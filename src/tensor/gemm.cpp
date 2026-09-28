#include "inference/tensor/gemm.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm_kernels.hpp"

namespace inference {

namespace {

bool cpu_has_avx2_fma() {
#if NAWA_HAVE_AVX2_KERNEL && (defined(__GNUC__) || defined(__clang__))
    // Checks both the CPU and that the OS saves the 256-bit registers (XSAVE/OSXSAVE).
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;  // non-x86 build, or a compiler without __builtin_cpu_supports
#endif
}

}  // namespace

// The step 9.1 kernel: i-k-j over a row-major B. Runs on any CPU.
void detail::gemm_ikj(const float* a, std::size_t M, std::size_t K, std::size_t N, const float* b,
                      float* c) {
    for (std::size_t i = 0; i < M * N; ++i) c[i] = 0.0f;
    for (std::size_t i = 0; i < M; ++i) {
        float* c_row = c + i * N;
        const float* a_row = a + i * K;
        for (std::size_t k = 0; k < K; ++k) {
            const float a_ik = a_row[k];
            const float* b_row = b + k * N;
            for (std::size_t j = 0; j < N; ++j) c_row[j] += a_ik * b_row[j];
        }
    }
}

bool kernel_available(GemmKernel kernel) {
    return kernel != GemmKernel::Avx2 || cpu_has_avx2_fma();
}

const char* kernel_name(GemmKernel kernel) {
    switch (kernel) {
        case GemmKernel::Auto:
            return "auto";
        case GemmKernel::Portable:
            return "portable";
        case GemmKernel::Avx2:
            return "avx2";
    }
    return "unknown";
}

GemmKernel resolve_kernel(GemmKernel requested) {
    if (requested == GemmKernel::Auto) {
        // Read once: the choice must not change while packed matrices exist.
        static const GemmKernel automatic = [] {
            if (const char* env = std::getenv("NAWA_KERNEL")) {
                const std::string value = env;
                if (value == "portable") return GemmKernel::Portable;
                if (value == "avx2") return GemmKernel::Avx2;
                if (value != "auto" && !value.empty()) {
                    throw std::runtime_error("NAWA_KERNEL must be auto, portable or avx2, got '" +
                                             value + "'");
                }
            }
            return cpu_has_avx2_fma() ? GemmKernel::Avx2 : GemmKernel::Portable;
        }();
        requested = automatic;
    }
    if (!kernel_available(requested)) {
        throw std::runtime_error(std::string("GEMM kernel '") + kernel_name(requested) +
                                 "' is not supported by this CPU or build");
    }
    return requested;
}

PackedMatrix PackedMatrix::pack(const float* b, std::size_t K, std::size_t N, GemmKernel kernel) {
    PackedMatrix packed;
    packed.kernel_ = resolve_kernel(kernel);
    packed.K_ = K;
    packed.N_ = N;
    if (packed.kernel_ == GemmKernel::Portable) {
        packed.data_.assign(b, b + K * N);
        return packed;
    }
    // Avx2: panels of kNR columns, each stored as K rows of kNR floats, zero-padded.
    const std::size_t nr = detail::kNR;
    const std::size_t panels = (N + nr - 1) / nr;
    packed.data_.assign(panels * K * nr, 0.0f);
    float* out = packed.data_.data();
    for (std::size_t p = 0; p < panels; ++p) {
        const std::size_t col = p * nr;
        const std::size_t cols = N - col < nr ? N - col : nr;
        for (std::size_t k = 0; k < K; ++k) {
            const float* src = b + k * N + col;
            for (std::size_t j = 0; j < cols; ++j) out[j] = src[j];
            out += nr;
        }
    }
    return packed;
}

void gemm(const float* a, std::size_t M, const PackedMatrix& b, float* c,
          const GemmEpilogue& epilogue) {
    const std::size_t K = b.rows();
    const std::size_t N = b.cols();
    if (b.kernel() == GemmKernel::Portable) {
        detail::gemm_ikj(a, M, K, N, b.data(), c);
        // Epilogue as a pass over each row right after it is computed (still in cache).
        if (epilogue.bias || epilogue.relu) {
            for (std::size_t i = 0; i < M; ++i) {
                float* row = c + i * N;
                if (epilogue.bias)
                    for (std::size_t j = 0; j < N; ++j) row[j] = row[j] + epilogue.bias[j];
                if (epilogue.relu)
                    for (std::size_t j = 0; j < N; ++j)
                        if (row[j] < 0.0f) row[j] = 0.0f;
            }
        }
        return;
    }
#if NAWA_HAVE_AVX2_KERNEL
    // Scratch space for packing blocks of A. thread_local: each thread gets its own, it is
    // allocated on the first call only, and it is reused by every later call.
    thread_local std::vector<float> a_pack(detail::kAvx2APackFloats);
    detail::gemm_avx2(a, M, K, N, b.data(), c, a_pack.data(), epilogue.bias, epilogue.relu);
#else
    (void)a;
    (void)c;
    (void)epilogue;
    throw std::logic_error("gemm: AVX2 kernel not compiled into this build");
#endif
}

}  // namespace inference
