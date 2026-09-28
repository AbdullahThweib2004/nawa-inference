#include "inference/tensor/gemm.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm_kernels.hpp"
#include "inference/runtime/thread_pool.hpp"

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

std::size_t parallel_threshold_macs() {
    static const std::size_t threshold = [] {
        if (const char* env = std::getenv("NAWA_MIN_PARALLEL_MACS")) {
            try {
                return static_cast<std::size_t>(std::stoull(env));
            } catch (const std::exception&) {
                // invalid value: keep the default
            }
        }
        return kMinParallelMacs;
    }();
    return threshold;
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

namespace {

// Epilogue for the portable kernel: a pass over each row right after it is computed.
void portable_rows(const float* a, std::size_t rows, std::size_t K, std::size_t N, const float* b,
                   float* c, const GemmEpilogue& epilogue) {
    detail::gemm_ikj(a, rows, K, N, b, c);
    if (!epilogue.bias && !epilogue.relu) return;
    for (std::size_t i = 0; i < rows; ++i) {
        float* row = c + i * N;
        if (epilogue.bias)
            for (std::size_t j = 0; j < N; ++j) row[j] = row[j] + epilogue.bias[j];
        if (epilogue.relu)
            for (std::size_t j = 0; j < N; ++j)
                if (row[j] < 0.0f) row[j] = 0.0f;
    }
}

}  // namespace

void gemm(const float* a, std::size_t M, const PackedMatrix& b, float* c,
          const GemmEpilogue& epilogue) {
    const std::size_t K = b.rows();
    const std::size_t N = b.cols();
    ThreadPool& pool = default_thread_pool();
    const std::size_t threads = pool.num_threads();
    // Small products stay on the calling thread: handing work to other threads costs a few
    // microseconds of wake-up and synchronization (threshold measured, docs/performance.md).
    const bool parallel = threads > 1 && M * K * N >= parallel_threshold_macs();

    if (b.kernel() == GemmKernel::Portable) {
        if (!parallel || M < 2) {
            portable_rows(a, M, K, N, b.data(), c, epilogue);
            return;
        }
        const std::size_t rows_per_chunk = (M + threads - 1) / threads;
        pool.parallel_for(M, rows_per_chunk, [&](std::size_t begin, std::size_t end) {
            portable_rows(a + begin * K, end - begin, K, N, b.data(), c + begin * N, epilogue);
        });
        return;
    }

#if NAWA_HAVE_AVX2_KERNEL
    // Each thread packs blocks of A into its own scratch buffer. thread_local: allocated on a
    // thread's first call only and reused afterwards.
    const auto run = [&](std::size_t row_begin, std::size_t row_end, std::size_t col_begin,
                         std::size_t col_end) {
        thread_local std::vector<float> a_pack(detail::kAvx2APackFloats);
        detail::gemm_avx2(a + row_begin * K, row_end - row_begin, K, N, b.data(), c + row_begin * N,
                          a_pack.data(), epilogue.bias, epilogue.relu, col_begin, col_end);
    };
    if (!parallel) {
        run(0, M, 0, N);
        return;
    }
    // Split so each thread gets one contiguous share. Each element of C is computed by exactly
    // one thread with the same sequence of operations as in a single-threaded run, so the
    // result is bit-identical for any number of threads.
    if (M >= detail::kMR * threads) {
        // Enough rows: split M into row blocks (multiples of the 6-row micro-tile).
        const std::size_t tiles = (M + detail::kMR - 1) / detail::kMR;
        const std::size_t rows_per_chunk = (tiles + threads - 1) / threads * detail::kMR;
        pool.parallel_for(M, rows_per_chunk,
                          [&](std::size_t begin, std::size_t end) { run(begin, end, 0, N); });
    } else {
        // Few rows (e.g. batch 1): split the 16-column panels of B instead. Each thread then
        // streams only its part of the weights, from its own core's L2 cache.
        const std::size_t panels = (N + detail::kNR - 1) / detail::kNR;
        const std::size_t panels_per_chunk = (panels + threads - 1) / threads;
        pool.parallel_for(panels, panels_per_chunk, [&](std::size_t begin, std::size_t end) {
            run(0, M, begin * detail::kNR, std::min(N, end * detail::kNR));
        });
    }
#else
    (void)a;
    (void)c;
    (void)epilogue;
    (void)parallel;
    throw std::logic_error("gemm: AVX2 kernel not compiled into this build");
#endif
}

}  // namespace inference
