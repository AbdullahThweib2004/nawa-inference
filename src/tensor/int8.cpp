#include "inference/tensor/int8.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "inference/runtime/thread_pool.hpp"
#include "int8_kernels.hpp"

namespace inference {

namespace {

constexpr float kQMax = 127.0f;  // symmetric: -127..127 (-128 is never used)

std::size_t pad_to(std::size_t n, std::size_t multiple) {
    return (n + multiple - 1) / multiple * multiple;
}

// Rounds v (|v| <= 127) to the nearest integer, ties to even: the same result as
// std::nearbyint in the default rounding mode. Adding 1.5 * 2^23 pushes the fraction bits out
// of the float's 24-bit mantissa, and the hardware rounds them away (to nearest even);
// subtracting leaves the rounded value. Two additions instead of a libm call per value: the
// portable build targets baseline x86-64, which has no SSE4.1 rounding instruction, so
// std::nearbyint compiled to a function call and dominated activation quantization.
inline float round_nearest_even(float v) {
    constexpr float kMagic = 12582912.0f;  // 1.5 * 2^23
    const float shifted = v + kMagic;
    return shifted - kMagic;
}

std::int8_t quantize_value(float x, float inv_scale) {
    // Clamp first (x * (1/scale) can land a hair outside +-127 because the reciprocal is
    // rounded), then round. For |v| <= 127.x this equals rounding and then clamping.
    const float v = std::clamp(x * inv_scale, -kQMax, kQMax);
    return static_cast<std::int8_t>(round_nearest_even(v));
}

}  // namespace

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

QuantizedMatrix QuantizedMatrix::quantize(const float* w, std::size_t K, std::size_t N) {
    // The largest possible |sum| is K * 127 * 127; it must fit in int32.
    if (K > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) / (127 * 127)) {
        throw std::invalid_argument("QuantizedMatrix: in_features " + std::to_string(K) +
                                    " is too large for int32 accumulation");
    }
    QuantizedMatrix m;
    m.K_ = K;
    m.N_ = N;
    m.K_pad_ = pad_to(K, kPad);
    m.q_.assign(N * m.K_pad_, 0);
    m.scales_.assign(N, 1.0f);
    for (std::size_t j = 0; j < N; ++j) {
        float max_abs = 0.0f;
        for (std::size_t k = 0; k < K; ++k) max_abs = std::max(max_abs, std::fabs(w[k * N + j]));
        // An all-zero column keeps scale 1 (its weights quantize to 0 either way).
        const float scale = max_abs > 0.0f ? max_abs / kQMax : 1.0f;
        m.scales_[j] = scale;
        const float inv = 1.0f / scale;
        for (std::size_t k = 0; k < K; ++k)
            m.q_[j * m.K_pad_ + k] = quantize_value(w[k * N + j], inv);
    }
    return m;
}

QuantizedMatrix QuantizedMatrix::from_quantized(const std::int8_t* q, const float* scales,
                                                std::size_t K, std::size_t N) {
    QuantizedMatrix m;
    m.K_ = K;
    m.N_ = N;
    m.K_pad_ = pad_to(K, kPad);
    m.q_.assign(N * m.K_pad_, 0);
    m.scales_.assign(scales, scales + N);
    for (std::size_t j = 0; j < N; ++j)
        std::copy(q + j * K, q + (j + 1) * K,
                  m.q_.begin() + static_cast<std::ptrdiff_t>(j * m.K_pad_));
    return m;
}

float quantize_row(const float* x, std::size_t K, std::size_t K_pad, std::int16_t* out) {
    float max_abs = 0.0f;
    for (std::size_t k = 0; k < K; ++k) max_abs = std::max(max_abs, std::fabs(x[k]));
    const float scale = max_abs > 0.0f ? max_abs / kQMax : 1.0f;
    const float inv = 1.0f / scale;
    for (std::size_t k = 0; k < K; ++k) out[k] = quantize_value(x[k], inv);
    for (std::size_t k = K; k < K_pad; ++k) out[k] = 0;
    return scale;
}

// ---------------------------------------------------------------------------
// Scalar reference kernel
// ---------------------------------------------------------------------------

void detail::int8_rows_scalar(const std::int16_t* qa, const float* a_scales, std::size_t rows,
                              std::size_t K_pad, const std::int8_t* qw, const float* w_scales,
                              std::size_t n_begin, std::size_t n_end, float* c, std::size_t ldc,
                              const float* bias, bool relu) {
    for (std::size_t r = 0; r < rows; ++r) {
        const std::int16_t* a = qa + r * K_pad;
        for (std::size_t j = n_begin; j < n_end; ++j) {
            const std::int8_t* w = qw + j * K_pad;
            std::int32_t acc = 0;
            for (std::size_t k = 0; k < K_pad; ++k) {
                acc += static_cast<std::int32_t>(a[k]) * static_cast<std::int32_t>(w[k]);
            }
            c[r * ldc + j] = finish_int8(acc, a_scales[r], w_scales[j], bias, j, relu);
        }
    }
}

// ---------------------------------------------------------------------------
// Driver: quantize activations per thread, run the kernel, split across threads
// ---------------------------------------------------------------------------

void gemm_int8(const float* a, std::size_t M, const QuantizedMatrix& w, float* c,
               const GemmEpilogue& epilogue, GemmKernel kernel) {
    const GemmKernel chosen = resolve_kernel(kernel);
    const std::size_t K = w.in_features();
    const std::size_t N = w.out_features();
    const std::size_t K_pad = w.padded_in();
#if NAWA_HAVE_AVX2_KERNEL
    const auto rows_kernel =
        chosen == GemmKernel::Avx2 ? detail::int8_rows_avx2 : detail::int8_rows_scalar;
#else
    (void)chosen;
    const auto rows_kernel = detail::int8_rows_scalar;
#endif

    // Quantizes rows [row_begin, row_end) into this thread's scratch buffers (grown on first
    // use, then reused: no allocations in steady state), then computes columns
    // [col_begin, col_end) of those rows.
    const auto run = [&](std::size_t row_begin, std::size_t row_end, std::size_t col_begin,
                         std::size_t col_end) {
        thread_local std::vector<std::int16_t> qa;
        thread_local std::vector<float> scales;
        const std::size_t rows = row_end - row_begin;
        if (qa.size() < rows * K_pad) qa.resize(rows * K_pad);
        if (scales.size() < rows) scales.resize(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            scales[r] = quantize_row(a + (row_begin + r) * K, K, K_pad, qa.data() + r * K_pad);
        }
        rows_kernel(qa.data(), scales.data(), rows, K_pad, w.data(), w.scales(), col_begin, col_end,
                    c + row_begin * N, N, epilogue.bias, epilogue.relu);
    };

    ThreadPool& pool = default_thread_pool();
    const std::size_t threads = pool.num_threads();
    if (threads == 1 || M * K * N < parallel_threshold_macs()) {
        run(0, M, 0, N);
    } else if (M >= threads) {
        const std::size_t rows_per_chunk = (M + threads - 1) / threads;
        pool.parallel_for(M, rows_per_chunk,
                          [&](std::size_t begin, std::size_t end) { run(begin, end, 0, N); });
    } else {
        const std::size_t cols_per_chunk = (N + threads - 1) / threads;
        pool.parallel_for(N, cols_per_chunk,
                          [&](std::size_t begin, std::size_t end) { run(0, M, begin, end); });
    }
}

}  // namespace inference
