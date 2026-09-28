#pragma once

// Private interface between int8.cpp (portable) and int8_avx2.cpp (compiled with -mavx2).
// Declarations with built-in types only (see the ODR note in gemm_avx2.cpp).

#include <cstddef>
#include <cstdint>

namespace inference::detail {

// For rows [0, rows) of the quantized activations `qa` ({rows, K_pad} 8-bit values stored as
// int16, row scales a_scales)
// and output channels [n_begin, n_end) of the quantized weights `qw` ({N, K_pad}, scales
// w_scales): computes the exact int32 dot products and writes
//     c[r * ldc + j] = finish_int8(acc, a_scales[r], w_scales[j], bias, j, relu)
// Both kernels must produce the same int32 sums; the float conversion is shared.
void int8_rows_scalar(const std::int16_t* qa, const float* a_scales, std::size_t rows,
                      std::size_t K_pad, const std::int8_t* qw, const float* w_scales,
                      std::size_t n_begin, std::size_t n_end, float* c, std::size_t ldc,
                      const float* bias, bool relu);

void int8_rows_avx2(const std::int16_t* qa, const float* a_scales, std::size_t rows,
                    std::size_t K_pad, const std::int8_t* qw, const float* w_scales,
                    std::size_t n_begin, std::size_t n_end, float* c, std::size_t ldc,
                    const float* bias, bool relu);

// int32 sum -> float output, with the epilogue. Defined inline HERE, identically for both
// kernels, and both files are compiled with -ffp-contract=off, so the multiply and the add
// always round separately: the float results match bit for bit in every build. `static`
// gives it internal linkage (no ODR sharing between the AVX2 and portable files).
static inline float finish_int8(std::int32_t acc, float a_scale, float w_scale, const float* bias,
                                std::size_t j, bool relu) {
    float v = static_cast<float>(acc) * (a_scale * w_scale);
    if (bias) v = v + bias[j];
    if (relu && v < 0.0f) v = 0.0f;
    return v;
}

}  // namespace inference::detail
