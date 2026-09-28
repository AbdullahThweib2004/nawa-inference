// AVX2 int8 kernel. Compiled with -mavx2 -ffp-contract=off (and int8.cpp, the scalar
// reference, with -ffp-contract=off too): the shared float epilogue (finish_int8) must never be
// fused into an FMA in one file and not the other, even in a -march=native build (see the
// root CMakeLists.txt). Same ODR rules as gemm_avx2.cpp: nothing but intrinsics and the
// declarations in int8_kernels.hpp, internal linkage for helpers.
//
// Per 16 values of k:
//   vpmovsxbw  sign-extend 16 int8 WEIGHTS to int16 (activations are already int16)
//   vpmaddwd   multiply 16 int16 pairs and add adjacent products -> 8 int32
//   vpaddd     accumulate
// |product pair| <= 2 * 127 * 127 = 32258, so int32 lanes cannot overflow for any K that
// QuantizedMatrix accepts. Integer arithmetic is exact, so the sums equal the scalar
// reference's whatever the order.
//
// REGISTER BLOCKING, as in the float micro-kernel: the main loop computes 4 rows x 2 output
// channels (8 accumulators). Each widened weight vector is used for 4 rows, so widening costs
// 2 instructions per 8 multiply-adds instead of 1 per multiply-add. (Widening runs on a single
// execution port, and was the bottleneck of the first, unblocked version: 2.4x slower than
// float32 at batch 256.)

#include <immintrin.h>

#include "int8_kernels.hpp"

namespace inference::detail {

namespace {

std::int32_t hsum_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));  // swap 64-bit halves
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));  // swap 32-bit pairs
    return _mm_cvtsi128_si32(s);
}

// 16 int8 values -> 16 int16 (sign-extended).
inline __m256i widen16(const std::int8_t* p) {
    return _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// 16 int16 values.
inline __m256i load16(const std::int16_t* p) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

}  // namespace

void int8_rows_avx2(const std::int16_t* qa, const float* a_scales, std::size_t rows,
                    std::size_t K_pad, const std::int8_t* qw, const float* w_scales,
                    std::size_t n_begin, std::size_t n_end, float* c, std::size_t ldc,
                    const float* bias, bool relu) {
    std::size_t r = 0;
    // Blocks of 4 rows: each widened weight vector feeds 4 rows.
    for (; r + 4 <= rows; r += 4) {
        const std::int16_t* a0 = qa + (r + 0) * K_pad;
        const std::int16_t* a1 = qa + (r + 1) * K_pad;
        const std::int16_t* a2 = qa + (r + 2) * K_pad;
        const std::int16_t* a3 = qa + (r + 3) * K_pad;
        std::size_t j = n_begin;
        for (; j + 2 <= n_end; j += 2) {
            const std::int8_t* w0 = qw + j * K_pad;
            const std::int8_t* w1 = w0 + K_pad;
            __m256i acc[4][2];
            for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = _mm256_setzero_si256();
            for (std::size_t k = 0; k < K_pad; k += 16) {
                const __m256i v0 = widen16(w0 + k);
                const __m256i v1 = widen16(w1 + k);
                const __m256i x0 = load16(a0 + k);
                const __m256i x1 = load16(a1 + k);
                const __m256i x2 = load16(a2 + k);
                const __m256i x3 = load16(a3 + k);
                acc[0][0] = _mm256_add_epi32(acc[0][0], _mm256_madd_epi16(x0, v0));
                acc[0][1] = _mm256_add_epi32(acc[0][1], _mm256_madd_epi16(x0, v1));
                acc[1][0] = _mm256_add_epi32(acc[1][0], _mm256_madd_epi16(x1, v0));
                acc[1][1] = _mm256_add_epi32(acc[1][1], _mm256_madd_epi16(x1, v1));
                acc[2][0] = _mm256_add_epi32(acc[2][0], _mm256_madd_epi16(x2, v0));
                acc[2][1] = _mm256_add_epi32(acc[2][1], _mm256_madd_epi16(x2, v1));
                acc[3][0] = _mm256_add_epi32(acc[3][0], _mm256_madd_epi16(x3, v0));
                acc[3][1] = _mm256_add_epi32(acc[3][1], _mm256_madd_epi16(x3, v1));
            }
            for (std::size_t i = 0; i < 4; ++i) {
                float* out = c + (r + i) * ldc;
                out[j] =
                    finish_int8(hsum_epi32(acc[i][0]), a_scales[r + i], w_scales[j], bias, j, relu);
                out[j + 1] = finish_int8(hsum_epi32(acc[i][1]), a_scales[r + i], w_scales[j + 1],
                                         bias, j + 1, relu);
            }
        }
        for (; j < n_end; ++j) {  // odd channel left over
            const std::int8_t* w = qw + j * K_pad;
            __m256i acc[4];
            for (int i = 0; i < 4; ++i) acc[i] = _mm256_setzero_si256();
            for (std::size_t k = 0; k < K_pad; k += 16) {
                const __m256i v = widen16(w + k);
                acc[0] = _mm256_add_epi32(acc[0], _mm256_madd_epi16(load16(a0 + k), v));
                acc[1] = _mm256_add_epi32(acc[1], _mm256_madd_epi16(load16(a1 + k), v));
                acc[2] = _mm256_add_epi32(acc[2], _mm256_madd_epi16(load16(a2 + k), v));
                acc[3] = _mm256_add_epi32(acc[3], _mm256_madd_epi16(load16(a3 + k), v));
            }
            for (std::size_t i = 0; i < 4; ++i) {
                c[(r + i) * ldc + j] =
                    finish_int8(hsum_epi32(acc[i]), a_scales[r + i], w_scales[j], bias, j, relu);
            }
        }
    }
    // Remaining rows (fewer than 4, e.g. batch 1): one row at a time, 4 channels at once.
    for (; r < rows; ++r) {
        const std::int16_t* a = qa + r * K_pad;
        std::size_t j = n_begin;
        for (; j + 4 <= n_end; j += 4) {
            const std::int8_t* w0 = qw + j * K_pad;
            __m256i acc[4];
            for (int q = 0; q < 4; ++q) acc[q] = _mm256_setzero_si256();
            for (std::size_t k = 0; k < K_pad; k += 16) {
                const __m256i x = load16(a + k);
                for (int q = 0; q < 4; ++q) {
                    acc[q] = _mm256_add_epi32(
                        acc[q], _mm256_madd_epi16(
                                    x, widen16(w0 + static_cast<std::size_t>(q) * K_pad + k)));
                }
            }
            for (std::size_t q = 0; q < 4; ++q) {
                c[r * ldc + j + q] = finish_int8(hsum_epi32(acc[q]), a_scales[r], w_scales[j + q],
                                                 bias, j + q, relu);
            }
        }
        for (; j < n_end; ++j) {
            const std::int8_t* w = qw + j * K_pad;
            __m256i acc = _mm256_setzero_si256();
            for (std::size_t k = 0; k < K_pad; k += 16) {
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(load16(a + k), widen16(w + k)));
            }
            c[r * ldc + j] = finish_int8(hsum_epi32(acc), a_scales[r], w_scales[j], bias, j, relu);
        }
    }
}

}  // namespace inference::detail
