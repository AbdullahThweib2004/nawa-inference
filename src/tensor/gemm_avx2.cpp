// Cache-blocked GEMM with an AVX2 + FMA register-blocked micro-kernel.
//
// THIS FILE IS COMPILED WITH -mavx2 -mfma, the rest of the library is not. It only runs after
// a runtime check (__builtin_cpu_supports in gemm.cpp) confirmed the CPU has AVX2 and FMA.
//
// ODR hazard, and why this file avoids the standard library: inline functions and templates
// (std::min, std::vector<float>::operator[], ...) are compiled into EVERY object file that
// uses them, and the linker keeps just one copy. If that copy came from this file, it could
// contain AVX2 instructions, and the "portable" default build would then crash with
// "illegal instruction" on a CPU without AVX2. So this file includes nothing but
// <immintrin.h> (intrinsics are always inlined) and gemm_kernels.hpp (declarations only),
// and every helper here has internal linkage (anonymous namespace).
//
// Structure (the BLIS/GotoBLAS scheme):
//   for each block of NC columns of B          (B block stays in L2)
//     for each block of KC rows of the depth   (B micro-panel, KC x 16, stays in L1)
//       for each block of MC rows of A         (packed A block stays in L2)
//         pack the MC x KC block of A into 6-row panels
//         for each 16-column panel of B
//           for each 6-row panel of A: micro-kernel -> 6 x 16 tile of C

#include <immintrin.h>

#include "gemm_kernels.hpp"

namespace inference::detail {

namespace {

std::size_t min_size(std::size_t a, std::size_t b) { return a < b ? a : b; }

// Packs rows [0, mc) x depth [0, kc) of A (row stride lda) into panels of kMR rows: panel r
// holds, for each k, the kMR values A[r*kMR + 0..kMR)[k] next to each other. Rows past mc are
// zero, so the micro-kernel can always compute full kMR-row tiles.
void pack_a(const float* a, std::size_t lda, std::size_t mc, std::size_t kc, float* out) {
    for (std::size_t r0 = 0; r0 < mc; r0 += kMR) {
        const std::size_t rows = min_size(kMR, mc - r0);
        for (std::size_t k = 0; k < kc; ++k) {
            for (std::size_t r = 0; r < rows; ++r) out[r] = a[(r0 + r) * lda + k];
            for (std::size_t r = rows; r < kMR; ++r) out[r] = 0.0f;
            out += kMR;
        }
    }
}

// Epilogue on one register of 8 outputs: + bias, then ReLU. max_ps(zero, x) (in THIS order)
// returns x when x is NaN or -0.0, exactly like the scalar `x < 0 ? 0 : x` of the ReLU layer.
inline __m256 epilogue8(__m256 x, const float* bias8, bool relu) {
    if (bias8) x = _mm256_add_ps(x, _mm256_loadu_ps(bias8));
    if (relu) x = _mm256_max_ps(_mm256_setzero_ps(), x);
    return x;
}

float epilogue1(float x, const float* bias, std::size_t j, bool relu) {
    if (bias) x = x + bias[j];
    if (relu && x < 0.0f) x = 0.0f;
    return x;
}

// The micro-kernel: a ROWS x 16 tile of C, held entirely in registers
// (ROWS x 2 ymm accumulators, 12 for ROWS = 6) for the whole depth loop.
//
// Per step of k: 2 loads of B (16 floats), ROWS broadcasts of A, 2*ROWS FMAs. Each loaded B
// vector feeds ROWS FMAs, and the 12 accumulators are independent, which hides the 4-cycle
// FMA latency: 2 FMA units x 4 cycles = 8 FMAs must be in flight, and there are 12.
//
// `accumulate`: add to the existing C (later depth blocks), or overwrite it (first block).
// `bias16` / `relu`: epilogue, passed only for the LAST depth block (bias16 = 16 values).
template <int ROWS>
void micro_kernel(std::size_t kc, const float* a_panel, const float* b_panel, float* c,
                  std::size_t ldc, bool accumulate, const float* bias16, bool relu) {
    __m256 acc[ROWS][2];
    for (int r = 0; r < ROWS; ++r) {
        if (accumulate) {
            acc[r][0] = _mm256_loadu_ps(c + r * ldc);
            acc[r][1] = _mm256_loadu_ps(c + r * ldc + 8);
        } else {
            acc[r][0] = _mm256_setzero_ps();
            acc[r][1] = _mm256_setzero_ps();
        }
    }
    for (std::size_t k = 0; k < kc; ++k) {
        const __m256 b0 = _mm256_loadu_ps(b_panel);
        const __m256 b1 = _mm256_loadu_ps(b_panel + 8);
        for (int r = 0; r < ROWS; ++r) {
            const __m256 a = _mm256_broadcast_ss(a_panel + r);
            acc[r][0] = _mm256_fmadd_ps(a, b0, acc[r][0]);
            acc[r][1] = _mm256_fmadd_ps(a, b1, acc[r][1]);
        }
        a_panel += kMR;  // packed A panels are always kMR wide
        b_panel += kNR;
    }
    const bool has_epilogue = bias16 != nullptr || relu;
    for (int r = 0; r < ROWS; ++r) {
        __m256 lo = acc[r][0], hi = acc[r][1];
        if (has_epilogue) {
            lo = epilogue8(lo, bias16, relu);
            hi = epilogue8(hi, bias16 ? bias16 + 8 : nullptr, relu);
        }
        _mm256_storeu_ps(c + r * ldc, lo);
        _mm256_storeu_ps(c + r * ldc + 8, hi);
    }
}

// Runs the micro-kernel for a tile of `rows` x `cols` (rows <= 6, cols <= 16). Full-width
// tiles write C directly. Edge tiles (the last columns when N isn't a multiple of 16) go
// through a 6 x 16 temporary so the kernel never touches memory outside C (or outside the
// bias); their epilogue is applied per element when copying back.
// `bias` points at this tile's first column (or is nullptr); `last` = last depth block.
void tile(std::size_t rows, std::size_t cols, std::size_t kc, const float* a_panel,
          const float* b_panel, float* c, std::size_t ldc, bool accumulate, const float* bias,
          bool relu, bool last) {
    float tmp[kMR * kNR];
    const bool full = cols == kNR;
    float* target = full ? c : tmp;
    const std::size_t ld = full ? ldc : kNR;
    if (!full && accumulate) {
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t j = 0; j < cols; ++j) tmp[r * kNR + j] = c[r * ldc + j];
    }
    const float* kernel_bias = full && last ? bias : nullptr;
    const bool kernel_relu = full && last && relu;
    switch (rows) {
        case 6:
            micro_kernel<6>(kc, a_panel, b_panel, target, ld, accumulate, kernel_bias, kernel_relu);
            break;
        case 5:
            micro_kernel<5>(kc, a_panel, b_panel, target, ld, accumulate, kernel_bias, kernel_relu);
            break;
        case 4:
            micro_kernel<4>(kc, a_panel, b_panel, target, ld, accumulate, kernel_bias, kernel_relu);
            break;
        case 3:
            micro_kernel<3>(kc, a_panel, b_panel, target, ld, accumulate, kernel_bias, kernel_relu);
            break;
        case 2:
            micro_kernel<2>(kc, a_panel, b_panel, target, ld, accumulate, kernel_bias, kernel_relu);
            break;
        default:
            micro_kernel<1>(kc, a_panel, b_panel, target, ld, accumulate, kernel_bias, kernel_relu);
            break;
    }
    if (!full) {
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t j = 0; j < cols; ++j) {
                const float v = tmp[r * kNR + j];
                c[r * ldc + j] = last ? epilogue1(v, bias, j, relu) : v;
            }
        }
    }
}

// Matrix-vector product (M = 1) straight from the packed B panels, without packing A.
// A single row has too little work for the 6-row kernel (it would compute 5 rows of zeros),
// and 2 accumulators per panel would leave the FMA units waiting on latency. So 4 panels are
// processed together: 8 independent accumulators, the same A value broadcast into all.
void gemv(const float* a, std::size_t K, std::size_t N, const float* b_panels, float* c,
          const float* bias, bool relu, std::size_t panel_begin, std::size_t panel_end) {
    const std::size_t panels = panel_end;
    const std::size_t panel_stride = K * kNR;  // floats between consecutive panels
    std::size_t p = panel_begin;
    for (; p + 4 <= panels; p += 4) {
        const float* b = b_panels + p * panel_stride;
        __m256 acc[4][2];
        for (int q = 0; q < 4; ++q) acc[q][0] = acc[q][1] = _mm256_setzero_ps();
        for (std::size_t k = 0; k < K; ++k) {
            const __m256 av = _mm256_broadcast_ss(a + k);
            for (int q = 0; q < 4; ++q) {
                const float* bq = b + q * panel_stride + k * kNR;
                acc[q][0] = _mm256_fmadd_ps(av, _mm256_loadu_ps(bq), acc[q][0]);
                acc[q][1] = _mm256_fmadd_ps(av, _mm256_loadu_ps(bq + 8), acc[q][1]);
            }
        }
        for (int q = 0; q < 4; ++q) {
            const std::size_t col = (p + static_cast<std::size_t>(q)) * kNR;
            float out[kNR];
            _mm256_storeu_ps(out, acc[q][0]);
            _mm256_storeu_ps(out + 8, acc[q][1]);
            const std::size_t cols = min_size(kNR, N - col);
            for (std::size_t j = 0; j < cols; ++j)
                c[col + j] = epilogue1(out[j], bias, col + j, relu);
        }
    }
    for (; p < panels; ++p) {  // leftover panels (fewer than 4)
        const float* b = b_panels + p * panel_stride;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        for (std::size_t k = 0; k < K; ++k) {
            const __m256 av = _mm256_broadcast_ss(a + k);
            acc0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b + k * kNR), acc0);
            acc1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b + k * kNR + 8), acc1);
        }
        float out[kNR];
        _mm256_storeu_ps(out, acc0);
        _mm256_storeu_ps(out + 8, acc1);
        const std::size_t col = p * kNR;
        const std::size_t cols = min_size(kNR, N - col);
        for (std::size_t j = 0; j < cols; ++j) c[col + j] = epilogue1(out[j], bias, col + j, relu);
    }
}

}  // namespace

void gemm_avx2(const float* a, std::size_t M, std::size_t K, std::size_t N, const float* b_panels,
               float* c, float* a_pack, const float* bias, bool relu, std::size_t col_begin,
               std::size_t col_end) {
    if (M == 0 || N == 0) return;
    if (K == 0) {  // empty sum: C = 0 (can't happen with Tensor, whose dims are >= 1)
        for (std::size_t i = 0; i < M; ++i)
            for (std::size_t j = col_begin; j < col_end; ++j)
                c[i * N + j] = epilogue1(0.0f, bias, j, relu);
        return;
    }
    if (M == 1) {
        gemv(a, K, N, b_panels, c, bias, relu, col_begin / kNR, (col_end + kNR - 1) / kNR);
        return;
    }
    const std::size_t panel_stride = K * kNR;
    for (std::size_t jc = col_begin; jc < col_end; jc += kNC) {
        const std::size_t nc = min_size(kNC, col_end - jc);
        for (std::size_t pc = 0; pc < K; pc += kKC) {
            const std::size_t kc = min_size(kKC, K - pc);
            const bool accumulate = pc > 0;  // first depth block overwrites C
            const bool last = pc + kc == K;  // the epilogue runs on the final values only
            for (std::size_t ic = 0; ic < M; ic += kMC) {
                const std::size_t mc = min_size(kMC, M - ic);
                pack_a(a + ic * K + pc, K, mc, kc, a_pack);
                for (std::size_t jr = 0; jr < nc; jr += kNR) {
                    const std::size_t col = jc + jr;
                    const std::size_t cols = min_size(kNR, col_end - col);
                    const float* b = b_panels + (col / kNR) * panel_stride + pc * kNR;
                    for (std::size_t ir = 0; ir < mc; ir += kMR) {
                        const std::size_t rows = min_size(kMR, mc - ir);
                        tile(rows, cols, kc, a_pack + ir * kc, b, c + (ic + ir) * N + col, N,
                             accumulate, bias ? bias + col : nullptr, relu, last);
                    }
                }
            }
        }
    }
}

}  // namespace inference::detail
