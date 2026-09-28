#pragma once

// Private interface between gemm.cpp (portable code) and gemm_avx2.cpp (compiled with
// -mavx2 -mfma). Only plain declarations with built-in types: nothing here may generate code,
// because anything inline that both files compiled would risk an AVX2 copy leaking into the
// portable build (see the note at the top of gemm_avx2.cpp).

#include <cstddef>

namespace inference::detail {

// Blocking parameters of the AVX2 GEMM (chosen in docs/performance.md, stage 9.3).
inline constexpr std::size_t kMR = 6;   // rows of C per micro-kernel call
inline constexpr std::size_t kNR = 16;  // columns of C per micro-kernel call (2 ymm)
// NAWA_GEMM_MC / NAWA_GEMM_KC exist only for tuning experiments (-D on the command line).
#ifndef NAWA_GEMM_MC
#define NAWA_GEMM_MC 96
#endif
#ifndef NAWA_GEMM_KC
#define NAWA_GEMM_KC 256
#endif
inline constexpr std::size_t kMC = NAWA_GEMM_MC;  // rows of A per cache block (A block in L2)
inline constexpr std::size_t kKC = NAWA_GEMM_KC;  // depth per cache block (B micro-panel in L1)
inline constexpr std::size_t kNC = 512;  // columns of B per cache block (B block lives in L2)

// Size in floats of the scratch buffer gemm_avx2 needs for packing A.
inline constexpr std::size_t kAvx2APackFloats = kMC * kKC;

// C {M,N} = A {M,K} · B {K,N}, all row-major: the portable i-k-j loop from step 9.1.
// Defined in gemm.cpp (portable code). Used as the fallback kernel and by matmul() for small
// M, where packing B would cost more than it saves.
void gemm_ikj(const float* a, std::size_t M, std::size_t K, std::size_t N, const float* b,
              float* c);

// C {M,N} = A {M,K} · B, with B in the 16-column panel layout (see gemm.hpp).
// `a_pack` must hold kAvx2APackFloats floats.
// `bias` (N values, or nullptr) and `relu` form the epilogue (see GemmEpilogue in gemm.hpp).
void gemm_avx2(const float* a, std::size_t M, std::size_t K, std::size_t N, const float* b_panels,
               float* c, float* a_pack, const float* bias, bool relu);

}  // namespace inference::detail
