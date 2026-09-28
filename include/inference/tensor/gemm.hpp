#pragma once

#include <cstddef>
#include <vector>

// General matrix multiply (GEMM) with pre-packed right-hand matrices and runtime kernel
// selection. This is the engine behind matmul() and Linear (see docs/performance.md, 9.3).
namespace inference {

// Which implementation computes the product.
//  - Auto:     the fastest one this CPU supports (see resolve_kernel).
//  - Portable: the i-k-j loop from step 9.1. Plain C++, runs on any CPU.
//  - Avx2:     cache-blocked GEMM with a 6x16 AVX2 + FMA micro-kernel (x86-64 only).
enum class GemmKernel { Auto, Portable, Avx2 };

// True if `kernel` can run on this machine (Portable and Auto: always).
bool kernel_available(GemmKernel kernel);

// Turns Auto into a concrete kernel. The environment variable NAWA_KERNEL=portable|avx2
// overrides the automatic choice, which lets the whole test suite run on the fallback path.
// Throws std::runtime_error if a kernel is requested (explicitly or via NAWA_KERNEL) that
// this CPU can't run, or if NAWA_KERNEL has an unknown value.
GemmKernel resolve_kernel(GemmKernel requested = GemmKernel::Auto);

const char* kernel_name(GemmKernel kernel);

// A {K, N} right-hand matrix rearranged for one kernel. Packing costs a full pass over the
// matrix, so constant matrices (Linear weights) are packed once and reused for every call.
//
// Layouts:
//  - Portable: a plain row-major copy.
//  - Avx2: column panels of 16 floats. Panel p holds columns [16p, 16p+16) as K consecutive
//    rows of 16 floats, zero-padded past column N. The micro-kernel then reads B strictly
//    sequentially.
class PackedMatrix {
public:
    PackedMatrix() = default;

    // Packs the row-major {K, N} matrix `b`.
    static PackedMatrix pack(const float* b, std::size_t K, std::size_t N,
                             GemmKernel kernel = GemmKernel::Auto);

    std::size_t rows() const noexcept { return K_; }  // K
    std::size_t cols() const noexcept { return N_; }  // N
    GemmKernel kernel() const noexcept { return kernel_; }
    const float* data() const noexcept { return data_.data(); }
    std::size_t size_bytes() const noexcept { return data_.size() * sizeof(float); }

private:
    GemmKernel kernel_ = GemmKernel::Portable;
    std::size_t K_ = 0;
    std::size_t N_ = 0;
    std::vector<float> data_;
};

// Work done on each element of C as it is written, while it is still in a register
// ("fused" into the GEMM instead of separate passes over memory):
//     c = (A·B)[i][j] + bias[j]        if bias is set  (a Linear layer's bias)
//     c = max(c, 0)                    if relu is set  (a following ReLU layer)
// The results are bit-identical to doing the same steps separately afterwards.
struct GemmEpilogue {
    const float* bias = nullptr;  // N values, or nullptr for no bias
    bool relu = false;
};

// C = A · B (+ epilogue), where A is row-major {M, K}, B is packed {K, N}, and C is
// row-major {M, N} (overwritten). The kernel is the one B was packed for.
void gemm(const float* a, std::size_t M, const PackedMatrix& b, float* c,
          const GemmEpilogue& epilogue = {});

}  // namespace inference
