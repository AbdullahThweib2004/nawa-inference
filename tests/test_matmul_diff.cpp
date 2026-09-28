// Differential tests: the optimized matmul against the reference matmul_naive.
//
// The reference is kept forever (reference.hpp). Every future optimization of matmul
// (loop order, compiler flags, tiling, SIMD, threads) must keep passing these tests.

#include <gtest/gtest.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "inference/tensor/ops.hpp"
#include "inference/tensor/reference.hpp"

using namespace inference;

namespace {

// Deterministic pseudo-random values in [-1, 1).
Tensor random_matrix(std::size_t rows, std::size_t cols, std::uint32_t seed) {
    std::vector<float> data(rows * cols);
    std::uint32_t state = seed * 2654435761u + 12345u;
    for (float& v : data) {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>(state >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
    }
    return Tensor({rows, cols}, std::move(data));
}

Tensor abs_of(const Tensor& t) {
    Tensor out = t;
    for (std::size_t i = 0; i < out.numel(); ++i) out.data()[i] = std::fabs(out.data()[i]);
    return out;
}

struct Shape3 {
    std::size_t M, K, N;
};

}  // namespace

// WHY NOT EXACT EQUALITY: float addition is not associative. (a + b) + c and a + (b + c)
// can differ in the last bits, because each addition rounds to 24 significant bits. An
// optimized matmul is allowed to add the K products of a dot product in a different order
// (tiling, several accumulators, threads) or to fuse a*b+c into one FMA instruction, which
// rounds once instead of twice. Both change the last bits of the result without either
// version being "wrong".
//
// So the tolerance is the standard worst-case error bound for a float dot product of
// length K:
//     |computed - exact| <= K * eps * sum_k |a_ik * b_kj|        (eps = FLT_EPSILON)
// Both versions stay within that bound of the exact value, so they differ from each other
// by at most twice it. sum_k |a_ik * b_kj| is itself a matrix product, of |A| and |B|,
// computed here with the reference. The bound scales with K and with the magnitudes
// involved, unlike a fixed tolerance, which would be too loose for small results and too
// strict for large K.
TEST(MatmulDifferential, MatchesReferenceOnManyShapes) {
    const std::vector<Shape3> shapes = {
        {1, 1, 1},    {1, 784, 128}, {3, 5, 7},   {17, 31, 13}, {127, 129, 65}, {256, 784, 128},
        {64, 64, 64}, {1, 1, 100},   {100, 1, 1}, {33, 1, 17},  {2, 1000, 3},   {256, 128, 10}};
    std::uint32_t seed = 1;
    for (const auto& [M, K, N] : shapes) {
        SCOPED_TRACE(::testing::Message() << "shape " << M << "x" << K << "x" << N);
        const Tensor a = random_matrix(M, K, seed++);
        const Tensor b = random_matrix(K, N, seed++);

        const Tensor fast = matmul(a, b);
        const Tensor ref = matmul_naive(a, b);
        const Tensor magnitude = matmul_naive(abs_of(a), abs_of(b));
        ASSERT_EQ(fast.shape(), ref.shape());

        const float factor = 2.0f * static_cast<float>(K) * FLT_EPSILON;
        std::size_t violations = 0;
        float worst_ratio = 0.0f;  // largest error / bound seen
        for (std::size_t i = 0; i < fast.numel(); ++i) {
            const float err = std::fabs(fast.data()[i] - ref.data()[i]);
            const float bound = factor * magnitude.data()[i];
            if (err > bound) ++violations;
            if (bound > 0.0f) worst_ratio = std::max(worst_ratio, err / bound);
        }
        EXPECT_EQ(violations, 0u) << "worst error / bound = " << worst_ratio;
    }
}

TEST(MatmulDifferential, ReferenceStillCorrect) {
    // The reference itself must stay right: the 2x2 example from test_ops.cpp.
    const Tensor a({2, 2}, {1, 2, 3, 4});
    const Tensor b({2, 2}, {5, 6, 7, 8});
    EXPECT_TRUE(allclose(matmul_naive(a, b), Tensor({2, 2}, {19, 22, 43, 50})));
    EXPECT_THROW(matmul_naive(Tensor({2, 3}), Tensor({4, 2})), std::invalid_argument);
    EXPECT_THROW(matmul_naive(Tensor({3}), Tensor({3, 2})), std::invalid_argument);
}

TEST(MatmulDifferential, SpecialValuesPropagateLikeReference) {
    // inf and NaN must propagate the same way (e.g. no "optimization" that skips zeros).
    Tensor a = random_matrix(3, 4, 7);
    Tensor b = random_matrix(4, 5, 8);
    a.at({1, 2}) = INFINITY;
    b.at({0, 3}) = NAN;
    const Tensor fast = matmul(a, b);
    const Tensor ref = matmul_naive(a, b);
    for (std::size_t i = 0; i < fast.numel(); ++i) {
        const float f = fast.data()[i], r = ref.data()[i];
        EXPECT_EQ(std::isnan(f), std::isnan(r)) << "element " << i;
        EXPECT_EQ(std::isinf(f), std::isinf(r)) << "element " << i;
        if (std::isfinite(r)) {
            EXPECT_NEAR(f, r, 1e-5f) << "element " << i;
        }
    }
}

// ---------------------------------------------------------------------------
// The GEMM kernels directly: every kernel this CPU supports, packed B, awkward shapes.
// ---------------------------------------------------------------------------

namespace {

std::vector<GemmKernel> available_kernels() {
    std::vector<GemmKernel> kernels;
    for (GemmKernel k : {GemmKernel::Portable, GemmKernel::Avx2}) {
        if (kernel_available(k)) kernels.push_back(k);
    }
    return kernels;
}

}  // namespace

TEST(GemmKernels, EveryKernelMatchesReference) {
    // Chosen to hit every edge of the AVX2 blocking (MR=6, NR=16, KC=256, NC=512):
    // M not a multiple of 6, N not a multiple of 16, K > KC, N > NC, the M=1 gemv path with
    // leftover panels, and exact multiples.
    const std::vector<Shape3> shapes = {
        {1, 1, 1},   {1, 3, 5},    {1, 784, 128},  {1, 300, 70},  {2, 17, 33},     {5, 7, 16},
        {6, 16, 16}, {7, 300, 17}, {13, 520, 530}, {97, 257, 49}, {256, 784, 128}, {256, 128, 10}};
    for (GemmKernel kernel : available_kernels()) {
        std::uint32_t seed = 100;
        for (const auto& [M, K, N] : shapes) {
            SCOPED_TRACE(::testing::Message()
                         << kernel_name(kernel) << " " << M << "x" << K << "x" << N);
            const Tensor a = random_matrix(M, K, seed++);
            const Tensor b = random_matrix(K, N, seed++);
            const PackedMatrix packed = PackedMatrix::pack(b.data(), K, N, kernel);
            ASSERT_EQ(packed.kernel(), kernel);
            Tensor c({M, N});
            c.fill(12345.0f);  // must be fully overwritten
            gemm(a.data(), M, packed, c.data());

            const Tensor ref = matmul_naive(a, b);
            const Tensor magnitude = matmul_naive(abs_of(a), abs_of(b));
            const float factor = 2.0f * static_cast<float>(K) * FLT_EPSILON;
            const auto violations = [&](const Tensor& result) {
                std::size_t count = 0;
                for (std::size_t i = 0; i < result.numel(); ++i) {
                    if (std::fabs(result.data()[i] - ref.data()[i]) >
                        factor * magnitude.data()[i]) {
                        ++count;
                    }
                }
                return count;
            };
            EXPECT_EQ(violations(c), 0u);

            // The explicit-kernel matmul overload: same code path (and so the same bits) as
            // gemm when it packs B; for M < kMatmulPackMinRows it uses the unpacked i-k-j loop
            // instead, whose separate multiply and add can differ from FMA in the last bits.
            const Tensor via_matmul = matmul(a, b, kernel);
            EXPECT_EQ(violations(via_matmul), 0u) << "matmul overload";
            if (kernel == GemmKernel::Portable || M >= kMatmulPackMinRows) {
                EXPECT_TRUE(allclose(via_matmul, c, 0.0f, 0.0f)) << "matmul overload, same path";
            }
        }
    }
}

TEST(GemmKernels, OneRowIsBitIdenticalToARowOfABatch) {
    // Batch 1 (the gemv path) and a row inside a batch (the blocked path) must give the same
    // bits: both apply the multiply-adds for each output in the same k order.
    for (GemmKernel kernel : available_kernels()) {
        const Tensor a = random_matrix(9, 784, 3);
        const Tensor b = random_matrix(784, 130, 4);
        const PackedMatrix packed = PackedMatrix::pack(b.data(), 784, 130, kernel);
        Tensor batch({9, 130});
        gemm(a.data(), 9, packed, batch.data());
        for (std::size_t row = 0; row < 9; ++row) {
            Tensor single({1, 130});
            gemm(a.data() + row * 784, 1, packed, single.data());
            EXPECT_EQ(std::memcmp(single.data(), batch.data() + row * 130, 130 * sizeof(float)), 0)
                << kernel_name(kernel) << " row " << row;
        }
    }
}

TEST(GemmKernels, SelectionAndAvailability) {
    EXPECT_TRUE(kernel_available(GemmKernel::Portable));
    EXPECT_TRUE(kernel_available(GemmKernel::Auto));
    EXPECT_NE(resolve_kernel(), GemmKernel::Auto);  // Auto always resolves to a real kernel
    EXPECT_EQ(resolve_kernel(GemmKernel::Portable), GemmKernel::Portable);
    if (!kernel_available(GemmKernel::Avx2)) {
        EXPECT_THROW(resolve_kernel(GemmKernel::Avx2), std::runtime_error);
    }
    std::cout << "[          ] automatic kernel: " << kernel_name(resolve_kernel()) << "\n";
}
