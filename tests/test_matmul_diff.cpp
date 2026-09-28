// Differential tests: the optimized matmul against the reference matmul_naive.
//
// The reference is kept forever (reference.hpp). Every future optimization of matmul
// (loop order, compiler flags, tiling, SIMD, threads) must keep passing these tests.

#include <gtest/gtest.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
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
