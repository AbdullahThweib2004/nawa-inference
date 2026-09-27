#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "inference/tensor/ops.hpp"

using namespace inference;

namespace {

// Tensor of the given shape filled with 0, 1, 2, ... in row-major order.
Tensor arange(const Shape& shape) {
    std::vector<float> data(numel_of(shape));
    std::iota(data.begin(), data.end(), 0.0f);
    return Tensor(shape, std::move(data));
}

// Deterministic values that look random: no symmetry, mixed signs, not all integers.
// Useful when a simple pattern like 0, 1, 2, ... could hide an indexing bug.
Tensor pseudo_random(const Shape& shape, std::size_t seed) {
    std::vector<float> data(numel_of(shape));
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(static_cast<int>((i * 7 + seed * 13) % 11) - 5) * 0.5f;
    }
    return Tensor(shape, std::move(data));
}

Tensor identity(std::size_t n) {
    Tensor eye({n, n});
    for (std::size_t i = 0; i < n; ++i) eye.at({i, i}) = 1.0f;
    return eye;
}

}  // namespace

// ---------------------------------------------------------------------------
// matmul
// ---------------------------------------------------------------------------

TEST(Matmul, TwoByTwoHandExample) {
    const Tensor a({2, 2}, {1, 2, 3, 4});
    const Tensor b({2, 2}, {5, 6, 7, 8});
    // [1*5 + 2*7, 1*6 + 2*8]   [19, 22]
    // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
    EXPECT_TRUE(allclose(matmul(a, b), Tensor({2, 2}, {19, 22, 43, 50})));
}

TEST(Matmul, NonSquare) {
    const Tensor a = arange({2, 3});  // [[0,1,2],[3,4,5]]
    const Tensor b = arange({3, 4});  // [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    const Tensor c = matmul(a, b);
    EXPECT_EQ(c.shape(), (Shape{2, 4}));
    // Row 0: 0*[0,1,2,3] + 1*[4,5,6,7] + 2*[8,9,10,11] = [20, 23, 26, 29]
    // Row 1: 3*[0,1,2,3] + 4*[4,5,6,7] + 5*[8,9,10,11] = [56, 68, 80, 92]
    EXPECT_TRUE(allclose(c, Tensor({2, 4}, {20, 23, 26, 29, 56, 68, 80, 92})));
}

TEST(Matmul, MnistLayerShape) {
    // One flattened 28x28 image through a 784 -> 128 layer.
    const Tensor x = Tensor::ones({1, 784});
    const Tensor w = Tensor::full({784, 128}, 0.5f);
    const Tensor y = matmul(x, w);
    EXPECT_EQ(y.shape(), (Shape{1, 128}));
    EXPECT_TRUE(allclose(y, Tensor::full({1, 128}, 392.0f)));  // 784 * 0.5
}

TEST(Matmul, IdentityLeavesMatrixUnchanged) {
    const Tensor a = pseudo_random({3, 4}, 1);
    EXPECT_TRUE(allclose(matmul(identity(3), a), a));
    EXPECT_TRUE(allclose(matmul(a, identity(4)), a));
}

TEST(Matmul, TransposeOfProductIsProductOfTransposes) {
    // (A·B)ᵀ == Bᵀ·Aᵀ
    const Tensor a = pseudo_random({3, 5}, 1);
    const Tensor b = pseudo_random({5, 2}, 2);
    EXPECT_TRUE(allclose(transpose(matmul(a, b)), matmul(transpose(b), transpose(a))));
}

TEST(Matmul, InnerDimensionMismatchThrows) {
    EXPECT_THROW(matmul(Tensor({2, 3}), Tensor({4, 2})), std::invalid_argument);
    EXPECT_THROW(matmul(Tensor({2, 3}), Tensor({2, 3})), std::invalid_argument);
}

TEST(Matmul, NonTwoDimensionalThrows) {
    EXPECT_THROW(matmul(Tensor({3}), Tensor({3, 2})), std::invalid_argument);
    EXPECT_THROW(matmul(Tensor({2, 3}), Tensor({3})), std::invalid_argument);
    EXPECT_THROW(matmul(Tensor({1, 2, 3}), Tensor({3, 2})), std::invalid_argument);
    EXPECT_THROW(matmul(Tensor::scalar(1.0f), Tensor::scalar(1.0f)), std::invalid_argument);
}

TEST(Matmul, ErrorMessageContainsBothShapes) {
    try {
        matmul(Tensor({2, 3}), Tensor({4, 5}));
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("[2, 3]"), std::string::npos) << msg;
        EXPECT_NE(msg.find("[4, 5]"), std::string::npos) << msg;
    }
}

TEST(Matmul, MemberMatchesFreeFunction) {
    const Tensor x = pseudo_random({2, 4}, 3);
    const Tensor w = pseudo_random({4, 3}, 4);
    EXPECT_TRUE(allclose(x.matmul(w), matmul(x, w)));
}

// ---------------------------------------------------------------------------
// transpose
// ---------------------------------------------------------------------------

TEST(Transpose, TwoByThree) {
    const Tensor t = arange({2, 3});  // [[0,1,2],[3,4,5]]
    const Tensor tt = transpose(t);
    EXPECT_EQ(tt.shape(), (Shape{3, 2}));
    EXPECT_TRUE(allclose(tt, Tensor({3, 2}, {0, 3, 1, 4, 2, 5})));
}

TEST(Transpose, TwiceReturnsOriginal) {
    const Tensor t = pseudo_random({4, 7}, 5);
    EXPECT_TRUE(allclose(transpose(transpose(t)), t));
}

TEST(Transpose, NonTwoDimensionalThrows) {
    EXPECT_THROW(transpose(Tensor({3})), std::invalid_argument);
    EXPECT_THROW(transpose(Tensor({2, 3, 4})), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// broadcast_shape
// ---------------------------------------------------------------------------

TEST(BroadcastShape, ValidCases) {
    EXPECT_EQ(broadcast_shape({4, 128}, {128}), (Shape{4, 128}));
    EXPECT_EQ(broadcast_shape({128}, {4, 128}), (Shape{4, 128}));
    EXPECT_EQ(broadcast_shape({4, 1}, {1, 3}), (Shape{4, 3}));
    EXPECT_EQ(broadcast_shape({2, 3}, {2, 3}), (Shape{2, 3}));
    EXPECT_EQ(broadcast_shape({2, 1, 4}, {3, 1}), (Shape{2, 3, 4}));
}

TEST(BroadcastShape, ScalarWithAnything) {
    EXPECT_EQ(broadcast_shape({}, {}), Shape{});
    EXPECT_EQ(broadcast_shape({}, {5}), (Shape{5}));
    EXPECT_EQ(broadcast_shape({2, 3, 4}, {}), (Shape{2, 3, 4}));
}

TEST(BroadcastShape, IncompatibleThrows) {
    EXPECT_THROW(broadcast_shape({4, 128}, {64}), std::invalid_argument);
    EXPECT_THROW(broadcast_shape({2, 3}, {3, 2}), std::invalid_argument);
    EXPECT_THROW(broadcast_shape({2, 3}, {2}), std::invalid_argument);
}

TEST(BroadcastShape, ErrorMessageContainsBothShapes) {
    try {
        broadcast_shape({4, 128}, {64});
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("[4, 128]"), std::string::npos) << msg;
        EXPECT_NE(msg.find("[64]"), std::string::npos) << msg;
    }
}

// ---------------------------------------------------------------------------
// Element-wise ops
// ---------------------------------------------------------------------------

TEST(Elementwise, SameShape) {
    const Tensor a({2, 2}, {1, 2, 3, 4});
    const Tensor b({2, 2}, {10, 20, 30, 40});
    EXPECT_TRUE(allclose(add(a, b), Tensor({2, 2}, {11, 22, 33, 44})));
    EXPECT_TRUE(allclose(sub(a, b), Tensor({2, 2}, {-9, -18, -27, -36})));
    EXPECT_TRUE(allclose(mul(a, b), Tensor({2, 2}, {10, 40, 90, 160})));
    EXPECT_TRUE(allclose(div(b, a), Tensor({2, 2}, {10, 10, 10, 10})));
}

TEST(Elementwise, BiasStyleRowBroadcast) {
    // {2,3} + {3}: the bias row is added to every row, like in a Linear layer.
    const Tensor x = arange({2, 3});  // [[0,1,2],[3,4,5]]
    const Tensor bias({3}, {10, 20, 30});
    const Tensor y = add(x, bias);
    EXPECT_EQ(y.shape(), (Shape{2, 3}));
    EXPECT_TRUE(allclose(y, Tensor({2, 3}, {10, 21, 32, 13, 24, 35})));
    // Broadcasting works with the smaller tensor on either side.
    EXPECT_TRUE(allclose(add(bias, x), y));
    EXPECT_TRUE(allclose(sub(x, bias), Tensor({2, 3}, {-10, -19, -28, -7, -16, -25})));
}

TEST(Elementwise, ColumnBroadcast) {
    // {2,1} op {2,3}: each row of x is combined with that row's single value.
    const Tensor col({2, 1}, {10, 100});
    const Tensor x = arange({2, 3});
    EXPECT_TRUE(allclose(add(col, x), Tensor({2, 3}, {10, 11, 12, 103, 104, 105})));
    EXPECT_TRUE(allclose(mul(x, col), Tensor({2, 3}, {0, 10, 20, 300, 400, 500})));
}

TEST(Elementwise, OuterProductStyleBroadcast) {
    // {2,1} * {1,3} -> {2,3}: both inputs are stretched.
    const Tensor col({2, 1}, {1, 2});
    const Tensor row({1, 3}, {1, 10, 100});
    EXPECT_TRUE(allclose(mul(col, row), Tensor({2, 3}, {1, 10, 100, 2, 20, 200})));
}

TEST(Elementwise, ScalarTensorBroadcast) {
    const Tensor x = arange({2, 3});
    EXPECT_TRUE(allclose(add(x, Tensor::scalar(1.0f)), add(x, 1.0f)));
}

TEST(Elementwise, IncompatibleShapesThrow) {
    EXPECT_THROW(add(Tensor({4, 128}), Tensor({64})), std::invalid_argument);
}

TEST(Elementwise, ScalarOnEitherSide) {
    const Tensor x({3}, {1, 2, 4});
    EXPECT_TRUE(allclose(add(x, 1.0f), Tensor({3}, {2, 3, 5})));
    EXPECT_TRUE(allclose(sub(x, 1.0f), Tensor({3}, {0, 1, 3})));
    EXPECT_TRUE(allclose(mul(x, 2.0f), Tensor({3}, {2, 4, 8})));
    EXPECT_TRUE(allclose(div(x, 2.0f), Tensor({3}, {0.5f, 1, 2})));

    EXPECT_TRUE(allclose(add(1.0f, x), Tensor({3}, {2, 3, 5})));
    EXPECT_TRUE(allclose(sub(1.0f, x), Tensor({3}, {0, -1, -3})));  // order matters
    EXPECT_TRUE(allclose(mul(2.0f, x), Tensor({3}, {2, 4, 8})));
    EXPECT_TRUE(allclose(div(4.0f, x), Tensor({3}, {4, 2, 1})));  // order matters
}

TEST(Elementwise, Operators) {
    const Tensor a({2}, {6, 8});
    const Tensor b({2}, {2, 4});
    EXPECT_TRUE(allclose(a + b, add(a, b)));
    EXPECT_TRUE(allclose(a - b, sub(a, b)));
    EXPECT_TRUE(allclose(a * b, Tensor({2}, {12, 32})));  // element-wise, not matmul
    EXPECT_TRUE(allclose(a / b, Tensor({2}, {3, 2})));

    EXPECT_TRUE(allclose(a + 1.0f, Tensor({2}, {7, 9})));
    EXPECT_TRUE(allclose(a - 1.0f, Tensor({2}, {5, 7})));
    EXPECT_TRUE(allclose(a * 0.5f, Tensor({2}, {3, 4})));
    EXPECT_TRUE(allclose(a / 2.0f, Tensor({2}, {3, 4})));
    EXPECT_TRUE(allclose(1.0f + a, Tensor({2}, {7, 9})));
    EXPECT_TRUE(allclose(10.0f - a, Tensor({2}, {4, 2})));
    EXPECT_TRUE(allclose(0.5f * a, Tensor({2}, {3, 4})));
    EXPECT_TRUE(allclose(24.0f / a, Tensor({2}, {4, 3})));

    // Operators compose: y = 2x + b
    EXPECT_TRUE(allclose(2.0f * a + b, Tensor({2}, {14, 20})));
}

TEST(Elementwise, UnaryMinus) {
    const Tensor x({3}, {1, -2, 0});
    EXPECT_TRUE(allclose(-x, Tensor({3}, {-1, 2, 0})));
    EXPECT_TRUE(allclose(-(-x), x));
}

TEST(Elementwise, DivisionByZeroFollowsIeee) {
    const Tensor num({3}, {1, -1, 0});
    const Tensor q = num / 0.0f;
    EXPECT_TRUE(std::isinf(q.data()[0]) && q.data()[0] > 0);
    EXPECT_TRUE(std::isinf(q.data()[1]) && q.data()[1] < 0);
    EXPECT_TRUE(std::isnan(q.data()[2]));
}

TEST(Elementwise, InputsAreNotModified) {
    const Tensor a = arange({2, 3});
    const Tensor b({3}, {1, 1, 1});
    const Tensor a_before = a;
    (void)(a + b);
    EXPECT_TRUE(allclose(a, a_before));
}

// ---------------------------------------------------------------------------
// Reductions (all on t = [[0, 1, 2], [3, 4, 5]] unless noted)
// ---------------------------------------------------------------------------

TEST(Reduce, SumAlongAxes) {
    const Tensor t = arange({2, 3});
    EXPECT_TRUE(allclose(sum(t, 0), Tensor({3}, {3, 5, 7})));
    EXPECT_TRUE(allclose(sum(t, 1), Tensor({2}, {3, 12})));
    EXPECT_TRUE(allclose(sum(t, -1), sum(t, 1)));
    EXPECT_TRUE(allclose(sum(t, -2), sum(t, 0)));
}

TEST(Reduce, SumKeepdims) {
    const Tensor t = arange({2, 3});
    EXPECT_TRUE(allclose(sum(t, 0, true), Tensor({1, 3}, {3, 5, 7})));
    EXPECT_TRUE(allclose(sum(t, 1, true), Tensor({2, 1}, {3, 12})));
    EXPECT_TRUE(allclose(sum(t, -1, true), Tensor({2, 1}, {3, 12})));
}

TEST(Reduce, MaxAlongAxes) {
    const Tensor t({2, 3}, {1, 9, -2, 7, 0, 3});
    EXPECT_TRUE(allclose(max(t, 0), Tensor({3}, {7, 9, 3})));
    EXPECT_TRUE(allclose(max(t, 1), Tensor({2}, {9, 7})));
    EXPECT_TRUE(allclose(max(t, -1), Tensor({2}, {9, 7})));
    EXPECT_TRUE(allclose(max(t, 0, true), Tensor({1, 3}, {7, 9, 3})));
    EXPECT_TRUE(allclose(max(t, 1, true), Tensor({2, 1}, {9, 7})));
}

TEST(Reduce, MaxOfAllNegativeValues) {
    // Guards against starting the running maximum at 0 instead of the first element.
    const Tensor t({3}, {-5, -2, -9});
    EXPECT_TRUE(allclose(max(t, 0), Tensor::scalar(-2.0f)));
}

TEST(Reduce, MeanAlongAxes) {
    const Tensor t = arange({2, 3});
    EXPECT_TRUE(allclose(mean(t, 0), Tensor({3}, {1.5f, 2.5f, 3.5f})));
    EXPECT_TRUE(allclose(mean(t, 1), Tensor({2}, {1, 4})));
    EXPECT_TRUE(allclose(mean(t, -1), Tensor({2}, {1, 4})));
    EXPECT_TRUE(allclose(mean(t, 0, true), Tensor({1, 3}, {1.5f, 2.5f, 3.5f})));
    EXPECT_TRUE(allclose(mean(t, 1, true), Tensor({2, 1}, {1, 4})));
}

TEST(Reduce, ArgmaxAlongAxes) {
    const Tensor t({2, 3}, {1, 9, -2, 7, 0, 3});
    EXPECT_TRUE(allclose(argmax(t, 0), Tensor({3}, {1, 0, 1})));
    EXPECT_TRUE(allclose(argmax(t, 1), Tensor({2}, {1, 0})));
    EXPECT_TRUE(allclose(argmax(t, -1), Tensor({2}, {1, 0})));
    EXPECT_TRUE(allclose(argmax(t, 0, true), Tensor({1, 3}, {1, 0, 1})));
    EXPECT_TRUE(allclose(argmax(t, 1, true), Tensor({2, 1}, {1, 0})));
}

TEST(Reduce, ArgmaxTieKeepsFirstOccurrence) {
    const Tensor t({2, 4}, {3, 7, 7, 1,  //
                            5, 5, 5, 5});
    EXPECT_TRUE(allclose(argmax(t, 1), Tensor({2}, {1, 0})));
    EXPECT_TRUE(allclose(argmax(Tensor({2, 2}, {4, 4, 4, 4}), 0), Tensor({2}, {0, 0})));
}

TEST(Reduce, ThreeDimensionalMiddleAxis) {
    // {2,3,2} along axis 1 exercises both `outer` and `inner` in the reduction loop.
    const Tensor t = arange({2, 3, 2});
    // Block 0: [[0,1],[2,3],[4,5]] -> [6, 9];  block 1: [[6,7],[8,9],[10,11]] -> [24, 27]
    EXPECT_TRUE(allclose(sum(t, 1), Tensor({2, 2}, {6, 9, 24, 27})));
    EXPECT_TRUE(allclose(argmax(t, 1), Tensor({2, 2}, {2, 2, 2, 2})));
}

TEST(Reduce, OneDimensionalGivesScalar) {
    const Tensor t({4}, {1, 2, 3, 4});
    EXPECT_EQ(sum(t, 0).shape(), Shape{});
    EXPECT_EQ(sum(t, 0, true).shape(), (Shape{1}));
}

TEST(Reduce, InvalidAxisThrows) {
    const Tensor t = arange({2, 3});
    EXPECT_THROW(sum(t, 2), std::out_of_range);
    EXPECT_THROW(sum(t, -3), std::out_of_range);
    EXPECT_THROW(max(t, 5), std::out_of_range);
    EXPECT_THROW(mean(t, 2), std::out_of_range);
    EXPECT_THROW(argmax(t, -3), std::out_of_range);
    EXPECT_THROW(sum(Tensor::scalar(1.0f), 0), std::out_of_range);  // scalar has no axes
}

TEST(Reduce, TotalReductions) {
    const Tensor t({2, 3}, {1, 9, -2, 7, 0, 3});
    const Tensor s = sum(t);
    EXPECT_EQ(s.shape(), Shape{});
    EXPECT_TRUE(allclose(s, Tensor::scalar(18.0f)));

    const Tensor m = max(t);
    EXPECT_EQ(m.shape(), Shape{});
    EXPECT_TRUE(allclose(m, Tensor::scalar(9.0f)));

    EXPECT_TRUE(allclose(max(Tensor({3}, {-5, -2, -9})), Tensor::scalar(-2.0f)));
    EXPECT_TRUE(allclose(sum(Tensor::scalar(4.0f)), Tensor::scalar(4.0f)));
}
