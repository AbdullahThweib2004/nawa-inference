#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "inference/tensor/tensor.hpp"

using inference::allclose;
using inference::compute_strides;
using inference::numel_of;
using inference::Shape;
using inference::Tensor;

namespace {

// Tensor of the given shape filled with 0, 1, 2, ... in row-major order.
Tensor arange(const Shape& shape) {
    std::vector<float> data(numel_of(shape));
    std::iota(data.begin(), data.end(), 0.0f);
    return Tensor(shape, std::move(data));
}

}  // namespace

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

TEST(ShapeHelpers, ComputeStrides) {
    EXPECT_EQ(compute_strides({}), Shape{});
    EXPECT_EQ(compute_strides({5}), (Shape{1}));
    EXPECT_EQ(compute_strides({2, 3}), (Shape{3, 1}));
    EXPECT_EQ(compute_strides({2, 3, 4}), (Shape{12, 4, 1}));
    EXPECT_EQ(compute_strides({1, 784}), (Shape{784, 1}));
}

TEST(ShapeHelpers, NumelOf) {
    EXPECT_EQ(numel_of({}), 1u);
    EXPECT_EQ(numel_of({5}), 5u);
    EXPECT_EQ(numel_of({2, 3, 4}), 24u);
    EXPECT_EQ(numel_of({1, 784}), 784u);
}

TEST(ShapeHelpers, NumelOfOverflowThrows) {
    const std::size_t big = std::numeric_limits<std::size_t>::max() / 2 + 1;
    EXPECT_THROW(numel_of({big, 2}), std::invalid_argument);
}

TEST(ShapeHelpers, ShapeToString) {
    EXPECT_EQ(inference::shape_to_string({}), "[]");
    EXPECT_EQ(inference::shape_to_string({2, 3}), "[2, 3]");
}

// ---------------------------------------------------------------------------
// Construction and factories
// ---------------------------------------------------------------------------

TEST(TensorConstruction, ScalarHasEmptyShapeAndOneElement) {
    const Tensor t = Tensor::scalar(3.5f);
    EXPECT_EQ(t.shape(), Shape{});
    EXPECT_EQ(t.strides(), Shape{});
    EXPECT_EQ(t.ndim(), 0u);
    EXPECT_EQ(t.numel(), 1u);
    EXPECT_FLOAT_EQ(t.at({}), 3.5f);
}

TEST(TensorConstruction, ShapeOnlyIsZeroInitialized) {
    const Tensor t(Shape{2, 3});
    EXPECT_EQ(t.ndim(), 2u);
    EXPECT_EQ(t.numel(), 6u);
    for (std::size_t i = 0; i < t.numel(); ++i) EXPECT_EQ(t.data()[i], 0.0f);
}

TEST(TensorConstruction, NumelAndNdimForSeveralRanks) {
    EXPECT_EQ(Tensor(Shape{7}).numel(), 7u);
    EXPECT_EQ(Tensor(Shape{7}).ndim(), 1u);

    EXPECT_EQ(Tensor(Shape{2, 5}).numel(), 10u);
    EXPECT_EQ(Tensor(Shape{2, 5}).ndim(), 2u);

    // 4-D, like a batch of images: {batch, channels, height, width}.
    const Tensor t(Shape{2, 3, 4, 5});
    EXPECT_EQ(t.numel(), 120u);
    EXPECT_EQ(t.ndim(), 4u);
    EXPECT_EQ(t.strides(), (Shape{60, 20, 5, 1}));
}

TEST(TensorConstruction, WithDataKeepsValuesInOrder) {
    const Tensor t({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_EQ(t.data()[0], 1.0f);
    EXPECT_EQ(t.data()[3], 4.0f);
}

TEST(TensorConstruction, Factories) {
    const Tensor z = Tensor::zeros({3, 2});
    const Tensor o = Tensor::ones({3, 2});
    const Tensor f = Tensor::full({3, 2}, -2.5f);
    EXPECT_EQ(z.shape(), (Shape{3, 2}));
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(z.data()[i], 0.0f);
        EXPECT_EQ(o.data()[i], 1.0f);
        EXPECT_EQ(f.data()[i], -2.5f);
    }
}

TEST(TensorConstruction, DataSizeMismatchThrows) {
    EXPECT_THROW(Tensor({2, 3}, std::vector<float>(5)), std::invalid_argument);
    EXPECT_THROW(Tensor({2, 3}, std::vector<float>(7)), std::invalid_argument);
    EXPECT_THROW(Tensor(Shape{}, std::vector<float>{}), std::invalid_argument);
}

TEST(TensorConstruction, ErrorMessageMentionsShape) {
    try {
        Tensor({2, 3}, std::vector<float>(5));
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("[2, 3]"), std::string::npos) << e.what();
    }
}

TEST(TensorConstruction, ZeroSizedDimensionThrows) {
    EXPECT_THROW(Tensor(Shape{0}), std::invalid_argument);
    EXPECT_THROW(Tensor(Shape{2, 0, 3}), std::invalid_argument);
    EXPECT_THROW(Tensor::zeros({3, 0}), std::invalid_argument);
    EXPECT_THROW(Tensor({0}, std::vector<float>{}), std::invalid_argument);
}

TEST(TensorAccess, SizeOfDimension) {
    const Tensor t(Shape{2, 3, 4});
    EXPECT_EQ(t.size(0), 2u);
    EXPECT_EQ(t.size(2), 4u);
    EXPECT_THROW(t.size(3), std::out_of_range);
    EXPECT_THROW(Tensor::scalar(1.0f).size(0), std::out_of_range);
}

// ---------------------------------------------------------------------------
// Element access
// ---------------------------------------------------------------------------

TEST(TensorAccess, AtMatchesManualOffsetMath) {
    // Filled with 0..23, so each element's value equals its flat offset. The offset of
    // {i, j, k} with strides {12, 4, 1} is i*12 + j*4 + k.
    const Tensor t = arange({2, 3, 4});
    EXPECT_EQ(t.at({0, 0, 0}), 0.0f);
    EXPECT_EQ(t.at({0, 0, 3}), 3.0f);
    EXPECT_EQ(t.at({0, 1, 0}), 4.0f);
    EXPECT_EQ(t.at({0, 2, 1}), 9.0f);
    EXPECT_EQ(t.at({1, 0, 0}), 12.0f);
    EXPECT_EQ(t.at({1, 2, 3}), 23.0f);

    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 4; ++k)
                EXPECT_EQ(t.at({i, j, k}), static_cast<float>(i * 12 + j * 4 + k));
}

TEST(TensorAccess, AtWritesTheRightElement) {
    Tensor t(Shape{2, 3});
    t.at({1, 2}) = 7.0f;
    t.at({0, 1}) = -1.0f;
    EXPECT_EQ(t.data()[1 * 3 + 2], 7.0f);
    EXPECT_EQ(t.data()[0 * 3 + 1], -1.0f);
    EXPECT_EQ(t.data()[0], 0.0f);  // untouched

    Tensor s = Tensor::scalar(1.0f);
    s.at({}) = 4.0f;
    EXPECT_EQ(s.at({}), 4.0f);
}

TEST(TensorAccess, AtWrongIndexCountThrows) {
    Tensor t(Shape{2, 3});
    EXPECT_THROW(t.at({1}), std::out_of_range);
    EXPECT_THROW(t.at({1, 1, 1}), std::out_of_range);
    EXPECT_THROW(t.at({}), std::out_of_range);
    EXPECT_THROW(Tensor::scalar(1.0f).at({0}), std::out_of_range);
}

TEST(TensorAccess, AtIndexOutOfRangeThrows) {
    const Tensor t(Shape{2, 3});
    EXPECT_THROW(t.at({2, 0}), std::out_of_range);
    EXPECT_THROW(t.at({0, 3}), std::out_of_range);
    EXPECT_NO_THROW(t.at({1, 2}));
}

TEST(TensorAccess, FillSetsEveryElement) {
    Tensor t(Shape{3, 3});
    t.fill(0.5f);
    for (std::size_t i = 0; i < t.numel(); ++i) EXPECT_EQ(t.data()[i], 0.5f);
}

// ---------------------------------------------------------------------------
// Reshape and flatten
// ---------------------------------------------------------------------------

TEST(TensorReshape, KeepsDataOrder) {
    const Tensor t = arange({2, 3});
    const Tensor r = t.reshape(Shape{3, 2});
    EXPECT_EQ(r.shape(), (Shape{3, 2}));
    EXPECT_EQ(r.strides(), (Shape{2, 1}));
    for (std::size_t i = 0; i < 6; ++i) EXPECT_EQ(r.data()[i], static_cast<float>(i));
    // Element {1, 0} of the 3x2 result is flat position 2.
    EXPECT_EQ(r.at({1, 0}), 2.0f);
}

TEST(TensorReshape, InfersMinusOne) {
    const Tensor t = arange({2, 3, 4});
    EXPECT_EQ(t.reshape({-1}).shape(), (Shape{24}));
    EXPECT_EQ(t.reshape({4, -1}).shape(), (Shape{4, 6}));
    EXPECT_EQ(t.reshape({-1, 2, 3}).shape(), (Shape{4, 2, 3}));
    EXPECT_EQ(t.reshape(std::vector<std::int64_t>{2, -1, 2}).shape(), (Shape{2, 6, 2}));
}

TEST(TensorReshape, BracedListWithoutMinusOneWorks) {
    const Tensor t = arange({2, 3});
    EXPECT_EQ(t.reshape({6, 1}).shape(), (Shape{6, 1}));
}

TEST(TensorReshape, InvalidReshapeThrows) {
    const Tensor t = arange({2, 3});
    EXPECT_THROW(t.reshape(Shape{4, 2}), std::invalid_argument);
    EXPECT_THROW(t.reshape({4, -1}), std::invalid_argument);    // 6 is not divisible by 4
    EXPECT_THROW(t.reshape({2, 0, 3}), std::invalid_argument);  // zero dimension
    EXPECT_THROW(t.reshape({-2, 3}), std::invalid_argument);    // negative, not -1
}

TEST(TensorReshape, TwoMinusOnesThrow) {
    const Tensor t = arange({2, 3});
    EXPECT_THROW(t.reshape({-1, -1}), std::invalid_argument);
}

TEST(TensorReshape, ReturnsIndependentCopy) {
    const Tensor t = arange({2, 3});
    Tensor r = t.reshape(Shape{6});
    r.at({0}) = 100.0f;
    EXPECT_EQ(t.at({0, 0}), 0.0f);
}

TEST(TensorReshape, Flatten) {
    const Tensor t = arange({2, 3, 4});
    const Tensor f = t.flatten();
    EXPECT_EQ(f.shape(), (Shape{24}));
    EXPECT_TRUE(allclose(f.reshape(Shape{2, 3, 4}), t));

    EXPECT_EQ(Tensor::scalar(2.0f).flatten().shape(), (Shape{1}));
}

// ---------------------------------------------------------------------------
// Copy and move
// ---------------------------------------------------------------------------

TEST(TensorCopyMove, CopyIsDeep) {
    const Tensor original = arange({2, 2});
    Tensor copy = original;
    copy.at({0, 0}) = 42.0f;
    EXPECT_EQ(original.at({0, 0}), 0.0f);
    EXPECT_NE(copy.data(), original.data());

    Tensor assigned(Shape{1});
    assigned = original;
    assigned.fill(9.0f);
    EXPECT_EQ(original.at({1, 1}), 3.0f);
    EXPECT_EQ(assigned.shape(), original.shape());
}

TEST(TensorCopyMove, MoveTransfersData) {
    Tensor source = arange({2, 3});
    const float* buffer = source.data();

    Tensor moved = std::move(source);
    EXPECT_EQ(moved.shape(), (Shape{2, 3}));
    EXPECT_EQ(moved.data(), buffer);  // same buffer: nothing was copied
    EXPECT_EQ(moved.at({1, 2}), 5.0f);

    Tensor target(Shape{1});
    target = std::move(moved);
    EXPECT_EQ(target.data(), buffer);
    EXPECT_EQ(target.strides(), (Shape{3, 1}));
}

// ---------------------------------------------------------------------------
// allclose
// ---------------------------------------------------------------------------

TEST(TensorAllclose, EqualAndNearlyEqual) {
    const Tensor a({3}, {1.0f, 2.0f, 3.0f});
    EXPECT_TRUE(allclose(a, a));
    EXPECT_TRUE(allclose(a, Tensor({3}, {1.0f, 2.0f + 1e-6f, 3.0f})));
}

TEST(TensorAllclose, DifferentValues) {
    const Tensor a({3}, {1.0f, 2.0f, 3.0f});
    EXPECT_FALSE(allclose(a, Tensor({3}, {1.0f, 2.1f, 3.0f})));
    // A looser tolerance accepts the same difference.
    EXPECT_TRUE(allclose(a, Tensor({3}, {1.0f, 2.1f, 3.0f}), 0.0f, 0.2f));
}

TEST(TensorAllclose, DifferentShapes) {
    EXPECT_FALSE(allclose(Tensor::zeros({2, 3}), Tensor::zeros({3, 2})));
    EXPECT_FALSE(allclose(Tensor::zeros({6}), Tensor::zeros({6, 1})));
}

TEST(TensorAllclose, NanIsNeverClose) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const Tensor a({1}, {nan});
    EXPECT_FALSE(allclose(a, a));
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

TEST(TensorPrint, Scalar) {
    EXPECT_EQ(Tensor::scalar(1.5f).to_string(), "tensor(1.5000, shape=[])");
}

TEST(TensorPrint, Small2D) {
    const Tensor t({2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, -5.0f});
    EXPECT_EQ(t.to_string(),
              "tensor([[ 0.0000,  1.0000,  2.0000],\n"
              "        [ 3.0000,  4.0000, -5.0000]], shape=[2, 3])");
}

TEST(TensorPrint, ThreeDimensionalHasBlankLineBetweenMatrices) {
    const std::string s = arange({2, 1, 2}).to_string();
    EXPECT_EQ(s,
              "tensor([[[0.0000, 1.0000]],\n"
              "\n"
              "        [[2.0000, 3.0000]]], shape=[2, 1, 2])");
}

TEST(TensorPrint, LargeTensorIsSummarized) {
    const std::string s = arange({2000}).to_string();
    EXPECT_NE(s.find("..."), std::string::npos);
    EXPECT_NE(s.find("1999.0000"), std::string::npos);  // last element shown
    EXPECT_EQ(s.find("1000.0000"), std::string::npos);  // middle element hidden
}

TEST(TensorPrint, StreamOperatorMatchesToString) {
    const Tensor t = arange({2, 2});
    std::ostringstream os;
    os << t;
    EXPECT_EQ(os.str(), t.to_string());
}
