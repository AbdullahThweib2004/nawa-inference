#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "inference/layers/layers.hpp"
#include "inference/tensor/ops.hpp"

using namespace inference;

namespace {

// True if every element is a finite number (no NaN, no inf).
bool all_finite(const Tensor& t) {
    for (std::size_t i = 0; i < t.numel(); ++i) {
        if (!std::isfinite(t.data()[i])) return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Layer interface
// ---------------------------------------------------------------------------

// Checked at compile time: layers must be movable but not copyable.
static_assert(!std::is_copy_constructible_v<Linear>);
static_assert(!std::is_copy_assignable_v<Linear>);
static_assert(std::is_move_constructible_v<Linear>);
static_assert(std::is_move_assignable_v<Linear>);
static_assert(!std::is_copy_constructible_v<ReLU>);
static_assert(std::is_abstract_v<Layer>);

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

TEST(Linear, HandComputedExample) {
    // x {1,2} · W {2,3} + b {3}
    const Tensor x({1, 2}, {1.0f, 2.0f});
    const Tensor W({2, 3}, {1.0f, 2.0f, 3.0f,  //
                            4.0f, 5.0f, 6.0f});
    const Tensor b({3}, {0.5f, -1.0f, 0.0f});
    const Linear layer(W, b);
    // x · W = [1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6] = [9, 12, 15]; + b = [9.5, 11, 15]
    EXPECT_TRUE(allclose(layer.forward(x), Tensor({1, 3}, {9.5f, 11.0f, 15.0f})));
}

TEST(Linear, BatchInput) {
    const Tensor W({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor b({3}, {1, 1, 1});
    const Linear layer(W, b);
    const Tensor x({3, 2}, {1, 0,  //
                            0, 1,  //
                            1, 1});
    // Row 0 picks W's first row, row 1 the second, row 2 their sum; then + 1.
    const Tensor y = layer.forward(x);
    EXPECT_EQ(y.shape(), (Shape{3, 3}));
    EXPECT_TRUE(allclose(y, Tensor({3, 3}, {2, 3, 4, 5, 6, 7, 6, 8, 10})));
}

TEST(Linear, WithoutBias) {
    const Tensor W({2, 3}, {1, 2, 3, 4, 5, 6});
    const Linear layer(W);
    EXPECT_FALSE(layer.bias().has_value());
    const Tensor x({1, 2}, {1.0f, 2.0f});
    EXPECT_TRUE(allclose(layer.forward(x), Tensor({1, 3}, {9, 12, 15})));
    EXPECT_EQ(layer.num_parameters(), 6u);
    EXPECT_EQ(layer.name(), "Linear(2 -> 3, no bias)");
}

TEST(Linear, WrongInputThrows) {
    const Linear layer(Tensor({2, 3}), Tensor({3}));
    EXPECT_THROW(layer.forward(Tensor({1, 4})), std::invalid_argument);     // wrong features
    EXPECT_THROW(layer.forward(Tensor({2})), std::invalid_argument);        // 1-D
    EXPECT_THROW(layer.forward(Tensor({1, 1, 2})), std::invalid_argument);  // 3-D
    EXPECT_THROW(layer.forward(Tensor::scalar(1.0f)), std::invalid_argument);
}

TEST(Linear, ForwardErrorMessageContainsShapes) {
    const Linear layer(Tensor({2, 3}));
    try {
        layer.forward(Tensor({1, 4}));
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("[1, 4]"), std::string::npos) << msg;
        EXPECT_NE(msg.find("[2, 3]"), std::string::npos) << msg;
    }
}

TEST(Linear, InvalidConstructorArgumentsThrow) {
    EXPECT_THROW(Linear(Tensor({2, 3}), Tensor({2})), std::invalid_argument);     // wrong size
    EXPECT_THROW(Linear(Tensor({2, 3}), Tensor({1, 3})), std::invalid_argument);  // wrong rank
    EXPECT_THROW(Linear(Tensor({6})), std::invalid_argument);                     // 1-D weight
    EXPECT_THROW(Linear(Tensor({2, 3, 1})), std::invalid_argument);               // 3-D weight
}

TEST(Linear, AccessorsNameAndParameterCount) {
    const Linear layer(Tensor({784, 128}), Tensor({128}));
    EXPECT_EQ(layer.in_features(), 784u);
    EXPECT_EQ(layer.out_features(), 128u);
    EXPECT_EQ(layer.weight().shape(), (Shape{784, 128}));
    ASSERT_TRUE(layer.bias().has_value());
    EXPECT_EQ(layer.bias()->shape(), (Shape{128}));
    EXPECT_EQ(layer.num_parameters(), 784u * 128u + 128u);  // 100,480
    EXPECT_EQ(layer.name(), "Linear(784 -> 128)");
}

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

TEST(ReLU, NegativeZeroPositive) {
    const ReLU relu;
    const Tensor x({5}, {-2.0f, -0.5f, 0.0f, 0.5f, 3.0f});
    EXPECT_TRUE(allclose(relu.forward(x), Tensor({5}, {0, 0, 0, 0.5f, 3})));
}

TEST(ReLU, KeepsShapeFor1D2D4D) {
    const ReLU relu;
    for (const Shape& shape : {Shape{6}, Shape{2, 3}, Shape{2, 1, 3, 2}}) {
        const Tensor x = Tensor::full(shape, -1.0f);
        const Tensor y = relu.forward(x);
        EXPECT_EQ(y.shape(), shape);
        EXPECT_TRUE(allclose(y, Tensor::zeros(shape)));
        EXPECT_TRUE(allclose(relu.forward(-x), -x));  // positive values pass through
    }
    EXPECT_EQ(relu.num_parameters(), 0u);
    EXPECT_EQ(relu.name(), "ReLU");
}

TEST(ReLU, DoesNotModifyInput) {
    const Tensor x({2}, {-1.0f, 1.0f});
    ReLU{}.forward(x);
    EXPECT_EQ(x.data()[0], -1.0f);
}

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------

TEST(Sigmoid, KnownValues) {
    const Sigmoid sigmoid;
    EXPECT_FLOAT_EQ(sigmoid.forward(Tensor::scalar(0.0f)).at({}), 0.5f);
    EXPECT_NEAR(sigmoid.forward(Tensor::scalar(1.0f)).at({}), 0.7310586f, 1e-6f);
    EXPECT_NEAR(sigmoid.forward(Tensor::scalar(-1.0f)).at({}), 0.26894142f, 1e-6f);
}

TEST(Sigmoid, Symmetry) {
    // sigmoid(-x) == 1 - sigmoid(x)
    const Sigmoid sigmoid;
    const Tensor x({2, 3}, {0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f});
    EXPECT_TRUE(allclose(sigmoid.forward(-x), 1.0f - sigmoid.forward(x), 1e-5f, 1e-6f));
}

TEST(Sigmoid, ExtremeInputsSaturateWithoutNanOrInf) {
    const Sigmoid sigmoid;
    const Tensor y = sigmoid.forward(Tensor({2}, {-1000.0f, 1000.0f}));
    EXPECT_TRUE(all_finite(y));
    EXPECT_EQ(y.data()[0], 0.0f);
    EXPECT_EQ(y.data()[1], 1.0f);
}

TEST(Sigmoid, KeepsShape) {
    const Tensor y = Sigmoid{}.forward(Tensor({2, 1, 3}));
    EXPECT_EQ(y.shape(), (Shape{2, 1, 3}));
    EXPECT_TRUE(allclose(y, Tensor::full({2, 1, 3}, 0.5f)));
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

TEST(Softmax, KnownValues) {
    const Softmax softmax;
    const Tensor y = softmax.forward(Tensor({3}, {1.0f, 2.0f, 3.0f}));
    EXPECT_TRUE(allclose(y, Tensor({3}, {0.09003057f, 0.24472847f, 0.66524096f})));
}

TEST(Softmax, EveryRowSumsToOne) {
    const Softmax softmax;
    const Tensor x({3, 4}, {1, 2, 3, 4,   //
                            -1, 0, 1, 0,  //
                            5, 5, 5, 5});
    const Tensor y = softmax.forward(x);
    EXPECT_EQ(y.shape(), (Shape{3, 4}));
    EXPECT_TRUE(allclose(sum(y, -1), Tensor::ones({3})));
    // A row of equal values gives a uniform distribution.
    EXPECT_NEAR(y.at({2, 0}), 0.25f, 1e-6f);
}

TEST(Softmax, InvariantToAddingAConstant) {
    const Softmax softmax;
    const Tensor x({2, 3}, {0.5f, -1.0f, 2.0f, 3.0f, 3.0f, 0.0f});
    EXPECT_TRUE(allclose(softmax.forward(x + 5.0f), softmax.forward(x)));
    EXPECT_TRUE(allclose(softmax.forward(x - 7.0f), softmax.forward(x)));
}

TEST(Softmax, LargeInputsDoNotOverflow) {
    const Softmax softmax;
    const Tensor big = softmax.forward(Tensor({3}, {1000.0f, 1001.0f, 1002.0f}));
    EXPECT_TRUE(all_finite(big));
    EXPECT_TRUE(allclose(big, softmax.forward(Tensor({3}, {0.0f, 1.0f, 2.0f}))));
}

TEST(Softmax, AxisParameter) {
    const Tensor x({2, 3}, {1, 2, 3,  //
                            1, 4, 9});
    const Tensor cols = Softmax(0).forward(x);
    EXPECT_TRUE(allclose(sum(cols, 0), Tensor::ones({3})));
    EXPECT_NEAR(cols.at({0, 0}), 0.5f, 1e-6f);  // column 0 has equal values

    EXPECT_TRUE(allclose(Softmax(1).forward(x), Softmax(-1).forward(x)));
    EXPECT_EQ(Softmax(0).name(), "Softmax(axis=0)");
    EXPECT_EQ(Softmax().axis(), -1);
}

TEST(Softmax, InvalidAxisThrows) {
    EXPECT_THROW(Softmax(2).forward(Tensor({2, 3})), std::out_of_range);
    EXPECT_THROW(Softmax().forward(Tensor::scalar(1.0f)), std::out_of_range);
}

// ---------------------------------------------------------------------------
// Polymorphism
// ---------------------------------------------------------------------------

TEST(LayerStack, RunsForwardThroughBasePointers) {
    std::vector<std::unique_ptr<Layer>> layers;
    layers.push_back(std::make_unique<Linear>(Tensor::full({4, 3}, 0.1f), Tensor({3})));
    layers.push_back(std::make_unique<ReLU>());
    layers.push_back(std::make_unique<Linear>(Tensor::full({3, 2}, 0.2f), Tensor({2})));
    layers.push_back(std::make_unique<Softmax>());

    const std::vector<Shape> expected_shapes = {{5, 3}, {5, 3}, {5, 2}, {5, 2}};
    const std::vector<std::string> expected_names = {"Linear(4 -> 3)", "ReLU", "Linear(3 -> 2)",
                                                     "Softmax(axis=-1)"};

    Tensor x = Tensor::ones({5, 4});  // batch of 5 samples, 4 features each
    for (std::size_t i = 0; i < layers.size(); ++i) {
        // Each call goes to the right derived class through the vtable.
        x = layers[i]->forward(x);
        EXPECT_EQ(x.shape(), expected_shapes[i]) << layers[i]->name();
        EXPECT_EQ(layers[i]->name(), expected_names[i]);
    }
    EXPECT_TRUE(allclose(sum(x, -1), Tensor::ones({5})));

    std::size_t total = 0;
    for (const auto& layer : layers) total += layer->num_parameters();
    EXPECT_EQ(total, (4u * 3u + 3u) + (3u * 2u + 2u));
}

TEST(LayerStack, LayersCanBeMoved) {
    Linear a(Tensor({2, 3}), Tensor({3}));
    const Linear b = std::move(a);
    EXPECT_EQ(b.in_features(), 2u);
    EXPECT_EQ(b.out_features(), 3u);
}
