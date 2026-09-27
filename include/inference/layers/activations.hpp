#pragma once

#include <cstdint>
#include <string>

#include "inference/layers/layer.hpp"
#include "inference/tensor/tensor.hpp"

// Activation layers. They have no parameters and keep the input shape.
namespace inference {

// ReLU(x) = max(0, x), element-wise. Works on any shape.
class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) const override;
    std::string name() const override { return "ReLU"; }
};

// Sigmoid(x) = 1 / (1 + e^-x), element-wise. Works on any shape. Output is in [0, 1].
//
// Computed in a numerically stable way: e^-x overflows to inf for large negative x, so
// for x < 0 the equivalent form e^x / (1 + e^x) is used instead. That way exp() only
// ever receives a value <= 0, and its result stays in (0, 1].
class Sigmoid : public Layer {
public:
    Tensor forward(const Tensor& input) const override;
    std::string name() const override { return "Sigmoid"; }
};

// Softmax along one axis (default: last): exp(x_i) / sum_j exp(x_j).
// Each slice along the axis becomes a probability distribution (non-negative, sums to 1).
//
// Numerically stable form: the maximum along the axis is subtracted first. Softmax is
// unchanged by adding a constant to its inputs, and after the shift the largest exponent
// is exp(0) = 1, so nothing can overflow and the sum is at least 1 (no division by 0).
class Softmax : public Layer {
public:
    // Negative axis counts from the end. An invalid axis throws std::out_of_range in
    // forward(), when the input's rank is known.
    explicit Softmax(std::int64_t axis = -1) : axis_(axis) {}

    Tensor forward(const Tensor& input) const override;
    std::string name() const override;

    std::int64_t axis() const noexcept { return axis_; }

private:
    std::int64_t axis_;
};

}  // namespace inference
