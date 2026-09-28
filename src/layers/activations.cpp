#include "inference/layers/activations.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "inference/tensor/ops.hpp"

namespace inference {

Tensor ReLU::forward(const Tensor& input) const {
    Tensor output = input;  // copy, then transform in place
    relu_inplace(output.data(), output.numel());
    return output;
}

Tensor Sigmoid::forward(const Tensor& input) const {
    Tensor output = input;
    sigmoid_inplace(output.data(), output.numel());
    return output;
}

Tensor Softmax::forward(const Tensor& input) const {
    const auto ndim = static_cast<std::int64_t>(input.ndim());
    const std::int64_t axis = axis_ < 0 ? axis_ + ndim : axis_;
    if (axis < 0 || axis >= ndim) {
        throw std::out_of_range("Softmax: axis " + std::to_string(axis_) +
                                " is out of range for shape " + shape_to_string(input.shape()));
    }
    if (axis == ndim - 1) {
        // The common case: softmax over each contiguous row, in one pass per row.
        Tensor output = input;
        const std::size_t cols = input.shape().back();
        softmax_rows_inplace(output.data(), output.numel() / cols, cols);
        return output;
    }
    // Any other axis: the tensor-level form. keepdims keeps the reduced axis as size 1 so the
    // results broadcast back against the input, e.g. {2, 3} - {2, 1}.
    const Tensor shifted = input - max(input, axis_, true);
    const Tensor e = exp(shifted);
    return e / sum(e, axis_, true);
}

std::string Softmax::name() const { return "Softmax(axis=" + std::to_string(axis_) + ")"; }

}  // namespace inference
