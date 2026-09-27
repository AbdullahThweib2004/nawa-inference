#include "inference/layers/activations.hpp"

#include <cmath>
#include <string>

#include "inference/tensor/ops.hpp"

namespace inference {

Tensor ReLU::forward(const Tensor& input) const {
    Tensor output = input;  // copy, then transform in place
    float* p = output.data();
    for (std::size_t i = 0; i < output.numel(); ++i) {
        if (p[i] < 0.0f) p[i] = 0.0f;
    }
    return output;
}

Tensor Sigmoid::forward(const Tensor& input) const {
    Tensor output = input;
    float* p = output.data();
    for (std::size_t i = 0; i < output.numel(); ++i) {
        const float x = p[i];
        // Both branches only call exp() with a non-positive argument (see header).
        if (x >= 0.0f) {
            p[i] = 1.0f / (1.0f + std::exp(-x));
        } else {
            const float e = std::exp(x);
            p[i] = e / (1.0f + e);
        }
    }
    return output;
}

Tensor Softmax::forward(const Tensor& input) const {
    // keepdims = true keeps the reduced axis as size 1, so the results broadcast back
    // against the input: e.g. {2, 3} - {2, 1} subtracts each row's max from that row.
    const Tensor shifted = input - max(input, axis_, true);
    const Tensor e = exp(shifted);
    return e / sum(e, axis_, true);
}

std::string Softmax::name() const { return "Softmax(axis=" + std::to_string(axis_) + ")"; }

}  // namespace inference
