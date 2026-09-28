#include "inference/layers/linear.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace inference {

Linear::Linear(Tensor weight, std::optional<Tensor> bias)
    : weight_(std::move(weight)), bias_(std::move(bias)) {
    if (weight_.ndim() != 2) {
        throw std::invalid_argument(
            "Linear: weight must be 2-D {in_features, out_features}, got shape " +
            shape_to_string(weight_.shape()));
    }
    if (bias_ && bias_->shape() != Shape{out_features()}) {
        throw std::invalid_argument("Linear: bias must have shape " +
                                    shape_to_string({out_features()}) + " to match weight " +
                                    shape_to_string(weight_.shape()) + ", got " +
                                    shape_to_string(bias_->shape()));
    }
    packed_ = PackedMatrix::pack(weight_.data(), in_features(), out_features());
}

Tensor Linear::forward(const Tensor& input) const {
    if (input.ndim() != 2 || input.size(1) != in_features()) {
        throw std::invalid_argument("Linear::forward: expected input of shape {batch, " +
                                    std::to_string(in_features()) + "}, got " +
                                    shape_to_string(input.shape()) + " (weight " +
                                    shape_to_string(weight_.shape()) + ")");
    }
    // {batch, in} · {in, out} -> {batch, out}, using the weights packed at construction.
    Tensor output({input.size(0), out_features()});
    forward_into(input.data(), input.size(0), output.data());
    return output;
}

void Linear::forward_into(const float* input, std::size_t rows, float* output,
                          bool fuse_relu) const {
    // The bias is added in the GEMM epilogue, as each output is written, instead of a
    // separate broadcast add over the whole output afterwards (bit-identical results).
    gemm(input, rows, packed_, output, GemmEpilogue{bias_ ? bias_->data() : nullptr, fuse_relu});
}

std::string Linear::name() const {
    std::string result =
        "Linear(" + std::to_string(in_features()) + " -> " + std::to_string(out_features());
    if (!bias_) result += ", no bias";
    return result + ")";
}

std::size_t Linear::num_parameters() const {
    return weight_.numel() + (bias_ ? bias_->numel() : 0);
}

}  // namespace inference
