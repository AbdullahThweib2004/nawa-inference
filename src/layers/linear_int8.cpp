#include "inference/layers/linear_int8.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include "inference/layers/linear.hpp"

namespace inference {

LinearInt8::LinearInt8(QuantizedMatrix weight, std::optional<Tensor> bias)
    : weight_(std::move(weight)), bias_(std::move(bias)) {
    if (bias_ && bias_->shape() != Shape{out_features()}) {
        throw std::invalid_argument("LinearInt8: bias must have shape " +
                                    shape_to_string({out_features()}) + ", got " +
                                    shape_to_string(bias_->shape()));
    }
}

LinearInt8 LinearInt8::from_linear(const Linear& linear) {
    return LinearInt8(QuantizedMatrix::quantize(linear.weight().data(), linear.in_features(),
                                                linear.out_features()),
                      linear.bias());
}

Tensor LinearInt8::forward(const Tensor& input) const {
    if (input.ndim() != 2 || input.size(1) != in_features()) {
        throw std::invalid_argument("LinearInt8::forward: expected input of shape {batch, " +
                                    std::to_string(in_features()) + "}, got " +
                                    shape_to_string(input.shape()));
    }
    Tensor output({input.size(0), out_features()});
    forward_into(input.data(), input.size(0), output.data());
    return output;
}

void LinearInt8::forward_into(const float* input, std::size_t rows, float* output,
                              bool fuse_relu) const {
    gemm_int8(input, rows, weight_, output,
              GemmEpilogue{bias_ ? bias_->data() : nullptr, fuse_relu});
}

std::string LinearInt8::name() const {
    std::string result =
        "LinearInt8(" + std::to_string(in_features()) + " -> " + std::to_string(out_features());
    if (!bias_) result += ", no bias";
    return result + ")";
}

std::size_t LinearInt8::num_parameters() const {
    return in_features() * out_features() + (bias_ ? bias_->numel() : 0);
}

}  // namespace inference
