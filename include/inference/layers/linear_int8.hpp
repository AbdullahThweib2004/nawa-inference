#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "inference/layers/layer.hpp"
#include "inference/tensor/int8.hpp"
#include "inference/tensor/tensor.hpp"

namespace inference {

class Linear;

// A fully connected layer with INT8 weights (step 9.6): y = x · W + b, computed as
//     dequantize(quantize_rows(x) · W_int8) + b
// (see tensor/int8.hpp). Weights are quantized per output channel; activations are quantized
// per row at runtime. The bias stays float32, and so does everything after this layer.
class LinearInt8 : public DenseLayer {
public:
    // Throws std::invalid_argument if the bias shape isn't {out_features}.
    LinearInt8(QuantizedMatrix weight, std::optional<Tensor> bias);

    // Quantizes a float32 Linear layer.
    static LinearInt8 from_linear(const Linear& linear);

    // input: {batch, in_features} -> {batch, out_features}. Throws std::invalid_argument on
    // a wrong rank or feature size.
    Tensor forward(const Tensor& input) const override;
    void forward_into(const float* input, std::size_t rows, float* output,
                      bool fuse_relu = false) const override;

    std::string name() const override;
    // Weights + biases, counted like Linear (the per-channel scales are derived values).
    std::size_t num_parameters() const override;

    std::size_t in_features() const noexcept override { return weight_.in_features(); }
    std::size_t out_features() const noexcept override { return weight_.out_features(); }
    const QuantizedMatrix& weight() const noexcept { return weight_; }
    const std::optional<Tensor>& bias() const noexcept { return bias_; }

private:
    QuantizedMatrix weight_;
    std::optional<Tensor> bias_;
};

}  // namespace inference
