#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "inference/layers/layer.hpp"
#include "inference/tensor/gemm.hpp"
#include "inference/tensor/tensor.hpp"

namespace inference {

// Fully connected layer: y = x · W + b.
//
// WEIGHT LAYOUT: weight has shape {in_features, out_features}. This is the TRANSPOSE of
// PyTorch's nn.Linear, which stores {out_features, in_features} and computes x · Wᵀ + b.
// Storing {in, out} lets forward() be a plain matmul with no transpose. The Python
// exporter (step 5) transposes PyTorch weights before saving them.
class Linear : public Layer {
public:
    // weight: {in_features, out_features}. bias (optional): {out_features}.
    // Throws std::invalid_argument if weight is not 2-D or the bias shape doesn't match.
    explicit Linear(Tensor weight, std::optional<Tensor> bias = std::nullopt);

    // input: {batch, in_features} -> output: {batch, out_features}.
    // Throws std::invalid_argument on a wrong rank or feature size.
    Tensor forward(const Tensor& input) const override;

    std::string name() const override;
    std::size_t num_parameters() const override;

    std::size_t in_features() const noexcept { return weight_.shape()[0]; }
    std::size_t out_features() const noexcept { return weight_.shape()[1]; }
    const Tensor& weight() const noexcept { return weight_; }
    const std::optional<Tensor>& bias() const noexcept { return bias_; }

    // The weight packed for the GEMM kernel. Weights are constant during inference, so they
    // are packed once here, at construction (= model load), instead of on every forward().
    const PackedMatrix& packed_weight() const noexcept { return packed_; }

private:
    Tensor weight_;               // {in_features, out_features}
    std::optional<Tensor> bias_;  // {out_features}, or empty for no bias
    PackedMatrix packed_;         // weight_, packed for the active GEMM kernel
};

}  // namespace inference
