#pragma once

#include <cstddef>
#include <string>

#include "inference/tensor/tensor.hpp"

namespace inference {

// Base class for every neural network layer. Inference only: layers compute outputs from
// inputs and fixed parameters; there are no gradients and no training.
//
// Layers are used polymorphically, e.g. a model holds a std::vector<std::unique_ptr<Layer>>
// and calls forward() on each one in turn without knowing its concrete type.
class Layer {
public:
    // Virtual so that deleting a derived layer through a Layer* (which is what
    // std::unique_ptr<Layer> does) runs the derived destructor and frees its tensors.
    virtual ~Layer() = default;

    // Non-copyable: parameters can be large, so an accidental copy would be expensive.
    // Deleting copy here also stops "slicing" (copying only the Layer part of a derived
    // object). Movable, so layers can be built and then moved into a model.
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    Layer(Layer&&) = default;
    Layer& operator=(Layer&&) = default;

    // Computes the layer's output. const because inference layers hold no mutable state,
    // so the same layer can safely be run many times (and later, from several threads).
    virtual Tensor forward(const Tensor& input) const = 0;

    // Human-readable description, e.g. "Linear(784 -> 128)".
    virtual std::string name() const = 0;

    // Number of learned values (weights + biases). 0 for layers without parameters.
    virtual std::size_t num_parameters() const { return 0; }

protected:
    // Only derived classes can construct a Layer.
    Layer() = default;
};

// A fully connected layer, float32 (Linear) or int8 (LinearInt8). The model treats both the
// same way: it checks that in/out sizes chain together, and runs them through forward_into,
// optionally with a following ReLU fused into the output pass.
class DenseLayer : public Layer {
public:
    virtual std::size_t in_features() const noexcept = 0;
    virtual std::size_t out_features() const noexcept = 0;

    // output {rows, out_features} = layer(input {rows, in_features}), raw buffers, no shape
    // checks (the caller guarantees them). fuse_relu applies a following ReLU in the same pass.
    virtual void forward_into(const float* input, std::size_t rows, float* output,
                              bool fuse_relu = false) const = 0;
};

}  // namespace inference
