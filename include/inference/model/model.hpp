#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "inference/layers/layer.hpp"
#include "inference/tensor/tensor.hpp"

namespace inference {

// A network loaded from a .nawa file (docs/model_format.md): normalization metadata plus an
// ordered list of layers. Immutable after loading; every method is const.
//
//     Model model("models/mnist_mlp.nawa");
//     auto predictions = model.classify(raw_pixels);   // raw_pixels: {N, 784}, values 0..255
class Model {
public:
    struct Metadata {
        Shape input_shape;         // one sample, without the batch dimension, e.g. {784}
        float pixel_scale = 1.0f;  // x = (raw * pixel_scale - mean) / stddev
        std::vector<float> mean;
        std::vector<float> stddev;
    };

    struct Prediction {
        std::size_t label;  // index of the largest output
        float confidence;   // that output's value (a probability if the model ends in Softmax)
    };

    // Loads and validates a model file. Throws ModelFormatError if the file breaks the spec
    // or the layers don't chain together, and std::runtime_error if it can't be read.
    static Model load(const std::string& path);

    // Same as load(path), so `Model model("model.nawa");` works.
    explicit Model(const std::string& path);

    // Movable (it owns its layers through unique_ptr), not copyable.
    Model(Model&&) noexcept = default;
    Model& operator=(Model&&) noexcept = default;

    // Normalizes raw input: (raw * pixel_scale - mean) / std.
    // Input: {N, F} or a single sample {F} (F = input_features()); the result is {N, F}.
    Tensor preprocess(const Tensor& raw) const;

    // Runs every layer on already-normalized input ({N, F} or {F}). Returns {N, outputs}.
    Tensor forward(const Tensor& normalized) const;

    // preprocess + forward.
    Tensor predict(const Tensor& raw) const;

    // Like forward(), but returns the output of every layer, in order (for debugging).
    std::vector<Tensor> forward_trace(const Tensor& normalized) const;

    // One prediction per batch row of raw input.
    std::vector<Prediction> classify(const Tensor& raw) const;

    // Human-readable description: layers, shapes, parameter counts, metadata.
    std::string summary() const;

    std::size_t num_parameters() const;

    // Number of values in one input sample (the product of metadata().input_shape).
    std::size_t input_features() const noexcept { return input_features_; }
    const Metadata& metadata() const noexcept { return metadata_; }
    const std::vector<std::unique_ptr<Layer>>& layers() const noexcept { return layers_; }

private:
    Model() = default;  // used by load()

    // Checks that input is {N, F} or {F} and returns it as {N, F}.
    Tensor as_batch(const Tensor& input, const char* fn) const;

    std::string path_;
    Metadata metadata_;
    std::size_t input_features_ = 0;
    std::vector<std::unique_ptr<Layer>> layers_;
};

}  // namespace inference
