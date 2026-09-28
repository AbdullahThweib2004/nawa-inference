#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "inference/layers/layer.hpp"
#include "inference/runtime/workspace.hpp"
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

    // preprocess + all layers, through the execution plan (fused Linear+ReLU). Convenience
    // version: allocates a temporary Workspace and returns a copy of the result.
    Tensor predict(const Tensor& raw) const;

    // The allocation-free version: all intermediate results and the output live in
    // `workspace`, whose buffers are reused across calls (0 heap allocations once they have
    // grown to the batch size). Returns a reference into the workspace, valid until its next
    // use. One Workspace per thread; the Model itself can be shared (see workspace.hpp).
    const Tensor& predict(const Tensor& raw, Workspace& workspace) const;

    // One recorded step of predict_trace().
    struct TraceStep {
        std::string name;  // "preprocess", a layer name, or "<Linear...> + ReLU" when fused
        Tensor output;     // that step's output, {N, features}
    };

    // Runs EXACTLY what predict(raw, workspace) runs (the fused execution plan) and returns
    // every step's output, starting with the normalized input. With a fused Linear+ReLU, the
    // hidden activations appear after the ReLU, as the fused step's output. Results are
    // bit-identical to predict(). Allocates (it copies each output); for inspection only.
    std::vector<TraceStep> predict_trace(const Tensor& raw, Workspace& workspace) const;

    // Like forward(), but returns the output of every layer, in order (for debugging).
    std::vector<Tensor> forward_trace(const Tensor& normalized) const;

    // One prediction per batch row of raw input.
    std::vector<Prediction> classify(const Tensor& raw) const;
    std::vector<Prediction> classify(const Tensor& raw, Workspace& workspace) const;

    // Human-readable description: layers, shapes, parameter counts, metadata.
    std::string summary() const;

    std::size_t num_parameters() const;

    // Writes the model in the .nawa format (docs/model_format.md): version 1 for float32
    // models, version 2 if it contains LinearInt8 layers. A loaded float32 model saves
    // byte-identical to the file it came from.
    void save(const std::string& path) const;

    // A copy with every Linear layer quantized to LinearInt8 (tensor/int8.hpp). Metadata and
    // the other layers are unchanged; the execution plan is rebuilt (Linear+ReLU stays fused).
    Model quantize() const;

    // 1 for float32 models, 2 if any layer is LinearInt8.
    std::uint32_t format_version() const;

    // Number of values in one input sample (the product of metadata().input_shape).
    std::size_t input_features() const noexcept { return input_features_; }
    const Metadata& metadata() const noexcept { return metadata_; }
    const std::vector<std::unique_ptr<Layer>>& layers() const noexcept { return layers_; }

    // One step of the execution plan built at load time. Usually one layer; a Linear followed
    // by a ReLU becomes ONE step (fuse_relu), applied in the GEMM epilogue.
    struct Step {
        const Layer* layer;      // points into layers()
        bool fuse_relu = false;  // Linear only: also apply the following ReLU
    };
    const std::vector<Step>& plan() const noexcept { return plan_; }

private:
    Model() = default;  // used by load()

    // Shared by predict() and predict_trace(): runs the plan; records steps if trace != nullptr.
    const Tensor& run_plan(const Tensor& raw, Workspace& workspace,
                           std::vector<TraceStep>* trace) const;

    // Builds plan_ from layers_ (fusing Linear + ReLU).
    void build_plan();

    // Normalizes n raw values into `out` (the formula of preprocess()).
    void preprocess_into(const float* raw, std::size_t n, float* out) const;

    // Checks that input is {N, F} or {F} and returns it as {N, F}.
    Tensor as_batch(const Tensor& input, const char* fn) const;

    std::string path_;
    Metadata metadata_;
    std::size_t input_features_ = 0;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::vector<Step> plan_;
};

}  // namespace inference
