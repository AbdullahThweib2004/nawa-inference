#include "inference/model/model.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "inference/layers/activations.hpp"
#include "inference/layers/linear.hpp"
#include "inference/layers/linear_int8.hpp"
#include "inference/model/binary_io.hpp"
#include "inference/model/tensor_io.hpp"
#include "inference/tensor/ops.hpp"

namespace inference {

namespace {

constexpr std::string_view kModelMagic = "NAWA";
// Version 1: layer types 1-4. Version 2 adds type 5 (LinearInt8). Readers accept both;
// save() writes version 1 unless the model contains an int8 layer.
constexpr std::uint32_t kModelVersionFloat = 1;
constexpr std::uint32_t kModelVersionInt8 = 2;
constexpr std::uint32_t kMaxInputNdim = 8;

// Layer type ids (docs/model_format.md, "Layers").
constexpr std::uint32_t kLayerLinear = 1;
constexpr std::uint32_t kLayerReLU = 2;
constexpr std::uint32_t kLayerSigmoid = 3;
constexpr std::uint32_t kLayerSoftmax = 4;
constexpr std::uint32_t kLayerLinearInt8 = 5;

bool is_finite_positive(float x) { return std::isfinite(x) && x > 0.0f; }

// Reads the metadata section (everything between the version and num_layers).
Model::Metadata read_metadata(BinaryReader& r) {
    Model::Metadata meta;

    const std::size_t ndim_offset = r.offset();
    const std::uint32_t ndim = r.read_u32("input_ndim");
    if (ndim < 1 || ndim > kMaxInputNdim) {
        r.fail(ndim_offset, "expected input_ndim in 1.." + std::to_string(kMaxInputNdim) +
                                ", found " + std::to_string(ndim));
    }
    for (std::uint32_t i = 0; i < ndim; ++i) {
        const std::size_t dim_offset = r.offset();
        const std::uint64_t dim = r.read_u64("input_dims[" + std::to_string(i) + "]");
        if (dim == 0) {
            r.fail(dim_offset, "expected input_dims[" + std::to_string(i) + "] >= 1, found 0");
        }
        meta.input_shape.push_back(static_cast<std::size_t>(dim));
    }

    const std::size_t scale_offset = r.offset();
    meta.pixel_scale = r.read_f32("pixel_scale");
    if (!is_finite_positive(meta.pixel_scale)) {
        r.fail(scale_offset,
               "expected a finite pixel_scale > 0, found " + std::to_string(meta.pixel_scale));
    }

    const std::size_t count_offset = r.offset();
    const std::uint32_t norm_count = r.read_u32("norm_count");
    // The spec allows one mean/std pair per channel. The runtime only implements a single
    // pair (MNIST is one grayscale channel); reject anything else clearly rather than guess
    // at a channel layout.
    if (norm_count != 1) {
        r.fail(count_offset,
               "expected norm_count 1 (the only value this runtime supports), "
               "found " +
                   std::to_string(norm_count));
    }
    const std::size_t mean_offset = r.offset();
    meta.mean = r.read_f32_array(norm_count, "mean");
    const std::size_t std_offset = r.offset();
    meta.stddev = r.read_f32_array(norm_count, "std");
    if (!std::isfinite(meta.mean[0])) {
        r.fail(mean_offset, "expected a finite mean, found " + std::to_string(meta.mean[0]));
    }
    if (!is_finite_positive(meta.stddev[0])) {
        r.fail(std_offset, "expected a finite std > 0, found " + std::to_string(meta.stddev[0]));
    }
    return meta;
}

// THE LAYER FACTORY: reads one layer (u32 type id + payload) and builds the matching
// concrete Layer. The caller only ever sees std::unique_ptr<Layer>; this switch is the one
// place that knows about every layer type. Supporting a new layer means adding a type id
// to the spec and one case here.
std::unique_ptr<Layer> read_layer(BinaryReader& r, std::size_t index, std::uint32_t version) {
    const std::string name = "layer " + std::to_string(index);
    const std::size_t type_offset = r.offset();
    const std::uint32_t type_id = r.read_u32(name + " type id");

    switch (type_id) {
        case kLayerLinear: {
            const std::size_t bias_flag_offset = r.offset();
            const std::uint8_t has_bias = r.read_u8(name + " has_bias");
            if (has_bias > 1) {
                r.fail(bias_flag_offset, name + " (Linear): expected has_bias 0 or 1, found " +
                                             std::to_string(has_bias));
            }
            const std::size_t weight_offset = r.offset();
            Tensor weight = read_tensor_block(r, name + " weight");
            if (weight.ndim() != 2) {
                r.fail(weight_offset, name +
                                          " (Linear): expected a 2-D weight "
                                          "{in_features, out_features}, found shape " +
                                          shape_to_string(weight.shape()));
            }
            std::optional<Tensor> bias;
            if (has_bias == 1) {
                const std::size_t bias_offset = r.offset();
                bias = read_tensor_block(r, name + " bias");
                const Shape expected{weight.size(1)};
                if (bias->shape() != expected) {
                    r.fail(bias_offset, name + " (Linear): expected bias shape " +
                                            shape_to_string(expected) + " to match weight " +
                                            shape_to_string(weight.shape()) + ", found " +
                                            shape_to_string(bias->shape()));
                }
            }
            return std::make_unique<Linear>(std::move(weight), std::move(bias));
        }
        case kLayerReLU:
            return std::make_unique<ReLU>();
        case kLayerSigmoid:
            return std::make_unique<Sigmoid>();
        case kLayerSoftmax:
            return std::make_unique<Softmax>(r.read_i32(name + " axis"));
        case kLayerLinearInt8: {
            if (version < kModelVersionInt8) {
                r.fail(type_offset, name +
                                        ": layer type 5 (LinearInt8) requires format version 2, "
                                        "found it in a version " +
                                        std::to_string(version) + " file");
            }
            const std::size_t bias_flag_offset = r.offset();
            const std::uint8_t has_bias = r.read_u8(name + " has_bias");
            if (has_bias > 1) {
                r.fail(bias_flag_offset, name + " (LinearInt8): expected has_bias 0 or 1, found " +
                                             std::to_string(has_bias));
            }
            const std::size_t dims_offset = r.offset();
            const std::uint32_t in = r.read_u32(name + " in_features");
            const std::uint32_t out = r.read_u32(name + " out_features");
            // in >= 1, out >= 1, and in * 127 * 127 must fit in int32 (exact accumulation).
            if (in == 0 || out == 0 || in > 2147483647u / (127u * 127u)) {
                r.fail(dims_offset, name +
                                        " (LinearInt8): expected 1 <= in_features <= 133144 and "
                                        "out_features >= 1, found " +
                                        std::to_string(in) + " x " + std::to_string(out));
            }
            const std::size_t scales_offset = r.offset();
            const std::vector<float> scales = r.read_f32_array(out, name + " scales");
            for (std::size_t j = 0; j < scales.size(); ++j) {
                if (!is_finite_positive(scales[j])) {
                    r.fail(scales_offset + 4 * j,
                           name + " (LinearInt8): expected scale[" + std::to_string(j) +
                               "] finite and > 0, found " + std::to_string(scales[j]));
                }
            }
            const std::size_t weights_offset = r.offset();
            const std::size_t count = static_cast<std::size_t>(in) * out;
            const std::span<const std::byte> raw = r.read_bytes(count, name + " int8 weights");
            std::vector<std::int8_t> q(count);
            std::memcpy(q.data(), raw.data(), count);
            for (std::size_t i = 0; i < count; ++i) {
                if (q[i] == -128) {
                    r.fail(weights_offset + i, name +
                                                   " (LinearInt8): expected weights in "
                                                   "-127..127 (symmetric), found -128");
                }
            }
            std::optional<Tensor> bias;
            if (has_bias == 1) {
                const std::size_t bias_offset = r.offset();
                bias = read_tensor_block(r, name + " bias");
                if (bias->shape() != Shape{out}) {
                    r.fail(bias_offset, name + " (LinearInt8): expected bias shape " +
                                            shape_to_string({out}) + ", found " +
                                            shape_to_string(bias->shape()));
                }
            }
            return std::make_unique<LinearInt8>(
                QuantizedMatrix::from_quantized(q.data(), scales.data(), in, out), std::move(bias));
        }
        default:
            r.fail(type_offset, name +
                                    ": expected a layer type id 1-5 "
                                    "(Linear, ReLU, Sigmoid, Softmax, LinearInt8), found " +
                                    std::to_string(type_id));
    }
}

// Formats 101770 as "101,770".
std::string with_commas(std::size_t n) {
    std::string digits = std::to_string(n);
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(digits.size()) - 3; i > 0; i -= 3) {
        digits.insert(static_cast<std::size_t>(i), ",");
    }
    return digits;
}

}  // namespace

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

Model Model::load(const std::string& path) {
    const std::vector<std::byte> bytes = read_file_bytes(path);
    BinaryReader r(bytes, path);

    r.expect_magic(kModelMagic);
    const std::size_t version_offset = r.offset();
    const std::uint32_t version = r.read_u32("version");
    if (version != kModelVersionFloat && version != kModelVersionInt8) {
        r.fail(version_offset,
               "expected model format version 1 or 2, found " + std::to_string(version));
    }

    Model model;
    model.path_ = path;
    model.metadata_ = read_metadata(r);
    model.input_features_ = numel_of(model.metadata_.input_shape);

    const std::size_t count_offset = r.offset();
    const std::uint32_t num_layers = r.read_u32("num_layers");
    if (num_layers == 0) r.fail(count_offset, "expected num_layers >= 1, found 0");

    // Validate the layer chain while reading. `features` is the width of the data flowing
    // between layers: it starts at the input size and only Linear changes it. Checking here
    // means a model that loads is a model that can run (see CLAUDE.md).
    std::size_t features = model.input_features_;
    for (std::uint32_t i = 0; i < num_layers; ++i) {
        const std::size_t layer_offset = r.offset();
        std::unique_ptr<Layer> layer = read_layer(r, i, version);

        if (const auto* linear = dynamic_cast<const DenseLayer*>(layer.get())) {
            if (linear->in_features() != features) {
                r.fail(layer_offset,
                       "layer " + std::to_string(i) + " (" + linear->name() +
                           "): expected in_features " + std::to_string(features) +
                           (i == 0 ? " (the input size " +
                                         shape_to_string(model.metadata_.input_shape) + ")"
                                   : " (the previous layer's output size)") +
                           ", found " + std::to_string(linear->in_features()));
            }
            features = linear->out_features();
        } else if (const auto* softmax = dynamic_cast<const Softmax*>(layer.get())) {
            // Layers see 2-D {batch, features} data, so only axes -2..1 exist.
            if (softmax->axis() < -2 || softmax->axis() > 1) {
                r.fail(layer_offset, "layer " + std::to_string(i) +
                                         " (Softmax): expected axis in -2..1 for 2-D "
                                         "{batch, features} data, found " +
                                         std::to_string(softmax->axis()));
            }
        }
        model.layers_.push_back(std::move(layer));
    }
    r.expect_end();
    model.build_plan();
    return model;
}

// Delegates to load() and then moves the loaded model into *this.
Model::Model(const std::string& path) : Model(load(path)) {}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

Tensor Model::as_batch(const Tensor& input, const char* fn) const {
    if (input.ndim() == 2 && input.size(1) == input_features_) return input;
    if (input.ndim() == 1 && input.size(0) == input_features_) {
        return input.reshape(Shape{1, input_features_});  // one sample -> a batch of one
    }
    throw std::invalid_argument(
        std::string(fn) + ": expected input of shape [N, " + std::to_string(input_features_) +
        "] or [" + std::to_string(input_features_) + "], got " + shape_to_string(input.shape()));
}

void Model::preprocess_into(const float* raw, std::size_t n, float* out) const {
    // The spec's formula, x = (raw * pixel_scale - mean) / std, in float32. One function for
    // both preprocess() and predict(), so they give identical results.
    const float scale = metadata_.pixel_scale;
    const float mean = metadata_.mean[0];
    const float stddev = metadata_.stddev[0];
    for (std::size_t i = 0; i < n; ++i) out[i] = (raw[i] * scale - mean) / stddev;
}

Tensor Model::preprocess(const Tensor& raw) const {
    const Tensor x = as_batch(raw, "Model::preprocess");
    Tensor out(x.shape());
    preprocess_into(x.data(), x.numel(), out.data());
    return out;
}

void Model::build_plan() {
    plan_.clear();
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const Layer* layer = layers_[i].get();
        const bool next_is_relu =
            i + 1 < layers_.size() && dynamic_cast<const ReLU*>(layers_[i + 1].get()) != nullptr;
        if (dynamic_cast<const DenseLayer*>(layer) && next_is_relu) {
            plan_.push_back({layer, true});
            ++i;  // the ReLU is part of this step
        } else {
            plan_.push_back({layer, false});
        }
    }
}

Tensor Model::forward(const Tensor& normalized) const {
    Tensor x = as_batch(normalized, "Model::forward");
    for (const auto& layer : layers_) x = layer->forward(x);
    return x;
}

Tensor Model::predict(const Tensor& raw) const {
    Workspace workspace;
    return predict(raw, workspace);  // copies the result out of the temporary workspace
}

const Tensor& Model::predict(const Tensor& raw, Workspace& workspace) const {
    return run_plan(raw, workspace, nullptr);
}

std::vector<Model::TraceStep> Model::predict_trace(const Tensor& raw, Workspace& workspace) const {
    std::vector<TraceStep> trace;
    trace.reserve(plan_.size() + 1);
    run_plan(raw, workspace, &trace);
    return trace;
}

const Tensor& Model::run_plan(const Tensor& raw, Workspace& workspace,
                              std::vector<TraceStep>* trace) const {
    // Shape check without building strings or temporaries (unless it fails).
    std::size_t rows = 0;
    if (raw.ndim() == 2 && raw.size(1) == input_features_) {
        rows = raw.size(0);
    } else if (raw.ndim() == 1 && raw.size(0) == input_features_) {
        rows = 1;  // a single sample is a batch of one; its data is already contiguous
    } else {
        (void)as_batch(raw, "Model::predict");  // throws with the usual message
    }

    // Two buffers, used alternately: each step reads one and writes the other (or works in
    // place). Their memory is reused on the next call.
    Tensor* current = &workspace.buffer(0);
    Tensor* other = &workspace.buffer(1);
    current->resize(rows, input_features_);
    preprocess_into(raw.data(), rows * input_features_, current->data());
    if (trace) trace->push_back({"preprocess", *current});

    for (const Step& step : plan_) {
        if (const auto* linear = dynamic_cast<const DenseLayer*>(step.layer)) {
            other->resize(rows, linear->out_features());
            linear->forward_into(current->data(), rows, other->data(), step.fuse_relu);
            std::swap(current, other);
        } else if (dynamic_cast<const ReLU*>(step.layer)) {
            relu_inplace(current->data(), current->numel());
        } else if (dynamic_cast<const Sigmoid*>(step.layer)) {
            sigmoid_inplace(current->data(), current->numel());
        } else if (const auto* softmax = dynamic_cast<const Softmax*>(step.layer);
                   softmax && (softmax->axis() == -1 || softmax->axis() == 1)) {
            softmax_rows_inplace(current->data(), rows, current->size(1));
        } else {
            // Anything without an in-place kernel (e.g. Softmax over axis 0): the layer's
            // own forward(), which allocates. Not used by the MNIST model.
            *other = step.layer->forward(*current);
            std::swap(current, other);
        }
        // Tracing copies each step's output (the in-place steps that follow would otherwise
        // overwrite it, e.g. Softmax turning the logits into probabilities). predict() passes
        // no trace, so its hot path copies and allocates nothing.
        if (trace) {
            trace->push_back(
                {step.fuse_relu ? step.layer->name() + " + ReLU" : step.layer->name(), *current});
        }
    }
    return *current;
}

std::vector<Tensor> Model::forward_trace(const Tensor& normalized) const {
    std::vector<Tensor> outputs;
    outputs.reserve(layers_.size());
    Tensor x = as_batch(normalized, "Model::forward_trace");
    for (const auto& layer : layers_) {
        x = layer->forward(x);
        outputs.push_back(x);
    }
    return outputs;
}

std::vector<Model::Prediction> Model::classify(const Tensor& raw) const {
    Workspace workspace;
    return classify(raw, workspace);
}

std::vector<Model::Prediction> Model::classify(const Tensor& raw, Workspace& workspace) const {
    const Tensor& output = predict(raw, workspace);  // {N, classes}
    const std::size_t classes = output.size(1);
    std::vector<Prediction> predictions;
    predictions.reserve(output.size(0));
    for (std::size_t n = 0; n < output.size(0); ++n) {
        // First maximum wins on ties, like argmax().
        const float* row = output.data() + n * classes;
        std::size_t best = 0;
        for (std::size_t j = 1; j < classes; ++j) {
            if (row[j] > row[best]) best = j;
        }
        predictions.push_back({best, row[best]});
    }
    return predictions;
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

std::size_t Model::num_parameters() const {
    std::size_t total = 0;
    for (const auto& layer : layers_) total += layer->num_parameters();
    return total;
}

std::string Model::summary() const {
    std::ostringstream os;
    os << "Model: " << path_ << " (format v" << format_version() << ")\n";
    os << "Input: " << shape_to_string(metadata_.input_shape) << " per sample, normalized as"
       << " (x * " << metadata_.pixel_scale << " - " << metadata_.mean[0] << ") / "
       << metadata_.stddev[0] << '\n';
    os << "Layers:\n";

    std::size_t features = input_features_;
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const Layer& layer = *layers_[i];
        const std::size_t in = features;
        if (const auto* linear = dynamic_cast<const DenseLayer*>(&layer)) {
            features = linear->out_features();
        }
        std::string shapes =
            "[N, " + std::to_string(in) + "] -> [N, " + std::to_string(features) + "]";
        os << "  " << i << "  " << layer.name();
        os << std::string(layer.name().size() < 24 ? 24 - layer.name().size() : 1, ' ');
        os << shapes;
        if (layer.num_parameters() > 0) {
            os << std::string(shapes.size() < 24 ? 24 - shapes.size() : 1, ' ')
               << with_commas(layer.num_parameters()) << " parameters";
        }
        os << '\n';
    }
    os << "Execution plan:";
    for (std::size_t i = 0; i < plan_.size(); ++i) {
        os << (i ? " -> " : " ") << plan_[i].layer->name();
        if (plan_[i].fuse_relu) os << " + ReLU (fused)";
    }
    os << '\n';
    os << "Total parameters: " << with_commas(num_parameters());
    return os.str();
}

// ---------------------------------------------------------------------------
// Saving and quantization
// ---------------------------------------------------------------------------

std::uint32_t Model::format_version() const {
    for (const auto& layer : layers_) {
        if (dynamic_cast<const LinearInt8*>(layer.get())) return kModelVersionInt8;
    }
    return kModelVersionFloat;
}

void Model::save(const std::string& path) const {
    BinaryWriter w;
    w.write_bytes(std::as_bytes(std::span(kModelMagic)));
    w.write_u32(format_version());

    w.write_u32(static_cast<std::uint32_t>(metadata_.input_shape.size()));
    for (std::size_t d : metadata_.input_shape) w.write_u64(d);
    w.write_f32(metadata_.pixel_scale);
    w.write_u32(static_cast<std::uint32_t>(metadata_.mean.size()));
    for (float m : metadata_.mean) w.write_f32(m);
    for (float sd : metadata_.stddev) w.write_f32(sd);

    w.write_u32(static_cast<std::uint32_t>(layers_.size()));
    for (const auto& layer : layers_) {
        if (const auto* linear = dynamic_cast<const Linear*>(layer.get())) {
            w.write_u32(kLayerLinear);
            w.write_u8(linear->bias() ? 1 : 0);
            write_tensor_block(w, linear->weight());
            if (linear->bias()) write_tensor_block(w, *linear->bias());
        } else if (const auto* q = dynamic_cast<const LinearInt8*>(layer.get())) {
            const QuantizedMatrix& m = q->weight();
            w.write_u32(kLayerLinearInt8);
            w.write_u8(q->bias() ? 1 : 0);
            w.write_u32(static_cast<std::uint32_t>(m.in_features()));
            w.write_u32(static_cast<std::uint32_t>(m.out_features()));
            w.write_f32_array({m.scales(), m.out_features()});
            // {out_features, in_features}: the unpadded part of each channel's row.
            for (std::size_t j = 0; j < m.out_features(); ++j) {
                w.write_bytes(
                    std::as_bytes(std::span(m.data() + j * m.padded_in(), m.in_features())));
            }
            if (q->bias()) write_tensor_block(w, *q->bias());
        } else if (dynamic_cast<const ReLU*>(layer.get())) {
            w.write_u32(kLayerReLU);
        } else if (dynamic_cast<const Sigmoid*>(layer.get())) {
            w.write_u32(kLayerSigmoid);
        } else if (const auto* softmax = dynamic_cast<const Softmax*>(layer.get())) {
            w.write_u32(kLayerSoftmax);
            w.write_i32(static_cast<std::int32_t>(softmax->axis()));
        } else {
            throw std::logic_error("Model::save: layer '" + layer->name() + "' has no file format");
        }
    }
    write_file_bytes(path, w.bytes());
}

Model Model::quantize() const {
    Model q;
    q.path_ = path_ + " (int8)";
    q.metadata_ = metadata_;
    q.input_features_ = input_features_;
    for (const auto& layer : layers_) {
        if (const auto* linear = dynamic_cast<const Linear*>(layer.get())) {
            q.layers_.push_back(std::make_unique<LinearInt8>(LinearInt8::from_linear(*linear)));
        } else if (const auto* already = dynamic_cast<const LinearInt8*>(layer.get())) {
            q.layers_.push_back(std::make_unique<LinearInt8>(already->weight(), already->bias()));
        } else if (dynamic_cast<const ReLU*>(layer.get())) {
            q.layers_.push_back(std::make_unique<ReLU>());
        } else if (dynamic_cast<const Sigmoid*>(layer.get())) {
            q.layers_.push_back(std::make_unique<Sigmoid>());
        } else if (const auto* softmax = dynamic_cast<const Softmax*>(layer.get())) {
            q.layers_.push_back(std::make_unique<Softmax>(softmax->axis()));
        } else {
            throw std::logic_error("Model::quantize: unsupported layer '" + layer->name() + "'");
        }
    }
    q.build_plan();
    return q;
}

}  // namespace inference
