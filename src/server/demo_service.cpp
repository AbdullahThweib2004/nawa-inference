#include "inference/server/demo_service.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "inference/data/digit_preprocess.hpp"
#include "inference/layers/activations.hpp"
#include "inference/layers/linear.hpp"
#include "inference/layers/linear_int8.hpp"
#include "inference/runtime/thread_pool.hpp"
#include "inference/tensor/gemm.hpp"

namespace inference::server {

namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kTopContributions = 20;
constexpr std::size_t kImageSide = 28;

ApiResponse error(int status, const std::string& message) {
    return {status, json{{"error", message}}.dump()};
}

double micros(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::micro>(to - from).count();
}

// Weight (k, j) and bias j of a float32 or int8 fully connected layer, as float32 (int8
// weights are dequantized: q * scale). Used only for visualization, not for inference.
float dense_weight(const Layer& layer, std::size_t k, std::size_t j) {
    if (const auto* l = dynamic_cast<const Linear*>(&layer)) {
        return l->weight().data()[k * l->out_features() + j];
    }
    return dynamic_cast<const LinearInt8&>(layer).weight().dequantized(k, j);
}

float dense_bias(const Layer& layer, std::size_t j) {
    if (const auto* l = dynamic_cast<const Linear*>(&layer)) {
        return l->bias() ? l->bias()->data()[j] : 0.0f;
    }
    const auto& q = dynamic_cast<const LinearInt8&>(layer);
    return q.bias() ? q.bias()->data()[j] : 0.0f;
}

// The demo explains Linear(+ReLU) -> Linear -> Softmax(last axis) networks.
void check_structure(const Model& model, const std::string& what) {
    const auto& plan = model.plan();
    const auto* hidden =
        plan.size() == 3 ? dynamic_cast<const DenseLayer*>(plan[0].layer) : nullptr;
    const auto* output =
        plan.size() == 3 ? dynamic_cast<const DenseLayer*>(plan[1].layer) : nullptr;
    const auto* softmax = plan.size() == 3 ? dynamic_cast<const Softmax*>(plan[2].layer) : nullptr;
    if (!hidden || !plan[0].fuse_relu || !output || plan[1].fuse_relu || !softmax ||
        (softmax->axis() != -1 && softmax->axis() != 1) ||
        model.input_features() != kImageSide * kImageSide) {
        throw std::invalid_argument(what +
                                    " model: the web demo needs a 784-input network of "
                                    "the form Linear + ReLU -> Linear -> Softmax");
    }
}

const DenseLayer& hidden_layer(const Model& m) {
    return dynamic_cast<const DenseLayer&>(*m.plan()[0].layer);
}
const DenseLayer& output_layer(const Model& m) {
    return dynamic_cast<const DenseLayer&>(*m.plan()[1].layer);
}

json describe_model(const Model& model) {
    json layers = json::array();
    for (const auto& layer : model.layers()) {
        json entry{{"name", layer->name()}, {"parameters", layer->num_parameters()}};
        if (const auto* d = dynamic_cast<const DenseLayer*>(layer.get())) {
            entry["in"] = d->in_features();
            entry["out"] = d->out_features();
        }
        layers.push_back(entry);
    }
    json plan = json::array();
    for (const auto& step : model.plan()) {
        plan.push_back(step.fuse_relu ? step.layer->name() + " + ReLU" : step.layer->name());
    }
    return {{"format_version", model.format_version()},
            {"parameters", model.num_parameters()},
            {"layers", layers},
            {"plan", plan},
            {"summary", model.summary()}};
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction and the workspace pool
// ---------------------------------------------------------------------------

DemoService::DemoService(Model fp32, std::optional<Model> int8)
    : fp32_(std::move(fp32)), int8_(std::move(int8)) {
    check_structure(fp32_, "fp32");
    if (int8_) {
        check_structure(*int8_, "int8");
        if (hidden_layer(*int8_).out_features() != hidden_layer(fp32_).out_features() ||
            output_layer(*int8_).out_features() != output_layer(fp32_).out_features()) {
            throw std::invalid_argument("the int8 model's shapes don't match the fp32 model's");
        }
    }
}

std::unique_ptr<Workspace> DemoService::acquire_workspace() const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (pool_.empty()) return std::make_unique<Workspace>();  // more requests in flight than ever
    std::unique_ptr<Workspace> w = std::move(pool_.back());
    pool_.pop_back();
    return w;
}

void DemoService::release_workspace(std::unique_ptr<Workspace> workspace) const {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    pool_.push_back(std::move(workspace));
}

const Model& DemoService::model_for(const std::string& variant) const {
    if (variant == "fp32") return fp32_;
    if (variant == "int8" && int8_) return *int8_;
    throw std::out_of_range(variant);  // callers turn this into a JSON error
}

// ---------------------------------------------------------------------------
// GET /api/model
// ---------------------------------------------------------------------------

ApiResponse DemoService::model_info() const {
    const Model::Metadata& meta = fp32_.metadata();
    json variants = json::array({"fp32"});
    if (int8_) variants.push_back("int8");
    json models{{"fp32", describe_model(fp32_)}};
    if (int8_) models["int8"] = describe_model(*int8_);
    const json body{{"variants", variants},
                    {"models", models},
                    {"input",
                     {{"shape", meta.input_shape},
                      {"image", {kImageSide, kImageSide}},
                      {"pixel_scale", meta.pixel_scale},
                      {"mean", meta.mean},
                      {"std", meta.stddev}}},
                    {"hidden_size", hidden_layer(fp32_).out_features()},
                    {"classes", output_layer(fp32_).out_features()},
                    {"kernel", kernel_name(resolve_kernel())},
                    {"threads", default_thread_pool().num_threads()},
                    {"threads_note", describe_thread_choice()},
                    {"limits", {{"max_side", kMaxSide}, {"max_body_bytes", kMaxBodyBytes}}}};
    return {200, body.dump()};
}

// ---------------------------------------------------------------------------
// POST /api/predict
// ---------------------------------------------------------------------------

ApiResponse DemoService::predict(const std::string& request_body) const {
    const auto t_start = Clock::now();
    if (request_body.size() > kMaxBodyBytes) return error(413, "request body too large");

    // --- Parse and validate. Every failure is a 4xx with a message the UI can show. ---
    const json request = json::parse(request_body, nullptr, /*allow_exceptions=*/false);
    if (request.is_discarded() || !request.is_object()) {
        return error(400, "the request body must be a JSON object");
    }
    const auto int_field = [&](const char* name) -> std::optional<long long> {
        const auto it = request.find(name);
        if (it == request.end() || !it->is_number_integer()) return std::nullopt;
        return it->get<long long>();
    };
    const auto width = int_field("width");
    const auto height = int_field("height");
    if (!width || !height || *width < 1 || *height < 1 || *width > kMaxSide || *height > kMaxSide) {
        return error(400, "\"width\" and \"height\" must be integers between 1 and " +
                              std::to_string(kMaxSide));
    }
    const auto pixels_it = request.find("pixels");
    const std::size_t expected =
        static_cast<std::size_t>(*width) * static_cast<std::size_t>(*height);
    if (pixels_it == request.end() || !pixels_it->is_array()) {
        return error(400, "\"pixels\" must be an array of grayscale values 0..255");
    }
    if (pixels_it->size() != expected) {
        return error(400, "\"pixels\" has " + std::to_string(pixels_it->size()) +
                              " values, but width x height = " + std::to_string(expected));
    }
    std::string variant = "fp32";
    if (const auto v = request.find("variant"); v != request.end()) {
        if (!v->is_string()) return error(400, "\"variant\" must be \"fp32\" or \"int8\"");
        variant = v->get<std::string>();
    }
    if (variant != "fp32" && variant != "int8") {
        return error(400, "unknown variant \"" + variant + "\": use \"fp32\" or \"int8\"");
    }
    if (variant == "int8" && !int8_) {
        return error(409, "the int8 model is not loaded (start the server with --int8)");
    }

    GrayImage image;
    image.width = static_cast<std::size_t>(*width);
    image.height = static_cast<std::size_t>(*height);
    image.pixels.resize(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        const json& p = (*pixels_it)[i];
        if (!p.is_number()) return error(400, "pixel " + std::to_string(i) + " is not a number");
        const double v = p.get<double>();
        if (!(v >= 0.0 && v <= 255.0)) {  // also rejects NaN
            return error(400, "pixel " + std::to_string(i) + " is outside 0..255");
        }
        image.pixels[i] = static_cast<float>(v);
    }

    // --- Preprocess (the same C++ pipeline as `nawa predict`), then run the model. ---
    const Model& model = model_for(variant);
    const auto t_pre = Clock::now();
    const PreprocessResult prepared = mnist_preprocess(image);
    const auto t_pre_done = Clock::now();
    if (!prepared.box) {
        return error(422, "the canvas is empty: draw a digit first");
    }
    const Tensor input = to_model_input(prepared.image);  // {784}, raw 0..255

    std::unique_ptr<Workspace> workspace = acquire_workspace();
    // Timed: the real, allocation-free prediction path.
    const auto t_fwd = Clock::now();
    const Tensor& result = model.predict(input, *workspace);
    const auto t_fwd_done = Clock::now();
    // Copy it out before the workspace is reused by predict_trace below.
    const Tensor probs = result;
    // Not part of the prediction: the same plan again, recording every step for the UI.
    const std::vector<Model::TraceStep> trace = model.predict_trace(input, *workspace);
    const auto t_trace_done = Clock::now();
    release_workspace(std::move(workspace));

    // trace: [0] normalized input, [1] hidden after ReLU, [2] logits, [3] probabilities.
    const Tensor& hidden = trace[1].output;
    const Tensor& logits = trace[2].output;
    const std::size_t H = hidden.numel();
    const std::size_t C = probs.numel();

    std::vector<std::size_t> order(C);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return probs.data()[a] > probs.data()[b];
    });
    const std::size_t winner = order[0];

    // How much each hidden neuron pushed the winning logit: activation x weight.
    const Layer& out_layer = *model.plan()[1].layer;
    std::vector<std::size_t> neurons(H);
    std::iota(neurons.begin(), neurons.end(), std::size_t{0});
    const auto contribution = [&](std::size_t i) {
        return hidden.data()[i] * dense_weight(out_layer, i, winner);
    };
    std::stable_sort(neurons.begin(), neurons.end(), [&](std::size_t a, std::size_t b) {
        return std::fabs(contribution(a)) > std::fabs(contribution(b));
    });
    json contributions = json::array();
    for (std::size_t n = 0; n < std::min(kTopContributions, H); ++n) {
        const std::size_t i = neurons[n];
        contributions.push_back({{"neuron", i},
                                 {"activation", hidden.data()[i]},
                                 {"weight", dense_weight(out_layer, i, winner)},
                                 {"contribution", contribution(i)}});
    }

    json top3 = json::array();
    for (std::size_t n = 0; n < std::min<std::size_t>(3, C); ++n) {
        top3.push_back({{"digit", order[n]}, {"probability", probs.data()[order[n]]}});
    }
    std::vector<int> input28(prepared.image.pixels.size());
    for (std::size_t i = 0; i < input28.size(); ++i) {
        input28[i] = static_cast<int>(std::lround(prepared.image.pixels[i]));
    }
    const std::size_t active = static_cast<std::size_t>(
        std::count_if(hidden.data(), hidden.data() + H, [](float h) { return h > 0.0f; }));
    const auto as_vector = [](const Tensor& t) {
        return std::vector<float>(t.data(), t.data() + t.numel());
    };

    const auto t_end = Clock::now();
    const json response{{"variant", variant},
                        {"input28", input28},
                        {"preprocess",
                         {{"inverted", prepared.inverted},
                          {"box",
                           {{"x", prepared.box->x0},
                            {"y", prepared.box->y0},
                            {"width", prepared.box->width()},
                            {"height", prepared.box->height()}}}}},
                        {"hidden", as_vector(hidden)},
                        {"logits", as_vector(logits)},
                        {"probabilities", as_vector(probs)},
                        {"prediction", winner},
                        {"confidence", probs.data()[winner]},
                        {"top3", top3},
                        {"active_neurons", active},
                        {"hidden_size", H},
                        {"contributions", contributions},
                        {"timing_us",
                         {{"preprocess", micros(t_pre, t_pre_done)},
                          {"forward", micros(t_fwd, t_fwd_done)},
                          {"trace", micros(t_fwd_done, t_trace_done)},
                          {"total", micros(t_start, t_end)}}}};
    return {200, response.dump()};
}

// ---------------------------------------------------------------------------
// GET /api/neuron/<id>
// ---------------------------------------------------------------------------

ApiResponse DemoService::neuron(const std::string& id_text, const std::string& variant) const {
    if (id_text.empty() || id_text.size() > 9 ||
        !std::all_of(id_text.begin(), id_text.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        return error(400, "neuron id must be a non-negative integer");
    }
    if (variant != "fp32" && variant != "int8") {
        return error(400, "unknown variant \"" + variant + "\": use \"fp32\" or \"int8\"");
    }
    if (variant == "int8" && !int8_) return error(409, "the int8 model is not loaded");
    const Model& model = model_for(variant);
    const DenseLayer& hidden = hidden_layer(model);
    const DenseLayer& output = output_layer(model);
    const std::size_t id = std::stoul(id_text);
    if (id >= hidden.out_features()) {
        return error(404, "no hidden neuron " + id_text + ": ids are 0.." +
                              std::to_string(hidden.out_features() - 1));
    }
    std::vector<float> weights(hidden.in_features());
    for (std::size_t k = 0; k < weights.size(); ++k) weights[k] = dense_weight(hidden, k, id);
    std::vector<float> outgoing(output.out_features());
    for (std::size_t j = 0; j < outgoing.size(); ++j) outgoing[j] = dense_weight(output, id, j);
    const auto [lo, hi] = std::minmax_element(weights.begin(), weights.end());
    const json body{{"id", id},
                    {"variant", variant},
                    {"shape", {kImageSide, kImageSide}},
                    {"weights", weights},
                    {"bias", dense_bias(hidden, id)},
                    {"outgoing", outgoing},
                    {"min", *lo},
                    {"max", *hi}};
    return {200, body.dump()};
}

}  // namespace inference::server
