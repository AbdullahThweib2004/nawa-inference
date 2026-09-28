// Web demo: the request -> response functions (no network), the fused-plan trace, and one
// real HTTP round trip through cpp-httplib on a free port.

#include <gtest/gtest.h>
#include <httplib.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "inference/layers/linear.hpp"
#include "inference/model/model.hpp"
#include "inference/model/tensor_io.hpp"
#include "inference/server/demo_service.hpp"
#include "inference/server/http_server.hpp"
#include "inference/tensor/ops.hpp"

using namespace inference;
using nlohmann::json;
using server::DemoService;

namespace {

const std::string kModel = std::string(NAWA_MODELS_DIR) + "/mnist_mlp.nawa";
const std::string kModelInt8 = std::string(NAWA_MODELS_DIR) + "/mnist_mlp_int8.nawa";
const std::string kFixtures = NAWA_FIXTURES_DIR;

// The first fixture image (a 7) as a /api/predict request body.
json fixture_request(const std::string& variant = "fp32", std::size_t index = 0) {
    const Tensor images = read_tensor_file(kFixtures + "/mnist_test100_images.ntsr");
    std::vector<int> pixels(784);
    for (std::size_t i = 0; i < 784; ++i) {
        pixels[i] = static_cast<int>(images.data()[index * 784 + i]);
    }
    return {{"width", 28}, {"height", 28}, {"pixels", pixels}, {"variant", variant}};
}

const DemoService& service() {
    // Braces, not parentheses: `DemoService s(Model(a), Model(b));` would declare a function
    // (C++'s "most vexing parse").
    static const DemoService s{Model(kModel), Model(kModelInt8)};
    return s;
}

json body_of(const server::ApiResponse& r) { return json::parse(r.body); }

bool bit_equal(const Tensor& a, const Tensor& b) {
    return a.shape() == b.shape() &&
           std::memcmp(a.data(), b.data(), a.numel() * sizeof(float)) == 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// predict_trace: the hidden layer stays visible even though Linear+ReLU are fused
// ---------------------------------------------------------------------------

TEST(PredictTrace, MatchesFixturesAndUnfusedTrace) {
    const Model model(kModel);
    ASSERT_TRUE(model.plan()[0].fuse_relu);
    const Tensor images = read_tensor_file(kFixtures + "/mnist_test100_images.ntsr");
    const Tensor first3({3, 784}, std::vector<float>(images.data(), images.data() + 3 * 784));
    Workspace ws;
    const auto trace = model.predict_trace(first3, ws);
    ASSERT_EQ(trace.size(), 4u);  // preprocess, Linear+ReLU (fused), Linear, Softmax
    EXPECT_EQ(trace[0].name, "preprocess");
    EXPECT_EQ(trace[1].name, "Linear(784 -> 128) + ReLU");

    // Against PyTorch's fixtures (same scaled tolerance as test_model.cpp).
    const char* fixtures[] = {"input_normalized", "relu", "linear2", "softmax"};
    for (std::size_t i = 0; i < 4; ++i) {
        const Tensor expected =
            read_tensor_file(kFixtures + "/mnist_test3_" + fixtures[i] + ".ntsr");
        float max_abs = 0.0f, max_ref = 0.0f;
        for (std::size_t k = 0; k < expected.numel(); ++k) {
            max_abs = std::max(max_abs, std::fabs(trace[i].output.data()[k] - expected.data()[k]));
            max_ref = std::max(max_ref, std::fabs(expected.data()[k]));
        }
        EXPECT_LT(max_abs / std::max(1.0f, max_ref), 1e-5f) << fixtures[i];
    }

    // Bit-identical to the unfused layer-by-layer trace, and to predict() itself.
    const auto unfused = model.forward_trace(trace[0].output);  // linear1, relu, linear2, softmax
    EXPECT_TRUE(bit_equal(trace[1].output, unfused[1]));
    EXPECT_TRUE(bit_equal(trace[2].output, unfused[2]));
    EXPECT_TRUE(bit_equal(trace[3].output, unfused[3]));
    EXPECT_TRUE(bit_equal(trace[3].output, model.predict(first3)));
}

TEST(PredictTrace, WorksForInt8Models) {
    const Model model(kModelInt8);
    const Tensor images = read_tensor_file(kFixtures + "/mnist_test100_images.ntsr");
    const Tensor one({784}, std::vector<float>(images.data(), images.data() + 784));
    Workspace ws;
    const auto trace = model.predict_trace(one, ws);
    ASSERT_EQ(trace.size(), 4u);
    EXPECT_EQ(trace[1].output.numel(), 128u);
    for (std::size_t i = 0; i < 128; ++i) EXPECT_GE(trace[1].output.data()[i], 0.0f);  // post-ReLU
    EXPECT_TRUE(bit_equal(trace[3].output, model.predict(one)));
}

// ---------------------------------------------------------------------------
// DemoService: request -> response, without a network
// ---------------------------------------------------------------------------

TEST(DemoService, PredictsAFixtureDigit) {
    for (const std::string variant : {"fp32", "int8"}) {
        const auto r = service().predict(fixture_request(variant).dump());
        ASSERT_EQ(r.status, 200) << r.body;
        const json b = body_of(r);
        EXPECT_EQ(b["variant"], variant);
        EXPECT_EQ(b["prediction"], 7);  // fixture 0 is a 7
        EXPECT_GT(b["confidence"].get<double>(), 0.9);
        EXPECT_EQ(b["input28"].size(), 784u);
        EXPECT_EQ(b["hidden"].size(), 128u);
        EXPECT_EQ(b["logits"].size(), 10u);
        EXPECT_EQ(b["probabilities"].size(), 10u);
        EXPECT_EQ(b["top3"].size(), 3u);
        EXPECT_EQ(b["top3"][0]["digit"], 7);
        EXPECT_EQ(b["contributions"].size(), 20u);
        int active = 0;
        for (const auto& h : b["hidden"]) active += h.get<double>() > 0.0;
        EXPECT_EQ(b["active_neurons"], active);
        // Contributions are activation x weight, sorted by magnitude.
        double previous = 1e30;
        for (const auto& c : b["contributions"]) {
            EXPECT_NEAR(c["contribution"].get<double>(),
                        c["activation"].get<double>() * c["weight"].get<double>(), 1e-5);
            EXPECT_LE(std::fabs(c["contribution"].get<double>()), previous + 1e-9);
            previous = std::fabs(c["contribution"].get<double>());
        }
        for (const char* t : {"preprocess", "forward", "trace", "total"}) {
            EXPECT_GE(b["timing_us"][t].get<double>(), 0.0) << t;
        }
    }
}

TEST(DemoService, MatchesTheModelDirectly) {
    // Same pipeline as `nawa predict`: preprocessing + model. The probabilities must equal
    // running Model::predict on the preprocessed input the service reports.
    const json b = body_of(service().predict(fixture_request().dump()));
    std::vector<float> input(784);
    for (std::size_t i = 0; i < 784; ++i) input[i] = b["input28"][i].get<float>();
    const Model model(kModel);
    const Tensor probs = model.predict(Tensor({784}, input));
    for (std::size_t j = 0; j < 10; ++j) {
        EXPECT_NEAR(b["probabilities"][j].get<double>(), probs.data()[j], 1e-4);
    }
}

TEST(DemoService, RejectsInvalidRequests) {
    const auto status = [](const json& body) { return service().predict(body.dump()).status; };
    EXPECT_EQ(service().predict("not json").status, 400);
    EXPECT_EQ(service().predict("[1, 2, 3]").status, 400);
    json r = fixture_request();
    r.erase("width");
    EXPECT_EQ(status(r), 400);
    r = fixture_request();
    r["width"] = 0;
    EXPECT_EQ(status(r), 400);
    r["width"] = 5000;  // above the limit
    EXPECT_EQ(status(r), 400);
    r = fixture_request();
    r["width"] = 27;  // 27 x 28 != 784 pixels
    EXPECT_EQ(status(r), 400);
    r = fixture_request();
    r["pixels"][10] = 256;
    EXPECT_EQ(status(r), 400);
    r["pixels"][10] = -1;
    EXPECT_EQ(status(r), 400);
    r["pixels"][10] = "white";
    EXPECT_EQ(status(r), 400);
    r = fixture_request("fp16");
    EXPECT_EQ(status(r), 400);
    const auto err = body_of(service().predict(r.dump()));
    EXPECT_NE(err["error"].get<std::string>().find("fp16"), std::string::npos);
}

TEST(DemoService, EmptyCanvasIs422) {
    const json empty = {
        {"width", 280}, {"height", 280}, {"pixels", std::vector<int>(280 * 280, 255)}};
    const auto r = service().predict(empty.dump());  // all white: nothing drawn
    EXPECT_EQ(r.status, 422);
    EXPECT_NE(body_of(r)["error"].get<std::string>().find("empty"), std::string::npos);
}

TEST(DemoService, Int8NotLoadedIs409) {
    const DemoService fp32_only{Model(kModel)};
    EXPECT_EQ(fp32_only.predict(fixture_request("int8").dump()).status, 409);
    EXPECT_EQ(fp32_only.neuron("3", "int8").status, 409);
    const json info = body_of(fp32_only.model_info());
    EXPECT_EQ(info["variants"], json::array({"fp32"}));
}

TEST(DemoService, ModelInfo) {
    const auto r = service().model_info();
    ASSERT_EQ(r.status, 200);
    const json b = body_of(r);
    EXPECT_EQ(b["variants"], json::array({"fp32", "int8"}));
    EXPECT_EQ(b["hidden_size"], 128);
    EXPECT_EQ(b["classes"], 10);
    EXPECT_EQ(b["models"]["fp32"]["parameters"], 101770);
    EXPECT_EQ(b["models"]["int8"]["format_version"], 2);
    EXPECT_EQ(b["models"]["fp32"]["plan"][0], "Linear(784 -> 128) + ReLU");
    EXPECT_TRUE(b.contains("kernel"));
    EXPECT_GE(b["threads"].get<int>(), 1);
}

TEST(DemoService, NeuronEndpoint) {
    const auto r = service().neuron("5");
    ASSERT_EQ(r.status, 200);
    const json b = body_of(r);
    EXPECT_EQ(b["weights"].size(), 784u);
    EXPECT_EQ(b["outgoing"].size(), 10u);
    const Model model(kModel);
    const auto& layer = dynamic_cast<const Linear&>(*model.layers()[0]);
    EXPECT_FLOAT_EQ(b["weights"][100].get<float>(), layer.weight().at({100, 5}));
    EXPECT_FLOAT_EQ(b["bias"].get<float>(), layer.bias()->at({5}));

    EXPECT_EQ(service().neuron("127").status, 200);
    EXPECT_EQ(service().neuron("128").status, 404);  // out of range
    EXPECT_EQ(service().neuron("99999999").status, 404);
    EXPECT_EQ(service().neuron("-1").status, 400);  // not a number
    EXPECT_EQ(service().neuron("abc").status, 400);
    EXPECT_EQ(service().neuron("").status, 400);
    EXPECT_EQ(service().neuron("5", "int8").status, 200);
    EXPECT_EQ(service().neuron("5", "bf16").status, 400);
}

TEST(DemoService, ConcurrentRequestsShareTheModel) {
    // Many threads, one DemoService: each request borrows its own Workspace. Run under TSan
    // (ENABLE_TSAN) this also checks for data races.
    const std::string body = fixture_request().dump();
    const json expected = body_of(service().predict(body))["probabilities"];
    std::vector<std::thread> clients;
    std::atomic<int> wrong{0};
    for (int t = 0; t < 6; ++t) {
        clients.emplace_back([&, t] {
            for (int i = 0; i < 10; ++i) {
                const auto r = service().predict(t % 2 ? body : fixture_request("int8").dump());
                if (r.status != 200) ++wrong;
                if (t % 2 && body_of(r)["probabilities"] != expected) ++wrong;
            }
        });
    }
    for (auto& c : clients) c.join();
    EXPECT_EQ(wrong.load(), 0);
}

// ---------------------------------------------------------------------------
// HTTP: one real server on a free port, in this process
// ---------------------------------------------------------------------------

TEST(HttpServer, RoundTripOverHttp) {
    server::ServerOptions options;
    options.port = 0;  // any free port
    options.web_dir = std::string(NAWA_SOURCE_DIR) + "/web";
    server::HttpServer http(service(), options);
    const int port = http.bind();
    std::thread serving([&] { http.run(); });

    httplib::Client client("127.0.0.1", port);
    const auto info = client.Get("/api/model");
    ASSERT_TRUE(info);
    EXPECT_EQ(info->status, 200);
    EXPECT_EQ(json::parse(info->body)["hidden_size"], 128);

    const auto predict = client.Post("/api/predict", fixture_request().dump(), "application/json");
    ASSERT_TRUE(predict);
    EXPECT_EQ(predict->status, 200);
    EXPECT_EQ(json::parse(predict->body)["prediction"], 7);

    const auto bad = client.Post("/api/predict", "{}", "application/json");
    ASSERT_TRUE(bad);
    EXPECT_EQ(bad->status, 400);
    EXPECT_TRUE(json::parse(bad->body).contains("error"));

    const auto neuron = client.Get("/api/neuron/200");
    ASSERT_TRUE(neuron);
    EXPECT_EQ(neuron->status, 404);

    const auto missing = client.Get("/api/unknown");
    ASSERT_TRUE(missing);
    EXPECT_EQ(missing->status, 404);
    EXPECT_TRUE(json::parse(missing->body).contains("error"));

    const auto page = client.Get("/");
    ASSERT_TRUE(page);
    EXPECT_EQ(page->status, 200);
    EXPECT_NE(page->body.find("<title>Nawa"), std::string::npos);
    EXPECT_EQ(page->get_header_value("X-Content-Type-Options"), "nosniff");

    http.stop();
    serving.join();
}
