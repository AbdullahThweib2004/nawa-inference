// nawa: command-line front end for the inference engine.
//
//   nawa info    <model.nawa>
//   nawa eval    <model.nawa> --mnist <dir> [--batch N]
//   nawa predict <model.nawa> <image> [--no-preprocess] [--show] [--json]

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "inference/data/digit_preprocess.hpp"
#include "inference/data/idx.hpp"
#include "inference/data/image_io.hpp"
#include "inference/model/model.hpp"
#include "inference/server/demo_service.hpp"
#include "inference/server/http_server.hpp"

using namespace inference;

namespace {

constexpr std::size_t kNumClasses = 10;
constexpr std::size_t kMnistSide = 28;

// Thrown for bad command-line usage; main() prints the usage text and exits with 2.
struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void print_usage(std::ostream& os) {
    os << "Usage:\n"
          "  nawa info    <model.nawa>\n"
          "      Print the model's layers, shapes and parameter count.\n"
          "  nawa eval    <model.nawa> --mnist <dir> [--batch N] [--save-predictions FILE]\n"
          "      Evaluate on the MNIST test set (t10k-images-idx3-ubyte and\n"
          "      t10k-labels-idx1-ubyte in <dir>, e.g. data/MNIST/raw). Default batch: 256.\n"
          "      --save-predictions writes 'index true predicted confidence' per image, so two\n"
          "      builds can be compared with diff.\n"
          "  nawa quantize <in.nawa> <out.nawa>\n"
          "      Convert a float32 model to INT8 weights (format version 2, ~4x smaller).\n"
          "      eval and predict accept the result like any model.\n"
          "  nawa serve <model.nawa> [--int8 <model.nawa>] [--port 8080] [--host 127.0.0.1]\n"
          "             [--web <dir>]\n"
          "      Web demo: draw a digit in the browser, the C++ engine classifies it.\n"
          "      Serves web/ and a JSON API; --port 0 picks a free port.\n"
          "  nawa predict <model.nawa> <image> [--no-preprocess] [--show] [--json]\n"
          "      Classify a digit image (PNG, JPEG, BMP, ...). --show prints the 28x28 input\n"
          "      as ASCII art; --no-preprocess only resizes to 28x28 (for comparison);\n"
          "      --json prints the prediction, all 10 probabilities and the 28x28 input.\n";
}

// Splits arguments into positional values and --flags. `value_flags` take a value.
struct Args {
    std::vector<std::string> positional;
    std::vector<std::string> switches;                        // e.g. --show
    std::vector<std::pair<std::string, std::string>> values;  // e.g. --batch 256

    bool has(const std::string& name) const {
        return std::find(switches.begin(), switches.end(), name) != switches.end();
    }
    std::optional<std::string> value(const std::string& name) const {
        for (const auto& [k, v] : values) {
            if (k == name) return v;
        }
        return std::nullopt;
    }
};

Args parse_args(int argc, char** argv, int first, const std::vector<std::string>& switch_flags,
                const std::vector<std::string>& value_flags) {
    Args args;
    for (int i = first; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) {
            args.positional.push_back(arg);
        } else if (std::find(switch_flags.begin(), switch_flags.end(), arg) != switch_flags.end()) {
            args.switches.push_back(arg);
        } else if (std::find(value_flags.begin(), value_flags.end(), arg) != value_flags.end()) {
            if (i + 1 >= argc) throw UsageError(arg + " needs a value");
            args.values.emplace_back(arg, argv[++i]);
        } else {
            throw UsageError("unknown option " + arg);
        }
    }
    return args;
}

std::string with_commas(std::size_t n) {
    std::string s = std::to_string(n);
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(s.size()) - 3; i > 0; i -= 3) {
        s.insert(static_cast<std::size_t>(i), ",");
    }
    return s;
}

std::string percent(double fraction, int decimals = 2) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(decimals) << fraction * 100.0 << '%';
    return os.str();
}

// Draws a square image as ASCII art, two characters per pixel so it isn't squashed.
void print_ascii(const GrayImage& image) {
    static constexpr std::string_view kShades = " .:-=+*#%@";  // dark -> bright
    std::cout << '+' << std::string(image.width * 2, '-') << "+\n";
    for (std::size_t y = 0; y < image.height; ++y) {
        std::cout << '|';
        for (std::size_t x = 0; x < image.width; ++x) {
            const float v = std::clamp(image.at(x, y), 0.0f, 255.0f);
            const auto level = static_cast<std::size_t>(v / 255.0f * (kShades.size() - 1) + 0.5f);
            std::cout << kShades[level] << kShades[level];
        }
        std::cout << "|\n";
    }
    std::cout << '+' << std::string(image.width * 2, '-') << "+\n";
}

// ---------------------------------------------------------------------------
// nawa info
// ---------------------------------------------------------------------------

int cmd_info(int argc, char** argv) {
    const Args args = parse_args(argc, argv, 2, {}, {});
    if (args.positional.size() != 1) throw UsageError("info takes exactly one model path");
    const Model model(args.positional[0]);
    std::cout << model.summary() << '\n';
    return 0;
}

// ---------------------------------------------------------------------------
// nawa eval
// ---------------------------------------------------------------------------

struct Mistake {
    std::size_t index;
    std::size_t truth;
    std::size_t predicted;
    float confidence;
};

int cmd_eval(int argc, char** argv) {
    using Clock = std::chrono::steady_clock;
    const Args args = parse_args(argc, argv, 2, {}, {"--mnist", "--batch", "--save-predictions"});
    if (args.positional.size() != 1) throw UsageError("eval takes exactly one model path");
    const auto dir = args.value("--mnist");
    if (!dir) throw UsageError("eval needs --mnist <dir>");
    std::size_t batch_size = 256;
    if (const auto b = args.value("--batch")) {
        try {
            batch_size = std::stoul(*b);
        } catch (const std::exception&) {
            batch_size = 0;
        }
        if (batch_size == 0) throw UsageError("--batch must be a positive integer");
    }

    const auto t_start = Clock::now();
    const Model model(args.positional[0]);
    const std::filesystem::path root(*dir);
    const Tensor images = load_mnist_images((root / "t10k-images-idx3-ubyte").string());
    const std::vector<std::uint8_t> labels =
        load_mnist_labels((root / "t10k-labels-idx1-ubyte").string());
    const auto t_loaded = Clock::now();

    const std::size_t n = images.size(0);
    const std::size_t features = images.size(1);
    if (labels.size() != n) {
        throw std::runtime_error("image and label files disagree: " + std::to_string(n) +
                                 " images, " + std::to_string(labels.size()) + " labels");
    }
    if (features != model.input_features()) {
        throw std::runtime_error("images have " + std::to_string(features) +
                                 " pixels, but the model expects " +
                                 std::to_string(model.input_features()));
    }

    std::cout << "Evaluating " << args.positional[0] << " on " << with_commas(n)
              << " MNIST test images (batch " << batch_size << ")\n\n";

    std::array<std::array<std::size_t, kNumClasses>, kNumClasses> confusion{};
    std::vector<Model::Prediction> all_predictions;
    all_predictions.reserve(n);
    std::vector<Mistake> mistakes;
    std::size_t correct = 0;

    Workspace workspace;  // reused by every batch: no allocations after the first one
    for (std::size_t start = 0; start < n; start += batch_size) {
        const std::size_t rows = std::min(batch_size, n - start);
        const float* first = images.data() + start * features;
        const Tensor batch({rows, features}, std::vector<float>(first, first + rows * features));
        const auto predictions = model.classify(batch, workspace);
        for (std::size_t r = 0; r < rows; ++r) {
            const std::size_t truth = labels[start + r];
            const auto& p = predictions[r];
            all_predictions.push_back(p);
            if (p.label >= kNumClasses) throw std::runtime_error("model predicted a class >= 10");
            ++confusion[truth][p.label];
            if (p.label == truth) {
                ++correct;
            } else {
                mistakes.push_back({start + r, truth, p.label, p.confidence});
            }
        }
    }
    const auto t_done = Clock::now();

    if (const auto out_path = args.value("--save-predictions")) {
        std::ofstream out(*out_path);
        if (!out) throw std::runtime_error("cannot write " + *out_path);
        // Full float precision (9 significant digits round-trips a float exactly), so even
        // last-bit differences between two builds show up in a diff.
        out << std::setprecision(9);
        for (std::size_t i = 0; i < n; ++i) {
            out << i << ' ' << int{labels[i]} << ' ' << all_predictions[i].label << ' '
                << all_predictions[i].confidence << '\n';
        }
        std::cout << "Wrote " << n << " predictions to " << *out_path << "\n\n";
    }

    std::cout << "Accuracy: " << percent(static_cast<double>(correct) / static_cast<double>(n))
              << "  (" << with_commas(correct) << " / " << with_commas(n) << " correct)\n\n";

    std::cout << "Confusion matrix (rows = true digit, columns = predicted digit):\n      ";
    for (std::size_t c = 0; c < kNumClasses; ++c) std::cout << std::setw(6) << c;
    std::cout << '\n';
    for (std::size_t t = 0; t < kNumClasses; ++t) {
        std::cout << std::setw(6) << t;
        for (std::size_t c = 0; c < kNumClasses; ++c) std::cout << std::setw(6) << confusion[t][c];
        std::cout << '\n';
    }

    std::cout << "\nPer-digit accuracy:\n";
    for (std::size_t t = 0; t < kNumClasses; ++t) {
        const std::size_t total =
            std::accumulate(confusion[t].begin(), confusion[t].end(), std::size_t{0});
        const double acc =
            total ? static_cast<double>(confusion[t][t]) / static_cast<double>(total) : 0.0;
        std::cout << "  " << t << "  " << std::setw(7) << percent(acc) << "  (" << confusion[t][t]
                  << " / " << total << ")\n";
    }

    const std::size_t shown = std::min<std::size_t>(5, mistakes.size());
    std::partial_sort(
        mistakes.begin(), mistakes.begin() + static_cast<std::ptrdiff_t>(shown), mistakes.end(),
        [](const Mistake& a, const Mistake& b) { return a.confidence > b.confidence; });
    std::cout << "\nMost confident mistakes:\n  image  true  predicted  confidence\n";
    for (std::size_t i = 0; i < shown; ++i) {
        const Mistake& m = mistakes[i];
        std::cout << std::setw(7) << m.index << std::setw(6) << m.truth << std::setw(11)
                  << m.predicted << std::setw(12) << percent(m.confidence) << '\n';
    }

    const auto seconds = [](auto d) { return std::chrono::duration<double>(d).count(); };
    const double load_s = seconds(t_loaded - t_start);
    const double infer_s = seconds(t_done - t_loaded);
    std::cout << std::fixed << std::setprecision(3) << "\nTiming (std::chrono::steady_clock):\n"
              << "  load model + data  " << load_s << " s\n"
              << "  inference          " << infer_s << " s  ("
              << with_commas(static_cast<std::size_t>(static_cast<double>(n) / infer_s))
              << " images/s)\n"
              << "  total              " << load_s + infer_s << " s\n";
    return 0;
}

// ---------------------------------------------------------------------------
// nawa quantize
// ---------------------------------------------------------------------------

int cmd_quantize(int argc, char** argv) {
    const Args args = parse_args(argc, argv, 2, {}, {});
    if (args.positional.size() != 2) throw UsageError("quantize takes an input and an output path");
    const Model model(args.positional[0]);
    const Model quantized = model.quantize();
    quantized.save(args.positional[1]);
    const auto in_size = std::filesystem::file_size(args.positional[0]);
    const auto out_size = std::filesystem::file_size(args.positional[1]);
    std::cout << quantized.summary() << "\n\n"
              << "Wrote " << args.positional[1] << ": " << with_commas(out_size) << " bytes (was "
              << with_commas(in_size) << ", " << std::fixed << std::setprecision(2)
              << static_cast<double>(in_size) / static_cast<double>(out_size)
              << "x smaller), format version " << quantized.format_version() << '\n';
    return 0;
}

// ---------------------------------------------------------------------------
// nawa serve
// ---------------------------------------------------------------------------

// Set by SIGINT/SIGTERM. A signal handler may only do async-signal-safe work, so it just sets
// this flag; a watcher thread notices it and stops the server from normal code.
volatile std::sig_atomic_t g_stop_requested = 0;
extern "C" void on_stop_signal(int) { g_stop_requested = 1; }

int cmd_serve(int argc, char** argv) {
    const Args args = parse_args(argc, argv, 2, {}, {"--int8", "--port", "--host", "--web"});
    if (args.positional.size() != 1) throw UsageError("serve takes exactly one model path");
    inference::server::ServerOptions options;
    options.host = args.value("--host").value_or("127.0.0.1");
    options.web_dir = args.value("--web").value_or(std::string(NAWA_SOURCE_DIR) + "/web");
    if (const auto p = args.value("--port")) {
        try {
            options.port = std::stoi(*p);
        } catch (const std::exception&) {
            options.port = -1;
        }
        if (options.port < 0 || options.port > 65535) throw UsageError("--port must be 0..65535");
    }

    std::optional<Model> int8;
    if (const auto path = args.value("--int8")) int8.emplace(*path);
    const inference::server::DemoService service(Model(args.positional[0]), std::move(int8));
    inference::server::HttpServer http(service, options);
    const int port = http.bind();
    std::cout << "Nawa web demo: http://" << options.host << ":" << port << "/"
              << "   (Ctrl+C to stop)" << std::endl;  // flushed: scripts read the port from it

    std::signal(SIGINT, on_stop_signal);
    std::signal(SIGTERM, on_stop_signal);
    std::atomic<bool> finished{false};
    std::thread watcher([&] {
        while (!finished.load() && !g_stop_requested) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        http.stop();
    });
    http.run();  // blocks until stop()
    finished = true;
    watcher.join();
    std::cout << "Stopped." << std::endl;
    return 0;
}

// ---------------------------------------------------------------------------
// nawa predict
// ---------------------------------------------------------------------------

int cmd_predict(int argc, char** argv) {
    const Args args = parse_args(argc, argv, 2, {"--no-preprocess", "--show", "--json"}, {});
    // With --json the human-readable lines go nowhere and only one JSON object is printed.
    const bool as_json = args.has("--json");
    std::ostringstream discarded;
    std::ostream& out = as_json ? static_cast<std::ostream&>(discarded) : std::cout;
    if (args.positional.size() != 2) throw UsageError("predict takes a model path and an image");
    const Model model(args.positional[0]);
    if (model.input_features() != kMnistSide * kMnistSide) {
        throw std::runtime_error("predict expects a 28x28 (784-input) model");
    }

    const GrayImage original = load_grayscale_image(args.positional[1]);
    out << "Image: " << args.positional[1] << " (" << original.width << "x" << original.height
        << ")\n";

    GrayImage input;
    if (args.has("--no-preprocess")) {
        input = resize_only(original, kMnistSide);
        out << "Preprocessing: none (resized to 28x28 only)\n";
    } else {
        const PreprocessResult result = mnist_preprocess(original);
        input = result.image;
        out << "Preprocessing: " << (result.inverted ? "inverted (light background), " : "");
        if (result.box) {
            out << "cropped to " << result.box->width() << "x" << result.box->height() << " at ("
                << result.box->x0 << ", " << result.box->y0
                << "), scaled to fit 20x20, centered by mass in 28x28\n";
        } else {
            out << "no digit found (image is empty after thresholding)\n";
        }
    }
    if (args.has("--show") && !as_json) print_ascii(input);

    const Tensor probs = model.predict(to_model_input(input));  // {1, 10}
    std::vector<std::size_t> order(probs.size(1));
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return probs.at({0, a}) > probs.at({0, b}); });

    if (as_json) {
        nlohmann::json result;
        result["prediction"] = order[0];
        result["preprocessed"] = !args.has("--no-preprocess");
        result["probabilities"] = nlohmann::json::array();
        for (std::size_t c = 0; c < probs.size(1); ++c)
            result["probabilities"].push_back(probs.at({0, c}));
        result["input28"] = input.pixels;
        std::cout << result.dump() << '\n';
        return 0;
    }
    out << "Predicted digit: " << order[0] << '\n' << "Top 3:\n";
    for (std::size_t i = 0; i < 3 && i < order.size(); ++i) {
        out << "  " << order[i] << "  " << std::setw(7) << percent(probs.at({0, order[i]})) << '\n';
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage(argc < 2 ? std::cerr : std::cout);
        return argc < 2 ? 2 : 0;
    }
    const std::string command = argv[1];
    try {
        if (command == "info") return cmd_info(argc, argv);
        if (command == "eval") return cmd_eval(argc, argv);
        if (command == "predict") return cmd_predict(argc, argv);
        if (command == "quantize") return cmd_quantize(argc, argv);
        if (command == "serve") return cmd_serve(argc, argv);
        throw UsageError("unknown command '" + command + "'");
    } catch (const UsageError& e) {
        std::cerr << "nawa: " << e.what() << "\n\n";
        print_usage(std::cerr);
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "nawa: error: " << e.what() << '\n';
        return 1;
    }
}
