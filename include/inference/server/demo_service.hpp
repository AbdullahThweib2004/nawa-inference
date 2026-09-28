#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "inference/model/model.hpp"
#include "inference/runtime/workspace.hpp"

// The web demo's request -> response logic, with no networking in it: every endpoint is a
// plain function from request text to {status, JSON body}. The HTTP layer (http_server.hpp)
// only routes requests to these functions, so tests can call them directly.
namespace inference::server {

struct ApiResponse {
    int status = 200;  // HTTP status code
    std::string body;  // JSON text
};

// Holds the loaded models and answers API requests. Safe to call from many threads at once:
//  - The Models are immutable after loading and are SHARED by all requests (they are large:
//    the weights, and their packed/quantized forms).
//  - Everything a request writes goes into a Workspace (see runtime/workspace.hpp), and a
//    Workspace is NOT thread-safe. Each request therefore borrows its own Workspace from a
//    small pool and returns it when done. The pool only grows to the number of requests that
//    were ever in flight at the same time, and reused workspaces keep their memory, so
//    steady-state requests don't allocate model buffers.
class DemoService {
public:
    // The demo explains an MLP classifier, so both models must be
    // Linear(+ReLU) -> Linear -> Softmax with the same shapes (e.g. models/mnist_mlp.nawa and
    // its `nawa quantize` output). Throws std::invalid_argument otherwise.
    explicit DemoService(Model fp32, std::optional<Model> int8 = std::nullopt);

    // GET /api/model
    ApiResponse model_info() const;

    // POST /api/predict with body {"width", "height", "pixels": [0..255 ...], "variant"}.
    ApiResponse predict(const std::string& request_body) const;

    // GET /api/neuron/<id>?variant=fp32|int8
    ApiResponse neuron(const std::string& id_text, const std::string& variant = "fp32") const;

    // Limits (also enforced by the HTTP layer where it can).
    static constexpr int kMaxSide = 1024;                   // width and height at most this
    static constexpr std::size_t kMaxBodyBytes = 8u << 20;  // 8 MiB request body

private:
    const Model& model_for(const std::string& variant) const;  // throws if unavailable

    // Borrowing a Workspace for one request (see the class comment).
    std::unique_ptr<Workspace> acquire_workspace() const;
    void release_workspace(std::unique_ptr<Workspace> workspace) const;

    Model fp32_;
    std::optional<Model> int8_;
    mutable std::mutex pool_mutex_;
    mutable std::vector<std::unique_ptr<Workspace>> pool_;
};

}  // namespace inference::server
