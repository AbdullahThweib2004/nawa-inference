#include "inference/server/http_server.hpp"

#include <httplib.h>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace inference::server {

struct HttpServer::Impl {
    Impl(const DemoService& s, ServerOptions o) : service(s), options(std::move(o)) {}

    const DemoService& service;
    ServerOptions options;
    httplib::Server server;
};

namespace {

void reply(httplib::Response& res, const ApiResponse& api) {
    res.status = api.status;
    res.set_content(api.body, "application/json");
}

}  // namespace

HttpServer::HttpServer(const DemoService& service, ServerOptions options)
    : impl_(std::make_unique<Impl>(service, std::move(options))) {
    httplib::Server& svr = impl_->server;
    const std::string& web_dir = impl_->options.web_dir;
    if (!std::filesystem::is_directory(web_dir)) {
        throw std::runtime_error("web directory not found: " + web_dir);
    }

    // Limits: requests can't be arbitrarily large or slow.
    svr.set_payload_max_length(DemoService::kMaxBodyBytes);
    svr.set_read_timeout(5, 0);
    svr.set_write_timeout(5, 0);
    svr.set_keep_alive_max_count(100);

    // Security headers on every response. The page loads only its own files (no CDN, no
    // inline scripts); data: images are allowed for the CSS pen cursor.
    svr.set_default_headers({
        {"X-Content-Type-Options", "nosniff"},
        {"Referrer-Policy", "no-referrer"},
        {"Content-Security-Policy",
         "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
         "connect-src 'self'; frame-ancestors 'none'"},
    });

    // Static frontend: web/index.html, style.css, app.js at /. (cpp-httplib rejects paths
    // that try to leave the directory, e.g. /../CMakeLists.txt.)
    if (!svr.set_mount_point("/", web_dir)) {
        throw std::runtime_error("cannot serve web directory: " + web_dir);
    }

    // API routes: thin wrappers around DemoService.
    const DemoService& s = impl_->service;
    svr.Get("/api/model",
            [&s](const httplib::Request&, httplib::Response& res) { reply(res, s.model_info()); });
    svr.Post("/api/predict", [&s](const httplib::Request& req, httplib::Response& res) {
        reply(res, s.predict(req.body));
    });
    svr.Get(R"(/api/neuron/([^/]+))", [&s](const httplib::Request& req, httplib::Response& res) {
        const std::string variant =
            req.has_param("variant") ? req.get_param_value("variant") : std::string("fp32");
        reply(res, s.neuron(req.matches[1], variant));
    });

    // Errors that didn't come from our handlers (unknown route, body too large, ...) also
    // get a JSON body. Responses that already have one (our own 4xx) are left alone.
    svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        if (!res.body.empty()) return httplib::Server::HandlerResponse::Unhandled;
        std::string message = res.status == 404   ? "not found: " + req.path
                              : res.status == 413 ? "request body too large"
                                                  : "request failed";
        res.set_content(nlohmann::json{{"error", message}}.dump(), "application/json");
        return httplib::Server::HandlerResponse::Handled;
    });
    // A bug in a handler must not take the server down or leak internals to the client.
    svr.set_exception_handler(
        [](const httplib::Request&, httplib::Response& res, std::exception_ptr) {
            res.status = 500;
            res.set_content(nlohmann::json{{"error", "internal server error"}}.dump(),
                            "application/json");
        });
}

HttpServer::~HttpServer() = default;

int HttpServer::bind() {
    httplib::Server& svr = impl_->server;
    const ServerOptions& o = impl_->options;
    const int port = o.port == 0 ? svr.bind_to_any_port(o.host)
                                 : (svr.bind_to_port(o.host, o.port) ? o.port : -1);
    if (port < 0) {
        throw std::runtime_error("cannot listen on " + o.host + ":" + std::to_string(o.port) +
                                 " (port in use?)");
    }
    return port;
}

void HttpServer::run() { impl_->server.listen_after_bind(); }

void HttpServer::stop() { impl_->server.stop(); }

}  // namespace inference::server
