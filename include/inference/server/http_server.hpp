#pragma once

#include <functional>
#include <memory>
#include <string>

#include "inference/server/demo_service.hpp"

// The HTTP layer of the web demo (cpp-httplib). Deliberately thin: it maps routes to
// DemoService functions, serves the static frontend from a directory, and adds limits and
// headers. All request logic lives in DemoService.
namespace inference::server {

struct ServerOptions {
    std::string host = "127.0.0.1";  // localhost only by default: nothing is exposed
    int port = 8080;                 // 0 = pick a free port
    std::string web_dir;             // static files served at /
};

class HttpServer {
public:
    // Registers the routes. Throws std::runtime_error if web_dir is missing.
    HttpServer(const DemoService& service, ServerOptions options);
    ~HttpServer();
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // Binds the socket; returns the actual port (useful with port 0). Throws on failure.
    int bind();

    // Serves requests until stop() is called (from another thread). Call bind() first.
    void run();

    // Makes run() return. Safe to call from any thread.
    void stop();

private:
    struct Impl;  // hides cpp-httplib from everything that includes this header
    std::unique_ptr<Impl> impl_;
};

}  // namespace inference::server
