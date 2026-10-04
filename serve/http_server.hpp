#pragma once

// The HTTP surface on cpp-httplib: POST /v1/chat/completions (JSON or SSE),
// GET /v1/models, /health, /metrics (Prometheus text), /cache (prefix-reuse counters, for hbench). Requests are
// rendered and tokenized on the HTTP thread, then queued on the Engine; a streaming response relays the engine's
// events as they come, with SSE comment keep-alives while the request waits or prefills; a client that goes away
// cancels its request.

#include "serve/engine.hpp"
#include "serve/tokenizer.hpp"

#include <atomic>
#include <memory>
#include <string>

namespace httplib {
class Server;
}

namespace strix {

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 5300;
    std::string model_id = "qwen3.8-flash-next";
    SamplingParams defaults;  // generation_config.json's
    double keepalive_s = 10;  // SSE comment interval while nothing else is sent
    size_t max_body_bytes = 64u << 20;
};

class HttpServer {
public:
    HttpServer(Engine &engine, const Tokenizer &tok, ServerConfig cfg);
    ~HttpServer();
    // Binds (port 0 = any free port; returns the port) - then serve() blocks until stop().
    int bind();
    void serve();
    void stop();

private:
    Engine &engine_;
    const Tokenizer &tok_;
    ServerConfig cfg_;
    int64_t started_;
    std::atomic<int64_t> next_id_{0};  // request ids for the log ("req <id> ...")
    std::unique_ptr<httplib::Server> svr_;
    void routes();
};

}  // namespace strix
