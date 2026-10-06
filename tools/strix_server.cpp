// strix-server: Qwen3.8-Flash-Next behind the OpenAI Chat Completions API. Loads the converted weights (the shipping
// layout U - gdn_in - g128 unless --weights), the n-gram table and the tokenizer, then serves on --host:--port (default
// 0.0.0.0:5300) until SIGINT / SIGTERM.
//
// Settings (serve/server_config.hpp): `--config PATH` reads a key = value file
// (deploy/strix-server.conf is the served one), `--name value` flags override it, `--print-config` prints the
// effective settings and exits, `--help` lists every option with its default and range.
// Default paths are under $STRIX_MODELS_DIR (else ~/models/strix-infer).

#include "common/check.hpp"
#include "runtime/qwen4exp.hpp"
#include "serve/engine.hpp"
#include "serve/http_server.hpp"
#include "serve/json.hpp"
#include "serve/log.hpp"
#include "serve/prompt_cache.hpp"
#include "serve/qwen4exp_backend.hpp"
#include "serve/server_config.hpp"
#include "serve/tokenizer.hpp"

#include <csignal>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>

using namespace strix;

namespace {

std::string models_dir() {
    const char *env = std::getenv("STRIX_MODELS_DIR");
    if (env && *env) return env;
    const char *home = std::getenv("HOME");
    STRIX_CHECK(home && *home, "strix_server: neither STRIX_MODELS_DIR nor HOME is set");
    return std::string(home) + "/models/strix-infer";
}

// generation_config.json's sampling defaults (temperature, top_k, top_p); every one must be there.
SamplingParams sampling_defaults(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    STRIX_CHECK(f.good(), "strix_server: can't open generation config '", path, "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    const json::Value g = json::Value::parse(ss.str());
    SamplingParams p;
    const auto need = [&](const char *k) -> const json::Value & {
        const json::Value *v = g.find(k);
        STRIX_CHECK(v != nullptr, "strix_server: '", path, "' has no '", k, "'");
        return *v;
    };
    p.temperature = need("temperature").as_double(path + ".temperature");
    p.top_k = need("top_k").as_int(path + ".top_k");
    p.top_p = need("top_p").as_double(path + ".top_p");
    return p;
}

HttpServer *g_server = nullptr;
void on_signal(int) {
    if (g_server) g_server->stop();
}

int run(int argc, char **argv) {
    ServerSettings settings(strix_server_options(models_dir(), NgramTableRows::kDefaultCacheRows));
    settings.load(std::vector<std::string>(argv + 1, argv + argc));
    if (settings.help()) {
        std::fputs(settings.usage("strix_server").c_str(), stdout);
        return 0;
    }
    if (settings.print_config()) {
        std::fputs(settings.render().c_str(), stdout);
        return 0;
    }
    slog(LogLevel::Info, "startup: config file %s",
         settings.config_path().empty() ? "none (defaults and flags only)" : settings.config_path().c_str());
    for (const ConfigOption &o : settings.options()) {
        const ConfigSetting &s = settings.setting(o.name);
        slog(LogLevel::Info, "startup: config %s = %s (%s)", o.name.c_str(), s.value.c_str(), s.source.c_str());
    }
    const std::string weights = settings.text("weights"), ngram = settings.text("ngram");
    const std::string tokenizer = settings.text("tokenizer"), gen_config = settings.text("generation-config");
    ServerConfig cfg;
    cfg.host = settings.text("host");
    cfg.port = (int)settings.integer("port");
    cfg.model_id = settings.text("model-id");
    const int64_t capacity = settings.integer("capacity"), chunk = settings.integer("chunk");
    const float yarn_factor = (float)settings.number("rope-yarn-factor");
    const std::string cache_dir = settings.text("prompt-cache-dir");
    const int64_t cache_gib = settings.integer("prompt-cache-gib");
    PromptCache::Options cache_opts;
    cache_opts.ram_margin = (uint64_t)settings.integer("prompt-cache-ram-margin-gib") << 30;
    cache_opts.idle_seconds = settings.number("prompt-cache-idle-s");
    cache_opts.write_gib_per_hour = settings.number("prompt-cache-write-gib-per-hour");
    cache_opts.checkpoints_per_chain = (int)settings.integer("prompt-cache-checkpoints");
    cache_opts.delta_max_tokens = settings.integer("prompt-cache-delta-max-tokens");
    cache_opts.disk_free_margin = (uint64_t)settings.integer("prompt-cache-disk-free-gib") << 30;
    const bool use_mtp = settings.on("mtp");
    const int64_t mtp_draft = settings.integer("mtp-draft"), mtp_vocab = settings.integer("mtp-vocab");
    const float mtp_margin = (float)settings.number("mtp-margin");
    const int64_t ngram_cache_rows = settings.integer("ngram-cache-rows");
    const std::string capture_dir = settings.text("capture-dir");  // empty: no capture
    const bool think_nudge = settings.on("think-nudge");
    cfg.defaults = sampling_defaults(gen_config);
    const Tokenizer tok(tokenizer);
    slog(LogLevel::Info, "startup: tokenizer %s (%d tokens)", tokenizer.c_str(), tok.size());
    const Qwen4ExpModel model(weights, ngram, kernels::Act::BF16, /*allow_truncated=*/false, ngram_cache_rows,
                              yarn_factor);
    if (yarn_factor > 1.0f)
        slog(LogLevel::Info, "startup: YaRN factor %g: %lld positions (%lld trained), cos/sin factor %.4f",
             (double)yarn_factor, (long long)model.dims().max_positions(), (long long)model.dims().trained_positions,
             (double)model.rope_scale());
    slog(LogLevel::Info, "startup: %.1f GiB of weights loaded in %.1f s",
                 (double)model.weights().data_bytes() / (1ull << 30), model.weights().load_seconds());
    Qwen4ExpBackend backend(model, capacity, chunk, use_mtp, mtp_vocab);
    std::unique_ptr<PromptCache> cache;
    if (cache_gib > 0) {
        cache = std::make_unique<PromptCache>(cache_dir, (uint64_t)cache_gib << 30, backend.state_fingerprint(), cache_opts);
        const PromptCacheStats s = cache->stats();
        slog(LogLevel::Info, "startup: prompt cache %s: %lld entries, %.1f of %lld GiB (%lld invalid dropped)",
             cache_dir.c_str(), (long long)s.entries, (double)s.bytes / (1ull << 30), (long long)cache_gib, (long long)s.invalid);
        slog(LogLevel::Info, "startup: prompt cache RAM tier: margin %.0f GiB, idle write after %.0f s, budget %.0f GiB/h, "
             "disk free margin %.0f GiB", (double)cache_opts.ram_margin / (1ull << 30), cache_opts.idle_seconds,
             cache_opts.write_gib_per_hour, (double)cache_opts.disk_free_margin / (1ull << 30));
    }
    Engine::Options eng_opts;
    eng_opts.cache = cache.get();
    eng_opts.mtp_margin = mtp_margin;
    eng_opts.mtp_reject_forward = settings.on("mtp-reject-forward");
    eng_opts.mtp_draft = mtp_draft;
    eng_opts.think_nudge = think_nudge;
    slog(LogLevel::Info, "startup: thinking nudge %s", think_nudge ? "on" : "off");
    if (!capture_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(capture_dir, ec);
        STRIX_CHECK(!ec, "--capture-dir ", capture_dir, ": ", ec.message());
        eng_opts.capture_dir = capture_dir;
        slog(LogLevel::Info, "startup: capturing every request's tokens and MTP steps to %s", capture_dir.c_str());
    }
    Engine engine(backend, tok, eng_opts);
    HttpServer server(engine, tok, cfg);
    const int port = server.bind();
    g_server = &server;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    slog(LogLevel::Info, "startup: %s", backend.describe().c_str());
    slog(LogLevel::Info, "startup: serving %s on %s:%d", cfg.model_id.c_str(), cfg.host.c_str(), port);
    server.serve();
    slog(LogLevel::Info, "stopped");
    g_server = nullptr;
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        slog(LogLevel::Error, "fatal: %s", e.what());
        return 1;
    }
}
