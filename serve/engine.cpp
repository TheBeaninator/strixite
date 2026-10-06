#include "serve/engine.hpp"
#include "serve/think_nudge.hpp"

#include "serve/capture.hpp"
#include "serve/log.hpp"
#include "serve/resume_loss.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <cstdio>

namespace strix {

namespace {
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

Engine::Engine(LmBackend &backend, const Tokenizer &tok, const Options &options)
    : be_(backend),
      tok_(tok),
      options_(options),
      cache_(options.cache),
      capacity_(backend.capacity()),
      im_start_(tok.id_of("<|im_start|>")),
      im_end_(tok.id_of("<|im_end|>")),
      eot_(tok.id_of("<|endoftext|>")),
      thinking_stop_(tok.encode(kThinkingStop)),
      nudge_{tok.encode(kThinkNudge1), tok.encode(kThinkNudge2)} {
    STRIX_CHECK(backend.logits_row() >= tok.size(), "Engine: logits rows of ", backend.logits_row(),
                " are shorter than the tokenizer's ", tok.size(), " tokens");
    STRIX_CHECK(backend.max_chunk() >= (int64_t)thinking_stop_.size() && backend.capacity() >= 2, "Engine: backend chunk ",
                backend.max_chunk(), " (the thinking stop is ", thinking_stop_.size(), " tokens), capacity ", backend.capacity());
    STRIX_CHECK(std::count(thinking_stop_.begin(), thinking_stop_.end(), tok.id_of("</think>")) == 1,
                "Engine: the thinking stop text doesn't tokenize to exactly one </think>");
    for (const auto &n : nudge_)
        STRIX_CHECK(!n.empty() && (int64_t)n.size() <= backend.max_chunk() &&
                        std::count(n.begin(), n.end(), tok.id_of("</think>")) == 0,
                    "Engine: a thinking nudge tokenizes to ", n.size(), " tokens (backend chunk ", backend.max_chunk(),
                    ") or contains </think> - it must leave the think block open");
    STRIX_CHECK(options_.mtp_margin >= 0.0f, "Engine: mtp_margin ", options_.mtp_margin, ", expected >= 0");
    STRIX_CHECK(options_.mtp_draft >= 1 && options_.mtp_draft <= 15, "Engine: mtp_draft ", options_.mtp_draft,
                ", expected 1..15");
    STRIX_CHECK(options_.capture_dir.empty() || std::filesystem::is_directory(options_.capture_dir),
                "Engine: capture_dir '", options_.capture_dir, "' is not a directory");
    be_.reset();
    idle_since_ = now_s();
    thread_ = std::thread([this] { worker(); });
}

Engine::Engine(LmBackend &backend, const Tokenizer &tok, PromptCache *cache, float mtp_margin)
    : Engine(backend, tok, [&] {
          Options o;
          o.cache = cache, o.mtp_margin = mtp_margin;
          return o;
      }()) {}

Engine::~Engine() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    for (Job &j : queue_) {
        GenerationResult r;
        r.finish_reason = "error", r.error = "server shutting down";
        j.sink->on_done(r);
    }
}

void Engine::submit(GenerationRequest req, std::shared_ptr<GenerationSink> sink) {
    STRIX_CHECK(sink != nullptr, "Engine::submit: null sink");
    STRIX_CHECK(!req.prompt.empty(), "Engine::submit: empty prompt");
    STRIX_CHECK(req.thinking_budget >= -1, "Engine::submit: thinking budget ", req.thinking_budget, " (-1 = none)");
    STRIX_CHECK(req.max_tokens >= 1 && (int64_t)req.prompt.size() + req.max_tokens <= capacity_,
                "Engine::submit: prompt ", req.prompt.size(), " + max_tokens ", req.max_tokens, " exceed the capacity ",
                capacity_, " (the API layer clamps)");
    for (int32_t id : req.prompt)
        STRIX_CHECK(id >= 0 && id < tok_.size(), "Engine::submit: prompt token ", id, " outside [0, ", tok_.size(), ")");
    {
        std::lock_guard<std::mutex> lock(mu_);
        STRIX_CHECK(!stop_, "Engine::submit: shutting down");
        queue_.push_back({std::move(req), std::move(sink), now_s()});
        ++stats_.queued;
    }
    cv_.notify_one();
}

EngineStats Engine::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    EngineStats s = stats_;
    s.idle_seconds = s.busy || s.queued ? 0.0 : std::max(0.0, now_s() - idle_since_);
    return s;
}

void Engine::worker() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            job = std::move(queue_.front());
            queue_.pop_front();
            --stats_.queued, stats_.busy = 1;
        }
        run(job);
        std::lock_guard<std::mutex> lock(mu_);
        stats_.busy = 0;
        idle_since_ = now_s();
        stats_.live_tokens = (int64_t)seq_.size();
        stats_.snapshot_pos = be_.snapshot_pos(LmBackend::kTurnSlot);
    }
}

int64_t Engine::resume_point(const std::vector<int32_t> &prompt, int64_t id, bool &from_disk, bool &from_ram) {
    from_disk = false, from_ram = false;
    const int64_t P = (int64_t)prompt.size();
    int64_t lcp = 0;
    while (lcp < (int64_t)seq_.size() && lcp < P && seq_[(size_t)lcp] == prompt[(size_t)lcp]) ++lcp;
    // In memory: the live state if the prompt extends it, else the better of the snapshots.
    int64_t best = 0, how = -2;  // -1 = live, >= 0 = slot
    if (lcp == (int64_t)seq_.size() && lcp < P && be_.pos() == lcp) best = lcp, how = -1;
    for (int slot : {LmBackend::kTurnSlot, LmBackend::kSystemSlot, LmBackend::kUserSlot}) {
        const int64_t sp = be_.snapshot_pos(slot);
        if (sp > best && sp <= lcp && sp < P && be_.can_restore_snapshot(slot)) best = sp, how = slot;
    }
    // In the prompt cache (RAM, else disk), if it covers more.
    if (cache_) {
        const int64_t dn = cache_->best(prompt, P - 1);
        if (dn > best) {
            const double t0 = now_s();
            size_t bytes = 0;
            bool ram = false;
            int64_t delta_from = 0;
            const bool loaded = cache_->with_state(prompt, dn, [&](const HostBuffer &state, int64_t from, int64_t to) {
                be_.import_state(state, to, from);  // a delta entry: its base (0, base), then the delta (base, dn)
                bytes += state.size();
                if (from > 0) delta_from = from;
            }, &ram);
            if (loaded) {
                slog(LogLevel::Info, "req %lld resume: %lld of %lld tokens from the prompt cache's %s (%.0f MB in %.0f ms%s)",
                     (long long)id, (long long)dn, (long long)P, ram ? "RAM" : "disk", bytes / 1e6, (now_s() - t0) * 1e3,
                     delta_from > 0 ? (", base " + std::to_string(delta_from) + " + delta").c_str() : "");
                from_ram = ram;
                seq_.assign(prompt.begin(), prompt.begin() + dn);
                from_disk = true;
                return dn;
            }
        }
    }
    if (how == -1 && best > 0) {
        slog(LogLevel::Info, "req %lld resume: %lld of %lld tokens from the live session", (long long)id, (long long)best,
             (long long)P);
        return best;
    }
    if (how == -1) {  // the live session, but nothing of it matches
        slog(LogLevel::Info, "req %lld resume: none - all %lld tokens prefilled", (long long)id, (long long)P);
        return 0;
    }
    if (how >= 0) {
        be_.restore_snapshot((int)how);
        seq_.resize((size_t)best);
        slog(LogLevel::Info, "req %lld resume: %lld of %lld tokens from the %s snapshot", (long long)id, (long long)best,
             (long long)P, how == LmBackend::kTurnSlot ? "turn" : how == LmBackend::kSystemSlot ? "system" : "user-turn");
        return best;
    }
    be_.reset();
    seq_.clear();
    slog(LogLevel::Info, "req %lld resume: none - all %lld tokens prefilled", (long long)id, (long long)P);
    return 0;
}

void Engine::capture(const GenerationRequest &req, const std::vector<CaptureStep> &steps, const std::string &reason) {
    // A side channel: a failure here is logged as an error and never touches the session or the response.
    try {
        const int64_t P = (int64_t)req.prompt.size();
        STRIX_CHECK((int64_t)seq_.size() >= P && std::equal(req.prompt.begin(), req.prompt.end(), seq_.begin()),
                    "Engine::capture: the session (", seq_.size(), " tokens) doesn't start with the ", P, "-token prompt");
        CaptureRecord rec;
        rec.id = req.id, rec.prompt_n = P, rec.tokens = seq_, rec.steps = steps;
        rec.tools = req.parser.tools && req.parser.tools->is_array() ? (int64_t)req.parser.tools->as_array("tools").size() : 0;
        rec.one_shot = req.one_shot, rec.user_turn = req.user_turn, rec.finish = reason;
        rec.mtp_draft = options_.mtp_draft, rec.mtp_margin = options_.mtp_margin;
        const int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count();
        write_capture(options_.capture_dir, rec, ms);
    } catch (const std::exception &e) {
        slog(LogLevel::Error, "req %lld capture failed: %s", (long long)req.id, e.what());
    }
}

void Engine::report_resume_loss(const std::vector<int32_t> &prompt, int64_t start, int64_t live_common, int64_t live_n,
                                int64_t id) {
    const int64_t P = (int64_t)prompt.size();
    if (P - 1 - start < kResumeLossReportTokens) return;  // nothing big enough to lose: skip the cache scan
    int64_t common = live_common, their_n = live_n;
    std::string what = "the live session";
    if (cache_) {
        const PromptCache::Nearest n = cache_->nearest(prompt);
        if (n.common > common) {
            common = n.common, their_n = n.tokens;
            what = std::string("the prompt cache's ") +
                   (n.kind == PromptCache::Kind::System ? "system-prefix" : n.kind == PromptCache::Kind::Checkpoint ? "user-turn" : "turn") +
                   " entry (" + (n.in_ram ? "RAM" : "disk") + ")";
        }
    }
    const std::string text = resume_loss_text(prompt, start, common, their_n, what, im_start_);
    if (text.empty()) return;
    const int64_t lost = std::min(common, P - 1) - start;
    slog(lost >= kResumeLossWarnTokens ? LogLevel::Warning : LogLevel::Info, "req %lld   %s", (long long)id, text.c_str());
}

namespace {
// A finished request: one headline (what a human scans for), the MTP and PLE detail on their own debug lines.
void log_done(int64_t id, const GenerationResult &r, bool stream) {
    const int64_t prefilled = r.prompt_tokens - r.cached_tokens;
    const LogLevel level = r.finish_reason == "error" ? LogLevel::Error
                           : r.finish_reason == "cancelled" ? LogLevel::Warning
                                                            : LogLevel::Info;
    // The numbers are on the prefill / generate lines; this one closes the turn: how it ended, and the wait before.
    slog(level, "req %lld done: %s%s%s (queue %.2f s, prompt %lld = %lld cached + %lld new, out %lld)", (long long)id,
         r.finish_reason.c_str(), r.error.empty() ? "" : " - ", r.error.c_str(), r.queue_ms / 1e3,
         (long long)r.prompt_tokens, (long long)r.cached_tokens, (long long)prefilled, (long long)r.completion_tokens);
    if (r.mtp_drafted_tokens > 0)
        slog(LogLevel::Debug, "req %lld   mtp %lld/%lld accepted (%.1f%%), %lld rollbacks, after 0/1/2/3+: %lld/%lld/%lld/%lld",
             (long long)id, (long long)r.mtp_accepted_tokens, (long long)r.mtp_drafted_tokens,
             100.0 * (double)r.mtp_accepted_tokens / (double)r.mtp_drafted_tokens, (long long)r.mtp_rollbacks,
             (long long)r.mtp_reject_at[0], (long long)r.mtp_reject_at[1], (long long)r.mtp_reject_at[2],
             (long long)r.mtp_reject_at[3]);
    const BackendStats &b = r.backend;
    if (b.ple_gathers > 0)
        slog(b.ple_wait_seconds > 1.0 ? LogLevel::Warning : LogLevel::Debug,
             "req %lld   ple %lld rows, %lld distinct, %.1f%% cached, %lld read, %lld prefetched, waited %.3f s over %lld",
             (long long)id, (long long)b.ngram_rows_requested, (long long)b.ngram_rows_unique,
             b.ngram_rows_unique > 0 ? 100.0 * (double)b.ngram_rows_cached / (double)b.ngram_rows_unique : 0.0,
             (long long)b.ngram_rows_read, (long long)b.ngram_rows_prefetched, b.ple_wait_seconds, (long long)b.ple_waits);
}

}  // namespace

void Engine::run(Job &job) {
    GenerationRequest &req = job.req;
    GenerationSink &sink = *job.sink;
    GenerationResult res;
    const double t_start = now_s();
    res.queue_ms = (t_start - job.enqueued) * 1e3;
    res.prompt_tokens = (int64_t)req.prompt.size();
    const BackendStats b0 = be_.backend_stats();
    // A turn's lines end with a blank line, after the saves - whichever way run() returns. One
    // no-break space (U+00A0): journald drops a line that is empty or only ASCII whitespace after the "<N>" prefix.
    struct BlankLine {
        ~BlankLine() { slog(LogLevel::Info, "\xc2\xa0"); }
    } blank_line;
    const auto finish = [&](const std::string &reason) {
        res.finish_reason = reason;
        const BackendStats b = be_.backend_stats();
        res.backend.ngram_rows_requested = b.ngram_rows_requested - b0.ngram_rows_requested;
        res.backend.ngram_rows_unique = b.ngram_rows_unique - b0.ngram_rows_unique;
        res.backend.ngram_rows_cached = b.ngram_rows_cached - b0.ngram_rows_cached;
        res.backend.ngram_rows_read = b.ngram_rows_read - b0.ngram_rows_read;
        res.backend.ngram_rows_prefetched = b.ngram_rows_prefetched - b0.ngram_rows_prefetched;
        res.backend.ple_prefetches = b.ple_prefetches - b0.ple_prefetches;
        res.backend.ple_prefetch_skipped = b.ple_prefetch_skipped - b0.ple_prefetch_skipped;
        res.backend.ple_prefetch_seconds = b.ple_prefetch_seconds - b0.ple_prefetch_seconds;
        res.backend.ple_gathers = b.ple_gathers - b0.ple_gathers, res.backend.ple_waits = b.ple_waits - b0.ple_waits;
        res.backend.ple_gather_seconds = b.ple_gather_seconds - b0.ple_gather_seconds;
        res.backend.ple_wait_seconds = b.ple_wait_seconds - b0.ple_wait_seconds;
        res.backend.ple_wait_max_seconds = b.ple_wait_max_seconds;
        {
            std::lock_guard<std::mutex> lock(mu_);
            (reason == "error" ? stats_.requests_error : reason == "cancelled" ? stats_.requests_cancelled : stats_.requests_ok)++;
            stats_.generated_tokens += res.completion_tokens;
            stats_.mtp_drafted_tokens += res.mtp_drafted_tokens;
            stats_.think_nudges += res.think_nudges;
            stats_.mtp_accepted_tokens += res.mtp_accepted_tokens;
            stats_.mtp_rollbacks += res.mtp_rollbacks;
            for (int64_t i = 0; i < kMtpMaxDraft; ++i) stats_.mtp_reject_at[i] += res.mtp_reject_at[i];
            stats_.decode_seconds += res.decode_ms / 1e3;
            if (res.decode_ms > 0 && res.completion_tokens > 1)
                stats_.last_decode_tps = (double)(res.completion_tokens - 1) / (res.decode_ms / 1e3);
        }
        log_done(req.id, res, req.stream);  // here, not in the HTTP thread: the turn's lines stay in order
        sink.on_done(res);
    };
    try {
        sink.on_start();
        const std::vector<int32_t> &prompt = req.prompt;
        const int64_t P = (int64_t)prompt.size();
        bool from_disk = false, from_ram = false;
        int64_t live_common = 0;  // before resume_point moves the live session
        while (live_common < (int64_t)seq_.size() && live_common < P && seq_[(size_t)live_common] == prompt[(size_t)live_common])
            ++live_common;
        const int64_t live_n = (int64_t)seq_.size();
        const int64_t start = resume_point(prompt, req.id, from_disk, from_ram);
        report_resume_loss(prompt, start, live_common, live_n, req.id);
        res.cached_tokens = start;
        // Snapshot points: the prompt's last <|im_start|> (where the generation prompt begins; the next turn resumes
        // there), its second-to-last <|im_start|> (the start of the last user message - the user-turn
        // checkpoint: a client rewriting that message, e.g. dropping a reminder it appended, still matches everything
        // before it), and - for the disk cache - the second <|im_start|> overall (the end of the system
        // prompt, shared by every conversation that has it), unless the disk already holds that prefix.
        int64_t cut = -1, ustart = -1, sys = -1;
        for (int64_t k = P - 1; k > 0; --k)
            if (prompt[(size_t)k] == im_start_) {
                if (cut < 0) cut = k;
                else if (ustart < 0) ustart = k;
                sys = k;
            }
        // A prompt without <|im_start|> (a raw /v1/completions prompt): the turn point is the prompt's end, so a
        // next request that extends it - a benchmark priming a context, then adding to it - resumes there instead of
        // prefilling the primed part again. Chat prompts always have one; nothing changes for them.
        if (cut < 0) cut = P;
        const bool save_sys = cache_ && sys > start && sys < cut && !cache_->contains(prompt.data(), sys);
        // The user-turn checkpoint: only for user turns, only past the system prefix (the first user message's
        // start is the system entry) and not behind the resume point (then it was saved by an earlier request).
        const int64_t ust = cache_ && req.user_turn && ustart > sys && ustart >= start ? ustart : -1;
        // Logits rows as the sampler takes them: for greedy / top_k 1..20 (the served defaults) each row's top 20
        // candidates, reduced where the logits are (kernels/logits_topk on the real model) - ~160 bytes a row to the
        // host instead of ~1 MB and a full-row scan; same tokens either way.
        const bool cands = Sampler(req.sampling, tok_.size()).takes_candidates();
        const int64_t n_valid = tok_.size();
        LogitRows logits;
        // A chunk ends at the backend's max chunk, and at the turn / user-start / system snapshot points.
        auto chunk_end = [&](int64_t pos) {
            int64_t end = std::min(P, pos + be_.max_chunk());
            if (cut > pos && cut < end) end = cut;
            if (ust > pos && ust < end) end = ust;
            if (save_sys && sys > pos && sys < end) end = sys;
            return end;
        };
        if (start == ust)  // resumed exactly at the user turn: its start state is the live state right now
            be_.save_snapshot(LmBackend::kUserSlot);
        for (int64_t pos = start; pos < P;) {
            const int64_t end = chunk_end(pos);
            if (end < P)
                be_.set_lookahead(std::vector<int32_t>(prompt.begin() + end, prompt.begin() + chunk_end(end)));
            logits = be_.forward_rows(std::vector<int32_t>(prompt.begin() + pos, prompt.begin() + end), end == P ? 1 : 0,
                                      cands, n_valid);
            seq_.insert(seq_.end(), prompt.begin() + pos, prompt.begin() + end);
            pos = end;
            if (pos == cut) be_.save_snapshot(LmBackend::kTurnSlot);
            if (pos == ust) be_.save_snapshot(LmBackend::kUserSlot);
            if (save_sys && pos == sys) be_.save_snapshot(LmBackend::kSystemSlot);
            sink.on_progress(pos, P);
            if (pos < P && sink.cancelled()) {
                res.prompt_ms = (now_s() - t_start) * 1e3;
                return finish("cancelled");
            }
        }
        const double t_first = now_s();
        res.prompt_ms = (t_first - t_start) * 1e3;
        slog(LogLevel::Info, "req %lld prefill: %lld new tokens after %lld cached in %.2f s = %.0f t/s", (long long)req.id,
             (long long)(P - start), (long long)start, res.prompt_ms / 1e3,
             res.prompt_ms > 0 ? (double)(P - start) / (res.prompt_ms / 1e3) : 0.0);
        {
            std::lock_guard<std::mutex> lock(mu_);
            stats_.prompt_tokens += P, stats_.cached_tokens += start, stats_.prefill_tokens += P - start;
            (start > 0 ? stats_.cache_hits : stats_.cache_misses)++;
            if (from_disk) ++stats_.disk_hits, stats_.disk_tokens += start, stats_.ram_hits += from_ram;
            stats_.prefill_seconds += res.prompt_ms / 1e3;
            if (P > start) stats_.last_prefill_tps = (double)(P - start) / (res.prompt_ms / 1e3);
        }

        Sampler sampler(req.sampling, tok_.size());
        OutputParser parser(tok_, req.parser);
        ThinkWatch watch{ThinkWatch::Policy{}};
        // A sampled token just fed to the parser: the nudge watch follows the reasoning ones.
        const auto watch_tok = [&](int32_t t) {
            if (options_.think_nudge && parser.in_reasoning()) watch.observe(t, tok_.decode({t}));
        };
        std::vector<OutputEvent> events;
        std::string reason;
        std::vector<CaptureStep> steps;  // what each decode step did, for --capture-dir
        auto step = [&](int64_t drafted, int64_t accepted, int64_t emitted) {
            if (!options_.capture_dir.empty())
                steps.push_back({(uint8_t)drafted, (uint8_t)accepted, (uint8_t)emitted});
        };
        // PF-1: a token sampled at a rejected draft's position, already counted and fed to the parser but not yet
        // forwarded - the next step starts from it (drafts from it, then verifies it as row 0).
        int32_t carry = -1;
        for (;;) {
            if (sink.cancelled()) {
                reason = "cancelled";
                break;
            }
            // Thinking budget spent: the engine closes the think block itself (fed, not sampled); sampling goes on.
            if (carry < 0 && req.thinking_budget >= 0 && !res.thinking_budget_hit && parser.in_reasoning() &&
                parser.reasoning_tokens() >= req.thinking_budget) {
                const int64_t n = (int64_t)thinking_stop_.size();
                if (res.completion_tokens + n >= req.max_tokens) {  // no room left for it and an answer
                    reason = "length";
                    break;
                }
                res.thinking_budget_hit = true;
                for (int32_t id : thinking_stop_) parser.feed(id, events);
                if (!events.empty()) sink.on_events(events), events.clear();
                res.completion_tokens += n;
                logits = be_.forward_rows(thinking_stop_, 1, cands, n_valid);
                seq_.insert(seq_.end(), thinking_stop_.begin(), thinking_stop_.end());
                step(kThinkingStopStep, 0, n);
                continue;
            }
            // Thinking going in circles: feed a nudge into the think block (never a </think>), then go on sampling.
            if (carry < 0 && options_.think_nudge && parser.in_reasoning()) {  // injections wait until carry is forwarded
                if (const int d = watch.due()) {
                    const std::vector<int32_t> &nt = nudge_[d - 1];
                    const int64_t n = (int64_t)nt.size();
                    if (res.completion_tokens + n < req.max_tokens) {
                        slog(LogLevel::Info, "req %lld think nudge %d at %lld thinking tokens (self-copy %.0f%% over the last %lld)",
                             (long long)req.id, d, (long long)watch.thinking_tokens(), 100.0 * watch.window_rate(),
                             (long long)ThinkWatch::Policy{}.window);
                        for (int32_t t : nt) parser.feed(t, events);
                        if (!events.empty()) sink.on_events(events), events.clear();
                        res.completion_tokens += n;
                        ++res.think_nudges;
                        logits = be_.forward_rows(nt, 1, cands, n_valid);
                        seq_.insert(seq_.end(), nt.begin(), nt.end());
                        step(kThinkingStopStep, 0, n);
                        watch.fired();
                        continue;
                    }
                }
            }
            int32_t id;
            if (carry >= 0) {  // sampled, counted and parsed at the rejected position: straight to drafting
                id = carry;
                carry = -1;
            } else {
                id = sampler.sample(logits, 0);
                ++res.completion_tokens;
                if (id == im_end_ || id == eot_) {
                    reason = "stop";
                    break;
                }
                const bool stopped = parser.feed(id, events);
                watch_tok(id);
                if (!events.empty()) sink.on_events(events), events.clear();
                if (stopped) {
                    reason = "stop";
                    break;
                }
                if (res.completion_tokens >= req.max_tokens) {
                    reason = "length";
                    break;
                }
            }

            // MTP: up to mtp_draft greedy drafts from the head (chained), each while its top-1 margin reaches
            // mtp_margin (and no end-of-turn); one verify forward of id + the drafts. Each position's token is still
            // sampled from the trunk's own logits (row j) - a draft only saves the forward when it matches.
            if (be_.has_mtp()) {
                const int64_t room = req.max_tokens - res.completion_tokens;  // >= 1 here
                const int64_t steps = std::min(options_.mtp_draft, room);
                std::vector<int32_t> ids{id};
                // Each token's PLE rows start loading as soon as it is known, while the next draft runs - so the
                // verify (or id's own forward, when no draft passes) finds them in the row cache.
                be_.prefetch_ple(ids, 0);
                for (int64_t step = 0; step < steps; ++step) {
                    const Top2 t = be_.forward_mtp_top2(ids.back(), step);
                    STRIX_CHECK(!t.nan, "Engine: NaN in the MTP draft logits (step ", step, ")");
                    const int32_t d = t.best;
                    if (t.best_v - t.second_v < options_.mtp_margin || d == im_end_ || d == eot_) break;
                    ids.push_back(d);
                    if (step + 1 < steps) be_.prefetch_ple(ids, (int64_t)ids.size() - 1);  // the last: nothing to hide behind
                }
                const int64_t k = (int64_t)ids.size() - 1;
                if (k > 0) {
                    res.mtp_drafted_tokens += k;
                    const LogitRows ml = be_.forward_verify_rows(ids, k + 1, cands, n_valid);
                    STRIX_CHECK(ml.rows == k + 1, "Engine: verify of ", k + 1, " tokens returned ", ml.rows,
                                " logits rows");
                    // Accept drafts while the sampled token is the draft.
                    int64_t j = 0;
                    int32_t v = -1;
                    bool ended = false;
                    for (; j < k; ++j) {
                        v = sampler.sample(ml, j);
                        if (v != ids[(size_t)j + 1]) break;
                        ++res.mtp_accepted_tokens, ++res.completion_tokens;
                        const bool st = parser.feed(v, events);
                        watch_tok(v);
                        if (!events.empty()) sink.on_events(events), events.clear();
                        if (st || res.completion_tokens >= req.max_tokens) {
                            reason = st ? "stop" : "length";
                            ended = true;
                            ++j;
                            break;
                        }
                    }
                    if (ended) {  // keep id + the j accepted drafts in the session, drop the rest
                        if (j < k) be_.keep_verify_prefix(ids, j + 1);
                        else be_.keep_verify();
                        seq_.insert(seq_.end(), ids.begin(), ids.begin() + j + 1);
                        step(k, j, j + 1);
                        break;
                    }
                    if (j == k) {  // all accepted: the last row is the next token's logits
                        be_.keep_verify();
                        seq_.insert(seq_.end(), ids.begin(), ids.end());
                        step(k, k, k + 1);
                        logits = ml.row_of(k);
                        continue;
                    }
                    // Draft j rejected; v is this position's sampled token. Keep id and the j accepted drafts from the
                    // verify (the backend doesn't run them again - keep_verify_prefix). The session's state is then
                    // "after ids[j]", exactly what a step's drafting starts from, so v carries into the next step
                    // (PF-1) instead of a forward of its own (mtp_reject_forward: the old path).
                    ++res.mtp_rollbacks, ++res.completion_tokens, ++res.mtp_reject_at[std::min<int64_t>(j, kMtpMaxDraft - 1)];
                    {  // v's PLE rows load while the prefix is kept (its forward comes right after)
                        std::vector<int32_t> next(ids.begin(), ids.begin() + j + 1);
                        next.push_back(v);
                        be_.prefetch_ple(next, j + 1);
                    }
                    be_.keep_verify_prefix(ids, j + 1);
                    seq_.insert(seq_.end(), ids.begin(), ids.begin() + j + 1);
                    if (v == im_end_ || v == eot_) {
                        step(k, j, j + 1);
                        reason = "stop";
                        break;
                    }
                    const bool st = parser.feed(v, events);
                    watch_tok(v);
                    if (!events.empty()) sink.on_events(events), events.clear();
                    if (options_.mtp_reject_forward) {
                        logits = be_.forward_rows({v}, 1, cands, n_valid);
                        seq_.push_back(v);
                        step(k, j, j + 2);
                    } else {
                        step(k, j, j + 1);  // v is emitted by the next step
                    }
                    if (st || res.completion_tokens >= req.max_tokens) {
                        reason = st ? "stop" : "length";
                        break;
                    }
                    if (!options_.mtp_reject_forward) carry = v;
                    continue;
                }
            }
            logits = be_.forward_rows({id}, 1, cands, n_valid);
            seq_.push_back(id);
            step(0, 0, 1);
        }
        parser.finish(events);
        if (!events.empty()) sink.on_events(events);
        res.reasoning_tokens = parser.reasoning_tokens();
        res.dropped_partial_call = parser.dropped_partial_call();
        res.decode_ms = (now_s() - t_first) * 1e3;
        if (reason == "stop" && parser.tool_calls() > 0) reason = "tool_calls";
        slog(LogLevel::Info, "req %lld generate: %lld tokens (think %lld) in %.2f s = %.1f t/s", (long long)req.id,
             (long long)res.completion_tokens, (long long)res.reasoning_tokens, res.decode_ms / 1e3,
             res.decode_ms > 0 && res.completion_tokens > 1 ? (res.completion_tokens - 1) / (res.decode_ms / 1e3) : 0.0);
        finish(reason);
        if (!options_.capture_dir.empty() && reason != "cancelled") capture(req, steps, reason);
        // After the response: the new prefixes go to the prompt cache's RAM tier (the disk only later, if at all).
        if (cache_ && reason != "cancelled") {
            const double t0 = now_s();
            int saved = 0;
            const auto save = [&](int slot, int64_t n, PromptCache::Kind kind, const char *what) {
                // Already cached first: a request resumed exactly at its turn entry has no snapshot there, and needs none.
                const char *skip = n <= 0                                ? "no such point in the prompt"
                                   : cache_->contains(prompt.data(), n) ? nullptr
                                   : be_.snapshot_pos(slot) != n         ? "no snapshot there"
                                   : !be_.can_restore_snapshot(slot)     ? "snapshot no longer valid"
                                                                         : "";
                if (skip == nullptr) return;  // already cached
                if (*skip) {
                    slog(LogLevel::Debug, "req %lld save: not the %s at %lld: %s (snapshot at %lld)", (long long)req.id, what,
                         (long long)n, skip, (long long)be_.snapshot_pos(slot));
                    return;
                }
                const double t_e = now_s();
                // The replaced turn's buffer first (grown by this turn's few MB), else a spare: a fresh multi-GB buffer
                // costs ~100 ms per GB of page population on the request path. A Turn of a deep conversation goes as a
                // delta on the whole entry it extends (PromptCache "Delta entries").
                const int64_t base_n = cache_->delta_base(prompt.data(), n, kind);
                HostBuffer state = kind == PromptCache::Kind::System ? HostBuffer()
                                                                      : cache_->take_replaced_buffer(prompt.data(), n, base_n);
                if (state.capacity() == 0) state = cache_->take_buffer();
                be_.export_snapshot(slot, state, base_n);
                const double ms = (now_s() - t_e) * 1e3;  // > 0.5 s: the request path waited - worth seeing
                const std::string delta = base_n > 0 ? ", delta on " + std::to_string(base_n) : "";
                slog(ms > 500 ? LogLevel::Warning : LogLevel::Debug, "req %lld save: the %s at %lld tokens (%.0f MB, export %.0f ms%s%s)",
                     (long long)req.id, what, (long long)n, state.size() / 1e6, ms, delta.c_str(),
                     req.one_shot ? ", RAM only" : "");
                if (!cache_->put(std::vector<int32_t>(prompt.begin(), prompt.begin() + n), kind, std::move(state),
                                 req.one_shot, base_n))
                    return;
                ++saved;
            };
            if (save_sys) save(LmBackend::kSystemSlot, sys, PromptCache::Kind::System, "system prefix");
            // The checkpoint is the start of the user message, not its end: clients rewrite the latest user message
            // on the next one (opencode's plan mode appends a reminder to it and strips it later), so an entry that
            // includes it never matches again. The turn itself is a plain Turn, replaced by the tool loop's next.
            if (ust >= 0) save(LmBackend::kUserSlot, ust, PromptCache::Kind::Checkpoint, "user-turn checkpoint");
            save(LmBackend::kTurnSlot, cut, PromptCache::Kind::Turn, "turn");
            std::lock_guard<std::mutex> lock(mu_);
            stats_.disk_saves += saved, stats_.export_seconds += now_s() - t0;
        }
    } catch (const std::exception &e) {
        // A failed forward leaves the session unusable until reset: start clean for the next request.
        slog(LogLevel::Error, "req %lld failed: %s", (long long)req.id, e.what());
        try {
            be_.reset();
        } catch (const std::exception &e2) {
            slog(LogLevel::Error, "req %lld reset after the failure failed too: %s", (long long)req.id, e2.what());
        }
        seq_.clear();
        res.error = e.what();
        finish("error");
    }
}

}  // namespace strix
