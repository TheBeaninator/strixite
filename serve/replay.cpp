#include "serve/replay.hpp"

#include <algorithm>
#include <chrono>
#include <memory>

#include "common/check.hpp"

namespace strix {

namespace {
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

ReplayResult replay_teacher_forced(LmBackend &be, const CaptureRecord &rec, const ReplayOptions &opt) {
    const int64_t P = rec.prompt_n, T = (int64_t)rec.tokens.size();
    STRIX_CHECK(P >= 1 && P <= T, "replay: capture req ", rec.id, ": prompt_n ", P, " of ", T, " tokens");
    STRIX_CHECK(opt.max_context >= 1 && opt.max_gen >= 0 && opt.mtp_draft >= 0 && opt.mtp_draft <= 15 && opt.mtp_margin >= 0,
                "replay: options max_context ", opt.max_context, ", max_gen ", opt.max_gen, ", mtp_draft ", opt.mtp_draft,
                ", mtp_margin ", opt.mtp_margin);
    STRIX_CHECK(opt.mtp_draft == 0 || be.has_mtp(), "replay: ", opt.mtp_draft, " drafts asked for, the backend has no MTP head");
    using S = ReplayOptions::Sample;
    STRIX_CHECK(opt.sample == S::Off || (opt.n_valid >= 1 && opt.n_valid <= be.logits_row()), "replay: n_valid ",
                opt.n_valid, " with sampling on, expected 1..", be.logits_row());
    const bool sampling = opt.sample != S::Off, cands = opt.sample == S::Candidates;
    std::unique_ptr<Sampler> sampler;
    if (sampling) {
        sampler = std::make_unique<Sampler>(opt.sampling, opt.n_valid);
        STRIX_CHECK(!cands || sampler->takes_candidates(), "replay: candidates asked for, but temperature ",
                    opt.sampling.temperature, " / top_k ", opt.sampling.top_k, " need whole rows");
    }
    ReplayResult r;
    // Forwards as the engine runs them: with sampling, the rows through forward_rows / forward_verify_rows and rows
    // [0, n) sampled; without, the plain calls (logits copied back, unused).
    auto sample_rows = [&](const LogitRows &l, int64_t n) {
        for (int64_t q = 0; q < n; ++q) r.sample_hash = r.sample_hash * 1000003u + (uint64_t)sampler->sample(l, q);
        r.sampled += n;
    };
    auto fwd = [&](const std::vector<int32_t> &ids, bool want) {
        if (!sampling) return (void)be.forward(ids, want);
        const LogitRows l = be.forward_rows(ids, want ? 1 : 0, cands, opt.n_valid);
        if (want) sample_rows(l, 1);
    };
    r.context_n = std::min(P, opt.max_context);
    const int64_t G = opt.max_gen > 0 ? std::min(T - P, opt.max_gen) : T - P;
    STRIX_CHECK(G >= 1, "replay: capture req ", rec.id, " generated no tokens");
    STRIX_CHECK(r.context_n + G <= be.capacity(), "replay: capture req ", rec.id, ": ", r.context_n, " + ", G,
                " tokens exceed the backend's capacity ", be.capacity());
    const int32_t *ctx = rec.tokens.data() + (P - r.context_n);
    const int32_t *gen = rec.tokens.data() + P;

    be.reset();
    double t0 = now_ms();
    for (int64_t pos = 0; pos < r.context_n;) {  // prefill in the backend's chunks; the last one's logits, as the engine
        const int64_t end = std::min(r.context_n, pos + be.max_chunk());
        fwd(std::vector<int32_t>(ctx + pos, ctx + end), end == r.context_n);
        pos = end;
    }
    r.prefill_ms = now_ms() - t0;
    if (opt.after_prefill) opt.after_prefill();
    const BackendStats s0 = be.backend_stats();

    // gen[i] is the token the engine would have just sampled: known, not yet forwarded.
    t0 = now_ms();
    for (int64_t i = 0; i < G;) {
        const int32_t id = gen[i];
        std::vector<int32_t> ids{id};
        const int64_t steps = std::min(opt.mtp_draft, G - 1 - i);  // drafts only for tokens that exist in the capture
        if (steps > 0) be.prefetch_ple(ids, 0);
        for (int64_t s = 0; s < steps; ++s) {
            const Top2 t = be.forward_mtp_top2(ids.back(), s);
            STRIX_CHECK(!t.nan, "replay: capture req ", rec.id, ": NaN in the MTP draft logits at token ", i, " step ", s);
            if (t.best_v - t.second_v < opt.mtp_margin) break;
            ids.push_back(t.best);
            if (s + 1 < steps) be.prefetch_ple(ids, (int64_t)ids.size() - 1);
        }
        const int64_t k = (int64_t)ids.size() - 1;
        ++r.forwards;
        if (k == 0) {
            fwd({id}, true);
            i += 1;
            continue;
        }
        r.drafted += k;
        LogitRows vl;
        if (sampling) vl = be.forward_verify_rows(ids, k + 1, cands, opt.n_valid);
        else be.forward_verify(ids, k + 1);
        int64_t j = 0;
        while (j < k && ids[(size_t)j + 1] == gen[i + 1 + j]) ++j;
        if (sampling) sample_rows(vl, j + 1);  // the engine samples rows 0..j (row k: the next step's first token)
        r.accepted += j;
        if (j == k) {
            be.keep_verify();
            i += k + 1;
            continue;
        }
        // Draft j rejected: keep id + the j accepted drafts, then forward the captured token in its place.
        ++r.rollbacks;
        be.keep_verify_prefix(ids, j + 1);
        const int64_t v = i + j + 1;  // < G: drafts only reach tokens that exist
        be.prefetch_ple({gen[v]}, 0);
        fwd({gen[v]}, true);
        ++r.forwards;
        i = v + 1;
    }
    r.decode_ms = now_ms() - t0;
    const BackendStats s1 = be.backend_stats();
    r.ple_rows_from_file = (s1.ngram_rows_read + s1.ngram_rows_prefetched) - (s0.ngram_rows_read + s0.ngram_rows_prefetched);
    r.ple_waits = s1.ple_waits - s0.ple_waits;
    r.ple_wait_s = s1.ple_wait_seconds - s0.ple_wait_seconds;
    r.gen_n = G;
    return r;
}

}  // namespace strix
