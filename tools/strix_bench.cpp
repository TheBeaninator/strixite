// strix_bench: single-stream decode timing of Qwen4ExpSession at given context depths,
// on WikiText-2 text, with the engine's own MTP policy replayed teacher-forced over the model's greedy continuation
// (serve/replay.cpp's loop, timed per part), plus golden logits, the 1-node noise floor and needles.
//
//   whole model:          strix_bench --weights W --ngram T --tokenizer J --corpus a.raw,b.raw --out r.json
//   one rank of N:        ... --tp-world 2 --tp-rank 0   (exchanges stubbed: every all-reduce a no-op, counted)
//
// Per depth D (ascending; one session, the context grown by chunked prefill, a snapshot at D restored between parts):
//   ar_tf     T=1 forwards teacher-forced on the corpus continuation (the gate's T1 / T_half: median ms; MTP catch-up off)
//   ar_gen    greedy generation with the MTP catch-up on (as served with drafting gated off) - its tokens are the
//             "capture" the MTP replay decodes (whole model only)
//   verify_k  forward_verify(k) + drop for k = 2..6 (catch-up off and on), drafts: forward_mtp_top2 per chained step
//   mtp       the engine's draft / verify / keep loop over ar_gen's tokens: tokens/s, accepted per step
// --hash-run 1: instead, one fixed call sequence with every result hashed (two builds compared bit for bit).
// Golden + noise floor (whole model): teacher-forced logits over WikiText-2 test after a prefill in chunks of 8192 vs
// 4096 (only the arithmetic order differs) -> mean KL, top-1 agreement, NLL; rows saved for later gates.
// Needles (whole model): 8 passcodes spliced into the context between 4k and 64k, asked for at each depth >= 64k.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "formats/strixw.hpp"  // strix_hash64
#include "kernels/linear_q8.hpp"
#include "runtime/qwen4exp.hpp"
#include "serve/tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace strix;

namespace {

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Args {
    std::map<std::string, std::string> kv;
    std::string get(const std::string &k, const std::string &d = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? d : it->second;
    }
    int64_t num(const std::string &k, int64_t d) const { return kv.count(k) ? std::stoll(kv.at(k)) : d; }
    double real(const std::string &k, double d) const { return kv.count(k) ? std::stod(kv.at(k)) : d; }
};

std::vector<std::string> split(const std::string &s, char c) {
    std::vector<std::string> r;
    std::stringstream ss(s);
    for (std::string x; std::getline(ss, x, c);)
        if (!x.empty()) r.push_back(x);
    return r;
}

struct Stats {
    double p50 = 0, mean = 0, p10 = 0, p90 = 0, min = 0, max = 0;
    int64_t n = 0;
};
Stats stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n = (int64_t)v.size();
    auto q = [&](double f) { return v[(size_t)std::min<double>((double)v.size() - 1, std::floor(f * (double)(v.size() - 1) + 0.5))]; };
    s.p50 = q(0.5), s.p10 = q(0.1), s.p90 = q(0.9), s.min = v.front(), s.max = v.back();
    for (double x : v) s.mean += x;
    s.mean /= (double)v.size();
    return s;
}
std::string js(const Stats &s) {
    char b[256];
    std::snprintf(b, sizeof b, "{\"p50\":%.4f,\"mean\":%.4f,\"p10\":%.4f,\"p90\":%.4f,\"min\":%.4f,\"max\":%.4f,\"n\":%lld}", s.p50,
                  s.mean, s.p10, s.p90, s.min, s.max, (long long)s.n);
    return b;
}

std::string read_file(const std::string &p) {
    std::ifstream f(p, std::ios::binary);
    STRIX_CHECK(f.good(), "strix_bench: cannot read ", p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// log-softmax helpers over a full row
double lse(const float *x, int64_t n) {
    double m = -INFINITY;
    for (int64_t i = 0; i < n; ++i) m = std::max(m, (double)x[i]);
    double s = 0;
    for (int64_t i = 0; i < n; ++i) s += std::exp((double)x[i] - m);
    return m + std::log(s);
}
double kl_rows(const float *p, const float *q, int64_t n) {  // KL(P || Q), P = softmax(p)
    const double lp = lse(p, n), lq = lse(q, n);
    double k = 0;
    for (int64_t i = 0; i < n; ++i) {
        const double a = (double)p[i] - lp;
        k += std::exp(a) * (a - ((double)q[i] - lq));
    }
    return k;
}
int64_t argmax(const float *x, int64_t n) {
    int64_t b = 0;
    for (int64_t i = 1; i < n; ++i)
        if (x[i] > x[b]) b = i;
    return b;
}

}  // namespace

// "on" / "off" / 1 / 0 (the Q4 draft-head switches).
bool onoff_arg(const std::string &v, const char *what) {
    if (v == "on" || v == "1" || v == "true") return true;
    if (v == "off" || v == "0" || v == "false" || v.empty()) return false;
    STRIX_FAIL(what, ": '", v, "', expected on or off");
}

int main(int argc, char **argv) {
    try {
        Args a;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            STRIX_CHECK(k.rfind("--", 0) == 0 && i + 1 < argc, "strix_bench: bad argument ", k);
            a.kv[k.substr(2)] = argv[++i];
        }
        const std::string weights = a.get("weights"), ngram = a.get("ngram"), tokj = a.get("tokenizer");
        const std::vector<std::string> corpus_files = split(a.get("corpus"), ',');
        std::vector<int64_t> depths;
        for (const std::string &d : split(a.get("depths", "4096,65536,163840,400000"), ',')) depths.push_back(std::stoll(d));
        std::sort(depths.begin(), depths.end());
        const int64_t ar_steps = a.num("ar-steps", 128), gen_n = a.num("gen", 256), chunk = a.num("chunk", 8192);
        const int64_t mtp_draft = a.num("mtp-draft", 5), mtp_vocab = a.num("mtp-vocab", 65536);
        const double margin = a.real("mtp-margin", 2.0);
        const bool reject_forward = a.num("mtp-reject-forward", 0) != 0;  // the old reject path (A/B)
        const int64_t cache_rows = a.num("ngram-cache-rows", 8388608);
        const float yarn = (float)a.real("yarn", 2.0);
        const int tp_world = (int)a.num("tp-world", 1), tp_rank = (int)a.num("tp-rank", 0);
        const bool whole = tp_world == 1;
        const int64_t kl_prefix = a.num("kl-prefix", 8192), kl_n = a.num("kl-n", 256);
        const std::string golden_dir = a.get("golden-dir"), out = a.get("out", "/dev/stdout");
        const bool do_quality = whole && a.num("quality", 1) != 0;
        const int64_t verify_reps = a.num("verify-reps", 8), draft_reps = a.num("draft-reps", 8);
        STRIX_CHECK(!weights.empty() && !ngram.empty() && !tokj.empty() && !corpus_files.empty(),
                    "usage: strix_bench --weights W --ngram T --tokenizer J --corpus a,b [--depths ..] [--tp-world N --tp-rank r]");

        const Tokenizer tok(tokj);
        double t0 = now_ms();
        std::vector<int32_t> corpus;
        for (const std::string &f : corpus_files) {
            const std::vector<int32_t> ids = tok.encode(read_file(f));
            corpus.insert(corpus.end(), ids.begin(), ids.end());
        }
        std::fprintf(stderr, "strix_bench: corpus %zu tokens (%.1f s)\n", corpus.size(), (now_ms() - t0) / 1000);
        const int64_t maxD = depths.back();
        STRIX_CHECK((int64_t)corpus.size() > maxD + ar_steps + 64, "strix_bench: corpus has ", corpus.size(),
                    " tokens, need > ", maxD + ar_steps + 64);
        // Needles: spliced into [4096, 65536) of the context (whole model only).
        struct Needle {
            std::string key, code;
            int64_t at;
        };
        std::vector<Needle> needles;
        const char *colors[] = {"crimson", "amber", "cobalt", "violet", "emerald", "silver", "ochre", "teal"};
        const char *animals[] = {"heron", "lynx", "otter", "falcon", "badger", "gecko", "marten", "ibis"};
        if (whole && maxD >= 65536) {
            // A chat turn (the model is an instruct model: raw text makes it close the turn): the context opens a user
            // message, each question ends it and opens the answer with thinking off.
            std::vector<int32_t> ctx = tok.encode("<|im_start|>user\n");
            ctx.insert(ctx.end(), corpus.begin(), corpus.begin() + maxD + ar_steps + 64);
            for (int i = 7; i >= 0; --i) {
                char code[16];
                std::snprintf(code, sizeof code, "%06d", (int)((i * 7919 + 104729) * 37 % 1000000));
                Needle n{std::string(colors[i]) + " " + animals[i], code, 4096 + 2000 + i * 7400};
                const std::vector<int32_t> s = tok.encode(" The secret passcode of the " + n.key + " is " + n.code + ". ");
                ctx.insert(ctx.begin() + n.at, s.begin(), s.end());
                needles.insert(needles.begin(), n);
            }
            ctx.resize((size_t)(maxD + ar_steps + 64));
            std::copy(ctx.begin(), ctx.end(), corpus.begin());
        }

        t0 = now_ms();
        Qwen4ExpModel model(weights, ngram, kernels::Act::BF16, false, cache_rows, yarn,
                                  TpConfig{tp_world, tp_rank, std::max<int64_t>(1, mtp_vocab)});
        const double load_s = (now_ms() - t0) / 1000;
        const Qwen4ExpDims &Dm = model.dims();
        std::fprintf(stderr, "strix_bench: %.1f GiB of weights (rank %d of %d) loaded in %.1f s\n",
                     (double)model.weights().data_bytes() / (1ull << 30), tp_rank, tp_world, load_s);
        const int64_t capacity = ((maxD + gen_n + 4096) / 4096 + 1) * 4096;
        Qwen4ExpSession ses(model, capacity, chunk, PrefillMath::WmmaBf16, model.has_mtp());
        if (model.has_mtp()) ses.set_mtp_vocab(mtp_vocab);
        // --mtp-draft-q4 on: draft over a Q4 copy of the draft vocabulary rows (made here, once)
        const bool draft_q4 = onoff_arg(a.get("mtp-draft-q4", "off"), "strix_bench --mtp-draft-q4");
        if (draft_q4 && model.has_mtp()) {
            const double q0 = now_ms();
            model.make_draft_head_q4(ses.mtp_vocab(), a.num("mtp-draft-q4-group", 64));
            ses.set_mtp_draft_q4(true);
            std::fprintf(stderr, "strix_bench: Q4 draft head: %lld rows made in %.1f s\n", (long long)model.draft_head_q4().N(),
                         (now_ms() - q0) / 1000);
        }
        // Draft equivalence: --mtp-equiv-out FILE (whole model) runs one forward of --equiv-n corpus tokens,
        // keeps the trunk's streams after the last layer (L<last>.out) and the head's 5 chained drafts from the
        // forward's greedy token; --mtp-equiv-in FILE (rank 0 of a TP group) feeds the same ids + streams to its MTP
        // catch-up (Qwen4ExpSession::debug_mtp_feed) and drafts the same chain: the top-2s must be bit-equal.
        std::string equiv_json;
        const std::string equiv_out = a.get("mtp-equiv-out"), equiv_in = a.get("mtp-equiv-in");
        if (!equiv_out.empty() || !equiv_in.empty()) {
            STRIX_CHECK(model.has_mtp(), "strix_bench: --mtp-equiv-* needs the MTP head");
            const int64_t Q = a.num("equiv-n", 4096), n4 = Dm.H * Dm.d, nvq = (int64_t)tok.size();
            STRIX_CHECK(Q >= 1 && Q <= chunk, "strix_bench: --equiv-n ", Q, " vs --chunk ", chunk);
            std::vector<int32_t> ids(corpus.begin(), corpus.begin() + Q);
            std::vector<uint8_t> X((size_t)(Q * n4) * 2);
            int32_t first = 0;
            struct Draft {
                int32_t best;
                float bv, sv;
                int32_t nan;
            };
            std::vector<Draft> want(5), got(5);
            const std::string last = "L" + std::to_string(Dm.layers - 1) + ".out";
            ses.reset();
            if (!equiv_out.empty()) {
                bool seen = false;
                const std::vector<float> l = ses.forward(ids, 1, [&](const std::string &name, const void *dev, int64_t rows,
                                                                      int64_t cols, ProbeType type) {
                    if (name != last) return;
                    STRIX_CHECK(type == ProbeType::BF16 && rows == Q && cols == n4, "strix_bench: ", name, " probe shape");
                    STRIX_HIP_CHECK(hipMemcpy(X.data(), dev, X.size(), hipMemcpyDeviceToHost), "equiv probe");
                    seen = true;
                });
                STRIX_CHECK(seen, "strix_bench: no ", last, " probe");
                first = (int32_t)argmax(l.data(), nvq);
            } else {
                std::ifstream f(equiv_in, std::ios::binary);
                int64_t hdr[3];
                f.read(reinterpret_cast<char *>(hdr), sizeof hdr);
                STRIX_CHECK(f.good() && hdr[0] == Q && hdr[1] == n4, "strix_bench: ", equiv_in, " is for ", hdr[0], " x ", hdr[1]);
                f.read(reinterpret_cast<char *>(ids.data()), (std::streamsize)(ids.size() * 4));
                f.read(reinterpret_cast<char *>(X.data()), (std::streamsize)X.size());
                f.read(reinterpret_cast<char *>(&first), 4);
                f.read(reinterpret_cast<char *>(want.data()), (std::streamsize)(want.size() * sizeof(Draft)));
                STRIX_CHECK(f.good(), "strix_bench: ", equiv_in, " short");
                ses.debug_mtp_feed(ids, X.data(), X.size());
            }
            int32_t t = first;
            for (int s2 = 0; s2 < 5; ++s2) {
                const Qwen4ExpSession::MtpTop2 p = ses.forward_mtp_top2(t, s2);
                got[(size_t)s2] = {p.best, p.best_v, p.second_v, p.nan ? 1 : 0};
                t = p.best;
            }
            if (!equiv_out.empty()) {
                std::ofstream f(equiv_out, std::ios::binary);
                const int64_t hdr[3] = {Q, n4, 2};
                f.write(reinterpret_cast<const char *>(hdr), sizeof hdr);
                f.write(reinterpret_cast<const char *>(ids.data()), (std::streamsize)(ids.size() * 4));
                f.write(reinterpret_cast<const char *>(X.data()), (std::streamsize)X.size());
                f.write(reinterpret_cast<const char *>(&first), 4);
                f.write(reinterpret_cast<const char *>(got.data()), (std::streamsize)(got.size() * sizeof(Draft)));
                STRIX_CHECK(f.good(), "strix_bench: cannot write ", equiv_out);
                equiv_json = ",\"mtp_equiv\":{\"role\":\"reference\",\"file\":\"" + equiv_out + "\"}";
            } else {
                const bool eq = std::memcmp(want.data(), got.data(), want.size() * sizeof(Draft)) == 0;
                std::ostringstream ej;
                ej << ",\"mtp_equiv\":{\"role\":\"check\",\"file\":\"" << equiv_in << "\",\"positions\":" << Q
                   << ",\"first_token\":" << first << ",\"drafts_top2_equal\":" << (eq ? "true" : "false") << ",\"drafts\":[";
                for (size_t k = 0; k < got.size(); ++k)
                    ej << (k ? "," : "") << "{\"want\":[" << want[k].best << "," << want[k].bv << "," << want[k].sv << "],\"got\":["
                       << got[k].best << "," << got[k].bv << "," << got[k].sv << "]}";
                ej << "]}";
                equiv_json = ej.str();
                std::fprintf(stderr, "strix_bench: MTP draft equivalence vs %s: %s\n", equiv_in.c_str(), eq ? "bit-equal" : "DIFFERENT");
            }
            ses.reset();
        }
        const int64_t n_valid = whole ? (int64_t)tok.size() : Dm.lm_rows;
        size_t free_b = 0, total_b = 0;
        STRIX_HIP_CHECK(hipMemGetInfo(&free_b, &total_b), "hipMemGetInfo");

        auto greedy = [&](const std::vector<int32_t> &ids) {  // forward, the best candidate (engine's cand path)
            ses.want_candidates(n_valid);
            ses.forward(ids, 1);
            return ses.candidates().cand[0].id;
        };

        // --hash-run 1 (the "world 1 bit-identical" check): one fixed sequence of calls, every result hashed -
        // the prefill's logits, --gen greedy T = 1 forwards (whole rows), then the MTP loop over that continuation twice
        // (carry, then --mtp-reject-forward's path) with whole rows on every verify / single forward and every draft's
        // top-2, and the whole exported state at the end (trunk + MTP K / V, block keys, tails, streams). Two builds that
        // print the same hashes ran bit-identical arithmetic on these paths.
        if (a.num("hash-run", 0) != 0) {
            const int64_t D = a.num("hash-depth", 4096);
            STRIX_CHECK(D + gen_n + 64 < capacity, "strix_bench --hash-run: depth ", D, " + ", gen_n, " past the capacity");
            // A rank of N (stubbed) returns its vocabulary share's rows, lm_rows wide (world 1: the whole vocabulary):
            // rows and greedy picks over that share only, never past the row.
            const int64_t nv = std::min<int64_t>((int64_t)tok.size(), Dm.lm_rows);
            uint64_t hl = 0x9e3779b97f4a7c15ull, hd = hl, n_rows = 0, n_drafts = 0;
            auto mix = [](uint64_t &h, const void *p, size_t n) {
                const uint64_t x = strix_hash64(p, n);
                h = (h ^ x) * 0x100000001b3ull + (x >> 29);
            };
            auto rows = [&](const std::vector<float> &l) {
                mix(hl, l.data(), l.size() * 4);
                n_rows += (int64_t)(l.size() / (size_t)Dm.lm_rows);
            };
            auto best = [&](const float *l) { return (int32_t)argmax(l, nv); };
            ses.reset();
            for (int64_t p = 0; p < D;) {
                const int64_t end = std::min(D, p + chunk);
                if (end < D) ses.set_lookahead(std::vector<int32_t>(corpus.begin() + end, corpus.begin() + std::min(D, end + chunk)));
                const std::vector<float> l = ses.forward(std::vector<int32_t>(corpus.begin() + p, corpus.begin() + end), end == D ? 1 : 0);
                if (end == D) rows(l);
                p = end;
            }
            Qwen4ExpSnapshot hs = ses.make_snapshot();
            ses.save(hs);
            std::vector<int32_t> gen{corpus[(size_t)D]};
            for (int64_t i = 1; i < gen_n; ++i) {
                const std::vector<float> l = ses.forward({gen.back()}, 1);
                rows(l);
                gen.push_back(best(l.data()));
            }
            mix(hl, gen.data(), gen.size() * 4);
            int64_t verifies = 0, rollbacks_h = 0;
            for (int pass = 0; pass < 2 && model.has_mtp(); ++pass) {
                ses.restore(hs);
                for (int64_t i = 0; i < gen_n;) {
                    std::vector<int32_t> ids{gen[(size_t)i]};
                    const int64_t nsteps = std::min(mtp_draft, gen_n - 1 - i);
                    for (int64_t s = 0; s < nsteps; ++s) {
                        const Qwen4ExpSession::MtpTop2 t = ses.forward_mtp_top2(ids.back(), s);
                        uint32_t bv, sv;
                        std::memcpy(&bv, &t.best_v, 4), std::memcpy(&sv, &t.second_v, 4);
                        const uint32_t ww[4] = {(uint32_t)t.best, bv, sv, t.nan ? 1u : 0u};
                        mix(hd, ww, sizeof ww);
                        ++n_drafts;
                        if (t.best_v - t.second_v < margin) break;
                        ids.push_back(t.best);
                    }
                    const int64_t k = (int64_t)ids.size() - 1;
                    if (k == 0) {
                        rows(ses.forward(ids, 1));
                        i += 1;
                        continue;
                    }
                    rows(ses.forward_verify(ids, k + 1));
                    ++verifies;
                    int64_t j = 0;
                    while (j < k && ids[(size_t)j + 1] == gen[(size_t)(i + 1 + j)]) ++j;
                    if (j == k) {
                        ses.keep_verify();
                        i += k + 1;
                        continue;
                    }
                    ++rollbacks_h;
                    ses.keep_verify_prefix(j + 1);
                    const int64_t v = i + j + 1;
                    if (pass == 1) {
                        rows(ses.forward({gen[(size_t)v]}, 1));
                        i = v + 1;
                    } else {
                        i = v;
                    }
                }
            }
            Qwen4ExpSnapshot end = ses.make_snapshot();
            ses.save(end);
            std::vector<uint8_t> st(ses.state_bytes(end.pos()));
            ses.export_state(end, st.data(), st.size());
            uint64_t hst = strix_hash64(st.data(), st.size());
            auto hex = [](uint64_t h) {
                char b[24];
                std::snprintf(b, sizeof b, "%016llx", (unsigned long long)h);
                return std::string(b);
            };
            std::ofstream o(out);
            o << "{\"tool\":\"strix_bench\",\"mode\":\"hash-run\",\"weights\":\"" << weights << "\",\"tp_world\":" << tp_world
              << ",\"mtp\":" << (model.has_mtp() ? "true" : "false") << ",\"depth\":" << D << ",\"gen\":" << gen_n
              << ",\"mtp_draft\":" << mtp_draft << ",\"mtp_margin\":" << margin << ",\"mtp_vocab\":" << mtp_vocab
              << ",\"logits_rows\":" << n_rows << ",\"drafts\":" << n_drafts << ",\"verifies\":" << verifies
              << ",\"rollbacks\":" << rollbacks_h << ",\"state_pos\":" << end.pos() << ",\"state_bytes\":" << st.size()
              << ",\"logits_hash\":\"" << hex(hl) << "\",\"draft_hash\":\"" << hex(hd) << "\",\"state_hash\":\"" << hex(hst)
              << "\"" << equiv_json << ",\"canaries_ok\":" << (ses.canaries_ok() ? "true" : "false") << "}\n";
            std::fprintf(stderr, "strix_bench: hash-run: %lld rows, %lld drafts, logits %s drafts %s state %s\n",
                         (long long)n_rows, (long long)n_drafts, hex(hl).c_str(), hex(hd).c_str(), hex(hst).c_str());
            return 0;
        }

        // --row-invariance 1 (whole model): is a verify row's arithmetic a T = 1 forward's? Per depth, for
        // --ri-positions positions q (each from a snapshot of the state at q): T = 1 forwards of ids[q..q+6) (whole rows,
        // every layer's L<i>.out row hashed), then forward_verify(ids[q..q+T)) + drop for T = 2..6 with the same rows and
        // hashes, compared bit for bit; and one T = 1 forward at q with the decode linears' split-K x2 - the p99 of
        // |top-2 gap (x1) - top-2 gap (x2)| is the near-tie eps of the design's 3.1.
        if (a.num("row-invariance", 0) != 0) {
            STRIX_CHECK(whole, "strix_bench --row-invariance: whole model only");
            const int64_t npos = a.num("ri-positions", 64), nv = (int64_t)tok.size(), V = Dm.vocab;
            constexpr int kMaxT = 6;
            STRIX_CHECK((int64_t)corpus.size() > maxD + npos + kMaxT + 1, "strix_bench --row-invariance: corpus too short");
            struct RowHashes {
                std::vector<std::vector<uint64_t>> h;  // [row][layer]
                Qwen4ExpProbe probe() {
                    return [this](const std::string &name, const void *dev, int64_t rows, int64_t cols, ProbeType type) {
                        if (name.size() < 5 || name.compare(name.size() - 4, 4, ".out") != 0 || name[0] != 'L') return;
                        const size_t rb = (size_t)cols * (type == ProbeType::BF16 ? 2 : 4);
                        std::vector<uint8_t> host(rb * (size_t)rows);
                        STRIX_HIP_CHECK(hipMemcpy(host.data(), dev, host.size(), hipMemcpyDeviceToHost), "row hash probe");
                        if (h.size() < (size_t)rows) h.resize((size_t)rows);
                        for (int64_t r = 0; r < rows; ++r) h[(size_t)r].push_back(strix_hash64(host.data() + r * rb, rb));
                    };
                }
            };
            auto gap = [&](const float *l) {
                float b1 = -INFINITY, b2 = -INFINITY;
                for (int64_t i = 0; i < nv; ++i)
                    if (l[i] > b1) b2 = b1, b1 = l[i];
                    else if (l[i] > b2) b2 = l[i];
                return (double)b1 - (double)b2;
            };
            struct PerT {
                int64_t rows = 0, logits_equal = 0, hashes_equal = 0, top1_equal = 0;
                double max_abs = 0;
                std::vector<double> dgap;
            };
            std::map<int, PerT> perT;
            std::vector<double> eps_samples;
            int64_t positions = 0;
            std::ostringstream dj;
            Qwen4ExpSnapshot rs = ses.make_snapshot();
            int64_t pos = 0;
            const double r0 = now_ms();
            for (const int64_t D : depths) {
                while (pos < D) {
                    const int64_t end = std::min(D, pos + chunk);
                    if (end < D) ses.set_lookahead(std::vector<int32_t>(corpus.begin() + end, corpus.begin() + std::min(D, end + chunk)));
                    ses.forward(std::vector<int32_t>(corpus.begin() + pos, corpus.begin() + end), 0);
                    pos = end;
                }
                std::map<int, PerT> here;
                for (int64_t q = D; q < D + npos; ++q) {
                    ses.save(rs);  // the state at q
                    std::vector<std::vector<float>> one;
                    RowHashes oneh;
                    for (int t = 0; t < kMaxT; ++t) {
                        RowHashes h1;
                        one.push_back(ses.forward({corpus[(size_t)(q + t)]}, 1, h1.probe()));
                        oneh.h.push_back(h1.h.at(0));
                    }
                    ses.restore(rs);
                    kernels::set_split_k_scale(2);
                    const std::vector<float> two = ses.forward({corpus[(size_t)q]}, 1);
                    kernels::set_split_k_scale(1);
                    eps_samples.push_back(std::fabs(gap(one[0].data()) - gap(two.data())));
                    for (int T = 2; T <= kMaxT; ++T) {
                        ses.restore(rs);
                        RowHashes vh;
                        const std::vector<float> v =
                            ses.forward_verify(std::vector<int32_t>(corpus.begin() + q, corpus.begin() + q + T), T, vh.probe());
                        ses.drop_verify();
                        for (PerT *pt : {&perT[T], &here[T]})
                            for (int r = 0; r < T; ++r) {
                                const float *a1 = one[(size_t)r].data(), *b1 = v.data() + (size_t)r * V;
                                ++pt->rows;
                                pt->logits_equal += std::memcmp(a1, b1, (size_t)V * 4) == 0;
                                pt->hashes_equal += vh.h.at((size_t)r) == oneh.h[(size_t)r];
                                pt->top1_equal += argmax(a1, nv) == argmax(b1, nv);
                                double m = 0;
                                for (int64_t i = 0; i < nv; ++i) m = std::max(m, (double)std::fabs(a1[i] - b1[i]));
                                pt->max_abs = std::max(pt->max_abs, m);
                                pt->dgap.push_back(std::fabs(gap(a1) - gap(b1)));
                            }
                    }
                    ses.restore(rs);
                    ses.forward({corpus[(size_t)q]}, 0);  // on to q + 1
                    ++positions;
                }
                pos = D + npos;  // the context runs on from here for the next depth
                dj << (dj.tellp() > 0 ? "," : "") << "\"" << D << "\":{";
                bool f = true;
                for (auto &[T, p] : here)
                    dj << (f ? "" : ",") << "\"" << T << "\":{\"rows\":" << p.rows << ",\"logits_bitexact\":" << p.logits_equal
                       << ",\"out_hashes_equal\":" << p.hashes_equal << "}",
                        f = false;
                dj << "}";
                std::fprintf(stderr, "strix_bench: row-invariance: depth %lld done (%.1f s)\n", (long long)D, (now_ms() - r0) / 1000);
            }
            std::sort(eps_samples.begin(), eps_samples.end());
            auto pct = [](const std::vector<double> &v, double f) {
                return v.empty() ? 0.0 : v[(size_t)std::min<double>((double)v.size() - 1, std::floor(f * (double)(v.size() - 1) + 0.5))];
            };
            const double eps = pct(eps_samples, 0.99);
            std::ofstream o(out);
            o << "{\"tool\":\"strix_bench\",\"mode\":\"row-invariance\",\"weights\":\"" << weights << "\",\"positions\":" << positions
              << ",\"positions_per_depth\":" << npos << ",\"depths\":[";
            for (size_t i = 0; i < depths.size(); ++i) o << (i ? "," : "") << depths[i];
            o << "],\"compare\":\"forward_verify(ids[q..q+T)) rows r vs T = 1 forwards of ids[q+r] from the same state: "
                 "whole logits rows bit for bit, every layer's L<i>.out row hash\",\"bitexact_by_T\":{";
            bool f = true, all_exact = true;
            for (auto &[T, p] : perT) {
                o << (f ? "" : ",") << "\"" << T << "\":" << (p.logits_equal == p.rows && p.hashes_equal == p.rows ? "true" : "false");
                f = false;
            }
            o << "},\"by_T\":{";
            f = true;
            for (auto &[T, p] : perT) {
                std::sort(p.dgap.begin(), p.dgap.end());
                all_exact = all_exact && p.logits_equal == p.rows && p.hashes_equal == p.rows;
                o << (f ? "" : ",") << "\"" << T << "\":{\"rows\":" << p.rows << ",\"logits_bitexact\":" << p.logits_equal
                  << ",\"out_hashes_equal\":" << p.hashes_equal << ",\"top1_equal\":" << p.top1_equal << ",\"max_abs_logit_diff\":"
                  << p.max_abs << ",\"dgap_p50\":" << pct(p.dgap, 0.5) << ",\"dgap_p99\":" << pct(p.dgap, 0.99)
                  << ",\"dgap_max\":" << (p.dgap.empty() ? 0.0 : p.dgap.back()) << "}";
                f = false;
            }
            o << "},\"by_depth\":{" << dj.str() << "},\"all_bitexact\":" << (all_exact ? "true" : "false")
              << ",\"eps_top2_gap_p99\":" << eps << ",\"eps_top2_gap_p50\":" << pct(eps_samples, 0.5)
              << ",\"eps_top2_gap_max\":" << (eps_samples.empty() ? 0.0 : eps_samples.back())
              << ",\"eps_settings\":\"|top-2 gap| of T = 1 rows, decode linears split-K x1 vs x2, same positions\""
              << ",\"seconds\":" << (now_ms() - r0) / 1000 << "}\n";
            std::fprintf(stderr, "strix_bench: row-invariance: %lld positions, all bit-exact %s, eps p99 %.5g\n",
                         (long long)positions, all_exact ? "yes" : "no", eps);
            return 0;
        }

        std::ostringstream J;
        J << "{\"tool\":\"strix_bench\",\"weights\":\"" << weights << "\",\"ngram\":\"" << ngram << "\",\"tp_world\":" << tp_world
          << ",\"tp_rank\":" << tp_rank << ",\"weights_gib\":" << (double)model.weights().data_bytes() / (1ull << 30)
          << ",\"load_s\":" << load_s << ",\"capacity\":" << capacity << ",\"chunk\":" << chunk << ",\"yarn\":" << yarn
          << ",\"mtp\":" << (model.has_mtp() ? "true" : "false") << ",\"mtp_draft\":" << mtp_draft << ",\"mtp_margin\":" << margin
          << ",\"mtp_reject_forward\":" << (reject_forward ? "true" : "false")
          << ",\"mtp_draft_q4\":" << (draft_q4 ? "true" : "false")
          << ",\"mtp_vocab\":" << mtp_vocab << ",\"ngram_cache_rows\":" << cache_rows
          << ",\"gpu_mem_free_gib_after_session\":" << (double)free_b / (1ull << 30) << ",\"depths\":{";

        Qwen4ExpSnapshot snap = ses.make_snapshot();
        int64_t pos = 0;
        double prefill_ms_total = 0;
        std::map<int64_t, double> t1_ms, ar_tps, mtp_tps;
        bool first_depth = true;
        for (const int64_t D : depths) {
            // grow the context to D (chunked, the engine's lookahead hint for the next chunk)
            double tp0 = now_ms();
            const int64_t from = pos;
            while (pos < D) {
                const int64_t end = std::min(D, pos + chunk);
                const std::vector<int32_t> ids(corpus.begin() + pos, corpus.begin() + end);
                const int64_t nend = std::min(D, end + chunk);
                if (end < D) ses.set_lookahead(std::vector<int32_t>(corpus.begin() + end, corpus.begin() + nend));
                ses.forward(ids, end == D ? 1 : 0);
                pos = end;
            }
            const double pf_ms = now_ms() - tp0;
            prefill_ms_total += pf_ms;
            ses.save(snap);
            std::fprintf(stderr, "strix_bench: depth %lld: prefill %lld..%lld in %.1f s (%.0f t/s; cumulative %.0f t/s)\n",
                         (long long)D, (long long)from, (long long)D, pf_ms / 1000, (double)(D - from) / pf_ms * 1000,
                         (double)D / prefill_ms_total * 1000);
            const int64_t ex0 = ses.exchanges();
            // (1) AR, teacher-forced on the corpus continuation, MTP catch-up off.
            ses.set_mtp_catchup(false);
            std::vector<double> tf;
            for (int64_t i = 0; i < ar_steps; ++i) {
                const double s0 = now_ms();
                greedy({corpus[(size_t)(D + i)]});
                tf.push_back(now_ms() - s0);
            }
            const int64_t ex_per_fwd = (ses.exchanges() - ex0) / std::max<int64_t>(1, ar_steps);
            // verify (k = 2..6) with catch-up off
            std::map<int, std::vector<double>> ver_off, ver_on;
            auto run_verify = [&](std::map<int, std::vector<double>> &into) {
                for (int k = 2; k <= 6; ++k)
                    for (int64_t rep = 0; rep < verify_reps; ++rep) {
                        ses.restore(snap);
                        std::vector<int32_t> ids(corpus.begin() + D, corpus.begin() + D + k);
                        ses.want_candidates(n_valid);
                        const double s0 = now_ms();
                        ses.forward_verify(ids, k);
                        into[k].push_back(now_ms() - s0);
                        ses.drop_verify();
                    }
            };
            run_verify(ver_off);
            ses.restore(snap);
            ses.set_mtp_catchup(true);
            std::ostringstream dj;
            const Stats tfs = stats(tf);
            t1_ms[D] = tfs.p50;
            ar_tps[D] = 1000.0 / tfs.mean;
            dj << "\"" << D << "\":{\"prefill_s\":" << pf_ms / 1000 << ",\"prefill_from\":" << from
               << ",\"prefill_tps_cumulative\":" << (double)D / prefill_ms_total * 1000 << ",\"ar_tf_ms\":" << js(tfs)
               << ",\"exchanges_per_forward\":" << ex_per_fwd;
            dj << ",\"verify_ms_catchup_off\":{";
            for (int k = 2; k <= 6; ++k) dj << (k > 2 ? "," : "") << "\"" << k << "\":" << js(stats(ver_off[k]));
            dj << "}";
            if (model.has_mtp()) {
                run_verify(ver_on);
                dj << ",\"verify_ms\":{";
                for (int k = 2; k <= 6; ++k) dj << (k > 2 ? "," : "") << "\"" << k << "\":" << js(stats(ver_on[k]));
                dj << "}";
                // (2) greedy generation, catch-up on: the capture for the MTP replay
                ses.restore(snap);
                std::vector<int32_t> gen;
                std::vector<double> gm;
                // The capture starts with the corpus token at D (teacher-forced first token), then greedy.
                int32_t cur = corpus[(size_t)D];
                gen.push_back(cur);
                for (int64_t i = 1; i < gen_n; ++i) {
                    const double s0 = now_ms();
                    cur = greedy({cur});
                    gm.push_back(now_ms() - s0);
                    gen.push_back(cur);
                }
                const Stats gms = stats(gm);
                // (3) drafts: chained forward_mtp_top2 from D (state at D: gen[0] not yet forwarded)
                std::vector<std::vector<double>> dr((size_t)mtp_draft);
                for (int64_t rep = 0; rep < draft_reps; ++rep) {
                    ses.restore(snap);
                    greedy({gen[0]});  // the trunk forward of gen[0]; the head then drafts gen[2..]
                    int32_t t = gen[1];
                    for (int64_t s = 0; s < mtp_draft; ++s) {
                        const double s0 = now_ms();
                        const Qwen4ExpSession::MtpTop2 p = ses.forward_mtp_top2(t, s);
                        dr[(size_t)s].push_back(now_ms() - s0);
                        t = p.best;
                    }
                }
                // (4) the engine's MTP loop over gen (serve/replay.cpp), teacher-forced, timed
                ses.restore(snap);
                int64_t forwards = 0, drafted = 0, accepted = 0, rollbacks = 0, steps = 0;
                std::vector<double> step_ms, draft_ms, verify_ms, single_ms;
                std::map<int64_t, int64_t> acc_hist;
                const double r0 = now_ms();
                for (int64_t i = 0; i < gen_n;) {
                    const double st0 = now_ms();
                    const int32_t id = gen[(size_t)i];
                    std::vector<int32_t> ids{id};
                    const int64_t nsteps = std::min(mtp_draft, gen_n - 1 - i);
                    if (nsteps > 0) ses.prefetch_ple(ids, 0);
                    for (int64_t s = 0; s < nsteps; ++s) {
                        const double d0 = now_ms();
                        const Qwen4ExpSession::MtpTop2 t = ses.forward_mtp_top2(ids.back(), s);
                        draft_ms.push_back(now_ms() - d0);
                        if (t.best_v - t.second_v < margin) break;
                        ids.push_back(t.best);
                        if (s + 1 < nsteps) ses.prefetch_ple(ids, (int64_t)ids.size() - 1);
                    }
                    const int64_t k = (int64_t)ids.size() - 1;
                    ++forwards, ++steps;
                    if (k == 0) {
                        const double f0 = now_ms();
                        greedy({id});
                        single_ms.push_back(now_ms() - f0);
                        i += 1;
                        ++acc_hist[0];
                        step_ms.push_back(now_ms() - st0);
                        continue;
                    }
                    drafted += k;
                    ses.want_candidates(n_valid);
                    const double v0 = now_ms();
                    ses.forward_verify(ids, k + 1);
                    verify_ms.push_back(now_ms() - v0);
                    int64_t j = 0;
                    while (j < k && ids[(size_t)j + 1] == gen[(size_t)(i + 1 + j)]) ++j;
                    accepted += j;
                    ++acc_hist[j];
                    if (j == k) {
                        ses.keep_verify();
                        i += k + 1;
                        step_ms.push_back(now_ms() - st0);
                        continue;
                    }
                    ++rollbacks;
                    ses.keep_verify_prefix(j + 1);
                    const int64_t v = i + j + 1;
                    if (reject_forward) {  // the old path: forward the rejected position's token alone
                        ses.prefetch_ple({gen[(size_t)v]}, 0);
                        const double f0 = now_ms();
                        greedy({gen[(size_t)v]});
                        single_ms.push_back(now_ms() - f0);
                        ++forwards;
                        i = v + 1;
                    } else {
                        i = v;  // carry: gen[v] is the next step's first token (known, not yet forwarded)
                    }
                    step_ms.push_back(now_ms() - st0);
                }
                const double dec_ms = now_ms() - r0;
                mtp_tps[D] = (double)gen_n / dec_ms * 1000;
                dj << ",\"ar_gen_ms\":" << js(gms) << ",\"ar_gen_tps\":" << 1000.0 / gms.mean << ",\"draft_ms\":{";
                for (int64_t s = 0; s < mtp_draft; ++s) dj << (s ? "," : "") << "\"" << s << "\":" << js(stats(dr[(size_t)s]));
                dj << "},\"mtp\":{\"gen_n\":" << gen_n << ",\"decode_ms\":" << dec_ms << ",\"tps\":" << mtp_tps[D]
                   << ",\"steps\":" << steps << ",\"forwards\":" << forwards << ",\"drafted\":" << drafted
                   << ",\"accepted\":" << accepted << ",\"rollbacks\":" << rollbacks
                   << ",\"tokens_per_step\":" << (double)gen_n / (double)steps << ",\"step_ms\":" << js(stats(step_ms))
                   << ",\"draft_call_ms\":" << js(stats(draft_ms)) << ",\"verify_call_ms\":" << js(stats(verify_ms))
                   << ",\"single_forward_ms\":" << js(stats(single_ms)) << ",\"accepted_hist\":{";
                bool f = true;
                for (auto [k, n] : acc_hist) dj << (f ? "" : ",") << "\"" << k << "\":" << n, f = false;
                dj << "}}";
                // greedy continuation text (first 48 tokens) for the record
                std::string txt = tok.decode(std::vector<int32_t>(gen.begin(), gen.begin() + std::min<int64_t>(48, gen_n)));
                std::string esc;
                for (unsigned char c : txt)
                    if (c == '"' || c == '\\') esc += '\\', esc += (char)c;
                    else if (c < 0x20) esc += ' ';
                    else esc += (char)c;
                dj << ",\"gen_text\":\"" << esc << "\"";
                // (5) needles at depth >= 64k
                if (!needles.empty() && D >= 65536) {
                    int found = 0;
                    dj << ",\"needles\":[";
                    for (size_t ni = 0; ni < needles.size(); ++ni) {
                        ses.restore(snap);
                        const std::vector<int32_t> q = tok.encode(
                            "\n\nQuestion: What is the secret passcode of the " + needles[ni].key +
                            "?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe secret passcode of the " +
                            needles[ni].key + " is");
                        int32_t c = greedy(q);
                        std::vector<int32_t> ans{c};
                        for (int s = 0; s < 6; ++s) ans.push_back(c = greedy({c}));
                        const std::string at = tok.decode(ans);
                        const bool ok = at.find(needles[ni].code) != std::string::npos;
                        found += ok;
                        std::string e2;
                        for (unsigned char ch : at)
                            if (ch == '"' || ch == '\\') e2 += '\\', e2 += (char)ch;
                            else if (ch < 0x20) e2 += ' ';
                            else e2 += (char)ch;
                        dj << (ni ? "," : "") << "{\"key\":\"" << needles[ni].key << "\",\"code\":\"" << needles[ni].code
                           << "\",\"at\":" << needles[ni].at << ",\"answer\":\"" << e2 << "\",\"ids\":[";
                        for (size_t z = 0; z < ans.size(); ++z) dj << (z ? "," : "") << ans[z];
                        dj << "],\"ok\":" << (ok ? "true" : "false") << "}";
                    }
                    dj << "],\"needles_found\":" << found << ",\"needles_n\":" << needles.size();
                }
            }
            const Qwen4ExpSession::PleStats ps = ses.ple_stats();
            dj << ",\"ple_waits\":" << ps.waits << ",\"ple_wait_s\":" << ps.wait_seconds << ",\"ple_wait_max_s\":" << ps.wait_max_seconds
               << "}";
            J << (first_depth ? "" : ",") << dj.str();
            first_depth = false;
            std::fprintf(stderr, "strix_bench: depth %lld: AR %.2f ms (p50) %.1f t/s%s\n", (long long)D, tfs.p50, ar_tps[D],
                         model.has_mtp() ? (", MTP " + std::to_string(mtp_tps[D]) + " t/s").c_str() : "");
            ses.restore(snap);
        }
        J << "}";
        auto dict = [&](const std::map<int64_t, double> &m) {
            std::ostringstream o;
            o << "{";
            bool f = true;
            for (auto [k, v] : m) o << (f ? "" : ",") << "\"" << k << "\":" << v, f = false;
            o << "}";
            return o.str();
        };
        J << ",\"t1_ms\":" << dict(t1_ms) << ",\"ar_tps\":" << dict(ar_tps);
        if (model.has_mtp()) J << ",\"mtp_tps\":" << dict(mtp_tps);

        // Golden logits + the noise floor: WikiText-2 test (the last corpus file is not assumed: --kl-corpus).
        if (do_quality) {
            const std::string klc = a.get("kl-corpus", corpus_files.front());
            const std::vector<int32_t> kt = tok.encode(read_file(klc));
            STRIX_CHECK((int64_t)kt.size() > kl_prefix + kl_n + 1, "strix_bench: kl corpus too short");
            const int64_t V = Dm.vocab;
            auto run = [&](int64_t pchunk, std::vector<float> &rows) {
                ses.reset();
                ses.set_mtp_catchup(true);
                for (int64_t p = 0; p < kl_prefix; p += pchunk)
                    ses.forward(std::vector<int32_t>(kt.begin() + p, kt.begin() + std::min(kl_prefix, p + pchunk)), 0);
                rows.resize((size_t)(kl_n * V));
                for (int64_t i = 0; i < kl_n; ++i) {
                    std::vector<float> l = ses.forward({kt[(size_t)(kl_prefix + i)]}, 1);
                    std::memcpy(rows.data() + (size_t)(i * V), l.data(), (size_t)V * 4);
                }
            };
            std::vector<float> A, B;
            const int64_t c2 = a.num("kl-chunk-b", 4096);
            STRIX_CHECK(chunk >= c2 && kl_prefix % chunk == 0 && kl_prefix % c2 == 0, "strix_bench: kl chunks");
            run(chunk, A);
            run(c2, B);
            std::vector<float> C;
            const int ks = (int)a.num("noise-split-k", 2);
            kernels::set_split_k_scale(ks);
            run(chunk, C);
            kernels::set_split_k_scale(1);
            const int64_t nv = (int64_t)tok.size();
            double kl_chunk = 0;
            for (int64_t i = 0; i < kl_n; ++i)
                kl_chunk += kl_rows(A.data() + (size_t)(i * V), B.data() + (size_t)(i * V), nv);
            kl_chunk /= (double)kl_n;
            double kl = 0, nll = 0;
            int64_t agree = 0;
            std::vector<double> kls;
            for (int64_t i = 0; i < kl_n; ++i) {
                const float *pa = A.data() + (size_t)(i * V), *pb = C.data() + (size_t)(i * V);
                const double k = kl_rows(pa, pb, nv);
                kls.push_back(k);
                kl += k;
                agree += argmax(pa, nv) == argmax(pb, nv);
                const int32_t nxt = kt[(size_t)(kl_prefix + i + 1)];
                nll += lse(pa, nv) - (double)pa[nxt];
            }
            kl /= (double)kl_n, nll /= (double)kl_n;
            J << ",\"noise_floor_kl\":" << kl << ",\"noise_floor\":{\"settings\":\"decode linears split-K x" << ks
              << " (summation order only) vs as tuned; teacher-forced T=1 decode after a " << kl_prefix << "-token prefill\""
              << ",\"kl_prefill_chunk_" << chunk << "_vs_" << c2 << "\":" << kl_chunk << ",\"kl_mean\":" << kl << ",\"kl\":" << js(stats(kls)) << ",\"top1_agree\":" << (double)agree / (double)kl_n
              << ",\"prefix\":" << kl_prefix << ",\"positions\":" << kl_n << ",\"nll\":" << nll << ",\"ppl\":" << std::exp(nll)
              << ",\"corpus\":\"" << klc << "\"}";
            if (!golden_dir.empty()) {
                const std::string gp = golden_dir + "/golden-wt2test-p" + std::to_string(kl_prefix) + "-n" + std::to_string(kl_n);
                std::ofstream g(gp + ".f32", std::ios::binary);
                g.write(reinterpret_cast<const char *>(A.data()), (std::streamsize)(A.size() * 4));
                std::ofstream gi(gp + ".json");
                gi << "{\"rows\":" << kl_n << ",\"vocab\":" << V << ",\"n_valid\":" << nv << ",\"prefix\":" << kl_prefix
                   << ",\"chunk\":" << chunk << ",\"dtype\":\"f32\",\"corpus\":\"" << klc << "\",\"weights\":\"" << weights
                   << "\",\"tokens\":[";
                for (int64_t i = 0; i <= kl_prefix + kl_n; ++i) gi << (i ? "," : "") << kt[(size_t)i];
                gi << "]}\n";
                J << ",\"golden\":\"" << gp << ".f32\"";
            }
        }
        J << equiv_json << ",\"canaries_ok\":" << (ses.canaries_ok() ? "true" : "false") << ",\"exchanges_total\":" << ses.exchanges() << "}\n";
        std::ofstream o(out);
        o << J.str();
        std::fprintf(stderr, "strix_bench: wrote %s\n", out.c_str());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "strix_bench: fatal: %s\n", e.what());
        return 1;
    }
}
