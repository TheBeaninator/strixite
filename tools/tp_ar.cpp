// tp_ar: the TP-N autoregressive forward checked and timed - and the same measurements on the whole
// model (world 1), so the reference and the TP run come from one code path (runtime/tp_mirror.hpp TpDriver).
//
//   whole model:  tp_ar --weights W --ngram T --tokenizer J --corpus train.raw --kl-corpus test.raw
//                       --golden-out FILE|- [--noise-split-k 2] --out r.json
//   TP rank 0:    tp_ar ... --tp-world 2 --tp-rank 0 --tp-peers 192.0.2.1,192.0.2.2 --golden-in FILE --out r.json
//   TP rank r>0:  tp_ar ... --tp-world 2 --tp-rank r --tp-peers ...      (executes rank 0's calls, then exits)
//
// Quality (rank 0 / world 1): prefill --kl-prefix tokens of the KL corpus (one chunk of --chunk), then --kl-n
// teacher-forced T = 1 forwards, each row's whole logits (every rank's vocabulary share gathered). Per row: the NLL of
// the next corpus token; world 1 writes the rows (FP32 [kl-n, vocab]) to --golden-out; with --golden-in, KL(golden ||
// this), top-1 agreement and the golden's own NLL. --hash 1: every forward of the quality pass compares the replicated
// activations' hashes across ranks (TpDriver::set_hash_check: X after the embedding, the PLE and every layer; the
// reduced mixer / MoE outputs; the final mix). --cand-check K: K more positions, each run once with GPU candidates and
// once with whole rows (restored between): the merged candidates must equal the whole rows' top kLogitCands exactly.
// --noise-split-k S (world 1): the quality pass again with the decode linears' split-K x S (summation order only),
// KL / top-1 against the first pass - the noise floor on the same positions.
// Speed: per --depths D, the context (--corpus) grown to D by chunked prefill (lookahead hints mirrored), a snapshot,
// then --ar-steps T = 1 forwards teacher-forced with GPU candidates (the engine's decode path); p50 / mean ms.
//
// MTP (--mtp 1: rank 0's session keeps the draft head, the executors' don't): per depth, after AR,
//   verify_k  (--verify-k 1) forward_verify(k) + drop_verify for k = 2..6, mirrored, timed
//   ar_gen    --gen greedy T = 1 forwards (candidates): the continuation the MTP loop replays
//   q4 head   --mtp-draft-q4 on|off: the draft scores a Q4 copy of the draft vocabulary rows (made once at load,
//             --mtp-draft-q4-group 64) instead of the loaded Q8 head; --mtp-free-q4 1: per depth also the free-running
//             greedy MTP decode with the Q8 head and with the Q4 head, compared token by token ("free_q4").
//   sweep     --mtp-sweep 1.0x3,2.0x5,2.0x5q4,..: per depth, the MTP loop over the same ar_gen for each cell (margin x
//             most drafts, "q4" = the Q4 draft head), --sweep-reps passes (odd passes in reverse cell order), "sweep".
//   mtp       the engine's draft / verify / keep loop over ar_gen teacher-forced (strix_bench's, on TpDriver: carry, or
//             --mtp-reject-forward 1): steps, forwards, drafted, accepted, rollbacks, draft / verify / single ms, t/s,
//             exchanges per token; with --hash 1 first one pass with every forward's / verify's replicated activations
//             and every keep / drop's replicated state compared across ranks ("verify_hash"), then the timed pass.
// Rank 1's PLE stats come back at the end (kStats): ple_wait_s_rank1.
//
// --shadow 1 (world 1): the mirror-shadow test - one process, one weight copy, two
// sessions (the driver's with MTP, an executor's without) joined by an in-process control channel (TpLoopback), the
// executor on a thread. Both are the whole model, so after every mirrored forward / verify their logits rows and
// replicated activations must be bit-identical, and after every keep / drop / restore / reset their replicated
// state. Drives the real MTP loop over --shadow-tokens generated tokens (carry and reject-forward rounds alternating)
// with injected drops, mid-run snapshot restores and rank-0 failures followed by reset().

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "formats/strixw.hpp"  // strix_hash64
#include "kernels/linear_q8.hpp"
#include "kernels/logits_topk.hpp"
#include "runtime/qwen4exp.hpp"
#include "runtime/tp_comm.hpp"
#include "runtime/tp_mirror.hpp"
#include "serve/qwen4exp_backend.hpp"
#include "serve/replay.hpp"
#include "serve/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <chrono>
#include <random>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
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
    double p50 = 0, mean = 0, p10 = 0, p90 = 0, p99 = 0, min = 0, max = 0;
    int64_t n = 0;
};
Stats stats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n = (int64_t)v.size();
    auto q = [&](double f) { return v[(size_t)std::min<double>((double)v.size() - 1, std::floor(f * (double)(v.size() - 1) + 0.5))]; };
    s.p50 = q(0.5), s.p10 = q(0.1), s.p90 = q(0.9), s.p99 = q(0.99), s.min = v.front(), s.max = v.back();
    for (double x : v) s.mean += x;
    s.mean /= (double)v.size();
    return s;
}
std::string js(const Stats &s) {
    char b[320];
    std::snprintf(b, sizeof b,
                  "{\"p50\":%.6g,\"mean\":%.6g,\"p10\":%.6g,\"p90\":%.6g,\"p99\":%.6g,\"min\":%.6g,\"max\":%.6g,\"n\":%lld}", s.p50,
                  s.mean, s.p10, s.p90, s.p99, s.min, s.max, (long long)s.n);
    return b;
}
std::string esc(const std::string &x) {
    std::string o;
    for (unsigned char c : x)
        if (c == '"' || c == '\\') o += '\\', o += (char)c;
        else if (c < 0x20) o += ' ';
        else o += (char)c;
    return o;
}

std::string read_file(const std::string &p) {
    std::ifstream f(p, std::ios::binary);
    STRIX_CHECK(f.good(), "tp_ar: cannot read ", p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

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

// The engine's MTP loop (serve/replay.cpp, strix_bench's) over gen teacher-forced, on the driver: drafts on rank 0,
// the trunk's forwards / verifies / keeps mirrored. The state before: gen[0] known, not yet forwarded.
struct MtpRun {
    int64_t steps = 0, forwards = 0, drafted = 0, accepted = 0, rollbacks = 0, kept_all = 0, draft_calls = 0;
    double decode_ms = 0;
    std::vector<double> step_ms, draft_ms, verify_ms, single_ms;
    std::map<int64_t, int64_t> acc_hist;
};
MtpRun mtp_loop(TpDriver &drv, const std::vector<int32_t> &gen, int64_t n_valid, int64_t draft, double margin,
                bool reject_forward) {
    MtpRun r;
    const int64_t G = (int64_t)gen.size();
    const double r0 = now_ms();
    for (int64_t i = 0; i < G;) {
        const double st0 = now_ms();
        const int32_t id = gen[(size_t)i];
        std::vector<int32_t> ids{id};
        const int64_t nsteps = std::min(draft, G - 1 - i);
        if (nsteps > 0) drv.prefetch_ple(ids, 0);
        for (int64_t s = 0; s < nsteps; ++s) {
            const double d0 = now_ms();
            const Qwen4ExpSession::MtpTop2 t = drv.forward_mtp_top2(ids.back(), s);
            r.draft_ms.push_back(now_ms() - d0);
            ++r.draft_calls;
            STRIX_CHECK(!t.nan, "tp_ar: NaN in an MTP draft at token ", i, " step ", s);
            if (t.best_v - t.second_v < margin) break;
            ids.push_back(t.best);
            if (s + 1 < nsteps) drv.prefetch_ple(ids, (int64_t)ids.size() - 1);
        }
        const int64_t k = (int64_t)ids.size() - 1;
        ++r.forwards, ++r.steps;
        if (k == 0) {
            drv.want_candidates(n_valid);
            const double f0 = now_ms();
            drv.forward({id}, 1);
            r.single_ms.push_back(now_ms() - f0);
            i += 1;
            ++r.acc_hist[0];
            r.step_ms.push_back(now_ms() - st0);
            continue;
        }
        r.drafted += k;
        drv.want_candidates(n_valid);
        const double v0 = now_ms();
        drv.forward_verify(ids, k + 1);
        r.verify_ms.push_back(now_ms() - v0);
        int64_t j = 0;
        while (j < k && ids[(size_t)j + 1] == gen[(size_t)(i + 1 + j)]) ++j;
        r.accepted += j;
        ++r.acc_hist[j];
        if (j == k) {
            drv.keep_verify();
            ++r.kept_all;
            i += k + 1;
            r.step_ms.push_back(now_ms() - st0);
            continue;
        }
        ++r.rollbacks;
        drv.keep_verify_prefix(j + 1);
        const int64_t v = i + j + 1;
        if (reject_forward) {  // the old path: forward the rejected position's token alone
            drv.prefetch_ple({gen[(size_t)v]}, 0);
            drv.want_candidates(n_valid);
            const double f0 = now_ms();
            drv.forward({gen[(size_t)v]}, 1);
            r.single_ms.push_back(now_ms() - f0);
            ++r.forwards;
            i = v + 1;
        } else {
            i = v;  // carry: gen[v] is the next step's first token (known, not yet forwarded)
        }
        r.step_ms.push_back(now_ms() - st0);
    }
    r.decode_ms = now_ms() - r0;
    return r;
}

// Per-row hashes of a forward's L<i>.out rows (every layer, combined): rank 0's local probe (TpDriver::set_probe) -
// the cross-run row-hash check.
struct RowHashes {
    std::vector<uint64_t> h;  // [rows] of the last forward
    std::vector<uint8_t> host;
    Qwen4ExpProbe probe() {
        return [this](const std::string &name, const void *dev, int64_t rows, int64_t cols, ProbeType type) {
            if (rows > Qwen4ExpSession::kMaxLogits) return;  // prefill chunks: not compared
            if (name.size() < 5 || name[0] != 'L' || name.compare(name.size() - 4, 4, ".out") != 0) return;
            if (name == "L0.out") h.assign((size_t)rows, 0x9e3779b97f4a7c15ull);
            const size_t rb = (size_t)cols * (type == ProbeType::BF16 ? 2 : 4);
            host.resize(rb * (size_t)rows);
            STRIX_HIP_CHECK(hipMemcpy(host.data(), dev, host.size(), hipMemcpyDeviceToHost), "row hash probe");
            for (int64_t r = 0; r < rows && (size_t)r < h.size(); ++r) {
                const uint64_t x = strix_hash64(host.data() + (size_t)r * rb, rb);
                h[(size_t)r] = (h[(size_t)r] ^ x) * 0x100000001b3ull + (x >> 29);
            }
        };
    }
};

// "on" / "off" / 1 / 0 (the Q4 draft-head switches).
bool onoff_arg(const std::string &v, const char *what) {
    if (v == "on" || v == "1" || v == "true") return true;
    if (v == "off" || v == "0" || v == "false" || v.empty()) return false;
    STRIX_FAIL(what, ": '", v, "', expected on or off");
}

double cand_gap(const Qwen4ExpSession::Candidates &c, int64_t row) {
    const kernels::LogitCand *x = c.cand.data() + row * kernels::kLogitCands;
    return (double)x[0].v - (double)x[1].v;
}

// Free-running greedy decode with MTP, as the engine runs it at temperature 0 (drafts accepted while they equal the
// verify rows' greedy picks; carry: the first rejected row's pick is the next step's known token). From the state
// before `first` (known, not yet forwarded): n tokens (first included) and, per token t >= 1, the top-2 gap of the
// row that chose it.
struct FreeRun {
    std::vector<int32_t> seq;
    std::vector<double> gap;
};
FreeRun mtp_free(TpDriver &drv, int32_t first, int64_t n, int64_t n_valid, int64_t draft, double margin) {
    FreeRun f;
    f.seq.push_back(first), f.gap.push_back(0);
    while ((int64_t)f.seq.size() < n) {
        std::vector<int32_t> ids{f.seq.back()};
        const int64_t left = n - (int64_t)f.seq.size();
        const int64_t nsteps = std::min(draft, left - 1);
        if (nsteps > 0) drv.prefetch_ple(ids, 0);
        for (int64_t s = 0; s < nsteps; ++s) {
            const Qwen4ExpSession::MtpTop2 t = drv.forward_mtp_top2(ids.back(), s);
            if (t.nan || t.best_v - t.second_v < margin) break;
            ids.push_back(t.best);
            if (s + 1 < nsteps) drv.prefetch_ple(ids, (int64_t)ids.size() - 1);
        }
        const int64_t k = (int64_t)ids.size() - 1;
        drv.want_candidates(n_valid);
        if (k == 0) {
            drv.forward(ids, 1);
            const Qwen4ExpSession::Candidates &c = drv.candidates();
            f.seq.push_back(c.cand[0].id), f.gap.push_back(cand_gap(c, 0));
            continue;
        }
        drv.forward_verify(ids, k + 1);
        const Qwen4ExpSession::Candidates c = drv.candidates();
        int64_t j = 0;
        while (j < k && ids[(size_t)j + 1] == c.cand[(size_t)(j * kernels::kLogitCands)].id) {
            f.seq.push_back(ids[(size_t)j + 1]), f.gap.push_back(cand_gap(c, j));
            ++j;
        }
        f.seq.push_back(c.cand[(size_t)(j * kernels::kLogitCands)].id), f.gap.push_back(cand_gap(c, j));
        if (j == k) drv.keep_verify();
        else drv.keep_verify_prefix(j + 1);
    }
    f.seq.resize((size_t)n), f.gap.resize((size_t)n);
    return f;
}

std::string mtp_json(const MtpRun &r, int64_t G, double exch_per_token) {
    std::ostringstream o;
    o << "{\"gen_n\":" << G << ",\"decode_ms\":" << r.decode_ms << ",\"tps\":" << (double)G / r.decode_ms * 1000
      << ",\"steps\":" << r.steps << ",\"forwards\":" << r.forwards << ",\"drafted\":" << r.drafted
      << ",\"accepted\":" << r.accepted << ",\"rollbacks\":" << r.rollbacks << ",\"kept_all\":" << r.kept_all
      << ",\"draft_calls\":" << r.draft_calls << ",\"tokens_per_step\":" << (double)G / (double)std::max<int64_t>(1, r.steps)
      << ",\"exchanges_per_token\":" << exch_per_token << ",\"step_ms\":" << js(stats(r.step_ms))
      << ",\"draft_call_ms\":" << js(stats(r.draft_ms)) << ",\"verify_call_ms\":" << js(stats(r.verify_ms))
      << ",\"single_forward_ms\":" << js(stats(r.single_ms)) << ",\"accepted_hist\":{";
    bool f = true;
    for (auto [k, n] : r.acc_hist) o << (f ? "" : ",") << "\"" << k << "\":" << n, f = false;
    o << "}}";
    return o.str();
}

}  // namespace

int main(int argc, char **argv) {
    try {
        Args a;
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            STRIX_CHECK(k.rfind("--", 0) == 0 && i + 1 < argc, "tp_ar: bad argument ", k);
            a.kv[k.substr(2)] = argv[++i];
        }
        const std::string weights = a.get("weights"), ngram = a.get("ngram"), tokj = a.get("tokenizer");
        const int world = (int)a.num("tp-world", 1), rank = (int)a.num("tp-rank", 0);
        const int64_t chunk = a.num("chunk", 8192), cache_rows = a.num("ngram-cache-rows", 8388608);
        const float yarn = (float)a.real("yarn", 2.0);
        const int64_t kl_prefix = a.num("kl-prefix", 8192), kl_n = a.num("kl-n", 5120), cand_check = a.num("cand-check", 32);
        const int64_t ar_steps = a.num("ar-steps", 128);
        std::vector<int64_t> depths;
        for (const std::string &d : split(a.get("depths", "4096,65536"), ',')) depths.push_back(std::stoll(d));
        std::sort(depths.begin(), depths.end());
        const int64_t maxD = depths.empty() ? 0 : depths.back();
        const bool hash = a.num("hash", 1) != 0;
        const int noise_k = (int)a.num("noise-split-k", 0);
        const std::string golden_out = a.get("golden-out"), golden_in = a.get("golden-in"), out = a.get("out", "/dev/stderr");
        STRIX_CHECK(!weights.empty() && !ngram.empty() && !tokj.empty(),
                    "usage: tp_ar --weights W --ngram T --tokenizer J [--corpus ..] [--kl-corpus ..] [--tp-world N ...]");
        const int64_t capacity =
            ((std::max(kl_prefix + kl_n + cand_check + 16, maxD + ar_steps + 64) + 4095) / 4096 + 1) * 4096;

        double t0 = now_ms();
        const bool use_mtp = a.num("mtp", 0) != 0, shadow = a.num("shadow", 0) != 0;
        const int64_t mtp_draft = a.num("mtp-draft", 7), mtp_vocab = a.num("mtp-vocab", 65536), gen_n = a.num("gen", 256);
        const double margin = a.real("mtp-margin", 1.5);  // default since the Q4 draft head (was 2.0 x 5, Q8 head)
        const bool reject_forward = a.num("mtp-reject-forward", 0) != 0, verify_k = a.num("verify-k", 1) != 0;
        const int64_t verify_reps = a.num("verify-reps", 8);
        STRIX_CHECK(!shadow || world == 1, "tp_ar: --shadow runs the whole model in one process (world 1)");
        Qwen4ExpModel model(weights, ngram, kernels::Act::BF16, false, cache_rows, yarn,
                                  TpConfig{world, rank, std::max<int64_t>(1, mtp_vocab)});
        const double load_s = (now_ms() - t0) / 1000;
        const Qwen4ExpDims &Dm = model.dims();
        std::fprintf(stderr, "tp_ar: rank %d of %d: %.2f GiB of weights (%.2f GiB read) loaded in %.1f s\n", rank, world,
                     (double)model.weights().data_bytes() / (1ull << 30), (double)model.weights().bytes_read() / (1ull << 30),
                     load_s);
        // Rank 0 (or the whole model) keeps the MTP head with --mtp 1; executors never draft (no head loaded).
        const bool ses_mtp = (use_mtp || shadow) && rank == 0;
        STRIX_CHECK(!ses_mtp || model.has_mtp(), "tp_ar: --mtp 1 but the weights have no MTP head");
        const int64_t cap = shadow ? ((a.num("shadow-prefix", 4096) + a.num("shadow-tokens", 2048) + 1024) / 4096 + 1) * 4096
                                   : capacity + (use_mtp ? ((gen_n + 4095) / 4096) * 4096 : 0);
        // --replay 1 (rank 0): the session is a Qwen4ExpBackend's, driven through TpDriver (set_tp_driver) by
        // serve/replay's teacher-forced engine loop at the end - the backend-over-TP path.
        const bool replay = a.num("replay", 0) != 0 && rank == 0 && !shadow;
        std::unique_ptr<Qwen4ExpBackend> be;
        std::unique_ptr<Qwen4ExpSession> own;
        if (replay) be = std::make_unique<Qwen4ExpBackend>(model, cap, chunk, ses_mtp, ses_mtp ? mtp_vocab : 0);
        else own = std::make_unique<Qwen4ExpSession>(model, cap, chunk, PrefillMath::WmmaBf16, ses_mtp);
        Qwen4ExpSession &ses = be ? be->session() : *own;
        if (ses_mtp) ses.set_mtp_vocab(mtp_vocab);
        // The Q4 draft head (rank 0 / world 1 only: executors never draft)
        const bool draft_q4 = onoff_arg(a.get("mtp-draft-q4", "on"), "tp_ar --mtp-draft-q4");
        const bool free_q4 = a.num("mtp-free-q4", 0) != 0;
        // control for free_q4: the same free-running decode with the Q8 head under another policy (MARGINxDRAFTS):
        // how far any change of the drafts moves the verify-decided tokens
        const std::string free_alt = a.get("mtp-free-alt");
        struct SweepCell {
            std::string key;
            double margin;
            int64_t draft;
            bool q4;
        };
        std::vector<SweepCell> sweep;
        for (const std::string &c : split(a.get("mtp-sweep"), ',')) {
            const size_t x = c.find('x');
            STRIX_CHECK(x != std::string::npos, "tp_ar --mtp-sweep: cell '", c, "', expected MARGINxDRAFTS[q4]");
            const bool q4 = c.size() > 2 && c.compare(c.size() - 2, 2, "q4") == 0;
            sweep.push_back({c, std::stod(c.substr(0, x)), std::stoll(c.substr(x + 1, c.size() - x - 1 - (q4 ? 2 : 0))), q4});
        }
        const int64_t sweep_reps = a.num("sweep-reps", 2);
        double draft_q4_s = 0;
        if (ses_mtp && (draft_q4 || free_q4 || std::any_of(sweep.begin(), sweep.end(), [](const SweepCell &c) { return c.q4; }))) {
            const double q0 = now_ms();
            model.make_draft_head_q4(ses.mtp_vocab(), a.num("mtp-draft-q4-group", 64));
            draft_q4_s = (now_ms() - q0) / 1000;
            std::fprintf(stderr, "tp_ar: Q4 draft head: %lld rows, group %lld, made in %.1f s\n",
                         (long long)model.draft_head_q4().N(), (long long)model.draft_head_q4().q4.G, draft_q4_s);
        }
        if (ses_mtp) ses.set_mtp_draft_q4(draft_q4);
        // FP32 partials are the session's default under TP; 0 switches them off (the BF16 A/B).
        const bool f32_mixer = world > 1 && a.num("tp-f32-mixer", 1) != 0;
        const bool f32_moe = world > 1 && a.num("tp-f32-moe", 1) != 0;
        ses.set_grouped_min_tokens(a.num("grouped-min-tokens", Qwen4ExpSession::kGroupedMinTokens));  // A/B
        if (world > 1) {
            ses.set_tp_f32_mixer(f32_mixer);
            ses.set_tp_f32_moe(f32_moe);
        }
        const int64_t f32_max_t = a.num("tp-f32-max-tokens", 0);  // > 0: FP32 partials only for forwards of <= this
        if (f32_max_t > 0) ses.set_tp_f32_max_tokens(f32_max_t);
        std::unique_ptr<TpComm> comm;
        if (world > 1) {
            TpCommConfig cc;
            cc.rank = rank;
            cc.peers = split(a.get("tp-peers"), ',');
            STRIX_CHECK((int)cc.peers.size() == world, "tp_ar: --tp-peers lists ", cc.peers.size(), " addresses for world ", world);
            cc.dev = a.get("tp-dev", "");  // empty: the mlx5 device with a RoCE v2 GID for this rank's address
            cc.port = (int)a.num("tp-port", 18600);
            cc.max_bytes = (size_t)(chunk * Dm.d * (f32_mixer || f32_moe ? 4 : 2));
            cc.exchange_timeout_s = a.real("tp-timeout", 900);
            t0 = now_ms();
            comm = std::make_unique<TpComm>(cc);
            tp_attach(ses, model, *comm);
            std::fprintf(stderr, "tp_ar: rank %d of %d: communicator up in %.1f s (%s, port %d)\n", rank, world,
                         (now_ms() - t0) / 1000, comm->device().c_str(), cc.port + rank);
        }
        if (rank > 0) {
            tp_executor(ses, *comm);
            comm->barrier();
            std::fprintf(stderr, "tp_ar: rank %d: done (%llu exchanges)\n", rank, (unsigned long long)comm->exchanges());
            return 0;
        }

        // ---- rank 0 / the whole model ----
        const Tokenizer tok(tokj);
        const int64_t nv = (int64_t)tok.size(), V = Dm.vocab;
        auto corpus_ids = [&]() {
            std::vector<int32_t> corpus;
            for (const std::string &f : split(a.get("corpus"), ',')) {
                const std::vector<int32_t> ids = tok.encode(read_file(f));
                corpus.insert(corpus.end(), ids.begin(), ids.end());
            }
            return corpus;
        };

        if (shadow) {
            // ---- the mirror-shadow test (one process; see the header) ----
            const int64_t P = a.num("shadow-prefix", 4096), target = a.num("shadow-tokens", 2048), G = a.num("shadow-gen", 256);
            const std::vector<int32_t> corpus = corpus_ids();
            STRIX_CHECK((int64_t)corpus.size() > P + 8, "tp_ar: corpus too short");
            Qwen4ExpSession exe(model, cap, chunk, PrefillMath::WmmaBf16, false);
            TpLoopback lb;
            std::string exe_err;
            std::thread ex([&] {
                try {
                    tp_executor(exe, lb.end(1));
                } catch (const std::exception &e) {
                    exe_err = e.what();
                }
            });
            TpDriver drv(ses, &lb.end(0));
            drv.set_hash_check(true);
            drv.set_state_check(true);
            drv.set_shadow(true);
            std::mt19937_64 rng((uint64_t)a.num("shadow-seed", 1));
            std::uniform_real_distribution<double> U(0.0, 1.0);
            const double p_drop = a.real("shadow-p-drop", 0.08), p_restore = a.real("shadow-p-restore", 0.04),
                         p_fail = a.real("shadow-p-fail", 0.006);
            const int64_t max_fail = a.num("shadow-max-fail", 3);
            const bool plant = a.num("shadow-plant", 0) != 0;  // negative control: one mis-mirrored keep_verify_prefix
            if (plant) drv.debug_plant_prefix_bug();
            std::map<std::string, int64_t> cases;
            std::vector<int32_t> seq(corpus.begin(), corpus.begin() + P);  // every token the state has run so far
            auto prefill = [&](int64_t n) {  // reset, then the state after seq[0, n) by chunked forwards
                drv.reset();
                for (int64_t p = 0; p < n;) {
                    const int64_t end = std::min(n, p + chunk);
                    if (end < n) drv.set_lookahead(std::vector<int32_t>(seq.begin() + end, seq.begin() + std::min(n, end + chunk)));
                    drv.forward(std::vector<int32_t>(seq.begin() + p, seq.begin() + end), end == n ? 1 : 0);
                    p = end;
                }
            };
            const double t_start = now_ms();
            prefill(P);
            const int gsnap = drv.make_snapshot(), msnap = drv.make_snapshot();
            int64_t done = 0, round = 0, fails = 0;
            int32_t next = corpus[(size_t)P];  // the known, not yet forwarded token
            while (done < target) {
                const bool pf1 = round % 2 == 0;
                drv.save(gsnap);
                const int64_t base = (int64_t)seq.size();
                // the capture: greedy from `next`
                std::vector<int32_t> gen{next};
                for (int64_t i = 1; i < G; ++i) {
                    drv.want_candidates(nv);
                    drv.forward({gen.back()}, 1);
                    gen.push_back(drv.candidates().cand[0].id);
                }
                drv.restore(gsnap);
                // the MTP loop over it, with the injected cases
                int64_t i = 0, restore_at = -1, saved_i = -1;
                while (i < G) {
                    if (restore_at < 0 && U(rng) < p_restore) {  // a snapshot here, restored a few steps on
                        drv.save(msnap);
                        saved_i = i, restore_at = i + 1 + (int64_t)(U(rng) * 8);
                        ++cases["snapshot_saved"];
                    } else if (restore_at >= 0 && i >= restore_at) {
                        drv.restore(msnap);
                        i = saved_i, restore_at = -1;
                        ++cases["snapshot_restored"];
                    }
                    std::vector<int32_t> ids{gen[(size_t)i]};
                    const int64_t nsteps = std::min(mtp_draft, G - 1 - i);
                    if (nsteps > 0) drv.prefetch_ple(ids, 0);
                    for (int64_t st = 0; st < nsteps; ++st) {
                        const Qwen4ExpSession::MtpTop2 t = drv.forward_mtp_top2(ids.back(), st);
                        if (t.best_v - t.second_v < margin) break;
                        ids.push_back(t.best);
                        if (st + 1 < nsteps) drv.prefetch_ple(ids, (int64_t)ids.size() - 1);
                    }
                    const int64_t k = (int64_t)ids.size() - 1;
                    if (k == 0) {
                        drv.want_candidates(nv);
                        drv.forward(ids, 1);
                        ++cases["single"];
                        i += 1;
                        continue;
                    }
                    drv.want_candidates(nv);
                    drv.forward_verify(ids, k + 1);
                    if (fails < max_fail && U(rng) < p_fail) {
                        // A rank-0-only failure between a verify and its keep (an engine exception in drafting): the
                        // backend's error path resets every rank; the state is then rebuilt up to gen[i].
                        bool threw = false;
                        try {
                            drv.forward_mtp_top2(ids[0], 0);  // refused: a verify awaits its keep / drop
                        } catch (const std::exception &) {
                            threw = true;
                        }
                        STRIX_CHECK(threw, "tp_ar --shadow: the injected failure didn't fail");
                        ++fails, ++cases["failure_reset"];
                        std::vector<int32_t> cur = seq;
                        cur.insert(cur.end(), gen.begin(), gen.begin() + i);
                        std::swap(seq, cur);
                        prefill((int64_t)seq.size());
                        std::swap(seq, cur);
                        restore_at = -1;
                        continue;
                    }
                    if (U(rng) < p_drop) {
                        drv.drop_verify();
                        ++cases["drop"];
                        continue;  // the same step again (the drafts are the same: the state is)
                    }
                    int64_t j = 0;
                    while (j < k && ids[(size_t)j + 1] == gen[(size_t)(i + 1 + j)]) ++j;
                    if (j == k) {
                        drv.keep_verify();
                        ++cases["ended_at_k"];
                        i += k + 1;
                        continue;
                    }
                    drv.keep_verify_prefix(j + 1);
                    ++cases["ended_at_j_lt_k"];
                    const int64_t v = i + j + 1;
                    if (!pf1) {
                        drv.want_candidates(nv);
                        drv.forward({gen[(size_t)v]}, 1);
                        ++cases["reject_forward"];
                        i = v + 1;
                    } else {
                        ++cases["pf1_carry"];
                        i = v;
                    }
                }
                if (restore_at >= 0) restore_at = -1;
                // gen[G - 1] ran (every path forwards up to and including the last token); the next round starts
                // from the corpus again (a fresh known token after the generated text).
                seq.insert(seq.end(), gen.begin(), gen.end());
                STRIX_CHECK(ses.pos() == (int64_t)seq.size(), "tp_ar --shadow: position ", ses.pos(), " after round ", round,
                            ", expected ", seq.size(), " (base ", base, ")");
                next = corpus[(size_t)(P + round + 1)];
                done += G, ++round;
                const TpDriver::HashStats &h = drv.hash_stats();
                std::fprintf(stderr, "tp_ar: shadow: round %lld (%s) done: %lld tokens, %lld forwards, %lld mismatched, %lld state checks, %lld mismatched (%.0f s)\n",
                             (long long)round, pf1 ? "carry" : "reject-forward", (long long)done, (long long)h.forwards,
                             (long long)h.mismatched, (long long)h.state_checks, (long long)h.state_mismatched,
                             (now_ms() - t_start) / 1000);
                if (h.mismatched || h.state_mismatched) break;
            }
            const TpDriver::HashStats hs = drv.hash_stats();
            const std::map<std::string, int64_t> ops = drv.ops();
            drv.finish();
            ex.join();
            std::ostringstream S;
            S << "{\"tool\":\"tp_ar\",\"mode\":\"shadow\",\"weights\":\"" << weights << "\",\"prefix\":" << P
              << ",\"tokens\":" << done << ",\"rounds\":" << round << ",\"gen_per_round\":" << G << ",\"mtp_draft\":" << mtp_draft
              << ",\"mtp_margin\":" << margin << ",\"mtp_vocab\":" << mtp_vocab
              << ",\"planted_bug\":" << (plant ? "true" : "false")
              << ",\"planted_caught\":" << (plant && (hs.mismatched > 0 || hs.state_mismatched > 0) ? "true" : "false")
              << ",\"forwards\":" << hs.forwards << ",\"verify_forwards\":" << hs.verify_forwards << ",\"compared\":" << hs.compared
              << ",\"mismatched\":" << hs.mismatched << ",\"verify_mismatched\":" << hs.verify_mismatched
              << ",\"x_compared\":" << hs.x_compared << ",\"first_mismatch\":\"" << esc(hs.first_mismatch)
              << "\",\"state_checks\":" << hs.state_checks << ",\"state_hash_mismatched\":" << hs.state_mismatched
              << ",\"first_state_mismatch\":\"" << esc(hs.first_state_mismatch) << "\",\"executor_error\":\"" << esc(exe_err)
              << "\",\"compare\":\"per forward / verify: every replicated activation (embed, PLE out, L<i>.mixer/moe/out, final mix), "
                 "the candidates and the returned logits rows; per keep / drop / restore / reset: position, PLE state, n-gram "
                 "history, indexer tails, newest block keys\",\"ops\":{";
            bool f = true;
            for (auto [k, n] : ops) S << (f ? "" : ",") << "\"" << k << "\":" << n, f = false;
            S << "},\"cases\":{";
            f = true;
            for (auto [k, n] : cases) S << (f ? "" : ",") << "\"" << k << "\":" << n, f = false;
            S << "},\"seconds\":" << (now_ms() - t_start) / 1000 << "}\n";
            std::ofstream o(out);
            o << S.str();
            std::fprintf(stderr, "tp_ar: shadow: %lld forwards, %lld mismatched, %lld state mismatched; wrote %s\n",
                         (long long)hs.forwards, (long long)hs.mismatched, (long long)hs.state_mismatched, out.c_str());
            return exe_err.empty() || plant ? 0 : 1;
        }

        TpDriver drv(ses, comm.get());
        if (be) be->set_tp_driver(&drv);
        std::vector<int32_t> last_gen, last_ctx;  // the last depth's continuation and context (for --replay)
        MtpRun last_mtp;
        std::ostringstream J;
        J << "{\"tool\":\"tp_ar\",\"weights\":\"" << weights << "\",\"ngram\":\"" << ngram << "\",\"tp_world\":" << world
          << ",\"weights_gib\":" << (double)model.weights().data_bytes() / (1ull << 30)
          << ",\"read_gib\":" << (double)model.weights().bytes_read() / (1ull << 30) << ",\"load_s\":" << load_s
          << ",\"capacity\":" << cap << ",\"chunk\":" << chunk << ",\"yarn\":" << yarn
          << ",\"activations\":\"BF16\",\"partials\":\""
          << (world == 1 ? "n/a"
              : f32_mixer && f32_moe ? "mixer and MoE FP32 (unrounded accumulators); summed in FP32 in rank order, rounded once (RNE)"
              : f32_moe ? "MoE FP32 (unrounded accumulators), mixer BF16; summed in FP32 in rank order, rounded once (RNE)"
              : f32_mixer ? "mixer FP32 (unrounded accumulators), MoE BF16; summed in FP32 in rank order, rounded once (RNE)"
                          : "BF16, summed in FP32 in rank order, rounded once (RNE)")
          << "\"";
        auto exch = [&]() -> uint64_t { return comm ? comm->exchanges() : 0; };

        if (kl_n > 0) {
            const std::string klc = a.get("kl-corpus");
            const std::vector<int32_t> kt = tok.encode(read_file(klc));
            STRIX_CHECK((int64_t)kt.size() > kl_prefix + kl_n + cand_check + 2, "tp_ar: KL corpus has ", kt.size(), " tokens");
            STRIX_CHECK(kl_prefix % chunk == 0 || kl_prefix < chunk, "tp_ar: --kl-prefix ", kl_prefix, " vs --chunk ", chunk);
            FILE *gout = nullptr, *gin = nullptr;
            if (!golden_out.empty()) {
                gout = golden_out == "-" ? stdout : std::fopen(golden_out.c_str(), "wb");
                STRIX_CHECK(gout, "tp_ar: cannot write ", golden_out);
            }
            if (!golden_in.empty()) {
                gin = std::fopen(golden_in.c_str(), "rb");
                STRIX_CHECK(gin, "tp_ar: cannot read ", golden_in);
            }
            std::vector<float> keep;  // the first pass's rows, for the noise pass
            if (noise_k > 0) keep.resize((size_t)(kl_n * V));
            auto pass = [&](bool first, std::vector<double> &kls, int64_t &agree, double &nll, double &nll_g) {
                drv.reset();
                drv.set_hash_check(first && hash && world > 1);
                const double p0 = now_ms();
                for (int64_t p = 0; p < kl_prefix; p += chunk)
                    drv.forward(std::vector<int32_t>(kt.begin() + p, kt.begin() + std::min(kl_prefix, p + chunk)), 0);
                std::fprintf(stderr, "tp_ar: quality: prefill %lld in %.1f s\n", (long long)kl_prefix, (now_ms() - p0) / 1000);
                std::vector<float> g((size_t)V);
                const double d0 = now_ms();
                for (int64_t i = 0; i < kl_n; ++i) {
                    const std::vector<float> l = drv.forward({kt[(size_t)(kl_prefix + i)]}, 1, true);
                    const int32_t nxt = kt[(size_t)(kl_prefix + i + 1)];
                    nll += lse(l.data(), nv) - (double)l[(size_t)nxt];
                    const float *ref = nullptr;
                    if (first && gout) STRIX_CHECK(std::fwrite(l.data(), 4, (size_t)V, gout) == (size_t)V, "tp_ar: golden write");
                    if (first && noise_k > 0) std::memcpy(keep.data() + (size_t)(i * V), l.data(), (size_t)V * 4);
                    if (!first) ref = keep.data() + (size_t)(i * V);
                    else if (gin) {
                        STRIX_CHECK(std::fread(g.data(), 4, (size_t)V, gin) == (size_t)V, "tp_ar: golden row ", i, " short");
                        ref = g.data();
                    }
                    if (ref) {
                        kls.push_back(kl_rows(ref, l.data(), nv));
                        agree += argmax(ref, nv) == argmax(l.data(), nv);
                        nll_g += lse(ref, nv) - (double)ref[(size_t)nxt];
                    }
                    if ((i + 1) % 512 == 0)
                        std::fprintf(stderr, "tp_ar: quality: %lld / %lld rows (%.1f ms/row)%s\n", (long long)(i + 1),
                                     (long long)kl_n, (now_ms() - d0) / (double)(i + 1),
                                     kls.empty() ? "" : cat(", KL so far ", [&] {
                                                           double s = 0;
                                                           for (double k : kls) s += k;
                                                           return s / (double)kls.size();
                                                       }()).c_str());
                }
            };
            std::vector<double> kls;
            int64_t agree = 0;
            double nll = 0, nll_g = 0;
            // Row-hash check: rank 0's per-position L<i>.out row hashes of this AR pass, for the MTP pass below
            const bool mtp_q = use_mtp && drv.has_mtp() && a.num("mtp-quality", 1) != 0;
            RowHashes rhp;
            std::vector<uint64_t> ar_rows;
            if (mtp_q)
                drv.set_probe([&, p = rhp.probe()](const std::string &n, const void *d, int64_t r, int64_t c, ProbeType t) {
                    p(n, d, r, c, t);
                    if (r == 1 && n == "L" + std::to_string(Dm.layers - 1) + ".out") ar_rows.push_back(rhp.h.at(0));
                });
            pass(true, kls, agree, nll, nll_g);
            drv.set_probe(nullptr);
            if (gout && gout != stdout) std::fclose(gout);
            if (gout == stdout) std::fflush(stdout);
            if (gin) std::fclose(gin);
            const TpDriver::HashStats hs = drv.hash_stats();
            J << ",\"quality\":{\"corpus\":\"" << klc << "\",\"prefix\":" << kl_prefix << ",\"positions\":" << kl_n
              << ",\"n_valid\":" << nv << ",\"nll\":" << nll / (double)kl_n << ",\"ppl\":" << std::exp(nll / (double)kl_n);
            if (!kls.empty()) {
                double km = 0;
                for (double k : kls) km += k;
                km /= (double)kls.size();
                J << ",\"vs_golden\":{\"golden\":\"" << golden_in << "\",\"kl_mean\":" << km << ",\"kl\":" << js(stats(kls))
                  << ",\"top1_agree\":" << (double)agree / (double)kls.size() << ",\"top1_agree_n\":" << agree
                  << ",\"golden_nll\":" << nll_g / (double)kls.size() << ",\"golden_ppl\":" << std::exp(nll_g / (double)kls.size());
                // the first 256 positions alone (strix_bench's golden span, when the weights are the same)
                if (kls.size() >= 256) {
                    double k256 = 0;
                    for (size_t i = 0; i < 256; ++i) k256 += kls[i];
                    J << ",\"kl_mean_first256\":" << k256 / 256;
                }
                J << "}";
            }
            if (world > 1 && hash)
                J << ",\"hash\":{\"forwards\":" << hs.forwards << ",\"compared\":" << hs.compared << ",\"mismatched\":" << hs.mismatched
                  << ",\"x_compared\":" << hs.x_compared << ",\"x_mismatched\":" << hs.x_mismatched << ",\"first_mismatch\":\""
                  << esc(hs.first_mismatch) << "\",\"x_hash_equal\":"
                  << (hs.mismatched == 0 && hs.x_compared > 0 ? "true" : "false") << "}";
            J << "}";
            std::fprintf(stderr, "tp_ar: quality: ppl %.5f%s\n", std::exp(nll / (double)kl_n),
                         world > 1 && hash ? cat(", hashes ", hs.compared, " compared, ", hs.mismatched, " mismatched").c_str() : "");

            // candidates (GPU, merged over the ranks) vs the whole rows' top kLogitCands
            if (cand_check > 0) {
                const int snap = drv.make_snapshot();
                int64_t bad = 0, top1_bad = 0;
                std::string first_bad;
                for (int64_t j = 0; j < cand_check; ++j) {
                    const int32_t id = kt[(size_t)(kl_prefix + kl_n + j)];
                    drv.save(snap);
                    drv.want_candidates(nv);
                    drv.forward({id}, 1);
                    const std::vector<kernels::LogitCand> c = drv.candidates().cand;
                    drv.restore(snap);
                    const std::vector<float> l = drv.forward({id}, 1, true);
                    std::vector<int32_t> order((size_t)nv);
                    for (int64_t k = 0; k < nv; ++k) order[(size_t)k] = (int32_t)k;
                    std::partial_sort(order.begin(), order.begin() + kernels::kLogitCands, order.end(), [&](int32_t x, int32_t y) {
                        return l[(size_t)x] > l[(size_t)y] || (l[(size_t)x] == l[(size_t)y] && x < y);
                    });
                    bool ok = true;
                    for (int k = 0; k < kernels::kLogitCands; ++k)
                        if (c[(size_t)k].id != order[(size_t)k] || c[(size_t)k].v != l[(size_t)order[(size_t)k]]) ok = false;
                    if (c[0].id != order[0]) ++top1_bad;
                    if (!ok && first_bad.empty())
                        first_bad = cat("position ", j, ": cand[0] ", c[0].id, " = ", c[0].v, ", row top ", order[0], " = ",
                                        l[(size_t)order[0]]);
                    bad += !ok;
                }
                J << ",\"cand_check\":{\"positions\":" << cand_check << ",\"mismatched\":" << bad << ",\"top1_mismatched\":" << top1_bad
                  << ",\"first\":\"" << esc(first_bad) << "\"}";
                std::fprintf(stderr, "tp_ar: candidate check: %lld / %lld mismatched\n", (long long)bad, (long long)cand_check);
            }
            if (mtp_q) {
                // MTP quality (+ row hashes): the same 5120 positions teacher-forced through the MTP loop (reject path as
                // set): every position gets exactly one logits row - a verify row (row 0: the step's known token; rows
                // >= 1: accepted drafts) or a single forward - scored against the goldens; each row's L<i>.out hash
                // compared with the AR pass's at the same position (a rate: verify rows are not bit-equal, step 0).
                FILE *g2 = golden_in.empty() ? nullptr : std::fopen(golden_in.c_str(), "rb");
                drv.clear_hash_stats();
                drv.set_hash_check(hash && world > 1);
                drv.reset();
                for (int64_t p = 0; p < kl_prefix; p += chunk)
                    drv.forward(std::vector<int32_t>(kt.begin() + p, kt.begin() + std::min(kl_prefix, p + chunk)), 0);
                RowHashes mh;
                drv.set_probe(mh.probe());
                struct Acc {
                    int64_t n = 0, agree = 0, hash_eq = 0;
                    double kl = 0, nll = 0;
                };
                std::map<std::string, Acc> by;  // "verify_row0", "verify_draft_rows", "single"
                std::vector<double> mkls;
                std::vector<float> g((size_t)V);
                const std::vector<int32_t> seq(kt.begin() + kl_prefix, kt.begin() + kl_prefix + kl_n);
                int64_t steps = 0, verifies = 0;
                const double q0 = now_ms();
                auto score = [&](const float *l, int64_t posn, const char *type) {
                    const int32_t nxt = kt[(size_t)(kl_prefix + posn + 1)];
                    Acc &A = by[type];
                    ++A.n;
                    A.nll += lse(l, nv) - (double)l[(size_t)nxt];
                    if (g2) {
                        STRIX_CHECK(std::fread(g.data(), 4, (size_t)V, g2) == (size_t)V, "tp_ar: golden row ", posn, " short");
                        const double k = kl_rows(g.data(), l, nv);
                        mkls.push_back(k);
                        A.kl += k;
                        A.agree += argmax(g.data(), nv) == argmax(l, nv);
                    }
                };
                int64_t next_pos = 0;
                for (int64_t i = 0; i < kl_n;) {
                    std::vector<int32_t> ids{seq[(size_t)i]};
                    const int64_t nsteps = std::min(mtp_draft, kl_n - 1 - i);
                    if (nsteps > 0) drv.prefetch_ple(ids, 0);
                    for (int64_t st = 0; st < nsteps; ++st) {
                        const Qwen4ExpSession::MtpTop2 t = drv.forward_mtp_top2(ids.back(), st);
                        if (t.nan || t.best_v - t.second_v < margin) break;
                        ids.push_back(t.best);
                        if (st + 1 < nsteps) drv.prefetch_ple(ids, (int64_t)ids.size() - 1);
                    }
                    const int64_t k = (int64_t)ids.size() - 1;
                    ++steps;
                    if (k == 0) {
                        const std::vector<float> l = drv.forward(ids, 1, true);
                        STRIX_CHECK(next_pos == i, "tp_ar: MTP quality: position ", i, " after ", next_pos);
                        score(l.data(), i, "single");
                        by["single"].hash_eq += mh.h.at(0) == ar_rows.at((size_t)i);
                        ++next_pos, i += 1;
                        continue;
                    }
                    const std::vector<float> l = drv.forward_verify(ids, k + 1, true);
                    ++verifies;
                    const std::vector<uint64_t> rh = mh.h;
                    int64_t j = 0;
                    while (j < k && ids[(size_t)j + 1] == seq[(size_t)(i + 1 + j)]) ++j;
                    for (int64_t r = 0; r <= j; ++r) {  // rows 0..j are positions i..i+j
                        const char *type = r == 0 ? "verify_row0" : "verify_draft_rows";
                        STRIX_CHECK(next_pos == i + r, "tp_ar: MTP quality: position ", i + r, " after ", next_pos);
                        score(l.data() + (size_t)(r * V), i + r, type);
                        by[type].hash_eq += rh.at((size_t)r) == ar_rows.at((size_t)(i + r));
                        ++next_pos;
                    }
                    if (j == k) {
                        drv.keep_verify();
                        i += k + 1;
                        continue;
                    }
                    drv.keep_verify_prefix(j + 1);
                    const int64_t v = i + j + 1;
                    if (reject_forward) {
                        const std::vector<float> l1 = drv.forward({seq[(size_t)v]}, 1, true);
                        STRIX_CHECK(next_pos == v, "tp_ar: MTP quality: position ", v, " after ", next_pos);
                        score(l1.data(), v, "single");
                        by["single"].hash_eq += mh.h.at(0) == ar_rows.at((size_t)v);
                        ++next_pos;
                        i = v + 1;
                    } else {
                        i = v;
                    }
                    if (next_pos / 512 != (next_pos - j - 1) / 512)
                        std::fprintf(stderr, "tp_ar: MTP quality: %lld / %lld positions (%.1f s)\n", (long long)next_pos,
                                     (long long)kl_n, (now_ms() - q0) / 1000);
                }
                drv.set_probe(nullptr);
                drv.set_hash_check(false);
                if (g2) std::fclose(g2);
                const TpDriver::HashStats qh = drv.hash_stats();
                Acc all;
                for (auto &[k, A] : by) all.n += A.n, all.agree += A.agree, all.hash_eq += A.hash_eq, all.kl += A.kl, all.nll += A.nll;
                J << ",\"quality_mtp\":{\"positions\":" << all.n << ",\"steps\":" << steps << ",\"verifies\":" << verifies
                  << ",\"nll\":" << all.nll / (double)all.n << ",\"ppl\":" << std::exp(all.nll / (double)all.n);
                if (!mkls.empty())
                    J << ",\"vs_golden\":{\"golden\":\"" << golden_in << "\",\"kl_mean\":" << all.kl / (double)all.n
                      << ",\"kl\":" << js(stats(mkls)) << ",\"top1_agree\":" << (double)all.agree / (double)all.n
                      << ",\"top1_agree_n\":" << all.agree << "}";
                J << ",\"row_hash\":{\"compared\":" << all.n << ",\"equal\":" << all.hash_eq << ",\"rate\":"
                  << (double)all.hash_eq / (double)all.n << ",\"what\":\"rank 0's L0..L47.out row hash of each position's row "
                     "in this pass vs the AR pass's at the same position\"}";
                J << ",\"by_row_type\":{";
                bool f = true;
                for (auto &[k, A] : by) {
                    J << (f ? "" : ",") << "\"" << k << "\":{\"n\":" << A.n << ",\"ppl\":" << std::exp(A.nll / (double)A.n)
                      << ",\"row_hash_equal\":" << A.hash_eq;
                    if (!mkls.empty()) J << ",\"kl_mean\":" << A.kl / (double)A.n << ",\"top1_agree\":" << (double)A.agree / (double)A.n;
                    J << "}";
                    f = false;
                }
                J << "}";
                if (world > 1 && hash)
                    J << ",\"hash\":{\"forwards\":" << qh.forwards << ",\"verify_forwards\":" << qh.verify_forwards
                      << ",\"compared\":" << qh.compared << ",\"mismatched\":" << qh.mismatched << ",\"x_compared\":" << qh.x_compared
                      << ",\"x_mismatched\":" << qh.x_mismatched << ",\"first_mismatch\":\"" << esc(qh.first_mismatch)
                      << "\",\"x_hash_equal\":" << (qh.mismatched == 0 && qh.x_compared > 0 ? "true" : "false") << "}";
                J << "}";
                std::fprintf(stderr, "tp_ar: MTP quality: %lld positions, KL %.5f, top-1 %.4f, ppl %.5f, row hashes equal %lld / %lld, hashes mismatched %lld\n",
                             (long long)all.n, mkls.empty() ? 0.0 : all.kl / (double)all.n,
                             (double)all.agree / (double)all.n, std::exp(all.nll / (double)all.n), (long long)all.hash_eq,
                             (long long)all.n, (long long)qh.mismatched);
            }
            if (noise_k > 0 && world == 1) {
                kernels::set_split_k_scale(noise_k);
                std::vector<double> nk;
                int64_t nagree = 0;
                double nnll = 0, dummy = 0;
                pass(false, nk, nagree, nnll, dummy);
                kernels::set_split_k_scale(1);
                double km = 0;
                for (double k : nk) km += k;
                km /= (double)nk.size();
                J << ",\"noise_floor\":{\"settings\":\"decode linears split-K x" << noise_k
                  << " (summation order only) vs as tuned, same positions\",\"kl_mean\":" << km << ",\"kl\":" << js(stats(nk))
                  << ",\"top1_agree\":" << (double)nagree / (double)nk.size() << ",\"ppl\":" << std::exp(nnll / (double)kl_n) << "}";
                std::fprintf(stderr, "tp_ar: noise floor (split-K x%d): KL %.5f, top-1 %.4f\n", noise_k, km,
                             (double)nagree / (double)nk.size());
            }
        }

        // ---- speed ----
        if (!depths.empty() && ar_steps > 0) {
            std::vector<int32_t> corpus = corpus_ids();
            STRIX_CHECK((int64_t)corpus.size() > maxD + ar_steps + gen_n + 64, "tp_ar: corpus has ", corpus.size(), " tokens");
            const bool mtp_on = use_mtp && drv.has_mtp();
            // Needles (strix_bench's): 8 passcodes spliced into [4096, 65536) of a chat turn, asked for at each depth
            // >= 64k; with MTP on the answer is decoded free-running with drafts.
            struct Needle {
                std::string key, code;
                int64_t at;
            };
            std::vector<Needle> needles;
            if (maxD >= 65536 && a.num("needles", 1) != 0) {
                const char *colors[] = {"crimson", "amber", "cobalt", "violet", "emerald", "silver", "ochre", "teal"};
                const char *animals[] = {"heron", "lynx", "otter", "falcon", "badger", "gecko", "marten", "ibis"};
                const int64_t need = maxD + ar_steps + gen_n + 64;
                std::vector<int32_t> ctx = tok.encode("<|im_start|>user\n");
                ctx.insert(ctx.end(), corpus.begin(), corpus.begin() + need);
                for (int i = 7; i >= 0; --i) {
                    char code[16];
                    std::snprintf(code, sizeof code, "%06d", (int)((i * 7919 + 104729) * 37 % 1000000));
                    Needle n{std::string(colors[i]) + " " + animals[i], code, 4096 + 2000 + i * 7400};
                    const std::vector<int32_t> sp = tok.encode(" The secret passcode of the " + n.key + " is " + n.code + ". ");
                    ctx.insert(ctx.begin() + n.at, sp.begin(), sp.end());
                    needles.insert(needles.begin(), n);
                }
                ctx.resize((size_t)need);
                std::copy(ctx.begin(), ctx.end(), corpus.begin());
            }
            const double eps = a.real("near-tie-eps", 1.7155);  // p99 top-2 logit gap (strix_bench --row-invariance)
            int64_t needles_found_total = 0, needles_asked_total = 0;
            std::ostringstream divj;
            int64_t div_runs = 0, div_diverged = 0, div_not_near_tie = 0;
            TpDriver::HashStats vh_total;
            MtpRun mtp_total;
            int64_t mtp_gen_total = 0;
            double mtp_exch_total = 0;
            drv.set_hash_check(false);
            drv.reset();
            const int snap = drv.make_snapshot();
            int64_t pos = 0;
            double prefill_ms = 0;
            J << ",\"speed\":{";
            bool firstd = true;
            for (const int64_t D : depths) {
                const double p0 = now_ms();
                while (pos < D) {
                    const int64_t end = std::min(D, pos + chunk), nend = std::min(D, end + chunk);
                    if (end < D) drv.set_lookahead(std::vector<int32_t>(corpus.begin() + end, corpus.begin() + nend));
                    drv.forward(std::vector<int32_t>(corpus.begin() + pos, corpus.begin() + end), end == D ? 1 : 0);
                    pos = end;
                }
                const double pf = now_ms() - p0;
                prefill_ms += pf;
                drv.save(snap);
                const uint64_t e0 = exch();
                std::vector<double> tf;
                for (int64_t i = 0; i < ar_steps; ++i) {
                    drv.want_candidates(nv);
                    const double s0 = now_ms();
                    drv.forward({corpus[(size_t)(D + i)]}, 1);
                    tf.push_back(now_ms() - s0);
                }
                const double epf = (double)(exch() - e0) / (double)ar_steps;
                drv.restore(snap);
                const Stats s = stats(tf);
                J << (firstd ? "" : ",") << "\"" << D << "\":{\"ar_tf_ms\":" << js(s) << ",\"ar_tps\":" << 1000.0 / s.mean
                  << ",\"exchanges_per_forward\":" << epf << ",\"prefill_s\":" << pf / 1000
                  << ",\"prefill_tps_cumulative\":" << (double)D / prefill_ms * 1000;
                if (mtp_on) {
                    // verify_k: forward_verify(k) + drop, mirrored
                    if (verify_k) {
                        J << ",\"verify_ms\":{";
                        for (int k = 2; k <= 6; ++k) {
                            std::vector<double> vt;
                            for (int64_t rep = 0; rep < verify_reps; ++rep) {
                                drv.restore(snap);
                                drv.want_candidates(nv);
                                const double v0 = now_ms();
                                drv.forward_verify(std::vector<int32_t>(corpus.begin() + D, corpus.begin() + D + k), k);
                                vt.push_back(now_ms() - v0);
                                drv.drop_verify();
                            }
                            J << (k > 2 ? "," : "") << "\"" << k << "\":" << js(stats(vt));
                        }
                        J << "}";
                    }
                    // ar_gen: the greedy continuation (gen[0] = the corpus token at D, teacher-forced)
                    drv.restore(snap);
                    std::vector<int32_t> gen{corpus[(size_t)D]};
                    std::vector<double> gm, ar_gap{0};
                    for (int64_t i = 1; i < gen_n; ++i) {
                        drv.want_candidates(nv);
                        const double s0 = now_ms();
                        drv.forward({gen.back()}, 1);
                        gm.push_back(now_ms() - s0);
                        gen.push_back(drv.candidates().cand[0].id);
                        ar_gap.push_back(cand_gap(drv.candidates(), 0));
                    }
                    // check 6: free-running greedy with MTP vs this AR greedy - the first divergence and its top-2 gaps
                    {
                        drv.restore(snap);
                        const FreeRun fr = mtp_free(drv, gen[0], gen_n, nv, mtp_draft, margin);
                        int64_t d = -1;
                        for (int64_t t = 0; t < gen_n && d < 0; ++t)
                            if (fr.seq[(size_t)t] != gen[(size_t)t]) d = t;
                        ++div_runs;
                        J << ",\"free_run\":{\"tokens\":" << gen_n << ",\"first_divergence\":" << d;
                        if (d >= 0) {
                            const bool near = ar_gap[(size_t)d] < eps;
                            ++div_diverged, div_not_near_tie += !near;
                            J << ",\"ar_top2_gap\":" << ar_gap[(size_t)d] << ",\"mtp_top2_gap\":" << fr.gap[(size_t)d]
                              << ",\"near_tie\":" << (near ? "true" : "false") << ",\"ar_token\":" << gen[(size_t)d]
                              << ",\"mtp_token\":" << fr.seq[(size_t)d];
                        }
                        J << ",\"eps\":" << eps << "}";
                        divj << (div_runs > 1 ? "," : "") << "\"" << D << "\":{\"first_divergence\":" << d;
                        if (d >= 0)
                            divj << ",\"ar_top2_gap\":" << ar_gap[(size_t)d] << ",\"mtp_top2_gap\":" << fr.gap[(size_t)d]
                                 << ",\"near_tie\":" << (ar_gap[(size_t)d] < eps ? "true" : "false");
                        divj << "}";
                        std::fprintf(stderr, "tp_ar: depth %lld: free-running MTP vs AR greedy: first divergence %lld%s\n",
                                     (long long)D, (long long)d,
                                     d >= 0 ? cat(" (AR top-2 gap ", ar_gap[(size_t)d], ", MTP ", fr.gap[(size_t)d], ")").c_str() : "");
                    }
                    if (!needles.empty() && D >= 65536) {
                        int found = 0;
                        J << ",\"needles\":[";
                        for (size_t ni = 0; ni < needles.size(); ++ni) {
                            drv.restore(snap);
                            const std::vector<int32_t> q = tok.encode(
                                "\n\nQuestion: What is the secret passcode of the " + needles[ni].key +
                                "?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe secret passcode of the " +
                                needles[ni].key + " is");
                            drv.want_candidates(nv);
                            drv.forward(q, 1);
                            const FreeRun ans = mtp_free(drv, drv.candidates().cand[0].id, 7, nv, mtp_draft, margin);
                            const std::string at = tok.decode(ans.seq);
                            const bool ok = at.find(needles[ni].code) != std::string::npos;
                            found += ok;
                            J << (ni ? "," : "") << "{\"key\":\"" << needles[ni].key << "\",\"code\":\"" << needles[ni].code
                              << "\",\"answer\":\"" << esc(at) << "\",\"ok\":" << (ok ? "true" : "false") << "}";
                        }
                        J << "],\"needles_found\":" << found << ",\"needles_n\":" << needles.size();
                        needles_found_total += found, needles_asked_total += (int64_t)needles.size();
                        std::fprintf(stderr, "tp_ar: depth %lld: needles with MTP %d / %zu\n", (long long)D, found, needles.size());
                    }
                    J << ",\"ar_gen_ms\":" << js(stats(gm)) << ",\"ar_gen_tps\":" << 1000.0 / stats(gm).mean;
                    last_gen = gen;
                    last_ctx.assign(corpus.begin(), corpus.begin() + D);
                    // the MTP loop: first checked (hashes + state hashes on every rank), then timed
                    if (hash && world > 1) {
                        drv.restore(snap);
                        drv.clear_hash_stats();
                        drv.set_hash_check(true);
                        drv.set_state_check(true);
                        const MtpRun hr = mtp_loop(drv, gen, nv, mtp_draft, margin, reject_forward);
                        drv.set_hash_check(false);
                        drv.set_state_check(false);
                        const TpDriver::HashStats vh = drv.hash_stats();
                        J << ",\"verify_hash\":{\"forwards\":" << vh.forwards << ",\"verify_forwards\":" << vh.verify_forwards
                          << ",\"compared\":" << vh.compared << ",\"mismatched\":" << vh.mismatched
                          << ",\"verify_mismatched\":" << vh.verify_mismatched << ",\"x_compared\":" << vh.x_compared
                          << ",\"x_mismatched\":" << vh.x_mismatched << ",\"state_checks\":" << vh.state_checks
                          << ",\"state_mismatched\":" << vh.state_mismatched << ",\"first_mismatch\":\"" << esc(vh.first_mismatch)
                          << "\",\"first_state_mismatch\":\"" << esc(vh.first_state_mismatch) << "\",\"steps\":" << hr.steps << "}";
                        vh_total.forwards += vh.forwards, vh_total.verify_forwards += vh.verify_forwards;
                        vh_total.compared += vh.compared, vh_total.mismatched += vh.mismatched;
                        vh_total.verify_mismatched += vh.verify_mismatched, vh_total.x_compared += vh.x_compared;
                        vh_total.x_mismatched += vh.x_mismatched, vh_total.state_checks += vh.state_checks;
                        vh_total.state_mismatched += vh.state_mismatched;
                        if (vh_total.first_mismatch.empty()) vh_total.first_mismatch = vh.first_mismatch;
                        if (vh_total.first_state_mismatch.empty()) vh_total.first_state_mismatch = vh.first_state_mismatch;
                        std::fprintf(stderr, "tp_ar: depth %lld: MTP check: %lld forwards (%lld verifies), %lld mismatched, %lld state checks, %lld mismatched\n",
                                     (long long)D, (long long)vh.forwards, (long long)vh.verify_forwards, (long long)vh.mismatched,
                                     (long long)vh.state_checks, (long long)vh.state_mismatched);
                    }
                    drv.restore(snap);
                    // --router-overlap 1: rank 0's routed expert ids of every verify row in the timed MTP loop
                    // (the router_ids probe; it syncs per probe point, so that run's timings are not speed numbers).
                    // adj = |top-k(row r) & top-k(row r+1)| / k over adjacent rows; uniq = distinct experts / (T k)
                    // over a verify's rows, per layer - what expert dedup across rows could save.
                    const bool rov = a.num("router-overlap", 0) != 0;
                    std::map<int64_t, std::array<double, 4>> rov_t;  // T -> {adj shared, adj pairs*k, uniq, T*k}
                    if (rov) {
                        drv.set_probe([&](const std::string &n, const void *d, int64_t rows, int64_t cols, ProbeType) {
                            if (rows < 2 || rows > 16 || n.size() < 10 || n.compare(n.size() - 10, 10, "router_ids") != 0)
                                return;
                            std::vector<int32_t> h((size_t)(rows * cols));
                            STRIX_HIP_CHECK(hipMemcpy(h.data(), d, h.size() * 4, hipMemcpyDeviceToHost), "router_ids probe");
                            std::vector<std::vector<int32_t>> sets((size_t)rows);
                            for (int64_t r = 0; r < rows; ++r) {
                                for (int64_t c = 0; c < cols; ++c) {
                                    const int32_t e = h[(size_t)(r * cols + c)];
                                    if (e >= 0 && e < (int32_t)Dm.experts) sets[(size_t)r].push_back(e);
                                }
                                std::sort(sets[(size_t)r].begin(), sets[(size_t)r].end());
                            }
                            auto &acc = rov_t[rows];
                            std::vector<int32_t> all;
                            for (int64_t r = 0; r < rows; ++r) {
                                all.insert(all.end(), sets[(size_t)r].begin(), sets[(size_t)r].end());
                                acc[3] += (double)sets[(size_t)r].size();
                                if (r + 1 < rows) {
                                    std::vector<int32_t> x;
                                    std::set_intersection(sets[(size_t)r].begin(), sets[(size_t)r].end(),
                                                          sets[(size_t)r + 1].begin(), sets[(size_t)r + 1].end(),
                                                          std::back_inserter(x));
                                    acc[0] += (double)x.size(), acc[1] += (double)sets[(size_t)r].size();
                                }
                            }
                            std::sort(all.begin(), all.end());
                            acc[2] += (double)(std::unique(all.begin(), all.end()) - all.begin());
                        });
                    }
                    const uint64_t m0 = exch();
                    const MtpRun r = mtp_loop(drv, gen, nv, mtp_draft, margin, reject_forward);
                    const double ept = (double)(exch() - m0) / (double)gen_n;
                    if (rov) {
                        drv.set_probe(nullptr);
                        J << ",\"router_overlap\":{";
                        std::array<double, 4> tot{};
                        for (auto &[t, v] : rov_t) {
                            J << "\"" << t << "\":{\"adj_shared_frac\":" << v[0] / std::max(1.0, v[1])
                              << ",\"unique_frac\":" << v[2] / std::max(1.0, v[3]) << ",\"row_layers\":" << v[3] << "},";
                            for (int q = 0; q < 4; ++q) tot[(size_t)q] += v[(size_t)q];
                        }
                        J << "\"all\":{\"adj_shared_frac\":" << tot[0] / std::max(1.0, tot[1])
                          << ",\"unique_frac\":" << tot[2] / std::max(1.0, tot[3]) << "}}";
                    }
                    J << ",\"mtp\":" << mtp_json(r, gen_n, ept);
                    last_mtp = r;
                    std::fprintf(stderr, "tp_ar: depth %lld: MTP %.2f t/s (%lld steps, %lld forwards, %lld drafted, %lld accepted, %lld rollbacks; %.1f exchanges/token)\n",
                                 (long long)D, (double)gen_n / r.decode_ms * 1000, (long long)r.steps, (long long)r.forwards,
                                 (long long)r.drafted, (long long)r.accepted, (long long)r.rollbacks, ept);
                    mtp_total.steps += r.steps, mtp_total.forwards += r.forwards, mtp_total.drafted += r.drafted;
                    mtp_total.accepted += r.accepted, mtp_total.rollbacks += r.rollbacks, mtp_total.kept_all += r.kept_all;
                    mtp_total.draft_calls += r.draft_calls, mtp_total.decode_ms += r.decode_ms;
                    for (auto [k, n] : r.acc_hist) mtp_total.acc_hist[k] += n;
                    mtp_total.step_ms.insert(mtp_total.step_ms.end(), r.step_ms.begin(), r.step_ms.end());
                    mtp_total.draft_ms.insert(mtp_total.draft_ms.end(), r.draft_ms.begin(), r.draft_ms.end());
                    mtp_total.verify_ms.insert(mtp_total.verify_ms.end(), r.verify_ms.begin(), r.verify_ms.end());
                    mtp_total.single_ms.insert(mtp_total.single_ms.end(), r.single_ms.begin(), r.single_ms.end());
                    mtp_gen_total += gen_n, mtp_exch_total += ept * (double)gen_n;
                    // Free-running greedy MTP with the Q8 head vs the Q4 head (the verify decides every token;
                    // drafts differing changes which rows verify them, so near-ties can move)
                    if (free_q4) {
                        FreeRun fr[2];
                        for (int h = 0; h < 2; ++h) {
                            ses.set_mtp_draft_q4(h == 1);
                            drv.restore(snap);
                            fr[h] = mtp_free(drv, gen[0], gen_n, nv, mtp_draft, margin);
                        }
                        ses.set_mtp_draft_q4(draft_q4);
                        int64_t same = 0, d = -1, same8 = 0, same4 = 0;
                        for (int64_t t = 0; t < gen_n; ++t) {
                            const bool e = fr[0].seq[(size_t)t] == fr[1].seq[(size_t)t];
                            same += e;
                            if (!e && d < 0) d = t;
                            same8 += fr[0].seq[(size_t)t] == gen[(size_t)t];
                            same4 += fr[1].seq[(size_t)t] == gen[(size_t)t];
                        }
                        J << ",\"free_q4\":{\"tokens\":" << gen_n << ",\"identical\":" << (d < 0 ? "true" : "false")
                          << ",\"first_divergence\":" << d << ",\"positions_equal\":" << same
                          << ",\"q8_vs_ar_equal\":" << same8 << ",\"q4_vs_ar_equal\":" << same4;
                        if (d >= 0)
                            J << ",\"q8_gap\":" << fr[0].gap[(size_t)d] << ",\"q4_gap\":" << fr[1].gap[(size_t)d]
                              << ",\"near_tie\":" << (std::min(fr[0].gap[(size_t)d], fr[1].gap[(size_t)d]) < eps ? "true" : "false");
                        if (!free_alt.empty()) {
                            const size_t x = free_alt.find('x');
                            STRIX_CHECK(x != std::string::npos, "tp_ar --mtp-free-alt: '", free_alt, "', expected MARGINxDRAFTS");
                            ses.set_mtp_draft_q4(false);
                            drv.restore(snap);
                            const FreeRun fa = mtp_free(drv, gen[0], gen_n, nv, std::stoll(free_alt.substr(x + 1)),
                                                        std::stod(free_alt.substr(0, x)));
                            ses.set_mtp_draft_q4(draft_q4);
                            int64_t da = -1;
                            for (int64_t t = 0; t < gen_n && da < 0; ++t)
                                if (fa.seq[(size_t)t] != fr[0].seq[(size_t)t]) da = t;
                            J << ",\"alt\":{\"policy\":\"" << free_alt << "\",\"first_divergence_vs_q8\":" << da;
                            if (da >= 0) J << ",\"q8_gap\":" << fr[0].gap[(size_t)da] << ",\"alt_gap\":" << fa.gap[(size_t)da];
                            J << "}";
                            std::fprintf(stderr, "tp_ar: depth %lld: free-running MTP Q8 %s vs Q8 main policy: first divergence %lld\n",
                                         (long long)D, free_alt.c_str(), (long long)da);
                        }
                        J << "}";
                        std::fprintf(stderr, "tp_ar: depth %lld: free-running MTP Q8 vs Q4 draft head: %s (first divergence %lld, %lld/%lld equal)\n",
                                     (long long)D, d < 0 ? "identical" : "DIFFERENT", (long long)d, (long long)same, (long long)gen_n);
                    }
                    if (!sweep.empty()) {
                        std::vector<std::vector<MtpRun>> runs(sweep.size());
                        for (int64_t rp = 0; rp < sweep_reps; ++rp)
                            for (size_t ci = 0; ci < sweep.size(); ++ci) {
                                const size_t c = rp % 2 ? sweep.size() - 1 - ci : ci;
                                ses.set_mtp_draft_q4(sweep[c].q4);
                                drv.restore(snap);
                                runs[c].push_back(mtp_loop(drv, gen, nv, sweep[c].draft, sweep[c].margin, reject_forward));
                            }
                        ses.set_mtp_draft_q4(draft_q4);
                        J << ",\"sweep\":{";
                        for (size_t c = 0; c < sweep.size(); ++c) {
                            const MtpRun &f = runs[c].front();
                            std::vector<double> tps, dms, vms, sms;
                            for (const MtpRun &x : runs[c]) {
                                tps.push_back((double)gen_n / x.decode_ms * 1000);
                                dms.insert(dms.end(), x.draft_ms.begin(), x.draft_ms.end());
                                vms.insert(vms.end(), x.verify_ms.begin(), x.verify_ms.end());
                                sms.insert(sms.end(), x.single_ms.begin(), x.single_ms.end());
                            }
                            const Stats ts = stats(tps);
                            J << (c ? "," : "") << "\"" << sweep[c].key << "\":{\"margin\":" << sweep[c].margin
                              << ",\"drafts\":" << sweep[c].draft << ",\"q4\":" << (sweep[c].q4 ? "true" : "false")
                              << ",\"tps_mean\":" << ts.mean << ",\"tps\":[";
                            for (size_t z = 0; z < tps.size(); ++z) J << (z ? "," : "") << tps[z];
                            J << "],\"steps\":" << f.steps << ",\"tokens_per_step\":" << (double)gen_n / (double)f.steps
                              << ",\"forwards\":" << f.forwards << ",\"singles\":" << (f.steps - (int64_t)f.verify_ms.size())
                              << ",\"draft_calls\":" << f.draft_calls << ",\"drafted\":" << f.drafted << ",\"accepted\":" << f.accepted
                              << ",\"margin_failed_calls\":" << f.draft_calls - f.drafted << ",\"rejected_drafts\":" << f.drafted - f.accepted
                              << ",\"wasted_drafts\":" << f.draft_calls - f.accepted << ",\"rollbacks\":" << f.rollbacks
                              << ",\"kept_all\":" << f.kept_all << ",\"draft_call_ms\":" << js(stats(dms))
                              << ",\"verify_call_ms\":" << js(stats(vms)) << ",\"single_forward_ms\":" << js(stats(sms)) << ",\"accepted_hist\":{";
                            bool fh = true;
                            for (auto [k, n] : f.acc_hist) J << (fh ? "" : ",") << "\"" << k << "\":" << n, fh = false;
                            J << "}}";
                            std::fprintf(stderr, "tp_ar: depth %lld: sweep %s: %.2f t/s (%.3f tok/step, %lld calls, %lld drafted, %lld accepted, draft %.3f ms)\n",
                                         (long long)D, sweep[c].key.c_str(), ts.mean, (double)gen_n / (double)f.steps,
                                         (long long)f.draft_calls, (long long)f.drafted, (long long)f.accepted, stats(dms).mean);
                        }
                        J << "}";
                    }
                    drv.restore(snap);
                }
                J << "}";
                firstd = false;
                std::fprintf(stderr, "tp_ar: depth %lld: AR %.3f ms p50 (%.1f t/s), %.0f exchanges/forward, prefill %.1f s\n",
                             (long long)D, s.p50, 1000.0 / s.mean, epf, pf / 1000);
            }
            J << "}";
            if (mtp_on)
                J << ",\"free_run\":{\"eps\":" << eps << ",\"runs\":" << div_runs << ",\"diverged\":" << div_diverged
                  << ",\"diverged_not_near_tie\":" << div_not_near_tie << ",\"by_depth\":{" << divj.str() << "}}"
                  << ",\"needles_found\":" << needles_found_total << ",\"needles_n\":" << needles_asked_total;
            if (mtp_on) {  // over every depth (the per-depth numbers are under speed)
                J << ",\"mtp\":" << mtp_json(mtp_total, mtp_gen_total, mtp_exch_total / (double)std::max<int64_t>(1, mtp_gen_total));
                if (hash && world > 1)
                    J << ",\"verify_hash\":{\"forwards\":" << vh_total.forwards << ",\"verify_forwards\":" << vh_total.verify_forwards
                      << ",\"compared\":" << vh_total.compared << ",\"mismatched\":" << vh_total.mismatched
                      << ",\"verify_mismatched\":" << vh_total.verify_mismatched << ",\"x_compared\":" << vh_total.x_compared
                      << ",\"x_mismatched\":" << vh_total.x_mismatched << ",\"state_checks\":" << vh_total.state_checks
                      << ",\"state_mismatched\":" << vh_total.state_mismatched << ",\"first_mismatch\":\""
                      << esc(vh_total.first_mismatch) << "\",\"first_state_mismatch\":\"" << esc(vh_total.first_state_mismatch)
                      << "\",\"x_hash_equal_verify\":" << (vh_total.mismatched == 0 && vh_total.x_compared > 0 ? "true" : "false") << "}";
            }
        }
        if (be && !last_gen.empty()) {
            // serve/replay over the backend over TpDriver: the same context + continuation as the last depth's MTP
            // loop, greedy sampled from GPU candidates, with every forward / verify and keep / drop hash-checked.
            CaptureRecord rec;
            rec.id = 1, rec.prompt_n = (int64_t)last_ctx.size();
            rec.tokens = last_ctx;
            rec.tokens.insert(rec.tokens.end(), last_gen.begin(), last_gen.end());
            ReplayOptions opt;
            opt.max_context = rec.prompt_n, opt.max_gen = (int64_t)last_gen.size(), opt.mtp_draft = mtp_draft;
            opt.mtp_margin = (float)margin, opt.mtp_reject_forward = reject_forward;
            opt.sample = ReplayOptions::Sample::Candidates, opt.sampling.temperature = 0, opt.n_valid = nv;
            drv.clear_hash_stats();
            drv.set_hash_check(hash && world > 1);
            drv.set_state_check(hash && world > 1);
            const ReplayResult rr = replay_teacher_forced(*be, rec, opt);
            drv.set_hash_check(false);
            drv.set_state_check(false);
            const TpDriver::HashStats rh = drv.hash_stats();
            const bool same = rr.drafted == last_mtp.drafted && rr.accepted == last_mtp.accepted &&
                              rr.rollbacks == last_mtp.rollbacks && rr.forwards == last_mtp.forwards;
            J << ",\"replay\":{\"path\":\"serve/replay over Qwen4ExpBackend over TpDriver (greedy from GPU candidates)\""
              << ",\"context\":" << rr.context_n << ",\"gen\":" << rr.gen_n << ",\"forwards\":" << rr.forwards
              << ",\"drafted\":" << rr.drafted << ",\"accepted\":" << rr.accepted << ",\"rollbacks\":" << rr.rollbacks
              << ",\"sampled\":" << rr.sampled << ",\"decode_ms\":" << rr.decode_ms << ",\"hash_forwards\":" << rh.forwards
              << ",\"hash_mismatched\":" << rh.mismatched << ",\"state_checks\":" << rh.state_checks
              << ",\"state_mismatched\":" << rh.state_mismatched << ",\"same_steps_as_mtp_loop\":" << (same ? "true" : "false") << "}";
            std::fprintf(stderr, "tp_ar: replay over the TP backend: %lld forwards, %lld drafted, %lld accepted, %lld rollbacks; %lld hash mismatches, %lld state mismatches; same as the MTP loop: %s\n",
                         (long long)rr.forwards, (long long)rr.drafted, (long long)rr.accepted, (long long)rr.rollbacks,
                         (long long)rh.mismatched, (long long)rh.state_mismatched, same ? "yes" : "no");
        }
        const Qwen4ExpSession::PleStats ps = ses.ple_stats();
        J << ",\"mtp_on\":" << (use_mtp && drv.has_mtp() ? "true" : "false") << ",\"mtp_draft\":" << mtp_draft
          << ",\"mtp_margin\":" << margin << ",\"mtp_vocab\":" << mtp_vocab << ",\"mtp_draft_q4\":"
          << (draft_q4 ? "true" : "false") << ",\"draft_q4_make_s\":" << draft_q4_s << ",\"mtp_reject_forward\":"
          << (reject_forward ? "true" : "false") << ",\"ple_wait_s\":" << ps.wait_seconds << ",\"ple_waits\":" << ps.waits;
        if (world > 1) {
            const std::vector<Qwen4ExpSession::PleStats> es = drv.executor_stats();
            J << ",\"ple_wait_s_rank1\":" << es.at(0).wait_seconds << ",\"ple_waits_rank1\":" << es.at(0).waits
              << ",\"ple_wait_max_s_rank1\":" << es.at(0).wait_max_seconds;
        }
        J << ",\"canaries_ok\":" << (ses.canaries_ok() ? "true" : "false") << ",\"exchanges_total\":" << exch() << "}\n";
        drv.finish();
        if (comm) comm->barrier();
        std::ofstream o(out);
        o << J.str();
        std::fprintf(stderr, "tp_ar: wrote %s\n", out.c_str());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "tp_ar: fatal: %s\n", e.what());
        return 1;
    }
}
