// tp_ar (strixite-tp2 ST-2): the TP-N autoregressive forward checked and timed - and the same measurements on the whole
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
// MTP (ST-3; --mtp 1: rank 0's session keeps the draft head, the executors' don't): per depth, after AR,
//   verify_k  (--verify-k 1) forward_verify(k) + drop_verify for k = 2..6, mirrored, timed
//   ar_gen    --gen greedy T = 1 forwards (candidates): the continuation the MTP loop replays
//   mtp       the engine's draft / verify / keep loop over ar_gen teacher-forced (strix_bench's, on TpDriver: PF-1, or
//             --mtp-reject-forward 1): steps, forwards, drafted, accepted, rollbacks, draft / verify / single ms, t/s,
//             exchanges per token; with --hash 1 first one pass with every forward's / verify's replicated activations
//             and every keep / drop's replicated state compared across ranks ("verify_hash"), then the timed pass.
// Rank 1's PLE stats come back at the end (kStats): ple_wait_s_rank1.
//
// --shadow 1 (world 1, ST-3 step 2, design 3.2 check 1): the mirror-shadow test - one process, one weight copy, two
// sessions (the driver's with MTP, an executor's without) joined by an in-process control channel (TpLoopback), the
// executor on a thread. Both are the whole model, so after every mirrored forward / verify their logits rows and
// replicated activations must be bit-identical, and after every keep / drop / restore / reset their replicated
// state. Drives the real MTP loop over --shadow-tokens generated tokens (PF-1 and reject-forward rounds alternating)
// with injected drops, mid-run snapshot restores and rank-0 failures followed by reset().

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "kernels/linear_q8.hpp"
#include "kernels/logits_topk.hpp"
#include "runtime/qwen4exp.hpp"
#include "runtime/tp_comm.hpp"
#include "runtime/tp_mirror.hpp"
#include "serve/tokenizer.hpp"

#include <algorithm>
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
        if (reject_forward) {  // pre-PF-1: forward the rejected position's token alone
            drv.prefetch_ple({gen[(size_t)v]}, 0);
            drv.want_candidates(n_valid);
            const double f0 = now_ms();
            drv.forward({gen[(size_t)v]}, 1);
            r.single_ms.push_back(now_ms() - f0);
            ++r.forwards;
            i = v + 1;
        } else {
            i = v;  // PF-1: gen[v] is the next step's first token (known, not yet forwarded)
        }
        r.step_ms.push_back(now_ms() - st0);
    }
    r.decode_ms = now_ms() - r0;
    return r;
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
        const int64_t mtp_draft = a.num("mtp-draft", 5), mtp_vocab = a.num("mtp-vocab", 65536), gen_n = a.num("gen", 256);
        const double margin = a.real("mtp-margin", 2.0);
        const bool reject_forward = a.num("mtp-reject-forward", 0) != 0, verify_k = a.num("verify-k", 1) != 0;
        const int64_t verify_reps = a.num("verify-reps", 8);
        STRIX_CHECK(!shadow || world == 1, "tp_ar: --shadow runs the whole model in one process (world 1)");
        const Qwen4ExpModel model(weights, ngram, kernels::Act::BF16, false, cache_rows, yarn,
                                  TpConfig{world, rank, std::max<int64_t>(1, mtp_vocab)});
        const double load_s = (now_ms() - t0) / 1000;
        const Qwen4ExpDims &Dm = model.dims();
        std::fprintf(stderr, "tp_ar: rank %d of %d: %.2f GiB of weights (%.2f GiB read) loaded in %.1f s\n", rank, world,
                     (double)model.weights().data_bytes() / (1ull << 30), (double)model.weights().bytes_read() / (1ull << 30),
                     load_s);
        // ST-3: rank 0 (or the whole model) keeps the MTP head with --mtp 1; executors never draft (no head loaded).
        const bool ses_mtp = (use_mtp || shadow) && rank == 0;
        STRIX_CHECK(!ses_mtp || model.has_mtp(), "tp_ar: --mtp 1 but the weights have no MTP head");
        const int64_t cap = shadow ? ((a.num("shadow-prefix", 4096) + a.num("shadow-tokens", 2048) + 1024) / 4096 + 1) * 4096
                                   : capacity + (use_mtp ? ((gen_n + 4095) / 4096) * 4096 : 0);
        Qwen4ExpSession ses(model, cap, chunk, PrefillMath::WmmaBf16, ses_mtp);
        if (ses_mtp) ses.set_mtp_vocab(mtp_vocab);
        const bool f32_mixer = a.num("tp-f32-mixer", 0) != 0;  // experiment: FP32 mixer partials (ST-2 KL study)
        if (f32_mixer && world > 1) ses.set_tp_f32_mixer(true);
        std::unique_ptr<TpComm> comm;
        if (world > 1) {
            TpCommConfig cc;
            cc.rank = rank;
            cc.peers = split(a.get("tp-peers"), ',');
            STRIX_CHECK((int)cc.peers.size() == world, "tp_ar: --tp-peers lists ", cc.peers.size(), " addresses for world ", world);
            cc.dev = a.get("tp-dev", "");  // empty: the mlx5 device with a RoCE v2 GID for this rank's rail-0 address
            cc.port = (int)a.num("tp-port", 18600);
            cc.max_bytes = (size_t)(chunk * Dm.d * (f32_mixer ? 4 : 2));
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
                             (long long)round, pf1 ? "PF-1" : "reject-forward", (long long)done, (long long)h.forwards,
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
            return exe_err.empty() ? 0 : 1;
        }

        TpDriver drv(ses, comm.get());
        std::ostringstream J;
        J << "{\"tool\":\"tp_ar\",\"weights\":\"" << weights << "\",\"ngram\":\"" << ngram << "\",\"tp_world\":" << world
          << ",\"weights_gib\":" << (double)model.weights().data_bytes() / (1ull << 30)
          << ",\"read_gib\":" << (double)model.weights().bytes_read() / (1ull << 30) << ",\"load_s\":" << load_s
          << ",\"capacity\":" << cap << ",\"chunk\":" << chunk << ",\"yarn\":" << yarn
          << ",\"activations\":\"BF16\",\"partials\":\""
          << (world == 1 ? "n/a"
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
            pass(true, kls, agree, nll, nll_g);
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
                // the first 256 positions alone (ST-0a's golden span, when the weights are ST-0a's)
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
            const std::vector<int32_t> corpus = corpus_ids();
            STRIX_CHECK((int64_t)corpus.size() > maxD + ar_steps + 64, "tp_ar: corpus has ", corpus.size(), " tokens");
            const bool mtp_on = use_mtp && drv.has_mtp();
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
                    std::vector<double> gm;
                    for (int64_t i = 1; i < gen_n; ++i) {
                        drv.want_candidates(nv);
                        const double s0 = now_ms();
                        drv.forward({gen.back()}, 1);
                        gm.push_back(now_ms() - s0);
                        gen.push_back(drv.candidates().cand[0].id);
                    }
                    J << ",\"ar_gen_ms\":" << js(stats(gm)) << ",\"ar_gen_tps\":" << 1000.0 / stats(gm).mean;
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
                    const uint64_t m0 = exch();
                    const MtpRun r = mtp_loop(drv, gen, nv, mtp_draft, margin, reject_forward);
                    const double ept = (double)(exch() - m0) / (double)gen_n;
                    J << ",\"mtp\":" << mtp_json(r, gen_n, ept);
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
                    drv.restore(snap);
                }
                J << "}";
                firstd = false;
                std::fprintf(stderr, "tp_ar: depth %lld: AR %.3f ms p50 (%.1f t/s), %.0f exchanges/forward, prefill %.1f s\n",
                             (long long)D, s.p50, 1000.0 / s.mean, epf, pf / 1000);
            }
            J << "}";
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
        const Qwen4ExpSession::PleStats ps = ses.ple_stats();
        J << ",\"mtp_on\":" << (use_mtp && drv.has_mtp() ? "true" : "false") << ",\"mtp_draft\":" << mtp_draft
          << ",\"mtp_margin\":" << margin << ",\"mtp_vocab\":" << mtp_vocab << ",\"mtp_reject_forward\":"
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
