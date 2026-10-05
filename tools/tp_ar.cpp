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
        const Qwen4ExpModel model(weights, ngram, kernels::Act::BF16, false, cache_rows, yarn, TpConfig{world, rank});
        const double load_s = (now_ms() - t0) / 1000;
        const Qwen4ExpDims &Dm = model.dims();
        std::fprintf(stderr, "tp_ar: rank %d of %d: %.2f GiB of weights (%.2f GiB read) loaded in %.1f s\n", rank, world,
                     (double)model.weights().data_bytes() / (1ull << 30), (double)model.weights().bytes_read() / (1ull << 30),
                     load_s);
        Qwen4ExpSession ses(model, capacity, chunk, PrefillMath::WmmaBf16, false);
        const bool f32_mixer = a.num("tp-f32-mixer", 0) != 0;  // experiment: FP32 mixer partials (ST-2 KL study)
        if (f32_mixer && world > 1) ses.set_tp_f32_mixer(true);
        std::unique_ptr<TpComm> comm;
        if (world > 1) {
            TpCommConfig cc;
            cc.rank = rank;
            cc.peers = split(a.get("tp-peers"), ',');
            STRIX_CHECK((int)cc.peers.size() == world, "tp_ar: --tp-peers lists ", cc.peers.size(), " addresses for world ", world);
            cc.dev = a.get("tp-dev", "mlx5_0");
            cc.port = (int)a.num("tp-port", 18600);
            cc.max_bytes = (size_t)(chunk * Dm.d * (f32_mixer ? 4 : 2));
            cc.exchange_timeout_s = a.real("tp-timeout", 900);
            t0 = now_ms();
            comm = std::make_unique<TpComm>(cc);
            tp_attach(ses, model, *comm);
            std::fprintf(stderr, "tp_ar: rank %d of %d: communicator up in %.1f s\n", rank, world, (now_ms() - t0) / 1000);
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
        TpDriver drv(ses, comm.get());
        std::ostringstream J;
        J << "{\"tool\":\"tp_ar\",\"weights\":\"" << weights << "\",\"ngram\":\"" << ngram << "\",\"tp_world\":" << world
          << ",\"weights_gib\":" << (double)model.weights().data_bytes() / (1ull << 30)
          << ",\"read_gib\":" << (double)model.weights().bytes_read() / (1ull << 30) << ",\"load_s\":" << load_s
          << ",\"capacity\":" << capacity << ",\"chunk\":" << chunk << ",\"yarn\":" << yarn << ",\"mtp\":false"
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
            std::vector<int32_t> corpus;
            for (const std::string &f : split(a.get("corpus"), ',')) {
                const std::vector<int32_t> ids = tok.encode(read_file(f));
                corpus.insert(corpus.end(), ids.begin(), ids.end());
            }
            STRIX_CHECK((int64_t)corpus.size() > maxD + ar_steps + 64, "tp_ar: corpus has ", corpus.size(), " tokens");
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
                  << ",\"prefill_tps_cumulative\":" << (double)D / prefill_ms * 1000 << "}";
                firstd = false;
                std::fprintf(stderr, "tp_ar: depth %lld: AR %.3f ms p50 (%.1f t/s), %.0f exchanges/forward, prefill %.1f s\n",
                             (long long)D, s.p50, 1000.0 / s.mean, epf, pf / 1000);
            }
            J << "}";
        }
        const Qwen4ExpSession::PleStats ps = ses.ple_stats();
        J << ",\"ple_wait_s\":" << ps.wait_seconds << ",\"exchanges_total\":" << exch() << "}\n";
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
