// PF-6: the WMMA kernels at a K tail (K % 64 == 32, group size 32: a TP-4 rank's expert / shared-expert down,
// K = 640 / 4 = 160) against the FP32 kernels (MoeMath::F32, linear_q4 / linear_q8), and those against a host double
// reference, at K = 160, 320, 640. Every WMMA output is also hashed (FNV-1a over its bytes): K = 320 / 640 hashes are
// the same before and after PF-6 (the TAIL = false instantiations are the unchanged kernels).
//
//   test_wmma_ktail [--tol 1e-2] [--json out.json]
//
// Error: max |y - ref| / max |ref| over the output (ref: the FP32 kernel). Exit 1 if any case exceeds --tol.

#include "common/hip_check.hpp"
#include "formats/q4.hpp"
#include "formats/q8.hpp"
#include "kernels/linear_q4.hpp"
#include "kernels/linear_q8.hpp"
#include "kernels/linear_wmma.hpp"
#include "kernels/moe_grouped.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q4_device.hpp"
#include "runtime/q8_device.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <vector>

using namespace strix;
using kernels::Act;
using kernels::MoeMath;

namespace {

uint16_t f2bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint64_t fnv(const void *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t *>(p)[i]) * 1099511628211ull;
    return h;
}

struct Acts {  // values as the device sees them (BF16-rounded) + the device buffer in the chosen dtype
    std::vector<float> v;
    DeviceBuffer<uint8_t> d;
};
Acts make_acts(std::mt19937 &rng, int64_t n, float scale, Act act) {
    std::normal_distribution<float> nd(0.f, scale);
    Acts a;
    a.v.resize(n);
    std::vector<uint16_t> b(n);
    for (int64_t i = 0; i < n; ++i) b[i] = f2bf(nd(rng)), a.v[i] = bf2f(b[i]);
    if (act == Act::BF16) {
        a.d = DeviceBuffer<uint8_t>((size_t)n * 2, "x");
        STRIX_HIP_CHECK(hipMemcpy(a.d.get(), b.data(), n * 2, hipMemcpyHostToDevice), "x");
    } else {
        a.d = DeviceBuffer<uint8_t>((size_t)n * 4, "x");
        STRIX_HIP_CHECK(hipMemcpy(a.d.get(), a.v.data(), n * 4, hipMemcpyHostToDevice), "x");
    }
    return a;
}
std::vector<float> download(const DeviceBuffer<uint8_t> &d, int64_t n, Act act, uint64_t *hash) {
    std::vector<float> out(n);
    STRIX_HIP_CHECK(hipDeviceSynchronize(), "sync");
    if (act == Act::BF16) {
        std::vector<uint16_t> b(n);
        STRIX_HIP_CHECK(hipMemcpy(b.data(), d.get(), n * 2, hipMemcpyDeviceToHost), "y");
        for (int64_t i = 0; i < n; ++i) out[i] = bf2f(b[i]);
        if (hash) *hash = fnv(b.data(), n * 2);
    } else {
        STRIX_HIP_CHECK(hipMemcpy(out.data(), d.get(), n * 4, hipMemcpyDeviceToHost), "y");
        if (hash) *hash = fnv(out.data(), n * 4);
    }
    return out;
}
double rel_err(const std::vector<float> &y, const std::vector<double> &ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!std::isfinite(y[i])) return INFINITY;
        num = std::max(num, std::fabs((double)y[i] - ref[i])), den = std::max(den, std::fabs(ref[i]));
    }
    return den > 0 ? num / den : num;
}
std::vector<double> widen(const std::vector<float> &v) { return {v.begin(), v.end()}; }

struct Result {
    std::string name;
    double err = -1, ref_err = -1;  // WMMA vs FP32 kernel; FP32 kernel vs host double
    uint64_t hash = 0;
    std::string note;
};
std::vector<Result> results;
double worst = 0;

void report(Result r) {
    if (r.err >= 0) worst = std::max(worst, r.err);
    std::printf("%-44s err %-11.3e fp32-vs-host %-11.3e hash %016llx %s\n", r.name.c_str(), r.err, r.ref_err,
                (unsigned long long)r.hash, r.note.c_str());
    results.push_back(std::move(r));
}

const char *math_name(MoeMath m) { return m == MoeMath::WmmaF16 ? "wmma-f16" : m == MoeMath::WmmaBf16 ? "wmma-bf16" : "f32"; }

// The grouped experts at one K: gather (y[slot] = W_e x[slot / A], W [E*N, K]) and combine (y[m] = sum_a coef W_e
// h[m*A + a]), every math against MoeMath::F32.
void grouped_case(int64_t K, int64_t G, Act act, uint32_t seed) {
    const int64_t E = 8, N = 320, M = 40, A = 4;  // N: two full 128-row blocks + a partial one; 160 slots
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<float> wf((size_t)(E * N * K));
    for (auto &v : wf) v = nd(rng);
    const Q4Weight wq = quantize_q4(wf.data(), E * N, K, G);
    const std::vector<float> wd = dequantize_q4(wq);
    const Q4Device w = Q4Device::upload(wq, "w");
    std::vector<int32_t> ids(M * A);
    std::vector<float> coef(M * A);
    std::uniform_real_distribution<float> ud(0.05f, 0.5f);
    for (int64_t m = 0; m < M; ++m) {
        // skewed: expert 0 popular (multi-tile buckets), the rest distinct per token
        std::vector<int32_t> e = {0, 1, 2, 3, 4, 5, 6, 7};
        std::shuffle(e.begin() + 1, e.end(), rng);
        for (int64_t a = 0; a < A; ++a) ids[m * A + a] = e[a], coef[m * A + a] = ud(rng);
    }
    const auto ids_d = DeviceBuffer<int32_t>::from_host(ids, "ids");
    const auto coef_d = DeviceBuffer<float>::from_host(coef, "coef");
    const size_t wsb = kernels::moe_route_workspace_bytes(M, A, E);
    DeviceBuffer<int32_t> ws(wsb / 4 + 1, "ws");
    DeviceBuffer<uint32_t> err(3, "err");
    STRIX_HIP_CHECK(hipMemset(err.get(), 0, 12), "err");
    kernels::moe_group_routes(ids_d.get(), M, A, E, ws.get(), wsb, err.get(), nullptr);
    const int64_t es = act == Act::F32 ? 4 : 2;
    const std::string tag = "K=" + std::to_string(K) + " G=" + std::to_string(G) + " " + (act == Act::F32 ? "f32" : "bf16");

    // gather
    {
        Acts x = make_acts(rng, M * K, 1.0f, act);
        std::vector<double> host((size_t)(M * A * N));
        for (int64_t s = 0; s < M * A; ++s)
            for (int64_t n = 0; n < N; ++n) {
                double acc = 0;
                const float *wr = wd.data() + (ids[s] * N + n) * K, *xr = x.v.data() + (s / A) * K;
                for (int64_t k = 0; k < K; ++k) acc += (double)wr[k] * xr[k];
                host[s * N + n] = acc;
            }
        std::vector<float> ref;
        for (MoeMath math : {MoeMath::F32, MoeMath::WmmaBf16, MoeMath::WmmaF16}) {
            Result r;
            r.name = "grouped gather " + std::string(math_name(math)) + " " + tag;
            try {
                DeviceBuffer<uint8_t> y((size_t)(M * A * N * es), "y");
                kernels::linear_q4_experts_gather_grouped(x.d.get(), w.view(), E, ws.get(), wsb, M, A, y.get(), act, math,
                                                          nullptr);
                auto out = download(y, M * A * N, act, &r.hash);
                if (math == MoeMath::F32) ref = out, r.err = 0, r.ref_err = rel_err(out, host);
                else r.err = rel_err(out, widen(ref));
            } catch (const std::exception &e) {
                r.note = std::string("refused: ") + e.what();
                r.note = r.note.substr(0, 120);
            }
            report(r);
        }
    }
    // combine
    {
        Acts h = make_acts(rng, M * A * K, 0.5f, act);
        std::vector<double> host((size_t)(M * N));
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) {
                double v = 0;
                for (int64_t a = 0; a < A; ++a) {
                    double acc = 0;
                    const float *wr = wd.data() + (ids[m * A + a] * N + n) * K, *hr = h.v.data() + (m * A + a) * K;
                    for (int64_t k = 0; k < K; ++k) acc += (double)wr[k] * hr[k];
                    v += coef[m * A + a] * acc;
                }
                host[m * N + n] = v;
            }
        std::vector<float> ref;
        DeviceBuffer<float> partial((size_t)(M * A * N), "partial");
        for (MoeMath math : {MoeMath::F32, MoeMath::WmmaBf16, MoeMath::WmmaF16}) {
            Result r;
            r.name = "grouped combine " + std::string(math_name(math)) + " " + tag;
            try {
                DeviceBuffer<uint8_t> y((size_t)(M * N * es), "y");
                kernels::linear_q4_experts_combine_grouped(h.d.get(), w.view(), E, ids_d.get(), coef_d.get(), ws.get(),
                                                           wsb, partial.get(), M, A, y.get(), act, math, nullptr);
                auto out = download(y, M * N, act, &r.hash);
                if (math == MoeMath::F32) ref = out, r.err = 0, r.ref_err = rel_err(out, host);
                else r.err = rel_err(out, widen(ref));
            } catch (const std::exception &e) {
                r.note = std::string("refused: ") + e.what();
                r.note = r.note.substr(0, 120);
            }
            report(r);
        }
    }
    uint32_t he[3];
    STRIX_HIP_CHECK(hipMemcpy(he, err.get(), 12, hipMemcpyDeviceToHost), "err");
    STRIX_CHECK(he[0] == 0, "route error flagged");
}

// A dense Q4 / Q8 projection [N, K] at M rows: linear_q*_wmma against linear_q* (the non-WMMA kernel).
template <int BITS>
void dense_case(int64_t K, int64_t G, int64_t M, Act act, uint32_t seed) {
    const int64_t N = 2560 + 64;  // the hidden size + a partial 128-column block
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<float> wf((size_t)(N * K));
    for (auto &v : wf) v = nd(rng);
    Acts x = make_acts(rng, M * K, 0.5f, act);
    std::vector<float> wd;
    Q4Device w4;
    Q8Device w8;
    if constexpr (BITS == 4) {
        const Q4Weight q = quantize_q4(wf.data(), N, K, G);
        wd = dequantize_q4(q), w4 = Q4Device::upload(q, "w");
    } else {
        const Q8Weight q = quantize_q8(wf.data(), N, K, G);
        wd = dequantize_q8(q), w8 = Q8Device::upload(q, "w");
    }
    std::vector<double> host((size_t)(M * N));
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0;
            for (int64_t k = 0; k < K; ++k) acc += (double)wd[n * K + k] * x.v[m * K + k];
            host[m * N + n] = acc;
        }
    const int64_t es = act == Act::F32 ? 4 : 2;
    const std::string tag = "q" + std::to_string(BITS) + " K=" + std::to_string(K) + " G=" + std::to_string(G) +
                            " M=" + std::to_string(M) + " " + (act == Act::F32 ? "f32" : "bf16");
    std::vector<float> ref;
    for (int wm = 0; wm < 2; ++wm) {
        Result r;
            r.name = std::string(wm ? "dense wmma " : "dense f32 ") + tag;
        try {
            DeviceBuffer<uint8_t> y((size_t)(M * N * es), "y");
            if constexpr (BITS == 4) {
                if (wm) kernels::linear_q4_wmma(x.d.get(), w4.view(), y.get(), M, act, act, nullptr);
                else kernels::linear_q4(x.d.get(), w4.view(), y.get(), M, act, act, nullptr);
            } else {
                if (wm) kernels::linear_q8_wmma(x.d.get(), w8.view(), y.get(), M, act, act, nullptr);
                else kernels::linear_q8(x.d.get(), w8.view(), y.get(), M, act, act, nullptr);
            }
            auto out = download(y, M * N, act, &r.hash);
            if (!wm) ref = out, r.err = 0, r.ref_err = rel_err(out, host);
            else r.err = rel_err(out, widen(ref));
        } catch (const std::exception &e) {
            r.note = std::string("refused: ") + e.what();
            r.note = r.note.substr(0, 120);
        }
        report(r);
    }
}

}  // namespace

int main(int argc, char **argv) {
    double tol = 1e-2;
    std::string json;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tol" && i + 1 < argc) tol = std::atof(argv[++i]);
        else if (a == "--json" && i + 1 < argc) json = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--tol 1e-2] [--json out.json]\n", argv[0]);
            return 2;
        }
    }
    try {
        uint32_t seed = 1;
        for (int64_t K : {160, 320, 640}) {
            for (Act act : {Act::BF16, Act::F32}) grouped_case(K, 32, act, seed++);
            for (int64_t M : {12, 64, 200}) {
                dense_case<4>(K, 32, M, Act::BF16, seed++);
                dense_case<8>(K, 32, M, Act::BF16, seed++);
            }
            dense_case<8>(K, 32, 64, Act::F32, seed++);
        }
        // group size 64 at K = 320 / 640 (the single-node layouts' g64 classes): no tail, the unchanged kernels
        for (int64_t K : {320, 640}) {
            grouped_case(K, 64, Act::BF16, seed++);
            dense_case<8>(K, 64, 64, Act::BF16, seed++);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "test_wmma_ktail: %s\n", e.what());
        return 1;
    }
    double worst_k[3] = {0, 0, 0}, worst_ref = 0;
    int refused = 0;
    for (const auto &r : results) {
        if (!r.note.empty()) ++refused;
        worst_ref = std::max(worst_ref, r.ref_err);
        const int ki = r.name.find("K=160") != std::string::npos ? 0 : r.name.find("K=320") != std::string::npos ? 1 : 2;
        worst_k[ki] = std::max(worst_k[ki], r.err);
    }
    std::printf("max rel err: K=160 %.3e  K=320 %.3e  K=640 %.3e  (tol %.1e); fp32 kernel vs host %.3e; refused %d\n",
                worst_k[0], worst_k[1], worst_k[2], tol, worst_ref, refused);
    if (!json.empty()) {
        FILE *f = std::fopen(json.c_str(), "w");
        STRIX_CHECK(f != nullptr, "cannot write ", json);
        std::fprintf(f, "{\"unit_max_rel_err\": %.6e, \"unit_tol\": %.6e, \"unit_err_by_k\": {\"160\": %.6e, \"320\": %.6e, "
                        "\"640\": %.6e}, \"unit_fp32_vs_host\": %.6e, \"unit_refused\": %d, \"cases\": [",
                     worst, tol, worst_k[0], worst_k[1], worst_k[2], worst_ref, refused);
        for (size_t i = 0; i < results.size(); ++i)
            std::fprintf(f, "%s{\"name\": \"%s\", \"err\": %.6e, \"fp32_vs_host\": %.6e, \"hash\": \"%016llx\", \"refused\": %s}",
                         i ? ", " : "", results[i].name.c_str(), results[i].err, results[i].ref_err,
                         (unsigned long long)results[i].hash, results[i].note.empty() ? "false" : "true");
        std::fprintf(f, "]}\n");
        std::fclose(f);
    }
    return worst <= tol && refused == 0 ? 0 : 1;
}
