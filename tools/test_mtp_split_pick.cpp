// The split draft pick (kernels::mtp_pick_part per rank + mtp_pick_merge) equals kernels::mtp_pick over the
// whole draft vocabulary - best id, best value bits, NaN flag, and the runner-up value (zero sign canonicalized: the
// only order-dependent bit) - for N = 2 and 4 over random logits and the hard cases (ties across the rank boundaries,
// NaN, +-0, all equal, one rank all NaN, an empty last part). Exit 1 on any mismatch.
#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "kernels/embedding.hpp"
#include "kernels/mtp_pick.hpp"
#include "runtime/device_buffer.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strix;

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static uint32_t canon(float f) { return f == 0.0f ? 0u : bits(f); }

int main() {
    const int64_t V = 65536;
    DeviceBuffer<float> d_logits((size_t)V, "logits");
    DeviceBuffer<uint32_t> d_err(9, "err");
    DeviceBuffer<uint8_t> d_emb(sizeof(kernels::EmbeddingError), "emb");
    DeviceBuffer<uint8_t> d_pick(sizeof(kernels::MtpPick) * 2, "pick");
    DeviceBuffer<uint8_t> d_parts(sizeof(kernels::MtpPart) * 8, "parts");
    STRIX_HIP_CHECK(hipMemset(d_err.get(), 0, 9 * 4), "err");
    STRIX_HIP_CHECK(hipMemset(d_emb.get(), 0, sizeof(kernels::EmbeddingError)), "emb");
    auto *emb = reinterpret_cast<const kernels::EmbeddingError *>(d_emb.get());
    auto *pick = reinterpret_cast<kernels::MtpPick *>(d_pick.get());
    auto *parts = reinterpret_cast<kernels::MtpPart *>(d_parts.get());
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    int fails = 0, cases = 0;
    auto run = [&](const std::vector<float> &x, int64_t n, const char *what) {
        STRIX_HIP_CHECK(hipMemcpy(d_logits.get(), x.data(), n * 4, hipMemcpyHostToDevice), "logits");
        kernels::mtp_pick(d_logits.get(), n, d_err.get(), d_err.get() + 3, d_err.get() + 6, emb, pick, nullptr);
        for (int N : {2, 4}) {
            const int64_t Dr = V / N;
            for (int r = 0; r < N; ++r) {
                const int64_t base = r * Dr, m = std::max<int64_t>(0, std::min(Dr, n - base));
                kernels::mtp_pick_part(d_logits.get() + base, m, base, parts + r, nullptr);
            }
            kernels::mtp_pick_merge(parts, N, d_err.get(), d_err.get() + 3, d_err.get() + 6, emb, pick + 1, nullptr);
            kernels::MtpPick h[2];
            STRIX_HIP_CHECK(hipMemcpy(h, pick, sizeof h, hipMemcpyDeviceToHost), "pick");
            const bool same = h[0].nan == h[1].nan && (h[0].nan || (h[0].best == h[1].best && bits(h[0].best_v) == bits(h[1].best_v) &&
                                                                    canon(h[0].second_v) == canon(h[1].second_v)));
            ++cases;
            if (!same) {
                ++fails;
                std::printf("FAIL %s N=%d n=%lld: whole (%d %g %g nan %u) split (%d %g %g nan %u)\n", what, N, (long long)n,
                            h[0].best, h[0].best_v, h[0].second_v, h[0].nan, h[1].best, h[1].best_v, h[1].second_v, h[1].nan);
            }
        }
    };
    std::vector<float> x((size_t)V);
    for (int t = 0; t < 200; ++t) {  // random logits, some with planted maxima / ties
        for (float &v : x) v = nd(rng);
        const int64_t n = t % 4 == 0 ? V - (rng() % 5000) : V;  // a smaller draft vocabulary: an empty / short last part
        if (t % 3 == 0) { const int64_t a = rng() % n, b = rng() % n; x[a] = x[b] = 50.0f; }  // tie for the best
        if (t % 5 == 0) { const int64_t a = rng() % n; x[a] = 60.0f; x[(a + V / 4) % n] = 60.0f; }  // tie across ranks
        run(x, n, "random");
    }
    for (int N : {2, 4}) {  // a tie straddling each rank boundary
        for (float &v : x) v = nd(rng);
        for (int r = 1; r < N; ++r) { x[r * (V / N) - 1] = 40.0f; x[r * (V / N)] = 40.0f; }
        run(x, V, "boundary tie");
    }
    for (float &v : x) v = 1.25f; run(x, V, "all equal");
    for (float &v : x) v = nd(rng); x[123] = NAN; run(x, V, "one NaN");
    for (float &v : x) v = NAN; run(x, V, "all NaN");
    for (int64_t i = 0; i < V; ++i) x[i] = i < V / 4 ? NAN : nd(rng); run(x, V, "rank 0 all NaN");
    for (float &v : x) v = -1.0f; x[5] = 0.0f; x[V - 5] = -0.0f; run(x, V, "+0 / -0");
    for (float &v : x) v = -INFINITY; x[777] = 2.0f; run(x, V, "one finite");
    x.assign((size_t)V, 0.0f); run(x, 1, "one logit");
    std::printf("{\"tool\":\"test_mtp_split_pick\",\"cases\":%d,\"failures\":%d,\"pass\":%s}\n", cases, fails, fails ? "false" : "true");
    return fails ? 1 : 0;
}
