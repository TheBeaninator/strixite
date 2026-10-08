#include "formats/q4_from_q8.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strix {

namespace {

// Rows per work item: a worker holds one item's FP32 values at a time (1024 x K floats - 10 MB at K = 2560), so the
// transient memory doesn't grow with the row count.
constexpr int64_t kRowsPerItem = 1024;

// Rows [r0, r1) of src as their own Q8 weight (row-major: a contiguous slice of every array).
Q8Weight q8_rows(const Q8Weight &src, int64_t r0, int64_t r1) {
    const int64_t gpr = src.K / src.G;  // groups per row
    Q8Weight w;
    w.N = r1 - r0, w.K = src.K, w.G = src.G;
    w.q.assign(src.q.begin() + r0 * src.K, src.q.begin() + r1 * src.K);
    w.scale.assign(src.scale.begin() + r0 * gpr, src.scale.begin() + r1 * gpr);
    w.minv.assign(src.minv.begin() + r0 * gpr, src.minv.begin() + r1 * gpr);
    return w;
}

}  // namespace

Q4Weight quantize_q4_from_q8(const Q8Weight &src, int64_t G, int threads) {
    check_q8(src, "quantize_q4_from_q8: src");
    STRIX_CHECK(q4_group_size_supported(G), "quantize_q4_from_q8: Q4 group size ", G, ", expected 32, 64 or 128");
    STRIX_CHECK(src.K % G == 0, "quantize_q4_from_q8: K = ", src.K, " is not a multiple of the Q4 group size ", G);
    STRIX_CHECK(threads >= 1 && threads <= 64, "quantize_q4_from_q8: threads ", threads, ", expected 1..64");

    const int64_t items = (src.N + kRowsPerItem - 1) / kRowsPerItem;
    std::vector<Q4Weight> parts((size_t)items);
    std::atomic<int64_t> next{0};
    std::mutex err_mu;
    std::string first_err;  // the first worker failure, rethrown on this thread after the join
    auto worker = [&] {
        for (;;) {
            const int64_t i = next.fetch_add(1);
            if (i >= items) return;
            {
                std::lock_guard<std::mutex> lk(err_mu);
                if (!first_err.empty()) return;  // another worker failed: stop taking work
            }
            const int64_t r0 = i * kRowsPerItem, r1 = std::min(src.N, r0 + kRowsPerItem);
            try {
                const Q8Weight rows = q8_rows(src, r0, r1);
                const std::vector<float> f = dequantize_q8(rows);
                parts[(size_t)i] = quantize_q4(f.data(), rows.N, rows.K, G);
            } catch (const std::exception &e) {
                std::lock_guard<std::mutex> lk(err_mu);
                if (first_err.empty())
                    first_err = "rows [" + std::to_string(r0) + ", " + std::to_string(r1) + "): " + e.what();
                return;
            }
        }
    };
    const int n = (int)std::min<int64_t>(threads, items);
    std::vector<std::thread> pool;
    pool.reserve((size_t)n);
    try {
        for (int t = 0; t < n; ++t) pool.emplace_back(worker);
    } catch (const std::exception &e) {
        // Thread creation failed: stop the started workers, join them, then report.
        next.store(items);
        for (std::thread &th : pool) th.join();
        STRIX_FAIL("quantize_q4_from_q8: starting worker ", pool.size() + 1, " of ", n, " failed: ", e.what());
    }
    for (std::thread &th : pool) th.join();
    STRIX_CHECK(first_err.empty(), "quantize_q4_from_q8: ", first_err);

    Q4Weight out;
    for (const Q4Weight &p : parts) append_rows_q4(out, p, "quantize_q4_from_q8");
    STRIX_CHECK(out.N == src.N && out.K == src.K && out.G == G, "quantize_q4_from_q8: result [", out.N, ", ", out.K,
                "] G ", out.G, ", expected [", src.N, ", ", src.K, "] G ", G);
    check_q4(out, "quantize_q4_from_q8: result");
    return out;
}

}  // namespace strix
