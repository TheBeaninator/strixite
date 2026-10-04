#include "runtime/ple_hash.hpp"

#include "common/check.hpp"

namespace strix {

void check_ple_hash_params(const PleHashParams &p) {
    STRIX_CHECK(p.eos >= 0, "PLE hash: eos = ", p.eos, ", expected >= 0");
    STRIX_CHECK(p.table_rows >= 1, "PLE hash: table_rows = ", p.table_rows, ", expected >= 1");
    for (size_t j = 0; j < p.vocab.size(); ++j) {
        STRIX_CHECK(p.vocab[j] >= 1, "PLE hash: head ", j, " vocab size ", p.vocab[j], ", expected >= 1");
        STRIX_CHECK(p.offset[j] >= 0 && p.offset[j] + p.vocab[j] <= p.table_rows, "PLE hash: head ", j,
                    " rows [", p.offset[j], ", ", p.offset[j] + p.vocab[j], ") outside the table's ", p.table_rows,
                    " rows");
    }
}

void ple_ngram_ids(const PleHashParams &p, const int64_t *ids, int64_t T, PleHistory &hist, int64_t *out) {
    STRIX_CHECK(ids != nullptr && out != nullptr, "ple_ngram_ids: null pointer (ids=", (const void *)ids, ", out=",
                (void *)out, ")");
    STRIX_CHECK(T >= 1, "ple_ngram_ids: T = ", T, ", expected >= 1");
    check_ple_hash_params(p);
    // Two's-complement wrap-around multiply (int64 overflow is undefined in C++; unsigned isn't).
    auto mul = [](int64_t a, int64_t b) { return (int64_t)((uint64_t)a * (uint64_t)b); };
    for (int64_t t = 0; t < T; ++t) {
        STRIX_CHECK(ids[t] >= 0, "ple_ngram_ids: token id ", ids[t], " at position ", t, " is negative");
        const int64_t s0 = ids[t], s1 = hist.prev[1], s2 = hist.prev[1] == p.eos ? p.eos : hist.prev[0];
        const int64_t h2 = mul(s0, p.multipliers[0]) ^ mul(s1, p.multipliers[1]);
        const int64_t h3 = h2 ^ mul(s2, p.multipliers[2]);
        for (int j = 0; j < 16; ++j) {
            const int64_t h = j < 8 ? h2 : h3, v = p.vocab[(size_t)j];
            const int64_t m = ((h % v) + v) % v;
            out[t * 16 + j] = m + p.offset[(size_t)j];
        }
        hist.prev = {hist.prev[1], ids[t]};
    }
}

}  // namespace strix
