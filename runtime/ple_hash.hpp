#pragma once

// qwen4_exp PLE n-gram ids (PLE steps 1-3), on the host: the ids select rows of
// the SSD-backed n-gram table, so they're needed where the table reads are issued, and they're 16 integer
// hashes per token. Validated against HF's own id code (reference/golden_ple_ids.py fixture) and, through
// the row gather, against the L1.ple goldens.

#include <array>
#include <cstdint>

namespace strix {

// From the checkpoint's ple.ple_embedding buffers (never recomputed): the per-position multipliers, and per
// head (order-2 heads 0..7, then order-3 heads 8..15) the prime vocab size and the row offset.
struct PleHashParams {
    std::array<int64_t, 3> multipliers{};
    std::array<int64_t, 16> vocab{}, offset{};
    int64_t eos = 0;
    int64_t table_rows = 0;  // padded table rows; every id must be below it
};

// Throws unless the params are usable: vocab sizes >= 1, ids in range [offset, offset + vocab) within
// table_rows, eos >= 0.
void check_ple_hash_params(const PleHashParams &p);

// The last two tokens before the next call's first one (oldest first); a fresh sequence starts as {eos,
// eos} - the reference's padding before the start. Updated by ple_ngram_ids, so a prompt in pieces (and
// decode) gives the same ids as one call.
struct PleHistory {
    std::array<int64_t, 2> prev{};
    static PleHistory fresh(const PleHashParams &p) { return {{p.eos, p.eos}}; }
};

// ids [T] -> out [T, 16] row ids. Per token t with shifted tokens s0 = t's token, s1 = the one before,
// s2 = the one before that - except a shift never crosses an EOS (an EOS ends its segment, so after an EOS
// the older shifts are EOS): s1 = prev token, s2 = (prev token == EOS ? EOS : the one before). Order n hash
// = XOR over k < n of s_k * M_k (64-bit wrap-around), row = (hash mod vocab_j, non-negative) + offset_j.
void ple_ngram_ids(const PleHashParams &p, const int64_t *ids, int64_t T, PleHistory &hist, int64_t *out);

}  // namespace strix
