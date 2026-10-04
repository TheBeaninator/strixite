#include "kernels/logits_topk.hpp"

#include "common/check.hpp"
#include "common/hip_check.hpp"

#include <climits>
#include <cmath>

namespace strix::kernels {

namespace {

// A workgroup reduces up to kGroup inputs (32 a thread) to their best kLogitCands without sequential rounds:
//  1. each thread takes the best of its 32 inputs (its head);
//  2. tau = the kLogitCands-th best head, by rank counting over the 256 heads (rank = heads better than mine);
//     at least kLogitCands inputs (those heads) are at least tau, so every input of the top kLogitCands is too;
//  3. an input at least tau sits in a thread whose head is at least tau - rank < kLogitCands, so at most
//     kLogitCands x 32 candidates (usually ~kLogitCands: mostly the heads); those threads append them;
//  4. each candidate's place in the output = how many candidates are better (the order is total: ties on the
//     value go to the lower id, ids are unique), so no sort.
// Levels repeat it on the workgroups' lists until one is left: 248,077 logits -> 31 workgroups -> 620 candidates -> 1.
// Placing every one of the kSlots slots against all of them (no compaction) took 148 us; earlier versions picked
// kLogitCands times in sequence (an arg-max a round): 79 us for one row across a workgroup
// (two barriers a round), 128 us a wave at a time (gfx1100) - the rounds' latency, not the 1 MB read.
constexpr int kThreads = 256;
constexpr int kPer = 32;                    // inputs a thread holds
constexpr int kGroup = kThreads * kPer;     // inputs a workgroup reduces
constexpr int kSlots = kLogitCands * kPer;  // candidate slots: kLogitCands threads x their inputs

// A candidate as one 64-bit key whose unsigned order is the candidates' order: the value's bits made monotone in the
// high word (a float's sign flips the rest), the id inverted in the low word (lower id = larger key on equal values).
// 0 = nothing (no real candidate maps there: a key's high word is 0 only for a NaN bit pattern, and NaNs are left
// out). Comparing (value, id) pairs with branches made the head ranking ~13 of ~24 us a level (deletion probe).
__device__ inline uint64_t key_of(float v, int32_t id) {
    const uint32_t b = v == 0.f ? 0u : __float_as_uint(v);  // -0 ties with +0 (a float compare's equal), as on the host
    const uint32_t hi = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((uint64_t)hi << 32) | (uint64_t)(0xFFFFFFFFu - (uint32_t)id);
}
// The candidate back, its value read from the row itself (the key lost -0's sign).
__device__ inline LogitCand cand_of(uint64_t k, const float *row) {
    if (k == 0) return LogitCand{-INFINITY, INT32_MAX};
    const int32_t id = (int32_t)(0xFFFFFFFFu - (uint32_t)k);
    return LogitCand{row[id], id};
}

// Inputs of workgroup g of row r: logits (ids = positions, < n_valid, NaN flagged and left out) or an earlier level's
// candidate lists (n_in a row). Writes kLogitCands sorted candidates to out[(r * groups + g) * kLogitCands ...],
// (-inf, INT32_MAX) past the number of inputs; for logits the group's NaN flag to nan_out[r * groups + g]; on the last
// level (groups == 1, nan != nullptr) the row's NaN flag to nan[r] (the OR of level 1's).
template <bool kLogits>
__global__ void __launch_bounds__(kThreads)
    reduce_kernel(const float *logits, int64_t ld, int n_valid, const LogitCand *in, int n_in, int groups,
                  LogitCand *out, uint32_t *nan_out, const uint32_t *nan1, int groups1, uint32_t *nan) {
    __shared__ uint64_t s_head[kThreads], s_cand[kSlots];
    __shared__ uint64_t s_tau;
    __shared__ int s_n;
    __shared__ uint32_t s_nan;
    const int t = (int)threadIdx.x, g = (int)blockIdx.x;
    const int64_t r = blockIdx.y;
    uint64_t key[kPer];
    uint32_t my_nan = 0;
    uint64_t head = 0;
    for (int p = 0; p < kPer; ++p) {
        const int j = g * kGroup + p * kThreads + t;  // coalesced
        key[p] = 0;
        if (kLogits) {
            if (j < n_valid) {
                const float x = logits[r * ld + j];
                if (x != x) my_nan = 1u;
                else key[p] = key_of(x, j);
            }
        } else if (j < n_in) {
            const LogitCand c = in[r * n_in + j];
            if (c.id != INT32_MAX) key[p] = key_of(c.v, c.id);
        }
        head = key[p] > head ? key[p] : head;
    }
    s_head[t] = head;
    if (t == 0) s_tau = 0, s_n = 0, s_nan = 0u;
    __syncthreads();
    if (my_nan) atomicOr(&s_nan, 1u);
    int rank = kThreads;
    if (head != 0) {
        rank = 0;
#pragma unroll 16
        for (int k = 0; k < kThreads; ++k) rank += s_head[k] > head ? 1 : 0;
        if (rank == kLogitCands - 1) s_tau = head;
    }
    __syncthreads();
    if (rank < kLogitCands) {  // real heads only (an empty one has rank kThreads)
        const uint64_t tau = s_tau;  // 0 when fewer than kLogitCands real heads: everything real is a candidate
        int n = 0;
        for (int p = 0; p < kPer; ++p) n += key[p] != 0 && key[p] >= tau ? 1 : 0;
        int at = atomicAdd(&s_n, n);  // appended in any order: places come from counting
        for (int p = 0; p < kPer; ++p)
            if (key[p] != 0 && key[p] >= tau) s_cand[at++] = key[p];
    }
    __syncthreads();
    const int nc = s_n;  // usually ~kLogitCands (mostly the heads), at most kSlots
    LogitCand *o = out + (r * groups + g) * kLogitCands;
    for (int c = t; c < nc; c += kThreads) {
        const uint64_t ck = s_cand[c];
        int place = 0;
        for (int k = 0; k < nc; ++k) place += s_cand[k] > ck ? 1 : 0;
        if (place < kLogitCands) o[place] = cand_of(ck, logits + r * ld);
    }
    if (t >= nc && t < kLogitCands) o[t] = LogitCand{-INFINITY, INT32_MAX};  // fewer inputs than candidates
    if (kLogits && t == 0) {
        nan_out[r * groups + g] = s_nan;
        if (nan && groups == 1) nan[r] = s_nan;  // the only level
    }
    if (!kLogits && nan && t == 0) {
        uint32_t f = 0;
        for (int c = 0; c < groups1; ++c) f |= nan1[r * groups1 + c];
        nan[r] = f;
    }
}

int64_t groups_of(int64_t n) { return (n + kGroup - 1) / kGroup; }

}  // namespace

size_t logits_topk_workspace_bytes(int64_t rows, int64_t n_valid) {
    STRIX_CHECK(rows >= 1 && n_valid >= 1 && n_valid <= kLogitsTopkMaxValid, "logits_topk_workspace_bytes: rows = ",
                rows, ", n_valid = ", n_valid, " (expected rows >= 1, n_valid 1..", kLogitsTopkMaxValid, ")");
    const int64_t g1 = groups_of(n_valid);
    // Two ping-pong candidate buffers (level 1's output is the largest), then the level-1 NaN flags.
    return 2 * (size_t)(rows * g1 * kLogitCands) * sizeof(LogitCand) + (size_t)(rows * g1) * sizeof(uint32_t);
}

void logits_topk(const float *logits, int64_t rows, int64_t ld, int64_t n_valid, LogitCand *out, uint32_t *nan,
                 void *workspace, size_t workspace_bytes, hipStream_t stream) {
    const char *fn = "logits_topk";
    STRIX_CHECK(logits && out && nan && workspace, fn, ": null pointer (logits=", (const void *)logits, ", out=",
                (void *)out, ", nan=", (void *)nan, ", workspace=", workspace, ")");
    STRIX_CHECK(rows >= 1 && rows <= 65535, fn, ": rows = ", rows, ", expected 1..65535");
    STRIX_CHECK(n_valid >= 1 && n_valid <= kLogitsTopkMaxValid, fn, ": n_valid = ", n_valid, ", expected 1..",
                kLogitsTopkMaxValid);
    STRIX_CHECK(ld >= n_valid, fn, ": row stride ld = ", ld, " < n_valid = ", n_valid);
    STRIX_CHECK(workspace_bytes >= logits_topk_workspace_bytes(rows, n_valid), fn, ": workspace ", workspace_bytes,
                " bytes, need ", logits_topk_workspace_bytes(rows, n_valid), " for ", rows, " rows of ", n_valid);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(workspace) % 8 == 0, fn, ": workspace must be 8-byte aligned");
    const int64_t g1 = groups_of(n_valid);
    auto *buf0 = static_cast<LogitCand *>(workspace);
    LogitCand *buf1 = buf0 + rows * g1 * kLogitCands;
    auto *nan1 = reinterpret_cast<uint32_t *>(buf1 + rows * g1 * kLogitCands);
    LogitCand *cur = g1 == 1 ? out : buf0;
    reduce_kernel<true><<<dim3((unsigned)g1, (unsigned)rows), kThreads, 0, stream>>>(
        logits, ld, (int)n_valid, nullptr, 0, (int)g1, cur, nan1, nullptr, 0, g1 == 1 ? nan : nullptr);
    STRIX_HIP_CHECK_LAUNCH(fn, " level 1 rows=", rows, " n_valid=", n_valid);
    for (int64_t groups = g1, level = 2; groups > 1; ++level) {
        const int64_t n_in = groups * kLogitCands, next = groups_of(n_in);
        LogitCand *dst = next == 1 ? out : (cur == buf0 ? buf1 : buf0);
        reduce_kernel<false><<<dim3((unsigned)next, (unsigned)rows), kThreads, 0, stream>>>(
            logits, ld, 0, cur, (int)n_in, (int)next, dst, nullptr, nan1, (int)g1, next == 1 ? nan : nullptr);
        STRIX_HIP_CHECK_LAUNCH(fn, " level ", level, " rows=", rows, " candidates=", n_in);
        cur = dst, groups = next;
    }
}

}  // namespace strix::kernels
