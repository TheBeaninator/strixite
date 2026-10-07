#pragma once

// The MTP draft's pick on the GPU: the largest logit (lowest index on a tie), the runner-up's value (the multiset
// second - equal to the best on a tie, -inf for one logit) and whether any logit is NaN - serve/sampler.hpp top2()'s
// result for non-NaN rows - written with the forward's error words into one small struct, so a draft step ends with
// one ~64-byte copy to the host instead of five copies and the whole draft vocabulary's logits (256 KB for 65536 ids)
// scanned on the CPU. A 2026-09-28 trace of the draft loop (tools/trace_gaps.py) had ~0.3 ms of each ~1.2 ms draft
// step idle around those copies and the host scan.

#include "kernels/embedding.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::kernels {

// Device and host layout (one copy): the pick, then the router / expert / attention error words (3 each, as
// read_back lays them out) and the embedding error slot.
struct MtpPick {
    int32_t best;
    float best_v, second_v;
    uint32_t nan;  // 1 if any logit was NaN (best / values are then meaningless)
    uint32_t err[9];
    uint32_t pad;
    EmbeddingError emb;
};
static_assert(sizeof(MtpPick) == 64, "MtpPick is copied as one 64-byte block");

// logits [n] FP32 (1 <= n <= 2^31 - 1); the error words and slot are the forward's device ones (read, not reset);
// out on the device. One block.
void mtp_pick(const float *logits, int64_t n, const uint32_t *router_err, const uint32_t *expert_err,
              const uint32_t *attn_err, const EmbeddingError *emb_err, MtpPick *out, hipStream_t stream);

// The draft head split across TP ranks by vocabulary rows: one rank's partial top-2 over its n local logits
// (global ids base .. base + n - 1; n = 0 gives the empty part), and the merge of N ranks' parts in rank order. The
// merge is mtp_pick's own (lowest id on a tie; the runner-up = max(winner's runner-up, loser's best); NaN OR-ed), so
// the merged pick equals mtp_pick over the whole draft vocabulary (up to the sign of a zero runner-up).
struct MtpPart {
    float b1, b2;
    int32_t i1;  // INT32_MAX: nothing seen
    uint32_t nan;
};
static_assert(sizeof(MtpPart) == 16, "MtpPart is gathered as 16 bytes per rank");
void mtp_pick_part(const float *logits, int64_t n, int64_t base, MtpPart *out, hipStream_t stream);
void mtp_pick_merge(const MtpPart *parts, int N, const uint32_t *router_err, const uint32_t *expert_err,
                    const uint32_t *attn_err, const EmbeddingError *emb_err, MtpPick *out, hipStream_t stream);

}  // namespace strix::kernels
