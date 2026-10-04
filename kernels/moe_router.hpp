#pragma once

// MoE router for qwen4_exp: from the router
// logits, the top-k experts per token and their weights, softmax-renormalized
// over the k; plus the shared expert's slot with coefficient sigmoid(its gate
// logit). The output is what linear_q4_experts_gather / _combine consume: ids
// [M, A] int32 and coef [M, A] F32, A = k (+ 1 with the shared slot), the
// shared expert being the stacked experts' row block `shared_id`
// (formats/merge_plan: expert #E).
//
// Selection is on the logits (softmax is monotone, so it's the same set and
// order as selecting on the probabilities): largest first, a tie goes to the
// lower expert index. Weights: w_j = exp(l_j - l_max) / sum over the k of
// exp(l_i - l_max) - the same as the full softmax's p_j renormalized over the
// k (the full denominator cancels). All FP32; coef is stored FP32, so BF16
// runs don't round it (transformers' BF16 run does; engine precision is FP32
// math).
//
// A non-finite logit (routed or shared) fails that token: its ids are -1, its
// coef NaN, and err (>= 3 zeroed uint32 on the device) gets err[0] = 1,
// err[1] = the token, err[2] = the expert index (E for the shared logit);
// the first one wins. check_router_error() turns it into an exception (it
// synchronizes). Downstream, the experts kernels also flag id -1.

#include "kernels/norm.hpp"  // Act

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::kernels {

// logits: token m's E router logits at logits[m * ld .. m * ld + E) (ld >= E,
// so they can sit in a merged projection's output), F32 or BF16 (act).
// shared_logit: null for no shared slot (A = k); else token m's gate logit at
// shared_logit[m * shared_ld], same dtype, and slot k gets id shared_id.
// k in 1..min(E, 32); E in 1..1024.
void moe_router(const void *logits, int64_t ld, int64_t M, int64_t E, int64_t k, const void *shared_logit,
                int64_t shared_ld, int32_t shared_id, int32_t *ids, float *coef, Act act, uint32_t *err,
                hipStream_t stream);
void check_router_error(const uint32_t *err, const char *what);
// The same check on the 3 words already on the host (read back with the forward's one sync).
void report_router_error(const uint32_t e[3], const char *what);

// The shared expert's contribution when it isn't stacked with the routed experts (its format differs, so it
// runs as dense projections; the router then runs without a shared slot):
//   y[m, :] += sigmoid(gate_logit[m * gate_ld]) * s[m, :]
// y, s [M, N] in the activation dtype (row stride N), gate_logit FP32 (the router projection's output); FP32
// math, one rounding per element to the activation dtype. s must not overlap y. A non-finite gate logit fails
// that token like the router does: its y row becomes NaN and err (the router's error word) gets err[0] = 1,
// err[1] = the token, err[2] = gate_index (pass E, so check_router_error names the shared-expert gate); the
// first flag wins.
void moe_shared_add(void *y, const void *s, const float *gate_logit, int64_t gate_ld, int64_t M, int64_t N,
                    uint32_t gate_index, Act act, uint32_t *err, hipStream_t stream);

// moe_shared_add followed by the residual injection (hc_inject) in one launch - the MoE sublayer's end in decode /
// verify / the MTP head: y as moe_shared_add leaves it, then x[m, s*N + n] += w_in[m, s] * y[m, n] for the H = 4
// streams of x [M, H*N] (one FP32 fma, one rounding to the activation dtype - hc_inject's arithmetic on the stored
// y). Bit-identical to the two kernels; saves a launch and a kernel boundary per MoE sublayer. y, s and x must not
// overlap; the same non-finite gate handling.
void moe_shared_add_inject(void *y, const void *s, const float *gate_logit, int64_t gate_ld, int64_t M, int64_t N,
                           uint32_t gate_index, void *x, const float *w_in, int64_t H, Act act, uint32_t *err,
                           hipStream_t stream);

}  // namespace strix::kernels
