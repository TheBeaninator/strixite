#pragma once

// Gated DeltaNet (linear-attention) mixer pieces.
//
// gdn_front: steps 2-4 - from the merged in_proj output (formats/merge_plan:
// each row is [qkv (C) | z (Hv*Dv) | b (Hv) | a (Hv)], C = 2*Hk*Dk + Hv*Dv)
// to the delta rule's inputs:
//   qkv_out [T, C] = silu(causal depthwise conv, width 4, over time), laid
//       out q [Hk, Dk] | k [Hk, Dk] | v [Hv, Dv] per token;
//   beta [T, Hv] = sigmoid(b), g [T, Hv] = -exp(A_log) * softplus(a + dt_bias)
//       (softplus as torch: x for x > 20, else log1p(exp(x))).
// It ends where the goldens' `delta.*` are captured: q/k are NOT L2-normed or
// scaled here; that belongs to the delta-rule kernel. q/k stay per key head;
// with Hv > Hk (target: 16 key / 48 value heads) value head h pairs with key
// head h / (Hv/Hk), as transformers' repeat_interleave does.
//
// Conv state: [3, C] in the activation dtype, the last 3 conv inputs (raw
// qkv, pre-conv), oldest first - per layer, one conversation at a time.
// Each call reads conv_state_in as the
// inputs before token 0 and writes the last 3 inputs of (state ++ this call's
// tokens) to conv_state_out, so decode (T = 1) and a prompt fed in pieces
// give the same result as one call. The two may be the same buffer (in place,
// plain decode) or separate: MTP verify writes to a second buffer and keeps
// the old state for re-running only the accepted tokens. reset_state treats
// the old state as zeros (a fresh sequence) without reading it - no memset
// launch; conv_state_in may then be null.
//
// Precision (engine semantics): conv, silu and the gates in FP32 on the
// activation-dtype inputs; qkv_out rounded once to the activation dtype.
// beta and g are always FP32 (per-head scalars the delta rule consumes in
// FP32; g is FP32 in the reference too). A_log, dt_bias and the conv weights
// are FP32 on the device (A_log is FP32 in the checkpoint; the others are
// BF16 values, exact in FP32).

#include "kernels/norm.hpp"  // Act

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

constexpr int64_t kGdnConvWidth = 4;  // both models (linear_conv_kernel_dim)

struct GdnFrontShape {
    int64_t T = 0;                       // tokens this call (1 for decode)
    int64_t Hk = 0, Dk = 0, Hv = 0, Dv = 0;
    int64_t in_ts = 0;                   // row stride of the in_proj output (elements); qkv at column 0
    int64_t b_col = 0, a_col = 0;        // columns of b and a (Hv each) within a row
    int64_t channels() const { return 2 * Hk * Dk + Hv * Dv; }
};

// in_proj [T, in_ts] (activation dtype), conv_w [C, 4] FP32, conv_state_in /
// conv_state_out [3, C] (activation dtype; equal = in place), A_log / dt_bias
// [Hv] FP32 -> qkv_out [T, C] (activation dtype), beta / g [T, Hv] FP32.
// Buffers must not overlap, except conv_state_in == conv_state_out.
void gdn_front(const void *in_proj, const GdnFrontShape &s, const float *conv_w, const void *conv_state_in,
               void *conv_state_out, bool reset_state, const float *A_log, const float *dt_bias, void *qkv_out,
               float *beta, float *g, Act act, hipStream_t stream);

// Gated delta rule (GDN steps 5-6). From gdn_front's outputs:
// qkv [T, qkv_ts] (q [Hk, 128] | k [Hk, 128] | v [Hv, 128] from column 0),
// beta, g [T, Hv] FP32. Per value head h (key head h / (Hv/Hk)) and token t:
//   q, k L2-normalized (x * rsqrt(sum(x^2) + 1e-6)), q scaled by 1/sqrt(128);
//   S = exp(g) S;  S += k (beta (v - S^T k))^T;  out = S^T q.
// out [T, Hv, 128] in the activation dtype (rounded once) - the gated norm's
// input. Math and the state are FP32.
//
// State: [Hv, 128(k), 128(v)] FP32 per layer, read from state_in before token 0 and the state after the last
// token written to state_out. The same buffer = in place (each block owns a
// disjoint slice, read before it is written); separate buffers let MTP verify
// keep the old state. reset_state starts from zeros without reading state_in
// (which may then be null). A call reads and writes the state once whatever
// T is, so k+1 verify tokens cost one state round trip, like one decode step.
//
// Two paths, same results to FP32 rounding (not bit for bit):
//   PerToken - the recurrence token by token, state in registers; decode and
//              MTP verify.
//   Chunked  - 32-token chunks in matrix form (prefill): per chunk, with
//              cumulative log decay G, solve (I + A) U = beta (V - e^G K S0)
//              for the corrections U (A strictly lower, A_ts = beta_t
//              e^(G_t - G_s) k_t.k_s), then O = e^G Q S0 + (e^(G_t - G_s)
//              q_t.k_s)_{s <= t} U and S = e^(G_last) S0 + (e^(G_last - G_s)
//              k_s)^T U. Chunks start at the call's first token.
//   Auto picks PerToken for T < kDeltaRuleChunkMinT - for now every T: the
//   chunked kernel is slower at every measured size (bench_gdn, results.jsonl)
//   and is kept as the correct baseline to tune. MTP rollback is bit-exact
//   only when the verify and the re-run take the same path.
constexpr int64_t kGdnHeadDim = 128;  // Dk = Dv, both models
constexpr int64_t kDeltaRuleChunkMinT = INT64_MAX;  // chunked not faster yet (see above)
//   ChunkedWmma - the chunk form on the matrix units (one block per value head, the state in WMMA accumulators,
//              every product as BF16 hi + lo on both operands): prefill. Not bit-identical to the others (FP32
//              rounding-level differences), so MTP rollback must not mix it with PerToken for one step.
enum class DeltaRulePath { Auto, PerToken, Chunked, ChunkedWmma };

struct DeltaRuleShape {
    int64_t T = 0;
    int64_t Hk = 0, Hv = 0;  // Dk = Dv = kGdnHeadDim
    int64_t qkv_ts = 0;      // row stride of qkv (elements), >= (2*Hk + Hv) * 128
};

// Buffers must not overlap, except state_in == state_out.
void delta_rule(const void *qkv, const float *beta, const float *g, const DeltaRuleShape &s, const float *state_in,
                float *state_out, bool reset_state, void *out, Act act, hipStream_t stream,
                DeltaRulePath path = DeltaRulePath::Auto);

}  // namespace strix::kernels
