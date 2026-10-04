#pragma once

// RMSNorm kernels (the model has two different RMSNorms).
//
// Activations are F32 or BF16 (BF16 stored as raw uint16_t bits, RNE
// rounding); math is FP32 and the casts back to the activation dtype
// happen at the same points as in the reference, so BF16 output lands
// within one BF16 rounding step of it. Weights are FP32 on device.
//
// Both operate on `rows` independent rows of length `d`, contiguous.
// In place is allowed (out == x): each element is read and written by the
// same thread, and all reads of the row finish before any write.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::kernels {

enum class Act { F32, BF16 };
const char *act_name(Act a);

// out = cast(x * rsqrt(mean(x^2) + eps) * (1 + w)); w has d elements.
// Used for input/post-attention/final norms (d = hidden) and q_norm/k_norm
// (rows = tokens * heads, d = head_dim).
void rmsnorm_zc(const void *x, const float *w, void *out, int64_t rows, int64_t d, float eps, Act act,
                hipStream_t stream);

// Grouped zero-centred RMSNorm:
// x is [tokens, groups * d]; each d-wide slice is normalized on its own and
// scaled by its own slice of w (groups * d elements):
//   out[t, g*d + j] = cast(x * rsqrt(mean over the slice of x^2 + eps) * (1 + w[g*d + j])).
// qwen4_exp: the hyper-connection hc_norm and the PLE's norm_key /
// norm_query / norm_conv (groups = 4 streams, d = 2560). Same cast points as
// rmsnorm_zc; groups = 1 is rmsnorm_zc.
void group_rmsnorm_zc(const void *x, const float *w, void *out, int64_t tokens, int64_t groups, int64_t d, float eps,
                      Act act, hipStream_t stream);

// The GDN output gate's activation, per model: Qwen3.5 uses silu(z), qwen4_exp
// sigmoid(z) (its config's output_gate_type). Required
// at every call - no default, so a target call site can't silently get
// Qwen3.5's gate.
enum class GateAct { SiLU, Sigmoid };
const char *gate_act_name(GateAct g);

// out = cast(w * cast(x * rsqrt(mean(x^2) + eps)) * gate(z)); z has x's dtype. GDN output norm: rows = tokens *
// value heads, d = 128. z_heads = 0: z has x's shape (contiguous rows). z_heads > 0: z's rows come z_heads to a
// token, each token's z_heads * d values contiguous, tokens z_token_stride elements apart - z read in place from
// inside the in_proj rows (no copy); rows must be a multiple of z_heads.
void rmsnorm_gated(const void *x, const void *z, const float *w, void *out, int64_t rows, int64_t d, float eps,
                   GateAct gate, Act act, hipStream_t stream, int64_t z_heads = 0, int64_t z_token_stride = 0);

}  // namespace strix::kernels
