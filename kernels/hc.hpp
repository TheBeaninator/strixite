#pragma once

// qwen4_exp hyper-connections:
// the per-sublayer input mix over the H = 4 residual streams X [T, H*d], and
// the residual injection. Two launches per mix instead of five (norm, down,
// activations, up, combine) - ~97 mixes per token make launch count matter.
//
// Engine semantics: X is F32 or BF16 (the streams' activation dtype);
// everything inside is FP32 - the normed streams n are never stored, the
// low-rank h, w_in and the per-group inverse RMS are F32 - and u is rounded
// once to the activation dtype.
//
// Weights are Q4 or Q8 (the same code, one byte per
// code - an overload per format) and, for hc_mix_down, stored with
// the hc_norm weight folded in: rows [W_down * (1 + w) ; W_inj * (1 + w)]
// (per input column, before quantizing - measured to cost nothing). Then W_down n = W'_down n0 with n0 = X_s *
// inv_s the unweighted normalized streams, and each row's dot splits into
// per-stream partial dots that need only X: a = sum_s inv_s * (W'_s . X_s),
// with inv_s = 1/sqrt(mean(X_s^2) + eps) accumulated from the same X loads.

#include "kernels/norm.hpp"  // Act
#include "runtime/q4_device.hpp"
#include "runtime/q8_device.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::kernels {

// X [T, H*d] -> h [T, r] = silu(a / H) for the first r rows of w,
// w_in [T, H] = 2 sigmoid(a / H) for the next H rows (when inject; the final
// mixer has none), inv [T, H] = the per-stream inverse RMS (hc_mix_up needs it).
// w: Q4 [r (+ H), H*d], folded as above; H == 4; d % 32 == 0 (a 32-code load
// never straddles two streams); x 16-byte aligned. w_in may be null iff !inject.
void hc_mix_down(const void *x, const Q4DeviceView &w, int64_t T, int64_t H, int64_t d, int64_t r, bool inject,
                 float eps, float *h, float *w_in, float *inv, Act act, hipStream_t stream);
void hc_mix_down(const void *x, const Q8DeviceView &w, int64_t T, int64_t H, int64_t d, int64_t r, bool inject,
                 float eps, float *h, float *w_in, float *inv, Act act, hipStream_t stream);

// u[t, j] = (1/H) sum_s sigmoid(W_up[s*d + j] . h[t]) * n[t, s*d + j],
// n = X * inv[t, s] * (1 + norm_w) - the combine in the up projection's
// epilogue. w_up: Q4 [H*d, r], not folded, chunk-major (each thread owns a
// row; consecutive threads' rows are adjacent in that layout); norm_w: the
// hc_norm weight [H*d] FP32; r % 32 == 0, r <= 1024. u [T, d] in the
// activation dtype.
void hc_mix_up(const float *h, const Q4ChunkMajorView &w_up, const void *x, const float *norm_w, const float *inv,
               int64_t T, int64_t H, int64_t d, void *u, Act act, hipStream_t stream);
void hc_mix_up(const float *h, const Q8ChunkMajorView &w_up, const void *x, const float *norm_w, const float *inv,
               int64_t T, int64_t H, int64_t d, void *u, Act act, hipStream_t stream);

// Residual injection, in place: X[t, s*d + j] += w_in[t, s] * y[t, j], one
// rounding per element to the activation dtype. y must not overlap X.
void hc_inject(void *x, const float *w_in, const void *y, int64_t T, int64_t H, int64_t d, Act act,
               hipStream_t stream);

}  // namespace strix::kernels
