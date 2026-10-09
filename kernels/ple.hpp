#pragma once

// qwen4_exp PLE injection on the GPU, after the n-gram ids (runtime/ple_hash,
// host) have selected the table rows e [T, 2560] and the merged [W_key; W_val] projection has produced
// kv [T, ld]: key at columns [0, H*d), value at [H*d, H*d + d). Two launches:
//
//   ple_gate: per (token, stream), from key, value and the streams X - with the norms folded into sums:
//     g_s = inv_k * inv_q * sum_j key_j X_j (1 + wk_j)(1 + wq_j) / sqrt(d)   (inv_* = the group inverse RMS)
//     g_s = sign(g_s) sqrt(max(|g_s|, 1e-6)),  sigma_s = sigmoid(g_s)
//     vn_s = sigma_s value * inv_v * (1 + wc),  inv_v = 1/sqrt(sigma_s^2 mean(value^2) + eps)
//   writes vn into the history hist [S + T, H*d] after its S = (K-1)*dilation state rows (copied there from
//   state_in), and sigma [T, H] F32.
//   ple_conv: X[t, c] += sigma * value + silu(sum_k conv[c, k] * vn[t - (K-1-k) * dilation, c]), one
//   rounding; state_out = the last S rows of hist (may equal state_in: hist is a separate buffer).
//
// Engine semantics: math FP32; key/value arrive in the activation dtype (linear outputs), vn and the state
// are stored in it, sigma F32, X rounded once per element. State: zeros for a fresh sequence (the reference
// pads with zeros).

#include "kernels/norm.hpp"  // Act

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

struct PleShape {
    int64_t T = 0, H = 4, d = 0;
    int64_t K = 4, dilation = 3;  // depthwise conv taps and dilation (S = (K-1)*dilation state rows)
    int64_t ld = 0;               // kv row stride (>= H*d + d)
    int64_t state_rows() const { return (K - 1) * dilation; }
};

void ple_gate(const void *kv, const void *x, const float *norm_key, const float *norm_query, const float *norm_conv,
              const void *state_in, const PleShape &sh, float eps, void *hist, float *sigma, Act act,
              hipStream_t stream);

void ple_conv(const void *hist, const float *sigma, const void *kv, const float *conv_w, const PleShape &sh, void *x,
              void *state_out, Act act, hipStream_t stream);

}  // namespace strix::kernels
