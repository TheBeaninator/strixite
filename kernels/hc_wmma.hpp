#pragma once

// Hyper-connection mixes (kernels/hc.hpp - the same math and weight layouts) for multi-token forwards on the gfx11
// matrix units: the prefill path (x / h rounded once to F16, the weights
// dequantized to F16 (code * scale + min, one rounding), FP32 accumulation over each GEMM's whole K range; the norms,
// gates and the stream combine stay FP32). Per-token kernels (hc_mix_down / _up) read each weight once
// per token; here once per 64-token tile.
//   down: inv[t, s] = 1 / rms(X_s[t]) first (a small pass); then per stream s a GEMM P_s = X_s W'_s^T over that
//         stream's d inputs into FP32 partials; then a[t, n] = sum_s inv[t, s] P_s[t, n], h = silu(a / H) for
//         n < r, w_in = 2 sigmoid(a / H) for the H inject rows.
//   up:   a block takes 32 output positions j for all H streams (128 rows of W_up, chunk-major as stored): z =
//         h W_up^T, u[t, j] = (1/H) sum_s sigmoid(z_s) * X[t, s d + j] * inv[t, s] * (1 + norm_w[s d + j]).
// Requires H == 4, d % 64 == 0 (d = 2560), r % 64 == 0 (320).

#include "kernels/norm.hpp"  // Act
#include "runtime/q4_device.hpp"
#include "runtime/q8_device.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::kernels {

// Bytes of FP32 partials hc_mix_down_wmma needs for T tokens and R = r (+ H) weight rows.
size_t hc_wmma_workspace_bytes(int64_t T, int64_t H, int64_t R);

// As hc_mix_down (w: folded [r (+ H), H*d], Q4 or Q8 row-major), plus the partials workspace (device). inv_ready:
// inv already holds X's inverse RMS (hc_inject_inv computed it) - not recomputed.
void hc_mix_down_wmma(const void *x, const Q4DeviceView &w, int64_t T, int64_t H, int64_t d, int64_t r, bool inject,
                      float eps, float *h, float *w_in, float *inv, float *ws, size_t ws_bytes, Act act,
                      hipStream_t stream, bool inv_ready = false);
void hc_mix_down_wmma(const void *x, const Q8DeviceView &w, int64_t T, int64_t H, int64_t d, int64_t r, bool inject,
                      float eps, float *h, float *w_in, float *inv, float *ws, size_t ws_bytes, Act act,
                      hipStream_t stream, bool inv_ready = false);

// hc_inject (X += w_in * y per stream) fused with the next mix's inverse RMS of the updated X into inv [T, H] -
// bit-identical to hc_inject then the WMMA mix's own inv, in one pass over X. H must be 4.
void hc_inject_inv(void *x, const float *w_in, const void *y, int64_t T, int64_t H, int64_t d, float eps, float *inv,
                   Act act, hipStream_t stream);

// As hc_mix_up (w_up: [H*d, r] chunk-major, Q4 or Q8).
void hc_mix_up_wmma(const float *h, const Q4ChunkMajorView &w_up, const void *x, const float *norm_w, const float *inv,
                    int64_t T, int64_t H, int64_t d, void *u, Act act, hipStream_t stream);
void hc_mix_up_wmma(const float *h, const Q8ChunkMajorView &w_up, const void *x, const float *norm_w, const float *inv,
                    int64_t T, int64_t H, int64_t d, void *u, Act act, hipStream_t stream);

}  // namespace strix::kernels
