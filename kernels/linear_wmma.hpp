#pragma once

// Multi-token (prefill) Q4 / Q5 / Q6 / Q8 linear on the gfx11 matrix units: y[M, N] = x[M, K] * deq(W)^T, W row-major in
// formats/q4 / q5 / q6 / q8. A lower-precision path (kernels/wmma_gemm.hpp for the exact
// numerics): x rounded to BF16, the weights exact (raw codes on the matrix units, each group's scale and
// min applied in FP32), FP32 accumulation, y rounded once to its dtype - the prefill default (PrefillMath).
// Tiles of 64 rows of x x 128 rows of W, K in steps of 64 (so K % 64 == 0: every K of the target is).

#include "kernels/norm.hpp"  // Act
#include "runtime/q4_device.hpp"
#include "runtime/q5_device.hpp"
#include "runtime/q6_device.hpp"
#include "runtime/q8_device.hpp"

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

// x 16-byte aligned; y must not overlap x. out_act == act, or BF16 x -> F32 y.
void linear_q4_wmma(const void *x, const Q4DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream);
void linear_q5_wmma(const void *x, const Q5DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream);
void linear_q6_wmma(const void *x, const Q6DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream);
void linear_q8_wmma(const void *x, const Q8DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream);
// BF16 weights W [N, K] (row-major, 16-byte aligned rows: K % 64 == 0): y = bf16(x) * W^T, FP32 accumulation - the
// products are exact, only the summation order differs from linear_bf16w (the router: logits FP32).
void linear_bf16w_wmma(const void *x, const uint16_t *w, void *y, int64_t M, int64_t N, int64_t K, Act act, Act out_act,
                       hipStream_t stream);

}  // namespace strix::kernels
