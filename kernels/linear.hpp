#pragma once

// BF16-weight linear layer: y[M, N] = x[M, K] * W^T, W [N, K] as raw BF16
// bits in the checkpoint's row-major [out, in] layout; x and y are F32 or
// BF16 activations; FP32 accumulation, one rounding at the end.
//
// Deliberately simple and untuned: the
// correctness baseline and the path for tensors that stay BF16. The tuned
// hot path will be the 4-bit kernel.

#include "kernels/norm.hpp"  // Act

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

// y must not overlap x (other blocks may still be reading x while y is written).
void linear_bf16w(const void *x, const uint16_t *w, void *y, int64_t M, int64_t N, int64_t K, Act act,
                  hipStream_t stream);
// Same with y in its own dtype: out_act == act, or BF16 x -> F32 y (outputs
// kept FP32 in a BF16 run - the router logits); the FP32
// accumulator is stored unrounded.
void linear_bf16w(const void *x, const uint16_t *w, void *y, int64_t M, int64_t N, int64_t K, Act act, Act out_act,
                  hipStream_t stream);

}  // namespace strix::kernels
