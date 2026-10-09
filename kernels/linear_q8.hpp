#pragma once
// Q8-weight linear layer: y[M, N] = x[M, K] * dequant(W)^T, W in the Q8 layout (formats/q8.hpp), x/y F32 or BF16
// activations, FP32 accumulation. linear_q4's structure (kernels/linear_q4.hpp) at one code per byte: each lane
// loads 16 codes per 16-byte load (always within one group, G >= 32) and computes s * sum(x*q) + m * sum(x) per
// chunk; 1-8 waves per output row (split-K), partial sums meet in LDS; M == 1 has its own instantiation.
#include "kernels/norm.hpp"  // Act
#include "runtime/q8_device.hpp"
#include "common/hip_runtime.hpp"
#include <cstdint>
namespace strix::kernels {
// Requires: q 16-byte aligned (so K % 16 == 0 rows stay aligned: implied by K % G == 0); x 16-byte aligned;
// y must not overlap x. out_act == act, or BF16 x -> F32 y (the accumulator stored unrounded).
void linear_q8(const void *x, const Q8DeviceView &w, void *y, int64_t M, Act act, hipStream_t stream);
void linear_q8(const void *x, const Q8DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream);
}  // namespace strix::kernels
