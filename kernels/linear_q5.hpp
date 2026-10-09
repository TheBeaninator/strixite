#pragma once
// Q5-weight linear layer: y[M, N] = x[M, K] * dequant(W)^T, W in the Q5 layout (formats/q5.hpp), x/y F32 or BF16
// activations, FP32 accumulation. linear_q6's kernel (kernels/linear_q56.hip) with a 1-bit high plane: each lane
// loads 32 codes as their low nibbles (16 bytes, Q4's packing) and high bits (4 bytes), always within one group
// (G >= 32), and computes s * sum(x*q) + m * sum(x) per chunk; 1-8 waves per output row (split-K, ~2048 codes a
// wave as Q4), partial sums meet in LDS; M == 1 has its own instantiation.
#include "kernels/norm.hpp"  // Act
#include "runtime/q5_device.hpp"
#include "common/hip_runtime.hpp"
#include <cstdint>
namespace strix::kernels {
// Requires: q 16-byte aligned; K % 128 == 0 (the format's: every row's planes stay 16-byte aligned); x 16-byte
// aligned; y must not overlap x. out_act == act, or BF16 x -> F32 y (the accumulator stored unrounded).
void linear_q5(const void *x, const Q5DeviceView &w, void *y, int64_t M, Act act, hipStream_t stream);
void linear_q5(const void *x, const Q5DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream);
}  // namespace strix::kernels
