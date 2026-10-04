#pragma once

// Q4-weight linear layer: y[M, N] = x[M, K] * dequant(W)^T, W in the Q4
// layout (formats/q4.hpp), x/y F32 or BF16 activations, FP32 accumulation.
//
// Tuned for decode (small M, memory-bound): each output row is handled by
// 1-8 wavefronts of a block (split-K, more for longer K, partial sums meet in
// LDS); each lane loads 32 codes per 16-byte load (always within one group,
// since G >= 32), computing s * sum(x*q) + m * sum(x) per chunk - one FMA per
// weight - and a wave-shuffle reduction. M == 1 has its own instantiation
// (register use); larger M is processed 8 rows per pass - correct, but
// prefill is not what this kernel is built for.

#include "kernels/norm.hpp"  // Act
#include "runtime/q4_device.hpp"
#include "runtime/q5_device.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::kernels {

// Requires: q 16-byte aligned; x 16-byte aligned; K % 32 == 0 (implied by
// K % G == 0); y must not overlap x.
void linear_q4(const void *x, const Q4DeviceView &w, void *y, int64_t M, Act act, hipStream_t stream);
// Same with y in its own dtype: out_act == act, or BF16 x -> F32 y (outputs
// kept FP32 in a BF16 run - the router logits). The FP32
// accumulator is stored unrounded.
void linear_q4(const void *x, const Q4DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream);

// MoE experts. w holds E experts stacked as one [E*N, K] Q4 matrix (expert e
// = rows e*N .. e*N+N-1; the same bytes as E separate Q4 weights). ids [M, A]
// int32 on the device: the A active experts of each of M tokens. One launch
// covers every token and expert.
//   gather:  y[m, a, :] = x[m, :] * W[ids[m, a]]^T      x [M, K], y [M, A, N]
//   combine: y[m, :] = sum_a coef[m, a] * h[m, a, :] * W[ids[m, a]]^T
//            h [M, A, K], coef [M, A] F32, y [M, N]
// err: >= 3 zeroed uint32 on the device. An out-of-range id sets err[0] = 1,
// err[1] = the id, err[2] = its slot (m * A + a), and that output is NaN;
// check_expert_error() turns it into an exception (it synchronizes).
void linear_q4_experts_gather(const void *x, const Q4DeviceView &w, int64_t E, const int32_t *ids, int64_t M,
                              int64_t A, void *y, Act act, uint32_t *err, hipStream_t stream);
void linear_q4_experts_combine(const void *h, const Q4DeviceView &w, int64_t E, const int32_t *ids,
                               const float *coef, int64_t M, int64_t A, void *y, Act act, uint32_t *err,
                               hipStream_t stream);
// The same with Q5 experts (formats/q5.hpp rows: K % 128 == 0) - one kernel template with the Q4 ones.
void linear_q5_experts_gather(const void *x, const Q5DeviceView &w, int64_t E, const int32_t *ids, int64_t M,
                              int64_t A, void *y, Act act, uint32_t *err, hipStream_t stream);
void linear_q5_experts_combine(const void *h, const Q5DeviceView &w, int64_t E, const int32_t *ids,
                               const float *coef, int64_t M, int64_t A, void *y, Act act, uint32_t *err,
                               hipStream_t stream);
// Gather with the SwiGLU in its epilogue: w's expert rows are the merged [gate (I) | up (I)] projection (N = 2I per
// expert) and h [M, A, I] = silu(gate) * up - bit-identical to the gather into [M, A, 2I] then kernels::swiglu.
void linear_q4_experts_gather_swiglu(const void *x, const Q4DeviceView &w, int64_t E, const int32_t *ids, int64_t M,
                                     int64_t A, void *h, Act act, uint32_t *err, hipStream_t stream);
void linear_q5_experts_gather_swiglu(const void *x, const Q5DeviceView &w, int64_t E, const int32_t *ids, int64_t M,
                                     int64_t A, void *h, Act act, uint32_t *err, hipStream_t stream);
void check_expert_error(const uint32_t *err, const char *what);
// The same check on the 3 words already on the host (read back with the forward's one sync).
void report_expert_error(const uint32_t e[3], const char *what);

}  // namespace strix::kernels
