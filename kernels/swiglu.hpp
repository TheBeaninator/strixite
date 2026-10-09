#pragma once

// SwiGLU activation (Swish-gated linear unit) between an MLP's gate/up and
// down projections: h = silu(gate) * up, silu(v) = v * sigmoid(v).
//
// Input is the merged gate_up projection's output (formats/merge_plan):
// each row is [gate (I) | up (I)]. That covers the 0.8B dense MLP
// ([M, 2I]), and qwen4_exp's routed + shared experts (the expert gather's
// [M, A, 2I] output is M*A such rows). Math is FP32; BF16 rounds where the
// reference does: h = bf16(bf16(silu(gate)) * up).

#include "kernels/norm.hpp"  // Act

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

// gu [rows, 2I] -> h [rows, I], F32 or BF16 (raw bits). h must not overlap gu.
void swiglu(const void *gu, void *h, int64_t rows, int64_t I, Act act, hipStream_t stream);

}  // namespace strix::kernels
