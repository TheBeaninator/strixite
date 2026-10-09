#pragma once

// Partial rotary position embedding (full-attention step 4): the first rot_dim dims of each q/k head are rotated,
// pairing dim i with i + rot_dim/2 ("rotate half"); the rest pass through. Token t sits at position pos0 + t (text-only
// mRoPE reduces to this, same doc). Both target models: theta 1e7, rot_dim 64 of head_dim 256.
//
// Matches transformers' arithmetic: angle = fp32(pos * inv_freq[i]),
// out = x*cos + rotate_half(x)*sin with each product and the sum rounded
// separately (no FMA); BF16 also rounds cos/sin and each step through BF16.

#include "kernels/norm.hpp"  // Act

#include "common/hip_runtime.hpp"

#include <cstdint>
#include <vector>

namespace strix::kernels {

// inv_freq[j] = 1 / theta^(2j / rot_dim), j < rot_dim/2, computed exactly as
// transformers does (FP32 pow, then FP32 divide - bit-identical to torch; a
// more accurate double-precision version differs in 15 of 32 values, and the
// angle error that grows with position is visible at FP32 tolerance).
std::vector<float> rope_inv_freq(float theta, int64_t rot_dim);

// YaRN context extension (Peng et al. 2023, "YaRN: Efficient Context Window Extension of Large Language Models";
// transformers' rope_type "yarn" semantics, which the model card's recommended config uses): the positions a model
// was trained on (original_max_pos) stretch by `factor`.
// - Frequencies: pair j turns L * inv_freq[j] / 2 pi times over the original L = original_max_pos positions. Pairs
//   that turn more than beta_fast = 32 times (high frequency, short wavelength) keep their frequency; pairs that turn
//   less than beta_slow = 1 time are slowed by `factor` (interpolated); the pairs between are blended with a linear
//   ramp over the pair index, whose ends (where the turn counts hit 32 and 1) are rounded outward to whole pairs.
//   For theta 1e7, rot_dim 64, L 262144: pairs 0-14 keep, 15-21 blend, 22-31 are divided by the factor.
// - cos_sin_scale: the attention factor 0.1 ln(factor) + 1 that multiplies cos and sin (so only the rotated dims;
//   1.069 at factor 2), sharpening attention that interpolation would otherwise flatten.
// Every step in FP32 in transformers' order (bit-identical to its table; rope_yarn_matches_torch).
struct YarnRope {
    std::vector<float> inv_freq;  // rot_dim / 2
    float cos_sin_scale = 1.0f;
};
YarnRope rope_yarn(float theta, int64_t rot_dim, float factor, int64_t original_max_pos);

// In place. Element (t, h, d) of x is at x[t * token_stride + h * head_stride
// + d] (so q/k heads can be rotated where they sit in the merged qkv output).
// inv_freq: device array of rot_dim/2 floats from rope_inv_freq (or rope_yarn). cos_sin_scale: YaRN's attention
// factor (rope_yarn), 1..2; the default 1 is plain RoPE.
void rope(void *x, int64_t T, int64_t heads, int64_t token_stride, int64_t head_stride, int64_t rot_dim, int64_t pos0,
          const float *inv_freq, Act act, hipStream_t stream, float cos_sin_scale = 1.0f);

}  // namespace strix::kernels
