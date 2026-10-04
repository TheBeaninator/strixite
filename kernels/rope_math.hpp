#pragma once

// Device-side RoPE arithmetic shared by kernels/rope.hip and the fused
// q/k norm + RoPE in kernels/attention.hip, so both rotate identically
// (transformers' arithmetic: kernels/rope.hpp).

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::kernels::rope_math {

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ inline float rb(float v) { return bf16_bits_to_f32(f32_to_bf16_bits(v)); }

// cos/sin of fp32(pos * inv_freq), each times scale in FP32 (YaRN's attention factor - transformers multiplies its
// cos/sin tables by it before the cast; 1 for plain RoPE, which leaves every value bit-identical), then rounded to
// BF16 when kBF16.
template <bool kBF16>
__device__ inline void cos_sin(int64_t pos, float inv_freq, float scale, float &c, float &s) {
    sincosf(__fmul_rn((float)pos, inv_freq), &s, &c);
    c = __fmul_rn(c, scale), s = __fmul_rn(s, scale);
    if (kBF16) c = rb(c), s = rb(s);
}

// (x0, x1) = dims (i, i + rot_dim/2) -> rotated pair; products and sum rounded separately.
template <bool kBF16>
__device__ inline void rotate(float x0, float x1, float c, float s, float &y0, float &y1) {
    if (kBF16) {
        y0 = __fsub_rn(rb(__fmul_rn(x0, c)), rb(__fmul_rn(x1, s)));
        y1 = __fadd_rn(rb(__fmul_rn(x1, c)), rb(__fmul_rn(x0, s)));
    } else {
        y0 = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
        y1 = __fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s));
    }
}

}  // namespace strix::kernels::rope_math
