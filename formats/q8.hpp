#pragma once

// Q8: int8 asymmetric per-group weight format - Q4's scheme (formats/q4.hpp)
// at 8 bits, for the classes the quantization study shows need it (quant_study_qwen4exp.py, "q8gG").
// For W [N, K], group size G (32/64/128, K % G == 0):
//   q      [N, K] bytes: one unsigned code 0..255 per weight, row-major
//   scale  [N, K/G] BF16 bits,  min [N, K/G] BF16 bits
//   w[n,k] = q[n,k] * scale[n,k/G] + min[n,k/G]
// Per group: s = bf16((max - min) / 255), m = bf16(min), q = clamp(round_half_even((w - m) / s), 0, 255), all
// FP32 (a constant group, s == 0, gets q = 0) - bit-identical to int_asym(bits=8) in reference/quant_study.py.
// 8.5 bits per weight at G = 64.

#include <cstdint>
#include <vector>

namespace strix {

struct Q8Weight {
    int64_t N = 0, K = 0, G = 0;
    std::vector<uint8_t> q;        // N * K
    std::vector<uint16_t> scale;   // N * K / G
    std::vector<uint16_t> minv;    // N * K / G

    size_t bytes() const { return q.size() + 2 * scale.size() + 2 * minv.size(); }
    double bits_per_weight() const { return 8.0 * (double)bytes() / (double)(N * K); }
};

// w: N*K finite FP32 values. Throws on bad shapes, unsupported G (32/64/128), or non-finite input.
Q8Weight quantize_q8(const float *w, int64_t N, int64_t K, int64_t G);
// Throws unless the arrays are consistent with N, K, G.
void check_q8(const Q8Weight &q, const char *what);
// Appends src's rows below dst's (dst empty = copy); exact, as for Q4. Throws unless K and G match.
void append_rows_q8(Q8Weight &dst, const Q8Weight &src, const char *what);
// Dequantized FP32 [N, K] - the exact values the kernel must use (q*s then +m, no FMA contraction).
std::vector<float> dequantize_q8(const Q8Weight &q);

// Chunk-major Q8 (formats/q4.hpp Q4ChunkMajor's layout at one code per byte), for kernels whose lanes each own
// a row (kernels/hc.hpp hc_mix_up): 32-code chunk c of row n (32 bytes) at q[(c * N + n) * 32 ..], group g's
// scale/min at [g * N + n]. K % 32 == 0.
struct Q8ChunkMajor {
    int64_t N = 0, K = 0, G = 0;
    std::vector<uint8_t> q;
    std::vector<uint16_t> scale, minv;
    size_t bytes() const { return q.size() + 2 * scale.size() + 2 * minv.size(); }
};
Q8ChunkMajor to_chunk_major(const Q8Weight &w, const char *what);

}  // namespace strix
