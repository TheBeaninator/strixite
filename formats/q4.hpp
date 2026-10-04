#pragma once

// Q4: int4 asymmetric per-group weight format. For W [N, K], group size G (32/64/128, K % G == 0):
//   q      [N, K/2] bytes: two 4-bit codes per byte along K (low nibble = even k)
//   scale  [N, K/G] BF16 bits,  min [N, K/G] BF16 bits
//   w[n,k] = q[n,k] * scale[n,k/G] + min[n,k/G]
// Quantization is bit-identical to int_asym in reference/quant_study.py, so
// the engine runs exactly what the study measured.

#include <cstdint>
#include <vector>

namespace strix {

struct Q4Weight {
    int64_t N = 0, K = 0, G = 0;
    std::vector<uint8_t> q;        // N * K / 2
    std::vector<uint16_t> scale;   // N * K / G
    std::vector<uint16_t> minv;    // N * K / G

    int64_t groups_per_row() const { return K / G; }
    size_t bytes() const { return q.size() + 2 * scale.size() + 2 * minv.size(); }
    double bits_per_weight() const { return 8.0 * (double)bytes() / (double)(N * K); }
};

bool q4_group_size_supported(int64_t G);  // 32, 64 or 128

// w: N*K finite FP32 values (e.g. BF16 weights widened). Throws on bad shapes,
// unsupported G, or non-finite input.
Q4Weight quantize_q4(const float *w, int64_t N, int64_t K, int64_t G);

// Throws unless the arrays are consistent with N, K, G.
void check_q4(const Q4Weight &q, const char *what);

// Appends src's rows below dst's (dst empty = copy). Q4 groups run along K
// within a row, so this is exact: the result is byte-identical to quantizing
// the row-concatenated FP32 matrix. Throws unless K and G match.
void append_rows_q4(Q4Weight &dst, const Q4Weight &src, const char *what);

// Dequantized FP32 [N, K] - the exact values the kernel must use.
std::vector<float> dequantize_q4(const Q4Weight &q);

// The same codes, scales and mins re-laid out chunk-major for kernels whose
// lanes each own a row (kernels/hc.hpp hc_mix_up): 32-code chunk c of row n
// (16 bytes) at q[(c * N + n) * 16 ..], group g's scale/min at [g * N + n].
// Consecutive rows' chunk c are then adjacent, so a wave's 16-byte loads over
// consecutive rows are one contiguous run. A separate type, so a kernel can't
// be handed the row-major layout by mistake. K % 32 == 0.
struct Q4ChunkMajor {
    int64_t N = 0, K = 0, G = 0;
    std::vector<uint8_t> q;
    std::vector<uint16_t> scale, minv;
    size_t bytes() const { return q.size() + 2 * scale.size() + 2 * minv.size(); }
};
Q4ChunkMajor to_chunk_major(const Q4Weight &w, const char *what);

}  // namespace strix
