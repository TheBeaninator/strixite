#pragma once

// Q5: 5-bit asymmetric per-group weight format - formats/q6.hpp's two-plane layout with a 1-bit high plane, for the
// dense classes the per-class shrink study finds tolerant (reference/quant_study_qwen4exp.py "q5gG"). For W [N, K],
// group size G (32/64/128, K % G == 0, K % 128 == 0):
//   q      [N, 5K/8] bytes, per row two planes back to back:
//            low  [K/2]: bits 0..3 of the codes, two per byte along K (low nibble = even k) - Q4's packing
//            high [K/8]: bit 4, eight per byte (code k at bit k % 8 of byte k / 8)
//   scale  [N, K/G] BF16 bits,  min [N, K/G] BF16 bits
//   w[n,k] = q[n,k] * scale[n,k/G] + min[n,k/G]
// Per group: s = bf16((max - min) / 31), m = bf16(min), q = clamp(round_half_even((w - m) / s), 0, 31), all FP32
// (a constant group, s == 0, gets q = 0) - bit-identical to int_asym(bits=5) in reference/quant_study.py.
// 5.25 bits per weight at G = 128, 5.5 at G = 64. K % 128 == 0 keeps every row (5K/8 bytes) and both planes 16-byte
// aligned (the kernels' vector loads).

#include <cstdint>
#include <vector>

namespace strix {

struct Q5Weight {
    int64_t N = 0, K = 0, G = 0;
    std::vector<uint8_t> q;        // N * 5K/8
    std::vector<uint16_t> scale;   // N * K / G
    std::vector<uint16_t> minv;    // N * K / G

    size_t bytes() const { return q.size() + 2 * scale.size() + 2 * minv.size(); }
    double bits_per_weight() const { return 8.0 * (double)bytes() / (double)(N * K); }
};

constexpr int64_t q5_row_bytes(int64_t K) { return K / 2 + K / 8; }
// The 5-bit code of weight k in a row's code bytes (both planes).
inline int q5_code(const uint8_t *row, int64_t K, int64_t k) {
    const int lo = (row[k / 2] >> (4 * (k % 2))) & 0xF, hi = (row[K / 2 + k / 8] >> (k % 8)) & 0x1;
    return lo | (hi << 4);
}

// w: N*K finite FP32 values. Throws on bad shapes, unsupported G (32/64/128), K % 128 != 0, or non-finite input.
Q5Weight quantize_q5(const float *w, int64_t N, int64_t K, int64_t G);
// Throws unless the arrays are consistent with N, K, G.
void check_q5(const Q5Weight &q, const char *what);
// Appends src's rows below dst's (dst empty = copy); exact. Throws unless K and G match.
void append_rows_q5(Q5Weight &dst, const Q5Weight &src, const char *what);
// Dequantized FP32 [N, K] - the exact values the kernel must use (q*s then +m, no FMA contraction).
std::vector<float> dequantize_q5(const Q5Weight &q);

}  // namespace strix
