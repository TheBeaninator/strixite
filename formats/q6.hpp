#pragma once

// Q6: 6-bit asymmetric per-group weight format - Q4's / Q8's scheme (formats/q4.hpp, formats/q8.hpp) at 6 bits, for
// dense classes that tolerate less than 8 (a per-class shrink study, quant_study_qwen4exp.py
// "q6gG"). For W [N, K], group size G (32/64/128, K % G == 0, K % 64 == 0):
//   q      [N, 3K/4] bytes, per row two planes back to back:
//            low  [K/2]: bits 0..3 of the codes, two per byte along K (low nibble = even k) - Q4's packing
//            high [K/4]: bits 4..5, four per byte (code k at bits 2 * (k % 4) of byte k / 4)
//   scale  [N, K/G] BF16 bits,  min [N, K/G] BF16 bits
//   w[n,k] = q[n,k] * scale[n,k/G] + min[n,k/G]
// Per group: s = bf16((max - min) / 63), m = bf16(min), q = clamp(round_half_even((w - m) / s), 0, 63), all FP32
// (a constant group, s == 0, gets q = 0) - bit-identical to int_asym(bits=6) in reference/quant_study.py.
// 6.25 bits per weight at G = 128, 6.5 at G = 64. K % 64 == 0 keeps every row's planes 16-byte aligned (the kernels'
// vector loads).

#include <cstdint>
#include <vector>

namespace strix {

struct Q6Weight {
    int64_t N = 0, K = 0, G = 0;
    std::vector<uint8_t> q;        // N * 3K/4
    std::vector<uint16_t> scale;   // N * K / G
    std::vector<uint16_t> minv;    // N * K / G

    size_t bytes() const { return q.size() + 2 * scale.size() + 2 * minv.size(); }
    double bits_per_weight() const { return 8.0 * (double)bytes() / (double)(N * K); }
};

constexpr int64_t q6_row_bytes(int64_t K) { return K / 2 + K / 4; }
// The 6-bit code of weight k in a row's code bytes (both planes).
inline int q6_code(const uint8_t *row, int64_t K, int64_t k) {
    const int lo = (row[k / 2] >> (4 * (k % 2))) & 0xF, hi = (row[K / 2 + k / 4] >> (2 * (k % 4))) & 0x3;
    return lo | (hi << 4);
}

// w: N*K finite FP32 values. Throws on bad shapes, unsupported G (32/64/128), K % 64 != 0, or non-finite input.
Q6Weight quantize_q6(const float *w, int64_t N, int64_t K, int64_t G);
// Throws unless the arrays are consistent with N, K, G.
void check_q6(const Q6Weight &q, const char *what);
// Appends src's rows below dst's (dst empty = copy); exact, as for Q4 / Q8. Throws unless K and G match.
void append_rows_q6(Q6Weight &dst, const Q6Weight &src, const char *what);
// Dequantized FP32 [N, K] - the exact values the kernel must use (q*s then +m, no FMA contraction).
std::vector<float> dequantize_q6(const Q6Weight &q);

}  // namespace strix
