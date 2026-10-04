#include "formats/q6.hpp"

#include "common/check.hpp"
#include "formats/q4.hpp"  // q4_group_size_supported: the same group sizes

#include <cfenv>
#include <cmath>
#include <cstring>

namespace strix {

namespace {

// Same RNE conversion as formats/q4 / q8, the kernels and torch's .to(bfloat16).
uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

void check_q6(const Q6Weight &q, const char *what) {
    STRIX_CHECK(q.N >= 1 && q.K >= 1, what, ": Q6 shape [", q.N, ", ", q.K, "] must be positive");
    STRIX_CHECK(q4_group_size_supported(q.G), what, ": Q6 group size ", q.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(q.K % q.G == 0, what, ": Q6 K = ", q.K, " is not a multiple of group size ", q.G);
    STRIX_CHECK(q.K % 64 == 0, what, ": Q6 K = ", q.K, " is not a multiple of 64 (16-byte aligned code planes)");
    const size_t n_groups = (size_t)(q.N * (q.K / q.G));
    STRIX_CHECK(q.q.size() == (size_t)(q.N * q6_row_bytes(q.K)), what, ": Q6 code array has ", q.q.size(),
                " bytes, expected N*3K/4 = ", q.N * q6_row_bytes(q.K));
    STRIX_CHECK(q.scale.size() == n_groups && q.minv.size() == n_groups, what, ": Q6 scale/min arrays have ",
                q.scale.size(), "/", q.minv.size(), " entries, expected N*K/G = ", n_groups);
}

Q6Weight quantize_q6(const float *w, int64_t N, int64_t K, int64_t G) {
    STRIX_CHECK(w != nullptr, "quantize_q6: input is null");
    STRIX_CHECK(N >= 1 && K >= 1, "quantize_q6: shape [", N, ", ", K, "] must be positive");
    STRIX_CHECK(q4_group_size_supported(G), "quantize_q6: group size ", G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(K % G == 0, "quantize_q6: K = ", K, " is not a multiple of group size ", G);
    STRIX_CHECK(K % 64 == 0, "quantize_q6: K = ", K, " is not a multiple of 64 (16-byte aligned code planes)");
    STRIX_CHECK(std::fegetround() == FE_TONEAREST, "quantize_q6: FP rounding mode isn't round-to-nearest-even");
    Q6Weight out;
    out.N = N, out.K = K, out.G = G;
    const int64_t rb = q6_row_bytes(K);
    out.q.assign((size_t)(N * rb), 0);
    out.scale.resize((size_t)(N * (K / G)));
    out.minv.resize((size_t)(N * (K / G)));
    for (int64_t n = 0; n < N; ++n) {
        uint8_t *row = out.q.data() + n * rb;
        for (int64_t g = 0; g < K / G; ++g) {
            const float *grp = w + n * K + g * G;
            float lo = grp[0], hi = grp[0];
            for (int64_t i = 0; i < G; ++i) {
                STRIX_CHECK(std::isfinite(grp[i]), "quantize_q6: non-finite weight ", grp[i], " at [", n, ", ",
                            g * G + i, "]");
                lo = std::fmin(lo, grp[i]);
                hi = std::fmax(hi, grp[i]);
            }
            const uint16_t s_bits = f32_to_bf16((hi - lo) / 63.0f), m_bits = f32_to_bf16(lo);
            const float s = bf16_to_f32(s_bits), m = bf16_to_f32(m_bits);
            const size_t gi = (size_t)(n * (K / G) + g);
            out.scale[gi] = s_bits;
            out.minv[gi] = m_bits;
            for (int64_t i = 0; i < G; ++i) {
                int code = 0;
                if (s > 0.0f) {
                    const float r = std::nearbyint((grp[i] - m) / s);  // round half to even, like torch.round
                    code = (int)std::fmin(std::fmax(r, 0.0f), 63.0f);
                }
                const int64_t k = g * G + i;
                row[k / 2] |= (uint8_t)((code & 0xF) << (4 * (k % 2)));
                row[K / 2 + k / 4] |= (uint8_t)(((code >> 4) & 0x3) << (2 * (k % 4)));
            }
        }
    }
    return out;
}

void append_rows_q6(Q6Weight &dst, const Q6Weight &src, const char *what) {
    check_q6(src, what);
    if (dst.N == 0 && dst.q.empty() && dst.scale.empty() && dst.minv.empty()) {
        dst = src;
        return;
    }
    check_q6(dst, what);
    STRIX_CHECK(dst.K == src.K && dst.G == src.G, what, ": appending rows of K = ", src.K, ", G = ", src.G,
                " to a Q6 weight with K = ", dst.K, ", G = ", dst.G, " (both must match)");
    dst.q.insert(dst.q.end(), src.q.begin(), src.q.end());
    dst.scale.insert(dst.scale.end(), src.scale.begin(), src.scale.end());
    dst.minv.insert(dst.minv.end(), src.minv.begin(), src.minv.end());
    dst.N += src.N;
}

std::vector<float> dequantize_q6(const Q6Weight &q) {
#pragma clang fp contract(off)
    check_q6(q, "dequantize_q6");
    std::vector<float> w((size_t)(q.N * q.K));
    const int64_t rb = q6_row_bytes(q.K);
    for (int64_t n = 0; n < q.N; ++n)
        for (int64_t k = 0; k < q.K; ++k) {
            const size_t gi = (size_t)(n * (q.K / q.G) + k / q.G);
            w[(size_t)(n * q.K + k)] = (float)q6_code(q.q.data() + n * rb, q.K, k) * bf16_to_f32(q.scale[gi]) +
                                       bf16_to_f32(q.minv[gi]);
        }
    return w;
}

}  // namespace strix
