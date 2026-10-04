#include "formats/q8.hpp"

#include "common/check.hpp"
#include "formats/q4.hpp"  // q4_group_size_supported: the same group sizes

#include <cfenv>
#include <cmath>
#include <cstring>

namespace strix {

namespace {

// Same RNE conversion as formats/q4, the kernels and torch's .to(bfloat16).
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

void check_q8(const Q8Weight &q, const char *what) {
    STRIX_CHECK(q.N >= 1 && q.K >= 1, what, ": Q8 shape [", q.N, ", ", q.K, "] must be positive");
    STRIX_CHECK(q4_group_size_supported(q.G), what, ": Q8 group size ", q.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(q.K % q.G == 0, what, ": Q8 K = ", q.K, " is not a multiple of group size ", q.G);
    const size_t n_groups = (size_t)(q.N * (q.K / q.G));
    STRIX_CHECK(q.q.size() == (size_t)(q.N * q.K), what, ": Q8 code array has ", q.q.size(), " bytes, expected N*K = ",
                q.N * q.K);
    STRIX_CHECK(q.scale.size() == n_groups && q.minv.size() == n_groups, what, ": Q8 scale/min arrays have ",
                q.scale.size(), "/", q.minv.size(), " entries, expected N*K/G = ", n_groups);
}

Q8Weight quantize_q8(const float *w, int64_t N, int64_t K, int64_t G) {
    STRIX_CHECK(w != nullptr, "quantize_q8: input is null");
    STRIX_CHECK(N >= 1 && K >= 1, "quantize_q8: shape [", N, ", ", K, "] must be positive");
    STRIX_CHECK(q4_group_size_supported(G), "quantize_q8: group size ", G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(K % G == 0, "quantize_q8: K = ", K, " is not a multiple of group size ", G);
    STRIX_CHECK(std::fegetround() == FE_TONEAREST, "quantize_q8: FP rounding mode isn't round-to-nearest-even");
    Q8Weight out;
    out.N = N, out.K = K, out.G = G;
    out.q.assign((size_t)(N * K), 0);
    out.scale.resize((size_t)(N * (K / G)));
    out.minv.resize((size_t)(N * (K / G)));
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < K / G; ++g) {
            const float *grp = w + n * K + g * G;
            float lo = grp[0], hi = grp[0];
            for (int64_t i = 0; i < G; ++i) {
                STRIX_CHECK(std::isfinite(grp[i]), "quantize_q8: non-finite weight ", grp[i], " at [", n, ", ",
                            g * G + i, "]");
                lo = std::fmin(lo, grp[i]);
                hi = std::fmax(hi, grp[i]);
            }
            const uint16_t s_bits = f32_to_bf16((hi - lo) / 255.0f), m_bits = f32_to_bf16(lo);
            const float s = bf16_to_f32(s_bits), m = bf16_to_f32(m_bits);
            const size_t gi = (size_t)(n * (K / G) + g);
            out.scale[gi] = s_bits;
            out.minv[gi] = m_bits;
            for (int64_t i = 0; i < G; ++i) {
                int code = 0;
                if (s > 0.0f) {
                    const float r = std::nearbyint((grp[i] - m) / s);  // round half to even, like torch.round
                    code = (int)std::fmin(std::fmax(r, 0.0f), 255.0f);
                }
                out.q[(size_t)(n * K + g * G + i)] = (uint8_t)code;
            }
        }
    }
    return out;
}

void append_rows_q8(Q8Weight &dst, const Q8Weight &src, const char *what) {
    check_q8(src, what);
    if (dst.N == 0 && dst.q.empty() && dst.scale.empty() && dst.minv.empty()) {
        dst = src;
        return;
    }
    check_q8(dst, what);
    STRIX_CHECK(dst.K == src.K && dst.G == src.G, what, ": appending rows of K = ", src.K, ", G = ", src.G,
                " to a Q8 weight with K = ", dst.K, ", G = ", dst.G, " (both must match)");
    dst.q.insert(dst.q.end(), src.q.begin(), src.q.end());
    dst.scale.insert(dst.scale.end(), src.scale.begin(), src.scale.end());
    dst.minv.insert(dst.minv.end(), src.minv.begin(), src.minv.end());
    dst.N += src.N;
}

Q8ChunkMajor to_chunk_major(const Q8Weight &w, const char *what) {
    check_q8(w, what);
    STRIX_CHECK(w.K % 32 == 0, what, ": chunk-major Q8 needs K % 32 == 0 (32-code chunks), got K = ", w.K);
    Q8ChunkMajor c;
    c.N = w.N, c.K = w.K, c.G = w.G;
    const int64_t chunks = w.K / 32, groups = w.K / w.G;
    c.q.resize(w.q.size());
    c.scale.resize(w.scale.size());
    c.minv.resize(w.minv.size());
    for (int64_t n = 0; n < w.N; ++n) {
        for (int64_t k = 0; k < chunks; ++k)
            std::memcpy(&c.q[(size_t)((k * w.N + n) * 32)], &w.q[(size_t)(n * w.K + k * 32)], 32);
        for (int64_t g = 0; g < groups; ++g) {
            c.scale[(size_t)(g * w.N + n)] = w.scale[(size_t)(n * groups + g)];
            c.minv[(size_t)(g * w.N + n)] = w.minv[(size_t)(n * groups + g)];
        }
    }
    return c;
}

std::vector<float> dequantize_q8(const Q8Weight &q) {
#pragma clang fp contract(off)
    check_q8(q, "dequantize_q8");
    std::vector<float> w((size_t)(q.N * q.K));
    for (int64_t n = 0; n < q.N; ++n)
        for (int64_t k = 0; k < q.K; ++k) {
            const size_t gi = (size_t)(n * (q.K / q.G) + k / q.G);
            w[(size_t)(n * q.K + k)] = (float)q.q[(size_t)(n * q.K + k)] * bf16_to_f32(q.scale[gi]) + bf16_to_f32(q.minv[gi]);
        }
    return w;
}

}  // namespace strix
