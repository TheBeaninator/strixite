#include "formats/q4.hpp"

#include "common/check.hpp"

#include <cfenv>
#include <cmath>
#include <cstring>

namespace strix {

namespace {

// Same RNE conversion as the kernels and the Python study (torch .to(bfloat16)).
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

bool q4_group_size_supported(int64_t G) { return G == 32 || G == 64 || G == 128; }

void check_q4(const Q4Weight &q, const char *what) {
    STRIX_CHECK(q.N >= 1 && q.K >= 1, what, ": Q4 shape [", q.N, ", ", q.K, "] must be positive");
    STRIX_CHECK(q4_group_size_supported(q.G), what, ": Q4 group size ", q.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(q.K % q.G == 0, what, ": Q4 K = ", q.K, " is not a multiple of group size ", q.G);
    size_t n_groups = (size_t)(q.N * (q.K / q.G));
    STRIX_CHECK(q.q.size() == (size_t)(q.N * q.K / 2), what, ": Q4 code array has ", q.q.size(), " bytes, expected N*K/2 = ",
                q.N * q.K / 2);
    STRIX_CHECK(q.scale.size() == n_groups && q.minv.size() == n_groups, what, ": Q4 scale/min arrays have ",
                q.scale.size(), "/", q.minv.size(), " entries, expected N*K/G = ", n_groups);
}

Q4Weight quantize_q4(const float *w, int64_t N, int64_t K, int64_t G) {
    STRIX_CHECK(w != nullptr, "quantize_q4: input is null");
    STRIX_CHECK(N >= 1 && K >= 1, "quantize_q4: shape [", N, ", ", K, "] must be positive");
    STRIX_CHECK(q4_group_size_supported(G), "quantize_q4: group size ", G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(K % G == 0, "quantize_q4: K = ", K, " is not a multiple of group size ", G);
    STRIX_CHECK(std::fegetround() == FE_TONEAREST, "quantize_q4: FP rounding mode isn't round-to-nearest-even");

    Q4Weight out;
    out.N = N, out.K = K, out.G = G;
    out.q.assign((size_t)(N * K / 2), 0);
    out.scale.resize((size_t)(N * (K / G)));
    out.minv.resize((size_t)(N * (K / G)));
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < K / G; ++g) {
            const float *grp = w + n * K + g * G;
            float lo = grp[0], hi = grp[0];
            for (int64_t i = 0; i < G; ++i) {
                STRIX_CHECK(std::isfinite(grp[i]), "quantize_q4: non-finite weight ", grp[i], " at [", n, ", ",
                            g * G + i, "]");
                lo = std::fmin(lo, grp[i]);
                hi = std::fmax(hi, grp[i]);
            }
            uint16_t s_bits = f32_to_bf16((hi - lo) / 15.0f);
            uint16_t m_bits = f32_to_bf16(lo);
            float s = bf16_to_f32(s_bits), m = bf16_to_f32(m_bits);
            size_t gi = (size_t)(n * (K / G) + g);
            out.scale[gi] = s_bits;
            out.minv[gi] = m_bits;
            for (int64_t i = 0; i < G; ++i) {
                int code = 0;
                if (s > 0.0f) {
                    float r = std::nearbyint((grp[i] - m) / s);  // round half to even, like torch.round
                    code = (int)std::fmin(std::fmax(r, 0.0f), 15.0f);
                }
                int64_t k = g * G + i;
                out.q[(size_t)((n * K + k) / 2)] |= (uint8_t)(code << ((k & 1) ? 4 : 0));
            }
        }
    }
    return out;
}

void append_rows_q4(Q4Weight &dst, const Q4Weight &src, const char *what) {
    check_q4(src, what);
    if (dst.N == 0 && dst.q.empty() && dst.scale.empty() && dst.minv.empty()) {
        dst = src;
        return;
    }
    check_q4(dst, what);
    STRIX_CHECK(dst.K == src.K && dst.G == src.G, what, ": appending rows of K = ", src.K, ", G = ", src.G,
                " to a Q4 weight with K = ", dst.K, ", G = ", dst.G, " (both must match)");
    dst.q.insert(dst.q.end(), src.q.begin(), src.q.end());
    dst.scale.insert(dst.scale.end(), src.scale.begin(), src.scale.end());
    dst.minv.insert(dst.minv.end(), src.minv.begin(), src.minv.end());
    dst.N += src.N;
}

Q4ChunkMajor to_chunk_major(const Q4Weight &w, const char *what) {
    check_q4(w, what);
    STRIX_CHECK(w.K % 32 == 0, what, ": chunk-major Q4 needs K % 32 == 0 (32-code chunks), got K = ", w.K);
    Q4ChunkMajor c;
    c.N = w.N, c.K = w.K, c.G = w.G;
    const int64_t chunks = w.K / 32, groups = w.K / w.G;
    c.q.resize(w.q.size());
    c.scale.resize(w.scale.size());
    c.minv.resize(w.minv.size());
    for (int64_t n = 0; n < w.N; ++n) {
        for (int64_t k = 0; k < chunks; ++k)
            std::memcpy(&c.q[(size_t)((k * w.N + n) * 16)], &w.q[(size_t)(n * (w.K / 2) + k * 16)], 16);
        for (int64_t g = 0; g < groups; ++g) {
            c.scale[(size_t)(g * w.N + n)] = w.scale[(size_t)(n * groups + g)];
            c.minv[(size_t)(g * w.N + n)] = w.minv[(size_t)(n * groups + g)];
        }
    }
    return c;
}

std::vector<float> dequantize_q4(const Q4Weight &q) {
    // q*s then +m, each rounded separately like torch's `q * s + m`: no FMA contraction, or the
    // result could differ from the Python study quantizer in the last bit.
#pragma clang fp contract(off)
    check_q4(q, "dequantize_q4");
    std::vector<float> w((size_t)(q.N * q.K));
    for (int64_t n = 0; n < q.N; ++n)
        for (int64_t k = 0; k < q.K; ++k) {
            size_t gi = (size_t)(n * (q.K / q.G) + k / q.G);
            int code = (q.q[(size_t)((n * q.K + k) / 2)] >> ((k & 1) ? 4 : 0)) & 0xF;
            w[(size_t)(n * q.K + k)] = (float)code * bf16_to_f32(q.scale[gi]) + bf16_to_f32(q.minv[gi]);
        }
    return w;
}

}  // namespace strix
