#pragma once

// Row layout of the Q4-based expert weights as the expert kernels read it: per row K/2 bytes of 4-bit low codes
// (formats/q4.hpp's nibble order), then - for Q5 (formats/q5.hpp) - K/8 bytes of the codes' 5th bit. HB = the high
// bits per code: 0 (Q4; the high-bit code compiles away) or 1 (Q5). Q5 rows need K % 128 == 0 (16-byte aligned).

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::kernels {

template <int HB>
__host__ __device__ constexpr int64_t qsplit_row_bytes(int64_t K) {
    static_assert(HB == 0 || HB == 1, "Q4 (HB 0) or Q5 (HB 1)");
    return K / 2 + K * HB / 8;
}

// The high bits of the 32 codes of a row starting at k (k % 32 == 0), already shifted into code bit 4.
template <int HB>
struct QHigh32 {
    uint32_t w = 0;
    __device__ static QHigh32 load(const uint8_t *row, int64_t K, int64_t k) {
        QHigh32 h;
        if constexpr (HB == 1) h.w = *reinterpret_cast<const uint32_t *>(row + K / 2 + k / 8);
        return h;
    }
    __device__ uint32_t at(int i) const {
        if constexpr (HB == 1) return ((w >> i) & 1u) << 4;
        else return 0u;
    }
};

}  // namespace strix::kernels
