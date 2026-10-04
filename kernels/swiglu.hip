#include "kernels/swiglu.hpp"

#include "common/hip_check.hpp"

#include <cstdint>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxElems = 1ll << 40;  // far above any real rows * I; catches a rows/I mix-up

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__device__ inline float silu(float v) { return v / (1.0f + expf(-v)); }  // same form as rmsnorm_gated

__global__ void __launch_bounds__(kThreads) swiglu_f32(const float *gu, float *h, int64_t n, int64_t I) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = i / I, j = i % I;
        h[i] = silu(gu[r * 2 * I + j]) * gu[r * 2 * I + I + j];
    }
}

__global__ void __launch_bounds__(kThreads) swiglu_bf16(const uint16_t *gu, uint16_t *h, int64_t n, int64_t I) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = i / I, j = i % I;
        const float s = bf16_bits_to_f32(f32_to_bf16_bits(silu(bf16_bits_to_f32(gu[r * 2 * I + j]))));
        h[i] = f32_to_bf16_bits(s * bf16_bits_to_f32(gu[r * 2 * I + I + j]));
    }
}

}  // namespace

void swiglu(const void *gu, void *h, int64_t rows, int64_t I, Act act, hipStream_t stream) {
    STRIX_CHECK(gu != nullptr && h != nullptr, "swiglu: null pointer (gu=", gu, ", h=", h, ")");
    STRIX_CHECK(rows >= 1 && I >= 1, "swiglu: rows = ", rows, ", I = ", I, ", both must be >= 1");
    STRIX_CHECK(rows <= kMaxElems / I / 2, "swiglu: rows * 2I = ", rows, " * ", 2 * I, " exceeds ", kMaxElems,
                " elements (rows/I swapped?)");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "swiglu: unsupported activation dtype ", (int)act);
    const int64_t es = act == Act::F32 ? 4 : 2, n = rows * I;
    auto gb = reinterpret_cast<uintptr_t>(gu), hb = reinterpret_cast<uintptr_t>(h);
    STRIX_CHECK(hb + (uintptr_t)(n * es) <= gb || gb + (uintptr_t)(2 * n * es) <= hb, "swiglu: output h [", h, ", +",
                n * es, " B) overlaps input gu [", gu, ", +", 2 * n * es, " B)");
    const int64_t blocks = (n + kThreads - 1) / kThreads;
    dim3 grid((unsigned)(blocks < 65535 * 16 ? blocks : 65535 * 16)), block(kThreads);
    if (act == Act::F32)
        swiglu_f32<<<grid, block, 0, stream>>>(static_cast<const float *>(gu), static_cast<float *>(h), n, I);
    else
        swiglu_bf16<<<grid, block, 0, stream>>>(static_cast<const uint16_t *>(gu), static_cast<uint16_t *>(h), n, I);
    STRIX_HIP_CHECK_LAUNCH("swiglu ", act_name(act), " rows=", rows, " I=", I);
}

}  // namespace strix::kernels
