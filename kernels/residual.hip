#include "kernels/residual.hpp"

#include "common/hip_check.hpp"

#include <cstdint>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxElems = 1ll << 40;  // far above any real activation; catches a size mix-up

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__global__ void __launch_bounds__(kThreads) add_f32(float *x, const float *y, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        x[i] += y[i];
}

__global__ void __launch_bounds__(kThreads) add_bf16(uint16_t *x, const uint16_t *y, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        x[i] = f32_to_bf16_bits(bf16_bits_to_f32(x[i]) + bf16_bits_to_f32(y[i]));
}

}  // namespace

void residual_add(void *x, const void *y, int64_t n, Act act, hipStream_t stream) {
    STRIX_CHECK(x != nullptr && y != nullptr, "residual_add: null pointer (x=", x, ", y=", y, ")");
    STRIX_CHECK(n >= 1 && n <= kMaxElems, "residual_add: n = ", n, ", expected 1..", kMaxElems);
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "residual_add: unsupported activation dtype ", (int)act);
    const int64_t es = act == Act::F32 ? 4 : 2;
    STRIX_CHECK(reinterpret_cast<uintptr_t>(x) % es == 0 && reinterpret_cast<uintptr_t>(y) % es == 0,
                "residual_add: x = ", x, ", y = ", y, " not aligned to ", es, " bytes for ", act_name(act));
    const auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    STRIX_CHECK(x == y || xb + (uintptr_t)(n * es) <= yb || yb + (uintptr_t)(n * es) <= xb, "residual_add: y [", y,
                ", +", n * es, " B) partly overlaps x [", x, ", +", n * es, " B)");
    const int64_t blocks = (n + kThreads - 1) / kThreads;
    dim3 grid((unsigned)(blocks < 65535 * 16 ? blocks : 65535 * 16)), block(kThreads);
    if (act == Act::F32)
        add_f32<<<grid, block, 0, stream>>>(static_cast<float *>(x), static_cast<const float *>(y), n);
    else
        add_bf16<<<grid, block, 0, stream>>>(static_cast<uint16_t *>(x), static_cast<const uint16_t *>(y), n);
    STRIX_HIP_CHECK_LAUNCH("residual_add ", act_name(act), " n=", n);
}

}  // namespace strix::kernels
