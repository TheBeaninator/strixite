#include "kernels/wmma.hpp"

#include "common/hip_check.hpp"
#include "kernels/device_info.hpp"

namespace strix::kernels {

namespace {

__global__ void __launch_bounds__(32) wmma_tile_kernel(const uint16_t *a, const uint16_t *bt, float *c) {
    __shared__ uint16_t as[16 * 16], bs[16 * 16];
    for (int i = threadIdx.x; i < 256; i += 32) as[i] = a[i], bs[i] = bt[i];
    __syncthreads();
    wmma_f32x8 acc = {};
    acc = wmma_bf16(wmma_load_rows(as, 16), wmma_load_rows(bs, 16), acc);
#pragma unroll
    for (int v = 0; v < 8; ++v) c[wmma_c_row(v) * 16 + threadIdx.x % 16] = acc[v];
}

}  // namespace

void wmma_tile_product(const uint16_t *a, const uint16_t *bt, float *c) {
    STRIX_CHECK(a != nullptr && bt != nullptr && c != nullptr, "wmma_tile_product: null pointer (a=", (const void *)a,
                ", bt=", (const void *)bt, ", c=", (const void *)c, ")");
    STRIX_CHECK(device_warp_size(32) == 32, "wmma_tile_product: needs wave32 (gfx11 WMMA _w32)");
    wmma_tile_kernel<<<1, 32>>>(a, bt, c);
    STRIX_HIP_CHECK_LAUNCH("wmma_tile_product");
    STRIX_HIP_CHECK(hipDeviceSynchronize(), "wmma_tile_product");
}

}  // namespace strix::kernels
