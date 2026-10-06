#include "kernels/linear_wmma.hpp"

#include "common/hip_check.hpp"
#include "kernels/device_info.hpp"
#include "kernels/wmma_gemm.hpp"

#include <cstdint>
#include <type_traits>

namespace strix::kernels {

namespace {

constexpr int BM = 64, BN = 128;  // block tile: x rows x W rows; 8 waves as 2 x 4, each 32 x 32 (2 x 2 WMMA tiles)
constexpr int64_t kMaxM = 65535ll * BM;

__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = wmma_f32_to_bf16(v); }

// Block tile BMT x BN (BMT 64 or 128 rows of x), 8 waves as 2 x 4, each (BMT / 2) x 32: TM = BMT / 32 by 2 WMMA
// tiles. A taller tile stages the weight codes once for more rows (the kernel is VALU-bound: staging and the scale
// epilogue compete with WMMA for the SIMD - SQ counters, perf/gemm-core).
template <typename TX, typename TY, int BITS, int G, int BMT>
__global__ void __launch_bounds__(kWmmaThreads)
    linear_wmma_kernel(const TX *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv, TY *y, int64_t M,
                       int64_t N, int64_t K) {
    constexpr int NSUB = kWmmaKC / wmma_sub<G>(), TM = BMT / 32;
    __shared__ __attribute__((aligned(16))) uint16_t xs[BMT * kWmmaLd], ws[BN * kWmmaLd];
    __shared__ float rs[BMT * NSUB];
    __shared__ uint16_t sw[BN * WmmaWCodes<BITS, G, BN>::kWin], mw[BN * WmmaWCodes<BITS, G, BN>::kWin];
    __shared__ int64_t xoff[BMT];
    const int64_t m0 = (int64_t)blockIdx.y * BMT, n0 = (int64_t)blockIdx.x * BN;
    const int rows_x = (int)(M - m0 < BMT ? M - m0 : BMT), rows_w = (int)(N - n0 < BN ? N - n0 : BN);
    if (threadIdx.x < BMT) xoff[threadIdx.x] = (m0 + threadIdx.x) * K;
    const int wave = threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    wmma_f32x8 acc[TM][2] = {};
    __syncthreads();
    // Software-pipelined: step k0 + kWmmaKC's global loads are in flight during step k0's WMMA work.
    WmmaXTile<TX, BMT, G> xt;
    WmmaWCodes<BITS, G, BN> wt;
    xt.fetch(x, xoff, rows_x, 0);
    wt.fetch(q, scale, minv, n0, rows_w, K, 0);
    for (int64_t k0 = 0; k0 < K; k0 += kWmmaKC) {
        xt.stage(xs, rs);
        wt.stage(ws, sw, mw);
        __syncthreads();
        if (k0 + kWmmaKC < K) {
            xt.fetch(x, xoff, rows_x, k0 + kWmmaKC);
            wt.fetch(q, scale, minv, n0, rows_w, K, k0 + kWmmaKC);
        }
        wmma_step_scaled<TM, 2, G>(xs, ws, rs, sw, mw, k0, wm * (BMT / 2), wn * 32, acc);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int64_t m = m0 + wm * (BMT / 2) + i * 16 + wmma_c_row(v), n = n0 + wn * 32 + j * 16 + threadIdx.x % 16;
                if (m < M && n < N) store(y, m * N + n, acc[i][j][v]);
            }
}

// F16 (Q4 / Q8 weights): the same tiles, but the weight codes are dequantized
// once while staged (code * scale + min -> F16 in LDS; the two waves sharing a weight fragment read it decoded) and x
// is staged as F16, so the K loop is plain matrix work accumulating the whole K extent - no row sums, no per-group
// accumulator rescale. A step that opens a scale window stages it first and synchronizes before the decode.
// TAIL (K % kWmmaKC == 32, G = 32: a TP-4 rank's shared expert down, K = 160): the last step's second half fetches as
// zeros (kernels/wmma_gemm.hpp wmma_kvalid) - the TAIL = false instantiation is the unchanged kernel.
template <typename TX, typename TY, int BITS, int G, int BMT, bool TAIL = false>
__global__ void __launch_bounds__(kWmmaThreads)
    linear_wmma_f16_kernel(const TX *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv, TY *y, int64_t M,
                           int64_t N, int64_t K) {
    static_assert(BITS == 4 || BITS == 8, "F16 path: Q4 / Q8 weights");
    constexpr int TM = BMT / 32;
    using WT = WmmaWCodes<BITS, G, BN>;
    __shared__ __attribute__((aligned(16))) uint16_t xs[BMT * kWmmaLd], ws[BN * kWmmaLd];
    __shared__ uint16_t sw[BN * WT::kWin], mw[BN * WT::kWin];
    __shared__ int64_t xoff[BMT];
    const int64_t m0 = (int64_t)blockIdx.y * BMT, n0 = (int64_t)blockIdx.x * BN;
    const int rows_x = (int)(M - m0 < BMT ? M - m0 : BMT), rows_w = (int)(N - n0 < BN ? N - n0 : BN);
    if (threadIdx.x < BMT) xoff[threadIdx.x] = (m0 + threadIdx.x) * K;
    const int wave = threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    wmma_f32x8 acc[TM][2] = {};
    __syncthreads();
    WmmaXTile<TX, BMT, G, false> xt;
    WT wt;
    if constexpr (TAIL) {
        xt.fetch(x, xoff, rows_x, 0, wmma_kvalid(K, 0));
        wt.fetch(q, scale, minv, n0, rows_w, K, 0, 0, wmma_kvalid(K, 0));
    } else {
        xt.fetch(x, xoff, rows_x, 0);
        wt.fetch(q, scale, minv, n0, rows_w, K, 0);
    }
    for (int64_t k0 = 0; k0 < K; k0 += kWmmaKC) {
        if (wt.win) {  // uniform: depends on k0 only
            wt.stage_window(sw, mw);
            __syncthreads();
        }
        xt.stage_f16(xs);
        wt.stage_f16(ws, sw, mw, k0);
        __syncthreads();
        if (k0 + kWmmaKC < K) {
            if constexpr (TAIL) {
                const int kv = wmma_kvalid(K, k0 + kWmmaKC);
                xt.fetch(x, xoff, rows_x, k0 + kWmmaKC, kv);
                wt.fetch(q, scale, minv, n0, rows_w, K, k0 + kWmmaKC, 0, kv);
            } else {
                xt.fetch(x, xoff, rows_x, k0 + kWmmaKC);
                wt.fetch(q, scale, minv, n0, rows_w, K, k0 + kWmmaKC);
            }
        }
        wmma_step_f16<TM, 2>(xs, ws, wm * (BMT / 2), wn * 32, acc);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int64_t m = m0 + wm * (BMT / 2) + i * 16 + wmma_c_row(v), n = n0 + wn * 32 + j * 16 + threadIdx.x % 16;
                if (m < M && n < N) store(y, m * N + n, acc[i][j][v]);
            }
}

template <typename TX, typename TY>
__global__ void __launch_bounds__(kWmmaThreads)
    linear_bf16w_wmma_kernel(const TX *x, const uint16_t *w, TY *y, int64_t M, int64_t N, int64_t K) {
    __shared__ __attribute__((aligned(16))) uint16_t xs[BM * kWmmaLd], ws[BN * kWmmaLd];
    __shared__ int64_t xoff[BM];
    const int64_t m0 = (int64_t)blockIdx.y * BM, n0 = (int64_t)blockIdx.x * BN;
    const int rows_x = (int)(M - m0 < BM ? M - m0 : BM), rows_w = (int)(N - n0 < BN ? N - n0 : BN);
    if (threadIdx.x < BM) xoff[threadIdx.x] = (m0 + threadIdx.x) * K;
    const int wave = threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    wmma_f32x8 acc[2][2] = {};
    __syncthreads();
    // Software-pipelined as linear_wmma_kernel.
    WmmaXTile<TX, BM, 64, false> xt;
    WmmaWBf16<BN> wt;
    xt.fetch(x, xoff, rows_x, 0);
    wt.fetch(w, n0, rows_w, K, 0);
    for (int64_t k0 = 0; k0 < K; k0 += kWmmaKC) {
        xt.stage(xs, nullptr);
        wt.stage(ws);
        __syncthreads();
        if (k0 + kWmmaKC < K) {
            xt.fetch(x, xoff, rows_x, k0 + kWmmaKC);
            wt.fetch(w, n0, rows_w, K, k0 + kWmmaKC);
        }
        wmma_step<2, 2>(xs, ws, wm * 32, wn * 32, acc);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int64_t m = m0 + wm * 32 + i * 16 + wmma_c_row(v), n = n0 + wn * 32 + j * 16 + threadIdx.x % 16;
                if (m < M && n < N) store(y, m * N + n, acc[i][j][v]);
            }
}

struct View {
    const uint8_t *q;
    const uint16_t *scale, *minv;
    int64_t N, K, G;
};

// Multiprocessors as HIP reports them (WGPs on gfx1151: 20 for 40 CUs).
int device_mps() {
    static const int n = [] {
        int dev = 0, v = 0;
        STRIX_HIP_CHECK(hipGetDevice(&dev), "linear_wmma: current device");
        STRIX_HIP_CHECK(hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, dev), "linear_wmma: MP count");
        STRIX_CHECK(v >= 1, "linear_wmma: device reports ", v, " multiprocessors");
        return v;
    }();
    return n;
}

// Row tile 128 when that still gives >= 8 blocks per WGP, else 64. Measured (bench_prefill_gemm, 16c005d-bm128 vs
// 94cebaf): 128 rows +10-15% on in_proj / qkv at T = 512 / 2048 and out_proj / o_proj at 2048 (the weight codes
// staged once for twice the rows), but -12% on out_proj / o_proj at 512 (N = 2560: 80 blocks for 20 WGPs).
template <typename TX, typename TY, int BITS>
void launch(const TX *x, const View &w, TY *y, int64_t M, hipStream_t stream) {
    const int64_t col_blocks = (w.N + BN - 1) / BN;
    auto go = [&](auto bmt) {
        constexpr int BMT = decltype(bmt)::value;
        dim3 grid((unsigned)col_blocks, (unsigned)((M + BMT - 1) / BMT)), block(kWmmaThreads);
        // Q4 / Q8 on F16; Q5 / Q6 on BF16 scaled codes until their staging moves over.
        if constexpr (BITS == 4 || BITS == 8) {
            switch (w.G) {
                case 32:
                    if (w.K % kWmmaKC != 0)  // the K tail (only G = 32 has one: K % G == 0)
                        linear_wmma_f16_kernel<TX, TY, BITS, 32, BMT, true><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K);
                    else
                        linear_wmma_f16_kernel<TX, TY, BITS, 32, BMT><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K);
                    break;
                case 64: linear_wmma_f16_kernel<TX, TY, BITS, 64, BMT><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K); break;
                default: linear_wmma_f16_kernel<TX, TY, BITS, 128, BMT><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K); break;
            }
        } else {
            switch (w.G) {
                case 32: linear_wmma_kernel<TX, TY, BITS, 32, BMT><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K); break;
                case 64: linear_wmma_kernel<TX, TY, BITS, 64, BMT><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K); break;
                default: linear_wmma_kernel<TX, TY, BITS, 128, BMT><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K); break;
            }
        }
    };
    if (col_blocks * ((M + 127) / 128) >= 8 * (int64_t)device_mps()) go(std::integral_constant<int, 128>{});
    else go(std::integral_constant<int, 64>{});
}

template <int BITS>
void linear_wmma(const char *fn, const void *x, const View &w, void *y, int64_t M, Act act, Act out_act,
                 hipStream_t stream) {
    STRIX_CHECK(x != nullptr && y != nullptr && w.q != nullptr && w.scale != nullptr && w.minv != nullptr, fn,
                ": null pointer (x=", x, ", y=", y, ", q=", (const void *)w.q, ", scale=", w.scale, ", min=", w.minv, ")");
    STRIX_CHECK(w.G == 32 || w.G == 64 || w.G == 128, fn, ": group size ", w.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(M >= 1 && M <= kMaxM, fn, ": M (rows of x) = ", M, ", expected 1..", kMaxM);
    STRIX_CHECK(w.N >= 1 && w.N <= 0x7fffffff, fn, ": N = ", w.N, ", expected 1..2^31-1");
    // Q4 / Q8 (the F16 kernel) also take a K tail of 32 (K % 32 == 0: a TP-4 rank's K = 160 at G = 32).
    const int64_t kstep = BITS == 4 || BITS == 8 ? 32 : kWmmaKC;
    STRIX_CHECK(w.K >= kstep && w.K % kstep == 0 && w.K % w.G == 0 && w.K <= (1 << 24), fn, ": K = ", w.K,
                ", expected a multiple of ", kstep, " (the K step) and of the group size ", w.G, ", <= 2^24");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w.q) % 16 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0, fn,
                ": code array ", (const void *)w.q, " and x ", x, " must be 16-byte aligned");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    STRIX_CHECK(out_act == act || (act == Act::BF16 && out_act == Act::F32), fn, ": output dtype ", act_name(out_act),
                " with input ", act_name(act), " unsupported (same dtype, or BF16 -> F32)");
    STRIX_CHECK(device_warp_size(kWmmaThreads) == 32, fn, ": needs wave32 (gfx11 WMMA)");
    const int64_t es = act == Act::F32 ? 4 : 2, oes = out_act == Act::F32 ? 4 : 2;
    const auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    STRIX_CHECK(yb + (uintptr_t)(M * w.N * oes) <= xb || xb + (uintptr_t)(M * w.K * es) <= yb, fn, ": output y [", y,
                ", +", M * w.N * oes, " B) overlaps input x [", x, ", +", M * w.K * es, " B)");
    if (act == Act::F32) launch<float, float, BITS>(static_cast<const float *>(x), w, static_cast<float *>(y), M, stream);
    else if (out_act == Act::F32)
        launch<uint16_t, float, BITS>(static_cast<const uint16_t *>(x), w, static_cast<float *>(y), M, stream);
    else launch<uint16_t, uint16_t, BITS>(static_cast<const uint16_t *>(x), w, static_cast<uint16_t *>(y), M, stream);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), "->", act_name(out_act), " M=", M, " N=", w.N, " K=", w.K, " G=", w.G);
}

}  // namespace

void linear_q4_wmma(const void *x, const Q4DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    linear_wmma<4>("linear_q4_wmma", x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, y, M, act, out_act, stream);
}
void linear_q6_wmma(const void *x, const Q6DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    STRIX_CHECK(w.K % 64 == 0, "linear_q6_wmma: K = ", w.K, " is not a multiple of 64 (the Q6 layout's)");
    linear_wmma<6>("linear_q6_wmma", x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, y, M, act, out_act, stream);
}
void linear_q5_wmma(const void *x, const Q5DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    STRIX_CHECK(w.K % 128 == 0, "linear_q5_wmma: K = ", w.K, " is not a multiple of 128 (the Q5 layout's)");
    linear_wmma<5>("linear_q5_wmma", x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, y, M, act, out_act, stream);
}
void linear_q8_wmma(const void *x, const Q8DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    linear_wmma<8>("linear_q8_wmma", x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, y, M, act, out_act, stream);
}

void linear_bf16w_wmma(const void *x, const uint16_t *w, void *y, int64_t M, int64_t N, int64_t K, Act act, Act out_act,
                       hipStream_t stream) {
    const char *fn = "linear_bf16w_wmma";
    STRIX_CHECK(x != nullptr && w != nullptr && y != nullptr, fn, ": null pointer (x=", x, ", w=", (const void *)w,
                ", y=", y, ")");
    STRIX_CHECK(M >= 1 && M <= kMaxM, fn, ": M (rows of x) = ", M, ", expected 1..", kMaxM);
    STRIX_CHECK(N >= 1 && N <= 0x7fffffff, fn, ": N = ", N, ", expected 1..2^31-1");
    STRIX_CHECK(K >= kWmmaKC && K % kWmmaKC == 0 && K <= (1 << 24), fn, ": K = ", K, ", expected a multiple of ",
                kWmmaKC, " (the K step), <= 2^24");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w) % 16 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0, fn,
                ": weights ", (const void *)w, " and x ", x, " must be 16-byte aligned");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    STRIX_CHECK(out_act == act || (act == Act::BF16 && out_act == Act::F32), fn, ": output dtype ", act_name(out_act),
                " with input ", act_name(act), " unsupported (same dtype, or BF16 -> F32)");
    STRIX_CHECK(device_warp_size(kWmmaThreads) == 32, fn, ": needs wave32 (gfx11 WMMA)");
    const int64_t es = act == Act::F32 ? 4 : 2, oes = out_act == Act::F32 ? 4 : 2;
    const auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    STRIX_CHECK(yb + (uintptr_t)(M * N * oes) <= xb || xb + (uintptr_t)(M * K * es) <= yb, fn, ": output y [", y,
                ", +", M * N * oes, " B) overlaps input x [", x, ", +", M * K * es, " B)");
    const dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((M + BM - 1) / BM)), block(kWmmaThreads);
    if (act == Act::F32)
        linear_bf16w_wmma_kernel<float, float><<<grid, block, 0, stream>>>(static_cast<const float *>(x), w,
                                                                           static_cast<float *>(y), M, N, K);
    else if (out_act == Act::F32)
        linear_bf16w_wmma_kernel<uint16_t, float><<<grid, block, 0, stream>>>(static_cast<const uint16_t *>(x), w,
                                                                              static_cast<float *>(y), M, N, K);
    else
        linear_bf16w_wmma_kernel<uint16_t, uint16_t><<<grid, block, 0, stream>>>(static_cast<const uint16_t *>(x), w,
                                                                                 static_cast<uint16_t *>(y), M, N, K);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), "->", act_name(out_act), " M=", M, " N=", N, " K=", K);
}

}  // namespace strix::kernels
