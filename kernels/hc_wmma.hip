#include "kernels/hc_wmma.hpp"

#include "common/hip_check.hpp"
#include "kernels/device_info.hpp"
#include "kernels/wmma_gemm.hpp"

#include <cmath>
#include <cstdint>

namespace strix::kernels {

namespace {

constexpr int kH = 4;
constexpr int BM = 64, BN = 128;  // tokens x weight rows per block; 8 waves as 2 x 4, each 32 x 32
constexpr int kUpJ = BN / kH;     // up: output positions j per block (all H streams of each)
constexpr int kInvThreads = 256;
constexpr int64_t kMaxTokens = 65535ll * BM;

__device__ inline float to_f32(float v) { return v; }
__device__ inline float to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = wmma_f32_to_bf16(v); }

// inv[t, s] = 1 / sqrt(mean(X_s[t]^2) + eps), FP32 (a block per token).
template <typename T>
__global__ void __launch_bounds__(kInvThreads) hc_inv_kernel(const T *x, int64_t d, float eps, float *inv) {
    __shared__ float red[kInvThreads];
    const int64_t t = blockIdx.x;
    for (int s = 0; s < kH; ++s) {
        const T *xs = x + (t * kH + s) * d;
        float ss = 0.f;
        for (int64_t j = threadIdx.x; j < d; j += kInvThreads) {
            const float v = to_f32(xs[j]);
            ss = fmaf(v, v, ss);
        }
        red[threadIdx.x] = ss;
        __syncthreads();
        for (int off = kInvThreads / 2; off > 0; off >>= 1) {
            if ((int)threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
            __syncthreads();
        }
        if (threadIdx.x == 0) inv[t * kH + s] = 1.0f / sqrtf(red[0] / (float)d + eps);
        __syncthreads();
    }
}

// Injection fused with the next mix's inverse RMS (a block per token): X_s[t, j] += w_in[t, s] y[t, j] (one FP32
// fma, then the activation dtype's rounding - hc_inject's arithmetic), and inv[t, s] from the stored values with
// hc_inv_kernel's reduction shape - so both are bit-identical to hc_inject then hc_inv, in one pass over X.
template <typename T>
__global__ void __launch_bounds__(kInvThreads)
    hc_inject_inv_kernel(T *x, const float *w_in, const T *y, int64_t d, float eps, float *inv) {
    __shared__ float red[kInvThreads];
    const int64_t t = blockIdx.x;
    for (int s = 0; s < kH; ++s) {
        T *xs = x + (t * kH + s) * d;
        const float w = w_in[t * kH + s];
        float ss = 0.f;
        for (int64_t j = threadIdx.x; j < d; j += kInvThreads) {
            store(xs, j, fmaf(w, to_f32(y[t * d + j]), to_f32(xs[j])));
            const float v = to_f32(xs[j]);  // the stored (rounded) value, as hc_inv reads it
            ss = fmaf(v, v, ss);
        }
        red[threadIdx.x] = ss;
        __syncthreads();
        for (int off = kInvThreads / 2; off > 0; off >>= 1) {
            if ((int)threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
            __syncthreads();
        }
        if (threadIdx.x == 0) inv[t * kH + s] = 1.0f / sqrtf(red[0] / (float)d + eps);
        __syncthreads();
    }
}

// Down, one stream per blockIdx.z: P_s [T, R] FP32 = X_s W'_s^T over the stream's d inputs, on F16: the codes
// dequantized once while staged (code * scale + min -> F16 in LDS), x staged as F16, the whole stream's K range
// accumulated in FP32 by the matrix units. A step that opens a scale window stages it first and synchronizes before the
// decode.
template <typename T, int BITS, int G>
__global__ void __launch_bounds__(kWmmaThreads)
    hc_down_wmma_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv, int64_t R,
                        int64_t d, int64_t Tn, float *partial) {
    using WCodes = WmmaWCodes<BITS, G, BN>;
    __shared__ __attribute__((aligned(16))) uint16_t xs[BM * kWmmaLd], ws[BN * kWmmaLd];
    __shared__ uint16_t sw[BN * WCodes::kWin], mw[BN * WCodes::kWin];
    __shared__ int64_t xoff[BM];
    const int64_t K = kH * d, s = blockIdx.z, m0 = (int64_t)blockIdx.y * BM, n0 = (int64_t)blockIdx.x * BN;
    const int rows_x = (int)(Tn - m0 < BM ? Tn - m0 : BM), rows_w = (int)(R - n0 < BN ? R - n0 : BN);
    if (threadIdx.x < BM) xoff[threadIdx.x] = (m0 + threadIdx.x) * K + s * d;
    const int wave = threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    wmma_f32x8 acc[2][2] = {};
    __syncthreads();
    // Over the stream's K range [s d, (s + 1) d), software-pipelined two steps deep: two register sets (a, b) take
    // turns, so while a step's tiles are staged and multiplied the loads for the next two steps are in flight (one
    // step ahead left 43% of wave cycles waiting on memory - counters 50bdc6e; two deep: -6% / -14% at 2048 / 512
    // tokens, dd1778f).
    WmmaXTile<T, BM, G, false> xa, xb;
    WCodes wa, wb;
    xa.fetch(x, xoff, rows_x, 0);
    wa.fetch(q, scale, minv, n0, rows_w, K, 0, s * d);
    if (kWmmaKC < d) {
        xb.fetch(x, xoff, rows_x, kWmmaKC);
        wb.fetch(q, scale, minv, n0, rows_w, K, kWmmaKC, s * d);
    }
    auto step = [&](WmmaXTile<T, BM, G, false> &xt, WCodes &wt, int64_t k0) {
        if (wt.win) {  // uniform: depends on k0 only
            wt.stage_window(sw, mw);
            __syncthreads();
        }
        xt.stage_f16(xs);
        wt.stage_f16(ws, sw, mw, k0);
        __syncthreads();
        if (k0 + 2 * kWmmaKC < d) {
            xt.fetch(x, xoff, rows_x, k0 + 2 * kWmmaKC);
            wt.fetch(q, scale, minv, n0, rows_w, K, k0 + 2 * kWmmaKC, s * d);
        }
        wmma_step_f16<2, 2>(xs, ws, wm * 32, wn * 32, acc);
        __syncthreads();
    };
    for (int64_t k0 = 0; k0 < d; k0 += 2 * kWmmaKC) {
        step(xa, wa, k0);
        if (k0 + kWmmaKC < d) step(xb, wb, k0 + kWmmaKC);
    }
    float *P = partial + s * Tn * R;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int64_t m = m0 + wm * 32 + i * 16 + wmma_c_row(v), n = n0 + wn * 32 + j * 16 + threadIdx.x % 16;
                if (m < Tn && n < R) P[m * R + n] = acc[i][j][v];
            }
}

// a = sum_s inv[t, s] P_s[t, n] (s in order), then h = silu(a / H) for n < r, w_in = 2 sigmoid(a / H) after.
__global__ void __launch_bounds__(256)
    hc_down_epilogue_kernel(const float *partial, const float *inv, int64_t Tn, int64_t R, int64_t r, float *h,
                            float *w_in) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Tn * R) return;
    const int64_t t = i / R, n = i % R;
    float a = 0.f;
#pragma unroll
    for (int s = 0; s < kH; ++s) a = fmaf(inv[t * kH + s], partial[(s * Tn + t) * R + n], a);
    const float z = a / (float)kH;
    if (n < r) h[t * r + n] = z / (1.0f + expf(-z));
    else w_in[t * kH + (n - r)] = 2.0f / (1.0f + expf(-z));
}

// Up: block (j0 = 32 blockIdx.x, m0 = 64 blockIdx.y); local weight row lr -> stream lr / 32, position j0 + lr % 32,
// global row (lr / 32) d + j0 + lr % 32 of the chunk-major W_up [H d, r] (codes [K / 32][N] chunks, scales / mins
// [K / G][N]). Wave wn's 32 columns are stream wn. As WmmaWCodes: fetch the step's codes (and, when the step opens
// one, a window of kWin groups' scales / mins - consecutive lanes on consecutive rows, which the [K / G][N] layout
// makes coalesced) into registers; stage the window, then the codes dequantized to F16, into LDS.
template <int BITS, int G>
struct UpCodes {
    static constexpr int kBytesPerChunk = 32 * BITS / 8, kLoadsPerChunk = kBytesPerChunk / 16;
    static constexpr int kLoadsPerRow = (kWmmaKC / 32) * kLoadsPerChunk, kCodesPerLoad = 16 * 8 / BITS;
    static constexpr int kLoads = BN * kLoadsPerRow / kWmmaThreads, kWin = 8, kWinLoads = BN * kWin / kWmmaThreads;
    static_assert(BN * kLoadsPerRow % kWmmaThreads == 0 && BN * kWin % kWmmaThreads == 0, "whole loads per thread");
    uint4 q[kLoads];
    uint16_t s[kWinLoads], m[kWinLoads];
    bool win = false;

    __device__ static int64_t row_of(int lr, int64_t d, int64_t j0) { return (int64_t)(lr / kUpJ) * d + j0 + lr % kUpJ; }
    __device__ void fetch(const uint8_t *codes, const uint16_t *scale, const uint16_t *minv, int64_t N, int64_t K,
                          int64_t d, int64_t j0, int64_t k0) {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, lr = l / kLoadsPerRow, li = l % kLoadsPerRow;
            const int64_t k = k0 + (int64_t)li * kCodesPerLoad;
            q[i] = *reinterpret_cast<const uint4 *>(codes + ((k / 32) * N + row_of(lr, d, j0)) * kBytesPerChunk +
                                                     (k % 32) * BITS / 8);
        }
        win = k0 % G == 0 && (k0 / G) % kWin == 0;
        if (win) {
            const int64_t groups = K / G, g0 = k0 / G;
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads, lr = e % BN;
                const int64_t g = g0 + e / BN, at = g * N + row_of(lr, d, j0);
                const bool ok = g < groups;
                s[i] = ok ? scale[at] : (uint16_t)0, m[i] = ok ? minv[at] : (uint16_t)0;
            }
        }
    }
    __device__ void stage_window(uint16_t *sw, uint16_t *mw) const {
        if (win)
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads;
                sw[(e % BN) * kWin + e / BN] = s[i], mw[(e % BN) * kWin + e / BN] = m[i];
            }
    }
    // The step at k0: this thread's codes dequantized with their groups' scale / min from the staged window.
    __device__ void stage_f16(uint16_t *lds, const uint16_t *sw, const uint16_t *mw, int64_t k0) const {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, lr = l / kLoadsPerRow, kin = (l % kLoadsPerRow) * kCodesPerLoad;
            const int slot = (int)(((k0 + kin) / G) % kWin);  // a load's codes lie in one group (G >= 32)
            wmma_stage_codes_f16<BITS>(q[i], wmma_bf16_bits_to_f16x2(sw[lr * kWin + slot]),
                                       wmma_bf16_bits_to_f16x2(mw[lr * kWin + slot]), lds + lr * kWmmaLd + kin);
        }
    }
};

template <typename T, int BITS, int G>
__global__ void __launch_bounds__(kWmmaThreads)
    hc_up_wmma_kernel(const float *h, const uint8_t *q, const uint16_t *scale, const uint16_t *minv, const T *x,
                      const float *norm_w, const float *inv, int64_t Tn, int64_t d, int64_t r, T *u) {
    // The tiles, then (after the last step) the per-stream contributions [H][BM][kUpJ] in the same LDS.
    __shared__ __attribute__((aligned(16))) unsigned char smem[kH * BM * kUpJ * 4];
    static_assert((BM + BN) * kWmmaLd * 2 <= kH * BM * kUpJ * 4, "tiles fit the contribution buffer");
    uint16_t *xs = reinterpret_cast<uint16_t *>(smem), *ws = xs + BM * kWmmaLd;
    float *contrib = reinterpret_cast<float *>(smem);
    __shared__ uint16_t sw[BN * UpCodes<BITS, G>::kWin], mw[BN * UpCodes<BITS, G>::kWin];
    __shared__ int64_t hoff[BM];
    const int64_t j0 = (int64_t)blockIdx.x * kUpJ, m0 = (int64_t)blockIdx.y * BM, N = kH * d;
    const int rows_x = (int)(Tn - m0 < BM ? Tn - m0 : BM);
    if (threadIdx.x < BM) hoff[threadIdx.x] = (m0 + threadIdx.x) * r;
    const int wave = threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    wmma_f32x8 acc[2][2] = {};
    __syncthreads();
    // F16: h rounded once to F16, the codes dequantized to F16 while staged;
    // software-pipelined (kernels/wmma_gemm.hpp).
    WmmaXTileF32ToF16<BM> xt;
    UpCodes<BITS, G> wt;
    xt.fetch(h, hoff, rows_x, 0);
    wt.fetch(q, scale, minv, N, r, d, j0, 0);
    for (int64_t k0 = 0; k0 < r; k0 += kWmmaKC) {
        if (wt.win) {  // uniform: depends on k0 only
            wt.stage_window(sw, mw);
            __syncthreads();
        }
        xt.stage_f16(xs);
        wt.stage_f16(ws, sw, mw, k0);
        __syncthreads();
        if (k0 + kWmmaKC < r) {
            xt.fetch(h, hoff, rows_x, k0 + kWmmaKC);
            wt.fetch(q, scale, minv, N, r, d, j0, k0 + kWmmaKC);
        }
        wmma_step_f16<2, 2>(xs, ws, wm * 32, wn * 32, acc);
        __syncthreads();
    }
    const int s = wn;  // this wave's columns are stream s
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = wm * 32 + i * 16 + wmma_c_row(v), jj = j * 16 + threadIdx.x % 16;
                const int64_t t = m0 + row, g = (int64_t)s * d + j0 + jj;
                float c = 0.f;
                if (t < Tn) {
                    const float z = 1.0f / (1.0f + expf(-acc[i][j][v]));
                    c = z * (to_f32(x[t * N + g]) * inv[t * kH + s] * (1.0f + norm_w[g]));
                }
                contrib[(s * BM + row) * kUpJ + jj] = c;
            }
    __syncthreads();
    for (int i = threadIdx.x; i < BM * kUpJ; i += kWmmaThreads) {
        const int row = i / kUpJ, jj = i % kUpJ;
        const int64_t t = m0 + row;
        if (t >= Tn) continue;
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < kH; ++k) sum += contrib[(k * BM + row) * kUpJ + jj];
        store(u, t * d + j0 + jj, sum / (float)kH);
    }
}

struct QArrays {
    const uint8_t *q;
    const uint16_t *scale, *minv;
    int64_t N, K, G;
};

void check_common(const char *fn, int64_t T, int64_t H, int64_t d, const QArrays &w, Act act) {
    STRIX_CHECK(H == kH, fn, ": H = ", H, " streams, only ", kH, " is supported");
    STRIX_CHECK(T >= 1 && T <= kMaxTokens, fn, ": T = ", T, " tokens, expected 1..", kMaxTokens);
    STRIX_CHECK(d >= kWmmaKC && d % kWmmaKC == 0 && d <= (1 << 20), fn, ": d = ", d, ", expected a multiple of ",
                kWmmaKC, " up to 2^20");
    STRIX_CHECK(w.q != nullptr && w.scale != nullptr && w.minv != nullptr, fn, ": null weight pointer (q=",
                (const void *)w.q, ", scale=", w.scale, ", min=", w.minv, ")");
    STRIX_CHECK(w.G == 32 || w.G == 64 || w.G == 128, fn, ": group size ", w.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(w.K % w.G == 0, fn, ": K = ", w.K, " not a multiple of group size ", w.G);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w.q) % 16 == 0, fn, ": code array ", (const void *)w.q,
                " must be 16-byte aligned");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    STRIX_CHECK(device_warp_size(kWmmaThreads) == 32, fn, ": needs wave32 (gfx11 WMMA)");
}

template <int BITS>
void mix_down(const void *x, const QArrays &w, int64_t T, int64_t H, int64_t d, int64_t r, bool inject, float eps,
              float *h, float *w_in, float *inv, float *ws, size_t ws_bytes, Act act, hipStream_t stream,
              bool inv_ready) {
    const char *fn = "hc_mix_down_wmma";
    check_common(fn, T, H, d, w, act);
    STRIX_CHECK(d % w.G == 0, fn, ": d = ", d, " not a multiple of group size ", w.G,
                " (each stream's K range must start a group)");
    const int64_t R = r + (inject ? H : 0);
    STRIX_CHECK(x != nullptr && h != nullptr && inv != nullptr && ws != nullptr, fn, ": null pointer (x=", x, ", h=",
                (void *)h, ", inv=", (void *)inv, ", ws=", (void *)ws, ")");
    STRIX_CHECK(inject == (w_in != nullptr), fn, ": w_in must be given exactly when inject (inject=", inject,
                ", w_in=", (void *)w_in, ")");
    STRIX_CHECK(r >= 1 && r <= 1024, fn, ": r = ", r, ", expected 1..1024");
    STRIX_CHECK(w.K == H * d && w.N == R, fn, ": weight [", w.N, ", ", w.K, "], expected [r", inject ? " + H" : "",
                " = ", R, ", H*d = ", H * d, "]");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(x) % 16 == 0, fn, ": x ", x, " must be 16-byte aligned");
    STRIX_CHECK(std::isfinite(eps) && eps > 0.f, fn, ": eps must be finite and > 0, got ", eps);
    STRIX_CHECK(ws_bytes >= hc_wmma_workspace_bytes(T, H, R), fn, ": workspace of ", ws_bytes, " bytes, needs ",
                hc_wmma_workspace_bytes(T, H, R));
    dim3 grid((unsigned)((R + BN - 1) / BN), (unsigned)((T + BM - 1) / BM), (unsigned)H), block(kWmmaThreads);
    const unsigned epi_blocks = (unsigned)((T * R + 255) / 256);
    auto run = [&](auto tx) {
        using TX = decltype(tx);
        const TX *xp = static_cast<const TX *>(x);
        if (!inv_ready) hc_inv_kernel<TX><<<(unsigned)T, kInvThreads, 0, stream>>>(xp, d, eps, inv);
        switch (w.G) {
            case 32: hc_down_wmma_kernel<TX, BITS, 32><<<grid, block, 0, stream>>>(xp, w.q, w.scale, w.minv, R, d, T, ws); break;
            case 64: hc_down_wmma_kernel<TX, BITS, 64><<<grid, block, 0, stream>>>(xp, w.q, w.scale, w.minv, R, d, T, ws); break;
            default: hc_down_wmma_kernel<TX, BITS, 128><<<grid, block, 0, stream>>>(xp, w.q, w.scale, w.minv, R, d, T, ws); break;
        }
        hc_down_epilogue_kernel<<<epi_blocks, 256, 0, stream>>>(ws, inv, T, R, r, h, w_in);
    };
    if (act == Act::F32) run(float{});
    else run(uint16_t{});
    STRIX_HIP_CHECK_LAUNCH(fn, " Q", BITS, " ", act_name(act), " T=", T, " R=", R, " d=", d, " G=", w.G);
}

template <int BITS>
void mix_up(const float *h, const QArrays &w, const void *x, const float *norm_w, const float *inv, int64_t T,
            int64_t H, int64_t d, void *u, Act act, hipStream_t stream) {
    const char *fn = "hc_mix_up_wmma";
    check_common(fn, T, H, d, w, act);
    STRIX_CHECK(h != nullptr && x != nullptr && norm_w != nullptr && inv != nullptr && u != nullptr, fn,
                ": null pointer (h=", (const void *)h, ", x=", x, ", norm_w=", (const void *)norm_w, ", inv=",
                (const void *)inv, ", u=", u, ")");
    STRIX_CHECK(w.N == H * d, fn, ": W_up has ", w.N, " rows, expected H*d = ", H * d);
    STRIX_CHECK(w.K >= kWmmaKC && w.K % kWmmaKC == 0 && w.K <= 1024, fn, ": W_up K (the rank r) = ", w.K,
                ", expected a multiple of ", kWmmaKC, " up to 1024");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(h) % 16 == 0, fn, ": h ", (const void *)h, " must be 16-byte aligned");
    const size_t es = act == Act::F32 ? 4 : 2;
    const auto ub = reinterpret_cast<uintptr_t>(u), xb = reinterpret_cast<uintptr_t>(x);
    STRIX_CHECK(ub + (uintptr_t)(T * d) * es <= xb || xb + (uintptr_t)(T * H * d) * es <= ub, fn,
                ": u overlaps the streams x");
    dim3 grid((unsigned)(d / kUpJ), (unsigned)((T + BM - 1) / BM)), block(kWmmaThreads);
    auto run = [&](auto tx) {
        using TX = decltype(tx);
        const TX *xp = static_cast<const TX *>(x);
        TX *up = static_cast<TX *>(u);
        switch (w.G) {
            case 32: hc_up_wmma_kernel<TX, BITS, 32><<<grid, block, 0, stream>>>(h, w.q, w.scale, w.minv, xp, norm_w, inv, T, d, w.K, up); break;
            case 64: hc_up_wmma_kernel<TX, BITS, 64><<<grid, block, 0, stream>>>(h, w.q, w.scale, w.minv, xp, norm_w, inv, T, d, w.K, up); break;
            default: hc_up_wmma_kernel<TX, BITS, 128><<<grid, block, 0, stream>>>(h, w.q, w.scale, w.minv, xp, norm_w, inv, T, d, w.K, up); break;
        }
    };
    if (act == Act::F32) run(float{});
    else run(uint16_t{});
    STRIX_HIP_CHECK_LAUNCH(fn, " Q", BITS, " ", act_name(act), " T=", T, " d=", d, " r=", w.K, " G=", w.G);
}

}  // namespace

size_t hc_wmma_workspace_bytes(int64_t T, int64_t H, int64_t R) {
    STRIX_CHECK(T >= 1 && H >= 1 && R >= 1, "hc_wmma_workspace_bytes: T = ", T, ", H = ", H, ", R = ", R);
    return (size_t)(H * T * R) * 4;
}

void hc_mix_down_wmma(const void *x, const Q4DeviceView &w, int64_t T, int64_t H, int64_t d, int64_t r, bool inject,
                      float eps, float *h, float *w_in, float *inv, float *ws, size_t ws_bytes, Act act,
                      hipStream_t stream, bool inv_ready) {
    mix_down<4>(x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, T, H, d, r, inject, eps, h, w_in, inv, ws, ws_bytes, act,
                stream, inv_ready);
}
void hc_mix_down_wmma(const void *x, const Q8DeviceView &w, int64_t T, int64_t H, int64_t d, int64_t r, bool inject,
                      float eps, float *h, float *w_in, float *inv, float *ws, size_t ws_bytes, Act act,
                      hipStream_t stream, bool inv_ready) {
    mix_down<8>(x, {w.q, w.scale, w.minv, w.N, w.K, w.G}, T, H, d, r, inject, eps, h, w_in, inv, ws, ws_bytes, act,
                stream, inv_ready);
}
void hc_mix_up_wmma(const float *h, const Q4ChunkMajorView &w, const void *x, const float *norm_w, const float *inv,
                    int64_t T, int64_t H, int64_t d, void *u, Act act, hipStream_t stream) {
    mix_up<4>(h, {w.q, w.scale, w.minv, w.N, w.K, w.G}, x, norm_w, inv, T, H, d, u, act, stream);
}
void hc_mix_up_wmma(const float *h, const Q8ChunkMajorView &w, const void *x, const float *norm_w, const float *inv,
                    int64_t T, int64_t H, int64_t d, void *u, Act act, hipStream_t stream) {
    mix_up<8>(h, {w.q, w.scale, w.minv, w.N, w.K, w.G}, x, norm_w, inv, T, H, d, u, act, stream);
}

}  // namespace strix::kernels

namespace strix::kernels {

void hc_inject_inv(void *x, const float *w_in, const void *y, int64_t T, int64_t H, int64_t d, float eps, float *inv,
                   Act act, hipStream_t stream) {
    const char *fn = "hc_inject_inv";
    STRIX_CHECK(H == kH, fn, ": H = ", H, " streams, only ", kH, " is supported");
    STRIX_CHECK(T >= 1 && T <= 0x7fffffff && d >= 1 && d <= (1 << 20), fn, ": T = ", T, ", d = ", d);
    STRIX_CHECK(x != nullptr && w_in != nullptr && y != nullptr && inv != nullptr, fn, ": null pointer (x=", x,
                ", w_in=", (const void *)w_in, ", y=", y, ", inv=", (void *)inv, ")");
    STRIX_CHECK(std::isfinite(eps) && eps > 0.f, fn, ": eps must be finite and > 0, got ", eps);
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    const size_t es = act == Act::F32 ? 4 : 2;
    const auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    STRIX_CHECK(yb + (uintptr_t)(T * d) * es <= xb || xb + (uintptr_t)(T * H * d) * es <= yb, fn,
                ": y overlaps the streams x");
    if (act == Act::F32)
        hc_inject_inv_kernel<float><<<(unsigned)T, kInvThreads, 0, stream>>>(static_cast<float *>(x), w_in,
                                                                             static_cast<const float *>(y), d, eps, inv);
    else
        hc_inject_inv_kernel<uint16_t><<<(unsigned)T, kInvThreads, 0, stream>>>(static_cast<uint16_t *>(x), w_in,
                                                                                static_cast<const uint16_t *>(y), d, eps,
                                                                                inv);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), " T=", T, " d=", d);
}

}  // namespace strix::kernels
