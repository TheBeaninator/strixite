#include "kernels/norm.hpp"

#include "common/hip_check.hpp"

#include <cmath>
#include <cstdint>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;           // one block per row; LDS tree reduction, no wave-size assumption
constexpr int64_t kMaxRowLen = 1 << 20; // far above any real norm width; catches a rows/d mix-up
constexpr int64_t kMaxRows = 0x7fffffff;
constexpr int64_t kMaxGroups = 64;      // qwen4_exp has 4; catches a groups/d or tokens/groups mix-up

// BF16 is carried as raw bits. RNE rounding, NaN kept a NaN - same as the
// host-side round_to_bf16 the tests use.
__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

template <typename T> __device__ inline float load(const T *p, int64_t i);
template <> __device__ inline float load<float>(const float *p, int64_t i) { return p[i]; }
template <> __device__ inline float load<uint16_t>(const uint16_t *p, int64_t i) { return bf16_bits_to_f32(p[i]); }

template <typename T> __device__ inline void store(T *p, int64_t i, float v);
template <> __device__ inline void store<float>(float *p, int64_t i, float v) { p[i] = v; }
template <> __device__ inline void store<uint16_t>(uint16_t *p, int64_t i, float v) { p[i] = f32_to_bf16_bits(v); }

// Round through the activation dtype (identity for F32).
template <typename T> __device__ inline float round_act(float v);
template <> __device__ inline float round_act<float>(float v) { return v; }
template <> __device__ inline float round_act<uint16_t>(float v) { return bf16_bits_to_f32(f32_to_bf16_bits(v)); }

// 1 / sqrt(mean(x^2) + eps) for this block's row; all threads get the result.
template <typename T>
__device__ inline float row_inv_rms(const T *xr, int64_t d, float eps, float *red) {
    float ss = 0.f;
    for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
        float v = load(xr, j);
        ss += v * v;
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    // 1/sqrtf rather than rsqrtf: sqrtf/division are correctly rounded by default in HIP,
    // rsqrtf is an approximation.
    return 1.0f / sqrtf(red[0] / (float)d + eps);
}

// One block per d-wide row. Row r uses weight slice r % groups (groups = 1: plain rmsnorm_zc; grouped: the rows
// are a [tokens, groups * d] tensor's slices, so consecutive rows walk the groups).
template <typename T>
__global__ void __launch_bounds__(kThreads)
    rmsnorm_zc_kernel(const T *x, const float *w, T *out, int64_t d, int groups, float eps) {
    __shared__ float red[kThreads];
    const T *xr = x + (int64_t)blockIdx.x * d;
    T *orow = out + (int64_t)blockIdx.x * d;
    const float *wr = w + (int64_t)(blockIdx.x % (unsigned)groups) * d;
    float inv = row_inv_rms(xr, d, eps, red);
    for (int64_t j = threadIdx.x; j < d; j += blockDim.x) store(orow, j, (load(xr, j) * inv) * (1.0f + wr[j]));
}

void launch_rmsnorm_zc(const char *op, const void *x, const float *w, void *out, int64_t rows, int64_t d, int groups,
                       float eps, Act act, hipStream_t stream) {
    dim3 grid((unsigned)rows), block(kThreads);
    switch (act) {
        case Act::F32:
            rmsnorm_zc_kernel<float><<<grid, block, 0, stream>>>(static_cast<const float *>(x), w,
                                                                 static_cast<float *>(out), d, groups, eps);
            break;
        case Act::BF16:
            rmsnorm_zc_kernel<uint16_t><<<grid, block, 0, stream>>>(static_cast<const uint16_t *>(x), w,
                                                                    static_cast<uint16_t *>(out), d, groups, eps);
            break;
        default: STRIX_FAIL(op, ": unsupported activation dtype ", (int)act);
    }
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    rmsnorm_gated_kernel(const T *x, const T *z, const float *w, T *out, int64_t d, float eps, bool sigmoid_gate,
                         int64_t z_heads, int64_t z_ts) {
    __shared__ float red[kThreads];
    int64_t base = (int64_t)blockIdx.x * d;
    const int64_t zbase = z_heads > 0 ? (int64_t)blockIdx.x / z_heads * z_ts + (int64_t)blockIdx.x % z_heads * d : base;
    float inv = row_inv_rms(x + base, d, eps, red);
    for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
        float n = round_act<T>(load(x, base + j) * inv);  // reference casts the normed value back first
        float zv = load(z, zbase + j);
        // silu keeps its original form z / (1 + e^-z): z * sigmoid(z) rounds differently, and the Qwen3.5
        // BF16 forward is sensitive to that (a one-ulp shift here moved its logits error by ~28%).
        const float den = 1.0f + expf(-zv);  // sigmoid_gate is uniform across the block: no divergence
        store(out, base + j, (w[j] * n) * (sigmoid_gate ? 1.0f / den : zv / den));
    }
}

void check_common(const char *op, const void *x, const float *w, const void *out, int64_t rows, int64_t d, float eps) {
    STRIX_CHECK(x != nullptr && w != nullptr && out != nullptr, op, ": null pointer (x=", x, ", w=", w, ", out=", out,
                ")");
    STRIX_CHECK(rows >= 1 && rows <= kMaxRows, op, ": rows = ", rows, ", expected 1..", kMaxRows);
    STRIX_CHECK(d >= 1 && d <= kMaxRowLen, op, ": d = ", d, ", expected 1..", kMaxRowLen);
    STRIX_CHECK(std::isfinite(eps) && eps > 0.f, op, ": eps must be finite and > 0, got ", eps);
}

}  // namespace

const char *act_name(Act a) {
    switch (a) {
        case Act::F32: return "F32";
        case Act::BF16: return "BF16";
    }
    return "?";
}

void rmsnorm_zc(const void *x, const float *w, void *out, int64_t rows, int64_t d, float eps, Act act,
                hipStream_t stream) {
    check_common("rmsnorm_zc", x, w, out, rows, d, eps);
    launch_rmsnorm_zc("rmsnorm_zc", x, w, out, rows, d, 1, eps, act, stream);
    STRIX_HIP_CHECK_LAUNCH("rmsnorm_zc ", act_name(act), " rows=", rows, " d=", d);
}

void group_rmsnorm_zc(const void *x, const float *w, void *out, int64_t tokens, int64_t groups, int64_t d, float eps,
                      Act act, hipStream_t stream) {
    STRIX_CHECK(groups >= 1 && groups <= kMaxGroups, "group_rmsnorm_zc: groups = ", groups, ", expected 1..",
                kMaxGroups);
    STRIX_CHECK(tokens >= 1 && tokens <= kMaxRows / groups, "group_rmsnorm_zc: tokens = ", tokens, ", expected 1..",
                kMaxRows / groups, " (tokens * groups = rows must fit the grid)");
    // rows/d/eps/pointers: the same limits as rmsnorm_zc, on the [tokens * groups, d] view.
    check_common("group_rmsnorm_zc", x, w, out, tokens * groups, d, eps);
    launch_rmsnorm_zc("group_rmsnorm_zc", x, w, out, tokens * groups, d, (int)groups, eps, act, stream);
    STRIX_HIP_CHECK_LAUNCH("group_rmsnorm_zc ", act_name(act), " tokens=", tokens, " groups=", groups, " d=", d);
}

const char *gate_act_name(GateAct g) {
    switch (g) {
        case GateAct::SiLU: return "silu";
        case GateAct::Sigmoid: return "sigmoid";
    }
    return "?";
}

void rmsnorm_gated(const void *x, const void *z, const float *w, void *out, int64_t rows, int64_t d, float eps,
                   GateAct gate, Act act, hipStream_t stream, int64_t z_heads, int64_t z_token_stride) {
    check_common("rmsnorm_gated", x, w, out, rows, d, eps);
    STRIX_CHECK(z != nullptr, "rmsnorm_gated: gate z is null");
    STRIX_CHECK(z_heads >= 0 && (z_heads == 0 || (rows % z_heads == 0 && z_token_stride >= z_heads * d)),
                "rmsnorm_gated: z layout z_heads = ", z_heads, ", z_token_stride = ", z_token_stride, " for rows = ", rows,
                ", d = ", d, " (expected z_heads 0, or dividing rows with a token stride >= z_heads * d)");
    STRIX_CHECK(gate == GateAct::SiLU || gate == GateAct::Sigmoid, "rmsnorm_gated: unknown gate activation ",
                (int)gate);
    const bool sig = gate == GateAct::Sigmoid;
    dim3 grid((unsigned)rows), block(kThreads);
    switch (act) {
        case Act::F32:
            rmsnorm_gated_kernel<float><<<grid, block, 0, stream>>>(
                static_cast<const float *>(x), static_cast<const float *>(z), w, static_cast<float *>(out), d, eps, sig,
                z_heads, z_token_stride);
            break;
        case Act::BF16:
            rmsnorm_gated_kernel<uint16_t><<<grid, block, 0, stream>>>(static_cast<const uint16_t *>(x),
                                                                       static_cast<const uint16_t *>(z), w,
                                                                       static_cast<uint16_t *>(out), d, eps, sig,
                                                                       z_heads, z_token_stride);
            break;
        default: STRIX_FAIL("rmsnorm_gated: unsupported activation dtype ", (int)act);
    }
    STRIX_HIP_CHECK_LAUNCH("rmsnorm_gated ", gate_act_name(gate), " ", act_name(act), " rows=", rows, " d=", d);
}

}  // namespace strix::kernels
