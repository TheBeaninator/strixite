// Your first kernel - the companion program to tutorial/first-kernel/README.md.
//
// One small piece of a transformer layer, done the way every kernel in strixite is done: a CPU reference first, GPU
// kernels checked against it bit for bit, a deliberate bug to prove the check works, timing - and then a fusion
// that passes every test but runs slower, which only a trace shows.
//
// The math (per token t, hidden column c, residual stream s):
//   y[t][c]    = moe[t][c] + shared[t][c]                  ("add":    the MoE output plus the shared expert's)
//   x[t][s][c] = fma(w[t][s], y[t][c], x[t][s][c])        ("inject": y written into each of S residual streams)
//
// Usage: first_kernel [--test] [--mutate 1|2] [--bench] [--trace] [--reps N]
//   --test       run every kernel and compare it with the CPU reference (the default when nothing is given)
//   --mutate N   break one kernel on purpose (1: a grid one block too small, 2: the wrong stream weight) and show
//                that --test catches it
//   --bench      time every variant at T = 1 (one token, like generating) and T = 512 (like reading a prompt)
//   --trace      run each variant a few times at T = 1 - for a profiler trace (README, phase 6)
//   --reps N     timed launches per measurement (default 200)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Every HIP call is checked; a failure says what was called and why it failed, then stops.
#define HIP_CHECK(expr)                                                                                  \
    do {                                                                                                 \
        hipError_t err_ = (expr);                                                                        \
        if (err_ != hipSuccess) {                                                                        \
            std::fprintf(stderr, "HIP error at %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,          \
                         hipGetErrorString(err_));                                                       \
            std::exit(1);                                                                                \
        }                                                                                                \
    } while (0)

static void fail(const std::string &msg) {
    std::fprintf(stderr, "first_kernel: %s\n", msg.c_str());
    std::exit(1);
}

constexpr int kD = 2560;      // hidden size: columns per token
constexpr int kS = 4;         // residual streams
constexpr int kThreads = 256; // threads per workgroup (8 waves of 32 on RDNA)
static_assert(kD % kThreads == 0, "the column-tiled kernels assume D is a multiple of the workgroup size");

// ---- phase 1: the CPU reference - the spec, written as plainly as possible ------------------------------------
static void reference(int T, const float *moe, const float *shared, const float *w, float *x) {
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < kD; ++c) {
            const float y = moe[t * kD + c] + shared[t * kD + c];
            for (int s = 0; s < kS; ++s) {
                float &xv = x[((size_t)t * kS + s) * kD + c];
                xv = std::fmaf(w[t * kS + s], y, xv);
            }
        }
}

// ---- phase 2: two separate GPU kernels, one thread per element -----------------------------------------------
__global__ void add_kernel(int T, const float *moe, const float *shared, float *y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * kD) y[i] = moe[i] + shared[i];
}

__global__ void inject_kernel(int T, const float *w, const float *y, float *x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // one thread per (t, s, c)
    if (i >= T * kS * kD) return;
    const int c = i % kD, s = (i / kD) % kS, t = i / (kS * kD);
    x[i] = fmaf(w[t * kS + s], y[t * kD + c], x[i]);
}

// ---- phase 5: the two fused into one - the slow way -------------------------------------------------------------
// One workgroup per token, looping over all of the token's columns. For a prompt (T = 512) that's 512 workgroups and
// the GPU is busy; for one token (T = 1) it's ONE workgroup doing all 2560 x 4 updates while the rest of the GPU
// waits. The result is bit-identical to the separate pair, so every correctness test passes.
__global__ void fused_one_block_per_token(int T, const float *moe, const float *shared, const float *w, float *x,
                                          int wrong_weight) {
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < kD; c += blockDim.x) {
        const float y = moe[t * kD + c] + shared[t * kD + c];
        for (int s = 0; s < kS; ++s) {
            const int ws = wrong_weight ? 0 : s; // --mutate 2: every stream gets stream 0's weight
            float &xv = x[((size_t)t * kS + s) * kD + c];
            xv = fmaf(w[t * kS + ws], y, xv);
        }
    }
}

// ---- phase 7: the fusion done right - the columns spread over many workgroups ------------------------------------
// grid = (D / 256 column tiles, T tokens): even for one token that's 10 workgroups working at once.
__global__ void fused_column_tiles(int T, const float *moe, const float *shared, const float *w, float *x) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= kD || t >= T) return;
    const float y = moe[t * kD + c] + shared[t * kD + c];
    for (int s = 0; s < kS; ++s) {
        float &xv = x[((size_t)t * kS + s) * kD + c];
        xv = fmaf(w[t * kS + s], y, xv);
    }
}

// ---- host side ------------------------------------------------------------------------------------------------
enum class Variant { SeparatePair, FusedOneBlockPerToken, FusedColumnTiles };
static const char *name(Variant v) {
    switch (v) {
    case Variant::SeparatePair: return "separate add + inject (2 launches)";
    case Variant::FusedOneBlockPerToken: return "fused, one block per token";
    case Variant::FusedColumnTiles: return "fused, column tiles";
    }
    return "?";
}

struct Buffers {
    int T = 0;
    float *moe = nullptr, *shared = nullptr, *w = nullptr, *x = nullptr, *y = nullptr;
    explicit Buffers(int t) : T(t) {
        HIP_CHECK(hipMalloc(&moe, sizeof(float) * T * kD));
        HIP_CHECK(hipMalloc(&shared, sizeof(float) * T * kD));
        HIP_CHECK(hipMalloc(&w, sizeof(float) * T * kS));
        HIP_CHECK(hipMalloc(&x, sizeof(float) * T * kS * kD));
        HIP_CHECK(hipMalloc(&y, sizeof(float) * T * kD));
    }
    ~Buffers() {
        (void)hipFree(moe), (void)hipFree(shared), (void)hipFree(w), (void)hipFree(x), (void)hipFree(y);
    }
};

// Deterministic inputs, so every run (and every reader) sees the same numbers.
static std::vector<float> random_floats(size_t n, uint32_t seed) {
    std::vector<float> v(n);
    for (auto &f : v) {
        seed = seed * 1664525u + 1013904223u;
        f = (float)((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f; // in [-1, 1)
    }
    return v;
}

// mutate: 0 = correct, 1 = the column-tile grid one block too small, 2 = the wrong stream weight
static void launch(Variant v, Buffers &b, int mutate) {
    const int T = b.T;
    switch (v) {
    case Variant::SeparatePair:
        add_kernel<<<(T * kD + kThreads - 1) / kThreads, kThreads>>>(T, b.moe, b.shared, b.y);
        inject_kernel<<<(T * kS * kD + kThreads - 1) / kThreads, kThreads>>>(T, b.w, b.y, b.x);
        break;
    case Variant::FusedOneBlockPerToken:
        fused_one_block_per_token<<<T, kThreads>>>(T, b.moe, b.shared, b.w, b.x, mutate == 2);
        break;
    case Variant::FusedColumnTiles: {
        const int tiles = kD / kThreads - (mutate == 1 ? 1 : 0); // --mutate 1 leaves the last 256 columns alone
        fused_column_tiles<<<dim3(tiles, T), kThreads>>>(T, b.moe, b.shared, b.w, b.x);
        break;
    }
    }
    HIP_CHECK(hipGetLastError());
}

// ---- phases 2-3: the test - every variant against the reference, bit for bit -------------------------------------
static bool test(int mutate) {
    bool all_ok = true;
    for (int T : {1, 7, 512}) {
        const auto moe = random_floats((size_t)T * kD, 1), shared = random_floats((size_t)T * kD, 2),
                   w = random_floats((size_t)T * kS, 3), x0 = random_floats((size_t)T * kS * kD, 4);
        std::vector<float> want = x0;
        reference(T, moe.data(), shared.data(), w.data(), want.data());
        for (Variant v : {Variant::SeparatePair, Variant::FusedOneBlockPerToken, Variant::FusedColumnTiles}) {
            Buffers b(T);
            HIP_CHECK(hipMemcpy(b.moe, moe.data(), sizeof(float) * moe.size(), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(b.shared, shared.data(), sizeof(float) * shared.size(), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(b.w, w.data(), sizeof(float) * w.size(), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(b.x, x0.data(), sizeof(float) * x0.size(), hipMemcpyHostToDevice));
            launch(v, b, mutate);
            std::vector<float> got(x0.size());
            HIP_CHECK(hipMemcpy(got.data(), b.x, sizeof(float) * got.size(), hipMemcpyDeviceToHost));
            size_t bad = 0, first = 0;
            for (size_t i = 0; i < got.size(); ++i)
                if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0 && bad++ == 0) first = i;
            if (bad) {
                all_ok = false;
                std::printf("FAIL  T=%-3d %-36s %zu of %zu values differ (first at index %zu: got %.9g, want %.9g)\n",
                            T, name(v), bad, got.size(), first, got[first], want[first]);
            } else {
                std::printf("ok    T=%-3d %-36s all %zu values bit-identical to the reference\n", T, name(v),
                            got.size());
            }
        }
    }
    return all_ok;
}

// ---- phase 4: timing ------------------------------------------------------------------------------------------
static void bench(int reps) {
    std::printf("%-6s %-36s %10s %12s %10s\n", "tokens", "variant", "us/launch", "bytes moved", "GB/s");
    for (int T : {1, 512}) {
        Buffers b(T);
        HIP_CHECK(hipMemset(b.moe, 0, sizeof(float) * T * kD));
        HIP_CHECK(hipMemset(b.shared, 0, sizeof(float) * T * kD));
        HIP_CHECK(hipMemset(b.w, 0, sizeof(float) * T * kS));
        HIP_CHECK(hipMemset(b.x, 0, sizeof(float) * T * kS * kD));
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        for (Variant v : {Variant::SeparatePair, Variant::FusedOneBlockPerToken, Variant::FusedColumnTiles}) {
            for (int i = 0; i < 20; ++i) launch(v, b, 0); // warm up
            std::vector<float> runs;
            for (int r = 0; r < 7; ++r) { // the median of 7 measurements, each over `reps` launches
                HIP_CHECK(hipEventRecord(start));
                for (int i = 0; i < reps; ++i) launch(v, b, 0);
                HIP_CHECK(hipEventRecord(stop));
                HIP_CHECK(hipEventSynchronize(stop));
                float ms = 0;
                HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
                runs.push_back(ms * 1000.0f / reps);
            }
            std::sort(runs.begin(), runs.end());
            const double us = runs[3];
            // Bytes each variant must move: the pair writes y and reads it back; the fused kernels never store it.
            const double f = sizeof(float);
            const double x_rw = 2.0 * T * kS * kD * f, inputs = 2.0 * T * kD * f + T * kS * f;
            const double bytes = v == Variant::SeparatePair ? inputs + x_rw + 2.0 * T * kD * f : inputs + x_rw;
            std::printf("%-6d %-36s %10.2f %12.0f %10.1f\n", T, name(v), us, bytes, bytes / (us * 1e3));
        }
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
}

// ---- phase 6: a short run to capture in a profiler trace ----------------------------------------------------------
static void trace_run() {
    Buffers b(1);
    HIP_CHECK(hipMemset(b.x, 0, sizeof(float) * kS * kD));
    for (int round = 0; round < 5; ++round)
        for (Variant v : {Variant::SeparatePair, Variant::FusedOneBlockPerToken, Variant::FusedColumnTiles})
            launch(v, b, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::printf("ran each variant 5 times at T = 1 - open the trace and look for the three kernel names\n");
}

int main(int argc, char **argv) {
    bool do_test = false, do_bench = false, do_trace = false;
    int mutate = 0, reps = 200;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next_int = [&](const char *flag, int lo, int hi) {
            if (i + 1 >= argc) fail(std::string(flag) + " needs a value");
            const std::string v = argv[++i];
            char *end = nullptr;
            const long n = std::strtol(v.c_str(), &end, 10);
            if (end == v.c_str() || *end != '\0' || n < lo || n > hi)
                fail(std::string(flag) + " expects a whole number in " + std::to_string(lo) + ".." +
                     std::to_string(hi) + ", got '" + v + "'");
            return (int)n;
        };
        if (a == "--test") do_test = true;
        else if (a == "--bench") do_bench = true;
        else if (a == "--trace") do_trace = true;
        else if (a == "--mutate") mutate = next_int("--mutate", 1, 2), do_test = true;
        else if (a == "--reps") reps = next_int("--reps", 1, 100000);
        else fail("unknown option '" + a + "' (expected --test, --mutate N, --bench, --trace, --reps N)");
    }
    if (!do_test && !do_bench && !do_trace) do_test = true;

    int dev = 0;
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDevice(&dev));
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    std::printf("GPU: %s (%s), %d compute units reported\n\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);

    int rc = 0;
    if (do_test) {
        if (mutate) std::printf("mutation %d is on - the test below should FAIL\n", mutate);
        const bool ok = test(mutate);
        std::printf("\n%s\n\n", ok ? "all variants match the reference" : "the test caught a difference");
        if (!ok && !mutate) rc = 1;
        if (ok && mutate) {
            std::printf("mutation %d was NOT caught - the test is too weak\n", mutate);
            rc = 1;
        }
    }
    if (do_bench) bench(reps);
    if (do_trace) trace_run();
    return rc;
}
