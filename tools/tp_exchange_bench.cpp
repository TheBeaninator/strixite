// tp_exchange_bench: strixite-tp2 ST-1 / ST-N1 exchange microbenchmark (HIP + ibverbs over rail 0, RoCE v2, stock mlx5).
//
// Models the TP-N decode exchange: per forward, E (default 96) all-reduces of a BF16 partial of S bytes. Each exchange:
//   produce kernel  writes the rank's partial into send slot e%3, __threadfence_system, publishes ready_seq = e+1
//   communicator    (host thread) sees ready_seq, posts RDMA WRITE(_WITH_IMM) of the partial into window e%3 of every
//                   peer (chunked per --chunks, inline when <= --inline), polls the recv CQ for the peers' immediates
//                   and publishes arrival[p] = exchanges fully arrived from peer p (host-coherent words)
//   wait kernel     one wave, spins until arrival[p] >= e+1 for every peer
//   sum kernel      adds the N partials in fixed rank order (bit-identical on every rank), checks them exactly
// The GPU-stream stall per exchange = wait kernel exit - producer flag publish (GPU wall clock). Optional bandwidth
// load between exchanges (--loads inline) and/or on a second stream (stream). --prefill: 20/40 MiB exchanges, <= 32 MiB
// WRITEs. Windows: 3 per (peer, rank) rotating, header-free (gufo's scheme): a window is reused 3 exchanges later,
// which the all-reduce's own dependencies make safe. Every QP carries exactly one immediate per exchange (a zero-length
// WRITE_WITH_IMM when it got no chunk), so arrival = min over the peer's QPs of that QP's immediates.
//
// usage: tp_exchange_bench --rank R --peers 192.0.2.1,192.0.2.2[,..] [--dev DEVICE] [--gid-index auto|N]
//          [--port 18515] [--sizes 5120,10240,30720,61440] [--chunks 0,16384,32768,65536] [--inlines 0,220]
//          [--loads off,inline,inline+stream] [--load-bytes 67108864] [--qps-per-peer 1] [--exchanges 96]
//          [--forwards 200] [--warmup 20] [--mem host|device] [--prefill] [--timeout 120]
//        tp_exchange_bench --selftest            (CPU only: write plans, window rotation, exact sums)
// One JSON line per configuration on stdout; progress on stderr.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace tpx {

constexpr int kWindows = 3;
constexpr uint64_t kMaxWrite = 32ull << 20;  // never one WRITE above 32 MiB (plan v2)

struct Wr {
    int qp;
    uint64_t off, len;
    bool imm;
};
// The WRITEs of one exchange of S bytes on Q QPs: chunks of `chunk` bytes (0 = one WRITE, capped at kMaxWrite)
// round-robin over the QPs; the last WRITE on each QP carries the immediate; a QP with no chunk gets a zero-length one.
std::vector<Wr> plan_writes(uint64_t S, uint64_t chunk, int Q) {
    const uint64_t c = chunk == 0 ? std::min<uint64_t>(S, kMaxWrite) : std::min<uint64_t>(chunk, kMaxWrite);
    std::vector<Wr> w;
    int i = 0;
    for (uint64_t off = 0; off < S; off += c, ++i) w.push_back({i % Q, off, std::min(c, S - off), false});
    for (int q = 0; q < Q; ++q) {
        int last = -1;
        for (int k = 0; k < (int)w.size(); ++k)
            if (w[k].qp == q) last = k;
        if (last >= 0) w[last].imm = true;
        else w.push_back({q, 0, 0, true});
    }
    return w;
}
// Byte offset of window (src rank, exchange e) inside a rank's receive region.
uint64_t window_off(int src, uint64_t e, uint64_t slot) { return ((uint64_t)src * kWindows + e % kWindows) * slot; }
// The producer's value: exact in BF16 (<= 8 * 7 * 3 = 168), so the N-rank sum is exact in FP32.
inline float part_value(int rank, uint64_t e, uint64_t i) { return (float)((rank + 1) * (int)(i % 7 + 1) * (int)(e % 3 + 1)); }
inline float sum_value(int N, uint64_t e, uint64_t i) { return (float)(N * (N + 1) / 2 * (int)(i % 7 + 1) * (int)(e % 3 + 1)); }

int selftest() {
    int bad = 0;
    for (uint64_t S : {5120ull, 10240ull, 30720ull, 61440ull, 20ull << 20, 40ull << 20})
        for (uint64_t ch : {0ull, 16384ull, 32768ull, 65536ull})
            for (int Q : {1, 2, 4}) {
                auto w = plan_writes(S, ch, Q);
                std::vector<int> cover(S / 1024 + 1, 0), imms(Q, 0);
                uint64_t total = 0;
                for (size_t k = 0; k < w.size(); ++k) {
                    total += w[k].len;
                    if (w[k].len > kMaxWrite) ++bad;
                    if (w[k].imm) {
                        ++imms[w[k].qp];
                        for (size_t j = k + 1; j < w.size(); ++j)
                            if (w[j].qp == w[k].qp && w[j].len) ++bad;  // imm must be the QP's last
                    }
                }
                if (total != S) ++bad, fprintf(stderr, "cover S=%llu ch=%llu Q=%d total=%llu\n", (unsigned long long)S,
                                                (unsigned long long)ch, Q, (unsigned long long)total);
                for (int q = 0; q < Q; ++q)
                    if (imms[q] != 1) ++bad, fprintf(stderr, "imm count S=%llu Q=%d q=%d n=%d\n", (unsigned long long)S, Q, q, imms[q]);
            }
    // windows: distinct for 3 consecutive exchanges and per source; disjoint across sources
    for (int N : {2, 4}) {
        const uint64_t slot = 61440;
        for (uint64_t e = 0; e < 10; ++e)
            for (int a = 0; a < N; ++a)
                for (int b = 0; b < N; ++b)
                    for (uint64_t d = 0; d < 3; ++d) {
                        if (a == b && d == 0) continue;
                        if (window_off(a, e, slot) == window_off(b, e + d, slot) && !(a == b && d % 3 == 0)) ++bad;
                    }
        // fixed-order sum is exact and equals the closed form
        for (uint64_t e = 0; e < 6; ++e)
            for (uint64_t i = 0; i < 100; ++i) {
                float s = 0;
                for (int r = 0; r < N; ++r) s += part_value(r, e, i);
                if (s != sum_value(N, e, i)) ++bad;
            }
    }
    printf("{\"selftest\":\"%s\",\"failures\":%d}\n", bad ? "FAIL" : "ok", bad);
    return bad ? 1 : 0;
}

}  // namespace tpx

#ifdef TPX_SELFTEST_ONLY
int main() { return tpx::selftest(); }
#else

#include <hip/hip_runtime.h>
#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <thread>

using namespace tpx;

#define DIE(...) (fprintf(stderr, "tp_exchange_bench: " __VA_ARGS__), fputc('\n', stderr), exit(2))
#define HIPC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) DIE("%s: %s", #x, hipGetErrorString(e_)); } while (0)

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
std::vector<std::string> split(const std::string &s) {
    std::vector<std::string> r;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(',', a);
        if (b == std::string::npos) b = s.size();
        if (b > a) r.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return r;
}
std::vector<uint64_t> splitu(const std::string &s) {
    std::vector<uint64_t> r;
    for (auto &x : split(s)) r.push_back(std::stoull(x));
    return r;
}

// ---------------- GPU kernels ----------------
__global__ void k_produce(uint16_t *dst, uint64_t n, int rank, uint64_t e, volatile uint64_t *ready, uint64_t *t_prod,
                          uint64_t slot_i) {
    for (uint64_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = (float)((rank + 1) * (int)(i % 7 + 1) * (int)(e % 3 + 1));
        dst[i] = (uint16_t)(__float_as_uint(v) >> 16);  // exact: v has <= 8 significant bits
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *ready = e + 1;
        __threadfence_system();
        t_prod[slot_i] = wall_clock64();
    }
}
__global__ void k_fill(uint16_t *dst, uint64_t n, int rank, uint64_t e) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        float v = (float)((rank + 1) * (int)(i % 7 + 1) * (int)(e % 3 + 1));
        dst[i] = (uint16_t)(__float_as_uint(v) >> 16);
    }
}
__global__ void k_flag(uint64_t e, volatile uint64_t *ready, uint64_t *t_prod, uint64_t slot_i) {
    __threadfence_system();
    *ready = e + 1;
    __threadfence_system();
    t_prod[slot_i] = wall_clock64();
}
__global__ void k_wait(volatile uint64_t *arrival, int N, int rank, uint64_t e, uint64_t *t_done, uint64_t slot_i) {
    if (threadIdx.x == 0) {
        for (int p = 0; p < N; ++p) {
            if (p == rank) continue;
            while (arrival[p] < e + 1) __builtin_amdgcn_s_sleep(1);
        }
        __threadfence_system();
        t_done[slot_i] = wall_clock64();
    }
}
__global__ void k_sum(const uint16_t *recv, const uint16_t *self, uint64_t slot_elems, int N, int rank, uint64_t e,
                      uint64_t n, uint16_t *out, unsigned *err) {
    const uint64_t w = e % kWindows;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        float s = 0;
        for (int r = 0; r < N; ++r) {
            const uint16_t *src = r == rank ? self : recv + ((uint64_t)r * kWindows + w) * slot_elems;
            s += __uint_as_float((uint32_t)src[i] << 16);
        }
        const float want = (float)(N * (N + 1) / 2 * (int)(i % 7 + 1) * (int)(e % 3 + 1));
        if (s != want) atomicAdd(err, 1u);
        out[i] = (uint16_t)(__float_as_uint(s) >> 16);
    }
}
__global__ void k_load(const float4 *buf, uint64_t n4, uint64_t off4, uint64_t total4, float *sink) {
    float4 acc = {0, 0, 0, 0};
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n4; i += (uint64_t)gridDim.x * blockDim.x) {
        float4 v = buf[(off4 + i) % total4];
        acc.x += v.x, acc.y += v.y, acc.z += v.z, acc.w += v.w;
    }
    if (acc.x + acc.y + acc.z + acc.w == 1.2345e-30f) *sink = acc.x;
}
__global__ void k_load_forever(const float4 *buf, uint64_t total4, volatile int *stop, float *sink) {
    float acc = 0;
    while (!*stop)
        for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < total4 && !*stop;
             i += (uint64_t)gridDim.x * blockDim.x) {
            float4 v = buf[i];
            acc += v.x + v.w;
        }
    if (acc == 1.2345e-30f) *sink = acc;
}

// ---------------- out-of-band TCP mesh ----------------
int tcp_listen(const std::string &ip, int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET, a.sin_port = htons(port);
    inet_pton(AF_INET, ip.c_str(), &a.sin_addr);
    if (bind(s, (sockaddr *)&a, sizeof a) || listen(s, 16)) DIE("listen %s:%d: %s", ip.c_str(), port, strerror(errno));
    return s;
}
int tcp_connect(const std::string &ip, int port, double deadline) {
    while (now_s() < deadline) {
        int s = socket(AF_INET, SOCK_STREAM, 0), one = 1;
        sockaddr_in a{};
        a.sin_family = AF_INET, a.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &a.sin_addr);
        if (connect(s, (sockaddr *)&a, sizeof a) == 0) {
            setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            return s;
        }
        close(s);
        usleep(200000);
    }
    DIE("timed out connecting to %s:%d", ip.c_str(), port);
}
void xfer(int fd, void *p, size_t n, bool send_, double deadline) {
    auto *b = (uint8_t *)p;
    while (n) {
        pollfd pf{fd, (short)(send_ ? POLLOUT : POLLIN), 0};
        int ms = (int)std::max(0.0, (deadline - now_s()) * 1000);
        if (poll(&pf, 1, ms) <= 0) DIE("out-of-band socket timed out");
        ssize_t r = send_ ? send(fd, b, n, MSG_NOSIGNAL) : recv(fd, b, n, 0);
        if (r <= 0) DIE("out-of-band socket closed: %s", r < 0 ? strerror(errno) : "eof");
        b += r, n -= (size_t)r;
    }
}

struct QpInfo {
    uint32_t qpn, psn;
};
struct PairInfo {
    int32_t rank;
    uint8_t gid[16];
    uint32_t rkey;
    uint64_t region;  // the sender-to-me receive region base (remote address the peer writes to)
    QpInfo qp[8];
};

struct Peer {
    int fd = -1;
    std::vector<ibv_qp *> qps;
    PairInfo remote{};
    std::vector<uint64_t> imm_count;   // per QP: immediates received (exchanges)
    std::vector<uint64_t> outstanding;  // per QP: unsignaled + signaled WRs not yet retired
    std::vector<uint64_t> since_sig;
};

int pick_gid(const std::string &dev, const std::string &ip) {
    in_addr a;
    inet_pton(AF_INET, ip.c_str(), &a);
    const uint8_t *b = (const uint8_t *)&a;
    char want[64];
    snprintf(want, sizeof want, "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x", b[0], b[1], b[2], b[3]);
    for (int i = 0; i < 256; ++i) {
        std::ifstream g("/sys/class/infiniband/" + dev + "/ports/1/gids/" + std::to_string(i));
        std::ifstream t("/sys/class/infiniband/" + dev + "/ports/1/gid_attrs/types/" + std::to_string(i));
        std::string gs, ts;
        if (!std::getline(g, gs)) continue;
        std::getline(t, ts);
        if (gs == want && ts.find("v2") != std::string::npos) return i;
    }
    DIE("no RoCE v2 GID for %s on %s (wanted %s)", ip.c_str(), dev.c_str(), want);
}

void pct(std::vector<double> v, double *p50, double *p90, double *p99, double *mx, double *mean) {
    std::sort(v.begin(), v.end());
    auto at = [&](double q) { return v.empty() ? 0 : v[(size_t)std::min<double>(v.size() - 1, q * (v.size() - 1) + 0.5)]; };
    *p50 = at(0.5), *p90 = at(0.9), *p99 = at(0.99), *mx = v.empty() ? 0 : v.back();
    double s = 0;
    for (double x : v) s += x;
    *mean = v.empty() ? 0 : s / v.size();
}

}  // namespace

int main(int argc, char **argv) {
    std::string dev = "mlx5_0", peers_s, gid_s = "auto", mem = "host";
    std::string sizes_s = "5120,10240,30720,61440", chunks_s = "0,16384,32768,65536", inl_s = "0,220",
                loads_s = "off,inline,inline+stream";
    int rank = -1, port = 18515, Q = 1;
    uint64_t E = 96, F = 200, W = 20, load_bytes = 64ull << 20;
    double timeout = 120;
    bool prefill = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto v = [&]() -> std::string { if (i + 1 >= argc) DIE("%s needs a value", a.c_str()); return argv[++i]; };
        if (a == "--selftest") return selftest();
        else if (a == "--rank") rank = std::stoi(v());
        else if (a == "--peers") peers_s = v();
        else if (a == "--dev") dev = v();
        else if (a == "--gid-index") gid_s = v();
        else if (a == "--port") port = std::stoi(v());
        else if (a == "--sizes") sizes_s = v();
        else if (a == "--chunks") chunks_s = v();
        else if (a == "--inlines") inl_s = v();
        else if (a == "--loads") loads_s = v();
        else if (a == "--load-bytes") load_bytes = std::stoull(v());
        else if (a == "--qps-per-peer") Q = std::stoi(v());
        else if (a == "--exchanges") E = std::stoull(v());
        else if (a == "--forwards") F = std::stoull(v());
        else if (a == "--warmup") W = std::stoull(v());
        else if (a == "--mem") mem = v();
        else if (a == "--timeout") timeout = std::stod(v());
        else if (a == "--prefill") prefill = true;
        else DIE("unknown option %s (see the header of tools/tp_exchange_bench.cpp)", a.c_str());
    }
    if (prefill) {
        if (sizes_s == "5120,10240,30720,61440") sizes_s = std::to_string(20ull << 20) + "," + std::to_string(40ull << 20);
        if (chunks_s == "0,16384,32768,65536") chunks_s = "0," + std::to_string(4ull << 20);
        if (inl_s == "0,220") inl_s = "0";
        if (loads_s == "off,inline,inline+stream") loads_s = "off";
        if (E == 96) E = 4;
        if (F == 200) F = 20;
        if (W == 20) W = 3;
    }
    auto peers = split(peers_s);
    const int N = (int)peers.size();
    if (N < 2 || rank < 0 || rank >= N) DIE("--rank R --peers ip0,ip1,... (N >= 2, 0 <= R < N)");
    if (Q < 1 || Q > 8) DIE("--qps-per-peer 1..8");
    auto sizes = splitu(sizes_s), chunks = splitu(chunks_s), inls = splitu(inl_s);
    auto loads = split(loads_s);
    const uint64_t slot = (*std::max_element(sizes.begin(), sizes.end()) + 4095) & ~4095ull;
    const double deadline = now_s() + timeout;

    // ---- HIP memory ----
    HIPC(hipSetDevice(0));
    int clk_khz = 0;
    HIPC(hipDeviceGetAttribute(&clk_khz, hipDeviceAttributeWallClockRate, 0));
    const uint64_t send_bytes = slot * kWindows, recv_bytes = slot * kWindows * N;
    uint8_t *send_h = nullptr, *recv_h = nullptr, *send_d = nullptr, *recv_d = nullptr;
    uint64_t *flags_h = nullptr, *flags_d = nullptr;  // [0] ready_seq, [8..8+N) arrival
    bool dev_mem = mem == "device";
    HIPC(hipHostMalloc((void **)&flags_h, 4096, hipHostMallocCoherent | hipHostMallocMapped));
    memset(flags_h, 0, 4096);
    HIPC(hipHostGetDevicePointer((void **)&flags_d, flags_h, 0));
    volatile uint64_t *ready = flags_h, *arrival = flags_h + 8;
    auto host_alloc = [&]() {
        HIPC(hipHostMalloc((void **)&send_h, send_bytes, hipHostMallocCoherent | hipHostMallocMapped));
        HIPC(hipHostMalloc((void **)&recv_h, recv_bytes, hipHostMallocCoherent | hipHostMallocMapped));
        HIPC(hipHostGetDevicePointer((void **)&send_d, send_h, 0));
        HIPC(hipHostGetDevicePointer((void **)&recv_d, recv_h, 0));
    };
    if (dev_mem) {
        HIPC(hipMalloc((void **)&send_d, send_bytes));
        HIPC(hipMalloc((void **)&recv_d, recv_bytes));
        send_h = send_d, recv_h = recv_d;  // the NIC is handed the device VA (works only with peer-mem / dmabuf)
    } else host_alloc();

    // ---- verbs ----
    int ndev = 0;
    ibv_device **dl = ibv_get_device_list(&ndev);
    ibv_context *ctx = nullptr;
    for (int i = 0; i < ndev; ++i)
        if (dev == ibv_get_device_name(dl[i])) ctx = ibv_open_device(dl[i]);
    if (!ctx) DIE("RDMA device %s not found (%d devices; is the stock mlx5 provider on the path?)", dev.c_str(), ndev);
    ibv_port_attr pa;
    if (ibv_query_port(ctx, 1, &pa)) DIE("ibv_query_port");
    const int gidx = gid_s == "auto" ? pick_gid(dev, peers[rank]) : std::stoi(gid_s);
    ibv_gid my_gid;
    if (ibv_query_gid(ctx, 1, gidx, &my_gid)) DIE("ibv_query_gid %d", gidx);
    ibv_pd *pd = ibv_alloc_pd(ctx);
    const int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    ibv_mr *smr = ibv_reg_mr(pd, send_h, send_bytes, acc), *rmr = ibv_reg_mr(pd, recv_h, recv_bytes, acc);
    if (dev_mem && (!smr || !rmr)) {
        fprintf(stderr, "tp_exchange_bench: ibv_reg_mr on hipMalloc memory failed (%s): falling back to --mem host\n",
                strerror(errno));
        if (smr) ibv_dereg_mr(smr);
        if (rmr) ibv_dereg_mr(rmr);
        HIPC(hipFree(send_d));
        HIPC(hipFree(recv_d));
        host_alloc();
        dev_mem = false, mem = "host(fallback)";
        smr = ibv_reg_mr(pd, send_h, send_bytes, acc), rmr = ibv_reg_mr(pd, recv_h, recv_bytes, acc);
    }
    if (!smr || !rmr) DIE("ibv_reg_mr (%llu + %llu bytes): %s - raise the memlock limit (LimitMEMLOCK=infinity)",
                          (unsigned long long)send_bytes, (unsigned long long)recv_bytes, strerror(errno));
    const int SQD = 4096, RQD = 1024;
    ibv_cq *scq = ibv_create_cq(ctx, SQD * Q * N, nullptr, nullptr, 0), *rcq = ibv_create_cq(ctx, RQD * Q * N, nullptr, nullptr, 0);
    if (!scq || !rcq) DIE("ibv_create_cq");
    uint64_t max_inline = *std::max_element(inls.begin(), inls.end());
    std::vector<Peer> P(N);
    std::vector<std::array<uint32_t, 3>> qp_map;  // {qp_num, peer, index}
    for (int p = 0; p < N; ++p) {
        if (p == rank) continue;
        for (int q = 0; q < Q; ++q) {
            ibv_qp_init_attr ia{};
            ia.send_cq = scq, ia.recv_cq = rcq, ia.qp_type = IBV_QPT_RC;
            ia.cap.max_send_wr = SQD, ia.cap.max_recv_wr = RQD, ia.cap.max_send_sge = 1, ia.cap.max_recv_sge = 1;
            ia.cap.max_inline_data = (uint32_t)max_inline;
            ibv_qp *qp = ibv_create_qp(pd, &ia);
            if (!qp && max_inline) {
                fprintf(stderr, "tp_exchange_bench: QP with %llu inline bytes refused; retrying with 0\n", (unsigned long long)max_inline);
                ia.cap.max_inline_data = 0, max_inline = 0;
                qp = ibv_create_qp(pd, &ia);
            }
            if (!qp) DIE("ibv_create_qp: %s", strerror(errno));
            max_inline = std::min<uint64_t>(max_inline, ia.cap.max_inline_data);
            ibv_qp_attr at{};
            at.qp_state = IBV_QPS_INIT, at.pkey_index = 0, at.port_num = 1, at.qp_access_flags = acc;
            if (ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) DIE("QP INIT");
            qp_map.push_back({qp->qp_num, (uint32_t)p, (uint32_t)q});
            P[p].qps.push_back(qp);
        }
        P[p].imm_count.assign(Q, 0), P[p].outstanding.assign(Q, 0), P[p].since_sig.assign(Q, 0);
    }
    // TCP mesh: listen on my rail-0 address, connect to lower ranks, accept higher ones
    int ls = tcp_listen(peers[rank], port);
    for (int p = 0; p < rank; ++p) {
        P[p].fd = tcp_connect(peers[p], port, deadline);
        int32_t me = rank;
        xfer(P[p].fd, &me, 4, true, deadline);
    }
    for (int k = rank + 1; k < N; ++k) {
        pollfd pf{ls, POLLIN, 0};
        if (poll(&pf, 1, (int)std::max(0.0, (deadline - now_s()) * 1000)) <= 0) DIE("timed out waiting for peer connections");
        int fd = accept(ls, nullptr, nullptr), one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        int32_t r = -1;
        xfer(fd, &r, 4, false, deadline);
        if (r <= rank || r >= N) DIE("bad peer rank %d", r);
        P[r].fd = fd;
    }
    close(ls);
    srand48(getpid() ^ (long)(now_s() * 1e6));
    for (int p = 0; p < N; ++p) {
        if (p == rank) continue;
        PairInfo mine{};
        mine.rank = rank;
        memcpy(mine.gid, my_gid.raw, 16);
        mine.rkey = rmr->rkey;
        mine.region = (uint64_t)(uintptr_t)recv_h;  // peer p writes at window_off(p, e) inside it
        for (int q = 0; q < Q; ++q) mine.qp[q] = {P[p].qps[q]->qp_num, (uint32_t)(lrand48() & 0xffffff)};
        xfer(P[p].fd, &mine, sizeof mine, true, deadline);
        xfer(P[p].fd, &P[p].remote, sizeof mine, false, deadline);
        if (P[p].remote.rank != p) DIE("peer %d answered as rank %d", p, P[p].remote.rank);
        for (int q = 0; q < Q; ++q) {
            ibv_qp_attr at{};
            at.qp_state = IBV_QPS_RTR, at.path_mtu = std::min(pa.active_mtu, IBV_MTU_4096);
            at.dest_qp_num = P[p].remote.qp[q].qpn, at.rq_psn = P[p].remote.qp[q].psn;
            at.max_dest_rd_atomic = 1, at.min_rnr_timer = 12;
            at.ah_attr.is_global = 1, at.ah_attr.port_num = 1;
            memcpy(at.ah_attr.grh.dgid.raw, P[p].remote.gid, 16);
            at.ah_attr.grh.sgid_index = (uint8_t)gidx, at.ah_attr.grh.hop_limit = 64;
            if (ibv_modify_qp(P[p].qps[q], &at, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                                    IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
                DIE("QP RTR (peer %d): %s", p, strerror(errno));
            at.qp_state = IBV_QPS_RTS, at.timeout = 14, at.retry_cnt = 7, at.rnr_retry = 7, at.sq_psn = mine.qp[q].psn;
            at.max_rd_atomic = 1;
            if (ibv_modify_qp(P[p].qps[q], &at, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                                    IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
                DIE("QP RTS (peer %d): %s", p, strerror(errno));
            for (int k = 0; k < RQD; ++k) {
                ibv_recv_wr rw{}, *bad;
                rw.wr_id = 0;
                if (ibv_post_recv(P[p].qps[q], &rw, &bad)) DIE("ibv_post_recv");
            }
        }
    }
    auto barrier = [&]() {
        for (int p = 0; p < N; ++p) {
            if (p == rank) continue;
            uint8_t b = 1;
            xfer(P[p].fd, &b, 1, true, now_s() + timeout);
            xfer(P[p].fd, &b, 1, false, now_s() + timeout);
        }
    };
    barrier();
    fprintf(stderr, "tp_exchange_bench: rank %d/%d up on %s gid %d mtu %d, mem %s, inline max %llu, %d QP(s)/peer, clock %d kHz\n",
            rank, N, dev.c_str(), gidx, 128 << pa.active_mtu, mem.c_str(), (unsigned long long)max_inline, Q, clk_khz);

    // ---- GPU working set ----
    hipStream_t cs, ls2;
    HIPC(hipStreamCreateWithFlags(&cs, hipStreamNonBlocking));
    HIPC(hipStreamCreateWithFlags(&ls2, hipStreamNonBlocking));
    const uint64_t lbuf_bytes = 1ull << 30;
    float4 *lbuf;
    float *sink;
    unsigned *err_d;
    uint16_t *out_d;
    int *stop_h, *stop_d;
    HIPC(hipMalloc((void **)&lbuf, lbuf_bytes));
    HIPC(hipMemset(lbuf, 0, lbuf_bytes));
    HIPC(hipMalloc((void **)&sink, 64));
    HIPC(hipMalloc((void **)&err_d, 4));
    HIPC(hipMalloc((void **)&out_d, slot));
    HIPC(hipHostMalloc((void **)&stop_h, 64, hipHostMallocCoherent | hipHostMallocMapped));
    HIPC(hipHostGetDevicePointer((void **)&stop_d, stop_h, 0));
    const uint64_t maxX = (W + F) * E;
    uint64_t *t_prod, *t_done;
    HIPC(hipMalloc((void **)&t_prod, maxX * 8));
    HIPC(hipMalloc((void **)&t_done, maxX * 8));
    std::vector<hipEvent_t> ev0(W + F), ev1(W + F);
    for (auto &e : ev0) HIPC(hipEventCreate(&e));
    for (auto &e : ev1) HIPC(hipEventCreate(&e));

    uint64_t seq = 0;  // global exchange counter, identical on every rank (same sweep everywhere)
    for (uint64_t S : sizes)
        for (uint64_t chunk : chunks)
            for (uint64_t inl : inls)
                for (const std::string &load : loads) {
                    if (chunk && chunk >= S && std::count(chunks.begin(), chunks.end(), 0ull)) continue;  // = single WRITE
                    if (inl > max_inline) continue;
                    const auto plan = plan_writes(S, chunk, Q);
                    const uint64_t X = (W + F) * E, base = seq;
                    const bool linl = load.find("inline") != std::string::npos, lstr = load.find("stream") != std::string::npos;
                    // "+noex": the same forward without the exchanges (partials filled, never published, no wait / sum) -
                    // the baseline for the per-exchange GPU-stream cost (fwd_ms with minus without, / exchanges).
                    const bool noex = load.find("noex") != std::string::npos;
                    HIPC(hipMemset(err_d, 0, 4));
                    barrier();
                    std::atomic<bool> comm_fail{false};
                    auto lookup = [&](uint32_t qpn, int &pp, int &qq) {
                        for (auto &m : qp_map)
                            if (m[0] == qpn) { pp = (int)m[1], qq = (int)m[2]; return; }
                        DIE("completion on unknown QP %u", qpn);
                    };
                    std::thread comm([&]() {
                        if (noex) return;  // nothing is published: no WRITEs, no immediates
                        // post exchange e's WRITEs once the GPU published it; harvest immediates throughout
                        uint64_t next = base;
                        ibv_wc wc[64];
                        const double tdead = now_s() + timeout + 60.0 * (double)X / 1e5;
                        while (true) {
                            bool done_recv = true;
                            for (int p = 0; p < N; ++p)
                                if (p != rank && arrival[p] < base + X) done_recv = false;
                            if (next == base + X && done_recv) break;
                            if (now_s() > tdead) { comm_fail = true; fprintf(stderr, "comm thread: timeout at exchange %llu\n", (unsigned long long)next); return; }
                            if (next < base + X && *ready >= next + 1) {
                                const uint64_t w = next % kWindows;
                                for (int p = 0; p < N; ++p) {
                                    if (p == rank) continue;
                                    Peer &pr = P[p];
                                    for (const Wr &x : plan) {
                                        while (pr.outstanding[x.qp] > (uint64_t)SQD - 256) {  // reap send completions
                                            int n = ibv_poll_cq(scq, 64, wc);
                                            for (int k = 0; k < n; ++k) {
                                                if (wc[k].status) { fprintf(stderr, "send WC error %s\n", ibv_wc_status_str(wc[k].status)); comm_fail = true; return; }
                                                int pp = 0, qq = 0;
                                                lookup(wc[k].qp_num, pp, qq);
                                                P[pp].outstanding[qq] -= wc[k].wr_id;
                                            }
                                        }
                                        ibv_sge sg{(uint64_t)(uintptr_t)(send_h + w * slot + x.off), (uint32_t)x.len, smr->lkey};
                                        ibv_send_wr wr{}, *bad;
                                        wr.sg_list = x.len ? &sg : nullptr, wr.num_sge = x.len ? 1 : 0;
                                        wr.opcode = x.imm ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE;
                                        wr.imm_data = htonl((uint32_t)next);
                                        wr.wr.rdma.remote_addr = pr.remote.region + window_off(rank, next, slot) + x.off;
                                        wr.wr.rdma.rkey = pr.remote.rkey;
                                        ++pr.since_sig[x.qp], ++pr.outstanding[x.qp];
                                        if (x.imm || pr.since_sig[x.qp] >= 64) {
                                            wr.send_flags |= IBV_SEND_SIGNALED, wr.wr_id = pr.since_sig[x.qp];
                                            pr.since_sig[x.qp] = 0;
                                        }
                                        if (inl && x.len && x.len <= inl) wr.send_flags |= IBV_SEND_INLINE;
                                        if (int rc = ibv_post_send(pr.qps[x.qp], &wr, &bad)) {
                                            fprintf(stderr, "ibv_post_send: %s\n", strerror(rc)); comm_fail = true; return;
                                        }
                                    }
                                }
                                ++next;
                            }
                            int n = ibv_poll_cq(rcq, 64, wc);
                            for (int k = 0; k < n; ++k) {
                                if (wc[k].status) { fprintf(stderr, "recv WC error %s\n", ibv_wc_status_str(wc[k].status)); comm_fail = true; return; }
                                int pp = 0, qq = 0;
                                                lookup(wc[k].qp_num, pp, qq);
                                Peer &pr = P[pp];
                                if (ntohl(wc[k].imm_data) != (uint32_t)pr.imm_count[qq]) {
                                    fprintf(stderr, "immediate %u from peer %d QP %d, expected %u\n", ntohl(wc[k].imm_data), pp, qq, (uint32_t)pr.imm_count[qq]);
                                    comm_fail = true; return;
                                }
                                ++pr.imm_count[qq];
                                ibv_recv_wr rw{}, *bad;
                                ibv_post_recv(pr.qps[qq], &rw, &bad);
                                arrival[pp] = *std::min_element(pr.imm_count.begin(), pr.imm_count.end());
                            }
                            int s = ibv_poll_cq(scq, 64, wc);
                            for (int k = 0; k < s; ++k) {
                                if (wc[k].status) { fprintf(stderr, "send WC error %s\n", ibv_wc_status_str(wc[k].status)); comm_fail = true; return; }
                                int pp = 0, qq = 0;
                                                lookup(wc[k].qp_num, pp, qq);
                                P[pp].outstanding[qq] -= wc[k].wr_id;
                            }
                        }
                    });
                    *stop_h = 0;
                    if (lstr) hipLaunchKernelGGL(k_load_forever, dim3(16), dim3(256), 0, ls2, lbuf, lbuf_bytes / 16, stop_d, sink);
                    const uint64_t elems = S / 2;
                    uint64_t loff = 0;
                    const double t0 = now_s();
                    for (uint64_t f = 0; f < W + F; ++f) {
                        HIPC(hipEventRecord(ev0[f], cs));
                        for (uint64_t x = 0; x < E; ++x) {
                            const uint64_t e = seq + f * E + x, li = f * E + x;
                            if (linl) {
                                hipLaunchKernelGGL(k_load, dim3(256), dim3(256), 0, cs, lbuf, load_bytes / 16, loff,
                                                   lbuf_bytes / 16, sink);
                                loff = (loff + load_bytes / 16) % (lbuf_bytes / 16);
                            }
                            if (noex) {
                                hipLaunchKernelGGL(k_fill, dim3(1), dim3(1024), 0, cs, (uint16_t *)(send_d + (e % kWindows) * slot),
                                                   elems, rank, e);
                                continue;
                            }
                            if (elems <= 65536) {  // decode sizes: one block fills and publishes (one launch)
                                hipLaunchKernelGGL(k_produce, dim3(1), dim3(1024), 0, cs, (uint16_t *)(send_d + (e % kWindows) * slot),
                                                   elems, rank, e, flags_d, t_prod, li);
                            } else {  // prefill sizes: fill on the whole GPU, then publish
                                hipLaunchKernelGGL(k_fill, dim3(1024), dim3(256), 0, cs, (uint16_t *)(send_d + (e % kWindows) * slot),
                                                   elems, rank, e);
                                hipLaunchKernelGGL(k_flag, dim3(1), dim3(1), 0, cs, e, flags_d, t_prod, li);
                            }
                            hipLaunchKernelGGL(k_wait, dim3(1), dim3(32), 0, cs, flags_d + 8, N, rank, e, t_done, li);
                            hipLaunchKernelGGL(k_sum, dim3((unsigned)std::min<uint64_t>(1024, (elems + 255) / 256)), dim3(256), 0,
                                               cs, (const uint16_t *)recv_d, (const uint16_t *)(send_d + (e % kWindows) * slot),
                                               slot / 2, N, rank, e, elems, out_d, err_d);
                        }
                        HIPC(hipEventRecord(ev1[f], cs));
                        HIPC(hipStreamSynchronize(cs));  // between forwards only (not on the exchange path)
                        if (comm_fail) break;
                    }
                    const double wall = now_s() - t0;
                    *stop_h = 1;
                    HIPC(hipStreamSynchronize(ls2));
                    comm.join();
                    if (comm_fail) DIE("communicator failed (config S=%llu chunk=%llu inline=%llu load=%s)",
                                       (unsigned long long)S, (unsigned long long)chunk, (unsigned long long)inl, load.c_str());
                    if (!noex) seq += X;  // a no-exchange run consumes no sequence numbers
                    std::vector<uint64_t> tp(X), td(X);
                    unsigned errs = 0;
                    HIPC(hipMemcpy(tp.data(), t_prod, X * 8, hipMemcpyDeviceToHost));
                    HIPC(hipMemcpy(td.data(), t_done, X * 8, hipMemcpyDeviceToHost));
                    HIPC(hipMemcpy(&errs, err_d, 4, hipMemcpyDeviceToHost));
                    std::vector<double> stall, fwd, fwdx;
                    for (uint64_t f = W; f < W + F; ++f) {
                        double sx = 0;
                        for (uint64_t x = 0; x < E; ++x) {
                            const uint64_t li = f * E + x;
                            const double us = (double)(int64_t)(td[li] - tp[li]) * 1000.0 / clk_khz;
                            stall.push_back(us), sx += us;
                        }
                        float ms = 0;
                        HIPC(hipEventElapsedTime(&ms, ev0[f], ev1[f]));
                        fwd.push_back(ms), fwdx.push_back(sx / 1000.0);
                    }
                    double c50, c90, c99, cmx, cmean, f50, f90, f99, fmx, fmean, x50, x90, x99, xmx, xmean;
                    pct(stall, &c50, &c90, &c99, &cmx, &cmean);
                    pct(fwd, &f50, &f90, &f99, &fmx, &fmean);
                    pct(fwdx, &x50, &x90, &x99, &xmx, &xmean);
                    printf("{\"mode\":\"%s\",\"N\":%d,\"rank\":%d,\"size_bytes\":%llu,\"chunk_bytes\":%llu,\"writes\":%zu,"
                           "\"inline\":%llu,\"load\":\"%s\",\"load_bytes\":%llu,\"qps_per_peer\":%d,\"mem\":\"%s\","
                           "\"exchanges_per_forward\":%llu,\"forwards\":%llu,\"c_p50_us\":%.2f,\"c_p90_us\":%.2f,"
                           "\"c_p99_us\":%.2f,\"c_max_us\":%.2f,\"c_mean_us\":%.2f,\"fwd_ms_p50\":%.3f,\"fwd_ms_p99\":%.3f,"
                           "\"fwd_exchange_ms_p50\":%.3f,\"fwd_exchange_ms_p99\":%.3f,\"gbps_at_p50\":%.2f,\"sum_errors\":%u,"
                           "\"wall_s\":%.1f,\"clock_khz\":%d}\n",
                           prefill ? "prefill" : "decode", N, rank, (unsigned long long)S, (unsigned long long)chunk,
                           plan.size(), (unsigned long long)inl, load.c_str(), (unsigned long long)(linl ? load_bytes : 0), Q,
                           mem.c_str(), (unsigned long long)E, (unsigned long long)F, c50, c90, c99, cmx, cmean, f50, f99,
                           x50, x99, c50 > 0 ? S * 8.0 / (c50 * 1000.0) : 0.0, errs, wall, clk_khz);
                    fflush(stdout);
                    if (errs) fprintf(stderr, "tp_exchange_bench: %u wrong sums in this configuration\n", errs);
                }
    barrier();
    return 0;
}

#endif
