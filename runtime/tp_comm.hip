// strixite-tp2: TpComm (runtime/tp_comm.hpp) - HIP kernels + ibverbs over rail 0. The transport is
// tools/tp_exchange_bench's (ST-1: c_p50 15.1 us per 5 KiB exchange under load), made a library.

#include "runtime/tp_comm.hpp"

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "kernels/logits_topk.hpp"

#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

namespace strix {

namespace {

constexpr int kWindows = 3;
constexpr uint64_t kRing = 1u << 16;  // queued exchange sizes the comm thread hasn't posted yet
constexpr int kSqDepth = 4096, kRqDepth = 1024;
constexpr uint32_t kPoisonImm = 0xffffffffu;  // a zero-length WRITE_WITH_IMM carrying it poisons the receiver

// The immediate of exchange seq of `bytes`: seq's low 20 bits, then a 12-bit tag of the size (never the poison word).
uint32_t exchange_imm(uint64_t seq, uint64_t bytes) {
    const uint32_t tag = (uint32_t)(((bytes >> 2) ^ (bytes >> 14) ^ (bytes >> 26)) & 0xfffu);
    uint32_t imm = ((uint32_t)(seq & 0xfffffu) << 12) | tag;
    if (imm == 0xffffffffu) imm ^= 1u;
    return imm;
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

__device__ inline float bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f2bf(float f) {  // round to nearest even (kernels/rope_math.hpp f32_to_bf16_bits)
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// Copy n words into the send slot; with publish, one block: fence + flag (decode-sized payloads, one launch).
template <typename W>
__global__ void k_copy(W *dst, const W *src, uint64_t n, volatile uint64_t *ready, uint64_t seq, int publish) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
    if (publish) {
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
            *ready = seq + 1;
            __threadfence_system();
        }
    }
}
__global__ void k_flag(volatile uint64_t *ready, uint64_t seq) {
    __threadfence_system();
    *ready = seq + 1;
    __threadfence_system();
}
__global__ void k_wait(volatile uint64_t *arrival, volatile uint64_t *poison, int N, int rank, uint64_t need) {
    if (threadIdx.x == 0) {
        for (int p = 0; p < N; ++p) {
            if (p == rank) continue;
            while (arrival[p] < need && *poison == 0) __builtin_amdgcn_s_sleep(1);
        }
        __threadfence_system();
    }
}
__global__ void k_sum_bf16(uint16_t *buf, const uint16_t *recv, uint64_t slot_elems, int N, int rank, uint64_t w,
                           uint64_t n) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        float s = 0.0f;
        for (int r = 0; r < N; ++r) s += bf2f(r == rank ? buf[i] : recv[((uint64_t)r * kWindows + w) * slot_elems + i]);
        buf[i] = f2bf(s);
    }
}
__global__ void k_sum_f32(float *buf, const float *recv, uint64_t slot_elems, int N, int rank, uint64_t w, uint64_t n) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        float s = 0.0f;
        for (int r = 0; r < N; ++r) s += r == rank ? buf[i] : recv[((uint64_t)r * kWindows + w) * slot_elems + i];
        buf[i] = s;
    }
}
__global__ void k_sum_f32_bf16(uint16_t *out, const float *buf, const float *recv, uint64_t slot_elems, int N, int rank,
                               uint64_t w, uint64_t n) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        float s = 0.0f;
        for (int r = 0; r < N; ++r) s += r == rank ? buf[i] : recv[((uint64_t)r * kWindows + w) * slot_elems + i];
        out[i] = f2bf(s);
    }
}
__global__ void k_gather(uint32_t *dst, const uint32_t *src, const uint32_t *recv, uint64_t slot_words, int N, int rank,
                         uint64_t w, uint64_t n) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n * N; i += (uint64_t)gridDim.x * blockDim.x) {
        const int r = (int)(i / n);
        const uint64_t j = i % n;
        dst[i] = r == rank ? src[j] : recv[((uint64_t)r * kWindows + w) * slot_words + j];
    }
}

__global__ void k_merge_cands(const uint8_t *g, int N, int64_t rows, int64_t lm_rows, uint8_t *out) {
    using kernels::LogitCand;
    constexpr int K = kernels::kLogitCands;
    const int64_t row = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const size_t blk = (size_t)rows * (K * sizeof(LogitCand) + 4);
    int at[4] = {0, 0, 0, 0};
    auto *o = reinterpret_cast<LogitCand *>(out) + row * K;
    for (int k = 0; k < K; ++k) {
        int best = -1;
        LogitCand bc{0.0f, 0};
        for (int r = 0; r < N; ++r) {
            if (at[r] >= K) continue;
            LogitCand c = reinterpret_cast<const LogitCand *>(g + r * blk)[row * K + at[r]];
            if (c.id != INT32_MAX) c.id += (int32_t)(r * lm_rows);
            // (value descending, lower id first); -inf fillers carry INT32_MAX and sort last among -inf
            if (best < 0 || c.v > bc.v || (c.v == bc.v && c.id < bc.id)) best = r, bc = c;
        }
        o[k] = bc;
        ++at[best];
    }
    uint32_t nan = 0;
    for (int r = 0; r < N; ++r)
        nan |= reinterpret_cast<const uint32_t *>(g + r * blk + (size_t)rows * K * sizeof(LogitCand))[row];
    reinterpret_cast<uint32_t *>(out + (size_t)rows * K * sizeof(LogitCand))[row] = nan;
}

// ---------------- TCP (handshake + control) ----------------
int tcp_listen(const std::string &ip, int port) {
    int s = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0), one = 1;
    STRIX_CHECK(s >= 0, "TpComm: socket: ", std::strerror(errno));
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET, a.sin_port = htons((uint16_t)port);
    STRIX_CHECK(inet_pton(AF_INET, ip.c_str(), &a.sin_addr) == 1, "TpComm: bad address '", ip, "'");
    if (bind(s, (sockaddr *)&a, sizeof a) || listen(s, 16)) {
        const std::string e = std::strerror(errno);
        close(s);
        STRIX_FAIL("TpComm: listen on ", ip, ":", port, ": ", e);
    }
    return s;
}
int tcp_connect(const std::string &ip, int port, double deadline) {
    while (now_s() < deadline) {
        int s = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0), one = 1;
        sockaddr_in a{};
        a.sin_family = AF_INET, a.sin_port = htons((uint16_t)port);
        inet_pton(AF_INET, ip.c_str(), &a.sin_addr);
        if (connect(s, (sockaddr *)&a, sizeof a) == 0) {
            setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            return s;
        }
        close(s);
        usleep(200000);
    }
    STRIX_FAIL("TpComm: timed out connecting to ", ip, ":", port);
}
void xfer(int fd, void *p, size_t n, bool out, double deadline) {
    auto *b = static_cast<uint8_t *>(p);
    while (n) {
        pollfd pf{fd, (short)(out ? POLLOUT : POLLIN), 0};
        const double left = deadline - now_s();
        const int ms = left > 2e6 ? -1 : (int)std::max(0.0, left * 1000);
        const int pr = poll(&pf, 1, ms);
        if (pr < 0 && errno == EINTR) continue;
        STRIX_CHECK(pr > 0, "TpComm: control socket timed out (", out ? "send" : "recv", ", ", n, " bytes left)");
        const ssize_t r = out ? ::send(fd, b, n, MSG_NOSIGNAL) : ::recv(fd, b, n, 0);
        if (r < 0 && (errno == EINTR || errno == EAGAIN)) continue;
        STRIX_CHECK(r > 0, "TpComm: control socket closed (", r < 0 ? std::strerror(errno) : "peer gone", ")");
        b += r, n -= (size_t)r;
    }
}

int find_gid(const std::string &dev, const std::string &ip) {
    in_addr a;
    inet_pton(AF_INET, ip.c_str(), &a);
    const uint8_t *b = (const uint8_t *)&a;
    char want[64];
    std::snprintf(want, sizeof want, "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x", b[0], b[1], b[2], b[3]);
    for (int i = 0; i < 256; ++i) {
        std::ifstream g("/sys/class/infiniband/" + dev + "/ports/1/gids/" + std::to_string(i));
        std::ifstream t("/sys/class/infiniband/" + dev + "/ports/1/gid_attrs/types/" + std::to_string(i));
        std::string gs, ts;
        if (!std::getline(g, gs)) continue;
        std::getline(t, ts);
        if (gs == want && ts.find("v2") != std::string::npos) return i;
    }
    return -1;
}
int pick_gid(const std::string &dev, const std::string &ip) {
    const int g = find_gid(dev, ip);
    STRIX_CHECK(g >= 0, "TpComm: no RoCE v2 GID for ", ip, " on ", dev);
    return g;
}

struct PairInfo {
    uint32_t magic, version;
    int32_t rank, world;
    uint64_t slot;
    uint8_t gid[16];
    uint32_t rkey, qpn, psn, pad;
    uint64_t region;
};
constexpr uint32_t kMagic = 0x54503243;  // "TP2C"

}  // namespace

std::string tp_pick_device(const std::string &dev, const std::string &ip) {
    if (!dev.empty()) return dev;
    // By driver (mlx5; the USB4 / Thunderbolt RDMA devices are others) and address: device names follow the PCI
    // address and change on a renumber (e.g. rocep195s0).
    std::vector<std::string> seen;
    int ndev = 0;
    ibv_device **dl = ibv_get_device_list(&ndev);
    std::string found;
    for (int i = 0; dl && i < ndev; ++i) {
        const std::string name = ibv_get_device_name(dl[i]);
        char drv[512] = {0};
        const ssize_t n = readlink(("/sys/class/infiniband/" + name + "/device/driver").c_str(), drv, sizeof drv - 1);
        const std::string d = n > 0 ? std::string(drv, (size_t)n) : std::string();
        const std::string base = d.substr(d.find_last_of('/') + 1);
        seen.push_back(name + " (" + base + ")");
        if (found.empty() && base.rfind("mlx5", 0) == 0 && find_gid(name, ip) >= 0) found = name;
    }
    if (dl) ibv_free_device_list(dl);
    std::string all;
    for (const std::string &x : seen) all += (all.empty() ? "" : ", ") + x;
    STRIX_CHECK(!found.empty(), "TpComm: no mlx5 RDMA device with a RoCE v2 GID for ", ip, " (devices: ", all, ")");
    return found;
}

struct TpComm::Impl {
    TpCommConfig cfg;
    int N = 1, rank = 0;
    uint64_t slot = 0;
    uint8_t *send_h = nullptr, *recv_h = nullptr, *send_d = nullptr, *recv_d = nullptr;
    uint64_t *flags_h = nullptr, *flags_d = nullptr;  // [0] ready, [4] poison, [8 + p] arrival from p
    ibv_context *ctx = nullptr;
    ibv_pd *pd = nullptr;
    ibv_mr *smr = nullptr, *rmr = nullptr;
    ibv_cq *scq = nullptr, *rcq = nullptr;
    struct Peer {
        int fd = -1;
        ibv_qp *qp = nullptr;
        PairInfo remote{};
        uint64_t imms = 0, outstanding = 0, since_sig = 0;
        uint64_t checked = 0;  // exchanges from this peer whose size tag matched this rank's (arrival published)
        std::vector<uint16_t> tags = std::vector<uint16_t>(kRing);  // received size tags by seq % kRing
    };
    std::vector<Peer> P;
    std::vector<uint64_t> ring = std::vector<uint64_t>(kRing);
    std::atomic<uint64_t> queued{0}, posted{0};
    std::atomic<bool> stop{false}, poison_out{false};  // poison_out: tell the peers (an immediate of kPoisonImm)
    mutable std::mutex err_mu;
    std::string err;
    std::thread comm;

    volatile uint64_t *ready() const { return flags_h; }
    volatile uint64_t *arrival() const { return flags_h + 8; }
    void set_poison(const std::string &why) {
        {
            std::lock_guard<std::mutex> g(err_mu);
            if (err.empty()) err = why;
        }
        __atomic_store_n(flags_h + 4, (uint64_t)1, __ATOMIC_SEQ_CST);
        poison_out = true;
    }
    bool is_poisoned() const { return __atomic_load_n(flags_h + 4, __ATOMIC_SEQ_CST) != 0; }
    int peer_of(uint32_t qpn) const {
        for (int p = 0; p < N; ++p)
            if (p != rank && P[p].qp->qp_num == qpn) return p;
        return -1;
    }
    void loop();
    void run();  // loop(), then (after a failure) keep telling / idling until stopped
    void tell_peers() {
        for (int p = 0; p < N; ++p) {
            if (p == rank) continue;
            ibv_send_wr wr{}, *bad = nullptr;
            wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM, wr.imm_data = htonl(kPoisonImm), wr.send_flags = IBV_SEND_SIGNALED;
            wr.wr.rdma.remote_addr = P[p].remote.region, wr.wr.rdma.rkey = P[p].remote.rkey;
            (void)ibv_post_send(P[p].qp, &wr, &bad);  // best effort: a peer whose QP is gone can't be told
        }
    }
};

void TpComm::Impl::run() {
    loop();
    bool told = false;
    while (!stop.load(std::memory_order_relaxed)) {
        if (poison_out.load() && !told) told = true, tell_peers();
        usleep(1000);
    }
}

void TpComm::Impl::loop() {
    ibv_wc wc[64];
    uint64_t next = 0;
    double last_progress = now_s(), last_activity = now_s();
    auto reap_send = [&]() -> bool {
        const int n = ibv_poll_cq(scq, 64, wc);
        for (int k = 0; k < n; ++k) {
            if (wc[k].status) {
                set_poison(cat("TpComm: send completion error ", ibv_wc_status_str(wc[k].status)));
                return false;
            }
            const int p = peer_of(wc[k].qp_num);
            if (p >= 0) P[p].outstanding -= wc[k].wr_id;
        }
        return n >= 0;
    };
    while (!stop.load(std::memory_order_relaxed)) {
        bool did = false;
        if (is_poisoned()) return;  // run() tells the peers
        // post exchange `next` once queued by the host and published by the GPU
        if (queued.load(std::memory_order_acquire) > next && *ready() >= next + 1) {
            const uint64_t bytes = ring[next % kRing], w = next % kWindows;
            const uint64_t c = std::min<uint64_t>(cfg.chunk ? cfg.chunk : bytes, 32ull << 20);
            for (int p = 0; p < N; ++p) {
                if (p == rank) continue;
                Peer &pr = P[p];
                uint64_t off = 0;
                do {
                    const uint64_t len = std::min(c, bytes - off);
                    const bool last = off + len >= bytes;
                    while (pr.outstanding > (uint64_t)kSqDepth - 64)
                        if (!reap_send()) return;
                    ibv_sge sg{(uint64_t)(uintptr_t)(send_h + w * slot + off), (uint32_t)len, smr->lkey};
                    ibv_send_wr wr{}, *bad = nullptr;
                    wr.sg_list = len ? &sg : nullptr, wr.num_sge = len ? 1 : 0;
                    wr.opcode = last ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE;
                    wr.imm_data = htonl(exchange_imm(next, bytes));
                    wr.wr.rdma.remote_addr = pr.remote.region + ((uint64_t)rank * kWindows + w) * slot + off;
                    wr.wr.rdma.rkey = pr.remote.rkey;
                    ++pr.since_sig, ++pr.outstanding;
                    if (last || pr.since_sig >= 64) wr.send_flags |= IBV_SEND_SIGNALED, wr.wr_id = pr.since_sig, pr.since_sig = 0;
                    if (const int rc = ibv_post_send(pr.qp, &wr, &bad)) {
                        set_poison(cat("TpComm: ibv_post_send to rank ", p, ": ", std::strerror(rc)));
                        return;
                    }
                    off += len;
                } while (off < bytes);
            }
            posted.store(++next, std::memory_order_release);
            did = true;
        }
        const int n = ibv_poll_cq(rcq, 64, wc);
        for (int k = 0; k < n; ++k) {
            if (wc[k].status) {
                set_poison(cat("TpComm: receive completion error ", ibv_wc_status_str(wc[k].status)));
                return;
            }
            const int p = peer_of(wc[k].qp_num);
            if (p < 0) {
                set_poison(cat("TpComm: completion on unknown QP ", wc[k].qp_num));
                return;
            }
            Peer &pr = P[p];
            if (ntohl(wc[k].imm_data) == kPoisonImm) {
                set_poison(cat("TpComm: rank ", p, " failed and poisoned the communicator"));
                return;
            }
            const uint32_t imm = ntohl(wc[k].imm_data);
            if ((imm >> 12) != (uint32_t)(pr.imms & 0xfffffu)) {
                set_poison(cat("TpComm: immediate for exchange ", imm >> 12, " (mod 2^20) from rank ", p, ", expected ",
                               pr.imms & 0xfffffu, " (the ranks ran different calls)"));
                return;
            }
            pr.tags[pr.imms % kRing] = (uint16_t)(imm & 0xfffu);
            ++pr.imms;
            ibv_recv_wr rw{}, *bad = nullptr;
            ibv_post_recv(pr.qp, &rw, &bad);
            did = true;
        }
        // Arrivals are published once this rank has queued the same exchange and its size tag agrees.
        const uint64_t q = queued.load(std::memory_order_acquire);
        for (int p = 0; p < N; ++p) {
            if (p == rank) continue;
            Peer &pr = P[p];
            const uint64_t c0 = pr.checked;
            while (pr.checked < pr.imms && pr.checked < q) {
                const uint64_t s = pr.checked, bytes = ring[s % kRing];
                const uint32_t want = exchange_imm(s, bytes) & 0xfffu;
                if (pr.tags[s % kRing] != want) {
                    set_poison(cat("TpComm: exchange ", s, ": rank ", p, "'s payload size tag ", pr.tags[s % kRing],
                                   " != this rank's ", want, " (", bytes, " bytes) - the ranks ran different shapes"));
                    return;
                }
                ++pr.checked;
            }
            if (pr.checked != c0) {
                __atomic_store_n(flags_h + 8 + p, pr.checked, __ATOMIC_SEQ_CST);
                did = true;
            }
        }
        if (!reap_send()) return;
        const double t = now_s();
        if (did) {
            last_progress = last_activity = t;
            continue;
        }
        // waiting on a peer for an exchange we posted: the watchdog
        bool behind = false;
        for (int p = 0; p < N; ++p)
            if (p != rank && P[p].checked < next) behind = true;
        if (!behind) last_progress = t;
        else if (t - last_progress > cfg.exchange_timeout_s) {
            set_poison(cat("TpComm: no data from a peer for ", cfg.exchange_timeout_s, " s at exchange ", next));
            return;
        }
        if (t - last_activity > 0.005) usleep(20);  // idle: back off (a forward's exchanges come ~0.1 ms apart)
    }
}

TpComm::TpComm(const TpCommConfig &cfg) : impl_(std::make_unique<Impl>()) {
    Impl &I = *impl_;
    I.cfg = cfg;
    I.N = world_ = (int)cfg.peers.size();
    I.rank = rank_ = cfg.rank;
    STRIX_CHECK(world_ >= 2 && world_ <= 4 && rank_ >= 0 && rank_ < world_, "TpComm: rank ", rank_, " of ", world_,
                " (2..4 peers)");
    STRIX_CHECK(cfg.max_bytes > 0 && cfg.max_bytes % 4 == 0, "TpComm: max_bytes = ", cfg.max_bytes);
    max_bytes_ = cfg.max_bytes;
    I.slot = (cfg.max_bytes + 4095) & ~(uint64_t)4095;
    const double deadline = now_s() + cfg.connect_timeout_s;
    const uint64_t send_bytes = I.slot * kWindows, recv_bytes = I.slot * kWindows * (uint64_t)world_;
    STRIX_HIP_CHECK(hipHostMalloc((void **)&I.flags_h, 4096, hipHostMallocCoherent | hipHostMallocMapped), "TpComm flags");
    std::memset(I.flags_h, 0, 4096);
    STRIX_HIP_CHECK(hipHostGetDevicePointer((void **)&I.flags_d, I.flags_h, 0), "TpComm flags");
    STRIX_HIP_CHECK(hipHostMalloc((void **)&I.send_h, send_bytes, hipHostMallocCoherent | hipHostMallocMapped),
                    "TpComm send windows (", send_bytes, " bytes)");
    STRIX_HIP_CHECK(hipHostMalloc((void **)&I.recv_h, recv_bytes, hipHostMallocCoherent | hipHostMallocMapped),
                    "TpComm receive windows (", recv_bytes, " bytes)");
    STRIX_HIP_CHECK(hipHostGetDevicePointer((void **)&I.send_d, I.send_h, 0), "TpComm send windows");
    STRIX_HIP_CHECK(hipHostGetDevicePointer((void **)&I.recv_d, I.recv_h, 0), "TpComm receive windows");

    dev_ = I.cfg.dev = tp_pick_device(cfg.dev, cfg.peers[(size_t)rank_]);
    int ndev = 0;
    ibv_device **dl = ibv_get_device_list(&ndev);
    for (int i = 0; dl && i < ndev; ++i)
        if (dev_ == ibv_get_device_name(dl[i])) I.ctx = ibv_open_device(dl[i]);
    if (dl) ibv_free_device_list(dl);
    STRIX_CHECK(I.ctx, "TpComm: RDMA device ", dev_, " not found (", ndev, " devices; stock mlx5 provider?)");
    ibv_port_attr pa;
    STRIX_CHECK(ibv_query_port(I.ctx, 1, &pa) == 0, "TpComm: ibv_query_port");
    const int gidx = pick_gid(dev_, cfg.peers[(size_t)rank_]);
    ibv_gid my_gid;
    STRIX_CHECK(ibv_query_gid(I.ctx, 1, gidx, &my_gid) == 0, "TpComm: ibv_query_gid ", gidx);
    I.pd = ibv_alloc_pd(I.ctx);
    STRIX_CHECK(I.pd, "TpComm: ibv_alloc_pd");
    const int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    I.smr = ibv_reg_mr(I.pd, I.send_h, send_bytes, acc);
    I.rmr = ibv_reg_mr(I.pd, I.recv_h, recv_bytes, acc);
    STRIX_CHECK(I.smr && I.rmr, "TpComm: ibv_reg_mr (", send_bytes, " + ", recv_bytes, " bytes): ", std::strerror(errno),
                " - run with LimitMEMLOCK=infinity");
    I.scq = ibv_create_cq(I.ctx, kSqDepth * world_, nullptr, nullptr, 0);
    I.rcq = ibv_create_cq(I.ctx, kRqDepth * world_, nullptr, nullptr, 0);
    STRIX_CHECK(I.scq && I.rcq, "TpComm: ibv_create_cq");
    I.P.resize((size_t)world_);
    for (int p = 0; p < world_; ++p) {
        if (p == rank_) continue;
        ibv_qp_init_attr ia{};
        ia.send_cq = I.scq, ia.recv_cq = I.rcq, ia.qp_type = IBV_QPT_RC;
        ia.cap.max_send_wr = kSqDepth, ia.cap.max_recv_wr = kRqDepth, ia.cap.max_send_sge = 1, ia.cap.max_recv_sge = 1;
        ibv_qp *qp = ibv_create_qp(I.pd, &ia);
        STRIX_CHECK(qp, "TpComm: ibv_create_qp: ", std::strerror(errno));
        ibv_qp_attr at{};
        at.qp_state = IBV_QPS_INIT, at.pkey_index = 0, at.port_num = 1, at.qp_access_flags = acc;
        STRIX_CHECK(ibv_modify_qp(qp, &at, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
                    "TpComm: QP INIT");
        I.P[(size_t)p].qp = qp;
    }
    // TCP mesh on the rail-0 addresses: connect to lower ranks, accept higher ones.
    const int ls = tcp_listen(cfg.peers[(size_t)rank_], cfg.port + rank_);
    for (int p = 0; p < rank_; ++p) {
        I.P[(size_t)p].fd = tcp_connect(cfg.peers[(size_t)p], cfg.port + p, deadline);
        int32_t me = rank_;
        xfer(I.P[(size_t)p].fd, &me, 4, true, deadline);
    }
    for (int k = rank_ + 1; k < world_; ++k) {
        pollfd pf{ls, POLLIN, 0};
        STRIX_CHECK(poll(&pf, 1, (int)std::max(0.0, (deadline - now_s()) * 1000)) > 0,
                    "TpComm: timed out waiting for the higher ranks to connect");
        int fd = accept4(ls, nullptr, nullptr, SOCK_CLOEXEC), one = 1;
        STRIX_CHECK(fd >= 0, "TpComm: accept: ", std::strerror(errno));
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        int32_t r = -1;
        xfer(fd, &r, 4, false, deadline);
        STRIX_CHECK(r > rank_ && r < world_ && I.P[(size_t)r].fd < 0, "TpComm: bad peer rank ", r);
        I.P[(size_t)r].fd = fd;
    }
    close(ls);
    srand48((long)getpid() ^ (long)(now_s() * 1e6));
    for (int p = 0; p < world_; ++p) {
        if (p == rank_) continue;
        Impl::Peer &pr = I.P[(size_t)p];
        PairInfo mine{};
        mine.magic = kMagic, mine.version = 1, mine.rank = rank_, mine.world = world_, mine.slot = I.slot;
        std::memcpy(mine.gid, my_gid.raw, 16);
        mine.rkey = I.rmr->rkey, mine.qpn = pr.qp->qp_num, mine.psn = (uint32_t)(lrand48() & 0xffffff);
        mine.region = (uint64_t)(uintptr_t)I.recv_h;
        xfer(pr.fd, &mine, sizeof mine, true, deadline);
        xfer(pr.fd, &pr.remote, sizeof mine, false, deadline);
        STRIX_CHECK(pr.remote.magic == kMagic && pr.remote.version == 1 && pr.remote.rank == p &&
                        pr.remote.world == world_ && pr.remote.slot == I.slot,
                    "TpComm: peer ", p, " answered rank ", pr.remote.rank, " of ", pr.remote.world, ", slot ",
                    pr.remote.slot, " (want rank ", p, " of ", world_, ", slot ", I.slot, ")");
        ibv_qp_attr at{};
        at.qp_state = IBV_QPS_RTR, at.path_mtu = std::min(pa.active_mtu, IBV_MTU_4096);
        at.dest_qp_num = pr.remote.qpn, at.rq_psn = pr.remote.psn;
        at.max_dest_rd_atomic = 1, at.min_rnr_timer = 12;
        at.ah_attr.is_global = 1, at.ah_attr.port_num = 1;
        std::memcpy(at.ah_attr.grh.dgid.raw, pr.remote.gid, 16);
        at.ah_attr.grh.sgid_index = (uint8_t)gidx, at.ah_attr.grh.hop_limit = 64;
        STRIX_CHECK(ibv_modify_qp(pr.qp, &at, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                                                  IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0,
                    "TpComm: QP RTR (peer ", p, "): ", std::strerror(errno));
        at.qp_state = IBV_QPS_RTS, at.timeout = 14, at.retry_cnt = 7, at.rnr_retry = 7, at.sq_psn = mine.psn;
        at.max_rd_atomic = 1;
        STRIX_CHECK(ibv_modify_qp(pr.qp, &at, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                                                  IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) == 0,
                    "TpComm: QP RTS (peer ", p, "): ", std::strerror(errno));
        for (int k = 0; k < kRqDepth; ++k) {
            ibv_recv_wr rw{}, *bad = nullptr;
            STRIX_CHECK(ibv_post_recv(pr.qp, &rw, &bad) == 0, "TpComm: ibv_post_recv");
        }
    }
    barrier();  // every QP is ready to receive before anyone writes
    I.comm = std::thread([this] { impl_->run(); });
}

TpComm::~TpComm() {
    Impl &I = *impl_;
    I.stop = true;
    if (I.comm.joinable()) I.comm.join();
    for (Impl::Peer &p : I.P) {
        if (p.qp) ibv_destroy_qp(p.qp);
        if (p.fd >= 0) close(p.fd);
    }
    if (I.scq) ibv_destroy_cq(I.scq);
    if (I.rcq) ibv_destroy_cq(I.rcq);
    if (I.smr) ibv_dereg_mr(I.smr);
    if (I.rmr) ibv_dereg_mr(I.rmr);
    if (I.pd) ibv_dealloc_pd(I.pd);
    if (I.ctx) ibv_close_device(I.ctx);
    if (I.send_h) (void)hipHostFree(I.send_h);
    if (I.recv_h) (void)hipHostFree(I.recv_h);
    if (I.flags_h) (void)hipHostFree(I.flags_h);
}

bool TpComm::poisoned() const { return impl_->is_poisoned(); }

void TpComm::check() const {
    if (!impl_->is_poisoned()) return;
    std::lock_guard<std::mutex> g(impl_->err_mu);
    STRIX_FAIL("TpComm (rank ", rank_, " of ", world_, ") poisoned: ", impl_->err.empty() ? "poison()" : impl_->err);
}

void TpComm::poison(const std::string &why) { impl_->set_poison(why); }

void TpComm::enqueue(const void *src, size_t bytes, hipStream_t stream) {
    Impl &I = *impl_;
    STRIX_CHECK(bytes > 0 && bytes <= max_bytes_ && bytes % 4 == 0, "TpComm: exchange of ", bytes, " bytes (max ",
                max_bytes_, ", a multiple of 4)");
    while (seq_ - I.posted.load(std::memory_order_acquire) >= kRing / 2 && !I.is_poisoned())
        usleep(50);  // never in practice: a forward queues ~100 exchanges, then waits for its logits
    const uint64_t s = seq_, w = s % kWindows;
    I.ring[s % kRing] = bytes;
    I.queued.store(s + 1, std::memory_order_release);
    uint8_t *dst = I.send_d + w * I.slot;
    if (bytes <= (256u << 10)) {
        if (bytes % 16 == 0)
            k_copy<uint4><<<1, 1024, 0, stream>>>((uint4 *)dst, (const uint4 *)src, bytes / 16, I.flags_d, s, 1);
        else
            k_copy<uint32_t><<<1, 1024, 0, stream>>>((uint32_t *)dst, (const uint32_t *)src, bytes / 4, I.flags_d, s, 1);
    } else {
        if (bytes % 16 == 0)
            k_copy<uint4><<<512, 256, 0, stream>>>((uint4 *)dst, (const uint4 *)src, bytes / 16, I.flags_d, s, 0);
        else
            k_copy<uint32_t><<<512, 256, 0, stream>>>((uint32_t *)dst, (const uint32_t *)src, bytes / 4, I.flags_d, s, 0);
        k_flag<<<1, 1, 0, stream>>>(I.flags_d, s);
    }
    STRIX_HIP_CHECK_LAUNCH("TpComm publish");
    k_wait<<<1, 32, 0, stream>>>(I.flags_d + 8, I.flags_d + 4, world_, rank_, s + 1);
    STRIX_HIP_CHECK_LAUNCH("TpComm wait");
    ++seq_;
}

void TpComm::allreduce(void *buf, int64_t elems, kernels::Act act, hipStream_t stream) {
    Impl &I = *impl_;
    const size_t e = act == kernels::Act::F32 ? 4 : 2;
    const size_t bytes = (size_t)elems * e;
    const uint64_t w = seq_ % kWindows;
    enqueue(buf, bytes, stream);
    const unsigned grid = (unsigned)std::min<int64_t>(1024, (elems + 255) / 256);
    if (act == kernels::Act::F32)
        k_sum_f32<<<grid, 256, 0, stream>>>((float *)buf, (const float *)I.recv_d, I.slot / 4, world_, rank_, w, (uint64_t)elems);
    else
        k_sum_bf16<<<grid, 256, 0, stream>>>((uint16_t *)buf, (const uint16_t *)I.recv_d, I.slot / 2, world_, rank_, w,
                                             (uint64_t)elems);
    STRIX_HIP_CHECK_LAUNCH("TpComm sum");
}

void TpComm::allreduce_f32_to_bf16(const float *buf, int64_t elems, void *out, hipStream_t stream) {
    Impl &I = *impl_;
    const uint64_t w = seq_ % kWindows;
    enqueue(buf, (size_t)elems * 4, stream);
    const unsigned grid = (unsigned)std::min<int64_t>(1024, (elems + 255) / 256);
    k_sum_f32_bf16<<<grid, 256, 0, stream>>>((uint16_t *)out, buf, (const float *)I.recv_d, I.slot / 4, world_, rank_, w,
                                             (uint64_t)elems);
    STRIX_HIP_CHECK_LAUNCH("TpComm sum (FP32 -> BF16)");
}

void TpComm::allgather(const void *src, size_t bytes, void *dst, hipStream_t stream) {
    Impl &I = *impl_;
    const uint64_t w = seq_ % kWindows;
    enqueue(src, bytes, stream);
    const uint64_t n = bytes / 4;
    const unsigned grid = (unsigned)std::min<uint64_t>(1024, (n * world_ + 255) / 256);
    k_gather<<<grid, 256, 0, stream>>>((uint32_t *)dst, (const uint32_t *)src, (const uint32_t *)I.recv_d, I.slot / 4,
                                       world_, rank_, w, n);
    STRIX_HIP_CHECK_LAUNCH("TpComm gather");
}

void TpComm::send(int peer, const void *p, size_t n) {
    STRIX_CHECK(peer >= 0 && peer < world_ && peer != rank_, "TpComm::send to ", peer);
    xfer(impl_->P[(size_t)peer].fd, const_cast<void *>(p), n, true, now_s() + 1e9);
}

void TpComm::recv(int peer, void *p, size_t n, double timeout_s) {
    STRIX_CHECK(peer >= 0 && peer < world_ && peer != rank_, "TpComm::recv from ", peer);
    xfer(impl_->P[(size_t)peer].fd, p, n, false, now_s() + timeout_s);
}

void TpComm::barrier() {
    for (int p = 0; p < world_; ++p) {
        if (p == rank_) continue;
        uint32_t b = 0xba77;
        send(p, &b, 4);
        recv(p, &b, 4, impl_->cfg.connect_timeout_s);
        STRIX_CHECK(b == 0xba77, "TpComm::barrier: rank ", p, " sent ", b, " (the control stream is out of step)");
    }
}

size_t tp_candidate_block_bytes(int64_t rows) {
    return (size_t)rows * (kernels::kLogitCands * sizeof(kernels::LogitCand) + 4);
}

void tp_merge_candidates(const void *gathered, int world, int64_t rows, int64_t lm_rows, void *out, hipStream_t stream) {
    STRIX_CHECK(world >= 1 && world <= 4 && rows >= 1, "tp_merge_candidates: world ", world, ", rows ", rows);
    k_merge_cands<<<1, 64, 0, stream>>>(static_cast<const uint8_t *>(gathered), world, rows, lm_rows,
                                         static_cast<uint8_t *>(out));
    STRIX_HIP_CHECK_LAUNCH("tp_merge_candidates");
}

}  // namespace strix
