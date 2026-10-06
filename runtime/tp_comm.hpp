#pragma once

// The tensor-parallel communicator (world size N in {2, 4}) over RDMA (RoCE v2, ConnectX-6 Dx,
// stock rdma-core mlx5 provider) - the exchange tools/tp_exchange_bench measures, as a library.
//
// Every exchange (an all-reduce or an all-gather) is queued on a HIP stream without any host wait:
//   publish   the rank's payload is copied into send slot seq % 3 (host-coherent pinned memory registered with the NIC),
//             then a flag kernel publishes ready = seq + 1 (__threadfence_system first)
//   comm      a host thread sees ready, posts RDMA WRITE(s) of the payload into window (rank, seq % 3) of every peer
//             (chunks <= 4 MiB, the last WRITE carries the immediate), polls the receive CQ and publishes
//             arrival[p] = exchanges fully arrived from peer p
//   wait      one wave spins until arrival[p] >= seq + 1 for every peer (or the communicator is poisoned)
//   combine   all-reduce: the N partials summed in FP32 in rank order 0..N-1, rounded once (BF16: RNE) - bit-identical
//             on every rank; all-gather: the N payloads copied out in rank order.
// Windows rotate over 3 (gufo's header-free scheme): a window is written again 3 exchanges later, which the exchange's
// own dependencies make safe (a peer publishes seq + 3 only after its wait for my seq + 2, which I publish after my
// combine of seq has read the window). Sequence numbers are global and identical on every rank as long as every rank
// runs the same calls (the executors mirror rank 0: runtime/tp_mirror.hpp).
//
// Failure: a WC error, a peer silent past exchange_timeout_s, or poison() poisons the communicator: every wait kernel
// exits at once (the sums are then garbage) and check() throws - callers check after each forward.
//
// The control channel: the TCP mesh set up for the RDMA handshake stays open (one socket per peer, on the RDMA
// addresses, TCP_NODELAY); send() / recv() carry the mirrored calls (tp_mirror) and replies. Rank r listens on
// port + r, so ranks may share a host (the one-node two-process loopback: two QPs on one port).
//
// Size check: the immediate of an exchange's last WRITE carries its sequence number (low 20 bits) and a 12-bit
// tag of its size; a peer's exchange whose size disagrees with this rank's same exchange poisons the communicator
// (a mirror bug - e.g. a verify of another T - would otherwise sum garbage silently). Arrival is published only after
// the check.
//
// Needs LimitMEMLOCK=infinity (the windows are 9 x max_bytes of pinned memory at N = 2).

#include "kernels/norm.hpp"  // Act

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strix {

struct TpCommConfig {
    int rank = 0;
    std::vector<std::string> peers;  // RDMA-interface IPv4 address of every rank, in rank order (world = peers.size())
    std::string dev;                 // RDMA device; empty: the mlx5 device with a RoCE v2 GID for this rank's address
    int port = 18600;                // TCP port of the handshake / control mesh (every rank listens on its own address)
    size_t max_bytes = 0;            // the largest payload of one exchange per rank
    uint64_t chunk = 4ull << 20;     // WRITE size for large payloads (plan v2: never one WRITE above 32 MiB)
    double connect_timeout_s = 600;  // waiting for the peers to come up
    double exchange_timeout_s = 900; // a peer this late on an exchange poisons the communicator
};

// The control channel as tp_mirror uses it (TpDriver / tp_executor): TpComm's TCP mesh, or an in-process loopback
// (TpLoopback, runtime/tp_mirror.hpp - the mirror-shadow test).
class TpControl {
public:
    virtual ~TpControl() = default;
    virtual int rank() const = 0;
    virtual int world() const = 0;
    // blocking, whole buffers, to / from rank `peer` (!= rank())
    virtual void send(int peer, const void *p, size_t n) = 0;
    virtual void recv(int peer, void *p, size_t n, double timeout_s = 1e9) = 0;
    virtual void check() const = 0;                   // throws (with the first error) once poisoned
    virtual void poison(const std::string &why) = 0;  // stops every wait now and later
};

// The RDMA device for a rank's address: dev if given, else the mlx5 device with a RoCE v2 GID for ip.
std::string tp_pick_device(const std::string &dev, const std::string &ip);

class TpComm : public TpControl {
public:
    explicit TpComm(const TpCommConfig &cfg);
    ~TpComm();
    TpComm(const TpComm &) = delete;
    TpComm &operator=(const TpComm &) = delete;

    int rank() const override { return rank_; }
    int world() const override { return world_; }
    size_t max_bytes() const { return max_bytes_; }
    const std::string &device() const { return dev_; }

    // In place on stream: buf [elems] of act (BF16 or F32) := the sum over the ranks of every rank's buf, in rank order.
    void allreduce(void *buf, int64_t elems, kernels::Act act, hipStream_t stream);
    // FP32 partials buf [elems] -> out [elems] BF16 := their sum over the ranks in rank order, rounded once (RNE).
    void allreduce_f32_to_bf16(const float *buf, int64_t elems, void *out, hipStream_t stream);
    // On stream: dst [world][bytes] (device) := every rank's src [bytes] (device), in rank order.
    void allgather(const void *src, size_t bytes, void *dst, hipStream_t stream);

    uint64_t exchanges() const { return seq_; }  // queued so far
    bool poisoned() const;
    void check() const override;
    void poison(const std::string &why) override;

    // Control channel (blocking, whole buffers): to / from rank `peer` (!= rank()).
    void send(int peer, const void *p, size_t n) override;
    void recv(int peer, void *p, size_t n, double timeout_s = 1e9) override;
    void barrier();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int rank_ = 0, world_ = 1;
    std::string dev_;
    size_t max_bytes_ = 0;
    uint64_t seq_ = 0;
    void enqueue(const void *src, size_t bytes, hipStream_t stream);  // publish + wait for exchange seq_
};

// The LM head's candidate merge (vocabulary split by rank): gathered [N][rows * kLogitCands LogitCand, then rows NaN
// words] (rank r's ids local to its rows [r * lm_rows, (r + 1) * lm_rows)) -> out (the same layout as one rank's,
// global ids): per row the best kLogitCands by (value descending, lower id first) - logits_topk's order on the whole row
// - and the NaN words OR-ed.
void tp_merge_candidates(const void *gathered, int world, int64_t rows, int64_t lm_rows, void *out, hipStream_t stream);
size_t tp_candidate_block_bytes(int64_t rows);

}  // namespace strix
