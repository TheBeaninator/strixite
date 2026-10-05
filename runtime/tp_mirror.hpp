#pragma once

// strixite-tp2: rank 0 drives, ranks 1..N-1 execute (plan v1 "Control", v3 "rank 0 + N-1 executors").
//
// TpDriver (rank 0) has the state-changing calls of Qwen4ExpSession that serving and the benches use; each one is sent
// to every executor over the communicator's control channel (runtime/tp_comm.hpp) BEFORE rank 0 makes the same call,
// so every rank runs the same sequence of forwards and therefore of exchanges (whose sequence numbers must agree).
// tp_executor() is the loop on the other ranks. With no communicator (world 1) the driver is a pass-through, so tools
// run the whole model and a TP group through one code path.
//
// MTP is off under TP (ST-2): drafts on rank 0 only come with ST-3.
//
// Replicated-state check (ST-2's hard gate): with set_hash_check(true) every forward hashes, on every rank, the
// activations that must be identical across ranks - the 4 residual streams X after the embedding ("embed_streams"),
// after the PLE layer ("L<i>.ple_out") and after each layer ("L<i>.out"), the reduced sublayer outputs ("L<i>.mixer",
// "L<i>.moe") and the final mix ("final_mixed") - and rank 0 compares the executors' hashes with its own after the
// forward. It runs the forward with the session's probe (a stream sync per probe): for checking, not for timing.

#include "runtime/qwen4exp.hpp"
#include "runtime/tp_comm.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace strix {

// The session's exchanges through the communicator: kinds 0 / 1 all-reduced in place (the session's activation
// dtype), kind 2 the candidates gathered and merged (tp_merge_candidates). The session and comm must outlive it.
void tp_attach(Qwen4ExpSession &ses, const Qwen4ExpModel &model, TpComm &comm);

class TpDriver {
public:
    TpDriver(Qwen4ExpSession &ses, TpComm *comm);  // comm: null for world 1
    int world() const { return comm_ ? comm_->world() : 1; }

    void reset();
    // forward on every rank. full: whole logits rows [n_logits, vocab] FP32 (every rank's share gathered over the
    // control channel; slow - quality runs only); else what Qwen4ExpSession::forward returns on rank 0 (empty with
    // candidates; rank 0's share of the rows otherwise).
    std::vector<float> forward(const std::vector<int32_t> &ids, int64_t n_logits, bool full = false);
    void want_candidates(int64_t n_valid);
    const Qwen4ExpSession::Candidates &candidates() const { return ses_.candidates(); }
    void set_lookahead(const std::vector<int32_t> &next_ids);
    void prefetch_ple(const std::vector<int32_t> &ids, int64_t first);
    int make_snapshot();  // a snapshot slot on every rank
    void save(int id);
    void restore(int id);
    void set_hash_check(bool on) { hash_ = on; }
    void finish();  // the executors leave their loop (also on destruction)
    ~TpDriver();

    struct HashStats {
        int64_t forwards = 0, compared = 0, mismatched = 0, x_compared = 0, x_mismatched = 0;
        std::string first_mismatch;  // "forward f, <name>, rank r"
    };
    const HashStats &hash_stats() const { return hs_; }

private:
    Qwen4ExpSession &ses_;
    TpComm *comm_;
    bool hash_ = false, finished_ = false;
    uint64_t seq_ = 0;
    std::map<int, Qwen4ExpSnapshot> snaps_;
    HashStats hs_;
    void send_all(uint32_t op, int64_t a = 0, int64_t b = 0, const std::vector<int32_t> *ids = nullptr);
};

// Ranks 1..N-1: run the calls rank 0 sends until finish() (or a failure: the communicator is poisoned so rank 0's
// waits end, and the error rethrown).
void tp_executor(Qwen4ExpSession &ses, TpComm &comm);

}  // namespace strix
