#pragma once

// Tensor parallelism: rank 0 drives, ranks 1..N-1 execute.
//
// TpDriver (rank 0) has the state-changing calls of Qwen4ExpSession that serving and the benches use; each one is sent
// to every executor over the control channel (TpControl: the communicator's TCP mesh, runtime/tp_comm.hpp, or an
// in-process loopback) BEFORE rank 0 makes the same call, so every rank runs the same sequence of forwards and
// therefore of exchanges (whose sequence numbers must agree). tp_executor() is the loop on the other ranks. With no
// control channel (world 1) the driver is a pass-through, so tools run the whole model and a TP group through one code
// path.
//
// MTP: the draft head lives on rank 0 only, whole (Qwen4ExpModel::mtp_dims);
// drafts (forward_mtp_top2) and the catch-up at the end of each forward run there and need no exchange, so they are
// not mirrored. Executors mirror only the trunk's state machine: forward / forward_verify (the candidate n_valid
// folded into the message - a forgotten want_candidates can't leave a rank one exchange short), keep_verify,
// keep_verify_prefix(rows), drop_verify. Messages are fire-and-forget except the hash / stats replies.
//
// Replicated-state checks: with set_hash_check(true) every forward / verify hashes, on every rank, the activations
// that must be identical across ranks - the 4 residual streams X after the embedding ("embed_streams"), after the PLE
// layer ("L<i>.ple_out") and after each layer ("L<i>.out"), the reduced sublayer outputs ("L<i>.mixer", "L<i>.moe"), the
// final mix ("final_mixed") and the merged candidates - and rank 0 compares the executors' hashes with its own after
// the forward. It runs the forward with the session's probe (a stream sync per probe): for checking, not for timing.
// set_state_check(true): after every keep / drop the replicated state (Qwen4ExpSession::state_hash: position, PLE
// state, n-gram history, indexer tails and newest block keys) is compared the same way. set_shadow(true) (the
// one-process shadow test, where the executor is the whole model too): also the returned logits.

#include "runtime/qwen4exp.hpp"
#include "runtime/tp_comm.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strix {

// The session's exchanges through the communicator: kinds 0 / 1 all-reduced in place (the session's activation
// dtype), kind 2 the candidates gathered and merged (tp_merge_candidates). The session and comm must outlive it.
void tp_attach(Qwen4ExpSession &ses, const Qwen4ExpModel &model, TpComm &comm);

// An in-process control channel between ranks 0 and 1 of a world of 2 (byte queues): the mirror-shadow test runs
// TpDriver and tp_executor in one process over it.
class TpLoopback {
public:
    TpLoopback();
    ~TpLoopback();
    TpControl &end(int rank);  // 0 or 1

private:
    struct Shared;
    struct End;
    std::shared_ptr<Shared> sh_;
    std::unique_ptr<End> ends_[2];
};

class TpDriver {
public:
    TpDriver(Qwen4ExpSession &ses, TpControl *ctl);  // ctl: null for world 1
    int world() const { return ctl_ ? ctl_->world() : 1; }
    Qwen4ExpSession &session() { return ses_; }

    void reset();
    // forward on every rank. full: whole logits rows [n_logits, vocab] FP32 (every rank's share gathered over the
    // control channel; slow - quality runs only); else what Qwen4ExpSession::forward returns on rank 0 (empty with
    // candidates; rank 0's share of the rows otherwise).
    std::vector<float> forward(const std::vector<int32_t> &ids, int64_t n_logits, bool full = false);
    // For the next forward / forward_verify only, on every rank (sent inside that forward's message).
    void want_candidates(int64_t n_valid);
    const Qwen4ExpSession::Candidates &candidates() const { return ses_.candidates(); }
    // Verifying drafts (Qwen4ExpSession::forward_verify and its three ways out), mirrored.
    std::vector<float> forward_verify(const std::vector<int32_t> &ids, int64_t n_logits, bool full = false);
    void keep_verify();
    void keep_verify_prefix(int64_t rows);
    void drop_verify();
    // The MTP head (rank 0 only). With the split draft head (Qwen4ExpSession::set_draft_split) each draft is
    // mirrored as kDraftHead first: the executors score their rows of the draft vocabulary in the same two exchanges.
    bool has_mtp() const { return ses_.has_mtp(); }
    Qwen4ExpSession::MtpTop2 forward_mtp_top2(int32_t token_id, int64_t step);
    void set_mtp_vocab(int64_t n) { ses_.set_mtp_vocab(n); }
    void set_lookahead(const std::vector<int32_t> &next_ids);
    void prefetch_ple(const std::vector<int32_t> &ids, int64_t first);
    int make_snapshot();  // a snapshot slot on every rank
    void save(int id);
    void restore(int id);
    bool can_restore(int id) const { return ses_.can_restore(snaps_.at(id)); }
    const Qwen4ExpSnapshot &snapshot(int id) const { return snaps_.at(id); }
    void set_hash_check(bool on) { hash_ = on; }
    void set_state_check(bool on) { state_check_ = on; }
    void set_shadow(bool on) { shadow_ = on; }
    // A local probe on rank 0's forwards / verifies too (alongside the hash check's), e.g. per-row hashes. Not mirrored.
    void set_probe(Qwen4ExpProbe p) { probe_ = std::move(p); }
    // Negative control for the shadow test: the next keep_verify_prefix(rows >= 2) tells the executors rows - 1 - a
    // planted mirror bug the state check must catch. Tests only.
    void debug_plant_prefix_bug() { plant_ = true; }
    // Every executor's PLE stats (kStats), in rank order 1..N-1.
    std::vector<Qwen4ExpSession::PleStats> executor_stats();
    void finish();  // the executors leave their loop (also on destruction)
    ~TpDriver();

    struct HashStats {
        int64_t forwards = 0, compared = 0, mismatched = 0, x_compared = 0, x_mismatched = 0;
        int64_t verify_forwards = 0, verify_mismatched = 0;  // the forward_verify share of the above
        int64_t state_checks = 0, state_mismatched = 0;
        std::string first_mismatch, first_state_mismatch;  // "forward f, <name>, rank r"
    };
    const HashStats &hash_stats() const { return hs_; }
    void clear_hash_stats() { hs_ = HashStats{}; }
    // Calls mirrored so far by kind (forward, forward_verify, keep_verify, keep_verify_prefix, drop_verify, save,
    // restore, reset, prefetch_ple, ...).
    const std::map<std::string, int64_t> &ops() const { return ops_; }

private:
    Qwen4ExpSession &ses_;
    TpControl *ctl_;
    bool hash_ = false, state_check_ = false, shadow_ = false, finished_ = false, plant_ = false;
    Qwen4ExpProbe probe_;
    int64_t cand_ = 0;  // want_candidates' n_valid for the next forward
    uint64_t seq_ = 0;
    std::map<int, Qwen4ExpSnapshot> snaps_;
    std::map<std::string, int64_t> ops_;
    HashStats hs_;
    void send_all(uint32_t op, int64_t a = 0, int64_t b = 0, int64_t c = 0, const std::vector<int32_t> *ids = nullptr);
    std::vector<float> run_forward(bool verify, const std::vector<int32_t> &ids, int64_t n_logits, bool full);
    void check_state(const char *after);
};

// Ranks 1..N-1: run the calls rank 0 sends until finish() (or a failure: the control channel / communicator is
// poisoned so rank 0's waits end, and the error rethrown).
void tp_executor(Qwen4ExpSession &ses, TpControl &ctl);

}  // namespace strix
