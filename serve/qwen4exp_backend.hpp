#pragma once

// LmBackend on the real model: one Qwen4ExpSession (BF16 activations, WMMA prefill - the shipping settings) and one
// Qwen4ExpSnapshot slot.
//
// Tensor parallelism (rank 0 of a TP group): set_tp_driver(drv) routes every call that changes the trunk's state
// (forwards, verifies and their keeps / drops, resets, snapshots, PLE hints) through drv - mirrored to the executors
// (runtime/tp_mirror.hpp). Drafts go through drv too: with the split draft head (TpConfig::draft_split) every
// rank scores its share of the draft vocabulary, so the executors must be told to run their half (TpDriver::
// forward_mtp_top2); without the split the head lives on rank 0 and drv just forwards the call to the session.
// Under TP a forward's logits are rank 0's vocabulary share, so forward_rows / forward_verify_rows refuse samplers that
// need whole rows (the served defaults - greedy, top_k <= 20 - take candidates); forward / forward_verify gather whole
// rows over the control channel (slow: quality runs). The prompt cache's export / import is not mirrored yet.

#include "runtime/qwen4exp.hpp"
#include "serve/engine.hpp"

namespace strix {

class TpDriver;

class Qwen4ExpBackend : public LmBackend {
public:
    // mtp_vocab: the draft's vocabulary (Qwen4ExpSession::set_mtp_vocab), 0 = the whole one.
    Qwen4ExpBackend(const Qwen4ExpModel &model, int64_t capacity, int64_t chunk, bool use_mtp = true,
                    int64_t mtp_vocab = 0);
    int64_t capacity() const override { return session_.capacity(); }
    int64_t max_chunk() const override { return session_.max_tokens(); }
    int64_t logits_row() const override { return model_.dims().vocab; }
    int64_t pos() const override { return session_.pos(); }
    void reset() override;
    std::vector<float> forward(const std::vector<int32_t> &ids, bool want_logits) override {
        return forward(ids, (int64_t)(want_logits ? 1 : 0));
    }
    void set_lookahead(const std::vector<int32_t> &next_ids) override;
    std::vector<float> forward(const std::vector<int32_t> &ids, int64_t n_logits) override;
    bool has_mtp() const override { return session_.has_mtp(); }
    std::vector<float> forward_mtp(int32_t token_id, int64_t step) override {
        return session_.forward_mtp(token_id, step);
    }
    Top2 forward_mtp_top2(int32_t token_id, int64_t step) override;  // reduced on the GPU, one small copy back
    std::vector<float> forward_verify(const std::vector<int32_t> &ids, int64_t n_logits) override;
    LogitRows forward_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid) override;
    LogitRows forward_verify_rows(const std::vector<int32_t> &ids, int64_t n_logits, bool cands, int64_t n_valid) override;
    void keep_verify() override;
    void drop_verify() override;
    BackendStats backend_stats() const override;
    void prefetch_ple(const std::vector<int32_t> &ids, int64_t first) override;
    void keep_verify_prefix(const std::vector<int32_t> &ids, int64_t rows) override;
    void save_snapshot(int slot) override;
    int64_t snapshot_pos(int slot) const override;
    bool can_restore_snapshot(int slot) const override;
    void restore_snapshot(int slot) override;
    void export_snapshot(int slot, HostBuffer &out, int64_t from = 0) override;
    void import_state(const HostBuffer &state, int64_t n, int64_t from = 0) override;
    std::string state_fingerprint() const override;
    std::string describe() const override;

    // Tensor parallelism: drv drives session() (rank 0); null = world 1 (the default). Allocates drv's 4 slots.
    void set_tp_driver(TpDriver *drv);
    Qwen4ExpSession &session() { return session_; }

private:
    LogitRows candidate_rows(int64_t rows) const;  // session_.candidates() of the last forward, checked
    const Qwen4ExpModel &model_;
    Qwen4ExpSession session_;
    Qwen4ExpSnapshot snapshots_[4];
    Qwen4ExpSnapshot &snap(int slot);
    const Qwen4ExpSnapshot &snap(int slot) const;
    TpDriver *tp_ = nullptr;
    int tp_slot_[4] = {-1, -1, -1, -1};  // the driver's snapshot ids for slots 0..3
    int tp_id(int slot) const {
        (void)snap(slot);  // checks the slot
        return tp_slot_[slot];
    }
};

}  // namespace strix
