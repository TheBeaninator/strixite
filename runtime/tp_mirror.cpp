#include "runtime/tp_mirror.hpp"

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "formats/strixw.hpp"  // strix_hash64
#include "kernels/logits_topk.hpp"
#include "runtime/device_buffer.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>

namespace strix {

namespace {

// v2: + c (the candidate n_valid folded into the forwards), the verify ops, state hash and stats.
constexpr uint32_t kMsgMagic = 0x54503233, kReplyMagic = 0x54503252;
enum Op : uint32_t {
    kReset = 1,
    kForward,
    kWantCandidates,  // kept for compatibility; TpDriver folds n_valid into the forward messages
    kSetLookahead,
    kPrefetchPle,
    kMakeSnapshot,
    kSave,
    kRestore,
    kExit,
    kForwardVerify,
    kKeepVerify,
    kKeepVerifyPrefix,
    kDropVerify,
    kStateHash,
    kStats,
};
enum : int64_t { kFull = 1, kHash = 2, kLogitsHash = 4 };

struct Msg {
    uint32_t magic, op;
    uint64_t seq;
    int64_t a, b, c;
    uint64_t n;  // int32 ids following
};
static_assert(sizeof(Msg) == 48, "Msg v2 is 48 bytes");
struct Reply {
    uint32_t magic, status;
    uint64_t seq, n_floats, n_hash;
};

bool replicated_name(const std::string &n) {
    auto ends = [&](const char *e) {
        const size_t k = std::strlen(e);
        return n.size() >= k && n.compare(n.size() - k, k, e) == 0;
    };
    return n == "embed_streams" || n == "final_mixed" || ends(".out") || ends(".mixer") || ends(".moe") || ends(".ple_out");
}

// Hashes the replicated probes of one forward (names in call order: the same on every rank), then its outputs.
struct Hasher {
    std::vector<uint64_t> h;
    std::vector<std::string> names;
    std::vector<uint8_t> host;
    Qwen4ExpProbe probe() {
        return [this](const std::string &name, const void *dev, int64_t rows, int64_t cols, ProbeType type) {
            if (!replicated_name(name)) return;
            const size_t bytes = (size_t)(rows * cols) * (type == ProbeType::BF16 ? 2 : 4);
            host.resize(bytes);
            STRIX_HIP_CHECK(hipMemcpy(host.data(), dev, bytes, hipMemcpyDeviceToHost), "hash probe '", name, "'");
            add(name, host.data(), bytes);
        };
    }
    void add(const std::string &name, const void *p, size_t bytes) {
        h.push_back(strix_hash64(p, bytes));
        names.push_back(name);
    }
    // The forward's outputs: the merged candidates (replicated under TP) and, kLogitsHash, the returned rows.
    void outputs(const Qwen4ExpSession &ses, int64_t cand, int64_t n_logits, const std::vector<float> &l, int64_t flags) {
        if (cand > 0 && n_logits > 0) {
            const Qwen4ExpSession::Candidates &c = ses.candidates();
            uint64_t x[2] = {strix_hash64(c.cand.data(), c.cand.size() * sizeof(kernels::LogitCand)),
                             strix_hash64(c.nan.data(), c.nan.size() * 4)};
            add("candidates", x, sizeof x);
        }
        if (flags & kLogitsHash) add("logits", l.data(), l.size() * 4);
    }
};

}  // namespace

// ---------------- in-process loopback ----------------

struct TpLoopback::Shared {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<uint8_t> q[2];  // bytes waiting for rank r
    bool poisoned = false;
    std::string err;
};

struct TpLoopback::End : TpControl {
    std::shared_ptr<Shared> sh;
    int r;
    End(std::shared_ptr<Shared> s, int rank) : sh(std::move(s)), r(rank) {}
    int rank() const override { return r; }
    int world() const override { return 2; }
    void send(int peer, const void *p, size_t n) override {
        STRIX_CHECK(peer == 1 - r, "TpLoopback::send to ", peer, " from ", r);
        {
            std::lock_guard<std::mutex> g(sh->mu);
            const auto *b = static_cast<const uint8_t *>(p);
            sh->q[peer].insert(sh->q[peer].end(), b, b + n);
        }
        sh->cv.notify_all();
    }
    void recv(int peer, void *p, size_t n, double timeout_s) override {
        STRIX_CHECK(peer == 1 - r, "TpLoopback::recv from ", peer, " on ", r);
        std::unique_lock<std::mutex> g(sh->mu);
        const bool ok = sh->cv.wait_for(g, std::chrono::duration<double>(std::min(timeout_s, 1e7)),
                                        [&] { return sh->q[r].size() >= n || sh->poisoned; });
        STRIX_CHECK(ok, "TpLoopback: recv timed out (", n, " bytes)");
        STRIX_CHECK(sh->q[r].size() >= n, "TpLoopback (rank ", r, ") poisoned: ", sh->err);
        auto *b = static_cast<uint8_t *>(p);
        std::copy(sh->q[r].begin(), sh->q[r].begin() + (std::ptrdiff_t)n, b);
        sh->q[r].erase(sh->q[r].begin(), sh->q[r].begin() + (std::ptrdiff_t)n);
    }
    void check() const override {
        std::lock_guard<std::mutex> g(sh->mu);
        STRIX_CHECK(!sh->poisoned, "TpLoopback (rank ", r, ") poisoned: ", sh->err);
    }
    void poison(const std::string &why) override {
        {
            std::lock_guard<std::mutex> g(sh->mu);
            if (!sh->poisoned) sh->err = why;
            sh->poisoned = true;
        }
        sh->cv.notify_all();
    }
};

TpLoopback::TpLoopback() : sh_(std::make_shared<Shared>()) {
    for (int r = 0; r < 2; ++r) ends_[r] = std::make_unique<End>(sh_, r);
}

TpLoopback::~TpLoopback() = default;

TpControl &TpLoopback::end(int rank) {
    STRIX_CHECK(rank == 0 || rank == 1, "TpLoopback::end(", rank, ")");
    return *ends_[rank];
}

// ---------------- rank 0 ----------------

TpDriver::TpDriver(Qwen4ExpSession &ses, TpControl *ctl) : ses_(ses), ctl_(ctl) {
    STRIX_CHECK(!ctl || ctl->rank() == 0, "TpDriver runs on rank 0 (this is rank ", ctl ? ctl->rank() : 0, ")");
}

TpDriver::~TpDriver() {
    try {
        finish();
    } catch (...) {
    }
}

void TpDriver::send_all(uint32_t op, int64_t a, int64_t b, int64_t c, const std::vector<int32_t> *ids) {
    if (!ctl_) return;
    Msg m{kMsgMagic, op, ++seq_, a, b, c, ids ? (uint64_t)ids->size() : 0};
    for (int r = 1; r < ctl_->world(); ++r) {
        ctl_->send(r, &m, sizeof m);
        if (ids && !ids->empty()) ctl_->send(r, ids->data(), ids->size() * 4);
    }
}

void TpDriver::reset() {
    ++ops_["reset"];
    cand_ = 0;
    send_all(kReset);
    ses_.reset();
    check_state("reset");
}

std::vector<float> TpDriver::forward(const std::vector<int32_t> &ids, int64_t n_logits, bool full) {
    return run_forward(false, ids, n_logits, full);
}

std::vector<float> TpDriver::forward_verify(const std::vector<int32_t> &ids, int64_t n_logits, bool full) {
    return run_forward(true, ids, n_logits, full);
}

std::vector<float> TpDriver::run_forward(bool verify, const std::vector<int32_t> &ids, int64_t n_logits, bool full) {
    const int64_t cand = cand_;
    cand_ = 0;
    const int64_t flags = (full ? kFull : 0) | (hash_ ? kHash : 0) | (hash_ && shadow_ ? kLogitsHash : 0);
    ++ops_[verify ? "forward_verify" : "forward"];
    send_all(verify ? kForwardVerify : kForward, n_logits, flags, cand, &ids);
    Hasher hasher;
    std::vector<float> out;
    try {
        Qwen4ExpProbe probe = hash_ ? hasher.probe() : Qwen4ExpProbe{};
        if (probe_) {
            if (probe) {
                Qwen4ExpProbe a = std::move(probe), b = probe_;
                probe = [a, b](const std::string &n, const void *d, int64_t r, int64_t c, ProbeType t) {
                    a(n, d, r, c, t);
                    b(n, d, r, c, t);
                };
            } else {
                probe = probe_;
            }
        }
        out = verify ? ses_.forward_verify(ids, n_logits, probe) : ses_.forward(ids, n_logits, probe);
    } catch (...) {
        if (ctl_) ctl_->poison(verify ? "rank 0's forward_verify failed" : "rank 0's forward failed");
        throw;
    }
    if (hash_) hasher.outputs(ses_, cand, n_logits, out, flags);
    if (!ctl_) return out;
    ctl_->check();
    if (!flags) return out;
    const Qwen4ExpDims &D = ses_.model().dims();
    std::vector<float> whole;
    if (full) {
        STRIX_CHECK(out.size() == (size_t)(n_logits * D.lm_rows), "TpDriver::forward: full logits with candidates on");
        whole.resize((size_t)(n_logits * D.vocab));
        for (int64_t i = 0; i < n_logits; ++i)
            std::memcpy(whole.data() + i * D.vocab, out.data() + i * D.lm_rows, (size_t)D.lm_rows * 4);
    }
    if (hash_) ++hs_.forwards, hs_.verify_forwards += verify;
    std::vector<float> part;
    std::vector<uint64_t> h;
    for (int r = 1; r < ctl_->world(); ++r) {
        Reply rp{};
        ctl_->recv(r, &rp, sizeof rp);
        STRIX_CHECK(rp.magic == kReplyMagic && rp.seq == seq_, "TpDriver: reply from rank ", r, " out of step (seq ", rp.seq,
                    ", want ", seq_, ")");
        STRIX_CHECK(rp.status == 0, "TpDriver: rank ", r, " failed its forward");
        part.resize(rp.n_floats);
        h.resize(rp.n_hash);
        if (rp.n_floats) ctl_->recv(r, part.data(), rp.n_floats * 4);
        if (rp.n_hash) ctl_->recv(r, h.data(), rp.n_hash * 8);
        if (full) {
            STRIX_CHECK(rp.n_floats == (uint64_t)(n_logits * D.lm_rows), "TpDriver: rank ", r, " sent ", rp.n_floats, " logits");
            for (int64_t i = 0; i < n_logits; ++i)
                std::memcpy(whole.data() + i * D.vocab + r * D.lm_rows, part.data() + i * D.lm_rows, (size_t)D.lm_rows * 4);
        }
        if (hash_) {
            const char *what = verify ? "verify" : "forward";
            if (h.size() != hasher.h.size()) {
                ++hs_.mismatched, hs_.verify_mismatched += verify;
                if (hs_.first_mismatch.empty())
                    hs_.first_mismatch = cat(what, " ", hs_.forwards - 1, ": rank ", r, " hashed ", h.size(), " probes, rank 0 ",
                                             hasher.h.size());
                continue;
            }
            for (size_t k = 0; k < h.size(); ++k) {
                const bool x = hasher.names[k] == "embed_streams" || hasher.names[k].find(".out") != std::string::npos ||
                               hasher.names[k].find(".ple_out") != std::string::npos;
                ++hs_.compared, hs_.x_compared += x;
                if (h[k] != hasher.h[k]) {
                    ++hs_.mismatched, hs_.x_mismatched += x, hs_.verify_mismatched += verify;
                    if (hs_.first_mismatch.empty())
                        hs_.first_mismatch = cat(what, " ", hs_.forwards - 1, " (", ids.size(), " tokens): ", hasher.names[k],
                                                 " rank ", r);
                }
            }
        }
    }
    return full ? whole : out;
}

void TpDriver::check_state(const char *after) {
    if (!state_check_ || !ctl_) return;
    send_all(kStateHash);
    const uint64_t mine = ses_.state_hash();
    ++hs_.state_checks;
    for (int r = 1; r < ctl_->world(); ++r) {
        Reply rp{};
        ctl_->recv(r, &rp, sizeof rp);
        STRIX_CHECK(rp.magic == kReplyMagic && rp.seq == seq_ && rp.n_hash == 1, "TpDriver: state hash reply from rank ", r,
                    " out of step (seq ", rp.seq, ", want ", seq_, ")");
        STRIX_CHECK(rp.status == 0, "TpDriver: rank ", r, " failed its state hash");
        uint64_t h = 0;
        ctl_->recv(r, &h, 8);
        if (h != mine) {
            ++hs_.state_mismatched;
            if (hs_.first_state_mismatch.empty())
                hs_.first_state_mismatch = cat("state check ", hs_.state_checks - 1, " after ", after, ": rank ", r,
                                               " at position ", ses_.pos());
        }
    }
}

void TpDriver::want_candidates(int64_t n_valid) {
    ses_.want_candidates(n_valid);  // rank 0 now; the executors with the forward's message
    cand_ = n_valid;
}

void TpDriver::keep_verify() {
    ++ops_["keep_verify"];
    send_all(kKeepVerify);
    try {
        ses_.keep_verify();
    } catch (...) {
        if (ctl_) ctl_->poison("rank 0's keep_verify failed");
        throw;
    }
    check_state("keep_verify");
}

void TpDriver::keep_verify_prefix(int64_t rows) {
    ++ops_["keep_verify_prefix"];
    const bool plant = plant_ && rows >= 2;
    if (plant) plant_ = false, ++ops_["planted_bug"];
    send_all(kKeepVerifyPrefix, plant ? rows - 1 : rows);
    try {
        ses_.keep_verify_prefix(rows);
    } catch (...) {
        if (ctl_) ctl_->poison("rank 0's keep_verify_prefix failed");
        throw;
    }
    check_state("keep_verify_prefix");
}

void TpDriver::drop_verify() {
    ++ops_["drop_verify"];
    send_all(kDropVerify);
    try {
        ses_.drop_verify();
    } catch (...) {
        if (ctl_) ctl_->poison("rank 0's drop_verify failed");
        throw;
    }
    check_state("drop_verify");
}

void TpDriver::set_lookahead(const std::vector<int32_t> &next_ids) {
    send_all(kSetLookahead, 0, 0, 0, &next_ids);
    ses_.set_lookahead(next_ids);
}

void TpDriver::prefetch_ple(const std::vector<int32_t> &ids, int64_t first) {
    ++ops_["prefetch_ple"];
    send_all(kPrefetchPle, first, 0, 0, &ids);
    ses_.prefetch_ple(ids, first);
}

int TpDriver::make_snapshot() {
    const int id = (int)snaps_.size();
    send_all(kMakeSnapshot, id);
    snaps_.emplace(id, ses_.make_snapshot());
    return id;
}

void TpDriver::save(int id) {
    ++ops_["save"];
    send_all(kSave, id);
    ses_.save(snaps_.at(id));
}

void TpDriver::restore(int id) {
    ++ops_["restore"];
    send_all(kRestore, id);
    ses_.restore(snaps_.at(id));
    check_state("restore");
}

std::vector<Qwen4ExpSession::PleStats> TpDriver::executor_stats() {
    std::vector<Qwen4ExpSession::PleStats> out;
    if (!ctl_) return out;
    send_all(kStats);
    for (int r = 1; r < ctl_->world(); ++r) {
        Reply rp{};
        ctl_->recv(r, &rp, sizeof rp);
        STRIX_CHECK(rp.magic == kReplyMagic && rp.seq == seq_ && rp.status == 0 &&
                        rp.n_floats * 4 == sizeof(Qwen4ExpSession::PleStats),
                    "TpDriver: stats reply from rank ", r, " out of step or failed");
        Qwen4ExpSession::PleStats s;
        ctl_->recv(r, &s, sizeof s);
        out.push_back(s);
    }
    return out;
}

void TpDriver::finish() {
    if (finished_) return;
    finished_ = true;
    send_all(kExit);
}

// ---------------- ranks 1..N-1 ----------------

void tp_executor(Qwen4ExpSession &ses, TpControl &ctl) {
    static_assert(sizeof(Qwen4ExpSession::PleStats) % 4 == 0, "PleStats goes as floats");
    std::map<int64_t, Qwen4ExpSnapshot> snaps;
    std::vector<int32_t> ids;
    uint64_t want_seq = 0;
    while (true) {
        Msg m{};
        ctl.recv(0, &m, sizeof m);
        STRIX_CHECK(m.magic == kMsgMagic && m.seq == ++want_seq, "tp_executor: message out of step (magic ", m.magic, ", seq ",
                    m.seq, ", want ", want_seq, ")");
        ids.resize(m.n);
        if (m.n) ctl.recv(0, ids.data(), m.n * 4);
        const bool replies = ((m.op == kForward || m.op == kForwardVerify) && m.b) || m.op == kStateHash || m.op == kStats;
        try {
            switch (m.op) {
                case kReset: ses.reset(); break;
                case kForward:
                case kForwardVerify: {
                    Hasher hasher;
                    const bool hash = m.b & kHash;
                    if (m.c) ses.want_candidates(m.c);
                    const Qwen4ExpProbe probe = hash ? hasher.probe() : Qwen4ExpProbe{};
                    std::vector<float> l = m.op == kForwardVerify ? ses.forward_verify(ids, m.a, probe) : ses.forward(ids, m.a, probe);
                    ctl.check();
                    if (hash) hasher.outputs(ses, m.c, m.a, l, m.b);
                    if (m.b) {
                        const bool full = m.b & kFull;
                        Reply rp{kReplyMagic, 0, m.seq, full ? l.size() : 0, hash ? hasher.h.size() : 0};
                        ctl.send(0, &rp, sizeof rp);
                        if (rp.n_floats) ctl.send(0, l.data(), l.size() * 4);
                        if (rp.n_hash) ctl.send(0, hasher.h.data(), hasher.h.size() * 8);
                    }
                    break;
                }
                case kWantCandidates: ses.want_candidates(m.a); break;
                case kKeepVerify: ses.keep_verify(); break;
                case kKeepVerifyPrefix: ses.keep_verify_prefix(m.a); break;
                case kDropVerify: ses.drop_verify(); break;
                case kStateHash: {
                    const uint64_t h = ses.state_hash();
                    Reply rp{kReplyMagic, 0, m.seq, 0, 1};
                    ctl.send(0, &rp, sizeof rp);
                    ctl.send(0, &h, 8);
                    break;
                }
                case kStats: {
                    const Qwen4ExpSession::PleStats s = ses.ple_stats();
                    Reply rp{kReplyMagic, 0, m.seq, sizeof s / 4, 0};
                    ctl.send(0, &rp, sizeof rp);
                    ctl.send(0, &s, sizeof s);
                    break;
                }
                case kSetLookahead: ses.set_lookahead(ids); break;
                case kPrefetchPle: ses.prefetch_ple(ids, m.a); break;
                case kMakeSnapshot: snaps.emplace(m.a, ses.make_snapshot()); break;
                case kSave: ses.save(snaps.at(m.a)); break;
                case kRestore: ses.restore(snaps.at(m.a)); break;
                case kExit: return;
                default: STRIX_FAIL("tp_executor: unknown op ", m.op);
            }
        } catch (const std::exception &e) {
            ctl.poison(cat("rank ", ctl.rank(), ": ", e.what()));
            if (replies) {
                Reply rp{kReplyMagic, 1, m.seq, 0, 0};
                try {
                    ctl.send(0, &rp, sizeof rp);
                } catch (...) {
                }
            }
            throw;
        }
    }
}

}  // namespace strix
