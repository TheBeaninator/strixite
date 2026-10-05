#include "runtime/tp_mirror.hpp"

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "formats/strixw.hpp"  // strix_hash64
#include "kernels/logits_topk.hpp"
#include "runtime/device_buffer.hpp"

#include <cstdio>
#include <cstring>
#include <memory>

namespace strix {

namespace {

constexpr uint32_t kMsgMagic = 0x5450324d, kReplyMagic = 0x54503252;
enum Op : uint32_t { kReset = 1, kForward, kWantCandidates, kSetLookahead, kPrefetchPle, kMakeSnapshot, kSave, kRestore, kExit };
enum : int64_t { kFull = 1, kHash = 2 };

struct Msg {
    uint32_t magic, op;
    uint64_t seq;
    int64_t a, b;
    uint64_t n;  // int32 ids following
};
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

// Hashes the replicated probes of one forward (names in call order: the same on every rank).
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
            h.push_back(strix_hash64(host.data(), bytes));
            names.push_back(name);
        };
    }
};

}  // namespace

void tp_attach(Qwen4ExpSession &ses, const Qwen4ExpModel &model, TpComm &comm) {
    const Qwen4ExpDims &D = model.dims();
    STRIX_CHECK(D.tp_world == comm.world() && D.tp_rank == comm.rank(), "tp_attach: model rank ", D.tp_rank, " of ",
                D.tp_world, ", communicator rank ", comm.rank(), " of ", comm.world());
    const kernels::Act act = model.act();
    const int64_t rows = Qwen4ExpSession::kMaxLogits;
    auto gathered = std::make_shared<DeviceBuffer<uint8_t>>(tp_candidate_block_bytes(rows) * (size_t)comm.world(),
                                                            "TP gathered candidates");
    const int64_t lm_rows = D.lm_rows;
    TpComm *c = &comm;
    ses.set_exchange([c, act, gathered, lm_rows](void *buf, int64_t elems, int kind, hipStream_t stream, void *out) {
        if (kind < 2) {
            c->allreduce(buf, elems, act, stream);
            return;
        }
        if (kind == 3) {
            c->allreduce_f32_to_bf16(static_cast<const float *>(buf), elems, out, stream);
            return;
        }
        STRIX_CHECK(buf && elems >= 1 && elems <= Qwen4ExpSession::kMaxLogits, "TP candidate exchange: ", elems, " rows");
        c->allgather(buf, tp_candidate_block_bytes(elems), gathered->get(), stream);
        tp_merge_candidates(gathered->get(), c->world(), elems, lm_rows, buf, stream);
    });
}

TpDriver::TpDriver(Qwen4ExpSession &ses, TpComm *comm) : ses_(ses), comm_(comm) {
    STRIX_CHECK(!comm || comm->rank() == 0, "TpDriver runs on rank 0 (this is rank ", comm ? comm->rank() : 0, ")");
}

TpDriver::~TpDriver() {
    try {
        finish();
    } catch (...) {
    }
}

void TpDriver::send_all(uint32_t op, int64_t a, int64_t b, const std::vector<int32_t> *ids) {
    if (!comm_) return;
    Msg m{kMsgMagic, op, ++seq_, a, b, ids ? (uint64_t)ids->size() : 0};
    for (int r = 1; r < comm_->world(); ++r) {
        comm_->send(r, &m, sizeof m);
        if (ids && !ids->empty()) comm_->send(r, ids->data(), ids->size() * 4);
    }
}

void TpDriver::reset() {
    send_all(kReset);
    ses_.reset();
}

std::vector<float> TpDriver::forward(const std::vector<int32_t> &ids, int64_t n_logits, bool full) {
    const int64_t flags = (full ? kFull : 0) | (hash_ ? kHash : 0);
    send_all(kForward, n_logits, flags, &ids);
    Hasher hasher;
    std::vector<float> out;
    try {
        out = ses_.forward(ids, n_logits, hash_ ? hasher.probe() : Qwen4ExpProbe{});
    } catch (...) {
        if (comm_) comm_->poison("rank 0's forward failed");
        throw;
    }
    if (!comm_) return out;
    comm_->check();
    if (!flags) return out;
    const Qwen4ExpDims &D = ses_.model().dims();
    std::vector<float> whole;
    if (full) {
        STRIX_CHECK(out.size() == (size_t)(n_logits * D.lm_rows), "TpDriver::forward: full logits with candidates on");
        whole.resize((size_t)(n_logits * D.vocab));
        for (int64_t i = 0; i < n_logits; ++i)
            std::memcpy(whole.data() + i * D.vocab, out.data() + i * D.lm_rows, (size_t)D.lm_rows * 4);
    }
    if (hash_) ++hs_.forwards;
    std::vector<float> part;
    std::vector<uint64_t> h;
    for (int r = 1; r < comm_->world(); ++r) {
        Reply rp{};
        comm_->recv(r, &rp, sizeof rp);
        STRIX_CHECK(rp.magic == kReplyMagic && rp.seq == seq_, "TpDriver: reply from rank ", r, " out of step (seq ", rp.seq,
                    ", want ", seq_, ")");
        STRIX_CHECK(rp.status == 0, "TpDriver: rank ", r, " failed its forward");
        part.resize(rp.n_floats);
        h.resize(rp.n_hash);
        if (rp.n_floats) comm_->recv(r, part.data(), rp.n_floats * 4);
        if (rp.n_hash) comm_->recv(r, h.data(), rp.n_hash * 8);
        if (full) {
            STRIX_CHECK(rp.n_floats == (uint64_t)(n_logits * D.lm_rows), "TpDriver: rank ", r, " sent ", rp.n_floats, " logits");
            for (int64_t i = 0; i < n_logits; ++i)
                std::memcpy(whole.data() + i * D.vocab + r * D.lm_rows, part.data() + i * D.lm_rows, (size_t)D.lm_rows * 4);
        }
        if (hash_) {
            if (h.size() != hasher.h.size()) {
                ++hs_.mismatched;
                if (hs_.first_mismatch.empty())
                    hs_.first_mismatch = cat("forward ", hs_.forwards - 1, ": rank ", r, " hashed ", h.size(), " probes, rank 0 ",
                                             hasher.h.size());
                continue;
            }
            for (size_t k = 0; k < h.size(); ++k) {
                const bool x = hasher.names[k] == "embed_streams" || hasher.names[k].find(".out") != std::string::npos ||
                               hasher.names[k].find(".ple_out") != std::string::npos;
                ++hs_.compared, hs_.x_compared += x;
                if (h[k] != hasher.h[k]) {
                    ++hs_.mismatched, hs_.x_mismatched += x;
                    if (hs_.first_mismatch.empty())
                        hs_.first_mismatch = cat("forward ", hs_.forwards - 1, " (", ids.size(), " tokens): ", hasher.names[k],
                                                 " rank ", r);
                }
            }
        }
    }
    return full ? whole : out;
}

void TpDriver::want_candidates(int64_t n_valid) {
    send_all(kWantCandidates, n_valid);
    ses_.want_candidates(n_valid);
}

void TpDriver::set_lookahead(const std::vector<int32_t> &next_ids) {
    send_all(kSetLookahead, 0, 0, &next_ids);
    ses_.set_lookahead(next_ids);
}

void TpDriver::prefetch_ple(const std::vector<int32_t> &ids, int64_t first) {
    send_all(kPrefetchPle, first, 0, &ids);
    ses_.prefetch_ple(ids, first);
}

int TpDriver::make_snapshot() {
    const int id = (int)snaps_.size();
    send_all(kMakeSnapshot, id);
    snaps_.emplace(id, ses_.make_snapshot());
    return id;
}

void TpDriver::save(int id) {
    send_all(kSave, id);
    ses_.save(snaps_.at(id));
}

void TpDriver::restore(int id) {
    send_all(kRestore, id);
    ses_.restore(snaps_.at(id));
}

void TpDriver::finish() {
    if (finished_) return;
    finished_ = true;
    send_all(kExit);
}

void tp_executor(Qwen4ExpSession &ses, TpComm &comm) {
    std::map<int64_t, Qwen4ExpSnapshot> snaps;
    std::vector<int32_t> ids;
    uint64_t want_seq = 0;
    while (true) {
        Msg m{};
        comm.recv(0, &m, sizeof m);
        STRIX_CHECK(m.magic == kMsgMagic && m.seq == ++want_seq, "tp_executor: message out of step (magic ", m.magic, ", seq ",
                    m.seq, ", want ", want_seq, ")");
        ids.resize(m.n);
        if (m.n) comm.recv(0, ids.data(), m.n * 4);
        try {
            switch (m.op) {
                case kReset: ses.reset(); break;
                case kForward: {
                    Hasher hasher;
                    const bool hash = m.b & kHash;
                    std::vector<float> l = ses.forward(ids, m.a, hash ? hasher.probe() : Qwen4ExpProbe{});
                    comm.check();
                    if (m.b) {
                        const bool full = m.b & kFull;
                        Reply rp{kReplyMagic, 0, m.seq, full ? l.size() : 0, hash ? hasher.h.size() : 0};
                        comm.send(0, &rp, sizeof rp);
                        if (rp.n_floats) comm.send(0, l.data(), l.size() * 4);
                        if (rp.n_hash) comm.send(0, hasher.h.data(), hasher.h.size() * 8);
                    }
                    break;
                }
                case kWantCandidates: ses.want_candidates(m.a); break;
                case kSetLookahead: ses.set_lookahead(ids); break;
                case kPrefetchPle: ses.prefetch_ple(ids, m.a); break;
                case kMakeSnapshot: snaps.emplace(m.a, ses.make_snapshot()); break;
                case kSave: ses.save(snaps.at(m.a)); break;
                case kRestore: ses.restore(snaps.at(m.a)); break;
                case kExit: return;
                default: STRIX_FAIL("tp_executor: unknown op ", m.op);
            }
        } catch (const std::exception &e) {
            comm.poison(cat("rank ", comm.rank(), ": ", e.what()));
            if (m.op == kForward && m.b) {
                Reply rp{kReplyMagic, 1, m.seq, 0, 0};
                try {
                    comm.send(0, &rp, sizeof rp);
                } catch (...) {
                }
            }
            throw;
        }
    }
}

}  // namespace strix
