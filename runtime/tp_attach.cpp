// strixite-tp2: tp_attach (runtime/tp_mirror.hpp) - a session's exchanges through the RDMA communicator. Its own
// file (in strix_tp, with runtime/tp_comm.hip) so that TpDriver / tp_executor need no RDMA library.

#include "runtime/tp_mirror.hpp"

#include "common/check.hpp"
#include "runtime/device_buffer.hpp"

#include <memory>

namespace strix {

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

}  // namespace strix
