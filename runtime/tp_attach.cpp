// tp_attach (runtime/tp_mirror.hpp) - a session's exchanges through the RDMA communicator. Its own
// file (in strix_tp, with runtime/tp_comm.hip) so that TpDriver / tp_executor need no RDMA library.

#include "runtime/tp_mirror.hpp"

#include "common/check.hpp"
#include "common/hip_check.hpp"
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
    // Split draft head: rank 0's draft-head input u (d activations) to every rank: an all-gather, executors keep block 0
    const size_t u_bytes = (size_t)D.d * (act == kernels::Act::F32 ? 4 : 2);
    auto gathered_u = std::make_shared<DeviceBuffer<uint8_t>>(u_bytes * (size_t)comm.world(), "TP gathered draft input");
    const int64_t lm_rows = D.lm_rows;
    TpComm *c = &comm;
    ses.set_exchange([c, act, gathered, gathered_u, u_bytes, lm_rows](void *buf, int64_t elems, int kind, hipStream_t stream, void *out) {
        if (kind == 4) {  // split draft head: broadcast rank 0's u (elems activations)
            STRIX_CHECK((size_t)elems * (act == kernels::Act::F32 ? 4 : 2) == u_bytes, "TP draft input: ", elems, " elems");
            c->allgather(buf, u_bytes, gathered_u->get(), stream);
            if (c->rank() != 0)
                STRIX_HIP_CHECK(hipMemcpyAsync(buf, gathered_u->get(), u_bytes, hipMemcpyDeviceToDevice, stream), "TP draft input");
            return;
        }
        if (kind == 5) {  // split draft head: every rank's draft part (16 bytes each) into out, in rank order
            c->allgather(buf, (size_t)elems * 16, out, stream);
            return;
        }
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
