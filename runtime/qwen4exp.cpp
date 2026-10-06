#include "runtime/qwen4exp.hpp"

#include "runtime/ngram_table.hpp"
#include "formats/strixw.hpp"  // strix_hash64

#include "formats/q4_from_q8.hpp"

#include "common/hip_check.hpp"
#include "common/trace.hpp"
#include "kernels/attention.hpp"
#include "kernels/gdn.hpp"
#include "kernels/hc.hpp"
#include "kernels/hc_wmma.hpp"
#include "kernels/linear.hpp"
#include "kernels/moe_grouped.hpp"
#include "kernels/moe_router.hpp"
#include "kernels/mtp_pick.hpp"
#include "kernels/ple.hpp"
#include "kernels/qsa.hpp"
#include "kernels/residual.hpp"
#include "kernels/rope.hpp"
#include "kernels/swiglu.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <optional>
#include <thread>

namespace strix {

using kernels::Act;

namespace {

const std::string kLm = "model.language_model.";
constexpr int64_t kHD = kernels::kGdnHeadDim;

size_t es(Act a) { return a == Act::F32 ? 4 : 2; }

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

Qwen4ExpModel::Hc load_hc(const StrixwDevice &w, const std::string &p, const Qwen4ExpDims &D, bool inject) {
    const int64_t n4 = D.H * D.d;
    return {w.qw(p + "mix_down", {D.r + (inject ? D.H : 0), n4}), w.qw_chunk_major(p + "mix_up", {n4, D.r}),
            w.f32(p + "hc_norm.weight", {n4})};
}

}  // namespace

namespace {

bool ends_with(const std::string &s, const std::string &e) {
    return s.size() >= e.size() && s.compare(s.size() - e.size(), e.size(), e) == 0;
}

// The rows rank 0's LM head holds: its vocabulary share, at least the draft's rows (TpConfig::draft_rows).
int64_t rank0_head_rows(const Qwen4ExpDims &F, int N, int64_t draft_rows) {
    return std::min(F.vocab, std::max(F.vocab / N, draft_rows));
}

// The first n rows of a row-major quantized weight (the same data: a prefix view).
QWeightView rows_prefix(QWeightView v, int64_t n, const char *what) {
    STRIX_CHECK(n >= 1 && n <= v.N(), what, ": a prefix of ", n, " rows of ", v.N());
    switch (v.bits) {
        case 8: v.q8.N = n; break;
        case 6: v.q6.N = n; break;
        case 5: v.q5.N = n; break;
        case 4: v.q4.N = n; break;
        default: STRIX_FAIL(what, ": ", v.bits, " bits, expected 4, 5, 6 or 8");
    }
    return v;
}

// Rank r of N's share of each tensor (StrixwSlice): F the whole model's dims. See Qwen4ExpDims::tp_world.
std::optional<StrixwSlice> tp_slice(const StrixwTensor &t, const Qwen4ExpDims &F, int N, int r, int64_t draft_rows) {
    const std::string &n = t.name;
    // ST-3: the MTP draft head lives on rank 0 only, whole (its drafts and the catch-up need no exchange).
    if (n.rfind("mtp.", 0) == 0) return r == 0 ? std::nullopt : std::optional<StrixwSlice>(StrixwSlice{true, {}, 0, 0});
    if (n.rfind(kLm + "layers.", 0) != 0) {
        if (n == "lm_head.weight") {
            const int64_t V = F.vocab / N;
            if (r == 0) return StrixwSlice{false, {{0, rank0_head_rows(F, N, draft_rows)}}, 0, 0};
            return StrixwSlice{false, {{r * V, (r + 1) * V}}, 0, 0};
        }
        return std::nullopt;
    }
    const int64_t hd = kHD, gk = F.gk / N, gv = F.gv / N, hq = F.hq / N, hkv = std::max<int64_t>(1, F.hkv / N);
    const int64_t kvh = (r * hq) / (F.hq / F.hkv), I = F.inter / N;
    const int64_t qk = F.gk * hd, vw = F.gv * hd;  // whole q (= k) and v (= z) widths of the GDN projection
    StrixwSlice s;
    if (ends_with(n, "linear_attn.in_proj")) {  // [q | k | v | z | b | a], heads contiguous
        s.rows = {{r * gk * hd, (r + 1) * gk * hd},
                  {qk + r * gk * hd, qk + (r + 1) * gk * hd},
                  {2 * qk + r * gv * hd, 2 * qk + (r + 1) * gv * hd},
                  {2 * qk + vw + r * gv * hd, 2 * qk + vw + (r + 1) * gv * hd},
                  {2 * qk + 2 * vw + r * gv, 2 * qk + 2 * vw + (r + 1) * gv},
                  {2 * qk + 2 * vw + F.gv + r * gv, 2 * qk + 2 * vw + F.gv + (r + 1) * gv}};
    } else if (ends_with(n, "linear_attn.conv1d.weight")) {  // the q | k | v channels
        s.rows = {{r * gk * hd, (r + 1) * gk * hd},
                  {qk + r * gk * hd, qk + (r + 1) * gk * hd},
                  {2 * qk + r * gv * hd, 2 * qk + (r + 1) * gv * hd}};
    } else if (ends_with(n, "linear_attn.A_log") || ends_with(n, "linear_attn.dt_bias")) {
        s.rows = {{r * gv, (r + 1) * gv}};
    } else if (ends_with(n, "linear_attn.out_proj.weight")) {
        s.k0 = r * gv * hd, s.k1 = (r + 1) * gv * hd;
    } else if (ends_with(n, "self_attn.qkv")) {  // [q|gate per head | k | v | indexer q heads | indexer k]
        const int64_t ah = F.hd, qw = F.hq * 2 * ah, kv = F.hkv * ah;  // attention heads are F.hd wide (GDN: kHD)
        s.rows = {{r * hq * 2 * ah, (r + 1) * hq * 2 * ah},
                  {qw + kvh * ah, qw + (kvh + hkv) * ah},
                  {qw + kv + kvh * ah, qw + kv + (kvh + hkv) * ah},
                  {qw + 2 * kv, F.astride}};
    } else if (ends_with(n, "self_attn.o_proj.weight")) {
        s.k0 = r * hq * F.hd, s.k1 = (r + 1) * hq * F.hd;
    } else if (ends_with(n, "mlp.experts_gate_up")) {  // per expert [gate I | up I]
        const int64_t E = t.shape.at(0) / (2 * F.inter);
        for (int64_t e = 0; e < E; ++e) {
            const int64_t b = e * 2 * F.inter;
            s.rows.push_back({b + r * I, b + (r + 1) * I});
            s.rows.push_back({b + F.inter + r * I, b + F.inter + (r + 1) * I});
        }
    } else if (ends_with(n, "mlp.shared_expert.gate_up")) {
        s.rows = {{r * I, (r + 1) * I}, {F.inter + r * I, F.inter + (r + 1) * I}};
    } else if (ends_with(n, "mlp.experts_down") || ends_with(n, "mlp.shared_expert.down_proj.weight")) {
        s.k0 = r * I, s.k1 = (r + 1) * I;
    } else {
        return std::nullopt;
    }
    return s;
}

}  // namespace

Qwen4ExpModel::Qwen4ExpModel(const std::string &weights, const std::string &ngram, Act act,
                             bool allow_truncated, int64_t ngram_cache_rows, float yarn_factor, TpConfig tp)
    : act_(act) {
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, "Qwen4ExpModel: unsupported activation dtype ", (int)act);
    STRIX_CHECK(std::isfinite(yarn_factor) && yarn_factor >= 1.0f && yarn_factor <= 8.0f, "Qwen4ExpModel: yarn_factor = ",
                yarn_factor, ", expected 1 (RoPE as trained) .. 8");
    Qwen4ExpDims &D = dims_;
    const Qwen4ExpDims whole = D;  // the MTP head's (mtp_dims_)
    STRIX_CHECK(tp.draft_rows >= 1, "Qwen4ExpModel: draft_rows = ", tp.draft_rows);
    STRIX_CHECK((tp.world == 1 && tp.rank == 0) || ((tp.world == 2 || tp.world == 4) && tp.rank >= 0 && tp.rank < tp.world),
                "Qwen4ExpModel: tensor parallelism world ", tp.world, " rank ", tp.rank, " (world 1, 2 or 4)");
    if (tp.world > 1) {
        const Qwen4ExpDims F = D;  // the whole model's
        const int N = tp.world;
        STRIX_CHECK(F.gk % N == 0 && F.gv % N == 0 && F.hq % N == 0 && F.inter % N == 0 && F.vocab % N == 0,
                    "Qwen4ExpModel: dims don't split ", N, " ways");
        D.tp_world = N, D.tp_rank = tp.rank;
        D.gk = F.gk / N, D.gv = F.gv / N, D.hq = F.hq / N, D.hkv = std::max<int64_t>(1, F.hkv / N);
        D.inter = F.inter / N, D.lm_rows = F.vocab / N;
        D.conv_c = 2 * D.gk * kHD + D.gv * kHD, D.gz = D.gv * kHD, D.gstride = D.conv_c + D.gz + 2 * D.gv;
        D.idx_col = D.hq * 2 * D.hd + 2 * D.hkv * D.hd, D.astride = D.idx_col + D.idx_h * D.idx_d + D.idx_d;
        const int rank = tp.rank;
        const int64_t draft_rows = tp.draft_rows;
        w_ = std::make_unique<StrixwDevice>(
            weights, [F, N, rank, draft_rows](const StrixwTensor &t) { return tp_slice(t, F, N, rank, draft_rows); });
    } else {
        w_ = std::make_unique<StrixwDevice>(weights);
    }
    const StrixwFile &f = w_->file();
    while (f.find(kLm + "layers." + std::to_string(D.layers) + ".attn_hyper_connection.mix_down")) ++D.layers;
    STRIX_CHECK(D.layers >= 1, "Qwen4ExpModel: no decoder layers in '", weights, "'");
    std::string prefix;  // "0,1,..,k-1": what a truncated conversion's layers metadata reads
    for (int64_t i = 0; i < D.layers; ++i) prefix += (i ? "," : "") + std::to_string(i);
    const std::string &layers_meta = f.meta("layers");
    truncated_ = layers_meta != "all";
    STRIX_CHECK(f.meta("arch") == "qwen4exp" && f.meta("globals") == "yes" &&
                    (layers_meta == "all" || (allow_truncated && layers_meta == prefix)),
                "Qwen4ExpModel: '", weights, "' is arch '", f.meta("arch"), "', layers '", layers_meta, "', globals '",
                f.meta("globals"), "' - needs a qwen4exp conversion with the globals and every layer",
                allow_truncated ? " (or the first k: --layers 0,..,k-1)" : " (tools/convert_qwen4exp without --layers)");
    // Shared expert stacked as #512 or separate: the tensors decide; the metadata, where present, must agree.
    shared_separate_ = f.find(kLm + "layers.0.mlp.shared_expert.gate_up") != nullptr;
    if (f.meta().count("shared_expert"))
        STRIX_CHECK(f.meta("shared_expert") == (shared_separate_ ? "separate" : "stacked"), "Qwen4ExpModel: '", weights,
                    "' has shared_expert metadata '", f.meta("shared_expert"), "' but layer 0's shared expert is ",
                    shared_separate_ ? "separate" : "stacked");
    const int64_t n4 = D.H * D.d, Estack = D.experts + (shared_separate_ ? 0 : 1);
    const auto &emb_info = f.get(kLm + "embed_tokens.weight");
    if (emb_info.encoding == StrixwEncoding::Q8RowMajor) {
        embed_q8_ = w_->q8(kLm + "embed_tokens.weight", {D.vocab, D.d});
    } else {
        embed_ = w_->bf16(kLm + "embed_tokens.weight", {D.vocab, D.d});
    }
    // Rank 0 of N may hold more rows than its share (the draft's; tp_slice): the trunk's head is the first lm_rows.
    const int64_t head_rows = D.tp_world > 1 && D.tp_rank == 0 ? rank0_head_rows(whole, D.tp_world, tp.draft_rows) : D.lm_rows;
    draft_lm_ = w_->qw("lm_head.weight", {head_rows, D.d});
    lm_head_ = head_rows == D.lm_rows ? draft_lm_ : rows_prefix(draft_lm_, D.lm_rows, "Qwen4ExpModel: LM head");
    final_ = load_hc(*w_, kLm + "hyper_connection_mixer.", D, false);
    D.yarn_factor = yarn_factor;
    if (yarn_factor > 1.0f) {
        const kernels::YarnRope yarn = kernels::rope_yarn(D.rope_theta, D.rot, yarn_factor, D.trained_positions);
        inv_freq_ = DeviceBuffer<float>::from_host(yarn.inv_freq, "inv_freq (YaRN)");
        D.rope_scale = yarn.cos_sin_scale;
    } else {
        inv_freq_ = DeviceBuffer<float>::from_host(kernels::rope_inv_freq(D.rope_theta, D.rot), "inv_freq");
        D.rope_scale = 1.0f;
    }
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string L = kLm + "layers." + std::to_string(i) + ".";
        const bool attn = f.find(L + "self_attn.qkv") != nullptr;
        STRIX_CHECK(attn != (f.find(L + "linear_attn.in_proj") != nullptr), "Qwen4ExpModel: layer ", i,
                    " must have exactly one of self_attn.qkv / linear_attn.in_proj");
        D.is_attention.push_back(attn);
        Layer l;
        l.hc_attn = load_hc(*w_, L + "attn_hyper_connection.", D, true);
        l.hc_mlp = load_hc(*w_, L + "mlp_hyper_connection.", D, true);
        if (attn) {
            const std::string A = L + "self_attn.";
            l.qkv = w_->qw(A + "qkv", {D.astride, D.d});
            l.o_proj = w_->qw(A + "o_proj.weight", {D.d, D.hq * D.hd});
            l.q_norm = w_->f32(A + "q_norm.weight", {D.hd});
            l.k_norm = w_->f32(A + "k_norm.weight", {D.hd});
            l.idx_q_norm = w_->f32(A + "indexer.q_layernorm.weight", {D.idx_d});
            l.idx_k_norm = w_->f32(A + "indexer.k_layernorm.weight", {D.idx_d});
        } else {
            const std::string G = L + "linear_attn.";
            l.in_proj = w_->qw(G + "in_proj", {D.gstride, D.d});
            l.out_proj = w_->qw(G + "out_proj.weight", {D.d, D.gz});
            l.conv_w = w_->f32(G + "conv1d.weight", {D.conv_c, 1, kernels::kGdnConvWidth});
            l.A_log = w_->f32(G + "A_log", {D.gv});
            l.dt_bias = w_->f32(G + "dt_bias", {D.gv});
            l.gdn_norm = w_->f32(G + "norm.weight", {kHD});
        }
        l.gate_up = w_->qw(L + "mlp.experts_gate_up", {Estack * 2 * D.inter, D.d});
        l.down = w_->qw(L + "mlp.experts_down", {Estack * D.d, D.inter});
        l.router = w_->bf16(L + "mlp.router", {D.experts + 1, D.d});  // + the shared expert's gate, either way
        const bool sep = f.find(L + "mlp.shared_expert.gate_up") != nullptr;
        STRIX_CHECK(sep == shared_separate_, "Qwen4ExpModel: layer ", i, "'s shared expert is ",
                    sep ? "separate" : "stacked", " but layer 0's is ", shared_separate_ ? "separate" : "stacked");
        if (sep) {
            l.shared_gate_up = w_->qw(L + "mlp.shared_expert.gate_up", {2 * D.inter, D.d});
            l.shared_down = w_->qw(L + "mlp.shared_expert.down_proj.weight", {D.d, D.inter});
        }
        if (f.find(L + "ple.kv_proj")) {
            STRIX_CHECK(D.ple_layer < 0, "Qwen4ExpModel: PLE at layers ", D.ple_layer, " and ", i, "; expected one");
            D.ple_layer = i;
            const std::string P = L + "ple.";
            l.ple_kv = w_->qw(P + "kv_proj", {D.ple_ld, D.ple_e});
            l.ple_conv = w_->f32(P + "conv1d.weight", {n4, 1, D.ple_taps});
            l.ple_norm_key = w_->f32(P + "norm_key.weight", {n4});
            l.ple_norm_query = w_->f32(P + "norm_query.weight", {n4});
            l.ple_norm_conv = w_->f32(P + "norm_conv.weight", {n4});
            if (std::filesystem::is_regular_file(ngram)) {
                auto table = std::make_unique<NgramTableRows>(ngram, ngram_cache_rows);
                const std::string &want = f.meta("source_fingerprint"), &got = table->file().info().source_fingerprint;
                STRIX_CHECK(got == want, "Qwen4ExpModel: n-gram table '", ngram, "' was converted from checkpoint ", got,
                            ", the weights '", weights, "' from ", want, " - convert both from the same checkpoint");
                ngram_ = std::move(table);
            } else {
                STRIX_CHECK(std::filesystem::is_directory(ngram), "Qwen4ExpModel: n-gram source '", ngram,
                            "' is neither an ngram.table file nor the HF checkpoint directory");
                STRIX_CHECK(ngram_cache_rows == NgramTableRows::kDefaultCacheRows, "Qwen4ExpModel: ngram_cache_rows = ",
                            ngram_cache_rows, " with the checkpoint directory '", ngram,
                            "' as the n-gram source - only a table file has a row cache");
                ngram_ = std::make_unique<NgramRowsFromShards>(ngram, P + "ple_embedding.");
            }
            STRIX_CHECK(ngram_->row_dim() * 16 == D.ple_e, "Qwen4ExpModel: 16 n-gram rows of ", ngram_->row_dim(),
                        " don't make the PLE input width ", D.ple_e);
        }
        layers_.push_back(l);
    }
    // The MTP head: whole, on rank 0 only (tp_slice), so its shapes are the whole model's.
    mtp_dims_ = D;
    {
        Qwen4ExpDims &M = mtp_dims_;
        M.gk = whole.gk, M.gv = whole.gv, M.conv_c = whole.conv_c, M.gz = whole.gz, M.gstride = whole.gstride;
        M.hq = whole.hq, M.hkv = whole.hkv, M.idx_col = whole.idx_col, M.astride = whole.astride, M.inter = whole.inter;
    }
    const Qwen4ExpDims &MD = mtp_dims_;
    has_mtp_ = D.tp_rank == 0 && f.find("mtp.fc_embedding.weight") != nullptr;
    if (has_mtp_) {
        mtp_.fc_embedding = w_->qw("mtp.fc_embedding.weight", {D.d, D.d});
        mtp_.fc_hidden = w_->qw("mtp.fc_hidden.weight", {D.d, D.d});
        mtp_.pre_fc_norm_embedding = w_->f32("mtp.pre_fc_norm_embedding.weight", {D.d});
        mtp_.pre_fc_norm_hidden = w_->f32("mtp.pre_fc_norm_hidden.weight", {n4});
        mtp_.hc_mixer = load_hc(*w_, "mtp.hyper_connection_mixer.", D, false);
        const std::string M = "mtp.layers.0.";
        Layer &l = mtp_.layer;
        l.hc_attn = load_hc(*w_, M + "attn_hyper_connection.", D, true);
        l.hc_mlp = load_hc(*w_, M + "mlp_hyper_connection.", D, true);
        const std::string A = M + "self_attn.";
        l.qkv = w_->qw(A + "qkv", {MD.astride, D.d});
        l.o_proj = w_->qw(A + "o_proj.weight", {D.d, MD.hq * D.hd});
        l.q_norm = w_->f32(A + "q_norm.weight", {D.hd});
        l.k_norm = w_->f32(A + "k_norm.weight", {D.hd});
        l.idx_q_norm = w_->f32(A + "indexer.q_layernorm.weight", {D.idx_d});
        l.idx_k_norm = w_->f32(A + "indexer.k_layernorm.weight", {D.idx_d});
        l.gate_up = w_->qw(M + "mlp.experts_gate_up", {Estack * 2 * MD.inter, D.d});
        l.down = w_->qw(M + "mlp.experts_down", {Estack * D.d, MD.inter});
        l.router = w_->bf16(M + "mlp.router", {D.experts + 1, D.d});
        if (shared_separate_) {
            l.shared_gate_up = w_->qw(M + "mlp.shared_expert.gate_up", {2 * MD.inter, D.d});
            l.shared_down = w_->qw(M + "mlp.shared_expert.down_proj.weight", {D.d, MD.inter});
        }
    }
}

void Qwen4ExpModel::make_draft_head_q4(int64_t rows, int64_t G, int threads) {
    const char *fn = "Qwen4ExpModel::make_draft_head_q4";
    STRIX_CHECK(has_mtp_, fn, ": the model has no MTP head - nothing drafts");
    STRIX_CHECK(draft_head_.bits == 0, fn, ": a draft head copy of ", draft_head_.N(), " rows was already made (call once)");
    // The copy is made from draft_lm(): the LM head at world 1; under tensor parallelism the rows rank 0 holds whole.
    const QWeightView &src = draft_lm_;
    STRIX_CHECK(src.bits != 0, fn, ": the LM head isn't loaded");
    const int64_t N = src.N(), K = src.K();
    STRIX_CHECK(N == (dims_.tp_world == 1 ? dims_.vocab : draft_rows()), fn, ": the LM head has ", N, " rows, expected ",
                dims_.tp_world == 1 ? dims_.vocab : draft_rows());
    STRIX_CHECK(rows >= 1 && rows <= N, fn, ": rows ", rows, ", expected 1..", N, " (the LM head's rows)");
    STRIX_CHECK(q4_group_size_supported(G), fn, ": group size ", G, ", expected 32, 64 or 128");
    STRIX_CHECK(K % G == 0, fn, ": the LM head's K = ", K, " is not a multiple of group size ", G);
    STRIX_CHECK(threads >= 1 && threads <= 64, fn, ": threads ", threads, ", expected 1..64");

    if (src.bits == 4) {  // already Q4: the draft reads the loaded head (a row prefix of it)
        STRIX_CHECK(src.q4.G == G, fn, ": the LM head is Q4 with group size ", src.q4.G, ", asked for ", G,
                    " - a Q4 head is used as loaded, not requantized");
        draft_head_ = src;
        draft_head_.q4.N = rows;
        return;
    }
    STRIX_CHECK(src.bits == 8, fn, ": the LM head has ", src.bits, " bits, expected 8 (requantized) or 4 (used ",
                "as loaded)");

    // Read the Q8 rows back (row-major: the first `rows` rows are a prefix of each array).
    const Q8DeviceView &v = src.q8;
    STRIX_CHECK(v.q && v.scale && v.minv && v.K == K && v.G >= 1 && K % v.G == 0, fn, ": the Q8 LM head view is ",
                "incomplete (codes ", (const void *)v.q, ", scales ", (const void *)v.scale, ", mins ",
                (const void *)v.minv, ", K ", v.K, ", G ", v.G, ")");
    Q8Weight q8;
    q8.N = rows, q8.K = K, q8.G = v.G;
    q8.q.resize((size_t)(rows * K));
    q8.scale.resize((size_t)(rows * (K / v.G)));
    q8.minv.resize(q8.scale.size());
    STRIX_HIP_CHECK(hipMemcpy(q8.q.data(), v.q, q8.q.size(), hipMemcpyDeviceToHost), fn, ": read back ", q8.q.size(),
                    " bytes of Q8 codes");
    STRIX_HIP_CHECK(hipMemcpy(q8.scale.data(), v.scale, q8.scale.size() * 2, hipMemcpyDeviceToHost), fn, ": read back ",
                    q8.scale.size(), " Q8 scales");
    STRIX_HIP_CHECK(hipMemcpy(q8.minv.data(), v.minv, q8.minv.size() * 2, hipMemcpyDeviceToHost), fn, ": read back ",
                    q8.minv.size(), " Q8 mins");

    const Q4Weight q4 = quantize_q4_from_q8(q8, G, threads);
    draft_q4_ = std::make_unique<Q4Device>(Q4Device::upload(q4, "lm_head.mtp_draft_q4"));
    draft_q4_bytes_ = q4.bytes();
    draft_head_ = qweight(draft_q4_->view());
    STRIX_CHECK(draft_head_.N() == rows && draft_head_.K() == K, fn, ": the uploaded copy is [", draft_head_.N(), ", ",
                draft_head_.K(), "], expected [", rows, ", ", K, "]");
}

int64_t make_served_mtp_draft_head(Qwen4ExpModel &model, int64_t mtp_vocab) {
    const int64_t vocab = model.dims().vocab;
    STRIX_CHECK(mtp_vocab >= 0 && mtp_vocab <= vocab, "make_served_mtp_draft_head: mtp_vocab ", mtp_vocab, ", expected 0..",
                vocab, " (0 = the whole vocabulary)");
    const int64_t rows = mtp_vocab == 0 ? vocab : mtp_vocab;
    model.make_draft_head_q4(rows, /*G=*/64, /*threads=*/16);
    return rows;
}

const Qwen4ExpModel::Layer &Qwen4ExpModel::layer(int64_t i) const {
    STRIX_CHECK(i >= 0 && i < (int64_t)layers_.size(), "Qwen4ExpModel::layer: ", i, " outside [0, ", layers_.size(), ")");
    return layers_[(size_t)i];
}

const NgramRowSource &Qwen4ExpModel::ngram_rows() const {
    STRIX_CHECK(ngram_ != nullptr, "Qwen4ExpModel: no PLE layer, so no n-gram rows");
    return *ngram_;
}

Qwen4ExpSession::Qwen4ExpSession(const Qwen4ExpModel &model, int64_t capacity, int64_t max_tokens,
                                 PrefillMath prefill_math, bool mtp)
    : m_(model), capacity_(capacity), max_tokens_(max_tokens), prefill_math_(prefill_math), mtp_(mtp) {
    STRIX_CHECK(prefill_math == PrefillMath::F32 || prefill_math == PrefillMath::WmmaBf16,
                "Qwen4ExpSession: unknown prefill math ", (int)prefill_math);
    STRIX_CHECK(!mtp || model.has_mtp(), "Qwen4ExpSession: MTP asked for, but the loaded weights have no MTP head "
                "(no mtp.fc_embedding.weight)");
    const Qwen4ExpDims &D = m_.dims(), &MD = m_.mtp_dims();
    STRIX_CHECK(capacity >= 1 && capacity <= (1ll << 24), "Qwen4ExpSession: capacity = ", capacity, ", expected 1..2^24");
    STRIX_CHECK(capacity <= D.max_positions(), "Qwen4ExpSession: capacity = ", capacity, " is past the model's ",
                D.max_positions(), " positions (", D.trained_positions, " trained x YaRN factor ", D.yarn_factor,
                ") - positions beyond it aren't meaningful; load the model with a larger yarn_factor");
    STRIX_CHECK(max_tokens >= 1 && max_tokens <= capacity && max_tokens <= 16384, "Qwen4ExpSession: max_tokens = ",
                max_tokens, ", expected 1..min(capacity = ", capacity, ", 16384)");
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "Qwen4ExpSession: stream");
    const size_t e = es(m_.act());
    const int64_t M = max_tokens, n4 = D.H * D.d, A = D.top_k + 1;
    cap_blocks_ = capacity / kernels::kQsaBlock + 1;
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string tag = "layer " + std::to_string(i);
        const bool attn = D.is_attention[(size_t)i];
        k_cache_.emplace_back(attn ? DeviceBuffer<uint8_t>((size_t)(capacity * D.hkv * D.hd) * e, "K cache " + tag)
                                   : DeviceBuffer<uint8_t>());
        v_cache_.emplace_back(attn ? DeviceBuffer<uint8_t>((size_t)(capacity * D.hkv * D.hd) * e, "V cache " + tag)
                                   : DeviceBuffer<uint8_t>());
        block_keys_.emplace_back(attn ? DeviceBuffer<uint8_t>((size_t)(cap_blocks_ * D.idx_d) * e, "block keys " + tag)
                                      : DeviceBuffer<uint8_t>());
        for (int k = 0; k < 2; ++k)
            tail_[k].emplace_back(attn ? DeviceBuffer<uint8_t>((size_t)(3 * D.idx_d) * e, "indexer tail " + tag)
                                       : DeviceBuffer<uint8_t>());
        conv_state_.emplace_back(!attn ? DeviceBuffer<uint8_t>((size_t)((kernels::kGdnConvWidth - 1) * D.conv_c) * e,
                                                               "conv state " + tag)
                                       : DeviceBuffer<uint8_t>());
        rec_state_.emplace_back(!attn ? DeviceBuffer<float>((size_t)(D.gv * kHD * kHD), "recurrent state " + tag)
                                      : DeviceBuffer<float>());
    }
    auto act_buf = [&](int64_t per_token, const std::string &name) {
        return DeviceBuffer<uint8_t>((size_t)(M * per_token) * e, name);
    };
    // Buffers sized from both the per-rank dims D and the MTP head's whole dims MD (ST-3; equal at world 1): at least
    // `bytes`, plus a guard (canaries_ok) past the end.
    auto guarded = [&](size_t bytes, const std::string &name) {
        DeviceBuffer<uint8_t> b(bytes + kCanaryBytes, name);
        canaries_.emplace_back(b.get() + bytes, (size_t)kCanaryBytes);
        return b;
    };
    ids_ = DeviceBuffer<int32_t>((size_t)M, "token ids");
    x_ = act_buf(n4, "residual streams");
    u_ = act_buf(D.d, "sublayer input");
    y_ = act_buf(D.d, "sublayer output");
    if (m_.shared_separate()) sh_y_ = act_buf(D.d, "shared expert output");
    // The MTP catch-up projects T rows of the whole head's q|k|v|idx (MD.astride) into proj_ too.
    proj_ = guarded((size_t)(M * std::max({D.gstride, D.astride, mtp ? MD.astride : (int64_t)0})) * e, "projection");
    qkv_ = act_buf(D.conv_c, "gdn q|k|v");
    STRIX_CHECK(D.gz == D.gv * kHD, "Qwen4ExpSession: GDN z width ", D.gz, " != ", D.gv, " value heads x ", kHD,
                " (rmsnorm_gated reads z in place as gv heads of kHD a token)");
    // A draft's row (T = 1) of the whole head: its attention core MD.hq x hd, its experts' A x MD.inter (gate|up 2x).
    core_ = guarded((size_t)std::max(M * std::max(D.gz, D.hq * D.hd), mtp ? MD.hq * MD.hd : (int64_t)0) * e, "mixer core");
    gnorm_ = act_buf(D.gz, "gdn normed");
    gu_ = guarded((size_t)std::max(M * A * 2 * D.inter, mtp ? A * 2 * MD.inter : (int64_t)0) * e, "experts gate|up");
    hh_ = guarded((size_t)std::max(M * A * D.inter, mtp ? A * MD.inter : (int64_t)0) * e, "experts swiglu");
    h_ = DeviceBuffer<float>((size_t)(M * D.r), "hc low-rank");
    w_in_ = DeviceBuffer<float>((size_t)(M * D.H), "hc w_in");
    inv_ = DeviceBuffer<float>((size_t)(M * D.H), "hc inverse rms");
    beta_ = DeviceBuffer<float>((size_t)(M * D.gv), "gdn beta");
    g_ = DeviceBuffer<float>((size_t)(M * D.gv), "gdn g");
    router_logits_ = DeviceBuffer<float>((size_t)(M * (D.experts + 1)), "router logits");
    route_ids_ = DeviceBuffer<int32_t>((size_t)(M * A), "expert ids");
    route_coef_ = DeviceBuffer<float>((size_t)(M * A), "expert coefficients");
    logits_ = DeviceBuffer<float>((size_t)(std::min(M, kMaxLogits) * D.vocab), "logits");
    readback_ = PinnedHostBuffer(kReadbackLogitsAt + logits_.size() * 4, "forward read-back (errors + logits)");
    {
        const int64_t rows = std::min(M, kMaxLogits);
        cand_dev_ = DeviceBuffer<uint8_t>((size_t)rows * (kernels::kLogitCands * sizeof(kernels::LogitCand) + 4),
                                          "logits candidates + NaN flags");
        cand_ws_bytes_ = kernels::logits_topk_workspace_bytes(rows, D.vocab);
        cand_ws_ = DeviceBuffer<uint8_t>(cand_ws_bytes_, "logits_topk workspace");
        STRIX_CHECK(cand_dev_.size() <= logits_.size() * 4, "Qwen4ExpSession: the candidates (", cand_dev_.size(),
                    " bytes) don't fit the logits read-back area (", logits_.size() * 4, ")");
    }
    if (D.ple_layer >= 0) {
        const int64_t S = (D.ple_taps - 1) * D.ple_dil;
        ple_state_ = DeviceBuffer<uint8_t>((size_t)(S * n4) * e, "PLE conv state");
        ple_e_ = act_buf(D.ple_e, "PLE n-gram rows");
        for (int k = 0; k < 2; ++k)
            ple_host_[k] = PinnedHostBuffer((size_t)(M * D.ple_e) * e, "PLE n-gram rows (host) " + std::to_string(k));
        ple_kv_ = act_buf(D.ple_ld, "PLE key|value");
        ple_hist_buf_ = DeviceBuffer<uint8_t>((size_t)((S + M) * n4) * e, "PLE history");
        ple_sigma_ = DeviceBuffer<float>((size_t)(M * D.H), "PLE sigma");
    }
    const int64_t qsa_k = D.qsa_budget / kernels::kQsaBlock;
    const bool qsa = capacity > D.dense_key_limit();
    const bool wmma_attn = prefill_math_ == PrefillMath::WmmaBf16 && m_.act() == Act::BF16;
    for (int64_t T = 1; T <= M; ++T) {
        const kernels::AttentionShape s{T, capacity - T, D.hq, D.hkv, D.hd, D.astride, 2 * D.hd};
        attn_ws_bytes_ = std::max(attn_ws_bytes_, kernels::attention_workspace_bytes(s));
        if (qsa) attn_ws_bytes_ = std::max(attn_ws_bytes_, kernels::attention_gathered_workspace_bytes(s, qsa_k));
        if (wmma_attn && T >= kWmmaMinTokens) {
            // The matrix-unit attention for prefill-sized forwards, dense at any depth below the limit.
            for (int64_t p0 : {(int64_t)0, std::min(capacity, D.dense_key_limit()) - T})
                if (p0 >= 0)
                    attn_ws_bytes_ = std::max(attn_ws_bytes_, kernels::attention_wmma_workspace_bytes(
                                                                  {T, p0, D.hq, D.hkv, D.hd, D.astride, 2 * D.hd}));
            if (qsa) attn_ws_bytes_ = std::max(attn_ws_bytes_, kernels::attention_gathered_wmma_workspace_bytes(s, qsa_k));
        }
    }
    if (mtp) {  // a draft's row through the whole head's attention (T = 1 at any position)
        const kernels::AttentionShape s{1, capacity - 1, MD.hq, MD.hkv, D.hd, MD.astride, 2 * D.hd};
        attn_ws_bytes_ = std::max(attn_ws_bytes_, kernels::attention_workspace_bytes(s));
        if (qsa) attn_ws_bytes_ = std::max(attn_ws_bytes_, kernels::attention_gathered_workspace_bytes(s, qsa_k));
    }
    if (qsa) {
        qsa_scores_ = DeviceBuffer<float>((size_t)(M * cap_blocks_), "QSA block scores");
        qsa_sel_ = DeviceBuffer<int32_t>((size_t)(M * qsa_k), "QSA kept blocks");
        qsa_nsel_ = DeviceBuffer<int32_t>((size_t)M, "QSA kept block counts");
        qsa_topk_ws_bytes_ = kernels::qsa_topk_workspace_bytes(M);
        qsa_topk_ws_ = DeviceBuffer<uint8_t>(qsa_topk_ws_bytes_, "QSA top-k workspace");
        STRIX_HIP_CHECK(hipMemset(qsa_topk_ws_.get(), 0, qsa_topk_ws_bytes_), "zero the QSA top-k workspace");
    }
    attn_err_ = DeviceBuffer<uint32_t>(3, "attention error");
    STRIX_HIP_CHECK(hipMemset(attn_err_.get(), 0, 12), "zero attention error");
    attn_ws_ = DeviceBuffer<float>(attn_ws_bytes_ / 4 + 1 + kCanaryBytes / 4, "attention workspace");
    canaries_.emplace_back(reinterpret_cast<const uint8_t *>(attn_ws_.get() + attn_ws_bytes_ / 4 + 1), (size_t)kCanaryBytes);
    expert_err_ = DeviceBuffer<uint32_t>(3, "expert error");
    if (prefill_math_ == PrefillMath::WmmaBf16 && M >= kWmmaMinTokens) {
        hc_ws_bytes_ = kernels::hc_wmma_workspace_bytes(M, D.H, D.r + D.H);
        hc_ws_ = DeviceBuffer<float>(hc_ws_bytes_ / 4, "hc mix partials");
    }
    if (M >= kGroupedMinTokens) {
        const int64_t Estack = D.experts + (m_.shared_separate() ? 0 : 1);
        group_ws_bytes_ = kernels::moe_route_workspace_bytes(M, A, Estack);
        group_ws_ = DeviceBuffer<int32_t>(group_ws_bytes_ / 4, "expert grouping");
        group_partial_ = DeviceBuffer<float>((size_t)(M * A * D.d), "experts combine partials");
    }
    router_err_ = DeviceBuffer<uint32_t>(3, "router error");
    emb_err_ = DeviceBuffer<kernels::EmbeddingError>(1, "embedding error");
    STRIX_HIP_CHECK(hipMemset(expert_err_.get(), 0, 12), "zero expert error");
    STRIX_HIP_CHECK(hipMemset(router_err_.get(), 0, 12), "zero router error");
    mtp_vocab_ = std::min(D.vocab, m_.draft_rows());
    if (mtp_) {
        mtp_emb_ = act_buf(D.d, "MTP embedding");
        mtp_norm_emb_ = act_buf(D.d, "MTP normed embedding");
        mtp_norm_hid_ = act_buf(n4, "MTP normed streams");
        mtp_proj_emb_ = act_buf(D.d, "MTP fc_embedding output");
        mtp_x_ = act_buf(n4, "MTP streams");
        mtp_prev_ = DeviceBuffer<uint8_t>((size_t)n4 * e, "MTP previous streams");
        k_cache_mtp_ = guarded((size_t)(capacity * MD.hkv * D.hd) * e, "MTP K cache");
        v_cache_mtp_ = guarded((size_t)(capacity * MD.hkv * D.hd) * e, "MTP V cache");
        block_keys_mtp_ = DeviceBuffer<uint8_t>((size_t)(cap_blocks_ * D.idx_d) * e, "MTP block keys");
        for (int k = 0; k < 2; ++k)
            tail_mtp_[k] = DeviceBuffer<uint8_t>((size_t)(3 * D.idx_d) * e, "MTP indexer tail " + std::to_string(k));
        // mtp_x_'s twin: a chain step swaps the two, so the step before's output is kept without a copy.
        mtp_chain_prev_ = act_buf(n4, "MTP chain previous streams");
        for (int k = 0; k < 2; ++k)
            mtp_chain_tail_[k] = DeviceBuffer<uint8_t>((size_t)(3 * D.idx_d) * e, "MTP chain indexer tail " + std::to_string(k));
        std::vector<float> ones((size_t)(M * D.H), 1.0f);
        mtp_ones_ = DeviceBuffer<float>(ones.size(), "MTP ones");
        STRIX_HIP_CHECK(hipMemcpy(mtp_ones_.get(), ones.data(), ones.size() * 4, hipMemcpyHostToDevice), "MTP ones");
    }
    for (const auto &[at, n] : canaries_)
        STRIX_HIP_CHECK(hipMemset(const_cast<uint8_t *>(at), kCanaryByte, n), "Qwen4ExpSession: guard words");
    STRIX_HIP_CHECK(hipDeviceSynchronize(), "Qwen4ExpSession: guard words");
    // Tensor parallelism: FP32 partials by default (ST-N2: TP4 perplexity within 0.5% of one node;
    // BF16 partials gave +0.71% AR / +0.55% MTP, FP32 +0.15% / +0.20%). The communicator's slots must hold FP32 rows.
    if (D.tp_world > 1 && m_.act() == Act::BF16) {
        set_tp_f32_mixer(true);
        set_tp_f32_moe(true);
    }
    reset();
}

uint64_t Qwen4ExpSession::state_hash() const {
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::state_hash");
    const Qwen4ExpDims &D = m_.dims();
    const size_t e = es(m_.act());
    std::vector<uint64_t> parts;
    const int64_t head[2] = {pos_, verify_pending_ ? 1 : 0};
    parts.push_back(strix_hash64(head, sizeof head));
    parts.push_back(strix_hash64(&ple_hist_, sizeof ple_hist_));
    std::vector<uint8_t> h;
    auto dev = [&](const void *p, size_t n) {
        h.resize(n);
        if (n) STRIX_HIP_CHECK(hipMemcpy(h.data(), p, n, hipMemcpyDeviceToHost), "Qwen4ExpSession::state_hash");
        parts.push_back(strix_hash64(h.data(), n));
    };
    dev(ple_state_.get(), ple_state_.size());
    const int64_t b = pos_ / kernels::kQsaBlock - 1;
    for (size_t l = 0; l < conv_state_.size(); ++l) {
        if (!D.is_attention[l]) continue;
        dev(tail_[tail_cur_][l].get(), (size_t)((pos_ % kernels::kQsaBlock) * D.idx_d) * e);
        if (b >= 0) {  // block b's keys: idx_d / 8 strided rows of 8 (chunk-major, kernels/qsa.hpp)
            const size_t row = 8 * e, pitch = (size_t)(cap_blocks_ * 8) * e, rows = (size_t)(D.idx_d / 8);
            h.resize(row * rows);
            STRIX_HIP_CHECK(hipMemcpy2D(h.data(), row, block_keys_[l].get() + (size_t)(b * 8) * e, pitch, row, rows,
                                        hipMemcpyDeviceToHost),
                            "Qwen4ExpSession::state_hash: block keys");
            parts.push_back(strix_hash64(h.data(), h.size()));
        }
    }
    return strix_hash64(parts.data(), parts.size() * 8);
}

bool Qwen4ExpSession::canaries_ok() const {
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::canaries_ok");
    std::vector<uint8_t> h;
    for (const auto &[at, n] : canaries_) {
        h.resize(n);
        STRIX_HIP_CHECK(hipMemcpy(h.data(), at, n, hipMemcpyDeviceToHost), "Qwen4ExpSession::canaries_ok");
        for (uint8_t b : h)
            if (b != kCanaryByte) return false;
    }
    return true;
}

Qwen4ExpSession::~Qwen4ExpSession() {
    drop_ple_pending();
    if (stream_) {
        (void)hipStreamSynchronize(stream_);  // no upload may still read a pinned buffer when it's freed
        (void)hipStreamDestroy(stream_);
    }
    if (mask_uploaded_) (void)hipEventDestroy(mask_uploaded_);
}

Qwen4ExpSession::PleWorker::PleWorker() : thread_([this] { loop(); }) {}

Qwen4ExpSession::PleWorker::~PleWorker() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();  // a job in flight finishes first
}

std::future<void> Qwen4ExpSession::PleWorker::submit_alone(std::function<void()> job) {
    STRIX_CHECK(job != nullptr, "Qwen4ExpSession::PleWorker::submit_alone: empty job");
    std::packaged_task<void()> task(std::move(job));
    std::future<void> f = task.get_future();
    {
        // One at a time: normally the forward that submitted the last job waited for it; after a forward that failed
        // midway it may still be running - and it writes the same pinned buffer - so wait for it.
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return jobs_.empty() && !busy_; });
        jobs_.push_back(std::move(task));
    }
    cv_.notify_all();
    return f;
}

bool Qwen4ExpSession::PleWorker::post(std::function<void()> job, size_t max_queued) {
    STRIX_CHECK(job != nullptr, "Qwen4ExpSession::PleWorker::post: empty job");
    STRIX_CHECK(max_queued >= 1, "Qwen4ExpSession::PleWorker::post: max_queued = 0");
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (jobs_.size() >= max_queued) return false;
        jobs_.emplace_back(std::move(job));
    }
    cv_.notify_all();
    return true;
}

void Qwen4ExpSession::PleWorker::loop() {
    for (;;) {
        std::packaged_task<void()> task;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [&] { return stop_ || !jobs_.empty(); });
            if (jobs_.empty()) return;  // stop_, nothing left
            task = std::move(jobs_.front()), busy_ = true;
            jobs_.pop_front();
        }
        task();  // an exception goes into the future
        {
            std::lock_guard<std::mutex> lock(mu_);
            busy_ = false;
        }
        cv_.notify_all();  // a submit_alone waiting for this job
    }
}

Qwen4ExpSession::PleStats Qwen4ExpSession::ple_stats() const {
    PleStats s;
    s.gathers = ple_gathers_.load(), s.waits = ple_waits_.load();
    s.gather_seconds = (double)ple_gather_ns_.load() * 1e-9, s.wait_seconds = (double)ple_wait_ns_.load() * 1e-9;
    s.wait_max_seconds = (double)ple_wait_max_ns_.load() * 1e-9;
    s.prefetches = ple_prefetches_.load(), s.prefetch_skipped = ple_prefetch_skipped_.load();
    s.prefetch_failed = ple_prefetch_failed_.load(), s.prefetch_seconds = (double)ple_prefetch_ns_.load() * 1e-9;
    return s;
}

void Qwen4ExpSession::prefetch_ple(const std::vector<int32_t> &ids, int64_t first) {
    const int64_t T = (int64_t)ids.size();
    STRIX_CHECK(T >= 1 && T <= max_tokens_, "Qwen4ExpSession::prefetch_ple: ", T, " ids, expected 1..", max_tokens_);
    STRIX_CHECK(first >= 0 && first < T, "Qwen4ExpSession::prefetch_ple: first = ", first, ", expected 0..", T - 1);
    if (m_.dims().ple_layer < 0) return;
    // Rows of every id (the history runs through them), then the ones asked for. While a verify awaits its outcome
    // the ids start where the verify did (a rejection's kept prefix + the corrected token).
    std::vector<int64_t> tok(ids.begin(), ids.end()), rows((size_t)(T * 16));
    PleHistory h = verify_pending_ ? verify_hist_ : ple_hist_;
    ple_ngram_ids(m_.ngram_rows().hash_params(), tok.data(), T, h, rows.data());
    rows.erase(rows.begin(), rows.begin() + first * 16);
    // A prefetch that doesn't keep up is worth nothing: past a few queued, skip (the gather reads the rows itself).
    constexpr size_t kMaxQueued = 8;
    const bool queued = ple_prefetcher_.post(
        [this, rows = std::move(rows)] {
            const auto t0 = std::chrono::steady_clock::now();
            try {
                m_.ngram_rows().prefetch(rows.data(), (int64_t)rows.size());
            } catch (const std::exception &ex) {
                // Not the forward's failure: the gather that needs these rows reads them and reports its own.
                if (ple_prefetch_failed_++ < 10)
                    std::fprintf(stderr, "Qwen4ExpSession::prefetch_ple: prefetch failed (the gather retries): %s\n",
                                 ex.what());
            }
            ++ple_prefetches_;
            ple_prefetch_ns_ +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        },
        kMaxQueued);
    if (!queued) ++ple_prefetch_skipped_;
}

// Row ids [T, 16] -> ple_host_[buf] as the activation dtype: on the persistent worker (small forwards) or an async
// thread (prefill chunks, lookahead). The caller guarantees nothing reads the buffer (no upload from it pending)
// until the returned future is ready.
std::future<void> Qwen4ExpSession::start_ple_gather(std::vector<int64_t> rows, int64_t T, int buf, PleRun run) {
    const int64_t pe = m_.dims().ple_e;
    STRIX_CHECK(buf == 0 || buf == 1, "Qwen4ExpSession: PLE host buffer ", buf);
    STRIX_CHECK(T >= 1 && T <= max_tokens_ && (int64_t)rows.size() == T * 16, "Qwen4ExpSession: PLE gather of ", T,
                " tokens with ", rows.size(), " row ids, expected 1..", max_tokens_, " tokens x 16");
    STRIX_CHECK(ple_host_[buf].size() >= (size_t)(T * pe) * es(m_.act()), "Qwen4ExpSession: PLE host buffer ", buf,
                " holds ", ple_host_[buf].size(), " bytes, needs ", (size_t)(T * pe) * es(m_.act()));
    auto job = [this, rows = std::move(rows), T, pe, buf] {
        const auto t0 = std::chrono::steady_clock::now();
        const NgramRowSource &ng = m_.ngram_rows();
        if (m_.act() == Act::BF16) {
            ng.gather(rows.data(), (int64_t)rows.size(), static_cast<uint16_t *>(ple_host_[buf].get()));
        } else {
            std::vector<uint16_t> bits((size_t)(T * pe));
            ng.gather(rows.data(), (int64_t)rows.size(), bits.data());
            float *f = static_cast<float *>(ple_host_[buf].get());
            for (size_t k = 0; k < bits.size(); ++k) f[k] = bf16_to_f32(bits[k]);
        }
        ++ple_gathers_;
        ple_gather_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    };
    if (run == PleRun::Worker) return ple_worker_.submit_alone(std::move(job));
    return std::async(std::launch::async, std::move(job));
}

void Qwen4ExpSession::drop_ple_pending() {
    if (ple_pending_.active && ple_pending_.done.valid()) {
        try {
            ple_pending_.done.get();
        } catch (...) {  // a hint's gather: its failure only matters to a forward that uses it
        }
    }
    ple_pending_ = PleGather{};
}

void Qwen4ExpSession::set_lookahead(std::vector<int32_t> next_ids) {
    STRIX_CHECK((int64_t)next_ids.size() <= max_tokens_, "Qwen4ExpSession::set_lookahead: ", next_ids.size(),
                " ids, expected 0..", max_tokens_);
    ple_lookahead_ = std::move(next_ids);
}

void Qwen4ExpSession::reset() {
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::reset");
    pos_ = 0, tail_cur_ = 0, mtp_tail_cur_ = 0, broken_ = false, mtp_chain_step_ = -1, verify_pending_ = false;  // GDN states reset on the first call (reset_state); KV overwritten
    imported_at_ = -1;
    if (mtp_)  // position 0's MTP row pairs its token with no trunk state: zeros
        STRIX_HIP_CHECK(hipMemset(mtp_prev_.get(), 0, mtp_prev_.size()), "Qwen4ExpSession: zero MTP previous streams");
    ple_lookahead_.clear();  // a pending gather stays: the next forward's ids / history decide whether it's used
    rewind_to_.push_back(0);
    if (m_.dims().ple_layer >= 0) {
        STRIX_HIP_CHECK(hipMemset(ple_state_.get(), 0, ple_state_.size()), "Qwen4ExpSession: zero PLE state");
        ple_hist_ = PleHistory::fresh(m_.ngram_rows().hash_params());
    }
}

Qwen4ExpSnapshot Qwen4ExpSession::make_snapshot() const {
    const Qwen4ExpDims &D = m_.dims();
    Qwen4ExpSnapshot s;
    for (int64_t i = 0; i < D.layers; ++i) {
        const std::string tag = "snapshot layer " + std::to_string(i);
        const size_t l = (size_t)i;
        s.conv_.emplace_back(conv_state_[l].size() ? DeviceBuffer<uint8_t>(conv_state_[l].size(), "conv state " + tag)
                                                    : DeviceBuffer<uint8_t>());
        s.rec_.emplace_back(rec_state_[l].size() ? DeviceBuffer<float>(rec_state_[l].size(), "recurrent state " + tag)
                                                  : DeviceBuffer<float>());
        s.tail_.emplace_back(tail_[0][l].size() ? DeviceBuffer<uint8_t>(tail_[0][l].size(), "indexer tail " + tag)
                                                 : DeviceBuffer<uint8_t>());
    }
    if (ple_state_.size()) s.ple_state_ = DeviceBuffer<uint8_t>(ple_state_.size(), "snapshot PLE conv state");
    if (mtp_) {
        s.mtp_tail_ = DeviceBuffer<uint8_t>(tail_mtp_[0].size(), "snapshot MTP indexer tail");
        s.mtp_prev_ = DeviceBuffer<uint8_t>(mtp_prev_.size(), "snapshot MTP previous streams");
    }
    return s;
}

namespace {
template <typename T>
void copy_state(const DeviceBuffer<T> &from, const DeviceBuffer<T> &to, hipStream_t stream, const char *what) {
    STRIX_CHECK(from.size() == to.size(), "Qwen4ExpSession snapshot: ", what, " has ", from.size(), " elements, expected ",
                to.size(), " (a snapshot from another session?)");
    if (from.size())
        STRIX_HIP_CHECK(hipMemcpyAsync(to.get(), from.get(), from.size() * sizeof(T), hipMemcpyDeviceToDevice, stream),
                        "Qwen4ExpSession snapshot: copying ", what);
}
}  // namespace

void Qwen4ExpSession::save(Qwen4ExpSnapshot &s) {
    STRIX_CHECK(!broken_, "Qwen4ExpSession::save: an earlier call failed midway; the state is not worth saving");
    STRIX_CHECK(!verify_pending_, "Qwen4ExpSession::save: a forward_verify awaits keep_verify / drop_verify");
    STRIX_CHECK(s.conv_.size() == conv_state_.size(), "Qwen4ExpSession::save: snapshot has ", s.conv_.size(),
                " layers, the session ", conv_state_.size(), " (make it with make_snapshot)");
    for (size_t l = 0; l < conv_state_.size(); ++l) {
        copy_state(conv_state_[l], s.conv_[l], stream_, "conv state");
        copy_state(rec_state_[l], s.rec_[l], stream_, "recurrent state");
        copy_state(tail_[tail_cur_][l], s.tail_[l], stream_, "indexer tail");
    }
    copy_state(ple_state_, s.ple_state_, stream_, "PLE conv state");
    copy_state(tail_mtp_[mtp_tail_cur_], s.mtp_tail_, stream_, "MTP indexer tail");
    copy_state(mtp_prev_, s.mtp_prev_, stream_, "MTP previous streams");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::save");
    s.ple_hist_ = ple_hist_;
    s.pos_ = pos_;
    s.epoch_ = rewind_to_.size() - 1;
}

bool Qwen4ExpSession::can_restore(const Qwen4ExpSnapshot &s) const {
    if (s.pos_ < 0 || s.conv_.size() != conv_state_.size() || s.epoch_ >= rewind_to_.size()) return false;
    for (size_t e = s.epoch_ + 1; e < rewind_to_.size(); ++e)
        if (rewind_to_[e] < s.pos_) return false;
    // Every rewind since was to >= s.pos_, and forwards only write at or past the position they start from, so
    // KV and block keys below s.pos_ are the ones the snapshot's state was built with.
    return true;
}

void Qwen4ExpSession::restore(const Qwen4ExpSnapshot &s) {
    STRIX_CHECK(s.pos_ >= 0, "Qwen4ExpSession::restore: the snapshot was never saved");
    verify_pending_ = false;  // a restore replaces all state a pending verify could have left
    STRIX_CHECK(can_restore(s), "Qwen4ExpSession::restore: the session rewound below the snapshot's position ", s.pos_,
                " since it was saved, so its KV cache no longer matches");
    for (size_t l = 0; l < conv_state_.size(); ++l) {
        copy_state(s.conv_[l], conv_state_[l], stream_, "conv state");
        copy_state(s.rec_[l], rec_state_[l], stream_, "recurrent state");
        copy_state(s.tail_[l], tail_[tail_cur_][l], stream_, "indexer tail");
    }
    copy_state(s.ple_state_, ple_state_, stream_, "PLE conv state");
    copy_state(s.mtp_tail_, tail_mtp_[mtp_tail_cur_], stream_, "MTP indexer tail");
    copy_state(s.mtp_prev_, mtp_prev_, stream_, "MTP previous streams");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::restore");
    ple_hist_ = s.ple_hist_;
    pos_ = s.pos_;
    broken_ = false, mtp_chain_step_ = -1;
    rewind_to_.push_back(s.pos_);
}

size_t Qwen4ExpSession::state_bytes(int64_t n, int64_t from) const {
    STRIX_CHECK(n >= 0 && n <= capacity_, "Qwen4ExpSession::state_bytes: n = ", n, ", expected 0..", capacity_);
    STRIX_CHECK(from >= 0 && from <= n, "Qwen4ExpSession::state_bytes: from = ", from, ", expected 0..n = ", n);
    const Qwen4ExpDims &D = m_.dims();
    const size_t e = es(m_.act());
    // The rows [from, n) and the blocks [from / 4, n / 4) (see export_state).
    const size_t kv = 2 * (size_t)((n - from) * D.hkv * D.hd) * e,
                 keys = (size_t)((n / kernels::kQsaBlock - from / kernels::kQsaBlock) * D.idx_d) * e;
    size_t b = sizeof(PleHistory);
    for (size_t l = 0; l < conv_state_.size(); ++l) {
        if (D.is_attention[l])
            b += kv + keys + tail_[0][l].size();
        else
            b += conv_state_[l].size() + rec_state_[l].size() * sizeof(float);
    }
    b += ple_state_.size();
    if (mtp_) b += 2 * (size_t)((n - from) * m_.mtp_dims().hkv * D.hd) * e + keys + tail_mtp_[0].size() + mtp_prev_.size();
    return b;
}

namespace {
// Copies between one device region and the next `bytes` of a host cursor, in either direction.
// The block keys of blocks [b0, b1): the cache is chunk-major (kernels/qsa.hpp: element d of block b at
// ((d / 8) * cap_blocks + b) * 8 + d % 8), so they're D / 8 strided rows of (b1 - b0) * 8 elements - packed on the
// host as [chunk][block][8], which doesn't depend on the session's capacity.
void block_keys_copy(uint8_t *&cur, const void *dev, int64_t b0, int64_t b1, int64_t cap_blocks, int64_t D, size_t e,
                     bool to_host, hipStream_t stream) {
    STRIX_CHECK(b0 >= 0 && b0 <= b1 && b1 <= cap_blocks, "Qwen4ExpSession block keys copy: blocks [", b0, ", ", b1,
                ") of ", cap_blocks);
    if (b1 == b0) return;
    const size_t row = (size_t)((b1 - b0) * 8) * e, pitch = (size_t)(cap_blocks * 8) * e, rows = (size_t)(D / 8);
    uint8_t *d = static_cast<uint8_t *>(const_cast<void *>(dev)) + (size_t)(b0 * 8) * e;
    if (to_host)
        STRIX_HIP_CHECK(hipMemcpy2DAsync(cur, row, d, pitch, row, rows, hipMemcpyDeviceToHost, stream),
                        "Qwen4ExpSession export: block keys");
    else
        STRIX_HIP_CHECK(hipMemcpy2DAsync(d, pitch, cur, row, row, rows, hipMemcpyHostToDevice, stream),
                        "Qwen4ExpSession import: block keys");
    cur += row * rows;
}

void host_copy(uint8_t *&cur, const void *dev, size_t bytes, bool to_host, hipStream_t stream, const char *what) {
    if (!bytes) return;
    if (to_host)
        STRIX_HIP_CHECK(hipMemcpyAsync(cur, dev, bytes, hipMemcpyDeviceToHost, stream), "Qwen4ExpSession export: ", what);
    else
        STRIX_HIP_CHECK(hipMemcpyAsync(const_cast<void *>(dev), cur, bytes, hipMemcpyHostToDevice, stream),
                        "Qwen4ExpSession import: ", what);
    cur += bytes;
}
}  // namespace

void Qwen4ExpSession::export_state(const Qwen4ExpSnapshot &s, uint8_t *out, size_t bytes, int64_t from) {
    STRIX_CHECK(out != nullptr, "Qwen4ExpSession::export_state: null output");
    STRIX_CHECK(can_restore(s), "Qwen4ExpSession::export_state: the snapshot (position ", s.pos(),
                ") is not valid for this session's KV cache any more");
    const int64_t n = s.pos();
    STRIX_CHECK(from >= 0 && from < n, "Qwen4ExpSession::export_state: from = ", from, ", expected 0..", n - 1,
                " (the snapshot's position - 1)");
    STRIX_CHECK(bytes == state_bytes(n, from), "Qwen4ExpSession::export_state: buffer ", bytes, " bytes, the state at ",
                n, from > 0 ? " (a delta from " + std::to_string(from) + ")" : std::string(), " is ", state_bytes(n, from));
    // Rows [from, n), blocks [from / 4, n / 4).
    const int64_t b0 = from / kernels::kQsaBlock, b1 = n / kernels::kQsaBlock;
    const Qwen4ExpDims &D = m_.dims();
    const size_t e = es(m_.act());
    uint8_t *cur = out;
    std::memcpy(cur, &s.ple_hist_, sizeof(PleHistory));
    cur += sizeof(PleHistory);
    // An indexer tail holds 3 rows but only n % 4 are state; the rest is whatever an earlier forward left there
    // (after an MTP rejection: the rejected tokens' keys). Exported as zeros, so a state's bytes depend only on its
    // sequence. The zeroing waits for the copies.
    const size_t tail_valid = (size_t)((n % kernels::kQsaBlock) * D.idx_d) * e;
    std::vector<std::pair<uint8_t *, size_t>> tail_unused;
    for (size_t l = 0; l < conv_state_.size(); ++l) {
        if (D.is_attention[l]) {
            const size_t row = (size_t)(D.hkv * D.hd) * e, kv = (size_t)(n - from) * row;
            host_copy(cur, k_cache_[l].get() + (size_t)from * row, kv, true, stream_, "K cache");
            host_copy(cur, v_cache_[l].get() + (size_t)from * row, kv, true, stream_, "V cache");
            block_keys_copy(cur, block_keys_[l].get(), b0, b1, cap_blocks_, D.idx_d, e, true, stream_);
            STRIX_CHECK(s.tail_[l].size() >= tail_valid, "Qwen4ExpSession::export_state: indexer tail of ",
                        s.tail_[l].size(), " bytes, ", tail_valid, " valid");
            tail_unused.emplace_back(cur + tail_valid, s.tail_[l].size() - tail_valid);
            host_copy(cur, s.tail_[l].get(), s.tail_[l].size(), true, stream_, "indexer tail");
        } else {
            host_copy(cur, s.conv_[l].get(), s.conv_[l].size(), true, stream_, "conv state");
            host_copy(cur, s.rec_[l].get(), s.rec_[l].size() * sizeof(float), true, stream_, "recurrent state");
        }
    }
    host_copy(cur, s.ple_state_.get(), s.ple_state_.size(), true, stream_, "PLE conv state");
    if (mtp_) {
        const size_t row = (size_t)(m_.mtp_dims().hkv * D.hd) * e, kv = (size_t)(n - from) * row;
        host_copy(cur, k_cache_mtp_.get() + (size_t)from * row, kv, true, stream_, "MTP K cache");
        host_copy(cur, v_cache_mtp_.get() + (size_t)from * row, kv, true, stream_, "MTP V cache");
        block_keys_copy(cur, block_keys_mtp_.get(), b0, b1, cap_blocks_, D.idx_d, e, true, stream_);
        STRIX_CHECK(s.mtp_tail_.size() >= tail_valid, "Qwen4ExpSession::export_state: MTP indexer tail of ",
                    s.mtp_tail_.size(), " bytes, ", tail_valid, " valid");
        tail_unused.emplace_back(cur + tail_valid, s.mtp_tail_.size() - tail_valid);
        host_copy(cur, s.mtp_tail_.get(), s.mtp_tail_.size(), true, stream_, "MTP indexer tail");
        host_copy(cur, s.mtp_prev_.get(), s.mtp_prev_.size(), true, stream_, "MTP previous streams");
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::export_state");
    for (const auto &[at, len] : tail_unused) std::memset(at, 0, len);
    STRIX_CHECK((size_t)(cur - out) == bytes, "Qwen4ExpSession::export_state: wrote ", cur - out, " of ", bytes, " bytes");
}

void Qwen4ExpSession::import_state(const uint8_t *in, size_t bytes, int64_t n, int64_t from) {
    STRIX_CHECK(in != nullptr, "Qwen4ExpSession::import_state: null input");
    STRIX_CHECK(n >= 1 && n < capacity_, "Qwen4ExpSession::import_state: n = ", n, ", expected 1..", capacity_ - 1);
    STRIX_CHECK(from >= 0 && from < n, "Qwen4ExpSession::import_state: from = ", from, ", expected 0..", n - 1);
    STRIX_CHECK(bytes == state_bytes(n, from), "Qwen4ExpSession::import_state: ", bytes, " bytes, the state at ", n,
                from > 0 ? " (a delta from " + std::to_string(from) + ")" : std::string(), " is ", state_bytes(n, from),
                " (from another model, layout or activation dtype?)");
    // A delta lands on the base it was cut from: the rows below `from` must be that base's, imported just before.
    STRIX_CHECK(from == 0 || (pos_ == from && imported_at_ == from), "Qwen4ExpSession::import_state: a delta from ",
                from, " needs its base imported just before (position ", pos_, ", last import ended at ", imported_at_,
                ")");
    const int64_t b0 = from / kernels::kQsaBlock, b1 = n / kernels::kQsaBlock;
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::import_state");
    const Qwen4ExpDims &D = m_.dims();
    const size_t e = es(m_.act());
    uint8_t *cur = const_cast<uint8_t *>(in);
    PleHistory hist;
    std::memcpy(&hist, cur, sizeof(PleHistory));
    cur += sizeof(PleHistory);
    for (size_t l = 0; l < conv_state_.size(); ++l) {
        if (D.is_attention[l]) {
            const size_t row = (size_t)(D.hkv * D.hd) * e, kv = (size_t)(n - from) * row;
            host_copy(cur, k_cache_[l].get() + (size_t)from * row, kv, false, stream_, "K cache");
            host_copy(cur, v_cache_[l].get() + (size_t)from * row, kv, false, stream_, "V cache");
            block_keys_copy(cur, block_keys_[l].get(), b0, b1, cap_blocks_, D.idx_d, e, false, stream_);
            host_copy(cur, tail_[tail_cur_][l].get(), tail_[tail_cur_][l].size(), false, stream_, "indexer tail");
        } else {
            host_copy(cur, conv_state_[l].get(), conv_state_[l].size(), false, stream_, "conv state");
            host_copy(cur, rec_state_[l].get(), rec_state_[l].size() * sizeof(float), false, stream_, "recurrent state");
        }
    }
    host_copy(cur, ple_state_.get(), ple_state_.size(), false, stream_, "PLE conv state");
    if (mtp_) {
        const size_t row = (size_t)(m_.mtp_dims().hkv * D.hd) * e, kv = (size_t)(n - from) * row;
        host_copy(cur, k_cache_mtp_.get() + (size_t)from * row, kv, false, stream_, "MTP K cache");
        host_copy(cur, v_cache_mtp_.get() + (size_t)from * row, kv, false, stream_, "MTP V cache");
        block_keys_copy(cur, block_keys_mtp_.get(), b0, b1, cap_blocks_, D.idx_d, e, false, stream_);
        host_copy(cur, tail_mtp_[mtp_tail_cur_].get(), tail_mtp_[mtp_tail_cur_].size(), false, stream_,
                  "MTP indexer tail");
        host_copy(cur, mtp_prev_.get(), mtp_prev_.size(), false, stream_, "MTP previous streams");
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::import_state");
    ple_hist_ = hist;
    STRIX_CHECK((size_t)(cur - in) == bytes, "Qwen4ExpSession::import_state: read ", cur - in, " of ", bytes, " bytes");
    pos_ = n, broken_ = false, mtp_chain_step_ = -1, verify_pending_ = false;
    imported_at_ = n;
    rewind_to_.push_back(0);  // the KV cache below n was rewritten (or completed): no older snapshot matches it
}

// The end of a forward: the error words and the logits come back in one go - async copies into pinned memory on
// stream_ behind the forward, one synchronize, then the checks on the host copies. It used to be a synchronize and
// five blocking hipMemcpy calls (~40 us each on gfx1151, HIP API trace dea5523-apitrace), per forward - and the MTP
// cycle runs several forwards per token. The pinned area is reused only after this synchronize (the gfx1151
// pinned-reuse hazard).
std::vector<float> Qwen4ExpSession::read_back(size_t n_floats, const char *what, int64_t n_cand_rows) {
    STRIX_CHECK(n_floats * 4 <= readback_.size() - kReadbackLogitsAt, "Qwen4ExpSession::read_back: ", n_floats,
                " logits, the pinned area holds ", (readback_.size() - kReadbackLogitsAt) / 4);
    STRIX_CHECK(n_cand_rows >= 0 && (n_cand_rows == 0 || n_floats == 0), "Qwen4ExpSession::read_back: ", n_cand_rows,
                " candidate rows with ", n_floats, " logits (one or the other)");
    const size_t cand_bytes = (size_t)n_cand_rows * kernels::kLogitCands * sizeof(kernels::LogitCand);
    STRIX_CHECK(cand_bytes + (size_t)n_cand_rows * 4 <= cand_dev_.size(), "Qwen4ExpSession::read_back: ", n_cand_rows,
                " candidate rows, the device area holds ", cand_dev_.size(), " bytes");
    auto *host = static_cast<uint8_t *>(readback_.get());
    auto *words = reinterpret_cast<uint32_t *>(host);  // router [0, 3), experts [3, 6), attention [6, 9)
    auto *emb = reinterpret_cast<kernels::EmbeddingError *>(host + kReadbackEmbAt);
    auto d2h = [&](void *to, const void *from, size_t bytes, const char *name) {
        STRIX_HIP_CHECK(hipMemcpyAsync(to, from, bytes, hipMemcpyDeviceToHost, stream_), "Qwen4ExpSession::read_back",
                        what, ": ", name);
    };
    d2h(words, router_err_.get(), 12, "router error");
    d2h(words + 3, expert_err_.get(), 12, "expert error");
    d2h(words + 6, attn_err_.get(), 12, "attention error");
    d2h(emb, emb_err_.get(), sizeof(kernels::EmbeddingError), "embedding error");
    if (n_floats) d2h(host + kReadbackLogitsAt, logits_.get(), n_floats * 4, "logits");
    if (n_cand_rows) d2h(host + kReadbackLogitsAt, cand_dev_.get(), cand_bytes + (size_t)n_cand_rows * 4, "candidates");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::read_back", what, ": end of the forward");
    const std::string tag = std::string("Qwen4ExpSession") + what;
    kernels::report_router_error(words, (tag + " router").c_str());
    kernels::report_expert_error(words + 3, (tag + " experts").c_str());
    kernels::report_attention_error(words + 6, (tag + " QSA attention").c_str());
    kernels::report_embedding_error(*emb, m_.dims().vocab);
    if (n_cand_rows) {
        const auto *c = reinterpret_cast<const kernels::LogitCand *>(host + kReadbackLogitsAt);
        const auto *f = reinterpret_cast<const uint32_t *>(host + kReadbackLogitsAt + cand_bytes);
        cand_host_.rows = n_cand_rows;
        cand_host_.cand.assign(c, c + n_cand_rows * kernels::kLogitCands);
        cand_host_.nan.assign(f, f + n_cand_rows);
    }
    const float *l = reinterpret_cast<const float *>(host + kReadbackLogitsAt);
    return std::vector<float>(l, l + n_floats);
}

void Qwen4ExpSession::set_tp_f32_mixer(bool on) {
    STRIX_CHECK(!on || m_.act() == Act::BF16, "Qwen4ExpSession::set_tp_f32_mixer: needs BF16 activations");
    if (on && y32_.size() == 0) y32_ = DeviceBuffer<float>((size_t)(max_tokens_ * m_.dims().d), "mixer FP32 partials");
    tp_f32_mixer_ = on;
}

void Qwen4ExpSession::set_tp_f32_moe(bool on) {
    STRIX_CHECK(!on || m_.act() == Act::BF16, "Qwen4ExpSession::set_tp_f32_moe: needs BF16 activations");
    const size_t n = (size_t)(max_tokens_ * m_.dims().d);
    if (on && y32_.size() == 0) y32_ = DeviceBuffer<float>(n, "FP32 partials");
    if (on && m_.shared_separate() && sh_y32_.size() == 0) sh_y32_ = DeviceBuffer<float>(n, "shared expert FP32 output");
    tp_f32_moe_ = on;
}

void Qwen4ExpSession::want_candidates(int64_t n_valid, const uint32_t *masks, int64_t mask_rows, int64_t mask_words) {
    const char *fn = "Qwen4ExpSession::want_candidates";
    const Qwen4ExpDims &D = m_.dims();
    const int64_t vocab = D.vocab;
    STRIX_CHECK(n_valid >= 1 && n_valid <= vocab, fn, ": n_valid = ", n_valid, ", expected 1..", vocab);
    // Tensor parallelism: n_valid counts the whole vocabulary; this rank scores its rows [rank * lm_rows, ..) of it.
    const int64_t local = std::min(D.lm_rows, n_valid - (int64_t)D.tp_rank * D.lm_rows);
    STRIX_CHECK(local >= 1, fn, ": n_valid = ", n_valid, " leaves rank ", D.tp_rank, " (rows from ", D.tp_rank * D.lm_rows,
                ") no valid id");
    if (masks == nullptr) {
        STRIX_CHECK(mask_rows == 0 && mask_words == 0, fn, ": mask_rows = ", mask_rows, ", mask_words = ", mask_words,
                    " without masks (pass 0 for both)");
    } else {
        // A rank holds lm_rows of the vocabulary under TP; the masks index all of it (structured output: world 1).
        STRIX_CHECK(D.tp_world == 1, fn, ": structured-output masks under tensor parallelism are not supported");
        STRIX_CHECK(mask_rows >= 1 && mask_rows <= kMaxLogits, fn, ": mask_rows = ", mask_rows, ", expected 1..",
                    kMaxLogits);
        STRIX_CHECK(mask_words >= (vocab + 31) / 32, fn, ": mask_words = ", mask_words, " covers ", mask_words * 32,
                    " ids, the vocabulary has ", vocab);
        const size_t bytes = (size_t)(mask_rows * mask_words) * 4;
        if (mask_host_.size() < bytes) {
            STRIX_CHECK(!mask_upload_recorded_ || hipEventSynchronize(mask_uploaded_) == hipSuccess, fn,
                        ": waiting for the last mask upload before replacing its buffer");
            mask_upload_recorded_ = false;
            const size_t capacity = (size_t)(kMaxLogits * mask_words) * 4;
            mask_host_ = PinnedHostBuffer(capacity, "structured-output masks (host)");
            mask_dev_ = DeviceBuffer<uint32_t>(capacity / 4, "structured-output masks");
            if (mask_uploaded_ == nullptr)
                STRIX_HIP_CHECK(hipEventCreateWithFlags(&mask_uploaded_, hipEventDisableTiming), fn, ": mask upload event");
        }
        // The gate: the previous upload from this pinned buffer has completed before it is written again.
        if (mask_upload_recorded_) STRIX_HIP_CHECK(hipEventSynchronize(mask_uploaded_), fn, ": last mask upload");
        std::memcpy(mask_host_.get(), masks, bytes);
    }
    mask_rows_ = masks == nullptr ? 0 : mask_rows;
    mask_words_ = masks == nullptr ? 0 : mask_words;
    cand_n_valid_ = local;
}

std::vector<float> Qwen4ExpSession::forward(const std::vector<int32_t> &ids, int64_t n_logits, const Qwen4ExpProbe &probe) {
    STRIX_CHECK(!broken_, "Qwen4ExpSession::forward: an earlier call failed midway; reset() the session first");
    STRIX_CHECK(!verify_pending_, "Qwen4ExpSession::forward: a forward_verify awaits keep_verify / drop_verify");
    const Qwen4ExpDims &D = m_.dims();
    const Act act = m_.act();
    const size_t e = es(act);
    const bool sep = m_.shared_separate();
    const bool wmma = prefill_math_ == PrefillMath::WmmaBf16;
    STRIX_TRACE_RANGE(std::string(gdn_to_spare_ ? "verify" : ids.size() == 1 ? "decode" : "prefill") + " T=" +
                      std::to_string(ids.size()));
    STRIX_TRACE_STAGE(stage);
    STRIX_TRACE_SET(stage, "embed");
    // Dense Q4 / Q8 projection: on the matrix units for large enough forwards when switched on - and when the kernel
    // takes its K: a multiple of the K step (kernels/wmma_gemm.hpp kWmmaKC = 64), or of 32 for Q4 / Q8 (PF-6: a TP-4
    // rank's shared expert down, K = 160, runs the K-tail instantiation).
    auto lin = [&](const void *x, const QWeightView &w, void *y, int64_t M, Act out_act) {
        const bool wmma_k = w.K() % 64 == 0 || ((w.bits == 4 || w.bits == 8) && w.K() % 32 == 0);
        if (wmma && M >= kWmmaMinTokens && wmma_k) kernels::linear_qw_wmma(x, w, y, M, act, out_act, stream_);
        else kernels::linear_qw(x, w, y, M, act, out_act, stream_);
    };
    const kernels::MoeMath moe_math = wmma ? kernels::MoeMath::WmmaBf16 : kernels::MoeMath::F32;
    // Expert slots per token: top-k + the shared expert's when it's stacked.
    const int64_t T = (int64_t)ids.size(), n4 = D.H * D.d, A = D.top_k + (sep ? 0 : 1), hd = D.hd;
    STRIX_CHECK(T >= 1 && T <= max_tokens_, "Qwen4ExpSession::forward: ", T, " tokens, expected 1..", max_tokens_);
    STRIX_CHECK(pos_ + T <= capacity_, "Qwen4ExpSession::forward: positions [", pos_, ", ", pos_ + T,
                ") exceed the capacity ", capacity_);
    STRIX_CHECK(n_logits >= 0 && n_logits <= std::min(T, kMaxLogits), "Qwen4ExpSession::forward: n_logits = ",
                n_logits, ", expected 0..", std::min(T, kMaxLogits));
    const int64_t cand_n_valid = cand_n_valid_;  // want_candidates() covers this one forward, whatever happens
    const bool tp = D.tp_world > 1;
    auto tp_exchange = [&](void *buf, int64_t elems, int kind, void *out = nullptr) {
        if (!tp) return;
        ++exchanges_;
        if (exchange_) exchange_(buf, elems, kind, stream_, out);
    };
    const bool f32_mixer = tp && tp_f32_mixer_ && T <= tp_f32_max_tokens_;
    const bool f32_moe = tp && tp_f32_moe_ && T <= tp_f32_max_tokens_;  // y32_ is free again by the MoE: the mixer's exchange consumed it
    const int64_t mask_rows = mask_rows_, mask_words = mask_words_;
    cand_n_valid_ = 0, mask_rows_ = 0, mask_words_ = 0;
    STRIX_CHECK(mask_rows == 0 || (cand_n_valid > 0 && mask_rows == n_logits), "Qwen4ExpSession::forward: ",
                mask_rows, " structured-output mask rows for a forward of ", n_logits, " logits rows");
    broken_ = true;
    const bool fresh = pos_ == 0;
    auto at = [&](const DeviceBuffer<uint8_t> &b, int64_t elem) { return b.get() + (size_t)elem * e; };
    const ProbeType pt = act == Act::F32 ? ProbeType::F32 : ProbeType::BF16;
    auto show = [&](const std::string &name, const void *dev, int64_t cols, ProbeType type) {
        if (!probe) return;
        STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "before probe '", name, "'");
        probe(name, dev, T, cols, type);
    };
    // inv_ready: inv_ already holds X's inverse RMS (the preceding injection was hc_inject_inv). xb: the streams
    // to mix, x_ unless given (the MTP layer's mtp_x_).
    auto hc_mix = [&](const Qwen4ExpModel::Hc &hc, bool inject, bool inv_ready = false, uint8_t *xb = nullptr) {
        float *w_in = inject ? w_in_.get() : nullptr;
        uint8_t *X = xb ? xb : x_.get();
        STRIX_CHECK((hc.down.bits == 4 || hc.down.bits == 8) && (hc.up.bits == 4 || hc.up.bits == 8),
                    "Qwen4ExpSession: HC weights not loaded (down Q", hc.down.bits, ", up Q", hc.up.bits, ")");
        if (wmma && T >= kWmmaMinTokens) {  // the prefill path on the matrix units (kernels/hc_wmma)
            float *ws = hc_ws_.get();
            if (hc.down.bits == 8)
                kernels::hc_mix_down_wmma(X, hc.down.q8, T, D.H, D.d, D.r, inject, D.eps, h_.get(), w_in,
                                          inv_.get(), ws, hc_ws_bytes_, act, stream_, inv_ready);
            else
                kernels::hc_mix_down_wmma(X, hc.down.q4, T, D.H, D.d, D.r, inject, D.eps, h_.get(), w_in,
                                          inv_.get(), ws, hc_ws_bytes_, act, stream_, inv_ready);
            if (hc.up.bits == 8)
                kernels::hc_mix_up_wmma(h_.get(), hc.up.q8, X, hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act,
                                        stream_);
            else
                kernels::hc_mix_up_wmma(h_.get(), hc.up.q4, X, hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act,
                                        stream_);
            return;
        }
        if (hc.down.bits == 8)
            kernels::hc_mix_down(X, hc.down.q8, T, D.H, D.d, D.r, inject, D.eps, h_.get(), w_in, inv_.get(), act,
                                 stream_);
        else
            kernels::hc_mix_down(X, hc.down.q4, T, D.H, D.d, D.r, inject, D.eps, h_.get(), w_in, inv_.get(), act,
                                 stream_);
        if (hc.up.bits == 8)
            kernels::hc_mix_up(h_.get(), hc.up.q8, X, hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act, stream_);
        else
            kernels::hc_mix_up(h_.get(), hc.up.q4, X, hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act, stream_);
    };

    // PLE rows for this forward: the ids advance the n-gram history now; the rows come from set_lookahead's gather
    // when it was for exactly these ids and history, else a gather starts now on a worker - overlapping the
    // embedding and the layers before the PLE layer. Joined there (a gather failure fails the forward).
    int ple_buf = 0;
    std::future<void> ple_ready;
    std::vector<int32_t> lookahead = std::move(ple_lookahead_);
    ple_lookahead_.clear();
    if (D.ple_layer >= 0) {
        const PleHistory before = ple_hist_;
        std::vector<int64_t> tok(ids.begin(), ids.end()), rows((size_t)(T * 16));
        ple_ngram_ids(m_.ngram_rows().hash_params(), tok.data(), T, ple_hist_, rows.data());
        if (ple_pending_.active && ple_pending_.ids == ids && ple_pending_.hist.prev == before.prev) {
            ple_buf = ple_pending_.buf, ple_ready = std::move(ple_pending_.done);
            ple_pending_ = PleGather{};
        } else {
            drop_ple_pending();  // a stale hint: let it finish before its buffer is reused
            // Small forwards (decode, MTP verify): on the persistent worker, overlapping the embedding and layer 0.
            ple_ready = start_ple_gather(std::move(rows), T, ple_buf,
                                         T >= kWmmaMinTokens ? PleRun::Async : PleRun::Worker);
        }
    }

    // Injection; on the WMMA mix path fused with the next mix's inverse RMS (hc_inject_inv). Returns whether inv_
    // is ready for that mix.
    const bool fuse_inv = wmma && T >= kWmmaMinTokens;
    auto inject_y = [&]() {
        if (fuse_inv) kernels::hc_inject_inv(x_.get(), w_in_.get(), y_.get(), T, D.H, D.d, D.eps, inv_.get(), act, stream_);
        else kernels::hc_inject(x_.get(), w_in_.get(), y_.get(), T, D.H, D.d, act, stream_);
        return fuse_inv;
    };
    bool inv_ready = false;  // X changed since inv_ was computed (the embedding)

    // Embedding, written straight into all 4 streams (one launch).
    STRIX_HIP_CHECK(hipMemcpyAsync(ids_.get(), ids.data(), (size_t)T * 4, hipMemcpyHostToDevice, stream_),
                    "uploading ", T, " token ids");
    kernels::reset_embedding_error(emb_err_.get(), stream_);
    if (m_.is_embed_q8()) {
        kernels::embedding_lookup(ids_.get(), T, m_.embed_q8(), x_.get(), act, emb_err_.get(), stream_, D.H, n4);
    } else {
        kernels::embedding_lookup(ids_.get(), T, m_.embed(), D.vocab, D.d, x_.get(), act, emb_err_.get(), stream_, D.H, n4);
    }
    show("embed_streams", x_.get(), n4, pt);  // [T, H * d]: the embedding in each of the H streams

    for (int64_t i = 0; i < D.layers; ++i) {
        const Qwen4ExpModel::Layer &l = m_.layer(i);
        const std::string L = "L" + std::to_string(i) + ".";
        if (i == D.ple_layer) STRIX_TRACE_SET(stage, "L" + std::to_string(i) + " ple");
        if (i == D.ple_layer) {
            // n-gram rows (gathered on the host since the forward started), key|value projection, gate + conv into X.
            show(L + "ple_in", x_.get(), n4, pt);
            {
                const auto t0 = std::chrono::steady_clock::now();
                ple_ready.get();
                const int64_t ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
                ++ple_waits_, ple_wait_ns_ += ns;
                for (int64_t m = ple_wait_max_ns_.load(); ns > m && !ple_wait_max_ns_.compare_exchange_weak(m, ns);) {
                }
            }
            STRIX_HIP_CHECK(hipMemcpyAsync(ple_e_.get(), ple_host_[ple_buf].get(), (size_t)(T * D.ple_e) * e,
                                           hipMemcpyHostToDevice, stream_),
                            "uploading PLE rows");
            // The next forward's rows (set_lookahead) into the other buffer, during this forward's GPU work: that
            // buffer's last upload was in an earlier forward, which ended with a stream sync.
            if (!lookahead.empty()) {
                std::vector<int64_t> tok(lookahead.begin(), lookahead.end()), rows(lookahead.size() * 16);
                PleHistory h = ple_hist_;
                ple_ngram_ids(m_.ngram_rows().hash_params(), tok.data(), (int64_t)tok.size(), h, rows.data());
                ple_pending_.buf = 1 - ple_buf;
                ple_pending_.done = start_ple_gather(std::move(rows), (int64_t)tok.size(), ple_pending_.buf,
                                                     PleRun::Async);
                ple_pending_.ids = std::move(lookahead), ple_pending_.hist = ple_hist_, ple_pending_.active = true;
            }
            lin(ple_e_.get(), l.ple_kv, ple_kv_.get(), T, act);
            const kernels::PleShape sh{T, D.H, D.d, D.ple_taps, D.ple_dil, D.ple_ld};
            kernels::ple_gate(ple_kv_.get(), x_.get(), l.ple_norm_key, l.ple_norm_query, l.ple_norm_conv,
                              ple_state_.get(), sh, D.eps, ple_hist_buf_.get(), ple_sigma_.get(), act, stream_);
            kernels::ple_conv(ple_hist_buf_.get(), ple_sigma_.get(), ple_kv_.get(), l.ple_conv, sh, x_.get(),
                              ple_state_.get(), act, stream_);
            show(L + "ple_out", x_.get(), n4, pt);
        }

        if (i == D.ple_layer) inv_ready = false;  // the PLE changed X after the last injection
        STRIX_TRACE_SET(stage, "L" + std::to_string(i) + (D.is_attention[(size_t)i] ? " attn" : " gdn"));
        hc_mix(l.hc_attn, true, inv_ready);
        show(L + "attn_in", u_.get(), D.d, pt);
        show(L + "attn_w_in", w_in_.get(), D.H, ProbeType::F32);
        if (!D.is_attention[(size_t)i]) {
            // A verify writes its in_proj rows straight into the layer's verify_in_ buffer (keep_verify_prefix may
            // replay them) instead of proj_ and a copy.
            uint8_t *gp = gdn_to_spare_ ? verify_in_[(size_t)i].get() : proj_.get();
            lin(u_.get(), l.in_proj, gp, T, act);
            kernels::gdn_front(gp, {T, D.gk, kHD, D.gv, kHD, D.gstride, D.conv_c + D.gz, D.conv_c + D.gz + D.gv},
                               l.conv_w, conv_state_[(size_t)i].get(),
                               (gdn_to_spare_ ? conv_spare_ : conv_state_)[(size_t)i].get(), fresh, l.A_log,
                               l.dt_bias, qkv_.get(), beta_.get(), g_.get(), act, stream_);
            // Prefill-sized WMMA forwards: the chunked delta rule on the matrix units; decode / MTP verify per token.
            kernels::delta_rule(qkv_.get(), beta_.get(), g_.get(), {T, D.gk, D.gv, D.conv_c}, rec_state_[(size_t)i].get(),
                                (gdn_to_spare_ ? rec_spare_ : rec_state_)[(size_t)i].get(), fresh, core_.get(), act,
                                stream_,
                                wmma && T >= kWmmaMinTokens ? kernels::DeltaRulePath::ChunkedWmma
                                                            : kernels::DeltaRulePath::Auto);
            // z (gv heads of kHD a token) read in place from the in_proj rows.
            kernels::rmsnorm_gated(core_.get(), gp + (size_t)D.conv_c * e, l.gdn_norm, gnorm_.get(), T * D.gv, kHD, D.eps,
                                   kernels::GateAct::Sigmoid, act, stream_, D.gv, D.gstride);
            show(L + "gdn_core", gnorm_.get(), D.gz, pt);
            if (f32_mixer) lin(gnorm_.get(), l.out_proj, y32_.get(), T, Act::F32);
            else lin(gnorm_.get(), l.out_proj, y_.get(), T, act);
        } else {
            uint8_t *kc = k_cache_[(size_t)i].get(), *vc = v_cache_[(size_t)i].get();
            lin(u_.get(), l.qkv, proj_.get(), T, act);
            // One launch: q normed + roped in place, k into the K cache, v copied into the V cache, past the dense
            // limit the indexer queries (normed + roped in place), and in a verify the raw indexer keys copied into
            // the layer's verify_in_ (keep_verify_prefix may replay them) - kernels/attention.hpp qk_norm_rope_jobs.
            const bool qsa_layer = pos_ + T > D.dense_key_limit();
            kernels::NormRopeJob nr[5] = {
                {proj_.get(), D.astride, 2 * hd, proj_.get(), D.astride, 2 * hd, D.hq, hd, l.q_norm},
                {at(proj_, D.hq * 2 * hd), D.astride, hd, kc + (size_t)(pos_ * D.hkv * hd) * e, D.hkv * hd, hd, D.hkv, hd,
                 l.k_norm},
                {at(proj_, D.hq * 2 * hd + D.hkv * hd), D.astride, hd, vc + (size_t)(pos_ * D.hkv * hd) * e, D.hkv * hd, hd,
                 D.hkv, hd, nullptr},
                {at(proj_, D.idx_col), D.astride, D.idx_d, at(proj_, D.idx_col), D.astride, D.idx_d, D.idx_h, D.idx_d,
                 l.idx_q_norm}};
            int njobs = 3;
            if (qsa_layer) ++njobs;
            else nr[3] = {};
            if (gdn_to_spare_)
                nr[njobs++] = {at(proj_, D.idx_col + D.idx_h * D.idx_d), D.astride, D.idx_d, verify_in_[(size_t)i].get(),
                               D.idx_d, D.idx_d, 1, D.idx_d, nullptr};
            kernels::qk_norm_rope_jobs(nr, njobs, T, D.eps, D.rot, pos_, m_.inv_freq(), act, stream_, m_.rope_scale());
            // Indexer block keys (every completed block, this call's included) - kept current for when selection
            // starts to bite; the queries and scores are only needed past the dense limit.
            kernels::qsa_block_keys(at(proj_, D.idx_col + D.idx_h * D.idx_d), D.astride, T, pos_,
                                    pos_ % kernels::kQsaBlock ? tail_[tail_cur_][(size_t)i].get() : nullptr,
                                    tail_[1 - tail_cur_][(size_t)i].get(), D.idx_d, l.idx_k_norm, D.eps, m_.inv_freq(),
                                    D.rot, block_keys_[(size_t)i].get(), cap_blocks_, act, stream_, m_.rope_scale());
            const kernels::AttentionShape as{T, pos_, D.hq, D.hkv, hd, D.astride, 2 * hd};
            const float scale = 1.0f / std::sqrt((float)hd);
            // Prefill-sized forwards with BF16 activations: the attention on the matrix units (kernels/attention.hpp).
            const bool wmma_attn = wmma && act == Act::BF16 && T >= kWmmaMinTokens;
            if (!qsa_layer) {
                if (wmma_attn)
                    kernels::attention_wmma(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, scale, core_.get(),
                                            attn_ws_.get(), attn_ws_bytes_, act, stream_);
                else
                    kernels::attention(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, scale, core_.get(),
                                       attn_ws_.get(), attn_ws_bytes_, act, stream_);
            } else {
                // QSA: indexer queries (normed + roped in place), block scores, the top budget / 4 blocks, gathered
                // attention. Queries that still see every key keep all their blocks - the dense set.
                const int64_t qsa_k = D.qsa_budget / kernels::kQsaBlock;
                void *iq = at(proj_, D.idx_col);  // normed + roped by the qk_norm_rope_jobs above
                kernels::qsa_scores(iq, D.astride, D.idx_d, T, pos_, D.idx_h, D.idx_d, block_keys_[(size_t)i].get(),
                                    cap_blocks_, qsa_scores_.get(), cap_blocks_, act, stream_);
                kernels::qsa_topk(qsa_scores_.get(), cap_blocks_, T, pos_, qsa_k, qsa_sel_.get(), qsa_nsel_.get(),
                                  qsa_topk_ws_.get(), qsa_topk_ws_bytes_, stream_);
                show(L + "qsa_sel", qsa_sel_.get(), qsa_k, ProbeType::I32);
                show(L + "qsa_nsel", qsa_nsel_.get(), 1, ProbeType::I32);
                if (wmma_attn)
                    kernels::attention_gathered_wmma(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, qsa_sel_.get(),
                                                     qsa_nsel_.get(), qsa_k, scale, core_.get(), attn_ws_.get(),
                                                     attn_ws_bytes_, attn_err_.get(), act, stream_);
                else
                    kernels::attention_gathered(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, qsa_sel_.get(),
                                                qsa_nsel_.get(), qsa_k, scale, core_.get(), attn_ws_.get(),
                                                attn_ws_bytes_, attn_err_.get(), act, stream_);
            }
            show(L + "attn_core", core_.get(), D.hq * D.hd, pt);
            if (f32_mixer) lin(core_.get(), l.o_proj, y32_.get(), T, Act::F32);
            else lin(core_.get(), l.o_proj, y_.get(), T, act);
        }
        // the row-parallel o_proj / out_proj partials
        if (f32_mixer) tp_exchange(y32_.get(), T * D.d, 3, y_.get());
        else tp_exchange(y_.get(), T * D.d, 0);
        show(L + "mixer", y_.get(), D.d, pt);
        inv_ready = inject_y();

        STRIX_TRACE_SET(stage, "L" + std::to_string(i) + " moe");
        hc_mix(l.hc_mlp, true, inv_ready);
        show(L + "mlp_in", u_.get(), D.d, pt);
        show(L + "mlp_w_in", w_in_.get(), D.H, ProbeType::F32);
        // Router: logits FP32, 512 routed + the shared gate in one [T, 513] output; prefill-sized
        // forwards on the matrix units (exact BF16 products, FP32 sums - only the summation order differs).
        if (wmma && T >= kWmmaMinTokens)
            kernels::linear_bf16w_wmma(u_.get(), l.router, router_logits_.get(), T, D.experts + 1, D.d, act, Act::F32,
                                       stream_);
        else
            kernels::linear_bf16w(u_.get(), l.router, router_logits_.get(), T, D.experts + 1, D.d, act, Act::F32,
                                  stream_);
        show(L + "router_logits", router_logits_.get(), D.experts + 1, ProbeType::F32);
        kernels::moe_router(router_logits_.get(), D.experts + 1, T, D.experts, D.top_k,
                            sep ? nullptr : router_logits_.get() + D.experts, D.experts + 1, (int32_t)D.experts,
                            route_ids_.get(), route_coef_.get(), Act::F32, router_err_.get(), stream_);
        show(L + "router_ids", route_ids_.get(), A, ProbeType::I32);
        show(L + "router_coef", route_coef_.get(), A, ProbeType::F32);
        const int64_t Estack = D.experts + (sep ? 0 : 1);
        if (T >= kGroupedMinTokens) {
            // Q4 experts on the matrix units take F16; Q5 stays on BF16 scaled codes
            // until its kernel moves over.
            // The WMMA expert kernels step K by kWmmaKC (64); Q4 experts also take a K tail of 32 (PF-6: a TP-4 rank's
            // intermediate 640 / 4 = 160 at G = 32). Anything else runs the FP32 math.
            const bool wmma_k = l.down.K() % 64 == 0 || (l.gate_up.bits == 4 && l.down.bits == 4 && l.down.K() % 32 == 0);
            const kernels::MoeMath em = !wmma_k ? kernels::MoeMath::F32
                                        : moe_math == kernels::MoeMath::WmmaBf16 && l.gate_up.bits == 4 && l.down.bits == 4
                                            ? kernels::MoeMath::WmmaF16
                                            : moe_math;
            kernels::moe_group_routes(route_ids_.get(), T, A, Estack, group_ws_.get(), group_ws_bytes_,
                                      expert_err_.get(), stream_);
            kernels::linear_qw_experts_gather_grouped(u_.get(), l.gate_up, Estack, group_ws_.get(), group_ws_bytes_, T,
                                                      A, gu_.get(), act, em, stream_);
            kernels::swiglu(gu_.get(), hh_.get(), T * A, D.inter, act, stream_);
            if (f32_moe)
                kernels::linear_qw_experts_combine_grouped_f32(hh_.get(), l.down, Estack, route_ids_.get(),
                                                               route_coef_.get(), group_ws_.get(), group_ws_bytes_,
                                                               group_partial_.get(), T, A, y32_.get(), em, stream_);
            else
                kernels::linear_qw_experts_combine_grouped(hh_.get(), l.down, Estack, route_ids_.get(),
                                                           route_coef_.get(), group_ws_.get(), group_ws_bytes_,
                                                           group_partial_.get(), T, A, y_.get(), act, em, stream_);
        } else {
            kernels::linear_qw_experts_gather_swiglu(u_.get(), l.gate_up, Estack, route_ids_.get(), T, A, hh_.get(),
                                                     act, expert_err_.get(), stream_);
            if (f32_moe)
                kernels::linear_qw_experts_combine_f32(hh_.get(), l.down, Estack, route_ids_.get(), route_coef_.get(),
                                                       T, A, y32_.get(), expert_err_.get(), stream_);
            else
                kernels::linear_qw_experts_combine(hh_.get(), l.down, Estack, route_ids_.get(), route_coef_.get(), T,
                                                   A, y_.get(), act, expert_err_.get(), stream_);
        }
        show(L + "experts_h", hh_.get(), A * D.inter, pt);  // the routed experts' down input, T rows of A slots
        if (sep) {  // the shared expert as dense projections, then y += sigmoid(its gate logit) * its output
            lin(u_.get(), l.shared_gate_up, gu_.get(), T, act);
            kernels::swiglu(gu_.get(), hh_.get(), T, D.inter, act, stream_);
            show(L + "shared_h", hh_.get(), D.inter, pt);
            if (f32_moe) lin(hh_.get(), l.shared_down, sh_y32_.get(), T, Act::F32);
            else lin(hh_.get(), l.shared_down, sh_y_.get(), T, act);
        }
        if (sep && !fuse_inv && !tp) {  // decode / verify: the gated add and the injection in one launch (bit-identical)
            kernels::moe_shared_add_inject(y_.get(), sh_y_.get(), router_logits_.get() + D.experts, D.experts + 1, T,
                                           D.d, (uint32_t)D.experts, x_.get(), w_in_.get(), D.H, act,
                                           router_err_.get(), stream_);
            show(L + "moe", y_.get(), D.d, pt);
            inv_ready = false;
        } else {
            if (sep && f32_moe)
                kernels::moe_shared_add(y32_.get(), sh_y32_.get(), router_logits_.get() + D.experts, D.experts + 1,
                                        T, D.d, (uint32_t)D.experts, Act::F32, router_err_.get(), stream_);
            else if (sep)
                kernels::moe_shared_add(y_.get(), sh_y_.get(), router_logits_.get() + D.experts, D.experts + 1, T,
                                        D.d, (uint32_t)D.experts, act, router_err_.get(), stream_);
            // the experts' down partials (+ the gated shared expert's)
            if (f32_moe) tp_exchange(y32_.get(), T * D.d, 3, y_.get());
            else tp_exchange(y_.get(), T * D.d, 1);
            show(L + "moe", y_.get(), D.d, pt);
            inv_ready = inject_y();
        }
        show(L + "out", x_.get(), n4, pt);
    }
    // Indexer tails: every attention layer wrote tail_out = the other buffer this call.
    tail_cur_ = 1 - tail_cur_;

    STRIX_TRACE_SET(stage, "head");
    hc_mix(m_.final_mix(), false, inv_ready);
    show("final_mixed", u_.get(), D.d, pt);
    if (n_logits > 0)
        lin(at(u_, (T - n_logits) * D.d), m_.lm_head(), logits_.get(), n_logits, Act::F32);
    const bool cands = cand_n_valid > 0 && n_logits > 0;
    if (cands) {
        auto *cd = reinterpret_cast<kernels::LogitCand *>(cand_dev_.get());
        if (mask_rows > 0) {
            STRIX_HIP_CHECK(hipMemcpyAsync(mask_dev_.get(), mask_host_.get(), (size_t)(mask_rows * mask_words) * 4,
                                           hipMemcpyHostToDevice, stream_),
                            "Qwen4ExpSession::forward: structured-output masks upload");
            STRIX_HIP_CHECK(hipEventRecord(mask_uploaded_, stream_), "Qwen4ExpSession::forward: mask upload event");
            mask_upload_recorded_ = true;
        }
        kernels::logits_topk(logits_.get(), n_logits, D.lm_rows, cand_n_valid, cd,
                             reinterpret_cast<uint32_t *>(cd + n_logits * kernels::kLogitCands), cand_ws_.get(),
                             cand_ws_bytes_, stream_, mask_rows > 0 ? mask_dev_.get() : nullptr,
                             mask_rows > 0 ? mask_words : 0);
    }

    // MTP: the layer's K / V (and indexer keys) for these positions, each from its token and the trunk's streams
    // after the position before - what a later draft attends to. The rest of the
    // layer only matters for a draft's own row, so it's skipped here.
    // Tensor parallelism: the vocabulary shares' candidates gathered and merged on every rank (cand_dev_ in place, global
    // ids). Full logits rows (no candidates) stay this rank's share [n_logits, lm_rows]: the caller gathers them.
    if (cands) tp_exchange(cand_dev_.get(), n_logits, 2);
    if (mtp_ && mtp_catchup_) mtp_catchup(T);
    std::vector<float> logits = cands ? read_back(0, "", n_logits) : read_back((size_t)(n_logits * D.lm_rows), "");
    pos_ += T;
    broken_ = false;
    mtp_chain_step_ = -1;
    return logits;
}

void Qwen4ExpSession::mtp_catchup(int64_t T) {
    const Qwen4ExpDims &D = m_.dims(), &MD = m_.mtp_dims();
    const Act act = m_.act();
    const size_t e = es(act);
    const bool wmma = prefill_math_ == PrefillMath::WmmaBf16;
    const int64_t n4 = D.H * D.d, hd = D.hd;
    auto at = [&](const DeviceBuffer<uint8_t> &b, int64_t elem) { return b.get() + (size_t)elem * e; };
    const Qwen4ExpModel::Layer &ml = m_.mtp().layer;
    mtp_input(T, mtp_prev_.get());
    // The layer's attention mix over mtp_x_ (inject: w_in_ for nothing after; the trunk's hc mix, kernels as forward()).
    const Qwen4ExpModel::Hc &hc = ml.hc_attn;
    STRIX_CHECK((hc.down.bits == 4 || hc.down.bits == 8) && (hc.up.bits == 4 || hc.up.bits == 8),
                "Qwen4ExpSession: HC weights not loaded (down Q", hc.down.bits, ", up Q", hc.up.bits, ")");
    uint8_t *X = mtp_x_.get();
    if (wmma && T >= kWmmaMinTokens) {
        float *ws = hc_ws_.get();
        if (hc.down.bits == 8)
            kernels::hc_mix_down_wmma(X, hc.down.q8, T, D.H, D.d, D.r, true, D.eps, h_.get(), w_in_.get(), inv_.get(), ws,
                                      hc_ws_bytes_, act, stream_, false);
        else
            kernels::hc_mix_down_wmma(X, hc.down.q4, T, D.H, D.d, D.r, true, D.eps, h_.get(), w_in_.get(), inv_.get(), ws,
                                      hc_ws_bytes_, act, stream_, false);
        if (hc.up.bits == 8)
            kernels::hc_mix_up_wmma(h_.get(), hc.up.q8, X, hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act, stream_);
        else
            kernels::hc_mix_up_wmma(h_.get(), hc.up.q4, X, hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act, stream_);
        kernels::linear_qw_wmma(u_.get(), ml.qkv, proj_.get(), T, act, act, stream_);
    } else {
        if (hc.down.bits == 8)
            kernels::hc_mix_down(X, hc.down.q8, T, D.H, D.d, D.r, true, D.eps, h_.get(), w_in_.get(), inv_.get(), act, stream_);
        else
            kernels::hc_mix_down(X, hc.down.q4, T, D.H, D.d, D.r, true, D.eps, h_.get(), w_in_.get(), inv_.get(), act, stream_);
        if (hc.up.bits == 8)
            kernels::hc_mix_up(h_.get(), hc.up.q8, X, hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act, stream_);
        else
            kernels::hc_mix_up(h_.get(), hc.up.q4, X, hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act, stream_);
        kernels::linear_qw(u_.get(), ml.qkv, proj_.get(), T, act, act, stream_);
    }
    // The whole head's widths (MD; the trunk's D are this rank's share under tensor parallelism).
    const kernels::NormRopeJob nr[2] = {
        {at(proj_, MD.hq * 2 * hd), MD.astride, hd, k_cache_mtp_.get() + (size_t)(pos_ * MD.hkv * hd) * e, MD.hkv * hd, hd,
         MD.hkv, hd, ml.k_norm},
        {at(proj_, MD.hq * 2 * hd + MD.hkv * hd), MD.astride, hd, v_cache_mtp_.get() + (size_t)(pos_ * MD.hkv * hd) * e,
         MD.hkv * hd, hd, MD.hkv, hd, nullptr}};
    kernels::qk_norm_rope_jobs(nr, 2, T, D.eps, D.rot, pos_, m_.inv_freq(), act, stream_, m_.rope_scale());  // k, v into the MTP cache
    kernels::qsa_block_keys(at(proj_, MD.idx_col + D.idx_h * D.idx_d), MD.astride, T, pos_,
                            pos_ % kernels::kQsaBlock ? tail_mtp_[mtp_tail_cur_].get() : nullptr,
                            tail_mtp_[1 - mtp_tail_cur_].get(), D.idx_d, ml.idx_k_norm, D.eps, m_.inv_freq(),
                            D.rot, block_keys_mtp_.get(), cap_blocks_, act, stream_, m_.rope_scale());
    mtp_tail_cur_ = 1 - mtp_tail_cur_;
    STRIX_HIP_CHECK(hipMemcpyAsync(mtp_prev_.get(), at(x_, (T - 1) * n4), (size_t)n4 * e, hipMemcpyDeviceToDevice,
                                   stream_),
                    "MTP previous streams");
}

void Qwen4ExpSession::debug_mtp_feed(const std::vector<int32_t> &ids, const void *x_host, size_t bytes) {
    STRIX_CHECK(mtp_, "Qwen4ExpSession::debug_mtp_feed: MTP is off");
    STRIX_CHECK(!broken_ && !verify_pending_, "Qwen4ExpSession::debug_mtp_feed: broken or a verify pending");
    const Qwen4ExpDims &D = m_.dims();
    const size_t e = es(m_.act());
    const int64_t T = (int64_t)ids.size(), n4 = D.H * D.d;
    STRIX_CHECK(T >= 1 && T <= max_tokens_ && pos_ + T <= capacity_, "Qwen4ExpSession::debug_mtp_feed: ", T,
                " tokens at ", pos_);
    STRIX_CHECK(x_host && bytes == (size_t)(T * n4) * e, "Qwen4ExpSession::debug_mtp_feed: ", bytes, " bytes of streams, expected ",
                (size_t)(T * n4) * e);
    broken_ = true;
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::debug_mtp_feed");
    STRIX_HIP_CHECK(hipMemcpy(ids_.get(), ids.data(), (size_t)T * 4, hipMemcpyHostToDevice), "debug_mtp_feed: ids");
    STRIX_HIP_CHECK(hipMemcpy(x_.get(), x_host, bytes, hipMemcpyHostToDevice), "debug_mtp_feed: streams");
    kernels::reset_embedding_error(emb_err_.get(), stream_);
    mtp_catchup(T);
    read_back(0, " debug_mtp_feed");
    pos_ += T;
    mtp_chain_step_ = -1;
    broken_ = false;
}

std::vector<float> Qwen4ExpSession::forward_verify(const std::vector<int32_t> &ids, int64_t n_logits,
                                                    const Qwen4ExpProbe &probe) {
    STRIX_CHECK(!verify_pending_, "Qwen4ExpSession::forward_verify: the last one awaits keep_verify / drop_verify");
    STRIX_CHECK(!broken_, "Qwen4ExpSession::forward_verify: an earlier call failed midway; reset() the session first");
    if (conv_spare_.empty()) {
        for (size_t l = 0; l < conv_state_.size(); ++l) {
            const std::string tag = "spare layer " + std::to_string(l);
            conv_spare_.emplace_back(conv_state_[l].size() ? DeviceBuffer<uint8_t>(conv_state_[l].size(), "conv state " + tag)
                                                           : DeviceBuffer<uint8_t>());
            rec_spare_.emplace_back(rec_state_[l].size() ? DeviceBuffer<float>(rec_state_[l].size(), "recurrent state " + tag)
                                                         : DeviceBuffer<float>());
        }
        if (ple_state_.size()) ple_state_kept_ = DeviceBuffer<uint8_t>(ple_state_.size(), "kept PLE conv state");
        if (mtp_) mtp_prev_kept_ = DeviceBuffer<uint8_t>(mtp_prev_.size(), "kept MTP previous streams");
        const Qwen4ExpDims &D = m_.dims();
        const size_t e = es(m_.act());
        for (size_t l = 0; l < conv_state_.size(); ++l)
            verify_in_.emplace_back((size_t)(kMaxLogits * (D.is_attention[l] ? D.idx_d : D.gstride)) * e,
                                    "verify rows, layer " + std::to_string(l));
    }
    // What the forward changes in place (small): kept for drop_verify. The GDN states go to the spare set.
    copy_state(ple_state_, ple_state_kept_, stream_, "PLE conv state");
    copy_state(mtp_prev_, mtp_prev_kept_, stream_, "MTP previous streams");
    STRIX_CHECK((int64_t)ids.size() <= kMaxLogits, "Qwen4ExpSession::forward_verify: ", ids.size(), " tokens, expected <= ",
                kMaxLogits, " (the saved rows for keep_verify_prefix)");
    verify_pos_ = pos_, verify_tail_cur_ = tail_cur_, verify_mtp_tail_cur_ = mtp_tail_cur_, verify_hist_ = ple_hist_;
    verify_ids_ = ids;
    gdn_to_spare_ = true;
    try {
        std::vector<float> logits = forward(ids, n_logits, probe);
        verify_caught_up_ = mtp_ && mtp_catchup_;
        gdn_to_spare_ = false;
        verify_pending_ = true;
        return logits;
    } catch (...) {
        gdn_to_spare_ = false;
        throw;
    }
}

void Qwen4ExpSession::keep_verify() {
    STRIX_CHECK(verify_pending_, "Qwen4ExpSession::keep_verify: no forward_verify to keep");
    std::swap(conv_state_, conv_spare_);
    std::swap(rec_state_, rec_spare_);
    verify_pending_ = false;
}

void Qwen4ExpSession::drop_verify() {
    STRIX_CHECK(verify_pending_, "Qwen4ExpSession::drop_verify: no forward_verify to drop");
    copy_state(ple_state_kept_, ple_state_, stream_, "PLE conv state");
    copy_state(mtp_prev_kept_, mtp_prev_, stream_, "MTP previous streams");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::drop_verify");
    pos_ = verify_pos_, tail_cur_ = verify_tail_cur_, mtp_tail_cur_ = verify_mtp_tail_cur_, ple_hist_ = verify_hist_;
    mtp_chain_step_ = -1;
    rewind_to_.push_back(verify_pos_);  // rows past it were overwritten: snapshots beyond it no longer match
    verify_pending_ = false;
}

void Qwen4ExpSession::keep_verify_prefix(int64_t rows) {
    STRIX_CHECK(verify_pending_, "Qwen4ExpSession::keep_verify_prefix: no forward_verify pending");
    const int64_t T = (int64_t)verify_ids_.size();
    STRIX_CHECK(rows >= 1 && rows < T, "Qwen4ExpSession::keep_verify_prefix: keep ", rows, " rows of a ", T,
                "-row verify, expected 1..", T - 1, " (all of them: keep_verify; none: drop_verify)");
    const Qwen4ExpDims &D = m_.dims();
    const Act act = m_.act();
    const size_t e = es(act);
    const int64_t n4 = D.H * D.d, p0 = verify_pos_;
    broken_ = true;  // until the state is whole again
    for (int64_t i = 0; i < D.layers; ++i) {
        const Qwen4ExpModel::Layer &l = m_.layer(i);
        const size_t li = (size_t)i;
        if (!D.is_attention[li]) {
            // GDN: the verify wrote its states to the spare set, so conv_state_ / rec_state_ are still the ones before
            // it - replay rows [0, rows) on them in place, with the verify's own path (Auto: per token), so the state
            // is exactly the verify's after that row (split invariance, tests/test_gdn_kernel *_mtp_rollback).
            kernels::gdn_front(verify_in_[li].get(),
                               {rows, D.gk, kHD, D.gv, kHD, D.gstride, D.conv_c + D.gz, D.conv_c + D.gz + D.gv},
                               l.conv_w, conv_state_[li].get(), conv_state_[li].get(), false, l.A_log, l.dt_bias,
                               qkv_.get(), beta_.get(), g_.get(), act, stream_);
            kernels::delta_rule(qkv_.get(), beta_.get(), g_.get(), {rows, D.gk, D.gv, D.conv_c}, rec_state_[li].get(),
                                rec_state_[li].get(), false, core_.get(), act, stream_, kernels::DeltaRulePath::Auto);
        } else {
            // Attention: the tail (and any block completed within the kept rows - same keys the verify wrote) from
            // the tail before the verify, which it only read.
            kernels::qsa_block_keys(verify_in_[li].get(), D.idx_d, rows, p0,
                                    p0 % kernels::kQsaBlock ? tail_[verify_tail_cur_][li].get() : nullptr,
                                    tail_[1 - verify_tail_cur_][li].get(), D.idx_d, l.idx_k_norm, D.eps,
                                    m_.inv_freq(), D.rot, block_keys_[li].get(), cap_blocks_, act, stream_, m_.rope_scale());
        }
    }
    if (D.ple_layer >= 0) {
        // PLE: the history held [S state rows][T verify rows]; the state after row rows - 1 is its rows [rows, rows + S).
        const int64_t S = (D.ple_taps - 1) * D.ple_dil;
        STRIX_HIP_CHECK(hipMemcpyAsync(ple_state_.get(), ple_hist_buf_.get() + (size_t)(rows * n4) * e,
                                       (size_t)(S * n4) * e, hipMemcpyDeviceToDevice, stream_),
                        "Qwen4ExpSession::keep_verify_prefix: PLE state");
    }
    if (mtp_) {
        // The MTP layer's projections are the last thing the verify wrote to proj_; its streams are still in x_.
        // (Invariant, ST-3 design 1.2: nothing writes proj_ between the verify's catch-up and this keep - drafts come
        // after it; a verify run with set_mtp_catchup(false) left no projections here.)
        STRIX_CHECK(verify_caught_up_, "Qwen4ExpSession::keep_verify_prefix: the verify ran without the MTP catch-up "
                    "(set_mtp_catchup(false)), so there are no MTP projections to keep");
        const Qwen4ExpModel::Layer &ml = m_.mtp().layer;
        const Qwen4ExpDims &MD = m_.mtp_dims();
        kernels::qsa_block_keys(proj_.get() + (size_t)(MD.idx_col + D.idx_h * D.idx_d) * e, MD.astride, rows, p0,
                                p0 % kernels::kQsaBlock ? tail_mtp_[verify_mtp_tail_cur_].get() : nullptr,
                                tail_mtp_[1 - verify_mtp_tail_cur_].get(), D.idx_d, ml.idx_k_norm, D.eps, m_.inv_freq(),
                                D.rot, block_keys_mtp_.get(), cap_blocks_, act, stream_, m_.rope_scale());
        STRIX_HIP_CHECK(hipMemcpyAsync(mtp_prev_.get(), x_.get() + (size_t)((rows - 1) * n4) * e, (size_t)n4 * e,
                                       hipMemcpyDeviceToDevice, stream_),
                        "Qwen4ExpSession::keep_verify_prefix: MTP previous streams");
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::keep_verify_prefix");
    // The n-gram history: the one before the verify, advanced over the kept ids.
    PleHistory h = verify_hist_;
    if (D.ple_layer >= 0) {
        std::vector<int64_t> tok(verify_ids_.begin(), verify_ids_.begin() + rows), out((size_t)(rows * 16));
        ple_ngram_ids(m_.ngram_rows().hash_params(), tok.data(), rows, h, out.data());
    }
    ple_hist_ = h;
    pos_ = p0 + rows;
    tail_cur_ = 1 - verify_tail_cur_, mtp_tail_cur_ = 1 - verify_mtp_tail_cur_;
    mtp_chain_step_ = -1;
    rewind_to_.push_back(pos_);  // the verify's rows past pos_ get overwritten: snapshots beyond it no longer match
    verify_pending_ = false;
    broken_ = false;
}

void Qwen4ExpSession::set_mtp_vocab(int64_t n) {
    // The draft scores the LM head's first n rows - under tensor parallelism rank 0's (Qwen4ExpModel::draft_rows).
    const int64_t rows = std::min(m_.dims().vocab, m_.draft_rows());
    STRIX_CHECK(n >= 1 && n <= rows, "Qwen4ExpSession::set_mtp_vocab: ", n, ", expected 1..", rows,
                rows < m_.dims().vocab ? " (the LM head rows this rank holds)" : "");
    mtp_vocab_ = n;
}

void Qwen4ExpSession::set_mtp_draft_q4(bool on) { mtp_draft_q4_ = on; }

void Qwen4ExpSession::mtp_input(int64_t T, const void *prev0) {
    STRIX_CHECK(mtp_, "Qwen4ExpSession::mtp_input: MTP is off");
    STRIX_CHECK(T >= 1 && T <= max_tokens_, "Qwen4ExpSession::mtp_input: ", T, " rows, expected 1..", max_tokens_);
    const Qwen4ExpDims &D = m_.dims();
    const Act act = m_.act();
    const size_t e = es(act);
    const int64_t n4 = D.H * D.d;
    const Qwen4ExpModel::MtpHead &mtp = m_.mtp();
    const bool wmma = prefill_math_ == PrefillMath::WmmaBf16;
    auto lin = [&](const void *x, const QWeightView &w, void *y, int64_t M) {
        if (wmma && M >= kWmmaMinTokens) kernels::linear_qw_wmma(x, w, y, M, act, act, stream_);
        else kernels::linear_qw(x, w, y, M, act, act, stream_);
    };
    // e = rmsnorm_zc(embed(token)) [T, d]; h = the previous streams, each stream normed [T, H, d].
    if (m_.is_embed_q8())
        kernels::embedding_lookup(ids_.get(), T, m_.embed_q8(), mtp_emb_.get(), act, emb_err_.get(), stream_);
    else
        kernels::embedding_lookup(ids_.get(), T, m_.embed(), D.vocab, D.d, mtp_emb_.get(), act, emb_err_.get(), stream_);
    kernels::rmsnorm_zc(mtp_emb_.get(), mtp.pre_fc_norm_embedding, mtp_norm_emb_.get(), T, D.d, D.eps, act, stream_);
    STRIX_CHECK(prev0 != nullptr, "Qwen4ExpSession::mtp_input: null previous streams");
    kernels::group_rmsnorm_zc(prev0, mtp.pre_fc_norm_hidden, mtp_norm_hid_.get(), 1, D.H, D.d, D.eps, act, stream_);
    if (T > 1)
        kernels::group_rmsnorm_zc(x_.get(), mtp.pre_fc_norm_hidden, mtp_norm_hid_.get() + (size_t)n4 * e, T - 1, D.H,
                                  D.d, D.eps, act, stream_);
    // Per stream s: x_s = fc_embedding(e) + fc_hidden(h_s) - fc_hidden over the T * H stream rows ([T, H, d] is
    // [T * H, d]), then fc_embedding's row added to each of its token's streams (an injection with weight 1).
    lin(mtp_norm_emb_.get(), mtp.fc_embedding, mtp_proj_emb_.get(), T);
    lin(mtp_norm_hid_.get(), mtp.fc_hidden, mtp_x_.get(), T * D.H);
    kernels::hc_inject(mtp_x_.get(), mtp_ones_.get(), mtp_proj_emb_.get(), T, D.H, D.d, act, stream_);
}

std::vector<float> Qwen4ExpSession::forward_mtp(int32_t token_id, int64_t step) {
    run_mtp(token_id, step);
    std::vector<float> logits = read_back((size_t)mtp_vocab_, " MTP");
    mtp_done(step);
    return logits;
}

Qwen4ExpSession::MtpTop2 Qwen4ExpSession::forward_mtp_top2(int32_t token_id, int64_t step) {
    run_mtp(token_id, step);
    if (mtp_pick_.size() == 0) mtp_pick_ = DeviceBuffer<uint8_t>(sizeof(kernels::MtpPick), "MTP pick");
    auto *dev = reinterpret_cast<kernels::MtpPick *>(mtp_pick_.get());
    kernels::mtp_pick(logits_.get(), mtp_vocab_, router_err_.get(), expert_err_.get(), attn_err_.get(),
                      emb_err_.get(), dev, stream_);
    // One copy into the pinned area (the logits' part - unused here) behind the forward, one synchronize.
    STRIX_CHECK(readback_.size() >= kReadbackLogitsAt + sizeof(kernels::MtpPick),
                "Qwen4ExpSession::forward_mtp_top2: pinned area of ", readback_.size(), " bytes");
    auto *host = reinterpret_cast<kernels::MtpPick *>(static_cast<uint8_t *>(readback_.get()) + kReadbackLogitsAt);
    STRIX_HIP_CHECK(hipMemcpyAsync(host, dev, sizeof(kernels::MtpPick), hipMemcpyDeviceToHost, stream_),
                    "Qwen4ExpSession::forward_mtp_top2: the pick");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream_), "Qwen4ExpSession::forward_mtp_top2: end of the draft");
    const kernels::MtpPick p = *host;
    kernels::report_router_error(p.err, "Qwen4ExpSession MTP router");
    kernels::report_expert_error(p.err + 3, "Qwen4ExpSession MTP experts");
    kernels::report_attention_error(p.err + 6, "Qwen4ExpSession MTP QSA attention");
    kernels::report_embedding_error(p.emb, m_.dims().vocab);
    STRIX_CHECK(p.nan || (p.best >= 0 && p.best < mtp_vocab_), "Qwen4ExpSession::forward_mtp_top2: pick ", p.best,
                " outside the draft vocabulary [0, ", mtp_vocab_, ")");
    mtp_done(step);
    return {p.best, p.best_v, p.second_v, p.nan != 0};
}

void Qwen4ExpSession::mtp_done(int64_t step) {
    broken_ = false;
    mtp_chain_step_ = step, mtp_chain_pos_ = pos_;
}

void Qwen4ExpSession::run_mtp(int32_t token_id, int64_t step) {
    STRIX_TRACE_RANGE("draft " + std::to_string(step));
    STRIX_CHECK(mtp_, "Qwen4ExpSession::forward_mtp: the session was made without MTP");
    STRIX_CHECK(!broken_, "Qwen4ExpSession::forward_mtp: an earlier call failed midway; reset() the session first");
    const Qwen4ExpDims &D = m_.dims(), &MD = m_.mtp_dims();  // MD: the whole head's widths (ST-3)
    STRIX_CHECK(token_id >= 0 && token_id < D.vocab, "Qwen4ExpSession::forward_mtp: token ", token_id,
                " outside [0, ", D.vocab, ")");
    STRIX_CHECK(step >= 0, "Qwen4ExpSession::forward_mtp: step ", step, ", expected >= 0");
    STRIX_CHECK(!verify_pending_, "Qwen4ExpSession::forward_mtp: a forward_verify awaits keep_verify / drop_verify");
    STRIX_CHECK(step == 0 || (mtp_chain_pos_ == pos_ && mtp_chain_step_ == step - 1), "Qwen4ExpSession::forward_mtp: step ",
                step, " doesn't continue a chain (last step ", mtp_chain_step_, " at position ", mtp_chain_pos_,
                ", now at ", pos_, ")");
    const int64_t p = pos_ + step;  // this row's position
    STRIX_CHECK(p < capacity_, "Qwen4ExpSession::forward_mtp: position ", p, " is at the capacity ", capacity_);
    const Act act = m_.act();
    const size_t e = es(act);
    const bool sep = m_.shared_separate();
    const int64_t T = 1, A = D.top_k + (sep ? 0 : 1), hd = D.hd;
    const Qwen4ExpModel::MtpHead &mtp = m_.mtp();
    const Qwen4ExpModel::Layer &l = mtp.layer;
    auto at = [&](const DeviceBuffer<uint8_t> &b, int64_t elem) { return b.get() + (size_t)elem * e; };
    auto hc_mix = [&](const Qwen4ExpModel::Hc &hc, bool inject) {
        float *w_in = inject ? w_in_.get() : nullptr;
        if (hc.down.bits == 8)
            kernels::hc_mix_down(mtp_x_.get(), hc.down.q8, T, D.H, D.d, D.r, inject, D.eps, h_.get(), w_in, inv_.get(), act, stream_);
        else
            kernels::hc_mix_down(mtp_x_.get(), hc.down.q4, T, D.H, D.d, D.r, inject, D.eps, h_.get(), w_in, inv_.get(), act, stream_);
        if (hc.up.bits == 8)
            kernels::hc_mix_up(h_.get(), hc.up.q8, mtp_x_.get(), hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act, stream_);
        else
            kernels::hc_mix_up(h_.get(), hc.up.q4, mtp_x_.get(), hc.norm, inv_.get(), T, D.H, D.d, u_.get(), act, stream_);
    };
    broken_ = true;
    mtp_chain_step_ = -1;
    // Indexer tails: step 0 continues the committed tail into the spare buffer (not flipped); chain steps continue
    // from the step before into their own pair.
    const void *tail_in = step == 0 ? tail_mtp_[mtp_tail_cur_].get()
                          : step == 1 ? tail_mtp_[1 - mtp_tail_cur_].get()
                                      : mtp_chain_tail_[step % 2].get();
    void *tail_out = step == 0 ? tail_mtp_[1 - mtp_tail_cur_].get() : mtp_chain_tail_[(step + 1) % 2].get();
    if (step > 0) {  // the step before's output streams, kept by swapping buffers (mtp_input overwrites mtp_x_)
        STRIX_CHECK(mtp_chain_prev_.size() == mtp_x_.size(), "Qwen4ExpSession: MTP chain buffers ", mtp_chain_prev_.size(),
                    " / ", mtp_x_.size(), " bytes, expected equal (they're swapped)");
        std::swap(mtp_x_, mtp_chain_prev_);
    }
    STRIX_HIP_CHECK(hipMemcpyAsync(ids_.get(), &token_id, 4, hipMemcpyHostToDevice, stream_), "uploading the MTP token id");
    kernels::reset_embedding_error(emb_err_.get(), stream_);
    mtp_input(T, step == 0 ? mtp_prev_.get() : mtp_chain_prev_.get());

    // The layer, as a trunk full-attention layer at position p. Step 0's KV row and block keys are the ones the next
    // forward's catch-up writes again from the same inputs; a chain step's, until the position runs, its guess.
    hc_mix(l.hc_attn, true);
    uint8_t *kc = k_cache_mtp_.get(), *vc = v_cache_mtp_.get();
    kernels::linear_qw(u_.get(), l.qkv, proj_.get(), T, act, act, stream_);
    // q, k, v and (past the dense limit) the indexer queries in one launch, as in the trunk's layers.
    const bool qsa_layer = p + T > D.dense_key_limit();
    const kernels::NormRopeJob nr[4] = {
        {proj_.get(), MD.astride, 2 * hd, proj_.get(), MD.astride, 2 * hd, MD.hq, hd, l.q_norm},
        {at(proj_, MD.hq * 2 * hd), MD.astride, hd, kc + (size_t)(p * MD.hkv * hd) * e, MD.hkv * hd, hd, MD.hkv, hd, l.k_norm},
        {at(proj_, MD.hq * 2 * hd + MD.hkv * hd), MD.astride, hd, vc + (size_t)(p * MD.hkv * hd) * e, MD.hkv * hd, hd, MD.hkv,
         hd, nullptr},
        {at(proj_, MD.idx_col), MD.astride, D.idx_d, at(proj_, MD.idx_col), MD.astride, D.idx_d, D.idx_h, D.idx_d,
         l.idx_q_norm}};
    kernels::qk_norm_rope_jobs(nr, qsa_layer ? 4 : 3, T, D.eps, D.rot, p, m_.inv_freq(), act, stream_, m_.rope_scale());
    kernels::qsa_block_keys(at(proj_, MD.idx_col + D.idx_h * D.idx_d), MD.astride, T, p,
                            p % kernels::kQsaBlock ? tail_in : nullptr, tail_out, D.idx_d, l.idx_k_norm, D.eps, m_.inv_freq(),
                            D.rot, block_keys_mtp_.get(), cap_blocks_, act, stream_, m_.rope_scale());
    const kernels::AttentionShape as{T, p, MD.hq, MD.hkv, hd, MD.astride, 2 * hd};
    const float scale = 1.0f / std::sqrt((float)hd);
    if (!qsa_layer) {
        kernels::attention(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, scale, core_.get(), attn_ws_.get(),
                           attn_ws_bytes_, act, stream_);
    } else {
        const int64_t qsa_k = D.qsa_budget / kernels::kQsaBlock;
        void *iq = at(proj_, MD.idx_col);  // normed + roped by the qk_norm_rope_jobs above
        kernels::qsa_scores(iq, MD.astride, D.idx_d, T, p, D.idx_h, D.idx_d, block_keys_mtp_.get(), cap_blocks_,
                            qsa_scores_.get(), cap_blocks_, act, stream_);
        kernels::qsa_topk(qsa_scores_.get(), cap_blocks_, T, p, qsa_k, qsa_sel_.get(), qsa_nsel_.get(),
                          qsa_topk_ws_.get(), qsa_topk_ws_bytes_, stream_);
        kernels::attention_gathered(proj_.get(), at(proj_, hd), kc, vc, capacity_, as, qsa_sel_.get(), qsa_nsel_.get(),
                                    qsa_k, scale, core_.get(), attn_ws_.get(), attn_ws_bytes_, attn_err_.get(), act,
                                    stream_);
    }
    kernels::linear_qw(core_.get(), l.o_proj, y_.get(), T, act, act, stream_);
    kernels::hc_inject(mtp_x_.get(), w_in_.get(), y_.get(), T, D.H, D.d, act, stream_);

    hc_mix(l.hc_mlp, true);
    kernels::linear_bf16w(u_.get(), l.router, router_logits_.get(), T, D.experts + 1, D.d, act, Act::F32, stream_);
    kernels::moe_router(router_logits_.get(), D.experts + 1, T, D.experts, D.top_k,
                        sep ? nullptr : router_logits_.get() + D.experts, D.experts + 1, (int32_t)D.experts,
                        route_ids_.get(), route_coef_.get(), Act::F32, router_err_.get(), stream_);
    const int64_t Estack = D.experts + (sep ? 0 : 1);
    kernels::linear_qw_experts_gather_swiglu(u_.get(), l.gate_up, Estack, route_ids_.get(), T, A, hh_.get(), act,
                                             expert_err_.get(), stream_);
    kernels::linear_qw_experts_combine(hh_.get(), l.down, Estack, route_ids_.get(), route_coef_.get(), T, A, y_.get(),
                                       act, expert_err_.get(), stream_);
    if (sep) {
        kernels::linear_qw(u_.get(), l.shared_gate_up, gu_.get(), T, act, act, stream_);
        kernels::swiglu(gu_.get(), hh_.get(), T, MD.inter, act, stream_);
        kernels::linear_qw(hh_.get(), l.shared_down, sh_y_.get(), T, act, act, stream_);
        kernels::moe_shared_add_inject(y_.get(), sh_y_.get(), router_logits_.get() + D.experts, D.experts + 1, T, D.d,
                                       (uint32_t)D.experts, mtp_x_.get(), w_in_.get(), D.H, act, router_err_.get(),
                                       stream_);
    } else {
        kernels::hc_inject(mtp_x_.get(), w_in_.get(), y_.get(), T, D.H, D.d, act, stream_);
    }

    // The head's own collapse, then the LM head's first mtp_vocab_ rows (row-major: a prefix view) - from the model's
    // Q4 draft copy when it covers mtp_vocab_ (Qwen4ExpModel::make_draft_head_q4: drafts only, the verify decides),
    // else from the LM head as loaded.
    hc_mix(mtp.hc_mixer, false);
    const QWeightView &draft = m_.draft_head_q4();
    // mtp_draft_q4_ off (tp_ar's Q8 comparison cells): the Q8 rows, as before the copy.
    QWeightView head = mtp_draft_q4_ && draft.bits != 0 && mtp_vocab_ <= draft.N() ? draft : m_.draft_lm();
    STRIX_CHECK(head.K() == m_.lm_head().K() && mtp_vocab_ <= head.N(), "Qwen4ExpSession::forward_mtp: draft head [",
                head.N(), ", ", head.K(), "] for mtp_vocab ", mtp_vocab_, " and the LM head's K ", m_.lm_head().K());
    switch (head.bits) {
        case 8: head.q8.N = mtp_vocab_; break;
        case 6: head.q6.N = mtp_vocab_; break;
        case 5: head.q5.N = mtp_vocab_; break;
        case 4: head.q4.N = mtp_vocab_; break;
        default: STRIX_FAIL("Qwen4ExpSession::forward_mtp: LM head has ", head.bits, " bits, expected 4, 5, 6 or 8");
    }
    kernels::linear_qw(u_.get(), head, logits_.get(), 1, act, Act::F32, stream_);
}

}  // namespace strix
