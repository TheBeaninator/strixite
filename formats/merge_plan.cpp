#include "formats/merge_plan.hpp"

#include "common/check.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace strix {

std::vector<MergeGroup> merge_plan(Arch arch, const std::string &p, LayerKind kind) {
    STRIX_CHECK(!p.empty() && p.back() == '.', "merge_plan: layer prefix '", p, "' must end with '.'");
    STRIX_CHECK(arch == Arch::Qwen3_5 || arch == Arch::Qwen4Exp, "merge_plan: unknown arch ", (int)arch);
    STRIX_CHECK(kind == LayerKind::LinearAttention || kind == LayerKind::FullAttention, "merge_plan: unknown layer kind ",
                (int)kind);
    std::vector<MergeGroup> g;
    const std::string w = ".weight";
    if (kind == LayerKind::LinearAttention) {
        // GDN input projections all read the normed hidden state.
        const std::string a = p + "linear_attn.";
        g.push_back({a + "in_proj", {a + "in_proj_qkv" + w, a + "in_proj_z" + w, a + "in_proj_b" + w, a + "in_proj_a" + w}});
    } else {
        const std::string a = p + "self_attn.";
        MergeGroup qkv{a + "qkv", {a + "q_proj" + w, a + "k_proj" + w, a + "v_proj" + w}};
        // qwen4_exp's QSA indexer projects the same hidden state as q/k/v.
        if (arch == Arch::Qwen4Exp) qkv.parts.push_back(a + "indexer.index_qk_proj" + w);
        g.push_back(qkv);
    }
    const std::string m = p + "mlp.";
    if (arch == Arch::Qwen3_5) {
        g.push_back({m + "gate_up", {m + "gate_proj" + w, m + "up_proj" + w}});
    } else {
        // Shared expert = expert #E of the stacked routed experts: its gate + up rows
        // match an expert's [gate | up] gate_up_proj layout; its combine coefficient
        // is sigmoid(shared_expert_gate . x).
        const std::string s = m + "shared_expert.";
        g.push_back({m + "experts_gate_up", {m + "experts.gate_up_proj", s + "gate_proj" + w, s + "up_proj" + w}});
        g.push_back({m + "experts_down", {m + "experts.down_proj", s + "down_proj" + w}});
        // Router logits (512) + the shared expert's gate logit: both read the MoE input, both are
        // stored FP32, and moe_router reads them from one [T, 513] output.
        g.push_back({m + "router", {m + "gate" + w, m + "shared_expert_gate" + w}});
    }
    return g;
}

namespace {

const TensorInfo &need(const TensorLookup &lookup, const MergeGroup &g, const std::string &name) {
    const TensorInfo *t = lookup(name);
    STRIX_CHECK(t != nullptr, "merge group '", g.name, "': tensor '", name, "' not found in the checkpoint");
    return *t;
}

std::string shape_str(const std::vector<int64_t> &s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "]";
}

}  // namespace

MergedLayout resolve_merge(const MergeGroup &g, const TensorLookup &lookup) {
    STRIX_CHECK(lookup != nullptr, "resolve_merge('", g.name, "'): null lookup");
    STRIX_CHECK(g.parts.size() >= 2, "merge group '", g.name, "' has ", g.parts.size(), " parts, expected >= 2");
    MergedLayout m;
    m.group = g;
    for (const std::string &name : g.parts) {
        const TensorInfo &t = need(lookup, g, name);
        STRIX_CHECK(t.dtype == Dtype::BF16, "merge group '", g.name, "': '", name, "' is ", dtype_name(t.dtype),
                    ", expected BF16");
        STRIX_CHECK(t.shape.size() == 2 || t.shape.size() == 3, "merge group '", g.name, "': '", name, "' has shape ",
                    shape_str(t.shape), ", expected [N, K] or [E, N, K]");
        for (int64_t d : t.shape)
            STRIX_CHECK(d >= 1, "merge group '", g.name, "': '", name, "' has shape ", shape_str(t.shape));
        const int64_t K = t.shape.back();
        const int64_t rows = t.shape.size() == 3 ? t.shape[0] * t.shape[1] : t.shape[0];
        STRIX_CHECK(m.K == 0 || K == m.K, "merge group '", g.name, "': '", name, "' has K = ", K, " (shape ",
                    shape_str(t.shape), "), but '", g.parts[0], "' has K = ", m.K, "; all parts must share K");
        m.K = K;
        m.offset.push_back(m.rows);
        m.part_rows.push_back(rows);
        m.rows += rows;
    }
    // Experts: the per-expert row count comes from the 3D part.
    for (const std::string &name : g.parts) {
        const TensorInfo &t = need(lookup, g, name);
        if (t.shape.size() == 3) {
            STRIX_CHECK(m.experts == 0, "merge group '", g.name, "': more than one 3D (expert) part");
            m.group.expert_rows = t.shape[1];
            STRIX_CHECK(m.rows % t.shape[1] == 0, "merge group '", g.name, "': ", m.rows,
                        " merged rows don't split into whole experts of ", t.shape[1], " rows ('", name, "' is ",
                        shape_str(t.shape), ")");
            m.experts = m.rows / t.shape[1];
        }
    }
    return m;
}

Q4Weight quantize_merged(const MergedLayout &m, const TensorLookup &lookup, const TensorData &data, int64_t G) {
    STRIX_CHECK(lookup != nullptr && data != nullptr, "quantize_merged('", m.group.name, "'): null lookup or data");
    STRIX_CHECK(m.K >= 1 && m.rows >= 1 && m.offset.size() == m.group.parts.size(), "quantize_merged('", m.group.name,
                "'): layout not resolved (K = ", m.K, ", rows = ", m.rows, ")");
    STRIX_CHECK(m.K % G == 0 && q4_group_size_supported(G), "quantize_merged('", m.group.name, "'): K = ", m.K,
                " with group size ", G, " (must be 32/64/128 and divide K)");
    Q4Weight out;
    const int64_t kChunkRows = std::max<int64_t>(1, (64ll << 20) / (m.K * 4));  // ~64 MB of FP32 at a time
    std::vector<float> f;
    for (size_t i = 0; i < m.group.parts.size(); ++i) {
        const TensorInfo &t = need(lookup, m.group, m.group.parts[i]);
        STRIX_CHECK(t.dtype == Dtype::BF16 && t.shape.back() == m.K && t.byte_length == (size_t)(m.part_rows[i] * m.K) * 2,
                    "quantize_merged('", m.group.name, "'): '", t.name, "' changed since resolve_merge (",
                    dtype_name(t.dtype), " ", shape_str(t.shape), ", ", t.byte_length, " bytes)");
        const auto *src = static_cast<const uint16_t *>(data(t));
        STRIX_CHECK(src != nullptr && reinterpret_cast<uintptr_t>(src) % 2 == 0, "quantize_merged('", m.group.name,
                    "'): data for '", t.name, "' is null or not 2-byte aligned (", (const void *)src, ")");
        for (int64_t r0 = 0; r0 < m.part_rows[i]; r0 += kChunkRows) {
            const int64_t n = std::min(kChunkRows, m.part_rows[i] - r0);
            f.resize((size_t)(n * m.K));
            const uint16_t *b = src + r0 * m.K;
            for (size_t j = 0; j < f.size(); ++j) {
                uint32_t u = (uint32_t)b[j] << 16;
                std::memcpy(&f[j], &u, 4);
            }
            append_rows_q4(out, quantize_q4(f.data(), n, m.K, G), m.group.name.c_str());
        }
    }
    STRIX_CHECK(out.N == m.rows, "quantize_merged('", m.group.name, "'): produced ", out.N, " rows, expected ", m.rows);
    return out;
}

}  // namespace strix
