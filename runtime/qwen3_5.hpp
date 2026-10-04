#pragma once

// Qwen3.5 text model on the GPU - stage 1's full
// forward: embedding, 24 decoder layers (GDN or full attention mixer + SwiGLU
// MLP, pre-norm residual), final norm, tied LM head.
//
// Qwen35Model holds the weights on the device: the checkpoint's BF16 linear
// weights as stored, with the load-time merges (formats/merge_plan: GDN
// in_proj qkv|z|b|a, attention q|k|v, MLP gate|up); norm weights, conv
// weights, A_log and dt_bias as FP32. BF16 linears use the simple
// linear_bf16w kernel - the correctness baseline; the tuned 4-bit path swaps
// in once its layout is decided.
//
// Qwen35Session is one conversation: the KV
// cache of each attention layer, the conv + recurrent state of each GDN
// layer, the position, and preallocated workspaces for up to max_tokens
// tokens per forward call. forward() runs a batch of tokens at the current
// position (prefill or decode) and returns logits for the last few.
//
// Dims come from the checkpoint's tensor shapes; the scalar hyperparameters
// (eps, RoPE theta, rotary fraction) are Qwen3.5's config.json values,
// written down here because this stage had no JSON parser, and pinned by the golden tests.

#include "formats/safetensors.hpp"
#include "kernels/embedding.hpp"
#include "kernels/norm.hpp"  // Act
#include "runtime/device_buffer.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strix {

struct Qwen35Dims {
    int64_t d = 0, vocab = 0, layers = 0, inter = 0;
    std::vector<bool> is_attention;  // per layer: full attention (else GDN)
    // Full attention.
    int64_t hq = 0, hkv = 0, head_dim = 0, rot_dim = 0;
    // GDN (head dims are kernels::kGdnHeadDim).
    int64_t gk = 0, gv = 0, conv_c = 0;
    float eps = 1e-6f, rope_theta = 1e7f;
};

class Qwen35Model {
public:
    // path: the 0.8B safetensors checkpoint. act: activation dtype of every run on this model.
    Qwen35Model(const std::string &path, kernels::Act act);
    const Qwen35Dims &dims() const { return dims_; }
    kernels::Act act() const { return act_; }

    struct Layer {
        DeviceBuffer<float> in_norm, post_norm;         // [d], applied as 1 + w
        DeviceBuffer<uint16_t> gate_up, down;           // [2I, d], [d, I]
        // GDN
        DeviceBuffer<uint16_t> in_proj, out_proj;       // [C + Z + 2 Hv, d], [d, Z]
        DeviceBuffer<float> conv_w, A_log, dt_bias, gdn_norm;
        // Full attention
        DeviceBuffer<uint16_t> qkv, o_proj;             // [Hq 2D + 2 Hkv D, d], [d, Hq D]
        DeviceBuffer<float> q_norm, k_norm;
    };
    const Layer &layer(int64_t i) const;
    const DeviceBuffer<uint16_t> &embed() const { return embed_; }  // [vocab, d]; also the LM head
    const DeviceBuffer<float> &final_norm() const { return final_norm_; }
    const DeviceBuffer<float> &inv_freq() const { return inv_freq_; }

private:
    Qwen35Dims dims_;
    kernels::Act act_;
    DeviceBuffer<uint16_t> embed_;
    DeviceBuffer<float> final_norm_, inv_freq_;
    std::vector<Layer> layers_;
};

// Called with each intermediate the goldens capture, under the goldens' names
// ("embed", "L<i>.in_norm", "L<i>.mixer", "L<i>.post_norm", "L<i>.mlp",
// "L<i>.out", "final_norm"): a device pointer to rows x cols values in the
// activation dtype, valid only during the call (the stream is synchronized first).
using ForwardProbe = std::function<void(const std::string &name, const void *dev, int64_t rows, int64_t cols)>;

class Qwen35Session {
public:
    // capacity: max positions (KV cache rows); max_tokens: max tokens per forward call.
    Qwen35Session(const Qwen35Model &model, int64_t capacity, int64_t max_tokens);
    ~Qwen35Session();
    Qwen35Session(const Qwen35Session &) = delete;
    Qwen35Session &operator=(const Qwen35Session &) = delete;

    void reset();  // a fresh conversation: position 0, states cleared
    int64_t pos() const { return pos_; }

    // Runs ids at positions pos()..pos()+T-1 and advances pos(). Returns the
    // logits of the last n_logits (0..kMaxLogits) tokens, [n_logits, vocab]
    // row-major, as FP32 (BF16 activations: the BF16 logits widened). If it
    // throws, the session refuses further calls until reset().
    static constexpr int64_t kMaxLogits = 16;
    std::vector<float> forward(const std::vector<int32_t> &ids, int64_t n_logits, const ForwardProbe &probe = nullptr);

private:
    const Qwen35Model &m_;
    int64_t capacity_, max_tokens_, pos_ = 0;
    hipStream_t stream_ = nullptr;
    // Per-layer state.
    std::vector<DeviceBuffer<uint8_t>> k_cache_, v_cache_, conv_state_;
    std::vector<DeviceBuffer<float>> rec_state_;
    // Workspaces [max_tokens, ...] in the activation dtype unless noted.
    DeviceBuffer<int32_t> ids_;
    DeviceBuffer<uint8_t> x_, n_, proj_, qkv_, core_, z_, gnorm_, mix_, gu_, h_, logits_;
    DeviceBuffer<float> beta_, g_, attn_ws_;
    DeviceBuffer<kernels::EmbeddingError> emb_err_;
    size_t attn_ws_bytes_ = 0;
    bool broken_ = false;  // a forward threw midway: state is inconsistent until reset()
};

}  // namespace strix
