#pragma once

// Load-time merges of linear weights that read the same input. Small projections are launch-bound on the GPU
// (e.g. 512x2560 at <50% of read bandwidth vs ~89% for 12288x2560), so each
// group below is stored as one row-concatenated weight and run as one launch;
// the consumers read their slice of the output by row offset.
//
// Row concatenation happens at quantization time and is exact (append_rows_q4):
// a merged launch computes each row exactly as a separate launch would, as
// long as the members share K, format and group size - a member the bit
// budget puts in another format can't be in a group.

#include "formats/q4.hpp"
#include "formats/safetensors.hpp"

#include <functional>
#include <string>
#include <vector>

namespace strix {

enum class Arch { Qwen3_5, Qwen4Exp };  // Qwen3.5 dense (0.8B), qwen4_exp (Qwen3.8-Flash-Next)
enum class LayerKind { LinearAttention, FullAttention };

struct MergeGroup {
    std::string name;                // prefix + a short name, e.g. "...layers.3.self_attn.qkv"
    std::vector<std::string> parts;  // checkpoint tensor names, in output-row order
    // > 0: the merged rows are stacked experts of this many rows each (a 3D
    // [E, N, K] part counts as E*N rows; e.g. the shared expert's gate + up
    // together make expert #E). 0: an ordinary merged linear.
    int64_t expert_rows = 0;
};

// The groups for one decoder layer. prefix: the layer's tensor-name prefix,
// e.g. "model.language_model.layers.3." or "mtp.layers.0.".
std::vector<MergeGroup> merge_plan(Arch arch, const std::string &prefix, LayerKind kind);

using TensorLookup = std::function<const TensorInfo *(const std::string &name)>;  // nullptr if absent
using TensorData = std::function<const void *(const TensorInfo &t)>;

struct MergedLayout {
    MergeGroup group;
    int64_t K = 0, rows = 0;
    int64_t experts = 0;                    // rows / expert_rows, or 0
    std::vector<int64_t> offset, part_rows;  // per part: first output row, row count
};

// Checks every part exists, is BF16, 2D [N, K] or 3D [E, N, K], and that all
// share K (and, for experts, the rows split into whole experts). Throws naming
// the group, part and shapes otherwise.
MergedLayout resolve_merge(const MergeGroup &g, const TensorLookup &lookup);

// Quantizes the merged weight part by part, in row chunks (a 512-expert part
// never needs a full FP32 copy). Byte-identical to quantizing each part alone
// and appending.
Q4Weight quantize_merged(const MergedLayout &m, const TensorLookup &lookup, const TensorData &data, int64_t G);

}  // namespace strix
