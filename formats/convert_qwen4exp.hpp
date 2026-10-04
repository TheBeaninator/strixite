#pragma once

// Offline converter: Qwen3.8-Flash-Next HF safetensors -> weights.strixw (formats/strixw.hpp; layout A by default).
// Plans every output tensor from the checkpoint headers alone (merge groups from formats/merge_plan, the HC
// fold, chunk-major W_up, the router in BF16, small tensors widened to F32), then streams the sources
// (pread in ~256 MB chunks, the next chunk read while this one is quantized on all cores, page cache
// dropped behind) and writes each tensor as soon as it's done. Verbose by design: the log explains each
// transform the first time it appears and reports per-tensor size, bits per weight, sampled quantization
// error and time, and per-layer progress with throughput and an ETA.

#include "formats/strixw.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strix {

struct ConvertOptions {
    std::string src_dir;      // HF checkpoint directory (*.safetensors + model.safetensors.index.json)
    std::string out_dir;      // created if missing; must not already hold weights.strixw
    std::string git;          // converter git hash, recorded in the metadata
    // Per-class formats: "default=q4g64" is layout A; e.g. ",gdn_in=q8g64" puts a class
    // at Q8 (only where a kernel exists). The study's class names (reference/quant_study_qwen4exp.py).
    std::string layout = "default=q4g64";
    int threads = 0;          // 0: all hardware threads
    std::vector<int64_t> layers;  // decoder layers to convert; empty = all
    bool globals = true;          // embedding, LM head, final HC mixer
    bool mtp = true;              // the MTP head
    bool plan_only = false;       // log the plan and stop (reads headers only)
    FILE *console = stderr;       // log sink besides out_dir/convert.log (nullptr: file only)
};

enum class ConvertOp { Q4Merge, Q4FoldHC, Q4ChunkMajor, BF16Rows, F32Widen, I64Copy };
const char *convert_op_name(ConvertOp op);

struct ConvertStep {
    StrixwTensor out;                  // name, encoding, shape, group size, experts, parts
    ConvertOp op = ConvertOp::F32Widen;
    std::vector<std::string> sources;  // Q4Merge / BF16Rows: the parts; Q4FoldHC: down, inject (or ""), norm
    int64_t unit = -1;                 // decoder layer index, -1 globals, -2 MTP (for progress grouping)
};

struct ConvertPlan {
    std::vector<ConvertStep> steps;
    std::vector<std::pair<std::string, std::string>> skipped;  // source tensor, why
    int64_t layers_total = 0;
    bool shared_split = false;  // shared expert written separately (format differs from the routed experts')
};

// Summary of a finished (or planned) run.
struct ConvertResult {
    std::string weights_path;
    uint64_t source_bytes = 0, output_bytes = 0;
    size_t tensors = 0;
    double seconds = 0;
};

ConvertResult convert_qwen4exp(const ConvertOptions &opt);

// The source checkpoint's fingerprint as the converter records it (strixw metadata source_fingerprint): a hash of
// the index JSON and every shard's name, size and header. Files converted from the same checkpoint carry the same.
std::string checkpoint_fingerprint(const std::string &dir);

}  // namespace strix
