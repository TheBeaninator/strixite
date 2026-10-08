#pragma once

// Each logits row's top kLogitCands candidates on the GPU, so a forward ends with ~160 bytes a row copied to the host
// instead of the whole row (248,320 floats, ~1 MB) scanned on the CPU - the sampler's top-k path needs only these
// (serve/sampler.hpp, Sampler::sample_candidates). Order: value descending, the lower id first on a tie - the order
// the sampler's own one-pass list keeps - so sampling from the candidates is bit-identical to sampling from the row.
// Only ids [0, n_valid) count (the LM head is padded past the tokenizer). NaN logits are left out and flagged per row
// (the sampler refuses a row with any NaN, as on the full row). A row with fewer than kLogitCands logits fills the
// rest with (-inf, INT32_MAX).
//
// Measured before building: after every MTP verify the GPU idled ~0.3 ms
// (bench) while the host copied and scanned the rows; Sampler on one cold row 0.059 ms (bench_sampler 738810b
// cold variants, gfx1151).
//
// A workgroup reduces 8,192 inputs to their best kLogitCands by rank counting (a threshold from the threads' bests, then
// each candidate's place = the candidates better than it - no sequential picking); levels repeat it on the lists:
// 2 launches for 248,077 logits (31 workgroups, then 1). Details in logits_topk.hip.

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::kernels {

constexpr int kLogitCands = 20;
constexpr int64_t kLogitsTopkMaxValid = 1 << 24;  // ids stay well inside int32

struct LogitCand {
    float v;
    int32_t id;
};
static_assert(sizeof(LogitCand) == 8, "LogitCand is copied to the host as 8 bytes");

size_t logits_topk_workspace_bytes(int64_t rows, int64_t n_valid);

// logits [rows, ld] FP32 on the device; n_valid <= ld, 1 <= n_valid <= kLogitsTopkMaxValid, rows >= 1.
// out [rows, kLogitCands], nan [rows] (1 if the row has a NaN among its first n_valid logits), both on the device.
// workspace: >= logits_topk_workspace_bytes(rows, n_valid) bytes, no initialisation needed.
// mask (structured output: response_format): null, or [rows, mask_words] uint32 on the device - bit
// (id % 32) of word (id / 32) of a row set = id allowed. Disallowed ids are left out like padding (a row with fewer
// allowed ids than kLogitCands fills the rest with (-inf, INT32_MAX)); a NaN is flagged whether its id is allowed or
// not. mask_words >= ceil(n_valid / 32); 0 when mask is null. Applied in the first level only, where logits are read.
void logits_topk(const float *logits, int64_t rows, int64_t ld, int64_t n_valid, LogitCand *out, uint32_t *nan,
                 void *workspace, size_t workspace_bytes, hipStream_t stream, const uint32_t *mask = nullptr,
                 int64_t mask_words = 0);

}  // namespace strix::kernels
