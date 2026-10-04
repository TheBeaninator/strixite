#pragma once

// Token embedding lookup: out[t, :] = table[ids[t], :].
// table is BF16 (raw bits) [vocab, d] on device, as stored in the checkpoint;
// out is F32 or BF16 [n_tokens, d]; ids are int32 on device.
//
// A kernel can't throw, and clamping a bad id would be a silent default
// (never allowed here). Instead an out-of-range id zeroes that row and records the
// first bad (position, id) in a device-side EmbeddingError; the caller must
// pass it to check_embedding_error() at its next sync point, which throws
// naming both.

#include "kernels/norm.hpp"  // Act
#include "runtime/q8_device.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::kernels {

// Device-resident; initialise with reset_embedding_error() before use.
// (position << 32 | uint32(id)) of the lowest bad position, updated with
// atomicMin so the result doesn't depend on block scheduling order; one
// word keeps the id paired with its position. ~0ull = no error.
struct EmbeddingError {
    unsigned long long packed;
};

void reset_embedding_error(EmbeddingError *err_dev, hipStream_t stream);

// copies > 1 writes each token's row copies times, side by side: copy c of token t at out + t * out_ts + c * d
// (out_ts >= copies * d; 0 = copies * d) - the embedding broadcast into the hyper-connection streams in one launch.
void embedding_lookup(const int32_t *ids, int64_t n_tokens, const uint16_t *table, int64_t vocab, int64_t d, void *out,
                      Act act, EmbeddingError *err_dev, hipStream_t stream, int64_t copies = 1, int64_t out_ts = 0);

void embedding_lookup(const int32_t *ids, int64_t n_tokens, const Q8DeviceView &table, void *out,
                      Act act, EmbeddingError *err_dev, hipStream_t stream, int64_t copies = 1, int64_t out_ts = 0);

// Synchronizes `stream`, reads the error slot, throws if a bad id was seen.
void check_embedding_error(const EmbeddingError *err_dev, int64_t vocab, hipStream_t stream);
// The same check on a slot already on the host (read back with the forward's one sync).
void report_embedding_error(const EmbeddingError &e, int64_t vocab);

}  // namespace strix::kernels
