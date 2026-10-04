#pragma once

// PLE n-gram table rows straight from the HF checkpoint's shards - the interim source from before the converted
// table file (runtime/ngram_table), kept for tests. Reads the 16 BF16 rows of 160 per token
// straight from the checkpoint's 128 `ngram_embedding.shard_N` tensors with pread (~320 B each; one 4 KiB
// page read per row through the page cache), deduplicated and 32 at a time. Correct and exact - the rows are the
// checkpoint's own values - but it needs the HF checkpoint next to the converted weights and does no caching. It's the
// only PLE row source until the table is designed; the designed one replaces it.

#include "runtime/ngram_table.hpp"  // NgramRowSource
#include "runtime/ple_hash.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strix {

class NgramRowsFromShards : public NgramRowSource {
public:
    // dir: the HF checkpoint; layer_prefix e.g. "model.language_model.layers.1.ple.ple_embedding.".
    NgramRowsFromShards(const std::string &dir, const std::string &layer_prefix);
    ~NgramRowsFromShards();
    NgramRowsFromShards(const NgramRowsFromShards &) = delete;
    NgramRowsFromShards &operator=(const NgramRowsFromShards &) = delete;

    const PleHashParams &hash_params() const override { return params_; }
    int64_t row_dim() const override { return row_dim_; }  // 160
    // rows [n] table row ids (ple_ngram_ids output) -> out [n, row_dim] BF16 bits.
    void gather(const int64_t *rows, int64_t n, uint16_t *out) const override;

private:
    struct Shard {
        int fd = -1;
        uint64_t offset = 0;  // the tensor's first byte in its file
        std::string path;
    };
    std::vector<Shard> shards_;
    int64_t rows_per_shard_ = 0, row_dim_ = 0;
    PleHashParams params_;
};

}  // namespace strix
