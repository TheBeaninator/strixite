#include "runtime/ngram_rows.hpp"

#include "common/check.hpp"
#include "formats/safetensors.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <exception>
#include <filesystem>
#include <thread>

namespace strix {

namespace {
constexpr int64_t kEos = 248044;  // config.json eos_token_id (PLE step 1)
constexpr int kShards = 128;
constexpr size_t kReadThreads = 32;  // concurrent row reads per gather (SSD queue depth)
}  // namespace

NgramRowsFromShards::NgramRowsFromShards(const std::string &dir, const std::string &P) {
    namespace fs = std::filesystem;
    STRIX_CHECK(fs::is_directory(dir), "NgramRowsFromShards: '", dir, "' is not a directory (the PLE rows still come ",
                "from the HF checkpoint - the table file is OPEN)");
    shards_.resize(kShards);
    std::vector<int64_t> mult, vocab, offs;
    for (const auto &e : fs::directory_iterator(dir)) {
        if (e.path().extension() != ".safetensors") continue;
        const SafetensorsFile f(e.path().string());
        auto i64 = [&](const std::string &name, std::vector<int64_t> &out) {
            const TensorInfo *t = f.find(P + name);
            if (!t) return;
            STRIX_CHECK(t->dtype == Dtype::I64, "NgramRowsFromShards: '", P + name, "' is ", dtype_name(t->dtype));
            out.resize((size_t)t->numel());
            std::memcpy(out.data(), f.data(*t), t->byte_length);
        };
        i64("layer_multipliers", mult), i64("ngram_heads_vocab_sizes", vocab), i64("ngram_heads_offsets", offs);
        for (int s = 0; s < kShards; ++s) {
            const TensorInfo *t = f.find(P + "ngram_embedding.shard_" + std::to_string(s) + ".weight");
            if (!t) continue;
            STRIX_CHECK(t->dtype == Dtype::BF16 && t->shape.size() == 2, "NgramRowsFromShards: shard ", s, " is ",
                        dtype_name(t->dtype), " with ", t->shape.size(), " dims, expected BF16 [rows, 160]");
            STRIX_CHECK(rows_per_shard_ == 0 || (t->shape[0] == rows_per_shard_ && t->shape[1] == row_dim_),
                        "NgramRowsFromShards: shard ", s, " shape differs from the others");
            rows_per_shard_ = t->shape[0], row_dim_ = t->shape[1];
            Shard &sh = shards_[(size_t)s];
            STRIX_CHECK(sh.fd < 0, "NgramRowsFromShards: shard ", s, " found twice");
            sh.fd = ::open(f.path().c_str(), O_RDONLY | O_CLOEXEC);
            STRIX_CHECK(sh.fd >= 0, "NgramRowsFromShards: open '", f.path(), "' failed: ", std::strerror(errno));
            sh.offset = t->byte_offset, sh.path = f.path();
        }
    }
    for (int s = 0; s < kShards; ++s)
        STRIX_CHECK(shards_[(size_t)s].fd >= 0, "NgramRowsFromShards: shard ", s, " of ", kShards, " ('", P,
                    "ngram_embedding.shard_", s, ".weight') not found in '", dir, "'");
    STRIX_CHECK(mult.size() == 3 && vocab.size() == 16 && offs.size() == 16, "NgramRowsFromShards: hashing constants ",
                "under '", P, "': ", mult.size(), " multipliers (expected 3), ", vocab.size(), " vocab sizes and ",
                offs.size(), " offsets (expected 16 each)");
    std::copy(mult.begin(), mult.end(), params_.multipliers.begin());
    std::copy(vocab.begin(), vocab.end(), params_.vocab.begin());
    std::copy(offs.begin(), offs.end(), params_.offset.begin());
    params_.eos = kEos;
    params_.table_rows = (int64_t)kShards * rows_per_shard_;
    check_ple_hash_params(params_);
}

NgramRowsFromShards::~NgramRowsFromShards() {
    for (const Shard &s : shards_)
        if (s.fd >= 0) ::close(s.fd);
}

// The rows are random 320-byte reads across 95 GiB of shards: one at a time they cost ~0.25 ms each cold (a 512-token
// prefill chunk wants 8192 of them - ~2 s, rocprofv3 + wall clock at depth 8192, git d9dc5cc). So: sort and
// dedupe the ids (repeats are common - the hash heads share rows across tokens), fetch the unique rows with up to
// kReadThreads concurrent preads (queue depth for the SSD), then scatter to the output order.
void NgramRowsFromShards::gather(const int64_t *rows, int64_t n, uint16_t *out) const {
    STRIX_CHECK((rows && out) || n == 0, "NgramRowsFromShards::gather: null pointer (rows=", (const void *)rows,
                ", out=", (void *)out, ") with n = ", n);
    STRIX_CHECK(n >= 0, "NgramRowsFromShards::gather: n = ", n);
    for (int64_t i = 0; i < n; ++i)
        STRIX_CHECK(rows[i] >= 0 && rows[i] < params_.table_rows, "NgramRowsFromShards::gather: row id ", rows[i],
                    " (entry ", i, ") outside the table's ", params_.table_rows, " rows");
    std::vector<int64_t> uniq(rows, rows + n);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    const size_t rb = (size_t)row_dim_ * 2;
    std::vector<uint16_t> buf(uniq.size() * (size_t)row_dim_);
    auto read_one = [&](size_t u) {
        const int64_t r = uniq[u];
        const Shard &s = shards_[(size_t)(r / rows_per_shard_)];
        const uint64_t at = s.offset + (uint64_t)(r % rows_per_shard_) * rb;
        auto *p = reinterpret_cast<uint8_t *>(buf.data() + u * (size_t)row_dim_);
        size_t left = rb, done = 0;
        while (left > 0) {
            const ssize_t got = ::pread(s.fd, p + done, left, (off_t)(at + done));
            if (got < 0 && errno == EINTR) continue;
            STRIX_CHECK(got > 0, "NgramRowsFromShards::gather: pread of row ", r, " from '", s.path, "' failed: ",
                        got < 0 ? std::strerror(errno) : "end of file");
            done += (size_t)got, left -= (size_t)got;
        }
    };
    parallel_rows(uniq.size(), kReadThreads, read_one);
    for (int64_t i = 0; i < n; ++i) {
        const size_t u = (size_t)(std::lower_bound(uniq.begin(), uniq.end(), rows[i]) - uniq.begin());
        std::memcpy(out + i * row_dim_, buf.data() + u * (size_t)row_dim_, rb);
    }
}

}  // namespace strix
