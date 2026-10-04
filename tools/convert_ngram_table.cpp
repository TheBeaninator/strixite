// Offline converter for the PLE n-gram table: the HF checkpoint's 128 `ngram_embedding.shard_N` tensors (BF16, rows
// of 160) -> <out>/ngram.table (formats/ngram_table.hpp), BF16 exact, rows
// in table order, compression off for the file. The hashing constants and the checkpoint fingerprint go in the
// header. Afterwards re-reads a sample of rows through the finished file and compares them with the source.
//
// Usage: convert_ngram_table --src DIR --out DIR --git HASH [--layer 1]

#include "common/args.hpp"
#include "common/check.hpp"
#include "formats/convert_qwen4exp.hpp"
#include "formats/ngram_table.hpp"
#include "formats/safetensors.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace strix;
namespace fs = std::filesystem;

namespace {

constexpr int kShards = 128;
constexpr int64_t kEos = 248044;  // config.json eos_token_id (PLE step 1)
constexpr size_t kChunkBytes = 64u << 20;

void pread_all(int fd, void *p, size_t n, uint64_t at, const std::string &what) {
    auto *b = static_cast<uint8_t *>(p);
    while (n > 0) {
        const ssize_t got = ::pread(fd, b, n, (off_t)at);
        if (got < 0 && errno == EINTR) continue;
        STRIX_CHECK(got > 0, "convert_ngram_table: read of ", what, " failed: ", got < 0 ? std::strerror(errno) : "end of file");
        b += got, n -= (size_t)got, at += (uint64_t)got;
    }
}

int run(int argc, char **argv) {
    std::string src, out, git;
    int64_t layer = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char *flag) {
            STRIX_CHECK(i + 1 < argc, flag, " needs a value");
            return std::string(argv[++i]);
        };
        if (a == "--src") src = value("--src");
        else if (a == "--out") out = value("--out");
        else if (a == "--git") git = value("--git");
        else if (a == "--layer") layer = parse_int(value("--layer"), "--layer", 0, 1000);
        else STRIX_FAIL("convert_ngram_table: unknown argument '", a, "'");
    }
    STRIX_CHECK(!src.empty() && !out.empty() && !git.empty(), "usage: convert_ngram_table --src DIR --out DIR --git HASH [--layer 1]");
    const auto t0 = std::chrono::steady_clock::now();
    auto el = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    const std::string P = "model.language_model.layers." + std::to_string(layer) + ".ple.ple_embedding.";

    // Locate the shards and the hashing constants.
    struct ShardSrc {
        std::string path;
        uint64_t offset = 0, bytes = 0;
        int64_t rows = 0, dim = 0;
    };
    std::vector<ShardSrc> shards(kShards);
    std::vector<int64_t> mult, vocab, offs;
    for (const auto &e : fs::directory_iterator(src)) {
        if (e.path().extension() != ".safetensors") continue;
        const SafetensorsFile f(e.path().string());
        auto i64 = [&](const std::string &name, std::vector<int64_t> &v) {
            const TensorInfo *t = f.find(P + name);
            if (!t) return;
            STRIX_CHECK(t->dtype == Dtype::I64, "convert_ngram_table: '", P + name, "' is ", dtype_name(t->dtype));
            v.resize((size_t)t->numel());
            std::memcpy(v.data(), f.data(*t), t->byte_length);
        };
        i64("layer_multipliers", mult), i64("ngram_heads_vocab_sizes", vocab), i64("ngram_heads_offsets", offs);
        for (int s = 0; s < kShards; ++s) {
            const TensorInfo *t = f.find(P + "ngram_embedding.shard_" + std::to_string(s) + ".weight");
            if (!t) continue;
            STRIX_CHECK(t->dtype == Dtype::BF16 && t->shape.size() == 2, "convert_ngram_table: shard ", s, " is ",
                        dtype_name(t->dtype), " with ", t->shape.size(), " dims, expected BF16 [rows, dim]");
            STRIX_CHECK(shards[(size_t)s].path.empty(), "convert_ngram_table: shard ", s, " found twice");
            shards[(size_t)s] = {f.path(), t->byte_offset, t->byte_length, t->shape[0], t->shape[1]};
        }
    }
    for (int s = 0; s < kShards; ++s) {
        STRIX_CHECK(!shards[(size_t)s].path.empty(), "convert_ngram_table: shard ", s, " of ", kShards, " ('", P,
                    "ngram_embedding.shard_", s, ".weight') not found in '", src, "'");
        STRIX_CHECK(shards[(size_t)s].rows == shards[0].rows && shards[(size_t)s].dim == shards[0].dim,
                    "convert_ngram_table: shard ", s, " is [", shards[(size_t)s].rows, ", ", shards[(size_t)s].dim,
                    "], shard 0 [", shards[0].rows, ", ", shards[0].dim, "]");
    }
    STRIX_CHECK(mult.size() == 3 && vocab.size() == 16 && offs.size() == 16, "convert_ngram_table: hashing constants under '",
                P, "': ", mult.size(), " multipliers (expected 3), ", vocab.size(), " vocab sizes, ", offs.size(),
                " offsets (expected 16 each)");
    NgramTableInfo info;
    info.rows = shards[0].rows * kShards, info.row_dim = shards[0].dim;
    std::copy(mult.begin(), mult.end(), info.params.multipliers.begin());
    std::copy(vocab.begin(), vocab.end(), info.params.vocab.begin());
    std::copy(offs.begin(), offs.end(), info.params.offset.begin());
    info.params.eos = kEos, info.params.table_rows = info.rows;
    std::fprintf(stderr, "[%7.1fs] fingerprinting the checkpoint ...\n", el());
    info.source_fingerprint = checkpoint_fingerprint(src);
    info.converter_git = git;
    std::fprintf(stderr, "[%7.1fs] %lld rows x %lld (BF16) = %.2f GiB from %d shards, fingerprint %s\n", el(),
                 (long long)info.rows, (long long)info.row_dim, (double)info.rows * info.row_bytes() / (1ull << 30), kShards,
                 info.source_fingerprint.c_str());

    fs::create_directories(out);
    const std::string path = out + "/ngram.table";
    NgramTableWriter w(path, info);
    std::fprintf(stderr, "[%7.1fs] writing %s - compression %s\n", el(), path.c_str(),
                 w.compression_disabled() ? "disabled for the file (FS_NOCOMP_FL)" : w.compression_note().c_str());
    std::vector<uint16_t> buf(kChunkBytes / 2);
    const uint64_t rb = info.row_bytes(), rows_per_chunk = kChunkBytes / rb;
    for (int s = 0; s < kShards; ++s) {
        const ShardSrc &sh = shards[(size_t)s];
        const int fd = ::open(sh.path.c_str(), O_RDONLY | O_CLOEXEC);
        STRIX_CHECK(fd >= 0, "convert_ngram_table: open '", sh.path, "' failed: ", std::strerror(errno));
        for (int64_t r0 = 0; r0 < sh.rows; r0 += (int64_t)rows_per_chunk) {
            const int64_t n = std::min<int64_t>((int64_t)rows_per_chunk, sh.rows - r0);
            pread_all(fd, buf.data(), (size_t)n * rb, sh.offset + (uint64_t)r0 * rb, "shard " + std::to_string(s));
            w.write_rows(buf.data(), n);
        }
        (void)::posix_fadvise(fd, (off_t)sh.offset, (off_t)sh.bytes, POSIX_FADV_DONTNEED);  // don't keep the source
        ::close(fd);
        if (s % 16 == 15)
            std::fprintf(stderr, "[%7.1fs]   %d/%d shards, %.1f GiB written\n", el(), s + 1, kShards,
                         (double)w.rows_written() * rb / (1ull << 30));
    }
    std::fprintf(stderr, "[%7.1fs] header + fsync ...\n", el());
    w.finish();

    // Verify a sample through the finished file (a fresh reader: header, size, then rows vs the source).
    const NgramTableFile tf(path);
    std::mt19937_64 rng(7);
    std::vector<uint16_t> a((size_t)info.row_dim), b((size_t)info.row_dim);
    for (int i = 0; i < 4096; ++i) {
        const int64_t r = i == 0 ? 0 : i == 1 ? info.rows - 1 : (int64_t)(rng() % (uint64_t)info.rows);
        const ShardSrc &sh = shards[(size_t)(r / shards[0].rows)];
        const int fd = ::open(sh.path.c_str(), O_RDONLY | O_CLOEXEC);
        STRIX_CHECK(fd >= 0, "convert_ngram_table: reopen '", sh.path, "': ", std::strerror(errno));
        pread_all(fd, a.data(), rb, sh.offset + (uint64_t)(r % shards[0].rows) * rb, "verify source row");
        ::close(fd);
        pread_all(tf.fd(), b.data(), rb, kNgramTableDataOffset + (uint64_t)r * rb, "verify table row");
        STRIX_CHECK(a == b, "convert_ngram_table: verify: row ", r, " differs from the source");
    }
    std::fprintf(stderr, "[%7.1fs] done: %s, %.2f GiB, 4096 sampled rows verified against the source\n", el(),
                 path.c_str(), (double)info.file_bytes() / (1ull << 30));
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
