#pragma once

// ngram.table: the PLE n-gram embedding table as its own flat file (the table stays on SSD and
// rows are read on demand). A 512-byte hashed header, then at data_offset
// (4096) every row back to back - row r's row_dim values at data_offset + r * row_bytes - so a row is one
// positioned read. The header carries the PLE hashing constants and the source checkpoint's fingerprint, so a
// table pairs with the weights converted from the same checkpoint. Written by tools/convert_ngram_table with
// filesystem compression turned off for the file (a random 320-byte read from a zstd btrfs extent decompresses
// up to 128 KiB).

#include "runtime/ple_hash.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

namespace strix {

// Version 2 added the 8-bit row formats and the FP8 table scale; version 1 files (BF16) still read.
constexpr uint32_t kNgramTableVersion = 2;
constexpr uint64_t kNgramTableHeaderBytes = 512, kNgramTableDataOffset = 4096;
// Row formats (the values are tiny - |x| < 0.05 - so 8-bit formats need a
// scale):
//   BF16    - row_dim BF16 values, exact.
//   FP8E4M3 - row_dim E4M3 (fn: max 448, no inf) codes of x / fp8_scale, one scale for the whole table
//             (absmax / 448); round to nearest even, saturating.
//   Q8Row   - a BF16 scale s = bf16(absmax(row) / 127), then row_dim int8 codes q = clamp(rne(x / s), -127, 127).
// Rows decode to BF16 on the host (value in FP32, rounded to BF16 once): the forward sees BF16 rows either way.
enum class NgramDtype : uint32_t { BF16 = 1, FP8E4M3 = 2, Q8Row = 3 };
const char *ngram_dtype_name(NgramDtype d);
NgramDtype parse_ngram_dtype(const std::string &s);  // "bf16" | "fp8" | "q8"; throws otherwise

struct NgramTableInfo {
    int64_t rows = 0, row_dim = 0;
    NgramDtype dtype = NgramDtype::BF16;
    float fp8_scale = 0;              // FP8E4M3 only: value = code * fp8_scale
    PleHashParams params;             // table_rows == rows
    std::string source_fingerprint;   // <= 31 chars (the converter's checkpoint fingerprint)
    std::string converter_git;        // <= 31 chars
    uint64_t row_bytes() const;
    uint64_t file_bytes() const { return kNgramTableDataOffset + (uint64_t)rows * row_bytes(); }
};

// E4M3 (fn) codes: round to nearest even, saturating to +-448 (NaN / inf inputs are the caller's to refuse).
uint8_t f32_to_e4m3(float x);
float e4m3_to_f32(uint8_t code);
// n BF16 rows -> n rows in info's format (row_bytes() each) / one stored row -> row_dim BF16 values.
void ngram_encode_rows(const NgramTableInfo &info, const uint16_t *bf16, int64_t n, uint8_t *out);
void ngram_decode_row(const NgramTableInfo &info, const uint8_t *in, uint16_t *bf16);

// Streams the rows in order into path + ".partial" (compression off where the filesystem supports turning it
// off - see compression_disabled()), then finish() writes the header, syncs and renames to path. Refuses to
// overwrite an existing path.
class NgramTableWriter {
public:
    NgramTableWriter(const std::string &path, const NgramTableInfo &info);
    ~NgramTableWriter();
    NgramTableWriter(const NgramTableWriter &) = delete;
    NgramTableWriter &operator=(const NgramTableWriter &) = delete;
    // The next n rows (n * row_dim BF16 values), encoded in the table's format.
    void write_rows(const uint16_t *rows, int64_t n);
    // The next n rows already in the table's format (n * row_bytes() bytes, ngram_encode_rows) - for encoding on
    // several threads.
    void write_encoded(const uint8_t *bytes, int64_t n);
    void finish();
    int64_t rows_written() const { return written_; }
    // Whether the filesystem accepted "no compression" for the file (btrfs FS_NOCOMP_FL), and why not if not.
    bool compression_disabled() const { return nocomp_; }
    const std::string &compression_note() const { return nocomp_note_; }

private:
    std::string path_, partial_;
    NgramTableInfo info_;
    int fd_ = -1;
    int64_t written_ = 0;
    bool finished_ = false, nocomp_ = false;
    std::string nocomp_note_;
};

// Opens and validates a table (magic, version, header size and hash, dtype, the hashing constants, and the file
// size against rows x row bytes). The data is read by runtime/ngram_table (positioned reads on fd()).
class NgramTableFile {
public:
    explicit NgramTableFile(const std::string &path);
    ~NgramTableFile();
    NgramTableFile(const NgramTableFile &) = delete;
    NgramTableFile &operator=(const NgramTableFile &) = delete;
    const NgramTableInfo &info() const { return info_; }
    const std::string &path() const { return path_; }
    int fd() const { return fd_; }

private:
    std::string path_;
    NgramTableInfo info_;
    int fd_ = -1;
};

}  // namespace strix
