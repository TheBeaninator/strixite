#include "formats/ngram_table.hpp"

#include "common/check.hpp"
#include "formats/strixw.hpp"  // strix_hash64

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <vector>

namespace strix {

namespace {

constexpr char kMagic[8] = {'S', 'T', 'R', 'X', 'N', 'G', 'R', 'M'};
// Header layout (little endian; x86-64 host, memcpy is the encoding):
//   0 magic[8]  8 u32 version  12 u32 header bytes  16 u64 rows  24 u32 row_dim  28 u32 dtype  32 u64 data offset
//   40 u64 file bytes  48 i64 eos  56 i64 multipliers[3]  80 i64 vocab[16]  208 i64 offsets[16]
//   336 char source_fingerprint[32]  368 char converter_git[32]  400 f32 fp8_scale (v2; 0 otherwise)
//   404..504 zero  504 u64 hash of bytes 0..504
constexpr size_t kHashAt = 504;

void put_str(uint8_t *at, const std::string &s, const char *what) {
    STRIX_CHECK(s.size() < 32, "ngram table: ", what, " '", s, "' is ", s.size(), " chars, at most 31");
    std::memcpy(at, s.data(), s.size());
}
std::string get_str(const uint8_t *at) { return std::string(reinterpret_cast<const char *>(at), strnlen(reinterpret_cast<const char *>(at), 31)); }

std::vector<uint8_t> encode(const NgramTableInfo &in) {
    std::vector<uint8_t> h(kNgramTableHeaderBytes, 0);
    auto put = [&](size_t off, auto v) { std::memcpy(h.data() + off, &v, sizeof(v)); };
    std::memcpy(h.data(), kMagic, 8);
    put(8, kNgramTableVersion), put(12, (uint32_t)kNgramTableHeaderBytes), put(16, (uint64_t)in.rows);
    put(24, (uint32_t)in.row_dim), put(28, (uint32_t)in.dtype), put(32, kNgramTableDataOffset), put(40, in.file_bytes());
    put(48, in.params.eos);
    for (int i = 0; i < 3; ++i) put(56 + 8 * i, in.params.multipliers[(size_t)i]);
    for (int i = 0; i < 16; ++i) put(80 + 8 * i, in.params.vocab[(size_t)i]), put(208 + 8 * i, in.params.offset[(size_t)i]);
    put_str(h.data() + 336, in.source_fingerprint, "source fingerprint");
    put_str(h.data() + 368, in.converter_git, "converter git");
    put(400, in.fp8_scale);
    put(kHashAt, strix_hash64(h.data(), kHashAt));
    return h;
}

void check_info(const NgramTableInfo &in, const std::string &where) {
    STRIX_CHECK(in.rows >= 1 && in.rows <= (1ll << 40), where, ": rows = ", in.rows, ", expected 1..2^40");
    STRIX_CHECK(in.row_dim >= 1 && in.row_dim <= 65536, where, ": row_dim = ", in.row_dim, ", expected 1..65536");
    STRIX_CHECK(in.dtype == NgramDtype::BF16 || in.dtype == NgramDtype::FP8E4M3 || in.dtype == NgramDtype::Q8Row, where,
                ": dtype ", (uint32_t)in.dtype, ", expected 1 (BF16), 2 (FP8E4M3) or 3 (Q8Row)");
    if (in.dtype == NgramDtype::FP8E4M3)
        STRIX_CHECK(std::isfinite(in.fp8_scale) && in.fp8_scale > 0, where, ": FP8 table scale ", in.fp8_scale,
                    ", expected finite > 0");
    else
        STRIX_CHECK(in.fp8_scale == 0, where, ": fp8_scale ", in.fp8_scale, " set on a ", ngram_dtype_name(in.dtype), " table");
    STRIX_CHECK(in.params.table_rows == in.rows, where, ": hashing constants say ", in.params.table_rows,
                " table rows, the table has ", in.rows);
}

void pwrite_all(int fd, const void *p, size_t n, uint64_t at, const std::string &where) {
    const auto *b = static_cast<const uint8_t *>(p);
    while (n > 0) {
        const ssize_t w = ::pwrite(fd, b, n, (off_t)at);
        if (w < 0 && errno == EINTR) continue;
        STRIX_CHECK(w > 0, where, ": write of ", n, " bytes at ", at, " failed: ", w < 0 ? std::strerror(errno) : "0 bytes");
        b += w, n -= (size_t)w, at += (uint64_t)w;
    }
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t f32_to_bf16(float f) {  // round to nearest even (no NaN here: decoded values are finite)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

}  // namespace

const char *ngram_dtype_name(NgramDtype d) {
    switch (d) {
    case NgramDtype::BF16: return "bf16";
    case NgramDtype::FP8E4M3: return "fp8";
    case NgramDtype::Q8Row: return "q8";
    }
    return "?";
}

NgramDtype parse_ngram_dtype(const std::string &s) {
    if (s == "bf16") return NgramDtype::BF16;
    if (s == "fp8") return NgramDtype::FP8E4M3;
    if (s == "q8") return NgramDtype::Q8Row;
    STRIX_FAIL("n-gram table dtype '", s, "', expected bf16, fp8 or q8");
}

uint64_t NgramTableInfo::row_bytes() const {
    switch (dtype) {
    case NgramDtype::BF16: return (uint64_t)row_dim * 2;
    case NgramDtype::FP8E4M3: return (uint64_t)row_dim;
    case NgramDtype::Q8Row: return (uint64_t)row_dim + 2;
    }
    STRIX_FAIL("NgramTableInfo::row_bytes: dtype ", (uint32_t)dtype);
}

uint8_t f32_to_e4m3(float x) {
    const uint8_t sign = std::signbit(x) ? 0x80 : 0;
    const float a = std::fabs(x);
    if (a < 0.015625f) {  // below 2^-6: subnormal, m * 2^-9 (m = 8 rounds up into the smallest normal, code 8)
        return sign | (uint8_t)std::nearbyint(a * 512.0f);
    }
    int ex;
    std::frexp(a, &ex);
    int e = ex - 1;                                               // a = 1.f * 2^e
    int m = (int)std::nearbyint((std::ldexp(a, -e) - 1.0f) * 8);  // exact before rounding: nearest even
    if (m == 8) m = 0, ++e;
    if (e > 8 || (e == 8 && m > 6)) e = 8, m = 6;  // saturate at 448 (1.75 * 2^8); S.1111.111 is NaN
    return sign | (uint8_t)(((e + 7) << 3) | m);
}

float e4m3_to_f32(uint8_t code) {
    const int e = (code >> 3) & 15, m = code & 7;
    const float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + (float)m / 8, e - 7);
    return code & 0x80 ? -v : v;
}

void ngram_encode_rows(const NgramTableInfo &info, const uint16_t *bf16, int64_t n, uint8_t *out) {
    STRIX_CHECK(bf16 && out && n >= 0, "ngram_encode_rows: bf16 = ", (const void *)bf16, ", out = ", (void *)out, ", n = ", n);
    const int64_t D = info.row_dim;
    switch (info.dtype) {
    case NgramDtype::BF16: std::memcpy(out, bf16, (size_t)(n * D) * 2); return;
    case NgramDtype::FP8E4M3: {
        STRIX_CHECK(info.fp8_scale > 0, "ngram_encode_rows: FP8 scale ", info.fp8_scale);
        const float inv = 1.0f / info.fp8_scale;
        for (int64_t i = 0; i < n * D; ++i) out[i] = f32_to_e4m3(bf16_to_f32(bf16[i]) * inv);
        return;
    }
    case NgramDtype::Q8Row:
        for (int64_t r = 0; r < n; ++r) {
            const uint16_t *x = bf16 + r * D;
            uint8_t *o = out + r * (D + 2);
            float amax = 0;
            for (int64_t k = 0; k < D; ++k) amax = std::max(amax, std::fabs(bf16_to_f32(x[k])));
            const uint16_t sb = f32_to_bf16(amax / 127.0f);
            const float s = bf16_to_f32(sb);
            std::memcpy(o, &sb, 2);
            for (int64_t k = 0; k < D; ++k) {
                const float q = s > 0 ? std::nearbyint(bf16_to_f32(x[k]) / s) : 0.0f;
                o[2 + k] = (uint8_t)(int8_t)std::max(-127.0f, std::min(127.0f, q));
            }
        }
        return;
    }
    STRIX_FAIL("ngram_encode_rows: dtype ", (uint32_t)info.dtype);
}

void ngram_decode_row(const NgramTableInfo &info, const uint8_t *in, uint16_t *bf16) {
    const int64_t D = info.row_dim;
    switch (info.dtype) {
    case NgramDtype::BF16: std::memcpy(bf16, in, (size_t)D * 2); return;
    case NgramDtype::FP8E4M3:
        for (int64_t k = 0; k < D; ++k) bf16[k] = f32_to_bf16(e4m3_to_f32(in[k]) * info.fp8_scale);
        return;
    case NgramDtype::Q8Row: {
        uint16_t sb;
        std::memcpy(&sb, in, 2);
        const float s = bf16_to_f32(sb);
        for (int64_t k = 0; k < D; ++k) bf16[k] = f32_to_bf16((float)(int8_t)in[2 + k] * s);
        return;
    }
    }
    STRIX_FAIL("ngram_decode_row: dtype ", (uint32_t)info.dtype);
}

NgramTableWriter::NgramTableWriter(const std::string &path, const NgramTableInfo &info)
    : path_(path), partial_(path + ".partial"), info_(info) {
    check_info(info, "NgramTableWriter '" + path + "'");
    STRIX_CHECK(!std::filesystem::exists(path), "NgramTableWriter: '", path, "' exists; refusing to overwrite");
    fd_ = ::open(partial_.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    STRIX_CHECK(fd_ >= 0, "NgramTableWriter: create '", partial_, "' failed: ", std::strerror(errno));
    // Compression off while the file is empty (btrfs applies FS_NOCOMP_FL to data written afterwards).
    int flags = 0;
    if (::ioctl(fd_, FS_IOC_GETFLAGS, &flags) == 0) {
        flags |= FS_NOCOMP_FL;
        nocomp_ = ::ioctl(fd_, FS_IOC_SETFLAGS, &flags) == 0;
        if (!nocomp_) nocomp_note_ = std::string("FS_IOC_SETFLAGS(FS_NOCOMP_FL) failed: ") + std::strerror(errno);
    } else {
        nocomp_note_ = std::string("filesystem has no inode flags (FS_IOC_GETFLAGS: ") + std::strerror(errno) +
                       ") - no per-file compression to turn off";
    }
}

NgramTableWriter::~NgramTableWriter() {
    if (fd_ >= 0) ::close(fd_);
    if (!finished_) std::filesystem::remove(partial_);  // an unfinished table is never left looking finished
}

void NgramTableWriter::write_rows(const uint16_t *rows, int64_t n) {
    STRIX_CHECK(!finished_, "NgramTableWriter::write_rows after finish()");
    STRIX_CHECK(rows != nullptr && n >= 1, "NgramTableWriter::write_rows: rows = ", (const void *)rows, ", n = ", n);
    STRIX_CHECK(written_ + n <= info_.rows, "NgramTableWriter::write_rows: ", written_, " + ", n, " rows exceed the ",
                info_.rows, " planned");
    std::vector<uint8_t> enc((size_t)n * info_.row_bytes());
    ngram_encode_rows(info_, rows, n, enc.data());
    pwrite_all(fd_, enc.data(), enc.size(), kNgramTableDataOffset + (uint64_t)written_ * info_.row_bytes(),
               "NgramTableWriter '" + partial_ + "'");
    written_ += n;
}

void NgramTableWriter::write_encoded(const uint8_t *bytes, int64_t n) {
    STRIX_CHECK(!finished_, "NgramTableWriter::write_encoded after finish()");
    STRIX_CHECK(bytes != nullptr && n >= 1, "NgramTableWriter::write_encoded: bytes = ", (const void *)bytes, ", n = ", n);
    STRIX_CHECK(written_ + n <= info_.rows, "NgramTableWriter::write_encoded: ", written_, " + ", n, " rows exceed the ",
                info_.rows, " planned");
    pwrite_all(fd_, bytes, (size_t)n * info_.row_bytes(), kNgramTableDataOffset + (uint64_t)written_ * info_.row_bytes(),
               "NgramTableWriter '" + partial_ + "'");
    written_ += n;
}

void NgramTableWriter::finish() {
    STRIX_CHECK(!finished_, "NgramTableWriter::finish called twice");
    STRIX_CHECK(written_ == info_.rows, "NgramTableWriter::finish: ", written_, " rows written of ", info_.rows);
    const std::vector<uint8_t> h = encode(info_);
    pwrite_all(fd_, h.data(), h.size(), 0, "NgramTableWriter header '" + partial_ + "'");
    STRIX_CHECK(::fsync(fd_) == 0, "NgramTableWriter: fsync '", partial_, "' failed: ", std::strerror(errno));
    ::close(fd_);
    fd_ = -1;
    std::filesystem::rename(partial_, path_);
    finished_ = true;
}

NgramTableFile::NgramTableFile(const std::string &path) : path_(path) {
    const std::string where = "ngram table '" + path + "'";
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    STRIX_CHECK(fd_ >= 0, where, ": open failed: ", std::strerror(errno));
    try {
        std::vector<uint8_t> h(kNgramTableHeaderBytes);
        const ssize_t got = ::pread(fd_, h.data(), h.size(), 0);
        STRIX_CHECK(got == (ssize_t)h.size(), where, ": header read returned ", got, " of ", h.size(), " bytes");
        auto rd = [&](size_t off, auto v) {
            std::memcpy(&v, h.data() + off, sizeof(v));
            return v;
        };
        STRIX_CHECK(std::memcmp(h.data(), kMagic, 8) == 0, where, ": not an n-gram table (bad magic)");
        const uint32_t version = rd(8, uint32_t{}), hsize = rd(12, uint32_t{});
        STRIX_CHECK(version == 1 || version == kNgramTableVersion, where, ": version ", version, ", this build reads 1..",
                    kNgramTableVersion);
        STRIX_CHECK(hsize == kNgramTableHeaderBytes, where, ": header size ", hsize, ", expected ", kNgramTableHeaderBytes);
        STRIX_CHECK(rd(kHashAt, uint64_t{}) == strix_hash64(h.data(), kHashAt), where, ": header hash mismatch (corrupt header)");
        info_.rows = (int64_t)rd(16, uint64_t{}), info_.row_dim = rd(24, uint32_t{});
        info_.dtype = (NgramDtype)rd(28, uint32_t{});
        info_.params.eos = rd(48, int64_t{});
        for (int i = 0; i < 3; ++i) info_.params.multipliers[(size_t)i] = rd(56 + 8 * i, int64_t{});
        for (int i = 0; i < 16; ++i)
            info_.params.vocab[(size_t)i] = rd(80 + 8 * i, int64_t{}), info_.params.offset[(size_t)i] = rd(208 + 8 * i, int64_t{});
        info_.params.table_rows = info_.rows;
        info_.source_fingerprint = get_str(h.data() + 336), info_.converter_git = get_str(h.data() + 368);
        info_.fp8_scale = version >= 2 ? rd(400, float{}) : 0.0f;
        STRIX_CHECK(version >= 2 || info_.dtype == NgramDtype::BF16, where, ": a version 1 table must be BF16");
        check_info(info_, where);
        STRIX_CHECK(rd(32, uint64_t{}) == kNgramTableDataOffset, where, ": data offset ", rd(32, uint64_t{}), ", expected ",
                    kNgramTableDataOffset);
        struct stat st;
        STRIX_CHECK(::fstat(fd_, &st) == 0, where, ": fstat failed: ", std::strerror(errno));
        STRIX_CHECK(rd(40, uint64_t{}) == info_.file_bytes() && (uint64_t)st.st_size == info_.file_bytes(), where,
                    ": ", info_.rows, " rows of ", info_.row_bytes(), " bytes make ", info_.file_bytes(),
                    " bytes; the header says ", rd(40, uint64_t{}), ", the file has ", (uint64_t)st.st_size,
                    " (truncated or appended to)");
    } catch (...) {
        ::close(fd_);
        throw;
    }
}

NgramTableFile::~NgramTableFile() {
    if (fd_ >= 0) ::close(fd_);
}

}  // namespace strix
