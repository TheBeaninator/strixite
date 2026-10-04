// transcode_ngram_table: a BF16 ngram.table (tools/convert_ngram_table) -> the same table in an 8-bit row format
// (formats/ngram_table.hpp: fp8 = E4M3 with one table scale, q8 = int8 with a BF16 scale per row). Streams the
// input in chunks; FP8 first scans it for the table's absmax (scale = absmax / 448). Encodes on every core, then
// verifies a sample of rows read back through the runtime reader: decode(encode(row)) bit for bit, and reports the
// relative row error against the BF16 rows (median / p99 / max) as a JSON line on stdout.
//
// Usage: transcode_ngram_table --in PATH/ngram.table --out DIR --dtype fp8|q8 --git HASH

#include "common/check.hpp"
#include "formats/ngram_table.hpp"
#include "runtime/ngram_table.hpp"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>

using namespace strix;

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

float bf(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

void pread_all(int fd, void *p, size_t n, uint64_t at, const std::string &where) {
    auto *b = static_cast<uint8_t *>(p);
    while (n > 0) {
        const ssize_t got = ::pread(fd, b, n, (off_t)at);
        if (got < 0 && errno == EINTR) continue;
        STRIX_CHECK(got > 0, where, ": read of ", n, " bytes at ", at, " failed: ", got < 0 ? std::strerror(errno) : "end of file");
        b += got, n -= (size_t)got, at += (uint64_t)got;
    }
}

// fn(first_row, rows) over [0, total) in chunks of `chunk`, on `threads` threads.
template <typename Fn>
void for_chunks(int64_t total, int64_t chunk, unsigned threads, Fn fn) {
    std::vector<std::thread> ts;
    const int64_t per = (total + threads - 1) / threads;
    for (unsigned k = 0; k < threads; ++k)
        ts.emplace_back([&, k] {
            const int64_t a = std::min(total, (int64_t)k * per), b = std::min(total, a + per);
            for (int64_t r = a; r < b; r += chunk) fn(r, std::min(chunk, b - r));
        });
    for (auto &t : ts) t.join();
}

int run(int argc, char **argv) {
    std::string in_path, out_dir, dtype_s, git;
    STRIX_CHECK(argc % 2 == 1, "usage: transcode_ngram_table --in PATH --out DIR --dtype fp8|q8 --git HASH");
    for (int i = 1; i < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--in") in_path = v;
        else if (k == "--out") out_dir = v;
        else if (k == "--dtype") dtype_s = v;
        else if (k == "--git") git = v;
        else STRIX_FAIL("transcode_ngram_table: unknown option '", k, "'");
    }
    STRIX_CHECK(!in_path.empty() && !out_dir.empty() && !dtype_s.empty() && !git.empty(),
                "usage: transcode_ngram_table --in PATH --out DIR --dtype fp8|q8 --git HASH");
    const NgramDtype dtype = parse_ngram_dtype(dtype_s);
    STRIX_CHECK(dtype != NgramDtype::BF16, "transcode_ngram_table: --dtype bf16 is the input format; use fp8 or q8");
    const NgramTableFile in(in_path);
    STRIX_CHECK(in.info().dtype == NgramDtype::BF16, "transcode_ngram_table: '", in_path, "' is ",
                ngram_dtype_name(in.info().dtype), ", expected a BF16 table");
    const int64_t R = in.info().rows, D = in.info().row_dim;
    const unsigned threads = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
    constexpr int64_t kChunk = 16384;
    const double t0 = now_s();

    NgramTableInfo out = in.info();
    out.dtype = dtype, out.converter_git = git;
    if (dtype == NgramDtype::FP8E4M3) {
        std::vector<float> amax(threads, 0.f);
        std::atomic<unsigned> slot{0};
        std::atomic<int64_t> bad{-1};
        for_chunks(R, kChunk, threads, [&](int64_t r0, int64_t n) {
            thread_local unsigned me = slot++;
            thread_local std::vector<uint16_t> buf;
            buf.resize((size_t)(n * D));
            pread_all(in.fd(), buf.data(), buf.size() * 2, kNgramTableDataOffset + (uint64_t)r0 * D * 2, in_path);
            float m = amax[me];
            for (size_t i = 0; i < buf.size(); ++i) {
                const float x = bf(buf[i]);
                if (!std::isfinite(x)) bad = r0 + (int64_t)(i / (size_t)D);
                m = std::max(m, std::fabs(x));
            }
            amax[me] = m;
        });
        STRIX_CHECK(bad < 0, "transcode_ngram_table: row ", bad.load(), " holds a non-finite value");
        const float a = *std::max_element(amax.begin(), amax.end());
        STRIX_CHECK(a > 0, "transcode_ngram_table: the table is all zeros");
        out.fp8_scale = a / 448.0f;
        std::fprintf(stderr, "transcode_ngram_table: absmax %.9g -> FP8 scale %.9g (%.1f s)\n", a, out.fp8_scale, now_s() - t0);
    }

    std::filesystem::create_directories(out_dir);
    const std::string out_path = out_dir + "/ngram.table";
    NgramTableWriter w(out_path, out);
    if (!w.compression_disabled()) std::fprintf(stderr, "transcode_ngram_table: note: %s\n", w.compression_note().c_str());
    // Encode chunk by chunk: each chunk's rows split over the threads, then written in order.
    constexpr int64_t kBig = 1 << 20;  // rows per written chunk (~160 MiB of FP8)
    std::vector<uint16_t> src((size_t)(kBig * D));
    std::vector<uint8_t> enc((size_t)kBig * out.row_bytes());
    for (int64_t r0 = 0; r0 < R; r0 += kBig) {
        const int64_t n = std::min(kBig, R - r0);
        for_chunks(n, kChunk, threads, [&](int64_t a, int64_t m) {
            pread_all(in.fd(), src.data() + a * D, (size_t)(m * D) * 2, kNgramTableDataOffset + (uint64_t)(r0 + a) * D * 2, in_path);
            ngram_encode_rows(out, src.data() + a * D, m, enc.data() + (size_t)a * out.row_bytes());
        });
        w.write_encoded(enc.data(), n);
        if ((r0 / kBig) % 32 == 0)
            std::fprintf(stderr, "transcode_ngram_table: %lld / %lld rows (%.0f s)\n", (long long)(r0 + n), (long long)R, now_s() - t0);
    }
    w.finish();
    const double t_write = now_s() - t0;

    // Verify: sampled rows through the runtime reader = decode(encode(BF16 row)); row errors vs BF16.
    const NgramTableRows check(out_path, 0);
    std::mt19937_64 rng(7);
    std::vector<int64_t> ids{0, R - 1};
    for (int i = 0; i < 200000; ++i) ids.push_back((int64_t)(rng() % (uint64_t)R));
    std::vector<uint16_t> got(ids.size() * (size_t)D), row((size_t)D), want((size_t)D);
    check.gather(ids.data(), (int64_t)ids.size(), got.data());
    std::vector<uint8_t> e(out.row_bytes());
    std::vector<double> rel;
    for (size_t k = 0; k < ids.size(); ++k) {
        pread_all(in.fd(), row.data(), (size_t)D * 2, kNgramTableDataOffset + (uint64_t)ids[k] * D * 2, in_path);
        ngram_encode_rows(out, row.data(), 1, e.data());
        ngram_decode_row(out, e.data(), want.data());
        STRIX_CHECK(std::equal(want.begin(), want.end(), got.begin() + (int64_t)k * D), "transcode_ngram_table: verify: row ",
                    ids[k], " read back differs from decode(encode(row))");
        double num = 0, den = 0;
        for (int64_t d = 0; d < D; ++d) {
            const double x = bf(row[(size_t)d]), y = bf(got[(size_t)k * D + (size_t)d]);
            num += (x - y) * (x - y), den += x * x;
        }
        rel.push_back(den > 0 ? std::sqrt(num / den) : 0);
    }
    std::sort(rel.begin(), rel.end());
    const auto q = [&](double f) { return rel[(size_t)(f * (double)(rel.size() - 1))]; };
    std::printf("{\"git\":\"%s\",\"test\":\"transcode_ngram_table\",\"dtype\":\"%s\",\"rows\":%lld,\"row_dim\":%lld,"
                "\"fp8_scale\":%.9g,\"bytes\":%llu,\"seconds\":%.1f,\"sampled_rows\":%zu,\"rel_row_err_median\":%.5f,"
                "\"rel_row_err_p99\":%.5f,\"rel_row_err_max\":%.5f}\n",
                git.c_str(), ngram_dtype_name(dtype), (long long)R, (long long)D, out.fp8_scale,
                (unsigned long long)out.file_bytes(), t_write, rel.size(), q(0.5), q(0.99), rel.back());
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "transcode_ngram_table: %s\n", e.what());
        return 1;
    }
}
