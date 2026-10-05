#include "runtime/strixw_loader.hpp"

#include "common/check.hpp"
#include "common/hip_check.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

namespace strix {

namespace {

constexpr size_t kStageBytes = 256ull << 20;

std::string shape_str(const std::vector<int64_t> &s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "]";
}

// n bytes at file offset off into dst, split over `threads` concurrent preads (NVMe queue depth).
void pread_parallel(int fd, void *dst, size_t n, uint64_t off, int threads, const std::string &path) {
    std::vector<std::thread> pool;
    std::vector<std::string> errs((size_t)threads);
    const size_t per = (n + (size_t)threads - 1) / (size_t)threads;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            size_t a = std::min(n, (size_t)t * per), b = std::min(n, a + per);
            auto *p = static_cast<uint8_t *>(dst) + a;
            while (a < b) {
                const ssize_t r = ::pread(fd, p, b - a, (off_t)(off + a));
                if (r < 0 && errno == EINTR) continue;
                if (r <= 0) {
                    errs[(size_t)t] = r < 0 ? std::strerror(errno) : "unexpected end of file";
                    return;
                }
                a += (size_t)r, p += r;
            }
        });
    for (std::thread &th : pool) th.join();
    for (const std::string &e : errs)
        STRIX_CHECK(e.empty(), "StrixwDevice: reading '", path, "' at ", off, " (", n, " bytes) failed: ", e);
}

// Byte ranges {file offset, length, offset in dst} into dst: pieces of <= 4 MiB handed to `threads` readers in turn
// (a rank's row slices of a tensor: e.g. 1,026 runs of 320 expert rows each - only the bytes it keeps are read).
struct ReadRange {
    uint64_t off, len, dst;
};
void pread_ranges(int fd, uint8_t *dst, const std::vector<ReadRange> &ranges, int threads, const std::string &path) {
    constexpr uint64_t kPiece = 4ull << 20;
    std::vector<ReadRange> pieces;
    for (const ReadRange &r : ranges)
        for (uint64_t a = 0; a < r.len; a += kPiece) pieces.push_back({r.off + a, std::min(kPiece, r.len - a), r.dst + a});
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    std::vector<std::string> errs((size_t)threads);
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            for (size_t i; (i = next.fetch_add(1)) < pieces.size();) {
                const ReadRange &r = pieces[i];
                uint64_t a = 0;
                while (a < r.len) {
                    const ssize_t n = ::pread(fd, dst + r.dst + a, r.len - a, (off_t)(r.off + a));
                    if (n < 0 && errno == EINTR) continue;
                    if (n <= 0) {
                        errs[(size_t)t] = n < 0 ? std::strerror(errno) : "unexpected end of file";
                        return;
                    }
                    a += (uint64_t)n;
                }
            }
        });
    for (std::thread &th : pool) th.join();
    for (const std::string &e : errs) STRIX_CHECK(e.empty(), "StrixwDevice: reading '", path, "' failed: ", e);
}

}  // namespace

StrixwDevice::StrixwDevice(const std::string &path, int read_threads) {
    STRIX_CHECK(read_threads >= 1 && read_threads <= 64, "StrixwDevice: read_threads = ", read_threads);
    const auto t0 = std::chrono::steady_clock::now();
    file_ = std::make_unique<StrixwFile>(path);
    const uint64_t off = file_->data_offset(), n = file_->file_bytes() - off;
    STRIX_CHECK(n > 0, "StrixwDevice: '", path, "' has an empty data region");
    data_ = DeviceBuffer<uint8_t>((size_t)n, "strixw data region of '" + path + "'");
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    STRIX_CHECK(fd >= 0, "StrixwDevice: open '", path, "' failed: ", std::strerror(errno));
    void *stage[2] = {nullptr, nullptr};
    hipEvent_t done[2] = {nullptr, nullptr};
    hipStream_t stream = nullptr;
    try {
        STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "StrixwDevice: stream");
        for (int i = 0; i < 2; ++i) {
            STRIX_HIP_CHECK(hipHostMalloc(&stage[i], kStageBytes, hipHostMallocDefault), "StrixwDevice: staging ", i);
            STRIX_HIP_CHECK(hipEventCreateWithFlags(&done[i], hipEventDisableTiming), "StrixwDevice: event ", i);
        }
        bool pending[2] = {false, false};
        int cur = 0;
        for (uint64_t at = 0; at < n; at += kStageBytes, cur ^= 1) {
            const size_t len = (size_t)std::min<uint64_t>(kStageBytes, n - at);
            // Gate: this staging buffer's previous copy must be complete before it's overwritten.
            if (pending[cur]) STRIX_HIP_CHECK(hipEventSynchronize(done[cur]), "StrixwDevice: staging gate");
            pread_parallel(fd, stage[cur], len, off + at, read_threads, path);
            STRIX_HIP_CHECK(hipMemcpyAsync(data_.get() + at, stage[cur], len, hipMemcpyHostToDevice, stream),
                            "StrixwDevice: copy at ", at, " (", len, " bytes)");
            STRIX_HIP_CHECK(hipEventRecord(done[cur], stream), "StrixwDevice: event record");
            pending[cur] = true;
        }
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "StrixwDevice: final sync");
    } catch (...) {
        for (int i = 0; i < 2; ++i) {
            if (done[i]) (void)hipEventDestroy(done[i]);
            if (stage[i]) (void)hipHostFree(stage[i]);
        }
        if (stream) (void)hipStreamDestroy(stream);
        ::close(fd);
        throw;
    }
    for (int i = 0; i < 2; ++i) (void)hipEventDestroy(done[i]), (void)hipHostFree(stage[i]);
    (void)hipStreamDestroy(stream);
    (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);  // the device copy is the one kept
    ::close(fd);
    bytes_read_ = n;
    load_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

namespace {

// Bytes per K element of a component of a row-major quantized tensor (codes / scales / mins), as num / den.
std::pair<int64_t, int64_t> bytes_per_k(const StrixwTensor &t, StrixwRole role) {
    const bool q4 = t.encoding == StrixwEncoding::Q4RowMajor, q8 = t.encoding == StrixwEncoding::Q8RowMajor;
    STRIX_CHECK(q4 || q8, "StrixwDevice split: column slice of '", t.name, "' (", strixw_encoding_name(t.encoding),
                "): only Q4 / Q8 row-major");
    if (role == StrixwRole::Q) return {1, q4 ? 2 : 1};
    return {2, t.group_size};  // one BF16 scale / min per group
}

}  // namespace

StrixwDevice::StrixwDevice(const std::string &path, const StrixwSlicePlan &plan, int read_threads) : split_(true) {
    STRIX_CHECK(read_threads >= 1 && read_threads <= 64, "StrixwDevice: read_threads = ", read_threads);
    const auto t0 = std::chrono::steady_clock::now();
    file_ = std::make_unique<StrixwFile>(path);
    // 1. Plan: each kept tensor's sliced shape and component sizes, at 256-aligned offsets of one device region.
    struct Job {
        const StrixwTensor *src;
        StrixwSlice sl;
        bool whole;
    };
    std::vector<Job> jobs;
    uint64_t total = 0;
    for (const StrixwTensor &t : file_->tensors()) {
        std::optional<StrixwSlice> o = plan(t);
        if (o && o->skip) continue;
        Job j{&t, o ? *o : StrixwSlice{}, !o.has_value()};
        StrixwTensor lt = t;
        const int64_t N = t.shape.at(0);
        if (j.sl.rows.empty()) j.sl.rows.push_back({0, N});
        int64_t rows = 0;
        for (auto [a, b] : j.sl.rows) {
            STRIX_CHECK(0 <= a && a < b && b <= N, "StrixwDevice split: '", t.name, "' rows [", a, ", ", b, ") of ", N);
            rows += b - a;
        }
        lt.shape[0] = rows;
        if (j.sl.k1 > 0) {
            STRIX_CHECK(t.shape.size() == 2 && t.group_size > 0 && 0 <= j.sl.k0 && j.sl.k0 < j.sl.k1 &&
                            j.sl.k1 <= t.shape[1] && j.sl.k0 % t.group_size == 0 && j.sl.k1 % t.group_size == 0,
                        "StrixwDevice split: '", t.name, "' columns [", j.sl.k0, ", ", j.sl.k1, ") not on whole groups of ",
                        t.group_size);
            lt.shape[1] = j.sl.k1 - j.sl.k0;
        }
        lt.parts.clear();
        for (StrixwComponent &c : lt.components) {
            const uint64_t row_bytes = c.bytes / (uint64_t)N;
            STRIX_CHECK(row_bytes * (uint64_t)N == c.bytes, "StrixwDevice split: '", t.name, "' component ",
                        strixw_role_name(c.role), " of ", c.bytes, " bytes isn't ", N, " equal rows");
            uint64_t out_row = row_bytes;
            if (j.sl.k1 > 0) {
                const auto [num, den] = bytes_per_k(t, c.role);
                out_row = (uint64_t)((j.sl.k1 - j.sl.k0) * num / den);
            }
            c.bytes = out_row * (uint64_t)rows;
            total = (total + 255) & ~(uint64_t)255;
            dev_off_[{t.name, (int)c.role}] = total;
            total += c.bytes;
        }
        local_[t.name] = std::move(lt);
        jobs.push_back(std::move(j));
    }
    STRIX_CHECK(total > 0, "StrixwDevice split: nothing to load from '", path, "'");
    data_ = DeviceBuffer<uint8_t>((size_t)total, "strixw split region of '" + path + "'");
    // 2. Load: each component read whole, sliced on the host, uploaded through two gated pinned staging buffers.
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    STRIX_CHECK(fd >= 0, "StrixwDevice: open '", path, "' failed: ", std::strerror(errno));
    void *stage[2] = {nullptr, nullptr};
    hipEvent_t done[2] = {nullptr, nullptr};
    hipStream_t stream = nullptr;
    std::vector<uint8_t> src, out;
    try {
        STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "StrixwDevice: stream");
        for (int i = 0; i < 2; ++i) {
            STRIX_HIP_CHECK(hipHostMalloc(&stage[i], kStageBytes, hipHostMallocDefault), "StrixwDevice: staging ", i);
            STRIX_HIP_CHECK(hipEventCreateWithFlags(&done[i], hipEventDisableTiming), "StrixwDevice: event ", i);
        }
        bool pending[2] = {false, false};
        int cur = 0;
        auto upload = [&](const uint8_t *p, uint64_t n, uint64_t dst) {
            for (uint64_t at = 0; at < n; at += kStageBytes, cur ^= 1) {
                const size_t len = (size_t)std::min<uint64_t>(kStageBytes, n - at);
                if (pending[cur]) STRIX_HIP_CHECK(hipEventSynchronize(done[cur]), "StrixwDevice: staging gate");
                std::memcpy(stage[cur], p + at, len);
                STRIX_HIP_CHECK(hipMemcpyAsync(data_.get() + dst + at, stage[cur], len, hipMemcpyHostToDevice, stream),
                                "StrixwDevice: copy");
                STRIX_HIP_CHECK(hipEventRecord(done[cur], stream), "StrixwDevice: event record");
                pending[cur] = true;
            }
        };
        for (const Job &j : jobs) {
            const StrixwTensor &t = *j.src;
            const int64_t N = t.shape[0];
            for (const StrixwComponent &c : t.components) {
                const uint64_t dst = dev_off_.at({t.name, (int)c.role});
                const uint64_t row_bytes = c.bytes / (uint64_t)N;
                if (!j.whole && j.sl.k1 == 0) {  // whole rows: read only this rank's row ranges
                    std::vector<ReadRange> rr;
                    uint64_t at = 0;
                    for (auto [a, b] : j.sl.rows) {
                        rr.push_back({c.offset + (uint64_t)a * row_bytes, (uint64_t)(b - a) * row_bytes, at});
                        at += (uint64_t)(b - a) * row_bytes;
                    }
                    out.resize(at);
                    pread_ranges(fd, out.data(), rr, read_threads, path);
                    bytes_read_ += at;
                    upload(out.data(), at, dst);
                    continue;
                }
                src.resize(c.bytes);
                pread_parallel(fd, src.data(), c.bytes, c.offset, read_threads, path);
                bytes_read_ += c.bytes;
                if (j.whole) {
                    upload(src.data(), c.bytes, dst);
                    continue;
                }
                uint64_t b0 = 0, b1 = row_bytes;
                if (j.sl.k1 > 0) {
                    const auto [num, den] = bytes_per_k(t, c.role);
                    b0 = (uint64_t)(j.sl.k0 * num / den), b1 = (uint64_t)(j.sl.k1 * num / den);
                }
                const uint64_t ob = b1 - b0;
                uint64_t rows = 0;
                for (auto [a, b] : j.sl.rows) rows += (uint64_t)(b - a);
                out.resize(ob * rows);
                uint8_t *w = out.data();
                for (auto [a, b] : j.sl.rows) {
                    if (b0 == 0 && b1 == row_bytes) {  // whole rows: one run
                        std::memcpy(w, src.data() + (uint64_t)a * row_bytes, (uint64_t)(b - a) * row_bytes);
                        w += (uint64_t)(b - a) * row_bytes;
                    } else {
                        for (int64_t r = a; r < b; ++r, w += ob) std::memcpy(w, src.data() + (uint64_t)r * row_bytes + b0, ob);
                    }
                }
                upload(out.data(), out.size(), dst);
            }
        }
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "StrixwDevice: final sync");
    } catch (...) {
        for (int i = 0; i < 2; ++i) {
            if (done[i]) (void)hipEventDestroy(done[i]);
            if (stage[i]) (void)hipHostFree(stage[i]);
        }
        if (stream) (void)hipStreamDestroy(stream);
        ::close(fd);
        throw;
    }
    for (int i = 0; i < 2; ++i) (void)hipEventDestroy(done[i]), (void)hipHostFree(stage[i]);
    (void)hipStreamDestroy(stream);
    (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
    load_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

const StrixwTensor &StrixwDevice::tensor(const std::string &name, StrixwEncoding enc,
                                         const std::vector<int64_t> &shape) const {
    if (split_) {
        const auto it = local_.find(name);
        STRIX_CHECK(it != local_.end(), "StrixwDevice: '", name, "' was not loaded on this rank (split plan)");
        const StrixwTensor &t = it->second;
        STRIX_CHECK(t.encoding == enc && t.shape == shape, "StrixwDevice: '", name, "' (this rank) is ",
                    strixw_encoding_name(t.encoding), " ", shape_str(t.shape), ", expected ", strixw_encoding_name(enc),
                    " ", shape_str(shape));
        return t;
    }
    const StrixwTensor &t = file_->get(name);
    STRIX_CHECK(t.encoding == enc && t.shape == shape, "StrixwDevice: '", name, "' in '", file_->path(), "' is ",
                strixw_encoding_name(t.encoding), " ", shape_str(t.shape), ", expected ", strixw_encoding_name(enc),
                " ", shape_str(shape));
    return t;
}

const void *StrixwDevice::ptr(const StrixwTensor &t, StrixwRole role) const {
    if (split_) {
        const auto it = dev_off_.find({t.name, (int)role});
        STRIX_CHECK(it != dev_off_.end(), "StrixwDevice: '", t.name, "' has no ", strixw_role_name(role), " component here");
        return data_.get() + it->second;
    }
    for (const StrixwComponent &c : t.components)
        if (c.role == role) return data_.get() + (c.offset - file_->data_offset());
    STRIX_CHECK(false, "StrixwDevice: '", t.name, "' has no ", strixw_role_name(role), " component");
    return nullptr;
}

Q4DeviceView StrixwDevice::q4(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = tensor(name, StrixwEncoding::Q4RowMajor, shape);
    return {static_cast<const uint8_t *>(ptr(t, StrixwRole::Q)), static_cast<const uint16_t *>(ptr(t, StrixwRole::Scale)),
            static_cast<const uint16_t *>(ptr(t, StrixwRole::Min)), shape[0], shape[1], t.group_size};
}

Q8DeviceView StrixwDevice::q8(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = tensor(name, StrixwEncoding::Q8RowMajor, shape);
    return {static_cast<const uint8_t *>(ptr(t, StrixwRole::Q)), static_cast<const uint16_t *>(ptr(t, StrixwRole::Scale)),
            static_cast<const uint16_t *>(ptr(t, StrixwRole::Min)), shape[0], shape[1], t.group_size};
}

Q6DeviceView StrixwDevice::q6(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = tensor(name, StrixwEncoding::Q6RowMajor, shape);
    return {static_cast<const uint8_t *>(ptr(t, StrixwRole::Q)), static_cast<const uint16_t *>(ptr(t, StrixwRole::Scale)),
            static_cast<const uint16_t *>(ptr(t, StrixwRole::Min)), shape[0], shape[1], t.group_size};
}

Q5DeviceView StrixwDevice::q5(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = tensor(name, StrixwEncoding::Q5RowMajor, shape);
    return {static_cast<const uint8_t *>(ptr(t, StrixwRole::Q)), static_cast<const uint16_t *>(ptr(t, StrixwRole::Scale)),
            static_cast<const uint16_t *>(ptr(t, StrixwRole::Min)), shape[0], shape[1], t.group_size};
}

QWeightView StrixwDevice::qw(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = file_->get(name);
    switch (t.encoding) {
        case StrixwEncoding::Q4RowMajor: return qweight(q4(name, shape));
        case StrixwEncoding::Q8RowMajor: return qweight(q8(name, shape));
        case StrixwEncoding::Q6RowMajor: return qweight(q6(name, shape));
        case StrixwEncoding::Q5RowMajor: return qweight(q5(name, shape));
        default:
            STRIX_FAIL("StrixwDevice: '", name, "' in '", file_->path(), "' is ", strixw_encoding_name(t.encoding), " ",
                       shape_str(t.shape), ", expected a dense projection: q4, q5, q6 or q8 row-major ", shape_str(shape));
    }
}

Q4ChunkMajorView StrixwDevice::q4_chunk_major(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = tensor(name, StrixwEncoding::Q4ChunkMajor, shape);
    return {static_cast<const uint8_t *>(ptr(t, StrixwRole::Q)), static_cast<const uint16_t *>(ptr(t, StrixwRole::Scale)),
            static_cast<const uint16_t *>(ptr(t, StrixwRole::Min)), shape[0], shape[1], t.group_size};
}

Q8ChunkMajorView StrixwDevice::q8_chunk_major(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = tensor(name, StrixwEncoding::Q8ChunkMajor, shape);
    return {static_cast<const uint8_t *>(ptr(t, StrixwRole::Q)), static_cast<const uint16_t *>(ptr(t, StrixwRole::Scale)),
            static_cast<const uint16_t *>(ptr(t, StrixwRole::Min)), shape[0], shape[1], t.group_size};
}

QChunkMajorView StrixwDevice::qw_chunk_major(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = file_->get(name);
    switch (t.encoding) {
        case StrixwEncoding::Q4ChunkMajor: return qweight(q4_chunk_major(name, shape));
        case StrixwEncoding::Q8ChunkMajor: return qweight(q8_chunk_major(name, shape));
        default:
            STRIX_FAIL("StrixwDevice: '", name, "' in '", file_->path(), "' is ", strixw_encoding_name(t.encoding), " ",
                       shape_str(t.shape), ", expected q4 or q8 chunk-major ", shape_str(shape));
    }
}

const float *StrixwDevice::f32(const std::string &name, const std::vector<int64_t> &shape) const {
    return static_cast<const float *>(ptr(tensor(name, StrixwEncoding::F32, shape), StrixwRole::Data));
}

const uint16_t *StrixwDevice::bf16(const std::string &name, const std::vector<int64_t> &shape) const {
    return static_cast<const uint16_t *>(ptr(tensor(name, StrixwEncoding::BF16, shape), StrixwRole::Data));
}

}  // namespace strix
