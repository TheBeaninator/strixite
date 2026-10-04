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
    load_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

const StrixwTensor &StrixwDevice::tensor(const std::string &name, StrixwEncoding enc,
                                         const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = file_->get(name);
    STRIX_CHECK(t.encoding == enc && t.shape == shape, "StrixwDevice: '", name, "' in '", file_->path(), "' is ",
                strixw_encoding_name(t.encoding), " ", shape_str(t.shape), ", expected ", strixw_encoding_name(enc),
                " ", shape_str(shape));
    return t;
}

const void *StrixwDevice::ptr(const StrixwTensor &t, StrixwRole role) const {
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
