#include "runtime/weight_loader.hpp"

#include "common/hip_check.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>

namespace strix {

namespace {

// kDirectIOAlign (4096) covers virtually all local filesystems/block sizes
// in practice; querying a block device's logical sector size via
// ioctl(BLKSSZGET) doesn't apply to a regular file on a filesystem. If it's
// ever wrong for a given machine, open()/pread() fail loudly with EINVAL
// rather than silently corrupting data.
constexpr size_t kAlign = kDirectIOAlign;
constexpr int kMaxWorkers = 256;

size_t round_up(size_t n, size_t align) { return (n + align - 1) / align * align; }

// Reads [start, end) of fd into buf, retrying on short reads. end may
// exceed the file's actual size (the buffer is over-allocated to
// direct_io_buffer_size()); a pread() whose requested range crosses EOF just
// returns fewer bytes than asked, which is fine as long as [start, real_eof)
// is fully covered - real_eof is enforced via file_size.
void read_range(const std::string &path, int fd, unsigned char *buf, size_t start, size_t end, size_t file_size,
                bool direct) {
    STRIX_CHECK(start <= end, "bad range [", start, ", ", end, ") for '", path, "'");
    size_t done = start;
    size_t must_reach = end < file_size ? end : file_size;
    while (done < must_reach) {
        ssize_t got = pread(fd, buf + done, end - done, static_cast<off_t>(done));
        STRIX_CHECK(got >= 0, "pread", direct ? " (O_DIRECT)" : "", " of '", path, "' failed at offset ", done,
                    " (range [", start, ", ", end, "), ", end - done, " bytes requested): ", std::strerror(errno),
                    direct && errno == EINVAL ? " - buffer/offset/length alignment wrong for this filesystem?" : "");
        STRIX_CHECK(got > 0, "unexpected EOF reading '", path, "' at offset ", done, " of expected size ", file_size,
                    " (file shrank while reading?)");
        done += static_cast<size_t>(got);
    }
}

}  // namespace

size_t direct_io_buffer_size(size_t file_size) { return round_up(file_size, kAlign); }

void read_file_parallel(const std::string &path, void *buf, size_t file_size, bool direct, int num_workers) {
    STRIX_CHECK(!path.empty(), "path is empty");
    STRIX_CHECK(buf != nullptr, "destination buffer for '", path, "' is null");
    STRIX_CHECK(file_size > 0, "file_size for '", path, "' is 0 - nothing to read (wrong size passed?)");
    STRIX_CHECK(num_workers >= 1 && num_workers <= kMaxWorkers, "num_workers = ", num_workers, " for '", path,
                "', expected 1..", kMaxWorkers);
    STRIX_CHECK(!direct || reinterpret_cast<uintptr_t>(buf) % kAlign == 0, "O_DIRECT buffer for '", path, "' at ", buf,
                " is not ", kAlign, "-byte aligned");

    int fd = open(path.c_str(), O_RDONLY | (direct ? O_DIRECT : 0));
    STRIX_CHECK(fd >= 0, "open", direct ? "(O_DIRECT)" : "", " failed on '", path, "': ", std::strerror(errno),
                direct && errno == EINVAL ? " - filesystem may not support O_DIRECT" : "");
    struct stat st{};
    int stat_rc = fstat(fd, &st);
    if (stat_rc != 0 || (size_t)st.st_size != file_size) close(fd);
    STRIX_CHECK(stat_rc == 0, "fstat failed on '", path, "'");
    STRIX_CHECK((size_t)st.st_size == file_size, "'", path, "' is ", (size_t)st.st_size,
                " bytes on disk but the caller expects ", file_size, " (stale metadata or wrong file?)");

    // pread() with an explicit offset doesn't touch the shared file
    // position, so concurrent threads sharing one fd is safe (POSIX).
    // Aligned, roughly-equal, disjoint ranges - what actually drives NVMe
    // queue depth; a single synchronous O_DIRECT read (no kernel readahead
    // to overlap with) undersells the drive.
    size_t span = direct_io_buffer_size(file_size);
    auto *bytes = static_cast<unsigned char *>(buf);
    size_t chunk = round_up((span + (size_t)num_workers - 1) / (size_t)num_workers, kAlign);
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors(num_workers);
    std::exception_ptr spawn_error;
    try {
        for (int i = 0; i < num_workers; ++i) {
            size_t start = (size_t)i * chunk;
            if (start >= span) break;
            size_t end = start + chunk < span ? start + chunk : span;
            workers.emplace_back([&, i, start, end] {
                try {
                    read_range(path, fd, bytes, start, end, file_size, direct);
                } catch (...) {
                    errors[i] = std::current_exception();
                }
            });
        }
    } catch (...) {
        spawn_error = std::current_exception();  // join what started before rethrowing, or std::terminate
    }
    for (auto &w : workers) w.join();
    close(fd);
    if (spawn_error) std::rethrow_exception(spawn_error);
    for (auto &e : errors)
        if (e) std::rethrow_exception(e);
}

void drop_page_cache(const std::string &path) {
    STRIX_CHECK(!path.empty(), "path is empty");
    int fd = open(path.c_str(), O_RDONLY);
    STRIX_CHECK(fd >= 0, "cannot open '", path, "' to drop its page cache: ", std::strerror(errno));
    int sync_rc = fdatasync(fd);
    int sync_errno = errno;
    int advise_rc = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);  // returns the error number directly
    close(fd);
    STRIX_CHECK(sync_rc == 0, "fdatasync of '", path, "' failed: ", std::strerror(sync_errno));
    STRIX_CHECK(advise_rc == 0, "posix_fadvise(DONTNEED) of '", path, "' failed: ", std::strerror(advise_rc));
}

const char *load_mode_name(LoadMode m) {
    switch (m) {
        case LoadMode::Buffered: return "buffered";
        case LoadMode::Direct: return "direct";
    }
    return "?";
}

WeightLoader::WeightLoader(const SafetensorsFile &sf, LoadMode mode, int num_workers) {
    STRIX_CHECK(mode == LoadMode::Buffered || mode == LoadMode::Direct, "unknown load mode ", (int)mode, " for '",
                sf.path(), "'");
    file_size_ = sf.file_size();
    STRIX_CHECK(file_size_ > 0, "'", sf.path(), "' reports size 0");
    alloc_size_ = direct_io_buffer_size(file_size_);
    STRIX_HIP_CHECK(hipHostMalloc(&pinned_, alloc_size_), "pinned buffer for '", sf.path(), "', ", alloc_size_,
                    " bytes (", (double)alloc_size_ / (1 << 30), " GiB)");
    try {
        read_file_parallel(sf.path(), pinned_, file_size_, mode == LoadMode::Direct, num_workers);
        drop_page_cache(sf.path());  // the pinned buffer is the copy kept
    } catch (...) {
        (void)hipHostFree(pinned_);
        pinned_ = nullptr;
        throw;
    }
    path_ = sf.path();
}

WeightLoader::~WeightLoader() {
    if (pinned_) (void)hipHostFree(pinned_);
}

const void *WeightLoader::data(const TensorInfo &t) const {
    STRIX_CHECK(pinned_ != nullptr, "loader for '", path_, "' holds no data");
    STRIX_CHECK(t.byte_offset + t.byte_length <= file_size_, "tensor '", t.name, "' (bytes ", t.byte_offset, "..",
                t.byte_offset + t.byte_length, ") is outside the ", file_size_, " bytes loaded from '", path_,
                "' - TensorInfo from a different file?");
    return static_cast<const unsigned char *>(pinned_) + t.byte_offset;
}

}  // namespace strix
