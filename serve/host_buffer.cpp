#include "serve/host_buffer.hpp"

#include "common/check.hpp"

#include <sys/mman.h>

#include <cerrno>
#include <cstring>
#include <utility>

namespace strix {

HostBuffer::~HostBuffer() { release(); }

HostBuffer::HostBuffer(HostBuffer &&o) noexcept
    : p_(std::exchange(o.p_, nullptr)), size_(std::exchange(o.size_, 0)), cap_(std::exchange(o.cap_, 0)) {}

HostBuffer &HostBuffer::operator=(HostBuffer &&o) noexcept {
    if (this != &o) {
        release();
        p_ = std::exchange(o.p_, nullptr), size_ = std::exchange(o.size_, 0), cap_ = std::exchange(o.cap_, 0);
    }
    return *this;
}

void HostBuffer::release() {
    if (p_) ::munmap(p_, cap_);  // can't fail for a mapping made here; nothing useful to do if it did
    p_ = nullptr, size_ = 0, cap_ = 0;
}

void HostBuffer::resize(size_t bytes) {
    if (bytes <= cap_) {
        size_ = bytes;
        return;
    }
    constexpr size_t kHuge = size_t{2} << 20;  // 2 MiB: the transparent huge page size on x86-64
    const size_t cap = (bytes + kHuge - 1) / kHuge * kHuge;
    STRIX_CHECK(cap >= bytes, "HostBuffer::resize: ", bytes, " bytes overflows when rounded up to 2 MiB");
    // Populating is not a hint: a failure means the memory isn't there, and the copy into it would fault anyway.
    auto populate = [](void *p, size_t off, size_t len, size_t total) {
        if (::madvise(static_cast<uint8_t *>(p) + off, len, MADV_POPULATE_WRITE) != 0) {
            const int err = errno;
            ::munmap(p, total);
            STRIX_FAIL("HostBuffer::resize: populating ", len, " bytes at ", off, " of a ", total, "-byte mapping failed: ",
                       std::strerror(err), " (MADV_POPULATE_WRITE needs Linux >= 5.14)");
        }
    };
    if (p_) {
        // Growing: mremap keeps the pages already there (the VMA's MADV_NOHUGEPAGE too) and only the new tail is
        // populated. A conversation's state grows 3-80 MB a turn, so re-mapping the whole buffer cost ~100 ms per GB
        // on every large export once the RAM tier sat at its margin (2026-09-30: 2-6 GB exports 278-546 ms median).
        void *p = ::mremap(p_, cap_, cap, MREMAP_MAYMOVE);
        STRIX_CHECK(p != MAP_FAILED, "HostBuffer::resize: growing a ", cap_, "-byte mapping to ", cap, " bytes failed: ",
                    std::strerror(errno));
        const size_t old = cap_;
        p_ = nullptr, size_ = 0, cap_ = 0;  // populate() unmaps on failure: never leave a dangling p_
        populate(p, old, cap - old, cap);
        p_ = static_cast<uint8_t *>(p), size_ = bytes, cap_ = cap;
        return;
    }
    void *p = ::mmap(nullptr, cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    STRIX_CHECK(p != MAP_FAILED, "HostBuffer::resize: mmap of ", cap, " bytes failed: ", std::strerror(errno));
    // 4 KiB pages, not transparent huge pages: with free memory ~0 (the n-gram table in the page cache) a
    // MADV_HUGEPAGE fault compacts in the faulting thread - a fresh 5 GiB buffer 287 ms median but up to 1,094 ms
    // under pressure, and the prompt cache's exports stalled seconds (bench_host_buffer 0bc9472-pressure; the warm
    // switch test). 4 KiB: 517 ms fresh, flat; copies 3-6% slower (66 -> 69 ms per 5 GiB). The prompt cache
    // prefaults and reuses buffers, so the fresh cost stays off the request path.
    (void)::madvise(p, cap, MADV_NOHUGEPAGE);
    populate(p, 0, cap, cap);
    p_ = static_cast<uint8_t *>(p), size_ = bytes, cap_ = cap;
}

}  // namespace strix
