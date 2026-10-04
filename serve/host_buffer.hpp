#pragma once

// Host memory for GiB-sized session states on their way between the GPU and the disk prompt cache
// (the prompt cache's host buffers). A std::vector was the stall: resize() zero-fills and
// faults in 4 KiB pages, most of the ~0.7 s the engine spent exporting an entry after every response at 160-200k
// context. This maps anonymous memory and faults it all in up front, and resize() keeps the mapping when it is big
// enough, so a reused buffer costs nothing but the copy (bench_host_buffer, 6dac2e9: 5 GiB 782 ms with a vector, 287
// fresh with huge pages, 66 reused). 4 KiB pages since 2026-09-29 (host_buffer.cpp: huge pages compact under memory
// pressure; 517 ms fresh). Contents after resize() are unspecified.

#include <cstddef>
#include <cstdint>

namespace strix {

class HostBuffer {
public:
    HostBuffer() = default;
    explicit HostBuffer(size_t bytes) { resize(bytes); }
    ~HostBuffer();
    HostBuffer(HostBuffer &&o) noexcept;
    HostBuffer &operator=(HostBuffer &&o) noexcept;
    HostBuffer(const HostBuffer &) = delete;
    HostBuffer &operator=(const HostBuffer &) = delete;

    // Size it to `bytes`: keeps the mapping if its capacity suffices; grows an existing mapping in place (mremap - only
    // the added pages are populated; it may move); maps a new one if there is none.
    void resize(size_t bytes);
    void release();  // unmaps; size and capacity 0

    uint8_t *data() { return p_; }
    const uint8_t *data() const { return p_; }
    size_t size() const { return size_; }
    size_t capacity() const { return cap_; }
    bool empty() const { return size_ == 0; }

private:
    uint8_t *p_ = nullptr;
    size_t size_ = 0, cap_ = 0;
};

}  // namespace strix
