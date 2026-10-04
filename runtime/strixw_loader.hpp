#pragma once

// strixw weights resident on the device (formats/strixw.hpp): the file's whole data region is one device
// allocation, filled by concurrent pread() into two pinned staging buffers and async copies, each staging
// buffer reused only after an event says its previous copy finished (the gfx1151 host-buffer hazard:
// reusing a pinned buffer for the next async copy without a gate corrupts data). Tensors keep their file
// offsets, so a component's device pointer is base + (offset - data offset) - alignment carries over (4096
// per tensor, 256 per component). On gfx1151 hipMalloc comes out of GTT (system RAM: 124 GiB of a 128 GB machine).

#include "formats/strixw.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q4_device.hpp"
#include "runtime/q5_device.hpp"
#include "runtime/q6_device.hpp"
#include "runtime/q8_device.hpp"
#include "runtime/qweight.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace strix {

class StrixwDevice {
public:
    // Loads path. Throws on any header/index problem (StrixwFile validates) or I/O / HIP failure. Component
    // hashes aren't re-checked (inspect_strixw --verify does that); the header and index hashes are.
    explicit StrixwDevice(const std::string &path, int read_threads = 8);

    const StrixwFile &file() const { return *file_; }
    double load_seconds() const { return load_seconds_; }
    uint64_t data_bytes() const { return data_.size(); }

    // Device pointers, checked: the tensor must exist with the expected encoding and shape.
    const StrixwTensor &tensor(const std::string &name, StrixwEncoding enc, const std::vector<int64_t> &shape) const;
    const void *ptr(const StrixwTensor &t, StrixwRole role) const;
    Q4DeviceView q4(const std::string &name, const std::vector<int64_t> &shape) const;
    Q8DeviceView q8(const std::string &name, const std::vector<int64_t> &shape) const;
    Q6DeviceView q6(const std::string &name, const std::vector<int64_t> &shape) const;
    Q5DeviceView q5(const std::string &name, const std::vector<int64_t> &shape) const;
    // A dense projection stored as Q4, Q5, Q6 or Q8 row-major, whichever the file has (anything else throws).
    QWeightView qw(const std::string &name, const std::vector<int64_t> &shape) const;
    Q4ChunkMajorView q4_chunk_major(const std::string &name, const std::vector<int64_t> &shape) const;
    Q8ChunkMajorView q8_chunk_major(const std::string &name, const std::vector<int64_t> &shape) const;
    // A chunk-major weight stored as Q4 or Q8, whichever the file has (anything else throws).
    QChunkMajorView qw_chunk_major(const std::string &name, const std::vector<int64_t> &shape) const;
    const float *f32(const std::string &name, const std::vector<int64_t> &shape) const;
    const uint16_t *bf16(const std::string &name, const std::vector<int64_t> &shape) const;

private:
    std::unique_ptr<StrixwFile> file_;
    DeviceBuffer<uint8_t> data_;
    double load_seconds_ = 0;
};

}  // namespace strix
