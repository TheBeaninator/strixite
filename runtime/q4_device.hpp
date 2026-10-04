#pragma once

// Q4 weight resident on the device (formats/q4.hpp layout), plus the
// non-owning view the kernels take.

#include "formats/q4.hpp"
#include "runtime/device_buffer.hpp"

#include <string>

namespace strix {

struct Q4DeviceView {
    const uint8_t *q = nullptr;
    const uint16_t *scale = nullptr;
    const uint16_t *minv = nullptr;
    int64_t N = 0, K = 0, G = 0;
};

struct Q4Device {
    DeviceBuffer<uint8_t> q;
    DeviceBuffer<uint16_t> scale, minv;
    int64_t N = 0, K = 0, G = 0;

    static Q4Device upload(const Q4Weight &w, const std::string &name) {
        check_q4(w, name.c_str());
        Q4Device d;
        d.q = DeviceBuffer<uint8_t>::from_host(w.q, name + ".q");
        d.scale = DeviceBuffer<uint16_t>::from_host(w.scale, name + ".scale");
        d.minv = DeviceBuffer<uint16_t>::from_host(w.minv, name + ".min");
        d.N = w.N, d.K = w.K, d.G = w.G;
        return d;
    }
    Q4DeviceView view() const { return {q.get(), scale.get(), minv.get(), N, K, G}; }
};

// Chunk-major Q4 (formats/q4.hpp Q4ChunkMajor) on the device.
struct Q4ChunkMajorView {
    const uint8_t *q = nullptr;
    const uint16_t *scale = nullptr;
    const uint16_t *minv = nullptr;
    int64_t N = 0, K = 0, G = 0;
};

struct Q4ChunkMajorDevice {
    DeviceBuffer<uint8_t> q;
    DeviceBuffer<uint16_t> scale, minv;
    int64_t N = 0, K = 0, G = 0;

    static Q4ChunkMajorDevice upload(const Q4ChunkMajor &w, const std::string &name) {
        STRIX_CHECK(w.N >= 1 && w.K >= 32 && w.K % 32 == 0 && q4_group_size_supported(w.G) && w.K % w.G == 0 &&
                        w.q.size() == (size_t)(w.N * w.K / 2) && w.scale.size() == (size_t)(w.N * (w.K / w.G)) &&
                        w.minv.size() == w.scale.size(),
                    name, ": inconsistent chunk-major Q4 (N=", w.N, ", K=", w.K, ", G=", w.G, ", ", w.q.size(),
                    " code bytes, ", w.scale.size(), "/", w.minv.size(), " scales/mins)");
        Q4ChunkMajorDevice d;
        d.q = DeviceBuffer<uint8_t>::from_host(w.q, name + ".q");
        d.scale = DeviceBuffer<uint16_t>::from_host(w.scale, name + ".scale");
        d.minv = DeviceBuffer<uint16_t>::from_host(w.minv, name + ".min");
        d.N = w.N, d.K = w.K, d.G = w.G;
        return d;
    }
    Q4ChunkMajorView view() const { return {q.get(), scale.get(), minv.get(), N, K, G}; }
};

}  // namespace strix
