#pragma once

// Q8 weight resident on the device (formats/q8.hpp layout), plus the non-owning view the kernels take.

#include "formats/q8.hpp"
#include "runtime/device_buffer.hpp"

#include <string>

namespace strix {

struct Q8DeviceView {
    const uint8_t *q = nullptr;
    const uint16_t *scale = nullptr;
    const uint16_t *minv = nullptr;
    int64_t N = 0, K = 0, G = 0;
};

struct Q8Device {
    DeviceBuffer<uint8_t> q;
    DeviceBuffer<uint16_t> scale, minv;
    int64_t N = 0, K = 0, G = 0;

    static Q8Device upload(const Q8Weight &w, const std::string &name) {
        check_q8(w, name.c_str());
        Q8Device d;
        d.q = DeviceBuffer<uint8_t>::from_host(w.q, name + ".q");
        d.scale = DeviceBuffer<uint16_t>::from_host(w.scale, name + ".scale");
        d.minv = DeviceBuffer<uint16_t>::from_host(w.minv, name + ".min");
        d.N = w.N, d.K = w.K, d.G = w.G;
        return d;
    }
    Q8DeviceView view() const { return {q.get(), scale.get(), minv.get(), N, K, G}; }
};

// Chunk-major Q8 (formats/q8.hpp Q8ChunkMajor) on the device.
struct Q8ChunkMajorView {
    const uint8_t *q = nullptr;
    const uint16_t *scale = nullptr;
    const uint16_t *minv = nullptr;
    int64_t N = 0, K = 0, G = 0;
};

struct Q8ChunkMajorDevice {
    DeviceBuffer<uint8_t> q;
    DeviceBuffer<uint16_t> scale, minv;
    int64_t N = 0, K = 0, G = 0;

    static Q8ChunkMajorDevice upload(const Q8ChunkMajor &w, const std::string &name) {
        STRIX_CHECK(w.N >= 1 && w.K >= 32 && w.K % 32 == 0 && (w.G == 32 || w.G == 64 || w.G == 128) &&
                        w.K % w.G == 0 && w.q.size() == (size_t)(w.N * w.K) &&
                        w.scale.size() == (size_t)(w.N * (w.K / w.G)) && w.minv.size() == w.scale.size(),
                    name, ": inconsistent chunk-major Q8 (N=", w.N, ", K=", w.K, ", G=", w.G, ", ", w.q.size(),
                    " code bytes, ", w.scale.size(), "/", w.minv.size(), " scales/mins)");
        Q8ChunkMajorDevice d;
        d.q = DeviceBuffer<uint8_t>::from_host(w.q, name + ".q");
        d.scale = DeviceBuffer<uint16_t>::from_host(w.scale, name + ".scale");
        d.minv = DeviceBuffer<uint16_t>::from_host(w.minv, name + ".min");
        d.N = w.N, d.K = w.K, d.G = w.G;
        return d;
    }
    Q8ChunkMajorView view() const { return {q.get(), scale.get(), minv.get(), N, K, G}; }
};

}  // namespace strix
