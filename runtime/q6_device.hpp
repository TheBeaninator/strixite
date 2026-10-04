#pragma once

// Q6 weight resident on the device (formats/q6.hpp layout: per row the low then the high code plane), plus the
// non-owning view the kernels take.

#include "formats/q6.hpp"
#include "runtime/device_buffer.hpp"

#include <string>

namespace strix {

struct Q6DeviceView {
    const uint8_t *q = nullptr;  // N rows of 3K/4 bytes: low plane [K/2], high plane [K/4]
    const uint16_t *scale = nullptr;
    const uint16_t *minv = nullptr;
    int64_t N = 0, K = 0, G = 0;
};

struct Q6Device {
    DeviceBuffer<uint8_t> q;
    DeviceBuffer<uint16_t> scale, minv;
    int64_t N = 0, K = 0, G = 0;

    static Q6Device upload(const Q6Weight &w, const std::string &name) {
        check_q6(w, name.c_str());
        Q6Device d;
        d.q = DeviceBuffer<uint8_t>::from_host(w.q, name + ".q");
        d.scale = DeviceBuffer<uint16_t>::from_host(w.scale, name + ".scale");
        d.minv = DeviceBuffer<uint16_t>::from_host(w.minv, name + ".min");
        d.N = w.N, d.K = w.K, d.G = w.G;
        return d;
    }
    Q6DeviceView view() const { return {q.get(), scale.get(), minv.get(), N, K, G}; }
};

}  // namespace strix
