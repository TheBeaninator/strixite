#pragma once

// Owning, move-only device allocation with checked upload/download. Every
// buffer carries a name so allocation and copy errors say which buffer failed.

#include "common/hip_check.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace strix {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(size_t n, std::string name) : n_(n), name_(std::move(name)) {
        STRIX_CHECK(n > 0, "device buffer '", name_, "' requested with 0 elements");
        STRIX_CHECK(n <= SIZE_MAX / sizeof(T), "device buffer '", name_, "': ", n, " elements overflow size_t");
        STRIX_HIP_CHECK(hipMalloc(&ptr_, n * sizeof(T)), "allocating device buffer '", name_, "', ", n, " x ",
                        sizeof(T), " B = ", n * sizeof(T), " bytes");
    }
    static DeviceBuffer from_host(const std::vector<T> &host, std::string name) {
        DeviceBuffer b(host.size(), std::move(name));
        b.upload(host);
        return b;
    }
    ~DeviceBuffer() {
        if (ptr_) (void)hipFree(ptr_);
    }
    DeviceBuffer(DeviceBuffer &&o) noexcept : ptr_(o.ptr_), n_(o.n_), name_(std::move(o.name_)) {
        o.ptr_ = nullptr;
        o.n_ = 0;
    }
    DeviceBuffer &operator=(DeviceBuffer &&o) noexcept {
        if (this != &o) {
            if (ptr_) (void)hipFree(ptr_);
            ptr_ = o.ptr_;
            n_ = o.n_;
            name_ = std::move(o.name_);
            o.ptr_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *get() const {
        STRIX_CHECK(ptr_ != nullptr, "device buffer '", name_, "' is empty (moved-from or default-constructed)");
        return ptr_;
    }
    size_t size() const { return n_; }
    const std::string &name() const { return name_; }

    void upload(const std::vector<T> &host) {
        STRIX_CHECK(host.size() == n_, "upload to '", name_, "': host has ", host.size(), " elements, buffer has ", n_);
        STRIX_HIP_CHECK(hipMemcpy(get(), host.data(), n_ * sizeof(T), hipMemcpyHostToDevice), "upload to '", name_,
                        "', ", n_ * sizeof(T), " bytes");
    }
    std::vector<T> to_host() const {
        std::vector<T> host(n_);
        STRIX_HIP_CHECK(hipMemcpy(host.data(), get(), n_ * sizeof(T), hipMemcpyDeviceToHost), "download of '", name_,
                        "', ", n_ * sizeof(T), " bytes");
        return host;
    }

private:
    T *ptr_ = nullptr;
    size_t n_ = 0;
    std::string name_;
};

// Owning, move-only pinned (page-locked) host allocation, for async host -> device copies. gfx1151 hazard:
// never write it again for the next transfer until the previous copy from it has completed (an event or
// stream sync) - ungated reuse corrupts data.
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;
    PinnedHostBuffer(size_t bytes, std::string name) : bytes_(bytes), name_(std::move(name)) {
        STRIX_CHECK(bytes > 0, "pinned host buffer '", name_, "' requested with 0 bytes");
        STRIX_HIP_CHECK(hipHostMalloc(&ptr_, bytes, hipHostMallocDefault), "allocating pinned host buffer '", name_,
                        "', ", bytes, " bytes");
    }
    ~PinnedHostBuffer() {
        if (ptr_) (void)hipHostFree(ptr_);
    }
    PinnedHostBuffer(PinnedHostBuffer &&o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_), name_(std::move(o.name_)) {
        o.ptr_ = nullptr;
        o.bytes_ = 0;
    }
    PinnedHostBuffer &operator=(PinnedHostBuffer &&o) noexcept {
        if (this != &o) {
            if (ptr_) (void)hipHostFree(ptr_);
            ptr_ = o.ptr_, bytes_ = o.bytes_, name_ = std::move(o.name_);
            o.ptr_ = nullptr, o.bytes_ = 0;
        }
        return *this;
    }
    PinnedHostBuffer(const PinnedHostBuffer &) = delete;
    PinnedHostBuffer &operator=(const PinnedHostBuffer &) = delete;
    void *get() const { return ptr_; }
    size_t size() const { return bytes_; }  // bytes
    const std::string &name() const { return name_; }

private:
    void *ptr_ = nullptr;
    size_t bytes_ = 0;
    std::string name_;
};

}  // namespace strix
