#pragma once

// Hand-rolled reader for the safetensors format (HF safetensors are read
// directly, no GGUF). Format: 8-byte little-endian header
// length, then a JSON header (tensor name -> {dtype, shape, data_offsets}),
// then raw tensor bytes. data_offsets are [start, end) relative to the
// first byte after the header. Read-only, mmap'd — no copy of tensor data.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strix {

enum class Dtype {
    F64, F32, F16, BF16,
    I64, I32, I16, I8, U8,
    Bool,
    Unknown,
};

const char *dtype_name(Dtype d);
size_t dtype_size(Dtype d);  // bytes per element; 0 for Unknown

struct TensorInfo {
    std::string name;
    Dtype dtype = Dtype::Unknown;
    std::vector<int64_t> shape;
    size_t byte_offset = 0;  // absolute offset into the mapped file
    size_t byte_length = 0;

    int64_t numel() const;
};

class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::string &path);
    ~SafetensorsFile();
    SafetensorsFile(const SafetensorsFile &) = delete;
    SafetensorsFile &operator=(const SafetensorsFile &) = delete;

    const std::vector<TensorInfo> &tensors() const { return tensors_; }
    const TensorInfo *find(const std::string &name) const;  // nullptr if absent
    const TensorInfo &get(const std::string &name) const;   // throws, naming the file, if absent
    // Pointer into the mapping at t.byte_offset; valid for this file's lifetime.
    // Throws if t doesn't lie inside this file (e.g. a TensorInfo from another file).
    const void *data(const TensorInfo &t) const;
    size_t file_size() const { return file_size_; }
    const std::string &path() const { return path_; }

private:
    std::string path_;
    void *map_ = nullptr;
    size_t file_size_ = 0;
    size_t data_start_ = 0;  // first byte after the header
    std::vector<TensorInfo> tensors_;
};

}  // namespace strix
