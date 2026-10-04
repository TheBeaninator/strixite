#pragma once

// Bulk weight loader: reads a safetensors file's raw bytes into pinned host
// memory with concurrent pread() workers, then drops the file from the page
// cache - the bytes now live in the pinned buffer, and at target scale
// (~115 GiB quantized vs 124 GiB RAM) keeping a second, cached copy would
// crowd out everything else.
//
// Default mode is buffered (2026-09-24): the test machine's
// /home is btrfs with zstd compression, where O_DIRECT silently falls back
// to buffered reads anyway. O_DIRECT stays available as LoadMode::Direct
// (re-derived from man 2 open, not from llama.cpp) for uncompressed storage;
// on a compressed btrfs file it behaves like Buffered.
//
// Distinct from formats::SafetensorsFile (mmap-based): that one is for cheap
// header/metadata inspection. This one is the "get tensor bytes into
// GPU-accessible memory for inference" path; it reuses SafetensorsFile's
// parsed layout (byte_offset is an absolute file offset valid against
// either reader's base pointer).

#include "formats/safetensors.hpp"

#include <cstddef>
#include <string>

namespace strix {

// Alignment O_DIRECT needs for buffer address, file offset and length.
constexpr size_t kDirectIOAlign = 4096;
size_t direct_io_buffer_size(size_t file_size);  // file_size rounded up to kDirectIOAlign

// Reads all file_size bytes of path into buf using num_workers concurrent
// pread() calls over disjoint aligned ranges. direct=true opens with
// O_DIRECT; buf must then be kDirectIOAlign-aligned. buf must hold at least
// direct_io_buffer_size(file_size) bytes either way. Throws on any failure.
void read_file_parallel(const std::string &path, void *buf, size_t file_size, bool direct, int num_workers);

// fdatasync + posix_fadvise(DONTNEED): evicts path's clean pages from the page
// cache (dirty ones are flushed first so they can be dropped too). Throws on failure.
void drop_page_cache(const std::string &path);

enum class LoadMode { Buffered, Direct };
const char *load_mode_name(LoadMode m);

class WeightLoader {
public:
    // Loads the file sf was parsed from (sf.path()), so layout and bytes
    // can't come from different files. num_workers concurrent readers drive
    // NVMe queue depth; a single synchronous read undersells the drive.
    explicit WeightLoader(const SafetensorsFile &sf, LoadMode mode = LoadMode::Buffered, int num_workers = 8);
    ~WeightLoader();
    WeightLoader(const WeightLoader &) = delete;
    WeightLoader &operator=(const WeightLoader &) = delete;

    // Pointer into the pinned buffer at t.byte_offset; valid for this loader's lifetime.
    // Throws if t lies outside the loaded file.
    const void *data(const TensorInfo &t) const;
    size_t bytes_loaded() const { return file_size_; }

private:
    std::string path_;
    void *pinned_ = nullptr;
    size_t file_size_ = 0;
    size_t alloc_size_ = 0;  // file_size_ rounded up to kDirectIOAlign
};

}  // namespace strix
