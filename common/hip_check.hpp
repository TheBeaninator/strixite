#pragma once

// HIP return-code check in the common/check.hpp format:
//   STRIX_HIP_CHECK(hipMalloc(&p, bytes), "device buffer for '", name, "', ", bytes, " bytes");
//   -> "... check `hipMalloc(&p, bytes)` failed: hipErrorOutOfMemory (out of memory): device buffer ..."
// Separate from check.hpp so host-only code doesn't need the HIP headers.

#include "common/check.hpp"

#include "common/hip_runtime.hpp"

#define STRIX_HIP_CHECK(expr, ...)                                                               \
    do {                                                                                         \
        hipError_t strix_hip_err_ = (expr);                                                      \
        if (strix_hip_err_ != hipSuccess)                                                        \
            ::strix::check_failed(__FILE__, __LINE__, __func__, #expr,                           \
                                  ::strix::cat(hipGetErrorName(strix_hip_err_), " (",            \
                                               hipGetErrorString(strix_hip_err_), "): ",         \
                                               __VA_ARGS__));                                    \
    } while (0)

// After a kernel launch: catches bad launch configs immediately rather than at the next sync.
#define STRIX_HIP_CHECK_LAUNCH(...) STRIX_HIP_CHECK(hipGetLastError(), "kernel launch: ", __VA_ARGS__)
