#pragma once

// Residual add for the decoder layers:
// x = x + y in place, element-wise, FP32 add and one rounding to the
// activation dtype - the same values as transformers' add in either dtype.

#include "kernels/norm.hpp"  // Act

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strix::kernels {

// x, y: n elements of the activation dtype. y must not overlap x (except
// y == x, which doubles x).
void residual_add(void *x, const void *y, int64_t n, Act act, hipStream_t stream);

}  // namespace strix::kernels
