#pragma once

// <hip/hip_runtime.h> for this repo: include this, never the HIP header directly (a ctest checks it).
//
// Outside a HIP compile (host-only .cpp files), HIP's hip/amd_detail/host_defines.h defines __noinline__ to nothing.
// GCC 16's libstdc++ writes an attribute as [[__gnu__::__noinline__]] in <format> (which <chrono> includes), so a
// standard header included after HIP's reads [[__gnu__::]] - "error: expected identifier". Including <format> first
// lets its include guard keep the macro out of it.

#include <format>

#include <hip/hip_runtime.h>
