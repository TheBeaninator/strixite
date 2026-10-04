#pragma once

// ROCTX ranges for profiler timelines (rocprofv3 --marker-trace; Optiq): compiled in only when the build sets
// STRIX_ROCTX (CMake option, preset strix-roctx). Off - every shipping build - the
// macros expand to nothing and their name arguments are never evaluated, so they cost nothing.
//
// Ranges are host-side: they mark when the host *enqueued* the work, which runs ahead of the GPU. A kernel's range is
// the one around the HIP launch that queued it (rocpd links each dispatch to its launch call; tools/trace_gaps.py
// labels kernels that way).
//
//   STRIX_TRACE_RANGE("forward");        // pushed here, popped at the end of the scope
//   STRIX_TRACE_STAGE(stage);            // a range that moves along a scope: set() pops the last one, pushes
//   STRIX_TRACE_SET(stage, "L3 mixer");  // the next; the last pops at the end of the scope

#if defined(STRIX_ROCTX) && STRIX_ROCTX
#include <rocprofiler-sdk-roctx/roctx.h>

#include <string>

namespace strix::trace {

class Range {
public:
    explicit Range(const std::string &name) { roctxRangePushA(name.c_str()); }
    ~Range() { roctxRangePop(); }
    Range(const Range &) = delete;
    Range &operator=(const Range &) = delete;
};

class Stage {
public:
    Stage() = default;
    ~Stage() {
        if (open_) roctxRangePop();
    }
    void set(const std::string &name) {
        if (open_) roctxRangePop();
        roctxRangePushA(name.c_str());
        open_ = true;
    }
    Stage(const Stage &) = delete;
    Stage &operator=(const Stage &) = delete;

private:
    bool open_ = false;
};

}  // namespace strix::trace

#define STRIX_TRACE_CAT2(a, b) a##b
#define STRIX_TRACE_CAT(a, b) STRIX_TRACE_CAT2(a, b)
#define STRIX_TRACE_RANGE(name) ::strix::trace::Range STRIX_TRACE_CAT(strix_trace_range_, __LINE__)(name)
#define STRIX_TRACE_STAGE(var) ::strix::trace::Stage var
#define STRIX_TRACE_SET(var, name) (var).set(name)
#else
#define STRIX_TRACE_RANGE(name) static_cast<void>(0)
#define STRIX_TRACE_STAGE(var) static_cast<void>(0)
#define STRIX_TRACE_SET(var, name) static_cast<void>(0)
#endif
