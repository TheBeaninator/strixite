cmake_minimum_required(VERSION 3.28)

# --- host C++ stdlib -----------------------------------------------------------
# therock-tarball's clang ships no usable host libstdc++/libc++ (its bundled
# libc++ is device-target only, no x86_64 __config_site), so it relies on a
# GCC install's. On Ubuntu the system g++-13/14 is found unaided. Fedora 43
# without gcc-c++/libstdc++-devel can use Linuxbrew's gcc-15, which clang
# doesn't find on its own - so point at it when it's there. Must be set before project(): that runs CMake's
# compiler ABI check, which already links a trivial C++ program.
set(STRIX_BREW_GCC_ROOT "/home/linuxbrew/.linuxbrew" CACHE PATH "Linuxbrew prefix providing gcc-15's C++ stdlib")
set(STRIX_BREW_CXX_INCLUDE "${STRIX_BREW_GCC_ROOT}/include/c++/15")
if(EXISTS "${STRIX_BREW_CXX_INCLUDE}/vector")
    set(CMAKE_CXX_FLAGS "-isystem${STRIX_BREW_CXX_INCLUDE} -isystem${STRIX_BREW_CXX_INCLUDE}/x86_64-pc-linux-gnu")
    set(CMAKE_EXE_LINKER_FLAGS "-L${STRIX_BREW_GCC_ROOT}/lib/gcc/15")
endif()

# The ROCm clang compiles host and device alike; it is selected by the preset
# (CMAKE_CXX_COMPILER) before project() so the ABI check runs on the right one.
project(strix-infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)   # no C++20 modules; CMake 3.28 would otherwise want clang-scan-deps
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- ROCm --------------------------------------------------------------------
# No hipcc, no CMake HIP language: HIP sources are plain CXX sources compiled
# with `-x hip --offload-arch=...`. Host-only files that call the HIP runtime
# API only need the headers and __HIP_PLATFORM_AMD__. gfx1151 is the only
# shipping target; gfx1100 exists only as a local dev-loop build for
# correctness checks (xtx-dev preset).
set(STRIX_GPU_ARCH "gfx1151" CACHE STRING "Device code target (gfx1151 ships; gfx1100 = local dev only)")
set(STRIX_ROCM_ROOT "$ENV{ROCM_PATH}" CACHE PATH "ROCm install (therock tarball)")
if(NOT STRIX_ROCM_ROOT)
    set(STRIX_ROCM_ROOT "$ENV{HOME}/tools/therock-tarball/install")
endif()
if(NOT EXISTS "${STRIX_ROCM_ROOT}/include/hip/hip_runtime.h")
    message(FATAL_ERROR "STRIX_ROCM_ROOT=${STRIX_ROCM_ROOT} has no hip/hip_runtime.h")
endif()
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR NOT CMAKE_CXX_COMPILER MATCHES "${STRIX_ROCM_ROOT}")
    message(FATAL_ERROR "CMAKE_CXX_COMPILER must be the ROCm clang under ${STRIX_ROCM_ROOT} "
                        "(got ${CMAKE_CXX_COMPILER}); use the strix preset from CMakePresets.json")
endif()
message(STATUS "ROCm at ${STRIX_ROCM_ROOT}")

find_library(STRIX_HIP_LIB amdhip64 PATHS "${STRIX_ROCM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_library(strix_rocm INTERFACE)
target_include_directories(strix_rocm SYSTEM INTERFACE "${STRIX_ROCM_ROOT}/include")
target_compile_definitions(strix_rocm INTERFACE __HIP_PLATFORM_AMD__)
target_link_libraries(strix_rocm INTERFACE "${STRIX_HIP_LIB}")
target_link_options(strix_rocm INTERFACE "-Wl,-rpath,${STRIX_ROCM_ROOT}/lib")

# Mark a target's .hip sources for device compilation.
function(strix_hip_sources target)
    foreach(src IN LISTS ARGN)
        set_source_files_properties(${src} PROPERTIES
            LANGUAGE CXX
            COMPILE_OPTIONS "-x;hip;--offload-arch=${STRIX_GPU_ARCH};--rocm-path=${STRIX_ROCM_ROOT}")
    endforeach()
    target_sources(${target} PRIVATE ${ARGN})
endfunction()

# --- host flags ----------------------------------------------------------------
add_compile_options(-Wall -Wextra -Wno-unused-parameter)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=znver3 -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=znver3 -flto -g -DNDEBUG")
add_link_options(-fuse-ld=lld)
if(CMAKE_BUILD_TYPE MATCHES "Rel")
    add_link_options(-flto)
endif()

# --- targets -------------------------------------------------------------------
find_package(Threads REQUIRED)
add_library(strix_formats STATIC
    formats/safetensors.cpp
    formats/q4.cpp
    formats/q5.cpp
    formats/q6.cpp
    formats/q8.cpp
    formats/merge_plan.cpp
    formats/strixw.cpp
    formats/ngram_table.cpp
    formats/convert_qwen4exp.cpp)
target_include_directories(strix_formats PUBLIC "${CMAKE_SOURCE_DIR}")
target_link_libraries(strix_formats PUBLIC Threads::Threads)

add_executable(convert_qwen4exp tools/convert_qwen4exp.cpp)
target_link_libraries(convert_qwen4exp PRIVATE strix_formats)

add_executable(convert_ngram_table tools/convert_ngram_table.cpp)
target_link_libraries(convert_ngram_table PRIVATE strix_formats)

add_executable(inspect_strixw tools/inspect_strixw.cpp)
target_link_libraries(inspect_strixw PRIVATE strix_formats)

# Development tools (benches, studies, inspectors) and tests/ are built when their sources are in the tree; the
# engine, the server and the converters build without them.
if(EXISTS "${CMAKE_SOURCE_DIR}/tools/bench_linear.cpp")
    set(STRIX_DEV_TOOLS_DEFAULT ON)
else()
    set(STRIX_DEV_TOOLS_DEFAULT OFF)
endif()
option(STRIX_DEV_TOOLS "Build the bench / study / inspection tools" ${STRIX_DEV_TOOLS_DEFAULT})

if(STRIX_DEV_TOOLS)
add_executable(inspect_safetensors tools/inspect_safetensors.cpp)
target_link_libraries(inspect_safetensors PRIVATE strix_formats)
endif()

add_library(strix_runtime STATIC
    runtime/weight_loader.cpp
    runtime/ple_hash.cpp)
target_include_directories(strix_runtime PUBLIC "${CMAKE_SOURCE_DIR}")
target_link_libraries(strix_runtime PUBLIC strix_formats strix_rocm Threads::Threads)

if(STRIX_DEV_TOOLS)
add_executable(load_weights tools/load_weights.cpp)
target_link_libraries(load_weights PRIVATE strix_runtime)

add_executable(bench_load tools/bench_load.cpp)
target_link_libraries(bench_load PRIVATE strix_runtime)

endif()

# The int8 n-gram table from convert_ngram_table's BF16 one (the published table is made this way).
add_executable(transcode_ngram_table tools/transcode_ngram_table.cpp)
target_link_libraries(transcode_ngram_table PRIVATE strix_model)

add_library(strix_kernels STATIC)
strix_hip_sources(strix_kernels kernels/norm.hip kernels/embedding.hip kernels/linear.hip kernels/linear_q4.hip kernels/linear_q56.hip kernels/linear_q8.hip
    kernels/swiglu.hip kernels/rope.hip kernels/attention.hip kernels/gdn.hip
    kernels/residual.hip kernels/moe_router.hip kernels/moe_grouped.hip kernels/wmma.hip kernels/linear_wmma.hip kernels/hc_wmma.hip kernels/hc.hip kernels/ple.hip kernels/qsa.hip
    kernels/mtp_pick.hip kernels/logits_topk.hip)
target_include_directories(strix_kernels PUBLIC "${CMAKE_SOURCE_DIR}")
target_link_libraries(strix_kernels PUBLIC strix_rocm strix_formats)

add_library(strix_model STATIC runtime/qwen3_5.cpp runtime/qwen4exp.cpp runtime/strixw_loader.cpp
    runtime/ngram_rows.cpp runtime/ngram_table.cpp runtime/tp_mirror.cpp)
target_link_libraries(strix_model PUBLIC strix_kernels strix_formats strix_runtime)

# The WMMA kernels at a K tail (K % 64 == 32: a TP-4 rank's expert down, K = 160) against the FP32 kernels.
add_executable(test_wmma_ktail)
strix_hip_sources(test_wmma_ktail tools/test_wmma_ktail.cpp)
target_link_libraries(test_wmma_ktail PRIVATE strix_kernels strix_formats)
# ROCTX ranges (common/trace.hpp) for profiler timelines - off in every shipping build.
option(STRIX_ROCTX "Compile in ROCTX ranges (forward / layer / sublayer) for rocprofv3 --marker-trace" OFF)
if(STRIX_ROCTX)
    find_package(rocprofiler-sdk-roctx REQUIRED PATHS "${STRIX_ROCM_ROOT}/lib/cmake" NO_DEFAULT_PATH)
    target_link_libraries(strix_model PUBLIC rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
    target_compile_definitions(strix_model PUBLIC STRIX_ROCTX=1)
endif()

if(STRIX_DEV_TOOLS)
add_executable(bench_linear)
strix_hip_sources(bench_linear tools/bench_linear.cpp)
target_link_libraries(bench_linear PRIVATE strix_kernels strix_formats)

add_executable(bench_prefill_gemm)
strix_hip_sources(bench_prefill_gemm tools/bench_prefill_gemm.cpp)
target_link_libraries(bench_prefill_gemm PRIVATE strix_kernels strix_formats)

add_executable(bench_attention)
strix_hip_sources(bench_attention tools/bench_attention.cpp)
target_link_libraries(bench_attention PRIVATE strix_kernels)

add_executable(bench_forward)
strix_hip_sources(bench_forward tools/bench_forward.cpp)
target_link_libraries(bench_forward PRIVATE strix_model)

add_executable(bench_gdn)
strix_hip_sources(bench_gdn tools/bench_gdn.cpp)
target_link_libraries(bench_gdn PRIVATE strix_kernels)

add_executable(bench_forward_qwen4exp)
strix_hip_sources(bench_forward_qwen4exp tools/bench_forward_qwen4exp.cpp)
target_link_libraries(bench_forward_qwen4exp PRIVATE strix_model)

add_executable(act_range)
strix_hip_sources(act_range tools/act_range.cpp)
target_link_libraries(act_range PRIVATE strix_model)

add_executable(route_stats)
strix_hip_sources(route_stats tools/route_stats.cpp)
target_link_libraries(route_stats PRIVATE strix_model)

add_executable(bench_hc)
strix_hip_sources(bench_hc tools/bench_hc.cpp)
target_link_libraries(bench_hc PRIVATE strix_kernels strix_formats)

add_executable(bench_router)
strix_hip_sources(bench_router tools/bench_router.cpp)
target_link_libraries(bench_router PRIVATE strix_kernels)

add_executable(bench_boundary)
strix_hip_sources(bench_boundary tools/bench_boundary.cpp)
target_link_libraries(bench_boundary PRIVATE strix_kernels strix_formats)

add_executable(bench_wmma_peak)
strix_hip_sources(bench_wmma_peak tools/bench_wmma_peak.cpp)
target_link_libraries(bench_wmma_peak PRIVATE strix_kernels)

add_executable(bench_logits_topk)
strix_hip_sources(bench_logits_topk tools/bench_logits_topk.cpp)
target_link_libraries(bench_logits_topk PRIVATE strix_kernels)

add_executable(bench_qsa)
strix_hip_sources(bench_qsa tools/bench_qsa.cpp)
target_link_libraries(bench_qsa PRIVATE strix_kernels)

add_executable(bench_ple)
strix_hip_sources(bench_ple tools/bench_ple.cpp)
target_link_libraries(bench_ple PRIVATE strix_kernels)
endif()

# Serving: host-only pieces - JSON, tokenizer, chat template, ...
add_library(strix_serve STATIC
    serve/json.cpp
    serve/unicode.cpp
    serve/tokenizer.cpp
    serve/chat_template.cpp
    serve/sampler.cpp
    serve/output_parser.cpp
    serve/engine.cpp
    serve/think_nudge.cpp
    serve/server_config.cpp
    serve/openai.cpp
    serve/http_server.cpp
    serve/prompt_cache.cpp
    serve/resume_loss.cpp
    serve/capture.cpp
    serve/replay.cpp
    serve/host_buffer.cpp)
target_include_directories(strix_serve PUBLIC "${CMAKE_SOURCE_DIR}")
target_include_directories(strix_serve SYSTEM PRIVATE "${CMAKE_SOURCE_DIR}/third_party/cpp-httplib")
target_link_libraries(strix_serve PUBLIC strix_formats Threads::Threads)

# The server on the real model (HIP runtime via strix_model).
add_library(strix_serve_model STATIC serve/qwen4exp_backend.cpp)
target_link_libraries(strix_serve_model PUBLIC strix_serve strix_model)

if(STRIX_DEV_TOOLS)
# The prompt cache under memory pressure, host only (no model, no GPU): tools/stress_prompt_cache.cpp.
add_executable(stress_prompt_cache tools/stress_prompt_cache.cpp)
target_link_libraries(stress_prompt_cache PRIVATE strix_serve)

add_executable(bench_host_buffer tools/bench_host_buffer.cpp serve/host_buffer.cpp)
target_include_directories(bench_host_buffer PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(bench_host_buffer PRIVATE strix_rocm)

add_executable(bench_sampler tools/bench_sampler.cpp)
target_link_libraries(bench_sampler PRIVATE strix_serve)
endif()

add_executable(strix_server tools/strix_server.cpp)
target_link_libraries(strix_server PRIVATE strix_serve_model)

if(STRIX_DEV_TOOLS)
# Decode A/Bs on captured real traffic, teacher-forced (serve/replay.hpp).
add_executable(bench_replay tools/bench_replay.cpp)
target_link_libraries(bench_replay PRIVATE strix_serve_model)
endif()

# strix_bench: single-stream decode timing at depth, MTP replay, golden logits + noise floor, needles;
# --tp-world / --tp-rank load one rank's share with every exchange stubbed (tools/strix_bench.cpp).
add_executable(strix_bench tools/strix_bench.cpp)
target_link_libraries(strix_bench PRIVATE strix_serve_model)

# Tensor parallelism over RDMA (docs/tp-rdma.md): the decode / prefill exchange microbenchmark (RoCE v2, stock
# rdma-core mlx5), the TP-N communicator and tools/tp_ar. Optional: needs libibverbs; the engine and the server never
# link it.
option(STRIX_TP_RDMA "Build the RDMA tensor-parallel library and tools (needs libibverbs)" ON)
if(STRIX_TP_RDMA)
    find_library(STRIX_IBVERBS_LIB ibverbs)
    if(NOT STRIX_IBVERBS_LIB)
        message(STATUS "strixite: libibverbs not found - tp_exchange_bench / strix_tp / tp_ar skipped")
    endif()
endif()
if(STRIX_TP_RDMA AND STRIX_IBVERBS_LIB)
add_executable(tp_exchange_bench)
strix_hip_sources(tp_exchange_bench tools/tp_exchange_bench.cpp)
target_link_libraries(tp_exchange_bench PRIVATE strix_rocm ${STRIX_IBVERBS_LIB} Threads::Threads)

# The TP-N communicator (runtime/tp_comm) and rank 0's driver / the executors (runtime/tp_mirror);
# tools/tp_ar checks and times the TP forward against the whole model.
# (TpDriver / tp_executor / TpLoopback - runtime/tp_mirror.cpp - are in strix_model: no RDMA; tp_attach joins a session
# to the communicator.)
add_library(strix_tp STATIC runtime/tp_attach.cpp)
strix_hip_sources(strix_tp runtime/tp_comm.hip)
target_link_libraries(strix_tp PUBLIC strix_model ${STRIX_IBVERBS_LIB} Threads::Threads)
add_executable(tp_ar tools/tp_ar.cpp)
target_link_libraries(tp_ar PRIVATE strix_tp strix_serve_model)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

# The tutorials live in the public repo only; built when present.
if(EXISTS "${CMAKE_SOURCE_DIR}/tutorial/CMakeLists.txt")
    add_subdirectory(tutorial)
endif()
