cmake_minimum_required(VERSION 3.20)

project(inference
    VERSION 0.1.0
    DESCRIPTION "A lightweight neural network inference engine written from scratch"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ENABLE_TESTS "Build the GoogleTest unit tests" ON)
option(ENABLE_BENCHMARKS "Build the benchmarks" OFF)
option(BUILD_EXAMPLES "Build the example programs" ON)
option(ENABLE_SANITIZERS "Enable AddressSanitizer + UBSan in Debug builds (GCC/Clang)" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer (GCC/Clang; cannot be combined with ENABLE_SANITIZERS)" OFF)
option(NAWA_NATIVE "Optimize for this machine's CPU (-march=native); binaries may not run elsewhere" OFF)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
# Default to an optimized build. Single-config generators (Makefiles, Ninja)
# otherwise build with no optimization flags at all, which is misleading for
# an inference engine where performance is part of the point.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # use -std=c++20, not -std=gnu++20

# Writes build/compile_commands.json so clangd / editors understand the flags.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# NAWA_NATIVE: -march=native lets the compiler use every instruction set this CPU has
# (AVX2, FMA, AVX-512 on the development machine) instead of baseline x86-64 (SSE2).
# Off by default: the resulting binary may crash with "illegal instruction" on any CPU
# that lacks one of those extensions (older or different CPUs, CI runners). Applied to all
# targets, so the library, tests and benchmarks are compiled the same way.
# ---------------------------------------------------------------------------
if(NAWA_NATIVE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-march=native)
        message(STATUS "NAWA_NATIVE: compiling with -march=native")
    else()
        message(WARNING "NAWA_NATIVE is only implemented for GCC/Clang; ignoring.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Warnings: an INTERFACE target carrying only flags. Linking it to our own
# targets applies strict warnings to our code without forcing -Werror on
# third-party code such as GoogleTest.
# ---------------------------------------------------------------------------
add_library(inference_warnings INTERFACE)
if(MSVC)
    target_compile_options(inference_warnings INTERFACE /W4 /WX)
else()
    target_compile_options(inference_warnings INTERFACE -Wall -Wextra -Wpedantic -Werror)
endif()

# ---------------------------------------------------------------------------
# Sanitizers: also INTERFACE, because both compile AND link flags are needed.
# ---------------------------------------------------------------------------
add_library(inference_sanitizers INTERFACE)
if(ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(_san_flags -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_compile_options(inference_sanitizers INTERFACE $<$<CONFIG:Debug>:${_san_flags}>)
        target_link_options(inference_sanitizers INTERFACE $<$<CONFIG:Debug>:${_san_flags}>)
    else()
        message(WARNING "ENABLE_SANITIZERS is only supported on GCC/Clang; ignoring.")
    endif()
endif()

# ThreadSanitizer: finds data races between threads (the thread pool, the parallel GEMM).
# Separate from ASan because the two can't be combined in one binary. Applied in every build
# type (RelWithDebInfo is a good choice: TSan slows code down ~5-15x).
if(ENABLE_TSAN)
    if(ENABLE_SANITIZERS)
        message(FATAL_ERROR "ENABLE_TSAN and ENABLE_SANITIZERS can't be used together")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(_tsan_flags -fsanitize=thread -fno-omit-frame-pointer)
        target_compile_options(inference_sanitizers INTERFACE ${_tsan_flags})
        target_link_options(inference_sanitizers INTERFACE ${_tsan_flags})
    else()
        message(WARNING "ENABLE_TSAN is only supported on GCC/Clang; ignoring.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(inference_core STATIC
    src/version.cpp
    src/tensor/tensor.cpp
    src/tensor/ops.cpp
    src/tensor/reference.cpp
    src/tensor/gemm.cpp
    src/tensor/int8.cpp
    src/layers/linear.cpp
    src/layers/activations.cpp
    src/layers/linear_int8.cpp
    src/model/binary_io.cpp
    src/model/tensor_io.cpp
    src/model/model.cpp
    src/runtime/workspace.cpp
    src/runtime/thread_pool.cpp
    src/data/idx.cpp
    src/data/digit_preprocess.cpp
)
# INT8: the scalar reference (int8.cpp) and the AVX2 kernel (int8_avx2.cpp) must give
# bit-identical results. Their integer sums are exact, but the float epilogue
# (acc * scale + bias) could be fused into an FMA in one file and not the other: whether the
# compiler "contracts" a*b+c is decided separately at each inlined copy, and with
# -march=native (NAWA_NATIVE) FMA is available everywhere. -ffp-contract=off forbids that in
# both files, so the rounding is the same in every build.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/tensor/int8.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
# The AVX2 + FMA GEMM kernel is compiled with -mavx2 -mfma in THIS FILE ONLY; the library
# chooses it at runtime if the CPU supports it (src/tensor/gemm.cpp). The default build stays
# portable to any x86-64 CPU, yet uses AVX2 where available.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i[3-6]86" AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_sources(inference_core PRIVATE src/tensor/gemm_avx2.cpp src/tensor/int8_avx2.cpp)
    set_source_files_properties(src/tensor/gemm_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    # The int8 kernel needs AVX2 only (no -mfma). -ffp-contract=off: see the int8 note below.
    set_source_files_properties(src/tensor/int8_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
    target_compile_definitions(inference_core PRIVATE NAWA_HAVE_AVX2_KERNEL=1)
else()
    target_compile_definitions(inference_core PRIVATE NAWA_HAVE_AVX2_KERNEL=0)
endif()

find_package(Threads REQUIRED)
target_link_libraries(inference_core PUBLIC Threads::Threads)

target_include_directories(inference_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
# PRIVATE: warnings apply when compiling the library itself.
# PUBLIC: sanitizer flags must also reach every executable that links it.
target_link_libraries(inference_core
    PRIVATE inference_warnings
    PUBLIC inference_sanitizers
)

# ---------------------------------------------------------------------------
# Image loading: stb_image (third-party, image decoding only; not math)
# ---------------------------------------------------------------------------
include(FetchContent)
# The single header, pinned to a commit and verified by hash: if the file ever changes,
# the build fails instead of silently compiling different code.
FetchContent_Declare(
    stb_image
    URL      https://raw.githubusercontent.com/nothings/stb/2c980bb59875b0d32144a71867fbdebb2f77cd20/stb_image.h
    URL_HASH SHA256=594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(stb_image)

# stb_image's implementation, compiled once, with warnings disabled for this target only.
add_library(stb_image_impl STATIC src/data/stb_image_impl.cpp)
# SYSTEM: code that includes <stb_image.h> doesn't get warnings from inside the header.
target_include_directories(stb_image_impl SYSTEM PUBLIC ${stb_image_SOURCE_DIR})
target_compile_options(stb_image_impl PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>)
target_link_libraries(stb_image_impl PRIVATE inference_sanitizers)

# Our image loader: normal strict warnings. Separate from inference_core, which stays free
# of third-party code; only the CLI and the tests link it.
add_library(nawa_image STATIC src/data/image_io.cpp)
target_link_libraries(nawa_image
    PUBLIC inference_core
    PRIVATE inference_warnings stb_image_impl
)

# ---------------------------------------------------------------------------
# Web demo server (`nawa serve`): cpp-httplib + nlohmann/json, both single headers, pinned to a
# release and verified by SHA-256 like stb_image. Used only by the nawa_server library.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    cpp_httplib
    URL      https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.58.0/httplib.h
    URL_HASH SHA256=aa14e7e7bd2703694e0a6b6855af3b8c406102ab1fc56ac905fe33619b31faa5
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_Declare(
    nlohmann_json
    URL      https://raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp
    URL_HASH SHA256=aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(cpp_httplib nlohmann_json)
# The header is included as <nlohmann/json.hpp>, so give it that directory layout.
file(COPY ${nlohmann_json_SOURCE_DIR}/json.hpp
     DESTINATION ${CMAKE_BINARY_DIR}/_deps/nlohmann_include/nlohmann)

add_library(nawa_server STATIC
    src/server/demo_service.cpp
    src/server/http_server.cpp
)
# SYSTEM: warnings from inside the two big third-party headers don't fail our -Werror build.
target_include_directories(nawa_server SYSTEM PUBLIC
    ${cpp_httplib_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/_deps/nlohmann_include
)
target_link_libraries(nawa_server
    PUBLIC inference_core Threads::Threads
    PRIVATE inference_warnings
)

# ---------------------------------------------------------------------------
# Subprojects
# ---------------------------------------------------------------------------
if(ENABLE_TESTS)
    enable_testing()  # creates CTestTestfile.cmake so `ctest` works in build/
    add_subdirectory(tests)
endif()

add_subdirectory(tools)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
