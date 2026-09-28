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

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(inference_core STATIC
    src/version.cpp
    src/tensor/tensor.cpp
    src/tensor/ops.cpp
    src/tensor/reference.cpp
    src/layers/linear.cpp
    src/layers/activations.cpp
    src/model/binary_io.cpp
    src/model/tensor_io.cpp
    src/model/model.cpp
    src/data/idx.cpp
    src/data/digit_preprocess.cpp
)
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
