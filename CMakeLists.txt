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
    src/layers/linear.cpp
    src/layers/activations.cpp
    src/model/binary_io.cpp
    src/model/tensor_io.cpp
    src/model/model.cpp
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
# Subprojects
# ---------------------------------------------------------------------------
if(ENABLE_TESTS)
    enable_testing()  # creates CTestTestfile.cmake so `ctest` works in build/
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(ENABLE_BENCHMARKS)
    if(EXISTS ${PROJECT_SOURCE_DIR}/benchmarks/CMakeLists.txt)
        add_subdirectory(benchmarks)
    else()
        message(STATUS "ENABLE_BENCHMARKS is ON but benchmarks/ has no CMakeLists.txt yet")
    endif()
endif()
