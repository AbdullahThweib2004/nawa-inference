# Benchmarks: built only with -DENABLE_BENCHMARKS=ON. See docs/performance.md.
include(FetchContent)

# Google Benchmark, pinned to a release. SYSTEM keeps warnings from its headers out of our
# -Werror build (CMake >= 3.25; older versions just skip it).
set(_bench_system "")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
    set(_bench_system SYSTEM)
endif()
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.5
    GIT_SHALLOW    TRUE
    ${_bench_system}
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)
# Third-party code: its own warnings are not our concern.
target_compile_options(benchmark PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>)

add_executable(nawa_bench
    bench_main.cpp
    bench_ops.cpp
    bench_model.cpp
    alloc_counter.cpp
    cycle_counter.cpp
)
target_link_libraries(nawa_bench PRIVATE inference_core benchmark::benchmark inference_warnings)
set_target_properties(nawa_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Recorded in every result file, so numbers are never separated from how they were built.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)
target_compile_definitions(nawa_bench PRIVATE
    NAWA_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    NAWA_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    NAWA_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_build_type_upper}}"
    NAWA_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
)
