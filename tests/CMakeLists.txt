include(FetchContent)

# SYSTEM marks GoogleTest's headers as system headers, so the compiler doesn't
# report warnings from inside them. Without it, our -Werror test code could
# fail because of warnings in GoogleTest itself (e.g. new Clang diagnostics).
# The SYSTEM keyword needs CMake >= 3.25; older versions just skip it.
set(_gtest_system "")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
    set(_gtest_system SYSTEM)
endif()

# Download GoogleTest at configure time, pinned to a release tag so builds are
# reproducible. It is built from source as part of this project.
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
    GIT_SHALLOW    TRUE
    ${_gtest_system}
)
# On Windows, make GoogleTest use the same C runtime as our code.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# We only need GoogleTest, not GoogleMock, and we don't want its install rules.
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Silence warnings when compiling GoogleTest's own sources: it's third-party
# code we don't control, and its warnings would only be noise in our build.
foreach(_gtest_target gtest gtest_main)
    if(MSVC)
        target_compile_options(${_gtest_target} PRIVATE /w)
    else()
        target_compile_options(${_gtest_target} PRIVATE -w)
    endif()
endforeach()

add_executable(inference_tests
    test_version.cpp
)
target_link_libraries(inference_tests PRIVATE
    inference_core
    inference_warnings
    GTest::gtest_main  # provides main(), so test files contain only TEST()s
)

# Asks the test binary for its list of TEST()s after it is built, and registers
# each one with CTest individually.
include(GoogleTest)
gtest_discover_tests(inference_tests)
