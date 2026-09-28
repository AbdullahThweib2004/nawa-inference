#pragma once

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

// A directory for the current test's temporary files. With fresh = true (the default) it is
// emptied first; temp-file helpers that are called several times per test pass false.
//
// The name includes the process id. CTest runs every test twice (normally and as
// portable_kernel.*, with NAWA_KERNEL=portable), possibly AT THE SAME TIME under `ctest -j`.
// A directory named only after the test would be shared by both processes, and one would
// delete or overwrite files the other is reading.
inline std::filesystem::path test_temp_dir(bool fresh = true) {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
#if defined(__unix__) || defined(__APPLE__)
    const std::string pid = std::to_string(static_cast<long>(::getpid()));
#else
    const std::string pid = "0";
#endif
    const std::filesystem::path dir =
        std::filesystem::path(::testing::TempDir()) /
        (std::string("nawa_") + info->test_suite_name() + "_" + info->name() + "_" + pid);
    if (fresh) std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}
