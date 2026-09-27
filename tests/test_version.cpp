#include "inference/version.hpp"

#include <gtest/gtest.h>

TEST(VersionTest, ReturnsCurrentVersion) {
    EXPECT_EQ(inference::version(), "0.1.0");
}
