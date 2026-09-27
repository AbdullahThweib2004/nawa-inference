#include <gtest/gtest.h>

#include "inference/version.hpp"

TEST(VersionTest, ReturnsCurrentVersion) { EXPECT_EQ(inference::version(), "0.1.0"); }
