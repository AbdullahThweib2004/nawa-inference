#pragma once

#include <string_view>

namespace inference {

// Returns the library version as "MAJOR.MINOR.PATCH".
std::string_view version() noexcept;

}  // namespace inference
