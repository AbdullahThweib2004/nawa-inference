#include "inference/version.hpp"

namespace inference {

std::string_view version() noexcept {
    // A string literal has static storage duration, so returning a view to it is safe.
    return "0.1.0";
}

}  // namespace inference
