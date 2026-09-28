#include "inference/runtime/workspace.hpp"

namespace inference {

Tensor& Workspace::buffer(std::size_t i) {
    while (buffers_.size() <= i) buffers_.push_back(std::make_unique<Tensor>(Shape{1}));
    return *buffers_[i];
}

std::size_t Workspace::bytes() const noexcept {
    std::size_t total = 0;
    for (const auto& b : buffers_) total += b->numel() * sizeof(float);
    return total;
}

}  // namespace inference
