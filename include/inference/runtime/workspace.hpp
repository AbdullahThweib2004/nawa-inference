#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "inference/tensor/tensor.hpp"

namespace inference {

// Reusable scratch memory for running a model without heap allocations.
//
// Model::predict(raw, workspace) writes its intermediate results and its output into the
// workspace's buffers. The buffers grow the first time they are needed and keep their memory
// afterwards, so repeated calls with the same (or a smaller) batch size allocate nothing.
//
// THREAD SAFETY: a Workspace is mutable scratch space. Use one Workspace per thread (or per
// concurrent request). A Model is immutable after loading and can be shared by any number of
// threads, each with its own Workspace.
class Workspace {
public:
    Workspace() = default;

    // Buffer number i, created on first use. Its address stays valid for the lifetime of
    // the workspace (buffers are held by pointer, so adding more doesn't move them).
    Tensor& buffer(std::size_t i);

    std::size_t num_buffers() const noexcept { return buffers_.size(); }

    // Total floats currently reserved by all buffers, in bytes.
    std::size_t bytes() const noexcept;

private:
    std::vector<std::unique_ptr<Tensor>> buffers_;
};

}  // namespace inference
