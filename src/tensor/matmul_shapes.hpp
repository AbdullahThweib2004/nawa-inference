#pragma once

// Private helper shared by matmul (ops.cpp) and matmul_naive (reference.cpp). Not installed:
// it lives in src/, not include/.

#include <cstddef>
#include <stdexcept>
#include <string>

#include "inference/tensor/tensor.hpp"

namespace inference::detail {

struct MatmulDims {
    std::size_t M, K, N;  // a is {M, K}, b is {K, N}, the result is {M, N}
};

// Validates the operands of a matrix product and returns its dimensions.
inline MatmulDims check_matmul_shapes(const Tensor& a, const Tensor& b, const char* fn) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument(std::string(fn) + ": expected two 2-D tensors, got shapes " +
                                    shape_to_string(a.shape()) + " and " +
                                    shape_to_string(b.shape()));
    }
    const MatmulDims d{a.size(0), a.size(1), b.size(1)};
    if (b.size(0) != d.K) {
        throw std::invalid_argument(
            std::string(fn) + ": inner dimensions don't match: " + shape_to_string(a.shape()) +
            " x " + shape_to_string(b.shape()) + " (" + std::to_string(d.K) +
            " != " + std::to_string(b.size(0)) + ")");
    }
    return d;
}

}  // namespace inference::detail
