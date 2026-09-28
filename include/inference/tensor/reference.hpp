#pragma once

#include "inference/tensor/tensor.hpp"

// Reference implementations: the original, deliberately simple versions of operations that
// have since been optimized. They are kept FOREVER as ground truth. Tests compare the fast
// versions against them, and benchmarks measure how much faster the fast versions are.
// Never optimize anything in this file.
namespace inference {

// The original naive matrix product (i-j-k loop order): a is {M, K}, b is {K, N}.
// Same contract and errors as matmul() in ops.hpp.
Tensor matmul_naive(const Tensor& a, const Tensor& b);

}  // namespace inference
