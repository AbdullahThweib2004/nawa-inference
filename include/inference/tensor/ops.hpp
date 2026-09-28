#pragma once

#include <cstddef>
#include <cstdint>

#include "inference/tensor/gemm.hpp"
#include "inference/tensor/tensor.hpp"

// Math operations on tensors. Every function returns a NEW contiguous tensor; inputs are
// never modified (see the tensor design in CLAUDE.md).
namespace inference {

// ---------------------------------------------------------------------------
// Matrix operations (2-D only)
// ---------------------------------------------------------------------------

// Matrix product: a is {M, K}, b is {K, N}, the result is {M, N}.
// Optimized; the original naive version is kept as matmul_naive() in reference.hpp.
// Throws std::invalid_argument if either input is not 2-D or the inner dimensions differ.
Tensor matmul(const Tensor& a, const Tensor& b);

// Same, with an explicit kernel (tests use this to exercise every path).
Tensor matmul(const Tensor& a, const Tensor& b, GemmKernel kernel);

// matmul() packs B only when A has at least this many rows (see matmul for why).
inline constexpr std::size_t kMatmulPackMinRows = 3;

// Swaps rows and columns: {M, N} -> {N, M}. Copies the data.
// Throws std::invalid_argument if t is not 2-D.
Tensor transpose(const Tensor& t);

// ---------------------------------------------------------------------------
// Element-wise operations with broadcasting
// ---------------------------------------------------------------------------

// Shape of the result of an element-wise op between tensors of shapes a and b, using
// NumPy broadcasting rules: align the shapes from the right; each pair of dimensions must
// be equal, or one of them must be 1 (or missing), and the result takes the larger one.
// Examples: {4, 128} and {128} -> {4, 128};  {4, 1} and {1, 3} -> {4, 3}.
// Throws std::invalid_argument (with both shapes in the message) if they are incompatible.
Shape broadcast_shape(const Shape& a, const Shape& b);

// Element-wise arithmetic. Tensor-tensor versions broadcast (see broadcast_shape).
// Division by zero follows IEEE float rules (inf or nan); it never throws.
Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

Tensor add(const Tensor& a, float s);
Tensor sub(const Tensor& a, float s);
Tensor mul(const Tensor& a, float s);
Tensor div(const Tensor& a, float s);

Tensor add(float s, const Tensor& b);
Tensor sub(float s, const Tensor& b);
Tensor mul(float s, const Tensor& b);
Tensor div(float s, const Tensor& b);

// Operators are element-wise (like NumPy/PyTorch). Note that `*` is NOT matrix
// multiplication; use matmul() for that.
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);

Tensor operator+(const Tensor& a, float s);
Tensor operator-(const Tensor& a, float s);
Tensor operator*(const Tensor& a, float s);
Tensor operator/(const Tensor& a, float s);

Tensor operator+(float s, const Tensor& b);
Tensor operator-(float s, const Tensor& b);
Tensor operator*(float s, const Tensor& b);
Tensor operator/(float s, const Tensor& b);

Tensor operator-(const Tensor& t);

// e^x for every element. Large inputs overflow to inf (IEEE); callers that need
// stability, like Softmax, shift their inputs first.
Tensor exp(const Tensor& t);

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------
// The axis version reduces along one axis. A negative axis counts from the end
// (-1 = last axis). With keepdims the reduced axis stays as size 1, so {2, 3} reduced
// along axis 1 is {2, 1} instead of {2}. Throws std::out_of_range for an invalid axis;
// a scalar has no axes, so every axis is invalid for it.

Tensor sum(const Tensor& t, std::int64_t axis, bool keepdims = false);
Tensor max(const Tensor& t, std::int64_t axis, bool keepdims = false);
Tensor mean(const Tensor& t, std::int64_t axis, bool keepdims = false);

// Position of the largest value along the axis. When the maximum appears more than
// once, the first occurrence wins.
// NOTE: the indices are stored as float values (0.0f, 1.0f, ...) because Tensor is
// float32-only for now. Floats represent integers exactly up to 2^24, far more than any
// axis length we will use.
Tensor argmax(const Tensor& t, std::int64_t axis, bool keepdims = false);

// Reductions over all elements. The result is a scalar tensor (shape {}).
Tensor sum(const Tensor& t);
Tensor max(const Tensor& t);

}  // namespace inference
