#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <string>
#include <vector>

namespace inference {

// Size of each dimension, outermost first. An empty shape {} is a scalar.
using Shape = std::vector<std::size_t>;

// Number of elements a tensor of this shape holds (product of all dimensions).
// The empty shape {} returns 1, because a scalar holds exactly one value.
// Throws std::invalid_argument if the product overflows std::size_t.
std::size_t numel_of(const Shape& shape);

// Row-major (C-order) strides, in elements: strides[i] is how far to move in the flat
// data array when index i increases by one. The last dimension always has stride 1.
// Example: shape {2, 3, 4} -> strides {12, 4, 1}.
Shape compute_strides(const Shape& shape);

// Human-readable shape for messages, e.g. "[2, 3]".
std::string shape_to_string(const Shape& shape);

// An n-dimensional array of float32 values.
//
// Design (see CLAUDE.md):
//  - Always contiguous and row-major.
//  - Always owns its data: no views, no shared storage. Copying is a deep copy.
//  - Every dimension must be >= 1 (zero-sized dimensions are rejected).
//
// Like standard library containers, a moved-from Tensor may only be assigned to or
// destroyed.
class Tensor {
public:
    // Zero-initialized tensor of the given shape.
    explicit Tensor(Shape shape);

    // Tensor that takes ownership of `data`, which must hold exactly numel_of(shape)
    // values in row-major order.
    Tensor(Shape shape, std::vector<float> data);

    static Tensor zeros(Shape shape);
    static Tensor ones(Shape shape);
    static Tensor full(Shape shape, float value);
    static Tensor scalar(float value);

    const Shape& shape() const noexcept { return shape_; }
    const Shape& strides() const noexcept { return strides_; }
    std::size_t ndim() const noexcept { return shape_.size(); }
    std::size_t numel() const noexcept { return data_.size(); }

    // Size of dimension `dim`. Throws std::out_of_range if dim >= ndim().
    std::size_t size(std::size_t dim) const;

    // Raw pointer to the first element. The data is contiguous and row-major.
    float* data() noexcept { return data_.data(); }
    const float* data() const noexcept { return data_.data(); }

    // Element access with full bounds checking, e.g. t.at({1, 2}).
    // A scalar is accessed with t.at({}).
    float& at(std::initializer_list<std::size_t> indices);
    float at(std::initializer_list<std::size_t> indices) const;

    // Returns a NEW tensor (deep copy) with the same data in the same order and a
    // different shape. Throws std::invalid_argument if the element count differs.
    Tensor reshape(Shape new_shape) const;

    // Same, but one dimension may be -1, meaning "infer it from the element count".
    Tensor reshape(const std::vector<std::int64_t>& new_shape) const;

    // Lets t.reshape({2, -1}) and t.reshape({2, 3}) compile. Without it, a braced list
    // could convert to both Shape and std::vector<int64_t>, and the call is ambiguous.
    Tensor reshape(std::initializer_list<std::int64_t> new_shape) const;

    // Returns a 1-D copy with shape {numel()}.
    Tensor flatten() const;

    void fill(float value);

    // Multi-line, PyTorch-like representation. Large tensors are summarized with "...".
    std::string to_string() const;

private:
    // Position of the element at `indices` in data_, using the stride formula:
    //   offset = sum_i indices[i] * strides_[i]
    // Throws std::out_of_range on a wrong number of indices or an index out of range.
    std::size_t offset(std::initializer_list<std::size_t> indices) const;

    Shape shape_;
    Shape strides_;  // in elements, not bytes
    // Kept private and reached only through the API, so it can later be replaced with
    // 64-byte-aligned storage for SIMD without changing any calling code.
    std::vector<float> data_;
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

// True if a and b have the same shape and every pair of elements satisfies
//   |a - b| <= atol + rtol * |b|      (same rule as numpy.allclose)
// Returns false if the shapes differ or any element is NaN.
bool allclose(const Tensor& a, const Tensor& b, float rtol = 1e-5f, float atol = 1e-8f);

}  // namespace inference
