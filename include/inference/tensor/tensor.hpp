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

// Heap storage for float32 values, aligned to 64 bytes (one cache line, and the width of an
// AVX-512 register): a SIMD load of a whole cache line never straddles two lines, and
// every tensor starts at the same position relative to cache lines, which makes timings
// reproducible. Like std::vector it owns its memory (deep copy, cheap move), and it keeps
// its capacity when it shrinks, so reused buffers stop allocating once they are big enough.
class AlignedBuffer {
public:
    static constexpr std::size_t kAlignment = 64;

    AlignedBuffer() noexcept = default;
    explicit AlignedBuffer(std::size_t size, float value = 0.0f);
    AlignedBuffer(const float* values, std::size_t size);  // copies `size` values
    AlignedBuffer(const AlignedBuffer& other);
    AlignedBuffer& operator=(const AlignedBuffer& other);
    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;
    ~AlignedBuffer();

    float* data() noexcept { return data_; }
    const float* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }
    float& operator[](std::size_t i) noexcept { return data_[i]; }
    float operator[](std::size_t i) const noexcept { return data_[i]; }

    // Sets the size to n. Reallocates only if n > capacity(); the contents are then
    // UNSPECIFIED (not preserved), because every caller overwrites them anyway.
    void resize_uninitialized(std::size_t n);

private:
    float* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
};

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

    // Matrix product, same as the free function matmul(*this, other) in ops.hpp.
    // Lets layer code read naturally: input.matmul(weights).
    Tensor matmul(const Tensor& other) const;

    void fill(float value);

    // Changes the shape in place, reusing the existing memory when it is large enough
    // (no allocation in steady state). The contents are unspecified afterwards: this is for
    // output buffers that are about to be overwritten. Throws like the constructor for a
    // zero-sized dimension.
    void resize(const Shape& new_shape);

    // Same for a 2-D shape, without building a Shape (so a no-op resize allocates nothing).
    void resize(std::size_t rows, std::size_t cols);

    // Multi-line, PyTorch-like representation. Large tensors are summarized with "...".
    std::string to_string() const;

private:
    Tensor() = default;  // used internally (reshape) before filling in all members

    // Position of the element at `indices` in data_, using the stride formula:
    //   offset = sum_i indices[i] * strides_[i]
    // Throws std::out_of_range on a wrong number of indices or an index out of range.
    std::size_t offset(std::initializer_list<std::size_t> indices) const;

    Shape shape_;
    Shape strides_;  // in elements, not bytes
    // 64-byte-aligned storage (step 9.4). It was a std::vector<float> before; because it
    // was only ever reached through the API, no calling code had to change.
    AlignedBuffer data_;
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

// True if a and b have the same shape and every pair of elements satisfies
//   |a - b| <= atol + rtol * |b|      (same rule as numpy.allclose)
// Returns false if the shapes differ or any element is NaN.
bool allclose(const Tensor& a, const Tensor& b, float rtol = 1e-5f, float atol = 1e-8f);

}  // namespace inference
