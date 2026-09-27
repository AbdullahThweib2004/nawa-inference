#include "inference/tensor/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "inference/tensor/ops.hpp"

namespace inference {

namespace {

// to_string() summarizes tensors with more elements than this (same default as PyTorch).
constexpr std::size_t kSummarizeThreshold = 1000;
// When summarizing, each dimension shows this many items at the start and at the end.
constexpr std::size_t kEdgeItems = 3;
// Width of the "tensor(" prefix, used to line up nested rows under the first one.
constexpr std::size_t kPrefixWidth = 7;

// Formats any list of integers as "[a, b, c]".
template <typename Int>
std::string dims_to_string(const std::vector<Int>& dims) {
    std::ostringstream os;
    os << '[';
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) os << ", ";
        os << dims[i];
    }
    os << ']';
    return os.str();
}

// Throws if any dimension is zero, otherwise returns the shape unchanged. Used in
// constructor initializer lists, so the check runs before any memory is allocated.
Shape validated(Shape shape) {
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) {
            throw std::invalid_argument("Tensor: dimension " + std::to_string(i) + " of shape " +
                                        shape_to_string(shape) +
                                        " has size 0; zero-sized dimensions are not supported");
        }
    }
    return shape;
}

std::string format_value(float value) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(4) << value;
    return os.str();
}

// Recursively prints nested brackets, one bracket level per dimension.
struct Printer {
    std::ostream& os;
    const float* data;
    const Shape& shape;
    const Shape& strides;
    bool summarize;
    std::size_t width;  // every value is right-aligned to this width so columns line up

    // Prints dimension `dim` whose first element is at data[offset].
    void block(std::size_t dim, std::size_t offset) const {
        const std::size_t n = shape[dim];
        const bool innermost = (dim + 1 == shape.size());
        const bool skip_middle = summarize && n > 2 * kEdgeItems;
        bool first = true;

        // Separator before every item except the first. Innermost values go on one line.
        // Outer blocks go on new lines: one line break between rows of a matrix, two
        // between matrices, and so on, like PyTorch.
        auto separator = [&] {
            if (!first) {
                os << ',';
                if (innermost) {
                    os << ' ';
                } else {
                    os << std::string(shape.size() - dim - 1, '\n')
                       << std::string(kPrefixWidth + dim + 1, ' ');
                }
            }
            first = false;
        };

        os << '[';
        for (std::size_t i = 0; i < n; ++i) {
            if (skip_middle && i == kEdgeItems) {
                separator();
                os << "...";
                i = n - kEdgeItems;  // jump to the last kEdgeItems items
            }
            separator();
            const std::size_t item_offset = offset + i * strides[dim];
            if (innermost) {
                os << std::setw(static_cast<int>(width)) << format_value(data[item_offset]);
            } else {
                block(dim + 1, item_offset);
            }
        }
        os << ']';
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

std::size_t numel_of(const Shape& shape) {
    std::size_t n = 1;
    for (std::size_t dim : shape) {
        if (dim != 0 && n > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::invalid_argument("numel_of: element count of shape " +
                                        shape_to_string(shape) + " overflows std::size_t");
        }
        n *= dim;
    }
    return n;
}

Shape compute_strides(const Shape& shape) {
    // Walk from the last dimension to the first. Moving one step in dimension i skips
    // over one full block of all the dimensions after it, so its stride is the product
    // of their sizes.
    Shape strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

std::string shape_to_string(const Shape& shape) { return dims_to_string(shape); }

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Tensor::Tensor(Shape shape)
    : shape_(validated(std::move(shape))),
      strides_(compute_strides(shape_)),
      data_(numel_of(shape_), 0.0f) {}

Tensor::Tensor(Shape shape, std::vector<float> data)
    : shape_(validated(std::move(shape))),
      strides_(compute_strides(shape_)),
      data_(std::move(data)) {
    const std::size_t expected = numel_of(shape_);
    if (data_.size() != expected) {
        throw std::invalid_argument("Tensor: shape " + shape_to_string(shape_) + " needs " +
                                    std::to_string(expected) + " values, but data has " +
                                    std::to_string(data_.size()));
    }
}

Tensor Tensor::zeros(Shape shape) { return Tensor(std::move(shape)); }

Tensor Tensor::ones(Shape shape) { return full(std::move(shape), 1.0f); }

Tensor Tensor::full(Shape shape, float value) {
    Tensor t(std::move(shape));
    t.fill(value);
    return t;
}

Tensor Tensor::scalar(float value) { return Tensor(Shape{}, std::vector<float>{value}); }

// ---------------------------------------------------------------------------
// Access
// ---------------------------------------------------------------------------

std::size_t Tensor::size(std::size_t dim) const {
    if (dim >= ndim()) {
        throw std::out_of_range("Tensor::size: dimension " + std::to_string(dim) +
                                " is out of range for shape " + shape_to_string(shape_) + " with " +
                                std::to_string(ndim()) + " dimensions");
    }
    return shape_[dim];
}

std::size_t Tensor::offset(std::initializer_list<std::size_t> indices) const {
    if (indices.size() != ndim()) {
        throw std::out_of_range("Tensor::at: shape " + shape_to_string(shape_) + " needs " +
                                std::to_string(ndim()) + " indices, got " +
                                std::to_string(indices.size()) + " " +
                                shape_to_string(Shape(indices)));
    }
    std::size_t result = 0;
    std::size_t dim = 0;
    for (std::size_t index : indices) {
        if (index >= shape_[dim]) {
            throw std::out_of_range(
                "Tensor::at: index " + std::to_string(index) + " is out of range for dimension " +
                std::to_string(dim) + " of size " + std::to_string(shape_[dim]) + " (indices " +
                shape_to_string(Shape(indices)) + ", shape " + shape_to_string(shape_) + ")");
        }
        result += index * strides_[dim];
        ++dim;
    }
    return result;
}

float& Tensor::at(std::initializer_list<std::size_t> indices) { return data_[offset(indices)]; }

float Tensor::at(std::initializer_list<std::size_t> indices) const {
    return data_[offset(indices)];
}

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

Tensor Tensor::reshape(Shape new_shape) const {
    // Row-major data doesn't depend on the shape, so the same values in the same order
    // are valid for any shape with the same element count. Only that count must match.
    if (numel_of(new_shape) != numel()) {
        throw std::invalid_argument("Tensor::reshape: cannot reshape " + shape_to_string(shape_) +
                                    " (" + std::to_string(numel()) + " elements) to " +
                                    shape_to_string(new_shape) + " (" +
                                    std::to_string(numel_of(new_shape)) + " elements)");
    }
    return Tensor(std::move(new_shape), data_);  // copies data_
}

Tensor Tensor::reshape(const std::vector<std::int64_t>& new_shape) const {
    constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
    std::size_t infer_dim = kNone;
    Shape resolved;
    resolved.reserve(new_shape.size());

    for (std::size_t i = 0; i < new_shape.size(); ++i) {
        const std::int64_t d = new_shape[i];
        if (d == -1) {
            if (infer_dim != kNone) {
                throw std::invalid_argument("Tensor::reshape: only one dimension can be -1, got " +
                                            dims_to_string(new_shape));
            }
            infer_dim = i;
            resolved.push_back(1);  // placeholder, replaced below
        } else if (d <= 0) {
            throw std::invalid_argument("Tensor::reshape: invalid dimension " + std::to_string(d) +
                                        " in " + dims_to_string(new_shape));
        } else {
            resolved.push_back(static_cast<std::size_t>(d));
        }
    }

    if (infer_dim != kNone) {
        // With the placeholder set to 1, this is the product of the known dimensions.
        const std::size_t known = numel_of(resolved);
        if (numel() % known != 0) {
            throw std::invalid_argument("Tensor::reshape: cannot reshape " +
                                        shape_to_string(shape_) + " (" + std::to_string(numel()) +
                                        " elements) to " + dims_to_string(new_shape));
        }
        resolved[infer_dim] = numel() / known;
    }
    return reshape(std::move(resolved));
}

Tensor Tensor::reshape(std::initializer_list<std::int64_t> new_shape) const {
    return reshape(std::vector<std::int64_t>(new_shape));
}

Tensor Tensor::flatten() const { return reshape(Shape{numel()}); }

// Qualified call: inside the class, plain `matmul` would find this member function
// instead of the free function.
Tensor Tensor::matmul(const Tensor& other) const { return inference::matmul(*this, other); }

void Tensor::fill(float value) { std::fill(data_.begin(), data_.end(), value); }

// ---------------------------------------------------------------------------
// Printing and comparison
// ---------------------------------------------------------------------------

std::string Tensor::to_string() const {
    std::size_t width = 0;
    for (float v : data_) width = std::max(width, format_value(v).size());

    std::ostringstream os;
    os << "tensor(";
    if (ndim() == 0) {
        os << format_value(data_[0]);
    } else {
        const Printer printer{os,   data_.data(), shape_, strides_, numel() > kSummarizeThreshold,
                              width};
        printer.block(0, 0);
    }
    os << ", shape=" << shape_to_string(shape_) << ')';
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    return os << tensor.to_string();
}

bool allclose(const Tensor& a, const Tensor& b, float rtol, float atol) {
    if (a.shape() != b.shape()) return false;
    const float* x = a.data();
    const float* y = b.data();
    for (std::size_t i = 0; i < a.numel(); ++i) {
        if (x[i] == y[i]) continue;  // exact match, including equal infinities
        // Written as !(diff <= tol) so that NaN, which fails every comparison, counts as
        // "not close".
        if (!(std::fabs(x[i] - y[i]) <= atol + rtol * std::fabs(y[i]))) return false;
    }
    return true;
}

}  // namespace inference
