#include "inference/tensor/ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "gemm_kernels.hpp"
#include "matmul_shapes.hpp"

namespace inference {

namespace {

// ---------------------------------------------------------------------------
// Broadcasting helpers
// ---------------------------------------------------------------------------

// Strides for reading an input of shape `in` while walking an output of shape `out`
// (out must be a broadcast of in). The result has one stride per OUTPUT dimension:
//  - a dimension that `in` doesn't have (missing on the left) gets stride 0;
//  - a dimension where `in` has size 1 gets stride 0;
//  - any other dimension keeps the normal row-major stride of `in`.
// Stride 0 means "moving along this output dimension doesn't move in the input", so
// the same input value is reused. That is the whole trick behind broadcasting: no data
// is ever copied or expanded.
Shape broadcast_strides(const Shape& in, const Shape& out) {
    const Shape in_strides = compute_strides(in);
    const std::size_t missing = out.size() - in.size();  // dims to add on the left
    Shape strides(out.size(), 0);
    for (std::size_t i = missing; i < out.size(); ++i) {
        const std::size_t in_dim = i - missing;
        strides[i] = (in[in_dim] == 1) ? 0 : in_strides[in_dim];
    }
    return strides;
}

// Applies `op` element-wise to a and b with broadcasting and returns a new tensor.
template <typename Op>
Tensor binary_op(const Tensor& a, const Tensor& b, Op op) {
    const float* pa = a.data();
    const float* pb = b.data();

    // Fast path: identical shapes need no index math, just one linear pass.
    if (a.shape() == b.shape()) {
        Tensor out(a.shape());
        float* po = out.data();
        for (std::size_t i = 0; i < out.numel(); ++i) po[i] = op(pa[i], pb[i]);
        return out;
    }

    Tensor out(broadcast_shape(a.shape(), b.shape()));
    const Shape& out_shape = out.shape();
    const Shape a_strides = broadcast_strides(a.shape(), out_shape);
    const Shape b_strides = broadcast_strides(b.shape(), out_shape);
    float* po = out.data();

    // Walk the output in row-major order. `index` holds the multi-dimensional position of
    // output element i; each input offset is sum(index[d] * stride[d]), the same formula
    // Tensor uses internally, just with the broadcast strides.
    Shape index(out_shape.size(), 0);
    for (std::size_t i = 0; i < out.numel(); ++i) {
        std::size_t a_off = 0;
        std::size_t b_off = 0;
        for (std::size_t d = 0; d < index.size(); ++d) {
            a_off += index[d] * a_strides[d];
            b_off += index[d] * b_strides[d];
        }
        po[i] = op(pa[a_off], pb[b_off]);

        // Advance `index` like an odometer: bump the last dimension, and when it
        // reaches its size, reset it to 0 and carry into the dimension before it.
        for (std::size_t d = index.size(); d-- > 0;) {
            if (++index[d] < out_shape[d]) break;
            index[d] = 0;
        }
    }
    return out;
}

// Applies `op(x)` to every element and returns a new tensor of the same shape.
template <typename Op>
Tensor unary_op(const Tensor& t, Op op) {
    Tensor out(t.shape());
    const float* pt = t.data();
    float* po = out.data();
    for (std::size_t i = 0; i < t.numel(); ++i) po[i] = op(pt[i]);
    return out;
}

// ---------------------------------------------------------------------------
// Reduction helpers
// ---------------------------------------------------------------------------

// Converts a possibly negative axis into an index in [0, ndim).
std::size_t normalize_axis(const Tensor& t, std::int64_t axis, const char* fn) {
    const auto ndim = static_cast<std::int64_t>(t.ndim());
    const std::int64_t resolved = axis < 0 ? axis + ndim : axis;
    if (resolved < 0 || resolved >= ndim) {
        throw std::out_of_range(std::string(fn) + ": axis " + std::to_string(axis) +
                                " is out of range for shape " + shape_to_string(t.shape()) +
                                " with " + std::to_string(ndim) + " dimensions");
    }
    return static_cast<std::size_t>(resolved);
}

// Reduces t along `axis`. For every output element, `reduce_slice(first, count, stride)`
// receives the 1-D slice being reduced: `count` values starting at `first`, `stride`
// elements apart, and returns the reduced value.
//
// The tensor is viewed as three blocks {outer, axis_len, inner}: `outer` is the product
// of the dimensions before the axis and `inner` the product of those after it. The
// slice for output position (o, j) then starts at o * axis_len * inner + j, with stride
// `inner`. Example: {2, 3} along axis 0 -> outer 1, axis_len 2, inner 3.
template <typename ReduceSlice>
Tensor reduce_axis(const Tensor& t, std::int64_t axis, bool keepdims, const char* fn,
                   ReduceSlice reduce_slice) {
    const std::size_t ax = normalize_axis(t, axis, fn);
    const Shape& shape = t.shape();

    std::size_t outer = 1;
    for (std::size_t d = 0; d < ax; ++d) outer *= shape[d];
    const std::size_t axis_len = shape[ax];
    std::size_t inner = 1;
    for (std::size_t d = ax + 1; d < shape.size(); ++d) inner *= shape[d];

    Shape out_shape = shape;
    if (keepdims) {
        out_shape[ax] = 1;
    } else {
        out_shape.erase(out_shape.begin() + static_cast<std::ptrdiff_t>(ax));
    }

    // Removing or shrinking the axis doesn't change the order of the remaining
    // positions, so the output is written in plain row-major order.
    Tensor out(std::move(out_shape));
    const float* pt = t.data();
    float* po = out.data();
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t j = 0; j < inner; ++j) {
            po[o * inner + j] = reduce_slice(pt + o * axis_len * inner + j, axis_len, inner);
        }
    }
    return out;
}

float slice_sum(const float* first, std::size_t count, std::size_t stride) {
    float total = 0.0f;
    for (std::size_t k = 0; k < count; ++k) total += first[k * stride];
    return total;
}

// Index of the largest value; ties keep the first one because only a strictly greater
// value replaces the current best.
std::size_t slice_argmax(const float* first, std::size_t count, std::size_t stride) {
    std::size_t best = 0;
    for (std::size_t k = 1; k < count; ++k) {
        if (first[k * stride] > first[best * stride]) best = k;
    }
    return best;
}

}  // namespace

// ---------------------------------------------------------------------------
// Matrix operations
// ---------------------------------------------------------------------------

Tensor matmul(const Tensor& a, const Tensor& b) { return matmul(a, b, GemmKernel::Auto); }

Tensor matmul(const Tensor& a, const Tensor& b, GemmKernel kernel) {
    const auto [M, K, N] = detail::check_matmul_shapes(a, b, "matmul");
    Tensor out({M, N});
    const GemmKernel chosen = resolve_kernel(kernel);
    // Packing B costs one pass over B (K*N values) on every call, because here B is not
    // known to be constant. With only a few rows of A there isn't enough work to repay that,
    // so small products use the unpacked i-k-j loop. (Linear packs its weights once at
    // construction and always takes the packed path.) Threshold measured in
    // docs/performance.md, stage 9.3.
    if (chosen == GemmKernel::Portable || M < kMatmulPackMinRows) {
        detail::gemm_ikj(a.data(), M, K, N, b.data(), out.data());
    } else {
        gemm(a.data(), M, PackedMatrix::pack(b.data(), K, N, chosen), out.data());
    }
    return out;
}

Tensor transpose(const Tensor& t) {
    if (t.ndim() != 2) {
        throw std::invalid_argument("transpose: expected a 2-D tensor, got shape " +
                                    shape_to_string(t.shape()));
    }
    const std::size_t rows = t.size(0);
    const std::size_t cols = t.size(1);
    Tensor out({cols, rows});
    const float* in = t.data();
    float* po = out.data();
    // Element (i, j) of the input becomes element (j, i) of the output.
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            po[j * rows + i] = in[i * cols + j];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Element-wise operations
// ---------------------------------------------------------------------------

Shape broadcast_shape(const Shape& a, const Shape& b) {
    const std::size_t ndim = std::max(a.size(), b.size());
    Shape result(ndim);
    // Walk both shapes from the right. A missing dimension behaves like size 1.
    for (std::size_t i = 0; i < ndim; ++i) {
        const std::size_t da = i < a.size() ? a[a.size() - 1 - i] : 1;
        const std::size_t db = i < b.size() ? b[b.size() - 1 - i] : 1;
        if (da != db && da != 1 && db != 1) {
            throw std::invalid_argument("broadcast_shape: shapes " + shape_to_string(a) + " and " +
                                        shape_to_string(b) +
                                        " are not broadcast-compatible (dimension " +
                                        std::to_string(da) + " vs " + std::to_string(db) + ")");
        }
        result[ndim - 1 - i] = std::max(da, db);
    }
    return result;
}

Tensor add(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x + y; });
}
Tensor sub(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x - y; });
}
Tensor mul(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x * y; });
}
Tensor div(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x / y; });
}

Tensor add(const Tensor& a, float s) {
    return unary_op(a, [s](float x) { return x + s; });
}
Tensor sub(const Tensor& a, float s) {
    return unary_op(a, [s](float x) { return x - s; });
}
Tensor mul(const Tensor& a, float s) {
    return unary_op(a, [s](float x) { return x * s; });
}
Tensor div(const Tensor& a, float s) {
    return unary_op(a, [s](float x) { return x / s; });
}

Tensor add(float s, const Tensor& b) {
    return unary_op(b, [s](float x) { return s + x; });
}
Tensor sub(float s, const Tensor& b) {
    return unary_op(b, [s](float x) { return s - x; });
}
Tensor mul(float s, const Tensor& b) {
    return unary_op(b, [s](float x) { return s * x; });
}
Tensor div(float s, const Tensor& b) {
    return unary_op(b, [s](float x) { return s / x; });
}

Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }

Tensor operator+(const Tensor& a, float s) { return add(a, s); }
Tensor operator-(const Tensor& a, float s) { return sub(a, s); }
Tensor operator*(const Tensor& a, float s) { return mul(a, s); }
Tensor operator/(const Tensor& a, float s) { return div(a, s); }

Tensor operator+(float s, const Tensor& b) { return add(s, b); }
Tensor operator-(float s, const Tensor& b) { return sub(s, b); }
Tensor operator*(float s, const Tensor& b) { return mul(s, b); }
Tensor operator/(float s, const Tensor& b) { return div(s, b); }

Tensor operator-(const Tensor& t) {
    return unary_op(t, [](float x) { return -x; });
}

Tensor exp(const Tensor& t) {
    return unary_op(t, [](float x) { return std::exp(x); });
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

Tensor sum(const Tensor& t, std::int64_t axis, bool keepdims) {
    return reduce_axis(t, axis, keepdims, "sum", slice_sum);
}

Tensor max(const Tensor& t, std::int64_t axis, bool keepdims) {
    return reduce_axis(t, axis, keepdims, "max",
                       [](const float* first, std::size_t count, std::size_t stride) {
                           return first[slice_argmax(first, count, stride) * stride];
                       });
}

Tensor mean(const Tensor& t, std::int64_t axis, bool keepdims) {
    return reduce_axis(t, axis, keepdims, "mean",
                       [](const float* first, std::size_t count, std::size_t stride) {
                           return slice_sum(first, count, stride) / static_cast<float>(count);
                       });
}

Tensor argmax(const Tensor& t, std::int64_t axis, bool keepdims) {
    return reduce_axis(t, axis, keepdims, "argmax",
                       [](const float* first, std::size_t count, std::size_t stride) {
                           return static_cast<float>(slice_argmax(first, count, stride));
                       });
}

Tensor sum(const Tensor& t) { return Tensor::scalar(slice_sum(t.data(), t.numel(), 1)); }

Tensor max(const Tensor& t) {
    const std::size_t best = slice_argmax(t.data(), t.numel(), 1);
    return Tensor::scalar(t.data()[best]);
}

}  // namespace inference
