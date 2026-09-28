#include "inference/data/idx.hpp"

#include <cstring>
#include <limits>
#include <span>
#include <utility>

#include "inference/model/binary_io.hpp"

namespace inference {

namespace {

constexpr std::uint8_t kTypeUnsignedByte = 0x08;
constexpr std::size_t kMnistSide = 28;

// Reverses the byte order of a 32-bit value: 0x11223344 <-> 0x44332211.
// (C++23 has std::byteswap; this project targets C++20.)
std::uint32_t byteswap32(std::uint32_t v) {
    return ((v & 0x000000FFu) << 24) | ((v & 0x0000FF00u) << 8) | ((v & 0x00FF0000u) >> 8) |
           ((v & 0xFF000000u) >> 24);
}

// Reads a big-endian u32. The bytes are copied with memcpy (no pointer casts, for the
// strict-aliasing reasons explained in binary_io.cpp) into a native integer, which on our
// little-endian CPU holds them in the wrong order, so they are swapped afterwards. The
// static_assert in binary_io.hpp guarantees the CPU really is little-endian.
std::uint32_t read_u32_be(BinaryReader& r, std::string_view what) {
    const std::span<const std::byte> bytes = r.read_bytes(4, what);
    std::uint32_t value;
    std::memcpy(&value, bytes.data(), 4);
    return byteswap32(value);
}

}  // namespace

IdxArray read_idx(const std::string& path) {
    const std::vector<std::byte> bytes = read_file_bytes(path);
    BinaryReader r(bytes, path);

    // Magic: two zero bytes, the element type, and the number of dimensions.
    const std::span<const std::byte> magic = r.read_bytes(4, "IDX magic");
    const auto m0 = static_cast<unsigned>(magic[0]);
    const auto m1 = static_cast<unsigned>(magic[1]);
    const auto type = static_cast<unsigned>(magic[2]);
    const auto ndim = static_cast<unsigned>(magic[3]);
    if (m0 != 0 || m1 != 0) {
        r.fail(0, "expected IDX magic to start with bytes 00 00, found " + std::to_string(m0) +
                      " " + std::to_string(m1) + " (not an IDX file?)");
    }
    if (type != kTypeUnsignedByte) {
        r.fail(2, "expected IDX type 0x08 (unsigned byte), found " + std::to_string(type));
    }
    if (ndim == 0) r.fail(3, "expected at least 1 IDX dimension, found 0");

    IdxArray result;
    std::size_t count = 1;
    for (unsigned i = 0; i < ndim; ++i) {
        const std::size_t dim_offset = r.offset();
        const std::uint32_t dim = read_u32_be(r, "IDX dims[" + std::to_string(i) + "]");
        if (dim == 0) {
            r.fail(dim_offset, "expected IDX dims[" + std::to_string(i) + "] >= 1, found 0");
        }
        if (dim > std::numeric_limits<std::size_t>::max() / count) {
            r.fail(dim_offset, "IDX element count overflows");
        }
        count *= dim;
        result.dims.push_back(dim);
    }

    // One byte per element: the file size must match the header exactly.
    const std::span<const std::byte> data = r.read_bytes(count, "IDX data");
    r.expect_end();
    result.data.resize(count);
    std::memcpy(result.data.data(), data.data(), count);
    return result;
}

Tensor load_mnist_images(const std::string& path) {
    IdxArray idx = read_idx(path);
    if (idx.dims.size() != 3 || idx.dims[1] != kMnistSide || idx.dims[2] != kMnistSide) {
        throw ModelFormatError(
            path, 0, "expected MNIST image dims [N, 28, 28], found " + shape_to_string(idx.dims));
    }
    const std::size_t n = idx.dims[0];
    std::vector<float> pixels(idx.data.begin(), idx.data.end());  // uint8 -> float, 0..255
    return Tensor({n, kMnistSide * kMnistSide}, std::move(pixels));
}

std::vector<std::uint8_t> load_mnist_labels(const std::string& path) {
    IdxArray idx = read_idx(path);
    if (idx.dims.size() != 1) {
        throw ModelFormatError(path, 0,
                               "expected MNIST label dims [N], found " + shape_to_string(idx.dims));
    }
    for (std::size_t i = 0; i < idx.data.size(); ++i) {
        if (idx.data[i] > 9) {
            throw ModelFormatError(path, 8 + i,
                                   "expected a label 0..9 at index " + std::to_string(i) +
                                       ", found " + std::to_string(idx.data[i]));
        }
    }
    return std::move(idx.data);
}

}  // namespace inference
