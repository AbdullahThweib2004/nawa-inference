#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "inference/tensor/tensor.hpp"

// Reader for the IDX format used by the original MNIST files (t10k-images-idx3-ubyte, ...).
//
// Layout: 4-byte magic [0x00, 0x00, type, ndim], then ndim BIG-endian u32 dims, then the
// data in row-major order. Only type 0x08 (unsigned byte) is supported, which is what MNIST
// uses. Malformed files throw ModelFormatError (file, byte offset, expected vs. found).
namespace inference {

struct IdxArray {
    std::vector<std::size_t> dims;
    std::vector<std::uint8_t> data;  // row-major, product(dims) values
};

// Reads any unsigned-byte IDX file.
IdxArray read_idx(const std::string& path);

// MNIST images: an IDX file with dims {N, 28, 28}. Returns a {N, 784} tensor of RAW pixel
// values 0..255 (as floats), the same form Model::preprocess expects.
Tensor load_mnist_images(const std::string& path);

// MNIST labels: an IDX file with dims {N}, each value 0..9.
std::vector<std::uint8_t> load_mnist_labels(const std::string& path);

}  // namespace inference
