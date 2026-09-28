#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "inference/model/binary_io.hpp"
#include "inference/tensor/tensor.hpp"

// Tensor blocks and standalone tensor files (.ntsr), as specified in docs/model_format.md.
namespace inference {

// Largest ndim a tensor block may have (the spec's limit).
inline constexpr std::uint32_t kMaxTensorBlockNdim = 8;

// Reads one tensor block at the reader's position: u32 ndim, u64 dims[ndim], f32 data.
// Throws ModelFormatError if ndim > 8, a dim is 0, the element or byte count overflows, or
// the data is truncated. `what` names the tensor in error messages (e.g. "layer 0 weight").
Tensor read_tensor_block(BinaryReader& reader, std::string_view what);

// Appends a tensor block for `tensor`.
void write_tensor_block(BinaryWriter& writer, const Tensor& tensor);

// Reads a .ntsr file: magic "NTSR", u32 version 1, one tensor block, nothing after it.
Tensor read_tensor_file(const std::string& path);

// Writes a .ntsr file (replacing any existing file).
void write_tensor_file(const std::string& path, const Tensor& tensor);

}  // namespace inference
