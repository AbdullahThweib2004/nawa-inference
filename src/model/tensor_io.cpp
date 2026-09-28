#include "inference/model/tensor_io.hpp"

#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace inference {

namespace {

constexpr std::string_view kTensorFileMagic = "NTSR";
constexpr std::uint32_t kTensorFileVersion = 1;

}  // namespace

Tensor read_tensor_block(BinaryReader& reader, std::string_view what) {
    const std::string name(what);
    const std::size_t ndim_offset = reader.offset();
    const std::uint32_t ndim = reader.read_u32(name + " ndim");
    if (ndim > kMaxTensorBlockNdim) {
        reader.fail(ndim_offset, name +
                                     ": expected ndim <= " + std::to_string(kMaxTensorBlockNdim) +
                                     ", found " + std::to_string(ndim));
    }

    Shape shape;
    shape.reserve(ndim);
    std::size_t numel = 1;
    for (std::uint32_t i = 0; i < ndim; ++i) {
        const std::size_t dim_offset = reader.offset();
        const std::uint64_t dim = reader.read_u64(name + " dims[" + std::to_string(i) + "]");
        if (dim == 0) {
            reader.fail(dim_offset,
                        name + ": expected dims[" + std::to_string(i) + "] >= 1, found 0");
        }
        // Overflow check before multiplying (numel * dim must fit in size_t).
        if (dim > std::numeric_limits<std::size_t>::max() / numel) {
            reader.fail(dim_offset, name + ": element count overflows at dims[" +
                                        std::to_string(i) + "] = " + std::to_string(dim));
        }
        numel *= static_cast<std::size_t>(dim);
        shape.push_back(static_cast<std::size_t>(dim));
    }

    // read_f32_array checks that numel * 4 bytes are actually present before allocating.
    std::vector<float> data = reader.read_f32_array(numel, name + " data");
    return Tensor(std::move(shape), std::move(data));
}

void write_tensor_block(BinaryWriter& writer, const Tensor& tensor) {
    writer.write_u32(static_cast<std::uint32_t>(tensor.ndim()));
    for (std::size_t dim : tensor.shape()) writer.write_u64(dim);
    writer.write_f32_array({tensor.data(), tensor.numel()});
}

Tensor read_tensor_file(const std::string& path) {
    const std::vector<std::byte> bytes = read_file_bytes(path);
    BinaryReader reader(bytes, path);

    reader.expect_magic(kTensorFileMagic);
    const std::size_t version_offset = reader.offset();
    const std::uint32_t version = reader.read_u32("version");
    if (version != kTensorFileVersion) {
        reader.fail(version_offset, "expected tensor file version " +
                                        std::to_string(kTensorFileVersion) + ", found " +
                                        std::to_string(version));
    }
    Tensor tensor = read_tensor_block(reader, "tensor");
    reader.expect_end();
    return tensor;
}

void write_tensor_file(const std::string& path, const Tensor& tensor) {
    BinaryWriter writer;
    writer.write_bytes(std::as_bytes(std::span(kTensorFileMagic)));
    writer.write_u32(kTensorFileVersion);
    write_tensor_block(writer, tensor);
    write_file_bytes(path, writer.bytes());
}

}  // namespace inference
