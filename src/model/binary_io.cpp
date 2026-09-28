#include "inference/model/binary_io.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <utility>

namespace inference {

ModelFormatError::ModelFormatError(const std::string& source, std::size_t offset,
                                   const std::string& message)
    : std::runtime_error(source + ": byte offset " + std::to_string(offset) + ": " + message),
      source_(source),
      offset_(offset) {}

// ---------------------------------------------------------------------------
// Whole-file I/O
// ---------------------------------------------------------------------------

std::vector<std::byte> read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);  // ate: start at the end
    if (!in) throw std::runtime_error("cannot open file for reading: " + path);

    const std::streamsize size = in.tellg();
    if (size < 0) throw std::runtime_error("cannot determine size of file: " + path);
    in.seekg(0);

    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    // Viewing the buffer as char* is allowed: char (like std::byte) may alias any object.
    if (!in.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("error while reading file: " + path);
    }
    return bytes;
}

void write_file_bytes(const std::string& path, std::span<const std::byte> bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open file for writing: " + path);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("error while writing file: " + path);
}

// ---------------------------------------------------------------------------
// BinaryReader
// ---------------------------------------------------------------------------

BinaryReader::BinaryReader(std::span<const std::byte> data, std::string source)
    : data_(data), source_(std::move(source)) {}

void BinaryReader::fail(std::size_t at, const std::string& message) const {
    throw ModelFormatError(source_, at, message);
}

void BinaryReader::require(std::size_t n, std::string_view what) const {
    if (n > remaining()) {
        fail(offset_, "expected " + std::to_string(n) + " bytes for " + std::string(what) +
                          ", found only " + std::to_string(remaining()) + " (truncated file)");
    }
}

template <typename T>
T BinaryReader::read_scalar(std::string_view what) {
    require(sizeof(T), what);
    // Copy the bytes into a real T object instead of doing
    //     *reinterpret_cast<const T*>(data_.data() + offset_)
    // That cast would break C++'s STRICT ALIASING rule: an object may only be read through
    // a pointer to its own type (or char/std::byte). The buffer holds std::byte objects, not
    // a T, so reading them as a T is undefined behaviour, and the optimizer is allowed to
    // assume it never happens. The address may also be misaligned for T (fields are packed
    // with no padding). std::memcpy is always defined, and compilers turn this fixed-size
    // copy into a single load, so it costs nothing.
    T value;
    std::memcpy(&value, data_.data() + offset_, sizeof(T));
    offset_ += sizeof(T);
    return value;  // no byte swap needed: the file and the CPU are both little-endian
}

std::uint8_t BinaryReader::read_u8(std::string_view what) {
    return read_scalar<std::uint8_t>(what);
}
std::uint32_t BinaryReader::read_u32(std::string_view what) {
    return read_scalar<std::uint32_t>(what);
}
std::uint64_t BinaryReader::read_u64(std::string_view what) {
    return read_scalar<std::uint64_t>(what);
}
std::int32_t BinaryReader::read_i32(std::string_view what) {
    return read_scalar<std::int32_t>(what);
}
float BinaryReader::read_f32(std::string_view what) { return read_scalar<float>(what); }

std::span<const std::byte> BinaryReader::read_bytes(std::size_t n, std::string_view what) {
    require(n, what);
    const std::span<const std::byte> bytes = data_.subspan(offset_, n);
    offset_ += n;
    return bytes;
}

std::vector<float> BinaryReader::read_f32_array(std::size_t count, std::string_view what) {
    if (count > remaining() / sizeof(float)) {
        // Also covers counts so large that count * 4 would overflow.
        fail(offset_, "expected " + std::to_string(count) + " float32 values (" +
                          std::to_string(count) + " x 4 bytes) for " + std::string(what) +
                          ", found only " + std::to_string(remaining()) +
                          " bytes (truncated file)");
    }
    std::vector<float> values(count);
    std::memcpy(values.data(), data_.data() + offset_, count * sizeof(float));
    offset_ += count * sizeof(float);
    return values;
}

void BinaryReader::expect_magic(std::string_view magic) {
    const std::size_t start = offset_;
    require(magic.size(), "magic");
    std::string found;
    bool matches = true;
    for (std::size_t i = 0; i < magic.size(); ++i) {
        const auto c = static_cast<unsigned char>(data_[offset_ + i]);
        matches = matches && c == static_cast<unsigned char>(magic[i]);
        // Show printable bytes as-is and others as \xNN, so binary junk stays readable.
        if (c >= 0x20 && c < 0x7f) {
            found += static_cast<char>(c);
        } else {
            char hex[5];
            std::snprintf(hex, sizeof(hex), "\\x%02X", c);
            found += hex;
        }
    }
    offset_ += magic.size();
    if (!matches) {
        fail(start, "expected magic \"" + std::string(magic) + "\", found \"" + found + "\"");
    }
}

void BinaryReader::expect_end() const {
    if (remaining() != 0) {
        fail(offset_, "expected end of file, found " + std::to_string(remaining()) +
                          " unexpected trailing bytes");
    }
}

// ---------------------------------------------------------------------------
// BinaryWriter
// ---------------------------------------------------------------------------

template <typename T>
void BinaryWriter::write_scalar(T value) {
    // memcpy for the same aliasing reason as in the reader.
    std::byte raw[sizeof(T)];
    std::memcpy(raw, &value, sizeof(T));
    bytes_.insert(bytes_.end(), raw, raw + sizeof(T));
}

void BinaryWriter::write_u8(std::uint8_t value) { write_scalar(value); }
void BinaryWriter::write_u32(std::uint32_t value) { write_scalar(value); }
void BinaryWriter::write_u64(std::uint64_t value) { write_scalar(value); }
void BinaryWriter::write_i32(std::int32_t value) { write_scalar(value); }
void BinaryWriter::write_f32(float value) { write_scalar(value); }

void BinaryWriter::write_f32_array(std::span<const float> values) {
    write_bytes(std::as_bytes(values));
}

void BinaryWriter::write_bytes(std::span<const std::byte> bytes) {
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
}

}  // namespace inference
