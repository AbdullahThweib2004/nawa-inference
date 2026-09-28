#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Low-level helpers for the binary formats in docs/model_format.md.
namespace inference {

// The file formats are little-endian, and the readers copy bytes straight into integers and
// floats. That is only correct on a little-endian machine (x86-64, ARM64 in its usual mode).
// A big-endian port would need to byte-swap every field.
static_assert(std::endian::native == std::endian::little,
              "Nawa's file readers assume a little-endian CPU (the file format is little-endian)");

// A file does not follow docs/model_format.md. The message names the file, the byte offset
// of the offending field, and what was expected vs. what was found.
class ModelFormatError : public std::runtime_error {
public:
    ModelFormatError(const std::string& source, std::size_t offset, const std::string& message);

    const std::string& source() const noexcept { return source_; }
    std::size_t offset() const noexcept { return offset_; }

private:
    std::string source_;
    std::size_t offset_;
};

// Reads an entire file into memory. Throws std::runtime_error if it can't be opened or read.
std::vector<std::byte> read_file_bytes(const std::string& path);

// Writes (and replaces) a file. Throws std::runtime_error on failure.
void write_file_bytes(const std::string& path, std::span<const std::byte> bytes);

// A cursor over an in-memory byte buffer. Every read is bounds-checked and advances the
// cursor; running past the end throws ModelFormatError instead of reading garbage.
//
// Each read takes a short description of the field (e.g. "layer 2 type id") that is used
// in error messages.
class BinaryReader {
public:
    // `source` names the data in error messages (usually the file path). The reader does not
    // own `data`: the buffer must outlive the reader.
    BinaryReader(std::span<const std::byte> data, std::string source);

    std::size_t offset() const noexcept { return offset_; }
    std::size_t remaining() const noexcept { return data_.size() - offset_; }
    const std::string& source() const noexcept { return source_; }

    std::uint8_t read_u8(std::string_view what);
    std::uint32_t read_u32(std::string_view what);
    std::uint64_t read_u64(std::string_view what);
    std::int32_t read_i32(std::string_view what);
    float read_f32(std::string_view what);

    // Reads `count` consecutive float32 values. The size is checked BEFORE allocating, so a
    // corrupt count can't trigger a huge allocation.
    std::vector<float> read_f32_array(std::size_t count, std::string_view what);

    // Reads 4 bytes and checks they equal `magic` (e.g. "NAWA").
    void expect_magic(std::string_view magic);

    // Throws if any bytes are left: a well-formed file ends exactly after its last field.
    void expect_end() const;

    // Throws ModelFormatError for the field that starts at byte `at`.
    [[noreturn]] void fail(std::size_t at, const std::string& message) const;

private:
    // Throws unless at least n more bytes are available.
    void require(std::size_t n, std::string_view what) const;

    template <typename T>
    T read_scalar(std::string_view what);

    std::span<const std::byte> data_;
    std::size_t offset_ = 0;
    std::string source_;
};

// Appends little-endian fields to a growing byte buffer (the mirror of BinaryReader).
class BinaryWriter {
public:
    void write_u8(std::uint8_t value);
    void write_u32(std::uint32_t value);
    void write_u64(std::uint64_t value);
    void write_i32(std::int32_t value);
    void write_f32(float value);
    void write_f32_array(std::span<const float> values);
    void write_bytes(std::span<const std::byte> bytes);

    const std::vector<std::byte>& bytes() const noexcept { return bytes_; }

private:
    template <typename T>
    void write_scalar(T value);

    std::vector<std::byte> bytes_;
};

}  // namespace inference
