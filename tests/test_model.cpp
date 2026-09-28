#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "inference/layers/layers.hpp"
#include "inference/model/binary_io.hpp"
#include "inference/model/model.hpp"
#include "inference/model/tensor_io.hpp"
#include "inference/tensor/ops.hpp"

using namespace inference;
namespace fs = std::filesystem;

// Set by tests/CMakeLists.txt so the tests find the committed files from any directory.
#ifndef NAWA_FIXTURES_DIR
#error "NAWA_FIXTURES_DIR must be defined by the build"
#endif
#ifndef NAWA_MODELS_DIR
#error "NAWA_MODELS_DIR must be defined by the build"
#endif

namespace {

const std::string kFixtures = NAWA_FIXTURES_DIR;
const std::string kModelPath = std::string(NAWA_MODELS_DIR) + "/mnist_mlp.nawa";

std::string fixture(const std::string& name) { return kFixtures + "/" + name + ".ntsr"; }

// A fresh, empty directory for the current test's temporary files.
fs::path temp_dir() {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    const fs::path dir = fs::path(::testing::TempDir()) /
                         (std::string("nawa_") + info->test_suite_name() + "_" + info->name());
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

// Builds binary files byte by byte, independently of the library's BinaryWriter, so a bug
// in the writer can't hide the same bug in the reader.
class Bytes {
public:
    Bytes& magic(const char* m) {
        for (int i = 0; i < 4; ++i) data_.push_back(static_cast<unsigned char>(m[i]));
        return *this;
    }
    Bytes& u8(std::uint8_t v) { return raw(&v, 1); }
    Bytes& u32(std::uint32_t v) { return raw(&v, 4); }
    Bytes& u64(std::uint64_t v) { return raw(&v, 8); }
    Bytes& i32(std::int32_t v) { return raw(&v, 4); }
    Bytes& f32(float v) { return raw(&v, 4); }
    // A tensor block with the given dims, filled with 0.1, 0.2, 0.3, ...
    Bytes& tensor(const std::vector<std::uint64_t>& dims) {
        u32(static_cast<std::uint32_t>(dims.size()));
        std::uint64_t numel = 1;
        for (auto d : dims) {
            u64(d);
            numel *= d;
        }
        for (std::uint64_t i = 0; i < numel; ++i) f32(0.1f * static_cast<float>(i + 1));
        return *this;
    }

    std::vector<unsigned char>& data() { return data_; }

    std::string write(const fs::path& path) const {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data_.data()),
                  static_cast<std::streamsize>(data_.size()));
        return path.string();
    }

private:
    Bytes& raw(const void* p, std::size_t n) {  // host is little-endian (static_assert)
        const auto* b = static_cast<const unsigned char*>(p);
        data_.insert(data_.end(), b, b + n);
        return *this;
    }
    std::vector<unsigned char> data_;
};

// Knobs for building a small, otherwise valid model:
//   input {2} -> Linear(2 -> 3) -> ReLU -> Linear(3 -> 2) -> Softmax(-1)
struct TinyModel {
    const char* magic = "NAWA";
    std::uint32_t version = 1;
    std::uint32_t input_dim = 2;
    float stddev = 1.0f;
    std::uint32_t second_layer_type = 2;  // ReLU
    std::uint8_t has_bias = 1;
    std::uint64_t bias_size = 3;
    std::uint64_t second_linear_in = 3;
    std::int32_t softmax_axis = -1;

    Bytes build() const {
        Bytes b;
        b.magic(magic).u32(version);
        b.u32(1).u64(input_dim);                   // input_dims
        b.f32(1.0f).u32(1).f32(0.0f).f32(stddev);  // pixel_scale, norm_count, mean, std
        b.u32(4);                                  // num_layers
        b.u32(1).u8(has_bias).tensor({2, 3});      // Linear(2 -> 3)
        if (has_bias == 1) b.tensor({bias_size});
        b.u32(second_layer_type);                                  // ReLU (normally)
        b.u32(1).u8(1).tensor({second_linear_in, 2}).tensor({2});  // Linear(3 -> 2)
        b.u32(4).i32(softmax_axis);                                // Softmax
        return b;
    }
};

// Largest absolute difference, and that difference divided by max(1, largest |expected|).
// The scaled value is what the Python verifier checks (see python/verify_export.py): float32
// rounding error grows with magnitude, so layers with large values need a relative measure.
struct ErrorStats {
    float max_abs;
    float scaled;
};

ErrorStats compare(const Tensor& actual, const Tensor& expected) {
    float max_abs = 0.0f;
    float max_ref = 0.0f;
    for (std::size_t i = 0; i < expected.numel(); ++i) {
        max_abs = std::max(max_abs, std::fabs(actual.data()[i] - expected.data()[i]));
        max_ref = std::max(max_ref, std::fabs(expected.data()[i]));
    }
    return {max_abs, max_abs / std::max(1.0f, max_ref)};
}

}  // namespace

// ---------------------------------------------------------------------------
// Tensor files
// ---------------------------------------------------------------------------

TEST(TensorFile, RoundTrip) {
    const fs::path dir = temp_dir();
    for (const Tensor& t : {Tensor::scalar(-2.5f), Tensor({3}, {1, 2, 3}),
                            Tensor({2, 1, 3}, {0.1f, -0.2f, 3e8f, 1e-7f, 0, -0.0f})}) {
        const std::string path = (dir / "t.ntsr").string();
        write_tensor_file(path, t);
        const Tensor back = read_tensor_file(path);
        EXPECT_EQ(back.shape(), t.shape());
        EXPECT_EQ(std::memcmp(back.data(), t.data(), t.numel() * sizeof(float)), 0);
    }
}

TEST(TensorFile, WriterMatchesSpecExampleByteForByte) {
    // The 2x3 tensor block example from docs/model_format.md, after the 8-byte file header.
    const std::string path = (temp_dir() / "spec.ntsr").string();
    write_tensor_file(path, Tensor({2, 3}, {1, 2, 3, 4, 5, 6}));
    const std::vector<std::byte> bytes = read_file_bytes(path);

    const std::vector<unsigned char> expected = {
        'N', 'T', 'S',  'R',  1, 0, 0,    0,     // magic, version
        2,   0,   0,    0,                       // ndim
        2,   0,   0,    0,    0, 0, 0,    0,     // dims[0]
        3,   0,   0,    0,    0, 0, 0,    0,     // dims[1]
        0,   0,   0x80, 0x3F, 0, 0, 0,    0x40,  // 1.0, 2.0
        0,   0,   0x40, 0x40, 0, 0, 0x80, 0x40,  // 3.0, 4.0
        0,   0,   0xA0, 0x40, 0, 0, 0xC0, 0x40,  // 5.0, 6.0
    };
    ASSERT_EQ(bytes.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(static_cast<unsigned char>(bytes[i]), expected[i]) << "byte " << i;
    }
}

TEST(TensorFile, ReadsCommittedFixture) {
    const Tensor labels = read_tensor_file(fixture("mnist_test100_labels"));
    EXPECT_EQ(labels.shape(), (Shape{100}));
    for (std::size_t i = 0; i < labels.numel(); ++i) {
        const float v = labels.data()[i];
        EXPECT_TRUE(v >= 0.0f && v <= 9.0f && v == std::floor(v)) << "label " << i << " = " << v;
    }
    EXPECT_EQ(read_tensor_file(fixture("mnist_test100_images")).shape(), (Shape{100, 784}));
}

TEST(TensorFile, MalformedFilesThrow) {
    const fs::path dir = temp_dir();
    auto header = [] { return Bytes().magic("NTSR").u32(1); };

    EXPECT_THROW(read_tensor_file(Bytes().magic("NAWA").u32(1).tensor({2}).write(dir / "a")),
                 ModelFormatError);  // bad magic
    EXPECT_THROW(read_tensor_file(Bytes().magic("NTSR").u32(2).tensor({2}).write(dir / "b")),
                 ModelFormatError);  // bad version
    EXPECT_THROW(read_tensor_file(header().u32(9).write(dir / "c")), ModelFormatError);  // ndim 9
    EXPECT_THROW(read_tensor_file(header().u32(1).u64(0).write(dir / "d")),
                 ModelFormatError);  // dim 0
    EXPECT_THROW(read_tensor_file(header().u32(2).u64(1ull << 40).u64(1ull << 40).write(dir / "e")),
                 ModelFormatError);  // numel overflow
    // A huge but non-overflowing element count with no data behind it must fail cleanly
    // (without trying to allocate 4 TB first).
    EXPECT_THROW(read_tensor_file(header().u32(1).u64(1ull << 40).write(dir / "f")),
                 ModelFormatError);
    Bytes truncated = header().tensor({2, 2});
    truncated.data().pop_back();
    EXPECT_THROW(read_tensor_file(truncated.write(dir / "g")), ModelFormatError);
    EXPECT_THROW(read_tensor_file(header().tensor({2}).u8(0).write(dir / "h")),
                 ModelFormatError);  // trailing byte
}

TEST(TensorFile, MissingFileIsAnIoError) {
    EXPECT_THROW(read_tensor_file("/nonexistent/nawa/file.ntsr"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Malformed model files
// ---------------------------------------------------------------------------

TEST(ModelFormat, TinyValidModelLoads) {
    const Model model(TinyModel{}.build().write(temp_dir() / "tiny.nawa"));
    ASSERT_EQ(model.layers().size(), 4u);
    EXPECT_EQ(model.input_features(), 2u);
    const Tensor y = model.predict(Tensor({1, 2}, {1.0f, 2.0f}));
    EXPECT_EQ(y.shape(), (Shape{1, 2}));
    EXPECT_NEAR(sum(y).at({}), 1.0f, 1e-6f);
}

TEST(ModelFormat, BadMagic) {
    TinyModel spec;
    spec.magic = "NAWB";
    EXPECT_THROW(Model::load(spec.build().write(temp_dir() / "m")), ModelFormatError);
}

TEST(ModelFormat, BadVersion) {
    TinyModel spec;
    spec.version = 3;  // 1 and 2 are valid
    EXPECT_THROW(Model::load(spec.build().write(temp_dir() / "m")), ModelFormatError);
}

TEST(ModelFormat, UnknownLayerType) {
    TinyModel spec;
    spec.second_layer_type = 99;
    EXPECT_THROW(Model::load(spec.build().write(temp_dir() / "m")), ModelFormatError);
}

TEST(ModelFormat, HasBiasTwo) {
    TinyModel spec;
    spec.has_bias = 2;
    EXPECT_THROW(Model::load(spec.build().write(temp_dir() / "m")), ModelFormatError);
}

TEST(ModelFormat, TruncatedAtEveryLength) {
    // Cutting a valid file at ANY length must be detected, never crash or read garbage.
    const fs::path dir = temp_dir();
    Bytes full = TinyModel{}.build();
    const std::size_t size = full.data().size();
    for (std::size_t len = 0; len < size; ++len) {
        Bytes cut = full;
        cut.data().resize(len);
        EXPECT_THROW(Model::load(cut.write(dir / "m")), ModelFormatError) << "length " << len;
    }
}

TEST(ModelFormat, TrailingBytes) {
    Bytes b = TinyModel{}.build();
    b.u8(0);
    EXPECT_THROW(Model::load(b.write(temp_dir() / "m")), ModelFormatError);
}

TEST(ModelFormat, BiasShapeMismatch) {
    TinyModel spec;
    spec.bias_size = 2;  // weight is {2, 3}, so the bias must be {3}
    EXPECT_THROW(Model::load(spec.build().write(temp_dir() / "m")), ModelFormatError);
}

TEST(ModelFormat, BrokenLinearChain) {
    TinyModel spec;
    spec.second_linear_in = 4;  // previous Linear outputs 3
    EXPECT_THROW(Model::load(spec.build().write(temp_dir() / "m")), ModelFormatError);
}

TEST(ModelFormat, FirstLinearMustMatchInputSize) {
    TinyModel spec;
    spec.input_dim = 5;  // first Linear takes 2
    EXPECT_THROW(Model::load(spec.build().write(temp_dir() / "m")), ModelFormatError);
}

TEST(ModelFormat, InvalidMetadataAndAxis) {
    TinyModel zero_std;
    zero_std.stddev = 0.0f;
    EXPECT_THROW(Model::load(zero_std.build().write(temp_dir() / "a")), ModelFormatError);
    TinyModel bad_axis;
    bad_axis.softmax_axis = 2;
    EXPECT_THROW(Model::load(bad_axis.build().write(temp_dir() / "b")), ModelFormatError);
}

TEST(ModelFormat, ErrorMessageHasPathOffsetExpectedAndFound) {
    TinyModel spec;
    spec.second_layer_type = 99;
    const std::string path = spec.build().write(temp_dir() / "bad_type.nawa");
    try {
        Model::load(path);
        FAIL() << "expected ModelFormatError";
    } catch (const ModelFormatError& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find(path), std::string::npos) << msg;
        EXPECT_NE(msg.find("byte offset"), std::string::npos) << msg;
        EXPECT_NE(msg.find("found 99"), std::string::npos) << msg;
        EXPECT_EQ(e.source(), path);
        // Header (8) + input_ndim/dims (12) + normalization (16) + num_layers (4)
        // + Linear: type id (4) + has_bias (1) + weight block (4 + 16 + 24) + bias block
        // (4 + 8 + 12) = the second layer's type id starts at byte 113.
        EXPECT_EQ(e.offset(), 113u);
    }
}

// ---------------------------------------------------------------------------
// The real MNIST model
// ---------------------------------------------------------------------------

class MnistModel : public ::testing::Test {
protected:
    static void SetUpTestSuite() { model_ = new Model(kModelPath); }
    static void TearDownTestSuite() {
        delete model_;
        model_ = nullptr;
    }
    static const Model& model() { return *model_; }

private:
    static Model* model_;  // loaded once and shared by every test in this suite
};
Model* MnistModel::model_ = nullptr;

TEST_F(MnistModel, Metadata) {
    const auto& meta = model().metadata();
    EXPECT_EQ(meta.input_shape, (Shape{784}));
    EXPECT_FLOAT_EQ(meta.pixel_scale, 1.0f / 255.0f);
    ASSERT_EQ(meta.mean.size(), 1u);
    EXPECT_FLOAT_EQ(meta.mean[0], 0.1307f);
    EXPECT_FLOAT_EQ(meta.stddev[0], 0.3081f);
    EXPECT_EQ(model().input_features(), 784u);
}

TEST_F(MnistModel, LayersAndParameterCount) {
    const auto& layers = model().layers();
    ASSERT_EQ(layers.size(), 4u);
    EXPECT_EQ(layers[0]->name(), "Linear(784 -> 128)");
    EXPECT_EQ(layers[1]->name(), "ReLU");
    EXPECT_EQ(layers[2]->name(), "Linear(128 -> 10)");
    EXPECT_EQ(layers[3]->name(), "Softmax(axis=-1)");
    EXPECT_EQ(model().num_parameters(), 101770u);  // 784*128 + 128 + 128*10 + 10

    const std::string summary = model().summary();
    EXPECT_NE(summary.find("101,770"), std::string::npos) << summary;
    EXPECT_NE(summary.find("Linear(784 -> 128)"), std::string::npos) << summary;
}

TEST_F(MnistModel, MatchesPyTorchLayerByLayer) {
    const Tensor images = read_tensor_file(fixture("mnist_test100_images"));
    // The first 3 images: rows 0..2 of the {100, 784} tensor are its first 3 * 784 values.
    const Tensor first3({3, 784}, std::vector<float>(images.data(), images.data() + 3 * 784));

    const Tensor normalized = model().preprocess(first3);
    const ErrorStats pre =
        compare(normalized, read_tensor_file(fixture("mnist_test3_input_normalized")));
    EXPECT_LT(pre.scaled, 1e-5f) << "stage 'normalize' failed: max abs error " << pre.max_abs;

    const std::vector<Tensor> trace = model().forward_trace(normalized);
    const char* names[] = {"linear1", "relu", "linear2", "softmax"};
    ASSERT_EQ(trace.size(), 4u);
    for (std::size_t i = 0; i < trace.size(); ++i) {
        const Tensor expected = read_tensor_file(fixture(std::string("mnist_test3_") + names[i]));
        ASSERT_EQ(trace[i].shape(), expected.shape()) << "layer " << names[i];
        const ErrorStats err = compare(trace[i], expected);
        EXPECT_LT(err.scaled, 1e-5f)
            << "layer " << i << " (" << names[i] << ") differs from PyTorch: max abs error "
            << err.max_abs << ", scaled error " << err.scaled;
    }
}

TEST_F(MnistModel, MatchesPyTorchOnAll100Images) {
    const Tensor images = read_tensor_file(fixture("mnist_test100_images"));
    const Tensor labels = read_tensor_file(fixture("mnist_test100_labels"));
    const Tensor expected = read_tensor_file(fixture("mnist_test100_probs"));

    const Tensor probs = model().predict(images);
    ASSERT_EQ(probs.shape(), (Shape{100, 10}));
    const ErrorStats err = compare(probs, expected);
    EXPECT_LT(err.max_abs, 1e-5f) << "max abs error vs PyTorch probabilities: " << err.max_abs;

    const Tensor ours = argmax(probs, -1);
    const Tensor theirs = argmax(expected, -1);
    std::size_t correct = 0;
    for (std::size_t n = 0; n < 100; ++n) {
        EXPECT_EQ(ours.at({n}), theirs.at({n})) << "image " << n;
        if (ours.at({n}) == labels.at({n})) ++correct;
    }
    std::cout << "[          ] max abs error " << err.max_abs << ", accuracy " << correct
              << "/100 vs labels\n";
    EXPECT_GE(correct, 95u);
}

TEST_F(MnistModel, SingleSampleMatchesBatchRow) {
    const Tensor images = read_tensor_file(fixture("mnist_test100_images"));
    const Tensor one(Shape{784}, std::vector<float>(images.data(), images.data() + 784));
    const Tensor single = model().predict(one);
    EXPECT_EQ(single.shape(), (Shape{1, 10}));

    const Tensor batch = model().predict(images);
    const Tensor row0({1, 10}, std::vector<float>(batch.data(), batch.data() + 10));
    EXPECT_TRUE(allclose(single, row0, 0.0f, 0.0f));  // bit-identical
}

TEST_F(MnistModel, ClassifyReturnsOnePredictionPerRow) {
    const Tensor images = read_tensor_file(fixture("mnist_test100_images"));
    const Tensor probs = read_tensor_file(fixture("mnist_test100_probs"));
    const auto predictions = model().classify(images);
    ASSERT_EQ(predictions.size(), 100u);
    for (std::size_t n = 0; n < 5; ++n) {
        EXPECT_EQ(static_cast<float>(predictions[n].label), argmax(probs, -1).at({n}));
        EXPECT_NEAR(predictions[n].confidence, max(probs, -1).at({n}), 1e-5f);
    }
}

TEST_F(MnistModel, WrongInputShapeThrows) {
    EXPECT_THROW(model().predict(Tensor({2, 783})), std::invalid_argument);
    EXPECT_THROW(model().predict(Tensor({28, 28})), std::invalid_argument);
    EXPECT_THROW(model().forward(Tensor({1, 1, 784})), std::invalid_argument);
    EXPECT_THROW(model().preprocess(Tensor::scalar(0.0f)), std::invalid_argument);
}
