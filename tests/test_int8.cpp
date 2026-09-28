// Stage 9.6: INT8 quantization, kernels, the LinearInt8 layer, and format version 2.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "inference/data/idx.hpp"
#include "inference/layers/linear.hpp"
#include "inference/layers/linear_int8.hpp"
#include "inference/model/binary_io.hpp"
#include "inference/model/model.hpp"
#include "inference/model/tensor_io.hpp"
#include "inference/runtime/thread_pool.hpp"
#include "inference/runtime/workspace.hpp"
#include "inference/tensor/int8.hpp"
#include "inference/tensor/ops.hpp"
#include "inference/tensor/reference.hpp"
#include "test_paths.hpp"

using namespace inference;
namespace fs = std::filesystem;

namespace {

const std::string kModel = std::string(NAWA_MODELS_DIR) + "/mnist_mlp.nawa";
const std::string kModelInt8 = std::string(NAWA_MODELS_DIR) + "/mnist_mlp_int8.nawa";
const std::string kFixtures = NAWA_FIXTURES_DIR;

Tensor random_tensor(const Shape& shape, std::uint32_t seed) {
    std::vector<float> data(numel_of(shape));
    std::uint32_t state = seed * 2654435761u + 5u;
    for (float& v : data) {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>(state >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
    }
    return Tensor(shape, std::move(data));
}

bool bit_equal(const Tensor& a, const Tensor& b) {
    return a.shape() == b.shape() &&
           std::memcmp(a.data(), b.data(), a.numel() * sizeof(float)) == 0;
}

fs::path temp_file(const std::string& name) { return test_temp_dir(false) / name; }

std::vector<GemmKernel> kernels() {
    std::vector<GemmKernel> out;
    for (GemmKernel k : {GemmKernel::Portable, GemmKernel::Avx2})
        if (kernel_available(k)) out.push_back(k);
    return out;
}

struct DefaultThreadsGuard {
    ~DefaultThreadsGuard() { set_num_threads(default_num_threads()); }
};

}  // namespace

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

TEST(Int8Quantize, PerChannelScalesAndRoundingError) {
    const std::size_t K = 37, N = 11;
    Tensor w = random_tensor({K, N}, 1);
    for (std::size_t k = 0; k < K; ++k) w.at({k, 3}) = 0.0f;  // an all-zero channel
    const QuantizedMatrix q = QuantizedMatrix::quantize(w.data(), K, N);
    EXPECT_EQ(q.padded_in() % 16, 0u);
    for (std::size_t j = 0; j < N; ++j) {
        float max_abs = 0.0f;
        for (std::size_t k = 0; k < K; ++k) max_abs = std::max(max_abs, std::fabs(w.at({k, j})));
        EXPECT_FLOAT_EQ(q.scales()[j], j == 3 ? 1.0f : max_abs / 127.0f);
        for (std::size_t k = 0; k < K; ++k) {
            EXPECT_GE(q.at(k, j), -127);
            EXPECT_LE(q.at(k, j), 127);
            // Rounding to the nearest step: at most half a step away.
            EXPECT_LE(std::fabs(q.dequantized(k, j) - w.at({k, j})), q.scales()[j] * 0.5001f);
        }
        for (std::size_t k = K; k < q.padded_in(); ++k)
            EXPECT_EQ(q.data()[j * q.padded_in() + k], 0);
    }
}

TEST(Int8Quantize, RowQuantization) {
    const float x[5] = {0.5f, -1.0f, 0.25f, 0.0f, 1.0f / 254.0f};
    std::int16_t q[16];
    const float scale = quantize_row(x, 5, 16, q);
    EXPECT_FLOAT_EQ(scale, 1.0f / 127.0f);
    EXPECT_EQ(q[1], -127);
    EXPECT_EQ(q[0], 64);  // 63.5 rounds to even
    for (int k = 5; k < 16; ++k) EXPECT_EQ(q[k], 0);
    const float zeros[3] = {0, 0, 0};
    EXPECT_FLOAT_EQ(quantize_row(zeros, 3, 16, q), 1.0f);
}

// ---------------------------------------------------------------------------
// Kernels: scalar reference vs AVX2, threads, and the error bound vs float32
// ---------------------------------------------------------------------------

TEST(Int8Gemm, KernelsAndThreadCountsAreBitIdentical) {
    DefaultThreadsGuard guard;
    for (const auto& [M, K, N] : std::vector<std::array<std::size_t, 3>>{{1, 1, 1},
                                                                         {1, 784, 128},
                                                                         {3, 17, 5},
                                                                         {7, 300, 13},
                                                                         {256, 784, 128},
                                                                         {256, 128, 10},
                                                                         {2, 1000, 3},
                                                                         {64, 784, 128}}) {
        SCOPED_TRACE(::testing::Message() << M << "x" << K << "x" << N);
        const Tensor a = random_tensor({M, K}, 2);
        const Tensor w = random_tensor({K, N}, 3);
        const Tensor bias = random_tensor({N}, 4);
        const QuantizedMatrix q = QuantizedMatrix::quantize(w.data(), K, N);
        set_num_threads(1);
        Tensor reference({M, N});
        gemm_int8(a.data(), M, q, reference.data(), {bias.data(), true}, GemmKernel::Portable);
        for (GemmKernel kernel : kernels()) {
            for (std::size_t threads : {1u, 4u}) {
                set_num_threads(threads);
                Tensor c({M, N});
                gemm_int8(a.data(), M, q, c.data(), {bias.data(), true}, kernel);
                EXPECT_TRUE(bit_equal(c, reference))
                    << kernel_name(kernel) << ", " << threads << " threads";
            }
        }
    }
}

TEST(Int8Gemm, ErrorIsWithinTheQuantizationBound) {
    // With |a - q_a*s_a| <= s_a/2 and |w - q_w*s_w| <= s_w/2 per value, the int8 result
    // differs from the exact product by at most
    //   s_w/2 * sum|a| + s_a/2 * sum|w| + K * s_a * s_w / 4    (+ float rounding).
    const std::size_t M = 16, K = 784, N = 32;
    const Tensor a = random_tensor({M, K}, 5);
    const Tensor w = random_tensor({K, N}, 6);
    const QuantizedMatrix q = QuantizedMatrix::quantize(w.data(), K, N);
    Tensor c({M, N});
    gemm_int8(a.data(), M, q, c.data());
    const Tensor exact = matmul_naive(a, w);
    for (std::size_t m = 0; m < M; ++m) {
        float max_a = 0.0f, sum_a = 0.0f;
        for (std::size_t k = 0; k < K; ++k) {
            max_a = std::max(max_a, std::fabs(a.at({m, k})));
            sum_a += std::fabs(a.at({m, k}));
        }
        const float s_a = max_a / 127.0f;
        for (std::size_t j = 0; j < N; ++j) {
            float sum_w = 0.0f;
            for (std::size_t k = 0; k < K; ++k) sum_w += std::fabs(w.at({k, j}));
            const float s_w = q.scales()[j];
            const float bound = s_w / 2 * sum_a + s_a / 2 * sum_w + K * s_a * s_w / 4 + 1e-3f;
            EXPECT_LE(std::fabs(c.at({m, j}) - exact.at({m, j})), bound) << m << "," << j;
        }
    }
}

TEST(Int8Gemm, SteadyStateIsDeterministicAcrossCalls) {
    const Tensor a = random_tensor({8, 100}, 7);
    const Tensor w = random_tensor({100, 20}, 8);
    const QuantizedMatrix q = QuantizedMatrix::quantize(w.data(), 100, 20);
    Tensor c1({8, 20}), c2({8, 20});
    gemm_int8(a.data(), 8, q, c1.data());
    gemm_int8(a.data(), 8, q, c2.data());
    EXPECT_TRUE(bit_equal(c1, c2));
}

// ---------------------------------------------------------------------------
// LinearInt8 layer and quantized models
// ---------------------------------------------------------------------------

TEST(LinearInt8, LayerBasics) {
    const Linear fp32(random_tensor({20, 6}, 9), random_tensor({6}, 10));
    const LinearInt8 q = LinearInt8::from_linear(fp32);
    EXPECT_EQ(q.name(), "LinearInt8(20 -> 6)");
    EXPECT_EQ(q.num_parameters(), fp32.num_parameters());
    const Tensor x = random_tensor({3, 20}, 11);
    const Tensor y = q.forward(x);
    EXPECT_TRUE(allclose(y, fp32.forward(x), 0.0f, 0.05f));  // close to float32
    Tensor into({3, 6});
    q.forward_into(x.data(), 3, into.data());
    EXPECT_TRUE(bit_equal(into, y));
    EXPECT_THROW(q.forward(Tensor({3, 21})), std::invalid_argument);
    EXPECT_THROW(LinearInt8(q.weight(), Tensor({5})), std::invalid_argument);
}

TEST(QuantizedModel, CloseToFloatOnFixtures) {
    const Model fp32(kModel);
    const Model int8 = fp32.quantize();
    EXPECT_EQ(int8.format_version(), 2u);
    EXPECT_EQ(int8.plan().size(), 3u);  // LinearInt8 + ReLU still fused
    EXPECT_TRUE(int8.plan()[0].fuse_relu);
    const Tensor images = read_tensor_file(kFixtures + "/mnist_test100_images.ntsr");
    const Tensor p32 = fp32.predict(images);
    const Tensor p8 = int8.predict(images);
    float max_diff = 0.0f;
    std::size_t changed = 0;
    for (std::size_t n = 0; n < 100; ++n) {
        std::size_t a32 = 0, a8 = 0;
        for (std::size_t j = 0; j < 10; ++j) {
            max_diff = std::max(max_diff, std::fabs(p32.at({n, j}) - p8.at({n, j})));
            if (p32.at({n, j}) > p32.at({n, a32})) a32 = j;
            if (p8.at({n, j}) > p8.at({n, a8})) a8 = j;
        }
        changed += a32 != a8;
    }
    std::cout << "[          ] 100 fixtures: max probability difference " << max_diff << ", "
              << changed << " predictions changed\n";
    EXPECT_LE(changed, 1u);
    EXPECT_LT(max_diff, 0.1f);
}

// ---------------------------------------------------------------------------
// Saving and format version 2
// ---------------------------------------------------------------------------

TEST(ModelSave, Float32ModelSavesByteIdentical) {
    const Model model(kModel);
    const std::string out = temp_file("copy.nawa").string();
    model.save(out);
    EXPECT_EQ(model.format_version(), 1u);
    EXPECT_EQ(read_file_bytes(out), read_file_bytes(kModel));  // matches the Python writer
}

TEST(ModelSave, Int8RoundTripAndCommittedFile) {
    const Model int8 = Model(kModel).quantize();
    const std::string out = temp_file("int8.nawa").string();
    int8.save(out);
    const std::vector<std::byte> bytes = read_file_bytes(out);
    std::uint32_t version = 0;
    std::memcpy(&version, bytes.data() + 4, 4);
    EXPECT_EQ(version, 2u);
    // Quantization is deterministic: the committed file is exactly what quantize() produces.
    EXPECT_EQ(bytes, read_file_bytes(kModelInt8));

    const Model loaded(out);
    const Tensor images = read_tensor_file(kFixtures + "/mnist_test100_images.ntsr");
    EXPECT_TRUE(bit_equal(loaded.predict(images), int8.predict(images)));
    EXPECT_EQ(loaded.layers()[0]->name(), "LinearInt8(784 -> 128)");
}

TEST(ModelSave, MalformedInt8LayersAreRejected) {
    const std::vector<std::byte> good = read_file_bytes(kModelInt8);
    // Layout after the 40-byte header/metadata: u32 type (5), u8 has_bias, u32 in, u32 out,
    // f32 scales[out], i8 weights[out*in], bias block.
    const std::size_t type_at = 40, in_at = 45, scale0_at = 53, weights_at = 53 + 128 * 4;
    const auto write_mutated = [&](const std::string& name, auto mutate) {
        std::vector<std::byte> bytes = good;
        mutate(bytes);
        const std::string path = temp_file(name).string();
        write_file_bytes(path, bytes);
        return path;
    };
    std::uint32_t type = 0;
    std::memcpy(&type, good.data() + type_at, 4);
    ASSERT_EQ(type, 5u);

    // Type 5 in a version-1 file.
    EXPECT_THROW(Model::load(write_mutated("v1.nawa", [](auto& b) { b[4] = std::byte{1}; })),
                 ModelFormatError);
    // A zero scale.
    EXPECT_THROW(Model::load(write_mutated(
                     "scale.nawa", [&](auto& b) { std::memset(b.data() + scale0_at, 0, 4); })),
                 ModelFormatError);
    // A weight of -128 (outside the symmetric range).
    EXPECT_THROW(
        Model::load(write_mutated("w.nawa", [&](auto& b) { b[weights_at] = std::byte{0x80}; })),
        ModelFormatError);
    // in_features = 0.
    EXPECT_THROW(Model::load(write_mutated("in.nawa",
                                           [&](auto& b) { std::memset(b.data() + in_at, 0, 4); })),
                 ModelFormatError);
    // Truncated inside the weights.
    EXPECT_THROW(
        Model::load(write_mutated("cut.nawa", [&](auto& b) { b.resize(weights_at + 10); })),
        ModelFormatError);
}

TEST(QuantizedModel, FullTestSetAccuracy) {
    const std::string dir = std::string(NAWA_SOURCE_DIR) + "/data/MNIST/raw";
    if (!fs::exists(dir + "/t10k-images-idx3-ubyte")) {
        GTEST_SKIP() << "MNIST not downloaded (run python/train.py); skipping full evaluation";
    }
    const Tensor images = load_mnist_images(dir + "/t10k-images-idx3-ubyte");
    const auto labels = load_mnist_labels(dir + "/t10k-labels-idx1-ubyte");
    const Model fp32(kModel);
    const Model int8(kModelInt8);
    Workspace ws32, ws8;
    std::size_t correct32 = 0, correct8 = 0, changed = 0;
    float max_diff = 0.0f;
    const std::size_t n = images.size(0);
    for (std::size_t start = 0; start < n; start += 1000) {
        const std::size_t rows = std::min<std::size_t>(1000, n - start);
        const Tensor batch({rows, 784}, std::vector<float>(images.data() + start * 784,
                                                           images.data() + (start + rows) * 784));
        const Tensor& p32 = fp32.predict(batch, ws32);
        const Tensor& p8 = int8.predict(batch, ws8);
        for (std::size_t r = 0; r < rows; ++r) {
            std::size_t a32 = 0, a8 = 0;
            for (std::size_t j = 0; j < 10; ++j) {
                max_diff = std::max(max_diff, std::fabs(p32.at({r, j}) - p8.at({r, j})));
                if (p32.at({r, j}) > p32.at({r, a32})) a32 = j;
                if (p8.at({r, j}) > p8.at({r, a8})) a8 = j;
            }
            correct32 += a32 == labels[start + r];
            correct8 += a8 == labels[start + r];
            changed += a32 != a8;
        }
    }
    std::cout << "[          ] fp32 " << correct32 << "/" << n << ", int8 " << correct8 << "/" << n
              << ", " << changed << " predictions changed, max probability difference " << max_diff
              << "\n";
    EXPECT_GE(correct8, 9700u);
    EXPECT_LE(changed, 20u);
}
