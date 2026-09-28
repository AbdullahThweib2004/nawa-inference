// Stage 9.4: aligned storage, reusable buffers, broadcast fast paths, the fused GEMM
// epilogue, and allocation-free Model::predict.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "alloc_hook.hpp"
#include "inference/model/model.hpp"
#include "inference/model/tensor_io.hpp"
#include "inference/runtime/workspace.hpp"
#include "inference/tensor/gemm.hpp"
#include "inference/tensor/ops.hpp"

using namespace inference;

namespace {

Tensor random_tensor(const Shape& shape, std::uint32_t seed) {
    std::vector<float> data(numel_of(shape));
    std::uint32_t state = seed * 2654435761u + 7u;
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

// Broadcasting computed the slow, obvious way: for every output index, map it to each input
// by dropping missing leading dims and using index 0 for size-1 dims.
Tensor reference_add(const Tensor& a, const Tensor& b) {
    const Shape out_shape = broadcast_shape(a.shape(), b.shape());
    Tensor out(out_shape);
    const Shape out_strides = compute_strides(out_shape);
    const auto input_offset = [&](const Tensor& t, std::size_t flat) {
        std::size_t offset = 0;
        const std::size_t missing = out_shape.size() - t.ndim();
        for (std::size_t d = 0; d < out_shape.size(); ++d) {
            const std::size_t idx = (flat / out_strides[d]) % out_shape[d];
            if (d >= missing && t.shape()[d - missing] != 1) {
                offset += idx * t.strides()[d - missing];
            }
        }
        return offset;
    };
    for (std::size_t i = 0; i < out.numel(); ++i) {
        out.data()[i] = a.data()[input_offset(a, i)] + b.data()[input_offset(b, i)];
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Aligned storage and reusable buffers
// ---------------------------------------------------------------------------

TEST(AlignedStorage, TensorDataIs64ByteAligned) {
    for (const Shape& shape : {Shape{1}, Shape{3}, Shape{7, 13}, Shape{256, 784}, Shape{}}) {
        const Tensor t(shape);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(t.data()) % 64, 0u) << shape_to_string(shape);
        const Tensor copy = t;  // copies are aligned too
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(copy.data()) % 64, 0u);
    }
}

TEST(AlignedStorage, ResizeReusesMemory) {
    Tensor t({256, 128});
    const float* original = t.data();
    t.resize(Shape{10, 10});  // smaller: same memory
    EXPECT_EQ(t.data(), original);
    EXPECT_EQ(t.shape(), (Shape{10, 10}));
    EXPECT_EQ(t.strides(), (Shape{10, 1}));
    EXPECT_EQ(t.numel(), 100u);
    t.resize(256, 128);  // back to the old size: still no new memory
    EXPECT_EQ(t.data(), original);
    t.resize(Shape{512, 128});  // larger: new memory, still aligned
    EXPECT_EQ(t.numel(), 512u * 128u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(t.data()) % 64, 0u);
    EXPECT_THROW(t.resize(Shape{3, 0}), std::invalid_argument);
}

TEST(AlignedStorage, AlignedBufferCopyAndMove) {
    AlignedBuffer a(5, 2.0f);
    AlignedBuffer b = a;
    b[0] = 9.0f;
    EXPECT_EQ(a[0], 2.0f);  // deep copy
    const float* p = b.data();
    AlignedBuffer c = std::move(b);
    EXPECT_EQ(c.data(), p);  // move steals the memory
    EXPECT_EQ(c.size(), 5u);
    a = c;
    EXPECT_EQ(a[0], 9.0f);
}

TEST(AlignedStorage, IntoOpsReuseTheOutput) {
    const Tensor a = random_tensor({64, 32}, 1);
    const Tensor b = random_tensor({32}, 2);
    Tensor out({64, 32});
    const float* p = out.data();
    add_into(a, b, out);
    EXPECT_EQ(out.data(), p);
    EXPECT_TRUE(bit_equal(out, a + b));
    Tensor mm({1, 1});
    matmul_into(a, random_tensor({32, 16}, 3), mm);
    EXPECT_EQ(mm.shape(), (Shape{64, 16}));
}

// ---------------------------------------------------------------------------
// Broadcasting: fast paths and the incremental general walk vs a direct reference
// ---------------------------------------------------------------------------

TEST(BroadcastPaths, MatchReferenceForManyShapePairs) {
    const std::vector<std::pair<Shape, Shape>> pairs = {
        {{256, 128}, {128}},       {{128}, {256, 128}},  // row, both orders
        {{4, 2, 3}, {2, 3}},       {{2, 3}, {4, 2, 3}},  // multi-dim row
        {{5, 7}, {5, 1}},          {{5, 1}, {5, 7}},     // column, both orders
        {{3, 4, 6}, {3, 4, 1}},    {{2, 3}, {}},         // 3-D column, scalar
        {{2, 1, 4}, {3, 1}},       {{4, 1}, {1, 3}},     // general: both stretched
        {{2, 3, 1, 5}, {3, 4, 1}}, {{1, 6}, {6, 1}}};
    std::uint32_t seed = 1;
    for (const auto& [sa, sb] : pairs) {
        const Tensor a = random_tensor(sa, seed++);
        const Tensor b = random_tensor(sb, seed++);
        EXPECT_TRUE(bit_equal(a + b, reference_add(a, b)))
            << shape_to_string(sa) << " + " << shape_to_string(sb);
    }
}

// ---------------------------------------------------------------------------
// GEMM epilogue
// ---------------------------------------------------------------------------

TEST(GemmEpilogue, BitIdenticalToSeparatePasses) {
    for (GemmKernel kernel : {GemmKernel::Portable, GemmKernel::Avx2}) {
        if (!kernel_available(kernel)) continue;
        for (const auto& [M, K, N] : std::vector<std::array<std::size_t, 3>>{
                 {1, 784, 128}, {1, 10, 21}, {7, 300, 17}, {256, 128, 10}, {13, 520, 530}}) {
            SCOPED_TRACE(::testing::Message()
                         << kernel_name(kernel) << " " << M << "x" << K << "x" << N);
            const Tensor a = random_tensor({M, K}, 11);
            const Tensor b = random_tensor({K, N}, 12);
            Tensor bias = random_tensor({N}, 13);
            // NaN and -0.0 must pass through ReLU exactly like the separate ReLU layer.
            bias.data()[0] = std::numeric_limits<float>::quiet_NaN();
            const PackedMatrix packed = PackedMatrix::pack(b.data(), K, N, kernel);

            Tensor plain({M, N});
            gemm(a.data(), M, packed, plain.data());
            Tensor expected = plain + bias;
            relu_inplace(expected.data(), expected.numel());

            Tensor fused({M, N});
            gemm(a.data(), M, packed, fused.data(), GemmEpilogue{bias.data(), true});
            EXPECT_TRUE(bit_equal(fused, expected));
        }
    }
}

// ---------------------------------------------------------------------------
// Fused execution plan and allocation-free predict
// ---------------------------------------------------------------------------

TEST(ModelPlan, FusesLinearAndRelu) {
    const Model model(std::string(NAWA_MODELS_DIR) + "/mnist_mlp.nawa");
    ASSERT_EQ(model.layers().size(), 4u);  // the loaded layers are unchanged
    ASSERT_EQ(model.plan().size(), 3u);    // Linear+ReLU, Linear, Softmax
    EXPECT_TRUE(model.plan()[0].fuse_relu);
    EXPECT_FALSE(model.plan()[1].fuse_relu);
    EXPECT_NE(model.summary().find("+ ReLU (fused)"), std::string::npos);
}

TEST(ModelPlan, FusedPredictIsBitIdenticalToLayerByLayer) {
    const Model model(std::string(NAWA_MODELS_DIR) + "/mnist_mlp.nawa");
    const Tensor images =
        read_tensor_file(std::string(NAWA_FIXTURES_DIR) + "/mnist_test100_images.ntsr");
    Tensor unfused = model.preprocess(images);
    for (const auto& layer : model.layers()) unfused = layer->forward(unfused);

    Workspace workspace;
    EXPECT_TRUE(bit_equal(model.predict(images, workspace), unfused));
    EXPECT_TRUE(bit_equal(model.predict(images), unfused));  // convenience overload
}

TEST(ModelPlan, SteadyStatePredictMakesNoHeapAllocations) {
    if (!NAWA_ALLOC_HOOK) GTEST_SKIP() << "allocation counting is disabled under sanitizers";
    const Model model(std::string(NAWA_MODELS_DIR) + "/mnist_mlp.nawa");
    const Tensor images =
        read_tensor_file(std::string(NAWA_FIXTURES_DIR) + "/mnist_test100_images.ntsr");
    const Tensor one({784}, std::vector<float>(images.data(), images.data() + 784));
    Workspace workspace;
    for (const Tensor* input : {&images, &one}) {
        (void)model.predict(*input, workspace);  // first call: buffers grow
        const std::size_t before = test_allocation_count();
        for (int i = 0; i < 3; ++i) (void)model.predict(*input, workspace);
        EXPECT_EQ(test_allocation_count() - before, 0u) << "batch " << input->numel() / 784;
    }
    // A smaller batch after a larger one reuses the same memory too.
    const std::size_t before = test_allocation_count();
    (void)model.predict(one, workspace);
    EXPECT_EQ(test_allocation_count() - before, 0u);
}
