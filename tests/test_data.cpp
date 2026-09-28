#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "inference/data/digit_preprocess.hpp"
#include "inference/data/idx.hpp"
#include "inference/data/image_io.hpp"
#include "inference/model/binary_io.hpp"
#include "inference/model/model.hpp"
#include "inference/tensor/ops.hpp"
#include "test_paths.hpp"

using namespace inference;
namespace fs = std::filesystem;

#ifndef NAWA_SOURCE_DIR
#error "NAWA_SOURCE_DIR must be defined by the build"
#endif

namespace {

const std::string kRoot = NAWA_SOURCE_DIR;
const std::string kModelPath = kRoot + "/models/mnist_mlp.nawa";

fs::path temp_file(const std::string& name) { return test_temp_dir(false) / name; }

// Writes an IDX file byte by byte: magic [0, 0, type, ndim], BIG-endian u32 dims, data.
std::string write_idx(const std::string& name, std::vector<std::uint8_t> magic,
                      const std::vector<std::uint32_t>& dims,
                      const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> bytes = std::move(magic);
    for (std::uint32_t d : dims) {
        // Most significant byte first.
        bytes.push_back(static_cast<std::uint8_t>(d >> 24));
        bytes.push_back(static_cast<std::uint8_t>(d >> 16));
        bytes.push_back(static_cast<std::uint8_t>(d >> 8));
        bytes.push_back(static_cast<std::uint8_t>(d));
    }
    bytes.insert(bytes.end(), data.begin(), data.end());
    const fs::path path = temp_file(name);
    std::ofstream(path, std::ios::binary)
        .write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    return path.string();
}

// A w x h image filled with `background`, with the rectangle [x0, x1) x [y0, y1) set to `fg`.
GrayImage rect_image(std::size_t w, std::size_t h, float background, std::size_t x0, std::size_t y0,
                     std::size_t x1, std::size_t y1, float fg) {
    GrayImage img{w, h, std::vector<float>(w * h, background)};
    for (std::size_t y = y0; y < y1; ++y)
        for (std::size_t x = x0; x < x1; ++x) img.pixels[y * w + x] = fg;
    return img;
}

std::size_t predict_label(const Model& model, const GrayImage& image28) {
    return static_cast<std::size_t>(argmax(model.predict(to_model_input(image28)), -1).at({0}));
}

}  // namespace

// ---------------------------------------------------------------------------
// IDX reader
// ---------------------------------------------------------------------------

TEST(Idx, DimsAreBigEndian) {
    // 258 = 0x00000102 is stored as bytes 00 00 01 02. Read little-endian by mistake, it
    // would come out as 0x02010000 = 33,619,968.
    std::vector<std::uint8_t> data(258 * 2);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<std::uint8_t>(i % 251);
    const IdxArray idx = read_idx(write_idx("a.idx", {0, 0, 0x08, 2}, {258, 2}, data));
    EXPECT_EQ(idx.dims, (std::vector<std::size_t>{258, 2}));
    EXPECT_EQ(idx.data, data);
}

TEST(Idx, MnistImagesAndLabels) {
    std::vector<std::uint8_t> pixels(2 * 784);
    for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<std::uint8_t>(i % 256);
    const Tensor images =
        load_mnist_images(write_idx("img.idx", {0, 0, 0x08, 3}, {2, 28, 28}, pixels));
    EXPECT_EQ(images.shape(), (Shape{2, 784}));
    EXPECT_EQ(images.at({0, 0}), 0.0f);
    EXPECT_EQ(images.at({0, 255}), 255.0f);  // raw 0..255, not normalized
    EXPECT_EQ(images.at({1, 0}), static_cast<float>(784 % 256));

    const auto labels = load_mnist_labels(write_idx("lab.idx", {0, 0, 0x08, 1}, {3}, {7, 2, 1}));
    EXPECT_EQ(labels, (std::vector<std::uint8_t>{7, 2, 1}));
}

TEST(Idx, MalformedFilesThrow) {
    EXPECT_THROW(read_idx(write_idx("m1", {1, 0, 0x08, 1}, {2}, {0, 0})),
                 ModelFormatError);  // magic
    EXPECT_THROW(read_idx(write_idx("m2", {0, 0, 0x0D, 1}, {2}, {0, 0})),
                 ModelFormatError);  // float type
    EXPECT_THROW(read_idx(write_idx("m3", {0, 0, 0x08, 1}, {3}, {0, 0})),
                 ModelFormatError);  // truncated
    EXPECT_THROW(read_idx(write_idx("m4", {0, 0, 0x08, 1}, {1}, {0, 0})),
                 ModelFormatError);  // trailing
    EXPECT_THROW(read_idx(write_idx("m5", {0, 0, 0x08, 2}, {5}, {})),
                 ModelFormatError);                                             // missing dim
    EXPECT_THROW(read_idx(write_idx("m6", {0, 0}, {}, {})), ModelFormatError);  // cut inside magic
    // Valid IDX, wrong shape for MNIST images / labels.
    EXPECT_THROW(load_mnist_images(write_idx("m7", {0, 0, 0x08, 3}, {1, 27, 28},
                                             std::vector<std::uint8_t>(27 * 28))),
                 ModelFormatError);
    EXPECT_THROW(load_mnist_labels(write_idx("m8", {0, 0, 0x08, 1}, {2}, {3, 10})),
                 ModelFormatError);  // label 10
}

// ---------------------------------------------------------------------------
// Preprocessing steps
// ---------------------------------------------------------------------------

TEST(Preprocess, InversionDetection) {
    const GrayImage dark_on_light = rect_image(10, 10, 255.0f, 3, 3, 7, 7, 0.0f);
    const GrayImage light_on_dark = rect_image(10, 10, 0.0f, 3, 3, 7, 7, 255.0f);
    EXPECT_TRUE(needs_inversion(dark_on_light));
    EXPECT_FALSE(needs_inversion(light_on_dark));
    const GrayImage inverted = invert(dark_on_light);
    EXPECT_EQ(inverted.pixels, light_on_dark.pixels);
    EXPECT_FALSE(needs_inversion(inverted));
}

TEST(Preprocess, BoundingBoxAndCrop) {
    GrayImage img = rect_image(10, 8, 0.0f, 0, 0, 0, 0, 0.0f);
    img.pixels[3 * 10 + 2] = 200.0f;  // (2, 3)
    img.pixels[5 * 10 + 6] = 200.0f;  // (6, 5)
    img.pixels[0 * 10 + 9] = 20.0f;   // faint noise below the threshold: ignored
    const auto box = bounding_box(img, 50.0f);
    ASSERT_TRUE(box.has_value());
    EXPECT_EQ(box->x0, 2u);
    EXPECT_EQ(box->y0, 3u);
    EXPECT_EQ(box->x1, 7u);
    EXPECT_EQ(box->y1, 6u);

    const GrayImage cropped = crop(img, *box);
    EXPECT_EQ(cropped.width, 5u);
    EXPECT_EQ(cropped.height, 3u);
    EXPECT_EQ(cropped.at(0, 0), 200.0f);
    EXPECT_EQ(cropped.at(4, 2), 200.0f);

    EXPECT_FALSE(bounding_box(rect_image(4, 4, 10.0f, 0, 0, 0, 0, 0.0f), 50.0f).has_value());
}

TEST(Preprocess, ResizeOutputSizes) {
    const GrayImage wide(rect_image(40, 30, 255.0f, 0, 0, 0, 0, 0.0f));
    const GrayImage fitted = fit_longer_side(wide, 20);
    EXPECT_EQ(fitted.width, 20u);
    EXPECT_EQ(fitted.height, 15u);  // aspect ratio kept

    const GrayImage tall_small(rect_image(5, 10, 255.0f, 0, 0, 0, 0, 0.0f));
    const GrayImage enlarged = fit_longer_side(tall_small, 20);
    EXPECT_EQ(enlarged.width, 10u);
    EXPECT_EQ(enlarged.height, 20u);

    // A 1-pixel-wide line stays at least 1 pixel wide.
    EXPECT_EQ(fit_longer_side(rect_image(1, 100, 255.0f, 0, 0, 0, 0, 0.0f), 20).width, 1u);
}

TEST(Preprocess, ResizeAreaAverages) {
    // Uniform stays uniform, at any scale.
    for (const GrayImage& out : {resize_area(rect_image(7, 5, 100.0f, 0, 0, 0, 0, 0.0f), 3, 2),
                                 resize_area(rect_image(3, 2, 100.0f, 0, 0, 0, 0, 0.0f), 7, 5)}) {
        for (float v : out.pixels) EXPECT_NEAR(v, 100.0f, 1e-3f);
    }
    // 4x4 -> 2x2: each output pixel is the mean of a 2x2 block.
    const GrayImage src = rect_image(4, 4, 0.0f, 0, 0, 1, 1, 255.0f);  // only (0,0) is lit
    const GrayImage half = resize_area(src, 2, 2);
    EXPECT_NEAR(half.at(0, 0), 255.0f / 4.0f, 1e-3f);
    EXPECT_NEAR(half.at(1, 1), 0.0f, 1e-3f);
    // 3 -> 2 columns: output pixel 0 covers input [0, 1.5): all of pixel 0 and half of pixel 1.
    const GrayImage row{3, 1, {90.0f, 30.0f, 0.0f}};
    const GrayImage two = resize_area(row, 2, 1);
    EXPECT_NEAR(two.at(0, 0), (90.0f * 1.0f + 30.0f * 0.5f) / 1.5f, 1e-3f);
    EXPECT_NEAR(two.at(1, 0), (30.0f * 0.5f + 0.0f * 1.0f) / 1.5f, 1e-3f);
}

TEST(Preprocess, CenterOfMass) {
    const GrayImage dot = rect_image(3, 3, 0.0f, 0, 0, 1, 1, 255.0f);
    EXPECT_FLOAT_EQ(center_of_mass(dot).x, 0.5f);  // center of pixel (0, 0)
    EXPECT_FLOAT_EQ(center_of_mass(dot).y, 0.5f);
    const Point c = center_of_mass(rect_image(28, 28, 1.0f, 0, 0, 0, 0, 0.0f));
    EXPECT_FLOAT_EQ(c.x, 14.0f);  // uniform image: the geometric center
    EXPECT_FLOAT_EQ(c.y, 14.0f);
}

TEST(Preprocess, PasteCenteredPutsMassInTheMiddle) {
    // A lopsided 10x20 shape: its mass is in the right half.
    const GrayImage shape = rect_image(10, 20, 0.0f, 6, 2, 10, 18, 255.0f);
    const GrayImage canvas = paste_centered(shape, 28);
    EXPECT_EQ(canvas.width, 28u);
    const Point c = center_of_mass(canvas);
    EXPECT_NEAR(c.x, 14.0f, 0.5f);
    EXPECT_NEAR(c.y, 14.0f, 0.5f);
}

TEST(Preprocess, FullPipelineOnSmallOffCenterDarkDigit) {
    // A dark, tall "stroke" in the corner of a large light image.
    const GrayImage img = rect_image(120, 90, 250.0f, 5, 8, 11, 38, 10.0f);
    const PreprocessResult r = mnist_preprocess(img);
    EXPECT_TRUE(r.inverted);
    ASSERT_TRUE(r.box.has_value());
    EXPECT_EQ(r.box->width(), 6u);
    EXPECT_EQ(r.box->height(), 30u);
    EXPECT_EQ(r.image.width, 28u);
    EXPECT_EQ(r.image.height, 28u);
    const auto content = bounding_box(r.image, 50.0f);
    ASSERT_TRUE(content.has_value());
    EXPECT_EQ(content->height(), 20u);  // longer side scaled to 20
    const Point c = center_of_mass(r.image);
    EXPECT_NEAR(c.x, 14.0f, 0.5f);
    EXPECT_NEAR(c.y, 14.0f, 0.5f);
}

TEST(Preprocess, EmptyImageGivesBlankCanvas) {
    const PreprocessResult r = mnist_preprocess(rect_image(50, 50, 0.0f, 0, 0, 0, 0, 0.0f));
    EXPECT_FALSE(r.box.has_value());
    EXPECT_EQ(r.image.pixels, std::vector<float>(28 * 28, 0.0f));
}

// ---------------------------------------------------------------------------
// Example images and full evaluation
// ---------------------------------------------------------------------------

TEST(ExampleImages, ClassifiedCorrectlyWithPreprocessing) {
    const Model model(kModelPath);
    const std::vector<std::pair<std::string, std::size_t>> cases = {
        {"digit7_mnist.png", 7}, {"digit2_inverted.png", 2}, {"digit4_offcenter.png", 4}};
    for (const auto& [file, label] : cases) {
        const GrayImage image = load_grayscale_image(kRoot + "/examples/images/" + file);
        EXPECT_EQ(predict_label(model, mnist_preprocess(image).image), label) << file;
        std::cout << "[          ] " << file << ": with preprocessing -> "
                  << predict_label(model, mnist_preprocess(image).image) << ", resize only -> "
                  << predict_label(model, resize_only(image)) << " (true " << label << ")\n";
    }
}

TEST(ExampleImages, LoaderHandlesMissingFile) {
    EXPECT_THROW(load_grayscale_image(kRoot + "/examples/images/no_such.png"), std::runtime_error);
}

TEST(MnistTestSet, FullEvaluationAccuracy) {
    const std::string dir = kRoot + "/data/MNIST/raw";
    if (!fs::exists(dir + "/t10k-images-idx3-ubyte")) {
        GTEST_SKIP() << "MNIST not downloaded (run python/train.py); skipping full evaluation";
    }
    const Model model(kModelPath);
    const Tensor images = load_mnist_images(dir + "/t10k-images-idx3-ubyte");
    const auto labels = load_mnist_labels(dir + "/t10k-labels-idx1-ubyte");
    ASSERT_EQ(images.size(0), labels.size());

    std::size_t correct = 0;
    const std::size_t n = images.size(0);
    constexpr std::size_t kBatch = 1000;
    for (std::size_t start = 0; start < n; start += kBatch) {
        const std::size_t rows = std::min(kBatch, n - start);
        const float* first = images.data() + start * 784;
        const auto predictions =
            model.classify(Tensor({rows, 784}, std::vector<float>(first, first + rows * 784)));
        for (std::size_t r = 0; r < rows; ++r) correct += predictions[r].label == labels[start + r];
    }
    const double accuracy = static_cast<double>(correct) / static_cast<double>(n);
    std::cout << "[          ] accuracy " << correct << "/" << n << "\n";
    EXPECT_GE(accuracy, 0.97);
}
