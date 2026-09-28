#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "inference/tensor/tensor.hpp"

// Turning an arbitrary picture of a digit into an MNIST-style 28x28 input.
//
// MNIST digits are white on black, cropped to their bounding box, scaled so the longer side
// is 20 px, and placed in a 28x28 canvas so their center of mass is in the middle. The
// model only ever saw digits like that, so a user's image must be converted the same way.
namespace inference {

// A grayscale image, row-major, values 0..255 (0 = black).
struct GrayImage {
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<float> pixels;  // width * height values; pixel (x, y) is pixels[y * width + x]

    float at(std::size_t x, std::size_t y) const { return pixels[y * width + x]; }
};

// Rectangle [x0, x1) x [y0, y1) in pixel coordinates (end-exclusive).
struct BoundingBox {
    std::size_t x0, y0, x1, y1;
    std::size_t width() const { return x1 - x0; }
    std::size_t height() const { return y1 - y0; }
};

struct PreprocessOptions {
    // Pixels brighter than this (after inversion) count as part of the digit when cropping.
    // Well above 0, so light noise or JPEG artifacts in the background don't enlarge the box.
    float threshold = 50.0f;
    std::size_t box_size = 20;     // longer side of the digit after scaling (MNIST: 20)
    std::size_t canvas_size = 28;  // output side (MNIST: 28)
};

// What mnist_preprocess did, for reporting.
struct PreprocessResult {
    GrayImage image;                 // canvas_size x canvas_size
    bool inverted = false;           // step a) was applied
    std::optional<BoundingBox> box;  // step b) crop, or empty if no pixel was above threshold
};

// a) True if the average brightness of the outermost ring of pixels is above mid-gray,
//    i.e. the background is light (a dark digit on a light background).
bool needs_inversion(const GrayImage& image);
GrayImage invert(const GrayImage& image);  // 255 - value

// b) Smallest box containing every pixel > threshold, or nullopt if there is none.
std::optional<BoundingBox> bounding_box(const GrayImage& image, float threshold);
GrayImage crop(const GrayImage& image, const BoundingBox& box);

// c) Resizes by AREA AVERAGING: each output pixel covers a rectangle of the input and gets
//    the average of the input over that rectangle (partially covered pixels count by the
//    fraction covered). Works for shrinking and enlarging.
GrayImage resize_area(const GrayImage& image, std::size_t width, std::size_t height);

//    Scales so the longer side becomes `size`, keeping the aspect ratio (shorter side >= 1).
GrayImage fit_longer_side(const GrayImage& image, std::size_t size);

// d) Intensity-weighted mean position, in continuous coordinates where pixel (x, y) covers
//    [x, x+1) x [y, y+1), so a 28x28 image's geometric center is (14, 14).
//    Returns the image center for an all-black image.
struct Point {
    float x, y;
};
Point center_of_mass(const GrayImage& image);

//    Pastes `image` into a black canvas_size x canvas_size canvas so its center of mass is
//    as close to the canvas center as whole-pixel shifts allow (the image stays inside).
GrayImage paste_centered(const GrayImage& image, std::size_t canvas_size);

// The full pipeline: a) invert if needed, b) crop, c) fit to box_size, d) center.
PreprocessResult mnist_preprocess(const GrayImage& image, const PreprocessOptions& options = {});

// No preprocessing: just resize the whole image to size x size (for comparison).
GrayImage resize_only(const GrayImage& image, std::size_t size = 28);

// Flattens a square image into a {width * height} tensor of raw 0..255 values, the input
// Model::predict expects.
Tensor to_model_input(const GrayImage& image);

}  // namespace inference
