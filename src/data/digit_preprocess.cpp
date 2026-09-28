#include "inference/data/digit_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace inference {

namespace {

void require_non_empty(const GrayImage& image, const char* fn) {
    if (image.width == 0 || image.height == 0 ||
        image.pixels.size() != image.width * image.height) {
        throw std::invalid_argument(std::string(fn) + ": invalid image " +
                                    std::to_string(image.width) + "x" +
                                    std::to_string(image.height) + " with " +
                                    std::to_string(image.pixels.size()) + " pixels");
    }
}

GrayImage blank(std::size_t width, std::size_t height) {
    return {width, height, std::vector<float>(width * height, 0.0f)};
}

}  // namespace

// ---------------------------------------------------------------------------
// a) Inversion
// ---------------------------------------------------------------------------

bool needs_inversion(const GrayImage& image) {
    require_non_empty(image, "needs_inversion");
    // Sum every pixel on the outer ring once: top and bottom rows, then the left and right
    // columns without their corners.
    double sum = 0.0;
    std::size_t count = 0;
    const std::size_t w = image.width;
    const std::size_t h = image.height;
    for (std::size_t x = 0; x < w; ++x) {
        sum += image.at(x, 0);
        ++count;
        if (h > 1) {
            sum += image.at(x, h - 1);
            ++count;
        }
    }
    for (std::size_t y = 1; y + 1 < h; ++y) {
        sum += image.at(0, y);
        ++count;
        if (w > 1) {
            sum += image.at(w - 1, y);
            ++count;
        }
    }
    return sum / static_cast<double>(count) > 127.5;
}

GrayImage invert(const GrayImage& image) {
    GrayImage out = image;
    for (float& v : out.pixels) v = 255.0f - v;
    return out;
}

// ---------------------------------------------------------------------------
// b) Bounding box and crop
// ---------------------------------------------------------------------------

std::optional<BoundingBox> bounding_box(const GrayImage& image, float threshold) {
    require_non_empty(image, "bounding_box");
    std::size_t x0 = image.width, y0 = image.height, x1 = 0, y1 = 0;
    for (std::size_t y = 0; y < image.height; ++y) {
        for (std::size_t x = 0; x < image.width; ++x) {
            if (image.at(x, y) > threshold) {
                x0 = std::min(x0, x);
                y0 = std::min(y0, y);
                x1 = std::max(x1, x + 1);
                y1 = std::max(y1, y + 1);
            }
        }
    }
    if (x1 == 0) return std::nullopt;  // nothing above the threshold
    return BoundingBox{x0, y0, x1, y1};
}

GrayImage crop(const GrayImage& image, const BoundingBox& box) {
    if (box.x1 > image.width || box.y1 > image.height || box.x0 >= box.x1 || box.y0 >= box.y1) {
        throw std::invalid_argument("crop: box is empty or outside the image");
    }
    GrayImage out = blank(box.width(), box.height());
    for (std::size_t y = 0; y < out.height; ++y) {
        for (std::size_t x = 0; x < out.width; ++x) {
            out.pixels[y * out.width + x] = image.at(box.x0 + x, box.y0 + y);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// c) Resizing
// ---------------------------------------------------------------------------

GrayImage resize_area(const GrayImage& image, std::size_t width, std::size_t height) {
    require_non_empty(image, "resize_area");
    if (width == 0 || height == 0) throw std::invalid_argument("resize_area: zero size");

    // Output pixel (ox, oy) covers the input rectangle
    //   [ox * sx, (ox + 1) * sx) x [oy * sy, (oy + 1) * sy)
    // Each input pixel it touches contributes its value times the overlap area; dividing by
    // the rectangle's area (sx * sy) gives the average. When enlarging (sx < 1) a rectangle
    // lies within one or two input pixels, which blends neighbours smoothly at their edges.
    const double sx = static_cast<double>(image.width) / static_cast<double>(width);
    const double sy = static_cast<double>(image.height) / static_cast<double>(height);
    GrayImage out = blank(width, height);

    for (std::size_t oy = 0; oy < height; ++oy) {
        const double y_begin = static_cast<double>(oy) * sy;
        const double y_end = y_begin + sy;
        for (std::size_t ox = 0; ox < width; ++ox) {
            const double x_begin = static_cast<double>(ox) * sx;
            const double x_end = x_begin + sx;

            double total = 0.0;
            const auto iy_last = std::min(image.height, static_cast<std::size_t>(std::ceil(y_end)));
            const auto ix_last = std::min(image.width, static_cast<std::size_t>(std::ceil(x_end)));
            for (auto iy = static_cast<std::size_t>(y_begin); iy < iy_last; ++iy) {
                // Overlap of input row [iy, iy+1) with [y_begin, y_end).
                const double wy = std::min(y_end, static_cast<double>(iy + 1)) -
                                  std::max(y_begin, static_cast<double>(iy));
                if (wy <= 0.0) continue;
                for (auto ix = static_cast<std::size_t>(x_begin); ix < ix_last; ++ix) {
                    const double wx = std::min(x_end, static_cast<double>(ix + 1)) -
                                      std::max(x_begin, static_cast<double>(ix));
                    if (wx <= 0.0) continue;
                    total += wx * wy * image.at(ix, iy);
                }
            }
            out.pixels[oy * width + ox] = static_cast<float>(total / (sx * sy));
        }
    }
    return out;
}

GrayImage fit_longer_side(const GrayImage& image, std::size_t size) {
    require_non_empty(image, "fit_longer_side");
    const double scale =
        static_cast<double>(size) / static_cast<double>(std::max(image.width, image.height));
    const auto scaled = [&](std::size_t side) {
        return std::max<std::size_t>(
            1, static_cast<std::size_t>(std::lround(static_cast<double>(side) * scale)));
    };
    // The longer side is set exactly, so rounding can't make it 19 or 21.
    const std::size_t w = image.width >= image.height ? size : scaled(image.width);
    const std::size_t h = image.height > image.width ? size : scaled(image.height);
    return resize_area(image, w, h);
}

// ---------------------------------------------------------------------------
// d) Centering
// ---------------------------------------------------------------------------

Point center_of_mass(const GrayImage& image) {
    require_non_empty(image, "center_of_mass");
    double mass = 0.0, mx = 0.0, my = 0.0;
    for (std::size_t y = 0; y < image.height; ++y) {
        for (std::size_t x = 0; x < image.width; ++x) {
            const double v = image.at(x, y);
            mass += v;
            // Pixel (x, y) covers [x, x+1), so its own center is at x + 0.5.
            mx += v * (static_cast<double>(x) + 0.5);
            my += v * (static_cast<double>(y) + 0.5);
        }
    }
    if (mass <= 0.0) {
        return {static_cast<float>(image.width) / 2.0f, static_cast<float>(image.height) / 2.0f};
    }
    return {static_cast<float>(mx / mass), static_cast<float>(my / mass)};
}

GrayImage paste_centered(const GrayImage& image, std::size_t canvas_size) {
    require_non_empty(image, "paste_centered");
    if (image.width > canvas_size || image.height > canvas_size) {
        throw std::invalid_argument("paste_centered: image is larger than the canvas");
    }
    // Shift so the center of mass moves to the canvas center, rounded to whole pixels and
    // clamped so the image stays fully inside the canvas.
    const Point com = center_of_mass(image);
    const double center = static_cast<double>(canvas_size) / 2.0;
    const auto place = [&](double com_coord, std::size_t side) {
        const double ideal = std::round(center - com_coord);
        const double max_offset = static_cast<double>(canvas_size - side);
        return static_cast<std::size_t>(std::clamp(ideal, 0.0, max_offset));
    };
    const std::size_t ox = place(com.x, image.width);
    const std::size_t oy = place(com.y, image.height);

    GrayImage canvas = blank(canvas_size, canvas_size);
    for (std::size_t y = 0; y < image.height; ++y) {
        for (std::size_t x = 0; x < image.width; ++x) {
            canvas.pixels[(oy + y) * canvas_size + (ox + x)] = image.at(x, y);
        }
    }
    return canvas;
}

// ---------------------------------------------------------------------------
// Pipelines
// ---------------------------------------------------------------------------

PreprocessResult mnist_preprocess(const GrayImage& image, const PreprocessOptions& options) {
    require_non_empty(image, "mnist_preprocess");
    PreprocessResult result;

    // a) MNIST digits are white on black.
    result.inverted = needs_inversion(image);
    const GrayImage oriented = result.inverted ? invert(image) : image;

    // b) Crop away the empty margin, so position and size in the original don't matter.
    result.box = bounding_box(oriented, options.threshold);
    if (!result.box) {
        result.image = blank(options.canvas_size, options.canvas_size);  // nothing drawn
        return result;
    }
    const GrayImage cropped = crop(oriented, *result.box);

    // c) Normalize the size: longer side = 20 px, aspect ratio kept.
    const GrayImage fitted = fit_longer_side(cropped, options.box_size);

    // d) Center by mass in the 28x28 canvas.
    result.image = paste_centered(fitted, options.canvas_size);
    return result;
}

GrayImage resize_only(const GrayImage& image, std::size_t size) {
    return resize_area(image, size, size);
}

Tensor to_model_input(const GrayImage& image) {
    require_non_empty(image, "to_model_input");
    return Tensor(Shape{image.width * image.height}, image.pixels);
}

}  // namespace inference
