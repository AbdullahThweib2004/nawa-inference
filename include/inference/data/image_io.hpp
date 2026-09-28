#pragma once

#include <string>

#include "inference/data/digit_preprocess.hpp"

// Image file loading (PNG, JPEG, BMP, GIF, ...), built on stb_image. Lives in the separate
// `nawa_image` library so inference_core stays free of third-party code.
namespace inference {

// Loads any image stb_image understands as grayscale 0..255.
// Color is converted to luminance by stb_image. Transparent pixels are composited over a
// WHITE background: drawing apps often save black strokes on a transparent canvas, which
// would otherwise turn into black-on-black. Throws std::runtime_error if loading fails.
GrayImage load_grayscale_image(const std::string& path);

}  // namespace inference
