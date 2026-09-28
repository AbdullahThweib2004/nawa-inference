#include "inference/data/image_io.hpp"

#include <stb_image.h>

#include <memory>
#include <stdexcept>

namespace inference {

GrayImage load_grayscale_image(const std::string& path) {
    int width = 0;
    int height = 0;
    int channels_in_file = 0;
    // Ask for 2 channels, gray + alpha, whatever the file contains: stb_image converts RGB
    // to luminance and adds alpha = 255 for images without transparency.
    constexpr int kGrayAlpha = 2;
    // unique_ptr with stbi_image_free as the deleter frees the buffer on every path.
    const std::unique_ptr<unsigned char, void (*)(void*)> data(
        stbi_load(path.c_str(), &width, &height, &channels_in_file, kGrayAlpha), stbi_image_free);
    if (!data) {
        throw std::runtime_error("cannot load image '" + path + "': " + stbi_failure_reason());
    }

    GrayImage image;
    image.width = static_cast<std::size_t>(width);
    image.height = static_cast<std::size_t>(height);
    image.pixels.resize(image.width * image.height);
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        const float gray = data.get()[2 * i];
        const float alpha = data.get()[2 * i + 1] / 255.0f;
        // Composite over white: fully transparent pixels become 255 (white paper).
        image.pixels[i] = gray * alpha + 255.0f * (1.0f - alpha);
    }
    return image;
}

}  // namespace inference
