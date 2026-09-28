// Loads the trained MNIST model and classifies the first 10 test images.
// Run: ./build/bin/predict_digits [model.nawa] [images.ntsr] [labels.ntsr]

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "inference/model/model.hpp"
#include "inference/model/tensor_io.hpp"

using namespace inference;

int main(int argc, char** argv) {
    // Defaults point into the source tree (set by examples/CMakeLists.txt), so the example
    // works from any directory.
    const std::string root = NAWA_SOURCE_DIR;
    const std::string model_path = argc > 1 ? argv[1] : root + "/models/mnist_mlp.nawa";
    const std::string images_path =
        argc > 2 ? argv[2] : root + "/tests/fixtures/mnist_test100_images.ntsr";
    const std::string labels_path =
        argc > 3 ? argv[3] : root + "/tests/fixtures/mnist_test100_labels.ntsr";

    try {
        const Model model(model_path);
        std::cout << model.summary() << "\n\n";

        // Raw pixels 0..255: the model normalizes them itself (see Model::preprocess).
        const Tensor images = read_tensor_file(images_path);
        const Tensor labels = read_tensor_file(labels_path);
        constexpr std::size_t kCount = 10;
        const Tensor first({kCount, 784},
                           std::vector<float>(images.data(), images.data() + kCount * 784));

        std::size_t correct = 0;
        std::cout << "image  predicted  confidence  true label\n";
        const auto predictions = model.classify(first);
        for (std::size_t i = 0; i < predictions.size(); ++i) {
            const auto truth = static_cast<std::size_t>(labels.at({i}));
            const bool ok = predictions[i].label == truth;
            correct += ok ? 1 : 0;
            std::cout << std::setw(5) << i << std::setw(11) << predictions[i].label << std::setw(11)
                      << std::fixed << std::setprecision(2) << predictions[i].confidence * 100.0f
                      << "%" << std::setw(12) << truth << (ok ? "" : "   <-- wrong") << '\n';
        }
        std::cout << correct << "/" << kCount << " correct\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
