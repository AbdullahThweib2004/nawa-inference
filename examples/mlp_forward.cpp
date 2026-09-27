// A tiny 2-layer network with hand-chosen weights:
//   input {1,4} -> Linear(4 -> 3) -> ReLU -> Linear(3 -> 2) -> Softmax
// Run: ./build/bin/mlp_forward

#include <iostream>
#include <memory>
#include <vector>

#include "inference/layers/layers.hpp"
#include "inference/tensor/ops.hpp"

using namespace inference;

int main() {
    std::vector<std::unique_ptr<Layer>> layers;

    // Weights use the {in_features, out_features} layout (see linear.hpp).
    layers.push_back(std::make_unique<Linear>(Tensor({4, 3}, {0.5f, -0.3f, 0.2f,  //
                                                              -0.4f, 0.8f, 0.1f,  //
                                                              0.3f, 0.2f, -0.6f,  //
                                                              0.1f, -0.5f, 0.4f}),
                                              Tensor({3}, {0.1f, 0.0f, -0.5f})));
    layers.push_back(std::make_unique<ReLU>());
    layers.push_back(std::make_unique<Linear>(Tensor({3, 2}, {1.0f, -1.0f,  //
                                                              0.5f, 0.5f,   //
                                                              -1.0f, 2.0f}),
                                              Tensor({2}, {0.0f, 0.1f})));
    layers.push_back(std::make_unique<Softmax>());

    Tensor x({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    std::cout << "input:\n" << x << "\n\n";

    for (const auto& layer : layers) {
        x = layer->forward(x);
        std::cout << layer->name() << ":\n" << x << "\n\n";
    }

    std::cout << "probabilities:   " << x << '\n';
    std::cout << "predicted class: " << argmax(x, -1).at({0}) << '\n';

    std::size_t total = 0;
    for (const auto& layer : layers) total += layer->num_parameters();
    std::cout << "parameters:      " << total << '\n';
    return 0;
}
