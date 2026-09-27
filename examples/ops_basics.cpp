// A "manual Linear layer": y = x · W + b, printing every intermediate tensor.
// Run: ./build/bin/ops_basics

#include <iostream>

#include "inference/tensor/ops.hpp"

using inference::Tensor;

int main() {
    // One input sample with 4 features (a batch of size 1).
    const Tensor x({1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});

    // Weights: 4 inputs -> 3 outputs. Column j holds the weights of output neuron j.
    const Tensor W({4, 3}, {0.1f, 0.2f, 0.3f,     //
                            0.4f, 0.5f, 0.6f,     //
                            -0.1f, -0.2f, -0.3f,  //
                            0.0f, 1.0f, -1.0f});

    // One bias per output neuron.
    const Tensor b({3}, {0.5f, -0.5f, 1.0f});

    std::cout << "x {1,4}:\n" << x << "\n\n";
    std::cout << "W {4,3}:\n" << W << "\n\n";
    std::cout << "b {3}:\n" << b << "\n\n";

    // {1,4} · {4,3} -> {1,3}
    const Tensor xW = x.matmul(W);
    std::cout << "x · W {1,3}:\n" << xW << "\n\n";

    // {1,3} + {3}: b is broadcast across the batch dimension.
    const Tensor y = xW + b;
    std::cout << "y = x · W + b {1,3}:\n" << y << "\n\n";

    // Which output is largest? (For a classifier this would be the predicted class.)
    std::cout << "argmax(y, -1):\n" << inference::argmax(y, -1) << '\n';
    return 0;
}
