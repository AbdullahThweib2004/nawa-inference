// Creates a few tensors and prints them. Run: ./build/bin/tensor_basics

#include <iostream>
#include <numeric>
#include <vector>

#include "inference/tensor/tensor.hpp"

using inference::Shape;
using inference::Tensor;

int main() {
    const Tensor s = Tensor::scalar(3.14f);
    std::cout << "Scalar:\n" << s << "\n\n";

    const Tensor m({2, 3}, {1, 2, 3, 4, 5, 6});
    std::cout << "2x3 matrix:\n" << m << '\n';
    std::cout << "strides = " << inference::shape_to_string(m.strides())
              << ", element {1, 2} = " << m.at({1, 2}) << "\n\n";

    std::cout << "Reshaped to {3, -1}:\n" << m.reshape({3, -1}) << "\n\n";

    std::vector<float> values(24);
    std::iota(values.begin(), values.end(), 0.0f);
    const Tensor cube({2, 3, 4}, values);
    std::cout << "2x3x4 tensor:\n" << cube << "\n\n";

    // A batch of three 28x28 images (MNIST size): 2352 elements, which is more than
    // the 1000-element threshold, so it is printed in summarized form.
    const Tensor images = Tensor::full({3, 28, 28}, 0.5f);
    std::cout << "3x28x28 tensor (summarized):\n" << images << '\n';
    return 0;
}
