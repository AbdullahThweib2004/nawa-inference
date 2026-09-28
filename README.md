# Nawa

[![CI](https://github.com/AbdullahThweib2004/nawa-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdullahThweib2004/nawa-inference/actions/workflows/ci.yml)

A lightweight neural network inference engine built from scratch in C++.

## Why

Nawa is a learning project. The goal is to understand what actually happens when a trained
neural network runs by rebuilding the core of an inference engine from the ground up:
tensors, layers, model loading, and a computational graph, followed by CPU optimization
(multithreading, SIMD, INT8 quantization).

All math is written by hand in C++20, with no Eigen or BLAS. Python (PyTorch) is used only
to train models and export their weights.

## Status

Early development. Done so far:

- the tensor data structure
- math operations: matmul, transpose, element-wise ops with broadcasting, reductions
- inference layers: Linear, ReLU, Sigmoid, Softmax
- a trained MNIST model exported to the [Nawa model format](docs/model_format.md)
- a C++ model loader and runtime that matches PyTorch's outputs (max error < 1e-6)

End-to-end MNIST inference tooling is next.

- [x] 1. Project scaffolding
- [x] 2. Tensor core
- [x] 3. Tensor operations
- [x] 4. Layers
- [x] 5. Python training and weight export (MNIST)
- [x] 6. Model loading and runtime
- [ ] 7. End-to-end MNIST inference
- [ ] 8. Benchmarking
- [ ] 9. Optimization (threads, SIMD, INT8)
- [ ] 10. Extensions

## Prerequisites

- CMake >= 3.20
- A C++20 compiler (GCC >= 11 or Clang >= 14)
- Git (GoogleTest is downloaded automatically on the first configure)
- Optional: Ninja for faster builds

On Arch Linux:

```bash
sudo pacman -S --needed base-devel cmake git ninja
```

## Build and test

```bash
cmake -S . -B build                          # configure (Release by default)
cmake --build build -j                       # build
ctest --test-dir build --output-on-failure   # run the tests
```

Run the examples (all example programs are built into `build/bin/`):

```bash
./build/bin/tensor_basics   # creating, indexing, reshaping and printing tensors
./build/bin/ops_basics      # a manual Linear layer: y = x · W + b
./build/bin/mlp_forward     # a tiny 2-layer network built from Layer objects
./build/bin/predict_digits  # load models/mnist_mlp.nawa and classify 10 test digits
```

Debug build with AddressSanitizer + UndefinedBehaviorSanitizer:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
cmake --build build-debug -j
ctest --test-dir build-debug --output-on-failure
```

## Train and export (Python)

Python is used only to train the MNIST model and export it for the C++ engine. The
exported model (`models/mnist_mlp.nawa`) and test fixtures (`tests/fixtures/`) are
committed, so you only need these steps to retrain.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt

python python/train.py           # downloads MNIST to data/, trains, saves models/mnist_mlp.pt
python python/export.py          # writes models/mnist_mlp.nawa and tests/fixtures/*.ntsr
python python/verify_export.py   # numpy-only forward pass, checked against PyTorch's outputs
python -m pytest python/tests    # format round-trip and validation tests
```

The file format is specified in [docs/model_format.md](docs/model_format.md).

### CMake options

| Option              | Default | Description                                   |
|---------------------|---------|-----------------------------------------------|
| `ENABLE_TESTS`      | ON      | Build the GoogleTest unit tests               |
| `ENABLE_BENCHMARKS` | OFF     | Build the benchmarks                          |
| `BUILD_EXAMPLES`    | ON      | Build the example programs in `examples/`     |
| `ENABLE_SANITIZERS` | OFF     | ASan + UBSan in Debug builds (GCC/Clang only) |

## Project layout

```
include/inference/   public headers (tensor, layers, model, runtime, operators, optimization)
src/                 implementation files, mirroring include/
tests/               GoogleTest unit tests
examples/            example programs
benchmarks/          performance benchmarks
models/              exported model weights (not committed)
python/              training and export scripts
```

## License

Released under the [MIT License](LICENSE).
