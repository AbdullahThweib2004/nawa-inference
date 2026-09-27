# Nawa

A lightweight neural network inference engine built from scratch in C++.

## Why

Nawa is a learning project. The goal is to understand what actually happens when a trained
neural network runs by rebuilding the core of an inference engine from the ground up:
tensors, layers, model loading, and a computational graph, followed by CPU optimization
(multithreading, SIMD, INT8 quantization).

All math is written by hand in C++20, with no Eigen or BLAS. Python (PyTorch) is used only
to train models and export their weights.

## Status

Early development. Only the build and test scaffolding exists so far.

- [x] 1. Project scaffolding
- [ ] 2. Tensor core
- [ ] 3. Tensor operations
- [ ] 4. Layers
- [ ] 5. Python training and weight export (MNIST)
- [ ] 6. Model loading and runtime
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

Debug build with AddressSanitizer + UndefinedBehaviorSanitizer:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
cmake --build build-debug -j
ctest --test-dir build-debug --output-on-failure
```

### CMake options

| Option              | Default | Description                                   |
|---------------------|---------|-----------------------------------------------|
| `ENABLE_TESTS`      | ON      | Build the GoogleTest unit tests               |
| `ENABLE_BENCHMARKS` | OFF     | Build the benchmarks                          |
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
