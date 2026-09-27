# inference

A lightweight neural network inference engine written from scratch in C++20.

This is a learning project. The goal is to understand how a trained network is executed
at the systems level: tensors, layers, model loading, and the computational graph, and
then CPU optimization with threads, SIMD, and INT8 quantization. No external math libraries
are used. Python (PyTorch) is used only to train models and export their weights.

## Prerequisites

- CMake >= 3.20
- A C++20 compiler (GCC >= 11, Clang >= 14, or MSVC 2022)
- Git and network access on the first configure (GoogleTest is downloaded automatically)
- Optional: Ninja (`-G Ninja`) for faster builds

## Build and test

```bash
# Configure (Release by default)
cmake -S . -B build

# Build
cmake --build build -j

# Run the tests
ctest --test-dir build --output-on-failure
```

Debug build with AddressSanitizer + UndefinedBehaviorSanitizer:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
cmake --build build-debug -j
ctest --test-dir build-debug --output-on-failure
```

## CMake options

| Option              | Default | Description                                   |
|---------------------|---------|-----------------------------------------------|
| `ENABLE_TESTS`      | ON      | Build the GoogleTest unit tests               |
| `ENABLE_BENCHMARKS` | OFF     | Build the benchmarks                          |
| `ENABLE_SANITIZERS` | OFF     | ASan + UBSan in Debug builds (GCC/Clang only) |

## Layout

```
include/inference/   public headers (tensor, layers, model, runtime, operators, optimization)
src/                 implementation files, mirroring include/
tests/               GoogleTest unit tests
examples/            example programs
benchmarks/          performance benchmarks
models/              exported model weights (not committed)
python/              training and export scripts
```
