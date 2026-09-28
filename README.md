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
- the `nawa` command-line tool: 97.15% on the MNIST test set, plus predictions on your own images
- a benchmarking baseline ([docs/performance.md](docs/performance.md)): the naive engine reaches
  1.6% of the CPU's peak FP32 throughput, and matmul takes 98% of the inference time

CPU optimization (cache-friendly matmul, SIMD, threads, INT8) is next.

- [x] 1. Project scaffolding
- [x] 2. Tensor core
- [x] 3. Tensor operations
- [x] 4. Layers
- [x] 5. Python training and weight export (MNIST)
- [x] 6. Model loading and runtime
- [x] 7. End-to-end MNIST inference
- [x] 8. Benchmarking
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
./build/bin/nawa --help       # the command-line tool (see below)
```

Debug build with AddressSanitizer + UndefinedBehaviorSanitizer:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
cmake --build build-debug -j
ctest --test-dir build-debug --output-on-failure
```

## Using the `nawa` CLI

After building, the tool is at `build/bin/nawa`:

```bash
# Model summary: layers, shapes, parameter count
./build/bin/nawa info models/mnist_mlp.nawa

# Accuracy, confusion matrix, per-digit accuracy and speed on the 10,000 MNIST test images
# (data/MNIST/raw is created by python/train.py)
./build/bin/nawa eval models/mnist_mlp.nawa --mnist data/MNIST/raw --batch 256

# Classify an image; --show draws the 28x28 model input as ASCII art
./build/bin/nawa predict models/mnist_mlp.nawa examples/images/digit2_inverted.png --show

# Same image without MNIST-style preprocessing, for comparison
./build/bin/nawa predict models/mnist_mlp.nawa examples/images/digit2_inverted.png --no-preprocess
```

`examples/images/` has three demo images made from MNIST test digits: `digit7_mnist.png`
(unchanged), `digit2_inverted.png` (dark on light) and `digit4_offcenter.png` (small, in a
corner). With preprocessing all three are classified correctly; with `--no-preprocess` only
the first one is.

### Draw your own digit

1. Draw a single digit in any paint program: a dark pen on a white background (or white on
   black), with **thick strokes**. MNIST digits are about 2-3 px thick at 28×28, so at
   280×280 use a brush around 20-25 px.
2. Fill most of the canvas with one digit and leave a margin. Nothing else should be in the
   picture.
3. Save it as PNG (JPEG and BMP work too; transparent backgrounds are treated as white).
4. Run `./build/bin/nawa predict models/mnist_mlp.nawa my_digit.png --show`.

The preprocessing inverts dark-on-light images, crops to the digit, scales it to 20 px and
centers it the way MNIST does. `--show` lets you check what the model actually sees. Photos
with shadows or uneven lighting may need cropping and more contrast first.

## Benchmarks

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_BENCHMARKS=ON
cmake --build build -j
./build/bin/nawa_bench                                   # all benchmarks
./build/bin/nawa_bench --benchmark_filter=BM_Predict     # a subset

# Compare two recorded result files (standard-library Python)
python3 python/compare_bench.py benchmarks/results/baseline.json new.json --metric cycles_per_iter
```

See [docs/performance.md](docs/performance.md) for the baseline numbers, how to record a
comparable run, and why CPU cycles are more trustworthy than seconds on a laptop.

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
| `ENABLE_BENCHMARKS` | OFF     | Build `nawa_bench` (Google Benchmark)         |
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
