# Project rules

## Goal

A lightweight neural network inference engine in C++, written from scratch. It is a
**learning project**: the point is to understand how trained networks run at the systems level.

- Prefer clear, readable code over clever code.
- Explain non-obvious decisions in comments (the *why*, not the *what*).

## Technical constraints

- C++20. All code lives in the `inference` namespace.
- Public headers go in `include/inference/<module>/`. Implementations go in `src/<module>/`.
- **No external math libraries** (no Eigen, BLAS, etc.). All core math is implemented by hand.
  GoogleTest is the only third-party dependency, and it is used only for tests.
- The build must have zero warnings (`-Wall -Wextra -Wpedantic -Werror` / `/W4 /WX`).
- Format code with `.clang-format` (Google-based, 4-space indent, 100 columns).

## Tensor design (`include/inference/tensor/tensor.hpp`)

- `using Shape = std::vector<std::size_t>;`. The empty shape `{}` is a scalar with `numel() == 1`.
- **float32 only.**
- **Contiguous and row-major** (C-order) always.
- **Owning:** every `Tensor` owns its data. There are no views and no shared storage.
  Copying is a deep copy. Moving is cheap (the default move). `reshape()` and `flatten()`
  return new tensors (copies).
- **Strides are in elements, not bytes.** They are computed from the shape
  (`compute_strides`) and stored. The element at `{i0, i1, ...}` is at offset
  `sum(ik * strides[k])`.
- A dimension of size 0 is rejected.
- Storage is a private `std::vector<float>`, reached only through the class API (`data()`,
  `at()`, ...). It will be swapped for 64-byte-aligned memory in the optimization step, so
  nothing outside `Tensor` may depend on it being a `std::vector`.
- Errors: `std::invalid_argument` for bad shapes or data sizes, and `std::out_of_range` for
  bad indices or dimensions. Messages include the shapes/indices involved.
- `allclose(a, b, rtol, atol)` is the comparison to use in tests (numpy rule; different
  shapes → false).

## Workflow rules

- Every new feature comes with GoogleTest tests in `tests/`.
- A step is done only when all tests pass.
- Correctness first, optimization later.
- Every optimization is measured with a benchmark before and after the change. Record the numbers.
- Python (`python/`) is used only for training (PyTorch, MNIST) and exporting weights.

## Commands

```bash
cmake -S . -B build                             # configure (Release by default)
cmake --build build -j                          # build
ctest --test-dir build --output-on-failure      # run all tests

# Debug + AddressSanitizer/UBSan
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON
cmake --build build-debug -j && ctest --test-dir build-debug --output-on-failure
```

To add a test file, create `tests/test_<name>.cpp` and add it to `inference_tests` in
`tests/CMakeLists.txt`. To add a source file, add it to `inference_core` in the root
`CMakeLists.txt`.

## Git workflow

- Repository: https://github.com/AbdullahThweib2004/nawa-inference (remote `origin`, branch `main`).
- For every roadmap step: finish the step → all tests pass → commit → `git push`.
- Never commit build output, model binaries, or secrets (see `.gitignore`).

## Roadmap

1. Project scaffolding ✅
2. Tensor core (storage, shape, strides, indexing) ✅
3. **Tensor operations (elementwise, matmul, reductions, activations)** ← *current step*
4. Layers (Linear, ReLU, Softmax, Conv2D, ...)
5. Python training and weight export (PyTorch, MNIST)
6. Model loading and runtime (file format, computational graph, executor)
7. End-to-end MNIST inference
8. Benchmarking infrastructure
9. Optimization (threads, SIMD, INT8 quantization)
10. Extensions
