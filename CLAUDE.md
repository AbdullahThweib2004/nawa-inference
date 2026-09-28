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
  Third-party code is limited to GoogleTest (tests only) and **stb_image** (image *decoding*
  for the CLI, which is not math). stb_image is fetched as a single header pinned to a commit
  and verified by SHA-256. It is compiled in its own `stb_image_impl` target with warnings off,
  and used only through the separate `nawa_image` library, so `inference_core` has no
  third-party code.
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
- Storage is a private 64-byte-aligned `AlignedBuffer` (since stage 9.4), reached only through
  the class API (`data()`, `at()`, `resize()`, ...). Nothing outside `Tensor` may depend on
  how it is stored.
- Errors: `std::invalid_argument` for bad shapes or data sizes, and `std::out_of_range` for
  bad indices or dimensions. Messages include the shapes/indices involved.
- `allclose(a, b, rtol, atol)` is the comparison to use in tests (numpy rule; different
  shapes → false).

## Tensor ops design (`include/inference/tensor/ops.hpp`)

- Free functions in `inference`. Every op returns a **new contiguous tensor** and never
  modifies its inputs.
- **Broadcasting** (NumPy rules: align from the right; dims equal, 1, or missing).
  `broadcast_shape()` computes the output shape. Each input gets **broadcast strides**
  with one entry per output dim: 0 for a missing or size-1 dim, otherwise its normal
  stride. The output is walked in row-major order with an odometer-style index, and
  each input offset is `sum(index[d] * bstride[d])`. Nothing is expanded or copied.
  Identical shapes take a fast path: one linear loop, no index math.
- Operators `+ - * /` are **element-wise** (like NumPy/PyTorch), never matmul. Division
  by zero follows IEEE (inf/nan) and doesn't throw.
- **matmul is deliberately naive**: 2-D only, i-j-k loop over raw pointers. It is
  cache-unfriendly for `b` (the inner loop walks down a column). Leave it that way until
  the optimization step, then change it with a benchmark before and after.
  `Tensor::matmul(other)` forwards to the free function.
- `transpose` is 2-D only and copies.
- Reductions (`sum`, `max`, `mean`, `argmax`) take one axis (negative counts from the
  end) and `keepdims`. A bad axis throws `std::out_of_range`. `sum(t)` and `max(t)` reduce
  everything to a scalar. `max`/`argmax` ties keep the first occurrence.
- **argmax returns indices as float values** (float32-only tensors). This is exact up
  to 2^24. Revisit if an integer dtype is added.

## Layers design (`include/inference/layers/`)

- **`Layer` interface** (`layer.hpp`): `virtual Tensor forward(const Tensor&) const = 0`,
  `virtual std::string name() const = 0`, and `virtual std::size_t num_parameters() const`
  (defaults to 0). It has a virtual destructor, and it is **non-copyable but movable**.
  Layers are owned through `std::unique_ptr<Layer>` (the model in step 6 holds a
  `std::vector<std::unique_ptr<Layer>>`). `forward` is `const`: inference layers have no
  mutable state. Inference only, with no gradients.
- **Linear weight convention: `weight` is `{in_features, out_features}`**, and
  `forward` computes `y = x · W + b` with `x` shaped `{batch, in_features}`. **This is the
  transpose of PyTorch's `nn.Linear.weight` (`{out, in}`).** The Python exporter must
  transpose PyTorch weights (`weight.T`) before saving. The bias is optional, shape
  `{out_features}`, and broadcast over the batch.
- **Numerical stability:**
  - Sigmoid uses `1/(1+e^-x)` for x ≥ 0 and `e^x/(1+e^x)` for x < 0, so `exp` never gets a
    positive argument. ±1000 gives exactly 0/1 with no NaN or inf.
  - Softmax subtracts the max along the axis before `exp` (built from
    `max(keepdims)`, `exp`, `sum(keepdims)` and broadcasting). The largest exponent is
    `exp(0) = 1`, so there is no overflow and the denominator is ≥ 1.
- Activations (ReLU, Sigmoid, Softmax) keep the input shape and have no parameters.
  Softmax takes an `axis` (default -1).

## Model format and Python export

- **[docs/model_format.md](docs/model_format.md) is the contract** between Python and C++.
  It covers `.nawa` model files and `.ntsr` tensor files: little-endian, no padding, with
  strict validation. Change the spec first, and bump the version for any layout change.
  `python/nawa_format.py` is the reference reader/writer (numpy only).
- The **exporter transposes** PyTorch Linear weights (`{out, in}` → `{in, out}`).
- Training uses `CrossEntropyLoss` on logits, so the training model has no Softmax. The
  **exporter appends `Softmax(axis=-1)`**, and the `.nawa` model outputs probabilities.
- **Normalization lives in the model metadata** (`pixel_scale = 1/255`, `mean = 0.1307`,
  `std = 0.3081`). The C++ runtime must apply `x = (raw * pixel_scale - mean) / std` to
  raw 0..255 input before the first layer. Fixtures hold RAW pixels.
- Committed artifacts: `models/mnist_mlp.nawa`, `tests/fixtures/*.ntsr` (per-layer outputs
  for the first 3 test images, and probabilities for the first 100). `data/` and `*.pt`
  are gitignored.
- Python lives in `python/` and runs from the repo-root venv `.venv`
  (`pip install -r python/requirements.txt`). CI runs `pytest python/tests` and
  `verify_export.py` with numpy only (no torch).

## Model loading and runtime (`include/inference/model/`)

- **Reading approach** (`binary_io.hpp`): read the whole file into a
  `std::vector<std::byte>`, then parse it with `BinaryReader`, a cursor that bounds-checks
  every read. Values are copied out with `std::memcpy`, **never** by `reinterpret_cast`ing
  the buffer to `float*`/`uint32_t*` (that breaks strict aliasing and may be misaligned).
  A `static_assert` requires a little-endian host. Array sizes are checked **before**
  allocating.
- **`ModelFormatError`** (derives from `std::runtime_error`): its message has the file
  path, the byte offset of the bad field, and expected vs. found. I/O failures (a missing
  file) throw plain `std::runtime_error`.
- `tensor_io.hpp`: `read_tensor_block`/`write_tensor_block` and
  `read_tensor_file`/`write_tensor_file` (`.ntsr`), with the same checks as the Python
  reader.
- **`Model`** (`model.hpp`): `Model::load(path)` or `Model model(path)`. It holds the
  metadata (`input_shape`, `pixel_scale`, `mean`, `stddev`) and a
  `std::vector<std::unique_ptr<Layer>>` built by a factory switch on the layer type id.
  - **The layer chain is validated at load time**: the first Linear's `in_features` must
    equal `numel(input_shape)`, each Linear's must equal the previous Linear's
    `out_features`, and Softmax's axis must be in -2..1. A model that loads can run.
  - The runtime supports `norm_count == 1` only (the spec allows more). Anything else is
    rejected with a clear error.
  - API: `preprocess(raw)` = `(raw * pixel_scale - mean) / stddev`; `forward(normalized)`;
    `predict(raw)` = both; `forward_trace(normalized)` = every layer's output;
    `classify(raw)` → `{label, confidence}` per row; `summary()`; `num_parameters()`.
  - Input must be `{N, F}` or a single sample `{F}`, where `F = input_features()`.
    `{F}` is treated as `{1, F}`, and anything else throws `std::invalid_argument`.
- Tests find `models/` and `tests/fixtures/` through the compile definitions
  `NAWA_MODELS_DIR`/`NAWA_FIXTURES_DIR`; examples use `NAWA_SOURCE_DIR`. Fixture
  comparisons use the verifier's scaled error: `max_abs / max(1, max|expected|)`.

## Data and CLI (`include/inference/data/`, `tools/`)

- `idx.hpp`: MNIST IDX reader. **IDX headers are big-endian**: fields are copied with
  `memcpy` and then byte-swapped explicitly. Magic, type (0x08 only), dims and exact file
  size are validated, and errors are `ModelFormatError`. `load_mnist_images` returns
  `{N, 784}` RAW 0..255.
- `digit_preprocess.hpp`: MNIST-style preprocessing for user images. The steps are
  separate functions: a) invert if the border is light, b) crop to the bounding box of
  pixels > threshold (50), c) scale the longer side to 20 with hand-written **area
  averaging**, d) paste into 28×28 so the center of mass lands at (14, 14). The steps
  compose into `mnist_preprocess()`, and `resize_only()` is the no-preprocessing baseline.
- `image_io.hpp` (library `nawa_image`): `load_grayscale_image` via stb_image. Transparent
  pixels are composited over white.
- `nawa` CLI (`tools/nawa_cli.cpp`, built into `build/bin/`): `info`, `eval --mnist <dir>
  [--batch N]`, `predict <image> [--no-preprocess] [--show]`. CTest runs CLI smoke tests
  (`cli.*`).
- The full-test-set test skips (GTEST_SKIP) when `data/MNIST/raw` is missing, as in CI.
  The C++ engine gets **97.15%** (9,715/10,000), identical to PyTorch.
- `examples/images/` holds 3 demo PNGs generated by `python/make_example_images.py`.

## GEMM (`include/inference/tensor/gemm.hpp`, `src/tensor/gemm*.cpp`)

- `gemm(A, M, PackedMatrix, C)`: C = A·B with B **pre-packed** (`PackedMatrix::pack`).
  `Linear` packs its weights once in its constructor, i.e. at model load.
- Kernels (`GemmKernel`): `Portable` (the 9.1 i-k-j loop) and `Avx2` (BLIS-style blocking:
  MR×NR = 6×16 micro-kernel, KC = 256, MC = 96, NC = 512; a 1-row gemv path for M = 1).
  `Auto` picks AVX2+FMA at runtime via `__builtin_cpu_supports`. **`NAWA_KERNEL=portable|avx2`**
  forces a kernel; `matmul(a, b, GemmKernel)` does it per call.
- `gemm_avx2.cpp` is the only file compiled with `-mavx2 -mfma`. **It must not include
  standard-library headers or define non-internal inline functions/templates**: the linker
  could pick its AVX2 copy for the whole program (ODR), and the portable build would crash
  on CPUs without AVX2. Shared declarations go in `src/tensor/gemm_kernels.hpp` (declarations
  and constants only).
- `matmul()` (B not pre-packed) packs B only when M ≥ `kMatmulPackMinRows` (3, measured).
- CTest runs every test twice: normally, and as `portable_kernel.*` with
  `NAWA_KERNEL=portable`. `test_matmul_diff.cpp` checks every available kernel against
  `matmul_naive`, with shapes that hit all edge tiles.

## Memory and execution plan (stage 9.4)

- `Tensor` storage is `AlignedBuffer` (64-byte aligned, keeps capacity). `Tensor::resize()`
  and `resize(rows, cols)` reuse memory; `resize(rows, cols)` on a 2-D tensor allocates
  nothing.
- `*_into` ops (`matmul_into`, `add_into`, ...) write into existing tensors. In-place
  kernels: `relu_inplace`, `sigmoid_inplace`, `softmax_rows_inplace`.
- **`Model::predict(raw, Workspace&)` is the hot path: 0 heap allocations in steady state**
  (`tests/test_memory.cpp` counts them via `tests/alloc_hook.cpp`; that hook is disabled
  under sanitizers). **Rule: one Workspace per thread; Model is immutable and shareable.**
- `Model::plan()` is built at load: Linear followed by ReLU becomes one step, with bias and
  ReLU applied in the GEMM epilogue (`GemmEpilogue`). `layers()` and `forward_trace()` stay
  unfused (4 layers). Fused and unfused results must stay bit-identical (tested).
- Broadcasting: fast paths for row `{…,N}+{N}` and column `{…,R,C}+{…,R,1}`; the general
  path uses incremental offsets.

## Threads (stage 9.5)

- `include/inference/runtime/thread_pool.hpp`: `ThreadPool`, created once; `parallel_for`
  with static round-robin chunks; the caller participates; exceptions are rethrown; nested
  calls run inline. `default_thread_pool()` is used by `gemm()`. `set_num_threads(n)` is for
  tests and benchmarks only (not while the pool is in use).
- Thread count: `NAWA_NUM_THREADS`, else physical cores (P-cores on hybrid CPUs), capped by
  CPU affinity.
- GEMM goes parallel only from `parallel_threshold_macs()` = 2²² multiply-adds (measured;
  `NAWA_MIN_PARALLEL_MACS` overrides it for experiments). It splits M in multiples of 6
  rows, or N panels when M is small.
- **Invariant: results are bit-identical for any thread count** (each output is computed
  by exactly one thread, in k order). `tests/test_threads.cpp` checks this. Keep it true.
- `-DENABLE_TSAN=ON` (separate from `ENABLE_SANITIZERS`); CI runs the suite under TSan
  (clang).
- Multithreaded benchmarks use wall time (`UseRealTime`). `LoopMeter` cycles only count the
  calling thread.

## INT8 (stage 9.6)

- `include/inference/tensor/int8.hpp`: `QuantizedMatrix` (per-output-channel symmetric int8,
  stored `{out, K_pad}`, K_pad a multiple of 16), `quantize_row` (per-row activations, values
  -127..127 stored as int16), and `gemm_int8` (exact int32 sums, then
  `float(acc) * (scale_a * scale_w[j])` + bias + ReLU).
- Kernels: `int8_rows_scalar` (the reference) and `int8_rows_avx2` (`vpmovsxbw` +
  `vpmaddwd`, 4 rows × 2 channels register-blocked), chosen like the float kernels
  (`NAWA_KERNEL`, `GemmKernel`), and threaded like `gemm()`.
- **Invariant: the scalar and AVX2 INT8 kernels are bit-identical** in every build and for
  any thread count (`tests/test_int8.cpp`). That's why `int8.cpp` and `int8_avx2.cpp` are
  compiled with `-ffp-contract=off` (no FMA contraction of the float epilogue) and why both
  use the shared `finish_int8` and `quantize_row`.
- Rounding uses `round_nearest_even` (add/subtract 1.5·2²³), not `std::nearbyint`, which
  was a libm call per value in the portable build.
- `LinearInt8` (a `DenseLayer`, like `Linear`) is fused with ReLU in the plan.
  `Model::quantize()` converts every Linear layer; `Model::save()` writes version 1 for
  float32 and version 2 if any LinearInt8 is present. A loaded float32 model saves
  byte-identical to its file (tested).
- `models/mnist_mlp_int8.nawa` is committed. A test checks that quantizing
  `mnist_mlp.nawa` reproduces it byte for byte, so **regenerate it (`nawa quantize`) if the
  quantization code changes on purpose.**
- Next INT8 step if needed: an AVX-512 VNNI kernel (`vpdpbusd`). Without VNNI, int8 is slower
  than the float32 GEMM at large batch on this CPU.

## Build option `NAWA_NATIVE` (default OFF)

- `-DNAWA_NATIVE=ON` adds `-march=native` to every target. The compiler then targets
  exactly this CPU (AVX2, FMA, AVX-512 on the dev machine) instead of baseline x86-64 (SSE2).
- **Why it's off by default:** the binary then contains instructions that older or
  different CPUs don't have, and it crashes there with "illegal instruction". That would
  break CI runners (unknown CPU model) and any binary copied to another machine. Portable
  builds stay the default; `NAWA_NATIVE=ON` is for local benchmarking and local use.
- Use a separate build folder: `cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
  -DNAWA_NATIVE=ON -DENABLE_BENCHMARKS=ON`.
- **Numerics:** with FMA available, GCC and Clang contract `a*b + c` into one fused
  multiply-add, which rounds once instead of twice. Results can differ in the last bits
  (MNIST: 770 of 10,000 confidences change by at most 1.1e-6; no prediction changes). Tests
  compare against `matmul_naive` with an error bound, not bitwise, for this reason.
- To compare predictions between two builds bit for bit, use
  `nawa eval ... --save-predictions FILE` and `diff` the files.

## Benchmarking (`benchmarks/`, `docs/performance.md`)

- Google Benchmark v1.9.5, built only with `-DENABLE_BENCHMARKS=ON`, into
  `build/bin/nawa_bench`. It **refuses to run unless the build is Release**
  (`--nawa_allow_non_release` overrides this and prints a warning). CI compiles the
  benchmarks but never runs them.
- The baseline is `benchmarks/results/baseline.json`: the naive engine, -O3, no
  `-march=native`, i7-11370H.
- **Record a result:**
  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_BENCHMARKS=ON && cmake --build build -j
  taskset -c 2 ./build/bin/nawa_bench --benchmark_repetitions=5 \
      --benchmark_enable_random_interleaving=true --benchmark_report_aggregates_only=true \
      --benchmark_out=benchmarks/results/<name>.json --benchmark_out_format=json
  ```
- **Compare:** `python3 python/compare_bench.py benchmarks/results/baseline.json
  benchmarks/results/<name>.json --metric cycles_per_iter` (and again without `--metric`
  for wall time). `--filter <regex>` selects benchmarks.
- **Trust cycles over seconds on this laptop.** Turbo and thermals move wall time by 10-50%
  for identical code, while cycle counters (`cycles_per_iter`, `cycles_per_image`,
  `FLOP_per_cycle`, via perf events in `benchmarks/cycle_counter.cpp`) vary by about 1%.
- Heap allocations per `predict` are counted by a replaced `operator new` in the benchmark
  binary only (`benchmarks/alloc_counter.cpp`). The baseline makes 65 per predict at any
  batch size.
- Baseline facts: matmul is 97.7% of `predict`; naive matmul reaches 0.51 FLOP/cycle
  against a peak of 32 (1.6%), because of the serial `addss` dependency chain and B's
  column-stride cache misses.
- **`matmul_naive` (`include/inference/tensor/reference.hpp`) is the permanent reference.**
  Never optimize it. Every matmul change must pass `tests/test_matmul_diff.cpp`, whose
  tolerance is the float dot-product error bound `2*K*eps*sum|a*b|`, not exact equality.
  `BM_MatmulNaive` benchmarks it in the same run.
- Progress so far: see the table at the top of `docs/performance.md`. After 9.2, matmul
  reaches ~5.4-5.8 FLOP/cycle (17-18% of peak). The native build made the general
  broadcasting path 4-5× slower (known regression, documented there).
- Cycle counts are ~1% stable for latency-bound code, but the fast native matmul varies
  ±25% between separate runs. Compare stages over several runs.

## Workflow rules

- Every new feature comes with GoogleTest tests in `tests/`.
- A step is done only when all tests pass.
- Correctness first, optimization later.
- Every optimization is measured with a benchmark before and after the change. **Every
  optimization commit must include before/after numbers from `python/compare_bench.py`**
  (cycles first, wall time alongside) in its message or in `docs/performance.md`. Results
  must match the fixtures just as before: the tests, `verify_export.py` and the full MNIST
  eval (97.15%) must still pass.
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
- No co-author trailers or AI attribution lines in commit messages or PR descriptions
  (no `Co-Authored-By:` for AI tools, no "Generated with ..." lines). `.claude/settings.json`
  sets `attribution` to empty strings to enforce this.

## Roadmap

1. Project scaffolding ✅
2. Tensor core (storage, shape, strides, indexing) ✅
3. Tensor operations (elementwise, matmul, reductions) ✅
4. Layers (Linear, ReLU, Sigmoid, Softmax) ✅
5. Python training and weight export (PyTorch, MNIST) ✅
6. Model loading and runtime (file format, computational graph, executor) ✅
7. End-to-end MNIST inference (`nawa` CLI) ✅
8. Benchmarking infrastructure ✅
9. **Optimization (threads, SIMD, INT8 quantization)** ← *current step*
   - 9.1 i-k-j loop order in matmul ✅ (`matmul_naive` kept as the reference; differential tests)
   - 9.2 optional `-march=native` (`NAWA_NATIVE`) ✅
   - 9.3 cache-blocked GEMM + AVX2 micro-kernel + runtime dispatch + pre-packed weights ✅
   - 9.4 memory and fusion ✅ (0 allocations per workspace predict, fused Linear+ReLU
     epilogue, broadcast fast paths, 64-byte-aligned storage)
   - 9.5 multithreading ✅ (ThreadPool, parallel GEMM, bit-identical for any thread count,
     TSan CI job; scaling limited by the laptop's power budget, see docs/performance.md)
   - 9.6 INT8 quantization ✅ (`nawa quantize`, format v2; 4x smaller, 97.14% vs 97.15%;
     faster at batch 1, slower at batch 256 without VNNI, see docs/performance.md)
   - **Wrap-up** ← *current*: full progress table, README "Performance", step 9 done
10. Extensions
