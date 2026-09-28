# Performance

This file starts with the **optimization progress** of step 9, then documents the
**baseline**: the naive engine from steps 1-7, measured before any optimization. Every
optimization is compared against `benchmarks/results/baseline.json` with
`python/compare_bench.py`.

## Optimization progress (step 9)

| stage | change | result file |
|---|---|---|
| baseline | naive i-j-k matmul, `-O3`, SSE2 only | `baseline.json` |
| 9.1 | i-k-j loop order in matmul (same flags) | `step9_1_loop_order.json` |
| 9.2 | `-DNAWA_NATIVE=ON` (`-march=native`: AVX2, FMA) on top of 9.1 | `step9_2_native.json` |

Medians of 5 repetitions, pinned to one core, interleaved. Peak = 32 FLOP/cycle
(see "Peak single-core FP32 throughput" below).

| benchmark | baseline | 9.1 | 9.2 |
|---|---|---|---|
| matmul 512³, FLOP/cycle (% of peak) | 0.51 (1.6%) | 3.77 (11.8%) | 5.80 (18.1%) |
| matmul 1024³, FLOP/cycle (% of peak) | 0.24 (0.7%) | 3.38 (10.6%) | 4.44 (13.9%) |
| MNIST layer 1 {256,784}×{784,128}, FLOP/cycle (% of peak) | 0.51 (1.6%) | 4.18 (13.1%) | 5.44 (17.0%)¹ |
| MNIST layer 1, one image {1,784}×{784,128}, FLOP/cycle | 0.51 | 3.60 | 4.79 |
| MNIST layer 2 {256,128}×{128,10}, FLOP/cycle | 0.65 | 1.73 | 1.91 |
| `predict` batch 1, cycles per image | 407k | 60.0k (6.8×) | 58.2k (7.0×) |
| `predict` batch 256, cycles per image | 407k | 56.0k (7.3×) | 50.3k (8.1×) |
| `predict` batch-1 latency p50 / p99 (µs, wall) | 134 / 172 | 18.0 / 25.6 | 18.9 / 28.3² |
| `predict` batch 256, images/s (wall) | 6,194 | 55,927 | 65,770 |
| `nawa eval`, 10,000 images, accuracy | 97.15% | 97.15% (bit-identical to naive) | 97.15% (same 285 mistakes) |

¹ This benchmark varies more from run to run in the native build: separate runs gave 5.9M,
8.3M (three times) and 9.5M cycles (5.44 FLOP/cycle is from the recorded 9.5M). Fast, memory-heavy
kernels are more sensitive to where their buffers sit in memory than the old latency-bound
loop. Compare stages with several runs, not one.

² Wall-clock latency depends on the clock at the time of the run. In cycles, batch 1
improved 1.03× from 9.1 to 9.2.

### What each stage did

- **9.1, i-k-j.** The inner loop walks contiguous rows of B and C with A[i][k] in a
  register. Iterations are independent, so GCC vectorizes it with 4-wide SSE
  (`movups, mulps, addps, movups`) instead of the serial `addss` chain. Results are
  bit-identical to `matmul_naive`: the summation order per element is unchanged.
- **9.2, `-march=native`.** The same loop becomes 8-wide AVX2 with a fused multiply-add
  (`vmovups ymm, vfmadd213ps ymm, vmovups`). GCC keeps 256-bit vectors on this AVX-512 CPU by
  default (`-mprefer-vector-width=256`). FMA rounds once, so 770 of 10,000 confidences change
  in the last bits (max 1.1e-6), with no prediction changes.

### 9.3: cache-blocked GEMM, packing, AVX2 micro-kernel, runtime dispatch

`src/tensor/gemm.cpp` (portable) and `src/tensor/gemm_avx2.cpp` (compiled with
`-mavx2 -mfma`, chosen at runtime by `__builtin_cpu_supports`; `NAWA_KERNEL=portable|avx2`
overrides it). The **default portable build** gets the AVX2 kernel on any CPU that has it.

- **Packing B:** columns in panels of 16, each stored as K consecutive rows of 16 floats, so
  the micro-kernel reads B strictly sequentially. `Linear` packs its weights once, at model
  load. Packing the 784×128 layer-1 weights costs **69k cycles**, which is **2.7× a whole
  batch-1 `predict`** now, and pre-packing saves it on every call.
- **Blocking (BLIS scheme):** KC = 256, so a 256×16 B micro-panel (16 KB) and a 6×256 A
  micro-panel (6 KB) fit in the 48 KB L1. MC = 96, so the packed A block (96 KB) stays in
  the 1.25 MB L2. NC = 512.

  Tuning over MC ∈ {48, 96, 144, 192} × KC ∈ {128, 256, 384, 512} moved results by at most
  ~5%, so the cache-derived values stay.
- **Micro-kernel:** a 6×16 tile of C in 12 ymm accumulators. Per k: 2 loads of B,
  6 broadcasts of A, 12 FMAs, and no memory traffic for C inside the loop (objdump):
  ```
  vmovups (%r8),%ymm2 ; vmovups 0x20(%r8),%ymm1          B[k][0..16)
  vbroadcastss -0x18(%rax),%ymm3                          A[row 0][k]
  vfmadd231ps %ymm2,%ymm3,%ymm14 ; vfmadd231ps %ymm1,%ymm3,%ymm13
  ... 5 more broadcast + 2 FMA pairs (ymm12..ymm0) ...
  ```
  12 independent accumulators hide the FMA latency: 2 units × 4 cycles = 8 must be in
  flight.
- **Edge tiles:** row counts 1-5 use smaller instantiations of the same kernel. Partial
  column panels go through a 6×16 temporary.
- **M = 1 (batch 1)** uses a separate matrix-vector loop over the packed panels (4 panels at
  once, 8 accumulators). It never packs A, and it gives bit-identical results to a row
  inside a batch (same k order).
- **Plain `matmul()`** (B not pre-packed) packs B only from **M ≥ 3**. Measured at
  K = 784, N = 128, in cycles:

  | M | unpacked i-k-j | pack + GEMM |
  |---|---|---|
  | 1 | **48k** | 91k |
  | 2 | **97k** | 100k |
  | 4 | 194k | **106k** |
  | 8 | 388k | **146k** |
  | 32 | 1.55M | **323k** |
- **AVX-512 was not implemented.** On this CPU it has the same 32 FLOP/cycle peak (one
  512-bit FMA unit instead of two 256-bit ones), and the AVX2 kernel already reaches 84-90%
  of that on large shapes. At most ~10% is left, and 512-bit code risks lower clocks.

**FLOP/cycle per shape** (median, default build, peak = 32):

| M×K×N | `matmul()` (packs B per call) | % of peak | pre-packed B (as in Linear) | % of peak |
|---|---|---|---|---|
| 32³ | 13.4 | 42% | 17.3 | 54% |
| 64³ | 20.7 | 65% | 23.3 | 73% |
| 128³ | 24.1 | 75% | 27.0 | 84% |
| 256³ | 27.4 | 86% | 28.7 | 90% |
| 512³ | 26.5 | 83% | 28.1 | 88% |
| 1024³ | 25.0 | 78% | 26.5 | 83% |
| 1×784×128 | 4.3 (unpacked i-k-j) | 13% | 8.6 (gemv) | 27% |
| 256×784×128 | 25.5 | 80% | 26.8 | 84% |
| 256×128×10 | 9.9 | 31% | 10.4 | 32% |

Small shapes and N = 10 can't fill the 6×16 tiles or amortize loop overhead. Batch 1 is
limited by streaming the 401 KB of weights from L2 (about 17 bytes per cycle), not by FMAs.

**After 9.3,** batch-256 `predict` takes 3.74M cycles: Linear(784→128) 2.25M (60%, of which
the bias add ~0.33M), preprocess 0.72M (19%), ReLU 0.48M (13%), Linear(128→10) 0.09M, and
Softmax 0.08M. **The non-GEMM work is now about half the time.** Still 65 allocations per call.

In the `NAWA_NATIVE=ON` build `predict` is *slower* after 9.3 (39k vs 26k cycles at batch
1), because of the broadcasting regression below. Fixed in 9.4.

### 9.4: memory and fusion

- **Tensor storage is 64-byte aligned** (`AlignedBuffer`, replacing `std::vector<float>`).
  `Tensor::resize()` reuses capacity.
- **`Workspace` + `Model::predict(raw, workspace)`:** two ping-pong buffers, reused across
  calls. **0 heap allocations per predict in steady state** (was 65). A test asserts this
  with a counting `operator new` in the test binary. The convenience `predict(raw)`, which
  creates a fresh workspace each call, makes 21.
- **Execution plan built at load:** `Linear(784→128) + ReLU (fused) → Linear(128→10) →
  Softmax`. Bias and ReLU are applied in the **GEMM epilogue**, as each C register is
  stored. The result is bit-identical to the layer-by-layer path (tested, including NaN and
  −0.0 through ReLU).
- **Broadcasting:**
  - Fast paths for row (`{M,N}+{N}`) and column (`{M,N}+{M,1}`) broadcasts.
  - The general path now updates offsets incrementally instead of recomputing
    `Σ index·stride` per element.
  - The bias-broadcast add fell from **329k to 21.5k cycles (15.3×)**. This also removed the
    9.2 native-build regression (1.77M → 20k).
- **Preprocess and Softmax** write into the workspace: Softmax is one pass per row in place
  (no temporaries), preprocess is one loop.

| cycles per call | 9.3 | 9.4 | speedup |
|---|---|---|---|
| `predict` batch 256 | 3.74M | 2.16M | 1.73× |
| `predict` batch 32 / batch 8 | 410k / 118k | 283k / 84.8k | 1.45× / 1.39× |
| bias-broadcast add {256,128}+{128} | 329k | 21.5k | 15.3× |
| Softmax {256,10} | 84.5k | 41.2k | 2.05× |
| preprocess {256,784} | 722k | 341k | 2.12× |
| heap allocations per `predict` (workspace) | 65 | **0** | |
| batch-1 latency p50 / p90 / p99 (µs, wall) | 8.8 / 9.1 / 13.8 | **7.6 / 7.6 / 8.9** | |

**Batch 1 in cycles is noisy in a new way.** The 1-row GEMV streams the 401 KB of weights
from L2 and runs in either a ~15.5k-cycle or a ~23k-cycle mode, even within one recorded run
(`MatmulPacked` M=1: 15.4k, `Linear` batch 1: 23.7k). The probable cause is activity on the
sibling hyperthread of the pinned core, which shares L1 and L2. What does hold within a
single run: batch-1 `predict` minus its first layer (preprocess, second layer, softmax, glue)
fell from **~7.8k cycles (9.3) to ~1.7k (9.4)**. The p99 latency, which is where
allocations and extra passes showed up, dropped from 13.8 to 8.9 µs.

**Memory:** peak RSS after a batch-256 predict fell from 9.1 MiB (baseline) to 7.4 MiB.
The weights are held twice, once row-major for `Linear::weight()` and once packed (+398 KB).

### 9.5: multithreading

- **`ThreadPool`** (`runtime/thread_pool.hpp`):
  - Fixed workers created once; the caller participates as thread 0.
  - `parallel_for` with **static round-robin chunks** (the same thread always gets the same
    part of the weights, which keeps them in its core's private L2).
  - A job queue with a condition variable; exceptions propagate to the caller; clean shutdown.
  - Nested calls run inline. Allocation-free in steady state.
- **Parallel GEMM:**
  - Split over **M** (row blocks in multiples of 6) when there are at least 6 rows per
    thread, otherwise over the **N panels** (16 columns each).
  - Each output is computed by one thread with the same k-ordered FMAs, so results are
    **bit-identical for 1, 2, 3, 4 and 8 threads** (tested for GEMM shapes and full
    `predict`).
- **Threads:** `NAWA_NUM_THREADS`, default = physical cores (P-cores on hybrid CPUs), capped
  by the process's CPU affinity. Here: *4 threads (4 physical cores, 8 logical CPUs)*.
- **`ENABLE_TSAN`:** ThreadSanitizer, clean on GCC and Clang, now a CI job. A deliberately
  racy test program was flagged, so TSan really is active.
- **Minimum work for threads: 2²² ≈ 4.2M multiply-adds**, measured (forced parallel,
  K = 784, N = 128, wall time):

  | M | multiply-adds | 1 thread | 4 threads |
  |---|---|---|---|
  | 1 | 100k | 5.8 µs | 6.0 µs |
  | 4 | 401k | 10.5 µs | 18.1 µs |
  | 16 | 1.6M | 43.6 µs | 41.6 µs |
  | 64 | 6.4M | 161 µs | 113 µs |
  | 256 | 25.7M | 643 µs | 379 µs |

  So batch 1 and the small second layer stay single-threaded. Only the batch-256 first
  layer runs in parallel.
- **A spin-wait before sleeping was tried and removed.** Workers spinning up to ~1 ms after
  each job gave no measurable gain (spin 0 / 500 / 2,000 / 20,000 iterations were all within
  noise). It only burned power that the working core needed.

**Scaling (wall time, medians, `step9_5_threads.json`; speedup and parallel efficiency):**

| | 1 thread | 2 | 4 | 8 (with hyperthreads) |
|---|---|---|---|---|
| `predict` batch 1 | 7.9 µs | 9.4 µs (0.84×) | 8.1 µs (0.98×) | 8.0 µs (0.99×) |
| `predict` batch 256 | 705 µs | 465 µs (1.52×, 76%) | 488 µs (1.44×, 36%) | 423 µs (1.67×, 21%) |
| GEMM 256×784×128 | 623 µs | 400 µs (1.56×, 78%) | 331 µs (1.88×, 47%) | 277 µs (2.25×, 28%) |
| GEMM 1024³ | 26.6 ms | 15.5 ms (1.72×, 86%) | 11.0 ms (2.42×, 61%) | 12.2 ms (2.18×, 27%) |

Batch 1 doesn't change: it stays on one thread by design. Its cycles per image are identical
in 9.4 and 9.5 (25.2k), and the wall differences are noise.

**Where scaling stops, and why (`perf stat`, GEMM 1024³):**

| threads | CPUs busy | clock per busy CPU | IPC per busy CPU |
|---|---|---|---|
| 1 | 1.00 | 3.61 GHz | 3.66 |
| 2 | 1.98 | 2.08 GHz | 3.70 |
| 4 | 3.70 | 1.46 GHz | 3.52 |
| 8 | 6.68 | 1.40 GHz | 2.03 |

- **Each core does the same work per cycle at every thread count** (IPC ≈ 3.6).
- **The clock collapses.** The i7-11370H is a 35 W laptop chip, and 4 cores running AVX2
  FMAs exceed its power budget, so it lowers every core's frequency. Total cycles per second
  grow only from 3.6 G (1 thread) to about 5.4 G (4 threads), which caps the speedup at
  ~1.5-2.4× depending on thermal state.
- **8 threads add nothing,** because two hyperthreads share one core's FMA units (IPC per
  thread halves).
- **Batch 256 `predict` scales less than its GEMM,** because the rest of `predict`
  (preprocess, second layer, softmax: about 20% of the single-thread time) stays
  single-threaded. That's Amdahl's law.

On a desktop CPU or a server with a bigger power budget, the same code should scale much
closer to the core count.

### Known regression in 9.2: broadcasting got slower

**Fixed in 9.4** (incremental offsets and row/column fast paths).

With `-march=native`, GCC also vectorizes the tiny per-element loop in the general
broadcasting path (`offset += index[d] * stride[d]`, 2 iterations for 2-D tensors) using
AVX-512 64-bit multiplies (`vpmullq`), mask registers and cross-lane shuffles. For 2 iterations
that setup costs far more than the scalar code. Measured in the same run:

| cycles per call | portable (9.1) | native (9.2) |
|---|---|---|
| bias-broadcast add {256,128}+{128} | 0.32-0.33M | **1.77M (5.4×)** |
| Softmax {256,10} (uses broadcast subtract and divide) | 83k | **329k (4×)** |

`predict` still got faster overall, because matmul dominates. But in the native build the
bias add is now ~17-23% of a Linear(784→128) forward pass at batch 256.

### Where the time goes after 9.2

| `predict` part (native build) | batch 1 | batch 256 |
|---|---|---|
| Linear(784→128): matmul + bias add | 72% | 85% (matmul ~62-77%, bias add ~17-23%) |
| Linear(128→10) | 4.5% | 4.4% |
| Softmax | 5.3% | 3.0% |
| preprocess | 3.9% | 6.1% |
| ReLU | 0.4% | 0.2% |
| other (input copy, 65 allocations, glue) | 14% | 1.3% |

Matmul went from ~98% to roughly 60-75% of `predict`. Amdahl's law is now visible: the 1.3-1.6×
matmul gain from 9.2 turned into only 1.03-1.11× for `predict`, because the rest didn't get
faster. Some of it (broadcasting) even got slower.


## How to run

```bash
# Benchmarks need a Release build (nawa_bench refuses to run otherwise)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_BENCHMARKS=ON
cmake --build build -j

# Everything, once (quick look)
./build/bin/nawa_bench

# A recorded result: pinned to one core, 5 repetitions, interleaved, medians saved as JSON
taskset -c 2 ./build/bin/nawa_bench --benchmark_repetitions=5 \
    --benchmark_enable_random_interleaving=true --benchmark_report_aggregates_only=true \
    --benchmark_out=benchmarks/results/new.json --benchmark_out_format=json

# Compare (cycles are the reliable metric on this laptop, see "Noise" below)
python3 python/compare_bench.py benchmarks/results/baseline.json benchmarks/results/new.json \
    --metric cycles_per_iter
python3 python/compare_bench.py benchmarks/results/baseline.json benchmarks/results/new.json
```

- `taskset -c 2` keeps the benchmark on one core. Otherwise the scheduler can move it between
  cores that have different cache contents.
- `--benchmark_enable_random_interleaving` runs the repetitions of all benchmarks in a
  shuffled order. Without it, the benchmarks that run first get the CPU while it's cool and
  turbo-boosted, and the later ones get it once it has throttled.

## Machine and build

| | |
|---|---|
| CPU | 11th Gen Intel Core i7-11370H (Tiger Lake, Willow Cove cores), 4 cores / 8 threads |
| Clock | 3.3 GHz base, up to 4.8 GHz single-core turbo; `powersave` governor (intel_pstate) |
| Caches (per core) | L1d 48 KiB (12-way), L1i 32 KiB, L2 1.25 MiB |
| Caches (shared) | L3 12 MiB |
| SIMD the CPU supports | SSE4.2, AVX2, FMA, AVX-512 (F, VL, BW, VNNI) |
| **SIMD the build uses** | **Baseline x86-64 only (SSE2)**: flags are `-O3 -DNDEBUG` with no `-march=native`, so AVX2, FMA and AVX-512 are never used |
| Compiler | GCC 16.2.1, Release |
| Benchmark library | Google Benchmark v1.9.5 |
| Date | 2026-09-28 |

`baseline.json` records all of this in its `context` block (the `nawa_*` keys and the cache
list).

## Noise: why cycles, not only seconds

A laptop CPU doesn't run at a fixed clock:
- It turbo-boosts to ~4.8 GHz until its short-term power budget runs out (tens of seconds).
- It then settles around 2.5-3.2 GHz, depending on temperature.

During the baseline run the measured clock ranged from **2.46 to 3.11 GHz** between
benchmarks. So `nawa_bench` also counts **CPU cycles** with Linux perf events
(`benchmarks/cycle_counter.cpp`, user space only, no root needed). It reports `GHz`,
`cycles_per_iter`, `cycles_per_image` and `FLOP_per_cycle`.

| Across 5 repetitions | wall time CV | cycles CV |
|---|---|---|
| matmul 512² | 7.0% | 0.4% |
| MNIST layer 1 matmul {256,784}×{784,128} | 11.6% | 0.4% |
| `predict` batch 256 | 15.4% | 1.2% |
| `predict` batch 1 | 12.6% | 0.5% |

(CV is the coefficient of variation: stddev / mean.)

**A demonstration.** Running the *identical* binary again a few minutes later, as a short run
that got full turbo, `compare_bench.py` reported:

| benchmark | wall time "speedup" | cycles "speedup" |
|---|---|---|
| matmul 512² | 1.36× | 0.99× |
| `predict` batch 256 | 1.51× | 0.99× |
| `predict` batch 1 | 1.26× | 1.00× |

Nothing changed. The 1.5× was the clock. **Judge optimizations by cycles first**, and report
wall time alongside.

For steadier wall-clock numbers you can disable turbo while benchmarking. This needs root, so
run it yourself if you want it:

```bash
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo   # disable turbo
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo   # re-enable afterwards
```

## Peak single-core FP32 throughput

Each Willow Cove core has **two** 256-bit FMA units (on execution ports 0 and 1). For AVX-512
they fuse into **one** 512-bit FMA unit (client Tiger Lake has no second 512-bit FMA on port 5).
Either way:

```
AVX2 + FMA:  2 FMA units × 8 floats (256 bit) × 2 FLOP (multiply + add) = 32 FLOP/cycle
AVX-512:     1 FMA unit  × 16 floats (512 bit) × 2 FLOP                 = 32 FLOP/cycle

Peak = 32 FLOP/cycle × 4.8 GHz (max turbo) = 153.6 GFLOPS
     = 32 FLOP/cycle × ~3.0 GHz (clock sustained during the baseline) ≈ 96 GFLOPS
```

That is an upper bound, reachable only by hand-tuned matmul kernels with data in L1. The
clock-independent comparison is FLOP/cycle against **32**:

| matmul | FLOP/cycle | % of peak (32) | GFLOPS (median) |
|---|---|---|---|
| 32×32×32 | 0.95 | 3.0% | 2.94 |
| 64³ | 0.79 | 2.5% | 2.45 |
| 128³ | 0.62 | 1.9% | 1.95 |
| 256³ | 0.55 | 1.7% | 1.56 |
| 512³ | 0.51 | **1.6%** | 1.47 |
| 1024³ | 0.23 | **0.7%** | 0.65 |
| MNIST {1,784}×{784,128} | 0.51 | 1.6% | 1.36 |
| MNIST {256,784}×{784,128} | 0.51 | 1.6% | 1.51 |
| MNIST {256,128}×{128,10} | 0.65 | 2.0% | 2.00 |

### Why exactly 0.5 FLOP/cycle: the generated code

`objdump` of `inference::matmul` shows GCC's inner loop. It handles 4 values of `k` per
iteration:

```
movss ×4, unpcklps ×2   gather 4 values of B from 4 different rows (one at a time)
movups                  load 4 consecutive values of A
mulps                   4 multiplications in one SSE instruction     <- vectorized
addss ×4                acc += p0; acc += p1; acc += p2; acc += p3   <- serial
```

The multiplications are vectorized, but the additions into `acc` stay a **serial dependency
chain**. Floating-point addition is not associative ((a+b)+c ≠ a+(b+c) after rounding), and
without `-ffast-math` the compiler must add in source order.
- Each `addss` waits for the previous one: about **4 cycles** of latency on this core.
- So 4 multiply-adds (8 FLOP) take ≥ 16 cycles, which is **0.5 FLOP/cycle**.
- Every large-K case measures 0.51.

Small matrices do slightly better (0.95 at 32³) because each `acc` chain is short. The
out-of-order core then overlaps the end of one output element's chain with the start of the
next.

## Baseline results

Medians of 5 repetitions; `baseline.json` has the full aggregates (mean, median, stddev, CV).
Times are wall clock at whatever clock the CPU had, so read them with the noise section in mind.

### Matmul

| shape (M×K×N) | time | GFLOPS | cycles / call |
|---|---|---|---|
| 32³ | 22.4 µs | 2.94 | 69.1k |
| 64³ | 215 µs | 2.45 | 665k |
| 128³ | 2.16 ms | 1.95 | 6.80M |
| 256³ | 21.6 ms | 1.56 | 61.3M |
| 512³ | 186 ms | 1.47 | 526M |
| 1024³ | 3.31 s | 0.65 | 9.14G |
| {1,784}×{784,128} | 148 µs | 1.36 | 394k |
| {256,784}×{784,128} | 34.4 ms | 1.51 | 101M |
| {256,128}×{128,10} | 328 µs | 2.00 | 1.01M |

### Memory-bound operations

GB/s here is the **minimum** traffic: every input byte read once and every output byte written
once.

| operation | time | GB/s | cycles / call |
|---|---|---|---|
| add, same shape {256,128} (128 KiB per tensor) | 8.5 µs | 46.9 | 25.1k |
| add, same shape {1024,1024} (4 MiB per tensor) | 952 µs | 13.6 | 2.36M |
| add, bias broadcast {256,128} + {128} | 145 µs | **1.86** | 340k |
| transpose 512×512 | 842 µs | 2.50 | 2.57M |
| softmax {256,10} | 29.3 µs | 0.70 | 86.5k |

- **Same-shape add is fast** (47 GB/s while the data fits in L2). It's one linear loop the
  compiler vectorizes. At 4 MiB per tensor it falls to 13.6 GB/s, because the data now streams
  from L3 or DRAM.
- **The bias-broadcast add is 13.5× slower per element** than the same-shape add, on the same
  number of elements. The general broadcasting path recomputes every input offset from an
  index "odometer" for every element, which is an inner loop over dimensions per value. This
  shape (a row vector added to every row) runs in *every* Linear layer.
- **Transpose** reads rows and writes columns. Each write to the output lands 2 KiB after the
  previous one, so the writes hit a new cache line every time.
- **Softmax** does 5 passes (max, subtract, exp, sum, divide) and creates 5 temporary tensors
  for only 2,560 values. The fixed per-operation overhead dominates.

### Layers and preprocessing (real model weights)

| | batch 1 | batch 256 |
|---|---|---|
| preprocess | 0.92 µs, 2.7k cycles | 274 µs, 772k cycles |
| Linear(784→128) | 159 µs, 398k cycles | 33.2 ms, 101M cycles |
| ReLU | 0.21 µs, 582 cycles | 200 µs, 496k cycles |
| Linear(128→10) | 1.6 µs, 5.0k cycles | 368 µs, 1.06M cycles |
| Softmax | 0.77 µs, 2.3k cycles | 28 µs, 84k cycles |

### Full `Model::predict`

| batch | time / batch | images / s | cycles / image | heap allocations / predict | bytes allocated / predict |
|---|---|---|---|---|---|
| 1 | 136 µs | 7,396 | 407k | 65 | 17.8 KiB |
| 8 | 1.07 ms | 7,512 | 403k | 65 | 137 KiB |
| 32 | 4.62 ms | 7,023 | 406k | 65 | 545 KiB |
| 256 | 42.3 ms | 6,194 | 407k | 65 | 4.36 MiB |

- **Batch-1 latency** over single calls: **p50 134 µs, p90 142 µs, p99 172 µs**.
- **Batching doesn't help yet.** Cycles per image are identical (~405k) from batch 1 to batch
  256. The naive matmul does the same work per row either way. It never reuses a loaded weight
  for several images while it is still in cache, which is the main benefit batching gives an
  optimized engine.
- The step-7 CLI measured about 9,500 images/s. That was a short run at full turbo; the
  cycles are the same.

### Memory

| moment | peak resident memory (`ru_maxrss`) |
|---|---|
| program start | 4.5 MiB |
| after loading the model | 5.5 MiB (+1.0 MiB: 398 KiB of weights plus the 398 KiB file buffer during parsing) |
| after a batch-256 `predict` | 9.1 MiB (+3.6 MiB of temporary tensors) |

**65 heap allocations per `predict`, whatever the batch size.**
- The count is fixed by the number of operations, not the data size. Every intermediate result
  is a new `Tensor`, and each `Tensor` holds three `std::vector`s (data, shape and strides), so
  up to 3 allocations each.
- Broadcasting also allocates temporary shape, stride and index vectors.
- Examples: `preprocess` alone creates 4 tensors (the input copy, `* scale`, `- mean`,
  `/ std`), and Softmax creates 5.
- At batch 256 the same 65 allocations move 4.36 MiB per call.

## `perf stat`: the 512² matmul

This was measured without root: user-space events only, allowed by the default
`perf_event_paranoid = 2`. Each run is one benchmark iteration plus the start-up code.

```bash
taskset -c 2 perf stat -e cycles:u,instructions:u,cache-references:u,cache-misses:u,\
L1-dcache-loads:u,L1-dcache-load-misses:u,dTLB-load-misses:u \
    ./build/bin/nawa_bench --benchmark_filter='BM_Matmul/M:512/' --benchmark_min_time=1x
```

If perf ever reports "permission denied", check `cat /proc/sys/kernel/perf_event_paranoid`.
`sudo sysctl kernel.perf_event_paranoid=2` (or `1`) restores access. That's a system setting,
so change it yourself if needed.

| counter | 512³ | 1024³ |
|---|---|---|
| instructions / cycle (IPC) | **1.36** | **0.64** |
| L1d line fills per L1d load (`L1-dcache-load-misses` / `L1-dcache-loads`) | **85%** | 81% |
| last-level cache references | 5.9M (≈0.04 per multiply-add) | 1.08G (**≈1.0 per multiply-add**) |
| last-level cache miss rate (`cache-misses` / `cache-references`) | 2.7% | 1.2% |
| dTLB misses that needed a page walk | 4.6k | 126k |

On Intel, `L1-dcache-load-misses` counts L1 line replacements, which includes hardware
prefetches. So "85%" means close to one new cache line per load, not literally 85% of loads
stalling.

### What this says about the i-j-k loop

- **Almost every load of B misses L1.** The inner loop walks a *column* of B:
  `B[k*N + j]`, `B[(k+1)*N + j]`, … are N floats apart, 2 KiB at N = 512. Each access needs a
  new 64-byte cache line and uses 4 bytes of it.
- **It's worse than the stride alone.** With a stride that's a power of two, the whole column
  maps to the same few cache **sets**. The L1 is 12-way with 64 sets. Only address bit 11
  changes between rows, so the 512 lines of one column compete for **2 sets × 12 ways = 24
  lines**. The line holding `B[k][j+1]`, which the next `j` needs, is long gone by then.
  (This is a set-associativity *conflict*; the L1 isn't too small.)
- **At 512 those misses are mostly hidden.** B is 1 MiB and fits in the 1.25 MiB L2, so each
  L1 miss costs ~14 cycles. The serial `addss` chain already takes 16 cycles per 4 k-steps, so
  the misses overlap with it and IPC stays at 1.36.
- **At 1024 they aren't.** B is 4 MiB and no longer fits in L2, so nearly every
  multiply-add now goes to L3 (last-level references jump to ~1 per multiply-add), at ~40-50
  cycles each. IPC halves to 0.64 and FLOP/cycle drops from 0.51 to 0.23.
- **The low last-level miss rate (1-3%) is misleading.** It only says the matrices fit in the
  12 MiB L3, so DRAM is rarely touched. The damage happens in L1 and L2.
- **TLB misses don't matter much.** Page walks stay small (126k over 1.07 billion
  multiply-adds at 1024), since the L2 TLB covers the working set.

## Where does the time go? (`predict`, batch 256)

These are cycles per call from the individual benchmarks at the same shapes, compared with the
full `predict`:

| part | cycles | share of `predict` |
|---|---|---|
| preprocess (`(x·scale − mean) / std`) | 0.77M | 0.7% |
| Linear(784→128): **matmul** {256,784}×{784,128} | **100.7M** | **96.7%** |
| Linear(784→128): bias add (broadcast) | 0.34M | 0.3% |
| ReLU | 0.50M | 0.5% |
| Linear(128→10): **matmul** {256,128}×{128,10} | **1.01M** | **1.0%** |
| Linear(128→10): bias add | 0.04M | 0.04% |
| Softmax | 0.08M | 0.1% |
| everything else (input copy, allocations, glue) | ~0.7M | ~0.7% |
| **total `predict`** | **104.1M** | 100% |

**About 97.7% of the time is the two matmuls, and 96.7% is the first layer's matmul alone.**
The biggest remaining item is the bias-broadcast add, and it's 300× smaller.

At batch 1 the picture is the same: the first matmul takes 394k of 407k cycles (97%).

Consequences for step 9, in order of payoff:

1. **Matmul first.** Loop order (i-k-j) or independent accumulators remove the serial
   dependency chain, and the compiler can then vectorize. Blocking or tiling keeps B's lines in
   cache. Compiler flags (`-march=native` for AVX2/FMA/AVX-512) widen the vectors. Threads come
   last. Matmul is ~98% of the runtime, so everything else combined can be at most ~2% of any
   speedup.
2. **After matmul is fast,** the small costs grow in relative terms: the 13.5× slower broadcast
   path, the 65 allocations, and softmax's 5 passes. Allocations matter most for batch-1
   latency: each call pays the same fixed cost for them, however little data it processes.
