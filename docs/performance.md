# Performance

This is the **baseline**: the naive engine from steps 1-7, measured before any optimization.
Every optimization in step 9 is compared against `benchmarks/results/baseline.json` with
`python/compare_bench.py`.

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
