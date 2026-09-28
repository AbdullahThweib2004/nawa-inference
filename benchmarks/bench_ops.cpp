// Benchmarks for individual tensor operations.

#include <benchmark/benchmark.h>

#include <cstdint>

#include "bench_common.hpp"
#include "inference/layers/activations.hpp"
#include "inference/tensor/ops.hpp"
#include "inference/tensor/reference.hpp"

using namespace inference;
using nawa_bench::random_tensor;

namespace {

// A rate counter: `per_iteration` units of work per iteration, reported per second.
benchmark::Counter rate(double per_iteration) {
    return benchmark::Counter(per_iteration, benchmark::Counter::kIsIterationInvariantRate);
}

// ---------------------------------------------------------------------------
// matmul: compute-bound in principle, so measured in GFLOPS.
// One multiply-add per (i, j, k) = 2 floating-point operations, so 2*M*K*N FLOP per call.
// ---------------------------------------------------------------------------

// Shared body for the optimized and the reference matmul, so both are measured identically.
void run_matmul(benchmark::State& state, Tensor (*fn)(const Tensor&, const Tensor&)) {
    const auto M = static_cast<std::size_t>(state.range(0));
    const auto K = static_cast<std::size_t>(state.range(1));
    const auto N = static_cast<std::size_t>(state.range(2));
    const Tensor a = random_tensor({M, K}, 1);
    const Tensor b = random_tensor({K, N}, 2);
    nawa_bench::LoopMeter meter;
    for (auto _ : state) {
        Tensor c = fn(a, b);
        benchmark::DoNotOptimize(c.data());  // the result must be treated as used
        benchmark::ClobberMemory();
    }
    meter.report(state);
    const double flop = 2.0 * static_cast<double>(M * K * N);
    state.counters["GFLOPS"] = rate(flop / 1e9);
    if (meter.has_cycles()) {
        // Clock-independent efficiency; this core's peak is 32 FLOP/cycle (docs/performance.md).
        state.counters["FLOP_per_cycle"] =
            flop * static_cast<double>(state.iterations()) / static_cast<double>(meter.cycles());
    }
}

// The same shapes for both, so one run compares them directly.
void matmul_shapes(benchmark::Benchmark* b) {
    b->ArgNames({"M", "K", "N"})
        ->Args({32, 32, 32})
        ->Args({64, 64, 64})
        ->Args({128, 128, 128})
        ->Args({256, 256, 256})
        ->Args({512, 512, 512})
        ->Args({1024, 1024, 1024})
        ->Args({1, 784, 128})    // MNIST layer 1, one image
        ->Args({256, 784, 128})  // MNIST layer 1, batch 256
        ->Args({256, 128, 10})   // MNIST layer 2, batch 256
        ->Unit(benchmark::kMicrosecond);
}

void BM_Matmul(benchmark::State& state) { run_matmul(state, matmul); }
void BM_MatmulNaive(benchmark::State& state) { run_matmul(state, matmul_naive); }
BENCHMARK(BM_Matmul)->Apply(matmul_shapes);
BENCHMARK(BM_MatmulNaive)->Apply(matmul_shapes);

// ---------------------------------------------------------------------------
// Memory-bound ops: measured in GB/s of the MINIMUM traffic (every input byte read once,
// every output byte written once). Real traffic can be higher; this is the useful-work rate.
// ---------------------------------------------------------------------------

void BM_AddSameShape(benchmark::State& state) {
    const auto rows = static_cast<std::size_t>(state.range(0));
    const auto cols = static_cast<std::size_t>(state.range(1));
    const Tensor a = random_tensor({rows, cols}, 1);
    const Tensor b = random_tensor({rows, cols}, 2);
    nawa_bench::LoopMeter meter;
    for (auto _ : state) {
        Tensor c = add(a, b);
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }
    meter.report(state);
    state.counters["GBps"] = rate(3.0 * static_cast<double>(rows * cols) * 4 / 1e9);  // 2 in, 1 out
}
BENCHMARK(BM_AddSameShape)
    ->ArgNames({"rows", "cols"})
    ->Args({256, 128})    // 128 KB per tensor: fits in L2
    ->Args({1024, 1024})  // 4 MB per tensor: 12 MB total, beyond L2
    ->Unit(benchmark::kMicrosecond);

void BM_AddBiasBroadcast(benchmark::State& state) {
    const Tensor x = random_tensor({256, 128}, 1);
    const Tensor bias = random_tensor({128}, 2);
    nawa_bench::LoopMeter meter;
    for (auto _ : state) {
        Tensor y = add(x, bias);  // {256,128} + {128}: the bias row is reused for every row
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    meter.report(state);
    state.counters["GBps"] = rate((2.0 * 256 * 128 + 128) * 4 / 1e9);
}
BENCHMARK(BM_AddBiasBroadcast)->Unit(benchmark::kMicrosecond);

void BM_Transpose(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const Tensor t = random_tensor({n, n}, 1);
    nawa_bench::LoopMeter meter;
    for (auto _ : state) {
        Tensor u = transpose(t);
        benchmark::DoNotOptimize(u.data());
        benchmark::ClobberMemory();
    }
    meter.report(state);
    state.counters["GBps"] = rate(2.0 * static_cast<double>(n * n) * 4 / 1e9);
}
BENCHMARK(BM_Transpose)->ArgName("n")->Arg(512)->Unit(benchmark::kMicrosecond);

void BM_Softmax(benchmark::State& state) {
    const Tensor x = random_tensor({256, 10}, 1);
    const Softmax softmax;
    nawa_bench::LoopMeter meter;
    for (auto _ : state) {
        Tensor y = softmax.forward(x);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    meter.report(state);
    state.counters["GBps"] = rate(2.0 * 256 * 10 * 4 / 1e9);
}
BENCHMARK(BM_Softmax)->Unit(benchmark::kMicrosecond);

}  // namespace
