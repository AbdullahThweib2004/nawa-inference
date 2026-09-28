// Benchmarks for the real MNIST model: each layer, preprocessing, and full predictions.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <vector>

#include "alloc_counter.hpp"
#include "bench_common.hpp"
#include "inference/model/model.hpp"
#include "inference/runtime/thread_pool.hpp"
#include "inference/runtime/workspace.hpp"
#include "inference/tensor/gemm.hpp"

using namespace inference;
using nawa_bench::random_pixels;
using nawa_bench::random_tensor;

namespace {

// Loaded once, on first use, and shared by every benchmark.
const Model& model() {
    static const Model m(nawa_bench::kModelPath);
    return m;
}

// ---------------------------------------------------------------------------
// Individual layers of the real model, at MNIST shapes
// ---------------------------------------------------------------------------

// Runs layer `index` of the model on a {batch, features} input.
void run_layer(benchmark::State& state, std::size_t index, std::size_t features) {
    const auto batch = static_cast<std::size_t>(state.range(0));
    const Layer& layer = *model().layers().at(index);
    const Tensor x = random_tensor({batch, features}, 3);
    nawa_bench::LoopMeter meter;
    for (auto _ : state) {
        Tensor y = layer.forward(x);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    meter.report(state);
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(batch));
    state.SetLabel(layer.name());
}

void BM_Layer0_Linear784x128(benchmark::State& s) { run_layer(s, 0, 784); }
void BM_Layer1_ReLU(benchmark::State& s) { run_layer(s, 1, 128); }
void BM_Layer2_Linear128x10(benchmark::State& s) { run_layer(s, 2, 128); }
void BM_Layer3_Softmax(benchmark::State& s) { run_layer(s, 3, 10); }
BENCHMARK(BM_Layer0_Linear784x128)
    ->ArgName("batch")
    ->Arg(1)
    ->Arg(256)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Layer1_ReLU)->ArgName("batch")->Arg(1)->Arg(256)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Layer2_Linear128x10)
    ->ArgName("batch")
    ->Arg(1)
    ->Arg(256)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Layer3_Softmax)->ArgName("batch")->Arg(1)->Arg(256)->Unit(benchmark::kMicrosecond);

void BM_Preprocess(benchmark::State& state) {
    const auto batch = static_cast<std::size_t>(state.range(0));
    const Tensor raw = random_pixels({batch, 784});
    nawa_bench::LoopMeter meter;
    for (auto _ : state) {
        Tensor x = model().preprocess(raw);
        benchmark::DoNotOptimize(x.data());
        benchmark::ClobberMemory();
    }
    meter.report(state);
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(batch));
}
BENCHMARK(BM_Preprocess)->ArgName("batch")->Arg(1)->Arg(256)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// Full predictions: throughput and heap allocations
// ---------------------------------------------------------------------------

void BM_Predict(benchmark::State& state) {
    const auto batch = static_cast<std::size_t>(state.range(0));
    const Tensor raw = random_pixels({batch, 784});
    const nawa_bench::AllocStats before = nawa_bench::alloc_snapshot();
    nawa_bench::LoopMeter meter;
    for (auto _ : state) {
        Tensor probs = model().predict(raw);
        benchmark::DoNotOptimize(probs.data());
        benchmark::ClobberMemory();
    }
    meter.report(state);
    const nawa_bench::AllocStats used = nawa_bench::alloc_snapshot() - before;
    const auto iters = static_cast<double>(state.iterations());
    // items_per_second = images per second.
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(batch));
    state.counters["allocs_per_predict"] = static_cast<double>(used.count) / iters;
    state.counters["alloc_KB_per_predict"] = static_cast<double>(used.bytes) / iters / 1024.0;
    if (meter.has_cycles()) {
        state.counters["cycles_per_image"] =
            static_cast<double>(meter.cycles()) / (iters * static_cast<double>(batch));
    }
}
BENCHMARK(BM_Predict)
    ->ArgName("batch")
    ->Arg(1)
    ->Arg(8)
    ->Arg(32)
    ->Arg(256)
    ->Unit(benchmark::kMicrosecond);

// The allocation-free path: predict(raw, workspace) with a workspace reused across calls.
void BM_PredictWorkspace(benchmark::State& state) {
    const auto batch = static_cast<std::size_t>(state.range(0));
    const Tensor raw = random_pixels({batch, 784});
    Workspace workspace;
    (void)model().predict(raw, workspace);  // let the buffers grow before measuring
    const nawa_bench::AllocStats before = nawa_bench::alloc_snapshot();
    nawa_bench::LoopMeter meter;
    for (auto _ : state) {
        const Tensor& probs = model().predict(raw, workspace);
        benchmark::DoNotOptimize(probs.data());
        benchmark::ClobberMemory();
    }
    meter.report(state);
    const nawa_bench::AllocStats used = nawa_bench::alloc_snapshot() - before;
    const auto iters = static_cast<double>(state.iterations());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(batch));
    state.counters["allocs_per_predict"] = static_cast<double>(used.count) / iters;
    if (meter.has_cycles()) {
        state.counters["cycles_per_image"] =
            static_cast<double>(meter.cycles()) / (iters * static_cast<double>(batch));
    }
}
BENCHMARK(BM_PredictWorkspace)
    ->ArgName("batch")
    ->Arg(1)
    ->Arg(8)
    ->Arg(32)
    ->Arg(256)
    ->Unit(benchmark::kMicrosecond);

// Batch-1 latency distribution, allocation-free path.
void BM_PredictLatencyWorkspace(benchmark::State& state) {
    using Clock = std::chrono::steady_clock;
    const Tensor raw = random_pixels({1, 784});
    Workspace workspace;
    (void)model().predict(raw, workspace);
    std::vector<double> micros;
    micros.reserve(1 << 20);
    for (auto _ : state) {
        const auto start = Clock::now();
        const Tensor& probs = model().predict(raw, workspace);
        benchmark::DoNotOptimize(probs.data());
        benchmark::ClobberMemory();
        const auto end = Clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        state.SetIterationTime(seconds);
        if (micros.size() < micros.capacity()) micros.push_back(seconds * 1e6);
    }
    if (micros.empty()) return;
    const auto percentile = [&](double p) {
        const auto k = static_cast<std::size_t>(p * static_cast<double>(micros.size() - 1));
        std::nth_element(micros.begin(), micros.begin() + static_cast<std::ptrdiff_t>(k),
                         micros.end());
        return micros[k];
    };
    state.counters["p50_us"] = percentile(0.50);
    state.counters["p90_us"] = percentile(0.90);
    state.counters["p99_us"] = percentile(0.99);
}
BENCHMARK(BM_PredictLatencyWorkspace)->UseManualTime()->Unit(benchmark::kMicrosecond);

// Batch-1 latency distribution. Each call is timed individually and reported through
// manual timing, so the percentiles describe single requests rather than averages.
void BM_PredictLatency(benchmark::State& state) {
    using Clock = std::chrono::steady_clock;
    const Tensor raw = random_pixels({1, 784});
    std::vector<double> micros;
    micros.reserve(1 << 20);  // reserved up front so push_back doesn't reallocate mid-run
    for (auto _ : state) {
        const auto start = Clock::now();
        Tensor probs = model().predict(raw);
        benchmark::DoNotOptimize(probs.data());
        benchmark::ClobberMemory();
        const auto end = Clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        state.SetIterationTime(seconds);
        if (micros.size() < micros.capacity()) micros.push_back(seconds * 1e6);
    }
    if (micros.empty()) return;
    const auto percentile = [&](double p) {
        const auto k = static_cast<std::size_t>(p * static_cast<double>(micros.size() - 1));
        std::nth_element(micros.begin(), micros.begin() + static_cast<std::ptrdiff_t>(k),
                         micros.end());
        return micros[k];
    };
    state.counters["p50_us"] = percentile(0.50);
    state.counters["p90_us"] = percentile(0.90);
    state.counters["p99_us"] = percentile(0.99);
}
BENCHMARK(BM_PredictLatency)->UseManualTime()->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// Multithreading (stage 9.5). WALL time (UseRealTime): CPU cycles summed over threads would
// hide the speedup. Args: threads, batch.
// ---------------------------------------------------------------------------

void BM_PredictThreads(benchmark::State& state) {
    const auto threads = static_cast<std::size_t>(state.range(0));
    const auto batch = static_cast<std::size_t>(state.range(1));
    set_num_threads(threads);
    const Tensor raw = random_pixels({batch, 784});
    Workspace workspace;
    (void)model().predict(raw, workspace);
    for (auto _ : state) {
        const Tensor& probs = model().predict(raw, workspace);
        benchmark::DoNotOptimize(probs.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(batch));
    set_num_threads(default_num_threads());
}
BENCHMARK(BM_PredictThreads)
    ->ArgNames({"threads", "batch"})
    ->ArgsProduct({{1, 2, 4, 8}, {1, 256}})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

void BM_PredictLatencyThreads(benchmark::State& state) {
    using Clock = std::chrono::steady_clock;
    const auto threads = static_cast<std::size_t>(state.range(0));
    set_num_threads(threads);
    const Tensor raw = random_pixels({1, 784});
    Workspace workspace;
    (void)model().predict(raw, workspace);
    std::vector<double> micros;
    micros.reserve(1 << 20);
    for (auto _ : state) {
        const auto start = Clock::now();
        const Tensor& probs = model().predict(raw, workspace);
        benchmark::DoNotOptimize(probs.data());
        benchmark::ClobberMemory();
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        state.SetIterationTime(seconds);
        if (micros.size() < micros.capacity()) micros.push_back(seconds * 1e6);
    }
    set_num_threads(default_num_threads());
    if (micros.empty()) return;
    const auto percentile = [&](double p) {
        const auto k = static_cast<std::size_t>(p * static_cast<double>(micros.size() - 1));
        std::nth_element(micros.begin(), micros.begin() + static_cast<std::ptrdiff_t>(k),
                         micros.end());
        return micros[k];
    };
    state.counters["p50_us"] = percentile(0.50);
    state.counters["p90_us"] = percentile(0.90);
    state.counters["p99_us"] = percentile(0.99);
}
BENCHMARK(BM_PredictLatencyThreads)
    ->ArgName("threads")
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// The GEMM alone (packed B), wall time, for scaling beyond the small MNIST layers.
void BM_GemmThreads(benchmark::State& state) {
    const auto threads = static_cast<std::size_t>(state.range(0));
    const auto M = static_cast<std::size_t>(state.range(1));
    const auto K = static_cast<std::size_t>(state.range(2));
    const auto N = static_cast<std::size_t>(state.range(3));
    set_num_threads(threads);
    const Tensor a = random_tensor({M, K}, 1);
    const Tensor b = random_tensor({K, N}, 2);
    const PackedMatrix packed = PackedMatrix::pack(b.data(), K, N);
    Tensor c({M, N});
    for (auto _ : state) {
        gemm(a.data(), M, packed, c.data());
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOPS"] = benchmark::Counter(2.0 * static_cast<double>(M * K * N) / 1e9,
                                                  benchmark::Counter::kIsIterationInvariantRate);
    set_num_threads(default_num_threads());
}
BENCHMARK(BM_GemmThreads)
    ->ArgNames({"threads", "M", "K", "N"})
    ->ArgsProduct({{1, 2, 4, 8}, {1024}, {1024}, {1024}})
    ->ArgsProduct({{1, 2, 4, 8}, {1, 4, 16, 64, 256}, {784}, {128}})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace
