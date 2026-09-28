#pragma once

#include <benchmark/benchmark.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "cycle_counter.hpp"
#include "inference/tensor/tensor.hpp"

namespace nawa_bench {

inline const std::string kModelPath = std::string(NAWA_SOURCE_DIR) + "/models/mnist_mlp.nawa";

// Tensor filled with deterministic values in [-1, 1). The values don't affect timing (as
// long as there are no denormals), but a fixed seed keeps every run identical.
inline inference::Tensor random_tensor(const inference::Shape& shape, std::uint32_t seed = 1) {
    std::vector<float> data(inference::numel_of(shape));
    std::uint32_t state = seed * 2654435761u + 1;
    for (float& v : data) {
        state = state * 1664525u + 1013904223u;  // LCG (Numerical Recipes constants)
        v = static_cast<float>(state >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
    }
    return inference::Tensor(shape, std::move(data));
}

// Tensor of raw "pixels" 0..255, like MNIST input before preprocessing.
inline inference::Tensor random_pixels(const inference::Shape& shape, std::uint32_t seed = 1) {
    inference::Tensor t = random_tensor(shape, seed);
    float* p = t.data();
    for (std::size_t i = 0; i < t.numel(); ++i) p[i] = (p[i] + 1.0f) * 127.5f;
    return t;
}

// Measures CPU cycles and wall time around a whole benchmark loop:
//
//     LoopMeter meter;            // starts measuring
//     for (auto _ : state) { ... }
//     meter.report(state);        // adds GHz and cycles_per_iter counters
//
// The effective GHz shows what clock the CPU actually ran at (turbo, thermal throttling);
// cycle-based counters stay comparable when it changes.
class LoopMeter {
public:
    LoopMeter() : start_(std::chrono::steady_clock::now()) { counter_.start(); }

    std::uint64_t cycles() const noexcept { return cycles_; }
    bool has_cycles() const noexcept { return cycles_ > 0; }

    void report(benchmark::State& state) {
        cycles_ = counter_.stop();
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
        if (!has_cycles() || state.iterations() == 0) return;  // counter unavailable
        state.counters["GHz"] = static_cast<double>(cycles_) / seconds / 1e9;
        state.counters["cycles_per_iter"] =
            static_cast<double>(cycles_) / static_cast<double>(state.iterations());
    }

private:
    CycleCounter counter_;
    std::chrono::steady_clock::time_point start_;
    std::uint64_t cycles_ = 0;
};

}  // namespace nawa_bench
