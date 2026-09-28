// nawa_bench entry point: checks the build type, records machine and build information,
// measures memory, then runs the Google Benchmark suite.
//
//   ./build/bin/nawa_bench                                  # all benchmarks
//   ./build/bin/nawa_bench --benchmark_filter=BM_Matmul     # a subset
//
// Recording a result file (pinned to one core, 5 repetitions): see docs/performance.md.
// (No shell line continuations here: a backslash at the end of a // comment line would
// continue the comment onto the next line, which -Wcomment rightly rejects.)

#include <benchmark/benchmark.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__)
#include <sys/resource.h>
#endif

#include "alloc_counter.hpp"
#include "bench_common.hpp"
#include "inference/model/model.hpp"

namespace {

constexpr const char* kAllowFlag = "--nawa_allow_non_release";

// Peak resident set size of this process so far, in KiB (Linux reports ru_maxrss in KiB).
// It only ever grows, so read it at the moments you care about.
long peak_rss_kib() {
#if defined(__unix__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) return usage.ru_maxrss;
#endif
    return -1;
}

std::string cpu_model_name() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    for (std::string line; std::getline(cpuinfo, line);) {
        if (line.rfind("model name", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) return line.substr(colon + 2);
        }
    }
    return "unknown";
}

// What the CPU can do (runtime check) ...
std::string cpu_simd_support() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    std::string s;
    if (__builtin_cpu_supports("avx2")) s += "AVX2 ";
    if (__builtin_cpu_supports("fma")) s += "FMA ";
    if (__builtin_cpu_supports("avx512f")) s += "AVX-512F ";
    return s.empty() ? "none of AVX2/FMA/AVX-512" : s.substr(0, s.size() - 1);
#else
    return "unknown (non-x86 or non-GCC/Clang)";
#endif
}

// ... versus what this binary was compiled to use (compile-time macros). Without flags such
// as -march=native, the compiler targets baseline x86-64 and uses none of the wide SIMD units.
std::string compiled_simd() {
    std::string s;
#if defined(__AVX512F__)
    s += "AVX-512F ";
#endif
#if defined(__AVX2__)
    s += "AVX2 ";
#endif
#if defined(__FMA__)
    s += "FMA ";
#endif
    return s.empty() ? "baseline only (SSE2 on x86-64)" : s.substr(0, s.size() - 1);
}

bool is_release_build() {
#if defined(NDEBUG)
    return std::string(NAWA_BUILD_TYPE) == "Release";
#else
    return false;
#endif
}

void add_context(const std::string& key, const std::string& value) {
    benchmark::AddCustomContext(key, value);
    std::cout << "  " << key << ": " << value << '\n';
}

// Peak memory and heap allocations of a real predict call.
void memory_report() {
    using inference::Model;
    using inference::Tensor;
    std::cout << "Memory:\n";
    add_context("peak_rss_kib_at_start", std::to_string(peak_rss_kib()));

    const Model model(nawa_bench::kModelPath);
    add_context("peak_rss_kib_after_model_load", std::to_string(peak_rss_kib()));

    for (const std::size_t batch : {std::size_t{1}, std::size_t{256}}) {
        const Tensor raw = nawa_bench::random_pixels({batch, 784});
        (void)model.predict(raw);  // warm-up, so one-time effects don't count
        const nawa_bench::AllocStats before = nawa_bench::alloc_snapshot();
        const Tensor probs = model.predict(raw);
        const nawa_bench::AllocStats used = nawa_bench::alloc_snapshot() - before;
        benchmark::DoNotOptimize(probs.data());
        const std::string b = std::to_string(batch);
        add_context("heap_allocs_per_predict_batch" + b, std::to_string(used.count));
        add_context("heap_bytes_per_predict_batch" + b, std::to_string(used.bytes));
        if (batch == 256) {
            add_context("peak_rss_kib_after_predict_batch256", std::to_string(peak_rss_kib()));
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    // Handle (and remove) our own flag before Google Benchmark sees the arguments.
    bool allow_non_release = false;
    std::vector<char*> args;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], kAllowFlag) == 0) {
            allow_non_release = true;
        } else {
            args.push_back(argv[i]);
        }
    }
    int args_count = static_cast<int>(args.size());

    if (!is_release_build()) {
        std::cerr << "\n"
                     "*****************************************************************\n"
                     "*  WARNING: nawa_bench was built as '" NAWA_BUILD_TYPE
                     "', not Release.\n"
                     "*  Timings from unoptimized builds are meaningless for comparison.\n"
                     "*  Rebuild: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
                     "*                -DENABLE_BENCHMARKS=ON\n"
                     "*****************************************************************\n\n";
        if (!allow_non_release) {
            std::cerr << "Refusing to run. Pass " << kAllowFlag << " to run anyway.\n";
            return 1;
        }
    }

    std::cout << "Build:\n";
    add_context("nawa_build_type", NAWA_BUILD_TYPE);
    add_context("nawa_compiler", NAWA_COMPILER);
    add_context("nawa_cxx_flags", NAWA_CXX_FLAGS);
    add_context("nawa_compiled_simd", compiled_simd());
    std::cout << "Machine:\n";
    add_context("nawa_cpu_model", cpu_model_name());
    add_context("nawa_cpu_simd_support", cpu_simd_support());
    add_context("nawa_hardware_threads", std::to_string(std::thread::hardware_concurrency()));
    memory_report();
    std::cout << '\n';

    benchmark::Initialize(&args_count, args.data());
    if (benchmark::ReportUnrecognizedArguments(args_count, args.data())) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
