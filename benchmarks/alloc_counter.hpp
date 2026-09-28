#pragma once

#include <cstddef>

// Heap-allocation counting for the benchmark binary ONLY (alloc_counter.cpp replaces the
// global operator new/delete; the library and tests are unaffected).
namespace nawa_bench {

struct AllocStats {
    std::size_t count = 0;  // number of operator new calls
    std::size_t bytes = 0;  // total bytes requested
};

// Totals since program start. Take two snapshots and subtract to measure a region.
AllocStats alloc_snapshot() noexcept;

inline AllocStats operator-(const AllocStats& a, const AllocStats& b) noexcept {
    return {a.count - b.count, a.bytes - b.bytes};
}

}  // namespace nawa_bench
