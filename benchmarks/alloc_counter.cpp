// Replaces the global allocation functions to count heap allocations.
//
// C++ lets a program provide its own `operator new`/`operator delete`: the linker then uses
// these instead of the standard library's. Every `new`, every std::vector growth, every
// std::string that doesn't fit its small buffer ends up here. We count, then forward to
// malloc/free. This file is linked into nawa_bench only.

#include "alloc_counter.hpp"

#include <atomic>
#include <cstdlib>
#include <new>

namespace {

// Relaxed atomics: we only need correct totals, not ordering between threads. An uncontended
// relaxed increment costs a few nanoseconds, far less than the allocation itself.
std::atomic<std::size_t> g_count{0};
std::atomic<std::size_t> g_bytes{0};

void* counted_alloc(std::size_t size) {
    g_count.fetch_add(1, std::memory_order_relaxed);
    g_bytes.fetch_add(size, std::memory_order_relaxed);
    // malloc(0) may return nullptr; operator new must return a unique non-null pointer.
    if (void* p = std::malloc(size == 0 ? 1 : size)) return p;
    throw std::bad_alloc();
}

void* counted_aligned_alloc(std::size_t size, std::align_val_t align) {
    g_count.fetch_add(1, std::memory_order_relaxed);
    g_bytes.fetch_add(size, std::memory_order_relaxed);
    const auto a = static_cast<std::size_t>(align);
    // std::aligned_alloc requires the size to be a multiple of the alignment.
    const std::size_t rounded = (size + a - 1) / a * a;
    if (void* p = std::aligned_alloc(a, rounded == 0 ? a : rounded)) return p;
    throw std::bad_alloc();
}

}  // namespace

namespace nawa_bench {

AllocStats alloc_snapshot() noexcept {
    return {g_count.load(std::memory_order_relaxed), g_bytes.load(std::memory_order_relaxed)};
}

}  // namespace nawa_bench

// The replaceable global allocation functions. The nothrow and array forms of the standard
// library call these ordinary forms by default, so they are counted too.
void* operator new(std::size_t size) { return counted_alloc(size); }
void* operator new[](std::size_t size) { return counted_alloc(size); }
void* operator new(std::size_t size, std::align_val_t align) {
    return counted_aligned_alloc(size, align);
}
void* operator new[](std::size_t size, std::align_val_t align) {
    return counted_aligned_alloc(size, align);
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
