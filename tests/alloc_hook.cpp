// Counts heap allocations in the test binary, so tests can assert "no allocations here".
// Replacing the global operator new/delete affects the whole test program (GoogleTest too),
// which is fine: tests only look at the difference across a region they control.

#include "alloc_hook.hpp"

#if NAWA_ALLOC_HOOK

#include <atomic>
#include <cstdlib>
#include <new>

namespace {
std::atomic<std::size_t> g_allocations{0};

void* counted(std::size_t size) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size == 0 ? 1 : size)) return p;
    throw std::bad_alloc();
}

void* counted_aligned(std::size_t size, std::align_val_t align) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    const auto a = static_cast<std::size_t>(align);
    if (void* p = std::aligned_alloc(a, (size + a - 1) / a * a == 0 ? a : (size + a - 1) / a * a)) {
        return p;
    }
    throw std::bad_alloc();
}
}  // namespace

std::size_t test_allocation_count() noexcept {
    return g_allocations.load(std::memory_order_relaxed);
}

void* operator new(std::size_t size) { return counted(size); }
void* operator new[](std::size_t size) { return counted(size); }
void* operator new(std::size_t size, std::align_val_t a) { return counted_aligned(size, a); }
void* operator new[](std::size_t size, std::align_val_t a) { return counted_aligned(size, a); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

#else  // sanitized build: no replacement, nothing counted

std::size_t test_allocation_count() noexcept { return 0; }

#endif
