#pragma once

#include <cstddef>

// Sanitizers (ASan, TSan) replace operator new/delete themselves to track every allocation.
// A second replacement in the test binary would mix the two (memory from one allocator freed
// by the other), so the counting hook is only installed in non-sanitized builds.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define NAWA_ALLOC_HOOK 0
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define NAWA_ALLOC_HOOK 0
#endif
#endif
#ifndef NAWA_ALLOC_HOOK
#define NAWA_ALLOC_HOOK 1
#endif

// Number of heap allocations (operator new calls) since the test program started.
// Always 0 when NAWA_ALLOC_HOOK is 0 (tests that need it skip themselves).
std::size_t test_allocation_count() noexcept;
