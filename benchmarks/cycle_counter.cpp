#include "cycle_counter.hpp"

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstring>
#endif

namespace nawa_bench {

#if defined(__linux__)

CycleCounter::CycleCounter() {
    perf_event_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_HARDWARE;
    attr.size = sizeof(attr);
    attr.config = PERF_COUNT_HW_CPU_CYCLES;
    attr.disabled = 1;        // start() enables it
    attr.exclude_kernel = 1;  // user space only: allowed without root at paranoid level 2
    attr.exclude_hv = 1;
    // pid 0 = this thread, cpu -1 = whichever CPU it runs on. glibc has no wrapper for
    // this system call, so it is invoked directly.
    fd_ = static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0));
}

CycleCounter::~CycleCounter() {
    if (fd_ >= 0) close(fd_);
}

void CycleCounter::start() noexcept {
    if (fd_ < 0) return;
    ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
}

std::uint64_t CycleCounter::stop() noexcept {
    if (fd_ < 0) return 0;
    ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
    std::uint64_t cycles = 0;
    if (read(fd_, &cycles, sizeof(cycles)) != static_cast<ssize_t>(sizeof(cycles))) return 0;
    return cycles;
}

#else  // not Linux: counter unavailable

CycleCounter::CycleCounter() = default;
CycleCounter::~CycleCounter() = default;
void CycleCounter::start() noexcept {}
std::uint64_t CycleCounter::stop() noexcept { return 0; }

#endif

}  // namespace nawa_bench
