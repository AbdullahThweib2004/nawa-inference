#pragma once

#include <cstdint>

// Counts CPU cycles spent in user space by this thread, using Linux perf events.
//
// Why: a laptop CPU's clock moves between ~1.5 and 4.8 GHz depending on turbo budget and
// temperature, so wall-clock time alone mixes "the code got faster" with "the CPU was
// cooler". Cycles are (mostly) independent of the clock, so FLOP per cycle and cycles per
// image stay comparable between runs.
//
// Needs /proc/sys/kernel/perf_event_paranoid <= 2 (the default on most distributions allows
// counting your own process's user-space cycles). If the counter is unavailable (other OS,
// container, stricter setting), valid() is false and benchmarks omit the cycle counters.
namespace nawa_bench {

class CycleCounter {
public:
    CycleCounter();
    ~CycleCounter();
    CycleCounter(const CycleCounter&) = delete;
    CycleCounter& operator=(const CycleCounter&) = delete;

    bool valid() const noexcept { return fd_ >= 0; }
    void start() noexcept;          // reset to 0 and start counting
    std::uint64_t stop() noexcept;  // stop and return the cycles counted since start()

private:
    int fd_ = -1;
};

}  // namespace nawa_bench
