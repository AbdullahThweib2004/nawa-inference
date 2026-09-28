#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace inference {

// A fixed set of worker threads, created once and reused for every parallel operation.
// Creating threads costs tens of microseconds, far more than a whole batch-1 prediction, so
// threads are never created per call.
//
//  - parallel_for(count, grain, fn) splits [0, count) into chunks of `grain` and runs
//    fn(begin, end) on them in parallel. The calling thread participates.
//  - Chunks are assigned STATICALLY: chunk c always runs on participant c % num_threads()
//    (the caller is participant 0). The same thread therefore works on the same data every
//    call, which keeps it in that core's private L2 cache. The assignment never changes which
//    operations compute an output, so results don't depend on the number of threads.
//  - If fn throws, the remaining chunks are skipped and the first exception is rethrown in
//    the caller, after every thread has stopped touching the job.
//  - Called from inside a pool worker (nested parallelism), parallel_for runs serially
//    instead of deadlocking.
//  - Several threads may call parallel_for concurrently; their jobs queue up and run in order.
//  - The destructor finishes queued work and joins all workers.
//  - A steady-state parallel_for makes no heap allocations (jobs live on the caller's stack;
//    the queue holds pointers in preallocated storage).
class ThreadPool {
public:
    // num_threads counts the calling thread, so num_threads - 1 workers are started.
    // 0 is treated as 1 (no workers: everything runs on the caller).
    explicit ThreadPool(std::size_t num_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::size_t num_threads() const noexcept { return workers_.size() + 1; }

    template <typename Fn>
    void parallel_for(std::size_t count, std::size_t grain, Fn&& fn) {
        // Type-erased without std::function (which may allocate): a plain function pointer
        // plus a pointer to the caller's callable, which outlives the call.
        run(
            count, grain,
            [](void* ctx, std::size_t begin, std::size_t end) {
                (*static_cast<std::remove_reference_t<Fn>*>(ctx))(begin, end);
            },
            static_cast<void*>(&fn));
    }

private:
    using ChunkFn = void (*)(void*, std::size_t, std::size_t);

    struct Job {
        Job(ChunkFn f, void* c, std::size_t n, std::size_t g, std::size_t ch, std::size_t p)
            : fn(f), ctx(c), count(n), grain(g), chunks(ch), participants(p) {}

        ChunkFn fn;
        void* ctx;
        std::size_t count;
        std::size_t grain;
        std::size_t chunks;
        std::size_t participants;         // num_threads(): caller = 0, worker i = i + 1
        std::atomic<bool> failed{false};  // set on the first exception; later chunks skip
        std::exception_ptr error;         // written once, under error_mutex
        std::mutex error_mutex;
        std::size_t acks = 0;  // workers that have finished with this job (mutex_)
    };

    void run(std::size_t count, std::size_t grain, ChunkFn fn, void* ctx);
    void run_share(Job& job, std::size_t participant) noexcept;
    void worker_loop(std::size_t index);

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_cv_;  // workers wait here for jobs
    std::condition_variable done_cv_;  // callers wait here for their job to finish
    // Pending jobs in FIFO order; queue_[0] is job number base_. Each points into the stack of
    // a caller that is still waiting inside run(): a caller removes its job (and any finished
    // jobs before it) before returning, so no pointer in the queue ever dangles.
    std::vector<Job*> queue_;
    std::size_t base_ = 0;
    std::vector<std::size_t> next_job_;  // per worker: number of the next job it will take
    bool stop_ = false;
};

// Physical CPU layout, from Linux sysfs (with fallbacks elsewhere).
struct CpuTopology {
    std::size_t logical_cpus = 1;       // hardware threads the OS lets this process use
    std::size_t physical_cores = 1;     // distinct cores among them (SMT siblings counted once)
    std::size_t performance_cores = 0;  // hybrid CPUs only: P-cores (0 if not hybrid)
    bool hybrid = false;                // P-cores and E-cores (e.g. Intel Alder Lake and later)
};
CpuTopology detect_cpu_topology();

// Default thread count: NAWA_NUM_THREADS if set, else the physical P-cores on a hybrid CPU,
// else the physical cores (hyperthread siblings share one core's FMA units, so they add
// little to compute-bound GEMM). Never more than the CPUs this process may run on.
std::size_t default_num_threads();

// Human-readable explanation of the default choice, e.g. "4 threads (4 physical cores,
// 8 logical CPUs)".
std::string describe_thread_choice();

// The process-wide pool used by gemm(), created on first use with default_num_threads().
ThreadPool& default_thread_pool();

// Replaces the default pool with one of n threads (benchmarks and tests). Must not be called
// while another thread is using the pool.
void set_num_threads(std::size_t n);

}  // namespace inference
