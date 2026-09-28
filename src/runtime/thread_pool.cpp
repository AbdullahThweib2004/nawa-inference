#include "inference/runtime/thread_pool.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>

#if defined(__linux__)
#include <sched.h>
#endif

namespace inference {

namespace {

// Set in pool workers, so a nested parallel_for runs inline instead of waiting on itself.
thread_local bool t_in_worker = false;

}  // namespace

// ---------------------------------------------------------------------------
// ThreadPool
// ---------------------------------------------------------------------------

ThreadPool::ThreadPool(std::size_t num_threads) {
    const std::size_t workers = num_threads > 0 ? num_threads - 1 : 0;
    queue_.reserve(64);  // pointers only; enough for many concurrent callers
    next_job_.assign(workers, 0);
    workers_.reserve(workers);
    for (std::size_t i = 0; i < workers; ++i) {
        workers_.emplace_back([this, i] { worker_loop(i); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    work_cv_.notify_all();
    for (std::thread& t : workers_) t.join();
}

void ThreadPool::run_share(Job& job, std::size_t participant) noexcept {
    // Static round-robin: this participant runs chunks participant, participant + P, ...
    for (std::size_t c = participant; c < job.chunks; c += job.participants) {
        if (job.failed.load(std::memory_order_relaxed)) break;
        const std::size_t begin = c * job.grain;
        const std::size_t end = std::min(job.count, begin + job.grain);
        try {
            job.fn(job.ctx, begin, end);
        } catch (...) {
            std::lock_guard<std::mutex> lock(job.error_mutex);
            if (!job.error) job.error = std::current_exception();
            job.failed.store(true, std::memory_order_relaxed);
        }
    }
}

void ThreadPool::run(std::size_t count, std::size_t grain, ChunkFn fn, void* ctx) {
    if (count == 0) return;
    if (grain == 0) grain = 1;
    const std::size_t chunks = (count + grain - 1) / grain;

    // Serial: one chunk, no workers, or already inside a worker (nested call).
    if (chunks == 1 || workers_.empty() || t_in_worker) {
        for (std::size_t c = 0; c < chunks; ++c) {
            const std::size_t begin = c * grain;
            fn(ctx, begin, std::min(count, begin + grain));
        }
        return;
    }

    Job job(fn, ctx, count, grain, chunks, num_threads());
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(&job);
    }
    work_cv_.notify_all();

    run_share(job, 0);  // the caller is participant 0

    {
        // Every worker must acknowledge the job (even one with no chunks to run) before it
        // may leave the queue: then no worker can still hold a pointer to it.
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&] { return job.acks == workers_.size(); });
        // Workers take jobs in order, so every job before this one is fully acknowledged
        // too. Remove all of them (their callers are still inside run(), waiting to wake up).
        while (!queue_.empty() && queue_.front()->acks == workers_.size()) {
            queue_.erase(queue_.begin());
            ++base_;
        }
    }
    if (job.error) std::rethrow_exception(job.error);
}

void ThreadPool::worker_loop(std::size_t index) {
    t_in_worker = true;
    const std::size_t participant = index + 1;  // the caller of each job is participant 0
    for (;;) {
        // Idle workers sleep on the condition variable. (A spin-wait before sleeping was
        // measured in stage 9.5 and gave no gain on this CPU, so it was removed.)
        Job* job = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_cv_.wait(lock, [&] { return stop_ || next_job_[index] < base_ + queue_.size(); });
            if (next_job_[index] >= base_ + queue_.size()) return;  // stop_ and no work left
            job = queue_[next_job_[index] - base_];
            ++next_job_[index];
        }
        run_share(*job, participant);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++job->acks;
            if (job->acks == workers_.size()) done_cv_.notify_all();
        }
    }
}

// ---------------------------------------------------------------------------
// Topology and the default pool
// ---------------------------------------------------------------------------

namespace {

// Parses a Linux CPU list like "0-3,8,10-11".
std::vector<std::size_t> parse_cpu_list(const std::string& text) {
    std::vector<std::size_t> cpus;
    std::stringstream ss(text);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        const auto dash = part.find('-');
        try {
            if (dash == std::string::npos) {
                cpus.push_back(std::stoul(part));
            } else {
                const std::size_t lo = std::stoul(part.substr(0, dash));
                const std::size_t hi = std::stoul(part.substr(dash + 1));
                for (std::size_t c = lo; c <= hi; ++c) cpus.push_back(c);
            }
        } catch (const std::exception&) {
            return {};
        }
    }
    return cpus;
}

std::string read_first_line(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return line;
}

}  // namespace

CpuTopology detect_cpu_topology() {
    CpuTopology topo;
    topo.logical_cpus = std::max(1u, std::thread::hardware_concurrency());
    topo.physical_cores = topo.logical_cpus;
#if defined(__linux__)
    // CPUs this process may run on (respects taskset / cgroup CPU sets).
    std::vector<std::size_t> allowed;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (std::size_t c = 0; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &set)) allowed.push_back(c);
        }
    }
    if (allowed.empty()) return topo;
    topo.logical_cpus = allowed.size();

    // Physical cores: CPUs with the same thread_siblings_list share one core.
    std::set<std::string> cores;
    for (std::size_t c : allowed) {
        const std::string siblings = read_first_line(
            "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        cores.insert(siblings.empty() ? std::to_string(c) : siblings);
    }
    topo.physical_cores = cores.size();

    // Hybrid Intel CPUs expose separate PMUs for P-cores (cpu_core) and E-cores (cpu_atom).
    const std::string pcores = read_first_line("/sys/devices/cpu_core/cpus");
    if (!pcores.empty() && !read_first_line("/sys/devices/cpu_atom/cpus").empty()) {
        topo.hybrid = true;
        std::set<std::string> p_physical;
        for (std::size_t c : parse_cpu_list(pcores)) {
            if (std::find(allowed.begin(), allowed.end(), c) == allowed.end()) continue;
            p_physical.insert(read_first_line("/sys/devices/system/cpu/cpu" + std::to_string(c) +
                                              "/topology/thread_siblings_list"));
        }
        topo.performance_cores = p_physical.size();
    }
#endif
    return topo;
}

std::size_t default_num_threads() {
    if (const char* env = std::getenv("NAWA_NUM_THREADS")) {
        try {
            const unsigned long n = std::stoul(env);
            if (n >= 1) return n;
        } catch (const std::exception&) {
            // fall through to the automatic choice
        }
    }
    const CpuTopology topo = detect_cpu_topology();
    const std::size_t cores =
        topo.hybrid && topo.performance_cores > 0 ? topo.performance_cores : topo.physical_cores;
    return std::max<std::size_t>(1, std::min(cores, topo.logical_cpus));
}

std::string describe_thread_choice() {
    const CpuTopology topo = detect_cpu_topology();
    std::ostringstream os;
    os << default_num_threads() << " threads (";
    if (std::getenv("NAWA_NUM_THREADS")) os << "from NAWA_NUM_THREADS; ";
    os << topo.physical_cores << " physical cores, " << topo.logical_cpus << " logical CPUs";
    if (topo.hybrid) os << ", hybrid: " << topo.performance_cores << " P-cores used";
    os << ")";
    return os.str();
}

namespace {
std::unique_ptr<ThreadPool>& pool_slot() {
    static std::unique_ptr<ThreadPool> pool;
    return pool;
}
std::mutex& pool_mutex() {
    static std::mutex m;
    return m;
}
}  // namespace

ThreadPool& default_thread_pool() {
    std::lock_guard<std::mutex> lock(pool_mutex());
    auto& pool = pool_slot();
    if (!pool) pool = std::make_unique<ThreadPool>(default_num_threads());
    return *pool;
}

void set_num_threads(std::size_t n) {
    std::lock_guard<std::mutex> lock(pool_mutex());
    auto& pool = pool_slot();
    pool.reset();  // joins the old workers first
    pool = std::make_unique<ThreadPool>(std::max<std::size_t>(1, n));
}

}  // namespace inference
