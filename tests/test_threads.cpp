// Stage 9.5: the thread pool, and bit-identical results for any number of threads.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "inference/model/model.hpp"
#include "inference/model/tensor_io.hpp"
#include "inference/runtime/thread_pool.hpp"
#include "inference/runtime/workspace.hpp"
#include "inference/tensor/gemm.hpp"

using namespace inference;

namespace {

Tensor random_tensor(const Shape& shape, std::uint32_t seed) {
    std::vector<float> data(numel_of(shape));
    std::uint32_t state = seed * 2654435761u + 3u;
    for (float& v : data) {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>(state >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
    }
    return Tensor(shape, std::move(data));
}

bool bit_equal(const Tensor& a, const Tensor& b) {
    return a.shape() == b.shape() &&
           std::memcmp(a.data(), b.data(), a.numel() * sizeof(float)) == 0;
}

// Restores the default pool size when a test changes it.
struct DefaultThreadsGuard {
    ~DefaultThreadsGuard() { set_num_threads(default_num_threads()); }
};

}  // namespace

// ---------------------------------------------------------------------------
// ThreadPool
// ---------------------------------------------------------------------------

TEST(ThreadPool, EveryIndexExactlyOnce) {
    for (std::size_t threads : {1u, 2u, 3u, 4u, 8u}) {
        ThreadPool pool(threads);
        EXPECT_EQ(pool.num_threads(), threads);
        for (std::size_t count : {0u, 1u, 7u, 100u, 1001u}) {
            for (std::size_t grain : {1u, 3u, 64u, 5000u}) {
                std::vector<std::atomic<int>> hits(count);
                pool.parallel_for(count, grain, [&](std::size_t begin, std::size_t end) {
                    ASSERT_LE(end - begin, grain);
                    for (std::size_t i = begin; i < end; ++i) hits[i].fetch_add(1);
                });
                for (std::size_t i = 0; i < count; ++i) {
                    ASSERT_EQ(hits[i].load(), 1) << threads << " threads, count " << count
                                                 << ", grain " << grain << ", index " << i;
                }
            }
        }
    }
}

TEST(ThreadPool, UsesSeveralThreads) {
    ThreadPool pool(4);
    std::vector<std::thread::id> ids(8);
    pool.parallel_for(8, 1, [&](std::size_t begin, std::size_t) {
        ids[begin] = std::this_thread::get_id();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    });
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(std::unique(ids.begin(), ids.end()) - ids.begin(), 4);  // static: 4 participants
}

TEST(ThreadPool, ExceptionReachesTheCaller) {
    ThreadPool pool(4);
    EXPECT_THROW(pool.parallel_for(100, 1,
                                   [](std::size_t begin, std::size_t) {
                                       if (begin == 37) throw std::runtime_error("chunk 37");
                                   }),
                 std::runtime_error);
    // The pool still works afterwards.
    std::atomic<int> sum{0};
    pool.parallel_for(10, 1, [&](std::size_t b, std::size_t) { sum += static_cast<int>(b); });
    EXPECT_EQ(sum.load(), 45);
}

TEST(ThreadPool, NestedParallelForRunsInline) {
    ThreadPool pool(4);
    std::atomic<int> inner{0};
    pool.parallel_for(8, 1, [&](std::size_t, std::size_t) {
        pool.parallel_for(10, 1, [&](std::size_t, std::size_t) { inner.fetch_add(1); });
    });
    EXPECT_EQ(inner.load(), 80);  // completes (no deadlock) and runs everything
}

TEST(ThreadPool, ConcurrentCallersShareOnePool) {
    ThreadPool pool(4);
    std::vector<std::thread> callers;
    std::atomic<long> total{0};
    for (int t = 0; t < 4; ++t) {
        callers.emplace_back([&] {
            for (int rep = 0; rep < 50; ++rep) {
                pool.parallel_for(64, 4, [&](std::size_t b, std::size_t e) {
                    total.fetch_add(static_cast<long>(e - b));
                });
            }
        });
    }
    for (auto& c : callers) c.join();
    EXPECT_EQ(total.load(), 4L * 50 * 64);
}

TEST(ThreadPool, CreateAndDestroyRepeatedly) {
    for (int i = 0; i < 20; ++i) {
        ThreadPool pool(3);
        std::atomic<int> n{0};
        pool.parallel_for(6, 1, [&](std::size_t, std::size_t) { n.fetch_add(1); });
        EXPECT_EQ(n.load(), 6);
    }  // each destructor joins its workers
}

TEST(ThreadPool, TopologyAndDefaults) {
    const CpuTopology topo = detect_cpu_topology();
    EXPECT_GE(topo.physical_cores, 1u);
    EXPECT_LE(topo.physical_cores, topo.logical_cpus);
    EXPECT_GE(default_num_threads(), 1u);
    std::cout << "[          ] " << describe_thread_choice() << "\n";
#if defined(__unix__)
    setenv("NAWA_NUM_THREADS", "3", 1);
    EXPECT_EQ(default_num_threads(), 3u);
    unsetenv("NAWA_NUM_THREADS");
#endif
}

// ---------------------------------------------------------------------------
// Determinism: bit-identical results for 1, 2, 4 and all threads
// ---------------------------------------------------------------------------

TEST(ThreadDeterminism, GemmIsBitIdenticalForAnyThreadCount) {
    DefaultThreadsGuard guard;
    const std::size_t all = std::max<std::size_t>(std::thread::hardware_concurrency(), 1);
    for (GemmKernel kernel : {GemmKernel::Portable, GemmKernel::Avx2}) {
        if (!kernel_available(kernel)) continue;
        for (const auto& [M, K, N] : std::vector<std::array<std::size_t, 3>>{
                 {1, 784, 128}, {1, 300, 70}, {256, 784, 128}, {97, 257, 530}, {5, 64, 200}}) {
            const Tensor a = random_tensor({M, K}, 1);
            const Tensor b = random_tensor({K, N}, 2);
            const Tensor bias = random_tensor({N}, 3);
            const PackedMatrix packed = PackedMatrix::pack(b.data(), K, N, kernel);
            Tensor reference({M, N});
            set_num_threads(1);
            gemm(a.data(), M, packed, reference.data(), GemmEpilogue{bias.data(), true});
            for (std::size_t threads : {2u, 3u, 4u, static_cast<unsigned>(all)}) {
                set_num_threads(threads);
                Tensor c({M, N});
                gemm(a.data(), M, packed, c.data(), GemmEpilogue{bias.data(), true});
                EXPECT_TRUE(bit_equal(c, reference)) << kernel_name(kernel) << " " << M << "x" << K
                                                     << "x" << N << ", " << threads << " threads";
            }
        }
    }
}

TEST(ThreadDeterminism, PredictIsBitIdenticalForAnyThreadCount) {
    DefaultThreadsGuard guard;
    const Model model(std::string(NAWA_MODELS_DIR) + "/mnist_mlp.nawa");
    const Tensor images =
        read_tensor_file(std::string(NAWA_FIXTURES_DIR) + "/mnist_test100_images.ntsr");
    const Tensor one({784}, std::vector<float>(images.data(), images.data() + 784));
    set_num_threads(1);
    const Tensor batch_ref = model.predict(images);
    const Tensor one_ref = model.predict(one);
    for (std::size_t threads : {2u, 4u, std::max(1u, std::thread::hardware_concurrency())}) {
        set_num_threads(threads);
        Workspace workspace;
        EXPECT_TRUE(bit_equal(model.predict(images, workspace), batch_ref)) << threads;
        EXPECT_TRUE(bit_equal(model.predict(one, workspace), one_ref)) << threads;
    }
}

TEST(ThreadDeterminism, ModelSharedAcrossThreadsWithOwnWorkspaces) {
    const Model model(std::string(NAWA_MODELS_DIR) + "/mnist_mlp.nawa");
    const Tensor images =
        read_tensor_file(std::string(NAWA_FIXTURES_DIR) + "/mnist_test100_images.ntsr");
    const Tensor expected = model.predict(images);
    std::vector<std::thread> users;
    std::atomic<int> mismatches{0};
    for (int t = 0; t < 4; ++t) {
        users.emplace_back([&] {
            Workspace workspace;  // one per thread; the model is shared
            for (int rep = 0; rep < 20; ++rep) {
                if (!bit_equal(model.predict(images, workspace), expected)) ++mismatches;
            }
        });
    }
    for (auto& u : users) u.join();
    EXPECT_EQ(mismatches.load(), 0);
}
