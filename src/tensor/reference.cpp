#include "inference/tensor/reference.hpp"

#include "matmul_shapes.hpp"

namespace inference {

Tensor matmul_naive(const Tensor& a, const Tensor& b) {
    const auto [M, K, N] = detail::check_matmul_shapes(a, b, "matmul_naive");

    Tensor out({M, N});
    const float* A = a.data();
    const float* B = b.data();
    float* C = out.data();

    // Naive i-j-k loop: C[i][j] = sum over k of A[i][k] * B[k][j].
    //
    // Two problems, measured in docs/performance.md (0.51 FLOP/cycle, 1.6% of peak):
    //  - The inner loop walks B down a COLUMN: B[k*N + j] and B[(k+1)*N + j] are N floats
    //    apart, so nearly every read of B needs a new cache line.
    //  - `acc += ...` is a serial dependency chain: each addition waits for the previous
    //    one (~4 cycles), and the compiler may not reorder float additions to break it.
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
    return out;
}

}  // namespace inference
