#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "inference/tensor/gemm.hpp"

// INT8 post-training quantization (step 9.6).
//
// Weights: symmetric, PER OUTPUT CHANNEL. For each output j,
//     scale_w[j] = max_k |W[k][j]| / 127,   q_w[k][j] = round(W[k][j] / scale_w[j]) in [-127, 127]
// Activations: symmetric, PER ROW, computed at runtime for each input row m:
//     scale_a[m] = max_k |x[m][k]| / 127,   q_a[m][k] = round(x[m][k] / scale_a[m])
// Products are accumulated EXACTLY in int32, then converted back once per output:
//     y[m][j] = float(sum_k q_a[m][k] * q_w[k][j]) * (scale_a[m] * scale_w[j])  (+ bias) (ReLU)
//
// Only 8-bit weights are stored (4x smaller than float32), and the multiply-adds are small
// integer operations. Accuracy is lost only in the two roundings to 8 bits.
namespace inference {

// Per-channel int8 weights. Stored as {out_features, K_pad}: each output channel's weights
// are contiguous (so a dot product reads one contiguous run), zero-padded from in_features to
// K_pad, a multiple of 16 (one SIMD load of int8 values).
class QuantizedMatrix {
public:
    static constexpr std::size_t kPad = 16;

    QuantizedMatrix() = default;

    // Quantizes the row-major {K, N} float matrix (the Linear weight layout).
    // Throws std::invalid_argument if K is so large that int32 sums could overflow.
    static QuantizedMatrix quantize(const float* w, std::size_t K, std::size_t N);

    // Builds one from already-quantized values: `q` is {N, K} row-major (as stored in a
    // .nawa file), `scales` has N positive values.
    static QuantizedMatrix from_quantized(const std::int8_t* q, const float* scales, std::size_t K,
                                          std::size_t N);

    std::size_t in_features() const noexcept { return K_; }
    std::size_t out_features() const noexcept { return N_; }
    std::size_t padded_in() const noexcept { return K_pad_; }
    const std::int8_t* data() const noexcept { return q_.data(); }  // {N, K_pad}
    const float* scales() const noexcept { return scales_.data(); }
    std::int8_t at(std::size_t k, std::size_t j) const noexcept { return q_[j * K_pad_ + k]; }
    float dequantized(std::size_t k, std::size_t j) const noexcept {
        return static_cast<float>(at(k, j)) * scales_[j];
    }
    std::size_t size_bytes() const noexcept { return N_ * K_ + N_ * sizeof(float); }

private:
    std::size_t K_ = 0;
    std::size_t N_ = 0;
    std::size_t K_pad_ = 0;
    std::vector<std::int8_t> q_;
    std::vector<float> scales_;
};

// Quantizes one row of K floats symmetrically to 8-bit values (-127..127, zero-padded to K_pad)
// and returns its scale. The values are STORED as int16: the AVX2 kernel multiplies int16
// pairs (vpmaddwd), and storing the row widened once saves re-widening it for every output
// channel. The same function is used by every kernel, so they all see the same integers.
float quantize_row(const float* x, std::size_t K, std::size_t K_pad, std::int16_t* out);

// C {M, N} = dequantized(quantize_rows(A {M, K}) · W) (+ epilogue). Uses the kernel B would
// use for float GEMM (resolve_kernel: AVX2 when available, NAWA_KERNEL=portable forces the
// scalar reference), and the thread pool for large products. The integer sums are exact, so
// every kernel and every thread count gives bit-identical results.
void gemm_int8(const float* a, std::size_t M, const QuantizedMatrix& w, float* c,
               const GemmEpilogue& epilogue = {}, GemmKernel kernel = GemmKernel::Auto);

}  // namespace inference
