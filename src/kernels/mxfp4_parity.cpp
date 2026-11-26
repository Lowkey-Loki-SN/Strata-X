// src/kernels/mxfp4_parity.cpp (moex) - Strata-X's MXFP4 decode MMVQ against ggml's own dequantizer, on the GPU.
//
//     build/mxfp4_parity
//
// DeepSeek-V4-Flash (one layer's experts) and MiMo-V2.6-Flash (down projections of a few layers) keep native
// MXFP4 expert tensors, so the VRAM expert tier needs a decode kernel for them.
// 1. native_mmvq(39) (activations quantized to Q8_1, then vec_dot_mxfp4_q8_1's arithmetic) against a float64 dot of
//    ggml's dequantize_row_mxfp4 (ggml-base's to_float) with the float activations, at q2k_parity's 2e-2 of the
//    row's absolute dot mass (the Q8_1 activation rounding is the only expected difference).
// 2. ncols 2..8: every column BITWISE equal to the single-column call on that column (mmvq_multi_parity's rule).
// 3. Negative control: a reference without the E8M0 scale's 0.5 must FAIL the tolerance, or (1) has no power.
// Weights are random codes with exponents in [118, 131] (scales 2^-10 .. 2^3).
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kType = 39;   // GGML_TYPE_MXFP4
constexpr int QK = 32, BLOCK_BYTES = 17;

bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "mxfp4_parity: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

}  // namespace

int main() {
    if (GGML_TYPE_MXFP4 != kType || ggml_type_size(GGML_TYPE_MXFP4) != BLOCK_BYTES || ggml_blck_size(GGML_TYPE_MXFP4) != QK) {
        std::fprintf(stderr, "mxfp4_parity: ggml's MXFP4 layout is not the one the kernel assumes\n");
        return 1;
    }
    const int n_in = 4096, n_out = 640, max_cols = 8;
    const int bpr = n_in / QK;
    std::mt19937 rng(20261007);
    std::vector<uint8_t> w((size_t) n_out * bpr * BLOCK_BYTES);
    for (size_t b = 0; b < (size_t) n_out * bpr; ++b) {
        uint8_t* blk = w.data() + b * BLOCK_BYTES;
        blk[0] = (uint8_t) (118 + rng() % 14);
        for (int i = 1; i < BLOCK_BYTES; ++i) blk[i] = (uint8_t) rng();
    }
    std::normal_distribution<float> nd(0.6f, 0.5f);
    std::vector<float> x((size_t) n_in * max_cols);
    for (float& v : x) v = nd(rng);

    // the reference: ggml's dequantization, float64 dots
    const ggml_type_traits* tr = ggml_get_type_traits(GGML_TYPE_MXFP4);
    std::vector<float> row((size_t) n_in);
    std::vector<double> ref((size_t) n_out * max_cols), mass((size_t) n_out * max_cols), bad((size_t) n_out * max_cols);
    for (int r = 0; r < n_out; ++r) {
        tr->to_float(w.data() + (size_t) r * bpr * BLOCK_BYTES, row.data(), n_in);
        for (int c = 0; c < max_cols; ++c) {
            double acc = 0, m = 0;
            for (int i = 0; i < n_in; ++i) {
                const double p = (double) row[(size_t) i] * x[(size_t) c * n_in + i];
                acc += p;
                m += std::fabs(p);
            }
            ref[(size_t) c * n_out + r] = acc;
            mass[(size_t) c * n_out + r] = m;
            bad[(size_t) c * n_out + r] = 2.0 * acc;   // the scale without vec_dot's 0.5
        }
    }

    cudaStream_t s = nullptr;
    void *dw = nullptr, *dxq = nullptr;
    float *dx = nullptr, *dy = nullptr;
    const size_t xq_bytes = strata::kernels::native_q8_1_bytes(n_in * max_cols);
    if (!ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream") || !ck(cudaMalloc(&dw, w.size()), "malloc") ||
        !ck(cudaMalloc(&dx, x.size() * sizeof(float)), "malloc") || !ck(cudaMalloc(&dxq, xq_bytes), "malloc") ||
        !ck(cudaMalloc(&dy, (size_t) n_out * max_cols * sizeof(float)), "malloc"))
        return 1;
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (!strata::kernels::native_mmvq_supported(kType) ||
        strata::kernels::native_mmvq_weight_bytes(kType, n_in, n_out) != w.size()) {
        std::fprintf(stderr, "mxfp4_parity: native MMVQ does not report MXFP4 (or its byte count is wrong)\n");
        return 1;
    }

    // 1. single columns
    std::vector<float> single((size_t) n_out * max_cols);
    double worst = 0, worst_bad = 1e30;
    for (int c = 0; c < max_cols; ++c) {
        strata::kernels::native_quantize_q8_1(dx + (size_t) c * n_in, dxq, n_in, 1, s);
        strata::kernels::native_mmvq(kType, dw, dxq, dy, n_in, n_out, 1, s);
        if (!ck(cudaStreamSynchronize(s), "single")) return 1;
        cudaMemcpy(single.data() + (size_t) c * n_out, dy, (size_t) n_out * sizeof(float), cudaMemcpyDeviceToHost);
        for (int r = 0; r < n_out; ++r) {
            const size_t k = (size_t) c * n_out + r;
            const double e = std::fabs(single[k] - ref[k]) / std::max(mass[k], 1e-30);
            const double eb = std::fabs(single[k] - bad[k]) / std::max(mass[k], 1e-30);
            worst = std::max(worst, e);
            worst_bad = std::min(worst_bad, eb);
        }
    }
    // the negative control: the WORST row against the wrong reference must be far off
    double ctl = 0;
    for (int c = 0; c < max_cols; ++c)
        for (int r = 0; r < n_out; ++r) {
            const size_t k = (size_t) c * n_out + r;
            ctl = std::max(ctl, std::fabs(single[k] - bad[k]) / std::max(mass[k], 1e-30));
        }
    const bool ok1 = worst <= 2e-2;
    const bool ctl_fails = ctl > 2e-2 && ctl > 10 * worst;
    std::printf("mxfp4_parity: MMVQ vs ggml dequant (float64): worst %.3g of the row's dot mass (tol 2e-2): %s\n", worst,
                ok1 ? "PASS" : "FAIL");
    std::printf("mxfp4_parity: negative control (no 0.5): worst %.3g: %s\n", ctl, ctl_fails ? "detected" : "NOT detected");

    // 2. multi-column: bitwise against the single-column calls
    int diff = 0;
    for (int nc = 2; nc <= max_cols; ++nc) {
        strata::kernels::native_quantize_q8_1(dx, dxq, n_in, nc, s);
        strata::kernels::native_mmvq(kType, dw, dxq, dy, n_in, n_out, nc, s);
        std::vector<float> multi((size_t) n_out * nc);
        if (!ck(cudaStreamSynchronize(s), "multi")) return 1;
        cudaMemcpy(multi.data(), dy, multi.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t k = 0; k < multi.size(); ++k)
            if (std::memcmp(&multi[k], &single[k], sizeof(float)) != 0) ++diff;
    }
    std::printf("mxfp4_parity: ncols 2..8 vs single columns: %d values differ (bitwise): %s\n", diff, diff ? "FAIL" : "PASS");
    const bool pass = ok1 && ctl_fails && diff == 0;
    std::printf("mxfp4_parity: %s\n", pass ? "ALL PASS" : "FAILED");
    return pass ? 0 : 1;
}
