// src/kernels/nvfp4_parity.cpp (moex) - Strata-X's NVFP4 decode MMVQ against ggml's own dequantizer, on the GPU.
//
//     build/nvfp4_parity
//
// Hy4-flash's SPARK-4Bit sets keep their routed experts (and the -Fast set most dense projections) as ggml's NVFP4:
// 64 E2M1 codes a block in four runs of 16, each run scaled by a UE4M3 byte.  At each width (1,536: Hy4-flash's experts,
// 24 blocks a row - fewer than one iteration of the kernel's threads; 3,072; 4,096):
// 1. native_mmvq(40) against a float64 dot of ggml's dequantize_row_nvfp4 (ggml-base's to_float) with the float
//    activations, at mxfp4_parity's 2e-2 of the row's absolute dot mass (the Q8_1 activation rounding is the only
//    expected difference).
// 2. The same with the activations as the kernel reads them (the device's Q8_1 blocks, d x q in float64): only the float
//    sums' order is left, so 1e-5 of the dot mass - a code, nibble or scale read wrong shows here.
// 3. ncols 2..8: every column BITWISE equal to the single-column call on that column (mmvq_multi_parity's rule).
// 4. Negative control: a reference with each byte's nibbles swapped must FAIL (2), or (2) has no power.
// Scales are random UE4M3 bytes 0x00..0x7E with exponents 3..10 (2^-5 .. 2^2 after ggml's 0.5) and, one block in 64,
// 0x7F (NaN: 0, as ggml-cpu reads it).
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kType = 40;   // GGML_TYPE_NVFP4
constexpr int QK = 64, BLOCK_BYTES = 36;

bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "nvfp4_parity: %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// the device's Q8_1 activations back as floats: a block is {half d, half s, int8 qs[32]}
std::vector<double> q8_1_values(const std::vector<uint8_t>& xq, int n) {
    std::vector<double> v((size_t) n);
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* blk = xq.data() + (size_t) b * 36;
        __half d;
        std::memcpy(&d, blk, 2);
        const double dd = (double) __half2float(d);
        for (int i = 0; i < 32; ++i) v[(size_t) b * 32 + i] = dd * (double) (int8_t) blk[4 + i];
    }
    return v;
}

int run(int n_in) {
    const int n_out = 640, max_cols = 8;
    const int bpr = n_in / QK;
    std::mt19937 rng(20261118u + (unsigned) n_in);
    std::vector<uint8_t> w((size_t) n_out * bpr * BLOCK_BYTES);
    for (size_t b = 0; b < (size_t) n_out * bpr; ++b) {
        uint8_t* blk = w.data() + b * BLOCK_BYTES;
        for (int k = 0; k < 4; ++k) blk[k] = rng() % 64 == 0 ? 0x7F : (uint8_t) (((3 + rng() % 8) << 3) | (rng() & 7));
        for (int i = 4; i < BLOCK_BYTES; ++i) blk[i] = (uint8_t) rng();
    }
    // the negative control's weights: every code byte's nibbles swapped (the run's values 0-7 and 8-15 exchanged)
    std::vector<uint8_t> w_swap(w);
    for (size_t b = 0; b < (size_t) n_out * bpr; ++b)
        for (int i = 4; i < BLOCK_BYTES; ++i) {
            uint8_t& c = w_swap[b * BLOCK_BYTES + (size_t) i];
            c = (uint8_t) ((c >> 4) | (c << 4));
        }
    std::normal_distribution<float> nd(0.6f, 0.5f);
    std::vector<float> x((size_t) n_in * max_cols);
    for (float& v : x) v = nd(rng);

    const ggml_type_traits* tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    std::vector<float> rows((size_t) n_out * n_in), rows_swap((size_t) n_out * n_in);
    for (int r = 0; r < n_out; ++r) {
        tr->to_float(w.data() + (size_t) r * bpr * BLOCK_BYTES, rows.data() + (size_t) r * n_in, n_in);
        tr->to_float(w_swap.data() + (size_t) r * bpr * BLOCK_BYTES, rows_swap.data() + (size_t) r * n_in, n_in);
    }

    cudaStream_t s = nullptr;
    void *dw = nullptr, *dxq = nullptr;
    float *dx = nullptr, *dy = nullptr;
    const size_t xq_col = strata::kernels::native_q8_1_bytes(n_in);
    const size_t xq_bytes = strata::kernels::native_q8_1_bytes(n_in * max_cols);
    if (!ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream") || !ck(cudaMalloc(&dw, w.size()), "malloc") ||
        !ck(cudaMalloc(&dx, x.size() * sizeof(float)), "malloc") || !ck(cudaMalloc(&dxq, xq_bytes), "malloc") ||
        !ck(cudaMalloc(&dy, (size_t) n_out * max_cols * sizeof(float)), "malloc"))
        return 1;
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 1, 2 and 4: single columns
    std::vector<float> single((size_t) n_out * max_cols);
    double worst_f = 0, worst_q = 0, ctl = 1e30;
    std::vector<uint8_t> xq(xq_col);
    for (int c = 0; c < max_cols; ++c) {
        strata::kernels::native_quantize_q8_1(dx + (size_t) c * n_in, dxq, n_in, 1, s);
        strata::kernels::native_mmvq(kType, dw, dxq, dy, n_in, n_out, 1, s);
        if (!ck(cudaStreamSynchronize(s), "single")) return 1;
        cudaMemcpy(single.data() + (size_t) c * n_out, dy, (size_t) n_out * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(xq.data(), dxq, xq_col, cudaMemcpyDeviceToHost);
        const std::vector<double> xa = q8_1_values(xq, n_in);
        double ctl_c = 0;
        for (int r = 0; r < n_out; ++r) {
            double acc_f = 0, acc_q = 0, acc_s = 0, mass = 0;
            for (int i = 0; i < n_in; ++i) {
                const double wv = rows[(size_t) r * n_in + i];
                acc_f += wv * x[(size_t) c * n_in + i];
                acc_q += wv * xa[(size_t) i];
                acc_s += (double) rows_swap[(size_t) r * n_in + i] * xa[(size_t) i];
                mass += std::fabs(wv * xa[(size_t) i]);
            }
            const double y = single[(size_t) c * n_out + r], m = std::max(mass, 1e-30);
            worst_f = std::max(worst_f, std::fabs(y - acc_f) / m);
            worst_q = std::max(worst_q, std::fabs(y - acc_q) / m);
            ctl_c = std::max(ctl_c, std::fabs(y - acc_s) / m);
        }
        ctl = std::min(ctl, ctl_c);
    }
    const bool ok1 = worst_f <= 2e-2, ok2 = worst_q <= 1e-5, ctl_fails = ctl > 1e-3;

    // 3. multi-column: bitwise against the single-column calls
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
    std::printf("nvfp4_parity n_in %d: vs ggml dequant x float activations: worst %.3g of the dot mass (tol 2e-2): %s\n", n_in,
                worst_f, ok1 ? "PASS" : "FAIL");
    std::printf("nvfp4_parity n_in %d: vs ggml dequant x the kernel's Q8_1 activations: worst %.3g (tol 1e-5): %s\n", n_in, worst_q,
                ok2 ? "PASS" : "FAIL");
    std::printf("nvfp4_parity n_in %d: negative control (nibbles swapped): best column's worst %.3g: %s\n", n_in, ctl,
                ctl_fails ? "detected" : "NOT detected");
    std::printf("nvfp4_parity n_in %d: ncols 2..8 vs single columns: %d values differ (bitwise): %s\n", n_in, diff,
                diff ? "FAIL" : "PASS");
    cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy); cudaStreamDestroy(s);
    return ok1 && ok2 && ctl_fails && diff == 0 ? 0 : 1;
}

}  // namespace

int main() {
    if (GGML_TYPE_NVFP4 != kType || ggml_type_size(GGML_TYPE_NVFP4) != BLOCK_BYTES || ggml_blck_size(GGML_TYPE_NVFP4) != QK) {
        std::fprintf(stderr, "nvfp4_parity: ggml's NVFP4 layout is not the one the kernel assumes\n");
        return 1;
    }
    if (!strata::kernels::native_mmvq_supported(kType) ||
        strata::kernels::native_mmvq_weight_bytes(kType, 1536, 640) != (size_t) 640 * 24 * BLOCK_BYTES) {
        std::fprintf(stderr, "nvfp4_parity: native MMVQ does not report NVFP4 (or its byte count is wrong)\n");
        return 1;
    }
    int failures = 0;
    for (int n_in : {1536, 3072, 4096}) failures += run(n_in);
    std::printf("nvfp4_parity: %s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
