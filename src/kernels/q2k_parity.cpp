// src/kernels/q2k_parity.cpp (moex) - Strata-X's Q2_K expert kernels against ggml's arithmetic, on the GPU.
//
//     build/q2k_parity
//
// 1. native_q2_k_f32 (decode MMVQ: activations quantized to Q8_1, then the pinned vec_dot_q2_K_q8_1) against a
//    float64 transcription of ggml's dequantize_row_q2_K times the float activations, at iq_parity's 2e-2 of the
//    row's absolute dot mass (the Q8_1 activation rounding is the only expected difference).
// 2. dequant_bf16 for type 10 (the prompt path) against the same transcription in float32, rounded to BF16 with
//    round-to-nearest-even: bitwise.
// 3. Negative control: the MMVQ result against a reference that drops the min term (the classic Q2_K mistake) must
//    FAIL the tolerance, or the check in (1) has no power.
// Weights are random bytes with sane fp16 d / dmin (normal values in +-[2^-10, 2^-5), as mmvq_multi_parity does).
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int QK = 256, BLOCK_BYTES = 84;

float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = std::ldexp((float) m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float) (m | 0x400), (int) e - 25);
    return s ? -v : v;
}
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
uint16_t bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);   // quiet NaN
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
// ggml dequantize_row_q2_K, one 256-element block, in T (float for the bitwise path, double for the reference)
template <typename T>
void dequant_block(const uint8_t* b, T* y, bool with_min = true) {
    const float d = h2f((uint16_t) (b[80] | (b[81] << 8))), dmin = h2f((uint16_t) (b[82] | (b[83] << 8)));
    const uint8_t* q = b + 16;
    int is = 0;
    for (int n = 0; n < QK; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            uint8_t sc = b[is++];
            float dl = d * (float) (sc & 0xF), ml = with_min ? dmin * (float) (sc >> 4) : 0.0f;
            for (int l = 0; l < 16; ++l) *y++ = (T) (dl * (float) ((int8_t) ((q[l] >> shift) & 3)) - ml);
            sc = b[is++];
            dl = d * (float) (sc & 0xF); ml = with_min ? dmin * (float) (sc >> 4) : 0.0f;
            for (int l = 0; l < 16; ++l) *y++ = (T) (dl * (float) ((int8_t) ((q[l + 16] >> shift) & 3)) - ml);
            shift += 2;
        }
        q += 32;
    }
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)

}  // namespace

int main() {
    std::mt19937 rng(20261007);
    const int n_in = 2048, n_out = 96, bpr = n_in / QK;
    std::vector<uint8_t> w((size_t) n_out * bpr * BLOCK_BYTES);
    for (auto& v : w) v = (uint8_t) rng();
    for (size_t blk = 0; blk < (size_t) n_out * bpr; ++blk) {
        uint8_t* b = w.data() + blk * BLOCK_BYTES;
        const uint16_t d = sane_half(rng), dm = sane_half(rng);
        std::memcpy(b + 80, &d, 2); std::memcpy(b + 82, &dm, 2);
    }
    int failures = 0;
    // native MMVQ refuses the legacy (null) stream: one explicit stream for every call below
    cudaStream_t stream = nullptr;
    CK(cudaStreamCreate(&stream));
    // ---- (2) prompt path: dequant_bf16 type 10, bitwise
    {
        std::vector<uint16_t> want((size_t) n_out * n_in), got(want.size());
        std::vector<float> row(n_in);
        for (int r = 0; r < n_out; ++r) {
            for (int k = 0; k < bpr; ++k) dequant_block<float>(w.data() + ((size_t) r * bpr + k) * BLOCK_BYTES, row.data() + k * QK);
            for (int i = 0; i < n_in; ++i) want[(size_t) r * n_in + i] = bf16_rne(row[i]);
        }
        void* dw; uint16_t* dout;
        CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dout, got.size() * 2));
        CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
        strata::kernels::dequant_bf16(10, dw, 0, n_out, n_in, dout, stream);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), dout, got.size() * 2, cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < got.size(); ++i) bad += got[i] != want[i];
        std::printf("dequant_bf16 Q2_K: %zu of %zu values differ from ggml's arithmetic (want 0)\n", bad, got.size());
        failures += bad != 0;
        CK(cudaFree(dw)); CK(cudaFree(dout));
    }
    // ---- (1) decode MMVQ vs float64, and (3) the negative control
    for (int ncols : {1, 2, 3}) {
        std::vector<float> x((size_t) n_in * ncols);
        // a non-zero mean: Q2_K's min term is dmin * m * sum(x) per sub-block, invisible when sum(x) ~ 0 - and then the
        // negative control (the min term dropped) could not be told from the right answer
        std::normal_distribution<float> nd(0.6f, 0.5f);
        for (auto& v : x) v = nd(rng);
        void *dw, *dq; float *dx, *dy;
        CK(cudaMalloc(&dw, w.size())); CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMalloc(&dq, (size_t) ncols * (n_in / 32) * 36)); CK(cudaMalloc(&dy, (size_t) ncols * n_out * 4));
        CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        strata::kernels::native_q2_k_f32(dw, dx, dq, dy, n_in, n_out, ncols, stream);
        CK(cudaDeviceSynchronize());
        std::vector<float> y((size_t) ncols * n_out);
        CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        // the control's WORST row: dmin's random sign and size make the min term cancel in some rows, so its best row
        // says nothing; the check has power when dropping the term is caught somewhere, far outside the tolerance
        double worst = 0, worst_ctl = 0;
        std::vector<double> rowd(n_in), rown(n_in);
        for (int c = 0; c < ncols; ++c)
            for (int r = 0; r < n_out; ++r) {
                for (int k = 0; k < bpr; ++k) {
                    const uint8_t* b = w.data() + ((size_t) r * bpr + k) * BLOCK_BYTES;
                    dequant_block<double>(b, rowd.data() + k * QK, true);
                    dequant_block<double>(b, rown.data() + k * QK, false);
                }
                double ref = 0, mass = 0, ctl = 0;
                for (int i = 0; i < n_in; ++i) {
                    const double xi = x[(size_t) c * n_in + i];
                    ref += rowd[i] * xi; mass += std::fabs(rowd[i] * xi); ctl += rown[i] * xi;
                }
                const double got = y[(size_t) c * n_out + r];
                worst = std::max(worst, std::fabs(got - ref) / std::max(mass, 1e-12));
                worst_ctl = std::max(worst_ctl, std::fabs(got - ctl) / std::max(mass, 1e-12));
            }
        const bool ok = worst <= 2e-2, ctl_fails = worst_ctl > 2e-2 && worst_ctl > 10 * worst;
        std::printf("native_q2_k_f32 ncols=%d: worst error %.2e of the row's dot mass (tolerance 2e-2) %s; "
                    "negative control (no min term) worst %.2e -> %s\n", ncols, worst, ok ? "OK" : "FAIL",
                    worst_ctl, ctl_fails ? "detected, the check has power" : "NOT detected");
        failures += !ok + !ctl_fails;
        CK(cudaFree(dw)); CK(cudaFree(dx)); CK(cudaFree(dq)); CK(cudaFree(dy));
    }
    std::printf("%s\n", failures ? "q2k_parity: FAILED" : "q2k_parity: all checks passed");
    return failures ? 1 : 0;
}
