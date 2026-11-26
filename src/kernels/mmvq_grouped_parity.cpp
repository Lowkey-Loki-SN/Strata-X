// src/kernels/mmvq_grouped_parity.cpp (moex) - the grouped MMVQ (one launch over every (token, expert) pair, the weights
// picked from a slot array by a device-side slot table) against one single-column native_mmvq call per pair: BITWISE.
//
//     build/mmvq_grouped_parity
//
// For each type: a slot array of S slots whose bytes are random blocks (finite scales), T tokens' activations quantized
// to Q8_1 by the engine's own quantizer, k pairs per token with random slots - some -1 (a miss: the pair must leave its
// output untouched).  Gate and up are two matrices of one slot, computed in one PAIR launch.
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
struct Ty {
    int type;
    const char* name;
    int elems, bytes;   // values and bytes per block
};
constexpr Ty kTypes[] = {{2, "Q4_0", 32, 18},   {6, "Q5_0", 32, 22}, {8, "Q8_0", 32, 34},    {10, "Q2_K", 256, 84},
                         {11, "Q3_K", 256, 110}, {12, "Q4_K", 256, 144}, {13, "Q5_K", 256, 176}, {14, "Q6_K", 256, 210},
                         {20, "IQ4_NL", 32, 18}, {23, "IQ4_XS", 256, 136}, {39, "MXFP4", 32, 17},
                         {16, "IQ2_XXS", 256, 66}, {17, "IQ2_XS", 256, 74}, {18, "IQ3_XXS", 256, 98},
                         {21, "IQ3_S", 256, 110}, {22, "IQ2_S", 256, 82}, {19, "IQ1_S", 256, 50},   // moex: IQ1_S (UD-IQ1_M)
                         {40, "NVFP4", 64, 36}};   // moex: NVFP4 (Hy4-flash's SPARK-4Bit)

uint16_t sane_half(std::mt19937& rng) {   // a normal half in +-[2^-10, 2^-5)
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::fprintf(stderr, "mmvq_grouped_parity: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}
}  // namespace

int main(int argc, char** argv) {
    namespace kn = strata::kernels;
    const int n_in = argc > 1 ? std::atoi(argv[1]) : 2048, n_out = 192, S = 6, T = 3, k = 4, P = T * k;   // (moex: N_IN)
    std::mt19937 rng(20261007);
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    int failures = 0;
    for (const Ty& ty : kTypes) {
        if (!kn::native_mmvq_grouped_supported(ty.type, n_in)) {
            std::printf("%-7s grouped launch: not supported at n_in %d (skipped)\n", ty.name, n_in);
            continue;
        }
        const size_t row_bytes = (size_t) (n_in / ty.elems) * ty.bytes, mat = row_bytes * n_out;
        const size_t slot_bytes = (2 * mat + 255) & ~(size_t) 255;   // gate | up
        std::vector<uint8_t> w(slot_bytes * S);
        for (auto& b : w) b = (uint8_t) rng();
        // sane scales: every block's fp16 scale fields hold normal values (the K-quants' d / dmin, Q8_0's d, ...) - patch
        // the first two bytes of each block and, for the K-quants, the scale halves wherever they live
        const size_t nblk = w.size() / ty.bytes;
        for (size_t i = 0; i < nblk; ++i) {
            uint8_t* b = w.data() + i * ty.bytes;
            if (ty.elems == 32 && ty.type != 39) { const uint16_t h = sane_half(rng); std::memcpy(b, &h, 2); }
            if (ty.type == 39) b[0] = (uint8_t) (118 + rng() % 12);
            if (ty.type == 40)   // four UE4M3 run scales, exponents 3..10
                for (int q = 0; q < 4; ++q) b[q] = (uint8_t) (((3 + rng() % 8) << 3) | (rng() & 7));
            if (ty.type == 10) { const uint16_t h1 = sane_half(rng), h2 = sane_half(rng); std::memcpy(b + 80, &h1, 2); std::memcpy(b + 82, &h2, 2); }
            if (ty.type == 11) { const uint16_t h = sane_half(rng); std::memcpy(b + 108, &h, 2); }
            if (ty.type == 12 || ty.type == 13) { const uint16_t h1 = sane_half(rng), h2 = sane_half(rng); std::memcpy(b, &h1, 2); std::memcpy(b + 2, &h2, 2); }
            if (ty.type == 14) { const uint16_t h = sane_half(rng); std::memcpy(b + 208, &h, 2); }
            if (ty.type == 23) { const uint16_t h = sane_half(rng); std::memcpy(b, &h, 2); }
            if (ty.type == 16 || ty.type == 17 || ty.type == 18 || ty.type == 19 || ty.type == 21 || ty.type == 22) {   // the i-quants: d first
                const uint16_t h = sane_half(rng);
                std::memcpy(b, &h, 2);
            }
        }
        std::normal_distribution<float> nd(0.3f, 0.7f);
        std::vector<float> x((size_t) T * n_in);
        for (float& v : x) v = nd(rng);
        std::vector<int32_t> slot(P);
        for (int i = 0; i < P; ++i) slot[i] = (rng() % 4 == 0) ? -1 : (int32_t) (rng() % S);
        uint8_t* dw = nullptr;
        float *dx = nullptr, *dyg = nullptr, *dyu = nullptr, *dref = nullptr;
        void* dxq = nullptr;
        int32_t* dslot = nullptr;
        if (!ck(cudaMalloc(&dw, w.size()), "malloc") || !ck(cudaMalloc(&dx, x.size() * 4), "malloc") ||
            !ck(cudaMalloc(&dxq, kn::native_q8_1_bytes(n_in * T)), "malloc") ||
            !ck(cudaMalloc(&dyg, (size_t) P * n_out * 4), "malloc") || !ck(cudaMalloc(&dyu, (size_t) P * n_out * 4), "malloc") ||
            !ck(cudaMalloc(&dref, (size_t) 2 * P * n_out * 4), "malloc") || !ck(cudaMalloc(&dslot, P * 4), "malloc"))
            return 1;
        cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dslot, slot.data(), P * 4, cudaMemcpyHostToDevice);
        cudaMemset(dyg, 0x7f, (size_t) P * n_out * 4);   // sentinel: a skipped pair must not touch its output
        cudaMemset(dyu, 0x7f, (size_t) P * n_out * 4);
        // Q8_1 columns, one per token, contiguous (the engine's layout)
        kn::native_quantize_q8_1(dx, dxq, n_in, T, s);
        kn::native_mmvq_grouped(ty.type, dw, slot_bytes, 0, mat, dslot, dxq, k, dyg, n_out, dyu, n_out, n_in, n_out, P, s);
        // the reference: one single-column call per pair and matrix
        const size_t xq_col = kn::native_q8_1_bytes(n_in);
        for (int p = 0; p < P; ++p) {
            if (slot[p] < 0) continue;
            const uint8_t* wg = dw + (size_t) slot[p] * slot_bytes;
            const uint8_t* xc = static_cast<const uint8_t*>(dxq) + (size_t) (p / k) * xq_col;
            kn::native_mmvq(ty.type, wg, xc, dref + (size_t) p * n_out, n_in, n_out, 1, s);
            kn::native_mmvq(ty.type, wg + mat, xc, dref + (size_t) (P + p) * n_out, n_in, n_out, 1, s);
        }
        if (!ck(cudaStreamSynchronize(s), "sync")) return 1;
        std::vector<float> g((size_t) P * n_out), u((size_t) P * n_out), r((size_t) 2 * P * n_out);
        cudaMemcpy(g.data(), dyg, g.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(u.data(), dyu, u.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(r.data(), dref, r.size() * 4, cudaMemcpyDeviceToHost);
        long diff = 0, touched = 0, finite_bad = 0;
        for (int p = 0; p < P; ++p)
            for (int o = 0; o < n_out; ++o) {
                const float gv = g[(size_t) p * n_out + o], uv = u[(size_t) p * n_out + o];
                if (slot[p] < 0) {
                    uint32_t a, b;
                    std::memcpy(&a, &gv, 4);
                    std::memcpy(&b, &uv, 4);
                    if (a != 0x7f7f7f7fu || b != 0x7f7f7f7fu) ++touched;
                    continue;
                }
                if (std::memcmp(&gv, &r[(size_t) p * n_out + o], 4) != 0) ++diff;
                if (std::memcmp(&uv, &r[(size_t) (P + p) * n_out + o], 4) != 0) ++diff;
                if (!(gv == gv) || !(uv == uv)) ++finite_bad;
            }
        const bool ok = diff == 0 && touched == 0 && finite_bad == 0;
        std::printf("%-7s grouped vs single-column: %ld values differ, %ld skipped outputs touched, %ld NaN: %s\n", ty.name,
                    diff, touched, finite_bad, ok ? "PASS" : "FAIL");
        failures += !ok;
        cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dyg); cudaFree(dyu); cudaFree(dref); cudaFree(dslot);
    }
    std::printf("mmvq_grouped_parity: %s\n", failures ? "FAILED" : "ALL PASS");
    return failures ? 1 : 0;
}
