// moex/tools/hy4/indexer_switch_check.cpp - the engine's own hy4_scores_w + hy4_select_w (whatever STRATA_HY4_SCORES_TILE /
// STRATA_HY4_SCORES_MM / STRATA_HY4_SELECT_SAMPLED pick) on fixed data with ties, zero pools and a cluster of strong groups:
// every valid score and cell of 20 cases into argv[1], and the per-layer times.  Two runs under different switches compare by
// cmp (2026-11-19: the defaults = the row kernel + radix select, byte for byte, 637 MB).  Built by hand against the build tree:
// g++ -O2 -std=c++17 -Iinclude -I/usr/local/cuda/include moex/tools/hy4/indexer_switch_check.cpp
//     build-dsv41-dev/libstrata_dsv4_kernels.a -L/usr/local/cuda/lib64 -lcudart -lcublas -o indexer_switch_check
#include "strata/x/hy4_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
using namespace strata::x;
int main(int argc, char** argv) {
    const int kp = 4, n_top = 511, max_cells = 2048 + 4, ID = 128, IH = 16, TMAX = 512;
    const int64_t n_ctx = 1048576;
    const int max_pool = (int) (n_ctx / kp + 1);
    float *pooled, *iq, *w, *score; int32_t* cells; int64_t* pd;
    cudaMalloc(&pooled, (size_t) max_pool * ID * 4); cudaMalloc(&iq, (size_t) TMAX * IH * ID * 4); cudaMalloc(&w, TMAX * IH * 4);
    cudaMalloc(&score, (size_t) TMAX * max_pool * 4); cudaMalloc(&cells, (size_t) TMAX * max_cells * 4); cudaMalloc(&pd, 8);
    std::mt19937 rng(11); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> h((size_t) max_pool * ID);
    for (auto& v : h) v = nd(rng);
    for (int P = 997; P < max_pool; P += 997) std::copy(h.begin() + 5 * ID, h.begin() + 6 * ID, h.begin() + (size_t) P * ID);
    for (int P = 1009; P < max_pool; P += 1009) std::fill(h.begin() + (size_t) P * ID, h.begin() + (size_t) (P + 1) * ID, 0.0f);
    for (int P = 150000; P < 152000; ++P) for (int d = 0; d < ID; ++d) h[(size_t) P * ID + d] *= 3.0f;   // a cluster of strong groups
    cudaMemcpy(pooled, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    h.assign((size_t) TMAX * IH * ID, 0); for (auto& v : h) v = nd(rng); cudaMemcpy(iq, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    h.assign(TMAX * IH, 0); for (auto& v : h) v = 0.1f * nd(rng); cudaMemcpy(w, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t s; cudaStreamCreate(&s);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    FILE* f = fopen(argv[1], "wb");
    struct C { int T; int64_t p; bool dev; };
    const C cs[] = {{1, 5, false}, {1, 2045, false}, {1, 2049, true}, {5, 4000, false}, {1, 16387, false}, {6, 65533, true},
                    {7, 131070, false}, {8, 131072, true}, {1, 600001, false}, {3, 610002, true}, {4, 620003, false},
                    {1, 819200, true}, {2, 819201, false}, {6, 819198, true}, {8, 819200, false}, {37, 700001, false},
                    {512, 300000, false}, {512, 819200, false}, {1, 1048575, false}, {6, 1048569, true}};
    for (const C& c : cs) {
        cudaMemcpy(pd, &c.p, 8, cudaMemcpyHostToDevice);
        const int64_t* dp = c.dev ? pd : nullptr;
        cudaMemset(score, 0, (size_t) c.T * max_pool * 4);
        auto run = [&] { hy4_scores_w(iq, pooled, w, score, c.p, c.T, kp, max_pool, s, dp);
                         hy4_select_w(score, max_pool, n_top, kp, c.p, c.T, cells, max_cells, s, dp); };
        run();
        cudaStreamSynchronize(s);
        if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "CUDA error\n"); return 1; }
        std::vector<float> sc((size_t) c.T * max_pool); std::vector<int32_t> cl((size_t) c.T * max_cells);
        cudaMemcpy(sc.data(), score, sc.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(cl.data(), cells, cl.size() * 4, cudaMemcpyDeviceToHost);
        for (int t = 0; t < c.T; ++t) {
            const int64_t pos = c.p + t;
            fwrite(sc.data() + (size_t) t * max_pool, 4, (size_t) (pos / kp), f);
            fwrite(cl.data() + (size_t) t * max_cells, 4, (size_t) hy4_cells(pos, n_top, kp), f);
        }
        const int R = c.T >= 64 ? 2 : 10;
        cudaEventRecord(a, s); for (int r = 0; r < R; ++r) run(); cudaEventRecord(b, s); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        printf("T %3d p %7lld%s: indexer %.3f ms a layer\n", c.T, (long long) c.p, c.dev ? " (dev)" : "", ms / R);
    }
    fclose(f);
    return 0;
}
