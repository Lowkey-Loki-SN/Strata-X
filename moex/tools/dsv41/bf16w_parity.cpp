// moex (DSV41_PLAN §62): STRATA_DSV41_BF16_W's kernels against the F32 ones they replace, byte for byte.  gemv_bf16_w on BF16
// weights against gemv_f32_w on the F32 copy of the same values, at DeepSeek-V4.1's shapes (the router 5,120 x 384, the
// compressor 5,120 x 512, the index keys 512 x 128, the indexer's q_b 1,280 x 4,096 and its weights 5,120 x 32), 1 to 8 columns,
// many draws; and bf16_widen against the widening on the host.  The two forms read STRATA_DSV41_PRELOAD once: run it with and
// without it.
//   bf16w_parity [draws]          PASS / FAIL and the first mismatch
#include "strata/x/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace g = strata::x;

#define CK(x)                                                                                              \
    do {                                                                                                   \
        const cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_));                                 \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

int main(int argc, char** argv) {
    const int draws = argc > 1 ? std::atoi(argv[1]) : 20;
    std::mt19937 rng(20261030);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int shapes[][2] = {{5120, 384}, {5120, 512}, {512, 128}, {1280, 4096}, {5120, 32}};
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    long cases = 0, bad = 0;
    for (const auto& sh : shapes) {
        const int n_in = sh[0], n_out = sh[1];
        const size_t nw = (size_t) n_in * n_out;
        std::vector<uint16_t> wb(nw);
        std::vector<float> wf(nw);
        uint16_t* dwb;
        float *dwf, *dx, *dy0, *dy1, *dwide;
        CK(cudaMalloc(&dwb, nw * 2));
        CK(cudaMalloc(&dwf, nw * 4));
        CK(cudaMalloc(&dwide, nw * 4));
        CK(cudaMalloc(&dx, (size_t) 8 * n_in * 4));
        CK(cudaMalloc(&dy0, (size_t) 8 * n_out * 4));
        CK(cudaMalloc(&dy1, (size_t) 8 * n_out * 4));
        for (int d = 0; d < draws; ++d) {
            const float sc = d % 3 == 0 ? 0.02f : (d % 3 == 1 ? 1.0f : 30.0f);
            for (size_t i = 0; i < nw; ++i) {
                const float v = nd(rng) * sc;
                uint32_t b;
                std::memcpy(&b, &v, 4);
                wb[i] = (uint16_t) (b >> 16);
                const uint32_t up = (uint32_t) wb[i] << 16;
                std::memcpy(&wf[i], &up, 4);
            }
            std::vector<float> x((size_t) 8 * n_in);
            for (auto& v : x) v = nd(rng);
            CK(cudaMemcpy(dwb, wb.data(), nw * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dwf, wf.data(), nw * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            g::bf16_widen(dwb, dwide, nw, s);
            CK(cudaStreamSynchronize(s));
            std::vector<float> wide(nw);
            CK(cudaMemcpy(wide.data(), dwide, nw * 4, cudaMemcpyDeviceToHost));
            ++cases;
            if (std::memcmp(wide.data(), wf.data(), nw * 4) != 0) {
                if (!bad) std::fprintf(stderr, "FAIL: bf16_widen differs from the host's widening (%d x %d)\n", n_in, n_out);
                ++bad;
            }
            for (int T = 1; T <= 8; ++T) {
                g::gemv_f32_w(dwf, dx, dy0, n_in, n_out, T, s);
                g::gemv_bf16_w(dwb, dx, dy1, n_in, n_out, T, s);
                CK(cudaStreamSynchronize(s));
                std::vector<float> y0((size_t) T * n_out), y1((size_t) T * n_out);
                CK(cudaMemcpy(y0.data(), dy0, y0.size() * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost));
                ++cases;
                if (std::memcmp(y0.data(), y1.data(), y0.size() * 4) != 0) {
                    if (!bad)
                        for (size_t i = 0; i < y0.size(); ++i)
                            if (std::memcmp(&y0[i], &y1[i], 4) != 0) {
                                std::fprintf(stderr, "FAIL: %d x %d, T %d, draw %d: y[%zu] %.9g (F32) against %.9g (BF16)\n", n_in,
                                             n_out, T, d, i, y0[i], y1[i]);
                                break;
                            }
                    ++bad;
                }
            }
        }
        cudaFree(dwb), cudaFree(dwf), cudaFree(dwide), cudaFree(dx), cudaFree(dy0), cudaFree(dy1);
    }
    const char* pre = std::getenv("STRATA_DSV41_PRELOAD");
    std::printf("%s: %ld of %ld cases byte for byte (STRATA_DSV41_PRELOAD=%s)\n", bad ? "FAIL" : "PASS", cases - bad, cases,
                pre ? pre : "unset");
    return bad ? 1 : 0;
}
