// src/prefill/mmq_dense_parity.cpp (moex) - llama.cpp's MMQ as a dense projection (strata::prefill::mmq, the ordinary
// mode: no expert bounds) against a float64 product of ggml's own dequantization, per weight type and column count.
//
//     build/mmq_dense_parity [rows cols]
//
// Column counts are chosen to hit both of MMQ's launch modes on the card: the stream-k kernel with its fixup (tile
// efficiency < 90%) and the same kernel launched as a plain tiling (efficiency >= 90%: one block per tile).  The
// tolerance is the Q8_1 activation rounding's: 2e-2 of each output's absolute dot mass.
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;

int main(int argc, char** argv) {
    const int64_t N = argc > 2 ? atoll(argv[1]) : 4096, K = argc > 2 ? atoll(argv[2]) : 4096;
    ggml_cpu_init();
    const ggml_type types[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K, GGML_TYPE_Q3_K,
                               GGML_TYPE_Q2_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S, GGML_TYPE_MXFP4,
                               GGML_TYPE_NVFP4};   // moex: Hy4-flash's SPARK-4Bit
    const int cols[] = {64, 128, 256, 384, 512, 768, 1024};
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> wf((size_t) (N * K));
    for (float& v : wf) v = nd(rng) * 0.02f;
    const int Tmax = 1024;
    std::vector<float> x((size_t) Tmax * K);
    for (float& v : x) v = nd(rng);
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    mmq::Context ctx;
    float *dx = nullptr, *dy = nullptr;
    void *dw = nullptr, *dxq = nullptr;
    cudaMalloc(&dx, x.size() * sizeof(float));
    cudaMalloc(&dy, (size_t) Tmax * N * sizeof(float));
    cudaMalloc(&dxq, mmq::q8_bytes(Tmax, K));
    cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    int bad_total = 0;
    for (ggml_type t : types) {
        if (!mmq::supported((int) t) || !mmq::fits((int) t, N)) {
            printf("%-8s not covered by this build's MMQ\n", ggml_type_name(t));
            continue;
        }
        if (ggml_get_type_traits_cpu(t)->from_float == nullptr && t != GGML_TYPE_IQ3_XXS && t != GGML_TYPE_IQ3_S &&
            t != GGML_TYPE_IQ2_S) {
            printf("%-8s no quantizer\n", ggml_type_name(t));
            continue;
        }
        // quantize the weights with ggml (an i-quant needs an importance matrix: uniform)
        const size_t rb = ggml_row_size(t, K);
        std::vector<uint8_t> wq((size_t) N * rb);
        std::vector<float> imat((size_t) K, 1.0f);
        if (ggml_quantize_requires_imatrix(t)) ggml_quantize_init(t);
        ggml_quantize_chunk(t, wf.data(), wq.data(), 0, N, K, ggml_quantize_requires_imatrix(t) ? imat.data() : nullptr);
        // the reference: ggml's dequantization, float64 dots
        std::vector<float> wd((size_t) (N * K));
        ggml_get_type_traits(t)->to_float(wq.data(), wd.data(), N * K);
        cudaFree(dw);
        cudaMalloc(&dw, wq.size());
        cudaMemcpy(dw, wq.data(), wq.size(), cudaMemcpyHostToDevice);
        printf("%-8s", ggml_type_name(t));
        for (int T : cols) {
            mmq::quantize(dx, nullptr, dxq, (int) t, K, K, T, s);
            mmq::Product p;
            p.w = dw;
            p.type = (int) t;
            p.w_rows = N;
            p.w_cols = K;
            p.expert_bytes = wq.size();
            p.n = 1;
            p.xq = dxq;
            p.bounds = nullptr;
            p.ids = nullptr;
            p.total_rows = T;
            p.max_rows = T;
            p.dst = dy;
            p.ld_dst = N;
            ctx.run(p, s);
            std::vector<float> y((size_t) T * N);
            cudaStreamSynchronize(s);
            cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost);
            double worst = 0;
            int nbad = 0;
            for (int c = 0; c < T; c += 7)
                for (int64_t r = 0; r < N; r += 5) {
                    double acc = 0, mass = 0;
                    for (int64_t k = 0; k < K; ++k) {
                        const double pr = (double) wd[(size_t) (r * K + k)] * x[(size_t) c * K + k];
                        acc += pr;
                        mass += std::fabs(pr);
                    }
                    const double e = std::fabs(y[(size_t) c * N + r] - acc) / std::max(mass, 1e-30);
                    worst = std::max(worst, e);
                    nbad += e > 2e-2;
                }
            printf("  T=%-4d %s %.1e", T, nbad ? "FAIL" : "ok  ", worst);
            bad_total += nbad > 0;
        }
        printf("\n");
    }
    printf("mmq_dense_parity (%lld x %lld): %s\n", (long long) N, (long long) K, bad_total ? "FAILURES" : "all ok");
    return bad_total ? 1 : 0;
}
