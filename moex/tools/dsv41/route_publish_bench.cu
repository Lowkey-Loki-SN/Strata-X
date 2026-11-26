// moex/tools/dsv41/route_publish_bench.cu - DSV41_BACKLOG #112: a decode window's routing and the doorbell's publish as the
// decode graph runs them, two launches (dsv4_route, doorbell_publish) against one (dsv4_route_publish, STRATA_DSV41_ROUTE_PUBLISH=1):
// DeepSeek-V4.1's 384 experts top-6, the 5,120 activations of each of T tokens to mapped host memory, each way captured in a
// CUDA graph of 40 such layers (a token's MoE layers) with a small kernel between them (the shared expert's place), the graph
// replayed 200 times - its time a replay; then both ways' ids, weights and published bytes compared.
//
//   nvcc -O3 -std=c++17 -arch=sm_86 -Iinclude moex/tools/dsv41/route_publish_bench.cu build-dsv41-dev/libstrata_dsv4_kernels.a \
//        build-dsv41-dev/libstrata_x_kernels.a -o build-dsv41-dev/route_publish_bench
//   moex_gpus_free && CUDA_DEVICE_ORDER=PCI_BUS_ID build-dsv41-dev/route_publish_bench
#include "strata/x/dsv4_kernels.hpp"
#include "strata/x/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        const cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "route_publish_bench: %s: %s\n", #x, cudaGetErrorString(e_));   \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

namespace g = strata::x;

__global__ void spacer(float* x, int n) {   // a layer's other work, so the launches do not run back to back alone
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 0.999f + 0.001f;
}

int main() {
    const int E = 384, k = 6, n = 5120, layers = 40, reps = 200;
    for (int dev = 0; dev < 2; ++dev) {
        if (cudaSetDevice(dev) != cudaSuccess) break;
        for (int T : {1, 8}) {
            std::mt19937 rng(112);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float> lg((size_t) T * E), bias(E), x((size_t) T * n);
            for (auto& v : lg) v = nd(rng);
            for (auto& v : bias) v = 0.1f * nd(rng);
            for (auto& v : x) v = nd(rng);
            float *d_lg, *d_bias, *d_x, *d_w, *d_sp;
            int32_t* d_ids;
            uint32_t* d_cnt;
            CK(cudaMalloc(&d_lg, lg.size() * 4));
            CK(cudaMalloc(&d_bias, E * 4));
            CK(cudaMalloc(&d_x, x.size() * 4));
            CK(cudaMalloc(&d_w, (size_t) T * k * 4));
            CK(cudaMalloc(&d_ids, (size_t) T * k * 4));
            CK(cudaMalloc(&d_cnt, 4));
            CK(cudaMalloc(&d_sp, 4096 * 4));
            CK(cudaMemcpy(d_lg, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_bias, bias.data(), E * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemset(d_cnt, 0, 4));
            CK(cudaMemset(d_sp, 0, 4096 * 4));
            // the mapped block: ids, activations, the word (a separate block per way, compared at the end)
            int32_t* h_ids[2];
            float* h_x[2];
            uint32_t* h_seq[2];
            for (int w = 0; w < 2; ++w) {
                CK(cudaHostAlloc((void**) &h_ids[w], (size_t) T * k * 4, cudaHostAllocMapped));
                CK(cudaHostAlloc((void**) &h_x[w], x.size() * 4, cudaHostAllocMapped));
                CK(cudaHostAlloc((void**) &h_seq[w], 64, cudaHostAllocMapped));
            }
            cudaStream_t s;
            CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
            float ms[2];
            std::vector<int32_t> ids_out[2];
            std::vector<float> w_out[2];
            for (int way = 0; way < 2; ++way) {
                int32_t* di;
                float* dx;
                uint32_t* ds;
                CK(cudaHostGetDevicePointer((void**) &di, h_ids[way], 0));
                CK(cudaHostGetDevicePointer((void**) &dx, h_x[way], 0));
                CK(cudaHostGetDevicePointer((void**) &ds, h_seq[way], 0));
                cudaGraph_t gr;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                for (int l = 0; l < layers; ++l) {
                    spacer<<<16, 256, 0, s>>>(d_sp, 4096);
                    if (way == 0) {
                        g::dsv4_route(d_lg, d_bias, nullptr, E, k, 2.5f, d_ids, d_w, T, s, nullptr, true);
                        g::doorbell_publish(d_ids, T * k, d_x, T * n, di, dx, ds, d_cnt, s);
                    } else {
                        g::dsv4_route_publish(d_lg, d_bias, nullptr, E, k, 2.5f, d_ids, d_w, T, s, nullptr, true, d_x, T * n, di, dx, ds,
                                              d_cnt);
                    }
                }
                CK(cudaStreamEndCapture(s, &gr));
                CK(cudaGraphInstantiate(&ge, gr, 0));
                CK(cudaGraphLaunch(ge, s));   // warm
                CK(cudaStreamSynchronize(s));
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a));
                CK(cudaEventCreate(&b));
                CK(cudaEventRecord(a, s));
                for (int r = 0; r < reps; ++r) CK(cudaGraphLaunch(ge, s));
                CK(cudaEventRecord(b, s));
                CK(cudaEventSynchronize(b));
                CK(cudaEventElapsedTime(&ms[way], a, b));
                ms[way] /= reps;
                ids_out[way].resize((size_t) T * k);
                w_out[way].resize((size_t) T * k);
                CK(cudaMemcpy(ids_out[way].data(), d_ids, (size_t) T * k * 4, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(w_out[way].data(), d_w, (size_t) T * k * 4, cudaMemcpyDeviceToHost));
                CK(cudaGraphExecDestroy(ge));
                CK(cudaGraphDestroy(gr));
            }
            const bool same = ids_out[0] == ids_out[1] && std::memcmp(w_out[0].data(), w_out[1].data(), w_out[0].size() * 4) == 0 &&
                              std::memcmp(h_ids[0], h_ids[1], (size_t) T * k * 4) == 0 &&
                              std::memcmp(h_x[0], h_x[1], x.size() * 4) == 0;
            std::printf("GPU %d, windows of %d: 40 layers' routing + publish %.1f us as two launches, %.1f us as one (%.2f us a layer); "
                        "ids, weights and the published bytes %s\n", dev, T, ms[0] * 1000.0f, ms[1] * 1000.0f,
                        (ms[0] - ms[1]) * 1000.0f / layers, same ? "the same" : "DIFFER");
            for (int w = 0; w < 2; ++w) {
                cudaFreeHost(h_ids[w]);
                cudaFreeHost(h_x[w]);
                cudaFreeHost(h_seq[w]);
            }
            cudaFree(d_lg);
            cudaFree(d_bias);
            cudaFree(d_x);
            cudaFree(d_w);
            cudaFree(d_ids);
            cudaFree(d_cnt);
            cudaFree(d_sp);
            cudaStreamDestroy(s);
        }
    }
    return 0;
}
