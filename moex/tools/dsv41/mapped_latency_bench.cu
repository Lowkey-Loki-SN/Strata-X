// moex/tools/dsv41/mapped_latency_bench.cu - DSV41_BACKLOG #106: what a GPU's reads of mapped host memory cost while the same
// GPU's link carries an expert copy.  A decode layer's doorbell wait (the GPU polling a word the CPU pool writes) and the
// combine (the GPU reading the pool's partial sums) are such reads; the adaptive tier's swap copies (13 MiB records, ~2 ms over
// GPU 0's x8) run under the next window's layers.  Per GPU: one thread's chain of dependent 4-byte reads of mapped memory, alone
// and with host-to-device copies of 12,812,288 bytes running back to back on another stream; then 64 KB reads (a block's
// 256 threads, 64 x 4 KB rows - a combine's size) the same two ways.
//
//   nvcc -O3 -std=c++17 -arch=sm_86 moex/tools/dsv41/mapped_latency_bench.cu -o build-dsv41/mapped_latency_bench
//   moex_gpus_free && CUDA_DEVICE_ORDER=PCI_BUS_ID build-dsv41/mapped_latency_bench
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define CK(x)                                                                                    \
    do {                                                                                         \
        const cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "mapped_latency_bench: %s: %s\n", #x, cudaGetErrorString(e_));  \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

// a chain of n dependent reads: each read's value is the next index (the host fills a permutation cycle), so no read starts
// before the previous one returned
__global__ void chain_kernel(const volatile uint32_t* h, int n, unsigned long long* cycles, uint32_t* sink) {
    uint32_t i = 0;
    const long long t0 = clock64();
    for (int k = 0; k < n; ++k) i = h[i];
    *cycles = (unsigned long long) (clock64() - t0);
    *sink = i;
}

// one block reading `rows` rows of 4 KB (each thread 16 bytes a row), the combine's pattern
__global__ void rows_kernel(const uint4* h, int rows, int reps, unsigned long long* cycles, uint4* sink) {
    uint4 acc = {0, 0, 0, 0};
    const long long t0 = clock64();
    for (int r = 0; r < reps; ++r)
        for (int row = 0; row < rows; ++row) {
            const uint4 v = h[(size_t) row * 256 + threadIdx.x + (size_t) r * 0];
            acc.x ^= v.x, acc.y ^= v.y, acc.z ^= v.z, acc.w ^= v.w;
        }
    __syncthreads();
    if (threadIdx.x == 0) *cycles = (unsigned long long) (clock64() - t0);
    sink[threadIdx.x] = acc;
}

int main() {
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    const size_t kRec = 12812288;
    const int n_chain = 4000, rows = 16, reps = 64;
    for (int d = 0; d < ndev && d < 2; ++d) {
        CK(cudaSetDevice(d));
        int clk_khz = 0;
        CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, d));
        uint32_t* h = nullptr;
        CK(cudaHostAlloc((void**) &h, 1 << 20, cudaHostAllocMapped | cudaHostAllocPortable));
        for (uint32_t k = 0; k < (1u << 18); ++k) h[k] = (k * 9973u + 1u) % (1u << 18);   // (a long cycle over 1 MB)
        uint32_t* hd = nullptr;
        CK(cudaHostGetDevicePointer((void**) &hd, h, 0));
        uint8_t* src = nullptr;
        uint8_t* dst = nullptr;
        CK(cudaHostAlloc((void**) &src, kRec, cudaHostAllocPortable));
        CK(cudaMalloc(&dst, kRec));
        unsigned long long* cyc = nullptr;
        uint32_t* sink = nullptr;
        uint4* sink4 = nullptr;
        CK(cudaMalloc(&cyc, sizeof(unsigned long long)));
        CK(cudaMalloc(&sink, sizeof(uint32_t)));
        CK(cudaMalloc(&sink4, 256 * sizeof(uint4)));
        cudaStream_t sk, sc;
        CK(cudaStreamCreateWithFlags(&sk, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking));
        auto run = [&](bool copies, double& chain_us, double& rows_us) {
            std::atomic<bool> stop{false};
            std::thread cp;
            if (copies) {
                cp = std::thread([&] {
                    CK(cudaSetDevice(d));
                    while (!stop.load()) {
                        CK(cudaMemcpyAsync(dst, src, kRec, cudaMemcpyHostToDevice, sc));
                        CK(cudaStreamSynchronize(sc));
                    }
                });
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            unsigned long long c = 0;
            double best_chain = 1e30, best_rows = 1e30, sum_chain = 0, sum_rows = 0;
            const int iters = 10;
            for (int it = 0; it < iters; ++it) {
                chain_kernel<<<1, 1, 0, sk>>>(hd, n_chain, cyc, sink);
                CK(cudaMemcpyAsync(&c, cyc, sizeof c, cudaMemcpyDeviceToHost, sk));
                CK(cudaStreamSynchronize(sk));
                const double us = (double) c / (clk_khz * 1e-3) / n_chain;
                sum_chain += us;
                best_chain = std::min(best_chain, us);
                rows_kernel<<<1, 256, 0, sk>>>((const uint4*) hd, rows, reps, cyc, sink4);
                CK(cudaMemcpyAsync(&c, cyc, sizeof c, cudaMemcpyDeviceToHost, sk));
                CK(cudaStreamSynchronize(sk));
                const double ur = (double) c / (clk_khz * 1e-3) / reps;
                sum_rows += ur;
                best_rows = std::min(best_rows, ur);
            }
            stop = true;
            if (cp.joinable()) cp.join();
            chain_us = sum_chain / iters;
            rows_us = sum_rows / iters;
        };
        double a0, b0, a1, b1;
        run(false, a0, b0);
        run(true, a1, b1);
        std::printf("GPU %d: a mapped 4-byte read %.2f us alone, %.2f us with copies on its link; 64 KB of mapped rows %.1f us alone, "
                    "%.1f us with copies\n", d, a0, a1, b0, b1);
        CK(cudaFreeHost(h));
        CK(cudaFreeHost(src));
        CK(cudaFree(dst));
    }
    return 0;
}
