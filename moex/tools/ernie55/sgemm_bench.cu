// moex (ERNIE55_PLAN.md, the hill-climb): the prompt path's dense projections under the cards' power limit: cuBLAS SGEMM (W row-major n_out x n_in, x row-major
// T x n_in, as Stage::gemv_t calls it) against one BF16 tensor-core GEMM with FP32 accumulation (x3 = the exact 3-way BF16
// split of x), each timed over 1.5 s of back-to-back calls (the limiter's steady state), the SM clock sampled by NVML.
// Bits: cuBLAS's other GemmEx algorithms against the default at T 512, and the default's first 512 columns at T 8192.
//   nvcc -O3 -std=c++17 -arch=sm_86 moex/tools/ernie55/sgemm_bench.cu -lcublas -lnvidia-ml -o build-dsv41-dev/sgemm_bench
//   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 build-dsv41-dev/sgemm_bench
// 2026-11-25, RTX 3080 at 180 W: SGEMM 9.5-11.9 TFLOPS held (915-1,170 MHz), one BF16 GEMM 43-54 (3 of them 0.55-0.79x the
// SGEMM's time: STRATA_X_TC3's arithmetic); the 24 FP32 algorithms = the default's bits at T 512, other T other kernels
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

__global__ void to_bf16(const float* a, __nv_bfloat16* b, size_t n) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) b[i] = __float2bfloat16_rn(a[i]);
}

int main() {
    struct Shape { int n_in, n_out; const char* what; };
    const Shape shapes[] = {{4096, 1024, "q_a"}, {1024, 32768, "q_b HCA"}, {8192, 4096, "output_b"}, {4096, 2048, "shexp g/u"}};
    nvmlInit();
    nvmlDevice_t nd;
    {
        int dev = 0; CK(cudaGetDevice(&dev));
        char bus[64]; CK(cudaDeviceGetPCIBusId(bus, sizeof bus, dev));
        nvmlDeviceGetHandleByPciBusId(bus, &nd);
    }
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return 1;
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cublasSetStream(h, s);
    std::mt19937 rng(5);
    std::printf("%-10s %6s %6s %5s | %9s %6s %5s | %9s %6s %5s | %6s | %s\n", "shape", "n_in", "n_out", "T", "SGEMM us", "TFLOPS", "MHz",
                "BF16 us", "TFLOPS", "MHz", "3xBF16", "bits");
    for (const Shape& sh : shapes) {
        const size_t nw = (size_t) sh.n_in * sh.n_out;
        std::vector<float> hw(nw);
        for (auto& v : hw) { const int m = (int) (rng() % 16), e = (int) (rng() % 8); v = std::ldexp((8 + m % 8) * (m < 8 ? 1.f : -1.f), -12 - e); }
        std::vector<float> hx((size_t) 8192 * sh.n_in);
        for (auto& v : hx) v = (float) ((int) (rng() % 20001) - 10000) / 7777.0f;
        float *dw, *dx, *dy0, *dy1;
        __nv_bfloat16 *bw, *bx;
        CK(cudaMalloc(&dw, nw * 4));
        CK(cudaMalloc(&bw, nw * 2));
        CK(cudaMemcpy(dw, hw.data(), nw * 4, cudaMemcpyHostToDevice));
        to_bf16<<<1024, 256, 0, s>>>(dw, bw, nw);
        std::vector<float> y512;
        for (int T : {512, 1024, 2048, 8192}) {
            CK(cudaMalloc(&dx, (size_t) T * sh.n_in * 4));
            CK(cudaMalloc(&bx, (size_t) T * sh.n_in * 2));
            CK(cudaMalloc(&dy0, (size_t) T * sh.n_out * 4));
            CK(cudaMalloc(&dy1, (size_t) T * sh.n_out * 4));
            CK(cudaMemcpy(dx, hx.data(), (size_t) T * sh.n_in * 4, cudaMemcpyHostToDevice));
            to_bf16<<<1024, 256, 0, s>>>(dx, bx, (size_t) T * sh.n_in);
            const float one = 1.0f, zero = 0.0f;
            auto sg = [&](float* y) { return cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, sh.n_out, T, sh.n_in, &one, dw, sh.n_in, dx, sh.n_in, &zero, y, sh.n_out); };
            auto bf = [&](float* y) {
                return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, sh.n_out, T, sh.n_in, &one, bw, CUDA_R_16BF, sh.n_in, bx, CUDA_R_16BF, sh.n_in,
                                    &zero, y, CUDA_R_32F, sh.n_out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            };
            auto sustained = [&](auto&& f, float* y, unsigned& mhz) -> double {   // 0.5 s warm, then calls for 1.5 s; the mean
                if (f(y) != CUBLAS_STATUS_SUCCESS) return -1;
                auto t0 = std::chrono::steady_clock::now();
                while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.5) { f(y); CK(cudaStreamSynchronize(s)); }
                long n = 0;
                std::vector<unsigned> clk;
                t0 = std::chrono::steady_clock::now();
                double el = 0;
                while ((el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()) < 1.5) {
                    for (int k = 0; k < 4; ++k) f(y);
                    n += 4;
                    CK(cudaStreamSynchronize(s));
                    unsigned c = 0; nvmlDeviceGetClockInfo(nd, NVML_CLOCK_SM, &c); clk.push_back(c);
                }
                std::sort(clk.begin(), clk.end());
                mhz = clk.empty() ? 0 : clk[clk.size() / 2];
                return el / n * 1e6;
            };
            const double fl = 2.0 * sh.n_in * sh.n_out * (double) T;
            unsigned m0 = 0, m1 = 0;
            const double t_sg = sustained(sg, dy0, m0);
            const double t_bf = sustained(bf, dy1, m1);
            std::vector<float> y0((size_t) T * sh.n_out);
            CK(cudaMemcpy(y0.data(), dy0, y0.size() * 4, cudaMemcpyDeviceToHost));
            char bits[160] = "";
            if (T == 512) {
                y512 = y0;
                int same = 0, ok = 0;
                std::vector<float> y1(y0.size());
                for (int algo = 0; algo <= 23; ++algo) {
                    if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, sh.n_out, T, sh.n_in, &one, dw, CUDA_R_32F, sh.n_in, dx, CUDA_R_32F, sh.n_in, &zero,
                                     dy1, CUDA_R_32F, sh.n_out, CUBLAS_COMPUTE_32F, (cublasGemmAlgo_t) algo) != CUBLAS_STATUS_SUCCESS) continue;
                    CK(cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost));
                    ++ok;
                    same += std::memcmp(y0.data(), y1.data(), y0.size() * 4) == 0;
                }
                std::snprintf(bits, sizeof bits, "%d of %d FP32 algos = the default's bits", same, ok);
            } else {
                std::snprintf(bits, sizeof bits, "first 512 columns %s T 512's", std::memcmp(y512.data(), y0.data(), y512.size() * 4) == 0 ? "=" : "DIFFER from");
            }
            std::printf("%-10s %6d %6d %5d | %9.1f %6.2f %5u | %9.1f %6.2f %5u | %5.2fx | %s\n", sh.what, sh.n_in, sh.n_out, T, t_sg,
                        fl / (t_sg * 1e-6) / 1e12, m0, t_bf, fl / (t_bf * 1e-6) / 1e12, m1, 3 * t_bf / t_sg, bits);
            CK(cudaFree(dx)); CK(cudaFree(bx)); CK(cudaFree(dy0)); CK(cudaFree(dy1));
        }
        CK(cudaFree(dw)); CK(cudaFree(bw));
    }
    nvmlShutdown();
    return 0;
}
