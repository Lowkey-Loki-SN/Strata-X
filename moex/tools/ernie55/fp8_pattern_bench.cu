// moex (ERNIE55_PLAN.md, the hill-climb): the FP8 warp GEMV's read pattern against plain streaming reads, one card - is ~500 GB/s
// the pattern's limit or the math's?  2026-11-25, RTX 3080: the pattern alone (8 B a lane, or 16 B over half a warp) 670-705
// GB/s on the big shapes, a linear read 701-716, the kernel 470-560: the decode math and the loads per byte, not the pattern.
//   nvcc -O3 -std=c++17 -arch=sm_86 -Iinclude moex/tools/ernie55/fp8_pattern_bench.cu build-dsv41-dev/libstrata_x_kernels.a \
//        -o build-dsv41-dev/fp8_pattern_bench
//   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 build-dsv41-dev/fp8_pattern_bench
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "strata/x/kernels.hpp"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

__global__ void k_linear(const uint4* __restrict__ p, size_t n16, unsigned* out) {   // grid-stride, 16 B a lane
    unsigned a = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n16; i += (size_t) gridDim.x * blockDim.x) {
        const uint4 v = __ldg(p + i);
        a ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (a == 0x9E3779B9u) out[0] = a;
}

// the warp kernel's pattern: R rows a warp, lane l reads 8 B at 8 l + 256 k of each row
template <int R, int U>
__global__ void __launch_bounds__(256) k_rows8(const uint8_t* W, int n_in, int n_out, unsigned* out) {
    const int lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * R;
    if (r0 >= n_out) return;
    const uint8_t* wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wr[r] = W + (size_t) min(r0 + r, n_out - 1) * n_in;
    unsigned a = 0;
#pragma unroll U
    for (int i0 = 8 * lane; i0 < n_in; i0 += 256) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint2 q = __ldg(reinterpret_cast<const uint2*>(wr[r] + i0));
            a ^= q.x ^ q.y;
        }
    }
    if (a == 0x9E3779B9u) out[0] = a;
}

// half a warp a row: lane l of a half reads 16 B at 16 l + 256 k (the same 256 B a step a row, LDG.128)
template <int R, int U>
__global__ void __launch_bounds__(256) k_rows16(const uint8_t* W, int n_in, int n_out, unsigned* out) {
    const int lane = threadIdx.x & 31, hl = lane & 15;
    const int r0 = ((blockIdx.x * 8 + (threadIdx.x >> 5)) * 2 + (lane >> 4)) * R;
    if (r0 >= n_out) return;
    const uint8_t* wr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) wr[r] = W + (size_t) min(r0 + r, n_out - 1) * n_in;
    unsigned a = 0;
#pragma unroll U
    for (int i0 = 16 * hl; i0 < n_in; i0 += 256) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 q = __ldg(reinterpret_cast<const uint4*>(wr[r] + i0));
            a ^= q.x ^ q.y ^ q.z ^ q.w;
        }
    }
    if (a == 0x9E3779B9u) out[0] = a;
}

int main() {
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, 0));
    std::printf("%s: %d SMs, L2 %d KB, bus %d bit\n", pr.name, pr.multiProcessorCount, pr.l2CacheSize >> 10,
                pr.memoryBusWidth);
    const size_t ring = 512u << 20;   // a ring of copies far past the L2
    uint8_t* buf;
    CK(cudaMalloc(&buf, ring + (64u << 20)));
    unsigned* out;
    float *x, *y;
    CK(cudaMalloc(&out, 64));
    CK(cudaMalloc(&x, 65536 * 4));
    CK(cudaMalloc(&y, 65536 * 4));
    {
        std::vector<float> hx(65536);
        for (size_t i = 0; i < hx.size(); ++i) hx[i] = (float) ((int) (i * 2654435761u % 2001) - 1000) / 1000.0f;
        CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    }
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    {   // clocks up
        const auto t0 = clock();
        while ((double) (clock() - t0) / CLOCKS_PER_SEC < 1.5) { CK(cudaMemsetAsync(buf, 1, 256u << 20, s)); CK(cudaStreamSynchronize(s)); }
    }
    {
        std::vector<uint8_t> h(ring + (64u << 20));
        std::mt19937 rng(7);
        for (auto& c : h) { c = (uint8_t) (rng() & 0xFFu); if ((c & 0x78u) == 0) c |= 0x40u; if ((c & 0x7Fu) == 0x7Fu) c &= 0xFEu; }
        CK(cudaMemcpy(buf, h.data(), h.size(), cudaMemcpyHostToDevice));
    }
    struct Shape { int n_in, n_out; const char* what; };
    const Shape shapes[] = {{4096, 1024, "q_a"}, {1024, 32768, "q_b HCA"}, {1024, 8192, "gate HCA"}, {8192, 4096, "output_b"},
                            {4096, 2048, "shexp g/u"}, {2048, 4096, "shexp down"}, {4096, 14336, "ffn g/u L0"}, {14336, 4096, "ffn down L0"}};
    auto time_it = [&](auto&& launch, size_t per) {   // median of 15 reps of 64 launches, cycling copies in the ring
        const int copies = (int) std::max<size_t>(1, ring / per);
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        std::vector<float> ms;
        for (int rep = 0; rep < 15; ++rep) {
            CK(cudaEventRecord(a, s));
            for (int k = 0; k < 64; ++k) launch(buf + (size_t) (k % copies) * per);
            CK(cudaEventRecord(b, s));
            CK(cudaEventSynchronize(b));
            float t;
            CK(cudaEventElapsedTime(&t, a, b));
            ms.push_back(t * 1000.0f / 64.0f);
        }
        CK(cudaGetLastError());
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    };
    std::printf("%-12s %6s %6s %7s | GB/s: %6s %6s | %6s %6s %6s %6s %6s | %6s %6s %6s | %6s\n", "shape", "n_in", "n_out", "MB",
                "linear", "FP8", "r8R1u2", "r8R2u2", "r8R4u2", "r8R2u8", "r8R4u4", "r16R1", "r16R2", "r16R2u4", "FP8 us");
    for (const Shape& sh : shapes) {
        const size_t nv = (size_t) sh.n_in * sh.n_out, per = (nv + nv / 32 + 255) & ~(size_t) 255;
        const double mb = (nv + nv / 32) / 1e6;
        const int nb = pr.multiProcessorCount * 8;
        auto bw = [&](float us) { return mb * 1e6 / (us * 1e-6) / 1e9; };
        const float t_lin = time_it([&](const uint8_t* w) { k_linear<<<nb, 256, 0, s>>>(reinterpret_cast<const uint4*>(w), nv / 16, out); }, per);
        for (size_t c = 0; c < std::max<size_t>(1, ring / per); ++c)   // each copy's scales: one exponent, no exponent-0 mark
            CK(cudaMemsetAsync(buf + c * per + nv, 0x6F, nv / 32, s));
        const float t_fp8 = time_it([&](const uint8_t* w) { strata::x::gemv_fp8_w(w, w + nv, x, y, sh.n_in, sh.n_out, 1, s); }, per);
        auto rows8 = [&](auto R, auto U) {
            constexpr int RR = decltype(R)::value, UU = decltype(U)::value;
            const unsigned g = (unsigned) ((sh.n_out + 8 * RR - 1) / (8 * RR));
            return time_it([&](const uint8_t* w) { k_rows8<RR, UU><<<g, 256, 0, s>>>(w, sh.n_in, sh.n_out, out); }, per);
        };
        auto rows16 = [&](auto R, auto U) {
            constexpr int RR = decltype(R)::value, UU = decltype(U)::value;
            const unsigned g = (unsigned) ((sh.n_out + 16 * RR - 1) / (16 * RR));
            return time_it([&](const uint8_t* w) { k_rows16<RR, UU><<<g, 256, 0, s>>>(w, sh.n_in, sh.n_out, out); }, per);
        };
        using I1 = std::integral_constant<int, 1>; using I2 = std::integral_constant<int, 2>;
        using I4 = std::integral_constant<int, 4>; using I8 = std::integral_constant<int, 8>;
        const float a1 = rows8(I1{}, I2{}), a2 = rows8(I2{}, I2{}), a4 = rows8(I4{}, I2{}), a28 = rows8(I2{}, I8{}), a44 = rows8(I4{}, I4{});
        const float b1 = rows16(I1{}, I2{}), b2 = rows16(I2{}, I2{}), b24 = rows16(I2{}, I4{});
        std::printf("%-12s %6d %6d %7.1f | GB/s: %6.0f %6.0f | %6.0f %6.0f %6.0f %6.0f %6.0f | %6.0f %6.0f %6.0f | %6.1f\n", sh.what,
                    sh.n_in, sh.n_out, mb, bw(t_lin), bw(t_fp8), bw(a1), bw(a2), bw(a4), bw(a28), bw(a44), bw(b1), bw(b2), bw(b24), t_fp8);
    }
    return 0;
}
