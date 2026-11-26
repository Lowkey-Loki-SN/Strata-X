// moex/tools/hy4/heads_q8_0_decode_parity.cu - heads_q8_0 for a decode window (T <= 8 columns): the reference kernel verbatim
// against R rows a warp (the engine's heads_q8_0_rows_kernel) - bit for bit at 1-8 columns, then timed (nvcc -O3 -arch=sm_86).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
struct HQ80 { half d; int8_t qs[32]; };
struct HQ81 { half2 ds; int8_t qs[32]; };
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ int load_i16x2(const int8_t* p, int i) {
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p) + 2 * i;
    return (int) ((uint32_t) q[0] | ((uint32_t) q[1] << 16));
}
__global__ void __launch_bounds__(128) heads_q8_0_kernel(const HQ80* __restrict__ W, const HQ81* __restrict__ X,
                                                         float* __restrict__ Y, int nb, int n_out, int T, int T_total,
                                                         int c0) {
    const int h = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 4 + warp;
    if (r >= n_out) return;
    const HQ80* w = W + ((size_t) h * n_out + r) * nb;
    const HQ81* x = X + ((size_t) h * T_total + c0) * nb;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int e = lane; e < nb * 8; e += 32) {
        const int b = e >> 3, i = e & 7;
        const int v = load_i16x2(w[b].qs, i);
        const float dw = __half2float(w[b].d);
#pragma unroll
        for (int t = 0; t < 8; ++t)
            if (t < T) {
                const HQ81* xb = x + (size_t) t * nb + b;
                const int u = reinterpret_cast<const int*>(xb->qs)[i];
                acc[t] += dw * __low2float(xb->ds) * (float) __dp4a(v, u, 0);
            }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t)
        if (t < T) {
            const float v = warp_sum(acc[t]);
            if (lane == 0) Y[((size_t) h * T_total + c0 + t) * n_out + r] = v;
        }
}
// R rows a warp (rows (blockIdx.x * 4 + warp) * R ..), TC columns at compile time
template <int R, int TC>
__global__ void __launch_bounds__(128) heads_q8_0_rows_kernel(const HQ80* __restrict__ W, const HQ81* __restrict__ X,
                                                              float* __restrict__ Y, int nb, int n_out, int T_total, int c0) {
    const int h = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * 4 + warp) * R;
    if (r0 >= n_out) return;
    const HQ81* x = X + ((size_t) h * T_total + c0) * nb;
    float acc[R][TC];
#pragma unroll
    for (int k = 0; k < R; ++k)
#pragma unroll
        for (int t = 0; t < TC; ++t) acc[k][t] = 0.0f;
    for (int e = lane; e < nb * 8; e += 32) {
        const int b = e >> 3, i = e & 7;
        int u[TC];
        float dx[TC];
#pragma unroll
        for (int t = 0; t < TC; ++t) {
            const HQ81* xb = x + (size_t) t * nb + b;
            u[t] = reinterpret_cast<const int*>(xb->qs)[i];
            dx[t] = __low2float(xb->ds);
        }
        int v[R];
        float dw[R];
#pragma unroll
        for (int k = 0; k < R; ++k) {
            const int r = min(r0 + k, n_out - 1);   // (a row past the end: read, never written)
            const HQ80* wb = W + ((size_t) h * n_out + r) * nb + b;
            v[k] = load_i16x2(wb->qs, i);
            dw[k] = __half2float(wb->d);
        }
#pragma unroll
        for (int k = 0; k < R; ++k)
#pragma unroll
            for (int t = 0; t < TC; ++t)
                acc[k][t] = __fmaf_rn(__fmul_rn(dw[k], dx[t]), __int2float_rn(__dp4a(v[k], u[t], 0)), acc[k][t]);
    }
#pragma unroll
    for (int k = 0; k < R; ++k)
#pragma unroll
        for (int t = 0; t < TC; ++t) {
            const float s = warp_sum(acc[k][t]);
            if (lane == 0 && r0 + k < n_out) Y[((size_t) h * T_total + c0 + t) * n_out + r0 + k] = s;
        }
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)
template <int R>
void launch_rows(const HQ80* w, const HQ81* x, float* y, int H, int nb, int n_out, int T, cudaStream_t s) {
    const dim3 g((n_out + 4 * R - 1) / (4 * R), H);
    switch (T) {
#define CASE(TC) case TC: heads_q8_0_rows_kernel<R, TC><<<g, 128, 0, s>>>(w, x, y, nb, n_out, T, 0); break;
        CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
    }
}
int main() {
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> q8(-127, 127);
    std::normal_distribution<float> nd(0.f, 1.f);
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    int fails = 0;
    struct C { int H, n_in, n_out; };
    for (const C& c : {C{64, 192, 512}, C{64, 512, 128}, C{32, 192, 512}, C{32, 512, 128}, C{16, 96, 130}})
        for (int T = 1; T <= 8; ++T) {
            const int nb = c.n_in / 32;
            std::vector<HQ80> hw((size_t) c.H * c.n_out * nb);
            for (auto& x : hw) { x.d = __float2half(0.01f * (1.0f + nd(rng) * 0.3f)); for (auto& q : x.qs) q = (int8_t) q8(rng); }
            std::vector<HQ81> hx((size_t) c.H * T * nb);
            for (auto& x : hx) { x.ds = __floats2half2_rn(0.02f * (1.0f + nd(rng) * 0.3f) * (nd(rng) < -1.5f ? -1.f : 1.f), 0.f); for (auto& q : x.qs) q = (int8_t) q8(rng); }
            for (size_t i = 0; i < hx.size(); i += 29) for (auto& q : hx[i].qs) q = 0;
            HQ80* dw; HQ81* dx; float *y0, *y1;
            CK(cudaMalloc(&dw, hw.size() * sizeof(HQ80))); CK(cudaMalloc(&dx, hx.size() * sizeof(HQ81)));
            CK(cudaMalloc(&y0, (size_t) c.H * T * c.n_out * 4)); CK(cudaMalloc(&y1, (size_t) c.H * T * c.n_out * 4));
            CK(cudaMemcpy(dw, hw.data(), hw.size() * sizeof(HQ80), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(HQ81), cudaMemcpyHostToDevice));
            auto ref = [&] { heads_q8_0_kernel<<<dim3((c.n_out + 3) / 4, c.H), 128, 0, s>>>(dw, dx, y0, nb, c.n_out, T, T, 0); };
            auto nw = [&] { launch_rows<4>(dw, dx, y1, c.H, nb, c.n_out, T, s); };
            CK(cudaMemset(y0, 0xff, (size_t) c.H * T * c.n_out * 4)); CK(cudaMemset(y1, 0x7f, (size_t) c.H * T * c.n_out * 4));
            ref(); nw(); CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
            std::vector<float> r0((size_t) c.H * T * c.n_out), r1(r0.size());
            CK(cudaMemcpy(r0.data(), y0, r0.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r1.data(), y1, r1.size() * 4, cudaMemcpyDeviceToHost));
            long bad = 0;
            for (size_t i = 0; i < r0.size(); ++i) bad += memcmp(&r0[i], &r1[i], 4) != 0;
            fails += bad != 0;
            auto tm = [&](auto&& f) { f(); cudaStreamSynchronize(s); cudaEventRecord(a, s); for (int k = 0; k < 50; ++k) f(); cudaEventRecord(b, s);
                                      cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return 1000.0f * ms / 50; };
            if (T == 1 || T == 6 || bad) printf("H %2d n_in %3d n_out %3d T %d: %s (%ld differ)  reference %.1f us, 4 rows a warp %.1f us\n", c.H, c.n_in, c.n_out, T,
                                               bad ? "DIFF" : "same", bad, tm(ref), tm(nw));
            cudaFree(dw); cudaFree(dx); cudaFree(y0); cudaFree(y1);
        }
    printf(fails ? "FAILURES %d\n" : "ALL SAME\n", fails);
    return fails != 0;
}
