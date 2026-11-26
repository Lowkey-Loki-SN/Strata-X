// moex/tools/hy4/heads_q8_0_parity.cu - the MLA's per-head Q8_0 projections (src/x/kernels.cu heads_q8_0) for a prompt's
// T > 8 columns: the reference (heads_q8_0_kernel verbatim, launched 8 columns at a time as heads_q8_0 does) against tiled
// kernels that compute each output's 32 lane partials and add them in warp_sum's butterfly order - bit for bit, then timed.
// heads_q8_0_t3_kernel is the engine's heads_q8_0_tiled_kernel.  Built by hand: nvcc -O3 -DNDEBUG -arch=sm_86 -o
// heads_q8_0_parity heads_q8_0_parity.cu && ./heads_q8_0_parity.  moex/docs/HY4_PORT.md, 2026-11-19.
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

// ---- new: one thread an output tile of RT rows x CT columns, for nb <= 8 (n_in <= 256: at most two 4-value groups a lane
// in the reference, e = l and l + 32).  Lane l's partial is the reference's chain over its groups from 0: fma(d_w d_x, dot,
// acc) (0 past the row); the 32 partials are added as warp_sum adds them for lane 0 (l + (l ^ 16), then ^ 8, ^ 4, ^ 2, ^ 1).
template <int RT, int CT>
__global__ void __launch_bounds__(256) heads_q8_0_tiled_kernel(const HQ80* __restrict__ W, const HQ81* __restrict__ X,
                                                               float* __restrict__ Y, int nb, int n_out, int T) {
    constexpr int RB = 16 * RT, CB = 16 * CT, NG = 64;
    __shared__ int Ws[RB][NG + 1];   // a row's groups (+1: banks)
    __shared__ float Wd[RB][8];
    __shared__ int Xs[CB][NG + 1];
    __shared__ float Xd[CB][8];
    const int h = blockIdx.z, r0 = blockIdx.x * RB, t0 = blockIdx.y * CB;
    const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const int ng = nb * 8;   // groups a row
    for (int e = tid; e < RB * NG; e += 256) {
        const int rr = e / NG, g = e % NG, r = r0 + rr;
        int v = 0;
        if (r < n_out && g < ng) v = load_i16x2(W[((size_t) h * n_out + r) * nb + (g >> 3)].qs, g & 7);
        Ws[rr][g] = v;
        if (g < 8) Wd[rr][g] = (r < n_out && g < nb) ? __half2float(W[((size_t) h * n_out + r) * nb + g].d) : 0.0f;
    }
    for (int e = tid; e < CB * NG; e += 256) {
        const int cc = e / NG, g = e % NG, t = t0 + cc;
        int v = 0;
        if (t < T && g < ng) v = reinterpret_cast<const int*>(X[((size_t) h * T + t) * nb + (g >> 3)].qs)[g & 7];
        Xs[cc][g] = v;
        if (g < 8) Xd[cc][g] = (t < T && g < nb) ? __low2float(X[((size_t) h * T + t) * nb + g].ds) : 0.0f;
    }
    __syncthreads();
    auto part = [&](int j, int i, int l) -> float {   // lane l's acc: its groups l, l + 32 in order (0 past the row)
        const int rw = ty * RT + j, cl = tx * CT + i;
        float acc = 0.0f;
#pragma unroll
        for (int e = l; e < NG; e += 32)
            if (e < ng) {
                const int d = __dp4a(Ws[rw][e], Xs[cl][e], 0);
                acc = __fmaf_rn(__fmul_rn(Wd[rw][e >> 3], Xd[cl][e >> 3]), __int2float_rn(d), acc);
            }
        return acc;
    };
#pragma unroll
    for (int j = 0; j < RT; ++j) {
        const int r = r0 + ty * RT + j;
#pragma unroll
        for (int i = 0; i < CT; ++i) {
            float v1[16];
#pragma unroll
            for (int l = 0; l < 16; ++l) v1[l] = __fadd_rn(part(j, i, l), part(j, i, l + 16));
            float v2[8];
#pragma unroll
            for (int l = 0; l < 8; ++l) v2[l] = __fadd_rn(v1[l], v1[l + 8]);
            float v3[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) v3[l] = __fadd_rn(v2[l], v2[l + 4]);
            const float o = __fadd_rn(__fadd_rn(v3[0], v3[2]), __fadd_rn(v3[1], v3[3]));
            const int t = t0 + tx * CT + i;
            if (r < n_out && t < T) Y[((size_t) h * T + t) * n_out + r] = o;
        }
    }
}

// ---- v2: the groups transposed in shared memory (a thread's 4 rows / 4 columns of one group: one 16-byte read each), the
// int -> float of each dot by the exact magic-number add (|dot| <= 4 x 127 x 128 < 2^22), the partials added in the
// butterfly's order as they come (bit-reversed pairs: a few live sums an output)
template <int RT, int CT>
__global__ void __launch_bounds__(256) heads_q8_0_tiled2_kernel(const HQ80* __restrict__ W, const HQ81* __restrict__ X,
                                                                float* __restrict__ Y, int nb, int n_out, int T) {
    constexpr int RB = 16 * RT, CB = 16 * CT;
    __shared__ __align__(16) int Ws[32][RB + 4];
    __shared__ __align__(16) float Wd[4][RB + 4];
    __shared__ __align__(16) int Xs[32][CB + 4];
    __shared__ __align__(16) float Xd[4][CB + 4];
    const int h = blockIdx.z, r0 = blockIdx.x * RB, t0 = blockIdx.y * CB;
    const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const int ng = nb * 8;
    for (int e = tid; e < RB * 32; e += 256) {
        const int rr = e >> 5, g = e & 31, r = r0 + rr;
        int v = 0;
        if (r < n_out && g < ng) v = load_i16x2(W[((size_t) h * n_out + r) * nb + (g >> 3)].qs, g & 7);
        Ws[g][rr] = v;
        if (g < 4) Wd[g][rr] = (r < n_out && g < nb) ? __half2float(W[((size_t) h * n_out + r) * nb + g].d) : 0.0f;
    }
    for (int e = tid; e < CB * 32; e += 256) {
        const int cc = e >> 5, g = e & 31, t = t0 + cc;
        int v = 0;
        if (t < T && g < ng) v = reinterpret_cast<const int*>(X[((size_t) h * T + t) * nb + (g >> 3)].qs)[g & 7];
        Xs[g][cc] = v;
        if (g < 4) Xd[g][cc] = (t < T && g < nb) ? __low2float(X[((size_t) h * T + t) * nb + g].ds) : 0.0f;
    }
    __syncthreads();
    float pr[4][RT][CT];
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int i = 0; i < CT; ++i) pr[b][j][i] = __fmul_rn(Wd[b][ty * RT + j], Xd[b][tx * CT + i]);
    float v2s[RT][CT], v3s[RT][CT], v4s[RT][CT], o[RT][CT];
    // lane l's partial for this thread's outputs: fma(d_w d_x, dot, +0), 0 past the row
    auto pairs = [&](int l, float (&dst)[RT][CT]) {   // v1 = p_l + p_(l+16)
        int wa[RT], wb[RT], xa[CT], xb[CT];
        {
            const int4 a4 = *reinterpret_cast<const int4*>(&Ws[l][ty * RT]);
            const int4 b4 = *reinterpret_cast<const int4*>(&Ws[l + 16][ty * RT]);
            const int4 c4 = *reinterpret_cast<const int4*>(&Xs[l][tx * CT]);
            const int4 d4 = *reinterpret_cast<const int4*>(&Xs[l + 16][tx * CT]);
            wa[0] = a4.x; wa[1] = a4.y; wa[2] = a4.z; wa[3] = a4.w;
            wb[0] = b4.x; wb[1] = b4.y; wb[2] = b4.z; wb[3] = b4.w;
            xa[0] = c4.x; xa[1] = c4.y; xa[2] = c4.z; xa[3] = c4.w;
            xb[0] = d4.x; xb[1] = d4.y; xb[2] = d4.z; xb[3] = d4.w;
        }
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int i = 0; i < CT; ++i) {
                const float fa = __fsub_rn(__int_as_float(0x4B400000 + __dp4a(wa[j], xa[i], 0)), 12582912.0f);
                const float fb = __fsub_rn(__int_as_float(0x4B400000 + __dp4a(wb[j], xb[i], 0)), 12582912.0f);
                const float pa = l < ng ? __fmaf_rn(pr[l >> 3][j][i], fa, 0.0f) : 0.0f;
                const float pb = l + 16 < ng ? __fmaf_rn(pr[(l + 16) >> 3][j][i], fb, 0.0f) : 0.0f;
                dst[j][i] = __fadd_rn(pa, pb);
            }
    };
    auto add2 = [&](float (&a)[RT][CT], const float (&b)[RT][CT]) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int i = 0; i < CT; ++i) a[j][i] = __fadd_rn(a[j][i], b[j][i]);
    };
    float t1[RT][CT], t2[RT][CT];
    // v4[0] = ((v1[0] + v1[8]) + (v1[4] + v1[12])) + ((v1[2] + v1[10]) + (v1[6] + v1[14]))
    pairs(0, v2s); pairs(8, t1); add2(v2s, t1);                     // v2[0]
    pairs(4, v3s); pairs(12, t1); add2(v3s, t1); add2(v2s, v3s);    // v3[0] in v2s
    pairs(2, v3s); pairs(10, t1); add2(v3s, t1);                    // v2[2]
    pairs(6, v4s); pairs(14, t1); add2(v4s, t1); add2(v3s, v4s);    // v3[2] in v3s
    add2(v2s, v3s);                                                 // v4[0] in v2s
    pairs(1, v3s); pairs(9, t1); add2(v3s, t1);                     // v2[1]
    pairs(5, v4s); pairs(13, t1); add2(v4s, t1); add2(v3s, v4s);    // v3[1] in v3s
    pairs(3, v4s); pairs(11, t1); add2(v4s, t1);                    // v2[3]
    pairs(7, t2); pairs(15, t1); add2(t2, t1); add2(v4s, t2);       // v3[3] in v4s
    add2(v3s, v4s);                                                 // v4[1] in v3s
#pragma unroll
    for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int i = 0; i < CT; ++i) o[j][i] = __fadd_rn(v2s[j][i], v3s[j][i]);
#pragma unroll
    for (int j = 0; j < RT; ++j) {
        const int r = r0 + ty * RT + j;
        if (r >= n_out) continue;
#pragma unroll
        for (int i = 0; i < CT; ++i) {
            const int t = t0 + tx * CT + i;
            if (t < T) Y[((size_t) h * T + t) * n_out + r] = o[j][i];
        }
    }
}

// ---- v3: NB (blocks a row) at compile time - 4, 6 or 8 (n_in 128 / 192 / 256: e = l and, past 128, l + 32 a lane); the
// groups transposed in shared memory (a thread's 4 rows or 4 columns of one group in one 16-byte read, their scales too);
// each term the reference's FMUL(d_w, d_x) then FFMA(that, dot, acc), the dot's int -> float by the exact magic add; the
// partials added in warp_sum's order as they come (bit-reversed pairs: a few live sums an output)
__device__ __forceinline__ float i2f_exact(int d) { return __fsub_rn(__int_as_float(0x4B400000 + d), 12582912.0f); }
template <int NB, int RT, int CT>
__global__ void __launch_bounds__(256) heads_q8_0_t3_kernel(const HQ80* __restrict__ W, const HQ81* __restrict__ X,
                                                            float* __restrict__ Y, int n_out, int T) {
    constexpr int RB = 16 * RT, CB = 16 * CT, NG = NB * 8;
    static_assert(RT == 4 && CT == 4, "one 16-byte read a group");
    __shared__ __align__(16) int Ws[NG][RB + 4];
    __shared__ __align__(16) float Wd[NB][RB + 4];
    __shared__ __align__(16) int Xs[NG][CB + 4];
    __shared__ __align__(16) float Xd[NB][CB + 4];
    const int h = blockIdx.z, r0 = blockIdx.x * RB, t0 = blockIdx.y * CB;
    const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    for (int e = tid; e < RB * NG; e += 256) {
        const int rr = e / NG, g = e % NG, r = r0 + rr;
        Ws[g][rr] = r < n_out ? load_i16x2(W[((size_t) h * n_out + r) * NB + (g >> 3)].qs, g & 7) : 0;
        if (g < NB) Wd[g][rr] = r < n_out ? __half2float(W[((size_t) h * n_out + r) * NB + g].d) : 0.0f;
    }
    for (int e = tid; e < CB * NG; e += 256) {
        const int cc = e / NG, g = e % NG, t = t0 + cc;
        Xs[g][cc] = t < T ? reinterpret_cast<const int*>(X[((size_t) h * T + t) * NB + (g >> 3)].qs)[g & 7] : 0;
        if (g < NB) Xd[g][cc] = t < T ? __low2float(X[((size_t) h * T + t) * NB + g].ds) : 0.0f;
    }
    __syncthreads();
    // lane l's partials for this thread's 16 outputs: the chain over its groups e = l, l + 32 (< NG), from 0
    auto lane_part = [&](int l, float (&p)[RT][CT]) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int i = 0; i < CT; ++i) p[j][i] = 0.0f;
#pragma unroll
        for (int e = l; e < NG; e += 32) {
            const int4 w4 = *reinterpret_cast<const int4*>(&Ws[e][ty * RT]);
            const int4 x4 = *reinterpret_cast<const int4*>(&Xs[e][tx * CT]);
            const float4 wd = *reinterpret_cast<const float4*>(&Wd[e >> 3][ty * RT]);
            const float4 xd = *reinterpret_cast<const float4*>(&Xd[e >> 3][tx * CT]);
            const int wv[4] = {w4.x, w4.y, w4.z, w4.w}, xv[4] = {x4.x, x4.y, x4.z, x4.w};
            const float dw[4] = {wd.x, wd.y, wd.z, wd.w}, dx[4] = {xd.x, xd.y, xd.z, xd.w};
#pragma unroll
            for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int i = 0; i < CT; ++i)
                    p[j][i] = __fmaf_rn(__fmul_rn(dw[j], dx[i]), i2f_exact(__dp4a(wv[j], xv[i], 0)), p[j][i]);
        }
    };
    auto pair = [&](int l, float (&v)[RT][CT]) {   // v1[l] = p_l + p_(l+16)
        float a[RT][CT], b[RT][CT];
        lane_part(l, a);
        lane_part(l + 16, b);
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int i = 0; i < CT; ++i) v[j][i] = __fadd_rn(a[j][i], b[j][i]);
    };
    auto add2 = [&](float (&a)[RT][CT], const float (&b)[RT][CT]) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int i = 0; i < CT; ++i) a[j][i] = __fadd_rn(a[j][i], b[j][i]);
    };
    // v4[0] = ((v1[0] + v1[8]) + (v1[4] + v1[12])) + ((v1[2] + v1[10]) + (v1[6] + v1[14])); v4[1] the same from 1
    float A[RT][CT], Bv[RT][CT], C[RT][CT], D[RT][CT];
    pair(0, A); pair(8, D); add2(A, D);
    pair(4, Bv); pair(12, D); add2(Bv, D); add2(A, Bv);
    pair(2, Bv); pair(10, D); add2(Bv, D);
    pair(6, C); pair(14, D); add2(C, D); add2(Bv, C);
    add2(A, Bv);   // v4[0]
    pair(1, Bv); pair(9, D); add2(Bv, D);
    pair(5, C); pair(13, D); add2(C, D); add2(Bv, C);
    pair(3, C); pair(11, D); add2(C, D);
    float E[RT][CT];
    pair(7, E); pair(15, D); add2(E, D); add2(C, E);
    add2(Bv, C);   // v4[1]
#pragma unroll
    for (int j = 0; j < RT; ++j) {
        const int r = r0 + ty * RT + j;
#pragma unroll
        for (int i = 0; i < CT; ++i) {
            const int t = t0 + tx * CT + i;
            if (r < n_out && t < T) Y[((size_t) h * T + t) * n_out + r] = __fadd_rn(A[j][i], Bv[j][i]);
        }
    }
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

int main() {
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> q8(-127, 127);
    std::normal_distribution<float> nd(0.f, 1.f);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    int fails = 0;
    struct C { int H, n_in, n_out, T; };
    const C cs[] = {{64, 192, 512, 9}, {64, 192, 512, 512}, {32, 192, 512, 512}, {64, 192, 512, 77}, {16, 256, 300, 100},
                    {8, 224, 64, 40}, {8, 160, 64, 17}, {64, 128, 512, 9}, {64, 128, 512, 64}, {64, 128, 512, 100}, {64, 128, 512, 512}, {32, 128, 512, 512},
                    {64, 128, 512, 4096}, {16, 64, 200, 77}, {8, 96, 130, 33}, {64, 32, 512, 40}};
    for (const C& c : cs) {
        const int nb = c.n_in / 32;
        std::vector<HQ80> hw((size_t) c.H * c.n_out * nb);
        for (auto& x : hw) { x.d = __float2half(0.01f * (1.0f + nd(rng) * 0.3f)); for (auto& q : x.qs) q = (int8_t) q8(rng); }
        std::vector<HQ81> hx((size_t) c.H * c.T * nb);
        for (auto& x : hx) { x.ds = __floats2half2_rn(0.02f * (1.0f + nd(rng) * 0.3f) * (nd(rng) < -1.5f ? -1.0f : 1.0f), 0.f);
                             for (auto& q : x.qs) q = (int8_t) q8(rng); }
        // some zero blocks (a row or a column of zeros: -0 / +0 paths)
        for (size_t i = 0; i < hx.size(); i += 37) for (auto& q : hx[i].qs) q = 0;
        HQ80* dw; HQ81* dx; float *y0, *y1;
        CK(cudaMalloc(&dw, hw.size() * sizeof(HQ80)));
        CK(cudaMalloc(&dx, hx.size() * sizeof(HQ81)));
        CK(cudaMalloc(&y0, (size_t) c.H * c.T * c.n_out * 4));
        CK(cudaMalloc(&y1, (size_t) c.H * c.T * c.n_out * 4));
        CK(cudaMemcpy(dw, hw.data(), hw.size() * sizeof(HQ80), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(HQ81), cudaMemcpyHostToDevice));
        auto ref = [&] {
            for (int c0 = 0; c0 < c.T; c0 += 8)
                heads_q8_0_kernel<<<dim3((c.n_out + 3) / 4, c.H), 128, 0, s>>>(dw, dx, y0, nb, c.n_out, std::min(8, c.T - c0), c.T, c0);
        };
        auto nw = [&] {
            constexpr int RT = 4, CT = 4;
            const dim3 g((c.n_out + 16 * RT - 1) / (16 * RT), (c.T + 16 * CT - 1) / (16 * CT), c.H);
            if (nb == 4) heads_q8_0_t3_kernel<4, RT, CT><<<g, 256, 0, s>>>(dw, dx, y1, c.n_out, c.T);
            else if (nb == 6) heads_q8_0_t3_kernel<6, RT, CT><<<g, 256, 0, s>>>(dw, dx, y1, c.n_out, c.T);
            else if (nb == 8) heads_q8_0_t3_kernel<8, RT, CT><<<g, 256, 0, s>>>(dw, dx, y1, c.n_out, c.T);
            else heads_q8_0_tiled_kernel<RT, CT><<<g, 256, 0, s>>>(dw, dx, y1, nb, c.n_out, c.T);
        };
        CK(cudaMemset(y0, 0xff, (size_t) c.H * c.T * c.n_out * 4));
        CK(cudaMemset(y1, 0x7f, (size_t) c.H * c.T * c.n_out * 4));
        ref();
        nw();
        CK(cudaStreamSynchronize(s));
        CK(cudaGetLastError());
        std::vector<float> r0((size_t) c.H * c.T * c.n_out), r1(r0.size());
        CK(cudaMemcpy(r0.data(), y0, r0.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(r1.data(), y1, r1.size() * 4, cudaMemcpyDeviceToHost));
        long bad = 0;
        for (size_t i = 0; i < r0.size(); ++i) if (memcmp(&r0[i], &r1[i], 4)) { if (bad < 3) fprintf(stderr, "  [%zu] %.9g vs %.9g\n", i, r0[i], r1[i]); ++bad; }
        fails += bad != 0;
        auto tm = [&](auto&& f) { f(); cudaStreamSynchronize(s); cudaEventRecord(a, s); for (int k = 0; k < 10; ++k) f(); cudaEventRecord(b, s);
                                  cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms / 10; };
        const float t0 = tm(ref), t1 = tm(nw);
        printf("H %2d n_in %3d n_out %3d T %4d: %s (%ld differ)  reference %.3f ms, tiled %.3f ms\n", c.H, c.n_in, c.n_out, c.T,
               bad ? "DIFF" : "same", bad, t0, t1);
        cudaFree(dw); cudaFree(dx); cudaFree(y0); cudaFree(y1);
    }
    printf(fails ? "FAILURES %d\n" : "ALL SAME\n", fails);
    return fails != 0;
}
