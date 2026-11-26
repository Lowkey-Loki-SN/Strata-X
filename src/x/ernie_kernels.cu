// src/x/ernie_kernels.cu - moex: ERNIE-5.5-Flash's kernels (include/strata/x/ernie_kernels.hpp; moex/docs/ERNIE55_PLAN.md).
//
// Each op follows the reference model (moex/tools/ernie55/ref/model.py) in fp32: elementwise steps in its order with each
// operation rounded (__fmul_rn / __fadd_rn where a contraction would change a bit); the reductions are float trees or
// in-order sums, so they differ from torch's in the last bits.
#include "strata/x/ernie_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace strata::x {
namespace {

inline cudaStream_t st(void* s) { return static_cast<cudaStream_t>(s); }

inline void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "ernie_kernels: %s: %s\n", what, cudaGetErrorString(e));
        throw std::runtime_error(what);
    }
}

__device__ __forceinline__ int64_t pos_of(const int64_t* __restrict__ p, int64_t arg) { return p ? *p : arg; }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/// Sum over the block (blockDim.x a multiple of 32, <= 1024); every thread gets the total.  Every thread must call it.
__device__ float block_sum(float v) {
    __shared__ float red[32];
    __shared__ float total;
    v = warp_sum(v);
    const int nw = (int) (blockDim.x >> 5);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float w = (int) threadIdx.x < nw ? red[threadIdx.x] : 0.0f;
        w = warp_sum(w);
        if (threadIdx.x == 0) total = w;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}

__device__ __forceinline__ float sigmoid_rn(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x))); }

// ------------------------------------------------------------------ VHA

constexpr int kVhaH = 64;   ///< heads
constexpr int kVhaD = 64;   ///< values a tile

/// grid (D / 64, 64 / (4 JN), T), 256 threads: M's rows h0 .. h0 + 4 JN - 1 and the token's 64 x 64 input tile in shared memory;
/// thread (d = tid % 64, hb = tid / 64) computes out[t, h0 + hb + 4 j, d0 + d] for j < JN as sum_g M[h, g] in[t, g, d0 + d], g in
/// order (one fmaf chain an output, whichever block runs it: the same bits for every JN).  moex (the ERNIE-5.5 hill-climb): JN 4,
/// the 64 heads over four blocks a tile - JN 16 (one block a tile) read shared memory on 8 SMs, 8.8 us a decode layer
template <int JN>
__global__ void __launch_bounds__(256) vha_kernel(const float* __restrict__ in, float* __restrict__ out, const float* __restrict__ M,
                                                  int D) {
    constexpr int HB = 4 * JN;   // the heads a block
    __shared__ float ms[HB][kVhaH];
    __shared__ float tile[kVhaH][kVhaD];
    const int d0 = blockIdx.x * kVhaD, h0 = blockIdx.y * HB, t = blockIdx.z;
    const float* src = in + (size_t) t * kVhaH * D;
    for (int i = threadIdx.x; i < HB * kVhaH; i += 256) ms[i / kVhaH][i % kVhaH] = M[(size_t) h0 * kVhaH + i];
    for (int i = threadIdx.x; i < kVhaH * kVhaD; i += 256) tile[i / kVhaD][i % kVhaD] = src[(size_t) (i / kVhaD) * D + d0 + i % kVhaD];
    __syncthreads();
    const int d = threadIdx.x % kVhaD, hb = threadIdx.x / kVhaD;
    float acc[JN];
#pragma unroll
    for (int j = 0; j < JN; ++j) acc[j] = 0.0f;
    for (int g = 0; g < kVhaH; ++g) {
        const float v = tile[g][d];
#pragma unroll
        for (int j = 0; j < JN; ++j) acc[j] = fmaf(ms[hb + 4 * j][g], v, acc[j]);
    }
    float* dst = out + (size_t) t * kVhaH * D + d0 + d;
#pragma unroll
    for (int j = 0; j < JN; ++j) dst[(size_t) (h0 + hb + 4 * j) * D] = acc[j];
}

// ------------------------------------------------------------------ routing

/// one block (256 threads) per token: the scores and choices in shared memory, then k rounds of a block argmax
__global__ void __launch_bounds__(256) route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                    const float* __restrict__ escale, int E, int k, int32_t* __restrict__ ids,
                                                    float* __restrict__ wts) {
    __shared__ float score[1024], choice[1024];
    __shared__ float rv[8];
    __shared__ int ri[8];
    __shared__ int s_ids[32];
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float* l = logits + (size_t) t * 2 * E;
    for (int e = threadIdx.x; e < E; e += 256) {
        const float s = __fadd_rn(sigmoid_rn(l[e]), sigmoid_rn(l[E + e]));
        score[e] = s;
        choice[e] = __fadd_rn(s, bias ? bias[e] : 0.0f);
    }
    __syncthreads();
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY;
        int bi = INT_MAX;
        for (int e = threadIdx.x; e < E; e += 256) {
            const float v = choice[e];
            if (v > bv || (v == bv && e < bi)) {
                bv = v;
                bi = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            rv[warp] = bv;
            ri[warp] = bi;
        }
        __syncthreads();
        if (warp == 0) {
            bv = lane < 8 ? rv[lane] : -INFINITY;
            bi = lane < 8 ? ri[lane] : INT_MAX;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bi = oi;
                }
            }
            if (lane == 0) {
                if (bi < 0 || bi >= E) bi = -1;   // (no finite choice left: the engine refuses the id)
                s_ids[j] = bi;
                if (bi >= 0) choice[bi] = -INFINITY;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) sum = __fadd_rn(sum, s_ids[j] >= 0 ? score[s_ids[j]] : 0.0f);
        sum = fmaxf(sum, 1e-20f);
        for (int j = 0; j < k; ++j) {
            const int e = s_ids[j];
            ids[(size_t) t * k + j] = e;
            wts[(size_t) t * k + j] = e >= 0 ? __fmul_rn(__fdiv_rn(score[e], sum), escale ? escale[e] : 1.0f) : 0.0f;
        }
    }
}

/// moex (ERNIE55_PLAN P6): route_kernel with its k rounds in ONE WARP - the block's 256 threads write the scores and choices
/// to shared memory as route_kernel does (one sync), then warp 0 holds experts lane + 32 m (m < M) in registers and each round
/// takes every lane's best untaken expert, then the warp's by xor shuffles, with route_kernel's rule (the larger choice, a tie
/// to the lower id): the unique (largest value, smallest id) expert, so the ids, their order and the weights (summed and
/// divided as route_kernel does) are route_kernel's.  Only finite choices are taken (route_kernel could take a chosen expert's
/// -inf again once every other choice was NaN; here the id is -1).  route_kernel spent 12.9 us a decode layer (k = 10 rounds of
/// two block syncs); STRATA_ERNIE_ROUTE_WARP=0 keeps it.
//
/// moex (the ERNIE-5.5 hill-climb), PUB: a one-token decode window's routing and its doorbell publish in one launch - the routing
/// above line for line (the same ids and weights), its ids into the mapped block too, warps 1-7 copying the activations x
/// there meanwhile, then doorbell_publish_kernel's fences and word (kernels.cu).  STRATA_ERNIE_ROUTE_PUBLISH=0: two launches
template <int M, bool PUB = false>
__global__ void __launch_bounds__(256) route_warp_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                         const float* __restrict__ escale, int E, int k,
                                                         int32_t* __restrict__ ids, float* __restrict__ wts,
                                                         const float* __restrict__ x = nullptr, int n_x = 0,
                                                         int32_t* __restrict__ m_ids = nullptr, float* __restrict__ m_x = nullptr,
                                                         volatile uint32_t* __restrict__ m_seq = nullptr,
                                                         uint32_t* __restrict__ d_cnt = nullptr) {
    __shared__ float score[32 * M], choice[32 * M];
    const int t = blockIdx.x, lane = threadIdx.x & 31;
    const float* l = logits + (size_t) t * 2 * E;
    for (int e = threadIdx.x; e < 32 * M; e += 256) {
        if (e < E) {
            const float sv = __fadd_rn(sigmoid_rn(l[e]), sigmoid_rn(l[E + e]));
            score[e] = sv;
            choice[e] = __fadd_rn(sv, bias ? bias[e] : 0.0f);
        } else {
            score[e] = 0.0f;
            choice[e] = -INFINITY;
        }
    }
    __syncthreads();
    if (threadIdx.x >= 32) {
        if constexpr (PUB) {   // doorbell_publish_kernel's copy of the activations, by the warps that route nothing
            for (int i = threadIdx.x - 32; i < n_x; i += 256 - 32) m_x[i] = x[i];
        } else {
            return;
        }
    } else {
        float ch[M];
#pragma unroll
        for (int m = 0; m < M; ++m) ch[m] = choice[lane + 32 * m];
        float sum = 0.0f;
        int my_id = -1;   // lane j's id (j < k), written at the end
        float my_sc = 0.0f;
        for (int j = 0; j < k; ++j) {
            float bv = -INFINITY;
            int bi = INT_MAX;
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const float v = ch[m];
                const int e = lane + 32 * m;
                if (v > -INFINITY && (v > bv || (v == bv && e < bi))) {   // (finite or +inf, never NaN)
                    bv = v;
                    bi = e;
                }
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bi = oi;
                }
            }
            const int id = bi >= 0 && bi < E ? bi : -1;
            if (id >= 0 && lane == (id & 31)) {
#pragma unroll
                for (int m = 0; m < M; ++m)
                    if (m == (id >> 5)) ch[m] = -INFINITY;
            }
            const float s_id = id >= 0 ? score[id] : 0.0f;
            sum = __fadd_rn(sum, s_id);
            if (lane == j) {
                my_id = id;
                my_sc = s_id;
            }
        }
        sum = fmaxf(sum, 1e-20f);
        if (lane < k) {
            ids[(size_t) t * k + lane] = my_id;
            wts[(size_t) t * k + lane] = my_id >= 0 ? __fmul_rn(__fdiv_rn(my_sc, sum), escale ? escale[my_id] : 1.0f) : 0.0f;
            if constexpr (PUB) m_ids[(size_t) t * k + lane] = my_id;
        }
    }   // (warp 0)
    if constexpr (PUB) {   // doorbell_publish_kernel's fences and word (one token: block 0 alone), every thread at one barrier
        __threadfence_system();   // every thread's payload is visible to the host before the word moves
        __syncthreads();
        if (threadIdx.x == 0) {
            const uint32_t v = *d_cnt + 1u;
            *d_cnt = v;
            *m_seq = v;
            __threadfence_system();
        }
    }
}

// ------------------------------------------------------------------ MultiMax

__global__ void multimax_kernel(float* __restrict__ x, int64_t V, int64_t ld, float r0, float r1, float r2, float r3, float t0,
                                float t1, float t2, float t3) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V) return;
    float* p = x + (size_t) blockIdx.y * ld + i;
    const float v = *p;
    float y = __fadd_rn(v, __fmul_rn(t0, fmaxf(__fsub_rn(r0, v), 0.0f)));
    y = __fadd_rn(y, __fmul_rn(t1, fmaxf(__fsub_rn(v, r1), 0.0f)));
    const float a = fmaxf(__fsub_rn(r2, v), 0.0f);
    y = __fadd_rn(y, __fmul_rn(t2, __fmul_rn(a, a)));
    const float b = fmaxf(__fsub_rn(v, r3), 0.0f);
    *p = __fadd_rn(y, __fmul_rn(t3, __fmul_rn(b, b)));
}

// ------------------------------------------------------------------ the indexer's FP8 rounding

/// one warp per group of 128 values (4 a lane)
__global__ void fq_e4m3_kernel(float* __restrict__ x, int64_t groups) {
    const int64_t g = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (g >= groups) return;
    const int lane = threadIdx.x & 31;
    float* p = x + g * 128;
    float v[4], am = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = p[lane + 32 * i];
        am = fmaxf(am, fabsf(v[i]));
    }
    am = warp_max(am);
    // the reference's fp32 steps: max(amax, 1e-10) / 448, log2, ceil, exp2 (a power of two: x / s and q * s are exact)
    const float s = exp2f(ceilf(log2f(__fdiv_rn(fmaxf(am, 1e-10f), 448.0f))));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float y = fminf(fmaxf(__fdiv_rn(v[i], s), -448.0f), 448.0f);
        const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(y, __NV_SATFINITE, __NV_E4M3);
        const __half_raw hr = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
        p[lane + 32 * i] = __fmul_rn(__half2float(__half(hr)), s);
    }
}

// ------------------------------------------------------------------ the DSA layers' cells

__global__ void dsa_cells_kernel(const int32_t* __restrict__ sel, int top_k, int window, int32_t* __restrict__ cells, int max_cells,
                                 int32_t* __restrict__ counts, int64_t pos0, const int64_t* __restrict__ pos_dev) {
    const int t = blockIdx.x;
    const int64_t p = pos_of(pos_dev, pos0) + t;
    const int64_t n_cand = p + 1 - window > 0 ? p + 1 - window : 0;
    const int n_sel = (int) (n_cand < (int64_t) top_k ? n_cand : (int64_t) top_k);
    const int64_t w0 = p - window + 1 > 0 ? p - window + 1 : 0;
    const int n_win = (int) (p - w0 + 1);
    int32_t* o = cells + (size_t) t * max_cells;
    const int32_t* s = sel + (size_t) t * top_k;
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) o[i] = s[i];
    for (int i = threadIdx.x; i < n_win; i += blockDim.x) o[n_sel + i] = (int32_t) (w0 + i);
    if (threadIdx.x == 0) counts[t] = n_sel + n_win;
}

// ------------------------------------------------------------------ mHC's flat-stream factor

// moex (the ERNIE-5.5 hill-climb): each thread's values loaded 16 at a time ahead of their fmaf chain and of their products - the
// same chain in the same order, the same products: the same bits (a decode token's 16,384 values: one block waited for one
// load after another, ~4.4 us a mixer)
__global__ void __launch_bounds__(256) rms_eo_kernel(const float* __restrict__ x, float* __restrict__ y, int n, float eps) {
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.0f;
    for (int i0 = threadIdx.x; i0 < n; i0 += 256 * 16) {
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) v[j] = i0 + 256 * j < n ? xr[i0 + 256 * j] : 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            if (i0 + 256 * j < n) s = fmaf(v[j], v[j], s);
    }
    s = block_sum(s);
    const float f = __fdiv_rn(1.0f, __fadd_rn(sqrtf(__fdiv_rn(s, (float) n)), eps));
    for (int i0 = threadIdx.x; i0 < n; i0 += 256 * 16) {
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) v[j] = i0 + 256 * j < n ? xr[i0 + 256 * j] : 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            if (i0 + 256 * j < n) yr[i0 + 256 * j] = __fmul_rn(v[j], f);
    }
}

// ------------------------------------------------------------------ the index keys' rows

__global__ void store_rows_kernel(const float* __restrict__ x, __half* __restrict__ cache, int width, int64_t cache_rows, int64_t pos0,
                                  const int64_t* __restrict__ pos_dev) {
    const int t = blockIdx.x;
    const int64_t p = pos_of(pos_dev, pos0) + t;
    if (p < 0 || p >= cache_rows) return;
    const float* src = x + (size_t) t * width;
    __half* dst = cache + (size_t) p * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[i] = __float2half_rn(src[i]);
}

}  // namespace

void ernie_vha_w(const float* in, float* out, const float* M, int T, int H, int D, void* stream) {
    if (H != kVhaH || D <= 0 || D % kVhaD) throw std::runtime_error("ernie_vha_w: 64 heads of a multiple of 64 values");
    if (in == out) throw std::runtime_error("ernie_vha_w: out aliases in");
    if (T <= 0) return;
    static const bool split = [] {   // moex (the hill-climb): STRATA_ERNIE_VHA_SPLIT=0 - one block a tile
        const char* e = std::getenv("STRATA_ERNIE_VHA_SPLIT");
        return !(e && e[0] == '0');
    }();
    if (T > 65535) throw std::runtime_error("ernie_vha_w: too many tokens");
    // (decode windows only: a prompt chunk's rows already fill the card with one block a tile, which reads each input tile once)
    if (split && T <= 8) vha_kernel<4><<<dim3(D / kVhaD, kVhaH / 16, T), 256, 0, st(stream)>>>(in, out, M, D);
    else vha_kernel<16><<<dim3(D / kVhaD, 1, T), 256, 0, st(stream)>>>(in, out, M, D);
    check_launch("ernie_vha_w");
}

void ernie_route_w(const float* logits, const float* bias, const float* escale, int E, int k, int32_t* ids, float* wts, int T,
                   void* stream) {
    if (E <= 0 || E > 1024 || k <= 0 || k > 32 || k > E) throw std::runtime_error("ernie_route_w: E <= 1024 and k <= 32");
    if (T <= 0) return;
    static const bool warp = [] {   // moex (P6): STRATA_ERNIE_ROUTE_WARP=0 - the block kernel
        const char* e = std::getenv("STRATA_ERNIE_ROUTE_WARP");
        return !(e && e[0] == '0');
    }();
    if (warp) {
        if (E <= 512) route_warp_kernel<16><<<T, 256, 0, st(stream)>>>(logits, bias, escale, E, k, ids, wts);
        else route_warp_kernel<32><<<T, 256, 0, st(stream)>>>(logits, bias, escale, E, k, ids, wts);
        check_launch("ernie_route_w (warp)");
        return;
    }
    route_kernel<<<T, 256, 0, st(stream)>>>(logits, bias, escale, E, k, ids, wts);
    check_launch("ernie_route_w");
}

bool ernie_route_publish_w(const float* logits, const float* bias, const float* escale, int E, int k, int32_t* ids, float* wts,
                           const float* x, int n_x, int32_t* m_ids, float* m_x, volatile uint32_t* m_seq, uint32_t* d_cnt,
                           int T, void* stream) {
    static const bool on = [] {   // STRATA_ERNIE_ROUTE_PUBLISH=0: the routing and the publish as two launches
        const char* e = std::getenv("STRATA_ERNIE_ROUTE_PUBLISH");
        const char* w = std::getenv("STRATA_ERNIE_ROUTE_WARP");
        return !(e && e[0] == '0') && !(w && w[0] == '0');
    }();
    if (!on || T != 1 || E <= 0 || E > 1024 || k <= 0 || k > 32 || k > E) return false;
    if (E <= 512) route_warp_kernel<16, true><<<1, 256, 0, st(stream)>>>(logits, bias, escale, E, k, ids, wts, x, n_x, m_ids, m_x, m_seq, d_cnt);
    else route_warp_kernel<32, true><<<1, 256, 0, st(stream)>>>(logits, bias, escale, E, k, ids, wts, x, n_x, m_ids, m_x, m_seq, d_cnt);
    check_launch("ernie_route_publish_w");
    return true;
}

void ernie_route_block_w(const float* logits, const float* bias, const float* escale, int E, int k, int32_t* ids, float* wts, int T,
                         void* stream) {   // the block kernel always (the checks' reference)
    if (E <= 0 || E > 1024 || k <= 0 || k > 32 || k > E) throw std::runtime_error("ernie_route_block_w: E <= 1024 and k <= 32");
    if (T <= 0) return;
    route_kernel<<<T, 256, 0, st(stream)>>>(logits, bias, escale, E, k, ids, wts);
    check_launch("ernie_route_block_w");
}

void ernie_multimax_w(float* logits, int64_t V, int rows, int64_t ld, const ErnieMultiMax& mm, void* stream) {
    if (rows <= 0 || V <= 0) return;
    if (rows > 65535) throw std::runtime_error("ernie_multimax_w: too many rows");
    multimax_kernel<<<dim3((unsigned) ((V + 255) / 256), (unsigned) rows), 256, 0, st(stream)>>>(
        logits, V, ld, mm.r[0], mm.r[1], mm.r[2], mm.r[3], mm.t[0], mm.t[1], mm.t[2], mm.t[3]);
    check_launch("ernie_multimax_w");
}

void ernie_fq_e4m3_w(float* x, int64_t groups, void* stream) {
    if (groups <= 0) return;
    fq_e4m3_kernel<<<(unsigned) ((groups + 7) / 8), 256, 0, st(stream)>>>(x, groups);
    check_launch("ernie_fq_e4m3_w");
}

void ernie_dsa_cells_w(const int32_t* sel, int top_k, int window, int32_t* cells, int max_cells, int32_t* counts, int64_t pos0, int T,
                       void* stream, const int64_t* pos_dev) {
    if (max_cells < top_k + window) throw std::runtime_error("ernie_dsa_cells_w: max_cells < top_k + window");
    if (T <= 0) return;
    dsa_cells_kernel<<<T, 256, 0, st(stream)>>>(sel, top_k, window, cells, max_cells, counts, pos0, pos_dev);
    check_launch("ernie_dsa_cells_w");
}

void ernie_rms_eo_w(const float* x, float* y, int rows, int n, float eps, void* stream) {
    if (rows <= 0) return;
    rms_eo_kernel<<<rows, 256, 0, st(stream)>>>(x, y, n, eps);
    check_launch("ernie_rms_eo_w");
}

void ernie_store_rows_w(const float* x, uint16_t* cache, int width, int64_t cache_rows, int64_t pos0, int T, void* stream,
                        const int64_t* pos_dev) {
    if (T <= 0) return;
    store_rows_kernel<<<T, 128, 0, st(stream)>>>(x, reinterpret_cast<__half*>(cache), width, cache_rows, pos0, pos_dev);
    check_launch("ernie_store_rows_w");
}

}  // namespace strata::x
