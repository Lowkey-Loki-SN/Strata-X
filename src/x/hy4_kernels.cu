// src/x/hy4_kernels.cu - moex: Hy4-flash's kernels (include/strata/x/hy4_kernels.hpp; moex/docs/HY4_PORT.md).
//
// The ops of llama.cpp's hy_v4_flash graph (PR #32517, src/models/hy-v4-flash.cpp) that strata-x's GLM-5.3 and
// DeepSeek-V4 kernels do not already compute: iHC's coefficients, RoPE on consecutive pairs, the 576-value MLA cache row,
// the indexer's pooled keys (pool, norm, RoPE), its 16-head scores and Hy4's selection (the best 511 groups before the
// token's own, then its own group), the attention over the cache with sinks, the output gate, and the Engram's math.
// Elementwise steps keep llama.cpp's operation order and roundings (__fmul_rn / __fadd_rn where a contraction would
// change a bit); the reductions are float trees, so they differ from ggml-cpu's (double) sums in the last bits.
#include "strata/x/hy4_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace strata::x {
namespace {

inline cudaStream_t st(void* s) { return static_cast<cudaStream_t>(s); }

inline void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "hy4_kernels: %s: %s\n", what, cudaGetErrorString(e));
        throw std::runtime_error(what);
    }
}

inline int blocks(int64_t n, int bs) { return (int) ((n + bs - 1) / bs); }

__device__ __forceinline__ int64_t pos_of(const int64_t* __restrict__ p, int64_t arg) { return p ? *p : arg; }

/// moex vision: the coordinate pair k (of the 32) of position p rotates by - Hy4-flash's M-RoPE (vLLM's mrope_interleaved,
/// sections [12, 10, 10]: pairs 1, 4, .. 28 take the row h, 2, 5, .. 29 the column w, the rest the time t).  `mpos`
/// (hy4_kernels.hpp Hy4MPos): null, or a header {b, e, shift, 0} then (e - b) triples (t, h, w) for the positions [b, e);
/// a position before b is its own coordinate, one at or past e is p - shift in all three (the text after the pictures).
__device__ __forceinline__ int64_t mrope_coord(const int64_t* __restrict__ mpos, int64_t p, int k) {
    if (!mpos) return p;
    const int64_t b = mpos[0], e = mpos[1];
    if (p < b) return p;
    if (p >= e) return p - mpos[2];
    return reinterpret_cast<const int32_t*>(mpos + 4)[(p - b) * 3 + (k < 30 ? k % 3 : 0)];
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
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
    __syncthreads();   // red / total reused by the next call
    return r;
}

__device__ __forceinline__ float sigmoid_rn(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-x))); }

/// ggml's rms_norm scale: 1 / sqrtf(mean + eps)
__device__ __forceinline__ float rms_scale(float sum_sq, int n, float eps) {
    return __fdiv_rn(1.0f, sqrtf(__fadd_rn(__fdiv_rn(sum_sq, (float) n), eps)));
}

// ------------------------------------------------------------------ iHC

__global__ void ihc_coef_kernel(const float* __restrict__ mixes, const float* __restrict__ sc, const float* __restrict__ bs,
                                float* __restrict__ coef, float eps, float mag, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float* m = mixes + (size_t) t * 8;
    float* c = coef + (size_t) t * 24;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        // ggml_mul, ggml_add, ggml_sigmoid, ggml_scale_bias(1, eps) - and the post side's ggml_scale(mag) before the bias
        c[h] = __fadd_rn(sigmoid_rn(__fadd_rn(__fmul_rn(m[h], sc[0]), bs[h])), eps);
        c[4 + h] = __fadd_rn(__fmul_rn(sigmoid_rn(__fadd_rn(__fmul_rn(m[4 + h], sc[1]), bs[4 + h])), mag), eps);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) c[8 + i] = i % 5 == 0 ? 1.0f : 0.0f;   // comb[dst + 4 src]: the identity
}

// ------------------------------------------------------------------ RoPE (consecutive pairs)

__device__ __forceinline__ void rot_pair(float* v, float theta) {
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = v[0], x1 = v[1];
    v[0] = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
    v[1] = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
}

__global__ void rope_pairs_kernel(float* __restrict__ x, int row_stride, int head_off, int n_heads, int head_dim, int rot_off,
                                  int n_rot, float theta_scale, int64_t pos0, const int64_t* __restrict__ pos_dev,
                                  const int64_t* __restrict__ mpos) {
    const int t = blockIdx.y;
    const int64_t p = pos_of(pos_dev, pos0) + t;
    const int half = n_rot / 2, pairs = n_heads * half;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pairs; i += gridDim.x * blockDim.x) {
        const int h = i / half, k = i % half;
        float* v = x + (size_t) t * row_stride + head_off + (size_t) h * head_dim + rot_off + 2 * k;
        rot_pair(v, (float) mrope_coord(mpos, p, k) * powf(theta_scale, (float) k));
    }
}

// ------------------------------------------------------------------ the MLA cache row

constexpr int kLat = 512, kRope = 64, kRow = kLat + kRope;

__global__ void __launch_bounds__(256) kv_row_kernel(float* __restrict__ kv, const float* __restrict__ nw, float eps,
                                                     float theta_scale, __half* __restrict__ cache, int64_t rows, int64_t pos0,
                                                     const int64_t* __restrict__ pos_dev, const int64_t* __restrict__ mpos) {
    const int t = blockIdx.x, i = threadIdx.x;
    const int64_t p = pos_of(pos_dev, pos0) + t;
    float* r = kv + (size_t) t * kRow;
    const float a = r[i], b = r[i + 256];
    const float scale = rms_scale(block_sum(__fadd_rn(__fmul_rn(a, a), __fmul_rn(b, b))), kLat, eps);
    __half* c = cache ? cache + (size_t) (p % rows) * kRow : nullptr;
    const float ya = __fmul_rn(__fmul_rn(a, scale), nw[i]), yb = __fmul_rn(__fmul_rn(b, scale), nw[i + 256]);
    r[i] = ya;
    r[i + 256] = yb;
    if (c) {
        c[i] = __float2half_rn(ya);
        c[i + 256] = __float2half_rn(yb);
    }
    if (i < kRope / 2) {
        float* v = r + kLat + 2 * i;
        rot_pair(v, (float) mrope_coord(mpos, p, i) * powf(theta_scale, (float) i));
        if (c) {
            c[kLat + 2 * i] = __float2half_rn(v[0]);
            c[kLat + 2 * i + 1] = __float2half_rn(v[1]);
        }
    }
}

__global__ void store_kv_kernel(const float* __restrict__ kv, __half* __restrict__ cache, int64_t rows, int64_t pos0,
                                const int64_t* __restrict__ pos_dev) {
    const int t = blockIdx.x;
    const int64_t p = pos_of(pos_dev, pos0) + t;
    const float* r = kv + (size_t) t * kRow;
    __half* c = cache + (size_t) (p % rows) * kRow;
    for (int i = threadIdx.x; i < kRow; i += blockDim.x) c[i] = __float2half_rn(r[i]);
}

// ------------------------------------------------------------------ the indexer

constexpr int kIdx = 128, kIdxHeads = 16;

/// Block b: the b-th pool completing in the window (a block with none returns).
__global__ void __launch_bounds__(kIdx) kpool_kernel(const float* __restrict__ iv, const float* __restrict__ ig, int ring_rows,
                                                     const float* __restrict__ ape, const float* __restrict__ nw, float eps,
                                                     float* __restrict__ pooled, float theta_scale, int64_t pos0_arg,
                                                     const int64_t* __restrict__ pos_dev, int T, int kp,
                                                     const int64_t* __restrict__ mpos) {
    __shared__ float sv[kIdx];
    const int d = threadIdx.x;
    const int64_t pos0 = pos_of(pos_dev, pos0_arg);
    const int64_t p_first = pos0 + ((kp - (pos0 + 1) % kp) % kp);   // the first position in the window that ends a group
    const int64_t p_last = p_first + (int64_t) blockIdx.x * kp;
    if (p_last >= pos0 + T) return;   // (uniform over the block)
    const int64_t P = (p_last + 1) / kp - 1;
    float lg[8];
    float mx = -INFINITY;
    for (int k = 0; k < kp; ++k) {
        const size_t row = (size_t) ((P * kp + k) % ring_rows);
        lg[k] = __fadd_rn(ig[row * kIdx + d], ape[d + (size_t) kIdx * k]);
        mx = fmaxf(mx, lg[k]);
    }
    float sum = 0.0f;
    for (int k = 0; k < kp; ++k) {
        lg[k] = expf(lg[k] - mx);
        sum += lg[k];
    }
    const float inv = __fdiv_rn(1.0f, sum);
    float acc = 0.0f;
    for (int k = 0; k < kp; ++k)
        acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(lg[k], inv), iv[(size_t) ((P * kp + k) % ring_rows) * kIdx + d]));
    const float scale = rms_scale(block_sum(__fmul_rn(acc, acc)), kIdx, eps);
    sv[d] = __fmul_rn(__fmul_rn(acc, scale), nw[d]);
    __syncthreads();
    if (d < kRope / 2) rot_pair(sv + (kIdx - kRope) + 2 * d, (float) mrope_coord(mpos, P * kp, d) * powf(theta_scale, (float) d));
    __syncthreads();
    pooled[(size_t) P * kIdx + d] = sv[d];
}

/// grid (pool blocks, T), 4 warps a block, a pool a warp at a time: lane = (head h = lane % 16, half = lane / 16), its
/// head's 64 query values of that half in registers; the two halves' dots joined, relu, times the head's weight, the 16
/// heads summed.
__global__ void __launch_bounds__(128) scores_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                                     const float* __restrict__ w, float* __restrict__ score, int64_t pos0_arg,
                                                     const int64_t* __restrict__ pos_dev, int kp, int max_pool) {
    __shared__ float key[4][2 * 65];
    const int t = blockIdx.y;
    const int64_t p = pos_of(pos_dev, pos0_arg) + t;
    const int n_pool = (int) (p / kp);
    if ((int) blockIdx.x * 4 >= n_pool) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = lane & 15, half = lane >> 4;
    float qr[64];
    const float* qs = iq + ((size_t) t * kIdxHeads + h) * kIdx + half * 64;
#pragma unroll
    for (int d = 0; d < 64; ++d) qr[d] = qs[d];
    const float wh = w[(size_t) t * kIdxHeads + h];
    float* sc = score + (size_t) t * max_pool;
    float* kw = key[warp];
    for (int P = blockIdx.x * 4 + warp; P < n_pool; P += gridDim.x * 4) {
        const float* src = pooled + (size_t) P * kIdx;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = lane + 32 * j;
            kw[(d >> 6) * 65 + (d & 63)] = src[d];
        }
        __syncwarp();
        const float* kk = kw + half * 65;
        float acc = 0.0f;
#pragma unroll 16
        for (int d = 0; d < 64; ++d) acc = fmaf(qr[d], kk[d], acc);
        acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 16));
        float sv = __fmul_rn(fmaxf(acc, 0.0f), wh);
        sv = __fadd_rn(sv, __shfl_xor_sync(0xffffffffu, sv, 8));
        sv = __fadd_rn(sv, __shfl_xor_sync(0xffffffffu, sv, 4));
        sv = __fadd_rn(sv, __shfl_xor_sync(0xffffffffu, sv, 2));
        sv = __fadd_rn(sv, __shfl_xor_sync(0xffffffffu, sv, 1));
        if (lane == 0) sc[P] = sv;
        __syncwarp();
    }
}

/// The same scores for a decode window (T <= 8 rows), bit for bit: each warp takes a tile of G pools into shared memory once
/// and every row of the window scores them from there - G independent dots a lane interleaved (each the same sequential
/// fmaf chain over its 64 values as scores_kernel's, the same shuffle tree after it), so the dependent chains overlap and a
/// pool's 512 bytes are read once a window instead of once a row.
template <int G>
__global__ void __launch_bounds__(128) scores_tile_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                                          const float* __restrict__ w, float* __restrict__ score, int64_t pos0_arg,
                                                          const int64_t* __restrict__ pos_dev, int kp, int max_pool, int T) {
    __shared__ float key[4][G][2 * 65];
    const int64_t pos0 = pos_of(pos_dev, pos0_arg);
    const int n_last = (int) ((pos0 + T - 1) / kp);   // the window's last row scores the most pools
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = lane & 15, half = lane >> 4;
    float (*kw)[2 * 65] = key[warp];
    for (int P0 = (blockIdx.x * 4 + warp) * G; P0 < n_last; P0 += gridDim.x * 4 * G) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const int P = P0 + g;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = lane + 32 * j;
                kw[g][(d >> 6) * 65 + (d & 63)] = P < n_last ? pooled[(size_t) P * kIdx + d] : 0.0f;
            }
        }
        __syncwarp();
        for (int t = 0; t < T; ++t) {
            const int n_pool = (int) ((pos0 + t) / kp);
            if (P0 >= n_pool) continue;   // (uniform over the warp)
            float qr[64];
            const float* qs = iq + ((size_t) t * kIdxHeads + h) * kIdx + half * 64;
#pragma unroll
            for (int d = 0; d < 64; ++d) qr[d] = qs[d];
            const float wh = w[(size_t) t * kIdxHeads + h];
            float acc[G];
#pragma unroll
            for (int g = 0; g < G; ++g) acc[g] = 0.0f;
#pragma unroll 16
            for (int d = 0; d < 64; ++d)
#pragma unroll
                for (int g = 0; g < G; ++g) acc[g] = fmaf(qr[d], kw[g][half * 65 + d], acc[g]);
            float* sc = score + (size_t) t * max_pool;
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float a = __fadd_rn(acc[g], __shfl_xor_sync(0xffffffffu, acc[g], 16));
                float sv = __fmul_rn(fmaxf(a, 0.0f), wh);
                sv = __fadd_rn(sv, __shfl_xor_sync(0xffffffffu, sv, 8));
                sv = __fadd_rn(sv, __shfl_xor_sync(0xffffffffu, sv, 4));
                sv = __fadd_rn(sv, __shfl_xor_sync(0xffffffffu, sv, 2));
                sv = __fadd_rn(sv, __shfl_xor_sync(0xffffffffu, sv, 1));
                if (lane == 0 && P0 + g < n_pool) sc[P0 + g] = sv;
            }
        }
        __syncwarp();
    }
}

/// The same scores again, bit for bit, register-tiled (2026-11-19): a block takes BT rows x BP pools with BT * WP warps,
/// warp (r, ps) row r and PW = 8 TN pools; lane bit 4 is the half (values 0-63 / 64-127), bit 3 the head group (heads
/// 0-7 / 8-15), bits 0-2 TN consecutive pools.  Each (row, head, half, pool) stays one thread's sequential fmaf chain over
/// its 64 values from 0.0f, as scores_kernel's lane; the halves are joined lo + hi, relu, times the head's weight, and the 16
/// heads summed in that kernel's shuffle tree (h + (h ^ 8), then ^ 4, ^ 2, ^ 1, each own + other) - so every score is its
/// bits.  The values come 16 at a time through shared memory, transposed so a thread's 8 heads and TN pools are 16-byte
/// reads: 8 TN products a step from 8 + TN values (scores_tile_kernel: one), and a pool tile read once for all the rows (the
/// row tiles of a pool tile are neighbouring blocks).  hy4_idx_parity-style check: 17 cases (1 .. 512 rows, 4 .. 1,048,569
/// positions, the device position, ties and zero pools) give every score bit for bit (scratchpad idx3.cu, 2026-11-19).
template <int BT, int WP, int TN>
__global__ void __launch_bounds__(BT * WP * 32) scores_mm_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                                                const float* __restrict__ w, float* __restrict__ score,
                                                                int64_t pos0_arg, const int64_t* __restrict__ pos_dev, int kp,
                                                                int max_pool, int T) {
    constexpr int NW = BT * WP, NTH = NW * 32, PW = 8 * TN, BP = PW * WP, DC = 16, BPS = BP + 4;
    __shared__ __align__(16) float Ks[2][DC][BPS];
    __shared__ __align__(16) float Qs[BT][2][DC][16];
    const int64_t pos0 = pos_of(pos_dev, pos0_arg);
    const int n_last = (int) ((pos0 + T - 1) / kp);   // the last row scores the most pools
    const int tid = (int) threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int hf = lane >> 4, hg = (lane >> 3) & 1, pg = lane & 7;
    const int r = warp / WP, ps = warp % WP;
    const int n_rt = (T + BT - 1) / BT;
    for (int P0 = (int) blockIdx.y * BP; P0 < n_last; P0 += (int) gridDim.y * BP)
        for (int rt = (int) blockIdx.x; rt < n_rt; rt += (int) gridDim.x) {
            const int t = rt * BT + r;
            float acc[8][TN];
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
#pragma unroll 1
            for (int c = 0; c < 64 / DC; ++c) {
                __syncthreads();   // the previous chunk is read
                for (int e = tid; e < BP * 2 * (DC / 4); e += NTH) {   // a pool's 2 x 16 values: 8 neighbouring threads
                    const int p = e / (2 * (DC / 4)), h2 = (e / (DC / 4)) & 1, q4 = e % (DC / 4);
                    const int P = P0 + p;
                    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    if (P < n_last) v = *reinterpret_cast<const float4*>(pooled + (size_t) P * kIdx + h2 * 64 + c * DC + q4 * 4);
                    Ks[h2][q4 * 4 + 0][p] = v.x;
                    Ks[h2][q4 * 4 + 1][p] = v.y;
                    Ks[h2][q4 * 4 + 2][p] = v.z;
                    Ks[h2][q4 * 4 + 3][p] = v.w;
                }
                for (int e = tid; e < BT * 16 * 2 * (DC / 4); e += NTH) {
                    const int rr = e / (16 * 2 * (DC / 4)), h = (e / (2 * (DC / 4))) & 15, h2 = (e / (DC / 4)) & 1, q4 = e % (DC / 4);
                    const int tt = rt * BT + rr;
                    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    if (tt < T) v = *reinterpret_cast<const float4*>(iq + ((size_t) tt * kIdxHeads + h) * kIdx + h2 * 64 + c * DC + q4 * 4);
                    Qs[rr][h2][q4 * 4 + 0][h] = v.x;
                    Qs[rr][h2][q4 * 4 + 1][h] = v.y;
                    Qs[rr][h2][q4 * 4 + 2][h] = v.z;
                    Qs[rr][h2][q4 * 4 + 3][h] = v.w;
                }
                __syncthreads();
#pragma unroll
                for (int dd = 0; dd < DC; ++dd) {
                    const float4 qa = *reinterpret_cast<const float4*>(&Qs[r][hf][dd][hg * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&Qs[r][hf][dd][hg * 8 + 4]);
                    const float qv[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
                    float kv[TN];
#pragma unroll
                    for (int j4 = 0; j4 < TN / 4; ++j4) {
                        const float4 k4 = *reinterpret_cast<const float4*>(&Ks[hf][dd][ps * PW + pg * TN + 4 * j4]);
                        kv[4 * j4 + 0] = k4.x;
                        kv[4 * j4 + 1] = k4.y;
                        kv[4 * j4 + 2] = k4.z;
                        kv[4 * j4 + 3] = k4.w;
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i)
#pragma unroll
                        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(qv[i], kv[j], acc[i][j]);
                }
            }
            if (t < T) {   // (uniform over the warp)
                const int n_pool = (int) ((pos0 + t) / kp);
                float wv[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) wv[i] = w[(size_t) t * kIdxHeads + hg * 8 + i];
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) {   // lo + hi (the half-0 lane's own + other), relu, the weight
                        const float a = __fadd_rn(acc[i][j], __shfl_xor_sync(0xffffffffu, acc[i][j], 16));
                        acc[i][j] = __fmul_rn(fmaxf(a, 0.0f), wv[i]);
                    }
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] = __fadd_rn(acc[i][j], __shfl_xor_sync(0xffffffffu, acc[i][j], 8));
                if (hf == 0 && hg == 0) {   // head i now holds s_i + s_(i+8); then ^ 4, ^ 2, ^ 1 as lane 0 adds them
                    float* sc = score + (size_t) t * max_pool;
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        const float a0 = __fadd_rn(acc[0][j], acc[4][j]), a1 = __fadd_rn(acc[1][j], acc[5][j]);
                        const float a2 = __fadd_rn(acc[2][j], acc[6][j]), a3 = __fadd_rn(acc[3][j], acc[7][j]);
                        const float b0 = __fadd_rn(a0, a2), b1 = __fadd_rn(a1, a3);
                        const int P = P0 + ps * PW + pg * TN + j;
                        if (P < n_pool) sc[P] = __fadd_rn(b0, b1);
                    }
                }
            }
        }
}

__device__ __forceinline__ uint32_t okey(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// Exclusive prefix sum of one value per thread over the block (blockDim.x a multiple of 32, at most 1024); `tmp` holds
/// 33 words of shared memory.  Every thread must call it.  (kernels.cu's, for the same deterministic placement.)
__device__ uint32_t block_excl_scan(uint32_t v, uint32_t* tmp) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (int) (blockDim.x >> 5);
    uint32_t x = v;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) x += y;
    }
    __syncthreads();
    if (lane == 31) tmp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        uint32_t w = lane < nw ? tmp[lane] : 0u;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, w, d);
            if (lane >= d) w += y;
        }
        if (lane < nw) tmp[lane] = w;
    }
    __syncthreads();
    return (warp ? tmp[warp - 1] : 0u) + x - v;
}

/// The selected groups expanded to their kp cells in group order, ties at the threshold the lowest group first
/// (kernels.cu's select_ordered).
__device__ void select_ordered(const float* __restrict__ sc, int n, uint32_t thr, uint32_t remaining, int kp,
                               int32_t* __restrict__ out, uint32_t* scan_tmp) {
    const int per = (n + (int) blockDim.x - 1) / (int) blockDim.x;
    const int i0 = (int) threadIdx.x * per, i1 = min(n, i0 + per);
    uint32_t my_gt = 0, my_eq = 0;
    for (int i = i0; i < i1; ++i) {
        const uint32_t key = okey(sc[i]);
        my_gt += key > thr;
        my_eq += key == thr;
    }
    const uint32_t eq_before = block_excl_scan(my_eq, scan_tmp);
    const uint32_t take_eq = eq_before >= remaining ? 0u : min(my_eq, remaining - eq_before);
    uint32_t o = block_excl_scan(my_gt + take_eq, scan_tmp), eq_seen = 0;
    for (int i = i0; i < i1; ++i) {
        const uint32_t key = okey(sc[i]);
        if (key > thr || (key == thr && eq_seen++ < take_eq)) {
            for (int m = 0; m < kp; ++m) out[o * kp + m] = i * kp + m;
            ++o;
        }
    }
}

/// One block per token: its groups before its own (all when they fit the budget, else the radix top n_top), then its own
/// group's positions.
__global__ void __launch_bounds__(1024) select_kernel(const float* __restrict__ score, int max_pool, int n_top, int kp,
                                                      int64_t pos0_arg, const int64_t* __restrict__ pos_dev,
                                                      int32_t* __restrict__ cells, int max_cells) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_digit, s_rem;
    __shared__ uint32_t scan_tmp[33];
    const int t = blockIdx.x;
    const int64_t pos = pos_of(pos_dev, pos0_arg) + t;
    const int n_pool = (int) (pos / kp);
    const int n_sel = n_pool < n_top ? n_pool : n_top;
    int32_t* out = cells + (size_t) t * max_cells;
    const float* sc = score + (size_t) t * max_pool;
    if (n_pool <= n_top) {
        for (int i = threadIdx.x; i < n_sel * kp; i += blockDim.x) out[i] = i;
    } else {
        uint32_t prefix = 0, mask = 0, remaining = (uint32_t) n_top;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int i = threadIdx.x; i < n_pool; i += blockDim.x) {
                const uint32_t key = okey(sc[i]);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                uint32_t cum = 0;
                for (int dg = 255; dg >= 0; --dg) {
                    if (cum + hist[dg] >= remaining) {
                        s_digit = (uint32_t) dg;
                        s_rem = remaining - cum;
                        break;
                    }
                    cum += hist[dg];
                }
            }
            __syncthreads();
            prefix |= s_digit << shift;
            mask |= 255u << shift;
            remaining = s_rem;
            __syncthreads();
        }
        select_ordered(sc, n_pool, prefix, remaining, kp, out, scan_tmp);
    }
    const int n_tail = (int) (pos % kp) + 1;   // the token's own group: kp * (pos / kp) .. pos
    if ((int) threadIdx.x < n_tail) out[n_sel * kp + threadIdx.x] = (int32_t) (pos - n_tail + 1 + threadIdx.x);
}

/// Descending bitonic sort of n (a power of two) values in shared memory; every thread of the block calls it.
__device__ __forceinline__ void bitonic_desc(unsigned long long* v, int n) {
    for (int k = 2; k <= n; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = (int) threadIdx.x; i < n / 2; i += (int) blockDim.x) {
                const int lo = 2 * i - (i & (j - 1)), hi = lo + j;   // the pair (lo, lo + j), lo's bit j clear
                const bool desc = (lo & k) == 0;
                const unsigned long long a = v[lo], b = v[hi];
                if ((a < b) == desc) {
                    v[lo] = b;
                    v[hi] = a;
                }
            }
            __syncthreads();
        }
}

constexpr int kSelCap = 4096, kSelSamp = 4096, kSelAim = 2000;

/// select_kernel's cells with the scores read once (2026-11-19): tau = the K-th largest of 4,096 evenly spaced keys, K
/// aiming at 2,000 keys >= tau; every key >= tau goes with its group into shared memory, and when there are n_top .. 4,096
/// of them, the best n_top of those - key descending, group ascending: select_kernel's tie rule - are the best n_top of all:
/// sorted there and written in group order.  Otherwise (a miss: fewer than n_top or more than 4,096 keys >= tau) and below
/// 32,768 groups, select_kernel's radix passes.  The same cells either way (idx3.cu: 17 cases, ties and zero pools); a miss
/// only costs time.  A row's 1 MiB of scores is read four times by the radix passes, once here: 0.31 -> 0.10 ms a row at
/// 819,200 positions.
__global__ void __launch_bounds__(1024) select_sampled_kernel(const float* __restrict__ score, int max_pool, int n_top, int kp,
                                                              int64_t pos0_arg, const int64_t* __restrict__ pos_dev,
                                                              int32_t* __restrict__ cells, int max_cells) {
    __shared__ unsigned long long cand[kSelCap];
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_digit, s_rem, s_cnt;
    __shared__ uint32_t scan_tmp[33];
    const int t = blockIdx.x;
    const int64_t pos = pos_of(pos_dev, pos0_arg) + t;
    const int n_pool = (int) (pos / kp);
    const int n_sel = n_pool < n_top ? n_pool : n_top;
    int32_t* out = cells + (size_t) t * max_cells;
    const float* sc = score + (size_t) t * max_pool;
    const int tid = (int) threadIdx.x, lane = tid & 31, nt = (int) blockDim.x;
    bool done = false;
    if (n_pool <= n_top) {
        for (int i = tid; i < n_sel * kp; i += nt) out[i] = i;
        done = true;
    } else if (n_pool >= 8 * kSelSamp && n_top <= kSelCap / 2) {
        for (int i = tid; i < kSelSamp; i += nt)
            cand[i] = ((unsigned long long) okey(sc[(int) (((int64_t) i * n_pool) / kSelSamp)]) << 32) | (uint32_t) ~(uint32_t) i;
        if (tid == 0) s_cnt = 0;
        __syncthreads();
        bitonic_desc(cand, kSelSamp);
        const int64_t kq = ((int64_t) kSelAim * kSelSamp) / n_pool;
        const int K = kq < 1 ? 1 : kq > kSelSamp ? kSelSamp : (int) kq;
        const uint32_t tau = (uint32_t) (cand[K - 1] >> 32);
        __syncthreads();
        // every key >= tau: a warp's lanes adjacent, one atomicAdd a warp a step, four float4 a thread in flight; the keys
        // before the row's first 16-byte boundary and after its last whole float4 one at a time
        const int hd0 = (int) (((16u - (unsigned) (reinterpret_cast<uintptr_t>(sc) & 15u)) & 15u) >> 2);
        const int hd = hd0 < n_pool ? hd0 : n_pool;
        const int n4 = (n_pool - hd) >> 2;
        const float4* s4 = reinterpret_cast<const float4*>(sc + hd);
        for (int b = 0; b < n4; b += 4 * nt) {
            float4 v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int i = b + u * nt + tid;
                v[u] = i < n4 ? s4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int i = b + u * nt + tid;
                const float e[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint32_t key = okey(e[c]);
                    const bool take = i < n4 && key >= tau;
                    const unsigned bl = __ballot_sync(0xffffffffu, take);
                    if (bl) {
                        uint32_t base = 0;
                        if (lane == 0) base = atomicAdd(&s_cnt, (uint32_t) __popc(bl));
                        base = __shfl_sync(0xffffffffu, base, 0);
                        const uint32_t k = base + (uint32_t) __popc(bl & ((1u << lane) - 1u));
                        if (take && k < (uint32_t) kSelCap)
                            cand[k] = ((unsigned long long) key << 32) | (uint32_t) ~(uint32_t) (hd + 4 * i + c);
                    }
                }
            }
        }
        for (int j = tid; j < hd + (n_pool - hd - 4 * n4); j += nt) {   // the head, then the tail
            const int i = j < hd ? j : hd + 4 * n4 + (j - hd);
            const uint32_t key = okey(sc[i]);
            if (key >= tau) {
                const uint32_t k = atomicAdd(&s_cnt, 1u);
                if (k < (uint32_t) kSelCap) cand[k] = ((unsigned long long) key << 32) | (uint32_t) ~(uint32_t) i;
            }
        }
        __syncthreads();
        const uint32_t cnt = s_cnt;
        if (cnt >= (uint32_t) n_top && cnt <= (uint32_t) kSelCap) {
            int np = 1;
            while (np < (int) cnt) np <<= 1;
            for (int i = (int) cnt + tid; i < np; i += nt) cand[i] = 0ull;   // below every entry (a group's ~index is >= 2^31)
            __syncthreads();
            bitonic_desc(cand, np);
            // the best n_top's groups ascending: their ~index words sorted descending, in the upper half
            int nq = 1;
            while (nq < n_top) nq <<= 1;
            for (int i = tid; i < nq; i += nt) cand[kSelCap / 2 + i] = i < n_top ? (unsigned long long) (uint32_t) cand[i] : 0ull;
            __syncthreads();
            bitonic_desc(cand + kSelCap / 2, nq);
            for (int i = tid; i < n_top; i += nt) {
                const int g = (int) ~(uint32_t) cand[kSelCap / 2 + i];
                for (int m = 0; m < kp; ++m) out[i * kp + m] = g * kp + m;
            }
            done = true;
        }
        __syncthreads();
    }
    if (!done) {   // select_kernel's radix passes
        uint32_t prefix = 0, mask = 0, remaining = (uint32_t) n_top;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = tid; i < 256; i += nt) hist[i] = 0;
            __syncthreads();
            for (int i = tid; i < n_pool; i += nt) {
                const uint32_t key = okey(sc[i]);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (tid == 0) {
                uint32_t cum = 0;
                for (int dg = 255; dg >= 0; --dg) {
                    if (cum + hist[dg] >= remaining) {
                        s_digit = (uint32_t) dg;
                        s_rem = remaining - cum;
                        break;
                    }
                    cum += hist[dg];
                }
            }
            __syncthreads();
            prefix |= s_digit << shift;
            mask |= 255u << shift;
            remaining = s_rem;
            __syncthreads();
        }
        select_ordered(sc, n_pool, prefix, remaining, kp, out, scan_tmp);
    }
    const int n_tail = (int) (pos % kp) + 1;   // the token's own group
    if (tid < n_tail) out[n_sel * kp + tid] = (int32_t) (pos - n_tail + 1 + tid);
}

// ------------------------------------------------------------------ the attention

constexpr int kAc = 32;   ///< cells a tile
constexpr int kAh = 16;   ///< heads a block (2 a warp)
constexpr int kPart = kLat + 2;

__device__ __forceinline__ int n_cells_of(int64_t p, int n_top, int kp, int window, bool sparse) {
    if (sparse) return min((int) (p / kp), n_top) * kp + (int) (p % kp) + 1;
    return (int) (p + 1 < (int64_t) window ? p + 1 : (int64_t) window);
}

/// grid (chunks, H / 16, T), 256 threads.  n_chunks == 1: the block walks all of the token's cells and writes the
/// normalized output (sink included); else block `chunk` takes cells [32 chunk, 32 chunk + 32) into the partials.
__global__ void __launch_bounds__(256) attend_kernel(const float* __restrict__ q, const __half* __restrict__ cache, int64_t rows,
                                                     const int32_t* __restrict__ cells, int max_cells, int n_top, int kp, int window,
                                                     const float* __restrict__ sinks, float scale, float* __restrict__ ctx,
                                                     float* __restrict__ partial, int n_chunks, int H, int64_t pos0_arg,
                                                     const int64_t* __restrict__ pos_dev, const int32_t* __restrict__ counts) {
    __shared__ __align__(16) __half tile[kAc][kRow];
    const int chunk = blockIdx.x, hg = blockIdx.y, t = blockIdx.z;
    const int64_t p = pos_of(pos_dev, pos0_arg) + t;
    const bool sparse = cells != nullptr;
    const int nc_all = counts ? counts[t] : n_cells_of(p, n_top, kp, window, sparse);   // (counts: DFly2's rows)
    int c_begin = 0, c_end = nc_all;
    if (n_chunks > 1) {
        c_begin = chunk * kAc;
        c_end = min(nc_all, c_begin + kAc);
        if (c_begin >= c_end) return;   // past this token's cells (the merge reads only its own chunks)
    }
    const int64_t first = sparse ? 0 : p - nc_all + 1;
    const int32_t* cl = sparse ? cells + (size_t) t * max_cells : nullptr;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int PK = kRow / 32, PV = kLat / 32;   // 18 and 16 values a lane
    float qr[2][PK], acc[2][PV], m[2], l[2];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        const int h = hg * kAh + warp * 2 + hh;
        const float* qh = q + ((size_t) t * H + h) * kRow;
#pragma unroll
        for (int i = 0; i < PK; ++i) qr[hh][i] = qh[lane + 32 * i];
#pragma unroll
        for (int i = 0; i < PV; ++i) acc[hh][i] = 0.0f;
        m[hh] = -INFINITY;
        l[hh] = 0.0f;
    }
    for (int c0 = c_begin; c0 < c_end; c0 += kAc) {
        const int nc = min(kAc, c_end - c0);
        __syncthreads();   // the previous tile is read
        for (int e = threadIdx.x; e < kAc * (kRow / 8); e += 256) {
            const int c = e / (kRow / 8), part = e % (kRow / 8);
            uint4 val = make_uint4(0, 0, 0, 0);
            if (c < nc) {
                const int64_t pos = sparse ? (int64_t) cl[c0 + c] : first + c0 + c;
                val = reinterpret_cast<const uint4*>(cache + (size_t) (pos % rows) * kRow)[part];
            }
            reinterpret_cast<uint4*>(&tile[c][0])[part] = val;
        }
        __syncthreads();
        for (int c = 0; c < nc; ++c) {
            float kv[PK];
#pragma unroll
            for (int i = 0; i < PK; ++i) kv[i] = __half2float(tile[c][lane + 32 * i]);
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                float sv = 0.0f;
#pragma unroll
                for (int i = 0; i < PK; ++i) sv = fmaf(qr[hh][i], kv[i], sv);
                sv = warp_sum(sv) * scale;
                const float mn = fmaxf(m[hh], sv);
                const float corr = expf(m[hh] - mn), pr = expf(sv - mn);
                l[hh] = l[hh] * corr + pr;
#pragma unroll
                for (int i = 0; i < PV; ++i) acc[hh][i] = acc[hh][i] * corr + pr * kv[i];
                m[hh] = mn;
            }
        }
    }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        const int h = hg * kAh + warp * 2 + hh;
        if (n_chunks == 1) {
            float M = m[hh], L = l[hh];
            if (sinks) {
                const float sk = sinks[h];
                M = fmaxf(m[hh], sk);
                L = l[hh] * expf(m[hh] - M) + expf(sk - M);
            }
            const float f = expf(m[hh] - M) / L;
            float* o = ctx + ((size_t) t * H + h) * kLat;
#pragma unroll
            for (int i = 0; i < PV; ++i) o[lane + 32 * i] = acc[hh][i] * f;
        } else {
            float* o = partial + (((size_t) t * H + h) * n_chunks + chunk) * kPart;
#pragma unroll
            for (int i = 0; i < PV; ++i) o[lane + 32 * i] = acc[hh][i];
            if (lane == 0) {
                o[kLat] = m[hh];
                o[kLat + 1] = l[hh];
            }
        }
    }
}

/// grid (H, T): a token's chunks joined, its head's sink in the denominator.
__global__ void __launch_bounds__(256) attend_merge_kernel(const float* __restrict__ partial, const float* __restrict__ sinks,
                                                           float* __restrict__ ctx, int n_chunks, int H, int n_top, int kp,
                                                           int window, int sparse, int64_t pos0_arg,
                                                           const int64_t* __restrict__ pos_dev, const int32_t* __restrict__ counts) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int64_t p = pos_of(pos_dev, pos0_arg) + t;
    const int nc = ((counts ? counts[t] : n_cells_of(p, n_top, kp, window, sparse != 0)) + kAc - 1) / kAc;
    const float* base = partial + ((size_t) t * H + h) * n_chunks * kPart;
    float M = sinks ? sinks[h] : -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, base[(size_t) c * kPart + kLat]);
    float L = sinks ? expf(sinks[h] - M) : 0.0f;
    for (int c = 0; c < nc; ++c) L += expf(base[(size_t) c * kPart + kLat] - M) * base[(size_t) c * kPart + kLat + 1];
    const float inv = 1.0f / L;
    float* out = ctx + ((size_t) t * H + h) * kLat;
    for (int d = threadIdx.x; d < kLat; d += 256) {
        float a = 0.0f;
        for (int c = 0; c < nc; ++c) a += expf(base[(size_t) c * kPart + kLat] - M) * base[(size_t) c * kPart + d];
        out[d] = a * inv;
    }
}

__global__ void q_split_kernel(const float* __restrict__ q, float* __restrict__ qn, float* __restrict__ qf, int H, int T) {
    const int h = blockIdx.x, t = blockIdx.y;
    const float* src = q + ((size_t) t * H + h) * 256;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        if (i < 192) qn[((size_t) h * T + t) * 192 + i] = src[i];
        else qf[((size_t) t * H + h) * kRow + kLat + (i - 192)] = src[i];
    }
}

__global__ void q_join_kernel(const float* __restrict__ qa, float* __restrict__ qf, int H, int T) {
    const int h = blockIdx.x, t = blockIdx.y;
    for (int i = threadIdx.x; i < kLat; i += blockDim.x) qf[((size_t) t * H + h) * kRow + i] = qa[((size_t) h * T + t) * kLat + i];
}

__global__ void gate_mul_kernel(float* __restrict__ o, const float* __restrict__ g, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __fmul_rn(o[i], sigmoid_rn(g[i]));
}

// ------------------------------------------------------------------ the Engram

/// OCP FP8 E4M3 (FN): 1 sign, 4 exponent (bias 7), 3 mantissa bits; exponent 0 the subnormals m x 2^-9; 0x7F / 0xFF NaN.
__device__ __forceinline__ float e4m3_f(uint32_t c) {
    const uint32_t e = (c >> 3) & 15u, m = c & 7u;
    float v;
    if (e == 0) v = (float) m * 0.001953125f;
    else if (e == 15u && m == 7u) v = __int_as_float(0x7fc00000);
    else v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (c & 0x80u) ? -v : v;
}

/// a table row's value d: fmt 0 F8_E4M3 (a byte a value), fmt 1 ggml's Q8_0 (blocks of 32: an FP16 d, 32 int8 codes - d x q,
/// exact in F32: dequantize_row_q8_0's value)
__device__ __forceinline__ float eg_value(const uint8_t* __restrict__ row, int fmt, int d) {
    if (fmt == 0) return e4m3_f(row[d]);
    const uint8_t* b = row + (size_t) (d / 32) * 34;
    const float sc = __half2float(__ushort_as_half((unsigned short) (b[0] | (b[1] << 8))));
    return __fmul_rn((float) (int8_t) b[2 + d % 32], sc);
}

__global__ void eg_rows_kernel(const uint8_t* __restrict__ rows, int fmt, const float* __restrict__ gamma, const float* __restrict__ valid,
                               float* __restrict__ out, int cols, int dim, float eps) {
    const int c = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const size_t at = ((size_t) t * cols + c) * dim;
    const size_t rb = fmt == 0 ? (size_t) dim : (size_t) (dim / 32) * 34;
    const float x = d < dim ? eg_value(rows + ((size_t) t * cols + c) * rb, fmt, d) : 0.0f;
    const float scale = rms_scale(block_sum(__fmul_rn(x, x)), dim, eps);
    if (d < dim) out[at + d] = __fmul_rn(__fmul_rn(__fmul_rn(x, scale), gamma[(size_t) c * dim + d]), valid[(size_t) t * cols + c]);
}

__global__ void __launch_bounds__(1024) eg_apply_kernel(float* __restrict__ R, const float* __restrict__ kv,
                                                        const float* __restrict__ kvb, const float* __restrict__ egate,
                                                        const float* __restrict__ pg, const float* __restrict__ hcs,
                                                        const float* __restrict__ valid, int cols, int n, float eps,
                                                        float eps_qk) {
    __shared__ float s_any;
    const int t = blockIdx.x;
    if (threadIdx.x == 0) {
        float a = 0.0f;
        for (int c = 0; c < cols; ++c) a += valid[(size_t) t * cols + c];
        s_any = a > 0.0f ? 1.0f : 0.0f;   // ggml_step(sum_rows(valid))
    }
    __syncthreads();
    const float any = s_any;
    float* r = R + (size_t) t * 4 * n;
    const float* k = kv + (size_t) t * 5 * n;
    float ss[9];
#pragma unroll
    for (int j = 0; j < 9; ++j) ss[j] = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float x = r[(size_t) c * n + i], y = __fadd_rn(k[(size_t) c * n + i], kvb[(size_t) c * n + i]);
            ss[c] = fmaf(x, x, ss[c]);
            ss[4 + c] = fmaf(y, y, ss[4 + c]);
        }
        const float v = __fadd_rn(k[(size_t) 4 * n + i], kvb[(size_t) 4 * n + i]);
        ss[8] = fmaf(v, v, ss[8]);
    }
    float sc[9];
#pragma unroll
    for (int j = 0; j < 9; ++j) sc[j] = rms_scale(block_sum(ss[j]), n, j < 8 ? eps_qk : eps);   // (the streams and keys: norm_query / norm_key)
    float dt[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float xn = __fmul_rn(r[(size_t) c * n + i], sc[c]);
            const float kn = __fmul_rn(__fadd_rn(k[(size_t) c * n + i], kvb[(size_t) c * n + i]), sc[4 + c]);
            dt[c] = __fadd_rn(dt[c], __fmul_rn(__fmul_rn(xn, kn), egate[(size_t) c * n + i]));
        }
    }
    float g[4];
    const float inv_sqrt_n = __fdiv_rn(1.0f, sqrtf((float) n));
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const float d = __fmul_rn(block_sum(dt[c]), inv_sqrt_n);
        const float mag = sqrtf(fmaxf(fabsf(d), 1e-6f));
        g[c] = sigmoid_rn(d < 0.0f ? -mag : mag);   // sign(0) = +1 (ggml_step of -dot)
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float sig = __fmul_rn(__fmul_rn(__fadd_rn(k[(size_t) 4 * n + i], kvb[(size_t) 4 * n + i]), sc[8]), pg[i]);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float delta = __fmul_rn(__fmul_rn(sig, __fadd_rn(g[c], hcs[(size_t) c * n + i])), any);
            r[(size_t) c * n + i] = __fadd_rn(r[(size_t) c * n + i], delta);
        }
    }
}

}  // namespace

// ------------------------------------------------------------------ the launches

void hy4_ihc_coef_w(const float* mixes, const float* scale2, const float* base8, float* coef, float eps, float mag, int T,
                    void* stream) {
    if (T <= 0) return;
    ihc_coef_kernel<<<blocks(T, 64), 64, 0, st(stream)>>>(mixes, scale2, base8, coef, eps, mag, T);
    check_launch("hy4_ihc_coef_w");
}

void hy4_rope_w(float* x, int row_stride, int head_off, int n_heads, int head_dim, int rot_off, int n_rot, float base, int64_t pos0,
                int T, void* stream, const int64_t* pos_dev, const int64_t* mpos) {
    if (T <= 0) return;
    if (n_rot % 2 || rot_off + n_rot > head_dim) throw std::runtime_error("hy4_rope_w: the rotated pairs leave the head");
    if (mpos && n_rot != 2 * 32) throw std::runtime_error("hy4_rope_w: M-RoPE's sections are for 32 pairs");
    const float theta_scale = powf(base, -2.0f / (float) n_rot);
    const int pairs = n_heads * (n_rot / 2);
    rope_pairs_kernel<<<dim3(std::max(1, std::min(64, blocks(pairs, 256))), T), 256, 0, st(stream)>>>(
        x, row_stride, head_off, n_heads, head_dim, rot_off, n_rot, theta_scale, pos0, pos_dev, mpos);
    check_launch("hy4_rope_w");
}

void hy4_kv_row_w(float* kv, const float* norm_w, float eps, float base, uint16_t* cache, int64_t rows, int64_t pos0, int T,
                  void* stream, const int64_t* pos_dev, const int64_t* mpos) {
    if (T <= 0) return;
    kv_row_kernel<<<T, 256, 0, st(stream)>>>(kv, norm_w, eps, powf(base, -2.0f / (float) kRope), reinterpret_cast<__half*>(cache),
                                             rows, pos0, pos_dev, mpos);
    check_launch("hy4_kv_row_w");
}

void hy4_store_kv_w(const float* kv, uint16_t* cache, int64_t rows, int64_t pos0, int T, void* stream, const int64_t* pos_dev) {
    if (T <= 0) return;
    store_kv_kernel<<<T, 192, 0, st(stream)>>>(kv, reinterpret_cast<__half*>(cache), rows, pos0, pos_dev);
    check_launch("hy4_store_kv_w");
}

void hy4_kpool_w(const float* ring_iv, const float* ring_ig, int ring_rows, const float* ape, const float* norm_w, float eps,
                 float* pooled, float base, int64_t pos0, int T, int kp, void* stream, const int64_t* pos_dev,
                 const int64_t* mpos) {
    if (T <= 0) return;
    if (kp < 1 || kp > 8) throw std::runtime_error("hy4_kpool_w: groups of 1 .. 8");
    // at most ceil(T / kp) groups end in a window of T positions; the blocks without one return
    kpool_kernel<<<(T + kp - 1) / kp, kIdx, 0, st(stream)>>>(ring_iv, ring_ig, ring_rows, ape, norm_w, eps, pooled,
                                                             powf(base, -2.0f / (float) kRope), pos0, pos_dev, T, kp, mpos);
    check_launch("hy4_kpool_w");
}

void hy4_scores_w(const float* iq, const float* pooled, const float* w, float* score, int64_t pos0, int T, int kp, int max_pool,
                  void* stream, const int64_t* pos_dev) {
    if (T <= 0) return;
    // enough blocks for the longest row (a captured graph's position is read on the device: the context's pools)
    const int64_t last = pos_dev ? (int64_t) max_pool * kp : pos0 + T;
    const int n_pool = (int) std::min<int64_t>(max_pool, last / kp);
    // the pool tiles read once for every row of the window or the prompt's sub-batch (scores_tile_kernel, the same bits;
    // STRATA_HY4_SCORES_TILE=0 keeps the block-row-per-row kernel)
    static const bool tile = [] { const char* e = std::getenv("STRATA_HY4_SCORES_TILE"); return !(e && e[0] == '0'); }();
    // moex (2026-11-19): register-tiled (scores_mm_kernel, the same bits again; STRATA_HY4_SCORES_MM=0 keeps the tile
    // kernel).  The block shape by the rows - each the fastest on an RTX 3080 at 819,200 positions (idx3.cu): 1 row 0.715 ->
    // 0.162 ms, 6 rows 4.31 -> 0.52 ms, 8 rows 5.85 -> 0.65, a 512-row sub-batch 404 -> 39.4 ms
    static const bool mm = [] { const char* e = std::getenv("STRATA_HY4_SCORES_MM"); return !(e && e[0] == '0'); }();
    if (tile && mm) {
        auto go = [&](auto kern, int bt, int bp, int threads) {
            const int n_rt = (T + bt - 1) / bt;
            const int n_pt = std::max(1, std::min(65535, (n_pool + bp - 1) / bp));
            kern<<<dim3((unsigned) n_rt, (unsigned) n_pt), threads, 0, st(stream)>>>(iq, pooled, w, score, pos0, pos_dev, kp, max_pool, T);
        };
        if (T == 1) go(scores_mm_kernel<1, 8, 4>, 1, 256, 256);
        else if (T == 2) go(scores_mm_kernel<2, 4, 4>, 2, 128, 256);
        else if (T == 3) go(scores_mm_kernel<3, 4, 4>, 3, 128, 384);
        else if (T == 4) go(scores_mm_kernel<4, 2, 4>, 4, 64, 256);
        else if (T <= 6) go(scores_mm_kernel<6, 2, 4>, 6, 64, 384);
        else go(scores_mm_kernel<8, 2, 8>, 8, 128, 512);
        check_launch("hy4_scores_w");
        return;
    }
    if (tile) {
        constexpr int G = 8;
        const int nbt = std::max(1, std::min(2048, (n_pool + 4 * G - 1) / (4 * G)));
        scores_tile_kernel<G><<<nbt, 128, 0, st(stream)>>>(iq, pooled, w, score, pos0, pos_dev, kp, max_pool, T);
        check_launch("hy4_scores_w");
        return;
    }
    const int nb = std::max(1, std::min(1024, blocks(n_pool, 4)));
    scores_kernel<<<dim3(nb, T), 128, 0, st(stream)>>>(iq, pooled, w, score, pos0, pos_dev, kp, max_pool);
    check_launch("hy4_scores_w");
}

void hy4_select_w(const float* score, int max_pool, int n_top, int kp, int64_t pos0, int T, int32_t* cells, int max_cells,
                  void* stream, const int64_t* pos_dev) {
    if (T <= 0) return;
    if (n_top * kp + kp > max_cells) throw std::runtime_error("hy4_select_w: the cells do not fit max_cells");
    // moex (2026-11-19): the scores read once by a sampled threshold (select_sampled_kernel, the same cells;
    // STRATA_HY4_SELECT_SAMPLED=0 keeps the radix passes alone)
    static const bool sampled = [] { const char* e = std::getenv("STRATA_HY4_SELECT_SAMPLED"); return !(e && e[0] == '0'); }();
    if (sampled) {
        select_sampled_kernel<<<T, 1024, 0, st(stream)>>>(score, max_pool, n_top, kp, pos0, pos_dev, cells, max_cells);
        check_launch("hy4_select_w");
        return;
    }
    select_kernel<<<T, 1024, 0, st(stream)>>>(score, max_pool, n_top, kp, pos0, pos_dev, cells, max_cells);
    check_launch("hy4_select_w");
}

int hy4_cells(int64_t pos, int n_top, int kp) { return (int) std::min<int64_t>(pos / kp, n_top) * kp + (int) (pos % kp) + 1; }

void hy4_attend_w(const float* q, const uint16_t* cache, int64_t rows, const int32_t* cells, int max_cells, int n_top, int kp,
                  int window, const float* sinks, float scale, float* ctx, float* partial, int n_chunks, int H, int64_t pos0, int T,
                  void* stream, const int64_t* pos_dev, const int32_t* counts) {
    if (T <= 0) return;
    if (H % kAh) throw std::runtime_error("hy4_attend_w: a multiple of 16 heads");
    if (counts && !cells) throw std::runtime_error("hy4_attend_w: per-row counts without cells");
    if (T > 65535) throw std::runtime_error("hy4_attend_w: at most 65535 tokens a launch");
    if (n_chunks > 1 && !partial) throw std::runtime_error("hy4_attend_w: split-K without its scratch");
    const __half* c = reinterpret_cast<const __half*>(cache);
    attend_kernel<<<dim3(std::max(1, n_chunks), H / kAh, T), 256, 0, st(stream)>>>(q, c, rows, cells, max_cells, n_top, kp, window,
                                                                                  sinks, scale, ctx, partial, std::max(1, n_chunks),
                                                                                  H, pos0, pos_dev, counts);
    check_launch("hy4_attend_w");
    if (n_chunks > 1) {
        attend_merge_kernel<<<dim3(H, T), 256, 0, st(stream)>>>(partial, sinks, ctx, n_chunks, H, n_top, kp, window,
                                                                cells != nullptr ? 1 : 0, pos0, pos_dev, counts);
        check_launch("hy4_attend_w merge");
    }
}

size_t hy4_attend_scratch_bytes(int H, int n_chunks, int T) { return (size_t) T * H * n_chunks * kPart * sizeof(float); }

int hy4_attend_chunks(int max_cells) { return (max_cells + kAc - 1) / kAc; }

void hy4_q_split(const float* q, float* q_nope_hm, float* qf, int H, int T, void* stream) {
    if (T <= 0) return;
    q_split_kernel<<<dim3(H, T), 256, 0, st(stream)>>>(q, q_nope_hm, qf, H, T);
    check_launch("hy4_q_split");
}

void hy4_q_join(const float* qabs_hm, float* qf, int H, int T, void* stream) {
    if (T <= 0) return;
    q_join_kernel<<<dim3(H, T), 256, 0, st(stream)>>>(qabs_hm, qf, H, T);
    check_launch("hy4_q_join");
}

// ------------------------------------------------------------------ DFly2 (hy4_dfly.cpp)

/// rows [max(0, T - ring), T) of src (row_bytes each, a multiple of 4) into dst's ring at row (position % ring)
__global__ void snap_rows_kernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, int row_words, int T, int ring,
                                 int64_t pos0_arg, const int64_t* __restrict__ pos_dev) {
    const int t = max(0, T - ring) + (int) blockIdx.x;
    if (t >= T) return;
    const int64_t p = pos_of(pos_dev, pos0_arg) + t;
    const uint32_t* s = src + (size_t) t * row_words;
    uint32_t* d = dst + (size_t) (p % ring) * row_words;
    for (int i = threadIdx.x; i < row_words; i += blockDim.x) d[i] = s[i];
}

/// a draft row o's cells: the target row's n_hist cells (each < anchor), then the block's own positions anchor .. anchor + o
__global__ void dfly_cells_kernel(const int32_t* __restrict__ hist, int n_hist, int64_t anchor, int32_t* __restrict__ out,
                                  int stride, int32_t* __restrict__ counts) {
    const int o = blockIdx.x;
    int32_t* r = out + (size_t) o * stride;
    for (int i = threadIdx.x; i < n_hist; i += blockDim.x) {
        const int32_t c = hist[i];
        r[i] = c >= 0 && c < anchor ? c : (int32_t) (anchor - 1);   // (a stray cell: the row before the block, never past it)
    }
    if ((int) threadIdx.x <= o) r[n_hist + threadIdx.x] = (int32_t) (anchor + threadIdx.x);
    if (threadIdx.x == 0) counts[o] = n_hist + o + 1;
}

__global__ void gated_add_kernel(float* __restrict__ x, const float* __restrict__ g, const float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __fadd_rn(x[i], __fmul_rn(g[i], y[i]));
}

void hy4_snap_rows_w(const void* src, void* dst, int row_bytes, int T, int ring, int64_t pos0, void* stream, const int64_t* pos_dev) {
    if (T <= 0) return;
    if (row_bytes % 4) throw std::runtime_error("hy4_snap_rows_w: rows of whole words");
    snap_rows_kernel<<<std::min(T, ring), 256, 0, st(stream)>>>(static_cast<const uint32_t*>(src), static_cast<uint32_t*>(dst),
                                                                row_bytes / 4, T, ring, pos0, pos_dev);
    check_launch("hy4_snap_rows_w");
}

void hy4_dfly_cells_w(const int32_t* hist, int n_hist, int64_t anchor, int W, int32_t* out, int stride, int32_t* counts, void* stream) {
    if (W <= 0) return;
    if (W > 256 || n_hist + W > stride) throw std::runtime_error("hy4_dfly_cells_w: the block does not fit its rows");
    dfly_cells_kernel<<<W, 256, 0, st(stream)>>>(hist, n_hist, anchor, out, stride, counts);
    check_launch("hy4_dfly_cells_w");
}

void hy4_gated_add(float* x, const float* g, const float* y, int n, void* stream) {
    if (n <= 0) return;
    gated_add_kernel<<<blocks(n, 256), 256, 0, st(stream)>>>(x, g, y, n);
    check_launch("hy4_gated_add");
}

__global__ void round_bf16_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t u = __float_as_uint(x[i]);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) u |= 0x00400000u;   // a NaN stays one (quiet)
    else u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    y[i] = __uint_as_float(u);
}

void hy4_round_bf16(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    round_bf16_kernel<<<blocks(n, 256), 256, 0, st(stream)>>>(x, y, n);
    check_launch("hy4_round_bf16");
}

void hy4_gate_mul(float* o, const float* g, int64_t n, void* stream) {
    if (n <= 0) return;
    gate_mul_kernel<<<blocks(n, 256), 256, 0, st(stream)>>>(o, g, n);
    check_launch("hy4_gate_mul");
}

void hy4_eg_rows_w(const uint8_t* rows, int fmt, const float* gamma, const float* valid, float* out, int cols, int dim, float eps, int T,
                   void* stream) {
    if (T <= 0) return;
    if (dim < 1 || dim > 1024) throw std::runtime_error("hy4_eg_rows_w: rows of 1 .. 1024 values");
    if (fmt != 0 && (fmt != 1 || dim % 32)) throw std::runtime_error("hy4_eg_rows_w: F8_E4M3 rows, or Q8_0 rows of whole blocks");
    eg_rows_kernel<<<dim3(cols, T), ((dim + 31) / 32) * 32, 0, st(stream)>>>(rows, fmt, gamma, valid, out, cols, dim, eps);
    check_launch("hy4_eg_rows_w");
}

void hy4_eg_apply_w(float* R, const float* kv, const float* kv_b, const float* egate, const float* pre_gamma, const float* hc_scale,
                    const float* valid, int cols, int n, float eps, int T, void* stream, float eps_qk) {
    if (T <= 0) return;
    eg_apply_kernel<<<T, 1024, 0, st(stream)>>>(R, kv, kv_b, egate, pre_gamma, hc_scale, valid, cols, n, eps,
                                                eps_qk >= 0.0f ? eps_qk : eps);
    check_launch("hy4_eg_apply_w");
}

}  // namespace strata::x
