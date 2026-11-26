// src/x/kernels.cu - moex: strata-x's decode kernels, first written for GLM-5.3-Flash.  See include/strata/x/kernels.hpp; the
// arithmetic of each is moex/ref/glm_ref.cpp's (named per function), which matches llama.cpp op by op.
#include "strata/x/kernels.hpp"
#include "strata/kernels/glu_act.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace strata::x {
namespace {

inline cudaStream_t st(void* s) { return static_cast<cudaStream_t>(s); }

inline void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "strata-x kernels: %s: %s\n", what, cudaGetErrorString(e));
        throw std::runtime_error(what);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}


/// Sum over the block (BS a multiple of 32, <= 1024); every thread gets the total.
template <int BS>
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float red[BS / 32];
    __shared__ float total;
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float w = threadIdx.x < BS / 32 ? red[threadIdx.x] : 0.0f;
        w = warp_sum(w);
        if (threadIdx.x == 0) total = w;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();   // `red`/`total` may be reused by a second call
    return r;
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

// ------------------------------------------------------------------ generic

template <int BS>
__global__ void __launch_bounds__(BS) rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                      float* __restrict__ y, int cols, float eps) {
    const float* xr = x + (size_t) blockIdx.x * cols;
    float* yr = y + (size_t) blockIdx.x * cols;
    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BS) {
        const float v = xr[i];
        s += v * v;
    }
    s = block_sum<BS>(s);
    const float scale = 1.0f / sqrtf(s / (float) cols + eps);
    for (int i = threadIdx.x; i < cols; i += BS) {
        const float v = xr[i] * scale;
        yr[i] = w ? v * w[i] : v;
    }
}

/// moex (DSV41_PLAN §34), opt-in STRATA_DSV41_PRELOAD=1: rms_norm_kernel with each thread's (at most PT) values loaded first
/// and kept for the second pass - the same sums in the same order, the same block tree, the same products (in place too:
/// a thread rewrites only its own elements).  A decode row's norms (5,120 or 4 x 5,120 floats, one block) waited for one
/// load after another
template <int BS, int PT>
__global__ void __launch_bounds__(BS) rms_norm_pre_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                          float* __restrict__ y, int cols, float eps) {
    const float* xr = x + (size_t) blockIdx.x * cols;
    float* yr = y + (size_t) blockIdx.x * cols;
    float v[PT], wv[PT];
#pragma unroll
    for (int j = 0; j < PT; ++j) {
        const int i = threadIdx.x + j * BS;
        v[j] = i < cols ? xr[i] : 0.0f;
        wv[j] = w && i < cols ? w[i] : 0.0f;
    }
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < PT; ++j)
        if (threadIdx.x + j * BS < cols) s += v[j] * v[j];
    s = block_sum<BS>(s);
    const float scale = 1.0f / sqrtf(s / (float) cols + eps);
#pragma unroll
    for (int j = 0; j < PT; ++j) {
        const int i = threadIdx.x + j * BS;
        if (i < cols) {
            const float o = v[j] * scale;
            yr[i] = w ? o * wv[j] : o;
        }
    }
}

template <int BS>
__global__ void __launch_bounds__(BS) layer_norm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                        const float* __restrict__ b, float* __restrict__ y, int cols,
                                                        float eps) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BS) s += x[i];
    const float mean = block_sum<BS>(s) / (float) cols;
    float s2 = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BS) {
        const float d = x[i] - mean;
        s2 += d * d;
    }
    const float var = block_sum<BS>(s2) / (float) cols;
    const float scale = 1.0f / sqrtf(var + eps);
    for (int i = threadIdx.x; i < cols; i += BS) y[i] = (x[i] - mean) * scale * w[i] + b[i];
}

template <int BS>
__global__ void __launch_bounds__(BS) layer_norm_kernel_rows(const float* __restrict__ x, const float* __restrict__ w,
                                                             const float* __restrict__ b, float* __restrict__ y, int cols,
                                                             float eps) {
    const float* xr = x + (size_t) blockIdx.x * cols;
    float* yr = y + (size_t) blockIdx.x * cols;
    float sacc = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BS) sacc += xr[i];
    const float mean = block_sum<BS>(sacc) / (float) cols;
    float s2 = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BS) {
        const float d = xr[i] - mean;
        s2 += d * d;
    }
    const float var = block_sum<BS>(s2) / (float) cols;
    const float scale = 1.0f / sqrtf(var + eps);
    for (int i = threadIdx.x; i < cols; i += BS) yr[i] = (xr[i] - mean) * scale * w[i] + b[i];
}

/// One warp per output row; float4 loads when the row is 16-byte aligned and n_in % 128 == 0.
__global__ void __launch_bounds__(256) gemv_f32_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                       float* __restrict__ y, int n_in, int n_out) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n_out) return;
    const float* wr = W + (size_t) row * n_in;
    float acc = 0.0f;
    if ((n_in & 127) == 0) {
        const float4* w4 = reinterpret_cast<const float4*>(wr);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int i = lane; i < n_in / 4; i += 32) {
            const float4 a = w4[i], b = x4[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int i = lane; i < n_in; i += 32) acc += wr[i] * x[i];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

/// moex (ERNIE-5.5-Flash): SiTU-GLU, the activation a negative limit selects (strata/kernels/glu_act.hpp), in the reference's
/// order: ((4 tanh(g / 4)) sigmoid(g)) (25 tanh(u / 25))
__device__ __forceinline__ float situ_glu_d(float g, float u) {
    const float a = __fmul_rn(strata::kernels::kSituBeta, tanhf(__fdiv_rn(g, strata::kernels::kSituBeta)));
    const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g)));
    const float b = __fmul_rn(strata::kernels::kSituLinBeta, tanhf(__fdiv_rn(u, strata::kernels::kSituLinBeta)));
    return __fmul_rn(__fmul_rn(a, sg), b);
}

__global__ void swiglu_clamp_kernel(const float* __restrict__ g, const float* __restrict__ u, float* out, int n,
                                    float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (limit < 0.0f) {   // moex (ERNIE-5.5): SiTU-GLU
        out[i] = situ_glu_d(gv, uv);
        return;
    }
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    out[i] = gv / (1.0f + expf(-gv)) * uv;
}

__global__ void scale_kernel(float* x, int n, float s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void __launch_bounds__(1024) argmax_kernel(const float* __restrict__ x, int n, int32_t* __restrict__ out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float v = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += 1024)
        if (x[i] > v) {   // strictly greater: the first (lowest) index wins within a thread's stride
            v = x[i];
            idx = i;
        }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > v || (ov == v && oi < idx)) {
            v = ov;
            idx = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        bv[threadIdx.x >> 5] = v;
        bi[threadIdx.x >> 5] = idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float best = -INFINITY;
        int besti = 0;
        for (int w = 0; w < 32; ++w)
            if (bv[w] > best || (bv[w] == best && bi[w] < besti)) {
                best = bv[w];
                besti = bi[w];
            }
        *out = besti;
    }
}

/// The arg-max of a row and the probability softmax(x) gives it: 1 / sum(exp(x - max)).  One block per row, two passes over
/// the logits (the second only for the sum).  The drafter's token and its confidence (spec-min-p) in one launch.
__global__ void __launch_bounds__(1024) argmax_prob_kernel(const float* __restrict__ x, int n, int32_t* __restrict__ out,
                                                           float* __restrict__ outp) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ float bsum[32];
    __shared__ float gmax;
    float v = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += 1024)
        if (x[i] > v) {
            v = x[i];
            idx = i;
        }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > v || (ov == v && oi < idx)) {
            v = ov;
            idx = oi;
        }
    }
    if ((threadIdx.x & 31) == 0) {
        bv[threadIdx.x >> 5] = v;
        bi[threadIdx.x >> 5] = idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float best = -INFINITY;
        int besti = 0;
        for (int w = 0; w < 32; ++w)
            if (bv[w] > best || (bv[w] == best && bi[w] < besti)) {
                best = bv[w];
                besti = bi[w];
            }
        *out = besti;
        gmax = best;
    }
    __syncthreads();
    const float mx = gmax;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += 1024) sum += expf(x[i] - mx);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((threadIdx.x & 31) == 0) bsum[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < 32; ++w) total += bsum[w];
        *outp = 1.0f / total;
    }
}

/// y[t] = [a[t] | b[t]]: rows of n floats side by side (the NextN's eh_proj input).
__global__ void concat_rows_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t t = blockIdx.y;
    y[t * 2 * n + i] = a[t * n + i];
    y[t * 2 * n + n + i] = b[t * n + i];
}

// ------------------------------------------------------------------ mHC

/// One thread: 24 values and 20 Sinkhorn iterations are nothing; the launch is the cost.
__global__ void mhc_coef_kernel(const float* __restrict__ mixes, const float* __restrict__ sc,
                                const float* __restrict__ bs, float* __restrict__ coef, float eps, int iters) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int h = 0; h < 4; ++h) {
        coef[h] = sigmoid_f(mixes[h] * sc[0] + bs[h]) * 1.0f + eps;
        coef[4 + h] = sigmoid_f(mixes[4 + h] * sc[1] + bs[4 + h]) * 2.0f;
    }
    float c[16];
    for (int src = 0; src < 4; ++src) {
        float mx = -INFINITY;
        for (int dst = 0; dst < 4; ++dst) {
            const int idx = dst + 4 * src;
            c[idx] = mixes[8 + idx] * sc[2] + bs[8 + idx];
            mx = fmaxf(mx, c[idx]);
        }
        float sum = 0.0f;
        for (int dst = 0; dst < 4; ++dst) {
            const int idx = dst + 4 * src;
            c[idx] = expf(c[idx] - mx);
            sum += c[idx];
        }
        const float inv = 1.0f / sum;
        for (int dst = 0; dst < 4; ++dst) c[dst + 4 * src] = c[dst + 4 * src] * inv + eps;
    }
    auto norm_cols = [&]() {
        for (int dst = 0; dst < 4; ++dst) {
            float sum = eps;
            for (int src = 0; src < 4; ++src) sum += c[dst + 4 * src];
            const float inv = 1.0f / sum;
            for (int src = 0; src < 4; ++src) c[dst + 4 * src] *= inv;
        }
    };
    auto norm_rows = [&]() {
        for (int src = 0; src < 4; ++src) {
            float sum = eps;
            for (int dst = 0; dst < 4; ++dst) sum += c[dst + 4 * src];
            const float inv = 1.0f / sum;
            for (int dst = 0; dst < 4; ++dst) c[dst + 4 * src] *= inv;
        }
    };
    norm_cols();
    for (int i = 1; i < iters; ++i) {
        norm_rows();
        norm_cols();
    }
    for (int i = 0; i < 16; ++i) coef[8 + i] = c[i];
}

__global__ void mhc_pre_kernel(const float* __restrict__ R, const float* __restrict__ coef, float* __restrict__ out,
                               int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
#pragma unroll
    for (int h = 0; h < 4; ++h) s += R[(size_t) h * n + i] * coef[h];
    out[i] = s;
}

__global__ void mhc_post_kernel(const float* __restrict__ x, float* R, const float* __restrict__ coef, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float r[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) r[h] = R[(size_t) h * n + i];
    const float xv = x[i];
#pragma unroll
    for (int dst = 0; dst < 4; ++dst) {
        float s = xv * coef[4 + dst];
#pragma unroll
        for (int src = 0; src < 4; ++src) s += r[src] * coef[8 + dst + 4 * src];
        R[(size_t) dst * n + i] = s;
    }
}

__global__ void mhc_mean_kernel(const float* __restrict__ R, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (((R[i] + R[(size_t) n + i]) + R[(size_t) 2 * n + i]) + R[(size_t) 3 * n + i]) * 0.25f;
}

// ------------------------------------------------------------------ KDA

__global__ void kda_conv_silu_kernel(float* __restrict__ cs, const float* __restrict__ x, const float* __restrict__ w,
                                     float* __restrict__ out, int C, int dc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* s = cs + (size_t) c * (dc - 1);
    const float* wc = w + (size_t) c * dc;
    float in[8];
    for (int t = 0; t < dc - 1; ++t) in[t] = s[t];
    in[dc - 1] = x[c];
    float acc = 0.0f;
    for (int t = 0; t < dc; ++t) acc += in[t] * wc[t];
    for (int t = 0; t < dc - 1; ++t) s[t] = in[t + 1];
    out[c] = silu_f(acc);
}

__global__ void kda_gates_kernel(float* __restrict__ g, const float* __restrict__ dt, const float* __restrict__ A,
                                 float lb, float* __restrict__ beta, int H, int S) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) beta[i] = sigmoid_f(beta[i]);
    if (i >= H * S) return;
    const int h = i / S;
    float v = g[i] + dt[i];
    if (lb > -INFINITY) {
        v = v * A[h];
        v = sigmoid_f(v * -1.0f);
        v = v * lb;
    } else {
        v = (v > 20.0f ? v : log1pf(expf(v))) * A[h];
    }
    g[i] = v;
}

template <int BS>
__global__ void __launch_bounds__(BS) l2norm_kernel(float* __restrict__ x, int cols, float eps, float scale) {
    float* p = x + (size_t) blockIdx.x * cols;
    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BS) s += p[i] * p[i];
    s = block_sum<BS>(s);
    const float inv = 1.0f / sqrtf(s + eps) * scale;
    for (int i = threadIdx.x; i < cols; i += BS) p[i] *= inv;
}

/// FOUR threads own each state column (h, j), a quarter of its S rows each, held in REGISTERS (S/4 floats): the
/// state is read once and written once (GDN's kernel reads and writes it per pass), and the per-thread array is
/// small enough not to spill (a whole 128-row column per thread went to local memory: 1680 B of stack).  A warp
/// covers 8 consecutive j by 4 quarters, so each load is four 32-byte sectors; the column's two contractions
/// are summed over the quarters with two shuffles.
template <int S>
__global__ void __launch_bounds__(32) kda_step_kernel(float* __restrict__ state, const float* __restrict__ q,
                                                      const float* __restrict__ k, const float* __restrict__ v,
                                                      const float* __restrict__ g, const float* __restrict__ beta,
                                                      float* __restrict__ o, int H) {
    constexpr int PER = S / 4;
    __shared__ float ks[S], qs[S], ds[S];
    const int h = blockIdx.y;
    const int lane = threadIdx.x;
    const int jj = lane & 7, part = lane >> 3;
    const int j = blockIdx.x * 8 + jj;
    for (int i = lane; i < S; i += 32) {
        ks[i] = k[h * S + i];
        qs[i] = q[h * S + i];
        ds[i] = expf(g[h * S + i]);
    }
    __syncwarp();
    const float b = beta[h];
    float* col = state + (size_t) h * S + j;
    const size_t stride = (size_t) H * S;
    const int i0 = part * PER;
    float c[PER];
    float sk = 0.0f;
#pragma unroll
    for (int t = 0; t < PER; ++t) {
        c[t] = col[(size_t) (i0 + t) * stride] * ds[i0 + t];
        sk += c[t] * ks[i0 + t];
    }
    sk += __shfl_xor_sync(0xffffffffu, sk, 8);
    sk += __shfl_xor_sync(0xffffffffu, sk, 16);
    const float d = (v[h * S + j] - sk) * b;
    float dot = 0.0f;
#pragma unroll
    for (int t = 0; t < PER; ++t) {
        c[t] = fmaf(ks[i0 + t], d, c[t]);
        col[(size_t) (i0 + t) * stride] = c[t];
        dot += c[t] * qs[i0 + t];
    }
    dot += __shfl_xor_sync(0xffffffffu, dot, 8);
    dot += __shfl_xor_sync(0xffffffffu, dot, 16);
    if (part == 0) o[h * S + j] = dot;
}

template <int BS>
__global__ void __launch_bounds__(BS) kda_out_gate_kernel(const float* __restrict__ o, const float* __restrict__ g2,
                                                          const float* __restrict__ w, float* __restrict__ y, int S,
                                                          float eps) {
    const int h = blockIdx.x;
    const float* oh = o + (size_t) h * S;
    float s = 0.0f;
    for (int i = threadIdx.x; i < S; i += BS) s += oh[i] * oh[i];
    s = block_sum<BS>(s);
    const float scale = 1.0f / sqrtf(s / (float) S + eps);
    for (int i = threadIdx.x; i < S; i += BS) y[h * S + i] = oh[i] * scale * w[i] * sigmoid_f(g2[h * S + i]);
}

// ------------------------------------------------------------------ DSA

__global__ void store_latent_kernel(const float* __restrict__ kv, __half* __restrict__ cache, int L) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < L) cache[i] = __float2half_rn(kv[i]);
}

__global__ void kpool_kernel(const float* __restrict__ ik, const float* __restrict__ ig, const float* __restrict__ ape,
                             float* __restrict__ pooled, int D, int kp) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float lg[8];
    float mx = -INFINITY;
    for (int k = 0; k < kp; ++k) {
        lg[k] = ig[(size_t) k * D + d] + ape[d + (size_t) D * k];
        mx = fmaxf(mx, lg[k]);
    }
    float sum = 0.0f;
    for (int k = 0; k < kp; ++k) {
        lg[k] = expf(lg[k] - mx);
        sum += lg[k];
    }
    const float inv = 1.0f / sum;
    float acc = 0.0f;
    for (int k = 0; k < kp; ++k) acc += (lg[k] * inv) * ik[(size_t) k * D + d];
    pooled[d] = acc;
}

/// One warp per pool; lane h holds head h's query (IH == 32), the pool key is staged per warp in shared memory and
/// read as a broadcast; iq is transposed into shared so lane h's reads are conflict-free.
template <int D>
__global__ void __launch_bounds__(128) scores_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                                     const float* __restrict__ w, float* __restrict__ score,
                                                     int n_pool) {
    __shared__ float iqT[D][33];
    __shared__ float key[4][D];
    for (int e = threadIdx.x; e < 32 * D; e += 128) {
        const int h = e / D, d = e % D;
        iqT[d][h] = iq[e];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float wh = w[lane];
    for (int P = blockIdx.x * 4 + warp; P < n_pool; P += gridDim.x * 4) {
        for (int d = lane; d < D; d += 32) key[warp][d] = pooled[(size_t) P * D + d];
        __syncwarp();
        float acc = 0.0f;
#pragma unroll 8
        for (int d = 0; d < D; ++d) acc += iqT[d][lane] * key[warp][d];
        float s = fmaxf(acc, 0.0f) * wh;
        s = warp_sum(s);
        if (lane == 0) score[P] = s;
        __syncwarp();
    }
}

__device__ __forceinline__ uint32_t okey(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// Exclusive prefix sum of one value per thread over the block (blockDim.x a multiple of 32, at most 1024); `tmp` holds
/// 33 words of shared memory.  Every thread must call it (it synchronizes).
__device__ uint32_t block_excl_scan(uint32_t v, uint32_t* tmp) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (int) (blockDim.x >> 5);
    uint32_t x = v;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const uint32_t y = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) x += y;
    }
    __syncthreads();   // tmp may still be read by a previous scan
    if (lane == 31) tmp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        uint32_t w = lane < nw ? tmp[lane] : 0u;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, w, d);
            if (lane >= d) w += y;
        }
        if (lane < nw) tmp[lane] = w;   // inclusive warp totals
    }
    __syncthreads();
    return (warp ? tmp[warp - 1] : 0u) + x - v;
}

/// After the radix pass (threshold key `thr`, `remaining` ties to take): the selected pools expanded to cells IN POOL
/// ORDER (the reference sorts its cells ascending), ties at the threshold taken lowest index first.  Deterministic:
/// each thread takes a contiguous range of pools and two block scans place its picks.  (The first version claimed
/// slots with atomicAdd: the cells' order - so the attention's summation order - and the choice among tied pools,
/// which do occur (ReLU scores tie at 0), varied from run to run.)
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

/// The top `k` of `n` scores by MSB-first radix select (one block), then the pools expanded to cells (select_ordered).
__global__ void __launch_bounds__(1024) topk_pools_kernel(const float* __restrict__ score, int n, int k, int kp,
                                                          int32_t* __restrict__ cells) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_digit, s_rem;
    __shared__ uint32_t scan_tmp[33];
    uint32_t prefix = 0, mask = 0, remaining = (uint32_t) k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t key = okey(score[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t cum = 0;
            for (int dgt = 255; dgt >= 0; --dgt) {
                if (cum + hist[dgt] >= remaining) {
                    s_digit = (uint32_t) dgt;
                    s_rem = remaining - cum;
                    break;
                }
                cum += hist[dgt];
            }
        }
        __syncthreads();
        prefix |= s_digit << shift;
        mask |= 255u << shift;
        remaining = s_rem;
        __syncthreads();
    }
    select_ordered(score, n, prefix, remaining, kp, cells, scan_tmp);
}

__global__ void all_pools_kernel(int n_cells_pools, int32_t* __restrict__ cells) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_cells_pools) cells[i] = i;
}

__global__ void tail_kernel(int32_t* __restrict__ cells, int base, int n_tail, int64_t pos) {
    const int t = threadIdx.x;
    if (t < n_tail) cells[base + t] = (int32_t) (pos - n_tail + 1 + t);   // ascending, as the reference
}

constexpr int kCh = 32;   ///< cells per attention block
constexpr int kHg = 16;   ///< heads per attention block (2 per warp)

/// Flash-decoding over the latent: block (chunk, head group) scores its kCh cells for kHg heads and leaves, per
/// head, (max, sum, sum p.kv) for the merge.  The chunk's FP16 rows are staged once for the group's heads.
template <int L>
__global__ void __launch_bounds__(256) attend_partial_kernel(const float* __restrict__ q, const __half* __restrict__ kv,
                                                             const int32_t* __restrict__ cells, int n_cells,
                                                             float scale, float* __restrict__ partial, int n_chunks) {
    constexpr int PER = L / 32;   // dims per lane
    __shared__ __align__(16) __half tile[kCh][L];
    const int chunk = blockIdx.x, hg = blockIdx.y;
    const int c0 = chunk * kCh;
    const int nc = min(kCh, n_cells - c0);
    // stage the chunk's rows: kCh * L halves as uint4 (8 halves each)
    for (int e = threadIdx.x; e < kCh * L / 8; e += 256) {
        const int c = e / (L / 8), part = e % (L / 8);
        uint4 val = make_uint4(0, 0, 0, 0);
        if (c < nc) val = reinterpret_cast<const uint4*>(kv + (size_t) cells[c0 + c] * L)[part];
        reinterpret_cast<uint4*>(&tile[c][0])[part] = val;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int hh = 0; hh < 2; ++hh) {
        const int h = hg * kHg + warp * 2 + hh;
        float qr[PER];
#pragma unroll
        for (int t = 0; t < PER; ++t) qr[t] = q[(size_t) h * L + lane + 32 * t];
        float m = -INFINITY, l = 0.0f, acc[PER];
#pragma unroll
        for (int t = 0; t < PER; ++t) acc[t] = 0.0f;
        for (int c = 0; c < nc; ++c) {
            float s = 0.0f;
#pragma unroll
            for (int t = 0; t < PER; ++t) s += qr[t] * __half2float(tile[c][lane + 32 * t]);
            s = warp_sum(s) * scale;
            const float mn = fmaxf(m, s);
            const float corr = expf(m - mn), p = expf(s - mn);
            l = l * corr + p;
#pragma unroll
            for (int t = 0; t < PER; ++t) acc[t] = acc[t] * corr + p * __half2float(tile[c][lane + 32 * t]);
            m = mn;
        }
        float* out = partial + ((size_t) h * n_chunks + chunk) * (L + 2);
#pragma unroll
        for (int t = 0; t < PER; ++t) out[lane + 32 * t] = acc[t];
        if (lane == 0) {
            out[L] = m;
            out[L + 1] = l;
        }
    }
}

template <int L>
__global__ void __launch_bounds__(256) attend_merge_kernel(const float* __restrict__ partial, float* __restrict__ ctx,
                                                           int n_chunks) {
    const int h = blockIdx.x;
    const float* base = partial + (size_t) h * n_chunks * (L + 2);
    float M = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, base[(size_t) c * (L + 2) + L]);
    float lsum = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float* pc = base + (size_t) c * (L + 2);
        lsum += expf(pc[L] - M) * pc[L + 1];
    }
    const float inv = 1.0f / lsum;
    for (int d = threadIdx.x; d < L; d += 256) {
        float a = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            const float* pc = base + (size_t) c * (L + 2);
            a += expf(pc[L] - M) * pc[d];
        }
        ctx[(size_t) h * L + d] = a * inv;
    }
}

// ------------------------------------------------------------------ MoE

__global__ void __launch_bounds__(1024) route_kernel(const float* __restrict__ logits_all, const float* __restrict__ bias,
                                                     int E, int k, bool norm, float scale, int32_t* __restrict__ ids_all,
                                                     float* __restrict__ weights_all) {
    const float* logits = logits_all + (size_t) blockIdx.x * E;
    int32_t* ids = ids_all + (size_t) blockIdx.x * k;
    float* weights = weights_all + (size_t) blockIdx.x * k;
    __shared__ float prob[1024], sel[1024];
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int chosen[64];
    const int e = threadIdx.x;
    if (e < E) {
        prob[e] = sigmoid_f(logits[e]);
        sel[e] = prob[e] + (bias ? bias[e] : 0.0f);
    }
    __syncthreads();
    for (int r = 0; r < k; ++r) {
        float v = e < E ? sel[e] : -INFINITY;
        int idx = e < E ? e : 0x7fffffff;
        // warp arg-max, lowest index on ties
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) {
                v = ov;
                idx = oi;
            }
        }
        if ((e & 31) == 0) {
            bv[e >> 5] = v;
            bi[e >> 5] = idx;
        }
        __syncthreads();
        if (e == 0) {
            float best = -INFINITY;
            int besti = 0x7fffffff;
            for (int w = 0; w < (int) (blockDim.x >> 5); ++w)
                if (bv[w] > best || (bv[w] == best && bi[w] < besti)) {
                    best = bv[w];
                    besti = bi[w];
                }
            if (besti == 0x7fffffff) besti = r;   // no finite score (NaN activations): any unused index, never an access past sel
            chosen[r] = besti;
            sel[besti] = -INFINITY;
        }
        __syncthreads();
    }
    if (e == 0) {
        float w[64];
        float s = 0.0f;
        for (int r = 0; r < k; ++r) {
            w[r] = prob[chosen[r]];
            s += w[r];
        }
        if (norm) {
            const float sum = fmaxf(s, 6.103515625e-5f);
            for (int r = 0; r < k; ++r) w[r] = w[r] / sum;
        }
        const bool do_scale = scale != 0.0f && scale != 1.0f;
        for (int r = 0; r < k; ++r) {
            ids[r] = chosen[r];
            weights[r] = do_scale ? w[r] * scale : w[r];
        }
    }
}

__global__ void combine_kernel(const float* __restrict__ parts, const float* __restrict__ w,
                               const float* __restrict__ shared, float* __restrict__ out, int n, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = parts[i] * w[0];
    for (int j = 1; j < k; ++j) s = s + parts[(size_t) j * n + i] * w[j];
    out[i] = shared ? s + shared[i] : s;
}

inline int blocks(int n, int bs) { return (n + bs - 1) / bs; }


// ================================================================== WINDOWS (T tokens)

/// 16 lanes per token, lane k holding comb element k = dst + 4 * src: the per-src softmax and the row sums are over
/// the 4 consecutive lanes of a src (xor 1, 2), the column sums over the 4 lanes of a dst (xor 4, 8).  The same
/// arithmetic as the single-thread kernel, the sums in the same order ((a + b) + (c + d) vs ((a + b) + c) + d aside).
__global__ void __launch_bounds__(32) mhc_coef_w_kernel(const float* __restrict__ mixes, const float* __restrict__ sc,
                                                        const float* __restrict__ bs, float* __restrict__ coef, float eps,
                                                        int iters, int T) {
    const int lane = threadIdx.x & 31;
    const int t = blockIdx.x * 2 + (lane >> 4);
    const int k = lane & 15;
    const bool live = t < T;
    const float* mx_t = mixes + (size_t) (live ? t : 0) * 24;
    float* cf = coef + (size_t) (live ? t : 0) * 24;
    if (live && k < 4) {
        cf[k] = sigmoid_f(mx_t[k] * sc[0] + bs[k]) * 1.0f + eps;
        cf[4 + k] = sigmoid_f(mx_t[4 + k] * sc[1] + bs[4 + k]) * 2.0f;
    }
    float c = mx_t[8 + k] * sc[2] + bs[8 + k];
    // softmax over dst for this src (the 4 lanes sharing k / 4)
    float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    c = expf(c - mx);
    float sum = c + __shfl_xor_sync(0xffffffffu, c, 1);
    sum = sum + __shfl_xor_sync(0xffffffffu, sum, 2);
    c = c * (1.0f / sum) + eps;
    for (int it = 0; it < iters; ++it) {
        if (it > 0) {   // rows: per src, over dst
            float r = c + __shfl_xor_sync(0xffffffffu, c, 1);
            r = r + __shfl_xor_sync(0xffffffffu, r, 2);
            c *= 1.0f / (r + eps);
        }
        float col = c + __shfl_xor_sync(0xffffffffu, c, 4);   // cols: per dst, over src
        col = col + __shfl_xor_sync(0xffffffffu, col, 8);
        c *= 1.0f / (col + eps);
    }
    if (live) cf[8 + k] = c;
}

__global__ void mhc_pre_w_kernel(const float* __restrict__ R, const float* __restrict__ coef, float* __restrict__ out,
                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    const float* r = R + (size_t) t * 4 * n;
    const float* cf = coef + (size_t) t * 24;
    float s = 0.0f;
#pragma unroll
    for (int h = 0; h < 4; ++h) s += r[(size_t) h * n + i] * cf[h];
    out[(size_t) t * n + i] = s;
}

/// moex (DSV41_PLAN §34), opt-in STRATA_DSV41_MIX_FUSED=1: a decode window's (one token) DeepSeek-V4.1 mixer read after its
/// split gemv's partials, in ONE block of 1024 threads - the partials' fixed-order sums (gemv_f32_reduce_kernel), the
/// Sinkhorn coefficients (mhc_coef_w_kernel's warp for the token), the pre-mix with the carried coefficients
/// (mhc_pre_w_kernel; they are read before this half's replace them), that copy, and the weighted RMS norm (rms_norm_kernel
/// <1024>: each thread its own elements i = tid + 1024 j, the same block tree).  Each step is the kernel it replaces line for
/// line, so the same bits: five launches, ~12 us of a decode layer's mixer, become one.  n <= 5 x 1024
__global__ void __launch_bounds__(1024) mhc_mix_in_w1_kernel(const float* __restrict__ part, int S, const float* __restrict__ sc,
                                                             const float* __restrict__ bs, float* __restrict__ mixes,
                                                             float* __restrict__ coef, float* __restrict__ cpre,
                                                             const float* __restrict__ R, float* __restrict__ cur,
                                                             const float* __restrict__ nw, int n, float eps, int iters,
                                                             float rms_eps) {
    __shared__ float smix[24], scoef[24], sold[24];
    const int tid = threadIdx.x;
    if (tid < 24) {   // gemv_f32_reduce_kernel's row tid
        const float* p = part + (size_t) tid * S;
        float v = p[0];
        for (int k = 1; k < S; ++k) v += p[k];
        mixes[tid] = v;
        smix[tid] = v;
    } else if (tid >= 32 && tid < 56) {
        sold[tid - 32] = cpre[tid - 32];   // the carried pre, before this half's replaces it
    }
    __syncthreads();
    if (tid < 32) {   // mhc_coef_w_kernel, T = 1 (lanes 16-31: a token that is not there, on token 0's rows as there)
        const int lane = tid & 31;
        const int t = lane >> 4;
        const int k = lane & 15;
        const bool live = t < 1;
        const float* mx_t = smix;
        if (live && k < 4) {
            scoef[k] = sigmoid_f(mx_t[k] * sc[0] + bs[k]) * 1.0f + eps;
            scoef[4 + k] = sigmoid_f(mx_t[4 + k] * sc[1] + bs[4 + k]) * 2.0f;
        }
        float c = mx_t[8 + k] * sc[2] + bs[8 + k];
        float mx = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        c = expf(c - mx);
        float sum = c + __shfl_xor_sync(0xffffffffu, c, 1);
        sum = sum + __shfl_xor_sync(0xffffffffu, sum, 2);
        c = c * (1.0f / sum) + eps;
        for (int it = 0; it < iters; ++it) {
            if (it > 0) {
                float r = c + __shfl_xor_sync(0xffffffffu, c, 1);
                r = r + __shfl_xor_sync(0xffffffffu, r, 2);
                c *= 1.0f / (r + eps);
            }
            float col = c + __shfl_xor_sync(0xffffffffu, c, 4);
            col = col + __shfl_xor_sync(0xffffffffu, col, 8);
            c *= 1.0f / (col + eps);
        }
        if (live) scoef[8 + k] = c;
    }
    __syncthreads();
    if (tid < 24) {
        coef[tid] = scoef[tid];
        cpre[tid] = scoef[tid];   // the carried pre: this half's
    }
    constexpr int PT = 5;
    float v[PT];
#pragma unroll
    for (int j = 0; j < PT; ++j) {   // mhc_pre_w_kernel's element i
        const int i = tid + j * 1024;
        float s = 0.0f;
        if (i < n) {
#pragma unroll
            for (int h = 0; h < 4; ++h) s += R[(size_t) h * n + i] * sold[h];
        }
        v[j] = s;
    }
    float ss = 0.0f;   // rms_norm_kernel<1024>
#pragma unroll
    for (int j = 0; j < PT; ++j)
        if (tid + j * 1024 < n) ss += v[j] * v[j];
    ss = block_sum<1024>(ss);
    const float scale = 1.0f / sqrtf(ss / (float) n + rms_eps);
#pragma unroll
    for (int j = 0; j < PT; ++j) {
        const int i = tid + j * 1024;
        if (i < n) {
            const float o = v[j] * scale;
            cur[i] = nw ? o * nw[i] : o;
        }
    }
}

/// moex (the ERNIE-5.5 hill-climb): a one-token mixer read whose Sinkhorn chain runs beside it on s3 (Stage::mhc_half) - the four
/// pre coefficients computed here by mhc_coef_w_kernel's expression for its lanes k < 4, then mhc_pre_w_kernel's sum for each
/// element and rms_norm_kernel<1024>'s norm (each thread its own elements i = tid + 1024 j, the same block tree), in one block:
/// each step the kernel it replaces line for line (mhc_mix_in_w1_kernel's tail), so the same bits.  n <= 5 x 1024
__global__ void __launch_bounds__(1024) mhc_pre_norm_w1_kernel(float* __restrict__ mixes, const float* __restrict__ parts,
                                                               const float* __restrict__ sc, const float* __restrict__ bs, float eps,
                                                               const float* __restrict__ R, float* __restrict__ cur,
                                                               const float* __restrict__ nw, int n, float rms_eps) {
    __shared__ float spre[4], smix[24];
    const int tid = threadIdx.x;
    if (parts) {   // gemv_f32_w1_chunk_kernel's block sum of the warps' totals (gemv_f32_w1_warps_kernel's), in order from 0.0f
        if (tid < 24) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) v += parts[tid * 8 + w];
            mixes[tid] = v;
            smix[tid] = v;
        }
        __syncthreads();
    }
    const float* mx = parts ? smix : mixes;
    if (tid < 4) spre[tid] = sigmoid_f(mx[tid] * sc[0] + bs[tid]) * 1.0f + eps;   // mhc_coef_w_kernel, k < 4
    __syncthreads();
    constexpr int PT = 5;
    float v[PT];
#pragma unroll
    for (int j = 0; j < PT; ++j) {   // mhc_pre_w_kernel's element i
        const int i = tid + j * 1024;
        float s = 0.0f;
        if (i < n) {
#pragma unroll
            for (int h = 0; h < 4; ++h) s += R[(size_t) h * n + i] * spre[h];
        }
        v[j] = s;
    }
    float ss = 0.0f;   // rms_norm_kernel<1024>
#pragma unroll
    for (int j = 0; j < PT; ++j)
        if (tid + j * 1024 < n) ss += v[j] * v[j];
    ss = block_sum<1024>(ss);
    const float scale = 1.0f / sqrtf(ss / (float) n + rms_eps);
#pragma unroll
    for (int j = 0; j < PT; ++j) {
        const int i = tid + j * 1024;
        if (i < n) {
            const float o = v[j] * scale;
            cur[i] = nw ? o * nw[i] : o;
        }
    }
}

__global__ void mhc_post_w_kernel(const float* __restrict__ x, float* R, const float* __restrict__ coef, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    float* r = R + (size_t) t * 4 * n;
    const float* cf = coef + (size_t) t * 24;
    float rv[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) rv[h] = r[(size_t) h * n + i];
    const float xv = x[(size_t) t * n + i];
#pragma unroll
    for (int dst = 0; dst < 4; ++dst) {
        float acc = xv * cf[4 + dst];
#pragma unroll
        for (int src = 0; src < 4; ++src) acc += rv[src] * cf[8 + dst + 4 * src];
        r[(size_t) dst * n + i] = acc;
    }
}

__global__ void mhc_mean_w_kernel(const float* __restrict__ R, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    const float* r = R + (size_t) t * 4 * n;
    out[(size_t) t * n + i] = (((r[i] + r[(size_t) n + i]) + r[(size_t) 2 * n + i]) + r[(size_t) 3 * n + i]) * 0.25f;
}

__global__ void kda_conv_silu_w_kernel(float* __restrict__ cs, const float* __restrict__ x, const float* __restrict__ w,
                                       float* __restrict__ out, int C, int dc, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* s = cs + (size_t) c * (dc - 1);
    const float* wc = w + (size_t) c * dc;
    float win[8];
    for (int t = 0; t < dc - 1; ++t) win[t] = s[t];
    for (int t = 0; t < T; ++t) {
        win[dc - 1] = x[(size_t) t * C + c];
        float acc = 0.0f;
        for (int k = 0; k < dc; ++k) acc += win[k] * wc[k];
        out[(size_t) t * C + c] = silu_f(acc);
        for (int k = 0; k < dc - 1; ++k) win[k] = win[k + 1];
    }
    for (int t = 0; t < dc - 1; ++t) s[t] = win[t];
}

__global__ void kda_conv_rollback_kernel(const float* __restrict__ prev, const float* __restrict__ x,
                                         float* __restrict__ state, int C, int dc, int n_keep) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    // the input sequence is prev[0..dc-2] then x[0..n_keep-1]; the state is its last dc-1 entries
    float seq[8];
    for (int t = 0; t < dc - 1; ++t) seq[t] = prev[(size_t) c * (dc - 1) + t];
    for (int t = 0; t < n_keep; ++t) {
        for (int k = 0; k < dc - 2; ++k) seq[k] = seq[k + 1];
        seq[dc - 2] = x[(size_t) t * C + c];
    }
    for (int t = 0; t < dc - 1; ++t) state[(size_t) c * (dc - 1) + t] = seq[t];
}

__global__ void kda_gates_w_kernel(float* __restrict__ g, const float* __restrict__ dt, const float* __restrict__ A,
                                   float lb, float* __restrict__ beta, int H, int S, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * H) beta[i] = sigmoid_f(beta[i]);
    if (i >= T * H * S) return;
    const int hs = i % (H * S), h = hs / S;
    float v = g[i] + dt[hs];
    if (lb > -INFINITY) {
        v = v * A[h];
        v = sigmoid_f(v * -1.0f);
        v = v * lb;
    } else {
        v = (v > 20.0f ? v : log1pf(expf(v))) * A[h];
    }
    g[i] = v;
}

/// kda_step over T tokens, the column quarter in registers across all of them.
template <int S>
__global__ void __launch_bounds__(32) kda_scan_kernel(float* __restrict__ state, const float* __restrict__ q,
                                                      const float* __restrict__ k, const float* __restrict__ v,
                                                      const float* __restrict__ g, const float* __restrict__ beta,
                                                      float* __restrict__ o, int H, int T, float* __restrict__ snaps) {
    constexpr int PER = S / 4;
    __shared__ float ks[S], qs[S], ds[S];
    const int h = blockIdx.y;
    const int lane = threadIdx.x;
    const int jj = lane & 7, part = lane >> 3;
    const int j = blockIdx.x * 8 + jj;
    float* col = state + (size_t) h * S + j;
    const size_t stride = (size_t) H * S;
    const int i0 = part * PER;
    float c[PER];
#pragma unroll
    for (int t = 0; t < PER; ++t) c[t] = col[(size_t) (i0 + t) * stride];
    for (int tok = 0; tok < T; ++tok) {
        const size_t base = (size_t) tok * H * S + (size_t) h * S;
        __syncwarp();
        for (int i = lane; i < S; i += 32) {
            ks[i] = k[base + i];
            qs[i] = q[base + i];
            ds[i] = expf(g[base + i]);
        }
        __syncwarp();
        const float b = beta[(size_t) tok * H + h];
        float sk = 0.0f;
#pragma unroll
        for (int t = 0; t < PER; ++t) {
            c[t] = c[t] * ds[i0 + t];
            sk += c[t] * ks[i0 + t];
        }
        sk += __shfl_xor_sync(0xffffffffu, sk, 8);
        sk += __shfl_xor_sync(0xffffffffu, sk, 16);
        const float d = (v[base + j] - sk) * b;
        float dot = 0.0f;
#pragma unroll
        for (int t = 0; t < PER; ++t) {
            c[t] = fmaf(ks[i0 + t], d, c[t]);
            dot += c[t] * qs[i0 + t];
        }
        dot += __shfl_xor_sync(0xffffffffu, dot, 8);
        dot += __shfl_xor_sync(0xffffffffu, dot, 16);
        if (part == 0) o[base + j] = dot;
        if (snaps) {
            float* sc = snaps + (size_t) tok * S * stride + (size_t) h * S + j;
#pragma unroll
            for (int t = 0; t < PER; ++t) sc[(size_t) (i0 + t) * stride] = c[t];
        }
    }
#pragma unroll
    for (int t = 0; t < PER; ++t) col[(size_t) (i0 + t) * stride] = c[t];
}

// POSITIONS FROM DEVICE MEMORY (graphs): every kernel below that takes a window's first position takes it as
// `pos_ptr ? *pos_ptr : pos0` - a captured graph is replayed at every position, so the value cannot be a kernel
// argument (Strata's rule: "per-step values live in pinned/device buffers, never in kernel arguments").
__device__ __forceinline__ int64_t pos_of(const int64_t* __restrict__ p, int64_t arg) { return p ? *p : arg; }

__global__ void store_latent_w_kernel(const float* __restrict__ kv, __half* __restrict__ cache, int64_t pos0_arg,
                                      const int64_t* __restrict__ pos_ptr, int L) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    const int64_t pos0 = pos_of(pos_ptr, pos0_arg);
    if (i < L) cache[(size_t) (pos0 + t) * L + i] = __float2half_rn(kv[(size_t) t * L + i]);
}

__global__ void ring_store_w_kernel(const float* __restrict__ src, float* __restrict__ ring, int ring_rows,
                                    int64_t pos0_arg, const int64_t* __restrict__ pos_ptr, int D) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    const int64_t pos0 = pos_of(pos_ptr, pos0_arg);
    if (d < D) ring[(size_t) ((pos0 + t) % ring_rows) * D + d] = src[(size_t) t * D + d];
}

/// Block b: the b-th pool completing in the window.  `T` > 0: the launch is static (graphs) - the first completing
/// position and whether block b has a pool are worked out here from the window's position.
__global__ void kpool_w_kernel(const float* __restrict__ ik, const float* __restrict__ ig, int ring_rows,
                               const float* __restrict__ ape, float* __restrict__ pooled, int64_t p_first_arg,
                               const int64_t* __restrict__ pos_ptr, int T, int D, int kp) {
    const int d = threadIdx.x;
    if (d >= D) return;
    int64_t p_first = p_first_arg;
    if (pos_ptr) {
        const int64_t pos0 = *pos_ptr;
        p_first = pos0 + ((kp - (pos0 + 1) % kp) % kp);
        if (p_first + (int64_t) blockIdx.x * kp >= pos0 + T) return;   // no pool completes for this block
    }
    const int64_t p_last = p_first + (int64_t) blockIdx.x * kp;   // the pool's last member's position
    const int64_t P = (p_last + 1) / kp - 1;
    float lg[8];
    float mx = -INFINITY;
    for (int k = 0; k < kp; ++k) {
        const size_t row = (size_t) ((P * kp + k) % ring_rows);
        lg[k] = ig[row * D + d] + ape[d + (size_t) D * k];
        mx = fmaxf(mx, lg[k]);
    }
    float sum = 0.0f;
    for (int k = 0; k < kp; ++k) {
        lg[k] = expf(lg[k] - mx);
        sum += lg[k];
    }
    const float inv = 1.0f / sum;
    float acc = 0.0f;
    for (int k = 0; k < kp; ++k) acc += (lg[k] * inv) * ik[(size_t) ((P * kp + k) % ring_rows) * D + d];
    pooled[(size_t) P * D + d] = acc;
}

/// grid (pool blocks, T): query t scores its own n_pool(t) pools.
template <int D>
__global__ void __launch_bounds__(128) scores_w_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                                       const float* __restrict__ w, float* __restrict__ score,
                                                       int64_t pos0_arg, const int64_t* __restrict__ pos_ptr, int kp,
                                                       int max_pool) {
    __shared__ float iqT[D][33];
    __shared__ float key[4][D];
    const int t = blockIdx.y;
    const int64_t pos0 = pos_of(pos_ptr, pos0_arg);
    const int n_pool = (int) ((pos0 + t + 1) / kp);
    if ((int) blockIdx.x * 4 >= n_pool) return;
    const float* iqt = iq + (size_t) t * 32 * D;
    for (int e = threadIdx.x; e < 32 * D; e += 128) {
        const int hh = e / D, d = e % D;
        iqT[d][hh] = iqt[e];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float wh = w[(size_t) t * 32 + lane];
    float* sc = score + (size_t) t * max_pool;
    for (int P = blockIdx.x * 4 + warp; P < n_pool; P += gridDim.x * 4) {
        for (int d = lane; d < D; d += 32) key[warp][d] = pooled[(size_t) P * D + d];
        __syncwarp();
        float acc = 0.0f;
#pragma unroll 8
        for (int d = 0; d < D; ++d) acc = fmaf(iqT[d][lane], key[warp][d], acc);
        // spelled out (no contraction left to the compiler): scores_tiled_kernel computes these same operations
        float sv = __fmul_rn(fmaxf(acc, 0.0f), wh);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sv = __fadd_rn(sv, __shfl_xor_sync(0xffffffffu, sv, o));
        if (lane == 0) sc[P] = sv;
        __syncwarp();
    }
}

/// The same scores as scores_w_kernel, as a register-tiled FP32 GEMM for a prompt sub-batch (Project Maya's
/// dsa_score_tiled_kernel, MIT, rearranged to be BIT-IDENTICAL to scores_w_kernel): a block takes 4 tokens' 32 heads (128
/// query rows) against 64 pools, k in chunks of 32 through shared memory, 8 x 4 accumulators a thread.
///  - each dot is one fmaf chain over d = 0..127, as a lane of scores_w_kernel runs it;
///  - a thread's 8 rows are one token's heads i, i+4, ..., i+28 (i = its residue mod 4): the first three levels of
///    warp_sum's butterfly (h + h^16, then ^8, then ^4) pair exactly those heads, so they run in registers in that order,
///    and the last two (^2, ^1) combine the four residues as (c0 + c2) + (c1 + c3).
/// 32 heads of 128 only.  ~4-5x scores_w_kernel on this card at 10K-200K positions (x_kernels_parity dsa_bench).
constexpr int kTsT = 4, kTsP = 64, kTsK = 32;
__global__ void __launch_bounds__(256) scores_tiled_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                                           const float* __restrict__ w, float* __restrict__ score,
                                                           int64_t pos0, int kp, int T, int max_vis, int max_pool) {
    constexpr int H = 32, K = 128, R = kTsT * H;   // 128 query rows: row r = token (r >> 5), head (r >> 3 & 3) + 4 (r & 7)
    __shared__ float As[kTsK][R + 4];              // [k][row]
    __shared__ float Bs[kTsK][kTsP + 4];           // [k][pool]
    __shared__ float red[4][kTsT][kTsP];           // the four residues' partial sums
    const int t0 = blockIdx.y * kTsT, pbeg = blockIdx.x * kTsP;
    const int t_last = min(T, t0 + kTsT) - 1;
    if (pbeg >= (int) ((pos0 + t_last + 1) / kp)) return;   // no pool of this tile is visible to any of its tokens
    const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;   // rows ty*8 .. +7, pools tx*4 .. +3
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += kTsK) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {   // A: 128 rows x 32 k = 1024 float4, 4 a thread
            const int idx = tid + 256 * q, r = idx >> 3, kk = (idx & 7) * 4;
            const int t = t0 + (r >> 5), h = ((r >> 3) & 3) + 4 * (r & 7);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (t < T) v = *(const float4*) (iq + ((size_t) t * H + h) * K + k0 + kk);
            As[kk + 0][r] = v.x;
            As[kk + 1][r] = v.y;
            As[kk + 2][r] = v.z;
            As[kk + 3][r] = v.w;
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {   // B: 64 pools x 32 k = 512 float4, 2 a thread
            const int idx = tid + 256 * q, pp = idx >> 3, kk = (idx & 7) * 4;
            const int p = pbeg + pp;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (p < max_vis) v = *(const float4*) (pooled + (size_t) p * K + k0 + kk);
            Bs[kk + 0][pp] = v.x;
            Bs[kk + 1][pp] = v.y;
            Bs[kk + 2][pp] = v.z;
            Bs[kk + 3][pp] = v.w;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < kTsK; ++kk) {
            const float4 a0 = *(const float4*) &As[kk][ty * 8], a1 = *(const float4*) &As[kk][ty * 8 + 4];
            const float4 b = *(const float4*) &Bs[kk][tx * 4];
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w}, bb[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], bb[j], acc[i][j]);
        }
        __syncthreads();
    }
    // epilogue: relu(dot) * w per head, then warp_sum's tree - its first three levels here (heads i + 4m, m = 0..7)
    const int tt = ty >> 2, res = ty & 3, t = t0 + tt;
    float wv[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) wv[m] = t < T ? w[(size_t) t * H + res + 4 * m] : 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float v[8];
#pragma unroll
        for (int m = 0; m < 8; ++m) v[m] = __fmul_rn(fmaxf(acc[m][j], 0.0f), wv[m]);
        const float a0 = __fadd_rn(v[0], v[4]), a1 = __fadd_rn(v[1], v[5]), a2 = __fadd_rn(v[2], v[6]),
                    a3 = __fadd_rn(v[3], v[7]);                                            // ^16
        const float b0 = __fadd_rn(a0, a2), b1 = __fadd_rn(a1, a3);                         // ^8
        red[res][tt][tx * 4 + j] = __fadd_rn(b0, b1);                                       // ^4
    }
    __syncthreads();
    {   // 256 threads = 4 tokens x 64 pools: ^2, then ^1
        const int ot = tid >> 6, op = tid & 63, tok = t0 + ot, p = pbeg + op;
        if (tok < T && p < (int) ((pos0 + tok + 1) / kp))
            score[(size_t) tok * max_pool + p] =
                __fadd_rn(__fadd_rn(red[0][ot][op], red[2][ot][op]), __fadd_rn(red[1][ot][op], red[3][ot][op]));
    }
}

__device__ __forceinline__ int cells_of(int64_t pos, int n_top, int kp, bool tail) {
    const int n_pool = (int) ((pos + 1) / kp);
    const int n_sel = n_pool < n_top ? n_pool : n_top;
    return n_sel * kp + (tail ? (int) ((pos + 1) % kp) : 0);
}

/// One block per query: all pools (when they fit the budget) or the radix top-k, then the tail.
__global__ void __launch_bounds__(1024) select_w_kernel(const float* __restrict__ score, int max_pool, int n_top, int kp,
                                                        int64_t pos0_arg, const int64_t* __restrict__ pos_ptr, bool tail,
                                                        int32_t* __restrict__ cells, int max_cells) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_digit, s_rem;
    __shared__ uint32_t scan_tmp[33];
    const int t = blockIdx.x;
    const int64_t pos = pos_of(pos_ptr, pos0_arg) + t;
    const int n_pool = (int) ((pos + 1) / kp);
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
    const int n_tail = tail ? (int) ((pos + 1) % kp) : 0;   // the open pool's tokens, ascending after the pools
    if ((int) threadIdx.x < n_tail) out[n_sel * kp + threadIdx.x] = (int32_t) (pos - n_tail + 1 + threadIdx.x);
}

/// grid (chunks, head groups, T): query t's chunk of its own cells.
template <int L>
__global__ void __launch_bounds__(256) attend_partial_w_kernel(const float* __restrict__ q, const __half* __restrict__ kv,
                                                               const int32_t* __restrict__ cells, int max_cells,
                                                               int64_t pos0_arg, const int64_t* __restrict__ pos_ptr,
                                                               int n_top, int kp, bool tail, float scale,
                                                               float* __restrict__ partial, int max_chunks, int H) {
    constexpr int PER = L / 32;
    __shared__ __align__(16) __half tile[kCh][L];
    const int chunk = blockIdx.x, hg = blockIdx.y, t = blockIdx.z;
    const int64_t pos0 = pos_of(pos_ptr, pos0_arg);
    const int n_cells = cells_of(pos0 + t, n_top, kp, tail);
    const int c0 = chunk * kCh;
    if (c0 >= n_cells) return;
    const int nc = min(kCh, n_cells - c0);
    const int32_t* cl = cells + (size_t) t * max_cells;
    for (int e = threadIdx.x; e < kCh * L / 8; e += 256) {
        const int c = e / (L / 8), part = e % (L / 8);
        uint4 val = make_uint4(0, 0, 0, 0);
        if (c < nc) val = reinterpret_cast<const uint4*>(kv + (size_t) cl[c0 + c] * L)[part];
        reinterpret_cast<uint4*>(&tile[c][0])[part] = val;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int hh = 0; hh < 2; ++hh) {
        const int h = hg * kHg + warp * 2 + hh;
        const float* qh = q + ((size_t) t * H + h) * L;
        float qr[PER];
#pragma unroll
        for (int i = 0; i < PER; ++i) qr[i] = qh[lane + 32 * i];
        float m = -INFINITY, l = 0.0f, acc[PER];
#pragma unroll
        for (int i = 0; i < PER; ++i) acc[i] = 0.0f;
        for (int c = 0; c < nc; ++c) {
            float sv = 0.0f;
#pragma unroll
            for (int i = 0; i < PER; ++i) sv += qr[i] * __half2float(tile[c][lane + 32 * i]);
            sv = warp_sum(sv) * scale;
            const float mn = fmaxf(m, sv);
            const float corr = expf(m - mn), p = expf(sv - mn);
            l = l * corr + p;
#pragma unroll
            for (int i = 0; i < PER; ++i) acc[i] = acc[i] * corr + p * __half2float(tile[c][lane + 32 * i]);
            m = mn;
        }
        float* out = partial + (((size_t) t * H + h) * max_chunks + chunk) * (L + 2);
#pragma unroll
        for (int i = 0; i < PER; ++i) out[lane + 32 * i] = acc[i];
        if (lane == 0) {
            out[L] = m;
            out[L + 1] = l;
        }
    }
}

template <int L>
__global__ void __launch_bounds__(256) attend_merge_w_kernel(const float* __restrict__ partial, float* __restrict__ ctx,
                                                             int max_chunks, int64_t pos0_arg,
                                                             const int64_t* __restrict__ pos_ptr, int n_top, int kp,
                                                             bool tail, int H) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int n_chunks = (cells_of(pos_of(pos_ptr, pos0_arg) + t, n_top, kp, tail) + kCh - 1) / kCh;
    const float* base = partial + ((size_t) t * H + h) * max_chunks * (L + 2);
    float M = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, base[(size_t) c * (L + 2) + L]);
    float lsum = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float* pc = base + (size_t) c * (L + 2);
        lsum += expf(pc[L] - M) * pc[L + 1];
    }
    const float inv = 1.0f / lsum;
    float* out = ctx + ((size_t) t * H + h) * L;
    for (int d = threadIdx.x; d < L; d += 256) {
        float a = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            const float* pc = base + (size_t) c * (L + 2);
            a += expf(pc[L] - M) * pc[d];
        }
        out[d] = a * inv;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// THE PROMPT PATH'S DSA ATTENTION ON TENSOR CORES - Strata's qsa_prompt_attn v2 (src/kernels/cuda/qsa_prompt_attn.cu,
// "D-1") for the MLA latent.  Over the latent every one of a layer's 64 heads attends to the SAME cells with K = V = the
// cell's 512-value FP16 latent, so a block (one query, 16 heads) walks its cells in chunks of 32 with an online softmax,
// one MMA m16n8k16 per (16 heads, 8 cells, 16 dims):
//   * a warp owns 128 of the 512 dims (as Strata's warp owns 64 of 256): it gathers its own slice of the chunk's K/V
//     rows with cp.async into its own (optionally double-buffered) stage, holds its q fragments in registers, and the
//     latent slice in smem serves BOTH q.k (the warp's partial score, over its dims) and p.v (its output dims);
//   * only the q.k partial sums cross warps (smem, summed in a fixed order: deterministic), the softmax is the block's;
//   * q and p are split into FP16 hi + lo parts (two MMAs each) so they keep ~22 bits: the result differs from the
//     FP32 kernel by summation order and exp2's rounding, not by an FP16 cast (the latent itself is FP16, exact).
//   * q is pre-scaled by a power of two that puts its largest value near 2^14 (exact; keeps the lo half out of FP16's
//     subnormals), p by 2^14 (its max is 1); both are taken back out exactly.
// grid (head groups, queries): the four head groups of a query are neighbours in launch order and share its cells in L2.
// Not bitwise equal to attend_partial_w/merge; x_kernels_parity bounds the difference.  Decode windows (T <= 8) keep
// the split-K kernels (a single query needs the parallelism over cells).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define STRATA_X_MMA 1
#else
#define STRATA_X_MMA 0
#endif

constexpr int kMaCh = 32, kMaDims = 128, kMaStride = kMaDims + 8, kMaThreads = 128;

template <int NS>
struct MaSmem {
    __half kv[4][NS][kMaCh][kMaStride];   // warp, stage, cell, 128 dims (+8 pad: 272-byte rows spread the banks)
    float part[4][16][kMaCh + 1];
    float p[16][kMaCh + 1];
    float alpha[16], lsum[16], mrow[16], qmax[4];
};

__device__ __forceinline__ void ma_mma(float* c, const uint32_t* a, const uint32_t* b) {
#if STRATA_X_MMA
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}
__device__ __forceinline__ void ma_ldsm(uint32_t* r, const void* p) {
#if STRATA_X_MMA
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#endif
}
__device__ __forceinline__ void ma_ldsm_t(uint32_t* r, const void* p) {
#if STRATA_X_MMA
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#endif
}
__device__ __forceinline__ void ma_cp16(void* smem, const void* gmem, bool valid) {
#if STRATA_X_MMA
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
#endif
}
__device__ __forceinline__ void ma_commit() {
#if STRATA_X_MMA
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void ma_wait() {
#if STRATA_X_MMA
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}
__device__ __forceinline__ uint32_t ma_pack(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

template <int NS>
__global__ void __launch_bounds__(kMaThreads, (NS == 1 ? 2 : 1))
    attend_mma_kernel(const float* __restrict__ q, const __half* __restrict__ kv, const int32_t* __restrict__ cells,
                      int max_cells, int64_t pos0, int n_top, int kp, bool tail, float scale_log2,
                      float* __restrict__ ctx, int H) {
#if STRATA_X_MMA
    constexpr int L = 512;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    MaSmem<NS>& S = *reinterpret_cast<MaSmem<NS>*>(smem_raw);
    const int hg = blockIdx.x, t = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * kMaDims;
    const int n = cells_of(pos0 + t, n_top, kp, tail);
    const float* qb = q + ((size_t) t * H + (size_t) hg * 16) * L;
    float* ob = ctx + ((size_t) t * H + (size_t) hg * 16) * L;
    const int32_t* cl = cells + (size_t) t * max_cells;

    // q: the power-of-two prescale over the block's 16 heads, then this warp's dims as hi / lo A fragments
    float qm = 0.0f;
    for (int i = tid; i < 16 * L; i += kMaThreads) qm = fmaxf(qm, fabsf(qb[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (tid < 16) {
        S.mrow[tid] = -INFINITY;
        S.lsum[tid] = 0.0f;
    }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);   // qm < 2^qe
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    uint32_t qh[8][4], ql[8][4];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = gid + (r & 1) * 8, col = dim0 + kk * 16 + 2 * tig + (r >> 1) * 8;
            float2 x = *reinterpret_cast<const float2*>(qb + (size_t) row * L + col);
            x.x *= qup;
            x.y *= qup;
            const __half2 hi = __floats2half2_rn(x.x, x.y);
            const float2 hf = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(x.x - hf.x, x.y - hf.y);
            qh[kk][r] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t*>(&lo);
        }
    }

    const int n_chunks = (n + kMaCh - 1) / kMaCh;
    auto cell_of = [&](int c) -> int { return c < n ? __ldg(cl + c) : -1; };
    // the chunk's cells' slices of this warp's dims into stage `st`: lane j takes pieces lane + 32 j of 16 (cell, 16 B)
    auto issue = [&](int cell_lane, int st) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int idx = lane + 32 * j, c = idx >> 4, pc = idx & 15;
            const int cell = __shfl_sync(0xffffffffu, cell_lane, c);
            const bool ok = cell >= 0;
            ma_cp16(&S.kv[warp][st][c][pc * 8], kv + (size_t) (ok ? cell : 0) * L + dim0 + pc * 8, ok);
        }
    };
    float acc[16][4];   // 2^14 x this warp's output dims, 16 heads
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    int cell_next = cell_of(lane);
    issue(cell_next, 0);
    ma_commit();
    cell_next = cell_of(kMaCh + lane);

    for (int ci = 0; ci < n_chunks; ++ci) {
        const int st = NS == 2 ? (ci & 1) : 0, c0 = ci * kMaCh;
        if constexpr (NS == 2) {
            if (ci + 1 < n_chunks) issue(cell_next, st ^ 1);
            ma_commit();
            cell_next = cell_of((ci + 2) * kMaCh + lane);
            ma_wait<1>();
        } else {
            ma_wait<0>();
        }
        __syncwarp();
        // q.k over this warp's dims: 4 n-tiles of 8 cells, 8 k-steps (two per ldmatrix.x4), hi + lo
        {
            const __half* Kt = &S.kv[warp][st][0][0];
#pragma unroll
            for (int nt = 0; nt < kMaCh / 8; ++nt) {
                float tg[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kk2 = 0; kk2 < 4; ++kk2) {
                    uint32_t b[4];
                    ma_ldsm(b, Kt + (nt * 8 + (lane & 7)) * kMaStride + kk2 * 32 + (lane >> 3) * 8);
                    ma_mma(tg, qh[2 * kk2], b);
                    ma_mma(tg, ql[2 * kk2], b);
                    ma_mma(tg, qh[2 * kk2 + 1], b + 2);
                    ma_mma(tg, ql[2 * kk2 + 1], b + 2);
                }
                const int c = nt * 8 + 2 * tig;
                S.part[warp][gid][c] = tg[0];
                S.part[warp][gid][c + 1] = tg[1];
                S.part[warp][gid + 8][c] = tg[2];
                S.part[warp][gid + 8][c + 1] = tg[3];
            }
        }
        __syncthreads();
        // online softmax over the four warps' sum (fixed order): row tid/8, 4 cells per thread
        {
            const int r = tid >> 3, sub = tid & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = c0 + c < n ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v over this warp's output dims: the running sum rescaled, then the chunk's hi + lo MMAs straight into it
        {
            const __half* Vt = &S.kv[warp][st][0][0];
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                acc[j][0] *= a0;
                acc[j][1] *= a0;
                acc[j][2] *= a1;
                acc[j][3] *= a1;
            }
#pragma unroll
            for (int ks = 0; ks < kMaCh / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float p00 = S.p[gid][cA] * 16384.0f, p01 = S.p[gid][cA + 1] * 16384.0f;
                const float p10 = S.p[gid + 8][cA] * 16384.0f, p11 = S.p[gid + 8][cA + 1] * 16384.0f;
                const float p02 = S.p[gid][cB] * 16384.0f, p03 = S.p[gid][cB + 1] * 16384.0f;
                const float p12 = S.p[gid + 8][cB] * 16384.0f, p13 = S.p[gid + 8][cB + 1] * 16384.0f;
                uint32_t ah[4], al[4];
                ah[0] = ma_pack(p00, p01);
                ah[1] = ma_pack(p10, p11);
                ah[2] = ma_pack(p02, p03);
                ah[3] = ma_pack(p12, p13);
                {
                    const __half2* hh = reinterpret_cast<const __half2*>(ah);
                    float2 f;
                    f = __half22float2(hh[0]); al[0] = ma_pack(p00 - f.x, p01 - f.y);
                    f = __half22float2(hh[1]); al[1] = ma_pack(p10 - f.x, p11 - f.y);
                    f = __half22float2(hh[2]); al[2] = ma_pack(p02 - f.x, p03 - f.y);
                    f = __half22float2(hh[3]); al[3] = ma_pack(p12 - f.x, p13 - f.y);
                }
#pragma unroll
                for (int j2 = 0; j2 < 8; ++j2) {   // two n-tiles (16 dims) per transposed ldmatrix.x4
                    uint32_t b[4];
                    const int m = lane >> 3;
                    ma_ldsm_t(b, Vt + (ks * 16 + (lane & 7) + (m & 1) * 8) * kMaStride + (j2 * 2 + (m >> 1)) * 8);
                    ma_mma(acc[2 * j2], ah, b);
                    ma_mma(acc[2 * j2], al, b);
                    ma_mma(acc[2 * j2 + 1], ah, b + 2);
                    ma_mma(acc[2 * j2 + 1], al, b + 2);
                }
            }
        }
        __syncwarp();
        if constexpr (NS == 1) {   // this warp is done reading its stage: the next chunk's slice
            if (ci + 1 < n_chunks) issue(cell_next, 0);
            ma_commit();
            cell_next = cell_of((ci + 2) * kMaCh + lane);
        }
    }
    __syncthreads();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? (1.0f / 16384.0f) / l0 : 0.0f, i1 = l1 > 0.0f ? (1.0f / 16384.0f) / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int d = dim0 + j * 8 + 2 * tig;
        *reinterpret_cast<float2*>(ob + (size_t) gid * L + d) = make_float2(acc[j][0] * i0, acc[j][1] * i0);
        *reinterpret_cast<float2*>(ob + (size_t) (gid + 8) * L + d) = make_float2(acc[j][2] * i1, acc[j][3] * i1);
    }
#else
    __trap();   // built for a card without mma.sync m16n8k16 f16 (sm_80): the host keeps the split-K kernels
#endif
}

__global__ void combine_w_kernel(const float* __restrict__ parts, const float* __restrict__ w,
                                 const float* __restrict__ shared, float* __restrict__ out, int n, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    const float* pt = parts + (size_t) t * k * n;
    const float* wt = w + (size_t) t * k;
    float s = pt[i] * wt[0];
    for (int j = 1; j < k; ++j) s = s + pt[(size_t) j * n + i] * wt[j];
    out[(size_t) t * n + i] = shared ? s + shared[(size_t) t * n + i] : s;
}

/// A block per output row, 8 warps splitting n_in, all T columns at once (the row read once): 288 rows are 288
/// blocks - the warp-per-row form gave the router 36 blocks for 68 SMs (36 us a call).
__global__ void __launch_bounds__(256) gemv_f32_w_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                         float* __restrict__ y, int n_in, int n_out, int T) {
    __shared__ float red[8][8];
    const int row = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* wr = W + (size_t) row * n_in;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = threadIdx.x; i < n_in; i += 256) {
        const float wv = wr[i];
#pragma unroll
        for (int t = 0; t < 8; ++t)
            if (t < T) acc[t] += wv * x[(size_t) t * n_in + i];
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0) red[t][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < 8 && (int) threadIdx.x < T) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += red[threadIdx.x][w];
        y[(size_t) threadIdx.x * n_out + row] = v;
    }
}

/// moex (ERNIE55_PLAN P6): gemv_f32_w_kernel for one column at any width - each thread's values t, t + 256, ... loaded 16 at a
/// time ahead of their multiply-adds, which run in the same order on the same values, then the same warp and block sums: the
/// same bits.  The loop before waited one load a step: mHC's 16,384 x 24 read took 18 us a call (24 blocks, 64 steps each,
/// two calls a layer) in ERNIE-5.5's decode.  STRATA_X_F32_CHUNK=0: the loop as before.
__global__ void __launch_bounds__(256) gemv_f32_w1_chunk_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                                float* __restrict__ y, int n_in) {
    __shared__ float red[8];
    const int row = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* __restrict__ wr = W + (size_t) row * n_in;
    float acc = 0.0f;
    for (int i0 = threadIdx.x; i0 < n_in; i0 += 256 * 16) {
        float wv[16], xv[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int i = i0 + 256 * j;
            wv[j] = i < n_in ? __ldg(wr + i) : 0.0f;
            xv[j] = i < n_in ? __ldg(x + i) : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < 16; ++j)
            if (i0 + 256 * j < n_in) acc += wv[j] * xv[j];
    }
    const float v = warp_sum(acc);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) s += red[w];
        y[row] = s;
    }
}

/// moex (the ERNIE-5.5 hill-climb): gemv_f32_w1_chunk_kernel's rows cut into their 8 warps' shares - block (row, w) is that
/// kernel's warp w of block `row` (threads t = 32 w + lane: the same loads 16 ahead, the same chain, the same warp_sum) and
/// part[row * 8 + w] its warp's total; the block sum (0.0f + part[0] + ... + part[7], in order) is the reader's
/// (mhc_pre_norm_w1_kernel).  mHC's 24 rows of 16,384 on 192 blocks: the chunk kernel ran 24 blocks on 24 SMs (5.6 us a mixer)
__global__ void __launch_bounds__(32) gemv_f32_w1_warps_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                               float* __restrict__ part, int n_in) {
    const int row = blockIdx.x, warp = blockIdx.y, lane = threadIdx.x;
    const int t = 32 * warp + lane;
    const float* __restrict__ wr = W + (size_t) row * n_in;
    float acc = 0.0f;
    for (int i0 = t; i0 < n_in; i0 += 256 * 16) {
        float wv[16], xv[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int i = i0 + 256 * j;
            wv[j] = i < n_in ? __ldg(wr + i) : 0.0f;
            xv[j] = i < n_in ? __ldg(x + i) : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < 16; ++j)
            if (i0 + 256 * j < n_in) acc += wv[j] * xv[j];
    }
    const float v = warp_sum(acc);
    if (lane == 0) part[(size_t) row * 8 + warp] = v;
}

/// moex (DSV41_PLAN §36), STRATA_DSV41_PRELOAD=1: gemv_f32_w_kernel for one column (a decode window's) with each thread's
/// (at most PT) weights and inputs loaded first, then the same multiply-adds in the same order, the one warp sum a column has
/// (not eight), the same block sum: the same bits.  The routers' 5,120 x 384 read waited one load after another
template <int PT>
__global__ void __launch_bounds__(256) gemv_f32_w1_pre_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                              float* __restrict__ y, int n_in) {
    __shared__ float red[8];
    const int row = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* wr = W + (size_t) row * n_in;
    float wv[PT], xv[PT];
#pragma unroll
    for (int j = 0; j < PT; ++j) {
        const int i = threadIdx.x + 256 * j;
        wv[j] = i < n_in ? wr[i] : 0.0f;
        xv[j] = i < n_in ? x[i] : 0.0f;
    }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < PT; ++j)
        if (threadIdx.x + 256 * j < n_in) acc += wv[j] * xv[j];
    const float v = warp_sum(acc);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) s += red[w];
        y[row] = s;
    }
}

/// moex (DSV41_PLAN §62): a BF16 weight as the F32 its exact copy holds
__device__ __forceinline__ float bf16_up(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

/// gemv_f32_w_kernel with BF16 weights: each weight widened as it loads, every multiply-add and sum as there - the same bits
__global__ void __launch_bounds__(256) gemv_bf16_w_kernel(const uint16_t* __restrict__ W, const float* __restrict__ x,
                                                          float* __restrict__ y, int n_in, int n_out, int T) {
    __shared__ float red[8][8];
    const int row = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint16_t* wr = W + (size_t) row * n_in;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = threadIdx.x; i < n_in; i += 256) {
        const float wv = bf16_up(wr[i]);
#pragma unroll
        for (int t = 0; t < 8; ++t)
            if (t < T) acc[t] += wv * x[(size_t) t * n_in + i];
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0) red[t][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < 8 && (int) threadIdx.x < T) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += red[threadIdx.x][w];
        y[(size_t) threadIdx.x * n_out + row] = v;
    }
}

/// gemv_f32_w1_pre_kernel with BF16 weights, the same way
template <int PT>
__global__ void __launch_bounds__(256) gemv_bf16_w1_pre_kernel(const uint16_t* __restrict__ W, const float* __restrict__ x,
                                                               float* __restrict__ y, int n_in) {
    __shared__ float red[8];
    const int row = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint16_t* wr = W + (size_t) row * n_in;
    float wv[PT], xv[PT];
#pragma unroll
    for (int j = 0; j < PT; ++j) {
        const int i = threadIdx.x + 256 * j;
        wv[j] = i < n_in ? bf16_up(wr[i]) : 0.0f;
        xv[j] = i < n_in ? x[i] : 0.0f;
    }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < PT; ++j)
        if (threadIdx.x + 256 * j < n_in) acc += wv[j] * xv[j];
    const float v = warp_sum(acc);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) s += red[w];
        y[row] = s;
    }
}

/// moex (Hy4-flash's verify windows): gemv_bf16_w_kernel for NC = 2 .. 8 columns with the row preloaded - each thread's PT
/// values (i = tid + 256 j) read once into registers, then every column's multiply-adds over them in ascending j: the general
/// kernel's per-thread order, its eight warp sums and its warp order, so each output is that kernel's (and the one-column
/// preload's) bit for bit.  The general kernel reads a value at a time, 2 bytes a thread: 3.4 ms against ~1.1 for the
/// 3,072 x 120,832 head at 2 .. 6 columns.
template <int PT, int NC>
__global__ void __launch_bounds__(256) gemv_bf16_wn_pre_kernel(const uint16_t* __restrict__ W, const float* __restrict__ x,
                                                               float* __restrict__ y, int n_in, int n_out) {
    __shared__ float red[8][8];
    const int row = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint16_t* wr = W + (size_t) row * n_in;
    float wv[PT];
#pragma unroll
    for (int j = 0; j < PT; ++j) {
        const int i = threadIdx.x + 256 * j;
        wv[j] = i < n_in ? bf16_up(wr[i]) : 0.0f;
    }
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int t = 0; t < NC; ++t) {
        const float* xt = x + (size_t) t * n_in;
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            const int i = threadIdx.x + 256 * j;
            if (i < n_in) acc[t] += wv[j] * xt[i];
        }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0) red[t][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < NC) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += red[threadIdx.x][w];
        y[(size_t) threadIdx.x * n_out + row] = v;
    }
}

/// gemv_bf16_w_kernel with FP8 weights: the same multiply-adds, in the same order, on the same F32 values - the same bits -
/// but the row read in tiles of 4,096 values with one 16-byte load a thread into shared memory (gemv_bf16_w reads one value
/// a thread a step, so half the bytes would take the same number of loads); thread t then takes values t, t + 256, ... from
/// the tile, as gemv_bf16_w's thread t takes them from the row.  n_in a multiple of 16 (the loader takes multiples of 32)
/// moex (DSV41_PLAN §70): an e4m3 value (OCP E4M3FN: bias 7, no infinities) times its block's e8m0 scale - exact (4
/// significant bits times a power of two): the F32 its BF16 weight held, which the loader checked value by value
__device__ __forceinline__ float e4m3_val(uint32_t q, float sc) {   // branch-free: no table, no bank conflicts
    const uint32_t em = q & 0x7Fu;
    const float vn = __uint_as_float((em << 20) + (120u << 23));    // exponent >= 1: 2^(e - 7) (1 + m / 8)
    const float vs = (float) (em & 7u) * 0.001953125f;               // exponent 0: m x 2^-9
    const float v = (em >> 3) ? vn : vs;
    return __uint_as_float(__float_as_uint(v * sc) | ((q & 0x80u) << 24));   // (the sign last: -(v s) is (-v) s)
}
/// moex (DSV41_BACKLOG #137, DSV41_PLAN §75): the same value from the scale's EXPONENT, sa = (S - 7) << 23, instead of its
/// float - the multiply by a power of two becomes an integer add of the exponent field, 6 operations a weight against ~12.
///   exponent >= 1: e4m3_val's vn x 2^(S - 127) has the exponent (em >> 3) + 120 + S - 127 = (em >> 3) + (S - 7), and the
///   mantissa (em & 7) << 20 either way - so (em << 20) + sa holds the same bits (no rounding: a power of two).
///   exponent 0 (0.008% of the release's codes): em x 2^-9 x 2^(S - 127) = em x 2^(S - 136), the float sa - (2 << 23).
/// S >= 10 is checked at load (fp8_block refuses the matrix otherwise): sa - (2 << 23) is then the exponent field S - 9 >= 1,
/// a normal power of two (at S = 9 it would be 0.0 and the code's denormal value lost).
__device__ __noinline__ float e4m3_sa(uint32_t q, uint32_t sa) {   // (out of line: see gemv_fp8_rows_body's marked blocks)
    const uint32_t em = q & 0x7Fu;
    uint32_t b = (em << 20) + sa;
    if (em < 8u) b = __float_as_uint((float) em * __uint_as_float(sa - (2u << 23)));
    return __uint_as_float(b | ((q & 0x80u) << 24));
}
/// The scale byte without its mark (bit 7: the block holds an exponent-0 code; Engine::Impl's fp8_block sets it at load).
constexpr uint32_t kFp8ScaleMask = 0x7Fu;
__global__ void __launch_bounds__(256) gemv_fp8_w_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                                         const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out,
                                                         int T) {
    __shared__ float red[8][8];
    __shared__ __align__(16) uint8_t tq[4096];
    __shared__ float tsc[128];
    const int row = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint8_t* wr = W + (size_t) row * n_in;
    const uint8_t* sr = S + (size_t) row * (n_in >> 5);
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int base = 0; base < n_in; base += 4096) {
        const int len = min(4096, n_in - base);
        __syncthreads();   // (the tile before is consumed)
        if (16 * (int) threadIdx.x < len) *reinterpret_cast<uint4*>(tq + 16 * threadIdx.x) = *reinterpret_cast<const uint4*>(wr + base + 16 * threadIdx.x);
        if ((int) threadIdx.x < (len >> 5)) tsc[threadIdx.x] = __uint_as_float(((uint32_t) sr[(base >> 5) + threadIdx.x] & kFp8ScaleMask) << 23);
        __syncthreads();
        for (int i = threadIdx.x; i < len; i += 256) {
            const float wv = e4m3_val(tq[i], tsc[i >> 5]);
#pragma unroll
            for (int t = 0; t < 8; ++t)
                if (t < T) acc[t] += wv * x[(size_t) t * n_in + base + i];
        }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0) red[t][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < 8 && (int) threadIdx.x < T) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += red[threadIdx.x][w];
        y[(size_t) threadIdx.x * n_out + row] = v;
    }
}

/// gemv_fp8_w at T 1, R rows a block: each row's products and sums as the one-row kernels' (thread t takes values t, t + 256,
/// ... of every row, in that order - gemv_bf16_w_kernel's and gemv_bf16_w1_pre_kernel's order; the warps' partials and the warp
/// sum likewise) - the same bits - with x read once for the block's R rows and the rows read by 16-byte loads into shared
/// memory, in tiles of 2,048 values (moex, DSV41_PLAN §70: the one-row kernels ran the decode's projections at 150-500 GB/s)
template <int R, int TT>
__device__ __forceinline__ void gemv_fp8_rows_body(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                                   const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out,
                                                   int T) {
    __shared__ float red[R * TT][8];
    __shared__ __align__(16) uint8_t tq[R][2048];
    __shared__ uint32_t tsc[R][64];
    const int row0 = blockIdx.x * R;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[R * TT];
#pragma unroll
    for (int r = 0; r < R * TT; ++r) acc[r] = 0.0f;
    for (int base = 0; base < n_in; base += 2048) {
        const int len = min(2048, n_in - base);   // (a multiple of 16: n_in is one of 32)
        __syncthreads();   // (the tiles before are consumed)
        // moex (§75): a row past n_out (a block's last rows when n_out is no multiple of R) gets a tile of zeros and the scale
        // word 0, so the products below run for all R rows with no test a weight - its sums are computed and not written
        for (int o = 16 * threadIdx.x; o < R * len; o += 16 * 256) {
            const int r = o / len, c = o - r * len;
            *reinterpret_cast<uint4*>(&tq[r][c]) = row0 + r < n_out
                ? *reinterpret_cast<const uint4*>(W + (size_t) (row0 + r) * n_in + base + c) : make_uint4(0u, 0u, 0u, 0u);
        }
        const int lb = len >> 5;
        for (int o = threadIdx.x; o < R * lb; o += 256) {
            const int r = o / lb, c = o - r * lb;
            // the scale's exponent field, e4m3_sa's sa (its float is never formed), and the block's mark at bit 31
            uint32_t sw = 3u << 23;
            if (row0 + r < n_out) {
                const uint32_t sb = S[(size_t) (row0 + r) * (n_in >> 5) + (base >> 5) + c];
                sw = (((sb & kFp8ScaleMask) - 7u) << 23) | ((sb & 0x80u) << 24);
            }
            tsc[r][c] = sw;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < len; i += 256) {
            // A step's R codes and scale words; a warp's lanes sit in ONE 32-block of each row, so "is any of the R blocks
            // marked" is the same for the whole warp - a branch that does not split it, taken for ~1% of the steps.
            uint32_t q[R], sa[R], any = 0u;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                q[r] = tq[r][i];
                sa[r] = tsc[r][i >> 5];
                any |= sa[r];
            }
            float wv[R];
            if (__builtin_expect((int32_t) any < 0, 0)) {   // a marked block among them: the general decode, each code
#pragma unroll
                for (int r = 0; r < R; ++r) wv[r] = e4m3_sa(q[r], sa[r] & 0x7FFFFFFFu);
            } else {
                // every code's exponent >= 1: q x 2^20 + sa puts the exponent and the mantissa in place (and the sign at bit
                // 27); (q & 0x80) x 0xF00000 = sign x (2^31 - 2^27) moves the sign to bit 31.  Three integer instructions
#pragma unroll
                for (int r = 0; r < R; ++r) wv[r] = __uint_as_float(q[r] * 0x100000u + sa[r] + (q[r] & 0x80u) * 0xF00000u);
            }
            if constexpr (TT == 1) {   // one token: no test a weight
                const float xv = x[base + i];
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r] += wv[r] * xv;
            } else {
                float xv[TT];
#pragma unroll
                for (int t = 0; t < TT; ++t) xv[t] = t < T ? x[(size_t) t * n_in + base + i] : 0.0f;
#pragma unroll
                for (int r = 0; r < R; ++r) {
#pragma unroll
                    for (int t = 0; t < TT; ++t)
                        if (t < T) acc[r * TT + t] += wv[r] * xv[t];
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R * TT; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) red[r][warp] = v;
    }
    __syncthreads();
    if ((int) threadIdx.x < R * TT) {
        const int r = (int) threadIdx.x / TT, t = (int) threadIdx.x % TT;
        if (row0 + r < n_out && t < T) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) v += red[threadIdx.x][w];
            y[(size_t) t * n_out + row0 + r] = v;
        }
    }
}

template <int R, int TT>
__global__ void __launch_bounds__(256) gemv_fp8_rows_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                                            const float* __restrict__ x, float* __restrict__ y, int n_in,
                                                            int n_out, int T) {
    gemv_fp8_rows_body<R, TT>(W, S, x, y, n_in, n_out, T);
}

/// moex (ERNIE55_PLAN, P6): gemv_fp8_w at T 1 with ONE WARP A ROW and no shared memory - the same bits as the row kernels.
/// Those kernels' order: thread t (of 256) sums the products of values t, t + 256, t + 512, ... in turn (one FMA each, from
/// 0.0f), warp_sum's xor tree (16, 8, 4, 2, 1) joins each warp's 32 partials, and the 8 warps' totals are added w = 0 .. 7
/// from 0.0f.  Here lane l reads 8 codes at 8 l + 256 k by one 8-byte load (a warp: 256 contiguous bytes a step, every
/// step's load issued ahead of its use) and keeps the 8 partials of threads t = 8 l + j, adding term k exactly as thread t
/// did; thread t's warp is l / 4 and its lane in that warp 8 (l % 4) + j, so warp_sum's xor 16 and 8 are this warp's xor 2
/// and 1 (shuffles of each partial), its 4, 2, 1 the partials' own j ^ 4, j ^ 2, j ^ 1 (adds in the lane), and warp w's
/// total sits in lanes 4 w .. 4 w + 3.  An add takes the same two operands at every step (IEEE addition commutes), so every
/// output is the row kernels' to the bit (fp8_gemv_bench, ERNIE's shapes).  The row kernels stage each row in shared memory
/// and sync twice a tile: ERNIE's decode ran them at 190-460 GB/s (q_a 4.3 MB 13.9 us, q_b 34.6 MB 80.7 us).
template <int R>
__global__ void __launch_bounds__(256) gemv_fp8_warp_kernel(const uint8_t* W, const uint8_t* S, const float* x, float* y,
                                                            int n_in, int n_out) {
    // R rows a warp (rows r0 .. r0 + R - 1 share each step's 8 inputs); a row past n_out reads row r0's bytes again and is
    // not written (its sums are computed with the warp's - no test a step)
    const int lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * R;
    if (r0 >= n_out) return;   // (the whole warp)
    if (gridDim.y > 1) {   // gemv_fp8_heads: group blockIdx.y - its own weights, input and output (one token)
        const size_t h = blockIdx.y;
        W += h * (size_t) n_out * n_in;
        S += h * ((size_t) n_out * n_in >> 5);
        x += h * (size_t) n_in;
        y += h * (size_t) n_out;
    }
    const uint8_t* __restrict__ wr[R];
    const uint8_t* __restrict__ sr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = r0 + r < n_out ? r0 + r : r0;
        wr[r] = W + (size_t) row * n_in;
        sr[r] = S + (size_t) row * (n_in >> 5);
    }
    float acc[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r][j] = 0.0f;
    // n_in a multiple of 32: a chunk that starts below n_in ends there too
#pragma unroll 2
    for (int i0 = 8 * lane; i0 < n_in; i0 += 256) {
        const float4 xa = __ldg(reinterpret_cast<const float4*>(x + i0));
        const float4 xb = __ldg(reinterpret_cast<const float4*>(x + i0 + 4));
        const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        uint2 qq[R];
        uint32_t sb[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            qq[r] = __ldg(reinterpret_cast<const uint2*>(wr[r] + i0));
            sb[r] = __ldg(sr[r] + (i0 >> 5));
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint32_t sa = ((sb[r] & kFp8ScaleMask) - 7u) << 23;
            float wv[8];
            if (__builtin_expect((sb[r] & 0x80u) != 0u, 0)) {   // a block with an exponent-0 code: the general decode
#pragma unroll
                for (int j = 0; j < 8; ++j) wv[j] = e4m3_sa(((j < 4 ? qq[r].x : qq[r].y) >> (8 * (j & 3))) & 0xFFu, sa);
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t q = ((j < 4 ? qq[r].x : qq[r].y) >> (8 * (j & 3))) & 0xFFu;
                    wv[j] = __uint_as_float(q * 0x100000u + sa + (q & 0x80u) * 0xF00000u);
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[r][j] += wv[j] * xv[j];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        // warp_sum's xor 16 and 8: the threads 16 and 8 apart are this warp's lanes 2 and 1 apart (the same partial j)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r][j] += __shfl_xor_sync(0xffffffffu, acc[r][j], 2);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[r][j] += __shfl_xor_sync(0xffffffffu, acc[r][j], 1);
        // its xor 4, 2, 1: partials j and j ^ 4, then ^ 2, then ^ 1 (partial 0's chain: the total is in every one)
        const float s4a = acc[r][0] + acc[r][4], s4b = acc[r][2] + acc[r][6], s4c = acc[r][1] + acc[r][5], s4d = acc[r][3] + acc[r][7];
        const float s2a = s4a + s4b, s2b = s4c + s4d;
        const float wt = s2a + s2b;   // the total of the row kernels' warp lane / 4
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += __shfl_sync(0xffffffffu, wt, 4 * w);
        if (lane == 0 && r0 + r < n_out) y[r0 + r] = v;
    }
}

/// moex (ERNIE55_PLAN P6): gemv_fp8_warp_kernel for a window of T <= TT columns (2 .. 8: a verify window's, the prompt
/// lookup's) - one warp a row, each step's 8 weights decoded once and multiplied into the 8 partials of every column; each
/// (row, column) sum keeps the row kernels' order (their acc[r * TT + t] += wv * x[t]: thread t's values in turn, warp_sum's
/// tree, the 8 warps from 0.0f), so the bits are theirs.  The row kernels took 2.4-4x their one-column time for 4-8 columns
/// (fp8_gemv_bench: q_b 96 -> 228 / 368 / 380 us) though the weights are read once.
template <int TT>
__global__ void __launch_bounds__(256) gemv_fp8_warp_t_kernel(const uint8_t* W, const uint8_t* S, const float* x, float* y,
                                                              int n_in, int n_out, int T) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= n_out) return;   // (the whole warp)
    if (gridDim.y > 1) {   // gemv_fp8_heads: group blockIdx.y
        const size_t h = blockIdx.y;
        W += h * (size_t) n_out * n_in;
        S += h * ((size_t) n_out * n_in >> 5);
        x += h * (size_t) T * n_in;
        y += h * (size_t) T * n_out;
    }
    const uint8_t* __restrict__ wr = W + (size_t) row * n_in;
    const uint8_t* __restrict__ sr = S + (size_t) row * (n_in >> 5);
    float acc[TT][8];
#pragma unroll
    for (int t = 0; t < TT; ++t)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[t][j] = 0.0f;
    for (int i0 = 8 * lane; i0 < n_in; i0 += 256) {
        const uint2 qq = __ldg(reinterpret_cast<const uint2*>(wr + i0));
        const uint32_t sb = __ldg(sr + (i0 >> 5));
        const uint32_t sa = ((sb & kFp8ScaleMask) - 7u) << 23;
        float wv[8];
        if (__builtin_expect((sb & 0x80u) != 0u, 0)) {
#pragma unroll
            for (int j = 0; j < 8; ++j) wv[j] = e4m3_sa(((j < 4 ? qq.x : qq.y) >> (8 * (j & 3))) & 0xFFu, sa);
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t q = ((j < 4 ? qq.x : qq.y) >> (8 * (j & 3))) & 0xFFu;
                wv[j] = __uint_as_float(q * 0x100000u + sa + (q & 0x80u) * 0xF00000u);
            }
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t < T) {
                const float4 xa = __ldg(reinterpret_cast<const float4*>(x + (size_t) t * n_in + i0));
                const float4 xb = __ldg(reinterpret_cast<const float4*>(x + (size_t) t * n_in + i0 + 4));
                const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[t][j] += wv[j] * xv[j];
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        if (t >= T) break;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[t][j] += __shfl_xor_sync(0xffffffffu, acc[t][j], 2);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[t][j] += __shfl_xor_sync(0xffffffffu, acc[t][j], 1);
        const float s4a = acc[t][0] + acc[t][4], s4b = acc[t][2] + acc[t][6], s4c = acc[t][1] + acc[t][5], s4d = acc[t][3] + acc[t][7];
        const float s2a = s4a + s4b, s2b = s4c + s4d;
        const float wt = s2a + s2b;
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += __shfl_sync(0xffffffffu, wt, 4 * w);
        if (lane == 0) y[(size_t) t * n_out + row] = v;
    }
}

/// moex (ERNIE55_PLAN P6, the hill-climb): TWO one-column products of the same input in ONE launch - each row by the code
/// its own kernel runs, so each output is that kernel's to the bit: an FP8 row by gemv_fp8_warp_kernel<1>'s warp (8 rows a
/// block), a BF16 row by gemv_bf16_w1_pre_kernel's block (a row a block).  Blocks [0, nb_a) take matrix A's rows, the rest
/// B's: a block's path is uniform (the BF16 block's syncs see every thread).  One launch where there were two: the second's
/// launch and tail (a decode layer's q_a + KV row, the compressor's two rows, the routers + routed_down).
template <int PT>
__device__ __forceinline__ void pair_bf16_row(const uint16_t* __restrict__ W, const float* __restrict__ x, float* __restrict__ y,
                                              int n_in, int row) {
    __shared__ float red[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint16_t* wr = W + (size_t) row * n_in;
    float wv[PT], xv[PT];
#pragma unroll
    for (int j = 0; j < PT; ++j) {
        const int i = threadIdx.x + 256 * j;
        wv[j] = i < n_in ? bf16_up(wr[i]) : 0.0f;
        xv[j] = i < n_in ? x[i] : 0.0f;
    }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < PT; ++j)
        if (threadIdx.x + 256 * j < n_in) acc += wv[j] * xv[j];
    const float v = warp_sum(acc);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) s += red[w];
        y[row] = s;
    }
}

__device__ __forceinline__ void pair_fp8_row(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ x,
                                             float* __restrict__ y, int n_in, int row) {   // gemv_fp8_warp_kernel<1>'s warp
    const int lane = threadIdx.x & 31;
    const uint8_t* __restrict__ wr = W + (size_t) row * n_in;
    const uint8_t* __restrict__ sr = S + (size_t) row * (n_in >> 5);
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
#pragma unroll 2
    for (int i0 = 8 * lane; i0 < n_in; i0 += 256) {
        const float4 xa = __ldg(reinterpret_cast<const float4*>(x + i0));
        const float4 xb = __ldg(reinterpret_cast<const float4*>(x + i0 + 4));
        const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        const uint2 qq = __ldg(reinterpret_cast<const uint2*>(wr + i0));
        const uint32_t sb = __ldg(sr + (i0 >> 5));
        const uint32_t sa = ((sb & kFp8ScaleMask) - 7u) << 23;
        float wv[8];
        if (__builtin_expect((sb & 0x80u) != 0u, 0)) {
#pragma unroll
            for (int j = 0; j < 8; ++j) wv[j] = e4m3_sa(((j < 4 ? qq.x : qq.y) >> (8 * (j & 3))) & 0xFFu, sa);
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t q = ((j < 4 ? qq.x : qq.y) >> (8 * (j & 3))) & 0xFFu;
                wv[j] = __uint_as_float(q * 0x100000u + sa + (q & 0x80u) * 0xF00000u);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] += wv[j] * xv[j];
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 2);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], 1);
    const float s4a = acc[0] + acc[4], s4b = acc[2] + acc[6], s4c = acc[1] + acc[5], s4d = acc[3] + acc[7];
    const float s2a = s4a + s4b, s2b = s4c + s4d;
    const float wt = s2a + s2b;
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) v += __shfl_sync(0xffffffffu, wt, 4 * w);
    if (lane == 0) y[row] = v;
}

template <int PT, int MODE>   // MODE 0: FP8 + FP8, 1: BF16 + BF16, 2: BF16 + FP8 (each instantiation holds only its paths)
__global__ void __launch_bounds__(256) gemv_pair_kernel(const void* __restrict__ wa, const uint8_t* __restrict__ sa, float* __restrict__ ya,
                                                        int na, const void* __restrict__ wb, const uint8_t* __restrict__ sb,
                                                        float* __restrict__ yb, int nb, int nb_a, const float* __restrict__ x, int n_in) {
    const bool in_a = (int) blockIdx.x < nb_a;
    const int blk = in_a ? (int) blockIdx.x : (int) blockIdx.x - nb_a;
    const void* w = in_a ? wa : wb;      // scalar selects (a struct chosen by reference went to the stack: 64 bytes a thread)
    float* y = in_a ? ya : yb;
    const bool fp8 = MODE == 0 ? true : MODE == 1 ? false : !in_a;
    if constexpr (MODE != 1) {
        if (fp8) {
            const uint8_t* s = in_a ? sa : sb;
            const int n_out = in_a ? na : nb;
            const int row = blk * 8 + (threadIdx.x >> 5);
            if (row < n_out) pair_fp8_row(static_cast<const uint8_t*>(w), s, x, y, n_in, row);
            return;
        }
    }
    if constexpr (MODE != 0) pair_bf16_row<PT>(static_cast<const uint16_t*>(w), x, y, n_in, blk);
}

static bool fp8_warp_on() {   // STRATA_X_FP8_WARP=0: the row kernels for every width
    static const bool on = [] {
        const char* e = std::getenv("STRATA_X_FP8_WARP");
        return !(e && e[0] == '0');
    }();
    return on;
}

/// gemv_bf16_w1_pre_kernel with FP8 weights (T 1, n_in <= 256 PT): the row into shared memory by 16-byte loads while the inputs
/// load, then each thread's PT values widened and multiplied in gemv_bf16_w1_pre_kernel's order - the same bits
template <int PT>
__global__ void __launch_bounds__(256) gemv_fp8_w1_pre_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                                              const float* __restrict__ x, float* __restrict__ y, int n_in) {
    __shared__ float red[8];
    __shared__ __align__(16) uint8_t tq[256 * PT];
    __shared__ float tsc[256 * PT / 32];
    const int row = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint8_t* wr = W + (size_t) row * n_in;
    const uint8_t* sr = S + (size_t) row * (n_in >> 5);
    for (int o = 16 * threadIdx.x; o < n_in; o += 16 * 256)
        *reinterpret_cast<uint4*>(tq + o) = *reinterpret_cast<const uint4*>(wr + o);
    for (int o = threadIdx.x; o < (n_in >> 5); o += 256) tsc[o] = __uint_as_float(((uint32_t) sr[o] & kFp8ScaleMask) << 23);
    float wv[PT], xv[PT];
#pragma unroll
    for (int j = 0; j < PT; ++j) {
        const int i = threadIdx.x + 256 * j;
        xv[j] = i < n_in ? x[i] : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < PT; ++j) {
        const int i = threadIdx.x + 256 * j;
        wv[j] = i < n_in ? e4m3_val(tq[i], tsc[i >> 5]) : 0.0f;
    }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < PT; ++j)
        if (threadIdx.x + 256 * j < n_in) acc += wv[j] * xv[j];
    const float v = warp_sum(acc);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) s += red[w];
        y[row] = s;
    }
}

__global__ void fp8_widen_kernel(const uint8_t* __restrict__ src, const uint8_t* __restrict__ S, float* __restrict__ dst, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = e4m3_val(src[i], __uint_as_float(((uint32_t) S[i >> 5] & kFp8ScaleMask) << 23));
}

/// moex (ERNIE55_PLAN, opt-in STRATA_X_TC3=1): an FP8 matrix's values as BF16 - e4m3_val's exact value (at most 4 significant
/// bits times a power of two, the low 16 bits zero), its upper half: exactly the value fp8_widen gives
__global__ void fp8_bf16_kernel(const uint8_t* __restrict__ src, const uint8_t* __restrict__ S, uint16_t* __restrict__ dst, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = (uint16_t) (__float_as_uint(e4m3_val(src[i], __uint_as_float(((uint32_t) S[i >> 5] & kFp8ScaleMask) << 23))) >> 16);
}

/// x as three BF16 parts, x = hi + mid + lo exactly: hi = bf16(x), mid = bf16(x - hi), lo = (x - hi) - mid (each difference
/// exact in F32; the rest has at most 8 significant bits).  Planes of n: hi, mid, lo.  (Activations far from BF16's subnormals.)
__device__ __forceinline__ uint32_t bf16_rn_bits(float v) {   // round to nearest even, finite values
    const uint32_t u = __float_as_uint(v);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}
__global__ void split3_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ p, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const float v = x[i];
        const uint32_t h = bf16_rn_bits(v);
        const float r1 = __fsub_rn(v, __uint_as_float(h << 16));
        const uint32_t m = bf16_rn_bits(r1);
        const float r2 = __fsub_rn(r1, __uint_as_float(m << 16));
        p[i] = (uint16_t) h;
        p[n + i] = (uint16_t) m;
        p[2 * n + i] = (uint16_t) bf16_rn_bits(r2);
    }
}

__global__ void bf16_widen_kernel(const uint16_t* __restrict__ src, float* __restrict__ dst, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        dst[i] = bf16_up(src[i]);
}

/// THE PER-HEAD PROJECTIONS OF THE MLA (absorbing wk_b into q, wv_b out of the latent) in one launch: head h's
/// Q8_0 matrix (n_out rows of n_in) against its own T Q8_1 columns, y (H, T, n_out).  The Q8_0 x Q8_1 block dot of
/// the native kernels (DP4A per 4 values, d_w * d_x per block); one warp per row, T accumulators per lane.  Replaces
/// 2 x 64 MMVQ launches per DSA layer.
struct HQ80 {
    half d;
    int8_t qs[32];
};
struct HQ81 {
    half2 ds;
    int8_t qs[32];
};
__device__ __forceinline__ int load_i16x2(const int8_t* p, int i) {   // an int from 2-byte-aligned bytes
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

/// moex (2026-11-19): heads_q8_0_kernel for a decode window (TC <= 8 columns), R rows a warp: each row's lane chains (its groups
/// e = lane, lane + 32, ... in order; FMUL(d_w, d_x) then FFMA, as that kernel compiles them) and warp_sum per column exactly
/// as there - the same bits - with the activations' groups loaded once for the R rows and R rows' loads in flight a lane
/// (one row a warp left the per-head matrices at ~184 GB/s).  hq8d.cu (scratchpad): 1-8 columns on k_b / v_b shapes, 0
/// values differ; 64 heads x 512 x 192 at one column 39.4 -> 16.0 us, at six 104.6 -> 34.1.  STRATA_X_HEADS_ROWS=0 keeps the
/// one-row kernel.
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
            const float sum = warp_sum(acc[k][t]);
            if (lane == 0 && r0 + k < n_out) Y[((size_t) h * T_total + c0 + t) * n_out + r0 + k] = sum;
        }
}

/// moex (Hy4-flash's prompt path, 2026-11-19): heads_q8_0_kernel's outputs for T > 8 columns in one launch, bit for bit, for
/// rows of 128, 192 or 256 values (NB = 4 / 6 / 8 blocks; Hy4-flash's k_b rows are its 192 nope values).  heads_q8_0 ran that
/// kernel 8 columns at a time - a 512-row prompt sub-batch 64 launches, the heads' weights re-read by each: 2.8 of GPU 0's
/// 8.5 s of kernels in an 8K-token read.  Here a thread takes 4 rows x 4 columns of one head out of shared memory (each
/// group of 4 values transposed there: a thread's 4 rows or columns of it in one 16-byte read) and makes each output as the
/// kernel's lane 0 ends with it: lane l's partial is that lane's chain over its groups e = l, l + 32 (< 8 NB) from 0 -
/// FMUL(d_w, d_x), then FFMA(that, dot, acc), as the kernel compiles it - the dot's int -> float by the exact magic add (|dot|
/// <= 4 x 127 x 128 < 2^22: the same float as I2F, at full rate), and the 32 partials are added as warp_sum adds them (l +
/// (l ^ 16), then ^ 8, ^ 4, ^ 2, ^ 1), in bit-reversed pairs so an output keeps a few live sums.  hq8.cu (scratchpad): 16
/// shapes, 0 values differ; 64 heads x 512 x 192 at 512 columns 8.38 -> 0.85 ms.  STRATA_X_HEADS_TILED=0 keeps the loop.
__device__ __forceinline__ float i2f_exact(int d) { return __fsub_rn(__int_as_float(0x4B400000 + d), 12582912.0f); }

template <int NB>
__global__ void __launch_bounds__(256) heads_q8_0_tiled_kernel(const HQ80* __restrict__ W, const HQ81* __restrict__ X,
                                                               float* __restrict__ Y, int n_out, int T) {
    constexpr int RT = 4, CT = 4, RB = 16 * RT, CB = 16 * CT, NG = NB * 8;
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
    auto lane_part = [&](int l, float (&p)[RT][CT]) {   // lane l's acc for this thread's 16 outputs
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
    auto pair = [&](int l, float (&v)[RT][CT]) {   // warp_sum's first step: p_l + p_(l + 16)
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
    // v4[0] = ((v1[0] + v1[8]) + (v1[4] + v1[12])) + ((v1[2] + v1[10]) + (v1[6] + v1[14])), v4[1] the same from 1; out v4[0] + v4[1]
    float A[RT][CT], B[RT][CT], C[RT][CT], D[RT][CT], E[RT][CT];
    pair(0, A); pair(8, D); add2(A, D);
    pair(4, B); pair(12, D); add2(B, D); add2(A, B);
    pair(2, B); pair(10, D); add2(B, D);
    pair(6, C); pair(14, D); add2(C, D); add2(B, C);
    add2(A, B);
    pair(1, B); pair(9, D); add2(B, D);
    pair(5, C); pair(13, D); add2(C, D); add2(B, C);
    pair(3, C); pair(11, D); add2(C, D);
    pair(7, E); pair(15, D); add2(E, D); add2(C, E);
    add2(B, C);
#pragma unroll
    for (int j = 0; j < RT; ++j) {
        const int r = r0 + ty * RT + j;
#pragma unroll
        for (int i = 0; i < CT; ++i) {
            const int t = t0 + tx * CT + i;
            if (r < n_out && t < T) Y[((size_t) h * T + t) * n_out + r] = __fadd_rn(A[j][i], B[j][i]);
        }
    }
}

/// dst column i = src column cols[i] (columns of `bytes`, 16-byte aligned).
__global__ void gather_cols_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, const int32_t* __restrict__ cols,
                                   int words) {
    const int i = blockIdx.y;
    const uint4* s = src + (size_t) cols[i] * words;
    uint4* d = dst + (size_t) i * words;
    for (int w = blockIdx.x * blockDim.x + threadIdx.x; w < words; w += gridDim.x * blockDim.x) d[w] = s[w];
}

/// dst row rows[i] = src row i (rows of n floats).
__global__ void scatter_rows_kernel(const float* __restrict__ src, float* __restrict__ dst, const int32_t* __restrict__ rows,
                                    int n) {
    const int i = blockIdx.y;
    const float* s = src + (size_t) i * n;
    float* d = dst + (size_t) rows[i] * n;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n; c += gridDim.x * blockDim.x) d[c] = s[c];
}

/// MMQ's gate+up product rows [gate (nf) | up (nf)] -> the clamped SwiGLU (nf) per row.
__global__ void swiglu_clamp_split_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int nf,
                                          float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nf) return;
    const int64_t r = i / nf;
    const int c = (int) (i % nf);
    float g = gu[r * 2 * nf + c], u = gu[r * 2 * nf + nf + c];
    if (limit < 0.0f) {   // moex (ERNIE-5.5): SiTU-GLU
        h[i] = situ_glu_d(g, u);
        return;
    }
    if (limit > 0.0f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    h[i] = g / (1.0f + expf(-g)) * u;
}

/// swiglu_clamp_split_kernel in place: row r's SwiGLU written over its gate half (rows `ld` floats apart, 2 nf or more),
/// where the prompt path's down projection quantizes it from - no separate rows x nf buffer.  Each element reads its own
/// gate and up value and writes its own gate slot.
__global__ void swiglu_clamp_inplace_kernel(float* gu, int64_t rows, int nf, int64_t ld, float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nf) return;
    const int64_t r = i / nf;
    const int c = (int) (i % nf);
    float g = gu[r * ld + c], u = gu[r * ld + nf + c];
    if (limit < 0.0f) {   // moex (ERNIE-5.5): SiTU-GLU
        gu[r * ld + c] = situ_glu_d(g, u);
        return;
    }
    if (limit > 0.0f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    gu[r * ld + c] = g / (1.0f + expf(-g)) * u;
}

/// out[t] = sum_j w[t][j] * Dm[inv[t*k + j]] + shared[t]: the prompt path's expert rows, sorted by expert, back to
/// their tokens (same order of addition as moe_combine_w).
__global__ void combine_rows_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ inv,
                                    const float* __restrict__ w, const float* __restrict__ shared, float* __restrict__ out,
                                    int n, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    const int32_t* iv = inv + (size_t) t * k;
    const float* wt = w + (size_t) t * k;
    float acc = Dm[(size_t) iv[0] * n + i] * wt[0];
    for (int j = 1; j < k; ++j) acc = acc + Dm[(size_t) iv[j] * n + i] * wt[j];
    out[(size_t) t * n + i] = shared ? acc + shared[(size_t) t * n + i] : acc;
}

__global__ void permute_abc_kernel(const float* __restrict__ src, float* __restrict__ dst, int A, int B, int C) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n = (size_t) A * B * C;
    if (i >= n) return;
    const int c = (int) (i % C);
    const int b = (int) ((i / C) % B);
    const int a = (int) (i / ((size_t) B * C));
    dst[((size_t) b * A + a) * C + c] = src[i];
}

}  // namespace

// ------------------------------------------------------------------ the API

void rms_norm(const float* x, const float* w, float* y, int rows, int cols, float eps, void* stream) {
    static const bool pre = [] {   // moex (§34)
        const char* e = std::getenv("STRATA_DSV41_PRELOAD");
        return e && e[0] == '1';
    }();
    if (pre && cols > 2048 && cols <= 1024 * 20) {   // the 1024-thread form's rows: 5,120 (5 a thread) and 20,480 (20)
        if (cols <= 1024 * 5) rms_norm_pre_kernel<1024, 5><<<rows, 1024, 0, st(stream)>>>(x, w, y, cols, eps);
        else rms_norm_pre_kernel<1024, 20><<<rows, 1024, 0, st(stream)>>>(x, w, y, cols, eps);
        check_launch("rms_norm (preload)");
        return;
    }
    if (cols <= 256) rms_norm_kernel<128><<<rows, 128, 0, st(stream)>>>(x, w, y, cols, eps);
    else if (cols <= 2048) rms_norm_kernel<256><<<rows, 256, 0, st(stream)>>>(x, w, y, cols, eps);
    else rms_norm_kernel<1024><<<rows, 1024, 0, st(stream)>>>(x, w, y, cols, eps);
    check_launch("rms_norm");
}

void layer_norm(const float* x, const float* w, const float* b, float* y, int cols, float eps, void* stream) {
    layer_norm_kernel<128><<<1, 128, 0, st(stream)>>>(x, w, b, y, cols, eps);
    check_launch("layer_norm");
}

void gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, void* stream) {
    gemv_f32_kernel<<<blocks(n_out, 8), 256, 0, st(stream)>>>(W, x, y, n_in, n_out);
    check_launch("gemv_f32");
}

void swiglu_clamp(const float* g, const float* u, float* out, int n, float limit, void* stream) {
    swiglu_clamp_kernel<<<blocks(n, 256), 256, 0, st(stream)>>>(g, u, out, n, limit);
    check_launch("swiglu_clamp");
}

void scale_f32(float* x, int n, float s, void* stream) {
    scale_kernel<<<blocks(n, 256), 256, 0, st(stream)>>>(x, n, s);
    check_launch("scale_f32");
}

void add_f32(const float* a, const float* b, float* out, int n, void* stream) {
    add_kernel<<<blocks(n, 256), 256, 0, st(stream)>>>(a, b, out, n);
    check_launch("add_f32");
}

void argmax_f32(const float* x, int n, int32_t* out, void* stream) {
    argmax_kernel<<<1, 1024, 0, st(stream)>>>(x, n, out);
    check_launch("argmax_f32");
}

void mhc_coef(const float* mixes, const float* scale3, const float* base24, float* coef, float eps, int iters,
              void* stream) {
    mhc_coef_kernel<<<1, 32, 0, st(stream)>>>(mixes, scale3, base24, coef, eps, iters);
    check_launch("mhc_coef");
}

void mhc_pre(const float* R, const float* coef, float* out, int n, void* stream) {
    mhc_pre_kernel<<<blocks(n, 256), 256, 0, st(stream)>>>(R, coef, out, n);
    check_launch("mhc_pre");
}

void mhc_post(const float* x, float* R, const float* coef, int n, void* stream) {
    mhc_post_kernel<<<blocks(n, 256), 256, 0, st(stream)>>>(x, R, coef, n);
    check_launch("mhc_post");
}

void mhc_mean(const float* R, float* out, int n, void* stream) {
    mhc_mean_kernel<<<blocks(n, 256), 256, 0, st(stream)>>>(R, out, n);
    check_launch("mhc_mean");
}

void kda_conv_silu(float* conv_state, const float* x, const float* w, float* out, int channels, int d_conv,
                   void* stream) {
    if (d_conv < 2 || d_conv > 8) throw std::runtime_error("kda_conv_silu: d_conv out of range");
    kda_conv_silu_kernel<<<blocks(channels, 256), 256, 0, st(stream)>>>(conv_state, x, w, out, channels, d_conv);
    check_launch("kda_conv_silu");
}

void kda_gates(float* g, const float* dt_bias, const float* A, float lower_bound, float* beta, int H, int S,
               void* stream) {
    kda_gates_kernel<<<blocks(H * S, 256), 256, 0, st(stream)>>>(g, dt_bias, A, lower_bound, beta, H, S);
    check_launch("kda_gates");
}

void kda_l2norm(float* x, int rows, int cols, float eps, float scale, void* stream) {
    l2norm_kernel<128><<<rows, 128, 0, st(stream)>>>(x, cols, eps, scale);
    check_launch("kda_l2norm");
}

void kda_step(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
              float* o, int H, int S, void* stream) {
    if (S != 128) throw std::runtime_error("kda_step: S must be 128");
    kda_step_kernel<128><<<dim3(S / 8, H), 32, 0, st(stream)>>>(state, q, k, v, g, beta, o, H);
    check_launch("kda_step");
}

void kda_out_gate(const float* o, const float* g2, const float* w, float* y, int H, int S, float eps, void* stream) {
    kda_out_gate_kernel<128><<<H, 128, 0, st(stream)>>>(o, g2, w, y, S, eps);
    check_launch("kda_out_gate");
}

void dsa_store_latent(const float* kv, uint16_t* cache, int64_t pos, int L, void* stream) {
    store_latent_kernel<<<blocks(L, 256), 256, 0, st(stream)>>>(kv, reinterpret_cast<__half*>(cache) + pos * L, L);
    check_launch("dsa_store_latent");
}

void dsa_kpool(const float* ik, const float* ig, const float* ape, float* pooled, int D, int kpool, void* stream) {
    if (kpool > 8) throw std::runtime_error("dsa_kpool: kpool > 8");
    kpool_kernel<<<blocks(D, 128), 128, 0, st(stream)>>>(ik, ig, ape, pooled, D, kpool);
    check_launch("dsa_kpool");
}

void dsa_scores(const float* iq, const float* pooled, const float* w, float* score, int n_pool, int IH, int D,
                void* stream) {
    if (IH != 32 || D != 128) throw std::runtime_error("dsa_scores: needs 32 indexer heads of 128");
    if (n_pool <= 0) return;
    const int nb = std::min(blocks(n_pool, 4), 1024);
    scores_kernel<128><<<nb, 128, 0, st(stream)>>>(iq, pooled, w, score, n_pool);
    check_launch("dsa_scores");
}

std::size_t dsa_select_scratch_bytes(int) { return 0; }

int dsa_select(const float* score, int n_pool, int n_top, int kpool, int64_t pos, bool tail, int32_t* cells,
               void*, void* stream) {
    const int n_sel = std::min(n_pool, n_top);
    if (n_sel > 0) {
        if (n_pool <= n_top) all_pools_kernel<<<blocks(n_sel * kpool, 256), 256, 0, st(stream)>>>(n_sel * kpool, cells);
        else topk_pools_kernel<<<1, 1024, 0, st(stream)>>>(score, n_pool, n_top, kpool, cells);
        check_launch("dsa_select");
    }
    const int n_tail = tail ? (int) ((pos + 1) % kpool) : 0;
    if (n_tail > 0) {
        tail_kernel<<<1, 32, 0, st(stream)>>>(cells, n_sel * kpool, n_tail, pos);
        check_launch("dsa_select tail");
    }
    return n_sel * kpool + n_tail;
}

std::size_t dsa_attend_scratch_bytes(int H, int L, int max_cells) {
    const int n_chunks = (max_cells + kCh - 1) / kCh;
    return (std::size_t) H * n_chunks * (L + 2) * sizeof(float);
}

void dsa_attend(const float* q, const uint16_t* kv, const int32_t* cells, int n_cells, float scale, float* ctx,
                float* partial, int H, int L, void* stream) {
    if (L != 512 || H % kHg != 0) throw std::runtime_error("dsa_attend: needs L = 512 and H a multiple of 16");
    if (n_cells <= 0) throw std::runtime_error("dsa_attend: no cells");
    const int n_chunks = (n_cells + kCh - 1) / kCh;
    attend_partial_kernel<512><<<dim3(n_chunks, H / kHg), 256, 0, st(stream)>>>(
        q, reinterpret_cast<const __half*>(kv), cells, n_cells, scale, partial, n_chunks);
    check_launch("dsa_attend partial");
    attend_merge_kernel<512><<<H, 256, 0, st(stream)>>>(partial, ctx, n_chunks);
    check_launch("dsa_attend merge");
}

void moe_route(const float* logits, const float* bias, int E, int k, bool norm, float scale, int32_t* ids,
               float* weights, void* stream) {
    if (E > 1024 || k > 64) throw std::runtime_error("moe_route: E <= 1024, k <= 64");
    const int bs = ((E + 31) / 32) * 32;
    route_kernel<<<1, bs, 0, st(stream)>>>(logits, bias, E, k, norm, scale, ids, weights);
    check_launch("moe_route");
}

void moe_combine(const float* parts, const float* weights, const float* shared, float* out, int n, int k,
                 void* stream) {
    combine_kernel<<<blocks(n, 256), 256, 0, st(stream)>>>(parts, weights, shared, out, n, k);
    check_launch("moe_combine");
}

// ------------------------------------------------------------------ the window API

void mhc_coef_w(const float* mixes, const float* scale3, const float* base24, float* coef, float eps, int iters, int T,
                void* stream) {
    mhc_coef_w_kernel<<<blocks(T, 2), 32, 0, st(stream)>>>(mixes, scale3, base24, coef, eps, iters, T);
    check_launch("mhc_coef_w");
}

void mhc_mix_in_w1(const float* part, int S, const float* scale3, const float* base24, float* mixes, float* coef, float* cpre,
                   const float* R, float* cur, const float* norm_w, int n, float eps, int iters, float rms_eps, void* stream) {
    if (n > 5 * 1024) throw std::runtime_error("mhc_mix_in_w1: n <= 5120");
    mhc_mix_in_w1_kernel<<<1, 1024, 0, st(stream)>>>(part, S, scale3, base24, mixes, coef, cpre, R, cur, norm_w, n, eps, iters,
                                                     rms_eps);
    check_launch("mhc_mix_in_w1");
}

void mhc_pre_norm_w1(float* mixes, const float* scale3, const float* base24, float eps, const float* R, float* cur,
                     const float* norm_w, int n, float rms_eps, void* stream, const float* parts) {
    if (n > 5 * 1024) throw std::runtime_error("mhc_pre_norm_w1: n <= 5120");
    mhc_pre_norm_w1_kernel<<<1, 1024, 0, st(stream)>>>(mixes, parts, scale3, base24, eps, R, cur, norm_w, n, rms_eps);
    check_launch("mhc_pre_norm_w1");
}

bool gemv_f32_w1_warps(const float* W, const float* x, float* part, int n_in, int n_out, void* stream) {
    static const bool on = [] {   // moex (the hill-climb): STRATA_X_F32_WARPS=0 - the chunk kernel's blocks
        const char* e = std::getenv("STRATA_X_F32_WARPS");
        return !(e && e[0] == '0');
    }();
    if (!on || n_out <= 0 || n_out > 65535) return false;
    gemv_f32_w1_warps_kernel<<<dim3((unsigned) n_out, 8), 32, 0, st(stream)>>>(W, x, part, n_in);
    check_launch("gemv_f32_w1_warps");
    return true;
}

void mhc_pre_w(const float* R, const float* coef, float* out, int n, int T, void* stream) {
    mhc_pre_w_kernel<<<dim3(blocks(n, 256), T), 256, 0, st(stream)>>>(R, coef, out, n);
    check_launch("mhc_pre_w");
}

void mhc_post_w(const float* x, float* R, const float* coef, int n, int T, void* stream) {
    mhc_post_w_kernel<<<dim3(blocks(n, 256), T), 256, 0, st(stream)>>>(x, R, coef, n);
    check_launch("mhc_post_w");
}

void mhc_mean_w(const float* R, float* out, int n, int T, void* stream) {
    mhc_mean_w_kernel<<<dim3(blocks(n, 256), T), 256, 0, st(stream)>>>(R, out, n);
    check_launch("mhc_mean_w");
}

void kda_conv_silu_w(float* conv_state, const float* x, const float* w, float* out, int channels, int d_conv, int T,
                     void* stream) {
    if (d_conv < 2 || d_conv > 8) throw std::runtime_error("kda_conv_silu_w: d_conv out of range");
    kda_conv_silu_w_kernel<<<blocks(channels, 256), 256, 0, st(stream)>>>(conv_state, x, w, out, channels, d_conv, T);
    check_launch("kda_conv_silu_w");
}

void kda_conv_rollback(const float* prev, const float* x, float* state, int channels, int d_conv, int n_keep,
                       void* stream) {
    kda_conv_rollback_kernel<<<blocks(channels, 256), 256, 0, st(stream)>>>(prev, x, state, channels, d_conv, n_keep);
    check_launch("kda_conv_rollback");
}

void kda_gates_w(float* g, const float* dt_bias, const float* A, float lower_bound, float* beta, int H, int S, int T,
                 void* stream) {
    kda_gates_w_kernel<<<blocks(T * H * S, 256), 256, 0, st(stream)>>>(g, dt_bias, A, lower_bound, beta, H, S, T);
    check_launch("kda_gates_w");
}

void kda_scan(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
              float* o, int H, int S, int T, float* snaps, void* stream) {
    if (S != 128) throw std::runtime_error("kda_scan: S must be 128");
    kda_scan_kernel<128><<<dim3(S / 8, H), 32, 0, st(stream)>>>(state, q, k, v, g, beta, o, H, T, snaps);
    check_launch("kda_scan");
}

// `pos_dev` (every dsa_*_w below): the window's first position in DEVICE memory, for captured graphs - the launch is then
// STATIC (grids that depend on the position become their upper bounds, the blocks past the real work return at once) and
// `pos0` is ignored.  nullptr: `pos0` is the argument and the launch is sized exactly, as before.
void dsa_store_latent_w(const float* kv, uint16_t* cache, int64_t pos0, int L, int T, void* stream, const int64_t* pos_dev) {
    store_latent_w_kernel<<<dim3(blocks(L, 256), T), 256, 0, st(stream)>>>(kv, reinterpret_cast<__half*>(cache), pos0, pos_dev, L);
    check_launch("dsa_store_latent_w");
}

void dsa_ring_store_w(const float* src, float* ring, int ring_rows, int64_t pos0, int D, int T, void* stream,
                      const int64_t* pos_dev) {
    ring_store_w_kernel<<<dim3(blocks(D, 128), T), 128, 0, st(stream)>>>(src, ring, ring_rows, pos0, pos_dev, D);
    check_launch("dsa_ring_store_w");
}

void dsa_kpool_w(const float* ring_ik, const float* ring_ig, int ring_rows, const float* ape, float* pooled,
                 int64_t pos0, int T, int D, int kpool, void* stream, const int64_t* pos_dev) {
    if (kpool > 8 || D > 1024) throw std::runtime_error("dsa_kpool_w: kpool <= 8, D <= 1024");
    if (pos_dev) {   // at most ceil((T + kpool - 1) / kpool) pools complete in a window of T positions
        const int n = (T + 2 * kpool - 2) / kpool;
        kpool_w_kernel<<<n, ((D + 31) / 32) * 32, 0, st(stream)>>>(ring_ik, ring_ig, ring_rows, ape, pooled, 0, pos_dev, T, D, kpool);
        check_launch("dsa_kpool_w");
        return;
    }
    // the first position in the window that completes a pool, and how many do
    int64_t p_first = pos0 + ((kpool - (pos0 + 1) % kpool) % kpool);
    if (p_first >= pos0 + T) return;
    const int n = (int) ((pos0 + T - 1 - p_first) / kpool + 1);
    kpool_w_kernel<<<n, ((D + 31) / 32) * 32, 0, st(stream)>>>(ring_ik, ring_ig, ring_rows, ape, pooled, p_first, nullptr, 0, D, kpool);
    check_launch("dsa_kpool_w");
}

void dsa_scores_w(const float* iq, const float* pooled, const float* w, float* score, int64_t pos0, int T, int kpool,
                  int IH, int D, int max_pool, void* stream, const int64_t* pos_dev) {
    if (IH != 32 || D != 128) throw std::runtime_error("dsa_scores_w: needs 32 indexer heads of 128");
    // a prompt sub-batch: the tiled kernel (bit-identical scores; STRATA_X_DSA_TILED=0 = the warp kernel); a decode
    // window (T <= 8, or positions on the device for a captured graph) keeps the warp kernel
    static const bool tiled = !(std::getenv("STRATA_X_DSA_TILED") && std::getenv("STRATA_X_DSA_TILED")[0] == '0');
    if (tiled && !pos_dev && T > 8) {
        dsa_scores_tiled(iq, pooled, w, score, pos0, T, kpool, max_pool, stream);
        return;
    }
    dsa_scores_warp(iq, pooled, w, score, pos0, T, kpool, max_pool, stream, pos_dev);
}

void dsa_scores_tiled(const float* iq, const float* pooled, const float* w, float* score, int64_t pos0, int T, int kpool,
                      int max_pool, void* stream) {
    const int max_vis = (int) ((pos0 + T) / kpool);   // the last query's pools
    if (T <= 0 || max_vis <= 0) return;
    scores_tiled_kernel<<<dim3(blocks(max_vis, kTsP), blocks(T, kTsT)), 256, 0, st(stream)>>>(iq, pooled, w, score, pos0, kpool,
                                                                                               T, max_vis, max_pool);
    check_launch("dsa_scores_tiled");
}

void dsa_scores_warp(const float* iq, const float* pooled, const float* w, float* score, int64_t pos0, int T, int kpool,
                     int max_pool, void* stream, const int64_t* pos_dev) {
    int nb;
    if (pos_dev) {
        nb = std::min(blocks(max_pool, 4), 512);   // static: the blocks past the pools return at once
    } else {
        const int n_pool_max = (int) ((pos0 + T) / kpool);
        if (n_pool_max <= 0) return;
        nb = std::min(blocks(n_pool_max, 4), 1024);
    }
    scores_w_kernel<128><<<dim3(nb, T), 128, 0, st(stream)>>>(iq, pooled, w, score, pos0, pos_dev, kpool, max_pool);
    check_launch("dsa_scores_w");
}

int dsa_cells(int64_t pos, int n_top, int kpool, bool tail) {
    const int n_pool = (int) ((pos + 1) / kpool);
    return std::min(n_pool, n_top) * kpool + (tail ? (int) ((pos + 1) % kpool) : 0);
}

void dsa_select_w(const float* score, int max_pool, int n_top, int kpool, int64_t pos0, int T, bool tail,
                  int32_t* cells, int max_cells, void* stream, const int64_t* pos_dev) {
    select_w_kernel<<<T, 1024, 0, st(stream)>>>(score, max_pool, n_top, kpool, pos0, pos_dev, tail, cells, max_cells);
    check_launch("dsa_select_w");
}

bool dsa_attend_mma_supported() {
    static const bool off = [] { const char* e = std::getenv("STRATA_X_ATTN_MMA"); return e && e[0] == '0'; }();
    if (off) return false;
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return major >= 8;
}

bool dsa_attend_mma(const float* q, const uint16_t* kv, const int32_t* cells, int max_cells, int64_t pos0, int T, int n_top,
                    int kpool, bool tail, float scale, float* ctx, int H, int L, void* stream) {
    if (!dsa_attend_mma_supported() || L != 512 || H % 16 != 0 || T < 1 || T > 65535) return false;
    static const int stages = [] { const char* e = std::getenv("STRATA_X_ATTN_STAGES"); return e && e[0] == '1' ? 1 : 2; }();
    static bool attr[64][2] = {};   // the shared-memory opt-in is per device and per kernel
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    const int si = stages - 1;
    const int bytes = stages == 2 ? (int) sizeof(MaSmem<2>) : (int) sizeof(MaSmem<1>);
    if (!attr[dev][si]) {
        const cudaError_t e = stages == 2 ? cudaFuncSetAttribute(attend_mma_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)
                                          : cudaFuncSetAttribute(attend_mma_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (e != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev][si] = true;
    }
    const float scale_log2 = scale * 1.4426950408889634f;
    const dim3 grid((unsigned) (H / 16), (unsigned) T);
    if (stages == 2)
        attend_mma_kernel<2><<<grid, kMaThreads, bytes, st(stream)>>>(q, reinterpret_cast<const __half*>(kv), cells, max_cells,
                                                                       pos0, n_top, kpool, tail, scale_log2, ctx, H);
    else
        attend_mma_kernel<1><<<grid, kMaThreads, bytes, st(stream)>>>(q, reinterpret_cast<const __half*>(kv), cells, max_cells,
                                                                       pos0, n_top, kpool, tail, scale_log2, ctx, H);
    check_launch("dsa_attend_mma");
    return true;
}

std::size_t dsa_attend_w_scratch_bytes(int H, int L, int max_cells, int T) {
    const int max_chunks = (max_cells + kCh - 1) / kCh;
    return (std::size_t) T * H * max_chunks * (L + 2) * sizeof(float);
}

void dsa_attend_w(const float* q, const uint16_t* kv, const int32_t* cells, int max_cells, int64_t pos0, int T, int n_top,
                  int kpool, bool tail, float scale, float* ctx, float* partial, int H, int L, void* stream,
                  const int64_t* pos_dev) {
    if (L != 512 || H % kHg != 0) throw std::runtime_error("dsa_attend_w: needs L = 512 and H a multiple of 16");
    const int max_chunks = (max_cells + kCh - 1) / kCh;
    int n_chunks = max_chunks;   // static (graphs): the chunks past a query's cells return at once
    if (!pos_dev) {
        int cmax = 0;
        for (int t = 0; t < T; ++t) cmax = std::max(cmax, dsa_cells(pos0 + t, n_top, kpool, tail));
        if (cmax <= 0) throw std::runtime_error("dsa_attend_w: a query with no cells");
        n_chunks = (cmax + kCh - 1) / kCh;
    }
    attend_partial_w_kernel<512><<<dim3(n_chunks, H / kHg, T), 256, 0, st(stream)>>>(
        q, reinterpret_cast<const __half*>(kv), cells, max_cells, pos0, pos_dev, n_top, kpool, tail, scale, partial, max_chunks, H);
    check_launch("dsa_attend_w partial");
    attend_merge_w_kernel<512><<<dim3(H, T), 256, 0, st(stream)>>>(partial, ctx, max_chunks, pos0, pos_dev, n_top, kpool, tail, H);
    check_launch("dsa_attend_w merge");
}

// ---------------------------------------------------------------------------------------------------------------
// THE DOORBELL (Strata's decode handoff, STRATA_QWEN_DESIGN.md 3.2): a captured kernel copies the router's ids and the
// layer's activations into MAPPED PINNED host memory, fences, and bumps a sequence word the host spins on with no driver
// call; the CPU's rows come back the same way (mapped rows + a mask, a flag word the GPU's wait kernel spins on) and the
// combine reads them straight from host memory.  Counters live in device memory and are INCREMENTED by the kernels
// (a value passed as a kernel argument would be a literal baked into the graph: "a cloned literal").
__global__ void doorbell_publish_kernel(const int32_t* __restrict__ ids, int n_ids, const float* __restrict__ x, int n_x,
                                        int32_t* __restrict__ m_ids, float* __restrict__ m_x,
                                        volatile uint32_t* __restrict__ m_seq, uint32_t* __restrict__ d_cnt) {
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) m_ids[i] = ids[i];
    for (int i = threadIdx.x; i < n_x; i += blockDim.x) m_x[i] = x[i];
    __threadfence_system();   // every thread's payload is visible to the host before the word moves
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t v = *d_cnt + 1u;
        *d_cnt = v;
        *m_seq = v;
        __threadfence_system();
    }
}

__global__ void doorbell_wait_kernel(const volatile uint32_t* __restrict__ m_flag, const volatile uint32_t* __restrict__ m_abort,
                                     uint32_t* __restrict__ d_cnt) {
    if (threadIdx.x != 0) return;
    const uint32_t want = *d_cnt + 1u;
    // the host's counter reaches `want` once it has put this layer's rows in place; an abort word releases the GPU when
    // the host gives up (Strata: "no spin kernel outlives the request")
    while ((int32_t) (*m_flag - want) < 0 && *m_abort == 0u) __nanosleep(100);
    __threadfence_system();
    *d_cnt = want;
}

__global__ void hit_slots_kernel(const int32_t* __restrict__ ids, const int32_t* __restrict__ res_layer, int32_t* __restrict__ slot,
                                 int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) slot[i] = res_layer[ids[i]];
}

/// out[t] = sum_j w[t][j] * row(t, j) + shared[t], row(t, j) from the CPU's mapped rows when mask[t * k + j], else the
/// GPU's parts: moe_combine_w's arithmetic exactly (j ascending, the shared expert last).
__global__ void combine_pairs_kernel(const float* __restrict__ parts, const float* __restrict__ cpu_rows,
                                     const uint8_t* __restrict__ mask, const float* __restrict__ w,
                                     const float* __restrict__ shared, float* __restrict__ out, int n, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    const float* wt = w + (size_t) t * k;
    float s = 0.0f;
    for (int j = 0; j < k; ++j) {
        const size_t row = ((size_t) t * k + j) * n + i;
        const float v = mask[t * k + j] ? cpu_rows[row] : parts[row];
        s = j == 0 ? v * wt[0] : s + v * wt[j];
    }
    out[(size_t) t * n + i] = shared ? s + shared[(size_t) t * n + i] : s;
}

void doorbell_publish(const int32_t* ids, int n_ids, const float* x, int n_x, int32_t* m_ids, float* m_x,
                      volatile uint32_t* m_seq, uint32_t* d_cnt, void* stream) {
    doorbell_publish_kernel<<<1, 1024, 0, st(stream)>>>(ids, n_ids, x, n_x, m_ids, m_x, m_seq, d_cnt);
    check_launch("doorbell_publish");
}

void doorbell_wait(const volatile uint32_t* m_flag, const volatile uint32_t* m_abort, uint32_t* d_cnt, void* stream) {
    doorbell_wait_kernel<<<1, 32, 0, st(stream)>>>(m_flag, m_abort, d_cnt);
    check_launch("doorbell_wait");
}

void hit_slots(const int32_t* ids, const int32_t* res_layer, int32_t* slot, int n, void* stream) {
    hit_slots_kernel<<<blocks(n, 64), 64, 0, st(stream)>>>(ids, res_layer, slot, n);
    check_launch("hit_slots");
}

/// moex (DSV41_PLAN §34), opt-in STRATA_DSV41_PRELOAD=1: combine_pairs_kernel with its loads first - the k mask bytes (mapped),
/// then the k values (the CPU's rows mapped, the GPU's in VRAM), then the same sum: each load of the loop waited for the one
/// before (a PCIe round trip each, ~11 us a decode layer for 6 slots).  The same expression in the same order: the same bits
template <int K>
__global__ void combine_pairs_pre_kernel(const float* __restrict__ parts, const float* __restrict__ cpu_rows,
                                         const uint8_t* __restrict__ mask, const float* __restrict__ w,
                                         const float* __restrict__ shared, float* __restrict__ out, int n, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    const float* wt = w + (size_t) t * k;
    uint8_t mk[K];
#pragma unroll
    for (int j = 0; j < K; ++j) mk[j] = j < k ? mask[t * k + j] : 0;
    float v[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        const size_t row = ((size_t) t * k + j) * n + i;
        v[j] = j < k ? (mk[j] ? cpu_rows[row] : parts[row]) : 0.0f;
    }
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < K; ++j)
        if (j < k) s = j == 0 ? v[j] * wt[0] : s + v[j] * wt[j];
    out[(size_t) t * n + i] = shared ? s + shared[(size_t) t * n + i] : s;
}

static bool bf16_pre_default();   // (g_bf16_pre, below)

void moe_combine_pairs(const float* parts, const float* cpu_rows, const uint8_t* mask, const float* weights,
                       const float* shared, float* out, int n, int k, int T, void* stream) {
    // moex (the ERNIE-5.5 hill-climb): the preload kernel also where the engine sets the one-column preload default (ERNIE-5.5,
    // Hy4-flash) and for up to 16 slots (ERNIE-5.5 routes 10): the same expression in the same order, the same bits;
    // STRATA_DSV41_PRELOAD=0 keeps the loop kernel
    static const int pre_env = [] {
        const char* e = std::getenv("STRATA_DSV41_PRELOAD");
        return e ? (e[0] == '1' ? 1 : 0) : -1;
    }();
    const bool pre = pre_env >= 0 ? pre_env == 1 : bf16_pre_default();
    if (pre && k <= 8) {
        combine_pairs_pre_kernel<8><<<dim3(blocks(n, 256), T), 256, 0, st(stream)>>>(parts, cpu_rows, mask, weights, shared, out, n, k);
        check_launch("moe_combine_pairs (preload)");
        return;
    }
    if (pre && k <= 16) {
        combine_pairs_pre_kernel<16><<<dim3(blocks(n, 256), T), 256, 0, st(stream)>>>(parts, cpu_rows, mask, weights, shared, out, n, k);
        check_launch("moe_combine_pairs (preload, 16)");
        return;
    }
    combine_pairs_kernel<<<dim3(blocks(n, 256), T), 256, 0, st(stream)>>>(parts, cpu_rows, mask, weights, shared, out, n, k);
    check_launch("moe_combine_pairs");
}

void moe_route_w(const float* logits, const float* bias, int E, int k, bool norm, float scale, int32_t* ids,
                 float* weights, int T, void* stream) {
    if (E > 1024 || k > 64) throw std::runtime_error("moe_route_w: E <= 1024, k <= 64");
    const int bs = ((E + 31) / 32) * 32;
    route_kernel<<<T, bs, 0, st(stream)>>>(logits, bias, E, k, norm, scale, ids, weights);
    check_launch("moe_route_w");
}

void moe_combine_w(const float* parts, const float* weights, const float* shared, float* out, int n, int k, int T,
                   void* stream) {
    combine_w_kernel<<<dim3(blocks(n, 256), T), 256, 0, st(stream)>>>(parts, weights, shared, out, n, k);
    check_launch("moe_combine_w");
}

namespace {
/// gemv_f32_w for a few long rows (DeepSeek-V4.1's mHC mixers: 24 rows of 20,480): each row split S ways across blocks, the
/// partial sums written out and reduced in a fixed order by a second launch.  gemv_f32_w's block per row is 24 blocks for 68
/// SMs there: it reads the 2 MB in ~26 us, twice a layer (nsys, 2026-10-18)
__global__ void __launch_bounds__(256) gemv_f32_part_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                            float* __restrict__ part, int n_in, int n_out, int T, int chunk) {
    __shared__ float red[8][8];
    const int row = blockIdx.x, sp = blockIdx.y, S = gridDim.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int i1 = min(n_in, (sp + 1) * chunk);
    const float* wr = W + (size_t) row * n_in;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = sp * chunk + threadIdx.x; i < i1; i += 256) {
        const float wv = wr[i];
#pragma unroll
        for (int t = 0; t < 8; ++t)
            if (t < T) acc[t] += wv * x[(size_t) t * n_in + i];
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0) red[t][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < 8 && (int) threadIdx.x < T) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) v += red[threadIdx.x][w];
        part[((size_t) threadIdx.x * n_out + row) * S + sp] = v;
    }
}

/// moex (DSV41_PLAN §36), STRATA_DSV41_PRELOAD=1: gemv_f32_part_kernel for one column with each thread's (at most PT) weights
/// and inputs loaded first, the same multiply-adds in order, the one warp sum a column has, the same block sum: the same bits
template <int PT>
__global__ void __launch_bounds__(256) gemv_f32_part1_pre_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                                 float* __restrict__ part, int n_in, int n_out, int chunk) {
    __shared__ float red[8];
    const int row = blockIdx.x, sp = blockIdx.y, S = gridDim.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int i0 = sp * chunk, i1 = min(n_in, (sp + 1) * chunk);
    const float* wr = W + (size_t) row * n_in;
    float wv[PT], xv[PT];
#pragma unroll
    for (int j = 0; j < PT; ++j) {
        const int i = i0 + threadIdx.x + 256 * j;
        wv[j] = i < i1 ? wr[i] : 0.0f;
        xv[j] = i < i1 ? x[i] : 0.0f;
    }
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < PT; ++j)
        if (i0 + (int) threadIdx.x + 256 * j < i1) acc += wv[j] * xv[j];
    const float v = warp_sum(acc);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) s += red[w];
        part[(size_t) row * S + sp] = s;
    }
}

__global__ void gemv_f32_reduce_kernel(const float* __restrict__ part, float* __restrict__ y, int n, int S) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // t * n_out + row
    if (i >= n) return;
    const float* p = part + (size_t) i * S;
    float v = p[0];
    for (int k = 1; k < S; ++k) v += p[k];
    y[i] = v;
}
}  // namespace

int gemv_f32_split_parts(int n_in) { return std::max(1, std::min(16, n_in / 1024)); }

namespace {
void gemv_f32_part_launch(const float* W, const float* x, float* part, int n_in, int n_out, int T, int S, int chunk, void* stream) {
    static const bool pre = [] {   // moex (§36)
        const char* e = std::getenv("STRATA_DSV41_PRELOAD");
        return e && e[0] == '1';
    }();
    if (pre && T == 1 && chunk <= 256 * 5) {
        gemv_f32_part1_pre_kernel<5><<<dim3(n_out, S), 256, 0, st(stream)>>>(W, x, part, n_in, n_out, chunk);
        return;
    }
    gemv_f32_part_kernel<<<dim3(n_out, S), 256, 0, st(stream)>>>(W, x, part, n_in, n_out, T, chunk);
}
}  // namespace

void gemv_f32_split_part(const float* W, const float* x, float* part, int n_in, int n_out, int T, void* stream) {
    if (T < 1 || T > 8) throw std::runtime_error("gemv_f32_split_part: 1 <= T <= 8");
    const int S = gemv_f32_split_parts(n_in);
    const int chunk = (n_in + S - 1) / S;
    gemv_f32_part_launch(W, x, part, n_in, n_out, T, S, chunk, stream);
    check_launch("gemv_f32_split_part");
}

void gemv_f32_split(const float* W, const float* x, float* y, float* part, int n_in, int n_out, int T, void* stream) {
    if (T < 1 || T > 8) throw std::runtime_error("gemv_f32_split: 1 <= T <= 8");
    const int S = gemv_f32_split_parts(n_in);
    const int chunk = (n_in + S - 1) / S;
    gemv_f32_part_launch(W, x, part, n_in, n_out, T, S, chunk, stream);
    gemv_f32_reduce_kernel<<<(T * n_out + 255) / 256, 256, 0, st(stream)>>>(part, y, T * n_out, S);
    check_launch("gemv_f32_split");
}

namespace {
int g_f32_pre = -1;   // moex (2026-11-19): an engine's default for the F32 one-column preload (-1: STRATA_DSV41_PRELOAD's alone)
}

void gemv_f32_w_preload_default(bool on) { g_f32_pre = on ? 1 : 0; }

void gemv_f32_w(const float* W, const float* x, float* y, int n_in, int n_out, int T, void* stream) {
    if (T < 1) throw std::runtime_error("gemv_f32_w: T >= 1");
    static const int pre_env = [] {   // moex (§36): set, it wins over an engine's default (gemv_bf16_w's rule)
        const char* e = std::getenv("STRATA_DSV41_PRELOAD");
        return e && *e ? (e[0] == '1' ? 1 : 0) : -1;
    }();
    const bool pre = pre_env >= 0 ? pre_env == 1 : g_f32_pre == 1;
    if (pre && T == 1 && n_in <= 256 * 20) {
        if (n_in <= 256 * 4) gemv_f32_w1_pre_kernel<4><<<n_out, 256, 0, st(stream)>>>(W, x, y, n_in);
        else gemv_f32_w1_pre_kernel<20><<<n_out, 256, 0, st(stream)>>>(W, x, y, n_in);
        check_launch("gemv_f32_w (preload)");
        return;
    }
    static const bool chunk = [] {   // moex (ERNIE55_PLAN P6): the one-column loads ahead (the same bits); =0 the loop as before
        const char* e = std::getenv("STRATA_X_F32_CHUNK");
        return !(e && e[0] == '0');
    }();
    if (chunk && T == 1) {
        gemv_f32_w1_chunk_kernel<<<n_out, 256, 0, st(stream)>>>(W, x, y, n_in);
        check_launch("gemv_f32_w (one column)");
        return;
    }
    for (int c0 = 0; c0 < T; c0 += 8)
        gemv_f32_w_kernel<<<n_out, 256, 0, st(stream)>>>(W, x + (size_t) c0 * n_in, y + (size_t) c0 * n_out, n_in, n_out,
                                                         std::min(8, T - c0));
    check_launch("gemv_f32_w");
}

namespace {
int g_bf16_pre = -1;   // moex: an engine's default for the one-column preload (-1: STRATA_DSV41_PRELOAD's alone)
}

void gemv_bf16_w_preload_default(bool on) { g_bf16_pre = on ? 1 : 0; }
static bool bf16_pre_default() { return g_bf16_pre == 1; }

/// moex (the hill-climb): y_a = A x and y_b = B x in one launch (gemv_pair_kernel) where both rows' kernels are the ones a
/// one-column call would take (FP8: the warp kernel's rule; BF16: the preload kernel, an engine's default or
/// STRATA_DSV41_PRELOAD); otherwise the two calls as before.  The same bits either way.  STRATA_X_PAIR=0: the two calls.
static int fp8_rows_env();   // (defined below, with fp8_rows)
bool gemv_pair_w(const void* Wa, const uint8_t* Sa, bool a_fp8, float* ya, int na, const void* Wb, const uint8_t* Sb, bool b_fp8,
                 float* yb, int nb, const float* x, int n_in, int T, void* stream) {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_X_PAIR");
        return !(e && e[0] == '0');
    }();
    static const int pre_env = [] {
        const char* e = std::getenv("STRATA_DSV41_PRELOAD");
        return e && *e ? (e[0] == '1' ? 1 : 0) : -1;
    }();
    const bool bf16_pre = pre_env >= 0 ? pre_env == 1 : g_bf16_pre == 1;
    const bool fp8_warp = fp8_warp_on() && fp8_rows_env() < 0 && (reinterpret_cast<uintptr_t>(x) & 15) == 0;
    auto fits = [&](bool fp8, int n) {   // the row kernel a one-column call would run is the one the pair runs
        // (FP8: the warp kernel's rows - one a warp here, R a warp there: each row's order is the same, fp8_warp_bench)
        return fp8 ? fp8_warp && n > 0 && n_in % 32 == 0 : bf16_pre && n > 0 && n_in <= 256 * 20;
    };
    if (!on || T != 1 || !fits(a_fp8, na) || !fits(b_fp8, nb)) return false;
    if (a_fp8 && !b_fp8) {   // (BF16 + FP8 only, as the kernel's MODE 2 holds it: swap the pair)
        std::swap(Wa, Wb), std::swap(Sa, Sb), std::swap(ya, yb), std::swap(na, nb), std::swap(a_fp8, b_fp8);
    }
    const int mode = a_fp8 ? 0 : b_fp8 ? 2 : 1;
    static const int modes = [] {   // STRATA_X_PAIR_MODES: which kinds of pair run as one launch (bits: 1 FP8 + FP8, 2 BF16 + BF16,
        const char* e = std::getenv("STRATA_X_PAIR_MODES");   // 4 BF16 + FP8); unset: all three (each measured faster:
        return e && *e ? std::atoi(e) : 7;                         // fp8_warp_bench, 28.2 -> 17.0 / 18.1 -> 15.6 / 40.4 -> 37.2 us)
    }();
    if (!(modes & (1 << mode))) return false;
    const int nb_a = a_fp8 ? (na + 7) / 8 : na, nb_b = b_fp8 ? (nb + 7) / 8 : nb;
    const unsigned g = (unsigned) (nb_a + nb_b);
    auto go = [&](auto pt) {
        constexpr int PT = decltype(pt)::value;
        if (mode == 0) gemv_pair_kernel<PT, 0><<<g, 256, 0, st(stream)>>>(Wa, Sa, ya, na, Wb, Sb, yb, nb, nb_a, x, n_in);
        else if (mode == 1) gemv_pair_kernel<PT, 1><<<g, 256, 0, st(stream)>>>(Wa, Sa, ya, na, Wb, Sb, yb, nb, nb_a, x, n_in);
        else gemv_pair_kernel<PT, 2><<<g, 256, 0, st(stream)>>>(Wa, Sa, ya, na, Wb, Sb, yb, nb, nb_a, x, n_in);
    };
    if (n_in <= 256 * 4) go(std::integral_constant<int, 4>{});
    else go(std::integral_constant<int, 20>{});
    check_launch("gemv_pair_w");
    return true;
}

void gemv_bf16_w(const uint16_t* W, const float* x, float* y, int n_in, int n_out, int T, void* stream) {
    if (T < 1) throw std::runtime_error("gemv_bf16_w: T >= 1");
    static const int pre_env = [] {   // gemv_f32_w's choice, made the same way (set, it wins over an engine's default)
        const char* e = std::getenv("STRATA_DSV41_PRELOAD");
        return e && *e ? (e[0] == '1' ? 1 : 0) : -1;
    }();
    const bool pre = pre_env >= 0 ? pre_env == 1 : g_bf16_pre == 1;
    if (pre && T == 1 && n_in <= 256 * 20) {
        if (n_in <= 256 * 4) gemv_bf16_w1_pre_kernel<4><<<n_out, 256, 0, st(stream)>>>(W, x, y, n_in);
        else gemv_bf16_w1_pre_kernel<20><<<n_out, 256, 0, st(stream)>>>(W, x, y, n_in);
        check_launch("gemv_bf16_w (preload)");
        return;
    }
    if (pre && T >= 2 && T <= 8 && n_in <= 256 * 12) {   // moex: 2 .. 8 columns, the same bits (gemv_bf16_wn_pre_kernel)
        const bool small = n_in <= 256 * 4;
        switch (T) {
#define STRATA_BF16_WN(NC)                                                                                                     \
    case NC:                                                                                                                   \
        if (small) gemv_bf16_wn_pre_kernel<4, NC><<<n_out, 256, 0, st(stream)>>>(W, x, y, n_in, n_out);                        \
        else gemv_bf16_wn_pre_kernel<12, NC><<<n_out, 256, 0, st(stream)>>>(W, x, y, n_in, n_out);                             \
        break;
            STRATA_BF16_WN(2) STRATA_BF16_WN(3) STRATA_BF16_WN(4) STRATA_BF16_WN(5) STRATA_BF16_WN(6) STRATA_BF16_WN(7) STRATA_BF16_WN(8)
#undef STRATA_BF16_WN
        }
        check_launch("gemv_bf16_w (preload, columns)");
        return;
    }
    for (int c0 = 0; c0 < T; c0 += 8)
        gemv_bf16_w_kernel<<<n_out, 256, 0, st(stream)>>>(W, x + (size_t) c0 * n_in, y + (size_t) c0 * n_out, n_in, n_out,
                                                          std::min(8, T - c0));
    check_launch("gemv_bf16_w");
}

void gemv_fp8_w(const uint8_t* W, const uint8_t* S, const float* x, float* y, int n_in, int n_out, int T, void* stream);

/// moex (§70, §75): rows a block - the same bits whichever.  STRATA_DSV41_FP8_ROWS=R names one for every matrix (1: the one-row
/// kernels); unset, 4, and 8 for a matrix of 8,192 values a row or more (wo_b: 120.5 -> 106.8 us a call at 8, where the
/// narrower projections were as fast or faster at 4 - fp8_gemv_bench, 2026-11-10)
static int fp8_rows_env() {   // -1: unset
    static const int r = [] {
        const char* e = std::getenv("STRATA_DSV41_FP8_ROWS");
        if (!e || !*e) return -1;
        const int v = std::atoi(e);
        return v == 2 || v == 4 || v == 8 ? v : 1;
    }();
    return r;
}
static int fp8_rows(int n_in) {
    const int e = fp8_rows_env();
    return e >= 0 ? e : (n_in >= 8192 ? 8 : 4);
}

/// moex (DSV41_BACKLOG #138, DSV41_PLAN §75): H groups in ONE launch (grid.y = the group), each with its own weights, input
/// and output - gemv_fp8_rows_kernel's rows, products and sums, so bitwise the H separate calls (the Q8_0 path's WOA_BATCH:
/// eight launches of 8.5 us were 68 us of a decode layer, §34).  Used by Stage::heads_mm for DeepSeek-V4.1's wo_a.
template <int R, int TT>
__global__ void __launch_bounds__(256) gemv_fp8_heads_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                                             const float* __restrict__ x, float* __restrict__ y, int n_in,
                                                             int n_out, int T) {
    const size_t h = blockIdx.y;
    gemv_fp8_rows_body<R, TT>(W + h * (size_t) n_out * n_in, S + h * ((size_t) n_out * n_in >> 5), x + h * (size_t) T * n_in,
                              y + h * (size_t) T * n_out, n_in, n_out, T);
}

void gemv_fp8_heads(const uint8_t* W, const uint8_t* S, const float* x, float* y, int n_in, int n_out, int T, int H,
                    void* stream) {
    if (T < 1 || T > 8 || n_in % 32 || (reinterpret_cast<uintptr_t>(W) & 15)) throw std::runtime_error("gemv_fp8_heads: T 1..8, n_in a multiple of 32, W 16-byte aligned");
    if (fp8_rows_env() == 1) {   // (the A/B's one-row arm: the groups one after the other, as before)
        for (int h = 0; h < H; ++h)
            gemv_fp8_w(W + (size_t) h * n_out * n_in, S + ((size_t) h * n_out * n_in >> 5), x + (size_t) h * T * n_in,
                       y + (size_t) h * T * n_out, n_in, n_out, T, stream);
        return;
    }
    // moex (ERNIE55_PLAN P6): one token - the warp kernel, a group a grid row (the row kernels' bits, gemv_fp8_warp_kernel)
    if (T == 1 && H > 1 && fp8_warp_on() && fp8_rows_env() < 0 && (reinterpret_cast<uintptr_t>(x) & 15) == 0) {
        const int R = n_out >= 8192 ? 4 : n_out >= 2048 ? 2 : 1;
        const dim3 gw((unsigned) ((n_out + 8 * R - 1) / (8 * R)), (unsigned) H);
        if (R == 4) gemv_fp8_warp_kernel<4><<<gw, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out);
        else if (R == 2) gemv_fp8_warp_kernel<2><<<gw, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out);
        else gemv_fp8_warp_kernel<1><<<gw, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out);
        check_launch("gemv_fp8_heads (warp)");
        return;
    }
    if (T >= 2 && T <= 8 && H > 1 && fp8_warp_on() && fp8_rows_env() < 0 && (reinterpret_cast<uintptr_t>(x) & 15) == 0) {
        const dim3 gw((unsigned) ((n_out + 7) / 8), (unsigned) H);
        if (T <= 2) gemv_fp8_warp_t_kernel<2><<<gw, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out, T);
        else if (T <= 4) gemv_fp8_warp_t_kernel<4><<<gw, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out, T);
        else gemv_fp8_warp_t_kernel<8><<<gw, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out, T);
        check_launch("gemv_fp8_heads (warp, window)");
        return;
    }
    const dim3 g((unsigned) ((n_out + 3) / 4), (unsigned) H);
    if (T == 1) gemv_fp8_heads_kernel<4, 1><<<g, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out, T);
    else if (T <= 4) gemv_fp8_heads_kernel<4, 4><<<g, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out, T);
    else gemv_fp8_heads_kernel<4, 8><<<g, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out, T);
    check_launch("gemv_fp8_heads");
}

void gemv_fp8_w(const uint8_t* W, const uint8_t* S, const float* x, float* y, int n_in, int n_out, int T, void* stream) {
    if (T < 1 || n_in % 32 || (reinterpret_cast<uintptr_t>(W) & 15)) throw std::runtime_error("gemv_fp8_w: T >= 1, n_in a multiple of 32, W 16-byte aligned");
    static const bool pre = [] {   // gemv_bf16_w's choice, made the same way
        const char* e = std::getenv("STRATA_DSV41_PRELOAD");
        return e && e[0] == '1';
    }();
    // moex (ERNIE55_PLAN P6): one token - a warp a row, the row kernels' bits (gemv_fp8_warp_kernel); W and x 16-byte aligned
    if (T == 1 && fp8_warp_on() && fp8_rows_env() < 0 && (reinterpret_cast<uintptr_t>(x) & 15) == 0) {
        static const int r_env = [] { const char* e = std::getenv("STRATA_X_FP8_WARP_R"); return e ? std::atoi(e) : 0; }();
        const int R = r_env == 1 || r_env == 2 || r_env == 4 ? r_env : n_out >= 8192 ? 4 : n_out >= 2048 ? 2 : 1;
        const unsigned nb = (unsigned) ((n_out + 8 * R - 1) / (8 * R));
        if (R == 4) gemv_fp8_warp_kernel<4><<<nb, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out);
        else if (R == 2) gemv_fp8_warp_kernel<2><<<nb, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out);
        else gemv_fp8_warp_kernel<1><<<nb, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out);
        check_launch("gemv_fp8_w (warp)");
        return;
    }
    // moex (ERNIE55_PLAN P6): a window of 2 .. 8 columns - a warp a row, the row kernels' bits (gemv_fp8_warp_t_kernel)
    if (T >= 2 && T <= 8 && fp8_warp_on() && fp8_rows_env() < 0 && (reinterpret_cast<uintptr_t>(x) & 15) == 0 && n_in % 4 == 0) {
        const unsigned nb = (unsigned) ((n_out + 7) / 8);
        if (T <= 2) gemv_fp8_warp_t_kernel<2><<<nb, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out, T);
        else if (T <= 4) gemv_fp8_warp_t_kernel<4><<<nb, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out, T);
        else gemv_fp8_warp_t_kernel<8><<<nb, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out, T);
        check_launch("gemv_fp8_w (warp, window)");
        return;
    }
    // (8 rows x 8 columns would be 64 sums a thread: windows of 5 to 8 rows keep 4 rows a block)
    const int rows = fp8_rows(n_in) == 8 && T > 4 ? 4 : fp8_rows(n_in);
    // moex (DSV41_BACKLOG #137, #139; DSV41_PLAN §75): R rows a block at every width up to 8 - a verify window (the prompt
    // lookup's, DSpark's) ran the one-row kernel, 0.34-0.55x of BF16's preload kernel (§70).  Each row's products and sums
    // stay in gemv_bf16_w's order, so the bits are the one-row kernels'
    if (T >= 1 && T <= 8 && rows > 1) {
        const unsigned nb = (unsigned) ((n_out + rows - 1) / rows);
        auto go = [&](auto tt) {
            constexpr int TT = decltype(tt)::value;
            if (rows == 2) gemv_fp8_rows_kernel<2, TT><<<nb, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out, T);
            else if (rows == 4) gemv_fp8_rows_kernel<4, TT><<<nb, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out, T);
            else gemv_fp8_rows_kernel<8, TT><<<nb, 256, 0, st(stream)>>>(W, S, x, y, n_in, n_out, T);
        };
        if (T == 1) go(std::integral_constant<int, 1>{});
        else if (T <= 4) go(std::integral_constant<int, 4>{});
        else go(std::integral_constant<int, 8>{});
        check_launch("gemv_fp8_w (rows)");
        return;
    }
    if (pre && T == 1 && n_in <= 256 * 20) {   // gemv_bf16_w's preload choice, the same kernel shape (its sums are the main kernel's)
        if (n_in <= 256 * 4) gemv_fp8_w1_pre_kernel<4><<<n_out, 256, 0, st(stream)>>>(W, S, x, y, n_in);
        else gemv_fp8_w1_pre_kernel<20><<<n_out, 256, 0, st(stream)>>>(W, S, x, y, n_in);
        check_launch("gemv_fp8_w (preload)");
        return;
    }
    for (int c0 = 0; c0 < T; c0 += 8)
        gemv_fp8_w_kernel<<<n_out, 256, 0, st(stream)>>>(W, S, x + (size_t) c0 * n_in, y + (size_t) c0 * n_out, n_in, n_out,
                                                         std::min(8, T - c0));
    check_launch("gemv_fp8_w");
}

void fp8_widen(const uint8_t* src, const uint8_t* S, float* dst, size_t n, void* stream) {
    if (n == 0) return;
    const unsigned blocks = (unsigned) std::min<size_t>(1024, (n + 255) / 256);
    fp8_widen_kernel<<<blocks, 256, 0, st(stream)>>>(src, S, dst, n);
    check_launch("fp8_widen");
}

void fp8_bf16(const uint8_t* src, const uint8_t* S, uint16_t* dst, size_t n, void* stream) {
    if (n == 0) return;
    const unsigned blocks = (unsigned) std::min<size_t>(1024, (n + 255) / 256);
    fp8_bf16_kernel<<<blocks, 256, 0, st(stream)>>>(src, S, dst, n);
    check_launch("fp8_bf16");
}

void split3_bf16(const float* x, uint16_t* parts, size_t n, void* stream) {
    if (n == 0) return;
    const unsigned blocks = (unsigned) std::min<size_t>(1024, (n + 255) / 256);
    split3_bf16_kernel<<<blocks, 256, 0, st(stream)>>>(x, parts, n);
    check_launch("split3_bf16");
}

void bf16_widen(const uint16_t* src, float* dst, size_t n, void* stream) {
    if (n == 0) return;
    const unsigned blocks = (unsigned) std::min<size_t>(1024, (n + 255) / 256);
    bf16_widen_kernel<<<blocks, 256, 0, st(stream)>>>(src, dst, n);
    check_launch("bf16_widen");
}

void argmax_w(const float* x, int n, int32_t* out, int T, void* stream) {
    for (int t = 0; t < T; ++t) argmax_kernel<<<1, 1024, 0, st(stream)>>>(x + (size_t) t * n, n, out + t);
    check_launch("argmax_w");
}

namespace {
/// A decode graph's first node: the step's inputs from MAPPED host memory (no copy engine: a queued swap DMA can no longer
/// hold a window's start back).  d_pos <- *m_pos; d_res[0, n_res) <- m_res; dst (rows, width) <- m_rows, each source row of
/// src_width values repeated across the destination row (the embeddings broadcast to the 4 mHC streams) or copied as is.
__global__ void window_load_kernel(const int64_t* __restrict__ m_pos, const float* __restrict__ m_rows,
                                   const int32_t* __restrict__ m_res, int64_t* __restrict__ d_pos, int32_t* __restrict__ d_res,
                                   int n_res, float* __restrict__ dst, int rows, int width, int src_width) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x, nthr = (size_t) gridDim.x * blockDim.x;
    if (tid == 0 && m_pos != nullptr) *d_pos = *(const volatile int64_t*) m_pos;
    for (size_t i = tid; i < (size_t) n_res; i += nthr) d_res[i] = m_res[i];
    const size_t tot = (size_t) rows * width;
    for (size_t i = tid; i < tot; i += nthr) {
        const size_t r = i / width, c = i % width;
        dst[i] = m_rows[r * src_width + (c % src_width)];
    }
}

__global__ void store_mapped_kernel(const float* __restrict__ x, float* __restrict__ m_out, size_t count) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x, nthr = (size_t) gridDim.x * blockDim.x;
    for (size_t i = tid; i < count; i += nthr) m_out[i] = x[i];
    __threadfence_system();
}

__global__ void store_draft_kernel(const int32_t* __restrict__ tok, const float* __restrict__ prob, int32_t* __restrict__ m_tok,
                                   float* __restrict__ m_prob) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *m_tok = *tok;
        *m_prob = *prob;
        __threadfence_system();
    }
}

/// n drafts and their probabilities into mapped memory (DFlash's block)
__global__ void store_drafts_kernel(const int32_t* __restrict__ tok, const float* __restrict__ prob, int n, int32_t* __restrict__ m_tok,
                                    float* __restrict__ m_prob) {
    const int i = threadIdx.x;
    if (i < n) {
        m_tok[i] = tok[i];
        m_prob[i] = prob[i];
    }
    __threadfence_system();
}

/// rows of n floats from src (row stride src_stride) to dst (row stride dst_stride); either side may be mapped host memory
__global__ void copy_rows_kernel(const float* __restrict__ src, int64_t src_stride, float* __restrict__ dst, int64_t dst_stride, int n) {
    const int r = blockIdx.y;
    const float* sr = src + (size_t) r * src_stride;
    float* dr = dst + (size_t) r * dst_stride;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dr[i] = sr[i];
}

__global__ void copy_f32_kernel(const float* __restrict__ src, float* __restrict__ dst, size_t count) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x, nthr = (size_t) gridDim.x * blockDim.x;
    for (size_t i = tid; i < count; i += nthr) dst[i] = src[i];
}
}  // namespace

namespace {
/// One block per job: h = sum over the range's 64-bit words w_i of w_i * (2i + 1) (mod 2^64), a trailing partial word
/// zero-padded - order-sensitive enough that a moved, missing or flipped byte changes it, and a plain sum to reduce.
__global__ void slices_hash_kernel(const SliceJob* __restrict__ jobs, uint64_t* __restrict__ out) {
    const SliceJob j = jobs[blockIdx.x];
    const uint64_t nw = j.len / 8, tail = j.len % 8;
    uint64_t h = 0;
    for (uint64_t i = threadIdx.x; i < nw; i += blockDim.x) {
        uint64_t w;
        memcpy(&w, j.p + i * 8, 8);
        h += w * (2 * i + 1);
    }
    if (tail && threadIdx.x == 0) {
        uint64_t w = 0;
        for (uint64_t b = 0; b < tail; ++b) w |= (uint64_t) j.p[nw * 8 + b] << (8 * b);
        h += w * (2 * nw + 1);
    }
    __shared__ uint64_t red[256];
    red[threadIdx.x] = h;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = red[0];
        __threadfence_system();
    }
}
}  // namespace

namespace {
/// moex (DSV41_PLAN §35): slices_hash_kernel's sum over one P-th of a slice's words (block (p, j): words nw p / P .. nw (p + 1)
/// / P of job j, the tail with the last part), into parts[j P + p] - the hash is a sum of products modulo 2^64, so the parts'
/// sum is the one-block value whatever the split: one swapped-in slot's check ran on 3 blocks (1.9 ms), now on 3 P
__global__ void slices_hash_parts_kernel(const SliceJob* __restrict__ jobs, int P, uint64_t* __restrict__ parts) {
    const SliceJob j = jobs[blockIdx.y];
    const uint64_t nw = j.len / 8, tail = j.len % 8;
    const uint64_t i0 = nw * blockIdx.x / P, i1 = nw * (blockIdx.x + 1) / P;
    uint64_t h = 0;
    for (uint64_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) {
        uint64_t w;
        memcpy(&w, j.p + i * 8, 8);
        h += w * (2 * i + 1);
    }
    if (tail && threadIdx.x == 0 && (int) blockIdx.x == P - 1) {
        uint64_t w = 0;
        for (uint64_t b = 0; b < tail; ++b) w |= (uint64_t) j.p[nw * 8 + b] << (8 * b);
        h += w * (2 * nw + 1);
    }
    __shared__ uint64_t red[256];
    red[threadIdx.x] = h;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        parts[(size_t) blockIdx.y * P + blockIdx.x] = red[0];
        __threadfence_system();
    }
}
}  // namespace

void slices_hash_parts(const SliceJob* jobs, int n, int P, uint64_t* parts, void* stream) {
    if (n <= 0 || P <= 0) return;
    slices_hash_parts_kernel<<<dim3((unsigned) P, (unsigned) n), 256, 0, st(stream)>>>(jobs, P, parts);
    check_launch("slices_hash_parts");
}

void slices_hash(const SliceJob* jobs, int n, uint64_t* out, void* stream) {
    if (n <= 0) return;
    slices_hash_kernel<<<n, 256, 0, st(stream)>>>(jobs, out);
    check_launch("slices_hash");
}

uint64_t slices_hash_host(const uint8_t* p, uint64_t len) {
    const uint64_t nw = len / 8, tail = len % 8;
    uint64_t h = 0;
    for (uint64_t i = 0; i < nw; ++i) {
        uint64_t w;
        memcpy(&w, p + i * 8, 8);
        h += w * (2 * i + 1);
    }
    if (tail) {
        uint64_t w = 0;
        for (uint64_t b = 0; b < tail; ++b) w |= (uint64_t) p[nw * 8 + b] << (8 * b);
        h += w * (2 * nw + 1);
    }
    return h;
}

void window_load(const int64_t* m_pos, const float* m_rows, const int32_t* m_res, int64_t* d_pos, int32_t* d_res, int n_res,
                 float* dst, int rows, int width, int src_width, void* stream) {
    const size_t work = std::max<size_t>((size_t) rows * width, (size_t) n_res);
    const int nb = (int) std::max<size_t>(1, std::min<size_t>(256, (work + 255) / 256));
    window_load_kernel<<<nb, 256, 0, st(stream)>>>(m_pos, m_rows, m_res, d_pos, d_res, n_res, dst, rows, width, src_width);
    check_launch("window_load");
}

namespace {
/// moex (DSV41_PLAN §76): T rows of `width` values from rows of src_ld to rows of dst_ld (the head's two runs of rows into the
/// logits' rows) - a copy
__global__ void rows_copy_ld_kernel(const float* __restrict__ src, int src_ld, float* __restrict__ dst, int dst_ld, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i < width) dst[(size_t) t * dst_ld + i] = src[(size_t) t * src_ld + i];
}
}  // namespace

void rows_copy_ld(const float* src, int src_ld, float* dst, int dst_ld, int width, int T, void* stream) {
    if (width <= 0 || T <= 0) return;
    rows_copy_ld_kernel<<<dim3((unsigned) ((width + 255) / 256), (unsigned) T), 256, 0, st(stream)>>>(src, src_ld, dst, dst_ld, width);
    check_launch("rows_copy_ld");
}

void store_mapped(const float* x, float* m_out, size_t count, void* stream) {
    const int nb = (int) std::max<size_t>(1, std::min<size_t>(256, (count + 255) / 256));
    store_mapped_kernel<<<nb, 256, 0, st(stream)>>>(x, m_out, count);
    check_launch("store_mapped");
}

void store_draft(const int32_t* tok, const float* prob, int32_t* m_tok, float* m_prob, void* stream) {
    store_draft_kernel<<<1, 32, 0, st(stream)>>>(tok, prob, m_tok, m_prob);
    check_launch("store_draft");
}

void store_drafts(const int32_t* tok, const float* prob, int n, int32_t* m_tok, float* m_prob, void* stream) {
    if (n <= 0) return;
    store_drafts_kernel<<<1, 32, 0, st(stream)>>>(tok, prob, n, m_tok, m_prob);
    check_launch("store_drafts");
}

void copy_rows_w(const float* src, int64_t src_stride, float* dst, int64_t dst_stride, int n, int rows, void* stream) {
    if (rows <= 0 || n <= 0) return;
    copy_rows_kernel<<<dim3(std::max(1, std::min(8, (n + 255) / 256)), rows), 256, 0, st(stream)>>>(src, src_stride, dst, dst_stride, n);
    check_launch("copy_rows_w");
}

void copy_f32(const float* src, float* dst, size_t count, void* stream) {
    const int nb = (int) std::max<size_t>(1, std::min<size_t>(256, (count + 255) / 256));
    copy_f32_kernel<<<nb, 256, 0, st(stream)>>>(src, dst, count);
    check_launch("copy_f32");
}

namespace {
/// DSpark's Markov head step (V4.1): li[v] += w2[v, :] . w1[prev, :] with prev read from the device (the argmax before), a
/// warp per vocabulary row - the rows are only `rank` (256) wide, so gemv_f32_w's block per row spent 129,280 blocks on
/// 256-element dots (~1.3 ms a step; the matrix is 132 MB, ~0.2 ms at the card's bandwidth)
__global__ void markov_step_kernel(const float* __restrict__ w2, const float* __restrict__ w1, const int32_t* __restrict__ d_prev,
                                   int rank, int V, float* __restrict__ li) {
    const int64_t v = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (v >= V) return;   // warp-uniform
    const float* a = w2 + v * rank;
    const float* b = w1 + (int64_t) (*d_prev) * rank;
    float acc = 0.0f;
    for (int i = lane; i < rank; i += 32) acc += a[i] * b[i];
    acc = warp_sum(acc);
    if (lane == 0) li[v] += acc;
}
}  // namespace

void markov_step(const float* w2, const float* w1, const int32_t* d_prev, int rank, int V, float* li, void* stream) {
    markov_step_kernel<<<(V + 7) / 8, 256, 0, st(stream)>>>(w2, w1, d_prev, rank, V, li);
    check_launch("markov_step");
}

void argmax_prob_w(const float* x, int n, int32_t* out, float* out_p, int T, void* stream) {
    for (int t = 0; t < T; ++t) argmax_prob_kernel<<<1, 1024, 0, st(stream)>>>(x + (size_t) t * n, n, out + t, out_p + t);
    check_launch("argmax_prob_w");
}

void concat_rows(const float* a, const float* b, float* y, int n, int T, void* stream) {
    concat_rows_kernel<<<dim3(blocks(n, 256), T), 256, 0, st(stream)>>>(a, b, y, n);
    check_launch("concat_rows");
}

namespace {
__global__ void strided_bounds_kernel(int32_t* dst, int n, int step) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = i * step;
}
}  // namespace

void strided_bounds(int32_t* dst, int n, int step, void* stream) {
    strided_bounds_kernel<<<(n + 255) / 256, 256, 0, st(stream)>>>(dst, n, step);
    check_launch("strided_bounds");
}

void heads_q8_0(const void* W, const void* x_q8_1, float* y, int H, int n_in, int n_out, int T, void* stream) {
    if (n_in % 32 || T < 1) throw std::runtime_error("heads_q8_0: n_in % 32, T >= 1");
    static const bool tiled = [] { const char* e = std::getenv("STRATA_X_HEADS_TILED"); return !(e && e[0] == '0'); }();
    const int nb = n_in / 32;
    if (tiled && T > 8 && (nb == 4 || nb == 6 || nb == 8)) {   // a prompt's columns: the same bits in one launch
        const dim3 g(blocks(n_out, 64), blocks(T, 64), H);
        const auto* w = static_cast<const HQ80*>(W);
        const auto* x = static_cast<const HQ81*>(x_q8_1);
        if (nb == 4) heads_q8_0_tiled_kernel<4><<<g, 256, 0, st(stream)>>>(w, x, y, n_out, T);
        else if (nb == 6) heads_q8_0_tiled_kernel<6><<<g, 256, 0, st(stream)>>>(w, x, y, n_out, T);
        else heads_q8_0_tiled_kernel<8><<<g, 256, 0, st(stream)>>>(w, x, y, n_out, T);
        check_launch("heads_q8_0 (tiled)");
        return;
    }
    static const bool rows = [] { const char* e = std::getenv("STRATA_X_HEADS_ROWS"); return !(e && e[0] == '0'); }();
    if (rows && T <= 8) {   // a decode window: 4 rows a warp (heads_q8_0_rows_kernel: the same bits)
        const auto* w = static_cast<const HQ80*>(W);
        const auto* x = static_cast<const HQ81*>(x_q8_1);
        const dim3 g(blocks(n_out, 16), H);
        switch (T) {
#define STRATA_HEADS_ROWS(TC) \
    case TC: heads_q8_0_rows_kernel<4, TC><<<g, 128, 0, st(stream)>>>(w, x, y, n_in / 32, n_out, T, 0); break;
            STRATA_HEADS_ROWS(1) STRATA_HEADS_ROWS(2) STRATA_HEADS_ROWS(3) STRATA_HEADS_ROWS(4) STRATA_HEADS_ROWS(5)
            STRATA_HEADS_ROWS(6) STRATA_HEADS_ROWS(7) STRATA_HEADS_ROWS(8)
#undef STRATA_HEADS_ROWS
        }
        check_launch("heads_q8_0 (rows)");
        return;
    }
    for (int c0 = 0; c0 < T; c0 += 8)
        heads_q8_0_kernel<<<dim3(blocks(n_out, 4), H), 128, 0, st(stream)>>>(
            static_cast<const HQ80*>(W), static_cast<const HQ81*>(x_q8_1), y, n_in / 32, n_out, std::min(8, T - c0), T, c0);
    check_launch("heads_q8_0");
}

void gather_cols(const void* src, void* dst, const int32_t* cols, size_t col_bytes, int m, void* stream) {
    if (col_bytes % 16) throw std::runtime_error("gather_cols: columns must be 16-byte multiples");
    const int words = (int) (col_bytes / 16);
    gather_cols_kernel<<<dim3(std::min(blocks(words, 256), 64), m), 256, 0, st(stream)>>>(
        static_cast<const uint4*>(src), static_cast<uint4*>(dst), cols, words);
    check_launch("gather_cols");
}

void scatter_rows(const float* src, float* dst, const int32_t* rows, int n, int m, void* stream) {
    scatter_rows_kernel<<<dim3(std::min(blocks(n, 256), 64), m), 256, 0, st(stream)>>>(src, dst, rows, n);
    check_launch("scatter_rows");
}

void swiglu_clamp_inplace(float* gu, int64_t rows, int nf, float limit, void* stream, int64_t ld) {
    const int64_t n = rows * nf;
    if (n <= 0) return;
    swiglu_clamp_inplace_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st(stream)>>>(gu, rows, nf, ld > 0 ? ld : 2 * (int64_t) nf,
                                                                                      limit);
    check_launch("swiglu_clamp_inplace");
}

void swiglu_clamp_split(const float* gu, float* h, int64_t rows, int nf, float limit, void* stream) {
    const int64_t n = rows * nf;
    if (n <= 0) return;
    swiglu_clamp_split_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st(stream)>>>(gu, h, rows, nf, limit);
    check_launch("swiglu_clamp_split");
}

void moe_combine_rows(const float* Dm, const int32_t* inv, const float* weights, const float* shared, float* out, int n,
                      int k, int T, void* stream) {
    combine_rows_kernel<<<dim3(blocks(n, 256), T), 256, 0, st(stream)>>>(Dm, inv, weights, shared, out, n, k);
    check_launch("moe_combine_rows");
}

void permute_abc(const float* src, float* dst, int A, int B, int C, void* stream) {
    const size_t n = (size_t) A * B * C;
    permute_abc_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st(stream)>>>(src, dst, A, B, C);
    check_launch("permute_abc");
}

void layer_norm_w(const float* x, const float* w, const float* b, float* y, int rows, int cols, float eps, void* stream) {
    layer_norm_kernel_rows<128><<<rows, 128, 0, st(stream)>>>(x, w, b, y, cols, eps);
    check_launch("layer_norm_w");
}


// ---------------------------------------------------------------------------------------------------------------
// MiMo-V2 (mimo2): the GQA attention.  RoPE as ggml-cuda's rope_neox (theta = pos * powf(theta_scale, i), freq_scale 1, no
// YaRN, attn_factor 1 - with the accurate sinf/cosf/powf where ggml-cuda's -use_fast_math build takes the intrinsics);
// the caches FP16 as llama.cpp's (F32 -> F16 round-to-nearest); the attention in FP32 over the FP16 rows, softmax with the
// head's sink logit (unscaled) in the denominator, over the keys at positions (p - window, p] (all of [0, p] when window 0).
namespace {

__global__ void rope_neox_kernel(float* __restrict__ x, int row_stride, int head_off, int n_heads, int head_dim, int n_rot,
                                 float theta_scale, int64_t pos0, const int64_t* __restrict__ pos_dev) {
    const int t = blockIdx.y;
    const int64_t p = (pos_dev ? *pos_dev : pos0) + t;
    const int half = n_rot / 2, pairs = n_heads * half;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pairs; i += gridDim.x * blockDim.x) {
        const int h = i / half, k = i % half;
        float* v = x + (size_t) t * row_stride + head_off + (size_t) h * head_dim;
        const float theta = (float) p * powf(theta_scale, (float) k);
        const float c = cosf(theta), sn = sinf(theta);
        const float x0 = v[k], x1 = v[k + half];
        v[k] = x0 * c - x1 * sn;
        v[k + half] = x0 * sn + x1 * c;
    }
}

__global__ void kv_store_kernel(const float* __restrict__ src, int row_stride, int k_off, int v_off, __half* __restrict__ kc,
                                __half* __restrict__ vc, int64_t rows, int nk, int nv, int64_t pos0,
                                const int64_t* __restrict__ pos_dev) {
    const int t = blockIdx.y;
    const int64_t r = ((pos_dev ? *pos_dev : pos0) + t) % rows;
    const float* sr = src + (size_t) t * row_stride;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nk + nv; i += gridDim.x * blockDim.x) {
        if (i < nk) kc[(size_t) r * nk + i] = __float2half_rn(sr[k_off + i]);
        else vc[(size_t) r * nv + (i - nk)] = __float2half_rn(sr[v_off + i - nk]);
    }
}

constexpr int kGqaKT = 32;   ///< keys per tile (one per lane in the score pass)

/// One block = (key chunk c, KV head g, token t): the G query heads that share KV head g over the chunk's keys, an online
/// softmax per head.  Warp w owns heads [w * G/4, (w+1) * G/4): the score pass puts a key on each lane, the value pass 4 of
/// the DV dims on each lane.  n_chunks == 1 writes the normalized output (with the sink); else the chunk's (acc, m, l).
template <int DK, int DV>
__global__ void __launch_bounds__(128) gqa_attend_kernel(const float* __restrict__ q, int q_stride, const __half* __restrict__ kc,
                                                          const __half* __restrict__ vc, int64_t rows, int H, int n_kv, int window,
                                                          const float* __restrict__ sinks, float scale, float* __restrict__ out,
                                                          float* __restrict__ partial, int n_chunks, int64_t pos0,
                                                          const int64_t* __restrict__ pos_dev, int noncausal) {
    static_assert(DV == 128, "a lane owns 4 of 128 value dims");
    constexpr int KP = DK + 2, VP = DV + 4;   // padded rows: an odd word stride for K, 8-byte reads for V
    __shared__ float qs[16][DK];
    __shared__ __half ks[kGqaKT][KP];
    __shared__ __half vs[kGqaKT][VP];
    __shared__ float sc[kGqaKT][16];
    const int c = blockIdx.x, gk = blockIdx.y, t = blockIdx.z;
    const int G = H / n_kv, HW = G / 4;          // heads per group, heads per warp
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int64_t base = pos_dev ? *pos_dev : pos0;
    const int64_t p = base + t;
    // causal: keys up to the query's own position; non-causal (DFlash's noise block): up to the batch's last row - the lower
    // bound stays the query's own window (llama.cpp's SWA mask: p - p_key < window)
    const int64_t hi = noncausal ? base + (int64_t) gridDim.z - 1 : p;
    const int64_t lo = window > 0 ? (p - window + 1 > 0 ? p - window + 1 : 0) : 0;
    const int64_t len = hi - lo + 1;
    int64_t per = (len + n_chunks - 1) / n_chunks;
    per = (per + kGqaKT - 1) / kGqaKT * kGqaKT;
    const int64_t k0 = lo + (int64_t) c * per, k1 = k0 + per < hi + 1 ? k0 + per : hi + 1;
    for (int i = threadIdx.x; i < G * DK; i += 128) {
        const int h = i / DK, d = i % DK;
        qs[h][d] = q[(size_t) t * q_stride + (size_t) (gk * G + h) * DK + d];
    }
    float m[4], l[4], acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        m[a] = -INFINITY;
        l[a] = 0.0f;
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = 0.0f;
    }
    const int nk = n_kv * DK, nv = n_kv * DV;
    for (int64_t j0 = k0; j0 < k1; j0 += kGqaKT) {
        const int nt = (int) (k1 - j0 < kGqaKT ? k1 - j0 : kGqaKT);
        __syncthreads();   // the previous tile's readers are done (and qs is written, the first time)
        for (int i = threadIdx.x; i < nt * (DK / 2); i += 128) {
            const int j = i / (DK / 2), d2 = i % (DK / 2);
            const int64_t r = (j0 + j) % rows;
            reinterpret_cast<__half2*>(&ks[j][0])[d2] = reinterpret_cast<const __half2*>(kc + (size_t) r * nk + (size_t) gk * DK)[d2];
        }
        for (int i = threadIdx.x; i < nt * (DV / 4); i += 128) {
            const int j = i / (DV / 4), d4 = i % (DV / 4);
            const int64_t r = (j0 + j) % rows;
            reinterpret_cast<uint2*>(&vs[j][0])[d4] = reinterpret_cast<const uint2*>(vc + (size_t) r * nv + (size_t) gk * DV)[d4];
        }
        __syncthreads();
        // scores: lane j, the warp's heads
        for (int a = 0; a < HW; ++a) {
            const int h = w * HW + a;
            float dot = 0.0f;
            if (lane < nt) {
#pragma unroll 8
                for (int d = 0; d < DK; ++d) dot = fmaf(qs[h][d], __half2float(ks[lane][d]), dot);
            }
            const float sv = lane < nt ? dot * scale : -INFINITY;
            float mt = sv;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
            const float mn = fmaxf(m[a], mt);
            const float pj = lane < nt ? expf(sv - mn) : 0.0f;
            const float alpha = m[a] == -INFINITY ? 0.0f : expf(m[a] - mn);
            l[a] = l[a] * alpha + warp_sum(pj);
            m[a] = mn;
#pragma unroll
            for (int b = 0; b < 4; ++b) acc[a][b] *= alpha;
            sc[lane][h] = pj;
        }
        __syncwarp();
        // values: lane owns dims [4 lane, 4 lane + 4)
        for (int j = 0; j < nt; ++j) {
            const uint2 raw = reinterpret_cast<const uint2*>(&vs[j][0])[lane];
            const __half2 v01 = *reinterpret_cast<const __half2*>(&raw.x), v23 = *reinterpret_cast<const __half2*>(&raw.y);
            const float vv[4] = {__low2float(v01), __high2float(v01), __low2float(v23), __high2float(v23)};
            for (int a = 0; a < HW; ++a) {
                const float pj = sc[j][w * HW + a];
#pragma unroll
                for (int b = 0; b < 4; ++b) acc[a][b] = fmaf(pj, vv[b], acc[a][b]);
            }
        }
    }
    for (int a = 0; a < HW; ++a) {
        const int h = gk * G + w * HW + a;
        if (n_chunks == 1) {
            float M = m[a], L = l[a];
            const float sk = sinks ? sinks[h] : -INFINITY;
            float f = 1.0f;
            if (sinks) {
                M = fmaxf(m[a], sk);
                f = m[a] == -INFINITY ? 0.0f : expf(m[a] - M);
                L = l[a] * f + expf(sk - M);
            }
            const float inv = L > 0.0f ? 1.0f / L : 0.0f;
            float* o = out + ((size_t) t * H + h) * DV + 4 * lane;
#pragma unroll
            for (int b = 0; b < 4; ++b) o[b] = acc[a][b] * f * inv;
        } else {
            float* pr = partial + (((size_t) t * H + h) * n_chunks + c) * (DV + 2);
#pragma unroll
            for (int b = 0; b < 4; ++b) pr[4 * lane + b] = acc[a][b];
            if (lane == 0) {
                pr[DV] = m[a];
                pr[DV + 1] = l[a];
            }
        }
    }
}

/// out[t][h] = the chunks' partials merged: M = max(m_c, sink), L = sum l_c e^(m_c - M) + e^(sink - M), out = sum acc_c e^(m_c - M) / L.
template <int DV>
__global__ void gqa_merge_kernel(const float* __restrict__ partial, const float* __restrict__ sinks, float* __restrict__ out, int H,
                                 int n_chunks) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const float* pr = partial + ((size_t) t * H + h) * n_chunks * (DV + 2);
    float M = sinks ? sinks[h] : -INFINITY;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, pr[(size_t) c * (DV + 2) + DV]);
    float L = sinks ? expf(sinks[h] - M) : 0.0f, a = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float* pc = pr + (size_t) c * (DV + 2);
        if (pc[DV] == -INFINITY) continue;
        const float f = expf(pc[DV] - M);
        L += pc[DV + 1] * f;
        a += pc[d] * f;
    }
    out[((size_t) t * H + h) * DV + d] = L > 0.0f ? a / L : 0.0f;
}

/// THE PROMPT PATH'S GQA (a sub-batch of T queries): one block per (KV head g, 8 consecutive query tokens) - every K/V tile
/// loaded once into shared memory serves all 8 tokens' G heads (the per-token kernel above read each tile once per token: the
/// full layers' K/V traffic grew with T x the context).  Warp w owns token t0 + w and all G heads of group g: the score pass
/// puts a key on each lane (G dot products of DK), the value pass 4 of the DV dims on each lane; online softmax per (token,
/// head); a token's keys are its own window (p - window, p], so the tile's keys past a token's position are masked for it.
/// The sink logit (unscaled) joins the denominator at the end.  Dynamic shared memory: q (8 x G x DK halves), K and V tiles,
/// the scores.
template <int DK, int DV, int G, int BQ>
__global__ void __launch_bounds__(256) gqa_prefill_kernel(const float* __restrict__ q, int q_stride, const __half* __restrict__ kc,
                                                           const __half* __restrict__ vc, int64_t rows, int H, int n_kv, int window,
                                                           const float* __restrict__ sinks, float scale, float* __restrict__ out,
                                                           int64_t pos0, int T) {
    static_assert(DV == 128 && G <= 16 && BQ <= 8, "a lane owns 4 of 128 value dims; at most 16 heads per KV head");
    constexpr int KP = DK + 2, VP = DV + 4;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* qs = reinterpret_cast<float*>(smem_raw);                         // [BQ][G][DK], F32 as the decode kernel's
    __half* ks = reinterpret_cast<__half*>(qs + BQ * G * DK);                // [kGqaKT][KP]
    __half* vs = ks + kGqaKT * KP;                                            // [kGqaKT][VP]
    float* sc = reinterpret_cast<float*>(vs + kGqaKT * VP);                   // [BQ][kGqaKT][G]
    const int gk = blockIdx.y, t0 = blockIdx.x * BQ;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int t = t0 + w;
    const bool live = t < T;
    const int64_t p = pos0 + (live ? t : T - 1);
    const int64_t lo = window > 0 ? (p - window + 1 > 0 ? p - window + 1 : 0) : 0;
    // the block's key range: the union of its tokens' windows
    const int tl = min(T, t0 + BQ) - 1;
    const int64_t p_lo = pos0 + t0, p_hi = pos0 + tl;
    const int64_t k_lo = window > 0 ? (p_lo - window + 1 > 0 ? p_lo - window + 1 : 0) : 0, k_hi = p_hi + 1;
    for (int i = threadIdx.x; i < BQ * G * DK; i += blockDim.x) {
        const int tt = i / (G * DK), h = (i / DK) % G, d = i % DK;
        const int tq = t0 + tt < T ? t0 + tt : T - 1;
        qs[i] = q[(size_t) tq * q_stride + (size_t) (gk * G + h) * DK + d];
    }
    float m[G], l[G], acc[G][4];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.0f;
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[h][b] = 0.0f;
    }
    const int nk = n_kv * DK, nv = n_kv * DV;
    float* scw = sc + (size_t) w * kGqaKT * G;
    for (int64_t j0 = k_lo; j0 < k_hi; j0 += kGqaKT) {
        const int nt = (int) (k_hi - j0 < kGqaKT ? k_hi - j0 : kGqaKT);
        __syncthreads();
        for (int i = threadIdx.x; i < nt * (DK / 2); i += blockDim.x) {
            const int j = i / (DK / 2), d2 = i % (DK / 2);
            const int64_t r = (j0 + j) % rows;
            reinterpret_cast<__half2*>(ks + j * KP)[d2] = reinterpret_cast<const __half2*>(kc + (size_t) r * nk + (size_t) gk * DK)[d2];
        }
        for (int i = threadIdx.x; i < nt * (DV / 4); i += blockDim.x) {
            const int j = i / (DV / 4), d4 = i % (DV / 4);
            const int64_t r = (j0 + j) % rows;
            reinterpret_cast<uint2*>(vs + j * VP)[d4] = reinterpret_cast<const uint2*>(vc + (size_t) r * nv + (size_t) gk * DV)[d4];
        }
        __syncthreads();
        // this warp's token: is the tile's key `lane` in its window?
        const int64_t kp = j0 + lane;
        const bool vis = live && lane < nt && kp >= lo && kp <= p;
        const float* qt = qs + (size_t) w * G * DK;
#pragma unroll
        for (int h = 0; h < G; ++h) {
            float dot = 0.0f;
            if (vis) {
                const float* qh = qt + h * DK;
                const __half2* k2 = reinterpret_cast<const __half2*>(ks + lane * KP);
#pragma unroll 8
                for (int d2 = 0; d2 < DK / 2; ++d2) {
                    const float2 kf = __half22float2(k2[d2]);
                    dot = fmaf(qh[2 * d2], kf.x, dot);
                    dot = fmaf(qh[2 * d2 + 1], kf.y, dot);
                }
            }
            const float sv = vis ? dot * scale : -INFINITY;
            float mt = sv;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, o));
            const float mn = fmaxf(m[h], mt);
            const float pj = vis ? expf(sv - mn) : 0.0f;
            const float alpha = m[h] == -INFINITY ? (mn == -INFINITY ? 1.0f : 0.0f) : expf(m[h] - mn);
            l[h] = l[h] * alpha + warp_sum(pj);
            m[h] = mn;
#pragma unroll
            for (int b = 0; b < 4; ++b) acc[h][b] *= alpha;
            scw[lane * G + h] = pj;
        }
        __syncwarp();
        for (int j = 0; j < nt; ++j) {
            const uint2 raw = reinterpret_cast<const uint2*>(vs + j * VP)[lane];
            const __half2 v01 = *reinterpret_cast<const __half2*>(&raw.x), v23 = *reinterpret_cast<const __half2*>(&raw.y);
            const float vv[4] = {__low2float(v01), __high2float(v01), __low2float(v23), __high2float(v23)};
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float pj = scw[j * G + h];
#pragma unroll
                for (int b = 0; b < 4; ++b) acc[h][b] = fmaf(pj, vv[b], acc[h][b]);
            }
        }
        __syncwarp();
    }
    if (!live) return;
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const int hh = gk * G + h;
        float M = m[h], L = l[h], f = 1.0f;
        if (sinks) {
            const float sk = sinks[hh];
            M = fmaxf(m[h], sk);
            f = m[h] == -INFINITY ? 0.0f : expf(m[h] - M);
            L = l[h] * f + expf(sk - M);
        }
        const float inv = L > 0.0f ? 1.0f / L : 0.0f;
        float* o = out + ((size_t) t * H + hh) * DV + 4 * lane;
#pragma unroll
        for (int b = 0; b < 4; ++b) o[b] = acc[h][b] * f * inv;
    }
}

/// THE PROMPT PATH'S GQA ON TENSOR CORES (MiMo): gqa_prefill_kernel's tiles with Strata's qsa_prompt_attn numerics.  A
/// block of NW warps = (KV head g, NW x 16 / G consecutive query tokens); warp w owns 16 query rows, 16 / G tokens x the G
/// heads that share KV head g (row r: token r / G, head r % G).  The block stages 32-key chunks of K and V by cp.async,
/// double-buffered; every warp runs the flash-attention loop over the chunks on its own rows:
///   * s = q.k by m16n8k16 FP16 MMAs with FP32 accumulation, in registers; p = exp2(s - m) re-packed from those registers
///     as the A fragments of p.v (a warp owns all the dims of its rows: no cross-warp sum, no shared-memory round trip);
///   * q and p are split into FP16 hi + lo parts (two MMAs each), ~22 bits: the result differs from gqa_prefill_kernel by
///     summation order and exp2f's rounding, not by an FP16 cast (K and V are FP16 in the cache, exact);
///   * each chunk's p.v lands in fresh accumulators and joins the running output by FP32 adds (not through the MMA's C,
///     whose truncating accumulation would bias a sum over the context's thousands of chunks);
///   * q is pre-scaled per ROW by the power of two that puts its largest value near 2^14 (exact; keeps the lo half out of
///     FP16's subnormals), p by 2^14 (its max is 1); both come back out exactly;
///   * a row's keys are its token's window [p - window + 1, p] (window 0: [0, p]), the others -inf; the sink logit
///     (unscaled) joins the denominator at the end.
/// Scores in the log2 domain (s * scale * log2 e).  x_kernels_parity bounds the difference to the FP32 kernels.
constexpr int kGmCh = 32, kGmK = 192, kGmV = 128, kGmKS = kGmK + 8, kGmVS = kGmV + 8;   // 400- and 272-byte smem rows

template <int G, int NW>
__global__ void __launch_bounds__(NW * 32, 1) gqa_prefill_mma_kernel(const float* __restrict__ q, int q_stride,
                                                                      const __half* __restrict__ kc, const __half* __restrict__ vc,
                                                                      int64_t rows, int H, int n_kv, int window,
                                                                      const float* __restrict__ sinks, float scale,
                                                                      float* __restrict__ out, int64_t pos0, int T) {
#if STRATA_X_MMA
    static_assert(G == 8 || G == 16, "16 rows per warp: one token of 16 heads or two of 8");
    constexpr int TW = 16 / G, BQ = NW * TW, NT = NW * 32;
    constexpr float kL2e = 1.4426950408889634f;
    __shared__ __align__(16) __half Ks[2][kGmCh][kGmKS];
    __shared__ __align__(16) __half Vs[2][kGmCh][kGmVS];
    const int gk = blockIdx.y, t0 = blockIdx.x * BQ;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
    // this thread's rows gid (a) and gid + 8 (b): their tokens (clamped for a partial last tile), heads and windows
    const int ta = t0 + w * TW + gid / G, tb = t0 + w * TW + (gid + 8) / G;
    const int ha = gk * G + gid % G, hb = gk * G + (gid + 8) % G;
    const int64_t pa = pos0 + (ta < T ? ta : T - 1), pb = pos0 + (tb < T ? tb : T - 1);
    const int64_t loa = window > 0 && pa - window + 1 > 0 ? pa - window + 1 : 0;
    const int64_t lob = window > 0 && pb - window + 1 > 0 ? pb - window + 1 : 0;
    // the block's keys: the union of its tokens' windows; the warp's (to skip the chunks none of its rows sees)
    const int tl = (t0 + BQ < T ? t0 + BQ : T) - 1;
    const int64_t k_lo = window > 0 && pos0 + t0 - window + 1 > 0 ? pos0 + t0 - window + 1 : 0, k_hi = pos0 + tl + 1;
    const bool live = t0 + w * TW < T;
    const int64_t w_lo = loa < lob ? loa : lob, w_hi = pa > pb ? pa : pb;
    const bool wrap = k_hi > rows;   // a sliding-window ring: key j in row j % rows

    // q into A fragments: the row maxima first (the 4 threads of a quad hold a row), then hi / lo of q * 2^(14 - e)
    const float* qa = q + (size_t) (pa - pos0) * q_stride + (size_t) ha * kGmK;
    const float* qb = q + (size_t) (pb - pos0) * q_stride + (size_t) hb * kGmK;
    float ma = 0.0f, mb = 0.0f;
#pragma unroll
    for (int kk = 0; kk < kGmK / 16; ++kk)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int c = kk * 16 + 2 * tig + hf * 8;
            const float2 xa = *reinterpret_cast<const float2*>(qa + c), xb = *reinterpret_cast<const float2*>(qb + c);
            ma = fmaxf(ma, fmaxf(fabsf(xa.x), fabsf(xa.y)));
            mb = fmaxf(mb, fmaxf(fabsf(xb.x), fabsf(xb.y)));
        }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
        ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, o));
        mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, o));
    }
    int ea = 0, eb = 0;
    if (ma > 0.0f) frexpf(ma, &ea);   // ma < 2^ea
    if (mb > 0.0f) frexpf(mb, &eb);
    const float upa = ldexpf(1.0f, 14 - ea), upb = ldexpf(1.0f, 14 - eb);
    const float dna = ldexpf(scale * kL2e, ea - 14), dnb = ldexpf(scale * kL2e, eb - 14);
    uint32_t qh[kGmK / 16][4], ql[kGmK / 16][4];
#pragma unroll
    for (int kk = 0; kk < kGmK / 16; ++kk)
#pragma unroll
        for (int r = 0; r < 4; ++r) {   // a[0]: row a, cols 2 tig..; a[1]: row b; a[2] / a[3]: the same at cols + 8
            const int c = kk * 16 + 2 * tig + (r >> 1) * 8;
            float2 x = *reinterpret_cast<const float2*>(((r & 1) ? qb : qa) + c);
            const float up = (r & 1) ? upb : upa;
            x.x *= up;
            x.y *= up;
            const __half2 hi = __floats2half2_rn(x.x, x.y);
            const float2 hf = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(x.x - hf.x, x.y - hf.y);
            qh[kk][r] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t*>(&lo);
        }

    const int nk = n_kv * kGmK, nv = n_kv * kGmV;
    auto issue = [&](int64_t j0, int st) {   // chunk j0 .. j0 + 31 into stage st; keys past the block's last are zeros
        for (int i = tid; i < kGmCh * (kGmK / 8); i += NT) {
            const int j = i / (kGmK / 8), c = i % (kGmK / 8);
            const int64_t key = j0 + j;
            const bool ok = key < k_hi;
            const int64_t r = ok ? (wrap ? key % rows : key) : 0;
            ma_cp16(&Ks[st][j][c * 8], kc + (size_t) r * nk + (size_t) gk * kGmK + c * 8, ok);
        }
        for (int i = tid; i < kGmCh * (kGmV / 8); i += NT) {
            const int j = i / (kGmV / 8), c = i % (kGmV / 8);
            const int64_t key = j0 + j;
            const bool ok = key < k_hi;
            const int64_t r = ok ? (wrap ? key % rows : key) : 0;
            ma_cp16(&Vs[st][j][c * 8], vc + (size_t) r * nv + (size_t) gk * kGmV + c * 8, ok);
        }
        ma_commit();
    };

    float acc[kGmV / 8][4];   // 2^14 x the output: n-tile j = dims 8 j .. 8 j + 7, rows a (0, 1) and b (2, 3)
#pragma unroll
    for (int j = 0; j < kGmV / 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.0f, l_b = 0.0f;   // l: this thread's share of the row sums
    const int n_ch = (int) ((k_hi - k_lo + kGmCh - 1) / kGmCh);
    issue(k_lo, 0);
    for (int ci = 0; ci < n_ch; ++ci) {
        const int st = ci & 1;
        const int64_t j0 = k_lo + (int64_t) ci * kGmCh;
        if (ci + 1 < n_ch) {
            issue(j0 + kGmCh, st ^ 1);
            ma_wait<1>();
        } else {
            ma_wait<0>();
        }
        __syncthreads();
        if (live && j0 <= w_hi && j0 + kGmCh - 1 >= w_lo) {
            // s = q.k: 4 n-tiles of 8 keys, 12 k-steps of 16 dims (two per ldmatrix.x4), hi + lo
            float s[kGmCh / 8][4];
#pragma unroll
            for (int nt = 0; nt < kGmCh / 8; ++nt) {
                s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
                for (int kk2 = 0; kk2 < kGmK / 32; ++kk2) {
                    uint32_t b[4];
                    ma_ldsm(b, &Ks[st][nt * 8 + (lane & 7)][kk2 * 32 + (lane >> 3) * 8]);
                    ma_mma(s[nt], qh[2 * kk2], b);
                    ma_mma(s[nt], ql[2 * kk2], b);
                    ma_mma(s[nt], qh[2 * kk2 + 1], b + 2);
                    ma_mma(s[nt], ql[2 * kk2 + 1], b + 2);
                }
            }
            // the window, the scale into the log2 domain, the rows' maxima over the quad
            float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < kGmCh / 8; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int64_t key = j0 + nt * 8 + 2 * tig + e;
                    s[nt][e] = key >= loa && key <= pa ? s[nt][e] * dna : -INFINITY;
                    s[nt][2 + e] = key >= lob && key <= pb ? s[nt][2 + e] * dnb : -INFINITY;
                    mxa = fmaxf(mxa, s[nt][e]);
                    mxb = fmaxf(mxb, s[nt][2 + e]);
                }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, o));
                mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, o));
            }
            const float na = fmaxf(m_a, mxa), nb = fmaxf(m_b, mxb);
            const float ba = na == -INFINITY ? 0.0f : na, bb = nb == -INFINITY ? 0.0f : nb;   // a row with no key yet
            const float aa = exp2f(m_a - ba), ab = exp2f(m_b - bb);
            m_a = na;
            m_b = nb;
            float sa = 0.0f, sb = 0.0f;
#pragma unroll
            for (int nt = 0; nt < kGmCh / 8; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    s[nt][e] = exp2f(s[nt][e] - ba);
                    s[nt][2 + e] = exp2f(s[nt][2 + e] - bb);
                    sa += s[nt][e];
                    sb += s[nt][2 + e];
                }
            l_a = fmaf(l_a, aa, sa);
            l_b = fmaf(l_b, ab, sb);
#pragma unroll
            for (int j = 0; j < kGmV / 8; ++j) {
                acc[j][0] *= aa;
                acc[j][1] *= aa;
                acc[j][2] *= ab;
                acc[j][3] *= ab;
            }
            // acc += p.v: k-step ks = keys 16 ks .. +15 (n-tiles 2 ks, 2 ks + 1 of s are its A fragment), hi + lo
#pragma unroll
            for (int ks = 0; ks < kGmCh / 16; ++ks) {
                const float x[8] = {s[2 * ks][0] * 16384.0f,     s[2 * ks][1] * 16384.0f,     s[2 * ks][2] * 16384.0f,
                                    s[2 * ks][3] * 16384.0f,     s[2 * ks + 1][0] * 16384.0f, s[2 * ks + 1][1] * 16384.0f,
                                    s[2 * ks + 1][2] * 16384.0f, s[2 * ks + 1][3] * 16384.0f};
                uint32_t ah[4], al[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const __half2 hi = __floats2half2_rn(x[2 * r], x[2 * r + 1]);
                    const float2 hf = __half22float2(hi);
                    ah[r] = *reinterpret_cast<const uint32_t*>(&hi);
                    al[r] = ma_pack(x[2 * r] - hf.x, x[2 * r + 1] - hf.y);
                }
                const int mq = lane >> 3;
#pragma unroll
                for (int j2 = 0; j2 < kGmV / 16; ++j2) {   // two n-tiles (16 dims) per transposed ldmatrix.x4
                    // into fresh accumulators, then an FP32 add: the MMA's own accumulation truncates its addends to
                    // the largest one's exponent, so a running sum fed back through it drifts down chunk after chunk
                    // (~1e-4 relative at 8K keys, ~1e-3 at 64K - x_kernels_parity)
                    uint32_t b[4];
                    ma_ldsm_t(b, &Vs[st][ks * 16 + (lane & 7) + (mq & 1) * 8][(j2 * 2 + (mq >> 1)) * 8]);
                    float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, d1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    ma_mma(d0, ah, b);
                    ma_mma(d0, al, b);
                    ma_mma(d1, ah, b + 2);
                    ma_mma(d1, al, b + 2);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        acc[2 * j2][i] += d0[i];
                        acc[2 * j2 + 1][i] += d1[i];
                    }
                }
            }
        }
        __syncthreads();   // every warp is done with stage st: the next chunk's issue may overwrite it
    }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
        l_a += __shfl_xor_sync(0xffffffffu, l_a, o);
        l_b += __shfl_xor_sync(0xffffffffu, l_b, o);
    }
    // the sink (log2 domain) and the output; rows of tokens past T are not written
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int t = half ? tb : ta, h = half ? hb : ha;
        if (t >= T) continue;
        const float m = half ? m_b : m_a, l = half ? l_b : l_a;
        float M = m, L = l, f = 1.0f;
        if (sinks) {
            const float sk = sinks[h] * kL2e;
            M = fmaxf(m, sk);
            f = m == -INFINITY ? 0.0f : exp2f(m - M);
            L = l * f + exp2f(sk - M);
        }
        const float inv = L > 0.0f ? f / (16384.0f * L) : 0.0f;
        float* o = out + ((size_t) t * H + h) * kGmV + 2 * tig;
#pragma unroll
        for (int j = 0; j < kGmV / 8; ++j)
            *reinterpret_cast<float2*>(o + 8 * j) = make_float2(acc[j][2 * half] * inv, acc[j][2 * half + 1] * inv);
    }
    ma_wait<0>();
#else
    __trap();   // built for a card without mma.sync m16n8k16 f16 (sm_80): gqa_prefill_mma refuses such a device
#endif
}

/// THE DECODE WINDOW'S GQA ON TENSOR CORES (MiMo's full layers): gqa_prefill_mma_kernel's loop, split over the keys as
/// gqa_attend_kernel splits them.  One block = one warp = (key chunk c, KV head g, 16 / G tokens); its 16 rows are the G heads
/// that share KV head g (row r: token r / G, head r % G), so a decode token's 16 heads of a full layer are one m16 tile - the
/// work gqa_attend_kernel does as 16 scalar dot products a key, which made a 135K-token MiMo context decode at ~10 T/s.  The
/// chunk's keys stream through cp.async double buffers; the numerics are gqa_prefill_mma_kernel's (q and p in FP16 hi + lo,
/// fresh accumulators per 32-key tile joined by FP32 adds, the per-row power-of-two prescale); the position comes from the
/// device when given (the decode graphs).  Writes the chunk's (acc, m, l) in gqa_attend_kernel's layout and natural-log
/// domain, so gqa_merge_kernel joins the chunks and the sink unchanged.
template <int G, int KT>
__global__ void __launch_bounds__(32) gqa_decode_mma_kernel(const float* __restrict__ q, int q_stride, const __half* __restrict__ kc,
                                                            const __half* __restrict__ vc, int64_t rows, int H, int n_kv, int window,
                                                            float scale, float* __restrict__ partial, int n_chunks, int64_t pos0,
                                                            const int64_t* __restrict__ pos_dev, int T) {
#if STRATA_X_MMA
    static_assert(G == 8 || G == 16, "16 rows per warp: one token of 16 heads or two of 8");
    static_assert(KT == 16 || KT == 32, "16 or 32 keys a tile: 21.5 / 43 KB of double-buffered K and V a warp, 4 / 2 warps an SM");
    constexpr int TW = 16 / G;
    constexpr float kL2e = 1.4426950408889634f, kLn2 = 0.6931471805599453f;
    __shared__ __align__(16) __half Ks[2][KT][kGmKS];
    __shared__ __align__(16) __half Vs[2][KT][kGmVS];
    const int c = blockIdx.x, gk = blockIdx.y, t0 = blockIdx.z * TW;
    const int lane = threadIdx.x, gid = lane >> 2, tig = lane & 3;
    const int64_t base = pos_dev ? *pos_dev : pos0;
    // this thread's rows gid (a) and gid + 8 (b): their tokens (clamped past T), heads and key windows
    const int ta = t0 + gid / G, tb = t0 + (gid + 8) / G;
    const int ha = gk * G + gid % G, hb = gk * G + (gid + 8) % G;
    const int64_t pa = base + (ta < T ? ta : T - 1), pb = base + (tb < T ? tb : T - 1);
    const int64_t loa = window > 0 && pa - window + 1 > 0 ? pa - window + 1 : 0;
    const int64_t lob = window > 0 && pb - window + 1 > 0 ? pb - window + 1 : 0;
    // the block's keys: chunk c of the union of its tokens' windows, in whole KT-key tiles (a middle chunk never ends inside
    // a tile, so only the last chunk's last tile reaches past a token's position - masked below)
    const int tl = (t0 + TW < T ? t0 + TW : T) - 1;
    const int64_t p_lo = base + t0, p_hi = base + tl;
    const int64_t u_lo = window > 0 && p_lo - window + 1 > 0 ? p_lo - window + 1 : 0, u_hi = p_hi + 1;
    int64_t per = (u_hi - u_lo + n_chunks - 1) / n_chunks;
    per = (per + KT - 1) / KT * KT;
    const int64_t k_lo = u_lo + (int64_t) c * per, k_hi = k_lo + per < u_hi ? k_lo + per : u_hi;
    const bool wrap = u_hi > rows;   // a sliding-window ring: key j in row j % rows

    // q into A fragments (gqa_prefill_mma_kernel's): the rows' maxima over the quad, then hi / lo of q * 2^(14 - e)
    const float* qa = q + (size_t) (pa - base) * q_stride + (size_t) ha * kGmK;
    const float* qb = q + (size_t) (pb - base) * q_stride + (size_t) hb * kGmK;
    float ma = 0.0f, mb = 0.0f;
#pragma unroll
    for (int kk = 0; kk < kGmK / 16; ++kk)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int cc = kk * 16 + 2 * tig + hf * 8;
            const float2 xa = *reinterpret_cast<const float2*>(qa + cc), xb = *reinterpret_cast<const float2*>(qb + cc);
            ma = fmaxf(ma, fmaxf(fabsf(xa.x), fabsf(xa.y)));
            mb = fmaxf(mb, fmaxf(fabsf(xb.x), fabsf(xb.y)));
        }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
        ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, o));
        mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, o));
    }
    int ea = 0, eb = 0;
    if (ma > 0.0f) frexpf(ma, &ea);
    if (mb > 0.0f) frexpf(mb, &eb);
    const float upa = ldexpf(1.0f, 14 - ea), upb = ldexpf(1.0f, 14 - eb);
    const float dna = ldexpf(scale * kL2e, ea - 14), dnb = ldexpf(scale * kL2e, eb - 14);
    uint32_t qh[kGmK / 16][4], ql[kGmK / 16][4];
#pragma unroll
    for (int kk = 0; kk < kGmK / 16; ++kk)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int cc = kk * 16 + 2 * tig + (r >> 1) * 8;
            float2 x = *reinterpret_cast<const float2*>(((r & 1) ? qb : qa) + cc);
            const float up = (r & 1) ? upb : upa;
            x.x *= up;
            x.y *= up;
            const __half2 hi = __floats2half2_rn(x.x, x.y);
            const float2 hf = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(x.x - hf.x, x.y - hf.y);
            qh[kk][r] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t*>(&lo);
        }

    const int nk = n_kv * kGmK, nv = n_kv * kGmV;
    auto issue = [&](int64_t j0, int st) {   // tile j0 .. j0 + KT - 1 into stage st; keys past the chunk are zeros
        for (int i = lane; i < KT * (kGmK / 8); i += 32) {
            const int j = i / (kGmK / 8), cc = i % (kGmK / 8);
            const int64_t key = j0 + j;
            const bool ok = key < k_hi;
            const int64_t r = ok ? (wrap ? key % rows : key) : 0;
            ma_cp16(&Ks[st][j][cc * 8], kc + (size_t) r * nk + (size_t) gk * kGmK + cc * 8, ok);
        }
        for (int i = lane; i < KT * (kGmV / 8); i += 32) {
            const int j = i / (kGmV / 8), cc = i % (kGmV / 8);
            const int64_t key = j0 + j;
            const bool ok = key < k_hi;
            const int64_t r = ok ? (wrap ? key % rows : key) : 0;
            ma_cp16(&Vs[st][j][cc * 8], vc + (size_t) r * nv + (size_t) gk * kGmV + cc * 8, ok);
        }
        ma_commit();
    };

    float acc[kGmV / 8][4];   // 2^14 x the chunk's p.v: n-tile j = dims 8 j .. 8 j + 7, rows a (0, 1) and b (2, 3)
#pragma unroll
    for (int j = 0; j < kGmV / 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.0f, l_b = 0.0f;
    const int n_ch = k_hi > k_lo ? (int) ((k_hi - k_lo + KT - 1) / KT) : 0;
    if (n_ch > 0) issue(k_lo, 0);
    for (int ci = 0; ci < n_ch; ++ci) {
        const int st = ci & 1;
        const int64_t j0 = k_lo + (int64_t) ci * KT;
        if (ci + 1 < n_ch) {
            issue(j0 + KT, st ^ 1);
            ma_wait<1>();
        } else {
            ma_wait<0>();
        }
        __syncwarp();   // every lane's copies of stage st are visible
        float s[KT / 8][4];
#pragma unroll
        for (int nt = 0; nt < KT / 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
            for (int kk2 = 0; kk2 < kGmK / 32; ++kk2) {
                uint32_t b[4];
                ma_ldsm(b, &Ks[st][nt * 8 + (lane & 7)][kk2 * 32 + (lane >> 3) * 8]);
                ma_mma(s[nt], qh[2 * kk2], b);
                ma_mma(s[nt], ql[2 * kk2], b);
                ma_mma(s[nt], qh[2 * kk2 + 1], b + 2);
                ma_mma(s[nt], ql[2 * kk2 + 1], b + 2);
            }
        }
        float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < KT / 8; ++nt)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int64_t key = j0 + nt * 8 + 2 * tig + e;
                s[nt][e] = key >= loa && key <= pa ? s[nt][e] * dna : -INFINITY;
                s[nt][2 + e] = key >= lob && key <= pb ? s[nt][2 + e] * dnb : -INFINITY;
                mxa = fmaxf(mxa, s[nt][e]);
                mxb = fmaxf(mxb, s[nt][2 + e]);
            }
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) {
            mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, o));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, o));
        }
        const float na = fmaxf(m_a, mxa), nb = fmaxf(m_b, mxb);
        const float ba = na == -INFINITY ? 0.0f : na, bb = nb == -INFINITY ? 0.0f : nb;
        const float aa = exp2f(m_a - ba), ab = exp2f(m_b - bb);
        m_a = na;
        m_b = nb;
        float sa = 0.0f, sb = 0.0f;
#pragma unroll
        for (int nt = 0; nt < KT / 8; ++nt)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                s[nt][e] = exp2f(s[nt][e] - ba);
                s[nt][2 + e] = exp2f(s[nt][2 + e] - bb);
                sa += s[nt][e];
                sb += s[nt][2 + e];
            }
        l_a = fmaf(l_a, aa, sa);
        l_b = fmaf(l_b, ab, sb);
#pragma unroll
        for (int j = 0; j < kGmV / 8; ++j) {
            acc[j][0] *= aa;
            acc[j][1] *= aa;
            acc[j][2] *= ab;
            acc[j][3] *= ab;
        }
#pragma unroll
        for (int ks = 0; ks < KT / 16; ++ks) {
            const float x[8] = {s[2 * ks][0] * 16384.0f,     s[2 * ks][1] * 16384.0f,     s[2 * ks][2] * 16384.0f,
                                s[2 * ks][3] * 16384.0f,     s[2 * ks + 1][0] * 16384.0f, s[2 * ks + 1][1] * 16384.0f,
                                s[2 * ks + 1][2] * 16384.0f, s[2 * ks + 1][3] * 16384.0f};
            uint32_t ah[4], al[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const __half2 hi = __floats2half2_rn(x[2 * r], x[2 * r + 1]);
                const float2 hf = __half22float2(hi);
                ah[r] = *reinterpret_cast<const uint32_t*>(&hi);
                al[r] = ma_pack(x[2 * r] - hf.x, x[2 * r + 1] - hf.y);
            }
            const int mq = lane >> 3;
#pragma unroll
            for (int j2 = 0; j2 < kGmV / 16; ++j2) {   // fresh accumulators + an FP32 add (see gqa_prefill_mma_kernel)
                uint32_t b[4];
                ma_ldsm_t(b, &Vs[st][ks * 16 + (lane & 7) + (mq & 1) * 8][(j2 * 2 + (mq >> 1)) * 8]);
                float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, d1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                ma_mma(d0, ah, b);
                ma_mma(d0, al, b);
                ma_mma(d1, ah, b + 2);
                ma_mma(d1, al, b + 2);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[2 * j2][i] += d0[i];
                    acc[2 * j2 + 1][i] += d1[i];
                }
            }
        }
        __syncwarp();   // the warp is done with stage st: the next issue may overwrite it
    }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
        l_a += __shfl_xor_sync(0xffffffffu, l_a, o);
        l_b += __shfl_xor_sync(0xffffffffu, l_b, o);
    }
    // the chunk's partials (gqa_attend_kernel's: acc, then m in the natural-log domain of the scaled score, then l)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int t = half ? tb : ta, h = half ? hb : ha;
        if (t >= T) continue;
        const float m2 = half ? m_b : m_a, l = half ? l_b : l_a;
        float* pr = partial + (((size_t) t * H + h) * n_chunks + c) * (kGmV + 2);
        constexpr float inv = 1.0f / 16384.0f;
#pragma unroll
        for (int j = 0; j < kGmV / 8; ++j)
            *reinterpret_cast<float2*>(pr + 8 * j + 2 * tig) = make_float2(acc[j][2 * half] * inv, acc[j][2 * half + 1] * inv);
        if (tig == 0) {
            pr[kGmV] = m2 == -INFINITY ? -INFINITY : m2 * kLn2;
            pr[kGmV + 1] = l;
        }
    }
    ma_wait<0>();
#else
    __trap();   // built for a card without mma.sync m16n8k16 f16 (sm_80): gqa_decode_mma refuses such a device
#endif
}

/// A 32-bit word at a 2-byte-aligned shared address (a Q8_0 block's quants sit 2 bytes past its scale).
__device__ __forceinline__ uint32_t lds_u32_b2(const uint8_t* p) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
    return (uint32_t) h[0] | ((uint32_t) h[1] << 16);
}

/// MiMo's DFlash drafter: a Q8_0 matrix (GGUF blocks, row-major, as uploaded) times up to 8 Q8_1 columns on int8 tensor
/// cores - mma.m16n8k32: 16 weight rows x 32 of K against the 8 columns per instruction, where MMVQ's dp4a loop turns
/// compute-bound past four columns (a draft step's projections: 8.1 ms at 8 columns against 3.2 at one).  One block = NW
/// warps over one 16-row tile, K split among them in whole groups of 8 Q8_0 blocks (272 bytes a row: 16-byte aligned when
/// n_in % 256 == 0), each group staged through shared memory by cp.async (double-buffered); per block of 32 the exact int32
/// products times the row's and the column's scales into FP32 (MMVQ's expression); the warps' partial sums joined in shared
/// memory.  Not bitwise MMVQ (another summation order): the drafter only - a verify is exact whatever its drafts are.
template <int NW, int MT, bool BS>
__global__ void __launch_bounds__(NW * 32) q8_mma_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ xq,
                                                          float* __restrict__ y, int n_in, int n_out, int ncols) {
#if STRATA_X_MMA
    constexpr int KB = 8, RB = KB * 34, R = 16 * MT;   // Q8_0 blocks a group, a row's bytes of one group, the block's rows
    constexpr int XB = KB * 36, XS = XB + 16;          // BS: a column's Q8_1 blocks of one group (288 bytes), padded
    __shared__ __align__(16) uint8_t sm[NW][2][R * RB + (BS ? 8 * XS : 0)];
    __shared__ float red[NW > 1 ? NW - 1 : 1][32][4 * MT];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int nb = n_in / 32, ng = nb / KB;
    const int g0 = (int) ((int64_t) ng * warp / NW), g1 = (int) ((int64_t) ng * (warp + 1) / NW);
    const size_t row_bytes = (size_t) nb * 34;
    const int row0 = blockIdx.x * R, nrows = n_out - row0 < R ? n_out - row0 : R;
    const uint8_t* wt = w + (size_t) row0 * row_bytes;
    auto issue = [&](int g, int st) {   // the block's R rows x one group (17 chunks of 16 bytes a row)
        for (int i = lane; i < R * (RB / 16); i += 32) {
            const int r = i / (RB / 16), c = i % (RB / 16);
            const bool ok = r < nrows;
            ma_cp16(&sm[warp][st][r * RB + c * 16], wt + (size_t) (ok ? r : 0) * row_bytes + (size_t) g * RB + c * 16, ok);
        }
        if (BS)   // and the 8 columns' Q8_1 blocks of the group (18 chunks a column; a column past ncols: zeros)
            for (int i = lane; i < 8 * (XB / 16); i += 32) {
                const int col = i / (XB / 16), c = i % (XB / 16);
                const bool ok = col < ncols;
                ma_cp16(&sm[warp][st][R * RB + col * XS + c * 16], xq + ((size_t) (ok ? col : 0) * nb + (size_t) g * KB) * 36 + c * 16, ok);
            }
        ma_commit();
    };
    // the B fragment's column is gid (its K bytes tig * 4 and 16 + tig * 4); the C fragment's columns are 2 tig, 2 tig + 1.
    // MT tiles of 16 rows share each B fragment (one load from L1 feeds MT products)
    const bool bok = gid < ncols;
    const uint8_t* xb_col = xq + (size_t) (bok ? gid : 0) * nb * 36;
    const int c0 = 2 * tig, c1 = 2 * tig + 1;
    const uint8_t* xd0 = xq + (size_t) (c0 < ncols ? c0 : 0) * nb * 36;
    const uint8_t* xd1 = xq + (size_t) (c1 < ncols ? c1 : 0) * nb * 36;
    float acc[MT][4];
#pragma unroll
    for (int t = 0; t < MT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;
    if (g0 < g1) issue(g0, 0);
    for (int g = g0; g < g1; ++g) {
        const int st = (g - g0) & 1;
        if (g + 1 < g1) {
            issue(g + 1, st ^ 1);
            ma_wait<1>();
        } else {
            ma_wait<0>();
        }
        __syncwarp();
        const uint8_t* sg = sm[warp][st];
#pragma unroll
        for (int b = 0; b < KB; ++b) {
            const int kb = g * KB + b;
            uint32_t b0, b1;
            float dx0, dx1;
            if (BS) {   // from the staged group (zeros past ncols)
                const uint8_t* sx = sg + R * RB;
                b0 = *reinterpret_cast<const uint32_t*>(sx + gid * XS + b * 36 + 4 + tig * 4);
                b1 = *reinterpret_cast<const uint32_t*>(sx + gid * XS + b * 36 + 20 + tig * 4);
                dx0 = __half2float(*reinterpret_cast<const __half*>(sx + c0 * XS + b * 36));
                dx1 = __half2float(*reinterpret_cast<const __half*>(sx + c1 * XS + b * 36));
            } else {
                const uint8_t* xb = xb_col + (size_t) kb * 36 + 4;
                b0 = bok ? *reinterpret_cast<const uint32_t*>(xb + tig * 4) : 0u;
                b1 = bok ? *reinterpret_cast<const uint32_t*>(xb + 16 + tig * 4) : 0u;
                dx0 = __half2float(*reinterpret_cast<const __half*>(xd0 + (size_t) kb * 36));
                dx1 = __half2float(*reinterpret_cast<const __half*>(xd1 + (size_t) kb * 36));
            }
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                const uint8_t* ra = sg + (t * 16 + gid) * RB + b * 34;
                const uint8_t* rb = ra + 8 * RB;
                const uint32_t a0 = lds_u32_b2(ra + 2 + tig * 4), a1 = lds_u32_b2(rb + 2 + tig * 4);
                const uint32_t a2 = lds_u32_b2(ra + 18 + tig * 4), a3 = lds_u32_b2(rb + 18 + tig * 4);
                const float dwa = __half2float(*reinterpret_cast<const __half*>(ra));
                const float dwb = __half2float(*reinterpret_cast<const __half*>(rb));
                int cc[4] = {0, 0, 0, 0};
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(cc[0]), "+r"(cc[1]), "+r"(cc[2]), "+r"(cc[3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                acc[t][0] += dwa * dx0 * (float) cc[0];
                acc[t][1] += dwa * dx1 * (float) cc[1];
                acc[t][2] += dwb * dx0 * (float) cc[2];
                acc[t][3] += dwb * dx1 * (float) cc[3];
            }
        }
        __syncwarp();
    }
    if (NW > 1) {   // split K: the other warps' partial sums into warp 0
        if (warp > 0)
#pragma unroll
            for (int t = 0; t < MT; ++t)
#pragma unroll
                for (int i = 0; i < 4; ++i) red[warp - 1][lane][4 * t + i] = acc[t][i];
        __syncthreads();
        if (warp > 0) return;
#pragma unroll
        for (int v = 0; v < NW - 1; ++v)
#pragma unroll
            for (int t = 0; t < MT; ++t)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[t][i] += red[v][lane][4 * t + i];
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        const int ra = t * 16 + gid, rb = ra + 8;
        if (c0 < ncols) {
            if (ra < nrows) y[(size_t) c0 * n_out + row0 + ra] = acc[t][0];
            if (rb < nrows) y[(size_t) c0 * n_out + row0 + rb] = acc[t][2];
        }
        if (c1 < ncols) {
            if (ra < nrows) y[(size_t) c1 * n_out + row0 + ra] = acc[t][1];
            if (rb < nrows) y[(size_t) c1 * n_out + row0 + rb] = acc[t][3];
        }
    }
#else
    __trap();   // built for a card without int8 mma.sync m16n8k32 (sm_80): q8_mma_w refuses such a device
#endif
}

/// The key chunk a full layer's decode row at position p takes (the shared kernel below): the smallest power of two >= KT whose
/// n_chunks chunks cover [0, p] - a function of p alone, so a row's partials are the same whatever window it is verified in.
__device__ __forceinline__ int64_t gqa_abs_chunk(int64_t p, int n_chunks, int kt) {
    int64_t S = kt;
    while (S * n_chunks < p + 1) S <<= 1;
    return S;
}

/// THE DECODE WINDOW'S GQA WITH SHARED K / V (MiMo's full layers, 16 heads a KV head): gqa_decode_mma_kernel's per-row math,
/// but one block = (key chunk c, KV head g) for EVERY row of the window - warp w is row w - so each K / V tile is read from
/// memory once per window instead of once per row (a 4-row verify window at 156K keys read the cache 4 times: ~3.6 GB a
/// window more than one row).  A row's keys are cut into chunks of gqa_abs_chunk(its position): absolute boundaries that
/// depend on the row alone, so the row's partials - and the merge - are bitwise those of the same row in a window of one; a
/// block covers a chunk of the window's last row (the largest size), and a row whose own chunks are half as long (the window
/// straddles a doubling) closes its partial at its own boundary inside it.  Writes partials in gqa_attend_kernel's layout,
/// joined by gqa_merge_abs_kernel.
template <int KT>
__global__ void __launch_bounds__(256) gqa_decode_shared_kernel(const float* __restrict__ q, int q_stride, const __half* __restrict__ kc,
                                                                const __half* __restrict__ vc, int64_t rows, int H, int n_kv,
                                                                float scale, float* __restrict__ partial, int n_chunks, int64_t pos0,
                                                                const int64_t* __restrict__ pos_dev, int T) {
#if STRATA_X_MMA
    static_assert(KT == 16 || KT == 32, "16 or 32 keys a tile");
    constexpr int G = 16;
    constexpr float kL2e = 1.4426950408889634f, kLn2 = 0.6931471805599453f;
    __shared__ __align__(16) __half Ks[2][KT][kGmKS];
    __shared__ __align__(16) __half Vs[2][KT][kGmVS];
    const int c = blockIdx.x, gk = blockIdx.y, t = threadIdx.y;
    const int lane = threadIdx.x, gid = lane >> 2, tig = lane & 3;
    const int nthr = 32 * T, tid = t * 32 + lane;
    const int64_t base = pos_dev ? *pos_dev : pos0;
    const int64_t p = base + t, p_last = base + T - 1;
    const int64_t S = gqa_abs_chunk(p, n_chunks, KT), Smax = gqa_abs_chunk(p_last, n_chunks, KT);
    const int s_log = __ffsll((long long) S) - 1;   // S is a power of two
    const int64_t k_lo = (int64_t) c * Smax;
    if (k_lo > p_last) return;   // past every row (uniform across the block)
    const int64_t k_hi = k_lo + Smax < p_last + 1 ? k_lo + Smax : p_last + 1;
    const bool wrap = p_last + 1 > rows;
    const int ha = gk * G + gid, hb = ha + 8;
    // q into A fragments (gqa_decode_mma_kernel's): this row's 16 heads, hi / lo of q * 2^(14 - e)
    const float* qa = q + (size_t) t * q_stride + (size_t) ha * kGmK;
    const float* qb = q + (size_t) t * q_stride + (size_t) hb * kGmK;
    float ma = 0.0f, mb = 0.0f;
#pragma unroll
    for (int kk = 0; kk < kGmK / 16; ++kk)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int cc = kk * 16 + 2 * tig + hf * 8;
            const float2 xa = *reinterpret_cast<const float2*>(qa + cc), xb = *reinterpret_cast<const float2*>(qb + cc);
            ma = fmaxf(ma, fmaxf(fabsf(xa.x), fabsf(xa.y)));
            mb = fmaxf(mb, fmaxf(fabsf(xb.x), fabsf(xb.y)));
        }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
        ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, o));
        mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, o));
    }
    int ea = 0, eb = 0;
    if (ma > 0.0f) frexpf(ma, &ea);
    if (mb > 0.0f) frexpf(mb, &eb);
    const float upa = ldexpf(1.0f, 14 - ea), upb = ldexpf(1.0f, 14 - eb);
    const float dna = ldexpf(scale * kL2e, ea - 14), dnb = ldexpf(scale * kL2e, eb - 14);
    uint32_t qh[kGmK / 16][4], ql[kGmK / 16][4];
#pragma unroll
    for (int kk = 0; kk < kGmK / 16; ++kk)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int cc = kk * 16 + 2 * tig + (r >> 1) * 8;
            float2 x = *reinterpret_cast<const float2*>(((r & 1) ? qb : qa) + cc);
            const float up = (r & 1) ? upb : upa;
            x.x *= up;
            x.y *= up;
            const __half2 hi = __floats2half2_rn(x.x, x.y);
            const float2 hf = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(x.x - hf.x, x.y - hf.y);
            qh[kk][r] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t*>(&lo);
        }
    const int nk = n_kv * kGmK, nv = n_kv * kGmV;
    auto issue = [&](int64_t j0, int st) {   // tile j0 .. j0 + KT - 1, by the whole block; keys past the chunk are zeros
        for (int i = tid; i < KT * (kGmK / 8); i += nthr) {
            const int j = i / (kGmK / 8), cc = i % (kGmK / 8);
            const int64_t key = j0 + j;
            const bool ok = key < k_hi;
            const int64_t r = ok ? (wrap ? key % rows : key) : 0;
            ma_cp16(&Ks[st][j][cc * 8], kc + (size_t) r * nk + (size_t) gk * kGmK + cc * 8, ok);
        }
        for (int i = tid; i < KT * (kGmV / 8); i += nthr) {
            const int j = i / (kGmV / 8), cc = i % (kGmV / 8);
            const int64_t key = j0 + j;
            const bool ok = key < k_hi;
            const int64_t r = ok ? (wrap ? key % rows : key) : 0;
            ma_cp16(&Vs[st][j][cc * 8], vc + (size_t) r * nv + (size_t) gk * kGmV + cc * 8, ok);
        }
        ma_commit();
    };
    float acc[kGmV / 8][4];
#pragma unroll
    for (int j = 0; j < kGmV / 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.0f, l_b = 0.0f;
    // the row's partial of chunk `ci` (gqa_attend_kernel's: acc, m in the natural-log domain of the scaled score, l), then
    // fresh accumulators for its next chunk
    auto flush = [&](int64_t ci) {
        float la = l_a, lb = l_b;
#pragma unroll
        for (int o = 1; o < 4; o <<= 1) {
            la += __shfl_xor_sync(0xffffffffu, la, o);
            lb += __shfl_xor_sync(0xffffffffu, lb, o);
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int h = half ? hb : ha;
            const float m2 = half ? m_b : m_a, l = half ? lb : la;
            float* pr = partial + (((size_t) t * H + h) * n_chunks + ci) * (kGmV + 2);
            constexpr float inv = 1.0f / 16384.0f;
#pragma unroll
            for (int j = 0; j < kGmV / 8; ++j)
                *reinterpret_cast<float2*>(pr + 8 * j + 2 * tig) = make_float2(acc[j][2 * half] * inv, acc[j][2 * half + 1] * inv);
            if (tig == 0) {
                pr[kGmV] = m2 == -INFINITY ? -INFINITY : m2 * kLn2;
                pr[kGmV + 1] = l;
            }
        }
#pragma unroll
        for (int j = 0; j < kGmV / 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
        m_a = m_b = -INFINITY;
        l_a = l_b = 0.0f;
    };
    const int n_ch = (int) ((k_hi - k_lo + KT - 1) / KT);
    issue(k_lo, 0);
    for (int ci = 0; ci < n_ch; ++ci) {
        const int st = ci & 1;
        const int64_t j0 = k_lo + (int64_t) ci * KT;
        if (ci + 1 < n_ch) {
            issue(j0 + KT, st ^ 1);
            ma_wait<1>();
        } else {
            ma_wait<0>();
        }
        __syncthreads();   // every thread's copies of stage st are visible
        if (j0 <= p) {     // a tile past this row: nothing of its keys is the row's
            float s[KT / 8][4];
#pragma unroll
            for (int nt = 0; nt < KT / 8; ++nt) {
                s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
#pragma unroll
                for (int kk2 = 0; kk2 < kGmK / 32; ++kk2) {
                    uint32_t b[4];
                    ma_ldsm(b, &Ks[st][nt * 8 + (lane & 7)][kk2 * 32 + (lane >> 3) * 8]);
                    ma_mma(s[nt], qh[2 * kk2], b);
                    ma_mma(s[nt], ql[2 * kk2], b);
                    ma_mma(s[nt], qh[2 * kk2 + 1], b + 2);
                    ma_mma(s[nt], ql[2 * kk2 + 1], b + 2);
                }
            }
            float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < KT / 8; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int64_t key = j0 + nt * 8 + 2 * tig + e;
                    s[nt][e] = key <= p ? s[nt][e] * dna : -INFINITY;
                    s[nt][2 + e] = key <= p ? s[nt][2 + e] * dnb : -INFINITY;
                    mxa = fmaxf(mxa, s[nt][e]);
                    mxb = fmaxf(mxb, s[nt][2 + e]);
                }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, o));
                mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, o));
            }
            const float na = fmaxf(m_a, mxa), nb = fmaxf(m_b, mxb);
            const float ba = na == -INFINITY ? 0.0f : na, bb = nb == -INFINITY ? 0.0f : nb;
            const float aa = exp2f(m_a - ba), ab = exp2f(m_b - bb);
            m_a = na;
            m_b = nb;
            float sa = 0.0f, sb = 0.0f;
#pragma unroll
            for (int nt = 0; nt < KT / 8; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    s[nt][e] = exp2f(s[nt][e] - ba);
                    s[nt][2 + e] = exp2f(s[nt][2 + e] - bb);
                    sa += s[nt][e];
                    sb += s[nt][2 + e];
                }
            l_a = fmaf(l_a, aa, sa);
            l_b = fmaf(l_b, ab, sb);
#pragma unroll
            for (int j = 0; j < kGmV / 8; ++j) {
                acc[j][0] *= aa;
                acc[j][1] *= aa;
                acc[j][2] *= ab;
                acc[j][3] *= ab;
            }
#pragma unroll
            for (int ks = 0; ks < KT / 16; ++ks) {
                const float x[8] = {s[2 * ks][0] * 16384.0f,     s[2 * ks][1] * 16384.0f,     s[2 * ks][2] * 16384.0f,
                                    s[2 * ks][3] * 16384.0f,     s[2 * ks + 1][0] * 16384.0f, s[2 * ks + 1][1] * 16384.0f,
                                    s[2 * ks + 1][2] * 16384.0f, s[2 * ks + 1][3] * 16384.0f};
                uint32_t ah[4], al[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const __half2 hi = __floats2half2_rn(x[2 * r], x[2 * r + 1]);
                    const float2 hf = __half22float2(hi);
                    ah[r] = *reinterpret_cast<const uint32_t*>(&hi);
                    al[r] = ma_pack(x[2 * r] - hf.x, x[2 * r + 1] - hf.y);
                }
                const int mq = lane >> 3;
#pragma unroll
                for (int j2 = 0; j2 < kGmV / 16; ++j2) {   // fresh accumulators + an FP32 add (see gqa_prefill_mma_kernel)
                    uint32_t b[4];
                    ma_ldsm_t(b, &Vs[st][ks * 16 + (lane & 7) + (mq & 1) * 8][(j2 * 2 + (mq >> 1)) * 8]);
                    float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, d1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    ma_mma(d0, ah, b);
                    ma_mma(d0, al, b);
                    ma_mma(d1, ah, b + 2);
                    ma_mma(d1, al, b + 2);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        acc[2 * j2][i] += d0[i];
                        acc[2 * j2 + 1][i] += d1[i];
                    }
                }
            }
            // the end of one of this row's chunks (its own size, a power of two), or of its keys
            if (((j0 + KT) & (S - 1)) == 0 || j0 + KT > p) flush(j0 >> s_log);
        }
        __syncthreads();   // every warp is done with stage st: the next issue may overwrite it
    }
#else
    __trap();   // built for a card without mma.sync m16n8k16 f16 (sm_80)
#endif
}

/// gqa_merge_kernel for gqa_decode_shared_kernel's partials: row t joins its own chunks only (gqa_abs_chunk of its position).
template <int DV>
__global__ void gqa_merge_abs_kernel(const float* __restrict__ partial, const float* __restrict__ sinks, float* __restrict__ out, int H,
                                     int n_chunks, int64_t pos0, const int64_t* __restrict__ pos_dev, int kt) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int64_t p = (pos_dev ? *pos_dev : pos0) + t;
    const int64_t S = gqa_abs_chunk(p, n_chunks, kt);
    const int nc = (int) ((p + S) / S);   // chunks [0, nc): the last one holds key p
    const float* pr = partial + ((size_t) t * H + h) * n_chunks * (DV + 2);
    float M = sinks ? sinks[h] : -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pr[(size_t) c * (DV + 2) + DV]);
    float L = sinks ? expf(sinks[h] - M) : 0.0f, a = 0.0f;
    for (int c = 0; c < nc; ++c) {
        const float* pc = pr + (size_t) c * (DV + 2);
        if (pc[DV] == -INFINITY) continue;
        const float f = expf(pc[DV] - M);
        L += pc[DV + 1] * f;
        a += pc[d] * f;
    }
    out[((size_t) t * H + h) * DV + d] = L > 0.0f ? a / L : 0.0f;
}

}  // namespace

void rope_neox_w(float* x, int row_stride, int head_off, int n_heads, int head_dim, int n_rot, float base, int64_t pos0, int T,
                 void* stream, const int64_t* pos_dev) {
    if (T <= 0) return;
    const float theta_scale = powf(base, -2.0f / (float) n_rot);   // ggml_cuda_op_rope_impl's, on the host
    const int pairs = n_heads * (n_rot / 2);
    rope_neox_kernel<<<dim3(std::max(1, std::min(64, (pairs + 255) / 256)), T), 256, 0, st(stream)>>>(
        x, row_stride, head_off, n_heads, head_dim, n_rot, theta_scale, pos0, pos_dev);
    check_launch("rope_neox_w");
}

void kv_store_w(const float* src, int row_stride, int k_off, int v_off, uint16_t* kc, uint16_t* vc, int64_t rows, int n_kv, int dk,
                int dv, int64_t pos0, int T, void* stream, const int64_t* pos_dev) {
    if (T <= 0) return;
    const int nk = n_kv * dk, nv = n_kv * dv;
    kv_store_kernel<<<dim3(std::max(1, std::min(16, (nk + nv + 255) / 256)), T), 256, 0, st(stream)>>>(
        src, row_stride, k_off, v_off, reinterpret_cast<__half*>(kc), reinterpret_cast<__half*>(vc), rows, nk, nv, pos0, pos_dev);
    check_launch("kv_store_w");
}

bool q8_mma_w(const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream, int warps) {
    if (ncols < 1 || ncols > 8 || n_in % 256 != 0 || n_out < 1 || (reinterpret_cast<uintptr_t>(w) & 15) != 0) return false;
    static int cc[64] = {};   // the device's compute capability major, once
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (cc[dev] == 0) cudaDeviceGetAttribute(&cc[dev], cudaDevAttrComputeCapabilityMajor, dev);
    if (cc[dev] < 8) return false;
    // the shape: warps a block (K split among them) x 16-row tiles a warp.  `warps` (bench): 1 / 2 / 4 = (w, 1); 12 / 14 / 22 =
    // (warps 1, tiles 2), (1, 4), (2, 2); 0: by shape (measured on the drafter's matrices, sm_86)
    const int tiles = (n_out + 15) / 16, groups = n_in / 256;
    // STRATA_X_Q8MMA_BS=1: the activations staged in shared memory with the weights (measured no faster: off)
    static const bool bs_env = [] { const char* e = std::getenv("STRATA_X_Q8MMA_BS"); return e != nullptr && e[0] == '1'; }();
    bool bs = bs_env;
    if (warps >= 100) {   // bench: + 100 = the activations through L1 (not staged)
        bs = false;
        warps -= 100;
    }
    int nw = 0, mt = 1;
    switch (warps) {
        case 1: case 2: case 4: nw = warps; break;
        case 12: nw = 1; mt = 2; break;
        case 14: nw = 1; mt = 4; break;
        case 22: nw = 2; mt = 2; break;
        default:   // measured (x_kernels_parity dflash_bench, 8 columns): the head 1x4, gate/up 2x1, wq 1x1, wo/down/fc 2x2, wk/wv 4x1
            if (tiles >= 4096) { nw = 1; mt = 4; }
            else if (tiles >= 1024) { nw = 2; mt = 1; }
            else if (tiles >= 512) { nw = 1; mt = 1; }
            else if (n_in >= 8192) { nw = 2; mt = 2; }
            else { nw = 4; mt = 1; }
    }
    while (nw > 1 && groups < 2 * nw) nw /= 2;
    const auto* wb = static_cast<const uint8_t*>(w);
    const auto* xb = static_cast<const uint8_t*>(x_q8_1);
    const unsigned blocks = (unsigned) ((n_out + 16 * mt - 1) / (16 * mt));
    auto go = [&](auto kern, int threads) { kern<<<blocks, threads, 0, st(stream)>>>(wb, xb, y, n_in, n_out, ncols); };
    if (bs) {
        if (mt == 4) go(q8_mma_kernel<1, 4, true>, 32);
        else if (mt == 2 && nw == 2) go(q8_mma_kernel<2, 2, true>, 64);
        else if (mt == 2) go(q8_mma_kernel<1, 2, true>, 32);
        else if (nw == 4) go(q8_mma_kernel<4, 1, false>, 128);   // staged: 54 KB, past the static limit
        else if (nw == 2) go(q8_mma_kernel<2, 1, true>, 64);
        else go(q8_mma_kernel<1, 1, true>, 32);
    } else {
        if (mt == 4) go(q8_mma_kernel<1, 4, false>, 32);
        else if (mt == 2 && nw == 2) go(q8_mma_kernel<2, 2, false>, 64);
        else if (mt == 2) go(q8_mma_kernel<1, 2, false>, 32);
        else if (nw == 4) go(q8_mma_kernel<4, 1, false>, 128);
        else if (nw == 2) go(q8_mma_kernel<2, 1, false>, 64);
        else go(q8_mma_kernel<1, 1, false>, 32);
    }
    check_launch("q8_mma_w");
    return true;
}

size_t gqa_attend_scratch_bytes(int H, int dv, int n_chunks, int T) {
    return (size_t) std::max(T, 1) * H * std::max(n_chunks, 1) * (dv + 2) * sizeof(float);
}

bool gqa_prefill_mma(const float* q, int q_stride, const uint16_t* kc, const uint16_t* vc, int64_t rows, int H, int n_kv, int dk,
                     int dv, int window, const float* sinks, float scale, float* out, int64_t pos0, int T, void* stream, int warps) {
    if (T <= 0 || dk != kGmK || dv != kGmV || n_kv <= 0 || H % n_kv != 0 || (H / n_kv != 8 && H / n_kv != 16)) return false;
    static int cc[64] = {};   // the device's compute capability major, once
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (cc[dev] == 0) cudaDeviceGetAttribute(&cc[dev], cudaDevAttrComputeCapabilityMajor, dev);
    if (cc[dev] < 8) return false;
    static const int env_w = [] { const char* e = std::getenv("STRATA_X_GQA_MMA_WARPS"); return e && e[0] == '4' ? 4 : 8; }();
    const int nw = warps == 4 || warps == 8 ? warps : env_w;
    const __half* k = reinterpret_cast<const __half*>(kc);
    const __half* v = reinterpret_cast<const __half*>(vc);
    const int G = H / n_kv, bq = nw * 16 / G;
    auto go = [&](auto kern) {
        kern<<<dim3((T + bq - 1) / bq, n_kv), nw * 32, 0, st(stream)>>>(q, q_stride, k, v, rows, H, n_kv, window, sinks, scale, out,
                                                                        pos0, T);
    };
    if (G == 16) go(nw == 8 ? gqa_prefill_mma_kernel<16, 8> : gqa_prefill_mma_kernel<16, 4>);
    else go(nw == 8 ? gqa_prefill_mma_kernel<8, 8> : gqa_prefill_mma_kernel<8, 4>);
    check_launch("gqa_prefill_mma");
    return true;
}

void gqa_merge_w(const float* partial, const float* sinks, float* out, int H, int n_chunks, int T, void* stream) {
    if (T <= 0) return;
    gqa_merge_kernel<128><<<dim3(H, T), 128, 0, st(stream)>>>(partial, sinks, out, H, n_chunks);
    check_launch("gqa_merge_w");
}

bool gqa_decode_mma(const float* q, int q_stride, const uint16_t* kc, const uint16_t* vc, int64_t rows, int H, int n_kv, int dk, int dv,
                    int window, float scale, float* partial, int n_chunks, int64_t pos0, int T, void* stream, const int64_t* pos_dev,
                    int kt) {
    if (T <= 0 || dk != kGmK || dv != kGmV || n_kv <= 0 || H % n_kv != 0 || (H / n_kv != 8 && H / n_kv != 16) || n_chunks < 2 ||
        !partial)
        return false;
    static int cc[64] = {};   // the device's compute capability major, once
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (cc[dev] == 0) cudaDeviceGetAttribute(&cc[dev], cudaDevAttrComputeCapabilityMajor, dev);
    if (cc[dev] < 8) return false;
    const __half* k = reinterpret_cast<const __half*>(kc);
    const __half* v = reinterpret_cast<const __half*>(vc);
    const int G = H / n_kv, tw = 16 / G;
    const dim3 grid(n_chunks, n_kv, (T + tw - 1) / tw);
    // the tile: 16 keys (21.5 KB of double-buffered K / V a warp: 4 warps an SM, every tensor core busy) or 32 (43 KB: 2 an SM)
    static const int env_kt = [] { const char* e = std::getenv("STRATA_X_GQA_DECODE_KT"); return e && std::atoi(e) == 32 ? 32 : 16; }();
    const int t = kt == 16 || kt == 32 ? kt : env_kt;
    auto go = [&](auto kern) {
        kern<<<grid, 32, 0, st(stream)>>>(q, q_stride, k, v, rows, H, n_kv, window, scale, partial, n_chunks, pos0, pos_dev, T);
    };
    if (G == 16) go(t == 16 ? gqa_decode_mma_kernel<16, 16> : gqa_decode_mma_kernel<16, 32>);
    else go(t == 16 ? gqa_decode_mma_kernel<8, 16> : gqa_decode_mma_kernel<8, 32>);
    check_launch("gqa_decode_mma");
    return true;
}

bool gqa_decode_shared(const float* q, int q_stride, const uint16_t* kc, const uint16_t* vc, int64_t rows, int H, int n_kv, int dk,
                       int dv, const float* sinks, float scale, float* out, float* partial, int n_chunks, int64_t pos0, int T, void* stream,
                       const int64_t* pos_dev, int max_pos) {
    if (T < 1 || T > 8 || dk != kGmK || dv != kGmV || n_kv <= 0 || H != 16 * n_kv || n_chunks < 1 || !partial) return false;
    static int cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (cc[dev] == 0) cudaDeviceGetAttribute(&cc[dev], cudaDevAttrComputeCapabilityMajor, dev);
    if (cc[dev] < 8) return false;
    constexpr int kt = 16;
    (void) max_pos;
    // n_chunks blocks a KV head: a row's chunks never number more (gqa_abs_chunk), at any position a graph replays with; the
    // blocks past the window's last chunk return at once
    const int64_t n_super = n_chunks;
    const __half* k = reinterpret_cast<const __half*>(kc);
    const __half* v = reinterpret_cast<const __half*>(vc);
    gqa_decode_shared_kernel<kt><<<dim3((unsigned) n_super, n_kv), dim3(32, T), 0, st(stream)>>>(q, q_stride, k, v, rows, H, n_kv, scale,
                                                                                               partial, n_chunks, pos0, pos_dev, T);
    gqa_merge_abs_kernel<128><<<dim3(H, T), 128, 0, st(stream)>>>(partial, sinks, out, H, n_chunks, pos0, pos_dev, kt);
    check_launch("gqa_decode_shared");
    return true;
}

void gqa_attend_w(const float* q, int q_stride, const uint16_t* kc, const uint16_t* vc, int64_t rows, int H, int n_kv, int dk, int dv,
                  int window, const float* sinks, float scale, float* out, float* partial, int n_chunks, int64_t pos0, int T,
                  void* stream, const int64_t* pos_dev, bool noncausal) {
    if (T <= 0) return;
    const int G = H / n_kv;
    if (H % n_kv != 0 || G % 4 != 0 || G > 16) throw std::runtime_error("gqa_attend_w: 4, 8, 12 or 16 query heads per KV head");
    if (n_chunks < 1 || (n_chunks > 1 && !partial)) throw std::runtime_error("gqa_attend_w: split-K needs the partials");
    const __half* k = reinterpret_cast<const __half*>(kc);
    const __half* v = reinterpret_cast<const __half*>(vc);
    static const bool tiled = !(std::getenv("STRATA_X_GQA_TILED") && std::getenv("STRATA_X_GQA_TILED")[0] == '0');
    // the prompt path on tensor cores (STRATA_X_GQA_MMA=0: the FP32 tiles below)
    static const bool mma = !(std::getenv("STRATA_X_GQA_MMA") && std::getenv("STRATA_X_GQA_MMA")[0] == '0');
    if (n_chunks == 1 && T > 8 && mma && !pos_dev && !noncausal &&
        gqa_prefill_mma(q, q_stride, kc, vc, rows, H, n_kv, dk, dv, window, sinks, scale, out, pos0, T, stream))
        return;
    // a decode window's full layers (keys split) on tensor cores too (STRATA_X_GQA_DECODE_MMA=0: the FP32 kernel below); the
    // sliding-window layers' 128 keys stay there
    static const bool mma_dec = !(std::getenv("STRATA_X_GQA_DECODE_MMA") && std::getenv("STRATA_X_GQA_DECODE_MMA")[0] == '0');
    if (n_chunks > 1 && window == 0 && mma && mma_dec && !noncausal &&
        gqa_decode_mma(q, q_stride, kc, vc, rows, H, n_kv, dk, dv, window, scale, partial, n_chunks, pos0, T, stream, pos_dev)) {
        gqa_merge_kernel<128><<<dim3(H, T), 128, 0, st(stream)>>>(partial, sinks, out, H, n_chunks);
        check_launch("gqa_attend_w merge");
        return;
    }
    if (n_chunks == 1 && T > 8 && tiled && !noncausal && dk == 192 && dv == 128 && (G == 8 || G == 16)) {   // the prompt path: tokens per block
        // F32 queries in shared memory: 4 tokens of 16 heads (the full layers) or 8 of 8 (the sliding-window ones) fit sm_86's 99 KB
        auto launch = [&](auto kern, int g, int bq) {
            const size_t smem = (size_t) bq * g * 192 * 4 + (size_t) kGqaKT * (192 + 2) * 2 + (size_t) kGqaKT * (128 + 4) * 2 +
                                (size_t) bq * kGqaKT * g * 4;
            // the opt-in above 48 KB is the CURRENT device's: set it every launch (both GPUs run this kernel)
            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
            kern<<<dim3((T + bq - 1) / bq, n_kv), 32 * bq, smem, st(stream)>>>(q, q_stride, k, v, rows, H, n_kv, window, sinks, scale, out,
                                                                                pos0, T);
        };
        if (G == 16) launch(gqa_prefill_kernel<192, 128, 16, 4>, 16, 4);
        else launch(gqa_prefill_kernel<192, 128, 8, 8>, 8, 8);
        check_launch("gqa_attend_w (prompt tiles)");
        return;
    }
    const dim3 grid(n_chunks, n_kv, T);
    if (dk == 192 && dv == 128)
        gqa_attend_kernel<192, 128><<<grid, 128, 0, st(stream)>>>(q, q_stride, k, v, rows, H, n_kv, window, sinks, scale, out,
                                                                 partial, n_chunks, pos0, pos_dev, noncausal ? 1 : 0);
    else if (dk == 128 && dv == 128)
        gqa_attend_kernel<128, 128><<<grid, 128, 0, st(stream)>>>(q, q_stride, k, v, rows, H, n_kv, window, sinks, scale, out,
                                                                 partial, n_chunks, pos0, pos_dev, noncausal ? 1 : 0);
    else
        throw std::runtime_error("gqa_attend_w: head dims " + std::to_string(dk) + " / " + std::to_string(dv) + " have no kernel");
    check_launch("gqa_attend_w");
    if (n_chunks > 1) {
        gqa_merge_kernel<128><<<dim3(H, T), 128, 0, st(stream)>>>(partial, sinks, out, H, n_chunks);
        check_launch("gqa_attend_w merge");
    }
}

}  // namespace strata::x
