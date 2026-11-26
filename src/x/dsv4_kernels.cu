// src/x/dsv4_kernels.cu - moex: DeepSeek-V4-Flash attention-side kernels.  See include/strata/x/dsv4_kernels.hpp;
// the arithmetic of each is its DSV4_SPEC section as llama.cpp computes it, and each kernel's comment names the
// formulation it follows (ggml-cuda's where the backends differ; line references are to mainline ~/AI/llama.cpp).
//
// Built WITHOUT -use_fast_math (ggml-cuda is built with it, ggml/src/ggml-cuda/CMakeLists.txt:195): expf, sincosf, logf
// and the divisions here are the precise ones.  At position p, __sinf's argument reduction alone costs ggml-cuda about
// p * 2^-22 rad on the highest-frequency rope pairs; these kernels and ggml-cpu do not have that error.
#include "strata/x/dsv4_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
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
        fprintf(stderr, "dsv4_kernels: %s: %s\n", what, cudaGetErrorString(e));
        throw std::runtime_error(what);
    }
}

inline int blocks(int64_t n, int bs) { return (int) ((n + bs - 1) / bs); }

// POSITIONS FROM DEVICE MEMORY (graphs): `pos_ptr ? *pos_ptr : pos0`, as kernels.cu's window kernels.
__device__ __forceinline__ int64_t pos_of(const int64_t* __restrict__ p, int64_t arg) { return p ? *p : arg; }
__device__ __forceinline__ int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// moex (DSV41_BACKLOG #104): DeepSeek-V4.1's packed cache rows (dsv41_kernels.hpp dsv41_pack_row_bytes) - E2M1 codes two a
// byte, then the scales.  A value decodes to code x scale in F32 (exact: two and four significant bits) and then to F16 as
// dsv41_store_rows rounded it, so a reader gets the F16 cache's halves bit for bit.
constexpr int kPackComp = 288;   ///< a compressed row's bytes: 256 of codes, 32 E4M3 scales (one per 16 values)
constexpr int kPackLid = 80;     ///< an index key's: 64 of codes, 4 UE8M0 scales (one per 32), 12 of padding

__device__ __forceinline__ float e2m1_val(uint32_t c) {
    const uint32_t m = c & 7u;
    const float a = m < 4u ? 0.5f * (float) m : m == 7u ? 6.0f : (float) (m - 2u);
    return (c & 8u) ? -a : a;
}
__device__ __forceinline__ float e4m3_val(uint32_t b) {
    __nv_fp8_e4m3 f;
    f.__x = (__nv_fp8_storage_t) b;
    return (float) f;
}
__device__ __forceinline__ float ue8m0_val(uint32_t b) { return __uint_as_float(b << 23); }
/// The halves of the 8 values whose codes are the nibbles of `codes` (value j at bits 4j) and whose scale is `sc`
__device__ __forceinline__ uint4 unpack8(uint32_t codes, float sc) {
    uint32_t w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __half2 h = __halves2half2(__float2half_rn(e2m1_val((codes >> (8 * j)) & 15u) * sc),
                                         __float2half_rn(e2m1_val((codes >> (8 * j + 4)) & 15u) * sc));
        w[j] = *reinterpret_cast<const uint32_t*>(&h);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}
/// A compressed row's values [d, d + 8) (d % 8 == 0) as 8 halves
__device__ __forceinline__ uint4 comp_unpack8(const uint8_t* __restrict__ row, int d) {
    return unpack8(*reinterpret_cast<const uint32_t*>(row + d / 2), e4m3_val(row[256 + d / 16]));
}
/// An index key's values [d, d + 8)
__device__ __forceinline__ uint4 lid_unpack8(const uint8_t* __restrict__ row, int d) {
    return unpack8(*reinterpret_cast<const uint32_t*>(row + d / 2), ue8m0_val(row[64 + d / 32]));
}

/// c ? a : b as one selp on registers: written as `c ? arr[i] : arr[j]` the compiler folds the select into an indexed
/// load and moves the whole array to local memory.
__device__ __forceinline__ float sel_f(bool c, float a, float b) {
    float r;
    asm("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\tselp.f32 %0, %1, %2, p;\n\t}"
        : "=f"(r)
        : "f"(a), "f"(b), "r"((unsigned) c));
    return r;
}

/// One stage of the warp transpose-reduce of 2 * OFF values per lane: the lane keeps the half its bit OFF selects and
/// adds the partner's copy of that half (OFF a template argument: constant indices keep `v` in registers).
template <int OFF>
__device__ __forceinline__ void transpose_stage(float* v, int lane) {
    const bool up = (lane & OFF) != 0;
#pragma unroll
    for (int j = 0; j < OFF; ++j) {
        const float send = sel_f(up, v[j], v[j + OFF]);
        const float keep = sel_f(up, v[j + OFF], v[j]);
        v[j] = keep + __shfl_xor_sync(0xffffffffu, send, OFF);
    }
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

constexpr int kRopeDims = 64;            ///< n_rot: every DSV4 rope is 64 dims (32 pairs)
constexpr int kRopePairs = kRopeDims / 2;

// ------------------------------------------------------------------ RoPE

/// The per-call constants of ggml's RoPE, worked out on the host exactly as ggml does: theta_scale = powf(base, -2/64)
/// and the YaRN correction dims (ggml_rope_yarn_corr_dims) on the host in both backends; the per-pair factor
/// powf(theta_scale, k) that ggml-cuda evaluates per thread (rope.cu:107) is evaluated here once, with glibc's powf;
/// the ramp (rope.cu:15-18) and the magnitude (rope.cu:34) are the same float operations done once.
struct RopeK {
    float freq[kRopePairs];   ///< powf(theta_scale, k)
    float mix[kRopePairs];    ///< rope_yarn_ramp(c0, c1, 2k) * ext_factor
    float fs;                 ///< freq_scale
    float msc;                ///< attn_factor * (1 + 0.1 logf(1 / fs)) with YaRN, attn_factor without
    float sin_sign;           ///< -1: ggml_rope_ext_back (rope.cu:37-39)
    int yarn;                 ///< ext_factor != 0
};

/// ggml.c:4418-4430, transcribed (MAX/MIN are ggml's ternaries).
void corr_dims(int n_dims, int n_ctx_orig, float base, float beta_fast, float beta_slow, float dims[2]) {
    auto corr_dim = [&](float n_rot) {
        return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float) M_PI)) / (2 * logf(base));
    };
    const float start = floorf(corr_dim(beta_fast));
    const float end = ceilf(corr_dim(beta_slow));
    dims[0] = 0 > start ? 0 : start;
    dims[1] = (n_dims - 1) < end ? (float) (n_dims - 1) : end;
}

RopeK rope_consts(const Dsv4Rope& p, bool inverse) {
    RopeK k{};
    const float theta_scale = powf(p.freq_base, -2.0f / kRopeDims);
    float cd[2];
    corr_dims(kRopeDims, p.n_ctx_orig, p.freq_base, p.beta_fast, p.beta_slow, cd);
    for (int i = 0; i < kRopePairs; ++i) {
        k.freq[i] = powf(theta_scale, (float) (2 * i) / 2.0f);
        const float y = ((float) i - cd[0]) / std::max(0.001f, cd[1] - cd[0]);
        k.mix[i] = (1.0f - std::min(1.0f, std::max(0.0f, y))) * p.ext_factor;
    }
    k.fs = p.freq_scale;
    k.yarn = p.ext_factor != 0.0f;
    k.msc = k.yarn ? p.attn_factor * (1.0f + 0.1f * logf(1.0f / p.freq_scale)) : p.attn_factor;
    k.sin_sign = inverse ? -1.0f : 1.0f;
    return k;
}

/// cos/sin of pair k at position `pos` (rope.cu:23-40 with theta_base = pos * freq[k], rope.cu:107).
__device__ __forceinline__ void rope_cs(const RopeK& rk, float pos, int k, float& c, float& s) {
    const float te = pos * rk.freq[k];
    const float ti = rk.fs * te;
    float th = ti;
    if (rk.yarn) {
        const float mix = rk.mix[k];
        th = ti * (1 - mix) + te * mix;
    }
    float sn, cs;
    sincosf(th, &sn, &cs);
    c = cs * rk.msc;
    s = sn * rk.msc;
    s *= rk.sin_sign;
}

/// One thread per pair: in place the 32 rotated pairs of each row, out of place every pair of the row (the others
/// copied).  x/y may alias (no __restrict__).
__global__ void rope_kernel(const float* x, float* y, int64_t n_rows, int rows, int row_len, int offs, int pair0,
                            int ppr, RopeK rk, int64_t pos0_arg, const int64_t* __restrict__ pos_ptr, int64_t shift) {
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_rows * ppr) return;
    const int64_t r = gid / ppr;
    const int i = pair0 + (int) (gid % ppr);
    const float2 v = reinterpret_cast<const float2*>(x + r * row_len)[i];
    float2* o = reinterpret_cast<float2*>(y + r * row_len) + i;
    const int k = i - offs / 2;
    if (k < 0 || k >= kRopePairs) {
        *o = v;
        return;
    }
    const float pos = (float) (pos_of(pos_ptr, pos0_arg) + shift + r / rows);
    float c, s;
    rope_cs(rk, pos, k, c, s);
    // rope.cu:119, (x c - y s, x s + y c): moex (the ERNIE-5.5 hill-climb) spelled as the SASS this kernel compiled to before (ptxas
    // fused both: FFMA x c - (y s), FFMA y c + (x s), each with the other product rounded) so norm_rope_kernel's copy rounds the same
    *o = make_float2(__fmaf_rn(v.x, c, -__fmul_rn(v.y, s)), __fmaf_rn(v.y, c, __fmul_rn(v.x, s)));
}

// ------------------------------------------------------------------ Hadamard

/// fwht.cu's fwht_cuda transcribed: a warp per row, element i * 32 + lane in reg[i]; the scale first, then the
/// butterflies in ascending stride - lane stages by shuffle, register stages in place.  The same operations in the
/// same order as ggml-cpu's FWHT (ops.cpp:11990-12058): bitwise equal.
template <int N>
__global__ void __launch_bounds__(128) fwht_kernel(const float* x, float* y, int64_t rows, float scale) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= rows) return;
    constexpr int EL = N / 32;
    const int lane = threadIdx.x;
    float reg[EL];
#pragma unroll
    for (int i = 0; i < EL; ++i) reg[i] = x[r * N + i * 32 + lane] * scale;
#pragma unroll
    for (int h = 1; h < 32; h *= 2) {
#pragma unroll
        for (int j = 0; j < EL; ++j) {
            const float v = reg[j];
            const float v2 = __shfl_xor_sync(0xffffffffu, v, h);
            reg[j] = (lane & h) == 0 ? v + v2 : v2 - v;
        }
    }
#pragma unroll
    for (int h = 32; h < N; h *= 2) {
        const int step = h / 32;
#pragma unroll
        for (int j = 0; j < EL; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float a = reg[j + k], b = reg[j + k + step];
                reg[j + k] = a + b;
                reg[j + k + step] = a - b;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < EL; ++i) y[r * N + i * 32 + lane] = reg[i];
}

// ------------------------------------------------------------------ raw ring

constexpr int kD = 512;   ///< the head dim / KV width

__global__ void raw_store_kernel(const float* __restrict__ kv, __half* __restrict__ ring, int ring_rows,
                                 int64_t pos0_arg, const int64_t* __restrict__ pos_ptr) {
    const int t = blockIdx.x;
    const int64_t p = pos_of(pos_ptr, pos0_arg) + t;
    const float2* src = reinterpret_cast<const float2*>(kv + (size_t) t * kD);
    half2* dst = reinterpret_cast<half2*>(ring + (size_t) (p % ring_rows) * kD);
    for (int i = threadIdx.x; i < kD / 2; i += blockDim.x) dst[i] = __float22half2_rn(src[i]);   // set_rows: RNE
}

// moex (the ERNIE-5.5 hill-climb): kernels.cu's block_sum<256>, line for line (norm_rope_kernel's norm is rms_norm_kernel<256>'s)
__device__ __forceinline__ float block_sum256(float v) {
    __shared__ float red[8];
    __shared__ float total;
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float w = threadIdx.x < 8 ? red[threadIdx.x] : 0.0f;
        w = warp_sum(w);
        if (threadIdx.x == 0) total = w;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}

/// moex (the ERNIE-5.5 hill-climb): a row's RMS norm (kernels.cu rms_norm_kernel<256>: thread t's squares of values t, t + 256
/// in turn, block_sum<256>, 1 / sqrtf(s / cols + eps), the product, the weight), then rope_kernel's in-place rotation of its pairs
/// [offs / 2, offs / 2 + 32) and, with STORE (one row a token), raw_store_kernel's half copy into the ring - one block a row, each
/// step the kernel it replaces line for line: the same bits.  cols == kD
template <bool STORE>
__global__ void __launch_bounds__(256) norm_rope_kernel(float* __restrict__ x, const float* __restrict__ w, int cols, float eps, int rows,
                                                        int offs, RopeK rk, int64_t pos0_arg, const int64_t* __restrict__ pos_ptr,
                                                        __half* __restrict__ ring, int ring_rows) {
    __shared__ float row[kD];
    const int r = blockIdx.x;
    float* xr = x + (size_t) r * cols;
    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += 256) {
        const float v = xr[i];
        s += v * v;
    }
    s = block_sum256(s);
    const float scale = 1.0f / sqrtf(s / (float) cols + eps);
    for (int i = threadIdx.x; i < cols; i += 256) {
        const float v = xr[i] * scale;
        row[i] = w ? v * w[i] : v;
    }
    __syncthreads();
    if (threadIdx.x < kRopePairs) {   // rope_kernel's pair i = offs / 2 + k of row r (shift 0)
        const int k = threadIdx.x, i = offs / 2 + k;
        const float pos = (float) (pos_of(pos_ptr, pos0_arg) + r / rows);
        float c, sn;
        rope_cs(rk, pos, k, c, sn);
        const float2 v = make_float2(row[2 * i], row[2 * i + 1]);
        const float2 o = make_float2(__fmaf_rn(v.x, c, -__fmul_rn(v.y, sn)), __fmaf_rn(v.y, c, __fmul_rn(v.x, sn)));   // rope_kernel's
        row[2 * i] = o.x;
        row[2 * i + 1] = o.y;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += 256) xr[i] = row[i];
    if constexpr (STORE) {   // raw_store_kernel: token r at position pos + r, set_rows' RNE
        const int64_t p = pos_of(pos_ptr, pos0_arg) + r;
        half2* dst = reinterpret_cast<half2*>(ring + (size_t) (p % ring_rows) * kD);
        for (int i = threadIdx.x; i < kD / 2; i += 256) dst[i] = __float22half2_rn(make_float2(row[2 * i], row[2 * i + 1]));
    }
}

// ------------------------------------------------------------------ KV compressors

/// The 8-lane butterfly of ggml-cuda's warp_reduce_sum with 8 live lanes (softmax.cu:125 for the 8-slot softmax's
/// sum, sumrows.cu / reduce_rows.cuh:17-36 for the pooled sum): ((v0 + v4) + (v2 + v6)) + ((v1 + v5) + (v3 + v7)).
__device__ __forceinline__ float tree8(const float* v) {
    return ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
}
/// The 32-lane butterfly (strides 16, 8, 4, 2, 1) of 32 values: what warp_reduce_sum leaves in every lane.  The
/// strides are template arguments so every index is a constant (a runtime stride sends the array to local memory).
template <int S>
__device__ __forceinline__ void tree_stage(float* v) {
#pragma unroll
    for (int l = 0; l < S; ++l) v[l] = v[l] + v[l + S];
}
__device__ __forceinline__ float tree32(float* v) {
    tree_stage<16>(v);
    tree_stage<8>(v);
    tree_stage<4>(v);
    tree_stage<2>(v);
    tree_stage<1>(v);
    return v[0];
}

/// One block per completing compressed row, one thread per channel.  The CUDA ops of deepseek4.cpp:471-589 fused:
///   softmax (softmax.cu:102-141 soft_max_f32: max, expf(x - max), the float sum in warp_reduce_sum's butterfly order,
///            inv = 1.0f / sum, e * inv),  mul (v * w, rounded),  sum_rows (reduce_rows.cuh's lane sums then the
///            butterfly),  rms_norm + mul (norm.cu:133-150: mean = sum / d, rsqrtf(mean + eps), (scale * x) * w),
///   rope_norm at s0 (rope.cu), LID: fwht (scale first), set_rows (__float2half_rn).
/// (ggml-cpu sums in double and multiplies by (float)(1.0 / sum) - ops.cpp:5667-5683 - and normalizes with
/// 1.0f / sqrtf: ulp-level.)  The products feeding sums use __fmul_rn so they round on their own as the separate ggml
/// ops do (the compiler would otherwise contract them into the adds).
template <int D, int R, int COFF, bool HAD>
__global__ void __launch_bounds__(D) pool_kernel(const float* __restrict__ akv, const float* __restrict__ gs,
                                                 const float* __restrict__ ape, const float* __restrict__ ring_kv,
                                                 const float* __restrict__ ring_sc, int S, const float* __restrict__ norm_w,
                                                 float eps, RopeK rk, __half* __restrict__ cache, int64_t pos0_arg,
                                                 const int64_t* __restrict__ pos_ptr, int T) {
    constexpr int W = COFF * D;   // a projection / ring / ape row
    static_assert(COFF == 1 || 2 * R == 8, "the overlapping compressors pool 8 slots");
    static_assert(COFF == 2 || R == 128, "the plain compressor is HCA's 128");
    __shared__ float red[D / 32];
    __shared__ float hbuf[HAD ? D : 1];
    const int64_t pos0 = pos_of(pos_ptr, pos0_arg);
    const int64_t p = pos0 + ((R - (pos0 + 1) % R) % R) + (int64_t) blockIdx.x * R;   // this block's completing position
    if (p >= pos0 + T) return;                                                        // (static grids: no row here)
    const int64_t s0 = p + 1 - R;                                                     // = (p / R) * R, the rope position
    const int k = threadIdx.x;
    // a slot's (kv, score): the window's fresh rows (+ ape, as ggml_add rounds it), the ring before the window, or the
    // synthetic zero / -inf row of positions < 0 (deepseek4.cpp:209-215, 539-540)
    auto fetch = [&](int64_t q, int f, float& v, float& g) {
        if (q < 0) {
            v = 0.0f;
            g = -INFINITY;
        } else if (q >= pos0) {
            const size_t row = (size_t) (q - pos0) * W;
            v = akv[row + f];
            g = gs[row + f] + ape[(size_t) (q % R) * W + f];
        } else {
            const size_t row = (size_t) (q % S) * W;
            v = ring_kv[row + f];
            g = ring_sc[row + f];
        }
    };
    float c;
    if constexpr (COFF == 2) {
        // slots 0..R-1: the previous window's first half; R..2R-1: the current window's second half
        float v[8], g[8];
#pragma unroll
        for (int j = 0; j < R; ++j) fetch(s0 - R + j, k, v[j], g[j]);
#pragma unroll
        for (int j = 0; j < R; ++j) fetch(s0 + j, D + k, v[R + j], g[R + j]);
        float m = g[0];
#pragma unroll
        for (int j = 1; j < 8; ++j) m = fmaxf(m, g[j]);
        float e[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) e[j] = expf(g[j] - m);
        const float inv = 1.0f / tree8(e);
        float y[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(v[j], e[j] * inv);
        c = tree8(y);
    } else {
        // 128 slots: the softmax over 4 warps (each warp's butterfly, then (W0 + W2) + (W1 + W3)), the pooled sum as
        // reduce_rows_f32 with 32 lanes (lane l: ((y_l + y_l+32) + y_l+64) + y_l+96, then the butterfly)
        float m = -INFINITY;
        for (int j = 0; j < R; ++j) {
            float v_, g_;
            fetch(s0 + j, k, v_, g_);
            m = fmaxf(m, g_);
        }
        float wsum[4];
        for (int w = 0; w < 4; ++w) {
            float e[32];
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                float v_, g_;
                fetch(s0 + 32 * w + l, k, v_, g_);
                e[l] = expf(g_ - m);
            }
            wsum[w] = tree32(e);
        }
        const float inv = 1.0f / ((wsum[0] + wsum[2]) + (wsum[1] + wsum[3]));
        float a[32];
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            float acc = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float v_, g_;
                fetch(s0 + l + 32 * i, k, v_, g_);
                acc = acc + __fmul_rn(v_, expf(g_ - m) * inv);
            }
            a[l] = acc;
        }
        c = tree32(a);
    }
    // RMS over the d channels, then the weight
    const float sq = warp_sum(__fmul_rn(c, c));
    if ((k & 31) == 0) red[k >> 5] = sq;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < D / 32; ++w) tot += red[w];
    const float scale = rsqrtf(tot / (float) D + eps);
    float e = __fmul_rn(scale, c) * norm_w[k];
    // RoPE at s0 on [D - 64, D): pair partners are neighbouring lanes of whole warps
    if (k >= D - kRopeDims) {
        const float other = __shfl_xor_sync(0xffffffffu, e, 1);
        float cs, sn;
        rope_cs(rk, (float) s0, (k - (D - kRopeDims)) >> 1, cs, sn);
        e = (k & 1) == 0 ? e * cs - other * sn : other * sn + e * cs;
    }
    if constexpr (HAD) {
        // fwht over the row: scale first, strides 1..16 by shuffle, 32.. through shared memory
        float h = e * (1.0f / sqrtf((float) D));
#pragma unroll
        for (int s = 1; s < 32; s <<= 1) {
            const float o = __shfl_xor_sync(0xffffffffu, h, s);
            h = (k & s) == 0 ? h + o : o - h;
        }
#pragma unroll
        for (int s = 32; s < D; s <<= 1) {
            hbuf[k] = h;
            __syncthreads();
            const float o = hbuf[k ^ s];
            __syncthreads();
            h = (k & s) == 0 ? h + o : o - h;
        }
        e = h;
    }
    cache[(size_t) (p / R) * D + k] = __float2half_rn(e);
}

/// The last min(T, S) tokens of the window, one block each: they own distinct ring rows and are each row's latest.
__global__ void persist_kernel(const float* __restrict__ akv, const float* __restrict__ gs, const float* __restrict__ ape,
                               float* __restrict__ ring_kv, float* __restrict__ ring_sc, int R, int S, int W, int T,
                               int n_land, int64_t pos0_arg, const int64_t* __restrict__ pos_ptr) {
    const int t = T - n_land + (int) blockIdx.x;
    const int64_t p = pos_of(pos_ptr, pos0_arg) + t;
    const size_t src = (size_t) t * W, dst = (size_t) (p % S) * W, ap = (size_t) (p % R) * W;
    for (int f = threadIdx.x; f < W; f += blockDim.x) {
        ring_kv[dst + f] = akv[src + f];
        ring_sc[dst + f] = gs[src + f] + ape[ap + f];
    }
}

// ------------------------------------------------------------------ lightning indexer

constexpr int kLidHeads = 64, kLidDim = 128;   // the positions per row: the launchers' ratio (V4 4, V4.1 2 or 1)
constexpr int kLidRows = 128;   ///< K rows per tile
constexpr int kLidDc = 32;      ///< dims per stage
constexpr int kLidPad = kLidDc + 4;

/// Block (token t, slice y): the tiles y, y + gridDim.y, ... of 128 visible rows, all 64 heads.  A thread owns
/// 8 rows (tx + 16 i) x 4 heads (ty + 16 h) of q16 . k in F32 (fmaf over d ascending; the products of two halves are
/// exact), staged 32 dims at a time.  The epilogue is ggml-cuda's: per head relu(dot) * w rounded, then the heads
/// summed in order h = 0..63 from 0 (lightning-indexer.cu:175-197).  q is rounded to F16 as its tensor-core path does
/// (lightning-indexer.cu:61-68, 157-160), k is the cache's F16.
template <bool PK = false>   // moex (#104): K packed (kPackLid-byte rows)
__global__ void __launch_bounds__(256) lid_scores_kernel(const float* __restrict__ iq, const __half* __restrict__ K,
                                                         const float* __restrict__ w, float* __restrict__ score,
                                                         int max_rows, int64_t pos0_arg, const int64_t* __restrict__ pos_ptr,
                                                         int ratio) {
    struct Stage {
        float q[kLidHeads][kLidPad];
        float k[kLidRows][kLidPad];
    };
    union Smem {
        Stage s;
        float red[kLidHeads][kLidRows];
    };
    __shared__ __align__(16) Smem sm;
    __shared__ float wsh[kLidHeads];
    const int t = blockIdx.x;
    const int64_t pos = pos_of(pos_ptr, pos0_arg) + t;
    const int n_vis = (int) min64((pos + 1) / ratio, (int64_t) max_rows);
    const int n_tiles = (n_vis + kLidRows - 1) / kLidRows;
    if ((int) blockIdx.y >= n_tiles) return;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    if (tid < kLidHeads) wsh[tid] = w[(size_t) t * kLidHeads + tid];
    const float* qt = iq + (size_t) t * kLidHeads * kLidDim;
    float* out = score + (size_t) t * max_rows;
    for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y) {
        const int j0 = tile * kLidRows;
        float acc[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 4; ++h) acc[i][h] = 0.0f;
        for (int d0 = 0; d0 < kLidDim; d0 += kLidDc) {
            __syncthreads();   // the previous stage (or the previous tile's epilogue) is done with the buffer
            for (int e = tid; e < kLidHeads * kLidDc / 4; e += 256) {
                const int h = e / (kLidDc / 4), c4 = e % (kLidDc / 4);
                float4 v = *reinterpret_cast<const float4*>(qt + (size_t) h * kLidDim + d0 + c4 * 4);
                v.x = __half2float(__float2half_rn(v.x));
                v.y = __half2float(__float2half_rn(v.y));
                v.z = __half2float(__float2half_rn(v.z));
                v.w = __half2float(__float2half_rn(v.w));
                *reinterpret_cast<float4*>(&sm.s.q[h][c4 * 4]) = v;
            }
            for (int e = tid; e < kLidRows * kLidDc / 4; e += 256) {
                const int r = e / (kLidDc / 4), c4 = e % (kLidDc / 4);
                float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (j0 + r < n_vis && PK) {   // (moex, #104: the same halves from the packed row)
                    const uint8_t* row = reinterpret_cast<const uint8_t*>(K) + (size_t) (j0 + r) * kPackLid;
                    const int d = d0 + c4 * 4;
                    const uint32_t cw = *reinterpret_cast<const uint16_t*>(row + d / 2);
                    const float sc = ue8m0_val(row[64 + d / 32]);
                    v = make_float4(__half2float(__float2half_rn(e2m1_val(cw & 15u) * sc)),
                                    __half2float(__float2half_rn(e2m1_val((cw >> 4) & 15u) * sc)),
                                    __half2float(__float2half_rn(e2m1_val((cw >> 8) & 15u) * sc)),
                                    __half2float(__float2half_rn(e2m1_val((cw >> 12) & 15u) * sc)));
                } else if (j0 + r < n_vis) {
                    const uint2 u = *reinterpret_cast<const uint2*>(K + (size_t) (j0 + r) * kLidDim + d0 + c4 * 4);
                    const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
                    const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
                    v = make_float4(a.x, a.y, b.x, b.y);
                }
                *reinterpret_cast<float4*>(&sm.s.k[r][c4 * 4]) = v;
            }
            __syncthreads();
#pragma unroll
            for (int dd = 0; dd < kLidDc; dd += 4) {
                float4 qv[4], kv[8];
#pragma unroll
                for (int h = 0; h < 4; ++h) qv[h] = *reinterpret_cast<const float4*>(&sm.s.q[ty + 16 * h][dd]);
#pragma unroll
                for (int i = 0; i < 8; ++i) kv[i] = *reinterpret_cast<const float4*>(&sm.s.k[tx + 16 * i][dd]);
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        float a = acc[i][h];
                        a = fmaf(qv[h].x, kv[i].x, a);
                        a = fmaf(qv[h].y, kv[i].y, a);
                        a = fmaf(qv[h].z, kv[i].z, a);
                        a = fmaf(qv[h].w, kv[i].w, a);
                        acc[i][h] = a;
                    }
            }
        }
        __syncthreads();   // everyone is done with the stage buffers: they become `red`
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const float d = acc[i][h] > 0.0f ? acc[i][h] : 0.0f;
                sm.red[ty + 16 * h][tx + 16 * i] = __fmul_rn(d, wsh[ty + 16 * h]);
            }
        __syncthreads();
        if (tid < kLidRows && j0 + tid < n_vis) {
            float s = 0.0f;
#pragma unroll 8
            for (int h = 0; h < kLidHeads; ++h) s += sm.red[h][tid];
            out[j0 + tid] = s;
        }
    }
}

// THE INDEXER ON TENSOR CORES: q is rounded to F16 anyway (ggml-cuda's formulation), so mma.sync m16n8k16 f16 x f16 ->
// f32 computes the same exact products as lid_scores_kernel, accumulated in F32 in the hardware's order (ggml-cuda's
// lightning_indexer_kernel_wmma does the same with wmma).  The epilogue is the SIMT kernel's: relu(dot) * w rounded per
// head, the heads summed in order 0..63.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define STRATA_DSV4_MMA 1
#else
#define STRATA_DSV4_MMA 0
#endif

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if STRATA_DSV4_MMA
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
#if STRATA_DSV4_MMA
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#endif
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

constexpr int kLmRows = 64;             ///< K rows per tile of the tensor-core indexer
constexpr int kLmStride = kLidDim + 8;  ///< halves per staged row: 272 bytes, so ldmatrix's 8 rows hit distinct banks

__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool valid) {
#if STRATA_DSV4_MMA
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
#endif
}
__device__ __forceinline__ void cp_commit() {
#if STRATA_DSV4_MMA
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cp_wait() {
#if STRATA_DSV4_MMA
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

/// Block (token t, slice y): 4 warps, the tiles y, y + gridDim.y, ... of 64 visible rows, double-buffered by cp.async
/// (the next tile streams in while this one is multiplied).  Warp w: heads 32 (w & 1) .. +31 (2 m-tiles, A fragments of
/// q16 in registers for the whole block) x rows 32 (w >> 1) .. +31 (4 n-tiles), 8 k-steps of 16 dims; B by ldmatrix.
/// The terms relu(dot) * w go to shared memory (over the tile just consumed) and 64 threads sum them over h = 0..63.
/// moex (DSV41_PLAN §56 C), CAND: a tile's 64 rows are kLmRows / cbsize candidate blocks of clist (row r of the tile: block
/// clist[t * ccap + tile * bpt + r / cbsize], its row r % cbsize), so only the candidates' keys are read; each row's products,
/// relu, weight and heads' sum are the ones the full scan's tile gives it - the same bits - and the other rows are not written.
/// moex (DSV41_PLAN §59 P2), H32: V4.1's 32 heads (the caller's q and w padded to 64 with zeros, as for the 64-head tile): every
/// warp holds heads 0..31 and its own 16 rows of the tile (2 n-tiles), so no warp multiplies the zero heads - each (head, row)
/// product is the same m16n8k16 chain over the 8 k-steps, and the sum over h = 0..31 the 64-head one's before its zero terms.
template <bool PK = false, bool CAND = false, bool H32 = false>   // moex (#104): K packed - the tiles staged packed, then decoded
__global__ void __launch_bounds__(128) lid_scores_mma_kernel(const float* __restrict__ iq, const __half* __restrict__ K,
                                                             const float* __restrict__ w, float* __restrict__ score,
                                                             int max_rows, int64_t pos0_arg,
                                                             const int64_t* __restrict__ pos_ptr, int ratio,
                                                             const int32_t* __restrict__ clist = nullptr,
                                                             const int32_t* __restrict__ ccount = nullptr, int ccap = 0,
                                                             int cbsize = 1) {
#if STRATA_DSV4_MMA
    union Buf {
        __half k[kLmRows][kLmStride];
        float red[kLmRows][kLidHeads + 1];   // [row][head]
    };
    __shared__ __align__(16) Buf buf[2];
    __shared__ __align__(16) uint8_t pk[PK ? 2 : 1][PK ? kLmRows : 1][PK ? kPackLid : 16];
    __shared__ float wsh[kLidHeads];
    const int t = blockIdx.x;
    const int64_t pos = pos_of(pos_ptr, pos0_arg) + t;
    const int n_vis = (int) min64((pos + 1) / ratio, (int64_t) max_rows);
    const int bpt = CAND ? kLmRows / cbsize : 1;   // (CAND) candidate blocks a tile
    const int n_cand = CAND ? ccount[t] : 0;
    if (CAND && n_cand > ccap) __trap();           // dsv41_cand_list kept fewer than the token has: never score a part
    const int n_tiles = CAND ? (n_cand + bpt - 1) / bpt : (n_vis + kLmRows - 1) / kLmRows;
    if ((int) blockIdx.y >= n_tiles) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    // the key row behind row r of tile `tile`, or -1 (past n_vis, or past the token's candidates)
    auto row_of = [&](int tile, int r) -> int {
        if (!CAND) {
            const int j = tile * kLmRows + r;
            return j < n_vis ? j : -1;
        }
        const int i = tile * bpt + r / cbsize;
        if (i >= n_cand) return -1;
        const int j = clist[(size_t) t * ccap + i] * cbsize + r % cbsize;
        return j < n_vis ? j : -1;
    };
    // tile `tile`'s 64 rows into buf[b] (rows past n_vis zero-filled); packed, into pk[b]
    auto stage = [&](int tile, int b) {
        if (PK) {
            const uint8_t* Kp = reinterpret_cast<const uint8_t*>(K);
            for (int e = tid; e < kLmRows * (kPackLid / 16); e += blockDim.x) {
                const int r = e / (kPackLid / 16), c = e % (kPackLid / 16);
                const int j = row_of(tile, r);
                cp16(&pk[b][r][c * 16], Kp + (size_t) (j >= 0 ? j : 0) * kPackLid + c * 16, j >= 0);
            }
        } else {
            for (int e = tid; e < kLmRows * (kLidDim / 8); e += blockDim.x) {
                const int r = e / (kLidDim / 8), c8 = e % (kLidDim / 8);
                const int j = row_of(tile, r);
                cp16(&buf[b].k[r][c8 * 8], K + (size_t) (j >= 0 ? j : 0) * kLidDim + c8 * 8, j >= 0);
            }
        }
        cp_commit();
    };
    stage(blockIdx.y, 0);
    constexpr int NH = H32 ? 32 : kLidHeads, NT = H32 ? 2 : 4;   // the heads summed; a warp's n-tiles of 8 rows
    if (tid < NH) wsh[tid] = w[(size_t) t * kLidHeads + tid];
    // A: this warp's 32 heads x 128 dims of q, rounded to F16 (the m16n8k16 row-major A layout, as kernels.cu's
    // attend_mma_kernel loads q)
    const int hb = H32 ? 0 : 32 * (warp & 1);
    const float* qt = iq + (size_t) t * kLidHeads * kLidDim;
    uint32_t af[2][8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = hb + 16 * mt + gid + (r & 1) * 8, col = kk * 16 + 2 * tig + (r >> 1) * 8;
                const float2 x = *reinterpret_cast<const float2*>(qt + (size_t) row * kLidDim + col);
                af[mt][kk][r] = pack_h2(x.x, x.y);
            }
    float* out = score + (size_t) t * max_rows;
    const int key0 = H32 ? 16 * warp : 32 * (warp >> 1);
    int b = 0;
    for (int tile = blockIdx.y; tile < n_tiles; tile += gridDim.y, b ^= 1) {
        const int jrow = tid < kLmRows ? row_of(tile, tid) : -1;   // the score this thread writes (if any)
        cp_wait<0>();
        __syncthreads();   // this tile is in buf[b]; buf[b ^ 1]'s terms (the previous tile) are summed
        if (PK) {   // (moex, #104) the packed tile into the F16 tile, the halves the F16 cache holds (zero rows stay zero)
            for (int e = tid; e < kLmRows * (kLidDim / 8); e += blockDim.x) {
                const int r = e / (kLidDim / 8), c8 = e % (kLidDim / 8);
                *reinterpret_cast<uint4*>(&buf[b].k[r][c8 * 8]) = lid_unpack8(&pk[b][r][0], c8 * 8);
            }
            __syncthreads();
        }
        if (tile + (int) gridDim.y < n_tiles) stage(tile + gridDim.y, b ^ 1);
        float acc[2][NT][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            // B of n-tiles (0, 1) and (2, 3): matrix m of an x4 = rows 8 (m >> 1) .., dims 8 (m & 1) .. of the k-step
            uint32_t bf[NT / 2][4];
#pragma unroll
            for (int p = 0; p < NT / 2; ++p)
                ldsm_x4(bf[p], &buf[b].k[key0 + 16 * p + (lane & 7) + 8 * (lane >> 4)][kk * 16 + 8 * ((lane >> 3) & 1)]);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) mma16816(acc[mt][nt], af[mt][kk], &bf[nt >> 1][2 * (nt & 1)]);
        }
        __syncthreads();   // every warp is done reading buf[b].k: it becomes the terms
        // c0, c1 = (head gid, rows 2 tig, +1), c2, c3 = (head gid + 8, ...)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int h = hb + 16 * mt + gid, r = key0 + 8 * nt + 2 * tig;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int hh = h + (q >> 1) * 8, rr = r + (q & 1);
                    const float d = acc[mt][nt][q] > 0.0f ? acc[mt][nt][q] : 0.0f;
                    buf[b].red[rr][hh] = __fmul_rn(d, wsh[hh]);
                }
            }
        __syncthreads();
        if (jrow >= 0) {
            float sum = 0.0f;
#pragma unroll 8
            for (int h = 0; h < NH; ++h) sum += buf[b].red[tid][h];
            out[jrow] = sum;
        }
    }
    cp_wait<0>();
#else
    __trap();   // built for a card without mma.sync m16n8k16 f16 (sm_80): the host keeps lid_scores_kernel
#endif
}

/// float -> uint32 whose unsigned order is the float order (non-NaN; -0 is +0 first, so they tie as floats do).
__device__ __forceinline__ uint32_t okey(float f) {
    const uint32_t u = __float_as_uint(__fadd_rn(f, 0.0f));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// The exclusive prefix over the block's warps of two per-warp counts, and the block totals (warp 0 scans; the arrays
/// hold 33 words).  Synchronizes: every thread must call it.
__device__ __forceinline__ void block_prefix2(uint32_t a, uint32_t b, uint32_t* sa, uint32_t* sb, uint32_t& a_before,
                                              uint32_t& b_before, uint32_t& a_all, uint32_t& b_all) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (int) (blockDim.x >> 5);
    if (lane == 0) {
        sa[warp] = a;
        sb[warp] = b;
    }
    __syncthreads();
    if (warp == 0) {
        const uint32_t x0 = lane < nw ? sa[lane] : 0u, y0 = lane < nw ? sb[lane] : 0u;
        uint32_t x = x0, y = y0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const uint32_t xs = __shfl_up_sync(0xffffffffu, x, o), ys = __shfl_up_sync(0xffffffffu, y, o);
            if (lane >= o) {
                x += xs;
                y += ys;
            }
        }
        sa[lane] = x - x0;
        sb[lane] = y - y0;
        if (lane == 31) {
            sa[32] = x;
            sb[32] = y;
        }
    }
    __syncthreads();
    a_before = sa[warp];
    b_before = sb[warp];
    a_all = sa[32];
    b_all = sb[32];
    __syncthreads();   // the next call rewrites the arrays
}

/// MSB-first radix select over the n keys get(i): the k-th largest key (*thr) and how many of the keys equal to it are
/// among the top k (*rem).  8-bit digits; the histogram atomics aggregated per warp (__match_any_sync: a warp's keys
/// mostly share the high digits).  Every thread must call it.
template <typename Get>
__device__ void radix_select(Get get, int n, uint32_t k, uint32_t* hist, uint32_t* s_digit, uint32_t* s_rem,
                             uint32_t* thr, uint32_t* rem) {
    const int lane = threadIdx.x & 31;
    uint32_t prefix = 0, mask = 0, remaining = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        __syncthreads();   // the previous pass's histogram and digit are consumed
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int base = 0; base < n; base += blockDim.x) {
            const int i = base + (int) threadIdx.x;
            const uint32_t key = i < n ? get(i) : 0u;
            const bool live = i < n && (key & mask) == prefix;
            const uint32_t dg = (key >> shift) & 255u;
            const unsigned act = __ballot_sync(0xffffffffu, live);
            if (live) {
                const unsigned peers = __match_any_sync(act, dg);
                if (lane == __ffs(peers) - 1) atomicAdd(&hist[dg], (uint32_t) __popc(peers));
            }
        }
        __syncthreads();
        if (threadIdx.x < 32) {   // the digit where the count from the top reaches `remaining`: lane l owns bins 255-8l-j
            uint32_t c[8], own = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) own += (c[j] = hist[255 - (8 * lane + j)]);
            uint32_t inc = own;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const uint32_t y = __shfl_up_sync(0xffffffffu, inc, o);
                if (lane >= o) inc += y;
            }
            const uint32_t exc = inc - own;
            if (exc < remaining && remaining <= inc) {
                uint32_t cum = exc;
                for (int j = 0; j < 8; ++j) {
                    if (cum + c[j] >= remaining) {
                        *s_digit = 255u - (uint32_t) (8 * lane + j);
                        *s_rem = remaining - cum;
                        break;
                    }
                    cum += c[j];
                }
            }
        }
        __syncthreads();
        prefix |= *s_digit << shift;
        mask |= 255u << shift;
        remaining = *s_rem;
    }
    *thr = prefix;
    *rem = remaining;
}

/// The selected entries of n keys in index order: key > thr always, key == thr for the first `rem` (the lowest
/// indices), out[] = row(i), 1024 entries at a time (ballots + a block scan).  Every thread must call it.
template <typename Key, typename Row>
__device__ void ordered_select(Key key_of, Row row_of, int n, uint32_t thr, uint32_t rem, int32_t* __restrict__ out,
                               uint32_t* tot_gt, uint32_t* tot_eq) {
    const int lane = threadIdx.x & 31;
    const unsigned lt = (1u << lane) - 1u;
    uint32_t written = 0, eq_seen = 0;
    for (int base = 0; base < n; base += blockDim.x) {
        const int i = base + (int) threadIdx.x;
        const uint32_t key = i < n ? key_of(i) : 0u;
        const bool gt = i < n && key > thr, eq = i < n && key == thr;
        const unsigned bgt = __ballot_sync(0xffffffffu, gt), beq = __ballot_sync(0xffffffffu, eq);
        uint32_t gt_before, eq_before, all_gt, all_eq;
        block_prefix2((uint32_t) __popc(bgt), (uint32_t) __popc(beq), tot_gt, tot_eq, gt_before, eq_before, all_gt, all_eq);
        gt_before += (uint32_t) __popc(bgt & lt);
        eq_before += (uint32_t) __popc(beq & lt);
        const uint32_t eq_room = rem > eq_seen ? rem - eq_seen : 0u;   // ties still to take
        if (gt || (eq && eq_before < eq_room)) out[written + gt_before + min(eq_before, eq_room)] = row_of(i);
        written += all_gt + min(all_eq, eq_room);
        eq_seen += all_eq;
    }
}

constexpr int kSelThreads = 1024;
constexpr int kSelCap = 4096;   ///< candidate rows the fast path keeps in shared memory

/// The rows i < n whose key is >= tau into (ckey, crow) in ascending row order, at most kSelCap of them; returns how
/// many there are.  Warp w owns the contiguous rows of its 128-row groups, a lane rows g + l + 32 j (j < 4) of a group:
/// coalesced, four loads in flight per lane, a ballot per j orders the group.  Counted, the warps' offsets scanned,
/// then written (the second read hits the cache).  Every thread must call it.
__device__ uint32_t compact_ge(const float* __restrict__ sc, int n, uint32_t tau, uint32_t* ckey, int32_t* crow,
                               uint32_t* tmp) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (int) (blockDim.x >> 5);
    const unsigned lt = (1u << lane) - 1u;
    const int gpw = ((n + 127) / 128 + nw - 1) / nw;   // groups per warp
    const int r0 = min(n, warp * gpw * 128), r1 = min(n, r0 + gpw * 128);
    uint32_t cnt = 0;
#pragma unroll 2
    for (int g = r0; g < r1; g += 128) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = g + 32 * j + lane;
            v[j] = i < r1 ? sc[i] : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) cnt += (g + 32 * j + lane < r1 && okey(v[j]) >= tau) ? 1u : 0u;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
    __syncthreads();   // tmp may still be read by a previous call
    if (lane == 0) tmp[warp] = cnt;
    __syncthreads();
    uint32_t off = 0, total = 0;
    for (int w2 = 0; w2 < nw; ++w2) {
        const uint32_t c = tmp[w2];
        off += w2 < warp ? c : 0u;
        total += c;
    }
    if (total <= (uint32_t) kSelCap) {
        for (int g = r0; g < r1; g += 128) {
            uint32_t key[4];
            bool f[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int i = g + 32 * j + lane;
                key[j] = i < r1 ? okey(sc[i]) : 0u;
                f[j] = i < r1 && key[j] >= tau;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned b = __ballot_sync(0xffffffffu, f[j]);
                if (f[j]) {
                    const uint32_t o = off + (uint32_t) __popc(b & lt);
                    ckey[o] = key[j];
                    crow[o] = g + 32 * j + lane;
                }
                off += (uint32_t) __popc(b);
            }
        }
    }
    __syncthreads();
    return total;
}

/// One block (1024 threads) per token: every visible row when n_vis <= top_k; else the exact top_k by score with ties
/// to the lower row, written in ASCENDING row order.  Fast path: a radix select over 1024 sampled scores gives a
/// threshold with ~1.25 top_k rows above it, one pass compacts those candidates into shared memory, and the radix
/// select of the top_k runs there;
/// when the candidates miss [top_k, 4096] after a few thresholds (ties, odd distributions) the radix select runs over
/// all the scores instead.  Either way the same set in the same order: deterministic.
__global__ void __launch_bounds__(kSelThreads) lid_select_kernel(const float* __restrict__ score, int max_rows, int top_k,
                                                                 int32_t* __restrict__ sel, int32_t* __restrict__ count,
                                                                 int64_t pos0_arg, const int64_t* __restrict__ pos_ptr, int ratio,
                                                                 int excl) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_digit, s_rem;
    __shared__ uint32_t tot_a[33], tot_b[33];
    __shared__ uint32_t samp[kSelThreads];
    __shared__ uint32_t ckey[kSelCap];
    __shared__ int32_t crow[kSelCap];
    const int t = blockIdx.x;
    const int64_t pos = pos_of(pos_ptr, pos0_arg) + t;
    // moex (ERNIE-5.5): excl > 0 leaves the last excl positions out (its DSA layers: the window is attended apart)
    const int64_t vis = pos + 1 - excl;
    const int n = (int) min64((vis > 0 ? vis : 0) / ratio, (int64_t) max_rows);
    int32_t* out = sel + (size_t) t * top_k;
    const float* sc = score + (size_t) t * max_rows;
    if (count != nullptr && threadIdx.x == 0) count[t] = n < top_k ? n : top_k;
    if (n <= top_k) {
        for (int i = threadIdx.x; i < top_k; i += blockDim.x) out[i] = i < n ? i : -1;
        return;
    }
    uint32_t tau = 0;   // n <= 1024: every row is a candidate
    int s_idx = 0;
    uint32_t thr, rem;
    if (n > kSelThreads) {
        samp[threadIdx.x] = okey(sc[(int64_t) threadIdx.x * n / kSelThreads]);
        // sample s (descending) has ~(s + 1) n / 1024 rows at or above it: aim at 1.25 top_k plus 8 samples of margin
        s_idx = min(kSelThreads - 1, (int) ((int64_t) 5 * top_k * kSelThreads / (4 * (int64_t) n)) + 8);
        radix_select([&](int i) { return samp[i]; }, kSelThreads, (uint32_t) s_idx + 1, hist, &s_digit, &s_rem, &tau, &rem);
    }
    bool fast = false;
    uint32_t m = 0;
    for (int attempt = 0; attempt < 4; ++attempt) {
        m = compact_ge(sc, n, tau, ckey, crow, tot_a);
        if (m >= (uint32_t) top_k && m <= (uint32_t) kSelCap) {
            fast = true;
            break;
        }
        if (n <= kSelThreads) break;   // (cannot happen: n <= 1024 rows all fit)
        const int next = m < (uint32_t) top_k ? min(kSelThreads - 1, 2 * s_idx + 16) : s_idx / 2;
        if (next == s_idx) break;
        s_idx = next;
        radix_select([&](int i) { return samp[i]; }, kSelThreads, (uint32_t) s_idx + 1, hist, &s_digit, &s_rem, &tau, &rem);
    }
    if (fast) {
        radix_select([&](int i) { return ckey[i]; }, (int) m, (uint32_t) top_k, hist, &s_digit, &s_rem, &thr, &rem);
        ordered_select([&](int i) { return ckey[i]; }, [&](int i) { return crow[i]; }, (int) m, thr, rem, out, tot_a, tot_b);
    } else {
        radix_select([&](int i) { return okey(sc[i]); }, n, (uint32_t) top_k, hist, &s_digit, &s_rem, &thr, &rem);
        ordered_select([&](int i) { return okey(sc[i]); }, [&](int i) { return (int32_t) i; }, n, thr, rem, out, tot_a, tot_b);
    }
}

// ------------------------------------------------------------------ sparse attention with sinks

constexpr int kAttnHeads = 64;
constexpr int kAttnCh = 32;   ///< keys per chunk (staged in shared memory)
constexpr int kAttnHg = 16;   ///< heads per block: 8 warps x 2
constexpr int kAttnWin = 128; ///< the raw window (n_swa)
constexpr int kPart = kD + 2; ///< a split's partial: acc[512], m, l

/// Token at p: its raw keys (positions hi - n_raw + 1 .. hi) then its compressed rows.  Causal: hi = p and the 128-position
/// window below it.  A non-causal batch starting at b0 (moex: DSpark's noise block, DeepSeek-V4's image span - llama.cpp's
/// non-causal ubatch, llama-kv-cache.cpp:1677-1703: the batch's cells skip the future mask AND the window, older cells keep
/// the window): hi = the batch's last position, and the keys start at min(the token's window start, b0).
/// `b0`: the batch's first position for a non-causal batch, p for a causal one (min(lo, p) = lo: the window alone).
/// moex, DeepSeek-V4.1's DSpark block (nc == 2, model.py get_dspark_topk_idxs): every row sees the 128 positions before the
/// block and the whole block - the keys run from max(0, b0 - 128) to the batch's last position, whatever the row.
__device__ __forceinline__ void attn_counts(int64_t p, int64_t hi, int64_t b0, int ratio, int top_k, int& n_raw, int& n_comp,
                                            int nc = 0) {
    const int64_t lo = nc == 2 ? b0 - min64(b0, kAttnWin) : p - min64(p + 1, kAttnWin) + 1;
    n_raw = (int) (hi - (b0 < lo ? b0 : lo) + 1);
    n_comp = 0;
    if (ratio == 128) n_comp = (int) ((p + 1) / 128);
    else if (ratio > 0) n_comp = (int) min64((p + 1) / ratio, (int64_t) top_k);   // V4's 4, V4.1's 2 and 1: the selected rows
}

__device__ __forceinline__ const __half* attn_key(int i, int64_t hi, int n_raw, const __half* raw, int raw_rows,
                                                  const __half* comp, int ratio, const int32_t* sel_t) {
    if (i < n_raw) return raw + (size_t) ((hi - n_raw + 1 + i) % raw_rows) * kD;
    const int c = i - n_raw;
    return comp + (size_t) (ratio == 128 ? c : sel_t[c]) * kD;
}

/// Block (split s, 16-head group, token t): the chunks [s * cps, (s + 1) * cps) of the token's keys, 32 keys staged at
/// a time; warp w owns heads 2w, 2w + 1 of the group, a lane dims 2 * lane + 64 i (+1).  Per chunk: the 32 dots by
/// lane partials, summed across the warp by a transpose-reduce (lane L ends with key L & 15 of each 16), the online
/// softmax (max, exp, running sum per lane), then acc = acc * corr + sum_c p_c k_c with p in shared memory.  F32
/// throughout; s = (q . k) * scale as ggml-cpu's soft_max scales (ops.cpp:5647).  One split: the sink is folded in
/// and the output written (out = acc * a / (l * a + exp(sink - M)), M = max(m, sink): the FA sink handling of
/// fattn-mma-f16.cuh:1394-1412); several: (acc, m, l) to `part` for attn_merge_kernel.
template <bool PK = false>   // moex (#104): the compressed rows packed (kPackComp-byte rows; the raw ring stays F16)
__global__ void __launch_bounds__(256, 2) attn_partial_kernel(const float* __restrict__ q, const __half* __restrict__ raw,
                                                              int raw_rows, const __half* __restrict__ comp, int ratio,
                                                              const int32_t* __restrict__ sel, int top_k,
                                                              const float* __restrict__ sinks, float scale,
                                                              float* __restrict__ out, float* __restrict__ part, int n_split,
                                                              int64_t pos0_arg, const int64_t* __restrict__ pos_ptr, int nc) {
    __shared__ __align__(16) __half tile[kAttnCh][kD];
    __shared__ __align__(16) float pbuf[8][2][kAttnCh];
    const int s = blockIdx.x, hg = blockIdx.y, t = blockIdx.z;
    const int64_t p = pos_of(pos_ptr, pos0_arg) + t;
    const int64_t hi = nc ? p - t + (int64_t) gridDim.z - 1 : p;
    int n_raw, n_comp;
    attn_counts(p, hi, nc ? p - t : p, ratio, top_k, n_raw, n_comp, nc);
    const int n_keys = n_raw + n_comp;
    const int n_chunks = (n_keys + kAttnCh - 1) / kAttnCh;
    const int cps = (n_chunks + n_split - 1) / n_split;
    const int c_begin = s * cps, c_end = min(n_chunks, c_begin + cps);
    if (c_begin >= c_end) return;   // attn_merge_kernel reads only the splits that have chunks
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h0 = hg * kAttnHg + warp * 2;
    const int32_t* sel_t = sel != nullptr ? sel + (size_t) t * top_k : nullptr;
    float2 qv[2][8];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int i = 0; i < 8; ++i)
            qv[hh][i] = *reinterpret_cast<const float2*>(q + ((size_t) t * kAttnHeads + h0 + hh) * kD + 2 * lane + 64 * i);
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    float2 acc[2][8];
#pragma unroll
    for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[hh][i] = make_float2(0.0f, 0.0f);

    for (int c = c_begin; c < c_end; ++c) {
        const int k0 = c * kAttnCh;
        const int nk = min(kAttnCh, n_keys - k0);
        __syncthreads();   // the previous chunk's tile is consumed
        for (int e = threadIdx.x; e < kAttnCh * (kD / 8); e += blockDim.x) {
            const int r = e / (kD / 8), part8 = e % (kD / 8);
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (r < nk && PK && k0 + r >= n_raw) {   // (moex, #104: the same halves from the packed row)
                const int c = k0 + r - n_raw;
                v = comp_unpack8(reinterpret_cast<const uint8_t*>(comp) + (size_t) (ratio == 128 ? c : sel_t[c]) * kPackComp, part8 * 8);
            } else if (r < nk) {
                v = reinterpret_cast<const uint4*>(attn_key(k0 + r, hi, n_raw, raw, raw_rows, comp, ratio, sel_t))[part8];
            }
            reinterpret_cast<uint4*>(&tile[r][0])[part8] = v;
        }
        __syncthreads();
        // scores: 16 keys at a time; lanes L and L ^ 16 summed, then a 16-lane transpose-reduce (15 shuffles)
        float sc[2][2];
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            float ps[2][16];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const half2* row = reinterpret_cast<const half2*>(&tile[sub * 16 + j][0]);
                float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float2 kf = __half22float2(row[lane + 32 * i]);
                    d0 = fmaf(qv[0][i].x, kf.x, d0);
                    d0 = fmaf(qv[0][i].y, kf.y, d0);
                    d1 = fmaf(qv[1][i].x, kf.x, d1);
                    d1 = fmaf(qv[1][i].y, kf.y, d1);
                }
                ps[0][j] = d0;
                ps[1][j] = d1;
            }
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
                for (int j = 0; j < 16; ++j) ps[hh][j] += __shfl_xor_sync(0xffffffffu, ps[hh][j], 16);
                transpose_stage<8>(ps[hh], lane);
                transpose_stage<4>(ps[hh], lane);
                transpose_stage<2>(ps[hh], lane);
                transpose_stage<1>(ps[hh], lane);
                sc[hh][sub] = ps[hh][0];   // key sub * 16 + (lane & 15)
            }
        }
        float corr[2];
        const int ka = lane & 15, kb = 16 + (lane & 15);
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const float sa = ka < nk ? sc[hh][0] * scale : -INFINITY;
            const float sb = kb < nk ? sc[hh][1] * scale : -INFINITY;
            const float mn = fmaxf(m[hh], warp_max(fmaxf(sa, sb)));
            corr[hh] = expf(m[hh] - mn);   // m = -inf on the first chunk: 0
            const float pa = expf(sa - mn), pb = expf(sb - mn);
            l[hh] = l[hh] * corr[hh] + (lane < 16 ? pa + pb : 0.0f);
            m[hh] = mn;
            if (lane < 16) {
                pbuf[warp][hh][ka] = pa;
                pbuf[warp][hh][kb] = pb;
            }
        }
        __syncwarp();
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[hh][i].x *= corr[hh];
                acc[hh][i].y *= corr[hh];
            }
        for (int cc = 0; cc < nk; ++cc) {
            const float pa = pbuf[warp][0][cc], pb = pbuf[warp][1][cc];
            const half2* row = reinterpret_cast<const half2*>(&tile[cc][0]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float2 kf = __half22float2(row[lane + 32 * i]);
                acc[0][i].x = fmaf(pa, kf.x, acc[0][i].x);
                acc[0][i].y = fmaf(pa, kf.y, acc[0][i].y);
                acc[1][i].x = fmaf(pb, kf.x, acc[1][i].x);
                acc[1][i].y = fmaf(pb, kf.y, acc[1][i].y);
            }
        }
        __syncwarp();   // pbuf is rewritten by the next chunk
    }
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        const int h = h0 + hh;
        const float lsum = warp_sum(l[hh]);
        if (n_split == 1) {
            const float sink = sinks[h];
            const float M = fmaxf(m[hh], sink);
            const float a = expf(m[hh] - M);
            const float den = lsum * a + expf(sink - M);
            float* o = out + ((size_t) t * kAttnHeads + h) * kD;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                *reinterpret_cast<float2*>(o + 2 * lane + 64 * i) = make_float2(acc[hh][i].x * a / den, acc[hh][i].y * a / den);
        } else {
            float* pp = part + (((size_t) t * kAttnHeads + h) * n_split + s) * kPart;
#pragma unroll
            for (int i = 0; i < 8; ++i) *reinterpret_cast<float2*>(pp + 2 * lane + 64 * i) = acc[hh][i];
            if (lane == 0) {
                pp[kD] = m[hh];
                pp[kD + 1] = lsum;
            }
        }
    }
}

/// grid (64 heads, T): M = max(sink, the splits' maxima); out = sum_s e^(m_s - M) acc_s / (sum_s e^(m_s - M) l_s +
/// e^(sink - M)) - flash_attn_combine_results' arithmetic (fattn-common.cuh:960-972) with the sink in the denominator.
/// `log2`: the partials' maxima are in the log2 domain (attn_mma_kernel): exp2f, and the sink times log2 e.
__global__ void __launch_bounds__(256) attn_merge_kernel(const float* __restrict__ part, const float* __restrict__ sinks,
                                                         float* __restrict__ out, int ratio, int top_k, int n_split, bool log2,
                                                         int64_t pos0_arg, const int64_t* __restrict__ pos_ptr, int nc) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int64_t p = pos_of(pos_ptr, pos0_arg) + t;
    const int64_t hi = nc ? p - t + (int64_t) gridDim.y - 1 : p;
    int n_raw, n_comp;
    attn_counts(p, hi, nc ? p - t : p, ratio, top_k, n_raw, n_comp, nc);
    const int n_chunks = (n_raw + n_comp + kAttnCh - 1) / kAttnCh;
    const int cps = (n_chunks + n_split - 1) / n_split;
    const int n_used = (n_chunks + cps - 1) / cps;
    const float* base = part + ((size_t) t * kAttnHeads + h) * n_split * kPart;
    const float sink = log2 ? sinks[h] * 1.4426950408889634f : sinks[h];
    auto ex = [&](float x) { return log2 ? exp2f(x) : expf(x); };
    float M = sink;
    for (int sp = 0; sp < n_used; ++sp) M = fmaxf(M, base[(size_t) sp * kPart + kD]);
    float den = 0.0f;
    for (int sp = 0; sp < n_used; ++sp) {
        const float* pp = base + (size_t) sp * kPart;
        den += ex(pp[kD] - M) * pp[kD + 1];
    }
    den += ex(sink - M);
    float* o = out + ((size_t) t * kAttnHeads + h) * kD;
    for (int d = 2 * threadIdx.x; d < kD; d += 2 * blockDim.x) {
        float2 num = make_float2(0.0f, 0.0f);
        for (int sp = 0; sp < n_used; ++sp) {
            const float* pp = base + (size_t) sp * kPart;
            const float wgt = ex(pp[kD] - M);
            const float2 a = *reinterpret_cast<const float2*>(pp + d);
            num.x += wgt * a.x;
            num.y += wgt * a.y;
        }
        *reinterpret_cast<float2*>(o + d) = make_float2(num.x / den, num.y / den);
    }
}

// THE ATTENTION ON TENSOR CORES - kernels.cu's attend_mma_kernel (Strata's qsa_prompt_attn for an MLA latent: every
// head attends to the same F16 rows of 512, K = V) with DSV4's keys, split-K partials and the sink.  A block (16 heads of
// one token, one split) walks its chunks of 32 keys with an online softmax, one mma.sync m16n8k16 per (16 heads, 8 keys,
// 16 dims):
//   * a warp owns 128 of the 512 dims: it gathers its slice of the chunk's rows by cp.async, holds its q fragments in
//     registers, and the slice serves q.k (its partial score) and p.v (its output dims);
//   * the four warps' partial scores are summed in a fixed order through shared memory, the softmax is the block's;
//   * q and p are split into F16 hi + lo parts (two MMAs each), ~22 bits: the result differs from attn_partial_kernel by
//     summation order and exp2f's rounding, not by an F16 cast (the keys are F16, exact);
//   * q is pre-scaled by the power of two that puts its largest value near 2^14 (exact), p by 2^14; both are taken out.
// Scores in the log2 domain (s * scale * log2 e), and so are the maxima of the partials it leaves (merged with log2 = true).
constexpr int kAmCh = 32, kAmDims = 128, kAmStride = kAmDims + 8, kAmThreads = 128;
constexpr float kLog2e = 1.4426950408889634f;

struct AmSmem {
    __half kv[4][kAmCh][kAmStride];   // warp, key, the warp's 128 dims (+8 pad: 272-byte rows spread the banks)
    float part[4][16][kAmCh + 1];
    float p[16][kAmCh + 1];
    float alpha[16], lsum[16], mrow[16], qmax[4];
};

__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
#if STRATA_DSV4_MMA
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#endif
}

template <bool PK = false>   // moex (#104): the compressed rows packed - decoded as they are staged (the raw ring by cp.async)
__global__ void __launch_bounds__(kAmThreads, 2) attn_mma_kernel(const float* __restrict__ q, const __half* __restrict__ raw,
                                                                 int raw_rows, const __half* __restrict__ comp, int ratio,
                                                                 const int32_t* __restrict__ sel, int top_k,
                                                                 const float* __restrict__ sinks, float scale,
                                                                 float* __restrict__ out, float* __restrict__ part,
                                                                 int n_split, int64_t pos0_arg,
                                                                 const int64_t* __restrict__ pos_ptr, int nc) {
#if STRATA_DSV4_MMA
    __shared__ __align__(16) AmSmem S;
    const int s = blockIdx.x, hg = blockIdx.y, t = blockIdx.z;
    const int64_t p = pos_of(pos_ptr, pos0_arg) + t;
    const int64_t hi = nc ? p - t + (int64_t) gridDim.z - 1 : p;
    int n_raw, n_comp;
    attn_counts(p, hi, nc ? p - t : p, ratio, top_k, n_raw, n_comp, nc);
    const int n_keys = n_raw + n_comp;
    const int n_chunks = (n_keys + kAmCh - 1) / kAmCh;
    const int cps = (n_chunks + n_split - 1) / n_split;
    const int c_begin = s * cps, c_end = min(n_chunks, c_begin + cps);
    if (c_begin >= c_end) return;   // attn_merge_kernel reads only the splits that have chunks
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * kAmDims;
    const int32_t* sel_t = sel != nullptr ? sel + (size_t) t * top_k : nullptr;
    const float* qb = q + ((size_t) t * kAttnHeads + (size_t) hg * 16) * kD;

    // q, read once into this warp's A-fragment positions (16 heads x its 128 dims); the power-of-two prescale over the
    // block's 16 heads, then hi / lo fragments
    float2 qf[8][4];
    float qm = 0.0f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = gid + (r & 1) * 8, col = dim0 + kk * 16 + 2 * tig + (r >> 1) * 8;
            qf[kk][r] = *reinterpret_cast<const float2*>(qb + (size_t) row * kD + col);
            qm = fmaxf(qm, fmaxf(fabsf(qf[kk][r].x), fabsf(qf[kk][r].y)));
        }
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
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale * kLog2e, qe - 14);
    uint32_t qh[8][4], ql[8][4];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            float2 x = qf[kk][r];
            x.x *= qup;
            x.y *= qup;
            const __half2 hi = __floats2half2_rn(x.x, x.y);
            const float2 hf = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(x.x - hf.x, x.y - hf.y);
            qh[kk][r] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t*>(&lo);
        }

    // key k's row (lane k of a chunk computes it, the warp's 16 issuing lanes take it by shuffle)
    auto row_of = [&](int k) -> const __half* {
        if (PK && k < n_keys && k >= n_raw) {   // (moex, #104) a packed compressed row
            const int c = k - n_raw;
            return reinterpret_cast<const __half*>(reinterpret_cast<const uint8_t*>(comp) + (size_t) (ratio == 128 ? c : sel_t[c]) * kPackComp);
        }
        return k < n_keys ? attn_key(k, hi, n_raw, raw, raw_rows, comp, ratio, sel_t) : nullptr;
    };
    auto issue = [&](const __half* row_lane, int kc0) {   // the chunk's 32 rows (keys kc0..), this warp's 256 bytes of each
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int idx = lane + 32 * j, c = idx >> 4, pc = idx & 15;
            const __half* row = reinterpret_cast<const __half*>(__shfl_sync(0xffffffffu, (unsigned long long) row_lane, c));
            const bool ok = row != nullptr;
            if (PK && ok && kc0 + c >= n_raw)   // the packed row's 8 values, decoded into the slot cp.async would fill
                *reinterpret_cast<uint4*>(&S.kv[warp][c][pc * 8]) =
                    comp_unpack8(reinterpret_cast<const uint8_t*>(row), dim0 + pc * 8);
            else
                cp16(&S.kv[warp][c][pc * 8], (ok ? row : raw) + dim0 + pc * 8, ok);
        }
        cp_commit();
    };
    float acc[16][4];   // 2^14 x this warp's output dims, 16 heads
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    issue(row_of(c_begin * kAmCh + lane), c_begin * kAmCh);

    for (int ci = c_begin; ci < c_end; ++ci) {
        const int k0 = ci * kAmCh;
        cp_wait<0>();
        __syncwarp();
        // q.k over this warp's dims: 4 n-tiles of 8 keys, 8 k-steps (two per ldmatrix.x4), hi + lo
        {
            const __half* Kt = &S.kv[warp][0][0];
#pragma unroll
            for (int nt = 0; nt < kAmCh / 8; ++nt) {
                float tg[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kk2 = 0; kk2 < 4; ++kk2) {
                    uint32_t b[4];
                    ldsm_x4(b, Kt + (nt * 8 + (lane & 7)) * kAmStride + kk2 * 32 + (lane >> 3) * 8);
                    mma16816(tg, qh[2 * kk2], b);
                    mma16816(tg, ql[2 * kk2], b);
                    mma16816(tg, qh[2 * kk2 + 1], b + 2);
                    mma16816(tg, ql[2 * kk2 + 1], b + 2);
                }
                const int c = nt * 8 + 2 * tig;
                S.part[warp][gid][c] = tg[0];
                S.part[warp][gid][c + 1] = tg[1];
                S.part[warp][gid + 8][c] = tg[2];
                S.part[warp][gid + 8][c + 1] = tg[3];
            }
        }
        __syncthreads();
        // online softmax over the four warps' sum (fixed order): row tid / 8, 4 keys per thread
        {
            const int r = tid >> 3, sub = tid & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = k0 + c < n_keys ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
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
            const __half* Vt = &S.kv[warp][0][0];
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                acc[j][0] *= a0;
                acc[j][1] *= a0;
                acc[j][2] *= a1;
                acc[j][3] *= a1;
            }
#pragma unroll
            for (int ks = 0; ks < kAmCh / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float p00 = S.p[gid][cA] * 16384.0f, p01 = S.p[gid][cA + 1] * 16384.0f;
                const float p10 = S.p[gid + 8][cA] * 16384.0f, p11 = S.p[gid + 8][cA + 1] * 16384.0f;
                const float p02 = S.p[gid][cB] * 16384.0f, p03 = S.p[gid][cB + 1] * 16384.0f;
                const float p12 = S.p[gid + 8][cB] * 16384.0f, p13 = S.p[gid + 8][cB + 1] * 16384.0f;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const __half2* hh = reinterpret_cast<const __half2*>(ah);
                    float2 f;
                    f = __half22float2(hh[0]);
                    al[0] = pack_h2(p00 - f.x, p01 - f.y);
                    f = __half22float2(hh[1]);
                    al[1] = pack_h2(p10 - f.x, p11 - f.y);
                    f = __half22float2(hh[2]);
                    al[2] = pack_h2(p02 - f.x, p03 - f.y);
                    f = __half22float2(hh[3]);
                    al[3] = pack_h2(p12 - f.x, p13 - f.y);
                }
#pragma unroll
                for (int j2 = 0; j2 < 8; ++j2) {   // two n-tiles (16 dims) per transposed ldmatrix.x4
                    uint32_t b[4];
                    const int m = lane >> 3;
                    ldsm_x4_t(b, Vt + (ks * 16 + (lane & 7) + (m & 1) * 8) * kAmStride + (j2 * 2 + (m >> 1)) * 8);
                    mma16816(acc[2 * j2], ah, b);
                    mma16816(acc[2 * j2], al, b);
                    mma16816(acc[2 * j2 + 1], ah, b + 2);
                    mma16816(acc[2 * j2 + 1], al, b + 2);
                }
            }
        }
        __syncwarp();   // this warp is done with its slice: the next chunk's
        if (ci + 1 < c_end) issue(row_of((ci + 1) * kAmCh + lane), (ci + 1) * kAmCh);
    }
    __syncthreads();
    // rows gid and gid + 8 of this thread: one split folds the sink in (log2 domain) and writes the output; several leave
    // (acc, m, l) in natural units for attn_merge_kernel
    float* const pb = part != nullptr ? part : out;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int r = gid + 8 * half, h = hg * 16 + r;
        const float mr = S.mrow[r], lr = S.lsum[r];
        if (n_split == 1) {
            const float sk = sinks[h] * kLog2e;
            const float M = fmaxf(mr, sk);
            const float a = exp2f(mr - M);
            const float inv = (a / 16384.0f) / (lr * a + exp2f(sk - M));
            float* o = out + ((size_t) t * kAttnHeads + h) * kD;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int d = dim0 + j * 8 + 2 * tig;
                *reinterpret_cast<float2*>(o + d) = make_float2(acc[j][2 * half] * inv, acc[j][2 * half + 1] * inv);
            }
        } else {
            float* pp = pb + (((size_t) t * kAttnHeads + h) * n_split + s) * kPart;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int d = dim0 + j * 8 + 2 * tig;
                *reinterpret_cast<float2*>(pp + d) =
                    make_float2(acc[j][2 * half] * (1.0f / 16384.0f), acc[j][2 * half + 1] * (1.0f / 16384.0f));
            }
            if (warp == 0 && tig == 0) {
                pp[kD] = mr;   // log2 domain: attn_merge_kernel(log2 = true)
                pp[kD + 1] = lr;
            }
        }
    }
    cp_wait<0>();
#else
    __trap();   // built for a card without mma.sync m16n8k16 f16 (sm_80): the host keeps attn_partial_kernel
#endif
}

// ------------------------------------------------------------------ head hyper-connection reduce

/// pre (4 values, thread s < 4): sigmoid(fl(fl(mix * scale) + base)) + eps - dsv4_hc_affine's ggml_mul then ggml_add,
/// ggml_sigmoid (unary.cu:48), ggml_scale_bias(1, eps) (scale.cu: 1 * x + eps); then the four separate ggml_mul and
/// three ggml_add of build_hc_pre(.., -1), each rounded (__fmul_rn / __fadd_rn: no contraction).
__global__ void hc_head_kernel(const float* __restrict__ x, const float* __restrict__ mixes, const float* __restrict__ sc,
                               const float* __restrict__ base, float eps, float* __restrict__ y, int n) {
    __shared__ float pre[4];
    const int t = blockIdx.x;
    if (threadIdx.x < 4) {
        const int s = threadIdx.x;
        const float a = __fadd_rn(__fmul_rn(mixes[(size_t) t * 4 + s], sc[0]), base[s]);
        pre[s] = __fadd_rn(1.0f / (1.0f + expf(-a)), eps);
    }
    __syncthreads();
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* xt = x + (size_t) t * 4 * n;
    float acc = __fmul_rn(xt[i], pre[0]);
    acc = __fadd_rn(acc, __fmul_rn(xt[(size_t) n + i], pre[1]));
    acc = __fadd_rn(acc, __fmul_rn(xt[(size_t) 2 * n + i], pre[2]));
    acc = __fadd_rn(acc, __fmul_rn(xt[(size_t) 3 * n + i], pre[3]));
    y[(size_t) t * n + i] = acc;
}

// ------------------------------------------------------------------ router

/// sqrt(softplus(x)) as topk-moe.cu:72-80 (and unary.cu:93 + sqrtf): NaN becomes -FLT_MAX (topk-moe.cu:145-150).  `exact`
/// (moex, DSV41_BACKLOG #56; DeepSeek-V4.1): torch's softplus, log1p(e^x) - model.py Gate's F.softplus(scores).sqrt() -
/// where V4 keeps topk-moe's log(1 + e^x), which loses the small values (x below ~-9: >0.1% off; below ~-17: 0)
__device__ __forceinline__ float sqrt_softplus(float x, bool exact) {
    const float v = sqrtf(x > 20.0f ? x : exact ? log1pf(expf(x)) : logf(1.0f + expf(x)));
    return isnan(v) ? -FLT_MAX : v;
}

/// A warp per token, expert e = lane + 32 i in register i (EPL per lane) - topk_moe_cuda's layout and arithmetic:
/// k rounds of a lane-local best (strictly greater: the lower expert of the lane) then a butterfly arg-max with ties to
/// the lower expert; the owning lane adds the weight to its partial sum (topk-moe.cu:183-247), the partials summed by
/// warp_reduce_sum.  Hash ids: the weights as the unfused ops (get_rows, sum_rows = the same butterfly over k lanes,
/// clamp, div, scale).
template <int EPL, bool PRE = false>
__global__ void __launch_bounds__(128) route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                    const int32_t* __restrict__ hash_ids, int E, int k, float scale,
                                                    int32_t* __restrict__ ids, float* __restrict__ weights, int T,
                                                    const float* __restrict__ pen, bool exact) {
    const int t = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (t >= T) return;   // warp-uniform
    const int lane = threadIdx.x & 31;
    const float* lg = logits + (size_t) t * E;
    const bool do_scale = scale != 0.0f && scale != 1.0f;   // llama-graph.cpp:2097
    if (hash_ids != nullptr) {
        const int32_t id = lane < k ? hash_ids[(size_t) t * k + lane] : 0;
        const float wv = lane < k ? sqrt_softplus(lg[id], exact) : 0.0f;
        const float sum = fmaxf(warp_sum(wv), 6.103515625e-5f);
        if (lane < k) {
            const float wn = wv / sum;
            ids[(size_t) t * k + lane] = id;
            weights[(size_t) t * k + lane] = do_scale ? wn * scale : wn;
        }
        return;
    }
    float wt[EPL], sw[EPL];
    if (PRE) {   // moex (DSV41_PLAN §34), STRATA_DSV41_PRELOAD=1: every load issued first, then the same expressions
        float lv[EPL], bv[EPL], pv[EPL];
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            const int e = lane + 32 * i;
            const bool ok = e < E;
            lv[i] = ok ? lg[e] : 0.0f;
            bv[i] = ok && bias != nullptr ? bias[e] : 0.0f;
            pv[i] = ok && pen != nullptr ? pen[e] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            const int e = lane + 32 * i;
            const bool ok = e < E;
            wt[i] = ok ? sqrt_softplus(lv[i], exact) : -INFINITY;
            sw[i] = ok ? (bias != nullptr ? wt[i] + bv[i] : wt[i]) - (pen != nullptr ? pv[i] : 0.0f) : -INFINITY;
        }
    } else {
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            const int e = lane + 32 * i;
            const bool ok = e < E;
            wt[i] = ok ? sqrt_softplus(lg[e], exact) : -INFINITY;
            sw[i] = ok ? (bias != nullptr ? wt[i] + bias[e] : wt[i]) - (pen != nullptr ? pen[e] : 0.0f) : -INFINITY;
        }
    }
    float out_w = 0.0f, wsum = 0.0f;
    int out_id = 0;
    for (int r = 0; r < k; ++r) {
        float bv = sw[0], bw = wt[0];
        int be = lane;
#pragma unroll
        for (int i = 1; i < EPL; ++i) {
            const int e = lane + 32 * i;
            if (e < E && sw[i] > bv) {
                bv = sw[i];
                bw = wt[i];
                be = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o), ow = __shfl_xor_sync(0xffffffffu, bw, o);
            const int oe = __shfl_xor_sync(0xffffffffu, be, o);
            if (ov > bv || (ov == bv && oe < be)) {
                bv = ov;
                bw = ow;
                be = oe;
            }
        }
        if ((be & 31) == lane) {
#pragma unroll
            for (int i = 0; i < EPL; ++i)
                if (i == (be >> 5)) sw[i] = -INFINITY;
            wsum += bw;
        }
        if (lane == r) {
            out_w = bw;
            out_id = be;
        }
    }
    const float inv = 1.0f / fmaxf(warp_sum(wsum), 6.103515625e-5f);
    if (lane < k) {
        ids[(size_t) t * k + lane] = out_id;
        const float wn = out_w * inv;
        weights[(size_t) t * k + lane] = do_scale ? wn * scale : wn;
    }
}

/// moex (DSV41_BACKLOG #112), STRATA_DSV41_ROUTE_PUBLISH=1: a decode window's routing and its doorbell publish in one block -
/// warp t < T routes token t with route_kernel's arithmetic (the same expressions in the same order, so the same ids and
/// weights) and writes its ids to the mapped block too, the other warps copy the activations meanwhile; then
/// doorbell_publish_kernel's fences and word.  One launch a MoE layer instead of two, the copy beside the routing
template <int EPL, bool PRE = false>
__global__ void __launch_bounds__(1024) route_publish_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                             const int32_t* __restrict__ hash_ids, int E, int k, float scale,
                                                             int32_t* __restrict__ ids, float* __restrict__ weights, int T,
                                                             const float* __restrict__ pen, bool exact, const float* __restrict__ x,
                                                             int n_x, int32_t* __restrict__ m_ids, float* __restrict__ m_x,
                                                             volatile uint32_t* __restrict__ m_seq, uint32_t* __restrict__ d_cnt) {
    const int t = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (t >= T) {   // the activations' copy (doorbell_publish_kernel's), by the warps that route nothing
        for (int i = threadIdx.x - T * 32; i < n_x; i += blockDim.x - T * 32) m_x[i] = x[i];
    } else {
        const float* lg = logits + (size_t) t * E;
        const bool do_scale = scale != 0.0f && scale != 1.0f;
        if (hash_ids != nullptr) {
            const int32_t id = lane < k ? hash_ids[(size_t) t * k + lane] : 0;
            const float wv = lane < k ? sqrt_softplus(lg[id], exact) : 0.0f;
            const float sum = fmaxf(warp_sum(wv), 6.103515625e-5f);
            if (lane < k) {
                const float wn = wv / sum;
                ids[(size_t) t * k + lane] = id;
                m_ids[(size_t) t * k + lane] = id;
                weights[(size_t) t * k + lane] = do_scale ? wn * scale : wn;
            }
        } else {
            float wt[EPL], sw[EPL];
            if (PRE) {
                float lv[EPL], bv[EPL], pv[EPL];
#pragma unroll
                for (int i = 0; i < EPL; ++i) {
                    const int e = lane + 32 * i;
                    const bool ok = e < E;
                    lv[i] = ok ? lg[e] : 0.0f;
                    bv[i] = ok && bias != nullptr ? bias[e] : 0.0f;
                    pv[i] = ok && pen != nullptr ? pen[e] : 0.0f;
                }
#pragma unroll
                for (int i = 0; i < EPL; ++i) {
                    const int e = lane + 32 * i;
                    const bool ok = e < E;
                    wt[i] = ok ? sqrt_softplus(lv[i], exact) : -INFINITY;
                    sw[i] = ok ? (bias != nullptr ? wt[i] + bv[i] : wt[i]) - (pen != nullptr ? pv[i] : 0.0f) : -INFINITY;
                }
            } else {
#pragma unroll
                for (int i = 0; i < EPL; ++i) {
                    const int e = lane + 32 * i;
                    const bool ok = e < E;
                    wt[i] = ok ? sqrt_softplus(lg[e], exact) : -INFINITY;
                    sw[i] = ok ? (bias != nullptr ? wt[i] + bias[e] : wt[i]) - (pen != nullptr ? pen[e] : 0.0f) : -INFINITY;
                }
            }
            float out_w = 0.0f, wsum = 0.0f;
            int out_id = 0;
            for (int r = 0; r < k; ++r) {
                float bv = sw[0], bw = wt[0];
                int be = lane;
#pragma unroll
                for (int i = 1; i < EPL; ++i) {
                    const int e = lane + 32 * i;
                    if (e < E && sw[i] > bv) {
                        bv = sw[i];
                        bw = wt[i];
                        be = e;
                    }
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    const float ov = __shfl_xor_sync(0xffffffffu, bv, o), ow = __shfl_xor_sync(0xffffffffu, bw, o);
                    const int oe = __shfl_xor_sync(0xffffffffu, be, o);
                    if (ov > bv || (ov == bv && oe < be)) {
                        bv = ov;
                        bw = ow;
                        be = oe;
                    }
                }
                if ((be & 31) == lane) {
#pragma unroll
                    for (int i = 0; i < EPL; ++i)
                        if (i == (be >> 5)) sw[i] = -INFINITY;
                    wsum += bw;
                }
                if (lane == r) {
                    out_w = bw;
                    out_id = be;
                }
            }
            const float inv = 1.0f / fmaxf(warp_sum(wsum), 6.103515625e-5f);
            if (lane < k) {
                ids[(size_t) t * k + lane] = out_id;
                m_ids[(size_t) t * k + lane] = out_id;
                const float wn = out_w * inv;
                weights[(size_t) t * k + lane] = do_scale ? wn * scale : wn;
            }
        }
    }
    __threadfence_system();   // every thread's payload is visible to the host before the word moves
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t v = *d_cnt + 1u;
        *d_cnt = v;
        *m_seq = v;
        __threadfence_system();
    }
}

__global__ void hash_ids_kernel(const int32_t* __restrict__ tab, const int32_t* __restrict__ tok, int k, int64_t n,
                                int32_t* __restrict__ ids) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ids[i] = tab[(size_t) tok[i / k] * k + i % k];
}

}  // namespace

// ------------------------------------------------------------------ the API

Dsv4Rope dsv4_rope_p0(float freq_base) { return Dsv4Rope{freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0}; }

Dsv4Rope dsv4_rope_pc(float compress_base, float freq_scale, float ext_factor, int n_ctx_orig, float beta_fast,
                      float beta_slow) {
    // dsv4_rope_attn_factor (deepseek4.cpp:11-17): 1 / (1 + 0.1 logf(1 / fs)) when ext_factor != 0
    const float af = ext_factor == 0.0f ? 1.0f : 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
    return Dsv4Rope{compress_base, freq_scale, ext_factor, af, beta_fast, beta_slow, n_ctx_orig};
}

void dsv4_rope(const float* x, float* y, int T, int rows, int row_len, int offs, const Dsv4Rope& p, bool inverse,
               int64_t pos0, void* stream, const int64_t* pos_dev) {
    dsv4_rope_shift(x, y, T, rows, row_len, offs, p, inverse, pos0, 0, stream, pos_dev);
}

void dsv4_rope_shift(const float* x, float* y, int T, int rows, int row_len, int offs, const Dsv4Rope& p, bool inverse,
                     int64_t pos0, int64_t shift, void* stream, const int64_t* pos_dev) {
    if (row_len % 2 || offs % 2 || offs < 0 || offs + kRopeDims > row_len)
        throw std::runtime_error("dsv4_rope: needs even row_len and offs, offs + 64 <= row_len");
    if (T <= 0 || rows <= 0) return;
    const bool in_place = x == y;
    const int ppr = in_place ? kRopePairs : row_len / 2;
    const int pair0 = in_place ? offs / 2 : 0;
    const int64_t n_rows = (int64_t) T * rows;
    rope_kernel<<<blocks(n_rows * ppr, 256), 256, 0, st(stream)>>>(x, y, n_rows, rows, row_len, offs, pair0, ppr,
                                                                    rope_consts(p, inverse), pos0, pos_dev, shift);
    check_launch("dsv4_rope");
}

void dsv4_hadamard(const float* x, float* y, int rows, int n, void* stream) {
    if (rows <= 0) return;
    const dim3 bs(32, 4);
    const int nb = blocks(rows, 4);
    const float scale = 1 / sqrtf((float) n);   // fwht.cu: 1 / sqrtf(n); ops.cpp: 1.0f / sqrtf((float) n)
    switch (n) {
        case 64: fwht_kernel<64><<<nb, bs, 0, st(stream)>>>(x, y, rows, scale); break;
        case 128: fwht_kernel<128><<<nb, bs, 0, st(stream)>>>(x, y, rows, scale); break;
        case 256: fwht_kernel<256><<<nb, bs, 0, st(stream)>>>(x, y, rows, scale); break;
        case 512: fwht_kernel<512><<<nb, bs, 0, st(stream)>>>(x, y, rows, scale); break;
        default: throw std::runtime_error("dsv4_hadamard: n must be 64, 128, 256 or 512");
    }
    check_launch("dsv4_hadamard");
}

void dsv4_raw_store(const float* kv, uint16_t* ring, int ring_rows, int64_t pos0, int T, void* stream,
                    const int64_t* pos_dev) {
    if (ring_rows < T) throw std::runtime_error("dsv4_raw_store: the ring must hold the window");
    if (T <= 0) return;
    raw_store_kernel<<<T, 128, 0, st(stream)>>>(kv, reinterpret_cast<__half*>(ring), ring_rows, pos0, pos_dev);
    check_launch("dsv4_raw_store");
}

bool dsv4_norm_rope(float* x, const float* w, int T, int rows, int cols, float eps, int offs, const Dsv4Rope& p, int64_t pos0,
                    void* stream, const int64_t* pos_dev, uint16_t* ring, int ring_rows) {
    static const bool on = [] {   // moex (the hill-climb): STRATA_DSV4_NORM_ROPE=0 - the norm, the rope and the store apart
        const char* e = std::getenv("STRATA_DSV4_NORM_ROPE");
        return !(e && e[0] == '0');
    }();
    if (!on || cols != kD || offs % 2 || offs < 0 || offs + kRopeDims > cols || (ring && (rows != 1 || ring_rows < T))) return false;
    if (T <= 0 || rows <= 0) return true;
    const int64_t n_rows = (int64_t) T * rows;
    if (n_rows > 0x7fffffff) return false;
    if (ring)
        norm_rope_kernel<true><<<(unsigned) n_rows, 256, 0, st(stream)>>>(x, w, cols, eps, rows, offs, rope_consts(p, false), pos0, pos_dev,
                                                                         reinterpret_cast<__half*>(ring), ring_rows);
    else
        norm_rope_kernel<false><<<(unsigned) n_rows, 256, 0, st(stream)>>>(x, w, cols, eps, rows, offs, rope_consts(p, false), pos0,
                                                                          pos_dev, nullptr, 0);
    check_launch("dsv4_norm_rope");
    return true;
}

Dsv4Compressor dsv4_csa() { return Dsv4Compressor{4, 8, 2, 512, false}; }
Dsv4Compressor dsv4_lid() { return Dsv4Compressor{4, 8, 2, 128, true}; }
Dsv4Compressor dsv4_hca() { return Dsv4Compressor{128, 128, 1, 512, false}; }

void dsv4_compress_pool(const Dsv4Compressor& c, const float* akv, const float* gs, const float* ape, const float* ring_kv,
                        const float* ring_sc, const float* norm_w, float eps, const Dsv4Rope& rope, uint16_t* cache,
                        int64_t pos0, int T, void* stream, const int64_t* pos_dev) {
    if (T <= 0) return;
    int n = (T + c.ratio - 1) / c.ratio;   // static: the most rows T consecutive positions complete
    if (!pos_dev) {                        // exact: none completes -> nothing to launch
        const int64_t p_first = pos0 + ((c.ratio - (pos0 + 1) % c.ratio) % c.ratio);
        if (p_first >= pos0 + T) return;
        n = (int) ((pos0 + T - 1 - p_first) / c.ratio + 1);
    }
    const RopeK rk = rope_consts(rope, false);
    __half* ch = reinterpret_cast<__half*>(cache);
    if (c.ratio == 4 && c.coff == 2 && c.d == 512 && !c.hadamard && c.ring >= 8)
        pool_kernel<512, 4, 2, false><<<n, 512, 0, st(stream)>>>(akv, gs, ape, ring_kv, ring_sc, c.ring, norm_w, eps, rk, ch,
                                                                 pos0, pos_dev, T);
    else if (c.ratio == 4 && c.coff == 2 && c.d == 128 && c.hadamard && c.ring >= 8)
        pool_kernel<128, 4, 2, true><<<n, 128, 0, st(stream)>>>(akv, gs, ape, ring_kv, ring_sc, c.ring, norm_w, eps, rk, ch,
                                                                pos0, pos_dev, T);
    else if (c.ratio == 128 && c.coff == 1 && c.d == 512 && !c.hadamard && c.ring >= 128)
        pool_kernel<512, 128, 1, false><<<n, 512, 0, st(stream)>>>(akv, gs, ape, ring_kv, ring_sc, c.ring, norm_w, eps, rk,
                                                                   ch, pos0, pos_dev, T);
    else
        throw std::runtime_error("dsv4_compress_pool: not a DSV4 compressor (CSA, LID or HCA)");
    check_launch("dsv4_compress_pool");
}

void dsv4_compress_persist(const Dsv4Compressor& c, const float* akv, const float* gs, const float* ape, float* ring_kv,
                           float* ring_sc, int64_t pos0, int T, void* stream, const int64_t* pos_dev) {
    if (T <= 0) return;
    const int n_land = std::min(T, c.ring);
    persist_kernel<<<n_land, 256, 0, st(stream)>>>(akv, gs, ape, ring_kv, ring_sc, c.ratio, c.ring, c.coff * c.d, T, n_land,
                                                   pos0, pos_dev);
    check_launch("dsv4_compress_persist");
}

void dsv4_lid_scores(const float* iq, const uint16_t* lid_k, const float* w, float* score, int max_rows, int64_t pos0,
                     int T, void* stream, const int64_t* pos_dev, int ratio, bool packed) {
    if (T <= 0 || max_rows <= 0) return;
    if (ratio <= 0) throw std::runtime_error("dsv4_lid_scores: ratio must be positive");
    int64_t rows = max_rows;   // static: the capacity
    if (!pos_dev) {
        rows = std::min<int64_t>((pos0 + T) / ratio, max_rows);   // the last token sees the most
        if (rows <= 0) return;
    }
    const int tiles = (int) ((rows + kLidRows - 1) / kLidRows);
    const int per_token = std::max(1, std::min(tiles, 544 / T));   // ~8 blocks per SM of a 68-SM card in total
    (packed ? lid_scores_kernel<true> : lid_scores_kernel<false>)<<<dim3(T, per_token), 256, 0, st(stream)>>>(
        iq, reinterpret_cast<const __half*>(lid_k), w, score, max_rows, pos0, pos_dev, ratio);
    check_launch("dsv4_lid_scores");
}

bool dsv4_mma_supported() {
    static const bool off = [] { const char* e = std::getenv("STRATA_DSV4_MMA"); return e && e[0] == '0'; }();
    if (off) return false;
    // per device: compute capability 8+ AND the code this card runs built for sm_80+ (a build for older archs compiles
    // the mma kernels to __trap()); 0 unknown, 1 yes, 2 no
    static int known[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (known[dev] == 0) {
        int major = 0;
        cudaFuncAttributes fa{};
        const bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major >= 8 &&
                        cudaFuncGetAttributes(&fa, attn_mma_kernel<false>) == cudaSuccess && fa.binaryVersion >= 80 &&
                        cudaFuncGetAttributes(&fa, lid_scores_mma_kernel<false>) == cudaSuccess && fa.binaryVersion >= 80;
        if (!ok) cudaGetLastError();
        known[dev] = ok ? 1 : 2;
    }
    return known[dev] == 1;
}

bool dsv4_lid_scores_mma(const float* iq, const uint16_t* lid_k, const float* w, float* score, int max_rows, int64_t pos0,
                         int T, void* stream, const int64_t* pos_dev, int ratio, bool packed, const int32_t* clist,
                         const int32_t* ccount, int ccap, int cbsize, int heads) {
    static const bool lid_off = [] { const char* e = std::getenv("STRATA_DSV4_MMA_LID"); return e && e[0] == '0'; }();   // A/B
    if (!dsv4_mma_supported() || lid_off) return false;
    if (T <= 0 || max_rows <= 0) return true;
    if (ratio <= 0) throw std::runtime_error("dsv4_lid_scores_mma: ratio must be positive");
    if (clist && (!ccount || ccap <= 0 || cbsize <= 0 || kLmRows % cbsize != 0))
        throw std::runtime_error("dsv4_lid_scores_mma: a candidate list needs its counts, a capacity and a block size dividing 64");
    int64_t rows = max_rows;
    if (!pos_dev) {
        rows = std::min<int64_t>((pos0 + T) / ratio, max_rows);
        if (rows <= 0) return true;
    }
    // moex (§56 C): with a candidate list a token's tiles are its candidates', at most ccap blocks of them
    const int tiles = clist ? (int) std::min<int64_t>((ccap + kLmRows / cbsize - 1) / (kLmRows / cbsize), (rows + kLmRows - 1) / kLmRows)
                            : (int) ((rows + kLmRows - 1) / kLmRows);
    // ~2 blocks per SM in total (68-SM card): a block's q fragments (64 KB of loads) amortize over many tiles
    const int per_token = std::max(1, std::min(tiles, 136 / T));
    const __half* K = reinterpret_cast<const __half*>(lid_k);
    const dim3 grid(T, per_token);
    if (heads != kLidHeads && heads != 32) throw std::runtime_error("dsv4_lid_scores_mma: 64 heads, or V4.1's 32");
    if (heads == 32) {   // moex (§59 P2): the tile without the zero heads' half
        if (clist)
            (packed ? lid_scores_mma_kernel<true, true, true> : lid_scores_mma_kernel<false, true, true>)<<<grid, 128, 0,
                                                                                                           st(stream)>>>(
                iq, K, w, score, max_rows, pos0, pos_dev, ratio, clist, ccount, ccap, cbsize);
        else
            (packed ? lid_scores_mma_kernel<true, false, true> : lid_scores_mma_kernel<false, false, true>)<<<grid, 128, 0,
                                                                                                             st(stream)>>>(
                iq, K, w, score, max_rows, pos0, pos_dev, ratio, nullptr, nullptr, 0, 1);
    } else if (clist)
        (packed ? lid_scores_mma_kernel<true, true> : lid_scores_mma_kernel<false, true>)<<<grid, 128, 0, st(stream)>>>(
            iq, K, w, score, max_rows, pos0, pos_dev, ratio, clist, ccount, ccap, cbsize);
    else
        (packed ? lid_scores_mma_kernel<true> : lid_scores_mma_kernel<false>)<<<grid, 128, 0, st(stream)>>>(
            iq, K, w, score, max_rows, pos0, pos_dev, ratio, nullptr, nullptr, 0, 1);
    check_launch("dsv4_lid_scores_mma");
    return true;
}

void dsv4_lid_select(const float* score, int max_rows, int top_k, int32_t* sel, int32_t* count, int64_t pos0, int T,
                     void* stream, const int64_t* pos_dev, int ratio, int excl) {
    if (top_k <= 0 || max_rows <= 0) throw std::runtime_error("dsv4_lid_select: top_k and max_rows must be positive");
    if (ratio <= 0) throw std::runtime_error("dsv4_lid_select: ratio must be positive");
    if (excl < 0) throw std::runtime_error("dsv4_lid_select: excl must not be negative");
    if (T <= 0) return;
    lid_select_kernel<<<T, 1024, 0, st(stream)>>>(score, max_rows, top_k, sel, count, pos0, pos_dev, ratio, excl);
    check_launch("dsv4_lid_select");
}

int dsv4_attn_splits(int T) {
    // moex (#36c), opt-in: STRATA_DSV4_ATTN_SPLITS=N splits every decode window's keys N ways, whatever its rows (1 .. 8) - a
    // row's attention then sums its keys in the same order in a window of 8 as alone (the default below gives a single step 32
    // parts and a window of 8 four: the same products, added up in another order).  A prompt sub-batch keeps the default: its
    // split-K scratch is sized for that (stage.cpp, at most 31 rows)
    static const int fixed = [] {
        const char* e = std::getenv("STRATA_DSV4_ATTN_SPLITS");
        return e ? std::clamp(std::atoi(e), 1, 32) : 0;
    }();
    if (fixed > 0 && T <= 8) return fixed;
    // ~128 blocks of (split, 16 heads, token) in total, measured best on a 68-SM card (both kernels, T = 1..8, 640 to
    // 8320 keys: more splits only add partials to merge); 1 from T = 32 on (no scratch)
    return std::max(1, std::min(32, 128 / (4 * std::max(T, 1))));
}

std::size_t dsv4_attn_scratch_bytes(int T) {
    const int s = dsv4_attn_splits(T);
    return s > 1 ? (std::size_t) T * kAttnHeads * s * kPart * sizeof(float) : 0;
}

void dsv4_attn(const float* q, const uint16_t* raw, int raw_rows, const uint16_t* comp, int ratio, const int32_t* sel,
               int top_k, const float* sinks, float scale, float* out, float* scratch, int64_t pos0, int T, void* stream,
               const int64_t* pos_dev, int noncausal, bool packed) {
    if (ratio != 0 && ratio != 1 && ratio != 2 && ratio != 4 && ratio != 128)
        throw std::runtime_error("dsv4_attn: ratio must be 0, 1, 2 (DeepSeek-V4.1), 4 or 128");
    if (ratio > 0 && ratio != 128 && (sel == nullptr || top_k <= 0))
        throw std::runtime_error("dsv4_attn: ratios 1, 2 and 4 attend the selected rows: the selection is needed");
    if (ratio != 0 && comp == nullptr) throw std::runtime_error("dsv4_attn: no compressed cache");
    if (raw_rows < T + kAttnWin - 1) throw std::runtime_error("dsv4_attn: the raw ring must hold T + 127 rows");
    if (T <= 0) return;
    const int n_split = dsv4_attn_splits(T);
    if (n_split > 1 && scratch == nullptr) throw std::runtime_error("dsv4_attn: no split-K scratch");
    (packed ? attn_partial_kernel<true> : attn_partial_kernel<false>)<<<dim3(n_split, kAttnHeads / kAttnHg, T), 256, 0, st(stream)>>>(
        q, reinterpret_cast<const __half*>(raw), raw_rows, reinterpret_cast<const __half*>(comp), ratio, sel, top_k, sinks,
        scale, out, scratch, n_split, pos0, pos_dev, noncausal);
    check_launch("dsv4_attn partial");
    if (n_split > 1) {
        attn_merge_kernel<<<dim3(kAttnHeads, T), 256, 0, st(stream)>>>(scratch, sinks, out, ratio, top_k, n_split, false, pos0,
                                                                       pos_dev, noncausal);
        check_launch("dsv4_attn merge");
    }
}

bool dsv4_attn_mma(const float* q, const uint16_t* raw, int raw_rows, const uint16_t* comp, int ratio, const int32_t* sel,
                   int top_k, const float* sinks, float scale, float* out, float* scratch, int64_t pos0, int T, void* stream,
                   const int64_t* pos_dev, int noncausal, bool packed) {
    static const bool attn_off = [] { const char* e = std::getenv("STRATA_DSV4_MMA_ATTN"); return e && e[0] == '0'; }();   // A/B
    if (!dsv4_mma_supported() || attn_off) return false;
    if (ratio != 0 && ratio != 1 && ratio != 2 && ratio != 4 && ratio != 128)
        throw std::runtime_error("dsv4_attn_mma: ratio must be 0, 1, 2 (DeepSeek-V4.1), 4 or 128");
    if (ratio > 0 && ratio != 128 && (sel == nullptr || top_k <= 0))
        throw std::runtime_error("dsv4_attn_mma: ratios 1, 2 and 4 attend the selected rows: the selection is needed");
    if (ratio != 0 && comp == nullptr) throw std::runtime_error("dsv4_attn_mma: no compressed cache");
    if (raw_rows < T + kAttnWin - 1) throw std::runtime_error("dsv4_attn_mma: the raw ring must hold T + 127 rows");
    if (T <= 0) return true;
    const int n_split = dsv4_attn_splits(T);
    if (n_split > 1 && scratch == nullptr) throw std::runtime_error("dsv4_attn_mma: no split-K scratch");
    (packed ? attn_mma_kernel<true> : attn_mma_kernel<false>)<<<dim3(n_split, kAttnHeads / 16, T), kAmThreads, 0, st(stream)>>>(
        q, reinterpret_cast<const __half*>(raw), raw_rows, reinterpret_cast<const __half*>(comp), ratio, sel, top_k, sinks,
        scale, out, n_split > 1 ? scratch : nullptr, n_split, pos0, pos_dev, noncausal);
    check_launch("dsv4_attn_mma");
    if (n_split > 1) {
        attn_merge_kernel<<<dim3(kAttnHeads, T), 256, 0, st(stream)>>>(scratch, sinks, out, ratio, top_k, n_split, true, pos0,
                                                                       pos_dev, noncausal);
        check_launch("dsv4_attn_mma merge");
    }
    return true;
}

void dsv4_hc_head(const float* x, const float* mixes, const float* hc_scale, const float* hc_base, float eps, float* y,
                  int n, int T, void* stream) {
    if (T <= 0 || n <= 0) return;
    hc_head_kernel<<<dim3(T, blocks(n, 256)), 256, 0, st(stream)>>>(x, mixes, hc_scale, hc_base, eps, y, n);
    check_launch("dsv4_hc_head");
}

void dsv4_route(const float* logits, const float* bias, const int32_t* hash_ids, int E, int k, float scale, int32_t* ids,
                float* weights, int T, void* stream, const float* pen, bool exact_softplus) {
    if (E < 1 || E > 1024 || k < 1 || k > 32 || k > E) throw std::runtime_error("dsv4_route: 1 <= k <= 32, k <= E <= 1024");
    if (T <= 0) return;
    const int epl = (E + 31) / 32;
    const int nb = blocks(T, 4);
    static const bool pre = [] {   // moex (DSV41_PLAN §34): DeepSeek-V4.1's 384 experts (12 a lane) with their loads first
        const char* e = std::getenv("STRATA_DSV41_PRELOAD");
        return e && e[0] == '1';
    }();
    if (epl <= 1) route_kernel<1><<<nb, 128, 0, st(stream)>>>(logits, bias, hash_ids, E, k, scale, ids, weights, T, pen, exact_softplus);
    else if (epl <= 2) route_kernel<2><<<nb, 128, 0, st(stream)>>>(logits, bias, hash_ids, E, k, scale, ids, weights, T, pen, exact_softplus);
    else if (epl <= 4) route_kernel<4><<<nb, 128, 0, st(stream)>>>(logits, bias, hash_ids, E, k, scale, ids, weights, T, pen, exact_softplus);
    else if (epl <= 8) route_kernel<8><<<nb, 128, 0, st(stream)>>>(logits, bias, hash_ids, E, k, scale, ids, weights, T, pen, exact_softplus);
    else if (epl <= 16 && pre) route_kernel<16, true><<<nb, 128, 0, st(stream)>>>(logits, bias, hash_ids, E, k, scale, ids, weights, T, pen, exact_softplus);
    else if (epl <= 16) route_kernel<16><<<nb, 128, 0, st(stream)>>>(logits, bias, hash_ids, E, k, scale, ids, weights, T, pen, exact_softplus);
    else route_kernel<32><<<nb, 128, 0, st(stream)>>>(logits, bias, hash_ids, E, k, scale, ids, weights, T, pen, exact_softplus);
    check_launch("dsv4_route");
}

bool dsv4_route_publish(const float* logits, const float* bias, const int32_t* hash_ids, int E, int k, float scale, int32_t* ids,
                        float* weights, int T, void* stream, const float* pen, bool exact_softplus, const float* x, int n_x,
                        int32_t* m_ids, float* m_x, volatile uint32_t* m_seq, uint32_t* d_cnt) {
    if (E < 1 || E > 1024 || k < 1 || k > 32 || k > E) throw std::runtime_error("dsv4_route_publish: 1 <= k <= 32, k <= E <= 1024");
    if (T <= 0 || T > 16) return false;   // (16 routing warps at most: the copy keeps half the block)
    const int epl = (E + 31) / 32;
    static const bool pre = [] {
        const char* e = std::getenv("STRATA_DSV41_PRELOAD");
        return e && e[0] == '1';
    }();
#define RP_ARGS logits, bias, hash_ids, E, k, scale, ids, weights, T, pen, exact_softplus, x, n_x, m_ids, m_x, m_seq, d_cnt
    if (epl <= 1) route_publish_kernel<1><<<1, 1024, 0, st(stream)>>>(RP_ARGS);
    else if (epl <= 2) route_publish_kernel<2><<<1, 1024, 0, st(stream)>>>(RP_ARGS);
    else if (epl <= 4) route_publish_kernel<4><<<1, 1024, 0, st(stream)>>>(RP_ARGS);
    else if (epl <= 8) route_publish_kernel<8><<<1, 1024, 0, st(stream)>>>(RP_ARGS);
    else if (epl <= 16 && pre) route_publish_kernel<16, true><<<1, 1024, 0, st(stream)>>>(RP_ARGS);
    else if (epl <= 16) route_publish_kernel<16><<<1, 1024, 0, st(stream)>>>(RP_ARGS);
    else route_publish_kernel<32><<<1, 1024, 0, st(stream)>>>(RP_ARGS);
#undef RP_ARGS
    check_launch("dsv4_route_publish");
    return true;
}

void dsv4_hash_ids(const int32_t* tid2eid, const int32_t* tokens, int k, int T, int32_t* ids, void* stream) {
    if (T <= 0 || k <= 0) return;
    const int64_t n = (int64_t) T * k;
    hash_ids_kernel<<<blocks(n, 256), 256, 0, st(stream)>>>(tid2eid, tokens, k, n, ids);
    check_launch("dsv4_hash_ids");
}

}  // namespace strata::x
