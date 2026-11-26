// src/x/dsv41_kernels.cu - moex: DeepSeek-V4.1-Flash's attention-side kernels beyond V4's (include/strata/x/dsv41_kernels.hpp).
#include "strata/x/dsv41_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>

namespace strata::x {
namespace {

cudaStream_t st(void* s) { return static_cast<cudaStream_t>(s); }

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

int blocks_of(int64_t n, int per) { return (int) ((n + per - 1) / per); }

__device__ __forceinline__ int64_t pos_of(const int64_t* __restrict__ p, int64_t arg) { return p ? *p : arg; }

/// 2^ceil(log2(v)) for positive normal v, on the bits (kernel.py fast_log2_ceil / fast_pow2).
__device__ __forceinline__ float pow2_ceil_log2(float v) {
    const uint32_t b = __float_as_uint(v);
    const int e = (int) ((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
    return __uint_as_float((uint32_t) (e + 127) << 23);
}

/// E2M1 round to nearest, ties to even, for v in [-6, 6].
__device__ __forceinline__ float round_e2m1(float v) {
    const float a = fabsf(v);
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return copysignf(r, v);
}

/// One thread per value; the 32 lanes of a warp are one block of 32 values (rows contiguous, width % 32 == 0).
__global__ void fq_fp8_kernel(float* __restrict__ x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const float v = i < n ? x[i] : 0.0f;
    float a = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if (i >= n) return;
    a = fmaxf(a, 1e-4f);
    const float s = pow2_ceil_log2(a * (1.0f / 448.0f));
    const float q = fminf(fmaxf(v / s, -448.0f), 448.0f);
    const __nv_fp8_e4m3 f(q);   // round to nearest even (the value is in range)
    x[i] = (float) f * s;
}

/// The same with blocks of `block` (16 or 32) lanes.
template <int BLOCK>
__global__ void fq_fp4_kernel(float* __restrict__ x, int64_t n, bool e4m3_scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const float v = i < n ? x[i] : 0.0f;
    float a = fabsf(v);
#pragma unroll
    for (int o = BLOCK / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if (i >= n) return;
    float s;
    if (e4m3_scale) {
        a = fmaxf(a, 6.0f * 0.001953125f);   // 6 * 2^-9: a zero group keeps a nonzero scale (training's compressed KV)
        s = (float) __nv_fp8_e4m3(a / 6.0f);
    } else {
        a = fmaxf(a, 6.0f * 1.1754943508222875e-38f);   // 6 * 2^-126
        s = pow2_ceil_log2(a * (1.0f / 6.0f));
    }
    x[i] = round_e2m1(fminf(fmaxf(v / s, -6.0f), 6.0f)) * s;
}

/// moex (DSV41_BACKLOG #104): the E2M1 code of a value round_e2m1 returned - magnitudes 0 .5 1 1.5 2 3 4 6 as 0 .. 7, bit 3
/// the sign (-0 kept)
__device__ __forceinline__ uint32_t e2m1_code(float q) {
    const float a = fabsf(q);
    const uint32_t m = a <= 1.5f ? (uint32_t) (a * 2.0f) : a <= 2.0f ? 4u : a <= 3.0f ? 5u : a <= 4.0f ? 6u : 7u;
    return m | ((__float_as_uint(q) >> 31) << 3);
}

/// fq_fp4_kernel's rounding - the same expressions, the floats written back - and the codes and the scale of each row that
/// completes a cache row (dsv41_store_rows's: ratio 1 every position, ratio 2 the odd ones) packed into it: width / 2 bytes
/// of codes (value 2i in the low nibble), then width / BLOCK scale bytes (E4M3, or UE8M0: the scale's F32 exponent byte)
template <int BLOCK>
__global__ void fq_fp4_pack_kernel(float* __restrict__ x, int64_t n, int width, bool e4m3_scale, uint8_t* __restrict__ cache,
                                   int row_bytes, int cache_rows, int ratio, int64_t pos0_arg, const int64_t* __restrict__ pos_ptr) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const float v = i < n ? x[i] : 0.0f;
    float a = fabsf(v);
#pragma unroll
    for (int o = BLOCK / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float s;
    uint32_t sbyte;
    if (e4m3_scale) {
        a = fmaxf(a, 6.0f * 0.001953125f);
        const __nv_fp8_e4m3 f8(a / 6.0f);
        s = (float) f8;
        sbyte = (uint32_t) f8.__x;
    } else {
        a = fmaxf(a, 6.0f * 1.1754943508222875e-38f);
        s = pow2_ceil_log2(a * (1.0f / 6.0f));
        sbyte = (__float_as_uint(s) >> 23) & 0xFFu;
    }
    const float r = round_e2m1(fminf(fmaxf(v / s, -6.0f), 6.0f));
    const uint32_t code = e2m1_code(r), odd = __shfl_xor_sync(0xffffffffu, code, 1);
    if (i >= n) return;
    x[i] = r * s;
    const int64_t row = i / width;
    const int col = (int) (i % width);
    const int64_t p = pos_of(pos_ptr, pos0_arg) + row;
    if ((p + 1) % ratio != 0) return;
    const int64_t b = p / ratio;
    if (b >= cache_rows) return;
    uint8_t* dst = cache + (size_t) b * row_bytes;
    if ((col & 1) == 0) dst[col >> 1] = (uint8_t) (code | (odd << 4));
    if (col % BLOCK == 0) dst[width / 2 + col / BLOCK] = (uint8_t) sbyte;
}

constexpr int kCompThreads = 256;

/// Block t: token t's latent (D = 512, 2 channels per thread).
template <int D>
__global__ void __launch_bounds__(kCompThreads) compress_kernel(int ratio, const float* __restrict__ akv, const float* __restrict__ gs,
                                                               const float* __restrict__ ring_kv, const float* __restrict__ ring_sc,
                                                               int ring_rows, const float* __restrict__ norm_w, float eps,
                                                               float* __restrict__ latent, int64_t pos0_arg,
                                                               const int64_t* __restrict__ pos_ptr) {
    constexpr int PER = D / kCompThreads;
    __shared__ float red[kCompThreads / 32];
    const int t = blockIdx.x;
    const int64_t pos0 = pos_of(pos_ptr, pos0_arg), p = pos0 + t;
    float* out = latent + (size_t) t * D;
    if (ratio == 2 && (p & 1) == 0) {   // no row completes here
#pragma unroll
        for (int i = 0; i < PER; ++i) out[threadIdx.x + i * kCompThreads] = 0.0f;
        return;
    }
    float c[PER];
    const float* kv1 = akv + (size_t) t * D;
    if (ratio == 1) {
#pragma unroll
        for (int i = 0; i < PER; ++i) c[i] = kv1[threadIdx.x + i * kCompThreads];
    } else {
        const int64_t s0 = p - 1;   // the block's first slot: in this window or in the ring
        const float *kv0, *sc0;
        if (s0 >= pos0) {
            kv0 = akv + (size_t) (s0 - pos0) * D;
            sc0 = gs + (size_t) (s0 - pos0) * D;
        } else {
            const int64_t r = s0 % ring_rows;
            kv0 = ring_kv + (size_t) r * D;
            sc0 = ring_sc + (size_t) r * D;
        }
        const float* sc1 = gs + (size_t) t * D;
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int k = threadIdx.x + i * kCompThreads;
            const float a0 = sc0[k], a1 = sc1[k], m = fmaxf(a0, a1);
            const float e0 = expf(a0 - m), e1 = expf(a1 - m), inv = 1.0f / (e0 + e1);
            c[i] = kv0[k] * (e0 * inv) + kv1[k] * (e1 * inv);
        }
    }
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < PER; ++i) ss += c[i] * c[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < kCompThreads / 32 ? red[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float r = rsqrtf(red[0] / (float) D + eps);
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int k = threadIdx.x + i * kCompThreads;
        out[k] = c[i] * r * norm_w[k];
    }
}

__global__ void persist_kernel(const float* __restrict__ akv, const float* __restrict__ gs, float* __restrict__ ring_kv,
                               float* __restrict__ ring_sc, int ring_rows, int width, int T, int64_t pos0_arg,
                               const int64_t* __restrict__ pos_ptr) {
    const int t = blockIdx.y;
    if (t < T - ring_rows) return;   // a later token takes the row
    const int64_t p = pos_of(pos_ptr, pos0_arg) + t;
    const int64_t r = p % ring_rows;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < width; k += gridDim.x * blockDim.x) {
        ring_kv[(size_t) r * width + k] = akv[(size_t) t * width + k];
        ring_sc[(size_t) r * width + k] = gs[(size_t) t * width + k];
    }
}

__global__ void store_rows_kernel(int ratio, const float* __restrict__ rows, __half* __restrict__ cache, int width,
                                  int cache_rows, int64_t pos0_arg, const int64_t* __restrict__ pos_ptr) {
    const int t = blockIdx.y;
    const int64_t p = pos_of(pos_ptr, pos0_arg) + t;
    if ((p + 1) % ratio != 0) return;
    const int64_t b = p / ratio;
    if (b >= cache_rows) return;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < width; k += gridDim.x * blockDim.x)
        cache[(size_t) b * width + k] = __float2half_rn(rows[(size_t) t * width + k]);
}

/// float -> an unsigned key in the same order (larger float, larger key).
__device__ __forceinline__ uint32_t okey(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

constexpr int kCandThreads = 256;

/// Block t: token t's candidate blocks, one bit each.  The fast path (every block kept) covers contexts up to keep * bsize rows;
/// beyond, a binary search over the keys finds the keep-th largest block maximum, then the ties at it go to the lower blocks.
__global__ void __launch_bounds__(kCandThreads) candidates_kernel(const float* __restrict__ score, int max_rows, int ratio, int keep,
                                                                 int bsize, uint8_t* __restrict__ cand, int cand_bytes,
                                                                 int64_t pos0_arg, const int64_t* __restrict__ pos_ptr) {
    __shared__ uint32_t s_cnt;
    const int t = blockIdx.x;
    const int64_t p = pos_of(pos_ptr, pos0_arg) + t;
    const int n_vis = (int) min((p + 1) / ratio, (int64_t) max_rows);
    const int nblk = (n_vis + bsize - 1) / bsize, last = n_vis > 0 ? (n_vis - 1) / bsize : -1;
    uint8_t* c = cand + (size_t) t * cand_bytes;
    const float* sc = score + (size_t) t * max_rows;
    if (nblk <= keep) {
        for (int i = threadIdx.x; i < cand_bytes; i += blockDim.x) {
            const int lo = i * 8;
            c[i] = lo + 8 <= nblk ? 0xFF : lo >= nblk ? 0 : (uint8_t) ((1u << (nblk - lo)) - 1u);
        }
        return;
    }
    auto bkey = [&](int b) -> uint32_t {   // the block's best visible row; the newest block pinned in
        if (b == last) return 0xFFFFFFFFu;
        float m = -INFINITY;
        const int j1 = min(n_vis, (b + 1) * bsize);
        for (int j = b * bsize; j < j1; ++j) m = fmaxf(m, sc[j]);
        return okey(m);
    };
    uint32_t lo = 0, hi = 0xFFFFFFFFu;   // find the largest thr with count(key >= thr) >= keep
    while (lo < hi) {
        const uint32_t mid = (uint32_t) (((uint64_t) lo + hi + 1) >> 1);   // in 64 bits: hi - lo + 1 wraps at 2^32
        if (threadIdx.x == 0) s_cnt = 0;
        __syncthreads();
        uint32_t local = 0;
        for (int b = threadIdx.x; b < nblk; b += blockDim.x) local += bkey(b) >= mid ? 1u : 0u;
        atomicAdd(&s_cnt, local);
        __syncthreads();
        const bool ok = s_cnt >= (uint32_t) keep;
        __syncthreads();
        if (ok) lo = mid;
        else hi = mid - 1;
    }
    const uint32_t thr = lo;
    if (threadIdx.x == 0) s_cnt = 0;
    __syncthreads();
    uint32_t gt = 0;
    for (int i = threadIdx.x; i < cand_bytes; i += blockDim.x) {   // the blocks above the threshold, a byte per thread at a time
        uint32_t byte = 0;
        for (int k = 0; k < 8; ++k) {
            const int b = i * 8 + k;
            if (b < nblk && bkey(b) > thr) byte |= 1u << k;
        }
        c[i] = (uint8_t) byte;
        gt += (uint32_t) __popc(byte);
    }
    atomicAdd(&s_cnt, gt);
    __syncthreads();
    const int need = keep - (int) s_cnt;   // >= 1: the ties at the threshold that are kept, lower blocks first
    if (need == 1) {
        // moex (DSV41_BACKLOG #54): one tie - distinct block maxima, the usual case - found by every thread at once: the
        // lowest block at the threshold, the block the serial walk below would take.  That walk, alone on thread 0, read about
        // half the blocks in turn (~0.1-0.2 us each: est. 1.8-2.8 ms a decode window at 262K positions)
        __shared__ int s_low;
        if (threadIdx.x == 0) s_low = INT_MAX;
        __syncthreads();
        for (int b = threadIdx.x; b < nblk; b += blockDim.x)   // a thread's blocks in order: its first tie is its lowest
            if (bkey(b) == thr) {
                atomicMin(&s_low, b);
                break;
            }
        __syncthreads();
        if (threadIdx.x == 0 && s_low != INT_MAX) c[s_low >> 3] |= (uint8_t) (1u << (s_low & 7));
    } else if (threadIdx.x == 0) {   // several ties at the threshold, lower blocks first
        int left = need;
        for (int b = 0; b < nblk && left > 0; ++b)
            if (bkey(b) == thr) {
                c[b >> 3] |= (uint8_t) (1u << (b & 7));
                --left;
            }
    }
}

__global__ void cand_mask_kernel(float* __restrict__ score, int max_rows, int ratio, const uint8_t* __restrict__ cand,
                                 int cand_bytes, int bsize, int64_t pos0_arg, const int64_t* __restrict__ pos_ptr) {
    const int t = blockIdx.y;
    const int64_t p = pos_of(pos_ptr, pos0_arg) + t;
    const int n_vis = (int) min((p + 1) / ratio, (int64_t) max_rows);
    const uint8_t* c = cand + (size_t) t * cand_bytes;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_vis; j += gridDim.x * blockDim.x) {
        const int b = j / bsize;
        if (!((c[b >> 3] >> (b & 7)) & 1)) score[(size_t) t * max_rows + j] = -INFINITY;
    }
}

/// moex (DSV41_PLAN §56 C): block t - token t's kept visible blocks in ascending order: each thread a run of bytes, their bits
/// counted, an exclusive scan over the threads, then each thread writes its blocks from its offset.
constexpr int kCandListThreads = 256;
__global__ void __launch_bounds__(kCandListThreads) cand_list_kernel(const uint8_t* __restrict__ cand, int cand_bytes, int max_rows,
                                                                     int ratio, int bsize, int32_t* __restrict__ list,
                                                                     int32_t* __restrict__ count, int cap, int64_t pos0_arg,
                                                                     const int64_t* __restrict__ pos_ptr) {
    __shared__ int s_scan[kCandListThreads];
    const int t = blockIdx.x;
    const int64_t p = pos_of(pos_ptr, pos0_arg) + t;
    const int n_vis = (int) min((p + 1) / ratio, (int64_t) max_rows);
    const int nblk = n_vis > 0 ? (n_vis + bsize - 1) / bsize : 0;
    const int nbytes = min((nblk + 7) / 8, cand_bytes);
    const uint8_t* c = cand + (size_t) t * cand_bytes;
    const int per = (nbytes + kCandListThreads - 1) / kCandListThreads;
    const int b0 = min(nbytes, (int) threadIdx.x * per), b1 = min(nbytes, b0 + per);
    auto byte_at = [&](int i) -> uint32_t {   // the bits of visible blocks only
        uint32_t v = c[i];
        const int lo = i * 8;
        if (lo + 8 > nblk) v &= (1u << (nblk - lo)) - 1u;
        return v;
    };
    int mine = 0;
    for (int i = b0; i < b1; ++i) mine += __popc(byte_at(i));
    s_scan[threadIdx.x] = mine;
    __syncthreads();
    for (int off = 1; off < kCandListThreads; off <<= 1) {   // inclusive scan
        const int v = (int) threadIdx.x >= off ? s_scan[threadIdx.x - off] : 0;
        __syncthreads();
        s_scan[threadIdx.x] += v;
        __syncthreads();
    }
    int at = s_scan[threadIdx.x] - mine;
    int32_t* out = list + (size_t) t * cap;
    for (int i = b0; i < b1; ++i)
        for (uint32_t v = byte_at(i); v != 0; v &= v - 1u, ++at)
            if (at < cap) out[at] = i * 8 + (__ffs((int) v) - 1);
    if (threadIdx.x == 0) count[t] = s_scan[kCandListThreads - 1];
}

__global__ void engram_dequant_kernel(const uint8_t* __restrict__ rows, const uint8_t* __restrict__ scales, float* __restrict__ out,
                                      int64_t n, int dim) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / dim, d = i % dim;
    __nv_fp8_e4m3 f;
    f.__x = rows[i];
    const int sb = scales[r * (dim / 32) + d / 32];
    const float sc = sb == 255 ? __int_as_float(0x7fc00000) : ldexpf(1.0f, sb - 127);   // E8M0: 0xFF is NaN
    out[i] = (float) f * sc;
}

// moex: an IQ4_NL engram table's rows (unsloth's UD-IQ1_M / UD-IQ1_S): row r at rows + r * stride, dim / 32 blocks of
// [fp16 d, 16 bytes: value j's index in byte j's low nibble, value j + 16's in its high one] - ggml's dequantize_row_iq4_nl,
// d * kvalues_iq4nl[index] in F32 (exact: an 11-bit d times a 7-bit integer)
__constant__ int8_t kEgIq4nlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__global__ void engram_dequant_iq4nl_kernel(const uint8_t* __restrict__ rows, size_t stride, float* __restrict__ out, int64_t n,
                                            int dim) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / dim, e = i % dim;
    const uint8_t* blk = rows + (size_t) r * stride + (size_t) (e / 32) * 18;
    const int j = (int) (e % 32);
    __half_raw hr;
    hr.x = (unsigned short) (blk[0] | (blk[1] << 8));
    const float d = __half2float(__half(hr));
    const int q = blk[2 + (j & 15)];
    out[i] = d * (float) kEgIq4nlValues[j < 16 ? (q & 0x0F) : (q >> 4)];
}

constexpr int kEgThreads = 256;

/// Block (copy c, token t): the three sums over n, then the copy's update.
__global__ void __launch_bounds__(kEgThreads) engram_gate_kernel(float* __restrict__ R, const float* __restrict__ kv,
                                                               const float* __restrict__ qk, float eps, int n, int hc) {
    __shared__ float red[3][kEgThreads / 32];
    __shared__ float s_gate;
    const int c = blockIdx.x, t = blockIdx.y;
    float* h = R + ((size_t) t * hc + c) * n;
    const float* key = kv + (size_t) t * (hc + 1) * n + (size_t) c * n;
    const float* val = kv + (size_t) t * (hc + 1) * n + (size_t) hc * n;
    const float* w = qk + (size_t) c * n;
    float hh = 0.0f, kk = 0.0f, hk = 0.0f;
    for (int d = threadIdx.x; d < n; d += kEgThreads) {
        const float a = h[d], b = key[d];
        hh += a * a;
        kk += b * b;
        hk += a * w[d] * b;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        hh += __shfl_xor_sync(0xffffffffu, hh, o);
        kk += __shfl_xor_sync(0xffffffffu, kk, o);
        hk += __shfl_xor_sync(0xffffffffu, hk, o);
    }
    if ((threadIdx.x & 31) == 0) {
        red[0][threadIdx.x >> 5] = hh;
        red[1][threadIdx.x >> 5] = kk;
        red[2][threadIdx.x >> 5] = hk;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.0f, b = 0.0f, x = 0.0f;
        for (int i = 0; i < kEgThreads / 32; ++i) {
            a += red[0][i];
            b += red[1][i];
            x += red[2][i];
        }
        const float rstd = rsqrtf(a / (float) n + eps) * rsqrtf(b / (float) n + eps);
        const float dot = x * rstd * rsqrtf((float) n);
        const float g = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
        s_gate = 1.0f / (1.0f + expf(-g));
    }
    __syncthreads();
    const float g = s_gate;
    for (int d = threadIdx.x; d < n; d += kEgThreads) h[d] += g * val[d];
}

__global__ void imx_sumsq_kernel(const float* __restrict__ x, int64_t ld, const int32_t* __restrict__ rowmap,
                                 const int32_t* __restrict__ bounds, const int32_t* __restrict__ experts, int cols,
                                 float* __restrict__ acc) {
    const int j = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float s = 0.0f;
    for (int r = bounds[j]; r < bounds[j + 1]; ++r) {
        const float v = x[(size_t) (rowmap ? rowmap[r] : r) * ld + c];
        s += v * v;
    }
    acc[(size_t) experts[j] * cols + c] += s;
}

__global__ void pre_onehot_kernel(float* __restrict__ pre, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pre[i] = i % 24 == 0 ? 1.0f : 0.0f;
}

}  // namespace

void dsv41_fq_fp8(float* x, int64_t rows, int width, void* stream) {
    if (width % 32) throw std::runtime_error("dsv41_fq_fp8: width must be a multiple of 32");
    const int64_t n = rows * width;
    if (n <= 0) return;
    fq_fp8_kernel<<<blocks_of(n, 256), 256, 0, st(stream)>>>(x, n);
    check_launch("dsv41_fq_fp8");
}

void dsv41_fq_fp4(float* x, int64_t rows, int width, int block, bool e4m3_scale, void* stream) {
    if ((block != 16 && block != 32) || width % block) throw std::runtime_error("dsv41_fq_fp4: block 16 or 32 dividing width");
    const int64_t n = rows * width;
    if (n <= 0) return;
    if (block == 16) fq_fp4_kernel<16><<<blocks_of(n, 256), 256, 0, st(stream)>>>(x, n, e4m3_scale);
    else fq_fp4_kernel<32><<<blocks_of(n, 256), 256, 0, st(stream)>>>(x, n, e4m3_scale);
    check_launch("dsv41_fq_fp4");
}

void dsv41_fq_fp4_pack(float* x, int64_t rows, int width, int block, bool e4m3_scale, uint8_t* cache, int cache_rows, int ratio,
                       int64_t pos0, void* stream, const int64_t* pos_dev) {
    if ((block != 16 && block != 32) || width % 32) throw std::runtime_error("dsv41_fq_fp4_pack: block 16 or 32, width a multiple of 32");
    if (ratio != 1 && ratio != 2) throw std::runtime_error("dsv41_fq_fp4_pack: ratio 1 or 2");
    const int64_t n = rows * width;
    if (n <= 0) return;
    const int rb = (int) dsv41_pack_row_bytes(width, block);
    if (block == 16)
        fq_fp4_pack_kernel<16><<<blocks_of(n, 256), 256, 0, st(stream)>>>(x, n, width, e4m3_scale, cache, rb, cache_rows, ratio, pos0, pos_dev);
    else
        fq_fp4_pack_kernel<32><<<blocks_of(n, 256), 256, 0, st(stream)>>>(x, n, width, e4m3_scale, cache, rb, cache_rows, ratio, pos0, pos_dev);
    check_launch("dsv41_fq_fp4_pack");
}

void dsv41_compress(int ratio, const float* akv, const float* gs, const float* ring_kv, const float* ring_sc, int ring_rows,
                    const float* norm_w, float eps, float* latent, int64_t pos0, int T, void* stream, const int64_t* pos_dev) {
    if (ratio != 1 && ratio != 2) throw std::runtime_error("dsv41_compress: ratio 1 or 2");
    if (ratio == 2 && (gs == nullptr || ring_kv == nullptr || ring_sc == nullptr || ring_rows <= 0))
        throw std::runtime_error("dsv41_compress: ratio 2 needs the gate scores and the state ring");
    if (T <= 0) return;
    compress_kernel<512><<<T, kCompThreads, 0, st(stream)>>>(ratio, akv, gs, ring_kv, ring_sc, ring_rows, norm_w, eps, latent,
                                                             pos0, pos_dev);
    check_launch("dsv41_compress");
}

void dsv41_compress_persist(const float* akv, const float* gs, float* ring_kv, float* ring_sc, int ring_rows, int64_t pos0,
                            int T, void* stream, const int64_t* pos_dev) {
    if (T <= 0) return;
    persist_kernel<<<dim3(2, T), 256, 0, st(stream)>>>(akv, gs, ring_kv, ring_sc, ring_rows, 512, T, pos0, pos_dev);
    check_launch("dsv41_compress_persist");
}

void dsv41_store_rows(int ratio, const float* rows, uint16_t* cache, int width, int cache_rows, int64_t pos0, int T,
                      void* stream, const int64_t* pos_dev) {
    if (ratio != 1 && ratio != 2) throw std::runtime_error("dsv41_store_rows: ratio 1 or 2");
    if (T <= 0) return;
    store_rows_kernel<<<dim3(blocks_of(width, 256), T), 256, 0, st(stream)>>>(ratio, rows, reinterpret_cast<__half*>(cache), width,
                                                                              cache_rows, pos0, pos_dev);
    check_launch("dsv41_store_rows");
}

void dsv41_candidates(const float* score, int max_rows, int ratio, int keep, int bsize, uint8_t* cand, int cand_bytes,
                      int64_t pos0, int T, void* stream, const int64_t* pos_dev) {
    if (ratio <= 0 || keep <= 0 || bsize <= 0) throw std::runtime_error("dsv41_candidates: positive ratio, keep, bsize");
    if (cand_bytes < dsv41_cand_bytes(max_rows, bsize)) throw std::runtime_error("dsv41_candidates: rows of candidate bits too short");
    if (T <= 0) return;
    candidates_kernel<<<T, kCandThreads, 0, st(stream)>>>(score, max_rows, ratio, keep, bsize, cand, cand_bytes, pos0, pos_dev);
    check_launch("dsv41_candidates");
}

void dsv41_cand_mask(float* score, int max_rows, int ratio, const uint8_t* cand, int cand_bytes, int bsize, int64_t pos0,
                     int T, void* stream, const int64_t* pos_dev) {
    if (ratio <= 0 || bsize <= 0) throw std::runtime_error("dsv41_cand_mask: positive ratio and bsize");
    if (T <= 0) return;
    const int gx = std::min(blocks_of(max_rows, 256), 64);
    cand_mask_kernel<<<dim3(gx, T), 256, 0, st(stream)>>>(score, max_rows, ratio, cand, cand_bytes, bsize, pos0, pos_dev);
    check_launch("dsv41_cand_mask");
}

void dsv41_cand_list(const uint8_t* cand, int cand_bytes, int max_rows, int ratio, int bsize, int32_t* list, int32_t* count,
                     int cap, int64_t pos0, int T, void* stream, const int64_t* pos_dev) {
    if (ratio <= 0 || bsize <= 0 || cap <= 0) throw std::runtime_error("dsv41_cand_list: positive ratio, bsize and cap");
    if (cand_bytes < dsv41_cand_bytes(max_rows, bsize)) throw std::runtime_error("dsv41_cand_list: rows of candidate bits too short");
    if (T <= 0) return;
    cand_list_kernel<<<T, kCandListThreads, 0, st(stream)>>>(cand, cand_bytes, max_rows, ratio, bsize, list, count, cap, pos0,
                                                             pos_dev);
    check_launch("dsv41_cand_list");
}

void dsv41_engram_dequant(const uint8_t* rows, const uint8_t* scales, float* out, int64_t n_rows, int dim, void* stream) {
    if (dim % 32) throw std::runtime_error("dsv41_engram_dequant: dim must be a multiple of 32");
    const int64_t n = n_rows * dim;
    if (n <= 0) return;
    engram_dequant_kernel<<<blocks_of(n, 256), 256, 0, st(stream)>>>(rows, scales, out, n, dim);
    check_launch("dsv41_engram_dequant");
}

void dsv41_engram_dequant_iq4nl(const uint8_t* rows, size_t row_stride, float* out, int64_t n_rows, int dim, void* stream) {
    if (dim % 32 || row_stride < (size_t) dim / 32 * 18)
        throw std::runtime_error("dsv41_engram_dequant_iq4nl: dim must be a multiple of 32, the stride at least a row");
    const int64_t n = n_rows * dim;
    if (n <= 0) return;
    engram_dequant_iq4nl_kernel<<<blocks_of(n, 256), 256, 0, st(stream)>>>(rows, row_stride, out, n, dim);
    check_launch("dsv41_engram_dequant_iq4nl");
}

void dsv41_engram_gate(float* R, const float* kv, const float* qk, float eps, int n, int hc, int T, void* stream) {
    if (T <= 0) return;
    engram_gate_kernel<<<dim3(hc, T), kEgThreads, 0, st(stream)>>>(R, kv, qk, eps, n, hc);
    check_launch("dsv41_engram_gate");
}

void dsv41_imx_sumsq(const float* x, int64_t ld, const int32_t* rowmap, const int32_t* bounds, const int32_t* experts, int n_exp,
                     int cols, float* acc, void* stream) {
    if (n_exp <= 0 || cols <= 0) return;
    imx_sumsq_kernel<<<dim3(blocks_of(cols, 256), n_exp), 256, 0, st(stream)>>>(x, ld, rowmap, bounds, experts, cols, acc);
    check_launch("dsv41_imx_sumsq");
}

void dsv41_pre_onehot(float* pre, int T, void* stream) {
    if (T <= 0) return;
    pre_onehot_kernel<<<blocks_of((int64_t) T * 24, 256), 256, 0, st(stream)>>>(pre, T * 24);
    check_launch("dsv41_pre_onehot");
}

}  // namespace strata::x
