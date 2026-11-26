// src/x/hy4_vision.cu - moex: Hy4-flash's image encoder on the GPU (include/strata/x/hy4_vision.hpp).  serve/vision_hy4.py
// is the CPU encoder (torch, fp32) this follows step by step; checks/hy4_vision_parity.py holds the two against each other.
// The structure is DeepSeek-V4.1's (src/x/dsv41_vision.cu): the weights streamed a piece at a time through two staging
// buffers, the linear layers as K-blocked SGEMMs, the attention a block of queries at a time with each score row's softmax in
// shared memory.
#include "strata/x/hy4_vision.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::x {
namespace {

using V = Hy4Vision;
constexpr int kQkv = 3 * V::kDim;        // 3,456: q | k | v, each 16 heads of 72
constexpr int kChunk = 512;              // the projector's weights go up 512 output rows at a time
// A long sum in blocks of 128 terms (dsv41_vision.cu: 6.0e-6 against float64 there).  Here, a 640 x 480 picture against the
// CPU encoder's float64 rows: 1.54e-6 (its worst row 1.86e-5; the fp32 CPU encoder 2.66e-6); the attention's p v as one sum
// over every key 2.51e-6 (worst row 7.28e-5) for 9% less time at 1080p (3.61 -> 3.28 s) - kept at 128
constexpr int kSumBlock = 128;
constexpr size_t kScoreBytes = 160ull << 20;   // a block of queries' scores, every head
constexpr float kLnEps = 1e-5f;          // the blocks' LayerNorm and the projector's RMS norm (rms_norm_eps)
constexpr float kSqrtHalf = 0.70710678118654752440f;   // torch's M_SQRT1_2 in float (F.gelu, approximate "none")
static_assert(V::kMergeHid % kChunk == 0 && V::kOut % kChunk == 0, "the projector's chunks");

// THE HOST LAYOUT (bytes).  Piece P0 (f32): the patch embedding and its bias, the projector's norm and biases, the newline
// row.  The position grid (f32, the host's).  Then the 27 blocks, each contiguous: its bf16 matrices, then its f32 vectors
// (one copy a block).  Then the projector's two weights (f32, kChunk output rows a copy).
constexpr size_t kPeW = 0, kPeB = kPeW + (size_t) V::kDim * V::kPatchIn, kPre = kPeB + V::kDim, kB0 = kPre + V::kDim,
                 kB2 = kB0 + V::kMergeHid, kNl = kB2 + V::kOut, kP0Floats = kNl + V::kOut;
constexpr size_t kGridFloats = (size_t) V::kGrid * V::kGrid * V::kDim;
// a block's bf16 matrices (elements) ...
constexpr size_t kWqkv = 0, kWo = kWqkv + (size_t) kQkv * V::kDim, kWup = kWo + (size_t) V::kDim * V::kDim,
                 kWdown = kWup + (size_t) V::kInter * V::kDim, kMats = kWdown + (size_t) V::kDim * V::kInter;
// ... then its f32 vectors (floats)
constexpr size_t kLn1w = 0, kLn1b = kLn1w + V::kDim, kBqkv = kLn1b + V::kDim, kBo = kBqkv + kQkv, kLn2w = kBo + V::kDim,
                 kLn2b = kLn2w + V::kDim, kBup = kLn2b + V::kDim, kBdown = kBup + V::kInter, kVecs = kBdown + V::kDim;
constexpr size_t kBlockBytes = kMats * 2 + kVecs * 4;
constexpr size_t kChunkBytes = (size_t) kChunk * V::kMergeIn * 4;
constexpr size_t kHostP0 = 0, kHostGrid = kHostP0 + kP0Floats * 4, kHostBlocks = kHostGrid + kGridFloats * 4,
                 kHostM0 = kHostBlocks + (size_t) V::kLayers * kBlockBytes, kHostM2 = kHostM0 + (size_t) V::kMergeHid * V::kMergeIn * 4,
                 kHostBytes = kHostM2 + (size_t) V::kOut * V::kMergeHid * 4;
constexpr int kM0Chunks = V::kMergeHid / kChunk, kM2Chunks = V::kOut / kChunk;
constexpr int kPieces = V::kLayers + kM0Chunks + kM2Chunks;   // the blocks, then the projector's chunks
constexpr size_t kStageBytes = std::max(kBlockBytes, kChunkBytes);
constexpr size_t kWfFloats = kMats + kVecs;   // a block widened: its matrices, then its vectors
static_assert(kMats * 2 % 16 == 0, "the vectors' alignment in a block");

size_t al256(size_t b) { return (b + 255) & ~(size_t) 255; }

/// moex (2026-11-19): the attention through k_attn_fused (on; STRATA_HY4_VENC_FUSED=0: the SGEMM path, its score buffer)
bool venc_fused() {
    static const bool on = [] { const char* e = std::getenv("STRATA_HY4_VENC_FUSED"); return !(e && e[0] == '0'); }();
    return on;
}

/// The workspace of one picture: byte offsets into it
struct Plan {
    int64_t n = 0, L = 0;
    int qc = 0;   ///< queries a block of the attention
    size_t x = 0, t = 0, big = 0, sc = 0, st[2] = {0, 0}, wf = 0, w0 = 0, total = 0;
};

Plan plan_of(int gh, int gw) {
    Plan p;
    if (gh < V::kMerge || gw < V::kMerge || gh % V::kMerge || gw % V::kMerge || (int64_t) gh * gw > V::kMaxPatches) return p;
    p.n = (int64_t) gh * gw;
    p.L = p.n / (V::kMerge * V::kMerge);
    const size_t n = (size_t) p.n, L = (size_t) p.L;
    int qc = (int) std::min<size_t>(n, kScoreBytes / (sizeof(float) * V::kHeads * n));
    if (qc < (int) n) qc = std::max(32, qc / 32 * 32);
    p.qc = qc;
    // big holds in turn: the patches, q | k | v, the MLP's hidden rows, then the projector's cells, hidden and output rows
    const size_t big = std::max({n * V::kPatchIn, n * kQkv, n * V::kInter, L * (size_t) (V::kMergeIn + V::kMergeHid + V::kOut)});
    size_t o = 0;
    auto take = [&](size_t bytes) {
        const size_t at = o;
        o += al256(bytes);
        return at;
    };
    p.x = take(n * V::kDim * sizeof(float));
    p.t = take(n * V::kDim * sizeof(float));
    p.big = take(big * sizeof(float));
    p.sc = take(venc_fused() ? 0 : (size_t) V::kHeads * (size_t) qc * n * sizeof(float));   // (the fused attention keeps none)
    p.st[0] = take(kStageBytes);
    p.st[1] = take(kStageBytes);
    p.wf = take(kWfFloats * sizeof(float));
    p.w0 = take(kP0Floats * sizeof(float));
    p.total = o;
    return p;
}

int grid_of(int64_t n, int threads) { return (int) std::min<int64_t>((n + threads - 1) / threads, 65535); }

// ------------------------------------------------------------------------------------------------ the kernels

__global__ void k_widen(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        out[i] = __uint_as_float((uint32_t) in[i] << 16);
}

/// y = y + b per row (F.linear's bias), in place
__global__ void k_bias(float* __restrict__ y, const float* __restrict__ b, int64_t rows, int width) {
    const int64_t n = rows * width;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = __fadd_rn(y[i], b[i % width]);
}

/// x = x + (t + b) (the block's residual over a biased linear), x = x + t without a bias
__global__ void k_residual(float* __restrict__ x, const float* __restrict__ t, const float* __restrict__ b, int64_t rows, int width) {
    const int64_t n = rows * width;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] = __fadd_rn(x[i], b ? __fadd_rn(t[i], b[i % width]) : t[i]);
}

/// The sum over a block of 384 threads, in a fixed order (a warp's butterfly, then the 12 warps in turn); every thread gets it
__device__ float sum384(float v, float* red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < 12; ++i) t = __fadd_rn(t, red[i]);
    __syncthreads();   // red reused by the next call
    return t;
}

/// LayerNorm (F.layer_norm, eps 1e-5) on rows of kDim: mean, the biased variance about it, (x - mean) / sqrt(var + eps) * w + b
__global__ void __launch_bounds__(384) k_layernorm(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                                                   float* __restrict__ y) {
    __shared__ float red[12];
    const float* xr = x + (int64_t) blockIdx.x * V::kDim;
    float v[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) v[k] = xr[threadIdx.x + 384 * k];
    const float mean = __fdiv_rn(sum384(__fadd_rn(__fadd_rn(v[0], v[1]), v[2]), red), (float) V::kDim);
    float d[3], ss = 0.0f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        d[k] = __fsub_rn(v[k], mean);
        ss = __fadd_rn(ss, __fmul_rn(d[k], d[k]));
    }
    const float var = __fdiv_rn(sum384(ss, red), (float) V::kDim);
    const float rs = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(var, kLnEps)));
    float* yr = y + (int64_t) blockIdx.x * V::kDim;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int j = threadIdx.x + 384 * k;
        yr[j] = __fadd_rn(__fmul_rn(__fmul_rn(d[k], rs), w[j]), b[j]);
    }
}

/// The projector's RMS norm on rows of kDim: x / sqrt(mean(x^2) + eps), then times w
__global__ void __launch_bounds__(384) k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y) {
    __shared__ float red[12];
    const float* xr = x + (int64_t) blockIdx.x * V::kDim;
    float v[3], ss = 0.0f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        v[k] = xr[threadIdx.x + 384 * k];
        ss = __fadd_rn(ss, __fmul_rn(v[k], v[k]));
    }
    const float rs = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(sum384(ss, red), (float) V::kDim), kLnEps)));
    float* yr = y + (int64_t) blockIdx.x * V::kDim;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int j = threadIdx.x + 384 * k;
        yr[j] = __fmul_rn(__fmul_rn(v[k], rs), w[j]);
    }
}

/// Rows of n scores, in place: exp(s - max) / sum.  A block a row, the row in shared memory (one read, one write), the sum
/// in a fixed order (each thread's stride, a warp's butterfly, the 16 warps' butterfly) - dsv41_vision.cu's.
__global__ void __launch_bounds__(512) k_softmax(float* __restrict__ s, int n) {
    extern __shared__ float row[];
    __shared__ float red[16];
    __shared__ float res;
    float* g = s + (int64_t) blockIdx.x * n;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += 512) {
        const float v = g[i];
        row[i] = v;
        m = fmaxf(m, v);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < 16 ? red[threadIdx.x] : -INFINITY;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) res = v;
    }
    __syncthreads();
    const float mx = res;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += 512) {
        const float e = expf(__fsub_rn(row[i], mx));
        row[i] = e;
        sum = __fadd_rn(sum, e);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    __syncthreads();   // every thread read `res`
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < 16 ? red[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) res = v;
    }
    __syncthreads();
    const float tot = res;
    for (int i = threadIdx.x; i < n; i += 512) g[i] = __fdiv_rn(row[i], tot);
}

/// moex (2026-11-19): THE ATTENTION FUSED (STRATA_HY4_VENC_FUSED=0: the SGEMMs and k_softmax) - o = softmax(q k^T * scale) v
/// for every head in one launch: a block 128 queries of one head (8 warps, 16 queries a warp: 4 a lane, a group of 8 lanes a
/// group of 4 queries), the keys 64 at a time through shared memory, FlashAttention's online softmax in fp32 - the scores
/// never leave the SM (the SGEMM path writes, normalizes and reads back ~466 GB of them at 1080p, its attention 75% of the
/// encode).  The sums keep that path's two levels: a key tile's p v summed apart in key order, then added to the running
/// output after its rescale (one sum over every key had 4x the worst row's error there); a score is its 72 products in
/// order, times the scale; the row's denominator folded the same way (each lane's 8 p's in order, the 8 lanes' butterfly);
/// the output divided once, at the end.
constexpr int kFq = 128, kFk = 64, kFd = V::kHeadDim, kSkStride = kFd + 4;   // (76: the 8 lanes' float4 reads on 32 banks)
constexpr size_t kFusedSmem = (size_t) (kFq * kFd + kFk * kSkStride + kFk * kFd) * sizeof(float);   // 74,752 B
static_assert(kFd == 72 && V::kDim % 4 == 0 && kQkv % 4 == 0, "the fused attention's lanes: 9 dims each (4 + 4 + 1)");
__global__ void __launch_bounds__(256) k_attn_fused(const float* __restrict__ qkv, float* __restrict__ out, int n, float scale) {
    extern __shared__ __align__(16) float fsm[];
    float* sq = fsm;                    // [kFq][kFd]
    float* sk = sq + kFq * kFd;         // [kFk][kSkStride]
    float* sv = sk + kFk * kSkStride;   // [kFk][kFd]
    const int h = (int) blockIdx.y;
    const int64_t q0 = (int64_t) blockIdx.x * kFq;
    const int tid = (int) threadIdx.x, warp = tid >> 5, lane = tid & 31, lk = lane & 7;
    const int qr0 = warp * 16 + (lane >> 3) * 4;   // this lane's 4 queries (rows of sq)
    constexpr int F4 = kFd / 4;                    // 18 float4 a row
    for (int e = tid; e < kFq * F4; e += 256) {
        const int r = e / F4, c = e % F4;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (q0 + r < n) v = reinterpret_cast<const float4*>(qkv + (size_t) (q0 + r) * kQkv + h * kFd)[c];
        reinterpret_cast<float4*>(sq + r * kFd)[c] = v;
    }
    float o[4][9], m[4], l[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
#pragma unroll
        for (int c = 0; c < 9; ++c) o[i][c] = 0.0f;
    }
    for (int64_t k0 = 0; k0 < n; k0 += kFk) {
        const int nk = n - k0 < kFk ? (int) (n - k0) : kFk;
        __syncthreads();   // (the queries stored; every warp done with the last tile)
        for (int e = tid; e < kFk * F4; e += 256) {
            const int r = e / F4, c = e % F4;
            float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f), vv = kv;
            if (r < nk) {
                const float* row = qkv + (size_t) (k0 + r) * kQkv + h * kFd;
                kv = reinterpret_cast<const float4*>(row + V::kDim)[c];
                vv = reinterpret_cast<const float4*>(row + 2 * V::kDim)[c];
            }
            reinterpret_cast<float4*>(sk + r * kSkStride)[c] = kv;
            reinterpret_cast<float4*>(sv + r * kFd)[c] = vv;
        }
        __syncthreads();
        // the scores: the lane's 4 queries x the keys lk + 8 jj
        float s[4][8];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) s[i][jj] = 0.0f;
#pragma unroll 2
        for (int d4 = 0; d4 < F4; ++d4) {
            float4 qa[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) qa[i] = reinterpret_cast<const float4*>(sq + (qr0 + i) * kFd)[d4];
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const float4 kb = reinterpret_cast<const float4*>(sk + (lk + 8 * jj) * kSkStride)[d4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    s[i][jj] = fmaf(qa[i].x, kb.x, s[i][jj]);
                    s[i][jj] = fmaf(qa[i].y, kb.y, s[i][jj]);
                    s[i][jj] = fmaf(qa[i].z, kb.z, s[i][jj]);
                    s[i][jj] = fmaf(qa[i].w, kb.w, s[i][jj]);
                }
            }
        }
        // the online softmax: the tile's row maximum, its p's (in s), the denominator
        float corr[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float mx = -INFINITY;
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                s[i][jj] = lk + 8 * jj < nk ? __fmul_rn(s[i][jj], scale) : -INFINITY;
                mx = fmaxf(mx, s[i][jj]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            const float mn = fmaxf(m[i], mx);
            corr[i] = expf(__fsub_rn(m[i], mn));   // (the first tile: exp(-inf) = 0)
            float ls = 0.0f;
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                s[i][jj] = expf(__fsub_rn(s[i][jj], mn));   // (a key past the end: 0)
                ls = __fadd_rn(ls, s[i][jj]);
            }
            ls = __fadd_rn(ls, __shfl_xor_sync(0xffffffffu, ls, 1));
            ls = __fadd_rn(ls, __shfl_xor_sync(0xffffffffu, ls, 2));
            ls = __fadd_rn(ls, __shfl_xor_sync(0xffffffffu, ls, 4));
            l[i] = __fadd_rn(__fmul_rn(l[i], corr[i]), ls);
            m[i] = mn;
        }
        // the tile's p v, summed apart in key order (key src + 8 jj: its p's from lane (group, src)); the lane's dims
        // 4 lk .. 4 lk + 3, 32 + 4 lk .. 32 + 4 lk + 3 and 64 + lk
        float tv[4][9];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int c = 0; c < 9; ++c) tv[i][c] = 0.0f;
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
            const int n_src = nk - 8 * jj < 8 ? nk - 8 * jj : 8;   // (uniform: the tile's keys in order; none past its end)
#pragma unroll 2
            for (int src = 0; src < n_src; ++src) {
                const int key = src + 8 * jj;
                float pv[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) pv[i] = __shfl_sync(0xffffffffu, s[i][jj], (lane & ~7) | src);
                const float* vr = sv + key * kFd;
                const float4 va = reinterpret_cast<const float4*>(vr)[lk];
                const float4 vb = reinterpret_cast<const float4*>(vr + 32)[lk];
                const float vc = vr[64 + lk];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    tv[i][0] = fmaf(pv[i], va.x, tv[i][0]);
                    tv[i][1] = fmaf(pv[i], va.y, tv[i][1]);
                    tv[i][2] = fmaf(pv[i], va.z, tv[i][2]);
                    tv[i][3] = fmaf(pv[i], va.w, tv[i][3]);
                    tv[i][4] = fmaf(pv[i], vb.x, tv[i][4]);
                    tv[i][5] = fmaf(pv[i], vb.y, tv[i][5]);
                    tv[i][6] = fmaf(pv[i], vb.z, tv[i][6]);
                    tv[i][7] = fmaf(pv[i], vb.w, tv[i][7]);
                    tv[i][8] = fmaf(pv[i], vc, tv[i][8]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int c = 0; c < 9; ++c) o[i][c] = __fadd_rn(__fmul_rn(o[i][c], corr[i]), tv[i][c]);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int64_t qi = q0 + qr0 + i;
        if (qi >= n) continue;
        float* orow = out + (size_t) qi * V::kDim + h * kFd;
        reinterpret_cast<float4*>(orow)[lk] = make_float4(__fdiv_rn(o[i][0], l[i]), __fdiv_rn(o[i][1], l[i]),
                                                          __fdiv_rn(o[i][2], l[i]), __fdiv_rn(o[i][3], l[i]));
        reinterpret_cast<float4*>(orow + 32)[lk] = make_float4(__fdiv_rn(o[i][4], l[i]), __fdiv_rn(o[i][5], l[i]),
                                                               __fdiv_rn(o[i][6], l[i]), __fdiv_rn(o[i][7], l[i]));
        orow[64 + lk] = __fdiv_rn(o[i][8], l[i]);
    }
}

/// h = gelu(h + b) per row: torch's tanh form (0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))), its operation order) or,
/// with `erf`, F.gelu's (x * 0.5) * (1 + erf(x * sqrt(1/2)))
__global__ void k_bias_gelu(float* __restrict__ h, const float* __restrict__ b, int64_t rows, int width, bool erf) {
    const float kBeta = (float) (1.41421356237309504880 * 1.12837916709551257390 * 0.5), kKappa = 0.044715f;
    const int64_t n = rows * width;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float v = __fadd_rn(h[i], b[i % width]);
        if (erf) {
            h[i] = __fmul_rn(__fmul_rn(v, 0.5f), __fadd_rn(1.0f, erff(__fmul_rn(v, kSqrtHalf))));
        } else {
            const float cube = __fmul_rn(__fmul_rn(v, v), v);
            const float inner = __fmul_rn(kBeta, __fadd_rn(v, __fmul_rn(kKappa, cube)));
            h[i] = __fmul_rn(__fmul_rn(0.5f, v), __fadd_rn(1.0f, tanhf(inner)));
        }
    }
}

/// The final rows (gh x gw patches of kDim) -> the merged cells ((gh / 2) x (gw / 2) of kMergeIn): cell (oy, ox)'s column
/// c * 4 + kh * 2 + kw is patch (2 oy + kh, 2 ox + kw)'s value c (the 2 x 2 conv's weight order)
__global__ void k_merge(const float* __restrict__ t, float* __restrict__ u, int gw, int64_t cells) {
    const int ow = gw / V::kMerge;
    const int64_t n = cells * V::kMergeIn;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t l = i / V::kMergeIn;
        const int col = (int) (i % V::kMergeIn), c = col >> 2, kh = (col >> 1) & 1, kw = col & 1;
        const int y = 2 * (int) (l / ow) + kh, x = 2 * (int) (l % ow) + kw;
        u[i] = t[((int64_t) y * gw + x) * V::kDim + c];
    }
}

// ------------------------------------------------------------------------------------------------ the host side

struct Guard {   // the encode's end, however it ends: the streams drained, the events gone, the handle's math mode back
    cublasHandle_t h = nullptr;
    cudaStream_t s = nullptr, cs = nullptr;
    cublasMath_t math = CUBLAS_DEFAULT_MATH;
    bool math_set = false;
    cudaEvent_t ev[4] = {nullptr, nullptr, nullptr, nullptr};
    ~Guard() {
        if (s) cudaStreamSynchronize(s);
        if (cs) cudaStreamSynchronize(cs);
        for (cudaEvent_t e : ev)
            if (e) cudaEventDestroy(e);
        if (math_set) cublasSetMathMode(h, math);
    }
};

bool cuda_ok(const char* what, std::string& err) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    err = std::string("the image encoder on the GPU: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

bool blas_ok(cublasStatus_t st, const char* what, std::string& err) {
    if (st == CUBLAS_STATUS_SUCCESS) return true;
    err = std::string("the image encoder on the GPU: cuBLAS refused ") + what + " (status " + std::to_string((int) st) + ")";
    return false;
}

/// y (rows x out, row stride ldy) = x (rows x in, row stride ldx) W^T, W (out x in) row-major, row stride ldw - the sum over
/// `in` in blocks of kSumBlock (the first block's product written, each later one's added: beta 1)
bool linear(cublasHandle_t h, const float* x, int ldx, const float* w, int ldw, int in, int out, int64_t rows, float* y, int ldy,
            const char* what, std::string& err) {
    const float one = 1.0f, zero = 0.0f;
    for (int k0 = 0; k0 < in; k0 += kSumBlock) {
        const int kb = std::min(kSumBlock, in - k0);
        if (!blas_ok(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, out, (int) rows, kb, &one, w + k0, ldw, x + k0, ldx, k0 ? &one : &zero, y,
                                 ldy),
                     what, err))
            return false;
    }
    return true;
}

/// The learned grid resampled to gh x gw (F.interpolate(scale_factor=((gh + 0.1) / 128, (gw + 0.1) / 128), mode="bilinear",
/// align_corners=False) on the channels-last view vision.py makes): per axis the source coordinate (1 / sf) (i + 0.5) - 0.5 in
/// float, clamped at 0, its floor and the next (the last index repeated), and the fractions; each value the four corners'
/// weighted sum, x00 w00 + (x01 w01 + (x10 w10 + x11 w11)), w00 = lh0 lw0 ...
void resample_grid(const float* grid, int gh, int gw, float* out) {
    struct Axis {
        int i0, i1;
        float l0, l1;
    };
    auto axis = [](int n_out) {
        std::vector<Axis> a((size_t) n_out);
        const float scale = (float) (1.0 / (((double) n_out + 0.1) / (double) V::kGrid));
        for (int i = 0; i < n_out; ++i) {
            float src = scale * ((float) i + 0.5f) - 0.5f;
            if (src < 0.0f) src = 0.0f;
            const int i0 = std::min((int) std::floor(src), V::kGrid - 1);
            const float l1 = std::min(std::max(src - (float) i0, 0.0f), 1.0f);
            a[(size_t) i] = {i0, i0 + (i0 < V::kGrid - 1 ? 1 : 0), 1.0f - l1, l1};
        }
        return a;
    };
    const std::vector<Axis> ah = axis(gh), aw = axis(gw);
    for (int y = 0; y < gh; ++y)
        for (int x = 0; x < gw; ++x) {
            const Axis& h = ah[(size_t) y];
            const Axis& w = aw[(size_t) x];
            const float w00 = h.l0 * w.l0, w01 = h.l0 * w.l1, w10 = h.l1 * w.l0, w11 = h.l1 * w.l1;
            const float* p00 = grid + ((size_t) h.i0 * V::kGrid + w.i0) * V::kDim;
            const float* p01 = grid + ((size_t) h.i0 * V::kGrid + w.i1) * V::kDim;
            const float* p10 = grid + ((size_t) h.i1 * V::kGrid + w.i0) * V::kDim;
            const float* p11 = grid + ((size_t) h.i1 * V::kGrid + w.i1) * V::kDim;
            float* o = out + ((size_t) y * gw + x) * V::kDim;
            for (int c = 0; c < V::kDim; ++c) o[c] = p00[c] * w00 + (p01[c] * w01 + (p10[c] * w10 + p11[c] * w11));
        }
}

}  // namespace

Hy4Vision::Hy4Vision(HostAlloc alloc, HostFree free) : alloc_(std::move(alloc)), free_(std::move(free)) {
    const char* g = std::getenv("STRATA_HY4_VISION_GELU");
    gelu_erf_ = g && std::strcmp(g, "erf") == 0;
}

Hy4Vision::~Hy4Vision() {
    if (host_ && free_) free_(host_);
}

size_t Hy4Vision::workspace_bytes(int gh, int gw) { return plan_of(gh, gw).total; }

bool Hy4Vision::load(const TensorSource& src, std::string& err) {
    if (host_) return true;
    uint8_t* host = static_cast<uint8_t*>(alloc_ ? alloc_(kHostBytes) : nullptr);
    if (!host) {
        err = "the image encoder's weights: " + std::to_string(kHostBytes >> 20) + " MiB of pinned memory refused";
        return false;
    }
    auto fail = [&](const std::string& why) {
        free_(host);
        err = "the image encoder's weights (the mmproj): " + why;
        return false;
    };
    // one tensor into `dst`: its GGUF type (0 F32, 30 BF16) and shape (dim 0 fastest) as the encoder takes them
    auto take = [&](const std::string& name, uint32_t want_type, std::vector<uint64_t> want_shape, uint8_t* dst) -> bool {
        uint32_t type = 0;
        std::vector<uint64_t> shape;
        const uint8_t* data = nullptr;
        if (!src(name, type, shape, data) || !data) return fail("it has no " + name);
        if (type != want_type || shape != want_shape) {
            std::string s;
            for (uint64_t d : shape) s += (s.empty() ? "" : " x ") + std::to_string(d);
            return fail(name + " is ggml type " + std::to_string(type) + " [" + s + "], not the encoder's");
        }
        uint64_t elems = 1;
        for (uint64_t d : shape) elems *= d;
        std::memcpy(dst, data, (size_t) elems * (type == 30 ? 2 : 4));
        return true;
    };
    float* p0 = reinterpret_cast<float*>(host + kHostP0);
    const uint64_t D = kDim;
    if (!take("v.patch_embd.weight", 0, {kPatch, kPatch, 3, D}, reinterpret_cast<uint8_t*>(p0 + kPeW)) ||
        !take("v.patch_embd.bias", 0, {D}, reinterpret_cast<uint8_t*>(p0 + kPeB)) ||
        !take("mm.pre_norm.weight", 0, {D}, reinterpret_cast<uint8_t*>(p0 + kPre)) ||
        !take("mm.0.bias", 0, {(uint64_t) kMergeHid}, reinterpret_cast<uint8_t*>(p0 + kB0)) ||
        !take("mm.2.bias", 0, {(uint64_t) kOut}, reinterpret_cast<uint8_t*>(p0 + kB2)) ||
        !take("v.image_newline", 0, {(uint64_t) kOut}, reinterpret_cast<uint8_t*>(p0 + kNl)) ||
        !take("v.position_embd.weight", 0, {D, (uint64_t) kGrid * kGrid}, host + kHostGrid) ||
        !take("mm.0.weight", 0, {2, 2, D, (uint64_t) kMergeHid}, host + kHostM0) ||
        !take("mm.2.weight", 0, {1, 1, (uint64_t) kMergeHid, (uint64_t) kOut}, host + kHostM2))
        return false;
    for (int i = 0; i < kLayers; ++i) {
        const std::string b = "v.blk." + std::to_string(i) + ".";
        uint8_t* blk = host + kHostBlocks + (size_t) i * kBlockBytes;
        uint16_t* m = reinterpret_cast<uint16_t*>(blk);
        float* v = reinterpret_cast<float*>(blk + kMats * 2);
        auto mat = [&](size_t at) { return reinterpret_cast<uint8_t*>(m + at); };
        auto vec = [&](size_t at) { return reinterpret_cast<uint8_t*>(v + at); };
        if (!take(b + "attn_q.weight", 30, {D, D}, mat(kWqkv)) || !take(b + "attn_k.weight", 30, {D, D}, mat(kWqkv + D * D)) ||
            !take(b + "attn_v.weight", 30, {D, D}, mat(kWqkv + 2 * D * D)) || !take(b + "attn_out.weight", 30, {D, D}, mat(kWo)) ||
            !take(b + "ffn_up.weight", 30, {D, (uint64_t) kInter}, mat(kWup)) ||
            !take(b + "ffn_down.weight", 30, {(uint64_t) kInter, D}, mat(kWdown)) || !take(b + "ln1.weight", 0, {D}, vec(kLn1w)) ||
            !take(b + "ln1.bias", 0, {D}, vec(kLn1b)) || !take(b + "attn_q.bias", 0, {D}, vec(kBqkv)) ||
            !take(b + "attn_k.bias", 0, {D}, vec(kBqkv + D)) || !take(b + "attn_v.bias", 0, {D}, vec(kBqkv + 2 * D)) ||
            !take(b + "attn_out.bias", 0, {D}, vec(kBo)) || !take(b + "ln2.weight", 0, {D}, vec(kLn2w)) ||
            !take(b + "ln2.bias", 0, {D}, vec(kLn2b)) || !take(b + "ffn_up.bias", 0, {(uint64_t) kInter}, vec(kBup)) ||
            !take(b + "ffn_down.bias", 0, {D}, vec(kBdown)))
            return false;
    }
    host_ = host;
    host_bytes_ = kHostBytes;
    return true;
}

bool Hy4Vision::encode(void* blas, void* stream, void* copy_stream, void* workspace, size_t ws_bytes, const float* patches, int gh, int gw,
                       float* rows, std::string& err) {
    if (!host_) {
        err = "the image encoder's weights are not loaded";
        return false;
    }
    const Plan p = plan_of(gh, gw);
    if (p.total == 0) {
        err = "a picture of " + std::to_string(gh) + " x " + std::to_string(gw) + " patches is not one the encoder takes (both sides even, "
              "at most " + std::to_string(kMaxPatches) + " patches)";
        return false;
    }
    if (ws_bytes < p.total) {
        err = "the image encoder's workspace is " + std::to_string(ws_bytes >> 20) + " MiB, the picture needs " + std::to_string(p.total >> 20);
        return false;
    }
    const int64_t n = p.n, L = p.L;
    const int oh = gh / kMerge, ow = gw / kMerge;
    Guard g;
    g.h = static_cast<cublasHandle_t>(blas);
    g.s = static_cast<cudaStream_t>(stream);
    g.cs = static_cast<cudaStream_t>(copy_stream);
    uint8_t* ws = static_cast<uint8_t*>(workspace);
    float* x = reinterpret_cast<float*>(ws + p.x);
    float* t = reinterpret_cast<float*>(ws + p.t);
    float* big = reinterpret_cast<float*>(ws + p.big);
    float* sc = reinterpret_cast<float*>(ws + p.sc);
    uint8_t* st[2] = {ws + p.st[0], ws + p.st[1]};
    float* wf = reinterpret_cast<float*>(ws + p.wf);
    float* w0 = reinterpret_cast<float*>(ws + p.w0);
    if (!blas_ok(cublasSetStream(g.h, g.s), "its stream", err) || !blas_ok(cublasGetMathMode(g.h, &g.math), "its math mode", err))
        return false;
    if (!blas_ok(cublasSetMathMode(g.h, CUBLAS_PEDANTIC_MATH), "pedantic math", err)) return false;
    g.math_set = true;
    for (cudaEvent_t& e : g.ev)
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) return cuda_ok("its events", err);
    cudaEvent_t* copied = g.ev;       // [2]: a staging buffer's piece has landed
    cudaEvent_t* released = g.ev + 2; // [2]: ... and been used (the buffer may take the next piece)

    // THE PIECES: block i at i, then the 2 x 2 conv's chunks, then the 1 x 1 conv's
    auto piece = [&](int k, size_t& bytes) -> const uint8_t* {
        if (k < kLayers) {
            bytes = kBlockBytes;
            return host_ + kHostBlocks + (size_t) k * kBlockBytes;
        }
        bytes = kChunkBytes;
        const int c = k - kLayers;
        return c < kM0Chunks ? host_ + kHostM0 + (size_t) c * kChunkBytes : host_ + kHostM2 + (size_t) (c - kM0Chunks) * kChunkBytes;
    };
    auto issue = [&](int k) {   // piece k's copy into staging buffer k % 2, once piece k - 2 there was used
        size_t bytes = 0;
        const uint8_t* srcp = piece(k, bytes);
        if (k >= 2) cudaStreamWaitEvent(g.cs, released[k & 1], 0);
        cudaMemcpyAsync(st[k & 1], srcp, bytes, cudaMemcpyHostToDevice, g.cs);
        cudaEventRecord(copied[k & 1], g.cs);
    };
    auto done = [&](int k) {   // piece k's buffer used: the next copy may land there
        cudaEventRecord(released[k & 1], g.s);
        if (k + 2 < kPieces) issue(k + 2);
    };
    auto block_in = [&](int k) -> bool {   // block k widened into wf (its matrices, then its vectors copied after them)
        cudaStreamWaitEvent(g.s, copied[k & 1], 0);
        k_widen<<<grid_of((int64_t) kMats, 256), 256, 0, g.s>>>(reinterpret_cast<const uint16_t*>(st[k & 1]), wf, (int64_t) kMats);
        cudaMemcpyAsync(wf + kMats, st[k & 1] + kMats * 2, kVecs * sizeof(float), cudaMemcpyDeviceToDevice, g.s);
        done(k);
        return cuda_ok("a block's weights", err);
    };
    auto chunk_in = [&](int k) -> const float* {   // a projector chunk, read where it landed (f32: nothing to widen)
        cudaStreamWaitEvent(g.s, copied[k & 1], 0);
        return reinterpret_cast<const float*>(st[k & 1]);
    };

    // piece P0 to the GPU, the first two blocks on their way; the patches; the position rows resampled on the host
    cudaMemcpyAsync(w0, host_ + kHostP0, kP0Floats * sizeof(float), cudaMemcpyHostToDevice, g.s);
    issue(0);
    issue(1);
    cudaMemcpyAsync(big, patches, (size_t) n * kPatchIn * sizeof(float), cudaMemcpyHostToDevice, g.s);
    {
        std::vector<float> pos((size_t) n * kDim);
        resample_grid(reinterpret_cast<const float*>(host_ + kHostGrid), gh, gw, pos.data());
        cudaMemcpyAsync(t, pos.data(), pos.size() * sizeof(float), cudaMemcpyHostToDevice, g.s);
        if (cudaStreamSynchronize(g.s) != cudaSuccess) return cuda_ok("the patches' upload", err);   // (pos goes out of scope)
    }
    if (!cuda_ok("the patches", err)) return false;
    // the patch embedding: x = patches W^T + b, then + the position rows
    if (!linear(g.h, big, kPatchIn, w0 + kPeW, kPatchIn, kPatchIn, kDim, n, x, kDim, "the patch embedding", err)) return false;
    k_bias<<<grid_of(n * kDim, 256), 256, 0, g.s>>>(x, w0 + kPeB, n, kDim);
    k_residual<<<grid_of(n * kDim, 256), 256, 0, g.s>>>(x, t, nullptr, n, kDim);
    if (!cuda_ok("the patch embedding", err)) return false;

    const float one = 1.0f, zero = 0.0f, scale = (float) (1.0 / std::sqrt((double) kHeadDim));   // SDPA's scale, in float
    const size_t smem = (size_t) n * sizeof(float);
    if (!venc_fused() && smem > (48u << 10) &&
        cudaFuncSetAttribute(k_softmax, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) != cudaSuccess)
        return cuda_ok("the softmax's shared memory", err);
    if (venc_fused() && cudaFuncSetAttribute(k_attn_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kFusedSmem) != cudaSuccess)
        return cuda_ok("the fused attention's shared memory", err);
    for (int il = 0; il < kLayers; ++il) {
        if (!block_in(il)) return false;
        const float *wqkv = wf + kWqkv, *wo = wf + kWo, *wup = wf + kWup, *wdown = wf + kWdown, *vv = wf + kMats;
        // h = ln1(x); q | k | v = h Wqkv^T + b
        k_layernorm<<<(unsigned) n, 384, 0, g.s>>>(x, vv + kLn1w, vv + kLn1b, t);
        if (!linear(g.h, t, kDim, wqkv, kDim, kDim, kQkv, n, big, kQkv, "the attention's qkv", err)) return false;
        k_bias<<<grid_of(n * kQkv, 256), 256, 0, g.s>>>(big, vv + kBqkv, n, kQkv);
        if (!cuda_ok("the qkv bias", err)) return false;
        // o = softmax(q k^T / sqrt(72)) v, a block of queries at a time, every head in one call (into t: h is consumed)
        const float* q = big;
        const float* k = big + kDim;
        const float* v = big + 2 * kDim;
        if (venc_fused()) {   // (moex: the scores never leave the SM - k_attn_fused)
            k_attn_fused<<<dim3((unsigned) ((n + kFq - 1) / kFq), (unsigned) kHeads), 256, kFusedSmem, g.s>>>(big, t, (int) n, scale);
            if (!cuda_ok("the attention", err)) return false;
        }
        for (int64_t q0 = 0; q0 < n && !venc_fused(); q0 += p.qc) {
            const int qb = (int) std::min<int64_t>(p.qc, n - q0);
            if (!blas_ok(cublasSgemmStridedBatched(g.h, CUBLAS_OP_T, CUBLAS_OP_N, (int) n, qb, kHeadDim, &scale, k, kQkv, kHeadDim,
                                                   q + q0 * kQkv, kQkv, kHeadDim, &zero, sc, (int) n, (long long) qb * n, kHeads),
                         "the attention's scores", err))
                return false;
            k_softmax<<<(unsigned) (kHeads * qb), 512, smem, g.s>>>(sc, (int) n);
            if (!cuda_ok("the attention's softmax", err)) return false;
            for (int64_t j0 = 0; j0 < n; j0 += kSumBlock) {   // o = p v, the keys kSumBlock at a time (beta 1 after the first)
                const int kb = (int) std::min<int64_t>(kSumBlock, n - j0);
                if (!blas_ok(cublasSgemmStridedBatched(g.h, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, qb, kb, &one, v + j0 * kQkv, kQkv, kHeadDim,
                                                       sc + j0, (int) n, (long long) qb * n, j0 ? &one : &zero, t + q0 * kDim, kDim,
                                                       kHeadDim, kHeads),
                             "the attention's output", err))
                    return false;
            }
        }
        // x = x + (o Wo^T + bo)   (o Wo^T into big: qkv is consumed)
        if (!linear(g.h, t, kDim, wo, kDim, kDim, kDim, n, big, kDim, "the attention's output projection", err)) return false;
        k_residual<<<grid_of(n * kDim, 256), 256, 0, g.s>>>(x, big, vv + kBo, n, kDim);
        // x = x + (gelu(ln2(x) Wup^T + bup) Wdown^T + bdown)
        k_layernorm<<<(unsigned) n, 384, 0, g.s>>>(x, vv + kLn2w, vv + kLn2b, t);
        if (!linear(g.h, t, kDim, wup, kDim, kDim, kInter, n, big, kInter, "the mlp's up", err)) return false;
        k_bias_gelu<<<grid_of(n * kInter, 256), 256, 0, g.s>>>(big, vv + kBup, n, kInter, gelu_erf_);
        if (!linear(g.h, big, kInter, wdown, kInter, kInter, kDim, n, t, kDim, "the mlp's down", err)) return false;
        k_residual<<<grid_of(n * kDim, 256), 256, 0, g.s>>>(x, t, vv + kBdown, n, kDim);
        if (!cuda_ok("a block", err)) return false;
    }
    // the projector: rms(x) * w, the 2 x 2 cells, gelu(cells W0^T + b0) (erf), then W2^T + b2 - each weight kChunk rows at a time
    k_rmsnorm<<<(unsigned) n, 384, 0, g.s>>>(x, w0 + kPre, t);
    float* u = big;
    float* hid = u + L * kMergeIn;
    float* out = hid + L * kMergeHid;
    k_merge<<<grid_of(L * kMergeIn, 256), 256, 0, g.s>>>(t, u, gw, L);
    if (!cuda_ok("the projector's norm and cells", err)) return false;
    for (int c = 0; c < kM0Chunks; ++c) {
        const int k = kLayers + c;
        if (!linear(g.h, u, kMergeIn, chunk_in(k), kMergeIn, kMergeIn, kChunk, L, hid + (size_t) c * kChunk, kMergeHid,
                    "the projector's 2 x 2 conv", err))
            return false;
        done(k);
    }
    k_bias_gelu<<<grid_of(L * kMergeHid, 256), 256, 0, g.s>>>(hid, w0 + kB0, L, kMergeHid, true);
    for (int c = 0; c < kM2Chunks; ++c) {
        const int k = kLayers + kM0Chunks + c;
        if (!linear(g.h, hid, kMergeHid, chunk_in(k), kMergeHid, kMergeHid, kChunk, L, out + (size_t) c * kChunk, kOut,
                    "the projector's 1 x 1 conv", err))
            return false;
        done(k);
    }
    k_bias<<<grid_of(L * kOut, 256), 256, 0, g.s>>>(out, w0 + kB2, L, kOut);
    if (!cuda_ok("the projector", err)) return false;
    // the rows: each merged row's cells, then the newline (the host's copy of it)
    const float* nl = reinterpret_cast<const float*>(host_ + kHostP0) + kNl;
    for (int r = 0; r < oh; ++r) {
        float* dst = rows + (size_t) r * (ow + 1) * kOut;
        cudaMemcpyAsync(dst, out + (size_t) r * ow * kOut, (size_t) ow * kOut * sizeof(float), cudaMemcpyDeviceToHost, g.s);
        std::memcpy(dst + (size_t) ow * kOut, nl, (size_t) kOut * sizeof(float));
    }
    if (cudaStreamSynchronize(g.s) != cudaSuccess || cudaStreamSynchronize(g.cs) != cudaSuccess) return cuda_ok("the rows' copy", err);
    return cuda_ok("the encode", err);
}

}  // namespace strata::x
