// moex/tools/hy4/attend_parity.cu - Hy4-flash's attention (src/x/hy4_kernels.cu attend_kernel), verbatim, against the
// variant that walks its cells NC at a time (attend_nc_kernel: the NC cells' dot products and warp sums interleaved, then the
// online softmax's updates in the same order with the same operations) - bit for bit on every output (the sparse layers' 32
// heads over 2,048 + 1-4 selected cells, the sliding-window layers' 64 heads over 512 positions, the prompt's one-chunk
// mode and the decode's split mode with its merge, the device position, DFly2's per-row counts), then timed.
// Built by hand: nvcc -O3 -DNDEBUG -std=c++20 -arch=sm_86 -o attend_parity attend_parity.cu && ./attend_parity [quick]
// HY4_PORT.md, 2026-11-19.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

constexpr int kLat = 512, kRope = 64, kRow = kLat + kRope;
constexpr int kAc = 32;   ///< cells a tile
constexpr int kAh = 16;   ///< heads a block (2 a warp)
constexpr int kPart = kLat + 2;

__device__ __forceinline__ int64_t pos_of(const int64_t* __restrict__ p, int64_t arg) { return p ? *p : arg; }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ int n_cells_of(int64_t p, int n_top, int kp, int window, bool sparse) {
    if (sparse) return min((int) (p / kp), n_top) * kp + (int) (p % kp) + 1;
    return (int) (p + 1 < (int64_t) window ? p + 1 : (int64_t) window);
}

// ------------------------------------------------------------------ reference (hy4_kernels.cu, verbatim)
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

// ------------------------------------------------------------------ the variant: NC cells a step
template <int NC, int MINB>
__global__ void __launch_bounds__(256, MINB) attend_nc_kernel(const float* __restrict__ q, const __half* __restrict__ cache, int64_t rows,
                                                        const int32_t* __restrict__ cells, int max_cells, int n_top, int kp,
                                                        int window, const float* __restrict__ sinks, float scale,
                                                        float* __restrict__ ctx, float* __restrict__ partial, int n_chunks, int H,
                                                        int64_t pos0_arg, const int64_t* __restrict__ pos_dev,
                                                        const int32_t* __restrict__ counts) {
    static_assert(kAc % NC == 0, "a tile's cells in steps of NC");
    __shared__ __align__(16) __half tile[kAc][kRow];
    const int chunk = blockIdx.x, hg = blockIdx.y, t = blockIdx.z;
    const int64_t p = pos_of(pos_dev, pos0_arg) + t;
    const bool sparse = cells != nullptr;
    const int nc_all = counts ? counts[t] : n_cells_of(p, n_top, kp, window, sparse);
    int c_begin = 0, c_end = nc_all;
    if (n_chunks > 1) {
        c_begin = chunk * kAc;
        c_end = min(nc_all, c_begin + kAc);
        if (c_begin >= c_end) return;
    }
    const int64_t first = sparse ? 0 : p - nc_all + 1;
    const int32_t* cl = sparse ? cells + (size_t) t * max_cells : nullptr;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int PK = kRow / 32, PV = kLat / 32;
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
        __syncthreads();
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
        for (int c = 0; c < nc; c += NC) {   // (rows past nc are zeros in the tile: read, never applied)
            float kv[NC][PK], s[NC][2];
#pragma unroll
            for (int j = 0; j < NC; ++j)
#pragma unroll
                for (int i = 0; i < PK; ++i) kv[j][i] = __half2float(tile[c + j][lane + 32 * i]);
#pragma unroll
            for (int j = 0; j < NC; ++j)
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    float sv = 0.0f;
#pragma unroll
                    for (int i = 0; i < PK; ++i) sv = fmaf(qr[hh][i], kv[j][i], sv);
                    s[j][hh] = sv;
                }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)   // warp_sum's butterfly, the NC x 2 sums interleaved
#pragma unroll
                for (int j = 0; j < NC; ++j)
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) s[j][hh] += __shfl_xor_sync(0xffffffffu, s[j][hh], o);
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    if (c + j >= nc) break;
                    const float sv = s[j][hh] * scale;
                    const float mn = fmaxf(m[hh], sv);
                    const float corr = expf(m[hh] - mn), pr = expf(sv - mn);
                    l[hh] = l[hh] * corr + pr;
#pragma unroll
                    for (int i = 0; i < PV; ++i) acc[hh][i] = acc[hh][i] * corr + pr * kv[j][i];
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

// ------------------------------------------------------------------ the variant: two 16-cell tiles, the next read while the
// current is computed (cp.async), the same shared memory as one 32-cell tile; the cells walked one at a time as the reference
#include <cuda_pipeline.h>
constexpr int kAd = 16;
template <int MINB>
__global__ void __launch_bounds__(256, MINB) attend_db_kernel(const float* __restrict__ q, const __half* __restrict__ cache, int64_t rows,
                                                        const int32_t* __restrict__ cells, int max_cells, int n_top, int kp,
                                                        int window, const float* __restrict__ sinks, float scale,
                                                        float* __restrict__ ctx, float* __restrict__ partial, int n_chunks, int H,
                                                        int64_t pos0_arg, const int64_t* __restrict__ pos_dev,
                                                        const int32_t* __restrict__ counts) {
    __shared__ __align__(16) __half tile[2][kAd][kRow];
    const int chunk = blockIdx.x, hg = blockIdx.y, t = blockIdx.z;
    const int64_t p = pos_of(pos_dev, pos0_arg) + t;
    const bool sparse = cells != nullptr;
    const int nc_all = counts ? counts[t] : n_cells_of(p, n_top, kp, window, sparse);
    int c_begin = 0, c_end = nc_all;
    if (n_chunks > 1) {
        c_begin = chunk * kAc;
        c_end = min(nc_all, c_begin + kAc);
        if (c_begin >= c_end) return;
    }
    const int64_t first = sparse ? 0 : p - nc_all + 1;
    const int32_t* cl = sparse ? cells + (size_t) t * max_cells : nullptr;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int PK = kRow / 32, PV = kLat / 32;
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
    auto issue = [&](int c0, int b) {   // cells [c0, c0 + 16) of the token's into tile b (rows past the end: not read)
        const int nc = min(kAd, c_end - c0);
        for (int e = threadIdx.x; e < nc * (kRow / 8); e += 256) {
            const int c = e / (kRow / 8), part = e % (kRow / 8);
            const int64_t pos = sparse ? (int64_t) cl[c0 + c] : first + c0 + c;
            __pipeline_memcpy_async(reinterpret_cast<uint4*>(&tile[b][c][0]) + part,
                                    reinterpret_cast<const uint4*>(cache + (size_t) (pos % rows) * kRow) + part, 16);
        }
        __pipeline_commit();
    };
    issue(c_begin, 0);
    int b = 0;
    for (int c0 = c_begin; c0 < c_end; c0 += kAd, b ^= 1) {
        const int nc = min(kAd, c_end - c0);
        __syncthreads();   // every warp is done with the tile the next read goes to
        if (c0 + kAd < c_end) {
            issue(c0 + kAd, b ^ 1);
            __pipeline_wait_prior(1);   // this tile's copies landed (the next one's may still be in flight)
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();   // ... every thread's
        for (int c = 0; c < nc; ++c) {
            float kv[PK];
#pragma unroll
            for (int i = 0; i < PK; ++i) kv[i] = __half2float(tile[b][c][lane + 32 * i]);
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

// ------------------------------------------------------------------ the harness
#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

struct Case {
    const char* name;
    bool sparse;
    int H, T;
    int64_t pos0;
    bool split, dev_pos, counts;
};

typedef void (*Kern)(const float*, const __half*, int64_t, const int32_t*, int, int, int, int, const float*, float, float*, float*,
                     int, int, int64_t, const int64_t*, const int32_t*);

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "quick") == 0;
    const int n_top = 511, kp = 4, window = 512, max_cells = n_top * kp + kp;
    const int64_t rows_sparse = 1 << 16, rows_swa = 512 + 1024 + 64;   // (positions modulo the rows: ~290 MB of VRAM in all, so it
                                                                          // runs beside a server in a later GPU's spare memory)
    std::mt19937_64 rng(20261119);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    const int Hmax = 64, THmax = 32768;   // (T x H at most: 1,024 tokens x 32 heads, 512 x 64)
    const int Tmax = 1024;
    (void) quick;
    std::vector<float> hq((size_t) THmax * kRow), hs(Hmax);
    for (auto& v : hq) v = U(rng) * 2.0f;
    for (auto& v : hs) v = U(rng) * 3.0f;
    std::vector<__half> hc((size_t) rows_sparse * kRow);
    for (auto& v : hc) v = __float2half(U(rng));
    float *q, *sinks, *ctxa, *ctxb, *parta, *partb;
    __half* cache;
    int32_t *cells, *cnt;
    int64_t* dpos;
    const int n_chunks_max = (max_cells + kAc - 1) / kAc;
    CK(cudaMalloc(&q, hq.size() * 4));
    CK(cudaMalloc(&sinks, Hmax * 4));
    CK(cudaMalloc(&cache, hc.size() * 2));
    CK(cudaMalloc(&ctxa, (size_t) THmax * kLat * 4));
    CK(cudaMalloc(&ctxb, (size_t) THmax * kLat * 4));
    CK(cudaMalloc(&parta, (size_t) 8 * Hmax * n_chunks_max * kPart * 4));
    CK(cudaMalloc(&partb, (size_t) 8 * Hmax * n_chunks_max * kPart * 4));
    CK(cudaMalloc(&cells, (size_t) Tmax * max_cells * 4));
    CK(cudaMalloc(&cnt, Tmax * 4));
    CK(cudaMalloc(&dpos, 8));
    CK(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(sinks, hs.data(), Hmax * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(cache, hc.data(), hc.size() * 2, cudaMemcpyHostToDevice));

    const Case cases[] = {
        {"sparse 32 heads, prompt, 2,048 + 1-4 cells", true, 32, Tmax, 700001, false, false, false},
        {"sparse 32 heads, prompt, early positions (0-T)", true, 32, Tmax, 0, false, false, false},
        {"window 64 heads, prompt, 512 positions", false, 64, Tmax / 2, 300000, false, false, false},
        {"window 64 heads, prompt, early positions (0-T)", false, 64, Tmax / 2, 0, false, false, false},
        {"sparse 32 heads, decode split (T 6)", true, 32, 6, 812345, true, false, false},
        {"sparse 32 heads, decode split, device position (T 1)", true, 32, 1, 999999, true, true, false},
        {"window 64 heads, decode split (T 8)", false, 64, 8, 123457, true, false, false},
        {"sparse 32 heads, DFly2's counts, split (T 6)", true, 32, 6, 400003, true, false, true},
    };
    int bad_total = 0;
    for (const Case& cs : cases) {
        // the cells: per token, n_top distinct groups below its own (ascending, as the selection writes them), then its group
        std::vector<int32_t> hcl((size_t) cs.T * max_cells, 0), hcnt(cs.T);
        for (int t = 0; t < cs.T; ++t) {
            const int64_t p = cs.pos0 + t;
            const int ng = (int) std::min<int64_t>(p / kp, n_top);
            std::vector<int32_t> g;
            if (p / kp <= n_top) {
                for (int i = 0; i < ng; ++i) g.push_back(i);
            } else {
                std::vector<char> used((size_t) (p / kp), 0);
                while ((int) g.size() < ng) {
                    const int32_t x = (int32_t) (rng() % (uint64_t) (p / kp));
                    if (!used[(size_t) x]) {
                        used[(size_t) x] = 1;
                        g.push_back(x);
                    }
                }
                std::sort(g.begin(), g.end());
            }
            int n = 0;
            for (int32_t x : g)
                for (int mm = 0; mm < kp; ++mm) hcl[(size_t) t * max_cells + n++] = x * kp + mm;
            const int tail = (int) (p % kp) + 1;
            for (int i = 0; i < tail; ++i) hcl[(size_t) t * max_cells + n++] = (int32_t) (p - tail + 1 + i);
            hcnt[t] = cs.counts ? std::max(1, n - (int) (rng() % 7)) : n;   // (DFly2: a row's own count)
        }
        CK(cudaMemcpy(cells, hcl.data(), hcl.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(cnt, hcnt.data(), cs.T * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpos, &cs.pos0, 8, cudaMemcpyHostToDevice));
        const int max_l = cs.sparse ? max_cells : window;
        const int nch = cs.split ? (max_l + kAc - 1) / kAc : 1;
        const int64_t rows = cs.sparse ? rows_sparse : rows_swa;
        const int32_t* cl = cs.sparse ? cells : nullptr;
        const int64_t* pd = cs.dev_pos ? dpos : nullptr;
        const int32_t* cn = cs.counts ? cnt : nullptr;
        const float scale = 1.0f / std::sqrt(256.0f);
        auto run = [&](Kern k, float* ctx, float* part) {
            k<<<dim3(nch, cs.H / kAh, cs.T), 256>>>(q, cache, rows, cl, max_cells, n_top, kp, window, sinks, scale, ctx, part, nch,
                                                    cs.H, cs.dev_pos ? 0 : cs.pos0, pd, cn);
            if (nch > 1)
                attend_merge_kernel<<<dim3(cs.H, cs.T), 256>>>(part, sinks, ctx, nch, cs.H, n_top, kp, window, cs.sparse ? 1 : 0,
                                                               cs.dev_pos ? 0 : cs.pos0, pd, cn);
        };
        const size_t nout = (size_t) cs.T * cs.H * kLat;
        std::vector<float> ra(nout), rb(nout);
        CK(cudaMemset(ctxa, 0, nout * 4));
        run(attend_kernel, ctxa, parta);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ra.data(), ctxa, nout * 4, cudaMemcpyDeviceToHost));
        struct V {
            const char* n;
            Kern k;
        } vars[] = {{"NC2/1", attend_nc_kernel<2, 1>}, {"NC2/2", attend_nc_kernel<2, 2>}, {"DB/1", attend_db_kernel<1>},
                    {"DB/2", attend_db_kernel<2>}};
        printf("%s:\n", cs.name);
        for (const V& v : vars) {
            CK(cudaMemset(ctxb, 0xff, nout * 4));
            run(v.k, ctxb, partb);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(rb.data(), ctxb, nout * 4, cudaMemcpyDeviceToHost));
            size_t bad = 0;
            for (size_t i = 0; i < nout; ++i) bad += std::memcmp(&ra[i], &rb[i], 4) != 0;
            bad_total += bad != 0;
            // timing: the best of 5
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            float best_a = 1e30f, best_b = 1e30f;
            for (int rep = 0; rep < 5; ++rep) {
                float ms;
                cudaEventRecord(e0);
                run(attend_kernel, ctxa, parta);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms, e0, e1);
                best_a = std::min(best_a, ms);
                cudaEventRecord(e0);
                run(v.k, ctxb, partb);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms, e0, e1);
                best_b = std::min(best_b, ms);
            }
            printf("  %-5s %zu of %zu values differ | %.3f -> %.3f ms (x%.2f)\n", v.n, bad, nout, best_a, best_b, best_a / best_b);
        }
    }
    printf(bad_total ? "DIFFERENCES FOUND\n" : "ALL BIT FOR BIT\n");
    return bad_total ? 1 : 0;
}
