// moex/tools/hy4/indexer_parity.cu - Hy4-flash's indexer (src/x/hy4_kernels.cu): the reference scores / select kernels,
// verbatim, against the register-tiled scores (scores_mm_kernel) and the selections that read the scores once - bit for bit on
// every score and every cell (17 cases: 1-512 rows, 4-1,048,569 positions, the device position, ties and zero pools), then
// timed.  Built by hand: nvcc -O3 -DNDEBUG -arch=sm_86 -o indexer_parity indexer_parity.cu && ./indexer_parity [quick]
// (select2 is a tried variant; select3 is select_sampled_kernel's algorithm - the engine's takes an unaligned row's head
// apart instead of reading it all one key at a time; indexer_switch_check.cpp checks the engine's own.)  HY4_PORT.md, 2026-11-19.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

constexpr int kIdx = 128, kIdxHeads = 16;
__device__ __forceinline__ int64_t pos_of(const int64_t* __restrict__ p, int64_t arg) { return p ? *p : arg; }

// ------------------------------------------------------------------ reference (hy4_kernels.cu, verbatim)
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

template <int G>
__global__ void __launch_bounds__(128) scores_tile_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                                          const float* __restrict__ w, float* __restrict__ score, int64_t pos0_arg,
                                                          const int64_t* __restrict__ pos_dev, int kp, int max_pool, int T) {
    __shared__ float key[4][G][2 * 65];
    const int64_t pos0 = pos_of(pos_dev, pos0_arg);
    const int n_last = (int) ((pos0 + T - 1) / kp);
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
            if (P0 >= n_pool) continue;
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

__device__ __forceinline__ uint32_t okey(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

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
    const int n_tail = (int) (pos % kp) + 1;
    if ((int) threadIdx.x < n_tail) out[n_sel * kp + threadIdx.x] = (int32_t) (pos - n_tail + 1 + threadIdx.x);
}

// ------------------------------------------------------------------ new: register-tiled scores
// A block: BT rows x BP pools, BT * WP warps; warp (r, ps) takes row r and PW = 8 TN pools.  Lane bit 4: the half (dims
// 0-63 / 64-127), bit 3: the head group (heads 0-7 / 8-15), bits 0-2: TN consecutive pools.  Each (row, head, half, pool)
// is one thread's sequential fmaf chain over its 64 values from 0.0f, as scores_kernel's lane; the halves joined lo + hi,
// relu, times the head's weight, and the 16 heads summed in scores_kernel's shuffle tree (h + (h ^ 8), then ^ 4, ^ 2, ^ 1,
// each own + other) - every score that kernel's bits.  The dims come in chunks of DC through shared memory, transposed so a
// thread's 8 heads and TN pools are 16-byte reads.
template <int BT, int WP, int TN>
__global__ void __launch_bounds__(BT * WP * 32) scores_mm_kernel(const float* __restrict__ iq, const float* __restrict__ pooled,
                                                                const float* __restrict__ w, float* __restrict__ score,
                                                                int64_t pos0_arg, const int64_t* __restrict__ pos_dev, int kp,
                                                                int max_pool, int T) {
    constexpr int NW = BT * WP, NTH = NW * 32, PW = 8 * TN, BP = PW * WP, DC = 16, BPS = BP + 4;
    __shared__ __align__(16) float Ks[2][DC][BPS];
    __shared__ __align__(16) float Qs[BT][2][DC][16];
    const int64_t pos0 = pos_of(pos_dev, pos0_arg);
    const int n_last = (int) ((pos0 + T - 1) / kp);
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
                __syncthreads();
                for (int e = tid; e < BP * 2 * (DC / 4); e += NTH) {   // a pool's 2 x DC values: 8 consecutive threads
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
                    for (int j = 0; j < TN; ++j) {
                        const float a = __fadd_rn(acc[i][j], __shfl_xor_sync(0xffffffffu, acc[i][j], 16));
                        acc[i][j] = __fmul_rn(fmaxf(a, 0.0f), wv[i]);
                    }
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) acc[i][j] = __fadd_rn(acc[i][j], __shfl_xor_sync(0xffffffffu, acc[i][j], 8));
                if (hf == 0 && hg == 0) {
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

// ------------------------------------------------------------------ new: select
// select_kernel's cells: the threshold key (the n_top-th largest) and how many of its equals to take by a radix select in
// 11 / 11 / 10-bit digits (lanes with one digit add once: __match_any_sync), then the pick in index order with coalesced
// reads - warp w a contiguous run, its counts scanned over the warps, ballots inside it.
constexpr int kSelT = 1024;
__global__ void __launch_bounds__(kSelT) select2_kernel(const float* __restrict__ score, int max_pool, int n_top, int kp,
                                                        int64_t pos0_arg, const int64_t* __restrict__ pos_dev,
                                                        int32_t* __restrict__ cells, int max_cells) {
    __shared__ uint32_t hist[2048];
    __shared__ uint32_t s_digit, s_rem;
    __shared__ uint32_t scan_tmp[33];
    __shared__ uint32_t wgt[32], weq[32];
    const int t = blockIdx.x;
    const int64_t pos = pos_of(pos_dev, pos0_arg) + t;
    const int n_pool = (int) (pos / kp);
    const int n_sel = n_pool < n_top ? n_pool : n_top;
    int32_t* out = cells + (size_t) t * max_cells;
    const float* sc = score + (size_t) t * max_pool;
    const int tid = (int) threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (n_pool <= n_top) {
        for (int i = tid; i < n_sel * kp; i += kSelT) out[i] = i;
    } else {
        uint32_t prefix = 0, mask = 0, remaining = (uint32_t) n_top;
#pragma unroll 1
        for (int pass = 0; pass < 3; ++pass) {
            const int shift = pass == 0 ? 21 : pass == 1 ? 10 : 0;
            const uint32_t dmask = pass == 2 ? 1023u : 2047u;
            for (int i = tid; i < 2048; i += kSelT) hist[i] = 0;
            __syncthreads();
            for (int base = warp * 32; base < n_pool; base += kSelT) {   // a warp's 32 consecutive keys a step
                const int i = base + lane;
                const uint32_t key = i < n_pool ? okey(sc[i]) : 0u;
                const bool ok = i < n_pool && (key & mask) == prefix;
                const unsigned act = __ballot_sync(0xffffffffu, ok);
                if (ok) {
                    const uint32_t dg = (key >> shift) & dmask;
                    const unsigned peers = __match_any_sync(act, dg);
                    if (lane == __ffs(peers) - 1) atomicAdd(&hist[dg], (uint32_t) __popc(peers));
                }
            }
            __syncthreads();
            {   // the digit: suffix sums from the top bin, two bins a thread (bins 2047 - 2 tid, 2046 - 2 tid)
                const int d1 = 2047 - 2 * tid, d2 = d1 - 1;
                const uint32_t h1 = hist[d1], h2 = hist[d2];
                const uint32_t e = block_excl_scan(h1 + h2, scan_tmp);
                if (e < remaining && e + h1 >= remaining) {
                    s_digit = (uint32_t) d1;
                    s_rem = remaining - e;
                } else if (e + h1 < remaining && e + h1 + h2 >= remaining) {
                    s_digit = (uint32_t) d2;
                    s_rem = remaining - e - h1;
                }
            }
            __syncthreads();
            prefix |= s_digit << shift;
            mask |= dmask << shift;
            remaining = s_rem;
            __syncthreads();
        }
        const uint32_t thr = prefix;
        // the pick: warp w takes keys [w seg, (w + 1) seg), seg a multiple of 32
        const int seg = ((n_pool + 32 * 32 - 1) / (32 * 32)) * 32;
        const int b0 = warp * seg, b1 = min(n_pool, b0 + seg);
        uint32_t gt = 0, eq = 0;
        for (int i = b0 + lane; i < b1; i += 32) {
            const uint32_t key = okey(sc[i]);
            gt += key > thr;
            eq += key == thr;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            gt += __shfl_xor_sync(0xffffffffu, gt, o);
            eq += __shfl_xor_sync(0xffffffffu, eq, o);
        }
        if (lane == 0) {
            wgt[warp] = gt;
            weq[warp] = eq;
        }
        __syncthreads();
        uint32_t eq_before = 0, out_before = 0;
        for (int v = 0; v < warp; ++v) {   // (32 warps: the scan by each warp itself)
            const uint32_t te = eq_before >= remaining ? 0u : min(weq[v], remaining - eq_before);
            out_before += wgt[v] + te;
            eq_before += weq[v];
        }
        const uint32_t take_eq = eq_before >= remaining ? 0u : min(eq, remaining - eq_before);
        uint32_t o = out_before, eq_seen = 0;
        for (int base = b0; base < b1; base += 32) {
            const int i = base + lane;
            const uint32_t key = i < b1 ? okey(sc[i]) : 0u;
            const bool is_gt = i < b1 && key > thr, is_eq = i < b1 && key == thr;
            const unsigned eqb = __ballot_sync(0xffffffffu, is_eq);
            const unsigned below = (1u << lane) - 1u;
            const bool take = is_gt || (is_eq && eq_seen + (uint32_t) __popc(eqb & below) < take_eq);
            const unsigned tb = __ballot_sync(0xffffffffu, take);
            if (take) {
                const uint32_t k = o + (uint32_t) __popc(tb & below);
                for (int m = 0; m < kp; ++m) out[k * kp + m] = i * kp + m;
            }
            o += (uint32_t) __popc(tb);
            eq_seen += (uint32_t) __popc(eqb);
        }
    }
    const int n_tail = (int) (pos % kp) + 1;
    if (tid < n_tail) out[n_sel * kp + tid] = (int32_t) (pos - n_tail + 1 + tid);
}


// ------------------------------------------------------------------ new: select by a sampled threshold
// select_kernel's cells, read once: tau = the 16th largest of 1,024 evenly spaced keys; every key >= tau (with its index)
// into shared memory - when there are n_top .. kCap of them, the n_top best of those (key descending, index ascending: the
// tie rule) are the best n_top of all, sorted there and written in index order; otherwise select_kernel's radix passes.
constexpr int kCap = 4096, kSamp = 4096, kTarget = 2000;   // samples; the candidates aimed at
__device__ __forceinline__ void bitonic_desc(unsigned long long* v, int n) {   // n a power of two; every thread calls it
    for (int k = 2; k <= n; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = (int) threadIdx.x; i < n / 2; i += (int) blockDim.x) {
                const int lo = 2 * i - (i & (j - 1)), hi = lo + j;   // the pair (lo, lo + j), lo's bit j clear
                const bool up = (lo & k) == 0;                        // this run sorted descending when up
                const unsigned long long a = v[lo], b = v[hi];
                if ((a < b) == up) { v[lo] = b; v[hi] = a; }
            }
            __syncthreads();
        }
}
__global__ void __launch_bounds__(1024) select3_kernel(const float* __restrict__ score, int max_pool, int n_top, int kp,
                                                       int64_t pos0_arg, const int64_t* __restrict__ pos_dev,
                                                       int32_t* __restrict__ cells, int max_cells) {
    __shared__ unsigned long long cand[kCap];
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_digit, s_rem, s_cnt;
    __shared__ uint32_t scan_tmp[33];
    const int t = blockIdx.x;
    const int64_t pos = pos_of(pos_dev, pos0_arg) + t;
    const int n_pool = (int) (pos / kp);
    const int n_sel = n_pool < n_top ? n_pool : n_top;
    int32_t* out = cells + (size_t) t * max_cells;
    const float* sc = score + (size_t) t * max_pool;
    const int tid = (int) threadIdx.x, lane = tid & 31;
    bool done = false;
    if (n_pool <= n_top) {
        for (int i = tid; i < n_sel * kp; i += (int) blockDim.x) out[i] = i;
        done = true;
    } else if (n_pool >= 8 * kSamp) {   // (32,768 groups: positions from 131,072)
        // tau: of kSamp keys at i * n_pool / kSamp, the K-th largest - K the share of kTarget in n_pool (the count of keys >= tau:
        // kTarget +- 16% at n_pool 204,800; n_top .. kCap keeps the read: a miss costs the radix passes, never a cell)
        for (int i = tid; i < kSamp; i += (int) blockDim.x)
            cand[i] = ((unsigned long long) okey(sc[(int) (((int64_t) i * n_pool) / kSamp)]) << 32) | (uint32_t) ~(uint32_t) i;
        if (tid == 0) s_cnt = 0;
        __syncthreads();
        bitonic_desc(cand, kSamp);
        const int64_t kq = ((int64_t) kTarget * kSamp) / n_pool;
        const int K = kq < 1 ? 1 : kq > kSamp ? kSamp : (int) kq;
        const uint32_t tau = (uint32_t) (cand[K - 1] >> 32);
        __syncthreads();
        // every key >= tau: a warp's lanes in a row, one atomicAdd a warp a step; four float4 a thread in flight
        const int n4 = n_pool / 4;
        const float4* s4 = reinterpret_cast<const float4*>(sc);
        const bool al = (reinterpret_cast<uintptr_t>(sc) & 15) == 0;
        for (int b = 0; b < (al ? n4 : 0); b += 4 * (int) blockDim.x) {
            float4 v[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int i = b + u * (int) blockDim.x + tid;
                v[u] = i < n4 ? s4[i] : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int i = b + u * (int) blockDim.x + tid;
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
                        if (take && k < (uint32_t) kCap) cand[k] = ((unsigned long long) key << 32) | (uint32_t) ~(uint32_t) (4 * i + c);
                    }
                }
            }
        }
        for (int i = (al ? 4 * n4 : 0) + tid; i < n_pool; i += (int) blockDim.x) {   // the tail (or all, unaligned)
            const uint32_t key = okey(sc[i]);
            if (key >= tau) {
                const uint32_t k = atomicAdd(&s_cnt, 1u);
                if (k < (uint32_t) kCap) cand[k] = ((unsigned long long) key << 32) | (uint32_t) ~(uint32_t) i;
            }
        }
        __syncthreads();
        const uint32_t cnt = s_cnt;
        if (cnt >= (uint32_t) n_top && cnt <= (uint32_t) kCap) {
            int np = 1;
            while (np < (int) cnt) np <<= 1;
            for (int i = (int) cnt + tid; i < np; i += (int) blockDim.x) cand[i] = 0ull;   // (below every real entry: ~index != 0... see below)
            __syncthreads();
            bitonic_desc(cand, np);
            // the best n_top: their indices ascending (the low words are ~index: descending ~index = ascending index)
            int nq = 1;
            while (nq < n_top) nq <<= 1;
            for (int i = tid; i < nq; i += (int) blockDim.x)
                cand[kCap / 2 + i] = i < n_top ? (unsigned long long) (uint32_t) cand[i] : 0ull;
            __syncthreads();
            bitonic_desc(cand + kCap / 2, nq);
            for (int i = tid; i < n_top; i += (int) blockDim.x) {
                const int g = (int) ~(uint32_t) cand[kCap / 2 + i];
                for (int m = 0; m < kp; ++m) out[i * kp + m] = g * kp + m;
            }
            done = true;
        }
        __syncthreads();
    }
    if (!done) {   // select_kernel's passes
        uint32_t prefix = 0, mask = 0, remaining = (uint32_t) n_top;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = tid; i < 256; i += (int) blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int i = tid; i < n_pool; i += (int) blockDim.x) {
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
    const int n_tail = (int) (pos % kp) + 1;
    if (tid < n_tail) out[n_sel * kp + tid] = (int32_t) (pos - n_tail + 1 + tid);
}

// ------------------------------------------------------------------ the harness
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

template <int BT, int WP, int TN>
void launch_mm(const float* iq, const float* pooled, const float* w, float* score, int64_t pos0, const int64_t* pos_dev, int kp,
               int max_pool, int T, cudaStream_t s) {
    constexpr int BP = 8 * TN * WP;
    const int64_t last = pos_dev ? (int64_t) max_pool * kp : pos0 + T;
    const int n_pool = (int) std::min<int64_t>(max_pool, last / kp);
    const int n_rt = (T + BT - 1) / BT;
    const int n_pt = std::max(1, std::min(65535, (n_pool + BP - 1) / BP));
    scores_mm_kernel<BT, WP, TN><<<dim3(n_rt, n_pt), BT * WP * 32, 0, s>>>(iq, pooled, w, score, pos0, pos_dev, kp, max_pool, T);
}

typedef void (*MmFn)(const float*, const float*, const float*, float*, int64_t, const int64_t*, int, int, int, cudaStream_t);
struct MmCfg { const char* name; MmFn fn; };

int main(int argc, char** argv) {
    const int kp = 4, n_top = 511, max_cells = 2048 + 4;
    const int64_t n_ctx = 1048576;
    const int max_pool = (int) (n_ctx / kp + 1), TMAX = 512;
    float *pooled, *iq, *w, *sc_ref, *sc_new;
    int32_t *c_ref, *c_new;
    int64_t* pos_d;
    CK(cudaMalloc(&pooled, (size_t) max_pool * kIdx * 4));
    CK(cudaMalloc(&iq, (size_t) TMAX * kIdxHeads * kIdx * 4));
    CK(cudaMalloc(&w, (size_t) TMAX * kIdxHeads * 4));
    CK(cudaMalloc(&sc_ref, (size_t) TMAX * max_pool * 4));
    CK(cudaMalloc(&sc_new, (size_t) TMAX * max_pool * 4));
    CK(cudaMalloc(&c_ref, (size_t) TMAX * max_cells * 4));
    CK(cudaMalloc(&c_new, (size_t) TMAX * max_cells * 4));
    CK(cudaMalloc(&pos_d, 8));
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    {
        std::vector<float> h((size_t) max_pool * kIdx);
        for (auto& v : h) v = nd(rng);
        // ties: every 997th pool a copy of pool 5, every 1009th pool all zeros (score 0 for every row)
        for (int P = 997; P < max_pool; P += 997) std::copy(h.begin() + 5 * kIdx, h.begin() + 6 * kIdx, h.begin() + (size_t) P * kIdx);
        for (int P = 1009; P < max_pool; P += 1009) std::fill(h.begin() + (size_t) P * kIdx, h.begin() + (size_t) (P + 1) * kIdx, 0.0f);
        CK(cudaMemcpy(pooled, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        h.assign((size_t) TMAX * kIdxHeads * kIdx, 0.f);
        for (auto& v : h) v = nd(rng);
        CK(cudaMemcpy(iq, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        h.assign((size_t) TMAX * kIdxHeads, 0.f);
        for (auto& v : h) v = 0.1f * nd(rng);   // (negative weights too)
        CK(cudaMemcpy(w, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    }
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    const MmCfg cfgs[] = {
        {"bt1 wp8 tn4", launch_mm<1, 8, 4>}, {"bt2 wp4 tn4", launch_mm<2, 4, 4>}, {"bt4 wp2 tn4", launch_mm<4, 2, 4>},
        {"bt8 wp1 tn4", launch_mm<8, 1, 4>}, {"bt4 wp2 tn8", launch_mm<4, 2, 8>}, {"bt8 wp1 tn8", launch_mm<8, 1, 8>},
        {"bt8 wp2 tn8", launch_mm<8, 2, 8>}, {"bt4 wp4 tn8", launch_mm<4, 4, 8>}, {"bt6 wp2 tn4", launch_mm<6, 2, 4>},
        {"bt6 wp2 tn8", launch_mm<6, 2, 8>}, {"bt3 wp4 tn4", launch_mm<3, 4, 4>}, {"bt2 wp8 tn4", launch_mm<2, 8, 4>},
    };
    const bool quick = argc > 1 && !strcmp(argv[1], "quick");
    auto time_it = [&](auto&& fn, int R) {
        fn();
        CK(cudaStreamSynchronize(s));
        cudaEventRecord(a, s);
        for (int r = 0; r < R; ++r) fn();
        cudaEventRecord(b, s);
        CK(cudaEventSynchronize(b));
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        return ms / R;
    };
    std::vector<float> hr, hn;
    std::vector<int32_t> cr, cn;
    int fails = 0;
    struct Case { int T; int64_t p; bool dev; };
    std::vector<Case> cases = {{1, 4, false}, {1, 2047, false}, {1, 2049, false}, {3, 2046, false}, {1, 16385, false},
                               {6, 16383, false}, {8, 131070, true}, {1, 819200, false}, {1, 819203, true}, {2, 819201, false},
                               {6, 819198, true}, {8, 819200, false}, {64, 131072, false}, {512, 131071, false},
                               {512, 819200, false}, {1, 1048575 - 1, false}, {6, 1048575 - 6, true}};
    if (quick) cases = {{1, 819200, false}, {6, 819198, true}, {512, 819200, false}};
    for (const Case& cs : cases) {
        const int T = cs.T;
        const int64_t p = cs.p;
        const int64_t* pd = cs.dev ? pos_d : nullptr;
        CK(cudaMemcpy(pos_d, &p, 8, cudaMemcpyHostToDevice));
        // reference: the row kernel, then the select
        CK(cudaMemsetAsync(sc_ref, 0xff, (size_t) T * max_pool * 4, s));
        {
            const int64_t last = pd ? (int64_t) max_pool * kp : p + T;
            const int n_pool = (int) std::min<int64_t>(max_pool, last / kp);
            const int nb = std::max(1, std::min(1024, (n_pool + 3) / 4));
            scores_kernel<<<dim3(nb, T), 128, 0, s>>>(iq, pooled, w, sc_ref, p, pd, kp, max_pool);
        }
        select_kernel<<<T, 1024, 0, s>>>(sc_ref, max_pool, n_top, kp, p, pd, c_ref, max_cells);
        CK(cudaStreamSynchronize(s));
        CK(cudaGetLastError());
        hr.resize((size_t) T * max_pool);
        CK(cudaMemcpy(hr.data(), sc_ref, hr.size() * 4, cudaMemcpyDeviceToHost));
        cr.resize((size_t) T * max_cells);
        CK(cudaMemcpy(cr.data(), c_ref, cr.size() * 4, cudaMemcpyDeviceToHost));
        auto check_scores = [&](const char* what) {
            hn.resize((size_t) T * max_pool);
            CK(cudaMemcpy(hn.data(), sc_new, hn.size() * 4, cudaMemcpyDeviceToHost));
            long bad = 0;
            for (int t = 0; t < T; ++t) {
                const int n_pool = (int) ((p + t) / kp);
                if (memcmp(hr.data() + (size_t) t * max_pool, hn.data() + (size_t) t * max_pool, (size_t) n_pool * 4)) {
                    for (int P = 0; P < n_pool; ++P)
                        if (memcmp(&hr[(size_t) t * max_pool + P], &hn[(size_t) t * max_pool + P], 4)) {
                            if (bad < 3) fprintf(stderr, "  %s T %d p %lld: row %d pool %d: %.9g vs %.9g\n", what, T, (long long) p, t, P,
                                                 hr[(size_t) t * max_pool + P], hn[(size_t) t * max_pool + P]);
                            ++bad;
                        }
                }
            }
            return bad;
        };
        auto check_cells = [&](const char* what) {
            cn.resize((size_t) T * max_cells);
            CK(cudaMemcpy(cn.data(), c_new, cn.size() * 4, cudaMemcpyDeviceToHost));
            long bad = 0;
            for (int t = 0; t < T; ++t) {
                const int64_t pos = p + t;
                const int n = (int) (std::min<int64_t>(pos / kp, n_top) * kp + pos % kp + 1);
                if (memcmp(cr.data() + (size_t) t * max_cells, cn.data() + (size_t) t * max_cells, (size_t) n * 4)) {
                    if (bad < 3) fprintf(stderr, "  %s T %d p %lld: row %d cells differ\n", what, T, (long long) p, t);
                    ++bad;
                }
            }
            return bad;
        };
        // the tile kernel (today's default)
        {
            CK(cudaMemsetAsync(sc_new, 0xff, (size_t) T * max_pool * 4, s));
            const int64_t last = pd ? (int64_t) max_pool * kp : p + T;
            const int n_pool = (int) std::min<int64_t>(max_pool, last / kp);
            const int nbt = std::max(1, std::min(2048, (n_pool + 31) / 32));
            scores_tile_kernel<8><<<nbt, 128, 0, s>>>(iq, pooled, w, sc_new, p, pd, kp, max_pool, T);
            CK(cudaStreamSynchronize(s));
            const long bad = check_scores("tile");
            fails += bad != 0;
            const float ms = (quick || T <= 64) ? time_it([&] { scores_tile_kernel<8><<<nbt, 128, 0, s>>>(iq, pooled, w, sc_new, p, pd, kp, max_pool, T); }, T >= 64 ? 3 : 20) : -1.f;
            const float ms_sel = time_it([&] { select_kernel<<<T, 1024, 0, s>>>(sc_ref, max_pool, n_top, kp, p, pd, c_ref, max_cells); }, T >= 64 ? 3 : 20);
            printf("T %3d p %7lld%s: tile %s %.3f ms | select %.3f ms\n", T, (long long) p, pd ? " (dev)" : "", bad ? "DIFF" : "same", ms, ms_sel);
        }
        for (const MmCfg& c : cfgs) {
            CK(cudaMemsetAsync(sc_new, 0xff, (size_t) T * max_pool * 4, s));
            c.fn(iq, pooled, w, sc_new, p, pd, kp, max_pool, T, s);
            CK(cudaStreamSynchronize(s));
            CK(cudaGetLastError());
            const long bad = check_scores(c.name);
            fails += bad != 0;
            const float ms = time_it([&] { c.fn(iq, pooled, w, sc_new, p, pd, kp, max_pool, T, s); }, T >= 64 ? 3 : 20);
            printf("    %s: %s %.3f ms\n", c.name, bad ? "DIFF" : "same", ms);
        }
        {
            CK(cudaMemsetAsync(c_new, 0xff, (size_t) T * max_cells * 4, s));
            select2_kernel<<<T, kSelT, 0, s>>>(sc_ref, max_pool, n_top, kp, p, pd, c_new, max_cells);
            CK(cudaStreamSynchronize(s));
            CK(cudaGetLastError());
            const long bad = check_cells("select2");
            fails += bad != 0;
            const float ms = time_it([&] { select2_kernel<<<T, kSelT, 0, s>>>(sc_ref, max_pool, n_top, kp, p, pd, c_new, max_cells); }, T >= 64 ? 3 : 20);
            printf("    select2: %s %.3f ms\n", bad ? "DIFF" : "same", ms);
        }
        {
            CK(cudaMemsetAsync(c_new, 0xff, (size_t) T * max_cells * 4, s));
            select3_kernel<<<T, 1024, 0, s>>>(sc_ref, max_pool, n_top, kp, p, pd, c_new, max_cells);
            CK(cudaStreamSynchronize(s));
            CK(cudaGetLastError());
            const long bad = check_cells("select3");
            fails += bad != 0;
            const float ms = time_it([&] { select3_kernel<<<T, 1024, 0, s>>>(sc_ref, max_pool, n_top, kp, p, pd, c_new, max_cells); }, T >= 64 ? 3 : 20);
            printf("    select3: %s %.3f ms\n", bad ? "DIFF" : "same", ms);
        }
    }
    printf(fails ? "FAILURES: %d\n" : "ALL SAME\n", fails);
    return fails ? 1 : 0;
}
