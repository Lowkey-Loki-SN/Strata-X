// src/kernels/cpu/mx_avx512.cpp - moex: AVX-512BW multi-token dot products for MXFP4 rows against Q8_0 activations, bitwise
// equal per token to ggml-cpu's AVX2 ggml_vec_dot_mxfp4_q8_0 (which the CPU runs for these rows on every x86 build).
//
// Why: at one token the pool is DRAM-bound on MXFP4 experts (ggml-cpu's rows: 68.7 GB/s of the box's 72 on 20 threads),
// but an expert shared by a verify window's tokens is compute-bound (2 tokens: 48.5 GB/s) - ggml-cpu redoes every block's
// lookup, sign trick, scale and FMA per token, and mx_avx2.cpp's shared lookup saves little.
//
// How, block pair by block pair (ggml's loop takes the blocks two at a time into two accumulators):
//   * one zmm holds both blocks: 128-bit lanes = block ib's low nibbles, its high nibbles, block ib+1's low, its high -
//     the values in ggml's order (MM256_SET_M128I(high, low) per block), looked up once for all the tokens;
//   * AVX-512BW has no vpsignb, so the weights are offset to unsigned (w + 12, 0..24) and maddubs/madd give per 32-bit
//     lane sum((w + 12) y) = sum(w y) + 12 sum(y) EXACTLY (int16 pairs <= 24 * 127 * 2: no saturation); the activation-side
//     12 sum(y) per lane is computed once per call and subtracted - the very int32 lanes of ggml's sign trick;
//   * the scales: fp16(y.d) * e8m0_half(x.e) per block, one float product as ggml's, as a [8 x block ib | 8 x block ib+1]
//     vector; the FMA into one zmm accumulator per token whose low half IS ggml's accum1 (even blocks) and whose high half
//     IS its accum2 (odd blocks), lane for lane, in the same order;
//   * the end: low + high (= accum1 + accum2), ggml's hsum_float_8, ggml's scalar tail for an odd block count.
// Compiled with -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma -mf16c; callers gate on cpu_avx512bw_ok().
#include "strata/kernels/cpu/mx_avx2.hpp"

#include <immintrin.h>

#include <cstddef>
#include <cstring>
#include <new>

namespace strata::kernels::cpu {

namespace {

struct BlockMx {
    uint8_t e;
    uint8_t qs[16];
};
struct BlockQ8 {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(BlockMx) == 17 && sizeof(BlockQ8) == 34, "ggml block layouts");

alignas(16) constexpr int8_t kFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline float e8m0_half(uint8_t x) {
    const uint32_t bits = x < 2 ? 0x00200000u << x : (uint32_t) (x - 1) << 23;
    float r;
    std::memcpy(&r, &bits, sizeof r);
    return r;
}

inline float fp16(uint16_t h) { return _cvtsh_ss(h); }

inline float hsum_float_8(const __m256 x) {   // ggml's
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

/// A buffer of zmm values: resize grows it (the contents are not kept - every build() rewrites all of them).  Not a standard
/// vector: its members would be weak template instances compiled here with AVX-512, which the linker may pick for a caller
/// in any other file (tools/check_isa_guard.py); this one lives in the anonymous namespace, so its code stays this file's.
template <class T>
struct ZBuf {
    T* p = nullptr;
    size_t cap = 0;
    ZBuf() = default;
    ZBuf(const ZBuf&) = delete;
    ZBuf& operator=(const ZBuf&) = delete;
    ~ZBuf() {
        if (p) ::operator delete(p, std::align_val_t(alignof(T)));
    }
    void resize(size_t n) {
        if (n <= cap) return;
        if (p) ::operator delete(p, std::align_val_t(alignof(T)));
        p = static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(alignof(T))));
        cap = n;
    }
    T& operator[](size_t i) { return p[i]; }
    const T& operator[](size_t i) const { return p[i]; }
};

/// One token's activation, repacked for the kernel: per block pair the 64 quant bytes contiguous, the two blocks' fp16
/// scales as floats [8 x d_ib | 8 x d_ib+1], and the exact lane correction 12 * (sum of each 4-byte group).
struct ActPack {
    ZBuf<__m512i> q;             ///< [pair]: block ib's qs (bytes 0..31), block ib+1's (32..63)
    ZBuf<__m512> d;              ///< [pair]
    ZBuf<__m512i> corr;          ///< [pair]: 16 int32 lanes
    void build(const BlockQ8* y, int nb) {
        const int np = nb / 2;
        q.resize((size_t) np);
        d.resize((size_t) np);
        corr.resize((size_t) np);
        const __m512i twelve = _mm512_set1_epi8(12), ones16 = _mm512_set1_epi16(1);
        for (int p = 0; p < np; ++p) {
            const __m256i a = _mm256_loadu_si256((const __m256i*) y[2 * p].qs);
            const __m256i b = _mm256_loadu_si256((const __m256i*) y[2 * p + 1].qs);
            const __m512i v = _mm512_inserti64x4(_mm512_castsi256_si512(a), b, 1);
            q[(size_t) p] = v;
            d[(size_t) p] = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(fp16(y[2 * p].d))),
                                               _mm256_set1_ps(fp16(y[2 * p + 1].d)), 1);
            corr[(size_t) p] = _mm512_madd_epi16(_mm512_maddubs_epi16(twelve, v), ones16);   // 12 * sum of 4 activations
        }
    }
};

template <int NT>
void rows_nt512(const uint8_t* w, size_t row_bytes, int n, const BlockQ8* const* y, const ActPack* ap, float* const* out,
                int r0, int r1) {
    const int nb = n / 32, np = nb / 2;
    const __m512i table = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kFp4));
    const __m512i m4b = _mm512_set1_epi8(0x0f), twelve = _mm512_set1_epi8(12), ones16 = _mm512_set1_epi16(1);
    for (int r = r0; r < r1; ++r) {
        const BlockMx* x = (const BlockMx*) (w + (size_t) r * row_bytes);
        __m512 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
        for (int p = 0; p < np; ++p) {
            const __m128i b0 = _mm_loadu_si128((const __m128i*) x[2 * p].qs);
            const __m128i b1 = _mm_loadu_si128((const __m128i*) x[2 * p + 1].qs);
            // lanes: b0 low nibbles, b0 high, b1 low, b1 high
            __m512i nib = _mm512_castsi128_si512(b0);
            nib = _mm512_inserti32x4(nib, _mm_srli_epi16(b0, 4), 1);
            nib = _mm512_inserti32x4(nib, b1, 2);
            nib = _mm512_inserti32x4(nib, _mm_srli_epi16(b1, 4), 3);
            const __m512i wu = _mm512_add_epi8(_mm512_shuffle_epi8(table, _mm512_and_si512(nib, m4b)), twelve);   // 0..24
            const __m512 ev = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(e8m0_half(x[2 * p].e))),
                                                 _mm256_set1_ps(e8m0_half(x[2 * p + 1].e)), 1);
            for (int t = 0; t < NT; ++t) {
                const __m512i s = _mm512_sub_epi32(_mm512_madd_epi16(_mm512_maddubs_epi16(wu, ap[t].q[(size_t) p]), ones16),
                                                   ap[t].corr[(size_t) p]);
                acc[t] = _mm512_fmadd_ps(_mm512_mul_ps(ap[t].d[(size_t) p], ev), _mm512_cvtepi32_ps(s), acc[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const __m256 lo = _mm512_castps512_ps256(acc[t]), hi = _mm512_extractf32x8_ps(acc[t], 1);
            float sumf = hsum_float_8(_mm256_add_ps(lo, hi));
            for (int jb = 2 * np; jb < nb; ++jb) {   // ggml's scalar tail
                const float d = fp16(y[t][jb].d) * e8m0_half(x[jb].e);
                int sumi1 = 0, sumi2 = 0;
                for (int j = 0; j < 16; ++j) {
                    sumi1 += y[t][jb].qs[j + 0] * kFp4[x[jb].qs[j] & 0xf];
                    sumi2 += y[t][jb].qs[j + 16] * kFp4[x[jb].qs[j] >> 4];
                }
                sumf += d * (sumi1 + sumi2);
            }
            out[t][r] = sumf;
        }
    }
}

/// The activation repacked to the weights' natural unpack order: per pair [y0 0..15 | y1 0..15 | y0 16..31 | y1 16..31],
/// the scales as [d0 x4 | d1 x4 | d0 x4 | d1 x4], the lane corrections in the same order.
struct ActPackN {
    ZBuf<__m512i> q, corr;
    ZBuf<__m512> d;
    void build(const BlockQ8* y, int nb) {
        const int np = nb / 2;
        q.resize((size_t) np);
        d.resize((size_t) np);
        corr.resize((size_t) np);
        const __m512i twelve = _mm512_set1_epi8(12), ones16 = _mm512_set1_epi16(1);
        for (int p = 0; p < np; ++p) {
            const __m128i a0 = _mm_loadu_si128((const __m128i*) y[2 * p].qs), a1 = _mm_loadu_si128((const __m128i*) (y[2 * p].qs + 16));
            const __m128i b0 = _mm_loadu_si128((const __m128i*) y[2 * p + 1].qs), b1 = _mm_loadu_si128((const __m128i*) (y[2 * p + 1].qs + 16));
            __m512i v = _mm512_castsi128_si512(a0);
            v = _mm512_inserti32x4(v, b0, 1);
            v = _mm512_inserti32x4(v, a1, 2);
            v = _mm512_inserti32x4(v, b1, 3);
            q[(size_t) p] = v;
            const float d0 = fp16(y[2 * p].d), d1 = fp16(y[2 * p + 1].d);
            d[(size_t) p] = _mm512_setr_ps(d0, d0, d0, d0, d1, d1, d1, d1, d0, d0, d0, d0, d1, d1, d1, d1);
            corr[(size_t) p] = _mm512_madd_epi16(_mm512_maddubs_epi16(twelve, v), ones16);
        }
    }
};

/// The default (STRATA_MX512_V=1): ActPackN's order, one shuffle and two inserts a block pair.  Measured one thread, rows from
/// DRAM: +7% at 1 token over the lane-insert layout, +1-3% at 2-3; interleaving 2-4 rows (separate chains, shared activation
/// loads) was SLOWER (2 tokens: 3.70 vs 4.77 GB/s) - removed.
template <int NT>
void rows_nt512n(const uint8_t* w, size_t row_bytes, int n, const BlockQ8* const* y, const ActPackN* ap, float* const* out,
                 int r0, int r1) {
    const int nb = n / 32, np = nb / 2;
    const __m512i table = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kFp4));
    const __m512i m4b = _mm512_set1_epi8(0x0f), twelve = _mm512_set1_epi8(12), ones16 = _mm512_set1_epi16(1);
    const __m512i pat = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
    for (int r = r0; r < r1; ++r) {
        const BlockMx* x = (const BlockMx*) (w + (size_t) r * row_bytes);
        __m512 acc[NT];
        for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
        for (int p = 0; p < np; ++p) {
            // [q0 | q1] -> [q0 lo | q1 lo | q0 hi | q1 hi]: the order ActPackN put the activations in
            const __m256i a = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) x[2 * p].qs)),
                                                      _mm_loadu_si128((const __m128i*) x[2 * p + 1].qs), 1);
            const __m512i nib = _mm512_inserti64x4(_mm512_castsi256_si512(a), _mm256_srli_epi16(a, 4), 1);
            const __m512i wu = _mm512_add_epi8(_mm512_shuffle_epi8(table, _mm512_and_si512(nib, m4b)), twelve);
            const __m128 e2 = _mm_setr_ps(e8m0_half(x[2 * p].e), e8m0_half(x[2 * p + 1].e), 0.f, 0.f);
            const __m512 ev = _mm512_permutexvar_ps(pat, _mm512_castps128_ps512(e2));
            for (int t = 0; t < NT; ++t) {
                const __m512i s = _mm512_sub_epi32(_mm512_madd_epi16(_mm512_maddubs_epi16(wu, ap[t].q[(size_t) p]), ones16),
                                                   ap[t].corr[(size_t) p]);
                acc[t] = _mm512_fmadd_ps(_mm512_mul_ps(ap[t].d[(size_t) p], ev), _mm512_cvtepi32_ps(s), acc[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            // lanes [b0 lo 0-3 | b1 lo 4-7 | b0 hi 8-11 | b1 hi 12-15]: ggml's accum1 + accum2, lane for lane, is
            // [m0+m4 .. m3+m7 | m8+m12 .. m11+m15]
            const __m256 lo = _mm512_castps512_ps256(acc[t]), hi = _mm512_extractf32x8_ps(acc[t], 1);
            const __m256 c = _mm256_permute2f128_ps(lo, hi, 0x20), d = _mm256_permute2f128_ps(lo, hi, 0x31);
            float sumf = hsum_float_8(_mm256_add_ps(c, d));
            for (int jb = 2 * np; jb < nb; ++jb) {   // ggml's scalar tail
                const float dd = fp16(y[t][jb].d) * e8m0_half(x[jb].e);
                int sumi1 = 0, sumi2 = 0;
                for (int j = 0; j < 16; ++j) {
                    sumi1 += y[t][jb].qs[j + 0] * kFp4[x[jb].qs[j] & 0xf];
                    sumi2 += y[t][jb].qs[j + 16] * kFp4[x[jb].qs[j] >> 4];
                }
                sumf += dd * (sumi1 + sumi2);
            }
            out[t][r] = sumf;
        }
    }
}

}  // namespace

void mx512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
                int r1) {
    static const int ver = [] { const char* v = std::getenv("STRATA_MX512_V"); return v ? std::atoi(v) : 1; }();
    const BlockQ8* y[8];
    const int nb = n / 32;
    if (ver == 1) {   // the activations in the weights' unpack order (fewer shuffles a block pair)
        thread_local ActPackN pn[8];
        for (int t = 0; t < nt; ++t) {
            y[t] = (const BlockQ8*) act[t];
            pn[t].build(y[t], nb);
        }
        switch (nt) {
            case 1: rows_nt512n<1>(w, row_bytes, n, y, pn, out, r0, r1); break;
            case 2: rows_nt512n<2>(w, row_bytes, n, y, pn, out, r0, r1); break;
            case 3: rows_nt512n<3>(w, row_bytes, n, y, pn, out, r0, r1); break;
            case 4: rows_nt512n<4>(w, row_bytes, n, y, pn, out, r0, r1); break;
            case 5: rows_nt512n<5>(w, row_bytes, n, y, pn, out, r0, r1); break;
            case 6: rows_nt512n<6>(w, row_bytes, n, y, pn, out, r0, r1); break;
            case 7: rows_nt512n<7>(w, row_bytes, n, y, pn, out, r0, r1); break;
            default: rows_nt512n<8>(w, row_bytes, n, y, pn, out, r0, r1); break;
        }
        return;
    }
    thread_local ActPack packs[8];
    for (int t = 0; t < nt; ++t) {
        y[t] = (const BlockQ8*) act[t];
        packs[t].build(y[t], nb);
    }
    switch (nt) {
        case 1: rows_nt512<1>(w, row_bytes, n, y, packs, out, r0, r1); break;
        case 2: rows_nt512<2>(w, row_bytes, n, y, packs, out, r0, r1); break;
        case 3: rows_nt512<3>(w, row_bytes, n, y, packs, out, r0, r1); break;
        case 4: rows_nt512<4>(w, row_bytes, n, y, packs, out, r0, r1); break;
        case 5: rows_nt512<5>(w, row_bytes, n, y, packs, out, r0, r1); break;
        case 6: rows_nt512<6>(w, row_bytes, n, y, packs, out, r0, r1); break;
        case 7: rows_nt512<7>(w, row_bytes, n, y, packs, out, r0, r1); break;
        default: rows_nt512<8>(w, row_bytes, n, y, packs, out, r0, r1); break;
    }
}

}  // namespace strata::kernels::cpu
