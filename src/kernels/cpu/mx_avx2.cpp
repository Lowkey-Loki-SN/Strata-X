// src/kernels/cpu/mx_avx2.cpp - moex: AVX-2 multi-token dot products for MXFP4 rows against Q8_0 activations.
//
// The MXFP4 releases (DeepSeek-V4-Flash, MiMo-V2.6, gpt-oss) ship their routed experts as MXFP4, and ggml-cpu's dot is
// single-token: a verify window's tokens each re-decode every weight block.  Here a pair of 32-weight blocks is looked
// up once (nibble -> E2M1 value via pshufb) for all the tokens.  Per token the arithmetic is ggml-cpu's AVX2
// ggml_vec_dot_mxfp4_q8_0 operation for operation: the even blocks into one accumulator and the odd blocks into
// another, the same sign trick + maddubs + madd, the same scale (fp16 d times the E8M0 half), the same FMA order, the
// same horizontal sum, the same scalar tail.  So a token's result is bitwise the same alone, in any group, and as
// ggml-cpu's (build/mx_avx2_parity checks all three) - no #152 grouping rule is needed for these rows.
// Compiled with -mavx2 -mfma -mf16c (CMakeLists.txt); callers gate on cpu_avx2_ok().
#include "strata/kernels/cpu/mx_avx2.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {

namespace {

struct BlockMx {   // ggml's block_mxfp4
    uint8_t e;
    uint8_t qs[16];
};
struct BlockQ8 {   // ggml's block_q8_0
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(BlockMx) == 17 && sizeof(BlockQ8) == 34, "ggml block layouts");

alignas(16) constexpr int8_t kFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};   // kvalues_fp4

inline float e8m0_half(uint8_t x) {   // ggml_e8m0_to_fp32_half (ggml-cpu's x86 table holds exactly these values)
    const uint32_t bits = x < 2 ? 0x00200000u << x : (uint32_t) (x - 1) << 23;
    float r;
    std::memcpy(&r, &bits, sizeof r);
    return r;
}

inline float fp16(uint16_t h) { return _cvtsh_ss(h); }   // exact, as every fp16 -> fp32 conversion

inline float hsum_float_8(const __m256 x) {   // ggml's
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {   // ggml's
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

inline __m256i lookup(const __m128i bits, const __m128i values128, const __m128i m4b) {
    // ggml: MM256_SET_M128I(shuffle(high nibbles), shuffle(low nibbles)) - the low nibbles are values 0..15
    return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(bits, m4b))),
                                   _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(bits, 4), m4b)), 1);
}

template <int NT>
void rows_nt(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1, int pf) {
    const int nb = n / 32;
    const __m128i values128 = _mm_loadu_si128((const __m128i*) kFp4);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);
    const BlockQ8* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const BlockQ8*) act[t];
    for (int r = r0; r < r1; ++r) {
        const BlockMx* x = (const BlockMx*) (w + (size_t) r * row_bytes);
        if (pf > 0 && r + pf < r1) {   // STRATA_MX256_PF: the row pf ahead into L1 (the L2 streamer stops at 4 KiB pages)
            const char* p = (const char*) (w + (size_t) (r + pf) * row_bytes);
            for (size_t o = 0; o < row_bytes; o += 64) _mm_prefetch(p + o, _MM_HINT_T0);
        }
        __m256 accum1[NT], accum2[NT];
        for (int t = 0; t < NT; ++t) {
            accum1[t] = _mm256_setzero_ps();
            accum2[t] = _mm256_setzero_ps();
        }
        int ib = 0;
        for (; ib + 1 < nb; ib += 2) {
            const __m256i q4b_1 = lookup(_mm_loadu_si128((const __m128i*) x[ib + 0].qs), values128, m4b);
            const __m256i q4b_2 = lookup(_mm_loadu_si128((const __m128i*) x[ib + 1].qs), values128, m4b);
            const float e1 = e8m0_half(x[ib + 0].e), e2 = e8m0_half(x[ib + 1].e);
            for (int t = 0; t < NT; ++t) {
                const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*) y[t][ib + 0].qs);
                const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*) y[t][ib + 1].qs);
                const __m256i p_1 = _mm256_madd_epi16(mul_add_epi8(q4b_1, q8b_1), mone);
                const __m256i p_2 = _mm256_madd_epi16(mul_add_epi8(q4b_2, q8b_2), mone);
                const __m256 scale0 = _mm256_set1_ps(fp16(y[t][ib + 0].d) * e1);
                const __m256 scale1 = _mm256_set1_ps(fp16(y[t][ib + 1].d) * e2);
                accum1[t] = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(p_1), accum1[t]);
                accum2[t] = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(p_2), accum2[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            float sumf = hsum_float_8(_mm256_add_ps(accum1[t], accum2[t]));
            for (int jb = ib; jb < nb; ++jb) {   // ggml's scalar tail (an odd block count; not the experts' 4096 / 2048)
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

}  // namespace

void mx256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
                int r1) {
    static const int pf = [] { const char* v = std::getenv("STRATA_MX256_PF"); return v ? std::max(0, std::atoi(v)) : 0; }();
    switch (nt) {
        case 1: rows_nt<1>(w, row_bytes, n, act, out, r0, r1, pf); break;
        case 2: rows_nt<2>(w, row_bytes, n, act, out, r0, r1, pf); break;
        case 3: rows_nt<3>(w, row_bytes, n, act, out, r0, r1, pf); break;
        case 4: rows_nt<4>(w, row_bytes, n, act, out, r0, r1, pf); break;
        case 5: rows_nt<5>(w, row_bytes, n, act, out, r0, r1, pf); break;
        case 6: rows_nt<6>(w, row_bytes, n, act, out, r0, r1, pf); break;
        case 7: rows_nt<7>(w, row_bytes, n, act, out, r0, r1, pf); break;
        default: rows_nt<8>(w, row_bytes, n, act, out, r0, r1, pf); break;
    }
}

}  // namespace strata::kernels::cpu
