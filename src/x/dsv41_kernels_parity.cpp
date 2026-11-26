// src/x/dsv41_kernels_parity.cpp - moex: DeepSeek-V4.1-Flash's kernels (dsv41_kernels.hpp) and the V4 kernels V4.1 runs
// at its own ratios (dsv4_lid_scores / dsv4_lid_select / dsv4_attn at ratios 1 and 2, dsv4_rope_shift) against CPU
// transcriptions of the release's model.py / kernel.py arithmetic (moex/tools/dsv41/ref/ref_kernels.py is the same in torch).
// GPU, synthetic, no model.  DSV41_PORT.md stage S1.
//
// The roundings (FP8 / FP4 fake quants, the half stores, the candidate bits, the one-hot) must agree BITWISE: their CPU
// references do the same float operations, with an independent E4M3 / E2M1 rounding written here (not cuda_fp8.h's).  The
// reductions (the compressor's RMS, the indexer's dot products, the attention) agree to a relative L2 tolerance, and the
// selections as sets from the GPU's own scores.  Every family has a negative control: a deliberately wrong reference must
// FAIL the same comparison.
//
// Exit: 0 all pass, 1 a failure, 77 no CUDA device (skip).
#include "strata/x/dsv41_kernels.hpp"
#include "strata/x/dsv4_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace g = strata::x;

namespace {

int g_fail = 0;

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(2);                                                                             \
        }                                                                                        \
    } while (0)

template <typename T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t n_) : n(n_) { CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T))); }
    Dev(const std::vector<T>& v) : Dev(v.size()) { up(v); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void up(const std::vector<T>& v) { CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> down(size_t m = 0) const {
        std::vector<T> v(m ? m : n);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(v.data(), p, v.size() * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

std::mt19937 rng([] { const char* e = std::getenv("DSV41_PARITY_SEED"); return e ? (unsigned) std::strtoul(e, nullptr, 10) : 20261018u; }());
std::vector<float> randn(size_t n, float sd = 1.0f) {
    std::normal_distribution<float> d(0.0f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        const double e = (double) a[i] - (double) b[i];
        num += e * e;
        den += (double) b[i] * (double) b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

void report(const char* what, double r, double tol) {
    const bool ok = r <= tol && std::isfinite(r);
    printf("%-66s rel %.3e  tol %.0e  %s\n", what, r, tol, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}
void report_neg(const char* what, double r, double tol) {
    const bool ok = !(r <= tol);
    printf("%-66s rel %.3e  tol %.0e  %s (negative control: must fail the check)\n", what, r, tol, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}
void report_bool(const char* what, bool ok, const std::string& detail = "") {
    printf("%-66s %s %s\n", what, ok ? "PASS" : "FAIL", detail.c_str());
    if (!ok) ++g_fail;
}
void report_bool_neg(const char* what, bool equal) {   // a wrong reference must NOT match
    printf("%-66s %s (negative control: must differ)\n", what, !equal ? "PASS" : "FAIL");
    if (equal) ++g_fail;
}

/// Bitwise equality of float vectors (+0 / -0 distinguished: a fake quant keeps the sign of zero as the reference does).
size_t bit_mismatches(const std::vector<float>& a, const std::vector<float>& b) {
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) bad += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
    return bad;
}

uint16_t f2h(float x) {
    const __half h = __float2half_rn(x);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}
float h2f(uint16_t b) {
    __half h;
    std::memcpy(&h, &b, 2);
    return __half2float(h);
}
constexpr uint16_t kSentinel = 0x7e01;   // an F16 NaN no kernel writes

// ------------------------------------------------------------------ the CPU roundings (independent of cuda_fp8.h)

/// 2^ceil(log2(v)) on the bits (kernel.py fast_log2_ceil / fast_pow2; v positive and normal).
float pow2_ceil_log2(float v) {
    uint32_t b;
    std::memcpy(&b, &v, 4);
    const int e = (int) ((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0 ? 1 : 0);
    const uint32_t r = (uint32_t) (e + 127) << 23;
    float f;
    std::memcpy(&f, &r, 4);
    return f;
}

/// Round to nearest, ties to even, at a quantum (a power of two): v / q is exact, nearbyint in the default mode is RNE.
double rne_at(double v, double q) { return std::nearbyint(v / q) * q; }

/// FP8 E4M3 (fn: max 448, no infinities) round to nearest even, saturating: normals 2^-6 .. 448 with 3 mantissa bits,
/// subnormals in steps of 2^-9.
float e4m3(float x) {
    const double a = std::fabs((double) x);
    double r;
    if (a >= 464.0) r = 448.0;   // past the midpoint to the next (unrepresentable) step: saturate (satfinite)
    else {
        int e2 = 0;
        std::frexp(a, &e2);   // a = m 2^e2, m in [0.5, 1): floor(log2 a) = e2 - 1, exactly
        int e = a > 0.0 ? e2 - 1 : -6;
        if (e < -6) e = -6;
        r = rne_at(a, std::ldexp(1.0, e - 3));
        if (r > 448.0) r = 448.0;
    }
    return (float) std::copysign(r, (double) x);
}

/// E2M1 round to nearest even on {0, .5, 1, 1.5, 2, 3, 4, 6} (x already clamped to [-6, 6]): the codes' mantissa bit
/// decides the ties - 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4.
float e2m1(float x, bool ties_away = false) {
    static const float lv[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float a = std::fabs(x);
    int best = 0;
    for (int i = 1; i < 8; ++i) {
        const float d = std::fabs(a - lv[i]), db = std::fabs(a - lv[best]);
        if (d < db) best = i;
        else if (d == db) best = ties_away ? i : (i % 2 == 0 ? i : best);   // even code index = even mantissa
    }
    return std::copysign(lv[best], x);
}

/// act_quant(x, 32, ue8m0, inplace) on one row (width % 32 == 0).  plain_scale: the negative control's amax / 448.
void ref_fq_fp8(float* x, int width, bool plain_scale = false) {
    for (int b0 = 0; b0 < width; b0 += 32) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b0 + i]));
        amax = std::max(amax, 1e-4f);
        const float s = plain_scale ? amax * (1.0f / 448.0f) : pow2_ceil_log2(amax * (1.0f / 448.0f));
        for (int i = 0; i < 32; ++i) {
            const float q = std::min(std::max(x[b0 + i] / s, -448.0f), 448.0f);
            x[b0 + i] = e4m3(q) * s;
        }
    }
}

/// fp4_act_quant(x, block, inplace) on one row: E4M3 or E8M0 scales.  ties_away: the negative control.
void ref_fq_fp4(float* x, int width, int block, bool e4m3_scale, bool ties_away = false) {
    for (int b0 = 0; b0 < width; b0 += block) {
        float amax = 0.0f;
        for (int i = 0; i < block; ++i) amax = std::max(amax, std::fabs(x[b0 + i]));
        float s;
        if (e4m3_scale) {
            amax = std::max(amax, 6.0f * std::ldexp(1.0f, -9));
            s = e4m3(amax / 6.0f);
        } else {
            amax = std::max(amax, 6.0f * std::ldexp(1.0f, -126));
            s = pow2_ceil_log2(amax * (1.0f / 6.0f));
        }
        for (int i = 0; i < block; ++i) x[b0 + i] = e2m1(std::min(std::max(x[b0 + i] / s, -6.0f), 6.0f), ties_away) * s;
    }
}

/// Rows with the cases a fake quant must get right: ordinary normals, a block of tiny values (the scale floor), a block
/// spanning many binades, a zero block, and values placed exactly on rounding midpoints of their block's scale.
std::vector<float> quant_rows(int rows, int width, int block, bool fp4) {
    auto x = randn((size_t) rows * width, 1.5f);
    std::uniform_int_distribution<int> kind(0, 5);
    for (int r = 0; r < rows; ++r)
        for (int b0 = 0; b0 < width; b0 += block) {
            float* v = x.data() + (size_t) r * width + b0;
            switch (kind(rng)) {
            case 0: for (int i = 0; i < block; ++i) v[i] *= 1e-7f; break;                         // under the floor
            case 1: for (int i = 0; i < block; ++i) v[i] *= std::ldexp(1.0f, (i % 24) - 12); break;  // many binades
            case 2: for (int i = 0; i < block; ++i) v[i] = (i == 3) ? -0.0f : 0.0f; break;         // zeros (signed)
            case 3: {   // midpoints: the block's amax pins the scale, the others sit on rounding ties
                if (fp4) {
                    static const float ties[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
                    const float s = 0.5f;   // amax 3 -> E8M0 scale 2^ceil(log2(0.5)) = 0.5; E4M3 scale FP8(0.5) = 0.5
                    v[0] = 3.0f;
                    for (int i = 1; i < block; ++i) v[i] = (i & 1 ? -1.0f : 1.0f) * ties[i % 7] * s;
                } else {
                    v[0] = 448.0f;   // amax 448 -> scale 1: an E4M3 tie is a value halfway between two neighbours
                    for (int i = 1; i < block; ++i) v[i] = (i & 1 ? -1.0f : 1.0f) * (1.0f + (2 * (i % 8) + 1) / 16.0f) * (float) (1 << (i % 6));
                }
                break;
            }
            default: break;
            }
        }
    return x;
}

// ------------------------------------------------------------------ 1. fake quants

void test_fake_quants() {
    {
        const int rows = 97, width = 512;
        auto x = quant_rows(rows, width, 32, false);
        Dev<float> d(x);
        g::dsv41_fq_fp8(d.p, rows, width, nullptr);
        const auto got = d.down();
        auto ref = x, bad = x;
        for (int r = 0; r < rows; ++r) {
            ref_fq_fp8(ref.data() + (size_t) r * width, width);
            ref_fq_fp8(bad.data() + (size_t) r * width, width, true);
        }
        const size_t mm = bit_mismatches(got, ref);
        report_bool("dsv41_fq_fp8 (97 x 512, ties, floors, zeros) bitwise", mm == 0, std::to_string(mm) + " of " + std::to_string(got.size()) + " differ");
        report_bool_neg("dsv41_fq_fp8 vs a reference with a plain amax / 448 scale", bit_mismatches(got, bad) == 0);
        // the attention output's form: one row of 64 x 512 per token
        auto o = quant_rows(3, 64 * 512, 32, false);
        Dev<float> d2(o);
        g::dsv41_fq_fp8(d2.p, 3, 64 * 512, nullptr);
        auto ro = o;
        for (int r = 0; r < 3; ++r) ref_fq_fp8(ro.data() + (size_t) r * 64 * 512, 64 * 512);
        report_bool("dsv41_fq_fp8 (3 x 32768: the attention output) bitwise", bit_mismatches(d2.down(), ro) == 0);
    }
    for (int pass = 0; pass < 2; ++pass) {
        const bool e4 = pass == 0;
        const int block = e4 ? 16 : 32, width = e4 ? 512 : 128, rows = 211;
        auto x = quant_rows(rows, width, block, true);
        Dev<float> d(x);
        g::dsv41_fq_fp4(d.p, rows, width, block, e4, nullptr);
        const auto got = d.down();
        auto ref = x, bad = x;
        for (int r = 0; r < rows; ++r) {
            ref_fq_fp4(ref.data() + (size_t) r * width, width, block, e4);
            ref_fq_fp4(bad.data() + (size_t) r * width, width, block, e4, true);
        }
        const size_t mm = bit_mismatches(got, ref);
        char name[128];
        snprintf(name, sizeof name, "dsv41_fq_fp4 %s (%d x %d, block %d) bitwise", e4 ? "E4M3 scales" : "E8M0 scales", rows, width, block);
        report_bool(name, mm == 0, std::to_string(mm) + " of " + std::to_string(got.size()) + " differ");
        snprintf(name, sizeof name, "dsv41_fq_fp4 %s vs a reference rounding ties away", e4 ? "E4M3" : "E8M0");
        report_bool_neg(name, bit_mismatches(got, bad) == 0);
    }
}

// ------------------------------------------------------------------ 2. compressor, persist, row store

/// The latent of token t at position p (model.py Compressor, F32 throughout, without the reference's BF16 round of the
/// pooled row): ratio 1 RMS(akv_t) * w; ratio 2, p odd: per channel softmax over the slots p - 1 and p, then RMS * w; p even:
/// zeros.  kvp / scp: the slot p - 1's rows.  The sum and the norm in double, rounded once.
void ref_latent(int ratio, int64_t p, const float* kv1, const float* sc1, const float* kvp, const float* scp, const float* w,
                float eps, float* out, bool swap_softmax = false) {
    const int D = 512;
    std::vector<double> c(D);
    if (ratio == 2 && (p & 1) == 0) {
        for (int k = 0; k < D; ++k) out[k] = 0.0f;
        return;
    }
    for (int k = 0; k < D; ++k) {
        if (ratio == 1) {
            c[k] = kv1[k];
            continue;
        }
        const double a0 = scp[k], a1 = sc1[k], m = std::max(a0, a1);
        double e0 = std::exp(a0 - m), e1 = std::exp(a1 - m);
        if (swap_softmax) std::swap(e0, e1);
        c[k] = (kvp[k] * e0 + kv1[k] * e1) / (e0 + e1);
    }
    double ss = 0;
    for (int k = 0; k < D; ++k) ss += c[k] * c[k];
    const double r = 1.0 / std::sqrt(ss / D + eps);
    for (int k = 0; k < D; ++k) out[k] = (float) (c[k] * r * w[k]);
}

void test_compressor() {
    const int D = 512, S = 16;
    const float eps = 1e-20f;   // V4.1's norm_eps
    const auto w = randn(D, 0.3f);
    Dev<float> dw(w);
    for (int ratio : {1, 2}) {
        // one sequence of P positions read whole, then the same read as two windows split at an odd and an even cut
        const int P = 77;
        const auto akv = randn((size_t) P * D, 1.0f), gs = randn((size_t) P * D, 2.0f);
        std::vector<float> ref((size_t) P * D), bad((size_t) P * D);
        for (int p = 0; p < P; ++p) {
            const float* kvp = p > 0 ? akv.data() + (size_t) (p - 1) * D : nullptr;
            const float* scp = p > 0 ? gs.data() + (size_t) (p - 1) * D : nullptr;
            ref_latent(ratio, p, akv.data() + (size_t) p * D, gs.data() + (size_t) p * D, kvp, scp, w.data(), eps, ref.data() + (size_t) p * D);
            ref_latent(ratio, p, akv.data() + (size_t) p * D, gs.data() + (size_t) p * D, kvp, scp, w.data(), eps, bad.data() + (size_t) p * D,
                       true);
        }
        Dev<float> dkv(akv), dgs(gs), lat((size_t) P * D), rkv((size_t) S * D), rsc((size_t) S * D);
        CK(cudaMemset(rkv.p, 0, rkv.n * 4));
        CK(cudaMemset(rsc.p, 0, rsc.n * 4));
        g::dsv41_compress(ratio, dkv.p, ratio == 2 ? dgs.p : nullptr, rkv.p, rsc.p, ratio == 2 ? S : 0, dw.p, eps, lat.p, 0, P, nullptr);
        const auto whole = lat.down();
        char name[128];
        snprintf(name, sizeof name, "dsv41_compress ratio %d, %d positions in one window", ratio, P);
        report(name, rel(whole, ref), 2e-6);
        if (ratio == 2) {
            snprintf(name, sizeof name, "dsv41_compress ratio 2 vs a reference with the slots' weights swapped");
            report_neg(name, rel(whole, bad), 2e-6);
            size_t nz = 0;
            for (int p = 0; p < P; p += 2)
                for (int k = 0; k < D; ++k) nz += whole[(size_t) p * D + k] != 0.0f;
            report_bool("dsv41_compress ratio 2: the rows of even positions are zero", nz == 0);
        }
        // windows: [0, cut) then [cut, P) with the ring persisted between - every row bitwise the whole read's
        for (int cut : {33, 40}) {
            Dev<float> lat2((size_t) P * D);
            CK(cudaMemset(rkv.p, 0, rkv.n * 4));
            CK(cudaMemset(rsc.p, 0, rsc.n * 4));
            g::dsv41_compress(ratio, dkv.p, ratio == 2 ? dgs.p : nullptr, rkv.p, rsc.p, ratio == 2 ? S : 0, dw.p, eps, lat2.p, 0, cut, nullptr);
            if (ratio == 2) g::dsv41_compress_persist(dkv.p, dgs.p, rkv.p, rsc.p, S, 0, cut, nullptr);
            g::dsv41_compress(ratio, dkv.p + (size_t) cut * D, ratio == 2 ? dgs.p + (size_t) cut * D : nullptr, rkv.p, rsc.p, ratio == 2 ? S : 0,
                              dw.p, eps, lat2.p + (size_t) cut * D, cut, P - cut, nullptr);
            snprintf(name, sizeof name, "dsv41_compress ratio %d: windows split at %d == one window (bitwise)", ratio, cut);
            report_bool(name, bit_mismatches(lat2.down(), whole) == 0);
        }
        // the device position (a captured graph): the same rows
        {
            Dev<int64_t> dpos(std::vector<int64_t>{0});
            Dev<float> lat3((size_t) P * D);
            CK(cudaMemset(rkv.p, 0, rkv.n * 4));
            CK(cudaMemset(rsc.p, 0, rsc.n * 4));
            g::dsv41_compress(ratio, dkv.p, ratio == 2 ? dgs.p : nullptr, rkv.p, rsc.p, ratio == 2 ? S : 0, dw.p, eps, lat3.p, 12345, P, nullptr,
                              dpos.p);
            snprintf(name, sizeof name, "dsv41_compress ratio %d: position from the device word (bitwise)", ratio);
            report_bool(name, bit_mismatches(lat3.down(), whole) == 0);
        }
    }
    {   // persist: the ring holds the last S positions' rows at p % S, the earlier ones' rows untouched
        const int T = 37, pos0 = 101;
        const auto akv = randn((size_t) T * D), gs = randn((size_t) T * D);
        std::vector<float> init((size_t) S * D, -7.0f);
        Dev<float> dkv(akv), dgs(gs), rkv(init), rsc(init);
        g::dsv41_compress_persist(dkv.p, dgs.p, rkv.p, rsc.p, S, pos0, T, nullptr);
        auto ekv = init, esc = init;
        for (int t = 0; t < T; ++t) {
            const int r = (pos0 + t) % S;
            std::copy(akv.begin() + (size_t) t * D, akv.begin() + (size_t) (t + 1) * D, ekv.begin() + (size_t) r * D);
            std::copy(gs.begin() + (size_t) t * D, gs.begin() + (size_t) (t + 1) * D, esc.begin() + (size_t) r * D);
        }
        report_bool("dsv41_compress_persist (37 rows into a ring of 16) bitwise",
                    bit_mismatches(rkv.down(), ekv) == 0 && bit_mismatches(rsc.down(), esc) == 0);
        const int T2 = 5;   // a short window: only its rows change
        Dev<float> rkv2(init), rsc2(init);
        g::dsv41_compress_persist(dkv.p, dgs.p, rkv2.p, rsc2.p, S, 7, T2, nullptr);
        auto e2 = init;
        for (int t = 0; t < T2; ++t) std::copy(akv.begin() + (size_t) t * D, akv.begin() + (size_t) (t + 1) * D, e2.begin() + (size_t) ((7 + t) % S) * D);
        report_bool("dsv41_compress_persist (5 rows): the other rows untouched", bit_mismatches(rkv2.down(), e2) == 0);
    }
    for (int ratio : {1, 2}) {   // the row store: completed rows only, as halves, at p / ratio
        const int T = 23, width = 128, rows = 64;
        for (int pos0 : {10, 11}) {
            const auto x = randn((size_t) T * width);
            std::vector<uint16_t> init((size_t) rows * width, kSentinel);
            Dev<float> dx(x);
            Dev<uint16_t> cache(init);
            g::dsv41_store_rows(ratio, dx.p, cache.p, width, rows, pos0, T, nullptr);
            auto exp = init;
            for (int t = 0; t < T; ++t) {
                const int64_t p = pos0 + t;
                if ((p + 1) % ratio) continue;
                for (int k = 0; k < width; ++k) exp[(size_t) (p / ratio) * width + k] = f2h(x[(size_t) t * width + k]);
            }
            const auto got = cache.down();
            char name[128];
            snprintf(name, sizeof name, "dsv41_store_rows ratio %d at position %d (completed rows only) bitwise", ratio, pos0);
            report_bool(name, std::memcmp(got.data(), exp.data(), got.size() * 2) == 0);
        }
    }
}

// ------------------------------------------------------------------ 3. candidates

/// model.py select_candidate_blocks for one query: block scores = their best visible row, the newest block pinned in, the top
/// `keep` (ties to the lower block, as a stable sort - torch's topk leaves them open), only blocks that hold a visible row.
std::vector<uint8_t> ref_candidates(const float* sc, int n_vis, int keep, int bsize, int cbytes, bool pin = true) {
    std::vector<uint8_t> bits((size_t) cbytes, 0);
    const int nblk = (n_vis + bsize - 1) / bsize;
    if (nblk <= keep) {
        for (int b = 0; b < nblk; ++b) bits[(size_t) b / 8] |= (uint8_t) (1u << (b % 8));
        return bits;
    }
    std::vector<double> key((size_t) nblk);
    for (int b = 0; b < nblk; ++b) {
        double m = -INFINITY;
        for (int j = b * bsize; j < std::min(n_vis, (b + 1) * bsize); ++j) m = std::max(m, (double) sc[j]);
        key[(size_t) b] = m;
    }
    if (pin) key[(size_t) ((n_vis - 1) / bsize)] = INFINITY;
    std::vector<int> o((size_t) nblk);
    std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return key[(size_t) a] > key[(size_t) b]; });
    for (int i = 0; i < keep; ++i) bits[(size_t) o[(size_t) i] / 8] |= (uint8_t) (1u << (o[(size_t) i] % 8));
    return bits;
}

void test_candidates() {
    const int bsize = 8;
    struct Case {
        const char* what;
        int keep, ratio, max_rows, T;
        int64_t pos0;
        bool ties;
    };
    const Case cases[] = {
        {"fast path (blocks <= keep: every visible block)", 2048, 1, 4096, 6, 300, false},
        {"ratio 1, keep 16 of ~90 blocks", 16, 1, 1024, 8, 700, false},
        {"ratio 2, keep 16 of ~45 blocks", 16, 2, 1024, 8, 700, false},
        {"ties at the threshold go to the lower blocks", 16, 1, 1024, 4, 600, true},
        {"V4.1's shape: keep 2048 of 2400 blocks (19.2K rows)", 2048, 1, 20000, 3, 19197, false},
    };
    for (const Case& c : cases) {
        const int cbytes = g::dsv41_cand_bytes(c.max_rows, bsize);
        auto sc = randn((size_t) c.T * c.max_rows, 1.0f);
        if (c.ties)   // a few distinct values: many blocks share their maximum
            for (auto& v : sc) v = std::round(v * 2.0f) / 2.0f;
        Dev<float> dsc(sc);
        std::vector<uint8_t> init((size_t) c.T * cbytes, 0xA5);
        Dev<uint8_t> cand(init);
        g::dsv41_candidates(dsc.p, c.max_rows, c.ratio, c.keep, bsize, cand.p, cbytes, c.pos0, c.T, nullptr);
        const auto got = cand.down();
        bool ok = true, neg_equal = true;
        for (int t = 0; t < c.T; ++t) {
            const int n_vis = (int) std::min<int64_t>((c.pos0 + t + 1) / c.ratio, c.max_rows);
            const auto ref = ref_candidates(sc.data() + (size_t) t * c.max_rows, n_vis, c.keep, bsize, cbytes);
            const auto nopin = ref_candidates(sc.data() + (size_t) t * c.max_rows, n_vis, c.keep, bsize, cbytes, false);
            ok = ok && std::memcmp(got.data() + (size_t) t * cbytes, ref.data(), (size_t) cbytes) == 0;
            neg_equal = neg_equal && std::memcmp(got.data() + (size_t) t * cbytes, nopin.data(), (size_t) cbytes) == 0;
        }
        char name[160];
        snprintf(name, sizeof name, "dsv41_candidates: %s", c.what);
        report_bool(name, ok);
        if (c.keep < 2048 && !c.ties) {
            // the pin matters only when the newest block would lose: make it lose, then the no-pin reference must differ
            snprintf(name, sizeof name, "dsv41_candidates: %s vs a reference without the pin", c.what);
            for (int t = 0; t < c.T; ++t) {
                const int n_vis = (int) std::min<int64_t>((c.pos0 + t + 1) / c.ratio, c.max_rows);
                for (int j = (n_vis - 1) / bsize * bsize; j < n_vis; ++j) sc[(size_t) t * c.max_rows + j] = -50.0f;
            }
            dsc.up(sc);
            g::dsv41_candidates(dsc.p, c.max_rows, c.ratio, c.keep, bsize, cand.p, cbytes, c.pos0, c.T, nullptr);
            const auto got2 = cand.down();
            bool eq = true, ok2 = true;
            for (int t = 0; t < c.T; ++t) {
                const int n_vis = (int) std::min<int64_t>((c.pos0 + t + 1) / c.ratio, c.max_rows);
                const auto nopin = ref_candidates(sc.data() + (size_t) t * c.max_rows, n_vis, c.keep, bsize, cbytes, false);
                const auto ref = ref_candidates(sc.data() + (size_t) t * c.max_rows, n_vis, c.keep, bsize, cbytes);
                eq = eq && std::memcmp(got2.data() + (size_t) t * cbytes, nopin.data(), (size_t) cbytes) == 0;
                ok2 = ok2 && std::memcmp(got2.data() + (size_t) t * cbytes, ref.data(), (size_t) cbytes) == 0;
            }
            report_bool("  ... and with the newest block scored lowest, the pin keeps it", ok2);
            report_bool_neg(name, eq);
        }
        (void) neg_equal;
    }
    {   // the mask: unkept blocks' visible rows to -inf, everything else untouched
        const int max_rows = 1024, T = 5, ratio = 1, keep = 16;
        const int64_t pos0 = 700;
        const int cbytes = g::dsv41_cand_bytes(max_rows, bsize);
        const auto sc = randn((size_t) T * max_rows);
        Dev<float> dsc(sc), dsc2(sc);
        Dev<uint8_t> cand((size_t) T * cbytes);
        g::dsv41_candidates(dsc.p, max_rows, ratio, keep, bsize, cand.p, cbytes, pos0, T, nullptr);
        g::dsv41_cand_mask(dsc2.p, max_rows, ratio, cand.p, cbytes, bsize, pos0, T, nullptr);
        const auto bits = cand.down();
        const auto got = dsc2.down();
        auto exp = sc;
        for (int t = 0; t < T; ++t) {
            const int n_vis = (int) std::min<int64_t>((pos0 + t + 1) / ratio, max_rows);
            for (int j = 0; j < n_vis; ++j) {
                const int b = j / bsize;
                if (!((bits[(size_t) t * cbytes + b / 8] >> (b % 8)) & 1)) exp[(size_t) t * max_rows + j] = -INFINITY;
            }
        }
        report_bool("dsv41_cand_mask: unkept visible rows -inf, the rest untouched (bitwise)", bit_mismatches(got, exp) == 0);
    }
    {
        Dev<float> pre((size_t) 7 * 24);
        CK(cudaMemset(pre.p, 0x7f, pre.n * 4));
        g::dsv41_pre_onehot(pre.p, 7, nullptr);
        const auto got = pre.down();
        bool ok = true;
        for (int i = 0; i < 7 * 24; ++i) ok = ok && got[(size_t) i] == (i % 24 == 0 ? 1.0f : 0.0f);
        report_bool("dsv41_pre_onehot: rows {1, 0, ..., 0}", ok);
    }
}

// ------------------------------------------------------------------ 4. V4's kernels at V4.1's ratios

void test_rope_shift() {
    const int T = 9, D = 512;
    const auto x = randn((size_t) T * D);
    const g::Dsv4Rope pc = g::dsv4_rope_pc(160000.0f, 1.0f / 16.0f, 1.0f, 65536, 32.0f, 1.0f);
    Dev<float> a(x), b(x);
    g::dsv4_rope_shift(a.p, a.p, T, 1, D, D - 64, pc, false, 1001, -1, nullptr);
    g::dsv4_rope(b.p, b.p, T, 1, D, D - 64, pc, false, 1000, nullptr);
    report_bool("dsv4_rope_shift(pos0, -1) == dsv4_rope(pos0 - 1) (bitwise)", bit_mismatches(a.down(), b.down()) == 0);
    Dev<float> c(x);
    g::dsv4_rope(c.p, c.p, T, 1, D, D - 64, pc, false, 1001, nullptr);
    report_bool_neg("dsv4_rope_shift(pos0, -1) vs dsv4_rope(pos0) (must differ)", bit_mismatches(a.down(), c.down()) == 0);
}

/// sum_h relu(q16_h . k) w_h with q rounded to F16 (dsv4_lid_scores' definition), the dot in double.
float ref_score(const float* q, const float* kf, const float* w) {
    float score = 0.0f;
    for (int h = 0; h < 64; ++h) {
        double dot = 0;
        for (int d = 0; d < 128; ++d) dot += (double) h2f(f2h(q[h * 128 + d])) * kf[d];
        score = score + std::max((float) dot, 0.0f) * w[h];
    }
    return score;
}

std::set<int> ref_topk(const float* s, int n, int k) {
    std::set<int> out;
    if (n <= k) {
        for (int i = 0; i < n; ++i) out.insert(i);
        return out;
    }
    std::vector<int> o((size_t) n);
    std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return s[a] > s[b]; });
    for (int i = 0; i < k; ++i) out.insert(o[(size_t) i]);
    return out;
}

void test_indexer() {
    const int top_k = 512, IH = 32;
    for (int ratio : {1, 2}) {
        const int T = 4, max_rows = 1601;
        const int64_t pos0 = ratio == 1 ? 900 : 1500;
        // V4.1's 32 heads padded to the kernel's 64 with zero q and zero weights; q and k on the FP4 grid as the engine stores them
        auto q = randn((size_t) T * 64 * 128), w = randn((size_t) T * 64, 0.05f);
        for (int t = 0; t < T; ++t) {
            ref_fq_fp4(q.data() + (size_t) t * 64 * 128, IH * 128, 32, false);
            for (int h = IH; h < 64; ++h) {
                std::fill(q.begin() + ((size_t) t * 64 + h) * 128, q.begin() + ((size_t) t * 64 + h + 1) * 128, 0.0f);
                w[(size_t) t * 64 + h] = 0.0f;
            }
        }
        auto kf = randn((size_t) max_rows * 128);
        for (int r = 0; r < max_rows; ++r) ref_fq_fp4(kf.data() + (size_t) r * 128, 128, 32, false);
        std::vector<uint16_t> kh((size_t) max_rows * 128);
        for (size_t i = 0; i < kh.size(); ++i) kh[i] = f2h(kf[i]);
        Dev<float> dq(q), dw(w), score((size_t) T * max_rows), score2((size_t) T * max_rows);
        Dev<uint16_t> dk(kh);
        Dev<int32_t> sel((size_t) T * top_k), cnt(T);
        g::dsv4_lid_scores(dq.p, dk.p, dw.p, score.p, max_rows, pos0, T, nullptr, nullptr, ratio);
        const auto got = score.down();
        std::vector<float> g_vis, r_vis, bad_vis;
        for (int t = 0; t < T; ++t) {
            const int n_vis = (int) std::min<int64_t>((pos0 + t + 1) / ratio, max_rows);
            for (int j = 0; j < n_vis; ++j) {
                g_vis.push_back(got[(size_t) t * max_rows + j]);
                r_vis.push_back(ref_score(q.data() + (size_t) t * 64 * 128, kf.data() + (size_t) j * 128, w.data() + (size_t) t * 64));
                bad_vis.push_back(ref_score(q.data() + (size_t) t * 64 * 128, kf.data() + (size_t) (j + 1 < max_rows ? j + 1 : j) * 128,
                                            w.data() + (size_t) t * 64));
            }
        }
        char name[128];
        snprintf(name, sizeof name, "dsv4_lid_scores at ratio %d (32 heads padded to 64), visible rows", ratio);
        report(name, rel(g_vis, r_vis), 1e-5);
        snprintf(name, sizeof name, "dsv4_lid_scores at ratio %d vs a reference reading the next row", ratio);
        report_neg(name, rel(g_vis, bad_vis), 1e-5);
        if (g::dsv4_lid_scores_mma(dq.p, dk.p, dw.p, score2.p, max_rows, pos0, T, nullptr, nullptr, ratio)) {
            const auto g2 = score2.down();
            std::vector<float> m_vis;
            for (int t = 0; t < T; ++t) {
                const int n_vis = (int) std::min<int64_t>((pos0 + t + 1) / ratio, max_rows);
                for (int j = 0; j < n_vis; ++j) m_vis.push_back(g2[(size_t) t * max_rows + j]);
            }
            snprintf(name, sizeof name, "dsv4_lid_scores_mma at ratio %d vs the reference", ratio);
            report(name, rel(m_vis, r_vis), 1e-5);
        }
        g::dsv4_lid_select(score.p, max_rows, top_k, sel.p, cnt.p, pos0, T, nullptr, nullptr, ratio);
        const auto s = sel.down();
        const auto c = cnt.down();
        bool ok = true;
        for (int t = 0; t < T; ++t) {
            const int n_vis = (int) std::min<int64_t>((pos0 + t + 1) / ratio, max_rows);
            const auto want = ref_topk(got.data() + (size_t) t * max_rows, n_vis, top_k);
            std::set<int> have;
            for (int i = 0; i < c[(size_t) t]; ++i) have.insert(s[(size_t) t * top_k + i]);
            ok = ok && have == want && c[(size_t) t] == std::min(n_vis, top_k);
        }
        snprintf(name, sizeof name, "dsv4_lid_select at ratio %d: the top 512 of the GPU's scores (sets)", ratio);
        report_bool(name, ok);
    }
}

/// dsv4_attn's definition for one token and head: the raw window's rows and the listed compressed rows, the sink unscaled.
void ref_attn(const float* q, const std::vector<float>& rawf, int raw_rows, const std::vector<float>& compf, const int32_t* sel,
              int n_comp, float sink, float scale, int64_t p, float* out) {
    const int D = 512;
    std::vector<const float*> keys;
    for (int64_t pp = std::max<int64_t>(0, p - 127); pp <= p; ++pp) keys.push_back(rawf.data() + (size_t) (pp % raw_rows) * D);
    for (int c = 0; c < n_comp; ++c) keys.push_back(compf.data() + (size_t) sel[c] * D);
    std::vector<double> sc(keys.size());
    double m = sink;
    for (size_t j = 0; j < keys.size(); ++j) {
        double d = 0;
        for (int k = 0; k < D; ++k) d += (double) q[k] * keys[j][k];
        sc[j] = d * scale;
        m = std::max(m, sc[j]);
    }
    double den = std::exp((double) sink - m);
    std::vector<double> acc(D, 0.0);
    for (size_t j = 0; j < keys.size(); ++j) {
        const double e = std::exp(sc[j] - m);
        den += e;
        for (int k = 0; k < D; ++k) acc[k] += e * keys[j][k];
    }
    for (int k = 0; k < D; ++k) out[k] = (float) (acc[k] / den);
}

/// DSpark's block (nc 2): every row's keys are the positions max(0, b0 - 128) .. b0 + T - 1 (model.py get_dspark_topk_idxs).
void test_attention_block() {
    const int H = 64, D = 512, raw_rows = 192, T = 5;
    for (int64_t pos0 : {(int64_t) 300, (int64_t) 60}) {   // a full window before the block, and a short one
        const auto q = randn((size_t) T * H * D, 0.05f), sinks = randn(H, 1.0f);
        std::vector<float> rawf((size_t) raw_rows * D);
        {
            const auto r = randn(rawf.size());
            for (size_t i = 0; i < rawf.size(); ++i) rawf[i] = h2f(f2h(r[i]));
        }
        std::vector<uint16_t> rawh(rawf.size());
        for (size_t i = 0; i < rawf.size(); ++i) rawh[i] = f2h(rawf[i]);
        Dev<float> dq(q), dsk(sinks), out((size_t) T * H * D);
        Dev<uint16_t> draw(rawh);
        Dev<uint8_t> scratch(g::dsv4_attn_scratch_bytes(T) + 16);
        const float scale = 1.0f / std::sqrt((float) D);
        g::dsv4_attn(dq.p, draw.p, raw_rows, nullptr, 0, nullptr, 512, dsk.p, scale, out.p, reinterpret_cast<float*>(scratch.p), pos0, T,
                     nullptr, nullptr, 2);
        const auto got = out.down();
        std::vector<float> ref((size_t) T * H * D), bad((size_t) T * H * D);
        auto attend = [&](const float* qh, int64_t lo, int64_t hi, float sink, float* o) {
            std::vector<double> sc;
            double m = sink;
            for (int64_t pp = lo; pp <= hi; ++pp) {
                double d = 0;
                for (int k = 0; k < D; ++k) d += (double) qh[k] * rawf[(size_t) (pp % raw_rows) * D + k];
                sc.push_back(d * scale);
                m = std::max(m, sc.back());
            }
            double den = std::exp((double) sink - m);
            std::vector<double> acc(D, 0.0);
            for (int64_t pp = lo; pp <= hi; ++pp) {
                const double e = std::exp(sc[(size_t) (pp - lo)] - m);
                den += e;
                for (int k = 0; k < D; ++k) acc[k] += e * rawf[(size_t) (pp % raw_rows) * D + k];
            }
            for (int k = 0; k < D; ++k) o[k] = (float) (acc[k] / den);
        };
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < H; ++h) {
                const float* qh = q.data() + ((size_t) t * H + h) * D;
                attend(qh, std::max<int64_t>(0, pos0 - 128), pos0 + T - 1, sinks[(size_t) h], ref.data() + ((size_t) t * H + h) * D);
                // the negative control: V4's non-causal window (each row's own 128 positions)
                attend(qh, std::max<int64_t>(0, pos0 + t - 127), pos0 + T - 1, sinks[(size_t) h], bad.data() + ((size_t) t * H + h) * D);
            }
        char name[128];
        snprintf(name, sizeof name, "dsv4_attn block mode (DSpark, 5 rows at %lld)", (long long) pos0);
        report(name, rel(got, ref), 2e-5);
        if (pos0 >= 128) {
            snprintf(name, sizeof name, "dsv4_attn block mode vs V4's per-row non-causal window (at %lld)", (long long) pos0);
            report_neg(name, rel(got, bad), 2e-5);
        }
    }
}

/// The decode start: 1 and 8 rows at positions 0 .. 15, ratios 0 / 1 / 2, every visible row listed (n_vis < top_k) - the
/// engine's first windows; the SIMT kernel and the tensor-core twin each against the reference.
void test_attention_short() {
    const int H = 64, D = 512, top_k = 512, raw_rows = 640;
    for (int ratio : {0, 1, 2})
        for (int T : {1, 8})
            for (int64_t pos0 : {(int64_t) 0, (int64_t) 3, (int64_t) 8}) {
                const int comp_rows = (int) ((pos0 + T) / std::max(ratio, 1)) + 2;
                const auto q = randn((size_t) T * H * D, 0.05f), sinks = randn(H, 1.0f);
                std::vector<float> rawf((size_t) raw_rows * D), compf((size_t) comp_rows * D);
                {
                    const auto r = randn(rawf.size()), c = randn(compf.size());
                    for (size_t i = 0; i < rawf.size(); ++i) rawf[i] = h2f(f2h(r[i]));
                    for (size_t i = 0; i < compf.size(); ++i) compf[i] = h2f(f2h(c[i]));
                }
                std::vector<uint16_t> rawh(rawf.size()), comph(compf.size());
                for (size_t i = 0; i < rawf.size(); ++i) rawh[i] = f2h(rawf[i]);
                for (size_t i = 0; i < compf.size(); ++i) comph[i] = f2h(compf[i]);
                std::vector<int32_t> sel((size_t) T * top_k, -1);
                for (int t = 0; t < T; ++t)
                    if (ratio > 0)
                        for (int64_t j = 0; j < (pos0 + t + 1) / ratio; ++j) sel[(size_t) t * top_k + j] = (int32_t) j;
                Dev<float> dq(q), dsk(sinks), out((size_t) T * H * D), out2((size_t) T * H * D);
                Dev<uint16_t> draw(rawh), dcomp(comph);
                Dev<int32_t> dsel(sel);
                Dev<uint8_t> scratch(g::dsv4_attn_scratch_bytes(T) + 16);
                const float scale = 1.0f / std::sqrt((float) D);
                g::dsv4_attn(dq.p, draw.p, raw_rows, ratio ? dcomp.p : nullptr, ratio, ratio ? dsel.p : nullptr, top_k, dsk.p, scale, out.p,
                             reinterpret_cast<float*>(scratch.p), pos0, T, nullptr);
                const auto got = out.down();
                std::vector<float> ref((size_t) T * H * D);
                for (int t = 0; t < T; ++t) {
                    const int64_t p = pos0 + t;
                    const int n_comp = ratio ? (int) std::min<int64_t>((p + 1) / ratio, top_k) : 0;
                    for (int h = 0; h < H; ++h)
                        ref_attn(q.data() + ((size_t) t * H + h) * D, rawf, raw_rows, compf, sel.data() + (size_t) t * top_k, n_comp,
                                 sinks[(size_t) h], scale, p, ref.data() + ((size_t) t * H + h) * D);
                }
                char name[128];
                snprintf(name, sizeof name, "dsv4_attn (SIMT) ratio %d, %d rows at %lld", ratio, T, (long long) pos0);
                report(name, rel(got, ref), 2e-5);
                if (g::dsv4_attn_mma(dq.p, draw.p, raw_rows, ratio ? dcomp.p : nullptr, ratio, ratio ? dsel.p : nullptr, top_k, dsk.p, scale,
                                     out2.p, reinterpret_cast<float*>(scratch.p), pos0, T, nullptr)) {
                    snprintf(name, sizeof name, "dsv4_attn_mma ratio %d, %d rows at %lld", ratio, T, (long long) pos0);
                    report(name, rel(out2.down(), ref), 2e-5);
                }
            }
}

void test_attention() {
    const int H = 64, D = 512, top_k = 512, raw_rows = 192;
    for (int ratio : {1, 2}) {
        const int T = 3;
        const int64_t pos0 = ratio == 1 ? 700 : 1300;
        const int comp_rows = (int) ((pos0 + T) / ratio) + 2;
        const auto q = randn((size_t) T * H * D, 0.05f), sinks = randn(H, 1.0f);
        std::vector<float> rawf((size_t) raw_rows * D), compf((size_t) comp_rows * D);
        {
            const auto r = randn(rawf.size()), c = randn(compf.size());
            for (size_t i = 0; i < rawf.size(); ++i) rawf[i] = h2f(f2h(r[i]));
            for (size_t i = 0; i < compf.size(); ++i) compf[i] = h2f(f2h(c[i]));
        }
        std::vector<uint16_t> rawh(rawf.size()), comph(compf.size());
        for (size_t i = 0; i < rawf.size(); ++i) rawh[i] = f2h(rawf[i]);
        for (size_t i = 0; i < compf.size(); ++i) comph[i] = f2h(compf[i]);
        std::vector<int32_t> sel((size_t) T * top_k, -1);
        for (int t = 0; t < T; ++t) {   // 512 distinct visible rows, ascending
            const int n_vis = (int) ((pos0 + t + 1) / ratio);
            std::vector<int> all((size_t) n_vis);
            std::iota(all.begin(), all.end(), 0);
            std::shuffle(all.begin(), all.end(), rng);
            all.resize((size_t) std::min(n_vis, top_k));
            std::sort(all.begin(), all.end());
            std::copy(all.begin(), all.end(), sel.begin() + (size_t) t * top_k);
        }
        Dev<float> dq(q), dsk(sinks), out((size_t) T * H * D), out2((size_t) T * H * D);
        Dev<uint16_t> draw(rawh), dcomp(comph);
        Dev<int32_t> dsel(sel);
        Dev<uint8_t> scratch(g::dsv4_attn_scratch_bytes(T) + 16);
        const float scale = 1.0f / std::sqrt((float) D);
        g::dsv4_attn(dq.p, draw.p, raw_rows, dcomp.p, ratio, dsel.p, top_k, dsk.p, scale, out.p, reinterpret_cast<float*>(scratch.p), pos0, T,
                     nullptr);
        std::vector<float> ref((size_t) T * H * D), bad((size_t) T * H * D);
        for (int t = 0; t < T; ++t) {
            const int64_t p = pos0 + t;
            const int n_comp = (int) std::min<int64_t>((p + 1) / ratio, top_k);
            for (int h = 0; h < H; ++h) {
                const float* qh = q.data() + ((size_t) t * H + h) * D;
                ref_attn(qh, rawf, raw_rows, compf, sel.data() + (size_t) t * top_k, n_comp, sinks[(size_t) h], scale, p,
                         ref.data() + ((size_t) t * H + h) * D);
                ref_attn(qh, rawf, raw_rows, compf, sel.data() + (size_t) t * top_k, n_comp / 2, sinks[(size_t) h], scale, p,
                         bad.data() + ((size_t) t * H + h) * D);
            }
        }
        const auto got = out.down();
        char name[128];
        snprintf(name, sizeof name, "dsv4_attn at ratio %d (window + 512 listed rows, sinks)", ratio);
        report(name, rel(got, ref), 2e-5);
        snprintf(name, sizeof name, "dsv4_attn at ratio %d vs a reference with half the listed rows", ratio);
        report_neg(name, rel(got, bad), 2e-5);
        if (g::dsv4_attn_mma(dq.p, draw.p, raw_rows, dcomp.p, ratio, dsel.p, top_k, dsk.p, scale, out2.p, reinterpret_cast<float*>(scratch.p),
                             pos0, T, nullptr)) {
            snprintf(name, sizeof name, "dsv4_attn_mma at ratio %d vs the reference", ratio);
            report(name, rel(out2.down(), ref), 2e-5);
        }
    }
}

// ------------------------------------------------------------------ 5. the engram

/// E4M3 (fn) bits -> float, independent of cuda_fp8.h: s eeee mmm, subnormals at e = 0, NaN at e = 15, m = 7.
float e4m3_bits(uint8_t b) {
    const int sgn = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    return (float) (sgn ? -v : v);
}

void test_engram() {
    {
        const int64_t nr = 48 * 5;
        const int dim = 256;
        std::uniform_int_distribution<int> byte(0, 255), sc(100, 140);
        std::vector<uint8_t> rows((size_t) nr * dim), scales((size_t) nr * dim / 32);
        for (auto& b : rows) {
            do b = (uint8_t) byte(rng); while ((b & 0x7F) == 0x7F);   // no NaN codes
        }
        for (auto& b : scales) b = (uint8_t) sc(rng);
        Dev<uint8_t> dr(rows), ds(scales);
        Dev<float> out((size_t) nr * dim);
        g::dsv41_engram_dequant(dr.p, ds.p, out.p, nr, dim, nullptr);
        std::vector<float> ref((size_t) nr * dim), bad((size_t) nr * dim);
        for (int64_t r = 0; r < nr; ++r)
            for (int d = 0; d < dim; ++d) {
                const float v = e4m3_bits(rows[(size_t) r * dim + d]);
                ref[(size_t) r * dim + d] = v * std::ldexp(1.0f, scales[(size_t) r * (dim / 32) + d / 32] - 127);
                bad[(size_t) r * dim + d] = v * std::ldexp(1.0f, scales[(size_t) r * (dim / 32) + (d / 32 + 1) % (dim / 32)] - 127);
            }
        const auto got = out.down();
        report_bool("dsv41_engram_dequant (240 rows of 256, E8M0 per 32) bitwise", bit_mismatches(got, ref) == 0);
        report_bool_neg("dsv41_engram_dequant vs a reference with the next block's scale", bit_mismatches(got, bad) == 0);
    }
    {   // moex: an IQ4_NL table's rows (unsloth's UD-IQ1_M), each in a 256-byte place as eg_rows copies them
        const int64_t nr = 48 * 5;
        const int dim = 256, nb = dim / 32;
        const size_t stride = 256;
        static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        auto h2f = [](uint16_t h) {   // fp16 bits -> float (exact), independent of cuda_fp16.h
            const int sgn = h >> 15, e = (h >> 10) & 31, m = h & 1023;
            const double v = e == 0 ? std::ldexp((double) m, -24) : std::ldexp(1.0 + m / 1024.0, e - 15);
            return (float) (sgn ? -v : v);
        };
        std::uniform_int_distribution<int> byte(0, 255), mant(0, 0x0FFF);
        std::vector<uint8_t> rows((size_t) nr * stride, 0);
        for (int64_t r = 0; r < nr; ++r)
            for (int b = 0; b < nb; ++b) {
                uint8_t* blk = rows.data() + (size_t) r * stride + (size_t) b * 18;
                const uint16_t h = (uint16_t) ((byte(rng) & 1 ? 0x8000 : 0) | (0x1800 + mant(rng)));   // |d| in [2^-9, 2^-7)
                blk[0] = (uint8_t) (h & 0xFF);
                blk[1] = (uint8_t) (h >> 8);
                for (int q = 0; q < 16; ++q) blk[2 + q] = (uint8_t) byte(rng);
            }
        Dev<uint8_t> dr(rows);
        Dev<float> out((size_t) nr * dim);
        g::dsv41_engram_dequant_iq4nl(dr.p, stride, out.p, nr, dim, nullptr);
        std::vector<float> ref((size_t) nr * dim), bad((size_t) nr * dim);
        for (int64_t r = 0; r < nr; ++r)
            for (int b = 0; b < nb; ++b) {
                const uint8_t* blk = rows.data() + (size_t) r * stride + (size_t) b * 18;
                const float d = h2f((uint16_t) (blk[0] | (blk[1] << 8)));
                for (int j = 0; j < 16; ++j) {   // ggml's dequantize_row_iq4_nl
                    ref[(size_t) r * dim + b * 32 + j] = d * kv[blk[2 + j] & 0x0F];
                    ref[(size_t) r * dim + b * 32 + j + 16] = d * kv[blk[2 + j] >> 4];
                    bad[(size_t) r * dim + b * 32 + 2 * j] = d * kv[blk[2 + j] & 0x0F];   // (the nibbles interleaved: wrong)
                    bad[(size_t) r * dim + b * 32 + 2 * j + 1] = d * kv[blk[2 + j] >> 4];
                }
            }
        const auto got = out.down();
        report_bool("dsv41_engram_dequant_iq4nl (240 rows of 256 at a 256-byte stride) bitwise", bit_mismatches(got, ref) == 0);
        report_bool_neg("dsv41_engram_dequant_iq4nl vs a reference with the nibbles interleaved", bit_mismatches(got, bad) == 0);
    }
    {
        const int n = 5120, hc = 4, T = 5;
        const float eps = 1e-20f;
        const auto R0 = randn((size_t) T * hc * n), kv = randn((size_t) T * (hc + 1) * n, 0.7f), qk = randn((size_t) hc * n, 0.4f);
        Dev<float> dR(R0), dkv(kv), dqk(qk);
        g::dsv41_engram_gate(dR.p, dkv.p, dqk.p, eps, n, hc, T, nullptr);
        const auto got = dR.down();
        std::vector<float> ref = R0, bad = R0;
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < hc; ++c) {
                const float* h = R0.data() + ((size_t) t * hc + c) * n;
                const float* key = kv.data() + (size_t) t * (hc + 1) * n + (size_t) c * n;
                const float* val = kv.data() + (size_t) t * (hc + 1) * n + (size_t) hc * n;
                double hh = 0, kk = 0, hk = 0;
                for (int d = 0; d < n; ++d) {
                    hh += (double) h[d] * h[d];
                    kk += (double) key[d] * key[d];
                    hk += (double) h[d] * qk[(size_t) c * n + d] * key[d];
                }
                const double rstd = 1.0 / std::sqrt(hh / n + eps) / std::sqrt(kk / n + eps);
                const double dot = hk * rstd / std::sqrt((double) n);
                const double gt = 1.0 / (1.0 + std::exp(-std::copysign(std::sqrt(std::max(std::fabs(dot), 1e-6)), dot)));
                const double gb = 1.0 / (1.0 + std::exp(-dot));   // the negative control: no signed square root
                for (int d = 0; d < n; ++d) {
                    ref[((size_t) t * hc + c) * n + d] = (float) (h[d] + gt * val[d]);
                    bad[((size_t) t * hc + c) * n + d] = (float) (h[d] + gb * val[d]);
                }
            }
        // the update's error is relative to the change, not to R: compare R - R0
        std::vector<float> dg(got.size()), dr(got.size()), db(got.size());
        for (size_t i = 0; i < got.size(); ++i) {
            dg[i] = got[i] - R0[i];
            dr[i] = ref[i] - R0[i];
            db[i] = bad[i] - R0[i];
        }
        report("dsv41_engram_gate (5 tokens x 4 copies x 5120): the streams' change", rel(dg, dr), 1e-5);
        report_neg("dsv41_engram_gate vs a reference without the signed sqrt", rel(dg, db), 1e-5);
    }
}

}  // namespace

// ------------------------------------------------------------------ 6. the packed caches (moex, DSV41_BACKLOG #104)

/// A packed row decoded on the host with this file's own E4M3 decode: the halves the F16 cache would hold
void unpack_row(const uint8_t* row, int width, int block, bool e4m3, uint16_t* out, bool bad_scale = false) {
    for (int d = 0; d < width; ++d) {
        const uint32_t c = (row[d / 2] >> (4 * (d & 1))) & 15u, mg = c & 7u;
        const float a = mg < 4u ? 0.5f * (float) mg : mg == 7u ? 6.0f : (float) (mg - 2u);
        const uint8_t sb = (uint8_t) (row[width / 2 + d / block] + (bad_scale ? 1 : 0));
        const float sc = e4m3 ? e4m3_bits(sb) : (float) std::ldexp(1.0, (int) sb - 127);
        out[d] = f2h(((c & 8u) ? -a : a) * sc);
    }
}

void test_packed() {
    char name[160];
    // 1. dsv41_fq_fp4_pack against dsv41_fq_fp4 + dsv41_store_rows: the floats, and every row the store writes
    for (const auto& f : {std::make_tuple(512, 16, true), std::make_tuple(128, 32, false)})
        for (int ratio : {1, 2}) {
            const int width = std::get<0>(f), block = std::get<1>(f);
            const bool e4m3 = std::get<2>(f);
            const int T = 37, cache_rows = 64;
            const int64_t pos0 = 61;
            auto x = randn((size_t) T * width, 2.0f);
            for (int i = 0; i < block; ++i) x[(size_t) 3 * width + i] = 0.0f;                  // a zero block
            for (int i = 0; i < width; ++i) x[(size_t) 5 * width + i] *= 1e-6f;               // tiny rows: small scales
            for (int i = 0; i < width; ++i) x[(size_t) 7 * width + i] *= 1e-30f;
            x[(size_t) 9 * width + 1] = -0.0f;
            const size_t rb = g::dsv41_pack_row_bytes(width, block);
            Dev<float> xa(x), xb(x);
            Dev<uint16_t> half((size_t) cache_rows * width);
            Dev<uint8_t> pk((size_t) cache_rows * rb);
            CK(cudaMemset(half.p, 0, half.n * 2));
            CK(cudaMemset(pk.p, 0, pk.n));
            g::dsv41_fq_fp4(xa.p, T, width, block, e4m3, nullptr);
            g::dsv41_store_rows(ratio, xa.p, half.p, width, cache_rows, pos0, T, nullptr);
            g::dsv41_fq_fp4_pack(xb.p, T, width, block, e4m3, pk.p, cache_rows, ratio, pos0, nullptr);
            const auto fa = xa.down(), fb = xb.down();
            const auto hv = half.down();
            const auto pv = pk.down();
            std::vector<uint16_t> dec((size_t) cache_rows * width, 0), bad((size_t) cache_rows * width, 0);
            for (int b = 0; b < cache_rows; ++b) {
                bool any = false;
                for (size_t i = 0; i < rb; ++i) any |= pv[(size_t) b * rb + i] != 0;
                if (!any) continue;   // a row the store did not write (zero in both)
                unpack_row(&pv[(size_t) b * rb], width, block, e4m3, &dec[(size_t) b * width]);
                unpack_row(&pv[(size_t) b * rb], width, block, e4m3, &bad[(size_t) b * width], true);
            }
            snprintf(name, sizeof name, "dsv41_fq_fp4_pack (%d / %d, ratio %d): the rounded floats", width, block, ratio);
            report_bool(name, memcmp(fa.data(), fb.data(), fa.size() * 4) == 0);
            snprintf(name, sizeof name, "dsv41_fq_fp4_pack (%d / %d, ratio %d): the rows decoded = the F16 cache", width, block, ratio);
            report_bool(name, memcmp(dec.data(), hv.data(), hv.size() * 2) == 0);
            snprintf(name, sizeof name, "dsv41_fq_fp4_pack (%d / %d, ratio %d) decoded with each scale off by one", width, block, ratio);
            report_bool_neg(name, memcmp(bad.data(), hv.data(), hv.size() * 2) == 0);
        }
    // 2. the readers on the same rows kept both ways: the scores and the attention bit for bit
    const int top_k = 512, max_rows = 1601;
    for (int ratio : {1, 2}) {
        const int T = 4;
        const int64_t pos0 = ratio == 1 ? 900 : 1500;
        auto q = randn((size_t) T * 64 * 128), w = randn((size_t) T * 64, 0.05f);
        for (int t = 0; t < T; ++t) ref_fq_fp4(q.data() + (size_t) t * 64 * 128, 64 * 128, 32, false);
        const auto kraw = randn((size_t) max_rows * 128);
        Dev<float> ka(kraw), kb(kraw), dq(q), dw(w);
        Dev<uint16_t> kh((size_t) max_rows * 128);
        Dev<uint8_t> kp((size_t) max_rows * g::dsv41_pack_row_bytes(128, 32));
        g::dsv41_fq_fp4(ka.p, max_rows, 128, 32, false, nullptr);
        g::dsv41_store_rows(1, ka.p, kh.p, 128, max_rows, 0, max_rows, nullptr);
        g::dsv41_fq_fp4_pack(kb.p, max_rows, 128, 32, false, kp.p, max_rows, 1, 0, nullptr);
        Dev<float> s1((size_t) T * max_rows), s2((size_t) T * max_rows);
        CK(cudaMemset(s1.p, 0, s1.n * 4));
        CK(cudaMemset(s2.p, 0, s2.n * 4));
        g::dsv4_lid_scores(dq.p, kh.p, dw.p, s1.p, max_rows, pos0, T, nullptr, nullptr, ratio);
        g::dsv4_lid_scores(dq.p, reinterpret_cast<const uint16_t*>(kp.p), dw.p, s2.p, max_rows, pos0, T, nullptr, nullptr, ratio, true);
        snprintf(name, sizeof name, "dsv4_lid_scores at ratio %d: packed keys = F16 keys, bit for bit", ratio);
        report_bool(name, memcmp(s1.down().data(), s2.down().data(), s1.n * 4) == 0);
        CK(cudaMemset(s1.p, 0, s1.n * 4));
        CK(cudaMemset(s2.p, 0, s2.n * 4));
        if (g::dsv4_lid_scores_mma(dq.p, kh.p, dw.p, s1.p, max_rows, pos0, T, nullptr, nullptr, ratio) &&
            g::dsv4_lid_scores_mma(dq.p, reinterpret_cast<const uint16_t*>(kp.p), dw.p, s2.p, max_rows, pos0, T, nullptr, nullptr,
                                   ratio, true)) {
            snprintf(name, sizeof name, "dsv4_lid_scores_mma at ratio %d: packed keys = F16 keys, bit for bit", ratio);
            report_bool(name, memcmp(s1.down().data(), s2.down().data(), s1.n * 4) == 0);
        }
        // the attention: the window ring F16 as ever, the compressed rows both ways; one token (32 splits) and five (several)
        const int H = 64, D = 512, raw_rows = 192;
        for (int Ta : {1, 5}) {
            const int comp_rows = (int) ((pos0 + Ta) / ratio) + 2;
            const auto qa = randn((size_t) Ta * H * D, 0.05f), sinks = randn(H, 1.0f);
            std::vector<uint16_t> rawh((size_t) raw_rows * D);
            {
                const auto r = randn(rawh.size());
                for (size_t i = 0; i < rawh.size(); ++i) rawh[i] = f2h(r[i]);
            }
            const auto craw = randn((size_t) comp_rows * D);
            Dev<float> ca(craw), cb(craw);
            Dev<uint16_t> ch((size_t) comp_rows * D), draw(rawh);
            Dev<uint8_t> cp((size_t) comp_rows * g::dsv41_pack_row_bytes(D, 16));
            g::dsv41_fq_fp4(ca.p, comp_rows, D, 16, true, nullptr);
            g::dsv41_store_rows(1, ca.p, ch.p, D, comp_rows, 0, comp_rows, nullptr);
            g::dsv41_fq_fp4_pack(cb.p, comp_rows, D, 16, true, cp.p, comp_rows, 1, 0, nullptr);
            std::vector<int32_t> sel((size_t) Ta * top_k, -1);
            for (int t = 0; t < Ta; ++t) {
                const int n_vis = (int) ((pos0 + t + 1) / ratio);
                std::vector<int> all((size_t) n_vis);
                std::iota(all.begin(), all.end(), 0);
                std::shuffle(all.begin(), all.end(), rng);
                all.resize((size_t) std::min(n_vis, top_k));
                std::sort(all.begin(), all.end());
                std::copy(all.begin(), all.end(), sel.begin() + (size_t) t * top_k);
            }
            Dev<float> dqa(qa), dsk(sinks), o1((size_t) Ta * H * D), o2((size_t) Ta * H * D);
            Dev<int32_t> dsel(sel);
            Dev<uint8_t> scratch(g::dsv4_attn_scratch_bytes(Ta) + 16);
            float* sc = reinterpret_cast<float*>(scratch.p);
            const float scale = 1.0f / std::sqrt((float) D);
            const auto* cpk = reinterpret_cast<const uint16_t*>(cp.p);
            g::dsv4_attn(dqa.p, draw.p, raw_rows, ch.p, ratio, dsel.p, top_k, dsk.p, scale, o1.p, sc, pos0, Ta, nullptr);
            g::dsv4_attn(dqa.p, draw.p, raw_rows, cpk, ratio, dsel.p, top_k, dsk.p, scale, o2.p, sc, pos0, Ta, nullptr, nullptr, 0, true);
            snprintf(name, sizeof name, "dsv4_attn at ratio %d, %d token(s): packed rows = F16 rows, bit for bit", ratio, Ta);
            report_bool(name, memcmp(o1.down().data(), o2.down().data(), o1.n * 4) == 0);
            if (g::dsv4_attn_mma(dqa.p, draw.p, raw_rows, ch.p, ratio, dsel.p, top_k, dsk.p, scale, o1.p, sc, pos0, Ta, nullptr) &&
                g::dsv4_attn_mma(dqa.p, draw.p, raw_rows, cpk, ratio, dsel.p, top_k, dsk.p, scale, o2.p, sc, pos0, Ta, nullptr, nullptr, 0,
                                 true)) {
                snprintf(name, sizeof name, "dsv4_attn_mma at ratio %d, %d token(s): packed rows = F16 rows, bit for bit", ratio, Ta);
                report_bool(name, memcmp(o1.down().data(), o2.down().data(), o1.n * 4) == 0);
            }
            if (Ta == 1) {   // the negative control: the packed path reading the next row of every selection must differ
                std::vector<int32_t> sel2 = sel;
                for (auto& v : sel2)
                    if (v >= 0 && v + 1 < comp_rows) ++v;
                Dev<int32_t> dsel2(sel2);
                g::dsv4_attn(dqa.p, draw.p, raw_rows, cpk, ratio, dsel2.p, top_k, dsk.p, scale, o2.p, sc, pos0, Ta, nullptr, nullptr, 0, true);
                g::dsv4_attn(dqa.p, draw.p, raw_rows, ch.p, ratio, dsel.p, top_k, dsk.p, scale, o1.p, sc, pos0, Ta, nullptr);
                snprintf(name, sizeof name, "dsv4_attn at ratio %d: packed rows of shifted selections vs F16", ratio);
                report_bool_neg(name, memcmp(o1.down().data(), o2.down().data(), o1.n * 4) == 0);
            }
        }
    }
}

/// moex (#104): the indexer's scores over 200,000 keys (a ratio-2 source at 400K positions), F16 keys against packed - one
/// token, the tensor-core kernel (the decode's); DSV41_PARITY_ONLY=pack_bench
void bench_packed() {
    const int max_rows = 200000, T = 1;
    const int64_t pos0 = 2 * (int64_t) max_rows - 1;
    auto q = randn((size_t) T * 64 * 128), w = randn((size_t) T * 64, 0.05f);
    const auto kraw = randn((size_t) max_rows * 128);
    Dev<float> ka(kraw), kb(kraw), dq(q), dw(w), sc((size_t) T * max_rows);
    Dev<uint16_t> kh((size_t) max_rows * 128);
    Dev<uint8_t> kp((size_t) max_rows * g::dsv41_pack_row_bytes(128, 32));
    g::dsv41_fq_fp4(ka.p, max_rows, 128, 32, false, nullptr);
    for (int r0 = 0; r0 < max_rows; r0 += 50000)   // (the store's grid: at most 65,535 rows a launch)
        g::dsv41_store_rows(1, ka.p + (size_t) r0 * 128, kh.p, 128, max_rows, r0, std::min(50000, max_rows - r0), nullptr);
    g::dsv41_fq_fp4_pack(kb.p, max_rows, 128, 32, false, kp.p, max_rows, 1, 0, nullptr);
    auto time_ms = [](auto f) {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        f();
        cudaEventRecord(a);
        for (int i = 0; i < 50; ++i) f();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        return ms / 50;
    };
    for (bool mma : {true, false}) {
        const float t16 = time_ms([&] {
            if (!mma || !g::dsv4_lid_scores_mma(dq.p, kh.p, dw.p, sc.p, max_rows, pos0, T, nullptr, nullptr, 2))
                g::dsv4_lid_scores(dq.p, kh.p, dw.p, sc.p, max_rows, pos0, T, nullptr, nullptr, 2);
        });
        const auto* kpp = reinterpret_cast<const uint16_t*>(kp.p);
        const float tpk = time_ms([&] {
            if (!mma || !g::dsv4_lid_scores_mma(dq.p, kpp, dw.p, sc.p, max_rows, pos0, T, nullptr, nullptr, 2, true))
                g::dsv4_lid_scores(dq.p, kpp, dw.p, sc.p, max_rows, pos0, T, nullptr, nullptr, 2, true);
        });
        printf("the indexer's scores, 1 token x 200,000 keys (%s): F16 keys %.3f ms, packed %.3f ms\n", mma ? "tensor cores" : "SIMT",
               t16, tpk);
    }
}

/// moex (DSV41_PLAN §56 C): the index sources after cand_src scoring only the candidates' rows (dsv41_cand_list, then
/// dsv4_lid_scores_mma over the list) against the full scan: after the candidates' mask the scores are the same bits on every
/// visible row and the top-512 the same - F16 and packed keys, ratios 1 and 2, one token and a 37-row block, few blocks kept,
/// every block kept, only the pinned newest; a list shifted by a block must differ.
void test_cand_only() {
    if (!g::dsv4_mma_supported()) {
        printf("  (no tensor cores here: the candidates-only scores are the MMA kernel's - skipped)\n");
        return;
    }
    char name[256];
    const int bsize = 8, top_k = 512, max_rows = 3001;
    for (int ratio : {1, 2})
        for (int T : {1, 37})
            for (int keep : {40, 1000, 1}) {
                const int64_t pos0 = ratio == 1 ? 2900 - T : 5800 - T;   // n_vis around 2,900 (not a multiple of 8 or 64)
                auto q = randn((size_t) T * 64 * 128), w = randn((size_t) T * 64, 0.05f);
                for (int t = 0; t < T; ++t) ref_fq_fp4(q.data() + (size_t) t * 64 * 128, 64 * 128, 32, false);
                const auto kraw = randn((size_t) max_rows * 128), src_sc = randn((size_t) T * max_rows);
                Dev<float> ka(kraw), kb(kraw), dq(q), dw(w), dsrc(src_sc);
                Dev<uint16_t> kh((size_t) max_rows * 128);
                Dev<uint8_t> kp((size_t) max_rows * g::dsv41_pack_row_bytes(128, 32));
                g::dsv41_fq_fp4(ka.p, max_rows, 128, 32, false, nullptr);
                g::dsv41_store_rows(1, ka.p, kh.p, 128, max_rows, 0, max_rows, nullptr);
                g::dsv41_fq_fp4_pack(kb.p, max_rows, 128, 32, false, kp.p, max_rows, 1, 0, nullptr);
                const int cbytes = g::dsv41_cand_bytes(max_rows, bsize);
                Dev<uint8_t> cand((size_t) T * cbytes);
                g::dsv41_candidates(dsrc.p, max_rows, ratio, keep, bsize, cand.p, cbytes, pos0, T, nullptr);
                Dev<int32_t> list((size_t) T * keep), count((size_t) T);
                g::dsv41_cand_list(cand.p, cbytes, max_rows, ratio, bsize, list.p, count.p, keep, pos0, T, nullptr);
                for (bool packed : {false, true}) {
                    const uint16_t* K = packed ? reinterpret_cast<const uint16_t*>(kp.p) : kh.p;
                    Dev<float> s_full((size_t) T * max_rows), s_cand((size_t) T * max_rows);
                    CK(cudaMemset(s_full.p, 0, s_full.n * 4));
                    CK(cudaMemset(s_cand.p, 0x7f, s_cand.n * 4));   // other bits than the full run's: the mask must cover them
                    g::dsv4_lid_scores_mma(dq.p, K, dw.p, s_full.p, max_rows, pos0, T, nullptr, nullptr, ratio, packed);
                    g::dsv4_lid_scores_mma(dq.p, K, dw.p, s_cand.p, max_rows, pos0, T, nullptr, nullptr, ratio, packed, list.p,
                                           count.p, keep, bsize);
                    g::dsv41_cand_mask(s_full.p, max_rows, ratio, cand.p, cbytes, bsize, pos0, T, nullptr);
                    g::dsv41_cand_mask(s_cand.p, max_rows, ratio, cand.p, cbytes, bsize, pos0, T, nullptr);
                    const auto a = s_full.down(), b = s_cand.down();
                    bool same = true;
                    for (int t = 0; t < T; ++t) {
                        const int n_vis = (int) std::min<int64_t>((pos0 + t + 1) / ratio, max_rows);
                        same = same && std::memcmp(a.data() + (size_t) t * max_rows, b.data() + (size_t) t * max_rows,
                                                   (size_t) n_vis * 4) == 0;
                    }
                    snprintf(name, sizeof name, "dsv4_lid_scores_mma over the candidates (ratio %d, T %d, keep %d, %s keys): the masked "
                             "scores bit for bit", ratio, T, keep, packed ? "packed" : "F16");
                    report_bool(name, same);
                    Dev<int32_t> sel_a((size_t) T * top_k), sel_b((size_t) T * top_k);
                    g::dsv4_lid_select(s_full.p, max_rows, top_k, sel_a.p, nullptr, pos0, T, nullptr, nullptr, ratio);
                    g::dsv4_lid_select(s_cand.p, max_rows, top_k, sel_b.p, nullptr, pos0, T, nullptr, nullptr, ratio);
                    const auto xa = sel_a.down(), xb = sel_b.down();
                    snprintf(name, sizeof name, "  ... and the top-%d the same", top_k);
                    report_bool(name, std::memcmp(xa.data(), xb.data(), xa.size() * 4) == 0);
                    {   // moex (DSV41_PLAN §59 P2): V4.1's 32 heads - on q and w with heads 32..63 zero (as the engine pads them),
                        // the 32-head tile's scores, full and over the candidates, are the padded 64-head tile's bits
                        auto q32 = q;
                        auto w32 = w;
                        for (int t = 0; t < T; ++t)
                            for (int h = 32; h < 64; ++h) {
                                std::fill(q32.begin() + ((size_t) t * 64 + h) * 128, q32.begin() + ((size_t) t * 64 + h + 1) * 128, 0.0f);
                                w32[(size_t) t * 64 + h] = 0.0f;
                            }
                        Dev<float> dq32(q32), dw32(w32), f64((size_t) T * max_rows), f32((size_t) T * max_rows),
                            c64((size_t) T * max_rows), c32((size_t) T * max_rows);
                        for (Dev<float>* d : {&f64, &f32, &c64, &c32}) CK(cudaMemset(d->p, 0x7f, d->n * 4));
                        g::dsv4_lid_scores_mma(dq32.p, K, dw32.p, f64.p, max_rows, pos0, T, nullptr, nullptr, ratio, packed);
                        g::dsv4_lid_scores_mma(dq32.p, K, dw32.p, f32.p, max_rows, pos0, T, nullptr, nullptr, ratio, packed, nullptr,
                                               nullptr, 0, 0, 32);
                        g::dsv4_lid_scores_mma(dq32.p, K, dw32.p, c64.p, max_rows, pos0, T, nullptr, nullptr, ratio, packed, list.p,
                                               count.p, keep, bsize);
                        g::dsv4_lid_scores_mma(dq32.p, K, dw32.p, c32.p, max_rows, pos0, T, nullptr, nullptr, ratio, packed, list.p,
                                               count.p, keep, bsize, 32);
                        const auto a64 = f64.down(), a32 = f32.down(), b64 = c64.down(), b32 = c32.down();
                        snprintf(name, sizeof name, "dsv4_lid_scores_mma, 32 heads (ratio %d, T %d, keep %d, %s keys): the padded "
                                 "64-head tile's bits, full and over the candidates", ratio, T, keep, packed ? "packed" : "F16");
                        report_bool(name, std::memcmp(a64.data(), a32.data(), a64.size() * 4) == 0 &&
                                              std::memcmp(b64.data(), b32.data(), b64.size() * 4) == 0);
                    }
                    if (keep == 40 && !packed) {   // a list one block off must not give the same scores
                        auto bad = list.down();
                        for (auto& v : bad) v = v + 1;
                        Dev<int32_t> dbad(bad);
                        CK(cudaMemset(s_cand.p, 0, s_cand.n * 4));
                        g::dsv4_lid_scores_mma(dq.p, K, dw.p, s_cand.p, max_rows, pos0, T, nullptr, nullptr, ratio, packed, dbad.p,
                                               count.p, keep, bsize);
                        g::dsv41_cand_mask(s_cand.p, max_rows, ratio, cand.p, cbytes, bsize, pos0, T, nullptr);
                        const auto c = s_cand.down();
                        snprintf(name, sizeof name, "  ... vs a list shifted by a block (ratio %d, T %d)", ratio, T);
                        report_bool_neg(name, std::memcmp(a.data(), c.data(), (size_t) std::min<int64_t>((pos0 + 1) / ratio, max_rows) * 4) == 0);
                    }
                }
                if (keep == 40) {   // the list itself: every kept visible block, ascending
                    const auto bits = cand.down();
                    const auto lst = list.down(), cnt = count.down();
                    bool ok = true;
                    for (int t = 0; t < T; ++t) {
                        const int n_vis = (int) std::min<int64_t>((pos0 + t + 1) / ratio, max_rows);
                        std::vector<int32_t> exp;
                        for (int blk = 0; blk < (n_vis + bsize - 1) / bsize; ++blk)
                            if ((bits[(size_t) t * cbytes + blk / 8] >> (blk % 8)) & 1) exp.push_back(blk);
                        ok = ok && cnt[(size_t) t] == (int32_t) exp.size() &&
                             std::equal(exp.begin(), exp.end(), lst.begin() + (std::ptrdiff_t) t * keep);
                    }
                    snprintf(name, sizeof name, "dsv41_cand_list (ratio %d, T %d): the kept visible blocks, ascending", ratio, T);
                    report_bool(name, ok);
                }
            }
}

/// moex (DSV41_PLAN §56 C): DSV41_PARITY_ONLY=cand_bench - one token's scores at an index source after cand_src, the full scan
/// against the candidates' rows (2,048 blocks of 8), packed keys, at 300K, 600K and 1M visible rows (ratio 1).
void bench_cand() {
    const int bsize = 8, keep = 2048;
    for (int rows : {300000, 600000, 1000000}) {
        const int max_rows = rows + 16;
        const int64_t pos0 = rows - 1;
        auto q = randn((size_t) 64 * 128), w = randn((size_t) 64, 0.05f);
        ref_fq_fp4(q.data(), 64 * 128, 32, false);
        Dev<float> dq(q), dw(w), sc((size_t) max_rows), src(randn((size_t) max_rows));
        Dev<uint8_t> kp((size_t) max_rows * g::dsv41_pack_row_bytes(128, 32));
        {
            Dev<float> kraw(randn((size_t) max_rows * 128));
            g::dsv41_fq_fp4_pack(kraw.p, max_rows, 128, 32, false, kp.p, max_rows, 1, 0, nullptr);
        }
        const int cbytes = g::dsv41_cand_bytes(max_rows, bsize);
        Dev<uint8_t> cand((size_t) cbytes);
        Dev<int32_t> list((size_t) keep), count(1);
        g::dsv41_candidates(src.p, max_rows, 1, keep, bsize, cand.p, cbytes, pos0, 1, nullptr);
        const auto* K = reinterpret_cast<const uint16_t*>(kp.p);
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        float ms_full = 0, ms_cand = 0;
        for (int arm = 0; arm < 2; ++arm) {
            for (int it = 0; it < 60; ++it) {
                if (it == 10) CK(cudaEventRecord(e0));
                if (arm == 0) {
                    g::dsv4_lid_scores_mma(dq.p, K, dw.p, sc.p, max_rows, pos0, 1, nullptr, nullptr, 1, true);
                } else {
                    g::dsv41_cand_list(cand.p, cbytes, max_rows, 1, bsize, list.p, count.p, keep, pos0, 1, nullptr);
                    g::dsv4_lid_scores_mma(dq.p, K, dw.p, sc.p, max_rows, pos0, 1, nullptr, nullptr, 1, true, list.p, count.p, keep,
                                           bsize);
                }
                g::dsv41_cand_mask(sc.p, max_rows, 1, cand.p, cbytes, bsize, pos0, 1, nullptr);
            }
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(arm == 0 ? &ms_full : &ms_cand, e0, e1));
        }
        printf("  %d visible rows: the full scan + mask %.3f ms, the candidates' rows + their list + mask %.3f ms a call\n", rows,
               ms_full / 50, ms_cand / 50);
        CK(cudaEventDestroy(e0));
        CK(cudaEventDestroy(e1));
    }
}

int main() {
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd == 0) {
        printf("no CUDA device: skipped\n");
        return 77;
    }
    if (const char* only = std::getenv("DSV41_PARITY_ONLY"); only && std::string(only) == "pack_bench") {
        bench_packed();
        return 0;
    }
    if (const char* only = std::getenv("DSV41_PARITY_ONLY"); only && std::string(only) == "cand_bench") {
        bench_cand();
        return 0;
    }
    if (const char* only = std::getenv("DSV41_PARITY_ONLY"); only && std::string(only) == "cand") {
        test_cand_only();
        printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return g_fail ? 1 : 0;
    }
    test_fake_quants();
    test_compressor();
    test_candidates();
    test_rope_shift();
    test_indexer();
    test_attention();
    test_attention_short();
    test_attention_block();
    test_engram();
    test_packed();
    test_cand_only();
    printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
