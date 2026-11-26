// src/x/dsv4_kernels_parity.cpp - moex: every DeepSeek-V4-Flash attention-side kernel (dsv4_kernels.hpp) against CPU
// references transcribed from ggml-cpu, on random inputs with DSV4's shapes.  GPU, synthetic, no model.
//
// The references transcribe ggml-cpu's arithmetic (file:line of mainline ~/AI/llama.cpp in the comments), they do not call
// ggml: this test builds without it.  Where a kernel follows ggml-cuda's formula and that differs from ggml-cpu's by more
// than ulps (RoPE's theta, the indexer's F16 q), the CUDA formula is transcribed as well and checked tightly; the ggml-cpu
// comparison then carries a tolerance justified next to it.  Tolerances are relative L2 over each output; integer outputs
// (expert ids, selected rows) must agree as sets; F16 cache rows are compared in F16 ulps, and bitwise wherever the
// arithmetic is identical (windowed vs whole-sequence, device position vs argument, graph replay vs direct calls).  Every
// family has a negative control: a deliberately wrong reference must FAIL the same comparison.  The tensor-core twins
// (dsv4_lid_scores_mma, dsv4_attn_mma) are held to the same references and to their SIMT kernels when the card has
// sm_80 tensor cores; timings (RTX 3080 class) are printed for the long-context cases.
//
// Exit: 0 all pass, 1 a failure, 77 no CUDA device (skip).
#include "strata/x/dsv4_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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

/// DSV4_PARITY_SEED overrides the seed (the tolerances must not depend on one draw).
std::mt19937 rng([] { const char* e = std::getenv("DSV4_PARITY_SEED"); return e ? (unsigned) std::strtoul(e, nullptr, 10) : 20261007u; }());
std::vector<float> randn(size_t n, float sd = 1.0f, float mean = 0.0f) {
    std::normal_distribution<float> d(mean, sd);
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
    printf("%-62s rel %.3e  tol %.0e  %s\n", what, r, tol, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

/// A negative control: the metric against a deliberately wrong reference must EXCEED the tolerance.
void report_neg(const char* what, double r, double tol) {
    const bool ok = !(r <= tol);
    printf("%-62s rel %.3e  tol %.0e  %s (negative control: must fail the check)\n", what, r, tol, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

void report_bool(const char* what, bool ok, const std::string& detail = "") {
    printf("%-62s %s %s\n", what, ok ? "PASS" : "FAIL", detail.c_str());
    if (!ok) ++g_fail;
}

uint16_t f2h(float x) {
    const __half h = __float2half_rn(x);
    uint16_t b;
    memcpy(&b, &h, 2);
    return b;
}
float h2f(uint16_t b) {
    __half h;
    memcpy(&h, &b, 2);
    return __half2float(h);
}
constexpr uint16_t kSentinel = 0x7e01;   // an F16 NaN no kernel writes: cache rows nothing may touch
std::vector<uint16_t> to_f16(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = f2h(v[i]);
    return o;
}
std::vector<float> from_f16(const std::vector<uint16_t>& v) {
    std::vector<float> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = h2f(v[i]);
    return o;
}

/// F16 rows of width d compared: the largest distance in F16 ulps (ordered bit patterns), how many differ, rel L2 of the
/// values, and `bound`: the largest |a - b| / (ulp16(|b|) + tau * rms(row of b)) - one F16 rounding step plus an F32-level
/// error relative to the row's scale (near-zero outputs of the rope's cancellation or the Hadamard's mixing count many
/// F16 ulps for an absolute difference far below the row's precision).
struct F16Cmp {
    int max_ulp = 0;
    size_t n_diff = 0;
    double rel = 0;
    double bound = 0;
    size_t worst = 0;
};
double ulp16(float x) {
    const float a = std::fabs(x);
    if (a < 6.103515625e-05f) return 5.9604644775390625e-08;   // subnormal spacing 2^-24
    int e;
    std::frexp(a, &e);   // a in [2^(e-1), 2^e)
    return std::ldexp(1.0, e - 1 - 10);
}
F16Cmp cmp_f16(const uint16_t* a, const uint16_t* b, size_t n, int d = 0, double tau = 1e-5) {
    auto ord = [](uint16_t h) { return (h & 0x8000) ? -(int) (h & 0x7fff) : (int) (h & 0x7fff); };
    F16Cmp c;
    std::vector<float> fa(n), fb(n);
    for (size_t i = 0; i < n; ++i) {
        const int dd = std::abs(ord(a[i]) - ord(b[i]));
        c.max_ulp = std::max(c.max_ulp, dd);
        c.n_diff += dd != 0;
        fa[i] = h2f(a[i]);
        fb[i] = h2f(b[i]);
    }
    c.rel = rel(fa, fb);
    if (d > 0)
        for (size_t r = 0; r < n / d; ++r) {
            double ss = 0;
            for (int k = 0; k < d; ++k) ss += (double) fb[r * d + k] * fb[r * d + k];
            const double rms = std::sqrt(ss / d);
            for (int k = 0; k < d; ++k) {
                const size_t i = r * d + k;
                const double q = std::fabs((double) fa[i] - fb[i]) / (ulp16(fb[i]) + tau * rms);
                if (q > c.bound) {
                    c.bound = q;
                    c.worst = i;
                }
            }
        }
    return c;
}

bool bitwise(const void* a, const void* b, size_t bytes) { return memcmp(a, b, bytes) == 0; }

/// The mean time of `f` (enqueued on the null stream) over `iters` calls after one warm-up call (the first launch of a
/// kernel includes its lazy module load).
template <typename F>
float time_ms(F f, int iters = 20) {
    f();
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) f();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    return ms / iters;
}

template <typename T>
std::vector<T> slice(const std::vector<T>& v, size_t off, size_t n) {
    return std::vector<T>(v.begin() + off, v.begin() + off + n);
}

// ================================================================== references (ggml-cpu, mainline ~/AI/llama.cpp)

/// ggml.c:4418-4430 (MAX/MIN are ggml-impl.h:36-40's ternaries).
void ref_corr_dims(int n_dims, int n_ctx_orig, float base, float beta_fast, float beta_slow, float dims[2]) {
    auto corr_dim = [&](float n_rot) { return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float) M_PI)) / (2 * logf(base)); };
    const float start = floorf(corr_dim(beta_fast));
    const float end = ceilf(corr_dim(beta_slow));
    dims[0] = 0 > start ? 0 : start;
    dims[1] = (n_dims - 1) < end ? (float) (n_dims - 1) : end;
}

/// ops.cpp:5950-5973 (rope_yarn_ramp, rope_yarn).
void ref_rope_yarn(float theta_extrap, float freq_scale, const float corr[2], int64_t i0, float ext_factor, float mscale,
                   float* cos_theta, float* sin_theta) {
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float y = (i0 / 2 - corr[0]) / std::max(0.001f, corr[1] - corr[0]);
        const float ramp_mix = (1 - std::min(1.0f, std::max(0.0f, y))) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

/// RoPE NORMAL mode of the 64 dims [offs, offs + 64) of row v at position p.
/// cuda == false: ggml-cpu (ops.cpp:6082-6238): theta_k = theta_{k-1} * theta_scale, iteratively (rope_cache_init,
///   ops.cpp:5975-5989), rotate_pairs (ops.cpp:6063-6080).
/// cuda == true: ggml-cuda's rope_norm (rope.cu:107-119): theta_k = p * powf(theta_scale, k), powf evaluated on the host
///   as dsv4_rope does.
void ref_rope(float* v, int64_t p, const g::Dsv4Rope& P, int offs, bool inverse, bool cuda) {
    const float theta_scale = powf(P.freq_base, -2.0f / 64);
    float corr[2];
    ref_corr_dims(64, P.n_ctx_orig, P.freq_base, P.beta_fast, P.beta_slow, corr);
    float cache[64];
    float theta = (float) p;
    for (int i0 = 0; i0 < 64; i0 += 2) {
        const float te = cuda ? (float) p * powf(theta_scale, (float) i0 / 2.0f) : theta;
        ref_rope_yarn(te, P.freq_scale, corr, i0, P.ext_factor, P.attn_factor, &cache[i0], &cache[i0 + 1]);
        cache[i0 + 1] *= inverse ? -1.0f : 1.0f;
        theta *= theta_scale;
    }
    for (int i0 = 0; i0 < 64; i0 += 2) {
        const float c = cache[i0], s = cache[i0 + 1];
        const float x0 = v[offs + i0], x1 = v[offs + i0 + 1];
        v[offs + i0] = x0 * c - x1 * s;
        v[offs + i0 + 1] = x0 * s + x1 * c;
    }
}

/// ops.cpp:11990-12058: x * (1/sqrtf(n)) first, then the natural-order FWHT (the SIMD stages are u + v and fma(v, -1, u),
/// i.e. u - v exactly).
void ref_fwht(float* v, int n) {
    const float scale = 1.0f / sqrtf((float) n);
    for (int j = 0; j < n; ++j) v[j] *= scale;
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += 2 * len)
            for (int j = 0; j < len; ++j) {
                const float u = v[i + j], w = v[i + len + j];
                v[i + j] = u + w;
                v[i + len + j] = u - w;
            }
}

/// ggml_rms_norm then ggml_mul (ops.cpp:3924-3987): the squares in float summed in double, mean = (float)(sum / n),
/// scale = 1.0f / sqrtf(mean + eps), (x * scale) * w.
void ref_rms(float* x, const float* w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; ++i) ss += (double) (x[i] * x[i]);
    const float mean = (float) (ss / n);
    const float sc = 1.0f / sqrtf(mean + eps);
    for (int i = 0; i < n; ++i) x[i] = (x[i] * sc) * (w ? w[i] : 1.0f);
}

/// soft_max over n values (ops.cpp:5584-5683 with vec.cpp:531-596's scalar tail: ggml-cpu's AVX-512 ggml_v_expf on
/// 16-blocks is within ~1.5 ulp of expf): max, e = expf(x - max), the sum in double, then e * (float)(1.0 / sum).
void ref_softmax(float* x, int n) {
    float mx = -INFINITY;
    for (int i = 0; i < n; ++i) mx = mx > x[i] ? mx : x[i];
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += (double) (x[i] = expf(x[i] - mx));
    const float inv = (float) (1.0 / sum);
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

// ================================================================== 1. RoPE

void test_rope() {
    struct Case {
        const char* name;
        int T, rows, row_len, offs;
        bool pc;
        int64_t pos0;
        bool in_place, inverse;
    };
    const Case cases[] = {
        {"q 64x512 P0", 3, 64, 512, 448, false, 5, true, false},
        {"q 64x512 PC", 8, 64, 512, 448, true, 0, true, false},
        {"q 64x512 PC pos 1000", 4, 64, 512, 448, true, 1000, true, false},
        {"q 64x512 PC pos 65000", 2, 64, 512, 448, true, 65000, true, false},
        {"kv 1x512 PC out of place", 7, 1, 512, 448, true, 126, false, false},
        {"kv 1x512 P0 out of place pos 3000", 5, 1, 512, 448, false, 3000, false, false},
        {"indexer q 64x128 PC", 8, 64, 128, 64, true, 2600, true, false},
        {"de-rope 64x512 PC (inverse)", 5, 64, 512, 448, true, 777, true, true},
        {"de-rope 64x512 P0 (inverse) out of place", 1, 64, 512, 448, false, 9, false, true},
    };
    for (const Case& c : cases) {
        const g::Dsv4Rope P = c.pc ? g::dsv4_rope_pc() : g::dsv4_rope_p0();
        const size_t n = (size_t) c.T * c.rows * c.row_len;
        auto x = randn(n);
        std::vector<float> rc(x), rg(x);
        for (int t = 0; t < c.T; ++t)
            for (int r = 0; r < c.rows; ++r) {
                const size_t o = ((size_t) t * c.rows + r) * c.row_len;
                ref_rope(&rc[o], c.pos0 + t, P, c.offs, c.inverse, false);
                ref_rope(&rg[o], c.pos0 + t, P, c.offs, c.inverse, true);
            }
        Dev<float> dx(x), dy(n);
        float* out = c.in_place ? dx.p : dy.p;
        g::dsv4_rope(dx.p, out, c.T, c.rows, c.row_len, c.offs, P, c.inverse, c.pos0, nullptr);
        auto got = c.in_place ? dx.down() : dy.down();
        char name[128];
        // vs the ggml-cuda formula (theta = p * powf(ts, k), the same powf values): sincosf vs glibc ulps, and the device's
        // FMA in the YaRN blend (pairs 15-25, theta ~ 0.004 p): <= 1 ulp of theta, ~2e-10 p rad
        snprintf(name, sizeof name, "rope %s vs ggml-cuda formula", c.name);
        report(name, rel(got, rg), 2e-6 + 1e-10 * (double) (c.pos0 + c.T));
        // vs ggml-cpu (theta multiplied by ts once per pair): the iterative product rounds k times where the CUDA form
        // rounds twice, so theta_k differs by a few ulps of theta_k <= p, i.e. up to ~1e-7 * p rad on the high-frequency
        // pairs (the low ones and the rest of the row agree): 2e-6 + 2e-8 * p bounds it with margin
        const double tol_cpu = 2e-6 + 2e-8 * (double) (c.pos0 + c.T);
        snprintf(name, sizeof name, "rope %s vs ggml-cpu", c.name);
        report(name, rel(got, rc), tol_cpu);
        // the dims outside [offs, offs + 64) are copied bit for bit
        bool copy_ok = true;
        for (size_t row = 0; row < (size_t) c.T * c.rows && copy_ok; ++row)
            for (int i = 0; i < c.row_len; ++i)
                if ((i < c.offs || i >= c.offs + 64) && memcmp(&got[row * c.row_len + i], &x[row * c.row_len + i], 4) != 0) {
                    copy_ok = false;
                    break;
                }
        snprintf(name, sizeof name, "rope %s: other dims untouched", c.name);
        report_bool(name, copy_ok);
        if (c.inverse) {   // the negative control: the forward rotation compared with the de-rope
            std::vector<float> wrong(x);
            for (int t = 0; t < c.T; ++t)
                for (int r = 0; r < c.rows; ++r) ref_rope(&wrong[((size_t) t * c.rows + r) * c.row_len], c.pos0 + t, P, c.offs, false, true);
            snprintf(name, sizeof name, "rope %s vs the FORWARD rotation", c.name);
            report_neg(name, rel(got, wrong), 2e-6);
        }
    }
    {   // de-rope(rope(x)) = x up to rounding; the device position equals the argument bitwise
        const g::Dsv4Rope P = g::dsv4_rope_pc();
        const int T = 6, rows = 64, L = 512;
        auto x = randn((size_t) T * rows * L);
        Dev<float> d(x), d2(x.size());
        g::dsv4_rope(d.p, d.p, T, rows, L, 448, P, false, 4093, nullptr);
        g::dsv4_rope(d.p, d.p, T, rows, L, 448, P, true, 4093, nullptr);
        report("rope then de-rope = identity (PC, pos 4093)", rel(d.down(), x), 1e-6);
        Dev<int64_t> dpos(std::vector<int64_t>{4093});
        Dev<float> a(x), b(x);
        g::dsv4_rope(a.p, a.p, T, rows, L, 448, P, false, 4093, nullptr);
        g::dsv4_rope(b.p, b.p, T, rows, L, 448, P, false, -1, nullptr, dpos.p);
        auto va = a.down(), vb = b.down();
        report_bool("rope: device position == argument (bitwise)", bitwise(va.data(), vb.data(), va.size() * 4));
    }
}

/// The parameter sets of DSV4_SPEC 3.0: PC's attn_factor (deepseek4.cpp:11-17) is 0x3f486de1 and its effective magnitude
/// attn_factor * (1 + 0.1 logf(16)) is exactly 1.0f; the YaRN correction dims are 15 and 25; theta_scale 0.68765604 (PC)
/// and 0.74989426 (P0).
void test_rope_params() {
    const g::Dsv4Rope pc = g::dsv4_rope_pc(), p0 = g::dsv4_rope_p0();
    uint32_t af_bits;
    memcpy(&af_bits, &pc.attn_factor, 4);
    const float msc = pc.attn_factor * (1.0f + 0.1f * logf(1.0f / pc.freq_scale));
    float cd[2];
    ref_corr_dims(64, pc.n_ctx_orig, pc.freq_base, pc.beta_fast, pc.beta_slow, cd);
    const float ts_pc = powf(pc.freq_base, -2.0f / 64), ts_p0 = powf(p0.freq_base, -2.0f / 64);
    char d[160];
    snprintf(d, sizeof d, "(attn_factor 0x%08x, magnitude %.9g, corr dims %g %g, theta_scale %.8g / %.8g)", af_bits, msc, cd[0], cd[1],
             ts_pc, ts_p0);
    report_bool("rope parameter sets P0 / PC (DSV4_SPEC 3.0)",
                af_bits == 0x3f486de1u && msc == 1.0f && cd[0] == 15.0f && cd[1] == 25.0f && p0.ext_factor == 0.0f &&
                    p0.attn_factor == 1.0f && pc.freq_base == 160000.0f && pc.freq_scale == 0.0625f && pc.n_ctx_orig == 65536,
                d);
}

/// dsv4_raw_store: row (pos0 + t) % ring_rows = half(kv[t]) (RNE), every other row untouched; the device position the same.
void test_raw_store() {
    const int RS = 256, T = 8, D = 512;
    for (const int64_t pos0 : {(int64_t) 0, (int64_t) 251, (int64_t) 100003}) {
        const auto kv = randn((size_t) T * D, 1.5f);
        std::vector<uint16_t> want((size_t) RS * D, kSentinel);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < D; ++i) want[(size_t) ((pos0 + t) % RS) * D + i] = f2h(kv[(size_t) t * D + i]);
        Dev<float> dkv(kv);
        Dev<uint16_t> a(std::vector<uint16_t>((size_t) RS * D, kSentinel)), b(std::vector<uint16_t>((size_t) RS * D, kSentinel));
        Dev<int64_t> dpos(std::vector<int64_t>{pos0});
        g::dsv4_raw_store(dkv.p, a.p, RS, pos0, T, nullptr);
        g::dsv4_raw_store(dkv.p, b.p, RS, -5, T, nullptr, dpos.p);
        const auto ga = a.down(), gb = b.down();
        char name[96];
        snprintf(name, sizeof name, "raw_store T=8 at pos0 %lld (F16 RNE, ring %d), arg and device pos", (long long) pos0, RS);
        report_bool(name, ga == want && gb == want);
    }
}

// ================================================================== 2. Hadamard

void test_hadamard() {
    for (int n : {128, 512}) {
        const int rows = 64 * 5 + 3;
        auto x = randn((size_t) rows * n, 1.5f);
        std::vector<float> ref(x), wrong(x);
        for (int r = 0; r < rows; ++r) ref_fwht(&ref[(size_t) r * n], n);
        for (int r = 0; r < rows; ++r) {   // without the 1/sqrt(n): the negative control
            ref_fwht(&wrong[(size_t) r * n], n);
            for (int j = 0; j < n; ++j) wrong[(size_t) r * n + j] *= sqrtf((float) n);
        }
        Dev<float> dx(x), dy(x.size());
        g::dsv4_hadamard(dx.p, dy.p, rows, n, nullptr);
        auto got = dy.down();
        char name[96];
        snprintf(name, sizeof name, "hadamard %d x %d (ggml fwht order, bitwise)", rows, n);
        report_bool(name, bitwise(got.data(), ref.data(), got.size() * 4), "rel " + std::to_string(rel(got, ref)));
        g::dsv4_hadamard(dx.p, dx.p, rows, n, nullptr);   // in place
        auto inplace = dx.down();
        snprintf(name, sizeof name, "hadamard %d x %d in place (bitwise)", rows, n);
        report_bool(name, bitwise(inplace.data(), ref.data(), got.size() * 4));
        snprintf(name, sizeof name, "hadamard %d vs a reference without 1/sqrt(n)", n);
        report_neg(name, rel(got, wrong), 1e-6);
    }
}

// ================================================================== 3. KV compressors

/// The whole-sequence reference: every complete row b < n_blocks of compressor c from the full projections (N, coff*d),
/// as llama.cpp's graph computes it (deepseek4.cpp:471-589 on ggml-cpu): the slot gather with ape added
/// (deepseek4.cpp:955, 970, 1039) and the zero / -inf row for positions < 0 (deepseek4.cpp:209-215), soft_max over the
/// slots per channel, mul + sum_rows (double, ops.cpp:1460-1491), rms_norm * w, rope at s0 (PC, offs d - 64), LID: fwht,
/// set_rows F32 -> F16 RNE.  rope_cuda picks the theta formula; `wrong`: 1 = ape left out, 2 = prev/cur halves swapped.
std::vector<uint16_t> ref_compress(const g::Dsv4Compressor& c, const std::vector<float>& akv, const std::vector<float>& gs,
                                   const std::vector<float>& ape, const std::vector<float>& nw, float eps, const g::Dsv4Rope& P,
                                   int n_blocks, bool rope_cuda, int wrong = 0) {
    const int R = c.ratio, D = c.d, W = c.coff * D, NS = c.coff == 2 ? 2 * R : R;
    std::vector<uint16_t> out((size_t) n_blocks * D);
    std::vector<float> row(D), v(NS), gg(NS);
    for (int b = 0; b < n_blocks; ++b) {
        const int64_t s0 = (int64_t) b * R;
        for (int k = 0; k < D; ++k) {
            for (int j = 0; j < NS; ++j) {
                int64_t q;
                int f;
                if (c.coff == 2) {
                    q = j < R ? s0 - R + j : s0 + j - R;
                    const bool first = (j < R) != (wrong == 2);
                    f = first ? k : D + k;
                } else {
                    q = s0 + j;
                    f = k;
                }
                if (q < 0) {
                    v[j] = 0.0f;
                    gg[j] = -INFINITY;
                } else {
                    v[j] = akv[(size_t) q * W + f];
                    gg[j] = wrong == 1 ? gs[(size_t) q * W + f] : gs[(size_t) q * W + f] + ape[(size_t) (q % R) * W + f];
                }
            }
            ref_softmax(gg.data(), NS);
            double acc = 0;
            for (int j = 0; j < NS; ++j) acc += (double) (v[j] * gg[j]);
            row[k] = (float) acc;
        }
        ref_rms(row.data(), nw.data(), D, eps);
        ref_rope(row.data(), s0, P, D - 64, false, rope_cuda);
        if (c.hadamard) ref_fwht(row.data(), D);
        for (int k = 0; k < D; ++k) out[(size_t) b * D + k] = f2h(row[k]);
    }
    return out;
}

struct CompState {
    std::vector<uint16_t> cache;
    std::vector<float> ring_kv, ring_sc;
};

/// The GPU: the sequence fed through windows of the given sizes from position 0 (pool, then persist, per window), the
/// rings starting at zero (llama.cpp zero-fills the state, KVD:964-966) and the cache filled with a sentinel.
CompState gpu_compress(const g::Dsv4Compressor& c, const std::vector<float>& akv, const std::vector<float>& gs,
                       const std::vector<float>& ape, const std::vector<float>& nw, float eps, const g::Dsv4Rope& P,
                       const std::vector<int>& windows, int cache_rows, bool dev_pos) {
    const int W = c.coff * c.d;
    const int tmax = *std::max_element(windows.begin(), windows.end());
    Dev<float> dakv((size_t) tmax * W), dgs((size_t) tmax * W), dape(ape), dnw(nw), rkv((size_t) c.ring * W),
        rsc((size_t) c.ring * W);
    CK(cudaMemset(rkv.p, 0, rkv.n * sizeof(float)));
    CK(cudaMemset(rsc.p, 0, rsc.n * sizeof(float)));
    Dev<uint16_t> cache(std::vector<uint16_t>((size_t) cache_rows * c.d, kSentinel));
    Dev<int64_t> dpos(1);
    int64_t pos0 = 0;
    for (int T : windows) {
        CK(cudaMemcpy(dakv.p, akv.data() + (size_t) pos0 * W, (size_t) T * W * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dgs.p, gs.data() + (size_t) pos0 * W, (size_t) T * W * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpos.p, &pos0, sizeof pos0, cudaMemcpyHostToDevice));
        const int64_t* pd = dev_pos ? dpos.p : nullptr;
        const int64_t pa = dev_pos ? -12345 : pos0;   // the argument must be ignored on the device-position path
        g::dsv4_compress_pool(c, dakv.p, dgs.p, dape.p, rkv.p, rsc.p, dnw.p, eps, P, cache.p, pa, T, nullptr, pd);
        g::dsv4_compress_persist(c, dakv.p, dgs.p, dape.p, rkv.p, rsc.p, pa, T, nullptr, pd);
        pos0 += T;
    }
    return {cache.down(), rkv.down(), rsc.down()};
}

/// Windows of 1..8 tokens in a scrambled cycle, the sizes summing to exactly n.
std::vector<int> small_windows(int n, int phase) {
    static const int cyc[] = {1, 3, 8, 2, 5, 7, 4, 6, 8, 1, 2, 8, 3, 6, 5, 7};
    std::vector<int> w;
    int s = 0;
    for (int i = phase; s < n; ++i) {
        const int t = std::min(cyc[i % 16], n - s);
        w.push_back(t);
        s += t;
    }
    return w;
}

void test_compressors() {
    const float eps = 9.99999968e-21f;   // f32(1e-20), DSV4_SPEC 1.2
    const g::Dsv4Rope P = g::dsv4_rope_pc();
    struct Kind {
        const char* name;
        g::Dsv4Compressor c;
    };
    const Kind kinds[] = {{"CSA", g::dsv4_csa()}, {"LID", g::dsv4_lid()}, {"HCA", g::dsv4_hca()}};
    for (const Kind& kd : kinds) {
        const g::Dsv4Compressor& c = kd.c;
        const int W = c.coff * c.d;
        const int N = c.ratio == 128 ? 2200 : 1200;   // positions of the sequence
        const int nb = N / c.ratio;                    // its complete rows
        const int cache_rows = nb + 4;
        auto akv = randn((size_t) N * W, 1.0f), gs = randn((size_t) N * W, 2.0f);
        auto ape = randn((size_t) c.ratio * W, 0.5f), nw = randn(c.d, 0.2f, 1.0f);
        const auto ref_cpu = ref_compress(c, akv, gs, ape, nw, eps, P, nb, false);
        const auto ref_cuda = ref_compress(c, akv, gs, ape, nw, eps, P, nb, true);
        char name[160];
        auto check_rows = [&](const char* what, const CompState& s, int rows_done) {
            const size_t n = (size_t) rows_done * c.d;
            const F16Cmp a = cmp_f16(s.cache.data(), ref_cpu.data(), n, c.d);
            const F16Cmp b = cmp_f16(s.cache.data(), ref_cuda.data(), n, c.d);
            // vs the ggml-cuda rope: the GPU's F32 row differs from the reference's by ulps (float vs double sums, rsqrtf
            // vs 1/sqrtf, sincosf vs glibc), so each half differs by at most one F16 rounding step plus ~1e-6 of the
            // row's scale (tau 1e-5 leaves margin): bound <= 1 is the criterion.  An F32 difference d (relative) flips
            // ~d / 2^-11 of the roundings, each one step (2^-11..2^-10 of the value): rel L2 ~ sqrt(d 2^11) 2^-11, 3e-5
            // at d = 1e-6; 1e-4 bounds it
            snprintf(name, sizeof name, "%s %s: %d rows vs ggml-cuda rope (%zu/%zu differ, max %d ulp)", kd.name, what, rows_done,
                     b.n_diff, n, b.max_ulp);
            report(name, b.bound <= 1.0 ? b.rel : 1.0, 1e-4);
            if (b.bound > 1.0)
                printf("   worst %zu: gpu %.8g ref %.8g bound %.3g\n", b.worst, h2f(s.cache[b.worst]), h2f(ref_cuda[b.worst]), b.bound);
            // vs ggml-cpu: its rope's theta (one product per pair) differs from the CUDA form by a few ulps of theta, ~1e-4
            // rad at s0 ~ 2000 on the high-frequency pairs: rel L2 ~ 1e-4 * sqrt(2 * 3 / d) plus the F16 steps
            snprintf(name, sizeof name, "%s %s: %d rows vs ggml-cpu (%zu/%zu differ, max %d ulp)", kd.name, what, rows_done,
                     a.n_diff, n, a.max_ulp);
            report(name, a.rel, 2e-4);
            bool untouched = true;   // rows past the complete ones: never written
            for (size_t i = n; i < s.cache.size(); ++i) untouched &= s.cache[i] == kSentinel;
            snprintf(name, sizeof name, "%s %s: rows past the complete ones untouched", kd.name, what);
            report_bool(name, untouched);
        };
        auto check_ring = [&](const char* what, const CompState& s, int n_pos) {
            // the ring after the windows: row p % S holds (akv_p, gs_p + ape[p % r]) of the last S positions, bitwise
            bool ok = true;
            for (int p = std::max(0, n_pos - c.ring); p < n_pos && ok; ++p) {
                const size_t row = (size_t) (p % c.ring) * W;
                for (int f = 0; f < W; ++f) {
                    const float want_sc = gs[(size_t) p * W + f] + ape[(size_t) (p % c.ratio) * W + f];
                    if (memcmp(&s.ring_kv[row + f], &akv[(size_t) p * W + f], 4) || memcmp(&s.ring_sc[row + f], &want_sc, 4)) {
                        ok = false;
                        break;
                    }
                }
            }
            snprintf(name, sizeof name, "%s %s: ring = the last %d positions (bitwise)", kd.name, what, c.ring);
            report_bool(name, ok);
        };
        // (a) the whole sequence in one window (a prompt), and windows of 1..8 tokens: bitwise the same rows
        const CompState whole = gpu_compress(c, akv, gs, ape, nw, eps, P, {N}, cache_rows, false);
        check_rows("one window of N", whole, nb);
        const CompState small = gpu_compress(c, akv, gs, ape, nw, eps, P, small_windows(N, 0), cache_rows, false);
        snprintf(name, sizeof name, "%s windows of 1..8 = one window (bitwise cache)", kd.name);
        report_bool(name, bitwise(small.cache.data(), whole.cache.data(), whole.cache.size() * 2));
        check_ring("windows of 1..8", small, N);
        // (b) T = 300 at 0 then small windows; small windows to 517, T = 300 at 517, small windows
        std::vector<int> wa{300};
        for (int t : small_windows(N - 300, 3)) wa.push_back(t);
        const CompState sa = gpu_compress(c, akv, gs, ape, nw, eps, P, wa, cache_rows, false);
        snprintf(name, sizeof name, "%s T=300 at 0 then 1..8 = one window (bitwise)", kd.name);
        report_bool(name, bitwise(sa.cache.data(), whole.cache.data(), whole.cache.size() * 2));
        std::vector<int> wb = small_windows(517, 5);
        wb.push_back(300);
        for (int t : small_windows(N - 817, 9)) wb.push_back(t);
        const CompState sb = gpu_compress(c, akv, gs, ape, nw, eps, P, wb, cache_rows, false);
        snprintf(name, sizeof name, "%s 1..8 to 517, T=300 at 517, 1..8 = one window (bitwise)", kd.name);
        report_bool(name, bitwise(sb.cache.data(), whole.cache.data(), whole.cache.size() * 2));
        check_ring("T=300 at 517", sb, N);
        // (c) the device position: the same windows, bitwise the same cache and ring
        const CompState sd = gpu_compress(c, akv, gs, ape, nw, eps, P, wb, cache_rows, true);
        snprintf(name, sizeof name, "%s device position == argument (bitwise cache + ring)", kd.name);
        report_bool(name, bitwise(sd.cache.data(), sb.cache.data(), sb.cache.size() * 2) &&
                              bitwise(sd.ring_kv.data(), sb.ring_kv.data(), sb.ring_kv.size() * 4) &&
                              bitwise(sd.ring_sc.data(), sb.ring_sc.data(), sb.ring_sc.size() * 4));
        // (d) every window size 1..8 at every start 0..9 (single tokens before it, then two windows after it)
        bool grid_ok = true;
        double worst_bound = 0;
        for (int pos0 = 0; pos0 <= 9 && grid_ok; ++pos0)
            for (int T = 1; T <= 8; ++T) {
                std::vector<int> w(pos0, 1);
                w.push_back(T);
                w.push_back(5);
                w.push_back(3);
                const int n_pos = pos0 + T + 8;
                const CompState s = gpu_compress(c, akv, gs, ape, nw, eps, P, w, cache_rows, false);
                const int rows_done = n_pos / c.ratio;
                const F16Cmp a = cmp_f16(s.cache.data(), ref_cuda.data(), (size_t) rows_done * c.d, c.d);
                worst_bound = std::max(worst_bound, a.bound);
                bool untouched = true;
                for (size_t i = (size_t) rows_done * c.d; i < s.cache.size(); ++i) untouched &= s.cache[i] == kSentinel;
                if (a.bound > 1.0 || !untouched) {
                    grid_ok = false;
                    printf("   %s: pos0 %d T %d: bound %.3g, untouched %d\n", kd.name, pos0, T, a.bound, (int) untouched);
                    break;
                }
            }
        snprintf(name, sizeof name, "%s T=1..8 at pos0=0..9 (+2 windows): rows in bound, rest untouched", kd.name);
        report_bool(name, grid_ok, "worst bound " + std::to_string(worst_bound));
        // (e) negative controls: without ape, and (overlap compressors) with the halves swapped
        const auto no_ape = ref_compress(c, akv, gs, ape, nw, eps, P, nb, true, 1);
        const F16Cmp n1 = cmp_f16(whole.cache.data(), no_ape.data(), (size_t) nb * c.d, c.d);
        snprintf(name, sizeof name, "%s vs a reference without ape", kd.name);
        report_neg(name, n1.bound <= 1.0 ? n1.rel : 1.0, 1e-4);
        if (c.coff == 2) {
            const auto swapped = ref_compress(c, akv, gs, ape, nw, eps, P, nb, true, 2);
            const F16Cmp n2 = cmp_f16(whole.cache.data(), swapped.data(), (size_t) nb * c.d, c.d);
            snprintf(name, sizeof name, "%s vs a reference with prev/cur halves swapped", kd.name);
            report_neg(name, n2.bound <= 1.0 ? n2.rel : 1.0, 1e-4);
        }
    }
}

// ================================================================== 4. lightning indexer

/// ops.cpp:12084-12164 (ggml-cpu, F32 q): per row, per head the dot (ggml_vec_dot_f32; here in double, rounded), then
/// score += MAX(qk, 0) * w[h] sequentially.  With q16 (q already rounded to F16) it is ggml-cuda's tensor-core path
/// (lightning-indexer.cu:61-197): each head's relu(dot) * w rounded, the heads summed in order - the same float ops.
/// no_relu: the negative control.  q: (64, 128) as given, kf: the row as floats.
float ref_lid_score(const float* q, const float* kf, const float* w, bool no_relu = false) {
    float score = 0.0f;
    for (int h = 0; h < 64; ++h) {
        double dot = 0;
        for (int d = 0; d < 128; ++d) dot += (double) q[h * 128 + d] * kf[d];
        const float qk = (float) dot;
        const float r = no_relu ? qk : std::max(qk, 0.0f);
        score = score + r * w[h];
    }
    return score;
}

/// The selection the GPU must make from its own scores: the top k by score, ties to the lower row (a stable sort).
std::set<int> ref_topk(const float* s, int n, int k) {
    std::set<int> out;
    if (n <= k) {
        for (int i = 0; i < n; ++i) out.insert(i);
        return out;
    }
    std::vector<int> o(n);
    std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return s[a] > s[b]; });
    for (int i = 0; i < k; ++i) out.insert(o[i]);
    return out;
}

void lid_case(const char* what, int64_t pos0, int T, int max_rows, int stride_check, bool timing) {
    const int top_k = 512;
    auto iq = randn((size_t) T * 64 * 128, 1.0f), w = randn((size_t) T * 64, 0.05f);
    const auto kh = to_f16(randn((size_t) max_rows * 128, 1.0f));
    Dev<float> diq(iq), dw(w), score((size_t) T * max_rows);
    Dev<uint16_t> dk(kh);
    Dev<int32_t> sel((size_t) T * top_k), cnt(T);
    CK(cudaMemset(score.p, 0, score.n * sizeof(float)));
    g::dsv4_lid_scores(diq.p, dk.p, dw.p, score.p, max_rows, pos0, T, nullptr);
    g::dsv4_lid_select(score.p, max_rows, top_k, sel.p, cnt.p, pos0, T, nullptr);
    auto sg = score.down();
    auto sv = sel.down();
    auto cv = cnt.down();
    // the tensor-core path: the same products, F32 accumulation in the hardware's order
    Dev<float> score_mma((size_t) T * max_rows);
    CK(cudaMemset(score_mma.p, 0, score_mma.n * sizeof(float)));
    const bool have_mma = g::dsv4_lid_scores_mma(diq.p, dk.p, dw.p, score_mma.p, max_rows, pos0, T, nullptr);
    const auto sm = score_mma.down();
    float ms1 = 0, ms2 = 0, ms3 = 0;
    if (timing) {
        ms1 = time_ms([&] { g::dsv4_lid_scores(diq.p, dk.p, dw.p, score.p, max_rows, pos0, T, nullptr); });
        ms2 = time_ms([&] { g::dsv4_lid_select(score.p, max_rows, top_k, sel.p, cnt.p, pos0, T, nullptr); });
        if (have_mma) ms3 = time_ms([&] { g::dsv4_lid_scores_mma(diq.p, dk.p, dw.p, score_mma.p, max_rows, pos0, T, nullptr); });
    }
    // scores: every `stride_check`-th visible row against both references
    std::vector<float> got, got_mma, rq16, rcpu, rbad;
    const auto iq16 = from_f16(to_f16(iq));   // q as ggml-cuda's tensor-core indexer sees it
    bool sel_ok = true, order_ok = true;
    std::string detail;
    float kf[128];
    for (int t = 0; t < T; ++t) {
        const int n_vis = (int) std::min<int64_t>((pos0 + t + 1) / 4, max_rows);
        const float* qt = &iq[(size_t) t * 64 * 128];
        const float* qt16 = &iq16[(size_t) t * 64 * 128];
        const float* wt = &w[(size_t) t * 64];
        for (int j = 0; j < n_vis; j += stride_check) {
            for (int d = 0; d < 128; ++d) kf[d] = h2f(kh[(size_t) j * 128 + d]);
            got.push_back(sg[(size_t) t * max_rows + j]);
            got_mma.push_back(sm[(size_t) t * max_rows + j]);
            rq16.push_back(ref_lid_score(qt16, kf, wt));
            rcpu.push_back(ref_lid_score(qt, kf, wt));
            rbad.push_back(ref_lid_score(qt16, kf, wt, true));
        }
        // the selection: as a set the top 512 of the GPU's own scores (all rows when n_vis <= 512), ascending, -1 padded
        const std::set<int> want = ref_topk(&sg[(size_t) t * max_rows], n_vis, top_k);
        const int32_t* row = &sv[(size_t) t * top_k];
        const int n_sel = std::min(n_vis, top_k);
        std::set<int> have(row, row + n_sel);
        for (int i = 1; i < n_sel; ++i) order_ok &= row[i - 1] < row[i];
        for (int i = n_sel; i < top_k; ++i) order_ok &= row[i] == -1;
        if (have != want || cv[t] != n_sel) {
            sel_ok = false;
            detail = "token " + std::to_string(t) + ": count " + std::to_string(cv[t]) + " want " + std::to_string(n_sel);
        }
    }
    char name[160];
    if (got.empty()) {
        snprintf(name, sizeof name, "lid %s: no visible rows, selection empty", what);
        report_bool(name, sel_ok && order_ok, detail);
        return;
    }
    snprintf(name, sizeof name, "lid scores %s vs ggml-cuda formula (q16)", what);
    report(name, rel(got, rq16), 1e-5);
    // vs ggml-cpu's F32 q: the kernel rounds q to F16 as ggml-cuda does; each q element moves by <= 2^-12 relative (RNE),
    // a dot by ~2^-12/sqrt(3) relative for random signs (1.4e-4) and by <= 2^-12 sum|q k| at worst
    snprintf(name, sizeof name, "lid scores %s vs ggml-cpu (F32 q)", what);
    report(name, rel(got, rcpu), 1e-3);
    snprintf(name, sizeof name, "lid scores %s vs a reference without the ReLU", what);
    report_neg(name, rel(got, rbad), 1e-3);
    if (have_mma) {
        snprintf(name, sizeof name, "lid scores %s tensor cores vs ggml-cuda formula (q16)", what);
        report(name, rel(got_mma, rq16), 1e-5);
        snprintf(name, sizeof name, "lid scores %s tensor cores vs the SIMT kernel", what);
        report(name, rel(got_mma, got), 1e-5);
        snprintf(name, sizeof name, "lid scores %s tensor cores vs a reference without the ReLU", what);
        report_neg(name, rel(got_mma, rbad), 1e-3);
    }
    snprintf(name, sizeof name, "lid select %s = top-512 of the scores (sets, count)", what);
    report_bool(name, sel_ok, detail);
    snprintf(name, sizeof name, "lid select %s: ascending, -1 padded", what);
    report_bool(name, order_ok);
    if (timing)
        printf("   lid %s: scores %.3f ms (tensor cores %.3f ms), select %.3f ms (mean of 20 after a warm-up)\n", what, ms1, ms3, ms2);
}

/// The selection alone on crafted score rows (one token, n visible rows): random, all equal, heavy ties at the
/// boundary, few distinct values, sorted - the last ones drive the sampled threshold out of [512, 4096] and the kernel
/// to its full radix select.  Exact: the set of the top 512 with ties to the lower row, ascending, -1 padded.
void test_select() {
    const int top_k = 512;
    struct Case {
        const char* name;
        int n;
        int kind;
    };
    const Case cases[] = {{"random n=513", 513, 0},       {"random n=1000", 1000, 0},     {"random n=1024", 1024, 0},
                          {"random n=1025", 1025, 0},     {"random n=4096", 4096, 0},     {"random n=20000", 20000, 0},
                          {"negative n=30000", 30000, 1}, {"all equal n=20000", 20000, 2}, {"ties at the boundary n=10000", 10000, 3},
                          {"4 distinct values n=30000", 30000, 4}, {"ascending n=30000", 30000, 5}, {"descending n=65536", 65536, 6},
                          // moex (DSV41_BACKLOG #107): the decode's selection at long context - 131K-400K visible rows a token
                          {"random n=131072", 131072, 0}, {"random n=262144", 262144, 0}, {"random n=400000", 400000, 0}};
    for (const Case& c : cases) {
        std::vector<float> sc(c.n);
        std::uniform_int_distribution<int> u4(0, 3);
        std::uniform_real_distribution<float> u01(0.0f, 0.5f);
        switch (c.kind) {
            case 0: sc = randn(c.n, 0.3f); break;
            case 1: sc = randn(c.n, 2.0f, -5.0f); break;
            case 2: std::fill(sc.begin(), sc.end(), 0.25f); break;
            case 3: {   // 300 rows at 5, 5000 at 1, the rest below 0.5: the top 512 = the 300 and the first 212 of the 1s
                for (auto& v : sc) v = u01(rng);
                std::vector<int> idx(c.n);
                std::iota(idx.begin(), idx.end(), 0);
                std::shuffle(idx.begin(), idx.end(), rng);
                for (int i = 0; i < 300; ++i) sc[idx[i]] = 5.0f;
                for (int i = 300; i < 5300; ++i) sc[idx[i]] = 1.0f;
                break;
            }
            case 4: for (auto& v : sc) v = (float) u4(rng); break;
            case 5: for (int i = 0; i < c.n; ++i) sc[i] = (float) i * 1e-3f; break;
            case 6: for (int i = 0; i < c.n; ++i) sc[i] = (float) (c.n - i) * 1e-3f; break;
        }
        const int64_t pos0 = (int64_t) 4 * c.n - 1;   // n_vis = n for the one token
        Dev<float> ds(sc);
        Dev<int32_t> sel(top_k), cnt(1);
        g::dsv4_lid_select(ds.p, c.n, top_k, sel.p, cnt.p, pos0, 1, nullptr);
        auto sv = sel.down();
        const std::set<int> want = ref_topk(sc.data(), c.n, top_k);
        const int n_sel = std::min(c.n, top_k);
        bool ok = cnt.down()[0] == n_sel && std::set<int>(sv.begin(), sv.begin() + n_sel) == want;
        for (int i = 1; i < n_sel; ++i) ok &= sv[i - 1] < sv[i];
        char name[128];
        snprintf(name, sizeof name, "lid select %s = top-512, ties to the lower row", c.name);
        report_bool(name, ok);
        if (c.n >= 65536) {
            const float ms = time_ms([&] { g::dsv4_lid_select(ds.p, c.n, top_k, sel.p, nullptr, pos0, 1, nullptr); });
            printf("   lid select %s: %.3f ms\n", c.name, ms);
        }
    }
    {   // the negative control: a reference that keeps the LOWEST 512 must not match
        const int n = 5000;
        const auto sc = randn(n);
        Dev<float> ds(sc);
        Dev<int32_t> sel(top_k);
        g::dsv4_lid_select(ds.p, n, top_k, sel.p, nullptr, (int64_t) 4 * n - 1, 1, nullptr);
        auto sv = sel.down();
        std::vector<float> neg(sc);
        for (auto& v : neg) v = -v;
        report_bool("lid select vs the bottom-512 (negative control: must differ)",
                    std::set<int>(sv.begin(), sv.end()) != ref_topk(neg.data(), n, top_k));
    }
}

void test_indexer() {
    for (int64_t pos0 : {0, 1, 2, 3, 5, 9}) {   // the first blocks: n_vis 0, 1, 2 inside the windows
        char what[64];
        snprintf(what, sizeof what, "T=8 pos0=%lld", (long long) pos0);
        lid_case(what, pos0, 8, 64, 1, false);
    }
    lid_case("T=5 pos0=2600 (n_vis 650-651)", 2600, 5, 1024, 1, false);
    lid_case("T=3 pos0=2045 (crossing 512 visible rows)", 2045, 3, 1024, 1, false);
    lid_case("T=300 pos0=517 (prompt)", 517, 300, 256, 7, false);
    lid_case("T=8 pos0=262136 (65536 rows, 256K context)", 262136, 8, 65536, 61, true);
    lid_case("T=1 pos0=262143 (65536 rows)", 262143, 1, 65536, 97, true);
    {   // the device position, bitwise; and the static grid of the graph path with a larger capacity
        const int max_rows = 4096, T = 8, top_k = 512;
        const int64_t pos0 = 9000;
        auto iq = randn((size_t) T * 64 * 128), w = randn((size_t) T * 64, 0.05f);
        Dev<float> diq(iq), dw(w), s1((size_t) T * max_rows), s2((size_t) T * max_rows);
        Dev<uint16_t> dk(to_f16(randn((size_t) max_rows * 128)));
        Dev<int32_t> sel1((size_t) T * top_k), sel2((size_t) T * top_k), c1(T), c2(T);
        Dev<int64_t> dpos(std::vector<int64_t>{pos0});
        CK(cudaMemset(s1.p, 0, s1.n * 4));
        CK(cudaMemset(s2.p, 0, s2.n * 4));
        g::dsv4_lid_scores(diq.p, dk.p, dw.p, s1.p, max_rows, pos0, T, nullptr);
        g::dsv4_lid_select(s1.p, max_rows, top_k, sel1.p, c1.p, pos0, T, nullptr);
        g::dsv4_lid_scores(diq.p, dk.p, dw.p, s2.p, max_rows, 0, T, nullptr, dpos.p);
        g::dsv4_lid_select(s2.p, max_rows, top_k, sel2.p, c2.p, 0, T, nullptr, dpos.p);
        auto a = s1.down(), b = s2.down();
        auto x = sel1.down(), y = sel2.down();
        report_bool("lid scores + select: device position == argument (bitwise)",
                    bitwise(a.data(), b.data(), a.size() * 4) && bitwise(x.data(), y.data(), x.size() * 4));
        CK(cudaMemset(s1.p, 0, s1.n * 4));
        CK(cudaMemset(s2.p, 0, s2.n * 4));
        if (g::dsv4_lid_scores_mma(diq.p, dk.p, dw.p, s1.p, max_rows, pos0, T, nullptr) &&
            g::dsv4_lid_scores_mma(diq.p, dk.p, dw.p, s2.p, max_rows, 0, T, nullptr, dpos.p)) {
            auto c = s1.down(), d = s2.down();
            report_bool("lid scores tensor cores: device position == argument (bitwise)", bitwise(c.data(), d.data(), c.size() * 4));
        }
    }
}

// ================================================================== 5. sparse attention with sinks

/// Token t's keys as the kernel defines them (dsv4_kernels.hpp): the raw rows of positions max(0, p - 127) .. p, then the
/// compressed rows - as pointers into float copies of the F16 caches.
std::vector<const float*> ref_keys(int64_t p, int ratio, const std::vector<float>& raw, int raw_rows,
                                   const std::vector<float>& comp, const int32_t* sel_t, int top_k, int64_t hi = -1,
                                   int64_t b0 = -1) {
    std::vector<const float*> k;
    int64_t lo = std::max<int64_t>(0, p - 127);
    if (hi < p) hi = p;   // causal; a non-causal batch: through its last position (llama-kv-cache.cpp:1677-1703)
    else if (b0 >= 0) lo = std::min(lo, b0);   // ... and every row of the batch, past the window
    for (int64_t q = lo; q <= hi; ++q) k.push_back(&raw[(size_t) (q % raw_rows) * 512]);
    if (ratio == 4) {
        const int n = (int) std::min<int64_t>((p + 1) / 4, top_k);
        for (int i = 0; i < n; ++i) k.push_back(&comp[(size_t) sel_t[i] * 512]);
    } else if (ratio == 128) {
        for (int64_t j = 0; j < (p + 1) / 128; ++j) k.push_back(&comp[(size_t) j * 512]);
    }
    return k;
}

/// soft_max with the sink (ops.cpp:5584-5683: wp = kq * scale, max over the row and the sink, the exps summed in double
/// plus expf(sink - max), times (float)(1.0 / sum)), then kqv = sum_j p_j v_j.  The dots and the p.v sums are in double:
/// ggml-cpu's mul_mat rounds q (and its FA path the V accumulator) to F16, which this F32 kernel deliberately does not
/// (dsv4_kernels.hpp).  `llama_f16` applies the non-FA path's roundings instead (q -> F16 before the dot, p -> F16 before
/// kqv: mul_mat's vec_dot_type for F16 K and V), to show how far those move the output.  no_sink: the negative control.
void ref_attn_head(const float* q, const std::vector<const float*>& keys, float sink, float scale, float* out,
                   bool no_sink = false, bool llama_f16 = false) {
    const int n = (int) keys.size();
    float qr[512];
    for (int d = 0; d < 512; ++d) qr[d] = llama_f16 ? h2f(f2h(q[d])) : q[d];
    std::vector<float> s(n);
    for (int j = 0; j < n; ++j) {
        double dot = 0;
        for (int d = 0; d < 512; ++d) dot += (double) qr[d] * keys[j][d];
        s[j] = (float) dot * scale;
    }
    float mx = -INFINITY;
    for (float v : s) mx = std::max(mx, v);
    if (!no_sink) mx = std::max(mx, sink);
    double sum = 0;
    for (auto& v : s) sum += (double) (v = expf(v - mx));
    if (!no_sink) sum += (double) expf(sink - mx);
    const float inv = (float) (1.0 / sum);
    for (auto& v : s) v = llama_f16 ? h2f(f2h(v * inv)) : v * inv;
    std::vector<double> a(512, 0.0);
    for (int j = 0; j < n; ++j) {
        const double pj = s[j];
        const float* kj = keys[j];
        for (int d = 0; d < 512; ++d) a[d] += pj * kj[d];
    }
    for (int d = 0; d < 512; ++d) out[d] = (float) a[d];
}

struct AttnCase {
    const char* name;
    int ratio;
    int64_t pos0;
    int T;
    int raw_rows;
    int comp_rows;
    float q_sd;
    int tok_stride;   // check every tok_stride-th token on the CPU
    bool noncausal = false;   // moex: DSpark's noise block - every token also sees the batch's later rows
};

void attn_case(const AttnCase& c) {
    const int H = 64, D = 512, top_k = 512;
    const float scale = 1.0f / sqrtf(512.0f);
    auto q = randn((size_t) c.T * H * D, c.q_sd);
    const auto raw = to_f16(randn((size_t) c.raw_rows * D, 1.0f));
    const auto comp = to_f16(randn((size_t) std::max(c.comp_rows, 1) * D, 1.0f));
    auto sinks = randn(H, 2.0f);
    // CSA: each token's selection, ascending: all visible rows up to 512, else 512 random distinct visible rows
    std::vector<int32_t> sel((size_t) c.T * top_k, -1);
    if (c.ratio == 4)
        for (int t = 0; t < c.T; ++t) {
            const int n_vis = (int) ((c.pos0 + t + 1) / 4);
            std::vector<int32_t> rows(n_vis);
            std::iota(rows.begin(), rows.end(), 0);
            if (n_vis > top_k) {
                std::shuffle(rows.begin(), rows.end(), rng);
                rows.resize(top_k);
                std::sort(rows.begin(), rows.end());
            }
            std::copy(rows.begin(), rows.end(), sel.begin() + (size_t) t * top_k);
        }
    Dev<float> dq(q), dsink(sinks), out((size_t) c.T * H * D), out2((size_t) c.T * H * D);
    Dev<uint16_t> draw(raw), dcomp(comp);
    Dev<int32_t> dsel(sel);
    Dev<float> scratch(std::max<size_t>(g::dsv4_attn_scratch_bytes(c.T) / sizeof(float), 1));
    Dev<int64_t> dpos(std::vector<int64_t>{c.pos0});
    g::dsv4_attn(dq.p, draw.p, c.raw_rows, dcomp.p, c.ratio, dsel.p, top_k, dsink.p, scale, out.p, scratch.p, c.pos0, c.T, nullptr,
                 nullptr, c.noncausal);
    g::dsv4_attn(dq.p, draw.p, c.raw_rows, dcomp.p, c.ratio, dsel.p, top_k, dsink.p, scale, out2.p, scratch.p, -777, c.T, nullptr,
                 dpos.p, c.noncausal);
    auto got = out.down(), got2 = out2.down();
    const float ms = time_ms([&] {
        g::dsv4_attn(dq.p, draw.p, c.raw_rows, dcomp.p, c.ratio, dsel.p, top_k, dsink.p, scale, out2.p, scratch.p, c.pos0, c.T,
                     nullptr, nullptr, c.noncausal);
    });
    // the tensor-core path, through the same scratch
    Dev<float> out3((size_t) c.T * H * D), out4((size_t) c.T * H * D);
    const bool have_mma = g::dsv4_attn_mma(dq.p, draw.p, c.raw_rows, dcomp.p, c.ratio, dsel.p, top_k, dsink.p, scale, out3.p,
                                           scratch.p, c.pos0, c.T, nullptr, nullptr, c.noncausal);
    float ms_mma = 0;
    std::vector<float> got3, got4;
    if (have_mma) {
        g::dsv4_attn_mma(dq.p, draw.p, c.raw_rows, dcomp.p, c.ratio, dsel.p, top_k, dsink.p, scale, out4.p, scratch.p, -777, c.T,
                         nullptr, dpos.p, c.noncausal);
        got3 = out3.down();
        got4 = out4.down();
        ms_mma = time_ms([&] {
            g::dsv4_attn_mma(dq.p, draw.p, c.raw_rows, dcomp.p, c.ratio, dsel.p, top_k, dsink.p, scale, out4.p, scratch.p, c.pos0,
                             c.T, nullptr, nullptr, c.noncausal);
        });
    }
    std::vector<float> g1, g3, r1, rbad, rf16;
    const auto rawf = from_f16(raw), compf = from_f16(comp);
    int max_keys = 0;
    for (int t = 0; t < c.T; t += c.tok_stride) {
        const int64_t p = c.pos0 + t;
        const auto keys = ref_keys(p, c.ratio, rawf, c.raw_rows, compf, &sel[(size_t) t * top_k], top_k,
                                   c.noncausal ? c.pos0 + c.T - 1 : -1, c.noncausal ? c.pos0 : -1);
        max_keys = std::max(max_keys, (int) keys.size());
        std::vector<float> o(D), ob(D), of(D);
        for (int h = 0; h < H; ++h) {
            const float* qh = &q[((size_t) t * H + h) * D];
            ref_attn_head(qh, keys, sinks[h], scale, o.data());
            ref_attn_head(qh, keys, sinks[h], scale, ob.data(), true);
            ref_attn_head(qh, keys, sinks[h], scale, of.data(), false, true);
            const float* gh = &got[((size_t) t * H + h) * D];
            g1.insert(g1.end(), gh, gh + D);
            if (have_mma) g3.insert(g3.end(), &got3[((size_t) t * H + h) * D], &got3[((size_t) t * H + h) * D] + D);
            r1.insert(r1.end(), o.begin(), o.end());
            rbad.insert(rbad.end(), ob.begin(), ob.end());
            rf16.insert(rf16.end(), of.begin(), of.end());
        }
    }
    char name[160];
    // F32 dots and online softmax against double: the order of summation and expf ulps
    snprintf(name, sizeof name, "attn %s (<= %d keys)", c.name, max_keys);
    report(name, rel(g1, r1), 2e-6);
    snprintf(name, sizeof name, "attn %s vs a reference without the sink", c.name);
    report_neg(name, rel(g1, rbad), 2e-6);
    // information, with a gross bound: llama.cpp's own F16 roundings of q and p move the output by ~1e-3
    snprintf(name, sizeof name, "attn %s vs llama.cpp non-FA F16 roundings [info]", c.name);
    report(name, rel(g1, rf16), 2e-2);
    snprintf(name, sizeof name, "attn %s: device position == argument (bitwise)", c.name);
    report_bool(name, bitwise(got.data(), got2.data(), got.size() * 4));
    if (have_mma) {
        // q and p in F16 hi + lo: each operand to ~2^-22 relative (the lo half's own rounding; smaller q elements, scaled
        // against the block's largest, keep less), F32 accumulation in the tensor cores' order, exp2f(s log2 e): ~1e-6
        snprintf(name, sizeof name, "attn %s tensor cores", c.name);
        report(name, rel(g3, r1), 5e-6);
        snprintf(name, sizeof name, "attn %s tensor cores vs the SIMT kernel", c.name);
        report(name, rel(got3, got), 5e-6);
        snprintf(name, sizeof name, "attn %s tensor cores vs a reference without the sink", c.name);
        report_neg(name, rel(g3, rbad), 5e-6);
        snprintf(name, sizeof name, "attn %s tensor cores: device position == argument (bitwise)", c.name);
        report_bool(name, bitwise(got3.data(), got4.data(), got3.size() * 4));
    }
    printf("   attn %s: %.3f ms, tensor cores %.3f ms (mean of 20, %d split(s))\n", c.name, ms, ms_mma, g::dsv4_attn_splits(c.T));
}

void test_attn() {
    const AttnCase cases[] = {
        {"raw T=1 pos 0", 0, 0, 1, 256, 0, 1.0f, 1},
        {"raw T=8 pos 3", 0, 3, 8, 256, 0, 1.0f, 1},
        {"raw T=5 pos 125 (window edge)", 0, 125, 5, 256, 0, 1.0f, 1},
        {"raw T=7 pos 200", 0, 200, 7, 256, 0, 4.0f, 1},
        {"CSA T=3 pos 9 (2-4 rows, all)", 4, 9, 3, 256, 16, 1.0f, 1},
        {"CSA T=8 pos 2040 (510-512 rows, all)", 4, 2040, 8, 256, 520, 1.0f, 1},
        {"CSA T=5 pos 2600 (top 512 of 650)", 4, 2600, 5, 256, 700, 1.0f, 1},
        {"CSA T=1 pos 100000 (top 512 of 25000)", 4, 100000, 1, 256, 25001, 4.0f, 1},
        {"HCA T=2 pos 126 (rows 0..1)", 128, 126, 2, 256, 4, 1.0f, 1},
        {"HCA T=8 pos 299990 (2343 rows)", 128, 299990, 8, 256, 2344, 1.0f, 1},
        {"HCA T=1 pos 1048575 (8192 rows)", 128, 1048575, 1, 256, 8192, 1.0f, 1},
        {"CSA T=300 pos 517 (prompt)", 4, 517, 300, 512, 300, 1.0f, 13},
        {"HCA T=300 pos 517 (prompt)", 128, 517, 300, 512, 8, 1.0f, 13},
        {"raw T=70 pos 0 (one split)", 0, 0, 70, 256, 0, 1.0f, 3},
        // moex: DSpark's noise block (non-causal): every token sees the block's later rows too
        {"raw non-causal T=5 pos 0", 0, 0, 5, 256, 0, 1.0f, 1, true},
        {"raw non-causal T=5 pos 300", 0, 300, 5, 256, 0, 1.0f, 1, true},
        {"raw non-causal T=8 pos 125 (window edge)", 0, 125, 8, 256, 0, 4.0f, 1, true},
        // moex: DeepSeek-V4's image span (up to 384 rows): its rows see the whole span, past the 128-position window
        {"raw non-causal T=300 pos 700 (an image span)", 0, 700, 300, 512, 0, 1.0f, 7, true},
        {"CSA non-causal T=200 pos 3001 (an image span)", 4, 3001, 200, 512, 800, 1.0f, 9, true},
    };
    for (const AttnCase& c : cases) attn_case(c);
}

// ================================================================== 6. head hyper-connection reduce

void test_hc_head() {
    const int n = 4096, T = 5;
    const float eps = 9.99999997e-07f;   // hyper_connection.epsilon
    auto x = randn((size_t) T * 4 * n, 1.0f), mixes = randn((size_t) T * 4, 3.0f), sc = randn(1, 0.3f, 1.0f),
         base = randn(4, 1.0f);
    // deepseek4.cpp:449-469 on ggml-cpu: ggml_mul, ggml_add (rounded), sigmoid (unary-ops.cpp:31), scale_bias(1, eps),
    // then build_hc_pre(.., -1): four ggml_mul and three ggml_add (deepseek4.cpp:307-316)
    std::vector<float> ref((size_t) T * n), bad((size_t) T * n);
    for (int t = 0; t < T; ++t) {
        float pre[4];
        for (int s = 0; s < 4; ++s) {
            const float a = mixes[(size_t) t * 4 + s] * sc[0];
            const float b = a + base[s];
            pre[s] = 1.0f / (1.0f + expf(-b)) + eps;
        }
        for (int i = 0; i < n; ++i) {
            const float* xt = &x[(size_t) t * 4 * n];
            const float p0 = xt[i] * pre[0], p1 = xt[n + i] * pre[1], p2 = xt[2 * n + i] * pre[2], p3 = xt[3 * n + i] * pre[3];
            ref[(size_t) t * n + i] = ((p0 + p1) + p2) + p3;
            bad[(size_t) t * n + i] = ((xt[i] * pre[1] + xt[n + i] * pre[0]) + p2) + p3;   // pre0 and pre1 swapped
        }
    }
    Dev<float> dx(x), dm(mixes), dsc(sc), db(base), dy((size_t) T * n);
    g::dsv4_hc_head(dx.p, dm.p, dsc.p, db.p, eps, dy.p, n, T, nullptr);
    auto got = dy.down();
    // the same operations, each rounded; only expf may differ by an ulp (glibc vs CUDA's), moving pre by ~1e-7
    report("hc_head (unfused, every op rounded)", rel(got, ref), 1e-6);
    size_t same = 0;
    for (size_t i = 0; i < got.size(); ++i) same += memcmp(&got[i], &ref[i], 4) == 0;
    printf("   hc_head: %zu / %zu outputs bitwise equal to the reference\n", same, got.size());
    report_neg("hc_head vs a reference with pre0/pre1 swapped", rel(got, bad), 1e-6);
}

// ================================================================== 7. router

void test_router() {
    const int E = 256, k = 6, T = 7, n_vocab = 1000;
    const float scale = 1.5f;
    auto logits = randn((size_t) T * E, 2.0f), bias = randn(E, 0.3f);
    // probs: unary-ops.cpp:80 then sqrtf (unary-ops.cpp:51); top-k of probs + bias by a sort descending (ops.cpp:8489-8528;
    // stable here: ties to the lower id); weights the UNBIASED probs, sum_rows (double), clamp, div, scale
    // (llama-graph.cpp:1997-2099)
    auto probs = [&](int t, int e) {
        const float x = logits[(size_t) t * E + e];
        return sqrtf(x > 20.0f ? x : logf(1.0f + expf(x)));
    };
    auto ref_weights = [&](int t, const std::vector<int>& ids, bool biased) {
        std::vector<float> w(ids.size());
        double s = 0;
        for (size_t j = 0; j < ids.size(); ++j) s += (w[j] = probs(t, ids[j]) + (biased ? bias[ids[j]] : 0.0f));
        const float sum = std::max((float) s, 6.103515625e-5f);
        for (auto& v : w) v = v / sum * scale;
        return w;
    };
    for (int mode = 0; mode < 3; ++mode) {   // 0: bias, 1: no bias, 2: hash ids
        Dev<float> dl(logits), db(bias), dw((size_t) T * k);
        Dev<int32_t> di((size_t) T * k);
        std::vector<int32_t> hash((size_t) T * k);
        Dev<int32_t> dh((size_t) T * k);
        if (mode == 2) {   // the ids through dsv4_hash_ids from a tid2eid table and token ids
            std::uniform_int_distribution<int> ue(0, E - 1), ut(0, n_vocab - 1);
            std::vector<int32_t> tab((size_t) n_vocab * k), tok(T);
            for (auto& v : tab) v = ue(rng);
            for (auto& v : tok) v = ut(rng);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < k; ++j) hash[(size_t) t * k + j] = tab[(size_t) tok[t] * k + j];
            Dev<int32_t> dtab(tab), dtok(tok);
            g::dsv4_hash_ids(dtab.p, dtok.p, k, T, dh.p, nullptr);
            auto hv = dh.down();
            report_bool("router hash_ids = tid2eid[token] (exact)", hv == hash);
        }
        g::dsv4_route(dl.p, mode == 0 ? db.p : nullptr, mode == 2 ? dh.p : nullptr, E, k, scale, di.p, dw.p, T, nullptr);
        auto ids = di.down();
        auto ws = dw.down();
        bool ids_ok = true;
        std::vector<float> got, want, bad;
        for (int t = 0; t < T; ++t) {
            std::vector<int> rid;
            if (mode == 2) {
                rid.assign(hash.begin() + (size_t) t * k, hash.begin() + (size_t) (t + 1) * k);
            } else {
                std::vector<int> o(E);
                std::iota(o.begin(), o.end(), 0);
                std::stable_sort(o.begin(), o.end(), [&](int a, int b) {
                    const float sa = probs(t, a) + (mode == 0 ? bias[a] : 0.0f), sb = probs(t, b) + (mode == 0 ? bias[b] : 0.0f);
                    return sa > sb;
                });
                rid.assign(o.begin(), o.begin() + k);
            }
            const std::vector<int> gid(ids.begin() + (size_t) t * k, ids.begin() + (size_t) (t + 1) * k);
            if (mode == 2) ids_ok &= gid == rid;   // as given, in order
            else ids_ok &= std::set<int>(gid.begin(), gid.end()) == std::set<int>(rid.begin(), rid.end());
            // the weights matched by id
            const auto wr = ref_weights(t, rid, false), wb = ref_weights(t, rid, true);
            for (int j = 0; j < k; ++j) {
                const int pos = (int) (std::find(gid.begin(), gid.end(), rid[j]) - gid.begin());
                got.push_back(pos < k ? ws[(size_t) t * k + pos] : 0.0f);
                want.push_back(wr[j]);
                bad.push_back(wb[j]);
            }
        }
        static const char* mname[] = {"top-6 of probs + bias", "top-6 of probs (no bias)", "hash ids"};
        char name[128];
        snprintf(name, sizeof name, "router %s: ids (%s)", mname[mode], mode == 2 ? "exact" : "sets");
        report_bool(name, ids_ok);
        // the weights: sqrt/log/exp ulps, a 6-term sum in float vs double and w * (1/sum) vs w / sum
        snprintf(name, sizeof name, "router %s: weights", mname[mode]);
        report(name, rel(got, want), 2e-6);
        if (mode == 0) report_neg("router weights vs the BIASED probs", rel(got, bad), 2e-6);
    }
}

/// moex (DSV41_BACKLOG #56): DeepSeek-V4.1's router with torch's softplus (log1p(e^x), exact_softplus) against a CPU reference,
/// on logits around -14 (the chosen ones near -11) where log(1 + e^x) loses the small values - and, as the negative control,
/// unlike that form
void test_router_exact() {
    const int E = 384, k = 6, T = 7;
    const float scale = 1.5f;
    auto logits = randn((size_t) T * E, 1.0f), bias = randn(E, 1e-4f);
    for (auto& v : logits) v -= 14.0f;
    auto probs = [&](int t, int e, bool exact) {
        const float x = logits[(size_t) t * E + e];
        return sqrtf(x > 20.0f ? x : exact ? log1pf(expf(x)) : logf(1.0f + expf(x)));
    };
    for (int mode = 0; mode < 2; ++mode) {   // 0: bias, 1: no bias
        Dev<float> dl(logits), db(bias), dw((size_t) T * k);
        Dev<int32_t> di((size_t) T * k);
        g::dsv4_route(dl.p, mode == 0 ? db.p : nullptr, nullptr, E, k, scale, di.p, dw.p, T, nullptr, nullptr, true);
        auto ids = di.down();
        auto ws = dw.down();
        bool ids_ok = true;
        std::vector<float> got, want, bad;
        for (int t = 0; t < T; ++t) {
            std::vector<int> o(E);
            std::iota(o.begin(), o.end(), 0);
            std::stable_sort(o.begin(), o.end(), [&](int a, int b) {
                return probs(t, a, true) + (mode == 0 ? bias[a] : 0.0f) > probs(t, b, true) + (mode == 0 ? bias[b] : 0.0f);
            });
            const std::vector<int> rid(o.begin(), o.begin() + k);
            const std::vector<int> gid(ids.begin() + (size_t) t * k, ids.begin() + (size_t) (t + 1) * k);
            ids_ok &= std::set<int>(gid.begin(), gid.end()) == std::set<int>(rid.begin(), rid.end());
            double sx = 0, sl = 0;
            for (int j = 0; j < k; ++j) {
                sx += probs(t, rid[j], true);
                sl += probs(t, rid[j], false);
            }
            for (int j = 0; j < k; ++j) {
                const int pos = (int) (std::find(gid.begin(), gid.end(), rid[j]) - gid.begin());
                got.push_back(pos < k ? ws[(size_t) t * k + pos] : 0.0f);
                want.push_back(probs(t, rid[j], true) / std::max((float) sx, 6.103515625e-5f) * scale);
                bad.push_back(probs(t, rid[j], false) / std::max((float) sl, 6.103515625e-5f) * scale);
            }
        }
        report_bool(mode == 0 ? "router log1p (V4.1), probs + bias: ids (sets)" : "router log1p (V4.1), no bias: ids (sets)", ids_ok);
        report(mode == 0 ? "router log1p (V4.1), probs + bias: weights" : "router log1p (V4.1), no bias: weights", rel(got, want), 2e-6);
        report_neg(mode == 0 ? "router log1p (V4.1) vs log(1 + e^x), probs + bias" : "router log1p (V4.1) vs log(1 + e^x), no bias",
                   rel(got, bad), 2e-6);
    }
}

// ================================================================== 8. a decode window captured in a CUDA graph

/// The CSA layer's window kernels (raw store, rope, the CSA and LID compressors, the indexer's rope + Hadamard + scores +
/// selection, the attention, the de-rope) captured once with the position in device memory, replayed at two positions;
/// against the same calls made directly with the positions as arguments on the same starting state: bitwise.  `mma`:
/// with the tensor-core indexer and attention.
void test_graph(bool mma) {
    if (mma && !g::dsv4_mma_supported()) {
        printf("graph test (tensor cores): skipped, no sm_80 tensor cores here (or STRATA_DSV4_MMA=0)\n");
        return;
    }
    const int T = 4, H = 64, D = 512, RS = 256, top_k = 512, max_rows = 1024;
    const int64_t positions[2] = {2601, 2605};
    const g::Dsv4Rope P = g::dsv4_rope_pc();
    const g::Dsv4Compressor csa = g::dsv4_csa(), lid = g::dsv4_lid();
    const float eps = 9.99999968e-21f, scale = 1.0f / sqrtf(512.0f);
    // the starting state: a raw ring, the CSA / LID caches with their first 650 rows, rings of random history
    const auto raw0 = to_f16(randn((size_t) RS * D));
    auto csa_c0 = to_f16(randn((size_t) max_rows * D)), lid_c0 = to_f16(randn((size_t) max_rows * 128));
    auto rkv0 = randn((size_t) 8 * 1024), rsc0 = randn((size_t) 8 * 1024, 2.0f), lkv0 = randn((size_t) 8 * 256),
         lsc0 = randn((size_t) 8 * 256, 2.0f);
    const auto ape_c = randn((size_t) 4 * 1024, 0.5f), ape_l = randn((size_t) 4 * 256, 0.5f), nw_c = randn(512, 0.2f, 1.0f),
               nw_l = randn(128, 0.2f, 1.0f), sinks = randn(H, 2.0f);
    // per step inputs
    std::vector<std::vector<float>> q(2), kv(2), akv(2), gs(2), lakv(2), lgs(2), iq(2), w(2);
    for (int s = 0; s < 2; ++s) {
        q[s] = randn((size_t) T * H * D);
        kv[s] = randn((size_t) T * D);
        akv[s] = randn((size_t) T * 1024);
        gs[s] = randn((size_t) T * 1024, 2.0f);
        lakv[s] = randn((size_t) T * 256);
        lgs[s] = randn((size_t) T * 256, 2.0f);
        iq[s] = randn((size_t) T * 64 * 128);
        w[s] = randn((size_t) T * 64, 0.05f);
    }
    struct State {
        Dev<uint16_t> raw, csa_c, lid_c;
        Dev<float> rkv, rsc, lkv, lsc;
        Dev<float> q, kv, akv, gs, lakv, lgs, iq, w, score, out, scratch;
        Dev<int32_t> sel;
        State(const std::vector<uint16_t>& r, const std::vector<uint16_t>& cc, const std::vector<uint16_t>& lc,
              const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c, const std::vector<float>& d)
            : raw(r), csa_c(cc), lid_c(lc), rkv(a), rsc(b), lkv(c), lsc(d), q((size_t) T * H * D), kv((size_t) T * D),
              akv((size_t) T * 1024), gs((size_t) T * 1024), lakv((size_t) T * 256), lgs((size_t) T * 256),
              iq((size_t) T * 64 * 128), w((size_t) T * 64), score((size_t) T * max_rows), out((size_t) T * H * D),
              scratch(std::max<size_t>(g::dsv4_attn_scratch_bytes(T) / 4, 1)), sel((size_t) T * top_k) {}
    };
    Dev<float> dape_c(ape_c), dape_l(ape_l), dnw_c(nw_c), dnw_l(nw_l), dsink(sinks);
    Dev<int64_t> dpos(1);
    auto window = [&](State& S, int64_t pos0, const int64_t* pd, cudaStream_t s) {
        g::dsv4_raw_store(S.kv.p, S.raw.p, RS, pos0, T, s, pd);
        g::dsv4_rope(S.q.p, S.q.p, T, H, D, 448, P, false, pos0, s, pd);
        g::dsv4_compress_pool(csa, S.akv.p, S.gs.p, dape_c.p, S.rkv.p, S.rsc.p, dnw_c.p, eps, P, S.csa_c.p, pos0, T, s, pd);
        g::dsv4_compress_persist(csa, S.akv.p, S.gs.p, dape_c.p, S.rkv.p, S.rsc.p, pos0, T, s, pd);
        g::dsv4_compress_pool(lid, S.lakv.p, S.lgs.p, dape_l.p, S.lkv.p, S.lsc.p, dnw_l.p, eps, P, S.lid_c.p, pos0, T, s, pd);
        g::dsv4_compress_persist(lid, S.lakv.p, S.lgs.p, dape_l.p, S.lkv.p, S.lsc.p, pos0, T, s, pd);
        g::dsv4_rope(S.iq.p, S.iq.p, T, 64, 128, 64, P, false, pos0, s, pd);
        g::dsv4_hadamard(S.iq.p, S.iq.p, T * 64, 128, s);
        if (!mma || !g::dsv4_lid_scores_mma(S.iq.p, S.lid_c.p, S.w.p, S.score.p, max_rows, pos0, T, s, pd))
            g::dsv4_lid_scores(S.iq.p, S.lid_c.p, S.w.p, S.score.p, max_rows, pos0, T, s, pd);
        g::dsv4_lid_select(S.score.p, max_rows, top_k, S.sel.p, nullptr, pos0, T, s, pd);
        if (!mma || !g::dsv4_attn_mma(S.q.p, S.raw.p, RS, S.csa_c.p, 4, S.sel.p, top_k, dsink.p, scale, S.out.p, S.scratch.p,
                                      pos0, T, s, pd))
            g::dsv4_attn(S.q.p, S.raw.p, RS, S.csa_c.p, 4, S.sel.p, top_k, dsink.p, scale, S.out.p, S.scratch.p, pos0, T, s, pd);
        g::dsv4_rope(S.out.p, S.out.p, T, H, D, 448, P, true, pos0, s, pd);   // the de-rope
    };
    auto load = [&](State& S, int s) {
        S.q.up(q[s]);
        S.kv.up(kv[s]);
        S.akv.up(akv[s]);
        S.gs.up(gs[s]);
        S.lakv.up(lakv[s]);
        S.lgs.up(lgs[s]);
        S.iq.up(iq[s]);
        S.w.up(w[s]);
    };
    State A(raw0, csa_c0, lid_c0, rkv0, rsc0, lkv0, lsc0), B(raw0, csa_c0, lid_c0, rkv0, rsc0, lkv0, lsc0);
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    window(B, -1, dpos.p, s);
    CK(cudaStreamEndCapture(s, &graph));
    CK(cudaGraphInstantiate(&exec, graph, 0));
    std::vector<float> oa[2], ob[2];
    for (int st = 0; st < 2; ++st) {
        load(A, st);
        window(A, positions[st], nullptr, s);
        CK(cudaStreamSynchronize(s));
        oa[st] = A.out.down();
        load(B, st);
        CK(cudaMemcpy(dpos.p, &positions[st], sizeof(int64_t), cudaMemcpyHostToDevice));
        CK(cudaGraphLaunch(exec, s));
        CK(cudaStreamSynchronize(s));
        ob[st] = B.out.down();
    }
    bool ok = bitwise(oa[0].data(), ob[0].data(), oa[0].size() * 4) && bitwise(oa[1].data(), ob[1].data(), oa[1].size() * 4);
    auto ca = A.csa_c.down(), cb = B.csa_c.down(), la = A.lid_c.down(), lb = B.lid_c.down();
    auto ra = A.raw.down(), rb = B.raw.down();
    auto ka = A.rkv.down(), kb = B.rkv.down();
    ok = ok && bitwise(ca.data(), cb.data(), ca.size() * 2) && bitwise(la.data(), lb.data(), la.size() * 2) &&
         bitwise(ra.data(), rb.data(), ra.size() * 2) && bitwise(ka.data(), kb.data(), ka.size() * 4);
    report_bool(mma ? "a CSA decode window as a CUDA graph (tensor cores), replayed at 2601, 2605 = direct (bitwise)"
                    : "a CSA decode window as a CUDA graph (SIMT), replayed at 2601, 2605 = direct calls (bitwise)",
                ok);
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    CK(cudaStreamDestroy(s));
}

}  // namespace

// STRATA_DSV4_PARITY_ONLY=lid_bench: the lightning indexer's prompt-path scores on one 512-token sub-batch
// (EngineOptions::prefill_sub) at growing depth, the tensor-core kernel and the F32 one, and what one 8K chunk pays for
// them (16 sub-batches x the 21 ratio-4 layers).  Timing only (random data).
void bench_lid() {
    const int T = 512;
    for (const int64_t depth : {(int64_t) 10240, (int64_t) 51200, (int64_t) 102400, (int64_t) 204800}) {
        const int max_rows = (int) ((depth + T) / 4 + 1);
        Dev<float> diq(randn((size_t) T * 64 * 128)), dw(randn((size_t) T * 64, 0.05f)), sc((size_t) T * max_rows);
        Dev<uint16_t> dk(to_f16(randn((size_t) max_rows * 128)));
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        auto best_of = [&](auto&& f) {
            f();
            CK(cudaDeviceSynchronize());
            float best = 1e30f;
            for (int r = 0; r < 5; ++r) {
                CK(cudaEventRecord(a));
                f();
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, a, b));
                best = std::min(best, ms);
            }
            return best;
        };
        bool mma = true;
        const float t_mma = best_of([&] { mma = g::dsv4_lid_scores_mma(diq.p, dk.p, dw.p, sc.p, max_rows, depth, T, nullptr); });
        const float t_f32 = best_of([&] { g::dsv4_lid_scores(diq.p, dk.p, dw.p, sc.p, max_rows, depth, T, nullptr); });
        const double fma = (double) T * (double) (depth / 4) * 64 * 128;
        printf("   lid depth %6lldK, 512 queries: tensor cores %s%8.2f ms (%5.0f GFMA/s), F32 %8.2f ms -> an 8K chunk's 21 layers: "
               "%6.2f s (tensor cores)\n", (long long) (depth / 1024), mma ? "" : "(unavailable) ", t_mma, fma / (t_mma * 1e6),
               t_f32, t_mma * 16 * 21 / 1000.0);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
}

int main() {
    int dev = 0, n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0 || cudaGetDevice(&dev) != cudaSuccess) {
        printf("no CUDA device\n");
        return 77;
    }
    if (const char* only = std::getenv("STRATA_DSV4_PARITY_ONLY"); only && std::string(only) == "lid_bench") {
        bench_lid();
        return 0;
    }
    const auto t0 = std::chrono::steady_clock::now();
    test_rope_params();
    test_rope();
    test_raw_store();
    test_hadamard();
    test_compressors();
    test_select();
    test_indexer();
    test_attn();
    test_hc_head();
    test_router();
    test_router_exact();
    test_graph(false);
    test_graph(true);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("%s: %d failure(s) (%.1f s)\n", g_fail ? "FAIL" : "PASS", g_fail, s);
    return g_fail ? 1 : 0;
}
