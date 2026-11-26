// moex/tools/ernie55/ernie_kernels_check.cu - ERNIE-5.5-Flash's kernels (src/x/ernie_kernels.cu, SiTU-GLU in
// src/x/kernels.cu, dsv4_lid_select's window exclusion) against CPU transcriptions, on random inputs (no model).
//
//     ernie_kernels_check            (exit 0 when every check holds)
//
// Bitwise where the CPU can follow the kernel's operation order (MultiMax, the E4M3 rounding, the cells, the routing's ids),
// else within a stated tolerance against double sums (VHA, the RMS factor, SiTU-GLU's tanh / exp).
#include "strata/x/dsv4_kernels.hpp"
#include "strata/x/ernie_kernels.hpp"
#include "strata/x/kernels.hpp"
#include "strata/kernels/glu_act.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using namespace strata::x;

namespace {

int g_fail = 0;

void check(bool ok, const char* what, const char* detail = "") {
    printf("%s  %s%s%s\n", ok ? "ok  " : "FAIL", what, *detail ? ": " : "", detail);
    if (!ok) ++g_fail;
}

template <typename T>
T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T));
    if (!h.empty()) cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <typename T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

/// float -> E4M3 (finite, saturating) -> float, round to nearest even: the nearest of the 127 positive codes' values,
/// ties to the even code
float e4m3_round(float x) {
    static std::vector<float> vals = [] {
        std::vector<float> v;
        for (int c = 0; c < 127; ++c) {   // 0x00 .. 0x7E (0x7F is NaN)
            const int e = c >> 3, m = c & 7;
            v.push_back(e == 0 ? std::ldexp((float) m / 8.0f, -6) : std::ldexp(1.0f + (float) m / 8.0f, e - 7));
        }
        return v;
    }();
    const float a = std::fabs(x);
    if (a >= 448.0f) return std::copysign(448.0f, x);
    int best = 0;
    for (int c = 1; c < 127; ++c) {
        const float d0 = std::fabs(a - vals[(size_t) best]), d1 = std::fabs(a - vals[(size_t) c]);
        if (d1 < d0 || (d1 == d0 && (c & 1) == 0)) best = c;
    }
    return std::copysign(vals[(size_t) best], x);
}

float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

int main() {
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    char msg[256];

    // ---- VHA: out[t, h, :] = sum_g M[h, g] in[t, g, :]
    {
        const int T = 5, H = 64, D = 512;
        std::vector<float> in((size_t) T * H * D), M((size_t) H * H);
        for (auto& v : in) v = nd(rng);
        for (int a = 0; a < H; ++a)
            for (int b = 0; b < H; ++b) M[(size_t) (a * H + b)] = (a == b ? 1.0f : 0.0f) + 0.05f * nd(rng);
        float *din = dev(in), *dM = dev(M), *dout = dev(std::vector<float>(in.size()));
        ernie_vha_w(din, dout, dM, T, H, D, nullptr);
        const auto out = host(dout, in.size());
        // the recorder's measure (RMS of the error over RMS of the values) and the worst error over the largest value
        double num = 0, den = 0, worst = 0, big = 0;
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d) {
                    double s = 0;
                    for (int g = 0; g < H; ++g) s += (double) M[(size_t) (h * H + g)] * in[((size_t) t * H + g) * D + d];
                    const double e = s - out[((size_t) t * H + h) * D + d];
                    num += e * e;
                    den += s * s;
                    worst = std::max(worst, std::fabs(e));
                    big = std::max(big, std::fabs(s));
                }
        snprintf(msg, sizeof msg, "relative %.2e, worst error %.2e of the largest value (T 5, 64 heads of 512)", std::sqrt(num / den),
                 worst / big);
        check(std::sqrt(num / den) < 1e-6 && worst / big < 1e-6, "VHA", msg);
        cudaFree(din);
        cudaFree(dM);
        cudaFree(dout);
    }

    // ---- the two-router routing
    {
        const int T = 6, E = 512, k = 10;
        std::vector<float> lg((size_t) T * 2 * E), bias(E), esc(E);
        for (auto& v : lg) v = 2.0f * nd(rng);
        for (auto& v : bias) v = 0.3f * nd(rng);
        for (auto& v : esc) v = 1.0f + 0.5f * std::fabs(nd(rng));
        lg[(size_t) 0 * 2 * E + 7] = lg[(size_t) 0 * 2 * E + 9] = 1.5f;   // a tie in token 0: experts 7 and 9 alike
        lg[(size_t) 0 * 2 * E + E + 7] = lg[(size_t) 0 * 2 * E + E + 9] = 1.5f;
        bias[7] = bias[9] = 3.0f;
        float *dl = dev(lg), *db = dev(bias), *de = dev(esc), *dw = dev(std::vector<float>((size_t) T * k));
        int32_t* di = dev(std::vector<int32_t>((size_t) T * k));
        ernie_route_w(dl, db, de, E, k, di, dw, T, nullptr);
        const auto ids = host(di, (size_t) T * k);
        const auto wts = host(dw, (size_t) T * k);
        bool ids_ok = true;
        double wworst = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<float> sc(E), ch(E);
            for (int e = 0; e < E; ++e) {
                sc[(size_t) e] = sigm(lg[(size_t) t * 2 * E + e]) + sigm(lg[(size_t) t * 2 * E + E + e]);
                ch[(size_t) e] = sc[(size_t) e] + bias[(size_t) e];
            }
            std::vector<int> ord(E);
            std::iota(ord.begin(), ord.end(), 0);
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return ch[(size_t) a] > ch[(size_t) b]; });
            double sum = 0;
            for (int j = 0; j < k; ++j) sum += sc[(size_t) ord[(size_t) j]];
            for (int j = 0; j < k; ++j) {
                if (ids[(size_t) t * k + j] != ord[(size_t) j]) ids_ok = false;
                const double w = sc[(size_t) ord[(size_t) j]] / sum * esc[(size_t) ord[(size_t) j]];
                wworst = std::max(wworst, std::fabs(w - wts[(size_t) t * k + j]) / w);
            }
        }
        check(ids_ok, "routing: the ids (descending score + bias, ties to the lower expert)");
        snprintf(msg, sizeof msg, "worst relative %.2e", wworst);
        check(wworst < 1e-6, "routing: the weights (normalized, times the per-expert factor)", msg);
        cudaFree(dl);
        cudaFree(db);
        cudaFree(de);
        cudaFree(dw);
        cudaFree(di);
    }

    // ---- the warp routing kernel (the default) against the block kernel, bit for bit: ids in order and weights
    for (const auto& cfg : std::vector<std::array<int, 3>>{{2000, 512, 10}, {300, 512, 1}, {257, 1024, 32}, {64, 100, 7}}) {
        const int T = cfg[0], E = cfg[1], k = cfg[2];
        std::vector<float> lg((size_t) T * 2 * E), bias(E), esc(E);
        for (auto& v : lg) v = 2.0f * nd(rng);
        for (auto& v : bias) v = 0.3f * nd(rng);
        for (auto& v : esc) v = 1.0f + 0.5f * std::fabs(nd(rng));
        for (int t = 0; t < T; t += 3)   // ties: logits rounded to a coarse grid (equal scores across experts)
            for (int e = 0; e < 2 * E; ++e) lg[(size_t) t * 2 * E + e] = std::round(lg[(size_t) t * 2 * E + e] * 2.0f) / 2.0f;
        if (T > 10) {   // a token of +inf / -inf logits, one of NaN logits but k + 3 experts
            for (int e = 0; e < 2 * E; e += 5) lg[(size_t) 1 * 2 * E + e] = (e % 10 == 0) ? INFINITY : -INFINITY;
            for (int e = 0; e < 2 * E; ++e) lg[(size_t) 2 * 2 * E + e] = e % E < k + 3 ? 0.5f * (float) (e % 7) : NAN;
        }
        float *dl = dev(lg), *db = dev(bias), *de = dev(esc), *dw = dev(std::vector<float>((size_t) T * k)),
              *dw2 = dev(std::vector<float>((size_t) T * k));
        int32_t *di = dev(std::vector<int32_t>((size_t) T * k)), *di2 = dev(std::vector<int32_t>((size_t) T * k));
        ernie_route_w(dl, db, de, E, k, di, dw, T, nullptr);
        ernie_route_block_w(dl, db, de, E, k, di2, dw2, T, nullptr);
        const auto a = host(di, (size_t) T * k), b = host(di2, (size_t) T * k);
        const auto wa = host(dw, (size_t) T * k), wb = host(dw2, (size_t) T * k);
        const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0 && std::memcmp(wa.data(), wb.data(), wa.size() * 4) == 0;
        snprintf(msg, sizeof msg, "T %d, E %d, k %d (ties, +-inf and NaN tokens)", T, E, k);
        check(same, "routing: the warp kernel = the block kernel, ids and weights bit for bit", msg);
        cudaFree(dl);
        cudaFree(db);
        cudaFree(de);
        cudaFree(dw);
        cudaFree(dw2);
        cudaFree(di);
        cudaFree(di2);
    }

    // ---- MultiMax, bitwise against the reference's op order
    {
        const int rows = 3;
        const int64_t V = 70000, ld = 70016;
        std::vector<float> x((size_t) rows * ld);
        for (auto& v : x) v = 8.0f * nd(rng);
        ErnieMultiMax mm;
        const float r[4] = {-3.0f, 4.5f, -6.0f, 9.0f}, tt[4] = {0.25f, -0.125f, 0.03125f, -0.0078125f};
        for (int i = 0; i < 4; ++i) {
            mm.r[i] = r[i];
            mm.t[i] = tt[i];
        }
        float* dx = dev(x);
        ernie_multimax_w(dx, V, rows, ld, mm, nullptr);
        const auto y = host(dx, x.size());
        int64_t bad = 0;
        for (int rr = 0; rr < rows; ++rr)
            for (int64_t i = 0; i < ld; ++i) {
                const float v = x[(size_t) (rr * ld + i)];
                float want = v;
                if (i < V) {
                    volatile float a = v + tt[0] * std::max(r[0] - v, 0.0f);
                    volatile float b = a + tt[1] * std::max(v - r[1], 0.0f);
                    volatile float p = std::max(r[2] - v, 0.0f);
                    volatile float pp = p * p;
                    volatile float c = b + tt[2] * pp;
                    volatile float q = std::max(v - r[3], 0.0f);
                    volatile float qq = q * q;
                    want = c + tt[3] * qq;
                }
                if (std::memcmp(&want, &y[(size_t) (rr * ld + i)], 4) != 0) ++bad;
            }
        snprintf(msg, sizeof msg, "%lld of %lld values differ (the row padding untouched)", (long long) bad, (long long) (rows * ld));
        check(bad == 0, "MultiMax, bit for bit", msg);
        cudaFree(dx);
    }

    // ---- the indexer's E4M3 rounding per 128 with a power-of-two scale
    {
        const int64_t groups = 300;
        std::vector<float> x((size_t) groups * 128);
        for (int64_t g = 0; g < groups; ++g) {
            const float sc = std::ldexp(1.0f, (int) (g % 40) - 20) * (1.0f + 0.37f * (float) (g % 7));
            for (int i = 0; i < 128; ++i) x[(size_t) (g * 128 + i)] = sc * nd(rng);
        }
        for (int i = 0; i < 128; ++i) x[(size_t) (5 * 128 + i)] = 0.0f;   // an all-zero group
        float* dx = dev(x);
        ernie_fq_e4m3_w(dx, groups, nullptr);
        const auto y = host(dx, x.size());
        int64_t bad = 0;
        for (int64_t g = 0; g < groups; ++g) {
            float am = 0;
            for (int i = 0; i < 128; ++i) am = std::max(am, std::fabs(x[(size_t) (g * 128 + i)]));
            const float s = std::exp2(std::ceil(std::log2(std::max(am, 1e-10f) / 448.0f)));
            for (int i = 0; i < 128; ++i) {
                const float v = x[(size_t) (g * 128 + i)];
                const float want = e4m3_round(std::min(std::max(v / s, -448.0f), 448.0f)) * s;
                if (std::memcmp(&want, &y[(size_t) (g * 128 + i)], 4) != 0) ++bad;
            }
        }
        snprintf(msg, sizeof msg, "%lld of %lld values differ (scales 2^-20 .. 2^19, a zero group)", (long long) bad, (long long) x.size());
        check(bad == 0, "E4M3 rounding, bit for bit", msg);
        cudaFree(dx);
    }

    // ---- the DSA cells and the selection's window exclusion
    {
        const int K = 2048, W = 128, T = 6;
        const int64_t positions[] = {0, 100, 127, 128, 2175, 2176, 2177, 5000};
        bool ok = true, sel_ok = true;
        for (int64_t p0 : positions) {
            const int max_rows = 6000;
            std::vector<float> score((size_t) T * max_rows);
            for (auto& v : score) v = nd(rng);
            for (int t = 0; t < T; ++t)   // ties at the cut: equal scores, the lower rows first
                for (int j = 0; j < 40 && j < max_rows; ++j) score[(size_t) t * max_rows + j * 3] = 0.25f;
            float* ds = dev(score);
            int32_t *dsel = dev(std::vector<int32_t>((size_t) T * K)), *dcells = dev(std::vector<int32_t>((size_t) T * (K + W))),
                    *dcnt = dev(std::vector<int32_t>((size_t) T));
            dsv4_lid_select(ds, max_rows, K, dsel, nullptr, p0, T, nullptr, nullptr, 1, W);
            ernie_dsa_cells_w(dsel, K, W, dcells, K + W, dcnt, p0, T, nullptr);
            const auto sel = host(dsel, (size_t) T * K);
            const auto cells = host(dcells, (size_t) T * (K + W));
            const auto cnt = host(dcnt, (size_t) T);
            for (int t = 0; t < T; ++t) {
                const int64_t p = p0 + t;
                const int64_t nc = std::max<int64_t>(p + 1 - W, 0);
                std::vector<int> want;
                if (nc <= K) {
                    for (int j = 0; j < nc; ++j) want.push_back(j);
                } else {   // the top K by score, ties to the lower row, ascending
                    std::vector<int> ord((size_t) nc);
                    std::iota(ord.begin(), ord.end(), 0);
                    std::stable_sort(ord.begin(), ord.end(),
                                     [&](int a, int b) { return score[(size_t) t * max_rows + a] > score[(size_t) t * max_rows + b]; });
                    want.assign(ord.begin(), ord.begin() + K);
                    std::sort(want.begin(), want.end());
                }
                for (int j = 0; j < K; ++j)
                    if (sel[(size_t) t * K + j] != (j < (int) want.size() ? want[(size_t) j] : -1)) sel_ok = false;
                const int64_t w0 = std::max<int64_t>(0, p - W + 1);
                const int n_win = (int) (p - w0 + 1);
                if (cnt[(size_t) t] != (int) want.size() + n_win) ok = false;
                for (size_t j = 0; j < want.size(); ++j)
                    if (cells[(size_t) t * (K + W) + j] != want[j]) ok = false;
                for (int j = 0; j < n_win; ++j)
                    if (cells[(size_t) t * (K + W) + want.size() + (size_t) j] != (int32_t) (w0 + j)) ok = false;
            }
            cudaFree(ds);
            cudaFree(dsel);
            cudaFree(dcells);
            cudaFree(dcnt);
        }
        check(sel_ok, "the indexer's top-2048 before the window (excl 128), positions 0 .. 5005");
        check(ok, "the DSA cells: the choice, then the window; the counts");
    }

    // ---- mHC's flat-stream factor, eps outside the root
    {
        const int rows = 4, n = 16384;
        const float eps = 1e-6f;
        std::vector<float> x((size_t) rows * n);
        for (auto& v : x) v = 3.0f * nd(rng);
        float *dx = dev(x), *dy = dev(std::vector<float>(x.size()));
        ernie_rms_eo_w(dx, dy, rows, n, eps, nullptr);
        const auto y = host(dy, x.size());
        double worst = 0;
        for (int r = 0; r < rows; ++r) {
            double ss = 0;
            for (int i = 0; i < n; ++i) ss += (double) x[(size_t) (r * n + i)] * x[(size_t) (r * n + i)];
            const double f = 1.0 / (std::sqrt(ss / n) + eps);
            for (int i = 0; i < n; ++i)
                worst = std::max(worst, std::fabs(x[(size_t) (r * n + i)] * f - y[(size_t) (r * n + i)]) / (std::fabs(x[(size_t) (r * n + i)] * f) + 1e-6));
        }
        snprintf(msg, sizeof msg, "worst relative %.2e", worst);
        check(worst < 1e-5, "mHC flat factor 1 / (sqrt(mean) + eps)", msg);
        cudaFree(dx);
        cudaFree(dy);
    }

    // ---- SiTU-GLU through swiglu_clamp's sentinel, and the split / in-place forms
    {
        const int n = 100000;
        std::vector<float> g(n), u(n);
        for (int i = 0; i < n; ++i) {
            g[(size_t) i] = 6.0f * nd(rng);
            u[(size_t) i] = 40.0f * nd(rng);
        }
        float *dg = dev(g), *du = dev(u), *dout = dev(std::vector<float>(n));
        swiglu_clamp(dg, du, dout, n, strata::kernels::kSituGlu, nullptr);
        const auto y = host(dout, n);
        double worst = 0;
        for (int i = 0; i < n; ++i) {
            const double gd = g[(size_t) i], ud = u[(size_t) i];
            const double want = 4.0 * std::tanh(gd / 4.0) / (1.0 + std::exp(-gd)) * 25.0 * std::tanh(ud / 25.0);
            worst = std::max(worst, std::fabs(want - y[(size_t) i]) / (std::fabs(want) + 1e-6));
        }
        snprintf(msg, sizeof msg, "worst relative %.2e against double", worst);
        check(worst < 2e-6, "SiTU-GLU (swiglu_clamp at kSituGlu)", msg);
        // the MMQ forms: [gate | up] rows into h, and in place over the gate half
        const int rows = 3, nf = 2048;
        std::vector<float> gu((size_t) rows * 2 * nf);
        for (auto& v : gu) v = 5.0f * nd(rng);
        float *dgu = dev(gu), *dh = dev(std::vector<float>((size_t) rows * nf));
        swiglu_clamp_split(dgu, dh, rows, nf, strata::kernels::kSituGlu, nullptr);
        const auto h = host(dh, (size_t) rows * nf);
        swiglu_clamp_inplace(dgu, rows, nf, strata::kernels::kSituGlu, nullptr);
        const auto gi = host(dgu, gu.size());
        bool same = true;
        double w2 = 0;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < nf; ++c) {
                const float a = h[(size_t) (r * nf + c)], b = gi[(size_t) (r * 2 * nf + c)];
                if (std::memcmp(&a, &b, 4) != 0) same = false;
                const float want = strata::kernels::situ_glu(gu[(size_t) (r * 2 * nf + c)], gu[(size_t) (r * 2 * nf + nf + c)]);
                w2 = std::max(w2, (double) std::fabs(want - a) / (std::fabs(want) + 1e-6));
            }
        check(same, "SiTU-GLU: the split and in-place MMQ forms agree bit for bit");
        snprintf(msg, sizeof msg, "worst relative %.2e against the host's situ_glu", w2);
        check(w2 < 2e-6, "SiTU-GLU: GPU against the CPU pool's formula", msg);
        // the default path untouched: limit 0 is plain SwiGLU
        swiglu_clamp(dg, du, dout, n, 0.0f, nullptr);
        const auto y0 = host(dout, n);
        int64_t bad = 0;
        for (int i = 0; i < n; ++i) {
            volatile float gv = g[(size_t) i];
            volatile float den = 1.0f + std::exp(-gv);
            volatile float sl = gv / den;
            const float want = sl * u[(size_t) i];
            if (std::fabs(want - y0[(size_t) i]) > 1e-6f * (std::fabs(want) + 1.0f)) ++bad;
        }
        check(bad == 0, "SwiGLU at limit 0 unchanged");
        cudaFree(dg);
        cudaFree(du);
        cudaFree(dout);
        cudaFree(dgu);
        cudaFree(dh);
    }

    // ---- the index keys' F16 rows
    {
        const int T = 3, width = 128;
        const int64_t rows = 50, p0 = 47;   // positions 47 .. 49 (the last one in range)
        std::vector<float> x((size_t) T * width);
        for (auto& v : x) v = nd(rng);
        uint16_t* dc = dev(std::vector<uint16_t>((size_t) rows * width, 0));
        float* dx = dev(x);
        ernie_store_rows_w(dx, dc, width, rows, p0, T, nullptr);
        const auto c = host(dc, (size_t) rows * width);
        bool ok = true;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < width; ++i) {
                uint32_t bits = 0;
                const uint16_t hv = c[(size_t) ((p0 + t) * width + i)];
                const int e = (hv >> 10) & 31, m = hv & 1023;
                float f = e == 0 ? std::ldexp((float) m, -24) : std::ldexp(1.0f + (float) m / 1024.0f, e - 15);
                if (hv & 0x8000) f = -f;
                (void) bits;
                if (std::fabs(f - x[(size_t) (t * width + i)]) > std::fabs(x[(size_t) (t * width + i)]) * 1e-3f) ok = false;
            }
        check(ok, "the index keys into F16 rows at their positions");
        cudaFree(dc);
        cudaFree(dx);
    }

    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("FAIL  CUDA: %s\n", cudaGetErrorString(e));
        ++g_fail;
    }
    printf(g_fail ? "%d FAILED\n" : "ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
