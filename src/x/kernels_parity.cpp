// src/x/kernels_parity.cpp - moex: every GLM decode kernel against the CPU arithmetic of moex/ref/glm_ref.cpp
// (which matches llama.cpp op by op), on random inputs.  GPU, synthetic, no model.
//
// The references below are transcriptions of glm_ref.cpp's functions, not calls into it: this test must build
// without ggml.  Tolerances are relative L2 over each output; integer outputs (expert ids, cells) must agree as sets.
#include "strata/x/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/moe_mmq.hpp"

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
    void up(const std::vector<T>& v) { CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> down(size_t m = 0) const {
        std::vector<T> v(m ? m : n);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(v.data(), p, v.size() * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

std::mt19937 rng(20261007);
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
    printf("%-34s rel %.3e  tol %.0e  %s\n", what, r, tol, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

void report_bool(const char* what, bool ok, const std::string& detail = "") {
    printf("%-34s %s %s\n", what, ok ? "PASS" : "FAIL", detail.c_str());
    if (!ok) ++g_fail;
}

float sig(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// ---------------- references (glm_ref.cpp)

void ref_rms(const float* x, const float* w, float* y, int n, float eps) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double) x[i] * x[i];
    const float scale = 1.0f / std::sqrt((float) (s / n) + eps);
    for (int i = 0; i < n; ++i) y[i] = w ? x[i] * scale * w[i] : x[i] * scale;
}

void ref_mhc_coef(const float* mixes, const float* sc, const float* bs, float* coef, float eps, int iters) {
    for (int h = 0; h < 4; ++h) {
        coef[h] = sig(mixes[h] * sc[0] + bs[h]) + eps;
        coef[4 + h] = sig(mixes[4 + h] * sc[1] + bs[4 + h]) * 2.0f;
    }
    float c[16];
    for (int src = 0; src < 4; ++src) {
        float mx = -INFINITY;
        for (int dst = 0; dst < 4; ++dst) {
            c[dst + 4 * src] = mixes[8 + dst + 4 * src] * sc[2] + bs[8 + dst + 4 * src];
            mx = std::max(mx, c[dst + 4 * src]);
        }
        float sum = 0;
        for (int dst = 0; dst < 4; ++dst) sum += (c[dst + 4 * src] = std::exp(c[dst + 4 * src] - mx));
        for (int dst = 0; dst < 4; ++dst) c[dst + 4 * src] = c[dst + 4 * src] / sum + eps;
    }
    auto cols = [&]() {
        for (int dst = 0; dst < 4; ++dst) {
            float s = eps;
            for (int src = 0; src < 4; ++src) s += c[dst + 4 * src];
            for (int src = 0; src < 4; ++src) c[dst + 4 * src] /= s;
        }
    };
    auto rows = [&]() {
        for (int src = 0; src < 4; ++src) {
            float s = eps;
            for (int dst = 0; dst < 4; ++dst) s += c[dst + 4 * src];
            for (int dst = 0; dst < 4; ++dst) c[dst + 4 * src] /= s;
        }
    };
    cols();
    for (int i = 1; i < iters; ++i) {
        rows();
        cols();
    }
    for (int i = 0; i < 16; ++i) coef[8 + i] = c[i];
}

// ---------------- tests

void test_generic() {
    {   // rms_norm: per head (64 x 128, weighted), the mHC flat norm (16384, unweighted), the layer norm (4096)
        for (auto [rows, cols, weighted] : {std::tuple<int, int, bool>{64, 128, true}, {1, 16384, false}, {1, 4096, true}}) {
            auto x = randn((size_t) rows * cols, 0.3f);
            auto w = randn(cols, 0.5f, 1.0f);
            std::vector<float> ref(x.size());
            for (int r = 0; r < rows; ++r) ref_rms(&x[(size_t) r * cols], weighted ? w.data() : nullptr, &ref[(size_t) r * cols], cols, 1e-5f);
            Dev<float> dx(x), dw(w), dy(x.size());
            g::rms_norm(dx.p, weighted ? dw.p : nullptr, dy.p, rows, cols, 1e-5f, nullptr);
            char name[64];
            snprintf(name, sizeof name, "rms_norm %dx%d%s", rows, cols, weighted ? " w" : "");
            report(name, rel(dy.down(), ref), 1e-6);
        }
    }
    {   // layer_norm over the indexer key (128), eps 1e-6
        auto x = randn(128, 2.0f, 0.5f), w = randn(128, 0.3f, 1.0f), b = randn(128, 0.1f);
        double s = 0;
        for (float v : x) s += v;
        const float mean = (float) (s / 128);
        double s2 = 0;
        for (float v : x) s2 += (double) (v - mean) * (v - mean);
        const float scale = 1.0f / std::sqrt((float) (s2 / 128) + 1e-6f);
        std::vector<float> ref(128);
        for (int i = 0; i < 128; ++i) ref[i] = (x[i] - mean) * scale * w[i] + b[i];
        Dev<float> dx(x), dw(w), db(b), dy(128);
        g::layer_norm(dx.p, dw.p, db.p, dy.p, 128, 1e-6f, nullptr);
        report("layer_norm 128", rel(dy.down(), ref), 1e-6);
    }
    {   // the router's F32 GEMV, 4096 -> 288
        const int n_in = 4096, n_out = 288;
        auto W = randn((size_t) n_in * n_out, 0.02f), x = randn(n_in);
        std::vector<float> ref(n_out);
        for (int r = 0; r < n_out; ++r) {
            double s = 0;
            for (int i = 0; i < n_in; ++i) s += (double) W[(size_t) r * n_in + i] * x[i];
            ref[r] = (float) s;
        }
        Dev<float> dW(W), dx(x), dy(n_out);
        g::gemv_f32(dW.p, dx.p, dy.p, n_in, n_out, nullptr);
        report("gemv_f32 4096->288", rel(dy.down(), ref), 1e-5);
    }
    {   // swiglu_clamp with values on both sides of the limit
        const int n = 12288;
        auto gt = randn(n, 8.0f), ut = randn(n, 8.0f);
        std::vector<float> ref(n);
        for (int i = 0; i < n; ++i) {
            const float gg = std::min(gt[i], 10.0f), uu = std::clamp(ut[i], -10.0f, 10.0f);
            ref[i] = gg / (1.0f + std::exp(-gg)) * uu;
        }
        Dev<float> dg(gt), du(ut), dy(n);
        g::swiglu_clamp(dg.p, du.p, dy.p, n, 10.0f, nullptr);
        report("swiglu_clamp L=10", rel(dy.down(), ref), 1e-6);
    }
}

void test_mhc() {
    const int n = 4096;
    auto mixes = randn(24, 60.0f), sc = randn(3, 0.01f, 0.02f), bs = randn(24, 1.0f);
    std::vector<float> coef(24);
    ref_mhc_coef(mixes.data(), sc.data(), bs.data(), coef.data(), 1e-6f, 20);
    Dev<float> dm(mixes), ds(sc), db(bs), dc(24);
    g::mhc_coef(dm.p, ds.p, db.p, dc.p, 1e-6f, 20, nullptr);
    report("mhc_coef (sinkhorn 20)", rel(dc.down(), coef), 1e-6);

    auto R = randn(4 * (size_t) n, 0.05f), x = randn(n, 0.1f);
    std::vector<float> pre(n), post(4 * (size_t) n), mean(n);
    for (int i = 0; i < n; ++i) {
        float s = 0;
        for (int h = 0; h < 4; ++h) s += R[(size_t) h * n + i] * coef[h];
        pre[i] = s;
        for (int dst = 0; dst < 4; ++dst) {
            float t = x[i] * coef[4 + dst];
            for (int src = 0; src < 4; ++src) t += R[(size_t) src * n + i] * coef[8 + dst + 4 * src];
            post[(size_t) dst * n + i] = t;
        }
    }
    for (int i = 0; i < n; ++i)
        mean[i] = (((post[i] + post[(size_t) n + i]) + post[(size_t) 2 * n + i]) + post[(size_t) 3 * n + i]) * 0.25f;
    Dev<float> dR(R), dx(x), dpre(n), dmean(n);
    dc.up(coef);   // the reference coefficients, so pre/post are tested alone
    g::mhc_pre(dR.p, dc.p, dpre.p, n, nullptr);
    report("mhc_pre", rel(dpre.down(), pre), 1e-6);
    g::mhc_post(dx.p, dR.p, dc.p, n, nullptr);
    report("mhc_post (in place)", rel(dR.down(), post), 1e-6);
    g::mhc_mean(dR.p, dmean.p, n, nullptr);
    report("mhc_mean", rel(dmean.down(), mean), 1e-6);
}

void test_kda() {
    const int H = 64, S = 128, C = H * S, dc = 4;
    {   // conv + silu over three tokens, so the slide is exercised
        auto w = randn((size_t) C * dc, 0.3f);
        std::vector<float> state((size_t) C * (dc - 1), 0.0f);
        Dev<float> dstate(state), dw(w), dx(C), dout(C);
        double worst = 0;
        for (int step = 0; step < 3; ++step) {
            auto x = randn(C);
            std::vector<float> ref(C);
            for (int c = 0; c < C; ++c) {
                float in[4] = {state[(size_t) c * 3], state[(size_t) c * 3 + 1], state[(size_t) c * 3 + 2], x[c]};
                float acc = 0;
                for (int t = 0; t < 4; ++t) acc += in[t] * w[(size_t) c * 4 + t];
                for (int t = 0; t < 3; ++t) state[(size_t) c * 3 + t] = in[t + 1];
                ref[c] = acc / (1.0f + std::exp(-acc));
            }
            dx.up(x);
            g::kda_conv_silu(dstate.p, dx.p, dw.p, dout.p, C, dc, nullptr);
            worst = std::max(worst, rel(dout.down(), ref));
        }
        report("kda_conv_silu x3", worst, 1e-6);
        report("kda_conv state", rel(dstate.down(), state), 0.0);
    }
    {   // gates: lower-bound path (-5) and beta
        auto graw = randn(C, 2.0f), dt = randn(C, 0.5f), A = randn(H, 1.0f, -3.0f), beta = randn(H, 2.0f);
        std::vector<float> rg(C), rb(H);
        for (int i = 0; i < C; ++i) rg[i] = sig(((graw[i] + dt[i]) * A[i / S]) * -1.0f) * -5.0f;
        for (int h = 0; h < H; ++h) rb[h] = sig(beta[h]);
        Dev<float> dg(graw), ddt(dt), dA(A), db(beta);
        g::kda_gates(dg.p, ddt.p, dA.p, -5.0f, db.p, H, S, nullptr);
        report("kda_gates g", rel(dg.down(), rg), 1e-6);
        report("kda_gates beta", rel(db.down(), rb), 1e-6);
    }
    {   // l2 norm with the q scale
        auto x = randn(C, 0.4f);
        std::vector<float> ref(C);
        for (int h = 0; h < H; ++h) {
            double s = 0;
            for (int d = 0; d < S; ++d) s += (double) x[h * S + d] * x[h * S + d];
            const float inv = 1.0f / std::sqrt((float) s + 1e-6f) / std::sqrt(128.0f);
            for (int d = 0; d < S; ++d) ref[h * S + d] = x[h * S + d] * inv;
        }
        Dev<float> dx(x);
        g::kda_l2norm(dx.p, H, S, 1e-6f, 1.0f / std::sqrt(128.0f), nullptr);
        report("kda_l2norm (q scale)", rel(dx.down(), ref), 1e-6);
    }
    {   // the delta rule over five tokens; state in the kernel's (i, h, j) layout, reference in ggml's M[h][j][i]
        std::vector<float> M((size_t) H * S * S, 0.0f), st((size_t) S * H * S, 0.0f);
        Dev<float> dst(st), dq(C), dk(C), dv(C), dg(C), db(H), dout(C);
        double worst = 0;
        for (int step = 0; step < 5; ++step) {
            auto q = randn(C, 0.0884f), k = randn(C, 0.0884f), v = randn(C, 0.5f), gg = randn(C, 0.5f, -1.0f);
            auto beta = randn(H, 0.2f, 0.5f);
            std::vector<float> o(C);
            for (int h = 0; h < H; ++h) {
                float* m = &M[(size_t) h * S * S];
                for (int j = 0; j < S; ++j)
                    for (int i = 0; i < S; ++i) m[j * S + i] *= std::exp(gg[h * S + i]);
                std::vector<float> d(S);
                for (int j = 0; j < S; ++j) {
                    float s = 0;
                    for (int i = 0; i < S; ++i) s += m[j * S + i] * k[h * S + i];
                    d[j] = (v[h * S + j] - s) * beta[h];
                }
                for (int j = 0; j < S; ++j)
                    for (int i = 0; i < S; ++i) m[j * S + i] = std::fma(k[h * S + i], d[j], m[j * S + i]);
                for (int j = 0; j < S; ++j) {
                    float s = 0;
                    for (int i = 0; i < S; ++i) s += m[j * S + i] * q[h * S + i];
                    o[h * S + j] = s;
                }
            }
            dq.up(q);
            dk.up(k);
            dv.up(v);
            dg.up(gg);
            db.up(beta);
            g::kda_step(dst.p, dq.p, dk.p, dv.p, dg.p, db.p, dout.p, H, S, nullptr);
            worst = std::max(worst, rel(dout.down(), o));
        }
        report("kda_step x5 (output)", worst, 1e-5);
        auto sd = dst.down();
        std::vector<float> st_ref(sd.size());
        for (int h = 0; h < H; ++h)
            for (int j = 0; j < S; ++j)
                for (int i = 0; i < S; ++i) st_ref[(size_t) i * H * S + (size_t) h * S + j] = M[(size_t) h * S * S + j * S + i];
        report("kda_step x5 (state)", rel(sd, st_ref), 1e-5);
    }
    {   // output norm and gate
        auto o = randn(C, 0.02f), g2 = randn(C, 2.0f), w = randn(S, 0.3f, 1.0f);
        std::vector<float> ref(C);
        for (int h = 0; h < H; ++h) {
            ref_rms(&o[h * S], w.data(), &ref[h * S], S, 1e-5f);
            for (int d = 0; d < S; ++d) ref[h * S + d] *= sig(g2[h * S + d]);
        }
        Dev<float> d_o(o), dg2(g2), dw(w), dy(C);
        g::kda_out_gate(d_o.p, dg2.p, dw.p, dy.p, H, S, 1e-5f, nullptr);
        report("kda_out_gate", rel(dy.down(), ref), 1e-6);
    }
}

void test_dsa() {
    const int L = 512, H = 64, D = 128, IH = 32, KP = 4;
    {   // the latent store: FP16 round-to-nearest
        auto kv = randn(L, 1.0f);
        Dev<float> dkv(kv);
        Dev<uint16_t> cache((size_t) 8 * L);
        g::dsa_store_latent(dkv.p, cache.p, 5, L, nullptr);
        auto c = cache.down();
        bool ok = true;
        for (int i = 0; i < L; ++i) {
            const __half h = __float2half_rn(kv[i]);
            uint16_t bits;
            memcpy(&bits, &h, 2);
            ok &= c[(size_t) 5 * L + i] == bits;
        }
        report_bool("dsa_store_latent (fp16 rn)", ok);
    }
    {   // pooling
        auto ik = randn((size_t) KP * D, 1.5f), ig = randn((size_t) KP * D, 1.0f), ape = randn((size_t) D * KP, 0.5f);
        std::vector<float> ref(D);
        for (int d = 0; d < D; ++d) {
            float lg[4], mx = -INFINITY, s = 0;
            for (int k = 0; k < KP; ++k) mx = std::max(mx, lg[k] = ig[k * D + d] + ape[d + D * k]);
            for (int k = 0; k < KP; ++k) s += (lg[k] = std::exp(lg[k] - mx));
            double acc = 0;
            for (int k = 0; k < KP; ++k) acc += (double) (lg[k] / s) * ik[k * D + d];
            ref[d] = (float) acc;
        }
        Dev<float> dik(ik), dig(ig), dape(ape), dp(D);
        g::dsa_kpool(dik.p, dig.p, dape.p, dp.p, D, KP, nullptr);
        report("dsa_kpool", rel(dp.down(), ref), 1e-6);
    }
    const int n_pool = 3000;
    auto iq = randn((size_t) IH * D, 3.0f), pooled = randn((size_t) n_pool * D, 1.0f), w = randn(IH, 0.05f);
    std::vector<float> sref(n_pool);
    for (int P = 0; P < n_pool; ++P) {
        float s = 0;
        for (int h = 0; h < IH; ++h) {
            double dot = 0;
            for (int d = 0; d < D; ++d) dot += (double) iq[h * D + d] * pooled[(size_t) P * D + d];
            s += std::max((float) dot, 0.0f) * w[h];
        }
        sref[P] = s;
    }
    Dev<float> diq(iq), dpool(pooled), dw(w), dscore(n_pool);
    g::dsa_scores(diq.p, dpool.p, dw.p, dscore.p, n_pool, IH, D, nullptr);
    auto sgpu = dscore.down();
    report("dsa_scores 3000 pools", rel(sgpu, sref), 1e-5);
    {   // selection: 3000 pools -> the top 512 (as a set, against a CPU sort of the GPU's own scores) + tail
        Dev<int32_t> cells(4096);
        const int64_t pos = 3000 * 4 + 2;   // 2 cells into the next pool: n_tail = 3
        const int n = g::dsa_select(dscore.p, n_pool, 512, KP, pos, true, cells.p, nullptr, nullptr);
        auto c = cells.down(n);
        std::vector<int> order(n_pool);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return sgpu[a] > sgpu[b]; });
        std::set<int> want, got(c.begin(), c.end());
        for (int i = 0; i < 512; ++i)
            for (int k = 0; k < KP; ++k) want.insert(order[i] * KP + k);
        for (int t = 0; t < 3; ++t) want.insert((int) (pos - t));
        report_bool("dsa_select top-512 + tail", n == 512 * 4 + 3 && want == got, "n=" + std::to_string(n));
        // fewer pools than the budget: every pool, then the tail
        const int n2 = g::dsa_select(dscore.p, 100, 512, KP, 401, true, cells.p, nullptr, nullptr);
        auto c2 = cells.down(n2);
        std::set<int> got2(c2.begin(), c2.end()), want2;
        for (int i = 0; i < 400; ++i) want2.insert(i);
        want2.insert(401);
        want2.insert(400);
        report_bool("dsa_select all pools + tail", n2 == 402 && got2 == want2, "n=" + std::to_string(n2));
    }
    {   // the sparse latent attention, 2051 random distinct cells of a 6000-cell FP16 cache
        const int n_ctx = 6000, n_cells = 2051;
        auto kvf = randn((size_t) n_ctx * L, 1.0f);
        std::vector<uint16_t> kvh(kvf.size());
        for (size_t i = 0; i < kvf.size(); ++i) {
            const __half h = __float2half_rn(kvf[i]);
            memcpy(&kvh[i], &h, 2);
            kvf[i] = __half2float(h);   // the reference reads what the cache holds
        }
        std::vector<int32_t> all(n_ctx);
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        std::vector<int32_t> cl(all.begin(), all.begin() + n_cells);
        auto q = randn((size_t) H * L, 0.3f);
        const float scale = 1.0f / 16.0f;
        std::vector<float> ref((size_t) H * L);
        for (int h = 0; h < H; ++h) {
            std::vector<double> p(n_cells);
            double mx = -INFINITY;
            for (int s = 0; s < n_cells; ++s) {
                double dot = 0;
                for (int i = 0; i < L; ++i) dot += (double) q[(size_t) h * L + i] * kvf[(size_t) cl[s] * L + i];
                p[s] = dot * scale;
                mx = std::max(mx, p[s]);
            }
            double sum = 0;
            for (auto& v : p) sum += (v = std::exp(v - mx));
            for (int i = 0; i < L; ++i) {
                double a = 0;
                for (int s = 0; s < n_cells; ++s) a += p[s] * kvf[(size_t) cl[s] * L + i];
                ref[(size_t) h * L + i] = (float) (a / sum);
            }
        }
        Dev<float> dq(q), dctx((size_t) H * L);
        Dev<uint16_t> dkv(kvh);
        Dev<int32_t> dcells(cl);
        Dev<float> part(g::dsa_attend_scratch_bytes(H, L, n_cells) / sizeof(float));
        g::dsa_attend(dq.p, dkv.p, dcells.p, n_cells, scale, dctx.p, part.p, H, L, nullptr);
        report("dsa_attend 64 heads x 2051 cells", rel(dctx.down(), ref), 1e-5);
    }
}

void test_moe() {
    const int E = 288, k = 8;
    auto logits = randn(E, 2.0f), bias = randn(E, 0.5f, 7.5f);
    std::vector<float> prob(E), sel(E);
    for (int e = 0; e < E; ++e) sel[e] = (prob[e] = sig(logits[e])) + bias[e];
    std::vector<int> order(E);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sel[a] > sel[b]; });
    std::vector<float> wref(k);
    float s = 0;
    for (int j = 0; j < k; ++j) s += (wref[j] = prob[order[j]]);
    for (auto& v : wref) v = v / std::max(s, 6.103515625e-5f) * 2.5f;
    Dev<float> dl(logits), db(bias), dw(k);
    Dev<int32_t> di(k);
    g::moe_route(dl.p, db.p, E, k, true, 2.5f, di.p, dw.p, nullptr);
    auto ids = di.down();
    report_bool("moe_route ids (ordered)", std::equal(ids.begin(), ids.end(), order.begin()));
    report("moe_route weights", rel(dw.down(), wref), 1e-6);

    const int n = 4096;
    auto parts = randn((size_t) k * n, 0.3f), shared = randn(n, 0.2f);
    std::vector<float> ref(n);
    for (int i = 0; i < n; ++i) {
        float t = parts[i] * wref[0];
        for (int j = 1; j < k; ++j) t = t + parts[(size_t) j * n + i] * wref[j];
        ref[i] = t + shared[i];
    }
    Dev<float> dp(parts), dsh(shared), dout(n);
    dw.up(wref);
    g::moe_combine(dp.p, dw.p, dsh.p, dout.p, n, k, nullptr);
    report("moe_combine", rel(dout.down(), ref), 1e-6);
}


// ---------------- windows: T tokens at once against T sequential single-token calls (or their references)

void test_windows() {
    const int T = 5;
    {   // mHC
        const int n = 4096;
        auto mixes = randn((size_t) T * 24, 60.0f), sc = randn(3, 0.01f, 0.02f), bs = randn(24, 1.0f);
        Dev<float> dm(mixes), ds(sc), db(bs), dc((size_t) T * 24), dc1(24);
        g::mhc_coef_w(dm.p, ds.p, db.p, dc.p, 1e-6f, 20, T, nullptr);
        std::vector<float> want;
        for (int t = 0; t < T; ++t) {
            std::vector<float> c(24);
            ref_mhc_coef(&mixes[(size_t) t * 24], sc.data(), bs.data(), c.data(), 1e-6f, 20);
            want.insert(want.end(), c.begin(), c.end());
        }
        auto coef = dc.down();
        report("mhc_coef_w x5", rel(coef, want), 1e-6);
        auto R = randn((size_t) T * 4 * n, 0.05f), x = randn((size_t) T * n, 0.1f);
        std::vector<float> pre((size_t) T * n), post(R), mean((size_t) T * n);
        for (int t = 0; t < T; ++t) {
            const float* cf = &coef[(size_t) t * 24];
            for (int i = 0; i < n; ++i) {
                float a = 0;
                for (int hh = 0; hh < 4; ++hh) a += R[((size_t) t * 4 + hh) * n + i] * cf[hh];
                pre[(size_t) t * n + i] = a;
                for (int dst = 0; dst < 4; ++dst) {
                    float v = x[(size_t) t * n + i] * cf[4 + dst];
                    for (int src = 0; src < 4; ++src) v += R[((size_t) t * 4 + src) * n + i] * cf[8 + dst + 4 * src];
                    post[((size_t) t * 4 + dst) * n + i] = v;
                }
            }
            for (int i = 0; i < n; ++i) {
                const float* r = &post[(size_t) t * 4 * n];
                mean[(size_t) t * n + i] = (((r[i] + r[n + i]) + r[2 * n + i]) + r[3 * n + i]) * 0.25f;
            }
        }
        Dev<float> dR(R), dx(x), dpre((size_t) T * n), dmean((size_t) T * n);
        g::mhc_pre_w(dR.p, dc.p, dpre.p, n, T, nullptr);
        report("mhc_pre_w x5", rel(dpre.down(), pre), 1e-6);
        g::mhc_post_w(dx.p, dR.p, dc.p, n, T, nullptr);
        report("mhc_post_w x5", rel(dR.down(), post), 1e-6);
        g::mhc_mean_w(dR.p, dmean.p, n, T, nullptr);
        report("mhc_mean_w x5", rel(dmean.down(), mean), 1e-6);
    }
    const int H = 64, S = 128, C = H * S, dc = 4;
    {   // conv over a window = T single-token convs; and the rollback
        auto w = randn((size_t) C * dc, 0.3f), st0 = randn((size_t) C * (dc - 1), 0.5f), x = randn((size_t) T * C);
        Dev<float> sa(st0), sb(st0), dw(w), dx(x), oa((size_t) T * C), ob(C);
        g::kda_conv_silu_w(sa.p, dx.p, dw.p, oa.p, C, dc, T, nullptr);
        std::vector<float> seq;
        Dev<float> x1(C);
        std::vector<std::vector<float>> states;
        for (int t = 0; t < T; ++t) {
            x1.up(std::vector<float>(x.begin() + (size_t) t * C, x.begin() + (size_t) (t + 1) * C));
            g::kda_conv_silu(sb.p, x1.p, dw.p, ob.p, C, dc, nullptr);
            auto o1 = ob.down();
            seq.insert(seq.end(), o1.begin(), o1.end());
            states.push_back(sb.down());
        }
        report("kda_conv_silu_w x5", rel(oa.down(), seq), 0.0);
        report("kda_conv_silu_w state", rel(sa.down(), states.back()), 0.0);
        Dev<float> prev(st0), rb((size_t) C * (dc - 1));
        g::kda_conv_rollback(prev.p, dx.p, rb.p, C, dc, 2, nullptr);
        report("kda_conv_rollback (keep 2)", rel(rb.down(), states[1]), 0.0);
    }
    {   // the scan = T steps; snapshots = the state after each step
        auto q = randn((size_t) T * C, 0.0884f), k = randn((size_t) T * C, 0.0884f), v = randn((size_t) T * C, 0.5f);
        auto gg = randn((size_t) T * C, 0.5f, -1.0f), beta = randn((size_t) T * H, 0.2f, 0.5f);
        auto st0 = randn((size_t) S * H * S, 0.05f);
        Dev<float> sa(st0), sb(st0), dq(q), dk(k), dv(v), dg(gg), dbeta(beta), oa((size_t) T * C);
        Dev<float> snaps((size_t) T * S * H * S);
        g::kda_scan(sa.p, dq.p, dk.p, dv.p, dg.p, dbeta.p, oa.p, H, S, T, snaps.p, nullptr);
        Dev<float> q1(C), k1(C), v1(C), g1(C), b1(H), o1(C);
        std::vector<float> seq, snap_ref;
        for (int t = 0; t < T; ++t) {
            auto sl = [&](const std::vector<float>& a, size_t w_) {
                return std::vector<float>(a.begin() + (size_t) t * w_, a.begin() + (size_t) (t + 1) * w_);
            };
            q1.up(sl(q, C));
            k1.up(sl(k, C));
            v1.up(sl(v, C));
            g1.up(sl(gg, C));
            b1.up(sl(beta, H));
            g::kda_step(sb.p, q1.p, k1.p, v1.p, g1.p, b1.p, o1.p, H, S, nullptr);
            auto ov = o1.down();
            seq.insert(seq.end(), ov.begin(), ov.end());
            auto sv = sb.down();
            snap_ref.insert(snap_ref.end(), sv.begin(), sv.end());
        }
        report("kda_scan x5 (output)", rel(oa.down(), seq), 0.0);
        report("kda_scan x5 (state)", rel(sa.down(), std::vector<float>(snap_ref.end() - (size_t) S * H * S, snap_ref.end())), 0.0);
        report("kda_scan x5 (snapshots)", rel(snaps.down(), snap_ref), 0.0);
    }
    {   // DSA: latents, the pool ring, the per-query scores, selection and attention
        const int L = 512, D = 128, IH = 32, KP = 4, ring = 12, n_ctx = 3000, max_pool = n_ctx / KP + 1;
        const int max_cells = 2048 + KP, n_top = 512;
        const int64_t pos0 = 2600;   // past the budget: queries 2600..2604 select 512 of ~650 pools
        // a cache and pooled keys for every earlier position, then the window's own
        auto kvf = randn((size_t) n_ctx * L, 1.0f);
        std::vector<uint16_t> kvh(kvf.size());
        for (size_t i = 0; i < kvf.size(); ++i) {
            const __half hv = __float2half_rn(kvf[i]);
            memcpy(&kvh[i], &hv, 2);
        }
        Dev<uint16_t> cache(kvh);
        auto pooled = randn((size_t) max_pool * D, 1.0f);
        Dev<float> dpool(pooled);
        // the ring holds the open pool's members before pos0 (2600 % 4 = 0: none) and the window's
        auto ikw = randn((size_t) T * D, 1.5f), igw = randn((size_t) T * D, 1.0f), ape = randn((size_t) D * KP, 0.5f);
        Dev<float> rik((size_t) ring * D), rig((size_t) ring * D), dik(ikw), dig(igw), dape(ape);
        g::dsa_ring_store_w(dik.p, rik.p, ring, pos0, D, T, nullptr);
        g::dsa_ring_store_w(dig.p, rig.p, ring, pos0, D, T, nullptr);
        g::dsa_kpool_w(rik.p, rig.p, ring, dape.p, dpool.p, pos0, T, D, KP, nullptr);   // pool 650 completes at 2603
        {
            Dev<float> one(D), ik4((size_t) KP * D), ig4((size_t) KP * D);
            ik4.up(std::vector<float>(ikw.begin(), ikw.begin() + KP * D));
            ig4.up(std::vector<float>(igw.begin(), igw.begin() + KP * D));
            g::dsa_kpool(ik4.p, ig4.p, dape.p, one.p, D, KP, nullptr);
            auto pw = dpool.down();
            report("dsa_kpool_w (pool 650)", rel(std::vector<float>(pw.begin() + (size_t) 650 * D, pw.begin() + (size_t) 651 * D),
                                               one.down()), 0.0);
        }
        auto iq = randn((size_t) T * IH * D, 3.0f), w = randn((size_t) T * IH, 0.05f);
        Dev<float> diq(iq), dw(w), score((size_t) T * max_pool);
        g::dsa_scores_w(diq.p, dpool.p, dw.p, score.p, pos0, T, KP, IH, D, max_pool, nullptr);
        auto sw = score.down();
        Dev<int32_t> cells((size_t) T * max_cells);
        g::dsa_select_w(score.p, max_pool, n_top, KP, pos0, T, true, cells.p, max_cells, nullptr);
        auto cw = cells.down();
        auto q = randn((size_t) T * 64 * L, 0.3f);
        Dev<float> dq(q), ctx((size_t) T * 64 * L);
        Dev<float> part(g::dsa_attend_w_scratch_bytes(64, L, max_cells, T) / sizeof(float));
        g::dsa_attend_w(dq.p, cache.p, cells.p, max_cells, pos0, T, n_top, KP, true, 1.0f / 16.0f, ctx.p, part.p, 64, L, nullptr);
        auto cx = ctx.down();
        double worst_s = 0, worst_c = 0;
        bool sel_ok = true;
        for (int t = 0; t < T; ++t) {
            const int64_t pos = pos0 + t;
            const int n_pool = (int) ((pos + 1) / KP);
            Dev<float> iq1(std::vector<float>(iq.begin() + (size_t) t * IH * D, iq.begin() + (size_t) (t + 1) * IH * D));
            Dev<float> w1(std::vector<float>(w.begin() + (size_t) t * IH, w.begin() + (size_t) (t + 1) * IH));
            Dev<float> s1(n_pool);
            g::dsa_scores(iq1.p, dpool.p, w1.p, s1.p, n_pool, IH, D, nullptr);
            auto sv = s1.down();
            worst_s = std::max(worst_s, rel(std::vector<float>(sw.begin() + (size_t) t * max_pool,
                                                               sw.begin() + (size_t) t * max_pool + n_pool), sv));
            Dev<int32_t> c1(max_cells);
            const int nc = g::dsa_select(s1.p, n_pool, n_top, KP, pos, true, c1.p, nullptr, nullptr);
            auto cv1 = c1.down(nc);
            std::set<int> a(cv1.begin(), cv1.end()),
                b(cw.begin() + (size_t) t * max_cells, cw.begin() + (size_t) t * max_cells + g::dsa_cells(pos, n_top, KP, true));
            sel_ok &= nc == g::dsa_cells(pos, n_top, KP, true) && a == b;
            Dev<float> q1(std::vector<float>(q.begin() + (size_t) t * 64 * L, q.begin() + (size_t) (t + 1) * 64 * L)), c1o((size_t) 64 * L);
            Dev<float> p1(g::dsa_attend_scratch_bytes(64, L, max_cells) / sizeof(float));
            g::dsa_attend(q1.p, cache.p, c1.p, nc, 1.0f / 16.0f, c1o.p, p1.p, 64, L, nullptr);
            worst_c = std::max(worst_c, rel(std::vector<float>(cx.begin() + (size_t) t * 64 * L, cx.begin() + (size_t) (t + 1) * 64 * L),
                                            c1o.down()));
        }
        report("dsa_scores_w x5 (past the budget)", worst_s, 0.0);
        report_bool("dsa_select_w x5 = single", sel_ok);
        report("dsa_attend_w x5", worst_c, 1e-6);
    }
    {   // routing, combine, the F32 GEMV, arg-max, the permute
        const int E = 288, k = 8, n = 4096;
        auto logits = randn((size_t) T * E, 2.0f), bias = randn(E, 0.5f, 7.5f);
        Dev<float> dl(logits), db(bias), dw((size_t) T * k);
        Dev<int32_t> di((size_t) T * k);
        g::moe_route_w(dl.p, db.p, E, k, true, 2.5f, di.p, dw.p, T, nullptr);
        auto iw = di.down();
        auto ww = dw.down();
        bool ok = true;
        double worst = 0;
        for (int t = 0; t < T; ++t) {
            Dev<float> l1(std::vector<float>(logits.begin() + (size_t) t * E, logits.begin() + (size_t) (t + 1) * E)), w1(k);
            Dev<int32_t> i1(k);
            g::moe_route(l1.p, db.p, E, k, true, 2.5f, i1.p, w1.p, nullptr);
            auto iv = i1.down();
            ok &= std::equal(iv.begin(), iv.end(), iw.begin() + (size_t) t * k);
            worst = std::max(worst, rel(std::vector<float>(ww.begin() + (size_t) t * k, ww.begin() + (size_t) (t + 1) * k), w1.down()));
        }
        report_bool("moe_route_w x5 ids", ok);
        report("moe_route_w x5 weights", worst, 0.0);
        auto parts = randn((size_t) T * k * n, 0.3f), shared = randn((size_t) T * n, 0.2f);
        Dev<float> dp(parts), dsh(shared), dout((size_t) T * n);
        g::moe_combine_w(dp.p, dw.p, dsh.p, dout.p, n, k, T, nullptr);
        auto cw = dout.down();
        worst = 0;
        for (int t = 0; t < T; ++t) {
            Dev<float> p1(std::vector<float>(parts.begin() + (size_t) t * k * n, parts.begin() + (size_t) (t + 1) * k * n));
            Dev<float> w1(std::vector<float>(ww.begin() + (size_t) t * k, ww.begin() + (size_t) (t + 1) * k));
            Dev<float> s1(std::vector<float>(shared.begin() + (size_t) t * n, shared.begin() + (size_t) (t + 1) * n)), o1(n);
            g::moe_combine(p1.p, w1.p, s1.p, o1.p, n, k, nullptr);
            worst = std::max(worst, rel(std::vector<float>(cw.begin() + (size_t) t * n, cw.begin() + (size_t) (t + 1) * n), o1.down()));
        }
        report("moe_combine_w x5", worst, 0.0);
        auto W = randn((size_t) n * E, 0.02f), x = randn((size_t) T * n);
        Dev<float> dW(W), dx(x), dy((size_t) T * E);
        g::gemv_f32_w(dW.p, dx.p, dy.p, n, E, T, nullptr);
        auto yv = dy.down();
        std::vector<float> yref((size_t) T * E);
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < E; ++r) {
                double a = 0;
                for (int i = 0; i < n; ++i) a += (double) W[(size_t) r * n + i] * x[(size_t) t * n + i];
                yref[(size_t) t * E + r] = (float) a;
            }
        report("gemv_f32_w x5", rel(yv, yref), 1e-5);
        Dev<int32_t> am(T);
        g::argmax_w(dy.p, E, am.p, T, nullptr);
        auto av = am.down();
        bool aok = true;
        for (int t = 0; t < T; ++t)
            aok &= av[t] == (int) (std::max_element(yv.begin() + (size_t) t * E, yv.begin() + (size_t) (t + 1) * E) -
                                   (yv.begin() + (size_t) t * E));
        report_bool("argmax_w x5", aok);
        auto src = randn((size_t) T * 64 * 256);
        Dev<float> ds(src), dd(src.size());
        g::permute_abc(ds.p, dd.p, T, 64, 256, nullptr);
        auto dv = dd.down();
        bool pok = true;
        for (int a = 0; a < T && pok; ++a)
            for (int b = 0; b < 64 && pok; ++b)
                for (int c = 0; c < 256; ++c)
                    if (dv[((size_t) b * T + a) * 256 + c] != src[((size_t) a * 64 + b) * 256 + c]) {
                        pok = false;
                        break;
                    }
        report_bool("permute_abc", pok);
    }
}
// the prompt path's tensor-core DSA attention against the split-K kernels, at the three regimes of a prompt: the first
// positions (a handful of cells, a partial chunk), under the budget (all pools + the tail) and past it (512 pools).
void test_dsa_mma() {
    if (!g::dsa_attend_mma_supported()) {
        printf("dsa_attend_mma: skipped (no sm_80 tensor cores here, or STRATA_X_ATTN_MMA=0)\n");
        return;
    }
    const int L = 512, D = 128, IH = 32, KP = 4, n_ctx = 3000, max_pool = n_ctx / KP + 1, n_top = 512;
    const int max_cells = 2048 + KP, H = 64, T = 40;
    for (const int64_t pos0 : {(int64_t) 0, (int64_t) 90, (int64_t) 2600}) {
        auto kvf = randn((size_t) n_ctx * L, 1.0f);
        std::vector<uint16_t> kvh(kvf.size());
        for (size_t i = 0; i < kvf.size(); ++i) {
            const __half hv = __float2half_rn(kvf[i]);
            memcpy(&kvh[i], &hv, 2);
        }
        Dev<uint16_t> cache(kvh);
        auto pooled = randn((size_t) max_pool * D, 1.0f);
        Dev<float> dpool(pooled);
        auto iq = randn((size_t) T * IH * D, 3.0f), w = randn((size_t) T * IH, 0.05f);
        Dev<float> diq(iq), dw(w), score((size_t) T * max_pool);
        g::dsa_scores_w(diq.p, dpool.p, dw.p, score.p, pos0, T, KP, IH, D, max_pool, nullptr);
        Dev<int32_t> cells((size_t) T * max_cells);
        g::dsa_select_w(score.p, max_pool, n_top, KP, pos0, T, true, cells.p, max_cells, nullptr);
        // queries with a wide dynamic range (the softmax must stay stable) and a small one
        for (const float qs : {0.3f, 4.0f}) {
            auto q = randn((size_t) T * H * L, qs);
            Dev<float> dq(q), ref((size_t) T * H * L), got((size_t) T * H * L);
            const int Tb = 8;   // the split-K scratch for T queries at a time
            Dev<float> part(g::dsa_attend_w_scratch_bytes(H, L, max_cells, Tb) / sizeof(float));
            for (int t0 = 0; t0 < T; t0 += Tb)
                g::dsa_attend_w(dq.p + (size_t) t0 * H * L, cache.p, cells.p + (size_t) t0 * max_cells, max_cells, pos0 + t0,
                                std::min(Tb, T - t0), n_top, KP, true, 1.0f / 16.0f, ref.p + (size_t) t0 * H * L, part.p, H, L, nullptr);
            const bool ok = g::dsa_attend_mma(dq.p, cache.p, cells.p, max_cells, pos0, T, n_top, KP, true, 1.0f / 16.0f, got.p, H, L, nullptr);
            char name[96];
            snprintf(name, sizeof name, "dsa_attend_mma vs split-K (pos0 %lld, |q| %.1f)", (long long) pos0, qs);
            if (!ok) {
                report_bool(name, false);
                continue;
            }
            report(name, rel(got.down(), ref.down()), 2e-5);
        }
    }
}

// The prompt path's tiled scores kernel against the warp kernel decode windows run: BIT-IDENTICAL, every visible pool of
// every query, and nothing written past a query's pools (both outputs start from the same sentinel).  Edge cases: T not
// a multiple of the 4-token tile, positions inside a pool, early queries with no complete pool, depths past 64-pool tiles.
void test_dsa_tiled() {
    const int D = 128, IH = 32, KP = 4;
    for (const auto& c : std::vector<std::pair<int64_t, int>>{{0, 37}, {1, 9}, {2, 512}, {3, 509}, {130, 64}, {2049, 100},
                                                               {10240, 512}, {40961, 211}}) {
        const int64_t pos0 = c.first;
        const int T = c.second;
        const int max_pool = (int) ((pos0 + T) / KP + 1);
        Dev<float> dpool(randn((size_t) max_pool * D, 1.0f)), diq(randn((size_t) T * IH * D, 3.0f)),
            dw(randn((size_t) T * IH, 0.05f)), a((size_t) T * max_pool), b((size_t) T * max_pool);
        CK(cudaMemset(a.p, 0x7f, a.n * sizeof(float)));
        CK(cudaMemset(b.p, 0x7f, b.n * sizeof(float)));
        g::dsa_scores_warp(diq.p, dpool.p, dw.p, a.p, pos0, T, KP, max_pool, nullptr);
        g::dsa_scores_tiled(diq.p, dpool.p, dw.p, b.p, pos0, T, KP, max_pool, nullptr);
        const auto ha = a.down(), hb = b.down();
        const bool same = memcmp(ha.data(), hb.data(), ha.size() * sizeof(float)) == 0;
        size_t diff = 0;
        double worst = 0;   // the largest difference, in units of the warp kernel's last place
        for (size_t i = 0; i < ha.size(); ++i)
            if (memcmp(&ha[i], &hb[i], sizeof(float)) != 0) {
                ++diff;
                const float ulp = std::nextafter(std::fabs(ha[i]), INFINITY) - std::fabs(ha[i]);
                worst = std::max(worst, (double) std::fabs(ha[i] - hb[i]) / std::max(ulp, 1e-30f));
            }
        char name[96];
        snprintf(name, sizeof name, "dsa_scores tiled == warp, bitwise (pos0 %lld, T %d)", (long long) pos0, T);
        report_bool(name, same, std::to_string(diff) + " of " + std::to_string(ha.size()) + " floats differ, worst " +
                                   std::to_string(worst) + " ulp");
    }
}

// STRATA_X_PARITY_ONLY=dsa_bench: the prompt path's DSA kernels on one 512-token sub-batch (EngineOptions::prefill_sub)
// at growing depth - the indexer's scores, the top-512 selection, the sparse latent attention - and what one 8K chunk
// pays for them (16 sub-batches x 11 DSA layers).  Timing only (random data, no reference).
void bench_dsa() {
    const int L = 512, D = 128, IH = 32, KP = 4, H = 64, T = 512, n_top = 512, max_cells = 2048 + KP;
    const bool mma = g::dsa_attend_mma_supported();
    for (const int64_t depth : {(int64_t) 10240, (int64_t) 51200, (int64_t) 102400, (int64_t) 204800}) {
        const int64_t n_ctx = depth + T;
        const int max_pool = (int) (n_ctx / KP + 1);
        Dev<uint16_t> cache((size_t) n_ctx * L);
        CK(cudaMemset(cache.p, 0x2c, cache.n * sizeof(uint16_t)));   // small FP16 values
        Dev<float> dpool(randn((size_t) max_pool * D, 1.0f));
        Dev<float> diq(randn((size_t) T * IH * D, 3.0f)), dw(randn((size_t) T * IH, 0.05f)), score((size_t) T * max_pool);
        Dev<int32_t> cells((size_t) T * max_cells);
        Dev<float> dq(randn((size_t) T * H * L, 0.3f)), out((size_t) T * H * L);
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
        const float t_warp = best_of([&] { g::dsa_scores_warp(diq.p, dpool.p, dw.p, score.p, depth, T, KP, max_pool, nullptr); });
        const float t_sc = best_of([&] { g::dsa_scores_w(diq.p, dpool.p, dw.p, score.p, depth, T, KP, IH, D, max_pool, nullptr); });
        printf("   dsa depth %6lldK: scores warp kernel %8.2f ms, tiled %8.2f ms (%.1fx)\n", (long long) (depth / 1024), t_warp,
               t_sc, t_warp / t_sc);
        const float t_sel = best_of([&] { g::dsa_select_w(score.p, max_pool, n_top, KP, depth, T, true, cells.p, max_cells, nullptr); });
        const float t_att = mma ? best_of([&] {
            g::dsa_attend_mma(dq.p, cache.p, cells.p, max_cells, depth, T, n_top, KP, true, 1.0f / 16.0f, out.p, H, L, nullptr);
        }) : -1.0f;
        const double fma = (double) T * (double) (depth / KP) * IH * D;   // the scores' multiply-adds (pools at ~depth)
        printf("   dsa depth %6lldK, 512 queries: scores %8.2f ms (%5.0f GFMA/s), select %7.2f ms, attend %6.2f ms"
               " -> an 8K chunk's 11 layers: %6.2f s\n", (long long) (depth / 1024), t_sc, fma / (t_sc * 1e6), t_sel, t_att,
               (t_sc + t_sel + std::max(0.0f, t_att)) * 16 * 11 / 1000.0);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
}

// THE GRAPH PATH: the DSA window kernels with the position in DEVICE memory and static launch grids (what a captured
// graph replays at every position) against the same kernels with the position as an argument: bit for bit.
void test_dsa_static_positions() {
    const int L = 512, D = 128, IH = 32, KP = 4, ring = 12, n_ctx = 3000, max_pool = n_ctx / KP + 1, n_top = 512, H = 64;
    const int max_cells = 2048 + KP;
    for (const int T : {1, 2, 8}) {
        for (const int64_t pos0 : {(int64_t) 0, (int64_t) 2, (int64_t) 5, (int64_t) 98, (int64_t) 2599, (int64_t) 2600, (int64_t) 2602}) {
            auto kvf = randn((size_t) n_ctx * L, 1.0f);
            std::vector<uint16_t> kvh(kvf.size());
            for (size_t i = 0; i < kvf.size(); ++i) {
                const __half hv = __float2half_rn(kvf[i]);
                memcpy(&kvh[i], &hv, 2);
            }
            auto pooled0 = randn((size_t) max_pool * D, 1.0f);
            auto ikw = randn((size_t) T * D, 1.5f), igw = randn((size_t) T * D, 1.0f), ape = randn((size_t) D * KP, 0.5f);
            auto ring0 = randn((size_t) ring * D, 1.0f), ring1 = randn((size_t) ring * D, 1.0f);
            auto iq = randn((size_t) T * IH * D, 3.0f), w = randn((size_t) T * IH, 0.05f), q = randn((size_t) T * H * L, 0.3f);
            auto kvnew = randn((size_t) T * L, 1.0f);
            auto run = [&](bool dev) {
                Dev<uint16_t> cache(kvh);
                Dev<float> dpool(pooled0), rik(ring0), rig(ring1), dik(ikw), dig(igw), dape(ape), diq(iq), dw(w), dq(q), dkv(kvnew);
                Dev<float> score((size_t) T * max_pool), ctx((size_t) T * H * L);
                Dev<int32_t> cells((size_t) T * max_cells);
                Dev<float> part(g::dsa_attend_w_scratch_bytes(H, L, max_cells, T) / sizeof(float));
                for (auto* z : {score.p, ctx.p, part.p}) (void) z;
                cudaMemset(score.p, 0, score.n * sizeof(float));
                cudaMemset(ctx.p, 0, ctx.n * sizeof(float));
                cudaMemset(part.p, 0, part.n * sizeof(float));
                cudaMemset(cells.p, 0, cells.n * sizeof(int32_t));
                Dev<int64_t> dpos(std::vector<int64_t>{pos0});
                const int64_t* pd = dev ? dpos.p : nullptr;
                g::dsa_store_latent_w(dkv.p, cache.p, pos0, L, T, nullptr, pd);
                g::dsa_ring_store_w(dik.p, rik.p, ring, pos0, D, T, nullptr, pd);
                g::dsa_ring_store_w(dig.p, rig.p, ring, pos0, D, T, nullptr, pd);
                g::dsa_kpool_w(rik.p, rig.p, ring, dape.p, dpool.p, pos0, T, D, KP, nullptr, pd);
                g::dsa_scores_w(diq.p, dpool.p, dw.p, score.p, pos0, T, KP, IH, D, max_pool, nullptr, pd);
                g::dsa_select_w(score.p, max_pool, n_top, KP, pos0, T, true, cells.p, max_cells, nullptr, pd);
                g::dsa_attend_w(dq.p, cache.p, cells.p, max_cells, pos0, T, n_top, KP, true, 1.0f / 16.0f, ctx.p, part.p, H, L, nullptr, pd);
                std::vector<float> out = ctx.down();
                auto add = [&](const std::vector<float>& v) { out.insert(out.end(), v.begin(), v.end()); };
                add(dpool.down());
                add(rik.down());
                add(rig.down());
                add(score.down());
                for (int32_t c : cells.down()) out.push_back((float) c);
                for (uint16_t c : cache.down()) out.push_back((float) c);
                return out;
            };
            const auto a = run(false), b = run(true);
            char name[96];
            snprintf(name, sizeof name, "dsa window, device position == argument (T %d, pos0 %lld)", T, (long long) pos0);
            report_bool(name, a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
        }
    }
}

// MiMo's GQA against a double-precision reference (K and V rounded to FP16 as the cache holds them; the sink in the
// denominator): the decode windows' split-K kernel and the prompt path's FP32 tiles (gqa_attend_w, STRATA_X_GQA_MMA=0 -
// main() sets it) and its tensor-core kernel (gqa_prefill_mma, 4 and 8 warps), at the full layers (16 query heads per KV
// head, every earlier position) and the sliding-window ones (8 per KV head, 128 positions, a 640-row ring that wraps);
// partial last tiles (T not a multiple of the block's tokens), and a tensor-core timing at depth.
void test_gqa() {
    const int H = 64, DK = 192, DV = 128;
    const float scale = 1.0f / std::sqrt((float) DK);
    struct Case {
        int n_kv, window;
        int64_t rows, pos0;
        int T;
        float qs;
    };
    const Case cases[] = {{4, 0, 4096, 0, 40, 1.0f},     {4, 0, 4096, 1000, 37, 1.0f}, {4, 0, 4096, 3000, 64, 4.0f},
                          {4, 0, 4096, 2000, 5, 1.0f},   {8, 128, 640, 0, 40, 1.0f},   {8, 128, 640, 100, 33, 1.0f},
                          {8, 128, 640, 2000, 64, 4.0f}, {8, 128, 640, 2000, 5, 1.0f}};
    for (const Case& c : cases) {
        const int nk = c.n_kv * DK, nv = c.n_kv * DV, G = H / c.n_kv;
        const int q_stride = H * DK + nk + nv;   // the fused QKV row the engine passes
        auto kf = randn((size_t) c.rows * nk, 1.0f), vf = randn((size_t) c.rows * nv, 1.0f);
        std::vector<uint16_t> kh(kf.size()), vh(vf.size());
        for (size_t i = 0; i < kf.size(); ++i) {
            const __half x = __float2half_rn(kf[i]);
            memcpy(&kh[i], &x, 2);
            kf[i] = __half2float(x);
        }
        for (size_t i = 0; i < vf.size(); ++i) {
            const __half x = __float2half_rn(vf[i]);
            memcpy(&vh[i], &x, 2);
            vf[i] = __half2float(x);
        }
        auto q = randn((size_t) c.T * q_stride, c.qs), sinks = randn(H, 2.0f);
        auto ref = [&](bool with_sink) {
            std::vector<float> out((size_t) c.T * H * DV);
            std::vector<double> sc, acc(DV);
            for (int t = 0; t < c.T; ++t) {
                const int64_t p = c.pos0 + t, lo = c.window > 0 ? std::max<int64_t>(0, p - c.window + 1) : 0;
                for (int h = 0; h < H; ++h) {
                    const int g = h / G;
                    const float* qh = &q[(size_t) t * q_stride + (size_t) h * DK];
                    sc.assign((size_t) (p - lo + 1), 0.0);
                    double M = with_sink ? (double) sinks[h] : -INFINITY;
                    for (int64_t j = lo; j <= p; ++j) {
                        const float* kr = &kf[(size_t) (j % c.rows) * nk + (size_t) g * DK];
                        double d = 0;
                        for (int i = 0; i < DK; ++i) d += (double) qh[i] * kr[i];
                        sc[(size_t) (j - lo)] = d * scale;
                        M = std::max(M, sc[(size_t) (j - lo)]);
                    }
                    double L = with_sink ? std::exp((double) sinks[h] - M) : 0.0;
                    std::fill(acc.begin(), acc.end(), 0.0);
                    for (int64_t j = lo; j <= p; ++j) {
                        const double e = std::exp(sc[(size_t) (j - lo)] - M);
                        L += e;
                        const float* vr = &vf[(size_t) (j % c.rows) * nv + (size_t) g * DV];
                        for (int i = 0; i < DV; ++i) acc[i] += e * vr[i];
                    }
                    for (int i = 0; i < DV; ++i) out[((size_t) t * H + h) * DV + i] = (float) (acc[i] / L);
                }
            }
            return out;
        };
        const std::vector<float> want = ref(true), want_ns = ref(false);
        Dev<uint16_t> dk(kh), dv(vh);
        Dev<float> dq(q), ds(sinks), o1((size_t) c.T * H * DV), o2(o1.n), o4(o1.n);
        const int n_chunks = c.T <= 8 ? (c.window > 0 ? 4 : 32) : 1;
        Dev<float> part(g::gqa_attend_scratch_bytes(H, DV, n_chunks, c.T) / sizeof(float));
        g::gqa_attend_w(dq.p, q_stride, dk.p, dv.p, c.rows, H, c.n_kv, DK, DV, c.window, ds.p, scale, o1.p, part.p, n_chunks, c.pos0,
                        c.T, nullptr);
        char name[128];
        snprintf(name, sizeof name, "gqa %s G=%d pos %lld T=%d |q| %.0f FP32 %s", c.window ? "swa" : "full", G, (long long) c.pos0, c.T,
                 c.qs, c.T <= 8 ? "split-K" : "tiles");
        const std::vector<float> got1 = o1.down();
        report(name, rel(got1, want), 2e-6);
        const bool m8 = g::gqa_prefill_mma(dq.p, q_stride, dk.p, dv.p, c.rows, H, c.n_kv, DK, DV, c.window, ds.p, scale, o2.p, c.pos0,
                                           c.T, nullptr, 8);
        const bool m4 = g::gqa_prefill_mma(dq.p, q_stride, dk.p, dv.p, c.rows, H, c.n_kv, DK, DV, c.window, ds.p, scale, o4.p, c.pos0,
                                           c.T, nullptr, 4);
        if (!m8 || !m4) {
            printf("gqa_prefill_mma: skipped (no sm_80 tensor cores here)\n");
            continue;
        }
        const std::vector<float> got8 = o2.down(), got4 = o4.down();
        snprintf(name, sizeof name, "gqa %s G=%d pos %lld T=%d |q| %.0f tensor cores (8 warps)", c.window ? "swa" : "full", G,
                 (long long) c.pos0, c.T, c.qs);
        report(name, rel(got8, want), 5e-6);
        snprintf(name, sizeof name, "gqa %s G=%d pos %lld T=%d |q| %.0f tensor cores (4 warps)", c.window ? "swa" : "full", G,
                 (long long) c.pos0, c.T, c.qs);
        report(name, rel(got4, want), 5e-6);
        snprintf(name, sizeof name, "gqa %s G=%d pos %lld T=%d tensor cores vs FP32", c.window ? "swa" : "full", G, (long long) c.pos0,
                 c.T);
        report(name, rel(got8, got1), 5e-6);
        const double neg = rel(got8, want_ns);
        snprintf(name, sizeof name, "gqa %s G=%d pos %lld T=%d vs no sink (must fail)", c.window ? "swa" : "full", G, (long long) c.pos0,
                 c.T);
        report_bool(name, neg > 5e-6, "rel " + std::to_string(neg));
    }
    // timing: one sub-batch of 512 queries at depth, the FP32 tiles vs the tensor cores
    for (const auto& tc : {std::make_pair(4, 0), std::make_pair(8, 128)}) {
        const int n_kv = tc.first, window = tc.second, T = 512, nk = n_kv * DK, nv = n_kv * DV, q_stride = H * DK + nk + nv;
        const int64_t rows = window ? 640 : 66048;
        std::vector<uint16_t> kh((size_t) rows * nk), vh((size_t) rows * nv);
        for (size_t i = 0; i < kh.size(); ++i) kh[i] = (uint16_t) (0x3000 + (i * 2654435761u >> 22) % 0x0c00);
        for (size_t i = 0; i < vh.size(); ++i) vh[i] = (uint16_t) (0x3000 + (i * 2246822519u >> 22) % 0x0c00);
        Dev<uint16_t> dk(kh), dv(vh);
        auto q = randn((size_t) T * q_stride, 1.0f), sinks = randn(H, 2.0f);
        Dev<float> dq(q), ds(sinks), o1((size_t) T * H * DV), o2(o1.n);
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        for (const int64_t pos0 : {(int64_t) 7680, (int64_t) 65024}) {
            if (window && pos0 > 7680) continue;
            auto time = [&](auto fn) {
                fn();
                CK(cudaEventRecord(a));
                for (int i = 0; i < 10; ++i) fn();
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, a, b));
                return ms / 10;
            };
            const float t32 = time([&] {
                g::gqa_attend_w(dq.p, q_stride, dk.p, dv.p, rows, H, n_kv, DK, DV, window, ds.p, scale, o1.p, nullptr, 1, pos0, T, nullptr);
            });
            const float t8 = time([&] {
                g::gqa_prefill_mma(dq.p, q_stride, dk.p, dv.p, rows, H, n_kv, DK, DV, window, ds.p, scale, o2.p, pos0, T, nullptr, 8);
            });
            const float t4 = time([&] {
                g::gqa_prefill_mma(dq.p, q_stride, dk.p, dv.p, rows, H, n_kv, DK, DV, window, ds.p, scale, o2.p, pos0, T, nullptr, 4);
            });
            char name[128];
            snprintf(name, sizeof name, "gqa %s T=512 pos %lld: tensor cores vs FP32", window ? "swa" : "full", (long long) pos0);
            report(name, rel(o2.down(), o1.down()), 5e-6);
            printf("   gqa %s T=512 pos %lld: FP32 tiles %.3f ms, tensor cores %.3f ms (8 warps) %.3f ms (4 warps)\n", window ? "swa" : "full",
                   (long long) pos0, t32, t8, t4);
        }
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
}

/// The decode windows' full layers on tensor cores (gqa_decode_mma + gqa_merge_w) against float64 at depth, and against the
/// FP32 split-K kernel; then the time per window at 131K keys, the FP32 kernel vs the tensor cores by key splits.
void test_gqa_decode() {
    const int H = 64, DK = 192, DV = 128, n_kv = 4, G = H / n_kv, nk = n_kv * DK, nv = n_kv * DV, q_stride = H * DK + nk + nv;
    const float scale = 1.0f / std::sqrt((float) DK);
    const int64_t rows = 135168;
    auto kf = randn((size_t) rows * nk, 1.0f), vf = randn((size_t) rows * nv, 1.0f);
    std::vector<uint16_t> kh(kf.size()), vh(vf.size());
    for (size_t i = 0; i < kf.size(); ++i) {
        const __half x = __float2half_rn(kf[i]);
        memcpy(&kh[i], &x, 2);
        kf[i] = __half2float(x);
    }
    for (size_t i = 0; i < vf.size(); ++i) {
        const __half x = __float2half_rn(vf[i]);
        memcpy(&vh[i], &x, 2);
        vf[i] = __half2float(x);
    }
    Dev<uint16_t> dk(kh), dv(vh);
    auto sinks = randn(H, 2.0f);
    Dev<float> ds(sinks);
    struct Case {
        int64_t pos0;
        int T;
        float qs;
        int n_chunks;
    };
    const Case cases[] = {{37, 1, 1.0f, 32}, {2000, 5, 1.0f, 32}, {65000, 1, 1.0f, 128}, {131000, 1, 1.0f, 128}, {131000, 4, 4.0f, 128},
                          {131072, 8, 1.0f, 64}};
    for (const Case& c : cases) {
        auto q = randn((size_t) c.T * q_stride, c.qs);
        // float64 reference (with the sink), every token and head
        std::vector<double> want((size_t) c.T * H * DV, 0.0), sc;
        for (int t = 0; t < c.T; ++t)
            for (int h = 0; h < H; ++h) {
                const int g = h / G;
                const int64_t p = c.pos0 + t;
                const float* qh = &q[(size_t) t * q_stride + (size_t) h * DK];
                sc.assign((size_t) p + 1, 0.0);
                double M = sinks[h];
                for (int64_t j = 0; j <= p; ++j) {
                    const float* kr = &kf[(size_t) j * nk + (size_t) g * DK];
                    double d = 0;
                    for (int i = 0; i < DK; ++i) d += (double) qh[i] * kr[i];
                    sc[(size_t) j] = d * scale;
                    M = std::max(M, sc[(size_t) j]);
                }
                double L = std::exp((double) sinks[h] - M);
                std::vector<double> acc(DV, 0.0);
                for (int64_t j = 0; j <= p; ++j) {
                    const double e = std::exp(sc[(size_t) j] - M);
                    L += e;
                    const float* vr = &vf[(size_t) j * nv + (size_t) g * DV];
                    for (int i = 0; i < DV; ++i) acc[i] += e * vr[i];
                }
                for (int i = 0; i < DV; ++i) want[((size_t) t * H + h) * DV + i] = acc[i] / L;
            }
        std::vector<float> wantf(want.begin(), want.end());
        Dev<float> dq(q), o1((size_t) c.T * H * DV), o2(o1.n);
        Dev<float> part(g::gqa_attend_scratch_bytes(H, DV, c.n_chunks, c.T) / sizeof(float));
        g::gqa_attend_w(dq.p, q_stride, dk.p, dv.p, rows, H, n_kv, DK, DV, 0, ds.p, scale, o1.p, part.p, c.n_chunks, c.pos0, c.T, nullptr);
        const std::vector<float> got32 = o1.down();
        char name[160];
        snprintf(name, sizeof name, "gqa decode full G=16 pos %lld T=%d |q| %.0f chunks %d: FP32 split-K", (long long) c.pos0, c.T, c.qs,
                 c.n_chunks);
        report(name, rel(got32, wantf), c.pos0 > 8192 ? 5e-6 : 2e-6);   // the FP32 kernel's own rounding grows with the keys
        for (const int kt : {16, 32}) {   // the tile sizes: 16 keys (4 warps an SM), 32 (2)
            const bool ok = g::gqa_decode_mma(dq.p, q_stride, dk.p, dv.p, rows, H, n_kv, DK, DV, 0, scale, part.p, c.n_chunks, c.pos0,
                                              c.T, nullptr, nullptr, kt);
            if (!ok) {
                printf("gqa_decode_mma: skipped (no sm_80 tensor cores here)\n");
                return;
            }
            g::gqa_merge_w(part.p, ds.p, o2.p, H, c.n_chunks, c.T, nullptr);
            const std::vector<float> gotm = o2.down();
            snprintf(name, sizeof name, "gqa decode full G=16 pos %lld T=%d |q| %.0f chunks %d: tensor cores, %d-key tiles", (long long) c.pos0,
                     c.T, c.qs, c.n_chunks, kt);
            report(name, rel(gotm, wantf), 5e-6);
            snprintf(name, sizeof name, "gqa decode full G=16 pos %lld T=%d: tensor cores (%d-key tiles) vs FP32", (long long) c.pos0, c.T,
                     kt);
            report(name, rel(gotm, got32), 1e-5);   // each is checked against float64 at 5e-6 above: apart, their errors add
        }
        if (c.qs > 1.0f) continue;   // sharp scores: the sink's share (~4e-6) is below what the check can tell
        g::gqa_merge_w(part.p, nullptr, o2.p, H, c.n_chunks, c.T, nullptr);   // the partials once more without the sink
        const double neg = rel(o2.down(), wantf);
        snprintf(name, sizeof name, "gqa decode pos %lld T=%d vs no sink (must fail)", (long long) c.pos0, c.T);
        report_bool(name, neg > 5e-6, "rel " + std::to_string(neg));
    }
    // timing: a decode window at 131K keys (one full layer), FP32 split-K (32 chunks, the engine's) vs tensor cores by chunks
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (const int T : {1, 4}) {
        auto q = randn((size_t) T * q_stride, 1.0f);
        Dev<float> dq(q), o((size_t) T * H * DV);
        Dev<float> part(g::gqa_attend_scratch_bytes(H, DV, 256, T) / sizeof(float));
        auto time = [&](auto fn) {   // the best of 8 batches of 5 (a GPU shared with a running server is busy at times)
            fn();
            float best = 1e30f;
            for (int r = 0; r < 8; ++r) {
                CK(cudaEventRecord(a));
                for (int i = 0; i < 5; ++i) fn();
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, a, b));
                best = std::min(best, ms / 5);
            }
            return best;
        };
        const int64_t pos0 = 131072;
        const float t32 = time([&] {
            g::gqa_attend_w(dq.p, q_stride, dk.p, dv.p, rows, H, n_kv, DK, DV, 0, ds.p, scale, o.p, part.p, 32, pos0, T, nullptr);
        });
        printf("   gqa decode full T=%d pos %lld: FP32 split-K (32 chunks) %.3f ms\n", T, (long long) pos0, t32);
        for (const int kt : {32, 16}) {
            printf("      tensor cores, %d-key tiles:", kt);
            for (const int ch : {64, 128, 256}) {
                const float tm = time([&] {
                    g::gqa_decode_mma(dq.p, q_stride, dk.p, dv.p, rows, H, n_kv, DK, DV, 0, scale, part.p, ch, pos0, T, nullptr, nullptr, kt);
                    g::gqa_merge_w(part.p, ds.p, o.p, H, ch, T, nullptr);
                });
                printf(" %d chunks %.3f ms;", ch, tm);
            }
            printf("\n");
        }
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
}

/// DFlash's noise block (MiMo's drafter): 128-wide heads, 8 per KV head, a 1024-position window over a ring, every query seeing
/// the batch's rows up to its last (non-causal) and its own window's lower bound - against float64.
void test_gqa_noncausal() {
    const int H = 64, D = 128, n_kv = 8, G = H / n_kv, kvw = n_kv * D, qw = H * D, window = 1024;
    const int64_t rows = 1088;
    const float scale = 1.0f / std::sqrt((float) D);
    auto kf = randn((size_t) rows * kvw, 1.0f), vf = randn((size_t) rows * kvw, 1.0f);
    std::vector<uint16_t> kh(kf.size()), vh(vf.size());
    for (size_t i = 0; i < kf.size(); ++i) {
        const __half x = __float2half_rn(kf[i]);
        memcpy(&kh[i], &x, 2);
        kf[i] = __half2float(x);
        const __half y = __float2half_rn(vf[i]);
        memcpy(&vh[i], &y, 2);
        vf[i] = __half2float(y);
    }
    Dev<uint16_t> dk(kh), dv(vh);
    auto sinks = randn(H, 2.0f);
    Dev<float> ds(sinks);
    struct Case {
        int64_t pos0;
        int T, n_chunks;
        bool noncausal;
    };
    const Case cases[] = {{0, 8, 8, true}, {5, 8, 8, true}, {1000, 8, 8, true}, {1030, 8, 8, true}, {5000, 8, 8, true},
                          {70001, 8, 8, true}, {70001, 8, 1, true}, {4096, 3, 8, true}, {4096, 1, 8, true}, {4096, 8, 8, false}};
    for (const Case& c : cases) {
        auto q = randn((size_t) c.T * qw, 1.0f);
        std::vector<double> want((size_t) c.T * H * D, 0.0);
        for (int t = 0; t < c.T; ++t)
            for (int h = 0; h < H; ++h) {
                const int g = h / G;
                const int64_t p = c.pos0 + t, hi = c.noncausal ? c.pos0 + c.T - 1 : p, lo = std::max<int64_t>(0, p - window + 1);
                const float* qh = &q[(size_t) t * qw + (size_t) h * D];
                std::vector<double> sc((size_t) (hi - lo + 1));
                double M = sinks[h];
                for (int64_t j = lo; j <= hi; ++j) {
                    const float* kr = &kf[(size_t) (j % rows) * kvw + (size_t) g * D];
                    double dd = 0;
                    for (int i = 0; i < D; ++i) dd += (double) qh[i] * kr[i];
                    sc[(size_t) (j - lo)] = dd * scale;
                    M = std::max(M, sc[(size_t) (j - lo)]);
                }
                double L = std::exp((double) sinks[h] - M);
                std::vector<double> acc(D, 0.0);
                for (int64_t j = lo; j <= hi; ++j) {
                    const double e = std::exp(sc[(size_t) (j - lo)] - M);
                    L += e;
                    const float* vr = &vf[(size_t) (j % rows) * kvw + (size_t) g * D];
                    for (int i = 0; i < D; ++i) acc[i] += e * vr[i];
                }
                for (int i = 0; i < D; ++i) want[((size_t) t * H + h) * D + i] = acc[i] / L;
            }
        std::vector<float> wantf(want.begin(), want.end());
        Dev<float> dq(q), o((size_t) c.T * H * D);
        Dev<float> part(g::gqa_attend_scratch_bytes(H, D, std::max(1, c.n_chunks), c.T) / sizeof(float) + 1);
        g::gqa_attend_w(dq.p, qw, dk.p, dv.p, rows, H, n_kv, D, D, window, ds.p, scale, o.p, part.p, c.n_chunks, c.pos0, c.T, nullptr,
                        nullptr, c.noncausal);
        const std::vector<float> got = o.down();
        char name[160];
        snprintf(name, sizeof name, "gqa %s D=128 G=8 window 1024 pos %lld T=%d chunks %d", c.noncausal ? "non-causal" : "causal",
                 (long long) c.pos0, c.T, c.n_chunks);
        report(name, rel(got, wantf), 2e-6);
        if (c.noncausal && c.T > 1) {   // the causal kernel must differ here (the block's later rows)
            g::gqa_attend_w(dq.p, qw, dk.p, dv.p, rows, H, n_kv, D, D, window, ds.p, scale, o.p, part.p, c.n_chunks, c.pos0, c.T, nullptr);
            const double neg = rel(o.down(), wantf);
            snprintf(name, sizeof name, "gqa non-causal pos %lld T=%d vs causal (must fail)", (long long) c.pos0, c.T);
            report_bool(name, neg > 1e-4, "rel " + std::to_string(neg));
        }
    }
}

/// q8_mma_w (the drafter's tensor-core Q8_0 projections) against float64 over the same quantized operands, and MMVQ.
void test_q8_mma() {
    cudaStream_t st = nullptr;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    struct Case {
        int n_in, n_out;
    };
    for (const Case c : {Case{4096, 1024}, Case{4096, 8192}, Case{8192, 4096}, Case{16384, 4096}, Case{20480, 4096}, Case{4096, 1000},
                         Case{4096, 152576}}) {
        const int nb = c.n_in / 32;
        std::vector<uint8_t> w((size_t) nb * c.n_out * 34);
        std::mt19937 r(11);
        std::vector<float> dwv((size_t) nb * c.n_out);
        for (size_t i = 0; i < (size_t) nb * c.n_out; ++i) {
            const __half d = __float2half_rn(0.0005f + 0.01f * (float) (r() % 1000) / 1000.0f);
            memcpy(&w[i * 34], &d, 2);
            dwv[i] = __half2float(d);
            for (int j = 0; j < 32; ++j) w[i * 34 + 2 + j] = (uint8_t) (r() & 0xff);
        }
        Dev<uint8_t> dw(w);
        for (const int nc : {1, 3, 7, 8}) {
            auto x = randn((size_t) nc * c.n_in, 1.0f);
            Dev<float> dx(x), y1((size_t) nc * c.n_out), y2((size_t) nc * c.n_out);
            Dev<uint8_t> xq(strata::kernels::native_q8_1_bytes(c.n_in, nc));
            strata::kernels::native_quantize_q8_1(dx.p, xq.p, c.n_in, nc, st);
            const bool ok = g::q8_mma_w(dw.p, xq.p, y1.p, c.n_in, c.n_out, nc, st);
            if (!ok) {
                printf("q8_mma_w: skipped (no sm_80 int8 tensor cores here)\n");
                cudaStreamDestroy(st);
                return;
            }
            strata::kernels::native_mmvq(8, dw.p, xq.p, y2.p, c.n_in, c.n_out, nc, st);
            CK(cudaStreamSynchronize(st));
            const std::vector<float> got = y1.down(), mv = y2.down();
            const std::vector<uint8_t> xh = xq.down();
            std::vector<float> want((size_t) nc * c.n_out);
            for (int col = 0; col < nc; ++col)
                for (int row = 0; row < c.n_out; ++row) {
                    double acc = 0;
                    for (int kb = 0; kb < nb; ++kb) {
                        const uint8_t* xb = &xh[((size_t) col * nb + kb) * 36];
                        __half hd;
                        memcpy(&hd, xb, 2);
                        const uint8_t* wb = &w[((size_t) row * nb + kb) * 34];
                        int sum = 0;
                        for (int i = 0; i < 32; ++i) sum += (int) (int8_t) wb[2 + i] * (int) (int8_t) xb[4 + i];
                        acc += (double) dwv[(size_t) row * nb + kb] * (double) __half2float(hd) * sum;
                    }
                    want[(size_t) col * c.n_out + row] = (float) acc;
                }
            char name[160];
            snprintf(name, sizeof name, "q8_mma %d->%d cols %d vs float64", c.n_in, c.n_out, nc);
            report(name, rel(got, want), 2e-6);
            snprintf(name, sizeof name, "q8_mma %d->%d cols %d vs MMVQ", c.n_in, c.n_out, nc);
            report(name, rel(got, mv), 4e-6);
            if (c.n_out > 20000) break;   // the head's shape once (its float64 reference is slow)
        }
    }
    cudaStreamDestroy(st);
}

/// DFlash's draft step (MiMo's drafter) on random Q8_0 weights of its shapes: each projection's time by column count, in the
/// exact multi-column layout (every column bitwise a single-column call: what the trunk needs) and llama.cpp's upstream one.
void bench_dflash() {
    struct Shape {
        const char* name;
        int n_in, n_out, reps;   // reps: how many such matrices a draft step reads
    };
    const Shape shapes[] = {{"wq 4096->8192 x5", 4096, 8192, 5},   {"wk/wv 4096->1024 x10", 4096, 1024, 10},
                            {"wo 8192->4096 x5", 8192, 4096, 5},   {"gate/up 4096->16384 x10", 4096, 16384, 10},
                            {"down 16384->4096 x5", 16384, 4096, 5}, {"head 4096->152576 x1", 4096, 152576, 1},
                            {"fc 20480->4096 x1", 20480, 4096, 1}};
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    cudaStream_t st = nullptr;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    double tot[2][9] = {}, mmq_tot[9] = {}, q8m_tot[9] = {};
    for (const Shape& sh : shapes) {
        const size_t blocks = (size_t) sh.n_in / 32 * sh.n_out;
        std::vector<uint8_t> w(blocks * 34);
        std::mt19937 r(7);
        for (size_t i = 0; i < blocks; ++i) {
            const __half d = __float2half_rn(0.001f + 0.01f * (float) (r() % 1000) / 1000.0f);
            memcpy(&w[i * 34], &d, 2);
            for (int j = 0; j < 32; ++j) w[i * 34 + 2 + j] = (uint8_t) (r() & 0xff);
        }
        Dev<uint8_t> dw(w);
        auto x = randn((size_t) 8 * sh.n_in, 1.0f);
        Dev<float> dx(x), dy((size_t) 8 * sh.n_out);
        Dev<uint8_t> xq(strata::kernels::native_q8_1_bytes(sh.n_in, 8));
        strata::kernels::native_quantize_q8_1(dx.p, xq.p, sh.n_in, 8, st);
        CK(cudaStreamSynchronize(st));
        printf("   %-26s", sh.name);
        for (int ex = 1; ex >= 0; --ex) {
            strata::kernels::native_mmvq_set_multi_exact(ex == 1);
            for (const int nc : {1, 4, 7, 8}) {
                auto run = [&] { strata::kernels::native_mmvq(8, dw.p, xq.p, dy.p, sh.n_in, sh.n_out, nc, st); };
                run();
                float best = 1e30f;
                for (int rr = 0; rr < 6; ++rr) {
                    CK(cudaEventRecord(a, st));
                    for (int i = 0; i < 10; ++i) run();
                    CK(cudaEventRecord(b, st));
                    CK(cudaEventSynchronize(b));
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, a, b));
                    best = std::min(best, ms / 10);
                }
                tot[ex][nc] += best * sh.reps;
                if (ex == 1 || nc > 1) printf(" %s%d %.3f ms (%.0f GB/s);", ex ? "exact " : "upstream ", nc, best, blocks * 34 / (best * 1e6));
            }
        }
        for (const int nw : {0, 100, 1, 2, 4, 12, 14, 22}) {   // q8_mma_w (0: its own shape choice; 100: activations via L1)
            for (const int nc : {1, 4, 7, 8}) {
                auto run = [&] { g::q8_mma_w(dw.p, xq.p, dy.p, sh.n_in, sh.n_out, nc, st, nw); };
                run();
                float best = 1e30f;
                for (int rr = 0; rr < 6; ++rr) {
                    CK(cudaEventRecord(a, st));
                    for (int i = 0; i < 10; ++i) run();
                    CK(cudaEventRecord(b, st));
                    CK(cudaEventSynchronize(b));
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, a, b));
                    best = std::min(best, ms / 10);
                }
                if (nw == 0) q8m_tot[nc] += best * sh.reps;
                if (nc == 8 || nc == 1 || nw == 0) printf(" q8mma%s%d c%d %.3f ms (%.0f GB/s);", nw ? "w" : "auto", nw, nc, best, blocks * 34 / (best * 1e6));
            }
        }
        // llama.cpp's MMQ (int8 tensor cores) in its ordinary mode, as Stage::mmq_dense runs it
        if (strata::prefill::mmq::built() && strata::prefill::mmq::fits(8, sh.n_out)) {
            static strata::prefill::mmq::Context ctx;
            Dev<uint8_t> mxq(strata::prefill::mmq::q8_bytes(8, sh.n_in) + 4096);
            for (const int nc : {1, 4, 7, 8}) {
                auto run = [&] {
                    strata::prefill::mmq::quantize(dx.p, nullptr, mxq.p, 8, sh.n_in, sh.n_in, nc, st);
                    strata::prefill::mmq::Product p;
                    p.w = dw.p;
                    p.type = 8;
                    p.w_rows = sh.n_out;
                    p.w_cols = sh.n_in;
                    p.expert_bytes = strata::prefill::mmq::matrix_bytes(8, sh.n_out, sh.n_in);
                    p.n = 1;
                    p.xq = mxq.p;
                    p.total_rows = nc;
                    p.max_rows = nc;
                    p.dst = dy.p;
                    p.ld_dst = sh.n_out;
                    ctx.run(p, st);
                };
                run();
                float best = 1e30f;
                for (int rr = 0; rr < 6; ++rr) {
                    CK(cudaEventRecord(a, st));
                    for (int i = 0; i < 10; ++i) run();
                    CK(cudaEventRecord(b, st));
                    CK(cudaEventSynchronize(b));
                    float ms = 0;
                    CK(cudaEventElapsedTime(&ms, a, b));
                    best = std::min(best, ms / 10);
                }
                mmq_tot[nc] += best * sh.reps;
                printf(" mmq %d %.3f ms (%.0f GB/s);", nc, best, blocks * 34 / (best * 1e6));
            }
        }
        printf("\n");
    }
    printf("   MMQ: 8 cols %.2f ms, 7 cols %.2f, 4 cols %.2f, 1 col %.2f\n", mmq_tot[8], mmq_tot[7], mmq_tot[4], mmq_tot[1]);
    printf("   q8_mma (auto warps): 8 cols %.2f ms, 7 cols %.2f, 4 cols %.2f, 1 col %.2f\n", q8m_tot[8], q8m_tot[7], q8m_tot[4], q8m_tot[1]);
    strata::kernels::native_mmvq_set_multi_exact(true);
    cudaStreamDestroy(st);
    printf("   a draft step's projections (5 blocks + head, fc once): exact 8 cols %.2f ms, 7 cols %.2f; upstream 8 cols %.2f ms, 7 cols %.2f; "
           "exact 4 cols %.2f, 1 col %.2f\n", tot[1][8], tot[1][7], tot[0][8], tot[0][7], tot[1][4], tot[1][1]);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
}

/// gqa_decode_shared (a full layer's decode window, the rows sharing K / V tiles): against float64, and a row's output in a
/// window bitwise its output alone at the same position (what keeps drafting lossless); timing against the per-row kernel.
void test_gqa_shared() {
    const int H = 64, DK = 192, DV = 128, n_kv = 4, nk = n_kv * DK, nv = n_kv * DV, q_stride = H * DK + nk + nv;
    const float scale = 1.0f / std::sqrt((float) DK);
    const int64_t rows = 135168;
    auto kf = randn((size_t) rows * nk, 1.0f), vf = randn((size_t) rows * nv, 1.0f);
    std::vector<uint16_t> kh(kf.size()), vh(vf.size());
    for (size_t i = 0; i < kf.size(); ++i) {
        const __half x = __float2half_rn(kf[i]);
        memcpy(&kh[i], &x, 2);
        kf[i] = __half2float(x);
    }
    for (size_t i = 0; i < vf.size(); ++i) {
        const __half x = __float2half_rn(vf[i]);
        memcpy(&vh[i], &x, 2);
        vf[i] = __half2float(x);
    }
    Dev<uint16_t> dk(kh), dv(vh);
    auto sinks = randn(H, 2.0f);
    Dev<float> ds(sinks);
    const int NC = 64;
    // positions: short, mid, a window straddling a chunk-size doubling (64 x 2048 = 131072 keys), deep
    struct Case {
        int64_t pos0;
        int T;
    };
    for (const Case c : {Case{37, 1}, Case{37, 8}, Case{2000, 5}, Case{65000, 3}, Case{131068, 8}, Case{131072, 4}, Case{134000, 8}}) {
        auto q = randn((size_t) c.T * q_stride, 1.0f);
        std::vector<double> want((size_t) c.T * H * DV, 0.0), sc;
        for (int t = 0; t < c.T; ++t)
            for (int h = 0; h < H; ++h) {
                const int gq = h / 16;
                const int64_t p = c.pos0 + t;
                const float* qh = &q[(size_t) t * q_stride + (size_t) h * DK];
                sc.assign((size_t) p + 1, 0.0);
                double M = sinks[h];
                for (int64_t j = 0; j <= p; ++j) {
                    const float* kr = &kf[(size_t) j * nk + (size_t) gq * DK];
                    double d = 0;
                    for (int i = 0; i < DK; ++i) d += (double) qh[i] * kr[i];
                    sc[(size_t) j] = d * scale;
                    M = std::max(M, sc[(size_t) j]);
                }
                double L = std::exp((double) sinks[h] - M);
                std::vector<double> acc(DV, 0.0);
                for (int64_t j = 0; j <= p; ++j) {
                    const double e = std::exp(sc[(size_t) j] - M);
                    L += e;
                    const float* vr = &vf[(size_t) j * nv + (size_t) gq * DV];
                    for (int i = 0; i < DV; ++i) acc[i] += e * vr[i];
                }
                for (int i = 0; i < DV; ++i) want[((size_t) t * H + h) * DV + i] = acc[i] / L;
            }
        std::vector<float> wantf(want.begin(), want.end());
        Dev<float> dq(q), o((size_t) c.T * H * DV), o1((size_t) H * DV);
        Dev<float> part(g::gqa_attend_scratch_bytes(H, DV, NC, c.T) / sizeof(float));
        if (!g::gqa_decode_shared(dq.p, q_stride, dk.p, dv.p, rows, H, n_kv, DK, DV, ds.p, scale, o.p, part.p, NC, c.pos0, c.T, nullptr, nullptr,
                                  262144)) {
            printf("gqa_decode_shared: skipped (no sm_80 tensor cores here)\n");
            return;
        }
        const std::vector<float> got = o.down();
        char name[160];
        snprintf(name, sizeof name, "gqa shared decode pos %lld T=%d vs float64", (long long) c.pos0, c.T);
        report(name, rel(got, wantf), 5e-6);
        bool same = true;   // each row alone (a window of one at its position, its q row) must give the same bits
        for (int t = 0; t < c.T && same; ++t) {
            Dev<float> q1(std::vector<float>(q.begin() + (size_t) t * q_stride, q.begin() + (size_t) (t + 1) * q_stride));
            g::gqa_decode_shared(q1.p, q_stride, dk.p, dv.p, rows, H, n_kv, DK, DV, ds.p, scale, o1.p, part.p, NC, c.pos0 + t, 1, nullptr,
                                 nullptr, 262144);
            const std::vector<float> alone = o1.down();
            same = memcmp(alone.data(), got.data() + (size_t) t * H * DV, alone.size() * sizeof(float)) == 0;
        }
        snprintf(name, sizeof name, "gqa shared decode pos %lld T=%d: every row bitwise its window-of-one", (long long) c.pos0, c.T);
        report_bool(name, same);
    }
    // timing at 131K keys: the per-row tensor-core kernel (64 chunks) against the shared one, by window rows
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (const int T : {1, 2, 3, 4, 8}) {
        auto q = randn((size_t) T * q_stride, 1.0f);
        Dev<float> dq(q), o((size_t) T * H * DV);
        Dev<float> part(g::gqa_attend_scratch_bytes(H, DV, 128, T) / sizeof(float));
        auto time = [&](auto fn) {
            fn();
            float best = 1e30f;
            for (int r = 0; r < 8; ++r) {
                CK(cudaEventRecord(a));
                for (int i = 0; i < 5; ++i) fn();
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, a, b));
                best = std::min(best, ms / 5);
            }
            return best;
        };
        const int64_t pos0 = 131072;
        const float t_old = time([&] {
            g::gqa_decode_mma(dq.p, q_stride, dk.p, dv.p, rows, H, n_kv, DK, DV, 0, scale, part.p, 64, pos0, T, nullptr, nullptr, 16);
            g::gqa_merge_w(part.p, ds.p, o.p, H, 64, T, nullptr);
        });
        const float t64 = time([&] {
            g::gqa_decode_shared(dq.p, q_stride, dk.p, dv.p, rows, H, n_kv, DK, DV, ds.p, scale, o.p, part.p, 64, pos0, T, nullptr, nullptr, 262144);
        });
        const float t128 = time([&] {
            g::gqa_decode_shared(dq.p, q_stride, dk.p, dv.p, rows, H, n_kv, DK, DV, ds.p, scale, o.p, part.p, 128, pos0, T, nullptr, nullptr, 262144);
        });
        printf("   full-layer decode attention at 131K keys, T=%d: per-row kernel %.3f ms; shared, 64 chunks %.3f ms, 128 chunks %.3f ms\n", T,
               t_old, t64, t128);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
}

void test_heads() {
    // heads_q8_0 vs one native MMVQ per head (Q8_0, n_in 256 -> 512 like wk_b, T = 3)
    const int H = 64, n_in = 256, n_out = 512, T = 3, nb = n_in / 32;
    std::vector<uint8_t> w((size_t) H * n_out * nb * 34);
    std::uniform_int_distribution<int> byte(-127, 127);
    for (size_t b = 0; b < w.size() / 34; ++b) {
        const __half d = __float2half(std::ldexp(1.0f + (float) (b % 7) / 8.0f, -9));
        memcpy(&w[b * 34], &d, 2);
        for (int i = 0; i < 32; ++i) w[b * 34 + 2 + i] = (uint8_t) (int8_t) byte(rng);
    }
    auto x = randn((size_t) H * T * n_in, 0.5f);
    Dev<uint8_t> dw(w);
    Dev<float> dx(x), y1((size_t) H * T * n_out), y2((size_t) H * T * n_out);
    Dev<uint8_t> xq(strata::kernels::native_q8_1_bytes(H * T * n_in));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    strata::kernels::native_quantize_q8_1(dx.p, xq.p, H * T * n_in, 1, s);
    g::heads_q8_0(dw.p, xq.p, y1.p, H, n_in, n_out, T, s);
    for (int h = 0; h < H; ++h)
        strata::kernels::native_mmvq(8, dw.p + (size_t) h * n_out * nb * 34, xq.p + (size_t) h * T * nb * 36,
                                     y2.p + (size_t) h * T * n_out, n_in, n_out, T, s);
    CK(cudaStreamSynchronize(s));
    report("heads_q8_0 = per-head MMVQ", rel(y1.down(), y2.down()), 1e-6);
}

void bench_heads_v41() {
    // DeepSeek-V4.1's wo_a (8 groups of 1024 x 4096, Q8_0) in a decode window: heads_q8_0's one launch vs one native MMVQ per
    // group (STRATA_DSV41_HEADS_MMVQ) - the same products, and their times
    const int H = 8, n_in = 4096, n_out = 1024, nb = n_in / 32;
    std::vector<uint8_t> w((size_t) H * n_out * nb * 34);
    std::uniform_int_distribution<int> byte(-127, 127);
    for (size_t b = 0; b < w.size() / 34; ++b) {
        const __half d = __float2half(std::ldexp(1.0f + (float) (b % 7) / 8.0f, -9));
        memcpy(&w[b * 34], &d, 2);
        for (int i = 0; i < 32; ++i) w[b * 34 + 2 + i] = (uint8_t) (int8_t) byte(rng);
    }
    Dev<uint8_t> dw(w);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (const int T : {1, 2, 3}) {
        auto x = randn((size_t) H * T * n_in, 0.5f);
        Dev<float> dx(x), y1((size_t) H * T * n_out), y2((size_t) H * T * n_out);
        Dev<uint8_t> xq(strata::kernels::native_q8_1_bytes(H * T * n_in));
        strata::kernels::native_quantize_q8_1(dx.p, xq.p, H * T * n_in, 1, s);
        auto one = [&] { g::heads_q8_0(dw.p, xq.p, y1.p, H, n_in, n_out, T, s); };
        auto per = [&] {
            for (int h = 0; h < H; ++h)
                strata::kernels::native_mmvq(8, dw.p + (size_t) h * n_out * nb * 34, xq.p + (size_t) h * T * nb * 36,
                                             y2.p + (size_t) h * T * n_out, n_in, n_out, T, s);
        };
        auto time = [&](auto fn) {   // the best of 8 batches of 20
            float best = 1e30f;
            for (int k = 0; k < 8; ++k) {
                fn();
                CK(cudaEventRecord(a, s));
                for (int i = 0; i < 20; ++i) fn();
                CK(cudaEventRecord(b, s));
                CK(cudaEventSynchronize(b));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, a, b));
                best = std::min(best, ms / 20);
            }
            return best;
        };
        const float t1 = time(one), t2 = time(per);
        CK(cudaStreamSynchronize(s));
        const double mb = (double) w.size() / 1e6;
        printf("   V4.1 wo_a, T=%d: heads_q8_0 %.1f us (%.0f GB/s), per-group MMVQ %.1f us (%.0f GB/s)\n", T, t1 * 1e3,
               mb / t1, t2 * 1e3, mb / t2);
        report("V4.1 wo_a: heads_q8_0 = per-group MMVQ", rel(y1.down(), y2.down()), 1e-6);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
}

void bench_hc_v41() {
    // DeepSeek-V4.1's mHC mixers (24 rows of 4 x 5120, F32) in a decode window: gemv_f32_w vs gemv_f32_split
    // (STRATA_DSV41_HC_SPLIT) - the same products summed in another order, and their times
    const int n_in = 4 * 5120, n_out = 24;
    auto w = randn((size_t) n_out * n_in, 0.02f);
    Dev<float> dw(w);
    Dev<float> part((size_t) 8 * n_out * g::gemv_f32_split_parts(n_in));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (const int T : {1, 2, 3}) {
        auto x = randn((size_t) T * n_in, 1.0f);
        Dev<float> dx(x), y1((size_t) T * n_out), y2((size_t) T * n_out);
        auto one = [&] { g::gemv_f32_w(dw.p, dx.p, y1.p, n_in, n_out, T, s); };
        auto split = [&] { g::gemv_f32_split(dw.p, dx.p, y2.p, part.p, n_in, n_out, T, s); };
        auto time = [&](auto fn) {   // the best of 8 batches of 20
            float best = 1e30f;
            for (int k = 0; k < 8; ++k) {
                fn();
                CK(cudaEventRecord(a, s));
                for (int i = 0; i < 20; ++i) fn();
                CK(cudaEventRecord(b, s));
                CK(cudaEventSynchronize(b));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, a, b));
                best = std::min(best, ms / 20);
            }
            return best;
        };
        const float t1 = time(one), t2 = time(split);
        CK(cudaStreamSynchronize(s));
        printf("   V4.1 mHC mixer, T=%d: gemv_f32_w %.1f us, gemv_f32_split %.1f us\n", T, t1 * 1e3, t2 * 1e3);
        report("V4.1 mHC mixer: gemv_f32_split = gemv_f32_w", rel(y1.down(), y2.down()), 1e-5);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        printf("no CUDA device\n");
        return 77;
    }
    setenv("STRATA_X_GQA_MMA", "0", 1);   // gqa_attend_w's FP32 paths; test_gqa calls the tensor-core kernel itself
    // STRATA_X_PARITY_ONLY=gqa_decode: just that test (e.g. beside a running server)
    if (const char* only = std::getenv("STRATA_X_PARITY_ONLY"); only && std::string(only) == "gqa_decode") {
        test_gqa_decode();
        printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return g_fail ? 1 : 0;
    }
    if (const char* only = std::getenv("STRATA_X_PARITY_ONLY"); only && std::string(only) == "heads_v41") {
        bench_heads_v41();
        bench_hc_v41();
        printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return g_fail ? 1 : 0;
    }
    if (const char* only = std::getenv("STRATA_X_PARITY_ONLY"); only && std::string(only) == "dflash_bench") {
        bench_dflash();
        return 0;
    }
    if (const char* only = std::getenv("STRATA_X_PARITY_ONLY"); only && std::string(only) == "dsa_bench") {
        bench_dsa();
        return 0;
    }
    if (const char* only = std::getenv("STRATA_X_PARITY_ONLY"); only && std::string(only) == "dsa_tiled") {
        test_dsa_tiled();
        printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return g_fail ? 1 : 0;
    }
    if (const char* only = std::getenv("STRATA_X_PARITY_ONLY"); only && std::string(only) == "gqa_shared") {
        test_gqa_shared();
        printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return g_fail ? 1 : 0;
    }
    if (const char* only = std::getenv("STRATA_X_PARITY_ONLY"); only && std::string(only) == "q8_mma") {
        test_q8_mma();
        printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return g_fail ? 1 : 0;
    }
    if (const char* only = std::getenv("STRATA_X_PARITY_ONLY"); only && std::string(only) == "gqa_noncausal") {
        test_gqa_noncausal();
        printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return g_fail ? 1 : 0;
    }
    test_generic();
    test_mhc();
    test_kda();
    test_dsa();
    test_dsa_tiled();
    test_moe();
    test_windows();
    test_dsa_mma();
    test_dsa_static_positions();
    test_heads();
    test_gqa();
    test_gqa_decode();
    test_gqa_noncausal();
    test_q8_mma();
    test_gqa_shared();
    printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
