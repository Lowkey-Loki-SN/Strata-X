// moex/ref/glm_ref.cpp - see glm_ref.hpp.  Line references are to ~/AI/llama.cpp-glm5next-mtp (c92b4ae77).
#include "glm_ref.hpp"

#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

namespace moex {

// ------------------------------------------------------------------ hparams

bool GlmHparams::load(const GgufFile& f, std::string& err) {
    const std::string a = f.str("general.architecture");
    if (a != "glm5next") {
        err = "architecture is " + a + ", not glm5next";
        return false;
    }
    auto k = [&](const char* s) { return a + "." + s; };
    n_embd = f.u32(k("embedding_length"));
    n_layer_all = f.u32(k("block_count"));
    const int64_t nextn = f.has_key(k("nextn_predict_layers")) ? f.u32(k("nextn_predict_layers")) : 0;
    n_layer = n_layer_all - nextn;
    n_vocab = f.u32(k("vocab_size"));
    n_head = f.u32(k("attention.head_count"));
    hd_kda = f.u32(k("kda.head_dim"));
    d_conv = f.u32(k("ssm.conv_kernel"));
    n_dense = f.has_key(k("leading_dense_block_count")) ? f.u32(k("leading_dense_block_count")) : 0;
    n_ff = f.u32(k("feed_forward_length"));
    n_expert = f.u32(k("expert_count"));
    n_used = f.u32(k("expert_used_count"));
    n_ff_exp = f.u32(k("expert_feed_forward_length"));
    q_lora = f.u32(k("attention.q_lora_rank"));
    kv_lora = f.u32(k("attention.kv_lora_rank"));
    k_mla = f.u32(k("attention.key_length_mla"));
    v_mla = f.u32(k("attention.value_length_mla"));
    idx_heads = f.u32(k("attention.indexer.head_count"));
    idx_dim = f.u32(k("attention.indexer.key_length"));
    idx_topk = f.u32(k("attention.indexer.top_k"));
    kpool = f.u32(k("attention.indexer.kpool"));
    if (f.has_key(k("attention.indexer.kpool_select_tail"))) select_tail = f.boolean(k("attention.indexer.kpool_select_tail"));
    hc = f.u32(k("hyper_connection.count"));
    sinkhorn_iters = f.u32(k("hyper_connection.sinkhorn_iterations"));
    hc_eps = f.f32(k("hyper_connection.epsilon"));
    rms_eps = f.f32(k("attention.layer_norm_rms_epsilon"));
    ln_eps = f.has_key(k("attention.layer_norm_epsilon")) ? f.f32(k("attention.layer_norm_epsilon")) : 1e-6f;
    kda_lb = f.has_key(k("kda.gate_lower_bound")) ? f.f32(k("kda.gate_lower_bound")) : -INFINITY;
    ew_scale = f.has_key(k("expert_weights_scale")) ? f.f32(k("expert_weights_scale")) : 1.0f;
    ew_norm = f.has_key(k("expert_weights_norm")) && f.boolean(k("expert_weights_norm"));
    head_kv = f.iarr(k("attention.head_count_kv"));
    clamp_exp = f.farr(k("swiglu_clamp_exp"));
    clamp_sh = f.has_key(k("swiglu_clamp_shexp")) ? f.farr(k("swiglu_clamp_shexp")) : clamp_exp;
    if ((int64_t) head_kv.size() != n_layer_all || (int64_t) clamp_exp.size() != n_layer_all) {
        err = "per-layer arrays do not match block_count";
        return false;
    }
    if (hc != 4) {
        err = "mHC with hc != 4";
        return false;
    }
    if (f.has_key(k("expert_gating_func")) && f.u32(k("expert_gating_func")) != 2) {
        err = "expert gating is not sigmoid";
        return false;
    }
    if (f.has_key(k("rope.dimension_count")) && f.u32(k("rope.dimension_count")) != 0) {
        err = "GLM5-Next MLA is nope-only; this file has rope";
        return false;
    }
    return true;
}

// ------------------------------------------------------------------ elementwise, as ggml-cpu computes them

// ggml_compute_forward_rms_norm_f32 (ops.cpp ~3957): double sum of float squares, float mean, 1/sqrtf.
static void rms_norm(const float* x, int64_t n, float eps, float* y) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += (double) (x[i] * x[i]);
    const float mean = (float) (sum / (double) n);
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale;
}

static void rms_norm_w(const float* x, int64_t n, float eps, const float* w, float* y) {
    rms_norm(x, n, eps, y);
    for (int64_t i = 0; i < n; ++i) y[i] = y[i] * w[i];
}

// ggml_compute_forward_norm_f32 (ops.cpp ~3854): float mean of a double sum, centred variance, 1/sqrtf.
static void layer_norm(const float* x, int64_t n, float eps, const float* w, const float* b, float* y) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += (double) x[i];
    const float sum = (float) s;
    const float mean = sum / (float) n;
    double s2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        y[i] = x[i] - mean;
        s2 += (double) (y[i] * y[i]);
    }
    const float variance = (float) (s2 / (double) n);
    const float scale = 1.0f / sqrtf(variance + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = (y[i] * scale) * w[i] + b[i];
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float siluf(float x) { return x / (1.0f + expf(-x)); }

// ggml_compute_forward_swiglu_clamp_f32 (ops.cpp 3449): silu(min(gate, limit)) * clamp(up, -limit, limit)
static inline float swiglu_clamp(float gate, float up, float limit) {
    const float g = std::min(gate, limit);
    const float u = std::clamp(up, -limit, limit);
    return g / (1.f + expf(-g)) * u;
}

static float dot_f32(int64_t n, const float* a, const float* b) {
    float s = 0.0f;
    ggml_get_type_traits_cpu(GGML_TYPE_F32)->vec_dot((int) n, &s, 0, a, 0, b, 0, 1);
    return s;
}

// ggml_soft_max over n values: max, expf(x - max) summed in double, scaled by 1/sum.
static void softmax(float* x, int64_t n) {
    float mx = -INFINITY;
    for (int64_t i = 0; i < n; ++i) mx = std::max(mx, x[i]);
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        x[i] = expf(x[i] - mx);
        sum += (double) x[i];
    }
    const float inv = (float) (1.0 / sum);
    for (int64_t i = 0; i < n; ++i) x[i] *= inv;
}

// ------------------------------------------------------------------ the model

GlmRef::GlmRef(GgufFile& f, const GlmHparams& hp, int64_t n_ctx, int threads)
    : f_(f), hp_(hp), n_ctx_(n_ctx), threads_(std::max(1, threads)) {
    kda_.resize(hp_.n_layer);
    dsa_.resize(hp_.n_layer);
    const int64_t d_inner = hp_.n_head * hp_.hd_kda;
    for (int64_t il = 0; il < hp_.n_layer; ++il) {
        if (hp_.is_dsa(il)) {
            DsaState& s = dsa_[il];
            s.kv.assign(n_ctx_ * hp_.kv_lora, 0.0f);
            s.ik.assign(n_ctx_ * hp_.idx_dim, 0.0f);
            s.ig.assign(n_ctx_ * hp_.idx_dim, 0.0f);
            s.pooled.assign((n_ctx_ / hp_.kpool + 1) * hp_.idx_dim, 0.0f);
        } else {
            KdaState& s = kda_[il];
            for (auto& c : s.conv) c.assign(d_inner * (hp_.d_conv - 1), 0.0f);
            s.S.assign(hp_.n_head * hp_.hd_kda * hp_.hd_kda, 0.0f);
        }
    }
}

std::string GlmRef::tn(int64_t il, const char* suffix) const { return "blk." + std::to_string(il) + "." + suffix; }

void GlmRef::matvec(const uint8_t* w, ggml_type t, int64_t n_in, int64_t n_out, const float* x, float* y) const {
    const ggml_type_traits_cpu* tt = ggml_get_type_traits_cpu(t);
    const ggml_type vt = tt->vec_dot_type;
    std::vector<uint8_t> xq;
    const void* xa = x;
    if (vt != GGML_TYPE_F32) {
        xq.resize(ggml_row_size(vt, n_in));
        ggml_get_type_traits_cpu(vt)->from_float(x, xq.data(), n_in);
        xa = xq.data();
    }
    const size_t rb = ggml_row_size(t, n_in);
    auto run = [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) tt->vec_dot((int) n_in, &y[r], 0, w + r * rb, 0, xa, 0, 1);
    };
    const int nt = n_out >= 1024 ? threads_ : 1;
    if (nt == 1) {
        run(0, n_out);
        return;
    }
    std::vector<std::thread> pool;
    const int64_t chunk = (n_out + nt - 1) / nt;
    for (int i = 0; i < nt; ++i) {
        const int64_t r0 = i * chunk, r1 = std::min(n_out, r0 + chunk);
        if (r0 < r1) pool.emplace_back(run, r0, r1);
    }
    for (auto& th : pool) th.join();
}

void GlmRef::matvec(const std::string& name, const float* x, float* y) {
    const TensorInfo* t = f_.info(name);
    if (!t) {
        fprintf(stderr, "glm_ref: missing tensor %s\n", name.c_str());
        abort();
    }
    matvec(f_.data(name), t->type, t->ne[0], t->ne[1] * t->ne[2] * t->ne[3], x, y);
}

// ---- mHC: deepseek4.cpp build_hc_pre (357-413) with the fused hc_comb (ops.cpp 11142), hc_pre (11248), hc_post (11313)

GlmRef::HcCoef GlmRef::hc_coef(const std::vector<float>& R, int64_t il, const char* which) {
    const int64_t n = hp_.n_embd, hc = hp_.hc, dim = hc * n;
    std::vector<float> flat(dim);
    rms_norm(R.data(), dim, hp_.rms_eps, flat.data());   // ggml_rms_norm(flat, norm_rms_eps), no weight
    float mixes[24];
    const std::string base = std::string("hc_") + which;
    matvec(tn(il, (base + "_fn.weight").c_str()), flat.data(), mixes);
    rec_->f32("hc_mixes", (int) il, mixes, 24);
    const float* sc = reinterpret_cast<const float*>(f_.data(tn(il, (base + "_scale.weight").c_str())));
    const float* bs = reinterpret_cast<const float*>(f_.data(tn(il, (base + "_base.weight").c_str())));
    HcCoef c;
    for (int h = 0; h < 4; ++h) {
        c.pre[h] = sigmoidf(mixes[h] * sc[0] + bs[h]) * 1.0f + hp_.hc_eps;      // ggml_scale_bias(pre, 1, eps)
        c.post[h] = sigmoidf(mixes[4 + h] * sc[1] + bs[4 + h]) * 2.0f;
    }
    // the fused comb, verbatim: per src a softmax over dst, + eps, then norm_cols and (iters-1) x (rows, cols)
    float* comb = c.comb;
    const float eps = hp_.hc_eps;
    for (int src = 0; src < 4; ++src) {
        float mx = -INFINITY;
        for (int dst = 0; dst < 4; ++dst) {
            const int idx = dst + 4 * src;
            const float v = mixes[8 + idx] * sc[2] + bs[8 + idx];
            comb[idx] = v;
            mx = std::max(mx, v);
        }
        float sum = 0.0f;
        for (int dst = 0; dst < 4; ++dst) {
            const int idx = dst + 4 * src;
            comb[idx] = expf(comb[idx] - mx);
            sum += comb[idx];
        }
        const float inv = 1.0f / sum;
        for (int dst = 0; dst < 4; ++dst) comb[dst + 4 * src] = comb[dst + 4 * src] * inv + eps;
    }
    auto norm_cols = [&]() {   // per dst, over src
        for (int dst = 0; dst < 4; ++dst) {
            float sum = eps;
            for (int src = 0; src < 4; ++src) sum += comb[dst + 4 * src];
            const float inv = 1.0f / sum;
            for (int src = 0; src < 4; ++src) comb[dst + 4 * src] *= inv;
        }
    };
    auto norm_rows = [&]() {   // per src, over dst
        for (int src = 0; src < 4; ++src) {
            float sum = eps;
            for (int dst = 0; dst < 4; ++dst) sum += comb[dst + 4 * src];
            const float inv = 1.0f / sum;
            for (int dst = 0; dst < 4; ++dst) comb[dst + 4 * src] *= inv;
        }
    };
    norm_cols();
    for (int64_t i = 1; i < hp_.sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }
    rec_->f32("hc_pre", (int) il, c.pre, 4);
    rec_->f32("hc_post", (int) il, c.post, 4);
    rec_->f32("hc_comb", (int) il, c.comb, 4, 4);
    return c;
}

void GlmRef::hc_pre_apply(const std::vector<float>& R, const HcCoef& c, float* out) const {
    const int64_t n = hp_.n_embd;
    for (int64_t i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int h = 0; h < 4; ++h) sum += R[h * n + i] * c.pre[h];
        out[i] = sum;
    }
}

void GlmRef::hc_post(const float* x, const HcCoef& c, std::vector<float>& R) const {
    const int64_t n = hp_.n_embd;
    std::vector<float> out(4 * n);
    for (int dst = 0; dst < 4; ++dst)
        for (int64_t i = 0; i < n; ++i) {
            float sum = x[i] * c.post[dst];
            for (int src = 0; src < 4; ++src) sum += R[src * n + i] * c.comb[dst + 4 * src];
            out[dst * n + i] = sum;
        }
    R.swap(out);
}

// ---- KDA: glm5-next.cpp build_kda_layer (485-566), glm5_conv1d (197-230), fused gated_delta_net (ops.cpp 10895)

void GlmRef::kda_layer(int64_t il, const float* cur, float* out) {
    const int64_t H = hp_.n_head, D = hp_.hd_kda, d_inner = H * D, K = hp_.d_conv;
    KdaState& st = kda_[il];

    std::vector<float> qkv[3];
    const char* proj[3] = {"attn_q.weight", "attn_k.weight", "attn_v.weight"};
    const char* convw[3] = {"ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight"};
    const char* rname[3] = {"kda_q_conv", "kda_k_conv", "kda_v_conv"};
    for (int s = 0; s < 3; ++s) {
        std::vector<float> xp(d_inner);
        matvec(tn(il, proj[s]), cur, xp.data());
        const float* w = reinterpret_cast<const float*>(f_.data(tn(il, convw[s])));   // [d_conv, 1, d_inner]
        std::vector<float>& cs = st.conv[s];
        qkv[s].resize(d_inner);
        for (int64_t c = 0; c < d_inner; ++c) {
            // ggml_ssm_conv: sum over the window, oldest sample first, against w[c*K + t]
            float x4[8];
            for (int64_t t = 0; t < K - 1; ++t) x4[t] = cs[c * (K - 1) + t];
            x4[K - 1] = xp[c];
            float sumf = 0.0f;
            for (int64_t t = 0; t < K; ++t) sumf += x4[t] * w[c * K + t];
            for (int64_t t = 0; t < K - 1; ++t) cs[c * (K - 1) + t] = x4[t + 1];
            qkv[s][c] = siluf(sumf);
        }
        rec_->f32(rname[s], (int) il, qkv[s].data(), D, H);
    }

    // decay gate (504-520): g1 = lb * sigmoid(-(f_b(f_a x) + dt_bias) * A),  A = ssm_a = -exp(A_log)
    std::vector<float> fa(D), g(d_inner);
    matvec(tn(il, "ssm_f_a.weight"), cur, fa.data());
    matvec(tn(il, "ssm_f_b.weight"), fa.data(), g.data());
    const float* dtb = reinterpret_cast<const float*>(f_.data(tn(il, "ssm_dt.bias")));
    const float* A = reinterpret_cast<const float*>(f_.data(tn(il, "ssm_a")));
    for (int64_t h = 0; h < H; ++h)
        for (int64_t d = 0; d < D; ++d) {
            float v = g[h * D + d] + dtb[h * D + d];
            if (hp_.kda_lb > -INFINITY) {
                v = v * A[h];
                v = sigmoidf(v * -1.0f);
                v = v * hp_.kda_lb;
            } else {
                v = log1pf(expf(v)) * A[h];   // softplus path (not this model)
            }
            g[h * D + d] = v;
        }
    rec_->f32("kda_g1", (int) il, g.data(), D, H);

    std::vector<float> beta(H);
    matvec(tn(il, "ssm_beta.weight"), cur, beta.data());
    for (auto& b : beta) b = sigmoidf(b);
    rec_->f32("kda_beta", (int) il, beta.data(), 1, H);

    // FLA's l2 norm (533-537): rms_norm(eps / D) then * 1/sqrt(D), per head, q and k only
    const float l2_scale = 1.0f / std::sqrt((float) D);
    for (int s = 0; s < 2; ++s)
        for (int64_t h = 0; h < H; ++h) {
            float* p = qkv[s].data() + h * D;
            rms_norm(p, D, 1e-6f / (float) D, p);
            for (int64_t d = 0; d < D; ++d) p[d] = p[d] * l2_scale;
        }

    // the fused delta rule, per head: M[j][i] = S[i][j]; S[i][:] *= exp(g[i]); delta = (v - S^T k) * beta;
    // S += k delta^T; o = (S^T q) / sqrt(D)
    std::vector<float> o(d_inner);
    const float scale = 1.0f / sqrtf((float) D);
    auto head = [&](int64_t h0, int64_t h1) {
        std::vector<float> e(D), delta(D);
        for (int64_t h = h0; h < h1; ++h) {
            float* M = st.S.data() + h * D * D;
            const float* q = qkv[0].data() + h * D;
            const float* k = qkv[1].data() + h * D;
            const float* v = qkv[2].data() + h * D;
            for (int64_t i = 0; i < D; ++i) e[i] = expf(g[h * D + i]);
            for (int64_t j = 0; j < D; ++j)
                for (int64_t i = 0; i < D; ++i) M[j * D + i] *= e[i];
            for (int64_t j = 0; j < D; ++j) delta[j] = (v[j] - dot_f32(D, M + j * D, k)) * beta[h];
            for (int64_t j = 0; j < D; ++j)
                for (int64_t i = 0; i < D; ++i) M[j * D + i] = std::fma(k[i], delta[j], M[j * D + i]);
            for (int64_t j = 0; j < D; ++j) o[h * D + j] = dot_f32(D, M + j * D, q) * scale;
        }
    };
    {
        std::vector<std::thread> pool;
        const int64_t chunk = (H + threads_ - 1) / threads_;
        for (int i = 0; i < threads_; ++i) {
            const int64_t h0 = i * chunk, h1 = std::min(H, h0 + chunk);
            if (h0 < h1) pool.emplace_back(head, h0, h1);
        }
        for (auto& th : pool) th.join();
    }
    rec_->f32("kda_scan_out", (int) il, o.data(), D, H);

    // output gate (550-563): rms_norm(o) * ssm_norm * sigmoid(g_b(g_a x)), then wo
    std::vector<float> ga(D), g2(d_inner), normed(d_inner);
    matvec(tn(il, "ssm_g_a.weight"), cur, ga.data());
    matvec(tn(il, "ssm_g_b.weight"), ga.data(), g2.data());
    const float* onw = reinterpret_cast<const float*>(f_.data(tn(il, "ssm_norm.weight")));
    for (int64_t h = 0; h < H; ++h) rms_norm_w(o.data() + h * D, D, hp_.rms_eps, onw, normed.data() + h * D);
    rec_->f32("kda_g2", (int) il, g2.data(), D, H);
    rec_->f32("kda_normed", (int) il, normed.data(), D, H);
    for (int64_t i = 0; i < d_inner; ++i) normed[i] = normed[i] * sigmoidf(g2[i]);
    matvec(tn(il, "attn_output.weight"), normed.data(), out);
    rec_->f32("kda_out", (int) il, out, hp_.n_embd);
}

// ---- DSA: glm5-next.cpp build_dsa_layer (726-820) and build_kpool_select (570-722); the k-pool layout of
//      llama-memory-hybrid-idx.cpp (812-831, 1053-1110) for one sequence starting at position 0

void GlmRef::dsa_layer(int64_t il, int64_t pos, const float* cur, float* out) {
    const int64_t H = hp_.n_head, Lq = hp_.q_lora, Lkv = hp_.kv_lora, Dk = hp_.k_mla, Dv = hp_.v_mla;
    const int64_t IH = hp_.idx_heads, ID = hp_.idx_dim, KP = hp_.kpool;
    DsaState& st = dsa_[il];
    if (pos >= n_ctx_) {
        fprintf(stderr, "glm_ref: position %lld past n_ctx\n", (long long) pos);
        abort();
    }

    std::vector<float> qr(Lq), q(H * Dk), kv(Lkv), qabs(H * Lkv);
    matvec(tn(il, "attn_q_a.weight"), cur, qr.data());
    rms_norm_w(qr.data(), Lq, hp_.rms_eps, reinterpret_cast<const float*>(f_.data(tn(il, "attn_q_a_norm.weight"))), qr.data());
    rec_->f32("q_resid", (int) il, qr.data(), Lq);
    matvec(tn(il, "attn_q_b.weight"), qr.data(), q.data());
    matvec(tn(il, "attn_kv_a_mqa.weight"), cur, kv.data());
    rms_norm_w(kv.data(), Lkv, hp_.rms_eps, reinterpret_cast<const float*>(f_.data(tn(il, "attn_kv_a_norm.weight"))), kv.data());
    rec_->f32("kv_cmpr", (int) il, kv.data(), Lkv);
    std::copy(kv.begin(), kv.end(), st.kv.begin() + pos * Lkv);

    // absorb wk_b [Dk, Lkv, H]: per head, Lkv rows of Dk
    {
        const TensorInfo* t = f_.info(tn(il, "attn_k_b.weight"));
        const uint8_t* w = f_.data(t->name);
        const size_t head_bytes = t->row_bytes() * Lkv;
        for (int64_t h = 0; h < H; ++h) matvec(w + h * head_bytes, t->type, Dk, Lkv, q.data() + h * Dk, qabs.data() + h * Lkv);
    }
    rec_->f32("q_absorbed", (int) il, qabs.data(), Lkv, H);

    // ---- the indexer
    std::vector<float> iq(IH * ID), ik(ID), ig(ID), wts(IH);
    matvec(tn(il, "indexer.attn_q_b.weight"), qr.data(), iq.data());
    rec_->f32("indexer_q", (int) il, iq.data(), ID, IH);
    matvec(tn(il, "indexer.attn_k.weight"), cur, ik.data());
    layer_norm(ik.data(), ID, hp_.ln_eps, reinterpret_cast<const float*>(f_.data(tn(il, "indexer.k_norm.weight"))),
               reinterpret_cast<const float*>(f_.data(tn(il, "indexer.k_norm.bias"))), ik.data());
    rec_->f32("indexer_k", (int) il, ik.data(), ID);
    matvec(tn(il, "indexer_compressor_gate.weight"), cur, ig.data());
    rec_->f32("indexer_gate", (int) il, ig.data(), ID);
    std::copy(ik.begin(), ik.end(), st.ik.begin() + pos * ID);
    std::copy(ig.begin(), ig.end(), st.ig.begin() + pos * ID);

    // a pool completes when its last member arrives: pooled[d] = sum_k softmax_k(gate_k[d] + ape[d, k]) * key_k[d]
    if ((pos + 1) % KP == 0) {
        const int64_t P = pos / KP;
        const float* ape = reinterpret_cast<const float*>(f_.data(tn(il, "indexer_compressor_ape.weight")));   // [ID, KP]
        float* pooled = st.pooled.data() + P * ID;
        for (int64_t d = 0; d < ID; ++d) {
            float lg[16];
            for (int64_t k = 0; k < KP; ++k) lg[k] = st.ig[(P * KP + k) * ID + d] + ape[d + ID * k];
            softmax(lg, KP);
            double s = 0.0;
            for (int64_t k = 0; k < KP; ++k) s += (double) (lg[k] * st.ik[(P * KP + k) * ID + d]);
            pooled[d] = (float) s;
        }
        rec_->f32("indexer_pool_k_new", (int) il, pooled, ID);
    }

    matvec(tn(il, "indexer.proj.weight"), cur, wts.data());
    const float wscale = 1.0f / sqrtf(float(ID * IH));
    for (auto& w : wts) w = w * wscale;
    rec_->f32("indexer_weights", (int) il, wts.data(), IH);

    const int64_t n_pool = (pos + 1) / KP;   // complete pools, all ending at or before pos
    std::vector<float> score(n_pool);
    for (int64_t P = 0; P < n_pool; ++P) {
        float s = 0.0f;
        for (int64_t h = 0; h < IH; ++h) s += std::max(dot_f32(ID, iq.data() + h * ID, st.pooled.data() + P * ID), 0.0f) * wts[h];
        score[P] = s;
    }
    rec_->f32("indexer_score", (int) il, score.data(), n_pool);

    // selection: the top min(n_pool, top_k / kpool) pools (all of them below 2048 tokens), plus the tail
    std::vector<int64_t> cells;
    {
        const int64_t n_top = std::min<int64_t>(n_pool, hp_.idx_topk / KP);
        std::vector<int64_t> order(n_pool);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + n_top, order.end(),
                          [&](int64_t a, int64_t b) { return score[a] > score[b]; });
        // recorded as llama.cpp's indexer_top_k (the pool ids, by descending score); compared as a set
        std::vector<int32_t> top_pools(order.begin(), order.begin() + n_top);
        rec_->i32("indexer_top_k", (int) il, top_pools.data(), n_top);
        for (int64_t i = 0; i < n_top; ++i)
            for (int64_t k = 0; k < KP; ++k) cells.push_back((int64_t) top_pools[i] * KP + k);
        if (hp_.select_tail) {
            const int64_t n_tail = (pos + 1) % KP;
            for (int64_t k = 0; k < n_tail; ++k) cells.push_back(pos - k);
        }
        std::sort(cells.begin(), cells.end());
    }

    // ---- sparse MQA over the latent: kq = q_abs . kv * 1/sqrt(Dk), softmax over the selection, ctx = p . kv
    const float kq_scale = 1.0f / sqrtf((float) Dk);
    std::vector<float> ctx(H * Lkv), o(H * Dv);
    {
        const int64_t n_sel = (int64_t) cells.size();
        std::vector<float> p(n_sel);
        for (int64_t h = 0; h < H; ++h) {
            const float* qh = qabs.data() + h * Lkv;
            for (int64_t s = 0; s < n_sel; ++s) p[s] = dot_f32(Lkv, st.kv.data() + cells[s] * Lkv, qh) * kq_scale;
            softmax(p.data(), n_sel);
            float* c = ctx.data() + h * Lkv;
            std::fill(c, c + Lkv, 0.0f);
            for (int64_t s = 0; s < n_sel; ++s) {
                const float* kvc = st.kv.data() + cells[s] * Lkv;
                for (int64_t i = 0; i < Lkv; ++i) c[i] += p[s] * kvc[i];
            }
        }
    }
    rec_->f32("kqv", (int) il, ctx.data(), Lkv, 1, H);
    {
        const TensorInfo* tv = f_.info(tn(il, "attn_v_b.weight"));   // [Lkv, Dv, H]
        const uint8_t* wv = f_.data(tv->name);
        const size_t head_bytes = tv->row_bytes() * Dv;
        for (int64_t h = 0; h < H; ++h) matvec(wv + h * head_bytes, tv->type, Lkv, Dv, ctx.data() + h * Lkv, o.data() + h * Dv);
    }
    rec_->f32("kqv_out", (int) il, o.data(), H * Dv);
    matvec(tn(il, "attn_output.weight"), o.data(), out);
    rec_->f32("attn_out", (int) il, out, hp_.n_embd);
}

// ---- FFN: llama-graph.cpp build_ffn (LLM_FFN_SILU + LLM_FFN_PAR + swiglu_clamp_shexp) and build_moe_ffn

void GlmRef::swiglu_ffn(int64_t il, const std::string& gate, const std::string& up, const std::string& down, float limit,
                        const float* cur, float* out) {
    const TensorInfo* tg = f_.info(gate);
    const int64_t n_ff = tg->ne[1];
    std::vector<float> g(n_ff), u(n_ff);
    matvec(up, cur, u.data());
    rec_->f32("ffn_up", (int) il, u.data(), n_ff);
    matvec(gate, cur, g.data());
    rec_->f32("ffn_gate", (int) il, g.data(), n_ff);
    for (int64_t i = 0; i < n_ff; ++i) g[i] = limit > 1e-6f ? swiglu_clamp(g[i], u[i], limit) : siluf(g[i]) * u[i];
    rec_->f32("ffn_swiglu_limited", (int) il, g.data(), n_ff);
    matvec(down, g.data(), out);
}

void GlmRef::dense_ffn(int64_t il, const float* cur, float* out) {
    swiglu_ffn(il, tn(il, "ffn_gate.weight"), tn(il, "ffn_up.weight"), tn(il, "ffn_down.weight"), hp_.clamp_sh[il], cur, out);
    rec_->f32("ffn_out", (int) il, out, hp_.n_embd);
}

void GlmRef::moe_ffn(int64_t il, const float* cur, float* out) {
    const int64_t E = hp_.n_expert, k = hp_.n_used, n = hp_.n_embd, nf = hp_.n_ff_exp;
    std::vector<float> logits(E), probs(E), sel(E);
    matvec(tn(il, "ffn_gate_inp.weight"), cur, logits.data());
    rec_->f32("ffn_moe_logits", (int) il, logits.data(), E);
    for (int64_t e = 0; e < E; ++e) probs[e] = sigmoidf(logits[e]);
    rec_->f32("ffn_moe_probs", (int) il, probs.data(), E);
    const float* pb = reinterpret_cast<const float*>(f_.data(tn(il, "exp_probs_b.bias")));
    for (int64_t e = 0; e < E; ++e) sel[e] = probs[e] + pb[e];
    rec_->f32("ffn_moe_probs_biased", (int) il, sel.data(), E);
    std::vector<int32_t> order(E);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return sel[a] > sel[b]; });
    std::vector<int32_t> top(order.begin(), order.begin() + k);
    rec_->i32("ffn_moe_topk", (int) il, top.data(), k);
    std::vector<float> w(k);
    for (int64_t j = 0; j < k; ++j) w[j] = probs[top[j]];
    rec_->f32("ffn_moe_weights", (int) il, w.data(), 1, k);
    if (hp_.ew_norm) {
        double s = 0.0;
        for (float v : w) s += (double) v;
        const float sum = std::max((float) s, 6.103515625e-5f);
        for (auto& v : w) v = v / sum;
    }
    if (hp_.ew_scale != 0.0f && hp_.ew_scale != 1.0f)
        for (auto& v : w) v = v * hp_.ew_scale;
    rec_->f32("ffn_moe_weights_scaled", (int) il, w.data(), 1, k);

    const TensorInfo* tg = f_.info(tn(il, "ffn_gate_exps.weight"));
    const TensorInfo* tu = f_.info(tn(il, "ffn_up_exps.weight"));
    const TensorInfo* td = f_.info(tn(il, "ffn_down_exps.weight"));
    const float limit = hp_.clamp_exp[il];
    std::vector<float> G(nf * k), U(nf * k), A(nf * k), Dn(n * k), W(n * k), moe(n, 0.0f);
    std::vector<uint8_t> buf;
    auto expert = [&](const TensorInfo* t, int64_t e, int64_t rows) -> const uint8_t* {
        const uint64_t bytes = t->row_bytes() * rows;
        buf.resize(bytes);
        f_.read(*t, (uint64_t) e * bytes, bytes, buf.data());
        return buf.data();
    };
    // in two passes, so the recorded gate/up/activation (forced under an oracle) are what the down projections use
    for (int64_t j = 0; j < k; ++j) {
        const int64_t e = top[j];
        matvec(expert(tg, e, nf), tg->type, n, nf, cur, G.data() + j * nf);
        matvec(expert(tu, e, nf), tu->type, n, nf, cur, U.data() + j * nf);
    }
    rec_->f32("ffn_moe_gate", (int) il, G.data(), nf, k);
    rec_->f32("ffn_moe_up", (int) il, U.data(), nf, k);
    for (int64_t i = 0; i < nf * k; ++i) A[i] = swiglu_clamp(G[i], U[i], limit);
    rec_->f32("ffn_moe_swiglu_limited", (int) il, A.data(), nf, k);
    for (int64_t j = 0; j < k; ++j) matvec(expert(td, top[j], n), td->type, nf, n, A.data() + j * nf, Dn.data() + j * n);
    rec_->f32("ffn_moe_down", (int) il, Dn.data(), n, k);
    for (int64_t j = 0; j < k; ++j)
        for (int64_t i = 0; i < n; ++i) W[j * n + i] = Dn[j * n + i] * w[j];
    rec_->f32("ffn_moe_weighted", (int) il, W.data(), n, k);
    for (int64_t j = 0; j < k; ++j)
        for (int64_t i = 0; i < n; ++i) moe[i] = j == 0 ? W[i] : moe[i] + W[j * n + i];
    rec_->f32("ffn_moe_out", (int) il, moe.data(), n);

    std::vector<float> sh(n);
    swiglu_ffn(il, tn(il, "ffn_gate_shexp.weight"), tn(il, "ffn_up_shexp.weight"), tn(il, "ffn_down_shexp.weight"),
               hp_.clamp_sh[il], cur, sh.data());
    rec_->f32("ffn_shexp", (int) il, sh.data(), n);
    for (int64_t i = 0; i < n; ++i) out[i] = moe[i] + sh[i];
    rec_->f32("ffn_out", (int) il, out, n);
}

// ---- one token: glm5-next.cpp graph::graph (348-481)

void GlmRef::step(int32_t token, int64_t pos, std::vector<float>& logits, Recorder* rec) {
    Recorder none;
    rec_ = rec ? rec : &none;
    const int64_t n = hp_.n_embd;

    std::vector<float> x(n);
    {
        const TensorInfo* te = f_.info("token_embd.weight");
        std::vector<uint8_t> row(te->row_bytes());
        f_.read(*te, (uint64_t) token * te->row_bytes(), te->row_bytes(), row.data());
        ggml_get_type_traits(te->type)->to_float(row.data(), x.data(), n);
    }
    rec_->f32("inp_embd", -1, x.data(), n);
    std::vector<float> R(4 * n);
    for (int h = 0; h < 4; ++h) std::copy(x.begin(), x.end(), R.begin() + h * n);
    rec_->f32("hc_init", -1, R.data(), n, 4);

    std::vector<float> cur(n), out(n);
    for (int64_t il = 0; il < hp_.n_layer; ++il) {
        HcCoef c = hc_coef(R, il, "attn");
        hc_pre_apply(R, c, cur.data());
        rec_->f32("hc_attn_pre", (int) il, cur.data(), n);
        rms_norm_w(cur.data(), n, hp_.rms_eps, reinterpret_cast<const float*>(f_.data(tn(il, "attn_norm.weight"))), cur.data());
        rec_->f32("attn_norm", (int) il, cur.data(), n);
        if (hp_.is_dsa(il)) dsa_layer(il, pos, cur.data(), out.data());
        else kda_layer(il, cur.data(), out.data());
        hc_post(out.data(), c, R);
        rec_->f32("hc_attn_post", (int) il, R.data(), n, 4);

        c = hc_coef(R, il, "ffn");
        hc_pre_apply(R, c, cur.data());
        rec_->f32("hc_ffn_pre", (int) il, cur.data(), n);
        rms_norm_w(cur.data(), n, hp_.rms_eps, reinterpret_cast<const float*>(f_.data(tn(il, "ffn_norm.weight"))), cur.data());
        rec_->f32("ffn_norm", (int) il, cur.data(), n);
        if (il < hp_.n_dense) dense_ffn(il, cur.data(), out.data());
        else moe_ffn(il, cur.data(), out.data());
        hc_post(out.data(), c, R);
        rec_->f32("l_out", (int) il, R.data(), n, 4);
    }

    // build_hc_mean (deepseek4.cpp 270): ((R0 + R1) + R2) + R3, then * 1/hc
    std::vector<float> h(n);
    for (int64_t i = 0; i < n; ++i) h[i] = (((R[i] + R[n + i]) + R[2 * n + i]) + R[3 * n + i]) * (1.0f / 4.0f);
    rec_->f32("hc_head", -1, h.data(), n);
    rms_norm_w(h.data(), n, hp_.rms_eps, reinterpret_cast<const float*>(f_.data("output_norm.weight")), h.data());
    rec_->f32("result_norm", -1, h.data(), n);
    logits.resize(hp_.n_vocab);
    matvec("output.weight", h.data(), logits.data());
    rec_->f32("result_output", -1, logits.data(), hp_.n_vocab);
    rec_ = nullptr;
}

}  // namespace moex
