// moex/ref/mimo_ref.cpp - see mimo_ref.hpp.  Line references are to ~/AI/llama.cpp-glm5next-mtp (c92b4ae77).
#include "mimo_ref.hpp"

#include "ref_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace moex {

using namespace moex::ref;

// ------------------------------------------------------------------ hparams (mimo2.cpp load_arch_hparams 3-27)

bool MimoHparams::load(const GgufFile& f, std::string& err) {
    const std::string a = f.str("general.architecture");
    if (a != "mimo2") {
        err = "architecture is " + a + ", not mimo2";
        return false;
    }
    auto k = [&](const char* s) { return a + "." + s; };
    n_embd = f.u32(k("embedding_length"));
    n_layer_all = f.u32(k("block_count"));
    const int64_t nextn = f.has_key(k("nextn_predict_layers")) ? f.u32(k("nextn_predict_layers")) : 0;
    n_layer = n_layer_all - nextn;
    n_vocab = f.has_key(k("vocab_size")) ? f.u32(k("vocab_size")) : (int64_t) f.iarr("tokenizer.ggml.token_type").size();
    n_head = f.u32(k("attention.head_count"));
    head_k = f.u32(k("attention.key_length"));
    head_v = f.u32(k("attention.value_length"));
    n_rot = f.u32(k("rope.dimension_count"));
    n_swa = f.u32(k("attention.sliding_window"));
    n_ff = f.u32(k("feed_forward_length"));
    n_expert = f.u32(k("expert_count"));
    n_used = f.u32(k("expert_used_count"));
    n_ff_exp = f.u32(k("expert_feed_forward_length"));
    rms_eps = f.f32(k("attention.layer_norm_rms_epsilon"));
    rope_base = f.f32(k("rope.freq_base"));
    rope_base_swa = f.has_key(k("rope.freq_base_swa")) ? f.f32(k("rope.freq_base_swa")) : rope_base;
    v_scale = f.has_key(k("attention.value_scale")) ? f.f32(k("attention.value_scale")) : 0.0f;
    if (v_scale == 1.0f) v_scale = 0.0f;   // mimo2.cpp 17: only a scale other than 1 is applied
    ew_scale = f.has_key(k("expert_weights_scale")) ? f.f32(k("expert_weights_scale")) : 0.0f;
    head_kv = f.iarr(k("attention.head_count_kv"));
    swa = f.iarr(k("attention.sliding_window_pattern"));
    if (head_kv.size() == 1) head_kv.assign((size_t) n_layer_all, head_kv[0]);
    if ((int64_t) head_kv.size() < n_layer_all || (int64_t) swa.size() < n_layer) {
        err = "per-layer arrays shorter than block_count";
        return false;
    }
    if (n_rot > head_k || n_rot % 2) {
        err = "rope dims";
        return false;
    }
    return true;
}

// ------------------------------------------------------------------ the model

MimoRef::MimoRef(GgufFile& f, const MimoHparams& hp, int64_t n_ctx, int threads)
    : f_(f), hp_(hp), n_ctx_(n_ctx), threads_(std::max(1, threads)) {
    kv_.resize((size_t) hp_.n_layer);
    for (int64_t il = 0; il < hp_.n_layer; ++il) {
        kv_[(size_t) il].k.assign((size_t) (n_ctx_ * hp_.head_kv[(size_t) il] * hp_.head_k), 0.0f);
        kv_[(size_t) il].v.assign((size_t) (n_ctx_ * hp_.head_kv[(size_t) il] * hp_.head_v), 0.0f);
    }
}

std::string MimoRef::tn(int64_t il, const char* suffix) const { return "blk." + std::to_string(il) + "." + suffix; }

void MimoRef::matvec(const std::string& name, const float* x, float* y) {
    const TensorInfo* t = f_.info(name);
    if (!t) {
        fprintf(stderr, "mimo_ref: missing tensor %s\n", name.c_str());
        abort();
    }
    ref::matvec(f_.data(name), t->type, t->ne[0], t->ne[1] * t->ne[2] * t->ne[3], x, y, threads_);
}

// ---- attention: mimo2.cpp graph (85-152) + llama-graph.cpp build_attn_mha's non-flash path (2605-2650)
void MimoRef::attention(int64_t il, int64_t pos, const float* cur, float* out) {
    const int64_t H = hp_.n_head, Hk = hp_.head_kv[(size_t) il], Dk = hp_.head_k, Dv = hp_.head_v, n = hp_.n_embd;
    const bool swa = hp_.is_swa(il);
    // the fused projection: Q (H x Dk), K (Hk x Dk), V (Hk x Dv), in that order (mimo2.cpp 98-104)
    std::vector<float> qkv((size_t) (H * Dk + Hk * Dk + Hk * Dv));
    matvec(tn(il, "attn_qkv.weight"), cur, qkv.data());
    rec_->f32("wqkv", (int) il, qkv.data(), (int64_t) qkv.size());
    float* q = qkv.data();
    float* kk = q + H * Dk;
    float* vv = kk + Hk * Dk;
    // RoPE (NEOX, the first n_rot dims of each head; the cache spans the whole head as ggml's does)
    std::vector<float> cache;
    const float corr[2] = {0.0f, 0.0f};
    rope_cache(pos, Dk, (int) hp_.n_rot, swa ? hp_.rope_base_swa : hp_.rope_base, 1.0f, 0.0f, 1.0f, corr, cache);
    for (int64_t h = 0; h < H; ++h) rope_neox(q + h * Dk, (int) hp_.n_rot, cache);
    for (int64_t h = 0; h < Hk; ++h) rope_neox(kk + h * Dk, (int) hp_.n_rot, cache);
    rec_->f32("Qcur", (int) il, q, Dk, H);
    rec_->f32("Kcur", (int) il, kk, Dk, Hk);
    rec_->f32("Vcur", (int) il, vv, Dv, Hk);
    Kv& c = kv_[(size_t) il];
    std::memcpy(c.k.data() + (size_t) (pos * Hk * Dk), kk, (size_t) (Hk * Dk) * sizeof(float));
    std::memcpy(c.v.data() + (size_t) (pos * Hk * Dv), vv, (size_t) (Hk * Dv) * sizeof(float));
    // the keys: every cell 0..pos (one sequence: cell = position), masked outside the window (hparams.h 455-467)
    const int64_t n_kv = pos + 1;
    const float scale = 1.0f / sqrtf((float) Dk);
    const TensorInfo* ts = f_.info(tn(il, "attn_sinks.weight"));
    const float* sinks = ts ? reinterpret_cast<const float*>(f_.data(tn(il, "attn_sinks.weight"))) : nullptr;
    std::vector<float> w((size_t) n_kv), vt((size_t) n_kv), kqv((size_t) (H * Dv));
    for (int64_t h = 0; h < H; ++h) {
        const int64_t g = h / (H / Hk);
        for (int64_t p = 0; p < n_kv; ++p) {
            const bool masked = swa && pos - p >= hp_.n_swa;
            w[(size_t) p] = masked ? -INFINITY : dot_f32(Dk, c.k.data() + (size_t) ((p * Hk + g) * Dk), q + h * Dk) * scale;
        }
        softmax_sink(w.data(), n_kv, sinks ? sinks + h : nullptr);
        for (int64_t d = 0; d < Dv; ++d) {
            for (int64_t p = 0; p < n_kv; ++p) vt[(size_t) p] = c.v[(size_t) ((p * Hk + g) * Dv + d)];
            kqv[(size_t) (h * Dv + d)] = dot_f32(n_kv, vt.data(), w.data());
        }
    }
    rec_->f32("kqv_out", (int) il, kqv.data(), H * Dv);
    matvec(tn(il, "attn_output.weight"), kqv.data(), out);
    rec_->f32("attn_out", (int) il, out, n);
    if (hp_.v_scale != 0.0f) {
        for (int64_t i = 0; i < n; ++i) out[i] *= hp_.v_scale;
        rec_->f32("attn_out_scaled", (int) il, out, n);
    }
}

// build_ffn(LLM_FFN_SILU, LLM_FFN_PAR): silu(gate) * up, then down
void MimoRef::swiglu_ffn(int64_t il, const std::string& gate, const std::string& up, const std::string& down,
                         const float* cur, float* out) {
    const int64_t n_ff = f_.info(gate)->ne[1];
    std::vector<float> g((size_t) n_ff), u((size_t) n_ff);
    matvec(up, cur, u.data());
    rec_->f32("ffn_up", (int) il, u.data(), n_ff);
    matvec(gate, cur, g.data());
    rec_->f32("ffn_gate", (int) il, g.data(), n_ff);
    for (int64_t i = 0; i < n_ff; ++i) g[(size_t) i] = swiglu(g[(size_t) i], u[(size_t) i], 0.0f);
    rec_->f32("ffn_swiglu", (int) il, g.data(), n_ff);
    matvec(down, g.data(), out);
}

// build_moe_ffn (llama-graph.cpp 1953-2190), sigmoid gating with the selection bias, normalized weights
void MimoRef::moe_ffn(int64_t il, const float* cur, float* out) {
    const int64_t E = hp_.n_expert, k = hp_.n_used, n = hp_.n_embd, nf = hp_.n_ff_exp;
    std::vector<float> logits((size_t) E), probs((size_t) E), sel((size_t) E);
    matvec(tn(il, "ffn_gate_inp.weight"), cur, logits.data());
    rec_->f32("ffn_moe_logits", (int) il, logits.data(), E);
    for (int64_t e = 0; e < E; ++e) probs[(size_t) e] = sigmoidf(logits[(size_t) e]);
    rec_->f32("ffn_moe_probs", (int) il, probs.data(), E);
    const float* pb = reinterpret_cast<const float*>(f_.data(tn(il, "exp_probs_b.bias")));
    for (int64_t e = 0; e < E; ++e) sel[(size_t) e] = probs[(size_t) e] + pb[e];
    rec_->f32("ffn_moe_probs_biased", (int) il, sel.data(), E);
    std::vector<int32_t> order((size_t) E);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return sel[(size_t) a] > sel[(size_t) b]; });
    std::vector<int32_t> top(order.begin(), order.begin() + k);
    rec_->i32("ffn_moe_topk", (int) il, top.data(), k);
    std::vector<float> w((size_t) k);
    for (int64_t j = 0; j < k; ++j) w[(size_t) j] = probs[(size_t) top[(size_t) j]];
    rec_->f32("ffn_moe_weights", (int) il, w.data(), 1, k);
    float sum = sum_f32(k, w.data());
    rec_->f32("ffn_moe_weights_sum", (int) il, &sum, 1);
    sum = std::max(sum, 6.103515625e-5f);
    rec_->f32("ffn_moe_weights_sum_clamped", (int) il, &sum, 1);
    for (auto& v : w) v = v / sum;
    rec_->f32("ffn_moe_weights_norm", (int) il, w.data(), k);
    if (hp_.ew_scale != 0.0f && hp_.ew_scale != 1.0f) {
        for (auto& v : w) v = v * hp_.ew_scale;
        rec_->f32("ffn_moe_weights_scaled", (int) il, w.data(), 1, k);
    }

    const TensorInfo* tg = f_.info(tn(il, "ffn_gate_exps.weight"));
    const TensorInfo* tu = f_.info(tn(il, "ffn_up_exps.weight"));
    const TensorInfo* td = f_.info(tn(il, "ffn_down_exps.weight"));
    std::vector<float> G((size_t) (nf * k)), U((size_t) (nf * k)), A((size_t) (nf * k)), Dn((size_t) (n * k)),
        W((size_t) (n * k)), moe((size_t) n, 0.0f);
    std::vector<uint8_t> buf;
    auto expert = [&](const TensorInfo* t, int64_t e, int64_t rows) -> const uint8_t* {
        const uint64_t bytes = t->row_bytes() * (uint64_t) rows;
        buf.resize(bytes);
        f_.read(*t, (uint64_t) e * bytes, bytes, buf.data());
        return buf.data();
    };
    for (int64_t j = 0; j < k; ++j) {
        const int64_t e = top[(size_t) j];
        ref::matvec(expert(tg, e, nf), tg->type, n, nf, cur, G.data() + j * nf, threads_);
        ref::matvec(expert(tu, e, nf), tu->type, n, nf, cur, U.data() + j * nf, threads_);
    }
    rec_->f32("ffn_moe_gate", (int) il, G.data(), nf, k);
    rec_->f32("ffn_moe_up", (int) il, U.data(), nf, k);
    for (int64_t i = 0; i < nf * k; ++i) A[(size_t) i] = swiglu(G[(size_t) i], U[(size_t) i], 0.0f);
    rec_->f32("ffn_moe_swiglu", (int) il, A.data(), nf, k);
    for (int64_t j = 0; j < k; ++j)
        ref::matvec(expert(td, top[(size_t) j], n), td->type, nf, n, A.data() + j * nf, Dn.data() + j * n, threads_);
    rec_->f32("ffn_moe_down", (int) il, Dn.data(), n, k);
    for (int64_t j = 0; j < k; ++j)
        for (int64_t i = 0; i < n; ++i) W[(size_t) (j * n + i)] = Dn[(size_t) (j * n + i)] * w[(size_t) j];
    rec_->f32("ffn_moe_weighted", (int) il, W.data(), n, k);
    for (int64_t j = 0; j < k; ++j)
        for (int64_t i = 0; i < n; ++i) moe[(size_t) i] = j == 0 ? W[(size_t) i] : moe[(size_t) i] + W[(size_t) (j * n + i)];
    std::copy(moe.begin(), moe.end(), out);
    rec_->f32("ffn_moe_out", (int) il, out, n);
}

// ---- one token: mimo2.cpp graph::graph (60-207)
void MimoRef::step(int32_t token, int64_t pos, std::vector<float>& logits, Recorder* rec) {
    Recorder none;
    rec_ = rec ? rec : &none;
    const int64_t n = hp_.n_embd;
    std::vector<float> x((size_t) n);
    {
        const TensorInfo* te = f_.info("token_embd.weight");
        std::vector<uint8_t> row(te->row_bytes());
        f_.read(*te, (uint64_t) token * te->row_bytes(), te->row_bytes(), row.data());
        ggml_get_type_traits(te->type)->to_float(row.data(), x.data(), n);
    }
    rec_->f32("inp_embd", -1, x.data(), n);
    std::vector<float> cur((size_t) n), out((size_t) n), ffn_inp((size_t) n);
    for (int64_t il = 0; il < hp_.n_layer; ++il) {
        rms_norm_w(x.data(), n, hp_.rms_eps, reinterpret_cast<const float*>(f_.data(tn(il, "attn_norm.weight"))), cur.data());
        rec_->f32("attn_norm", (int) il, cur.data(), n);
        attention(il, pos, cur.data(), out.data());
        for (int64_t i = 0; i < n; ++i) ffn_inp[(size_t) i] = out[(size_t) i] + x[(size_t) i];
        rec_->f32("ffn_inp", (int) il, ffn_inp.data(), n);
        rms_norm_w(ffn_inp.data(), n, hp_.rms_eps, reinterpret_cast<const float*>(f_.data(tn(il, "ffn_norm.weight"))),
                   cur.data());
        rec_->f32("ffn_norm", (int) il, cur.data(), n);
        if (f_.info(tn(il, "ffn_gate_inp.weight")) == nullptr) {
            swiglu_ffn(il, tn(il, "ffn_gate.weight"), tn(il, "ffn_up.weight"), tn(il, "ffn_down.weight"), cur.data(),
                       out.data());
            rec_->f32("ffn_out", (int) il, out.data(), n);
        } else {
            moe_ffn(il, cur.data(), out.data());
        }
        for (int64_t i = 0; i < n; ++i) x[(size_t) i] = out[(size_t) i] + ffn_inp[(size_t) i];
        rec_->f32("l_out", (int) il, x.data(), n);
    }
    rms_norm_w(x.data(), n, hp_.rms_eps, reinterpret_cast<const float*>(f_.data("output_norm.weight")), x.data());
    rec_->f32("result_norm", -1, x.data(), n);
    logits.resize((size_t) hp_.n_vocab);
    matvec("output.weight", x.data(), logits.data());
    rec_->f32("result_output", -1, logits.data(), hp_.n_vocab);
    rec_ = nullptr;
}

}  // namespace moex
