// src/x/ernie.cpp - moex: ERNIE-5.5-Flash (GGUF arch ernie55, written by moex/tools/ernie55/convert_gguf.cpp from Baidu's
// FP8 release) in strata-x.  The plan: moex/docs/ERNIE55_PLAN.md; the reference model the engine is checked against op by
// op: moex/tools/ernie55/ref/model.py (written from vLLM-ERNIE ba322a5 and SGLang f4284f2).
//
// What runs here: the GGUF's keys and tensors, the decode state - the 37 HCA layers' DeepSeek-V4 state (the raw K = V ring,
// one compressed row per 128 positions, the compressor's ring) and the 6 DSA layers' MLA cache rows of 576 and index keys -
// and the attention of both kinds with ERNIE's additions: the sigmoid output gate from the query LoRA and VHA (the low-rank
// post-mix across heads).  The HCA attention is DeepSeek-V4's ratio-128 layer (dsv4_kernels: q LoRA, the per-head RMS, YaRN
// at 1.6e5, the compressor, the attention with sinks over the window and every compressed row, the de-RoPE, the grouped
// output); the DSA attention is Hy4-flash's MLA (hy4_kernels: the 576-value cache row, the absorbed query, the attention with
// sinks over a list of positions) with V4's lightning-indexer scores and top-k over every position before the window.  mHC
// (V4's coefficients; the flat streams' factor 1 / (sqrt(mean) + eps)), the two-router routing, the latent MoE (Hy4's ends
// with ERNIE's latent norm), SiTU-GLU (the "SwiGLU limit" kSituGlu on every expert path), the head reduce and MultiMax are
// dispatched from stage.cpp.
#include "engine_impl.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/x/ernie_kernels.hpp"
#include "strata/x/hy4_kernels.hpp"
#include "strata/kernels/glu_act.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace strata::x {

namespace {
constexpr int64_t kRow = 576;   ///< a DSA layer's MLA cache row: the 512 latent values, the 64 rotated ones
/// pj's quantized-activation input is read only for a quantized matrix: BF16 / FP8 projections take the F32 rows
inline bool needs_q(const DT& w) { return w.d && !(w.type == GGML_TYPE_BF16 || w.type == kDtFP8 || w.type == 0); }
/// the indexer's E4M3 rounding of its q and keys (the references'); STRATA_ERNIE_IDX_FP8=0: exact scores (a check's switch)
bool idx_fp8() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_ERNIE_IDX_FP8");
        return !(e && e[0] == '0');
    }();
    return on;
}
}  // namespace

// ------------------------------------------------------------------ the GGUF's hparams

bool Engine::Impl::load_hparams_ernie(std::string& err) {
    const strata::GgufFile& m = gm->meta();
    const std::string pre = "ernie55.";
    auto get = [&](const std::string& k) { return m.get(pre + k); };
    auto u = [&](const char* k, int64_t& dst) -> bool {
        const strata::MetaValue* v = get(k);
        if (!v || !v->is_num()) return fail(err, "missing " + pre + k);
        dst = (int64_t) v->num();
        return true;
    };
    auto f = [&](const char* k, float& dst, float def, bool required) -> bool {
        const strata::MetaValue* v = get(k);
        if (!v) {
            if (required) return fail(err, "missing " + pre + k);
            dst = def;
            return true;
        }
        dst = (float) v->num();
        return true;
    };
    Hparams& h = hp;
    h.arch = Arch::ernie;
    int64_t mla_heads = 0, mla_q_lora = 0, mla_nope = 0, mla_rope = 0, n_shared = 0;
    if (!u("embedding_length", h.n_embd) || !u("block_count", h.n_layer_all) || !u("feed_forward_length", h.n_ff) ||
        !u("leading_dense_block_count", h.n_dense) || !u("attention.head_count", h.n_head) ||
        !u("attention.key_length", h.hd_k) || !u("attention.value_length", h.hd_v) || !u("rope.dimension_count", h.n_rot) ||
        !u("attention.q_lora_rank", h.q_lora) || !u("attention.sliding_window", h.n_swa) ||
        !u("attention.output_group_count", h.o_groups) || !u("attention.output_lora_rank", h.o_lora) ||
        !u("attention.mla.head_count", mla_heads) || !u("attention.mla.kv_lora_rank", h.kv_lora) ||
        !u("attention.mla.q_lora_rank", mla_q_lora) || !u("attention.mla.key_length_nope", mla_nope) ||
        !u("attention.mla.rope_dimension_count", mla_rope) || !u("attention.mla.value_length", h.v_mla) ||
        !u("attention.indexer.head_count", h.idx_heads) || !u("attention.indexer.key_length", h.idx_dim) ||
        !u("attention.indexer.top_k", h.idx_topk) || !u("hyper_connection.count", h.hc) ||
        !u("hyper_connection.sinkhorn_iterations", h.sinkhorn_iters) || !u("expert_count", h.n_expert) ||
        !u("expert_used_count", h.n_used) || !u("expert_feed_forward_length", h.n_ff_exp) ||
        !u("expert_shared_count", n_shared) || !u("expert_latent_length", h.n_ex) ||
        !u("rope.scaling.original_context_length", h.n_ctx_orig))
        return false;
    h.n_layer = h.n_layer_all;   // (DSpark's stages are not in the GGUF yet)
    h.k_mla = mla_nope + mla_rope;
    float factor = 1.0f, beta = 0.0f, lin_beta = 0.0f;
    if (!f("attention.layer_norm_rms_epsilon", h.rms_eps, 0, true) || !f("hyper_connection.epsilon", h.hc_eps, 0, true) ||
        !f("attention.compress_rope_freq_base", h.crope_base, 0, true) || !f("rope.freq_base", h.rope_base, 0, true) ||
        !f("rope.scaling.factor", factor, 1.0f, true) || !f("rope.scaling.yarn_beta_fast", h.beta_fast, 32.0f, false) ||
        !f("rope.scaling.yarn_beta_slow", h.beta_slow, 1.0f, false) || !f("activation.situ_beta", beta, 0, true) ||
        !f("activation.situ_linear_beta", lin_beta, 0, true))
        return false;
    h.rope_scale = factor > 0.0f ? 1.0f / factor : 1.0f;
    if (const strata::MetaValue* v = get("rope.scaling.type"); !v || v->s != "yarn")
        return fail(err, "ernie55: the HCA layers' RoPE is YaRN (rope.scaling.type yarn)");
    if (const strata::MetaValue* v = get("expert_gating_func"); !v || v->s != "sigmoid2")
        return fail(err, "ernie55: expert gating is not the two routers' sigmoids (sigmoid2)");
    if (const strata::MetaValue* v = get("activation"); !v || v->s != "situ_glu") return fail(err, "ernie55: the activation is not situ_glu");
    // the kernels' SiTU-GLU has the release's betas built in (strata/kernels/glu_act.hpp)
    if (beta != kernels::kSituBeta || lin_beta != kernels::kSituLinBeta)
        return fail(err, "ernie55: SiTU-GLU betas other than the release's 4 / 25");
    h.ew_norm = true;
    if (const strata::MetaValue* v = get("expert_weights_norm")) h.ew_norm = v->u != 0;
    if (!h.ew_norm) return fail(err, "ernie55: unnormalized expert weights (the router kernel always normalizes)");
    h.ew_scale = 1.0f;   // (the per-expert factor is a tensor; the config's scalar routed_scaling_factor is applied nowhere)
    if (n_shared != 1) return fail(err, "ernie55: one shared expert expected");
    if (const strata::MetaValue* v = m.get("tokenizer.ggml.eos_token_id")) h.eos_id = (int64_t) v->num();
    {   // the layer kinds: 128 = HCA, <= 0 = MLA + DSA (vLLM-ERNIE config.py:156-160)
        const strata::MetaValue* v = get("attention.compress_ratios");
        if (!v || v->type != strata::MetaType::ARRAY || (int64_t) v->items.size() < h.n_layer_all)
            return fail(err, "ernie55.attention.compress_ratios is not a per-layer array");
        for (int64_t il = 0; il < h.n_layer_all; ++il) {
            const int r = (int) v->items[(size_t) il].num();
            if (r != 128 && r > 0) return fail(err, "ernie55: compress ratio " + std::to_string(r) + " (128, or <= 0 for DSA)");
            h.cratio.push_back(r);
        }
    }
    size_t sh = 0;
    const strata::TensorInfo* te = gm->find("token_embd.weight", &sh);
    if (!te || te->shape.size() < 2) return fail(err, "no token_embd.weight");
    h.n_vocab = (int64_t) te->shape[1];
    h.head_kv.assign((size_t) h.n_layer_all, 1);   // one shared K = V head (HCA), one MLA latent (DSA)
    h.head_l.assign((size_t) h.n_layer_all, h.n_head);
    h.shexp = true;
    // SiTU-GLU on every expert path: the "SwiGLU limit" every path carries (strata/kernels/glu_act.hpp)
    h.clamp_exp.assign((size_t) h.n_layer_all, kernels::kSituGlu);
    h.clamp_sh = h.clamp_exp;
    // the kernels' geometry: V4's HCA (64 heads of 512 with RoPE on the last 64, the 128 window, 8 x 1024 groups), Hy4's MLA
    // (the 512 + 64 row, 64 heads of 192 + 64, values of 256), V4's indexer (64 heads of 128), four streams
    if (h.hc != 4) return fail(err, "ernie55: mHC with hc != 4");
    if (h.n_head != 64 || h.hd_k != 512 || h.hd_v != 512 || h.n_rot != 64 || h.n_swa != 128 || h.o_groups != 8 || h.o_lora != 1024 ||
        h.n_head * h.hd_k % h.o_groups != 0)
        return fail(err, "ernie55: HCA geometry - 64 heads of 512 (RoPE 64), window 128, 8 x 1024 output");
    if (mla_heads != h.n_head || h.kv_lora != 512 || mla_rope != h.n_rot || mla_nope != 192 || h.v_mla != 256 || mla_q_lora != h.q_lora)
        return fail(err, "ernie55: DSA geometry - 64 heads, the 512 latent + 64 RoPE, 192 + 64 query heads, 256 values, one q LoRA");
    if (h.idx_heads != 64 || h.idx_dim != 128 || h.idx_topk <= 0 || h.idx_topk > 4096)
        return fail(err, "ernie55: indexer geometry - 64 heads of 128, top-k at most 4096");
    if (h.n_ex <= 0 || h.n_ex % 256 || h.n_ff_exp % 256) return fail(err, "ernie55: the expert latent / width are not multiples of 256");
    if (h.n_dense < 0 || h.n_dense > h.n_layer) return fail(err, "ernie55: leading_dense_block_count out of range");
    return true;
}

// ------------------------------------------------------------------ the tensors

bool Engine::Impl::bind_layers_ernie(std::string& err) {
    // the one-column BF16 / F32 GEMVs with their rows preloaded (the same bits): the head, the routers, the compressors
    gemv_bf16_w_preload_default(true);
    gemv_f32_w_preload_default(true);
    auto get = [&](const std::string& nm, DT& d, bool required) -> bool {
        auto it = t.find(nm);
        if (it == t.end()) return !required || fail(err, "missing tensor " + nm);
        d = it->second;
        return true;
    };
    auto proj = [&](const DT& d, const std::string& nm, int64_t n_in, int64_t n_out) -> bool {
        if (d.type != 0 && d.type != GGML_TYPE_BF16 && d.type != kDtFP8 && !kn::native_mmvq_supported(d.type))
            return fail(err, nm + ": type " + std::to_string(d.type) + " has no GPU kernel");
        if (d.n_in() != n_in || d.n_out() != n_out)
            return fail(err, nm + ": [" + std::to_string(d.n_in()) + ", " + std::to_string(d.n_out()) + "], expected [" +
                             std::to_string(n_in) + ", " + std::to_string(n_out) + "]");
        return true;
    };
    auto f32 = [&](const DT& d, const std::string& nm, int64_t count) -> bool {
        if (d.type != 0 || d.n_in() * d.n_out() != count) return fail(err, nm + ": not F32 of " + std::to_string(count) + " values");
        return true;
    };
    // a small F32 tensor's values on the host (the mapping)
    auto host_f32 = [&](const std::string& nm, std::vector<float>& out, size_t count) -> bool {
        size_t shd = 0;
        const strata::TensorInfo* ti = gm->find(nm, &shd);
        if (!ti || (int) ti->type != GGML_TYPE_F32) return fail(err, nm + ": missing or not F32");
        uint64_t nv = 1;
        for (auto d : ti->shape) nv *= d;
        if (nv != count) return fail(err, nm + ": " + std::to_string(nv) + " values, expected " + std::to_string(count));
        out.resize(count);
        std::memcpy(out.data(), gm->shard(shd).tensor_data(*ti), count * sizeof(float));
        return true;
    };
    const int64_t n = hp.n_embd, H = hp.n_head, D = hp.hd_k, Lq = hp.q_lora, G = hp.o_groups, OL = hp.o_lora, hc = hp.hc;
    const int64_t Lkv = hp.kv_lora, Dk = hp.k_mla, Dv = hp.v_mla, Dr = hp.n_rot, ID = hp.idx_dim, IH = hp.idx_heads;
    const int64_t E = hp.n_expert, nx = hp.ex();
    // (moex, the hill-climb: the head over both GPUs when load_dense split it - DeepSeek-V4.1's, bind_head_split)
    if (!get("output_norm.weight", output_norm, true) || (head_lo > 0 ? !bind_head_split(err) : !get("output.weight", output, true)) ||
        !get("output_hc_fn.weight", out_hc_fn, true) || !get("output_hc_base.weight", out_hc_base, true) ||
        !get("output_hc_scale.weight", out_hc_scale, true))
        return false;
    if ((head_lo == 0 && !proj(output, "output.weight", n, hp.n_vocab)) || !f32(output_norm, "output_norm.weight", n) ||
        !f32(out_hc_fn, "output_hc_fn.weight", hc * n * hc) || !f32(out_hc_base, "output_hc_base.weight", hc) ||
        !f32(out_hc_scale, "output_hc_scale.weight", 1))
        return false;
    {   // MultiMax: its 4 ranges and 4 factors, kernel arguments
        std::vector<float> r, tt;
        if (!host_f32("output.multimax_ranges", r, 4) || !host_f32("output.multimax_ts", tt, 4)) return false;
        for (int i = 0; i < 4; ++i) {
            mmax.r[i] = r[(size_t) i];
            mmax.t[i] = tt[(size_t) i];
        }
    }
    L.resize((size_t) n_load);
    for (int64_t il = 0; il < n_load; ++il) {
        Layer& l = L[(size_t) il];
        auto nm = [&](const char* sfx) { return "blk." + std::to_string(il) + "." + sfx; };
        const bool dsa = hp.ernie_dsa(il);
        bool ok = get(nm("attn_norm.weight"), l.attn_norm, true) && get(nm("ffn_norm.weight"), l.ffn_norm, true) &&
                  get(nm("hc_attn_fn.weight"), l.hc_attn_fn, true) && get(nm("hc_attn_base.weight"), l.hc_attn_base, true) &&
                  get(nm("hc_attn_scale.weight"), l.hc_attn_scale, true) && get(nm("hc_ffn_fn.weight"), l.hc_ffn_fn, true) &&
                  get(nm("hc_ffn_base.weight"), l.hc_ffn_base, true) && get(nm("hc_ffn_scale.weight"), l.hc_ffn_scale, true) &&
                  get(nm("attn_q_a.weight"), l.q_a, true) && get(nm("attn_q_a_norm.weight"), l.q_a_norm, true) &&
                  get(nm("attn_q_b.weight"), l.q_b, true) && get(nm("attn_kv_a_norm.weight"), l.kv_a_norm, true) &&
                  get(nm("attn_sinks.weight"), l.sinks, true) && get(nm("attn_gate.weight"), l.attn_gate, true);
        if (!ok) return false;
        // mHC: F32 mixers (gemv_f32: llama.cpp's F32 products, no activation quantization), 24 mixes
        ok = f32(l.attn_norm, nm("attn_norm"), n) && f32(l.ffn_norm, nm("ffn_norm"), n) && f32(l.q_a_norm, nm("attn_q_a_norm"), Lq) &&
             f32(l.sinks, nm("attn_sinks"), H) && proj(l.q_a, nm("attn_q_a"), n, Lq) &&
             f32(l.hc_attn_fn, nm("hc_attn_fn"), hc * n * 24) && f32(l.hc_attn_base, nm("hc_attn_base"), 24) &&
             f32(l.hc_attn_scale, nm("hc_attn_scale"), 3) && f32(l.hc_ffn_fn, nm("hc_ffn_fn"), hc * n * 24) &&
             f32(l.hc_ffn_base, nm("hc_ffn_base"), 24) && f32(l.hc_ffn_scale, nm("hc_ffn_scale"), 3);
        if (!ok) return false;
        if (l.hc_attn_fn.n_in() != hc * n || l.hc_ffn_fn.n_in() != hc * n) return fail(err, nm("hc_*_fn") + ": not [4 n, 24]");
        if (!dsa) {   // HCA: DeepSeek-V4's ratio-128 layer, its gate on the 8 x 1024 grouped output
            ok = get(nm("attn_kv.weight"), l.wkv, true) && get(nm("attn_output_a.weight"), l.wo_a, true) &&
                 get(nm("attn_output_b.weight"), l.wo_b, true) && get(nm("attn_compressor_kv.weight"), l.c_kv, true) &&
                 get(nm("attn_compressor_gate.weight"), l.c_gate, true) && get(nm("attn_compressor_ape.weight"), l.c_ape, true) &&
                 get(nm("attn_compressor_norm.weight"), l.c_norm, true);
            if (!ok) return false;
            ok = proj(l.q_b, nm("attn_q_b"), Lq, H * D) && proj(l.wkv, nm("attn_kv"), n, D) && f32(l.kv_a_norm, nm("attn_kv_a_norm"), D) &&
                 proj(l.wo_a, nm("attn_output_a"), H * D / G, G * OL) && proj(l.wo_b, nm("attn_output_b"), G * OL, n) &&
                 proj(l.attn_gate, nm("attn_gate"), Lq, G * OL) && proj(l.c_kv, nm("attn_compressor_kv"), n, D) &&
                 proj(l.c_gate, nm("attn_compressor_gate"), n, D) && f32(l.c_ape, nm("attn_compressor_ape"), D * 128) &&
                 f32(l.c_norm, nm("attn_compressor_norm"), D);
            if (!ok) return false;
            if (hp.cratio[(size_t) il] != 128) return fail(err, nm("") + ": an HCA layer needs ratio 128");
        } else {   // MLA + DSA: Hy4's MLA, V4's indexer, the gate on the 64 x 256 attention output
            ok = get(nm("attn_kv_a_mqa.weight"), l.kv_a, true) && get(nm("attn_k_b.weight"), l.k_b, true) &&
                 get(nm("attn_v_b.weight"), l.v_b, true) && get(nm("attn_output.weight"), l.wo, true) &&
                 get(nm("indexer.attn_q_b.weight"), l.idx_q_b, true) && get(nm("indexer.attn_k.weight"), l.idx_k, true) &&
                 get(nm("indexer.k_norm.weight"), l.idx_k_norm, true) && get(nm("indexer.k_norm.bias"), l.idx_k_norm_b, true) &&
                 get(nm("indexer.proj.weight"), l.idx_proj, true);
            if (!ok) return false;
            ok = proj(l.q_b, nm("attn_q_b"), Lq, H * Dk) && proj(l.kv_a, nm("attn_kv_a_mqa"), n, Lkv + Dr) &&
                 f32(l.kv_a_norm, nm("attn_kv_a_norm"), Lkv) && proj(l.k_b, nm("attn_k_b"), Dk - Dr, Lkv * H) &&
                 proj(l.v_b, nm("attn_v_b"), Lkv, Dv * H) && proj(l.wo, nm("attn_output"), H * Dv, n) &&
                 proj(l.attn_gate, nm("attn_gate"), Lq, H * Dv) && proj(l.idx_q_b, nm("indexer.attn_q_b"), Lq, IH * ID) &&
                 proj(l.idx_k, nm("indexer.attn_k"), n, ID) && f32(l.idx_k_norm, nm("indexer.k_norm.weight"), ID) &&
                 f32(l.idx_k_norm_b, nm("indexer.k_norm.bias"), ID) && proj(l.idx_proj, nm("indexer.proj"), n, IH);
            if (!ok) return false;
            if (l.k_b.type == 0 || l.v_b.type == 0) return fail(err, nm("attn_k_b / attn_v_b") + ": F32 heads have no per-head kernel");
        }
        {   // VHA: M = I + V U^T (64 x 64) from U and V (64 x 16 each), in double, onto the layer's GPU
            std::vector<float> U, V;
            if (!host_f32(nm("attn_vha_u.weight"), U, (size_t) H * 16) || !host_f32(nm("attn_vha_v.weight"), V, (size_t) H * 16))
                return false;
            std::vector<float> M((size_t) H * H);
            for (int64_t a = 0; a < H; ++a)
                for (int64_t b = 0; b < H; ++b) {
                    double s = a == b ? 1.0 : 0.0;
                    for (int64_t j = 0; j < 16; ++j) s += (double) V[(size_t) (a * 16 + j)] * (double) U[(size_t) (b * 16 + j)];
                    M[(size_t) (a * H + b)] = (float) s;
                }
            Stage& g = stage_of(il);
            set_dev(g.dev);
            float* dm = g.dalloc<float>(M.size(), err);
            if (!dm || cudaMemcpy(dm, M.data(), M.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
                return fail(err, nm("VHA") + ": " + err);
            l.vha.d = dm;
            l.vha.type = 0;
            l.vha.ne[0] = H;
            l.vha.ne[1] = H;
            l.vha.bytes = M.size() * sizeof(float);
        }
        if (!hp.is_moe(il)) {   // layer 0's dense SiTU-GLU FFN
            ok = get(nm("ffn_gate.weight"), l.ffn_gate, true) && get(nm("ffn_up.weight"), l.ffn_up, true) &&
                 get(nm("ffn_down.weight"), l.ffn_down, true) && proj(l.ffn_gate, nm("ffn_gate"), n, hp.n_ff) &&
                 proj(l.ffn_up, nm("ffn_up"), n, hp.n_ff) && proj(l.ffn_down, nm("ffn_down"), hp.n_ff, n);
        } else {   // the multi-router latent MoE
            ok = get(nm("ffn_gate_inp.weight"), l.gate_inp, true) && get(nm("exp_probs_b.bias"), l.probs_b, true) &&
                 get(nm("ffn_exp_scale.weight"), l.exp_scale, true) && get(nm("ffn_routed_down.weight"), l.r_down, true) &&
                 get(nm("ffn_routed_up.weight"), l.r_up, true) && get(nm("ffn_latent_norm.weight"), l.latent_norm, true) &&
                 get(nm("ffn_gate_shexp.weight"), l.sh_gate, true) && get(nm("ffn_up_shexp.weight"), l.sh_up, true) &&
                 get(nm("ffn_down_shexp.weight"), l.sh_down, true);
            if (!ok) return false;
            const int64_t nsh = l.sh_gate.n_out();
            ok = proj(l.gate_inp, nm("ffn_gate_inp"), n, 2 * E) && f32(l.probs_b, nm("exp_probs_b"), E) &&
                 f32(l.exp_scale, nm("ffn_exp_scale"), E) && proj(l.r_down, nm("ffn_routed_down"), n, nx) &&
                 proj(l.r_up, nm("ffn_routed_up"), nx, n) && f32(l.latent_norm, nm("ffn_latent_norm"), nx) &&
                 proj(l.sh_gate, nm("ffn_gate_shexp"), n, nsh) && proj(l.sh_up, nm("ffn_up_shexp"), n, nsh) &&
                 proj(l.sh_down, nm("ffn_down_shexp"), nsh, n);
        }
        if (!ok) return false;
    }
    reset_dev_cache();
    rope_pc = dsv4_rope_pc(hp.crope_base, hp.rope_scale, 1.0f, (int) hp.n_ctx_orig, hp.beta_fast, hp.beta_slow);
    return true;
}

// ------------------------------------------------------------------ the decode state

uint64_t Engine::Impl::ernie_state_bytes(int64_t il) const {
    if (!hp.ernie_dsa(il)) return dsv4_state_bytes(il);   // HCA: V4's ratio-128 state
    return (uint64_t) opt.n_ctx * (kRow + hp.idx_dim) * 2;  // DSA: the cache rows and the index keys (F16), every position
}

bool Engine::Impl::alloc_state_ernie(std::string& err) {
    const int64_t n = hp.n_embd, D = hp.hd_k, ID = hp.idx_dim;
    lid_rows = (int) (opt.n_ctx + 1);                 // the indexer's score stride: a DSA token scores every position
    max_cells = (int) (hp.idx_topk + hp.n_swa);       // a DSA token's cells: the indexer's choice, then the window
    for (auto& gp : stages)
        if (!gp->alloc(err) || !gp->alloc_decode(err)) return false;
    if (cudaMallocHost(&h_x, (size_t) kTmax * n * sizeof(float)) != cudaSuccess ||
        cudaMallocHost(&h_parts, (size_t) kTmax * hp.n_used * n * sizeof(float)) != cudaSuccess ||
        cudaMallocHost(&h_ids, (size_t) kTmax * 64 * sizeof(int32_t)) != cudaSuccess ||
        cudaMallocHost(&h_amax, (size_t) kTmax * sizeof(int32_t)) != cudaSuccess)
        return fail(err, "pinned host buffers");
    uint64_t bytes = 0;
    {   // KV on demand (Hy4-flash's, hy4.cpp hy4_kv_ensure): the DSA layers' caches on CUDA VMM, mapped as the positions
        // arrive, the expert slots giving way when a GPU runs short and coming back when a request starts short - the same
        // values in the same rows, only mapped later; STRATA_ERNIE_KV_GROW=0: the whole context allocated at the start
        const char* v = std::getenv("STRATA_ERNIE_KV_GROW");
        kv_grow = !(v && v[0] == '0');
    }
    for (int64_t il = 0; il < n_load; ++il) {
        Layer& l = L[(size_t) il];
        Stage& g = stage_of(il);
        cudaSetDevice(g.dev);
        if (!hp.ernie_dsa(il)) {   // HCA (alloc_state_dsv4's ratio-128 layer)
            const int64_t S = dsv4_ring(128);
            l.raw_rows = dsv4_raw_rows();
            l.c_rows = opt.n_ctx / 128 + 2;
            if (!(l.raw = g.dalloc<uint16_t>((size_t) (l.raw_rows * D), err)) || !(l.ccache = g.dalloc<uint16_t>((size_t) (l.c_rows * D), err)) ||
                !(l.c_ring_kv = g.dalloc<float>((size_t) (S * D), err)) || !(l.c_ring_sc = g.dalloc<float>((size_t) (S * D), err)))
                return false;
        } else {   // DSA: the MLA cache rows and the index keys, every position (row = position)
            l.kv_rows = opt.n_ctx;
            l.c_rows = opt.n_ctx;
            if (kv_grow) {   // the whole context's addresses now, the memory as the positions arrive
                if (kv_vmm.empty()) {
                    kv_vmm.resize((size_t) n_load);
                    pool_vmm.resize((size_t) n_load);
                }
                if (!vmm_reserve(kv_vmm[(size_t) il], g.dev, kvg_kv_bytes(opt.n_ctx), err) ||
                    !vmm_reserve(pool_vmm[(size_t) il], g.dev, kvg_pool_bytes(opt.n_ctx), err))
                    return false;
                l.kv_cache = reinterpret_cast<uint16_t*>(kv_vmm[(size_t) il].va);
                l.lcache = reinterpret_cast<uint16_t*>(pool_vmm[(size_t) il].va);
                continue;   // (counted once mapped, below)
            }
            if (!(l.kv_cache = g.dalloc<uint16_t>((size_t) (l.kv_rows * kRow), err)) ||
                !(l.lcache = g.dalloc<uint16_t>((size_t) (l.c_rows * ID), err)))
                return false;
        }
        bytes += ernie_state_bytes(il);
    }
    if (kv_grow) {   // the first rows (STRATA_ERNIE_KV_GROW_INIT, 65,536 by default) and the step (STRATA_ERNIE_KV_GROW_STEP, the same)
        const char* st = std::getenv("STRATA_ERNIE_KV_GROW_STEP");
        kv_step = st && *st ? std::max<int64_t>(256, std::atoll(st)) : 65536;
        const char* v = std::getenv("STRATA_ERNIE_KV_GROW_INIT");
        const int64_t init = std::min<int64_t>(opt.n_ctx, v && *v ? std::max<int64_t>(1024, std::atoll(v)) : 65536);
        if (!hy4_kv_ensure(init - 64, err)) return false;
        kv_rows_init = kv_rows_mapped;
        for (int64_t il = 0; il < n_load && il < (int64_t) kv_vmm.size(); ++il)
            if (kv_vmm[(size_t) il].va) bytes += kv_vmm[(size_t) il].mapped + pool_vmm[(size_t) il].mapped;
    }
    {   // each stage's widening scratch for its BF16 / FP8 matrices over a prompt chunk (gemv_dt: widened, then one GEMM; a
        // matrix larger than the scratch in row blocks), as DeepSeek-V4.1's §62 rule: its largest such matrix, at most 48M floats
        constexpr size_t kWidenMax = (size_t) 48 << 20;
        std::vector<size_t> need(stages.size(), 0);
        for (int64_t il = 0; il < n_load; ++il) {
            const Layer& l = L[(size_t) il];
            for (const DT* d : {&l.q_a, &l.q_b, &l.wkv, &l.wo_a, &l.wo_b, &l.attn_gate, &l.c_kv, &l.c_gate, &l.kv_a, &l.k_b, &l.v_b,
                                &l.wo, &l.idx_q_b, &l.idx_k, &l.idx_proj, &l.ffn_gate, &l.ffn_up, &l.ffn_down, &l.gate_inp, &l.r_down,
                                &l.r_up, &l.sh_gate, &l.sh_up, &l.sh_down})
                if (d->d && (d->type == GGML_TYPE_BF16 || d->type == kDtFP8)) {
                    size_t& nd = need[(size_t) ldev[(size_t) il]];
                    nd = std::max(nd, std::min(kWidenMax, (size_t) (d->n_in() * d->n_out())));
                }
        }
        for (size_t si = 0; si < stages.size(); ++si)
            if (need[si] && !stages[si]->bf16_f32) {
                set_dev(stages[si]->dev);
                if (!(stages[si]->bf16_f32 = stages[si]->dalloc<float>(need[si], err)))
                    return fail(err, "the BF16 / FP8 weights' widening scratch: " + err);
                stages[si]->bf16_f32_n = need[si];
            }
    }
    reset_dev_cache();
    if (opt.verbose)
        fprintf(stderr, "strata-x: ERNIE-5.5 attention state %.1f MiB for %lld positions (HCA raw rings of %lld rows)%s\n",
                bytes / 1048576.0, (long long) opt.n_ctx, (long long) dsv4_raw_rows(),
                kv_grow ? ("; the DSA caches mapped through " + std::to_string(kv_rows_mapped) + " positions, grown on demand").c_str() : "");
    return true;
}

// ------------------------------------------------------------------ the attention

void Stage::ernie_attn(int il, int64_t pos0, int T, const float* in, float* out_r) {
    if (hp.ernie_dsa(il)) ernie_dsa(il, pos0, T, in, out_r);
    else ernie_hca(il, pos0, T, in, out_r);
}

/// An HCA layer (the reference's attn_hca; DeepSeek-V4's ratio-128 layer, Stage::dsv4, with ERNIE's gate and VHA):
///   qr = RMS(q_a x);  g = gate qr (8 x 1024);  q = RoPE(RMS_heads(q_b qr));  kv = RoPE(RMS(attn_kv x)) -> the raw ring
///   the compressor's row for each block of 128 this batch completes; the attention with sinks over the window and every
///   completed row; VHA across the heads; the de-RoPE; z = concat_g(wo_a[g] heads 8g .. 8g + 7); out = wo_b (z sigmoid(g))
void Stage::ernie_hca(int il, int64_t pos0, int T, const float* in, float* out_r) {
    const Layer& l = L[(size_t) il];
    const int n = (int) hp.n_embd, H = (int) hp.n_head, D = (int) hp.hd_k, Lq = (int) hp.q_lora, nr = (int) hp.n_rot;
    const int G = (int) hp.o_groups, OL = (int) hp.o_lora;
    const float eps = rms_eps();
    const Dsv4Rope& rp = m.rope_pc;
    const int roff = D - nr;
    if (needs_q(l.q_a) || needs_q(l.wkv) || needs_q(l.c_kv) || needs_q(l.c_gate)) quant(in, xq, n, T);
    // the query and the gate's logits, both from the normed q LoRA output; moex (the hill-climb): a decode token's q_a and
    // KV row in one launch, the compressor's two rows in another (gemv_pair_w: each row by its own kernel's code)
    const bool q_kv_pair = l.q_a.type == kDtFP8 && l.wkv.type == kDtFP8 &&
                           gemv_pair_w(l.q_a.d, static_cast<const uint8_t*>(l.q_a.s), true, qr, (int) l.q_a.n_out(), l.wkv.d,
                                       static_cast<const uint8_t*>(l.wkv.s), true, dkv, (int) l.wkv.n_out(), in, n, T, s);
    if (!q_kv_pair) pj(l.q_a, in, xq, qr, T);
    rms_norm(qr, l.q_a_norm.f(), qr, T, Lq, eps, s);
    probe("q_resid", il, qr, Lq);
    if (needs_q(l.q_b) || needs_q(l.attn_gate)) quant(qr, xq2, Lq, T);
    pj(l.q_b, qr, xq2, dq, T);
    pj(l.attn_gate, qr, xq2, hgate, T);
    // moex (the hill-climb): the heads' norm and their RoPE in one launch (dsv4_norm_rope: the same bits)
    if (!dsv4_norm_rope(dq, nullptr, T, H, D, eps, roff, rp, pos0, s, gpos)) {
        rms_norm(dq, nullptr, dq, T * H, D, eps, s);   // per head, no weight
        dsv4_rope(dq, dq, T, H, D, roff, rp, false, pos0, s, gpos);
    }
    probe("hca_q", il, dq, D, H);
    // the shared K = V row into the raw ring
    if (!q_kv_pair) pj(l.wkv, in, xq, dkv, T);
    // (moex, the hill-climb: its norm, RoPE and store in one launch - dsv4_norm_rope, the same bits)
    const bool kv_fused = dsv4_norm_rope(dkv, l.kv_a_norm.f(), T, 1, D, eps, roff, rp, pos0, s, gpos, l.raw, (int) l.raw_rows);
    if (!kv_fused) {
        rms_norm(dkv, l.kv_a_norm.f(), dkv, T, D, eps, s);
        dsv4_rope(dkv, dkv, T, 1, D, roff, rp, false, pos0, s, gpos);
    }
    probe("hca_kv", il, dkv, D);
    if (!kv_fused) dsv4_raw_store(dkv, l.raw, (int) l.raw_rows, pos0, T, s, gpos);
    // the compressor: the rows of the blocks this batch completes, then its state ring
    const Dsv4Compressor c = m.dsv4_comp(128, false);
    const bool c_pair = l.c_kv.type == GGML_TYPE_BF16 && l.c_gate.type == GGML_TYPE_BF16 &&
                        gemv_pair_w(l.c_kv.d, nullptr, false, dakv, (int) l.c_kv.n_out(), l.c_gate.d, nullptr, false, dgs,
                                    (int) l.c_gate.n_out(), in, n, T, s);
    if (!c_pair) {
        pj(l.c_kv, in, xq, dakv, T);
        pj(l.c_gate, in, xq, dgs, T);
    }
    dsv4_compress_pool(c, dakv, dgs, l.c_ape.f(), l.c_ring_kv, l.c_ring_sc, l.c_norm.f(), eps, m.rope_pc, l.ccache, pos0, T, s, gpos);
    dsv4_compress_persist(c, dakv, dgs, l.c_ape.f(), l.c_ring_kv, l.c_ring_sc, pos0, T, s, gpos);
    // the attention with sinks over the raw window and every completed row
    const float scale = 1.0f / std::sqrt((float) D);
    if (!dsv4_attn_mma(dq, l.raw, (int) l.raw_rows, l.ccache, 128, dsel, (int) hp.idx_topk, l.sinks.f(), scale, datt, dapart, pos0, T,
                       s, gpos, attn_noncausal))
        dsv4_attn(dq, l.raw, (int) l.raw_rows, l.ccache, 128, dsel, (int) hp.idx_topk, l.sinks.f(), scale, datt, dapart, pos0, T, s,
                  gpos, attn_noncausal);
    probe("hca_o", il, datt, D, H);
    // VHA, then the de-RoPE (they commute: every head of a token rotates by the same angles); q is spent
    ernie_vha_w(datt, dq, l.vha.f(), T, H, D, s);
    dsv4_rope(dq, dq, T, H, D, roff, rp, true, pos0, s, gpos);
    // the grouped low-rank output: group g = heads [8g, 8g + 8) through wo_a's rows [g OL, (g + 1) OL)
    const int gw = H * D / G;
    // moex (the ERNIE-5.5 hill-climb): one token's (1, G, gw) and (G, 1, OL) are their permutes' own layouts, byte for byte -
    // the copies are skipped and the products read the buffers in place.  STRATA_ERNIE_PERMUTE_COPY=1: the copies as before
    static const bool copy1 = [] {
        const char* e = std::getenv("STRATA_ERNIE_PERMUTE_COPY");
        return e && e[0] == '1';
    }();
    const bool in_place = T == 1 && !copy1;
    if (!in_place) permute_abc(dq, datt, T, G, gw, s);   // (T, G, gw) -> (G, T, gw)
    heads_mm(l.wo_a, in_place ? dq : datt, doa_hm, G, gw, OL, T);   // (G, T, OL)
    float* z = in_place ? doa_hm : doa;
    if (!in_place) permute_abc(doa_hm, doa, G, T, OL, s);   // (T, G * OL)
    probe("hca_z", il, z, (int64_t) G * OL);
    hy4_gate_mul(z, hgate, (int64_t) T * G * OL, s);
    if (needs_q(l.wo_b)) quant(z, xq, G * OL, T);
    pj(l.wo_b, z, xq, out_r, T);
    probe("attn_out", il, out_r, n);
}

/// A DSA layer (the reference's attn_dsa): MLA in the absorbed form over the indexer's 2,048 positions before the window
/// and the window:
///   qc = RMS(q_a x);  g = gate qc (64 x 256);  q = q_b qc (64 x [192 | 64]);  [kv_c | k_pe] = kv_a x -> the cache row
///   [RMS(kv_c) | RoPE(k_pe)] (RoPE: consecutive pairs at 8e6); q_pe RoPE'd, q_nope absorbed through k_b into the latent
///   the indexer: k = Hadamard(NeoX(LayerNorm(wk x))) and q = Hadamard(NeoX(wq_b qc)) per head (RoPE on dims 0-63), both
///   rounded to E4M3 per 128; score[j] = sum_h w_h relu(q_h . k_j) over j < p + 1 - 128 (w = weights_proj x / sqrt(128 64));
///   the best 2,048 (all of them when fewer); the attention with sinks over those and the window; out_lat -> v_b; VHA;
///   out = wo (o sigmoid(g))
void Stage::ernie_dsa(int il, int64_t pos0, int T, const float* in, float* out_r) {
    const Layer& l = L[(size_t) il];
    const int n = (int) hp.n_embd, H = (int) hp.H_of(il), Lq = (int) hp.q_lora, Dk = (int) hp.k_mla, Dv = (int) hp.v_mla;
    const int Dr = (int) hp.n_rot, Dn = Dk - Dr, Lkv = (int) hp.kv_lora, IH = (int) hp.idx_heads, ID = (int) hp.idx_dim;
    const int K = (int) hp.idx_topk, W = (int) hp.n_swa;
    const float eps = rms_eps(), base = hp.rope_base;
    if (needs_q(l.q_a) || needs_q(l.kv_a) || needs_q(l.idx_k) || needs_q(l.idx_proj)) quant(in, xq, n, T);
    pj(l.q_a, in, xq, qr, T);
    pj(l.kv_a, in, xq, hkv, T);
    pj(l.idx_k, in, xq, ikw, T);
    pj(l.idx_proj, in, xq, w32, T);
    rms_norm(qr, l.q_a_norm.f(), qr, T, Lq, eps, s);
    probe("q_resid", il, qr, Lq);
    if (needs_q(l.q_b) || needs_q(l.attn_gate) || needs_q(l.idx_q_b)) quant(qr, xq2, Lq, T);
    pj(l.q_b, qr, xq2, q, T);
    pj(l.attn_gate, qr, xq2, hgate, T);
    pj(l.idx_q_b, qr, xq2, iq, T);
    // the cache row: the normed latent and the rotated k_pe
    hy4_kv_row_w(hkv, l.kv_a_norm.f(), eps, base, nullptr, l.kv_rows, pos0, T, s, gpos, nullptr);
    probe("kv_cmpr", il, hkv, Lkv);
    probe("k_pe", il, hkv + Lkv, Dr);
    hy4_store_kv_w(hkv, l.kv_cache, l.kv_rows, pos0, T, s, gpos);
    // q: RoPE on each head's last 64, the nope part absorbed through k_b into the latent
    hy4_rope_w(q, H * Dk, 0, H, Dk, Dn, Dr, base, pos0, T, s, gpos, nullptr);
    hy4_q_split(q, q_hm, hqf, H, T, s);
    heads_mm(l.k_b, q_hm, qabs_hm, H, Dn, Lkv, T);
    hy4_q_join(qabs_hm, hqf, H, T, s);
    // the indexer's key, stored at its position; its query
    layer_norm_w(ikw, l.idx_k_norm.f(), l.idx_k_norm_b.f(), ikw, T, ID, 1e-6f, s);
    rope_neox_w(ikw, ID, 0, 1, ID, Dr, base, pos0, T, s, gpos);
    dsv4_hadamard(ikw, ikw, T, ID, s);
    if (idx_fp8()) ernie_fq_e4m3_w(ikw, (int64_t) T * ID / 128, s);
    probe("idx_k", il, ikw, ID);
    ernie_store_rows_w(ikw, l.lcache, ID, l.c_rows, pos0, T, s, gpos);
    rope_neox_w(iq, IH * ID, 0, IH, ID, Dr, base, pos0, T, s, gpos);
    dsv4_hadamard(iq, iq, T * IH, ID, s);
    if (idx_fp8()) ernie_fq_e4m3_w(iq, (int64_t) T * IH * ID / 128, s);
    probe("idx_q", il, iq, ID, IH);
    scale_f32(w32, T * IH, 1.0f / std::sqrt((float) (ID * IH)), s);
    probe("idx_w", il, w32, IH);
    // the scores and the top-k in blocks of kDsv41ScoreRows rows (the score buffer's; a decode window is one block): every
    // kernel computes a row on its own, so the blocks give the bits one launch would
    if (gpos && T > kDsv41ScoreRows) throw std::runtime_error("ernie_dsa: a graphed window of more than kDsv41ScoreRows rows");
    for (int b0 = 0; b0 < T; b0 += kDsv41ScoreRows) {
        const int bt = std::min(kDsv41ScoreRows, T - b0);
        const float* biq = iq + (size_t) b0 * IH * ID;
        const float* bw = w32 + (size_t) b0 * IH;
        const int64_t bp = pos0 + b0;
        if (!dsv4_lid_scores_mma(biq, l.lcache, bw, dscore, m.lid_rows, bp, bt, s, gpos, 1, false))
            dsv4_lid_scores(biq, l.lcache, bw, dscore, m.lid_rows, bp, bt, s, gpos, 1, false);
        dsv4_lid_select(dscore, m.lid_rows, K, dsel + (size_t) b0 * K, nullptr, bp, bt, s, gpos, 1, W);
    }
    probe_i32("dsa_sel", il, dsel, K);
    ernie_dsa_cells_w(dsel, K, W, cells, m.max_cells, ecnt, pos0, T, s, gpos);
    // the attention with sinks over the cells (the indexer's, then the window)
    const int chunks = T <= kTmax && hpart ? hy4_attend_chunks(m.max_cells) : 1;
    hy4_attend_w(hqf, l.kv_cache, l.kv_rows, cells, m.max_cells, 0, 1, W, l.sinks.f(), 1.0f / std::sqrt((float) Dk), ctx, hpart, chunks,
                 H, pos0, T, s, gpos, ecnt);
    probe("kqv", il, ctx, Lkv, 1, H);
    // moex (the ERNIE-5.5 hill-climb): one token's permutes are identities (ernie_hca's note) - v_b reads ctx in place and VHA
    // takes its rows from o2_hm into o2; STRATA_ERNIE_PERMUTE_COPY=1: the copies as before
    static const bool copy1 = [] {
        const char* e = std::getenv("STRATA_ERNIE_PERMUTE_COPY");
        return e && e[0] == '1';
    }();
    const bool in_place = T == 1 && !copy1;
    if (!in_place) permute_abc(ctx, ctx_hm, T, H, Lkv, s);
    heads_mm(l.v_b, in_place ? ctx : ctx_hm, o2_hm, H, Lkv, Dv, T);
    if (!in_place) permute_abc(o2_hm, o2, H, T, Dv, s);
    probe("dsa_o", il, in_place ? o2_hm : o2, Dv, H);
    // VHA (into o2_hm, free again; one token: into o2), the gate, the output projection
    float* va = in_place ? o2 : o2_hm;
    ernie_vha_w(in_place ? o2_hm : o2, va, l.vha.f(), T, H, Dv, s);
    hy4_gate_mul(va, hgate, (int64_t) T * H * Dv, s);
    if (needs_q(l.wo)) quant(va, xq, H * Dv, T);
    pj(l.wo, va, xq, out_r, T);
    probe("attn_out", il, out_r, n);
}

}  // namespace strata::x
