// src/x/hy4.cpp - moex: Hy4-flash (GGUF arch hy_v4_flash) in strata-x.  The design and the op-by-op description are
// in moex/docs/HY4_PORT.md; the references are llama.cpp PR #32517 (src/models/hy-v4-flash.cpp: it reads these GGUFs, and
// its CPU build is the oracle) and vLLM PR #68296 (vllm/models/hy_v4_flash: the official serving code).
//
// What runs here: the hparams and tensors of the GGUF, the decode state (the MLA cache rows of 576, full on the sparse layers
// and a ring on the sliding-window ones; the indexer's rings and pooled keys), the attention (MLA in the absorbed form, the
// sparse layers' cells from the k-pool indexer, the sinks, the gate, the post-norm), the latent MoE's two projections and the
// Engram (the hash on the host, the rows read from the GGUF's FP8 table, the update of the streams on the GPU).  iHC, the
// router, the experts' tiers and the head are strata-x's own (stage.cpp's branches for Arch::hy4).
#include "engine_impl.hpp"

#include <cuda.h>

#include "strata/artifact/gguf_reader.hpp"
#include "strata/x/hy4_kernels.hpp"

#include "ggml.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace strata::x {

namespace {
/// The reference builds every Engram norm as an RMSNorm with its eps unset, which llama.cpp (and the GGUF it reads) takes as
/// the F32 machine epsilon (src/models/hy-v4-flash.cpp HY_V4_FLASH_ENGRAM_EPS)
constexpr float kEgEps = 1.1920929e-07f;
constexpr int64_t kRow = 576;   ///< an MLA cache row: the 512 latent values, the 64 rotated ones
}  // namespace

// ------------------------------------------------------------------ the GGUF's hparams

bool Engine::Impl::load_hparams_hy4(std::string& err) {
    const strata::GgufFile& m = gm->meta();
    const std::string pre = "hy_v4_flash.";
    auto get = [&](const std::string& k) { return m.get(pre + k); };
    auto u = [&](const char* k, int64_t& dst) -> bool {
        const strata::MetaValue* v = get(k);
        if (!v || !v->is_num()) return fail(err, "missing " + pre + k);
        dst = (int64_t) v->u;
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
    // a per-layer array (n_layer_all entries; a scalar for every layer); ints() reads an integer array whole, the BOOL ones
    // (the patterns) keep every item in the reader's sample (43 < 64)
    auto per_layer = [&](const char* k, std::vector<int64_t>& dst, bool required) -> bool {
        const strata::MetaValue* v = get(k);
        if (!v) return !required || fail(err, "missing " + pre + k);
        if (v->type != strata::MetaType::ARRAY) {
            dst.assign((size_t) hp.n_layer_all, (int64_t) v->num());
            return true;
        }
        if ((int64_t) v->count != hp.n_layer_all || v->items.size() != v->count) return fail(err, pre + k + " is not a per-layer array");
        dst.clear();
        for (const auto& it : v->items) dst.push_back((int64_t) it.num());
        return true;
    };
    Hparams& h = hp;
    h.arch = Arch::hy4;
    if (!u("embedding_length", h.n_embd) || !u("block_count", h.n_layer_all) || !u("vocab_size", h.n_vocab) ||
        !u("feed_forward_length", h.n_ff) || !u("expert_count", h.n_expert) || !u("expert_used_count", h.n_used) ||
        !u("expert_feed_forward_length", h.n_ff_exp) || !u("attention.q_lora_rank", h.q_lora) ||
        !u("attention.kv_lora_rank", h.kv_lora) || !u("attention.key_length_mla", h.k_mla) ||
        !u("attention.value_length_mla", h.v_mla) || !u("rope.dimension_count", h.n_rot) ||
        !u("attention.sliding_window", h.n_swa) || !u("hyper_connection.count", h.hc) ||
        !u("attention.indexer.head_count", h.idx_heads) || !u("attention.indexer.key_length", h.idx_dim) ||
        !u("attention.indexer.top_k", h.idx_topk) || !u("attention.indexer.kpool", h.kpool))
        return false;
    if (get("expert_latent_length") && !u("expert_latent_length", h.n_ex)) return false;
    int64_t nextn = 0;
    if (get("nextn_predict_layers") && !u("nextn_predict_layers", nextn)) return false;
    h.n_layer = h.n_layer_all - nextn;
    if (get("leading_dense_block_count") && !u("leading_dense_block_count", h.n_dense)) return false;
    if (const strata::MetaValue* v = get("attention.indexer.kpool_select_tail")) h.select_tail = v->u != 0;
    if (const strata::MetaValue* v = m.get("tokenizer.ggml.eos_token_id")) h.eos_id = (int64_t) v->num();
    if (!f("attention.layer_norm_rms_epsilon", h.rms_eps, 0, true) || !f("hyper_connection.epsilon", h.hc_eps, 0, true) ||
        !f("hyper_connection.magnitude", h.hc_mag, 0, true) || !f("rope.freq_base", h.rope_base, 0, true) ||
        !f("rope.freq_base_swa", h.rope_base_swa, 10000.0f, false) || !f("expert_weights_scale", h.ew_scale, 1.0f, false))
        return false;
    if (const strata::MetaValue* v = get("expert_weights_norm")) h.ew_norm = v->u != 0;
    if (const strata::MetaValue* v = get("expert_gating_func"); v && v->u != 2) return fail(err, "expert gating is not sigmoid");
    std::vector<int64_t> heads, kvh, swa, idx;
    std::vector<float> clamp;
    if (!per_layer("attention.head_count", heads, true) || !per_layer("attention.head_count_kv", kvh, true) ||
        !per_layer("attention.sliding_window_pattern", swa, true))
        return false;
    {   // the routed experts' SwiGLU limit, per layer (llama.cpp: the routed experts only)
        const strata::MetaValue* v = get("swiglu_clamp_exp");
        if (!v) return fail(err, "missing " + pre + "swiglu_clamp_exp");
        if (v->type != strata::MetaType::ARRAY) {
            h.clamp_exp.assign((size_t) h.n_layer_all, (float) v->num());
        } else {
            if ((int64_t) v->count != h.n_layer_all || v->items.size() != v->count) return fail(err, pre + "swiglu_clamp_exp length");
            for (const auto& it : v->items) h.clamp_exp.push_back((float) it.num());
        }
        h.clamp_sh.assign((size_t) h.n_layer_all, 0.0f);   // the shared expert and the dense FFN: no clamp
    }
    h.head_l = heads;
    h.n_head = *std::max_element(heads.begin(), heads.end());
    h.swa_l.assign((size_t) h.n_layer_all, 0);
    for (int64_t il = 0; il < h.n_layer_all; ++il) h.swa_l[(size_t) il] = swa[(size_t) il] != 0;
    // the layers that own an indexer: the sparse ones (llama.cpp refuses a window layer that owns one; the draft block owns one)
    if (const strata::MetaValue* v = get("attention.indexer.types")) {
        if (v->type != strata::MetaType::ARRAY || v->items.size() != v->count || (int64_t) v->count > h.n_layer_all)
            return fail(err, pre + "attention.indexer.types is not a per-layer array");
        for (size_t il = 0; il < v->items.size(); ++il)
            if ((v->items[il].u != 0) == h.is_swa((int64_t) il))
                return fail(err, "layer " + std::to_string(il) + ": an indexer on a sliding-window layer, or a sparse layer without one");
    }
    for (int64_t il = 0; il < h.n_layer_all; ++il)
        if (kvh[(size_t) il] != 1) return fail(err, "layer " + std::to_string(il) + ": MLA as MQA needs one KV head");
    h.hc = h.hc ? h.hc : 1;
    // the geometry the kernels take (hy4_kernels.hpp): the 512 + 64 cache row, 192 + 64 query heads of 256 values, 16 indexer
    // heads of 128, 4 streams
    if (h.kv_lora != 512 || h.n_rot != 64 || h.k_mla != 256 || h.v_mla != 256 || h.idx_heads != 16 || h.idx_dim != 128 ||
        h.hc != 4 || h.kpool < 2 || h.kpool > 8 || h.idx_topk % h.kpool || !h.select_tail)
        return fail(err, "kernel geometry: a 512 latent + 64 RoPE, 256-value heads, 16 indexer heads of 128, 4 streams and a k-pool "
                         "of 2 .. 8 with its tail group are what the Hy4 kernels take");
    for (int64_t il = 0; il < h.n_layer_all; ++il)
        if (h.H_of(il) % 16) return fail(err, "kernel geometry: a multiple of 16 heads a layer");
    if (h.n_ex <= 0) h.n_ex = h.n_embd;
    // Engram (llama.cpp's PLE keys): the 0-based layers, the n-gram size, the heads per size, the row width, the hash's inputs
    if (const strata::MetaValue* lv = get("ple.layers")) {
        int64_t ng = 0, hpn = 0, dim = 0;
        if (!u("ple.ngram_size", ng) || !u("ple.heads_per_ngram", hpn) || !u("embedding_length_per_layer_input", dim)) return false;
        const strata::MetaValue *mul = get("ple.layer_multipliers"), *offs = get("ple.head_offsets"), *pr = get("ple.head_vocab_sizes"),
                                *pad = get("engram.pad_token_id"), *tmv = get("engram.token_map");
        if (!mul || !offs || !pr || !pad || !tmv) return fail(err, "hy_v4_flash: an Engram key is missing");
        const std::vector<int64_t> ls = m.ints(*lv), mv = m.ints(*mul), ov = m.ints(*offs), pv = m.ints(*pr), tm = m.ints(*tmv);
        const int64_t L = (int64_t) ls.size(), C = (ng - 1) * hpn;
        if (L < 1 || ng < 2 || ng > 16 || hpn < 1 || dim < 1 || dim > 1024 || (int64_t) mv.size() != L * ng ||
            (int64_t) ov.size() != L * C || (int64_t) pv.size() != L * C || (int64_t) tm.size() < h.n_vocab ||
            pad->u >= tm.size())
            return fail(err, "hy_v4_flash: the Engram metadata is inconsistent");
        for (int64_t j = 0; j < L; ++j)
            if (ls[(size_t) j] < 0 || ls[(size_t) j] >= h.n_layer || (j > 0 && ls[(size_t) j] <= ls[(size_t) j - 1]))
                return fail(err, "hy_v4_flash: an Engram layer out of range or out of order");
        h.engram = true;
        h.eg_layers.assign(ls.begin(), ls.end());
        h.eg_ngram = ng;
        h.eg_heads = hpn;
        h.eg_dim = dim;
        h.eg_mul = mv;
        h.eg_offsets = ov;
        h.eg_primes = pv;
        h.eg_token_map.assign(tm.begin(), tm.end());
        h.eg_pad = h.eg_token_map[(size_t) pad->u];   // the compressed pad id (llama.cpp engram_hash, vLLM FlashNgramHashState)
        eg_pad_token = (int64_t) pad->u;
    }
    return true;
}

// ------------------------------------------------------------------ the tensors

bool Engine::Impl::bind_layers_hy4(std::string& err) {
    // the one-column BF16 GEMV with its row preloaded (the same bits; STRATA_DSV41_PRELOAD=0 turns it off): the BF16 head of
    // the Q4_K_M / Q3_K / IQ2_AH sets 2.6 -> 1.0 ms a token (nsys, the 4-layer slice: 182 -> 257 tok/s), IQ2_AH's BF16
    // projections as well
    gemv_bf16_w_preload_default(true);
    gemv_f32_w_preload_default(true);   // moex (2026-11-19): the routers' F32 rows the same way (1.9 ms a token at ~134 GB/s)
    auto get = [&](const std::string& n, DT& d, bool required) -> bool {
        auto it = t.find(n);
        if (it == t.end()) return !required || fail(err, "missing tensor " + n);
        d = it->second;
        return true;
    };
    auto proj = [&](const DT& d, const std::string& n, int64_t n_in, int64_t n_out) -> bool {
        if (d.type != 0 && d.type != GGML_TYPE_BF16 && !kn::native_mmvq_supported(d.type))
            return fail(err, n + ": type " + std::to_string(d.type) + " has no GPU kernel");
        if (d.n_in() != n_in || d.n_out() != n_out)
            return fail(err, n + ": [" + std::to_string(d.n_in()) + ", " + std::to_string(d.n_out()) + "], expected [" +
                             std::to_string(n_in) + ", " + std::to_string(n_out) + "]");
        return true;
    };
    auto f32 = [&](const DT& d, const std::string& n, int64_t count) -> bool {
        if (d.type != 0 || d.n_in() * d.n_out() != count) return fail(err, n + ": not F32 of " + std::to_string(count) + " values");
        return true;
    };
    // The stream mixers (hc_*_fn, output_hc_fn) and the indexer's head weights (indexer.proj): F32 in the Q4_K_M, Q3_K and
    // IQ2_AH sets (products in F32, as llama.cpp's), Q8_0 / NVFP4 in the SPARK-4Bit sets - those through Stage::pj like every
    // quantized projection, the activations rounded to Q8_1 as llama.cpp rounds them to Q8_0 for a quantized matrix.
    auto mixer = [&](const DT& d, const std::string& n, int64_t n_in, int64_t n_out) -> bool {
        return d.type == 0 ? f32(d, n, n_in * n_out) : proj(d, n, n_in, n_out);
    };
    const int64_t n = hp.n_embd, nx = hp.ex(), Lq = hp.q_lora, Lkv = hp.kv_lora, Dk = hp.k_mla, Dv = hp.v_mla, Dr = hp.n_rot;
    const int64_t ID = hp.idx_dim, IH = hp.idx_heads, KP = hp.kpool, hc = hp.hc;
    if (!get("output_norm.weight", output_norm, true) || !get("output.weight", output, true) ||
        !get("output_hc_fn.weight", out_hc_fn, true) || !get("output_hc_base.weight", out_hc_base, true) ||
        !get("output_hc_scale.weight", out_hc_scale, true))
        return false;
    if (!proj(output, "output.weight", n, hp.n_vocab) || !f32(output_norm, "output_norm.weight", n) ||
        !mixer(out_hc_fn, "output_hc_fn.weight", hc * n, hc) || !f32(out_hc_base, "output_hc_base.weight", hc) ||
        !f32(out_hc_scale, "output_hc_scale.weight", 1))
        return false;
    L.resize((size_t) n_load);
    for (int64_t il = 0; il < n_load; ++il) {
        Layer& l = L[(size_t) il];
        auto nm = [&](const char* sfx) { return "blk." + std::to_string(il) + "." + sfx; };
        const bool trunk = il < hp.n_layer, swa = hp.is_swa(il);
        const int64_t H = hp.H_of(il);
        bool ok = get(nm("attn_norm.weight"), l.attn_norm, true) && get(nm("ffn_norm.weight"), l.ffn_norm, true) &&
                  get(nm("post_attention_norm.weight"), l.attn_post_norm, true) && get(nm("post_ffw_norm.weight"), l.ffn_post_norm, true) &&
                  get(nm("attn_q_a.weight"), l.q_a, true) && get(nm("attn_q_a_norm.weight"), l.q_a_norm, true) &&
                  get(nm("attn_q_b.weight"), l.q_b, true) && get(nm("attn_kv_a_mqa.weight"), l.kv_a, true) &&
                  get(nm("attn_kv_a_norm.weight"), l.kv_a_norm, true) && get(nm("attn_k_b.weight"), l.k_b, true) &&
                  get(nm("attn_v_b.weight"), l.v_b, true) && get(nm("attn_output.weight"), l.wo, true) &&
                  get(nm("attn_gate.weight"), l.attn_gate, true) && get(nm("attn_sinks.weight"), l.sinks, true);
        if (!ok) return false;
        ok = f32(l.attn_norm, nm("attn_norm"), n) && f32(l.ffn_norm, nm("ffn_norm"), n) && f32(l.attn_post_norm, nm("post_attention_norm"), n) &&
             f32(l.ffn_post_norm, nm("post_ffw_norm"), n) && f32(l.q_a_norm, nm("attn_q_a_norm"), Lq) &&
             f32(l.kv_a_norm, nm("attn_kv_a_norm"), Lkv) && f32(l.sinks, nm("attn_sinks"), H) &&
             proj(l.q_a, nm("attn_q_a"), n, Lq) && proj(l.q_b, nm("attn_q_b"), Lq, H * Dk) &&
             proj(l.kv_a, nm("attn_kv_a_mqa"), n, Lkv + Dr) && proj(l.k_b, nm("attn_k_b"), Dk - Dr, Lkv * H) &&
             proj(l.v_b, nm("attn_v_b"), Lkv, Dv * H) && proj(l.wo, nm("attn_output"), H * Dv, n) &&
             proj(l.attn_gate, nm("attn_gate"), n, H * Dv);
        if (!ok) return false;
        if (l.k_b.type == 0 || l.v_b.type == 0) return fail(err, nm("attn_k_b / attn_v_b") + ": F32 heads have no per-head kernel");
        if (trunk) {
            ok = get(nm("hc_attn_fn.weight"), l.hc_attn_fn, true) && get(nm("hc_attn_base.weight"), l.hc_attn_base, true) &&
                 get(nm("hc_attn_scale.weight"), l.hc_attn_scale, true) && get(nm("hc_ffn_fn.weight"), l.hc_ffn_fn, true) &&
                 get(nm("hc_ffn_base.weight"), l.hc_ffn_base, true) && get(nm("hc_ffn_scale.weight"), l.hc_ffn_scale, true) &&
                 mixer(l.hc_attn_fn, nm("hc_attn_fn"), hc * n, 2 * hc) && f32(l.hc_attn_base, nm("hc_attn_base"), 2 * hc) &&
                 f32(l.hc_attn_scale, nm("hc_attn_scale"), 2) && mixer(l.hc_ffn_fn, nm("hc_ffn_fn"), hc * n, 2 * hc) &&
                 f32(l.hc_ffn_base, nm("hc_ffn_base"), 2 * hc) && f32(l.hc_ffn_scale, nm("hc_ffn_scale"), 2);
        } else {   // the draft block: the next token's embedding into the trunk's final hidden state (no streams of its own)
            ok = get(nm("nextn.eh_proj.weight"), eh_proj, true) && get(nm("nextn.enorm.weight"), enorm, true) &&
                 get(nm("nextn.hnorm.weight"), hnorm, true) && get(nm("nextn.shared_head_norm.weight"), shead_norm, true) &&
                 proj(eh_proj, nm("nextn.eh_proj"), 2 * n, n);
        }
        if (!ok) return false;
        if (!swa) {   // the k-pool indexer (the sparse layers and the draft block)
            ok = get(nm("indexer.attn_k.weight"), l.idx_k, true) && get(nm("indexer.k_norm.weight"), l.idx_k_norm, true) &&
                 get(nm("indexer.attn_q_b.weight"), l.idx_q_b, true) && get(nm("indexer.proj.weight"), l.idx_proj, true) &&
                 get(nm("indexer_compressor_value.weight"), l.kp_value, true) &&
                 get(nm("indexer_compressor_gate.weight"), l.idx_gate, true) &&
                 get(nm("indexer_compressor_ape.weight"), l.idx_ape, true) &&
                 get(nm("indexer_compressor_norm.weight"), l.kp_norm, true) &&
                 proj(l.idx_k, nm("indexer.attn_k"), n, ID) && f32(l.idx_k_norm, nm("indexer.k_norm"), ID) &&
                 proj(l.idx_q_b, nm("indexer.attn_q_b"), Lq, IH * ID) && proj(l.idx_proj, nm("indexer.proj"), n, IH) &&
                 proj(l.kp_value, nm("indexer_compressor_value"), ID, ID) && proj(l.idx_gate, nm("indexer_compressor_gate"), ID, ID) &&
                 f32(l.idx_ape, nm("indexer_compressor_ape"), ID * KP) && f32(l.kp_norm, nm("indexer_compressor_norm"), ID);
            if (!ok) return false;
            if (l.idx_k.type == 0 || l.kp_value.type == 0 || l.idx_gate.type == 0 || l.idx_q_b.type == 0)
                return fail(err, nm("indexer") + ": an F32 projection where a quantized one is expected");
        }
        if (!hp.is_moe(il)) {
            ok = get(nm("ffn_gate.weight"), l.ffn_gate, true) && get(nm("ffn_up.weight"), l.ffn_up, true) &&
                 get(nm("ffn_down.weight"), l.ffn_down, true) && proj(l.ffn_gate, nm("ffn_gate"), n, hp.n_ff) &&
                 proj(l.ffn_up, nm("ffn_up"), n, hp.n_ff) && proj(l.ffn_down, nm("ffn_down"), hp.n_ff, n);
        } else {
            ok = get(nm("ffn_gate_inp.weight"), l.gate_inp, true) && get(nm("exp_probs_b.bias"), l.probs_b, true) &&
                 get(nm("ffn_gate_shexp.weight"), l.sh_gate, true) && get(nm("ffn_up_shexp.weight"), l.sh_up, true) &&
                 get(nm("ffn_down_shexp.weight"), l.sh_down, true) && f32(l.gate_inp, nm("ffn_gate_inp"), n * hp.n_expert) &&
                 f32(l.probs_b, nm("exp_probs_b"), hp.n_expert) && proj(l.sh_gate, nm("ffn_gate_shexp"), n, hp.n_ff_exp) &&
                 proj(l.sh_up, nm("ffn_up_shexp"), n, hp.n_ff_exp) && proj(l.sh_down, nm("ffn_down_shexp"), hp.n_ff_exp, n);
            if (ok && nx != n)
                ok = get(nm("ffn_routed_down.weight"), l.r_down, true) && get(nm("ffn_routed_up.weight"), l.r_up, true) &&
                     proj(l.r_down, nm("ffn_routed_down"), n, nx) && proj(l.r_up, nm("ffn_routed_up"), nx, n);
        }
        if (!ok) return false;
        if (trunk && hy4_eg_index(il) >= 0) {
            const int64_t C = (hp.eg_ngram - 1) * hp.eg_heads, D = hp.eg_dim;
            ok = get(nm("engram_kv.weight"), l.hy_eg_kv, true) && get(nm("engram_kv_b.weight"), l.hy_eg_kvb, true) &&
                 get(nm("engram_gate.weight"), l.hy_eg_gate, true) && get(nm("engram_lookup_norm.weight"), l.hy_eg_lnorm, true) &&
                 get(nm("engram_pre_gamma.weight"), l.hy_eg_pg, true) && get(nm("engram_hc_scale.weight"), l.hy_eg_hcs, true) &&
                 proj(l.hy_eg_kv, nm("engram_kv"), C * D, n * (hc + 1)) && f32(l.hy_eg_kvb, nm("engram_kv_b"), n * (hc + 1)) &&
                 f32(l.hy_eg_gate, nm("engram_gate"), n * hc) && f32(l.hy_eg_lnorm, nm("engram_lookup_norm"), D * C) &&
                 f32(l.hy_eg_pg, nm("engram_pre_gamma"), n) && f32(l.hy_eg_hcs, nm("engram_hc_scale"), n * hc);
            if (!ok) return false;
        }
    }
    return true;
}

int Engine::Impl::hy4_eg_index(int64_t il) const {
    if (!hp.engram || hp.arch != Arch::hy4) return -1;
    for (size_t j = 0; j < hp.eg_layers.size(); ++j)
        if (hp.eg_layers[j] == il) return (int) j;
    return -1;
}

// ------------------------------------------------------------------ the decode state

namespace {
bool cu_ok(CUresult r, const char* what, std::string& err) {
    if (r == CUDA_SUCCESS) return true;
    const char* m = nullptr;
    cuGetErrorString(r, &m);
    err = std::string(what) + ": " + (m ? m : "CUDA driver error");
    return false;
}
}  // namespace

/// moex (STRATA_HY4_KV_GROW): `bytes` of addresses on device `dev` (rounded up to the granularity), nothing mapped yet
bool Engine::Impl::vmm_reserve(VmmArr& a, int dev, size_t bytes, std::string& err) {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    if (vmm_gran == 0) {
        size_t g = 0;
        if (!cu_ok(cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "the VMM granularity", err)) return false;
        vmm_gran = g;
    }
    a.dev = dev;
    a.reserved = (bytes + vmm_gran - 1) / vmm_gran * vmm_gran;
    a.mapped = 0;
    CUdeviceptr p = 0;
    if (!cu_ok(cuMemAddressReserve(&p, a.reserved, 0, 0, 0), "reserving the cache's addresses", err)) return false;
    a.va = (unsigned long long) p;
    return true;
}

/// ... memory for its first `bytes` (granules created on its device, mapped, read-write there; zeroed)
bool Engine::Impl::vmm_map_to(VmmArr& a, size_t bytes, std::string& err) {
    bytes = std::min(a.reserved, (bytes + vmm_gran - 1) / vmm_gran * vmm_gran);
    if (bytes <= a.mapped) return true;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = a.dev;
    CUmemAccessDesc acc{};
    acc.location = prop.location;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cudaSetDevice(a.dev);
    for (size_t off = a.mapped; off < bytes; off += vmm_gran) {
        CUmemGenericAllocationHandle h = 0;
        if (!cu_ok(cuMemCreate(&h, vmm_gran, &prop, 0), "the cache's memory (no VRAM left for it)", err)) return false;
        if (!cu_ok(cuMemMap((CUdeviceptr) (a.va + off), vmm_gran, 0, h, 0), "mapping the cache's memory", err)) {
            cuMemRelease(h);
            return false;
        }
        if (!cu_ok(cuMemSetAccess((CUdeviceptr) (a.va + off), vmm_gran, &acc, 1), "the cache's access", err)) return false;
        a.h.push_back((unsigned long long) h);
        a.mapped = off + vmm_gran;
        if (cudaMemset(reinterpret_cast<void*>(a.va + off), 0, vmm_gran) != cudaSuccess) return fail(err, "zeroing the cache's memory");
    }
    return true;
}

/// moex (STRATA_HY4_KV_GROW): what stage g keeps free beside the caches' growth - its plan's reserve (the first GPU's
/// --vram-reserve-mib, a later one's --vram-reserve-later-mib: the display's on this box), 256 MiB at least
size_t Engine::Impl::vram_keep(const Stage& g) const {
    const int64_t mib = g.di == 0 ? opt.vram_reserve_mib : opt.vram_reserve_later_mib;
    return std::max<size_t>(256ull << 20, (size_t) std::max<int64_t>(0, mib) << 20);
}

/// moex: the growable arrays' bytes for `rows` positions - the MLA cache rows of 576 F16 (Hy4-flash's sparse layers, ERNIE-5.5's
/// DSA layers), and the second array: Hy4's pooled keys (one F32 row of idx_dim a pool of kpool) or ERNIE's index keys (an
/// F16 row a position)
size_t Engine::Impl::kvg_kv_bytes(int64_t rows) const { return (size_t) rows * kRow * 2; }
size_t Engine::Impl::kvg_pool_bytes(int64_t rows) const {
    return hp.arch == Arch::ernie ? (size_t) rows * hp.idx_dim * 2 : (size_t) (rows / hp.kpool + 1) * hp.idx_dim * 4;
}

/// moex (STRATA_HY4_KV_GROW; ERNIE-5.5: on, STRATA_ERNIE_KV_GROW=0 off): every sparse layer's cache and its second array
/// mapped through position pos_end (+ a margin: a window's rows and DFly2's block), in steps of kv_step positions; a no-op
/// without the switch
bool Engine::Impl::hy4_kv_ensure(int64_t pos_end, std::string& err) {
    if (!kv_grow) return true;
    const int64_t want = std::min<int64_t>(opt.n_ctx, pos_end + 64);
    if (want <= kv_rows_mapped) return true;
    const int64_t rows = std::min<int64_t>(opt.n_ctx, std::max<int64_t>(want, kv_rows_mapped + kv_step));
    auto rnd = [&](size_t b) { return (b + vmm_gran - 1) / vmm_gran * vmm_gran; };
    // a stage short of memory for its layers' growth gives it its top expert slots first, so that its reserve stays free: the
    // plan's floor holds while nothing else takes the GPU's memory; past it the reserve wins (a request never fails for it
    // while a slot is left)
    for (auto& gp : stages) {
        Stage& g = *gp;
        size_t need = 0;
        for (int64_t il = g.lb; il < g.le && il < (int64_t) kv_vmm.size(); ++il)
            if (kv_vmm[(size_t) il].va)
                need += (rnd(kvg_kv_bytes(rows)) - kv_vmm[(size_t) il].mapped) + (rnd(kvg_pool_bytes(rows)) - pool_vmm[(size_t) il].mapped);
        if (need == 0 || slot_vmm.size() <= (size_t) g.di || !slot_vmm[(size_t) g.di].va) continue;
        set_dev(g.dev);
        const size_t keep = vram_keep(g);
        for (int pass = 0; pass < 4 && g.n_slots > 0; ++pass) {
            size_t fr = 0, tot = 0;
            cudaMemGetInfo(&fr, &tot);
            if (fr >= need + keep) break;
            const size_t short_b = need + keep - fr + vmm_gran;   // (a granule more: the retired slots' last one may stay mapped)
            const int64_t k = (int64_t) ((short_b + slot_bytes - 1) / slot_bytes);
            const int64_t before = g.n_slots;
            if (!hy4_retire_slots(g, std::max<int64_t>(0, g.n_slots - k), err)) return false;
            if (g.n_slots >= before) break;   // (its top slot pinned: nothing more to give)
        }
    }
    for (int64_t il = 0; il < n_load && il < (int64_t) kv_vmm.size(); ++il) {
        if (!kv_vmm[(size_t) il].va) continue;
        if (!vmm_map_to(kv_vmm[(size_t) il], kvg_kv_bytes(rows), err) || !vmm_map_to(pool_vmm[(size_t) il], kvg_pool_bytes(rows), err))
            return fail(err, "the cache's growth to " + std::to_string(rows) + " positions: " + err);
    }
    for (auto& gp : stages) {
        set_dev(gp->dev);
        cudaDeviceSynchronize();   // (the zeroing done before any row is written)
    }
    if (opt.verbose && kv_rows_mapped > 0)
        fprintf(stderr, "strata-x: the sparse layers' caches grown to %lld positions (KV on demand)\n", (long long) rows);
    kv_rows_mapped = rows;
    return true;
}

/// moex (STRATA_HY4_KV_GROW): a request that starts well short of the caches' mapped rows - a new conversation, a compacted
/// one - gives the rows past its need back (a step's worth at least, never below the first rows), and each stage takes its
/// retired slots back as far as its free memory goes beside its reserve: empty, for the tier's first-use fills.  run_request
/// calls it once the rows it keeps are known (the reused prefix's: within the read's), every GPU idle; nothing reads a row
/// past the request's own positions, and a later growth maps them again, zeroed
bool Engine::Impl::hy4_kv_release(int64_t pos_end, std::string& err) {
    if (!kv_grow || kv_vmm.empty()) return true;
    const int64_t rows = std::min<int64_t>(opt.n_ctx, std::max<int64_t>(kv_rows_init, pos_end + 64));
    if (kv_rows_mapped - rows < kv_step) return true;
    finalize_swaps(true);
    for (auto& gp : stages) {
        set_dev(gp->dev);
        if (cudaDeviceSynchronize() != cudaSuccess) return fail(err, "a GPU failed before the caches' release");
    }
    for (int64_t il = 0; il < (int64_t) kv_vmm.size(); ++il) {
        if (!kv_vmm[(size_t) il].va) continue;
        vmm_unmap_from(kv_vmm[(size_t) il], kvg_kv_bytes(rows));
        vmm_unmap_from(pool_vmm[(size_t) il], kvg_pool_bytes(rows));
    }
    const int64_t was = kv_rows_mapped;
    kv_rows_mapped = rows;
    int64_t back = 0;
    std::string per;
    for (auto& gp : stages) {
        Stage& g = *gp;
        const size_t di = (size_t) g.di;
        if (slot_vmm.size() <= di || !slot_vmm[di].va || slot_n0.size() <= di || g.n_slots >= slot_n0[di]) continue;
        set_dev(g.dev);
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        const size_t keep = vram_keep(g) + vmm_gran;   // (a granule more: the new slots' last one is mapped whole)
        if (fr <= keep) continue;
        int64_t to = g.n_slots + std::min<int64_t>(slot_n0[di] - g.n_slots, (int64_t) ((fr - keep) / slot_bytes));
        if (to <= g.n_slots) continue;
        if (!vmm_map_to(slot_vmm[di], (size_t) to * slot_bytes, err)) {   // (as far as it went: the slots mapped whole)
            err.clear();
            to = std::min<int64_t>(to, (int64_t) (slot_vmm[di].mapped / slot_bytes));
        }
        for (int64_t k = g.n_slots; k < to; ++k) {   // (the retirement left them so; a slot index is the stage's own range)
            owner[(size_t) (g.slot_base + k)] = -1;
            slot_pending[(size_t) (g.slot_base + k)] = 0;
        }
        if (to > g.n_slots) {
            back += to - g.n_slots;
            per += (per.empty() ? "" : ", ") + std::string("GPU ") + std::to_string(g.dev) + " " + std::to_string(to);
            g.n_slots = to;
            g.st.slots = g.n_slots;
        }
    }
    kv_retired -= back;
    if (opt.verbose)
        fprintf(stderr, "strata-x: the sparse layers' caches released to %lld positions of %lld (STRATA_HY4_KV_GROW); %lld expert "
                        "slots back%s%s%s\n", (long long) rows, (long long) was, (long long) back, per.empty() ? "" : " (",
                per.c_str(), per.empty() ? "" : ")");
    return true;
}

/// ... the granules wholly past its first `bytes` unmapped and released (their memory back to the driver)
void Engine::Impl::vmm_unmap_from(VmmArr& a, size_t bytes) {
    bytes = (bytes + vmm_gran - 1) / vmm_gran * vmm_gran;
    cudaSetDevice(a.dev);
    while (a.mapped > bytes && !a.h.empty()) {
        a.mapped -= vmm_gran;
        cuMemUnmap((CUdeviceptr) (a.va + a.mapped), vmm_gran);
        cuMemRelease((CUmemGenericAllocationHandle) a.h.back());
        a.h.pop_back();
    }
}

/// moex (STRATA_HY4_KV_GROW): stage g's slots [new_n, n_slots) leave the tier for good, their memory to the caches - the
/// slot check's way out (resident and owner cleared, the residency republished before the next window: the CPU computes the
/// expert), an expert held only in VRAM first copied to a free exchange blob (else the file serves it, as a victim with no
/// blob), every swap in flight landed first and every GPU idle
bool Engine::Impl::hy4_retire_slots(Stage& g, int64_t new_n, std::string& err) {
    finalize_swaps(true);
    for (auto& gp : stages) {
        set_dev(gp->dev);
        if (cudaDeviceSynchronize() != cudaSuccess) return fail(err, "a GPU failed before the slots' retirement");
    }
    const int64_t NE = hp.n_expert;
    int64_t to_blob = 0, to_file = 0, kept = 0;
    const int64_t old_end = g.slot_base + g.n_slots;
    set_dev(g.dev);
    while (new_n < g.n_slots && slot_pinned[(size_t) (g.slot_base + g.n_slots - 1)] == 0) {
        const int64_t slot = g.slot_base + g.n_slots - 1;
        const int32_t key = owner[(size_t) slot];
        if (key >= 0 && resident[(size_t) key] == (int32_t) slot) {
            if (ram_off[(size_t) key] < 0) {
                const uint64_t need = L[(size_t) (key / NE)].fmt.bytes;
                size_t best = xpool.size();
                for (size_t i = 0; i < xpool.size(); ++i)
                    if (xpool[i].second >= need && (best == xpool.size() || xpool[i].second < xpool[best].second)) best = i;
                if (best < xpool.size() && arena_pinned &&
                    cudaMemcpy(arena->data() + xpool[best].first, slot_ptr((int32_t) slot), need, cudaMemcpyDeviceToHost) == cudaSuccess) {
                    ram_off[(size_t) key] = (int64_t) xpool[best].first;
                    ram_cap[(size_t) key] = xpool[best].second;
                    xpool[best] = xpool.back();
                    xpool.pop_back();
                    ++to_blob;
                } else {
                    ++to_file;
                }
            } else {
                ++kept;
            }
            resident[(size_t) key] = -1;
        }
        owner[(size_t) slot] = -1;
        slot_pending[(size_t) slot] = 0;
        --g.n_slots;
        ++kv_retired;
    }
    g.st.slots = g.n_slots;
    res_dirty = true;
    // moex (2026-11-26): a fill or swap that landed in a retired slot (finalize_swaps above queues its check for the next
    // maintain_cache) is not checked any more - its memory is unmapped next, and slices_hash would read it (an Xid 31)
    const int64_t new_end = g.slot_base + g.n_slots;
    verify_queue.erase(std::remove_if(verify_queue.begin(), verify_queue.end(),
                                      [&](const std::pair<int32_t, int32_t>& v) { return v.first >= new_end && v.first < old_end; }),
                       verify_queue.end());
    vmm_unmap_from(slot_vmm[(size_t) g.di], (size_t) g.n_slots * slot_bytes);
    if (opt.verbose)
        fprintf(stderr, "strata-x: GPU %d: %lld expert slots left now (STRATA_HY4_KV_GROW: the caches took the rest; their experts: "
                        "%lld with RAM copies, %lld copied out, %lld from the file)\n",
                g.dev, (long long) g.n_slots, (long long) kept, (long long) to_blob, (long long) to_file);
    return true;
}

bool Engine::Impl::alloc_state_hy4(std::string& err) {
    const int64_t n = hp.n_embd, ID = hp.idx_dim;
    max_pool = (int) (opt.n_ctx / hp.kpool + 1);
    max_cells = (int) (hp.idx_topk + hp.kpool);
    const int64_t tw = std::max<int64_t>(kTmax, opt.prefill_sub);
    ring_rows = (int) (((tw + hp.kpool + hp.kpool - 1) / hp.kpool) * hp.kpool);
    // a sliding-window layer's ring: the window, plus the longest batch written before it is read (rows it overwrites are out
    // of every remaining query's window)
    swa_rows = std::min<int64_t>(opt.n_ctx, (hp.n_swa + tw + 63) / 64 * 64);
    for (auto& gp : stages)
        if (!gp->alloc(err) || !gp->alloc_decode(err)) return false;
    if (cudaMallocHost(&h_x, (size_t) kTmax * n * sizeof(float)) != cudaSuccess ||
        cudaMallocHost(&h_parts, (size_t) kTmax * hp.n_used * n * sizeof(float)) != cudaSuccess ||
        cudaMallocHost(&h_ids, (size_t) kTmax * 64 * sizeof(int32_t)) != cudaSuccess ||
        cudaMallocHost(&h_amax, (size_t) kTmax * sizeof(int32_t)) != cudaSuccess)
        return fail(err, "pinned host buffers");
    uint64_t state_bytes = 0;
    {   // moex (2026-11-19): STRATA_HY4_KV_GROW=1 - the sparse layers' caches mapped as the positions arrive (hy4_kv_ensure)
        const char* v = std::getenv("STRATA_HY4_KV_GROW");
        kv_grow = v && v[0] == '1';
    }
    for (int64_t il = 0; il < n_load; ++il) {
        Layer& l = L[(size_t) il];
        Stage& g = stage_of(il);
        cudaSetDevice(g.dev);
        const bool swa = hp.is_swa(il);
        l.kv_rows = swa ? swa_rows : opt.n_ctx;
        if (kv_grow && !swa) {   // (moex, STRATA_HY4_KV_GROW: the addresses for the whole context, the memory as it is reached)
            if (kv_vmm.empty()) {
                kv_vmm.resize((size_t) n_load);
                pool_vmm.resize((size_t) n_load);
            }
            if (!vmm_reserve(kv_vmm[(size_t) il], g.dev, (size_t) l.kv_rows * kRow * 2, err) ||
                !vmm_reserve(pool_vmm[(size_t) il], g.dev, (size_t) max_pool * ID * 4, err))
                return false;
            l.kv_cache = reinterpret_cast<uint16_t*>(kv_vmm[(size_t) il].va);
            l.pooled = reinterpret_cast<float*>(pool_vmm[(size_t) il].va);
        } else if (!(l.kv_cache = g.dalloc<uint16_t>((size_t) (l.kv_rows * kRow), err))) {
            return false;
        }
        if (!kv_grow || swa) state_bytes += (uint64_t) l.kv_rows * kRow * 2;
        if (!swa) {
            if (!(l.ik_ring = g.dalloc<float>((size_t) ring_rows * ID, err)) || !(l.ig_ring = g.dalloc<float>((size_t) ring_rows * ID, err)) ||
                (!kv_grow && !(l.pooled = g.dalloc<float>((size_t) max_pool * ID, err))))
                return false;
            state_bytes += (uint64_t) (2 * ring_rows + (kv_grow ? 0 : max_pool)) * ID * 4;
        }
    }
    if (kv_grow) {   // the first rows (STRATA_HY4_KV_GROW_INIT, 65,536 by default), the step (STRATA_HY4_KV_GROW_STEP, the same)
        const char* s = std::getenv("STRATA_HY4_KV_GROW_STEP");
        kv_step = s && *s ? std::max<int64_t>(256, std::atoll(s)) : 65536;
        const char* v = std::getenv("STRATA_HY4_KV_GROW_INIT");
        const int64_t init = std::min<int64_t>(opt.n_ctx, v && *v ? std::max<int64_t>(1024, std::atoll(v)) : 65536);
        if (!hy4_kv_ensure(init - 64, err)) return false;
        kv_rows_init = kv_rows_mapped;
        for (int64_t il = 0; il < n_load; ++il)
            if (!kv_vmm.empty() && kv_vmm[(size_t) il].va) state_bytes += kv_vmm[(size_t) il].mapped + pool_vmm[(size_t) il].mapped;
    }
    for (auto& gp : stages)
        if (heg && !gp->hy_eg_grow(kTmax, err)) return false;
    if (opt.images)   // moex vision: the M-RoPE table a stage, its header {0, 0, 0} (every position its own) until a picture
        for (auto& gp : stages) {
            set_dev(gp->dev);
            const size_t bytes = Hy4MPos::bytes(opt.n_ctx);
            if (!(gp->hy_mpos = gp->dalloc<int64_t>((bytes + 7) / 8, err))) return fail(err, "the M-RoPE table: " + err);
            if (cudaMemset(gp->hy_mpos, 0, Hy4MPos::kHeader * sizeof(int64_t)) != cudaSuccess)
                return fail(err, "the M-RoPE table's header");
        }
    {   // each stage's BF16 widening scratch (gemv_dt's GEMMs over a prompt chunk; DeepSeek-V4.1's §62 rule, dsv41.cpp):
        // its largest BF16 layer weight, at most 48M floats (a larger one is widened in row blocks).  The IQ2_AH set's dense
        // projections are BF16; the Q4_K_M set's only BF16 weight is the head, which the prompt path runs on one row - no
        // scratch.  STRATA_X_BF16_ACT=1 (llama.cpp A/B) rounds a decode window's input into it: kTmax x the widest input.
        constexpr size_t kWidenMax = (size_t) 48 << 20;
        static const bool bf16_act = [] { const char* e = std::getenv("STRATA_X_BF16_ACT"); return e && e[0] == '1'; }();
        std::vector<size_t> need(stages.size(), 0);
        for (int64_t il = 0; il < n_load; ++il) {
            const Layer& l = L[(size_t) il];
            for (const DT* d : {&l.q_a, &l.q_b, &l.kv_a, &l.k_b, &l.v_b, &l.wo, &l.attn_gate, &l.idx_k, &l.idx_q_b, &l.idx_proj,
                                &l.kp_value, &l.idx_gate, &l.ffn_gate, &l.ffn_up, &l.ffn_down, &l.sh_gate, &l.sh_up, &l.sh_down,
                                &l.r_down, &l.r_up, &l.hy_eg_kv})
                if (d->d && d->type == GGML_TYPE_BF16) {
                    size_t& nd = need[(size_t) ldev[(size_t) il]];
                    nd = std::max(nd, std::min(kWidenMax, (size_t) (d->n_in() * d->n_out())));
                    if (bf16_act) nd = std::max(nd, (size_t) kTmax * (size_t) max_proj_in(hp));
                }
        }
        if (bf16_act && output.d && output.type == GGML_TYPE_BF16)
            need.back() = std::max(need.back(), (size_t) kTmax * (size_t) max_proj_in(hp));
        for (size_t si = 0; si < stages.size(); ++si)
            if (need[si] && !stages[si]->bf16_f32) {
                set_dev(stages[si]->dev);
                if (!(stages[si]->bf16_f32 = stages[si]->dalloc<float>(need[si], err)))
                    return fail(err, "the BF16 weights' widening scratch: " + err);
                stages[si]->bf16_f32_n = need[si];
            }
    }
    reset_dev_cache();
    if (opt.verbose)
        fprintf(stderr, "strata-x: decode state %.1f MiB for %lld positions (sliding-window rings of %lld)\n", state_bytes / 1048576.0,
                (long long) opt.n_ctx, (long long) swa_rows);
    return true;
}

bool Engine::Impl::hy4_mpos_set(std::string& err) {
    if (stages.empty() || !stages[0]->hy_mpos) return true;   // (no pictures in this run: the token indices)
    if (img.empty() && !hy_mpos_live) return true;            // the tables say {0, 0, 0} already
    int64_t b = 0, e = 0, shift = 0;
    constexpr size_t kH = Hy4MPos::kHeader * 2;                // the header's int32s
    if (img.empty()) {
        h_mpos.assign(kH, 0);
    } else {
        b = img.front().start;
        e = img.back().start + img.back().n;
        if (b < 0 || e > opt.n_ctx || e <= b) return fail(err, "the pictures' positions leave the context");
        h_mpos.assign(kH + (size_t) (e - b) * 3, 0);
        int32_t* tab = h_mpos.data() + kH;
        int64_t p = b;
        for (const ImageSpan& im : img) {   // (sorted, disjoint: run_request checked)
            for (; p < im.start; ++p) {     // text between pictures: counted on from the last picture's end
                int32_t* r = tab + (size_t) (p - b) * 3;
                r[0] = r[1] = r[2] = (int32_t) (p - shift);
            }
            const bool grid = im.nx > 0 && im.ny > 0;
            const int64_t cols = grid ? im.nx : im.n, rows = grid ? im.ny : 1;
            if (cols * rows != im.n)
                return fail(err, "a picture of " + std::to_string(im.n) + " rows has a grid of " + std::to_string(cols) + " x " +
                                     std::to_string(rows));
            const int64_t s0 = im.start - shift;   // vLLM's start_index at the picture
            for (int64_t k = 0; k < im.n; ++k, ++p) {
                int32_t* r = tab + (size_t) (p - b) * 3;
                r[0] = (int32_t) s0;
                r[1] = (int32_t) (s0 + k / cols);
                r[2] = (int32_t) (s0 + k % cols);
            }
            shift += im.n - std::max(rows, cols);
        }
    }
    const int64_t hdr[Hy4MPos::kHeader] = {b, e, shift, 0};
    std::memcpy(h_mpos.data(), hdr, sizeof hdr);
    const size_t bytes = h_mpos.size() * sizeof(int32_t);
    for (auto& gp : stages) {   // on the stage's stream: after whatever it still runs of the last request
        set_dev(gp->dev);
        if (cudaMemcpyAsync(gp->hy_mpos, h_mpos.data(), bytes, cudaMemcpyHostToDevice, gp->s) != cudaSuccess ||
            cudaStreamSynchronize(gp->s) != cudaSuccess)
            return fail(err, "the M-RoPE table's copy to GPU " + std::to_string(gp->dev));
    }
    hy_mpos_live = !img.empty();
    // STRATA_HY4_MPOS_DUMP=FILE (a check's): the table as sent, the header's four int64 then the triples
    if (const char* dump = std::getenv("STRATA_HY4_MPOS_DUMP"); dump && *dump && hy_mpos_live)
        if (FILE* f = std::fopen(dump, "wb")) {
            std::fwrite(h_mpos.data(), 1, bytes, f);
            std::fclose(f);
        }
    if (opt.verbose && hy_mpos_live)
        fprintf(stderr, "strata-x: M-RoPE for %zu picture(s): positions %lld .. %lld from the table, %lld fewer after\n", img.size(),
                (long long) b, (long long) e - 1, (long long) shift);
    return true;
}

// ------------------------------------------------------------------ the Engram

bool Engine::Impl::load_engram_hy4(std::string& err) {
    if (!hp.engram) return true;
    size_t sh = 0;
    const strata::TensorInfo* ti = gm->find("per_layer_token_embd.weight", &sh);
    if (!ti) return fail(err, "hy_v4_flash: Engram keys but no per_layer_token_embd.weight");
    // ggml type 43 here is mainline ggml's F8_E4M3: one byte a value, no scale (the converter scales each BF16 row to 448 and
    // drops the scale; llama.cpp reads it so) - the Q4_K_M, Q3_K and IQ2_AH sets' table.  The SPARK-4Bit sets' is Q8_0 (type 8:
    // blocks of 32, an FP16 scale and 32 int8 codes).  The rows end where the next tensor of the shard starts, or at the file's
    // end.
    const strata::GgufFile& g = gm->shard(sh);
    const int64_t C = (hp.eg_ngram - 1) * hp.eg_heads, D = hp.eg_dim;
    const int fmt = ti->type == 43 ? 0 : ti->type == 8 && D % 32 == 0 ? 1 : -1;
    if (fmt < 0 || ti->shape.size() != 2 || (int64_t) ti->shape[0] != D)
        return fail(err, "per_layer_token_embd: not F8_E4M3 or Q8_0 rows of " + std::to_string(D) + " (ggml type " +
                             std::to_string(ti->type) + ")");
    const uint64_t RB = fmt == 0 ? (uint64_t) D : (uint64_t) (D / 32) * 34;
    const uint64_t rows = ti->shape[1], data = g.data_start() + ti->offset;
    if (data + rows * RB > g.file_size()) return fail(err, "per_layer_token_embd: its rows run past the end of " + g.path());
    for (int64_t j = 0; j < (int64_t) hp.eg_layers.size(); ++j)
        for (int64_t c = 0; c < C; ++c) {
            const int64_t pr = hp.eg_primes[(size_t) (j * C + c)], of = hp.eg_offsets[(size_t) (j * C + c)];
            if (pr < 2 || of < 0 || (uint64_t) (of + pr) > rows) return fail(err, "hy_v4_flash: an Engram bucket range leaves the table");
        }
    const int fd = open(g.path().c_str(), O_RDONLY);
    if (fd < 0) return fail(err, "cannot open " + g.path());
    struct stat sb {};
    if (fstat(fd, &sb) != 0 || (uint64_t) sb.st_size < data + rows * RB) {
        close(fd);
        return fail(err, g.path() + ": shorter than its Engram table");
    }
    void* p = mmap(nullptr, (size_t) sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return fail(err, "cannot map " + g.path());
    madvise(p, (size_t) sb.st_size, MADV_RANDOM);   // single rows: no read-ahead around them
    rec_maps.push_back({p, (size_t) sb.st_size});
    auto e = std::make_unique<Hy4Eg>();
    e->L = (int) hp.eg_layers.size();
    e->cols = (int) C;
    e->dim = (int) D;
    e->fmt = fmt;
    e->rb = (int) RB;
    e->table = static_cast<const uint8_t*>(p) + data;
    e->rows = rows;
    e->ids.assign((size_t) opt.n_ctx, Hy4Eg::kDead);
    e->cap = kTmax;
    if (cudaMallocHost(&e->h_rows, (size_t) e->L * kTmax * C * RB) != cudaSuccess ||
        cudaMallocHost(&e->h_valid, (size_t) kTmax * C * sizeof(float)) != cudaSuccess)
        return fail(err, "the Engram's pinned staging");
    {   // moex (2026-11-19): the row cache (Hy4Eg::RowCache)
        const char* v = std::getenv("STRATA_HY4_EG_CACHE_MIB");
        const int64_t mib = v && *v ? std::max<int64_t>(0, std::atoll(v)) : 2048;
        e->cache.init((int) RB, (uint64_t) mib << 20);
    }
    heg = std::move(e);
    if (opt.verbose)
        fprintf(stderr, "strata-x: Engram: %d layers, %lld columns of %lld %s values, %.2f GB of rows read in place from %s; "
                        "a row cache of %.0f MiB (%llu rows)\n",
                heg->L, (long long) C, (long long) D, fmt == 0 ? "FP8" : "Q8_0", (double) rows * RB / 1e9, g.path().c_str(),
                (double) heg->cache.n_sets * 4 * (8 + RB) / 1048576.0, (unsigned long long) heg->cache.n_sets * 4);
    return true;
}

void Engine::Impl::Hy4Eg::RowCache::init(int row_bytes, uint64_t bytes) {
    rb = row_bytes;
    const uint64_t per_set = 4 * (8 + (uint64_t) row_bytes) + 2;
    n_sets = row_bytes > 0 ? bytes / per_set : 0;
    if (n_sets == 0) return;
    key.reset(new uint64_t[n_sets * 4]());                       // zeros: every way empty
    data.reset(new uint8_t[n_sets * 4 * (uint64_t) row_bytes]);   // (untouched until a row lands there)
    lk.reset(new Stripe[kStripes]);
    victim.reset(new uint8_t[n_sets]());
}

namespace {
inline uint64_t eg_set_of(uint64_t row, uint64_t n_sets) { return (row * 0x9E3779B97F4A7C15ull >> 17) % n_sets; }

}  // namespace

bool Engine::Impl::Hy4Eg::RowCache::get(uint64_t row, uint8_t* out) {
    const uint64_t s = eg_set_of(row, n_sets);
    std::lock_guard<std::mutex> g(lk[s % kStripes].mu);
    for (int w = 0; w < 4; ++w)
        if (key[s * 4 + w] == row + 1) {
            std::memcpy(out, data.get() + (s * 4 + w) * (uint64_t) rb, (size_t) rb);
            hits.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
    misses.fetch_add(1, std::memory_order_relaxed);
    return false;
}

void Engine::Impl::Hy4Eg::RowCache::put(uint64_t row, const uint8_t* in) {
    const uint64_t s = eg_set_of(row, n_sets);
    std::lock_guard<std::mutex> g(lk[s % kStripes].mu);
    int w = -1;
    for (int i = 0; i < 4 && w < 0; ++i)
        if (key[s * 4 + i] == row + 1 || key[s * 4 + i] == 0) w = i;   // (two threads may bring the same row: one way)
    if (w < 0) {
        w = victim[s] & 3;
        victim[s] = (uint8_t) (w + 1);
    }
    std::memcpy(data.get() + (s * 4 + w) * (uint64_t) rb, in, (size_t) rb);
    key[s * 4 + w] = row + 1;
}

bool Stage::hy_eg_grow(int rows, std::string& err) {
    // the decode windows' rows: kTmax a slot, allocated once (the captured graphs read this buffer at its place)
    if (!m.heg || hy_eg_rows) return true;
    const Engine::Impl::Hy4Eg& e = *m.heg;
    cudaSetDevice(dev);
    if (!(hy_eg_rows = dalloc<uint8_t>((size_t) e.L * kTmax * e.cols * e.rb, err)) ||
        !(hy_eg_valid = dalloc<float>((size_t) kTmax * e.cols, err)))
        return false;
    hy_eg_cap = kTmax;
    (void) rows;
    return true;
}

bool Stage::hy_egp_grow(int rows, std::string& err) {
    // a prompt chunk's rows (this stage's layers' slots alone, slot j at j - hy_egp_j0), on the device and their pinned
    // staging - grown to the chunk, the old buffers freed
    if (!m.heg || rows <= hy_egp_cap) return true;
    const Engine::Impl::Hy4Eg& e = *m.heg;
    int j0 = e.L, j1 = 0;
    for (int j = 0; j < e.L; ++j)
        if (hp.eg_layers[(size_t) j] >= lb && hp.eg_layers[(size_t) j] < le) j0 = std::min(j0, j), j1 = std::max(j1, j + 1);
    if (j0 >= j1) return true;
    cudaSetDevice(dev);
    cudaStreamSynchronize(s);   // (the old buffers' last reads are done)
    host_free(h_egp_rows);
    host_free(h_egp_valid);
    dfree(hy_egp_rows);
    dfree(hy_egp_valid);
    h_egp_rows = nullptr;
    h_egp_valid = nullptr;
    hy_egp_rows = nullptr;
    hy_egp_valid = nullptr;
    hy_egp_cap = 0;
    hy_egp_j0 = j0;
    hy_egp_nj = j1 - j0;
    const size_t rb = (size_t) hy_egp_nj * rows * e.cols * e.rb, vb = (size_t) rows * e.cols * sizeof(float);
    if (!(hy_egp_rows = dalloc<uint8_t>(rb, err)) || !(hy_egp_valid = dalloc<float>((size_t) rows * e.cols, err))) return false;
    if (!(h_egp_rows = static_cast<uint8_t*>(host_buf(rb))) || !(h_egp_valid = static_cast<float*>(host_buf(vb))))
        return m.fail(err, "the Engram's chunk staging (out of memory)");
    hy_egp_cap = rows;
    return true;
}

void Engine::Impl::hy4_eg_note(const int32_t* toks, int64_t pos0, int T) {
    Hy4Eg& e = *heg;
    for (int t = 0; t < T; ++t) {
        const int64_t p = pos0 + t;
        if (p < 0 || p >= (int64_t) e.ids.size()) continue;
        const int32_t tk = toks[t];
        e.ids[(size_t) p] = image_row(p) ? Hy4Eg::kDead
                            : tk >= 0 && tk < (int32_t) hp.eg_token_map.size() ? hp.eg_token_map[(size_t) tk] : Hy4Eg::kDead;
    }
}

/// llama.cpp's engram_hash with vLLM's look-back (deepseek_v41/common/engram.py _hash_ids_kernel): the compressed ids of the
/// ngram positions ending at p - a position before the start, or from a DEAD position (an image row) back, takes the pad id;
/// per Engram layer j the rolling xor of id * multiplier, per n-gram size g and head h its value mod the head's prime plus the
/// head's offset into the one table.  A dead position (or the pad token itself, vLLM's no_hash) adds nothing.  The rows of
/// the slots [j0, j1) of positions pos0 .. pos0 + T - 1 go to rows_out ([j - j0][cap][cols][the table's row bytes]), the
/// validity to valid_out ([cap][cols]); their ids must be noted (hy4_eg_note), and the two positions before them
bool Engine::Impl::hy4_eg_rows(const int32_t* toks, int64_t pos0, int T, int j0, int j1, uint8_t* rows_out, float* valid_out, int cap,
                               std::string& err, int max_threads) {
    Hy4Eg& e = *heg;
    if (T > cap) return fail(err, "Engram rows for " + std::to_string(T) + " positions into a buffer of " + std::to_string(cap));
    if (pos0 < 0 || pos0 + T > (int64_t) e.ids.size()) return fail(err, "Engram rows past the context");
    const int M = (int) hp.eg_ngram, H = (int) hp.eg_heads, C = e.cols;
    const size_t RB = (size_t) e.rb;
    // A decode window's rows: every row's index first, then each page they lie on asked for at once (MADV_WILLNEED queues
    // the drive's reads together), then the copies.  Copied as found, each cold page's fault read one page after another:
    // a token's 64 rows took ~7 ms on the full Q4_K_M set (the table's 63 GB mostly outside the page cache).  The same bytes
    // either way; STRATA_HY4_EG_WILLNEED=0 copies as found.  (A prompt chunk's threads fault in parallel already.)
    static const bool willneed = [] { const char* v = std::getenv("STRATA_HY4_EG_WILLNEED"); return !(v && v[0] == '0'); }();
    // moex (2026-11-19): a prompt chunk's too, opt-in (STRATA_HY4_EG_WILLNEED_PROMPT=1): each thread asks for its rows' pages
    // first, then all copy - cold, an 8K read 17.8 -> 16.7 s.  Not by default: this drive (99% full) has slow spells either
    // way, and in one the chunk's ~400K asked pages fill its whole queue (~1,040 requests, ~1 s a read: any other program's
    // read waits behind them) where the 32 threads' faults keep ~170 there (31 ms a read) - the same ~60 s stall either way
    static const bool willneed_p = [] { const char* v = std::getenv("STRATA_HY4_EG_WILLNEED_PROMPT"); return v && v[0] == '1'; }();
    const bool defer = willneed && (T < 64 || willneed_p);
    std::vector<uint64_t> found;
    if (defer) found.assign((size_t) (j1 - j0) * T * C, 0);
    std::atomic<bool> bad{false};
    // moex (2026-11-19): the row cache first (Hy4Eg::RowCache) - a hit is copied at once and asks the drive for nothing; a miss
    // is copied out of the mapping and kept
    Hy4Eg::RowCache& rc = e.cache;
    constexpr uint64_t kHit = ~(uint64_t) 0;   // a found[] entry already copied from the cache
    auto take = [&](uint8_t* dst, uint64_t row) {
        if (rc.on() && rc.get(row, dst)) return;
        std::memcpy(dst, e.table + row * RB, RB);
        if (rc.on()) rc.put(row, dst);
    };
    auto dst_of = [&](int j, int t, int c) { return rows_out + (((size_t) (j - j0) * cap + t) * C + c) * RB; };
    auto fi = [&](int j, int t, int c) { return ((size_t) (j - j0) * T + t) * C + c; };
    static const uintptr_t pg = (uintptr_t) sysconf(_SC_PAGESIZE);
    // the rows of positions [t0, t1): the cache's copied, the others' pages asked for (MADV_WILLNEED queues their reads together)
    auto ask = [&](int t0, int t1) {
        for (int j = j0; j < j1; ++j)
            for (int t = t0; t < t1; ++t)
                for (int c = 0; c < C; ++c) {
                    uint64_t& row = found[fi(j, t, c)];
                    if (rc.on() && rc.get(row, dst_of(j, t, c))) {
                        row = kHit;
                        continue;
                    }
                    const uintptr_t a = (uintptr_t) (e.table + row * RB), p0 = a & ~(pg - 1), p1 = (a + RB - 1) & ~(pg - 1);
                    madvise((void*) p0, (size_t) (p1 - p0 + pg), MADV_WILLNEED);
                }
    };
    auto copy = [&](int t0, int t1) {   // ... then the others out of the mapping, kept in the cache
        for (int j = j0; j < j1; ++j)
            for (int t = t0; t < t1; ++t)
                for (int c = 0; c < C; ++c) {
                    const uint64_t row = found[fi(j, t, c)];
                    if (row == kHit) continue;
                    uint8_t* dst = dst_of(j, t, c);
                    std::memcpy(dst, e.table + row * RB, RB);
                    if (rc.on()) rc.put(row, dst);
                }
    };
    auto work = [&](int t0, int t1) {
        for (int t = t0; t < t1 && !bad.load(std::memory_order_relaxed); ++t) {
            const int64_t p = pos0 + t;
            int64_t slot[16];
            bool blocked = false;
            for (int sh = 0; sh < M; ++sh) {
                const int32_t src = p >= sh ? e.ids[(size_t) (p - sh)] : Hy4Eg::kDead;
                blocked = blocked || p < sh || src == Hy4Eg::kDead;
                slot[sh] = blocked ? hp.eg_pad : src;
            }
            const bool dead = e.ids[(size_t) p] == Hy4Eg::kDead || (eg_pad_token >= 0 && toks[t] == eg_pad_token);
            for (int c = 0; c < C; ++c) valid_out[(size_t) t * C + c] = dead ? 0.0f : 1.0f;
            for (int j = j0; j < j1; ++j) {
                const int64_t* mul = &hp.eg_mul[(size_t) j * M];
                int64_t rolling = slot[0] * mul[0];
                for (int gi = 1; gi < M; ++gi) {
                    rolling ^= slot[gi] * mul[gi];
                    for (int h = 0; h < H; ++h) {
                        const int c = (gi - 1) * H + h;
                        const int64_t pr = hp.eg_primes[(size_t) j * C + c];
                        int64_t r = rolling % pr;
                        if (r < 0) r += pr;
                        const int64_t row = hp.eg_offsets[(size_t) j * C + c] + r;
                        if (row < 0 || (uint64_t) row >= e.rows) {
                            bad = true;
                            return;
                        }
                        if (defer) found[((size_t) (j - j0) * T + t) * C + c] = (uint64_t) row;
                        else   // the row out of the cache, or the mapping (a page fault reads it from the drive the first time)
                            take(rows_out + (((size_t) (j - j0) * cap + t) * C + c) * RB, (uint64_t) row);
                    }
                }
            }
        }
    };
    if (T < 64) {
        work(0, T);
        if (defer && !bad) {
            ask(0, T);
            copy(0, T);
        }
    } else {   // a prompt chunk: its rows' page faults in parallel (the drive takes many requests at once)
        const int nt = std::min(std::max(1, max_threads), std::max(1, T / 16));
        auto each = [&](auto&& fn) {
            std::vector<std::thread> th;
            for (int w = 0; w < nt; ++w) th.emplace_back(fn, (int) ((int64_t) T * w / nt), (int) ((int64_t) T * (w + 1) / nt));
            for (auto& x : th) x.join();
        };
        static const bool trace = std::getenv("STRATA_HY4_EG_TRACE") != nullptr;   // a check's timing of each step
        const auto tq0 = std::chrono::steady_clock::now();
        const uint64_t h0 = rc.hits.load(), m0 = rc.misses.load();
        if (!defer) {
            each(work);
        } else {   // each thread: its rows' indices, the cache's copied and the others' pages asked for; then every thread copies
            each([&](int t0, int t1) {
                work(t0, t1);
                if (!bad.load(std::memory_order_relaxed)) ask(t0, t1);
            });
            const auto tq1 = std::chrono::steady_clock::now();
            if (!bad) each(copy);
            if (trace)
                fprintf(stderr, "strata-x: Engram rows: %d positions, slots %d-%d, %d threads: index + ask %.1f ms, copy %.1f ms; "
                                "cache hits %llu, misses %llu\n", T, j0, j1 - 1, nt,
                        std::chrono::duration<double, std::milli>(tq1 - tq0).count(),
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tq1).count(),
                        (unsigned long long) (rc.hits.load() - h0), (unsigned long long) (rc.misses.load() - m0));
        }
    }
    if (bad) return fail(err, "an Engram row out of range");
    return true;
}

bool Engine::Impl::hy4_eg_window(const int32_t* toks, int64_t pos0, int T, std::string& err) {
    if (!heg) return true;
    Hy4Eg& e = *heg;
    hy4_eg_note(toks, pos0, T);
    for (auto& gp : stages) cudaStreamSynchronize(gp->s);   // the staging may still be read by the last window's copies
    const auto t_rows = std::chrono::steady_clock::now();
    if (!hy4_eg_rows(toks, pos0, T, 0, e.L, e.h_rows, e.h_valid, kTmax, err)) return false;
    stages[0]->st.ms_eg += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_rows).count();
    const int C = e.cols;
    const size_t RB = (size_t) e.rb;
    for (auto& gp : stages) {
        bool has = false;
        for (int64_t il : hp.eg_layers) has = has || (il >= gp->lb && il < gp->le);
        if (!has) continue;
        set_dev(gp->dev);
        if (!gp->hy_eg_grow(kTmax, err)) return false;
        if (cudaMemcpyAsync(gp->hy_eg_rows, e.h_rows, (size_t) e.L * kTmax * C * RB, cudaMemcpyHostToDevice, gp->s) != cudaSuccess ||
            cudaMemcpyAsync(gp->hy_eg_valid, e.h_valid, (size_t) T * C * sizeof(float), cudaMemcpyHostToDevice, gp->s) != cudaSuccess)
            return fail(err, "the Engram rows' upload failed");
    }
    return true;
}

bool Engine::Impl::hy4_eg_chunk(Stage& g, const int32_t* toks, int64_t pos0, int T, std::string& err) {
    if (!heg) return true;
    Hy4Eg& e = *heg;
    int j0 = e.L, j1 = 0;   // this stage's slots
    for (int j = 0; j < e.L; ++j)
        if (hp.eg_layers[(size_t) j] >= g.lb && hp.eg_layers[(size_t) j] < g.le) j0 = std::min(j0, j), j1 = std::max(j1, j + 1);
    hy4_eg_note(toks, pos0, T);   // (every stage notes its chunk's: the same ids, at positions no other chunk in flight reads)
    if (j0 >= j1) return true;
    {   // a gathering ahead of this chunk's rows into the row cache, if one was started (below): the first stage's own, started
        // with its chunk before; a later stage's, started with the first stage's run of this chunk
        std::future<void> f;
        {
            std::lock_guard<std::mutex> lk(eg_pf_mu);
            auto& mp = g.di == 0 ? eg_pf0 : eg_pf;
            if (auto it = mp.find(pos0); it != mp.end()) {
                f = std::move(it->second);
                mp.erase(it);
            }
        }
        if (f.valid()) f.wait();
    }
    if (!g.hy_egp_grow(T, err)) return false;
    cudaSetDevice(g.dev);
    cudaStreamSynchronize(g.s);   // the staging may still be read by this stage's last chunk's copies
    if (j0 != g.hy_egp_j0 || j1 - j0 != g.hy_egp_nj) return fail(err, "the Engram's chunk buffers hold other slots");
    if (!hy4_eg_rows(toks, pos0, T, j0, j1, g.h_egp_rows, g.h_egp_valid, g.hy_egp_cap, err)) return false;
    const size_t slot = (size_t) g.hy_egp_cap * e.cols * e.rb;
    if (cudaMemcpyAsync(g.hy_egp_rows, g.h_egp_rows, (size_t) (j1 - j0) * slot, cudaMemcpyHostToDevice, g.s) != cudaSuccess ||
        cudaMemcpyAsync(g.hy_egp_valid, g.h_egp_valid, (size_t) T * e.cols * sizeof(float), cudaMemcpyHostToDevice, g.s) != cudaSuccess)
        return fail(err, "the Engram rows' upload failed");
    // moex (2026-11-19): the later stages' rows of this chunk gathered now, into the row cache, while this stage runs the chunk
    // (their reads waited for it before: a 131K-row gather at GPU 1's start, 0.4-1.8 s healthy and ~60 s in the drive's slow
    // spells); the later stage's own gather waits for them, then copies them out of the cache - the same bytes.
    // STRATA_HY4_EG_AHEAD=0: no gathering ahead
    static const bool ahead = [] { const char* v = std::getenv("STRATA_HY4_EG_AHEAD"); return !(v && v[0] == '0'); }();
    static const int ahead_nt = [] { const char* v = std::getenv("STRATA_HY4_EG_AHEAD_THREADS"); return v ? std::max(1, std::atoi(v)) : 32; }();
    // ... and this stage's own rows of the chunk after this one (the prompt loop's hint, eg_next_*: the plan's length, an upper
    // bound), so its next gathering finds them in the cache while this chunk's layers run: of a call's chunks only the first
    // waits for the drive then.  Their ids are noted here (the same prompt tokens their own chunk notes again)
    if (ahead && g.di == 0 && e.cache.on() && g.eg_next_T > 0 && g.eg_next_toks) {
        const int Tn = g.eg_next_T;
        const int64_t pn = g.eg_next_pos;
        hy4_eg_note(g.eg_next_toks, pn, Tn);
        std::vector<int32_t> tk(g.eg_next_toks, g.eg_next_toks + Tn);
        std::lock_guard<std::mutex> lk(eg_pf_mu);
        eg_pf0[pn] = std::async(std::launch::async, [this, tk = std::move(tk), pn, Tn, j0, j1] {
            const Hy4Eg& h = *heg;
            std::vector<uint8_t> rows((size_t) (j1 - j0) * Tn * h.cols * h.rb);
            std::vector<float> valid((size_t) Tn * h.cols);
            std::string e2;
            hy4_eg_rows(tk.data(), pn, Tn, j0, j1, rows.data(), valid.data(), Tn, e2, ahead_nt);
        });
    }
    if (ahead && g.di == 0 && e.cache.on() && stages.size() > 1) {
        int k0 = e.L, k1 = 0;
        for (int j = 0; j < e.L; ++j)
            if (hp.eg_layers[(size_t) j] >= g.le) k0 = std::min(k0, j), k1 = std::max(k1, j + 1);
        if (k0 < k1) {
            std::vector<int32_t> tk(toks, toks + T);   // (the task outlives no buffer of the call: its own copy)
            std::lock_guard<std::mutex> lk(eg_pf_mu);
            eg_pf[pos0] = std::async(std::launch::async, [this, tk = std::move(tk), pos0, T, k0, k1] {
                const Hy4Eg& h = *heg;
                std::vector<uint8_t> rows((size_t) (k1 - k0) * T * h.cols * h.rb);
                std::vector<float> valid((size_t) T * h.cols);
                std::string e2;
                // (STRATA_HY4_EG_AHEAD_THREADS: its threads; 32 as the gathers' own - 4 of 4 cold 16K reads 30-32 s, 8 no better)
                hy4_eg_rows(tk.data(), pos0, T, k0, k1, rows.data(), valid.data(), T, e2, ahead_nt);
            });
        }
    }
    return true;
}

void Stage::hy4_engram(int il, int j, int r0, int T, const uint8_t* rows_base, const float* valid_base, int cap) {
    const Layer& l = L[(size_t) il];
    const Engine::Impl::Hy4Eg& e = *m.heg;
    const int C = e.cols, D = e.dim, n = (int) hp.n_embd;
    const uint8_t* rows = rows_base + ((size_t) j * cap + r0) * C * (size_t) e.rb;
    const float* valid = valid_base + (size_t) r0 * C;
    hy4_eg_rows_w(rows, e.fmt, l.hy_eg_lnorm.f(), valid, hegr, C, D, kEgEps, T, s);
    probe("engram_rows_normed", il, hegr, (int64_t) C * D);
    quant(hegr, xq, C * D, T);
    pj(l.hy_eg_kv, hegr, xq, hegkv, T);
    float* Rr = R + (size_t) r0 * hp.hc * n;
    // (a check's switch, STRATA_HY4_EG_QK_EPS=bf16: norm_key / norm_query with vLLM's epsilon - nn.RMSNorm's default over BF16
    // tensors, 2^-7 - where llama.cpp takes the F32 one)
    static const float eps_qk = [] {
        const char* v = std::getenv("STRATA_HY4_EG_QK_EPS");
        if (!v || !*v) return kEgEps;
        return std::strcmp(v, "bf16") == 0 ? 0.0078125f : std::strtof(v, nullptr);
    }();
    hy4_eg_apply_w(Rr, hegkv, l.hy_eg_kvb.f(), l.hy_eg_gate.f(), l.hy_eg_pg.f(), l.hy_eg_hcs.f(), valid, C, n, kEgEps, T, s, eps_qk);
    probe("engram_out", il, Rr, n, hp.hc);
}

// ------------------------------------------------------------------ the attention

void Stage::hy4_attn(int il, int64_t pos0, int T, const float* in, float* out_r) {
    const Layer& l = L[(size_t) il];
    const int n = (int) hp.n_embd, H = (int) hp.H_of(il), Lq = (int) hp.q_lora, Dk = (int) hp.k_mla, Dv = (int) hp.v_mla;
    const int Dr = (int) hp.n_rot, Dn = Dk - Dr, Lkv = (int) hp.kv_lora, IH = (int) hp.idx_heads, ID = (int) hp.idx_dim;
    const int KP = (int) hp.kpool, n_top = m.hy4_n_top();
    const bool swa = hp.is_swa(il);
    const float base = swa ? hp.rope_base_swa : hp.rope_base;
    // the projections of the layer's input (its Q8_1 columns in xq until heads_mm below takes the buffer)
    quant(in, xq, n, T);
    pj(l.q_a, in, xq, qr, T);
    pj(l.kv_a, in, xq, hkv, T);
    pj(l.attn_gate, in, xq, hgate, T);
    if (!swa) {
        pj(l.idx_k, in, xq, ikw, T);
        if (l.idx_proj.type == 0) gemv_dt(l.idx_proj, in, w32, n, IH, T);   // F32 (BF16 and the SPARK sets' Q8_0: pj)
        else pj(l.idx_proj, in, xq, w32, T);
    }
    rms_norm(qr, l.q_a_norm.f(), qr, T, Lq, hp.rms_eps, s);
    probe("q_resid", il, qr, Lq);
    quant(qr, xq2, Lq, T);
    pj(l.q_b, qr, xq2, q, T);
    if (!swa) {
        pj(l.idx_q_b, qr, xq2, iq, T);
        rms_norm(ikw, l.idx_k_norm.f(), ikw, T, ID, hp.rms_eps, s);
        probe("indexer_k", il, ikw, ID);
        quant(ikw, xq2, ID, T);   // (q_resid's columns are done with)
        pj(l.kp_value, ikw, xq2, hiv, T);
        pj(l.idx_gate, ikw, xq2, igw, T);
        probe("indexer_gate", il, igw, ID);
    }
    // the cache row: the normed latent and the rotated k_pe, stored once the probes have had them
    hy4_kv_row_w(hkv, l.kv_a_norm.f(), hp.rms_eps, base, nullptr, l.kv_rows, pos0, T, s, gpos, hy_mpos);
    probe("kv_cmpr", il, hkv, Lkv);
    probe("k_pe", il, hkv + Lkv, Dr);
    hy4_store_kv_w(hkv, l.kv_cache, l.kv_rows, pos0, T, s, gpos);
    // q: RoPE on each head's last 64, the nope part absorbed through k_b into the latent
    hy4_rope_w(q, H * Dk, 0, H, Dk, Dn, Dr, base, pos0, T, s, gpos, hy_mpos);
    hy4_q_split(q, q_hm, hqf, H, T, s);
    heads_mm(l.k_b, q_hm, qabs_hm, H, Dn, Lkv, T);
    hy4_q_join(qabs_hm, hqf, H, T, s);
    const int32_t* cl = nullptr;
    if (!swa) {   // the indexer: its query's RoPE, this window's values and gates into the rings, the pools that complete here
        hy4_rope_w(iq, IH * ID, 0, IH, ID, ID - Dr, Dr, base, pos0, T, s, gpos, hy_mpos);
        probe("indexer_q", il, iq, ID, IH);
        dsa_ring_store_w(hiv, l.ik_ring, m.ring_rows, pos0, ID, T, s, gpos);
        dsa_ring_store_w(igw, l.ig_ring, m.ring_rows, pos0, ID, T, s, gpos);
        hy4_kpool_w(l.ik_ring, l.ig_ring, m.ring_rows, l.idx_ape.f(), l.kp_norm.f(), hp.rms_eps, l.pooled, base, pos0, T, KP, s, gpos,
                    hy_mpos);
        if (m.rec && (pos0 + 1) % KP == 0) probe("indexer_pool_k_new", il, l.pooled + (pos0 / KP) * ID, ID);
        scale_f32(w32, T * IH, 1.0f / std::sqrt((float) (ID * IH)), s);
        probe("indexer_weights", il, w32, IH);
        hy4_scores_w(iq, l.pooled, w32, score, pos0, T, KP, m.max_pool, s, gpos);
        if (m.rec && pos0 / KP > 0) probe("indexer_score", il, score, pos0 / KP);
        hy4_select_w(score, m.max_pool, n_top, KP, pos0, T, cells, m.max_cells, s, gpos);
        dfly_capture_cells(il, cells, pos0, T);   // DFly2: its target layers' cells (a window's rows, by position)
        cl = cells;
    }
    const int max_l = swa ? (int) std::min<int64_t>(hp.n_swa, opt.n_ctx) : n_top * KP + KP;
    const int chunks = T <= kTmax && hpart ? hy4_attend_chunks(max_l) : 1;
    hy4_attend_w(hqf, l.kv_cache, l.kv_rows, cl, m.max_cells, n_top, KP, (int) hp.n_swa, l.sinks.f(), 1.0f / std::sqrt((float) Dk),
                 ctx, hpart, chunks, H, pos0, T, s, gpos);
    probe("kqv", il, ctx, Lkv, 1, H);
    permute_abc(ctx, ctx_hm, T, H, Lkv, s);
    heads_mm(l.v_b, ctx_hm, o2_hm, H, Lkv, Dv, T);
    permute_abc(o2_hm, o2, H, T, Dv, s);
    probe("attn_kqv", il, o2, (int64_t) H * Dv);
    hy4_gate_mul(o2, hgate, (int64_t) T * H * Dv, s);
    probe("attn_gated", il, o2, (int64_t) H * Dv);
    quant(o2, xq, H * Dv, T);
    pj(l.wo, o2, xq, out_r, T);
    rms_norm(out_r, l.attn_post_norm.f(), out_r, T, n, hp.rms_eps, s);
    probe("attn_out", il, out_r, n);
}

// ------------------------------------------------------------------ the latent MoE's ends

void Stage::hy4_moe_in(int il, int T) {
    const Layer& l = L[(size_t) il];
    const int n = (int) hp.n_embd, nx = (int) hp.ex();
    if (!l.r_down) return;   // (no latent: the experts read cur)
    for (int b0 = 0; b0 < T; b0 += SB) {   // (one sub-batch in a decode window)
        const int tn = std::min(SB, T - b0);
        quant(cur + (size_t) b0 * n, xq, n, tn);
        pj(l.r_down, cur + (size_t) b0 * n, xq, hxl + (size_t) b0 * nx, tn);
    }
}

void Stage::hy4_moe_out(int il, int T, float* out_r) {
    const Layer& l = L[(size_t) il];
    const int n = (int) hp.n_embd, nx = (int) hp.ex();
    probe("ffn_moe_out", il, hml, nx);
    if (l.latent_norm) rms_norm(hml, l.latent_norm.f(), hml, T, nx, hp.rms_eps, s);   // moex (ERNIE-5.5): the latent's norm
    // moex (the ERNIE-5.5 hill-climb): a BF16 or FP8 routed_up reads hml itself (pj): its Q8_1 copy is not made
    const bool q_up = !(l.r_up.type == GGML_TYPE_BF16 || l.r_up.type == kDtFP8);
    for (int b0 = 0; b0 < T; b0 += SB) {
        const int tn = std::min(SB, T - b0);
        if (q_up) quant(hml + (size_t) b0 * nx, xq2, nx, tn);
        // (llama.cpp's name for this, ffn_moe_up, is also its experts' up: not probed)
        pj(l.r_up, hml + (size_t) b0 * nx, xq2, out_r + (size_t) b0 * n, tn);
    }
    probe("ffn_shexp", il, shared, n);
    add_f32(out_r, shared, out_r, T * n, s);
    if (l.ffn_post_norm) rms_norm(out_r, l.ffn_post_norm.f(), out_r, T, n, hp.rms_eps, s);   // (Hy4-flash's; ERNIE-5.5 has none)
    probe("ffn_out", il, out_r, n);
}

}  // namespace strata::x
