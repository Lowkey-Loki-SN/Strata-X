// src/x/dsv4.cpp - moex: DeepSeek-V4-Flash (llama.cpp's deepseek4) on the GLM engine's machinery - its hyperparameters,
// tensors and decode state, and the attention half of a layer.  Everything else is shared with GLM-5.3: mHC (the same
// Sinkhorn hyper-connections, stage.cpp's mhc_half), the routed experts (the VRAM tier, the CPU pool, the prompt path's
// MMQ groups), the decode graphs, the prompt path, the session.  What DeepSeek-V4 adds is here and in a few dispatches:
//   * the attention (Stage::dsv4): one shared K = V head of 512 over the raw window of the last 128 positions (an F16 ring)
//     plus, per layer, nothing (ratio 0), the CSA rows the lightning indexer picks (ratio 4) or every HCA row (ratio 128) -
//     the compressors pool each block of positions into one row as its last position arrives; sinks; the attention's output
//     de-roped, then the grouped low-rank output (8 groups of 1024 rows, then wo_b);
//   * the router (Stage::route): sqrt(softplus) scores, the selection bias, normalized weights times 1.5; the first n_hash
//     layers take their ids from a per-token table (tid2eid) instead;
//   * the head (Stage::head): the hyper-connections' learned reduce (output_hc_*), not GLM's mean.
// The operator-level specification is moex/docs/DSV4_SPEC.md (it cites llama.cpp line by line); the kernels are
// src/x/dsv4_kernels.cu, each checked against a CPU transcription of llama.cpp by build/dsv4_kernels_parity.  The probe names
// are llama.cpp's (DSV4_SPEC section 7), so `strata-x --oracle <llama_dump dir>` checks the layer op by op.
#include "engine_impl.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace strata::x {

// ------------------------------------------------------------------ loading

/// deepseek4's keys (llama_model_deepseek4::load_arch_hparams, DSV4_SPEC 1.1-1.2): every one the graph reads; the geometry
/// the kernels take is checked here, so a different file fails at load and not in a kernel.
bool Engine::Impl::load_hparams_dsv4(std::string& err) {
    const strata::GgufFile& m = gm->meta();
    const std::string pre = "deepseek4.";
    auto get = [&](const std::string& k) { return m.get(pre + k); };
    auto u = [&](const char* k, int64_t& dst) -> bool {
        const strata::MetaValue* v = get(k);
        if (!v) return fail(err, "missing " + pre + k);
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
    h.arch = Arch::dsv4;
    int64_t idx_heads = 0, n_shared = 0;
    if (!u("embedding_length", h.n_embd) || !u("block_count", h.n_layer_all) || !u("attention.head_count", h.n_head) ||
        !u("attention.key_length", h.hd_k) || !u("attention.value_length", h.hd_v) || !u("rope.dimension_count", h.n_rot) ||
        !u("attention.q_lora_rank", h.q_lora) || !u("attention.sliding_window", h.n_swa) || !u("expert_count", h.n_expert) ||
        !u("expert_used_count", h.n_used) || !u("expert_feed_forward_length", h.n_ff_exp) ||
        !u("attention.indexer.head_count", idx_heads) || !u("attention.indexer.key_length", h.idx_dim) ||
        !u("attention.indexer.top_k", h.idx_topk) || !u("attention.output_group_count", h.o_groups) ||
        !u("attention.output_lora_rank", h.o_lora) || !u("hyper_connection.count", h.hc) ||
        !u("hyper_connection.sinkhorn_iterations", h.sinkhorn_iters) || !u("hash_layer_count", h.n_hash) ||
        !u("expert_shared_count", n_shared))
        return false;
    h.idx_heads = idx_heads;
    int64_t nextn = 0;
    if (get("nextn_predict_layers")) u("nextn_predict_layers", nextn);
    h.n_layer = h.n_layer_all - nextn;
    h.n_ctx_orig = 0;
    if (get("rope.scaling.original_context_length")) u("rope.scaling.original_context_length", h.n_ctx_orig);
    float factor = 1.0f;
    if (!f("attention.layer_norm_rms_epsilon", h.rms_eps, 0, true) || !f("hyper_connection.epsilon", h.hc_eps, 0, true) ||
        !f("attention.compress_rope_freq_base", h.crope_base, 0, true) || !f("rope.freq_base", h.rope_base, 10000.0f, false) ||
        !f("rope.scaling.factor", factor, 1.0f, false) || !f("rope.scaling.yarn_beta_fast", h.beta_fast, 32.0f, false) ||
        !f("rope.scaling.yarn_beta_slow", h.beta_slow, 1.0f, false) || !f("expert_weights_scale", h.ew_scale, 1.0f, false))
        return false;
    // llama.cpp reads neither beta (the hparams' defaults 32 / 1 apply, DSV4_SPEC 1.1): the file's must be those
    if (h.beta_fast != 32.0f || h.beta_slow != 1.0f) return fail(err, "deepseek4: YaRN betas other than llama.cpp's 32 / 1");
    h.rope_scale = factor > 0.0f ? 1.0f / factor : 1.0f;
    if (h.n_ctx_orig <= 0) {
        if (const strata::MetaValue* v = get("context_length")) h.n_ctx_orig = (int64_t) v->num();
    }
    if (const strata::MetaValue* v = get("rope.scaling.type"); !v || v->s != "yarn")
        return fail(err, "deepseek4: the compressed layers' RoPE is YaRN (rope.scaling.type yarn)");
    h.ew_norm = true;
    if (const strata::MetaValue* v = get("expert_weights_norm")) h.ew_norm = v->u != 0;
    if (!h.ew_norm) return fail(err, "deepseek4: unnormalized expert weights (the router kernel always normalizes)");
    if (const strata::MetaValue* v = get("expert_gating_func"); !v || v->u != 4)
        return fail(err, "deepseek4: expert gating is not sqrt(softplus) (4)");
    if (n_shared != 1) return fail(err, "deepseek4: one shared expert expected");
    if (const strata::MetaValue* v = m.get("tokenizer.ggml.eos_token_id")) h.eos_id = (int64_t) v->num();
    {   // the per-layer compression ratios (at least block_count entries; the trailing ones are unused, DSV4_SPEC 1.2)
        const strata::MetaValue* v = get("attention.compress_ratios");
        if (!v || v->type != strata::MetaType::ARRAY || (int64_t) v->items.size() < h.n_layer_all)
            return fail(err, "deepseek4.attention.compress_ratios is not a per-layer array");
        for (int64_t il = 0; il < h.n_layer_all; ++il) {
            const int r = (int) v->items[(size_t) il].num();
            if (r != 0 && r != 4 && r != 128) return fail(err, "deepseek4: compress ratio " + std::to_string(r) + " (0, 4 or 128)");
            h.cratio.push_back(r);
        }
    }
    auto arr_f = [&](const char* k, std::vector<float>& dst) -> bool {
        const strata::MetaValue* v = get(k);
        if (!v) return fail(err, "missing " + pre + k);
        if (v->type != strata::MetaType::ARRAY) {
            dst.assign((size_t) h.n_layer_all, (float) v->num());
            return true;
        }
        if ((int64_t) v->items.size() < h.n_layer_all) return fail(err, pre + k + " is shorter than block_count");
        for (int64_t il = 0; il < h.n_layer_all; ++il) dst.push_back((float) v->items[(size_t) il].num());
        return true;
    };
    if (!arr_f("swiglu_clamp_exp", h.clamp_exp)) return false;
    if (get("swiglu_clamp_shexp")) {
        if (!arr_f("swiglu_clamp_shexp", h.clamp_sh)) return false;
    } else {
        h.clamp_sh = h.clamp_exp;   // DSV4_SPEC 1.2: the shared expert's limit defaults to the routed experts'
    }
    size_t sh = 0;
    const strata::TensorInfo* te = gm->find("token_embd.weight", &sh);
    if (!te || te->shape.size() < 2) return fail(err, "no token_embd.weight");
    h.n_vocab = (int64_t) te->shape[1];
    h.head_kv.assign((size_t) h.n_layer_all, 1);   // one shared K = V head
    h.n_dense = 0;                                 // every layer routes experts, with a shared expert
    h.shexp = true;
    // the kernels' geometry (dsv4_kernels.hpp): 64 heads of 512 with RoPE on the last 64 dims, a 128-position raw window, the
    // indexer's 64 heads of 128 and its top-512, 8 output groups of 1024 rows, four hyper-connection streams
    if (h.hc != 4) return fail(err, "deepseek4: mHC with hc != 4");
    if (h.n_head != 64 || h.hd_k != 512 || h.hd_v != 512 || h.n_rot != 64 || h.n_swa != 128 || h.idx_heads != 64 ||
        h.idx_dim != 128 || h.idx_topk != 512 || h.o_groups != 8 || h.o_lora != 1024 || h.n_head * h.hd_k % h.o_groups != 0)
        return fail(err, "deepseek4: kernel geometry - 64 heads of 512 (RoPE 64), window 128, indexer 64 x 128 top-512, 8 x 1024 output");
    if (h.n_hash < 0 || h.n_hash > h.n_layer) return fail(err, "deepseek4: hash_layer_count out of range");
    if (nextn != 0) return fail(err, "deepseek4: this engine has no DeepSeek-V4 MTP block yet");
    return true;
}

namespace {
/// BF16 -> F32 is exact (the high half of the float).
void bf16_to_f32(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t b = (uint32_t) src[i] << 16;
        std::memcpy(dst + i, &b, sizeof(float));
    }
}
}  // namespace

/// deepseek4's tensors (llama_model_deepseek4::load_arch_tensors, DSV4_SPEC section 2).  The router's BF16 weight becomes an F32
/// copy on the layer's GPU (exact, 4 MiB a layer), so the router runs through the same F32 GEMV as GLM's and MiMo's.
bool Engine::Impl::bind_layers_dsv4(std::string& err) {
    auto get = [&](const std::string& n, DT& d, bool required) -> bool {
        auto it = t.find(n);
        if (it == t.end()) {
            if (required) return fail(err, "missing tensor " + n);
            return true;
        }
        d = it->second;
        return true;
    };
    auto proj_ok = [&](const DT& d, const std::string& n) -> bool {
        if (d.type == 0) return true;
        if (!kn::native_mmvq_supported(d.type)) return fail(err, n + ": type " + std::to_string(d.type) + " has no GPU kernel");
        return true;
    };
    auto f32 = [&](const DT& d, const std::string& n, int64_t ne0, int64_t ne1) -> bool {
        if (d.type != 0 || d.ne[0] != ne0 || d.ne[1] * d.ne[2] * d.ne[3] != ne1)
            return fail(err, n + ": not F32 [" + std::to_string(ne0) + ", " + std::to_string(ne1) + "]");
        return true;
    };
    auto shape = [&](const DT& d, const std::string& n, int64_t n_in, int64_t n_out) -> bool {
        if (d.n_in() != n_in || d.n_out() != n_out)
            return fail(err, n + ": shape [" + std::to_string(d.n_in()) + ", " + std::to_string(d.n_out()) + "], expected [" +
                                 std::to_string(n_in) + ", " + std::to_string(n_out) + "]");
        return true;
    };
    const int64_t n = hp.n_embd, H = hp.n_head, D = hp.hd_k, ID = hp.idx_dim, IH = hp.idx_heads, E = hp.n_expert;
    if (!get("output_norm.weight", output_norm, true) || !get("output.weight", output, true) ||
        !get("output_hc_fn.weight", out_hc_fn, true) || !get("output_hc_base.weight", out_hc_base, true) ||
        !get("output_hc_scale.weight", out_hc_scale, true))
        return false;
    if (!proj_ok(output, "output.weight") || !f32(out_hc_fn, "output_hc_fn.weight", hp.hc * n, hp.hc) ||
        !f32(out_hc_base, "output_hc_base.weight", hp.hc, 1) || !f32(out_hc_scale, "output_hc_scale.weight", 1, 1))
        return false;
    L.resize((size_t) n_load);
    for (int64_t il = 0; il < n_load; ++il) {
        Layer& l = L[(size_t) il];
        auto nm = [&](const char* sfx) { return "blk." + std::to_string(il) + "." + sfx; };
        const int r = hp.cratio[(size_t) il];
        bool ok = get(nm("attn_norm.weight"), l.attn_norm, true) && get(nm("ffn_norm.weight"), l.ffn_norm, true) &&
                  get(nm("hc_attn_fn.weight"), l.hc_attn_fn, true) && get(nm("hc_attn_base.weight"), l.hc_attn_base, true) &&
                  get(nm("hc_attn_scale.weight"), l.hc_attn_scale, true) && get(nm("hc_ffn_fn.weight"), l.hc_ffn_fn, true) &&
                  get(nm("hc_ffn_base.weight"), l.hc_ffn_base, true) && get(nm("hc_ffn_scale.weight"), l.hc_ffn_scale, true) &&
                  get(nm("attn_q_a.weight"), l.q_a, true) && get(nm("attn_q_a_norm.weight"), l.q_a_norm, true) &&
                  get(nm("attn_q_b.weight"), l.q_b, true) && get(nm("attn_kv.weight"), l.wkv, true) &&
                  get(nm("attn_kv_a_norm.weight"), l.kv_a_norm, true) && get(nm("attn_sinks.weight"), l.sinks, true) &&
                  get(nm("attn_output_a.weight"), l.wo_a, true) && get(nm("attn_output_b.weight"), l.wo_b, true) &&
                  get(nm("ffn_gate_inp.weight"), l.gate_inp, true) && get(nm("exp_probs_b.bias"), l.probs_b, false) &&
                  get(nm("exp_probs_b_vl.bias"), l.probs_b_vl, false) && get(nm("ffn_gate_shexp.weight"), l.sh_gate, true) &&
                  get(nm("ffn_up_shexp.weight"), l.sh_up, true) && get(nm("ffn_down_shexp.weight"), l.sh_down, true);
        if (!ok) return false;
        // the mixers: F32 here (stage.cpp's mhc_half takes an F32 GEMV for them) or a type the GPU kernels take
        if (!proj_ok(l.hc_attn_fn, nm("hc_attn_fn")) || !proj_ok(l.hc_ffn_fn, nm("hc_ffn_fn")) ||
            !shape(l.hc_attn_fn, nm("hc_attn_fn"), hp.hc * n, 24) || !shape(l.hc_ffn_fn, nm("hc_ffn_fn"), hp.hc * n, 24))
            return false;
        for (const auto& [d, what] : {std::pair<const DT*, const char*>{&l.q_a, "attn_q_a"}, {&l.q_b, "attn_q_b"}, {&l.wkv, "attn_kv"},
                                      {&l.wo_a, "attn_output_a"}, {&l.wo_b, "attn_output_b"}, {&l.sh_gate, "ffn_gate_shexp"},
                                      {&l.sh_up, "ffn_up_shexp"}, {&l.sh_down, "ffn_down_shexp"}})
            if (!proj_ok(*d, nm(what))) return false;
        // attn_output_a [n_head * hd / groups, groups * lora]: group g is rows [g lora, (g + 1) lora) (llama.cpp reshapes it
        // to [4096, 1024, 8] at load, DSV4_SPEC 2.2)
        if (!shape(l.q_a, nm("attn_q_a"), n, hp.q_lora) || !shape(l.q_b, nm("attn_q_b"), hp.q_lora, H * D) ||
            !shape(l.wkv, nm("attn_kv"), n, D) || !shape(l.wo_a, nm("attn_output_a"), H * D / hp.o_groups, hp.o_groups * hp.o_lora) ||
            !shape(l.wo_b, nm("attn_output_b"), hp.o_groups * hp.o_lora, n) || !f32(l.q_a_norm, nm("attn_q_a_norm"), hp.q_lora, 1) ||
            !f32(l.kv_a_norm, nm("attn_kv_a_norm"), D, 1) || !f32(l.sinks, nm("attn_sinks"), H, 1))
            return false;
        if (l.probs_b && !f32(l.probs_b, nm("exp_probs_b.bias"), E, 1)) return false;
        if (l.probs_b_vl && !f32(l.probs_b_vl, nm("exp_probs_b_vl.bias"), E, 1)) return false;
        if (il < hp.n_hash) {   // ffn_gate_tid2eid [k, n_vocab] I32: row = token (DSV4_SPEC 2.3)
            if (!get(nm("ffn_gate_tid2eid.weight"), l.tid2eid, true)) return false;
            if (l.tid2eid.type != GGML_TYPE_I32 || l.tid2eid.ne[0] != hp.n_used || l.tid2eid.ne[1] != hp.n_vocab)
                return fail(err, nm("ffn_gate_tid2eid.weight") + ": not I32 [n_used, n_vocab]");
        }
        if (r > 0) {   // the layer's compressor: CSA (2 x 512, overlapping windows) or HCA (512)
            const int64_t cw = (r == 4 ? 2 : 1) * D;
            ok = get(nm("attn_compressor_kv.weight"), l.c_kv, true) && get(nm("attn_compressor_gate.weight"), l.c_gate, true) &&
                 get(nm("attn_compressor_ape.weight"), l.c_ape, true) && get(nm("attn_compressor_norm.weight"), l.c_norm, true);
            if (!ok || !proj_ok(l.c_kv, nm("attn_compressor_kv")) || !proj_ok(l.c_gate, nm("attn_compressor_gate")) ||
                !shape(l.c_kv, nm("attn_compressor_kv"), n, cw) || !shape(l.c_gate, nm("attn_compressor_gate"), n, cw) ||
                !f32(l.c_ape, nm("attn_compressor_ape"), cw, r) || !f32(l.c_norm, nm("attn_compressor_norm"), D, 1))
                return false;
        }
        if (r == 4) {   // the lightning indexer: its q from qr, its per-head weights from x, its key compressor (2 x 128)
            ok = get(nm("indexer.attn_q_b.weight"), l.idx_q_b, true) && get(nm("indexer.proj.weight"), l.idx_proj, true) &&
                 get(nm("indexer_compressor_kv.weight"), l.l_kv, true) && get(nm("indexer_compressor_gate.weight"), l.l_gate, true) &&
                 get(nm("indexer_compressor_ape.weight"), l.l_ape, true) && get(nm("indexer_compressor_norm.weight"), l.l_norm, true);
            if (!ok || !proj_ok(l.idx_q_b, nm("indexer.attn_q_b")) || !proj_ok(l.l_kv, nm("indexer_compressor_kv")) ||
                !proj_ok(l.l_gate, nm("indexer_compressor_gate")) || !shape(l.idx_q_b, nm("indexer.attn_q_b"), hp.q_lora, IH * ID) ||
                !f32(l.idx_proj, nm("indexer.proj"), n, IH) || !shape(l.l_kv, nm("indexer_compressor_kv"), n, 2 * ID) ||
                !shape(l.l_gate, nm("indexer_compressor_gate"), n, 2 * ID) || !f32(l.l_ape, nm("indexer_compressor_ape"), 2 * ID, 4) ||
                !f32(l.l_norm, nm("indexer_compressor_norm"), ID, 1))
                return false;
        }
        // the router: BF16 [n_embd, n_expert] (or already F32) -> an F32 copy on the layer's GPU
        if (l.gate_inp.ne[0] != n || l.gate_inp.n_out() != E) return fail(err, nm("ffn_gate_inp.weight") + ": shape");
        if (l.gate_inp.type == GGML_TYPE_BF16) {
            size_t shd = 0;
            const strata::TensorInfo* ti = gm->find(nm("ffn_gate_inp.weight"), &shd);
            if (!ti) return fail(err, "missing tensor " + nm("ffn_gate_inp.weight"));
            std::vector<float> w((size_t) (n * E));
            bf16_to_f32(reinterpret_cast<const uint16_t*>(gm->shard(shd).tensor_data(*ti)), w.data(), w.size());
            Stage& g = stage_of(il);
            set_dev(g.dev);
            float* dw = g.dalloc<float>(w.size(), err);
            if (!dw || cudaMemcpy(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
                return fail(err, "the router's F32 copy: " + err);
            l.gate_inp.d = dw;
            l.gate_inp.type = 0;
            l.gate_inp.bytes = w.size() * sizeof(float);
        } else if (l.gate_inp.type != 0) {
            return fail(err, nm("ffn_gate_inp.weight") + ": neither BF16 nor F32");
        }
    }
    reset_dev_cache();
    rope_pc = dsv4_rope_pc(hp.crope_base, hp.rope_scale, 1.0f, (int) hp.n_ctx_orig, hp.beta_fast, hp.beta_slow);
    return true;
}

int64_t Engine::Impl::dsv4_raw_rows() const {
    // dsv4_raw_store / dsv4_attn: the 127 positions before a batch and the batch itself (a decode window, a prompt sub-batch)
    const int64_t tw = std::max<int64_t>(kTmax, opt.prefill_sub);
    return (hp.n_swa - 1 + tw + 63) / 64 * 64;
}

/// The engine's compressor geometry: llama.cpp's (DSV4_SPEC 3.5), with each state ring kTmax rows longer than llama.cpp's
/// (8 -> 16, 128 -> 136).  The ring is read for the positions before a batch that a block completing in it pools; a verify window's
/// rejected rows are written past the accepted ones, and with the longer ring they never land on a row a later block still
/// reads - so a rollback needs no copy of the ring (the pooled rows themselves are rewritten as the positions return).
Dsv4Compressor Engine::Impl::dsv4_comp(int ratio, bool lid) const {
    Dsv4Compressor c = lid ? dsv4_lid() : ratio == 4 ? dsv4_csa() : dsv4_hca();
    c.ring = dsv4_ring(c.ratio);
    return c;
}

uint64_t Engine::Impl::dsv4_state_bytes(int64_t il) const {
    const int64_t D = hp.hd_k, ID = hp.idx_dim;
    const int r = hp.cratio[(size_t) il];
    uint64_t b = (uint64_t) dsv4_raw_rows() * D * 2;
    if (r > 0) {
        const int64_t rows = opt.n_ctx / r + 2, cw = (r == 4 ? 2 : 1) * D;
        b += (uint64_t) rows * D * 2 + (uint64_t) dsv4_ring(r) * cw * 4 * 2;
        if (r == 4) b += (uint64_t) rows * ID * 2 + (uint64_t) dsv4_ring(4) * 2 * ID * 4 * 2;
    }
    return b;
}

/// DeepSeek-V4's decode state per layer (DSV4_SPEC section 5): the raw K = V ring, the compressed rows of the whole context
/// (n_ctx / ratio + 2: the block at the context's end and a spare), the indexer's rows, the compressors' state rings - all on
/// the layer's GPU, zeroed.  ~6.9 KB per position over the model (F16), as llama.cpp's.
bool Engine::Impl::alloc_state_dsv4(std::string& err) {
    const int64_t n = hp.n_embd, D = hp.hd_k, ID = hp.idx_dim;
    for (auto& gp : stages)
        if (!gp->alloc(err) || !gp->alloc_decode(err)) return false;
    if (cudaMallocHost(&h_x, (size_t) kTmax * n * sizeof(float)) != cudaSuccess ||
        cudaMallocHost(&h_parts, (size_t) kTmax * hp.n_used * n * sizeof(float)) != cudaSuccess ||
        cudaMallocHost(&h_ids, (size_t) kTmax * 64 * sizeof(int32_t)) != cudaSuccess ||
        cudaMallocHost(&h_amax, (size_t) kTmax * sizeof(int32_t)) != cudaSuccess)
        return fail(err, "pinned host buffers");
    uint64_t bytes = 0;
    for (int64_t il = 0; il < n_load; ++il) {
        Layer& l = L[(size_t) il];
        Stage& g = stage_of(il);
        cudaSetDevice(g.dev);
        const int r = hp.cratio[(size_t) il];
        l.raw_rows = dsv4_raw_rows();
        if (!(l.raw = g.dalloc<uint16_t>((size_t) (l.raw_rows * D), err))) return false;
        if (r > 0) {
            const int64_t cw = (r == 4 ? 2 : 1) * D, S = dsv4_ring(r);
            l.c_rows = opt.n_ctx / r + 2;
            if (!(l.ccache = g.dalloc<uint16_t>((size_t) (l.c_rows * D), err)) || !(l.c_ring_kv = g.dalloc<float>((size_t) (S * cw), err)) ||
                !(l.c_ring_sc = g.dalloc<float>((size_t) (S * cw), err)))
                return false;
            if (r == 4 && (!(l.lcache = g.dalloc<uint16_t>((size_t) (l.c_rows * ID), err)) ||
                           !(l.l_ring_kv = g.dalloc<float>((size_t) (S * 2 * ID), err)) ||
                           !(l.l_ring_sc = g.dalloc<float>((size_t) (S * 2 * ID), err))))
                return false;
        }
        bytes += dsv4_state_bytes(il);
    }
    reset_dev_cache();
    if (opt.verbose)
        fprintf(stderr, "strata-x: DeepSeek-V4 attention state %.1f MiB for %lld positions (raw rings of %lld rows)\n",
                bytes / 1048576.0, (long long) opt.n_ctx, (long long) dsv4_raw_rows());
    return true;
}

// ------------------------------------------------------------------ the attention half

/// DeepSeek-V4's attention over T tokens at positions pos0.. (DSV4_SPEC 3.3-3.8; llama.cpp deepseek4.cpp build_attention):
///   q  = RoPE(RMS_heads(q_b . RMS(q_a . x)))                    64 heads of 512, RoPE on dims [448, 512)
///   kv = RoPE(RMS(attn_kv . x))  -> the raw ring                 the one shared K = V row
///   ratio 4:   CSA rows pooled from the compressor's projections (overlapping windows of 8), and the lightning indexer's
///              rows from its own compressor; the indexer scores every visible CSA row and keeps the top 512
///   ratio 128: HCA rows pooled from windows of 128 (every visible row is attended)
///   attention of each head over the raw window + the chosen compressed rows, with the head's sink
///   out = wo_b . concat_g(wo_a[g] . de-RoPE(attention)[heads 8g .. 8g + 7])
/// Every per-step value is device-resident: with gpos set (a graph capture) the kernels read the position from it.
void Stage::dsv4(int il, int64_t pos0, int T, const float* in, float* out_r) {
    const Layer& l = L[(size_t) il];
    const int n = (int) hp.n_embd, H = (int) hp.n_head, D = (int) hp.hd_k, Lq = (int) hp.q_lora, nr = (int) hp.n_rot;
    const int IH = (int) hp.idx_heads, ID = (int) hp.idx_dim, G = (int) hp.o_groups, OL = (int) hp.o_lora;
    const int ratio = hp.cratio[(size_t) il];
    const float eps = rms_eps();   // the trunk's, or the MTP head's own (dsv4_mtp.cpp)
    const Dsv4Rope rp = ratio == 0 ? dsv4_rope_p0(hp.rope_base) : m.rope_pc;   // the ratio-0 layers: plain RoPE, no YaRN
    const int roff = D - nr;
    quant(in, xq, n, T);
    // the query (DSV4_SPEC 3.3)
    pj(l.q_a, in, xq, qr, T);
    probe("qr", il, qr, Lq);
    rms_norm(qr, l.q_a_norm.f(), qr, T, Lq, eps, s);
    probe("qr_norm", il, qr, Lq);
    quant(qr, xq2, Lq, T);
    pj(l.q_b, qr, xq2, dq, T);
    rms_norm(dq, nullptr, dq, T * H, D, eps, s);   // per head, no weight
    probe("q_norm", il, dq, D, H);
    dsv4_rope(dq, dq, T, H, D, roff, rp, false, pos0, s, gpos);
    probe("q", il, dq, D, H);
    // the shared K = V row into the raw ring (3.4)
    pj(l.wkv, in, xq, dkv, T);
    rms_norm(dkv, l.kv_a_norm.f(), dkv, T, D, eps, s);
    probe("kv_norm", il, dkv, D);
    dsv4_rope(dkv, dkv, T, 1, D, roff, rp, false, pos0, s, gpos);
    probe("kv", il, dkv, D);
    dsv4_raw_store(dkv, l.raw, (int) l.raw_rows, pos0, T, s, gpos);
    if (ratio > 0) {   // the layer's compressor (3.5): its rows for the blocks this batch completes, then the state ring
        const Dsv4Compressor c = m.dsv4_comp(ratio, false);
        const int cw = c.coff * c.d;
        const std::string cn = ratio == 4 ? "csa" : "hca";
        pj(l.c_kv, in, xq, dakv, T);
        pj(l.c_gate, in, xq, dgs, T);
        probe((cn + "_state_kv").c_str(), il, dakv, cw);
        probe((cn + "_state_score").c_str(), il, dgs, cw);
        dsv4_compress_pool(c, dakv, dgs, l.c_ape.f(), l.c_ring_kv, l.c_ring_sc, l.c_norm.f(), eps, m.rope_pc, l.ccache, pos0, T, s,
                           gpos);
        dsv4_compress_persist(c, dakv, dgs, l.c_ape.f(), l.c_ring_kv, l.c_ring_sc, pos0, T, s, gpos);
        if (ratio == 4) {   // the lightning indexer (3.6): its key compressor, q, per-head weights, scores, the top-k
            const Dsv4Compressor lc = m.dsv4_comp(4, true);
            pj(l.l_kv, in, xq, dlakv, T);
            pj(l.l_gate, in, xq, dlgs, T);
            probe("lid_state_kv", il, dlakv, 2 * ID);
            probe("lid_state_score", il, dlgs, 2 * ID);
            dsv4_compress_pool(lc, dlakv, dlgs, l.l_ape.f(), l.l_ring_kv, l.l_ring_sc, l.l_norm.f(), eps, m.rope_pc, l.lcache, pos0,
                               T, s, gpos);
            dsv4_compress_persist(lc, dlakv, dlgs, l.l_ape.f(), l.l_ring_kv, l.l_ring_sc, pos0, T, s, gpos);
            pj(l.idx_q_b, qr, xq2, diq, T);   // qr's Q8_1 columns are still in xq2
            probe("lid_q", il, diq, ID, IH);
            dsv4_rope(diq, diq, T, IH, ID, ID - nr, m.rope_pc, false, pos0, s, gpos);
            probe("lid_q_rope", il, diq, ID, IH);
            dsv4_hadamard(diq, diq, T * IH, ID, s);
            probe("lid_q_rot", il, diq, ID, IH);
            gemv_dt(l.idx_proj, in, dw, n, IH, T);
            scale_f32(dw, T * IH, 1.0f / std::sqrt((float) (ID * IH)), s);
            probe("lid_weights", il, dw, IH);
            if (!dsv4_lid_scores_mma(diq, l.lcache, dw, dscore, m.lid_rows, pos0, T, s, gpos))
                dsv4_lid_scores(diq, l.lcache, dw, dscore, m.lid_rows, pos0, T, s, gpos);
            dsv4_lid_select(dscore, m.lid_rows, (int) hp.idx_topk, dsel, nullptr, pos0, T, s, gpos);
        }
    }
    // the attention with sinks (3.7): the raw window plus the layer's compressed rows; then the de-RoPE (3.8)
    const float scale = 1.0f / std::sqrt((float) D);
    if (!dsv4_attn_mma(dq, l.raw, (int) l.raw_rows, l.ccache, ratio, dsel, (int) hp.idx_topk, l.sinks.f(), scale, datt, dapart, pos0,
                       T, s, gpos, attn_noncausal))
        dsv4_attn(dq, l.raw, (int) l.raw_rows, l.ccache, ratio, dsel, (int) hp.idx_topk, l.sinks.f(), scale, datt, dapart, pos0, T,
                  s, gpos, attn_noncausal);
    probe(ratio == 0 ? "attn_raw" : ratio == 4 ? "attn_csa_lid" : "attn_hca", il, datt, H * D);
    dsv4_rope(datt, datt, T, H, D, roff, rp, true, pos0, s, gpos);
    probe("attn_derope", il, datt, D, H);
    // the grouped low-rank output (3.8): group g = heads [8g, 8g + 8) through wo_a's rows [g OL, (g + 1) OL), then wo_b
    const int gw = H * D / G;
    permute_abc(datt, dq, T, G, gw, s);           // (T, G, gw) -> (G, T, gw); q is spent
    heads_mm(l.wo_a, dq, doa_hm, G, gw, OL, T);   // (G, T, OL): llama.cpp's attn_wo_a [OL, T, G]
    probe("attn_wo_a", il, doa_hm, OL, T, G);
    permute_abc(doa_hm, doa, G, T, OL, s);        // (T, G * OL)
    quant(doa, xq, G * OL, T);
    pj(l.wo_b, doa, xq, out_r, T);
    probe("attn_out", il, out_r, n);
}

// ------------------------------------------------------------------ routing

void Stage::route(int il, int T, const int32_t* tok_d, const float* pen) {
    const Layer& l = L[(size_t) il];
    const int E = (int) hp.E_of(il), k = (int) hp.k_of(il);   // (DeepSeek-V4.1's DSpark blocks: 128, top-3)
    if (hp.arch == Arch::ernie) {   // moex (ERNIE-5.5): sigmoid(W0 x) + sigmoid(W1 x), its bias's top-k, the per-expert factor
        ernie_route_w(router, l.probs_b.f(), l.exp_scale.f(), E, k, ids, wts, T, s);
        return;
    }
    if (hp.arch != Arch::dsv4) {
        moe_route_w(router, l.probs_b ? l.probs_b.f() : nullptr, E, k, hp.ew_norm, hp.ew_scale, ids, wts, T, s);
        return;
    }
    // DSV4_SPEC 3.11: probs = sqrt(softplus(logits)); the hash layers' ids from the token table, the others' the top-k of
    // probs + exp_probs_b; the weights are the UNBIASED probs at the ids, normalized, times expert_weights_scale.  (Image rows
    // route by exp_probs_b_vl with no hash - the vision path's to wire: llama.cpp switches per embedding batch.)
    if (image_batch) {   // an image span (deepseek4.cpp is_media): exp_probs_b_vl (exp_probs_b without it) and no hash, any layer
        const DT& vb = l.probs_b_vl ? l.probs_b_vl : l.probs_b;
        dsv4_route(router, vb ? vb.f() : nullptr, nullptr, E, k, hp.ew_scale, ids, wts, T, s, nullptr, hp.v41);
        return;
    }
    const int32_t* hash = nullptr;
    if (il < hp.n_hash) {
        if (!tok_d) throw std::runtime_error("DeepSeek-V4 hash layer " + std::to_string(il) + " without the tokens on its GPU");
        dsv4_hash_ids(static_cast<const int32_t*>(l.tid2eid.d), tok_d, k, T, dhash, s);
        hash = dhash;
    }
    if (img_mix && img_pos0 >= 0 && !m.img.empty()) {   // moex (#113): text and picture rows in one batch - each run of rows
        const DT& vb = l.probs_b_vl ? l.probs_b_vl : l.probs_b;   // routes as its kind (the kernel's rows are independent)
        for (int r0 = 0; r0 < T;) {
            const bool pic = m.image_row(img_pos0 + r0) != nullptr;
            int r1 = r0 + 1;
            while (r1 < T && (m.image_row(img_pos0 + r1) != nullptr) == pic) ++r1;
            const size_t ro = (size_t) r0 * E, ko = (size_t) r0 * k;
            if (pic) dsv4_route(router + ro, vb ? vb.f() : nullptr, nullptr, E, k, hp.ew_scale, ids + ko, wts + ko, r1 - r0, s, nullptr, hp.v41);
            else dsv4_route(router + ro, hash ? nullptr : (l.probs_b ? l.probs_b.f() : nullptr), hash ? hash + ko : nullptr, E, k,
                            hp.ew_scale, ids + ko, wts + ko, r1 - r0, s, hash ? nullptr : pen, hp.v41);
            r0 = r1;
        }
        return;
    }
    dsv4_route(router, hash ? nullptr : (l.probs_b ? l.probs_b.f() : nullptr), hash, E, k, hp.ew_scale, ids, wts, T, s,
               hash ? nullptr : pen, hp.v41);
}

/// moex (DSV41_BACKLOG #112), opt-in STRATA_DSV41_ROUTE_PUBLISH=1: a decode window's route() and the doorbell's publish of
/// its ids and activations in one launch (dsv4_route_publish: the same ids and weights) - false where it does not apply, the
/// caller then running both
bool Stage::route_publish(int il, int T, const int32_t* tok_d, const float* pen) {
    if (hp.arch == Arch::ernie) {   // moex (the ERNIE-5.5 hill-climb): one token's routing and publish in one launch (route's
                                    // ernie_route_w and the doorbell_publish of the ids and the experts' input, the same bits)
        const Layer& l = L[(size_t) il];
        const bool lat = (bool) l.r_down;
        const int nx = lat ? (int) hp.ex() : (int) hp.n_embd;
        return ernie_route_publish_w(router, l.probs_b.f(), l.exp_scale.f(), (int) hp.E_of(il), (int) hp.k_of(il), ids, wts,
                                     lat ? hxl : cur, T * nx, dd_ids, dd_x, dd_seq, d_pub_cnt, T, s);
    }
    static const bool on = [] {
        const char* e = std::getenv("STRATA_DSV41_ROUTE_PUBLISH");
        return e && e[0] == '1';
    }();
    if (!on || hp.arch != Arch::dsv4 || !hp.v41 || image_batch || img_mix || T > 16) return false;
    const Layer& l = L[(size_t) il];
    const int E = (int) hp.E_of(il), k = (int) hp.k_of(il);
    const int32_t* hash = nullptr;
    if (il < hp.n_hash) {
        if (!tok_d) throw std::runtime_error("DeepSeek-V4 hash layer " + std::to_string(il) + " without the tokens on its GPU");
        dsv4_hash_ids(static_cast<const int32_t*>(l.tid2eid.d), tok_d, k, T, dhash, s);
        hash = dhash;
    }
    return dsv4_route_publish(router, hash ? nullptr : (l.probs_b ? l.probs_b.f() : nullptr), hash, E, k, hp.ew_scale, ids, wts, T, s,
                              hash ? nullptr : pen, hp.v41, cur, T * (int) hp.n_embd, dd_ids, dd_x, dd_seq, d_pub_cnt);
}

bool Stage::upload_toks(const int32_t* toks, int T, std::string& err) {
    if (T > tok_cap) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(s);   // the old buffer may still be read
        if (tok_buf) cudaFree(tok_buf);
        tok_buf = nullptr;
        tok_cap = 0;
        if (cudaMalloc(&tok_buf, (size_t) std::max(T, kTmax) * sizeof(int32_t)) != cudaSuccess)
            return m.fail(err, "the tokens' device buffer on GPU " + std::to_string(dev));
        tok_cap = std::max(T, kTmax);
    }
    if (cudaMemcpyAsync(tok_buf, toks, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, s) != cudaSuccess)
        return m.fail(err, "the tokens' upload");
    d_toks = tok_buf;
    return true;
}

}  // namespace strata::x
