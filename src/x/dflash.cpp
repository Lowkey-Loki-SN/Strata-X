// src/x/dflash.cpp - moex: MiMo-V2's DFlash block drafter, as llama.cpp runs it (--spec-type draft-dflash -md FILE):
//   * the model: src/models/dflash.cpp - load_arch_hparams / load_arch_tensors (the plain branch: no hyper-connections, no
//     selector, no Markov head) and graph<false>, "dual-mode by batch type": an embd batch injects target features into the
//     drafter's cache, a token batch decodes the noise block;
//   * the loop: common/speculative.cpp common_speculative_impl_draft_dflash - process() after every target batch, draft()
//     before every verify window; tools/server/server-context.cpp drops the drafter's cells past the committed position after
//     each draft and each verify, so only committed rows stay injected (the HF reference, dflash.py spec_generate, feeds the
//     accepted rows' features only and crops the noise block's cells).
// What it is (XiaomiMiMo/MiMo-V2.6-Flash-*/dflash, converted by convert_hf_to_gguf.py --target-model-dir): five dense Qwen3-style
// blocks - 64 query heads / 8 KV heads of 128, q / k RMS norms, NeoX RoPE on the first 64 dims, sinks, a 1024-position sliding
// window, the attention output scaled by attention.value_scale (0.612), a SwiGLU FFN of 16384 - over the TRUNK's features: the
// residual at the input of trunk layers 1, 12, 24, 36 and the trunk's output (48; mimo2.cpp t_layer_inp), fused by fc and an
// RMS norm (the "encoder") and projected through every block's k / v (k norm, RoPE at the row's position) into that block's
// cache ("inject").  A draft step decodes the noise block [anchor, mask x 7] at the positions after the committed ones -
// non-causal inside the block (llama_set_causal_attn(false); each query keeps its own window's lower bound) - through the five
// blocks, the final norm and the head; position i's logits are the draft for the token at pos + i.  Its embedding table is the
// trunk's with the mask row replaced by the drafter's own (mask_embedding.pt: "MiMo's target mask row is untrained"), its head
// the trunk's lm_head.
// Here: its weights on the last GPU; the trunk's head and embedding rows wherever the file's are byte-identical (checked at
// load); per block a K / V ring of the window plus the noise block; the features captured on whichever GPU runs the target
// layer and carried to the last one in mapped memory (a window's rows) or a pinned double buffer (a prompt chunk's last `cap`
// rows); drafts by argmax or Strata's coupled sampling against the verify rows; eager launches.
#include "engine_impl.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::x {

namespace {
const std::string kDf = "the DFlash drafter: ";
}  // namespace

// ------------------------------------------------------------------ the file: geometry, checks, what it costs

bool Engine::Impl::open_dflash(std::string& err) {
    if (opt.dflash.empty()) return true;
    if (hp.arch != Arch::mimo2) return fail(err, "--draft-dflash drafts for MiMo-V2 (a mimo2 GGUF)");
    auto d = std::make_unique<Dflash>();
    try {
        d->gm = std::make_unique<strata::GgufModel>(strata::GgufModel::open(opt.dflash));
    } catch (const std::exception& e) {
        return fail(err, kDf + e.what());
    }
    const strata::GgufFile& mt = d->gm->meta();
    const strata::MetaValue* a = mt.get("general.architecture");
    if (!a || a->s != "dflash") return fail(err, kDf + "not a dflash GGUF");
    if (mt.get("dflash.hyper_connection.count")) return fail(err, kDf + "a DeepSeek-V4 DSpark drafter (--draft-dspark)");
    auto num = [&](const char* k, double def, bool req, double& v) -> bool {
        const strata::MetaValue* x = mt.get(std::string("dflash.") + k);
        if (!x) {
            if (req) return fail(err, kDf + "missing dflash." + k);
            v = def;
            return true;
        }
        v = x->num();
        return true;
    };
    double nl = 0, bs = 0, emb = 0, nff = 0, heads = 0, hkv = 0, kl = 0, vl = 0, rot = 0, fb = 0, swa = 0, eps = 0, vs = 0;
    if (!num("block_count", 0, true, nl) || !num("block_size", 16, false, bs) || !num("embedding_length", 0, true, emb) ||
        !num("feed_forward_length", 0, true, nff) || !num("attention.head_count", 0, true, heads) ||
        !num("attention.head_count_kv", 0, true, hkv) || !num("attention.key_length", 0, true, kl) ||
        !num("attention.value_length", 0, true, vl) || !num("rope.dimension_count", 0, true, rot) ||
        !num("rope.freq_base", 10000, false, fb) || !num("attention.sliding_window", 0, false, swa) ||
        !num("attention.layer_norm_rms_epsilon", 0, true, eps) || !num("attention.value_scale", 0, false, vs))
        return false;
    // what this engine runs: mainline's plain DFlash decoder over full-vocabulary drafts sharing the trunk's vocabulary
    for (const char* k : {"dflash.embedding_scale", "dflash.logit_scale", "dflash.final_logit_softcapping", "dflash.rope.dimension_sections",
                          "dflash.conv_kernel_size", "dflash.selector_rank"})
        if (const strata::MetaValue* x = mt.get(k); x && (x->type == strata::MetaType::ARRAY || x->num() != 0.0))
            return fail(err, kDf + std::string(k) + " is set (a variant this engine does not run)");
    if (const strata::MetaValue* x = mt.get("dflash.attention.causal"); x && x->num() != 0.0)
        return fail(err, kDf + "a causal block (dflash.attention.causal): MiMo's is non-causal");
    for (const char* t : {"d2t", "markov_w1.weight", "selector_hidden.weight", "blk.0.attn_post_norm.weight", "blk.0.ffn_post_norm.weight",
                          "blk.0.layer_output_scale.weight", "rope_freqs.weight", "blk.0.dflash_attn_conv_proj.weight"})
        if (d->gm->find(t)) return fail(err, kDf + "tensor " + t + " (a variant this engine does not run)");
    d->n = (int64_t) nl;
    d->block = (int) bs;
    d->H = (int64_t) heads;
    d->nkv = (int64_t) hkv;
    d->hd = (int64_t) kl;
    d->n_rot = (int64_t) rot;
    d->n_ff = (int64_t) nff;
    d->window = (int64_t) swa;
    d->rope_base = (float) fb;
    d->eps = (float) eps;
    d->v_scale = vs == 1.0 ? 0.0f : (float) vs;   // llama.cpp keeps only a value != 1 (mimo2's rule)
    const int64_t n = hp.n_embd, V = hp.n_vocab;
    if ((int64_t) emb != n) return fail(err, kDf + "its embedding width differs from the trunk's");
    if (d->n < 1 || d->block < 2 || d->block > kTmax) return fail(err, kDf + "block size " + std::to_string(d->block) + " (2 .. 8 supported)");
    // the attention through gqa_attend_w's FP32 kernel: 128-wide heads, 4 / 8 / 12 / 16 query heads per KV head
    if ((int64_t) vl != d->hd || d->hd != 128 || d->nkv < 1 || d->H % d->nkv != 0 || (d->H / d->nkv) % 4 != 0 || d->H / d->nkv > 16)
        return fail(err, kDf + "heads of " + std::to_string(d->hd) + " / " + std::to_string((int64_t) vl) + ", " + std::to_string(d->H) +
                             " over " + std::to_string(d->nkv) + " (128-wide heads, 4..16 per KV head supported)");
    if (d->n_rot < 2 || d->n_rot > d->hd || d->n_rot % 2) return fail(err, kDf + "RoPE dimension count " + std::to_string(d->n_rot));
    // every block slides (the rings hold the window): llama.cpp's pattern array, all true
    if (d->window <= 0) return fail(err, kDf + "no sliding window (a full-attention drafter needs a whole-context cache)");
    if (const strata::MetaValue* x = mt.get("dflash.attention.sliding_window_pattern")) {
        if (x->type == strata::MetaType::ARRAY) {
            if ((int64_t) x->items.size() < d->n) return fail(err, kDf + "a short sliding_window_pattern");
            for (int64_t j = 0; j < d->n; ++j)
                if (x->items[(size_t) j].num() == 0.0) return fail(err, kDf + "block " + std::to_string(j) + " attends to the whole context");
        } else if (x->num() == 0.0) {
            return fail(err, kDf + "a full-attention block");
        }
    }
    const strata::MetaValue* tl = mt.get("dflash.target_layers");
    if (!tl || tl->type != strata::MetaType::ARRAY || tl->items.empty()) return fail(err, kDf + "no dflash.target_layers");
    for (const auto& it : tl->items) {
        const int64_t il = (int64_t) it.num();
        if (il < 0 || il > hp.n_layer) return fail(err, kDf + "target layer " + std::to_string(il) + " out of range");
        d->tl.push_back(il);
    }
    if (const strata::MetaValue* x = mt.get("tokenizer.ggml.mask_token_id")) d->mask = (int32_t) x->num();
    else return fail(err, kDf + "no tokenizer.ggml.mask_token_id");
    if (d->mask < 0 || d->mask >= V) return fail(err, kDf + "mask token out of the vocabulary");
    // the tensors' shapes and types
    auto shape = [&](const std::string& nm, int64_t ne0, int64_t ne1, bool f32) -> bool {
        const strata::TensorInfo* ti = d->gm->find(nm);
        if (!ti) return fail(err, kDf + "missing tensor " + nm);
        const int64_t a0 = ti->shape.size() > 0 ? (int64_t) ti->shape[0] : 1, a1 = ti->shape.size() > 1 ? (int64_t) ti->shape[1] : 1;
        if (a0 != ne0 || a1 != ne1) return fail(err, kDf + nm + " is not [" + std::to_string(ne0) + ", " + std::to_string(ne1) + "]");
        if (f32 ? ti->type != GGML_TYPE_F32 : !kn::native_mmvq_supported((int) ti->type))
            return fail(err, kDf + nm + (f32 ? " is not F32" : " is of a type with no GPU kernel"));
        return true;
    };
    const int64_t nt = (int64_t) d->tl.size(), qw = d->H * d->hd, kvw = d->nkv * d->hd;
    if (!shape("fc.weight", nt * n, n, false) || !shape("enc.output_norm.weight", n, 1, true) || !shape("output_norm.weight", n, 1, true))
        return false;
    for (int64_t j = 0; j < d->n; ++j) {
        auto bn = [&](const char* sfx) { return "blk." + std::to_string(j) + "." + sfx; };
        if (!shape(bn("attn_norm.weight"), n, 1, true) || !shape(bn("attn_q.weight"), n, qw, false) ||
            !shape(bn("attn_k.weight"), n, kvw, false) || !shape(bn("attn_v.weight"), n, kvw, false) ||
            !shape(bn("attn_output.weight"), qw, n, false) || !shape(bn("attn_q_norm.weight"), d->hd, 1, true) ||
            !shape(bn("attn_k_norm.weight"), d->hd, 1, true) || !shape(bn("attn_sinks.weight"), d->H, 1, true) ||
            !shape(bn("ffn_norm.weight"), n, 1, true) || !shape(bn("ffn_gate.weight"), n, d->n_ff, false) ||
            !shape(bn("ffn_up.weight"), n, d->n_ff, false) || !shape(bn("ffn_down.weight"), d->n_ff, n, false))
            return false;
    }
    // the embedding table: the trunk's (byte-identical but for the mask row: the anchor's row from the trunk's table, which the
    // host reads anyway), else its own; the mask row dequantized once
    {
        size_t shd = 0;
        const strata::TensorInfo* te = d->gm->find("token_embd.weight", &shd);
        if (!te || te->shape.size() < 2 || (int64_t) te->shape[0] != n || (int64_t) te->shape[1] != V)
            return fail(err, kDf + "token_embd is not [n_embd, n_vocab] (llama.cpp's DFlash takes the trunk's then; not supported)");
        const auto* tr = ggml_get_type_traits((ggml_type) te->type);
        if (!tr || !tr->to_float) return fail(err, kDf + "token_embd of a type with no dequantizer");
        d->embd = d->gm->shard(shd).tensor_data(*te);
        d->embd_type = (int) te->type;
        d->embd_row = ggml_row_size((ggml_type) te->type, n);
        d->mask_emb.resize((size_t) n);
        tr->to_float(d->embd + (size_t) d->mask * d->embd_row, d->mask_emb.data(), n);
        // the trunk's table (load_dense binds it later): rows [0, mask) and (mask, V), each range one compare (row-major)
        size_t tsh = 0;
        const strata::TensorInfo* tt = gm->find("token_embd.weight", &tsh);
        const size_t rb = d->embd_row, m0 = (size_t) d->mask;
        const uint8_t* te2 = tt ? gm->shard(tsh).tensor_data(*tt) : nullptr;
        const bool same = tt && tt->type == te->type && strata::tensor_payload_bytes(*tt) == strata::tensor_payload_bytes(*te) &&
                          std::memcmp(d->embd, te2, m0 * rb) == 0 &&
                          std::memcmp(d->embd + (m0 + 1) * rb, te2 + (m0 + 1) * rb, ((size_t) V - m0 - 1) * rb) == 0;
        d->own_embd = !same;
    }
    // the head: the trunk's when the file's is the same bytes (the converter copies the target's lm_head), else its own
    bool share_head = false;
    {
        size_t s1 = 0, s2 = 0;
        const strata::TensorInfo* o1 = d->gm->find("output.weight", &s1);
        const strata::TensorInfo* o2 = gm->find("output.weight", &s2);
        if (!o1) return fail(err, kDf + "no output.weight (llama.cpp's DFlash takes the trunk's then; not supported)");
        if (!shape("output.weight", n, V, false)) return false;
        const uint64_t b1 = strata::tensor_payload_bytes(*o1);
        share_head = o2 && o1->type == o2->type && b1 == strata::tensor_payload_bytes(*o2) &&
                     std::memcmp(d->gm->shard(s1).tensor_data(*o1), gm->shard(s2).tensor_data(*o2), b1) == 0;
    }
    // the rings: the window, the noise block written before it is read, a margin; a prompt chunk injects its last `cap` rows
    d->ring = (d->window + 2 * kTmax + 63) / 64 * 64;
    d->cap = (int) d->ring;
    d->chunks = (int) std::max<int64_t>(1, std::min<int64_t>(16, (d->window + kTmax + 127) / 128));
    // what it holds on the last GPU: the dense weights (not the shared head, not the embedding), the rings, the buffers
    uint64_t w = 0;
    for (size_t sh = 0; sh < d->gm->size(); ++sh)
        for (const auto& ti : d->gm->shard(sh).tensors()) {
            if (ti.name == "token_embd.weight" || (share_head && ti.name == "output.weight")) continue;
            w += (strata::tensor_payload_bytes(ti) + 255) & ~uint64_t(255);
        }
    const uint64_t rings = (uint64_t) d->n * d->ring * kvw * 2 * 2;
    const uint64_t bufs = (uint64_t) kTmax * (nt * n + 3 * n + 2 * qw + 2 * kvw + 2 * d->n_ff) * 4 * 2 +
                          gqa_attend_scratch_bytes((int) d->H, (int) d->hd, d->chunks, d->block) + (16ull << 20);
    dflash_vram = w + rings + bufs;
    if (!share_head) d->output.type = -2;   // load_dflash uploads its own
    if (opt.verbose)
        fprintf(stderr, "strata-x: DFlash drafter: %lld blocks, block size %d, %lld heads / %lld KV of %lld (RoPE %lld), window %lld, "
                        "value scale %.3f, mask %d, target layers",
                (long long) d->n, d->block, (long long) d->H, (long long) d->nkv, (long long) d->hd, (long long) d->n_rot,
                (long long) d->window, d->v_scale, d->mask);
    if (opt.verbose) {
        for (int64_t il : d->tl) fprintf(stderr, " %lld", (long long) il);
        fprintf(stderr, "; %s head, %s embedding rows; %.2f GiB on the last GPU\n", share_head ? "the trunk's" : "its own",
                d->own_embd ? "its own" : "the trunk's", dflash_vram / 1073741824.0);
    }
    dfl = std::move(d);
    return true;
}

// ------------------------------------------------------------------ the weights, the rings, the buffers

bool Engine::Impl::load_dflash(std::string& err) {
    if (!dfl) return true;
    Dflash& d = *dfl;
    Stage& lg = last_stage();
    const int di_last = (int) stages.size() - 1;
    const int64_t n = hp.n_embd, nt = (int64_t) d.tl.size(), qw = d.H * d.hd, kvw = d.nkv * d.hd;
    const bool own_head = d.output.type == -2;
    // where each target layer is captured: the stage that runs it (the trunk's output: the last); its column there
    d.tl_stage.assign((size_t) nt, di_last);
    d.tl_col.assign((size_t) nt, 0);
    for (size_t k = 0; k < d.tl.size(); ++k) {
        const int64_t il = d.tl[k];
        const int st = il < hp.n_layer ? ldev[(size_t) il] : di_last;
        d.tl_stage[k] = st;
        if (st == di_last) d.tl_col[k] = (int) k;
        else d.tl_col[k] = stages[(size_t) st]->df_ncol++;
    }
    // the dense tensors into one allocation on the last GPU
    struct Item {
        const strata::TensorInfo* ti;
        size_t shard;
        uint64_t bytes, off;
    };
    std::vector<Item> items;
    uint64_t total = 0;
    for (size_t sh = 0; sh < d.gm->size(); ++sh)
        for (const auto& ti : d.gm->shard(sh).tensors()) {
            if (ti.name == "token_embd.weight" || (!own_head && ti.name == "output.weight")) continue;
            const uint64_t b = strata::tensor_payload_bytes(ti);
            if (!b || !d.gm->in_bounds(ti, sh)) return fail(err, kDf + "tensor " + ti.name + " of unknown size");
            items.push_back({&ti, sh, b, total});
            total += (b + 255) & ~uint64_t(255);
        }
    set_dev(lg.dev);
    if (cudaMalloc(&d.dense, total) != cudaSuccess) {
        cudaGetLastError();
        return fail(err, kDf + std::to_string(total >> 20) + " MiB of weights on GPU " + std::to_string(lg.dev));
    }
    for (const auto& it : items) {
        uint8_t* dst = static_cast<uint8_t*>(d.dense) + it.off;
        if (cudaMemcpy(dst, d.gm->shard(it.shard).tensor_data(*it.ti), it.bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail(err, kDf + "upload of " + it.ti->name);
        DT x;
        x.d = dst;
        x.type = (int) it.ti->type;
        for (size_t k = 0; k < it.ti->shape.size() && k < 4; ++k) x.ne[k] = (int64_t) it.ti->shape[k];
        x.bytes = it.bytes;
        d.t[it.ti->name] = x;
    }
    auto get = [&](const std::string& nm) { return d.t.at(nm); };   // open_dflash checked every one
    d.fc = get("fc.weight");
    d.enc_norm = get("enc.output_norm.weight");
    d.out_norm = get("output_norm.weight");
    d.output = own_head ? get("output.weight") : DT{};
    d.b.resize((size_t) d.n);
    for (int64_t j = 0; j < d.n; ++j) {
        auto bn = [&](const char* sfx) { return "blk." + std::to_string(j) + "." + sfx; };
        Dflash::Blk& b = d.b[(size_t) j];
        b.attn_norm = get(bn("attn_norm.weight"));
        b.wq = get(bn("attn_q.weight"));
        b.wk = get(bn("attn_k.weight"));
        b.wv = get(bn("attn_v.weight"));
        b.wo = get(bn("attn_output.weight"));
        b.q_norm = get(bn("attn_q_norm.weight"));
        b.k_norm = get(bn("attn_k_norm.weight"));
        b.sinks = get(bn("attn_sinks.weight"));
        b.ffn_norm = get(bn("ffn_norm.weight"));
        b.gate = get(bn("ffn_gate.weight"));
        b.up = get(bn("ffn_up.weight"));
        b.down = get(bn("ffn_down.weight"));
        if (!(b.kc = lg.dalloc<uint16_t>((size_t) (d.ring * kvw), err)) || !(b.vc = lg.dalloc<uint16_t>((size_t) (d.ring * kvw), err)))
            return false;
    }
    // the last stage's buffers: a window's features and their projections, the noise block's
    const int64_t wide = std::max<int64_t>({nt * n, qw, d.n_ff, n});
    const int T = d.block;
    if (!(lg.df_feat = lg.dalloc<float>((size_t) (kTmax * nt * n), err)) || !(lg.df_g = lg.dalloc<float>((size_t) (kTmax * n), err)) ||
        !(lg.df_k = lg.dalloc<float>((size_t) (2 * kTmax * kvw), err)) ||
        !(lg.df_xq = lg.dalloc<uint8_t>(kn::native_q8_1_bytes((int) (kTmax * wide)), err)) ||
        !(lg.df_x = lg.dalloc<float>((size_t) (kTmax * n), err)) || !(lg.df_nn = lg.dalloc<float>((size_t) (kTmax * n), err)) ||
        !(lg.df_q = lg.dalloc<float>((size_t) (kTmax * qw), err)) || !(lg.df_kvn = lg.dalloc<float>((size_t) (2 * kTmax * kvw), err)) ||
        !(lg.df_att = lg.dalloc<float>((size_t) (kTmax * qw), err)) || !(lg.df_o = lg.dalloc<float>((size_t) (kTmax * n), err)) ||
        !(lg.df_fg = lg.dalloc<float>((size_t) (kTmax * d.n_ff), err)) || !(lg.df_fu = lg.dalloc<float>((size_t) (kTmax * d.n_ff), err)) ||
        !(lg.df_bxq = lg.dalloc<uint8_t>(kn::native_q8_1_bytes((int) (kTmax * wide)), err)) ||
        !(lg.df_part = lg.dalloc<float>(gqa_attend_scratch_bytes((int) d.H, (int) d.hd, d.chunks, T) / sizeof(float) + 1, err)))
        return false;
    auto mapped = [&](size_t bytes, void** hp_, void** dp_) {
        if (cudaHostAlloc(hp_, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return false;
        std::memset(*hp_, 0, bytes);
        return cudaHostGetDevicePointer(dp_, *hp_, 0) == cudaSuccess;
    };
    if (!mapped((size_t) kTmax * n * sizeof(float), (void**) &lg.df_in_h, (void**) &lg.df_in_d) ||
        !mapped(kTmax * sizeof(int32_t), (void**) &lg.df_tok_h, (void**) &lg.df_tok_d) ||
        !mapped(kTmax * sizeof(float), (void**) &lg.df_prob_h, (void**) &lg.df_prob_d) ||
        !mapped(kTmax * sizeof(int32_t), (void**) &lg.df_cell_h, (void**) &lg.df_cell_d) ||
        !mapped(2 * sizeof(int64_t), (void**) &lg.df_pos_h, (void**) &lg.df_pos_d))
        return fail(err, kDf + "its mapped blocks");
    for (int i = 1; i < T; ++i)   // the noise block's mask rows, once (row 0: the anchor's, per step)
        std::memcpy(lg.df_in_h + (size_t) i * n, d.mask_emb.data(), (size_t) n * sizeof(float));
    // the earlier stages': their targets' window rows (mapped) and chunk tails (pinned, by chunk parity)
    for (auto& gp : stages) {
        Stage& g = *gp;
        if (g.last || g.df_ncol == 0) continue;
        const size_t w_row = (size_t) g.df_ncol * n * sizeof(float);
        if (!mapped((size_t) kTmax * w_row, (void**) &g.df_map_h, (void**) &g.df_map_d)) return fail(err, kDf + "the window features' mapped block");
        for (float*& hb : g.h_dfl)
            if (cudaMallocHost(&hb, (size_t) d.cap * w_row) != cudaSuccess) return fail(err, kDf + "the chunk tails' pinned buffers");
    }
    cur_dev = -1;
    if (opt.verbose)
        fprintf(stderr, "strata-x: DFlash drafter: %.2f GiB of weights on GPU %d, rings of %lld rows, chunk tails of %d rows%s\n",
                total / 1073741824.0, lg.dev, (long long) d.ring, d.cap, own_head ? " (its own head)" : "");
    return true;
}

size_t Stage::dflash_tail_bytes(const Engine::Impl& m, int rows) {
    if (!m.dfl) return 0;
    const Engine::Impl::Dflash& d = *m.dfl;
    const int64_t r = std::min(rows, d.cap), n = m.hp.n_embd, nt = (int64_t) d.tl.size(), kvw = d.nkv * d.hd;
    return (size_t) (r * (nt * n + n + 2 * kvw)) * sizeof(float) + 3 * 256;
}

// ------------------------------------------------------------------ the trunk's features and the injection

void Stage::dflash_capture(int il, int r0, int rows, bool tail) {
    if (!m.dfl || rows <= 0) return;
    const Engine::Impl::Dflash& d = *m.dfl;
    int k = -1;
    for (size_t i = 0; i < d.tl.size(); ++i)
        if (d.tl[i] == il && d.tl_stage[i] == di) k = (int) i;
    if (k < 0) return;
    const int n = (int) hp.n_embd, nt = (int) d.tl.size();
    const float* src = R + (size_t) r0 * n;   // MiMo: one residual stream
    if (last) {
        if (tail) cudaMemcpy2DAsync(df_feat_pr + (size_t) k * n, (size_t) nt * n * sizeof(float), src, (size_t) n * sizeof(float),
                                    (size_t) n * sizeof(float), (size_t) std::min(rows, d.cap), cudaMemcpyDeviceToDevice, s);
        else copy_rows_w(src, n, df_feat + (size_t) k * n, (int64_t) nt * n, n, std::min(rows, kTmax), s);
        return;
    }
    const int col = d.tl_col[(size_t) k];
    if (tail) cudaMemcpy2DAsync(h_dfl[dfl_par] + (size_t) col * n, (size_t) df_ncol * n * sizeof(float), src, (size_t) n * sizeof(float),
                                (size_t) n * sizeof(float), (size_t) std::min(rows, d.cap), cudaMemcpyDeviceToHost, s);
    else copy_rows_w(src, n, df_map_d + (size_t) col * n, (int64_t) df_ncol * n, n, std::min(rows, kTmax), s);
}

void Stage::dflash_mm(const DT& w, const void* xq_, float* y, int T) {
    // Q8_0 on int8 tensor cores (q8_mma_w: half MMVQ's time at 8 columns); anything else, or an older card, MMVQ
    static const bool mma_off = std::getenv("STRATA_X_DFLASH_MMVQ") != nullptr;   // A/B: the drafter on MMVQ
    if (w.type == 8 && !mma_off && q8_mma_w(w.d, xq_, y, (int) w.n_in(), (int) w.n_out(), T, s)) return;
    mm_t(w, xq_, y, T);
}

void Stage::dflash_proj(const DT& w, const float* x, int rows, float* y) {
    if (rows > kTmax && mmq_ok(w)) {   // a chunk's tail: the prompt path's loan holds mb.xq
        mmq_dense(w, x, y, rows);
        return;
    }
    const int ni = (int) w.n_in(), no = (int) w.n_out();
    for (int r0 = 0; r0 < rows; r0 += kTmax) {
        const int tn = std::min(kTmax, rows - r0);
        quant(x + (size_t) r0 * ni, df_xq, ni, tn);
        dflash_mm(w, df_xq, y + (size_t) r0 * no, tn);
    }
}

/// dflash.cpp graph<false>, the embd batch: g = RMS(fc . [f_1 | .. | f_nt], enc norm); per block K = RMS_head(wk . g, k norm) with
/// NeoX RoPE at the row's position, V = wv . g, into the block's ring.  The kernels only (a window's: rows <= kTmax, the
/// position from pos_dev when given - a captured graph's), the earlier stages' columns from their mapped blocks.
void Stage::dflash_inject_ops(int rows, int64_t pos0, const int64_t* pos_dev) {
    Engine::Impl::Dflash& d = *m.dfl;
    const int n = (int) hp.n_embd, nt = (int) d.tl.size(), nf = nt * n, nkv = (int) d.nkv, hd = (int) d.hd, kvw = nkv * hd;
    for (size_t k = 0; k < d.tl.size(); ++k) {
        if (d.tl_stage[k] == di) continue;
        const Stage& o = *m.stages[(size_t) d.tl_stage[k]];
        copy_rows_w(o.df_map_d + (size_t) d.tl_col[k] * n, (int64_t) o.df_ncol * n, df_feat + k * n, nf, n, rows, s);
    }
    quant(df_feat, df_xq, nf, rows);
    dflash_mm(d.fc, df_xq, df_g, rows);
    rms_norm(df_g, d.enc_norm.f(), df_g, rows, n, d.eps, s);
    quant(df_g, df_xq, n, rows);
    for (const Engine::Impl::Dflash::Blk& b : d.b) {
        dflash_mm(b.wk, df_xq, df_k, rows);
        dflash_mm(b.wv, df_xq, df_k + (size_t) rows * kvw, rows);
        rms_norm(df_k, b.k_norm.f(), df_k, rows * nkv, hd, d.eps, s);
        rope_neox_w(df_k, kvw, 0, nkv, hd, (int) d.n_rot, d.rope_base, pos0, rows, s, pos_dev);
        kv_store_w(df_k, kvw, 0, rows * kvw, b.kc, b.vc, d.ring, nkv, hd, hd, pos0, rows, s, pos_dev);
    }
}

bool Stage::dflash_inject(int rows, int64_t pos0, bool tail, std::string& err) {
    Engine::Impl::Dflash& d = *m.dfl;
    const int n = (int) hp.n_embd, nt = (int) d.tl.size(), nf = nt * n, nkv = (int) d.nkv, hd = (int) d.hd, kvw = nkv * hd;
    rows = std::min(rows, tail ? d.cap : kTmax);
    if (rows <= 0) return true;
    try {
        if (!tail) {
            dflash_inject_ops(rows, pos0, nullptr);
            return true;
        }
        if (!dfl_in && m.stages.size() > 1)
            for (size_t k = 0; k < d.tl.size(); ++k)
                if (d.tl_stage[k] != di) return m.fail(err, kDf + "a chunk's tail without the earlier stage's features");
        for (size_t k = 0; k < d.tl.size(); ++k) {   // the earlier stages' columns beside this stage's own
            if (d.tl_stage[k] == di) continue;
            const Stage& o = *m.stages[(size_t) d.tl_stage[k]];
            cudaMemcpy2DAsync(df_feat_pr + k * n, (size_t) nf * sizeof(float), dfl_in + (size_t) d.tl_col[k] * n,
                              (size_t) o.df_ncol * n * sizeof(float), (size_t) n * sizeof(float), (size_t) rows, cudaMemcpyHostToDevice, s);
        }
        dflash_proj(d.fc, df_feat_pr, rows, df_g_pr);
        rms_norm(df_g_pr, d.enc_norm.f(), df_g_pr, rows, n, d.eps, s);
        for (const Engine::Impl::Dflash::Blk& b : d.b) {
            dflash_proj(b.wk, df_g_pr, rows, df_k_pr);
            dflash_proj(b.wv, df_g_pr, rows, df_k_pr + (size_t) rows * kvw);
            rms_norm(df_k_pr, b.k_norm.f(), df_k_pr, rows * nkv, hd, d.eps, s);
            rope_neox_w(df_k_pr, kvw, 0, nkv, hd, (int) d.n_rot, d.rope_base, pos0, rows, s);
            kv_store_w(df_k_pr, kvw, 0, rows * kvw, b.kc, b.vc, d.ring, nkv, hd, hd, pos0, rows, s);
        }
    } catch (const std::exception& e) {
        return m.fail(err, kDf + "inject: " + e.what());
    }
    // a chunk's tail: the earlier stage rewrites this h_dfl two chunks on, possibly before this stream ran the copies above
    if (dfl_in && cudaStreamSynchronize(s) != cudaSuccess) return m.fail(err, kDf + "a chunk's tail inject failed");
    return true;
}

// ------------------------------------------------------------------ a draft step

/// llama.cpp's draft(): the noise block [anchor, mask x (block - 1)] at positions pos.. through the drafter (graph<false>, the
/// token batch: the embeddings, per block attn_norm -> q / k / v -> q / k norms -> RoPE -> the block's rows into the ring ->
/// non-causal attention with the sinks -> wo -> x value_scale -> + residual -> ffn_norm -> SwiGLU -> + residual; the final
/// norm; the head) - block position i (>= 1) is the draft for the token at pos + i: its argmax (llama.cpp's draft sampler,
/// top-k 10, reads its top candidate) or, coupled, the request's sampler with verify row i - 1's own uniform (df_cell_d).
/// The kernels only: the embeddings from df_in_d, the position from pos_dev when given, the drafts into df_tok_d / df_prob_d.
void Stage::dflash_block_ops(int64_t pos, const int64_t* pos_dev) {
    Engine::Impl::Dflash& d = *m.dfl;
    const int n = (int) hp.n_embd, V = (int) hp.n_vocab, T = d.block, H = (int) d.H, nkv = (int) d.nkv, hd = (int) d.hd;
    const int qw = H * hd, kvw = nkv * hd, nff = (int) d.n_ff, ND = T - 1;
    const DT& head = d.output ? d.output : m.output;
    copy_f32(df_in_d, df_x, (size_t) T * n, s);
    for (const Engine::Impl::Dflash::Blk& b : d.b) {
        rms_norm(df_x, b.attn_norm.f(), df_nn, T, n, d.eps, s);
        quant(df_nn, df_bxq, n, T);
        dflash_mm(b.wq, df_bxq, df_q, T);
        dflash_mm(b.wk, df_bxq, df_kvn, T);
        dflash_mm(b.wv, df_bxq, df_kvn + (size_t) T * kvw, T);
        rms_norm(df_q, b.q_norm.f(), df_q, T * H, hd, d.eps, s);
        rms_norm(df_kvn, b.k_norm.f(), df_kvn, T * nkv, hd, d.eps, s);
        rope_neox_w(df_q, qw, 0, H, hd, (int) d.n_rot, d.rope_base, pos, T, s, pos_dev);
        rope_neox_w(df_kvn, kvw, 0, nkv, hd, (int) d.n_rot, d.rope_base, pos, T, s, pos_dev);
        kv_store_w(df_kvn, kvw, 0, T * kvw, b.kc, b.vc, d.ring, nkv, hd, hd, pos, T, s, pos_dev);
        gqa_attend_w(df_q, qw, b.kc, b.vc, d.ring, H, nkv, hd, hd, (int) d.window, b.sinks.f(), 1.0f / std::sqrt((float) hd), df_att,
                     df_part, d.chunks, pos, T, s, pos_dev, true);
        quant(df_att, df_bxq, qw, T);
        dflash_mm(b.wo, df_bxq, df_o, T);
        if (d.v_scale != 0.0f) scale_f32(df_o, T * n, d.v_scale, s);
        add_f32(df_x, df_o, df_x, T * n, s);
        rms_norm(df_x, b.ffn_norm.f(), df_nn, T, n, d.eps, s);
        quant(df_nn, df_bxq, n, T);
        dflash_mm(b.up, df_bxq, df_fu, T);
        dflash_mm(b.gate, df_bxq, df_fg, T);
        swiglu_clamp(df_fg, df_fu, df_fg, T * nff, 0.0f, s);
        quant(df_fg, df_bxq, nff, T);
        dflash_mm(b.down, df_bxq, df_o, T);
        add_f32(df_x, df_o, df_x, T * n, s);
    }
    // the final norm and the head over block positions 1 .. T - 1 (position 0's output is never read)
    rms_norm(df_x + n, d.out_norm.f(), df_nn, ND, n, d.eps, s);
    quant(df_nn, df_bxq, n, ND);
    dflash_mm(head, df_bxq, mtp_logits, ND);
    if (cd_on) {
        for (int i = 0; i < ND; ++i)
            strata::kernels::coupled_draft_sample(mtp_logits + (size_t) i * V, V, nullptr, nullptr, V, cd_params, cd_ring, kCdCap, i,
                                                  df_cell_d + i, cd_scratch, mtp_amax + i, mtp_prob + i, s);
    } else {
        argmax_prob_w(mtp_logits, V, mtp_amax, mtp_prob, ND, s);
    }
    store_drafts(mtp_amax, mtp_prob, ND, df_tok_d, df_prob_d, s);
}

/// The DFlash catch-up after a window (Stage::mtp_catchup): its kept rows [pos0, pos0 + rows) into the rings, then the block
/// anchored at `anchor` (the token after them) at pos0 + rows.  A window's two parts as ONE captured graph per (rows, coupled)
/// after a variant's first eager use (Strata's drafter graphs: its inputs in mapped memory the host fills before the launch).
bool Stage::dflash_step(int rows, int32_t anchor, int64_t pos0, std::string& err) {
    Engine::Impl::Dflash& d = *m.dfl;
    static const bool timing = std::getenv("STRATA_X_DFLASH_TIMING") != nullptr;   // the step's time (synced at its end anyway)
    static double t_all = 0;
    static int t_cnt = 0;
    const auto t0 = clk::now();
    const int n = (int) hp.n_embd, V = (int) hp.n_vocab, ND = d.block - 1;
    df_n = 0;
    if (rows < 1 || rows > kTmax) return m.fail(err, kDf + "a window of " + std::to_string(rows) + " rows");
    if (anchor < 0 || anchor >= V) return m.fail(err, kDf + "anchor token out of the vocabulary");
    const int64_t pos = pos0 + rows;
    // the step's inputs (mapped; the previous step's reads ended with its sync): the anchor's embedding (rows 1.. hold the
    // mask's since the load), the two positions, the coupled cells - draft i is checked by the window's row i (position
    // pos + i), which draws Philox(seed, pos + i): its cell pos + i - 1
    if (anchor == d.mask) std::memcpy(df_in_h, d.mask_emb.data(), (size_t) n * sizeof(float));
    else if (d.own_embd) ggml_get_type_traits((ggml_type) d.embd_type)->to_float(d.embd + (size_t) anchor * d.embd_row, df_in_h, n);
    else if (!embd_rows(&anchor, 1, df_in_h, -1)) return m.fail(err, kDf + "the anchor's embedding");
    *(volatile int64_t*) (df_pos_h + 0) = pos0;
    *(volatile int64_t*) (df_pos_h + 1) = pos;
    if (cd_on)
        for (int i = 0; i < ND; ++i) *(volatile int32_t*) (df_cell_h + i) = (int32_t) (pos + i - 1);
    const int key = rows * 2 + (cd_on ? 1 : 0);
    bool graph = m.graphs_on && !df_graphs_off;
    if (graph && !df_gx.count(key)) {
        if (++df_gx_uses[key] >= 2) {   // the first use of a variant runs eagerly
            cudaGraph_t g = nullptr;
            std::string why;
            if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
                try {
                    dflash_inject_ops(rows, 0, df_pos_d + 0);
                    dflash_block_ops(0, df_pos_d + 1);
                } catch (const std::exception& e) {
                    why = e.what();
                }
                const cudaError_t ce = cudaStreamEndCapture(s, &g);
                cudaGraphExec_t ex = nullptr;
                if (why.empty() && ce == cudaSuccess && g && cudaGraphInstantiate(&ex, g, 0) == cudaSuccess) {
                    df_gx[key] = ex;
                    if (m.opt.verbose)
                        fprintf(stderr, "strata-x: GPU %d: captured the DFlash graph for windows of %d row%s%s\n", dev, rows,
                                rows == 1 ? "" : "s", cd_on ? " (coupled drafts)" : "");
                } else {
                    fprintf(stderr, "strata-x: DFlash graphs off: %s\n", why.empty() ? cudaGetErrorString(ce) : why.c_str());
                    df_graphs_off = true;
                }
                if (g) cudaGraphDestroy(g);
            } else {
                df_graphs_off = true;
            }
            cudaGetLastError();
        }
        graph = df_gx.count(key) != 0;
    }
    try {
        if (graph) {
            if (cudaGraphLaunch(df_gx[key], s) != cudaSuccess) return m.fail(err, kDf + "the graph launch failed");
        } else {
            dflash_inject_ops(rows, pos0, nullptr);
            dflash_block_ops(pos, nullptr);
        }
    } catch (const std::exception& e) {
        return m.fail(err, kDf + "draft: " + e.what());
    }
    if (cudaStreamSynchronize(s) != cudaSuccess) return m.fail(err, kDf + "the draft step failed");
    for (int i = 0; i < ND; ++i) {
        df_tok[i] = *(volatile int32_t*) (df_tok_h + i);
        df_prob[i] = *(volatile float*) (df_prob_h + i);
    }
    df_n = ND;
    df_last_ms = ms_since(t0);
    static const bool print = std::getenv("STRATA_X_DFLASH_PRINT") != nullptr;   // round by round, against mainline's -lv 4 log
    if (print) {
        fprintf(stderr, "strata-x: DFlash round at %lld (anchor %d):", (long long) pos, anchor);
        for (int i = 0; i < ND; ++i) fprintf(stderr, " %d:%.3f", df_tok[i], df_prob[i]);
        fprintf(stderr, "\n");
    }
    if (timing) {
        t_all += df_last_ms;
        if (++t_cnt % 50 == 0)
            fprintf(stderr, "strata-x: DFlash draft step (%s): %.2f ms (mean of %d)\n", graph ? "graph" : "eager", t_all / t_cnt, t_cnt);
    }
    return true;
}

}  // namespace strata::x
