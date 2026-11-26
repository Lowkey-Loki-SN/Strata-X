// src/x/hy4_dfly.cpp - moex: Hy4-flash's DFly2 drafter (moex/docs/HY4_PORT.md, "Speculative decoding and images"), as vLLM
// runs it (PR #68296: vllm/models/hy_v4/nvidia/dfly.py, dfly_speculator.py HYV4DFlyQueryModel, dfly_inputs.py):
//   * the model: five dense MLA layers (HYV4DecoderLayer without iHC: x += attn(norm(x)); x += ffn(norm(x)); the gated MLA with
//     sinks and its post-projection norm, as the trunk's; a plain SwiGLU) and a final norm, over the trunk's embedding table and
//     head; the anchor row fused with the trunk's normed last row (mtp_fusion: x + gate * eh_proj(enorm(x) | hnorm(h))); per
//     draft position a hidden correction (h + down(silu(gate(c)) * up(c)), c = hidden_norm(h) | embed_norm(the previous token's
//     embedding)) before the head;
//   * the attention: draft layer i reads trunk layer target_kv_layers[i]'s KV cache - its latent rows, decoded by the draft's
//     own k_b / v_b - at the trunk's cells for the last kept row (target_index_layer[i], up to 2,048 positions, all before the
//     block) and the block's own rows up to its own (causal); the block's rows are written into that cache at their positions,
//     which the next verify window rewrites (vLLM's KV sharing and its reuse indices: _build_dfly_reuse_indices_kernel, no
//     local window here: dfly_context_local_window is null);
//   * the loop: the query block [bonus, mask x (block - 1)] at the position after the window's last kept row; drafts by argmax.
// Here: the weights on the last GPU (moex/tools/hy4/dfly_to_gguf.py's GGUF: Q8_0 matrices); the trunk's target layers must run
// there too; their cells and the head's normed rows are captured into rings of kTmax rows by position during every window; the
// step runs eagerly after each window (5 head passes, each synced for the next correction's embedding).
#include "engine_impl.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/x/hy4_kernels.hpp"
#include "strata/kernels/elementwise.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace strata::x {

namespace {
const std::string kDy = "the DFly2 drafter: ";
}  // namespace

bool Engine::Impl::open_dfly(std::string& err) {
    if (opt.dfly.empty()) return true;
    if (hp.arch != Arch::hy4) return fail(err, "--draft-dfly drafts for Hy4-flash (an hy_v4_flash GGUF)");
    auto d = std::make_unique<Hy4Dfly>();
    try {
        d->gm = std::make_unique<strata::GgufModel>(strata::GgufModel::open(opt.dfly));
    } catch (const std::exception& e) {
        return fail(err, kDy + e.what());
    }
    const strata::GgufFile& mt = d->gm->meta();
    const strata::MetaValue* a = mt.get("general.architecture");
    if (!a || a->s != "hy4_dfly") return fail(err, kDy + "not an hy4_dfly GGUF (moex/tools/hy4/dfly_to_gguf.py)");
    auto num = [&](const char* k, double& v) -> bool {
        const strata::MetaValue* x = mt.get(std::string("hy4_dfly.") + k);
        if (!x) return fail(err, kDy + "missing hy4_dfly." + k);
        v = x->num();
        return true;
    };
    auto arr = [&](const char* k, std::vector<int64_t>& v) -> bool {
        const strata::MetaValue* x = mt.get(std::string("hy4_dfly.") + k);
        if (!x || x->type != strata::MetaType::ARRAY) return fail(err, kDy + "missing the array hy4_dfly." + k);
        for (const auto& it : x->items) v.push_back((int64_t) it.num());
        return true;
    };
    double nl = 0, emb = 0, nff = 0, heads = 0, ql = 0, kvl = 0, kl = 0, vl = 0, rot = 0, eps = 0, mask = 0, topk = 0, win = 0;
    if (!num("block_count", nl) || !num("embedding_length", emb) || !num("feed_forward_length", nff) ||
        !num("attention.head_count", heads) || !num("attention.q_lora_rank", ql) || !num("attention.kv_lora_rank", kvl) ||
        !num("attention.key_length_mla", kl) || !num("attention.value_length_mla", vl) || !num("rope.dimension_count", rot) ||
        !num("attention.layer_norm_rms_epsilon", eps) || !num("mask_token_id", mask) || !num("target_index_topk", topk) ||
        !num("context_local_window", win) || !arr("target_kv_layers", d->tkv) || !arr("target_index_layers", d->tidx))
        return false;
    // what this engine runs: the trunk's MLA geometry (the attention kernels' 576-wide rows, 192 + 64 query heads), heads in
    // groups of 16, no local window
    if ((int64_t) emb != hp.n_embd || (int64_t) ql != hp.q_lora || (int64_t) kvl != hp.kv_lora || (int64_t) kl != hp.k_mla ||
        (int64_t) vl != hp.v_mla || (int64_t) rot != hp.n_rot)
        return fail(err, kDy + "its MLA geometry differs from the trunk's");
    if (heads < 16 || (int64_t) heads % 16) return fail(err, kDy + std::to_string((int64_t) heads) + " heads (a multiple of 16 supported)");
    if (win != 0) return fail(err, kDy + "a local window (dfly_context_local_window) is not supported");
    d->n = (int) nl;
    d->H = (int64_t) heads;
    d->n_ff = (int64_t) nff;
    d->eps = (float) eps;
    d->mask = (int32_t) mask;
    d->topk = (int) topk;
    if (d->n < 1 || (int) d->tkv.size() != d->n || (int) d->tidx.size() != d->n) return fail(err, kDy + "one target layer per draft layer");
    if (d->mask < 0 || d->mask >= hp.n_vocab) return fail(err, kDy + "its mask token is outside the vocabulary");
    d->slot.assign((size_t) hp.n_layer, -1);
    for (int i = 0; i < d->n; ++i)
        for (int64_t il : {d->tkv[(size_t) i], d->tidx[(size_t) i]}) {
            if (il < 1 || il >= hp.n_layer || hp.is_swa(il)) return fail(err, kDy + "target layer " + std::to_string(il) + " is not a sparse trunk layer");
        }
    for (int i = 0; i < d->n; ++i) {
        const int64_t il = d->tidx[(size_t) i];
        if (d->slot[(size_t) il] < 0) d->slot[(size_t) il] = d->n_slots++;
    }
    d->block = std::min(6, kTmax);   // vLLM's num_speculative_tokens 5 + the bonus row
    for (const char* t : {"dfly.fusion.eh_proj.weight", "dfly.corr.gate.weight", "dfly.corr.up.weight", "dfly.corr.down.weight",
                          "dfly.output_norm.weight", "dfly.blk.0.attn_k_b.weight", "dfly.blk.0.attn_v_b.weight"})
        if (!d->gm->find(t)) return fail(err, kDy + "tensor " + t + " missing");
    uint64_t w = 0;
    for (size_t sh = 0; sh < d->gm->size(); ++sh)
        for (const auto& ti : d->gm->shard(sh).tensors()) w += (strata::tensor_payload_bytes(ti) + 255) & ~uint64_t(255);
    const int64_t n = hp.n_embd, W = d->block, H = d->H;
    const int chunks = hy4_attend_chunks(d->topk + W);
    const uint64_t bufs = (uint64_t) (d->n_slots + 1) * kTmax * (2052 + n) * 4 +
                          (uint64_t) W * (4 * n + hp.q_lora + H * 256 + 576 + H * 576 + H * 192 + 3 * H * 512 + 3 * H * 256 + 2 * d->n_ff) * 4 +
                          hy4_attend_scratch_bytes((int) H, chunks, (int) W) + (uint64_t) (8 * n + hp.n_vocab) * 4 + (32ull << 20);
    dfly_vram = w + bufs;
    if (opt.verbose) {
        fprintf(stderr, "strata-x: DFly2 drafter: %d layers of %lld heads, FFN %lld, block %d, mask %d, top-k %d, target KV layers",
                d->n, (long long) d->H, (long long) d->n_ff, d->block, d->mask, d->topk);
        for (int64_t il : d->tkv) fprintf(stderr, " %lld", (long long) il);
        fprintf(stderr, "; %.2f GiB on the last GPU\n", dfly_vram / 1073741824.0);
    }
    dfy = std::move(d);
    return true;
}

bool Engine::Impl::load_dfly(std::string& err) {
    if (!dfy) return true;
    Hy4Dfly& d = *dfy;
    Stage& lg = last_stage();
    for (int i = 0; i < d.n; ++i)
        for (int64_t il : {d.tkv[(size_t) i], d.tidx[(size_t) i]})
            if (&stage_of(il) != &lg)
                return fail(err, kDy + "trunk layer " + std::to_string(il) + " runs on GPU " + std::to_string(stage_of(il).dev) +
                                     ", not the last GPU: its KV cache and cells must be there (a layer split before layer " +
                                     std::to_string(*std::min_element(d.tkv.begin(), d.tkv.end())) + ")");
    // the weights into one allocation on the last GPU
    struct Item {
        const strata::TensorInfo* ti;
        size_t shard;
        uint64_t bytes, off;
    };
    std::vector<Item> items;
    uint64_t total = 0;
    for (size_t sh = 0; sh < d.gm->size(); ++sh)
        for (const auto& ti : d.gm->shard(sh).tensors()) {
            const uint64_t b = strata::tensor_payload_bytes(ti);
            if (!b || !d.gm->in_bounds(ti, sh)) return fail(err, kDy + "tensor " + ti.name + " of unknown size");
            items.push_back({&ti, sh, b, total});
            total += (b + 255) & ~uint64_t(255);
        }
    set_dev(lg.dev);
    if (cudaMalloc(&d.dense, total) != cudaSuccess) {
        cudaGetLastError();
        return fail(err, kDy + std::to_string(total >> 20) + " MiB of weights on GPU " + std::to_string(lg.dev));
    }
    for (const auto& it : items) {
        uint8_t* dst = static_cast<uint8_t*>(d.dense) + it.off;
        if (cudaMemcpy(dst, d.gm->shard(it.shard).tensor_data(*it.ti), it.bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail(err, kDy + "upload of " + it.ti->name);
        DT x;
        x.d = dst;
        x.type = (int) it.ti->type;
        for (size_t k = 0; k < it.ti->shape.size() && k < 4; ++k) x.ne[k] = (int64_t) it.ti->shape[k];
        x.bytes = it.bytes;
        d.t[it.ti->name] = x;
    }
    auto get = [&](const std::string& nm, DT& x) -> bool {
        auto f = d.t.find(nm);
        if (f == d.t.end()) return fail(err, kDy + "tensor " + nm + " missing");
        x = f->second;
        return true;
    };
    bool ok = get("dfly.fusion.enorm.weight", d.f_enorm) && get("dfly.fusion.hnorm.weight", d.f_hnorm) &&
              get("dfly.fusion.gate.weight", d.f_gate) && get("dfly.fusion.eh_proj.weight", d.f_eh) &&
              get("dfly.corr.hidden_norm.weight", d.c_hnorm) && get("dfly.corr.embed_norm.weight", d.c_enorm) &&
              get("dfly.corr.gate.weight", d.c_gate) && get("dfly.corr.up.weight", d.c_up) && get("dfly.corr.down.weight", d.c_down) &&
              get("dfly.output_norm.weight", d.out_norm);
    if (!ok) return false;
    d.b.resize((size_t) d.n);
    for (int i = 0; i < d.n; ++i) {
        auto bn = [&](const char* sfx) { return "dfly.blk." + std::to_string(i) + "." + sfx; };
        Hy4Dfly::Blk& b = d.b[(size_t) i];
        ok = get(bn("attn_norm.weight"), b.attn_norm) && get(bn("attn_q_a.weight"), b.q_a) && get(bn("attn_q_a_norm.weight"), b.q_a_norm) &&
             get(bn("attn_q_b.weight"), b.q_b) && get(bn("attn_kv_a_mqa.weight"), b.kv_a) && get(bn("attn_kv_a_norm.weight"), b.kv_a_norm) &&
             get(bn("attn_k_b.weight"), b.k_b) && get(bn("attn_v_b.weight"), b.v_b) && get(bn("attn_sinks.weight"), b.sinks) &&
             get(bn("attn_gate.weight"), b.gate) && get(bn("attn_output.weight"), b.wo) && get(bn("attn_post_norm.weight"), b.post_norm) &&
             get(bn("ffn_norm.weight"), b.ffn_norm) && get(bn("ffn_gate.weight"), b.ffn_gate) && get(bn("ffn_up.weight"), b.ffn_up) &&
             get(bn("ffn_down.weight"), b.ffn_down);
        if (!ok) return false;
        for (const DT* x : {&b.q_a, &b.q_b, &b.kv_a, &b.gate, &b.wo, &b.ffn_gate, &b.ffn_up, &b.ffn_down})
            if (!kn::native_mmvq_supported(x->type)) return fail(err, kDy + "a matrix of ggml type " + std::to_string(x->type) + " (Q8_0 expected)");
    }
    // the mask row of the trunk's embedding table
    d.mask_emb.assign((size_t) hp.n_embd, 0.0f);
    if (!lg.embd_rows(&d.mask, 1, d.mask_emb.data(), -1)) return fail(err, kDy + "the mask token's embedding");
    // the last stage's buffers
    const int64_t n = hp.n_embd, W = d.block, H = d.H, Lq = hp.q_lora, V = hp.n_vocab;
    Stage::Dfy& b = lg.dfy;
    b.stride = (int) (max_cells + W);
    const int64_t wide = std::max<int64_t>({2 * n, H * 256, d.n_ff});
    const int chunks = hy4_attend_chunks(b.stride);
    auto fl = [&](float*& p, int64_t cnt) { return (p = lg.dalloc<float>((size_t) cnt, err)) != nullptr; };
    ok = (b.cells = lg.dalloc<int32_t>((size_t) d.n_slots * kTmax * max_cells, err)) != nullptr && fl(b.h, kTmax * n) &&
         fl(b.x, W * n) && fl(b.xn, W * n) && fl(b.qr, W * Lq) && fl(b.q, W * H * 256) && fl(b.kv, W * 576) && fl(b.qf, W * H * 576) &&
         fl(b.qhm, H * W * 192) && fl(b.qahm, H * W * 512) && fl(b.ctx, W * H * 512) && fl(b.ctxhm, H * W * 512) &&
         fl(b.o2hm, H * W * 256) && fl(b.o2, W * H * 256) && fl(b.gate, W * H * 256) && fl(b.o, W * n) && fl(b.fg, W * d.n_ff) &&
         fl(b.fu, W * d.n_ff) && fl(b.part, (int64_t) (hy4_attend_scratch_bytes((int) H, chunks, (int) W) / sizeof(float)) + 1) &&
         fl(b.cat, 2 * n) && fl(b.cg, n) && fl(b.cu, n) && fl(b.cd, n) && fl(b.hc, n) && fl(b.logits, V) && fl(b.emb0, n) &&
         (b.xq = lg.dalloc<uint8_t>(kn::native_q8_1_bytes((int) (W * wide)), err)) != nullptr &&
         (b.xq2 = lg.dalloc<uint8_t>(kn::native_q8_1_bytes((int) (W * wide)), err)) != nullptr &&
         (b.dcells = lg.dalloc<int32_t>((size_t) W * b.stride, err)) != nullptr && (b.counts = lg.dalloc<int32_t>((size_t) W, err)) != nullptr;
    if (!ok) return false;
    auto mapped = [&](size_t bytes, void** h, void** dptr) {
        if (cudaHostAlloc(h, bytes, cudaHostAllocMapped) != cudaSuccess || cudaHostGetDevicePointer(dptr, *h, 0) != cudaSuccess) return false;
        std::memset(*h, 0, bytes);
        return true;
    };
    if (!mapped((size_t) (kTmax + 1) * n * sizeof(float), (void**) &b.emb_h, (void**) &b.emb_d) ||
        !mapped(kTmax * sizeof(int32_t), (void**) &b.tok_h, (void**) &b.tok_d) ||
        !mapped(kTmax * sizeof(float), (void**) &b.prob_h, (void**) &b.prob_d) ||
        !mapped(kTmax * sizeof(int32_t), (void**) &b.cell_h, (void**) &b.cell_d))
        return fail(err, kDy + "its mapped buffers");
    reset_dev_cache();
    if (opt.verbose)
        fprintf(stderr, "strata-x: DFly2 drafter loaded: %.2f GiB of weights on GPU %d, its block's cells %d a row\n", total / 1073741824.0,
                lg.dev, b.stride);
    return true;
}

void Stage::dfly_capture_cells(int il, const int32_t* cells, int64_t pos0, int T) {
    if (!m.dfy || !dfy.cells || T > kTmax || il < 0 || il >= (int) m.dfy->slot.size()) return;
    const int k = m.dfy->slot[(size_t) il];
    if (k < 0) return;
    hy4_snap_rows_w(cells, dfy.cells + (size_t) k * kTmax * m.max_cells, m.max_cells * (int) sizeof(int32_t), T, kTmax, pos0, s, gpos);
}

void Stage::dfly_capture_h(int64_t pos0, int T) {
    if (!m.dfy || !dfy.h || T > kTmax) return;
    hy4_snap_rows_w(h, dfy.h, (int) (hp.n_embd * sizeof(float)), T, kTmax, pos0, s, nullptr);
}

bool Stage::dfly_step(int rows, int32_t bonus, int64_t pos0, std::string& err) {
    Engine::Impl::Hy4Dfly& d = *m.dfy;
    const auto t0 = clk::now();
    const int n = (int) hp.n_embd, V = (int) hp.n_vocab, H = (int) d.H, Lq = (int) hp.q_lora, nff = (int) d.n_ff;
    constexpr int Dk = 256, Dn = 192, Dr = 64, Lkv = 512, Dv = 256;
    const int KP = (int) hp.kpool, n_top = m.hy4_n_top();
    dfy_n = 0;
    if (rows < 1 || rows > kTmax) return m.fail(err, kDy + "a window of " + std::to_string(rows) + " rows");
    if (bonus < 0 || bonus >= V) return m.fail(err, kDy + "a bonus token outside the vocabulary");
    const int64_t anchor = pos0 + rows;   // the bonus token's position: the block's first row
    const int W = (int) std::min<int64_t>(d.block, m.opt.n_ctx - anchor);
    if (W < 2) return true;               // no room for a draft before the context's end
    const int src = (int) ((anchor - 1) % kTmax);   // the last kept row's slot in the rings
    cudaSetDevice(dev);
    // 1. the block's embeddings (mapped): the bonus token's, then the mask's
    if (!embd_rows(&bonus, 1, dfy.emb_h, -1)) return m.fail(err, kDy + "the bonus token's embedding");
    for (int r = 1; r < W; ++r) std::memcpy(dfy.emb_h + (size_t) r * n, d.mask_emb.data(), (size_t) n * sizeof(float));
    cudaMemcpyAsync(dfy.x, dfy.emb_d, (size_t) W * n * sizeof(float), cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(dfy.emb0, dfy.emb_d, (size_t) n * sizeof(float), cudaMemcpyDeviceToDevice, s);
    // 2. the bonus row fused with the trunk's normed row before it: x0 += gate * eh_proj(enorm(x0) | hnorm(h))
    rms_norm(dfy.x, d.f_enorm.f(), dfy.cat, 1, n, d.eps, s);
    rms_norm(dfy.h + (size_t) src * n, d.f_hnorm.f(), dfy.cat + n, 1, n, d.eps, s);
    quant(dfy.cat, dfy.xq, 2 * n, 1);
    pj(d.f_eh, dfy.cat, dfy.xq, dfy.o, 1);
    hy4_gated_add(dfy.x, d.f_gate.f(), dfy.o, n, s);
    // 3. the layers: x += attn(norm(x)); x += ffn(norm(x))
    const int n_hist = hy4_cells(anchor - 1, n_top, KP);
    const int chunks = hy4_attend_chunks(dfy.stride);
    const float scale = 1.0f / std::sqrt((float) Dk);
    for (int i = 0; i < d.n; ++i) {
        const Engine::Impl::Hy4Dfly::Blk& b = d.b[(size_t) i];
        const Layer& tl = L[(size_t) d.tkv[(size_t) i]];
        const int k = d.slot[(size_t) d.tidx[(size_t) i]];
        rms_norm(dfy.x, b.attn_norm.f(), dfy.xn, W, n, d.eps, s);
        quant(dfy.xn, dfy.xq, n, W);
        pj(b.q_a, dfy.xn, dfy.xq, dfy.qr, W);
        pj(b.kv_a, dfy.xn, dfy.xq, dfy.kv, W);
        pj(b.gate, dfy.xn, dfy.xq, dfy.gate, W);
        rms_norm(dfy.qr, b.q_a_norm.f(), dfy.qr, W, Lq, d.eps, s);
        quant(dfy.qr, dfy.xq2, Lq, W);
        pj(b.q_b, dfy.qr, dfy.xq2, dfy.q, W);
        // the block's latent rows into the trunk layer's cache at their positions (the next verify window rewrites them)
        hy4_kv_row_w(dfy.kv, b.kv_a_norm.f(), d.eps, hp.rope_base, nullptr, tl.kv_rows, anchor, W, s, nullptr, hy_mpos);
        hy4_store_kv_w(dfy.kv, tl.kv_cache, tl.kv_rows, anchor, W, s, nullptr);
        hy4_rope_w(dfy.q, H * Dk, 0, H, Dk, Dn, Dr, hp.rope_base, anchor, W, s, nullptr, hy_mpos);
        hy4_q_split(dfy.q, dfy.qhm, dfy.qf, H, W, s);
        heads_mm(b.k_b, dfy.qhm, dfy.qahm, H, Dn, Lkv, W);
        hy4_q_join(dfy.qahm, dfy.qf, H, W, s);
        // each row's cells: the trunk's for the last kept row, then the block up to the row (causal)
        hy4_dfly_cells_w(dfy.cells + ((size_t) k * kTmax + src) * m.max_cells, n_hist, anchor, W, dfy.dcells, dfy.stride, dfy.counts, s);
        hy4_attend_w(dfy.qf, tl.kv_cache, tl.kv_rows, dfy.dcells, dfy.stride, n_top, KP, 0, b.sinks.f(), scale, dfy.ctx, dfy.part, chunks,
                     H, anchor, W, s, nullptr, dfy.counts);
        permute_abc(dfy.ctx, dfy.ctxhm, W, H, Lkv, s);
        heads_mm(b.v_b, dfy.ctxhm, dfy.o2hm, H, Lkv, Dv, W);
        permute_abc(dfy.o2hm, dfy.o2, H, W, Dv, s);
        hy4_gate_mul(dfy.o2, dfy.gate, (int64_t) W * H * Dv, s);
        quant(dfy.o2, dfy.xq, H * Dv, W);
        pj(b.wo, dfy.o2, dfy.xq, dfy.o, W);
        rms_norm(dfy.o, b.post_norm.f(), dfy.o, W, n, d.eps, s);
        kn::add_inplace(dfy.x, dfy.o, (int64_t) W * n, s);
        rms_norm(dfy.x, b.ffn_norm.f(), dfy.xn, W, n, d.eps, s);
        quant(dfy.xn, dfy.xq, n, W);
        pj(b.ffn_gate, dfy.xn, dfy.xq, dfy.fg, W);
        pj(b.ffn_up, dfy.xn, dfy.xq, dfy.fu, W);
        swiglu_clamp(dfy.fg, dfy.fu, dfy.fg, W * nff, 0.0f, s);   // (a plain SwiGLU: the routed experts' limit is not the drafter's)
        quant(dfy.fg, dfy.xq, nff, W);
        pj(b.ffn_down, dfy.fg, dfy.xq, dfy.o, W);
        kn::add_inplace(dfy.x, dfy.o, (int64_t) W * n, s);
    }
    rms_norm(dfy.x, d.out_norm.f(), dfy.xn, W, n, d.eps, s);
    // 4. the drafts: row j + 1 corrected by the embedding of the token before it (the bonus, then each draft), then the head.
    // Coupled (cd_on: a sampled request, coupled_draft.hpp): draft j - the token at anchor + j + 1 - is sampled with the request's
    // chain and the noise of the row that verifies it, the next window's row j at anchor + j (cell anchor + j - 1); a draft
    // only decides how many tokens a window keeps, so the output is the same either way
    float* prev_emb = dfy.emb_d + (size_t) W * n;   // (mapped: the last draft's row, written after its sync)
    if (cd_on)
        for (int j = 0; j + 1 < W; ++j) *(volatile int32_t*) (dfy.cell_h + j) = (int32_t) (anchor + j - 1);
    for (int j = 0; j + 1 < W; ++j) {
        const float* row = dfy.xn + (size_t) (j + 1) * n;
        rms_norm(row, d.c_hnorm.f(), dfy.cat, 1, n, d.eps, s);
        rms_norm(j == 0 ? dfy.emb0 : prev_emb, d.c_enorm.f(), dfy.cat + n, 1, n, d.eps, s);
        quant(dfy.cat, dfy.xq, 2 * n, 1);
        pj(d.c_gate, dfy.cat, dfy.xq, dfy.cg, 1);
        pj(d.c_up, dfy.cat, dfy.xq, dfy.cu, 1);
        swiglu_clamp(dfy.cg, dfy.cu, dfy.cg, n, 0.0f, s);
        quant(dfy.cg, dfy.xq, n, 1);
        pj(d.c_down, dfy.cg, dfy.xq, dfy.cd, 1);
        cudaMemcpyAsync(dfy.hc, row, (size_t) n * sizeof(float), cudaMemcpyDeviceToDevice, s);
        kn::add_inplace(dfy.hc, dfy.cd, n, s);
        if (m.output.type == GGML_TYPE_BF16) {
            gemv_dt(m.output, dfy.hc, dfy.logits, n, V, 1);
        } else {
            quant(dfy.hc, dfy.xq, n, 1);
            mm_t(m.output, dfy.xq, dfy.logits, 1);
        }
        if (cd_on)
            strata::kernels::coupled_draft_sample(dfy.logits, V, nullptr, nullptr, V, cd_params, cd_ring, kCdCap, j, dfy.cell_d + j,
                                                  cd_scratch, dfy.tok_d + j, dfy.prob_d + j, s);
        else
            argmax_prob_w(dfy.logits, V, dfy.tok_d + j, dfy.prob_d + j, 1, s);
        if (cudaStreamSynchronize(s) != cudaSuccess) return m.fail(err, kDy + "a draft step failed on GPU " + std::to_string(dev));
        const int32_t t = *(volatile int32_t*) (dfy.tok_h + j);
        dfy_tok[j] = t;
        dfy_prob[j] = *(volatile float*) (dfy.prob_h + j);
        if (t < 0 || t >= V) return m.fail(err, kDy + "a draft outside the vocabulary");
        if (j + 2 < W && !embd_rows(&t, 1, dfy.emb_h + (size_t) W * n, -1)) return m.fail(err, kDy + "a draft's embedding");
    }
    dfy_n = W - 1;
    dfy_last_ms = ms_since(t0);
    static const bool trace = std::getenv("STRATA_HY4_DFLY_TRACE") != nullptr;
    if (trace) {
        fprintf(stderr, "strata-x: DFly2 at %lld (%.2f ms):", (long long) anchor, dfy_last_ms);
        for (int j = 0; j < dfy_n; ++j) fprintf(stderr, " %d:%.3f", dfy_tok[j], dfy_prob[j]);
        fprintf(stderr, "\n");
    }
    return true;
}

}  // namespace strata::x
