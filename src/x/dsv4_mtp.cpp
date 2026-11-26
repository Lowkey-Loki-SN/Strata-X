// src/x/dsv4_mtp.cpp - moex: DeepSeek-V4's NextN head, as llama.cpp runs it (-md mtp-*.gguf --spec-type draft-mtp):
//   * the model: the original DeepSeek-V4-Flash's single MTP block (deepseek-ai/DeepSeek-V4-Flash mtp.0; convert_hf_to_gguf.py
//     --mtp writes it as blk.<n_layer>.* with nextn.{eh_proj, enorm, hnorm, shared_head_norm}, the head reduce output_hc_*
//     and the checkpoint's own token_embd / output), run by deepseek4.cpp graph_mtp:
//         z = eh_proj([enorm(emb(tok)) | hnorm(h_s)]) for each of the trunk's four residual streams h_s (one row per stream)
//         -> one DeepSeek-V4 block (mHC x 4; ratio 0: the 128-position raw window, sinks, the grouped output; its own 256
//         routed experts and a shared one) -> the head reduce (its output_hc_*) -> shared_head_norm -> its head;
//     the block's output streams are the next chain step's h (llama.cpp's t_h_nextn);
//   * the loop: Strata's MTP loop as GLM's NextN runs it (mtp.cpp, spec.cpp): the catch-up over a window's kept rows
//     pairs the trunk's streams at position p with the token at p + 1; a chain step feeds the block's own streams back.
// Here: L[n_layer] on the last GPU, so the trunk's DeepSeek-V4 attention and mHC code run it unchanged; its routed experts in a
// pinned RAM arena of their own, computed by the CPU pool (as DSpark's); eager (the router's ids go to the host).
// The trunk need not be the one the head was trained with: DeepSeek-V4-Flash-0731 and -Vision-Exp ship DSpark instead, and
// the original's head drafts for Vision-Exp at 47% (temperature 1) to 50% (greedy) per draft in llama.cpp (3 drafts a step,
// 2026-10-08) - its weights are 7-26% (relative L2) away from that trunk's.
#include "engine_impl.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include "ggml.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace strata::x {

namespace {
/// A GGUF tensor's rows as F32 on the host (F32 copied; any type with ggml's to_float dequantized).
bool rows_to_f32(const strata::TensorInfo& ti, const uint8_t* src, int64_t ne0, int64_t rows, float* dst) {
    if (ti.type == GGML_TYPE_F32) {
        std::memcpy(dst, src, (size_t) (ne0 * rows) * sizeof(float));
        return true;
    }
    const auto* tr = ggml_get_type_traits((ggml_type) ti.type);
    if (!tr || !tr->to_float) return false;
    const size_t rb = ggml_row_size((ggml_type) ti.type, ne0);
    for (int64_t r = 0; r < rows; ++r) tr->to_float(src + (size_t) r * rb, dst + (size_t) (r * ne0), ne0);
    return true;
}
}  // namespace

bool Engine::Impl::load_draft_mtp(std::string& err) {
    if (opt.draft_mtp.empty()) return true;
    if (hp.arch != Arch::dsv4) return fail(err, "--draft-mtp drafts for DeepSeek-V4 (GLM-5.3 and MiMo carry their MTP blocks in the GGUF)");
    if (dsp) return fail(err, "one drafter: --draft-dspark or --draft-mtp");
    auto d = std::make_unique<Dsv4Mtp>();
    try {
        d->gm = std::make_unique<strata::GgufModel>(strata::GgufModel::open(opt.draft_mtp));
    } catch (const std::exception& e) {
        return fail(err, std::string("the MTP head: ") + e.what());
    }
    const strata::GgufFile& mt = d->gm->meta();
    const strata::MetaValue* a = mt.get("general.architecture");
    if (!a || a->s != "deepseek4") return fail(err, "the MTP head: not a deepseek4 GGUF");
    auto num = [&](const char* k, double& v) -> bool {
        const strata::MetaValue* x = mt.get(std::string("deepseek4.") + k);
        if (!x) return fail(err, std::string("the MTP head: missing deepseek4.") + k);
        v = x->num();
        return true;
    };
    const int64_t il = hp.n_layer;
    /// a per-layer key's entry for the block (an array of block_count entries, or one value for every layer)
    auto at_block = [&](const char* k, double def) -> double {
        const strata::MetaValue* x = mt.get(std::string("deepseek4.") + k);
        if (!x) return def;
        if (x->type != strata::MetaType::ARRAY) return x->num();
        return (int64_t) x->items.size() > il ? x->items[(size_t) il].num() : def;
    };
    double nl = 0, nn = 0, emb = 0, heads = 0, kl = 0, rot = 0, ql = 0, swa = 0, ne = 0, nu = 0, nff = 0, gate = 0, nsh = 0;
    double ews = 0, hcm = 0, sk = 0, hce = 0, eps = 0, fb = 0, og = 0, ol = 0;
    if (!num("block_count", nl) || !num("nextn_predict_layers", nn) || !num("embedding_length", emb) ||
        !num("attention.head_count", heads) || !num("attention.key_length", kl) || !num("rope.dimension_count", rot) ||
        !num("attention.q_lora_rank", ql) || !num("attention.sliding_window", swa) || !num("expert_count", ne) ||
        !num("expert_used_count", nu) || !num("expert_feed_forward_length", nff) || !num("expert_gating_func", gate) ||
        !num("expert_shared_count", nsh) || !num("expert_weights_scale", ews) || !num("hyper_connection.count", hcm) ||
        !num("hyper_connection.sinkhorn_iterations", sk) || !num("hyper_connection.epsilon", hce) ||
        !num("attention.layer_norm_rms_epsilon", eps) || !num("rope.freq_base", fb) || !num("attention.output_group_count", og) ||
        !num("attention.output_lora_rank", ol))
        return false;
    // one NextN block right after this trunk's layers, through the trunk's DeepSeek-V4 code: the same geometry, eps, mHC, routing
    if ((int64_t) nn != 1 || (int64_t) nl != il + 1)
        return fail(err, "the MTP head: not one NextN block after this trunk's " + std::to_string(il) + " layers");
    if ((int64_t) emb != hp.n_embd || (int64_t) heads != hp.n_head || (int64_t) kl != hp.hd_k || (int64_t) rot != hp.n_rot ||
        (int64_t) ql != hp.q_lora || (int64_t) swa != hp.n_swa || (int64_t) ne != hp.n_expert || (int64_t) nu != hp.n_used ||
        (int64_t) nff != hp.n_ff_exp || gate != 4 || nsh != 1 || (float) ews != hp.ew_scale || (int64_t) hcm != hp.hc ||
        (int64_t) sk != hp.sinkhorn_iters || (float) hce != hp.hc_eps || (float) fb != hp.rope_base ||
        (int64_t) og != hp.o_groups || (int64_t) ol != hp.o_lora)
        return fail(err, "the MTP head's geometry differs from the trunk's (it must be a DeepSeek-V4-Flash head)");
    if (il < hp.n_hash) return fail(err, "the MTP head: a hash-routed block");
    if (!(eps > 0.0)) return fail(err, "the MTP head: no RMS norm eps");
    d->eps = (float) eps;   // its own: the trunk's may differ (DeepSeek-V4-Flash-Vision-Exp says 1e-20, the original 1e-6)
    if ((int) at_block("attention.compress_ratios", -1) != 0)
        return fail(err, "the MTP head: its block is not a ratio-0 (window-only) block, as llama.cpp's graph_mtp needs");
    const float cl_exp = (float) at_block("swiglu_clamp_exp", hp.clamp_exp.empty() ? 0.0 : hp.clamp_exp.back());
    const float cl_sh = (float) at_block("swiglu_clamp_shexp", cl_exp);
    // STRATA_X_MTP_TRUNK_HEAD=1: the trunk's embedding and head instead of the head's own (an A/B; saves its head's VRAM)
    static const bool trunk_head = [] { const char* e = std::getenv("STRATA_X_MTP_TRUNK_HEAD"); return e && *e == '1'; }();
    Stage& lg = last_stage();
    const int di_last = (int) stages.size() - 1;

    // the dense tensors (all but the routed experts, the embedding and the unused output norm) into one allocation on the last GPU
    struct Item {
        const strata::TensorInfo* ti;
        size_t shard;
        uint64_t bytes, off;
    };
    std::vector<Item> items;
    uint64_t total = 0;
    for (size_t sh = 0; sh < d->gm->size(); ++sh)
        for (const auto& ti : d->gm->shard(sh).tensors()) {
            const std::string& nm = ti.name;
            if (nm.size() > 12 && nm.compare(nm.size() - 12, 12, "_exps.weight") == 0) continue;   // the arena's
            // the host reads the embedding from the file mapping; graph_mtp norms with shared_head_norm, not output_norm
            if (nm == "token_embd.weight" || nm == "output_norm.weight" || (trunk_head && nm == "output.weight")) continue;
            const uint64_t b = strata::tensor_payload_bytes(ti);
            if (!b || !d->gm->in_bounds(ti, sh)) return fail(err, "the MTP head: tensor " + nm + " of unknown size");
            items.push_back({&ti, sh, b, total});
            total += (b + 255) & ~uint64_t(255);
        }
    set_dev(lg.dev);
    if (cudaMalloc(&d->dense, total) != cudaSuccess) {
        cudaGetLastError();
        return fail(err, "the MTP head: " + std::to_string(total >> 20) + " MiB of dense weights on GPU " + std::to_string(lg.dev));
    }
    for (const auto& it : items) {
        uint8_t* dst = static_cast<uint8_t*>(d->dense) + it.off;
        if (cudaMemcpy(dst, d->gm->shard(it.shard).tensor_data(*it.ti), it.bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail(err, "the MTP head: upload of " + it.ti->name);
        DT x;
        x.d = dst;
        x.type = (int) it.ti->type;
        for (size_t k = 0; k < it.ti->shape.size() && k < 4; ++k) x.ne[k] = (int64_t) it.ti->shape[k];
        x.bytes = it.bytes;
        d->t[it.ti->name] = x;
    }
    auto get = [&](const std::string& nm, DT& x) -> bool {
        auto f = d->t.find(nm);
        if (f == d->t.end()) return fail(err, "the MTP head: missing tensor " + nm);
        x = f->second;
        return true;
    };
    auto f32_vec = [&](const DT& x, int64_t ne0, const std::string& nm) -> bool {
        if (x.type != 0 || x.ne[0] != ne0 || x.n_out() != 1) return fail(err, "the MTP head: " + nm + " is not F32 [" + std::to_string(ne0) + "]");
        return true;
    };
    const int64_t n = hp.n_embd, V = hp.n_vocab, E = hp.n_expert;
    auto bn = [&](const char* sfx) { return "blk." + std::to_string(il) + "." + sfx; };
    if (!get(bn("nextn.eh_proj.weight"), d->eh_proj) || !get(bn("nextn.enorm.weight"), d->enorm) ||
        !get(bn("nextn.hnorm.weight"), d->hnorm) || !get(bn("nextn.shared_head_norm.weight"), d->shead_norm) ||
        !get("output_hc_base.weight", d->hc_base) || !get("output_hc_scale.weight", d->hc_scale))
        return false;
    if (d->eh_proj.n_in() != 2 * n || d->eh_proj.n_out() != n || !kn::native_mmvq_supported(d->eh_proj.type))
        return fail(err, "the MTP head: nextn.eh_proj is not [2 n_embd, n_embd] of a GPU type");
    if (!f32_vec(d->enorm, n, "nextn.enorm") || !f32_vec(d->hnorm, n, "nextn.hnorm") ||
        !f32_vec(d->shead_norm, n, "nextn.shared_head_norm") || !f32_vec(d->hc_base, hp.hc, "output_hc_base") ||
        !f32_vec(d->hc_scale, 1, "output_hc_scale"))
        return false;
    // the head reduce's mixer [hc * n -> hc] as F32 on the last GPU (the trunk's head reduce runs an F32 GEMV)
    {
        size_t shd = 0;
        const strata::TensorInfo* ti = d->gm->find("output_hc_fn.weight", &shd);
        if (!ti || ti->shape.size() < 2 || (int64_t) ti->shape[0] != hp.hc * n || (int64_t) ti->shape[1] != hp.hc)
            return fail(err, "the MTP head: output_hc_fn is not [hc n_embd, hc]");
        std::vector<float> w((size_t) (hp.hc * n * hp.hc));
        if (!rows_to_f32(*ti, d->gm->shard(shd).tensor_data(*ti), hp.hc * n, hp.hc, w.data()))
            return fail(err, "the MTP head: output_hc_fn of a type with no dequantizer");
        d->hc_fn = lg.dalloc<float>(w.size(), err);
        if (!d->hc_fn || cudaMemcpy(d->hc_fn, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            return fail(err, "the MTP head: output_hc_fn's F32 copy");
    }
    // its embedding (the file mapping, read on the host like the trunk's) and its head, or the trunk's
    if (trunk_head) {
        d->output = output;
    } else {
        if (!get("output.weight", d->output)) return false;
        if (d->output.n_in() != n || d->output.n_out() != V || !kn::native_mmvq_supported(d->output.type))
            return fail(err, "the MTP head: output.weight is not [n_embd, n_vocab] of a GPU type");
        size_t shd = 0;
        const strata::TensorInfo* te = d->gm->find("token_embd.weight", &shd);
        if (!te || te->shape.size() < 2 || (int64_t) te->shape[0] != n || (int64_t) te->shape[1] != V)
            return fail(err, "the MTP head: token_embd is not [n_embd, n_vocab]");
        const auto* tr = ggml_get_type_traits((ggml_type) te->type);
        if (!tr || !tr->to_float) return fail(err, "the MTP head: token_embd of a type with no dequantizer");
        d->embd = d->gm->shard(shd).tensor_data(*te);
        d->embd_type = (int) te->type;
        d->embd_row = ggml_row_size((ggml_type) te->type, n);
    }

    // the block: L[n_layer] on the last GPU (ratio 0, its clamps)
    L.resize((size_t) il + 1);
    ldev.resize((size_t) il + 1, di_last);
    hp.cratio.resize((size_t) il + 1, 0);
    hp.cratio[(size_t) il] = 0;
    hp.head_kv.resize((size_t) il + 1, 1);
    hp.clamp_exp.resize((size_t) il + 1, cl_exp);
    hp.clamp_sh.resize((size_t) il + 1, cl_sh);
    hp.clamp_exp[(size_t) il] = cl_exp;
    hp.clamp_sh[(size_t) il] = cl_sh;
    Layer& l = L[(size_t) il];
    bool ok = get(bn("attn_norm.weight"), l.attn_norm) && get(bn("ffn_norm.weight"), l.ffn_norm) &&
              get(bn("hc_attn_fn.weight"), l.hc_attn_fn) && get(bn("hc_attn_base.weight"), l.hc_attn_base) &&
              get(bn("hc_attn_scale.weight"), l.hc_attn_scale) && get(bn("hc_ffn_fn.weight"), l.hc_ffn_fn) &&
              get(bn("hc_ffn_base.weight"), l.hc_ffn_base) && get(bn("hc_ffn_scale.weight"), l.hc_ffn_scale) &&
              get(bn("attn_q_a.weight"), l.q_a) && get(bn("attn_q_a_norm.weight"), l.q_a_norm) && get(bn("attn_q_b.weight"), l.q_b) &&
              get(bn("attn_kv.weight"), l.wkv) && get(bn("attn_kv_a_norm.weight"), l.kv_a_norm) && get(bn("attn_sinks.weight"), l.sinks) &&
              get(bn("attn_output_a.weight"), l.wo_a) && get(bn("attn_output_b.weight"), l.wo_b) &&
              get(bn("ffn_gate_inp.weight"), l.gate_inp) && get(bn("exp_probs_b.bias"), l.probs_b) &&
              get(bn("ffn_gate_shexp.weight"), l.sh_gate) && get(bn("ffn_up_shexp.weight"), l.sh_up) &&
              get(bn("ffn_down_shexp.weight"), l.sh_down);
    if (!ok) return false;
    for (const DT* x : {&l.q_a, &l.q_b, &l.wkv, &l.wo_a, &l.wo_b, &l.sh_gate, &l.sh_up, &l.sh_down})
        if (!kn::native_mmvq_supported(x->type)) return fail(err, "the MTP head: a block projection of a type with no GPU kernel");
    // the mixers: F32 or a type the GPU kernels take (mhc_half runs both: the BF16 file's F32, the MXFP4 file's Q8_0)
    for (const DT* x : {&l.hc_attn_fn, &l.hc_ffn_fn})
        if ((x->type != 0 && !kn::native_mmvq_supported(x->type)) || x->n_in() != hp.hc * n || x->n_out() != 24)
            return fail(err, "the MTP head: its mixers are not [hc n_embd, 24] of a GPU type");
    // the grouped output as the trunk's (attn_output_a [n_head hd / groups, groups lora])
    if (l.q_a.n_in() != n || l.q_a.n_out() != hp.q_lora || l.q_b.n_in() != hp.q_lora || l.q_b.n_out() != hp.n_head * hp.hd_k ||
        l.wkv.n_in() != n || l.wkv.n_out() != hp.hd_k || l.wo_a.n_in() != hp.n_head * hp.hd_k / hp.o_groups ||
        l.wo_a.n_out() != hp.o_groups * hp.o_lora || l.wo_b.n_in() != hp.o_groups * hp.o_lora || l.wo_b.n_out() != n)
        return fail(err, "the MTP head: its attention's shapes differ from the trunk's");
    if (!f32_vec(l.sinks, hp.n_head, bn("attn_sinks")) || !f32_vec(l.probs_b, E, bn("exp_probs_b")) ||
        !f32_vec(l.q_a_norm, hp.q_lora, bn("attn_q_a_norm")) || !f32_vec(l.kv_a_norm, hp.hd_k, bn("attn_kv_a_norm")) ||
        !f32_vec(l.attn_norm, n, bn("attn_norm")) || !f32_vec(l.ffn_norm, n, bn("ffn_norm")))
        return false;
    // the router: BF16 -> an F32 copy (exact), as the trunk's (dsv4.cpp)
    if (l.gate_inp.ne[0] != n || l.gate_inp.n_out() != E) return fail(err, "the MTP head: " + bn("ffn_gate_inp.weight") + " shape");
    if (l.gate_inp.type != 0) {
        size_t shd = 0;
        const strata::TensorInfo* ti = d->gm->find(bn("ffn_gate_inp.weight"), &shd);
        std::vector<float> w((size_t) (n * E));
        if (!ti || !rows_to_f32(*ti, d->gm->shard(shd).tensor_data(*ti), n, E, w.data()))
            return fail(err, "the MTP head: its router of a type with no dequantizer");
        float* dw = lg.dalloc<float>(w.size(), err);
        if (!dw || cudaMemcpy(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            return fail(err, "the MTP head: the router's F32 copy");
        l.gate_inp.d = dw;
        l.gate_inp.type = 0;
        l.gate_inp.bytes = w.size() * sizeof(float);
    }
    // the routed experts' native format, and its K = V ring (the trunk's raw-ring geometry: the window plus the longest batch)
    {
        size_t sg = 0, sd = 0;
        const strata::TensorInfo* g = d->gm->find(bn("ffn_gate_exps.weight"), &sg);
        const strata::TensorInfo* dn = d->gm->find(bn("ffn_down_exps.weight"), &sd);
        std::string ferr;
        if (!g || !dn || !kc::native_fmt((int) g->type, (int) dn->type, n, hp.n_ff_exp, l.fmt, ferr))
            return fail(err, "the MTP head's experts: " + ferr);
        l.fmt.swiglu_limit = cl_exp;
        d->blob = (l.fmt.bytes + 4095) & ~(uint64_t) 4095;   // the pinned arena's stride (STRATA_X_MTP_ARENA=1)
    }
    l.raw_rows = dsv4_raw_rows();
    if (!(l.raw = lg.dalloc<uint16_t>((size_t) (l.raw_rows * hp.hd_k), err))) return false;
    cudaMemset(l.raw, 0, (size_t) (l.raw_rows * hp.hd_k) * sizeof(uint16_t));

    // the routed experts.  By default IN PLACE in the file mapping, as llama.cpp's mmap: the page cache is reclaimable, so the
    // resident RAM mode still sizes the trunk's lend region as without the head (a 3.2 GiB pinned copy took 450 lend-region
    // experts and 6% of the prefill, 2026-10-08); the pages are read in at load.  STRATA_X_MTP_ARENA=1: the pinned copy.
    {
        const auto t0 = clk::now();
        const char* names[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        const strata::TensorInfo* tis[3] = {nullptr, nullptr, nullptr};
        size_t shs[3] = {0, 0, 0};
        uint64_t slices[3] = {0, 0, 0};
        for (int part = 0; part < 3; ++part) {
            if (!(tis[part] = d->gm->find(bn(names[part]), &shs[part]))) return fail(err, "the MTP head: missing " + bn(names[part]));
            // each expert's slice must be exactly its part of the blob (a type the reader cannot size reads as 0 bytes)
            const uint64_t pay = strata::tensor_payload_bytes(*tis[part]);
            const uint64_t want = part < 2 ? (uint64_t) hp.n_ff_exp * l.fmt.gu_row : (uint64_t) n * l.fmt.d_row;
            if (pay == 0 || pay % (uint64_t) E || pay / (uint64_t) E != want || !d->gm->in_bounds(*tis[part], shs[part]))
                return fail(err, "the MTP head: " + bn(names[part]) + " is not " + std::to_string(E) + " experts of " + std::to_string(want) +
                                     " bytes (read " + std::to_string(pay) + ")");
            slices[part] = want;
        }
        const bool pinned = [] { const char* e = std::getenv("STRATA_X_MTP_ARENA"); return e && *e == '1'; }();
        // STRATA_X_MTP_VRAM=1: Strata's drafter layout - every expert resident on the last GPU, the block's MoE grouped there
        const bool in_vram = [] { const char* e = std::getenv("STRATA_X_MTP_VRAM"); return e && *e == '1'; }();
        uint64_t bytes = 0;
        if (in_vram) {
            if (!kn::native_mmvq_grouped_supported(l.fmt.gu_type, (int) n) || !kn::native_mmvq_grouped_supported(l.fmt.d_type, (int) hp.n_ff_exp))
                return fail(err, "the MTP head: no grouped GPU kernels for its experts' type");
            bytes = (uint64_t) E * d->blob;
            set_dev(lg.dev);
            if (cudaMalloc(&d->vram, bytes) != cudaSuccess) {
                cudaGetLastError();
                return fail(err, "the MTP head: " + std::to_string(bytes >> 20) + " MiB of VRAM for its experts on GPU " + std::to_string(lg.dev));
            }
            const size_t offs[3] = {0, l.fmt.up_off, l.fmt.down_off};
            for (int64_t e = 0; e < E; ++e)
                for (int part = 0; part < 3; ++part)
                    if (cudaMemcpy(d->vram + (uint64_t) e * d->blob + offs[part],
                                   d->gm->shard(shs[part]).tensor_data(*tis[part]) + (uint64_t) e * slices[part], slices[part],
                                   cudaMemcpyHostToDevice) != cudaSuccess)
                        return fail(err, "the MTP head: its experts' upload");
        } else if (pinned) {
            const uint64_t arena_bytes = (uint64_t) E * d->blob;
            d->arena = std::make_unique<strata::core::PinnedArena>(arena_bytes);
            if (!d->arena->valid()) return fail(err, "the MTP head: " + std::to_string(arena_bytes >> 20) + " MiB of RAM for its experts");
            const size_t offs[3] = {0, l.fmt.up_off, l.fmt.down_off};
            std::vector<std::thread> th;
            std::atomic<int64_t> next{0};
            for (int w = 0; w < 8; ++w)
                th.emplace_back([&] {
                    for (int64_t e; (e = next.fetch_add(1)) < E;) {
                        uint8_t* dst = d->arena->data() + (uint64_t) e * d->blob;
                        for (int part = 0; part < 3; ++part)
                            std::memcpy(dst + offs[part], d->gm->shard(shs[part]).tensor_data(*tis[part]) + (uint64_t) e * slices[part],
                                        slices[part]);
                    }
                });
            for (auto& t : th) t.join();
            // their pages leave the page cache (the arena holds them now); the embedding stays mapped
            for (int part = 0; part < 3; ++part) {
                const strata::GgufFile& f = d->gm->shard(shs[part]);
                if (const int fd = open(f.path().c_str(), O_RDONLY); fd >= 0) {
                    posix_fadvise(fd, (off_t) (f.data_start() + tis[part]->offset), (off_t) strata::tensor_payload_bytes(*tis[part]),
                                  POSIX_FADV_DONTNEED);
                    close(fd);
                }
            }
            d->src.g = d->arena->data();
            d->src.gs = d->blob;
            bytes = arena_bytes;
        } else {
            // read every page in now (a few threads, one byte a page), so no draft step waits on the disk
            std::vector<std::thread> th;
            std::atomic<int64_t> next{0};
            std::atomic<uint64_t> sink{0};
            const int64_t chunk = 64ll << 20;
            int64_t n_chunks[3], total_chunks = 0;
            for (int part = 0; part < 3; ++part) {
                const uint64_t pay = strata::tensor_payload_bytes(*tis[part]);
                n_chunks[part] = (int64_t) ((pay + chunk - 1) / chunk);
                total_chunks += n_chunks[part];
                const uint8_t* base = d->gm->shard(shs[part]).tensor_data(*tis[part]);
                const uintptr_t lo = (uintptr_t) base & ~(uintptr_t) 4095;
                madvise((void*) lo, (size_t) ((uintptr_t) base + pay - lo), MADV_WILLNEED);
                bytes += pay;
            }
            for (int w = 0; w < 8; ++w)
                th.emplace_back([&] {
                    uint64_t acc = 0;
                    for (int64_t c; (c = next.fetch_add(1)) < total_chunks;) {
                        int part = 0;
                        int64_t k = c;
                        while (k >= n_chunks[part]) k -= n_chunks[part++];
                        const uint64_t pay = strata::tensor_payload_bytes(*tis[part]);
                        const volatile uint8_t* base = d->gm->shard(shs[part]).tensor_data(*tis[part]);
                        const uint64_t end = std::min<uint64_t>(pay, (uint64_t) (k + 1) * chunk);
                        for (uint64_t o = (uint64_t) k * chunk; o < end; o += 4096) acc += base[o];
                    }
                    sink += acc;
                });
            for (auto& t : th) t.join();
            d->src.g = d->gm->shard(shs[0]).tensor_data(*tis[0]);
            d->src.u = d->gm->shard(shs[1]).tensor_data(*tis[1]);
            d->src.d = d->gm->shard(shs[2]).tensor_data(*tis[2]);
            d->src.gs = slices[0];
            d->src.us = slices[1];
            d->src.ds = slices[2];
        }
        if (opt.verbose)
            fprintf(stderr, "strata-x: MTP head %s: one block, %lld experts (%.2f GiB %s) in %.1f s; dense %.2f GiB on GPU %d; %s "
                            "embedding and head\n",
                    opt.draft_mtp.c_str(), (long long) E, bytes / 1073741824.0,
                    in_vram ? "resident on the last GPU" : pinned ? ("pinned: " + d->arena->note).c_str() : "in place in the file mapping, read in",
                    ms_since(t0) / 1000.0,
                    total / 1073741824.0, lg.dev, trunk_head ? "the trunk's" : "its own");
    }
    cur_dev = -1;
    dmtp = std::move(d);
    return true;
}

// ------------------------------------------------------------------ the block

bool Stage::mtp_embd_rows(const int32_t* toks, int T, float* dst, int64_t pos0) const {
    const Engine::Impl::Dsv4Mtp& d = *m.dmtp;
    const int n = (int) hp.n_embd;
    const auto* tr = ggml_get_type_traits((ggml_type) d.embd_type);
    for (int t = 0; t < T; ++t) {
        if (toks[t] < 0 || toks[t] >= hp.n_vocab) return false;
        if (const float* r = pos0 >= 0 ? m.image_row(pos0 + t) : nullptr) {   // an image position: the encoder's row (embd_rows')
            std::memcpy(dst + (size_t) t * n, r, (size_t) n * sizeof(float));
            continue;
        }
        tr->to_float(d.embd + (size_t) toks[t] * d.embd_row, dst + (size_t) t * n, n);
    }
    return true;
}

/// deepseek4.cpp graph_mtp over Tm rows (the window scratch; T <= kTmax): mtp_e holds the embeddings of the tokens after the
/// rows, mtp_hin the trunk's (or the chain's) four streams per row.  The block's output streams at the last row go to
/// mtp_zlast (the chain's next h); with last_head its draft (draft_pick: argmax or the coupled sample).
bool Stage::dsv4_mtp_body(int Tm, int64_t pos0, bool last_head, std::string& err) {
    const Engine::Impl::Dsv4Mtp& d = *m.dmtp;
    const int n = (int) hp.n_embd, hc = (int) hp.hc, hcn = hc * n, il = (int) hp.n_layer;
    // every RMS norm of the block with the head's own eps (mhc_half and dsv4 read rms_eps()); and the block stays a causal text
    // block over an image span's rows (the trunk reads a span non-causally with exp_probs_b_vl: llama.cpp does not run its
    // drafter over embedding batches at all - here the block reads the encoder's rows causally, so drafts after a picture see them)
    struct Guard {
        Stage& g;
        bool nc, ib;
        Guard(Stage& g_, float e) : g(g_), nc(g_.attn_noncausal), ib(g_.image_batch) {
            g.eps_ovr = e;
            g.attn_noncausal = g.image_batch = false;
        }
        ~Guard() {
            g.eps_ovr = 0.0f;
            g.attn_noncausal = nc;
            g.image_batch = ib;
        }
    } guard{*this, d.eps};
    try {
        rms_norm(mtp_e, d.enorm.f(), mtp_e, Tm, n, d.eps, s);
        rms_norm(mtp_hin, d.hnorm.f(), mtp_h, Tm * hc, n, d.eps, s);   // per stream
        // stream by stream: [enorm(e) | hnorm(h_s)] -> eh_proj -> the block's stream s (llama.cpp concatenates e repeated over
        // the streams with h and projects every (row, stream) column; MMVQ takes at most kTmax columns a launch)
        cudaMemcpy2DAsync(mtp_cat, (size_t) 2 * n * sizeof(float), mtp_e, (size_t) n * sizeof(float), (size_t) n * sizeof(float),
                          (size_t) Tm, cudaMemcpyDeviceToDevice, s);
        for (int st = 0; st < hc; ++st) {
            cudaMemcpy2DAsync(mtp_cat + n, (size_t) 2 * n * sizeof(float), mtp_h + (size_t) st * n, (size_t) hcn * sizeof(float),
                              (size_t) n * sizeof(float), (size_t) Tm, cudaMemcpyDeviceToDevice, s);
            quant(mtp_cat, xq, 2 * n, Tm);
            pj(d.eh_proj, mtp_cat, xq, mtp_z, Tm);
            cudaMemcpy2DAsync(R + (size_t) st * n, (size_t) hcn * sizeof(float), mtp_z, (size_t) n * sizeof(float),
                              (size_t) n * sizeof(float), (size_t) Tm, cudaMemcpyDeviceToDevice, s);
        }
        // one DeepSeek-V4 block over the streams: the trunk's mHC, attention (ratio 0, its ring) and the MoE from its arena
        mix_in(il, Tm, R, cur, coef, true);
        attention(il, pos0, Tm, cur, out, false);
        mix_out(il, Tm, out, R, coef, true);
        mix_in(il, Tm, R, cur, coef, false);
        if (d.vram) {   // every expert in VRAM (STRATA_X_MTP_VRAM=1): mtp_moe_resident's grouped launches, the slot = the expert id
            const Layer& l = L[(size_t) il];
            const kc::NativeFmt& f = l.fmt;
            const int E = (int) hp.n_expert, k = (int) hp.n_used, nf = (int) hp.n_ff_exp;
            gemv_f32_w(l.gate_inp.f(), cur, router, n, E, Tm, s);
            route(il, Tm, nullptr);
            swiglu_ffn(il, Tm, l.sh_gate, l.sh_up, l.sh_down, hp.clamp_sh[(size_t) il], cur, shared);   // leaves cur's Q8_1 in xq
            kn::native_mmvq_grouped(f.gu_type, d.vram, (size_t) d.blob, 0, f.up_off, ids, xq, k, hit_g, (size_t) nf, hit_u, (size_t) nf,
                                    n, nf, Tm * k, s);
            swiglu_clamp(hit_g, hit_u, hit_g, Tm * k * nf, f.swiglu_limit, s);
            quant(hit_g, hit_xq, Tm * k * nf, 1);
            kn::native_mmvq_grouped(f.d_type, d.vram, (size_t) d.blob, f.down_off, 0, ids, hit_xq, 1, hit_parts, (size_t) n, nullptr, 0,
                                    nf, n, Tm * k, s);
            moe_combine_pairs(hit_parts, hit_parts, d_zero_mask, wts, shared, out, n, k, Tm, s);
        } else if (!arena_moe(il, Tm, d.src, err)) {
            return false;
        }
        mix_out(il, Tm, out, R, coef, false);
        const float* zl = R + (size_t) (Tm - 1) * hcn;
        cudaMemcpyAsync(mtp_zlast, zl, (size_t) hcn * sizeof(float), cudaMemcpyDeviceToDevice, s);   // the chain's next h
        if (last_head) {   // its head reduce, shared_head_norm and head on the last row
            rms_norm(zl, nullptr, flat, 1, hcn, d.eps, s);
            gemv_t(d.hc_fn, flat, mixes, hcn, hc, 1);
            dsv4_hc_head(zl, mixes, d.hc_scale.f(), d.hc_base.f(), hp.hc_eps, h, n, 1, s);
            rms_norm(h, d.shead_norm.f(), mtp_hnlast, 1, n, d.eps, s);
            quant(mtp_hnlast, xq, n, 1);
            mm_t(d.output, xq, mtp_logits, 1);
            draft_pick();
        }
    } catch (const std::exception& e) {
        return m.fail(err, e.what());
    }
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return m.fail(err, cudaGetErrorString(ce));
    return true;
}

/// The block over the rows of a prompt chunk the first decode window's drafter reads - those within its 128-position window of
/// the read's end (Impl::prompt_end) - through the decode path's step, kTmax rows at a time; the chunk's final streams are
/// the prompt scratch's (pr, active on entry and again on return).
bool Stage::prompt_mtp_dsv4(int T, int n_next, int64_t pos0, const int32_t* next_toks, std::string& err) {
    const int Tm = std::min(T, n_next);
    const int64_t from = std::max<int64_t>(pos0, m.prompt_end - hp.n_swa);
    if (from >= pos0 + Tm) return true;
    const float* Rp = R;
    const int64_t hw = mtp_hw();
    int32_t tok = 0;
    float prob = 0.0f;
    bool ok = true;
    for (int64_t p = from; ok && p < pos0 + Tm; p += kTmax) {
        const int tn = (int) std::min<int64_t>(kTmax, pos0 + Tm - p);
        const int64_t r = p - pos0;
        ok = mtp_run(0, tn, next_toks + r, Rp + (size_t) r * hw, p, false, tok, prob, err);
    }
    use(pr);
    return ok;
}

// ------------------------------------------------------------------ a drafter block's MoE from its own arena

bool Stage::arena_moe(int il, int T, const ExpertSrc& src, std::string& err) {
    const Layer& l = L[(size_t) il];
    const int n = (int) hp.n_embd, E = (int) hp.E_of(il), k = (int) hp.k_of(il);
    gemv_f32_w(l.gate_inp.f(), cur, router, n, E, T, s);
    route(il, T, nullptr);
    cudaMemcpyAsync(m.h_x, cur, (size_t) T * n * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(m.h_ids, ids, (size_t) T * k * sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    cudaEventRecord(ev, s);
    swiglu_ffn(il, T, l.sh_gate, l.sh_up, l.sh_down, hp.clamp_sh[(size_t) il], cur, shared);
    // STRATA_X_MTP_TIMING: where a drafter block's MoE goes (the wait for the router's ids, the pool)
    static const bool timing = std::getenv("STRATA_X_MTP_TIMING") != nullptr;
    static double t_wait = 0, t_pool = 0;
    static int64_t t_cnt = 0, t_jobs = 0;
    const auto tw = clk::now();
    if (cudaEventSynchronize(ev) != cudaSuccess) return m.fail(err, "a drafter block's router readback failed");
    if (timing) t_wait += ms_since(tw);
    struct Job {
        int e, nt;
        int t[kc::MAXT], j[kc::MAXT];
    };
    std::vector<Job> jobs_e;
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < k; ++j) {
            const int e = m.h_ids[t * k + j];
            if (e < 0 || e >= E) return m.fail(err, "a drafter block's router produced expert " + std::to_string(e));
            auto it = std::find_if(jobs_e.begin(), jobs_e.end(), [&](const Job& x) { return x.e == e && x.nt < kc::MAXT; });
            if (it == jobs_e.end()) {
                jobs_e.push_back(Job{e, 0, {}, {}});
                it = jobs_e.end() - 1;
            }
            it->t[it->nt] = t;
            it->j[it->nt] = j;
            it->nt++;
        }
    for (int t = 0; t < T; ++t)
        kc::native_quant_act(l.fmt, m.h_x + (size_t) t * n, m.act_q.data() + (size_t) t * kc::kNativeActBytes);
    std::vector<kc::ExpertJobMulti> jobs(jobs_e.size());
    for (size_t i = 0; i < jobs_e.size(); ++i) {
        kc::ExpertJobMulti& jb = jobs[i];
        const uint64_t e = (uint64_t) jobs_e[i].e;
        jb.blob = src.g + e * src.gs;   // the blob, or (in place) the gate rows
        jb.up = src.u ? src.u + e * src.us : nullptr;
        jb.down = src.d ? src.d + e * src.ds : nullptr;
        jb.nt = jobs_e[i].nt;
        for (int q = 0; q < jb.nt; ++q) {
            jb.nact[q] = m.act_q.data() + (size_t) jobs_e[i].t[q] * kc::kNativeActBytes;
            jb.out[q] = m.h_parts + ((size_t) jobs_e[i].t[q] * k + jobs_e[i].j[q]) * n;
        }
    }
    const auto t1 = clk::now();
    {
        std::lock_guard<std::mutex> lk(m.pool_mu);   // a prompt's tail rows run while another stage may read on the pool
        m.pool->run_split_multi_native(l.fmt, jobs.data(), (int) jobs.size());
    }
    st.ms_pool += ms_since(t1);
    if (timing) {
        t_pool += ms_since(t1);
        t_jobs += (int64_t) jobs.size();
        if (++t_cnt % 100 == 0)
            fprintf(stderr, "strata-x: drafter MoE (%lld calls): router wait %.2f ms, pool %.2f ms, %.1f experts a call\n",
                    (long long) t_cnt, t_wait / t_cnt, t_pool / t_cnt, (double) t_jobs / t_cnt);
    }
    cudaMemcpyAsync(parts, m.h_parts, (size_t) T * k * n * sizeof(float), cudaMemcpyHostToDevice, s);
    moe_combine_w(parts, wts, shared, out, n, k, T, s);
    return true;
}

}  // namespace strata::x
