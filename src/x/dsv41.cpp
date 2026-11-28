// src/x/dsv41.cpp - moex: DeepSeek-V4.1-Flash (deepseek41, moex/docs/DSV41_PORT.md) on the DeepSeek-V4 path.  The file
// is a deepseek41 GGUF of the dense weights (moex/tools/dsv41/write_dense_gguf.py) and a pack directory of per-layer expert
// record files (EngineOptions::pack); the model runs as Arch::dsv4 with Hparams::v41 set, sharing V4's mHC, router, MoE,
// decode graphs and prompt path.  What V4.1 changes is here and in a few dispatches:
//   * the attention (Stage::dsv41): no per-head q norm; the window's K = V row rounded to FP8 before its ring; compression
//     ratios 2 and 1 instead of 4 and 128, computed only by the KV sources (layers 2, 8, 14, 20) and read by the layers after
//     each; the indexer's 32 heads on FP4-rounded q and keys, run only by the index sources and its top-512 reused by the
//     layers after each; layer 20's candidate blocks narrowing the later indexers' scores; o rounded to FP8 before wo_a;
//   * the hyper-connections' order (stage.cpp mhc_half): each half reads the streams with the PREVIOUS half's pre-mix,
//     carried per row in Scratch::cpre across the halves, the stages and the head (no output_hc_* reduce);
//   * the routed experts: records [gate | up | down] read in place from the pack's files (load_experts_records).
// The specification is the release's inference/model.py and kernel.py; moex/tools/dsv41/ref/ is the CPU reference the
// engine is compared against, op by op through the probes named here.
#include "engine_impl.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/gguf_split.hpp"
#include "strata/x/dsv41_kernels.hpp"

#include "ggml.h"

#include <cuda_fp16.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <cstdio>
#include <map>
#include <sys/sysmacros.h>
#include <mutex>

namespace strata::x {

/// moex (DSV41_BACKLOG #138, DSV41_PLAN §75): Stage::pj and heads_mm run a BF16 or FP8 matrix on the F32 input (gemv_dt) and
/// never read the quantized copy, so the quantize that feeds only such matrices is a launch for nothing - about 1.5 us each,
/// four a decode layer since the dense part went back to FP8 (§70).  Whether anything else reads the copy is the caller's.
static inline bool pj_needs_q(const strata::x::DT& w) { return !(w.type == GGML_TYPE_BF16 || w.type == kDtFP8); }


// ------------------------------------------------------------------ loading

/// deepseek41's keys: deepseek4's names (the geometry the kernels take is checked as there), V4.1's ratios 0 / 2 / 1, the KV and
/// index source lists, the candidate settings, the engram tables' settings and the expert records' layout.
bool Engine::Impl::load_hparams_dsv41(std::string& err) {
    const strata::GgufFile& m = gm->meta();
    const std::string pre = "deepseek41.";
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
    auto arr = [&](const char* k, const strata::MetaValue*& v) -> bool {
        v = get(k);
        if (!v || v->type != strata::MetaType::ARRAY) return fail(err, pre + k + " is missing or not an array");
        // the reader keeps a sample of a long array (64 items): a key read here must be whole
        if (v->items.size() != v->count) return fail(err, pre + k + ": " + std::to_string(v->count) + " items, only a sample read");
        return true;
    };
    auto ival = [](const strata::MetaValue& v) { return (int64_t) v.u; };   // signed payloads are two's complement in u
    Hparams& h = hp;
    h.arch = Arch::dsv4;
    h.v41 = true;
    int64_t idx_heads = 0, n_shared = 0, cand_src = -1, cand_blocks = 0, cand_bsize = 0;
    if (!u("embedding_length", h.n_embd) || !u("block_count", h.n_layer_all) || !u("attention.head_count", h.n_head) ||
        !u("attention.key_length", h.hd_k) || !u("attention.value_length", h.hd_v) || !u("rope.dimension_count", h.n_rot) ||
        !u("attention.q_lora_rank", h.q_lora) || !u("attention.sliding_window", h.n_swa) || !u("expert_count", h.n_expert) ||
        !u("expert_used_count", h.n_used) || !u("expert_feed_forward_length", h.n_ff_exp) ||
        !u("attention.indexer.head_count", idx_heads) || !u("attention.indexer.key_length", h.idx_dim) ||
        !u("attention.indexer.top_k", h.idx_topk) || !u("attention.output_group_count", h.o_groups) ||
        !u("attention.output_lora_rank", h.o_lora) || !u("hyper_connection.count", h.hc) ||
        !u("hyper_connection.sinkhorn_iterations", h.sinkhorn_iters) || !u("hash_layer_count", h.n_hash) ||
        !u("expert_shared_count", n_shared) ||
        // (moex: llama.cpp's converter - gguf_convert.py's set, unsloth's - keeps the candidate keys under attention.indexer)
        !u(get("attention.candidate_source_layer") ? "attention.candidate_source_layer" : "attention.indexer.candidate_source_layer", cand_src) ||
        !u(get("attention.candidate_top_blocks") ? "attention.candidate_top_blocks" : "attention.indexer.candidate_top_k_blocks", cand_blocks) ||
        !u(get("attention.candidate_block_size") ? "attention.candidate_block_size" : "attention.indexer.candidate_block_size", cand_bsize))
        return false;
    h.idx_heads = idx_heads;
    h.n_layer = h.n_layer_all;   // the trunk only: DSpark's blocks come in their own file (stage S7)
    h.n_ctx_orig = 0;
    if (get("rope.scaling.original_context_length")) u("rope.scaling.original_context_length", h.n_ctx_orig);
    float factor = 1.0f;
    if (!f("attention.layer_norm_rms_epsilon", h.rms_eps, 0, true) || !f("hyper_connection.epsilon", h.hc_eps, 0, true) ||
        !f("attention.compress_rope_freq_base", h.crope_base, 0, true) || !f("rope.freq_base", h.rope_base, 10000.0f, false) ||
        !f("rope.scaling.factor", factor, 1.0f, false) || !f("rope.scaling.yarn_beta_fast", h.beta_fast, 32.0f, false) ||
        !f("rope.scaling.yarn_beta_slow", h.beta_slow, 1.0f, false) || !f("expert_weights_scale", h.ew_scale, 1.0f, false))
        return false;
    h.rope_scale = factor > 0.0f ? 1.0f / factor : 1.0f;
    if (h.n_ctx_orig <= 0) {
        if (const strata::MetaValue* v = get("context_length")) h.n_ctx_orig = (int64_t) v->num();
    }
    if (const strata::MetaValue* v = get("rope.scaling.type"); !v || v->s != "yarn")
        return fail(err, "deepseek41: the compressed layers' RoPE is YaRN (rope.scaling.type yarn)");
    h.ew_norm = true;
    if (const strata::MetaValue* v = get("expert_weights_norm")) h.ew_norm = v->u != 0;
    if (!h.ew_norm) return fail(err, "deepseek41: unnormalized expert weights (the router kernel always normalizes)");
    if (const strata::MetaValue* v = get("expert_gating_func"); !v || v->u != 4)
        return fail(err, "deepseek41: expert gating is not sqrt(softplus) (4)");
    if (n_shared != 1) return fail(err, "deepseek41: one shared expert expected");
    if (h.n_hash != 0) return fail(err, "deepseek41: no hash-routed layers expected");
    if (const strata::MetaValue* v = m.get("tokenizer.ggml.eos_token_id")) h.eos_id = (int64_t) v->num();
    const strata::MetaValue* v = nullptr;
    if (!arr("attention.compress_ratios", v)) return false;
    if ((int64_t) v->items.size() < h.n_layer_all) return fail(err, "deepseek41.attention.compress_ratios is shorter than block_count");
    for (int64_t il = 0; il < h.n_layer_all; ++il) {
        const int r = (int) ival(v->items[(size_t) il]);
        if (r != 0 && r != 1 && r != 2) return fail(err, "deepseek41: compress ratio " + std::to_string(r) + " (0, 1 or 2)");
        h.cratio.push_back(r);
    }
    // the sources: layer il attends the compressed rows of the last KV source at or before it, and the top-k list of the last
    // index source at or before it (model.py's shared_attn); both must exist for a compressed layer and share its ratio
    auto sources = [&](const char* k, std::vector<int>& per_layer) -> bool {
        const strata::MetaValue* a = nullptr;
        if (!arr(k, a)) return false;
        std::vector<int> list;
        for (const auto& it : a->items) list.push_back((int) ival(it));
        if (!std::is_sorted(list.begin(), list.end())) return fail(err, pre + k + " is not ascending");
        per_layer.assign((size_t) h.n_layer_all, -1);
        for (int64_t il = 0; il < h.n_layer_all; ++il) {
            if (h.cratio[(size_t) il] == 0) continue;
            for (int s : list)
                if (s <= il) per_layer[(size_t) il] = s;
            const int s = per_layer[(size_t) il];
            if (s < 0) return fail(err, "deepseek41: compressed layer " + std::to_string(il) + " has no " + k + " entry at or before it");
            if (h.cratio[(size_t) s] != h.cratio[(size_t) il])
                return fail(err, "deepseek41: layer " + std::to_string(il) + " and its source " + std::to_string(s) + " compress differently");
        }
        return true;
    };
    if (!sources("attention.kv_source_layers", h.kv_src) ||
        !sources(get("attention.index_source_layers") ? "attention.index_source_layers" : "attention.indexer.source_layers", h.idx_src))
        return false;
    for (int64_t il = 0; il < h.n_layer_all; ++il) {   // a top-k list indexes the rows of the cache the layer attends
        const int s = h.idx_src[(size_t) il];
        if (s >= 0 && h.kv_src[(size_t) s] != h.kv_src[(size_t) il])
            return fail(err, "deepseek41: layer " + std::to_string(il) + " reads a top-k list over another layer's compressed rows");
    }
    h.cand_src = (int) cand_src;
    h.cand_blocks = (int) cand_blocks;
    h.cand_bsize = (int) cand_bsize;
    if (h.cand_src >= 0) {
        if (h.cand_src >= h.n_layer_all || !h.idx_owner(h.cand_src) || h.cand_blocks <= 0 || h.cand_bsize <= 0)
            return fail(err, "deepseek41: the candidate source must be an index source, with positive top blocks and block size");
        for (int64_t il = h.cand_src + 1; il < h.n_layer_all; ++il)
            if (h.idx_owner(il) && h.kv_src[(size_t) il] != h.kv_src[(size_t) h.cand_src])
                return fail(err, "deepseek41: index source " + std::to_string(il) + " narrows by candidates over another cache");
    }
    if (!get("expert_record.type")) {   // moex: a GGUF set's own experts (load_experts_gguf) - no records named
        h.rec_type.assign((size_t) h.n_layer, -1);
        h.rec_dtype.assign((size_t) h.n_layer, -1);
        h.rec_stride.assign((size_t) h.n_layer, 0);
        h.rec_file.assign((size_t) h.n_layer, std::string());
    } else {   // the expert records (one ggml type and stride per layer; the files relative to the pack)
        const strata::MetaValue *ty = nullptr, *sd = nullptr, *fl = nullptr;
        if (!arr("expert_record.type", ty) || !arr("expert_record.stride", sd) || !arr("expert_record.file", fl)) return false;
        if ((int64_t) ty->items.size() < h.n_layer || (int64_t) sd->items.size() < h.n_layer || (int64_t) fl->items.size() < h.n_layer)
            return fail(err, "deepseek41.expert_record.* are shorter than block_count");
        for (int64_t il = 0; il < h.n_layer; ++il) {
            h.rec_type.push_back((int) ival(ty->items[(size_t) il]));
            h.rec_dtype.push_back(h.rec_type.back());
            h.rec_stride.push_back((uint64_t) ival(sd->items[(size_t) il]));
            h.rec_file.push_back(fl->items[(size_t) il].s);
        }
    }
    {   // the engram tables' settings (the projections ride in the GGUF; the tables in the pack)
        const strata::MetaValue* pr = get("engram.present");
        h.engram = pr && pr->u != 0;
        if (!pr && get("ple.layers")) {
            // moex (the GGUF set): llama.cpp's PLE keys (conversion/deepseek41.py) - the layers, n-gram size, heads per n-gram,
            // the multipliers ([layer][n-gram]), every hash head's bucket count and ABSOLUTE first row in the one concatenated
            // table (per_layer_token_embd: the layers' tables one after another), the pad id and the token map (a metadata
            // array).  Here: each layer's rows (its heads' counts summed) and its heads' offsets in its own table, as ours
            const strata::MetaValue *ids = nullptr, *mul = nullptr, *offs = nullptr, *sizes = nullptr;
            if (!arr("ple.layers", ids) || !arr("ple.layer_multipliers", mul) || !u("ple.ngram_size", h.eg_ngram) ||
                !u("ple.heads_per_ngram", h.eg_heads) || !u("embedding_length_per_layer_input", h.eg_dim))
                return false;
            sizes = get("ple.head_vocab_sizes");
            offs = get("ple.head_offsets");
            const strata::MetaValue* tmv = get("engram.token_map");
            if (!sizes || !offs || !tmv) return fail(err, "deepseek41.ple.*: head_vocab_sizes, head_offsets or engram.token_map missing");
            const std::vector<int64_t> sz = m.ints(*sizes), of = m.ints(*offs), tm = m.ints(*tmv);
            for (const auto& it : ids->items) h.eg_layers.push_back((int) ival(it));
            for (const auto& it : mul->items) h.eg_mul.push_back(ival(it));
            const size_t Lg = h.eg_layers.size(), C = (size_t) ((h.eg_ngram - 1) * h.eg_heads);
            if (Lg == 0 || sz.size() != Lg * C || of.size() != Lg * C || h.eg_mul.size() != Lg * (size_t) h.eg_ngram || tm.empty())
                return fail(err, "deepseek41.ple.*: the head lists do not match " + std::to_string(Lg) + " layers of " + std::to_string(C) + " heads");
            int64_t base = 0;
            for (size_t j = 0; j < Lg; ++j) {
                int64_t rows = 0;
                for (size_t c = 0; c < C; ++c) rows += sz[j * C + c];
                for (size_t c = 0; c < C; ++c) {
                    const int64_t rel = of[j * C + c] - base;
                    if (rel < 0 || rel + sz[j * C + c] > rows) return fail(err, "deepseek41.ple.head_offsets: a head leaves its layer's table");
                    h.eg_primes.push_back(sz[j * C + c]);
                    h.eg_offsets.push_back(rel);
                }
                h.eg_rows.push_back(rows);
                base += rows;
            }
            h.eg_token_map.reserve(tm.size());
            int64_t top = 0;
            for (int64_t x : tm) {
                h.eg_token_map.push_back((int32_t) x);
                top = std::max(top, x);
            }
            h.eg_ncomp = top + 1;
            h.eg_pad = 0;
            if (const strata::MetaValue* p = get("engram.pad_token_id")) h.eg_pad = (int32_t) p->u;
            h.eg_wfile.assign(Lg, std::string());   // (the table is the set's per_layer_token_embd: load_engram)
            h.eg_sfile.assign(Lg, std::string());
            h.engram = true;
            if (const char* ne = std::getenv("STRATA_DSV41_NO_ENGRAM"); ne && ne[0] == '1') {
                h.engram = false;
                fprintf(stderr, "strata-x: STRATA_DSV41_NO_ENGRAM=1: the engram is OFF (not the real model)\n");
            }
        } else if (h.engram) {
            const strata::MetaValue *ids = nullptr, *rows = nullptr, *mul = nullptr, *primes = nullptr, *offs = nullptr,
                                    *wf = nullptr, *sf = nullptr;
            int64_t pad = 0;
            if (!arr("engram.layer_ids", ids) || !arr("engram.num_embeddings", rows) || !arr("engram.multipliers", mul) ||
                !arr("engram.primes", primes) || !arr("engram.offsets", offs) ||
                !arr("engram.weight_file", wf) || !arr("engram.scale_file", sf) || !u("engram.max_ngram_size", h.eg_ngram) ||
                !u("engram.head_count", h.eg_heads) || !u("engram.head_dim", h.eg_dim) ||
                !u("engram.compressed_vocab_size", h.eg_ncomp))
                return false;
            if (const strata::MetaValue* p = get("engram.pad_id")) pad = (int64_t) (int32_t) p->u;
            h.eg_pad = (int32_t) pad;
            for (const auto& it : ids->items) h.eg_layers.push_back((int) ival(it));
            for (const auto& it : rows->items) h.eg_rows.push_back(ival(it));
            for (const auto& it : mul->items) h.eg_mul.push_back(ival(it));
            for (const auto& it : primes->items) h.eg_primes.push_back(ival(it));
            for (const auto& it : offs->items) h.eg_offsets.push_back(ival(it));
            {   // the token map: the GGUF tensor engram.token_map (I32 [n_vocab]; too long for a metadata array)
                size_t sh2 = 0;
                const strata::TensorInfo* ti = gm->find("engram.token_map", &sh2);
                if (!ti || ti->type != GGML_TYPE_I32 || ti->shape.empty())
                    return fail(err, "deepseek41: no I32 tensor engram.token_map (the GGUF predates it: rewrite it with write_dense_gguf.py)");
                const int32_t* tm = reinterpret_cast<const int32_t*>(gm->shard(sh2).tensor_data(*ti));
                h.eg_token_map.assign(tm, tm + ti->shape[0]);
            }
            for (const auto& it : wf->items) h.eg_wfile.push_back(it.s);
            for (const auto& it : sf->items) h.eg_sfile.push_back(it.s);
            if (h.eg_rows.size() != h.eg_layers.size() || h.eg_wfile.size() != h.eg_layers.size() || h.eg_sfile.size() != h.eg_layers.size())
                return fail(err, "deepseek41.engram.*: one table per engram layer expected");
            // STRATA_DSV41_NO_ENGRAM=1: the engram-less model (the --no-engram reference's twin, for like-for-like checks)
            const char* ne = std::getenv("STRATA_DSV41_NO_ENGRAM");
            if (ne && ne[0] == '1') {
                h.engram = false;
                fprintf(stderr, "strata-x: STRATA_DSV41_NO_ENGRAM=1: the engram is OFF (not the real model)\n");
            }
        }
    }
    {
        const strata::MetaValue* clamp = get("swiglu_clamp_exp");
        if (!clamp) return fail(err, "missing " + pre + "swiglu_clamp_exp");
        const strata::MetaValue* cs = get("swiglu_clamp_shexp");
        // (moex: a scalar in ours, one per layer in llama.cpp's)
        auto per_layer = [&](const strata::MetaValue* v, std::vector<float>& dst) -> bool {
            if (v->type != strata::MetaType::ARRAY) {
                dst.assign((size_t) h.n_layer_all, (float) v->num());
                return true;
            }
            if ((int64_t) v->items.size() < h.n_layer_all || v->items.size() != v->count) return fail(err, pre + "swiglu_clamp_*: shorter than block_count");
            dst.clear();
            for (int64_t il = 0; il < h.n_layer_all; ++il) dst.push_back((float) v->items[(size_t) il].num());
            return true;
        };
        if (!per_layer(clamp, h.clamp_exp) || !per_layer(cs ? cs : clamp, h.clamp_sh)) return false;
    }
    size_t sh = 0;
    const strata::TensorInfo* te = gm->find("token_embd.weight", &sh);
    if (!te || te->shape.size() < 2) return fail(err, "no token_embd.weight");
    h.n_vocab = (int64_t) te->shape[1];
    h.head_kv.assign((size_t) h.n_layer_all, 1);   // one shared K = V head
    h.n_dense = 0;
    h.shexp = true;
    // the kernels' geometry: V4's (64 heads of 512 with RoPE on the last 64 dims, a 128-position window, top-512 of 128-wide
    // index rows, 8 output groups of 1024 rows, four streams); the indexer's heads run padded to V4's 64
    if (h.hc != 4) return fail(err, "deepseek41: mHC with hc != 4");
    if (h.n_head != 64 || h.hd_k != 512 || h.hd_v != 512 || h.n_rot != 64 || h.n_swa != 128 || h.idx_heads > 64 || h.idx_heads < 1 ||
        h.idx_dim != 128 || h.idx_topk != 512 || h.o_groups != 8 || h.o_lora != 1024 || h.n_head * h.hd_k % h.o_groups != 0)
        return fail(err, "deepseek41: kernel geometry - 64 heads of 512 (RoPE 64), window 128, indexer <= 64 x 128 top-512, 8 x 1024 output");
    return true;
}

namespace {
void bf16_to_f32(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t b = (uint32_t) src[i] << 16;
        std::memcpy(dst + i, &b, sizeof(float));
    }
}
}  // namespace

/// deepseek41's tensors (write_dense_gguf.py's names): V4's per layer minus the hash table, the compressor and the index keys'
/// projections on the KV sources (F32), the indexer's q and weights on the index sources (F32), the engram projections on
/// the engram layers; the head has no hyper-connection reduce.
/// moex (DSV41_BACKLOG #140, DSV41_PLAN §76; the ERNIE-5.5 hill-climb: shared): the BF16 head's two runs of rows (load_dense)
/// and their buffers - rows [0, head_lo) on the first GPU, the rest on the last, both computing at once (Stage::head)
bool Engine::Impl::bind_head_split(std::string& err) {
    auto lo = t.find("output.weight.lo"), hi = t.find("output.weight.hi");
    if (lo == t.end() || hi == t.end()) return fail(err, "missing tensor output.weight.lo / .hi");
    output_lo = lo->second;
    output_hi = hi->second;
    const int64_t n = hp.n_embd, V = hp.n_vocab, V0 = head_lo, V1 = V - V0;
    if (stages.size() != 2 || output_lo.type != GGML_TYPE_BF16 || output_hi.type != GGML_TYPE_BF16 || output_lo.ne[0] != n ||
        output_hi.ne[0] != n || output_lo.ne[1] != V0 || output_hi.ne[1] != V1 || V1 < 1)
        return fail(err, "output.weight: its two runs of rows do not make the head");
    output = DT{};   // (no GPU holds the whole head: Stage::head runs the two)
    output.type = GGML_TYPE_BF16;
    output.ne[0] = n;
    output.ne[1] = V;
    Stage &g0 = *stages[0], &g1 = *stages[1];
    set_dev(g0.dev);
    if (!(g0.hs_h = g0.dalloc<float>((size_t) kTmax * n, err)) || !(g0.hs_lo = g0.dalloc<float>((size_t) kTmax * V0, err)) ||
        cudaEventCreateWithFlags(&g0.hs_ev, cudaEventDisableTiming) != cudaSuccess)
        return fail(err, "the head's buffers on the first GPU: " + err);
    set_dev(g1.dev);
    if (!(g1.hs_hi = g1.dalloc<float>((size_t) kTmax * V1, err)) || cudaEventCreateWithFlags(&g1.hs_ev, cudaEventDisableTiming) != cudaSuccess)
        return fail(err, "the head's buffers on the last GPU: " + err);
    if (cudaHostAlloc((void**) &hs_x_h, (size_t) kTmax * n * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostAlloc((void**) &hs_lo_h, (size_t) kTmax * V0 * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &hs_x_d, hs_x_h, 0) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &hs_lo_d, hs_lo_h, 0) != cudaSuccess)
        return fail(err, "the head's mapped blocks");
    reset_dev_cache();
    if (opt.verbose)
        fprintf(stderr, "strata-x: the head over both GPUs (STRATA_DSV41_HEAD_SPLIT=0: whole on the last): rows 0-%lld on GPU %d, "
                        "%lld-%lld on GPU %d, each computing its own - the same logits\n",
                (long long) V0 - 1, g0.dev, (long long) V0, (long long) V - 1, g1.dev);
    return true;
}

bool Engine::Impl::bind_layers_dsv41(std::string& err) {
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
        if (d.type == GGML_TYPE_BF16 || d.type == kDtFP8) return true;   // moex (the GGUF set; §70): by Stage::gemv_dt (pj, heads_mm)
        if (!kn::native_mmvq_supported(d.type)) return fail(err, n + ": type " + std::to_string(d.type) + " has no GPU kernel");
        return true;
    };
    auto f32 = [&](const DT& d, const std::string& n, int64_t ne0, int64_t ne1) -> bool {
        if (d.type != 0 || d.ne[0] != ne0 || d.ne[1] * d.ne[2] * d.ne[3] != ne1)
            return fail(err, n + ": not F32 [" + std::to_string(ne0) + ", " + std::to_string(ne1) + "]");
        return true;
    };
    // moex (§62): the F32 weights STRATA_DSV41_BF16_W keeps as BF16 (load_dense) - either form, the same values
    auto f32w = [&](const DT& d, const std::string& n, int64_t ne0, int64_t ne1) -> bool {
        if (d.type == GGML_TYPE_BF16 && dsv41_bf16_w() && d.ne[0] == ne0 && d.ne[1] * d.ne[2] * d.ne[3] == ne1) return true;
        return f32(d, n, ne0, ne1);
    };
    auto shape = [&](const DT& d, const std::string& n, int64_t n_in, int64_t n_out) -> bool {
        if (d.n_in() != n_in || d.n_out() != n_out)
            return fail(err, n + ": shape [" + std::to_string(d.n_in()) + ", " + std::to_string(d.n_out()) + "], expected [" +
                                 std::to_string(n_in) + ", " + std::to_string(n_out) + "]");
        return true;
    };
    const int64_t n = hp.n_embd, H = hp.n_head, D = hp.hd_k, ID = hp.idx_dim, IH = hp.idx_heads, E = hp.n_expert;
    if (!get("output_norm.weight", output_norm, true)) return false;
    if (head_lo > 0) {   // moex (DSV41_BACKLOG #140, DSV41_PLAN §76): the head's two runs of rows (load_dense) and their buffers
        if (!bind_head_split(err)) return false;
    } else {
        if (!get("output.weight", output, true)) return false;
        if (!proj_ok(output, "output.weight")) return false;
    }
    if (!f32(output_norm, "output_norm.weight", n, 1)) return false;
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
                  get(nm("ffn_gate_inp.weight"), l.gate_inp, true) && get(nm("exp_probs_b.bias"), l.probs_b, true) &&
                  get(nm("exp_probs_b_vl.bias"), l.probs_b_vl, false) && get(nm("ffn_gate_shexp.weight"), l.sh_gate, true) &&
                  get(nm("ffn_up_shexp.weight"), l.sh_up, true) && get(nm("ffn_down_shexp.weight"), l.sh_down, true);
        if (!ok) return false;
        if (!proj_ok(l.hc_attn_fn, nm("hc_attn_fn")) || !proj_ok(l.hc_ffn_fn, nm("hc_ffn_fn")) ||
            !shape(l.hc_attn_fn, nm("hc_attn_fn"), hp.hc * n, 24) || !shape(l.hc_ffn_fn, nm("hc_ffn_fn"), hp.hc * n, 24) ||
            !f32(l.hc_attn_base, nm("hc_attn_base"), 24, 1) || !f32(l.hc_ffn_base, nm("hc_ffn_base"), 24, 1) ||
            !f32(l.hc_attn_scale, nm("hc_attn_scale"), 3, 1) || !f32(l.hc_ffn_scale, nm("hc_ffn_scale"), 3, 1) ||
            !f32(l.attn_norm, nm("attn_norm"), n, 1) || !f32(l.ffn_norm, nm("ffn_norm"), n, 1))
            return false;
        for (const auto& [d, what] : {std::pair<const DT*, const char*>{&l.q_a, "attn_q_a"}, {&l.q_b, "attn_q_b"}, {&l.wkv, "attn_kv"},
                                      {&l.wo_a, "attn_output_a"}, {&l.wo_b, "attn_output_b"}, {&l.sh_gate, "ffn_gate_shexp"},
                                      {&l.sh_up, "ffn_up_shexp"}, {&l.sh_down, "ffn_down_shexp"}})
            if (!proj_ok(*d, nm(what))) return false;
        if (!shape(l.q_a, nm("attn_q_a"), n, hp.q_lora) || !shape(l.q_b, nm("attn_q_b"), hp.q_lora, H * D) ||
            !shape(l.wkv, nm("attn_kv"), n, D) || !shape(l.wo_a, nm("attn_output_a"), H * D / hp.o_groups, hp.o_groups * hp.o_lora) ||
            !shape(l.wo_b, nm("attn_output_b"), hp.o_groups * hp.o_lora, n) || !f32(l.q_a_norm, nm("attn_q_a_norm"), hp.q_lora, 1) ||
            !f32(l.kv_a_norm, nm("attn_kv_a_norm"), D, 1) || !f32(l.sinks, nm("attn_sinks"), H, 1) ||
            !shape(l.sh_gate, nm("ffn_gate_shexp"), n, hp.n_ff_exp) || !shape(l.sh_up, nm("ffn_up_shexp"), n, hp.n_ff_exp) ||
            !shape(l.sh_down, nm("ffn_down_shexp"), hp.n_ff_exp, n) || !f32(l.probs_b, nm("exp_probs_b.bias"), E, 1))
            return false;
        if (l.probs_b_vl && !f32(l.probs_b_vl, nm("exp_probs_b_vl.bias"), E, 1)) return false;
        if (r > 0 && hp.kv_owner(il)) {   // the compressor (ratio 2 pools with a gate, ratio 1 only projects) and the index keys
            ok = get(nm("attn_compressor_kv.weight"), l.c_kv, true) && get(nm("attn_compressor_norm.weight"), l.c_norm, true) &&
                 get(nm("indexer.attn_k.weight"), l.idx_k, true) && get(nm("indexer.k_norm.weight"), l.idx_k_norm, true) &&
                 (r == 1 || get(nm("attn_compressor_gate.weight"), l.c_gate, true));
            if (!ok || !f32w(l.c_kv, nm("attn_compressor_kv"), n, D) || !f32(l.c_norm, nm("attn_compressor_norm"), D, 1) ||
                !f32w(l.idx_k, nm("indexer.attn_k"), D, ID) || !f32(l.idx_k_norm, nm("indexer.k_norm"), ID, 1) ||
                (r == 2 && !f32w(l.c_gate, nm("attn_compressor_gate"), n, D)))
                return false;
        }
        if (r > 0 && hp.idx_owner(il)) {   // the indexer's q (from qr) and per-head weights (from x)
            ok = get(nm("indexer.attn_q_b.weight"), l.idx_q_b, true) && get(nm("indexer.proj.weight"), l.idx_proj, true);
            if (!ok || !f32w(l.idx_q_b, nm("indexer.attn_q_b"), hp.q_lora, IH * ID) || !f32w(l.idx_proj, nm("indexer.proj"), n, IH))
                return false;
        }
        if (hp.engram && std::find(hp.eg_layers.begin(), hp.eg_layers.end(), (int) il) != hp.eg_layers.end()) {
            // moex (the GGUF set): llama.cpp's names - engram_kv the projection, engram_gate the converter's q_weight * k_weight
            // (F32, exact: BF16 * BF16) - where ours keeps q and k apart (load_engram multiplies them, the same product)
            const bool folded = gm->find(nm("engram_gate.weight")) != nullptr;
            ok = folded ? get(nm("engram_kv.weight"), l.eg_wkv, true) && get(nm("engram_gate.weight"), l.eg_qw, true)
                        : get(nm("engram.wkv.weight"), l.eg_wkv, true) && get(nm("engram.q_weight"), l.eg_qw, true) &&
                              get(nm("engram.k_weight"), l.eg_kw, true);
            if (!ok || !proj_ok(l.eg_wkv, nm("engram.wkv")) || l.eg_qw.type != 0 || (!folded && l.eg_kw.type != 0))
                return fail(err, nm("engram.*") + ": missing, or of an unexpected type");
        }
        // the router: BF16 [n_embd, n_expert] -> an F32 copy on the layer's GPU (as V4's); moex (§62): with STRATA_DSV41_BF16_W the
        // GGUF's BF16 itself (Stage::gemv_dt reads it as that copy's values)
        if (l.gate_inp.ne[0] != n || l.gate_inp.n_out() != E) return fail(err, nm("ffn_gate_inp.weight") + ": shape");
        if (l.gate_inp.type == GGML_TYPE_BF16 && !dsv41_bf16_w()) {
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
        } else if (l.gate_inp.type != 0 && l.gate_inp.type != GGML_TYPE_BF16) {
            return fail(err, nm("ffn_gate_inp.weight") + ": neither BF16 nor F32");
        }
    }
    {   // (§62) each stage's widening scratch: its largest BF16 weight's floats (the prompt path's GEMMs) - the F32 weights
        // STRATA_DSV41_BF16_W keeps as BF16 and (moex, the GGUF set) BF16 projections and head, at most kWidenMax floats: a
        // larger matrix is widened and multiplied in row blocks (Stage::gemv_dt)
        constexpr size_t kWidenMax = (size_t) 48 << 20;   // 192 MB: q_b and wo_b whole (41.9M), the engram's wkv and the head in blocks
        std::vector<size_t> need(stages.size(), 0);
        for (int64_t il = 0; il < n_load; ++il) {
            const Layer& l = L[(size_t) il];
            for (const DT* d : {&l.gate_inp, &l.c_kv, &l.c_gate, &l.idx_k, &l.idx_q_b, &l.idx_proj, &l.q_a, &l.q_b, &l.wkv, &l.wo_a,
                                &l.wo_b, &l.sh_gate, &l.sh_up, &l.sh_down, &l.eg_wkv})
                if (d->d && (d->type == GGML_TYPE_BF16 || d->type == kDtFP8)) {
                    size_t& nd = need[(size_t) ldev[(size_t) il]];
                    nd = std::max(nd, std::min(kWidenMax, (size_t) (d->n_in() * d->n_out())));
                }
        }
        if (output.d && output.type == GGML_TYPE_BF16)
            need.back() = std::max(need.back(), std::min(kWidenMax, (size_t) (output.n_in() * output.n_out())));
        for (size_t si = 0; si < stages.size(); ++si)
            if (need[si]) {
                set_dev(stages[si]->dev);
                if (!(stages[si]->bf16_f32 = stages[si]->dalloc<float>(need[si], err)))
                    return fail(err, "the BF16 weights' widening scratch: " + err);
                stages[si]->bf16_f32_n = need[si];
            }
    }
    reset_dev_cache();
    rope_pc = dsv4_rope_pc(hp.crope_base, hp.rope_scale, 1.0f, (int) hp.n_ctx_orig, hp.beta_fast, hp.beta_slow);
    return true;
}

/// THE RECORD SOURCE: layer il's routed experts are records of rec_stride bytes in the pack's file rec_file[il], expert e's at
/// e * stride, each a NativeFmt blob [gate rows | up rows | down rows] (moex/tools/dsv41/convert_pack.py) read in place from
/// a read-only mapping - the file tier, as a GGUF's expert tensors are read.  The tiers take them through Impl::src (the
/// record's parts) and init_cache's load pass (one read role per layer, the blob copied once into its slot / RAM blob).
bool Engine::Impl::load_experts_records(std::string& err) {
    return load_record_layers(err) && load_experts_tail(err);
}

/// The pack's record files, every MoE layer (load_experts_records; load_experts_gguf first, for the layers its set lacks)
bool Engine::Impl::load_record_layers(std::string& err) {
    if (opt.pack.empty()) return fail(err, "deepseek41: the routed experts are in a pack directory (--pack DIR)");
    // a requantized pack (moex/tools/dsv41/requant_pack) names its records in experts.tsv: "layer type stride file" lines
    // override the GGUF's deepseek41.expert_record.* - the same dense file serves every expert pack.  moex: the type is "gu,d"
    // where the down matrices have their own (gate and up in gu, down in d); a plain "type" is both, as before
    if (FILE* f = fopen((opt.pack + "/experts.tsv").c_str(), "r")) {
        char line[1024];
        std::vector<int> seen((size_t) hp.n_layer, 0);   // a layer named twice: its last line counts (requant_pack appends)
        std::string twice;
        while (fgets(line, sizeof line, f)) {
            int il = -1, ty = -1, dty = -1;
            unsigned long long stride = 0;
            char tys[32], rel[768];
            if (line[0] == '#' || sscanf(line, "%d\t%31s\t%llu\t%767s", &il, tys, &stride, rel) != 4) continue;
            const int nty = sscanf(tys, "%d,%d", &ty, &dty);
            if (nty < 1 || ty < 0 || (nty == 2 && dty < 0)) {
                fclose(f);
                return fail(err, opt.pack + "/experts.tsv: layer " + std::to_string(il) + ": type '" + tys + "' is neither T nor GU,D");
            }
            if (il < 0 || il >= hp.n_layer) {
                fclose(f);
                return fail(err, opt.pack + "/experts.tsv: layer " + std::to_string(il) + " out of range");
            }
            if (seen[(size_t) il]++ == 1) twice += (twice.empty() ? "" : " ") + std::to_string(il);
            hp.rec_type[(size_t) il] = ty;
            hp.rec_dtype[(size_t) il] = nty == 2 ? dty : ty;
            hp.rec_stride[(size_t) il] = (uint64_t) stride;
            hp.rec_file[(size_t) il] = rel;
        }
        fclose(f);
        int n_over = 0;
        for (int c : seen) n_over += c > 0;
        if (opt.verbose) {
            fprintf(stderr, "strata-x: %s/experts.tsv: records of %d layers", opt.pack.c_str(), n_over);
            if (!twice.empty()) fprintf(stderr, " (layers %s named more than once: the last line of each counts)", twice.c_str());
            fprintf(stderr, "\n");
        }
    }
    const int64_t NE = hp.n_expert;
    for (int64_t il = hp.n_dense; il < n_xl; ++il) {
        if (!xg_has.empty() && xg_has[(size_t) il]) continue;   // moex: its experts come from the GGUF set (load_experts_gguf)
        Layer& l = L[(size_t) il];
        const std::string path = opt.pack + "/" + hp.rec_file[(size_t) il];
        if (!kc::native_fmt(hp.rec_type[(size_t) il], hp.rec_dtype[(size_t) il], hp.n_embd, hp.n_ff_exp, l.fmt, err))
            return fail(err, path + ": " + err);
        l.fmt.swiglu_limit = hp.clamp_exp[(size_t) il];
        const uint64_t stride = hp.rec_stride[(size_t) il];
        if (stride < l.fmt.bytes) return fail(err, path + ": record stride " + std::to_string(stride) + " below the blob's " +
                                                   std::to_string(l.fmt.bytes) + " bytes");
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return fail(err, "cannot open " + path);
        // exactly NE records: a stride that is not the file's (a stale experts.tsv line, a pack requantized again) used to pass
        // whenever the file was at least that long, and every expert past the first was read from the wrong place
        struct stat sb {};
        if (fstat(fd, &sb) != 0 || (uint64_t) sb.st_size != (uint64_t) NE * stride) {
            close(fd);
            return fail(err, path + ": " + std::to_string((unsigned long long) sb.st_size) + " bytes, not " + std::to_string(NE) +
                                 " records of " + std::to_string(stride) + " bytes (experts.tsv and the file disagree)");
        }
        void* p = mmap(nullptr, (size_t) sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (p == MAP_FAILED) return fail(err, "cannot map " + path);
        rec_maps.push_back({p, (size_t) sb.st_size});
        const uint8_t* base = static_cast<const uint8_t*>(p);
        l.eg = base;
        l.eu = base + l.fmt.up_off;
        l.ed = base + l.fmt.down_off;
        l.sg = l.su = l.fmt.gu_row * (size_t) hp.n_ff_exp;
        l.sd = l.fmt.d_row * (size_t) hp.n_embd;
        l.rec_stride = stride;
        l.blob_contiguous = true;
        blob_max = std::max(blob_max, l.fmt.bytes);
        if (mmq::built() && mmq::supported(l.fmt.gu_type) && mmq::supported(l.fmt.d_type)) {
            gub_max = std::max(gub_max, mmq::matrix_bytes(l.fmt.gu_type, 2 * hp.n_ff_exp, hp.n_embd));
            db_max = std::max(db_max, mmq::matrix_bytes(l.fmt.d_type, hp.n_embd, hp.n_ff_exp));
        } else {
            mmq_all = false;
        }
    }
    return true;
}

/// moex (the GGUF set, --experts-gguf; gguf_convert.py writes one from the release on disk, unsloth's are others): the
/// routed experts from a GGUF set as virtual records.  A layer's gate / up / down tensors ([n_embd, n_ff, NE] and [n_ff,
/// n_embd, NE]) hold expert e's slices at e * slice; rec_io reads the three into a slot - the gate at 0, where every
/// kernel reads it, up and down at 4 KiB-aligned places with room around them for the sectors a slice shares with its
/// neighbours (O_DIRECT reads whole 512-byte sectors; a GGUF aligns its tensors to 32 bytes).  Each slice must be whole
/// sectors (then an expert's slices all start at the same place in a sector), and the drive must take 32-byte aligned
/// buffers (its dma_alignment).  With --pack as well, a layer whose tensors are in a file not written yet keeps the pack's
/// record (a set written file by file is checked layer by layer); without, every layer must be in the set.  The tiers
/// work on a virtual record as on a record (fmt's places, rec_stride a slot).
bool Engine::Impl::load_experts_gguf(std::string& err) {
    const std::vector<std::string> paths = strata::gguf_split_paths(opt.experts_gguf, false);
    xg.clear();
    std::map<std::string, std::pair<size_t, const strata::TensorInfo*>> idx;
    std::string first;
    for (size_t i = 0; i < paths.size(); ++i) {
        std::unique_ptr<strata::GgufFile> g;
        struct stat sb {};
        if (::stat(paths[i].c_str(), &sb) == 0) {
            try {
                g = std::make_unique<strata::GgufFile>(paths[i]);
            } catch (const std::exception& ex) {
                return fail(err, std::string("--experts-gguf: ") + ex.what());
            }
            for (const auto& t : g->tensors()) idx[t.name] = {i, &t};
            if (first.empty()) first = paths[i];
        }
        xg.push_back(std::move(g));
    }
    if (first.empty()) return fail(err, "--experts-gguf " + opt.experts_gguf + ": no file of the set is there");
    // the layers the set holds; the pack's records (--pack) only for the others - so the pack's files of the layers the set
    // holds need not be there any more
    xg_has.assign((size_t) n_xl, 0);
    for (int64_t il = hp.n_dense; il < n_xl; ++il) {
        auto nm = [&](const char* r) { return "blk." + std::to_string(il) + ".ffn_" + r + "_exps.weight"; };
        xg_has[(size_t) il] = idx.count(nm("gate")) && idx.count(nm("up")) && idx.count(nm("down"));
    }
    if (!opt.pack.empty() && !load_record_layers(err)) return false;
    {   // the drive's O_DIRECT limits: 512-byte sectors, buffers aligned to at most 32 bytes (the slices' places allow no more)
        struct stat sb {};
        ::stat(first.c_str(), &sb);
        const std::string dev = "/sys/dev/block/" + std::to_string(major(sb.st_dev)) + ":" + std::to_string(minor(sb.st_dev));
        auto rd = [&](const char* what) -> long {
            for (const std::string& d : {dev + "/queue/", dev + "/../queue/"})
                if (FILE* f = fopen((d + what).c_str(), "r")) {
                    long v = -1;
                    if (fscanf(f, "%ld", &v) != 1) v = -1;
                    fclose(f);
                    return v;
                }
            return -1;
        };
        const long lbs = rd("logical_block_size"), dma = rd("dma_alignment");
        if (lbs > 512 || dma > 31 || lbs < 0 || dma < 0)
            return fail(err, "--experts-gguf: the drive under " + first + " reads O_DIRECT in " + std::to_string(lbs) +
                             "-byte blocks into buffers aligned to " + std::to_string(dma + 1) +
                             " bytes; the slices' reads need 512 and at most 32 (not read: " + dev + ")");
    }
    const int64_t NE = hp.n_expert;
    int n_gguf = 0, n_pack = 0;
    for (int64_t il = hp.n_dense; il < n_xl; ++il) {
        Layer& l = L[(size_t) il];
        auto nm = [&](const char* r) { return "blk." + std::to_string(il) + ".ffn_" + r + "_exps.weight"; };
        const auto ig = idx.find(nm("gate")), iu = idx.find(nm("up")), id = idx.find(nm("down"));
        if (ig == idx.end() || iu == idx.end() || id == idx.end()) {
            if (opt.pack.empty()) return fail(err, "--experts-gguf: no " + nm("*") + " in the set's files there are (and no --pack)");
            ++n_pack;   // (its record from the pack: load_record_layers)
            continue;
        }
        const strata::TensorInfo *g = ig->second.second, *u = iu->second.second, *d = id->second.second;
        if (g->type != u->type) return fail(err, nm("gate") + ": gate and up of different types");
        if (g->shape.size() != 3 || u->shape != g->shape || d->shape.size() != 3 || (int64_t) g->shape[0] != hp.n_embd ||
            (int64_t) g->shape[1] != hp.n_ff_exp || (int64_t) g->shape[2] != NE || (int64_t) d->shape[0] != hp.n_ff_exp ||
            (int64_t) d->shape[1] != hp.n_embd || (int64_t) d->shape[2] != NE)
            return fail(err, nm("*") + ": not [n_embd, n_ff, n_expert] / [n_ff, n_embd, n_expert]");
        kc::NativeFmt f;
        if (!kc::native_fmt((int) g->type, (int) d->type, hp.n_embd, hp.n_ff_exp, f, err)) return fail(err, nm("gate") + ": " + err);
        f.swiglu_limit = hp.clamp_exp[(size_t) il];
        const uint64_t pg = f.gu_row * (uint64_t) hp.n_ff_exp, pd = f.d_row * (uint64_t) hp.n_embd;
        if (pg % 512 || pd % 512)
            return fail(err, nm("*") + ": an expert's slices are no whole 512-byte sectors (" + std::to_string(pg) + ", " +
                             std::to_string(pd) + " bytes): no O_DIRECT reads of them");
        const strata::TensorInfo* t3[3] = {g, u, d};
        const size_t sh3[3] = {ig->second.first, iu->second.first, id->second.first};
        uint64_t head = 0;
        for (int r = 0; r < 3; ++r) {
            const strata::GgufFile& gf = *xg[sh3[r]];
            l.f_shard[r] = sh3[r];
            l.f_off[r] = gf.data_start() + t3[r]->offset;
            if (l.f_off[r] + (r == 2 ? pd : pg) * (uint64_t) NE > gf.file_size()) return fail(err, gf.path() + ": " + t3[r]->name + " runs past its end");
            head |= l.f_off[r] % 512;
        }
        if (!head && pg % 4096 == 0 && pd % 4096 == 0) {   // every slice on a page: the pack's own layout [gate | up | down]
            f.up_off = pg;
            f.down_off = 2 * pg;
            f.bytes = 2 * pg + pd;
            l.rec_stride = (f.bytes + 4095) & ~(uint64_t) 4095;
        } else {   // a slice's sectors spill up to 511 bytes past it and up to 480 before it: 1 KiB of room on each side
            f.up_off = (pg + 1024 + 4095) & ~(uint64_t) 4095;
            f.down_off = (f.up_off + pg + 1024 + 4095) & ~(uint64_t) 4095;
            f.bytes = f.down_off + pd;
            l.rec_stride = (f.bytes + 512 + 4095) & ~(uint64_t) 4095;
        }
        l.fmt = f;
        l.eg = xg[sh3[0]]->tensor_data(*g);
        l.eu = xg[sh3[1]]->tensor_data(*u);
        l.ed = xg[sh3[2]]->tensor_data(*d);
        l.sg = l.su = (size_t) pg;
        l.sd = (size_t) pd;
        l.rec_gguf = true;
        l.blob_contiguous = false;
        hp.rec_type[(size_t) il] = (int) g->type;
        hp.rec_dtype[(size_t) il] = (int) d->type;
        blob_max = std::max(blob_max, l.fmt.bytes);
        if (mmq::built() && mmq::supported(l.fmt.gu_type) && mmq::supported(l.fmt.d_type)) {
            gub_max = std::max(gub_max, mmq::matrix_bytes(l.fmt.gu_type, 2 * hp.n_ff_exp, hp.n_embd));
            db_max = std::max(db_max, mmq::matrix_bytes(l.fmt.d_type, hp.n_embd, hp.n_ff_exp));
        } else {
            mmq_all = false;
        }
        ++n_gguf;
    }
    if (opt.verbose)
        fprintf(stderr, "strata-x: routed experts of %d layers from the GGUF set %s (%zu files, %zu there)%s\n", n_gguf,
                opt.experts_gguf.c_str(), paths.size(), (size_t) std::count_if(xg.begin(), xg.end(), [](const auto& p) { return p != nullptr; }),
                n_pack ? (", " + std::to_string(n_pack) + " layers' records from the pack " + opt.pack).c_str() : "");
    return load_experts_tail(err);
}

/// The record readers and the CPU pool (after the layers' records: load_record_layers, load_experts_gguf)
bool Engine::Impl::load_experts_tail(std::string& err) {
    if (!rec_io_init(err)) return false;
    // moex (DSV41_PLAN §29), opt-in STRATA_POOL_TASKS=N: each of the pool's row phases cut into N tasks, not 3 a thread (63 with
    // 20 workers and the host) - a decode layer's ~2 experts in 63 tasks left the phase's tail to a few threads (58 GB/s over
    // the drains against 72-73 for whole experts); every row is still one thread's, so no result changes
    const char* pt = std::getenv("STRATA_POOL_TASKS");
    const int pool_tasks = pt ? std::clamp(std::atoi(pt), 0, kc::ExpertPool::kMaxTasks) : 0;
    pool = std::make_unique<kc::ExpertPool>(opt.threads, opt.pin, true, kc::PoolAffinity::All, pool_tasks);
    if (opt.verbose)
        fprintf(stderr, "strata-x: routed experts from %lld record files in %s; CPU expert pool, %d workers + the host%s\n",
                (long long) (n_xl - hp.n_dense), opt.pack.c_str(), pool->workers(),
                pool_tasks > 0 ? (", " + std::to_string(pool_tasks) + " tasks a phase (STRATA_POOL_TASKS)").c_str() : "");
    act_q.resize((size_t) kTmax * kc::kNativeActBytes);
    return true;
}

/// /proc/meminfo's MemAvailable in bytes (0 when unreadable)
static uint64_t mem_available_bytes() {
    uint64_t kb = 0;
    if (FILE* f = fopen("/proc/meminfo", "r")) {
        char line[256];
        while (fgets(line, sizeof line, f))
            if (strncmp(line, "MemAvailable:", 13) == 0) {
                kb = strtoull(line + 13, nullptr, 10);
                break;
            }
        fclose(f);
    }
    return kb * 1024ull;
}

/// STRATA_DSV41_BOUNCE: stage si's bounce slots - N for every stage, or (moex, DSV41_BACKLOG #66) N0,N1,... stage by stage
/// (a stage the list leaves out keeps Stage::kRecBounce); unset: Stage::kRecBounce
static int bounce_slots(size_t si) {
    const char* e = std::getenv("STRATA_DSV41_BOUNCE");
    if (!e) return Stage::kRecBounce;
    if (!strchr(e, ',')) return std::atoi(e);
    for (size_t i = 0;; ++i) {
        if (i == si) return *e && *e != ',' ? std::atoi(e) : Stage::kRecBounce;
        const char* c = strchr(e, ',');
        if (!c) return Stage::kRecBounce;
        e = c + 1;
    }
}

/// moex (DSV41_PLAN §21): pinned host blocks of the given sizes (4 KiB multiples) into out; reg[k] = block k's bytes when it was
/// registered (freed with cudaHostUnregister and munmap), 0 when cudaHostAlloc'ed (cudaFreeHost).  STRATA_DSV41_FAST_PIN=1
/// (opt-in): anonymous memory, its pages faulted in by 8 threads a block ahead of the block's cudaHostRegister (portable,
/// mapped) - cudaHostAlloc holds the driver's lock for the whole allocation (another thread's 14.5 MiB copy waited 9.7 s
/// behind 16 GiB of it) and pinned 2.3 GiB/s on this box, this way 8.4 GiB/s (16 GiB: 6.9 -> 1.9 s).  A block it cannot map
/// or register is cudaHostAlloc'ed as before; false: a cudaHostAlloc failed
bool huge_pin() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_DSV41_HUGE_PIN");
        return e && e[0] == '1';
    }();
    return on;
}

/// moex (DSV41_BACKLOG #104), opt-in STRATA_DSV41_KV_PACK=1: the KV sources' compressed rows and index keys kept packed as
/// the model rounds them - E2M1 codes and their scales (dsv41_pack_row_bytes: 288 and 80 bytes a row against 1,024 and 256
/// as F16) - and read back as the same halves, so the attention and the selection give the same bits: 3,200 -> 920 bytes a
/// position, ~1.2 GB of VRAM at 524,288 positions for the expert slots
bool kv_pack() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_DSV41_KV_PACK");
        return e && e[0] == '1';
    }();
    return on;
}

/// moex (DSV41_BACKLOG #102): an anonymous mapping of `bytes` starting on a 2 MiB boundary (the padding's head and tail
/// unmapped again, so munmap(p, bytes) frees it whole), or nullptr
static void* map_2m(uint64_t bytes) {
    constexpr uint64_t kAlign = 2ull << 20;
    void* raw = mmap(nullptr, (size_t) (bytes + kAlign), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return nullptr;
    const uintptr_t start = (uintptr_t) raw, aligned = (start + kAlign - 1) & ~(uintptr_t) (kAlign - 1);
    if (aligned > start) munmap(raw, aligned - start);
    const uintptr_t end = aligned + bytes, raw_end = start + bytes + kAlign;
    if (raw_end > end) munmap((void*) end, raw_end - end);
    return (void*) aligned;
}

namespace {
std::mutex huge_mu;
std::map<uint8_t*, uint64_t> huge_live;   // alloc_pinned's huge-page buffers (munmap'ed by free_pinned), the rest cudaHostAlloc'ed
}  // namespace

uint8_t* alloc_pinned(uint64_t bytes) {
    if (huge_pin()) {
        if (void* m = map_2m(bytes)) {
            bool huge = true;
            madvise(m, (size_t) bytes, MADV_DONTFORK);
            strata::core::fault_in_paced(m, bytes, 8, &huge, nullptr);
            if (cudaHostRegister(m, (size_t) bytes, cudaHostRegisterPortable) == cudaSuccess) {
                std::lock_guard<std::mutex> lk(huge_mu);
                huge_live[static_cast<uint8_t*>(m)] = bytes;
                return static_cast<uint8_t*>(m);
            }
            (void) cudaGetLastError();
            munmap(m, (size_t) bytes);
        }
    }
    uint8_t* p = nullptr;
    return cudaHostAlloc((void**) &p, (size_t) bytes, cudaHostAllocPortable) == cudaSuccess ? p : nullptr;
}

void free_pinned(uint8_t* p) {
    if (!p) return;
    {
        std::lock_guard<std::mutex> lk(huge_mu);
        if (auto it = huge_live.find(p); it != huge_live.end()) {
            cudaHostUnregister(p);
            munmap(p, (size_t) it->second);
            huge_live.erase(it);
            return;
        }
    }
    cudaFreeHost(p);
}

static bool pin_blocks(const std::vector<uint64_t>& bytes, std::vector<uint8_t*>& out, std::vector<uint64_t>& reg,
                       std::thread* async = nullptr, std::atomic<uint64_t>* pending = nullptr,
                       std::atomic<size_t>* faulted_out = nullptr, int dev = 0) {
    const size_t nb = bytes.size();
    out.assign(nb, nullptr);
    reg.assign(nb, 0);
    if (const char* e = std::getenv("STRATA_DSV41_FAST_PIN"); e && e[0] == '1' && async) {
        // moex (DSV41_PLAN §73): the blocks mapped now, faulted in and registered by *async (the caller joins it before the
        // memory is copied to a GPU): a block that maps not is cudaHostAlloc'ed now; one that registers not stays mapped,
        // unregistered (a copy from it takes the driver's staged path), its reg its size all the same (munmap at the end)
        const bool huge0 = huge_pin();
        std::vector<char> mapped(nb, 0);
        uint64_t tot = 0;
        for (size_t k = 0; k < nb; ++k) {
            void* m = huge0 ? map_2m(bytes[k])
                            : mmap(nullptr, (size_t) bytes[k], PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (m == MAP_FAILED || m == nullptr) {
                if (cudaHostAlloc((void**) &out[k], (size_t) bytes[k], cudaHostAllocPortable) != cudaSuccess) return false;
                continue;
            }
            madvise(m, (size_t) bytes[k], MADV_DONTFORK);
            out[k] = static_cast<uint8_t*>(m);
            mapped[k] = 1;
            tot += bytes[k];
        }
        pending->store(tot, std::memory_order_release);
        faulted_out->store(0, std::memory_order_release);
        *async = std::thread([bytes, mapped, &out, &reg, pending, faulted_out, huge0, dev] {
            bool huge = huge0;
            std::string huge_note;
            cudaSetDevice(dev);
            constexpr uint64_t kThreads = 8;
            for (size_t k = 0; k < bytes.size(); ++k) {
                if (mapped[k] && huge) {
                    strata::core::fault_in_paced(out[k], bytes[k], (int) kThreads, &huge, &huge_note);
                } else if (mapped[k]) {
                    if (huge_pin()) madvise(out[k], (size_t) bytes[k], MADV_NOHUGEPAGE);
                    const uint64_t pages = bytes[k] / 4096, per = (pages + kThreads - 1) / kThreads;
                    std::vector<std::thread> ts;
                    for (uint64_t t = 0; t < kThreads; ++t)
                        ts.emplace_back([&, t] {
                            volatile uint8_t* p = out[k];
                            for (uint64_t pg = t * per; pg < std::min(pages, (t + 1) * per); ++pg) p[pg * 4096] = 0;
                        });
                    for (auto& t : ts) t.join();
                }
                faulted_out->store(k + 1, std::memory_order_release);
                if (mapped[k]) {
                    if (cudaHostRegister(out[k], (size_t) bytes[k], cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess)
                        (void) cudaGetLastError();
                    reg[k] = bytes[k];
                    pending->fetch_sub(bytes[k], std::memory_order_acq_rel);
                }
            }
            if (!huge_note.empty()) fprintf(stderr, "strata-x: pinned blocks: %s\n", huge_note.c_str());
        });
        return true;
    }
    if (const char* e = std::getenv("STRATA_DSV41_FAST_PIN"); e && e[0] == '1') {
        // moex (DSV41_BACKLOG #102), opt-in STRATA_DSV41_HUGE_PIN=1: each block on 2 MiB pages - this box's IOMMU translates
        // the GPUs' reads of host memory (DMA-FQ), and from 4 KB pages GPU 1's copies ran at 8.1 GB/s, from huge pages 12.3
        // (hugepage_bench); GPU 0's x8 link is full either way.  Faulted in paced (strata::core::fault_in_paced): the request
        // stops for the rest of the blocks as soon as compaction shows
        bool huge = huge_pin();
        for (size_t k = 0; k < nb; ++k) {
            void* m = huge ? map_2m(bytes[k])
                           : mmap(nullptr, (size_t) bytes[k], PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (m == MAP_FAILED || m == nullptr) continue;
            madvise(m, (size_t) bytes[k], MADV_DONTFORK);   // like the driver's allocation: never copied into a child
            out[k] = static_cast<uint8_t*>(m);
        }
        std::atomic<size_t> faulted{0};
        std::string huge_note;
        std::thread fl([&] {
            constexpr uint64_t kThreads = 8;
            for (size_t k = 0; k < nb; ++k) {
                if (out[k] && huge) {
                    strata::core::fault_in_paced(out[k], bytes[k], (int) kThreads, &huge, &huge_note);
                } else if (out[k]) {
                    if (huge_pin()) madvise(out[k], (size_t) bytes[k], MADV_NOHUGEPAGE);   // (stopped: the rest on 4 KB pages)
                    const uint64_t pages = bytes[k] / 4096, per = (pages + kThreads - 1) / kThreads;
                    std::vector<std::thread> ts;
                    for (uint64_t t = 0; t < kThreads; ++t)
                        ts.emplace_back([&, t] {
                            volatile uint8_t* p = out[k];
                            for (uint64_t pg = t * per; pg < std::min(pages, (t + 1) * per); ++pg) p[pg * 4096] = 0;
                        });
                    for (auto& t : ts) t.join();
                }
                faulted.store(k + 1, std::memory_order_release);
            }
        });
        for (size_t k = 0; k < nb; ++k) {
            while (faulted.load(std::memory_order_acquire) <= k) std::this_thread::sleep_for(std::chrono::microseconds(200));
            if (!out[k]) continue;
            if (cudaHostRegister(out[k], (size_t) bytes[k], cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess) {
                reg[k] = bytes[k];
            } else {
                (void) cudaGetLastError();   // a sticky error would name a later call
                munmap(out[k], (size_t) bytes[k]);
                out[k] = nullptr;
            }
        }
        fl.join();
        if (!huge_note.empty()) fprintf(stderr, "strata-x: pinned blocks: %s\n", huge_note.c_str());
    }
    for (size_t k = 0; k < nb; ++k)
        if (!out[k] && cudaHostAlloc((void**) &out[k], (size_t) bytes[k], cudaHostAllocPortable) != cudaSuccess) {
            out[k] = nullptr;
            return false;
        }
    return true;
}

bool Engine::Impl::rec_io_init(std::string& err) {
    // moex (DSV41_PLAN §18), opt-in STRATA_DSV41_DRIVE_GUARD=1 (=floor: held at its floor, a check): the drive guard, before
    // the record readers and init_cache's load pass, which ask it for their limits (drive_guard.hpp)
    if (const char* g = std::getenv("STRATA_DSV41_DRIVE_GUARD"); g && (std::strcmp(g, "1") == 0 || std::strcmp(g, "floor") == 0)) {
        const char* st = std::getenv("STRATA_DSV41_DRIVE_GUARD_STAT");
        const std::string where = !opt.pack.empty() ? opt.pack : opt.experts_gguf.substr(0, opt.experts_gguf.find_last_of('/'));
        const std::string path = st && st[0] ? std::string(st) : dsv41::DriveGuard::disk_stat(where);
        if (path.empty())
            fprintf(stderr, "strata-x: the drive guard: no disk counters for %s - it stays off\n", where.c_str());
        else
            dsv41::DriveGuard::start(path, std::strcmp(g, "floor") == 0, opt.verbose);
    }
    auto r = std::make_unique<RecordIo>();
    r->fd.assign((size_t) n_xl, -1);
    for (int64_t il = hp.n_dense; il < n_xl; ++il) {
        if (!L[(size_t) il].rec_stride) continue;
        if (L[(size_t) il].rec_stride % 4096) return fail(err, "deepseek41: a record stride that is no 4 KiB multiple (no O_DIRECT reads)");
        if (L[(size_t) il].rec_gguf) {   // moex (the GGUF set): its slices are read from the set's files (opened below)
            r->slot = std::max<uint64_t>(r->slot, L[(size_t) il].rec_stride);
            continue;
        }
        const std::string path = opt.pack + "/" + hp.rec_file[(size_t) il];
        const int fd = open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd < 0) return fail(err, "cannot open " + path + " for O_DIRECT reads");
        // moex (DSV41_BACKLOG #102): the records are only read O_DIRECT, so what the page cache holds of them is dead weight -
        // dropped before the huge pages are asked for, so they find free memory instead of compacting (dsx's lesson)
        if (huge_pin()) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        r->fd[(size_t) il] = fd;
        r->slot = std::max<uint64_t>(r->slot, L[(size_t) il].rec_stride);
    }
    r->xfd.assign(xg.size(), -1);   // moex (the GGUF set): its files, O_DIRECT, as the record files
    for (size_t i = 0; i < xg.size(); ++i) {
        if (!xg[i]) continue;
        const int fd = open(xg[i]->path().c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd < 0) return fail(err, "cannot open " + xg[i]->path() + " for O_DIRECT reads");
        if (huge_pin()) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        r->xfd[i] = fd;
    }
    if (!r->slot) return true;   // no records
    // 4 MiB pieces of several records in flight; read-ahead on 2 readers at most (moex, §32: STRATA_DSV41_AHEAD_READERS=N)
    const char* ar = std::getenv("STRATA_DSV41_AHEAD_READERS");
    r->io = std::make_unique<dsv41::ExpertIo>(12, ar ? std::clamp(std::atoi(ar), 1, 12) : 2);
    set_dev(stages[0]->dev);
    // 64 slots: one window's layer misses, the next layer's read-ahead, and the last few layers' records again; more
    // (--ssd-cache-gib): the records the decode used lately, out of the RAM copy's budget (init_cache plans after this)
    r->layer_pool.assign((size_t) n_xl, 0);
    std::vector<uint64_t> pool_n;   // slots per pool
    if (opt.ssd_cache_gib > 0) {   // a pool per record size, its share of the bytes by its layers' record bytes
        std::map<uint64_t, uint64_t> share;   // stride -> the bytes of one record of every layer of that stride
        uint64_t all = 0;
        for (int64_t il = hp.n_dense; il < n_xl; ++il)
            if (const uint64_t st = L[(size_t) il].rec_stride) {
                share[st] += st;
                all += st;
            }
        double bytes = opt.ssd_cache_gib * 1073741824.0;
        // moex (DSV41_BACKLOG #48): the cache is pinned before the RAM copy is sized (init_cache), so it keeps the RAM copy's
        // headroom itself - at most MemAvailable minus resident_headroom_gib (28 GiB: 17 GiB free froze the desktop twice); the
        // RAM copy then gets what is left above the headroom, 0 when the cache took it all
        const double head = (opt.experts_in_ram == 1 ? std::min(opt.resident_headroom_gib, 4.0) : opt.resident_headroom_gib) * 1073741824.0;
        if (const uint64_t avail = mem_available_bytes(); avail > 0 && bytes > (double) avail - head) {
            const double room = std::max(0.0, (double) avail - head);
            fprintf(stderr, "strata-x: SSD tier: the cache cut from %.1f to %.1f GiB: MemAvailable %.1f GiB, headroom %.0f GiB\n",
                    bytes / 1073741824.0, room / 1073741824.0, (double) avail / 1073741824.0, head / 1073741824.0);
            bytes = room;
        }
        for (const auto& [st, b] : share) {
            r->pool_stride.push_back(st);
            pool_n.push_back(std::max<uint64_t>(16, (uint64_t) (bytes * (double) b / (double) all / (double) st)));
        }
        for (int64_t il = hp.n_dense; il < n_xl; ++il)
            if (const uint64_t st = L[(size_t) il].rec_stride)
                r->layer_pool[(size_t) il] = (int) (std::find(r->pool_stride.begin(), r->pool_stride.end(), st) - r->pool_stride.begin());
    } else {
        r->pool_stride.push_back(r->slot);
        pool_n.push_back(64);
    }
    r->pool_slots.resize(r->pool_stride.size());
    const auto ta = clk::now();
    std::vector<uint64_t> blk_bytes;   // the cache's blocks: kGroup slots of a pool each
    for (size_t pi = 0; pi < r->pool_stride.size(); ++pi)
        for (uint64_t i = 0; i < pool_n[pi]; i += RecordIo::kGroup)
            blk_bytes.push_back(std::min<uint64_t>(RecordIo::kGroup, pool_n[pi] - i) * r->pool_stride[pi]);
    static const bool pin_async = [] {   // moex (DSV41_PLAN §73): STRATA_DSV41_PIN_ASYNC=0: the cache pinned before this returns
        const char* e = std::getenv("STRATA_DSV41_PIN_ASYNC");
        return !(e && e[0] == '0');
    }();
    if (!(pin_async ? pin_blocks(blk_bytes, r->blocks, r->block_reg, &r->pin_thr, &r->pin_pending, &r->blocks_faulted, stages[0]->dev)
                    : pin_blocks(blk_bytes, r->blocks, r->block_reg)))
        return fail(err, "the SSD cache's pinned slots");
    for (size_t pi = 0, k = 0; pi < r->pool_stride.size(); ++pi) {
        const uint64_t st = r->pool_stride[pi];
        for (uint64_t i = 0; i < pool_n[pi]; i += RecordIo::kGroup, ++k) {
            const uint64_t n = std::min<uint64_t>(RecordIo::kGroup, pool_n[pi] - i);
            r->block_first.push_back((int) r->cs.size());   // (moex, §63 step 2: a block's slots, for lending it)
            r->block_n.push_back((int) n);
            for (uint64_t j = 0; j < n; ++j) {
                RecordIo::Slot c;
                c.p = r->blocks[k] + j * st;
                c.pool = (int) pi;
                r->pool_slots[pi].push_back((int) r->cs.size());
                r->cs.push_back(std::move(c));
            }
        }
    }
    r->slot_block.assign(r->cs.size(), 0);
    for (size_t k = 0; k < r->block_first.size(); ++k)
        for (int i = r->block_first[k]; i < r->block_first[k] + r->block_n[k]; ++i) r->slot_block[(size_t) i] = (int) k;
    const bool pin_bg = r->pin_thr.joinable();
    const size_t n_reg = pin_bg ? 0 : (size_t) std::count_if(r->block_reg.begin(), r->block_reg.end(), [](uint64_t b) { return b > 0; });
    r->ppin.assign(r->cs.size(), 0);
    r->hold.assign(r->cs.size(), 0);
    r->lent.assign(r->cs.size(), 0);
    r->dpin.assign(r->cs.size(), 0);
    // The lent experts' records kept in the SSD cache for the prompt (and its refill): on by default since 2026-10-18 (warmed
    // prefill 2,000 tokens 38.5 -> 36.1 s, 7,820 tokens 52.2 -> 47.5 s, every lent slot refilled from RAM; the prompt's logits
    // bitwise unchanged); STRATA_DSV41_PROMPT_KEEP=0 turns it off.  STRATA_DSV41_PROMPT_KEEP_SHARE=F (0 by default): the
    // prompt's other reads too, up to F of each pool - a share of 1 made a warmed cache's prefill slower (200 tokens 18.4 ->
    // 27.0 s, 2,000 tokens 38.5 -> 51.2 s): the prompt's insertions evicted records the same prompt needed a few layers on
    if (const char* e = std::getenv("STRATA_DSV41_PROMPT_KEEP"); !(e && e[0] == '0')) {
        const char* sh = std::getenv("STRATA_DSV41_PROMPT_KEEP_SHARE");
        const double f = sh ? std::clamp(std::atof(sh), 0.0, 1.0) : 0.0;
        r->keep_on = true;
        r->kept.assign(r->pool_slots.size(), 0);
        for (const auto& ps : r->pool_slots) r->keep_cap.push_back((int) (f * (double) ps.size()));
        const char* lg = std::getenv("STRATA_DSV41_PROMPT_KEEP_LONG");   // opt-in: a long prompt's reads kept for its later chunks
        const double fl = lg ? std::clamp(std::atof(lg), 0.0, 1.0) : 0.0;
        for (const auto& ps : r->pool_slots) r->keep_cap_long.push_back((int) (fl * (double) ps.size()));
    }
    // read-ahead is opt-in (STRATA_DSV41_AHEAD=1): its wrong guesses cost SSD reads the drive has no room for - 64 slots,
    // +21-28% SSD traffic; the 60 GiB cache, 3.43 vs 3.78 tok/s with 44% more reads (its slots evict useful records)
    // (moex, DSV41_PLAN §23: the switch only ever turned it off - ahead_on starts false - so =1 did nothing since it went opt-in;
    // §27: =d guesses d layers ahead, 1-8)
    if (const char* e = std::getenv("STRATA_DSV41_AHEAD"); e && e[0] >= '1' && e[0] <= '8') {
        r->ahead_on = true;
        ahead_d = e[0] - '0';
    }
    for (size_t si = 0; si < stages.size(); ++si) {   // the prompt stream's bounce slots, per stage (each stage's issuer reads
        Stage& g = *stages[si];                         // on its own thread)
        set_dev(g.dev);
        const int nb = std::clamp(bounce_slots(si), 4, 512);
        g.rbounce.assign((size_t) nb, nullptr);
        g.rbounce_ev.assign((size_t) nb, nullptr);
        std::vector<uint64_t> breg;   // a block each (never freed, as before: the process's end frees them)
        if (!pin_blocks(std::vector<uint64_t>((size_t) nb, r->slot), g.rbounce, breg))
            return fail(err, "the SSD tier's bounce slots on GPU " + std::to_string(g.dev));
        for (int i = 0; i < nb; ++i)
            if (cudaEventCreateWithFlags(&g.rbounce_ev[i], cudaEventDisableTiming) != cudaSuccess)
                return fail(err, "the SSD tier's bounce slots on GPU " + std::to_string(g.dev));
    }
    reset_dev_cache();
    if (opt.verbose) {
        std::string pools;
        double gib = 0;
        for (size_t pi = 0; pi < r->pool_stride.size(); ++pi) {
            char b[96];
            snprintf(b, sizeof b, "%s%zu of %.1f MiB", pi ? " + " : "", r->pool_slots[pi].size(), r->pool_stride[pi] / 1048576.0);
            pools += b;
            gib += r->pool_slots[pi].size() * (double) r->pool_stride[pi] / 1073741824.0;
        }
        char how[160] = "";
        if (pin_bg) snprintf(how, sizeof how, ": %zu blocks mapped, faulted in and registered beside the load (STRATA_DSV41_PIN_ASYNC=0: first)",
                             r->blocks.size());
        else if (n_reg) snprintf(how, sizeof how, ": %zu of %zu blocks faulted in and registered, STRATA_DSV41_FAST_PIN", n_reg, r->blocks.size());
        fprintf(stderr, "strata-x: SSD tier: records read with O_DIRECT (12 readers, a cache of %s slots = %.1f GiB pinned in %.1f s%s, "
                        "read-ahead %s), never through the page cache\n", pools.c_str(), gib, ms_since(ta) / 1000.0, how, r->ahead_on ? "on" : "off");
        if (r->keep_on)
            fprintf(stderr, "strata-x: SSD tier: a prompt's lent experts are kept in the SSD cache for it and its refill%s "
                            "(STRATA_DSV41_PROMPT_KEEP=0: off)\n", r->keep_cap[0] > 0 ? ", and its other reads up to STRATA_DSV41_PROMPT_KEEP_SHARE of "
                            "the slots" : "");
        // moex (DSV41_BACKLOG #57): every DeepSeek switch set for this start, so a log shows what the config applied
        std::string sw;
        for (char** e = environ; e && *e; ++e)
            if (strncmp(*e, "STRATA_DSV4", 11) == 0) sw += std::string(sw.empty() ? "" : " ") + (*e + 7);
        fprintf(stderr, "strata-x: DeepSeek switches: %s\n", sw.empty() ? "none" : sw.c_str());
    }
    rio = std::move(r);
    return true;
}

bool Engine::Impl::rec_read(int64_t il, const int64_t* es, int n, uint8_t* const* dst, std::string& err) {
    RecordIo& r = *rio;
    if (r.fd[(size_t) il] < 0 && !L[(size_t) il].rec_gguf) return fail(err, "rec_read: layer " + std::to_string(il) + " has no record file");
    std::vector<dsv41::ExpertIo::Handle> h((size_t) n);
    for (int i = 0; i < n; ++i) h[(size_t) i] = rec_io(il, es[i], dst[i], dsv41::IoClass::Demand);
    r.rec_reads.fetch_add(n, std::memory_order_relaxed);
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        r.io->wait(h[(size_t) i]);
        if (h[(size_t) i]->error.load()) ok = false;
    }
    if (!ok) return fail(err, "an O_DIRECT read of layer " + std::to_string(il) + "'s records failed");
    return true;
}

dsv41::ExpertIo::Handle Engine::Impl::rec_submit(int64_t il, int64_t e, uint8_t* dst) {
    RecordIo& r = *rio;
    r.rec_reads.fetch_add(1, std::memory_order_relaxed);
    return rec_io(il, e, dst, dsv41::IoClass::Demand);
}

/// moex (the GGUF set): expert e of layer il read into dst, a slot of rec_stride bytes (4 KiB aligned) - a pack's record in
/// one extent; a GGUF set's virtual record (load_experts_gguf) in its three slices, each into its place (0, up_off,
/// down_off): a slice that starts inside a sector is read from that sector's start, the bytes before it landing in the gap
/// before its place - except the gate's, whose first sector goes through the request's own buffer, so nothing is written
/// before the slot; the last sector's tail lands in the gap after the place (the layout leaves room: load_experts_gguf)
dsv41::ExpertIo::Handle Engine::Impl::rec_io(int64_t il, int64_t e, uint8_t* dst, dsv41::IoClass cls) {
    RecordIo& r = *rio;
    const Layer& l = L[(size_t) il];
    if (!l.rec_gguf) return r.io->submit(r.fd[(size_t) il], (uint64_t) e * l.rec_stride, l.rec_stride, dst, cls);
    dsv41::ExpertIo::Seg seg[dsv41::ExpertIo::kMaxSegs];
    dsv41::ExpertIo::Fix fix;
    int n = 0;
    const uint64_t len[3] = {l.sg, l.su, l.sd};
    const size_t place[3] = {0, l.fmt.up_off, l.fmt.down_off};
    for (int k = 0; k < 3; ++k) {
        const int fd = r.xfd[l.f_shard[k]];
        const uint64_t at = l.f_off[k] + (uint64_t) e * len[k], lo = at & ~(uint64_t) 511, h = at - lo;
        const uint64_t end = (at + len[k] + 511) & ~(uint64_t) 511;
        // (need: the slice's own bytes in each extent - the last sector may run past the end of the file)
        if (k == 0 && h) {
            fix.seg = n;
            fix.dst = dst;
            fix.skip = (uint32_t) h;
            fix.len = (uint32_t) std::min<uint64_t>(512 - h, len[k]);
            seg[n++] = {lo, 512, nullptr, fd, std::min<uint64_t>(512, h + len[k])};
            if (end > lo + 512) seg[n++] = {lo + 512, end - lo - 512, dst + (512 - h), fd, h + len[k] - 512};
        } else {
            seg[n++] = {lo, end - lo, dst + place[k] - h, fd, h + len[k]};
        }
    }
    return r.io->submit(r.xfd[l.f_shard[0]], seg, n, fix.seg >= 0 ? &fix : nullptr, cls);
}

namespace {
/// STRATA_DSV41_READS_BY_LAYER=1 (a diagnosis): the decode's demand reads and cache hits per layer, printed by strata-x per
/// run (dsv41_reads_by_layer); the host thread only (ssd_slot's callers)
std::vector<int64_t> g_layer_reads(256), g_layer_hits(256), g_layer_reads0(256), g_layer_hits0(256);
}  // namespace

void dsv41_reads_by_layer(double tokens, int n_layer) {
    fprintf(stderr, "strata-x:          SSD reads / cache hits per token by layer:");
    for (int il = 0; il < n_layer && il < 256; ++il) {
        fprintf(stderr, " %d:%.2f/%.1f", il, (g_layer_reads[(size_t) il] - g_layer_reads0[(size_t) il]) / tokens,
                (g_layer_hits[(size_t) il] - g_layer_hits0[(size_t) il]) / tokens);
        g_layer_reads0[(size_t) il] = g_layer_reads[(size_t) il];
        g_layer_hits0[(size_t) il] = g_layer_hits[(size_t) il];
    }
    fprintf(stderr, "\n");
}

int Engine::Impl::ssd_victim(int64_t key) {
    RecordIo& r = *rio;
    const int64_t il_k = key / hp.n_expert;
    const uint64_t need = L[(size_t) il_k].rec_stride;
    int best = -1, pref = -1;   // the least recently used slot; ... among those holding a layer >= prefer_ge (a long prompt's)
    for (size_t pi = (size_t) r.layer_pool[(size_t) il_k]; best < 0 && pi < r.pool_stride.size(); ++pi) {
        if (r.pool_stride[pi] < need) continue;
        for (const int i : r.pool_slots[pi]) {
            const RecordIo::Slot& c = r.cs[(size_t) i];
            if (c.pin == r.epoch || r.ppin[(size_t) i] || r.hold[(size_t) i] || r.lent[(size_t) i]) continue;
            if (c.h && !c.h->done.load(std::memory_order_acquire)) continue;
            if (r.prefer_ge >= 0 && (c.key < 0 || c.key / hp.n_expert >= r.prefer_ge) &&
                (pref < 0 || c.tick < r.cs[(size_t) pref].tick))
                pref = i;
            if (best < 0 || c.tick < r.cs[(size_t) best].tick) best = i;
        }
    }
    return pref >= 0 ? pref : best;
}

int Engine::Impl::ssd_slot(int64_t key, bool demand) {
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);   // (uncontended: no prompt runs during a decode)
    if (auto it = r.where.find(key); it != r.where.end()) {   // cached, or its read in flight
        RecordIo::Slot& c = r.cs[(size_t) it->second];
        if (demand && key / hp.n_expert < 256) ++g_layer_hits[(size_t) (key / hp.n_expert)];
        if (demand) {
            if (c.spec) {   // a read-ahead the layer asks for: its pieces not read yet move to the demand queue
                c.spec = false;
                ++r.ahead_used;
                if (c.h && !c.h->done.load(std::memory_order_acquire)) r.io->promote(c.h);
            }
            ++r.hits;
            c.tick = ++r.clock;
        }
        return it->second;
    }
    // the least recently used slot of the record's pool (its size class) that is not pinned by this call and has no read in
    // flight; that pool full, a pool of larger slots
    const int best = ssd_victim(key);
    if (best < 0) return -1;
    RecordIo::Slot& c = r.cs[(size_t) best];
    if (r.cached.empty()) r.cached.assign((size_t) n_xl * hp.n_expert, 0);
    if (c.key >= 0) {
        r.where.erase(c.key);
        r.cached[(size_t) c.key] = 0;
        if (c.spec) ++r.ahead_wasted;
    }
    const int64_t il = key / hp.n_expert, e = key % hp.n_expert;
    c.key = key;
    c.spec = !demand;
    c.tick = ++r.clock;
    c.h = rec_io(il, e, c.p, demand ? dsv41::IoClass::Demand : dsv41::IoClass::Ahead);
    ++r.reads;
    if (demand && il < 256) ++g_layer_reads[(size_t) il];
    if (!demand) ++r.ahead_issued;
    r.where[key] = best;
    r.cached[(size_t) key] = 1;
    return best;
}

// moex (DSV41_BACKLOG #67, opt-in STRATA_DSV41_PROMPT_LRU=1): a prompt's hit in the SSD cache refreshes the record's place
// in the LRU as a decode hit does - the prompt path is 73% of the drive's bytes, and only the decode's hits kept a record
// there.  The prompt's output is the same; with the route bias the decode's routing follows the cache, so measure it
static bool prompt_lru() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_DSV41_PROMPT_LRU");
        return e && e[0] == '1';
    }();
    return on;
}

// moex (DSV41_PLAN §41), two opt-in rules for a short prompt's reads past its share of the SSD cache (STRATA_DSV41_PROMPT_KEEP
// _SHARE, 0 by default: they went through bounce slots and the cache kept none of them):
//  * STRATA_DSV41_CACHE_FREE=1: a slot that holds no record takes it (nothing is evicted) - after a start the cache stayed
//    empty through the first prompt, and the first reply read 12.4 records a token from the drive;
//  * STRATA_DSV41_PROMPT_HOT=D: a record that 1 / D or more of the chunk's rows route to takes the least recently used slot -
//    the reply that follows routes much of what its prompt routed (a 9-turn trace replayed, pcsim: D = 26 took a turn's
//    drive reads 9-25% lower and the decode's 26%).
// Where a record is read from changes no result.
static bool cache_free() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_DSV41_CACHE_FREE");
        return e && e[0] == '1';
    }();
    return on;
}

/// moex (DSV41_PLAN §59 P6), STRATA_DSV41_KEEP_SPREAD=1: a long prompt's share of a pool (PROMPT_KEEP_LONG) split evenly among
/// the pool's layers below prefer_ge - the chunks' layers - instead of the first ones read taking all of it: every layer of a
/// window then reads part of its records from the drive and part from the cache, so the drive reads through the window rather
/// than only at its last layers.  The same slots and records: exact
static bool keep_spread() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_DSV41_KEEP_SPREAD");
        return e && e[0] == '1';
    }();
    return on;
}

static int prompt_hot() {
    static const int d = [] {
        const char* e = std::getenv("STRATA_DSV41_PROMPT_HOT");
        return e ? std::max(0, std::atoi(e)) : 0;
    }();
    return d;
}

int Engine::Impl::prompt_keep_slot(int64_t key, int cnt, int rows, int hot) {   // hot: PROMPT_HOT's divisor (-1: its own)
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    if (auto it = r.where.find(key); it != r.where.end()) {   // cached (or read by the other stage just now): copied from there
        r.ppin[(size_t) it->second] = 1;
        if (prompt_lru()) r.cs[(size_t) it->second].tick = ++r.clock;
        return -2;
    }
    const int64_t il = key / hp.n_expert, e = key % hp.n_expert;
    const int pool = r.layer_pool[(size_t) il];
    const bool lent = !lent_key.empty() && lent_key[(size_t) key];   // the refill reads it again: kept whatever the share
    const int cap = r.long_now ? r.keep_cap_long[(size_t) pool] : r.keep_cap[(size_t) pool];
    bool layer_room = true;   // (§59 P6) the layer's part of the pool's long share
    if (r.long_now && keep_spread() && !lent && r.prefer_ge > 0 && il < r.prefer_ge) {
        int nl = 0;
        for (int64_t j = 0; j < r.prefer_ge && j < n_xl; ++j) nl += hp.is_moe(j) && r.layer_pool[(size_t) j] == pool;
        if (r.kept_layer.size() != (size_t) n_xl) r.kept_layer.assign((size_t) n_xl, 0);
        layer_room = nl <= 0 || r.kept_layer[(size_t) il] < (cap + nl - 1) / nl;
    }
    int best = -1;
    bool share = true;   // counted in the prompt's share
    bool deep = false;   // (§63: a one-window call's read-ahead)
    if (lent || (r.kept[(size_t) pool] < cap && layer_room)) {
        best = ssd_victim(key);
    } else if (r.deep_cap > 0 && r.deep_n < r.deep_cap) {   // (moex, §63) past the share in a one-window call: the least recently
        best = ssd_victim(key);                             // used slot (a free one first), held until its layer has run
        deep = best >= 0;
        share = false;
    } else if (const int hd = hot >= 0 ? hot : prompt_hot(); !r.long_now && (cache_free() || hd > 0)) {   // (moex, §41) past the share
        const int v = ssd_victim(key);   // (the least recently used: a slot never filled has tick 0)
        const bool hot = hd > 0 && cnt > 0 && rows > 0 && (int64_t) cnt * hd >= rows;
        if (v >= 0 && cache_free() && r.cs[(size_t) v].key < 0) {
            best = v;
            r.adm_free.fetch_add(1, std::memory_order_relaxed);
        } else if (v >= 0 && hot) {
            best = v;
            r.adm_hot.fetch_add(1, std::memory_order_relaxed);
        }
        share = false;
    }
    if (best < 0) return -1;
    RecordIo::Slot& c = r.cs[(size_t) best];
    if (r.cached.empty()) r.cached.assign((size_t) n_xl * hp.n_expert, 0);
    if (c.key >= 0) {
        r.where.erase(c.key);
        r.cached[(size_t) c.key] = 0;
    }
    c.key = key;
    c.spec = false;
    c.tick = ++r.clock;
    c.h = rec_io(il, e, c.p, dsv41::IoClass::Demand);
    r.where[key] = best;
    r.cached[(size_t) key] = 1;
    r.ppin[(size_t) best] = 1;
    if (deep && !r.dpin[(size_t) best]) {
        r.dpin[(size_t) best] = 1;
        ++r.deep_n;
        r.adm_deep.fetch_add(1, std::memory_order_relaxed);
    }
    if (share) ++r.kept[(size_t) pool];
    if (share && r.kept_layer.size() == (size_t) n_xl) ++r.kept_layer[(size_t) il];
    r.kept_total.fetch_add(1, std::memory_order_relaxed);
    r.rec_reads.fetch_add(1, std::memory_order_relaxed);
    return best;
}

int Engine::Impl::prompt_keep_take(int64_t key) {
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    if (auto it = r.where.find(key); it != r.where.end()) {
        r.ppin[(size_t) it->second] = 1;
        if (prompt_lru()) r.cs[(size_t) it->second].tick = ++r.clock;
        return -1;
    }
    const int best = ssd_victim(key);
    if (best < 0) return -1;
    RecordIo::Slot& c = r.cs[(size_t) best];
    if (r.cached.empty()) r.cached.assign((size_t) n_xl * hp.n_expert, 0);
    if (c.key >= 0) {
        r.where.erase(c.key);
        r.cached[(size_t) c.key] = 0;
    }
    c.key = key;
    c.spec = false;
    c.tick = ++r.clock;
    c.h = nullptr;   // no read: the caller copies the record in (and waits for it) before anyone reads the slot
    r.where[key] = best;
    r.cached[(size_t) key] = 1;
    r.ppin[(size_t) best] = 1;
    return best;
}

void Engine::Impl::prompt_keep_drop(int slot) {
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    RecordIo::Slot& c = r.cs[(size_t) slot];
    if (c.key >= 0) {
        r.where.erase(c.key);
        if (!r.cached.empty()) r.cached[(size_t) c.key] = 0;
        c.key = -1;
    }
}

namespace {
/// moex (§63 step 2): block k of the cache can be lent - its bytes hold `bytes`, and none of its slots is pinned by a decode
/// layer's call, held, used by the running prompt, lent already or reading.  Under r.mu
bool block_free(const Engine::Impl::RecordIo& r, size_t k, size_t bytes) {
    const int f = r.block_first[k], n = r.block_n[k];
    if (n <= 0 || (size_t) n * r.pool_stride[(size_t) r.cs[(size_t) f].pool] < bytes) return false;
    for (int i = f; i < f + n; ++i) {
        const auto& c = r.cs[(size_t) i];
        if (c.pin == r.epoch || r.ppin[(size_t) i] || r.hold[(size_t) i] || r.lent[(size_t) i]) return false;
        if (c.h && !c.h->done.load(std::memory_order_acquire)) return false;
    }
    return true;
}
}  // namespace

int Engine::Impl::cache_lendable(size_t bytes) {
    if (!rio) return 0;
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    int n = 0;
    for (size_t k = 0; k < r.block_first.size(); ++k) n += block_free(r, k, bytes);
    // at most half the cache's blocks: the rest holds the call's lent VRAM experts for their refill and the records its later
    // windows read again - the 200K read lent all 34 of a 30 GiB cache's and read 473.6 GB, 1,008 of the refill's records among them
    return std::min(n, (int) r.block_first.size() / 2);
}

int Engine::Impl::cache_lend(int n, size_t bytes, std::vector<uint8_t*>& out) {
    out.clear();
    if (!rio || n <= 0) return 0;
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    std::vector<std::pair<uint64_t, size_t>> cand;   // (the block's latest use, the block): the least recently used lent first
    for (size_t k = 0; k < r.block_first.size(); ++k)
        if (block_free(r, k, bytes)) {
            uint64_t t = 0;
            for (int i = r.block_first[k]; i < r.block_first[k] + r.block_n[k]; ++i) t = std::max(t, r.cs[(size_t) i].tick);
            cand.push_back({t, k});
        }
    std::sort(cand.begin(), cand.end());
    if (r.cached.empty()) r.cached.assign((size_t) n_xl * hp.n_expert, 0);
    for (size_t q = 0; q < cand.size() && (int) out.size() < n; ++q) {
        const size_t k = cand[q].second;
        for (int i = r.block_first[k]; i < r.block_first[k] + r.block_n[k]; ++i) {
            RecordIo::Slot& c = r.cs[(size_t) i];
            if (c.key >= 0) {
                r.where.erase(c.key);
                r.cached[(size_t) c.key] = 0;
            }
            c.key = -1;
            c.spec = false;
            c.tick = 0;
            r.lent[(size_t) i] = 1;
        }
        r.lent_blocks.push_back((int) k);
        out.push_back(r.blocks[k]);
    }
    return (int) out.size();
}

void Engine::Impl::cache_unlend() {
    if (!rio) return;
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    for (const int k : r.lent_blocks)
        for (int i = r.block_first[(size_t) k]; i < r.block_first[(size_t) k] + r.block_n[(size_t) k]; ++i) r.lent[(size_t) i] = 0;
    r.lent_blocks.clear();
}

int Engine::Impl::ssd_hold(int64_t key) {
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    const auto it = r.where.find(key);
    if (it == r.where.end()) return -1;
    const RecordIo::Slot& c = r.cs[(size_t) it->second];
    if (c.h && (!c.h->done.load(std::memory_order_acquire) || c.h->error.load())) return -1;
    ++r.hold[(size_t) it->second];
    return it->second;
}

void Engine::Impl::ssd_release(int slot) {
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    if (r.hold[(size_t) slot] > 0) --r.hold[(size_t) slot];
}

// moex (DSV41_PLAN §44, Impl::swap_cache): a VRAM-only victim's record into a cache slot of its own - the least recently used,
// held and unpublished until the copy lands (ssd_publish); the incoming record's slot freed once its VRAM copy has landed
int Engine::Impl::ssd_take_for(int64_t key) {
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    if (r.where.count(key)) return -1;   // (cached already: nothing to copy)
    const int v = ssd_victim(key);
    if (v < 0) return -1;
    RecordIo::Slot& c = r.cs[(size_t) v];
    if (r.cached.empty()) r.cached.assign((size_t) n_xl * hp.n_expert, 0);
    if (c.key >= 0) {
        r.where.erase(c.key);
        r.cached[(size_t) c.key] = 0;
        if (c.spec) ++r.ahead_wasted;
    }
    c.key = -1;   // unpublished until the copy lands
    c.h = nullptr;
    c.spec = false;
    ++r.hold[(size_t) v];
    return v;
}

// moex (DSV41_PLAN §69): the SSD cache filled at the start - the profile's hottest experts that no VRAM slot and no RAM copy
// holds, as many as the pools have slots, read before the engine says READY - beside the VRAM experts' load pass (moex §73:
// started before it, finished after it, so the drive reads both at once). Upstream's RAM store starts this way (its
// 5,456 slots read in 37 s) and missed half as many experts a token in the head-to-head (§68), where ours started empty and
// learned its set from the decode's misses, one drive read each. The hottest go in last, so the LRU gives them up last; from
// then on the cache adapts as before. The model's math is the same; a VRAM swap may come sooner from a cache slot than from a
// later round (STRATA_DSV41_SWAP_CACHE), as with any change to where the experts sit. Measured (§69): the decode's drive reads
// a token the same as with the cache empty (26.2 / 20.6 / 22.6 against 26.6 / 22.3 / 21.6 on turns 1-3), the start 25 s
// longer - the decode replaces the profile's picks within a turn, and the size of the tiers is what bounds the misses. So it
// is opt-in: STRATA_DSV41_CACHE_WARM=1.
void Engine::Impl::ssd_warm(const std::vector<int64_t>& rank, size_t n_prof, const std::vector<int32_t>* planned) {
    if (!rio || !rio->io || n_prof == 0) return;
    if (const char* e = std::getenv("STRATA_DSV41_CACHE_WARM"); !(e && e[0] == '1')) return;
    RecordIo& r = *rio;
    const int64_t NE = hp.n_expert;
    std::vector<std::vector<int>> free_slots(r.pool_slots.size());   // each pool's empty slots
    for (size_t pi = 0; pi < r.pool_slots.size(); ++pi)
        for (const int i : r.pool_slots[pi])
            if (r.cs[(size_t) i].key < 0 && !r.lent[(size_t) i] && !r.hold[(size_t) i]) free_slots[pi].push_back(i);
    // (moex §73: started before the load pass, the VRAM plan says which experts the slots will hold - the residency table is
    //  filled after it; ending after it, the table)
    auto in_vram = [&](int64_t key) {
        if (planned) return key < (int64_t) planned->size() && (*planned)[(size_t) key] >= 0;
        return key < (int64_t) resident.size() && resident[(size_t) key] >= 0;
    };
    warm_picks.clear();
    for (size_t j = 0; j < n_prof && j < rank.size(); ++j) {
        const int64_t key = rank[j], il = key / NE;
        if (il < hp.n_dense || il >= n_xl || !L[(size_t) il].rec_stride) continue;
        if (in_vram(key) || (!ram_off.empty() && ram_off[(size_t) key] >= 0) || r.where.count(key)) continue;
        int slot = -1;   // its pool's, or a pool of larger slots' (as ssd_victim)
        for (size_t pi = (size_t) r.layer_pool[(size_t) il]; slot < 0 && pi < r.pool_stride.size(); ++pi)
            if (r.pool_stride[pi] >= L[(size_t) il].rec_stride && !free_slots[pi].empty()) {
                slot = free_slots[pi].back();
                free_slots[pi].pop_back();
            }
        if (slot >= 0) warm_picks.push_back({key, slot});
    }
    warm_t0 = clk::now();
    warm_hs.assign(warm_picks.size(), nullptr);
    // the reads submitted by a thread of their own, each once its slot's block is faulted in (§73: the fault-in writes a byte of
    // every page; it runs ahead of the drive); the demand class: before READY nothing else asks, the load pass reads beside
    warm_submit = std::thread([this, NE] {
        RecordIo& rr = *rio;
        for (size_t j = 0; j < warm_picks.size(); ++j) {
            const int slot = warm_picks[j].second;
            const size_t blk = (size_t) rr.slot_block[(size_t) slot];
            while (rr.blocks_faulted.load(std::memory_order_acquire) <= blk) std::this_thread::sleep_for(std::chrono::microseconds(200));
            warm_hs[j] = rec_io(warm_picks[j].first / NE, warm_picks[j].first % NE, rr.cs[(size_t) slot].p, dsv41::IoClass::Demand);
        }
    });
}

void Engine::Impl::ssd_warm_finish() {
    if (warm_submit.joinable()) warm_submit.join();
    if (warm_picks.empty() || !rio) return;
    RecordIo& r = *rio;
    const int64_t NE = hp.n_expert;
    for (auto& h : warm_hs) r.io->wait(h);
    size_t ok = 0, bad = 0;
    uint64_t bytes = 0;
    {
        std::lock_guard<std::mutex> lk(r.mu);
        if (r.cached.empty()) r.cached.assign((size_t) n_xl * NE, 0);
        for (size_t j = warm_picks.size(); j-- > 0;) {   // the coldest first: the hottest get the latest ticks
            const auto [key, slot] = warm_picks[j];
            if (warm_hs[j]->error.load() || r.where.count(key)) {
                bad += warm_hs[j]->error.load() ? 1 : 0;
                continue;
            }
            RecordIo::Slot& c = r.cs[(size_t) slot];
            c.key = key;
            c.spec = false;
            c.tick = ++r.clock;
            c.h = nullptr;
            r.where[key] = slot;
            r.cached[(size_t) key] = 1;
            bytes += L[(size_t) (key / NE)].fmt.bytes;
            ++ok;
        }
    }
    if (opt.verbose || bad)
        fprintf(stderr, "strata-x: SSD tier: the cache filled at the start with the profile's %zu hottest experts outside VRAM and the "
                        "RAM copy, beside the VRAM experts' load (%.1f GiB in %.1f s%s; STRATA_DSV41_CACHE_WARM=1)\n", ok,
                bytes / 1073741824.0, ms_since(warm_t0) / 1000.0,
                bad ? (", " + std::to_string(bad) + " reads failed and left their slots empty").c_str() : "");
    warm_picks.clear();
    warm_hs.clear();
}

void Engine::Impl::ssd_publish(int slot, int64_t key) {
    RecordIo& r = *rio;
    static std::atomic<int> checked{0};   // the first four victims' copies against their records, byte for byte (printed)
    if (const int k = checked.fetch_add(1); k < 4) {
        const int64_t il = key / hp.n_expert, e = key % hp.n_expert;
        const Layer& l = L[(size_t) il];
        uint8_t* buf = static_cast<uint8_t*>(std::aligned_alloc(4096, l.rec_stride));
        std::string err, parts;
        // the three slices, as the slot check compares them (moex, the GGUF set: its virtual record's gaps between them hold
        // whatever its reads' edge sectors and the slot's earlier use left there - no byte of the expert)
        const bool got = buf && rec_read(il, &e, 1, &buf, err);
        const uint8_t* p = r.cs[(size_t) slot].p;
        if (got) {
            const size_t at[3] = {0, l.fmt.up_off, l.fmt.down_off}, len[3] = {l.sg, l.su, l.sd};
            for (int s = 0; s < 3; ++s)
                if (std::memcmp(buf + at[s], p + at[s], len[s]) != 0)
                    parts += (parts.empty() ? " in its " : "+") + std::string(s == 0 ? "gate" : s == 1 ? "up" : "down");
        }
        std::free(buf);
        fprintf(stderr, "strata-x: SSD tier: a swap's victim (layer %lld expert %lld) into a cache slot from its VRAM copy: %s its "
                        "record%s (STRATA_DSV41_SWAP_CACHE, check %d of 4)\n", (long long) il, (long long) e,
                !got ? "NOT READ to compare with" : parts.empty() ? "byte for byte" : "UNLIKE", parts.c_str(), k + 1);
    }
    std::lock_guard<std::mutex> lk(r.mu);
    RecordIo::Slot& c = r.cs[(size_t) slot];
    if (r.cached.empty()) r.cached.assign((size_t) n_xl * hp.n_expert, 0);
    c.key = key;
    c.tick = ++r.clock;
    r.where[key] = slot;
    r.cached[(size_t) key] = 1;
    if (r.hold[(size_t) slot] > 0) --r.hold[(size_t) slot];
}

void Engine::Impl::ssd_free_dup(int slot, int64_t key) {
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    RecordIo::Slot& c = r.cs[(size_t) slot];
    if (c.key != key || r.hold[(size_t) slot] || r.ppin[(size_t) slot] || (c.h && !c.h->done.load(std::memory_order_acquire))) return;
    r.where.erase(key);
    if (!r.cached.empty()) r.cached[(size_t) key] = 0;
    c.key = -1;
    c.tick = 0;   // the next record taken goes here
}

void Engine::Impl::prompt_keep_reset() {
    if (!rio) return;
    tail_ahead_stop();   // (moex, §37: a call that ended early)
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    std::fill(r.ppin.begin(), r.ppin.end(), 0);
    std::fill(r.dpin.begin(), r.dpin.end(), 0);
    r.deep_n = r.deep_cap = 0;
    std::fill(r.kept.begin(), r.kept.end(), 0);
    std::fill(r.kept_layer.begin(), r.kept_layer.end(), 0);
    std::fill(lent_key.begin(), lent_key.end(), 0);
    r.long_now = false;
    r.prefer_ge = -1;
}

int Engine::Impl::prompt_keep_pin(int64_t key) {
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    const auto it = r.where.find(key);
    if (it == r.where.end()) return -1;
    r.ppin[(size_t) it->second] = 1;
    return it->second;
}

int Engine::Impl::prompt_keep_ahead(int64_t key) {
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    if (auto it = r.where.find(key); it != r.where.end()) {
        r.ppin[(size_t) it->second] = 1;
        return -2;
    }
    const int best = ssd_victim(key);
    if (best < 0) return -1;
    RecordIo::Slot& c = r.cs[(size_t) best];
    if (r.cached.empty()) r.cached.assign((size_t) n_xl * hp.n_expert, 0);
    if (c.key >= 0) {
        r.where.erase(c.key);
        r.cached[(size_t) c.key] = 0;
    }
    const int64_t il = key / hp.n_expert, e = key % hp.n_expert;
    c.key = key;
    c.spec = false;
    c.tick = ++r.clock;
    c.h = rec_io(il, e, c.p, dsv41::IoClass::Ahead);
    r.where[key] = best;
    r.cached[(size_t) key] = 1;
    r.ppin[(size_t) best] = 1;
    r.rec_reads.fetch_add(1, std::memory_order_relaxed);
    return best;
}

void Engine::Impl::prompt_keep_release(int64_t il) {
    RecordIo& r = *rio;
    std::lock_guard<std::mutex> lk(r.mu);
    for (size_t i = 0; i < r.cs.size(); ++i) {
        const RecordIo::Slot& c = r.cs[i];
        if (!r.ppin[i] || c.key < 0 || c.key / hp.n_expert != il) continue;
        if (!lent_key.empty() && lent_key[(size_t) c.key]) continue;   // the refill copies it back into its VRAM slot
        r.ppin[i] = 0;
        if (r.dpin[i]) {   // (§63: a one-window call's read-ahead slot back in its budget)
            r.dpin[i] = 0;
            --r.deep_n;
        }
    }
}

void Engine::Impl::tail_ahead_start(int64_t l0, int64_t l1) {
    tail_ahead_stop();
    std::vector<int64_t> keys;   // the tails' stream (prompt_tails): the layers' experts in no VRAM slot, by layer and id
    for (int64_t il = l0; il < l1 && il < n_xl; ++il) {
        if (!hp.is_moe(il) || !L[(size_t) il].rec_stride || last_ring_only(il)) continue;   // (§59 P1: no MoE there)
        for (int64_t e = 0; e < hp.n_expert; ++e) {
            const int64_t key = il * hp.n_expert + e;
            if (resident[(size_t) key] < 0 && on_ssd(key)) keys.push_back(key);
        }
    }
    {
        std::lock_guard<std::mutex> lk(rio->mu);
        rio->long_now = false;   // the last window's own reads are kept for no later window: their slots go to the tails
        rio->prefer_ge = -1;
    }
    tail_pf_stop = false;
    tail_pf_read = 0;
    tail_pf_held = 0;
    tail_pf_slots.clear();
    tail_pf = std::thread([this, keys] {
        std::vector<int64_t> want;
        for (const int64_t key : keys) {   // 1. those the cache holds, pinned for the tails before anything evicts them
            if (prompt_keep_pin(key) >= 0) ++tail_pf_held;
            else want.push_back(key);
        }
        for (size_t i = 0; i < want.size() && !tail_pf_stop.load(std::memory_order_acquire);) {   // 2. the others, in order
            const int ci = prompt_keep_ahead(want[i]);
            if (ci == -1) {   // no slot free yet: the window releases each layer it has passed
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            if (ci >= 0) {
                ++tail_pf_read;
                std::lock_guard<std::mutex> lk(rio->mu);
                tail_pf_slots.push_back(ci);
            } else {
                ++tail_pf_held;
            }
            ++i;
        }
    });
}

void Engine::Impl::tail_ahead_stop() {
    if (!tail_pf.joinable()) return;
    tail_pf_stop.store(true, std::memory_order_release);
    tail_pf.join();
    {   // the reads still queued: the tails need them now
        std::lock_guard<std::mutex> lk(rio->mu);
        for (const int ci : tail_pf_slots) {
            const RecordIo::Slot& c = rio->cs[(size_t) ci];
            if (c.h && !c.h->done.load(std::memory_order_acquire)) rio->io->promote(c.h);
        }
        tail_pf_slots.clear();
    }
    fprintf(stderr, "strata-x: the tails' records during the last window: %lld read ahead, %lld held from the cache "
                    "(STRATA_DSV41_TAIL_AHEAD)\n", (long long) tail_pf_read.load(), (long long) tail_pf_held.load());
}

void Engine::Impl::replay_init() {
    replay_layer = -1;
    replay_tail = 0;
    if (!hp.v41) return;
    // on: S5 passed 2026-10-18 - 7,820 and 12,000-token prompts give the same 64 greedy tokens with it and without it (the
    // 12K read 22% faster: its first chunk streams no expert of layers 21-39); STRATA_DSV41_REPLAY=0: the full pass
    if (const char* e = std::getenv("STRATA_DSV41_REPLAY"); e && e[0] == '0') return;
    int64_t last_src = -1;
    for (int64_t il = 0; il < hp.n_layer; ++il)
        if (hp.kv_owner(il)) last_src = il;
    if (last_src < 0 || last_src + 1 >= hp.n_layer) return;
    for (int64_t il = last_src + 1; il < hp.n_layer; ++il)   // every later layer reads the last source's rows, or none
        if (hp.kv_owner(il) || hp.idx_src[(size_t) il] > il || (hp.kv_src[(size_t) il] >= 0 && hp.kv_src[(size_t) il] != last_src))
            return;
    if (ldev[(size_t) last_src + 1] != (int) stages.size() - 1) return;   // the replayed layers all on the last stage
    replay_layer = last_src + 1;
    replay_tail = 128 * (hp.n_layer - last_src);   // each later layer's window reads 128 rows of the one below: 128 x 20 for V4.1
    if (opt.verbose)
        fprintf(stderr, "strata-x: bounded replay: layers %lld-%lld run over the last %lld rows before each read start (the prompt's end, "
                        "its checkpoints)\n", (long long) replay_layer, (long long) hp.n_layer - 1, (long long) replay_tail);
}

void Stage::replay_shift(int T, int Tr) {
    const int r0 = T - Tr;
    if (r0 <= 0 || Tr <= 0) return;
    // every per-row buffer a later layer reads from an earlier one: the streams, the carried pre-mix, and the last KV source's
    // top-k lists (the layers up to the next index source reuse them) and candidate masks (the later index sources' filter)
    const size_t cb = (size_t) dsv41_cand_bytes(m.dsv41_score_rows(), std::max(1, (int) hp.cand_bsize));
    const struct { void* p; size_t row; } bufs[] = {
        {R, (size_t) hp.hc * hp.n_embd * sizeof(float)},
        {cpre, 24 * sizeof(float)},
        {dsel, (size_t) hp.idx_topk * sizeof(int32_t)},
        {dcand, cb},
    };
    for (const auto& b : bufs) {
        if (!b.p || !b.row) continue;
        uint8_t* base = static_cast<uint8_t*>(b.p);
        for (int i = 0; i < Tr; i += r0) {   // forward, in pieces of at most r0 rows: no piece overlaps its source
            const int nr = std::min(r0, Tr - i);
            cudaMemcpyAsync(base + (size_t) i * b.row, base + (size_t) (r0 + i) * b.row, (size_t) nr * b.row, cudaMemcpyDeviceToDevice, s);
        }
    }
}

bool Engine::Impl::route_bias_init(std::string& err) {
    const char* e = std::getenv("STRATA_DSV41_ROUTE_BIAS");
    if (!hp.v41 || !rio || !e || std::atof(e) <= 0.0) return true;
    route_margin = (float) std::atof(e);
    if (const char* p = std::getenv("STRATA_DSV41_ROUTE_BIAS_PROMPT"); p && p[0] == '1') route_bias_prompt = true;
    const size_t n = (size_t) hp.n_layer * hp.n_expert;
    if (cudaHostAlloc((void**) &route_pen, n * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
        return fail(err, "the route bias table");
    float* dp = nullptr;
    if (cudaHostGetDevicePointer((void**) &dp, route_pen, 0) != cudaSuccess || dp != route_pen)
        return fail(err, "the route bias table is not addressable at the same pointer on the GPUs (no UVA)");
    if (const char* pd = std::getenv("STRATA_DSV41_PEN_DEVICE"); pd && pd[0] == '1') {   // moex (§33): each stage's copy in VRAM
        route_pen_dev.assign(stages.size(), nullptr);
        for (size_t si = 0; si < stages.size(); ++si) {
            cudaSetDevice(stages[si]->dev);
            if (cudaMalloc((void**) &route_pen_dev[si], n * sizeof(float)) != cudaSuccess)
                return fail(err, "the route bias table's copy on GPU " + std::to_string(stages[si]->dev));
        }
    }
    route_pen_update();
    if (opt.verbose)
        fprintf(stderr, "strata-x: ROUTE BIAS ON (STRATA_DSV41_ROUTE_BIAS=%g%s): an expert the SSD would read is chosen only when it leads "
                        "the best resident one by more than %g - the output differs from the release's routing\n", route_margin,
                route_bias_prompt ? ", the prompt path too" : ", decode only", route_margin);
    return true;
}

void Engine::Impl::route_pen_update() {
    if (!route_pen) return;
    const int64_t NE = hp.n_expert;
    if (pen_prompt && !route_bias_prompt) {   // a prompt being read: its tokens route exactly, windows or not
        std::fill(route_pen, route_pen + (size_t) hp.n_layer * NE, 0.0f);
        return;
    }
    for (int64_t il = 0; il < hp.n_layer; ++il)
        for (int64_t e = 0; e < NE; ++e) {
            const int64_t key = il * NE + e;
            const bool miss = !L[(size_t) il].rec_stride ? false
                              : (key < (int64_t) resident.size() && resident[(size_t) key] >= 0) ? false
                              : (!ram_off.empty() && ram_off[(size_t) key] >= 0) ? false
                              : !(key < (int64_t) rio->cached.size() && rio->cached[(size_t) key]);
            route_pen[key] = miss ? route_margin : 0.0f;
        }
}

const uint8_t* Engine::Impl::ssd_cached(int64_t key) {
    RecordIo& r = *rio;
    const RecordIo::Slot* cp = nullptr;
    {
        std::unique_lock<std::mutex> lk(r.mu, std::defer_lock);
        if (r.keep_on || prompt_lru()) lk.lock();   // STRATA_DSV41_PROMPT_KEEP: the stages' issuers insert into the cache meanwhile
        const auto it = r.where.find(key);
        if (it == r.where.end()) return nullptr;
        cp = &r.cs[(size_t) it->second];
        if (r.keep_on) r.ppin[(size_t) it->second] = 1;   // the copy from it is queued: it stays until the prompt ends
        if (prompt_lru()) r.cs[(size_t) it->second].tick = ++r.clock;
    }
    const RecordIo::Slot& c = *cp;
    if (c.h) {
        r.io->wait(c.h);
        if (c.h->error.load()) return nullptr;
    }
    r.rec_cached.fetch_add(1, std::memory_order_relaxed);
    return c.p;
}

bool Engine::Impl::ahead_from(int64_t il) const {
    const int64_t t = il + ahead_d;   // the layer guessed: on this layer's GPU (its router's weights are there)
    return rio && rio->ahead_on && t < hp.n_layer && hp.is_moe(t) && ldev[(size_t) t] == ldev[(size_t) il] && L[(size_t) t].rec_stride > 0;
}

void Engine::Impl::ssd_ahead(int64_t il, const int32_t* ids, int n) {
    if (!rio || !rio->ahead_on || il >= hp.n_layer) return;
    const int64_t E = hp.E_of(il);
    for (int i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= E) continue;
        const int64_t key = il * E + e;
        if ((!resident.empty() && resident[(size_t) key] >= 0) || !on_ssd(key)) continue;   // in VRAM or RAM already
        if (ssd_slot(key, false) < 0) return;   // the cache is busy: the rest are left to the demand
    }
}

bool Engine::Impl::save_cache_keys(const std::string& path, std::string& err) {
    if (!rio) return true;
    std::vector<std::pair<uint64_t, int64_t>> v;   // (tick, key): the records read for a use (not an unused read-ahead)
    {
        std::lock_guard<std::mutex> lk(rio->mu);
        for (const auto& c : rio->cs)
            if (c.key >= 0 && !c.spec) v.push_back({c.tick, c.key});
    }
    std::sort(v.begin(), v.end());
    FILE* f = fopen((path + ".tmp").c_str(), "w");
    if (!f) return fail(err, "cannot write " + path);
    fprintf(f, "# strata-x SSD cache: %zu records, least recently used first (layer expert)\n", v.size());
    for (const auto& [tick, key] : v) fprintf(f, "%lld %lld\n", (long long) (key / hp.n_expert), (long long) (key % hp.n_expert));
    const bool ok = fclose(f) == 0;
    if (!ok || std::rename((path + ".tmp").c_str(), path.c_str()) != 0) return fail(err, "cannot write " + path);
    return true;
}

int64_t Engine::Impl::refill_cache(const std::string& path) {
    if (!rio) return 0;
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return 0;
    int64_t n = 0;
    char line[96];
    while (fgets(line, sizeof line, f)) {
        long long il = -1, e = -1;
        if (line[0] == '#' || sscanf(line, "%lld %lld", &il, &e) != 2) continue;
        if (il < hp.n_dense || il >= n_xl || e < 0 || e >= hp.n_expert) continue;
        const int64_t key = il * hp.n_expert + e;
        if ((!resident.empty() && resident[(size_t) key] >= 0) || !on_ssd(key)) continue;   // in VRAM or RAM already
        if (ssd_slot(key, false) < 0) break;   // (in order: the last ones read become the most recent)
        ++n;
    }
    fclose(f);
    return n;
}

void Engine::Impl::trace_route(int64_t il, int T, int k, const int32_t* ids) {
    static FILE* f = [] {
        const char* p = std::getenv("STRATA_ROUTE_TRACE");
        return p && *p ? fopen(p, "wb") : nullptr;
    }();
    if (!f) return;
    static std::mutex mu;   // the stages' threads
    std::lock_guard<std::mutex> g(mu);
    const int32_t h[3] = {(int32_t) il, T, k};
    fwrite(h, sizeof h, 1, f);
    fwrite(ids, sizeof(int32_t), (size_t) T * k, f);
}

bool Engine::Impl::run_cpu_jobs(int64_t il, std::vector<kc::ExpertJobMulti>& jobs, const std::vector<int64_t>& keys, std::string& err,
                                const std::vector<int32_t>* ahead, int64_t ahead_layer) {
    const kc::NativeFmt& f = L[(size_t) il].fmt;
    std::vector<kc::ExpertJobMulti> ram, ssd;
    std::vector<int64_t> ssd_k;
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (on_ssd(keys[i])) {
            ssd.push_back(jobs[i]);
            ssd_k.push_back(keys[i]);
        } else {
            ram.push_back(jobs[i]);
        }
    }
    ssd_st.cpu_ram += (int64_t) ram.size();
    ssd_st.cpu_ssd += (int64_t) ssd.size();
    if (ssd.empty()) {
        if (ahead) ssd_ahead(ahead_layer, ahead->data(), (int) ahead->size());   // nothing to wait for here: the guesses at once
        if (!jobs.empty()) pool->run_split_multi_native(f, jobs.data(), (int) jobs.size());
        return true;
    }
    RecordIo& r = *rio;
    bool ram_done = false;
    for (size_t b0 = 0; b0 < ssd.size();) {   // as many as the cache takes at once (usually all of them)
        ++r.epoch;   // this batch's pins
        std::vector<int> slot;
        for (size_t i = b0; i < ssd.size(); ++i) {
            const int sl = ssd_slot(ssd_k[i], true);
            if (sl < 0) break;
            r.cs[(size_t) sl].pin = r.epoch;
            slot.push_back(sl);
        }
        if (slot.empty()) return fail(err, "the SSD cache has no free slot (every one pinned or being read)");
        if (ahead && b0 == 0) ssd_ahead(ahead_layer, ahead->data(), (int) ahead->size());   // queued behind this layer's demand
        // what the pool can compute now - the RAM experts and the records the cache holds already - runs while the missing
        // records are read; then those.  (It computed the cache's records after the reads: ~120 of the 147 CPU experts a
        // token, ~40 ms a token on the native pack, waited behind the SSD for nothing.)  Each job writes its own rows and
        // the pool splits by rows, so the order changes no result.
        std::vector<kc::ExpertJobMulti> now, rest;
        if (!ram_done) now = ram;
        ram_done = true;
        std::vector<size_t> later;
        auto take = [&](size_t i, std::vector<kc::ExpertJobMulti>& to) -> bool {
            RecordIo::Slot& c = r.cs[(size_t) slot[i]];
            if (c.h && c.h->error.load() && c.key >= 0) {   // moex (DSV41_BACKLOG #61): read once more into the same slot, as the
                const int64_t kl = c.key / hp.n_expert, ke = c.key % hp.n_expert;   // prompt stream does (#47); the second
                auto h2 = rec_io(kl, ke, c.p, dsv41::IoClass::Demand);              // failure ends the request as before
                r.io->wait(h2);
                r.rec_reads.fetch_add(1, std::memory_order_relaxed);
                if (!h2->error.load()) c.h = h2;
            }
            if (c.h && c.h->error.load()) {   // a failed read leaves the cache
                r.where.erase(c.key);
                if (!r.cached.empty()) r.cached[(size_t) c.key] = 0;
                c.key = -1;
                return fail(err, "an O_DIRECT read of layer " + std::to_string(il) + "'s records failed");
            }
            kc::ExpertJobMulti& jb = ssd[b0 + i];
            jb.blob = c.p;   // the whole record: the pool's one-blob form
            jb.up = jb.down = nullptr;
            to.push_back(jb);
            return true;
        };
        for (size_t i = 0; i < slot.size(); ++i) {
            const RecordIo::Slot& c = r.cs[(size_t) slot[i]];
            if (!c.h || c.h->done.load(std::memory_order_acquire)) {
                if (!take(i, now)) return false;
            } else {
                later.push_back(i);
            }
        }
        if (!now.empty()) pool->run_split_multi_native(f, now.data(), (int) now.size());   // meanwhile
        const auto tw = clk::now();
        for (size_t i : later) {
            r.io->wait(r.cs[(size_t) slot[i]].h);
            if (!take(i, rest)) return false;
        }
        ssd_st.ms_ssd_wait += ms_since(tw);
        if (!rest.empty()) pool->run_split_multi_native(f, rest.data(), (int) rest.size());
        b0 += slot.size();
    }
    return true;
}

/// A two-GPU split at K keeps every layer on the second GPU off the first's state: its KV source, its index source and the
/// candidate source it narrows by all at K or later.
bool Engine::Impl::dsv41_split_ok(int64_t K) const {
    if (hp.engram)   // the engram layers on the first GPU: its thread has the tokens
        for (int il : hp.eg_layers)
            if (il >= K) return false;
    for (int64_t il = K; il < hp.n_layer; ++il) {
        const int ks = hp.kv_src[(size_t) il], is = hp.idx_src[(size_t) il];
        if ((ks >= 0 && ks < K) || (is >= 0 && is < K)) return false;
        if (hp.cand_src >= 0 && hp.cand_src < K && il > hp.cand_src && hp.idx_owner(il)) return false;
    }
    return true;
}

uint64_t Engine::Impl::dsv41_state_bytes(int64_t il) const {
    const int64_t D = hp.hd_k, ID = hp.idx_dim;
    uint64_t b = (uint64_t) dsv4_raw_rows() * D * 2;
    if (hp.kv_owner(il)) {
        const int r = hp.cratio[(size_t) il];
        b += (uint64_t) (opt.n_ctx / r + 2) * (kv_pack() ? dsv41_pack_row_bytes((int) D, 16) + dsv41_pack_row_bytes((int) ID, 32)
                                                         : (uint64_t) (D + ID) * 2);
        if (r == 2) b += (uint64_t) dsv4_ring(2) * D * 4 * 2;
    }
    return b;
}

/// V4.1's decode state: every layer's raw K = V ring; on the KV sources the compressed rows and the index keys of the whole
/// context (n_ctx / ratio + 2 rows of 512 and of 128 halves) and, at ratio 2, the compressor's state ring - the layers after a
/// source read its rows (Stage::dsv41).  Zeroed, on the layer's GPU.
bool Engine::Impl::alloc_state_dsv41(std::string& err) {
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
        if (r > 0 && hp.kv_owner(il)) {
            l.c_rows = opt.n_ctx / r + 2;
            // (moex, #104: packed, a row's bytes in uint16_t units)
            const int64_t cw = kv_pack() ? (int64_t) dsv41_pack_row_bytes((int) D, 16) / 2 : D;
            const int64_t lw = kv_pack() ? (int64_t) dsv41_pack_row_bytes((int) ID, 32) / 2 : ID;
            if (!(l.ccache = g.dalloc<uint16_t>((size_t) (l.c_rows * cw), err)) ||
                !(l.lcache = g.dalloc<uint16_t>((size_t) (l.c_rows * lw), err)))
                return false;
            if (r == 2) {
                const int64_t S = dsv4_ring(2);
                if (!(l.c_ring_kv = g.dalloc<float>((size_t) (S * D), err)) || !(l.c_ring_sc = g.dalloc<float>((size_t) (S * D), err)))
                    return false;
            }
        }
        bytes += dsv41_state_bytes(il);
    }
    reset_dev_cache();
    if (opt.verbose)
        fprintf(stderr, "strata-x: DeepSeek-V4.1 attention state %.1f MiB for %lld positions (raw rings of %lld rows)\n",
                bytes / 1048576.0, (long long) opt.n_ctx, (long long) dsv4_raw_rows());
    return true;
}

// ------------------------------------------------------------------ DSpark (the release's mtp.0-2)

/// DeepSeek-V4.1's DSpark drafter (model.py DSparkBlock, forward_embed / forward_head): three ratio-0 blocks with the trunk's
/// geometry but 128 routed experts top-3 each, written into the deepseek41 GGUF as dspark.blk.J.* (their dense weights, on the
/// last GPU with the head) and the pack's experts/M<J>.bin records (into a pinned RAM arena, run by the CPU pool); the block
/// projection of the trunk's features (main_proj, main_norm), the last block's norm, the Markov head (its embed and head as
/// F32 copies) and the confidence head.  The blocks join the layer table at L[n_load ..] as V4's drafter's do (dspark.cpp:
/// the capture, the inject and the draft step are shared; their V4.1 turns are in there).
bool Engine::Impl::load_dspark_v41(std::string& err) {
    // moex: from the model's own GGUF (--draft-dspark builtin) or from another deepseek41 GGUF that holds the drafter
    // (--draft-dspark FILE, e.g. our dense file dsv41-1017-dense-q8_0-dsp-eg.gguf): a GGUF set as the model (gguf_convert.py's,
    // unsloth's) has no dspark.* tensors - llama.cpp's converter leaves the drafter out - while it keeps the release's FP8
    // dense weights (STRATA_DSV41_DENSE_FP8), which our dense file has as Q8_0.  The experts stay the pack's M<j>.bin either way
    const bool builtin = opt.dspark == "builtin";
    std::unique_ptr<strata::GgufModel> own;
    if (!builtin) {
        try {
            own = std::make_unique<strata::GgufModel>(strata::GgufModel::open(opt.dspark));
        } catch (const std::exception& e) {
            return fail(err, std::string("the DSpark drafter: ") + e.what());
        }
        const strata::MetaValue* a = own->meta().get("general.architecture");
        if (!a || a->s != "deepseek41") return fail(err, "the DSpark drafter: " + opt.dspark + " is not a deepseek41 GGUF");
    }
    const strata::GgufModel& dg = builtin ? *gm : *own;
    const strata::GgufFile& mt = dg.meta();
    const std::string q = "deepseek41.dspark.";
    auto num = [&](const char* k, int64_t& v) -> bool {
        const strata::MetaValue* x = mt.get(q + k);
        if (!x) return fail(err, "missing " + q + k + " (a GGUF written without the drafter)");
        v = (int64_t) x->num();
        return true;
    };
    auto arr = [&](const char* k, const strata::MetaValue*& v) -> bool {
        v = mt.get(q + k);
        if (!v || v->type != strata::MetaType::ARRAY || v->items.size() != v->count) return fail(err, q + k + ": not a whole array");
        return true;
    };
    int64_t nb = 0, bs = 0, noise = 0, rank = 0, E = 0, k = 0;
    const strata::MetaValue *tl = nullptr, *rty = nullptr, *rst = nullptr, *rfl = nullptr;
    if (!num("block_count", nb) || !num("block_size", bs) || !num("noise_token_id", noise) || !num("markov_rank", rank) ||
        !num("expert_count", E) || !num("expert_used_count", k) || !arr("target_layers", tl) || !arr("expert_record.type", rty) ||
        !arr("expert_record.stride", rst) || !arr("expert_record.file", rfl))
        return false;
    if (nb < 1 || bs < 2 || bs > kTmax || E < 1 || E > hp.n_expert || k < 1 || k > hp.n_used || (int64_t) rty->items.size() < nb ||
        (int64_t) rst->items.size() < nb || (int64_t) rfl->items.size() < nb || noise < 0 || noise >= hp.n_vocab)
        return fail(err, "deepseek41.dspark.*: implausible settings");
    if (opt.pack.empty()) return fail(err, "deepseek41: the DSpark experts are in the pack directory (--pack DIR)");
    auto d = std::make_unique<Dspark>();
    d->v41 = true;
    d->n = nb;
    d->block = (int) bs;
    d->base = n_load;
    d->mask = (int32_t) noise;
    d->rank = rank;
    Stage& lg = last_stage();
    const int di_last = (int) stages.size() - 1;
    for (const auto& it : tl->items) {
        const int64_t il = (int64_t) it.u;
        if (il < 0 || il > hp.n_layer) return fail(err, "the DSpark drafter: target layer " + std::to_string(il) + " out of range");
        if (il < hp.n_layer && ldev[(size_t) il] != di_last)
            return fail(err, "the DSpark drafter's target layer " + std::to_string(il) + " is not on the last GPU (--split)");
        if (eg_index(il) >= 0)   // the engine captures before a layer's engram write, the reference after it
            return fail(err, "the DSpark drafter: target layer " + std::to_string(il) + " is an engram layer");
        d->tl.push_back(il);
    }
    if (!builtin) {   // its dense tensors into one allocation on the last GPU as they are in the file - as load_dense puts the
                      // builtin ones there (Q8_0 projections, F32 norms and mixers; the BF16 router gets its F32 copy below)
        struct Item {
            const strata::TensorInfo* ti;
            size_t shard;
            uint64_t bytes, off;
        };
        std::vector<Item> items;
        uint64_t total = 0;
        for (size_t sh = 0; sh < dg.size(); ++sh)
            for (const auto& ti : dg.shard(sh).tensors()) {
                // (the Markov head goes from the file into its F32 copies below, not onto the GPU twice)
                if (ti.name.rfind("dspark.", 0) != 0 || ti.name.rfind("dspark.markov_", 0) == 0) continue;
                const uint64_t b = strata::tensor_payload_bytes(ti);
                if (!b || !dg.in_bounds(ti, sh)) return fail(err, "the DSpark drafter: tensor " + ti.name + " of unknown size");
                items.push_back({&ti, sh, b, total});
                total += (b + 255) & ~uint64_t(255);
            }
        if (items.empty()) return fail(err, "the DSpark drafter: no dspark.* tensors in " + opt.dspark);
        set_dev(lg.dev);
        if (cudaMalloc(&d->dense, total) != cudaSuccess)
            return fail(err, "the DSpark drafter: " + std::to_string(total >> 20) + " MiB of dense weights on GPU " + std::to_string(lg.dev));
        for (const auto& it : items) {
            uint8_t* dst = static_cast<uint8_t*>(d->dense) + it.off;
            if (cudaMemcpy(dst, dg.shard(it.shard).tensor_data(*it.ti), it.bytes, cudaMemcpyHostToDevice) != cudaSuccess)
                return fail(err, "the DSpark drafter: upload of " + it.ti->name);
            DT x;
            x.d = dst;
            x.type = (int) it.ti->type;
            for (size_t k = 0; k < it.ti->shape.size() && k < 4; ++k) x.ne[k] = (int64_t) it.ti->shape[k];
            x.bytes = it.bytes;
            d->t[it.ti->name] = x;
        }
        if (opt.verbose)
            fprintf(stderr, "strata-x: DSpark drafter (V4.1) from %s: %zu dense tensors, %.2f GiB on GPU %d\n", opt.dspark.c_str(),
                    items.size(), total / 1073741824.0, lg.dev);
    }
    const std::map<std::string, DT>& tm = builtin ? t : d->t;
    auto get = [&](const std::string& nm, DT& x) -> bool {
        auto f = tm.find(nm);
        if (f == tm.end()) return fail(err, "missing tensor " + nm);
        x = f->second;
        return true;
    };
    const int64_t n = hp.n_embd, V = hp.n_vocab;
    if (!get("dspark.main_proj.weight", d->fc) || !get("dspark.main_norm.weight", d->enc_norm) ||
        !get("dspark.output_norm.weight", d->out_norm) || !get("dspark.confidence.weight", d->conf))
        return false;
    if (d->fc.n_in() != (int64_t) d->tl.size() * n || d->fc.n_out() != n || !kn::native_mmvq_supported(d->fc.type))
        return fail(err, "dspark.main_proj: not [n_target * n_embd, n_embd] of a GPU type");
    if (d->enc_norm.type != 0 || d->out_norm.type != 0 || d->conf.type != 0 || d->conf.n_in() != n + rank)
        return fail(err, "dspark.main_norm / output_norm / confidence: F32 of the expected sizes");
    // the Markov head (BF16 [rank, n_vocab] each): F32 copies on the last GPU, row t = token t's rank vector
    for (int which = 0; which < 2; ++which) {
        const char* nm = which == 0 ? "dspark.markov_embed.weight" : "dspark.markov_head.weight";
        size_t shd = 0;
        const strata::TensorInfo* ti = dg.find(nm, &shd);
        if (!ti || ti->type != GGML_TYPE_BF16 || ti->shape.size() < 2 || (int64_t) ti->shape[0] != rank || (int64_t) ti->shape[1] != V)
            return fail(err, std::string(nm) + ": not BF16 [rank, n_vocab]");
        std::vector<float> h((size_t) (rank * V));
        bf16_to_f32(reinterpret_cast<const uint16_t*>(dg.shard(shd).tensor_data(*ti)), h.data(), h.size());
        set_dev(lg.dev);
        float* dv = lg.dalloc<float>(h.size(), err);
        if (!dv || cudaMemcpy(dv, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            return fail(err, "the DSpark drafter: the Markov head's F32 copy");
        (which == 0 ? d->w1 : d->w2) = dv;
    }
    // its blocks, appended to the layer table (ratio 0, on the last GPU): the trunk's names under dspark.blk.J
    const int64_t base = d->base;
    L.resize((size_t) (base + nb));
    ldev.resize((size_t) (base + nb), di_last);
    hp.cratio.resize((size_t) (base + nb), 0);
    hp.head_kv.resize((size_t) (base + nb), 1);
    hp.kv_src.resize((size_t) (base + nb), -1);
    hp.idx_src.resize((size_t) (base + nb), -1);
    const float clamp = hp.clamp_exp.empty() ? 0.0f : hp.clamp_exp.back();
    hp.clamp_exp.resize((size_t) (base + nb), clamp);
    hp.clamp_sh.resize((size_t) (base + nb), hp.clamp_sh.empty() ? clamp : hp.clamp_sh.back());
    hp.l_expert.resize((size_t) (base + nb), 0);
    hp.l_used.resize((size_t) (base + nb), 0);
    uint64_t blob = 0;
    for (int64_t j = 0; j < nb; ++j) {
        const int64_t il = base + j;
        hp.l_expert[(size_t) il] = (int) E;
        hp.l_used[(size_t) il] = (int) k;
        Layer& l = L[(size_t) il];
        auto nm = [&](const char* sfx) { return "dspark.blk." + std::to_string(j) + "." + sfx; };
        bool ok = get(nm("attn_norm.weight"), l.attn_norm) && get(nm("ffn_norm.weight"), l.ffn_norm) &&
                  get(nm("hc_attn_fn.weight"), l.hc_attn_fn) && get(nm("hc_attn_base.weight"), l.hc_attn_base) &&
                  get(nm("hc_attn_scale.weight"), l.hc_attn_scale) && get(nm("hc_ffn_fn.weight"), l.hc_ffn_fn) &&
                  get(nm("hc_ffn_base.weight"), l.hc_ffn_base) && get(nm("hc_ffn_scale.weight"), l.hc_ffn_scale) &&
                  get(nm("attn_q_a.weight"), l.q_a) && get(nm("attn_q_a_norm.weight"), l.q_a_norm) && get(nm("attn_q_b.weight"), l.q_b) &&
                  get(nm("attn_kv.weight"), l.wkv) && get(nm("attn_kv_a_norm.weight"), l.kv_a_norm) &&
                  get(nm("attn_sinks.weight"), l.sinks) && get(nm("attn_output_a.weight"), l.wo_a) &&
                  get(nm("attn_output_b.weight"), l.wo_b) && get(nm("ffn_gate_inp.weight"), l.gate_inp) &&
                  get(nm("exp_probs_b.bias"), l.probs_b) && get(nm("ffn_gate_shexp.weight"), l.sh_gate) &&
                  get(nm("ffn_up_shexp.weight"), l.sh_up) && get(nm("ffn_down_shexp.weight"), l.sh_down);
        if (!ok) return false;
        for (const DT* x : {&l.q_a, &l.q_b, &l.wkv, &l.wo_a, &l.wo_b, &l.sh_gate, &l.sh_up, &l.sh_down})
            if (!kn::native_mmvq_supported(x->type)) return fail(err, "the DSpark drafter: a block projection of a type with no GPU kernel");
        if (l.hc_attn_fn.type != 0 || l.hc_ffn_fn.type != 0 || l.sinks.type != 0 || l.probs_b.type != 0 || l.probs_b.n_in() != E)
            return fail(err, "the DSpark drafter: its mixers, sinks and bias must be F32 (the bias of its " + std::to_string(E) + " experts)");
        if (l.gate_inp.ne[0] != n || l.gate_inp.n_out() != E) return fail(err, nm("ffn_gate_inp.weight") + ": shape");
        if (l.gate_inp.type == GGML_TYPE_BF16) {   // the router as an exact F32 copy, as the trunk's
            size_t shd = 0;
            const strata::TensorInfo* ti = dg.find(nm("ffn_gate_inp.weight"), &shd);
            std::vector<float> w((size_t) (n * E));
            bf16_to_f32(reinterpret_cast<const uint16_t*>(dg.shard(shd).tensor_data(*ti)), w.data(), w.size());
            set_dev(lg.dev);
            float* dw = lg.dalloc<float>(w.size(), err);
            if (!dw || cudaMemcpy(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
                return fail(err, "the DSpark drafter: the router's F32 copy");
            l.gate_inp.d = dw;
            l.gate_inp.type = 0;
        } else if (l.gate_inp.type != 0) {
            return fail(err, nm("ffn_gate_inp.weight") + ": neither BF16 nor F32");
        }
        const int rt = (int) rty->items[(size_t) j].u;
        if (!kc::native_fmt(rt, rt, n, hp.n_ff_exp, l.fmt, err)) return fail(err, "the DSpark drafter's experts: " + err);
        l.fmt.swiglu_limit = hp.clamp_exp[(size_t) il];
        if (rst->items[(size_t) j].u < l.fmt.bytes) return fail(err, "the DSpark drafter: a record stride below its blob");
        blob = std::max<uint64_t>(blob, (l.fmt.bytes + 4095) & ~(uint64_t) 4095);
        l.raw_rows = dsv4_raw_rows();   // its K = V ring: the trunk's raw-ring geometry
        set_dev(lg.dev);
        if (!(l.raw = lg.dalloc<uint16_t>((size_t) (l.raw_rows * hp.hd_k), err))) return false;
    }
    // the routed experts: [gate | up | down] records from the pack into a pinned arena, read by a few threads
    d->blob = blob;
    const uint64_t arena_bytes = (uint64_t) nb * E * blob;
    d->arena = std::make_unique<strata::core::PinnedArena>(arena_bytes);
    if (!d->arena->valid()) return fail(err, "the DSpark drafter: " + std::to_string(arena_bytes >> 20) + " MiB of RAM for its experts");
    {
        const auto t0 = clk::now();
        std::vector<int> fds((size_t) nb, -1);
        for (int64_t j = 0; j < nb; ++j) {
            const std::string path = opt.pack + "/" + rfl->items[(size_t) j].s;
            fds[(size_t) j] = open(path.c_str(), O_RDONLY);
            if (fds[(size_t) j] < 0) return fail(err, "cannot open " + path);
            struct stat sb {};   // exactly E records of the GGUF's stride, as the trunk's (load_experts_records)
            if (fstat(fds[(size_t) j], &sb) != 0 || (uint64_t) sb.st_size != (uint64_t) E * rst->items[(size_t) j].u) {
                for (int64_t i = 0; i <= j; ++i) close(fds[(size_t) i]);
                return fail(err, "the DSpark drafter: " + path + " is not " + std::to_string(E) + " records of " +
                                     std::to_string((unsigned long long) rst->items[(size_t) j].u) + " bytes");
            }
        }
        std::vector<std::thread> th;
        std::atomic<int64_t> next{0};
        std::atomic<bool> bad{false};
        for (int w = 0; w < 8; ++w)
            th.emplace_back([&] {
                for (int64_t key; !bad && (key = next.fetch_add(1)) < nb * E;) {
                    const int64_t j = key / E, e = key % E;
                    const Layer& l = L[(size_t) (base + j)];
                    const uint64_t stride = rst->items[(size_t) j].u;
                    if (pread(fds[(size_t) j], d->arena->data() + (uint64_t) key * blob, l.fmt.bytes, (off_t) ((uint64_t) e * stride)) !=
                        (ssize_t) l.fmt.bytes)
                        bad = true;
                }
            });
        for (auto& x : th) x.join();
        for (int fd : fds) {
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // the pinned copies serve now
            close(fd);
        }
        if (bad) return fail(err, "the DSpark drafter: a short read of its expert records");
        if (opt.verbose)
            fprintf(stderr, "strata-x: DSpark drafter (V4.1): %lld blocks of %lld experts top-%lld (%.2f GiB pinned, read in %.1f s), "
                            "block size %d, noise token %d, targets",
                    (long long) nb, (long long) E, (long long) k, arena_bytes / 1073741824.0, ms_since(t0) / 1000.0, d->block, d->mask);
        if (opt.verbose) {
            for (int64_t il : d->tl) fprintf(stderr, " %lld", (long long) il);
            fprintf(stderr, "\n");
        }
    }
    // the last stage's buffers: the features of up to max(kTmax, the window) rows; the confidence head's rows
    lg.dsp_rows = (int) std::max<int64_t>(kTmax, hp.n_swa);
    const int64_t ntg = (int64_t) d->tl.size();
    set_dev(lg.dev);
    if (!(lg.dsp_feat = lg.dalloc<float>((size_t) (lg.dsp_rows * ntg * n), err)) ||
        !(lg.dsp_tmp = lg.dalloc<float>((size_t) (lg.dsp_rows * n), err)) || !(lg.dsp_g = lg.dalloc<float>((size_t) (lg.dsp_rows * n), err)) ||
        !(lg.dsp_logits = lg.dalloc<float>((size_t) kTmax * V, err)) || !(lg.dsp_bias = lg.dalloc<float>((size_t) V, err)) ||
        !(lg.dsp_cin = lg.dalloc<float>((size_t) kTmax * (n + rank), err)) || !(lg.dsp_cout = lg.dalloc<float>((size_t) kTmax, err)) ||
        !(lg.dsp_dprev = lg.dalloc<int32_t>((size_t) kTmax + 1, err)) || !(lg.dsp_dprob = lg.dalloc<float>((size_t) kTmax, err)))
        return false;
    reset_dev_cache();
    dsp = std::move(d);
    return true;
}

// ------------------------------------------------------------------ the engram

/// The engram's inputs: the hash tables from the GGUF's metadata (Hparams::eg_*, engram_tables.py's), the pack's table files
/// mapped read-only (rows of eg_dim FP8 bytes in L<il>.weight.bin, eg_dim / 32 E8M0 bytes in L<il>.scale.bin), q_weight *
/// k_weight on the layer's GPU, and the window buffer on the engram layers' stage.  Nothing without hp.engram, or for any family
/// but DeepSeek-V4.1 (the load stops when no loader took an Engram: engine.cpp).
bool Engine::Impl::load_engram(std::string& err) {
    if (!hp.engram || !hp.v41) return true;   // DeepSeek-V4.1's only (Hy4-flash's: load_engram_hy4, hy4.cpp)
    auto e = std::make_unique<Engram>();
    e->L = (int) hp.eg_layers.size();
    e->cols = (int) ((hp.eg_ngram - 1) * hp.eg_heads);
    e->dim = (int) hp.eg_dim;
    if (e->L < 1 || e->cols < 1 || e->dim % 32 || hp.eg_ngram > 16)
        return fail(err, "deepseek41.engram: implausible geometry");
    if ((int64_t) hp.eg_mul.size() != e->L * hp.eg_ngram || (int64_t) hp.eg_primes.size() != (int64_t) e->L * e->cols ||
        (int64_t) hp.eg_offsets.size() != (int64_t) e->L * e->cols || (int64_t) hp.eg_token_map.size() < hp.n_vocab)
        return fail(err, "deepseek41.engram: the hash tables' sizes do not match the geometry");
    for (int j = 0; j < e->L; ++j)   // every row a hash can produce lies inside its table (engram_hash.hpp's check)
        for (int c = 0; c < e->cols; ++c) {
            const int64_t pr = hp.eg_primes[(size_t) j * e->cols + c], of = hp.eg_offsets[(size_t) j * e->cols + c];
            if (pr <= 0 || of < 0 || of + pr > hp.eg_rows[(size_t) j]) return fail(err, "deepseek41.engram: a bucket range leaves its table");
        }
    // moex (the GGUF set, moex/tools/dsv41/gguf_convert.py): STRATA_DSV41_ENGRAM_GGUF=FILE - the tables from a GGUF file's
    // per_layer_token_embd.weight in place of the pack's weight and scale files: MXFP8 (ggml type 67, or 43 at 33 bytes a
    // block - unsloth's two ids; mainline ggml's 43 is F8_E4M3, one byte a value, which the size check below refuses), rows
    // of dim / 32 blocks of [e8m0, 32 x e4m3], the engram layers' tables one after another in their order.  The release's
    // codes and scales, so eg_rows writes the same bytes into the buffers.  Read through the mapping only: the O_DIRECT,
    // dedup and n-gram reader paths and the scales' pre-read are for the pack's files (off with this)
    std::string egs;
    if (const char* ev = std::getenv("STRATA_DSV41_ENGRAM_GGUF"); ev && *ev) {
        egs = ev;
    } else {   // (the model's own GGUF set holds the table: gguf_convert.py's, unsloth's)
        size_t sh = 0;
        if (gm->find("per_layer_token_embd.weight", &sh)) egs = gm->shard(sh).path();
    }
    const char* egg = egs.c_str();
    const bool from_gguf = !egs.empty();
    if (opt.pack.empty() && !from_gguf) return fail(err, "deepseek41: the engram tables are in the pack directory (--pack DIR)");
    auto map = [&](const std::string& rel, uint64_t want, const uint8_t*& base, size_t& len) -> bool {
        const std::string path = opt.pack + "/" + rel;
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return fail(err, "cannot open " + path + " (the engram tables must be in the pack)");
        struct stat sb {};
        if (fstat(fd, &sb) != 0 || (uint64_t) sb.st_size != want) {
            close(fd);
            return fail(err, path + ": " + std::to_string((uint64_t) sb.st_size) + " bytes, expected " + std::to_string(want));
        }
        void* p = mmap(nullptr, (size_t) sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (p == MAP_FAILED) return fail(err, "cannot map " + path);
        madvise(p, (size_t) sb.st_size, MADV_RANDOM);   // single rows: no read-ahead around them
        rec_maps.push_back({p, (size_t) sb.st_size});
        base = static_cast<const uint8_t*>(p);
        len = (size_t) sb.st_size;
        return true;
    };
    e->w.resize((size_t) e->L);
    e->s.resize((size_t) e->L);
    e->wlen.resize((size_t) e->L);
    e->slen.resize((size_t) e->L);
    e->wfd.assign((size_t) e->L, -1);
    e->sfd.assign((size_t) e->L, -1);
    // moex (the engram wait, DSV41_BACKLOG), opt-in: STRATA_DSV41_EG_DIRECT=1 reads a prompt chunk's rows with O_DIRECT by a
    // pool of threads (eg_rows); the mappings stay for the decode windows' few rows.  Off by default: on the cold 44K read
    // (2026-10-19) it took 594.9 s against the mapped reads' 430.9 s - the engram waits 253.0 s against 111.3 s, its 4 KB
    // requests queued behind the experts' - with the same first token
    static const bool eg_mmap = !(std::getenv("STRATA_DSV41_EG_DIRECT") && std::getenv("STRATA_DSV41_EG_DIRECT")[0] == '1');
    // a row read whole from one 4 KB page: only when a row and its scales divide the page (V4.1: 256 and 8 bytes)
    const bool pages_fit = 4096 % e->dim == 0 && 4096 % (e->dim / 32) == 0;
    auto direct = [&](const std::string& rel) {
        return eg_mmap || !pages_fit ? -1 : open((opt.pack + "/" + rel).c_str(), O_RDONLY | O_DIRECT);
    };
    // moex (DSV41_BACKLOG #59), opt-in: STRATA_DSV41_EG_READER=1 reads a prompt chunk's rows through Strata's n-gram reader
    // (Qwen's ngram/ple_reader) - each table's pages deduplicated and sorted, at most STRATA_DSV41_EG_INFLIGHT (64) of them in
    // flight a table (its O_DIRECT threads: STRATA_IO_THREADS, 16), and a row cache of STRATA_DSV41_EG_CACHE_MIB (1024, the
    // layers' rows and scales together).  The mapped tables stay for the decode windows' few rows, and take over when a
    // read fails.  Measured against the mapped reads before it is anyone's default (the O_DIRECT reads above, 32 in flight
    // and no cache, were slower)
    const char* rdv = std::getenv("STRATA_DSV41_EG_READER");
    const bool eg_reader = rdv && rdv[0] == '1';
    const char* ifv = std::getenv("STRATA_DSV41_EG_INFLIGHT");
    const char* cmv = std::getenv("STRATA_DSV41_EG_CACHE_MIB");
    const uint32_t rd_inflight = (uint32_t) std::clamp(ifv ? std::atoi(ifv) : 64, 1, 1024);
    const double rd_mib = cmv ? std::max(0.0, std::atof(cmv)) : 1024.0;
    const uint64_t rd_cache_rows = (uint64_t) (rd_mib * 1048576.0 / ((double) e->L * (e->dim + e->dim / 32)));
    e->qk.resize((size_t) e->L, nullptr);
    const int64_t n = hp.n_embd, hc = hp.hc;
    if (from_gguf) {
        uint64_t total = 0;
        for (int j = 0; j < e->L; ++j) total += (uint64_t) hp.eg_rows[(size_t) j];
        // moex: or IQ4_NL (ggml type 20; unsloth's UD-IQ1_M / UD-IQ1_S ship their tables so): dim / 32 blocks of [fp16 d,
        // 16 bytes of 4-bit indices into kvalues_iq4nl] - the rows copied as they are, dequantized on the GPU as ggml does
        uint64_t rb = (uint64_t) e->dim / 32 * 33;
        bool iq4 = false;
        uint64_t data = 0, end = 0;
        try {   // (the header only; the file is mapped again below for the rows)
            strata::GgufFile g(egg);
            const strata::TensorInfo* ti = g.find("per_layer_token_embd.weight");
            if (!ti) return fail(err, std::string(egg) + ": no per_layer_token_embd.weight (STRATA_DSV41_ENGRAM_GGUF)");
            if (ti->type != 67 && ti->type != 43 && ti->type != 20)
                return fail(err, std::string(egg) + ": per_layer_token_embd is ggml type " + std::to_string(ti->type) +
                                 ", not MXFP8 (67 or 43) or IQ4_NL (20)");
            iq4 = ti->type == 20;
            if (iq4) rb = (uint64_t) e->dim / 32 * 18;
            if (ti->shape.size() != 2 || ti->shape[0] != (uint64_t) e->dim || ti->shape[1] != total)
                return fail(err, std::string(egg) + ": per_layer_token_embd is not [" + std::to_string(e->dim) + ", " + std::to_string(total) +
                                 "] (the engram layers' rows)");
            data = g.data_start() + ti->offset;
            end = g.file_size();   // the table ends where the next tensor starts, or at the end of the file
            for (const auto& t : g.tensors())
                if (t.offset > ti->offset) end = std::min<uint64_t>(end, g.data_start() + t.offset);
            if (end < data || end - data < total * rb || end - data >= total * rb + g.alignment())
                return fail(err, std::string(egg) + ": per_layer_token_embd holds " + std::to_string(end - data) + " bytes, not " +
                                 std::to_string(total * rb) + (iq4 ? " (IQ4_NL: 18 bytes a 32 values)" : " (MXFP8: 33 bytes a 32 values)"));
        } catch (const std::exception& ex) {
            return fail(err, std::string("STRATA_DSV41_ENGRAM_GGUF: ") + ex.what());
        }
        const int fd = open(egg, O_RDONLY);
        if (fd < 0) return fail(err, std::string("cannot open ") + egg);
        struct stat sb {};
        if (fstat(fd, &sb) != 0 || (uint64_t) sb.st_size < end) {
            close(fd);
            return fail(err, std::string(egg) + ": changed while it was read");
        }
        void* p = mmap(nullptr, (size_t) sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (p == MAP_FAILED) return fail(err, std::string("cannot map ") + egg);
        madvise(p, (size_t) sb.st_size, MADV_RANDOM);   // single rows: no read-ahead around them
        rec_maps.push_back({p, (size_t) sb.st_size});
        e->mx8 = true;
        e->iq4nl = iq4;
        e->rb = (size_t) rb;
        e->gfd = open(egg, O_RDONLY | O_DIRECT | O_CLOEXEC);   // (moex §73: a prompt chunk's rows read O_DIRECT; -1: mapped only)
        e->goff.assign((size_t) e->L, 0);
        uint64_t row0 = 0;
        for (int j = 0; j < e->L; ++j) {
            e->goff[(size_t) j] = data + row0 * rb;
            e->w[(size_t) j] = static_cast<const uint8_t*>(p) + data + row0 * rb;
            e->wlen[(size_t) j] = (size_t) ((uint64_t) hp.eg_rows[(size_t) j] * rb);
            e->s[(size_t) j] = nullptr;
            e->slen[(size_t) j] = 0;
            row0 += (uint64_t) hp.eg_rows[(size_t) j];
        }
        if (eg_reader || !eg_mmap)
            fprintf(stderr, "strata-x: engram: STRATA_DSV41_EG_READER / EG_DIRECT read the pack's tables: off with STRATA_DSV41_ENGRAM_GGUF\n");
    }
    for (int j = 0; j < e->L; ++j) {
        const uint64_t rows = (uint64_t) hp.eg_rows[(size_t) j];
        if (!from_gguf && (!map(hp.eg_wfile[(size_t) j], rows * (uint64_t) e->dim, e->w[(size_t) j], e->wlen[(size_t) j]) ||
                           !map(hp.eg_sfile[(size_t) j], rows * (uint64_t) (e->dim / 32), e->s[(size_t) j], e->slen[(size_t) j])))
            return false;
        if (!from_gguf) {
            e->wfd[(size_t) j] = direct(hp.eg_wfile[(size_t) j]);
            e->sfd[(size_t) j] = direct(hp.eg_sfile[(size_t) j]);
        }
        if (eg_reader && !from_gguf) {   // the reader's row ids are 32-bit (V4.1: 384M rows a table)
            if (rows > 0xFFFFFFFFull) return fail(err, "deepseek41.engram: a table of more than 2^32 rows for the n-gram reader");
            auto rw = std::make_unique<strata::ngram::PleReader>();
            auto rs = std::make_unique<strata::ngram::PleReader>();
            std::string re;
            if (!rw->open(opt.pack + "/" + hp.eg_wfile[(size_t) j], 0, rows, rd_inflight, rd_cache_rows, re, true, (uint32_t) e->dim) ||
                !rs->open(opt.pack + "/" + hp.eg_sfile[(size_t) j], 0, rows, rd_inflight, rd_cache_rows, re, true,
                          (uint32_t) (e->dim / 32)))
                return fail(err, "the engram tables' n-gram reader (STRATA_DSV41_EG_READER): " + re);
            e->rd_w.push_back(std::move(rw));
            e->rd_s.push_back(std::move(rs));
        }
        const int il = hp.eg_layers[(size_t) j];
        if (il < 0 || il >= n_load) return fail(err, "deepseek41.engram: layer " + std::to_string(il) + " out of range");
        const Layer& l = L[(size_t) il];
        const bool folded = !l.eg_kw;   // (the GGUF set: the product as engram_gate)
        if (!l.eg_wkv || l.eg_wkv.n_in() != (int64_t) e->cols * e->dim || l.eg_wkv.n_out() != (hc + 1) * n ||
            l.eg_qw.ne[0] != n || l.eg_qw.n_out() != hc || (!folded && (l.eg_kw.ne[0] != n || l.eg_kw.n_out() != hc)))
            return fail(err, "blk." + std::to_string(il) + ".engram.*: shapes");
        // q_weight * k_weight (model.py uses them only as a product), F32 on the layer's GPU
        std::vector<float> q((size_t) (hc * n)), k((size_t) (hc * n));
        Stage& g = stage_of(il);
        set_dev(g.dev);
        if (cudaMemcpy(q.data(), l.eg_qw.d, q.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
            (!folded && cudaMemcpy(k.data(), l.eg_kw.d, k.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess))
            return fail(err, "the engram's q / k weights");
        if (!folded)
            for (size_t i = 0; i < q.size(); ++i) q[i] *= k[i];
        float* d = g.dalloc<float>(q.size(), err);
        if (!d || cudaMemcpy(d, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
            return fail(err, "the engram's q * k: " + err);
        e->qk[(size_t) j] = d;
        if (ldev[(size_t) il] != 0) return fail(err, "deepseek41: the engram layers run on the first GPU (the tokens' stage)");
    }
    e->ids.assign((size_t) opt.n_ctx, 0);
    {   // the window buffer on the first stage (mapped: the decode graphs read it)
        Stage& g = *stages[0];
        set_dev(g.dev);
        const size_t bytes = (size_t) e->L * kTmax * e->row_bytes();
        if (cudaHostAlloc((void**) &g.eg_win_h, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &g.eg_win_d, g.eg_win_h, 0) != cudaSuccess)
            return fail(err, "the engram's window buffer");
        std::memset(g.eg_win_h, 0, bytes);
    }
    reset_dev_cache();
    eg = std::move(e);
    if (opt.verbose) {
        fprintf(stderr, "strata-x: engram at %d layers (%d rows of %d bytes per token each), tables mapped from %s\n", eg->L,
                eg->cols, eg->mx8 ? (int) eg->rb : eg->dim + eg->dim / 32,   // (MXFP8's rb is the same 264)
                eg->mx8 ? (std::string(egg) + (eg->iq4nl ? " (IQ4_NL, STRATA_DSV41_ENGRAM_GGUF)" : " (MXFP8, STRATA_DSV41_ENGRAM_GGUF)")).c_str()
                        : opt.pack.c_str());
        if (!eg->rd_w.empty())
            fprintf(stderr, "strata-x: engram: a prompt chunk's rows through the n-gram reader (STRATA_DSV41_EG_READER: %u in "
                            "flight a table, a row cache of %.0f MiB, %llu rows a layer)\n",
                    rd_inflight, rd_mib, (unsigned long long) rd_cache_rows);
    }
    // moex (DSV41_PLAN §42), opt-in STRATA_DSV41_EG_LATE=1: a decode graph window's rows fetched on a helper thread, waited
    // for before the first engram layer is served (eg_late) - that layer on the first stage, whose window buffer they go to
    if (const char* v = std::getenv("STRATA_DSV41_EG_LATE"); v && v[0] == '1' && !hp.eg_layers.empty()) {
        eg_late_layer = *std::min_element(hp.eg_layers.begin(), hp.eg_layers.end());
        eg_late = eg_late_layer < (int64_t) ldev.size() && ldev[(size_t) eg_late_layer] == stages[0]->di;
        if (eg_late) {
            eg_help = std::make_unique<EgHelper>();
            EgHelper* h = eg_help.get();
            h->th = std::thread([this, h] {
                uint64_t ran = 0;
                for (;;) {
                    std::unique_lock<std::mutex> lk(h->mu);
                    h->cv.wait(lk, [h, ran] { return h->seq != ran || h->stop; });
                    if (h->stop) return;
                    ran = h->seq;   // (posted again meanwhile: the latest job only - the window buffer's rows are its)
                    const int64_t pos0 = h->pos0;
                    const int T = h->T, cap = h->cap, r0 = h->r0;
                    uint8_t* buf = h->buf;
                    lk.unlock();
                    std::string e2;
                    h->ok = eg_rows(pos0, T, buf, cap, r0, e2);
                    {   // (a test: STRATA_X_EG_FAIL=N - the N-th fetch reports a failure; speed_ab.py F)
                        static const int64_t fail_at = std::getenv("STRATA_X_EG_FAIL") ? std::atoll(std::getenv("STRATA_X_EG_FAIL")) : 0;
                        static int64_t n_job = 0;
                        if (fail_at > 0 && ++n_job == fail_at) {
                            h->ok = false;
                            e2 = "the engram rows' fetch failed (STRATA_X_EG_FAIL)";
                        }
                    }
                    h->err = e2;
                    {   // (moex §76, eg_ring: a graph already waiting for these rows is told here; a failed fetch tells nothing -
                        //  the host finds it in eg_help_wait and releases the GPUs)
                        std::lock_guard<std::mutex> lk2(h->mu);
                        h->done_seq.store(ran, std::memory_order_release);
                        if (h->arm && h->arm_seq == ran) {
                            if (h->ok) {
                                std::atomic_thread_fence(std::memory_order_release);
                                *h->ring_ptr = h->ring_val;
                            }
                            h->arm = false;
                        }
                    }
                }
            });
        }
        if (opt.verbose)
            fprintf(stderr, "strata-x: engram: a decode window's rows %s (STRATA_DSV41_EG_LATE)\n",
                    eg_late ? ("fetched on a thread of their own as it starts, waited for before layer " + std::to_string(eg_late_layer) +
                               "'s MoE").c_str()
                            : "as before: the first engram layer is not on the first GPU");
    }
    // moex (DSV41_BACKLOG #140, DSV41_PLAN §75; STRATA_DSV41_EG_PRE=0: off): an engram layer's rows' half - the dequant and the
    // wkv projection (a 6,144 -> 25,600 gemv, ~0.45 ms a token on GPU 0's chain) - in the decode graph of the layer before it,
    // between that layer's VRAM experts and its wait for the CPU's rows (layer_graph_body). Only where the window's rows are
    // in by then: with eg_late from the first engram layer on, whose graph waits for them - so not that layer's own; and only
    // an FP8 / BF16 wkv (a quantized one writes xq, which the layer before still reads)
    if (opt.graphs && !stages.empty()) {
        const char* v = std::getenv("STRATA_DSV41_EG_PRE");
        Stage& g = *stages[0];
        if (!(v && v[0] == '0')) {
            const int n = (int) hp.n_embd, hc = (int) hp.hc;
            for (int j = 0; j < eg->L; ++j) {
                const int il = hp.eg_layers[(size_t) j], host = il - 1;
                if (host < g.lb || il >= g.le || !hp.is_moe(host) || pj_needs_q(L[(size_t) il].eg_wkv)) continue;
                if (eg_late && host < eg_late_layer) {   // the first engram layer: only with the helper's own word (eg_ring)
                    const char* r = std::getenv("STRATA_DSV41_EG_RING");
                    if (il != eg_late_layer || (r && r[0] == '0')) continue;
                    eg_ring = true;
                }
                if (g.eg_pre_kv.empty()) g.eg_pre_kv.assign((size_t) eg->L, nullptr);
                set_dev(g.dev);
                if (!g.eg_pre_att && !(g.eg_pre_att = g.dalloc<float>((size_t) kTmax * eg->cols * eg->dim, err)))
                    return fail(err, "the engram's early rows: " + err);
                if (!(g.eg_pre_kv[(size_t) j] = g.dalloc<float>((size_t) kTmax * (hc + 1) * n, err)))
                    return fail(err, "the engram's early rows: " + err);
                if (opt.verbose)
                    fprintf(stderr, "strata-x: engram: layer %d's rows are dequantized and projected in layer %d's decode graph, "
                                    "while it waits for the CPU's experts (STRATA_DSV41_EG_PRE=0: in its own)%s\n", il, host,
                            eg_ring && il == eg_late_layer ? " - behind the fetch's own word (STRATA_DSV41_EG_RING=0: not this layer's)" : "");
            }
            reset_dev_cache();
        }
    }
    return true;
}

/// moex (DSV41_PLAN §18), opt-in STRATA_DSV41_EG_SCALES_PRELOAD=1: the scale tables (2 x 3.07 GB on V4.1) read once into the
/// page cache behind the load, in order, by one thread: half of a token's 96 page reads are scale pages (a page holds 512
/// rows' scales against 16 rows' weights), and a cold 8K-token chunk's fetch took 13.71 s with them against 9.61 s with them
/// cached (eg_fetch_bench.py, 2026-10-21).  Page cache, not pinned: an evicted page is read again on demand, as without.
/// The thread waits only while the drive guard is below full depth: waiting for the fetches too, it never finished under a
/// long read right after the start (serve_guard.py's arm C, 2026-10-22: layer-major fetches a window ahead), the case the
/// pre-read is for; one reader in order queues a few requests at a time beside them.
void Engine::Impl::eg_preload_start() {
    const char* v = std::getenv("STRATA_DSV41_EG_SCALES_PRELOAD");
    if (!eg || !(v && v[0] == '1')) return;
    if (eg->mx8) {   // (STRATA_DSV41_ENGRAM_GGUF: each row carries its scales - there is no scale table to pre-read)
        if (opt.verbose) fprintf(stderr, "strata-x: engram: the scales are inside the GGUF's rows: nothing to pre-read\n");
        return;
    }
    Engram& e = *eg;
    e.pre = std::thread([this, &e] {
        const auto t0 = clk::now();
        const dsv41::DriveGuard* gd = dsv41::DriveGuard::get();
        std::vector<uint8_t> buf((size_t) 4 << 20);
        uint64_t got = 0;
        double waited = 0;
        for (int j = 0; j < e.L && !e.pre_stop.load(); ++j) {
            const std::string path = opt.pack + "/" + hp.eg_sfile[(size_t) j];
            const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) continue;
            posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
            for (off_t off = 0; !e.pre_stop.load();) {
                const auto tw = clk::now();
                while (!e.pre_stop.load() && gd && gd->engaged()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
                waited += ms_since(tw) / 1000.0;
                const ssize_t n = pread(fd, buf.data(), buf.size(), off);
                if (n <= 0) break;
                off += n;
                got += (uint64_t) n;
            }
            close(fd);
        }
        if (opt.verbose && !e.pre_stop.load())
            fprintf(stderr, "strata-x: engram: the scale tables read into the page cache (%.2f GB in %.1f s, %.1f s of it waiting for "
                            "the drive guard)\n", (double) got / 1e9, ms_since(t0) / 1000.0, waited);
    });
}

int Engine::Impl::eg_index(int64_t il) const {
    if (!eg) return -1;
    for (size_t j = 0; j < hp.eg_layers.size(); ++j)
        if (hp.eg_layers[j] == (int) il) return (int) j;
    return -1;
}

void Engine::Impl::eg_note(const int32_t* toks, int64_t pos0, int64_t n) {
    if (!eg) return;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t p = pos0 + i;
        if (p < 0 || p >= (int64_t) eg->ids.size()) continue;
        const int32_t t = toks[i];
        // engram.py NgramHashState: a position inside an image span is DEAD - it takes part in no n-gram (model.py forward's
        // engram_mask); the request's spans (img) cover a restored checkpoint's images too (images_match)
        bool in_image = false;
        for (const ImageSpan& im : img) in_image |= p >= im.start && p < im.start + im.n;
        eg->ids[(size_t) p] = in_image ? Engram::kDead
                              : t >= 0 && t < (int32_t) hp.eg_token_map.size() ? hp.eg_token_map[(size_t) t] : hp.eg_pad;
    }
}

/// engram.py NgramHashState for positions pos0..: the compressed ids of the max_ngram positions ending at p (the look-back
/// stops at position 0 and at a DEAD position, an image's: that slot and older ones take the pad id), xor-folded with the
/// layer's multipliers one look-back at a time - the
/// (g + 1)-gram's value - each n-gram size's value mod its heads' primes, plus their offsets (moex/tools/dsv41/engram_hash.hpp,
/// checked against the release by test_engram_hash.py).  The rows then copied from the mapped tables.
namespace {
constexpr int kEgDirectMin = 64;   // a prompt chunk's rows by O_DIRECT from this many positions on (eg_rows)
bool eg_gguf_direct_on() {         // STRATA_DSV41_EG_GGUF_DIRECT=0: the GGUF table's mapped reads
    static const bool on = [] {
        const char* v = std::getenv("STRATA_DSV41_EG_GGUF_DIRECT");
        return !(v && v[0] == '0');
    }();
    return on;
}
}  // namespace

bool Engine::Impl::eg_split_ok(int T) const {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_DSV41_EG_SPLIT");
        return !(v && v[0] == '0');
    }();
    return on && eg && eg->L >= 2 && eg->mx8 && eg->gfd >= 0 && eg_gguf_direct_on() && T >= kEgDirectMin;
}

bool Engine::Impl::eg_rows(int64_t pos0, int T, uint8_t* buf, int cap, int r0, std::string& err, int j0, int j1) {
    const auto t_eg = clk::now();
    struct Acc {
        std::atomic<int64_t>& us;
        clk::time_point t0;
        ~Acc() { us += (int64_t) (ms_since(t0) * 1000.0); }
    } acc{us_eg_rows, t_eg};
    struct Busy {   // moex (#76): the prompt streams' issuers hold their drive reads meanwhile (Impl::eg_first)
        std::atomic<int>& n;
        explicit Busy(std::atomic<int>& b) : n(b) { n.fetch_add(1, std::memory_order_acq_rel); }
        ~Busy() { n.fetch_sub(1, std::memory_order_acq_rel); }
    } busy{eg_busy};
    const Engram& e = *eg;
    const int M = (int) hp.eg_ngram, H = (int) hp.eg_heads, C = e.cols, D = e.dim, SB = e.dim / 32;
    // 1. the row numbers of every (position, layer, column)
    std::vector<int64_t> rows((size_t) T * e.L * C);
    if (pos0 < 0 || pos0 + T > (int64_t) e.ids.size())   // the positions eg_note recorded (n_ctx of them)
        return fail(err, "engram rows for positions " + std::to_string(pos0) + ".." + std::to_string(pos0 + T - 1) + " past the context");
    for (int t = 0; t < T; ++t) {
        const int64_t p = pos0 + t;
        int64_t slot[16];
        bool blocked = false;
        for (int sh = 0; sh < M; ++sh) {
            const int32_t src = p >= sh ? e.ids[(size_t) (p - sh)] : e.ids[0];
            blocked = blocked || p < sh || src == Engram::kDead;
            slot[sh] = blocked ? hp.eg_pad : src;
        }
        for (int j = 0; j < e.L; ++j) {
            const int64_t* mul = &hp.eg_mul[(size_t) j * M];
            int64_t rolling = slot[0] * mul[0];
            for (int g = 1; g < M; ++g) {
                rolling ^= slot[g] * mul[g];
                for (int h = 0; h < H; ++h) {
                    const int c = (g - 1) * H + h;
                    const int64_t pr = hp.eg_primes[(size_t) j * C + c];
                    int64_t r = rolling % pr;
                    if (r < 0) r += pr;
                    const int64_t row = hp.eg_offsets[(size_t) j * C + c] + r;
                    if (row < 0 || row >= hp.eg_rows[(size_t) j]) return fail(err, "engram row out of range");
                    rows[((size_t) t * e.L + j) * C + c] = row;
                }
            }
        }
    }
    // 2'. a prompt chunk (T >= kEgDirectMin) with the tables open O_DIRECT: every row read from its 4 KB page by kEgThreads
    //    threads at once - the same bytes as the mapped copy below, so the same rows.  The mapped reads' page faults ran at
    //    ~18K a second (a 44K-token prompt waited 118 s for its rows, 2026-10-19); a row never crosses a page (D and SB divide
    //    4096), a table's last page may be short (a short read at its end is fine: the row ends before it)
    constexpr int kEgThreads = 32;
    // 2'mx8. moex (DSV41_PLAN §73): the GGUF's MXFP8 table (STRATA_DSV41_ENGRAM_GGUF, or the set's own) in a prompt chunk - each
    //    layer's distinct rows (an n-gram that repeats in the chunk asks for the same row) read once by kEgThreads threads,
    //    O_DIRECT, a row's rb bytes from its one or two 4 KB pages; then each position's rows split into codes and scales as put()
    //    splits them - the same bytes in the same places.  The mapped reads (MADV_RANDOM: one page a fault, a window of rows asked
    //    ahead) made a cold 16K read wait 5.5-7.4 s for its rows.  STRATA_DSV41_EG_GGUF_DIRECT=0: the mapped reads
    const bool eg_gguf_direct = eg_gguf_direct_on();
    if (j1 < 0 || j1 > e.L) j1 = e.L;
    j0 = std::clamp(j0, 0, j1);
    if (e.mx8 && e.gfd >= 0 && eg_gguf_direct && T >= kEgDirectMin) {
        const size_t rb = e.rb;
        std::vector<std::vector<int64_t>> uq((size_t) e.L);
        std::vector<int64_t> ustart((size_t) e.L + 1, 0);
        for (int j = 0; j < e.L; ++j) {
            auto& u = uq[(size_t) j];
            if (j < j0 || j >= j1) {   // (moex #147: not this call's layers - none of their rows)
                ustart[(size_t) j + 1] = ustart[(size_t) j];
                continue;
            }
            u.resize((size_t) T * C);
            for (int t = 0; t < T; ++t)
                for (int c = 0; c < C; ++c) u[(size_t) t * C + c] = rows[((size_t) t * e.L + j) * C + c];
            std::sort(u.begin(), u.end());
            u.erase(std::unique(u.begin(), u.end()), u.end());
            ustart[(size_t) j + 1] = ustart[(size_t) j] + (int64_t) u.size();
        }
        const int64_t nu = ustart[(size_t) e.L];
        std::vector<uint8_t> ub((size_t) nu * rb);   // the distinct rows, layer after layer
        std::atomic<bool> bad{false};
        {
            std::vector<std::thread> th;
            const int nt = (int) std::min<int64_t>(kEgThreads, std::max<int64_t>(1, nu));
            for (int w = 0; w < nt; ++w)
                th.emplace_back([&, w] {
                    uint8_t* page = static_cast<uint8_t*>(std::aligned_alloc(4096, 8192));   // O_DIRECT's aligned buffer
                    if (!page) {
                        bad = true;
                        return;
                    }
                    int j = 0;
                    for (int64_t k = w; k < nu && !bad.load(std::memory_order_relaxed); k += nt) {
                        while (k >= ustart[(size_t) j + 1]) ++j;
                        const int64_t row = uq[(size_t) j][(size_t) (k - ustart[(size_t) j])];
                        const uint64_t at = e.goff[(size_t) j] + (uint64_t) row * rb, pg = at & ~(uint64_t) 4095;
                        const size_t need = (size_t) (at - pg) + rb, len = need <= 4096 ? 4096 : 8192;
                        const ssize_t got = pread(e.gfd, page, len, (off_t) pg);
                        if (got < (ssize_t) need) {   // (a short read past the end of the file is fine: the row ends before it)
                            bad = true;
                            break;
                        }
                        std::memcpy(ub.data() + (size_t) k * rb, page + (at - pg), rb);
                    }
                    std::free(page);
                });
            for (auto& x : th) x.join();
        }
        if (!bad) {
            std::vector<std::thread> th;
            const int64_t nrow = (int64_t) T * e.L;
            const int nt = (int) std::min<int64_t>(kEgThreads, nrow);
            for (int w = 0; w < nt; ++w)
                th.emplace_back([&, w] {
                    for (int64_t q = w; q < nrow; q += nt) {
                        const int t = (int) (q / e.L), j = (int) (q % e.L);
                        if (j < j0 || j >= j1) continue;   // (moex #147)
                        const auto& u = uq[(size_t) j];
                        uint8_t* base = buf + (size_t) j * cap * e.row_bytes();
                        for (int c = 0; c < C; ++c) {
                            const int64_t row = rows[((size_t) t * e.L + j) * C + c];
                            const size_t i = (size_t) (std::lower_bound(u.begin(), u.end(), row) - u.begin());
                            const uint8_t* src = ub.data() + ((size_t) ustart[(size_t) j] + i) * rb;
                            uint8_t* wd = base + ((size_t) (r0 + t) * C + c) * D;
                            uint8_t* sd = base + (size_t) cap * C * D + ((size_t) (r0 + t) * C + c) * SB;
                            if (e.iq4nl) {   // (put()'s copy: the IQ4_NL row as it is)
                                std::memcpy(wd, src, rb);
                                continue;
                            }
                            for (int b = 0; b < SB; ++b) {   // (put()'s split)
                                sd[b] = src[(size_t) b * 33];
                                std::memcpy(wd + (size_t) b * 32, src + (size_t) b * 33 + 1, 32);
                            }
                        }
                    }
                });
            for (auto& x : th) x.join();
            eg_rows_asked += (uint64_t) T * (j1 - j0) * C;
            eg_rows_read += (uint64_t) nu;
            return true;
        }
        static std::atomic<bool> said_g{false};
        if (!said_g.exchange(true)) fprintf(stderr, "strata-x: an O_DIRECT read of the GGUF engram table failed: the mapped reads take over\n");
    }
    // 2''. moex (DSV41_BACKLOG #59, opt-in STRATA_DSV41_EG_READER=1): a prompt chunk through the n-gram reader - per layer one
    //     ticket for its rows and one for their scales, each row landing where the mapped copy puts it (a layer's T x C rows,
    //     then their scales, are consecutive in the buffer).  STRATA_DSV41_EG_VERIFY=1 compares every row with the mapped
    //     tables.  A failed read: the mapped copy below writes every row again
    if (T >= kEgDirectMin && !e.rd_w.empty()) {
        std::vector<std::vector<uint32_t>> rr((size_t) e.L, std::vector<uint32_t>((size_t) T * C));
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < e.L; ++j)
                for (int c = 0; c < C; ++c) rr[(size_t) j][(size_t) t * C + c] = (uint32_t) rows[((size_t) t * e.L + j) * C + c];
        std::vector<strata::ngram::PleReader::Ticket> tw((size_t) e.L), ts((size_t) e.L);
        for (int j = 0; j < e.L; ++j) {
            uint8_t* base = buf + (size_t) j * cap * e.row_bytes();
            tw[(size_t) j] = e.rd_w[(size_t) j]->issue(rr[(size_t) j].data(), rr[(size_t) j].size(), base + (size_t) r0 * C * D);
            ts[(size_t) j] = e.rd_s[(size_t) j]->issue(rr[(size_t) j].data(), rr[(size_t) j].size(),
                                                       base + (size_t) cap * C * D + (size_t) r0 * C * SB);
        }
        bool ok = true;
        std::string re;
        for (int j = 0; j < e.L; ++j) {   // every ticket collected, whatever failed: the buffer is the caller's after this
            ok = e.rd_w[(size_t) j]->collect(tw[(size_t) j], re) && ok;
            ok = e.rd_s[(size_t) j]->collect(ts[(size_t) j], re) && ok;
        }
        if (ok) {
            static const bool verify = [] {
                const char* v = std::getenv("STRATA_DSV41_EG_VERIFY");
                return v && v[0] == '1';
            }();
            if (verify) {
                size_t bad = 0;
                for (int t = 0; t < T; ++t)
                    for (int j = 0; j < e.L; ++j) {
                        const uint8_t* base = buf + (size_t) j * cap * e.row_bytes();
                        for (int c = 0; c < C; ++c) {
                            const int64_t row = rows[((size_t) t * e.L + j) * C + c];
                            bad += std::memcmp(base + ((size_t) (r0 + t) * C + c) * D, e.w[(size_t) j] + (size_t) row * D, (size_t) D) != 0 ||
                                   std::memcmp(base + (size_t) cap * C * D + ((size_t) (r0 + t) * C + c) * SB,
                                               e.s[(size_t) j] + (size_t) row * SB, (size_t) SB) != 0;
                        }
                    }
                fprintf(stderr, "strata-x: engram reader check (STRATA_DSV41_EG_VERIFY): positions %lld..%lld, %zu of %lld rows "
                                "unlike the mapped tables\n",
                        (long long) pos0, (long long) (pos0 + T - 1), bad, (long long) T * e.L * C);
            }
            return true;
        }
        static std::atomic<bool> said_rd{false};
        if (!said_rd.exchange(true))
            fprintf(stderr, "strata-x: the engram tables' n-gram reader failed (%s): the mapped reads take over\n", re.c_str());
    }
    bool direct = T >= kEgDirectMin;
    for (int j = 0; j < e.L && direct; ++j) direct = e.wfd[(size_t) j] >= 0 && e.sfd[(size_t) j] >= 0;
    // moex (DSV41_BACKLOG #103), opt-in STRATA_DSV41_EG_DEDUP=1: the direct path with each layer's rows of the chunk read once
    // (sorted and deduplicated - an n-gram that repeats in the chunk asks for the same row; dsx's prefill.rs:723) and the
    // scales copied from the mapped tables instead of read O_DIRECT: those pages stay in the page cache (EG_SCALES_PRELOAD
    // fills it), where every O_DIRECT read took the drive again - half of the 96 requests a position was the scales, each
    // 4 KB page re-read for every chunk.  The same bytes in the same places: the same rows
    static const bool eg_dedup = [] {
        const char* v = std::getenv("STRATA_DSV41_EG_DEDUP");
        return v && v[0] == '1';
    }();
    if (direct && eg_dedup) {
        std::vector<std::vector<int64_t>> uq((size_t) e.L);
        std::vector<std::vector<uint8_t>> ub((size_t) e.L);
        std::vector<int64_t> ustart((size_t) e.L + 1, 0);
        for (int j = 0; j < e.L; ++j) {
            auto& u = uq[(size_t) j];
            u.resize((size_t) T * C);
            for (int t = 0; t < T; ++t)
                for (int c = 0; c < C; ++c) u[(size_t) t * C + c] = rows[((size_t) t * e.L + j) * C + c];
            std::sort(u.begin(), u.end());
            u.erase(std::unique(u.begin(), u.end()), u.end());
            ub[(size_t) j].resize(u.size() * (size_t) D);
            ustart[(size_t) j + 1] = ustart[(size_t) j] + (int64_t) u.size();
        }
        const int64_t nu = ustart[(size_t) e.L];
        std::atomic<bool> bad{false};
        {   // the distinct rows, each from its 4 KB page
            std::vector<std::thread> th;
            const int nt = (int) std::min<int64_t>(kEgThreads, nu);
            for (int w = 0; w < nt; ++w)
                th.emplace_back([&, w] {
                    uint8_t* page = static_cast<uint8_t*>(std::aligned_alloc(4096, 4096));
                    if (!page) {
                        bad = true;
                        return;
                    }
                    for (int64_t k = w; k < nu && !bad.load(std::memory_order_relaxed); k += nt) {
                        int j = 0;
                        while (k >= ustart[(size_t) j + 1]) ++j;
                        const int64_t i = k - ustart[(size_t) j], row = uq[(size_t) j][(size_t) i];
                        const uint64_t at = (uint64_t) row * D, pg = at & ~(uint64_t) 4095;
                        const ssize_t got = pread(e.wfd[(size_t) j], page, 4096, (off_t) pg);
                        if (got < (ssize_t) (at - pg + D)) {
                            bad = true;
                            break;
                        }
                        std::memcpy(ub[(size_t) j].data() + (size_t) i * D, page + (at - pg), (size_t) D);
                    }
                    std::free(page);
                });
            for (auto& x : th) x.join();
        }
        if (!bad) {   // each position's rows from the distinct ones, its scales from the mapped tables (in parallel: a scale page
                      // not in the page cache faults in once)
            std::vector<std::thread> th;
            const int64_t nrow = (int64_t) T * e.L;
            const int nt = (int) std::min<int64_t>(kEgThreads, nrow);
            for (int w = 0; w < nt; ++w)
                th.emplace_back([&, w] {
                    for (int64_t q = w; q < nrow; q += nt) {
                        const int t = (int) (q / e.L), j = (int) (q % e.L);
                        const auto& u = uq[(size_t) j];
                        uint8_t* base = buf + (size_t) j * cap * e.row_bytes();
                        for (int c = 0; c < C; ++c) {
                            const int64_t row = rows[((size_t) t * e.L + j) * C + c];
                            const size_t i = (size_t) (std::lower_bound(u.begin(), u.end(), row) - u.begin());
                            std::memcpy(base + ((size_t) (r0 + t) * C + c) * D, ub[(size_t) j].data() + i * D, (size_t) D);
                            std::memcpy(base + (size_t) cap * C * D + ((size_t) (r0 + t) * C + c) * SB, e.s[(size_t) j] + (size_t) row * SB,
                                        (size_t) SB);
                        }
                    }
                });
            for (auto& x : th) x.join();
            eg_rows_asked += (uint64_t) T * e.L * C;
            eg_rows_read += (uint64_t) nu;
            return true;
        }
        static std::atomic<bool> said_dd{false};
        if (!said_dd.exchange(true)) fprintf(stderr, "strata-x: an O_DIRECT read of the engram tables failed: the mapped reads take over\n");
        direct = false;   // (the mapped copy below writes every row again)
    }
    if (direct) {
        const int64_t per = (int64_t) e.L * C * 2;   // reads a position takes: each layer's C rows and their C scales
        const int64_t total = (int64_t) T * per;
        std::atomic<bool> bad{false};
        std::vector<std::thread> th;
        const int nt = (int) std::min<int64_t>(kEgThreads, total);
        for (int w = 0; w < nt; ++w)
            th.emplace_back([&, w] {
                uint8_t* page = static_cast<uint8_t*>(std::aligned_alloc(4096, 4096));   // O_DIRECT's aligned buffer
                if (!page) {
                    bad = true;
                    return;
                }
                for (int64_t k = w; k < total && !bad.load(std::memory_order_relaxed); k += nt) {
                    const int t = (int) (k / per);
                    const int64_t rem = k % per;
                    const int j = (int) (rem / (C * 2)), c = (int) ((rem / 2) % C);
                    const bool scale = (rem & 1) != 0;
                    const int64_t row = rows[((size_t) t * e.L + j) * C + c];
                    const size_t len = scale ? (size_t) SB : (size_t) D;
                    const uint64_t at = (uint64_t) row * len, pg = at & ~(uint64_t) 4095;
                    const ssize_t got = pread(scale ? e.sfd[(size_t) j] : e.wfd[(size_t) j], page, 4096, (off_t) pg);
                    if (got < (ssize_t) (at - pg + len)) {
                        bad = true;
                        break;
                    }
                    uint8_t* base = buf + (size_t) j * cap * e.row_bytes();
                    uint8_t* dst = scale ? base + (size_t) cap * C * D + ((size_t) (r0 + t) * C + c) * SB
                                         : base + ((size_t) (r0 + t) * C + c) * D;
                    std::memcpy(dst, page + (at - pg), len);
                }
                std::free(page);
            });
        for (auto& x : th) x.join();
        if (!bad) return true;
        // a failed O_DIRECT read: the mapped copy below writes every row again
        static std::atomic<bool> said{false};
        if (!said.exchange(true)) fprintf(stderr, "strata-x: an O_DIRECT read of the engram tables failed: the mapped reads take over\n");
    }
    // 2. every row's pages asked for at once (the mappings are MADV_RANDOM: a fault reads only its page, one at a time), so
    //    the NVMe serves them in parallel; rows already in the page cache cost nothing here
    const uintptr_t pg = (uintptr_t) sysconf(_SC_PAGESIZE);
    auto want = [&](const uint8_t* a, size_t len) {
        const uintptr_t b = (uintptr_t) a & ~(pg - 1), end = ((uintptr_t) a + len + pg - 1) & ~(pg - 1);
        madvise((void*) b, end - b, MADV_WILLNEED);
    };
    // a row's pages asked for, and the row copied to wd (its D code bytes) and sd (its SB scales): the pack's tables hold
    // the two apart; a GGUF's MXFP8 table (STRATA_DSV41_ENGRAM_GGUF) holds them together, SB blocks of [e8m0, 32 codes] -
    // split here, so the same bytes land in the same places
    auto ask = [&](int j, int64_t row) {
        if (e.mx8) {
            want(e.w[(size_t) j] + (size_t) row * e.rb, e.rb);
        } else {
            want(e.w[(size_t) j] + (size_t) row * D, (size_t) D);
            want(e.s[(size_t) j] + (size_t) row * SB, (size_t) SB);
        }
    };
    auto put = [&](int j, int64_t row, uint8_t* wd, uint8_t* sd) {
        if (e.iq4nl) {   // moex: the IQ4_NL row as it is, into the codes' place (Stage::engram dequantizes it)
            std::memcpy(wd, e.w[(size_t) j] + (size_t) row * e.rb, e.rb);
        } else if (e.mx8) {
            const uint8_t* src = e.w[(size_t) j] + (size_t) row * e.rb;
            for (int b = 0; b < SB; ++b) {
                sd[b] = src[(size_t) b * 33];
                std::memcpy(wd + (size_t) b * 32, src + (size_t) b * 33 + 1, 32);
            }
        } else {
            std::memcpy(wd, e.w[(size_t) j] + (size_t) row * D, (size_t) D);
            std::memcpy(sd, e.s[(size_t) j] + (size_t) row * SB, (size_t) SB);
        }
    };
    // 2'''. moex (DSV41_PLAN §18), opt-in STRATA_DSV41_EG_WINDOW=W: the rows asked for at most W ahead of the copies, so the
    //    drive's queue holds at most W rows' pages (all of a chunk's at once - 8K tokens, 786K pages - filled it to ~1,000
    //    requests, and the drive's slow mode served them at ~1,000 a second, 2026-10-21); the drive guard's level lowers W (its
    //    floor: 1, the copies' faults one page at a time).  The same rows copied in the same order: the same bytes.  A cold
    //    8K-token chunk with the scales cached: 7.41 s at 1,024 rows against 9.61 s all at once (eg_fetch_bench.py)
    static const int eg_window = [] {
        const char* v = std::getenv("STRATA_DSV41_EG_WINDOW");
        return v ? std::max(0, std::atoi(v)) : 0;
    }();
    static const int eg_decode_rows = [] {
        const char* v = std::getenv("STRATA_DSV41_EG_DECODE_ROWS");
        return v ? std::max(0, std::atoi(v)) : 32;
    }();
    if (const dsv41::DriveGuard* gd = dsv41::DriveGuard::get(); eg_window > 0 || gd) {
        // rows[] runs in the copies' order (position t, engram layer j, column c): row q is rows[q], of layer (q / C) % L
        const int64_t nrow = (int64_t) T * e.L * C;
        int64_t asked = 0;   // rows asked for so far
        for (int64_t q = 0; q < nrow; ++q) {
            int64_t W = eg_window > 0 ? eg_window : INT64_MAX;
            if (gd) W = std::min<int64_t>(W, gd->eg_rows());
            // moex (DSV41_PLAN §75, DSV41_BACKLOG #147): a decode window's rows (T <= 8: 48 rows a token) below the guard's
            // level 3 - 32 pages in flight, not its 1 / 4 / 16.  The guard's limit was set for a prompt chunk's 100K+ rows; a
            // token's 48 at the floor were 48 faults one after the other (§71: ~3 ms each in an episode - 144 ms of a 41 ms
            // token; §64: an episode serves 4 KB reads ~5.4K a second at 32 in flight, ~1K one at a time).  The same rows in
            // the same order: the same bytes.  STRATA_DSV41_EG_DECODE_ROWS=N: another count (0: the guard's own)
            if (T <= 8 && W < eg_decode_rows) W = eg_decode_rows;
            // moex (DSV41_PLAN §64): a stage blocked on these rows with the record reads held (eg_urgent) - the drive is the
            // fetch's alone: 32 pages in flight (a slow-mode episode's 4 KB reads ~5.4K a second so, ~1K one at a time)
            if (W < 32 && eg_urgent && eg_need.load(std::memory_order_relaxed) > 0) W = std::min<int64_t>(eg_window > 0 ? eg_window : INT64_MAX, 32);
            if (W > 1) {
                for (const int64_t lim = std::min<int64_t>(nrow, q + W); asked < lim; ++asked)
                    ask((int) ((asked / C) % e.L), rows[(size_t) asked]);
            } else {
                asked = std::max(asked, q + 1);   // the copy below faults its two pages, one at a time
            }
            const int j = (int) ((q / C) % e.L), c = (int) (q % C);
            const int64_t t = q / ((int64_t) e.L * C), row = rows[(size_t) q];
            uint8_t* base = buf + (size_t) j * cap * e.row_bytes();
            put(j, row, base + ((size_t) (r0 + t) * C + c) * D, base + (size_t) cap * C * D + ((size_t) (r0 + t) * C + c) * SB);
        }
        return true;
    }
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < e.L; ++j)
            for (int c = 0; c < C; ++c) ask(j, rows[((size_t) t * e.L + j) * C + c]);
    // 3. the copies (per engram layer j: the weights of cap x cols rows, then their scales)
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < e.L; ++j) {
            uint8_t* wdst = buf + (size_t) j * cap * e.row_bytes() + (size_t) (r0 + t) * C * D;
            uint8_t* sdst = buf + (size_t) j * cap * e.row_bytes() + (size_t) cap * C * D + (size_t) (r0 + t) * C * SB;
            for (int c = 0; c < C; ++c) put(j, rows[((size_t) t * e.L + j) * C + c], wdst + (size_t) c * D, sdst + (size_t) c * SB);
        }
    return true;
}

void Engine::Impl::eg_help_arm(Stage& g) {
    EgHelper& h = *eg_help;
    std::lock_guard<std::mutex> lk(h.mu);
    const uint32_t v = ++g.host_egf;   // this window's wait passes at v (release_gpu_waits takes the device's count after a failure)
    if (h.done_seq.load(std::memory_order_acquire) >= h.seq) {   // the fetch has ended: its rows are in, or it failed (no word)
        if (h.ok) {
            std::atomic_thread_fence(std::memory_order_release);
            *g.db_egf = v;
        }
        h.arm = false;
    } else {
        h.ring_ptr = g.db_egf;
        h.ring_val = v;
        h.arm_seq = h.seq;
        h.arm = true;
    }
}

void Engine::Impl::eg_help_post(int64_t pos0, int T, uint8_t* buf, int cap, int r0) {
    EgHelper& h = *eg_help;
    {
        std::lock_guard<std::mutex> lk(h.mu);
        h.pos0 = pos0;
        h.T = T;
        h.buf = buf;
        h.cap = cap;
        h.r0 = r0;
        ++h.seq;
    }
    h.cv.notify_one();
}

bool Engine::Impl::eg_help_wait(std::string& err) {
    if (!eg_help) return true;
    EgHelper& h = *eg_help;
    uint64_t want;
    {
        std::lock_guard<std::mutex> lk(h.mu);
        want = h.seq;
    }
    while (h.done_seq.load(std::memory_order_acquire) < want) std::this_thread::yield();
    if (!h.ok) return fail(err, h.err);
    return true;
}

void Engine::Impl::eg_help_stop() {
    if (!eg_help) return;
    {
        std::lock_guard<std::mutex> lk(eg_help->mu);
        eg_help->stop = true;
    }
    eg_help->cv.notify_one();
    if (eg_help->th.joinable()) eg_help->th.join();
    eg_help.reset();
}

bool Stage::eg_pr_grow(int rows, std::string& err) {
    if (rows <= eg_pr_cap) return true;
    cudaSetDevice(dev);
    cudaStreamSynchronize(s);   // the old buffer may still be read
    if (eg_pr_h) cudaFreeHost(eg_pr_h);
    eg_pr_h = eg_pr_d = nullptr;
    eg_pr_cap = 0;
    const size_t bytes = (size_t) m.eg->L * rows * m.eg->row_bytes();
    if (cudaHostAlloc((void**) &eg_pr_h, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &eg_pr_d, eg_pr_h, 0) != cudaSuccess)
        return m.fail(err, "the engram's prompt buffer");
    eg_pr_cap = rows;
    return true;
}

bool Stage::eg_nx_grow(int rows, std::string& err) {   // moex (DSV41_BACKLOG #51): the second buffer, as eg_pr_grow
    if (rows <= eg_nx_cap) return true;
    if (eg_nx.valid()) eg_nx.get();   // a fetch into the old buffer is waited out first
    cudaSetDevice(dev);
    cudaStreamSynchronize(s);
    if (eg_nx_h) cudaFreeHost(eg_nx_h);
    eg_nx_h = eg_nx_d = nullptr;
    eg_nx_cap = 0;
    const size_t bytes = (size_t) m.eg->L * rows * m.eg->row_bytes();
    if (cudaHostAlloc((void**) &eg_nx_h, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &eg_nx_d, eg_nx_h, 0) != cudaSuccess)
        return m.fail(err, "the engram's second prompt buffer");
    eg_nx_cap = rows;
    return true;
}

/// model.py Engram.forward before layer il's mixes, rows [r0, r0 + T) of the streams: the rows dequantized (into datt, free
/// before the attention), wkv (into dq) as [key (hc, n) | value (n)], then the gate and the write (dsv41_engram_gate).
void Stage::engram_rows(int il, int r0, int T, const uint8_t* buf_d, int cap, float* att, float* kv) {
    const int j = m.eg_index(il);
    const Engine::Impl::Engram& e = *m.eg;
    const int n = (int) hp.n_embd, hc = (int) hp.hc, C = e.cols, D = e.dim, CD = C * D;
    const uint8_t* w = buf_d + (size_t) j * cap * e.row_bytes() + (size_t) (eg_rbase + r0) * CD;
    const uint8_t* sc = buf_d + (size_t) j * cap * e.row_bytes() + (size_t) cap * CD + (size_t) (eg_rbase + r0) * C * (D / 32);
    const Layer& l = L[(size_t) il];
    if (e.iq4nl) dsv41_engram_dequant_iq4nl(w, (size_t) D, att, (int64_t) T * C, D, s);   // (moex: each row in its D-byte place)
    else dsv41_engram_dequant(w, sc, att, (int64_t) T * C, D, s);
    probe("eg_emb", il, att, CD);
    if (pj_needs_q(l.eg_wkv)) quant(att, xq, CD, T);   // (moex §75: pj's only reader here)
    pj(l.eg_wkv, att, xq, kv, T);
    probe("eg_kv", il, kv, (int64_t) (hc + 1) * n);
}

void Stage::engram(int il, int r0, int T, const uint8_t* buf_d, int cap, const float* kv_pre) {
    const int j = m.eg_index(il);
    const Engine::Impl::Engram& e = *m.eg;
    const int n = (int) hp.n_embd, hc = (int) hp.hc;
    float* Rr = R + (size_t) r0 * hc * n;
    if (!kv_pre) engram_rows(il, r0, T, buf_d, cap, datt, dq);
    dsv41_engram_gate(Rr, kv_pre ? kv_pre : dq, e.qk[(size_t) j], rms_eps(), n, hc, T, s);
    probe("eg_out", il, Rr, n, hc);
}

// ------------------------------------------------------------------ the attention half

void Stage::v41_pre_reset(int T) { dsv41_pre_onehot(cpre, T, s); }

namespace {
float half_bits(uint16_t h) {   // IEEE half -> float (the index-key cache's storage, __float2half_rn)
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16, ex = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t f;
    if (ex == 0) {
        if (man == 0) f = sign;
        else {   // subnormal: normalize
            int e = -1;
            uint32_t mm = man;
            do { ++e; mm <<= 1; } while (!(mm & 0x400u));
            f = sign | ((uint32_t) (127 - 15 - e) << 23) | ((mm & 0x3FFu) << 13);
        }
    } else if (ex == 31) f = sign | 0x7F800000u | (man << 13);
    else f = sign | ((ex + 127 - 15) << 23) | (man << 13);
    float out;
    std::memcpy(&out, &f, sizeof out);
    return out;
}
}  // namespace

/// The deep check of the selection (the audit of 2026-10-19, #36b: the candidate filter and the top-k past 16K positions had
/// never been compared with model.py): for one decode row at pos0, what the index owner `il` selected from - its source's index
/// keys, rows [0, n_vis) of the FP16 cache; its scores (after the candidate mask on a narrowed layer); its candidate bits as
/// int32 words - so moex/tools/dsv41/check_selection.py can redo select_candidate_blocks and the top-k in numpy on the engine's
/// own inputs.  Only with a recorder, and only the records its --match names (the cache is ~n_vis x 128 values).
void Stage::dump_selection(int il, int ratio, const uint16_t* lcache, int srows, const uint8_t* cand, int cbytes, int64_t pos0) {
    const int ID = (int) hp.idx_dim;
    const int64_t n_vis = std::min<int64_t>((pos0 + 1) / std::max(1, ratio), srows);
    if (n_vis <= 0) return;
    cudaStreamSynchronize(s);
    if (m.rec->wants("idx_kc", il) && kv_pack()) {   // (moex, #104) the packed keys as the kernels read them
        const size_t rb = dsv41_pack_row_bytes(ID, 32);
        std::vector<uint8_t> pk((size_t) n_vis * rb);
        cudaMemcpy(pk.data(), lcache, pk.size(), cudaMemcpyDeviceToHost);
        std::vector<float> f((size_t) (n_vis * ID));
        for (int64_t j = 0; j < n_vis; ++j)
            for (int d = 0; d < ID; ++d) {
                const uint8_t* row = pk.data() + (size_t) j * rb;
                const uint32_t c = (row[d / 2] >> (4 * (d & 1))) & 15u, mg = c & 7u;
                const float a = mg < 4u ? 0.5f * (float) mg : mg == 7u ? 6.0f : (float) (mg - 2u);
                uint32_t sb = (uint32_t) row[ID / 2 + d / 32] << 23;
                float sc;
                std::memcpy(&sc, &sb, sizeof sc);
                f[(size_t) j * ID + d] = __half2float(__float2half_rn(((c & 8u) ? -a : a) * sc));
            }
        m.rec->f32("idx_kc", il, f.data(), ID, n_vis);
    } else if (m.rec->wants("idx_kc", il)) {
        std::vector<uint16_t> hv((size_t) (n_vis * ID));
        cudaMemcpy(hv.data(), lcache, hv.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        std::vector<float> f(hv.size());
        for (size_t i = 0; i < hv.size(); ++i) f[i] = half_bits(hv[i]);
        m.rec->f32("idx_kc", il, f.data(), ID, n_vis);
    }
    if (m.rec->wants("idx_score", il)) {
        std::vector<float> sc((size_t) n_vis);
        cudaMemcpy(sc.data(), dscore, sc.size() * sizeof(float), cudaMemcpyDeviceToHost);   // row 0 (T == 1)
        m.rec->f32("idx_score", il, sc.data(), n_vis);
    }
    if (m.rec->wants("idx_cand", il) && cand) {
        std::vector<int32_t> w((size_t) (cbytes + 3) / 4, 0);
        cudaMemcpy(w.data(), cand, (size_t) cbytes, cudaMemcpyDeviceToHost);
        m.rec->i32("idx_cand", il, w.data(), (int64_t) w.size());
    }
}

/// The read-ahead's guess (DSV41_PLAN 5 C.3, measured on the oracle: 65% of the next layer's experts at 65% precision): this
/// layer's FFN rows normed with the next layer's FFN norm, the next layer's router and its top-k with its bias - the same rule
/// as its real routing - and the ids copied into mapped memory before the doorbell, so the host reads them with the routing.
/// moex (DSV41_PLAN §23): with the route bias (STRATA_DSV41_ROUTE_BIAS) its penalty too, as the decode's routing has it -
/// without it the guess was the release's routing, whose SSD experts the biased routing mostly passes over
void Stage::ahead_compute(int il, int T) {
    const int t = il + m.ahead_d;   // (§27: the layer d ahead)
    const Layer& nl = L[(size_t) t];
    const int n = (int) hp.n_embd, E = (int) hp.E_of(t), k = (int) hp.k_of(t);
    rms_norm(cur, nl.ffn_norm.f(), apred_x, T, n, rms_eps(), s);
    gemv_dt(nl.gate_inp, apred_x, apred_logits, n, E, T);
    dsv4_route(apred_logits, nl.probs_b ? nl.probs_b.f() : nullptr, nullptr, E, k, hp.ew_scale, apred_ids, apred_wts, T, s,
               m.route_pen_for(t), true);
}

void Stage::ahead_predict(int il, int T) {
    ahead_compute(il, T);
    store_mapped(reinterpret_cast<const float*>(apred_ids), reinterpret_cast<float*>(dd_pred), (size_t) T * hp.k_of(il + m.ahead_d), s);
}

/// DeepSeek-V4.1's attention over T tokens at positions pos0.. (model.py Attention.forward):
///   qr = RMS(q_a . x);  q = RoPE(q_b . qr)                         64 heads of 512, RoPE on dims [448, 512), no head norm
///   kv = FP8(RoPE(RMS(attn_kv . x)))  -> the raw ring              the one shared K = V row (act_quant of the whole row)
///   a KV source (ratio 2 / 1): the compressor's latent - per block of `ratio` positions the gated softmax pool of
///     compressor_kv . x (ratio 2) or the projection itself (ratio 1), RMS-normed - then
///       the index key  FP4_32(RoPE(RMS(attn_k . latent)))    into the source's index rows   (RoPE on dims [64, 128))
///       the row        FP4_16(RoPE(latent))                   into the source's compressed rows
///     both at the block's first position (row b stands for positions [b r, b r + r))
///   an index source: q_i = FP4_32(RoPE(attn_q_b . qr)) (32 heads), w = proj . x / sqrt(128 * 32); score[j] = sum_h relu(q_i,h .
///     k_j) w_h over the KV source's visible index rows; layer cand_src keeps its best blocks, the later sources score only
///     inside them; the top 512 rows (the list the layers up to the next index source reuse)
///   attention of each head over the raw window + the listed compressed rows of the KV source, with the head's sink
///   out = wo_b . concat_g(wo_a[g] . FP8(de-RoPE(attention))[heads 8g .. 8g + 7])
/// The top-k list and the candidates are kept per row of the active scratch (r0: this batch's first row), so the prompt path's
/// sub-batches of a later layer read their own rows' lists.
void Stage::dsv41(int il, int64_t pos0, int T, const float* in, float* out_r) {
    const Layer& l = L[(size_t) il];
    const int n = (int) hp.n_embd, H = (int) hp.n_head, D = (int) hp.hd_k, Lq = (int) hp.q_lora, nr = (int) hp.n_rot;
    const int IH = (int) hp.idx_heads, ID = (int) hp.idx_dim, G = (int) hp.o_groups, OL = (int) hp.o_lora;
    const int ratio = hp.cratio[(size_t) il];
    const float eps = rms_eps();
    const Dsv4Rope rp = ratio == 0 ? dsv4_rope_p0(hp.rope_base) : m.rope_pc;   // the ratio-0 layers: plain RoPE, no YaRN
    const int roff = D - nr;
    if (in < cur || (in - cur) % n != 0) throw std::runtime_error("dsv41: the attention's input is not a row of the scratch's cur");
    const int64_t r0 = (in - cur) / n;
    const int srows = m.dsv41_score_rows(), cbytes = dsv41_cand_bytes(srows, std::max(1, hp.cand_bsize));
    int32_t* sel = dsel + r0 * hp.idx_topk;
    uint8_t* cand = dcand ? dcand + r0 * cbytes : nullptr;   // (§59 P3: none on a stage before the candidates' source)
    if (pj_needs_q(l.q_a) || pj_needs_q(l.wkv)) quant(in, xq, n, T);   // (moex §75: its only readers)
    // moex (DSV41_PLAN §39), opt-in STRATA_DSV41_GRAPH_FORK=1: in a captured decode graph the KV row and the compressor (they
    // read the input and its quantized copy, and write only their own rows and caches) run on a second stream beside the
    // query's projections; the indexer and the attention, which read all three, after the join.  The same kernels on the
    // same inputs: the same bits
    static const bool fork_env = [] {
        const char* e = std::getenv("STRATA_DSV41_GRAPH_FORK");
        return e && e[0] == '1';
    }();
    const bool fork = fork_env && gpos != nullptr && s2 != nullptr && !m.rec;
    const cudaStream_t s_main = s;
    if (fork) {
        cudaEventRecord(fork_ev, s);
        cudaStreamWaitEvent(s2, fork_ev, 0);
        s = s2;
    } else {
        // the query (the default order: query, KV row, compressor)
        pj(l.q_a, in, xq, qr, T);
        probe("qr", il, qr, Lq);
        rms_norm(qr, l.q_a_norm.f(), qr, T, Lq, eps, s);
        probe("qr_norm", il, qr, Lq);
        if (pj_needs_q(l.q_b)) quant(qr, xq2, Lq, T);
        pj(l.q_b, qr, xq2, dq, T);
        dsv4_rope(dq, dq, T, H, D, roff, rp, false, pos0, s, gpos);
        probe("q", il, dq, D, H);
    }
    // the shared K = V row into the raw ring
    pj(l.wkv, in, xq, dkv, T);
    rms_norm(dkv, l.kv_a_norm.f(), dkv, T, D, eps, s);
    dsv4_rope(dkv, dkv, T, 1, D, roff, rp, false, pos0, s, gpos);
    dsv41_fq_fp8(dkv, T, D, s);
    probe("kv", il, dkv, D);
    dsv4_raw_store(dkv, l.raw, (int) l.raw_rows, pos0, T, s, gpos);
    const Layer* src = ratio > 0 ? &L[(size_t) hp.kv_src[(size_t) il]] : nullptr;
    struct ForkJoin {   // (moex, §39) the query on the main stream, then the join, once the second stream's part is queued
        Stage& g;
        bool on;
        const cudaStream_t main;
        std::function<void()> query;
        void done() {
            if (!on) return;
            on = false;
            cudaEventRecord(g.join_ev, g.s);
            g.s = main;
            query();
            cudaStreamWaitEvent(g.s, g.join_ev, 0);
        }
        ~ForkJoin() {
            if (on) g.s = main;   // (an exception: the capture fails anyway)
        }
    } fj{*this, fork, s_main, [&] {
        pj(l.q_a, in, xq, qr, T);
        rms_norm(qr, l.q_a_norm.f(), qr, T, Lq, eps, s);
        if (pj_needs_q(l.q_b)) quant(qr, xq2, Lq, T);
        pj(l.q_b, qr, xq2, dq, T);
        dsv4_rope(dq, dq, T, H, D, roff, rp, false, pos0, s, gpos);
    }};
    if (ratio > 0 && hp.kv_owner(il)) {
        const int S = ratio == 2 ? m.dsv4_ring(2) : 0;
        gemv_dt(l.c_kv, in, dakv, n, D, T);
        if (ratio == 2) gemv_dt(l.c_gate, in, dgs, n, D, T);
        probe("cmp_kv", il, dakv, D);
        dsv41_compress(ratio, dakv, ratio == 2 ? dgs : nullptr, l.c_ring_kv, l.c_ring_sc, S, l.c_norm.f(), eps, dlat, pos0, T, s, gpos);
        if (ratio == 2) dsv41_compress_persist(dakv, dgs, l.c_ring_kv, l.c_ring_sc, S, pos0, T, s, gpos);
        probe("cmp_latent", il, dlat, D);
        // the index key from the latent before its RoPE (model.py: the indexer runs before the cache is written)
        gemv_dt(l.idx_k, dlat, dik, D, ID, T);
        rms_norm(dik, l.idx_k_norm.f(), dik, T, ID, eps, s);
        dsv4_rope_shift(dik, dik, T, 1, ID, ID - nr, m.rope_pc, false, pos0, 1 - ratio, s, gpos);
        if (kv_pack())   // (moex, #104) the rounding, and the completed rows packed into the cache
            dsv41_fq_fp4_pack(dik, T, ID, 32, false, reinterpret_cast<uint8_t*>(l.lcache), (int) l.c_rows, ratio, pos0, s, gpos);
        else
            dsv41_fq_fp4(dik, T, ID, 32, false, s);
        probe("idx_k", il, dik, ID);
        if (!kv_pack()) dsv41_store_rows(ratio, dik, l.lcache, ID, (int) l.c_rows, pos0, T, s, gpos);
        dsv4_rope_shift(dlat, dlat, T, 1, D, roff, m.rope_pc, false, pos0, 1 - ratio, s, gpos);
        if (kv_pack())
            dsv41_fq_fp4_pack(dlat, T, D, 16, true, reinterpret_cast<uint8_t*>(l.ccache), (int) l.c_rows, ratio, pos0, s, gpos);
        else
            dsv41_fq_fp4(dlat, T, D, 16, true, s);
        probe("cmp_row", il, dlat, D);
        if (!kv_pack()) dsv41_store_rows(ratio, dlat, l.ccache, D, (int) l.c_rows, pos0, T, s, gpos);
    }
    fj.done();   // (moex, §39: the query, then the join)
    if (ratio > 0 && hp.idx_owner(il)) {
        // q's IH heads in datt (free until the attention), then padded with zero heads (and zero weights) to the kernels' 64
        constexpr int kIH = 64;
        gemv_dt(l.idx_q_b, qr, datt, Lq, IH * ID, T);
        dsv4_rope(datt, datt, T, IH, ID, ID - nr, m.rope_pc, false, pos0, s, gpos);
        dsv41_fq_fp4(datt, (int64_t) T * IH, ID, 32, false, s);
        probe("idx_q", il, datt, ID, IH);
        gemv_dt(l.idx_proj, in, dlakv, n, IH, T);
        scale_f32(dlakv, T * IH, 1.0f / std::sqrt((float) (ID * IH)), s);
        probe("idx_w", il, dlakv, IH);
        if (IH < kIH) {
            cudaMemsetAsync(diq, 0, (size_t) T * kIH * ID * sizeof(float), s);
            cudaMemsetAsync(dw, 0, (size_t) T * kIH * sizeof(float), s);
        }
        cudaMemcpy2DAsync(diq, (size_t) kIH * ID * sizeof(float), datt, (size_t) IH * ID * sizeof(float),
                          (size_t) IH * ID * sizeof(float), (size_t) T, cudaMemcpyDeviceToDevice, s);
        cudaMemcpy2DAsync(dw, (size_t) kIH * sizeof(float), dlakv, (size_t) IH * sizeof(float), (size_t) IH * sizeof(float),
                          (size_t) T, cudaMemcpyDeviceToDevice, s);
        // the scores, the candidates and the top-k list in blocks of kDsv41ScoreRows rows (the score buffer's rows; a decode
        // window is one block): every kernel computes a row on its own, so the blocks give the bits one launch would
        if (gpos && T > kDsv41ScoreRows)   // a graph's positions come from the device: its window must be one block
            throw std::runtime_error("dsv41: a graphed window of more than kDsv41ScoreRows rows");
        // moex (DSV41_PLAN §56 C), STRATA_DSV41_CAND_ONLY=1: after cand_src, the candidates' rows only (their scores the same
        // bits; the mask below sets the others to -inf as it does after a full scan, so the top-k is the same)
        const bool cand_only = dclist && hp.cand_src >= 0 && il > hp.cand_src && hp.cand_bsize > 0 && 64 % hp.cand_bsize == 0;
        // moex (DSV41_PLAN §59 P2), STRATA_DSV41_IDX32=1: the tensor-core scores over V4.1's 32 heads, not the padded 64 (the
        // zero heads' half of each tile left out; the same bits)
        static const bool idx32 = std::getenv("STRATA_DSV41_IDX32") && std::getenv("STRATA_DSV41_IDX32")[0] == '1';
        const int lid_heads = idx32 && IH == 32 ? 32 : kIH;
        for (int b0 = 0; b0 < T; b0 += kDsv41ScoreRows) {
            const int bt = std::min(kDsv41ScoreRows, T - b0);
            const float* biq = diq + (size_t) b0 * kIH * ID;
            const float* bw = dw + (size_t) b0 * kIH;
            uint8_t* bcand = cand ? cand + (size_t) b0 * cbytes : nullptr;
            const int64_t bp = pos0 + b0;
            bool scored = false;
            if (cand_only) {
                dsv41_cand_list(bcand, cbytes, srows, ratio, hp.cand_bsize, dclist, dccount, hp.cand_blocks, bp, bt, s, gpos);
                scored = dsv4_lid_scores_mma(biq, src->lcache, bw, dscore, srows, bp, bt, s, gpos, ratio, kv_pack(), dclist, dccount,
                                             hp.cand_blocks, hp.cand_bsize, lid_heads);
            }
            if (!scored && !dsv4_lid_scores_mma(biq, src->lcache, bw, dscore, srows, bp, bt, s, gpos, ratio, kv_pack(), nullptr,
                                                nullptr, 0, 0, lid_heads))
                dsv4_lid_scores(biq, src->lcache, bw, dscore, srows, bp, bt, s, gpos, ratio, kv_pack());
            if (il == hp.cand_src) dsv41_candidates(dscore, srows, ratio, hp.cand_blocks, hp.cand_bsize, bcand, cbytes, bp, bt, s, gpos);
            else if (hp.cand_src >= 0 && il > hp.cand_src) dsv41_cand_mask(dscore, srows, ratio, bcand, cbytes, hp.cand_bsize, bp, bt, s, gpos);
            dsv4_lid_select(dscore, srows, (int) hp.idx_topk, sel + (size_t) b0 * hp.idx_topk, nullptr, bp, bt, s, gpos, ratio);
        }
        probe_i32("idx_sel", il, sel, hp.idx_topk);
        if (m.rec && T == 1) dump_selection(il, ratio, src->lcache, srows, cand, cbytes, pos0);
    }
    // the attention with sinks over the raw window and the listed compressed rows, then the de-RoPE and o's FP8 rounding
    const float scale = 1.0f / std::sqrt((float) D);
    const uint16_t* comp = src ? src->ccache : nullptr;
    const int32_t* lsel = ratio > 0 ? sel : nullptr;
    const int nc = attn_block ? 2 : attn_noncausal ? 1 : 0;   // DSpark's block (V4.1), a non-causal batch, or causal
    const bool cpk = kv_pack() && comp != nullptr;   // (moex, #104: the source's rows packed)
    if (!dsv4_attn_mma(dq, l.raw, (int) l.raw_rows, comp, ratio, lsel, (int) hp.idx_topk, l.sinks.f(), scale, datt, dapart, pos0, T, s,
                       gpos, nc, cpk))
        dsv4_attn(dq, l.raw, (int) l.raw_rows, comp, ratio, lsel, (int) hp.idx_topk, l.sinks.f(), scale, datt, dapart, pos0, T, s, gpos,
                  nc, cpk);
    probe("attn", il, datt, H * D);
    dsv4_rope(datt, datt, T, H, D, roff, rp, true, pos0, s, gpos);
    dsv41_fq_fp8(datt, T, H * D, s);
    probe("attn_derope", il, datt, D, H);
    // the grouped low-rank output: group g = heads [8g, 8g + 8) through wo_a's rows [g OL, (g + 1) OL), then wo_b
    const int gw = H * D / G;
    permute_abc(datt, dq, T, G, gw, s);           // (T, G, gw) -> (G, T, gw); q is spent
    // STRATA_DSV41_HEADS_MMVQ=1 (A/B): a decode window's groups as one native MMVQ each.  heads_q8_0's one launch reads wo_a
    // (8 groups of 1024 x 4096 Q8_0, 35.7 MB) at ~200 GB/s - 153-194 us a layer, 6-8 ms a token (nsys, 2026-10-18) - where
    // the native kernel reads wo_b at ~650 GB/s
    static const bool heads_mmvq = [] {
        const char* e = std::getenv("STRATA_DSV41_HEADS_MMVQ");
        return e && e[0] == '1';
    }();
    // moex (DSV41_PLAN §33), opt-in STRATA_DSV41_WOA_BATCH=1: one column's groups in one launch of the same kernel (each
    // group's rows bitwise its own launch's) - eight launches of 8.5 us each were 68 us of a decode layer
    static const bool woa_batch = [] {
        const char* e = std::getenv("STRATA_DSV41_WOA_BATCH");
        return e && e[0] == '1';
    }();
    if (heads_mmvq && T <= kTmax && l.wo_a.type == 8) {
        quant(dq, xq, G * T * gw, 1);
        const size_t head_w = (size_t) OL * kn::native_mmvq_weight_bytes(l.wo_a.type, gw, 1);
        if (!(woa_batch && T == 1 &&
              kn::native_q8_0_mmvq_batched(l.wo_a.d, head_w, xq, (size_t) q8(gw), doa_hm, (size_t) OL, gw, OL, G, s)))
            for (int g = 0; g < G; ++g)
                kn::native_mmvq(l.wo_a.type, static_cast<const uint8_t*>(l.wo_a.d) + g * head_w,
                                static_cast<const uint8_t*>(xq) + (size_t) g * T * q8(gw), doa_hm + (size_t) g * T * OL, gw, OL, T, s);
    } else {
        heads_mm(l.wo_a, dq, doa_hm, G, gw, OL, T);   // (G, T, OL)
    }
    probe("attn_wo_a", il, doa_hm, OL, T, G);
    permute_abc(doa_hm, doa, G, T, OL, s);        // (T, G * OL)
    if (pj_needs_q(l.wo_b)) quant(doa, xq, G * OL, T);
    pj(l.wo_b, doa, xq, out_r, T);
    probe("attn_out", il, out_r, n);
}

/// moex (DSV41_PLAN §57): rows [0, min(c_rows, len / ratio + 1)) of each compressed cache on this stage's layers, at their row
/// strides (V4: every compressed layer, the indexer's rows at ratio 4; V4.1: the KV sources, packed or F16) - the positional
/// state a conversation of `len` positions holds; a block still pooling is in the compressor's ring (the running state)
void Stage::positional_regions(int64_t len, std::vector<std::pair<void*, size_t>>& out) const {
    const int64_t D = hp.hd_k, ID = hp.idx_dim;
    const bool pk = hp.v41 && kv_pack();
    const size_t crow = pk ? (size_t) dsv41_pack_row_bytes((int) D, 16) : (size_t) D * 2;
    const size_t lrow = pk ? (size_t) dsv41_pack_row_bytes((int) ID, 32) : (size_t) ID * 2;
    for (int64_t il = lb; il < le; ++il) {
        const Layer& l = L[(size_t) il];
        const int r = hp.cratio[(size_t) il];
        if (r <= 0 || !l.ccache) continue;
        const int64_t rows = std::min<int64_t>(l.c_rows, std::max<int64_t>(0, len) / r + 1);
        out.push_back({l.ccache, (size_t) rows * crow});
        if (l.lcache) out.push_back({l.lcache, (size_t) rows * lrow});
    }
}

}  // namespace strata::x
