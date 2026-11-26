// src/x/dspark.cpp - moex: DeepSeek-V4's DSpark drafter, as llama.cpp runs it (--spec-type draft-dspark -md FILE):
//   * the model: src/models/dflash.cpp - load_arch_hparams / load_arch_tensors (the dsv4_hc_mult branch), graph_dsv4 (the
//     DeepSeek-V4 decoder, "dual-mode by batch type") and build_dspark_markov_head;
//   * the loop: common/speculative.cpp common_speculative_impl_draft_dflash (is_dspark) - process() after every target batch,
//     draft() before every verify window; tools/server/server-context.cpp drops the drafter's cells past the committed
//     position after every draft.
// What it is: three DeepSeek-V4 blocks (ratio 0: the raw 128-position window only, sinks, the grouped output, mHC x 4, their
// own 256 routed experts and a shared one) over the TRUNK's features.  After the trunk reads rows (a prompt chunk, a decode
// window), each row's features - the mean of the trunk's four residual streams at the input of each of the drafter's
// target layers, n_layer being the trunk's output (deepseek4.cpp build_hc_mean as layer_inp) - are fused (fc, enc norm) and
// projected through every block's own wkv (kv_norm, plain RoPE) into that block's K = V ring at the row's position
// ("inject").  A draft step decodes the noise block [anchor, mask x (n - 1)] at the positions after the committed ones -
// non-causal inside the block - through the three blocks, the drafter's head reduce and norm, the TRUNK's output head, then
// the Markov head: position i's logits plus w2 . w1[prev], prev the argmax of position i - 1 (the anchor first); the drafts
// are those argmaxes (llama.cpp's draft sampler takes its top candidate).  The block's own ring rows sit past the committed
// position and are overwritten by the next inject before any read - llama.cpp removes those cells instead.
//
// Here: its dense weights on the last GPU (the trunk's output head is there too), its blocks appended to the engine's layer
// table at L[n_load ..] (so the DeepSeek-V4 attention and mHC code runs them unchanged), its experts in a pinned RAM arena
// of their own, computed by the CPU pool.  Eager (no graphs) for now.
#include "engine_impl.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/x/dsv41_kernels.hpp"

#include "ggml.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>

namespace strata::x {

namespace {
void bf16_rows_to_f32(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t b = (uint32_t) src[i] << 16;
        std::memcpy(dst + i, &b, sizeof(float));
    }
}
}  // namespace

int Engine::Impl::dsp_k(int64_t il) const {
    if (!dsp) return -1;
    for (size_t k = 0; k < dsp->tl.size(); ++k)
        if (dsp->tl[k] == il) return (int) k;
    return -1;
}

bool Engine::Impl::load_dspark(std::string& err) {
    if (opt.dspark.empty()) return true;
    if (hp.v41) return load_dspark_v41(err);   // DeepSeek-V4.1's drafter rides in its own GGUF and pack (dsv41.cpp)
    if (hp.arch != Arch::dsv4) return fail(err, "--draft-dspark drafts for DeepSeek-V4 (a deepseek4 GGUF)");
    auto d = std::make_unique<Dspark>();
    try {
        d->gm = std::make_unique<strata::GgufModel>(strata::GgufModel::open(opt.dspark));
    } catch (const std::exception& e) {
        return fail(err, std::string("the DSpark drafter: ") + e.what());
    }
    const strata::GgufFile& mt = d->gm->meta();
    const strata::MetaValue* a = mt.get("general.architecture");
    if (!a || a->s != "dflash") return fail(err, "the DSpark drafter: not a dflash GGUF");
    auto num = [&](const char* k, double def, bool req, double& v) -> bool {
        const strata::MetaValue* x = mt.get(std::string("dflash.") + k);
        if (!x) {
            if (req) return fail(err, std::string("the DSpark drafter: missing dflash.") + k);
            v = def;
            return true;
        }
        v = x->num();
        return true;
    };
    double nl = 0, bs = 0, emb = 0, heads = 0, kl = 0, rot = 0, ql = 0, swa = 0, ne = 0, nu = 0, nff = 0, gate = 0, nsh = 0;
    double ews = 1, hcm = 0, sk = 0, hce = 0, eps = 0, fb = 10000, og = 0, ol = 0, nhash = 0;
    if (!num("block_count", 0, true, nl) || !num("block_size", 0, true, bs) || !num("embedding_length", 0, true, emb) ||
        !num("attention.head_count", 0, true, heads) || !num("attention.key_length", 0, true, kl) ||
        !num("rope.dimension_count", 0, true, rot) || !num("attention.q_lora_rank", 0, true, ql) ||
        !num("attention.sliding_window", 0, true, swa) || !num("expert_count", 0, true, ne) || !num("expert_used_count", 0, true, nu) ||
        !num("expert_feed_forward_length", 0, true, nff) || !num("expert_gating_func", 0, true, gate) ||
        !num("expert_shared_count", 0, true, nsh) || !num("expert_weights_scale", 1, false, ews) ||
        !num("hyper_connection.count", 0, true, hcm) || !num("hyper_connection.sinkhorn_iterations", 0, true, sk) ||
        !num("hyper_connection.epsilon", 0, true, hce) || !num("attention.layer_norm_rms_epsilon", 0, true, eps) ||
        !num("rope.freq_base", 10000, false, fb) || !num("attention.output_group_count", 0, true, og) ||
        !num("attention.output_lora_rank", 0, true, ol) || !num("hash_layer_count", 0, false, nhash))
        return false;
    // the drafter's blocks run through the trunk's DeepSeek-V4 code: the same geometry, eps, mHC and routing
    if ((int64_t) emb != hp.n_embd || (int64_t) heads != hp.n_head || (int64_t) kl != hp.hd_k || (int64_t) rot != hp.n_rot ||
        (int64_t) ql != hp.q_lora || (int64_t) swa != hp.n_swa || (int64_t) ne != hp.n_expert || (int64_t) nu != hp.n_used ||
        (int64_t) nff != hp.n_ff_exp || gate != 4 || nsh != 1 || (float) ews != hp.ew_scale || (int64_t) hcm != hp.hc ||
        (int64_t) sk != hp.sinkhorn_iters || (float) hce != hp.hc_eps || (float) eps != hp.rms_eps || (float) fb != hp.rope_base ||
        (int64_t) og != hp.o_groups || (int64_t) ol != hp.o_lora || nhash != 0)
        return fail(err, "the DSpark drafter's geometry differs from the trunk's (it must be this DeepSeek-V4's drafter)");
    if (const strata::MetaValue* x = mt.get("dflash.attention.compress_ratios"))
        for (const auto& it : x->items)
            if (it.num() != 0) return fail(err, "the DSpark drafter: compressed attention (llama.cpp requires ratio 0 on every block)");
    if (const strata::MetaValue* x = mt.get("dflash.sample_from_anchor"); x && x->s != "true")
        return fail(err, "the DSpark drafter: sample_from_anchor false (a bonus-anchor export) is not supported");
    const strata::MetaValue* tl = mt.get("dflash.target_layers");
    if (!tl || tl->type != strata::MetaType::ARRAY || tl->items.empty()) return fail(err, "the DSpark drafter: no dflash.target_layers");
    for (const auto& it : tl->items) {
        const int64_t il = (int64_t) it.num();
        if (il < 0 || il > hp.n_layer) return fail(err, "the DSpark drafter: target layer " + std::to_string(il) + " out of range");
        d->tl.push_back(il);
    }
    d->n = (int64_t) nl;
    d->block = (int) bs;
    d->base = n_load;
    Stage& lg = last_stage();
    const int di_last = (int) stages.size() - 1;
    for (int64_t il : d->tl)   // the features are captured on the last GPU
        if (il < hp.n_layer && ldev[(size_t) il] != di_last)
            return fail(err, "the DSpark drafter's target layer " + std::to_string(il) + " is not on the last GPU (--split)");

    // the dense tensors (everything but the routed experts) into one allocation on the last GPU
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
            if (nm.size() > 12 && nm.compare(nm.size() - 12, 12, "_exps.weight") == 0) continue;
            const uint64_t b = strata::tensor_payload_bytes(ti);
            if (!b || !d->gm->in_bounds(ti, sh)) return fail(err, "the DSpark drafter: tensor " + nm + " of unknown size");
            items.push_back({&ti, sh, b, total});
            total += (b + 255) & ~uint64_t(255);
        }
    set_dev(lg.dev);
    if (cudaMalloc(&d->dense, total) != cudaSuccess)
        return fail(err, "the DSpark drafter: " + std::to_string(total >> 20) + " MiB of dense weights on GPU " + std::to_string(lg.dev));
    for (const auto& it : items) {
        uint8_t* dst = static_cast<uint8_t*>(d->dense) + it.off;
        if (cudaMemcpy(dst, d->gm->shard(it.shard).tensor_data(*it.ti), it.bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            return fail(err, "the DSpark drafter: upload of " + it.ti->name);
        DT x;
        x.d = dst;
        x.type = (int) it.ti->type;
        for (size_t k = 0; k < it.ti->shape.size() && k < 4; ++k) x.ne[k] = (int64_t) it.ti->shape[k];
        x.bytes = it.bytes;
        d->t[it.ti->name] = x;
    }
    auto get = [&](const std::string& nm, DT& x, bool req) -> bool {
        auto f = d->t.find(nm);
        if (f == d->t.end()) {
            if (req) return fail(err, "the DSpark drafter: missing tensor " + nm);
            return true;
        }
        x = f->second;
        return true;
    };
    const int64_t n = hp.n_embd, V = hp.n_vocab, E = hp.n_expert;
    if (!get("fc.weight", d->fc, true) || !get("enc.output_norm.weight", d->enc_norm, true) ||
        !get("output_norm.weight", d->out_norm, true) || !get("output_hc_fn.weight", d->hc_fn, true) ||
        !get("output_hc_base.weight", d->hc_base, true) || !get("output_hc_scale.weight", d->hc_scale, true))
        return false;
    if (d->fc.n_in() != (int64_t) d->tl.size() * n || d->fc.n_out() != n || !kn::native_mmvq_supported(d->fc.type))
        return fail(err, "the DSpark drafter: fc.weight is not [n_target * n_embd, n_embd] of a GPU type");
    if (d->hc_fn.type != 0 || d->hc_base.type != 0 || d->hc_scale.type != 0 || d->enc_norm.type != 0 || d->out_norm.type != 0)
        return fail(err, "the DSpark drafter: its norms and head reduce must be F32");
    // the Markov head [rank, n_vocab] BF16 each: F32 copies on the last GPU, row t = token t's rank vector
    {
        DT w1, w2;
        if (!get("markov_w1.weight", w1, true) || !get("markov_w2.weight", w2, true)) return false;
        if (w1.type != GGML_TYPE_BF16 || w2.type != GGML_TYPE_BF16 || w1.ne[1] != V || w2.ne[1] != V || w1.ne[0] != w2.ne[0])
            return fail(err, "the DSpark drafter: markov_w1 / markov_w2 are not BF16 [rank, n_vocab]");
        d->rank = w1.ne[0];
        std::vector<float> h((size_t) (d->rank * V));
        for (int which = 0; which < 2; ++which) {
            size_t shd = 0;
            const strata::TensorInfo* ti = d->gm->find(which == 0 ? "markov_w1.weight" : "markov_w2.weight", &shd);
            bf16_rows_to_f32(reinterpret_cast<const uint16_t*>(d->gm->shard(shd).tensor_data(*ti)), h.data(), h.size());
            float* dv = lg.dalloc<float>(h.size(), err);
            if (!dv || cudaMemcpy(dv, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
                return fail(err, "the DSpark drafter: the Markov head's F32 copy");
            (which == 0 ? d->w1 : d->w2) = dv;
        }
    }
    // its blocks, appended to the layer table (ratio 0, on the last GPU)
    const int64_t base = d->base, NB = d->n;
    L.resize((size_t) (base + NB));
    ldev.resize((size_t) (base + NB), di_last);
    hp.cratio.resize((size_t) (base + NB), 0);
    hp.head_kv.resize((size_t) (base + NB), 1);
    std::vector<double> cl_exp((size_t) NB, hp.clamp_exp.empty() ? 0.0 : hp.clamp_exp.back()), cl_sh = cl_exp;
    if (const strata::MetaValue* x = mt.get("dflash.swiglu_clamp_exp"))
        for (size_t j = 0; j < (size_t) NB && j < x->items.size(); ++j) cl_exp[j] = x->items[j].num();
    if (const strata::MetaValue* x = mt.get("dflash.swiglu_clamp_shexp"))
        for (size_t j = 0; j < (size_t) NB && j < x->items.size(); ++j) cl_sh[j] = x->items[j].num();
    else
        cl_sh = cl_exp;
    hp.clamp_exp.resize((size_t) (base + NB));
    hp.clamp_sh.resize((size_t) (base + NB));
    uint64_t blob = 0;
    for (int64_t j = 0; j < NB; ++j) {
        hp.clamp_exp[(size_t) (base + j)] = (float) cl_exp[(size_t) j];
        hp.clamp_sh[(size_t) (base + j)] = (float) cl_sh[(size_t) j];
        Layer& l = L[(size_t) (base + j)];
        auto nm = [&](const char* sfx) { return "blk." + std::to_string(j) + "." + sfx; };
        bool ok = get(nm("attn_norm.weight"), l.attn_norm, true) && get(nm("ffn_norm.weight"), l.ffn_norm, true) &&
                  get(nm("hc_attn_fn.weight"), l.hc_attn_fn, true) && get(nm("hc_attn_base.weight"), l.hc_attn_base, true) &&
                  get(nm("hc_attn_scale.weight"), l.hc_attn_scale, true) && get(nm("hc_ffn_fn.weight"), l.hc_ffn_fn, true) &&
                  get(nm("hc_ffn_base.weight"), l.hc_ffn_base, true) && get(nm("hc_ffn_scale.weight"), l.hc_ffn_scale, true) &&
                  get(nm("attn_q_a.weight"), l.q_a, true) && get(nm("attn_q_a_norm.weight"), l.q_a_norm, true) &&
                  get(nm("attn_q_b.weight"), l.q_b, true) && get(nm("attn_kv.weight"), l.wkv, true) &&
                  get(nm("attn_kv_a_norm.weight"), l.kv_a_norm, true) && get(nm("attn_sinks.weight"), l.sinks, true) &&
                  get(nm("attn_output_a.weight"), l.wo_a, true) && get(nm("attn_output_b.weight"), l.wo_b, true) &&
                  get(nm("ffn_gate_inp.weight"), l.gate_inp, true) && get(nm("exp_probs_b.bias"), l.probs_b, true) &&
                  get(nm("ffn_gate_shexp.weight"), l.sh_gate, true) && get(nm("ffn_up_shexp.weight"), l.sh_up, true) &&
                  get(nm("ffn_down_shexp.weight"), l.sh_down, true);
        if (!ok) return false;
        for (const DT* x : {&l.q_a, &l.q_b, &l.wkv, &l.wo_a, &l.wo_b, &l.sh_gate, &l.sh_up, &l.sh_down})
            if (!kn::native_mmvq_supported(x->type)) return fail(err, "the DSpark drafter: a block projection of a type with no GPU kernel");
        if (l.hc_attn_fn.type != 0 || l.hc_ffn_fn.type != 0 || l.sinks.type != 0 || l.probs_b.type != 0)
            return fail(err, "the DSpark drafter: its mixers, sinks and bias must be F32");
        // the router: BF16 -> an F32 copy (exact), as the trunk's (dsv4.cpp)
        if (l.gate_inp.type == GGML_TYPE_BF16) {
            size_t shd = 0;
            const strata::TensorInfo* ti = d->gm->find(nm("ffn_gate_inp.weight"), &shd);
            std::vector<float> w((size_t) (n * E));
            bf16_rows_to_f32(reinterpret_cast<const uint16_t*>(d->gm->shard(shd).tensor_data(*ti)), w.data(), w.size());
            float* dw = lg.dalloc<float>(w.size(), err);
            if (!dw || cudaMemcpy(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
                return fail(err, "the DSpark drafter: the router's F32 copy");
            l.gate_inp.d = dw;
            l.gate_inp.type = 0;
        } else if (l.gate_inp.type != 0) {
            return fail(err, "the DSpark drafter: " + nm("ffn_gate_inp.weight") + " is neither BF16 nor F32");
        }
        // the routed experts' native format (one per block: its SwiGLU limit)
        size_t sg = 0, sd = 0;
        const strata::TensorInfo* g = d->gm->find(nm("ffn_gate_exps.weight"), &sg);
        const strata::TensorInfo* dn = d->gm->find(nm("ffn_down_exps.weight"), &sd);
        std::string ferr;
        if (!g || !dn || !kc::native_fmt((int) g->type, (int) dn->type, n, hp.n_ff_exp, l.fmt, ferr))
            return fail(err, "the DSpark drafter's experts: " + ferr);
        l.fmt.swiglu_limit = hp.clamp_exp[(size_t) (base + j)];
        blob = std::max<uint64_t>(blob, (l.fmt.bytes + 4095) & ~(uint64_t) 4095);
        // its K = V ring (the trunk's raw-ring geometry: the window plus the longest batch)
        l.raw_rows = dsv4_raw_rows();
        if (!(l.raw = lg.dalloc<uint16_t>((size_t) (l.raw_rows * hp.hd_k), err))) return false;
    }
    // the routed experts: [gate | up | down] blobs in a pinned arena, read from the file by a few threads
    d->blob = blob;
    const uint64_t arena_bytes = (uint64_t) NB * E * blob;
    d->arena = std::make_unique<strata::core::PinnedArena>(arena_bytes);
    if (!d->arena->valid()) return fail(err, "the DSpark drafter: " + std::to_string(arena_bytes >> 20) + " MiB of RAM for its experts");
    {
        const auto t0 = clk::now();
        std::vector<std::thread> th;
        std::atomic<int64_t> next{0};
        std::atomic<bool> bad{false};
        const int nthr = 8;
        for (int w = 0; w < nthr; ++w)
            th.emplace_back([&] {
                for (int64_t key; !bad && (key = next.fetch_add(1)) < NB * E;) {
                    const int64_t j = key / E, e = key % E;
                    const Layer& l = L[(size_t) (base + j)];
                    uint8_t* dst = d->arena->data() + (uint64_t) key * blob;
                    const char* names[3] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
                    const size_t offs[3] = {0, l.fmt.up_off, l.fmt.down_off};
                    for (int part = 0; part < 3; ++part) {
                        size_t shd = 0;
                        const strata::TensorInfo* ti = d->gm->find("blk." + std::to_string(j) + "." + names[part], &shd);
                        if (!ti) {
                            bad = true;
                            return;
                        }
                        const uint64_t pay = strata::tensor_payload_bytes(*ti);
                        const uint64_t want = part < 2 ? (uint64_t) hp.n_ff_exp * l.fmt.gu_row : (uint64_t) hp.n_embd * l.fmt.d_row;
                        if (pay == 0 || pay % (uint64_t) E || pay / (uint64_t) E != want) {   // a type the reader cannot size: 0 bytes
                            bad = true;
                            return;
                        }
                        const uint64_t slice = pay / (uint64_t) E;
                        std::memcpy(dst + offs[part], d->gm->shard(shd).tensor_data(*ti) + (uint64_t) e * slice, slice);
                    }
                }
            });
        for (auto& t : th) t.join();
        if (bad) return fail(err, "the DSpark drafter: an expert tensor is missing or not n_expert slices of its blob part");
        if (opt.verbose)
            fprintf(stderr, "strata-x: DSpark drafter: %lld blocks, %lld experts (%.2f GiB pinned) read in %.1f s, block size %d, "
                            "target layers",
                    (long long) NB, (long long) (NB * E), arena_bytes / 1073741824.0, ms_since(t0) / 1000.0, d->block);
        for (int64_t il : d->tl) fprintf(stderr, opt.verbose ? " %lld" : "", (long long) il);
        if (opt.verbose) fprintf(stderr, "; dense %.2f GiB on GPU %d\n", total / 1073741824.0, lg.dev);
        // the file's pages leave the page cache (the experts are pinned copies now)
        for (size_t sh = 0; sh < d->gm->size(); ++sh)
            if (const int fd = open(d->gm->shard(sh).path().c_str(), O_RDONLY); fd >= 0) {
                posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                close(fd);
            }
    }
    // the mask token: the drafter's block positions after the anchor
    // llama.cpp: llama_vocab_mask of the drafter's vocabulary - its tokenizer.ggml.mask_token_id (128799 in this drafter)
    if (opt.dspark_mask >= 0) d->mask = opt.dspark_mask;
    else if (const strata::MetaValue* x = mt.get("tokenizer.ggml.mask_token_id")) d->mask = (int32_t) x->num();
    else return fail(err, "the DSpark drafter declares no tokenizer.ggml.mask_token_id (--dspark-mask ID)");
    if (d->mask < 0 || d->mask >= V) return fail(err, "the DSpark drafter: mask token out of the vocabulary");
    // the last stage's buffers: the features of up to max(kTmax, the window) rows
    lg.dsp_rows = (int) std::max<int64_t>(kTmax, hp.n_swa);
    const int64_t nt = (int64_t) d->tl.size();
    if (!(lg.dsp_feat = lg.dalloc<float>((size_t) (lg.dsp_rows * nt * n), err)) ||
        !(lg.dsp_tmp = lg.dalloc<float>((size_t) (lg.dsp_rows * n), err)) || !(lg.dsp_g = lg.dalloc<float>((size_t) (lg.dsp_rows * n), err)) ||
        !(lg.dsp_logits = lg.dalloc<float>((size_t) kTmax * V, err)) || !(lg.dsp_bias = lg.dalloc<float>((size_t) V, err)) ||
        !(lg.dsp_dprev = lg.dalloc<int32_t>((size_t) kTmax + 1, err)) || !(lg.dsp_dprob = lg.dalloc<float>((size_t) kTmax, err)))
        return false;
    reset_dev_cache();
    dsp = std::move(d);
    return true;
}

// ------------------------------------------------------------------ the trunk's features and the injection

void Stage::dspark_capture(int il, int r0, int rows) {
    if (!m.dsp || !last || rows <= 0) return;
    const int k = m.dsp_k(il);
    if (k < 0) return;
    const int n = (int) hp.n_embd, nt = (int) m.dsp->tl.size();
    rows = std::min(rows, dsp_rows);
    mhc_mean_w(R + (size_t) r0 * hp.hc * n, dsp_tmp, n, rows, s);   // deepseek4.cpp build_hc_mean (layer_inp)
    cudaMemcpy2DAsync(dsp_feat + (size_t) k * n, (size_t) nt * n * sizeof(float), dsp_tmp, (size_t) n * sizeof(float),
                      (size_t) n * sizeof(float), (size_t) rows, cudaMemcpyDeviceToDevice, s);
}

/// dflash.cpp graph_dsv4, the embd batch: g = RMS(fc . features, enc norm); per block kv = RMS(wkv . g, kv_norm) with plain RoPE
/// on its last 64 dims (rope base, no YaRN: ggml_rope_ext(.., freq_base, 1, 0, 1, 0, 0)) into the block's ring at the row's position.
bool Stage::dspark_inject(int rows, int64_t pos0, std::string& err) {
    Engine::Impl::Dspark& d = *m.dsp;
    const int n = (int) hp.n_embd, D = (int) hp.hd_k, nf = (int) (d.tl.size() * n);
    const Dsv4Rope rp = dsv4_rope_p0(hp.rope_base);
    rows = std::min(rows, dsp_rows);
    try {
        for (int r0 = 0; r0 < rows; r0 += SB) {   // in the active scratch's sub-batches
            const int tn = std::min(SB, rows - r0);
            const float* f = dsp_feat + (size_t) r0 * nf;
            float* g = dsp_g + (size_t) r0 * n;
            quant(f, xq, nf, tn);
            pj(d.fc, f, xq, g, tn);
            rms_norm(g, d.enc_norm.f(), g, tn, n, hp.rms_eps, s);
            quant(g, xq, n, tn);
            for (int64_t j = 0; j < d.n; ++j) {
                const Layer& l = L[(size_t) (d.base + j)];
                pj(l.wkv, g, xq, dkv, tn);
                rms_norm(dkv, l.kv_a_norm.f(), dkv, tn, D, hp.rms_eps, s);
                dsv4_rope(dkv, dkv, tn, 1, D, D - (int) hp.n_rot, rp, false, pos0 + r0, s);
                if (hp.v41) dsv41_fq_fp8(dkv, tn, D, s);   // DeepSeek-V4.1: main_kv rounded to FP8 (model.py act_quant)
                dsv4_raw_store(dkv, l.raw, (int) l.raw_rows, pos0 + r0, tn, s);
            }
        }
    } catch (const std::exception& e) {
        return m.fail(err, std::string("DSpark inject: ") + e.what());
    }
    return true;
}

// ------------------------------------------------------------------ a draft step

/// One drafter block's MoE over T rows (DSV4_SPEC 3.11 as build_moe_ffn: sqrt-softplus top-k with its bias, normalized weights
/// times expert_weights_scale; the routed experts on the CPU pool from the drafter's arena, the shared expert on the GPU meanwhile).
bool Stage::dspark_moe(int il, int T, std::string& err) {
    const Engine::Impl::Dspark& d = *m.dsp;
    ExpertSrc src;
    src.g = d.arena->data() + (uint64_t) ((il - d.base) * hp.E_of(il)) * d.blob;
    src.gs = d.blob;
    return arena_moe(il, T, src, err);   // dsv4_mtp.cpp
}

/// llama.cpp's draft(): the noise block [anchor, mask x (n - 1)] at positions pos.. through the drafter (graph_dsv4, the token
/// batch: hc_init, three blocks with non-causal attention, the head reduce, its output norm, the trunk's output head), then the
/// Markov head along the block (build_dspark_markov_head: prev = the anchor; logits_i + w2 . w1[prev]; prev = their argmax) -
/// the drafts are the argmaxes (llama.cpp's draft sampler, top-k 10, reads its top candidate) with their probabilities.
bool Stage::dspark_draft(int32_t anchor, int64_t pos, int n_draft, std::string& err) {
    Engine::Impl::Dspark& d = *m.dsp;
    static const bool timing = std::getenv("STRATA_X_DSPARK_TIMING") != nullptr;   // where a draft step's time goes
    static double t_all = 0, t_pool = 0, t_markov = 0;
    static int t_cnt = 0;
    const auto t_start = clk::now();
    const double pool0 = st.ms_pool;
    const int n = (int) hp.n_embd, V = (int) hp.n_vocab, T = std::min(n_draft, std::min(d.block, kTmax));
    dsp_n = 0;
    if (T < 1) return true;
    use(win);
    int32_t toks[kTmax];
    toks[0] = anchor;
    for (int i = 1; i < T; ++i) toks[i] = d.mask;
    if (!upload_embd(toks, T, cur)) return m.fail(err, "DSpark: a token out of the vocabulary");
    for (int hh = 0; hh < (int) hp.hc; ++hh)   // hc_init: the embedding in every stream
        cudaMemcpy2DAsync(R + (size_t) hh * n, (size_t) hp.hc * n * sizeof(float), cur, (size_t) n * sizeof(float),
                          (size_t) n * sizeof(float), (size_t) T, cudaMemcpyDeviceToDevice, s);
    if (d.v41) v41_pre_reset(T);   // DeepSeek-V4.1 (forward_spec): the blocks' carried pre-mix starts at stream 0
    try {
        // V4: llama.cpp's non-causal batch; V4.1: DSpark's block (every row sees the 128 positions before it and the block)
        (d.v41 ? attn_block : attn_noncausal) = true;
        for (int64_t j = 0; j < d.n; ++j) {
            const int il = (int) (d.base + j);
            mix_in(il, T, R, cur, coef, true);
            attention(il, pos, T, cur, out, false);
            mix_out(il, T, out, R, coef, true);
            mix_in(il, T, R, cur, coef, false);
            if (!dspark_moe(il, T, err)) {
                attn_noncausal = attn_block = false;
                return false;
            }
            mix_out(il, T, out, R, coef, false);
        }
        attn_noncausal = attn_block = false;
        if (d.v41) {   // V4.1 (forward_head): the streams collapsed with the last block's FFN pre-mix, its norm, the trunk's head
            mhc_pre_w(R, cpre, h, n, T, s);
            cudaMemcpyAsync(out, h, (size_t) T * n * sizeof(float), cudaMemcpyDeviceToDevice, s);   // the confidence head reads x
            rms_norm(h, d.out_norm.f(), h, T, n, hp.rms_eps, s);
        } else {   // the drafter's head reduce (its output_hc_*), its output norm, the trunk's output head
            const int hcn = (int) hp.hc * n;
            rms_norm(R, nullptr, flat, T, hcn, hp.rms_eps, s);
            gemv_t(d.hc_fn.f(), flat, mixes, hcn, (int) hp.hc, T);
            dsv4_hc_head(R, mixes, d.hc_scale.f(), d.hc_base.f(), hp.hc_eps, h, n, T, s);
            rms_norm(h, d.out_norm.f(), h, T, n, hp.rms_eps, s);
        }
        if (m.output.type == GGML_TYPE_BF16) {   // moex: a GGUF set's BF16 head (the release's), as the trunk's (stage.cpp)
            gemv_dt(m.output, h, dsp_logits, (int) n, (int) hp.n_vocab, T);
        } else {
            quant(h, xq, n, T);
            mm_t(m.output, xq, dsp_logits, T);
        }
        // the Markov head, greedy along the block, all on the device: each step reads its previous token where the argmax
        // before it wrote it, so the block is queued at once and read back with one sync (it synced once a token)
        const auto t_mk = clk::now();
        cudaMemcpyAsync(dsp_dprev, &anchor, sizeof(int32_t), cudaMemcpyHostToDevice, s);
        for (int i = 0; i < T; ++i) {
            float* li = dsp_logits + (size_t) i * V;
            markov_step(d.w2, d.w1, dsp_dprev + i, (int) d.rank, V, li, s);
            argmax_prob_w(li, V, dsp_dprev + i + 1, dsp_dprob + i, 1, s);
        }
        cudaMemcpyAsync(dsp_tok, dsp_dprev + 1, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost, s);
        cudaMemcpyAsync(dsp_prob, dsp_dprob, (size_t) T * sizeof(float), cudaMemcpyDeviceToHost, s);
        if (cudaStreamSynchronize(s) != cudaSuccess) return m.fail(err, "DSpark: the Markov head failed");
        if (d.v41) {   // the confidence head: proj . [x_i | markov_embed(token before draft i)] (model.py DSparkConfidenceHead)
            const int R2 = (int) (n + d.rank);
            cudaMemcpy2DAsync(dsp_cin, (size_t) R2 * sizeof(float), out, (size_t) n * sizeof(float), (size_t) n * sizeof(float),
                              (size_t) T, cudaMemcpyDeviceToDevice, s);
            for (int i = 0; i < T; ++i) {
                const int32_t pv = i == 0 ? anchor : dsp_tok[i - 1];
                cudaMemcpyAsync(dsp_cin + (size_t) i * R2 + n, d.w1 + (size_t) pv * d.rank, (size_t) d.rank * sizeof(float),
                                cudaMemcpyDeviceToDevice, s);
            }
            gemv_t(d.conf.f(), dsp_cin, dsp_cout, R2, 1, T);
            cudaMemcpyAsync(dsp_conf, dsp_cout, (size_t) T * sizeof(float), cudaMemcpyDeviceToHost, s);
            if (cudaStreamSynchronize(s) != cudaSuccess) return m.fail(err, "DSpark: the confidence head failed");
        }
        static const bool print = std::getenv("STRATA_X_DSPARK_PRINT") != nullptr;   // draft by draft, against mainline's -lv 4 log
        if (print) {
            fprintf(stderr, "strata-x: DSpark round at %lld (anchor %d):", (long long) pos, anchor);
            for (int i = 0; i < T; ++i) fprintf(stderr, " %d:%.3f", dsp_tok[i], dsp_prob[i]);
            fprintf(stderr, "\n");
        }
        if (timing) {
            t_markov += ms_since(t_mk);
            t_pool += st.ms_pool - pool0;
            t_all += ms_since(t_start);
            if (++t_cnt % 20 == 0)
                fprintf(stderr, "strata-x: DSpark draft step (%d positions): %.1f ms, of which its experts on the CPU %.1f ms, the "
                                "Markov head %.1f ms (mean of %d)\n", T, t_all / t_cnt, t_pool / t_cnt, t_markov / t_cnt, t_cnt);
        }
    } catch (const std::exception& e) {
        attn_noncausal = attn_block = false;
        return m.fail(err, std::string("DSpark draft: ") + e.what());
    }
    dsp_n = T;
    return true;
}

}  // namespace strata::x
