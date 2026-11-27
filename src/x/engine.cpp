// src/x/engine.cpp - moex: strata-x's engine, first written for GLM-5.3-Flash on Strata's machinery.  See include/strata/x/engine.hpp; the
// internals are in engine_impl.hpp (a stage per GPU), stage.cpp (a stage's layers) and prompt.cpp (the
// prompt path).  This file: loading (the dense weights per GPU, the routed experts into Strata's pinned arena), the
// VRAM expert tier, the decode windows, MTP, and the public API.
//
// The op order is moex/ref/glm_ref.cpp's GlmRef::step, which is llama.cpp's glm5-next graph; the probe names are
// the same, so `strata-x --oracle <llama_dump dir>` checks every op on llama.cpp's own inputs.
#include "engine_impl.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include "ggml.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>

namespace strata::x {

Engine::Impl::~Impl() {
    eg_help_stop();   // moex (DSV41_PLAN §42): the engram rows' helper (STRATA_DSV41_EG_LATE)
    if (eg && eg->pre.joinable()) {   // moex (DSV41_PLAN §18): the scale tables' pre-read ends first
        eg->pre_stop = true;
        eg->pre.join();
    }
    {   // the warm thread reads the GGUF mapping: it ends first
        std::lock_guard<std::mutex> lk(warm_mu);
        warm_stop = true;
    }
    warm_cv.notify_all();
    if (warm_thr.joinable()) warm_thr.join();
    pool.reset();
    stages.clear();
    for (const auto& [p, len] : rec_maps) munmap(p, len);   // DeepSeek-V4.1's record files
    if (eg)   // ... its engram tables' O_DIRECT descriptors
        for (const auto* fds : {&eg->wfd, &eg->sfd})
            for (int fd : *fds)
                if (fd >= 0) close(fd);
    if (eg && eg->gfd >= 0) close(eg->gfd);   // (moex §73: the GGUF table's)
    if (warm_submit.joinable()) warm_submit.join();   // (moex §73: a start stopped part way)
    if (rio && rio->pin_thr.joinable()) rio->pin_thr.join();
    if (rio) {   // ... and its SSD tier: the readers first (they write into the staging slots)
        rio->io.reset();
        for (int fd : rio->fd)
            if (fd >= 0) close(fd);
        for (size_t k = 0; k < rio->blocks.size(); ++k)
            if (k < rio->block_reg.size() && rio->block_reg[k]) {   // faulted in and registered (STRATA_DSV41_FAST_PIN)
                cudaHostUnregister(rio->blocks[k]);
                munmap(rio->blocks[k], (size_t) rio->block_reg[k]);
            } else {
                cudaFreeHost(rio->blocks[k]);
            }
    }
    for (size_t di = 0; di < arenas.size(); ++di)
        if (arenas[di]) cudaFree(arenas[di]);
    for (void* p : dense_extra) cudaFree(p);
    for (void* p : {(void*) h_x, (void*) h_parts, (void*) h_ids, (void*) h_amax, (void*) staging, (void*) h_hist, (void*) nf_h, (void*) route_pen})
        if (p) cudaFreeHost(p);
    arena.reset();
}

// ------------------------------------------------------------------ loading

bool Engine::Impl::load_hparams(std::string& err) {
    const strata::GgufFile& m = gm->meta();
    // the architecture's name: "glm5next" (the llama.cpp PR tree's converter) or "glm5-next" (mainline's, unsloth's GGUFs);
    // every key and tensor name is the same under either (compared 2026-10-07, UD-IQ2_XXS against the local Q2_K), except
    // mainline's attention.indexer.index_share_mtp, which llama.cpp reads and never uses - nor does this engine
    const strata::MetaValue* arch = m.get("general.architecture");
    if (arch && arch->s == "mimo2") return load_hparams_mimo(err);
    if (arch && arch->s == "deepseek4") return load_hparams_dsv4(err);
    if (arch && arch->s == "deepseek41") return load_hparams_dsv41(err);   // DeepSeek-V4.1 (dsv41.cpp)
    if (arch && arch->s == "hy_v4_flash") return load_hparams_hy4(err);    // Hy4-flash (hy4.cpp)
    if (arch && arch->s == "ernie55") return load_hparams_ernie(err);      // ERNIE-5.5-Flash (ernie.cpp)
    if (!arch || (arch->s != "glm5next" && arch->s != "glm5-next"))
        return fail(err, "not a GLM-5.3 (glm5next / glm5-next), MiMo-V2 (mimo2), DeepSeek-V4 (deepseek4), DeepSeek-V4.1 (deepseek41), "
                         "Hy4-flash (hy_v4_flash) or ERNIE-5.5-Flash (ernie55) GGUF");
    const std::string pre = arch->s + ".";
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
    if (!u("embedding_length", h.n_embd) || !u("block_count", h.n_layer_all) || !u("vocab_size", h.n_vocab) ||
        !u("attention.head_count", h.n_head) || !u("kda.head_dim", h.hd_kda) || !u("ssm.conv_kernel", h.d_conv) ||
        !u("feed_forward_length", h.n_ff) || !u("expert_count", h.n_expert) || !u("expert_used_count", h.n_used) ||
        !u("expert_feed_forward_length", h.n_ff_exp) || !u("attention.q_lora_rank", h.q_lora) ||
        !u("attention.kv_lora_rank", h.kv_lora) || !u("attention.key_length_mla", h.k_mla) ||
        !u("attention.value_length_mla", h.v_mla) || !u("attention.indexer.head_count", h.idx_heads) ||
        !u("attention.indexer.key_length", h.idx_dim) || !u("attention.indexer.top_k", h.idx_topk) ||
        !u("attention.indexer.kpool", h.kpool) || !u("hyper_connection.count", h.hc) ||
        !u("hyper_connection.sinkhorn_iterations", h.sinkhorn_iters))
        return false;
    int64_t nextn = 0;
    if (get("nextn_predict_layers")) u("nextn_predict_layers", nextn);
    h.n_layer = h.n_layer_all - nextn;
    if (get("leading_dense_block_count")) u("leading_dense_block_count", h.n_dense);
    if (const strata::MetaValue* v = get("attention.indexer.kpool_select_tail")) h.select_tail = v->u != 0;
    if (const strata::MetaValue* v = m.get("tokenizer.ggml.eos_token_id")) h.eos_id = (int64_t) v->num();
    if (!f("hyper_connection.epsilon", h.hc_eps, 0, true) || !f("attention.layer_norm_rms_epsilon", h.rms_eps, 0, true))
        return false;
    f("attention.layer_norm_epsilon", h.ln_eps, 1e-6f, false);
    f("kda.gate_lower_bound", h.kda_lb, -INFINITY, false);
    f("expert_weights_scale", h.ew_scale, 1.0f, false);
    if (const strata::MetaValue* v = get("expert_weights_norm")) h.ew_norm = v->u != 0;
    if (const strata::MetaValue* v = get("expert_gating_func"); v && v->u != 2) return fail(err, "expert gating is not sigmoid");
    if (const strata::MetaValue* v = get("rope.dimension_count"); v && v->u != 0) return fail(err, "MLA with rope");
    auto arr_i = [&](const char* k, std::vector<int64_t>& dst) -> bool {
        const strata::MetaValue* v = get(k);
        if (!v || v->type != strata::MetaType::ARRAY || (int64_t) v->items.size() != h.n_layer_all)
            return fail(err, std::string("glm5next.") + k + " is not a per-layer array");
        for (const auto& it : v->items) dst.push_back((int64_t) it.num());
        return true;
    };
    auto arr_f = [&](const char* k, std::vector<float>& dst) -> bool {
        const strata::MetaValue* v = get(k);
        if (!v) return fail(err, std::string("missing glm5next.") + k);
        if (v->type != strata::MetaType::ARRAY) {
            dst.assign((size_t) h.n_layer_all, (float) v->num());
            return true;
        }
        if ((int64_t) v->items.size() != h.n_layer_all) return fail(err, std::string("glm5next.") + k + " length");
        for (const auto& it : v->items) dst.push_back((float) it.num());
        return true;
    };
    if (!arr_i("attention.head_count_kv", h.head_kv) || !arr_f("swiglu_clamp_exp", h.clamp_exp)) return false;
    if (get("swiglu_clamp_shexp")) {
        if (!arr_f("swiglu_clamp_shexp", h.clamp_sh)) return false;
    } else {
        h.clamp_sh = h.clamp_exp;
    }
    if (h.hc != 4) return fail(err, "mHC with hc != 4");
    if (h.hd_kda != 128 || h.idx_heads != 32 || h.idx_dim != 128 || h.kv_lora != 512)
        return fail(err, "kernel geometry: KDA head 128, indexer 32 x 128 and a 512 latent are what the kernels take");
    return true;
}

bool Engine::Impl::verify_dense(int il, const char* when, std::string& err) {
    const std::string pre = il >= 0 ? "blk." + std::to_string(il) + "." : "";
    std::vector<uint8_t> h;
    for (const auto& [name, hash] : dense_hash) {
        if (il >= 0 && name.rfind(pre, 0) != 0) continue;
        const DT& d = t[name];
        h.resize(d.bytes);
        int dev = (int) stages.size() - 1;
        if (name.rfind("blk.", 0) == 0) dev = ldev[(size_t) std::strtoll(name.c_str() + 4, nullptr, 10)];
        cudaSetDevice(stages[(size_t) dev]->dev);
        cudaDeviceSynchronize();
        if (cudaMemcpy(h.data(), d.d, d.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return fail(err, "readback of " + name + " failed");
        if (slices_hash_host(h.data(), d.bytes) != hash) return fail(err, std::string("dense tensor ") + name + " differs from the file " + when);
    }
    reset_dev_cache();
    return true;
}

/// MiMo-V2.6-Flash (llama.cpp's mimo2: src/models/mimo2.cpp, load_arch_hparams): the keys, and the per-layer facts the tensors
/// carry (which layers route experts; no shared expert).  No mHC: one residual stream.
bool Engine::Impl::load_hparams_mimo(std::string& err) {
    const strata::GgufFile& m = gm->meta();
    const std::string pre = "mimo2.";
    auto get = [&](const std::string& k) { return m.get(pre + k); };
    auto u = [&](const char* k, int64_t& dst) -> bool {
        const strata::MetaValue* v = get(k);
        if (!v) return fail(err, "missing " + pre + k);
        dst = (int64_t) v->num();
        return true;
    };
    auto f = [&](const char* k, float& dst, float def) {
        const strata::MetaValue* v = get(k);
        dst = v ? (float) v->num() : def;
        return v != nullptr;
    };
    Hparams& h = hp;
    h.arch = Arch::mimo2;
    if (!u("embedding_length", h.n_embd) || !u("block_count", h.n_layer_all) || !u("attention.head_count", h.n_head) ||
        !u("attention.key_length", h.hd_k) || !u("attention.value_length", h.hd_v) || !u("rope.dimension_count", h.n_rot) ||
        !u("attention.sliding_window", h.n_swa) || !u("feed_forward_length", h.n_ff) || !u("expert_count", h.n_expert) ||
        !u("expert_used_count", h.n_used) || !u("expert_feed_forward_length", h.n_ff_exp))
        return false;
    int64_t nextn = 0;
    if (get("nextn_predict_layers")) u("nextn_predict_layers", nextn);
    h.n_layer = h.n_layer_all - nextn;
    if (!f("attention.layer_norm_rms_epsilon", h.rms_eps, 0)) return fail(err, "missing mimo2.attention.layer_norm_rms_epsilon");
    f("rope.freq_base", h.rope_base, 10000.0f);
    f("rope.freq_base_swa", h.rope_base_swa, 10000.0f);   // llama_hparams' default
    f("attention.value_scale", h.v_scale, 0.0f);
    if (h.v_scale == 1.0f) h.v_scale = 0.0f;                // llama.cpp keeps only a value != 1
    f("expert_weights_scale", h.ew_scale, 0.0f);            // build_moe_ffn scales only by a value other than 0 and 1
    h.ew_norm = true;                                       // mimo2.cpp: build_moe_ffn(..., norm_w = true, ...)
    if (const strata::MetaValue* v = get("expert_gating_func"); v && v->u != 2) return fail(err, "expert gating is not sigmoid");
    if (const strata::MetaValue* v = get("expert_group_count"); v && v->num() > 1) return fail(err, "grouped expert routing");
    if (const strata::MetaValue* v = m.get("tokenizer.ggml.eos_token_id")) h.eos_id = (int64_t) v->num();
    auto per_layer = [&](const char* k, std::vector<int64_t>& dst) -> bool {
        const strata::MetaValue* v = get(k);
        if (!v) return fail(err, "missing " + pre + k);
        if (v->type != strata::MetaType::ARRAY) {
            dst.assign((size_t) h.n_layer_all, (int64_t) v->num());
            return true;
        }
        if ((int64_t) v->items.size() != h.n_layer_all) return fail(err, pre + k + " is not a per-layer array");
        for (const auto& it : v->items) dst.push_back((int64_t) it.num());
        return true;
    };
    std::vector<int64_t> swa;
    if (!per_layer("attention.head_count_kv", h.head_kv) || !per_layer("attention.sliding_window_pattern", swa)) return false;
    h.swa_l.assign((size_t) h.n_layer_all, 0);
    for (int64_t il = 0; il < h.n_layer_all; ++il) h.swa_l[(size_t) il] = swa[(size_t) il] != 0;
    // which blocks route experts, from the tensors (mimo2.cpp reads ffn_gate_inp as optional; its absence is the dense FFN)
    h.moe_l.assign((size_t) h.n_layer_all, 0);
    size_t sh = 0;
    for (int64_t il = 0; il < h.n_layer_all; ++il)
        h.moe_l[(size_t) il] = gm->find("blk." + std::to_string(il) + ".ffn_gate_inp.weight", &sh) != nullptr;
    h.n_dense = 0;
    while (h.n_dense < h.n_layer && !h.moe_l[(size_t) h.n_dense]) ++h.n_dense;
    for (int64_t il = h.n_dense; il < h.n_layer; ++il)
        if (!h.moe_l[(size_t) il]) return fail(err, "a dense block after the first MoE block");
    h.shexp = false;
    for (int64_t il = 0; il < h.n_layer_all; ++il)
        if (gm->find("blk." + std::to_string(il) + ".ffn_gate_shexp.weight", &sh)) return fail(err, "a shared expert in a mimo2 GGUF");
    const strata::TensorInfo* te = gm->find("token_embd.weight", &sh);
    if (!te || te->shape.size() < 2) return fail(err, "no token_embd.weight");
    h.n_vocab = (int64_t) te->shape[1];
    h.hc = 1;
    h.clamp_exp.assign((size_t) h.n_layer_all, 0.0f);   // no SwiGLU limits: plain SiLU
    h.clamp_sh = h.clamp_exp;
    if (h.n_rot <= 0 || h.n_rot > h.hd_k || h.n_rot % 2 != 0 || h.hd_k > 256 || h.hd_v > 256 || h.hd_k % 32 != 0 ||
        h.hd_v % 32 != 0)
        return fail(err, "kernel geometry: K / V heads of up to 256 (multiples of 32) and an even rotary width");
    for (int64_t il = 0; il < h.n_layer_all; ++il)
        if (h.head_kv[(size_t) il] <= 0 || h.n_head % h.head_kv[(size_t) il] != 0 || h.n_head / h.head_kv[(size_t) il] > 16)
            return fail(err, "kernel geometry: up to 16 query heads per KV head");
    return true;
}

// moex (DSV41_PLAN §70): 32 BF16 values of a row as 32 e4m3 codes and one e8m0 scale - the smallest power of two that holds
// them (448 = 1.75 x 2^8) - if they are FP8 values exactly; false if not (q null: only the check).  Each value read back the
// GPU's way (gemv_fp8_w's e4m3_val) must give its BF16 bits
static bool fp8_block(const uint16_t* x, uint8_t* q, uint8_t& sc) {
    int emax = -1;   // the largest biased exponent; its mantissa decides the scale
    uint32_t mmax = 0;
    for (int i = 0; i < 32; ++i) {
        const uint32_t ex = (x[i] >> 7) & 0xFFu, m7 = x[i] & 0x7Fu;
        if (ex == 0xFFu || (ex == 0 && m7)) return false;   // (not finite, or a subnormal: kept out)
        if (ex && ((int) ex > emax || ((int) ex == emax && m7 > mmax))) emax = (int) ex, mmax = m7;
    }
    const int k = emax < 0 ? 0 : (emax - 127) - (mmax <= 0x60u ? 8 : 7);   // 1.75 is the 7-bit mantissa 0x60
    // moex (DSV41_PLAN §75): the scale byte S = k + 127 must lie in [10, 127], not [1, 254].
    //   10: the GPU's e4m3_sa builds an exponent-0 code's value as em x 2^(S - 136), the float whose exponent field is
    //   S - 9, which needs S - 9 >= 1 (at S = 9 the factor would be 0.0, where e4m3_val gives the denormal em x 2^-127).
    //   127: the byte's bit 7 is the block's MARK (below), so a scale may not use it.
    // Outside that range a block's largest value is under 2^-110 (and not all zero: an all-zero block takes k = 0) or 256
    // and more - no weight of this model; such a matrix keeps its BF16, as one that is not exactly FP8 does
    if (k + 127 < 10 || k + 127 > 127) return false;
    uint8_t c8[32];
    for (int i = 0; i < 32; ++i) {
        const uint32_t sg = (x[i] >> 8) & 0x80u, ex = (x[i] >> 7) & 0xFFu, m7 = x[i] & 0x7Fu;
        uint32_t c = sg;
        if (ex) {
            const int E = (int) ex - 127 - k;   // the value's exponent over the scale
            if (E >= -6) {
                if ((m7 & 0xFu) || E > 8 || (E == 8 && (m7 >> 4) == 7)) return false;
                c |= (uint32_t) (E + 7) << 3 | (m7 >> 4);
            } else if (E == -7 && !(m7 & 31u)) c |= 4 + (m7 >> 5);   // e4m3 subnormals: m x 2^-9
            else if (E == -8 && !(m7 & 63u)) c |= 2 + (m7 >> 6);
            else if (E == -9 && !m7) c |= 1;
            else return false;
        }
        c8[i] = (uint8_t) c;
    }
    const uint32_t sb = (uint32_t) (k + 127) << 23;
    float scf;
    std::memcpy(&scf, &sb, 4);
    for (int i = 0; i < 32; ++i) {   // read back as e4m3_val does: the value times the scale, the sign last
        const uint32_t c = c8[i], em = c & 0x7Fu;
        float v;
        if (em >> 3) {
            const uint32_t vb = (em << 20) + (120u << 23);
            std::memcpy(&v, &vb, 4);
        } else {
            v = (float) (em & 7u) * 0.001953125f;
        }
        v *= scf;
        uint32_t got;
        std::memcpy(&got, &v, 4);
        got |= (c & 0x80u) << 24;
        if (got != (uint32_t) x[i] << 16) return false;
    }
    if (q) std::memcpy(q, c8, 32);
    // moex (DSV41_PLAN §75): bit 7 of the scale byte marks a block that holds an exponent-0 code (a zero or an e4m3
    // subnormal: 0.2-0.3% of the release's blocks, 0.008% of its codes).  An unmarked block's codes all have an exponent
    // >= 1, which the decode's kernels turn into F32 bits with three integer operations a weight and no test
    // (gemv_fp8_rows_body); a marked block takes the general decode.  Every reader of the scales masks the bit
    // (kFp8ScaleMask: gemv_fp8_w_kernel, gemv_fp8_w1_pre_kernel, fp8_widen_kernel)
    bool sub = false;
    for (int i = 0; i < 32; ++i) sub = sub || (c8[i] & 0x78u) == 0;
    sc = (uint8_t) ((k + 127) | (sub ? 0x80 : 0));
    return true;
}

bool Engine::Impl::load_dense(std::string& err) {
    // every tensor but the routed experts and the embedding table (the host reads one row per token); the NextN / MTP
    // blocks only with MTP
    struct Item {
        const strata::TensorInfo* ti;
        size_t shard;
        uint64_t bytes, off;
        int di;
        bool q6k = false;          ///< requantized to Q6_K as it loads (bytes: the Q6_K size)
        bool bf16 = false;         ///< moex (§62): an F32 tensor of BF16 values, kept as BF16 (bytes: half the file's)
        uint64_t file_bytes = 0;   ///< its payload in the file
        bool q8 = false;           ///< moex: a BF16 matrix quantized to Q8_0 as it loads (STRATA_DSV41_DENSE_Q8)
        bool f8 = false;           ///< moex (§70): a BF16 matrix back to its FP8 values as it loads (STRATA_DSV41_DENSE_FP8)
        uint64_t src_off = 0;      ///< moex (§76): a part of the tensor - its first byte in the payload (bytes: the part's)
        const char* alias = nullptr;   ///< ... the name the part is kept under
        int64_t rows = -1;         ///< ... and its rows (the DT's ne[1])
    };
    std::vector<Item> items;
    // STRATA_DSV41_DENSE_Q6K=1 (opt-in, lossy): DeepSeek-V4.1's Q8_0 matrices requantized to Q6_K as they load (Xolotl's
    // STRATA_DENSE_REQUANT; there KL 0.2281 -> 0.2306) - 23% fewer bytes through the GPU a decode token, ~2 GiB of VRAM to
    // expert slots
    static const bool dense_q6k = [] {
        const char* e = std::getenv("STRATA_DSV41_DENSE_Q6K");
        return e && e[0] == '1';
    }();
    auto q6k_of = [&](const strata::TensorInfo& ti) -> bool {
        return dense_q6k && hp.v41 && (int) ti.type == GGML_TYPE_Q8_0 && ti.shape.size() >= 2 && ti.shape[0] % 256 == 0;
    };
    // moex (the GGUF set), opt-in STRATA_DSV41_DENSE_Q8=1: llama.cpp's lossless set keeps the FP8 projections as BF16 (exact);
    // this quantizes them to Q8_0 as they load - ggml's quantize_row_q8_0_ref, the blocks write_dense_gguf.py wrote (gguf-py's
    // Q8_0: amax/127, roundf) from the same values - the same matrices our dense GGUF holds, so the same bits: half the
    // bytes through the GPU a decode token and ~6 GiB of VRAM back to expert slots, against the release's exact weights
    static const bool dense_q8 = [] {
        const char* e = std::getenv("STRATA_DSV41_DENSE_Q8");
        return e && e[0] == '1';
    }();
    auto release_fp8 = [&](const std::string& n) -> bool {   // the projections the release holds as FP8
        if (n.rfind("blk.", 0) != 0) return false;
        const size_t dot = n.find('.', 4);   // "blk.N." then the tensor's own name, matched whole (indexer.attn_q_b stays BF16)
        if (dot == std::string::npos) return false;
        const std::string rest = n.substr(dot + 1);
        for (const char* t : {"attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight", "attn_output_a.weight", "attn_output_b.weight",
                              "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", "engram_kv.weight"})
            if (rest == t) return true;
        return false;
    };
    // moex (ERNIE-5.5): the release's FP8 projections, as our converter names them (convert_gguf.cpp; the release's dtypes)
    auto ernie_fp8 = [&](const std::string& n) -> bool {
        if (n.rfind("blk.", 0) != 0) return false;
        const size_t dot = n.find('.', 4);
        if (dot == std::string::npos) return false;
        const std::string rest = n.substr(dot + 1);
        for (const char* t : {"attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight", "attn_output_a.weight", "attn_output_b.weight",
                              "attn_gate.weight", "attn_kv_a_mqa.weight", "attn_k_b.weight", "attn_v_b.weight", "attn_output.weight",
                              "indexer.attn_q_b.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "ffn_gate_shexp.weight",
                              "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_routed_down.weight", "ffn_routed_up.weight"})
            if (rest == t) return true;
        return false;
    };
    auto q8_of = [&](const strata::TensorInfo& ti) -> bool {
        if (!dense_q8 || !hp.v41 || (int) ti.type != GGML_TYPE_BF16 || ti.shape.size() < 2 || ti.shape[0] % 32) return false;
        return ti.name == "output.weight" || release_fp8(ti.name);
    };
    // moex (DSV41_PLAN §70), opt-in STRATA_DSV41_DENSE_FP8=1: those projections back in the release's own precision - llama.cpp
    // wrote each FP8 value times its block's power-of-two scale as the BF16 that holds it exactly, so each run of 32 values of a
    // row is again 32 e4m3 values and one e8m0 scale (the smallest power of two that holds the run: still exact), checked value
    // by value as it loads (a matrix that is not refuses the start). The GPU widens each value to the F32 the BF16 held, then
    // multiplies as before (gemv_fp8_w / fp8_widen): the same bits as BF16, from ~half the VRAM and half the bytes a token
    static const bool dense_fp8 = [] {   // (unset: on - the same bits as BF16; =0 off)
        const char* e = std::getenv("STRATA_DSV41_DENSE_FP8");
        return !(e && e[0] == '0');
    }();
    int n_f8_kept = 0;   // candidates that are no FP8 values: kept as BF16
    // (no check before planning: the conversion checks every value as it writes, and a matrix that is no FP8 values keeps its
    // BF16 in a buffer of its own - the plan's FP8 room for it unused; a check before planning read the 6.5 GB twice, ~9 s)
    auto fp8_of = [&](const strata::TensorInfo& ti, size_t) -> bool {
        if (!dense_fp8 || (int) ti.type != GGML_TYPE_BF16 || ti.shape.size() < 2 || ti.shape[0] % 32) return false;
        if (hp.arch == Arch::ernie) return ernie_fp8(ti.name);
        if (dense_q8 || !hp.v41) return false;
        return release_fp8(ti.name);
    };
    if (dense_fp8 && ((!dense_q8 && hp.v41) || hp.arch == Arch::ernie))   // (moex §70: the conversion reads them through the mapping: the kernel reads ahead;
        for (size_t sh = 0; sh < gm->size(); ++sh) {   //  O_DIRECT reads of them were slower, 22-26 s against 16.8)
            const int fd = open(gm->shard(sh).path().c_str(), O_RDONLY);
            if (fd < 0) continue;
            for (const auto& ti : gm->shard(sh).tensors())
                if ((int) ti.type == GGML_TYPE_BF16 && (hp.arch == Arch::ernie ? ernie_fp8(ti.name) : release_fp8(ti.name)))
                    posix_fadvise(fd, (off_t) (gm->shard(sh).data_start() + ti.offset), (off_t) strata::tensor_payload_bytes(ti),
                                  POSIX_FADV_WILLNEED);
            close(fd);
        }
    std::vector<uint64_t> totals(stages.size(), 0);
    auto dev_of = [&](const std::string& n) -> int {
        if (n.rfind("blk.", 0) == 0) {
            const int64_t il = std::strtoll(n.c_str() + 4, nullptr, 10);
            if (il >= 0 && il < (int64_t) ldev.size()) return ldev[(size_t) il];
        }
        return (int) stages.size() - 1;   // the head and its norm go with the last layers
    };
    // moex (§76): the head's split (below) - two GPUs, DeepSeek-V4.1, no drafter; the ERNIE-5.5 hill-climb: ERNIE-5.5 too (its BF16
    // head is 201,216 x 4,096: 1.65 GB a token, ~2.3 ms of the last GPU with the first idle)
    int64_t head_pct = 0;
    if ((hp.v41 || hp.arch == Arch::ernie) && stages.size() == 2 && opt.dspark.empty() && !mtp_on && !dfl) {
        const char* e = std::getenv("STRATA_DSV41_HEAD_SPLIT");
        head_pct = e && *e ? std::clamp<int64_t>(std::atoll(e), 0, 90) : 45;
    }
    head_lo = 0;
    for (size_t sh = 0; sh < gm->size(); ++sh)
        for (const auto& ti : gm->shard(sh).tensors()) {
            const std::string& n = ti.name;
            if (n == "token_embd.weight") continue;
            // DeepSeek-V4.1's DSpark drafter rides in the model's own GGUF (dspark.*): its dense weights only when it drafts
            // from there (--draft-dspark builtin) - 0.70 GiB on the last GPU otherwise held for nothing, ~48 expert slots
            // (moex: with --draft-dspark FILE they come from that file, load_dspark_v41)
            if (opt.dspark != "builtin" && n.rfind("dspark.", 0) == 0) continue;
            if (n.size() > 12 && n.compare(n.size() - 12, 12, "_exps.weight") == 0) continue;
            // moex (the GGUF set): the engram's table stays in its file (load_engram maps it; 203 GB)
            if (n == "per_layer_token_embd.weight") continue;
            // the GGUF's own MTP blocks only when they draft (MiMo with --draft-dflash leaves them on the disk)
            if ((!mtp_on || dfl || dfy) && n.rfind("blk.", 0) == 0 && std::strtoll(n.c_str() + 4, nullptr, 10) >= hp.n_layer) continue;
            const uint64_t fb = strata::tensor_payload_bytes(ti);
            if (!fb || !gm->in_bounds(ti, sh)) return fail(err, "tensor " + n + ": unknown size or out of bounds");
            // moex (DSV41_BACKLOG #140, DSV41_PLAN §76; STRATA_DSV41_HEAD_SPLIT=P: P% of the rows on the first GPU, 0: off):
            // DeepSeek-V4.1's BF16 head (129,280 x 5,120: 1.32 GB a token, 2.1 ms of the last GPU after its layers, with the
            // first GPU idle) as two runs of rows - the first on the first GPU, the rest on the last - so both compute their
            // rows at once (Stage::head). A row's logit is its own dot product: the same bits on either card. Only without a
            // drafter (they read the whole head on the last GPU)
            if (head_pct > 0 && n == "output.weight" && (int) ti.type == GGML_TYPE_BF16 && ti.shape.size() == 2 &&
                (int64_t) ti.shape[0] == hp.n_embd && (int64_t) ti.shape[1] >= 1024) {
                const uint64_t rowb = (uint64_t) ti.shape[0] * 2;
                const int64_t V = (int64_t) ti.shape[1];
                const int64_t V0 = std::clamp<int64_t>((V * head_pct / 100) & ~int64_t(63), 64, V - 64);
                const int last = (int) stages.size() - 1;
                Item lo{&ti, sh, (uint64_t) V0 * rowb, totals[0], 0};
                lo.file_bytes = lo.bytes;
                lo.alias = "output.weight.lo";
                lo.rows = V0;
                items.push_back(lo);
                totals[0] += (lo.bytes + 255) & ~uint64_t(255);
                Item hi{&ti, sh, (uint64_t) (V - V0) * rowb, totals[(size_t) last], last};
                hi.file_bytes = hi.bytes;
                hi.src_off = lo.bytes;
                hi.alias = "output.weight.hi";
                hi.rows = V - V0;
                items.push_back(hi);
                totals[(size_t) last] += (hi.bytes + 255) & ~uint64_t(255);
                head_lo = V0;
                continue;
            }
            const int di = dev_of(n);
            const bool q6 = q6k_of(ti);
            const bool b16 = hp.v41 && dsv41_bf16_w() && (int) ti.type == GGML_TYPE_F32 && dsv41_bf16_w_name(n);
            const bool q8 = q8_of(ti);
            const bool f8 = fp8_of(ti, sh);
            const uint64_t b = q6    ? (uint64_t) (fb / ggml_row_size(GGML_TYPE_Q8_0, (int64_t) ti.shape[0])) *
                                        ggml_row_size(GGML_TYPE_Q6_K, (int64_t) ti.shape[0])
                               : b16 ? fb / 2
                               : q8  ? (uint64_t) (fb / ggml_row_size(GGML_TYPE_BF16, (int64_t) ti.shape[0])) *
                                        ggml_row_size(GGML_TYPE_Q8_0, (int64_t) ti.shape[0])
                               : f8  ? fb / 2 + fb / 64   // (a byte a value, a scale byte for every 32)
                                     : fb;
            items.push_back({&ti, sh, b, totals[(size_t) di], di, q6, b16, fb, q8, f8});
            totals[(size_t) di] += (b + 255) & ~uint64_t(255);
        }
    arenas.assign(stages.size(), nullptr);
    uint64_t total = 0;
    for (size_t di = 0; di < stages.size(); ++di) {
        cudaSetDevice(stages[di]->dev);
        if (totals[di] && cudaMalloc(&arenas[di], totals[di]) != cudaSuccess)
            return fail(err, "cannot allocate " + std::to_string(totals[di] >> 20) + " MiB of dense weights on GPU " +
                                 std::to_string(stages[di]->dev));
        total += totals[di];
    }
    const auto t0 = clk::now();
    // THE READ: O_DIRECT in 4 MiB pieces through two pinned bounce buffers (the piece k + 1 is read while piece k is copied up) -
    // through the page cache, the kernel's readahead had queued deep enough on the NVMe to hold other processes' reads back by
    // ~100 ms (measured with a 4 KiB probe beside a MiMo load, 2026-10-07)
    constexpr size_t kPiece = 4u << 20;
    uint8_t* bounce[2] = {nullptr, nullptr};
    cudaEvent_t bev[2] = {nullptr, nullptr};
    if (cudaMallocHost(&bounce[0], kPiece + 8192) != cudaSuccess || cudaMallocHost(&bounce[1], kPiece + 8192) != cudaSuccess)
        return fail(err, "pinned bounce buffers for the dense weights");
    struct Bounce {
        uint8_t** b;
        cudaEvent_t* e;
        ~Bounce() {
            for (int i = 0; i < 2; ++i) {
                if (e[i]) {
                    cudaEventSynchronize(e[i]);
                    cudaEventDestroy(e[i]);
                }
                if (b[i]) cudaFreeHost(b[i]);
            }
        }
    } bounce_guard{bounce, bev};
    std::vector<int> fds(gm->size(), -1);
    struct Fds {
        std::vector<int>& f;
        ~Fds() {
            for (int x : f)
                if (x >= 0) close(x);
        }
    } fds_guard{fds};
    int cur_b = 0, ev_dev = -1;
    auto drain = [&]() {   // an event belongs to its GPU: drain and drop both when the upload moves to another GPU
        for (int i = 0; i < 2; ++i)
            if (bev[i]) {
                cudaEventSynchronize(bev[i]);
                cudaEventDestroy(bev[i]);
                bev[i] = nullptr;
            }
    };
    int n_q6k = 0, n_bf16 = 0, n_q8 = 0, n_f8 = 0;
    uint64_t q6k_saved = 0, f8_saved = 0;
    for (const auto& it : items) {
        uint8_t* dst = static_cast<uint8_t*>(arenas[(size_t) it.di]) + it.off;
        if (stages[(size_t) it.di]->dev != ev_dev) {
            drain();
            ev_dev = stages[(size_t) it.di]->dev;
        }
        cudaSetDevice(ev_dev);
        if (it.q6k) {   // Q8_0 rows -> F32 -> Q6_K (ggml's quantizer, no imatrix), the rows split over threads; then the upload
            const int64_t ne0 = (int64_t) it.ti->shape[0];
            const size_t rb8 = ggml_row_size(GGML_TYPE_Q8_0, ne0), rb6 = ggml_row_size(GGML_TYPE_Q6_K, ne0);
            const int64_t rows = (int64_t) (it.file_bytes / rb8);
            const uint8_t* src = gm->shard(it.shard).tensor_data(*it.ti);
            std::vector<uint8_t> q((size_t) it.bytes);
            ggml_quantize_init(GGML_TYPE_Q6_K);
            const int nth = (int) std::max<int64_t>(1, std::min<int64_t>(16, rows / 64));
            std::vector<std::thread> th;
            for (int w = 0; w < nth; ++w)
                th.emplace_back([&, w] {
                    const int64_t r0 = rows * w / nth, r1 = rows * (w + 1) / nth;
                    std::vector<float> f((size_t) (r1 - r0) * ne0);
                    for (int64_t r = r0; r < r1; ++r)
                        for (int64_t b = 0; b < ne0 / 32; ++b) {
                            const uint8_t* blk = src + (size_t) r * rb8 + (size_t) b * 34;
                            ggml_fp16_t dh;
                            std::memcpy(&dh, blk, 2);
                            const float dd = ggml_fp16_to_fp32(dh);
                            float* o = f.data() + (size_t) (r - r0) * ne0 + (size_t) b * 32;
                            for (int i = 0; i < 32; ++i) o[i] = dd * (float) (int8_t) blk[2 + i];
                        }
                    ggml_quantize_chunk(GGML_TYPE_Q6_K, f.data(), q.data() + (size_t) r0 * rb6, 0, r1 - r0, ne0, nullptr);
                });
            for (auto& x : th) x.join();
            if (cudaMemcpy(dst, q.data(), (size_t) it.bytes, cudaMemcpyHostToDevice) != cudaSuccess)
                return fail(err, "upload failed: " + it.ti->name);
            DT d;
            d.d = dst;
            d.type = (int) GGML_TYPE_Q6_K;
            for (size_t k = 0; k < it.ti->shape.size() && k < 4; ++k) d.ne[k] = (int64_t) it.ti->shape[k];
            d.bytes = it.bytes;
            t[it.ti->name] = d;
            ++n_q6k;
            q6k_saved += it.file_bytes - it.bytes;
            continue;
        }
        if (it.f8) {   // moex (§70, STRATA_DSV41_DENSE_FP8): each run of 32 BF16 values -> 32 e4m3 codes and an e8m0 scale, exactly
            const int64_t ne0 = (int64_t) it.ti->shape[0];
            const int64_t nv = (int64_t) (it.file_bytes / 2), rows = nv / ne0;
            const uint16_t* src = reinterpret_cast<const uint16_t*>(gm->shard(it.shard).tensor_data(*it.ti));
            std::vector<uint8_t> q((size_t) it.bytes);
            uint8_t* qs = q.data() + nv;   // the scales after the values
            std::atomic<int64_t> bad{-1};
            const int nth = (int) std::max<int64_t>(1, std::min<int64_t>(16, rows / 64));
            std::vector<std::thread> th;
            for (int w = 0; w < nth; ++w)
                th.emplace_back([&, w] {
                    const int64_t b0 = (nv / 32) * w / nth, b1 = (nv / 32) * (w + 1) / nth;   // blocks of 32 (never across rows)
                    for (int64_t b = b0; b < b1 && bad.load(std::memory_order_relaxed) < 0; ++b)
                        if (!fp8_block(src + b * 32, q.data() + b * 32, qs[b])) bad.store(b * 32);
                });
            for (auto& x : th) x.join();
            if (bad.load() >= 0) {   // no FP8 values: its BF16 as stored, in a buffer of its own
                void* b16 = nullptr;
                if (cudaMalloc(&b16, (size_t) it.file_bytes) != cudaSuccess ||
                    cudaMemcpy(b16, src, (size_t) it.file_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                    if (b16) cudaFree(b16);
                    return fail(err, "upload failed: " + it.ti->name + " (kept as BF16: no FP8 values)");
                }
                dense_extra.push_back(b16);
                DT d;
                d.d = b16;
                d.type = (int) GGML_TYPE_BF16;
                for (size_t k = 0; k < it.ti->shape.size() && k < 4; ++k) d.ne[k] = (int64_t) it.ti->shape[k];
                d.bytes = it.file_bytes;
                t[it.ti->name] = d;
                ++n_f8_kept;
                continue;
            }
            if (cudaMemcpy(dst, q.data(), (size_t) it.bytes, cudaMemcpyHostToDevice) != cudaSuccess)
                return fail(err, "upload failed: " + it.ti->name);
            DT d;
            d.d = dst;
            d.s = dst + nv;
            d.type = kDtFP8;
            for (size_t k = 0; k < it.ti->shape.size() && k < 4; ++k) d.ne[k] = (int64_t) it.ti->shape[k];
            d.bytes = it.bytes;
            t[it.ti->name] = d;
            ++n_f8;
            f8_saved += it.file_bytes - it.bytes;
            continue;
        }
        if (it.q8) {   // moex (the GGUF set, STRATA_DSV41_DENSE_Q8): BF16 rows -> F32 -> Q8_0 (ggml's reference rows), threaded
            const int64_t ne0 = (int64_t) it.ti->shape[0];
            const size_t rbb = ggml_row_size(GGML_TYPE_BF16, ne0), rb8 = ggml_row_size(GGML_TYPE_Q8_0, ne0);
            const int64_t rows = (int64_t) (it.file_bytes / rbb);
            const uint16_t* src = reinterpret_cast<const uint16_t*>(gm->shard(it.shard).tensor_data(*it.ti));
            std::vector<uint8_t> q((size_t) it.bytes);
            const int nth = (int) std::max<int64_t>(1, std::min<int64_t>(16, rows / 64));
            std::vector<std::thread> th;
            for (int w = 0; w < nth; ++w)
                th.emplace_back([&, w] {
                    const int64_t r0 = rows * w / nth, r1 = rows * (w + 1) / nth;
                    std::vector<float> fv((size_t) (r1 - r0) * ne0);
                    for (size_t i = 0; i < fv.size(); ++i) {
                        const uint32_t b = (uint32_t) src[(size_t) r0 * ne0 + i] << 16;
                        std::memcpy(&fv[i], &b, sizeof(float));
                    }
                    ggml_quantize_chunk(GGML_TYPE_Q8_0, fv.data(), q.data() + (size_t) r0 * rb8, 0, r1 - r0, ne0, nullptr);
                });
            for (auto& x : th) x.join();
            if (cudaMemcpy(dst, q.data(), (size_t) it.bytes, cudaMemcpyHostToDevice) != cudaSuccess)
                return fail(err, "upload failed: " + it.ti->name);
            DT d;
            d.d = dst;
            d.type = (int) GGML_TYPE_Q8_0;
            for (size_t k = 0; k < it.ti->shape.size() && k < 4; ++k) d.ne[k] = (int64_t) it.ti->shape[k];
            d.bytes = it.bytes;
            t[it.ti->name] = d;
            ++n_q8;
            continue;
        }
        if (it.bf16) {   // moex (§62): each F32 value's upper half, once every lower half is checked zero (the same values)
            const size_t nv = (size_t) (it.file_bytes / 4);
            const uint32_t* src = reinterpret_cast<const uint32_t*>(gm->shard(it.shard).tensor_data(*it.ti));
            std::vector<uint16_t> h(nv);
            for (size_t i = 0; i < nv; ++i) {
                if (src[i] & 0xFFFFu)
                    return fail(err, it.ti->name + ": value " + std::to_string(i) + " is not a BF16 one - STRATA_DSV41_BF16_W " +
                                         "needs the dense GGUF's BF16-valued F32 tensors (moex/tools/dsv41/write_dense_gguf.py)");
                h[i] = (uint16_t) (src[i] >> 16);
            }
            if (cudaMemcpy(dst, h.data(), (size_t) it.bytes, cudaMemcpyHostToDevice) != cudaSuccess)
                return fail(err, "upload failed: " + it.ti->name);
            DT d;
            d.d = dst;
            d.type = (int) GGML_TYPE_BF16;
            for (size_t k = 0; k < it.ti->shape.size() && k < 4; ++k) d.ne[k] = (int64_t) it.ti->shape[k];
            d.bytes = it.bytes;
            t[it.ti->name] = d;
            ++n_bf16;
            continue;
        }
        int& fd = fds[it.shard];
        if (fd < 0) fd = open(gm->shard(it.shard).path().c_str(), O_RDONLY | O_DIRECT);
        const uint64_t foff = gm->shard(it.shard).data_start() + it.ti->offset + it.src_off;   // (moex §76: a part's first byte)
        for (uint64_t c = 0; c < it.bytes; c += kPiece) {
            const size_t len = (size_t) std::min<uint64_t>(kPiece, it.bytes - c);
            uint8_t* b = bounce[cur_b];
            if (bev[cur_b]) cudaEventSynchronize(bev[cur_b]);   // its previous piece's copy has landed
            else cudaEventCreateWithFlags(&bev[cur_b], cudaEventDisableTiming);
            const uint64_t a0 = (foff + c) & ~(uint64_t) 4095;
            const size_t want = (size_t) (((foff + c + len + 4095) & ~(uint64_t) 4095) - a0), needb = (size_t) (foff + c - a0) + len;
            size_t got = 0;
            if (fd >= 0)
                while (got < needb) {
                    const ssize_t rd = pread(fd, b + got, want - got, (off_t) (a0 + got));
                    if (rd <= 0) break;
                    got += (size_t) rd;
                }
            if (got < needb) {   // no O_DIRECT here: the mapping (the page cache)
                std::memcpy(b, gm->shard(it.shard).tensor_data(*it.ti) + it.src_off + c, len);
                cudaMemcpy(dst + c, b, len, cudaMemcpyHostToDevice);
                continue;
            }
            if (cudaMemcpyAsync(dst + c, b + (foff + c - a0), len, cudaMemcpyHostToDevice, nullptr) != cudaSuccess)
                return fail(err, "upload failed: " + it.ti->name);
            cudaEventRecord(bev[cur_b], nullptr);
            cur_b ^= 1;
        }
        DT d;
        d.d = dst;
        d.type = (int) it.ti->type;
        for (size_t k = 0; k < it.ti->shape.size() && k < 4; ++k) d.ne[k] = (int64_t) it.ti->shape[k];
        if (it.rows >= 0) d.ne[1] = it.rows;   // (moex §76: a part of the tensor's rows, under its own name)
        d.bytes = it.bytes;
        t[it.alias ? std::string(it.alias) : it.ti->name] = d;
    }
    drain();
    if (std::getenv("STRATA_X_SYNC_CHECK") && std::getenv("STRATA_X_SYNC_CHECK")[0] == '2') {   // debugging: the uploads checked
        for (const auto& it : items)   // (a tensor requantized as it loaded is not the file's bytes)
            if (!it.q6k && !it.bf16 && !it.alias)   // (a tensor's part, §76, is not checked here: verify_dense reads whole tensors)
                dense_hash[it.ti->name] = slices_hash_host(gm->shard(it.shard).tensor_data(*it.ti), it.bytes);
        std::string verr;
        if (!verify_dense(-1, "after the upload", verr)) return fail(err, verr);
        fprintf(stderr, "strata-x: %zu dense tensors read back and verified after the upload\n", dense_hash.size());
    }
    // the dense weights are on the GPUs now: their pages leave the process and the page cache - 7.5 GiB of RAM that, with the
    // expert arena pinned, the kernel would otherwise keep by swapping the desktop out
    for (const auto& it : items) {
        const uintptr_t a = ((uintptr_t) gm->shard(it.shard).tensor_data(*it.ti) + it.src_off) & ~(uintptr_t) 4095;
        const uintptr_t e = ((uintptr_t) gm->shard(it.shard).tensor_data(*it.ti) + it.src_off + it.file_bytes + 4095) & ~(uintptr_t) 4095;
        madvise((void*) a, e - a, MADV_DONTNEED);
    }
    for (size_t s2 = 0; s2 < gm->size(); ++s2)
        if (const int fd = open(gm->shard(s2).path().c_str(), O_RDONLY); fd >= 0) {
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            close(fd);
        }
    if (opt.verbose) {
        fprintf(stderr, "strata-x: %zu dense tensors, %.2f GiB in %.1f s:", items.size(), total / 1073741824.0,
                ms_since(t0) / 1000.0);
        for (size_t di = 0; di < stages.size(); ++di)
            fprintf(stderr, " GPU %d %.2f GiB", stages[di]->dev, totals[di] / 1073741824.0);
        fprintf(stderr, "\n");
        if (n_q6k > 0)
            fprintf(stderr, "strata-x: DENSE Q6_K (STRATA_DSV41_DENSE_Q6K=1): %d Q8_0 matrices requantized as they loaded, %.2f GiB "
                            "less on the GPUs - the output differs from the release's weights\n", n_q6k, q6k_saved / 1073741824.0);
        if (n_bf16 > 0)
            fprintf(stderr, "strata-x: STRATA_DSV41_BF16_W: %d F32 tensors of BF16 values kept as BF16 (the same values)\n", n_bf16);
        if (n_q8 > 0)
            fprintf(stderr, "strata-x: STRATA_DSV41_DENSE_Q8: %d BF16 matrices quantized to Q8_0 as they loaded (our dense GGUF's "
                            "blocks; not the release's exact weights)\n", n_q8);
        if (n_f8 > 0 || n_f8_kept > 0)
            fprintf(stderr, "strata-x: DENSE FP8 (STRATA_DSV41_DENSE_FP8=0: off): %d BF16 matrices back to the release's FP8 values as "
                            "they loaded (every value checked: the same weights), %.2f GiB less on the GPUs%s\n", n_f8,
                            f8_saved / 1073741824.0, n_f8_kept ? (", " + std::to_string(n_f8_kept) + " kept as BF16 (no FP8 values)").c_str() : "");
    }
    size_t sh = 0;
    const strata::TensorInfo* te = gm->find("token_embd.weight", &sh);
    if (!te) return fail(err, "no token_embd.weight");
    embd = gm->shard(sh).tensor_data(*te);
    embd_type = (int) te->type;
    embd_row = ggml_row_size((ggml_type) embd_type, (int64_t) te->shape[0]);
    return true;
}

bool Engine::Impl::bind_layers(std::string& err) {
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
        if (d.type == 0) return true;   // F32 goes through gemv_f32
        if (!kn::native_mmvq_supported(d.type)) return fail(err, n + ": type " + std::to_string(d.type) + " has no GPU kernel");
        return true;
    };
    if (hp.arch == Arch::dsv4) return hp.v41 ? bind_layers_dsv41(err) : bind_layers_dsv4(err);
    if (hp.arch == Arch::hy4) return bind_layers_hy4(err);
    if (hp.arch == Arch::ernie) return bind_layers_ernie(err);
    if (!get("output_norm.weight", output_norm, true) || !get("output.weight", output, true)) return false;
    if (!proj_ok(output, "output.weight")) return false;
    L.resize((size_t) n_load);
    if (hp.arch == Arch::mimo2) {   // mimo2.cpp load_arch_tensors (the fused QKV form; the MTP blocks' dense FFN)
        for (int64_t il = 0; il < n_load; ++il) {
            Layer& l = L[(size_t) il];
            auto n = [&](const char* sfx) { return "blk." + std::to_string(il) + "." + sfx; };
            bool ok = get(n("attn_norm.weight"), l.attn_norm, true) && get(n("ffn_norm.weight"), l.ffn_norm, true) &&
                      get(n("attn_qkv.weight"), l.wqkv, true) && get(n("attn_output.weight"), l.wo, true) &&
                      get(n("attn_sinks.weight"), l.sinks, false);
            if (!ok || !proj_ok(l.wqkv, n("attn_qkv")) || !proj_ok(l.wo, n("attn_output"))) return false;
            if (l.wqkv.n_in() != hp.n_embd || l.wqkv.n_out() != hp.qkv_width(il)) return fail(err, n("attn_qkv.weight") + ": shape");
            if (l.wo.n_in() != hp.n_head * hp.hd_v || l.wo.n_out() != hp.n_embd) return fail(err, n("attn_output.weight") + ": shape");
            if (l.sinks && (l.sinks.type != 0 || l.sinks.ne[0] != hp.n_head)) return fail(err, n("attn_sinks.weight") + ": not F32 x n_head");
            if (il >= hp.n_layer) {
                ok = get(n("nextn.eh_proj.weight"), l.eh_proj, true) && get(n("nextn.enorm.weight"), l.enorm, true) &&
                     get(n("nextn.hnorm.weight"), l.hnorm, true) && get(n("layer_output_norm.weight"), l.out_norm, false) &&
                     proj_ok(l.eh_proj, n("nextn.eh_proj"));
                if (!ok) return false;
                if (!l.out_norm) l.out_norm = output_norm;   // mimo2.cpp graph_mtp: shared_head_norm -> layer_out_norm -> output_norm
            }
            if (hp.is_moe(il)) {
                ok = get(n("ffn_gate_inp.weight"), l.gate_inp, true) && get(n("exp_probs_b.bias"), l.probs_b, false);
                if (!ok || l.gate_inp.type != 0) return fail(err, n("ffn_gate_inp is missing or not F32"));
            } else {
                ok = get(n("ffn_gate.weight"), l.ffn_gate, true) && get(n("ffn_up.weight"), l.ffn_up, true) &&
                     get(n("ffn_down.weight"), l.ffn_down, true);
                if (!ok || !proj_ok(l.ffn_gate, n("ffn_gate")) || !proj_ok(l.ffn_up, n("ffn_up")) || !proj_ok(l.ffn_down, n("ffn_down")))
                    return false;
            }
        }
        return true;
    }
    for (int64_t il = 0; il < n_load; ++il) {
        Layer& l = L[(size_t) il];
        auto n = [&](const char* sfx) { return "blk." + std::to_string(il) + "." + sfx; };
        const bool trunk = il < hp.n_layer;
        bool ok = get(n("attn_norm.weight"), l.attn_norm, true) && get(n("ffn_norm.weight"), l.ffn_norm, true);
        if (trunk)
            ok = ok && get(n("hc_attn_fn.weight"), l.hc_attn_fn, true) && get(n("hc_attn_base.weight"), l.hc_attn_base, true) &&
                 get(n("hc_attn_scale.weight"), l.hc_attn_scale, true) && get(n("hc_ffn_fn.weight"), l.hc_ffn_fn, true) &&
                 get(n("hc_ffn_base.weight"), l.hc_ffn_base, true) && get(n("hc_ffn_scale.weight"), l.hc_ffn_scale, true);
        else
            ok = ok && get(n("nextn.eh_proj.weight"), eh_proj, true) && get(n("nextn.enorm.weight"), enorm, true) &&
                 get(n("nextn.hnorm.weight"), hnorm, true) && get(n("nextn.shared_head_norm.weight"), shead_norm, true) &&
                 proj_ok(eh_proj, n("nextn.eh_proj.weight"));
        if (!ok) return false;
        if (hp.is_dsa(il)) {
            ok = get(n("attn_q_a.weight"), l.q_a, true) && get(n("attn_q_a_norm.weight"), l.q_a_norm, true) &&
                 get(n("attn_q_b.weight"), l.q_b, true) && get(n("attn_kv_a_mqa.weight"), l.kv_a, true) &&
                 get(n("attn_kv_a_norm.weight"), l.kv_a_norm, true) && get(n("attn_k_b.weight"), l.k_b, true) &&
                 get(n("attn_v_b.weight"), l.v_b, true) && get(n("attn_output.weight"), l.wo, true) &&
                 get(n("indexer.k_norm.weight"), l.idx_k_norm, true) && get(n("indexer.k_norm.bias"), l.idx_k_norm_b, true) &&
                 get(n("indexer.proj.weight"), l.idx_proj, true) && get(n("indexer.attn_k.weight"), l.idx_k, true) &&
                 get(n("indexer.attn_q_b.weight"), l.idx_q_b, true) &&
                 get(n("indexer_compressor_gate.weight"), l.idx_gate, true) &&
                 get(n("indexer_compressor_ape.weight"), l.idx_ape, true);
            if (!ok) return false;
            for (auto* d : {&l.q_a, &l.q_b, &l.kv_a, &l.k_b, &l.v_b, &l.wo, &l.idx_k, &l.idx_q_b, &l.idx_gate})
                if (!proj_ok(*d, n("(DSA projection)"))) return false;
            if (l.idx_proj.type != 0) return fail(err, n("indexer.proj.weight") + " is not F32");
        } else {
            ok = get(n("attn_q.weight"), l.wq, true) && get(n("attn_k.weight"), l.wk, true) &&
                 get(n("attn_v.weight"), l.wv, true) && get(n("ssm_conv1d_q.weight"), l.conv_q, true) &&
                 get(n("ssm_conv1d_k.weight"), l.conv_k, true) && get(n("ssm_conv1d_v.weight"), l.conv_v, true) &&
                 get(n("ssm_f_a.weight"), l.f_a, true) && get(n("ssm_f_b.weight"), l.f_b, true) &&
                 get(n("ssm_beta.weight"), l.beta, true) && get(n("ssm_a"), l.a, true) &&
                 get(n("ssm_dt.bias"), l.dt_b, true) && get(n("ssm_g_a.weight"), l.g_a, true) &&
                 get(n("ssm_g_b.weight"), l.g_b, true) && get(n("ssm_norm.weight"), l.o_norm, true) &&
                 get(n("attn_output.weight"), l.wo, true);
            if (!ok) return false;
            for (auto* d : {&l.wq, &l.wk, &l.wv, &l.f_a, &l.f_b, &l.beta, &l.g_a, &l.g_b, &l.wo})
                if (!proj_ok(*d, n("(KDA projection)"))) return false;
            if (l.conv_q.type != 0 || l.a.type != 0 || l.dt_b.type != 0) return fail(err, n("KDA conv/A/dt are not F32"));
        }
        if (!hp.is_moe(il)) {
            ok = get(n("ffn_gate.weight"), l.ffn_gate, true) && get(n("ffn_up.weight"), l.ffn_up, true) &&
                 get(n("ffn_down.weight"), l.ffn_down, true);
            if (!ok || !proj_ok(l.ffn_gate, n("ffn_gate")) || !proj_ok(l.ffn_up, n("ffn_up")) || !proj_ok(l.ffn_down, n("ffn_down")))
                return false;
        } else {
            ok = get(n("ffn_gate_inp.weight"), l.gate_inp, true) && get(n("exp_probs_b.bias"), l.probs_b, true) &&
                 get(n("ffn_gate_shexp.weight"), l.sh_gate, true) && get(n("ffn_up_shexp.weight"), l.sh_up, true) &&
                 get(n("ffn_down_shexp.weight"), l.sh_down, true);
            if (!ok || l.gate_inp.type != 0) return fail(err, n("ffn_gate_inp is missing or not F32"));
            for (auto* d : {&l.sh_gate, &l.sh_up, &l.sh_down})
                if (!proj_ok(*d, n("(shared expert)"))) return false;
        }
    }
    return true;
}

/// MiMo's decode state: per layer the K and V caches (FP16) - the context on a full-attention layer, on a sliding-window
/// layer a ring of the window plus the longest batch written before it is read (a decode window, a prompt sub-batch).
bool Engine::Impl::alloc_state_mimo(std::string& err) {
    const int64_t n = hp.n_embd;
    const int64_t tw = std::max<int64_t>(kTmax, opt.prefill_sub);
    swa_rows = std::min<int64_t>(opt.n_ctx, (hp.n_swa + tw + 63) / 64 * 64);
    // a full layer's keys in splits per decode window: the tensor-core kernel runs one warp per (split, KV head, token) with
    // 16-key tiles (4 warps an SM), so 64 splits are one wave of 256 warps at one token - measured at 131K keys: 0.57 ms a
    // layer against 0.60 / 0.71 at 128 / 256 splits and 0.82 for 32-key tiles at 128 (STRATA_X_GQA_CHUNKS=N to change; the
    // FP32 fallback takes the same split)
    gqa_chunks = 128;   // gqa_decode_shared: a row's chunks of 2048 keys at 131K, ~260 blocks a full layer (64: 4096, half that)
    if (const char* e = std::getenv("STRATA_X_GQA_CHUNKS")) gqa_chunks = std::max(2, std::min(1024, std::atoi(e)));
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
        l.kv_rows = hp.is_swa(il) ? swa_rows : opt.n_ctx;
        const int64_t nkv = hp.n_kv(il);
        if (!(l.k_cache = g.dalloc<uint16_t>((size_t) (l.kv_rows * nkv * hp.hd_k), err)) ||
            !(l.v_cache = g.dalloc<uint16_t>((size_t) (l.kv_rows * nkv * hp.hd_v), err)))
            return false;
        bytes += (uint64_t) l.kv_rows * nkv * (hp.hd_k + hp.hd_v) * 2;
    }
    reset_dev_cache();
    if (opt.verbose)
        fprintf(stderr, "strata-x: KV caches %.1f MiB for %lld positions (sliding-window rings of %lld rows)\n", bytes / 1048576.0,
                (long long) opt.n_ctx, (long long) swa_rows);
    return true;
}

bool Engine::Impl::alloc_state(std::string& err) {
    if (hp.arch == Arch::mimo2) return alloc_state_mimo(err);
    if (hp.arch == Arch::hy4) return alloc_state_hy4(err);
    if (hp.arch == Arch::ernie) return alloc_state_ernie(err);
    if (hp.arch == Arch::dsv4 && hp.v41) return alloc_state_dsv41(err);   // its score stride: dsv41_score_rows()
    if (hp.arch == Arch::dsv4) {
        lid_rows = (int) (opt.n_ctx / 4 + 1);   // carve_into's score stride
        return alloc_state_dsv4(err);
    }
    const int64_t n = hp.n_embd, H = hp.n_head, D = hp.hd_kda, din = H * D;
    const int64_t Lkv = hp.kv_lora, ID = hp.idx_dim;
    max_pool = (int) (opt.n_ctx / hp.kpool + 1);
    max_cells = (int) (hp.idx_topk + hp.kpool);
    const int64_t tw = std::max<int64_t>(kTmax, opt.prefill_sub);
    ring_rows = (int) (((tw + hp.kpool + hp.kpool - 1) / hp.kpool) * hp.kpool);
    for (auto& gp : stages)
        if (!gp->alloc(err) || !gp->alloc_decode(err)) return false;
    if (cudaMallocHost(&h_x, (size_t) kTmax * n * sizeof(float)) != cudaSuccess ||
        cudaMallocHost(&h_parts, (size_t) kTmax * hp.n_used * n * sizeof(float)) != cudaSuccess ||
        cudaMallocHost(&h_ids, (size_t) kTmax * 64 * sizeof(int32_t)) != cudaSuccess ||
        cudaMallocHost(&h_amax, (size_t) kTmax * sizeof(int32_t)) != cudaSuccess)
        return fail(err, "pinned host buffers");
    uint64_t state_bytes = 0;
    for (int64_t il = 0; il < n_load; ++il) {
        Layer& l = L[(size_t) il];
        Stage& g = stage_of(il);
        cudaSetDevice(g.dev);
        auto F = [&](float*& p, int64_t cnt) { return (p = g.dalloc<float>((size_t) cnt, err)) != nullptr; };
        if (hp.is_dsa(il)) {
            if (!(l.kv_cache = g.dalloc<uint16_t>((size_t) (opt.n_ctx * Lkv), err)) || !F(l.ik_ring, ring_rows * ID) ||
                !F(l.ig_ring, ring_rows * ID) || !F(l.pooled, (int64_t) max_pool * ID))
                return false;
            state_bytes += opt.n_ctx * Lkv * 2 + (uint64_t) max_pool * ID * 4;
        } else {
            const int64_t cs = 3 * din * (hp.d_conv - 1);
            if (!F(l.conv, cs) || !F(l.S, D * H * D)) return false;
            state_bytes += (cs + D * H * D) * 4;
            if (snap_T > 0) {
                if (!F(l.snaps, (int64_t) snap_T * D * H * D) || !F(l.conv_prev, cs) || !F(l.conv_in, (int64_t) snap_T * 3 * din))
                    return false;
                state_bytes += ((uint64_t) snap_T * (D * H * D + 3 * din) + cs) * 4;
            }
        }
    }
    reset_dev_cache();
    if (opt.verbose)
        fprintf(stderr, "strata-x: decode state %.1f MiB for %lld positions%s\n", state_bytes / 1048576.0,
                (long long) opt.n_ctx, snap_T ? " (with MTP rollback snapshots)" : "");
    return true;
}

// ------------------------------------------------------------------ the VRAM expert tier

// ------------------------------------------------------------------ decode windows

/// The window's residual streams (T, 4, n) from one GPU to the next: the source stream's work so far is recorded,
/// the destination waits for it, and the copy (peer, or staged by the driver) runs on the destination's stream.
bool Engine::Impl::handoff(Stage& from, Stage& to, int T) {
    set_dev(from.dev);
    cudaEventRecord(from.hand, from.s);
    set_dev(to.dev);
    cudaStreamWaitEvent(to.s, from.hand, 0);
    if (cudaMemcpyPeerAsync(to.R, to.dev, from.R, from.dev, (size_t) T * hp.hc * hp.n_embd * sizeof(float), to.s) != cudaSuccess)
        return false;
    // DeepSeek-V4.1: the next layer's attention reads the streams with the last FFN's pre-mix
    return !hp.v41 || cudaMemcpyPeerAsync(to.cpre, to.dev, from.cpre, from.dev, (size_t) T * 24 * sizeof(float), to.s) == cudaSuccess;
}

/// The stages' device copies of the residency table (stage-local slot of (layer, expert), or -1): what the captured
/// graphs choose their VRAM hits by.  The host's `resident` is the master; the host classifies the routed pairs by it, so
/// the two decisions are the same as long as this ran after every change.
void Engine::Impl::sync_residency() {
    if (!res_dirty) return;
    const int64_t E = hp.n_expert;
    for (auto& gp : stages) {
        Stage& g = *gp;
        if (!g.in_res_h) continue;
        for (int64_t il = hp.n_dense; il < n_load; ++il) {
            if (ldev[(size_t) il] != g.di) continue;
            for (int64_t e = 0; e < E; ++e) {
                const int32_t slot = resident[(size_t) (il * E + e)];
                g.in_res_h[il * E + e] = slot < 0 || slot_dev[(size_t) slot] != g.di ? -1
                                         : slot >= n_slots ? (int32_t) (slot - n_slots)   // the drafter's region
                                         : !g.slot_off.empty() ? (int32_t) (g.slot_off[(size_t) (slot - g.slot_base)] / kSlotUnit)   // §24
                                                           : (int32_t) (slot - g.slot_base);
            }
        }
    }
    res_dirty = false;
}

/// The device table of the slices no window's first node loads: the NextN layer's (its experts pinned at load), copied once.
void Engine::Impl::push_static_res() {
    const int64_t E = hp.n_expert;
    sync_residency();
    res_dirty = true;   // the windows' slices still go through window_load
    if (!mtp_on || hp.arch != Arch::glm5) return;
    Stage& lg = last_stage();
    set_dev(lg.dev);
    cudaMemcpy(lg.d_res + hp.n_layer * E, lg.in_res_h + hp.n_layer * E, (size_t) E * sizeof(int32_t), cudaMemcpyHostToDevice);
}

/// A decode window through the stages' captured graphs.  Everything the host decides it decides in service_moe from
/// the doorbell; within each MoE layer the GPU's shared expert and VRAM hits run while the CPU pool computes the misses (the
/// chain is serial across layers, as Strata's).  The window's inputs go
/// through mapped memory (the graphs' first node reads them): the copy engines carry only the tier's swaps.
bool Engine::Impl::forward_graph(const int32_t* toks, int T, int64_t pos0, bool keep, std::string& err) {
    sync_residency();
    for (auto& gp : stages) {
        gp->use(gp->win);
        *gp->in_pos_h = pos0;
    }
    Stage* g0 = &stage_of(0);
    if (!g0->embd_rows(toks, T, g0->in_rows_h, pos0)) return fail(err, "token out of range");
    if (eg) {   // DeepSeek-V4.1's engram: the window's rows (mapped; the first stage's graph reads them)
        eg_note(toks, pos0, T);
        if (eg_late) {   // moex (DSV41_PLAN §42): on the helper, waited for before the first engram layer is served
            std::string e0;
            eg_help_wait(e0);   // (a window that failed before its wait: its job done before this one's)
            eg_help_post(pos0, T, g0->eg_win_h, kTmax, 0);
        } else if (!eg_rows(pos0, T, g0->eg_win_h, kTmax, 0, err)) {
            return false;
        }
    }
    if (g0->in_tok_h) std::memcpy(g0->in_tok_h, toks, (size_t) T * sizeof(int32_t));   // DeepSeek-V4's hash layers
    if (heg && !hy4_eg_window(toks, pos0, T, err)) return false;   // Hy4-flash's Engram: the rows, queued before the launches
    Stage* prev = nullptr;
    const int key = T * 2 + (keep ? 1 : 0);
    for (auto& gp : stages) {
        Stage& g = *gp;
        if (prev) {   // the next GPU's graph starts once the previous one has written its residuals (mapped memory)
            set_dev(prev->dev);
            cudaEventRecord(prev->hand, prev->s);
            set_dev(g.dev);
            cudaStreamWaitEvent(g.s, prev->hand, 0);
        }
        set_dev(g.dev);
        if (cudaGraphLaunch(g.gx[key], g.s) != cudaSuccess) return fail(err, "graph launch failed on GPU " + std::to_string(g.dev));
        // moex (§76, eg_ring): the graph's wait for the engram rows is in the layer before the first engram layer - released by
        // the fetch's thread as it ends (or here, if it has ended), not by this thread after that layer's CPU experts
        if (eg_ring && eg && &g == stages[0].get()) eg_help_arm(g);
        for (int64_t il = g.lb; il < g.le; ++il) {
            if (eg_late && eg && il == eg_late_layer) {   // moex (§42): the graph waits here for the rows (layer_graph_body)
                if (!eg_help_wait(err)) {
                    for (auto& x : stages) x->release_gpu_waits();
                    return false;
                }
                if (!eg_ring) *g.db_egf = ++g.host_egf;
            }
            if (hp.is_moe(il) && !g.service_moe((int) il, T, err)) {
                for (auto& x : stages) x->release_gpu_waits();
                return false;
            }
        }
        prev = &g;
    }
    Stage& lastg = last_stage();
    set_dev(lastg.dev);
    const auto th = clk::now();
    lastg.head(T, false);
    if (dfy) lastg.dfly_capture_h(pos0, T);   // DFly2: the normed rows its next step fuses
    stages[0]->st.ms_head += ms_since(th);
    if (sync_check)
        for (auto& gp : stages) {
            set_dev(gp->dev);
            if (!gp->sync_check("decode window (graph)", (int) gp->lb, err)) return false;
        }
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return fail(err, cudaGetErrorString(ce));
    last_T = T;
    return true;
}

bool Engine::Impl::forward(const int32_t* toks, int T, int64_t pos0, bool keep, std::string& err) {
    const int n = (int) hp.n_embd;
    if (T < 1 || T > kTmax) return fail(err, "window of " + std::to_string(T) + " tokens");
    if (rec && T != 1) return fail(err, "probe mode checks one token at a time");
    if (pos0 + T > opt.n_ctx) return fail(err, "position past the context");
    // moex (STRATA_HY4_KV_GROW): the sparse layers' caches mapped through this window (and DFly2's block after it)
    if ((hp.arch == Arch::hy4 || hp.arch == Arch::ernie) && kv_grow && pos0 + T + kTmax + 16 > kv_rows_mapped &&
        !hy4_kv_ensure(pos0 + T + kTmax + 16, err))
        return false;
    // only GLM-5.3's KDA layers keep per-row snapshots for a rollback; MiMo's and DeepSeek-V4's state is rewritten in place
    if (keep && T > snap_T && hp.arch == Arch::glm5) return fail(err, "an MTP window longer than the rollback snapshots");
    for (int t = 0; t < T; ++t)
        if (toks[t] < 0 || toks[t] >= hp.n_vocab) return fail(err, "token out of range");
    for (auto& gp : stages) gp->use(gp->win);
    if (graphs_on && !rec && T <= kTmax) {   // captured graphs, once every stage has a graph for this window variant
        bool ready = true;
        for (auto& gp : stages) {
            std::string cerr;
            const bool r = gp->graph_ready(T, keep, cerr);
            if (!cerr.empty()) {   // a capture failed: eager from here on, and say why
                fprintf(stderr, "strata-x: decode graphs off: %s\n", cerr.c_str());
                graphs_on = false;
            }
            ready = ready && r;
        }
        if (graphs_on && ready) return forward_graph(toks, T, pos0, keep, err);
    }
    Stage* g = &stage_of(0);
    set_dev(g->dev);
    if (!g->upload_embd(toks, T, g->cur, pos0)) return fail(err, "token out of range");
    if (eg) {   // DeepSeek-V4.1's engram: the window's rows (the first stage's mapped buffer, as the graphs read them)
        cudaStreamSynchronize(g->s);   // an earlier window's kernels may still read the buffer
        if (!eg_help_wait(err)) return false;   // (moex, §42: ... and the helper's job, STRATA_DSV41_EG_LATE)
        eg_note(toks, pos0, T);
        if (!eg_rows(pos0, T, g->eg_win_h, kTmax, 0, err)) return false;
    }
    if (heg && !hy4_eg_window(toks, pos0, T, err)) return false;   // Hy4-flash's Engram: the window's rows on the stages
    set_dev(g->dev);
    if (hp.arch == Arch::dsv4 && !g->upload_toks(toks, T, err)) return false;   // the hash layers' ids (the first stage's layers)
    g->probe("inp_embd", -1, g->cur, n);
    for (int hh = 0; hh < (int) hp.hc; ++hh)
        cudaMemcpy2DAsync(g->R + (size_t) hh * n, (size_t) hp.hc * n * sizeof(float), g->cur, (size_t) n * sizeof(float),
                          (size_t) n * sizeof(float), (size_t) T, cudaMemcpyDeviceToDevice, g->s);
    if (hp.hc > 1) g->probe("hc_init", -1, g->R, n, hp.hc);
    if (hp.v41) g->v41_pre_reset(T);   // DeepSeek-V4.1: layer 0's attention reads stream 0
    try {
        for (int il = 0; il < (int) hp.n_layer; ++il) {
            Stage& gl = stage_of(il);
            if (&gl != g) {
                if (!handoff(*g, gl, T)) return fail(err, "the residual hand-off failed");
                g = &gl;
            }
            set_dev(g->dev);
            if (!g->layer_window(il, T, pos0, keep, err)) return false;
            if (!g->sync_check("decode window (eager)", il, err)) return false;
        }
        Stage& lastg = last_stage();
        if (&lastg != g) {
            if (!handoff(*g, lastg, T)) return fail(err, "the residual hand-off failed");
            g = &lastg;
        }
        set_dev(g->dev);
        g->dspark_capture((int) hp.n_layer, 0, T);   // DSpark: the trunk's output as a target layer's features
        g->dflash_capture((int) hp.n_layer, 0, T, false);   // DFlash: ... and its
        g->head(T, false);
        if (dfy) g->dfly_capture_h(pos0, T);   // DFly2: the normed rows its next step fuses
    } catch (const std::exception& e) {
        return fail(err, e.what());
    }
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return fail(err, cudaGetErrorString(ce));
    last_T = T;
    return true;
}

}  // namespace strata::x

namespace strata::x {

Engine::Engine() : p_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

const Hparams& Engine::hparams() const { return p_->hp; }

const EngineStats& Engine::stats() const {
    Impl& m = *p_;
    EngineStats& s = m.st_sum;
    s = EngineStats{};
    for (const auto& gp : m.stages) {
        const EngineStats& x = gp->st;
        s.tokens = std::max(s.tokens, x.tokens);
        s.ms_total = std::max(s.ms_total, x.ms_total);
        s.ms_pool += x.ms_pool;
        s.ms_wait += x.ms_wait;
        s.ms_head += x.ms_head;
        s.ms_upkeep += x.ms_upkeep;
        s.ms_up_final += x.ms_up_final;
        s.ms_up_fill += x.ms_up_fill;
        s.ms_up_verify += x.ms_up_verify;
        s.ms_up_round += x.ms_up_round;
        s.ms_up_pen += x.ms_up_pen;
        s.ms_up_khash += x.ms_up_khash;
        s.up_khash += x.up_khash;
        s.hits += x.hits;
        s.misses += x.misses;
        s.ms_eg += x.ms_eg;
        s.cpu_ram += x.cpu_ram;
        s.cpu_ssd += x.cpu_ssd;
        s.ms_ssd_wait += x.ms_ssd_wait;
        s.swaps += x.swaps;
        s.fills += x.fills;
        s.slots += x.slots;
        s.streamed += x.streamed;
        s.ms_prefill_moe += x.ms_prefill_moe;
        s.ms_prefill_wait += x.ms_prefill_wait;
        s.ms_prefill_cpu += x.ms_prefill_cpu;
        s.prefill_cpu_experts += x.prefill_cpu_experts;
        s.prefill_share_experts += x.prefill_share_experts;
        s.ms_prefill = std::max(s.ms_prefill, x.ms_prefill);
        s.tokens_prefill = std::max(s.tokens_prefill, x.tokens_prefill);
        s.prefill_chunk = std::max(s.prefill_chunk, x.prefill_chunk);
        s.prefill_ring = std::max(s.prefill_ring, x.prefill_ring);
        s.lent_slots += x.lent_slots;
    }
    s.cpu_ram += m.ssd_st.cpu_ram;
    s.cpu_ssd += m.ssd_st.cpu_ssd;
    s.ms_ssd_wait += m.ssd_st.ms_ssd_wait;
    if (m.rio) {
        s.ssd_reads = m.rio->reads;
        s.ssd_hits = m.rio->hits;
        s.ahead_issued = m.rio->ahead_issued;
        s.ahead_used = m.rio->ahead_used;
        s.ahead_wasted = m.rio->ahead_wasted;
        s.fill_cached = m.rio->fill_cached;
        s.fill_reads = m.rio->fill_reads;
        s.rec_reads = m.rio->rec_reads.load();
        s.rec_cached = m.rio->rec_cached.load();
        s.prompt_kept = m.rio->kept_total.load();
        s.prompt_adm_free = m.rio->adm_free.load();
        s.prompt_adm_hot = m.rio->adm_hot.load();
        s.prompt_sync = m.rio->prompt_sync.load();
        s.prompt_orphan = m.rio->prompt_orphan.load();
        s.ms_idle_lend = m.rio->us_lend.load() / 1000.0;
        s.ms_idle_gate = m.rio->us_gate.load() / 1000.0;
        for (size_t si = 0; si < m.stages.size() && si < 2; ++si) s.ms_idle_gate_dev[si] = m.stages[si]->us_gate_own.load() / 1000.0;
        s.ms_idle_ring = m.rio->us_ring.load() / 1000.0;
        s.ms_idle_bounce = m.rio->us_bounce.load() / 1000.0;
        s.ms_idle_refill = m.rio->us_refill.load() / 1000.0;
        s.ms_eg_rows = m.us_eg_rows.load() / 1000.0;
        s.eg_dedup_asked = (int64_t) m.eg_rows_asked.load();
        s.eg_dedup_read = (int64_t) m.eg_rows_read.load();
    }
    if (m.heg) {   // moex: Hy4-flash's Engram row cache (no record files: outside DeepSeek-V4.1's block above)
        s.hy4_eg_hits = m.heg->cache.hits.load(std::memory_order_relaxed);
        s.hy4_eg_misses = m.heg->cache.misses.load(std::memory_order_relaxed);
    }
    if (m.eg)   // moex (#59): the n-gram reader's counters, every table summed (none without STRATA_DSV41_EG_READER)
        for (const auto* v : {&m.eg->rd_w, &m.eg->rd_s})
            for (const auto& r : *v) {
                const strata::ngram::ReaderStats rs = r->snapshot();
                s.eg_reader_rows += (int64_t) rs.requests;
                s.eg_reader_hits += (int64_t) rs.cache_hits;
                s.eg_reader_reads += (int64_t) rs.reads;
            }
    s.slot_mib = (double) m.slot_bytes / 1048576.0;
    if (m.pool) {
        s.pool_gu_ms = m.pool->ms_multi_gu;
        s.pool_q_ms = m.pool->ms_multi_q;
        s.pool_down_ms = m.pool->ms_multi_down;
        s.pool_bytes = m.pool->multi_bytes;
        s.pool_workers = m.pool->workers();
        m.pool->phase_ms(s.pool_park_ms, s.pool_drain_ms, s.pool_repark_ms);
    }
    return s;
}

void Engine::set_recorder(moex::Recorder* r) { p_->rec = r; }

/// STRATA'S --layer-split auto (generate.cpp: a predicted window time per placement, an exhaustive search) for two GPUs.  For
/// every first layer K of the second GPU: what each GPU would have left for expert slots - its free VRAM minus its reserve, its
/// layers' dense weights and decode state, and on the last GPU the head and the drafter's region - the expert profile filled
/// into those slots hottest first, as init_cache fills them, and the routed mass they would hold.  Strata's per-layer term
/// (layer_ms x layers) is the same for every K on two identical cards, so K maximising the held mass wins; Strata weighs ranks
/// by a fitted curve because its profiles carry only a ranking - GLM's profile carries the routing counts, so the mass is
/// measured, not modelled.  No profile: the halves, as before.
int64_t Engine::Impl::choose_split(const std::vector<int>& devs) {
    const int64_t NL = hp.n_layer, NE = hp.n_expert, half = NL / 2;
    if (devs.size() != 2 || opt.expert_profile.empty()) return half;
    FILE* fp = fopen(opt.expert_profile.c_str(), "r");
    if (!fp) return half;
    struct Hot {
        int64_t il;
        double c;
    };
    std::vector<Hot> prof;
    char line[256];
    while (fgets(line, sizeof line, fp)) {
        long long il, e;
        double c;
        if (line[0] != '#' && sscanf(line, "%lld %lld %lf", &il, &e, &c) == 3 && il >= hp.n_dense && il < NL && e >= 0 && e < NE)
            prof.push_back({il, c});
    }
    fclose(fp);
    if (prof.empty()) return half;
    // per layer: dense bytes on its GPU, decode state, the expert blob (the routed experts' native format)
    std::vector<uint64_t> dense((size_t) NL + 2, 0), state((size_t) NL + 1, 0), blob((size_t) NL + 1, 0);
    uint64_t head = 0;
    for (size_t sh = 0; sh < gm->size(); ++sh)
        for (const auto& ti : gm->shard(sh).tensors()) {
            const std::string& nm = ti.name;
            if (nm == "token_embd.weight" || (nm.size() > 12 && nm.compare(nm.size() - 12, 12, "_exps.weight") == 0)) continue;
            const uint64_t b = (strata::tensor_payload_bytes(ti) + 255) & ~(uint64_t) 255;
            if (nm.rfind("blk.", 0) == 0) {
                const int64_t il = std::strtoll(nm.c_str() + 4, nullptr, 10);
                if (il >= 0 && il < NL) dense[(size_t) il] += b;
                else if (il == NL && mtp_on && !dfl && !dfy) head += b;   // the NextN block: the last GPU's
            } else {
                head += b;   // the head and its norm
            }
        }
    const int64_t H = hp.n_head, D = hp.hd_kda, din = H * D, Lkv = hp.kv_lora, ID = hp.idx_dim;
    const int64_t tw = std::max<int64_t>(kTmax, opt.prefill_sub);
    const int64_t mp = hp.kpool ? opt.n_ctx / hp.kpool + 1 : 0, rr = hp.kpool ? ((tw + hp.kpool + hp.kpool - 1) / hp.kpool) * hp.kpool : 0;
    const int64_t swa_r = std::min<int64_t>(opt.n_ctx, (hp.n_swa + tw + 63) / 64 * 64);
    for (int64_t il = 0; il < NL; ++il) {
        if (hp.arch == Arch::mimo2) state[(size_t) il] = (uint64_t) (hp.is_swa(il) ? swa_r : opt.n_ctx) * hp.n_kv(il) * (hp.hd_k + hp.hd_v) * 2;
        else if (hp.arch == Arch::dsv4) state[(size_t) il] = hp.v41 ? dsv41_state_bytes(il) : dsv4_state_bytes(il);
        else if (hp.arch == Arch::ernie) state[(size_t) il] = ernie_state_bytes(il);
        else if (hp.arch == Arch::hy4)   // the MLA cache rows of 576 (the context, or the sliding-window ring), the indexer's state
            state[(size_t) il] = (uint64_t) (hp.is_swa(il) ? swa_r : opt.n_ctx) * 576 * 2 + (hp.is_swa(il) ? 0 : (uint64_t) (2 * rr + mp) * ID * 4);
        else if (hp.is_dsa(il)) state[(size_t) il] = (uint64_t) opt.n_ctx * Lkv * 2 + (uint64_t) (2 * rr + mp) * ID * 4;
        else {
            const int64_t cs = 3 * din * (hp.d_conv - 1);
            state[(size_t) il] = (uint64_t) (cs + D * H * D) * 4 + (snap_T > 0 ? ((uint64_t) snap_T * (D * H * D + 3 * din) + cs) * 4 : 0);
        }
        if (!hp.is_moe(il)) continue;
        size_t sg = 0, sd = 0;
        const strata::TensorInfo* g = gm->find("blk." + std::to_string(il) + ".ffn_gate_exps.weight", &sg);
        const strata::TensorInfo* d = gm->find("blk." + std::to_string(il) + ".ffn_down_exps.weight", &sd);
        kc::NativeFmt f;
        std::string ferr;
        if (g && d && kc::native_fmt((int) g->type, (int) d->type, hp.ex(), hp.n_ff_exp, f, ferr)) blob[(size_t) il] = (f.bytes + 255) & ~(size_t) 255;
    }
    if (mtp_on && hp.arch == Arch::glm5) {   // the drafter's region and its buffers ride on the last GPU
        size_t sg = 0, sd = 0;
        const strata::TensorInfo* g = gm->find("blk." + std::to_string(NL) + ".ffn_gate_exps.weight", &sg);
        const strata::TensorInfo* d = gm->find("blk." + std::to_string(NL) + ".ffn_down_exps.weight", &sd);
        kc::NativeFmt f;
        std::string ferr;
        if (g && d && kc::native_fmt((int) g->type, (int) d->type, hp.n_embd, hp.n_ff_exp, f, ferr)) head += (uint64_t) NE * ((f.bytes + 255) & ~(size_t) 255);
        head += (uint64_t) kTmax * hp.n_vocab * 4 * 2;   // the drafter's logits beside the head's
    }
    head += (uint64_t) kTmax * hp.n_vocab * 4;   // the head's logits
    head += dflash_vram;                          // MiMo's DFlash drafter: its weights, rings and buffers ride on the last GPU
    head += dfly_vram;                            // moex: Hy4-flash's DFly2 drafter, the same
    // the slot size init_cache will pick: the most experts per byte (a mixed GGUF's larger-blob layers stay on the CPU)
    uint64_t slot = 0;
    {
        std::map<uint64_t, int64_t> by_size;
        for (int64_t il = hp.n_dense; il < NL; ++il)
            if (blob[(size_t) il]) by_size[blob[(size_t) il]] += NE;
        double best_score = -1;
        int64_t covered = 0;
        for (const auto& [sz, cnt] : by_size) {
            covered += cnt;
            if ((double) covered / (double) sz > best_score) {
                best_score = (double) covered / (double) sz;
                slot = sz;
            }
        }
    }
    if (slot == 0) return half;
    // what each GPU has now, before anything of the model is on it, minus its reserve and a per-stage allowance for what else
    // a stage allocates before the slots (the window scratch, the MoE and doorbell buffers, cuBLAS, the allocator's rounding of
    // the per-layer state): 319 / 343 MiB measured on the two stages (ctx 16K; the prediction then matches init_cache's slots)
    uint64_t free_b[2] = {0, 0};
    for (int i = 0; i < 2; ++i) {
        size_t fr = 0, tot = 0;
        cudaSetDevice(devs[(size_t) i]);
        cudaMemGetInfo(&fr, &tot);
        const uint64_t reserve = (uint64_t) (i == 0 ? opt.vram_reserve_mib : opt.vram_reserve_later_mib) << 20;
        const uint64_t misc = 336ull << 20;
        free_b[i] = fr > reserve + misc ? fr - reserve - misc : 0;
    }
    reset_dev_cache();
    double best_mass = -1.0, all = 0;
    for (const Hot& h : prof) all += h.c;
    int64_t best = half, best_cap[2] = {0, 0}, half_cap[2] = {0, 0};
    double half_mass = 0;
    for (int64_t K = hp.n_dense + 1; K < NL; ++K) {
        int64_t cap[2];
        for (int i = 0; i < 2; ++i) {
            uint64_t need = i == 1 ? head : 0;
            for (int64_t il = i == 0 ? 0 : K; il < (i == 0 ? K : NL); ++il) need += dense[(size_t) il] + state[(size_t) il];
            cap[i] = free_b[i] > need ? (int64_t) ((free_b[i] - need) / slot) : 0;
        }
        double held = 0;
        int64_t used[2] = {0, 0};
        for (const Hot& h : prof) {
            if (blob[(size_t) h.il] == 0 || blob[(size_t) h.il] > slot) continue;
            const int st = h.il < K ? 0 : 1;
            if (used[st] >= cap[st]) continue;
            ++used[st];
            held += h.c;
        }
        if (K == half) {
            half_mass = held;
            half_cap[0] = cap[0];
            half_cap[1] = cap[1];
        }
        if (held > best_mass + 1e-9 * all) {
            best_mass = held;
            best = K;
            best_cap[0] = cap[0];
            best_cap[1] = cap[1];
        }
    }
    if (opt.verbose)
        fprintf(stderr, "strata-x: layer split auto: layers 0-%lld on GPU %d hold %.1f%% of the profile's routed mass (predicted slots "
                        "%lld + %lld); the halves: %.1f%% (%lld + %lld)\n",
                (long long) best - 1, devs[0], 100.0 * best_mass / std::max(1e-9, all), (long long) best_cap[0], (long long) best_cap[1],
                100.0 * half_mass / std::max(1e-9, all), (long long) half_cap[0], (long long) half_cap[1]);
    return best;
}

bool Engine::load(const std::string& gguf, const EngineOptions& o, std::string& err) {
    Impl& m = *p_;
    m.opt = o;
    m.snap_T = o.mtp_window > 0 ? std::min(o.mtp_window, kTmax) : 0;
    m.mtp_on = m.snap_T > 1;
    m.prompt_short = o.short_read;
    // 1: sync and check every step; 2: and scan the step's output for NaN / Inf (every scan needs the syncs)
    m.sync_check = std::getenv("STRATA_X_SYNC_CHECK") && (std::getenv("STRATA_X_SYNC_CHECK")[0] == '1' ||
                                                           std::getenv("STRATA_X_SYNC_CHECK")[0] == '2');
    m.graphs_on = !(std::getenv("STRATA_X_NO_GRAPHS") && std::getenv("STRATA_X_NO_GRAPHS")[0] == '1') && o.graphs;
    try {
        m.gm = std::make_unique<strata::GgufModel>(strata::GgufModel::open(gguf));
    } catch (const std::exception& e) {
        return m.fail(err, e.what());
    }
    if (!m.load_hparams(err)) return false;
    // moex (the GGUF set): DeepSeek-V4.1 from a set that holds its routed experts (gguf_convert.py's, unsloth's): they come from
    // it as virtual records (load_experts_gguf) unless --experts-gguf names another set; a pack's records keep the layers it lacks
    // moex (HY4_PORT.md): Hy4-flash's MTP block does not draft yet
    if (m.hp.arch == Arch::hy4 && m.mtp_on && o.dfly.empty())
        return m.fail(err, "hy_v4_flash: its MTP block does not draft yet - its DFly2 drafter does (--draft-dfly FILE with --spec N)");
    if (m.hp.v41 && m.opt.experts_gguf.empty() && m.gm->find("blk." + std::to_string(m.hp.n_dense) + ".ffn_gate_exps.weight"))
        m.opt.experts_gguf = gguf;
    if (m.mtp_on && m.hp.n_layer_all == m.hp.n_layer && o.dspark.empty() && o.draft_mtp.empty() && o.dflash.empty() && o.dfly.empty())
        return m.fail(err, "MTP asked for, but the GGUF has no NextN block (DeepSeek-V4: --draft-mtp FILE or --draft-dspark FILE)");
    if (!o.dflash.empty() && (!o.dspark.empty() || !o.draft_mtp.empty())) return m.fail(err, "one drafter: --draft-dflash, --draft-mtp or --draft-dspark");
    if (!o.dflash.empty() && !m.mtp_on) return m.fail(err, "--draft-dflash needs --spec N (N >= 2)");
    if (!o.dfly.empty() && (!o.dflash.empty() || !o.dspark.empty() || !o.draft_mtp.empty()))
        return m.fail(err, "one drafter: --draft-dfly, --draft-dflash, --draft-mtp or --draft-dspark");
    if (!o.dfly.empty() && !m.mtp_on) return m.fail(err, "--draft-dfly needs --spec N (N >= 2)");
    if (!m.open_dflash(err)) return false;
    if (!m.open_dfly(err)) return false;
    if (!o.dspark.empty() && m.hp.arch != Arch::dsv4) return m.fail(err, "--draft-dspark drafts for DeepSeek-V4 only");
    if (!o.draft_mtp.empty() && m.hp.arch != Arch::dsv4) return m.fail(err, "--draft-mtp drafts for DeepSeek-V4 only");
    if (m.hp.v41 && !o.draft_mtp.empty()) return m.fail(err, "DeepSeek-V4.1 drafts with its DSpark blocks (--draft-dspark builtin or FILE)");
    if (!o.draft_mtp.empty() && !o.dspark.empty()) return m.fail(err, "one drafter: --draft-mtp or --draft-dspark");
    if (!o.draft_mtp.empty() && !m.mtp_on) return m.fail(err, "--draft-mtp needs --mtp or --spec N (N >= 2)");
    // MiMo's three MTP blocks are chained heads (mainline's draft-mtp): at most one draft per block
    if (m.mtp_on && m.hp.arch == Arch::mimo2 && !m.dfl) m.snap_T = (int) std::min<int64_t>(m.snap_T, 1 + m.hp.n_layer_all - m.hp.n_layer);
    if (m.dfl) m.snap_T = std::min(m.snap_T, m.dfl->block);   // DFlash: the anchor and block - 1 drafts at most
    if (m.dfy) m.snap_T = std::min(m.snap_T, m.dfy->block);   // DFly2: the bonus row and block - 1 drafts at most
    // prompt lookup: Strata's window (the MTP's + 2, or 6 alone); GLM-5.3 within its KDA snapshot rows; not beside DSpark
    if (o.suffix_draft > 0 && o.dspark.empty()) {
        int S = o.lookup_window > 0 ? o.lookup_window : (m.mtp_on ? m.snap_T + 2 : 6);
        if (m.hp.arch == Arch::glm5) S = std::min(S, m.snap_T);
        m.lookup_S = std::min(S, kTmax) >= 2 ? std::min(S, kTmax) : 0;
        if (m.lookup_S > 0) {
            m.sfx = std::make_unique<strata::spec::SuffixDrafter>(std::max(1, o.suffix_draft), 64, (size_t) o.n_ctx + 4096);
            m.policy = std::make_unique<strata::spec::DraftPolicy>(m.lookup_S);
        }
    }
    m.n_load = m.hp.n_layer + (m.mtp_on && !m.dfl && !m.dfy ? (m.hp.arch == Arch::glm5 ? 1 : m.hp.n_layer_all - m.hp.n_layer) : 0);
    m.n_xl = m.hp.arch == Arch::glm5 ? m.n_load : m.hp.n_layer;
    {
        std::vector<int> devs = o.devices;
        if (devs.empty()) {
            int nd = 0;
            cudaGetDeviceCount(&nd);
            if (o.split == 0 || nd < 1) devs.push_back(o.device);
            else
                for (int d = 0; d < nd && d < 2; ++d) devs.push_back(d);
        }
        if (devs.size() > 2) return m.fail(err, "the layer split takes one or two GPUs");
        for (size_t i = 0; i < devs.size(); ++i) m.stages.push_back(std::make_unique<Stage>(m, (int) i, devs[i]));
        int64_t first =
            devs.size() == 1 ? m.hp.n_layer : (o.split > 0 ? std::min<int64_t>(o.split, m.hp.n_layer) : m.choose_split(devs));
        if (m.hp.v41 && devs.size() == 2 && !m.dsv41_split_ok(first)) {   // the nearest split sharing nothing (ties: lower)
            int64_t best = -1;
            for (int64_t K = 1; K < m.hp.n_layer; ++K)
                if (m.dsv41_split_ok(K) && (best < 0 || std::llabs(K - first) < std::llabs(best - first))) best = K;
            if (best < 0) return m.fail(err, "DeepSeek-V4.1: no layer split keeps the shared compressed rows on one GPU");
            if (o.verbose)
                fprintf(stderr, "strata-x: layer split %lld moved to %lld (V4.1's shared compressed rows stay on one GPU)\n",
                        (long long) first, (long long) best);
            first = best;
        }
        if (m.dfy && devs.size() == 2) {   // DFly2: its target layers' KV caches and cells on the last GPU, beside the drafter
            int64_t lo = m.hp.n_layer;
            for (int64_t il : m.dfy->tkv) lo = std::min(lo, il);
            for (int64_t il : m.dfy->tidx) lo = std::min(lo, il);
            if (first > lo) {
                if (o.verbose)
                    fprintf(stderr, "strata-x: layer split %lld moved to %lld (DFly2's target layers stay on the last GPU)\n",
                            (long long) first, (long long) lo);
                first = lo;
            }
        }
        m.ldev.assign((size_t) m.n_load, (int) devs.size() - 1);
        for (int64_t il = 0; il < first; ++il) m.ldev[(size_t) il] = 0;
        m.stages[0]->lb = 0;
        m.stages[0]->le = first;
        m.stages.back()->last = true;
        if (devs.size() == 2) {
            m.stages[1]->lb = first;
            m.stages[1]->le = m.hp.n_layer;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, devs[0], devs[1]);
            if (can) {
                cudaSetDevice(devs[0]);
                cudaDeviceEnablePeerAccess(devs[1], 0);
                cudaSetDevice(devs[1]);
                cudaDeviceEnablePeerAccess(devs[0], 0);
                cudaGetLastError();
            }
            if (o.verbose)
                fprintf(stderr, "strata-x: layer split: layers 0-%lld on GPU %d, %lld-%lld%s and the head on GPU %d (peer %s)\n",
                        (long long) first - 1, devs[0], (long long) first, (long long) m.hp.n_layer - 1,
                        m.dfl ? " + DFlash" : m.dfy ? " + DFly2" : m.mtp_on ? " + NextN" : "", devs[1], can ? "on" : "off: staged copies");
        } else {
            m.stages[0]->le = m.hp.n_layer;
        }
    }
    if (m.hp.arch == Arch::dsv4)   // the hash layers read the tokens, which only the first stage has
        for (int64_t il = 0; il < m.hp.n_hash; ++il)
            if (m.ldev[(size_t) il] != 0) return m.fail(err, "DeepSeek-V4's hash layers must be on the first GPU (--split > n_hash)");
    if (o.verbose)
        fprintf(stderr, "strata-x: %s, %lld layers (+%lld NextN), n_embd %lld, %lld experts top-%lld, ctx %lld\n",
                m.hp.arch == Arch::mimo2 ? "mimo2" : m.hp.v41 ? "deepseek41" : m.hp.arch == Arch::dsv4 ? "deepseek4"
                : m.hp.arch == Arch::hy4 ? "hy_v4_flash" : m.hp.arch == Arch::ernie ? "ernie55" : "glm5next",
                (long long) m.hp.n_layer,
                (long long) (m.hp.n_layer_all - m.hp.n_layer),
                (long long) m.hp.n_embd,
                (long long) m.hp.n_expert, (long long) m.hp.n_used, (long long) o.n_ctx);
    // the DSpark drafter before the trunk's experts: its pinned RAM comes out of what the resident RAM mode then sees
    if (!(m.load_dense(err) && m.bind_layers(err) && m.load_engram(err) && m.load_engram_hy4(err) && m.load_dspark(err) && m.load_draft_mtp(err) && m.load_dflash(err) &&
          m.load_experts(err) &&
          m.alloc_state(err) && m.load_dfly(err) && m.init_cache(err) && m.route_bias_init(err)))
        return false;
    m.replay_init();   // DeepSeek-V4.1's bounded replay, once the split is known
    m.push_static_res();
    m.warm_start();   // before the pin below: the warm thread keeps the load's CPUs, off the host's core
    // PIN THE HOST ONCE (Strata, session.cpp SessionLoopScratch): the thread that runs the windows spins on the doorbell and
    // drains with the pool; the pool's workers keep off the first physical core for it, but nothing pins it there - left
    // free, the spin lands on a worker's core or its SMT sibling and Strata measured the CPU path at 26.9 GB/s instead of
    // 36.32.  The thread that loads is the one that decodes (the CLI's and --serve's main thread).
    if (m.opt.pin) {
        const int host_core = kc::planned_host_core();
        if (host_core >= 0) {
            const kc::ThreadAffinity a = kc::pin_current_thread(host_core);
            if (m.opt.verbose) fprintf(stderr, "strata-x: the host thread %s core %d\n", a.valid ? "pinned to" : "could not be pinned to", host_core);
        }
    }
    if (const char* e = std::getenv("STRATA_X_MTP_CHAIN_H")) m.mtp_chain_res = std::strcmp(e, "norm") != 0;
    else m.mtp_chain_res = true;
    if (o.warmup) m.warm_decode();
    m.eg_preload_start();   // moex (DSV41_PLAN §18): the engram's scale tables into the page cache, behind the load (opt-in)
    return true;
}

void Engine::reset() {
    Impl& m = *p_;
    for (auto& gp : m.stages) {
        cudaSetDevice(gp->dev);
        cudaStreamSynchronize(gp->s);
    }
    const int64_t din = m.hp.n_head * m.hp.hd_kda;
    for (int64_t il = 0; il < m.n_load; ++il) {
        Layer& l = m.L[(size_t) il];
        cudaSetDevice(m.stage_of(il).dev);
        if (l.conv) cudaMemset(l.conv, 0, (size_t) 3 * din * (m.hp.d_conv - 1) * sizeof(float));
        if (l.S) cudaMemset(l.S, 0, (size_t) m.hp.hd_kda * din * sizeof(float));
        // Hy4-flash: its rows are 576 wide and a sliding-window layer's a ring of kv_rows (n_ctx x kv_lora ran past the ring's
        // end: cudaMemset refused it, unchecked, and the prompt path's next check failed on it - the KLD tool's chunks)
        // (moex, STRATA_HY4_KV_GROW: a growable layer's mapped rows only)
        const bool row576 = m.hp.arch == Arch::hy4 || m.hp.arch == Arch::ernie;   // (ERNIE-5.5's DSA rows are Hy4's 576)
        const bool grown = m.kv_grow && row576 && il < (int64_t) m.kv_vmm.size() && m.kv_vmm[(size_t) il].va;
        if (l.kv_cache)
            cudaMemset(l.kv_cache, 0, grown ? m.kv_vmm[(size_t) il].mapped
                                            : row576 ? (size_t) l.kv_rows * 576 * 2 : (size_t) m.opt.n_ctx * m.hp.kv_lora * 2);
        if (l.pooled) cudaMemset(l.pooled, 0, grown ? m.pool_vmm[(size_t) il].mapped : (size_t) m.max_pool * m.hp.idx_dim * sizeof(float));
        if (l.ik_ring) cudaMemset(l.ik_ring, 0, (size_t) m.ring_rows * m.hp.idx_dim * sizeof(float));
        if (l.ig_ring) cudaMemset(l.ig_ring, 0, (size_t) m.ring_rows * m.hp.idx_dim * sizeof(float));
    }
    for (auto& gp : m.stages) {
        cudaSetDevice(gp->dev);
        cudaDeviceSynchronize();
    }
    m.reset_dev_cache();
}

void Engine::set_short_read(int64_t n) { p_->prompt_short = n; }

bool Engine::save_profile(const std::string& path, std::string& err) const { return p_->save_profile(path, err); }
bool Engine::save_cache_keys(const std::string& path, std::string& err) { return p_->save_cache_keys(path, err); }
bool Engine::save_conversations(std::string& err) { return p_->disk_save(err); }
int64_t Engine::refill_cache(const std::string& path) { return p_->refill_cache(path); }
bool Engine::encode_pictures(const std::vector<std::pair<std::string, std::string>>& files, std::vector<int64_t>& rows, double& ms,
                             std::string& how, std::string& err) {
    return p_->encode_pictures(files, rows, ms, how, err);
}

/// The imatrix collection's sums, per MoE layer on its GPU (prompt.cpp moe_prompt adds to them).
bool Engine::imatrix_begin(std::string& err) {
    Impl& m = *p_;
    if (m.imx) return true;
    if (!(m.opt.prefill_chunk != -1 && m.gub_max > 0 && m.mmq_all && mmq::built()))
        return m.fail(err, "the imatrix collection reads through the prompt path, which is off for this model or build");
    auto x = std::make_unique<Impl::Imx>();
    const int64_t E = m.hp.n_expert, n = m.hp.ex(), nf = m.hp.n_ff_exp;   // (moex: the experts' input width, the latent's)
    x->gu.assign((size_t) m.hp.n_layer, nullptr);
    x->dn.assign((size_t) m.hp.n_layer, nullptr);
    x->cnt.assign((size_t) (m.hp.n_layer * E), 0);
    for (int64_t il = m.hp.n_dense; il < m.hp.n_layer; ++il) {
        if (!m.hp.is_moe(il)) continue;
        Stage& g = m.stage_of(il);
        m.set_dev(g.dev);
        float* a = g.dalloc<float>((size_t) (E * n), err);
        float* b = g.dalloc<float>((size_t) (E * nf), err);
        if (!a || !b) return m.fail(err, "the imatrix sums: " + err);
        cudaMemset(a, 0, (size_t) (E * n) * sizeof(float));
        cudaMemset(b, 0, (size_t) (E * nf) * sizeof(float));
        x->gu[(size_t) il] = a;
        x->dn[(size_t) il] = b;
        if (!g.imx_order && !(g.imx_order = g.dalloc<int32_t>((size_t) E, err))) return m.fail(err, "the imatrix order: " + err);
    }
    m.reset_dev_cache();
    m.imx = std::move(x);
    return true;
}

/// DSV41IMX v1 (moex/tools/dsv41/requant_pack.cpp): u32 n_layers, n_expert, n_embd, n_ff, then per layer and expert the gate/up
/// sums [n_embd], the down sums [n_ff] (F32) and the tokens routed (u32); a dense layer's entries are zeros.  moex: n_embd is
/// the experts' input width (HParams::ex(): a latent MoE's latent).
bool Engine::imatrix_write(const std::string& path, std::string& err) const {
    Impl& m = *p_;
    if (!m.imx) return m.fail(err, "no imatrix collection running (imatrix_begin)");
    const int64_t E = m.hp.n_expert, n = m.hp.ex(), nf = m.hp.n_ff_exp, NL = m.hp.n_layer;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return m.fail(err, "cannot write " + path);
    const uint32_t hdr[5] = {1, (uint32_t) NL, (uint32_t) E, (uint32_t) n, (uint32_t) nf};
    bool ok = fwrite("DSV41IMX", 1, 8, f) == 8 && fwrite(hdr, 4, 5, f) == 5;
    std::vector<float> gu((size_t) (E * n)), dn((size_t) (E * nf));
    for (int64_t il = 0; ok && il < NL; ++il) {
        std::fill(gu.begin(), gu.end(), 0.0f);
        std::fill(dn.begin(), dn.end(), 0.0f);
        if (m.imx->gu[(size_t) il]) {
            m.set_dev(m.stage_of(il).dev);
            cudaDeviceSynchronize();
            cudaMemcpy(gu.data(), m.imx->gu[(size_t) il], gu.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(dn.data(), m.imx->dn[(size_t) il], dn.size() * sizeof(float), cudaMemcpyDeviceToHost);
        }
        for (int64_t e = 0; ok && e < E; ++e) {
            const uint32_t c = (uint32_t) std::min<uint64_t>(m.imx->cnt[(size_t) (il * E + e)], 0xFFFFFFFFu);
            ok = fwrite(gu.data() + e * n, 4, (size_t) n, f) == (size_t) n && fwrite(dn.data() + e * nf, 4, (size_t) nf, f) == (size_t) nf &&
                 fwrite(&c, 4, 1, f) == 1;
        }
    }
    m.reset_dev_cache();
    if (fclose(f) != 0 || !ok) return m.fail(err, "writing " + path);
    return true;
}

void Engine::set_spec(int window_cap, float min_p) {
    p_->spec_cap = window_cap;
    if (min_p >= 0.0f) p_->opt.spec_min_p = min_p;
}

void Engine::set_stand_in(int rows) { p_->stand_in = rows; }

void Engine::set_prompt_options(int prefill_chunk, int prefill_ring, int prefill_sub) {
    p_->opt.prefill_chunk = prefill_chunk;
    p_->opt.prefill_ring = prefill_ring;
    p_->opt.prefill_sub = prefill_sub;
}

bool Engine::step(int32_t token, int64_t pos, std::string& err) { return window(&token, 1, pos, false, err); }

bool Engine::Impl::run_window(const int32_t* tokens, int T, int64_t pos0, bool keep, std::string& err) {
    const auto t0 = clk::now();
    if (!forward(tokens, T, pos0, keep, err)) return false;
    for (auto& gp : stages) gp->st.tokens += T;
    const auto tu = clk::now();
    maintain_cache();
    stages[0]->st.ms_upkeep += ms_since(tu);
    if (sync_check)
        for (auto& gp : stages) {
            set_dev(gp->dev);
            cudaDeviceSynchronize();
            if (!gp->sync_check("the tier's upkeep (fills, swaps, checks)", (int) gp->lb, err)) return false;
        }
    stages[0]->st.ms_total += ms_since(t0);
    return true;
}

bool Engine::window(const int32_t* tokens, int T, int64_t pos0, bool keep, std::string& err) {
    return p_->run_window(tokens, T, pos0, keep, err);
}

/// The prompt: Strata's prompt path in chunks, or - a segment of at most short_read tokens, or without the MMQ tiles
/// it needs - decode windows of 8 (Strata's short reads).
bool Engine::prefill(const int32_t* tokens, int n, int64_t pos0, std::string& err) {
    Impl& m = *p_;
    const bool batched = n > m.prompt_short && m.opt.prefill_chunk != -1 && m.gub_max > 0 && m.mmq_all && mmq::built();
    if (batched) return m.prompt(tokens, n, pos0, n, err);
    for (int i = 0; i < n; i += kTmax)
        if (!window(tokens + i, std::min(kTmax, n - i), pos0 + i, false, err)) return false;
    return true;
}

void Engine::rollback(int n_keep) { p_->rollback_to(n_keep); }

void Engine::Impl::rollback_to(int n_keep) {
    Impl& m = *this;
    if (n_keep >= m.last_T || m.snap_T == 0) return;
    const int64_t H = m.hp.n_head, D = m.hp.hd_kda, din = H * D;
    for (int64_t il = 0; il < m.hp.n_layer; ++il) {
        // only KDA's recurrent state rolls back: the latent caches, the pools and MiMo's K/V rows are rewritten when those
        // positions return (MiMo's sliding-window rings hold the window plus a whole batch: the rows a rejected draft wrote
        // are older than every remaining query's window)
        if (m.hp.arch != Arch::glm5 || m.hp.is_dsa(il)) continue;
        Layer& l = m.L[(size_t) il];
        Stage& g = m.stage_of(il);
        m.set_dev(g.dev);
        // the recurrent state after token n_keep - 1 (n_keep >= 1: the window's first token is always accepted)
        cudaMemcpyAsync(l.S, l.snaps + (size_t) (n_keep - 1) * D * H * D, (size_t) D * H * D * sizeof(float),
                        cudaMemcpyDeviceToDevice, g.s);
        const size_t cs = (size_t) din * (m.hp.d_conv - 1);
        for (int c = 0; c < 3; ++c) {
            // the conv inputs of stream c are rows of 3 * din at offset c * din: gather them for the rebuild
            for (int t = 0; t < n_keep; ++t)
                cudaMemcpyAsync(g.win.y + (size_t) t * din, l.conv_in + (size_t) t * 3 * din + (size_t) c * din,
                                din * sizeof(float), cudaMemcpyDeviceToDevice, g.s);
            kda_conv_rollback(l.conv_prev + c * cs, g.win.y, l.conv + c * cs, (int) din, (int) m.hp.d_conv, n_keep, g.s);
        }
    }
    m.last_T = n_keep;
}

bool Engine::generate(const std::vector<int32_t>& prompt, int n_gen, bool use_mtp, GenResult& r, std::string& err) {
    Impl& m = *p_;
    r = GenResult{};
    const int N = (int) prompt.size();
    if (N == 0) return m.fail(err, "empty prompt");
    if (use_mtp && !m.mtp_on) return m.fail(err, "MTP was not enabled at load (EngineOptions::mtp_window >= 2)");
    auto t0 = clk::now();
    // positions [0, N - 1): the prompt path (or windows); the last prompt token goes through the first decode window
    // (Strata: "the path all three model files share").  With MTP the NextN block's caches follow the trunk's.
    const int head_n = N - 1;
    const bool batched = head_n > m.prompt_short && m.opt.prefill_chunk != -1 && m.gub_max > 0 && m.mmq_all && mmq::built();
    if (head_n > 0) {
        if (batched) {
            if (!m.prompt(prompt.data(), head_n, 0, use_mtp ? N : 0, err)) return false;
        } else {
            for (int i = 0; i < head_n; i += kTmax) {
                const int T = std::min(kTmax, head_n - i);
                if (!window(prompt.data() + i, T, i, false, err)) return false;
                if (use_mtp && !m.drafter_read(T, prompt.data() + i + 1, i, err)) return false;
            }
        }
    }
    r.ms_prefill = ms_since(t0);
    t0 = clk::now();
    // the decode: Strata's window loop (spec.cpp), greedy
    const SamplingParams sp;
    const std::vector<int32_t> no_eos;
    std::vector<int32_t> hist{prompt.back()};
    Impl::SpecRun run;
    run.sp = &sp;
    run.eos = &no_eos;
    run.max_new = n_gen;
    run.pos = N - 1;
    run.x = prompt.back();
    run.hist = &hist;
    run.plain = !use_mtp;
    run.token = [&](int32_t t) {
        r.tokens.push_back(t);
        return true;
    };
    m.live.assign(prompt.begin(), prompt.begin() + head_n);
    if (!m.spec_decode(run, err)) return false;
    r.ms_gen = ms_since(t0);
    r.ms_draft = run.draft_ms;
    r.drafted = run.offered;
    r.accepted = run.accepted;
    r.windows = run.windows;
    for (int j = 0; j < 8; ++j) {
        r.offered_at[j] = run.offered_at[j];
        r.accepted_at[j] = run.accepted_at[j];
    }
    return true;
}

int32_t Engine::argmax(std::string& err) {
    std::vector<int32_t> r;
    if (!argmax_rows(r, err) || r.empty()) return -1;
    return r.back();
}

bool Engine::argmax_rows(std::vector<int32_t>& out, std::string& err) {
    Impl& m = *p_;
    Stage& g = m.last_stage();
    m.set_dev(g.dev);
    const int T = std::max(1, m.logit_rows);
    argmax_w(g.logits, (int) m.hp.n_vocab, g.amax, T, g.s);
    cudaMemcpyAsync(m.h_amax, g.amax, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost, g.s);
    if (cudaStreamSynchronize(g.s) != cudaSuccess) return m.fail(err, "argmax failed");
    out.assign(m.h_amax, m.h_amax + T);
    return true;
}

bool Engine::logits(std::vector<float>& out, std::string& err, int row) {
    Impl& m = *p_;
    Stage& g = m.last_stage();
    m.set_dev(g.dev);
    const int T = std::max(1, m.logit_rows);
    if (row < 0) row = T - 1;
    out.resize((size_t) m.hp.n_vocab);
    if (cudaMemcpyAsync(out.data(), g.logits + (size_t) row * m.hp.n_vocab, out.size() * sizeof(float),
                        cudaMemcpyDeviceToHost, g.s) != cudaSuccess ||
        cudaStreamSynchronize(g.s) != cudaSuccess)
        return m.fail(err, "logits readback failed");
    return true;
}

}  // namespace strata::x
