// src/x/stage.cpp - a GPU's share of GLM-5.3-Flash (see engine_impl.hpp): its buffers, and the compute of
// its layers - mHC, KDA, DSA, the dense FFN, a decode window's MoE (VRAM hits + the CPU pool), the head, the NextN
// block.  The op order is moex/ref/glm_ref.cpp's GlmRef::step (llama.cpp's glm5-next graph) and the probe names are
// the same, so `strata-x --oracle <llama_dump dir>` checks every op on llama.cpp's own inputs.
#include "engine_impl.hpp"

#include "strata/x/dsv41_kernels.hpp"

#include "ggml.h"

#include <cublas_v2.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>

namespace strata::x {

namespace {
std::mutex g_pageable_mu;
std::unordered_set<void*> g_pageable;   // host_buf's heap fallbacks (anything else host_free sees is pinned)
}  // namespace

void* host_buf(size_t bytes) {
    void* p = nullptr;
    // STRATA_X_TEST_PAGEABLE=1 (a test hook): every buffer takes the fallback, as if the driver refused each pin
    static const bool no_pin = [] { const char* e = std::getenv("STRATA_X_TEST_PAGEABLE"); return e && e[0] == '1'; }();
    if (!no_pin && cudaMallocHost(&p, std::max<size_t>(bytes, 1)) == cudaSuccess) return p;
    cudaGetLastError();   // the refused pin is not the next launch's error (forward_graph checks cudaGetLastError)
    p = std::malloc(std::max<size_t>(bytes, 1));
    if (!p) return nullptr;
    static std::atomic<int> said{0};
    if (said.fetch_add(1) < 3)
        fprintf(stderr, "strata-x: pinned host memory refused for %.1f MiB: that buffer is pageable (slower copies, the same "
                        "results)\n", bytes / 1048576.0);
    std::lock_guard<std::mutex> lk(g_pageable_mu);
    g_pageable.insert(p);
    return p;
}

void host_free(void* p) {
    if (!p) return;
    {
        std::lock_guard<std::mutex> lk(g_pageable_mu);
        if (g_pageable.erase(p)) {
            std::free(p);
            return;
        }
    }
    cudaFreeHost(p);
}

Stage::Stage(Engine::Impl& m_, int di_, int dev_) : m(m_), hp(m_.hp), opt(m_.opt), L(m_.L), di(di_), dev(dev_) {}

Stage::~Stage() {
    cudaSetDevice(dev);
    for (void* p : allocs) cudaFree(p);
    if (own_prompt) cudaFree(own_prompt);
    if (slots) cudaFree(slots);
    for (cudaEvent_t e : copied) cudaEventDestroy(e);
    for (cudaEvent_t e : used) cudaEventDestroy(e);
    for (cudaEvent_t e : sh_ev)
        if (e) cudaEventDestroy(e);
    for (void* p : {(void*) h_ids, (void*) h_grp, (void*) h_hand[0], (void*) h_hand[1], (void*) h_embd}) host_free(p);
    if (tok_buf) cudaFree(tok_buf);
    if (in_tok_h) cudaFreeHost(in_tok_h);
    if (!lmh_lent)   // (moex, §63: a host window's rows - not when the SSD cache lent them -, its copy streams and events)
        for (uint8_t* p : lmh) host_free(p);
    for (cudaStream_t cst : lm_cs)
        if (cst) cudaStreamDestroy(cst);
    for (auto* evs : {&lm_up_ev, &lm_done_ev, &lm_dn_ev})
        for (cudaEvent_t e : *evs) cudaEventDestroy(e);
    for (int i = 0; i < 2; ++i) {   // (moex, §62: the prompt guess's)
        if (gs_hb[i]) cudaFreeHost(gs_hb[i]);
        if (gs_ev[i]) cudaEventDestroy(gs_ev[i]);
    }
    for (void* p : {(void*) eg_win_h, (void*) eg_pr_h})   // DeepSeek-V4.1's engram rows (mapped)
        if (p) cudaFreeHost(p);
    for (size_t i = 0; i < rbounce.size(); ++i) {   // DeepSeek-V4.1's SSD tier: the stream's bounce slots
        if (rbounce[i]) cudaFreeHost(rbounce[i]);
        if (rbounce_ev[i]) cudaEventDestroy(rbounce_ev[i]);
    }
    for (void* p : {(void*) h_cx, (void*) h_cdm}) host_free(p);
    mq.reset();
    for (auto& kv : gx) cudaGraphExecDestroy(kv.second);
    for (auto& kv : df_gx) cudaGraphExecDestroy(kv.second);
    if (db_block) cudaFreeHost(db_block);
    if (h_pos) cudaFreeHost(h_pos);
    if (blas) cublasDestroy((cublasHandle_t) blas);
    if (ev) cudaEventDestroy(ev);
    if (hand) cudaEventDestroy(hand);
    if (swap_ev) cudaEventDestroy(swap_ev);
    if (cs) cudaStreamDestroy(cs);
    if (s) cudaStreamDestroy(s);
}

template <typename T>
T* Stage::dalloc(size_t n, std::string& err) {
    void* p = nullptr;
    if (cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)) != cudaSuccess) {
        err = "cudaMalloc failed on GPU " + std::to_string(dev);
        return nullptr;
    }
    cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T));
    allocs.push_back(p);
    return static_cast<T*>(p);
}

void Stage::dfree(void* p) {
    if (!p) return;
    auto it = std::find(allocs.begin(), allocs.end(), p);
    if (it == allocs.end()) return;
    allocs.erase(it);
    cudaFree(p);
}

// ------------------------------------------------------------------ buffers

namespace {
struct Lay {   // carves 256-byte aligned buffers out of a region (base null: only counts)
    uint8_t* base;
    size_t off = 0;
    template <typename T>
    T* take(size_t n) {
        off = (off + 255) & ~(size_t) 255;
        T* p = base ? reinterpret_cast<T*>(base + off) : nullptr;
        off += std::max<size_t>(n, 1) * sizeof(T);
        return p;
    }
};

void carve_into(const Hparams& hp, int64_t n_ctx, int gqa_chunks, Scratch& x, uint8_t* base, int rows, int sub, size_t* bytes,
                bool cand_rows = true) {
    const int64_t n = hp.n_embd, H = hp.n_head, D = hp.hd_kda, din = H * D;
    const int64_t Lq = hp.q_lora, Lkv = hp.kv_lora, Dk = hp.k_mla, Dv = hp.v_mla, IH = hp.idx_heads, ID = hp.idx_dim;
    const int64_t max_in = max_proj_in(hp);
    const bool glm = hp.arch == Arch::glm5;
    const bool kpool_arch = glm || hp.arch == Arch::hy4;   // the k-pool indexer's scores and cells (Hy4-flash's too)
    const bool ernie = hp.arch == Arch::ernie;             // ERNIE-5.5's DSA cells: the indexer's top-k, then the window
    const int64_t max_pool = kpool_arch ? n_ctx / hp.kpool + 1 : 0,
                  max_cells = kpool_arch ? hp.idx_topk + hp.kpool : ernie ? hp.idx_topk + hp.n_swa : 0;
    const int64_t ff = std::max(hp.n_ff, hp.n_ff_exp);
    Lay c{base};
    x.rows = rows;
    x.sub = sub;
    x.R = c.take<float>((size_t) rows * hp.hc * n);
    x.cur = c.take<float>((size_t) rows * n);
    x.out = c.take<float>((size_t) rows * n);
    x.coef = c.take<float>((size_t) rows * 24);
    x.router = c.take<float>((size_t) rows * hp.n_expert * (ernie ? 2 : 1));   // (ERNIE-5.5: its two routers' logits)
    x.wts = c.take<float>((size_t) rows * hp.n_used);
    x.ids = c.take<int32_t>((size_t) rows * hp.n_used);
    x.shared = c.take<float>((size_t) rows * n);
    x.h = c.take<float>((size_t) rows * n);
    x.flat = c.take<float>((size_t) sub * std::max<int64_t>(hp.hc, 2) * n);   // mHC's flat read, MTP's [e | h]
    x.mixes = c.take<float>((size_t) sub * 24);
    // (moex, the ERNIE-5.5 hill-climb: ERNIE-5.5's one-token mixer rows as their 8 warps' totals, gemv_f32_w1_warps)
    // (ERNIE-5.5's sized as V4.1's: STRATA_DSV41_HC_SPLIT's split partials fit as well)
    x.hc_part = hp.v41 || (hp.arch == Arch::ernie && hp.hc > 1)
                    ? c.take<float>((std::max)((size_t) 24 * 8, (size_t) kTmax * 24 * gemv_f32_split_parts((int) (hp.hc * n))))
                    : nullptr;
    x.xq = c.take<uint8_t>(kn::native_q8_1_bytes((int) (max_in * sub)));
    x.xq2 = c.take<uint8_t>(kn::native_q8_1_bytes((int) (max_in * sub)));
    for (float** p : {&x.pq, &x.pk, &x.pv, &x.cq, &x.ck, &x.cv, &x.g, &x.o, &x.g2, &x.y}) *p = c.take<float>((size_t) sub * din);
    x.fa = c.take<float>((size_t) sub * D);
    x.ga = c.take<float>((size_t) sub * D);
    x.beta = c.take<float>((size_t) sub * H);
    x.qr = c.take<float>((size_t) sub * Lq);
    x.q = c.take<float>((size_t) sub * H * Dk);
    x.q_hm = c.take<float>((size_t) sub * H * Dk);
    x.kv = c.take<float>((size_t) sub * Lkv);
    for (float** p : {&x.qabs, &x.qabs_hm, &x.ctx, &x.ctx_hm}) *p = c.take<float>((size_t) sub * H * Lkv);
    x.iq = c.take<float>((size_t) sub * IH * ID);
    x.ikw = c.take<float>((size_t) sub * ID);
    x.igw = c.take<float>((size_t) sub * ID);
    x.w32 = c.take<float>((size_t) sub * IH);
    x.score = c.take<float>((size_t) sub * max_pool);
    x.o2 = c.take<float>((size_t) sub * H * Dv);
    x.o2_hm = c.take<float>((size_t) sub * H * Dv);
    x.partial = glm ? c.take<float>(dsa_attend_w_scratch_bytes((int) H, (int) Lkv, (int) max_cells, kAttnQ) / sizeof(float))
                    : nullptr;
    x.cells = c.take<int32_t>((size_t) sub * max_cells);
    x.fg = c.take<float>((size_t) sub * ff);
    x.fu = c.take<float>((size_t) sub * ff);
    x.sg = x.fg;
    x.su = x.fu;
    if (hp.arch == Arch::mimo2) {   // the fused QKV rows, the attention's output, its split-K partials (decode windows only)
        int64_t qkv = 0;
        for (int64_t il = 0; il < hp.n_layer_all; ++il) qkv = std::max(qkv, hp.qkv_width(il));
        x.qkv = c.take<float>((size_t) sub * qkv);
        x.att = c.take<float>((size_t) sub * H * hp.hd_v);
        x.gpart = c.take<float>(gqa_attend_scratch_bytes((int) H, (int) hp.hd_v, gqa_chunks, std::min(sub, kTmax)) / sizeof(float));
    }
    if (hp.arch == Arch::dsv4 || ernie) {   // dsv4.cpp's buffers: per sub-batch token, but the hash ids per row of the window / chunk
        // DeepSeek-V4.1: the indexer's 32 heads run through V4's 64-head kernels padded with zero weights, and ratio 1 scores
        // every position (dsv41.cpp)
        // ERNIE-5.5: its HCA layers take V4's ratio-128 buffers, its DSA layers the indexer's (every position, in V4.1's row blocks)
        const int64_t Dh = hp.hd_k, IDd = hp.idx_dim, IHd = hp.v41 ? std::max<int64_t>(hp.idx_heads, 64) : hp.idx_heads,
                      lid_rows = hp.v41 || ernie ? n_ctx + 1 : n_ctx / 4 + 1;
        x.dq = c.take<float>((size_t) sub * H * Dh);
        x.datt = c.take<float>((size_t) sub * H * Dh);
        x.dkv = c.take<float>((size_t) sub * Dh);
        x.dakv = c.take<float>((size_t) sub * 2 * Dh);
        x.dgs = c.take<float>((size_t) sub * 2 * Dh);
        x.dlakv = c.take<float>((size_t) sub * 2 * IDd);
        x.dlgs = c.take<float>((size_t) sub * 2 * IDd);
        x.diq = c.take<float>((size_t) sub * IHd * IDd);
        x.dw = c.take<float>((size_t) sub * IHd);
        x.dscore = c.take<float>((size_t) (hp.v41 || ernie ? std::min(sub, kDsv41ScoreRows) : sub) * lid_rows);   // V4.1: in row blocks
        // the top-k list: per sub-batch row (V4: every compressed layer selects its own), per window / chunk row (V4.1: the
        // layers after an index source reuse its rows' lists, sub-batch by sub-batch)
        x.dsel = c.take<int32_t>((size_t) (hp.v41 ? rows : sub) * hp.idx_topk);
        x.doa = c.take<float>((size_t) sub * hp.o_groups * hp.o_lora);
        x.doa_hm = c.take<float>((size_t) sub * hp.o_groups * hp.o_lora);
        size_t ap = 0;   // the split-K partials: any batch of 1 .. 31 rows (a decode window, a chunk's last sub-batch)
        for (int tt = 1; tt <= std::min(sub, 31); ++tt) ap = std::max(ap, dsv4_attn_scratch_bytes(tt));
        x.dapart = c.take<float>(ap / sizeof(float) + 1);
        x.dhash = c.take<int32_t>((size_t) rows * hp.n_used);
        if (hp.v41) {   // taken after V4's buffers: a deepseek4 GGUF's layout is unchanged
            x.cpre = c.take<float>((size_t) rows * 24);
            x.dlat = c.take<float>((size_t) sub * Dh);
            x.dik = c.take<float>((size_t) sub * IDd);
            // per row; moex (DSV41_PLAN §59 P3): none on a stage whose layers all come before the candidates' source
            x.dcand = cand_rows ? c.take<uint8_t>((size_t) rows * dsv41_cand_bytes((int) lid_rows, std::max(1, hp.cand_bsize)))
                                : nullptr;
            x.cpre_b = c.take<float>((size_t) rows * 24);   // moex (#105): taken last, the other buffers' places unchanged
            if (dsv41_cand_only() && hp.cand_src >= 0 && hp.cand_blocks > 0) {   // moex (§56 C): after it, and only when on
                x.dclist = c.take<int32_t>((size_t) kDsv41ScoreRows * hp.cand_blocks);
                x.dccount = c.take<int32_t>((size_t) kDsv41ScoreRows);
            }
        }
    }
    if (hp.arch == Arch::hy4 || ernie) {   // hy4.cpp's buffers (H = n_head, the most heads a layer has)
        constexpr int64_t kRow = 576;   // [latent 512 | rope 64]
        x.hqf = c.take<float>((size_t) sub * H * kRow);
        x.hgate = c.take<float>((size_t) sub * H * Dv);
        x.hkv = c.take<float>((size_t) sub * kRow);
        x.hiv = c.take<float>((size_t) sub * ID);
        x.hxl = c.take<float>((size_t) rows * hp.ex());
        x.hml = c.take<float>((size_t) rows * hp.ex());
        const int64_t egw = hp.engram ? (hp.eg_ngram - 1) * hp.eg_heads * hp.eg_dim : 0;
        x.hegr = c.take<float>((size_t) sub * egw);
        x.hegkv = c.take<float>((size_t) sub * (egw ? 5 * n : 0));
        // the attention's split-K partials, for a decode window (a prompt sub-batch takes one block a token and none): the most
        // (heads x chunks) any layer has - a sparse layer's cells (the best groups and the token's own) or the window's
        if (sub <= kTmax && ernie) {   // ERNIE-5.5's DSA layers: the most cells a token has
            x.hpart = c.take<float>(hy4_attend_scratch_bytes(1, (int) (H * hy4_attend_chunks((int) max_cells)), sub) / sizeof(float));
        } else if (sub <= kTmax) {
            const int64_t n_top = hp.idx_topk / hp.kpool - (hp.select_tail ? 1 : 0);
            const int ch_sparse = hy4_attend_chunks((int) (n_top * hp.kpool + hp.kpool));
            const int ch_swa = hy4_attend_chunks((int) std::min<int64_t>(hp.n_swa, n_ctx));
            int64_t most = 0;
            for (int64_t il = 0; il < hp.n_layer_all; ++il)
                most = std::max<int64_t>(most, hp.H_of(il) * (hp.is_swa(il) ? ch_swa : ch_sparse));
            x.hpart = c.take<float>(hy4_attend_scratch_bytes(1, (int) most, sub) / sizeof(float));
        } else {
            // moex (2026-11-26): a prompt sub-batch takes none. The prompt path's Scratch lives on, so without this a read of
            // more than kTmax rows kept the last short read's pointer, into that read's loan, and its sub-batches of kTmax
            // rows or fewer wrote and read their split-K partials there (ernie_dsa / hy4.cpp: `T <= kTmax && hpart`)
            x.hpart = nullptr;
        }
    }
    if (ernie) x.ecnt = c.take<int32_t>((size_t) sub);
    if (bytes) *bytes = c.off + 256;
}
}  // namespace

size_t Stage::scratch_bytes(const Engine::Impl& m, int rows, int sub, bool cand_rows) {
    Scratch x;
    size_t b = 0;
    carve_into(m.hp, m.opt.n_ctx, m.gqa_chunks, x, nullptr, rows, sub, &b, cand_rows);
    return b;
}

int64_t Stage::lendable() const {
    return std::max<int64_t>(0, std::min<int64_t>(n_slots * 9 / 10, n_slots - 128));   // the drafter's experts are in their own region
}

int64_t Stage::lendable_lm() const {
    static const int64_t keep = std::getenv("STRATA_DSV41_LM_KEEP") ? std::max(0, std::atoi(std::getenv("STRATA_DSV41_LM_KEEP"))) : -1;
    return keep < 0 ? lendable() : std::max<int64_t>(0, n_slots - keep);
}

size_t Stage::lend_bytes(int64_t k) const {
    k = std::clamp<int64_t>(k, 0, n_slots);
    return slot_off.empty() ? (size_t) k * m.slot_bytes : slot_off[(size_t) n_slots] - slot_off[(size_t) (n_slots - k)];
}

void Stage::carve(Scratch& x, uint8_t* base, int rows, int sub) const {
    carve_into(hp, opt.n_ctx, m.gqa_chunks, x, base, rows, sub, nullptr, cand_rows());
}

void Stage::use(const Scratch& x) {
    R = x.R; cur = x.cur; out = x.out; coef = x.coef; router = x.router; wts = x.wts; ids = x.ids;
    shared = x.shared; h = x.h; flat = x.flat; mixes = x.mixes; hc_part = x.hc_part; xq = x.xq; xq2 = x.xq2;
    pq = x.pq; pk = x.pk; pv = x.pv; cq = x.cq; ck = x.ck; cv = x.cv; fa = x.fa; g = x.g; beta = x.beta;
    o = x.o; ga = x.ga; g2 = x.g2; y = x.y; qr = x.qr; q = x.q; q_hm = x.q_hm; kv = x.kv; qabs = x.qabs;
    qabs_hm = x.qabs_hm; iq = x.iq; ikw = x.ikw; igw = x.igw; w32 = x.w32; score = x.score; ctx = x.ctx;
    ctx_hm = x.ctx_hm; o2 = x.o2; o2_hm = x.o2_hm; partial = x.partial; cells = x.cells;
    fg = x.fg; fu = x.fu; sg = x.sg; su = x.su;
    qkv = x.qkv; att = x.att; gpart = x.gpart;
    dq = x.dq; dkv = x.dkv; dakv = x.dakv; dgs = x.dgs; dlakv = x.dlakv; dlgs = x.dlgs; diq = x.diq; dw = x.dw;
    dscore = x.dscore; datt = x.datt; doa = x.doa; doa_hm = x.doa_hm; dapart = x.dapart; dsel = x.dsel; dhash = x.dhash;
    cpre = x.cpre; dlat = x.dlat; dik = x.dik; dcand = x.dcand; cpre_b = x.cpre_b;
    dclist = x.dclist; dccount = x.dccount;
    hqf = x.hqf; hgate = x.hgate; hkv = x.hkv; hiv = x.hiv; hxl = x.hxl; hml = x.hml; hegr = x.hegr; hegkv = x.hegkv; hpart = x.hpart;
    ecnt = x.ecnt;
    SB = x.sub;
}

bool Stage::alloc(std::string& err) {
    cudaSetDevice(dev);
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&mix_fork, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&mix_join, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&fork_ev, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&join_ev, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&hand, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&swap_ev, cudaEventDisableTiming) != cudaSuccess)
        return m.fail(err, "streams/events on GPU " + std::to_string(dev));
    const size_t wb = scratch_bytes(m, kTmax, kTmax, cand_rows());
    uint8_t* w = dalloc<uint8_t>(wb, err);
    if (!w) return false;
    carve(win, w, kTmax, kTmax);
    use(win);
    const int64_t n = hp.n_embd;
    if (!(parts = dalloc<float>((size_t) kTmax * hp.n_used * n, err)) || !(eg = dalloc<float>((size_t) hp.n_ff_exp, err)) ||
        !(eu = dalloc<float>((size_t) hp.n_ff_exp, err)) || !(xq3 = dalloc<uint8_t>(q8((int) hp.n_ff_exp), err)) ||
        !(amax = dalloc<int32_t>(kTmax, err)) || !(dbnd = dalloc<int32_t>(2, err)) ||
        !(hbnd = dalloc<int32_t>((size_t) hp.n_head + 1, err)))
        return false;
    {
        cublasHandle_t hb = nullptr;
        if (cublasCreate(&hb) != CUBLAS_STATUS_SUCCESS) return m.fail(err, "cuBLAS on GPU " + std::to_string(dev));
        blas = hb;
    }
    if (last) {
        if (!(logits = dalloc<float>((size_t) kTmax * hp.n_vocab, err))) return false;
        const int64_t hw = mtp_hw();   // a drafter hidden row: n_embd, DeepSeek-V4's four streams
        if (m.mtp_on && (!(mtp_e = dalloc<float>((size_t) kTmax * n, err)) || !(mtp_h = dalloc<float>((size_t) kTmax * hw, err)) ||
                         !(mtp_cat = dalloc<float>((size_t) kTmax * 2 * n, err)) || !(mtp_z = dalloc<float>((size_t) kTmax * n, err)) ||
                         !(mtp_logits = dalloc<float>((size_t) kTmax * hp.n_vocab, err)) ||
                         !(mtp_amax = dalloc<int32_t>(kTmax, err))))
            return false;
        if (m.mtp_on) {   // the drafter's graph inputs and outputs (mtp.cpp)
            if (!(mtp_hin = dalloc<float>((size_t) kTmax * hw, err)) || !(mtp_chain = dalloc<float>((size_t) kTmax * n, err)) ||
                !(mtp_zlast = dalloc<float>((size_t) hw, err)) ||
                !(mtp_hnlast = dalloc<float>((size_t) n, err)) || !(mtp_prob = dalloc<float>(kTmax, err)) ||
                !(mtp_pos = dalloc<int64_t>(1, err)) || !(d_zero_mask = dalloc<uint8_t>((size_t) kTmax * hp.n_used, err)))
                return false;
            cudaMemset(d_zero_mask, 0, (size_t) kTmax * hp.n_used);
            // coupled draft sampling: the sampler's parameters, ring and scratch (none: the option stays off)
            const size_t cds = strata::kernels::coupled_draft_scratch_bytes((int) hp.n_vocab);
            if (cds > 0 && (!(cd_params = dalloc<strata::kernels::SamplerParams>(1, err)) ||
                            !(cd_ring = dalloc<int32_t>((size_t) kCdRing, err)) || !(cd_scratch = dalloc<uint8_t>(cds, err))))
                return false;
            if (cudaMallocHost(&h_mtp_pos, sizeof(int64_t)) != cudaSuccess || cudaMallocHost(&h_mtp_tok, kTmax * sizeof(int32_t)) != cudaSuccess ||
                cudaMallocHost(&h_mtp_prob, kTmax * sizeof(float)) != cudaSuccess)
                return m.fail(err, "the drafter's pinned words");
        }
    }
    if (cudaMallocHost(&h_embd, (size_t) kTmax * n * sizeof(float)) != cudaSuccess) return m.fail(err, "pinned host buffers");
    h_embd_rows = kTmax;
    if (mmq::built()) mq = std::make_unique<mmq::Context>();
    return true;
}

// ------------------------------------------------------------------ probes (synchronous; only with a recorder)

void Stage::probe(const char* name, int il, float* d, int64_t ne0, int64_t ne1, int64_t ne2) {
    if (!m.rec) return;
    const int64_t n = ne0 * ne1 * ne2;
    std::vector<float> hv((size_t) n);
    cudaStreamSynchronize(s);
    cudaMemcpy(hv.data(), d, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost);
    m.rec->f32(name, il, hv.data(), ne0, ne1, ne2);
    // the write-back (the oracle's values) on the compute stream, waited for: a plain cudaMemcpy from pageable memory returns
    // once the bytes are staged, and the stream is non-blocking - the next kernel ran while the DMA still wrote the buffer's
    // tail (DeepSeek-V4's q: heads 34-63 of 64 came out un-roped in the oracle run, 2026-10-08)
    cudaMemcpyAsync(d, hv.data(), (size_t) n * sizeof(float), cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);
}

void Stage::probe_i32(const char* name, int il, int32_t* d, int64_t n) {
    if (!m.rec) return;
    std::vector<int32_t> hv((size_t) n);
    cudaStreamSynchronize(s);
    cudaMemcpy(hv.data(), d, (size_t) n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    m.rec->i32(name, il, hv.data(), n);
    cudaMemcpyAsync(d, hv.data(), (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice, s);   // see probe()
    cudaStreamSynchronize(s);
}

// ------------------------------------------------------------------ projections

void Stage::mm_t(const DT& w, const void* xq_, float* y_, int T) {
    const int ni = (int) w.n_in(), no = (int) w.n_out();
    for (int c0 = 0; c0 < T; c0 += kTmax)
        kn::native_mmvq(w.type, w.d, static_cast<const uint8_t*>(xq_) + (size_t) c0 * q8(ni), y_ + (size_t) c0 * no, ni,
                        no, std::min(kTmax, T - c0), s);
}

void Stage::quant(const float* x, void* xq_, int n, int ncols) {
    if (ncols <= kTmax) kn::native_quantize_q8_1(x, xq_, n, ncols, s);
    else kn::native_quantize_q8_1(x, xq_, n * ncols, 1, s);   // the layout is the same: one vector of n * ncols
}

bool Stage::mmq_ok(const DT& w) const {
    static const bool off = std::getenv("STRATA_X_NO_DENSE_MMQ") != nullptr;   // A/B: the dense projections by MMVQ
    return !off && mq && mb.xq && mmq::supported(w.type) && w.n_in() % 256 == 0 && mmq::fits(w.type, w.n_out());
}

/// A dense projection over T > 8 columns through llama.cpp's MMQ in its ORDINARY mode (no expert bounds, no row map:
/// what ggml_cuda_mul_mat_q runs for every dense matmul; build/mmq_dense_parity checks it per type and shape in both of
/// MMQ's launch modes).  STRATA_X_NO_DENSE_MMQ=1 is the A/B arm (MMVQ in groups of 8 columns).  NB: the 4-layer
/// slice's long-prompt continuation sits on a near-tie (two logits 0.05 apart) that flips with the sub-batch size;
/// every path stays within +-0.035 of llama.cpp's logits there (moex/docs/MOEX.md 7d) - compare logits, not tokens.
void Stage::mmq_dense(const DT& w, const float* x, float* y_, int T) {
    const int ni = (int) w.n_in(), no = (int) w.n_out();
    mmq::quantize(x, nullptr, mb.xq, w.type, ni, ni, T, s);
    mmq::Product p;
    p.w = w.d;
    p.type = w.type;
    p.w_rows = no;
    p.w_cols = ni;
    p.expert_bytes = mmq::matrix_bytes(w.type, no, ni);
    p.n = 1;
    p.xq = mb.xq;
    p.bounds = nullptr;
    p.ids = nullptr;
    p.total_rows = T;
    p.max_rows = T;
    p.dst = y_;
    p.ld_dst = no;
    mq->run(p, s);
}

void Stage::pj(const DT& w, const float* x, const void* xqn, float* y_, int T) {
    // moex (the GGUF set): a BF16 matrix - llama.cpp's lossless set keeps DeepSeek-V4.1's FP8 projections so, exactly - by
    // gemv_dt on the F32 input (the release's weights, F32 arithmetic); the quantized ones as before
    if (w.type == GGML_TYPE_BF16 || w.type == kDtFP8) {   // (moex §70: or those FP8 values, STRATA_DSV41_DENSE_FP8)
        gemv_dt(w, x, y_, (int) w.n_in(), (int) w.n_out(), T);
        return;
    }
    if (T > kTmax && mmq_ok(w)) mmq_dense(w, x, y_, T);
    else mm_t(w, xqn, y_, T);
}

void Stage::gemv_t(const float* W, const float* x, float* y_, int n_in, int n_out, int T) {
    if (T <= kTmax || !blas) {
        gemv_f32_w(W, x, y_, n_in, n_out, T, s);
        return;
    }
    // row-major y (T, n_out) = x (T, n_in) W^T, W row-major (n_out, n_in): column-major y^T = W^T' x^T'
    cublasHandle_t hb = (cublasHandle_t) blas;
    const float one = 1.0f, zero = 0.0f;
    if (cublasSetStream(hb, s) != CUBLAS_STATUS_SUCCESS ||
        cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, W, n_in, x, n_in, &zero, y_, n_out) != CUBLAS_STATUS_SUCCESS) {
        // a GEMM cuBLAS refused left y_ as it was (the routers' logits of an earlier chunk): the plain kernel instead
        static std::atomic<int> said{0};
        if (said.fetch_add(1) == 0) fprintf(stderr, "strata-x: cuBLAS refused an F32 GEMM (%d x %d x %d): the plain kernel ran it\n", n_out, T, n_in);
        gemv_f32_w(W, x, y_, n_in, n_out, T, s);
    }
}

void Stage::gemv_dt(const DT& w, const float* x, float* y_, int n_in, int n_out, int T) {
    // moex (ERNIE55_PLAN, the hill-climb; opt-in STRATA_X_TC3=1): a prompt sub-batch's BF16 / FP8 projection on the tensor
    // cores - the weights' exact BF16 values (FP8: fp8_bf16), x split exactly into three BF16 planes (split3_bf16), three
    // GEMMs with F32 accumulation, the small planes first (y = W lo; y += W mid; y += W hi).  The products are exact; the sums
    // are F32 in another order than the SGEMM's (and the tensor cores' own adder): not the same bits - a KLD check's switch.
    // Held at the cards' power limit: 0.55-0.79x the SGEMM's time (sgemm_bench).  Falls back where the scratch is too small.
    static const bool tc3 = std::getenv("STRATA_X_TC3") && std::getenv("STRATA_X_TC3")[0] == '1';
    if (tc3 && T > kTmax && blas && bf16_f32 && (w.type == kDtFP8 || w.type == GGML_TYPE_BF16)) {
        const size_t scb = bf16_f32_n * sizeof(float);
        const size_t wbytes = w.type == kDtFP8 ? (((size_t) n_in * n_out * sizeof(uint16_t) + 255) & ~(size_t) 255) : 0;
        const size_t xbytes = 3 * (size_t) T * n_in * sizeof(uint16_t);
        if (wbytes + xbytes <= scb) {
            uint8_t* sc = reinterpret_cast<uint8_t*>(bf16_f32);
            const uint16_t* wb = static_cast<const uint16_t*>(w.d);
            if (w.type == kDtFP8) {
                fp8_bf16(static_cast<const uint8_t*>(w.d), static_cast<const uint8_t*>(w.s), reinterpret_cast<uint16_t*>(sc),
                         (size_t) n_in * n_out, s);
                wb = reinterpret_cast<const uint16_t*>(sc);
            }
            uint16_t* xp = reinterpret_cast<uint16_t*>(sc + wbytes);
            split3_bf16(x, xp, (size_t) T * n_in, s);
            cublasHandle_t hb = (cublasHandle_t) blas;
            bool ok = cublasSetStream(hb, s) == CUBLAS_STATUS_SUCCESS;
            const float one = 1.0f, zero = 0.0f;
            for (int p = 2; ok && p >= 0; --p)
                ok = cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, wb, CUDA_R_16BF, n_in, xp + (size_t) p * T * n_in,
                                  CUDA_R_16BF, n_in, p == 2 ? &zero : &one, y_, CUDA_R_32F, n_out, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
            if (ok) return;
            static std::atomic<int> said{0};
            if (said.fetch_add(1) == 0) fprintf(stderr, "strata-x: cuBLAS refused a BF16 GEMM (STRATA_X_TC3): the F32 path ran it\n");
        }
    }
    if (w.type == kDtFP8) {   // moex (DSV41_PLAN §70): the BF16 path below, step for step, from the FP8 copy of the same values
        const uint8_t* wq = static_cast<const uint8_t*>(w.d);
        const uint8_t* ws = static_cast<const uint8_t*>(w.s);
        if (T <= kTmax || !blas || !bf16_f32 || bf16_f32_n < (size_t) n_in) {
            gemv_fp8_w(wq, ws, x, y_, n_in, n_out, T, s);
            return;
        }
        if ((size_t) n_in * n_out <= bf16_f32_n) {
            fp8_widen(wq, ws, bf16_f32, (size_t) n_in * n_out, s);
            gemv_t(bf16_f32, x, y_, n_in, n_out, T);
            return;
        }
        cublasHandle_t hb = (cublasHandle_t) blas;
        const int R = (int) std::min<size_t>(bf16_f32_n / (size_t) n_in, (size_t) n_out);
        const float one = 1.0f, zero = 0.0f;
        for (int r0 = 0; r0 < n_out; r0 += R) {
            const int nr = std::min(R, n_out - r0);
            fp8_widen(wq + (size_t) r0 * n_in, ws + (size_t) r0 * (n_in / 32), bf16_f32, (size_t) nr * n_in, s);
            if (cublasSetStream(hb, s) != CUBLAS_STATUS_SUCCESS ||
                cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, nr, T, n_in, &one, bf16_f32, n_in, x, n_in, &zero, y_ + r0, n_out) !=
                    CUBLAS_STATUS_SUCCESS) {
                static std::atomic<int> said{0};
                if (said.fetch_add(1) == 0) fprintf(stderr, "strata-x: cuBLAS refused a widened FP8 GEMM block: the plain kernel ran the matrix\n");
                gemv_fp8_w(wq, ws, x, y_, n_in, n_out, T, s);
                return;
            }
        }
        return;
    }
    if (w.type != GGML_TYPE_BF16) {
        gemv_t(w.f(), x, y_, n_in, n_out, T);
        return;
    }
    const uint16_t* wb = static_cast<const uint16_t*>(w.d);
    // moex (A/B against llama.cpp's CPU oracle, Hy4-flash's BF16 sets): STRATA_X_BF16_ACT=1 rounds a decode window's input to
    // BF16 first, as ggml-cpu's BF16 mul_mat does (its vec_dot_type); off, the product takes the F32 input as it is
    static const bool bf16_act = [] { const char* e = std::getenv("STRATA_X_BF16_ACT"); return e && e[0] == '1'; }();
    if (bf16_act && T <= kTmax && hp.arch == Arch::hy4 && bf16_f32 && bf16_f32_n >= (size_t) T * n_in) {
        hy4_round_bf16(x, bf16_f32, (int64_t) T * n_in, s);
        gemv_bf16_w(wb, bf16_f32, y_, n_in, n_out, T, s);
        return;
    }
    if (T <= kTmax || !blas || !bf16_f32 || bf16_f32_n < (size_t) n_in) {
        gemv_bf16_w(wb, x, y_, n_in, n_out, T, s);
        return;
    }
    if ((size_t) n_in * n_out <= bf16_f32_n) {
        bf16_widen(wb, bf16_f32, (size_t) n_in * n_out, s);   // (stream order: the GEMM before has read it)
        gemv_t(bf16_f32, x, y_, n_in, n_out, T);
        return;
    }
    // moex (the GGUF set's BF16 projections): a matrix larger than the scratch (the engram's wkv, 157M values; the head, 662M)
    // in blocks of rows - each block widened, then its output rows by one GEMM (y's leading dimension stays n_out)
    cublasHandle_t hb = (cublasHandle_t) blas;
    const int R = (int) std::min<size_t>(bf16_f32_n / (size_t) n_in, (size_t) n_out);
    const float one = 1.0f, zero = 0.0f;
    for (int r0 = 0; r0 < n_out; r0 += R) {
        const int nr = std::min(R, n_out - r0);
        bf16_widen(wb + (size_t) r0 * n_in, bf16_f32, (size_t) nr * n_in, s);
        if (cublasSetStream(hb, s) != CUBLAS_STATUS_SUCCESS ||
            cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, nr, T, n_in, &one, bf16_f32, n_in, x, n_in, &zero, y_ + r0, n_out) !=
                CUBLAS_STATUS_SUCCESS) {
            static std::atomic<int> said{0};
            if (said.fetch_add(1) == 0) fprintf(stderr, "strata-x: cuBLAS refused a widened BF16 GEMM block: the plain kernel ran the matrix\n");
            gemv_bf16_w(wb, x, y_, n_in, n_out, T, s);
            return;
        }
    }
}

void Stage::heads_mm(const DT& w, const float* x_hm, float* y_hm, int H, int n_in, int n_out, int T) {
    if (w.type == GGML_TYPE_BF16 || w.type == kDtFP8) {   // moex (the GGUF set): each head's rows by gemv_dt on its F32 rows (head-major)
        const bool f8 = w.type == kDtFP8;   // (§70: an FP8 head's values n_out x n_in bytes, its scales a 32nd of that)
        const size_t hb = (size_t) n_out * n_in * (f8 ? 1 : sizeof(uint16_t));
        // moex (DSV41_BACKLOG #138, DSV41_PLAN §75): the FP8 heads in one launch - the same rows, products and sums as the
        // loop below, which the eight separate launches of a decode layer cost ~14 us (§34's WOA_BATCH on Q8_0)
        if (f8 && T >= 1 && T <= 8 && n_in % 32 == 0) {
            gemv_fp8_heads(static_cast<const uint8_t*>(w.d), static_cast<const uint8_t*>(w.s), x_hm, y_hm, n_in, n_out, T, H, s);
            return;
        }
        for (int hh = 0; hh < H; ++hh) {
            DT v = w;
            v.d = static_cast<const uint8_t*>(w.d) + hh * hb;
            if (f8) v.s = static_cast<const uint8_t*>(w.s) + hh * (hb / 32);
            v.ne[0] = n_in;
            v.ne[1] = n_out;
            v.ne[2] = v.ne[3] = 1;
            v.bytes = hb;
            gemv_dt(v, x_hm + (size_t) hh * T * n_in, y_hm + (size_t) hh * T * n_out, n_in, n_out, T);
        }
        return;
    }
    const size_t head_w = (size_t) n_out * kn::native_mmvq_weight_bytes(w.type, n_in, 1);
    if (T > kTmax && mq && mb.xq && mmq::supported(w.type) && n_in % 256 == 0 && mmq::fits(w.type, n_out)) {
        // one MMQ launch, each head an "expert" of T rows (llama.cpp's MoE mode; the rows are already head-major)
        mmq::quantize(x_hm, nullptr, mb.xq, w.type, n_in, n_in, (int64_t) H * T, s);
        if (hbnd_T != T || hbnd_H != H) {   // (Hy4-flash: the head count is the layer's - 32 or 64)
            strided_bounds(hbnd, H + 1, T, s);
            hbnd_T = T;
            hbnd_H = H;
        }
        mmq::Product p;
        p.w = w.d;
        p.type = w.type;
        p.w_rows = n_out;
        p.w_cols = n_in;
        p.expert_bytes = head_w;
        p.n = H;
        p.xq = mb.xq;
        p.bounds = hbnd;
        p.ids = mb.ident;
        p.total_rows = (int64_t) H * T;
        p.max_rows = T;
        p.dst = y_hm;
        p.ld_dst = n_out;
        mq->run(p, s);
        return;
    }
    quant(x_hm, xq, H * T * n_in, 1);
    if (w.type == 8) {   // Q8_0: every head in one launch
        heads_q8_0(w.d, xq, y_hm, H, n_in, n_out, T, s);
    } else {
        for (int hh = 0; hh < H; ++hh)
            kn::native_mmvq(w.type, static_cast<const uint8_t*>(w.d) + hh * head_w,
                            static_cast<const uint8_t*>(xq) + (size_t) hh * T * q8(n_in), y_hm + (size_t) hh * T * n_out,
                            n_in, n_out, T, s);
    }
}

// ------------------------------------------------------------------ a layer's parts

/// One mHC half's front, T tokens: mixes from the RMS-normed flat streams, the coefficients (into coef_r, valid until
/// the matching mhc_post_w), the weighted read into cur_r, the norm.
void Stage::mhc_half(int il, int T, const float* Rr, float* cur_r, float* coef_r, bool attn) {
    const Layer& l = L[(size_t) il];
    const DT& fn = attn ? l.hc_attn_fn : l.hc_ffn_fn;
    const DT& sc = attn ? l.hc_attn_scale : l.hc_ffn_scale;
    const DT& bs = attn ? l.hc_attn_base : l.hc_ffn_base;
    const DT& norm = attn ? l.attn_norm : l.ffn_norm;
    const int n = (int) hp.n_embd;
    // STRATA_DSV41_HC_SPLIT=1 (A/B): V4.1's decode windows split each mixer row across blocks (gemv_f32_split)
    static const bool hc_split = [] {
        const char* e = std::getenv("STRATA_DSV41_HC_SPLIT");
        return e && e[0] == '1';
    }();
    // moex (DSV41_PLAN §34), opt-in STRATA_DSV41_MIX_FUSED=1: a one-token window's split gemv partials, then the reduce, the
    // coefficients, the pre-mix, the carried pre's copy and the norm in one launch (mhc_mix_in_w1) - the same bits
    static const bool mix_fused = [] {
        const char* e = std::getenv("STRATA_DSV41_MIX_FUSED");
        return e && e[0] == '1';
    }();
    // moex (DSV41_BACKLOG #105), opt-in STRATA_DSV41_MIX_SIDE=1: in a one-token decode window V4.1 reads the half with the
    // PREVIOUS half's carried pre, so this half's coefficients - the streams' norm, the split gemv, the Sinkhorn chain - are
    // first needed at its post-mix: they run on s3 beside the half's attention or MoE, joined in mix_out.  The carried pre
    // alternates between cpre and cpre_b by the half's parity in the token (half 2 il + !attn reads one and writes the other),
    // so cpre holds it wherever the window's halves hand it on - the stage boundary (layer 19's FFN, an odd half) and the head
    // (layer 39's).  The unfused kernels MIX_FUSED's launch is bit for bit: the same bits.  Decode windows only (their own
    // scratch): a prompt chunk's rows may meet a half alone and the next in a batch, which would break the parity
    static const bool mix_side = [] {
        const char* e = std::getenv("STRATA_DSV41_MIX_SIDE");
        return e && e[0] == '1';
    }();
    if (mix_wait) {   // (a half whose post-mix did not run: its coefficients joined before this half's)
        cudaStreamWaitEvent(s, mix_join, 0);
        mix_wait = false;
    }
    if (hp.arch == Arch::hy4) {   // Hy4-flash's iHC: 8 mixes (pre, post), the identity for the streams' mix (hy4_ihc_coef_w)
        rms_norm(Rr, nullptr, flat, T, 4 * n, rms_eps(), s);
        if (fn.type == 0) {
            gemv_t(fn.f(), flat, mixes, 4 * n, 8, T);
        } else {   // the SPARK-4Bit sets' Q8_0 / NVFP4 mixers: a quantized projection (llama.cpp rounds the input to Q8_0 there)
            quant(flat, xq, 4 * n, T);
            pj(fn, flat, xq, mixes, T);
        }
        probe("hc_mixes", il, mixes, 8);
        hy4_ihc_coef_w(mixes, sc.f(), bs.f(), coef_r, hp.hc_eps, hp.hc_mag, T, s);
        probe("hc_pre", il, coef_r, 4);
        probe("hc_post_gate", il, coef_r + 4, 4);
        mhc_pre_w(Rr, coef_r, cur_r, n, T, s);
        rms_norm(cur_r, norm.f(), cur_r, T, n, rms_eps(), s);
        probe(attn ? "attn_norm" : "ffn_norm", il, cur_r, n);
        return;
    }
    if (mix_side && hp.v41 && fn.type == 0 && hc_split && hc_part && T == 1 && n <= 5 * 1024 && !m.rec && il < hp.n_layer &&
        cpre_b && cpre == win.cpre && s3) {
        const int half = 2 * il + (attn ? 0 : 1);
        const size_t off = (size_t) (coef_r - coef);
        float* cp_in = ((half & 1) ? cpre_b : cpre) + off;    // the previous half's coefficients: this half's read
        float* cp_out = ((half & 1) ? cpre : cpre_b) + off;   // this half's: the next half's read
        cudaEventRecord(mix_fork, s);
        cudaStreamWaitEvent(s3, mix_fork, 0);
        rms_norm(Rr, nullptr, flat, T, 4 * n, rms_eps(), s3);
        gemv_f32_split(fn.f(), flat, mixes, hc_part, 4 * n, 24, T, s3);
        mhc_coef_w(mixes, sc.f(), bs.f(), coef_r, hp.hc_eps, (int) hp.sinkhorn_iters, T, s3);
        cudaMemcpyAsync(cp_out, coef_r, (size_t) T * 24 * sizeof(float), cudaMemcpyDeviceToDevice, s3);
        cudaEventRecord(mix_join, s3);
        mix_wait = true;
        mhc_pre_w(Rr, cp_in, cur_r, n, T, s);
        rms_norm(cur_r, norm.f(), cur_r, T, n, rms_eps(), s);
        return;
    }
    if (hp.arch == Arch::ernie) ernie_rms_eo_w(Rr, flat, T, 4 * n, rms_eps(), s);   // ERNIE-5.5: 1 / (sqrt(mean) + eps)
    else rms_norm(Rr, nullptr, flat, T, 4 * n, rms_eps(), s);
    // moex (the ERNIE-5.5 hill-climb): a decode window's token - the Sinkhorn chain (mhc_coef_w, whose coefficients the half's
    // post-mix reads first) on s3 beside the half's attention or MoE, joined at mix_out; here the read with its four pre
    // coefficients computed in place and the norm in one launch (mhc_pre_norm_w1), the mixer's rows as their 8 warps' totals
    // (gemv_f32_w1_warps) summed there - the same bits.  STRATA_ERNIE_MIX_SIDE=0: off
    static const bool ernie_side = [] {
        const char* e = std::getenv("STRATA_ERNIE_MIX_SIDE");
        return !(e && e[0] == '0');
    }();
    const bool e_side = ernie_side && hp.arch == Arch::ernie && T == 1 && !m.rec && s3 && mixes == win.mixes && n <= 5 * 1024;
    if (e_side && fn.type == 0 && hc_part && gemv_f32_w1_warps(fn.f(), flat, hc_part, 4 * n, 24, s)) {
        mhc_pre_norm_w1(mixes, sc.f(), bs.f(), hp.hc_eps, Rr, cur_r, norm.f(), n, rms_eps(), s, hc_part);
        cudaEventRecord(mix_fork, s);
        cudaStreamWaitEvent(s3, mix_fork, 0);
        mhc_coef_w(mixes, sc.f(), bs.f(), coef_r, hp.hc_eps, (int) hp.sinkhorn_iters, T, s3);
        cudaEventRecord(mix_join, s3);
        mix_wait = true;
        return;
    }
    if (mix_fused && hp.v41 && fn.type == 0 && hc_split && hc_part && T == 1 && n <= 5 * 1024 && !m.rec) {
        gemv_f32_split_part(fn.f(), flat, hc_part, 4 * n, 24, T, s);
        float* cp = cpre + (coef_r - coef);
        mhc_mix_in_w1(hc_part, gemv_f32_split_parts(4 * n), sc.f(), bs.f(), mixes, coef_r, cp, Rr, cur_r, norm.f(), n, hp.hc_eps,
                      (int) hp.sinkhorn_iters, rms_eps(), s);
        return;
    }
    if (fn.type == 0 && hc_split && hc_part && T <= kTmax) {
        gemv_f32_split(fn.f(), flat, mixes, hc_part, 4 * n, 24, T, s);
    } else if (fn.type == 0) {   // DeepSeek-V4's mixers are F32 (llama.cpp's F32 matmul: no activation quantization)
        gemv_t(fn.f(), flat, mixes, 4 * n, 24, T);
    } else {              // GLM's are Q8_0 (Q8_1 activations, as llama.cpp's CUDA MMVQ)
        quant(flat, xq, 4 * n, T);
        pj(fn, flat, xq, mixes, T);
    }
    probe("hc_mixes", il, mixes, 24);
    if (e_side) {   // (the mixer's rows by the chunk kernel: STRATA_X_F32_WARPS=0)
        cudaEventRecord(mix_fork, s);
        cudaStreamWaitEvent(s3, mix_fork, 0);
        mhc_coef_w(mixes, sc.f(), bs.f(), coef_r, hp.hc_eps, (int) hp.sinkhorn_iters, T, s3);
        cudaEventRecord(mix_join, s3);
        mix_wait = true;
        mhc_pre_norm_w1(mixes, sc.f(), bs.f(), hp.hc_eps, Rr, cur_r, norm.f(), n, rms_eps(), s);
        return;
    }
    mhc_coef_w(mixes, sc.f(), bs.f(), coef_r, hp.hc_eps, (int) hp.sinkhorn_iters, T, s);
    probe("hc_pre", il, coef_r, 4);
    probe("hc_post", il, coef_r + 4, 4);
    probe("hc_comb", il, coef_r + 8, 4, 4);
    if (hp.v41) {   // DeepSeek-V4.1 (model.py Block.forward): the read takes the PREVIOUS half's pre - the last FFN's for the
                    // attention (stream 0 before layer 0), this layer's attention's for the FFN - and this half's goes on
        float* cp = cpre + (coef_r - coef);   // the same rows of the carried pre (the prompt path's sub-batches)
        mhc_pre_w(Rr, cp, cur_r, n, T, s);
        cudaMemcpyAsync(cp, coef_r, (size_t) T * 24 * sizeof(float), cudaMemcpyDeviceToDevice, s);
    } else {
        mhc_pre_w(Rr, coef_r, cur_r, n, T, s);
    }
    probe(attn ? "hc_attn_pre" : "hc_ffn_pre", il, cur_r, n);
    rms_norm(cur_r, norm.f(), cur_r, T, n, rms_eps(), s);
    probe(attn ? "attn_norm" : "ffn_norm", il, cur_r, n);
}

void Stage::kda(int il, int T, const float* in, float* out_r, bool keep) {
    Layer& l = L[(size_t) il];
    const int H = (int) hp.n_head, D = (int) hp.hd_kda, din = H * D, n = (int) hp.n_embd, dc = (int) hp.d_conv;
    quant(in, xq, n, T);
    pj(l.wq, in, xq, pq, T);
    pj(l.wk, in, xq, pk, T);
    pj(l.wv, in, xq, pv, T);
    pj(l.f_a, in, xq, fa, T);
    pj(l.beta, in, xq, beta, T);
    pj(l.g_a, in, xq, ga, T);
    const size_t cs_n = (size_t) din * (dc - 1);
    if (keep) {   // what an MTP rollback needs: the conv state before the window and the window's conv inputs
        cudaMemcpyAsync(l.conv_prev, l.conv, 3 * cs_n * sizeof(float), cudaMemcpyDeviceToDevice, s);
        for (int t = 0; t < T; ++t) {
            float* ci = l.conv_in + (size_t) t * 3 * din;
            cudaMemcpyAsync(ci, pq + (size_t) t * din, din * sizeof(float), cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(ci + din, pk + (size_t) t * din, din * sizeof(float), cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(ci + 2 * din, pv + (size_t) t * din, din * sizeof(float), cudaMemcpyDeviceToDevice, s);
        }
    }
    kda_conv_silu_w(l.conv, pq, l.conv_q.f(), cq, din, dc, T, s);
    kda_conv_silu_w(l.conv + cs_n, pk, l.conv_k.f(), ck, din, dc, T, s);
    kda_conv_silu_w(l.conv + 2 * cs_n, pv, l.conv_v.f(), cv, din, dc, T, s);
    probe("kda_q_conv", il, cq, D, H);
    probe("kda_k_conv", il, ck, D, H);
    probe("kda_v_conv", il, cv, D, H);
    quant(fa, xq2, D, T);
    pj(l.f_b, fa, xq2, g, T);
    kda_gates_w(g, l.dt_b.f(), l.a.f(), hp.kda_lb, beta, H, D, T, s);
    probe("kda_g1", il, g, D, H);
    probe("kda_beta", il, beta, 1, H);
    kda_l2norm(cq, T * H, D, 1e-6f, 1.0f / std::sqrt((float) D), s);
    kda_l2norm(ck, T * H, D, 1e-6f, 1.0f, s);
    kda_scan(l.S, cq, ck, cv, g, beta, o, H, D, T, keep ? l.snaps : nullptr, s);
    probe("kda_scan_out", il, o, D, H);
    quant(ga, xq2, D, T);
    pj(l.g_b, ga, xq2, g2, T);
    probe("kda_g2", il, g2, D, H);
    kda_out_gate(o, g2, l.o_norm.f(), y, T * H, D, hp.rms_eps, s);
    quant(y, xq2, din, T);
    pj(l.wo, y, xq2, out_r, T);
    probe("kda_out", il, out_r, n);
}

void Stage::dsa(int il, int64_t pos0, int T, const float* in, float* out_r) {
    const Layer& l = L[(size_t) il];
    const int H = (int) hp.n_head, Lq = (int) hp.q_lora, Lkv = (int) hp.kv_lora, Dk = (int) hp.k_mla, Dv = (int) hp.v_mla;
    const int IH = (int) hp.idx_heads, ID = (int) hp.idx_dim, KP = (int) hp.kpool, n = (int) hp.n_embd;
    const int n_top = (int) (hp.idx_topk / KP);
    quant(in, xq, n, T);
    pj(l.q_a, in, xq, qr, T);
    rms_norm(qr, l.q_a_norm.f(), qr, T, Lq, hp.rms_eps, s);
    probe("q_resid", il, qr, Lq);
    pj(l.kv_a, in, xq, kv, T);
    rms_norm(kv, l.kv_a_norm.f(), kv, T, Lkv, hp.rms_eps, s);
    probe("kv_cmpr", il, kv, Lkv);
    dsa_store_latent_w(kv, l.kv_cache, pos0, Lkv, T, s, gpos);
    // the indexer's per-token key and gate, into the pool ring
    pj(l.idx_k, in, xq, ikw, T);
    layer_norm_w(ikw, l.idx_k_norm.f(), l.idx_k_norm_b.f(), ikw, T, ID, hp.ln_eps, s);
    probe("indexer_k", il, ikw, ID);
    pj(l.idx_gate, in, xq, igw, T);
    probe("indexer_gate", il, igw, ID);
    dsa_ring_store_w(ikw, l.ik_ring, m.ring_rows, pos0, ID, T, s, gpos);
    dsa_ring_store_w(igw, l.ig_ring, m.ring_rows, pos0, ID, T, s, gpos);
    gemv_dt(l.idx_proj, in, w32, n, IH, T);
    scale_f32(w32, T * IH, 1.0f / std::sqrt((float) (ID * IH)), s);
    // queries: q from q_resid; absorb wk_b per head (head-major so each head's T columns are contiguous)
    quant(qr, xq2, Lq, T);
    pj(l.q_b, qr, xq2, q, T);
    pj(l.idx_q_b, qr, xq2, iq, T);
    probe("indexer_q", il, iq, ID, IH);
    permute_abc(q, q_hm, T, H, Dk, s);
    heads_mm(l.k_b, q_hm, qabs_hm, H, Dk, Lkv, T);
    permute_abc(qabs_hm, qabs, H, T, Lkv, s);
    probe("q_absorbed", il, qabs, Lkv, H);
    dsa_kpool_w(l.ik_ring, l.ig_ring, m.ring_rows, l.idx_ape.f(), l.pooled, pos0, T, ID, KP, s, gpos);
    if (m.rec && (pos0 + 1) % KP == 0) probe("indexer_pool_k_new", il, l.pooled + (pos0 / KP) * ID, ID);
    probe("indexer_weights", il, w32, IH);
    dsa_scores_w(iq, l.pooled, w32, score, pos0, T, KP, IH, ID, m.max_pool, s, gpos);
    if (m.rec && (pos0 + 1) / KP > 0) probe("indexer_score", il, score, (int) ((pos0 + 1) / KP));
    dsa_select_w(score, m.max_pool, n_top, KP, pos0, T, hp.select_tail, cells, m.max_cells, s, gpos);
    // a prompt sub-batch: all its queries in one tensor-core launch (no split-K scratch); a decode window, or a card
    // without mma.sync m16n8k16: the split-K kernels, a few queries at a time
    if (!(T > kTmax && dsa_attend_mma(qabs, l.kv_cache, cells, m.max_cells, pos0, T, n_top, KP, hp.select_tail,
                                       1.0f / std::sqrt((float) Dk), ctx, H, Lkv, s)))
        for (int t0 = 0; t0 < T; t0 += kAttnQ)
            dsa_attend_w(qabs + (size_t) t0 * H * Lkv, l.kv_cache, cells + (size_t) t0 * m.max_cells, m.max_cells, pos0 + t0,
                         std::min(kAttnQ, T - t0), n_top, KP, hp.select_tail, 1.0f / std::sqrt((float) Dk),
                         ctx + (size_t) t0 * H * Lkv, partial, H, Lkv, s, gpos);
    probe("kqv", il, ctx, Lkv, 1, H);
    permute_abc(ctx, ctx_hm, T, H, Lkv, s);
    heads_mm(l.v_b, ctx_hm, o2_hm, H, Lkv, Dv, T);
    permute_abc(o2_hm, o2, H, T, Dv, s);
    probe("kqv_out", il, o2, H * Dv);
    quant(o2, xq, H * Dv, T);
    pj(l.wo, o2, xq, out_r, T);
    probe("attn_out", il, out_r, n);
}

/// MiMo's attention over T tokens (llama.cpp mimo2.cpp graph 85-152): the fused QKV projection, NEOX RoPE on Q and K (the
/// sliding-window layers' base, or the full layers'), K and V into the layer's FP16 cache, GQA with the head's sink, the
/// output projection, the value scale.
void Stage::gqa(int il, int64_t pos0, int T, const float* in, float* out_r) {
    const Layer& l = L[(size_t) il];
    const int n = (int) hp.n_embd, H = (int) hp.n_head, nkv = (int) hp.n_kv(il), dk = (int) hp.hd_k, dv = (int) hp.hd_v;
    const int W = (int) hp.qkv_width(il);
    const bool swa = hp.is_swa(il);
    quant(in, xq, n, T);
    pj(l.wqkv, in, xq, qkv, T);
    probe("wqkv", il, qkv, W);
    const float base = swa ? hp.rope_base_swa : hp.rope_base;
    rope_neox_w(qkv, W, 0, H, dk, (int) hp.n_rot, base, pos0, T, s, gpos);
    rope_neox_w(qkv, W, H * dk, nkv, dk, (int) hp.n_rot, base, pos0, T, s, gpos);
    probe("Qcur", il, qkv, dk, H);
    probe("Kcur", il, qkv + (size_t) H * dk, dk, nkv);
    probe("Vcur", il, qkv + (size_t) (H + nkv) * dk, dv, nkv);
    kv_store_w(qkv, W, H * dk, (H + nkv) * dk, l.k_cache, l.v_cache, l.kv_rows, nkv, dk, dv, pos0, T, s, gpos);
    // a decode window splits the keys (the full layers' up to the context; the window's 128 in 4); a prompt sub-batch has a
    // block per (token, KV head) already
    const int chunks = T <= kTmax ? (swa ? 4 : m.gqa_chunks) : 1;
    static const bool timing = std::getenv("STRATA_X_ATTN_TIMING") != nullptr;   // the prompt path's attention alone (synced)
    cudaEvent_t ta = nullptr, tb = nullptr;
    if (timing && T > kTmax && !gpos) {
        cudaEventCreate(&ta);
        cudaEventCreate(&tb);
        cudaEventRecord(ta, s);
    }
    // a decode window's full layer: its rows share each K / V tile (gqa_decode_shared: a row's result whatever the window;
    // STRATA_X_GQA_SHARED=0: the per-row kernel)
    static const bool shared = !(std::getenv("STRATA_X_GQA_SHARED") && std::getenv("STRATA_X_GQA_SHARED")[0] == '0');
    if (!(shared && !swa && T <= kTmax &&
          gqa_decode_shared(qkv, W, l.k_cache, l.v_cache, l.kv_rows, H, nkv, dk, dv, l.sinks ? l.sinks.f() : nullptr,
                            1.0f / std::sqrt((float) dk), att, gpart, m.gqa_chunks, pos0, T, s, gpos, (int) m.opt.n_ctx)))
        gqa_attend_w(qkv, W, l.k_cache, l.v_cache, l.kv_rows, H, nkv, dk, dv, swa ? (int) hp.n_swa : 0, l.sinks ? l.sinks.f() : nullptr,
                     1.0f / std::sqrt((float) dk), att, gpart, chunks, pos0, T, s, gpos);
    if (ta) {
        cudaEventRecord(tb, s);
        cudaEventSynchronize(tb);
        float ms = 0;
        cudaEventElapsedTime(&ms, ta, tb);
        static double sum_full = 0, sum_swa = 0;
        (swa ? sum_swa : sum_full) += ms;
        if (pos0 + T >= 4096 && (pos0 + T) % 4096 < (int64_t) T && il == 0)
            fprintf(stderr, "strata-x: prompt attention so far: full layers %.1f s, sliding-window layers %.1f s (at position %lld)\n",
                    sum_full / 1000.0, sum_swa / 1000.0, (long long) (pos0 + T));
        cudaEventDestroy(ta);
        cudaEventDestroy(tb);
    }
    probe("kqv_out", il, att, H * dv);
    quant(att, xq2, H * dv, T);
    pj(l.wo, att, xq2, out_r, T);
    probe("attn_out", il, out_r, n);
    if (hp.v_scale != 0.0f) {
        scale_f32(out_r, T * n, hp.v_scale, s);
        probe("attn_out_scaled", il, out_r, n);
    }
}

/// The front of a layer half, T tokens of the residual Rr: mHC's read and norm (GLM), or the pre-norm (MiMo).
void Stage::mix_in(int il, int T, const float* Rr, float* cur_r, float* coef_r, bool attn) {
    if (hp.hc > 1) {
        mhc_half(il, T, Rr, cur_r, coef_r, attn);
        return;
    }
    const Layer& l = L[(size_t) il];
    rms_norm(Rr, (attn ? l.attn_norm : l.ffn_norm).f(), cur_r, T, (int) hp.n_embd, hp.rms_eps, s);
    probe(attn ? "attn_norm" : "ffn_norm", il, cur_r, hp.n_embd);
}

/// The back of a layer half: mHC's write of x into the streams, or the residual add.
void Stage::mix_out(int il, int T, const float* x, float* Rr, const float* coef_r, bool attn) {
    const int n = (int) hp.n_embd;
    if (hp.hc > 1) {
        if (mix_wait) {   // moex (#105): the half's coefficients from s3
            cudaStreamWaitEvent(s, mix_join, 0);
            mix_wait = false;
        }
        mhc_post_w(x, Rr, coef_r, n, T, s);
        return;
    }
    add_f32(Rr, x, Rr, T * n, s);
    if (attn) probe("ffn_inp", il, Rr, n);
}

/// The attention half's mixer of layer il.
void Stage::attention(int il, int64_t pos0, int T, const float* in, float* out_r, bool keep) {
    if (hp.arch == Arch::mimo2) gqa(il, pos0, T, in, out_r);
    else if (hp.arch == Arch::hy4) hy4_attn(il, pos0, T, in, out_r);
    else if (hp.arch == Arch::ernie) ernie_attn(il, pos0, T, in, out_r);
    else if (hp.arch == Arch::dsv4 && hp.v41) dsv41(il, pos0, T, in, out_r);
    else if (hp.arch == Arch::dsv4) dsv4(il, pos0, T, in, out_r);
    else if (hp.is_dsa(il)) dsa(il, pos0, T, in, out_r);
    else kda(il, T, in, out_r, keep);
}

void Stage::swiglu_ffn(int il, int T, const DT& gate, const DT& up, const DT& down, float limit, const float* in,
                       float* out_r) {
    const int nf = (int) gate.n_out();
    quant(in, xq, (int) hp.n_embd, T);
    pj(up, in, xq, fu, T);
    probe("ffn_up", il, fu, nf);
    pj(gate, in, xq, fg, T);
    probe("ffn_gate", il, fg, nf);
    swiglu_clamp(fg, fu, fg, T * nf, limit, s);
    probe(limit > 0.0f ? "ffn_swiglu_limited" : "ffn_swiglu", il, fg, nf);
    quant(fg, xq2, nf, T);
    pj(down, fg, xq2, out_r, T);
}

/// One routed expert from its VRAM slot for window token t: gate/up on token t's Q8_1 column of `cur` (in `xq`, left
/// there by the shared expert), the clamped SwiGLU, down into `row_out`.
void Stage::gpu_expert(int il, int t, const uint8_t* slot, float* row_out) {
    const kc::NativeFmt& f = L[(size_t) il].fmt;
    const int n = (int) hp.ex(), nf = (int) hp.n_ff_exp;   // (Hy4-flash: the latent's columns, hy4_moe_in's)
    const uint8_t* xt = static_cast<const uint8_t*>(xq) + (size_t) t * q8(n);
    kn::native_mmvq(f.gu_type, slot, xt, eg, n, nf, 1, s);
    kn::native_mmvq(f.gu_type, slot + f.up_off, xt, eu, n, nf, 1, s);
    swiglu_clamp(eg, eu, eg, nf, f.swiglu_limit, s);
    quant(eg, xq3, nf);
    kn::native_mmvq(f.d_type, slot + f.down_off, xq3, row_out, nf, n, 1, s);
}

/// A decode window's MoE (T <= kTmax): the routing to the host, the shared expert queued meanwhile; the experts the
/// VRAM tier holds on the GPU, the rest on the CPU pool (each missed expert's rows read once for all its tokens).
bool Stage::moe_window(int il, int T, std::string& err) {
    const Layer& l = L[(size_t) il];
    const int n = (int) hp.n_embd, E = (int) hp.n_expert, k = (int) hp.n_used;
    // Hy4-flash's latent MoE: the router and the shared expert read cur, the routed experts xl = routed_down(cur) (hy4.cpp);
    // nx is the experts' width (n elsewhere)
    const bool lat = (bool) l.r_down;   // (Hy4-flash's and ERNIE-5.5's latent MoE)
    const int nx = lat ? (int) hp.ex() : n;
    gemv_dt(l.gate_inp, cur, router, n, (int) l.gate_inp.n_out(), T);   // (ERNIE-5.5: its two routers' 2 E logits)
    probe("ffn_moe_logits", il, router, E);
    route(il, T, d_toks, m.route_pen_for(il));
    probe_i32("ffn_moe_topk", il, ids, k);
    probe("ffn_moe_weights_scaled", il, wts, 1, k);
    if (lat) hy4_moe_in(il, T);
    cudaMemcpyAsync(m.h_x, lat ? hxl : cur, (size_t) T * nx * sizeof(float), cudaMemcpyDeviceToHost, s);
    if (m.ahead_from(il)) ahead_predict(il, T);   // DeepSeek-V4.1: the next layer's experts guessed (db_pred, mapped)
    cudaMemcpyAsync(m.h_ids, ids, (size_t) T * k * sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    cudaEventRecord(ev, s);
    if (hp.shexp) swiglu_ffn(il, T, l.sh_gate, l.sh_up, l.sh_down, hp.clamp_sh[(size_t) il], cur, shared);
    else quant(cur, xq, n, T);   // the VRAM hits' Q8_1 columns (the shared expert leaves them otherwise)
    if (lat) quant(hxl, xq, nx, T);   // ... the latent's
    const auto t0 = clk::now();
    if (cudaEventSynchronize(ev) != cudaSuccess) return m.fail(err, "router readback failed");
    st.ms_wait += ms_since(t0);
    m.trace_route(il, T, k, m.h_ids);
    struct Miss {
        int e;
        int nt;
        int t[kc::MAXT], j[kc::MAXT];
    };
    std::vector<Miss> misses;
    m.miss_rows.clear();
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < k; ++j) {
            const int e = m.h_ids[t * k + j];
            if (e < 0 || e >= E) return m.fail(err, "router produced expert " + std::to_string(e));
            const size_t key = (size_t) il * E + e;
            m.heat[key] += 1.0f;
            m.routed[key]++;
            const int32_t slot = m.n_slots ? m.resident[key] : -1;
            if (slot >= 0) {
                gpu_expert(il, t, m.slot_ptr(slot), parts + ((size_t) t * k + j) * nx);
                st.hits++;
                continue;
            }
            st.misses++;
            m.token_misses.push_back((int32_t) key);
            m.miss_rows.push_back(t * k + j);
            auto it = std::find_if(misses.begin(), misses.end(), [&](const Miss& x) { return x.e == e && x.nt < kc::MAXT; });
            if (it == misses.end()) {
                misses.push_back(Miss{e, 0, {}, {}});
                it = misses.end() - 1;
            }
            it->t[it->nt] = t;
            it->j[it->nt] = j;
            it->nt++;
        }
    const auto t1 = clk::now();
    if (!misses.empty()) {
        for (int t = 0; t < T; ++t)
            kc::native_quant_act(l.fmt, m.h_x + (size_t) t * nx, m.act_q.data() + (size_t) t * kc::kNativeActBytes);
        std::vector<kc::ExpertJobMulti> jobs(misses.size());
        std::vector<int64_t> keys(misses.size());
        for (size_t mi = 0; mi < misses.size(); ++mi) {
            kc::ExpertJobMulti& jb = jobs[mi];
            const int e = misses[mi].e;
            keys[mi] = (int64_t) il * E + e;
            const Engine::Impl::Src sr = m.src((int64_t) il * E + e);
            jb.blob = sr.g;
            // one [g | u | d] blob (the RAM arena's): the pool's blob form, whose multi-token kernels decode each row once
            // for all the expert's tokens; an expert in place in the file mapping: the split form (ggml-cpu's rows)
            const bool one_blob = sr.u == sr.g + l.fmt.up_off && sr.d == sr.g + l.fmt.down_off;
            jb.up = one_blob ? nullptr : sr.u;
            jb.down = one_blob ? nullptr : sr.d;
            jb.nt = misses[mi].nt;
            for (int i = 0; i < jb.nt; ++i) {
                jb.nact[i] = m.act_q.data() + (size_t) misses[mi].t[i] * kc::kNativeActBytes;
                jb.out[i] = m.h_parts + ((size_t) misses[mi].t[i] * k + misses[mi].j[i]) * nx;
            }
        }
        std::vector<int32_t> pred;
        if (m.ahead_from(il) && db_pred) pred.assign(db_pred, db_pred + T * (int) hp.k_of(il + m.ahead_d));
        if (!m.run_cpu_jobs(il, jobs, keys, err, pred.empty() ? nullptr : &pred, il + m.ahead_d)) return false;
    } else if (m.ahead_from(il) && db_pred) {
        m.ssd_ahead(il + m.ahead_d, db_pred, T * (int) hp.k_of(il + m.ahead_d));
    }
    st.ms_pool += ms_since(t1);
    for (int r : m.miss_rows)
        cudaMemcpyAsync(parts + (size_t) r * nx, m.h_parts + (size_t) r * nx, (size_t) nx * sizeof(float),
                        cudaMemcpyHostToDevice, s);
    probe("ffn_moe_down", il, parts, nx, k);
    if (lat) {   // the weighted sum in the latent, then routed_up, the shared expert and the MLP's post-norm (hy4.cpp)
        moe_combine_w(parts, wts, nullptr, hml, nx, k, T, s);
        hy4_moe_out(il, T, out);
        return true;
    }
    if (hp.shexp) probe("ffn_shexp", il, shared, n);
    if (m.rec && hp.shexp) {   // the oracle records the MoE sum before the shared expert is added
        float* moe_out = h;   // free in a window until the head
        moe_combine_w(parts, wts, nullptr, moe_out, n, k, T, s);
        probe("ffn_moe_out", il, moe_out, n);
        add_f32(moe_out, shared, out, T * n, s);
    } else {
        moe_combine_w(parts, wts, hp.shexp ? shared : nullptr, out, n, k, T, s);
        if (!hp.shexp) probe("ffn_moe_out", il, out, n);
    }
    if (hp.shexp) probe("ffn_out", il, out, n);
    return true;
}

/// One trunk layer over a decode window (T <= kTmax) in the window scratch.
bool Stage::layer_window(int il, int T, int64_t pos0, bool keep, std::string& err) {
    const int n = (int) hp.n_embd;
    dspark_capture(il, 0, T);   // DSpark: the streams' mean at a target layer's input
    dflash_capture(il, 0, T, false);   // DFlash: the residual there
    if (m.eg_index(il) >= 0) engram(il, 0, T, eg_win_d, kTmax);   // DeepSeek-V4.1: the n-gram rows into the streams
    if (const int j = m.hy4_eg_index(il); j >= 0 && m.heg) hy4_engram(il, j, 0, T, hy_eg_rows, hy_eg_valid, kTmax);   // Hy4-flash's Engram
    mix_in(il, T, R, cur, coef, true);
    attention(il, pos0, T, cur, out, keep);
    mix_out(il, T, out, R, coef, true);
    if (hp.hc > 1) probe(hp.arch == Arch::hy4 ? "hc_attn_out" : "hc_attn_post", il, R, n, hp.hc);   // (llama.cpp's names)
    mix_in(il, T, R, cur, coef, false);
    if (hp.is_moe(il)) {
        if (!moe_window(il, T, err)) return false;
    } else {
        const Layer& l = L[(size_t) il];
        swiglu_ffn(il, T, l.ffn_gate, l.ffn_up, l.ffn_down, hp.clamp_sh[(size_t) il], cur, out);
        if (l.ffn_post_norm) rms_norm(out, l.ffn_post_norm.f(), out, T, n, hp.rms_eps, s);   // Hy4-flash's sandwich norm
        probe("ffn_out", il, out, n);
    }
    mix_out(il, T, out, R, coef, false);
    probe(hp.arch == Arch::dsv4 ? "l_last" : "l_out", il, R, n, hp.hc);   // llama.cpp's name for the layer's output streams
    return true;
}

/// The head over a window: the streams' mean, the norm, the logits (of the last token only, or every token).
void Stage::head(int T, bool last_only) {
    const int n = (int) hp.n_embd;
    if (hp.v41) {   // DeepSeek-V4.1: the streams collapsed with the last FFN's pre-mix (model.py: layer.hc_pre(h, pre_mix))
        mhc_pre_w(R, cpre, h, n, T, s);
        probe("hc_head", -1, h, n);
        rms_norm(h, m.output_norm.f(), h, T, n, hp.rms_eps, s);
    } else if (hp.arch == Arch::dsv4 || hp.arch == Arch::hy4 || hp.arch == Arch::ernie) {   // DSV4_SPEC 4 (Hy4-flash's, ERNIE's the same): the streams' learned
                                                                   // reduce - RMS of the flat streams, 4 mixes, sigmoid weights
        const int hcn = (int) hp.hc * n;
        rms_norm(R, nullptr, flat, T, hcn, hp.rms_eps, s);
        if (m.out_hc_fn.type == 0) {
            gemv_t(m.out_hc_fn.f(), flat, mixes, hcn, (int) hp.hc, T);
        } else {   // (Hy4-flash's SPARK-4Bit sets: a quantized mixer, as in mhc_half)
            quant(flat, xq, hcn, T);
            pj(m.out_hc_fn, flat, xq, mixes, T);
        }
        probe("hc_head_mixes", -1, mixes, hp.hc);
        dsv4_hc_head(R, mixes, m.out_hc_scale.f(), m.out_hc_base.f(), hp.hc_eps, h, n, T, s);
        probe("hc_head", -1, h, n);
        rms_norm(h, m.output_norm.f(), h, T, n, hp.rms_eps, s);
    } else if (hp.hc > 1) {
        mhc_mean_w(R, h, n, T, s);
        probe("hc_head", -1, h, n);
        rms_norm(h, m.output_norm.f(), h, T, n, hp.rms_eps, s);
    } else {
        rms_norm(R, m.output_norm.f(), h, T, n, hp.rms_eps, s);
    }
    probe("result_norm", -1, h, n);
    if (m.head_lo > 0) {
        // moex (DSV41_BACKLOG #140, DSV41_PLAN §76): the head's rows over both GPUs. The window's normed rows go to the first
        // GPU through mapped memory (an event of this stream lets its stream go on); it computes rows [0, head_lo) of the logits
        // and hands them back the same way while this GPU computes the rest; then the two are joined. gemv_bf16_w runs one
        // block a row on either card - each logit the dot product the whole head's launch gave it
        const int Rw = last_only ? 1 : T, V = (int) hp.n_vocab, V0 = (int) m.head_lo, V1 = V - V0;
        const float* hx = last_only ? h + (size_t) (T - 1) * n : h;
        Stage& g0 = *m.stages[0];
        store_mapped(hx, m.hs_x_d, (size_t) Rw * n, s);
        cudaEventRecord(hs_ev, s);
        m.set_dev(g0.dev);
        cudaStreamWaitEvent(g0.s, hs_ev, 0);
        store_mapped(m.hs_x_d, g0.hs_h, (size_t) Rw * n, g0.s);
        gemv_bf16_w(static_cast<const uint16_t*>(m.output_lo.d), g0.hs_h, g0.hs_lo, n, V0, Rw, g0.s);
        store_mapped(g0.hs_lo, m.hs_lo_d, (size_t) Rw * V0, g0.s);
        cudaEventRecord(g0.hs_ev, g0.s);
        m.set_dev(dev);
        float* hi = Rw == 1 ? logits + V0 : hs_hi;   // (one row: straight to its place)
        gemv_bf16_w(static_cast<const uint16_t*>(m.output_hi.d), hx, hi, n, V1, Rw, s);
        cudaStreamWaitEvent(s, g0.hs_ev, 0);
        if (Rw == 1) {
            store_mapped(m.hs_lo_d, logits, (size_t) V0, s);
        } else {
            rows_copy_ld(m.hs_lo_d, V0, logits, V, V0, Rw, s);
            rows_copy_ld(hs_hi, V1, logits + V0, V, V1, Rw, s);
        }
        m.logit_rows = Rw;
        // ERNIE-5.5: MultiMax on the joined logits (each logit's own piece-wise map: as below for the whole head)
        if (hp.arch == Arch::ernie) ernie_multimax_w(logits, hp.n_vocab, m.logit_rows, hp.n_vocab, m.mmax, s);
        probe("result_output", -1, logits, hp.n_vocab);
        return;
    }
    // (moex, the GGUF set: a BF16 head by gemv_dt on the F32 rows)
    if (last_only) {
        if (m.output.type == GGML_TYPE_BF16) {
            gemv_dt(m.output, h + (size_t) (T - 1) * n, logits, n, (int) hp.n_vocab, 1);
        } else {
            quant(h + (size_t) (T - 1) * n, xq, n, 1);
            mm_t(m.output, xq, logits, 1);
        }
        m.logit_rows = 1;
    } else {
        if (m.output.type == GGML_TYPE_BF16) {
            gemv_dt(m.output, h, logits, n, (int) hp.n_vocab, T);
        } else {
            quant(h, xq, n, T);
            mm_t(m.output, xq, logits, T);
        }
        m.logit_rows = T;
    }
    // ERNIE-5.5: MultiMax, the head's piece-wise temperature on the F32 logits
    if (hp.arch == Arch::ernie) ernie_multimax_w(logits, hp.n_vocab, m.logit_rows, hp.n_vocab, m.mmax, s);
    probe("result_output", -1, logits, hp.n_vocab);
}

bool Stage::embd_rows(const int32_t* toks, int T, float* dst, int64_t pos0) const {
    const int n = (int) hp.n_embd;
    for (int t = 0; t < T; ++t) {
        if (toks[t] < 0 || toks[t] >= hp.n_vocab) return false;
        // moex vision: an image position reads the encoder's row (mainline: the batch's embd input, build_inp_embd)
        if (const float* r = pos0 >= 0 ? m.image_row(pos0 + t) : nullptr) {
            std::memcpy(dst + (size_t) t * n, r, (size_t) n * sizeof(float));
            continue;
        }
        ggml_get_type_traits((ggml_type) m.embd_type)->to_float(m.embd + (size_t) toks[t] * m.embd_row, dst + (size_t) t * n, n);
    }
    return true;
}

/// The embedding rows of `toks`, dequantized on the host from the GGUF's table, uploaded into dst (T, n).
bool Stage::upload_embd(const int32_t* toks, int T, float* dst, int64_t pos0) {
    const int n = (int) hp.n_embd;
    cudaStreamSynchronize(s);   // h_embd may still be read by the previous upload
    if (T > h_embd_rows) {
        host_free(h_embd);
        h_embd_rows = 0;
        if (!(h_embd = static_cast<float*>(host_buf((size_t) T * n * sizeof(float))))) return false;
        h_embd_rows = T;
    }
    if (!embd_rows(toks, T, h_embd, pos0)) return false;
    cudaMemcpyAsync(dst, h_embd, (size_t) T * n * sizeof(float), cudaMemcpyHostToDevice, s);
    return true;
}

/// THE NEXTN (MTP) BLOCK over a decode window (T <= kTmax) at positions pos0..: `h_rows` (T, n) is the trunk's post-
/// norm hidden of each position (llama.cpp's t_h_nextn), `toks` the token AFTER each position.  DeepSeek's MTP:
/// z = eh_proj([enorm(emb(tok)) | hnorm(h)]), one pre-norm block (the DSA attention over the block's own caches, its
/// own MoE; no mHC), then shared_head_norm and the shared head.  Row t's logits predict the token two after pos0 + t.
bool Stage::mtp_forward(const float* h_rows, const int32_t* toks, int T, int64_t pos0, std::string& err) {
    const int n = (int) hp.n_embd, il = (int) hp.n_layer;
    const Layer& l = L[(size_t) il];
    if (!upload_embd(toks, T, mtp_e, pos0 + 1)) return m.fail(err, "MTP token out of range");
    try {
        rms_norm(mtp_e, m.enorm.f(), mtp_e, T, n, hp.rms_eps, s);
        rms_norm(h_rows, m.hnorm.f(), mtp_h, T, n, hp.rms_eps, s);
        for (int t = 0; t < T; ++t) {
            cudaMemcpyAsync(mtp_cat + (size_t) t * 2 * n, mtp_e + (size_t) t * n, (size_t) n * sizeof(float),
                            cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(mtp_cat + (size_t) t * 2 * n + n, mtp_h + (size_t) t * n, (size_t) n * sizeof(float),
                            cudaMemcpyDeviceToDevice, s);
        }
        quant(mtp_cat, xq, 2 * n, T);
        pj(m.eh_proj, mtp_cat, xq, mtp_z, T);
        rms_norm(mtp_z, l.attn_norm.f(), cur, T, n, hp.rms_eps, s);
        dsa(il, pos0, T, cur, out);
        add_f32(mtp_z, out, mtp_z, T * n, s);
        rms_norm(mtp_z, l.ffn_norm.f(), cur, T, n, hp.rms_eps, s);
        if (!moe_window(il, T, err)) return false;
        add_f32(mtp_z, out, mtp_z, T * n, s);
        rms_norm(mtp_z, m.shead_norm.f(), mtp_h, T, n, hp.rms_eps, s);
        if (m.output.type == GGML_TYPE_BF16) {   // (moex: a BF16 head)
            gemv_dt(m.output, mtp_h, mtp_logits, n, (int) hp.n_vocab, T);
        } else {
            quant(mtp_h, xq, n, T);
            mm_t(m.output, xq, mtp_logits, T);
        }
        argmax_w(mtp_logits, (int) hp.n_vocab, mtp_amax, T, s);
    } catch (const std::exception& e) {
        return m.fail(err, e.what());
    }
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return m.fail(err, cudaGetErrorString(ce));
    return true;
}

template uint8_t* Stage::dalloc<uint8_t>(size_t, std::string&);
template float* Stage::dalloc<float>(size_t, std::string&);
template int32_t* Stage::dalloc<int32_t>(size_t, std::string&);
template uint16_t* Stage::dalloc<uint16_t>(size_t, std::string&);
template uint32_t* Stage::dalloc<uint32_t>(size_t, std::string&);
template int64_t* Stage::dalloc<int64_t>(size_t, std::string&);

}  // namespace strata::x
