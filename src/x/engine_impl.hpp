// src/x/engine_impl.hpp - strata-x's internals, shared by engine.cpp (the model: load, the VRAM tier,
// the decode windows, the public API), stage.cpp (a GPU's layers: the mixers, mHC, the MoE of a decode window,
// the head, the NextN block) and prompt.cpp (Strata's prompt path: the slot loan, the streamed ring, the MoE as
// MMQ groups, the stages pipelined over the chunks).
//
// THE SHAPE (Strata's, moex/docs/STRATA_QWEN_DESIGN.md): the model is split into contiguous layer ranges, one per
// GPU ("stage").  A Stage owns everything its GPU runs with - its streams, its expert slots, its buffers - and the
// compute for its layers, so the prompt path can run two stages at once on two host threads (GPU 0 reads chunk c + 1
// while GPU 1 reads chunk c), exactly as Strata's Prefill hands each chunk to the next stage's run on its own thread.
// A Stage touches only its own buffers and its own layers' state; the model (weights, hparams, the arena) is shared
// read-only and the tier tables are per (layer, expert), so each stage writes only its own layers' entries.
#pragma once

#include "strata/core/pinned.hpp"
#include "strata/x/dsv4_kernels.hpp"
#include "strata/x/dsv41_vision.hpp"
#include "strata/x/ernie_kernels.hpp"
#include "strata/x/hy4_vision.hpp"
#include "strata/x/engine.hpp"
#include "strata/x/kernels.hpp"
#include "strata/x/hy4_kernels.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/ngram/ple_reader.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "recorder.hpp"
#include "expert_io.hpp"   // moex/tools/dsv41: priority O_DIRECT reads (DeepSeek-V4.1's SSD tier)

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace strata {
class GgufModel;
class GgufFile;
}

namespace strata::x {

namespace kc = strata::kernels::cpu;
namespace kn = strata::kernels;
namespace mmq = strata::prefill::mmq;

/// moex (DSV41_BACKLOG #102), opt-in STRATA_DSV41_HUGE_PIN=1: the SSD cache, the bounce slots and the RAM copy on 2 MiB pages
/// (dsv41.cpp)
bool huge_pin();
/// moex (DSV41_BACKLOG #104), opt-in STRATA_DSV41_KV_PACK=1: DeepSeek-V4.1's compressed rows and index keys packed (dsv41.cpp)
bool kv_pack();
/// pinned host memory for copies to the GPUs: on 2 MiB pages with STRATA_DSV41_HUGE_PIN=1, else cudaHostAlloc'ed; freed by free_pinned
uint8_t* alloc_pinned(uint64_t bytes);
void free_pinned(uint8_t* p);

using clk = std::chrono::steady_clock;
inline double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

/// The prompt path's host buffers (its tables, the CPU read's rows, the stages' hand-off, the embedding rows): memory only
/// copies and the CPU pool touch.  Pinned when the driver can pin it, else plain heap memory - cudaMemcpyAsync stages a
/// pageable buffer itself (slower, the same bytes) - so a pin refused under memory pressure slows that read instead of
/// failing the request, which ends the engine (a reload of over a minute here; upstream's 04dbe561 for its CPU share).
/// host_free frees either kind (nullptr: nothing).
void* host_buf(size_t bytes);
void host_free(void* p);

constexpr int kTmax = 8;
constexpr int kCdCap = kTmax;          ///< the coupled sampler's ring: history slots (penalties off: unread) ...
constexpr int kCdRing = kCdCap + kTmax;   ///< ... plus one slot per draft     ///< MMVQ's column limit: the longest decode / MTP verify window
constexpr int kAttnQ = 16;   ///< DSA queries per attention launch (bounds the split-K partials)
/// moex (DSV41_PLAN §19, DSV41_BACKLOG #84): DeepSeek-V4.1's indexer scores a prompt sub-batch in blocks of this many rows,
/// into a score buffer of that many rows of every position (n_ctx + 1 floats each: 2 MiB a row at 512K context, 1 GiB for
/// a sub-batch of 512 in GPU 0's loan).  The score, candidate and top-k kernels compute each row on its own, so the blocks
/// give the same bits as one launch
constexpr int kDsv41ScoreRows = 64;
constexpr int kGroup = 16;   ///< experts per MMQ group (Strata's MMQ_GROUP)

/// moex (DSV41_PLAN §56 C), opt-in STRATA_DSV41_CAND_ONLY=1: the index sources after layer cand_src score only its candidate
/// blocks' rows (dsv41_cand_list, then dsv4_lid_scores_mma over the list) - each the same bits as the full scan, the rest set
/// to -inf by the candidates' mask as before, so the same top-k
inline bool dsv41_cand_only() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_DSV41_CAND_ONLY");
        return e && e[0] == '1';
    }();
    return on;
}

/// moex (DSV41_PLAN §62), opt-in STRATA_DSV41_BF16_W=1: DeepSeek-V4.1's F32 weights that hold BF16 values - the routers (BF16
/// in the GGUF, an F32 copy beside them otherwise), the compressors' kv and gate, the index keys', the indexer's q_b and its
/// weights' projection (F32 in the GGUF, every lower half zero) - kept as BF16 on the GPUs and read by Stage::gemv_dt: the
/// same values, the same arithmetic, half the bytes; ~440 MiB of VRAM to expert slots
inline bool dsv41_bf16_w() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_DSV41_BF16_W");
        return e && e[0] == '1';
    }();
    return on;
}
/// ... the dense GGUF's F32 tensors it takes (their names' ends; the trunk's layers only)
inline bool dsv41_bf16_w_name(const std::string& n) {
    if (n.rfind("blk.", 0) != 0) return false;
    // (moex: and the router - BF16 in ours, F32 of BF16 values in llama.cpp's set: kept as BF16 either way, one arithmetic)
    for (const char* sfx : {".attn_compressor_kv.weight", ".attn_compressor_gate.weight", ".indexer.attn_k.weight",
                            ".indexer.attn_q_b.weight", ".indexer.proj.weight", ".ffn_gate_inp.weight"}) {
        const size_t k = std::strlen(sfx);
        if (n.size() > k && n.compare(n.size() - k, k, sfx) == 0) return true;
    }
    return false;
}

/// The widest input any dense projection quantizes per token: mHC's flat streams, the MTP's [e | h], the mixers' outputs,
/// the dense FFN's hidden.
inline int64_t max_proj_in(const Hparams& hp) {
    const int64_t n = hp.n_embd, H = hp.n_head;
    return std::max<int64_t>({hp.hc * n, 2 * n, H * hp.v_mla, H * hp.hd_kda, hp.n_ff, H * hp.kv_lora, H * hp.k_mla,
                              H * hp.hd_v});
}

/// A dense tensor in a device arena.
/// moex (DSV41_PLAN §70): an internal type code (no ggml type): an FP8 matrix - its e4m3 values row-major at d (n_out x n_in
/// bytes), then one power-of-two scale (an e8m0 exponent byte) for every 32 values of a row at s (STRATA_DSV41_DENSE_FP8)
constexpr int kDtFP8 = 1000;

struct DT {
    const void* d = nullptr;
    const void* s = nullptr;   ///< moex (§70): kDtFP8's scales
    int type = -1;
    int64_t ne[4] = {1, 1, 1, 1};
    uint64_t bytes = 0;
    int64_t n_in() const { return ne[0]; }
    int64_t n_out() const { return ne[1] * ne[2] * ne[3]; }
    explicit operator bool() const { return d != nullptr; }
    const float* f() const { return static_cast<const float*>(d); }
};

struct Layer {
    DT attn_norm, ffn_norm;
    DT hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    DT wq, wk, wv, conv_q, conv_k, conv_v, f_a, f_b, beta, a, dt_b, g_a, g_b, o_norm, wo;            // KDA
    DT q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b;                                               // DSA
    DT idx_k_norm, idx_k_norm_b, idx_proj, idx_k, idx_q_b, idx_gate, idx_ape;                       // indexer
    DT ffn_gate, ffn_up, ffn_down;                                                                   // dense FFN
    DT gate_inp, probs_b, sh_gate, sh_up, sh_down;                                                   // MoE
    DT wqkv, sinks;                                                                                  // MiMo GQA
    DT eh_proj, enorm, hnorm, out_norm;                                                              // MiMo's MTP blocks
    DT wkv, wo_a, wo_b, c_kv, c_gate, c_ape, c_norm, l_kv, l_gate, l_ape, l_norm, probs_b_vl, tid2eid;   // DeepSeek-V4
    DT eg_wkv, eg_qw, eg_kw;   // DeepSeek-V4.1's engram (idx_k / idx_k_norm hold its index keys' projection and norm)
    // Hy4-flash (hy4.cpp): the output gate and the sandwich post-norms; the latent MoE's down / up projections; the
    // indexer's pooled value projection and the pooled keys' norm (idx_gate / idx_ape: the pool's gate projection and its
    // position bias; idx_k / idx_k_norm: the per-token key and its norm); the Engram's fused key / value projection, its bias,
    // the folded gate norms, the rows' per-head norms, the value's gamma and the per-stream offsets
    DT attn_gate, attn_post_norm, ffn_post_norm, r_down, r_up, kp_value, kp_norm;
    // ERNIE-5.5-Flash (ernie.cpp): VHA's I + V U^T (64 x 64 F32, built at load), the latent's RMS norm before routed_up,
    // the per-expert factor after the weights' normalization
    DT vha, latent_norm, exp_scale;
    DT hy_eg_kv, hy_eg_kvb, hy_eg_gate, hy_eg_lnorm, hy_eg_pg, hy_eg_hcs;
    kc::NativeFmt fmt;           ///< the routed experts' native blob (MoE layers)
    uint64_t arena_off = 0;      ///< this layer's first blob in the host arena
    /// Expert e's gate / up / down rows for the CPU and the copies: base + e * stride (the arena: strides = the blob;
    /// in place in the GGUF mapping: one expert's slice of each tensor).
    const uint8_t *eg = nullptr, *eu = nullptr, *ed = nullptr;
    size_t sg = 0, su = 0, sd = 0;
    bool blob_contiguous = false;   ///< eg/eu/ed are one blob per expert (the arena): one copy per expert
    /// > 0: the experts are records of this many bytes in a pack file (DeepSeek-V4.1): eg / eu / ed are expert 0's parts and
    /// expert e's at + e * rec_stride, while sg / su / sd stay the parts' lengths
    uint64_t rec_stride = 0;
    /// moex (the GGUF set, --experts-gguf; dsv41.cpp load_experts_gguf): the records are virtual - expert e's gate / up
    /// / down slices at f_off[r] + e * sg / su / sd of the expert set's file f_shard[r] (Impl::xg), read by rec_io into a
    /// slot's places 0 / up_off / down_off (fmt), rec_stride bytes a slot; eg / eu / ed: the slices in the files' mappings
    bool rec_gguf = false;
    size_t f_shard[3] = {0, 0, 0};  ///< the gate / up / down tensors' shard and their data's file offset (the load's reads)
    uint64_t f_off[3] = {0, 0, 0};
    // decode state (on the layer's GPU)
    float* conv = nullptr;          ///< 3 x d_inner x (d_conv - 1)
    float* S = nullptr;             ///< (S, H, S)
    uint16_t* kv_cache = nullptr;   ///< n_ctx x kv_lora, FP16
    float* ik_ring = nullptr;       ///< ring_rows x idx_dim: the open pool's keys
    float* ig_ring = nullptr;       ///< ring_rows x idx_dim: the open pool's gates
    float* pooled = nullptr;        ///< (n_ctx / kpool + 1) x idx_dim
    float* snaps = nullptr;         ///< MTP rollback: the KDA state after every window token (snap_T, S, H, S)
    float* conv_prev = nullptr;     ///< MTP rollback: the conv state before the last window
    float* conv_in = nullptr;       ///< MTP rollback: the last window's conv inputs (T, 3 * d_inner)
    // MiMo: the layer's K and V caches, FP16, row r = (n_kv, hd_k) / (n_kv, hd_v) of position p at r = p % kv_rows - the full
    // context on a full-attention layer, a ring of Impl::swa_rows on a sliding-window one (the window, plus the longest batch
    // written before it is read: rows it overwrites are out of every remaining query's window)
    uint16_t* k_cache = nullptr;
    uint16_t* v_cache = nullptr;
    int64_t kv_rows = 0;
    // DeepSeek-V4's state: the raw K = V ring (F16, row p % raw_rows: the 128-position window plus the longest batch written
    // before it is read), the compressed rows (F16, row b = the block of positions [b r, b r + r): CSA's or HCA's), the lightning
    // indexer's rows (F16, CSA layers), and the compressors' F32 state rings (kv; score + ape; row p % the ring)
    uint16_t *raw = nullptr, *ccache = nullptr, *lcache = nullptr;
    int64_t raw_rows = 0, c_rows = 0;
    float *c_ring_kv = nullptr, *c_ring_sc = nullptr, *l_ring_kv = nullptr, *l_ring_sc = nullptr;
};

/// A stage's working buffers for windows of up to `rows` tokens whose mixers run in sub-batches of up to `sub`.
/// Laid out by Stage::carve in one region: the decode scratch (rows = sub = kTmax, its own cudaMalloc) or the prompt
/// path's (rows = the chunk, sub = the mixer sub-batch, in the slots the stage lends).
struct Scratch {
    int rows = 0, sub = 0;
    // per window token
    float *R = nullptr, *cur = nullptr, *out = nullptr, *coef = nullptr, *router = nullptr, *wts = nullptr;
    float *shared = nullptr, *h = nullptr;
    int32_t* ids = nullptr;
    // per sub-batch token (mHC's read, the mixers, the dense FFN, the shared expert)
    float *flat = nullptr, *mixes = nullptr;
    float* hc_part = nullptr;   ///< DeepSeek-V4.1: the mixers' split-row partial sums of a decode window (gemv_f32_split)
    void *xq = nullptr, *xq2 = nullptr;
    float *pq = nullptr, *pk = nullptr, *pv = nullptr, *cq = nullptr, *ck = nullptr, *cv = nullptr, *fa = nullptr;
    float *g = nullptr, *beta = nullptr, *o = nullptr, *ga = nullptr, *g2 = nullptr, *y = nullptr;
    float *qr = nullptr, *q = nullptr, *q_hm = nullptr, *kv = nullptr, *qabs = nullptr, *qabs_hm = nullptr;
    float *iq = nullptr, *ikw = nullptr, *igw = nullptr, *w32 = nullptr, *score = nullptr, *ctx = nullptr;
    float *ctx_hm = nullptr, *o2 = nullptr, *o2_hm = nullptr, *partial = nullptr;
    int32_t* cells = nullptr;
    float *fg = nullptr, *fu = nullptr, *sg = nullptr, *su = nullptr;
    float *qkv = nullptr, *att = nullptr, *gpart = nullptr;   // MiMo: the fused QKV rows, the attention output, split-K partials
    // DeepSeek-V4: q (64 x 512 per token), the K = V row, the compressors' and the indexer's projections, the indexer's q,
    // weights, scores and top-k, the attention output, the grouped output (token- and group-major), the split-K partials, and
    // the hash layers' ids (per row of the window / chunk)
    float *dq = nullptr, *dkv = nullptr, *dakv = nullptr, *dgs = nullptr, *dlakv = nullptr, *dlgs = nullptr, *diq = nullptr;
    float *dw = nullptr, *dscore = nullptr, *datt = nullptr, *doa = nullptr, *doa_hm = nullptr, *dapart = nullptr;
    int32_t *dsel = nullptr, *dhash = nullptr;
    // DeepSeek-V4.1 (dsv41.cpp): the carried pre-mix (per window row, the coef layout: pre in the first 4 of 24), the
    // compressor's latents (post-norm, pre-RoPE), the index keys, layer cand_src's candidate blocks (per row, one byte per
    // block)
    float *cpre = nullptr, *dlat = nullptr, *dik = nullptr;
    float* cpre_b = nullptr;   ///< moex (#105): the carried pre's second buffer (a decode window's halves alternate)
    uint8_t* dcand = nullptr;
    int32_t *dclist = nullptr, *dccount = nullptr;   ///< moex (§56 C, CAND_ONLY): a score block's rows' candidate lists, counts
    // Hy4-flash (hy4.cpp), per sub-batch token: the attention's query rows [q_abs | q_pe] (H x 576), its gate (H x 256), the
    // kv_a row (576), the indexer's pooled-value projection (128), the latent MoE's input and combined output (per row of the
    // window / chunk: n_ex), the Engram's normed rows (cols x dim) and its fused projection (5 x n_embd); per decode window the
    // attention's split-K partials
    float *hqf = nullptr, *hgate = nullptr, *hkv = nullptr, *hiv = nullptr, *hxl = nullptr, *hml = nullptr;
    float *hegr = nullptr, *hegkv = nullptr, *hpart = nullptr;
    int32_t* ecnt = nullptr;   ///< ERNIE-5.5 (ernie.cpp): a DSA layer's cell count per sub-batch token
};

/// The prompt path's MoE buffers for one chunk (Strata's prefill: GU / H / Hq / Dm / Xq, the group buffers, the
/// grouping tables, the identity row map).
struct MoeBufs {
    int64_t rows = 0;                 ///< T * k
    uint8_t *xq = nullptr, *hq = nullptr;
    float *gu = nullptr, *hb = nullptr, *dm = nullptr;
    int64_t ld_gu = 0;                ///< gu's row stride in floats: 2 nf, or n with STRATA_DSV41_DM_ALIAS (moex, §33)
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    int32_t *ident = nullptr, *grp = nullptr;   ///< grp: [src (rows) | slot (rows) | bounds ...] on the device
};

struct Stream;   ///< a chunk's streamed experts (prompt.cpp)

class Stage {
public:
    Stage(Engine::Impl& m, int di, int dev);
    ~Stage();
    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;

    Engine::Impl& m;
    const Hparams& hp;
    const EngineOptions& opt;
    std::vector<Layer>& L;
    int di = 0, dev = 0;
    int64_t lb = 0, le = 0;          ///< the trunk layers [lb, le)
    bool last = false;               ///< also the head and the NextN block
    cudaStream_t s = nullptr;        ///< compute
    cudaStream_t cs = nullptr;       ///< copies (the prompt path's streamed experts)
    /// moex (DSV41_PLAN §39), opt-in STRATA_DSV41_GRAPH_FORK=1: a captured decode graph's KV row and compressor on a second
    /// stream beside the query's projections (forked after the input's quantization, joined before the indexer)
    cudaStream_t s2 = nullptr;
    cudaEvent_t fork_ev = nullptr, join_ev = nullptr;
    /// moex (DSV41_BACKLOG #105), opt-in STRATA_DSV41_MIX_SIDE=1: a decode window's mixer coefficients - the streams' norm, the
    /// split gemv, the Sinkhorn chain - on a third stream beside the half's attention or MoE (V4.1 reads a half with the PREVIOUS
    /// half's carried pre, so this half's coefficients are first read at its post-mix), joined before mix_out's mhc_post_w
    cudaStream_t s3 = nullptr;
    cudaEvent_t mix_fork = nullptr, mix_join = nullptr;
    bool mix_wait = false;   ///< a half's coefficients are on s3: mix_out waits for mix_join first
    cudaEvent_t ev = nullptr, hand = nullptr, swap_ev = nullptr;
    EngineStats st;                  ///< this stage's counters (Engine::stats sums them)
    std::atomic<int64_t> us_gate_own{0};   ///< moex (§41): its issuer's share of RecordIo::us_gate (the routing waits, no read in flight)

    // ---- buffers
    Scratch win;                     ///< decode windows (kTmax rows), its own allocation
    Scratch pr;                      ///< the prompt path's (a chunk), carved in the loan
    MoeBufs mb;
    // decode-only buffers (kTmax rows): the routed experts' rows, a GPU expert's scratch, the head
    float *parts = nullptr, *eg = nullptr, *eu = nullptr, *logits = nullptr;
    void* xq3 = nullptr;
    int32_t* amax = nullptr;
    float *mtp_e = nullptr, *mtp_h = nullptr, *mtp_cat = nullptr, *mtp_z = nullptr, *mtp_logits = nullptr;
    int32_t* mtp_amax = nullptr;
    // THE DRAFTER AS A GRAPH (mtp.cpp, Strata's MtpDrafter): the NextN block over Tm rows with its experts resident on
    // this GPU - one captured graph per row count, its inputs in device buffers the host fills before each launch
    float *mtp_hin = nullptr;        ///< (kTmax, n): the hidden rows the block reads (the trunk's post-norm, or its own residual)
    float *mtp_zlast = nullptr;      ///< (n): the block's residual at its last row - the chain's next hidden
    float *mtp_hnlast = nullptr;     ///< (n): ... and its shared_head_norm'd form (the head's input; the alternative chain hidden)
    float* mtp_prob = nullptr;       ///< the draft's probability under softmax (the last row)
    int64_t* mtp_pos = nullptr;      ///< device: the first row's position
    int64_t* h_mtp_pos = nullptr;    ///< pinned source of mtp_pos
    uint8_t* d_zero_mask = nullptr;  ///< (kTmax * k) zeros: moe_combine_pairs with no CPU rows
    int32_t* h_mtp_tok = nullptr;    ///< pinned: the last row's draft token
    float* h_mtp_prob = nullptr;     ///< pinned: ... and its probability
    float* mtp_chain = nullptr;      ///< MiMo (kTmax, n): row i = MTP head i's residual at its last row - head i + 1's hidden
    std::map<int, cudaGraphExec_t> gx_mtp;                  ///< key: head * 64 + rows * 2 + draft
    std::map<int, int> gx_mtp_uses;
    bool mtp_resident = false;       ///< every NextN expert is in this GPU's pinned slots: the block runs without the host
    // the grouped hits' decode buffers are in alloc_decode
    int32_t* dbnd = nullptr;         ///< a dense MMQ product's bounds {0, rows}
    int32_t* hbnd = nullptr;         ///< the heads MMQ's bounds {0, T, 2T, ...} (H + 1)
    int hbnd_T = -1;                 ///< the T hbnd holds
    int hbnd_H = -1;                 ///< ... and the heads (H + 1 bounds)
    void* blas = nullptr;            ///< cuBLAS on this GPU (the prompt path's F32 GEMMs)
    float* bf16_f32 = nullptr;       ///< moex (§62): the largest BF16 weight of this stage's layers widened for a GEMM
    size_t bf16_f32_n = 0;           ///< ... its floats (moex: a larger BF16 matrix is widened and multiplied in row blocks)
    /// moex (DSV41_PLAN §62), STRATA_DSV41_PROMPT_GUESS_LOG=1 (a measurement, nothing else changes): in a gated chunk's MoE the
    /// next layer's routing guessed from this layer's FFN rows (its FFN norm, its router and its top-k: the decode read-ahead's
    /// guess, ahead_compute) and scored when that layer routes against the reads it then needs from the drive
    /// ... and STRATA_DSV41_PROMPT_GUESS=N: the issuer reads the N most guessed of the layer's drive records into SSD cache
    /// slots while its routing comes (Stream::guess_l)
    float* gs_x = nullptr;
    int32_t* gs_ids = nullptr;
    float* gs_wts = nullptr;
    int32_t* gs_hb[2] = {nullptr, nullptr};   ///< pinned: the guessed ids, by the layer's parity
    cudaEvent_t gs_ev[2] = {nullptr, nullptr};
    int gs_for[2] = {-1, -1};        ///< the layer each holds the guess for (-1: none), under gs_mu
    const void* gs_str[2] = {nullptr, nullptr};   ///< ... and the chunk's stream it was made in
    int gs_rows[2] = {0, 0};
    std::mutex gs_mu;
    int64_t gs_cap = 0;              ///< the rows they hold
    int64_t gs_n[10] = {};           ///< layers scored, their drive reads, then (taken, useful) of the top 8 / 16 / 24 / 32
    float* h_embd = nullptr;         ///< pinned: embedding rows on their way up (per stage: stages run concurrently)
    int h_embd_rows = 0;
    std::vector<void*> allocs;
    template <typename T>
    T* dalloc(size_t n, std::string& err);
    void dfree(void* p);   ///< moex: frees a dalloc'd buffer before the stage ends (a grown buffer's old one)

    // ---- the expert slots (this GPU's share of the VRAM tier)
    uint8_t* slots = nullptr;
    int64_t n_slots = 0, slot_base = 0;
    /// moex (DSV41_PLAN §24, STRATA_DSV41_SLOT_SIZES=1): each slot the size of the expert the plan put in it (its layer's blob:
    /// a swap stays within a layer, a lent slot gets its own expert back) - slot k at slot_off[k], n_slots + 1 offsets; empty
    /// with uniform slots (slot k at k * slot_bytes)
    std::vector<size_t> slot_off;
    size_t lend_bytes(int64_t k) const;   ///< the top k slots' bytes
    int64_t lent = 0;                ///< top slots lent to the prompt path (their experts are not resident meanwhile)
    /// how many of the top slots the prompt path may borrow: 90% of them, 128 stay, and the drafter's pinned ones (the last
    /// stage's first) are never lent
    int64_t lendable() const;
    /// moex (DSV41_PLAN §28): ... for a layer-major read - STRATA_DSV41_LM_KEEP=N: all but N (opt-in; unset: lendable()'s)
    int64_t lendable_lm() const;
    std::vector<int32_t> lent_keys;  ///< the experts the lent slots held (refilled afterwards)

    // ---- the decode graphs (decode.cpp): Strata's Verifier structure - one captured graph per stage per window variant
    // (T tokens, keep), the host's half of each MoE layer served from mapped memory (the doorbell), the VRAM tier's hits
    // chosen on the DEVICE from a residency table and computed by grouped launches
    volatile uint32_t *db_seq = nullptr, *db_flag = nullptr, *db_abort = nullptr;   // host views of the mapped words
    volatile uint32_t *dd_seq = nullptr, *dd_flag = nullptr, *dd_abort = nullptr;   // device views
    int32_t *db_ids = nullptr, *dd_ids = nullptr;       // the router's ids (published by the GPU)
    float *db_x = nullptr, *dd_x = nullptr;             // the layer's activations (published by the GPU)
    float *db_rows = nullptr, *dd_rows = nullptr;       // the CPU's expert rows (written by the host)
    uint8_t *db_mask = nullptr, *dd_mask = nullptr;     // which (token, expert) pairs the rows hold
    void* db_block = nullptr;
    uint32_t *d_pub_cnt = nullptr, *d_wait_cnt = nullptr;   // device counters the doorbell kernels increment
    uint32_t host_seq = 0, host_flag = 0;                   // publishes the host has consumed / flags it has released
    volatile uint32_t *db_egf = nullptr, *dd_egf = nullptr;   // moex (§42, STRATA_DSV41_EG_LATE): the engram rows' word (mapped),
    uint32_t* d_eg_cnt = nullptr;                             // the wait's device count and the words the host has written
    uint32_t host_egf = 0;
    int64_t* d_pos = nullptr;                               // the window's first position (device)
    int64_t* h_pos = nullptr;                               // ... and its pinned source (the eager path)
    // THE STEP'S INPUTS IN MAPPED HOST MEMORY: the host writes them, the graph's first node (window_load) reads them - no copy
    // engine on the window's critical path, where a queued swap DMA would hold the window's start back by milliseconds
    int64_t *in_pos_h = nullptr, *in_pos_d = nullptr;       // the window's first position
    float *in_rows_h = nullptr, *in_rows_d = nullptr;       // (kTmax, n): the first stage's embedding rows
    int32_t *in_res_h = nullptr, *in_res_d = nullptr;       // [n_load * E]: the residency table (stage-local slots), host master copy
    float *out_rows_h = nullptr, *out_rows_d = nullptr;     // (kTmax, 4, n): this stage's residuals for the next GPU
    int64_t res_lo = 0, res_n = 0;                          // this stage's trunk slice of the table, in keys
    int64_t *mtp_in_pos_h = nullptr, *mtp_in_pos_d = nullptr;   // the drafter's: its first position,
    float *mtp_in_rows_h = nullptr, *mtp_in_rows_d = nullptr;   // its embedding rows (kTmax, n),
    int32_t *mtp_out_tok_h = nullptr, *mtp_out_tok_d = nullptr; // its draft and the draft's probability
    float *mtp_out_prob_h = nullptr, *mtp_out_prob_d = nullptr;
    // COUPLED DRAFT SAMPLING (EngineOptions::coupled_draft; mtp.cpp draft_pick): the request's chain on the device, the
    // sampler's ring (only written: no penalties), its scratch, and the MTP cell of the draft being made (mapped: the
    // captured graphs read each round's own), whose verify row draws Philox(seed, cell + 1)
    bool cd_on = false;                                         ///< this request's drafts are coupled samples
    strata::kernels::SamplerParams* cd_params = nullptr;
    int32_t* cd_ring = nullptr;
    void* cd_scratch = nullptr;
    int32_t *cd_cell_h = nullptr, *cd_cell_d = nullptr;
    /// Embedding rows of `toks` dequantized on the host into `dst` (host memory: pinned, mapped, or the staging buffer).
    /// `pos0` >= 0: toks[t] sits at position pos0 + t, and a position inside one of the request's images (Impl::img) takes
    /// that image's encoder row instead (moex vision; mainline's embedding batch).
    bool embd_rows(const int32_t* toks, int T, float* dst, int64_t pos0 = -1) const;
    int32_t* d_res = nullptr;                               // [n_load * n_expert] stage-local slot of (layer, expert), or -1
    int32_t* d_hslot = nullptr;                             // [kTmax * k] this layer's pairs' slots
    float *hit_g = nullptr, *hit_u = nullptr, *hit_parts = nullptr;
    void* hit_xq = nullptr;
    const int64_t* gpos = nullptr;                          // set while a graph is captured: dsa() reads positions from d_pos
    std::map<int, cudaGraphExec_t> gx;                      // window variants, key T * 2 + keep
    std::map<int, int> gx_uses;
    std::vector<char> hit_ok;                               // per layer: the graph computes its VRAM hits (grouped launches)
    bool graph_ok = true;
    bool alloc_decode(std::string& err);
    bool graph_ready(int T, bool keep, std::string& err);   // captured on the second use of a variant
    bool capture_window(int T, bool keep, std::string& err);
    void layer_graph_body(int il, int T, bool keep);
    bool service_moe(int il, int T, std::string& err);      // the host's half of a MoE layer: doorbell, misses, release
    void release_gpu_waits();                               // the abort word: no spin kernel outlives a failed request
    /// the buffers a checkpoint copies: the recurrent state and the indexer rings of this stage's layers (and the NextN's)
    void state_regions(std::vector<std::pair<void*, size_t>>& out) const;
    void state_regions_layer(int64_t il, std::vector<std::pair<void*, size_t>>& out) const;   ///< layer il's part of them
    size_t state_layer_offset(int64_t il) const;   ///< where layer il's part starts, in state_regions' order
    /// moex (DSV41_PLAN §57): the POSITIONAL rows a conversation of `len` positions holds on this stage's layers - DeepSeek-V4's
    /// compressed rows and indexer rows, V4.1's KV sources' (packed or F16), rows [0, len / ratio] of each (dsv41.cpp)
    void positional_regions(int64_t len, std::vector<std::pair<void*, size_t>>& out) const;

    // ---- the prompt path
    std::unique_ptr<mmq::Context> mq;
    uint8_t* ring = nullptr;
    int ring_n = 0;
    size_t ring_stride = 0;
    std::vector<cudaEvent_t> copied, used;
    std::vector<int> used_of;
    std::vector<char> live;
    int32_t* h_ids = nullptr;        ///< pinned: a chunk's routed ids
    int32_t* h_grp = nullptr;        ///< pinned: the grouping tables (mirrors mb.grp)
    float* h_hand[2] = {nullptr, nullptr};   ///< pinned: the residual rows handed to the next stage (double buffer)
    uint8_t* own_prompt = nullptr;   ///< the prompt buffers' own allocation when nothing is lent
    /// moex (DSV41_BACKLOG #76), STRATA_DSV41_LAYER_MAJOR=1: the first stage reads a WINDOW of chunks layer by layer - every
    /// chunk through layer il before layer il + 1, each layer's streamed experts read once and held in the ring for all of
    /// them.  Each chunk keeps its own rows that cross layers (replay_shift's list: the streams, the carried pre-mix, the
    /// top-k lists, the candidate masks; chunk 0's are the scratch's) and its own engram rows; every (layer, chunk) runs
    /// exactly as the chunk loop runs it, so the output is the same bit for bit
    struct LmRows {
        float* R = nullptr;
        float* cpre = nullptr;
        int32_t* dsel = nullptr;
        uint8_t* dcand = nullptr;
    };
    struct LmChunk {
        const int32_t* toks;
        int T;
        int64_t pos0;
    };
    int lm_k = 0;                    ///< the window's chunks the next lend makes room for (Impl::prompt sets it; 0: off)
    std::vector<LmRows> lmw;         ///< [chunk of the window], carved by lend
    /// moex (DSV41_PLAN §63), opt-in STRATA_DSV41_LM_HOST=1: a window of more chunks than the loan holds rows for - lm_win of
    /// them, their rows in pinned host memory (lmh: per chunk [R | the carried pre-mix | the top-k lists | the candidate masks],
    /// each part T rows long, so the first two are the hand-off's layout), the loan's lm_k sets their staging ring: (layer,
    /// chunk) p runs in set p mod lm_k, its rows copied up on lm_cs[0] before and back on lm_cs[1] after
    int lm_win = 0;
    std::vector<uint8_t*> lmh;
    size_t lmh_bytes = 0;            ///< each lmh buffer's bytes
    /// moex (§63 step 2): lmh lent from the SSD cache's blocks for this call (Impl::cache_lend; given back at its end, never
    /// freed), and each block's chunk's engram rows after its rows - egc (host) / egc_d (device), egc_cap rows a layer
    bool lmh_lent = false;
    std::vector<uint8_t*> egc, egc_d;
    int egc_cap = 0;
    /// ... with two host windows in flight (STRATA_DSV41_LM_HOST=2): the running window's first buffer (lmh, egc) and the
    /// window's chunks - a window's set is (window mod 2) x lmh_wk; 0, 0: one set
    int lmh_off = 0, lmh_wk = 0;
    cudaStream_t lm_cs[2] = {nullptr, nullptr};
    std::vector<cudaEvent_t> lm_up_ev, lm_done_ev, lm_dn_ev;   ///< [set]: its rows up, its layer run, its rows back
    /// ... the windows' engram rows: two buffers (mapped; eg_pr_grow's layout), each a window's chunks' rows one after the
    /// other - the next window's fetched while this one runs (lm_next names it); eg_rbase: the running chunk's first row
    uint8_t *eg_lm_h[2] = {nullptr, nullptr}, *eg_lm_d[2] = {nullptr, nullptr};
    int eg_lm_cap = 0;
    std::vector<std::promise<std::string>> eg_lm_p[2];      ///< each chunk's rows in buffer b: "" or the error
    std::vector<std::shared_future<std::string>> eg_lm_cf[2];
    /// moex (#147, §76): ... of its engram layers after the first (eg_lm_p / eg_lm_cf: the first's) - the fetch reads every
    /// chunk's first-layer rows, then the others', so layer 1 does not wait for layer 14's rows
    std::vector<std::promise<std::string>> eg_lm_p2[2];
    std::vector<std::shared_future<std::string>> eg_lm_cf2[2];
    std::future<void> eg_lm_job[2];                         ///< the fetch filling buffer b, chunk by chunk
    int64_t eg_lm_pos[2] = {-1, -1};
    int eg_lm_T[2] = {0, 0};
    std::vector<LmChunk> lm_next;
    void eg_lm_fetch(int b, const std::vector<LmChunk>& w);   ///< the rows of window w into buffer b, on a worker
    void eg_lm_drop();                                        ///< every fetch waited out, the buffers named nothing
    std::vector<float*> h_lm;        ///< pinned: each chunk's rows handed on to the next stage
    int h_lm_rows = 0;
    bool eg_lm_grow(int k, int rows, std::string& err);
    size_t lm_extra_bytes(int rows, int k, int kw = 0) const;   ///< ... a window of k chunks (kw: its chunks when more, §63)
    bool prompt_window(const std::vector<LmChunk>& w, std::string& err);
    void lm_rows(int c);             ///< chunk c's rows active (R, cpre, dsel, dcand)

    // ---- compute: the active scratch's pointers (use())
    void use(const Scratch& x);
    float *R = nullptr, *cur = nullptr, *out = nullptr, *coef = nullptr, *router = nullptr, *wts = nullptr;
    float *shared = nullptr, *h = nullptr;
    int32_t* ids = nullptr;
    float *flat = nullptr, *mixes = nullptr, *hc_part = nullptr;
    void *xq = nullptr, *xq2 = nullptr;
    float *pq = nullptr, *pk = nullptr, *pv = nullptr, *cq = nullptr, *ck = nullptr, *cv = nullptr, *fa = nullptr;
    float *g = nullptr, *beta = nullptr, *o = nullptr, *ga = nullptr, *g2 = nullptr, *y = nullptr;
    float *qr = nullptr, *q = nullptr, *q_hm = nullptr, *kv = nullptr, *qabs = nullptr, *qabs_hm = nullptr;
    float *iq = nullptr, *ikw = nullptr, *igw = nullptr, *w32 = nullptr, *score = nullptr, *ctx = nullptr;
    float *ctx_hm = nullptr, *o2 = nullptr, *o2_hm = nullptr, *partial = nullptr;
    int32_t* cells = nullptr;
    float *fg = nullptr, *fu = nullptr, *sg = nullptr, *su = nullptr;
    float *qkv = nullptr, *att = nullptr, *gpart = nullptr;
    float *dq = nullptr, *dkv = nullptr, *dakv = nullptr, *dgs = nullptr, *dlakv = nullptr, *dlgs = nullptr, *diq = nullptr;
    float *dw = nullptr, *dscore = nullptr, *datt = nullptr, *doa = nullptr, *doa_hm = nullptr, *dapart = nullptr;
    int32_t *dsel = nullptr, *dhash = nullptr;
    float *cpre = nullptr, *dlat = nullptr, *dik = nullptr;   ///< DeepSeek-V4.1 (Scratch)
    float* cpre_b = nullptr;   ///< moex (#105): the carried pre's second buffer
    uint8_t* dcand = nullptr;
    int32_t *dclist = nullptr, *dccount = nullptr;   ///< moex (§56 C): CAND_ONLY's lists (Scratch)
    float *hqf = nullptr, *hgate = nullptr, *hkv = nullptr, *hiv = nullptr, *hxl = nullptr, *hml = nullptr;   ///< Hy4 (Scratch)
    float *hegr = nullptr, *hegkv = nullptr, *hpart = nullptr;
    int32_t* ecnt = nullptr;         ///< ERNIE-5.5 (Scratch)
    int SB = kTmax;                  ///< the active scratch's sub-batch
    // Hy4-flash (hy4.cpp): its attention, the latent MoE's ends, and the Engram's write into the streams of rows [r0, r0
    // + T) before layer il (its rows and validity on this stage's device: hy_eg_rows (cap rows of layer slot j at j * cap * cols
    // * dim), hy_eg_valid (cap x cols))
    void hy4_attn(int il, int64_t pos0, int T, const float* in, float* out_r);
    void hy4_engram(int il, int j, int r0, int T, const uint8_t* rows_base, const float* valid_base, int cap);
    void hy4_moe_in(int il, int T);                   ///< the experts' latent input xl = routed_down(cur), its Q8_1 columns
    void hy4_moe_out(int il, int T, float* out_r);     ///< out = routed_up(hml) + shared, then the MLP's post-norm
    // ERNIE-5.5-Flash (ernie.cpp): the attention of an HCA layer (DeepSeek-V4's ratio 128) or a DSA layer (Hy4's MLA over
    // the lightning indexer's positions and the window), each with the output gate and VHA
    void ernie_attn(int il, int64_t pos0, int T, const float* in, float* out_r);
    void ernie_hca(int il, int64_t pos0, int T, const float* in, float* out_r);
    void ernie_dsa(int il, int64_t pos0, int T, const float* in, float* out_r);
    uint8_t* hy_eg_rows = nullptr;      ///< a decode window's (kTmax a slot, allocated once: the graphs read it there)
    float* hy_eg_valid = nullptr;
    int hy_eg_cap = 0;
    bool hy_eg_grow(int rows, std::string& err);
    uint8_t* hy_egp_rows = nullptr;     ///< a prompt chunk's (this stage's slots; grown to the chunk) and its pinned staging
    float* hy_egp_valid = nullptr;
    uint8_t* h_egp_rows = nullptr;
    float* h_egp_valid = nullptr;
    int hy_egp_cap = 0;
    int hy_egp_j0 = 0, hy_egp_nj = 0;   ///< this stage's Engram slots j0 .. j0 + nj - 1: the chunk buffers' slot j - j0
    bool hy_egp_grow(int rows, std::string& err);
    /// moex vision (opt.images): the request's M-RoPE table on this stage's GPU (hy4_kernels.hpp Hy4MPos; Impl::hy4_mpos_set
    /// writes it as a request starts), passed to every rotation; null: the token indices
    int64_t* hy_mpos = nullptr;
    /// DeepSeek-V4's hash layers route by token: the rows' token ids on the device for the MoE that runs next - the decode
    /// graph's mapped in_tok_d, or tok_buf (the eager windows, the prompt path's chunk), set by the caller of the layers
    const int32_t* d_toks = nullptr;
    int32_t* tok_buf = nullptr;
    int tok_cap = 0;
    /// Uploads T token ids into tok_buf (grown as needed) on this stage's stream and points d_toks at it.
    bool upload_toks(const int32_t* toks, int T, std::string& err);
    // DeepSeek-V4's DSpark drafter (dspark.cpp; the last stage): the trunk's features of the rows a window or a prompt chunk
    // read - the streams' mean at the input of each of the drafter's target layers (n_layer: the trunk's output) - then
    // fused into its blocks' K = V rings (inject), and a block of drafts from an anchor token (draft)
    bool attn_noncausal = false;     ///< dsv4(): the batch's rows see each other (DSpark's noise block, an image span)
    bool attn_block = false;         ///< dsv41(): DeepSeek-V4.1 DSpark's block - every row sees the 128 positions before it and the block
    bool image_batch = false;        ///< route(): the batch is a DeepSeek-V4 image span (exp_probs_b_vl, no tid2eid)
    /// moex (DSV41_BACKLOG #113), STRATA_DSV41_PIC_MERGE=1: a DeepSeek-V4.1 call reading text and pictures together - route()
    /// and the engram take each row's kind from its position (img_pos0: the position of prompt_layer's row 0)
    bool img_mix = false;
    int64_t img_pos0 = -1;
    float* dsp_feat = nullptr;       ///< (dsp_rows, n_target * n): the captured features
    float* dsp_tmp = nullptr;        ///< (dsp_rows, n): one target layer's streams' mean
    float* dsp_g = nullptr;          ///< (dsp_rows, n): the fused features (fc, enc norm)
    float* dsp_logits = nullptr;     ///< (kTmax, n_vocab): the drafter's logits
    float* dsp_bias = nullptr;       ///< (n_vocab): the Markov head's bias for one block position
    float* dsp_cin = nullptr;        ///< V4.1: (kTmax, n_embd + rank) the confidence head's inputs
    float* dsp_cout = nullptr;       ///< V4.1: (kTmax) its outputs
    float dsp_conf[kTmax] = {0};     ///< V4.1: the last draft block's confidences (the head's logits, one per draft)
    int dsp_rows = 0;
    int32_t* dsp_dprev = nullptr;    ///< the Markov head on the device: [0] the anchor, [i + 1] draft i (kTmax + 1)
    float* dsp_dprob = nullptr;      ///< ... draft i's probability (kTmax)
    int32_t dsp_tok[kTmax] = {0};    ///< the last draft block's tokens and their probabilities (catch-up -> chain steps)
    float dsp_prob[kTmax] = {0};
    int dsp_n = 0;
    /// The features of rows [r0, r0 + rows) of the active scratch's streams at target layer il's input (il == n_layer: the
    /// trunk's output) into dsp_feat rows [0, rows); nothing when il is not a target layer.
    void dspark_capture(int il, int r0, int rows);
    /// The captured features' first `rows` rows (positions pos0..) into every drafter block's ring.
    bool dspark_inject(int rows, int64_t pos0, std::string& err);
    /// A block of n drafts after `anchor` at position pos (dsp_tok / dsp_prob).
    bool dspark_draft(int32_t anchor, int64_t pos, int n, std::string& err);
    bool dspark_moe(int il, int T, std::string& err);
    int32_t *in_tok_h = nullptr, *in_tok_d = nullptr;       ///< the decode window's token ids in mapped memory (the first stage)
    // MiMo's DFlash drafter (dflash.cpp): the trunk's residual at the input of each target layer (n_layer: its output) for
    // the rows a window or a prompt chunk read.  A target layer on an earlier stage writes its window rows into that stage's
    // mapped block (df_map) and its chunk tails into a pinned double buffer (h_dfl, by chunk parity); the last stage gathers
    // them beside its own (df_feat: windows; df_feat_pr: chunk tails, carved in the loan) and injects them into the drafter's
    // K / V rings (fc, the encoder norm, every block's k / v projections, k norm, RoPE).
    float* df_feat = nullptr;        ///< last: (kTmax, nt * n) a window's features, target k at column k
    float* df_g = nullptr;           ///< last: (kTmax, n) the fused features
    float* df_k = nullptr;           ///< last: (kTmax, nkv * hd) then V right after, kTmax rows of each
    void* df_xq = nullptr;           ///< last: Q8_1 of kTmax rows of the widest projection input (fc's nt * n)
    float *df_feat_pr = nullptr, *df_g_pr = nullptr, *df_k_pr = nullptr;   ///< last: the same for a chunk's tail (the loan)
    float *df_map_h = nullptr, *df_map_d = nullptr;   ///< earlier stage: (kTmax, df_ncol * n) mapped, its targets' window rows
    float* h_dfl[2] = {nullptr, nullptr};             ///< earlier stage: pinned (dfl_cap, df_ncol * n) chunk tails
    int df_ncol = 0;                 ///< target layers this stage captures for the last one (an earlier stage)
    int dfl_par = 0;                 ///< earlier stage: the h_dfl this chunk writes (the hand-off buffer's parity)
    const float* dfl_in = nullptr;   ///< last: the earlier stage's h_dfl of this chunk (set before its prompt_chunk)
    // the noise block (last stage): its residual, norm, Q, K / V, attention, output, FFN, Q8_1 rows, split-K partials, and the
    // block's tokens / probabilities / coupled-sampling cells in mapped memory
    float *df_x = nullptr, *df_nn = nullptr, *df_q = nullptr, *df_kvn = nullptr, *df_att = nullptr, *df_o = nullptr;
    float *df_fg = nullptr, *df_fu = nullptr, *df_part = nullptr;
    void* df_bxq = nullptr;
    float *df_in_h = nullptr, *df_in_d = nullptr;     ///< mapped (kTmax, n): the block's embeddings (anchor, then the masks)
    int32_t *df_tok_h = nullptr, *df_tok_d = nullptr;
    float *df_prob_h = nullptr, *df_prob_d = nullptr;
    int32_t *df_cell_h = nullptr, *df_cell_d = nullptr;   ///< mapped: draft j's MTP cell (its verify row's position - 1)
    int32_t df_tok[kTmax] = {0};     ///< the last block's drafts and their probabilities (catch-up -> chain steps)
    float df_prob[kTmax] = {0};
    int df_n = 0;
    /// The rows [r0, r0 + rows) of the active scratch's residual at target layer il's input into the drafter's buffers (a
    /// window: rows from 0; `tail`: a prompt chunk's last rows); nothing when il is not a target layer.
    void dflash_capture(int il, int r0, int rows, bool tail);
    /// The captured features' first `rows` rows (positions pos0..) into the drafter's rings (`tail`: the chunk's buffers).
    bool dflash_inject(int rows, int64_t pos0, bool tail, std::string& err);
    /// The catch-up after a window: its kept rows [pos0, pos0 + rows) into the rings, then the noise block [anchor, mask x
    /// (block - 1)] at pos0 + rows: df_tok / df_prob for block positions 1 .. block - 1 (the argmax and its probability, or coupled
    /// samples against the next window's rows).  One captured graph per (rows, coupled) after a variant's first use.
    bool dflash_step(int rows, int32_t anchor, int64_t pos0, std::string& err);
    void dflash_inject_ops(int rows, int64_t pos0, const int64_t* pos_dev);   ///< a window's inject, kernels only
    void dflash_block_ops(int64_t pos, const int64_t* pos_dev);               ///< the noise block + head + picks, kernels only
    /// y (rows, w.n_out) = x (rows, w.n_in) W^T: MMQ past a window (the prompt path's loan), else quantize + dflash_mm
    void dflash_proj(const DT& w, const float* x, int rows, float* y);
    /// y (T <= 8 columns) = W x for the drafter: Q8_0 on int8 tensor cores (q8_mma_w), else MMVQ
    void dflash_mm(const DT& w, const void* xq_, float* y, int T);
    int64_t *df_pos_h = nullptr, *df_pos_d = nullptr;   ///< mapped: the inject's first position, the block's
    std::map<int, cudaGraphExec_t> df_gx;               ///< key: rows * 2 + coupled
    std::map<int, int> df_gx_uses;
    bool df_graphs_off = false;
    double df_last_ms = 0;                              ///< the last draft step's wall time (the window policy's cost)
    // Hy4-flash's DFly2 drafter (hy4_dfly.cpp), on the last stage: the capture rings (kTmax rows by position: the trunk's
    // cells at the drafter's target layers - each target layer's stage holds its own -, the head's normed rows), the query
    // block's buffers, and the drafts
    struct Dfy {
        int32_t* cells = nullptr;     ///< (target slots, kTmax, max_cells): the trunk's cells at its target layers
        float* h = nullptr;           ///< (kTmax, n): the head's normed rows (vLLM's target hidden: norm(hc_head(x)))
        float *x = nullptr, *xn = nullptr, *qr = nullptr, *q = nullptr, *kv = nullptr, *qf = nullptr, *qhm = nullptr, *qahm = nullptr;
        float *ctx = nullptr, *ctxhm = nullptr, *o2hm = nullptr, *o2 = nullptr, *gate = nullptr, *o = nullptr, *fg = nullptr, *fu = nullptr;
        float *part = nullptr, *cat = nullptr, *cg = nullptr, *cu = nullptr, *cd = nullptr, *hc = nullptr, *logits = nullptr;
        float* emb0 = nullptr;        ///< (n): the bonus token's own embedding (the correction's first previous token)
        void *xq = nullptr, *xq2 = nullptr;
        int32_t *dcells = nullptr, *counts = nullptr;
        float *emb_h = nullptr, *emb_d = nullptr;     ///< mapped (kTmax + 1, n): the block's embeddings, then a draft's
        int32_t *tok_h = nullptr, *tok_d = nullptr;   ///< mapped: a step's draft (argmax or coupled sample) and its probability
        float *prob_h = nullptr, *prob_d = nullptr;
        int32_t *cell_h = nullptr, *cell_d = nullptr; ///< mapped: draft j's cell (its verify row's position - 1, coupled drafts)
        int stride = 0;               ///< a draft row's cells
    } dfy;
    int32_t dfy_tok[kTmax] = {0};     ///< the last block's drafts and their probabilities (catch-up -> chain steps)
    float dfy_prob[kTmax] = {0};
    int dfy_n = 0;
    double dfy_last_ms = 0;
    /// trunk layer il's cells (T rows from pos0; a graph: gpos) into its target slot's ring - nothing when il is not one
    void dfly_capture_cells(int il, const int32_t* cells, int64_t pos0, int T);
    /// the head's normed rows (T from pos0) into the ring (after head())
    void dfly_capture_h(int64_t pos0, int T);
    /// The catch-up after a window: the query block [bonus, mask x (block - 1)] at pos0 + rows, from the kept row before it -
    /// dfy_tok / dfy_prob for block positions 1 .. block - 1 (each the argmax of the head over the corrected row)
    bool dfly_step(int rows, int32_t bonus, int64_t pos0, std::string& err);
    /// The last stage's chunk-tail buffers (df_feat_pr, df_g_pr, df_k_pr) for a chunk of `rows`, carved in its loan
    static size_t dflash_tail_bytes(const Engine::Impl& m, int rows);

    bool alloc(std::string& err);    ///< streams, events, the decode scratch and buffers
    static size_t scratch_bytes(const Engine::Impl& m, int rows, int sub, bool cand_rows = true);
    void carve(Scratch& x, uint8_t* base, int rows, int sub) const;
    /// moex (DSV41_PLAN §59 P3), STRATA_DSV41_CAND_STAGE=1: DeepSeek-V4.1's candidate masks (16,385 bytes a row at 1M) only on a
    /// stage that has the candidates' source layer or a later one - GPU 0's layers 0-19 never write or read them
    bool cand_rows() const {
        static const bool on = [] {
            const char* e = std::getenv("STRATA_DSV41_CAND_STAGE");
            return e && e[0] == '1';
        }();
        return !(on && hp.v41 && hp.cand_src >= 0 && le <= hp.cand_src);
    }

    // probes (synchronous; only with a recorder)
    void probe(const char* name, int il, float* d, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1);
    void probe_i32(const char* name, int il, int32_t* d, int64_t n);
    /// moex (#36b): an index owner's selection inputs for one decode row - its source's index keys (every visible row), its
    /// scores and its candidate bits - for a recorder that asks for idx_kc / idx_score / idx_cand by name (dsv41.cpp)
    void dump_selection(int il, int ratio, const uint16_t* lcache, int srows, const uint8_t* cand, int cbytes, int64_t pos0);

    // projections
    size_t q8(int n) const { return kn::native_q8_1_bytes(n); }
    void mm_t(const DT& w, const void* xq_, float* y_, int T);
    void quant(const float* x, void* xq_, int n, int ncols = 1);
    bool mmq_ok(const DT& w) const;
    void mmq_dense(const DT& w, const float* x, float* y, int T);
    void pj(const DT& w, const float* x, const void* xqn, float* y, int T);
    /// y (T, n_out) = x (T, n_in) W^T for an F32 weight: gemv_f32_w on a window, cuBLAS SGEMM on a prompt sub-batch
    void gemv_t(const float* W, const float* x, float* y, int n_in, int n_out, int T);
    /// moex (DSV41_PLAN §62): gemv_t over an F32 weight or a BF16 one (STRATA_DSV41_BF16_W): on a window gemv_bf16_w, on a
    /// prompt sub-batch the F32 values widened into bf16_f32 and gemv_t's GEMM on them - the same bits either way
    void gemv_dt(const DT& w, const float* x, float* y, int n_in, int n_out, int T);
    /// per-head products (H heads of [n_out x n_in], head-major rows x_hm (H, T, n_in) -> y_hm (H, T, n_out)): one MMQ
    /// launch with the heads as experts on a prompt sub-batch; heads_q8_0 / per-head MMVQ on a window
    void heads_mm(const DT& w, const float* x_hm, float* y_hm, int H, int n_in, int n_out, int T);

    // a layer's parts, T tokens (T <= SB for the mixers): rows passed explicitly
    void mhc_half(int il, int T, const float* Rr, float* cur_r, float* coef_r, bool attn);
    void kda(int il, int T, const float* in, float* out_r, bool keep);
    void dsa(int il, int64_t pos0, int T, const float* in, float* out_r);
    void gqa(int il, int64_t pos0, int T, const float* in, float* out_r);   ///< MiMo
    void dsv4(int il, int64_t pos0, int T, const float* in, float* out_r);  ///< DeepSeek-V4 (dsv4.cpp)
    void dsv41(int il, int64_t pos0, int T, const float* in, float* out_r); ///< DeepSeek-V4.1 (dsv41.cpp)
    /// DeepSeek-V4.1: the carried pre-mix at the start of a window - one-hot (stream 0) on every row
    void v41_pre_reset(int T);
    /// DeepSeek-V4.1's engram rows (the stage of the engram layers: the first): a window's (mapped, kTmax rows - the graphs
    /// read it) and a prompt chunk's (mapped, grown to the chunk; never captured)
    uint8_t *eg_win_h = nullptr, *eg_win_d = nullptr, *eg_pr_h = nullptr, *eg_pr_d = nullptr;
    int eg_pr_cap = 0;
    /// moex (DSV41_BACKLOG #51), STRATA_DSV41_EG_AHEAD=1: the NEXT chunk's engram rows, fetched into a second buffer from the
    /// moment this chunk's own have arrived (prompt_layer); the next chunk takes buffer and fetch over (prompt_chunk).  The
    /// chunk loop (Impl::prompt) names the next chunk: its tokens, first position and the plan's length for it
    uint8_t *eg_nx_h = nullptr, *eg_nx_d = nullptr;
    int eg_nx_cap = 0;
    std::future<std::string> eg_nx;
    int64_t eg_nx_pos = -1;                  ///< the first position whose rows the second buffer holds (-1: none)
    int eg_nx_T = 0;
    const int32_t* eg_next_toks = nullptr;   ///< the chunk loop's hint (eg_next_T 0: no next chunk in this call)
    int64_t eg_next_pos = 0;
    int eg_next_T = 0;
    bool eg_nx_grow(int rows, std::string& err);
    int32_t* imx_order = nullptr;    ///< the imatrix collection: a layer's routed experts in id order (device, n_expert)
    /// DeepSeek-V4.1's read-ahead: the next layer's router on this layer's FFN rows (its norm, logits, top-k) and the predicted
    /// ids handed to the host beside the routing (mapped: the graphs publish them before the doorbell)
    float *apred_x = nullptr, *apred_logits = nullptr, *apred_wts = nullptr;
    int32_t* apred_ids = nullptr;
    int32_t *db_pred = nullptr, *dd_pred = nullptr;
    void ahead_predict(int il, int T);   ///< enqueue layer il + 1's prediction from `cur` and its copy into db_pred
    /// moex (DSV41_PLAN §32): the decode graphs compute the guess after the routing's doorbell (off its path) and publish it
    /// with a word of its own, as the doorbell does (the ids, then the count); the host waits for it only where it reads ahead
    void ahead_compute(int il, int T);   ///< ahead_predict without the copy (the ids stay in apred_ids)
    volatile uint32_t *db_pred_seq = nullptr, *dd_pred_seq = nullptr;
    uint32_t* d_pred_cnt = nullptr;
    uint32_t pred_expect = 0;            ///< the guesses' publishes the host expects so far
    /// DeepSeek-V4.1's SSD tier in the prompt stream: pinned bounce slots the issuer reads records into before each copy to
    /// the ring, and the events that free them
    /// the prompt stream's bounce slots (prompt.cpp, Prefetch: one fewer records read ahead): kRecBounce, or
    /// STRATA_DSV41_BOUNCE=N (every stage) or N0,N1,... (stage by stage) - the drive reads on while a layer's attention runs
    /// and the ring is full
    static constexpr int kRecBounce = 32;
    std::vector<uint8_t*> rbounce;
    std::vector<cudaEvent_t> rbounce_ev;
    int rbounce_next = 0;
    /// moex (DSV41_BACKLOG #47): 1 + the key of a record whose read failed twice in this chunk (issue_entry, the issuer's
    /// thread); prompt_layer ends the chunk with an error instead of computing that expert as zeros.  0: none
    std::atomic<int64_t> read_failed{0};
    bool eg_pr_grow(int rows, std::string& err);
    /// The engram's write into rows [r0, r0 + T) of the active streams before layer il, from a buffer of `cap` rows (its rows
    /// from eg_rbase + r0 on: moex #76, a layer-major window's buffer holds all its chunks; 0 otherwise)
    /// moex (§76, the head over two GPUs): the first stage's copy of a window's normed rows and its rows of logits; the last
    /// stage's rows of logits before they are joined (windows of more than one row); each stage's event for the other
    float *hs_h = nullptr, *hs_lo = nullptr, *hs_hi = nullptr;
    cudaEvent_t hs_ev = nullptr;
    void engram(int il, int r0, int T, const uint8_t* buf_d, int cap, const float* kv_pre = nullptr);
    /// ... its half that reads the rows alone (the dequant into `att`, the wkv projection into `kv`): engram() runs it into
    /// datt / dq, or takes `kv_pre` where a decode graph ran it earlier (moex, DSV41_BACKLOG #140: eg_pre_kv)
    void engram_rows(int il, int r0, int T, const uint8_t* buf_d, int cap, float* att, float* kv);
    /// moex (DSV41_BACKLOG #140, DSV41_PLAN §75): engram layer j's wkv output computed in the decode graph of the layer before
    /// it, between that layer's VRAM experts and its wait for the CPU's rows ([j]: its buffer of kTmax rows, or null: in place)
    std::vector<float*> eg_pre_kv;
    float* eg_pre_att = nullptr;   ///< ... its dequantized rows (kTmax x cols x dim)
    int eg_rbase = 0;
    /// The routed experts' ids and weights of layer il over T rows (the router's logits in `router`): GLM's and MiMo's sigmoid
    /// top-k with the selection bias; DeepSeek-V4's sqrt-softplus top-k with its bias - in its first n_hash layers the ids
    /// by token from tok_d (the rows' token ids on this GPU)
    void route(int il, int T, const int32_t* tok_d, const float* pen = nullptr);   ///< pen: the opt-in route bias (decode)
    bool route_publish(int il, int T, const int32_t* tok_d, const float* pen);   ///< moex (#112): route() + the doorbell's publish
    /// the family's layer-half front (mHC's read + norm, or the pre-norm) and back (mHC's write, or the residual add), and
    /// the attention half's mixer
    void mix_in(int il, int T, const float* Rr, float* cur_r, float* coef_r, bool attn);
    void mix_out(int il, int T, const float* x, float* Rr, const float* coef_r, bool attn);
    void attention(int il, int64_t pos0, int T, const float* in, float* out_r, bool keep);
    void swiglu_ffn(int il, int T, const DT& gate, const DT& up, const DT& down, float limit, const float* in,
                    float* out_r);
    bool moe_window(int il, int T, std::string& err);   ///< a decode window's MoE (T <= kTmax): VRAM hits + CPU pool
    void gpu_expert(int il, int t, const uint8_t* slot, float* row_out);
    /// one trunk layer over a window (T <= kTmax: the decode path; the probes)
    bool layer_window(int il, int T, int64_t pos0, bool keep, std::string& err);
    void head(int T, bool last_only);
    bool upload_embd(const int32_t* toks, int T, float* dst, int64_t pos0 = -1);   ///< embd_rows' rule, uploaded
    bool mtp_forward(const float* h_rows, const int32_t* toks, int T, int64_t pos0, std::string& err);
    /// The block's ops over Tm rows (inputs in mtp_e / mtp_hin; the first position pos0, or gpos in a capture); the head on
    /// the last row only with `last_head` (outputs mtp_amax[0], mtp_prob[0]).  `res`: the MoE runs from the pinned slots on
    /// the device (capturable); else through moe_window (the host).
    bool mtp_body(int head, int Tm, int64_t pos0, bool last_head, bool res, std::string& err);
    void mtp_moe_resident(int il, int T);
    /// One drafter step: embeddings of `toks`, hidden rows `h_src` (device), first position pos0; the last row's draft in
    /// tok / prob.  Through the captured graph for Tm rows (captured on its second use) when the experts are resident.
    bool mtp_run(int head, int Tm, const int32_t* toks, const float* h_src, int64_t pos0, bool draft, int32_t& tok, float& prob,
                 std::string& err);
    bool mtp_capture(int head, int Tm, bool draft, std::string& err);
    void draft_pick();                                                ///< the draft of the last row: argmax or coupled sample
    int mtp_key(int head, int Tm, bool draft) const { return head * 64 + Tm * 2 + (draft ? 1 : 0) + (draft && cd_on ? 4096 : 0); }
    /// The trunk's hidden rows the drafter reads after a window: GLM's post-norm hidden, MiMo's pre-norm residual, DeepSeek-V4's
    /// four residual streams (h_nextn)
    const float* mtp_hsrc() const { return hp.arch == Arch::glm5 ? win.h : win.R; }
    /// The width of a drafter hidden row: the trunk's streams (DeepSeek-V4's four) times n_embd
    int64_t mtp_hw() const { return (hp.arch == Arch::dsv4 ? hp.hc : 1) * hp.n_embd; }
    /// DeepSeek-V4's NextN block over Tm rows (dsv4_mtp.cpp; deepseek4.cpp graph_mtp): inputs in mtp_e / mtp_hin
    bool dsv4_mtp_body(int Tm, int64_t pos0, bool last_head, std::string& err);
    /// ... over the rows of a prompt chunk the first decode window's drafter reads (its 128-position window)
    bool prompt_mtp_dsv4(int T, int n_next, int64_t pos0, const int32_t* next_toks, std::string& err);
    /// Where a drafter block's routed experts are on the host: expert e's [gate | up | down] blob at g + e * gs (u, d null: a
    /// pinned arena of its own), or its gate / up / down rows at g + e * gs, u + e * us, d + e * ds (in place in its GGUF's mapping)
    struct ExpertSrc {
        const uint8_t *g = nullptr, *u = nullptr, *d = nullptr;
        uint64_t gs = 0, us = 0, ds = 0;
    };
    /// A drafter block's MoE over T rows, its routed experts computed by the CPU pool from `src`, the shared expert on the GPU
    /// meanwhile (DSpark's blocks, DeepSeek-V4's NextN head)
    bool arena_moe(int il, int T, const ExpertSrc& src, std::string& err);
    /// embd_rows from DeepSeek-V4's MTP head's own embedding table (Impl::dmtp)
    bool mtp_embd_rows(const int32_t* toks, int T, float* dst, int64_t pos0) const;
    /// The RMS norms' eps of the block being run (mhc_half, dsv4): the trunk's, or - while DeepSeek-V4's MTP head runs on this
    /// stage - its own file's (llama.cpp builds the draft model's graph with the draft's f_norm_rms_eps: Vision-Exp's trunk
    /// says 1e-20, the original head 1e-6).  Per stage: another stage may be reading a prompt meanwhile.
    float eps_ovr = 0.0f;
    float rms_eps() const { return eps_ovr > 0.0f ? eps_ovr : hp.rms_eps; }
    /// The catch-up over T kept rows (toks: the token after each, h_src: the trunk's rows): every MTP block's caches; the last
    /// row's draft from the first.  GLM: the NextN block (its residual is the chain's hidden); MiMo: its three blocks.
    bool mtp_catchup(int T, const int32_t* toks, const float* h_src, int64_t pos0, int32_t& tok, float& prob, std::string& err);
    /// Draft j >= 1 of a window at pos (d: the drafts so far): GLM chains its one block on its own hidden; MiMo runs block j
    /// over rows pos .. pos + j - 1 (mainline's chained heads: block j's caches hold the catch-up's rows, not the drafts').
    bool mtp_chain_step(int j, const int32_t* d, int64_t pos, int32_t& tok, float& prob, std::string& err);
    int mtp_heads() const;

    // ---- the prompt path (prompt.cpp)
    static size_t prompt_bytes(const Engine::Impl& m, int rows, int sub, int ring, bool last = false, int di = -1,
                               size_t stride = 0, bool cand_rows = true);   ///< last: + DFlash's tails; di: the stage (its
                                                                            ///< ring's stride); stride: that stride; cand_rows()
    bool lend(int rows, int sub, int ring_slots, std::string& err);   ///< carve pr / mb / the ring in lent slots
    /// The fewest top slots whose bytes hold `need`, at most `cap` of them, lent (their experts not resident meanwhile; with
    /// STRATA_DSV41_PROMPT_KEEP the SSD experts' records copied into SSD cache slots first, for the refill) -> the first lent
    /// slot's address; null when `cap` slots hold too little (nothing lent), or with `failed` when a copy out failed (the slots
    /// lent: give_back_slots).  lend()'s, and moex (DSV41_PLAN §54) the image encoder's workspace between requests.
    uint8_t* lend_top(size_t need, int64_t cap, bool& failed, std::string& err);
    void give_back_slots();                                           ///< queue the lent slots' refill (one sync)
    void refill_issue();                                              ///< give_back_slots' copies, queued
    void refill_finish();                                             ///< give_back_slots' wait, check and release
    std::vector<std::pair<int32_t, int32_t>> refilled;                ///< (slot, key) queued by refill_issue
    std::vector<int> refill_held;   ///< moex (DSV41_PLAN §45, STRATA_DSV41_REFILL_ASYNC): the SSD cache slots its copies read
    /// One chunk of the prompt through this stage's layers.  `toks` (the first stage) or `hand_in` (pinned rows from
    /// the previous stage); the last stage leaves the chunk's final hidden rows (post-norm) in pr.h for the NextN.
    bool prompt_chunk(const int32_t* toks, const float* hand_in, int T, int64_t pos0, const int32_t* next_toks,
                      int n_next, std::string& err);
    /// Bounded replay: rows [0, T - Tr) of the chunk shift out - the R rows and the carried pre-mix of rows [T - Tr, T) to
    /// the front - before replay_layer runs over the Tr rows the reads after the chunk need
    void replay_shift(int T, int Tr);
    bool replay_cone() const;   ///< moex (#108): STRATA_DSV41_REPLAY_CONE
    int cone_rows(int64_t il, int64_t pos, int T, int64_t cut) const;   ///< ... how many of [pos, pos + T) layer il needs for `cut`
    bool moe_prompt(int il, int T, Stream& stream, std::string& err);
    /// The chunk's MoE without the stream (EngineOptions::cpu_read): the resident experts as MMQ groups, the rest on the CPU
    /// pool concurrently, their rows after the GPU's in the row map
    bool moe_prompt_cpu(int il, int T, std::string& err);
    bool cpu_moe = false;            ///< this prompt run streams nothing (Engine::Impl::prompt sets it)
    float* h_cx = nullptr;           ///< pinned (rows, n): the chunk's FFN inputs for the CPU pool
    float* h_cdm = nullptr;          ///< pinned (rows * k, n): the CPU's expert rows
    int64_t h_cx_rows = 0, h_cdm_rows = 0;
    std::vector<uint8_t> cact;       ///< the chunk's activations in the experts' vec_dot type
    /// moex (Strata 0.1.41's CPU share): the chunk's MoE with the CPU pool and this stage's stream sharing the routed
    /// non-resident experts, split per layer where both sides end together (EngineOptions::prefill_share)
    bool moe_prompt_share(int il, int T, std::string& err);
    /// a prompt-path MoE layer by this run's mode: moe_prompt_cpu, moe_prompt_share or moe_prompt (the stream)
    bool moe_chunk(int il, int T, Stream& stream, std::string& err);
    bool share = false;              ///< this prompt run shares the experts with the CPU (Engine::Impl::prompt sets it)
    /// what this GPU's streamed experts cost: bytes per ms over the copy stream (an EMA over the shared layers; its link)
    double share_link = 0.0;
    /// a shared layer's events: [0] before its first copy, [1] after its last (this GPU's link), [2] its FFN inputs on the host
    cudaEvent_t sh_ev[3] = {nullptr, nullptr, nullptr};
    double sh_pend_bytes = 0.0;      ///< the last shared layer's streamed bytes, its events not read yet (0: none)
    /// STRATA_X_SYNC_CHECK=1: sync this stage's stream and fail with `what` and the layer on a CUDA error (debugging)
    bool sync_check(const char* what, int il, std::string& err);
    /// ... and with STRATA_X_SYNC_CHECK=2 also scan `n` floats at d for NaN / Inf
    bool nan_check(const char* what, int il, const float* d, size_t n, std::string& err);
    bool prompt_layer(int il, int T, int64_t pos0, Stream& stream, std::string& err, int half = 0);   ///< half: §59 P7
    bool prompt_nextn(int T, int n_next, int64_t pos0, const int32_t* next_toks, Stream& stream, std::string& err);
    bool prompt_mtp_mimo(int T, int n_next, int64_t pos0, const int32_t* next_toks, std::string& err);
    size_t own_prompt_bytes = 0;
    size_t h_grp_cap = 0;
    int h_hand_rows = 0;
    /// moex (DSV41_BACKLOG #78, Impl::stop_layers): this stage's running state (state_regions) at the start of its last
    /// chunks, device copies in the loan (lend) - stage si keeps n_stages - si of them (it runs at most that many chunks ahead
    /// of the last stage's unfinished one); run_pos is where its running state is (-1: inside a chunk a stop ended)
    struct RunSnap {
        int64_t pos = -1;
        uint8_t* d = nullptr;
    };
    std::vector<RunSnap> run_snaps;
    int run_snap_next = 0;
    int64_t run_pos = -1;
    /// moex (DSV41_PLAN §30), opt-in STRATA_DSV41_TAIL_BATCH=1: the bounded replay's tails of a call - the rows of the last
    /// replay_tail before each read start (a checkpoint cut, the call's end) through the layers from replay_layer on - run
    /// together once the call's chunks are done (prompt_tails), each of those layers streamed once for all of them instead
    /// of once a tail.  A tail's rows (replay_shift's list) wait in the loan after its chunk's layer replay_layer - 1, and the
    /// chunk ending at a cut leaves a copy of the running state there that the tails' layers complete; the cuts' checkpoints
    /// are taken after.  A piece: a tail's rows in one chunk (a tail across a chunk end is two, run in order)
    struct TailRows {
        float* R = nullptr;
        float* cpre = nullptr;
        int32_t* dsel = nullptr;
        uint8_t* dcand = nullptr;
        RunSnap* snap = nullptr;     ///< the chunk ends at its cut: the running state there, completed layer by layer
        int T = 0;
        int64_t pos = 0, cut = 0;
    };
    bool tail_batch = false;         ///< this call defers its tails (Impl::prompt sets it before the lend)
    int tail_cap = 0;                ///< ... the pieces the loan holds rows for
    int tail_rows_cap = 0;           ///< ... their rows, all together (a tail across a chunk end: two pieces, one tail's rows)
    int tail_snap_cap = 0;           ///< ... and the cuts' state copies (one for each piece that ends at its cut)
    size_t tail_rows_used = 0;       ///< (the rows' bytes the pieces so far took)
    int call_sub = 0;                ///< the call's mixers' sub-batch (the tails run with the chunks' own)
    uint8_t* loan_base = nullptr;    ///< the loan's start: the tails' scratch is carved there again
    size_t tail_chunk_bytes = 0;     ///< the loan's chunk part; the pieces' rows and state copies follow it
    std::vector<TailRows> tails;     ///< the pieces deferred so far, in position order
    std::vector<RunSnap> tail_snaps; ///< their cuts' state copies (in the loan)
    static size_t tail_rows_bytes(const Engine::Impl& m, int rows);
    bool tail_keep(int T, int64_t pos, int64_t cut, bool at_cut, std::string& err);
    bool prompt_tails(std::string& err, bool past_stop = false);   ///< past_stop: a stopped call's, run to the end
    /// moex (DSV41_PLAN §33), opt-in STRATA_DSV41_RING_KEEP=1: with the tails deferred, every chunk of a call streams only
    /// this stage's one layer before replay_layer (GPU 1's layer 20); its records stay in the ring (Stream::hold) and each
    /// chunk reads only the routed ones the ring lacks - the layer streamed about once a call, not once a chunk.  [entry]:
    /// the ring's slot of that stream entry holds its record; the layer (lend() forgets them)
    std::vector<uint8_t> ring_have;
    int ring_kept_l = -1;
    /// moex (DSV41_PLAN §37): this layer-major window is the call's last and the tails are read ahead meanwhile - each layer's
    /// kept cache slots are released once its chunks' MoE has run (an event on the compute stream after the layer)
    bool lm_release = false;
    std::vector<cudaEvent_t> rel_ev;
    size_t keep_stride = 0;   ///< the call's ring keeps those layers' records: a whole layer at their blob (lend); 0: no keep
    bool chunk_stopped = false;   ///< the last prompt_chunk ended at a stop before its layers were done
    /// moex (#81): the prompt heartbeat's view of this stage (Impl::prompt, diagnostics on): the layer and chunk it runs, its
    /// stream's records issued of all, and the reads its issuer has ahead
    std::atomic<int> hb_il{-1}, hb_c{-1}, hb_ahead{0};
    std::atomic<size_t> hb_issued{0}, hb_total{0};
    size_t run_state_bytes() const;    ///< state_regions' bytes
    int run_snap_count() const;        ///< n_stages - this stage's index
    void run_snap_take(int64_t pos);   ///< the running state now, as at position pos (queued on s)
    bool run_snap_put(int64_t pos);    ///< the copy taken at pos back (queued on s); false: none
    /// moex (#76): a layer-major window's copies are put together layer by layer - the next copy of the ring labelled pos
    /// (filled by run_snap_layer as each layer finishes the chunk ending at pos), and layer il's part of one
    RunSnap* run_snap_new(int64_t pos);
    void run_snap_layer(int64_t il, RunSnap& r);
    const RunSnap* run_snap_at(int64_t pos) const;
};

/// A checkpoint of the RUNNING state (Strata 1.5): the recurrent KDA state and conv history of every KDA layer and the open
/// pool's indexer rings of every DSA layer, per stage, in pinned host memory.  The positional state (the latent caches and
/// the pooled keys) is not copied: rewinding to a position just means writing from there again.
/// moex vision: a picture inside a token list - where its rows start, how many, and the hash of their content.
struct ImageKey {
    int64_t start = 0, n = 0;
    uint64_t hash = 0;
    bool operator==(const ImageKey& o) const { return start == o.start && n == o.n && hash == o.hash; }
};

struct Checkpoint {
    std::vector<int32_t> ids;           ///< the tokens whose state this is
    std::vector<ImageKey> imgs;         ///< moex: the images inside ids (a prefix is reused only when they match too)
    bool root = false;                  ///< the system-prompt root: never evicted by the LRU
    bool deep = false;                  ///< moex: at a multiple of ck_every - kept apart from the turns' LRU (ck_deep of them)
    uint64_t used = 0;
    std::vector<uint8_t*> blob;         ///< per stage, pinned
    std::vector<size_t> bytes;
    ~Checkpoint();
};

/// moex (DSV41_PLAN §57): a PARKED conversation - what a request that left it would have overwritten: its tokens and pictures'
/// keys, its positional rows per stage (pageable RAM, in positional_regions' order), the running state at its live end (a
/// checkpoint of it), its own checkpoints (the ones the request that left did not share), and where its last prompt's final
/// turn began (the leave rule's turn_pos).
struct Parked {
    std::vector<int32_t> ids;
    std::vector<ImageKey> imgs;
    int64_t turn_pos = 0;
    std::vector<std::unique_ptr<uint8_t[]>> rows;
    std::vector<size_t> row_bytes;
    std::unique_ptr<Checkpoint> snap;
    std::vector<std::unique_ptr<Checkpoint>> checks;
    uint64_t used = 0;
    size_t bytes() const;
};

/// Control of one prompt-path run from the serving layer (prompt.cpp reads it): chunk boundaries forced at checkpoint
/// positions, each stage's snapshot at them, per-chunk progress, a stop flag.
struct PromptCtl {
    std::vector<int64_t> cuts;                              ///< absolute positions where a chunk must end (sorted)
    std::function<void(int stage, int64_t pos)> on_cut;     ///< called by a stage's thread when it has finished the chunk ending at pos
    std::function<bool(int64_t reached)> progress;          ///< the LAST stage finished a chunk ending at `reached`; false: stop
    std::function<bool()> stop;
    /// moex (DSV41_PLAN §18): the call's (chunk, layer) units done of all (Impl::pl_add; a read of 4096 tokens or more), from
    /// the stages' threads and every 30 s while the engine reads from storage
    std::function<void(int64_t done, int64_t total)> work;
    std::atomic<bool> stopped{false};
    int64_t done = 0;                                       ///< the position every stage has reached (set by prompt())
    /// moex (DSV41_BACKLOG #78): a stop inside a chunk left a stage's running state where no copy could put it back to `done`
    /// (not expected: the stages keep the copies they need) - the state is exact nowhere, the live tokens go
    bool torn = false;
};

struct Engine::Impl {
    EngineOptions opt;
    Hparams hp;
    std::unique_ptr<strata::GgufModel> gm;
    std::map<std::string, DT> t;
    std::vector<Layer> L;
    DT output_norm, output;
    /// moex (DSV41_BACKLOG #140, DSV41_PLAN §76): the BF16 head as two runs of rows - [0, head_lo) on the first GPU, the rest on
    /// the last (load_dense; `output` then names no GPU copy) - and the mapped blocks that carry a window's normed rows to the
    /// first GPU and its logits back (Stage::head).  head_lo 0: the whole head on the last GPU
    int64_t head_lo = 0;
    DT output_lo, output_hi;
    float *hs_x_h = nullptr, *hs_x_d = nullptr, *hs_lo_h = nullptr, *hs_lo_d = nullptr;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
    std::vector<void*> arenas;          ///< per GPU, the dense weights
    std::vector<void*> dense_extra;     ///< moex (§70): the BF16 copies of FP8 candidates that were no FP8 values
    int64_t n_load = 0;                 ///< layers loaded: the trunk, plus the NextN / MTP blocks from index n_layer
    int64_t n_xl = 0;                   ///< the layers with routed experts end here (GLM's NextN has them, MiMo's MTP blocks not)
    bool mtp_on = false;
    bool mtp_graphs_off = false;        ///< a drafter capture failed: eager from here on
    int spec_cap = 0;                   ///< Engine::set_spec: the longest window (0: snap_T)
    int req_cap = 0;                    ///< moex: the running request's window cap (spec_decode; 0 outside): DSpark's drafts
    int stand_in = 0;                   ///< Engine::set_stand_in: windows of this many rows with stand-in drafts (a measurement)
    bool mtp_chain_res = true;          ///< STRATA_X_MTP_CHAIN_H=res (default): the chain feeds the block's residual, not its normed output
    int snap_T = 0;
    DT eh_proj, enorm, hnorm, shead_norm;
    int max_pool = 0, max_cells = 0, ring_rows = 0;
    int64_t swa_rows = 0;               ///< MiMo: the sliding-window layers' cache ring
    int gqa_chunks = 0;                 ///< MiMo: the attention's key splits (the split-K partials' count)
    size_t gub_max = 0, db_max = 0, blob_max = 0;   ///< the largest MoE layer's gate+up / down matrices and blob
    bool mmq_all = true;                ///< every MoE layer's expert types have MMQ tiles (the prompt path's condition)

    // THE EXPERTS' HOST TIERS (tiers.cpp; Strata's resident RAM mode, expert_source.cpp pin_cache_complement): every
    // expert can be read in place from the GGUF mapping (the file tier, eg/eu/ed); the pinned arena holds the CACHE
    // COMPLEMENT - the experts the VRAM tier does not - plus, as far as the RAM budget allows, the prompt path's lend
    // region; an expert entering VRAM hands its RAM blob to the one leaving (the exchange)
    std::unique_ptr<strata::core::PinnedArena> arena;
    bool arena_pinned = false;          ///< registered with CUDA: the copies DMA straight from it
    std::vector<int64_t> ram_off;       ///< [key]: its blob's offset in the arena, or -1 (read from the file)
    /// free exchange blobs: (arena offset, capacity) - the reserved ones hold the largest blob, a returned RAM blob only its
    /// own expert's (a mixed GGUF: MiMo's four blob sizes) - a victim takes the smallest that holds it
    std::vector<std::pair<int64_t, uint64_t>> xpool;
    std::vector<uint64_t> ram_cap;      ///< [key]: the capacity of the RAM blob it holds (ram_off), what it returns to the pool
    uint64_t ram_bytes = 0;             ///< the arena (the experts' blobs and the exchange blobs)
    /// moex (PORT_AUDIT_ALL §6, STRATA_X_RECYCLE=1): [slot] 1 where the plan gave the expert a RAM copy for the prompt path's
    /// lend region - an expert there keeps its RAM blob while the slot is live; anywhere else a double's blob goes back to the
    /// pool (recycle_doubles); the plan's blob count, for the check
    std::vector<char> lend_slot;
    int64_t ram_blobs_plan = 0;
    bool ram_whole = false;              ///< the RAM plan holds every expert outside VRAM (not "what fits", not exchange-only)
    int sync_fills_gap = 0;              ///< (§6) synchronous fills from the file finalize_swaps did in this gap between windows
    void ram_counts(int64_t& doubled, int64_t& file_only) const;   ///< (§6) experts in VRAM and RAM; in neither (not on the SSD)
    bool in_lend_cover(int32_t slot) const;
    void recycle_doubles(const std::vector<char>& spare);
    void tier_check() const;
    struct Src {
        const uint8_t *g, *u, *d;
        bool pinned;                    ///< one blob [g | u | d] in the pinned arena: a single asynchronous copy
    };
    Src src(int64_t key) const;
    // THE SLOTS' INTEGRITY (Strata's verify_slot): each expert slice's hash as read from the file at load; every VRAM slot is
    // hashed on its GPU after the load and after every fill or swap that lands, and compared - a mismatch is an error, and
    // the slot leaves the tier (the CPU computes that expert from its own copy)
    std::vector<uint64_t> khash;        ///< [key * 3 + part]: gate / up / down
    std::vector<char> khash_ok;         ///< [key]: hashed at load
    std::vector<std::pair<int32_t, int32_t>> verify_queue;   ///< (slot, key) filled since the last check
    SliceJob *vjobs_h = nullptr, *vjobs_d = nullptr;         ///< mapped job list
    uint64_t *vout_h = nullptr, *vout_d = nullptr;           ///< mapped results
    uint64_t *vpart_h = nullptr, *vpart_d = nullptr;         ///< ... their parts (moex, §35: STRATA_DSV41_HASH_WIDE)
    size_t vcap = 0;
    int64_t verify_bad = 0, verify_checked = 0;
    // moex (DSV41_BACKLOG #143, DSV41_PLAN §75): a swap's check queued on the copy stream behind its copy, read when the copy's
    // event has completed (finalize_swaps) - the slot is resident only after it, as before, and the host waits for nothing.
    // kVq areas of 3 jobs and 3 x kVqParts sums, mapped; none free: that swap's check the old way (verify_queue)
    static constexpr int kVq = 256, kVqParts = 32;
    SliceJob *vq_jobs_h = nullptr, *vq_jobs_d = nullptr;
    uint64_t *vq_out_h = nullptr, *vq_out_d = nullptr;
    std::vector<int> vq_free;
    bool vq_tried = false;
    int64_t vq_fault = 0;               ///< (test) STRATA_X_SLOT_CHECK_FAULT=N: the N-th such check reads as a mismatch
    bool verify_slots(const std::vector<std::pair<int32_t, int32_t>>& sk, bool fatal, std::string& err);
    std::unique_ptr<kc::ExpertPool> pool;
    std::mutex pool_mu;                 ///< the prompt path's CPU reads (moe_prompt_cpu) from the stages' threads
    std::vector<uint8_t> act_q;
    float *h_x = nullptr, *h_parts = nullptr, *h_embd = nullptr;
    int32_t *h_ids = nullptr, *h_amax = nullptr;
    // upstream's #879 guard: every verify window's head logits checked for a NaN or an infinity beside its sampling
    // (kn::logits_nonfinite_rows, one flag word per slice of a row, written into mapped memory and read after the sync the
    // sampling already does).  A flagged window emits none of its tokens; the request ends `nonfinite` and the engine
    // forgets every state that may carry the poison (Engine::forget), so the next request reads its prompt from token 0 -
    // the server sends a request that has produced nothing yet once more.  STRATA_NAN_GUARD=0: off (no kernel launched).
    uint32_t* nf_h = nullptr;
    uint32_t* nf_d = nullptr;
    int nf_on = 0;                      ///< 0 not set up yet, 1 on, -1 off
    int64_t nf_windows = 0;             ///< windows checked (STRATA_NAN_INJECT=n[,n...] flags row 0 of the n-th, from 1)
    int nf_row = -1;                    ///< the last sample_rows' first non-finite row (-1: none)
    bool nan_guard_ready(Stage& g);

    // moex: the lent experts' file pages read in while the prompt runs (warm_lend, prompt.cpp) - one thread, the newest job
    std::thread warm_thr;
    std::mutex warm_mu;
    std::condition_variable warm_cv;
    std::vector<std::pair<const uint8_t*, size_t>> warm_job;   ///< ranges of the GGUF mapping
    std::atomic<bool> warm_new{false}, warm_stop{false};
    void warm_start();                  ///< the thread, before the host thread pins itself (it inherits the load's CPUs)
    void warm_lend();                   ///< a new job: the experts the stages just lent that have no RAM copy

    // the stages (GPUs): layer il runs on stages[ldev[il]]; the head and the NextN block on the last
    std::vector<std::unique_ptr<Stage>> stages;
    std::vector<int> ldev;
    Stage& stage_of(int64_t il) { return *stages[(size_t) ldev[(size_t) il]]; }
    Stage& last_stage() { return *stages.back(); }
    /// moex (2026-11-26): the device the CALLING THREAD made current. CUDA's current device is per thread, and one cached int
    /// for the engine let a thread skip its cudaSetDevice because another thread had set that device (the prompt stages'
    /// threads and the head over both GPUs call set_dev too). reset_dev_cache() forgets it, on the calling thread.
    static int& dev_cache() {
        static thread_local int d = -1;
        return d;
    }
    void set_dev(int dev) const {
        int& c = dev_cache();
        if (dev != c) {
            cudaSetDevice(dev);
            c = dev;
        }
    }
    void reset_dev_cache() const { dev_cache() = -1; }

    // the VRAM expert tier: uniform slots of the largest blob; a residency table; decayed routing counts
    int64_t n_slots = 0;
    size_t slot_bytes = 0;
    /// moex (DSV41_PLAN §24): STRATA_DSV41_SLOT_SIZES=1 - every slot its expert's size (Stage::slot_off; slot_bytes stays the
    /// uniform size, every expert's bound); the decode's residency tables then hold slot offsets in kSlotUnit units
    bool slot_sized = false;
    static constexpr size_t kSlotUnit = 256;
    /// a prompt ring entry's bytes on stage di: the largest blob of every layer (256-rounded) - moex (DSV41_PLAN §25), opt-in
    /// STRATA_DSV41_STAGE_RING=1: of the stage's own layers (the ring holds only their experts); di < 0: every layer's
    size_t ring_blob(int di) const;
    size_t slot_cap(int32_t slot) const;   ///< a slot's bytes
    std::vector<int> slot_dev;
    std::vector<int32_t> resident;      ///< [layer * n_expert + e] -> slot, or -1
    std::vector<int32_t> owner;         ///< [slot] -> layer * n_expert + e, or -1
    std::vector<float> heat;
    std::vector<uint64_t> routed;       ///< [key]: the run's decode-window routings, undecayed (the expert profile --profile-save
                                        ///< writes; as Strata's #477, a prompt read's routing is not counted)
    std::vector<int32_t> prior_rank;    ///< [key]: its line in the --expert-profile this run started from (-1: not in it)
    std::vector<double> prior_count;    ///< [key]: that line's count
    bool save_profile(const std::string& path, std::string& err) const;
    bool sync_check = false;            ///< STRATA_X_SYNC_CHECK=1: the prompt path syncs and checks after every step (debugging)
    std::map<std::string, uint64_t> dense_hash;   ///< STRATA_X_SYNC_CHECK=2: each dense tensor's hash as read from the file
    /// ... and every dense tensor of layer il (-1: all) read back from its GPU and compared with it
    bool verify_dense(int il, const char* when, std::string& err);
    std::vector<char> slot_pinned;      ///< [slot]: held by the NextN drafter for good (never a swap's victim)
    /// THE DRAFTER'S REGION: the NextN layer's experts, all of them, in their own VRAM allocation on the last GPU (their blob
    /// may be larger than a trunk slot): virtual slots [n_slots, n_slots + nx_count) - slot_ptr() maps them, the residency
    /// table holds them, nothing lends or swaps them
    uint8_t* nx_base = nullptr;
    size_t nx_bytes = 0;
    int64_t nx_count = 0;
    std::vector<int32_t> token_misses;
    uint8_t* staging = nullptr;         ///< pinned, one blob (slot fills when the experts are not in the arena)
    std::vector<int> miss_rows;
    uint8_t* slot_ptr(int32_t slot) const;

    // decode bookkeeping
    int last_T = 0;
    int logit_rows = 0;
    int64_t prompt_short = 64;          ///< Strata's --short-read: a prompt segment this short goes through windows
    /// moex: what the two ways of reading a short segment have cost lately (EMAs, session.cpp): decode windows per token
    /// (they slow down with depth: MiMo ~20 ms a token shallow, ~40 at 100K) and the prompt path per chunk (its streamed
    /// experts, ~12-15 s whatever the chunk's length); short_read_limit() lowers prompt_short to where a chunk is cheaper
    double rw_ms_tok = 0.0, chunk_ms = 0.0;
    int last_prompt_chunks = 0;         ///< the chunks the last prompt() call ran
    int64_t cpu_read_req = -1;          ///< the running request's read_cpu (< 0: EngineOptions::cpu_read)
    /// EngineOptions::cpu_read auto: what a token costs through the CPU-assisted read and a chunk through the stream (EMAs of
    /// this process's reads, priors measured on MiMo); a segment reads without the stream up to where the two meet
    double cpu_ms_tok = 0.0, stream_chunk_ms = 0.0;
    int64_t cpu_read_limit() const;
    /// EngineOptions::prefill_share: the pool's speed on the prompt's expert jobs (blob bytes per ms, a job of nt tokens
    /// weighing 1 + 0.2 (nt - 1)), a shared read's cost per token, the running request's read_share; share_limit() is where a
    /// shared read of n tokens costs a streamed chunk (both measured, MiMo's priors on this box until then)
    std::atomic<double> share_cpu_rate{0.0};   ///< written under pool_mu (both stages' threads may share)
    int64_t share_req = -1;
    int64_t ring_req = -1;              ///< the running request's read_ring (< 0: ring_slots_for's own rule)
    /// the shared reads' cost as a + b n (ms): exponentially weighted sums (w, w n, w n^2, w ms, w n ms) of this process's
    /// shared prompt() calls, seeded with two priors measured on MiMo here; share_limit() solves a + b n = a streamed chunk
    double sh_fit[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    void share_observe(double n, double ms, double w = 1.0);
    int64_t share_limit() const;
    int64_t short_read_limit() const;
    moex::Recorder* rec = nullptr;
    mutable EngineStats st_sum;

    ~Impl();
    bool fail(std::string& err, const std::string& msg) {
        err = msg;
        return false;
    }
    bool load_hparams(std::string& err);
    bool load_hparams_mimo(std::string& err);
    // DeepSeek-V4 (dsv4.cpp)
    bool load_hparams_dsv4(std::string& err);
    bool bind_layers_dsv4(std::string& err);
    bool alloc_state_dsv4(std::string& err);
    // DeepSeek-V4.1 (dsv41.cpp): the deepseek41 GGUF's hparams and tensors, its shared decode state
    bool load_hparams_dsv41(std::string& err);
    bool bind_layers_dsv41(std::string& err);
    bool bind_head_split(std::string& err);   ///< the head over both GPUs (head_lo > 0: DeepSeek-V4.1, ERNIE-5.5)
    bool alloc_state_dsv41(std::string& err);
    // Hy4-flash (hy4.cpp): the hy_v4_flash GGUF's hparams and tensors, its decode state, its Engram
    bool load_hparams_hy4(std::string& err);
    bool bind_layers_hy4(std::string& err);
    bool alloc_state_hy4(std::string& err);
    // ERNIE-5.5-Flash (ernie.cpp)
    bool load_hparams_ernie(std::string& err);
    bool bind_layers_ernie(std::string& err);
    bool alloc_state_ernie(std::string& err);
    uint64_t ernie_state_bytes(int64_t il) const;   ///< a layer's decode state (HCA: dsv4_state_bytes; DSA: rows of 576 + the keys)
    ErnieMultiMax mmax;                             ///< the head's MultiMax (output.multimax_ranges / _ts)
    /// moex vision: the request's M-RoPE table from `img` into every stage's hy_mpos (a no-op while both this request and the
    /// last have no picture); false when a picture's grid does not fit its span
    bool hy4_mpos_set(std::string& err);
    bool hy_mpos_live = false;          ///< the device tables hold a picture's positions (a header other than {0, 0, 0})
    std::vector<int32_t> h_mpos;        ///< the host copy (header as int32 pairs, then the triples)
    bool load_engram_hy4(std::string& err);   ///< after bind_layers: the table mapped (no-op without an Engram)
    /// The Engram's table (per_layer_token_embd, FP8 E4M3 rows of `dim` bytes in the GGUF), the compressed id of every
    /// position read so far, and the staging of a window's rows ([slot j][cap][cols][dim]) and validity ([cap][cols])
    struct Hy4Eg {
        static constexpr int32_t kDead = -1;   ///< an image position: in no n-gram, and adds nothing
        int L = 0, cols = 0, dim = 0;
        int fmt = 0;   ///< the table's rows: 0 F8_E4M3 (a byte a value: Q4_K_M, Q3_K, IQ2_AH), 1 Q8_0 (the SPARK-4Bit sets)
        int rb = 0;    ///< bytes a row (dim, or dim / 32 blocks of 34)
        const uint8_t* table = nullptr;
        uint64_t rows = 0;
        std::vector<int32_t> ids;
        uint8_t* h_rows = nullptr;
        float* h_valid = nullptr;
        int cap = 0;
        /// moex (2026-11-19): the rows read lately, row by row - a 4-way set-associative cache, a lock a set.  The page cache
        /// keeps a 4 KB page for each 96-byte row the table's hashed order scatters: ~1 useful row a page, so ~24 GB of it held
        /// ~6M of a long session's rows and the rest came off the drive again.  Rows are copied out of the table on a miss: the
        /// same bytes.  STRATA_HY4_EG_CACHE_MIB sizes it (default 2048; 0: none); its memory is touched only as rows arrive.
        struct RowCache {
            int rb = 0;
            uint64_t n_sets = 0;
            std::unique_ptr<uint64_t[]> key;              ///< n_sets x 4: row + 1 (0: empty)
            std::unique_ptr<uint8_t[]> data;              ///< n_sets x 4 x rb
            /// a set's lock: one of 4,096 striped mutexes (a spin lock a set stalled the prompt's 2 x 32 gathering threads whenever
            /// a holder was preempted: a 16K read 28 -> 109 s)
            struct alignas(64) Stripe {
                std::mutex mu;
            };
            static constexpr int kStripes = 4096;
            std::unique_ptr<Stripe[]> lk;
            std::unique_ptr<uint8_t[]> victim;            ///< a set's next way to replace
            std::atomic<uint64_t> hits{0}, misses{0};
            bool on() const { return n_sets > 0; }
            void init(int row_bytes, uint64_t bytes);
            bool get(uint64_t row, uint8_t* out);
            void put(uint64_t row, const uint8_t* in);
        };
        RowCache cache;
    };
    std::unique_ptr<Hy4Eg> heg;
    /// moex (2026-11-19), opt-in STRATA_HY4_KV_GROW=1: Hy4-flash's sparse layers' caches and pooled keys on CUDA VMM - the
    /// whole context's addresses reserved at the start, memory mapped as the positions arrive (hy4_kv_ensure); the rows past
    /// the mapped ones are never read (attention and selection read positions before the token's own)
    struct VmmArr {
        unsigned long long va = 0;           ///< a CUdeviceptr
        size_t reserved = 0, mapped = 0;
        int dev = -1;
        std::vector<unsigned long long> h;   ///< a CUmemGenericAllocationHandle a mapped granule
    };
    bool kv_grow = false;
    size_t vmm_gran = 0;
    int64_t kv_rows_mapped = 0;             ///< every sparse layer's mapped rows (their pooled keys' with them)
    std::vector<VmmArr> kv_vmm, pool_vmm;   ///< [layer] (empty for a sliding-window layer)
    bool vmm_reserve(VmmArr& a, int dev, size_t bytes, std::string& err);
    bool vmm_map_to(VmmArr& a, size_t bytes, std::string& err);
    bool hy4_kv_ensure(int64_t pos_end, std::string& err);
    size_t kvg_kv_bytes(int64_t rows) const;     ///< a growable layer's cache bytes for `rows` positions (hy4.cpp)
    size_t kvg_pool_bytes(int64_t rows) const;   ///< ... its second array's: Hy4's pooled keys, ERNIE's index keys
    /// ... a request that starts well short of the mapped rows (a new conversation, a compacted one): the rows past its need
    /// unmapped and each stage's retired slots back in the tier, empty, as far as its free memory goes beside its reserve
    bool hy4_kv_release(int64_t pos_end, std::string& err);
    std::vector<VmmArr> slot_vmm;           ///< [stage] the expert slots on VMM (STRATA_HY4_KV_GROW), so their top can be retired
    std::vector<int64_t> slot_floor;        ///< [stage] the slots that fit beside the whole context's caches (the plan)
    std::vector<int64_t> slot_n0;           ///< [stage] the slots the plan gave it (its index range: never more)
    int64_t kv_retired = 0;                 ///< slots given to the caches so far (less the ones given back)
    int64_t kv_rows_init = 0;               ///< the rows mapped at the start (STRATA_HY4_KV_GROW_INIT): never released below
    int64_t kv_step = 65536;                ///< the growth's step (STRATA_HY4_KV_GROW_STEP); a release frees a step at least
    size_t vram_keep(const Stage& g) const;  ///< what a stage keeps free beside the caches' growth: its plan's reserve
    void vmm_unmap_from(VmmArr& a, size_t bytes);
    bool hy4_retire_slots(Stage& g, int64_t new_n, std::string& err);
    /// moex (2026-11-19): the later stages' Engram rows of a prompt chunk, gathered into the row cache while the first stage runs
    /// the chunk (by its first position; a later stage waits for its chunk's before its own gather)
    std::mutex eg_pf_mu;
    std::map<int64_t, std::future<void>> eg_pf;
    std::map<int64_t, std::future<void>> eg_pf0;   ///< ... and the first stage's own rows of its next chunk (by its first position)
    int64_t eg_pad_token = -1;            ///< the Engram's pad TOKEN id (a position holding it adds nothing; vLLM's no_hash)
    int hy4_eg_index(int64_t il) const;   ///< il's Engram slot, or -1
    /// Positions pos0 .. pos0 + T - 1 read these tokens: their ids noted, their rows hashed, read from the table and uploaded
    /// to every stage that runs an Engram layer (the rows of `cap` >= T a slot)
    bool hy4_eg_window(const int32_t* toks, int64_t pos0, int T, std::string& err);
    /// ... a prompt chunk's for stage g's layers alone (the chunk's tokens: each stage runs its chunk on its own thread)
    bool hy4_eg_chunk(Stage& g, const int32_t* toks, int64_t pos0, int T, std::string& err);
    void hy4_eg_note(const int32_t* toks, int64_t pos0, int T);
    bool hy4_eg_rows(const int32_t* toks, int64_t pos0, int T, int j0, int j1, uint8_t* rows_out, float* valid_out, int cap,
                     std::string& err, int max_threads = 32);
    /// The sparse layers' scored groups: the indexer's budget less the token's own group (511 of 2,048 / 4)
    int hy4_n_top() const { return (int) (hp.idx_topk / hp.kpool - (hp.select_tail ? 1 : 0)); }
    bool load_experts_records(std::string& err);   ///< the routed experts from the pack's record files (in place)
    /// DeepSeek-V4.1's SSD TIER (DSV41_PLAN.md section 5 B-C): an expert in neither VRAM nor RAM is read from its record file
    /// with O_DIRECT into pinned memory - never through the page cache.  Faulting the record mappings in instead churned the
    /// page cache with ~2-3 GB a token next to 87 GB of pinned RAM, until the session's memory pressure passed systemd-oomd's
    /// limit and it ended the terminal (2026-10-18).
    struct RecordIo {
        /// moex (DSV41_PLAN §73): the cache's blocks faulted in and registered by this thread after rec_io_init returns (beside
        /// the VRAM experts' load); joined before the engine says READY.  pin_pending: their bytes not faulted in yet (the RAM
        /// plan counts them as used); blocks_faulted: how many blocks, in order, are faulted in (a read may land in one only
        /// then: the fault-in writes a byte of every page); slot_block: each slot's block
        std::thread pin_thr;
        std::atomic<uint64_t> pin_pending{0};
        std::atomic<size_t> blocks_faulted{SIZE_MAX};
        std::vector<int> slot_block;
        std::vector<int> fd;                         ///< per layer: its record file opened O_DIRECT (-1: no records)
        std::vector<int> xfd;                        ///< moex: per file of the expert GGUF set, opened O_DIRECT (-1: absent)
        std::unique_ptr<dsv41::ExpertIo> io;         ///< the readers (4 MiB pieces in flight together)
        uint64_t slot = 0;                           ///< a cache slot: the largest record stride (a 4 KiB multiple)
        /// THE SSD CACHE (the decode's): pinned record slots, least recently used first - a demand read lands in one and stays
        /// for the next windows; a read-ahead (DSV41_PLAN 5 C.3: the next layer's experts predicted on the GPU from this
        /// layer's FFN rows) lands in one and is promoted when its layer asks for it.  A slot a layer computes from is pinned
        /// for that call (pin == epoch); a slot with a read in flight is never evicted.
        struct Slot {
            int64_t key = -1;
            uint8_t* p = nullptr;
            dsv41::ExpertIo::Handle h;
            uint64_t tick = 0;
            bool spec = false;   ///< read ahead, not asked for yet
            uint64_t pin = 0;
            int pool = 0;        ///< the size class it belongs to (pool_stride)
        };
        std::vector<Slot> cs;
        /// --ssd-cache-gib: one pool of slots per record size (a mixed pack's Q3_K and Q2_K layers), each its share of the
        /// bytes by its layers' record bytes - a slot of the largest record left 24% of a Q2_K record's slot empty.  The
        /// 64-slot default: one pool of the largest record for every layer.
        std::vector<uint64_t> pool_stride;
        std::vector<std::vector<int>> pool_slots;   ///< [pool]: its slots, in cs
        std::vector<int> layer_pool;                ///< [layer]: the pool its records go to
        std::vector<uint8_t> cached;                ///< [key]: a slot holds its record (or its read is in flight)
        std::vector<uint8_t*> blocks;                ///< the slots' pinned memory, kGroup slots an allocation
        std::vector<uint64_t> block_reg;             ///< [block]: its bytes when registered (STRATA_DSV41_FAST_PIN), 0: cudaHostAlloc'ed
        static constexpr int kGroup = 64;
        /// moex (DSV41_PLAN §63 step 2): [block] its first slot in cs and its slots; [slot] lent with its block to a long read's
        /// host rows (holds no record, never a victim); the blocks lent now
        std::vector<int> block_first, block_n;
        std::vector<uint8_t> lent;
        std::vector<int> lent_blocks;
        std::unordered_map<int64_t, int> where;
        uint64_t clock = 0, epoch = 1;
        int64_t reads = 0, hits = 0, ahead_issued = 0, ahead_used = 0, ahead_wasted = 0;
        int64_t fill_cached = 0, fill_reads = 0;     ///< the VRAM tier's fills of SSD experts: from a cache slot / read again
        std::atomic<int64_t> rec_reads{0};           ///< records read by rec_read (the prompt stream's, the fills'): the stages' threads
        std::atomic<int64_t> rec_cached{0};          ///< ... and the prompt stream's records taken from the cache instead
        bool ahead_on = false;                       ///< STRATA_DSV41_AHEAD=1: read-ahead on (it lost both A/Bs)
        /// STRATA_DSV41_PROMPT_KEEP (on by default; =0 off) with STRATA_DSV41_PROMPT_KEEP_SHARE=F (0 by default): the prompt
        /// stream reads its SSD records into cache slots - up to F of each pool a prompt, and the experts it lent from VRAM
        /// whatever the share - instead of into bounce slots, so the lent slots'
        /// refill, the decode after the prompt and the next prompts find them in RAM (they were read and thrown away).  The
        /// stages' issuers insert together: `mu` guards the cache's tables then, and a slot the running prompt reads into or
        /// copies from stays put until the prompt ends (ppin) - a copy from it may still be queued.
        std::mutex mu;
        std::vector<uint8_t> ppin;                   ///< [slot]: used by the running prompt
        std::vector<int> keep_cap, kept;             ///< [pool]: the slots a prompt may fill, and how many it has
        /// A prompt of two or more chunks streams layers 0 .. replay_layer - 1 once a chunk: with STRATA_DSV41_PROMPT_KEEP_LONG=F
        /// (opt-in, on in strata-dsv41.json: the cold 44K read's drive bytes -27%, DSV41_BACKLOG #29) its reads of them are
        /// kept, up to F of each pool, so its later chunks find them in RAM,
        /// and the victims are taken from the layers the chunks do not stream (prefer_ge and up: the replayed layers) first
        std::vector<int> keep_cap_long;
        std::vector<int> kept_layer;                 ///< moex (§59 P6): [layer] a long prompt's kept records (KEEP_SPREAD)
        bool long_now = false;                       ///< the running prompt is such a prompt
        int prefer_ge = -1;                          ///< ... the first layer its victims are taken from first (-1: plain LRU)
        bool keep_on = false;
        std::atomic<int64_t> kept_total{0};          ///< records the prompts read into the cache (stats)
        std::atomic<int64_t> adm_free{0}, adm_hot{0};   ///< moex (§41): of them past the share, into free slots / routed by 1 / D
        /// moex (DSV41_PLAN §63): a call read in one host window - its read-ahead takes the least recently used slot past the share,
        /// up to deep_cap held at once (dpin: such a slot, deep_n of them; a layer's go back once its chunks have run it,
        /// prompt_keep_release), so the drive reads layers ahead while GPU 0 runs a long one; adm_deep counts them
        int deep_cap = 0, deep_n = 0;
        std::vector<uint8_t> dpin;
        std::atomic<int64_t> adm_deep{0};
        std::atomic<int64_t> prompt_sync{0};         ///< the prompt stream's records read on the spot, not ahead (none expected)
        std::atomic<int64_t> prompt_orphan{0};       ///< ... records read ahead and then not taken (dropped from the queue)
        /// the prompt path's time with none of a stage's reads in flight, by what it waited for (microseconds, stats): the lend
        /// (the lent slots' copies out), each layer's routing, the ring (the GPU consuming), a bounce slot's copy, the refill
        std::atomic<int64_t> us_lend{0}, us_gate{0}, us_ring{0}, us_bounce{0}, us_refill{0};
        std::vector<int> hold;                       ///< [slot]: the VRAM tier's copies in flight from it (not evicted meanwhile)
    };
    std::unique_ptr<RecordIo> rio;
    bool rec_io_init(std::string& err);
    /// experts es[0 .. n) of layer il, their whole records, into dst[0 .. n) (pinned, 4 KiB aligned, rio->slot bytes each),
    /// read together; returns once all have landed
    bool rec_read(int64_t il, const int64_t* es, int n, uint8_t* const* dst, std::string& err);
    /// moex (the GGUF set): expert e of layer il into the slot dst - a pack's record, or a GGUF set's virtual record in its
    /// three slices (rec_gguf); every read of a record goes through here
    dsv41::ExpertIo::Handle rec_io(int64_t il, int64_t e, uint8_t* dst, dsv41::IoClass cls);
    /// moex (the GGUF set): the files of --experts-gguf's set (null: not written yet - those layers keep the pack's records)
    std::vector<std::unique_ptr<strata::GgufFile>> xg;
    std::vector<uint8_t> xg_has;   ///< moex: [layer] its experts are in the GGUF set (the pack's records only for the others)
    bool load_experts_gguf(std::string& err);
    bool load_record_layers(std::string& err);   ///< the pack's records of every MoE layer (no readers yet)
    bool load_experts_tail(std::string& err);    ///< the record readers and the CPU pool, after the layers
    /// ... one record submitted at demand priority, not waited for: its handle (the prompt stream's reads ahead)
    dsv41::ExpertIo::Handle rec_submit(int64_t il, int64_t e, uint8_t* dst);
    /// A layer's CPU expert jobs (blobs from src(); keys[i] = jobs[i]'s expert) with the SSD tier: the SSD experts' records
    /// are read into the staging slots while the pool computes the RAM-resident ones, then the pool computes those
    bool run_cpu_jobs(int64_t il, std::vector<kc::ExpertJobMulti>& jobs, const std::vector<int64_t>& keys, std::string& err,
                      const std::vector<int32_t>* ahead = nullptr, int64_t ahead_layer = -1);
    /// The SSD cache's slot for `key`, its read submitted when absent (demand or read-ahead); -1: every slot pinned or in flight
    int ssd_slot(int64_t key, bool demand);
    /// The cache's victim for key's record: the least recently used slot of its pool (that pool full: a larger one) not pinned
    /// by this layer's call, not used by the running prompt, with no read in flight; -1 if none.  Under rio->mu.
    int ssd_victim(int64_t key);
    /// moex (DSV41_PLAN §63 step 2): the SSD cache's blocks of `bytes` or more that could be lent now (no slot pinned, held,
    /// used by the running prompt or reading), and up to n of them lent - their records dropped, the least recently used
    /// blocks first - into out (how many); cache_unlend gives every lent block back empty (the caller has synchronized the
    /// GPUs' work on them)
    int cache_lendable(size_t bytes);
    int cache_lend(int n, size_t bytes, std::vector<uint8_t*>& out);
    void cache_unlend();
    /// STRATA_DSV41_PROMPT_KEEP: the prompt stream's read of key into a cache slot - the slot, its read submitted; -2 when the
    /// cache holds the record already (issue_entry copies it from there); -1 when the prompt has filled its share or no slot
    /// is free (a bounce slot then).  The stages' issuers call it together.
    int prompt_keep_slot(int64_t key, int cnt = 0, int rows = 0, int hot = -1);   ///< cnt: the chunk's rows routed to it (gated), rows: the chunk's
    void prompt_keep_drop(int slot);   ///< a failed read leaves the cache
    /// ... a cache slot for key's record without a read (the caller fills it - a lent VRAM slot's copy), kept for the prompt;
    /// -1 when the cache holds it already (kept for the prompt too) or no slot is free
    int prompt_keep_take(int64_t key);
    void prompt_keep_reset();          ///< the prompt is over: its slots are ordinary LRU slots again
    /// moex (DSV41_PLAN §37), opt-in STRATA_DSV41_TAIL_AHEAD=1: during GPU 0's last layer-major window the deferred tails'
    /// records (the last stage's layers from replay_layer on, neither in VRAM nor in the RAM copy) go into SSD cache slots at
    /// the read-ahead's class - the ones the cache holds pinned first, then the others read as slots come free (the window
    /// releases each layer it has passed: no later window reads them) - so the tails' run copies them from RAM
    int prompt_keep_pin(int64_t key);       ///< key's record pinned for the prompt if the cache holds it (its slot), else -1
    int prompt_keep_ahead(int64_t key);     ///< ... else read into a free slot at the read-ahead's class: -2 held, -1 no slot
    void prompt_keep_release(int64_t il);   ///< layer il's slots kept for the prompt unpinned (the lent experts' stay: the refill)
    void tail_ahead_start(int64_t l0, int64_t l1);
    void tail_ahead_stop();
    std::thread tail_pf;
    std::atomic<bool> tail_pf_stop{false};
    std::atomic<int64_t> tail_pf_read{0}, tail_pf_held{0};
    std::vector<int> tail_pf_slots;         ///< (the prefetcher's reads: promoted to demand when the tails start)
    /// The VRAM tier's swap of an SSD-tier expert: its record's cache slot, held against eviction until the copy from it lands
    /// (ssd_release); -1 when the cache does not hold it (or its read is still in flight)
    int ssd_hold(int64_t key);
    int ssd_take_for(int64_t key);                 ///< moex (§44): a held, unpublished cache slot for key's record copied in
    void ssd_warm(const std::vector<int64_t>& rank, size_t n_prof, const std::vector<int32_t>* planned = nullptr);   ///< moex (§69): the cache filled from the profile at the start (its reads; §73: before the load pass)
    void ssd_warm_finish();                        ///< moex (§73): ... its reads waited for and the slots published
    std::vector<std::pair<int64_t, int>> warm_picks;   ///< moex (§73): the warm fill's (expert, slot) in flight
    std::vector<dsv41::ExpertIo::Handle> warm_hs;
    std::thread warm_submit;                       ///< moex (§73): submits the warm fill's reads as their blocks are faulted in
    std::chrono::steady_clock::time_point warm_t0;
    void ssd_publish(int slot, int64_t key);       ///< ... its copy landed: the cache holds key there
    void ssd_free_dup(int slot, int64_t key);      ///< ... key's record in VRAM now: its cache slot free
    void ssd_release(int slot);
    std::vector<uint8_t> lent_key;     ///< [key]: an expert the running prompt lent from VRAM (its refill reads it again)
    /// The prompt stream's look into the SSD cache: key's record when a slot holds it (a read still in flight is waited for),
    /// else nullptr.  Read-only - the stages' issuers call it together, and no decode runs meanwhile to evict a slot.
    const uint8_t* ssd_cached(int64_t key);
    /// The prompt stream's gate on the SSD tier: a chunk of at most this many tokens streams only its routed experts, each
    /// layer's after its routing (STRATA_DSV41_STREAM_GATE=N; 0: every chunk streams every non-resident expert ahead).  Every
    /// chunk by default since 2026-10-18: a 2,000-token chunk routes ~11,200 of the ~13,800 experts it streamed ahead, and
    /// gated it read in 38.5 s instead of 47.1 (7,820 tokens: 52.2 instead of 55.8 s; the cache warmed) - it was 1,024
    static int64_t stream_gate() {
        static const int64_t v = [] {
            const char* e = std::getenv("STRATA_DSV41_STREAM_GATE");
            return e ? std::atoll(e) : INT64_MAX;
        }();
        return v;
    }
    /// moex (DSV41_PLAN §59 P1), STRATA_DSV41_LAST_RING=1: the prompt path's last layer runs only its attention half - its
    /// ring rows, which the decode after the read start reads - and no FFN half: no logits come from the prompt path (its
    /// last token goes through the first decode window, and --kld scores the decode windows' rows), so the last layer's
    /// output is never read.  Its experts leave the streams and the tails' read-ahead.  Off with a drafter (DSpark, DFlash,
    /// NextN read the trunk's last layers' rows).  Exact: every value read later is the one the full layer computes.
    bool last_ring_only(int64_t il) const {
        static const bool on = [] {
            const char* e = std::getenv("STRATA_DSV41_LAST_RING");
            return e && e[0] == '1';
        }();
        return on && hp.v41 && il == hp.n_layer - 1 && !dsp && !dfl && !mtp_on;
    }
    /// The VRAM tier's swaps copy an SSD-tier expert from its SSD cache slot asynchronously (like a RAM copy's) instead of
    /// filling the slot synchronously between windows; one the cache does not hold waits for a later round.  On by default
    /// since 2026-10-18 (the final decode config, 3 passes of the coding text: 10.69 / 15.73 / 19.67 -> 11.06 / 19.00 / 20.54
    /// tok/s; the host's time between windows 8.9 -> 4.4 ms a token); STRATA_DSV41_ASYNC_FILL=0: the synchronous fills
    static bool ssd_async_fill() {
        static const bool v = [] {
            const char* e = std::getenv("STRATA_DSV41_ASYNC_FILL");
            return !(e && e[0] == '0');
        }();
        return v;
    }
    /// moex (DSV41_PLAN §44), opt-in STRATA_DSV41_SWAP_CACHE=1: the swaps keep the SSD cache exclusive of VRAM.  A swap whose
    /// incoming record came from its cache slot gave a VRAM-only victim one of the 96 exchange blobs, which nothing handed back
    /// - all spent within the first reply, then such victims dropped (their next routing a drive read) - and left the
    /// incoming record in its slot beside its VRAM copy.  Now the victim goes into a cache slot of its own (ssd_take_for) and
    /// the incoming record's slot is freed (ssd_free_dup).  Where an expert is computed moves with the tier, as with any tier
    /// change; a slot holds the same bytes
    static bool swap_cache() {
        static const bool v = [] {
            const char* e = std::getenv("STRATA_DSV41_SWAP_CACHE");
            return e && e[0] == '1';
        }();
        return v;
    }
    /// The read-ahead: layer il's predicted experts (T * k ids, the GPU's guess from layer il - 1's FFN rows) not in VRAM or RAM
    void ssd_ahead(int64_t il, const int32_t* ids, int n);
    /// moex (DSV41_BACKLOG #52): the SSD cache's records as "layer expert" lines, least recently used first, and such a list
    /// read back into the cache (at the reads' lowest priority, in the background): the records it asked for
    bool save_cache_keys(const std::string& path, std::string& err);
    int64_t refill_cache(const std::string& path);
    /// moex (DSV41_PLAN §54): DeepSeek-V4.1's image encoder on GPU 0 (vision.cpp, Engine::encode_pictures; none without
    /// EngineOptions::vision_manifest), its weights read on the first picture
    std::unique_ptr<Dsv41Vision> vision;
    /// moex: Hy4-flash's (its mmproj GGUF as --vision-weights; vision.cpp), the same borrowed workspace
    std::unique_ptr<Hy4Vision> hvision;
    bool encode_pictures(const std::vector<std::pair<std::string, std::string>>& files, std::vector<int64_t>& rows, double& ms,
                         std::string& how, std::string& err);
    /// layer il predicts layer il + 1's experts on its GPU (DeepSeek-V4.1's records with read-ahead on, the next layer there)
    bool ahead_from(int64_t il) const;
    /// moex (DSV41_PLAN §27): STRATA_DSV41_AHEAD=d - the read-ahead guesses layer il + d from layer il's rows (1: the next)
    int ahead_d = 1;
    EngineStats ssd_st;   ///< run_cpu_jobs' counters (cpu_ram, cpu_ssd, ms_ssd_wait), added into Engine::stats
    std::atomic<uint64_t> eg_rows_asked{0}, eg_rows_read{0};   ///< moex (#103): the deduplicated direct path's rows, asked / read
    std::atomic<int64_t> us_eg_rows{0};   ///< DeepSeek-V4.1: the engram rows' fetch (row numbers, page reads, copies), microseconds
    /// moex (DSV41_BACKLOG #76): the engram rows' fetches in flight (eg_rows).  With eg_first (a layer-major call, opt-in
    /// STRATA_DSV41_EG_FIRST=1) the prompt streams' issuers start no record read from the drive meanwhile - behind a whole
    /// layer's stream the engram's 4 KB page reads waited ~55 ms each (2026-10-21) - and us_eg_first is their time held back
    std::atomic<int> eg_busy{0};
    bool eg_first = false;
    std::atomic<int64_t> us_eg_first{0};
    /// moex (DSV41_PLAN §64), opt-in STRATA_DSV41_EG_URGENT=1: while a prompt stage is blocked on its chunk's engram rows
    /// (eg_need: the stages waiting), the streams' issuers start no record read from the drive - their reads then only fill
    /// the bounce slots ahead, and behind the record readers' ~384 requests the rows' 4 KB pages came at ~13K a second against
    /// ~140K alone (eg_fetch_rate, 2026-10-30) - and the drive guard's floor lets the fetch keep 32 pages in flight, not 1
    /// (one slow-mode episode served 5.4K a second at 32 against ~1K at 1; a deeper one ~1K at 30).  Unlike eg_first, the
    /// records wait only while a GPU does
    bool eg_urgent = false;
    std::atomic<int> eg_need{0};
    std::atomic<int64_t> us_eg_urgent{0};
    bool eg_hold() const {
        return (eg_first && eg_busy.load(std::memory_order_acquire) > 0) || (eg_urgent && eg_need.load(std::memory_order_acquire) > 0);
    }
    /// a prompt stage's wait for its chunk's engram rows (the fetch's future): counted in eg_need while it blocks
    template <class F> std::string eg_await(F& f) {
        if (eg_urgent && f.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            eg_need.fetch_add(1, std::memory_order_acq_rel);
            f.wait();
            eg_need.fetch_sub(1, std::memory_order_acq_rel);
        }
        return f.get();
    }
    /// moex (#81): the prompt heartbeat's view of the call: its chunk (the chunk loop) or window (layer-major), of all, and the
    /// layer-major engram fetches done of those queued
    std::atomic<int> hb_win{-1}, hb_nwin{0}, hb_eg_done{0}, hb_eg_queued{0};
    /// moex (DSV41_PLAN §18): the prompt call's progress lines - on for this call, its (chunk, layer) units done and in all
    bool pl_on = false;
    std::atomic<int64_t> pl_done{0};
    int64_t pl_total = 0;
    void pl_add(int64_t k);
    std::atomic<int64_t> us_eg_wait{0};   ///< moex (#51): ... and how long the first engram layer waited for them, microseconds
    /// STRATA_ROUTE_TRACE=FILE: every decode routing appended to FILE as int32 - layer, T, k, then the T x k expert ids - for an
    /// offline study of the tiers (moex/tools/dsv41/tier_sim.py); unset, one check per layer call
    void trace_route(int64_t il, int T, int k, const int32_t* ids);
    /// key on the SSD tier: a record expert with no RAM copy (the decode, the stream and the fills read it with rec_read)
    bool on_ssd(int64_t key) const {
        return rio && L[(size_t) (key / hp.n_expert)].rec_stride > 0 && (ram_off.empty() || ram_off[(size_t) key] < 0);
    }
    /// DeepSeek-V4.1's engram (dsv41.cpp): its tables' mappings (rows of `dim` FP8 bytes, their E8M0 scales), the hash
    /// inputs (Hparams::eg_*), the compressed id of every position read so far, and q_weight * k_weight on the layer's GPU
    struct Engram {
        static constexpr int32_t kDead = -1;            ///< ids' mark of an image position (engram.py NgramHashState.DEAD)
        int L = 0, cols = 0, dim = 0;                  ///< engram layers, rows per position per layer, row width
        std::vector<int32_t> ids;                       ///< [n_ctx]: the compressed token id at each position read (kDead: an image)
        std::vector<const uint8_t*> w, s;               ///< per engram layer: the rows' and the scales' mappings
        std::vector<size_t> wlen, slen;
        /// moex (the GGUF set, gguf_convert.py): STRATA_DSV41_ENGRAM_GGUF=FILE - the tables are a GGUF file's MXFP8
        /// per_layer_token_embd: w[j] points at layer j's first row there, rows of rb bytes (dim / 32 blocks of [e8m0,
        /// 32 x e4m3]: a row's codes and scales together), s unused; eg_rows splits each row back into the same bytes
        bool mx8 = false;
        /// moex: the GGUF's table is IQ4_NL (unsloth's UD-IQ1_M / UD-IQ1_S): rows of rb = dim / 32 x 18 bytes, copied as they are
        /// into a row's codes place (the scales' place unused) and dequantized on the GPU (dsv41_engram_dequant_iq4nl)
        bool iq4nl = false;
        size_t rb = 0;
        std::vector<int> wfd, sfd;                      ///< per engram layer: the tables opened O_DIRECT (-1: mapped reads only)
        int gfd = -1;                                   ///< moex (§73): the GGUF's MXFP8 table opened O_DIRECT (-1: none)
        std::vector<uint64_t> goff;                     ///< ... and each engram layer's first row's offset in that file
        /// moex (DSV41_BACKLOG #59), opt-in STRATA_DSV41_EG_READER=1: per engram layer, Strata's n-gram reader on the rows'
        /// and on the scales' table (Qwen's ngram/ple_reader: a chunk's pages deduplicated and sorted, a bounded number in
        /// flight, a row cache); empty: off
        std::vector<std::unique_ptr<strata::ngram::PleReader>> rd_w, rd_s;
        std::vector<float*> qk;                         ///< per engram layer: (hc, n) on its GPU
        size_t row_bytes() const { return (size_t) cols * (dim + dim / 32); }   ///< one position's rows of one layer
        /// moex (DSV41_PLAN §18), opt-in STRATA_DSV41_EG_SCALES_PRELOAD=1: the thread reading the scale tables into the page
        /// cache after the load (eg_preload_start), and its stop
        std::thread pre;
        std::atomic<bool> pre_stop{false};
    };
    std::unique_ptr<Engram> eg;
    /// The imatrix collection (Engine::imatrix_begin): per MoE layer the sums of squared FFN inputs (E x n) and SwiGLU hidden
    /// values (E x n_ff) on the layer's GPU, added by the prompt path's MoE (moe_prompt, the stream mode), and the tokens
    /// routed to each expert
    struct Imx {
        std::vector<float*> gu, dn;
        std::vector<uint64_t> cnt;   ///< [layer * n_expert + e]
    };
    std::unique_ptr<Imx> imx;
    bool load_engram(std::string& err);               ///< after bind_layers: the tables mapped, qk uploaded (no-op without)
    void eg_preload_start();                          ///< moex (§18): the scale tables' pre-read, after the load (opt-in)
    int eg_index(int64_t il) const;                    ///< il's engram slot, or -1
    void eg_note(const int32_t* toks, int64_t pos0, int64_t n);   ///< positions pos0 .. pos0 + n - 1 read these tokens
    /// The rows of positions pos0 .. pos0 + T - 1 into an engram buffer of `cap` rows at row r0 (per engram layer j: the
    /// weights of cap x cols rows, then their scales); the positions' ids (eg_note) and the three before them must be set.
    /// (moex, DSV41_BACKLOG #147, DSV41_PLAN §76: [j0, j1) names a run of the engram layers - a prompt chunk's first engram
    /// layer waits for its own rows only, eg_split_ok; -1: to the last.  Only the GGUF table's O_DIRECT path reads a run; the
    /// other paths fetch every layer's rows whatever the run - the same bytes, written again)
    bool eg_rows(int64_t pos0, int T, uint8_t* buf, int cap, int r0, std::string& err, int j0 = 0, int j1 = -1);
    bool eg_split_ok(int T) const;   ///< a chunk of T rows is fetched layer by layer (STRATA_DSV41_EG_SPLIT=0: never)
    /// moex (DSV41_PLAN §42), opt-in STRATA_DSV41_EG_LATE=1: a decode graph window's engram rows (eg_rows: the row numbers,
    /// the pages asked for, the copies - ~0.45 ms of system calls and faults a token) on a thread of their own while the GPUs
    /// run the layers before the first engram layer; the host waits for them just before that layer's MoE is served, and its
    /// graph waits for the host's word (Stage::db_egf) before it reads them
    bool eg_late = false;
    int64_t eg_late_layer = -1;   ///< the first engram layer (eg_late's wait)
    struct EgHelper {
        std::thread th;
        std::mutex mu;
        std::condition_variable cv;
        bool stop = false;
        uint64_t seq = 0;                    ///< jobs posted (the host's; under mu)
        std::atomic<uint64_t> done_seq{0};   ///< the last job done
        int64_t pos0 = 0;
        int T = 0, cap = 0, r0 = 0;
        uint8_t* buf = nullptr;
        bool ok = true;                      ///< the last job's result (read after done_seq)
        std::string err;
        // moex (DSV41_BACKLOG #140, DSV41_PLAN §76; eg_ring): the graph's wait for job arm_seq's rows is released by whoever
        // comes second - the helper when it finishes (arm set), or the host when it arms and finds the job done.  Under mu
        bool arm = false;
        uint64_t arm_seq = 0;
        volatile uint32_t* ring_ptr = nullptr;
        uint32_t ring_val = 0;
    };
    std::unique_ptr<EgHelper> eg_help;
    void eg_help_post(int64_t pos0, int T, uint8_t* buf, int cap, int r0);   ///< eg_rows on the helper
    /// moex (DSV41_BACKLOG #140, DSV41_PLAN §76; STRATA_DSV41_EG_RING=0: off): the first engram layer's rows' half too runs in
    /// the layer before (eg_pre_kv), behind the graph's wait for the rows - which the helper then releases itself as its fetch
    /// ends, while the host is still in that layer's CPU experts (the host's own word came only after them).  eg_help_arm:
    /// after the first stage's graph is launched - the wait's count taken, the word written now if the rows are in already
    bool eg_ring = false;
    void eg_help_arm(Stage& g);
    bool eg_help_wait(std::string& err);   ///< ... done (its result); also before the window buffer is written another way
    void eg_help_stop();
    std::vector<std::pair<void*, size_t>> rec_maps;   ///< the record files' read-only mappings
    bool dsv41_split_ok(int64_t K) const;          ///< a two-GPU split at K shares no compressed rows, list or candidates
    uint64_t dsv41_state_bytes(int64_t il) const;
    int64_t dsv41_score_rows() const { return opt.n_ctx + 1; }   ///< the indexer's score stride: ratio 1 sees every position
    uint64_t dsv4_state_bytes(int64_t il) const;    ///< a layer's decode state (the split's estimate, alloc_state_dsv4's sizes)
    int64_t dsv4_raw_rows() const;                  ///< the raw ring: the 128-position window plus the longest batch
    int dsv4_ring(int r) const { return (r == 128 ? 128 : 8) + kTmax; }   ///< a compressor's state ring (see alloc_state_dsv4)
    Dsv4Compressor dsv4_comp(int ratio, bool lid) const;
    Dsv4Rope rope_pc{};                             ///< the compressed layers' RoPE (YaRN)
    /// DeepSeek-V4's DSpark drafter (dspark.cpp): its blocks are L[base, base + n) (ratio-0 DeepSeek-V4 blocks on the last
    /// GPU), its experts in their own pinned RAM arena computed by the CPU pool
    struct Dspark {
        std::unique_ptr<strata::GgufModel> gm;
        int64_t base = 0, n = 0;
        int block = 0;                              ///< dflash.block_size: drafts per step at most
        std::vector<int64_t> tl;                    ///< target layers (n_layer: the trunk's output)
        DT fc, enc_norm, out_norm, hc_fn, hc_base, hc_scale;
        float *w1 = nullptr, *w2 = nullptr;         ///< the Markov head (F32 copies: [n_vocab][rank])
        int64_t rank = 0;
        kc::NativeFmt fmt;
        std::unique_ptr<strata::core::PinnedArena> arena;
        uint64_t blob = 0;                          ///< one expert's blob (rounded) in the arena
        int32_t mask = -1;
        void* dense = nullptr;                      ///< its dense weights on the last GPU
        std::map<std::string, DT> t;
        bool v41 = false;                           ///< DeepSeek-V4.1's drafter (dsv41.cpp): from the trunk's GGUF and pack
        DT conf;                                    ///< V4.1: the confidence head, F32 [n_embd + rank] on the last GPU
    };
    std::unique_ptr<Dspark> dsp;
    bool load_dspark_v41(std::string& err);   ///< dsv41.cpp: the deepseek41 GGUF's dspark.* tensors, the pack's M<j>.bin
    bool load_dspark(std::string& err);   ///< after bind_layers, before load_experts (its RAM comes first)
    /// DeepSeek-V4's NextN head from its own GGUF (dsv4_mtp.cpp; EngineOptions::draft_mtp): L[n_layer], a ratio-0 block on
    /// the last GPU; its routed experts in their own pinned RAM arena (the CPU pool); its embedding (the file mapping, host),
    /// head reduce, norm and head its own - mainline's draft model
    struct Dsv4Mtp {
        std::unique_ptr<strata::GgufModel> gm;
        DT eh_proj, enorm, hnorm, shead_norm, hc_base, hc_scale, output;
        float eps = 0.0f;                           ///< its RMS norms' eps (its file's attention.layer_norm_rms_epsilon)
        float* hc_fn = nullptr;                     ///< output_hc_fn as F32 [hc * n -> hc] (the head reduce's GEMV)
        const uint8_t* embd = nullptr;              ///< token_embd.weight in the file mapping
        int embd_type = -1;
        size_t embd_row = 0;
        std::unique_ptr<strata::core::PinnedArena> arena;   ///< STRATA_X_MTP_ARENA=1: its experts copied into pinned RAM
        uint64_t blob = 0;                          ///< one expert's blob (rounded) in the arena
        Stage::ExpertSrc src;                       ///< where the pool reads its experts (the arena, or the file mapping)
        uint8_t* vram = nullptr;                    ///< STRATA_X_MTP_VRAM=1: all of them on the last GPU (blob stride), the MoE there
        void* dense = nullptr;                      ///< its dense weights on the last GPU
        std::map<std::string, DT> t;
    };
    std::unique_ptr<Dsv4Mtp> dmtp;
    bool load_draft_mtp(std::string& err);   ///< after load_dspark, before load_experts
    /// MiMo's DFlash drafter (dflash.cpp; EngineOptions::dflash): five dense Qwen3-style blocks on the last GPU, each with
    /// K / V rings (FP16, row p % ring) of its sliding window plus the noise block; the trunk's head (byte-identical to the
    /// drafter's own: checked at load, else its own is uploaded) and the trunk's embedding table (ditto, but for the mask row)
    struct Dflash {
        std::unique_ptr<strata::GgufModel> gm;
        int64_t n = 0;                              ///< blocks
        int block = 0;                              ///< dflash.block_size: the noise block's positions (anchor + masks)
        int64_t H = 0, nkv = 0, hd = 0, n_rot = 0, n_ff = 0, window = 0;
        float rope_base = 10000.0f, eps = 1e-6f, v_scale = 0.0f;
        std::vector<int64_t> tl;                    ///< target layers (n_layer: the trunk's output)
        std::vector<int> tl_stage, tl_col;          ///< per target: the stage that captures it, its column in that stage's buffers
        DT fc, enc_norm, out_norm, output;          ///< output: its own head (only when the trunk's differs)
        struct Blk {
            DT attn_norm, wq, wk, wv, wo, q_norm, k_norm, sinks, ffn_norm, gate, up, down;
            uint16_t *kc = nullptr, *vc = nullptr;
        };
        std::vector<Blk> b;
        int64_t ring = 0;                           ///< K / V ring rows: the window, the noise block, a margin
        int cap = 0;                                ///< a prompt chunk's tail rows injected (>= the ring)
        int chunks = 8;                             ///< the noise block's attention key splits
        int32_t mask = -1;
        std::vector<float> mask_emb;                ///< the mask token's row (its token_embd, dequantized)
        const uint8_t* embd = nullptr;              ///< its own token_embd (the anchor's row when the trunk's table differs)
        int embd_type = -1;
        size_t embd_row = 0;
        bool own_embd = false;
        void* dense = nullptr;
        std::map<std::string, DT> t;
    };
    std::unique_ptr<Dflash> dfl;
    /// Hy4-flash's DFly2 drafter (hy4_dfly.cpp; EngineOptions::dfly - moex/tools/hy4/dfly_to_gguf.py's GGUF of the release's
    /// model.dfly_layers): five dense MLA layers on the last GPU, each over a trunk sparse layer's own KV cache and indexer cells
    /// (dfly_target_kv_layers / dfly_target_index_layer: 25, 29, 33, 37, 41), the trunk's head and embedding table
    struct Hy4Dfly {
        std::unique_ptr<strata::GgufModel> gm;
        int n = 0;                         ///< layers
        int block = 6;                     ///< the query block: the bonus token and block - 1 masks (vLLM: num_speculative_tokens + 1)
        int64_t H = 0, n_ff = 0;
        float eps = 1e-5f;
        int32_t mask = -1;
        int topk = 0;
        std::vector<int64_t> tkv, tidx;    ///< per draft layer: the trunk layer whose KV cache / whose cells it reads
        std::vector<int> slot;             ///< per trunk layer: its capture slot (-1: none)
        int n_slots = 0;
        DT f_enorm, f_hnorm, f_gate, f_eh, c_hnorm, c_enorm, c_gate, c_up, c_down, out_norm;
        struct Blk {
            DT attn_norm, q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b, sinks, gate, wo, post_norm, ffn_norm, ffn_gate, ffn_up, ffn_down;
        };
        std::vector<Blk> b;
        std::vector<float> mask_emb;       ///< the mask token's row of the trunk's table
        void* dense = nullptr;
        std::map<std::string, DT> t;
    };
    std::unique_ptr<Hy4Dfly> dfy;
    bool open_dfly(std::string& err);   ///< the GGUF's settings (before the split: its VRAM rides on the last GPU)
    bool load_dfly(std::string& err);   ///< its weights and the last stage's buffers (after alloc_state)
    /// DFlash's cost-aware window (spec.cpp; EngineOptions::spec_policy): how many of the block's drafts a window verifies.
    /// The block's drafts arrive together with their probabilities; a window of n drafts commits E(n) = 1 + q_0 + q_0 q_1 + ...
    /// tokens in expectation (q_j: the measured acceptance of a draft at that depth and probability, its earlier ones accepted -
    /// decayed counts per (depth, probability) cell with the probability as the prior) and costs the measured verify round of
    /// n + 1 rows plus the draft step that follows (EMAs: this machine's, this context's) - n* = argmax E(n) / cost(n).  On this
    /// box a MiMo verify row past the first costs ~0.6 of a token (its experts mostly on the CPU): a draft must be likely to
    /// pay for its row, which a fixed probability gate cannot know.
    struct DflashGate {
        static constexpr int kP = 16, kD = 4;
        double ok[kD][kP] = {}, all[kD][kP] = {};
        double cost[kTmax + 1] = {}, cost_n[kTmax + 1] = {};
        double draft_ms = 0, draft_n = 0;
        double lam = 0.0, tok_ema = 0.0, ms_ema = 0.0;         ///< the long-run rate (tokens per ms): their EMAs' ratio
        int64_t windows = 0, explored = 0, hist[kTmax] = {};   ///< windows chosen, exploration windows, n* histogram
        static int pbin(float p);
        void print() const;   ///< STRATA_X_DFLASH_GATE=1: what it has learned (after each request)
        double q(int depth, float p) const;
        double c(int T) const;
        int choose(const float* p, int nmax);
        void observe(const float* p, int nd, int accepted);
        void observe_cost(int T, double verify_ms, double step_ms);
        void observe_rate(int tokens, double round_ms);   ///< a drafting round: its committed tokens and its time
    };
    DflashGate dgate;
    bool open_dflash(std::string& err);      ///< before the split: its file, geometry and VRAM (choose_split counts it)
    bool load_dflash(std::string& err);      ///< after bind_layers, before load_experts: its weights, rings and buffers
    uint64_t dflash_vram = 0;                ///< its weights and buffers on the last GPU (choose_split)
    uint64_t dfly_vram = 0;                  ///< moex: Hy4-flash's DFly2 drafter's, ditto
    int64_t prompt_end = 0;                  ///< the running prompt() call's read end (DeepSeek-V4's NextN reads its last rows)
    /// The drafter over T rows the trunk just READ at pos0 (a prompt's windows): MTP's catch-up with its draft discarded, or
    /// DSpark's inject (no draft: only the last read is drafted from)
    bool drafter_read(int T, const int32_t* next_toks, int64_t pos0, std::string& err);
    int dsp_k(int64_t il) const;          ///< the feature slot of target layer il, or -1
    int lid_rows = 0;                               ///< the indexer's score stride: the most CSA rows a position sees, + 1
    DT out_hc_fn, out_hc_base, out_hc_scale;        ///< DeepSeek-V4's head: the hyper-connections' reduce
    int64_t choose_split(const std::vector<int>& devs);   ///< Strata's --layer-split auto for two GPUs
    bool load_dense(std::string& err);
    bool bind_layers(std::string& err);
    bool load_experts(std::string& err);
    bool alloc_state(std::string& err);
    bool alloc_state_mimo(std::string& err);
    bool init_cache(std::string& err);
    bool fill_slot(int32_t slot, int64_t il, int64_t e, std::string& err);
    void maintain_cache();
    /// DeepSeek-V4.1's OPT-IN ROUTE BIAS (STRATA_DSV41_ROUTE_BIAS=M, off by default: it changes the output): the decode's
    /// top-k selection score of an expert the SSD would have to read (in no VRAM slot, no RAM copy, no SSD cache slot) is
    /// lowered by M, so such an expert is chosen only when it leads the best resident alternative by more than M.  The
    /// weights stay the release's (the unbiased probs of the chosen experts).  The table is mapped memory the routing kernel
    /// reads; route_pen_update rewrites it between windows.
    float route_margin = 0.0f;
    float* route_pen = nullptr;   ///< [layer][expert], mapped
    /// moex (DSV41_PLAN §33), opt-in STRATA_DSV41_PEN_DEVICE=1: each stage's copy of the table in VRAM, which its decode
    /// graph's first node pulls from the mapped table (the stage's rows; the host writes the table only between windows):
    /// the routing kernel read the mapped table over PCIe, 384 floats in 12 dependent loads - 18 us of a decode layer, 13 from
    /// VRAM - and reads the same values here.  (A copy-engine refresh queued behind the tier's swap copies and held the next
    /// window up.)  [stage] (null: the mapped table)
    std::vector<float*> route_pen_dev;
    bool route_bias_init(std::string& err);
    void route_pen_update();
    const float* route_pen_for(int64_t il) const { return route_pen ? route_pen + il * hp.n_expert : nullptr; }
    /// ... a decode graph's: the stage's VRAM copy where there is one
    const float* route_pen_win(int64_t il) const {
        if (!route_pen) return nullptr;
        if (!route_pen_dev.empty() && il < (int64_t) ldev.size() && route_pen_dev[(size_t) ldev[(size_t) il]])
            return route_pen_dev[(size_t) ldev[(size_t) il]] + il * hp.n_expert;
        return route_pen + il * hp.n_expert;
    }
    /// ... and in the prompt path too (STRATA_DSV41_ROUTE_BIAS_PROMPT=1, with STRATA_DSV41_ROUTE_BIAS=M set; off by default):
    /// the prompt's routing is biased away from the SSD the same way, so a chunk reads fewer records (it changes how the
    /// prompt is read, not only the decode)
    bool route_bias_prompt = false;
    const float* prompt_pen_for(int64_t il) const { return route_bias_prompt ? route_pen_for(il) : nullptr; }
    /// A prompt is being read (run_request, up to its decode): the table holds zeros, so the prompt tokens a short read or
    /// a tail sends through decode windows route as the release does - the bias is the decode's alone (the audit of
    /// 2026-10-19 found those tokens' K/V written from biased routing, 54% of a 174K-token context)
    bool pen_prompt = false;
    // the adaptive tier's copies in flight (engine.cpp swap_round / finalize_swaps)
    struct PendingSwap {
        int32_t slot, key;              ///< the slot and the expert coming in
        int di;
        cudaEvent_t ev;
        int32_t key_out = -1;           ///< phase 0: the victim being copied out to its exchange blob first
        int64_t xb = -1;                ///< that exchange blob
        uint64_t xb_cap = 0;
        int phase = 1;                  ///< 0: the victim's copy out is in flight; 1: the incoming copy is
        int64_t ret = -1;               ///< the incoming expert's RAM blob, returned to the pool once it is in VRAM
        uint64_t ret_cap = 0;
        int cslot = -1;                 ///< DeepSeek-V4.1: the SSD cache slot the incoming copy reads (held until it lands)
        int vq = -1;                    ///< moex (#143): its check queued behind the incoming copy (a vq_* area), or -1
        int vslot = -1;                 ///< moex (§44, STRATA_DSV41_SWAP_CACHE): phase 0's victim copied into this cache slot
    };
    std::vector<PendingSwap> pending;
    std::vector<char> slot_pending;     ///< [slot]: a swap's copy is in flight into it
    std::vector<char> key_pending;      ///< [layer * n_expert + e]: that expert is being copied in
    int64_t windows_seen = 0;
    void finalize_swaps(bool wait);
    void swap_round();
    // serving: the live sequence, the checkpoints, the sampler's buffers
    std::vector<int32_t> live;                       ///< the tokens whose state the engine holds
    std::vector<ImageKey> live_imgs;                 ///< moex: the images inside `live`
    bool live_valid = true;                          ///< false after an error: the next request starts from zero
    /// moex vision: the running request's images (GENI), sorted by start; empty outside run_request.  The rows are the
    /// caller's (serve.cpp keeps them until the request ends).
    std::vector<ImageSpan> img;
    bool first_top_pending = false;                  ///< STRATA_X_FIRST_TOP: the request's first sample is still to come
    void first_top_probe(const float* row, int64_t counter_pos);   ///< session.cpp
    /// The encoder row for position `pos`, or nullptr (no image there).
    const float* image_row(int64_t pos) const {
        if (img.empty() || pos < 0) return nullptr;
        auto it = std::upper_bound(img.begin(), img.end(), pos, [](int64_t p, const ImageSpan& s) { return p < s.start; });
        if (it == img.begin()) return nullptr;
        --it;
        return pos < it->start + it->n ? it->rows + (size_t) (pos - it->start) * (size_t) hp.n_embd : nullptr;
    }
    std::vector<std::unique_ptr<Checkpoint>> checks;
    std::vector<int32_t> turn_tokens;
    int ck_keep = 6;
    int64_t ck_every = 16384, ck_root = 2048;
    int ck_deep = 4;                    ///< moex: deep checkpoints kept (0: every ck_every fresh tokens, in the LRU, as before)
    uint64_t ck_clock = 0;
    PromptCtl* pctl = nullptr;
    /// moex (DSV41_BACKLOG #78), STRATA_DSV41_STOP_LAYERS=1 (DeepSeek-V4.1, two or more stages): a stop is looked for between
    /// a chunk's layers too, not only between chunks - each stage keeps its running state as it was at the start of the
    /// chunks the last stage has not finished, and after a stop every stage is put back to where the last stage got
    bool stop_layers = false;
    /// moex (DSV41_BACKLOG #76): Impl::snapshot reads stage snap_src_si's state from this device copy (a layer-major window's
    /// state at a chunk end, put together layer by layer) instead of its live buffers
    const uint8_t* snap_src = nullptr;
    int snap_src_si = -1;
    /// a stop asked (the serving layer's hook, or the progress hook's earlier answer): any stage's thread may ask
    bool stop_asked() const { return pctl && (pctl->stopped.load(std::memory_order_acquire) || (pctl->stop && pctl->stop())); }
    /// DeepSeek-V4.1's BOUNDED REPLAY (DSV41_PLAN §1, §5 E; Xolotl's ds41_generate.cpp): the layers after the last KV source
    /// keep no state but their 128-row raw rings and read that source's compressed rows and index keys, so a prompt runs
    /// them over the 128 x (layers from the source on) rows before each place a later read starts from - the prompt's end
    /// and each checkpoint cut - and their rings there are what the full pass leaves.  replay_layer: the first such layer
    /// (-1: off - V4, STRATA_DSV41_REPLAY=0, or a split that puts it on any stage but the last); replay_tail its rows.
    int64_t replay_layer = -1, replay_tail = 0;
    std::vector<int64_t> replay_ends;   ///< this prompt call's read-start positions (its end, its cuts), ascending
    void replay_init();
    int32_t* samp_hist = nullptr;                    ///< device: the penalty history (last stage)
    int32_t* samp_out = nullptr;                     ///< device: the sampled ids
    int32_t* h_hist = nullptr;                       ///< pinned
    int samp_hist_cap = 4096;
    // moex (DSV41_PLAN §57): the parked conversations (park.cpp; opt.conv_cache_mib > 0, DeepSeek-V4 / V4.1)
    std::vector<std::unique_ptr<Parked>> parked;
    uint64_t park_clock = 0;
    int64_t park_evictions = 0;
    int64_t live_turn_pos = 0;                       ///< where the live conversation's last prompt's final turn began
    bool park_on() const { return opt.conv_cache_mib > 0 && hp.arch == Arch::dsv4; }
    void park_note_turn(const std::vector<int32_t>& prompt);   ///< live_turn_pos from a request's prompt
    /// A parked conversation this prompt continues 1,024+ tokens further than `cur` (what the live state and the checkpoints
    /// give): the live one parked by the leave rule, that one put back.  true: restored (the caller finds its reuse again).
    bool park_try(const std::vector<int32_t>& prompt, const std::vector<ImageKey>& keys, int64_t n, int64_t cur);
    /// The live conversation parked when this prompt leaves it (park::worth_parking); `keep`: a record not to evict.
    void park_leaving(const std::vector<int32_t>& prompt, const std::vector<ImageKey>& keys, const Parked* keep = nullptr);
    bool park_live(const std::vector<int32_t>& prompt, const std::vector<ImageKey>& keys, const Parked* keep);
    void unpark(size_t i);
    uint64_t parked_bytes() const;
    // moex (DSV41_BACKLOG #53, the restart half; park_disk.cpp, opt.conv_dir): conversations saved by an earlier engine
    struct DiskCk {
        int64_t len = 0;
        uint64_t h[2] = {0, 0};             ///< the hashes of its tokens (tok_hashes)
        std::vector<ImageKey> imgs;
        bool root = false, deep = false;
        std::vector<uint64_t> bytes;        ///< per stage
    };
    struct DiskRec {
        std::string path;
        int64_t len = 0, turn_pos = 0;
        uint64_t h[2] = {0, 0};
        std::vector<ImageKey> imgs;
        std::vector<uint64_t> row_bytes, snap_bytes;   ///< per stage
        std::vector<DiskCk> checks;         ///< its checkpoints, all prefixes of its tokens, ascending
        uint64_t data_off = 0, data_bytes = 0, file_bytes = 0;
        int64_t mtime = 0;
        bool used = false;                  ///< put back in this run: its file is superseded at the next save
    };
    std::vector<DiskRec> disk;
    bool disk_indexed = false;
    void disk_index();
    /// A saved conversation this prompt continues 1,024+ tokens further than `cur` and than the parked ones: read into
    /// `parked`, for park_try to put back.
    void disk_offer(const std::vector<int32_t>& prompt, const std::vector<ImageKey>& keys, int64_t n, int64_t cur);
    bool disk_save(std::string& err);
    void snapshot(Checkpoint& c, int stage);         ///< one stage's running state into c.blob[stage]
    void restore(const Checkpoint& c);
    void reset_state();                              ///< zero the recurrent state and the rings (the positional caches are rewritten)
    int32_t sample_last(const SamplingParams& sp, int64_t counter_pos, const int32_t* hist, int hist_n, std::string& err);
    /// Every row of the last window (T rows): the token after each, by arg-max (greedy, no penalties) or Strata's sampler -
    /// Philox keyed by the position of the row's fed token (counter_pos + t), the penalty history row t = the tail, then
    /// win[0..t] (kn::penalty_rows).  `hist` ends with win[0].
    bool sample_rows(const SamplingParams& sp, int64_t counter_pos, const int32_t* win, int T, const int32_t* hist, int hist_n,
                     int32_t* out, std::string& err);
    /// THE DECODE LOOP (spec.cpp): Strata's serial window loop.  The drafter's chain (a token and its probability per step),
    /// gated by spec-min-p, makes the window [x, drafts...]; the trunk verifies it in one pass; the longest prefix of drafts
    /// the trunk's own samples agree with is accepted, plus the trunk's token after it; the KDA state rolls back to the last
    /// kept row and the drafter catches up over the kept rows (its last row is the next window's first draft).
    struct SpecRun {
        const SamplingParams* sp = nullptr;
        const std::vector<int32_t>* eos = nullptr;
        int max_new = 0;
        int64_t pos = 0;                       ///< the position of x
        int32_t x = 0;                         ///< the token fed first (the last prompt token)
        std::vector<int32_t>* hist = nullptr;  ///< the penalty history's tail, ending with x; extended with every emitted token
        std::function<bool(int32_t)> token;    ///< an emitted token; false: stop (the request is cancelled)
        std::function<bool()> stop;            ///< polled between windows
        bool plain = false;                    ///< no drafting (a request or a test asks for the plain loop)
        const int32_t* ctx = nullptr;          ///< the whole prompt (the suffix drafter's history), ctx_n tokens
        int64_t ctx_n = 0;
        // out
        int64_t produced = 0, offered = 0, accepted = 0, windows = 0;
        int64_t offered_at[kTmax] = {0}, accepted_at[kTmax] = {0};   ///< per draft depth (MiMo: per MTP block)
        bool eos_hit = false, cancelled = false;
        bool nonfinite = false;                ///< #879: a window's logits held a NaN or an infinity (its tokens dropped)
        double draft_ms = 0;
        int64_t lookup_windows = 0, lookup_accepted = 0;   ///< prompt lookup: windows it supplied, their drafts kept
    };
    bool spec_decode(SpecRun& r, std::string& err);
    // Strata's prompt lookup (EngineOptions::suffix_draft): the drafter over the request's context and the policy that
    // learns, over the whole process, which window pays (strata/spec/suffix_drafter.hpp, draft_policy.hpp)
    std::unique_ptr<strata::spec::SuffixDrafter> sfx;
    std::unique_ptr<strata::spec::DraftPolicy> policy;
    int lookup_S = 0;                   ///< the longest lookup window (0: no lookup)
    bool run_window(const int32_t* tokens, int T, int64_t pos0, bool keep, std::string& err);   ///< forward + the tier's upkeep
    void rollback_to(int n_keep);
    void warm_decode();                                                                          ///< capture the graphs at load
    bool res_dirty = true;              ///< the residency table changed since the stages' device copies were made
    bool graphs_on = true;              ///< decode windows through captured graphs (STRATA_X_NO_GRAPHS=1 / --no-graphs: eager)
    void sync_residency();
    void push_static_res();             ///< the NextN layer's slice of the device table, once (its experts are pinned)
    bool forward_graph(const int32_t* toks, int T, int64_t pos0, bool keep, std::string& err);
    bool handoff(Stage& from, Stage& to, int T);
    bool forward(const int32_t* toks, int T, int64_t pos0, bool keep, std::string& err);
    /// Strata's prompt path over toks[0, n) at positions pos0..; `n_avail` (>= n) tokens are readable at `toks`, so
    /// with MTP the NextN block's caches are filled for every position whose next token is among them.
    bool prompt(const int32_t* toks, int n, int64_t pos0, int n_avail, std::string& err);   ///< prompt.cpp
    int choose_chunk(int n, int cap = 0) const;   ///< prompt.cpp; cap: the largest chunk to consider (0: the auto / opt cap)
};

}  // namespace strata::x
