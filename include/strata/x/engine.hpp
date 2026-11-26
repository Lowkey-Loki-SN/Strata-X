// include/strata/x/engine.hpp - moex: strata-x's engine (moex/README.md), first written as GLM-5.3-Flash (glm5next)
// decode on Strata's machinery; DeepSeek-V4 / V4.1, MiMo-V2.6, Hy4-flash and ERNIE-5.5-Flash run on it since.  Version 1:
//
// WHAT VERSION 1 IS: the correct engine, not yet the fast one.  The dense weights (KDA, DSA/MLA, mHC, the shared
// expert, the dense FFN, the head) run on ONE GPU from the GGUF's own blocks (native MMVQ + kernels.cu); the 288
// routed experts per layer run on the CPU in Strata's ExpertPool (ggml-cpu arithmetic, the SwiGLU clamp), from a RAM
// arena laid out as Strata's native expert blobs [gate | up | down].  One token at a time; the host waits for each
// MoE layer's router, runs the pool while the GPU computes the shared expert, then uploads the parts.
//
// WHAT COMES NEXT (moex/docs/MOEX.md, "GLM bring-up"): the VRAM expert tier (ExpertCache + GPU experts for the
// hits), the doorbell overlap and per-layer CUDA graphs, then MTP and a batched prefill.  Each lands behind this
// interface, and each is checked the way this one is: `set_recorder` with an oracle attached forces llama.cpp's
// values at every probe point, so every op is compared on llama.cpp's own inputs (moex/ref/recorder.hpp).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace moex {
class Recorder;
}

namespace strata::x {

/// The model families the engine runs: GLM-5.3-Flash (glm5next / glm5-next: KDA + DSA, mHC x 4, a shared expert, one NextN
/// block), MiMo-V2.6-Flash (mimo2: GQA attention, sliding-window or full per layer, attention sinks, no mHC, no shared
/// expert, three MTP blocks with dense FFNs), DeepSeek-V4-Flash (deepseek4: mHC x 4; one shared K = V head attended over a
/// raw 128-position window plus, per layer, no compressed rows, CSA rows chosen by the lightning indexer, or HCA rows;
/// attention sinks, a grouped low-rank output, a shared expert, sqrt-softplus routing with the first layers hash-routed by
/// token; no MTP block - moex/docs/DSV4_SPEC.md), Hy4-flash (hy_v4_flash: iHC x 4 - mHC with the identity for its stream
/// mixing; MLA on every layer, sparse layers over the k-pool indexer's groups and sliding-window layers, both with sinks, a
/// gated output and a post-norm; a latent MoE with a shared expert; Engram - moex/docs/HY4_PORT.md).
enum class Arch : int { glm5 = 0, mimo2 = 1, dsv4 = 2, hy4 = 3, ernie = 4 };

struct Hparams {
    Arch arch = Arch::glm5;
    int64_t n_embd = 0, n_layer = 0, n_layer_all = 0, n_vocab = 0;
    int64_t n_head = 0, hd_kda = 0, d_conv = 0, n_dense = 0, n_ff = 0;
    int64_t n_expert = 0, n_used = 0, n_ff_exp = 0;
    int64_t q_lora = 0, kv_lora = 0, k_mla = 0, v_mla = 0;
    int64_t idx_heads = 0, idx_dim = 0, idx_topk = 0, kpool = 0;
    bool select_tail = true;
    int64_t eos_id = -1;   ///< tokenizer.ggml.eos_token_id
    int64_t hc = 0, sinkhorn_iters = 0;
    float hc_eps = 0, rms_eps = 0, ln_eps = 0, kda_lb = 0, ew_scale = 1;
    bool ew_norm = false;
    std::vector<int64_t> head_kv;   ///< GLM: nonzero on the DSA layers; MiMo: the layer's KV heads
    std::vector<float> clamp_exp, clamp_sh;
    // MiMo (mimo2): GQA with K heads of hd_k and V heads of hd_v, NEOX RoPE on the first n_rot dims (base rope_base, or
    // rope_base_swa on the sliding-window layers), the window n_swa, the attention output scaled by v_scale
    int64_t hd_k = 0, hd_v = 0, n_rot = 0, n_swa = 0;
    float rope_base = 0, rope_base_swa = 0, v_scale = 0;
    std::vector<char> swa_l;        ///< per layer (incl. the MTP blocks): sliding-window attention
    std::vector<char> moe_l;        ///< per layer: routed experts (empty: every layer from n_dense)
    bool shexp = true;              ///< a shared expert beside the routed ones
    // DeepSeek-V4 (deepseek4): per layer the KV compression ratio (0: the raw window only; 4: CSA + the lightning indexer;
    // 128: HCA); the first n_hash layers route by token (ffn_gate_tid2eid); the attention output in o_groups low-rank groups
    // of o_lora rows; RoPE on the last n_rot dims of the hd_k (= hd_v) head - plain at rope_base on the ratio-0 layers, YaRN
    // on the compressed ones (crope_base, rope_scale = 1 / the scaling factor, n_ctx_orig, beta_fast / beta_slow)
    std::vector<int> cratio;
    int64_t n_hash = 0, o_groups = 0, o_lora = 0, n_ctx_orig = 0;
    float crope_base = 0, rope_scale = 1, beta_fast = 32, beta_slow = 1;
    // DeepSeek-V4.1 (deepseek41: arch dsv4 with v41 set; moex/docs/DSV41_PORT.md): compression ratios 0 / 2 / 1; the
    // compressed rows and the index keys computed only by the KV sources and shared by the layers after each (kv_src[il]:
    // the layer whose rows layer il attends, -1 for none); the top-k list computed only by the index sources and reused by
    // the layers after each (idx_src[il]); layer cand_src's indexer keeps the cand_blocks best blocks of cand_bsize rows and
    // the later index sources score only inside them; engram n-gram rows added before its layers; the routed experts in
    // per-layer record files of a pack directory; the head collapses the streams with the last layer's FFN-side pre-mix.
    bool v41 = false;
    std::vector<int> kv_src, idx_src;
    int cand_src = -1, cand_blocks = 0, cand_bsize = 0;
    bool kv_owner(int64_t il) const { return v41 && kv_src[(size_t) il] == (int) il; }
    bool idx_owner(int64_t il) const { return v41 && idx_src[(size_t) il] == (int) il; }
    bool engram = false;                       ///< the GGUF carries the engram projections (and the pack its tables)
    std::vector<int> eg_layers;                ///< the engram layers (1, 14)
    std::vector<int64_t> eg_rows;              ///< their tables' rows
    int64_t eg_ngram = 0, eg_heads = 0, eg_dim = 0, eg_ncomp = 0;
    int32_t eg_pad = 0;                        ///< the compressed pad id
    std::vector<int64_t> eg_mul, eg_primes, eg_offsets;   ///< the hash inputs (moex/tools/dsv41/engram_hash.hpp)
    std::vector<int32_t> eg_token_map;
    std::vector<std::string> eg_wfile, eg_sfile;          ///< the tables' rows and scales, relative to the pack
    std::vector<int> rec_type;                 ///< per layer: the routed experts' ggml type in the record file (gate / up)
    std::vector<int> rec_dtype;                ///< ... and their down matrices' (rec_type unless experts.tsv names "gu,d")
    std::vector<uint64_t> rec_stride;          ///< per layer: bytes per record (expert e at e * stride)
    std::vector<std::string> rec_file;         ///< per layer: the record file, relative to the pack
    /// per layer: its routed experts and how many a token takes, where they differ from n_expert / n_used (DeepSeek-V4.1's
    /// DSpark blocks: 128, top-3); empty or 0: the model's
    std::vector<int> l_expert, l_used;
    // Hy4-flash (hy_v4_flash, moex/docs/HY4_PORT.md): the routed experts' width (the latent, 1536; 0: n_embd), each layer's
    // query heads (32 on the sparse layers, 64 on the sliding-window ones; n_head is the largest), iHC's post magnitude; the
    // sparse layers' RoPE base rope_base, the sliding-window ones' rope_base_swa, the window n_swa, the RoPE width n_rot
    int64_t n_ex = 0;
    std::vector<int64_t> head_l;
    float hc_mag = 0;
    int64_t ex() const { return n_ex > 0 ? n_ex : n_embd; }   ///< the routed experts' input and output width
    int64_t H_of(int64_t il) const { return il < (int64_t) head_l.size() ? head_l[(size_t) il] : n_head; }
    int64_t E_of(int64_t il) const { return il < (int64_t) l_expert.size() && l_expert[(size_t) il] > 0 ? l_expert[(size_t) il] : n_expert; }
    int64_t k_of(int64_t il) const { return il < (int64_t) l_used.size() && l_used[(size_t) il] > 0 ? l_used[(size_t) il] : n_used; }
    bool is_dsa(int64_t il) const { return arch == Arch::glm5 && head_kv[(size_t) il] != 0; }
    // ERNIE-5.5-Flash (ernie55, moex/docs/ERNIE55_PLAN.md; ernie.cpp): cratio 128 = an HCA layer (DeepSeek-V4's: n_head x
    // hd_k, n_rot, n_swa, o_groups x o_lora, YaRN at crope_base), <= 0 = an MLA + DSA layer (Hy4's MLA: kv_lora, k_mla = 192 +
    // n_rot, v_mla; plain RoPE at rope_base; V4's indexer, idx_heads x idx_dim keeping idx_topk positions before the window);
    // the routed experts in the n_ex latent; clamp_exp / clamp_sh hold kSituGlu (SiTU-GLU on every expert path)
    bool ernie_dsa(int64_t il) const { return arch == Arch::ernie && il < (int64_t) cratio.size() && cratio[(size_t) il] <= 0; }
    bool is_moe(int64_t il) const { return moe_l.empty() ? il >= n_dense : moe_l[(size_t) il] != 0; }
    bool is_swa(int64_t il) const { return !swa_l.empty() && swa_l[(size_t) il] != 0; }
    int64_t n_kv(int64_t il) const { return head_kv[(size_t) il]; }
    /// MiMo's fused QKV row: [q (n_head x hd_k) | k (n_kv x hd_k) | v (n_kv x hd_v)]
    int64_t qkv_width(int64_t il) const { return n_head * hd_k + n_kv(il) * (hd_k + hd_v); }
};

struct EngineOptions {
    int device = 0;      ///< with `devices` empty and `split` 0: the one GPU
    /// THE LAYER SPLIT: the GPUs, in order (empty: every visible GPU).  The first `split` trunk layers run on the
    /// first, the rest - and the head and the NextN block - on the last (two GPUs; -1: half each).  Each GPU fills
    /// what is left of its VRAM with expert slots for its own layers; only the 4 x n_embd residual crosses.
    std::vector<int> devices;
    int split = -1;
    int64_t vram_reserve_later_mib = 2048;   ///< what a later GPU keeps free (the display's, on this box)
    int64_t n_ctx = 4096;
    int threads = 0;     ///< CPU expert workers; 0 = the pool's default (every physical core but the first)
    /// DeepSeek-V4.1 (deepseek41): the pack directory - the routed experts' record files and the engram tables, at the
    /// relative paths the GGUF's deepseek41.expert_record.file / engram.*_file name.
    std::string pack;
    /// moex (the GGUF set, gguf_convert.py): DeepSeek-V4.1's routed experts from a GGUF set (any of its files) instead of the
    /// pack's record files - each expert a "virtual record": its gate, up and down slices read O_DIRECT into a record's
    /// places (dsv41.cpp load_experts_gguf); every tier works on them as on records
    std::string experts_gguf;
    /// ... its SSD cache: the decode's records read from the SSD stay in pinned RAM, least recently used out first, and the
    /// prompt path and the VRAM fills take them from there.  GiB, taken before the RAM copy is planned, so it comes out of
    /// the RAM copy's budget; 0 = 64 slots (a window's misses and the read-ahead).
    double ssd_cache_gib = 0;
    /// moex (DSV41_PLAN §57), --serve: PARKED CONVERSATIONS (DeepSeek-V4 / V4.1) - a request that leaves the live conversation
    /// copies its positional rows, its running state and its checkpoints to RAM, and a request that continues a parked one
    /// further than the live state puts it back.  conv_cache_mib: the budget (0 = off: the engine as before); at most
    /// conv_cache_slots records; only conversations of conv_cache_min_tokens or more; never below conv_cache_min_free_mib
    /// of MemAvailable after the copy.
    int64_t conv_cache_mib = 0;
    int conv_cache_slots = 4;
    int64_t conv_cache_min_tokens = 8192;
    int64_t conv_cache_min_free_mib = 8192;
    /// moex (DSV41_BACKLOG #53, the restart half), --conversation-dir: the live and parked conversations are written to this
    /// directory when the engine ends cleanly (QUIT, the end of its input, SIGINT / SIGTERM in --serve), and a later engine
    /// started the same way - the same binary, arguments, STRATA_* environment and model files (`conv_identity`) - puts one
    /// back when a prompt continues it, through the parked conversations' restore: the same bytes in the same buffers.
    /// The files hold the state and hashes of the tokens, not the tokens.  At most conv_dir_mib MiB (newest first); a week's
    /// files and other engines' are dropped.  Empty: off (needs conv_cache_mib > 0).
    std::string conv_dir;
    int64_t conv_dir_mib = 8192;
    std::string conv_identity;
    bool pin = true;     ///< pin the workers to physical cores
    bool verbose = true;
    /// THE VRAM EXPERT TIER (Strata's mechanism): routed experts held in GPU slots are computed there, the rest by
    /// the CPU pool, concurrently.  -1 = as many slots as fit once `vram_reserve_mib` stays free; 0 = none.
    int64_t expert_slots = -1;
    int64_t vram_reserve_mib = 700;   ///< what the first GPU keeps free (Strata's --vram-reserve-mib, the live install's 700)
    /// "<layer> <expert> <count>" lines, hottest first (moex/tools/glm_profile.py): the slots are filled from the
    /// top at load.  Empty: the slots fill as experts are first routed.
    std::string expert_profile;
    /// THE ADAPTIVE TIER (Strata's, STRATA_QWEN_DESIGN.md 2.3): every `adapt_every` windows, per layer the non-resident
    /// experts used at least 2.0 (decayed counts) pair hottest-first with the layer's coldest residents while the gain is
    /// at least 1.5; the `adapt_swaps` best by gain are taken, then every count is multiplied by `adapt_decay`.  A victim
    /// leaves at once (the CPU computes it meanwhile); the incoming expert is copied on a side stream and becomes
    /// resident once its copy has landed.  Strata's flags: --adapt-every 4 --adapt-swaps 96 --adapt-decay 0.7.
    bool adaptive = true;
    int adapt_every = 4;
    /// 0 = a byte budget: 256 MiB of copies per round (Strata's 96 swaps of its 1.4 MB blobs would be 130 MB; measured here,
    /// 24 / 32 / 96 swaps per 4 windows = 24.4 / 25.7 / more-interference T/s with 7.9 MiB blobs: the swap DMAs compete
    /// with the doorbell's small mapped reads), clamped to 8..96 swaps.
    int adapt_swaps = 0;
    float adapt_decay = 0.7f;
    /// The routed experts' RAM copy (tiers.cpp, Strata's resident RAM mode): -1 = the cache complement plus the lend
    /// region within MemAvailable minus resident_headroom_gib (the default); 1 = the same with Strata's --resident-experts
    /// headroom of 4 GiB; 0 = no RAM copy, every expert the VRAM tier does not hold is read in place from the GGUF.
    int experts_in_ram = -1;
    /// Strata's resident RAM mode (expert_source.cpp pin_cache_complement): the pinned RAM copy holds the experts the VRAM tier
    /// does not, then the prompt path's lend region, within MemAvailable minus this headroom; what does not fit is read from
    /// the GGUF.  Strata leaves 4-8 GiB on a dedicated machine; on a desktop 17 GiB free froze the session twice (2026-10-07),
    /// so 28 - about what the Qwen install leaves here (29 GB available).
    double resident_headroom_gib = 28.0;
    /// Blobs kept free for the adaptive tier's exchanges (Strata's reserve_exchanges: min(adapt_swaps, 96)).
    int exchange_blobs = 96;
    /// Transparent huge pages for the RAM copy (Strata asks for them with MADV_HUGEPAGE).  Off here: on this desktop the
    /// kernel's compaction for 85 GiB of them in ~9 s froze the session (2026-10-07), and Strata's #771 records the same
    /// failure (a 25 s start stretched to 432 s); 4 KB pages registered in 32 s with no memory stall, then filled at 2.5 GiB/s,
    /// and decoded within noise (28.5 vs 29.5 T/s).  --arena-thp asks for them.
    bool arena_thp = false;
    /// Longest MTP verify window to support (1 + drafts, <= 8): sizes the KDA rollback snapshots (~136 MiB per
    /// token on GLM-5.3-Flash).  0: no MTP.
    int mtp_window = 0;
    /// Strata's --spec-min-p: draft j enters a verify window only while drafts 0..j each have a draft probability of at
    /// least this (the drafter stops launching chain steps once one falls below); a request's `spec_min_p` key overrides it.
    float spec_min_p = 0.5f;
    /// Strata's COUPLED DRAFT SAMPLING (include/strata/core/coupled_draft.hpp, --coupled-draft / STRATA_SPEC_COUPLED=1):
    /// under sampling the MTP drafter SAMPLES its draft with the request's own chain and the uniform the verify row will
    /// draw (Philox(seed, cell + 1)) instead of proposing its argmax.  Verification stays exact-match against the target's
    /// sample, so the output is unchanged; drafts agree with it far more often when the draft's distribution is close.
    /// Requests with repetition penalties keep argmax drafts (the penalty history is not mirrored).
    bool coupled_draft = false;
    /// Strata's PROMPT LOOKUP (--suffix-draft N, on by default there; STRATA_QWEN_DESIGN.md 3.6): where the text being
    /// written repeats an earlier stretch of the context (prompt + output), the suffix drafter proposes what followed it,
    /// and when the MTP's own first draft agrees (or there is no MTP) DraftPolicy takes that window over the MTP's if its
    /// expected tokens per ms - learned online from the acceptance by match length and the measured cost of each window
    /// size - is 3% better.  N: the shortest match (Strata's 3); 0 = off.  Drafts only change the speed, never the output.
    int suffix_draft = 3;
    /// The longest lookup window (1 + drafts): 0 = Strata's rule, the MTP window + 2 (6 without MTP), at most 8; GLM-5.3's
    /// windows stay within its KDA snapshot rows (the MTP window; no lookup without MTP).
    int lookup_window = 0;
    /// moex: Hy4-flash's DFly2 drafter, a GGUF of moex/tools/hy4/dfly_to_gguf.py (--draft-dfly FILE; drafts with --spec N)
    std::string dfly;
    /// moex vision: requests may carry pictures (the server's --image-pad).  Hy4-flash then keeps a device M-RoPE table a stage
    /// (hy4_kernels.hpp Hy4MPos) its rotations read; without it the positions are the token indices, as always
    bool images = false;
    /// THE PROMPT PATH (Strata's prefill, STRATA_QWEN_DESIGN.md 4): the prompt in chunks; per layer the routings
    /// sorted by expert, the routed experts as MMQ products in groups of 16; from 1024 tokens EVERY non-resident
    /// expert streams in a fixed order through a ring, issued by its own thread; the chunk's buffers and the ring
    /// live in expert slots the stage LENDS for the prompt (refilled after); the stages pipelined over the chunks.
    /// 0 = Strata's --prefill auto: the largest chunk up to 8192 (a 256 grid) whose buffers and ring fit in the
    /// lendable slots of every stage; > 0: that chunk; 8 or less: the prompt in decode windows (no MMQ).
    int prefill_chunk = 0;
    /// The ring's slots; 0 = Strata's byte budget (384 Q2_0 blobs of 1,382,400 B, as slots of this model's blob).
    int prefill_ring = 0;
    /// moex (ERNIE55_PLAN, the hill-climb): the full ring (every expert pinned in RAM) as a fraction of a layer's experts, in
    /// whole groups; 0 = 0.55, GLM-5.3-Flash's measured best.  Unlike prefill_ring it keeps the short reads' smaller rings.
    double prefill_ring_frac = 0.0;
    /// The mixers' (KDA / DSA / the dense FFN / the shared expert) sub-batch inside a chunk: their buffers grow with
    /// it, the MoE's with the chunk.
    int prefill_sub = 512;
    /// Strata's --short-read: a prompt segment of at most this many tokens goes through decode windows instead.  Strata's 64
    /// fits its small experts; GLM's 7.9 MiB experts make one prompt-path chunk cost ~12 s whatever its size (every routed,
    /// non-resident expert streams), so windows (~50 T/s) win up to ~600 tokens - measured 2026-10-07: 128 / 256 / 512 / 1024 /
    /// 2048 tokens read at 35 / 45 / 52 / 51 / 51 T/s through windows, 10 / 22 / 42 / 83 / 160 through the prompt path.
    /// moex 10-08: 0 - every prompt segment reads as one batch through the prompt path (chunked past a chunk), like Strata's
    /// Qwen engine and mainline: the user found 8-token decode windows for every small turn slow at depth (~40 ms a token at
    /// 100K) and noisy; a small segment takes the CPU-assisted read (cpu_read) instead of the stream's fixed cost.
    int64_t short_read = 0;
    /// moex: a prompt segment longer than short_read and at most this long reads through the prompt path WITHOUT the stream:
    /// the experts the VRAM tier holds run as MMQ groups on the GPU, every other routed expert on the CPU pool - once over all
    /// of its tokens (the multi-token kernels in groups of 8) - at the same time.  Past it the stream's fixed cost (every
    /// non-resident expert over PCIe, ~11 s a chunk on this box) wins.  0: off (also STRATA_X_CPU_READ=N).  -1 (default): auto -
    /// the engine learns both paths' costs on this box and model (Engine::Impl::cpu_read_limit).
    int64_t cpu_read = -1;
    /// moex (Strata 0.1.41's STRATA_PREFILL_CPU_SHARE): a prompt segment of at most this many tokens reads with the CPU pool and
    /// the stream SHARING the routed experts the VRAM tier does not hold - per layer the routing first, then the experts few
    /// tokens route to on the CPU pool from RAM and the others streamed over PCIe (only routed ones), split where both sides
    /// end together by their measured speeds (Stage::moe_prompt_share).  It replaces the auto cpu_read.  0: off (the cpu_read
    /// rules as before; also STRATA_X_SHARE=0).  -1 (default): auto - up to where it costs what a streamed chunk costs, both
    /// learned on this box (Engine::Impl::share_limit).
    int64_t prefill_share = -1;
    /// In-place mode only: pin (cudaHostRegister, read-only) the experts' pages of the GGUF mapping so the prompt
    /// path DMAs them (slow to register: the page cache must be read in first).
    bool register_experts = false;   ///< unused since the resident RAM mode (the RAM copy is registered; the file tier is not)
    /// Decode windows through captured CUDA graphs + the doorbell (Strata's Verifier structure; decode.cpp).  Off:
    /// the eager per-layer launches (also STRATA_X_NO_GRAPHS=1).  The probe mode is always eager.
    bool graphs = true;
    /// Capture every decode graph (the trunk's windows, the drafter's steps) before the first request.
    bool warmup = true;
    /// DeepSeek-V4: the DSpark drafter (llama.cpp's --spec-type draft-dspark -md FILE; dspark.cpp) - a GGUF of arch
    /// dflash holding three DSV4 blocks over the trunk's hidden features; with --spec N it drafts up to min(N - 1, its block
    /// size) tokens per window.  Empty: no drafter (DeepSeek-V4 has no MTP block).
    std::string dspark;
    /// The token id DSpark's block positions after the first carry (its "mask"); -1: llama.cpp's (llama_vocab_mask of the
    /// drafter's vocabulary).
    int dspark_mask = -1;
    /// DeepSeek-V4: the NextN head from a GGUF of its own (llama.cpp's -md mtp-*.gguf --spec-type draft-mtp; dsv4_mtp.cpp)
    /// - the original DeepSeek-V4-Flash's single MTP block (convert_hf_to_gguf.py --mtp), chained like GLM's NextN; with --mtp /
    /// --spec N.  Empty: none.
    std::string draft_mtp;
    /// MiMo-V2: the DFlash block drafter (llama.cpp's --spec-type draft-dflash -md FILE; dflash.cpp) - Xiaomi's official
    /// drafter (MiMo-V2.6-Flash-*/dflash: five dense Qwen3-style layers over the trunk's features at five layers, an 8-position
    /// non-causal block per step), converted with mainline's convert_hf_to_gguf.py --target-model-dir.  It replaces the GGUF's
    /// own MTP blocks (not loaded); with --spec N it drafts up to min(N - 1, block - 1) tokens per window.  Empty: none.
    std::string dflash;
    /// DFlash's window: 1 (default) the cost-aware choice (spec.cpp DflashGate), 0 Strata's spec-min-p gate alone.
    int spec_policy = 1;
    /// moex (DSV41_PLAN §54): DeepSeek-V4.1's image encoder on the first GPU - the manifest of the release's ViT and aligner
    /// weights (serve/vision_dsv41_plan.py write_manifest; --vision-weights).  Empty: no encoder (encode_pictures refuses).
    /// Hy4-flash: the mmproj GGUF itself (AngelSlim's mmproj-Hy4-flash-bf16.gguf; strata/x/hy4_vision.hpp)
    std::string vision_manifest;
};

struct EngineStats {
    int64_t tokens = 0;
    double ms_total = 0;     ///< step() wall time
    double ms_pool = 0;      ///< CPU expert pool
    double ms_wait = 0;      ///< host waiting for the GPU's router results
    /// moex (DSV41_PLAN §34): a window's head (the last GPU's last layer and the logits, waited for) and the VRAM tier's
    /// upkeep after it (maintain_cache) - where a token's time outside the routing waits and the pool goes
    double ms_head = 0, ms_upkeep = 0;
    /// ... the upkeep's parts: landed copies (finalize_swaps, the first fills' hashes), first-use fills, the slot check, the swap
    /// rounds (their planning and copies queued), the route bias's table
    double ms_up_final = 0, ms_up_fill = 0, ms_up_verify = 0, ms_up_round = 0, ms_up_pen = 0;
    double ms_up_khash = 0;   ///< ... of which the host's hashes of first-time swapped-in records (issue_incoming)
    int64_t up_khash = 0;     ///< ... how many
    int64_t hits = 0, misses = 0;   ///< routed experts served by the VRAM tier / by the CPU
    double ms_eg = 0;               ///< moex: Hy4-flash's decode windows, the host reading their Engram rows (page faults included)
    uint64_t hy4_eg_hits = 0, hy4_eg_misses = 0;   ///< moex: Hy4-flash's Engram rows from its row cache / out of the table
    /// DeepSeek-V4.1's SSD tier in the decode: the CPU's experts from RAM, those read from the SSD first, the time the decode
    /// waited for those reads after its RAM experts were done
    int64_t cpu_ram = 0, cpu_ssd = 0;
    double ms_ssd_wait = 0;
    /// ... its cache: records read, demands served from the cache, read-aheads issued / asked for later / evicted unused
    int64_t ssd_reads = 0, ssd_hits = 0, ahead_issued = 0, ahead_used = 0, ahead_wasted = 0;
    int64_t fill_cached = 0, fill_reads = 0;   ///< ... the VRAM tier's fills of SSD experts: from a cache slot / read again
    int64_t rec_reads = 0;                     ///< ... records read outside the cache (the prompt stream, the fills)
    int64_t rec_cached = 0;                    ///< ... the prompt stream's records taken from the cache instead
    int64_t prompt_kept = 0;                   ///< ... the prompt stream's reads that went into the cache (STRATA_DSV41_PROMPT_KEEP)
    int64_t prompt_adm_free = 0, prompt_adm_hot = 0;   ///< ... of them past its share (moex, DSV41_PLAN §41): free slots / hot ones
    int64_t prompt_sync = 0;                   ///< ... its records read on the spot instead of ahead (none expected)
    int64_t prompt_orphan = 0;                 ///< ... its records read ahead and then not taken (none expected)
    /// ... the prompt path's time with none of a stage's reads in flight: lending, waiting for a layer's routing, for the ring,
    /// for a bounce slot's copy, refilling
    double ms_idle_lend = 0, ms_idle_gate = 0, ms_idle_ring = 0, ms_idle_bounce = 0, ms_idle_refill = 0;
    double ms_idle_gate_dev[2] = {0, 0};       ///< ... ms_idle_gate's part on stages 0 and 1 (moex, DSV41_PLAN §41)
    double ms_eg_rows = 0;                     ///< ... the engram rows' fetch (decode windows and the prompt path)
    /// moex (DSV41_BACKLOG #59): with STRATA_DSV41_EG_READER=1, the n-gram reader's rows asked for, served by its row cache,
    /// and its 4 KB reads (every table summed)
    int64_t eg_reader_rows = 0, eg_reader_hits = 0, eg_reader_reads = 0;
    /// moex (DSV41_BACKLOG #103, STRATA_DSV41_EG_DEDUP=1): the prompt chunks' engram rows asked for, and the distinct ones read
    int64_t eg_dedup_asked = 0, eg_dedup_read = 0;
    int64_t swaps = 0, fills = 0;   ///< adaptive replacements, and slots filled (load + first use)
    int64_t slots = 0;
    double slot_mib = 0;            ///< one VRAM slot (the expert blob size the tier was sized for)
    int64_t streamed = 0;           ///< experts streamed to the GPU by the prompt path
    double ms_prefill_moe = 0;      ///< host time in the prompt path's MoE (includes waiting for the GPU)
    double ms_prefill_wait = 0;     ///< the prompt path waiting for each layer's routing
    double ms_prefill = 0;          ///< the prompt path's wall time
    double ms_prefill_cpu = 0;      ///< the prompt path's CPU-pool experts (EngineOptions::cpu_read, prefill_share)
    int64_t prefill_cpu_experts = 0;   ///< routed non-resident experts the CPU pool computed (prefill_share)
    int64_t prefill_share_experts = 0; ///< ... of this many routed non-resident ones in the shared layers
    int64_t tokens_prefill = 0;     ///< tokens read by the prompt path
    int64_t prefill_chunk = 0, prefill_ring = 0, lent_slots = 0;
    // the CPU pool's own phase timers (ms, cumulative) and the bytes its multi-token path read
    double pool_gu_ms = 0, pool_q_ms = 0, pool_down_ms = 0, pool_park_ms = 0, pool_drain_ms = 0, pool_repark_ms = 0;
    int64_t pool_bytes = 0;
    int pool_workers = 0;
};

struct GenResult {
    std::vector<int32_t> tokens;     ///< the generated tokens (greedy)
    int64_t drafted = 0, accepted = 0;   ///< MTP drafts verified / accepted
    int64_t windows = 0;                 ///< verify windows (trunk passes) in the decode
    int64_t offered_at[8] = {0}, accepted_at[8] = {0};   ///< drafts offered / accepted per depth
    double ms_prefill = 0, ms_gen = 0, ms_draft = 0;
};

/// A request's sampling (Strata's GEN keys, STRATA_QWEN_DESIGN.md 5): penalties -> top_k -> top_p -> min_p -> temperature
/// -> pick, on the GPU, Philox keyed by (seed, position).  Absent keys: greedy, top_k 20, no penalties.
struct SamplingParams {
    float temperature = 0.0f;   // <= 0: greedy
    /// 1..64: the list; more than 64: the widest list, 64; 0 (moex): no cut - the whole vocabulary (kernels::SamplerParams::
    /// full_vocab), under top_p < 1 its whole nucleus (DSV41_BACKLOG #56); with min_p > 0 the widest list.  A GEN line without
    /// top_k keeps 20: the server sends top_k
    /// 0 for a family that samples the whole vocabulary (DeepSeek-V4.1), so its requests never reach this default
    int top_k = 20;
    float top_p = 1.0f, min_p = 0.0f;
    /// moex: top_p and min_p over softmax(logits / T) (kernels::SamplerParams::temp_first) - set by run_request for DeepSeek-V4.1
    bool temp_first = false;
    int penalty_last_n = 0;
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    uint64_t seed = 0;          // 0: taken from the clock
    float spec_min_p = -1.0f;   // < 0: the engine's (EngineOptions::spec_min_p)
    int spec_coupled = -1;      // < 0: the engine's (EngineOptions::coupled_draft); 0 / 1: this request's (setup's calibration)
    int spec_window = 0;        // 0: the engine's longest window; 1: no drafting; 2..8: this request's cap
    float dspark_conf = -1.0f;  // moex: < 0 the engine's (STRATA_DSV41_DSPARK_CONF): DeepSeek-V4.1 DSpark's confidence floor
    float dspark_cum = -1.0f;   // moex: < 0 the engine's (STRATA_DSV41_DSPARK_CUM): its cut on the confidences' product
    int spec_policy = -1;       // DFlash: < 0 the engine's (EngineOptions::spec_policy); 0 the spec-min-p gate; 1 cost-aware
    int64_t read_short = -1;    // < 0: the engine's short-read limit; else this request's (a segment up to it: decode windows)
    int64_t read_cpu = -1;      // < 0: EngineOptions::cpu_read; else this request's (a prompt-path segment up to it: no stream)
    int64_t read_share = -1;    // < 0: EngineOptions::prefill_share; else this request's (a segment up to it: the CPU shares)
    int64_t read_ring = -1;     // < 0: the ring's own size (55% of a layer); else this request's prompt ring in slots (an A/B key)
};

/// What the engine tells its caller while a request runs (the serve loop prints them as Strata's protocol lines).
struct RequestHooks {
    std::function<void(int64_t reused)> resume;                                // before reading: this many prompt tokens are reused
    std::function<bool(int64_t reached, int64_t total, double ms)> progress;   // a prompt chunk is done (all stages); false: stop reading
    std::function<void(int64_t reused)> reused;                                // the prompt is read; the first window comes next
    std::function<bool(int32_t id)> token;                                     // a generated token (EOS included); false: stop
    std::function<bool()> stop;                                                // polled between chunks and windows
    std::function<void(int64_t done, int64_t total)> work;                     // moex: a long read's (chunk, layer) units done
};

/// moex: one image of a request (GENI, Strata's SVE1 records): its encoder rows replace the token embeddings of the prompt
/// positions [start, start + n) - the run of pad tokens the server expanded for it.  `rows` (n x n_embd floats, host memory)
/// stays valid while run_request runs; `hash` (FNV-1a of the record) keys the image for prefix reuse and checkpoints: a cached
/// prefix is reused only when its images are the same (ids alone match for any two pictures of the same size).
/// Positions are 1-D (mainline mtmd's NORMAL type for glm5v / mimovl): row i at position start + i.
struct ImageSpan {
    int64_t start = 0, n = 0;
    uint64_t hash = 0;
    const float* rows = nullptr;
    /// moex vision: the record's grid (SVE1's nx x ny = n): Hy4-flash's M-RoPE gives row r, column c of it (h, w) = (s + r,
    /// s + c); 0 (or a 1-D record, n x 1): a row of n
    int64_t nx = 0, ny = 0;
};

struct RequestStats {
    int64_t generated = 0, prompt = 0, reused = 0, prompt_read = 0;
    double prompt_ms = 0, decode_ms = 0;
    std::string finish = "length";   // "stop" (an EOS id) | "length" | "cancel"
    int64_t hits = 0, lookups = 0;   // decode: routed (token, expert) pairs the VRAM tier served / all
    int64_t drafts_accepted = 0, drafts_offered = 0, windows = 0;
    int64_t lookup_windows = 0, lookup_accepted = 0;   // prompt lookup: the windows it supplied, their drafts kept
    int64_t offered_at[8] = {0}, accepted_at[8] = {0};   // drafts offered / accepted per depth (lookup's included)
    double draft_ms = 0;
};

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /// Opens the model (any shard of a split GGUF), uploads the dense weights, builds the expert arena and the
    /// pool, and allocates the decode state for `o.n_ctx` positions.
    bool load(const std::string& gguf, const EngineOptions& o, std::string& err);
    const Hparams& hparams() const;

    /// One token at position `pos` (one sequence; positions must arrive in order from 0).  The logits stay on
    /// the device; read them with argmax() or logits().
    bool step(int32_t token, int64_t pos, std::string& err);
    /// T <= 8 consecutive tokens at positions pos0.. in one pass (the weights read once for all of them): an MTP
    /// verify window, or a step of the prefill.  `keep`: record what rollback() needs (EngineOptions::mtp_window).
    bool window(const int32_t* tokens, int T, int64_t pos0, bool keep, std::string& err);
    /// SERVING (session.cpp): one request on the live sequence - what the engine already holds is reused when the new
    /// prompt extends it (the live tokens, or a checkpoint of the recurrent state at an exact prefix), the rest of the
    /// prompt is read (the prompt path, or windows when short), the last prompt token goes through the first decode
    /// window, and tokens are sampled until an EOS id, max_new, or hooks.stop().  Tokens stream through hooks.token.
    /// `eos`: the ids that end a request (emitted first).  Checkpoints (Strata's policy, prompt-cache 6 / root 2048 /
    /// every 16384 tokens) are taken during reading at `turn_tokens` boundaries; ckpt = false: a one-shot request.
    /// `images` (moex, GENI): the request's pictures, sorted by start, disjoint, inside [0, prompt.size() - 1).
    bool run_request(const std::vector<int32_t>& prompt, int max_new, const SamplingParams& sp, const std::vector<int32_t>& eos,
                     const RequestHooks& hooks, bool ckpt, RequestStats& out, std::string& err,
                     const std::vector<ImageSpan>* images = nullptr);
    /// The role-start token ids ("<|user|>" etc.): a checkpoint is taken before the last one in a prompt, and the root is
    /// the first one at least `root` tokens in.  moex: `deep` > 0 takes the periodic ones at the multiples of `every` and
    /// keeps `deep` of them apart from the `keep` of the root and the turns; 0, every `every` fresh tokens among the `keep`.
    void set_checkpoint_policy(std::vector<int32_t> turn_tokens, int keep = 6, int64_t every = 16384, int64_t root = 2048,
                               int deep = 4);
    /// Forget the live sequence and every checkpoint (a new model state: a different config, an engine error).
    void forget();

    /// The prompt (positions pos0..): Strata's prompt path in chunks (EngineOptions::prefill_chunk), or decode
    /// windows of 8 for a short segment.  No logits: the last prompt token goes through the first decode window.
    bool prefill(const int32_t* tokens, int n, int64_t pos0, std::string& err);
    /// After a `keep` window: keep only its first n_keep tokens (>= 1) - the recurrent KDA state and conv state
    /// go back to just after token n_keep - 1; the attention caches need nothing (rewritten as positions return).
    void rollback(int n_keep);
    /// Greedy generation of n_gen tokens after `prompt` (positions from 0, after a reset()).  With `use_mtp`, the
    /// NextN block drafts one token and a 2-token window verifies it: accepted, the window yields two tokens;
    /// rejected, the trunk rolls back to the first and yields one.  The output is the same either way (greedy).
    bool generate(const std::vector<int32_t>& prompt, int n_gen, bool use_mtp, GenResult& r, std::string& err);
    /// The last window's arg-max per token, or of its last token.
    bool argmax_rows(std::vector<int32_t>& out, std::string& err);
    int32_t argmax(std::string& err);
    /// One row of the last window's logits (-1: its last token).
    bool logits(std::vector<float>& out, std::string& err, int row = -1);
    /// Zeroes the decode state (KDA states, the latent and indexer caches) for a new sequence.
    void reset();
    /// The speculation's knobs between requests (a benchmark sweeps them on one loaded model): the longest window (1: plain
    /// decode, 0: the loaded EngineOptions::mtp_window) and spec-min-p.
    void set_spec(int window_cap, float min_p);
    /// A measurement (the CLI's --decode-runs fN): every window is filled to `rows` rows with stand-in drafts - the
    /// context's own last tokens, which the trunk rejects as a rule - so a run's speed is the verify cost of that many rows
    /// (the break-even of any drafter).  0: off.  Never in serving.
    void set_stand_in(int rows);
    /// Strata's --short-read between prompts: a prompt segment of at most this many tokens is read through decode windows.
    void set_short_read(int64_t n);
    /// The imatrix collection (moex/tools/dsv41/requant_pack's input): from here on every prompt read through the prompt path
    /// adds each routed expert's squared inputs; the load must read prompts in the stream mode (cpu_read 0, prefill_share 0).
    bool imatrix_begin(std::string& err);
    /// The sums so far as a DSV41IMX file (per layer and expert: gate/up sums [n_embd], down sums [n_ff], tokens).
    bool imatrix_write(const std::string& path, std::string& err) const;
    /// Writes every expert this load routed (prompts and decode, undecayed) as an expert profile, hottest first.
    bool save_profile(const std::string& path, std::string& err) const;
    /// moex (DSV41_BACKLOG #52): DeepSeek-V4.1's SSD cache as a list of its records for the next start, and such a list read
    /// back into the cache in the background (the number of records asked for; 0 without a file or an SSD tier)
    bool save_cache_keys(const std::string& path, std::string& err);
    /// moex (DSV41_BACKLOG #53): the live and parked conversations into EngineOptions::conv_dir (park_disk.cpp) - the
    /// clean end's step; false when nothing could be written (err says why).  A no-op without conv_dir.
    bool save_conversations(std::string& err);
    int64_t refill_cache(const std::string& path);
    /// moex (DSV41_PLAN §54): DeepSeek-V4.1's image encoder on the first GPU, between requests: each (patches, rows) pair's
    /// SVP1 patches file (serve/vision_dsv41_plan.py write_patches) encoded into an SVE1 rows file (the picture's span, its
    /// delimiter rows included) - in free VRAM when it holds the workspace with room to spare, else in the GPU's top slots,
    /// lent as a prompt lends them and refilled after.  -> each picture's rows, the call's time, `how` the workspace was had.
    bool encode_pictures(const std::vector<std::pair<std::string, std::string>>& files, std::vector<int64_t>& rows, double& ms,
                         std::string& how, std::string& err);
    /// The prompt path's knobs between prompts (a benchmark sweeps them on one loaded model): see EngineOptions.
    void set_prompt_options(int prefill_chunk, int prefill_ring, int prefill_sub);

    /// Probe mode: every op output is copied to the host, recorded, and (with an oracle attached) replaced by the
    /// oracle's value.  Synchronous, for checking only; null switches it off.
    void set_recorder(moex::Recorder* r);
    const EngineStats& stats() const;

    struct Impl;   ///< internal (src/x/engine_impl.hpp): opaque here

private:
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::x
