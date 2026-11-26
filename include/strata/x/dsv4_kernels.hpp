// include/strata/x/dsv4_kernels.hpp - moex: the DeepSeek-V4-Flash (deepseek4) kernels of the attention side that are
// not projections - RoPE with YaRN, the Hadamard rotation, the three KV compressors (CSA / LID / HCA), the lightning
// indexer and its top-k, the sparse attention with sinks - and the head's hyper-connection reduce and the router.
//
// The SPECIFICATION is moex/docs/DSV4_SPEC.md (operator level, citing llama.cpp); each function names its section.  Line
// references below are to mainline ~/AI/llama.cpp (ggml/ and src/llama-kv-cache-dsv4.cpp are identical in the
// glm5next-mtp tree the spec cites; src/models/deepseek4.cpp is offset by ~8 lines there).  Where ggml's CPU and CUDA
// backends round differently, the kernel follows ggml-cuda's formulation (llama.cpp's reference run executes these ops
// on CUDA) and its comment says which; src/x/dsv4_kernels_parity.cpp checks every kernel against CPU references
// transcribed from ggml-cpu (and, where the two differ by more than ulps, from the ggml-cuda formula as well).
//
// What is NOT copied from ggml-cuda: its -use_fast_math intrinsics (__expf, __sinf/__cosf, __powf, approximate
// division; ggml/src/ggml-cuda/CMakeLists.txt:195) - these kernels use the precise functions - and the half-precision
// shortcuts of its flash attention (Q*scale rounded to F16 with an F16 scale, P rounded to F16, V.P accumulated in F16:
// fattn-mma-f16.cuh:1246-1274, 1070-1076): dsv4_attn keeps q, the probabilities and the accumulators in F32, and
// dsv4_attn_mma splits q and p into F16 hi + lo parts (~22 bits), as the GLM engine's attention does.  The lightning
// indexer DOES round q to F16, as ggml-cuda's tensor-core indexer does (lightning-indexer.cu:61-68): llama.cpp's top-512
// on this hardware is taken on those scores, and the DeepSeek reference quantizes the indexer's q and k much further.
//
// Two kernels have a tensor-core twin (*_mma: sm_80+, returns false and launches nothing elsewhere) with the same
// semantics and arguments, different only in summation order: measured on an RTX 3080 the twins win from ~1000 keys
// per token or T >= 2 (attention: 2x at 2k-8k keys and on prompts; indexer: 2.5-3x at 64K rows) and are within ~15%
// of the F32 SIMT kernels on the smallest decode windows.
//
// Conventions (the engine's windows): activations are row-major (T, ...) for T tokens at positions pos0 .. pos0+T-1 of
// one sequence.  All pointers are DEVICE pointers; every call enqueues on `stream`, allocates nothing and never
// synchronizes, so it can be captured in a CUDA graph.  A POSITION-DEPENDENT call takes a trailing `pos_dev`: when
// non-null, pos0 is read from that device word (a captured graph replays at every position) and every grid is static
// (sized for the worst case; blocks past the real work return at once), and `pos0` is ignored.  The results are bitwise
// the same either way.  T is unbounded unless noted (grids put T in a dimension that holds 65535 at least).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::x {

// ------------------------------------------------------------------ RoPE (DSV4_SPEC 3.0 ROPE)

/// ggml_rope_ext's parameters (mode NORMAL: interleaved pairs, n_dims 64).  attn_factor is the one ggml receives
/// (DSV4 passes dsv4_rope_attn_factor, deepseek4.cpp:11-17), so the magnitude is attn_factor * (1 + 0.1 ln(1/fs))
/// when ext_factor != 0 and attn_factor otherwise.
struct Dsv4Rope {
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    int n_ctx_orig;
};
/// P0: ratio-0 layers (deepseek4.cpp:903-910 with use_compress_rope false): plain RoPE, no YaRN.
Dsv4Rope dsv4_rope_p0(float freq_base = 10000.0f);
/// PC: compressed layers (q, raw kv, compressed rows, indexer q): base 160000, YaRN fs 1/16, ext 1, n_ctx_orig 65536,
/// beta 32/1, attn_factor = 1 / (1 + 0.1 logf(1/fs)) = 0x3f486de1 (effective magnitude exactly 1.0f).
Dsv4Rope dsv4_rope_pc(float compress_base = 160000.0f, float freq_scale = 0.0625f, float ext_factor = 1.0f,
                      int n_ctx_orig = 65536, float beta_fast = 32.0f, float beta_slow = 1.0f);
/// RoPE of x (T, rows, row_len) into y (same shape; y == x is in place): in every row of token t, the 32 interleaved
/// pairs of dims [offs, offs + 64) rotate by position pos0 + t, the other dims are copied (untouched in place).  Pair k
/// (k = 0..31): theta = (pos0 + t) * powf(theta_scale, k) - ggml-cuda's rope_norm formula (rope.cu:107; ggml-cpu
/// multiplies theta by theta_scale once per pair instead, ops.cpp:5975-5989; powf on the host) - then YaRN's ramp
/// with ggml_rope_yarn_corr_dims (ggml.c:4418-4430), cos/sin times the magnitude, and
///   y[2k] = x0 cos - x1 sin,  y[2k+1] = x0 sin + x1 cos      (x0 = x[offs + 2k], x1 = x[offs + 2k + 1]).
/// `inverse` negates sin: ggml_rope_ext_back, the de-rope of the attention output (deepseek4.cpp:1205).
void dsv4_rope(const float* x, float* y, int T, int rows, int row_len, int offs, const Dsv4Rope& p, bool inverse,
               int64_t pos0, void* stream, const int64_t* pos_dev = nullptr);
/// dsv4_rope with token t at position pos0 + shift + t (DeepSeek-V4.1: a compressed row rotates at its block's first
/// position, p - 1 for ratio 2).  dsv4_rope is this with shift 0, bitwise.
void dsv4_rope_shift(const float* x, float* y, int T, int rows, int row_len, int offs, const Dsv4Rope& p, bool inverse,
                     int64_t pos0, int64_t shift, void* stream, const int64_t* pos_dev = nullptr);

// ------------------------------------------------------------------ Hadamard (DSV4_SPEC 3.0 HADAMARD_n)

/// y = H x per row of n (128 or 512; y may alias x): x * (1 / sqrtf(n)) first, then the FWHT in natural (Sylvester)
/// order, (u, w) -> (u + w, u - w) for len = 1, 2, ..., n/2 - ggml-cpu's fwht (ops.cpp:11990-12058) and ggml-cuda's
/// fwht_cuda (fwht.cu) do exactly this, so the result is bitwise theirs.
void dsv4_hadamard(const float* x, float* y, int rows, int n, void* stream);

// ------------------------------------------------------------------ raw K = V ring (DSV4_SPEC 3.4)

/// The window's rope'd kv rows (T, 512) F32 into the raw ring: ring[(pos0 + t) % ring_rows] = half(kv[t]) (RNE, as
/// set_rows F32 -> F16).  The ring must hold the 127 positions before the window and the window: ring_rows >= T + 127.
void dsv4_raw_store(const float* kv, uint16_t* ring, int ring_rows, int64_t pos0, int T, void* stream,
                    const int64_t* pos_dev = nullptr);
/// moex (the ERNIE-5.5 hill-climb): rms_norm (rows of cols = 512, weight w or none) then dsv4_rope's in-place rotation at offs,
/// and with a ring (rows == 1) dsv4_raw_store - in one launch, the same bits; false (nothing launched) where it does not apply
/// or with STRATA_DSV4_NORM_ROPE=0
bool dsv4_norm_rope(float* x, const float* w, int T, int rows, int cols, float eps, int offs, const Dsv4Rope& p, int64_t pos0,
                    void* stream, const int64_t* pos_dev = nullptr, uint16_t* ring = nullptr, int ring_rows = 0);

// ------------------------------------------------------------------ KV compressors (DSV4_SPEC 3.5)

/// The geometry of one compressor: CSA {4, 8, 2, 512, false}, LID {4, 8, 2, 128, true}, HCA {128, 128, 1, 512, false}.
/// Rows of the projections, the ring and ape are coff * d wide.
struct Dsv4Compressor {
    int ratio;       ///< r: positions per compressed row
    int ring;        ///< S: rows of the persistent F32 state ring (row = position % S)
    int coff;        ///< 2: overlapping windows (CSA, LID), 1: HCA
    int d;           ///< width of a compressed row (512 or 128)
    bool hadamard;   ///< LID: the row is Hadamard-rotated before the store
};
Dsv4Compressor dsv4_csa();
Dsv4Compressor dsv4_lid();
Dsv4Compressor dsv4_hca();
/// THE POOL (call it before dsv4_compress_persist in the same window).  For every token t whose position p = pos0 + t
/// completes a block ((p + 1) % r == 0): block b = p / r, s0 = b * r, and per channel k < d a softmax over the slots of
/// the gate scores weights their kv values:
///   HCA (r slots):      slot j = position s0 + j, feature k
///   CSA/LID (2r slots): j < r: position s0 - r + j, feature k (the previous window's FIRST half);
///                       j >= r: position s0 + j - r, feature d + k (the current window's SECOND half)
/// A slot's (kv, score) is (akv[q - pos0], gs[q - pos0] + ape[q % r]) when its position q is in the window, the ring's
/// row q % S (stored as kv, gs + ape) when q < pos0, and (0, -inf) when q < 0 (dsv4_append_zero_row).  Then
///   c = sum_j softmax_j(score) kv_j  ->  RMS(c) * norm_w (eps)  ->  RoPE at s0 (PC, dims [d - 64, d))
///   ->  LID only: Hadamard-128  ->  cache row b as half (RNE).
/// The ring is read BEFORE dsv4_compress_persist overwrites it (llama.cpp reads the state, then persists): any T works,
/// and a window that completes no block writes nothing.  ggml-cuda's formulation (deepseek4.cpp:471-589 as CUDA ops):
/// the softmax's float sum and 1.0f / sum (softmax.cu:114-141), float sum_rows, rms_norm's rsqrtf (norm.cu:140-141),
/// rope_norm's theta, F16 by __float2half_rn.  akv/gs: (T, coff*d); ape: (r, coff*d); ring_kv/ring_sc: (S, coff*d);
/// norm_w: d; cache: (rows, d) halves, written at row b only.
void dsv4_compress_pool(const Dsv4Compressor& c, const float* akv, const float* gs, const float* ape, const float* ring_kv,
                        const float* ring_sc, const float* norm_w, float eps, const Dsv4Rope& rope, uint16_t* cache,
                        int64_t pos0, int T, void* stream, const int64_t* pos_dev = nullptr);
/// THE PERSIST (after the pool): every token's (akv_t, gs_t + ape[p_t % r]) into ring row p_t % S, the largest position
/// winning a row (llama.cpp's persist rows, llama-kv-cache-dsv4.cpp:503-517): with T > S only the last S tokens land.
void dsv4_compress_persist(const Dsv4Compressor& c, const float* akv, const float* gs, const float* ape, float* ring_kv,
                           float* ring_sc, int64_t pos0, int T, void* stream, const int64_t* pos_dev = nullptr);

// ------------------------------------------------------------------ lightning indexer (DSV4_SPEC 3.6)

/// Per token t and visible LID row j < n_vis(t) = (pos0 + t + 1) / 4 (rows of 128 halves):
///   score[t][j] = sum_{h < 64} relu(q16[t][h] . k_j) * w[t][h]
/// with q16 = q rounded to F16 (RNE) as ggml-cuda's tensor-core indexer does on sm_75+ (lightning-indexer.cu:61-68;
/// ggml-cpu keeps q in F32, ops.cpp:12150-12157): the products are exact in F32 and summed in F32.  iq: (T, 64, 128),
/// already rope'd and Hadamard-rotated; w: (T, 64), already scaled by 1/sqrt(8192); score: (T, max_rows), entries at
/// j >= n_vis(t) untouched.  max_rows bounds the rows (the static grid of the graph path and the score stride).
/// `ratio`: positions per compressed row (n_vis(t) = (pos0 + t + 1) / ratio): V4's 4, V4.1's 2 or 1.
/// moex (DSV41_BACKLOG #104): `packed` - lid_k holds DeepSeek-V4.1's packed index keys (dsv41_pack_row_bytes(128, 32) = 80
/// bytes a row) instead of F16 rows; each reads the halves the F16 rows would hold, so the scores are the same bits.
void dsv4_lid_scores(const float* iq, const uint16_t* lid_k, const float* w, float* score, int max_rows, int64_t pos0,
                     int T, void* stream, const int64_t* pos_dev = nullptr, int ratio = 4, bool packed = false);
/// dsv4_lid_scores on tensor cores (mma.sync m16n8k16 f16 x f16 -> f32, as ggml-cuda's lightning_indexer_kernel_wmma):
/// the same q16 . k products, exact, summed in F32 in the hardware's order, the same epilogue - so the scores differ from
/// dsv4_lid_scores' by the accumulation order only.  Returns false and launches nothing when the card has no mma.sync
/// m16n8k16 (sm_80) or STRATA_DSV4_MMA=0: use dsv4_lid_scores then.
/// moex (DSV41_PLAN §56 C): with `clist` (and `ccount`, `ccap`, `cbsize`), token t scores only the rows of the candidate
/// blocks clist[t * ccap + i], i < ccount[t] (dsv41_cand_list: blocks of cbsize rows, 64 % cbsize == 0), gathered eight
/// blocks to a tile; each row's score is the same bits as the full scan's, and the other rows are left untouched (the
/// candidates' mask then sets them to -inf, as after the full scan).
bool dsv4_lid_scores_mma(const float* iq, const uint16_t* lid_k, const float* w, float* score, int max_rows, int64_t pos0,
                         int T, void* stream, const int64_t* pos_dev = nullptr, int ratio = 4, bool packed = false,
                         const int32_t* clist = nullptr, const int32_t* ccount = nullptr, int ccap = 0, int cbsize = 0,
                         int heads = 64);   // moex (DSV41_PLAN §59 P2): 32 = V4.1's heads (q and w still padded to 64)
/// Whether the *_mma kernels run here (compute capability 8.0+ and STRATA_DSV4_MMA != 0).
bool dsv4_mma_supported();
/// The rows token t attends among the compressed CSA rows: every visible row when n_vis(t) <= top_k, else the top_k
/// highest-scoring visible rows, ties at the threshold to the lower row (one 1024-thread block per token: a radix
/// select over 1024 sampled scores sets a threshold, the ~1.25 top_k rows above it are compacted into shared memory and
/// the exact radix select runs there - over all the scores when the sample misses; the same rows either way).
/// sel: (T, top_k), the rows ASCENDING (the order llama.cpp's masked attention sums them in), then -1 padding; count
/// (T, may be null) = min(n_vis(t), top_k).  As a set this is llama.cpp's top_k + mask (deepseek4.cpp:673, 679-706).
/// n_vis(t) = (p + 1 - excl) / ratio (at least 0) for token t at p; moex (ERNIE-5.5): excl = the window its DSA layers
/// attend apart (0: V4's and V4.1's, unchanged).
void dsv4_lid_select(const float* score, int max_rows, int top_k, int32_t* sel, int32_t* count, int64_t pos0, int T,
                     void* stream, const int64_t* pos_dev = nullptr, int ratio = 4, int excl = 0);

// ------------------------------------------------------------------ sparse attention with sinks (DSV4_SPEC 3.7)

/// The split-K blocks per (token, 16-head group) of dsv4_attn / dsv4_attn_mma for a window of T tokens: 32 at T = 1,
/// 4 at T = 8, 1 from T = 32 on (no scratch).
int dsv4_attn_splits(int T);
/// The scratch of dsv4_attn / dsv4_attn_mma for a window of T tokens (0 when dsv4_attn_splits(T) == 1).
std::size_t dsv4_attn_scratch_bytes(int T);
/// 64 query heads of 512 against ONE shared K = V head: for token t at p_t and head h, over its keys
///   (i)  the raw ring rows of positions max(0, p_t - 127) .. p_t, at row p % raw_rows (written before the call);
///   (ii) ratio 4 (CSA): the compressed rows sel[t][0 .. min(n_vis, top_k)) with n_vis = (p_t + 1) / 4;
///        ratio 128 (HCA): every compressed row j < (p_t + 1) / 128;  ratio 0: none,
/// s_j = (q_h . k_j) * scale,  M = max(max_j s_j, sinks[h]),
/// out[t][h] = sum_j exp(s_j - M) k_j / (sum_j exp(s_j - M) + exp(sinks[h] - M))   (the sink is unscaled).
/// Keys are F16 rows of 512 (the full row, rope dims included, is both key and value); q (T, 64, 512) and out
/// (T, 64, 512) F32; everything else in F32 (see the file comment).  Split-K: dsv4_attn_splits(T) blocks share a
/// token's keys and a merge combines them; `scratch`: dsv4_attn_scratch_bytes(T).
/// moex: `noncausal` 2 - DeepSeek-V4.1's DSpark block: every row sees the 128 positions before the batch and the whole batch
/// (model.py get_dspark_topk_idxs); the ring must hold T + 128.
/// moex: `noncausal` 1 - every token of the batch also sees the batch's later rows (the raw keys run to the batch's last
/// position; each token keeps its own 128-position window below): DSpark's noise block (llama.cpp's non-causal ubatch,
/// llama-kv-cache.cpp:1677-1703, as dflash.cpp's draft context runs with causal attention off).  The ring must hold T + 127.
/// moex (DSV41_BACKLOG #104): `packed` - comp holds DeepSeek-V4.1's packed compressed rows (dsv41_pack_row_bytes(512, 16) =
/// 288 bytes a row; the raw ring stays F16), read as the halves the F16 rows would hold: the same bits out.
void dsv4_attn(const float* q, const uint16_t* raw, int raw_rows, const uint16_t* comp, int ratio, const int32_t* sel,
               int top_k, const float* sinks, float scale, float* out, float* scratch, int64_t pos0, int T, void* stream,
               const int64_t* pos_dev = nullptr, int noncausal = 0, bool packed = false);

/// dsv4_attn on tensor cores (kernels.cu's attend_mma_kernel design): the same keys, sink and split-K partials, q and
/// the probabilities each split into F16 hi + lo parts (~22 bits; the keys are F16, exact), F32 accumulation - so it
/// differs from dsv4_attn by summation order and exp2f vs expf, not by an F16 cast.  Same arguments and scratch.
/// Returns false and launches nothing when the card has no mma.sync m16n8k16 (sm_80) or STRATA_DSV4_MMA=0.
bool dsv4_attn_mma(const float* q, const uint16_t* raw, int raw_rows, const uint16_t* comp, int ratio, const int32_t* sel,
                   int top_k, const float* sinks, float scale, float* out, float* scratch, int64_t pos0, int T, void* stream,
                   const int64_t* pos_dev = nullptr, int noncausal = 0, bool packed = false);

// ------------------------------------------------------------------ head hyper-connection reduce (DSV4_SPEC 4)

/// pre = sigmoid((mix * hc_scale[0]) + hc_base) + eps (the mul and the add rounded separately), then UNFUSED
///   y = ((x0 * pre0 + x1 * pre1) + x2 * pre2) + x3 * pre3
/// every product and sum rounded on its own (il = -1: separate ggml mul/add ops, deepseek4.cpp:449-469, 307-316).
/// mixes (T, 4) (RMS of the 16384-flat residual times output_hc_fn, by the engine); x (T, 4, n); y (T, n).
void dsv4_hc_head(const float* x, const float* mixes, const float* hc_scale, const float* hc_base, float eps, float* y,
                  int n, int T, void* stream);

// ------------------------------------------------------------------ router (DSV4_SPEC 3.11 steps 2-4)

/// probs = sqrtf(softplus(logits)), softplus(x) = x > 20 ? x : logf(1 + expf(x)) (unary-ops.cpp:80, unary.cu:93).
/// Selection: hash_ids (T, k) when non-null (the hash layers: ids as given), else the top k of probs + bias (bias may
/// be null) by k warp arg-max rounds, ties to the lower id - ggml-cuda's fused topk_moe (topk-moe.cu:183-247).  Weights:
/// the UNBIASED probs at the ids, normalized and scaled - with top-k as topk_moe does (w * (1 / max(sum, 6.1035e-5)),
/// topk-moe.cu:252-259) then * scale; with hash ids as the unfused ops do (sum_rows, clamp, w / sum, * scale:
/// llama-graph.cpp:2066-2099).  The weights are always normalized (DSV4's expert_weights_norm); `scale` applies unless
/// it is 0 or 1.  ids/weights: (T, k) in selection order.  E <= 1024, k <= 32.
/// `pen` (optional, E floats, read by the kernel when it runs: mapped memory the host may rewrite between windows): subtracted
/// from the selection score only - DeepSeek-V4.1's opt-in route bias away from experts the SSD would have to read
/// (STRATA_DSV41_ROUTE_BIAS); nullptr, the release's routing.
/// `exact_softplus` (moex, DSV41_BACKLOG #56): torch's log1p(e^x), as DeepSeek-V4.1's model.py; false: topk-moe's
/// log(1 + e^x), V4's (llama.cpp's)
void dsv4_route(const float* logits, const float* bias, const int32_t* hash_ids, int E, int k, float scale, int32_t* ids,
                float* weights, int T, void* stream, const float* pen = nullptr, bool exact_softplus = false);
/// moex (DSV41_BACKLOG #112): dsv4_route's ids and weights (the same bits) and doorbell_publish's hand-off of the ids and the
/// n_x activations `x` (m_ids, m_x: mapped; the word m_seq from the counter d_cnt) in one launch.  False (nothing launched)
/// past 16 tokens.
bool dsv4_route_publish(const float* logits, const float* bias, const int32_t* hash_ids, int E, int k, float scale, int32_t* ids,
                        float* weights, int T, void* stream, const float* pen, bool exact_softplus, const float* x, int n_x,
                        int32_t* m_ids, float* m_x, volatile uint32_t* m_seq, uint32_t* d_cnt);
/// The hash layers' ids: ids[t][j] = tid2eid[tokens[t] * k + j] (ffn_gate_tid2eid [k, n_vocab] I32, get_rows by token).
void dsv4_hash_ids(const int32_t* tid2eid, const int32_t* tokens, int k, int T, int32_t* ids, void* stream);

}  // namespace strata::x
