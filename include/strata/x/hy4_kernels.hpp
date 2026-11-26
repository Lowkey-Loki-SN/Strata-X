// include/strata/x/hy4_kernels.hpp - moex: Hy4-flash's kernels (GGUF arch hy_v4_flash; moex/docs/HY4_PORT.md).
//
// Each one is an op of llama.cpp's graph (PR #32517, src/models/hy-v4-flash.cpp) over F32 rows on the device, with the
// launch shapes strata-x's decode windows (T <= 8) and prompt sub-batches (T up to the sub-batch) take.  `pos_dev`
// (non-null): the window's first position is read on the device (a captured graph's), else pos0.  `mpos` (moex vision,
// non-null while the server sends pictures): the request's M-RoPE table (Hy4MPos) - the rotations take each position's
// (t, h, w) from it, a text-only request's header {0, 0, 0} leaving every position its own.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::x {

/// moex vision: the device table hy4_rope_w / hy4_kv_row_w / hy4_kpool_w read - int64 {b, e, shift, 0}, then (e - b) int32
/// triples (t, h, w) for the positions [b, e); a position before b is its own coordinate, one at or past e is p - shift.
/// vLLM's get_mrope_input_positions (hy_v4/nvidia/vl_model.py): text counts on from the last position; a picture of
/// rows x cols tokens (the newline column included) takes t = s, h = s + row, w = s + col at s, the next text s + max(rows,
/// cols) - so `shift` sums each picture's rows * cols - max(rows, cols).
struct Hy4MPos {
    static constexpr int kHeader = 4;   ///< int64s before the triples
    static size_t bytes(int64_t positions) { return kHeader * sizeof(int64_t) + (size_t) positions * 3 * sizeof(int32_t); }
};

/// iHC's coefficients for T tokens, in mHC's 24-float layout so mhc_pre_w / mhc_post_w apply them: pre[c] =
/// sigmoid(mixes[c] * scale[0] + base[c]) + eps, post[c] = mag * sigmoid(mixes[4 + c] * scale[1] + base[4 + c]) + eps,
/// and the stream-mixing matrix the identity (mhc_post_w then adds each stream to round(x * post[c]) unchanged, as
/// llama.cpp's ggml_add does).  mixes (T, 8).
void hy4_ihc_coef_w(const float* mixes, const float* scale2, const float* base8, float* coef, float eps, float mag, int T,
                    void* stream);

/// RoPE on consecutive pairs (ggml's NORM rope): in each of `n_heads` heads of `head_dim` values at x + t * row_stride +
/// head_off + h * head_dim, the `n_rot` values from `rot_off`, pair k = (v[2k], v[2k + 1]) rotated by the angle
/// (pos0 + t) * powf(base^(-2 / n_rot), k) (ggml-cuda's rope_norm, freq_scale 1, no YaRN).
void hy4_rope_w(float* x, int row_stride, int head_off, int n_heads, int head_dim, int rot_off, int n_rot, float base,
                int64_t pos0, int T, void* stream, const int64_t* pos_dev = nullptr, const int64_t* mpos = nullptr);

/// The MLA cache row of each token from its kv_a output (T, 576), in place: rms_norm of the first 512 times `norm_w` (eps),
/// RoPE on the last 64 (consecutive pairs, `base`); with `cache` also both as FP16 into row (pos0 + t) % rows of `cache`
/// (rows x 576).  Null `cache`: hy4_store_kv_w stores them later (after the probes).
void hy4_kv_row_w(float* kv, const float* norm_w, float eps, float base, uint16_t* cache, int64_t rows, int64_t pos0, int T,
                  void* stream, const int64_t* pos_dev = nullptr, const int64_t* mpos = nullptr);
/// kv (T, 576) as FP16 into the cache rows (pos0 + t) % rows.
void hy4_store_kv_w(const float* kv, uint16_t* cache, int64_t rows, int64_t pos0, int T, void* stream,
                    const int64_t* pos_dev = nullptr);

/// The indexer's pools that complete in the window (the last member of group P at position P * kp + kp - 1 in
/// [pos0, pos0 + T)): softmax over the kp members of (gate + ape) per value, the weighted sum of the members' values, the
/// RMS norm (norm_w, eps), and RoPE on the last 64 of the 128 at the group's first position (P * kp) - into pooled[P].
/// The members' value and gate rows are in the rings at (position % ring_rows).
void hy4_kpool_w(const float* ring_iv, const float* ring_ig, int ring_rows, const float* ape, const float* norm_w, float eps,
                 float* pooled, float base, int64_t pos0, int T, int kp, void* stream, const int64_t* pos_dev = nullptr,
                 const int64_t* mpos = nullptr);

/// The indexer's scores (T, max_pool): token t at position p scores every group before its own, g < p / kp, as
/// sum_h relu(iq[t, h] . pooled[g]) * w[t, h] over 16 heads of 128 (iq (T, 16, 128), w (T, 16)).
void hy4_scores_w(const float* iq, const float* pooled, const float* w, float* score, int64_t pos0, int T, int kp, int max_pool,
                  void* stream, const int64_t* pos_dev = nullptr);

/// The cells (positions) token t attends on a sparse layer, max_cells a row: the best n_top of the groups g < p / kp by
/// score (all of them when there are no more than n_top), in ascending group order, each as its kp positions; then its
/// own group's positions kp * (p / kp) .. p.  Ties at the cut: the lower group first.
void hy4_select_w(const float* score, int max_pool, int n_top, int kp, int64_t pos0, int T, int32_t* cells, int max_cells,
                  void* stream, const int64_t* pos_dev = nullptr);
/// The cells of position `pos` (host): min(pos / kp, n_top) * kp + pos % kp + 1.
int hy4_cells(int64_t pos, int n_top, int kp);

/// MLA attention over the cache, the absorbed form: q (T, H, 576) = [q_abs (512) | q_pe (64)] per head, cache rows of 576
/// FP16 (position j's row at j % rows).  Token t (position p) attends its cells (`cells` non-null: hy4_select_w's list,
/// hy4_cells(p) of them) or the window's positions max(0, p - window + 1) .. p; score = q . row * scale, the head's sink
/// logit (unscaled) in the softmax's denominator; ctx (T, H, 512) = sum_c p_c row_c[0:512].  `n_chunks` > 1 splits a
/// token's cells over blocks (decode windows) into `partial` (hy4_attend_scratch_bytes); 1: one block a (token, 16 heads),
/// no scratch.
/// counts (DFly2's draft rows): each row's cell count on the device, in place of the k-pool rule (cells required)
void hy4_attend_w(const float* q, const uint16_t* cache, int64_t rows, const int32_t* cells, int max_cells, int n_top, int kp,
                  int window, const float* sinks, float scale, float* ctx, float* partial, int n_chunks, int H, int64_t pos0,
                  int T, void* stream, const int64_t* pos_dev = nullptr, const int32_t* counts = nullptr);
size_t hy4_attend_scratch_bytes(int H, int n_chunks, int T);
/// The split-K chunks for at most `max_cells` cells a token (32 cells a chunk).
int hy4_attend_chunks(int max_cells);

/// q (T, H, 256) -> q_nope head-major (H, T, 192) and q_pe into qf (T, H, 576) at [512, 576).
void hy4_q_split(const float* q, float* q_nope_hm, float* qf, int H, int T, void* stream);
/// q_abs head-major (H, T, 512) -> qf (T, H, 576) at [0, 512).
void hy4_q_join(const float* qabs_hm, float* qf, int H, int T, void* stream);

/// o[i] *= sigmoid(g[i]), n values (ggml_sigmoid then ggml_mul).
void hy4_gate_mul(float* o, const float* g, int64_t n, void* stream);
/// y = x with every value rounded to BF16 (round to nearest even, as ggml's fp32_to_bf16) - STRATA_X_BF16_ACT's A/B
void hy4_round_bf16(const float* x, float* y, int64_t n, void* stream);
/// DFly2 (hy4_dfly.cpp): rows [max(0, T - ring), T) of src into dst's ring of `ring` rows at row (position % ring)
void hy4_snap_rows_w(const void* src, void* dst, int row_bytes, int T, int ring, int64_t pos0, void* stream,
                     const int64_t* pos_dev = nullptr);
/// a draft block's cells: row o = the target row's n_hist cells, then anchor .. anchor + o; counts[o] = n_hist + o + 1
void hy4_dfly_cells_w(const int32_t* hist, int n_hist, int64_t anchor, int W, int32_t* out, int stride, int32_t* counts, void* stream);
/// x += g * y (elementwise)
void hy4_gated_add(float* x, const float* g, const float* y, int n, void* stream);

/// The Engram's looked-up rows (T, cols, dim) of FP8 E4M3 bytes: each row RMS-normed (eps) times its column's gamma
/// (gamma (cols, dim)) times valid[t, cols] - into out (T, cols * dim).
void hy4_eg_rows_w(const uint8_t* rows, int fmt, const float* gamma, const float* valid, float* out, int cols, int dim, float eps, int T,
                   void* stream);

/// The Engram's update of the streams R (T, 4, n) from kv (T, 5, n) = the fused projection (keys 0..3, then the value),
/// before its bias kv_b (5, n): per stream c, dot = sum(rms(R_c) * rms(key_c) * egate_c) / sqrt(n), gate =
/// sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)); R_c += rms(value) * pre_gamma * (gate + hc_scale_c), and nothing where
/// the token has no valid column (valid (T, cols)).  egate / hc_scale (4, n).  The RMS norms take `eps`, without weights.
void hy4_eg_apply_w(float* R, const float* kv, const float* kv_b, const float* egate, const float* pre_gamma, const float* hc_scale,
                    const float* valid, int cols, int n, float eps, int T, void* stream, float eps_qk = -1.0f);

}  // namespace strata::x
