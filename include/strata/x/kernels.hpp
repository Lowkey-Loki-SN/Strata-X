// include/strata/x/kernels.hpp - moex: strata-x's decode kernels that are not projections, first written for
// GLM-5.3-Flash (glm5next).
//
// The projections are Strata's native MMVQ on the GGUF's own blocks (Q8_0 / Q6_K / Q5_K / Q4_K / Q3_K / Q2_K, the
// activation as Q8_1).  What lives here is everything between them, one token at a time, and the SPECIFICATION
// is moex/ref/glm_ref.cpp - the CPU reference that matches llama.cpp's graph op by op (worst 6e-7, most ops
// bitwise; `glm_ref --oracle`).  Each function names the reference function it transcribes; the parity test
// (src/x/kernels_parity.cpp) checks each against that arithmetic on random inputs.
//
// All pointers are DEVICE pointers; every call enqueues on `stream` and allocates nothing.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::x {

// ------------------------------------------------------------------ generic

/// y = x / sqrt(mean(x^2) + eps) * w, per row of `cols` (w may be null: no weight).  In place is allowed.
void rms_norm(const float* x, const float* w, float* y, int rows, int cols, float eps, void* stream);
/// ggml_norm then * w + b, one row of `cols` (the indexer key's LayerNorm).  In place is allowed.
void layer_norm(const float* x, const float* w, const float* b, float* y, int cols, float eps, void* stream);
/// y[r] = dot(W row r, x): an F32 matrix of n_out rows of n_in (the router, the indexer's head weights).
void gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, void* stream);
/// out = silu(min(g, L)) * clamp(u, -L, L) (ggml swiglu_clamp); L == 0: silu(g) * u; L < 0 (kSituGlu, ERNIE-5.5): SiTU-GLU
/// (strata/kernels/glu_act.hpp).  `out` may alias g or u.
void swiglu_clamp(const float* g, const float* u, float* out, int n, float limit, void* stream);
/// y = x * s, in place.
void scale_f32(float* x, int n, float s, void* stream);
/// out = a + b (out may alias either).
void add_f32(const float* a, const float* b, float* out, int n, void* stream);
/// The index of the largest of n values (lowest index on ties) into *out (device).
void argmax_f32(const float* x, int n, int32_t* out, void* stream);

// ------------------------------------------------------------------ mHC (glm_ref.cpp: hc_coef / hc_pre_apply / hc_post)

/// `coef` = pre[4] | post[4] | comb[16] (comb[dst + 4*src]) from the 24 mixes: the sigmoids and the fused
/// Sinkhorn (softmax over dst per src, + eps, then norm_cols and (iters-1) x (norm_rows, norm_cols)).
void mhc_coef(const float* mixes, const float* scale3, const float* base24, float* coef, float eps, int iters,
              void* stream);
/// out[i] = sum_h R[h*n + i] * pre[h].
void mhc_pre(const float* R, const float* coef, float* out, int n, void* stream);
/// R[dst*n + i] = x[i] * post[dst] + sum_src R[src*n + i] * comb[dst + 4*src], in place.
void mhc_post(const float* x, float* R, const float* coef, int n, void* stream);
/// out[i] = (((R0 + R1) + R2) + R3) * 0.25 (build_hc_mean).
void mhc_mean(const float* R, float* out, int n, void* stream);

// ------------------------------------------------------------------ KDA (glm_ref.cpp: kda_layer)

/// ggml_ssm_conv of one stream with the carried state, then SiLU: out[c] = silu(sum_t inp[t][c] * w[c*dc + t]),
/// inp = [state (dc-1 samples, oldest first) | x].  The state slides.  state: (channels, dc-1), [c*(dc-1) + t].
void kda_conv_silu(float* conv_state, const float* x, const float* w, float* out, int channels, int d_conv,
                   void* stream);
/// g (in: ssm_f_b's output, out: the per-channel log decay) and beta (in: raw, out: sigmoid):
///   lb > -inf: g = lb * sigmoid(-((g + dt_bias) * A[h]))      else: g = softplus(g + dt_bias) * A[h]
void kda_gates(float* g, const float* dt_bias, const float* A, float lower_bound, float* beta, int H, int S,
               void* stream);
/// Per row of `cols`: x = x / sqrt(sum(x^2) + eps) * scale.
void kda_l2norm(float* x, int rows, int cols, float eps, float scale, void* stream);
/// The delta rule, one token, IN PLACE on `state` (S, H, S) = (i key, h, j value), j fastest:
///   S[i][:] *= exp(g[h][i]);  d[j] = (v[j] - sum_i S[i][j] k[i]) * beta[h];  S[i][j] += k[i] d[j];
///   o[j] = sum_i S[i][j] q[i]
/// q must already carry its 1/sqrt(S) (kda_l2norm's scale).
void kda_step(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
              float* o, int H, int S, void* stream);
/// y = rms_norm(o, eps) * w * sigmoid(g2), per head of S (w: S values, shared by the heads).
void kda_out_gate(const float* o, const float* g2, const float* w, float* y, int H, int S, float eps, void* stream);

// ------------------------------------------------------------------ DSA (glm_ref.cpp: dsa_layer)

/// The latent at `pos` into the FP16 cache: cache[pos * L + i] = half(kv[i]).
void dsa_store_latent(const float* kv, uint16_t* cache, int64_t pos, int L, void* stream);
/// A completed pool: pooled[d] = sum_k softmax_k(ig[k*D + d] + ape[d + D*k]) * ik[k*D + d], k < kpool.
/// `ik`/`ig` point at the pool's first member's row; members are consecutive rows of D.
void dsa_kpool(const float* ik, const float* ig, const float* ape, float* pooled, int D, int kpool, void* stream);
/// score[P] = sum_h relu(dot(iq[h], pooled[P])) * w[h], P < n_pool (the lightning indexer, no mask).
void dsa_scores(const float* iq, const float* pooled, const float* w, float* score, int n_pool, int IH, int D,
                void* stream);
/// The cells a token at `pos` attends (one sequence from position 0): the members of the top `n_top` pools by
/// score (all `n_pool` complete pools when n_pool <= n_top), then the tail (pos, pos-1, ...) when `tail`.
/// Returns the cell count, which the host knows from `pos` alone: 4*min(n_pool, n_top) + n_tail.
int dsa_select(const float* score, int n_pool, int n_top, int kpool, int64_t pos, bool tail, int32_t* cells,
               void* scratch, void* stream);
std::size_t dsa_select_scratch_bytes(int max_pool);
/// Sparse MQA over the FP16 latent: per head h, ctx[h] = softmax_c(scale * q[h] . kv[c]) . kv[c] over `cells`.
/// `partial`: dsa_attend_scratch_bytes.
void dsa_attend(const float* q, const uint16_t* kv, const int32_t* cells, int n_cells, float scale, float* ctx,
                float* partial, int H, int L, void* stream);
std::size_t dsa_attend_scratch_bytes(int H, int L, int max_cells);

// ------------------------------------------------------------------ MoE (glm_ref.cpp: moe_ffn)

/// Sigmoid routing with a selection bias: probs = sigmoid(logits); the top `k` of probs + bias; weights =
/// probs[top], normalised to sum 1 (sum floored at 6.103515625e-5) when `norm`, then * scale (when not 0 or 1).
/// ids/weights: k each, in descending selection order.  E <= 1024.
void moe_route(const float* logits, const float* bias, int E, int k, bool norm, float scale, int32_t* ids,
               float* weights, void* stream);
/// out[i] = sum_j weights[j] * parts[j*n + i] + shared[i] (shared may be null).
void moe_combine(const float* parts, const float* weights, const float* shared, float* out, int n, int k,
                 void* stream);

// ------------------------------------------------------------------ WINDOWS: T tokens at once
//
// The same arithmetic over T consecutive tokens (positions pos0 .. pos0+T-1 of one sequence), for MTP verify
// windows and batched prefill.  Activations are token-major: x[t][...].  Each is checked against T sequential
// single-token calls by x_kernels_parity.  T is unbounded unless noted; the 8-column limit is MMVQ's.

void mhc_coef_w(const float* mixes, const float* scale3, const float* base24, float* coef, float eps, int iters, int T,
                void* stream);
/// moex (the ERNIE-5.5 hill-climb): one token's mixer read with its pre coefficients computed in place (mhc_coef_w's
/// expression) and the weighted RMS norm, one block (mhc_pre_w + rms_norm's bits); n <= 5120
/// With `parts` (gemv_f32_w1_warps' 8 warp totals a row, 24 rows), the rows' sums first, written to mixes (the chunk kernel's).
void mhc_pre_norm_w1(float* mixes, const float* scale3, const float* base24, float eps, const float* R, float* cur,
                     const float* norm_w, int n, float rms_eps, void* stream, const float* parts = nullptr);
/// moex (the ERNIE-5.5 hill-climb): one column's y = W x as gemv_f32_w (T 1) computes it, its 8 warps' totals a row into
/// part[row * 8 + w] (n_out x 8 floats) - their sum in order (mhc_pre_norm_w1 with parts) is y's bits; false with
/// STRATA_X_F32_WARPS=0 (nothing launched)
bool gemv_f32_w1_warps(const float* W, const float* x, float* part, int n_in, int n_out, void* stream);
void mhc_pre_w(const float* R, const float* coef, float* out, int n, int T, void* stream);
/// moex (DSV41_PLAN §34): gemv_f32_split's reduce of S partials per mixer row, mhc_coef_w, mhc_pre_w with the carried
/// coefficients `cpre` (then cpre = coef) and rms_norm with norm_w, for one token in one launch - bitwise those launches
void mhc_mix_in_w1(const float* part, int S, const float* scale3, const float* base24, float* mixes, float* coef, float* cpre,
                   const float* R, float* cur, const float* norm_w, int n, float eps, int iters, float rms_eps, void* stream);
void mhc_post_w(const float* x, float* R, const float* coef, int n, int T, void* stream);
void mhc_mean_w(const float* R, float* out, int n, int T, void* stream);

/// The conv over T tokens in order: x/out (T, channels); the state slides T times.
void kda_conv_silu_w(float* conv_state, const float* x, const float* w, float* out, int channels, int d_conv, int T,
                     void* stream);
/// g (T, H, S), beta (T, H).
void kda_gates_w(float* g, const float* dt_bias, const float* A, float lower_bound, float* beta, int H, int S, int T,
                 void* stream);
/// The delta rule over T tokens in order, the state held in registers across the tokens (read once, written once).
/// q/k/v/g/o: (T, H, S); beta: (T, H).  `snaps` (null: none): the state after every token, (T, S, H, S) - what an
/// MTP verify window rolls back to.
void kda_scan(float* state, const float* q, const float* k, const float* v, const float* g, const float* beta,
              float* o, int H, int S, int T, float* snaps, void* stream);
/// Rebuilds the conv state after the first `n_keep` of the window's T inputs (an MTP rollback): `prev` is the state
/// before the window, `x` the window's inputs (T, channels); writes `state`.
void kda_conv_rollback(const float* prev, const float* x, float* state, int channels, int d_conv, int n_keep,
                       void* stream);

/// Latents (T, L) into the FP16 cache at positions pos0 ..
void dsa_store_latent_w(const float* kv, uint16_t* cache, int64_t pos0, int L, int T, void* stream, const int64_t* pos_dev = nullptr);
/// Rows (T, D) into a ring of `ring_rows` rows at slot (pos0 + t) % ring_rows.
void dsa_ring_store_w(const float* src, float* ring, int ring_rows, int64_t pos0, int D, int T, void* stream, const int64_t* pos_dev = nullptr);
/// Every pool whose last member is in [pos0, pos0 + T): its members are ring slots (P*kpool + k) % ring_rows.
void dsa_kpool_w(const float* ring_ik, const float* ring_ig, int ring_rows, const float* ape, float* pooled,
                 int64_t pos0, int T, int D, int kpool, void* stream, const int64_t* pos_dev = nullptr);
/// Per query t: score[t][P] for its n_pool(t) = (pos0 + t + 1) / kpool complete pools.  iq (T, IH, D), w (T, IH),
/// score (T, max_pool).
void dsa_scores_w(const float* iq, const float* pooled, const float* w, float* score, int64_t pos0, int T, int kpool,
                  int IH, int D, int max_pool, void* stream, const int64_t* pos_dev = nullptr);
/// dsa_scores_w's two kernels (32 heads of 128), for the parity test: the tiled GEMM a prompt sub-batch takes, and the
/// warp kernel decode windows and captured graphs take.  Their scores are bit-identical.
void dsa_scores_tiled(const float* iq, const float* pooled, const float* w, float* score, int64_t pos0, int T, int kpool,
                      int max_pool, void* stream);
void dsa_scores_warp(const float* iq, const float* pooled, const float* w, float* score, int64_t pos0, int T, int kpool,
                     int max_pool, void* stream, const int64_t* pos_dev = nullptr);
/// Per query t: its cells (T, max_cells) - the top n_top pools (all when fewer) then its tail.  The count of query
/// t is dsa_cells(pos0 + t, ...).
void dsa_select_w(const float* score, int max_pool, int n_top, int kpool, int64_t pos0, int T, bool tail,
                  int32_t* cells, int max_cells, void* stream, const int64_t* pos_dev = nullptr);
/// How many cells a query at `pos` attends.
int dsa_cells(int64_t pos, int n_top, int kpool, bool tail);
/// Sparse MQA per query t over its cells: q (T, H, L), ctx (T, H, L); `partial`: dsa_attend_w_scratch_bytes.
void dsa_attend_w(const float* q, const uint16_t* kv, const int32_t* cells, int max_cells, int64_t pos0, int T, int n_top,
                  int kpool, bool tail, float scale, float* ctx, float* partial, int H, int L, void* stream, const int64_t* pos_dev = nullptr);
/// The prompt path's DSA attention on tensor cores (Strata's qsa_prompt_attn for the MLA latent): the same arguments
/// as dsa_attend_w without the split-K scratch, ALL T queries in one launch.  Returns false (nothing launched) when the
/// card has no mma.sync m16n8k16 (sm_80), L != 512, H % 16 != 0, or STRATA_X_ATTN_MMA=0: use dsa_attend_w then.
bool dsa_attend_mma(const float* q, const uint16_t* kv, const int32_t* cells, int max_cells, int64_t pos0, int T, int n_top,
                    int kpool, bool tail, float scale, float* ctx, int H, int L, void* stream);
bool dsa_attend_mma_supported();
std::size_t dsa_attend_w_scratch_bytes(int H, int L, int max_cells, int T);

/// Per token: logits (T, E) -> ids/weights (T, k).
/// THE DOORBELL (kernels.cu): publish copies `ids` and `x` into mapped host memory (m_*: the DEVICE views of
/// cudaHostAllocMapped buffers), fences, and sets *m_seq = ++*d_cnt; wait spins until the host's *m_flag reaches
/// ++*d_cnt (or *m_abort != 0).  Captured in graphs: the counters are device memory the kernels increment.
void doorbell_publish(const int32_t* ids, int n_ids, const float* x, int n_x, int32_t* m_ids, float* m_x,
                      volatile uint32_t* m_seq, uint32_t* d_cnt, void* stream);
void doorbell_wait(const volatile uint32_t* m_flag, const volatile uint32_t* m_abort, uint32_t* d_cnt, void* stream);
/// slot[i] = res_layer[ids[i]]: the VRAM tier's slot of each routed (token, expert) pair, or < 0, from the DEVICE's table.
void hit_slots(const int32_t* ids, const int32_t* res_layer, int32_t* slot, int n, void* stream);
/// The MoE combine with the CPU's rows read from mapped memory where mask[t * k + j] is set, the GPU's `parts` elsewhere:
/// moe_combine_w's arithmetic exactly.
void moe_combine_pairs(const float* parts, const float* cpu_rows, const uint8_t* mask, const float* weights,
                       const float* shared, float* out, int n, int k, int T, void* stream);
void moe_route_w(const float* logits, const float* bias, int E, int k, bool norm, float scale, int32_t* ids,
                 float* weights, int T, void* stream);
/// out (T, n) = sum_j w[t][j] parts[t][j] + shared[t] (shared may be null); parts (T, k, n).
void moe_combine_w(const float* parts, const float* weights, const float* shared, float* out, int n, int k, int T,
                   void* stream);
/// y (T, n_out) = W x[t] for T columns of n_in (F32 W), the weights read once per 8 columns.
void gemv_f32_w(const float* W, const float* x, float* y, int n_in, int n_out, int T, void* stream);
/// moex (DSV41_PLAN §62), STRATA_DSV41_BF16_W: gemv_f32_w with BF16 weights - each weight widened to the F32 an exact copy
/// holds (its upper half), then the same multiply-adds in the same order: the same bits as gemv_f32_w on that copy, from half
/// the bytes
void gemv_bf16_w(const uint16_t* W, const float* x, float* y, int n_in, int n_out, int T, void* stream);
/// moex: an engine's default for gemv_bf16_w's one-column preload kernel (gemv_bf16_w1_pre_kernel: the same per-thread order,
/// warp sums and warp order - the same bits; STRATA_DSV41_PRELOAD=0/1 still decides when set).  Hy4-flash turns it on: its
/// BF16 head (3,072 x 120,832) 2,615 -> 1,031 us a token on an RTX 3080.
void gemv_bf16_w_preload_default(bool on);
/// moex (2026-11-19): the same for gemv_f32_w's one-column preload (gemv_f32_w1_pre_kernel: the general kernel's per-thread order,
/// warp sums and warp order - the same bits); STRATA_DSV41_PRELOAD wins when set
void gemv_f32_w_preload_default(bool on);
/// n BF16 values widened to F32, exactly (each the upper half of its float)
void bf16_widen(const uint16_t* src, float* dst, size_t n, void* stream);
/// moex (DSV41_PLAN §70), STRATA_DSV41_DENSE_FP8: gemv_bf16_w with FP8 weights - each e4m3 value times its block's power-of-two
/// scale (S: one e8m0 byte for every 32 values of a row) is the F32 the BF16 weight held, then the same multiply-adds in the same
/// order: the same bits as gemv_bf16_w on the BF16 matrix, from half the bytes
void gemv_fp8_w(const uint8_t* W, const uint8_t* S, const float* x, float* y, int n_in, int n_out, int T, void* stream);
/// moex (the hill-climb): y_a = A x and y_b = B x (T 1) in ONE launch, each row by the kernel a one-column call would run
/// (FP8: kDtFP8 codes + scales; BF16: the preload kernel) - the same bits; false (nothing launched) where a row's kernel would
/// differ (T > 1, no BF16 preload, STRATA_X_PAIR=0): the caller runs the two calls
bool gemv_pair_w(const void* Wa, const uint8_t* Sa, bool a_fp8, float* ya, int na, const void* Wb, const uint8_t* Sb, bool b_fp8,
                 float* yb, int nb, const float* x, int n_in, int T, void* stream);
/// moex (DSV41_BACKLOG #138): H groups of the above in one launch - group h takes W / S / x / y at its own offset
/// (n_out x n_in weights, T x n_in input, T x n_out output), each group's rows bitwise its own call's.  T 1..8.
void gemv_fp8_heads(const uint8_t* W, const uint8_t* S, const float* x, float* y, int n_in, int n_out, int T, int H,
                    void* stream);
/// n FP8 values (row-major, a scale for every 32) widened to F32, exactly as bf16_widen widens their BF16 copies
void fp8_widen(const uint8_t* src, const uint8_t* S, float* dst, size_t n, void* stream);
/// moex (ERNIE55_PLAN, opt-in STRATA_X_TC3=1): the same values as BF16 (exact: fp8_widen's floats have zero low halves)
void fp8_bf16(const uint8_t* src, const uint8_t* S, uint16_t* dst, size_t n, void* stream);
/// x (n floats) as three BF16 planes hi, mid, lo with x = hi + mid + lo exactly
void split3_bf16(const float* x, uint16_t* parts, size_t n, void* stream);
/// gemv_f32_w for a few long rows, T <= 8: each row split gemv_f32_split_parts(n_in) ways across blocks, the partial sums
/// (part: T * n_out * parts floats) reduced in a fixed order - the same products, summed in another order
int gemv_f32_split_parts(int n_in);
void gemv_f32_split(const float* W, const float* x, float* y, float* part, int n_in, int n_out, int T, void* stream);
/// ... its first launch alone: the partials (part[(t n_out + row) S + s])
void gemv_f32_split_part(const float* W, const float* x, float* part, int n_in, int n_out, int T, void* stream);
/// DSpark's Markov head step: li[v] += w2[v, :] . w1[*d_prev, :] for v < V (w2 [V, rank], w1 [vocab, rank]), d_prev on the device
void markov_step(const float* w2, const float* w1, const int32_t* d_prev, int rank, int V, float* li, void* stream);
/// Per row: the arg-max of x (T, n) into out (T).
void argmax_w(const float* x, int n, int32_t* out, int T, void* stream);
/// Per row: the arg-max of x (T, n) into out (T) and the probability softmax(x) gives it into out_p (T): the drafter's
/// token and its confidence (spec-min-p).
void argmax_prob_w(const float* x, int n, int32_t* out, float* out_p, int T, void* stream);
/// THE SLOTS' INTEGRITY CHECK (Strata's verify_slot, "a slot table that is right about indices and wrong about bytes
/// produces a plausible token"): a weighted word sum of each byte range, on the device (one block per job; `jobs` and `out`
/// may be mapped host memory) and the same on the host, so a slot is compared with what was read for it from the file.
struct SliceJob {
    const uint8_t* p;
    uint64_t len;
};
void slices_hash(const SliceJob* jobs, int n, uint64_t* out, void* stream);
/// moex (DSV41_PLAN §35): each slice's hash in P parts (parts[j P + p]; their sum modulo 2^64 is slices_hash's value)
void slices_hash_parts(const SliceJob* jobs, int n, int P, uint64_t* parts, void* stream);
uint64_t slices_hash_host(const uint8_t* p, uint64_t len);
/// THE STEP INPUTS IN MAPPED MEMORY (Strata's Verifier: every per-step value lives in device memory, filled without the copy
/// engine).  A decode graph's first node: d_pos <- *m_pos (if m_pos), d_res[0, n_res) <- m_res, dst (rows, width) <- m_rows
/// with each source row (src_width values) repeated across the destination row.  m_*: device views of mapped host memory.
void window_load(const int64_t* m_pos, const float* m_rows, const int32_t* m_res, int64_t* d_pos, int32_t* d_res, int n_res,
                 float* dst, int rows, int width, int src_width, void* stream);
/// m_out[0, count) <- x: a stage's residual rows into mapped memory for the next GPU (fenced: visible once the kernel is done).
void store_mapped(const float* x, float* m_out, size_t count, void* stream);
/// moex (DSV41_PLAN §76): T rows of `width` values copied from rows of src_ld to rows of dst_ld
void rows_copy_ld(const float* src, int src_ld, float* dst, int dst_ld, int width, int T, void* stream);
/// The drafter's token and probability into mapped memory (the host reads them after the stream's work is done).
void store_draft(const int32_t* tok, const float* prob, int32_t* m_tok, float* m_prob, void* stream);
/// dst[0, count) = src: a device copy as a kernel (not the copy engine).
void copy_f32(const float* src, float* dst, size_t count, void* stream);
/// `rows` rows of n floats from src (row stride src_stride) to dst (row stride dst_stride), as a kernel: either side may be
/// mapped host memory (a decode window's features between the stages - no copy engine)
void copy_rows_w(const float* src, int64_t src_stride, float* dst, int64_t dst_stride, int n, int rows, void* stream);
/// n (<= 32) drafts and their probabilities into mapped memory
void store_drafts(const int32_t* tok, const float* prob, int n, int32_t* m_tok, float* m_prob, void* stream);
/// y (T, 2n) = [a | b] per row (a, b: (T, n)).
void concat_rows(const float* a, const float* b, float* y, int n, int T, void* stream);
/// H per-head Q8_0 matrices (n_out rows of n_in each, back to back) against each head's own T Q8_1 columns
/// (x: head h's T columns contiguous, n_in each): y (H, T, n_out).  One launch for the MLA's 64 heads.
/// dst[i] = i * step for i < n (the bounds of n - 1 equal row blocks: MMQ with each head as an "expert").
void strided_bounds(int32_t* dst, int n, int step, void* stream);
void heads_q8_0(const void* W, const void* x_q8_1, float* y, int H, int n_in, int n_out, int T, void* stream);
/// dst column i = src column cols[i] (device indices), columns of col_bytes (a multiple of 16).
void gather_cols(const void* src, void* dst, const int32_t* cols, size_t col_bytes, int m, void* stream);
/// dst row rows[i] = src row i, rows of n floats.
void scatter_rows(const float* src, float* dst, const int32_t* rows, int n, int m, void* stream);
/// MMQ gate+up rows [gate (nf) | up (nf)] -> h (rows, nf) = silu(min(g, L)) * clamp(u, -L, L) (L == 0: no clamp; L < 0: SiTU-GLU).
void swiglu_clamp_split(const float* gu, float* h, int64_t rows, int nf, float limit, void* stream);
/// The same in place: row r's SwiGLU over its gate half (row stride `ld` floats; 0: 2 nf).
void swiglu_clamp_inplace(float* gu, int64_t rows, int nf, float limit, void* stream, int64_t ld = 0);
/// out[t] = sum_j weights[t][j] * Dm[inv[t*k + j]] + shared[t] (shared may be null): expert rows sorted by expert
/// (the prompt path's order) back to their tokens.
void moe_combine_rows(const float* Dm, const int32_t* inv, const float* weights, const float* shared, float* out, int n,
                      int k, int T, void* stream);
/// (A, B, C) -> (B, A, C): token-major <-> head-major for the per-head absorptions.
void permute_abc(const float* src, float* dst, int A, int B, int C, void* stream);
/// LayerNorm per row (rows, cols).
void layer_norm_w(const float* x, const float* w, const float* b, float* y, int rows, int cols, float eps, void* stream);


// ---- MiMo-V2 (mimo2): GQA attention
/// NEOX RoPE in place on n_heads heads of head_dim floats at x + t * row_stride + head_off (token t at position pos0 + t): the
/// first n_rot dims rotate in pairs (i, i + n_rot / 2), theta = pos * base^(-2i / n_rot), the rest are untouched.
void rope_neox_w(float* x, int row_stride, int head_off, int n_heads, int head_dim, int n_rot, float base, int64_t pos0, int T,
                 void* stream, const int64_t* pos_dev = nullptr);
/// Token t's K (n_kv * dk floats at k_off) and V (n_kv * dv at v_off) of row src + t * row_stride into FP16 cache row
/// (pos0 + t) % rows of kc (n_kv * dk halves per row) and vc (n_kv * dv).
void kv_store_w(const float* src, int row_stride, int k_off, int v_off, uint16_t* kc, uint16_t* vc, int64_t rows, int n_kv, int dk,
                int dv, int64_t pos0, int T, void* stream, const int64_t* pos_dev = nullptr);
/// y (ncols, n_out) = x (ncols Q8_1 columns of n_in, native_quantize_q8_1's layout) times a Q8_0 matrix (GGUF blocks, n_out rows
/// of n_in) on int8 tensor cores (sm_80+; MiMo's DFlash drafter, 1..8 columns).  False (nothing launched): not this kernel
/// (n_in not a multiple of 256, w not 16-byte aligned, an older card) - MMVQ then.  `warps` a 16-row tile: 1, 2 or 4 (0: by shape).
bool q8_mma_w(const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream, int warps = 0);
/// The split-K partials gqa_attend_w writes for T tokens.
size_t gqa_attend_scratch_bytes(int H, int dv, int n_chunks, int T);
/// GQA attention per token t (position p = pos0 + t) and query head h (KV head h / (H / n_kv)): softmax over the keys at
/// positions (p - window, p] (window 0: [0, p]; row of position j = j % rows) of scale * q . k, with the head's sink logit
/// (unscaled; sinks may be null) in the denominator, times V.  q: head h of token t at q + t * q_stride + h * dk; out (T, H, dv).
/// n_chunks > 1 splits the keys (partials: gqa_attend_scratch_bytes) and merges; 1 normalizes in place.
/// `noncausal` (DFlash's noise block): every query of the batch sees keys up to the batch's last row (pos0 + T - 1), the lower
/// bound its own window - always the FP32 kernel.
void gqa_attend_w(const float* q, int q_stride, const uint16_t* kc, const uint16_t* vc, int64_t rows, int H, int n_kv, int dk, int dv,
                  int window, const float* sinks, float scale, float* out, float* partial, int n_chunks, int64_t pos0, int T,
                  void* stream, const int64_t* pos_dev = nullptr, bool noncausal = false);
/// gqa_attend_w's prompt path (T queries, one pass) on tensor cores: dk 192, dv 128, 8 or 16 query heads per KV head, sm_80+.
/// Returns false (nothing launched) otherwise.  `warps` 4 or 8 per block (0: STRATA_X_GQA_MMA_WARPS, default 8).  It does
/// not read STRATA_X_GQA_MMA (gqa_attend_w does).
/// gqa_attend_w's decode path (a window's T queries, the keys split in n_chunks: the partials then gqa_merge) on tensor cores:
/// dk 192, dv 128, 8 or 16 query heads per KV head, sm_80+; the position from pos_dev when given.  Writes only the partials;
/// false = not this kernel (the caller runs the FP32 one).  Does not read STRATA_X_GQA_DECODE_MMA (gqa_attend_w does).
/// The split-K partials of T tokens joined (dv 128): max, the sink in the denominator, the normalized output.
void gqa_merge_w(const float* partial, const float* sinks, float* out, int H, int n_chunks, int T, void* stream);
/// A full layer's decode window (T <= 8 rows, 16 heads a KV head) with the window's rows sharing each K / V tile: absolute key
/// chunks (a power of two from each row's own position, n_chunks of them at most), partials + gqa_merge_abs_kernel into out -
/// a row's result does not depend on the window it is in.  `max_pos`: the largest position a launch (or a captured graph's
/// replay) can have - the grid.  False: not this kernel (the caller runs gqa_decode_mma).
bool gqa_decode_shared(const float* q, int q_stride, const uint16_t* kc, const uint16_t* vc, int64_t rows, int H, int n_kv, int dk,
                       int dv, const float* sinks, float scale, float* out, float* partial, int n_chunks, int64_t pos0, int T, void* stream,
                       const int64_t* pos_dev, int max_pos);
bool gqa_decode_mma(const float* q, int q_stride, const uint16_t* kc, const uint16_t* vc, int64_t rows, int H, int n_kv, int dk, int dv,
                    int window, float scale, float* partial, int n_chunks, int64_t pos0, int T, void* stream,
                    const int64_t* pos_dev = nullptr, int kt = 0);   ///< kt: keys a tile, 16 or 32 (0: STRATA_X_GQA_DECODE_KT, 16)
bool gqa_prefill_mma(const float* q, int q_stride, const uint16_t* kc, const uint16_t* vc, int64_t rows, int H, int n_kv, int dk,
                     int dv, int window, const float* sinks, float scale, float* out, int64_t pos0, int T, void* stream, int warps = 0);

}  // namespace strata::x
