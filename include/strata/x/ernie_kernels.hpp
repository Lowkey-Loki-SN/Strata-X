// include/strata/x/ernie_kernels.hpp - moex: ERNIE-5.5-Flash's kernels (GGUF arch ernie55; moex/docs/ERNIE55_PLAN.md).
//
// The ops of ERNIE-5.5-Flash's forward pass (the reference model moex/tools/ernie55/ref/model.py; vLLM-ERNIE ba322a5 and
// SGLang f4284f2) that strata-x's DeepSeek-V4 and Hy4-flash kernels do not already compute: VHA (the post-mix across
// heads), the two-router routing with its per-expert factor, MultiMax on the logits, the lightning indexer's E4M3 rounding,
// the DSA layers' attended positions (the indexer's choice and the window), mHC's flat-stream factor with the eps outside
// the root, and the index keys' rows.  F32 on the device; `pos_dev` (non-null): the window's first position is read on the
// device (a captured graph's), else pos0.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::x {

/// VHA (vLLM-ERNIE attention.py:136-172): out[t, h, :] = sum_g M[h, g] in[t, g, :] for T tokens of H heads of D values,
/// with M = I + V U^T (H x H, row-major, built at load).  H 64, D a multiple of 64; out must not alias in.  Each output a
/// float sum over g in order.
void ernie_vha_w(const float* in, float* out, const float* M, int T, int H, int D, void* stream);

/// The multi-router MoE's routing (vLLM-ERNIE / SGLang): logits (T, 2E) = [W0 x | W1 x]; score[e] = sigmoid(l0[e]) +
/// sigmoid(l1[e]); the top k of score + bias in descending order (ties: the lower expert); the weights are the scores at
/// the ids over their sum (at least 1e-20), times escale[id] (gate.routed_scaling_factor_param).  E <= 1024, k <= 32.
void ernie_route_w(const float* logits, const float* bias, const float* escale, int E, int k, int32_t* ids, float* wts, int T,
                   void* stream);
/// moex (the ERNIE-5.5 hill-climb): a one-token decode window's routing (ernie_route_w's warp kernel, the same ids and weights)
/// and its doorbell publish (doorbell_publish's copy of the ids and of x, its fences and word) in one launch; false (nothing
/// launched) for T != 1, STRATA_ERNIE_ROUTE_PUBLISH=0 or STRATA_ERNIE_ROUTE_WARP=0
bool ernie_route_publish_w(const float* logits, const float* bias, const float* escale, int E, int k, int32_t* ids, float* wts,
                           const float* x, int n_x, int32_t* m_ids, float* m_x, volatile uint32_t* m_seq, uint32_t* d_cnt,
                           int T, void* stream);
/// route_kernel (one block a token) always - the reference ernie_kernels_check holds the warp kernel to
void ernie_route_block_w(const float* logits, const float* bias, const float* escale, int E, int k, int32_t* ids, float* wts, int T,
                         void* stream);

/// MultiMax's ranges and factors (output.multimax_ranges / output.multimax_ts, 4 each)
struct ErnieMultiMax {
    float r[4] = {0, 0, 0, 0};
    float t[4] = {0, 0, 0, 0};
};
/// The head's piece-wise temperature in place on `rows` rows of V logits, `ld` apart (vLLM-ERNIE nvidia/model.py:97-159):
/// y = x + t0 relu(r0 - x) + t1 relu(x - r1) + t2 relu(r2 - x)^2 + t3 relu(x - r3)^2, left to right, each op rounded.
void ernie_multimax_w(float* logits, int64_t V, int rows, int64_t ld, const ErnieMultiMax& mm, void* stream);

/// The lightning indexer's FP8 rounding (vLLM per_token_group_quant_fp8 with UE8M0 scales), in place on `groups` groups of
/// 128 values: s = 2^ceil(log2(max(amax, 1e-10) / 448)), x = e4m3(clamp(x / s, -448, 448)) * s (round to nearest even).
void ernie_fq_e4m3_w(float* x, int64_t groups, void* stream);

/// The positions a DSA layer's token attends (token t at p = pos0 + t): the indexer's choice - sel's first min(p + 1 -
/// window, top_k) entries (dsv4_lid_select with excl = window: ascending, or every candidate when there are no more than
/// top_k) - then the window max(0, p - window + 1) .. p; counts[t] how many.  cells: max_cells (>= top_k + window) a row.
void ernie_dsa_cells_w(const int32_t* sel, int top_k, int window, int32_t* cells, int max_cells, int32_t* counts, int64_t pos0,
                       int T, void* stream, const int64_t* pos_dev = nullptr);

/// y = x / (sqrt(mean(x^2)) + eps) per row of n (ERNIE's mHC mixer input: the eps outside the root, vLLM-ERNIE
/// mhc_kernels.py:262-317); y may alias x.
void ernie_rms_eo_w(const float* x, float* y, int rows, int n, float eps, void* stream);

/// T rows of `width` F32 values into F16 rows (pos0 + t) of `cache` (round to nearest even): the DSA layers' index keys.
void ernie_store_rows_w(const float* x, uint16_t* cache, int width, int64_t cache_rows, int64_t pos0, int T, void* stream,
                        const int64_t* pos_dev = nullptr);

}  // namespace strata::x
