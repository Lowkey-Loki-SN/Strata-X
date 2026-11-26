// include/strata/x/dsv41_kernels.hpp - moex: DeepSeek-V4.1-Flash's (deepseek41) attention-side kernels that V4's
// dsv4_kernels do not cover (moex/docs/DSV41_PORT.md): the reference's FP8 / FP4 rounding of cached values, the ratio-2 and
// ratio-1 KV compressors (no overlap, no ape), the store of completed compressed rows, and layer 20's candidate blocks.
//
// The SPECIFICATION is the release's inference/model.py and kernel.py (read 2026-10-18); moex/tools/dsv41/ref/ref_kernels.py
// is the CPU transcription the parity tests compare against.  Conventions as dsv4_kernels.hpp: device pointers, row-major
// (T, ...) activations for T tokens at positions pos0 .. pos0 + T - 1, every call enqueued on `stream` with no allocation
// and no synchronization (graph-capturable); a trailing `pos_dev` (when non-null) holds pos0 for a captured graph.
#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::x {

/// kernel.py act_quant(x, 32, "ue8m0", inplace=True), in place on F32 rows: per block of 32 consecutive values of a row,
/// amax = max(|x|, 1e-4), s = 2^ceil(log2(amax * (1/448))) (the bits, as fast_log2_ceil / fast_pow2), then
/// x = FP8_E4M3(clamp(x / s, +-448)) * s (round to nearest even).  width % 32 == 0.
void dsv41_fq_fp8(float* x, int64_t rows, int width, void* stream);

/// kernel.py fp4_act_quant(x, block, inplace=True), in place on F32 rows: per block of `block` (16 or 32) values,
///   e4m3_scale (the compressed KV, 16-wide): amax = max(amax, 6 * 2^-9), s = FP8_E4M3(amax / 6)
///   otherwise (E8M0: the index q / k, 32-wide): amax = max(amax, 6 * 2^-126), s = 2^ceil(log2(amax * (1/6)))
/// then x = E2M1(clamp(x / s, +-6)) * s (round to nearest even on {0, .5, 1, 1.5, 2, 3, 4, 6}).  width % block == 0.
void dsv41_fq_fp4(float* x, int64_t rows, int width, int block, bool e4m3_scale, void* stream);

/// moex (DSV41_BACKLOG #104): a packed cache row of `width` FP4 values in blocks of `block` - width / 2 bytes of E2M1 codes
/// (value 2i in the low nibble, bit 3 of a code the sign), then width / block scale bytes (E4M3 for the compressed rows'
/// blocks of 16, UE8M0 - the scale's F32 exponent byte - for the index keys' blocks of 32), padded to 16 bytes: 288 for a
/// compressed row (512), 80 for an index key (128).  The values are on that grid already (dsv41_fq_fp4), so the packing loses
/// nothing: a reader's code x scale in F32, rounded to F16, is the F16 cache's half bit for bit.
inline size_t dsv41_pack_row_bytes(int width, int block) { return ((size_t) width / 2 + (size_t) (width / block) + 15) / 16 * 16; }

/// dsv41_fq_fp4 (the same rounding, the floats written back) and, for each row that completes a cache row (as
/// dsv41_store_rows: ratio 1 every position, ratio 2 the odd ones), its packed row into cache row p / ratio of `cache`
/// (cache_rows rows of dsv41_pack_row_bytes(width, block)).  rows: the T rows at positions pos0..
void dsv41_fq_fp4_pack(float* x, int64_t rows, int width, int block, bool e4m3_scale, uint8_t* cache, int cache_rows, int ratio,
                       int64_t pos0, void* stream, const int64_t* pos_dev = nullptr);

/// The KV compressor of a ratio-1 or ratio-2 layer (model.py Compressor.forward; d = 512): token t at p = pos0 + t
///   ratio 1: latent[t] = RMS(akv[t]) * norm_w                                            (every token completes a row)
///   ratio 2: when p is odd, the slots s0 = p - 1 and p, per channel k: c_k = sum_j softmax_j(score_j,k) kv_j,k, then
///            latent[t] = RMS(c) * norm_w; slot s0's (kv, score) is the window's row s0 - pos0 when s0 >= pos0, else the
///            ring's row s0 % ring_rows (written by dsv41_compress_persist); rows of even p are zeroed (no row completes)
/// RMS(v) = v * rsqrt(mean(v^2) + eps), in F32 (the reference pools and normalizes in fp32; it rounds the pooled row to
/// BF16 before the norm, which this does not).  akv / gs / latent: (T, 512); ring_kv / ring_sc: (ring_rows, 512).
void dsv41_compress(int ratio, const float* akv, const float* gs, const float* ring_kv, const float* ring_sc, int ring_rows,
                    const float* norm_w, float eps, float* latent, int64_t pos0, int T, void* stream,
                    const int64_t* pos_dev = nullptr);

/// The ratio-2 compressor's state: ring row p % ring_rows = (akv[t], gs[t]) for the window's tokens, the largest position
/// winning (with T > ring_rows only the last ring_rows land).  Call after dsv41_compress in the same window.
void dsv41_compress_persist(const float* akv, const float* gs, float* ring_kv, float* ring_sc, int ring_rows, int64_t pos0,
                            int T, void* stream, const int64_t* pos_dev = nullptr);

/// The completed rows into a cache: for every token whose position p completes a row (ratio 1: all; ratio 2: odd p),
/// cache row p / ratio = half(rows[t]) (RNE).  rows: (T, width) F32; cache: (cache_rows, width) halves.
void dsv41_store_rows(int ratio, const float* rows, uint16_t* cache, int width, int cache_rows, int64_t pos0, int T,
                      void* stream, const int64_t* pos_dev = nullptr);

/// The bytes of one row of candidate bits: one bit per block of `bsize` rows among `max_rows` (bit b of byte b / 8).
inline int dsv41_cand_bytes(int max_rows, int bsize) { return ((max_rows + bsize - 1) / bsize + 7) / 8; }

/// Layer cand_src's candidate blocks (model.py select_candidate_blocks): per token, n_vis = (p + 1) / ratio visible rows in
/// blocks of `bsize`; the block holding row n_vis - 1 always kept; when the blocks number at most `keep` all are kept, else
/// the `keep` blocks with the highest max score (ties to the lower block).  cand: (T, cand_bytes) bits, 1 = kept, bit b of
/// byte b / 8; every byte of the row is written (0 past the visible blocks).  score: (T, max_rows), rows j >= n_vis ignored.
void dsv41_candidates(const float* score, int max_rows, int ratio, int keep, int bsize, uint8_t* cand, int cand_bytes,
                      int64_t pos0, int T, void* stream, const int64_t* pos_dev = nullptr);

/// The later index sources score only inside the candidates: score[t][j] = -inf where block j / bsize's bit is 0, j < n_vis.
void dsv41_cand_mask(float* score, int max_rows, int ratio, const uint8_t* cand, int cand_bytes, int bsize, int64_t pos0,
                     int T, void* stream, const int64_t* pos_dev = nullptr);

/// moex (DSV41_PLAN §56 C): token t's candidate blocks as a list - list[t * cap + i] the i-th kept visible block, ascending,
/// count[t] how many (dsv41_candidates keeps at most `keep` <= cap; a count above cap stops dsv4_lid_scores_mma's list
/// kernel with a trap rather than scoring a part).
void dsv41_cand_list(const uint8_t* cand, int cand_bytes, int max_rows, int ratio, int bsize, int32_t* list, int32_t* count,
                     int cap, int64_t pos0, int T, void* stream, const int64_t* pos_dev = nullptr);

/// The engram rows' dequant (model.py ParallelEngramEmbedding): rows (n_rows, dim) FP8 E4M3 bytes, scales (n_rows, dim / 32)
/// E8M0 bytes (2^(b - 127)); out (n_rows, dim) F32 = fp8 * scale - exact, the BF16 the reference casts to holds it.  dim % 32 == 0.
void dsv41_engram_dequant(const uint8_t* rows, const uint8_t* scales, float* out, int64_t n_rows, int dim, void* stream);
/// moex: the same from an IQ4_NL table's rows (unsloth's UD-IQ1_M / UD-IQ1_S), row r at rows + r * row_stride, dim / 32 blocks of
/// 18 bytes: out = ggml's dequantize_row_iq4_nl (fp16 d times kvalues_iq4nl[index], exact in F32).  dim % 32 == 0.
void dsv41_engram_dequant_iq4nl(const uint8_t* rows, size_t row_stride, float* out, int64_t n_rows, int dim, void* stream);

/// The engram's write into the streams (model.py Engram.forward) for T tokens: kv (T, (hc + 1) n) = [key (hc, n) | value (n)]
/// from wkv, qk (hc, n) = q_weight * k_weight; per token and copy c of R (T, hc, n):
///   rstd = rsqrt(mean(R_c^2) + eps) * rsqrt(mean(key_c^2) + eps),  dot = sum(R_c * qk_c * key_c) * rstd / sqrt(n),
///   gate = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))),  R_c += gate * value                    (the sums in F32)
void dsv41_engram_gate(float* R, const float* kv, const float* qk, float eps, int n, int hc, int T, void* stream);

/// The imatrix's sums (llama.cpp imatrix's in_sum2, per routed expert): for each of n_exp experts j, rows [bounds[j],
/// bounds[j + 1]) of x - row r at x + (rowmap ? rowmap[r] : r) * ld - add sum_r x[r][c]^2 to acc[experts[j]][c], c < cols.
/// One thread per (expert, column): no atomics, the experts of one call distinct.
void dsv41_imx_sumsq(const float* x, int64_t ld, const int32_t* rowmap, const int32_t* bounds, const int32_t* experts, int n_exp,
                     int cols, float* acc, void* stream);

/// The carried pre-mix at the start of a forward (model.py make_identity_pre_mix): rows of 24 floats (the engine's mHC
/// coefficient layout, pre in the first 4), row t = {1, 0, 0, ...}.
void dsv41_pre_onehot(float* pre, int T, void* stream);

}  // namespace strata::x
