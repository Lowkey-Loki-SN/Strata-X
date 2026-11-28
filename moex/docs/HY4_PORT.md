# Hy4-flash in strata-glm: the port design (moex, from 2026-11-18)

Tencent's Hy4-flash (`tencent/Hy4-flash`, model type `hy_v4_flash`) from AngelSlim's GGUF builds
(`AngelSlim/Hy4-flash-GGUF`), on this box: two RTX 3080 20 GB, a Xeon Gold 6148 (20 cores, AVX-512 without VNNI),
125 GiB of RAM, one 2 TB NVMe. This file keeps the decisions; the work log goes under "Status" at the end.

## The model

From the model card and `config.json` (`text_config`), checked against the GGUF's own metadata:

- 42 layers at hidden 3072, 120,832 tokens. 220B parameters in all: 155.5B in the backbone, 63.5B in the Engram
  tables, 0.43B in the image encoder, 0.85B in the DFly2 drafter. 11.3B active a text token.
- Residual: identity hyper-connections (iHC), 4 streams. Each half of a layer reads the streams with 4 sigmoid weights
  (`pre`) and writes its output back with 4 more (`post`, times 2.0). There is no stream mixing matrix (no Sinkhorn):
  this is DeepSeek-V4's mHC with the mixing matrix the identity.
- Attention: MLA on every layer (q_lora 1024, kv_lora 512, qk 192 + 64 RoPE, v 256), in a 1:3 pattern:
  - 12 sparse layers (0, 1, 5, 9, ..., 41): 32 heads, RoPE base 10,000,000, attending to the tokens of up to 511 groups
    of 4 picked by a pooled-key indexer ("IndexPool": 16 heads of 128) plus the current token's group;
  - 30 sliding-window layers: 64 heads, RoPE base 10,000, the last 512 positions.
  - Both: a learnable sink per head, an elementwise sigmoid gate on the output (from the layer input), and an RMS norm
    after the output projection ("sandwich" norm).
- MLP: layer 0 dense (13,824); layers 1-41 "LatentMoE": the router and one shared expert (1536) at hidden 3072, 512
  routed experts top-16 in a 1536 latent (`routed_expert_down_proj` 3072 -> 1536 before, `routed_expert_up_proj`
  1536 -> 3072 after), sigmoid scores plus a selection bias, normalized, times 3.998; SwiGLU clamped at 10 on the routed
  experts only. An RMS norm after the MLP.
- Engram before layers 1, 11, 21, 31 (the config's `engram_layer_ids` 2, 12, 22, 32 are 1-based): 2-grams and 3-grams,
  8 hash heads each, 96 values a row, 10.31M rows a head; the hash is DeepSeek-V4.1's (rolling XOR of compressed ids
  times per-layer multipliers, mod per-head primes).
- Positions: interleaved M-RoPE in the release; for text the three axes are equal, so it is plain RoPE on consecutive
  pairs (llama.cpp `LLAMA_ROPE_TYPE_NORM`; vLLM `is_neox_style=False`). No scaling; 1M context.
- Not in the backbone: a 27-layer image encoder (an mmproj, not in this repo), the 5-layer DFly2 drafter (not in the
  GGUFs), and one MTP block (`blk.42` in the GGUF, BF16, 7.5 GB).

## The GGUF set

`AngelSlim/Hy4-flash-GGUF` at commit e018c884: Q4_K_M in two files, 168.35 GB. File 2 is the Engram table alone
(`per_layer_token_embd.weight`, ggml type 43 = F8_E4M3, [96, 660,006,416], 63,360,616,128 bytes) and has the same
SHA-256 in the Q3_K and IQ2_AH sets, so one copy serves all three. File 1 holds everything else: Q4_K experts (down Q6_K
on 19 layers), Q8_0 attention, shared expert and latent projections, F32 norms / router / hc / indexer projection,
BF16 head, Q8_0 embedding, BF16 MTP block.

The public BF16 checkpoint stores the Engram table as BF16 with no row scales. The GGUF's table is not an exact cast:
each row is the release's BF16 row scaled so its largest value is 448 (E4M3's largest) and rounded to E4M3, and the scale is
dropped - the per-head RMS norm right after the lookup cancels it. Measured on the first 2,000 rows of layer 1's module
against model-00004-of-00148.safetensors: the scales 107-324 (median 193), every row's largest code 448; after the norm the
rows differ from the release's by 2.6% (median) to 3.3% (largest), relative - E4M3's own rounding. This file is the same in
the Q4_K_M, Q3_K and IQ2_AH sets, so every one of them carries this quantization of the Engram.

The other sets, from their headers (the first 6 MiB of file 1 each):

| set | routed experts | dense projections | stream mixers, indexer.proj | head | Engram table |
|---|---|---|---|---|---|
| Q4_K_M | Q4_K (down Q6_K on 19 layers) | Q8_0 | F32 | BF16 | F8_E4M3 |
| Q3_K | Q3_K / Q4_K | Q8_0 | F32 | BF16 | F8_E4M3 (the same file) |
| IQ2_AH | IQ2_XXS gate/up, Q4_K down | BF16 (attn_gate, attn_output Q8_0) | F32 | BF16 | F8_E4M3 (the same file) |
| SPARK-4Bit | NVFP4 | Q8_0 | Q8_0 | Q8_0 | Q8_0 |
| SPARK-4Bit-Fast | NVFP4 | NVFP4 (q_a, kv_a, k_b, v_b, indexer Q8_0) | NVFP4 / Q8_0 (indexer.proj) | NVFP4 | Q8_0 |

The embedding is Q8_0 in every set but IQ2_AH (BF16).

## References and how they differ

- **vLLM PR #68296** (jeejeelee/vllm bb9b882a, `vllm/models/hy_v4_flash/`), with `vllm/models/hy_v4/nvidia/hc.py` (iHC)
  and `deepseek_v41/common/engram.py` (the hash kernel): the official serving code. CUDA, Hopper/Blackwell only.
- **SGLang**: main has HY V4 (`hunyuan_v4.py`), the iHC residual ops and the Engram layers, and a Hy4-flash cookbook
  (#51160) that points at the `lmsysorg/sglang:hy4-flash` image. Its Hy4-flash model file is not on main as of
  2026-11-18, so vLLM's is the one read here.
- **llama.cpp PR #32517** (0a054d0b, by Tencent; the same graph as the `patch/` files in the GGUF repo): reads these
  GGUFs. Built CPU-only as the oracle (moex/ref's method: `llama_dump`, then strata-glm's probe mode op by op).

Where they differ, this port follows the GGUF (llama.cpp), and says so:

1. **The indexer's precision.** vLLM rotates the pooled keys and queries with a Walsh-Hadamard transform and quantizes
   both to FP8 before DeepGEMM scores them (the weights take the query's scale). llama.cpp scores in F32 with no
   rotation. In exact arithmetic the two scores are equal (the transform is orthogonal); only the rounding near the
   top-511 cut differs. F32 here.
2. **The Engram table's scale.** vLLM's FP8 checkpoints carry a BF16 scale a row; the GGUF has none (see above). The
   per-head RMS norm right after the lookup removes a row's scale anyway, apart from its eps (1.19e-7).
3. **The indexer's RoPE layout.** vLLM de-interleaves the rotated half and leaves it so; llama.cpp rotates the pairs in
   place. Both do it to queries and keys alike, so the scores are the same up to summation order.
4. **The hash's sign.** vLLM takes `rolling % prime` signed and adds the prime when negative; llama.cpp takes it unsigned.
   The multipliers are below 2^63 / compressed vocab, so `rolling` is never negative for a mapped token: the same rows.
5. **The Engram's norms' eps.** vLLM's FlashEngram builds `norm_key` / `norm_query` as `nn.RMSNorm` with eps unset over
   BF16 tensors, which PyTorch takes as BF16's epsilon (2^-7); llama.cpp (and its converter's folded `engram_gate`) uses
   the F32 epsilon, 1.19e-7, for every Engram norm. This port follows llama.cpp. It matters only for rows whose mean
   square is near 2^-7.
6. **The Engram's look-back past an image.** vLLM stops the look-back at a dead (image) position and pads it and every
   older slot; llama.cpp pads the dead slot alone. This port follows vLLM. Text-only prompts hash the same either way.

## The forward pass (what each op is)

Per token, `R` = 4 streams of 3072 (all four the token's embedding row, Q8_0 dequantized, at the start).

- **Engram** (layers 1, 11, 21, 31, before the attention's read): for each of 16 columns (n-gram size g = 2, 3 x head h),
  row = offset + (XOR over the g tokens ending here of compressed_id x multiplier) mod prime; a position before the start
  takes the compressed pad id. The 16 rows of 96 FP8 values: RMS norm each (eps 1.19e-7) times its head's gamma; one
  Q8_0 projection 1536 -> 5 x 3072 plus a bias (four keys, then the value). For stream c: dot = sum(rms(R_c) * rms(key_c) *
  gate_c) / sqrt(3072) (gate_c folds both norms' weights), gate = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6))); R_c +=
  rms(value) * pre_gamma * (gate + hc_scale_c). A position with no valid column (an image row) adds nothing.
- **iHC read** (each half): mixes = hc_fn (F32 [12288 -> 8]) of rms(flat R, 1e-5); pre = sigmoid(mixes[0:4] * scale[0] +
  base[0:4]) + 1e-6; post = 2.0 * sigmoid(mixes[4:8] * scale[1] + base[4:8]) + 1e-6; x = sum_c pre_c R_c; then the half's
  RMS norm (attn_norm / ffn_norm). **Write**: R_c += post_c * out.
- **Attention**: qr = rms(q_a x) (1024); q = q_b qr (H x 256: 192 nope | 64 rope); kv = kv_a x (576): latent = rms(kv[0:512]),
  k_pe = kv[512:576]; RoPE on q's and k_pe's 64 (consecutive pairs, the layer's base); the cache row [latent | k_pe] (FP16).
  q_abs_h = k_b_h^T q_nope_h (512); score = (q_abs . latent + q_pe . k_pe) / 16 over the layer's cells, with the head's
  sink logit in the softmax's denominator; ctx_h = sum p latent; o_h = v_b_h^T ctx_h (256); o = o * sigmoid(gate x)
  (H x 256); out = rms(wo o) (attn_post_norm).
  - Sliding-window cells: positions p-511 .. p.
  - Sparse cells: the indexer. Per token: k = rms(idx_k x) (128); value = kpool_value k, gate = kpool_gate k (Q8_0 128 x 128).
    Each complete group of 4 positions: pooled = rms(sum_m softmax_m(gate_m + ape_m) value_m) (kpool_norm), then RoPE on its
    last 64 values at the group's first position. Query: iq = idx_q_b qr (16 x 128), RoPE on each head's last 64;
    weights = idx_proj x / sqrt(16 x 128). Token p scores the groups before its own, g < p / 4:
    sum_h w_h relu(iq_h . pooled_g); it keeps the best 511 (all when there are fewer) and its own group's positions
    4 (p / 4) .. p (1 to 4 of them).
- **MLP**: layer 0: down(silu(gate x) * up x), no clamp. Layers 1-41: probs = sigmoid(router x) (F32 [3072 -> 512]); the top 16
  of probs + bias (ties: the lower id); weights = the probs there over their sum (at least 6.1e-5), times 3.998;
  xl = routed_down x (1536); each expert: down(silu(min(gate xl, 10)) * clamp(up xl, -10, 10)); moe = routed_up (sum_j w_j
  expert_j) + shared(x) (no clamp). Then the MLP's RMS norm (post_ffw_norm).
- **Head**: mixes = output_hc_fn (F32 [12288 -> 4]) of rms(flat R, 1e-5); pre = sigmoid(mixes * scale + base) + 1e-6;
  h = rms(sum_c pre_c R_c) (output_norm); logits = output h (BF16, F32 accumulation).

## Decisions

1. **A new arch, `Arch::hy4`** (GGUF `hy_v4_flash`), not a flag on GLM or DeepSeek-V4: it shares GLM-5.3's MLA and NextN
   shape and DeepSeek-V4's stateless caches, and each of the engine's ~150 arch branches is decided once for it.
2. **Reuse, unchanged:** GLM's projections, `heads_mm` (the q absorption and v_b), the sigmoid router (512 / 16 fits its
   1024 / 64 bounds), `swiglu_clamp`, `moe_combine_w`; the mHC kernels `mhc_pre_w` / `mhc_post_w` with iHC's coefficients
   in their layout (mixing matrix = the identity: `mhc_post_w` then computes round(round(x * post) + R_c), llama.cpp's
   add, bit for bit); DeepSeek-V4's `dsv4_hc_head` (the same formula); the VRAM tier, the CPU pool and the RAM copy (the
   GLM path's in-place experts, `native_fmt` at the 1536 latent); the token history and row reads of V4.1's Engram.
3. **New kernels (hy4_kernels.cu):** the iHC coefficients; the cache row store (576); RoPE on consecutive pairs (rows and
   heads); the pool's norm + RoPE; Hy4's selection (511 scored groups + the token's own group, always); the attention over
   576-value keys and 512-value values with sinks, over a cell list (sparse) or the window (sliding); the gate; the Engram
   math.
4. **The latent MoE:** `hp.n_ex` (1536) is the experts' width everywhere the MoE meant `n_embd` (`native_fmt`, the GPU
   experts, the CPU jobs, the combine); the down / up projections sit around it, the shared expert at 3072.
5. **Placement on this box:** dense weights on the GPUs (~7 GB at Q8_0, plus the BF16 head 0.74 GB); the routed experts
   (89.5 GB at Q4_K_M: 2.35 GB a layer with Q6_K downs, 2.04 GB with Q4_K) in VRAM slots and the pinned RAM copy - they
   fit without the SSD tier; the Engram table stays in file 2 and its rows (96 bytes, 64 a token) are read through
   the page cache.
6. **Text first.** The MTP block, the DFly2 drafter and images come after the trunk is right.

## Phases and checks

1. Decode windows (eager), prompts read through windows: every probe point against llama.cpp's dump (the oracle forces its
   inputs, so each op's error is its own), then the tokens of a greedy run.
2. The prompt path (MMQ chunks over the streamed or lent experts): its logits against the windows' on the same prompt.
3. Decode graphs (the doorbell): the same tokens as eager.
4. The server: the `hy4` family (tokenizer pre `hy_v4`, the GGUF's template, the reasoning and tool-call parsers),
   a config and a launcher.
5. Speed: measured against llama.cpp's own CUDA build on the same files.
6. The MTP block as a drafter; then images.

## Speculative decoding and images: what exists (2026-11-18)

- **DFly2** is the model's own drafter (tencent/Hy4-flash, `model.dfly_layers.*`, 87 tensors, 0.85B parameters, all in
  `model-00148-of-00148.safetensors`, 1.69 GB; not in any GGUF of AngelSlim/Hy4-flash-GGUF). Five dense MLA layers; each
  reads the KV cache and the indexer's selection (up to 2,048 positions) of one of the target's sparse layers 25, 29, 33,
  37, 41 (`dfly_target_kv_layers`, `dfly_target_index_layer`). A draft step (vLLM PR #68296, vllm/models/hy_v4/nvidia/
  dfly_speculator.py `HYV4DFlyQueryModel`): the query block [last accepted token, mask (120009) x N]; the first row's
  embedding fused with the target's last hidden state (`mtp_fusion`: anchor + gate * eh_proj(enorm(anchor) |
  hnorm(target_hidden))); the five layers over the block; then per draft position a hidden correction (SwiGLU over
  hidden_norm(h) | embed_norm(embedding of the token before)) and the target's head. vLLM runs it with 5 speculative tokens,
  SGLang with block 6.
- **MTP**: the GGUF's `blk.42` (a sparse attention layer with an indexer, 512 BF16 routed experts and a shared expert,
  `nextn.eh_proj / enorm / hnorm / shared_head_norm`; 6.97 GiB) is not in the BF16 release (its trunk ends at layer 41);
  AngelSlim's SPARK llama.cpp patch drafts with it (`--spec-type draft-mtp`).
- **DSpark** is DeepSeek-V4.1's drafter: nothing of it applies to Hy4-flash.
- **Images**: the release's vision tower (`vit.*`: SigLIP2-So400m adapted, 27 layers, hidden 1,152, 16 heads, patch 16, an
  MLP projector `vit.perceive`) is in AngelSlim/Hy4-flash-GGUF as `mmproj-Hy4-flash-bf16.gguf` (1,044,397,440 bytes, added
  at commit ecd803c8 on 2026-11-18; the Q4_K_M files there keep e018c884's SHA-256s). llama.cpp PR #32517 has the
  encoder (tools/mtmd/models/hy-v4-flash.cpp); vLLM's vision.py / mrope.py / mm_preprocess.py and the Rust processor
  (hy_v4_vl/processor.rs) give the preprocessing and the image rows' M-RoPE positions. Built 2026-11-18 (Status, "images").
- Already on: Strata's prompt lookup (`suffix_draft` 3, windows up to 6) drafts from repeats of the context, lossless, its
  window sizes chosen by the measured cost.
- What a verify window costs here (the full Q4_K_M set, two GPUs, the seed profile, 60% VRAM hits): a 256-token prompt read
  as 32 windows of 8 took 5.0 s (156 ms a window) against 35 ms a single decode token - 4.5x for 8 tokens, the CPU pool's
  distinct experts growing with the window. A drafter's gain on this box therefore rises with the VRAM hit rate: estimated
  ~1.0-1.2x at 60% and ~1.9x at 85% for DFly2 (5 drafts, an assumed 3.5 accepted), before its 0.9-1.7 GB of VRAM.

## Status

- 2026-11-18: Q4_K_M downloading (moex/tools/dsv41/hf_fetch.py, user service `hy4-q4km-download`, ~12 MB/s over Wi-Fi);
  llama.cpp PR #32517 fetched into ~/AI/llama.cpp-mainline (branch hy4-flash-pr32517), worktree
  ~/AI/Strata-X-data/llama.cpp-hy4, CPU-only build in ~/AI/Strata-X-data/build/llama-hy4-cpu-exact (`llama_dump_hy4` beside it).
- 2026-11-18, the engine (src/glm/glm_hy4.cpp, src/glm/hy4_kernels.cu, `Arch::hy4` branches in glm_engine / glm_stage /
  glm_decode / glm_prompt / glm_tiers), checked on a 4-layer slice of the Q4_K_M set (moex/tools/hy4/hy4_slice.py: layers
  0-3 - the dense layer, a sparse MoE layer with the Engram, two sliding-window layers - with the Engram on 16 buckets of
  ~1,000 rows, the first rows of the real table), one GPU:
  - **Op by op against llama.cpp** (its CPU build forcing every input, moex/ref's recorder), 20 tokens: every record within
    1e-3 (most 1e-7) except `ffn_moe_down` at 2e-2 - the VRAM experts' MMVQ rounds the activations as Q8_1 where ggml-cpu
    rounds them as Q8_K: with `--expert-slots 0` (every expert on the CPU pool, ggml-cpu's own kernels) 3e-7, and nothing over
    2e-2. 2,200 tokens (the indexer's top 511 from position 2,048 on, the window's ring past 1,024): the attention's `kqv`
    within 1.1e-3 at every position, `indexer_score` within 6.8e-6, nothing over 2e-2.
  - **Decode graphs**: the same 16 greedy tokens as the eager windows; the first four are llama.cpp's own greedy tokens.
  - **The prompt path** (chunks of 8, 64, 256, 512, 1,024 and auto over the 2,200 tokens): the same top logits within 0.02
    (MMQ against MMVQ); llama.cpp's own logits at the last position, unforced: 45366 7.399, 37172 7.371, 23721 7.153 -
    ours 45366 7.39-7.40, 37172 7.38-7.39, 23721 7.15-7.16.
  - Found on the way: `Stage::heads_mm`'s MMQ bounds were cached by the token count alone, and Hy4's layers have 32 or 64
    heads - a 64-head layer after a 32-head one read 32 stale bounds (NaN). The cache is keyed by the head count too.
  - The decode graphs computed no VRAM expert at all (hit rate 0 of 12,288, every pair on the CPU pool): the grouped MMVQ
    took the K-quants only from one full iteration of blocks (2,048 values) and Hy4's experts are 1,536 wide both ways. The
    grouped kernel's per-row work is the single-column SmallK kernel's, so the K-quants now take any whole number of
    blocks: `mmvq_grouped_parity N_IN` 0 values differ for Q2_K-Q6_K and IQ2_XXS at 256, 512, 768, 1,024, 1,280, 1,536,
    2,048 and 4,096. On the slice (all its experts in VRAM, one GPU): 113 -> 182 tok/s, the same 256 tokens as eager.
- 2026-11-18, two GPUs with an Engram layer on each, on a 5-layer slice (`hy4_slice.py --layers 0-3,11`: the 4-layer slice
  and layer 11, the Engram's second module, as layer 4; `--layer-split 4` puts it and the head on GPU 1, peer off):
  - **Op by op against llama.cpp**, 20 tokens, `--expert-slots 0`: nothing over 2e-2; `engram_out` 5.6e-4 on GPU 0 and
    6.6e-4 on GPU 1, the head's `result_output` 1.9e-3 (as on one GPU).
  - **Decode graphs**: the same 64 tokens as the eager windows; the first four are llama.cpp's greedy tokens.
  - **The prompt path**, 2,200 tokens, each expert through the same GPU kernel (`STRATA_GLM_SHARE=0 STRATA_GLM_CPU_READ=0`):
    one GPU, `--layer-split 4` (twice) and `--layer-split 2` give the same top logits to 4 decimals (15840 7.7407, 7153
    7.3053, 2050 7.0160, 51996 6.9899). With the share on (the default), the pool takes a timed share of the experts, so the
    low bits move from run to run (7153 7.29-7.38 over five runs).
  - This slice is far more sensitive to rounding than the 4-layer one (layer 11 reads a residual stream 4 layers deep):
    llama.cpp gives 15840 7.683, 7153 7.315, 2050 7.003, 51996 6.919 token by token and 15840 7.750, 7153 7.359, 51996
    7.030, 2050 6.926 with the prompt as one batch - its two orders of rounding differ as much as ours (chunks of 256 /
    1,024 / 2,199: 15840 7.707 / 7.751 / 7.741).
- 2026-11-18, the SPARK-4Bit sets' types (NVFP4, and Q8_0 where the Q4_K_M set has F32 or F8_E4M3), before any of their
  files is here:
  - **NVFP4** (ggml type 40: 64 E2M1 codes a block, a UE4M3 scale per 16): native_mmvq.cu's `NVFP4Traits` (Q2_0's layout:
    two threads a block, one Q8_1 block each; the scale decoded as ggml-cpu's `ggml_ue4m3_to_fp32`) and `native_nvfp4_mmvq`,
    one layout (`native_mmvq_multi_kernel<F, 1, WARPS, 1>`) at every width, so the VRAM tier's grouped launch takes the
    1,536-wide experts (24 blocks a row); llama.cpp's MMQ instance for the prompt path (mmq-instance-nvfp4.cu, as shipped);
    the CPU pool through ggml-cpu's own `ggml_vec_dot_nvfp4_q8_0` (AVX2), llama.cpp's kernel. `nvfp4_parity` (new) at 1,536,
    3,072 and 4,096: within 2.2e-8 of the dot mass of ggml's dequantizer times the kernel's own Q8_1 activations (1.2e-3
    with the float activations), a nibble swap caught (0.11-0.18), 2..8 columns bitwise the single column's;
    `mmvq_grouped_parity` NVFP4 bitwise at 1,536, 2,048 and 3,072; `mmq_dense_parity` NVFP4 within 5.9e-4 (as the others).
  - **Q8_0 stream mixers and indexer.proj** go through `Stage::pj` like every quantized projection (llama.cpp rounds the
    input of a quantized matrix to Q8_0; ours to Q8_1, the same int8 codes): F32 ones keep the F32 product. A first try
    widened them to F32 at load instead - 1e-2 off llama.cpp on the head's four mixes, so it was dropped.
  - **A Q8_0 Engram table** (blocks of 32: an FP16 scale, 32 int8 codes; 102 bytes a row): `Hy4Eg::fmt` / `rb`, the rows'
    kernel decodes d x q (dequantize_row_q8_0's value). The prompt chunks' row buffers hold the stage's own Engram slots
    only, and a grown buffer frees the old one (`Stage::dfree`; before, every growth kept the old allocation).
  - Checked on a SPARK-like fixture: moex/tools/hy4/hy4_requant.cpp (new; ggml's to_float, then ggml_quantize_chunk) made
    layers 0-2 of the 4-layer slice into NVFP4 experts, NVFP4 dense projections (q_b, attn_output, attn_gate, the shared
    expert, the latent projections, the dense FFN, engram_kv, indexer q_b), Q8_0 and NVFP4 mixers, Q8_0 indexer.proj, an
    NVFP4 head and a Q8_0 Engram table (35 NVFP4, 26 Q8_0 tensors; 5.0 GB). Against llama.cpp on the same file:
    - op by op, 20 tokens: nothing over 2e-2 with the experts on the CPU pool, in VRAM, and over two GPUs; the head
      1.8e-7, the mixers 1.5-2.9e-7 (the head's 8.5e-5), `engram_rows_normed` 7e-8, `engram_out` 8.1e-4;
      `ffn_moe_down` 1.2e-3 / 2.5e-3 at one step (the same on the CPU pool, llama.cpp's own kernel, as in VRAM).
    - decode graphs: the same 64 tokens as eager on one and two GPUs; the first four are llama.cpp's greedy tokens.
    - the prompt path (2,200 tokens, `STRATA_GLM_SHARE=0 STRATA_GLM_CPU_READ=0`): one GPU, two GPUs and the CPU pool give
      the same logits in one chunk (45366 8.0869, 95017 7.7019); chunks of 256 / 512 / 1,024 and the decode windows
      (`--prefill-chunk -1`) within 0.02 of it - llama.cpp's top four tokens in its order.
  - The level of llama.cpp's logits depends on its KV cache: llama_dump keeps it F32, our cache rows are FP16. On the
    fixture llama.cpp gives 45366 8.1200 with an F32 cache and 8.0660 with an F16 one (ours 8.08-8.09, between); on the
    4-layer Q4_K_M slice 45366 7.3990 / 37172 7.3709 with F32 and 7.3875 / 7.3882 with F16 (ours through windows 7.3838 /
    7.3769). A slice's end-to-end logits are compared within that spread, not to 1e-4.
- 2026-11-18, the IQ2_AH set's types on an IQ2_AH-like fixture (hy4_requant, layers 0-2 of the 4-layer slice: IQ2_XXS
  gate/up and Q4_K down experts - IQ2_XXS quantized with a uniform importance matrix -, BF16 q_a / q_b / kv_a / k_b / v_b /
  engram_kv / shared expert / latent projections / indexer k and pool projections / dense FFN / embedding, Q8_0 attn_gate
  and attn_output, as the set has them; 4.9 GB):
  - Found on the way: Hy4's stages had no BF16 widening scratch (`bf16_f32`: DeepSeek-V4.1's setup allocates it), so a
    prompt chunk's BF16 projections ran the plain GEMV kernel instead of cuBLAS. alloc_state_hy4 now sizes it from the
    stage's BF16 layer weights (at most 48M floats); the head is left out (the prompt path runs it on one row), so the
    Q4_K_M set allocates none.
  - op by op against llama.cpp, 20 tokens: the BF16 projections differ by 1.0-2.2e-3 and the experts' down by 1.6e-2 with
    the experts on the CPU pool - ggml-cpu rounds a BF16 matrix's input to BF16 (its vec_dot_type), ours stays F32.
    `STRATA_GLM_BF16_ACT=1` (A/B only, off by default) rounds the input of our BF16 products to BF16 the same way: then
    every record is within 1.6e-4, the experts' down 3.4e-7 / 1.6e-4, the head 1.2e-7 - llama.cpp's CPU path to the last
    bits. With the experts in VRAM the down stays at 1.9-2.1e-2: our GPU kernels round the activations as Q8_1 (a scale per
    32), ggml-cpu as Q8_K (per 256) - as on the Q4_K_M slice.
  - decode graphs: the same 64 tokens as eager on one and two GPUs; the first four are llama.cpp's greedy tokens.
  - the prompt path (2,200 tokens, fixed paths): one chunk (one GPU = two GPUs), chunks of 256 and the decode windows
    within 0.007 of each other (45366 7.877-7.878, 37172 7.857-7.862).
  - Not run: the SPARK, Q3_K and IQ2_AH sets themselves (none of their files is here).
- 2026-11-18, the decode's GPU time on the 4-layer slice (nsys, eager then graphs, all experts in VRAM): the BF16 head
  (3,072 x 120,832, 742 MB) took 2,615 us a token in gemv_bf16_w_kernel - 284 GB/s, half the token. gemv_bf16_w's
  one-column preload kernel (gemv_bf16_w1_pre_kernel, opt-in for DeepSeek-V4.1 as STRATA_DSV41_PRELOAD=1) sums in the same
  per-thread order, warp sums and warp order: 1,031 us (720 GB/s), the same 256 tokens bit for bit, 182 -> 255 tok/s on the
  slice. It is on for Hy4-flash (`gemv_bf16_w_preload_default`; STRATA_DSV41_PRELOAD=0 turns it off), DeepSeek-V4.1's
  default unchanged. With it the slice's GPU time is ~3.7 ms a token: the head 1.0 ms and the dense Q8_0 projections
  1.0 ms, both at the card's bandwidth, the grouped experts 0.5 ms. llama.cpp's CUDA build on the same slice: pp512 8,124
  tok/s, tg64 272 tok/s.
- 2026-11-18, **DFly2** (src/glm/glm_hy4_dfly.cpp; `--draft-dfly FILE --spec 6`): moex/tools/hy4/dfly_to_gguf.py makes
  hy4-dfly2-q8_0.gguf (0.90 GB: Q8_0 matrices, byte for byte ggml's quantize_row_q8_0_ref; kv_b split per head as
  llama.cpp's DeepSeek converter splits it) from the release's model-00148-of-00148.safetensors (moex/tools/dsv41/hf_fetch.py
  `--file`, SHA-256 checked). The drafter on the last GPU (the layer split moved to 25 so its target layers 25-41 are
  there); the trunk's cells at layers 25 / 29 / 33 / 37 / 41 and the head's normed rows captured into rings of 8 rows by
  position (hy4_snap_rows_w, graph-safe); the step after each window: the block [bonus, mask x 5] at the next position, the
  fusion, the five layers (each writing its block rows into its trunk layer's cache and attending over that layer's cells
  for the last kept row plus the block, causal - hy4_attend_w's per-row counts), the final norm, then per draft the hidden
  correction and the trunk's head, argmax (five syncs: the next correction takes the draft's embedding from the host table).
  - The first block after the chat prompt drafts 6594 198 2345 7779 46866 (p 0.99-1.00): llama.cpp's greedy continuation.
  - Exact: with the experts on the CPU pool (`--expert-slots 0`, placement fixed), 160 greedy tokens with and without the
    drafter are the same tokens (the verify rows are bit for bit the single-token rows: the column-independent kernels).
  - Speed, the CLI, greedy (two GPUs, the seed profile, 72-76% VRAM hits): 256 tokens 31.6 -> 39.2-39.5 tok/s, drafts
    accepted 75-80%, 3.4-3.7 tokens a window, the drafter 11.7-12.1 ms a window. The server at temperature 0.9 (three
    requests): code 26.8 -> 38.2 tok/s, prose 26.8 -> 25.9, a reasoning reply 28.1 -> 33.4; all 25.8 -> 29.8 (+15%).
  - Found on the way: the verify windows' BF16 head ran gemv_bf16_w_kernel at 3.4 ms (2-6 columns); gemv_bf16_wn_pre_kernel
    (the same per-thread order, warp sums and warp order) - the same 160 tokens bit for bit.
  - Open: coupled draft sampling at temperature > 0 (drafts are argmax: prose at 0.9 breaks even), a Q8_0 copy of the head
    for the drafts (5 x 1.08 ms of its 12 ms), its step as a graph.
- 2026-11-18, **images** (AngelSlim's mmproj-Hy4-flash-bf16.gguf, SHA-256 checked, and the release's processor_config.json;
  the run config's "vision" block):
  - **The CPU encoder**, serve/vision_hy4.py (llama.cpp's venv's torch, the server's encoder protocol): the release's
    processor (HYV4VLImageProcessor: smart_resize to multiples of 32 within 4,096 .. 4,194,304 pixels, Python's round;
    torchvision's antialiased bicubic of the uint8 picture; rescale and normalize fused, (byte - 127.5) / 127.5, as vLLM's
    port), the tower (the 16 x 16 patch conv, the 128 x 128 position grid resampled with scale_factor (g + 0.1) / 128, 27
    pre-LN blocks of 16 heads x 72, an MLP of 4,304 with GELU tanh - the release's bias_activation_fusion;
    STRATA_HY4_VISION_GELU=erf takes vLLM's), the projector (RMS norm, the 2 x 2 conv, GELU erf, the 1 x 1 conv, a newline
    row after each merged row: (w / 32 + 1) x (h / 32) rows, no extra begin / end rows: cat_extra_token 0). Against
    llama.cpp's mtmd (PR #32517, its CPU backend) on a 448 x 320 picture (no resize): every row's cosine >= 0.99996, the mean
    relative L2 0.36% (ggml-cpu rounds a BF16 matrix's input to BF16), the newline rows identical.
  - **Preprocessing** where llama.cpp's differs: mtmd resizes this projector's pictures with Lanczos; the release's processor
    and vLLM's port with torch's antialiased bicubic. PIL's bicubic is that filter (a = -0.5) rounded otherwise: within 2
    levels on 0.15-0.8% of the values. serve/vision_hy4_plan.py resize_like_torch (numpy: int16 weights, the horizontal
    pass then the vertical, uint8 between) gives torch's bytes in 19 of 19 cases, so the server plans and patches a picture
    without torch (patches_of: the CPU encoder's input bit for bit).
  - **M-RoPE** (vLLM's get_mrope_input_positions; the config's sections [12, 10, 10], interleaved, for both attention kinds):
    a device table a stage (hy4_kernels.hpp Hy4MPos: the positions [b, e) as (t, h, w), the text after p - shift) read by
    the four rotations (q, k_pe, the indexer's q, the pooled keys at their group's first position) and DFly2's (its block
    at p - shift, vLLM's mrope_delta). Allocated only when the server sends pictures (--image-pad); a request without
    pictures leaves every position its own. Against a port of vLLM's function on the prompts the server sent
    (STRATA_HY4_MPOS_DUMP): one picture (315 rows, shift 294) and two with text between (393 positions, shift 354), the same
    triples. llama.cpp's mtmd gives this projector 1-D positions.
  - **The GPU encoder** (src/glm/hy4_vision.cu, strata/glm/hy4_vision.hpp; "gpu": true): the engine encodes a request's new
    pictures between requests (VENC) in a later GPU's free VRAM, as DeepSeek-V4.1's (glm_vision.cpp; with Hy4-flash loaded
    GPU 1 keeps ~1.8 GB free, a 1080p picture needs 485 MiB, the largest 693 MiB): the block weights in pinned RAM (996 MiB
    from the first picture on), a block streamed while the last computes; pedantic SGEMMs summed in blocks of 128; the
    attention a block of queries at a time, each score row's softmax in shared memory (up to 16,384 patches).
    moex/tools/hy4/vision_gpu_parity.py (hy4_vision_parity): against the CPU encoder's float64 rows the GPU's error is 1.54e-6
    (worst row 1.86e-5), the fp32 CPU encoder's 2.66e-6; two encodes bit for bit the same. Time: 640 x 480 236 ms (the CPU
    encoder 1.8 s), 1920 x 1080 3.6 s (65 s), 2048 x 2048 16.6 s. nsys at 1080p: the attention 75% (its p v 1.65 s, the
    scores 0.59 s, the softmax 0.47 s, memory-bound), the linear layers 0.67 s; the p v as one sum over all keys 9% faster
    and 4x the worst row's error - not taken.
  - **The attention fused** (k_attn_fused, 2026-11-19, on; STRATA_HY4_VENC_FUSED=0: the SGEMM path): a block 128 queries of
    one head, the keys 64 at a time through shared memory, FlashAttention's online softmax in fp32 - the scores never leave
    the SM and the 160 MB score buffer is gone from the workspace. The sums keep the SGEMM path's two levels (a key tile's
    p v summed apart in key order, then added to the output after its rescale), a score its 72 products in order. Against
    one float64 encode of the same patches (moex/tools/hy4/vision_fused_ab.py): 448 x 320 rel 9.34e-7 (worst row 5.90e-6) against the SGEMM path's 9.93e-7
    (6.36e-6), the same 127 ms; 1920 x 1088 2.38e-6 (2.98e-5) against 2.46e-6 (3.47e-5), 3,212 -> 1,973 ms; two encodes
    bit for bit the same. Its rate: ~8.4 TFLOP of attention at 1080p in ~1.47 s, ~5.7 TFLOP/s of the card's
    ~29.8 - one block of 8 warps a SM (74.75 KB of shared memory, 176 registers), a score loop of 12 128-bit shared
    loads a 128 multiply-adds and a p v loop of 4 shuffles and 3 loads a 36: a layout bound at ~2/3 of the peak by the
    shared memory's bandwidth before the latency; more blocks a SM would need fewer outputs a lane (unmeasured).
  - **End to end** (the test server, two GPUs, DFly2 on; a 640 x 488 newspaper front page): the date, price and the four
    headlines read exactly; a crop of the page as a second picture placed where it is on it; a follow-up turn (the prefix
    reused, 0.8-0.9 s) reads the small box at the top left; two new pictures in one request (one VENC); a text request after
    them answers as before. Text with vision on (the three DFly2 requests at 0.9): 31.8 tok/s against 29.8 before.
  - Open: a fused attention kernel for the encoder (the score matrix's traffic and the 72-wide products), video.
- 2026-11-19, **a quality report checked** (the user's report: stray Chinese tokens in English and mixed-language
  replies, a short greeting answered in Chinese, worse than the hosted model). What was measured, on one corpus
  (corpus.txt: 16.7K tokens - Strata's DETAILS.md and a mixed-language conversation, the puzzle's reasoning, an SVG),
  against llama.cpp's own full distributions (llama-perplexity --kl-divergence-base, its CUDA build, the same GGUF):
  - **The engine is not the cause.** Perplexity llama.cpp 34.06, ours 31.24 (chunk 0 16.07 vs 13.99, chunk 1 72.16 vs 69.76):
    ours predicts the text better. KLD(llama.cpp || ours) 0.129 mean (median 0.016), 89.4% the same top token - large, but our
    engine against itself (an identical second run: the prompt path's timed CPU share moves low bits) is 0.027 / 94.0%, and our
    prompt path against our decode windows 0.037: this model amplifies rounding (the top-511 group cut, routing), so two valid
    orders of the same arithmetic part this much. CJK mass at English positions: llama.cpp 0.46% a position, ours x1.025.
  - **A semantic review against vLLM** (PR #68296's hy_v4_flash and the shared hy_v4 base at bb9b882a, a separate reader):
    every module computes vLLM's function - layer order and every eps, iHC, attention (scale 1/16, sinks, gate, window
    p-511..p), the indexer (pooling, norm, RoPE at the group's first position, the best 511 plus the own group), routing
    (sigmoid, the bias for the choice only, 16, renormalized, x 3.998), the Engram's hash and its 12 multipliers, the head -
    apart from three edge cases that cannot move an output (the routing sum's 6.1e-5 floor, sign(0) in the Engram gate, tie
    order) and the Engram eps below.
  - **The Engram's norm_key / norm_query eps**: vLLM's nn.RMSNorm(eps=None) over BF16 tensors takes BF16's eps (2^-7), llama.cpp
    (and so this port) the F32 one. STRATA_HY4_EG_QK_EPS=bf16 (a check's switch) computes vLLM's: perplexity by segment
    (llama.cpp / ours / vLLM's eps) - English docs 16.07 / 13.99 / 12.46, the conversation 25.34 / 25.15 / 36.33, docs before
    and after it 129.8 / 120.6 / 128.1 and 109.1 / 104.6 / 110.9. vLLM's eps predicts worse in 3 of 4 (the conversation 44%
    worse): the F32 eps stays.
  - **Sampling** (the release's T 0.9 over the whole vocabulary): from our distributions, 2.06 CJK tokens are drawn per 1,000
    English tokens; min_p 0.05 (llama.cpp's default) 1.25, top_p 0.95 1.38 - for 6.1% / 3.0% of each position's mass cut. A
    cut removes ~40% of them only: much of the CJK mass sits where the model rates a switch likely. The release's sampling
    stays. The sampler itself: sampler_parity's full-vocabulary frequencies = softmax (TV 0.0145 over 204,800 draws).
  - Found and fixed on the way: Engine::reset() (the CLI tools' - the KLD mode - not the server's) cleared every MLA cache as
    n_ctx x kv_lora x 2 bytes, past a sliding-window layer's ring: cudaMemset refused it unchecked and the prompt path's next
    check failed ("prefill mmq: iota: invalid argument"). Hy4 rows are cleared by kv_rows x 576 x 2 now.
- 2026-11-19, **speed, exact**:
  - **The indexer's scores at depth** (hy4_kernels.cu scores_tile_kernel): the row kernel ran one pool a warp at a time (a
    64-FMA dependent chain each) and read every pooled key once per row. The tile kernel takes 8 pools a warp into shared
    memory once and every row scores them there (8 chains interleaved, each the same sequential fmaf order and shuffle tree):
    bit for bit the row kernel's at 1 / 6 / 64 / 512 rows and 16K-1M positions; the same 64 tokens end to end (prompt path
    and decode, fixed paths). Per layer: a decode token at 800K 2.63 -> 0.69 ms (with the selection, 12 layers 35.8 -> 12.3
    ms), a 6-row window 16.7 -> 4.3 ms, a 512-row prompt sub-batch at 800K 1,420 -> 404 ms, at 131K 243 -> 63 ms.
    STRATA_HY4_SCORES_TILE=0 keeps the row kernel.
  - **Coupled DFly2 drafts** (on by default; STRATA_SPEC_COUPLED=0 or a request's spec_coupled 0 keeps argmax): a draft is
    sampled with the request's chain and the noise of the row that verifies it (cell anchor + j - 1); coupled_merge_kernel
    takes the Gumbel-max pick a whole-vocabulary target uses (it drew an unrelated inverse-CDF uniform before). The prose
    request at 0.9: 228 drafted / 129 kept -> 221 / 136, 20.9 -> 21.8 tok/s; three requests 23.8 -> 23.7 tok/s (noise).
  - ~~Found: a verify window's rows are not bit for bit a single token's once experts run from VRAM~~ - wrong, see the next
    entry: the runs that parted had different expert layouts.
  - **Deployed speeds** (2026-11-19, 256K context, --no-adaptive, the learned profile): prefill 16,684 tokens 361 tok/s, 4,579
    tokens 236 tok/s (a chunk costs ~18-23 s: ~15,400 experts streamed over each card's link in turn); decode at 0.9 plain 19.5
    tok/s, DFly2 23.8 (code 26.5-26.9, prose 19.7-20.5, reasoning 26.0-27.3). The decode token: the CPU pool 27.5 of 47.4 ms at
    a 36.9% hit rate - RAM-bandwidth-bound (1.82 GB a token at 64 GB/s).
  - **1M context** (the model's native 1,048,576; strata-hy4.json since 2026-11-19): the decode state 15.4 GB, the expert slots
    6,454 -> 2,975; a short-context decode (greedy, adaptive) 21.9 tok/s at a 45.5% hit rate. At 800K deep the indexer adds 12.3
    ms a token (est. ~17 tok/s plain), a 5K-token turn's indexer ~47 s (was ~166 s).
  - Next, by what they would buy: the KV rows on demand or the far ones in RAM (sparse attention reads ~2,052 positions a layer
    a token; 15.4 GB of VRAM back to the experts lifts both decode and prefill), the scores as a register-tiled kernel (~1
    TFLOP/s now), a chunk's second stage streaming while the first computes.
- 2026-11-19 (later), **DFly2 is exact; the indexer register-tiled**:
  - **Drafting does not change a token.** The CLI, greedy, the chat prompt + 200 tokens, the learned profile, `--no-adaptive`,
    fixed paths (`STRATA_GLM_SHARE=0 STRATA_GLM_CPU_READ=0`) and the same layout for both arms (`--split 24 --expert-slots 5800`,
    `STRATA_GLM_STAGE_SLOTS=3300,2500`): plain and DFly2 give the same 200 tokens (71 verify windows, 129 of 175 drafts kept;
    20.6 -> 26.3 tok/s). Two plain runs at one layout are the same 200 tokens. What parted the earlier runs is the layout:
    loading the drafter (0.9 GB on GPU 1) moves the auto split from 24 to 25 and ~400 slots off GPU 1, so other experts run
    on the CPU pool (activations rounded as Q8_K) instead of VRAM (Q8_1); two plain runs with 6,000 and 5,800 slots part at
    token 45 the same way. That is rounding of two exact kernels, with or without drafts (the adaptive tier and the prompt
    path's timed share move it from run to run too). With the new indexer below, DFly2 again gives the same 200 tokens
    (70 windows, 130 of 172 drafts kept: the drafting policy's timing moved the windows, not one token).
  - **Sampled, at the deployed layout** (a test server with the deployed config - 1M context, the auto layout with the
    drafter, the release's T 0.9 over the whole vocabulary - the tier frozen and the prompt path's paths fixed; the same
    seeded request plain (spec_window 1), with DFly2's coupled drafts, then plain again): the prose request 1,353 characters
    and the code request 1,154, each the same text in all three (DFly2 kept 128 of 211 and 204 of 271 drafts; code 18.6 ->
    25.0 tok/s). The earlier server comparison that parted (above) had the prompt path's timed CPU share on.
  - **The scores register-tiled** (hy4_kernels.cu scores_mm_kernel; STRATA_HY4_SCORES_MM=0 keeps the tile kernel): a block
    takes up to 8 rows x 64-256 pools; a thread 8 heads x 4-8 pools of one half, the values 16 at a time through shared memory,
    transposed so each step is two 16-byte reads of queries and one or two of keys for 32-64 products. Every (row, head, half,
    pool) is still one thread's sequential fmaf chain from 0.0f, joined lo + hi, relu, weighted and summed in the row
    kernel's shuffle tree (h + (h ^ 8), then ^ 4, ^ 2, ^ 1), so every score is its bits. The block shape by the rows, each the
    fastest of eight on this box at 819,200 positions (1 row 1 x 256 pools, 2: 2 x 128, 3: 3 x 128, 4: 4 x 64, 5-6: 6 x 64,
    7 and more: 8 x 128).
  - **The selection by a sampled threshold** (select_sampled_kernel; STRATA_HY4_SELECT_SAMPLED=0 keeps the radix passes
    alone): from 32,768 groups (131,072 positions), tau = the K-th largest of 4,096 evenly spaced keys, K aiming at 2,000
    keys >= tau; those keys and their groups go into shared memory, and with n_top .. 4,096 of them the best 511 there - key
    descending, group ascending, the radix passes' tie rule - are the best of all, sorted and written in group order. Else
    (fewer than 511 or more than 4,096) the radix passes: a miss costs time, never a cell. A row's scores are read once
    instead of four times.
  - **Checked**: a harness of the reference kernels verbatim against the new ones (17 cases: 1-512 rows, 4-1,048,569 positions,
    the device position, ties and zero pools), then the engine's own launchers (hy4_scores_w + hy4_select_w) under each
    switch on 20 cases with a cluster of strong groups as well: the new defaults give the reference's scores and cells byte
    for byte (637 MB). End to end (fixed layout and paths): the 16K prompt + 100 greedy tokens the same tokens with the old
    kernels and the new (prefill 36.8 / 36.6 s, decode 21.0 tok/s: at 16K the indexer is small); DFly2 above.
  - **Per layer** (RTX 3080, the engine's launchers, scores + selection; the deployed 9569a63f -> now): a decode token at
    819,200 1.123 -> 0.273 ms (12 layers 13.5 -> 3.3 ms a token), a 6-row verify window 4.48 -> 0.57 ms (53.7 -> 6.8 ms a
    window: before, a DFly2 window at 800K paid ~19 ms of indexer per kept token, more than plain decode's 13.5), 8 rows
    6.12 -> 0.68 ms, at 1,048,575 1 row 1.44 -> 0.32 ms, a 512-row prompt sub-batch at 819,200 414 -> 38.0 ms (12 layers 5.0
    -> 0.46 s per 512 tokens) and at 300,000 144 -> 13.9 ms. The scores alone: 1 row at 819,200 0.715 -> 0.162 ms (the
    keys' 105 MB at ~650 GB/s, near the card's 760), 512 rows 404 -> 39.4 ms (2.1e11 multiply-adds: ~10.9 TFLOP/s).
  - At 800K deep, a token then costs ~10 ms less than before (est. ~17 -> ~20 tok/s plain, from the short-context token
    plus the indexer: attention reads ~2,052 cells a layer at any depth); not run end to end at that depth.
- 2026-11-19 (later), **the prompt path, profiled and two exact fixes**:
  - **Where an 8K-token read goes** (nsys, one chunk, fixed layout and paths, 22.0 s): the stages run one after the other and
    each was its GPU's kernels - GPU 0 13.3 s (8.5 s of kernels, 84-98% busy; 41 GB streamed at 6.2 GB/s over its x8 link),
    GPU 1 7.9 s (6.6 s, 33 GB at 8.0 GB/s, x16). Of the kernels (15.2 s both GPUs): heads_q8_0_kernel 4.96 s (k_b, the
    per-head absorption), attend_kernel 4.87 s, MMQ ~3 s. And both GPUs idle ~3.5 s before the first layer: the host
    gathering the chunk's Engram rows (0.47 s of it the first call's pinned allocations).
  - **k_b in one launch** (glm_kernels.cu heads_q8_0_tiled_kernel; STRATA_GLM_HEADS_TILED=0 keeps the loop): heads_q8_0 ran its
    decode kernel 8 columns at a time - a 512-row sub-batch 64 launches, the heads' weights read by each. Hy4-flash's k_b
    rows are its 192 nope values (6 blocks: lanes 0-15 of the kernel take 2 groups, 16-31 one). The tiled kernel takes 4 rows
    x 4 columns a thread out of shared memory and makes each output as the kernel's lane 0 ends with it: each lane's chain
    over its groups (FMUL(d_w, d_x) then FFMA, as the kernel compiles it; the dot's int -> float by the exact magic add),
    then the 32 partials in warp_sum's butterfly order. hq8.cu (scratchpad): 16 shapes (128 / 192 / 256 wide, 9 .. 4,096
    columns, zero blocks), 0 values differ; 64 heads x 512 x 192 at 512 columns 8.38 -> 0.85 ms (GPU 0's 2.78 s -> 0.30 s).
  - **The Engram rows asked for first** (glm_hy4.cpp hy4_eg_rows; STRATA_HY4_EG_WILLNEED_PROMPT=0 keeps the faults): a prompt
    chunk's threads copied rows as found, ~32 page faults in flight; now each thread computes its rows, asks for their pages
    (MADV_WILLNEED, as a decode window's already did), then all copy - the same bytes. With the table's pages dropped from
    the page cache first: the 8K read 17.8 -> 16.7 s, the same 16 tokens.
  - **End to end** (the 16K prompt + 100 greedy tokens, fixed layout and paths): every new path against every old one (the
    four switches) - prefill 37.7 -> 28.4 s (424 -> 563 tok/s), the same 100 tokens. The 8K read 22.0 -> 16.7 s: GPU 0 is now
    link-bound (H2D 85-99% busy), GPU 1 compute-bound (attention 2.26 of its 4.85 s); ~1.8 s of Engram gathering remains
    (the drive's random reads, cold).
  - Pinned build-dsv41/strata-glm.hy4 = 24c24ed7 (9569a63f kept beside it).
- 2026-11-19 (later), **at depth, end to end** (a test server on port 18234 with the deployed strata-hy4.json - 1M context, the
  adaptive tier, DFly2, T 0.9 - but no profile save; reasoning off; the prompt 29 of the engine's own source files):
  - Turn 1, a 470,954-token read: 906 s (520 tok/s over the whole read, ~770 tok/s at the start, ~410-500 past 250K: the
    stages pipelined across chunks of 7,680), then 198 tokens at **23.8 tok/s at 471K deep** (DFly2 103 of 161 kept, 51.6% VRAM
    hits) - about the short-context rate. The answer summarized the code correctly.
  - Turns 2 and 3 (3,828 new tokens, then 26; the 470,948-token prefix reused): read in 45.4 s and 1.4 s; decode 10.3 and 10.9
    tok/s, and a fresh short conversation on the same server afterwards 9.4-12.2 - at 54-58% hits, the answers correct.
  - Why: **the drive.** The model's NVMe (2 TB, 99% full) had a slow spell then - sysstat's 10-minute rows, one in the
    spell: 4,011 requests a second at an average wait of 75 ms and a queue of 301 (an earlier half hour: 1.5-5.1 ms).
    Every prompt token takes 64 Engram rows and a decode token 64 more, from a 63 GB table mostly outside the page
    cache: a slow drive stalls a chunk's gathering (a 40K read in the spell: 30-60 s between chunks, both GPUs done and
    waiting) and every decode window. The same three turns at 250K with the drive healthy (diagnostics on, the engine's
    breakdown): 217,741 tokens read in 523 s (416 tok/s; 32,768 reused), then decode 25.5 / 22.9 / 24.6 tok/s, the
    follow-up's 3,627 tokens in 14.7 s - a token 12.1-13.4 ms waiting for the routing, 19.4-22.2 ms in the CPU pool
    (62-64 GB/s), 7.5 ms the rest. The same cold 16K read gave 28 s and, minutes later in a spell, 104-112 s (the
    gathering's own timing: a step 1.4-2.2 s when healthy).
  - **The Engram row cache** (Hy4Eg::RowCache; STRATA_HY4_EG_CACHE_MIB, default 2048, 0 off): the rows read lately, row by row
    (4-way set-associative, 4,096 striped locks), in front of the table. The page cache keeps a 4 KB page for each 96-byte row
    the hashed order scatters - ~1 useful row a page - so a long session's ~15M rows did not fit its ~24 GB and repeats came
    off the drive again; 2 GiB hold ~20M rows. Misses are copied out of the mapping as before: the same bytes (the 16K read's
    100 tokens the same with it off, on and at 128 MiB). In one 16K read it holds 15-25% of the rows (repeated n-grams); a
    session's follow-ups share far more. A spin lock a set, first tried, stalled whenever a holder was preempted. The serve
    line says the share: "the Engram's rows: N looked up, X% from the row cache"; STRATA_HY4_EG_TRACE=1 times a chunk's steps.
  - **The slow spells, measured** (the 8K read six times, the prompt's pages asked for first (ask) or faulted in by the 32
    threads (fault) alternating, the drive sampled every second): slow 2 of 3 ask, 1 of 3 fault (76-82 s against 18-19), each
    time GPU 1's layer-31 gather (one Engram layer, 131K rows) at its stage's start: ~60 s, its copy 0.3 s. In a spell the asked
    pages fill the drive's whole queue (~1,040 requests, ~1,020 ms a read - any other program's read waits behind them), the
    faults keep ~170 there (31 ms a read); healthy, both 170-675 at 7-9 ms. Asking first gains ~1 s an 8K chunk when healthy
    and the same stall either way in a spell, so it is opt-in now (STRATA_HY4_EG_WILLNEED_PROMPT=1): the drive's queue stays
    bounded, as DeepSeek-V4.1's drive guard keeps it.
  - **The later stages' rows gathered ahead** (STRATA_HY4_EG_AHEAD=0 off): once the first stage has its chunk's rows, a task
    gathers the later stages' (layer 31's) into the row cache while the first stage runs the chunk; the later stage waits for
    it, then copies them out of the cache - the same bytes (the 16K read's 100 tokens the same). Its reads no longer wait for
    GPU 0's whole stage. Cold 16K reads: 4 of 4 at 30-32 s with it, 2 of 3 stalled (85, 122 s) without - in a spell it still
    cannot hide everything (one run of 8 threads: 79 s, and its decode 11.7 tok/s at 3.9 s of Engram reads for 100 tokens).
  - **The first stage's next chunk gathered ahead too** (the prompt loop's next-chunk hint, DeepSeek-V4.1's #51 fields): once
    the first stage has its chunk's rows, a task gathers its own rows of the next chunk into the row cache while this chunk's
    layers run (their ids noted early: the same prompt tokens); its next gathering waits for it, then copies out of the cache.
    Of a call's chunks only the first waits for the drive now. Cold 16K reads, three in a row: 29.5 / 29.8 / 32.1 s, the same
    100 tokens (58% of the run's 1.66M rows from the cache). The third met a spell in its decode instead (3.9 s of Engram reads
    for 100 tokens, 11.8 tok/s): a decode token's 64 rows are known only once its token is - the cache is its only shield.
  - **The CPU pool's switches for Hy4-flash** (strata-hy4.json "env": STRATA_POOL_SPREAD=1, STRATA_POOL_PIPELINE=1,
    STRATA_POOL_TASKS=252 - DeepSeek-V4.1's, DSV41_PLAN §29 / §40 / #94; TASKS now read for GGUF experts too, glm_tiers.cpp):
    decode 20.81 -> 22.07 tok/s (+6%), the pool 65.9 -> 73.5 GB/s, the same 200 tokens and the same 8K read's tokens (the
    chat prompt, greedy, one layout, fixed paths). STRATA_DSV41_HUGE_PIN=1 on top: 22.20 tok/s, 74.0 GB/s, the RAM copy
    reserved in ~15 s instead of ~27 - no measured gain past noise, so not taken (the compaction risk of §44 / #102).
  - **The row cache in a session** (the test server, the deployed config and switches, a 40K read then the 3,627-token
    follow-up and a one-line one): 61.6% / 56.7% / 29.6% of the rows from the cache (3.2M / 299K / 7,760 looked up; the first
    turn's count takes each gathered-ahead layer-31 row twice); decode 20.9 / 27.4 / 26.1 tok/s at 0.70-2.47 ms of Engram rows
    a token (62.8 ms in a drive spell, 9.4 tok/s), the CPU pool 69-72 GB/s with the pool switches.
  - **The decode token's GPU work** (nsys with every graph node traced, plain decode at the 1M config): ~25 ms of kernels a
    token on the two GPUs besides their waits for the CPU pool's rows - the dense Q8_0 projections 9.9 ms (~6.3 GB at ~640 GB/s,
    near the card's 760), the VRAM hits 3.7, the per-head k_b / v_b 2.56 (one row a warp: ~184 GB/s), the routers' F32 rows
    1.92 (the general kernel: ~134 GB/s), rms_norm 1.0, the attention 0.9. Two exact fixes:
    - heads_q8_0_rows_kernel (glm_kernels.cu; STRATA_GLM_HEADS_ROWS=0 off): 4 rows a warp, each row's lane chains and
      warp_sum as the one-row kernel's, the activations' groups loaded once for the four (heads_q8_0_decode_parity.cu: 1-8
      columns on the k_b / v_b shapes, 0 values differ; 64 heads x 512 x 192 at one column 39.4 -> 16.0 us, six 104.6 -> 34.1).
    - The routers' F32 one-column preload on for Hy4-flash (gemv_f32_w_preload_default, as the BF16 head's: the general
      kernel's per-thread order, warp sums and warp order).
    - At the pinned layout (the chat prompt + 200 greedy tokens): the same 200 tokens plain and with DFly2; plain 20.56 ->
      21.03 tok/s, DFly2 26.34 -> 28.16 (its verify windows are the 2-6 column case).
  - **Quality, the whole corpus** (corpus.txt, 16.7K tokens - Strata's docs and a mixed-language conversation, the
    puzzle's reasoning, an SVG - against llama.cpp's own full distributions, at one layout and fixed
    paths, each chunk's first half through the prompt path and the scored half through decode windows of 8): the binary before
    this day's work (9569a63f) and the pinned one now save the same 1.98 GB of distributions byte for byte - every scored
    position's, the k_b kernel, the indexer, the row cache and the gathering ahead all in play. KLD against llama.cpp 0.1378
    (median 0.0168), 88.7% the same top token, perplexity 31.67 against llama.cpp's 33.76 (at the earlier layout 0.129 and
    31.24: the layout moves rounding, as above).
  - **What the 1M context's KV costs a short decode** (the deployed arguments - adaptive tier, auto layout, DFly2 - greedy, the
    chat prompt + 400 tokens, the pool switches): --max-context 262144: 5,436 slots, 56.1% hits, 36.22 tok/s; 1048576: 2,801
    slots, 42.8% hits, 35.65 tok/s - twice the slots buy 1.6%. KV on demand (the design: the slot array and the sparse layers'
    caches on CUDA VMM, the lend region's top slots retired into the caches as the context grows) is not worth its risk for
    decode; at most for a short context's prompt reads (fewer experts streamed).
  - **... and a read** (the same arguments, a 48K-token prompt + 32 tokens): --max-context 65536 (6,095 slots, chunks of 9,600)
    63.3 s, 758 tok/s; 1048576 (2,801 slots, chunks of 7,680 - --prefill-chunk 16384 still gets 7,680: the KV leaves the
    prompt path no more slots to lend) 79.6 s, 603 tok/s - the static 1M KV costs a read ~26% while the conversation is short,
    less as it fills (at 800K ~12.3 of its 15.4 GB are rows in use). Chunks of 16,384 where they fit (65536): 36% fewer experts
    streamed (55,333 against 86,185) and no faster (64.8 s: three chunks keep the two stages busy less than five). So KV on
    demand would buy a short conversation's reads ~20-25% and a decode ~1.6% - the remaining structural item (built since:
    KV on demand, below).
  - **The layer split for reads** (the same, 128 tokens): auto (23 on GPU 0: 1,419 + 1,382 slots, chunks of 7,680) 149.7 s - a
    drive spell, 40 s stalled between chunks 2 and 3 - and 27.3 tok/s; 21 (1,781 + 1,020, chunks of 6,000: GPU 1's head,
    drafter and the moved layers' KV leave it fewer slots to lend) 229.9 s, 25.0; 19 (1,860 + 942, chunks of 5,376) 93.8 s,
    29.4. One run an arm under the spells decides nothing; the auto split stays. Decode alone, 400 greedy tokens, two runs an
    arm alternating: auto 35.08 / 29.69 tok/s (42.1 / 43.3% hits), 19 32.16 / 30.68 (40.7 / 42.7%) - no gain past the runs' own
    18% spread (DFly2's acceptance and the adaptive tier).
  - **KV on demand** (STRATA_HY4_KV_GROW=1, 2026-11-19): the 12 sparse layers' caches and pooled keys on CUDA VMM - the whole
    context's addresses reserved at the start (cuMemAddressReserve), 2 MiB granules created, mapped and zeroed as the
    positions arrive (65,536 at the start, steps of 65,536: STRATA_HY4_KV_GROW_INIT / _STEP), by hy4_kv_ensure before a
    prompt read and before a decode window - and the expert slots on VMM too: until the caches need it, their memory holds
    experts (at the start 3,457 + 2,805 slots against 1,419 + 1,350 with the 1M caches allocated whole). A stage short of
    memory for a growth gives its top slots first (hy4_retire_slots: the swaps in flight landed, every GPU idle, an expert
    held only in VRAM copied to a free exchange blob, the residency republished before the next window - the CPU computes
    it), so that its plan's reserve stays free: 700 MiB on GPU 0, 2,048 on GPU 1, the display's and the image encoder's (a
    first build kept 256 MiB and would have used them up). A request that starts well short of the mapped rows - a new
    conversation, a compacted one - gives the rows past its need back (a step at least, never below the first rows:
    hy4_kv_release, from run_request once the reused prefix is known), and the stages take their slots back, empty, as far
    as their reserves allow; the tier's first-use fills take them. Nothing reads a row past the request's own positions
    (the attention and the selection read positions before the token's own), so a released row comes back zeroed.
    - Exact: the corpus (as above: one layout, fixed paths), its caches grown 17 times from 1,024 positions in steps of 256
      through the chunks, saves the same 1.98 GB of distributions as the pinned binary, byte for byte.
    - One engine process, four requests (the 48K prompt + 400 greedy tokens, the chat prompt + 200, the same two again),
      the caches from 4,096 positions in steps of 256 so that every read and decode grows them: grown to 48,084 at the
      read's start (GPU 0 gave 136 slots, GPU 1 111), twice in its decode; released to 4,096 by the short request (118 slots
      back); grown again (83 + 60 given); released again (143 back). The replies coherent; the free VRAM after each request
      640 / 2,010, 706 / 2,050, 704 / 2,054 and 706 / 2,050 MiB - at READY, before any growth, 474 / 1,827: the decode
      graphs and buffers allocated after the plan had used part of the reserves, which the first growth restored. Reads
      60.8 and 53.0 s, decode 38.7 and 40.2 tok/s at 48K, 35.9 and 37.9 short.
    - At depth (the test server, the run config; the 250K conversation of the depth runs above, three turns): the caches
      grew to 250,596 positions at the read (GPU 0 gave 413 slots, GPU 1 308) and to 316,132 at the follow-up (144 + 31);
      the read 250,509 tokens in 399.5 s (627 tok/s) against 217,741 in 523.1 s (416 tok/s) without it, a 3.6K follow-up
      13.5 s against 14.7, decode 25.4 / 30.8 / 32.8 tok/s against 25.5 / 22.9 / 24.6 (2,778 + 2,313 slots at 316K
      against 2,756 with the 1M caches whole); the replies right about the code, the free VRAM after 706 / 2,051 MiB. The
      drive had a slow spell in the read (~2 minutes at ~1,000 reads/s, 22-50 ms a read).
    - The deployed arguments (1M, adaptive tier, DFly2, the pool switches), its defaults against off, alternating, each run
      one engine process: the 48K prompt + 400 greedy tokens, then the chat prompt + 200; the drive sampled every 5 s:

      | run | slots | 48K read | decode at 48K | short decode | the drive |
      |---|---|---|---|---|---|
      | off | 1,419 + 1,350 | 82.9 s | 29.12 tok/s | 28.91 | healthy |
      | on | 3,338 + 2,720 | 60.3 s | 34.56 | 35.67 | a spell in the load only |
      | off | 1,419 + 1,349 | 136.8 s | 27.82 | 29.95 | a spell in the read (~1,000 reads/s, 7-58 ms) |
      | on | 3,338 + 2,724 | 60.5 s | 34.68 | 34.53 | a spell in the load only |

      Reads 27% faster than off's clean run, decode 19-25% faster at 48K and 15-23% on the short request (twice the slots:
      the CPU's share of a token's experts down by a third). The starts with it on took 137-139 s against 69: both loads
      met a drive spell (488 reads/s at ~430 ms for a minute; 100-102 s against 33.7 to load the experts) - the runs
      before, the same switch, loaded in 37.4 s (more slots to fill: +3.7 s).
    - The server with the run config (a test port, no profile save): thinking on, a stream with thinking off, a tool call,
      the Anthropic API, and a 1920 x 1080 picture (68 x 120 patches on GPU 1 in 3.2 s; 2,107 prompt tokens, the page read
      right) - the reserves leave the image encoder its room.
    - On in strata-hy4.json since 2026-11-19 ("env": STRATA_HY4_KV_GROW=1; 0 or the key's removal: the caches whole, as
      before).
  - **Against the box's limits** (2026-11-19; a Xeon Gold 6148 with 128 GiB, two RTX 3080 20 GB - GPU 0 on x8, GPU 1 on x16 -
    and the NVMe; the RAM's read bandwidth measured here: 73.5-76.0 GB/s, 10 to 40 threads summing a 16 GiB buffer with
    AVX-512 loads, the best of five passes):

    | where the time goes | bound by | measured | the limit | share |
    |---|---|---|---|---|
    | a decode token's experts on the CPU | RAM reads | 69-74 GB/s (the pool with its switches) | 73.5-76.0 GB/s | 91-97% |
    | a decode token's dense projections | VRAM reads | ~640 GB/s (nsys, every graph node) | 760 GB/s (the card) | ~84% |
    | a prompt chunk's streamed experts, GPU 0 | its x8 link | 6.2 GB/s | 6.25 GB/s (measured, MOEX.md) | ~99% |
    | ... GPU 1 | its kernels (the stages run in turn) | 84-98% busy | | |
    | an Engram row not in RAM | the drive | 0.22 ms a read healthy | 15-57 ms in a slow spell (a flat ~1,000 reads/s) | |

    So a decode token's cost is its CPU experts' bytes over the RAM's bandwidth plus the GPU's serial part, both near their
    limits; what moves it is how many experts VRAM holds (1.58 GB a token on the CPU with 2,756 slots, 1.03-1.07 GB with
    6,220: KV on demand while the context is short). A read is GPU 0's link and GPU 1's kernels; what moves it is fewer
    experts streamed (KV on demand's bigger chunks and more resident experts) and GPU 1's attention kernel (open: an exact
    rewrite estimated at ~20% fewer instructions for that kernel, unmeasured). A decode window waits for the slowest of its
    Engram rows; in a drive spell a row outside the row cache and the page cache costs the window 15-57 ms, and nothing
    exact avoids it: the table (63 GB) does not fit in RAM beside the experts' RAM copy (80 GiB of 128).
  - **A prompt chunk against its limits** (nsys, an 8K chunk, the earlier layout): GPU 0 - its 23 layers' streamed experts
    41.0 GB in 6.59 s (6.2 GB/s, ~99% of its x8 link) beside 6.13 s of kernels, the two overlapping to ~7.5 s; GPU 1 - 33.4
    GB in 4.19 s (8.0 GB/s) beside 4.85 s of kernels, ~5.5 s. A chunk is GPU 0's stage then GPU 1's, the next chunk's GPU 0
    stage beside this one's GPU 1 stage, so a long read runs at GPU 0's pace, within ~15% of the larger of its link and its
    kernels; KV on demand lowers both GPUs' streamed bytes (more experts resident, bigger chunks). attend_kernel is 47% of
    either GPU's kernels: its exact form fixes each lane's 18-term chains, the butterfly and the online softmax's order, and
    four bit-for-bit variants of it (moex/tools/hy4/attend_parity.cu: every output of the sparse and the window layers, the
    prompt's and the decode's modes, the device position, DFly2's counts) were no faster on GPU 1 - two or four cells a step
    (their dot products and sums interleaved: 150 / 177 registers, one block a SM: x0.49-0.80), two cells capped at 128
    registers (x0.96-1.09), two 16-cell tiles with the next read by cp.async while the current is computed (x0.73-1.07).
    The update compiles as fma(corr, acc, pr * kv), so a settled maximum (corr = 1) saves no multiply either. What is left
    for reads was the stages' balance (GPU 0 ~7.5 s, GPU 1 ~5.5 s an 8K chunk at that layout). Measured with KV on demand
    (2026-11-19, the 48K prompt + 400 greedy tokens then the chat prompt + 200, two runs an arm alternating): the automatic
    split - 22 layers on GPU 0 by now, the learned profile having moved - read in 58.0 and 58.0 s, 21 layers in 57.5 s and
    118.4 (a drive spell: ~1,000 reads/s at 47-59 ms for a minute); decode at 48K 34.2 / 32.2 tok/s against 35.9 / 37.2, its
    drafts kept 59-62% against 71-73% (other greedy texts), the short request's 36.2 / 36.7 against 37.1 / 36.4. No gain
    for reads: the split stays automatic.
- The server (serve/families.py `hy4`, serve/family_parsers.py `Hy4Syntax`, serve/family_templates/hy4.jinja,
  tools/strata_tokenizer_families.py `hy_v4` = joyai-llm's regexes): the exported tokenizer gives llama.cpp's ids; a reply
  with thinking, text and two calls parses into its reasoning, its content and the two calls with their schema's types.
  - **The template** (2026-11-18): the release's (tencent/Hy4-flash d88c9dfd chat_template.jinja), its tag suffix the
    GGUF vocabulary's - `:6124c78e` for `:opensource`, the same ids (all 24 suffixed special tokens of the release's
    tokenizer.json, the think tags among them). It replaced AngelSlim's GGUF template, which is the release's but for one
    thing: it also takes `reasoning_effort: low`, which the release's refuses ("should be no_think/high/max") and its model
    card does not name (it gives high, the default, and no_think). Strata's levels: low, medium and high are high, xhigh
    and max are max, thinking off is no_think; the family always loads the packaged file (the tokenizer directory's
    chat_template.jinja is the GGUF's). The server's prompts equal transformers' render of the release's template (its
    tojson / raise_exception) for a tool conversation and a multi-turn one at all three levels.
  - **Sampling**: the release's - generation_config.json T 0.9, top_p 1, top_k -1, no penalties; the model card
    "temperature=0.9, top_p=1.0" - as the run config's defaults, and the whole vocabulary (top_k 0 reaches the engine as no
    cut, the family's full_vocab) for a request that names no top_k. Run config strata-hy4.json (its "vision" block:
  pictures on GPU 1's free VRAM; --max-context 1048576 since 2026-11-19; its "env": the CPU pool's switches and
  STRATA_HY4_KV_GROW=1), launcher launch-hy4-flash.sh (pinned binary build-dsv41/strata-glm.hy4: 6dfa3041 since 2026-11-19 -
  the image encoder's fused attention on top of f20342cc (KV on demand on top of e51e7df5); f20342cc, e51e7df5 (the register-tiled indexer, k_b in one launch, the Engram row cache and the later
  stages' rows gathered ahead, the decode's per-head projections 4 rows a warp, the routers' F32 preload), 9569a63f (the tile
  scores kernel, coupled drafts, the reset fix), 5a0fde92, d471bddf and a75814f8 kept beside it).

2026-11-28: load_engram_hy4 is Hy4-flash's only. Since the port it ran for every model with an Engram and read the -m
set's own table: DeepSeek-V4.1 did not start on any build since 2026-11-18 from a set whose own table is any type but 43
or Q8_0 (type 67 in the sets here; IQ4_NL, Q5_K, Q6_K or BF16 in others), nor from a pack, and an own type-43 (or Q8_0)
table started with Hy4's row gathering running unread (DSV41_PLAN §77).
