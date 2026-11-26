# DeepSeek-V4.1-Flash-1017 in strata-glm: the port design (moex, from 2026-10-18)

Companion to DSV41_PLAN.md (phase 1). The map below comes from a read-only survey of strata-glm's DeepSeek-V4 path
against V4.1's reference model.py, with Xolotl's independent V4.1 engine as a second reading (file:line citations
are in the survey; this file keeps the decisions).

## Decisions

1. **One code path for V4 and V4.1.** A new GGUF architecture string `deepseek41` maps to `Arch::dsv4` with a
   `v41` flag. All 23 `Arch::dsv4` dispatch sites keep working, and only the deltas branch. V4 must stay
   byte-identical: dsv4_kernels_parity and a V4 decode are re-run after every stage.
2. **Dense weights in a V4.1 GGUF, written by `moex/tools/dsv41/write_dense_gguf.py`** from the pack's dense
   safetensors. strata-glm's loader, placement, split accounting and shape/type checks all key on GgufModel; a
   safetensors+JSON path would need new parsers and abstractions everywhere. Contents:
   - **Q8_0** from FP8 x E8M0 via f32: wq_a, wq_b, wkv, wo_a (dequantized with its 32x32 scales first, as
     convert.py does), wo_b, the shared experts, engram wkv, DSpark main_proj and blocks, and the head.
   - **F32, exact from BF16:** norms, compressor wkv/wgate (the reference pools in fp32), indexer wk / weights_proj
     / wq_b, router, the DSpark confidence and Markov heads.
   - **F32 as released:** hc_*, biases, sinks.
   - **Embedding as BF16**, dequantized on the host row by row.
   - **Metadata `deepseek41.*`:** compress ratios (0/1/2), kv and index source lists, candidate settings, engram
     settings (layer ids, table rows, hash constants from engram_tables.py), and the expert record layout per layer
     (ggml type, record stride, directory).
   - Q8_0 dense is the target recipe's dense format, and it costs a little against FP8: Xolotl measured KL 0.015
     on its own dense weights. It is re-measured on 1017 as a quality number, not a gate. An FP8 mode would need
     new MMVQ/MMQ kernels (a later option, as a custom GGUF type).
3. **Experts come from the pack's record files** through a new source in glm_tiers.
   - A record is already the NativeFmt blob, [gate rows | up rows | down rows]. Strides: MXFP4 18,800,640
     (up_off 6,266,880, down_off 12,533,760); Q3_K 15,206,400 of blob padded to 15,208,448; Q2_K 11,612,160.
   - Slice length and stride must be separated where the code conflates them (fill_slot, the prompt stream's
     issue_entry, warm_lend). A "contiguous source" flag lets them do one copy of fmt.bytes.
   - One ggml type per layer (Layer::fmt is already per layer). That is enough for the per-layer Q3_K/Q2_K pack.
4. **Hard blockers fixed first:** `kPoolMaxFF = 2048` (include/strata/kernels/cpu/pool.hpp:49) aborts on
   V4.1's 2304-wide experts. Raising it touches shared core, so every backend gets built.
5. **The two-GPU split snaps to layer 20.** Sharing groups ({2..7}, {8..13}, {14..19}, {20..39}) must not cross
   GPUs, and the prompt replay cut is at 20 too. DSpark's target layers 37-39 sit on the last GPU, as already
   checked.

## Component map (V4 counterpart in strata-glm, then the change)

| V4.1 component | change |
|---|---|
| Embedding | reuse: host rows, BF16 (type 30) |
| Hyper-connection mixes | reuse the kernels (same comb indexing, softmax + 19 x row/col) |
| Hyper-connection order | change the plumbing: attention collapses with the previous FFN's pre (one-hot at layer 0), the FFN with this layer's attention pre. A second coef buffer per row, carried through handoff, the mapped decode I/O, the prompt hand-off (h_hand), DSpark and the head. |
| q path | reuse minus the per-head q RMS norm (glm_dsv4.cpp:362) |
| Window KV ring | reuse, plus a fake-quant (FP8 E4M3, 32-wide, power-of-2 scale) before the F16 store |
| Compressor, ratio 2 | pool_kernel with R=2, no ape, relaxed static_asserts. Also emit the post-norm, pre-RoPE latent. After RoPE, FP4 fake-quant (16-wide, E4M3 scale). wkv/wgate as F32. |
| Compressor, ratio 1 | projection + norm (+RoPE + FP4 fake-quant); no gate, no persist |
| KV sharing | caches only on 2/8/14/20, aliased by the rest; split accounting follows |
| Index keys | new: k_norm(wk . latent) 512->128 F32, RoPE on dims [64,128) at the group's start, FP4 fake-quant (32-wide, power-of-2) into the source's cache; indexers 24-36 read layer 20's keys |
| Indexer q / weights / scores | 32 heads, ratio 1/2, no Hadamard, FP4 fake-quant of q, scale 1/sqrt(128*32), weights_proj F32 |
| Candidates (layer 20) | new: block max over 8, pin the query's newest block, top-2048 blocks, stored per query row; indexers 24-36 score only candidate rows |
| Top-512 and its reuse | lid_select with a ratio parameter; the selection (and candidates) kept per chunk row for reusing layers in the prompt path |
| Sparse attention + sinks | reuse; attn_counts / attn_key take ratios 1/2 (n_vis = (p+1)/r) and the source layer's cache |
| De-RoPE + grouped LoRA | reuse (wo_a [8192, 4096] as Q8_0 from the dequantized weight) |
| Router | reuse unchanged (384 experts -> route_kernel<16>, no hash layers; bias_vl only for images) |
| Shared expert | reuse (Q8_0, clamp 10) |
| Routed experts | reuse the logic; the record source; kPoolMaxFF 2304 |
| Engram | new: the hash (engram_hash.hpp) on the host as tokens are sampled or drafted; rows read (weights + scales); GPU FP8 x E8M0 dequant, wkv 6144->25600 (Q8_0), the gate (per-copy RMS dot, signed sqrt, sigmoid), R += gate * v before layers 1 and 14's mix_in |
| Final norm + head | mhc_pre with layer 39's FFN pre, then norm, then head (no output_hc_*) |
| DSpark | 128 experts top-3 per block (E and k per layer, not global), the V4.1 HC order, main_proj/main_norm, FP8 fake-quant of main_kv, a "block" window variant, the confidence head, experts from M0-M2.bin |

**Kernels unchanged:** rope (theta 160000, factor 16, orig 65536, beta 32/1 equal V4's), raw store, the attention
merge, route, rms_norm, mhc_*, gemv_f32_w, heads_mm, swiglu_clamp, moe_combine, and MMVQ/MMQ for Q8_0, MXFP4, Q3_K,
Q2_K.

**New kernels:** three fake-quants, the index-key kernel, candidate selection, the engram dequant + gate, and the
confidence head. **Unused by V4.1:** the Hadamard, the learned head reduce, hash ids, the LID compressor.

## Stages and gates

| stage | work | gate |
|---|---|---|
| S0 | the reference harness (moex/tools/dsv41/ref/, done 2026-10-18; ref_oracle.py writes the op-level oracle) on the engine's own weights: dense dequantized from the Q8_0 GGUF, experts from the records | harness == model.py on the release weights; the Q8_0-vs-FP8 dense KL recorded |
| S1 | kernels on synthetic data: fake-quants, compressors r=2/1, index key, the 32-head indexer at r=1/2, candidates, masked top-512, attention with sel at r=1/2, mHC with the carried pre, the engram gate; MMVQ at n_in = 2304 (tail iterations V4 never ran) | bitwise vs CPU transcriptions, with negative controls (the dsv4_kernels_parity.cpp patterns) |
| S2 | loader, hparams, one real layer at a time under --oracle probes: layers 0, 2, 3, 20, 21, 24 (engram off on both sides until the tables land) | per-op relative error within V4's tolerances; router top-6 sets equal; index sets equal before forcing |
| S3 | experts from records, all 40 layers, decode windows (file tier through mmap) | teacher-forced KL vs the fp32 reference ≈ noise, plus top-1; split at 20 == one GPU bitwise; T-row window == T single steps |
| S4 | engram | rows bitwise == model.py's; layers 1 and 14 match; the hasher's truncate holds under rejected drafts |
| S5 | prompt path + bounded replay (layer 20's KV front every chunk; layers 20-39 in full only over the last 2,560 rows; no expert streaming for 21-39 before the tail) | last-row logits and the rings of 21-39 bitwise == the full pass; 64 decoded tokens identical; checkpoint restore == uninterrupted |
| S6 | tiers from records (VRAM slots, RAM arena, file tier), then the Q3_K/Q2_K pack (per-layer fmt + stride) | slots and records vs crc32; hit and miss paths agree |
| S7 | DSpark | drafts == forward_spec's on the same trunk states; output with DSpark == plain greedy |
| S8 | server: the family entry, DeepSeek's V4.1 encoding, the parser | the serve tests |

## Risks carried

- **Prompt replay vs checkpoints.** With replay, the rings of layers 21-39 are valid only after a 2,560-row tail,
  so every checkpoint cut needs its own tail (~8% extra work at a 16K checkpoint spacing), or `live` is truncated
  to the last valid cut.
- **Layer 20's ratio-1 indexer scores every position:** 512 x 128K x 4 B = 268 MB, and 2 GB at 1M context. Tile
  the queries or fuse the top-k.
- **The wo_a trap:** its FP8 bits without the 32x32 scale are ~1000x too large; Xolotl hit this.
- **Ties in the index scores.** Compare sets, then force the reference's selection downstream.
- **Window K near-zero blocks:** after the 1e-4 scale floor they fall below F16's subnormal range.
- **Vision:** V4.1 image spans are causal; V4's non-causal image batch must not be reused. Out of phase 1 scope.
- **From the code review (2026-10-18, before the first GPU run).** Two bugs were found and fixed:
  1. The candidates kernel's binary search: its midpoint wrapped at 2^32, so the kernel never returned past 16K
     positions.
  2. The engram token map was stored as a GGUF metadata array, which the reader cuts to 64 items. It is now an I32
     tensor `engram.token_map`. Arrays read by the loader must now be whole.

  Notes carried forward:
  - **Long contexts:** the candidates kernel rescans every block's maximum 32 times. A radix select (as
    `dsv4_lid_select` does) is the fix if it shows at 128K and above.
  - **DSpark order (S7):** the engine captures DSpark's features before a layer's engram write, while the reference
    captures them after. This does not matter today: the targets are 37-39 and the engram layers are 1 and 14.
  - **DSpark attention (S7):** `Stage::dsv41` derives its row offset from `in - cur`, so a DSpark caller needs another
    way to pass it.

## Status (2026-10-18)

| stage | state |
|---|---|
| S0 | harness and op-level oracle written; 40-layer fp32 baseline (128 tokens) and a 12-step oracle on disk |
| S1 | **passed** 2026-10-18: `dsv41_kernels_parity` all pass; `dsv4_kernels_parity` unchanged (0 failures) |
| S2-S3 | **passed** 2026-10-18: selections and routing equal the reference at every step; KLD 0.0185 / same top 93.2% / PPL x1.011 on 1,023 tokens vs the fp32 reference; split at 20 == one GPU; windows 1/2/8 identical (DSV41_PLAN §11) |
| S4 | **passed** 2026-10-18 19:13: the 12-step oracle with the engram on both sides (`ref-runs/oracle-12-fp32-eg`) - layers 1 and 14's engram rows bitwise equal (`eg_emb` 0), the projection 5e-3 and the gated output 1.6e-3 as every other op, every routed expert set and index selection equal at every step; over the 5% tolerance only layers 8 and 20's FP4 index keys (max 5.1% / 6.3%, mean 0.8% / 1.0%), the rounding sites of the no-engram check |
| S5 | bounded replay written (layers 21-39 over the last 2,560 rows before each read start; the shift carries the streams, the carried pre-mix, layer 20's top-k lists and candidate masks); **passed** 2026-10-18: 7,820 and 12,000-token prompts + 64 greedy tokens identical with replay on/off; the 12K read 22% faster (156.8 vs 200.7 s); on by default |
| S6 | **done**: records feed the VRAM slots, the RAM copy and the SSD cache; the 217 GB pack (Q3_K on layers 0-19 + 32-39, Q2_K on 20-31) passes the DSV4 Q3_K mark (KLD 0.131, same top 92.38%, perplexity x1.090) |
| S7 | **passed** 2026-10-18: DSpark greedy == plain greedy, 71-73% acceptance - the drafter's dense weights in the GGUF (`dspark.*`), its experts from `M0-2.bin`, 128 experts top-3 per block, the carried pre-mix, the block attention (`dsv4_attn` mode 2: every row sees the 128 positions before the block), its norm, the trunk's head, the Markov and confidence heads (`--draft-dspark builtin`); its step since cut from 18.1 to 8.1 ms (the Markov head on the device, only the drafts a window verifies) |
| S8 | **passed** 2026-10-18 23:41: the `deepseek41` family (`serve/families.py`), its template (`serve/family_templates/deepseek41.jinja`, a port of the release's `encoding/encoding.py`: byte-identical on 756 rendered conversations + 300 random ones) and parser (DSML with V4.1's spaced tags; 8,478 checks against `parse_message_from_completion_text`, streamed in every split) - `serve/test_deepseek41.py`, 30 tests; the run config `strata-dsv41.json` and `run-deepseek-v4.1-flash.sh`. End to end on the API: two turns (the second reused 103 tokens and read 17 in 0.96 s), a tool call parsed (`get_weather {"city":"Paris","unit":"c"}`) and answered after its result (345 reused, 26 read in 1.5 s), a 1,795-token prompt read in 38.5 s and its follow-up (1,944 reused) in 0.57 s |
