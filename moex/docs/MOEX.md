# Strata-X (`moex`): Strata's mechanism for GLM-5.3-Flash, DeepSeek V4 Flash and MiMo V2.6 Flash

An experimental fork of Strata v0.1.40.1 (`82f46a8`), at `~/AI/Strata-X` on branch `moex`.
Status (2026-10-07): the research, the tools, the box model, a verdict and a port plan with its blockers are done.
**No engine code has been ported yet.** The loading experiments are prepared and deferred.

## 0. Isolation (the live install is never touched)

| | live install | this fork |
|---|---|---|
| code | `~/AI/Strata` | `~/AI/Strata-X` (`git remote local-strata` has a disabled push URL) |
| data | `~/AI/Strata-data` | `~/AI/Strata-X-data` |
| settings | `~/.config/strata/settings.json` | `$XDG_CONFIG_HOME=~/AI/Strata-X/.config` (set by `moex/env.sh`) |
| port | 1234 | 1240 |

`moex/env.sh` defines `moex_gpus_free`. Every loading experiment calls it and refuses while Strata, tabby or llama-server is running, or while a GPU holds more than 1.5 GB.
Strata's setup remembers its data folder and calibration per user, so a fork setup run without `env.sh` would rewrite the live install's settings.

## 1. Why Strata is about 3x on this box (measured on 2026-10-07, Qwen3.8-Flash-Next UD-Q4_K_XL)

- Decode is 69.5–75.9 T/s, against 24 T/s for tabby 6.05 bpw.
- Prefill is 1,058–1,207 T/s at 37K–172K tokens.

The mechanisms behind it:
1. **Global (layer, expert) GPU cache that adapts to the conversation.** Two 3080s hold 8,338 of 24,576 experts (34%). The live hit rate is 86–90%.
2. **CPU computes the misses at kernel speed.** Calibration chose `--pcie-frac 0.00`: on PCIe Gen3, computing a missing expert on the CPU beats copying it to the GPU.
3. **MTP drafting pays because the window mostly hits the cache.** About 2.09 tokens are accepted per verify window, at about 30 ms per window.
4. **KV streaming.** The full 262K KV cache lives in pinned RAM, with only a 32K window per QSA layer on the GPU.
5. **Expert geometry.** 512 experts × 4.9M params with top-10 makes 2.36B routed params per token. At ~4.6 bpw, all experts are 69 GB.

## 2. Strata's code: what carries over

Strata is 88K lines of C++/CUDA plus 15.5K lines of Python tools. The geometry is hard-coded in `include/strata/core/layout.hpp` (`ModelGeometry`): n_embd 2560, 48 layers, QSA every 4th layer, GDN 16/48 heads, HC 4 (low-rank 320), 512 experts × n_ff 640.

- **Model-specific:** PLE in 83 files, QSA attention in 42, Gated DeltaNet in 27, hyper-connections in 18, the top-10 router in 17, MTP in 19.
- **Reusable machinery (adapt for 4096×2048 experts):** `core/expert_cache`, `expert_source`, `remote_experts`, `remote_expert_opt.cu`, `pinned.cu`, `peer_experts`, `verify`, `mtp`, the CPU expert kernels and the layer split.
- **Expert formats the kernels run:** Q2_0, IQ1_M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS, Q4_K, Q5_K, Q5_1, Q8_0. There is **no Q3_K, Q6_K or MXFP4**.

## 3. The targets (`moex/tools/describe.py`, from each config.json)

| | Qwen3.8-Flash-Next (Strata) | GLM-5.3-Flash | DeepSeek V4 Flash | MiMo V2.6 Flash |
|---|---|---|---|---|
| layers (MoE) | 48 (48) | 45 (42) | 43 (43) | 48 (47) |
| experts | 512 × 640, top-10 + shared | 288 × 2048, top-8 + shared | 256 × 2048, **top-6** + shared | 256 × 2048, top-8 |
| routed params/token | 2.36B | 8.46B | **6.49B** | 9.46B |
| all routed params | 121B | 304B | **277B** | 303B |
| linear attention | GDN, 36 of 48 | KDA, 34 of 45 (same 3:1 pattern) | — | — |
| full attention | QSA: GQA + indexer (compress 4, budget 2048) | DSA: MLA + indexer (kpool 4, top-2048) | compressed sparse (×4/×128 per layer, top-512, SWA-128) | SWA-128 ×39 + full ×9 |
| hyper-connections | ×4 low-rank | ×4 Sinkhorn | ×4 Sinkhorn | — |
| MTP layers | 1 | 1 | 3 (not in the local GGUF) | 3 |
| vision | yes | yes | yes | yes |
| today on this box | 24 T/s (6.05 bpw tabby) | 11.1 T/s | 10.4 T/s | 11.5–12.6 T/s |

GLM is close to an architectural twin of Strata's model. DeepSeek V4 shares the indexer and Sinkhorn-HC ideas. MiMo shares only the MoE and the MTP head.

## 4. Routing evidence (`moex/tools/cache_capture.py`, 48-prompt code corpus, 4096 + 128 tokens)

Adaptive hit rate (the prompt's own prefill routing ranks the cache, a floor for Strata's live cache):

| cache share of pairs | Qwen3.8-Flash-Next | GLM-5.3-Flash |
|---|---|---|
| 15% | 55% | 37% |
| 22.5% | 66% | 46% |
| 34% | 77% (live Strata: 86–90%) | 59% |

GLM's routing is much more diffuse than Qwen's. DeepSeek V4 and MiMo have **no capture yet**, which is the deferred experiment in §7. Until then they borrow GLM's curve.

## 5. The box model (`moex/tools/fit.py` + `moex/box.json`)

A roofline per verify window: `t = fixed + max(t_gpu, t_cpu)`. It is calibrated on Strata's measured run here: a 30 ms window, about 20 ms of it fixed. The calibration check predicts 44–82 T/s for Qwen, against 69.5 measured.

| model @ bpw | cache share | decode T/s, no MTP / MTP | today | speed-up | all experts in RAM? |
|---|---|---|---|---|---|
| **DeepSeek V4 @ 2.6** | 25% | 18–36 / 22–52 | 10.4 | **2.1–5.0×** | yes, 90 GB |
| **DeepSeek V4 @ 3.06** | 21% | 16–34 / 18–44 | 10.4 | 1.8–4.2× | **yes** (up to 3.29 bpw) |
| GLM-5.3 @ 2.6 | 23% | 15–33 / 17–42 | 11.1 | 1.6–3.8× | yes, 99 GB (up to 3.00 bpw) |
| GLM-5.3 @ 3.06 | 20% | 13–29 / 14–35 | 11.1 | 1.3–3.2× | **no** |
| MiMo V2.6 @ 2.6 | 26% | 15–33 / 17–41 | 11.5 | 1.5–3.6× | yes, 98 GB (up to 3.01 bpw) |
| MiMo V2.6 @ 3.06 | 22% | 12–28 / 14–34 | 11.5 | 1.2–3.0× | **no** |

## 6. Verdict: the next best fit on this box is **DeepSeek V4 Flash**

1. **Lightest MoE per token.** Top-6 gives 6.49B routed params per token, 23% less than GLM and 31% less than MiMo. The CPU computes the misses, so this is the term that sets the speed.
2. **Fewest experts in total (277B).** It is the only target whose experts fit in RAM at ~3 bpw, up to 3.29 bpw. Strata's two-GPU mode needs all experts in RAM; GLM and MiMo must drop to 3.0 bpw or less.
3. **Biggest predicted gain.** 1.8–5.0× over today's 10.4 T/s, the highest band of the three.
4. **Other advantages:**
   - Cheap long context: compressed sparse attention with one latent KV head.
   - Images, from a local GGUF mmproj that mtmd already knows (pinned and mainline).
   - Three MTP heads, plus a local DSpark drafter.
   - Native-FP4 experts upstream.

**Runner-up: GLM-5.3-Flash, the cheapest port.** Its 3:1 KDA/DSA pattern, indexer (kpool 4, top-2048), HC×4, shared expert and single MTP layer map onto Strata's GDN/QSA/HC/MTP code. But its routing is measured diffuse (46% at its affordable cache) and it is RAM-bound at 3 bpw.

**Last: MiMo V2.6 Flash.** It has the heaviest per-token MoE (9.46B), no shared expert, and none of Strata's attention kernels carry over. Its standard SWA attention would be the simplest to write from scratch, and its light dense side gives it the biggest cache.

The verdict's main uncertainty is DeepSeek V4's routing concentration, which still borrows GLM's curve. Even at GLM's diffuse curve it ranks first. A capture more concentrated than GLM's only raises its band.

## 7. Port plan (`moex/tools/port_plan.py` checks the blockers live)

### DeepSeek V4 Flash (target 1)
- **Phase 0, no GPU:**
  - Bump `third_party/llama.cpp` to mainline (deepseek4 in gguf-py, the model code and mtmd).
  - Generalize `ModelGeometry` into an arch-tagged struct read from GGUF metadata. Started: `moex/generated/moex_geometry.hpp` holds the generated, compile-checked geometry of all four models.
  - Generalize `tools/iq_pack.py` to deepseek4 tensor names.
  - Add a Q3_K expert kernel, or requant.
- **Phase 1, GPU-light:** the expert cache, CPU pool and layer split on 4096×2048 experts, with parity tests against ggml's MoE (`src/kernels/*_parity` style).
- **Phase 2:** the attention stack (new), compressed sparse attention: latent 512-dim heads, 1 KV head, compression ×4/×128, indexer 64×128 top-512, SWA-128. Then Sinkhorn mHC.
- **Phase 3:** drafting (fetch the 3 MTP layers like `tools/mtp_fetch.py`, or a DSpark verify path), KV streaming, images.

**Blocked by:**
- **Disk.** 65 GB free; a Strata-supported ~2.6–3.06 bpw GGUF needs 90–106 GB. Free space first, or delete the 128 GB Q3_K GGUF after a requant.
- **Expert types.** The local Q3_K GGUF isn't covered by Strata's kernels. Either `llama-quantize --allow-requantize` to IQ3_XXS/IQ2_M (double quantization costs quality; needs the disk above), or add Q3_K to the fork's kernels. The Q3_K file (119 GB of experts) also exceeds the RAM limit at 3.44 bpw.
- **MTP weights.** They are not in the GGUF.
- **Routing capture.** It needs the GPUs: §8.

### GLM-5.3-Flash
**Adapt:** GDN→KDA, QSA→DSA (MLA), HC→Sinkhorn, router→sigmoid.
**Blocked by:**
- No GGUF on disk (UD-IQ3_XXS was deleted; ~100 GB to re-download).
- No GGUF mmproj.
- glm5-next is missing from Strata's pinned llama.cpp (mainline has it since 2026-09-30/10-01).
- llama.cpp MTP support for it is still an open PR (#27917).

### MiMo V2.6 Flash
**New:** GQA SWA/full attention with sink bias. **Drop:** HC, GDN, PLE.
**Blocked by:**
- No GGUF (EXL3 only; converting needs the original weights).
- No GGUF mmproj (the vision tower exists as safetensors in the extras dir).
- No routing capture.

## 7b. GLM-5.3-Flash testing ground (from 2026-10-07)

**File:** `~/AI/Models/GGUFs/glm53-flash/glm53-flash-q2k-fast-111.gguf` (110.9 GB, arch `glm5next`). It was copied by the user from `/mnt/shared/SharedMount/ModelStash/` (NTFS on the same NVMe). The copy was verified by size and 66 sampled ranges (384 MiB), all identical.

**Why GLM first:** it is the architecture closest to Strata's own model, so the shared machinery can be brought up with the fewest new kernels. Quality is irrelevant: the port is judged against llama.cpp **on the same file**.

**Kernel facts, corrected on 2026-10-07 from the source:**
- The CPU expert path takes any ggml `vec_dot` type.
- The GPU (`native_mmvq_supported`, `dequant_bf16`) covers Q4_0/5_0/5_1/8_0, **Q3_K**, Q4_K/5_K/6_K, IQ4_NL/XS, Q2_0, IQ2_*, IQ3_XXS/S and IQ1_M.
- §2's "no Q3_K" was wrong.
- Missing on the GPU were **Q2_K** (this file) and **MXFP4** (2.3% of DeepSeek's UD-IQ3_S).

**Done in the fork:** a Q2_K GPU port.
- `native_q2_k_mmvq` and `native_q2_k_f32` are exact transcriptions of the pinned ggml `vec_dot_q2_K_q8_1` (decode, including multi-column verify windows through `Q2KTraits`).
- `dequant_bf16` type 10 is ggml's `dequantize_row_q2_K` (prompt path).
- Tests:
  - `src/kernels/q2k_parity.cpp`: GPU against float64 at 2e-2, BF16 bitwise, with a negative control.
  - A Q2_K case in `mmvq_multi_parity`: multi-column ≡ single-column, bitwise.
- The tests need a GPU: §8.

**Loader contract** (tensor names in `glm5next` GGUFs):

| layer kind | tensors |
|---|---|
| KDA (34) | `attn_q/k/v` 4096→8192, `attn_output` 8192→4096, `ssm_conv1d_{q,k,v}` [4,1,8192], `ssm_a` [64], `ssm_dt.bias`, `ssm_beta` 4096→64, `ssm_f_a/f_b` 4096→128→8192 (channel-wise forget), `ssm_g_a/g_b` (output gate), `ssm_norm` [128] |
| DSA (11) | `attn_q_a` 4096→1536 + `attn_q_a_norm`, `attn_q_b` 1536→64×256, `attn_kv_a_mqa` 4096→512 + `attn_kv_a_norm`, `attn_k_b` [256,512,64], `attn_v_b` [512,256,64], `attn_output` 16384→4096. Indexer: `indexer.attn_k` 4096→128, `indexer.attn_q_b` 1536→32×128, `indexer.k_norm` (weight+bias), `indexer.proj` 4096→32, `indexer_compressor_gate` 4096→128, `indexer_compressor_ape` [128,4] |
| every layer | `hc_{attn,ffn}_fn` 16384→24, `hc_*_base` [24], `hc_*_scale` [3] (4-stream Sinkhorn mHC); `attn_norm`, `ffn_norm` |
| MoE (42, from blk.3) | `ffn_{gate,up,down}_exps` (288, Q2_K), `ffn_*_shexp` (Q6_K), `ffn_gate_inp` F32 [4096,288], `exp_probs_b.bias` [288] |
| dense (blk.0-2) | `ffn_gate/up` 4096→12288, `ffn_down` |
| MTP (blk.45) | DSA attention + `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm` |

**Bring-up order, each step parity-tested against llama.cpp on this file:**
1. Q2_K kernels: done; the GPU parity run is pending.
2. Runtime geometry from GGUF metadata. Start: `moex/generated/moex_geometry.hpp`.
3. Loader for the names above.
4. Router: sigmoid + `exp_probs_b` bias, top-8 + shared, scaling 2.5.
5. KDA from the GDN kernels: channel-wise decay through the low-rank `f` gate, `g` output gate.
6. DSA: MLA latent KV (512) + indexer top-2048 with compressor (kpool 4), reusing the QSA indexer and top-k.
7. Sinkhorn mHC.
8. MTP (1 layer).
9. End to end: KL divergence against `glm_reference_logits.sh`'s oracle, then speed against `fit.py`'s band (GLM at ~2.6 bpw: 15–33 T/s without MTP).

## 8. Deferred loading experiments (run LAST, by hand, with every model server stopped)

0. Strata-X GPU tests: `build/q2k_parity` and `build/mmvq_multi_parity` (Q2_K port; seconds; GPU needed).
1. `moex/experiments/glm_reference_logits.sh`: the GLM parity oracle, 10–20 min. It builds `llama-perplexity` in `llama.cpp-glm5next-mtp` if missing.
1. `moex/experiments/capture_routing.sh all` captures DeepSeek V4 (EXL3 3.04 bpw, `-mcs 192`) and MiMo Yamz (`-mcs 203`, through the Yamz shim, uncensor off). Each takes 30–40 min and replaces the borrowed curves. `fit.py` then re-runs automatically.
2. After Phase 0/1: expert-cache parity and throughput on DeepSeek V4 experts against the fit's t_cpu and t_gpu terms.

## 9. Files

- `moex/env.sh`: isolation and the GPU guard.
- `moex/box.json`: measured box constants, with sources.
- `moex/models/*.json`: normalized geometry, from `moex/tools/describe.py`.
- `moex/tools/gguf_inventory.py`: header-only GGUF breakdown and Strata kernel coverage.
- `moex/tools/cache_capture.py`: static and adaptive cache capture from per-prompt routing.
- `moex/tools/fit.py`: decode/RAM predictions and the ranking.
- `moex/tools/port_plan.py`: per-model component map and live blocker check.
- `moex/tools/gen_geometry.py` writes `moex/generated/moex_geometry.hpp` (constexpr `MoeGeometry` per target, the Phase 0 replacement for the hard-coded `ModelGeometry`). `geometry_check.cpp` compile-checks it: Qwen matches `layout.hpp`, and the per-token expert load orders DeepSeek V4 < GLM < MiMo.
- `moex/experiments/capture_routing.sh` and `profile_capture.py`: the deferred routing captures.
- `moex/out/`: results; ignored by git.

## 7c. GLM-5.3-Flash on Strata-X: first GPU results (2026-10-07, Q2_K test file, Qwen server stopped)

All runs: `build/strata-glm`, GLM-5.3-Flash Q2_K (111 GB), 2 x RTX 3080 20 GB, Xeon Gold 6148.

- **Kernel parity, GPU**: `glm_kernels_parity` (48 checks incl. every window kernel bitwise = sequential, batched heads),
  `q2k_parity` (dequant bitwise; MMVQ <= 6.3e-4 of row mass; the min-term negative control fails at 0.62),
  `mmvq_multi_parity` (Q2_K multi-column bitwise).  All pass.
- **Op-level oracle** (4-layer slice, llama.cpp CPU dump, every op forced to llama.cpp's inputs): Q8_0 / f32 ops
  1e-7, router ids identical, CPU experts bitwise.  K-quant projections differ 1-2.5%: the CPU oracle's Q8_K
  activations - against an exact float64 product the GPU path is 2.3-2.7x CLOSER (Q3_K ffn_up: 0.7-0.9% vs 1.5-2.5%).
- **MTP**: greedy output token-identical with and without MTP (64 tokens, full model; 40 on the slice, 2 GPUs);
  acceptance 83-96% (1-token drafts) once the draft readback waited for the non-blocking stream (it read the
  previous draft: 0% before).
- **Decode, 2 GPUs (layers 0-21 / 22-44 + head), experts in place from the GGUF mapping, 27 GB VRAM tier from the
  tabby routing profile**: 18.1 T/s plain, 17.9 with MTP; llama.cpp on the same quant ~13 T/s (user).  Per token
  ~52 ms: CPU pool ~30 (memory-bound: 38 unpinned threads slower; a huge-page RAM arena slower than the page
  cache), router wait ~9, rest GPU.  Hit rate 37.8% (the profile promises ~45% at 25% of pairs).
- **Prompt**: ~25 T/s (windows of 8 tokens through the CPU pool) - the next piece is the big-T prefill streaming
  experts to the GPUs.

## 7d. GLM prompt path (Strata's MMQ prefill), first measurements (2026-10-07, Q2_K, 23,943-token prompt)

The path: per layer, routings sorted by expert, activations quantized once, experts gathered 16 at a time into one
MMQ product each for gate+up and down, non-resident experts streamed through a 32-blob ring on a copy stream.
Token-identical to the window path at chunks 8 / 16 / 512 on the slice (after fixing the index block's offsets,
which assumed full chunks).

| config | prefill T/s | MoE host time | note |
|---|---|---|---|
| window path (T = 8) | ~25 | - | before |
| chunk 2048, pageable copies | 86.3 | 264.5 s of 277.5 | the host blocks in every copy |
| chunk 2048, GGUF pages pinned (read-only) | 121.4 | 48.6 s of 197.2 | pinning 93 GiB of page cache took 110 s |
| chunk 8192 | - | - | cudaMalloc failed: every window buffer is chunk-sized |

- Streamed 114.5K experts (877 GiB) per 24K-token prompt: each 2048-chunk re-streams ~73 GiB, the GPUs take turns
  (GPU0's x8 link, then GPU1's x16), so ~4.5 GiB/s overall.
- The 2048-row scratch shrank the VRAM tier from 3499 to ~2250 slots: decode after prefill 13-14 T/s (vs 18).
- Next (pending the Strata design audit, moex/docs/STRATA_QWEN_DESIGN.md): MoE at the full chunk with mixers in
  sub-batches, group-sized MoE buffers, the expert arena as Strata's PinnedArena, the two GPUs overlapped.
- MXFP4 (DeepSeek-V4's one MXFP4 expert layer, MiMo's MXFP4 down projections): MMQ tiles enabled, native MMVQ added
  (build/mxfp4_parity: 9e-4 of the dot mass, multi-column bitwise, negative control detected).

## 7e. GLM on Strata's structure: prefill sweep, decode graphs, adaptive tier (2026-10-07, Q2_K test file, 2 x RTX 3080)

**The engine** is now Strata's shape: a `Stage` per GPU (src/glm/glm_stage.cpp = the model's layers, glm_prompt.cpp = the prompt
path, glm_decode.cpp = the decode graphs, glm_engine.cpp = load / the VRAM tier / the API); the experts live in Strata's
`PinnedArena` (registered before a page is touched, filled 16 experts per pread).

**Prefill (24K-token prompt, three chunks of 8192; the first measurements are in 7d):**

| change | T/s |
|---|---|
| window path (before) | ~25 |
| MMQ prompt path, pinned GGUF pages, chunk 2048 | 121 |
| PinnedArena + slot loan + stages pipelined + chunk 8192 | 391 |
| + GEMM routing of small projections | 467 |
| + tensor-core DSA attention (Strata's qsa_prompt_attn v2 for the MLA latent: 16 heads per block, K = V = the FP16 latent, a warp owns 128 of the 512 dims, hi/lo q and p) | 624 (ring 64) |
| + ring 160 | **745** |

Sweep (one load, `--runs ring:chunk:sub`): ring 64 / 160 / 320 / 480 = 624 / 745 / 722 / 705; chunk 4096 / 6144 / 8192 = 542 /
646 / 722; mixer sub-batch 256 / 512 / 1024 = 686 / 722 / 697.  The default ring is 55% of a layer's experts, in whole groups.
Steady state (long prompts) is ~990 T/s: a 3-chunk prompt only reaches 3/4 pipeline efficiency.  The idle gaps left in the
profile are the compute stream waiting for an expert's PCIe copy (GPU0's link is x8).  Hardware: GPU0 6.25 GB/s, GPU1 8.1
GB/s on 4 KB pages and 10.5 with THP; the GPUs do not share a bus ceiling.

**Decode (19-token prompt, greedy, the full model; outputs identical across all variants):**

| variant | hit rate | pool ms | doorbell wait ms | T/s |
|---|---|---|---|---|
| eager (v1) | 35% | 30.6 | 8.6 (router sync) | 18.4 |
| graphs + doorbell + grouped GPU hits | 35% | 30.7 | 14.2 | 20.2 |
| + Strata's adaptive rule, static profile only (--no-adaptive) | 29% | 32.9 | 15.8 | 19.3 |
| + Strata's rule, every 8 windows, <= 32 swaps | 46% | 25.6 | 15.9 | 22.8 |
| + Strata's rule, every 4 windows, <= 96 swaps | 58% | 19.9 | 21.3 | 22.9 |

The profile that motivated the graphs: ~2,940 kernel launches per token = 19.4 ms of host launch time (+ 2.9 ms of memcpys) of
~56 ms; the GPUs busy ~12 ms.  After the graphs the token is the CPU pool plus a GPU dense floor of ~14 ms (the 7.5 GiB of
dense weights read once per token at ~760 GB/s is ~10 ms, the rest kernel-latency chains).  Heavy swapping gains hit rate and
loses ~5 ms of GPU chain time (swap DMAs of 7.9 MiB blobs vs the doorbell's tiny mapped reads on the same PCIe direction):
Strata's 96 swaps was tuned for 1.4 MB blobs.  MTP gives no gain at these hit rates (a T = 2 window reads ~1.75x the misses).
