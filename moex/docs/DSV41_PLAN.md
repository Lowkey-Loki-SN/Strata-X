# DeepSeek-V4.1-Flash-1017 on this box: the plan (moex, from 2026-10-17)

**Status (2026-10-21): serving.** The 212 GB pack `q3q2u` runs through the server (`run-deepseek-v4.1-flash.sh`,
`strata-dsv41.json`): our Q3_K/Q2_K placed as Unsloth's UD-Q2_K_XL places its bits, the release's MXFP4 downs where
Unsloth keeps them, Unsloth's imatrix; KLD 0.133 / same top 93% against the native release on wikitext, the route bias 0.3
decode-only, the release's sampling at max effort, pictures end to end. §12 holds the goal of 2026-10-19 (the margin, the
audit #33-#36, vision, speed); §13 the re-audit's fixes (#43-#57); §14 the server log of 2026-10-20 (the learned profile,
deep checkpoints, the drive's slow mode - read disturb, fixed by `refresh_pack.py` - and the configuration for this box);
§15 the audit of 2026-10-20 and Maya v1.1.16-17; §16 the q3q2u pack against the 217 GB one, and Maya v1.1.18-20; §17 a
stop between layers (in the config) and layer-major prompt reads (exact, the cold 44K read 166.9 -> 125.3 s; off again after
a cold 84K read from a reboot stalled behind it).
DSV41_BACKLOG.md lists what is left. The history below is kept as it was written.

**Status (2026-10-17 18:10): plan final; phase 0 passed all three gates (§9); phase 1 conversion running.**
- The release is streamed from Hugging Face straight into the native pack by `moex/tools/dsv41/convert_pack.py`,
  with no raw copy kept: each shard is checked against its SHA-256, and the run can be resumed.
  - Speed: ~20 MB/s, about 7 h for 502 GB.
  - Log: `~/AI/Models/dsv41-1017/convert.log`.
- Where the pack goes:
  - experts and engram (498.8 GB) in `~/AI/Models/dsv41-1017/native` on ext4, leaving ~21 GB free on /;
  - the dense safetensors (~11.5 GB, read by the reference and `write_dense_gguf.py`) in
    `~/AI/Models/dsv41-1017/dense`. They were on `/mnt/shared` (NTFS through FUSE) until 2026-10-18 evening, when the
    user moved them there to repartition that drive; all 48 files' sizes and tensor counts were checked after the
    move, and `native/manifest.json` now points there (the old copy is `manifest.json.bak-2026-10-18`).
  - the dense GGUF the engine loads (dense weights, the DSpark drafter, the engram projections) in
    `~/AI/Models/dsv41-1017/dsv41-1017-dense-q8_0-dsp-eg.gguf`.
- Shards 3 (layer 0) and 43 (head) were checked with `verify_pack.py` against fresh range reads: records 0, 68 and
  383 byte-identical, all 384 CRCs good, dense tensors identical.
- The engine port of phase 1 has not started.
- **2026-10-18 10:26: restarted after a shutdown.**
  - 34 of 48 shards were done (224 GB). `verify_pack.py --local` checked them: all 11,904 expert records match
    their crc32, and every dense file has the right size.
  - `~/AI/Models/dsv41-1017/resume.sh` runs the 12 small shards left, then the rest.
  - convert_pack.py now resumes inside the two ~101 GB engram shards: a kept part is cut back to its last 256 MiB
    flush point, hashed again, and continued with a Range request. The shard's SHA-256 still covers every byte.
    It also fsyncs the manifest and directories.
  - `test_convert_resume.py` passed all four cases against a local server: clean run, SIGKILL then resume (at the
    512 MiB flush point of a 640 MiB part), damaged part caught by the hash and redone from zero, dropped
    connection resumed within the run.
  - The Wi-Fi link now gives ~5-8 MB/s, against 17-20 MB/s the day before.
- **2026-10-18: the user's decisions on the weights and the quant.**
  - **Source: only deepseek-ai/DeepSeek-V4.1-Flash-1017 as released.** No abliterated or otherwise modified
    checkpoint. That rules out Xolotl's published GSQ quant, which is made from dealignai's abliterated V4.1-Flash
    (§3).
  - **Target pack: the DSV4-Flash Q3_K recipe, carried over** (§6):
    - experts Q3_K, with an imatrix from the native 1017 model;
    - dense Q8_0;
    - the small tensors BF16/F32, as in that recipe.
  - **Pass mark:** that Q3_K's own numbers against its MXFP4 release (wiki, 64x512 tokens): mean KLD 0.134, same top
    token 88.0%, perplexity x1.101.
  - **Expected speed:**
    - 234 GB of experts, 54% of them resident;
    - ~6.7 tok/s middle estimate [3.3..12.7] (with the measured Q3_K CPU rate, §4).
    - Q3_K's balance on DSV4 came from fitting in memory (119 GB of experts). V4.1's experts fit in VRAM + RAM
      only at <= 1.84 bpw (§4).
- **2026-10-18, later: the user's requirements.**
  - **Usable means >= 20 tok/s decode.**
  - **The hardware is fixed.** This plan targets this box only. Hardware levers are kept apart in §10, as reference
    for the user's own planning.
  - On this box, only the ~2.3-bpw class reaches ~20 tok/s at exact routing: §4 gives 21.2 tok/s middle
    [14.6..25.0], 160 GB.
  - Its expected quality is about that of Xolotl's 2.25-bpw quant at exact routing, measured against its own FP8
    original:
    - KL 0.229 (the DSV4 Q3_K: 0.134);
    - perplexity x1.15;
    - API top-1 86.1% vs 93.7%;
    - entropy +19%;
    - by domain, perplexity +14% on code and +31% on long agentic contexts.
  - ~~Proposed for phase 6: the ~2.3-bpw pack first~~. Superseded the same day by the user's choice below.
- **2026-10-18, the user's choice of pack and priority.**
  - **The pack: the per-layer Q3_K/Q2_K mix at ~203 GB of experts.**
    - Q3_K for the sensitive layers, Q2_K for the rest: §4 lists 202 GB for Q3_K on layers 4-19 + 32, and 207 GB
      for 0-19 + 32.
    - The exact split is set from the per-layer sensitivity measured on the 1017 weights. That measurement comes
      from the same phase 1 run as the imatrix.
  - **Priority: make that pack as fast as it can be on this box, before any lower-bit pack or route bias is
    considered.** §4's estimate for it is ~10 tok/s middle at exact routing, so the speed work (read-ahead, DSpark
    depth and confidence gating, arena adaptation, both GPUs' PCIe links for prompts) is what moves it.
  - **The engine port (phase 1) starts now**, while the engram tables download.

**Goal (user, 2026-10-17):** run deepseek-ai/DeepSeek-V4.1-Flash-1017 on this box at good speed. The box is capped
on DRAM and VRAM, so the experts and the engram tables stream from SSD. The design takes what works from
llama.cpp, exllamav3, Strata, Strata-X and the ds4 engines (antirez/ds4 and the forks that stream experts from SSD).

**The short answer:**
- On today's hardware, with the experts in their released FP4 bits (lossless), decode is SSD-bound at about
  **2-7 tok/s (middle estimate ~4)**.
- With a quality-gated expert pack of ~2.7-3.0 bits per weight it is about **6-17 tok/s (middle ~10-12)**.
- With the target Q3_K pack (3.44 bpw, the DSV4 recipe) it is about **3-13 tok/s (middle ~6.7)**.
- Prompts pay to stream experts (corrected 2026-10-18, §4):
  - a +200-token agent turn costs ~9-27 s of expert reads depending on the pack (Q3_K ~18 s);
  - a fresh 8K prompt costs 33-68 s (Q3_K ~45 s). That one is bound by PCIe 3.0, not the SSD.
- The user's bar is >= 20 tok/s. Only the ~2.3-bpw class reaches it here: ~21 tok/s middle [14.6..25.0], §4.
- Even with no SSD waits at all, decode is capped by the CPU's DRAM bandwidth over the RAM-resident experts: about
  15-25 tok/s, with the ~2-3 bit packs at the top of that range.

## 1. The model, in bytes (from all 48 safetensors headers)

| part | size | per decode token |
|---|---|---|
| routed experts, 40 layers x 384, top-6, FP4 (E2M1) + E8M0 scales per 32 | 288.8 GB (18,800,640 B each) | 240 picks = 4.51 GB |
| engram tables, layers 1 and 14, FP8 rows of 256 B + 8 B of scales | 202.8 GB (2 x 384M rows) | 48 rows = 12.7 KB |
| trunk dense (attention, shared experts, router, hyper-connections, engram wkv), FP8 / BF16 | 7.20 GB | all of it |
| embedding + head, BF16 | 2.65 GB | head 1.32 GB + 1 row |
| DSpark drafter (`mtp.0-2`: 3 blocks, 128 experts top-3 each, Markov + confidence heads) | 7.22 GB experts + 0.71 GB | ~0.8 GB per draft |
| vision | 0.97 GB | per image |

- Per decode token: 8.5 GB of dense reads (GPU) plus 4.5 GB of routed expert reads. The old V4-Flash read 3.45 GB of
  experts.
- KV cache, packed as the reference quantizes it: 890 B per token of context. That is 117 MB at 128K and 0.93 GB at
  1M, so context length is not the problem.
- Engram row numbers depend only on token ids: a 2-4 gram hash of the last 4 normalized ids, mod 8 primes per order.
  So they are known as soon as a token is sampled, or drafted.
- Layers 21-39 keep no KV of their own beyond a 128-token window. Their attention reads layer 20's caches. So in a
  prompt only a tail needs layers 21-39 ("bounded replay").
  - **Corrected 2026-10-18:** the tail is the last 128 x 20 = **2,560 tokens**, not 128. Each layer above 20 reads its
    own 128-token window of the layer below, so the dependency grows by 128 per layer.
  - Xolotl does exactly this (ds41_generate.cpp:11-12, 328-340), with last-row logits byte-identical to the full pass
    at 8,192 tokens.
  - Phase 1 checks it against the reference.
- The expert bits are ggml MXFP4 after a nibble shuffle inside each 32-block, with the same 4.25 bits. That is
  lossless, and llama.cpp already does it for V4 (`repack_mxfp4_blocks`). Strata-X's mx_avx512 CPU kernels run this
  format (§2).
- Source of these facts: model.py / engram.py / convert.py of the release, read on 2026-10-17. llama.cpp has no V4.1
  support, and nothing on this box has run V4.1.

## 2. The box, measured

| what | measured | where |
|---|---|---|
| NVMe (EVM NV500 2 TB, DRAM-less, on the chipset at Gen3 x4) | 2.76 GB/s with 4 MiB O_DIRECT reads, 8 threads; 2.3-2.5 GB/s at 0.25-18 MiB; **2.36 GB/s while 16 threads read DRAM at 74 GB/s** | nvme_bench 2026-10-17 |
| NVMe 4 KiB random reads | 170K IOPS (0.70 GB/s) at 64 threads | same |
| DRAM (4 of 6 channels, DDR4-2666) | 72.8 GB/s, 16 threads | membw 2026-10-17 |
| CPU expert kernels, 20 threads, 1 token | MXFP4 69.7 GB/s, IQ3_XXS 47.1 GB/s, EXL3 K3 trellis 41.4 GB/s | MXFP4.md:68-76, exl3 track1 |
| PCIe host to GPU | GPU0 6.25 GB/s (x8), GPU1 8.1 GB/s (x16) | MOEX.md:255 |
| memory for experts | VRAM ~21 GB after dense + DSpark + KV; RAM ~104 GB (125 GiB minus the 20 GiB headroom minus 8 GiB) | configs, glm_tiers.cpp |
| board | Supermicro X11DPi-NT, one CPU. GPU0 on CPU stack 16 at x8, GPU1 on stack 64 at x16, the NVMe on the chipset at Gen3 x4 | lspci 2026-10-17 |

- io_uring is enabled in the kernel, but liburing is not installed. A pool of threads doing O_DIRECT reads already
  reaches the drive's limit, so the plan does not need io_uring.
- The SATA SSD (Crucial BX500) holds the Windows install, so it is not used.

## 3. What each source taught (take / avoid)

- **Strata-X** (moex: GLM-5.3, MiMo, DSV4):
  - Take: the tier code (VRAM slots, a pinned RAM arena sized from MemAvailable, the lend ring for prefill, adaptive
    exchange swaps, learned profiles, the strata-warm read-in); the MXFP4 CPU kernels; two-GPU layer split; MTP
    window 2; checkpoints.
  - The lesson of the 2026-10-10 file-tier experiment (MXFP4.md): ~8 first-touch NVMe reads per token halved V4
    decode. The drive sat idle ~65% of the time, because a read could only start once its layer had routed. That
    idle time is the thing to remove.
  - Avoid: MADV_WILLNEED (128 KiB cap), touch faults (≤2.3 GB/s), page-cache cold reads under memory pressure.
- **Strata** (upstream): the learned profile order (#477) and the file-tier look-ahead idea, measured +25.7% on a
  32 GB card and -32.6% on a 16 GB card. So prefetch has to earn its bandwidth.
- **antirez/ds4:**
  - Take: one contiguous record per expert and per engram row; read-ahead entries inserted as the oldest, so a wasted
    prefetch cannot evict hot experts (`ds4_cuda.cu:29253`); prefetching layer 14's engram rows while layers 2-13 run.
  - Avoid: serial misses, 48 serial engram preads, one GPU only.
  - Measured: V4.1 re-quantized to Q2 decodes at 9.3 tok/s on a DGX Spark (128 GB unified) with an ~80 GiB cache.
- **0xBakeer** (V4.1 on a DGX Spark, native FP4):
  - Take: a transient ring for prefill misses that does not flush the hot set, with promotion on a decode hit and no
    re-read; a warm start from a routing trace; per-layer sizing; DSpark is lossless (64 of 64 greedy tokens
    identical).
  - Measured, native FP4: 2.68 tok/s with 26% of experts resident, a 0.83 hit rate and 0.92 GB read per token. Its
    measured LRU hit curve for V4.1 is the optimistic edge of §4.
  - Its pruning, top-k reduction and cb3 3-bit modes are lossy, and some failed outright ("clearTimeout" came out as
    "cleartimeout").
- **Anemll/ds4-ssd:** read misses while the shared expert computes; do not pin RAM that the page cache needs (a
  measured 13x decode cliff on Metal).
- **llama.cpp:** `repack_mxfp4_blocks` for the bit move; `TENSOR_READ_LAZY` reads engram-style tensors on demand. No
  V4.1, no SSD tier.
- **exllamav3:**
  - EXL3 trellis quants are the best quality per bit measured here (Qwen3.8 MoE KL: 3.05 bpw 0.018, against 0.035 for
    GGUF UD-IQ3_XXS).
  - DSV4 is supported, and it can requantize from FP4 sources.
  - The AVX-512 trellis CPU kernels from exl3 track1 run at 41-57 GB/s on this Xeon.
  - It has no SSD tier.
- **Maya** (GLM-5.3 quants): imatrix statistics from the full model itself, GPTQ-style error-feedback rounding, more
  bits in the first and last MoE layers, and a KL check against the full model.
- **Xolotl** (Indras-Mirror/Xolotl, MIT, a Strata fork; read 2026-10-18 at f7c561b7, cloned to `~/AI/Xolotl`; not
  built or run).
  - **What it is:** V4.1 on one RTX 4090 (PCIe 4.0), an 8-core AVX2 Ryzen, 90 GB of 2-channel DDR4 and a 5.2 GB/s
    NVMe. All numbers below are theirs, measured.
  - **Its headline speed is not exact routing.**
    - 17-30 tok/s (26.8 in its agentic table) comes from a route bias toward VRAM-resident experts.
    - Exact routing on its box: **3.5-6.9 tok/s**.
    - DeepSeek-API top-token agreement on 100 fixtures: FP8 teacher 93.65%; its 2.25-bpw GSQ quant at exact
      routing 86.13%; with bias x0.25 85.1%; x0.5 (served) 81.5%; x1.0 71.8%.
    - HumanEval 21 problems: 20/21 at x0.25 and x0.5, 16/21 at x1.0.
  - **Its published quant** is made from dealignai's abliterated V4.1-Flash (dense weights changed; the experts are
    byte-identical to base V4.1), not 1017. The user does not want it (2026-10-18).
  - **Its quant pipeline** reads the raw HF shards. It fails as published (`v41_calib.py:560`, `ABL` undefined: a
    private ablation step).
  - **Layer sensitivity** (held-out output error at ~2 bpw): layers 20-31 0.03-0.44%, 33-39 0.13-2.1%, 0-3
    2.4-6.8%, 4-19 1.9-12.9%, 32 14.8%. This drives §6's mixes.
  - **Routing is flat:** the hottest 11% of experts cover 42% of routings, 45% cover 78%, 56% cover 86%.
    - A +180-token turn touches ~7,000 distinct experts, ~2,600 of them in its NVMe tier.
    - At exact routing with 53% resident, its adaptive arena held NVMe reads to 24-30 per token (hit 0.875-0.90; our
      middle curve gives 0.904). Its static arena gave 48-85 per token.
  - **Dense:** Q8_0 vs FP8 costs KL 0.015 (top-1 93.60 vs 93.65%); Q6_K at load +1% KL, frees 1.6 GiB.
  - **DSpark:** the draft experts at Q2_K accept 3.233 drafts per pass against 3.280 at MXFP4, and free 2.76 GB.
    Keeping 64 of 128 draft experts gives identical tokens.
  - **Engine choices that differ from ours, with their reasons:**
    - **RAM hits go over PCIe DMA.** Their CPU (~24 GB/s) is level with PCIe 4.0. Ours computes at 74 GB/s against
      6-8 GB/s of PCIe 3.0, so ours computes RAM hits on the CPU.
    - **An expert is three spans** (gate/up/down tensors), read through a bounce buffer. Ours is one aligned record.
    - **One GPU only.**
  - **Taken from it:**
    - **The replay tail of 2,560 tokens**, which corrects our §1.
    - **Arena adaptation**, worth +26-60% at exact routing: swaps between requests, cancelled when a prompt starts.
    - **DSpark:** confidence-gated verify (0.6), a Q2_K or IQ3_XXS draft, and keep-64.
    - **FP4 KV:** bit-exact, +1.1 GiB at 512K.
    - **Read-ahead:** lead-1 recall 67% at top-16 for NVMe misses. Theirs lost speed only because it was issued from
      the decode thread.

## 4. How far it can go (the cost model)

`t_token = fixed + DSpark draft + max(GPU + CPU, SSD) x overlap + (GPU + CPU + SSD) x (1 - overlap)`

- GPU: dense reads (8.5 GB at ~600 GB/s effective) plus VRAM expert hits.
- CPU: RAM hits plus misses, at the kernel rates in §2.
- SSD: missed bytes at 2.4 GB/s.
- overlap: 70% with the read-ahead of §5; 0% without it.
- DSpark: verify window 2, 85% acceptance. The two rows touch 1.7x the experts of one row (0xBakeer measured ~21
  distinct per layer for a 6-row block, against 36 picks).
- Script: `moex/tools/dsv41/cost_model.py` (`--anchors` for the two checks below). It replaces the scratchpad
  scripts lost in the 2026-10-18 reboot, with the same inputs and the same results.

**Hit curves, as a band.** There is no V4.1 routing data from this box yet; phase 2 measures it.
- pessimistic: V4's static learned profile from this box;
- optimistic: 0xBakeer's measured V4.1 LRU curve (one task);
- middle: the geometric mean of the two miss rates.

**Check against two measured runs:**

| run | model | measured |
|---|---|---|
| 0xBakeer, 25.6% resident: hit rate | 0.85 | 0.83 |
| Strata-X V4 file-tier emulation, ~8 first-touch reads per token | 11.6 tok/s | 10.2 tok/s |

**Decode, tok/s, DSpark on.** Cells are the middle estimate [pessimistic .. optimistic].

| expert pack (hot = resident / cold = SSD tier) | avg bpw | pack size | resident | decode (the NVMe at 2.4 GB/s) |
|---|---|---|---|---|
| FP4 / FP4, the release's bits (lossless) | 4.25 | 289 GB | 43% | 3.8 [2.0..7.2] |
| Q3_K / Q3_K (the DSV4 Q3_K recipe) | 3.44 | 234 GB | 54% | 6.7 [3.3..12.7] |
| Q3_K / Q2_K | 3.06 | 208 GB | 54% | 8.5 [4.3..15.6] |
| IQ3_XXS / IQ3_XXS | 3.06 | 208 GB | 60% | 9.0 [4.5..15.7] |
| Q3_K / IQ2_XXS | 2.80 | 190 GB | 54% | 10.3 [5.3..17.4] |
| Q2_K / Q2_K | 2.62 | 178 GB | 70% | 15.9 [8.4..24.8] |

- **Updated 2026-10-18** with the CPU pool's measured rate per format: native_expert_cpu_bench on the quiet box, 20
  threads, clamp 10, 48 experts cycled from DRAM, plain read 76-78 GB/s. Weight GB/s at 1 and 2 tokens per expert:

| format | 1 token | 2 tokens |
|---|---|---|
| MXFP4 (mx512) | 74.4 | 70.9 |
| Q3_K (ggml generic) | 74.5 | 49.0 |
| IQ3_XXS | 50.1 | 42.4 |
| Q2_K | 76.5 | 57.7 |
| IQ2_S | 46.9 | 33.6 |
| IQ2_XXS | 39.0 | 31.7 |

  - In a 2-row verify step, the experts both rows picked run at the 2-token rate and the rest at the 1-token rate.
  - Q3_K and Q2_K reach the DRAM ceiling at one token; the IQ formats are compute-bound on this CPU.
  - An earlier run made while the user's model ran in the background (load average ~26) was discarded.

- **Compute ceiling** (no SSD waits at all): 17-22 tok/s with DSpark (14.5 plain, native). This is the CPU's DRAM
  bandwidth over the RAM-resident experts. For comparison, today's DSV4 Q3_K decodes at 20-27 tok/s with 2.66 GB of
  expert bytes per token; V4.1 has 1.7x the expert bytes.
- **Per-layer mixes** (2026-10-18, `cost_model.py`; formats chosen by Xolotl's measured layer sensitivity, §3). Speed
  follows the total size; quality follows where the bits go. Decode tok/s with DSpark, exact routing, middle
  [pessimistic .. optimistic]:

| mix | avg bpw | experts | resident | decode | +200-token turn |
|---|---|---|---|---|---|
| Q3_K everywhere (the DSV4 recipe) | 3.44 | 234 GB | 54% | 6.7 [3.3..12.7] | 17.6 s |
| Q3_K for layers 0-19 + 32, Q2_K for the 19 insensitive ones | 3.05 | 207 GB | 60% | 9.5 [4.7..17.8] | 13.3 s |
| Q3_K for layers 4-19 + 32, Q2_K for the other 23 | 2.97 | 202 GB | 62% | 10.3 [5.1..19.2] | 12.4 s |
| Q3_K for layers 12-13 + 32 only | 2.69 | 183 GB | 68% | 14.6 [7.5..23.8] | 9.3 s |
| Q2_K everywhere | 2.62 | 178 GB | 70% | 15.9 [8.4..24.8] | 8.6 s |
| **Q2_K for 0-19 + 32, IQ2_XXS the rest (Xolotl's size class; the only row at >= 20 tok/s)** | 2.36 | 160 GB | 78% | **21.2** [14.6..25.0] | **5.7 s** |

  - The Xolotl-class size runs ~21 tok/s here with exact routing. Its own box runs it at 3.5-6.9 exact, ~27 with
    the bias. The extra room here does buy speed for the same pack.
  - A bigger pack spends that room fast: the SSD is 2.2x slower than theirs, and each step down in size cuts both
    the miss count and the bytes per miss.
- **Why Q3_K balanced so well on DSV4, and less so here.** Every V4.1 expert resident in VRAM + RAM (~125 GB) needs
  <= 8.14 MB per expert, which is <= 1.84 bpw, ternary class. DSV4's Q3_K is 119 GB of experts, so it fits and
  never touches the SSD. V4.1 at Q3_K is 234 GB.
  - Q3_K is only 19% smaller than the lossless FP4 release (289 GB), so its speed is closer to the native pack's
    than to the ~2-bit class's.
  - The Q3_K CPU rate is now measured (74.5 GB/s at 1 token, see the table above). It was assumed 47.1 GB/s
    before, which put this row at 6.5 / 13.6.
- **Without read-ahead** (overlap 0), each row drops by about 10-30%.
- **Prefill** (corrected 2026-10-18). Any chunk over ~300 tokens touches nearly all of a layer's 384 experts.
  - A fresh prompt therefore reads layers 0-20's experts once per chunk and layers 21-39's once, for the last 2,560
    tokens.
  - Every expert not in VRAM crosses PCIe to the GPUs, at ~4.8 GB/s with the two cards taking turns. The
    non-resident ones also come off the SSD; the slower of the two sets the time.
  - A short agent turn reads only the experts its tokens route. Fitted on Xolotl's measurement: a +180-token turn
    touches ~7,000 distinct experts, ~2,600 of them in its NVMe tier (HOW_IT_WORKS.md:43-44).
  - Expert-streaming floors from `cost_model.py` (compute overlaps them):

| pack | +200-token turn | fresh 8K prompt | fresh 32K prompt |
|---|---|---|---|
| native FP4 | 26.5 s | 68 s | 176 s |
| Q3_K | 17.6 s | 45 s | 117 s |
| Q3_K / Q2_K | 13.4 s | 39 s | 100 s |
| Q2_K | 8.6 s | 33 s | 85 s |

  - Long prompts are bound by PCIe 3.0 on this box, short turns by the SSD.
  - The lever for long prompts is feeding both cards at once: pipelining prompt chunks across the layer split, so
    both PCIe links (6.25 + 8.1 GB/s) carry experts at the same time. That would be up to ~3x faster, and it is a
    phase 5 item.
  - A disk KV cache (890 B per token) keeps conversations already read from being read again.

## 5. Design

**A. Pack format** (converter: safetensors to a pack, one shard at a time, so peak disk use is pack + one shard).
1. `dense.gguf`: dense weights kept in their released FP8 (E4M3 + E8M0 32x32 blocks) and BF16, so nothing is lost;
   DSpark stage weights; tokenizer; the 129,280-entry engram normalizer table built offline with HF tokenizers.
2. `experts_LLL.bin`, one per layer, expert-major: [w1 | w3 | w2], each in ggml `block_mxfp4` order.
   - 18,800,640 B per expert, exactly 4,590 pages of 4 KiB, so each expert is one aligned O_DIRECT read with no
     padding.
   - Optional pack variants use the same record layout with other block types (IQ3_S, IQ2_S, Q3_K, EXL3) and one
     type per expert, listed in the manifest.
3. `engram_LL.bin`: 15 rows of 264 B per 4 KiB page, so no row crosses a page. That is 105 GB per layer, and one
   4 KiB read per lookup instead of the release's two.
4. `manifest.json`: offsets, types per expert, profile seed, checksums (xxh3 per expert, checked on load and on
   first read).

**B. Tiers.** Strata-X's tier code, extended with an SSD tier.
- **VRAM:**
  - dense (~10 GB incl. head);
  - DSpark (7.9 GB at native, or ~3.7 GB as a 2-bit copy). Draft quality changes only speed, never the output.
  - expert slots, hottest first.
- **RAM:** a pinned arena of expert slots, sized from MemAvailable minus headroom. The page cache is not used for
  experts (O_DIRECT both ways); it keeps the engram rows and the desktop.
- **SSD:** the pack. Placement:
  - warm start from a learned profile, weighted by saliency (gate x output norm);
  - Strata-X's adaptive exchange swaps;
  - a cold LRU of ~128 experts (Strata-X measured +17% on V4);
  - read-ahead entries inserted as the oldest;
  - the prefill ring never evicts the hot set, and a ring expert that decode hits is promoted without a re-read.

**C. Miss path and read-ahead** (the core new code; it removes the idle drive).
1. One I/O engine: a thread pool with O_DIRECT preads into pinned slots, 4 MiB per request, 8-12 in flight.
   - Four priority classes, served in order: demand misses, layer-ahead predictions, prefill sweep, background
     promotions.
   - Speculative reads may use at most 25% of the in-flight slots, so a demand read never waits behind more than
     one of them.
2. A layer's real top-6 issues demand reads at once. The CPU starts that layer's RAM hits, the GPU runs the shared
   expert, and the missed experts are computed on the CPU as each read lands.
3. **Read-ahead for layer L+1:** apply layer L+1's router (3.9 MB of BF16, kept on the GPU) to layer L's output
   stream. It is issued only above a probability threshold, and only for experts not resident.
   - It is enabled only if phase 2 measures enough precision. Phase 2 also measures leads of 2 and 3; Strata-X's
     pre-gate at lead 3 on V4 had 50% recall but only 16% precision.
4. **Engram:** row numbers are computed on the CPU when a token is sampled or drafted.
   - The 48 reads are issued in parallel, with a ~2 GB RAM row cache in front.
   - Layer 1 waits on them; layer 14's reads overlap layers 2-13.
   - Prompts: deduplicate n-grams, keep up to 128 reads in flight, and run them alongside the layer 0-13 sweep. A
     32K prompt is ~1.5M reads, ~9 s at 170K IOPS.

**D. Compute.**
- **GPU:**
  - FP8 to FP16 GEMV and GEMM kernels for the FP8 dense weights; Ampere has no FP8 tensor cores, so the
    dequant happens in-kernel and the weights stay as released.
  - V4.1 attention: Strata-X's DSV4 kernels carry over, with new code for ratio-2/ratio-1 compression without
    overlap, KV and indexer sharing across layers, and layer 20's candidate top-2048 blocks; the FP4 cache packing
    is the reference's own.
  - hyper-connections (hc_dim 20480); the engram gate and wkv; VRAM experts.
  - Layers 0-19 on GPU0 and 20-39 plus DSpark on GPU1, to be tuned by measurement as today.
- **CPU:** the mx_avx512 MXFP4 kernels (2-token path for DSpark), or the IQ / EXL3 kernels for packs.
- **DSpark:** verify window 2 to start, as on DSV4 today; the confidence head picks 1-3 per step once measured.

**E. Prefill.**
- Bounded replay: layers 0-20 over the whole prompt, then 21-39 over the last 2,560 tokens (128 x 20, see §1), plus
  the DSpark feature rows.
- Big chunks (8K-32K) so each pass over the experts serves more tokens.
- Per layer: resident experts first, while the SSD streams the rest into the lend ring (double-buffered) and on to
  the GPU.
- Disk KV checkpoints per conversation (the KV is small), extending Strata-X's checkpoints.

## 6. Quality policy (see the user's rule: reliability first)

- The native FP4 pack is the reference: the released bits, bit-exact after the shuffle.
  - Phase 1 checks it against the release's own model.py, layer by layer, with real weights (one layer of tensors at
    a time on the CPU), plus teacher-forced logits on held-out text.
  - It is the slow but always-available mode.
- Every other pack has to pass against the native pack on the same texts before it is offered:
  - KL divergence, same-top-token rate and NLL;
  - plus Maya's long-answer loop test (14K tokens at temperature 1 and greedy, no repeated passages).
  - **The pass mark** is the user's daily DSV4-Flash Q3_K against its own MXFP4 release, set before measuring.
    - Measured with llama.cpp `perplexity --kl-divergence` on wiki text, 64 chunks of 512 tokens
      (`~/AI/Models/dsv41-1017/ref-runs/dsv4-flash-q3k/kld-q3k.log`, with that quant's recipe, README and logs: copied
      from `~/AI/Models/GGUFs/dsv4-vision/Q3_K-tarruda/` on 2026-10-19, so the pass mark's log outlives the DSV4-Flash weights).
    - Mean KLD 0.134, same top token 88.0%, perplexity x1.101, 99th-percentile KLD 1.90.
    - A V4.1 pack passes at or below that KLD and at or above that same-top rate, on the same kind of text plus
      code and agent transcripts.
- **The target recipe** (the user's choice, 2026-10-18) is the DSV4-Flash Q3_K recipe
  (`~/AI/CustomQuantRecipes/tarruda-dsv4-vision/scripts/quantize.sh`), carried over to V4.1:
  - routed experts gate/up/down: **Q3_K** (3.44 bpw), weighted by an imatrix collected from the native 1017 model;
  - dense weights: **Q8_0** (that recipe's default), so the GPU runs the existing Q8_0 kernels instead of new FP8
    ones;
  - embedding and the compressor/indexer projections: BF16 (the output head went Q8_0: write_dense_gguf.py writes
    `output.weight` with put_q8, found by the audit of 2026-10-20); the hyper-connection, router-bias and other small
    tensors: F32, as in that recipe;
  - engram tables: as released (FP8 rows on the SSD; they are read 48 rows per token, so their size costs nothing);
  - DSpark experts: free to go lower. Draft quality only changes speed, never the output.
  - The quantizer is ggml's own (`ggml_quantize_chunk` with the imatrix, as llama-quantize calls it), so a block is
    what llama.cpp would write. llama.cpp has no V4.1 model code, so the tool reads the native pack directly.
  - The imatrix comes from the native model on calibration text. The DSV4 recipe's
    `data/imatrix-calibration.txt` is used, plus code and agent tool-call transcripts. Collecting it needs a working
    V4.1 forward, so it is a phase 1 output.
- **One improvement inside the same format**, measured before it is kept: GPTQ-style error-feedback rounding of the
  Q3_K blocks, with each expert's input Hessian from the same native run (Maya-M, and Xolotl's pipeline for its Q3_K
  experts). Same bytes and kernels as the plain imatrix Q3_K; the one with the lower KLD is kept.
- **Speed variants, only if the user wants more speed at the same pass mark:** Q3_K for the resident experts and Q2_K
  for the SSD tier (§4: 8.2 vs 6.5 tok/s middle). The per-expert bit choice is static, so the same prompt gets the
  same weights whatever the cache holds.
- **Activations:** the release's own forward pass rounds the input of every FP8 and FP4 matrix multiply to FP8
  (E4M3, one power-of-two scale per 32 values; model.py:181-202, `act_quant`). The engine's kernels keep more
  precision: ggml's MXFP4 path rounds activations to int8 with an fp16 scale per 32, and the FP8 dense kernels run
  in fp16/fp32. The weights are bit-exact either way.
  - Phase 1 measures two numbers on the same texts, each held to the bar of the next point:
    1. KL against an fp32-activation reference with the exact weights. This is the gate (≈ noise).
    2. KL against an emulation of the release's FP8 activations, which shows how far the engine sits from
       DeepSeek's own serving numerics.
  - An exact FP8-activation mode would need kernels that emulate FP8 activations; it is not planned unless (2) is
    large.
- **Not used by default** (each changes the output): expert pruning or substitution, top-k below 6, skipping cold
  experts, opportunistic speculative sampling, and activation-sparsity skipping of w2 columns.

## 7. Phases and gates

| phase | work | needs | gate (measured here) |
|---|---|---|---|
| 0 | converter design; engram hash + normalizer in C++, tested against engram.py; FP4 to MXFP4 repack, tested bitwise on real experts (range reads, ~40 MB); I/O engine micro-bench with priorities | tokenizer + range reads only | bit-exact repack; engram rows equal to the reference for 10K tokens; 2.4 GB/s demand reads with speculation capped |
| 1 | download (510 GB) and convert to the native pack; V4.1 forward in strata-glm (attention kinds, hyper-connections, engram, bounded replay), experts from RAM + SSD with no read-ahead | ~520 GB of disk | per-layer match with model.py; teacher-forced KL of native vs reference ≈ noise; bounded replay equal to the full prompt pass |
| 2 | route capture on the user's workloads (code, agent, prose; the learned profile); the real hit curve; read-ahead precision and recall at leads 1-3 | phase 1 | replaces the band in §4 with a measured curve; read-ahead on only if it nets a win |
| 3 | SSD tier, priority I/O, read-ahead, cold LRU, engram prefetch | phase 2 | decode on native within 20% of the model at the measured curve |
| 4 | DSpark (window 2, then the confidence head) | phase 3 | greedy output equal to plain decode; the speed gain measured |
| 5 | prefill: big chunks, double-buffered streaming, disk KV | phase 3 | the pass time per prompt within 20% of the model |
| 6 | the pack: the per-layer Q3_K/Q2_K mix at ~203 GB (the user's choice, 2026-10-18), from the clean 1017 weights with an imatrix from the native model and ggml's quantizer (the GPTQ-rounded variant kept if it measures better); the layer split from the measured sensitivity | phase 1 reference (imatrix + saved reference logits) + ~160-207 GB per pack (+ ~8 GB Q8_0 dense) | decode >= 20 tok/s (the user's bar); KLD, same top and task checks against the native pack, reported next to the DSV4 Q3_K's 0.134; the user picks |

## 8. Risks and open questions

- **The hit curve is the biggest unknown.** The bands span 3x. Hot sets differ by task: 0xBakeer measured a Jaccard
  overlap of 0.18-0.31 between the coding and general top-25% sets. Phase 2 measures it on the user's work.
- **The drive is DRAM-less and sits on the chipset.** It is shared with the desktop: Strata-X traced a desktop
  freeze to I/O starvation (PSI io full 22%). The I/O engine keeps its queue bounded, and long runs go under the
  existing io_watchdog.
- **The CPU pool's 70 GB/s and the SSD's DMA share four DDR4 channels.** Measured: 2.36 GB/s of SSD reads alongside
  74 GB/s of DRAM reads. The cost of the reads to the pool is not yet measured; it shows in phase 3.
- **FP8 dense kernels on Ampere are new code.** If they are slow, the dense weights could become Q8_0 (lossy, small),
  which would need the §6 gate.
- **Bounded replay and the candidate indexer are new to the engine.** Phase 1's gates check them against the
  reference.
- **Short agent turns pay to read the experts they route:** ~6-27 s depending on the pack (§4). The disk KV and
  smaller packs shorten it.
- **Disk for the phase 6 packs:** 160-207 GB each (+ ~8 GB dense), while the native pack (499 GB on /) leaves ~22 GB
  free. The options:
  - free that much more; or
  - once the native model has produced the imatrix and the saved reference logits, delete the native experts
    (289 GB, keeping the engram tables, which both packs share). The lossless mode would then need the 40 expert
    shards converted again, ~5-10 h of download.
  - The user decides when phase 6 starts.

## 9. Phase 0 results (2026-10-17, no weights, no GPU)

All the code is in `moex/tools/dsv41/`. The reference is the release's own `inference/engram.py`, `convert.py` and
`kernel.py`, run in the llama.cpp venv with the GPUs hidden.

**1. Engram row numbers** (`engram_tables.py`, `engram_hash.hpp`, `test_engram_hash.py`): **pass.**
- The tables equal the reference module's buffers: the token map (129,280 to 99,092 compressed ids, as the config
  asserts), the pad id, the 8 multipliers, the 48 primes and the offsets. The primes sum to each table's row count.
- The C++ row numbers equal NgramHashState's on 303,600 rows. The six sequences were 3,000 tokens of text, random
  ids including the special tokens, image spans (one across the prefill/decode split), pad tokens and a dead first
  token. Each was fed as a prefill followed by one token per step, as the engine will use it.
- The HF config.json calls the pad field `engram_pad_token_id` and the inference config calls it `engram_pad_id`
  (both 2); the builder reads either.

**2. FP4 to MXFP4** (`fp4_repack.py`, `test_fp4_repack.py`): **pass.**
- ggml's own MXFP4 decoding (gguf-py) of the repacked blocks gives exactly the release's floats, bit for bit, for
  w1, w3 and w2 of four real experts: layers.0 #0, layers.20 #191, layers.39 #383 and mtp.2 #127 (range reads from
  Hugging Face).
- Every record is 18,800,640 B, 4,590 whole pages.
- The scale bytes span only 117-127. A later lossless squeeze of the 1.1 MB of scales per expert is possible, about
  4%, but not planned.

**3. Priority I/O** (`expert_io.hpp`, `expert_io_bench.cpp`; 8 reader threads, 4 MiB pieces, O_DIRECT; random
experts in a GLM shard): **pass.**

| demand only | batch time | drive |
|---|---|---|
| 1 miss | 7.1 ms (p90 7.4) | 2.63 GB/s |
| 2 misses | 14.2 ms | 2.64 GB/s |
| 8 misses | 56.1 ms | 2.67 GB/s |

- One expert alone already fills the drive, because its pieces are read in parallel.

| read-ahead keeping 16 experts queued, cap on its threads | 2-miss demand batch | demand + speculative |
|---|---|---|
| 1 of 8 | 15.5 ms (+9%) | 2.43 + 0.24 GB/s |
| 2 of 8 | 16.7 ms (+18%) | 2.25 + 0.45 GB/s |
| 4 of 8 | 19.3 ms (+36%) | 1.95 + 0.79 GB/s |
| 8 of 8 | 24.9 ms (+75%) | 1.52 + 1.21 GB/s |

- The drive's total stays ~2.67 GB/s, so the cap is a direct trade between demand latency and read-ahead.
- Phase 2 sets the cap from the measured precision of the read-ahead. A wrong prediction costs a demand miss about
  7 ms times the share of the drive it took.

## 10. Hardware levers (reference for the user's own upgrade planning; not part of this plan)

The plan above targets the current box only (the user, 2026-10-18). This section keeps what the analysis found about
hardware so it is on record. The numbers come from `cost_model.py --hardware-levers`, as middle estimates at exact
routing.

**Storage bandwidth.**
- The NVMe sits on the chipset at Gen3 x4: 2.4-2.76 GB/s.
- CPU stack 16 runs GPU0 at x8, and its other x8 (ports 16:01-16:03) is empty (lspci 2026-10-17). Two more Gen3 x4
  drives there would total ~8.9 GB/s, if the slot and BIOS allow x4x4 bifurcation; not checked against the board
  manual.
- The engine would stripe the pack across drives by temperature. That would be a later phase.
- Xolotl's second-drive mirror gave +33% in a microbenchmark and nothing in its engine, but that drive was exFAT
  (0.96 GB/s).

| pack | decode, 1 NVMe | decode, 3 NVMe (~8.9 GB/s) | +200-token turn, 1 / 3 NVMe |
|---|---|---|---|
| native FP4 (lossless) | 3.8 | 11.7 | 26.5 / 7.1 s |
| Q3_K everywhere | 6.7 | 17.1 | 17.6 / 4.7 s |
| Q3_K for layers 0-19 + 32, Q2_K for the rest | 9.5 | 20.3 | 13.3 / 3.6 s |
| Q3_K for layers 4-19 + 32, Q2_K for the rest | 10.3 | 21.1 | 12.4 / 3.4 s |
| Q2_K everywhere | 15.9 | 25.3 | 8.6 / 2.3 s |
| ~2.3 bpw (Q2_K + IQ2_XXS) | 21.2 | 24.3 | 5.7 / 1.5 s |

- Long prompts would barely move, because they are bound by PCIe 3.0.

**Memory channels** (MXFP4.md:284-288): 4 of the Xeon's 6 DDR4 channels are populated. Slots C1 and F1 are empty.
Filling them would raise the CPU pool's bandwidth by ~50% and the expert memory by the added capacity. Not modelled
here.

**Network:** downloads ran over Wi-Fi at 5-20 MB/s. The board's two 10 GbE ports are unused.

## 11. Phase 1 progress (2026-10-18)

**The reference harness** (`moex/tools/dsv41/ref/`): DeepSeek's own `inference/model.py` running on the CPU from
the native pack, one layer at a time.
- `ref_kernels.py` stands in for the six tilelang kernels: act_quant, fp4_act_quant, fp8_gemm, fp4_gemm,
  sparse_attn and hc_split_sinkhorn.
  - It follows their arithmetic: fp32 compute, the same scale rules, round-to-nearest-even into FP8 and FP4.
  - Results agree to fp32 rounding, not bitwise, because the accumulation order differs.
  - `DSV41_REF_GEMM_ACT=fp8|fp32` sets whether GEMM inputs (including the wo_a input) are rounded to FP8 as
    DeepSeek serves them, or left in fp32 for the phase 1 gate. Storage roundings (window KV, compressed KV, index
    q/k) stay in both modes.
- `ref_model.py` builds one Block at a time:
  - dense tensors from the pack's dense safetensors, converted as `convert.py` does for one rank (wo_a to BF16);
  - the 384 experts from the record file, turned back into the release's FP4 layout by `fp4_repack.unrepack`,
    which is bit-exact on random round trips and on a real record;
  - engram rows read by number from the table files.
- `ref_run.py` dumps per-layer residuals, routed expert ids and logits.
- `--no-engram` skips layers 1 and 14's engram additions. It is a like-for-like mode for checks while the tables
  download, not the real model.
- `test_ref_kernels.py` passes, including the FP4 dequant against the phase 0 reference on a real expert.

**First runs:**
- Layer 0, 64 tokens (the real model): every dense tensor matched its parameter by name and shape. Load 37 s, run
  16 s, the output finite, 160 distinct experts routed.
- Layers 0-3 with `--no-engram`, through the ratio-2 compressor and the first KV/index source: finite, ~40 s per
  layer.
- Running: 40-layer baselines with `--no-engram` (128 tokens in fp32 and fp8 mode; 2,048 tokens in fp32 mode, so
  that the top-512 index selection is exercised) in `~/AI/Models/dsv41-1017/ref-runs/`.
- The candidate filter (2,048 blocks of 8) only cuts above 16,384 positions, which is too long for a CPU reference
  run, so it gets a unit-level check instead.

**The engine port, first pass written (2026-10-18 afternoon; built, not yet run on a GPU).** The user's GLM server
held both GPUs all afternoon, so nothing below has run on a GPU yet. Everything is built in a separate
`build-dsv41/` so the live `build/strata-glm` stays untouched.
- `src/glm/dsv41_kernels.cu` holds the new kernels:
  - FP8 and FP4 rounding of cached values;
  - the ratio-2 and ratio-1 compressors and the ring persist;
  - the completed-row store;
  - layer 20's candidate blocks, one bit per block per row, and the mask the later index sources apply;
  - the one-hot pre-mix;
  - the engram's row dequant and gate.
  V4's kernels take the ratio as a parameter; V4's own calls are unchanged.
- `src/glm/glm_dsv41.cpp` holds the rest:
  - the `deepseek41` loader (`deepseek41` maps to `Arch::dsv4` with `v41` set);
  - the attention (`Stage::dsv41`);
  - the shared decode state (caches only on the KV sources);
  - the record-file expert source;
  - the engram (S4: host hash per position, rows copied from the mapped tables, dequant → `wkv` → gate on the GPU before
    layers 1 and 14);
  - the split rule: a two-GPU split shares no compressed rows, list or candidates, and keeps the engram layers on the first GPU. That leaves only layer 20.
- The hyper-connection order is in `mhc_half`. The previous half's pre-mix is carried per row. It crosses the stage
  hand-off, the decode graphs' mapped hand-off and the prompt path's pinned hand-off, and the head reads it.
- The expert records also feed the VRAM slots and the RAM copy: `Impl::src` and the load pass read one role per
  layer and copy the blob once. So the tiers are not limited to in-place reads.
- `kPoolMaxFF` is now 2304. It only sets buffer sizes; narrower models compute the same.
- `src/glm/dsv41_kernels_parity.cpp` (S1) checks each new kernel against an independent CPU transcription, each with
  a negative control:
  - roundings, row stores, candidate bits, one-hot and dequant: bitwise;
  - compressor, indexer scores, attention and engram gate: relative L2.
  It also covers V4's kernels at ratios 1 and 2.
- `moex/tools/dsv41/ref/ref_oracle.py` (S2) runs model.py one token at a time. It writes every value the engine
  probes, in llama_dump's format, so `strata-glm --oracle` checks each op on the reference's own inputs.
  - On layers 0-3 its decode steps match the prefill harness (layer 0 within 1.6e-4; later layers differ where a
    route flips, as bf16 inputs make them).
  - A 40-layer, 12-step oracle in fp32 GEMM-activation mode is in `ref-runs/oracle-12-fp32`.
- Pending GPU time: the two parity tests (V4's must still pass unchanged), then the oracle check, then decode-window
  KL against the fp32 baseline (`ref-runs/noeng-128-fp32`, done in 45 min).

**First GPU run (2026-10-18 evening, `build-dsv41/`, one RTX 3080 unless noted; native MXFP4 experts read in place; no
engram on either side, since its tables are still downloading).** Logs are in `~/AI/Models/dsv41-1017/ref-runs/engine-checks/`.
- **S1, kernel parity:**
  - `dsv41_kernels_parity`: every check passes, every negative control fails as it should. The roundings, row
    stores, candidate bits and dequant are bitwise; the reductions are within ~1e-7. It covers the decode start
    (1 and 8 rows at positions 0-8) and the DSpark block mode.
  - `dsv4_kernels_parity`: 0 failures, so V4's kernels are unaffected.
- **S2, op-level oracle (12 tokens, 40 layers, fp32-activation reference):**
  - Every index selection and every routed expert set equals the reference's at every step.
  - The per-op errors are flat across layers:
    - mHC ~1e-7;
    - norms and layer outputs ~1.7e-3, the reference's bf16 rounding;
    - projections ~5e-3, from the Q8_0 weights;
    - expert rows ~1%, from the CPU kernels' Q8 activations.
  - Only four records pass 5%, all FP4/FP8 rounding sites where the reference rounds its inputs to bf16 first: the
    index keys of layers 2/8/14 (worst 6-10%, mean ≤ 1.5%) and layer 20's FP8 window key.
- **S3, end to end against the fp32 reference:**

| text | read | mean KLD | same top | PPL ratio |
|---|---|---|---|---|
| 128 tokens (63 scored) | windows | 0.032 ± 0.003 | 87.3% | 1.074 |
| 128 tokens (63 scored) | prompt path | 0.033 ± 0.004 | 93.7% | 1.078 |
| **2,048 tokens (1,023 scored)** | **prompt path** | **0.0185 ± 0.0008** | **93.2%** | **1.011** |
| reference fp8 mode vs fp32 mode, the 63 tokens | - | 0.034 ± 0.004 | 93.7% | - |

- **Splits and windows:**
  - Two GPUs split at layer 20 equal one GPU in every printed statistic.
  - Windows of 1, 2 and 8 rows are identical on a 16-token text.
- **Rounding sensitivity (a property of the model, not a bug):**
  - The SIMT attention and its tensor-core twin agree to ~1e-7 (both pass parity). FP8 rounding absorbs the
    difference on most layers.
  - Where one element sits on an FP8 step, or a routing near-tie sits within ~1e-5, it jumps: layer 39's 6th and 7th
    experts were 8.5e-6 apart at step 0.
  - So different kernels or read paths give per-token outputs that wander apart over a few dozen tokens. The model's
    own fp8 and fp32 modes do the same.
  - Quality comparisons therefore need long texts.
- **S7, DSpark (`--draft-dspark builtin --spec 2`):**
  - Greedy output equals plain greedy (32 tokens, the prompt through windows; two runs identical).
  - 71-73% of drafts are accepted, 1.5 tokens per verify window, 18-23 ms of drafter time per window.
- **Generation:** coherent English on a natural prompt through both read paths. Greedy repeats within ~40 tokens on
  a bare (non-chat) prompt.

**Speed work on the native pack (2026-10-18 evening).** Two GPUs split at layer 20, the DSpark drafter loaded, no
engram yet (its second table landed at 17:31). Logs, routing traces and profiles are in
`~/AI/Models/dsv41-1017/ref-runs/engine-checks/` and `~/AI/Models/dsv41-1017/profiles/`.
- **The SSD tier.** Experts in neither VRAM nor RAM are read with O_DIRECT into pinned slots (`expert_io.hpp`: 12
  readers, 4 MiB pieces, demand reads before read-ahead), never through the page cache.
  - The first speed run read them through an mmap of the record files. That churned ~2-3 GB of page cache per token
    next to 87 GB of pinned RAM, and systemd-oomd killed the terminal's scope (memory pressure 54.68% > 50% for 20 s).
  - Engine runs now go in their own `systemd-run --user --scope` units.
- **Where the time went first.** At load: 1,482 VRAM slots (842 + 640, 26 GB), a 76.9 GiB RAM copy (~4,290 experts)
  filled in layer order (no profile), the rest (166 GiB) on the SSD; loaded in 43.8 s (2.37 GiB/s).
  - Decode was SSD-bound: ~60 records of 17.9 MiB per token at ~2.76 GB/s (the drive's measured ceiling), ~410 of
    ~483 ms per token.
- **DSpark windows did not help while the SSD was the limit.** One text, warmed up on the same text (so the tiers had
  seen it; see the cache rows below for a fresh text):

| window | tok/s | tokens per window | SSD reads per token | SSD wait per token |
|---|---|---|---|---|
| 2 | 1.90 | 1.70 | 64.8 | 439 ms |
| 3 | 2.05 | 2.31 | 61.2 | 416 ms |
| 4 | 1.96 | 2.56 | 63.2 | 432 ms |
| 1 (plain) | 2.07 | 1.00 | 60.2 | 410 ms |
| 2 (again) | 2.09 | 1.79 | 59.5 | 403 ms |

  - A window verifies several tokens in one pass, but those tokens route mostly to different experts (81% of a
    3-token union is distinct), so SSD reads per output token stay the same at every window size.
- **Read-ahead (plan §5 C.3) with the 64-slot cache.** The GPU guesses layer L+1's experts from layer L's FFN rows
  (`Stage::ahead_predict`); the host reads the guesses that are on the SSD.
  - 71-72% of the guesses were used, as the offline estimate said (65-71%).
  - But the wrong guesses added 21-28% to the SSD traffic, on a drive already ~97% busy. Its off arm was not run:
    the cache below changed the picture first. Read-ahead in the cache's regime is still to be measured.
- **The routing trace and the tier simulator.** `STRATA_ROUTE_TRACE=FILE` writes every decode routing (int32:
  layer, T, k, ids); `moex/tools/dsv41/tier_sim.py` replays traces against tier policies.
  - Calibrated on a prose text's warm-up and replayed on a coding text, it gave today's tiers 88.5 SSD reads per
    token, 105 VRAM and 45 RAM hits; the engine measured 93.3, 100 and 46.
  - It predicted ~28 SSD reads per token if most of the RAM budget were a least-recently-used cache of SSD records
    instead of a fixed copy. A profile helped the fixed copy (88.5 → 56.4), not the cache.
- **The SSD cache (`--ssd-cache-gib`).** The decode's records stay in pinned slots, least recently used out first.
  The cache is allocated before the RAM copy is planned, so it comes out of the RAM copy's budget: the same memory.
  The prompt path and the VRAM fills take records from it instead of reading them again.
  - Same sequence as above (warm-up on a 182-token prose prompt, then a 77-token coding prompt), plain decode unless
    noted:

| run | 64 slots + 76.9 GiB copy (read-ahead on) | 60 GiB cache + 18.2 GiB copy (read-ahead off) |
|---|---|---|
| warm-up, prose text (DSpark w2) | 1.08 tok/s, 133 SSD reads/token | **2.55 tok/s**, 43.7 |
| coding text, first time | 1.20 tok/s, 93.3 (+25.8 wasted read-ahead) | **3.78 tok/s**, 26.5 |
| coding text, DSpark w2 | - | **4.41 tok/s**, 22.6 |
| coding text again (seen twice) | - | 5.02 tok/s, 19.6 |

  - The VRAM fills of SSD experts now come from the cache: 2.05 per token from it, 0.49 read again (64 slots: 0.48
    and 1.96).
  - With fewer SSD waits, DSpark helps again (+17% at window 2).
  - The simulator's estimate for the smaller packs with the same memory (no profile; not measured): ~20 SSD reads
    per token for the 203 GB Q3_K/Q2_K pack, ~12.5 for a ~160 GB pack.
- **Prefill before the stream gate.** One loaded model, prefixes of the 7,820-token prompt, one token generated:

| prompt | time | tok/s | records read from the SSD |
|---|---|---|---|
| 32 tokens | 81.7 s | 0.4 | 9,666 |
| 200 tokens | 81.8 s | 2.4 | 9,706 |
| 2,000 tokens | 84.1 s | 23.8 | 9,918 |
| 7,820 tokens | 97.6 s | 80.1 | 11,186 |
| 200 tokens again | 81.9 s | 2.4 | 9,706 |

  - The prompt stream issues every non-resident expert of a layer ahead of its routing. That costs nothing from RAM,
    but from the SSD it read the whole tier (~9,500 records, ~178 GB) for any prompt: a 32-token prompt routes to
    ~2,600 of them.
  - The fix (`Stream::gate`): a chunk of at most 1,024 tokens (`STRATA_DSV41_STREAM_GATE`) issues each layer's
    entries after its routing, and only the routed ones. Longer chunks route to nearly every expert and still stream
    ahead.
- **Prefill with the gate** (same runs; the first token is the same in every row):

| prompt | before | gated | records read from the SSD |
|---|---|---|---|
| 32 tokens | 81.7 s | **16.6 s** | 9,666 → 1,922 |
| 200 tokens | 81.8 s | **41.2 s** (41.4 s again) | 9,706 → 4,837 |
| 2,000 tokens (not gated) | 84.1 s | 83.8 s | 9,918 → 9,891 |
| 7,820 tokens (not gated) | 97.6 s | 95.7 s | 11,186 → 11,159 |

  - The gated reads ran at ~2.2 GB/s: the issuer read one record at a time. It now keeps records in flight in the
    bounce slots (`Prefetch`), as far ahead as the routing is known. With 7 in flight: 32 tokens 15.9 s, 200 tokens
    39.4 s (twice), 2,000 tokens 80.2 s, 7,820 tokens 96.1 s - a few percent. The long reads still average ~2.2-2.3
    GB/s: with the 32-slot ring full, the drive idles while each layer's attention runs.
- **Read-ahead with the 60 GiB cache** (same sequence as the cache table): 2.36 tok/s on the warm-up (2.55 off) and
  3.43 on the coding text (3.78 off), with 59.9 and 38.1 SSD reads per token (43.7 and 26.5 off). 74% of its guesses
  were used, but each guess takes a cache slot and evicts a record the decode wanted later. It is now opt-in
  (`STRATA_DSV41_AHEAD=1`); the default is off.

**The goal from here (the user, 2026-10-18 evening):**
1. Run the best quant this box can run: the best balance of quality and speed.
2. Run it at the highest prefill and decode speeds this box allows.
3. Make the theoretical speed of that quant match the measured speed.

The user added a rule: no test that does not change a decision toward these goals.

**Toward it (2026-10-18 evening):**
- **The engram is in.** The engine's GGUF is now `~/AI/Models/dsv41-1017/dsv41-1017-dense-q8_0-dsp-eg.gguf` (dense
  Q8_0, the DSpark drafter, the engram projections; 10.56 GB, 94 s to write).
  - 96 greedy tokens on the coding prompt are coherent with it, and with it off (`STRATA_DSV41_NO_ENGRAM=1`).
  - S4's op-level check is running: the 12-step reference with the engram (`ref-runs/oracle-12-fp32-eg`). The
    planned 1,024-token engram reference was dropped: S3 already checked every other op end to end.
- **The imatrix** (engram on): the DSV4 recipe's calibration text, 24 chunks of 8,192 tokens, each starting with
  BOS (196,608 tokens), collected through the prompt path with no RAM copy. 108 s a chunk.
- **The quality yardstick for packs** is the same as the DSV4 pass mark: wikitext-2 test, 512-token chunks (16 of
  them, `quality/wiki-16x512-tokens.bin`). The native engine writes its rows once (`--kld-save`); each pack is
  measured against them (`--kld`). That is quant against the release format, as the DSV4 Q3_K's 0.134 was.
- **Decode: the pool computes the cache's records while the missing ones are read** (`run_cpu_jobs`). It used to
  compute them after the reads. The result is the same (each job writes its own rows, the pool splits by rows).
- **The speed model** (`moex/tools/dsv41/speed_model.py`) replays a routing trace against a pack's tiers and times
  each layer from measured parts:
  - the GPU's own time per token (the router wait, 33.5 ms);
  - the CPU pool's rate and per-call cost;
  - the SSD's demand-read rate (2.76 GB/s);
  - per layer, the pool computes the experts it has while the missing records are read.
  - Calibrated on the native run, it gives 3.87 tok/s for that run (measured 3.78). Its predictions, plain decode
    on the coding text:

| pack | today's slots | slots per record size |
|---|---|---|
| native (289 GB) | 4.08 | - |
| Q3_K 4-19 + 32, Q2_K the rest (202 GB) | 6.81 | 7.65 |
| Q3_K 0-19 + 32, Q2_K the rest (207 GB) | 6.73 | - |
| Q2_K everywhere (178 GB) | 9.56 | - |

  - A mixed pack loses ~12% to its slots: every VRAM and SSD-cache slot is sized for the largest record (the
    engine's rule picks the size that holds the most experts per byte, which here is Q3_K's), so the Q2_K layers'
    records leave 24% of a slot empty.
- **The first pack: Q3_K on layers 0-19 and 32, Q2_K on 20-31 and 33-39 (207.4 GB, `~/AI/Models/dsv41-1017/q3q2`).**
  - Xolotl measured layers 0-3 as sensitive too (2.4-6.8% output error at ~2 bpw, against <= 0.44% for 20-31). The
    model puts protecting them at ~1% of decode speed (6.73 vs 6.81 tok/s for the 202 GB split).
  - `/` had 220.8 GB free after the repartition, so it fits with ~13 GB to spare. (An earlier note said it would not
    fit: that read `df -h`'s GiB as GB.)
  - The pack directory links the native engram tables and the drafter's experts (`experts/M0-2.bin`): nothing is
    copied.
  - `requant_pack` now drops each layer's source and output pages from the page cache once they are written.
- **The SSD cache now has one pool of slots per record size** (`RecordIo::pool_stride`), each its share of the
  bytes by its layers' record bytes. A pool falls back to a pool of larger slots when it is full.
  - On the 207 GB mix the model gives 4,771 slots instead of 4,236, and 6.73 → 7.22 tok/s.
  - VRAM slots sized per GPU would add ~2% more (GPU 1 sized for Q2_K, layer 32 then off VRAM). It touches ~25
    places in the tier code, so it was not done.
- **S4, the engram: passed** (19:13, the 12-step oracle with the engram on both sides). Layers 1 and 14's engram rows are
  bitwise equal; the projection and the gated output are within 5e-3 and 1.6e-3, as every other op. Every routed expert set
  and index selection is equal at every step.
- **Bounded replay is written** (`replay_init`, `Stage::replay_shift`; on for V4.1, `STRATA_DSV41_REPLAY=0` for the full
  pass).
  - Layers 21-39 keep only their 128-row raw rings and read layer 20's compressed rows and index keys. So a prompt runs them
    over the last 128 x 20 = 2,560 rows before each place a later read starts from: the prompt's end and each checkpoint
    cut.
  - A chunk with none of those rows skips them, and streams none of their experts.
  - Its gate (S5): a 7,820-token prompt and 64 greedy tokens must be identical with replay on and off.
  - It matters most past 8K tokens, where earlier chunks skip half the layers' expert streaming.
  - **First try: an illegal memory access in the prompt path.** The shift moved the streams and the carried pre-mix
    but not layer 20's per-row top-k lists (layers 21-23 reuse them) or its candidate masks (the later index sources
    filter by them).
    - With those shifted too, **S5 passes** on the 7,820-token prompt: the 64 greedy tokens are identical with replay
      on and off.
    - **The multi-chunk case passes too:** on a 12,000-token prompt the 64 greedy tokens are identical, and it reads
      in 156.8 s instead of 200.7 s (22% faster). 18,480 experts are streamed instead of 26,494, because the first
      8K chunk skips layers 21-39 entirely. Since then it is on by default (`STRATA_DSV41_REPLAY=0`: the full pass).
    - On one 8K chunk it saves nothing measurable: 109.7 vs 109.3 s with the pack build running beside it. The tail's
      2,560 rows still route to nearly every expert, and the prefill is SSD-bound.
- **An opt-in route bias is written** (`STRATA_DSV41_ROUTE_BIAS=M`, off by default: it changes the output). In decode, an
  expert the SSD would have to read - in no VRAM slot, no RAM copy, no SSD cache slot - has its top-k selection score
  lowered by M. The weights stay the release's (the unbiased probs of the chosen experts).
  - Unset, the routing is bitwise the release's: the kernel subtracts exactly 0.
  - On the pack it would remove up to the ~16.5 SSD reads a token, ~55% of the token's time. Its quality cost, against
    the speed, is to be measured at a few margins after the exact pack numbers.
  - **How far the router's choices sit apart** (the 12-step engram oracle's router logits, 480 layer-steps; selection score
    = sqrt(softplus(logit)) + the bias, top scores ~14, biases 9-58): the 6th expert leads the 7th by a median of 0.019
    (25th percentile 0.007), and a selected expert leads the 7th by a median of 0.13.

| margin | selected experts leading the 7th by less |
|---|---|
| 0.01 | 7% |
| 0.02 | 13% |
| 0.05 | 28% |
| 0.1 | 43% |
| 0.5 | 79% |

    So a near-tie bias (<= 0.02) can swap few of the ~16.5 SSD reads a token. Removing most of them needs margins of
    ~0.5, well past near-ties; the margins to measure are 0.1 and 0.5.
  - **Against Xolotl's** (its README.strata.md, README.ds4.md): the same kind - selection score only, weights exact, every
    expert reachable - but two terms on V4.1: a bonus for VRAM-resident experts (`--route-bias`, 1.0 default, 0.5 served)
    and a penalty for those read from NVMe (`--route-bias-file`, 0.6 / 0.3). In the same score units, a VRAM expert and an
    NVMe expert differ by 0.8 at its served values, past >90% of the selected experts' leads.
    - Its measured cost there: API top-1 agreement 86.1% → 81.5%; at the defaults, code perplexity 2.38 → 2.87.
    - Its box sends RAM hits over PCIe to the GPU, so VRAM is what pays there. Here the CPU computes RAM and cache hits
      at ~0.3 ms each and an SSD read costs ~5-7 ms, so only the SSD penalty is used.

**The 207 GB pack, measured (2026-10-18, 20:32-20:51; the engram on; two GPUs split at 20).**
- **Built** in 78 min. Q3_K layers took ~60 s each; Q2_K layers ~165-195 s each, because ggml's Q2_K imatrix search is
  ~3x the cost of Q3_K's.
- **Quality against native** (wikitext-2 test, 16 chunks of 512, 4,080 scored tokens; the native rows written by the same
  engine):

| | the pack | DSV4 Q3_K pass mark |
|---|---|---|
| mean KLD | 0.143 ± 0.007 | <= 0.134 |
| same top token | 91.35% | >= 88.0% |
| perplexity ratio | 1.089 | 1.101 |
| median KLD | 0.0018 | - |
| 99th percentile KLD | 2.22 | 1.90 |

  - Better than the mark on same top and perplexity; 0.009 over it on mean KLD (~1.2 standard errors), from a heavy
    tail.
  - The DSV4 mark compares two llama.cpp runs with the same kernels (a floor of ~0). This engine's two runs differ by
    where each expert ran (VRAM: GPU kernels; RAM or SSD: CPU kernels), and this model flips routing near-ties under
    tiny roundings. **The floor between two native runs** (the base with a 30 GiB cache and 55 GiB headroom, the second run
    with the pack's 60 GiB cache): KLD 0.0100 ± 0.0010, same top 98.1%, perplexity x1.015. So the pack's own KLD is
    ~0.133, at the mark. Q3_K on layers 33-39 as well (the most sensitive of the Q2_K ones: up to 2.1% at ~2 bpw per
    Xolotl; +9.7 GB, ~2% slower by the model) is the planned margin.
- **Decode** (the same sequence as the native cache runs: a prose warm-up, then the coding text 3 times; 60 GiB cache):

| run | tok/s | SSD reads/token | the speed model (native's constants, the pack's own trace) |
|---|---|---|---|
| warm-up, prose, cold (DSpark w2) | 4.52 | 32.6 | - |
| coding text, first time, plain | **7.14** | 16.0 | 7.51 |
| coding text again, DSpark w2 | 11.89 | 6.5 | - |
| coding text a third time, plain | 12.89 | 5.8 | 11.99 |

  - 1.9x native on a fresh text (3.78).
  - The speed model was calibrated on native only and not refit: 5% high on the fresh text, 7% low on the warmed one.
- **Prefill** (prefixes of the 7,820-token prompt; the default tiers, bounded replay on):

| prompt | native | the pack | SSD reads native → pack |
|---|---|---|---|
| 32 tokens | 15.9 s | **10.6 s** | 1,922 → 1,713 |
| 200 tokens | 39.4 s | **25.9 s** | 4,836 → 4,217 |
| 2,000 tokens | 80.2 s | **53.2 s** | 9,890 → 8,626 |
| 7,820 tokens | 96.1 s | **63.9 s** | 11,158 → 10,022 |

  - ~1.5x native.
  - The drive averages ~2.0-2.1 GB/s here (~75% of its ceiling): smaller records carry more per-record overhead, and
    on long prompts the drive idles in each layer's attention once the 32 bounce slots are full.
- **The opt-in route bias on the pack, margin 0.1** (KLD vs native, wikitext 16x512; decode, the same sequence as the exact
  pack):

| | exact routing | bias 0.1 |
|---|---|---|
| mean KLD / same top / perplexity ratio | 0.143 / 91.35% / 1.089 | 0.146 / 91.20% / 1.083 |
| prose warm-up, cold (DSpark w2) | 4.52 tok/s | **6.09** (+35%) |
| coding text, fresh, plain | 7.14 | **8.54** (+20%), SSD reads 16.0 → 11.5 a token |
| coding text, 2nd pass, DSpark w2 | 11.89 | **15.43** (+30%) |
| coding text, 3rd pass, plain | 12.89 | **16.79** (+30%), SSD reads 5.8 → 2.5 |

  - The quality differences are within the engine's own run-to-run floor (KLD 0.010). The KLD run itself took 418 s
    instead of 531 s.
- **The pack, layers 33-39 to Q3_K too** (21:57; Q3_K on 0-19 + 32-39, Q2_K on 20-31; 217.0 GB). Quality against native
  (the same yardstick):

| | the 207 GB pack | **the 217 GB pack** | DSV4 Q3_K pass mark |
|---|---|---|---|
| mean KLD | 0.143 ± 0.007 | **0.131 ± 0.007** | <= 0.134 |
| same top token | 91.35% | **92.38%** | >= 88.0% |
| perplexity ratio | 1.089 | 1.090 | 1.101 |
| median KLD / 99th percentile | 0.0018 / 2.22 | 0.0011 / 2.15 | - / 1.90 |

  - It passes on all three; the engine's own floor (0.010) is inside the 0.131.
- **The DSpark drafter's cost cut** (22:13; `STRATA_DSV41_DSPARK_ROWS=-1` and the Markov head on the device):
  - The Markov head ran gemv_f32_w's block per row (129,280 blocks of 256-element dots) and synced the host once per draft.
    It is now one kernel a step (a warp per vocabulary row, the previous token read where the argmax wrote it) and one sync
    a block.
  - A window of 3 verifies 2 drafts, so only those are drafted: DSpark's block attends across its rows, so the drafts can
    differ; the trunk verifies them, so the output does not.
  - **18.1 → 8.1 ms a window**: its experts on the CPU 8.8 → 4.6 ms, the Markov head 6.7 → 1.5 ms. Acceptance at w3 is
    unchanged (76-78% of the drafts, 2.09-2.11 tokens a window).
  - (The first run of it failed at once: V4.1's drafter has its own loader, `load_dspark_v41`, and the new buffers were
    only allocated in V4's.)
- **Decode on the 217 GB pack, exact routing** (w1, w3, w3, w1 after the prose warm-up; 60 GiB cache):

| run | tok/s | SSD reads/token | the 207 GB pack |
|---|---|---|---|
| prose warm-up, cold (w3) | 4.49 | 33.1 | 4.52 (w2) |
| coding text, fresh, plain | **6.77** | 16.3 | 7.14 |
| 2nd pass, DSpark w3 | 10.51 | 9.3 | 11.89 (w2) |
| 3rd pass, DSpark w3 | 11.39 | 8.3 | 13.08-13.09 (w3, 3rd-4th pass) |
| 4th pass, plain | **9.92** | 9.5 | 12.68-12.89 (plain, 3rd-6th pass) |

  - Fresh text: 5% slower than the 207 GB pack, as the model says (7.02 predicted, 16.3 reads predicted and measured).
  - **Warmed: 22% slower, and the model does not say so**: it predicts 4.1 reads a token on the 4th pass (4.8 with the
    engine's per-size pools), the engine reads 9.5. On the 207 GB pack the same check was within 15% (6.7 predicted, 5.8
    measured). Being traced: the engine now counts its reads per layer (`STRATA_DSV41_READS_BY_LAYER=1`).
- **Route bias 0.3 on the 217 GB pack, decode and prompt** (the same sequence):

| run | exact routing | **bias 0.3** |
|---|---|---|
| prose warm-up, cold (w3) | 4.49 | **7.02** |
| coding text, fresh, plain | 6.77 | **10.42** |
| 2nd pass, DSpark w3 | 10.51 | **18.48** |
| 3rd pass, DSpark w3 | 11.39 | **21.06** |
| 4th pass, plain | 9.92 | **19.09** |

  - Quality (wikitext 16x512 against native): mean KLD 0.1537 ± 0.0080 against the exact pack's 0.1307 (**+0.023**),
    same top 91.47% (-0.9 points), perplexity ratio 1.079 (1.090 exact). Just over the +0.02 line on KLD.
  - The prompt path's bias only ever saves reads in chunks of at most 1,024 tokens (`stream_gate`): a longer chunk streams
    every non-resident expert of a layer ahead of its routing. Next: the decode-only bias's KLD.
- **Where the GPUs' time goes in a decode window** (the nsys profile of a warmed DSpark w3 run on the 207 GB pack, re-read
  per layer from its sqlite):
  - Between two layers' doorbells (one layer's post-routing, the next one's pre-routing and its VRAM experts): GPU 0
    1,075 µs, GPU 1 954 µs, of which kernels 99% (no launch gaps: the decode graphs work).
  - Pre-routing, per layer (a 2-3 row window): the grouped output projection `heads_q8_0` (wo_a, 8 groups of 1024 x 4096
    Q8_0, 35.7 MB) **153-194 µs at ~200 GB/s** - wo_b, a plain native MMVQ of 44.6 MB, takes 67 µs (~660 GB/s); q_b 75-80
    µs (~460 GB/s); the two mHC mixers (`gemv_f32_w`, 24 rows of 20,480 F32: 24 blocks for 68 SMs) **26 µs each**; the
    CPU rows' combine (read over PCIe) 24-28 µs; the router 17 µs, route 13, doorbell publish 11-14, attention 18-22.
  - The bandwidth floor of a layer's pre-routing weights (~137 MB) is ~200 µs; measured ~530 µs.
  - So heads_q8_0 alone is ~6-8 ms of a token's ~30 ms of GPU time, the mixers ~2 ms. Written, opt-in for the A/B:
    `STRATA_DSV41_HEADS_MMVQ=1` (wo_a as one native MMVQ per group in decode windows) and `STRATA_DSV41_HC_SPLIT=1` (the
    mixers' rows split 16 ways across blocks, the partial sums reduced in a fixed order). DSV4's path is untouched.
- **The two GPU opt-ins measured** (22:33-22:36):
  - The kernels alone (`STRATA_GLM_PARITY_ONLY=heads_v41 glm_kernels_parity`, V4.1's shapes; the results agree to
    1.5e-7): wo_a 125 / 167 / 212 µs at 1 / 2 / 3 rows with `heads_q8_0`, 72.5 / 77.4 / 80.7 µs as one native MMVQ per
    group (286 → 492 GB/s at one row); the mixers 17.6-20.7 µs with `gemv_f32_w`, ~10 µs split (two launches).
  - In decode (route bias 0.3 on both paths, the same sequence as run E): DSpark w3 windows 115.4 → 109.5 ms and 101.3 →
    93.8 ms; plain warmed 19.09 → **20.46 tok/s**, and the time outside the pool 30.1 → 27.3 ms a token - the 2.8 ms the
    kernels' times add up to. (Tokens per second at w3 moved less, 18.48 → 19.28 and 21.06 → 20.47, because the drafts
    were accepted less often in that run: 1.92 vs 2.13 tokens a window. The output depends on the cache's state once the
    bias is on.)
- **Prefill with the prompt path's bias 0.3 and `--short-read 96`** (217 GB pack; 207 GB exact for reference):

| prompt | the 207 GB pack, exact | **217 GB, prompt bias 0.3, short-read 96** | SSD reads |
|---|---|---|---|
| 32 tokens | 10.6 s | **9.0 s** (decode windows) | 1,713 → 19 (the windows' reads are the decode's) |
| 200 tokens | 25.9 s | **22.9 s** | 4,217 → 3,226 |
| 2,000 tokens | 53.2 s | 56.1 s | 8,626 → 8,639 |
| 7,820 tokens | 63.9 s | 67.0 s | 10,022 → 10,035 |

  - Past 1,024 tokens the bias cannot help (every non-resident expert streams ahead of the routing); there the 217 GB
    pack's larger records cost ~5%. The engram rows take 0.2 / 1.2 / 3.2 s of the 200 / 2,000 / 7,820-token reads.
  - These runs start with an empty SSD cache. A server's cache is full (decode fills it), and the prompt path takes every
    cached record instead of reading it: `--warm-file` now also works with `--runs` (one generation first) to measure that.
- **Quality of the candidate configs** (wikitext 16x512 against native; the exact 217 GB pack: KLD 0.1307, same top 92.38%):

| config | mean KLD | vs exact | same top | vs exact | perplexity ratio |
|---|---|---|---|---|---|
| route bias 0.3, decode and prompt | 0.1537 | +0.023 | 91.47% | -0.9 | 1.079 |
| **route bias 0.3, decode only** + the GPU opt-ins (§12: this run never applied the bias - re-measured there) | **0.1423** | **+0.012** | **91.67%** | **-0.7** | 1.095 |
| the same + dense Q6_K at load | 0.1467 | +0.016 | 91.35% | -1.03 | 1.100 |

  - Decode-only bias 0.3 is inside the threshold (+0.02 / -1 point): it is the final decode bias. The dense Q6_K requant
    (`STRATA_DSV41_DENSE_Q6K=1`: 348 matrices, 1.81 GiB off the GPUs, +127 expert slots; fresh plain 10.43 → 11.01 tok/s)
    lands just past the same-top line, so it stays opt-in.
- **The DSpark window in the final config** (decode-only bias 0.3, the GPU opt-ins; w1 w2 w3 w2 w1 after the prose warm-up):

| pass | window | tok/s | tokens a window | pool / wait (ms a token) |
|---|---|---|---|---|
| fresh | plain | 10.41 | 1 | 70.2 / 16.8 |
| 2nd | w2 | 17.06 | 1.61 | 36.9 / 11.4 |
| 3rd | w3 | 20.84 | 1.98 | 28.4 / 10.2 |
| 4th | w2 | 20.81 | 1.59 | 25.9 / 12.8 |
| 5th | plain | **21.02** | 1 | 21.1 / 19.2 |

  - Warmed, plain, w2 and w3 are the same speed: a verify window's extra rows cost the CPU pool as much as the drafts save.
    The drafter pins 6.7 GiB of RAM; off, the RAM copy takes it (the 28 GiB headroom and the 60 GiB cache unchanged). The
    final config decodes plain; DSpark stays available.
  - A warmed plain token: 47.6 ms = the host waiting for the GPUs' routing 19.2 + the CPU pool 21.1 + the rest 7.3.
- **Prefill as a server sees it** (K: the cache filled by a 77 + 384-token decode first; the final decode config; the
  RAM: a 15.8 GiB copy + the 60 GiB cache, against run C's 76.9 GiB copy and a 64-slot cache):

| prompt | C (big RAM copy, empty cache) | **K (warmed cache)** | K: from the SSD / from the cache |
|---|---|---|---|
| 32 tokens | 9.0 s | **2.8 s** | 0 / 0 (decode windows, all in RAM) |
| 200 tokens | 22.9 s | **18.4 s** | 3,115 / 3,029 |
| 2,000 tokens | 56.1 s | **47.1 s** | 8,767 / 4,189 |
| 7,820 tokens | 67.0 s | **55.8 s** | 9,784 / 4,362 |

  - The drive is ~96% busy in K's long reads: prefill there is its reads x ~5.2 ms.
  - Two wastes found in those reads: (1) a chunk over 1,024 tokens streams every non-resident expert before its routing,
    and a 2,000-token chunk uses only ~11,200 of the ~13,800 it streams (~1,600 SSD reads, ~8 s, for nothing); (2) the
    VRAM slots a long prompt borrows for its buffers (258 at 2K, 956 at 8K) are refilled afterwards with one O_DIRECT read
    each - records the prompt had just streamed and thrown away.
  - Being measured: the stream gate for long chunks (`STRATA_DSV41_STREAM_GATE=8192`) and the prompt's reads kept in the SSD
    cache (`STRATA_DSV41_PROMPT_KEEP=1`, new: the stream reads into cache slots - up to a share of them a prompt, the lent
    experts always - so the refill, the decode after the prompt and the next prompts find them in RAM).
- **The stream gate on every chunk** (L1: `STRATA_DSV41_STREAM_GATE=8192`, K's warmed cache and sequence):

| prompt | ungated (K) | **gated** | SSD reads | the lent slots' refill: from the cache / read |
|---|---|---|---|---|
| 2,000 tokens | 47.1 s | **38.5 s** (-18%) | 8,767 → 6,590 | 93 / 167 |
| 7,820 tokens | 55.8 s | **52.2 s** (-6.5%) | 9,784 → 8,143 | 333 / 599 |

  - The gate costs each layer a wait for its routing before its reads start; at 2K tokens the reads it saves win by far, at
    8K by less. Since then every chunk is gated by default on V4.1 (`stream_gate`; `STRATA_DSV41_STREAM_GATE=N` sets a
    limit, 0 the old ahead-of-routing stream).
  - The refill still reads 167 / 599 records from the SSD after the 2K / 8K prompts (~0.9 / ~3.1 s): the prompt-keep change
    is to serve those from RAM.
- **An exactness gate that was too strict** (23:08-23:14, KG): the 7,820-token prompt and 64 greedy tokens with the prompt-keep
  change on and off gave the same first 40 tokens, then split at a near-tie (`2619 27297` against `24985`: the same text cut
  into tokens two ways). No slot check failed. The two runs' RAM copies differed (20.6 vs 23.3 GiB: the copy is sized by the
  MemAvailable at load), so the VRAM tier evolved a little differently (285 vs 291 swaps) and a few experts ran on the GPU in
  one run and on the CPU in the other - two kernels, two roundings. A decode's greedy tokens are therefore no exactness test
  for a change that moves records between tiers; the prompt path is (every prompt expert runs on the GPU): its last logits
  must be bitwise equal (KG3).
- **The final decode config, plain (no drafter: the RAM copy 22.7 GiB), and the VRAM tier's fills made asynchronous**
  (M1 / M2: the prose warm-up, then the coding text 3 times; 60 GiB cache):

| pass | M1: fills synchronous | M2: fills asynchronous | M1 per token: pool / GPU wait / rest | M2 |
|---|---|---|---|---|
| prose warm-up, cold | 7.24 | 7.00 | | |
| coding text, fresh | **10.69** | 10.56 | 67.0 / 16.9 / 9.7 ms | 66.3 / 17.4 / 11.0 |
| 2nd pass | 15.73 | **17.58** | 36.0 / 17.9 / 9.7 | 30.7 / 19.2 / 7.0 |
| 3rd pass | **19.67** | 19.47 | 23.1 / 18.8 / 8.9 | 27.4 / 19.4 / 4.6 |

  - The CPU pool runs at 58-59 GB/s over its drain with no barrier time (park, repark 0.00 ms): bandwidth-bound.
  - Asynchronous fills took 4.3 ms a token off the host's time between windows (the rest: 8.9 → 4.6 ms) but lifted the
    synchronous fills' cap of 8 SSD-tier swaps a round: the tier swapped ~4 experts a token instead of ~2.4, and the victims
    it dropped were read again (SSD reads 0.9 → 2.2 a token, its wait 4 → 11 ms). Under the old cap: M3.
- **The prompt path's three changes pass their exactness gate** (KG3, 23:21-23:26): the 7,820-token prompt's top-8 logits with
  prompt keep + the lent experts copied to the SSD cache before lending + the engram rows fetched beside the stream are bitwise
  those with all three off (`5:30.562 372:26.876 671:25.414 ...`). The same cold run (no warm-up, the cache empty):

| | off | **on** |
|---|---|---|
| 7,820 tokens | 72.5 s | **64.4 s** (-11%) |
| SSD reads | 12,121 | 10,447 |
| the 956 lent slots' refill | 956 read from the SSD | **956 from the cache** |
| records the prompt left in the cache | 0 | 3,550 (+ the 956 lent) |
- **Keeping every prompt read in the cache hurt a warmed cache** (L3, 23:26: K's sequence with `PROMPT_KEEP` at a share of 1):

| prompt | gated, no keep (L1 / K) | **keep, share 1 (L3)** | L3 SSD reads / cache hits |
|---|---|---|---|
| 200 tokens | 18.4 s (K) | 27.0 s | 4,909 / 1,153 (K: 3,115 / 3,029) |
| 2,000 tokens | 38.5 s | 51.2 s | 9,352 / 894 (L1: 6,590 / 3,771) |
| 7,820 tokens | 52.2 s | 64.3 s | 10,773 / 939 (L1: 8,143 / 4,097) |
| 200 again | - | 26.2 s | 4,133 / 1,936 |
| 2,000 again | - | 53.7 s | 9,903 / 345 |

  - The prompt's insertions took LRU victims - records the same prompt needed a few layers on - and every later prompt found
    less: the cache stopped holding anything useful (a scan through an LRU). The lent experts' 934-956 refills did come from
    the cache, as in KG3.
  - Since then `STRATA_DSV41_PROMPT_KEEP=1` keeps only the lent experts (their copies out of VRAM, for the prompt and its
    refill); `STRATA_DSV41_PROMPT_KEEP_SHARE=F` adds the prompt's other reads, off by default. Measuring (L4).
- **Asynchronous fills under the old cap (M3)** - the final decode config's numbers since:

| pass | M1: synchronous | **M3: asynchronous, 8 SSD-tier swaps a round** | M3 per token: pool / GPU wait / rest |
|---|---|---|---|
| prose warm-up, cold | 7.24 | 6.85 | |
| coding text, fresh | 10.69 | **11.06** | 66.6 / 16.9 / 6.9 ms |
| 2nd pass | 15.73 | **19.00** | 29.4 / 18.1 / 5.1 |
| 3rd pass | 19.67 | **20.54** | 25.3 / 19.0 / 4.4 |

  - On by default since (`STRATA_DSV41_ASYNC_FILL=0`: the synchronous fills). A warmed token: 48.7 ms = the CPU pool 25.3 (its
    drain 17.5 at 58 GB/s, the SSD 7) + the GPUs' routing 19.0 + the host's rest 4.4.
- **Prefill as a server sees it, with the prompt path's changes** (L4, 23:36: K's warm-up and sequence; the stream gate on every
  chunk, the lent experts kept in the SSD cache, the engram rows beside the stream):

| prompt | K (before) | L1 (the gate) | **L4 (all three)** | L4 SSD reads / cache hits | the lent slots' refill |
|---|---|---|---|---|---|
| 200 tokens | 18.4 s | - | **18.9 s** | 3,250 / 2,830 | 124 from the cache |
| 2,000 tokens | 47.1 s | 38.5 s | **36.1 s** | 6,388 / 3,900 | 251 from the cache, 0 read |
| 7,820 tokens | 55.8 s | 52.2 s | **47.5 s** | 7,538 / 4,202 | 940 from the cache, 0 read |
| 200 again | 18.0 s | - | 19.8 s | 3,586 / 2,495 | |
| 2,000 again | - | - | 37.7 s | 6,805 / 3,482 | |

  - Long prompts: -23% (2K) and -15% (8K) against K. A 200-token prompt was gated already: unchanged.
  - What is left is the drive: 6,388 reads x ~5.2 ms = 33 s of the 2K prompt's 36 s; 7,538 = 39 s of the 8K's 47.5 s.
- **The server, end to end** (S8, 23:41; `strata-dsv41.json`: the final config - route bias 0.3 in decode, the GPU opt-ins,
  plain decode with the engine's prompt-lookup drafts, the stream gate, the lent experts kept, the engram beside the stream;
  a 128K context then - 262,144 in every config since - and a learned expert profile):

| request | prompt (reused / read) | read in | decode | the answer |
|---|---|---|---|---|
| turn 1 | 0 / 33 | 3.4 s | 71 tokens, 8.5 tok/s | a correct iterative Fibonacci |
| turn 2 | 103 / 17 | **0.96 s** | 88 tokens, **20.0 tok/s** (54 of 56 lookup drafts kept) | the same with the check asked for |
| a tool call | 0 / 291 | 23.2 s | 55 tokens, 11.0 tok/s | `get_weather {"city":"Paris","unit":"c"}` |
| after the tool's result | 345 / 26 | **1.5 s** | 16 tokens, 11.1 tok/s | "It's currently 14°C in Paris, with cloudy skies." |
| a 1,795-token file to summarize | 0 / 1,795 | 38.5 s (46.6 tok/s) | 150 tokens, 11.4 tok/s | a correct 3-point summary |
| its follow-up | 1,944 / 16 | **0.57 s** | 120 tokens, **17.9 tok/s** | correct |

  - A follow-up turn reads only its new tokens: under 1.5 s here. New text costs what the drive reads for it: a 291-token
    prompt on a cold topic 23 s, 1,795 tokens 38.5 s.
  - With no DSpark loaded the engine drafts from the prompt (n-gram lookup): an edit of code it just wrote decoded at 20 tok/s.
- **Theory against practice, the final config** (goal 3; the floors from this box's measured rates: the GPUs ~700 GB/s of
  their 760, the DRAM ~73 GB/s, the drive 2.7 GB/s - one record, 14.1 MB on average, 5.2 ms):

  Decode, a warmed token (M3's 3rd pass, 20.54 tok/s = 48.7 ms):

| part | what bounds it | floor | measured | of the floor |
|---|---|---|---|---|
| the GPUs' routing (each layer's pre-routing work, the two GPUs in turn) | 7.15 GB of dense weights a token | 10.2 ms | 19.0 ms | 54% |
| the CPU pool's experts | 1.02 GB of records | 14.0 ms | 17.5 ms (58 GB/s) | 80% |
| the SSD | 1.5 records | 7.8 ms | 7 ms waited | ~100% (partly overlapped) |
| the host (dispatch, sampling, the tier's upkeep) | - | ~1-2 ms | 4.4 ms | - |
| a token | the parts in turn | ~33 ms (30 tok/s) | 48.7 ms (20.5 tok/s) | 68% |

  Prefill, a warmed cache (L4): the drive's time for the records read is the floor (the GPUs' work hides behind it).

| prompt | records read | floor | measured | of the floor |
|---|---|---|---|---|
| 200 tokens | 3,250 | 17.0 s | 18.9 s | 90% |
| 2,000 tokens | 6,388 | 33.4 s | 36.1 s | 93% |
| 7,820 tokens | 7,538 | 39.4 s | 47.5 s | 83% |

  - Prefill runs at 83-93% of what the drive allows for the records it reads; the GPU's dense kernels are the largest gap
    in decode (54% of their bandwidth floor: the per-layer kernel list in the nsys section above).
  - Where theory and practice still part: the tier simulator predicts 4.1-4.8 SSD reads a token for the exact-routing 217 GB
    pack warmed, the engine reads 9.5 (#21; the run with `STRATA_DSV41_READS_BY_LAYER=1` against `sim_layers.py` was done there).
- **Prefill against the drive, the server's shape** (2026-10-19, after the TRIM; the server's flags through the serve
  protocol; each request's prompt-path line: the drive's bytes from the kernel's disk counters, and the time with no read
  of a stage in flight by what it waited for). The drive's ceiling with the engine's reader pool (`expert_io_bench`, 12
  readers, 4 MiB pieces, Q3_K records): 2.58-2.76 GB/s, 5.6-5.9 ms a record. The kernel's busy counter reads 84-92% while
  the drive is saturated, so its busy share understates the drive's work (it read 70-80% in the rows below).

| request | records read | GB the drive read | floor at 2.73 GB/s | prompt path | of the floor | no read in flight: routing / refill / lend |
|---|---|---|---|---|---|---|
| 4,857 tokens, fresh | 9,992 | 141.8 | 51.9 s | 61.8 s | 84% | 3.6 / 1.2 / 1.1 s |
| 9,918 tokens on 4,873 | 9,802 | 139.7 | 51.2 s | 58.6 s | 87% | 5.9 / 2.4 / 1.1 s |
| session turn 2, 4,856 tokens | 8,167 | 116.6 | 42.7 s | 50.0 s | 85% | - |
| session turn 3, 9,918 tokens | 7,873 | 112.2 | 41.1 s | 51.2 s | 80% | - |

  - The gap to the floor is mostly time the drive waits for each layer's routing, then the refill and the lend: the
    first is #31 (written, opt-in, one A/B to go), the second #32.
- **The speed model against three runs it was not fitted to** (2026-10-19, #21 closed): the model's constants are still the
  native calibration (the GPUs 33.5 ms a token, the pool 58 GB/s, a pool call 0.05 ms, the drive 2.76 GB/s); what changed are its
  inputs, now the engine's own: each run's logged VRAM slots per GPU and RAM copy (`... GiB in RAM (pinned)`), the RAM copy filled
  by bytes in layer order, the SSD cache's per-size pools, and the swap rules (17 a round, at most 8 bringing in an expert with no
  RAM copy). The earlier 4.1-4.8 reads a token against the engine's 9.5 came from the inputs: the native pack's 18.17 GiB of RAM
  copy instead of the run's 15.37, and swaps without the SSD-tier cap - not from the engine.

| run (its trace, its tiers) | pass | SSD reads: model / engine | tok/s: model / engine |
|---|---|---|---|
| 217 GB pack, exact routing (D) | fresh | 17.0 / 16.3 | **6.75 / 6.77** |
| | warmed | 9.1 / 9.5 | **9.95 / 9.92** |
| 207 GB pack, exact routing | fresh | 16.5 / 16.0 | **7.15 / 7.14** |
| | warmed | 5.7 / 5.8 | **12.42 / 12.89** |
| 217 GB pack, route bias 0.3, GPU opt-ins (F; the GPUs' 33.5 ms less the opt-ins' 2.8 measured on their own) | fresh | 7.60 / 7.4 | **10.21 / 10.43** |
| | warmed | 1.15 / 1.1 | **18.63 / 20.46** |

  - Every routed expert's tier (VRAM / RAM / cache / read) now agrees with the engine's count within a few percent, and the
    speed within 0.1-4% with exact routing. The fast warmed pass with the bias is 9% low: there the GPUs' and host's part is
    27.3 ms measured against the 30.7 the model carries. (With the bias the model replays the routing the engine chose; it does
    not choose routes itself.)

## 12. Reliable, complete, fast (from 2026-10-19)

**The goal (the user, 2026-10-19), in this order:** (1) measure, then set the route bias and the sampling; (2) close the audit's
gaps (DSV41_BACKLOG #33-#36) and verify every fix; (3) vision end to end; (4) the speed this box allows within those decisions,
with theory against practice. Reliability first; a test only when it changes a decision.

**The route bias through decode windows** (the 217 GB pack; `--kld-mode windows`: the context AND the scored tokens through decode
windows, so the bias applies to both - harsher than a session, where only the generated tokens are routed with it; wikitext
16x512, 4,080 scored tokens, against the native base; the GPU opt-ins on; the 60 GiB cache):

| margin | mean KLD | vs exact | same top | perplexity ratio | decode tok/s: fresh / 2nd / 3rd pass |
|---|---|---|---|---|---|
| exact | 0.1341 ± 0.0070 | - | 91.59% | 1.080 | 6.97 / 10.44 / 12.04 |
| 0.02 | 0.1355 ± 0.0070 | +0.0013 | 91.47% | 1.082 | 8.23 / 11.89 / 15.49 |
| 0.05 | 0.1349 ± 0.0071 | +0.0008 | 91.81% | 1.089 | 8.51 / 15.08 / 16.97 |
| 0.1 | 0.1373 ± 0.0073 | +0.0032 | 91.72% | 1.088 | 9.81 / 13.81 / 16.68 |
| **0.3** | **0.1395 ± 0.0072** | **+0.0054** | **91.57%** | 1.087 | **11.07 / 17.55 / 21.00** |

- Every margin is inside the line (+0.02 KLD, -1 point same top) and inside the engine's own run-to-run floor (0.010). 0.3 is
  the fastest: +59% fresh, +74% on the third pass (decode: plain, 192 tokens, a prose warm-up then the coding prompt three times).
- The earlier "+0.012 at 0.3, decode only" (§11) never applied the bias: that KLD read its contexts through the prompt path,
  whose routing takes `prompt_pen_for` - null without `STRATA_DSV41_ROUTE_BIAS_PROMPT`. The numbers above do apply it.
- **Decode only, now true.** A prompt's tokens read through windows (a follow-up of up to 96 tokens, the tail after the last
  checkpoint, the last token) used to route with the bias: 54% of the 174K-token context of the 2026-10-19 loop was written
  that way. `run_request` now holds the table at zeros while a prompt is read, the first decode window included. Checked: the
  routing trace of a 60-token prompt read through windows is the same, route for route (2,400 of 2,400), with the bias at 0.3
  and off; the only differences are the two windows `warm_decode` runs at the engine's start, before any request.

**The release's sampling.** DeepSeek-V4.1's card: temperature 1.0, top_p 1.0, no top-k cut; its `generate.py` samples
softmax(logits / T) over the whole vocabulary (Gumbel-max).
- The engine sampled at most 64 candidates, and a GEN line without top_k takes the engine's default, 20 - which V4.1's requests
  did all night on 2026-10-18, after the config's top_k was taken out. top_k 0 now means no cut, and with top_p 1 and min_p 0
  the engine samples the whole vocabulary: Gumbel-max keyed by (seed, position, token id), so a verify window's rows draw what
  single steps would given the same logits (a draft's draws are a single step's; its logits are not quite, #36c below). The server sends top_k 0 to the deepseek41 family whenever
  a request names no positive top_k (absent, 0, or a client's -1), so V4.1 never reaches the default; the default itself stays
  20 (the user, 2026-10-19), so every other family samples as before.
- `sampler_parity` (all three sampled paths): the draws' frequencies are softmax(logits / T) (total variation 0.0145 over 204,800
  draws; 54% of that fixture's mass lies past the 64 a list keeps), -inf is never drawn, windows equal single steps, and on 64
  rows at V4.1's width (129,280) every pick equals a host Gumbel-max's, with the split kernels and with the one-block one.
- One block a row walked 129,280 entries with two double-precision logs each on one SM: 2,956 µs a token (`--bench`), 6% of a
  warmed decode token. Split over 32 blocks a row and merged (the same values, the same tie rule: the same token): **101 µs**
  (198 µs for 4 rows, 391 µs for 8).
- top_p and min_p act on the tempered distribution for V4.1 (`temp_first`, the order of HF's and vLLM's samplers, which the
  release names for serving); at T 1.0, or top_p 1 and min_p 0, both orders draw the same. The parity fixture at T 0.5 sees the
  order (142 and 131 of 256 rows draw differently with the flag off) and the kernels match the serial reference.
- `strata-dsv41.json`: temperature 1.0, top_p 1.0, no top_k (the server's 0 for this family), reasoning effort "max" (the server now accepts it;
  for this family a client's "high" is the release's 75 and "max" its 100).

**The serving gaps (the audit's #35), against the release's encoder.** Clients other than dsh met a server that rewrote their
messages before the template saw them. For deepseek41 the server now reads a client's messages as the release's
`encoding.py` does (`serve/frontend.py` Dialect; every other family's conversion is unchanged, byte for byte):
- tool results are put in the order of their calls by id (the server dropped the ids: OpenAI's `tool_call_id`, Anthropic's
  `tool_use_id`);
- a system or developer message after the first stays one (`<｜System｜>`), not a user turn;
- a message's text parts are joined by a blank line, not glued together;
- the effort words are the release's: low 50, high 75, max 100 (a client's "high" rendered 100);
- a past turn of reasoning alone is kept (the renderer dropped it as empty, #843) - in thinking mode without tools the encoder
  drops a past turn's reasoning, so such a turn renders as an empty one, as the release's encoder renders it;
- the parser gives the answer as written - its newlines after `</think>` kept, of those before a call only the blank line of
  `"\n\n<｜DSML｜ calls"` dropped - so the next turn's history is the model's own text;
- text that is JSON of JSON (arguments encoded twice) is read twice, for every family. Empty arguments stay a call with no
  parameters (the release would write a parameter "arguments" holding "", which the model never wrote: its parser gives "{}").
- Checked end to end: request bodies (OpenAI, Anthropic, Responses) through the server's path render byte for byte as the
  release's encoder renders the same bodies (`serve/test_deepseek41.py`, 35 tests); the whole serve suite passes (762 tests).

**The latent ones.** Two CPU gate/up branches skipped the SwiGLU limit (10): the native Q2_0 rows in the pool and the AVX-only
i-quant kernel (`iq128_gu_rows`); neither runs on this box with this pack, both clamp now (`iq_avx1_parity` checks the clamped
rows, and that the limit moves at least an eighth of them). A record file must hold exactly 384 records of its stride (the size
check was `<`: a stale `experts.tsv` line - q3q2 lists layers 33-39 twice, the last line counts - would have read every expert
past the first from the wrong place); every file of both packs is exact.

**The code/agent text, and what it measured (DSV41_BACKLOG #38).** The first code/agent text (2 x 4,096 tokens of a synthetic
coding-agent conversation, this repo's source as `read_file` results) gave KLD 0.964 against the native base, against 0.131 on
wikitext - and the release's own model.py, on the release's dense safetensors and the bitwise native experts, failed it too: a
listing line's opening ("  NNN\t") at mean logp -5.38 / -8.12 / -11.61 over positions 564-1,199 / 1,200-1,999 / 2,000-2,991,
certain of code indentation where every earlier line has a number. What it was not:
- the input: the token ids are tokenizer.json's own; the prompt is the release's encoding (and SGLang's `encoding_dsv41.py`,
  DeepSeek's encoding fixtures 1, 2 and 5, vLLM's V4.1 renderer fixtures 1 and 2, all byte for byte through the server);
- the selection: model.py with every row selected (`ref_run.py --index-topk 4096`, positions 564-1,198) moves the line openings
  from -5.10 to -4.60 and the perplexity from 22.07 to 18.97 - a cost, not the cause; SGLang and vLLM compute the indexer, the
  compressor and the attention exactly as model.py does (vLLM's defaults differ only in number formats: an FP8 indexer and FP8
  compressed KV where the release, SGLang and this engine use FP4);
- the engine: by role, model.py predicts the assistant's tokens at NLL 1.13 (PPL 3.1) and the system prompt and the tool output
  at 3.58 and 3.85 (PPL 36 and 47) - the model predicts its own turns, not what it is sent, as chat training that scores the
  assistant would leave it. Against model.py the engine (native pack) is at KLD 0.0156, same top 99.1%, on the assistant's
  112 positions and 0.219 / 82.2% on the tool output's 337: the gap of 0.597 over 1,496-2,990 was the untrained region's alone,
  where a flat distribution turns roundings into different tokens. Its engram is the reference's: without it the engine's
  layer-1 residual parts from model.py's by 24% instead of 6%, and the KLD over 1,496-2,990 goes from 0.597 to 1.945.
So a KLD scored on tool output measures nothing a session uses, and the misread listing says nothing about how the model reads
one. The code/agent text is scored on the model's own answers instead (the native pack writes them at the release's sampling).

**Windows against single steps (#36c).** The same wikitext rows at decode windows of 8 and of 1 (the tier frozen, the slots
fixed) are not the same bytes: KLD 0.0157 between them over 3 chunks (765 rows), 20 top-1 flips, with the GPU opt-ins on.
It is the engine's, not the opt-ins': without them chunk 0 parts by KLD 0.0114 (96.9% same top), with them 0.0237 (96.5%) -
the kernels round a row differently in a window of 8 than alone, and in a model whose routing near-ties sit within 1e-5 that
moves ~3% of greedy picks. It costs no quality: against the native base the windows-of-8 rows are at KLD 0.1281 (91.37% same
top) and the single steps at 0.1335 (92.16%), inside the engine's run-to-run floor (0.010) - so a KLD through decode windows
measures what decode gives. What it does change: a drafted window can draw another token than single steps would, so drafts
change the speed and, rarely, the text. The opt-ins stay on (+7%).

**The route bias, decided: 0.3.** The code/agent text scored on the model's own answers (two agent prompts - system, tools, a
request, one `read_file` and its result - with the answer turn rendered with no tools offered: with `read_file` on offer the
model reads on, 94-token turns that are each another call; the answers written by the 217 GB pack at exact routing and the
release's sampling, 2,417 and 2,806 tokens; chunk 0, 2,047 scored tokens, windows mode, against its native base):

| margin | mean KLD | vs exact | same top | vs exact | the run |
|---|---|---|---|---|---|
| exact | 0.0310 ± 0.0018 | - | 94.43% | - | 580 s |
| **0.3** | **0.0393 ± 0.0021** | **+0.0083** | **93.70%** | **-0.73** | 280 s |

- The three marks, on both texts: KLD +0.0054 (wikitext) and +0.0083 (code/agent) against +0.02; same top -0.02 and -0.73
  points against -1 (on 2,047 tokens its standard error is ~0.4 points; with wikitext's 4,080, -0.26); the long-generation
  loop test (the 44K report prompt, temperature 1.0, top_k 0, 8,000 tokens each): seed 1 and seed 2 at repeat coverage
  0.000 (4,749 and 4,564 words, all still reasoning at 8,000 tokens). 0.3 is the fastest margin measured: decode 11.07 /
  17.55 / 21.00 tok/s against exact's 6.97 / 10.44 / 12.04, and 21-25 tok/s past the first 512 tokens on the 44K context.
- The code/agent text's absolute numbers favour the pack a little (it wrote the text at exact routing); the margin is judged
  against exact on the same text.

**Vision, end to end.** The release's model.py on a generated picture ("7319", 640x480 -> a 12 x 16 grid, 206 rows at
position 2 of 225 tokens; the native pack, the CPU) against the engine's image path: the top token equal at 15 of the 17
positions after the picture, and the answer's first token ("731", logp -0.000 in both); the two that differ are in the user's
question (219, 222 - text the model does not predict, #38). Through the API with the production binary and
`strata-dsv41.json`: the number -> "7319" (27.9 s, 225 prompt tokens), two shapes -> "A blue circle on the left and a yellow
square on the right." (27.9 s), a text request -> "42" (7.3 s). dsh (web and headless): deepseek-v4.1-flash takes text and
image.

**Prefill against the drive, a cold 44K read** (the 44,095-token report prompt through the serve protocol, 60 GiB, the
engram tables evicted from the page cache first, the same first token in every arm; the floor: the drive's bytes at 2.73
GB/s):

| arm | records from the drive / the cache | GB the drive read | floor | prompt path | of the floor | engram waits | no read in flight: routing |
|---|---|---|---|---|---|---|---|
| (a) the mapped engram reads (the old default) | 51,678 / 5,685 | 751.5 | 275.3 s | 430.9 s | 64% | 111.3 s | 56.3 s |
| (b) + O_DIRECT engram reads (#39) | 51,700 / 5,685 | 762.5 | 279.3 s | 594.9 s | 47% | 253.0 s | 120.6 s |
| (c) + GATE_AHEAD 2048 (#31) | 52,559 / 5,683 | 763.7 | 279.7 s | 443.5 s | 63% | 46.7 s | 27.7 s |
| **(d) + PROMPT_KEEP_LONG=1 (#29, the config now)** | **37,431 / 19,954** | **545.7** | **199.9 s** | **240.7 s** | **83%** | 35.5 s | 75.6 s |

- Every chunk routes to nearly every expert, so a long prompt read the pack once a chunk; keeping a long prompt's reads for
  its later chunks (victims from the replayed layers 21-39 first) cuts the drive's bytes by 27% and the time by 44%, and the
  drive runs at 2.24 GB/s instead of 1.73. GATE_AHEAD halves the routing gaps but reads 934 records ahead for nothing; the
  O_DIRECT engram reads queue 4 KB requests behind the experts'. The lend and the refill are 4.4 s of the 430.9 (#32).
- The cold 110K read (the deep probe, before PROMPT_KEEP_LONG): 109,998 tokens in 760.3 s, 144.7 T/s.

**The SSD cache's size** (`tier_sim.py` on a recorded 217 GB-pack decode trace, the engine's tiers: VRAM 979 + 841 slots,
the RAM copy 1,465 records, the profile): SSD reads a decoded token at 30 / 45 / 60 / 75 / 90 / 120 GiB = 20.9 / 15.2 / 10.0
/ 6.8 / 4.9 / 4.3. But the engine sizes the RAM copy as the free memory less a 28 GiB headroom once the cache is pinned, so a
bigger cache takes its GiB from the RAM copy: at the same total, 60 GiB + the RAM copy = 10.0, 75 GiB = 11.0, 79 GiB and no
copy = 11.2 - a static copy of the profile's hottest beats LRU slots for them. 60 GiB stays; the gain needs RAM the desktop
keeps.

**The full model, for scale** (the native FP4 experts, 289 GB, the same box): decode 3.2-3.8 tok/s on fresh text (4.4 with
DSpark, 5.0 when the text repeats); a short prompt ~80 s whatever its length (1,000 tokens 78.4 s, 2,100 79.9 s, 2,990 81.5
s: each chunk streams ~11,000-12,000 records, ~200 GB). The 217 GB pack decodes ~6x faster at KLD 0.13 / 91.6% same top.

**The final config** (`strata-dsv41.json`, the production binary of 2026-10-19): the 217 GB pack, 60 GiB SSD cache, the
learned expert profile; the route bias 0.3 (decode only); the GPU opt-ins HEADS_MMVQ and HC_SPLIT; PROMPT_KEEP and
PROMPT_KEEP_LONG; temperature 1.0, top_p 1.0, the whole vocabulary (top_k 0); reasoning effort max, loop recovery "stop";
vision through serve/vision_dsv41.py.

**Correction: the cold read's time varies with the drive; its bytes do not.** The final config's cold 44K read, rerun alone
with the production binary: 367.1 s, 546.7 GB at 1.48 GB/s (the engram waits 34.2 s) - against arm (d)'s 240.7 s at 2.24
GB/s for the same 545.7 GB, and a run beside a compile 455.0 s at 1.19 GB/s (the engram waits 130.2 s: the compile's memory
took the engram pages). So the drive's own rate moved 1.48-2.24 GB/s between clean runs of the same reads, and the claim above
is too strong: PROMPT_KEEP_LONG cuts the drive's bytes by 27% in every run (751.5 -> 545.7-547.2 GB), which made the read
15-44% faster across today's clean runs. A timing run must have the box to itself; prefill numbers come with their drive rate.

**A session on the final config** (the production binary, `strata-dsv41.json`'s args and env, the learned profile):

| step at ~45K of context | the path | read | records: drive / cache | then |
|---|---|---|---|---|
| the cold 44K read (beside a compile) | prompt path, PROMPT_KEEP_LONG | 455.0 s | 37,528 / 19,949 | 1,024 tokens: 10.9 / 19.7 / 18.9 / 21.7 tok/s per 256 |
| a 56-token follow-up | decode windows (exact routing) | 9.49 s | - | 32 tokens in 2.58 s |
| a 113-token follow-up | prompt path | 9.6 s | 1,358 / 2,399 | 32 tokens in 2.19 s |
| a 422-token follow-up | prompt path | 19.0 s | 3,404 / 3,239 | 32 tokens in 2.13 s |
| a 1,522-token follow-up | prompt path | 30.0 s | 5,443 / 3,937 | 32 tokens in 1.64 s |

- Decode right after a PROMPT_KEEP_LONG read is decode as before (the loop arms without it: 11.7-13.4 tok/s for the first 512,
  then 21-25). A follow-up is its records from the drive (2.0-2.6 GB/s there): ~9.5 s for one under ~100 tokens, the windows
  and the prompt path costing alike at that size. The 1.3 s short follow-ups of the server's log of 2026-10-19 routed their
  prompt windows with the bias (#33); routing them exactly, as the goal required, costs the difference.

**Decode against its floor, the final config** (`--decode-runs w1,w1,w1`: a warm-up on wikitext, then 192 plain tokens three
times on an agent coding prompt; 11.24 / 22.90 / 20.59 tok/s, the third under nsys; the third run's decode kernels, after the
prompt's last prefill kernel, 19.6 tok/s = 51 ms a token):

| part | what bounds it | floor | measured | of the floor |
|---|---|---|---|---|
| the dense projections (`native_small_mmvq`, ~600 calls a token) | ~7.15 GB of Q8_0 a token at ~700 GB/s | 10.2 ms | 12.5 ms | 82% |
| the VRAM experts (`native_mmvq_grouped`, 2 calls a layer) | ~181 Q3_K/Q2_K experts, ~2.75 GB, at ~700 GB/s | 3.9 ms | 8.1 ms | 48% |
| the rest of the GPUs' kernels (mixers 1.9, attention 1.0, routing 0.8, norms 0.9 ms ...) | - | ~2 ms | 8.0 ms | - |
| the CPU pool (58.8 experts from RAM and the SSD cache a token) | 0.84 GB at the DRAM's ~73 GB/s | 11.5 ms | 14.5 ms (57.7 GB/s) | 79% |
| the drive | 1.2 records a token | ~6 ms | 6 ms waited | ~100% |

- The GPUs wait for the CPU pool 8.2 ms (GPU 0) and 10.2 ms (GPU 1) of a token (the doorbell). The largest gap left is the
  VRAM experts' K-quant dot products at 48% of the cards' bandwidth (unpacking-bound, not memory-bound on this GPU): a kernel at
  ~90% would save ~4 ms of ~44 (+10% decode) - kernel work, not a switch (DSV41_BACKLOG #40). The router's GEMV (~20 us a
  layer at ~390 GB/s) is under 1%.

**Windows against single steps, resolved (#36c).** The cause is the attention's split-K count: `dsv4_attn_splits(T)` gives one
row 32 parts and a window of 8 four, so the same products are summed in another order. With `STRATA_DSV4_ATTN_SPLITS=32`
(opt-in; decode windows of 1-8 rows only, prompt sub-batches keep the default) windows of 8 and single steps give the same
bytes on wikitext chunk 0 (the tier frozen, the slots fixed) - without the GPU opt-ins and with them, so HEADS_MMVQ and
HC_SPLIT are row-count invariant. It stays off in `strata-dsv41.json`: the difference it removes costs no quality (above),
and its price on a verify window was lost in the run-to-run noise (8-row stand-in windows 6.11 vs 5.53 tok/s, while the plain
runs, whose attention is the same in both arms, moved 10.99 vs 9.75). Set it when drafts must never change the text.

## 13. The re-audit of 2026-10-19 and its fixes

Six read-only reviews (the forward math against model.py, the tiers and sessions, the server's deepseek41 path, vision,
speed, config/docs/tests) found no error in the forward pass - the engine matches model.py, kernel.py and engram.py op for
op but for the rounding §6 documents (V4.1 has no hash layers; layer 20's candidates mask layers 24, 28, 32 and 36) - and
fifteen items around it (DSV41_BACKLOG #43-#57). What was fixed, and how each fix was checked. The engine's checks are in
`moex/tools/dsv41/checks/` (README there); the server's are CPU tests in `serve/`.

**Requests with a picture (#43).** Their text was read with no checkpoints, no progress lines and no STOP: a picture
question's follow-up read the whole conversation again (and a long silent part could outlast the server's patience).
V4.1's text parts now read with the prompt path's control and the text path's checkpoints (one planner for both), and a
last cut behind a short text - read through windows, where no chunk ends - is snapshot when the windows reach it. The same
conversation through the server (thinking on, no tools, a 640x480 picture of "7319"; `vision_followup43.py`):

| | the follow-up's prompt | reused | read | its read |
|---|---|---|---|---|
| before | 382 tokens | 0 | 382 | 23.3 s |
| after | 274 tokens | 243 | 31 | 3.7 s |

Both answers right ("7319", then "7320").

**Checkpoints every 65,536 tokens (#50).** Each periodic cut is a bounded-replay read start, so the chunk ending there ran
layers 21-39 over 2,560 rows. The cold 44K read (the production binary, the engram tables evicted, a fresh engine each):

| `--prompt-cache-every` | experts streamed | the drive's bytes | time |
|---|---|---|---|
| 16384 (the engine's default) | 58,705 | 498.6 GB | 647.7 s |
| **65536 (strata-dsv41.json)** | **40,985 (-30%)** | **350.5 GB (-30%)** | 251.9 s |

The time difference is partly the drive's (73 vs 27 ms a request in the two runs); the bytes are the sure part, as
modelled (-29%). The first token's top-4 agree; another chunking (8,819-token chunks) moves the logprobs by rounding. What
it gives up: a resume point every 16K tokens between the root and the last turn (a cancel or an edit far back falls back
to a 64K boundary). The same read was 9% fewer bytes than §12's recheck already (498.6 against 546 GB) from #42: the root
checkpoint at 34,859, inside the conversation, is no longer cut.

**The next chunk's engram rows (#51).** The rows depend on the token ids alone, so they are fetched into a second buffer from
the moment a chunk's own have arrived, and the next chunk takes buffer and fetch over (`STRATA_DSV41_EG_AHEAD=1`, in the
config; out for a test on 2026-10-20 and back - not the drive's slow mode, §14). The cold 44K read (the dev build, the config's other switches): the first token's distribution bitwise the
same; its engram layers waited 4.0 s for their rows instead of 16.6 s; the read took 243.9 s instead of 263.1 s (-7%), the
drive's bytes the same (350.4 / 349.9 GB).

**Startup (#52).** The load reads only the planned records of a pack (a 16-record run whenever one of them was planned):
41.08 GiB in 18.5 s, where it read 195.12 GiB in 81.2 s for the same tiers (~1,795 VRAM slots, a 17.3 GiB RAM copy).

**Layer 20's candidate tie (#54).** The single tie at the threshold (distinct block maxima) is found by every thread at
once, the serial walk kept for several ties: the same bits (`dsv41_kernels_parity`: V4.1's shape, the tie cases).

**Decode's pre-routing chain (#55): measured, nothing to take.** The nsys profile of §12 split at each layer's routing
hand-off (`nsys_layers.py`): within a GPU's half of a token, a layer's step from the last wait to its routing is 368 us
median and 99% kernel time - no launch gaps - most of it the dense Q8_0 projections (240-266 us a layer, the kernel at ~82%
of the bandwidth). At most ~4% is left there. The split re-prices #40 instead: GPU 0's layers wait ~0 for the CPU (median
3 us, its VRAM experts' 317 us set the pace), so a K-quant kernel at ~90% would save an estimated 1.5-2.8 ms a token
(+3-6%); GPU 1 waits for its CPU pool (median 189 us).

**The router's softplus (#56).** V4.1's router now takes torch's softplus, log1p(e^x), as model.py's Gate (V4 keeps
topk-moe's log(1 + e^x), llama.cpp's). A parity case against a log1p reference passes (rel 8e-8; 1.4e-3 off the old form on
logits near -11, the negative control). Wikitext 16x512 against the native base through the prompt path:

| the router's softplus | mean KLD | same top | perplexity ratio |
|---|---|---|---|
| log(1 + e^x) (the production binary before) | 0.1328 +- 0.0069 | 92.01% | 1.0902 |
| **log1p(e^x) (model.py)** | **0.1306 +- 0.0067** | 91.81% | 1.0909 |

Inside the noise: the KLD's standard error is 0.007, same-top's ~0.4 points on 4,080 tokens (the difference 0.2).

**Robustness.** A failed O_DIRECT read in the prompt stream is read once more, and a second failure ends the chunk with an
error (it was computed as zeros, and the wrong state kept for later requests) (#47). The SSD cache is pinned within
MemAvailable minus the RAM copy's 28 GiB headroom (#48): a start with `--resident-headroom-gib 100` logged "the cache cut
from 60.0 to 11.1 GiB: MemAvailable 111.1 GiB, headroom 100 GiB" and "no RAM copy", and served. The decode doorbell takes
the device's counts after an aborted window (#56: a latent hang). The server starts text-only when V4.1's image encoder
does not start, and the config names the encoder's interpreter (`"python"` in the vision block, #49).

**The server (#44-#46, #56, #57).** Tool-result pictures go in the prompt's order (#44); `chat_template_kwargs`'
`thinking`, Anthropic's thinking enabled without a budget or adaptive, and the config's level for a request that turns
thinking on without one (#45); the reasoning loop guard also counts call blocks held inside the thinking that repeat one
held before, and an opt-in `"answer_loop_recovery": "stop"` looks at the answer (off in the config: an HTML table of
numbers measured 0.60 and tests repeating their setup 0.95 against its 0.5) (#46); a code fence closed right before
`</think>`; `/props`' top_k 0; structured output in the release's `## Response Format:` section; a past turn's `reasoning`
and inline `<think>`; a request's pictures encoded in one pass and held until it ends (70 tool pictures over 7 turns: 70
encodes instead of 140); count_tokens counts a picture's rows; opt-in `STRATA_DSV41_VISION_ATTN_ROWS=1024` puts the ViT's
attention through in blocks of queries - bitwise the one call on every shape compared, a 2400x1800 picture's peak 10.35 ->
1.26 GiB. Every other family's routes, parsers and generation loop
were compared with the tree before (2,860 results): Qwen unchanged; GLM-5, V4 and MiMo only in the fence case; V4.1 only
where intended. The serve suite: 796 tests, OK.

**Startup (#52), the cache refill.** Opt-in (`STRATA_DSV41_CACHE_REFILL=1`): the SSD cache's records are listed with the
expert profile and read back at the next start, in the background. On a 2,000-token prompt sent right after the start the
cache gave 5,448 records instead of 4,225, but the read took 87.5 s instead of 67.5 s - the refill's own reads (~60 GB)
compete with the request's, and cold runs of the same prompt spread 67-135 s. Off in the config.

**Tests, tools, logs (#57).** One start line lists every DeepSeek switch set ("DeepSeek switches: DSV41_ROUTE_BIAS=0.3 ..."); a pack's expert profile names its pack (FNV-1a 64 of experts.tsv) and the server starts from a learned profile only on that pack, saying which profile it uses (the first builds took the FNV offset basis with its last digit lost, so the server took their profiles for another pack's and started from the config's - from the restart of 2026-10-20 01:37 on; the engine now writes the standard digest and the server accepts both); `phase1_checks.sh all` skips the dropped engram reference; speed_model's defaults are the 217 GB pack's; the checks behind this section are in `moex/tools/dsv41/checks/`; deepseek41's picture pad runs, its loop guard and `chat_defaults` have CPU tests.

**Not done.** #53 stays open; its first step is in: a request that shares far more of the engine's tokens than it can reuse
logs where they part ("this prompt shares N tokens with the engine's M but reuses R"), which shows what a client's long
sessions need.

## 14. The server log of 2026-10-20: the learned profile, deep checkpoints, the engram reader and the drive's slow mode

Read from the engine's statistics lines only, for the run of 2026-10-20 01:37-11:00 (470 requests, conversations up to 178K
tokens), and from a 1 Hz record of the drive, the memory and the pressure taken beside it (counters only).

**The learned profile (#57's digest).** The engine took 1469598103934665603 for FNV-1a's offset basis (the standard
14695981039346656037 with its last digit lost), so the digest it wrote into a profile's header never matched the server's:
from the restart of 01:37 the server took the learned profile for another pack's and started from the config's.  Fixed
(4a3824ea): the engine writes the standard digest, the server accepts both.

**Deep checkpoints (`--prompt-cache-deep`, every family).** In that run 11 requests read 45-163K tokens again from the
4,731-token system-prompt root: 6,616 s, 42% of the run's prompt time.  In 5 of them the prompt shared 67-142K tokens with
the engine, but no checkpoint was left in between: the periodic ones were counted from each read's start (a conversation
read a turn at a time got none), and the ones a long read made went with the turns' LRU (`--prompt-cache 6`) five turns
later.  Now they sit at the multiples of `--prompt-cache-every` (65,536 in strata-dsv41.json, the engine's 16,384
elsewhere) and at most `--prompt-cache-deep` (4) of them are kept apart from the turns' (over the cap, the one whose
neighbours lie closest goes); `--prompt-cache-deep 0` is the old policy.  `ck_deep.py`: a 4,777-token system prompt and a
first message to 65,177 tokens, then 13 short turns past 65,536; the conversation sent again with the message right after
the crossing changed resumed at 65,536 and read 159 tokens in 7.2 s - the old policy read ~61K tokens from the root.  The
turn whose read crossed 65,536 took 9.8 s against its neighbours' 9.1-9.4 s.  Those 5 requests of the run read 3,627 s
and 4.7 TB from the drive; resumed at their deep checkpoints, ~600 s.

**The engram rows through Strata's n-gram reader (#59, opt-in).** `STRATA_DSV41_EG_READER=1` reads a prompt chunk's rows
through Qwen's `ngram/ple_reader` (a chunk's pages deduplicated and sorted, 64 in flight a table, a 1 GiB row cache);
the decode keeps the mapped rows.  `eg_reader.py`, the engram tables dropped from the page cache first: a 12.8K-token text
cold, the same text behind another system prompt (no prefix: the rows read before), a 430-token follow-up.

| arm | cold | again | follow-up |
|---|---|---|---|
| mapped | (a slow-mode episode: 129.6 s) | 48.5 s | 19.7 s |
| reader | 54.2 s (engram wait 1.9 s; 256K reads) | (an episode: 115.1 s) | 22.6 s |
| mapped, look-ahead | **52.9 s** (engram wait 1.9 s) | **48.7 s** | **19.7 s** |
| reader, look-ahead | 56.1 s (1.1 s; 256K reads) | 49.9 s (100% from its cache) | 21.9 s (88%, 4,957 reads) |

The first token's top 8 is the same in the mapped and the reader arm, bit for bit (5 decimals), for the cold and the again
read.  Between runs its tail moves in the second decimal (the cold read's token 671 at -16.20070 in the arms without the
look-ahead and in the production binary with it, -16.18020 in the look-ahead arms; the top token at ~0 everywhere): the
decode window's CPU experts are summed in the order their records come in, which a run's timing sets.  On this box the
mapped reads with the look-ahead are the fastest: the reader stays opt-in, and the look-ahead is back in the config.  The
production binary through the server (the config, the look-ahead on): 52.9 / 48.6 / 19.8 s.

**The drive's slow mode (#59).** From the run of 2026-10-20 the drive dropped, ~64 s at a time, to ~490 requests a second
(63 MB/s; ~1,000 a second for 4 KB ones) whatever was queued, and came back to 2.7 GB/s within a second: 24 of 374 prompt
reads and 27 of 524 replies took it after the build of 2026-10-20 (a reply at 2.5-8.5 tok/s, a 450-token read 67-170 s
instead of ~15 s).  Not heat (the drive's log: no thermal event; 43-57 C in the record), no error, write, swap or other
program, and not the engram look-ahead (three episodes in this section's checks without it).  `drivetest.py` reads the
pack's records with O_DIRECT as the engine's stream does, without the engine (12 threads; a second under 150 MB/s with the
load on counts as slow):

| | what was read | the load | slow seconds |
|---|---|---|---|
| E1 | the pack | nonstop, 300 s (773 GB) | 0 |
| E3 | the pack | 8 s on / 0.4 s off, 240 s | 0 |
| E2 | the pack | 8 s on / 2 s off, 300 s | 108 (two episodes back to back, right after E1) |
| E5 | the pack | as E2, and a 4 KB read every 50 ms in the gaps (no power state) | 53 |
| E4 | the pack | 8 s on / 4 s off, 240 s | 46 |
| E8 | MiMo's and GLM-5.3's GGUFs (read rarely) | 8 s on / 2 s off, 300 s | 0 |
| E9 | the pack's L33-L36 | 8 s on / 2 s off, 300 s | 0 |
| E11 | the pack's L20-L31 | 8 s on / 2 s off, 180 s | an episode from ~170 s |
| E13 | fresh copies of L20-L31 | 8 s on / 2 s off, 510 s | 0 |

The drive rewrites the blocks the engine reads most - read disturb: the pack was read ~150 TB in two days, a warm record
~1,500 times - and it does that in the next lull of a second or more after it has read a worn block (never under a steady
load, E1 and E3; the keep-alive's 4 KB reads do not hold it off, E5); for ~64 s the host then gets its leftovers.  On
2026-10-19 the pack was a day old: its short reads never met it (its two longest re-reads ran at 1.61-1.76 GB/s, perhaps
the first).

The fix: `moex/tools/dsv41/refresh_pack.py PACK --verify` writes every record file back where it is, with O_DIRECT (16 MiB
pieces, each read back and compared): the same bytes on new blocks, no free space and no trim needed - 217.0 GB in
1,058 s, the server stopped.  After it, E2's load on the whole pack for 600 s (E14): 1,288 GB, no slow mode (three
seconds at a load phase's edge, the next ones at 2.7 GB/s), where the same load had it within 1-2 minutes every time
before; a load phase reads 2.6-2.8 GB/s.  The engine says when it is back: a prompt read of 10 s or more under 1.2 GB/s
with the drive 85% busy is the slow mode's, and three within an hour put a line in the log that names the tool.  How often:
the pack met it after about a day of this use (written 2026-10-18, the first slow short reads 2026-10-20 00:06); the deep
checkpoints read ~15% fewer bytes in a run like 2026-10-20's (4.7 of its 25.2 TB went to the five re-reads they shorten).
A refresh writes the pack once (217 GB); the drive's own refresh rewrites the same blocks, a minute at a time in the lulls.

**The configuration for this box (strata-dsv41.json).** The deep checkpoints (the engine's default `--prompt-cache-deep 4`,
every 65,536 tokens with the config's `--prompt-cache-every`); the engram look-ahead on (`STRATA_DSV41_EG_AHEAD=1`, not the slow mode's cause, ~7% on a cold 44K read,
§13) and the mapped engram rows (the reader opt-in); the rest as §12-§13 left it - the route bias 0.3, HEADS_MMVQ, HC_SPLIT,
PROMPT_KEEP and PROMPT_KEEP_LONG, ROOT_AT_SYSTEM, `--prompt-cache-every 65536`, the 60 GiB SSD cache.  And
`refresh_pack.py` with the server stopped when the log says so.

## 15. The audit of 2026-10-20 (and Maya v1.1.16-17)

Seven read-only reviews while the server ran - the forward math against the release's model.py / kernel.py / engram.py,
the tiers and the drive's I/O, sessions and checkpoints, the server's DeepSeek path, vision, speed from the log's
statistics, configuration and operations - and a comparison with Project Maya's new DeepSeek-V4.1 engine. What they found
is DSV41_BACKLOG #61-#76; the items below were checked against the code.

**Nothing wrong at the core.** The forward pass matches the release op for op (decode, prompt path, image rows), more
precise where it differs (fp32 streams and scores where the release keeps bf16). The template matches the release's
encoding.py, and dsh's round trip reuses: 738 of 745 requests that continued a turn reused all of it. No state-consistency
bug in reuse, cancel, rollback, images or restore. The image path matches the release's image plan, pixels, ViT and aligner.

**Bugs and gaps worth fixing** (each small): a failed record read in the decode ends the engine (#61; the prompt stream
reads it again); the deep checkpoints of §14 are made inside batched reads only, so a mark a reply or a short read crosses
gets none - ~58% of 2026-10-20's marks (#62; a snapshot at the next request's start, ~28 MB, fixes it); a user turn that
ends with a picture loses its last-turn checkpoint (#63); no loop guard inside a tool call's argument (#64); the image
encoder's cold first pass, the likely cause of #58 (#73); a DSML block parsed as a call when no tools were offered (#65).

**Speed, exact** - the per-token and per-request budget of the run before the refresh (559 requests, 39,930 s; decode 58%,
prompt reads 42%, the drive's slow mode ~8%):

| | |
|---|---|
| a decode token | 62.4 ms: routing wait 22.2, CPU pool 22.8 (78 experts), drive 14.5 (3.05 reads), rest 3.0 |
| a ~500-token follow-up | 19.8 s: ~7,200 experts streamed, ~3,340 from the drive (48 GB) = 1.10x the drive's floor; PCIe ~74% |
| a 148,838-token read | 815.6 s: 143,415 experts streamed, 72,948 from the drive (1,098 GB at 1.34 GB/s) |

The levers, largest first: layer-major prompt reads (#76, from Maya v1.1.17: a long prompt's experts read once per layer
instead of once per chunk - est. -82% drive bytes and ~1.6-2.1x on that 148,838-token read; large); GATE_AHEAD with a
deeper read-ahead for long chunks (#66: long reads spend 20-24% with no read in flight waiting for each layer's routing,
up to ~20%); the cache for the prompt's reads (#67: only decode hits refresh the LRU, the prompt path is 73% of the drive's
bytes; ~7% fewer drive reads on follow-ups modeled); a smaller ring, a better order of a layer's copies, the refill off the
critical path (#68, a few % each); `--threads 19` (#69); wider chunks (#8, with Maya's V100 evidence: -15-19% est. on cold
long reads); pipelined speculative decode across the GPU split (#70, modeled +12-25% decode, very large).

**Maya v1.1.16-17's DeepSeek-V4.1 engine** (src/ds4; read, not run): a 178 GB IQ2/IQ3 pack, experts read through the page
cache with a madvise read-ahead, no route bias, two checkpoint slots, a 256 MB engram row cache, every layer over every
chunk. Its one idea worth taking is v1.1.17's layer-major order (#76). Not to take: the page-cache tier (what Strata-X left
after oomd's kills), the decode read-ahead and early "likely" prompt reads (measured and dropped here, #31; Maya's own
counts show their misses), the lossy disk skip and Q6_K engram tables (no measurement in its repo), its pack (IQ types run
39-50 GB/s on this CPU against 74-77 for Q3_K/Q2_K). Its own DeepSeek figures (one RTX 3090, 64 GB RAM: 4-5 tok/s
answers) are below this box's ~15.

## 16. A pack on Unsloth's allocation and imatrix, and Maya v1.1.18-20 (2026-10-21)

The question (the user): our Q3_K / Q2_K types placed per layer as Unsloth's UD-Q2_K_XL places its bits, with the release's
MXFP4 downs where Unsloth keeps them, calibrated with the imatrix Unsloth published - better than the 217 GB pack?

- **Unsloth's imatrix** (`imatrix_unsloth.gguf` from unsloth/DeepSeek-V4.1-Flash-1017-GGUF, 789,186,368 bytes, SHA-256 as
  the repository lists it; read by our own parser): 558 chunks of 4,096 tokens of conversations plus a probe for rarely
  routed experts, ~800K routed tokens a layer against our 196,608 - a median of 9,529 tokens per expert against 2,522 (our
  rarest tenth: 476, theirs 4,594). Its importance vectors against ours: cosine 0.985 on gate/up, 0.71 on the downs.
  `moex/tools/dsv41/imatrix_from_gguf.py` converts it to our format: 456,192,028 bytes, our file's size (the GGUF keeps
  gate and up apart; they are the same numbers - the same input - and are checked equal).
- **The allocation**, from UD-Q2_K_XL's own headers, each of its types to the nearest of ours by bits:

| layers | gate / up | down | record |
|---|---|---|---|
| 0-1, 15-19 | Q2_K | MXFP4 (the release's bytes, copied) | 14,008,320 B |
| 2-4 | Q2_K | Q2_K | 11,612,160 B |
| 5-14, 20-24 | Q2_K | Q3_K | 12,812,288 B |
| 25-39 | Q3_K | Q3_K | 15,208,448 B |

- **The engine and the tool** (d0397ee6): `experts.tsv` can give a layer's downs a type of their own ("GU,D"),
  `requant_pack --types GU/D:LAYERS` writes such layers, and an MXFP4 part is the source record's bytes. A pack with one
  type a layer reads as before.
- **Built** in 64 min (20 threads, the server stopped; `requant_pack --src native --dst q3q2u --types q2_k/mxfp4:0-1,15-19
  --types q2_k:2-4 --types q2_k/q3_k:5-14,20-24 --types q3_k:25-39 --imatrix imatrix/imatrix-unsloth.dat`): 212.43 GB in 4
  record sizes. `checks/verify_q3q2u.py` checked every layer's types, stride and size and the copied downs against the
  native records: all pass. The slots stay 14.5 MiB (the engine's rule: the most experts per byte), so the GPUs hold as
  many experts as before (1,808-1,878 in the runs below).

**Quality against native** (2026-10-21, `checks/kld_q3q2u.sh`: the same binary, each arm a fresh engine; the agent text is
the generated code / agent text of §12 through decode windows):

| | the 217 GB pack (q3q2) | **q3q2u** |
|---|---|---|
| wikitext: mean KLD | 0.1317 ± 0.0069 | 0.1328 ± 0.0080 |
| wikitext: median / 99% / max | 0.00110 / 2.17 / 8.0 | 0.00090 / 2.25 / 10.2 |
| wikitext: same top / perplexity ratio | 91.79% / 1.091 | **92.55% / 1.024** |
| agent text: mean KLD | 0.0305 ± 0.0017 | 0.0317 ± 0.0015 |
| agent text: median / 99% / max | 0.00033 / 0.304 / 0.943 | 0.00045 / 0.318 / 0.728 |
| agent text: same top / perplexity ratio | 94.43% / 0.975 | 94.38% / 1.004 |
| agent text, route bias 0.3: mean KLD | 0.0379 ± 0.0021 | 0.0397 ± 0.0021 |
| agent text, route bias 0.3: median / 99% / max | 0.00046 / 0.389 / 1.43 | 0.00059 / 0.448 / 1.54 |
| agent text, route bias 0.3: same top / perplexity ratio | 93.75% / 0.988 | 93.01% / 1.011 |

  - Nothing here is beyond noise except the wikitext perplexity: the mean KLD gaps are 0.1, 0.5 and 0.6 standard errors,
    the same-top gaps ~1.3 standard errors on wikitext (the new pack ahead), ~0 on the agent text and ~1 with the bias (the
    old pack ahead) - and the metrics of one row block are views of the same tokens, not independent votes. The old pack's wikitext perplexity
    ratio was 1.089-1.091 in three runs (two packs, two days); the new pack's 1.024 is far outside that.
  - Both pass the DSV4 Q3_K mark (KLD <= 0.134, same top >= 88.0%, perplexity ratio <= 1.101).
  - A per-token paired test (`--kld-save` on both packs) would settle the biased agent text; not run (the user chose).

**Speed** (`checks/speed_q3q2u.sh`: ab_every.py, a fresh engine with the server's flags, the cold 44K report prompt, the engram
tables evicted first, then 512 tokens; one arm each):

| | q3q2 | q3q2u |
|---|---|---|
| the prompt | 448.5 s | **205.2 s** |
| the drive meanwhile | 337.3 GB at 0.75 GB/s, 90.2 ms a request | 294.4 GB at 1.41 GB/s, 20.8 ms a request |
| the engram rows' fetch | 250.5 s | 37.2 s |
| decode, a cold cache | 4.0 tok/s (191.0 ms a token for the drive, 5.50 reads) | **14.3 tok/s** (23.9 ms, 5.01 reads) |

  - The old pack ran in the drive's slow mode (§14): its most-read blocks were worn again - the slow mode was back ~6 hours
    after the refresh of 2026-10-20 14:46 (the 20:56 run). A freshly written pack reads at full speed until its own
    blocks wear, so the 2.2x is fresh blocks against worn ones, not the recipe.
  - The recipe's own share: 12.7% fewer bytes from the drive on this prompt (layers 0-20's records, which every chunk of a
    long read streams: 121.2 -> 105.2 GB, -13%); decode the same (the speed model on the route-biased trace: 14.52 / 14.53 tok/s).

**Kept: q3q2u** (the user, 2026-10-21): quality equal within noise, ahead on general text's perplexity, ~13% fewer bytes on
long prompts, 4.6 GB smaller. `strata-dsv41.json` reads it; q3q2 is deleted - its recipe (experts.tsv, the build commands,
the measurements above) is in `~/AI/Models/dsv41-1017/recipes/q3q2/`, q3q2u's in `recipes/q3q2u/`, both imatrix files in
`imatrix/`. The check scripts (ab_every.py, serve_cancel*.py, drivetest.py, eg_reader.py) read q3q2u or the native engram.

**Maya v1.1.18-20** (28 commits since v1.1.17, 8d9a9b5..35bb2b2; read, not run):
- **#318 "skip layers 20-39"** is the bounded replay of §11: one GPU only, only at a run's end, only for 3,692 or more new
  tokens; ours serves every read start (the end and each checkpoint) on both GPUs. Its one part ours lacks: our replay
  starts at layer 21 (`replay_layer = last_src + 1`), so layer 20 runs its attention and experts over every row, where the
  rows before the tail need only its KV front (DSV41_PORT.md's S5 planned that). That is #77: rounding only - layer 20's
  experts over fewer rows change MMQ's stream-k sums - est. 2-4% on cold long reads.
- **Layer-major order** (#76) is unchanged since 6ae9d20. Maya's is not exact (its experts' outputs are summed in the order
  the stream delivers them), runs on one GPU, keeps no checkpoint inside a run, and still uploads every routed expert over
  PCIe once a chunk. Here it can be bit-exact: each (layer, chunk) keeps today's launches, and `moe_combine_rows` sums a
  row's experts in routing order. With the per-chunk uploads kept, GPU 0's link becomes the floor (~280 s of the
  148,838-token read, est.); a layer's experts held in the lent VRAM across its chunks would lift that (unchecked).
- **Stop inside a long read**: Maya checks it between (layer, chunk) units; ours before each chunk only - up to ~51 s of
  reading after a Stop at the 149K read's pace (#78, exact).
- **#300** vendors DeepSeek's encoding.py: ours already is that encoder, tested byte for byte. **#312** (per-PC tuning)
  adds nothing for the prompt path here: borrowing VRAM is our lend and refill, bigger chunks are #8, its page-cache reads
  do not apply (O_DIRECT and the pinned cache). Its placement-independent rounding (the GPU quantizes low-bit experts'
  activations as its CPU lane does: 96 of 96 tokens equal across placements) is our #75, rounding only.

## 17. A stop between layers (#78) and layer-major prompt reads (#76), 2026-10-21

Both in the config from 2026-10-21 (`STRATA_DSV41_STOP_LAYERS=1`, `STRATA_DSV41_LAYER_MAJOR=1`, `STRATA_DSV41_BOUNCE=96,32`),
both exact: the checks compared every bit of the first token's logits (the `STRATA_GLM_FIRST_TOP` probe now prints a hash
of the whole row) against the chunk loop's.

**How the checks compare two engines.** One tier in both (`--no-adaptive --expert-slots 1800`). The slots fill what a GPU
has free, GPU 1's free memory moves with the display, and a decode window - the prompt's last token goes through one -
computes a VRAM expert on the GPU and the others on the CPU, rounding apart: two engines whose tiers differed by 8 slots
(1855 / 1863) gave first-token log-probabilities ~0.04 apart on the same read.  The checks before this compared the top 8
to 5 decimals, in one engine.

**#78, a stop between a chunk's layers.** Each stage copies its running state (`state_regions`, ~14 MB, device to device in
the prompt path's loan) at the start of each chunk - the first stage keeps two copies (it runs a chunk ahead), the last
one - and looks for a stop before each layer.  After a stop every stage goes back to where the last stage got, so the state
is exact where it was (the read's start, a checkpoint) and the next request continues there as it would have.
`checks/serve_stop_layers.py`:

| the stop | where it landed | STOP to DONE | the next request |
|---|---|---|---|
| 8 s into a read's first chunk | before layer 7, GPU 0; back at the read's start (16384) | 2.9 s | resumed there, bit for bit |
| 8 s after a checkpoint (16384) | before layer 6 of the chunk at 24576, GPU 0; back at 24576 (no checkpoint: the live tokens go) | 3.0 s | resumed at 16384, bit for bit |
| once GPU 0 had read its last chunk | before layer 21, GPU 1; both GPUs back at 34859 (the root) | 3.4 s | resumed there, bit for bit |

The switch on without a stop gives the output of it off.  Before, a stop waited for the chunks in flight: up to ~51 s at
the 149K read's pace.

**#76, layer-major prompt reads.** GPU 0 reads a window of chunks layer by layer - each layer over every chunk of the window
before the next - and each layer's non-resident experts stream once into a ring that holds a whole layer (384 slots), kept
until the window's last chunk has run the layer (`Stream::hold`).  Each chunk keeps its rows that cross layers (the
streams, the carried pre-mix, the top-k lists, the candidate masks: `replay_shift`'s list), and every (layer, chunk) runs as
the chunk loop runs it - the same rows, positions, sub-batches, routing and MoE groups - so the output is the same bit for
bit (MMQ's stream-k sums by launch shape, `moe_combine_rows` in routing order; Maya's version sums its experts in arrival
order and is not exact).  The window's running state at each chunk end is put together layer by layer in copies of its
own: the checkpoints there and a stop's way back read them.  GPU 1 stays the chunk loop (layer 20 and the replay tails),
the window's chunks handed on together.  A window is as many chunks as GPU 0's loan holds (`STRATA_DSV41_LM_CHUNKS` caps
it, 4): 4 at 64K context, 2 at the server's 512K (the indexer scores' scratch).

Checks: `checks/serve_layer_major.py` (windows of 4, a checkpoint every 16,384 tokens) - the 44K read from 0, the follow-ups
from the root checkpoint and from one inside a window, a stop inside a window and the read after it: all bit for bit with
the chunk loop; `checks/serve_pic_lm.py` - a picture behind a 31,042-token text through the whole server, its text read in
windows of 2: the first token bit for bit; `checks/serve_same_default.py` - with both switches off the build is the one it
replaced (the 44K read and a follow-up, 32 greedy tokens each, the top 8 to 5 decimals).  The final binary passed all three
(`3f84b96df45883c2`, the one installed).

**Speed** - the cold 44K report prompt, the engram tables evicted, fresh engines one after the other, the server's flags
(60 GiB cache, 512K context, a checkpoint every 65,536 tokens, the root at the system prompt):

| | the prompt | the drive meanwhile |
|---|---|---|
| the chunk loop (`STRATA_DSV41_EG_AHEAD=1`) | 166.9 s | 299 GB, 1.79 GB/s, 17.7 ms a request, busy 65% |
| **layer-major, windows of 2, 96 bounce slots on GPU 0** | **125.3 s** (x1.33) | 250 GB, 2.00 GB/s, 21.2 ms a request, busy 68% |
| ... with the n-gram reader (`STRATA_DSV41_EG_READER=1`) | 127.0 s | 253 GB, 8.5 ms a request |

All three arms' first-token logits were equal.  What it took, on the test flags (64K context, windows of 4, the cold 44K
read; the chunk loop 219.2-226.7 s):
- **The engram rows.** Layer-major streams a whole layer ahead with no gaps, and the engram's 4 KB page reads (~64 a token,
  ~2.8 M for the 44K prompt) queued behind the records - 55 ms a request against the chunk loop's 16: a chunk's rows
  fetched with the window waited 116.7 s (351.7 s in all), one fetch for a whole window 196.9 s (342.5 s).  Fetched chunk by
  chunk on one worker, the next window's queued right behind (each chunk's first engram layer waits for its own): 8.2 s.
- **The read-ahead.** With the ring full the drive had nothing to read while a window's later chunks ran (35 s): 96 bounce
  slots on GPU 0 (`STRATA_DSV41_BOUNCE=96,32`) took the read from 342.5 to 269.7 s; 17 s remain on the server's flags.
- **Not taken:** the engram reads first (`STRATA_DSV41_EG_FIRST=1`, opt-in: the record reads held while each fetch runs) -
  650.4 s against 204.5 s without (the drive slow, below; the records waited 522.6 s behind the slowest reads); the n-gram
  reader - no gain.

**The drive during these tests.** The same prompt was read cold ~20 times on 2026-10-21 (the pack written at 16:27, the
engram pages evicted before each engine) and the drive slowed by ~20:00: the same layer-major config 125.3 s, then 204.5 s
(2.00 -> 1.23 GB/s, 21 -> 48 ms a request) at 39-43 degrees C - the slow mode of §14, its worn blocks now the test prompt's.
At 20:16 a read with those pages cached ran at 2.56 GB/s, 13 ms a request; the next cold reads at 39-77 ms.  The desktop
lags with it (the root filesystem is on the drive: at 161 ms a request every program waits).  The speed numbers above are
from before; the bit-for-bit checks do not depend on it.

**Off again the same evening (the user, 22:43): `STRATA_DSV41_LAYER_MAJOR` and `STRATA_DSV41_BOUNCE` are out of the config.**
After a reboot (an empty page cache) the server read an 84,524-token prompt from 0 for 670 s without one chunk reported -
the console's "X of Y" never came: layer-major reports only after a whole window - and the desktop was sluggish throughout
(the load average 14).  No engine log (diagnostics off).  The drive was healthy afterwards (`drivetest.py 90 12 8 2`: 194.5 GB
at 2.4-2.8 GB/s, no slow second), so not its slow mode on the pack: the likely cause is the cold engram rows - ~5.4 M random
4 KB page reads for that prompt - behind the window's deep record queue (96 bounce slots on GPU 0, no gaps), where the chunk
loop lets them through its gaps; every other program's disk reads waited the same way.  The 44K checks evicted the engram
pages but read half as many, and none started from a reboot.  `STRATA_DSV41_STOP_LAYERS=1` stays (it reads nothing more).
Before layer-major goes back on: a read-ahead that leaves the engram's reads room, and the cold read from a reboot measured
(DSV41_BACKLOG #81).

**Also:** the decode line (with diagnostics on) gives each token's engram time - the slow tokens of a session can then be
matched with the engram's page reads (the user's question: can those 4 KB reads behind the records also be the slowdowns
seen at times?).  The user deferred DSpark for decode (DSV41_BACKLOG #80).

## 18. The drive's slow mode handled in the engine: a guard, a bounded engram fetch, the scales pre-read (2026-10-22)

The goal (the user, 2026-10-21): the speed-ups and no lag - an end to the stalls that froze the desktop. The slow mode
(§14) is the drive's firmware's; the engine decides only how much of the drive's queue is its own while it lasts.

**What the logs and counters showed.**
- **The 670 s read** (22:42-22:54, the first after the 22:38 reboot, §17): no engine statistics (diagnostics off). Memory
  and CPU played no part: from the boot on the kernel swapped and reclaimed nothing (`/proc/vmstat`: pswpin, pswpout,
  pgscan, pgsteal, workingset_refault all 0; `/proc/pressure`: memory 0.27 s in all, CPU 18 s).
- **The two reads after 23:05** (the heartbeat on) met an episode each, 60-70 s, in both read orders: layer-major (38,180
  tokens) at 4 MB/s and 1,020 ms a request while layer 1 waited for its engram rows; the chunk loop (45,839 tokens - its
  9,495-token chunks did not fit a window of 2 in GPU 0's loan) at 32-45 MB/s and 1,620-2,084 ms while layer 4's records
  streamed. Their normal samples: 9.8-39.4 ms. §17's reading - the engram's reads behind layer-major's queue - was wrong:
  the slow mode comes in either order, and layer-major's own fault was silence, no line until a whole window was done.
- **Why everything froze:** every request waits behind the queue - the record reader's (12 readers x 4 MiB calls = up to 384
  NVMe requests) and the engram fetch's (a chunk's 786K pages asked for at once, the queue full) - served at ~1,000 a second
  in the slow mode, and the root filesystem is on the same drive.

**The drive measured** (its normal state; `checks/floor_probe.py` and a depth sweep, reads only, the server stopped):

| reads in flight | 4 KB reads a second | 128 KB reads a second |
|---|---|---|
| 1 | 8,350 (0.115 ms median, p99 0.16) | 4,158 (0.222 ms, p99 0.34) |
| 2 | 16,588 | 7,682 |
| 8 | 39,914 | 12,407 |
| 64 | 93,854 | 17,526 |
| the engine's 256-384 | - | ~18,500 (the load pass, 2.26 GiB/s); 16-20K (prompts, 2.0-2.6 GB/s) |

One reader puts one request on the drive (`/sys/block/nvme0n1/inflight`: max 1; two readers, max 2), O_DIRECT reads and a
mapping's page faults alike. The disk's stat file reports 0 in flight throughout on this kernel (7.0); the heartbeat reads
the inflight file now.

**The engram fetch** (`checks/eg_fetch_bench.py`, one cold 8K-token chunk replayed as `eg_rows` reads it, 12 record readers
meanwhile, one run each): the scale pages cost 30-49% of it - all at once 13.71 s, the weights only 9.61 s; 1,024 rows
ahead 14.67 s, the weights only 7.41 s; no 250 ms sample over 22.1 ms a request. The scales' share: 50% of a token's 96
page accesses, 46.9% of a cold chunk's drive reads, 26.5% of a cold 84K prompt's (a scale page holds 512 rows' scales, a
weight page 16 rows'; from the files' sizes, rows uniformly at random).

**What the engine does (opt-in, all on in strata-dsv41.json):**
1. `STRATA_DSV41_EG_WINDOW=1024`: the rows' pages asked for at most 1,024 rows ahead of the copies (`eg_rows`).
2. `STRATA_DSV41_EG_SCALES_PRELOAD=1`: the scale tables (2 x 3.07 GB) read into the page cache once after the load, in
   order, by one thread that waits only while the guard is below full depth (6.14 GB in 1.8 s here). Page cache, not
   pinned: an evicted page is read again on demand.
3. `STRATA_DSV41_DRIVE_GUARD=1` (`moex/tools/dsv41/drive_guard.hpp`): the disk's counters every 250 ms. A sample at >= 80 ms
   a read and under 2,000 reads a second - never in the normal state, where the drive completes >= 4,158 a second at every
   depth measured - and it goes to its floor at once: one record reader, 128 KB a call; the engram rows' pages one at a time;
   the start-up load on one reader (the others wait between layers, holding none); the pre-read waits. Two samples at
   < 40 ms and >= 2,000 a second, and it steps back up a level a sample (1, 4, 16, 64, 256 engram rows ahead; 1, 1, 2, 2, 8
   readers; 128 KB, 512 KB, 1 MiB, 4 MiB calls) to no limit. It covers the prompts' and the decode's record reads, the
   engram fetch, the start-up load and the pre-read - not the dense weights' load at start (7.9 GiB in 3.9 s) or the opt-in
   O_DIRECT engram paths (`EG_DIRECT`, `EG_READER`). With diagnostics on it prints a line as it goes to its floor and one
   as it is back, with the drive's time a read at the floor - its lone reads' time in the slow mode - and in the first 2 s
   (the requests already queued, draining).
4. Layer-major back on (`STRATA_DSV41_LAYER_MAJOR=1`, `STRATA_DSV41_BOUNCE=96,32`).
5. Progress lines (`PL done all` on the serve protocol): one per (chunk, layer) of a read of 4,096 tokens or more, and one
   every 30 s while the engine reads from storage (`/proc/self/io`). The server's silence limit resets on them - an engine
   that is stuck reads nothing and still meets it - and its window says "reading the prompt: N tokens (37% read)".

**The checks** (`checks/serve_guard.py`, `serve_guard_episode.py`, `serve_guard_decode.py`: a 16,398-token text - this file
at 645844e9 - read from 0 and greedy tokens, one tier, the config's switches with layer-major; the first token's whole
logits row hashed):

| arm | what | the prompt | result |
|---|---|---|---|
| A | today's build | 150.4 s | logits 0808db712fac6f64; a real episode at 20-100 s, while the cold engram rows were read |
| B | the new one, the request at the guard's floor (counters the check writes) | 305.4 s | the same logits; the drive's in-flight count <= 2 in 99.91% of 134,329 samples (max 4) |
| C | the new one, 8 s of the slow mode's counters 20 s in | 69.8 s | the same logits; the floor 0.30 s after they began, full depth 1.56 s after they ended; the 221 requests queued drained within 0.5 s, then <= 2 in flight (none of 3,155 samples above) |
| D | the new one on the drive's own counters, the engram cold | 129.9 s | the same logits; a real episode 11.5 s in (366 ms a read, 1,000 a second): 66.0 s at the floor; another program's lone 4 KB reads meanwhile (`GLM-5.3`'s GGUF, one every 100 ms) median 1.77 ms, p99 2.84 ms (640 reads; max 541 ms as the queue drained) against median 14.3 ms, p99 22.7 at full depth before and after; the drive at the floor: 4 MB/s, 1.0 ms a request, 1 in flight |
| E | the route bias off: today's build, then the new one with its decode at the floor | 56.0 / 52.1 s | the same logits and the same 32 greedy tokens |

B, C and D's greedy tokens part from A's at the third: the route bias picks among the experts resident at the time, and the
RAM copy is sized by MemAvailable at each start (46.35 GiB in A, 50.44 GiB in B) - E checks the decode with it off. Also:
`checks/drive_guard_test.cpp` (the guard's rules on written counters) 9 of 9; serve/test_server.py 319 tests OK (three new:
PL lines keep a slow chunk alive, without them it is ended, the window's share). Found and fixed on the way: the load
pass's readers first waited holding a layer (held at the floor the load never ended); the floor's statistics left out its
slow samples; the pre-read waited for the prompts' fetches and so never ran beside a long read right after the start.

**In the slow mode the drive served one request at a time.** D's floor got ~1,000 reads a second at 1.0 ms each with one in
flight - the rate tonight's other episodes gave ~1,000 queued requests at 1,020 ms each. So the floor costs the engine
nothing there, and another program's read waits for at most one of the engine's. The bound for a drive that did serve
deeper queues faster: the engine reads at most ~63 MB/s in an episode (§14), so even with no progress at its floor the
work it held back takes <= 3.1% of the episode's length afterwards for the records (63 MB/s against >= 2.0 GB/s), <= 2.4%
for the engram (1,000 against >= 40,900 pages a second), <= 22% for the decode (<= 3.2 tok/s from 63 MB/s over >= 19.9 MB a
token, against 14.3-26.9 tok/s), plus <= ~1 s stepping back up.

**What stays.** The episodes: the drive's firmware chooses when (§14). Three of tonight's four began while a cold engram
fetch ran (the fourth while records streamed beside the next chunk's engram fetch) - a lead, DSV41_BACKLOG #83. The
user's run from a reboot, diagnostics on, is the scenario's own check; diagnostics go off after it.

## 19. The indexer scored in row blocks: layer-major for every long read (#84, 2026-10-22)

At the server's 512K context a read of 84,414 tokens plans 9 chunks of up to 9,438 tokens, and two of them did not fit GPU
0's loan (11.63 GiB) - the read ran the chunk loop (§18). The largest part that grows with the context is the indexer's
score buffer: one row of every position (n_ctx + 1 floats, 2 MiB at 512K) for each row of a prompt sub-batch, 512 rows -
1 GiB. Every kernel the scores feed computes a row on its own (`lid_scores_mma_kernel`: a row's tile of 64 positions in one
block, its MMA over the 128 dimensions and its sum over the 64 heads in a fixed order; the candidates, the candidate mask
and the top-k list: a block per row), so the sub-batch is now scored in blocks of 64 rows (`kDsv41ScoreRows`, V4.1 only)
into a buffer of 64 rows - the same bits, 0.88 GiB less loan at 512K. A decode window (at most 8 rows) is one block.

`checks/serve_score_blocks.py` (the server's context and switches, one tier: `--no-adaptive --expert-slots 1700`, the
engram tables in the page cache, the first token's logits hashed):

| arm | the read | the prompt | GPU 0's loan | the drive | logits |
|---|---|---|---|---|---|
| today's build | 16,398 tokens, windows of 2 | 60.2 s | 11.25 of 11.63 GiB | 116.5 GB, 1.94 GB/s | 47c719ff66b58cfc |
| the new one | the same | 60.3 s | 10.37 GiB | 118.4 GB, 1.96 GB/s | 47c719ff66b58cfc |
| the new one, the chunk loop | 84,414 tokens (this repo's docs) | 235.6 s | - | 488.1 GB, 2.07 GB/s | 7a165911a5d39ea0 |
| **the new one, layer-major** | the same, 9 chunks of up to 9,438 in windows of 2 | **159.5 s** (x1.48) | 11.01 GiB | 377.4 GB, 2.37 GB/s | 7a165911a5d39ea0 |

No slow-mode episode in any arm. The loan's 0.88 GiB is also 61 more expert slots left in VRAM during every prompt at 512K.

## 20. Where the speed stands, measured against this box's limits (the critic turn of 2026-10-22)

**Prefill.**
- **Short reads** (a session's turns: 176-610 tokens in 12.9-23.2 s) stream 31.6-52.0 GB from the drive at 2.2-2.4 GB/s -
  ~88% of the drive's floor (2.73 GB/s). What is left of the drive's time is the routing waits (1.2 s a read): reading a
  layer's records before its routing (`STRATA_DSV41_GATE_AHEAD`) would fill them, but a short chunk routes a fraction of a
  layer's experts (a 2,000-token one ~81%), so most of those reads would be for nothing - and more reads of the same blocks are
  what the drive's slow mode comes from (§14). Left off.
- **Long reads** run layer-major at every size now (§19). With the server's 60 GiB cache a long read's later windows find
  ~4,600 records in it (the prompt pins what it reads, `prompt_keep_slot`), so windows of 3 (the window's idle chunks parked
  in host memory) would save ~23 s of an 84K read against ~12 s of copies over GPU 0's PCIe x8 link - not taken. The ring
  stall (10.7 s of the 84K read, 4.0 of a 16K one) wants more bounce slots, ~0.9 GiB of the RAM copy for 64 - a decode cost
  (the RAM copy serves ~40 experts a token) for a prefill gain; the server's use is decode-heavy, so not taken either.

**Decode** (the server's way: the route bias, adaptive VRAM swaps; `checks/serve_decode_ab.py`, a 2,017-token prompt and
6 x 256 greedy tokens, one engine an arm): ~19-20 tok/s warmed - a token ~21 ms of GPU work before its routings (the dense
Q8_0 projections, 82% of the cards' bandwidth, §12) and ~28 ms in the CPU pool (8-10 ms of it waiting for 1.6-2.2 SSD
reads), the two in series layer by layer.
- **The CPU pool** reads ~75 experts a token from RAM at ~50-58 GB/s; this box's DRAM gives 72-75 GB/s to 8 or more cores
  (13.2 GB/s to one; a 2 GiB read-sum per thread count, 2026-10-22).
- **The VRAM tier**: a decode trace of 1,560 tokens (`STRATA_ROUTE_TRACE`) has its best fixed set of 945 + 808 experts - in
  hindsight - serving 71.9% of the routings; the adaptive swaps reach 67-71% from the 3rd pass on. The 1,700 most-routed
  experts serve 71%, 2,500 81%, 6,000 98.8%.
- **The SSD cache**: `tier_sim.py` on that trace (the server's tiers: VRAM 945 + 808, RAM copy 1,700, cache 4,656) - LRU 1.5
  SSD reads a token, LFU 1.5, 2Q 2.6 (with a RAM copy of the trace's own hottest; 2.8 / 5.5 / 3.3 in layer order). LRU stays.
- **The VRAM experts' kernel** (`native_mmvq_grouped`): RS rows a block (`STRATA_MMVQ_GROUPED_ROWS=2|4|8`, opt-in), each
  row's terms and sums those of the one-row kernel - `mmvq_grouped_parity` bitwise for all 16 types at 1, 2, 4 and 8 rows, and
  the first token's logits the same in the decode A/B. At 8 rows the kernel alone (a token's 5 VRAM pairs, the median of 7)
  takes Q2_K gate+up 131 -> 107 us, down 80 -> 58; Q3_K 208 -> 176 and 131 -> 93; MXFP4 down 98 -> 73: -21-28% a layer.
  Decode itself did not move (19.08 / 19.99 / 19.06 against 18.42 / 19.14 / 19.42 tok/s, passes 4-6): the VRAM experts run
  beside the CPU pool, which is the longer of the two. Off in the config (a default changes only where it measured faster).

**What is left, and why it is not taken tonight:** the GPU's kernels between the dense projections (the mixers, the
attention, the norms: ~8 ms a token against a ~2 ms floor, §12) - fusing them is several kernels' work and moves rounding;
the overlap of GPU 0's refill with GPU 1's last chunk (0.3-2.7 s a read) - GPU 0's loan holds the copies a stop puts back.

## 21. The start in 47 s, not 71: the cache and the RAM copy pinned another way (2026-10-22)

The server's start (the smoke test of the config, `checks/serve_smoke.py`) took 71 s to "ready". 28.1 s of it went to the
SSD cache's 60 GiB of pinned slots, 7.2 s to the RAM copy's 21.5 GiB, 19.0 s to the load pass (the drive at 2.24 GiB/s) and
7.4 s to the image encoder before the engine started. How this box pins host memory (16 GiB, 1 GiB at a time, driver 580):

| way | 16 GiB in | rate |
|---|---|---|
| `cudaHostAlloc` (the cache's way), 1 thread | 6.93-6.98 s | 2.3 GiB/s |
| ... 2, 4, 8 threads together | 7.41-7.87 s | 2.0-2.2 GiB/s: no gain |
| `cudaHostRegister` of untouched anonymous memory (the RAM copy's way) | 5.16 s | 3.1 GiB/s |
| the pages faulted in by 8 / 16 / 20 threads, then registered | 2.60-3.09 s (faults 1.0-1.4, registration 1.6-1.7) | 5.2-6.2 GiB/s |
| ... faulted by 16 threads a block ahead of the registration | 1.90 s | 8.4 GiB/s |

`cudaHostAlloc` also holds the driver's lock for the whole call: another thread's 14.5 MiB host-to-device copy waited 9.7 s
behind 16 GiB of it (2.4 ms alone), and a `cudaMalloc` the same. So it cannot hide behind the load either.

**`STRATA_DSV41_FAST_PIN=1`** (opt-in, on in strata-dsv41.json): the SSD cache's 75 blocks and the bounce slots are anonymous
memory, faulted in by 8 threads a block ahead of each block's `cudaHostRegister` (portable, mapped; `MADV_DONTFORK`, as the
driver's own mappings), freed with `cudaHostUnregister` and `munmap`; a block that cannot be mapped or registered is
`cudaHostAlloc`'ed as before. The RAM copy's arena is faulted in by 8 threads before its registration
(`STRATA_ARENA_PREFAULT`, which the switch sets). The same pinned memory either way: where a record's bytes sit changes no
arithmetic.

- **The engine** (`checks/serve_fast_pin.py`: the config's arguments and switches, `--no-adaptive --expert-slots 1700`, a
  2,000-token text): READY 63.0 -> 38.3 s; the cache 28.2 -> 7.3 s (75 of 75 blocks registered), the RAM copy 7.6 -> 5.0 s,
  the load pass 20.2 / 18.9 s; the engine's end after QUIT 13.7 -> 8.8 s; the first token's logits the same
  (2fe05b1ba835d9e9). The script's check of the registration matched the wrong part of the log line in that run and was
  corrected after it (the line reads "75 of 75 blocks faulted in and registered").
- **The server** (`checks/serve_smoke.py` with the config): ready 71 -> 47 s (the cache 6.3 s, the RAM copy 5.7 s); its
  681-token prompt 44.4 / 41.3 s, 200 tokens at 9.0 / 9.4 tok/s (a cold cache both times).
- **Memory.** The cache is the engine's own (anonymous) memory now, not the driver's: pinned, so it can be neither swapped
  nor moved, though the kernel's reclaim can still look at it. Since this boot (4.5 h with the server's runs and these
  tests) the reclaim scanned 81 anonymous pages and 3.35 million file pages: it takes page cache (the engram tables), as
  before. MemAvailable at the RAM copy's sizing was 50.3 GiB against 50.6 (the RAM copy 22.26 against 22.64 GiB), inside
  the spread of today's starts (21.33-22.64 GiB).

Not taken: the image encoder's start ahead of the engine's (7.2 s, one process after the other in serve/server.py); the load
pass (19 s, at the drive's rate for 42.5 GiB of records).

## 22. The image encoder warms up beside the engine's start: ready in 40 s (2026-10-22)

The server started the image encoder and waited for its READY (8.2 s: Python and torch 1.7 s, the ViT's weights 0.6-0.7 s,
the warm-up pass of #73 6.1-6.2 s) before it started the engine. The encoder's memory, read every 0.2 s from its start: 1.3
GiB from 2.4 s on, 1.48 GiB at the warm-up's peak - in place long before the engine first reads MemAvailable (the SSD
cache's sizing, after the CUDA start and the 3.8-4.0 s dense load).

**`STRATA_DSV41_VISION_EARLY=1`** (opt-in, on in strata-dsv41.json): serve/vision_dsv41.py says `LOADED` once its weights
are in memory, before the warm-up; the server (`start_vision(..., early=True)` at its first start only) starts the engine
then and reads the encoder's READY (`Vision.finish_start`) once the engine is ready, before it serves anything - so the
warm-up's point (#58: a picture's rows are never the process's first pass) holds. An encoder that says READY first, a lazy
start and a restart after an unload work as before; a failure after `LOADED` is handled as a failed start (an optional
encoder: one line, and text only).

- `checks/serve_smoke.py` (now with a picture request too, a generated gradient): ready 47 -> 40 s; the warm-up 6.8 s beside
  the engine's start (6.1-6.2 s alone); the engine's phases as before (dense 3.8 s, the cache 6.6 s, the RAM copy 4.6 s,
  the load pass 18.9 s); both replies streamed, the picture's 184 rows encoded in 5.7 s.
- The serve tests (41 modules, 815 tests): one failure, `test_responses.NoLeakedSampler.test_server_close_ends_the_sampler`,
  which fails the same way without this change when every module runs in one process (it counts the threads named
  "(_loop)") and passes in its own module.

The start tonight: 71 s -> 40 s. What is left is the drive (the load pass's 42.3 GiB at 2.24 GiB/s, the dense weights) and
~11 s of pinning.

## 23. The decode's read-ahead with the route bias: no gain one layer ahead (2026-10-22)

A warmed decode token waits 7-10 ms of its ~50 for 1.4-2.0 records read from the drive (§20). The read-ahead (5 C.3: the
next layer's router on this layer's FFN rows, the guessed SSD experts read into the cache before their layer asks) had lost
its A/Bs before the route bias. Two things were wrong with it since then:

- `STRATA_DSV41_AHEAD=1` turned nothing on: `ahead_on` starts false and the switch could only set it false. Fixed.
- The guess was the release's routing, without the route bias's penalty - the experts the biased routing mostly passes
  over. It now takes the penalty table the decode's routing takes (`route_pen_for(il + 1)`).

`checks/serve_decode_ab.py` (the server's decode: `DECODE_BIAS=1 DECODE_ADAPTIVE=1`, 6 passes of 256 tokens, the 30 GiB test
cache), passes 3-6:

| arm | tok/s | the drive a token | SSD records a token | waiting for the routing |
|---|---|---|---|---|
| off (the first run, `STRATA_DSV41_AHEAD=1` doing nothing) | 19.23 / 19.03 / 19.94 / 18.72 | 7.0-10.2 ms | 1.39-2.04 | 21.6-21.9 ms |
| off (the second run of the same) | 17.98 / 18.24 / 19.08 / 19.19 | 8.7-11.1 ms | 1.78-2.29 | 21.4-21.8 ms |
| on, with the bias | 17.54 / 18.40 / 18.45 / 18.77 | 8.6-11.1 ms | 2.33-3.10 | 23.0-23.8 ms |

The guesses are good - 1.39-2.02 records read ahead a token, 87-90% of them used - but they come one layer (~1.25 ms)
before a read that takes ~5 ms, and the guess itself (a norm, the 384 x 5120 router, the top-k, per layer, before the
doorbell) adds 1.5-2 ms a token to the GPU's side. The two "off" runs differ by 3%: the noise of this test. Off it stays.
A guess several layers ahead would cover more of a read, at a lower precision and more of the drive's reads (DSV41_BACKLOG
#86).

## 24. The VRAM slots sized by their experts (2026-10-22)

The VRAM tier's slots were all the largest blob's size, 14.50 MiB (q3q2u's layers 25-39). Every other layer's experts are
smaller: 13.36 MiB on layers 0-1 and 15-19, 11.07 on 2-4, 12.22 on 5-14 and 20-24 - so GPU 0, which holds layers 0-19 and
none of 14.50, left 8-24% of every slot empty. On the decode trace of §20 (1,538 windows, 374,400 routings), the best fixed
set in hindsight in the same VRAM: uniform slots 945 + 808 experts, 71.9% of the routings; GPU 0's slots at 13.36 MiB,
1,025 + 808, 73.0%; every slot its expert's size, 1,100 + 845, 74.6%.

**`STRATA_DSV41_SLOT_SIZES=1`** (opt-in, on in strata-dsv41.json): each slot the size of the expert the plan puts in it - the
profile's order, GPU by GPU, while the next expert fits what the GPU has left above its reserve. A swap stays within its
layer and a lent slot gets its own expert back, so a slot never needs another size; a first-use fill takes a slot that holds
it. Everything finds a slot through `slot_ptr` (its offset); the decode's residency table holds the offsets in 256-byte units,
so the grouped kernels (base + index x stride, in 64 bits) are unchanged; the prompt path lends the fewest top slots whose
bytes hold its loan.

- **Exact** (`checks/serve_slot_sizes.py`, the route bias off, `--no-adaptive`): the same 945 + 755 experts in uniform and
  in sized slots (`STRATA_GLM_STAGE_SLOTS=945,755`, a cap for checks) - the first token's logits (af3abf58f2b2a136) and the
  64 greedy tokens the same. Uncapped, the same VRAM holds 1,102 + 844 (+191, +10.9%).
- **Decode** (`checks/serve_decode_ab.py`, the server's decode: the bias, adaptive swaps, the slots that fit -
  `DECODE_SLOTS=-1` - four arms one after another), passes 3-6:

| arm | slots | tok/s | in VRAM | the CPU pool's own work a token (its time less the drive's) |
|---|---|---|---|---|
| uniform | 949 + 814 | 19.30 | 68.8% | 18.2 ms |
| sized | 1,107 + 847 | 20.23 | 71.5% | 16.8 ms |
| uniform | 949 + 815 | 18.86 | 68.5% | 18.5 ms |
| sized | 1,107 + 848 | 18.76 | 69.8% | 17.9 ms |

  +2 points in VRAM and ~1 ms less of the pool's work a token in both pairs; tok/s 19.08 -> 19.50 on average (+2.2%), inside
  this test's noise (two runs of one arm differ by 3%): the drive's waits move more from pass to pass than the pool's work.
- The server (`checks/serve_smoke.py`): 1,102 + 844 slots, GPU 0 707 MiB and GPU 1 2,057 MiB left free (as before), ready
  41 s, both replies streamed.

## 25. GPU 0's prompt ring by its own layers' size: layer-major windows of 3 at 512K (2026-10-22)

A layer-major window keeps a whole layer's streamed experts in its ring (384 entries), each the largest blob of all 40
layers - 14.50 MiB on GPU 0 too, whose layers hold at most 13.36: 0.43 GiB of GPU 0's 11.6 GiB loan, about what a window of 3
lacked at the server's context (an 84K read: 9,438-token chunks, 11.01 GiB in windows of 2, ~0.8 GiB more a chunk).
**`STRATA_DSV41_STAGE_RING=1`** (opt-in, on in strata-dsv41.json) sizes a stage's ring entries by its own layers' largest
blob (`ring_blob`). `checks/serve_lm_windows.py` (84,414 tokens of this repo's docs read from 0 at 512K context, the slots
that fit, `--no-adaptive`, the 30 GiB test cache, no slow-mode episode in any arm):

| arm | the plan (GPU 0's loan) | the read | from the drive | first token's logits |
|---|---|---|---|---|
| the config | 9 chunks of up to 9,438, windows of 2 (11.01 of 11.63 GiB) | 156.8 s (161.0 in a second run) | 373.2 GB | cb8c5aeb815abeb6 |
| `STAGE_RING=1` | the same chunks, windows of 3 (11.42 of 11.63 GiB) | **140.4 s** (x1.12) | **279.8 GB** (-25%) | cb8c5aeb815abeb6 |
| ... and `SLOT_SIZES=1` | windows of 3 (11.42 of 11.89 GiB) | 140.4 s | 277.2 GB | (other experts in VRAM) |

Bit for bit in windows of 3, as layer-major is (every chunk computes a layer as it would alone). The first token comes out of
a decode window - its VRAM experts on the GPU, the rest on the CPU pool - so its logits follow the VRAM tier's set: the run
with `--expert-slots 1700` (7a165911a5d39ea0) differs from the uncapped one, and so does any layout that holds other
experts. That capped run also showed what a GPU 1 short of loan does: with 593 sized slots left there (GPU 0 took 1,107 of
the 1,700) the plan fell to 15 chunks of 6,144 in windows of 4, which reads layer 20 once a chunk - 369.8 GB against 387.5,
246.1 s against 256.2 (one slow-mode episode in each); the server caps nothing.

Two checks of the new layout around layer-major. `checks/serve_layer_major.py` with `STAGE_RING=1 SLOT_SIZES=1` (64K
context, windows of 4): the 44K read from 0, the resumes at the root and at 16384, and a stop inside a window followed by
the read again - all equal to the chunk loop's, bit for bit (9861c987962a7c1c; STOP -> DONE 4.8 s). (Its own switches leave
out §18's engram ones, and its first layer-major read stood ~260 s at layer 0 of the first window with the tables evicted.)
`checks/serve_cold_start.py` (§18's scenario through the whole server as configured: the page cache emptied of the engram
tables and the dense weights, an 84.5K request, another program's 4 KB reads timed): the first token 384.1 s after the
request, with three slow-mode episodes taking 217 s of it; progress lines at most 35.6 s apart (the server's 30-s "still
reading" line, late by a few seconds, inside the episodes - the check asks for 30.5, so it fails there); another program's
reads: median 3.39 ms, none over 1 s (912 / 993 ms at two episodes' onsets, the queued requests draining); at the guard's
floor median 2.91 ms, p90 3.98, p99 80.5 (the check asks for under 10). The p99 is the third episode's: the drive began it at
176 reads a second, the engine had one read on it 90% of the time (the in-flight count), and that one 128 KB call took
40-100 ms - smaller calls at the floor would shorten it, at 4 MB/s instead of 63 (§18): not taken.

## 26. Past 262,144 positions: a 300,000-token read (2026-10-22)

No read had gone past 262K since the 512K context of 2026-10-20 (the server's log of the user's runs: 177,736 tokens at
most), and dsh compacts at ~400K. `checks/serve_long_context.py`: a passphrase in the text's first line, this repo's docs and
sources after it, a question for the passphrase at the end - 300,000 tokens read from 0 with the config's arguments (512K,
the 60 GiB cache) and switches as of §23, then 16 greedy tokens.

- The reply: "amber-walrus-7193" (its first token at -0.02, 98%) - recall across the whole read; no error; the first
  token's logits finite.
- The read: 1,236.8 s - 32 chunks of up to 9,464 tokens in windows of 2 (GPU 0's loan 11.03 of 11.57 GiB), 1,096.5 GB from
  the drive at 0.88 GB/s. Ten slow-mode episodes of 59-79 s (~706 s, 57% of the read; the guard at its floor through each),
  and 817.6 s of the chunks' waits for their engram rows, most of it inside them. Away from the episodes the drive gave
  ~2 GB/s.

So a long read's time is now mostly the drive's slow mode, and its episodes come with the bytes read: windows of 3 (§25)
read a quarter fewer bytes (not yet measured at 300K).

## 27. What the user's decode waits for, and the read-ahead several layers ahead (2026-10-22)

The server's log of the user's own replies (its statistics lines only: 601 replies of 100 tokens or more, 434,145 tokens):
16.0 tok/s at the median (p10 13.6, p90 19.0; 15.3 by tokens). A token, weighted by tokens: 22.1 ms waiting for the routing
and 40.1 ms in the CPU pool, **18.0 ms of it for the drive** (2.9 SSD reads a token; 61.7 from the SSD cache, 14.3 from the
RAM copy), 67% of the routings in VRAM. The benchmark's decode (§20, §24) waits 7-10 ms for 1.5-2 reads: the user's varied
text misses the caches more, and the drive is ~30% of the user's token.

**`STRATA_DSV41_AHEAD=d`** (opt-in, off; 1-8): the read-ahead's guess for layer il + d from layer il's rows (with the bias's
penalty, §23). `checks/serve_decode_ab.py` (the server's decode, four passes), passes 2-4:

| d | records read ahead a token | used | the demand's SSD reads it guessed | the drive a token | tok/s |
|---|---|---|---|---|---|
| 1 (§23) | 1.39-2.02 | 87-90% | ~62% | 8.2-11.1 ms | 17.5-18.8 |
| 2 | 1.46-1.74 | 83-93% | ~56% | 8.4-9.4 ms | 17.9-18.9 |
| 3 | 1.32-1.81 | 78-85% | ~51% | 7.4-9.2 ms | 18.4-18.8 |
| 4 | 1.51-2.26 | 70-86% | ~48% | 8.1-11.7 ms | 17.5-19.3 |

The drive's wait does not fall: half the demand's reads are never guessed, a read ahead runs on at most 2 of the 12 readers
(below the demand), and the guess costs 1.5-2 ms of GPU time before the doorbell. Off. What it would take is in DSV41_BACKLOG
#86 (the guess after the doorbell, while the CPU pool runs; the reads ahead on more readers; ~25% of them wasted).

## 28. Windows of 4 for long reads: a layer-major read lends all but 32 slots (2026-10-22)

A window of 4 lacked ~0.8 GiB of GPU 0's loan, and a prompt lent all but 128 of a GPU's slots. **`STRATA_DSV41_LM_KEEP=N`**
(opt-in, 32 in strata-dsv41.json): a layer-major read lends all but N (`Stage::lendable_lm`; the chunks are still planned
by the usual lend, so they do not change). The 96 more experts it lends are the hottest; `STRATA_DSV41_PROMPT_KEEP` copies
them into the SSD cache first, so the read streams them from RAM and the refill takes them back from there. A short read is
no layer-major read and lends as before. `checks/serve_lm_keep.py` (~120,000 tokens of this repo's docs and sources at 512K,
the slots that fit, `--no-adaptive`, the 30 GiB test cache):

| arm | the plan (GPU 0's loan) | the read | from the drive |
|---|---|---|---|
| the config | 13 chunks of up to 9,363, windows of 3 (11.37 of 11.89 GiB) | 395.0 s | 399.5 GB |
| `LM_KEEP=32` | the same chunks, windows of 4 (12.20 of 13.06 GiB) | 385.3 s | **351.9 GB** (-11.9%) |

The first token's logits bit for bit (31b226bd73ebf23c). Three slow-mode episodes in each arm, so the times are no
comparison; the bytes are: 5 windows -> 4 (the last of one chunk), the later windows' reads partly in the SSD cache. The
fewer bytes a long read takes, the fewer episodes it meets (§26). A read of 9 chunks or fewer is 3 windows either way.

## 29. The CPU pool's phases in smaller tasks, and two of the cold-start check's misses (2026-10-22)

**Where decode's CPU time goes.** The pool's kernels for this pack's formats, timed on its own records
(`native_expert_cpu_bench` with `STRATA_BENCH_RECORDS`, 5120 x 2304, one token):

| gate/up / down | the kernel, 1 thread | a plain read, 1 thread | the kernel, 20 threads | a plain read, 20 threads |
|---|---|---|---|---|
| Q3_K / Q3_K | 5.62 GB/s | 13.88 | 72.20 | 73.89 (0.98) |
| Q2_K / Q3_K | 6.26 | 14.21 | 72.29 | 74.58 (0.97) |
| Q2_K / MXFP4 | 6.42 | 14.46 | 72.67 | 74.94 (0.97) |
| Q2_K / Q2_K | 6.80 | 14.93 | 72.94 | 74.82 (0.97) |

A core is compute-bound (0.40-0.46 of its read), but 20 threads on whole experts take 97-98% of the DRAM's bandwidth: a
faster kernel (AVX-512 for the K-quants) has nothing to take. The serve decode line now says what the pool's time is
(`decode CPU pool:`): 0.84-1.06 GB a token over 14.4-18.3 ms of its drains, **58 GB/s** - a layer's ~2 experts split into 63
tasks (three for each of 20 workers and the host) a phase, two phases a layer, and a phase's tail left to a few threads.

**`STRATA_POOL_TASKS=N`** (opt-in, 252 in strata-dsv41.json): the pool's phases in N tasks. `checks/serve_decode_ab.py` (the
server's decode, four passes), passes 2-4, the pool's rate over its drains: 63 tasks 58.0-58.4 GB/s, 126 61.9-62.0, **252
61.9-62.1**, 504 61.2-61.3 - **+6.7%**, ~1.1 ms of a benchmark token's ~47 and ~1.4 of the user's ~62 (§27). Tok/s in those
arms (18.8-23.3) moved more with the drive's waits than with the pool. Every row is one thread's whole dot product, so the
split changes no result: `pool_tasks_test` (now with the model's geometry, `N_EMBD N_FF`) on records of each of the pack's
four formats - task counts 0, 1, 41, 128, 192, 252, 256 and 4,096, 1 and 3 workers, the host working or not - 192 bitwise
comparisons each, all equal; the decode A/B's first token's logits the same (8611aae60ecaffc8; its greedy tokens part later,
as two runs of one arm can, §14).

**The faster VRAM kernel again** (`STRATA_MMVQ_GROUPED_ROWS=8`, §20) with the sized slots (17% more experts on GPU 0, whose
layers wait for their VRAM experts, #55): passes 3-6, two runs an arm - 19.30 / 20.29 tok/s with one row a block, 19.42 /
20.14 with eight (-0.1%); 0.4-0.7 ms less waiting for the routing. Off.

**The cold-start check's two misses (§25).**
- Progress lines 35.6 s apart at most, against 30: the server's "still reading" line waited 30 s, but it is asked only on
  the engine's heartbeats, 10 s apart while the engine is quiet. It now waits 20 s (serve/server.py `_progress`): at most
  30 s apart.
- "At the guard's floor, another program's p99 under 10 ms" (80.5 ms): the probe's reads with nothing else on the drive -
  132 of them in the third episode, from the 2 ms in-flight samples - took up to 100 ms themselves (median 1.65 ms), against
  2.58 ms beside the engine's one read. The drive, not the queue: the check now asks that the engine's reads add under 3 ms
  to another program's median read at the floor (they added ~1.6 ms in that run), and falls back to the p90 under 10 ms where
  it met too few lone reads.

The RAM copy as more LRU slots (#88) has its answer in the 2026-10-19 measurement (§12's "The SSD cache's size"): at the same
total memory a static copy of the profile's hottest served 10.0 SSD reads a token, LRU slots in its place 11.2.

**The cold start with the config as it now is** (`checks/serve_cold_start.py`, the binary of 68745ad1: windows of 4, the pool's
252 tasks, the 20 s progress line): the server ready 40 s after its start; the first token 250.7 s after the 84,400-token
request (9 chunks of up to 9,455 in windows of 4, GPU 0's loan 12.27 of 13.00 GiB), with one slow-mode episode (71.3 s) - the
run of §25 took 384.1 s with three; progress lines at most 24.2 s apart; another program's reads none over 1 s (651 ms at
the episode's onset); at the guard's floor 1.76 ms median with nothing else on the drive and 1.82 ms beside the engine's
reads. All four criteria pass.

**The user's own reads before these changes** (the server log's statistics lines, 830 requests): 1-100 tokens read 4.2 s at
the median (p90 8.8), 100-500 13.3 s (20.2), 500-2,000 26.0 s (33.2), 2,000-8,000 38.9 s (104.0), 8,000-32,000 81.2 s
(142.0), 32,000-100,000 276.1 s (399.6), 100,000-400,000 778.4 s (1,063.6); the largest, 140K-166K tokens read from cold, in
778-1,093 s, in the chunk loop (#84). Reads under 2,000 tokens are 744 of the 830, and the drive bounds them (each chunk
reads every expert not resident once).

## 30. The bounded replay's tails in one run (2026-10-22)

A long read runs layers 21-39 only over the last 2,560 rows before each read start - a checkpoint cut, the call's end - and
each of those tails streamed layers 21-39's ~7,000 non-resident records: an 84K read with its cut at 65,536 streamed them
twice (14,096 of its 35,257 records), a 177K conversation (the root, 65,536, 131,072, the end) four times. Layers 21-39 keep
only their 128-row raw rings and read layer 20's compressed rows and index keys, which a cold read never rewrites, so a tail
run later sees what it saw at the time, and its end state does not depend on the rings it starts from (128 x 20 rows).

**`STRATA_DSV41_TAIL_BATCH=1`** (opt-in, on in strata-dsv41.json): the last stage keeps each tail's rows (replay_shift's list:
the streams, the carried pre-mix, layer 20's top-k lists and candidate masks) in its loan after its chunk's layer 20, and at
each cut a copy of the running state. After the call's last chunk it carves the loan again (the longest piece, the chunks'
own sub-batch, a ring of a whole layer), streams each of layers 21-39 once, and runs every piece through it in position
order (`Stream::hold` until the last); a cut's copy takes each layer as the cut's last piece finishes it, and its checkpoint
is taken from the copy. A tail across a chunk end is two pieces, run in order. The loan reserves each piece's own rows and
one state copy a cut, and without the room the tails run as before (GPU 1 capped at 625 slots in a 64K check had 6.7 GiB to
lend against the run's ~6.8). A call stopped among its chunks runs the tails it has to the end first, so the cuts it passed
keep their checkpoints as before; its live state goes (the next read starts at a checkpoint). That stop takes the tails'
run: STOP -> DONE 25.2 s in the check below, 4.8 s without (§17).

- `checks/serve_tail_batch.py` (84,414 tokens at 512K, the 60 GiB cache, `--no-adaptive`): the read's first-token logits
  the same (bea8e6d14ef4a5c5); a second request resumed at the checkpoint at 65,536 with the same logits (ab145ab4ea6c95bc)
  - the checkpoint put together from the tails' run is the one the per-chunk tails made. **318.9 -> 270.8 GB** from the drive
  (-15.1%), the tails' run 30.1 s (2 pieces); 437.1 / 341.5 s with 4 / 2 slow-mode episodes (no time comparison).
- `checks/serve_layer_major.py` with the tails batched (64K context, the slots that fit: `LM_SLOTS=-1`; 4 pieces in
  26.3-27.2 s): the 44K read from 0 the chunk loop's and layer-major's bit for bit (127138a9e52694d1), the resumes at the root
  (34859) and at 16384 equal, and a stop inside a window - the one tail it had run first, its checkpoint at 16384 kept - then
  the read again from there, equal to the read from 0.

Not taken: GPU 1's layer 20 in windows (#89's other half) - four 9,438-row chunks' rows (3.1 GB), a whole-layer ring (5.6 GB)
and the scratch (~4.5 GB) against GPU 1's ~9.9 GiB loan; the chunks' MoE rows run together instead (a rolling ring) would
change MMQ's launch shapes, so its bits.

**The SSD cache's size again** (the 30 GiB of the test runs against the server's 60, the RAM copy taking the rest:
`serve_decode_ab.py DECODE_CACHE=`, the server's decode, two runs each, passes 2-4): 60 GiB 21.35 / 21.99 tok/s, 30 GiB 21.03 /
20.60 - the LRU learns this test's repeated text (0.73-1.07 SSD reads a token by pass 4 against a flat 1.6-2.0). An 84K read
took 22,440 SSD reads with 60 GiB, 20,514 with 30 (the larger static copy). Neither settles it for the user's varied text;
60 GiB stays, the user's setting.

## 31. Where it stands after 2026-10-22, and what is left

All of it with the binary of ae6ef41c and strata-dsv41.json as committed (`--ssd-cache-gib 60`, 512K context).

| | before §19-§30 | after | the limit it meets |
|---|---|---|---|
| the server's start to "ready" | 71 s | 40 s | the load pass (42.5 GiB at the drive's 2.2 GiB/s) and the dense weights |
| an 84,414-token read, warm, no slow mode (30 GiB test cache) | 235.6 s | 140.4 s (windows of 3; 4 with LM_KEEP) | the drive (~2.0-2.4 GB/s) and the ring's stalls |
| its drive bytes | 488 GB | 280 GB (windows of 3); a 120K read 399.5 -> 351.9 GB (windows of 4); the tails in one run -15% more (§30) | every chunk window reads every expert not resident |
| reads under 2,000 tokens (744 of the user's 830) | 13.3 s median (100-500), 26.0 s (500-2,000) | the same | the drive: ~88% of its floor, the rest the routing's waits |
| decode, the benchmark | 19.08 tok/s | 19.50 (sized slots) and the pool's 252 tasks (+6.7% of its work) | the CPU pool (62 of the DRAM's 72 GB/s, whole experts 72-73) and the drive (7-10 ms a token) |
| decode, the user's replies (before) | 16.0 tok/s median | not measured since | 18 ms a token for the drive (2.9 reads) |

**What is left, by what it would take:**
- Long reads now wait on the drive's slow mode more than on anything the engine does (10 episodes in the 300K read, 57% of
  it); fewer bytes meet fewer episodes, and every exact way to read fewer is in (windows of 4, the tails in one run). GPU 1's
  layer 20 in windows does not fit its memory (#89); the tails' records read during GPU 0's ring stalls (#87) would hide at
  most the run's ~25-30 s, and the SSD cache cannot hold them beside the windows' kept records.
- Decode: the pool's remaining 14% is its two phases a layer (a start and a tail each; 252 tasks already cut the tail) -
  ~1.5% of a token to take by running each expert's down rows as soon as its own gate/up rows are done; the dense Q8_0
  projections at 82% of the bandwidth (#55) - ~1 ms a token (~2%) at 90%; the drive's 18 ms a token in the user's replies -
  the read-ahead several layers ahead guessed 48-62% of them at 70-93% precision and did not shorten the wait (#86); the RAM
  copy against LRU slots and the cache's size (#88, §30) - measured both ways, the user's routing would settle it.
- Short reads: the routing's waits (~1.2 s a read) are the drive's idle time; reading the next layer before its routing
  (GATE_AHEAD, #31) reads records the chunk does not route (~19% at 2,000 tokens) - more of the reads the slow mode comes
  with, for 5-9% of a short read.

## 32. The read-ahead off the GPU's path, with more readers: still no gain (2026-10-22)

§27's read-ahead lost on two counts: its guess ran before the routing's doorbell (1.5-2 ms of GPU time a token), and its
reads ran on at most 2 of the 12 readers. Both are changed now (opt-in, off): the decode graphs compute the guess after the
doorbell and publish it as the doorbell publishes the routing - the ids, then a count of their own (`doorbell_publish` on
`apred_ids`), which the host waits for only where it queues the reads ahead (at most 2 ms; a late guess reads none ahead,
and an aborted window takes the device's count back, as the doorbell's); `STRATA_DSV41_AHEAD_READERS=N` gives the reads ahead
N readers. `checks/serve_decode_ab.py` (the server's decode, the 30 GiB test cache - its 1.6-2.0 SSD reads a token nearer
the user's 2.9 than the 60 GiB cache's), passes 2-4:

| arm | tok/s | the drive a token | waiting for the routing |
|---|---|---|---|
| off | 17.93 / 20.14 / 21.39 | 14.4 / 11.0 / 9.2 ms | 18.8-19.9 ms |
| `AHEAD=2`, 6 readers | 20.55 / 20.42 / 19.84 | 7.5 / 8.9 / 10.2 ms (1.18-1.88 records read ahead, 84-89% used) | 19.4-19.6 ms |
| off | 21.47 / 21.15 / 19.78 | 7.1 / 9.0 / 10.9 ms | 19.3-19.8 ms |
| `AHEAD=3`, 6 readers | 18.88 / 19.31 / 21.39 | 10.6 / 11.0 / 7.6 ms (1.36-2.03 read ahead, 75-84% used) | 19.0-20.0 ms |

The guess no longer lengthens the wait for the routing; the drive's wait a token falls ~1 ms on the average (10.3 -> 9.3),
inside the passes' spread, and tok/s does not move (20.31 off, 20.07 on). Half the demand's reads are never guessed
(§27), and the others are no shorter. Off. The decode's drive waits have no more exact lever here than the caches' sizes
(§30) and their policy (#88).

## 33. A long read's bytes: the MoE's down rows over its gate/up rows, windows of 6, GPU 1's layer 20 read once (2026-10-22)

Three opt-in switches, on in strata-dsv41.json, each exact (the same arithmetic on the same values: only where the bytes
lie or how often they are read changes):

- **`STRATA_DSV41_DM_ALIAS=1`**: a chunk's MoE held its gate/up products (rows x 2 x 2304 floats) and its down products
  (rows x 5120) apart, because a down row is longer than a gate/up row here (MiMo, DSV4 and GLM put the down rows over the
  gate/up rows). gu's rows are now 5120 floats apart, so row r's down output lands in gu's row r - consumed (its group's
  SwiGLU quantized) before its group's down product writes there, and past every earlier group's rows: 1.04 GB less a
  9,439-row chunk, on both GPUs (the in-place SwiGLU and the quantizer take the row stride).
- **`STRATA_DSV41_LM_CHUNKS=8`**: layer-major windows as large as GPU 0's loan holds (the cap was 4). With the alias an 84K
  read's 9 chunks go in windows of 6 (GPU 0's loan 12.95 of 13.00 GiB): 2 windows, not 3, each streaming GPU 0's layers once.
- **`STRATA_DSV41_RING_KEEP=1`**: with the tails deferred (§30) GPU 1 runs only layer 20 for each chunk, and its ring held
  208 entries (55% of a layer), so every chunk streamed layer 20 again. A call that defers its tails now gets a ring of the
  whole layer at layer 20's blob (384 x 12.22 MiB; GPU 1's loan 8.79 of 9.91 GiB, else the old ring), its records stay
  there (`Stream::hold`), and each later chunk reads only the routed records the ring lacks (the streams are gated: each
  chunk reads its routed experts only) - a group's MoE then waits for each of its slots' copies, which landed at different
  times. The tails' run carves its own ring afterwards, as before.

`checks/serve_dm_alias.py` (84,414 tokens of this repo's docs at 512K, the 60 GiB cache, `--no-adaptive`; then a second
request resuming at the checkpoint at 65,536):

| arm | the plan | from the drive | records read / from the cache | the read (slow-mode episodes) | first-token logits (read / resume) |
|---|---|---|---|---|---|
| the config | windows of 4 (3) | 277.4 GB | 19,584 / 9,710 | 294.9 s (2) | bea8e6d14ef4a5c5 / ab145ab4ea6c95bc |
| `DM_ALIAS=1 LM_CHUNKS=8` | windows of 6 (2) | 240.6 GB | 16,716 / 5,817 | 297.3 s (2) | the same / the same |
| ... and `RING_KEEP=1` | windows of 6 (2) | **217.5 GB** (-21.6%) | **14,705** / 5,703 | 215.8 s (1) | the same / the same |

Every run of the day met the drive's slow mode at its read's start (DSV41_BACKLOG #83), so the times say little: the bytes do, and the
fewer a read takes, the fewer episodes it meets (§26). The tails' run took 30.3-35.3 s in all three.

## 34. A decode token's GPU part, kernel by kernel, and the CPU pool's rate (2026-10-22)

**Where the GPU's part goes** (`nsys --cuda-graph-trace=node`, 64 tokens at the config's switches with `--no-adaptive`):
a layer's work from the CPU's rows to the next routing doorbell is ~380 us of kernels in a row - the launch gaps between
them 0.11-0.2 ms a token in all (median 64 ns), so it is the kernels' own time. A non-engram layer on GPU 0:

| kernels | us | what bounds them |
|---|---|---|
| the combine of the CPU's rows (mapped) | 10.9-17.1 | PCIe |
| the mixers' read, twice (the streams' norm, the split gemv and its reduce, Sinkhorn, the pre-mix, the carried pre's copy, the norm) | 2 x ~24 | each small kernel's latency |
| q_a / q_b / wkv | 12.0 / 65.5 / 5.9 | q_b's 44.6 MB at ~680 GB/s (q_lora is 1,280; corrected 2026-10-22, §44) |
| the attention (MMA + merge), its ropes and FP8 roundings | ~27 | latency |
| wo_a: eight 1024-row groups | 8 x 8.4 | 525 GB/s in eight launches |
| wo_b | 63.9 | 44.6 MB at ~700 GB/s |
| the router's gemv / the routing / the doorbell | 16.3 / 17.7 / 7.3 | the routing read the route bias's table over PCIe |

After the doorbell the shared expert (61 us) and the VRAM experts run beside the CPU pool, off that path. The head reads
its 703 MB at ~719 GB/s, the engram layers' 25,600-row projection 167 MB at ~714 GB/s: at the bandwidth.

**Four opt-in switches, on in strata-dsv41.json**, each the same arithmetic on the same values:
- `STRATA_DSV41_PEN_DEVICE=1`: the route bias's table in each GPU's VRAM, pulled from the mapped table by the decode graph's
  first node (the host writes it only between windows). A copy-engine refresh instead queued behind the tier's 13 MiB swap
  copies and held the next window up - the routing wait rose 0.7 ms a token in the A/B, so the pull is a kernel.
- `STRATA_DSV41_WOA_BATCH=1`: wo_a's eight groups in one launch of the same kernel (`native_q8_0_mmvq_batched`, a copy of
  `native_small_mmvq_kernel` with blockIdx.y choosing the group): 68 -> 54 us a layer.
- `STRATA_DSV41_PRELOAD=1`: the routing, the combine and the 1024-thread norms issue their loads first, then the same
  expressions (the routing's 12 x 3 loads a lane waited one after another): the routing 18 -> 7.9 us.
- `STRATA_DSV41_MIX_FUSED=1`: a one-token window's mixer read after its split gemv's partials - the reduce, the Sinkhorn
  coefficients, the pre-mix with the carried coefficients, their copy, the weighted norm - in one launch
  (`mhc_mix_in_w1`, each step the kernel it replaces line for line): five launches of ~12 us -> one of 8.5.

`preload_parity` (new, moex/tools/dsv41: the same fixed-seed inputs at V4.1's shapes through each kernel, the default and the
opt-in form, 300 draws, 43 MB of outputs) - byte for byte equal for every form (the routing with and without the bias's
table, the combine over random masks, both norms in place and not, the whole mixer read). `checks/serve_decode_ab.py` with
the bias off and one tier (`--no-adaptive --expert-slots 1700`): the first token's logits 8611aae60ecaffc8 in every arm (the
value of §29's run), and with the first two switches all 2 x 256 greedy tokens the default's.

The GPU's part in the profile: the wait for the routing 16.5 -> 15.0 ms a token, 1,282 -> 983 kernels a token on GPU 0.
The server's decode (`serve_decode_ab.py`, the bias and the adaptive tier, the 30 GiB test cache; passes 2-4; the first
three switches, in a run where both arms read the same records - the same drive statistics pass for pass):

| arm | waiting for the routing | tok/s |
|---|---|---|
| off | 19.4 / 19.5 / 19.7 ms | 20.84 / 21.12 / 21.17 |
| on | 18.5 / 18.7 / 18.6 ms | 21.19 / 21.51 / 21.64 |
| on, again | 18.4 / 18.7 / 18.7 ms | 21.23 / 21.47 / 21.54 |

-0.93 ms a token and +1.9%, with the fused mixer read's ~0.3 ms more on top (the profile). Left on the path: q_b at ~680
GB/s (corrected, §44: near the bandwidth - a packed copy would gain at most ~0.14 ms a token), the combine's PCIe
reads of the CPU's rows, the router's f32 gemv (its sum order pins its threads), the Sinkhorn chain.

**The CPU pool's 62 GB/s.** `pool_phase_bench` (new: the pool as decode drives it - a layer's experts for one token through
`run_split_multi_native`, a 500 us gap between layers, the pack's records in 4 KB pages, 20 workers and the host):

| the experts' memory | 4 KB pages | 2 MiB pages (`MADV_HUGEPAGE`, all of it backed) |
|---|---|---|
| 1.6 GB (128 experts) | 70.4-70.5 GB/s | 62.4 |
| 8 GB (640) | 69.9 | 61.8 |
| 32.5 GB (2,600) | 64.2-64.8 | 60.5-61.1 |

So the engine's 62 GB/s is its experts' spread - the RAM copy and the SSD cache are ~80 GB of 4 KB pages, whose page-table
entries do not fit the caches - not the pool's phases: in the bench 2 experts a layer run at 70 GB/s, 1 at 67, 4 at 72,
and a split of the rows in shrinking ranges (opt-in, written and measured) moved nothing (+-0.3%). Huge pages are slower
on this box (2 MiB pages 61.8 GB/s against 4 KB 69.9 at 8 GB), and a software prefetch 4-16 KB ahead in the rows lost
3-7% (60.0-62.1 against 64.7 GB/s): both dropped, neither in the code.

## 35. A decode token's time outside the routing and the pool: the VRAM tier's upkeep (2026-10-22)

The serve decode line now splits a token's other time (`a token's other X ms: the head, the VRAM tier's upkeep, the rest`)
and the upkeep (`maintain_cache`) into its parts. In `serve_decode_ab.py` (the server's decode, four passes) a token spent
3.8-4.2 / 2.4-2.6 / 1.5-1.7 / 1.2-1.5 ms in the upkeep, nearly all of it the host hashing a record the tier swapped into
VRAM for the first time - 1.5 / 1.2 / 0.6 / 0.5 such records a token, ~2.0 ms each on one core (the slot check's reference
hashes, `slices_hash_host` over its 14.5 MiB); planning the swap rounds took ~0.1 ms a token. The user's varied text keeps
bringing such records in, so the cost does not fade as it does on this test's repeated text.

Two opt-in switches, on in strata-dsv41.json, each giving the same hashes (a sum of products modulo 2^64, whatever the split):
- `STRATA_DSV41_HASH_POOL=1`: those hashes summed over 16 word ranges a slice on the CPU pool's threads - idle between
  windows (`ExpertPool::run_tasks`, a batch of plain tasks): ~2.0 -> ~0.23 ms a record. The first record also the one-thread
  way, printed: "equal to the one-thread values" in every run.
- `STRATA_DSV41_HASH_WIDE=1`: the slot check of a few swapped-in slots in 32 parts a slice (`slices_hash_parts`; 3 blocks a
  slot ran ~1.9 ms): 0.47 -> 0.30 ms a token. `preload_parity` compares its sums with `slices_hash` and the host's, PASS.

| a token (passes 1..4) | the upkeep | hashing first-time records | the other time in all |
|---|---|---|---|
| off (two arms) | 3.9-4.0 / 2.6 / 1.6 / 1.5 ms | 3.03-3.16 / 1.85-1.86 / 0.91-0.93 / 0.76-0.81 | 6.4-6.6 / 3.9 / 2.8-2.9 / 2.7-2.8 |
| on (two arms) | 1.0 / 0.8 / 0.7 / 0.6 ms | 0.36 / 0.25 / 0.12 / 0.11 | 3.7-3.8 / 2.1 / 2.0 / 1.9-2.0 |

-0.85 to -2.7 ms a token, the more the tier swaps. No slot check reported a difference from the file. What is left of the
other time, ~1.2 ms a token, is not split further: the window's inputs (its engram rows 0.4-0.5 ms by the decode line), the
head and the sampling, the reply's lines.

## 36. The f32 gemvs' loads first; the slow mode's onsets; the pool's caches (2026-10-22)

**`STRATA_DSV41_PRELOAD=1` covers the one-column f32 gemvs now** (the routers' 5,120 x 384, the compressor's and the indexer's,
and the mixers' split gemv): each thread's weights and inputs loaded first, the same multiply-adds in the same order, the one
warp sum a column has (the kernels ran eight, one for each column they could hold), the same block sum. `preload_parity`
(the three gemv shapes added): byte for byte. The profile (the config's switches): the routers' gemv 18.6 / 21.7 -> 14.4 /
14.7 us (GPU 0 / 1), the wait for the routing 15.0 -> 14.8 ms a token.

**The slow mode's onsets (DSV41_BACKLOG #83).** Every 84K read of the day met an episode within seconds of its start,
while each chunk's engram rows were fetched cold (random 4 KB reads over the 196 GB tables) beside the record stream.
`checks/serve_eg_warm.py` read the text twice in one engine, the second time behind another first token (nothing reused):
the first read 281.3 s with 2 episodes, the second 463.5 s with 4 - back to back, through the whole read. The second read's
engram rows were not warm either (it waited 153.1 s for them against 89.4: an 84K read touches ~16 GB of engram pages, more
than the page cache keeps beside the engine), so the test does not separate the fetches from the reading as a whole; what
it shows is that the episodes come with sustained reading on this drive now (6 in ~12 minutes of it), wherever a read is.
The drive takes at most 128 KB a request (`max_hw_sectors_kb`), and in an episode it serves ~1,000 requests a second
whatever their depth (§18) - ~128 MB/s for the records, ~4 MB/s for 4 KB pages. The engine's part stays the guard's: the
desktop keeps its reads. No change.

**The pool's caches** (`pool_phase_bench BENCH_HOT=1`, the 32.5 GB set): a layer's drain run twice, the second timed - the
best a read of the next layer's experts into the caches during the GPU's part of a layer could leave - is 21% faster at
2 experts a layer (399 -> 317 us) and 28% slower at 1 (200 -> 256 us: its rows then sit in other cores' L2s; this CPU's L3
is a victim cache of 27.5 MB). A guess one layer ahead would do worse than that: not taken.

**The day's decode, measured whole** (`checks/serve_decode_ab.py`: the server's decode with the bias and the adaptive tier,
the 30 GiB test cache, four passes; "off" sets the six decode switches of §34-§36 to 0, "on" is strata-dsv41.json; two arms
each, interleaved):

| passes 2-4 | tok/s | a token | waiting for the routing | the CPU pool (the drive) | the rest |
|---|---|---|---|---|---|
| off | 20.09 / 21.62 / 20.68 | 49.6 / 46.1 / 48.4 ms | 19.2 / 20.0 / 20.1 | 26.5 / 23.2 / 25.8 (9.5 / 8.3 / 9.0) | 4.0 / 3.1 / 2.5 |
| on | 22.11 / 22.35 / 22.12 | 45.3 / 44.9 / 45.3 ms | 17.1 / 17.3 / 16.8 | 25.9 / 25.2 / 26.6 (8.8 / 8.6 / 9.8) | 2.2 / 2.2 / 1.8 |
| off | 20.68 / 21.47 / 21.33 | 48.4 / 46.5 / 46.9 ms | 19.2 / 19.4 / 19.9 | 25.4 / 24.2 / 24.0 (8.8 / 7.9 / 8.9) | 3.8 / 3.0 / 2.9 |
| on | 22.02 / 22.88 / 23.75 | 45.3 / 43.8 / 42.2 ms | 17.4 / 17.7 / 18.0 | 25.8 / 23.9 / 22.2 (8.8 / 8.2 / 7.0) | 2.2 / 2.1 / 1.9 |

20.98 -> 22.54 tok/s at the means (+7.4%), the pool and the drive the same in both: -2.2 ms a token waiting for the routing
and -1.1 ms between windows, all of it exact.

## 37. The tails' records read during the last window (DSV41_BACKLOG #87, 2026-10-22)

In an 84K read's last window the drive had room: 692 MB/s at that window's heartbeat (§33's run), its records mostly in the
SSD cache from the window before, while the tails' run after it streamed layers 21-39 from the drive for 30-35 s.
**`STRATA_DSV41_TAIL_AHEAD=1`** (opt-in, on in strata-dsv41.json): when GPU 0 starts the call's last window (its tails
deferred), a thread pins the tails' records the cache holds, then reads the others in the tails' order into SSD cache slots
at the read-ahead's class (`prompt_keep_ahead`, below the window's own reads) as slots come free - the last window keeps none
of its own reads (no later window reads them) and releases each layer's kept slots once that layer's MoE has run (an event
after its last chunk; every copy from them done). The lent experts' records stay for the refill. When the tails start, the
reads still queued move to the demand class, and the tails' stream finds the records in the cache.

`checks/serve_dm_alias.py` arm C with the switch (`EXTRA_ENV`): the read's and the resume's first-token logits the config's
bit for bit (bea8e6d14ef4a5c5 / ab145ab4ea6c95bc - with every switch of §33-§36 on); 2,356 records read ahead and 623 held
from the cache; the drive at 1,689 MB/s during the last window (692 without); the tails' run 24.2 s (30.3-35.3 without), the
last window and the tails 57 s against 70; the refill all from the cache (1,713, none read); 218.0 GB from the drive (217.5).
One run each, the drive in its slow mode five times in this one: the times are a lead, the bytes are the same.

## 38. q_b's rows two or four to a block: no gain (2026-10-22)

q_b (32,768 rows of 1,280 - not 1,024 as written here first: ~680 GB/s, near the bandwidth, §44) ran as one Q8_0 block a thread. A form
with 2 or 4 rows to a block, each row's blocks owned by the same threads in the same order (byte for byte in a parity test):
66.1 / 73.1 us against 65.5 - the block's cost was not what bound it; the 34-byte blocks' 2-byte loads are what the packed
layout (`STRATA_Q8_PACKED`, a second copy in VRAM) avoids. Dropped.

## 39. The KV row and the compressor beside the query: no clear gain (2026-10-22)

**`STRATA_DSV41_GRAPH_FORK=1`** (opt-in, off): in a captured decode graph the attention's KV row (its projection, norm, rope,
FP8 rounding, the ring's store) and, on the compressor layers, the compressor run on a second stream beside the query's
projections (q_a, q_b), forked after the input's quantization and joined before the indexer - the same kernels on the same
inputs. `serve_decode_ab.py` with the bias off and one tier: the first token's logits 8611aae60ecaffc8 in both arms and all
2 x 256 greedy tokens the default's. The server's decode (four arms interleaved, passes 2-4): waiting for the routing 17.3 /
17.8 / 17.9 and 17.2 / 17.8 / 17.9 ms off, 16.5 / 17.5 / 18.0 and 17.2 / 17.9 / 17.7 on - -0.18 ms, inside the arms' spread:
q_b's read takes the memory the small kernels beside it would use. Off.

(The same run's tok/s with every switch of §33-§37: 23.64 and 23.68 at passes 2-4's means, the drive 5.9-9.2 ms a token.)

## 40. The CPU pool's 62 GB/s: its threads' ranges side by side (2026-10-22)

§34 put the pool's 62 GB/s against the bench's 70.5 down to its experts' 4 KB pages over ~80 GB. Not so: the four 32 GB
DIMMs are two to each memory controller (`/sys/devices/system/edac/mc`), interleaved alike, and 2 MiB pages came out
slower (§34). What differs is where the threads read at once: a phase's tasks are consecutive row ranges, claimed in
order, so the 21 threads read ~21 neighbouring ranges of an expert at a time. **`STRATA_POOL_SPREAD=1`** (opt-in, on in
strata-dsv41.json): the k-th claim of a native multi phase takes range (k s) mod n - s coprime with the n ranges and about
n / threads - so the ranges read at once lie apart. Every row is still one thread's whole dot product: `pool_tasks_test`
(a spread pass added: 1, 3 and 20 workers, every task count) 576 bitwise comparisons on each of the pack's four formats.

`pool_phase_bench` (the pack's records, 4 KB pages, the host pinned, a 500 us gap between layers):

| a layer | the set | ranges in order | spread |
|---|---|---|---|
| 2 experts, Q2_K / Q3_K | 32.5 GB | 63.53 GB/s | **69.82** |
| ... on 2 MiB pages | 32.5 GB | 60.58 | 70.80 |
| 1 expert | 32.5 GB | 63.43 | 66.53 |
| 4 experts | 32.5 GB | 67.74 | 71.32 |
| 2 experts, Q3_K / Q3_K | 32 GB | 63.68 | 69.57 |
| 2 experts | 1.6 GB | 69.90 | 69.78 |

The server's decode (`serve_decode_ab.py`, four arms interleaved, the pool's rate over its drains, passes 2-4): 61.4-62.1
GB/s in order, **65.9-66.5** spread (+7%) - ~1 ms a token less in the pool at the same bytes. (Tok/s in that run followed
the drive: 8.0-13.5 ms a token of reads, the arms apart by more than the pool's gain.)

## 41. A short read's records in the SSD cache: free slots, and the ones its rows share (2026-10-22)

A read of one chunk (up to ~9,400 tokens; 744 of the user's 830 requests were under 2,000) streams every routed expert
outside VRAM and the RAM copy, from the SSD cache or the drive. With `STRATA_DSV41_PROMPT_KEEP_SHARE` at 0 (a share of 1 made
warmed reads slower: the prompt's insertions evicted records the same prompt needed a few layers on), the records the cache
lacked went through bounce slots and the cache kept none of them - after a start it stayed empty through the first request,
and what it held later was what the decodes had read.

**A conversation, traced and replayed.** `checks/serve_short_warm.py` (new) runs a conversation through the serve protocol at
the config's settings: a 3,011-token first turn of this repo's docs, then 8 follow-ups of 414 new tokens (other docs and a
question), 200 tokens generated a turn. `STRATA_ROUTE_TRACE` now records the prompt path's routing too (a chunk's layer
marked + 1000; `tier_sim.py` skips those), and `prompt_cache_sim.py` (new) replays a trace against the cache's policies with
the engine's tiers as `tier_sim.py` models them, the prompts' lend included. The replay of the config came within 2-6% of the
engine's own counts (each follow-up's drive reads, the decode's reads a token). On one recorded conversation:

| policy (the replay) | each follow-up's drive reads, turns 1-8 | the decode's drive reads a token |
|---|---|---|
| the config: a prompt's reads not kept, its hits not refreshed | 4,183 / 3,754 / 3,337 / 2,894 / 3,270 / 3,552 / 3,329 / 3,236 | 4.89 |
| a prompt's hits refresh (`PROMPT_LRU`, DSV41_BACKLOG #67) | 4,183 / 3,754 / 3,263 / 2,797 / 3,123 / 3,385 / 3,133 / 3,051 | 4.71 |
| a prompt's read takes a slot that holds nothing | 3,264 / 3,433 / 3,285 / 2,860 / 3,245 / 3,535 / 3,327 / 3,236 | 4.35 |
| both, and a read 1 / 26 of the chunk's rows route to takes the least recent slot | 3,286 / 3,274 / 2,945 / 2,557 / 2,940 / 3,219 / 3,020 / 2,956 | 3.95 |
| every prompt read kept, least recent evicted (a share of 1) | 4,841 - 5,059 | 8.61 |
| the profile's frequency choosing the victim (LFU-like; GreedyDual) | 3,081 - 3,732; 3,628 - 4,512 | 3.99; 3.99 |
| Belady's, knowing the future (a bound) | 1,713 - 2,433 | 1.32 |

The decode's recency beats the profile's frequency here, keeping every prompt read floods the cache, and the records many of
a chunk's rows route to are the ones its reply goes on to route. (A decode miss entered at the middle of the LRU order
instead of its end: no change, 3.62-3.71 reads a token.)

**Three opt-in switches, on in strata-dsv41.json**, for a short read's records only (a long read's share, order and victims
as before); where a record is read from changes no result:
- `STRATA_DSV41_CACHE_FREE=1`: a prompt's read takes a slot that holds nothing - nothing is evicted.
- `STRATA_DSV41_PROMPT_HOT=26`: a prompt's read that 1 / 26 or more of the chunk's rows route to takes the least recently
  used slot (`Stream::routed` now holds each expert's count, `prompt_keep_slot` takes it and the chunk's rows).
- `STRATA_DSV41_PROMPT_LRU=1` (#67, written 2026-10-20): a prompt's hit refreshes the record's place as a decode hit does.
The prompt-path line says what they kept ("the SSD cache kept N of its reads past its share: F in free slots, H routed by
1 / 26 of the rows or more").

The exactness check (`serve_short_warm.py NO_BIAS=1`: the decode then routes the same experts whatever the tiers hold; a
1,011-token first turn, 3 follow-ups of 414, 8 tokens each): every turn's first-token logits the same with the switches and
without (257b651756a66251, a9e802a593b6dce2, bbf5117ed1a76321, 8f4eed95f6ccaa17), and the tokens.
The 84K read and its resume with the three switches (`serve_dm_alias.py` arm C, `EXTRA_ENV`): bea8e6d14ef4a5c5 /
ab145ab4ea6c95bc, as before; 219.0 GB from the drive (217.5-218.0 in §33 and §37).

**The engine A/B** (`serve_short_warm.py`, the config's bias and adaptive tier, each arm a fresh engine, B with the switches;
the drive's slow mode met arm A once, in turns 5-6, which are left out):

| turn | the read A / B | its drive records A / B | the reply A / B | the reply's drive reads a token A / B |
|---|---|---|---|---|
| 0 (3,011 tokens) | 51.1 / 52.9 s | 9,279 / 9,280 | 9.7 / 10.9 tok/s | 12.01 / 8.56 |
| 1 | 22.5 / 19.5 s | 4,268 / 3,218 | 16.2 / 17.3 | 4.75 / 3.56 |
| 2 | 20.5 / 18.8 s | 3,944 / 3,231 | 19.7 / 20.8 | 3.05 / 2.39 |
| 3 | 19.0 / 17.6 s | 3,617 / 3,064 | 19.3 / 20.3 | 2.87 / 2.50 |
| 4 | 16.2 / 15.2 s | 2,975 / 2,523 | 23.6 / 23.3 | 1.74 / 1.62 |
| 7 | 18.4 / 18.4 s | 3,381 / 3,284 | 19.8 / 19.1 | 2.92 / 2.96 |
| 8 | 17.7 / 17.3 s | 3,215 / 3,010 | 20.2 / 21.6 | 2.73 / 2.09 |

Turns 1-4 and 7-8: the reads 114.3 -> 106.8 s (-6.6%) with 14.3% fewer drive records, the replies +3%; the first reply after
a start +12% (its drive reads a token 12.0 -> 8.6). The gain is the largest after a start and narrows as the LRU fills
(turns 7-8: 3-6% fewer records). The reads gain less than their records because the drive now waits for the GPUs more - no
read in flight while each layer's routing comes, 0.4-1.1 s a read in A, 1.4-2.8 s in B: a 414-token read streams ~7,400
records (~97 GB) through the two GPUs' rings, and a record's copy runs at 6.23 GB/s into GPU 0 and 8.51 GB/s into GPU 1
(pinned, 13 MiB copies, the same with both at once) - ~7.8 + 5.7 s of a read's 15-20 s, beside the drive's 15-19 s.

## 42. The decode's engram rows on a thread of their own (2026-10-22)

A decode window's engram rows (2 layers x 24 a token: the row numbers, 96 page requests, the copies) were fetched before its
graphs started - 0.45-0.54 ms a token on the host (the decode line's "reading its engram rows"), most of it the system calls
and the pages' faults, not the drive. **`STRATA_DSV41_EG_LATE=1`** (opt-in, off): `eg_rows` runs on a helper thread from the
window's start; the host waits for it just before the first engram layer's MoE (layer 1) is served, and that layer's graph
waits for the host's word (a doorbell wait kernel) before it reads the rows - the same function writing the same bytes. A
first form (the pages asked for before the start, the copies before layer 1) gained nothing: the system calls stayed on the
path (21.41 against 21.63 tok/s, passes 2-4 of two arms each).

`serve_decode_ab.py` (the bias off, one tier; it now prints where each arm's tokens first differ from arm 0's): the first
token's logits 8611aae60ecaffc8 and all 128 greedy tokens the same with and without (two runs of the first form, one of the
helper's); the helper's run spent 2.8 ms a token outside the routing, the pool and the head, against 3.5. Its server-mode
A/B was stopped before its end (the user, 2026-10-22: what is left is not worth chasing now): off (DSV41_BACKLOG #90).

## 43. Where it stands (2026-10-22)

| | before §19 | now |
|---|---|---|
| the server's start to "ready" | 71 s | 39-40 s |
| an 84K read's drive bytes | 488 GB | 219 GB (§33, §41) |
| an 84K read, warm, no slow-mode episode | 235.6 s | 140.4 s at 280 GB (§31); none clean since - every 84K run of the day met the slow mode |
| decode, `serve_decode_ab.py` (the server's decode) | 19.08 tok/s | 21.6-23.7 across the day's runs (the passes' drive reads 1.9-2.6 a token) |
| a 414-token follow-up in a conversation | 16.2-22.5 s | 15.2-19.5 s (§41) |

Every change of §19-§42 is exact: the deterministic check's logits (8611aae60ecaffc8, unchanged for days) and greedy tokens,
the 84K read and its resume (bea8e6d14ef4a5c5 / ab145ab4ea6c95bc) with every switch on, byte-for-byte kernel parity
(`preload_parity`), 576 bitwise comparisons a format (`pool_tasks_test`). The route bias (0.3, decode only) is the one
output-changing setting, as before §19; with more of the routed experts cached (§41) it has fewer to steer away from.

**Each part against its floor:**
- Decode (~45 ms a token): the GPUs' part 15 ms of kernels with no gaps between them (382 / 365 us a layer; the dense
  projections at 570-720 GB/s, ~100 us a layer of small kernels); the CPU pool ~15.8 ms at 66-67 GB/s (whole experts 72.5);
  the drive 9-12 ms (2-2.3 reads a token; Belady's bound on a conversation's trace 1.3, which no practical policy tried came
  near, §41); the rest ~2 ms (the head ~1 ms at 719 GB/s, the upkeep ~0.7, the engram rows ~0.5). The user's replies decoded
  at 16 tok/s median whatever their context (the server's log, under 16K to over 140K: 15.9-16.8).
- Short reads (a 414-token follow-up): the drive at 2.5-2.7 GB/s and the rings' copies (~97 GB a read through both GPUs, at
  6.23 / 8.51 GB/s) both near full; the drive idle 1.4-2.8 s a read while the routing comes (§41).
- Long reads: 2 windows for 84K tokens (the VRAM's minimum); the drive's slow mode sets their time.

What is left (DSV41_BACKLOG #90-#97, deferred by the user): exact items of 0.2-1.5% each - in all ~2.4-3 ms a decode token
(5-6.5%) and 4-7% of a short read. The larger decode levers are the speculative ones (#70, #80), deferred by the user too.

## 44. The VRAM tier's swaps keep the SSD cache exclusive; the drive guard's way back during a decode (2026-10-22)

**An audit found it** (a read of `glm_tiers.cpp` and a replay of four saved decode traces against the engine's swap rules,
`exsim.py`): a swap whose incoming record comes from its SSD cache slot gave a VRAM-only victim one of the 96 exchange blobs,
and only a swap whose incoming expert had a RAM copy handed one back - all 96 were spent within ~35 tokens, and from then on
such a victim was dropped (its next routing a drive read); the incoming record also stayed in its cache slot beside its new
VRAM copy (832 of the 1,958 VRAM experts doubled in the cache at a trace's end). Replayed, keeping the victims in the cache
and the cache free of VRAM's experts: 19-23% fewer drive reads a decoded token (3.95 -> 3.12 on one trace; the victim placed
as the most recent or at its last use, the same). The simulators of §41 had let the RAM exchange always succeed.

**`STRATA_DSV41_SWAP_CACHE=1`** (opt-in, on in strata-dsv41.json): such a swap copies the victim from VRAM into a cache slot of
its own (the least recently used, held and unpublished until the copy lands), and frees the incoming record's cache slot once
its VRAM copy has landed; the exchange blobs then serve only the RAM-copy swaps, which hand one back each time. The first four
victims are checked against their records on the drive and the line printed ("byte for byte its record", 4 of 4 in the smoke
run). With a fixed tier nothing changes; with the adaptive tier which experts the GPU computes moves, as with any tier change.

`serve_decode_ab.py` (the server's decode: the bias, the adaptive tier, the 60 GiB cache; passes 2-3):

| arm | drive reads a token | the drive a token | tok/s |
|---|---|---|---|
| the config | 1.91 / 2.12 | 9.4 / 10.5 ms | 20.21 / 20.13 |
| SWAP_CACHE (and a spin wait, below) | 1.91 / 1.28 | 9.7 / 6.4 ms | 19.88 / 21.93 |

21% fewer drive reads over the two passes (2.02 -> 1.60), the replay's figure, and +3.7% tok/s (+8.9% in the second, as the
drops built up in the config); the tier's upkeep +0.3 ms a token (more first-time records hashed). A spin instead of a
condition wait for a decode's reads (the host's core and its deep sleep states, an audit item): ~5.0 ms a read either way -
dropped, not in the code.

**The drive guard's way back** (DSV41_BACKLOG #99): it climbed back from its floor only after two samples at >= 2,000 reads a
second, and a decode at the floor reads in bursts (a record's 104 requests one at a time) - a reply after an episode ran at
5.0 tok/s, the guard still at its floor. A sample of 16 reads or more that the drive completed at >= 2,000 a second of its busy
time (the counters' io_ticks) counts as healthy now; the slow mode completes <= ~1,000 a busy second at every depth. On this
drive, idle: 128 KB reads one at a time complete 13,158 a busy second back to back, 2,083 two ms apart and 952 ten ms apart
(its power states between reads) - a request's time alone would not tell the two apart. `drive_guard_test.cpp` (two cases
added): the slow mode's floor (1,000 a busy second at 1 ms) holds; a decode's bursts on a healthy drive (520 reads a second,
13,000 a busy second) are back at full depth in 1.5 s; the old cases as before. ALL PASS.

**Corrections the audit made:** q_lora is 1,280, not 1,024 - q_b is 44.6 MB read at ~680 GB/s, near the bandwidth (§34, §38
corrected; DSV41_BACKLOG #93 can gain at most ~0.14 ms a token); with the spread (§40) 2 MiB pages measured 70.80 GB/s against
69.82 - huge pages stay off for the compaction freezes, not for speed.

## 45. The lent slots' refill from the SSD cache, queued (2026-10-22)

A prompt's refill copied each lent expert without a RAM copy through `fill_slot`, which synced the stream before and after it -
the copies one at a time, and GPU 1's after GPU 0's (205 after a 679-token read, 3,402 after an 84K one). **`STRATA_DSV41_
REFILL_ASYNC=1`** (opt-in, on in strata-dsv41.json): a record the SSD cache holds (its hashes known) is queued from its held
cache slot like a RAM copy, one sync a GPU at the end; the slot check verifies every refilled slot as before (no line). The
smoke run's two reads: no read in flight while refilling 0.4 -> 0.2 s and 0.3 -> 0.2 s (the same test before and after).

## 46. The three-engine study's items: what was measured, what went in (2026-10-23)

The candidates from the study of Maya v1.1.25 and lxe/dsx 01c4731 (DSV41_BACKLOG #102-#112), each measured on this box
before anything was built or kept. Every change below is opt-in in the code and on in `strata-dsv41.json` only after its
check; `serve_decode_ab.py` (one tier, the route bias off) compared each arm's first-token logits and greedy tokens.

**The probes** (`moex/tools/dsv41/hugepage_bench.cu`, `mapped_latency_bench.cu`, `checks/vision_encode_bench.py`,
`checks/serve_picture_read.py`, the 131K-400K cases of `dsv4_kernels_parity`):

| | measured | decided |
|---|---|---|
| copies to the GPUs from pinned memory, 4 KB pages / 2 MiB pages (8 GiB, random 12.8 MB records) | GPU 0 (x8) 6.25 / 6.26 GB/s, GPU 1 (x16) 8.12 / 12.28, both at once 14.38 / 18.55; the CPU's 20-thread reads 75.6 / 76.1 | #102 built: this box's IOMMU translates the GPUs' reads (DMA-FQ), and from 4 KB pages GPU 1 ran at two thirds of its link |
| the server's memory on huge pages (`/proc/PID/smaps`) | 0 of 102.5 GB resident | (the same) |
| the decode's top-k selection alone, one token | 0.046 ms at 131K scores, 0.079 at 262K, 0.108 at 400K | #107 dropped: at most ~0.2 ms a token, at 400K only |
| a GPU's dependent read of mapped memory, alone / with a 12.8 MB copy on its link | GPU 0 0.92 / 5.75 us, GPU 1 1.04 / 6.38 us | #106 dropped: ~3 such reads a layer, 20 layers on GPU 0 - at most ~0.3 ms a token if every layer met a copy |
| the image encoder (CPU), bf16 / fp32, the math path / the fused kernel | 640x480 4.36 / 3.42 / 2.08 / **1.41 s**; 1920x1080 (968 rows) 84.4 / 24.4 / 65.6 / **15.0 s**; error against fp32-math 2.8e-2 (bf16) and 3.7e-6 (fp32 fused) | the encoder in fp32 with the fused attention |

**What went in:**
- **`STRATA_DSV41_HUGE_PIN=1`** (#102): the SSD cache's blocks, the bounce slots and the load's bounce buffers on 2 MiB-aligned
  memory advised MADV_HUGEPAGE before its first touch, and the RAM copy's own request kept (`--arena-thp`'s); the faults go
  through `strata::core::fault_in_paced` - 1 GiB at a time, the request stopped for the rest once a GiB faults below 2 GB/s or
  the memory pressure (`/proc/pressure/memory` some avg10) stays above 20% for 30 s, since compaction for 85 GiB at once froze
  this desktop (2026-10-07) and got dsx's server killed by systemd-oomd; the record files' page cache (they are read O_DIRECT)
  dropped first. Measured: a 53.6 GiB RAM copy stopped after 38 GiB (a GiB at 1,488 MB/s), the start's pinning 51 -> 32 s in
  that run; with the server's 60 GiB cache 9.8 -> 10.3 s. **Decode +7-9%** in `serve_short_warm.py`'s four warm turns (12.2 /
  13.7 / 16.4 / 20.4 -> 13.3 / 14.8 / 17.6 / 22.3 tok/s, no slow-mode episode in either arm), the reads unchanged (15.3-21.5 s
  both: the drive's bytes set them); the same tokens and logits.
- **`STRATA_DSV41_MIX_SIDE=1`** (#105): a one-token decode window's mixer coefficients (the streams' norm, the split gemv, the
  Sinkhorn chain) on a third stream beside the half's attention or MoE - V4.1 reads a half with the previous half's carried
  pre, so they are first read at the half's post-mix; the carried pre alternates between two buffers by the half's parity,
  so it is in `cpre` wherever a window hands it on (the stage boundary, the head). The same kernels MIX_FUSED's launch was
  bit for bit: the same logits and tokens. The GPUs' part of a token outside the pool 17.4 -> 16.6 ms (-0.8 ms).
- **`STRATA_POOL_PIPELINE=1`** (#94): the pool's gate/up and down phases as one batch, an expert's down rows started once its
  own gate/up rows are in (the task that finishes them quantizes the expert, where the host did all of them between the
  phases). `pool_phase_bench` on L05's records: 2 experts a layer 70.53 -> 71.92 GB/s, 6 experts 71.25 -> 72.67 (+2%); the
  same logits and tokens.
- **The image encoder: `STRATA_DSV41_VISION_DTYPE=fp32`, `STRATA_DSV41_VISION_ATTN=fused`**: this CPU has no bf16 instructions
  (no avx512_bf16, no AMX), and the 3-D attention call took torch's math path. `serve_pic_encoder.py` on the "7319" picture:
  both settings answer "7319" (its first token at -0.00005 / -0.00004), the encode 6.19 -> 1.34 s; the smoke run's 184-row
  picture 1.4 s (5.7 s in §22).

**What was not needed or not taken:**
- #103: the prompt path's engram rows already go through the page cache in this config (the mapped reads; `EG_DIRECT` is
  opt-in) - its scales stay cached and a repeated page is read once. `STRATA_DSV41_EG_DEDUP=1` deduplicates the opt-in direct
  path's rows and copies its scales from the mapping (the same bytes); off.
- #112: the rope already spreads over every head (`blocks(n_rows * ppr, 256)`); the router's top-k with its hand-off would
  save one launch a MoE layer (~0.1-0.2 ms a token) - not done yet.
- #104 (the packed attention rows), #108, #109, #110, #111: open (DSV41_BACKLOG).

**Where a picture's time goes** (`serve_picture_read.py`, a 1920x1080 screenshot, 968 rows, behind 1,500 tokens of text, a
fresh read each): the picture's rows read in a call of their own (`glm_session.cpp`: the span as one batch), so the experts the
text and the picture share stream twice - 17,119 experts streamed against 11,390 for as many text tokens - and its rows route
with `bias_vl` to ~2,300 experts the text does not use. Both reads met slow-mode episodes (79 s, 66 s), so their times (133.7
/ 99.8 s) say little; the span's call is the next item for pictures (rounding only: the batch's shape changes).

**Maya v1.1.25 on the same files** (`~/AI/Maya-data/dsv41-shared`: q3q2u's records on 33 layers, the release's MXFP4 records on
layers 0-1 and 15-19 - Maya runs no Q2_K gate/up with MXFP4 down - the dense GGUF, the release's engram tables, Maya's own
engram hashes, which match ours; each engine cold, the page cache dropped, `ab_engines.py`: 16K tokens of docs, then four
~540-token follow-ups, 128 greedy tokens a turn):

| | Maya (exact routing) | Strata-X, exact | Strata-X, the route bias (the config) |
|---|---|---|---|
| ready | 130.9 s | 38.0 s | 35.0 s |
| the 16K read | 85.2 s | 82.2 s | 148.9 s (the drive at 1.36 GB/s) |
| the follow-ups | 19.7 / 16.6 / 18.7 / 98.1 s (an 80 s slow-mode episode) | 20.7 / 18.0 / 19.6 / 19.1 s | 23.0 / 92.8 / 18.9 / 18.5 s |
| decode | 6.1 / 6.7 / 10.5 / 9.9 / 10.8 tok/s | 5.0 / 8.0 / 10.1 / 10.5 / 9.5 | 6.9 / 13.9 / 19.0 / 17.6 / 17.7 |

At exact routing the two engines decode and read alike here; the route bias is ours. Maya's first 12 tokens were ours. Maya's
run froze the desktop for minutes: its RAM tier is the page cache (tens of GB read ahead at its start, the desktop's own pages
pushed out) and nothing in it handles the drive's slow mode.

## 47. The replay's dependency cone, a picture read with its text, the routing with its hand-off (2026-10-24)

Three of §46's items, each opt-in and now in `strata-dsv41.json` (46 switches); the installed engine `build-dsv41/strata-glm`
(sha256 6a74130e...), `serve_smoke.py` ALL PASS with it (ready 34 s).

**`STRATA_DSV41_REPLAY_CONE=1` (#108).** The bounded replay ran layers 21-39 over the last 2,560 rows before each read start.
A read start needs each later layer's 128-row window ring there, and layer L's rows there read layer L - 1's 127 positions
before them, so layer L now runs only [cut - 128 (40 - L), cut): 2,432 rows at layer 21, 128 at layer 39, half the replay's
rows (`Stage::cone_rows`, in the chunk loop and the deferred tails). The rows it drops were never read. Off with a drafter.
Rounding only: fewer rows a launch change which experts the CPU takes and how the kernels sum (as #77).
- Quality (`--kld` on the generated agent text of §12, 2,047 scored tokens against the native base, the GPU opt-ins on, one
  tier): mean KLD 0.0298 ± 0.0014 -> 0.0308 ± 0.0015, same top 94.53% -> 93.89%, perplexity ratio 1.0033 -> 1.0023. Inside
  the engine's run-to-run floor (0.010; the same top's standard error is ~0.4 points on 2,047 tokens), and the two engines
  held 848 and 849 slots on GPU 1 - a placement apart as well.
- Speed (`serve_short_warm.py SHORT_N=1500`: a 3,011-token first read, then four 1,514-token follow-ups; the config's bias
  and adaptive tier; each arm a fresh engine):

| turn | drive bytes, base -> cone | records from the SSD | the read's time |
|---|---|---|---|
| 0 (3,011 tokens) | 128.1 -> 114.8 GB | 9,201 -> 8,323 | 53.3 -> 45.7 s |
| 1 (1,514) | 67.0 -> 60.0 | 4,712 -> 4,264 | 103.6 -> 29.1 s (base: a slow-mode episode) |
| 2 | 62.1 -> 55.7 | 4,410 -> 3,997 | 103.7 -> 27.4 s (base: an episode) |
| 3 | 57.3 -> 51.3 | 4,078 -> 3,684 | 25.9 -> 24.4 s |
| 4 | 53.9 -> 48.2 | 3,837 -> 3,461 | 80.1 -> 95.4 s (both: an episode) |

  The drive's bytes -10.4% to -10.6% on every turn, the same first 12 greedy tokens on all five; the times follow the
  drive's slow mode more than anything else.

**`STRATA_DSV41_PIC_MERGE=1` (#113).** A request's new text and its pictures read in one prompt call: each row routes and
takes the engram by its position (a picture's: `exp_probs_b_vl`, no hash, no engram - `Stage::img_mix`, route() and the
engram by runs of rows), so the experts the text and the picture share stream once and the short text around a picture reads
in the picture's chunk instead of through windows. A periodic checkpoint no longer lands inside a picture's span (a merged
read's chunks could make one); the tail after the last cut still reads through windows. Off with a drafter. Rounding only:
the chunks' shapes change. `serve_pic_merge.py` (each arm a fresh engine, the config):

| | its own call (before) | one call |
|---|---|---|
| the "7319" picture (206 rows) behind a turn's start and a question, fresh: the read | 65.5 s (its 18 text tokens through windows in the drive's slow mode) | **21.6 s** |
| its answer; its first token's log-probability | "7319"; -0.00003 | "7319"; -0.00001 |
| a 1920x1080 screenshot (968 rows) after 1,500 tokens of docs, before a question (2,480 tokens, fresh) | 41.9 s; 14,185 records (6,984 from the SSD); 99.2 GB read | **27.8 s; 8,672 (4,732); 64.7 GB** |

The smoke run's picture (a 184-row gradient): one chunk of 219 rows, the read 17.1 -> 15.4 s.
Past one picture a request (`serve_pic_multi.py`, the installed engine and the config, greedy): two pictures in one turn
(the screenshot, then the number) - 1,194 tokens read in 34.9 s, "7319"; a follow-up reusing that prefix with both pictures
in it and bringing a third - 1,196 reused, 226 read in 11.8 s, "7319"; a text-only follow-up after it - 1,424 reused, 18
read in 2.4 s, "7319". No engine error.

**`STRATA_DSV41_ROUTE_PUBLISH=1` (#112).** A decode window's routing and the doorbell's hand-off of its ids and activations
in one launch (`dsv4_route_publish`): a warp a token runs `route_kernel`'s expressions in its order - the same ids and
weights - and writes its ids into the mapped block too, the block's other warps copy the activations meanwhile, then the
publish's fences and word. `serve_decode_ab.py` (the route bias off, one tier, 1,700 slots, 128 greedy tokens, two passes):
the same first-token logits and the same 128 tokens in the first pass; the second pass's lookup drafting differed (4 lookup
windows against 5 - a window rounds differently from single steps) and parted at token 118. That run's wait for a token's
routing, 14.3 -> 13.3 ms, is no measurement of it: the same figure was 13.7 and 17.3 ms in two engine starts of one setup
(the slots fixed, the same binary and config). Measured instead (`route_publish_bench.cu`: 40 layers' routing and publish in
a CUDA graph, a small kernel between them; the same ids, weights and published bytes): one token 766.3 -> 584.5 us on GPU 0
(x8: -4.55 us a layer) and 753.5 -> 680.1 us on GPU 1 (-1.83 us a layer) - about **-0.13 ms a token** over the two GPUs'
20 layers each; a window of 8 -10.3 / -12.6 us a layer (-0.45 ms). The rest of #112: the rope already spreads over the
heads (§46), and the mixers' coefficients already run beside the half (#105).

## 48. The packed attention rows (#104): exact, 85 more slots, a slower indexer (2026-10-24)

`STRATA_DSV41_KV_PACK=1` keeps the KV sources' compressed rows and index keys as the model rounds them: E2M1 codes two a
byte and their scales (E4M3 per 16 for the compressed rows, UE8M0 per 32 for the index keys - Maya's layout), 288 and 80
bytes a row against 1,024 and 256 as F16. `dsv41_fq_fp4_pack` rounds as `dsv41_fq_fp4` and packs each completed row; the
attention (`dsv4_attn`, `dsv4_attn_mma`) and the indexer (`dsv4_lid_scores`, `_mma`) decode a value to code x scale in F32
and round it to F16 as the store did, so they read the F16 cache's halves bit for bit. The raw window ring stays F16.
- `dsv41_kernels_parity`: the packed store's floats and every stored row decoded are the F16 path's (zero blocks, rows at
  1e-6 and 1e-30, -0); the four readers' outputs are bitwise equal at ratios 1 and 2, one and five tokens, SIMT and tensor
  cores; the negative controls (a scale off by one, shifted selections) differ.
- `serve_decode_ab.py`, both arms' slots fixed per GPU (`STRATA_GLM_STAGE_SLOTS=1111,589` - `--expert-slots` alone fills
  GPU 0 first, and the packed rows leave it more room: 1,165 + 535 slots against 1,111 + 589, another VRAM set, which parted
  the logits): the same first-token logits and the same 128 tokens in both passes.
- The VRAM: the attention state at 524,288 positions 1,625 -> 485 MiB; the config's slots 1,106 + 827 -> 1,161 + 857
  (+85, +4.4%).
- The speed (`serve_short_warm.py`, exact routing, otherwise the config): the five reads 47.1 / 21.3 / 17.7 / 16.0 / 13.6 ->
  44.4 / 18.8 / 18.6 / 15.9 / 13.7 s, the records from the SSD within 3%, the decode 5.0 / 7.9 / 10.0 / 12.1 / 14.6 ->
  4.9 / 8.2 / 9.5 / 12.1 / 14.7 tok/s: no change past the noise at 3-5K positions. The indexer over 200,000 keys (one
  token, a ratio-2 source at 400K positions; `DSV41_PARITY_ONLY=pack_bench`): 0.086 -> 0.097 ms on the tensor cores, 0.245
  -> 0.286 ms SIMT. The F16 kernel already streams its 51 MB near the card's bandwidth, and the unpacking costs more than
  the bytes it saves (a decode by F16 multiplies was slower still: 0.164 ms).
- **Off in the config**: exact, but not measured faster. What would make it pay: the decode folded into the tensor-core
  kernel's B fragments (no unpacking pass), then measured at a long context.

## 49. One activation rounding on both tiers (#109): measured, the planned way is a loss (2026-10-24)

The plan's first step for #109 was one expert against fp32 both ways (`checks/round109.py`: q3q2u's expert 0 of layer 2,
Q2_K, and of layer 39, Q3_K, its 2,304 gate rows dequantized; activations of layer 2's `ffn_norm` weight times N(0,1),
Student-t 3, and N(0,1) with 4 channels x30, 16 vectors each; the relative RMS error of the row dots against float64):

| activations | rows | per 256 (Q8_K, the CPU pool's) | per 32 (Q8_1, the GPU's) |
|---|---|---|---|
| normal | Q2_K / Q3_K | 7.0e-3 / 7.0e-3 | 5.4e-3 / 5.4e-3 (x1.3) |
| Student-t 3 | Q2_K / Q3_K | 1.7e-2 / 1.6e-2 | 8.4e-3 / 8.4e-3 (x2.0) |
| 4 outlier channels x30 | Q2_K / Q3_K | 1.9e-2 / 2.2e-2 | 7.8e-3 / 8.7e-3 (x2.4-2.5) |

Per 32 is the finer step everywhere, as it must be (a 32-block's step is never above its 256-block's). Maya's way - the GPU's
low-bit experts rounding their activations per 256 as ggml's CPU rows do - would make the GPU tier's dots 1.3-2.5x less
exact: a quality loss, not done. The other way - the CPU pool's K-quant rows on per-32 activations - would make the CPU tier's
dots more exact as well as the tiers alike, but needs new AVX-512 Q2_K / Q3_K kernels on the decode's critical path (this
Xeon has no VNNI: the per-32 sums take extra shuffles a block), with a speed cost to measure first. It is a quality project
of its own, like #110 / #111, and the reliability it would buy is smaller than planned: the lookup drafts' policy picks its
windows from measured round times, so greedy tokens can part between runs anyway (§47).

**The picture's experts read during its encoding (the plan's Phase 3, item 2): not done.** The prompt path's stream gate is
unlimited (`stream_gate()`), so every chunk reads only the experts its rows route to, layer by layer - the experts a
picture's chunk will read are not known until its rows reach each layer. Reading ahead on a guess spends drive reads (this
NVMe's slow mode comes from its most-read blocks) and, for a large picture, pushes the decode's records out of the 60 GiB
cache (the screenshot's read takes 4,732 records, ~60 GB, from the SSD). The merged read (#113) took the picture read's
largest excess instead.

## 50. The launch configuration (2026-10-24)

- **The engine:** `build-dsv41/strata-glm` (sha256 5e081b65...: §47-§49's code; `KV_PACK` built in, off). `serve_smoke.py`
  ALL PASS with it and the config: ready 34 s, the reply streamed, the picture's reply streamed, no engine error.
- **The config:** `strata-dsv41.json`, 46 switches, `"diagnostics": false`, the 60 GiB SSD cache, 524,288 positions. New
  in this round: `STRATA_DSV41_REPLAY_CONE=1` (#108), `STRATA_DSV41_PIC_MERGE=1` (#113), `STRATA_DSV41_ROUTE_PUBLISH=1`
  (#112); the round before (§46): `HUGE_PIN`, `MIX_SIDE`, `POOL_PIPELINE`, the encoder in fp32 with the fused attention.
- **Built and off:** `STRATA_DSV41_KV_PACK` (#104: exact, +85 slots, the indexer 13% slower a call), `STRATA_DSV41_EG_DEDUP`
  (#103: the config's engram reads do not need it).
- **Measured and dropped:** #106, #107 (§46); #109's planned direction (§49); the picture's experts read during its encoding
  (§49).
- **Open, each a project of its own:** the CPU pool on per-32 activations (#109's other direction), the dense weights in
  FP8 (#110), better expert rounding (#111), the packed rows' decode folded into the indexer's fragments (#104).

## 51. ISTA-DASLab's GSQ-RCO IQ2_XS experts in our engine: measured, lower quality than q3q2u (2026-10-24)

**What they are.** `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`: non-uniform GGUF quantizations of the release
(`deepseek-ai/DeepSeek-V4.1-Flash-1017`, MIT) - GSQ rounds each tensor at each candidate type, RCO picks one type per tensor
under a size budget. Builds at 1.75 (IQ1_M), 2.00 (IQ2_XXS) and 2.25 bpw (IQ2_XS), and an IQ4_NL engram file (110.6 GB).
Mainline Strata, whose `strata-dsv41` engine (a Maya port) runs them, measured against the release over 4,995 positions of
mixed text: IQ2_XXS KL 0.170 (perplexity +15.5%), IQ2_XS KL 0.107 (+7.3%), IQ2_XS on agent work +2.2%, the sets weakest on
prose.

**The IQ2_XS build's experts** (the shards' headers): per layer, gate/up and down among IQ2_XXS, IQ2_XS, IQ2_S and IQ3_XXS
(layer 23's down) - no IQ1_M, gate and up always one type; 149.1 GB of records (138.9 GiB), 8.70-10.81 MiB an expert, against
q3q2u's 212.4 GB. Every type has our kernels (the GPU's MMVQ and grouped MMVQ, the prompt MMQ, the CPU pool's AVX2 rows).

**The pack** (`gguf_pack.py`): the four shards streamed from Hugging Face (155.2 GB in 116 min, 18-24 MB/s), each layer's
three expert tensors copied byte for byte into our records (zero-padded to 4 KiB); each shard's SHA-256 matched; `--verify 2`
read experts 0 and 383 of all 40 layers back from the GGUF, byte for byte. Nothing requantized. Our Q8_0 dense GGUF, the
release's FP8 engram tables and the learned profile serve the rest (GSQ rounds each tensor on its own). `--sidecar` kept the
6.17 GB the pack does not hold (the headers, the gaps, the dense tensors), so `--rebuild DIR` can write the shards again,
each checked by its SHA-256 (not run: it needs 155 GB free). Config: `strata-dsv41-iq2xs.json`,
`run-deepseek-v4.1-flash-iq2xs.sh` (the pack, its sidecar and both configs deleted on 2026-10-24 at the user's word, the drive
94% -> 86% full; the configs are in git history). `serve_smoke.py` ALL PASS with it: ready 34 s, 2,795 experts in VRAM (q3q2u: 1,933), the
RAM copy 20 GiB (the 60 GiB SSD cache pinned first, the 28 GiB headroom kept).

**Quality** (`kld_iq2xs.sh`: kld_q3q2u.sh's three tests, the installed engine, against the native base; q3q2u's numbers are
§16's, 2026-10-21 - the engine's changes since are exact or rounding-only; the same engine as IQ2_XS's runs gave q3q2u
0.1307 ± 0.0079 on wikitext, 92.62% same top, on 2026-10-24: §52):

| test | q3q2u | IQ2_XS |
|---|---|---|
| wikitext 16x512, prompt path, exact routing: mean KLD | 0.1328 ± 0.0080 | **0.2838 ± 0.0134** |
| median / 99% / max | 0.00090 / 2.25 / 10.2 | 0.00386 / 4.38 / 13.3 |
| same top / perplexity ratio | 92.55% / 1.024 | **88.85% / 1.145** |
| agent text, windows, exact routing: mean KLD | 0.0317 ± 0.0015 | **0.0786 ± 0.0053** |
| median / 99% / max | 0.00045 / 0.318 / 0.728 | 0.00090 / 0.801 / 7.67 |
| same top / perplexity ratio | 94.38% / 1.004 | **90.82% / 1.044** |
| agent text, route bias 0.3 (production): mean KLD | 0.0397 ± 0.0021 | **0.0782 ± 0.0047** |
| same top / perplexity ratio | 93.01% / 1.011 | **90.67% / 1.045** |

IQ2_XS has 2.1-2.5x q3q2u's KL at exact routing and 2.0x in the production setting, 2.3-3.7 points fewer same top tokens,
and 4-14% perplexity against 0.4-2.4% - every gap many standard errors wide. The route bias costs IQ2_XS almost nothing
(0.0786 -> 0.0782: few of its experts are left on the drive to steer away from), where it costs q3q2u 0.008. Consistent with
mainline's figures: our agent text (0.079) below their mixed-text 0.107, our prose (0.284) above it. Not taken: q3q2u stays.

## 52. The model's whole context, and the server's quality checked as configured (2026-10-24)

**1,048,576 positions** - the model's own (`context_length`: 65,536 x YaRN 16). `strata-dsv41.json` (2e2a9e9b):
`--max-context 1048576`, `STRATA_DSV41_KV_PACK=1` (§48), `--prompt-cache-deep 16` (every 65,536-token mark below 1M kept; a
checkpoint holds only the running rings, ~26 MB); `strata-dsv41-iq2xs.json` the same; dsh's `contextWindow` 888000 in both
profiles (the 1/8 margin for dsh's count error kept; the web profile compacts at 843,600, its 44,400 / 35,520 unchanged).
`checks/serve_kv_pack_1m.py` (the config's arguments and switches, `--no-adaptive`, the 30 GiB test cache; ~84.4K tokens of
this repo's docs read from 0, then 16 greedy tokens):

| arm | the attention state | slots (GPU 0 + 1) | the read | the first token's logits, the 16 tokens |
|---|---|---|---|---|
| F16 rows | 3,225.2 MiB | 1,033 + 795 | 9 chunks of up to 9,438 in windows of 4 | 5947c1b5d1056b5e |
| packed rows, A's slots | 945.2 MiB | 1,033 + 795 | the same plan | 5947c1b5d1056b5e, the same 16 tokens |
| packed rows, the slots that fit | 945.2 MiB | 1,142 + 861 | - | - |

The packed rows read back the F16 cache's halves at 1M as at 512K. At 1M the VRAM holds 140 fewer experts with F16 rows than
at 512K (1,828 against 1,968) and 35 more with packed ones (2,003). The reads' times (383 and 205 s) are the first arm's cold
page cache, not the format. What grows with the depth a conversation reaches: per decode token the indexer scores every
visible row on eight layers - the three ratio-2 sources over p/2 rows, layer 20 and the four index sources after it over p
(layers 24-36 score every row, then mask to layer 20's candidates: DSV41_BACKLOG #100's item) - and selects 512 for each;
from §48's 0.086 ms per 200,000 keys and §46's top-k times, est. ~1 ms a token at 150K, ~3 at 512K, ~6 at 1M (not measured:
no read has passed 300,000 positions, §26).

**The server's quality as configured** (the user's report: a short question that went well on 10-19 thought for 32,000+
tokens on 10-24; the launch configuration's switches and the installed engine suspected, and with them §51's IQ2_XS runs).
`checks/kld_server.sh`, the installed engine (5e081b65), q3q2u, against the native base:

| test | 2026-10-21 (§16) | 2026-10-24 |
|---|---|---|
| agent text, exact routing: mean KLD / same top / perplexity ratio | 0.0317 ± 0.0015 / 94.38% / 1.004 (kld_q3q2u.sh's two switches, windows of 8) | 0.0330 ± 0.0016 / 94.04% / 1.002 (the server's 45 switches - 3d04e313's less the bias - and its profile, one-token windows) |
| wikitext 16x512 through the prompt path, kld_q3q2u.sh's switches | 0.1328 ± 0.0080 / 92.55% / 1.024 | 0.1307 ± 0.0079 / 92.62% / 1.023 |

Both inside a standard error: the engine's changes since 10-21 and the launch configuration's switches cost no measurable
quality, and §51 compares IQ2_XS with q3q2u on one engine. The rest since the good run (10-19 ~00:15, 4288c109's config):
the pack q3q2 -> q3q2u (equal, §16), the template (V4.1's since 10-19 00:13), the effort (100 since 02:18), dsh's AGENTS.md
(09-19) - unchanged or measured equal. The sampling is the one difference left: T 1.0 / top_p 0.95 / top_k 64 at 00:15;
T 0.6 / top_p 1.0 / the engine's top_k 20 from 02:18; from 15:39 the whole vocabulary at T 1.0 / top_p 1.0 (DeepSeek's
recommended settings; `sampler_full_kernel`: Gumbel-max, an exact sample of softmax(s / T)). The user keeps DeepSeek's
settings.

## 53. An outside correctness audit (2026-10-24): one engine hang and three server bugs fixed

A read-only audit the user ran beside this round compared the engine with the release's `model.py` and kernels and with
vLLM's and SGLang's choices. Each of its findings was checked here against the code before anything changed.

- **The replay cone's hang (#108, in the config since §47).** A prompt chunk with replay rows streamed all of layers
  21-39 (`prompt_chunk`'s stream), but the cone stops the layer loop at the first layer that needs none of the chunk's rows;
  the issuer, at most `ring_n` entries ahead of what the layers consumed, waited at a full ring, and the chunk's last wait
  (`issued == seq.size()`, no stop looked for) spun until the server's watchdog ended the engine - the request lost, and
  every checkpoint with the restart. It takes a call with one read start (no checkpoint cut inside it: the deferred tails
  need two) whose plan leaves the chunk before the last inside the replay's 2,560 rows: 8,192 + 2,049-2,559 tokens, as a
  follow-up adding ~10.5K tokens. §47's check read single-chunk calls. Fixed: the stream leaves out every layer whose cone
  holds none of the chunk's rows (pos0 + T at or before cut - 128 (40 - il), true of every later layer once true: exactly
  the layers the loop no longer reaches). `checks/serve_cone_hang.py` (a ~3K call, then 10,400 tokens more from the live
  state, the config's switches, `--no-adaptive`): the second call planned "2 chunks of up to 8192 tokens" (8,192 +
  2,208, the cone empty from layer 23, no deferred tails) and finished in 48.7 s - ALL PASS. Installed:
  `build-dsv41/strata-glm` sha256 07ac2a42... (the one before kept beside it as `strata-glm.5e081b65`).
- **The tails' read-ahead and the refill (§37).** `tail_ahead_stop` ran only on the deferred tails' way: a call whose
  last window stopped or failed left the read-ahead running while the refill read the SSD cache's map without its lock.
  It is stopped before every refill now (a no-op when it is not running).
- **The server** (`serve/`: 824 tests OK, the new cases with them):
  - a reasoning object without an effort (`{"reasoning": {"enabled": true}}`), and on /v1/messages an `output_config`
    without one, kept the config's level out - V4.1's template, which thinks only when asked, did not think. Only an
    effort names a level now; `{"enabled": false}` turns thinking off (it was read nowhere).
  - a reasoning that ended on ' before an answer that began with ' kept the answer as reasoning: #537's quoted-tag rule
    took an apostrophe's enclosure alone for a quotation. An apostrophe no longer quotes the tag; the release's parser
    gives the same split on the new cases.
  - /v1/responses had no keep-alive while `prepare()` ran (d915199b covered the chat and messages routes): the same beat
    now, and an error after the stream's start goes out as the Responses API's `error` event.
- **Left open** (DSV41_BACKLOG #114-#118): malformed tool calls (the release's parser rejects them too); three latent
  items - an RTX 50 build's MXFP4 prompt MMQ, a stop inside a merged picture read with the replay off, two buffers'
  sizes; and the calls that are the user's (the EXIF orientation, the level of a request that names none, xhigh's
  mapping). VISION.md's "up to 1,026 rows a picture" corrected to 1,024, the span's delimiters included.
- **Checked and as the release:** the forward pass (RoPE / YaRN, the window, the compressor, the FP8 / FP4 rounding, the
  packed KV layout, the indexer, candidate blocks and the top 512, the sink, the hyper-connections' Sinkhorn order, the
  router - the image bias, the route bias on generated tokens only - the engram, the head, the whole-vocabulary sampler),
  the weights' conversion and records, every size at 1,048,576 positions. Where vLLM and SGLang depart from the release -
  the layers after layer 20 over the last 128 tokens only, the window recomputed after a prefix-cache hit, an FP8
  indexer, bf16 logits - this engine follows the release.

## 54. The image encoder on the GPU (2026-10-24)

**Why.** Screenshots are a regular part of an agent's turns. The CPU encoder (§46: fp32, the fused attention, 20 threads)
takes 15.0 s for any picture of 1920x1080 or more and 1.41 s for 640x480. A 1920x1080 picture is 968 rows from 8,418
patches: 16.4 TFLOP (6.9 in the ViT's linear layers, 9.3 in its attention, 0.14 in the aligner). 640x480 is 206 rows,
1.7 TFLOP. That is ~1.1 TFLOPS on the 20 cores, whose fp32 peak is ~2.8, so no CPU path goes below ~6 s a screenshot. The
encoder's process holds 2.97 GB of RAM for the run (the ViT's 485M weights in fp32, and torch).

**The options:**

| option | for | against | taken |
|---|---|---|---|
| the CPU, as now | nothing to build | 15.0 s a screenshot | - |
| the CPU encode overlapped with the read of the text before the picture | small | an agent's screenshot comes at the end of a turn, behind a cached prefix: little to overlap; the encoder's 20 threads and the reads share the cores | no |
| PyTorch with CUDA in the encoder's process | the code exists | a ~3 GB install; a second CUDA context plus the weights and activations, 2-3 GB of VRAM that the expert slots hold (GPU 0 has 0.9 GB free with the server up): every token pays for the occasional picture, or two processes pass memory back and forth | no |
| in the engine, bf16 tensor cores (the release's own dtype) | ~3x faster again | the rows depart from today's fp32 ones (2.8e-2 relative, §46) | no |
| **in the engine, fp32, in a GPU's free VRAM or GPU 0's borrowed slots** | ~7x faster est., no VRAM kept, ~2 GB of RAM back | new code in the engine | **yes** |

GPU 1 was first left out (it drives the desktop; its loan is the smaller). The user's point during the checks - GPU 1 always
keeps ~2 GiB free in this config (`vram_reserve_later_mib` 2048: 2,050 MiB after the load) - made its free VRAM the first
choice: no slot is lent, and the rows go back through a file whichever GPU encodes them (the same card: the same bits).

**The design.**
- **When:** between requests. The server already encodes in the request queue's turn, with the engine idle
  (`Service._encode_pictures`). A new engine command, `VENC <n> <patches> <rows> ...`, encodes a request's new pictures
  in one go and answers `VOK <rows,...> <ms>` or `ERR <why>`.
- **The VRAM:** borrowed for the call. The workspace is ~0.55 GB at the largest picture (9,189 patches: activations
  0.28 GB, a block of 256 queries' scores 0.15 GB, two layers' bf16 weights and one in fp32 0.1 GB; the aligner's weights
  go through the same buffers 1,024 rows at a time). It comes from the first GPU whose free VRAM holds it with 512 MiB to
  spare - GPU 1 (it keeps ~2 GiB free for the desktop), then GPU 0 (704 MiB free after the load, less after a request) -
  and is freed after. Otherwise GPU 0's top slots are lent as a prompt lends them (`Stage::lend_top`, the prompt path's own
  code moved into a function: `STRATA_DSV41_PROMPT_KEEP`'s copies into the SSD cache, so the refill reads RAM, not the
  drive) and refilled after. Nothing is kept in VRAM between pictures. `STRATA_DSV41_VENC_LEND=1` always lends (a check's
  arm).
- **The weights:** read from the release's safetensors (the vision block's "mmproj" directory, as now) by offset, through
  a manifest the server writes at its start (names, files, offsets, shapes: the engine has no JSON reader). They are read
  on the first VENC and kept pinned in bf16 (0.97 GB). Each layer's weights go to the GPU while the previous layer
  computes (25.7 MB, ~4 ms at GPU 0's 6.25 GB/s) and are widened to fp32 there.
- **The arithmetic:** fp32 throughout, as the CPU encoder with `STRATA_DSV41_VISION_DTYPE=fp32`.
  - cuBLAS SGEMM in pedantic math (no TF32) for the linear layers.
  - The attention as two strided-batched SGEMMs per block of queries, with a row softmax between them: a row of up to
    9,216 scores held in shared memory, one read and one write, a fixed summation order.
  - Every long sum in blocks of 128 terms, each block's sum added to the result in turn (cuBLAS's beta 1) - as the CPU's
    GEMMs block their K and torch's fused attention its keys. Added after the first parity run (below).
  - RMSNorm, the 2-D RoPE (its cos / sin from double precision), SiLU, erf GELU, the 3x3 unfold.
  - The file builds without fast math. Deterministic: the same picture gives the same rows every time (the engine's
    prefix keys hash them).
- **The server:** decodes and plans each picture with PIL and numpy (vision_dsv41_plan.py, the release's resize and pad,
  its bf16 rounding of the pixels: the encoder's `load_image` bit for bit, Pillow 12.3.0 in both venvs) and writes its
  patches (bf16 after an 'SVP1' header) for VENC. The rows come back as the same SVE1 files, so the picture cache, GENI
  and the prompt path are unchanged. With `"vision": {"gpu": true}` (deepseek41) the encoder process is not started;
  `"gpu": false` keeps it, as today.

**The checks** (~15 min of GPU time, the server stopped):
1. The serve tests: the patches against torch's `load_image` bit for bit; the VENC exchange (answers, refusals, an
   engine that ends); every serve module.
2. `dsv41_vision_parity`: the GPU's rows against the CPU encoder's on four pictures (640x480 to 1920x1080, padded on
   both sides): relative L2 at most 1e-5 (the CPU's fused attention against its math path: 3.7e-6), the same bytes on a
   second encode, and a perturbed weight must fail.
3. `serve_pic_encoder.py` through the GPU encoder: "7319", its first token's log-probability against the CPU path's
   (-0.00005).
4. A 1920x1080 screenshot's encode time; the smoke run's picture; a text request after it.

**Expected:** a screenshot 15.0 -> ~2 s (est.: the SGEMMs ~0.45 s, the attention ~1.4 s - at 1920x1080 its scores' traffic,
~18 GB a layer, is most of it), 640x480 1.41 -> ~0.25 s; when slots are lent, their refill ~0.5 s a request with new
pictures; ~2 GB more RAM for the experts' copy. The read of the picture's rows is unchanged.

**Measured (2026-10-24, the server stopped; logs in ref-runs/engine-checks/vision-*).**
- **The first parity run failed** where it grows: against the CPU encoder 2.5e-6 to 5e-6 on the small pictures but 1.4e-5
  at 4,784 patches and 1.1e-4 at 8,418. Against a float64 encode (`vision_fp64_ref.py`: the release's steps in float64, the
  RoPE angles in float32 as vision.py forms them) the CPU encoder was at 2e-6 to 9e-6 and the GPU's error grew with the
  patches: cuBLAS summed each whole K (8,418 keys, 9,216 aligner inputs) in one fp32 run, where the CPU blocks it. On the
  1920x1080 picture against float64: whole sums 1.15e-4, blocks of 512 2.95e-5, blocks of 128 6.0e-6 (+15% time) - 128.
- **Accuracy** (`vision_gpu_parity.py`, against float64; PASS: within 1.5x on each picture, at most the CPU's on average):

| picture (patches) | the CPU encoder | the GPU encoder |
|---|---|---|
| "7319" (1,610) | 2.07e-6 | 1.70e-6 |
| 640x480 noise (1,610) | 1.75e-6 | 1.29e-6 |
| 1280x720 gradient (4,784) | 5.15e-6 | 3.43e-6 |
| 500x1600 noise (4,140) | 8.93e-6 | 1.22e-5 |
| 1920x1080 noise (8,418) | 9.27e-6 | 6.01e-6 |

  The GPU's error 0.81x the CPU encoder's on average (geometric mean); both are fp32's, ~2,500x below the bf16 arithmetic of
  the release itself (2.8e-2, §46). Every picture gave the same bytes on a second encode, GPU 1 GPU 0's bytes, and the
  negative control (block 16's mlp.w2 from block 17's) erred 0.2 - it fails.
- **Through the engine** (`serve_vision_gpu.py`, the config's switches, the route bias off): the "7319" picture's VENC rows
  read behind the question answer "7319", the first token at -0.00000 (the CPU encoder's rows: -0.00001; the second-ranked
  token, at -12.3 / -11.6, moves with rows 2.6e-6 apart); with GPU 0's slots lent (`STRATA_DSV41_VENC_LEND=1`) the same
  bytes and the same top 8, a text request after it fine, no slot check failed; GPU 0's free VRAM (`STRATA_GLM_STAGE_SLOTS`
  900,800) the same bytes. With the server's slots GPU 0 had too little free VRAM for the 512 MiB margin after a request,
  and GPU 1 took every picture.
- **Times** (the encode alone, `dsv41_vision_parity`'s second pass; the CPU encoder's from the same pictures):

| picture | rows | the CPU encoder | the GPU encoder |
|---|---|---|---|
| 640x480 | 206 | 1.4 s | 0.30 s |
| 500x1600 | 509 | 4.6 s | 1.07 s |
| 1280x720 | 578 | 5.8 s | 1.34 s |
| 1920x1080 | 968 | 15.5 s | 3.57 s |

  A VENC adds the patches' file and the workspace: a 1920x1080 picture 3.83 s on GPU 1's free VRAM, 3.69 s with GPU 0's
  slots lent (their refill ~0.1 s). The weights' read: 0.6-0.9 s, once a start, on the first picture. The estimate was
  ~2 s a screenshot: the blocked sums cost 15%, and the attention's scores traffic is the larger part, as estimated
  (DSV41_BACKLOG #120: the attention fused, the scores never in memory).
- **The whole server** (`serve_smoke.py`, the config as it stands): ready, both replies streamed, the picture's request
  (184 rows) in 16.4 s, its encode 0.30 s on GPU 1, no engine error.
- **Installed:** build-dsv41/strata-glm b4988786 (07ac2a42 kept beside it as strata-glm.07ac2a42); strata-dsv41.json
  `"vision": {"gpu": true}`. `"gpu": false` brings the CPU encoder back.

## 55. The drive's slow mode against the other engines' drive code (2026-10-25)

**Why.** The user, still suspicious of the slow mode (§14, §18), asked whether Maya or dsx reads or writes the drive in a
way that avoids it. Read only, nothing built or run: Maya v1.1.27 (4615f34), dsx 443bdce (72 commits past §46's
01c4731), Xolotl's and upstream Strata's docs.

| engine | experts in neither VRAM nor RAM | the RAM tier | the slow mode |
|---|---|---|---|
| Strata-X | O_DIRECT records, 12 readers x 4 MiB calls (up to ~384 requests of 128 KB) | the pinned copy and the 60 GiB SSD cache, exclusive with VRAM (§44) | the guard (§18): one reader and 128 KB calls 0.3 s into an episode, the engram's pages one at a time |
| Maya | prompts: O_DIRECT a part of an expert at a time unless the page cache holds it (`IoQueue`, no bound but its threads); decode: page faults on the file's mapping, `madvise(WILLNEED)` | the page cache (`--mmap-experts auto` copies only when all fit with 16 GB spare) | none |
| dsx | page faults on the GGUF's mapping (O_DIRECT only for the start's arena copy, 64f3ab3) | a hugepage arena filled at the start, the page cache | none |
| Xolotl | O_DIRECT, 32 persistent readers, a layer ahead | pinned | none |

Maya, on a failed read, ends a prompt, leaves the expert out of a decode token (`lane_copy`) and zero-fills an engram row
(ds4_model.cu:6403): the output changes silently where the drive errs. Nothing in any of them writes at run time but
debug dumps.

**The drive today** (counters only; the box up 6.9 h, the user's server 6.83 h): the engine read 11.90 TB
(`/proc/<pid>/io`) and wrote nothing - 1.74 TB an hour, the drive busy 33% of the time; the drive wrote 11.9 GB in all.
43-48 C. The host memory buffer 40 MiB (the size the drive asks for; Linux's cap is 128). Swap 620 KB used (156 pages out
since the boot); file refaults 13.6M pages, ~0.4% of the bytes read. Trimmed weekly (10-19; the user's fstrim 10-24;
next 10-26); 128 KB the largest request, no I/O scheduler.

**Nothing to adopt.** The slow mode is in the drive's blocks, not in how a read is issued (§14: the same gapped load ran
clean on fresh copies and met an episode within 1-2 minutes on the pack), and its episodes come in proportion to the
bytes read. Strata-X's bytes a request are at this RAM's floor: a short follow-up needs nearly every expert, 2,500-3,400
records (~35-50 GB) of them from the drive (§41), in any engine; a long read takes each expert once a layer window
(§17-§19); the deep checkpoints shorten re-reads (§14). The page-cache tiers of Maya and dsx would read at least as much
here: a pass over a set larger than RAM is the case LRU serves worst. Holding the drive busy (E1, E3) only postpones its
rewrites - after E1's 300 s, E2 met two episodes back to back - and leaves the worn blocks aging: not done. What would
change the rate is fewer bytes read an hour: the RAM tier (§10).

## 56. Pictures by reference, the drive's lulls, layers 24-36 on their candidates: the plan (2026-10-27)

**Why.** The user: "Any potential speedup/optimisation without quality loss is welcome! Proceed with the engine change along
with the one you proposed earlier that can mitigate slowdowns. Plan thoroughly first". Three changes, each exact (the same
tokens and logits) or outside the model's arithmetic, each opt-in or automatic only where the old path stays the fallback.

**A. A request's pictures by reference (GENI manifests).** Since 2026-10-27 dsh keeps up to 256 MiB of pictures a request
(the llamacpp provider's `maxRequestImageBytes`; under dsh's 20 MiB default it turned the oldest picture into text at each
new one, and the engine read the rest of a ~300K conversation again: 176-225K tokens, 14-16 min, three times in an
afternoon). Every turn the server now copies each picture's rows file (13.1 MB at full size: 800 rows of 4,096 floats)
into one file in /dev/shm, and the engine reads it whole and FNV-1a-hashes every byte to key the pictures for prefix reuse:
0.80 GB/s on this Xeon (measured), 0.56 s a turn for today's 34 pictures, ~4.8 s and ~7.6 GB of RAM in flight at the
~290 pictures 256 MiB allows.
- Engine (`glm_serve.cpp`): a GENI file that starts `SVM1` is a manifest - one line a picture in prompt order: rows to
  skip, rows, nx, ny, a 64-bit content key, the rows file's path. Each file's SVE1 header is checked against its line
  (magic, n, nx x ny, the model's width, the file's size) and the file mapped read-only; a span's rows point into its
  mapping, so only the pictures past the reused prefix are ever read, and nothing is hashed. The SVE1 file stays as it is
  (the other engines, the tests). `INFO ... geni=manifest` tells the server.
- Server: with `geni=manifest` (and `STRATA_GENI_COMBINED` unset - the old path, for A/B checks), prepare() writes the
  manifest instead of the combined file; the key is blake2b-64 of the rows file (header and rows), computed once a file
  (memoized by path, size and mtime), with the skipped rows folded in (deepseek4v's lead pads). The request holds its
  pictures until its run ends, not only until the copy: the cache cannot delete a file the engine has mapped.
- Exact: the same rows reach the same positions; only the key's function changes (both are content keys, compared within
  one engine run).
- Checks: the server's tests (the manifest written, the hold kept to the run's end, the old path without the flag);
  `serve_smoke.py` ALL PASS; a picture conversation through the server with the old path and the manifest - the first
  token's logits bit for bit, the second turn's reuse the same; the per-turn time with many pictures before and after.

**B. The drive's lulls (the slow mode, §14, §18, §55).** §14's loads: nonstop reading 300 s - no episode; 0.4 s pauses -
none; 2 s pauses - episodes within 1-2 minutes, and right after the nonstop run two back to back; a 4 KB read every 50 ms
in the 2 s pauses halved them (108 -> 53 slow seconds). The drive starts a rewrite in a lull. On the user's traffic
(2026-10-26, a 3-minute sample at 595K) the drive was never idle half a second inside a turn; the lulls were the 2-2.5 s
between turns. The work owed (a rewrite per so many reads of a block) does not change; when it is done might.
- B1 (running since 2026-10-27 19:04, read only): a 1 Hz log of the drive's counters, the engine's bytes read and the
  server's busy flag - the episodes on the user's own traffic, their length, and the lull before each.
- B2 (`moex/tools/dsv41/drive_lull_test.py`, the server stopped, ~40 min): §14's E2 load (12 readers, 4 MiB O_DIRECT calls
  over the pack, 8 s on / 2 s off) with and without a filler - one O_DIRECT read of a 256 MiB random file whenever the
  drive's `inflight` has been 0 for T ms (T = 10 and 2 ms; 4 KB and 128 KB), arms alternated with plain ones so deferred
  rewrites show; slow seconds counted as §14's (under 150 MB/s with the load on). Then one episode left with the host
  silent against one under the guard's floor: does silence end it sooner?
- B3 (only if B2's filler holds the episodes off): the filler in the engine's drive guard (opt-in `STRATA_DSV41_DRIVE_FILL`),
  on while a request runs and for a few minutes after the last one, so the drive's rewrites move into the user's breaks;
  B1's log before and after on the user's traffic. B4 (only if silence ends an episode sooner): the guard's floor stops
  and probes instead of reading one request at a time.
- Risk (B3): the drive's refreshes wait for a break of minutes; the firmware still decides when a block must be rewritten.

**C. Layers 24-36 score only layer 20's candidate blocks (DSV41_BACKLOG #100's largest item).** The release's candidates
are 2,048 blocks of 8 rows (16,384 rows); the four index sources after layer 20 (ratio 1, p rows each) score every row,
then `dsv41_cand_mask` sets all but those to -inf: at 600K positions 97% of those four layers' scores are computed to be
discarded. Each row's score is its own (the 64 heads' MMA over 128 dims, relu, weight, the heads' sum in one order), so the
candidates' rows gathered into the tensor-core kernel's 64-row tiles (eight blocks a tile) give the same bits; the mask
still writes -inf everywhere else, and the top-512 selection reads the same array.
- Layer 20's candidates as an ascending list as well as the bit mask (one small kernel after `dsv41_candidates`); a
  `lid_scores_mma` variant taking the list; opt-in `STRATA_DSV41_CAND_ONLY=1`. The SIMT kernel keeps the full scan.
- Checks: `dsv41_kernels_parity` - scores bitwise equal to full-then-mask (packed and F16 keys, T = 1 and a 37-row block,
  masks dense, sparse, all and none, n_vis off the 8- and 64-row edges), the selections equal; the server's decode A/B
  (`serve_decode_ab.py`) the same logits and tokens; the kernel timed at 300K, 600K and 1M.
- Expected (§48's 0.097 ms per 200,000 packed keys): ~0.5 ms a decode token at 300K, ~1.1 at 600K, ~2 at 1M; a 1,000-token
  follow-up at 600K ~1.1 s shorter; an 84K fresh read ~6 s.

**Order.** Now, with the user's server running: the code, builds in `build-dsv41-dev` at the lowest CPU and I/O priority
(the installed `build-dsv41/strata-glm` untouched until the checks pass), the CPU-only tests, B1. With the server stopped
(~1-1.5 h, the user's window): A's and C's GPU checks, then B2. Then the config (CAND_ONLY), the install, this section's
results; B3/B4 only on B2's evidence.

**Results, the server stopped (2026-10-27 19:44-19:55; `build-dsv41-dev`, sha256 566a1b1b).**
- C: `dsv41_kernels_parity` PASS - the 24 new checks (the masked scores bit for bit, F16 and packed keys, ratios 1 and 2, T 1
  and 37, 40 / 1,000 / 1 blocks kept; the lists) and their negative controls (a list one block off differs), the rest as
  before. `DSV41_PARITY_ONLY=cand_bench`, one token at an index source after layer 20 (2,048 blocks of 8, packed keys):

| visible rows | the full scan + mask | the candidates' rows + their list + mask | x 4 index sources, a decode token |
|---|---|---|---|
| 300,000 | 0.171 ms | 0.030 ms | -0.56 ms |
| 600,000 | 0.332 ms | 0.037 ms | -1.18 ms |
| 1,000,000 | 0.526 ms | 0.050 ms | -1.90 ms |

  `serve_decode_ab.py` (24,000 tokens of the plan's text so the candidates engage, the route bias off, slots pinned 1111,589,
  64 greedy tokens, two passes): the first token's logits 6dd9a8d091361e2a in both arms, the same tokens in both passes.
- A: `serve_pic_multi.py` (the screenshot, 968 rows, and the number, 206; three turns, greedy) with the combined file and with
  the manifest: the same in every turn - 1,194 read in 36.5 / 36.4 s, then 1,196 and 1,424 reused (both pictures in the
  prefix), 226 and 18 read, "7319" each time. `serve_smoke.py` with the new engine and CAND_ONLY: ALL PASS (ready 31 s, the
  picture's reply 17.9 s, no engine error).
- Installed: `build-dsv41/strata-glm` = 566a1b1b (b4988786 kept beside it); strata-dsv41.json `STRATA_DSV41_CAND_ONLY=1` (48
  switches); both from the user's next start, with the server's manifest path (it follows the engine's `geni=manifest`).

**B2's result (2026-10-27 19:56-20:41, the server stopped; `drive_lull_test.py`, 5-minute arms of §14's E2 load):**

| arm | slow seconds | episodes (longest) | the load's MB/s |
|---|---|---|---|
| plain | 36 | 1 (36 s) | 2,168 |
| filler: 4 KB after 10 ms idle | 21 | 1 (21 s) | 2,330 |
| plain | 0 | 0 | 2,628 |
| filler: 4 KB after 2 ms idle | 142 | 3 (64 s) | 1,081 |
| plain | 85 | 2 (54 s) | 1,640 |
| filler: 128 KB after 10 ms idle | 0 | 0 | 2,673 |
| plain | 174 | 3 (63 s) | 724 |

- The fillers do nothing the plain arms do not do by themselves (0-174 slow seconds): the densest one - the drive never
  idle 2 ms - met the slow mode as often as any arm. §14's lulls were not the trigger at this grain: what held it off
  there was a full-speed load, which only postpones the drive's work. **B3 not done.**
- An episode's length: 71 s with the host silent (a probe of 64 one-at-a-time 128 KB reads every 2 s) and 70 s under the
  guard's floor (one 128 KB reader nonstop); in it the drive completed ~497 requests a second one at a time, against
  ~4,400 healthy - a fixed ~2 ms a request, whatever the host does. **B4 not done**: stopping does not shorten it.
- Under this load (~2.2-2.6 GB/s for 8 s of every 10) an episode came about every 400 GB read. What is left for the slow
  mode is reading fewer bytes - §55's conclusion, now with the two software ideas measured out.

## 57. Parking a conversation: the plan (2026-10-28)

**Why.** The engine holds one conversation's state: a request from another conversation (a second dsh session, a subagent,
a client's side request such as a title) overwrites the attention state and deletes the first one's checkpoints, and the
first one's next request reads it again from its root - 15-60+ minutes at the user's 200-600K. The user: "parking is what I
would benefit from atm ... robust plan first, then build".

**What a conversation's state is** (strata-glm, DeepSeek-V4 / V4.1; read from glm_session.cpp, glm_dsv41.cpp, glm_dsv4.cpp):
- the positional caches - on V4.1 the KV sources' compressed rows and index keys (layers 2, 8, 14 at ratio 2, layer 20 at
  ratio 1; packed: 288 + 80 bytes a row, F16: 1,024 + 256), rows [0, len / ratio] of each: ~920 bytes a position, 0.55 GB at
  600K; on V4 every compressed layer's rows (and the indexer's at ratio 4);
- the running state at the live end - what a checkpoint copies (`Stage::state_regions`: the raw rings, the compressors'
  rings, the drafter's rings), ~28 MB on V4.1;
- the live tokens and their pictures' keys, the conversation's checkpoints (each ~28 MB, pinned), and the engram's
  per-position ids on the host, which `Impl::restore` already rebuilds from a checkpoint's tokens (`eg_note`).
Nothing else is per conversation: the VRAM tier, the SSD cache, the route bias's table, the decode graphs (positions come
from the device) and the sampler's history are global or per request.

**Design** (new `glm_park.cpp`; the policy as pure functions in `glm_park_policy.hpp` with a CPU test):
- **Park** the live conversation when a request leaves it: the prompt parts from the live tokens before the live
  conversation's last turn token, and at least `min-tokens` would be lost. A follow-up never parts before it: with tools
  (every dsh request) DeepSeek-V4.1's encoder keeps every turn's reasoning and the follow-up extends the live tokens; in a
  chat without tools it drops the reasoning of turns before the last user turn (encoding.py `_drop_thinking_messages`), so
  the follow-up parts right after that turn token - a rule on the previous prompt's end would park every such turn. The record: the positional rows (pageable RAM), a snapshot of the running state at the live end, and the
  conversation's own checkpoints - the ones that are not prefixes of the new prompt, which today's rule deletes. Prefixes
  the new prompt shares (the system prompt's root) stay in the engine's list for both.
- **Restore** a parked conversation when the request continues it further than the live state and checkpoints can (by 1,024
  tokens or more): its prefix length if its tokens and pictures are a prefix of the prompt, else its longest checkpoint that
  is. The live one is parked first by the rule above; then the rows go back, `restore()` puts the running state back and
  re-notes the engram ids, the live tokens and pictures and the checkpoints are the parked ones, and the request goes on as
  if the conversation had never left (the normal reuse finds its live end or its checkpoint).
- **Budget**: `--conversation-cache-mib` (0 = off), `--conversation-cache-slots` (4), `--conversation-cache-min-tokens` (8,192),
  `--conversation-cache-min-free-mib` (8,192: MemAvailable after the copy). Over budget: the least recently used records go
  first, then the record's oldest checkpoints (its live-end state and newest turn checkpoint last); still over: not parked
  (today's behaviour). A record another one extends is dropped (superseded); a conversation a record already extends is not
  parked again. The engine's INFO line names the budget and slots (the server's Monitor card); each park, restore and skip
  is a stderr line in the format the server already reads.
- **Exact**: the same bytes go back to the same buffers; rows past the restored length are written before any read sees them
  (the invariant a checkpoint's restore already relies on: a visible row is a completed block). The check below compares
  bits.

**RAM**: a parked 600K conversation ~0.55 GB of rows + 28 MB of state + its checkpoints (~16-22 x 28 MB) - about 1.2 GB; the
4,096 MiB budget holds three such or many smaller. Pageable for the rows (no mlock), the checkpoints stay where they are.

**Checks.** CPU: the policy test (the leave rule around the turn token, the budget's order, superseded records).
GPU (the server stopped, `checks/serve_park.py`: the config's arguments with the route bias off, `--no-adaptive`, the slots
pinned per GPU, greedy, first-token logits by hash):
1. uninterrupted: A (~16K tokens of this repo's docs, 32 tokens), then A's follow-up - its logits and tokens;
2. parked: A, then B (~8K other text), then A's follow-up - the same logits and tokens, its read resuming at A's live end;
   then B's follow-up - B restored the same way;
3. a budget too small: nothing parked, A's follow-up reads from the root as today.
The park and restore times, and the RAM.

**Left alone**: one request at a time (a parked conversation still waits its turn); GLM and MiMo (the flag is refused for
them); the default (0: off) is today's engine byte for byte.

## 58. ISTA-DASLab's GSQ-RCO IQ2_XXS as a second pack: IQ1_M in the engine (2026-10-29)

**Why** (the user, 2026-10-29): "Download the IQ2_XXS quant. Although I lose quality, it'll be my cheap day-to-day driver" -
§51's lower-bit build of the same series (2.00 bpw; mainline's KL 0.170 against IQ2_XS's 0.107), which nearly fits this
box's VRAM + RAM copy + SSD cache (§51's estimate on 2026-10-27: ~21-23.5 tok/s at 595K against q3q2u's 15.8, ~16 GB of it
on the drive). q3q2u stays the quality pack; `run-deepseek-v4.1-flash-iq2xxs.sh` / `strata-dsv41-iq2xxs.json` (the q3q2u
config with this pack) started this one - the pack and both configs deleted on 2026-11-02 at the user's word (the drive
93% -> 86% full; the configs are in git history).

**The pack** (`gguf_pack.py`, streamed as §51's, no sidecar): 138.2 GB of shards (6 now) into 132.1 GB of records (123.0
GiB), 7.4-9.8 MiB an expert; per layer gate/up and down among IQ1_M (16 layers' gate/up, 28 layers' down), IQ2_XXS and
IQ2_XS. The release's FP8 engram tables and the drafter's experts are q3q2u's links (`engram`, `experts/M0-M2.bin`).

**IQ1_M in the engine.** The decode's kernels had it (`native_mmvq.cu`'s single and grouped MMVQ, iq_multi_parity and
native_grouped_parity bitwise PASS) and the CPU pool takes it through ggml-cpu's own rows (`native_fmt`); the prompt path did
not - llama.cpp's MMQ has no IQ1_M tile, and a layer without one turns the whole prompt path off (`mmq_all`: every prompt
read through decode windows). `moex/patches/llama.cpp/0001-mmq-iq1_m.patch` (applied by CMake to the pinned checkout once,
`git apply --check --reverse` telling an applied one): `ggml_cuda_mmq_load_tiles_iq1_m` puts IQ1_M into IQ2_XS's Q8_0_16 tile
- each group of 8 as the int8 8 grid +- 1 (its +-1/8 delta folded in), each 16-value half its float scale d (2 s + 1) / 8, so
every weight is exactly `dequantize_row_iq1_m`'s and the products are IQ2_XS's vec_dot_q8_0_16_q8_1 as they are; the MMQ
tables' entries, IQ2_XS's tile configurations, an instance file; `moe_mmq.cu` lists the type (STRATA_MOEX_KQUANTS).
`tests/cuda/prefill_mmq_iq1m_test.cpp` (random IQ1_M blocks at V4.1's shapes - gate and up 4608 x 5120, down 5120 x 2304 - and
1000 x 768 for the fallback tiles, 5 batches of experts and rows): against ggml's dequantized weights in double precision
rel_l2 0.0022-0.0024 (the q8_1 activations), and against the activations rounded as quantize_mmq_q8_1 rounds them 1e-7-1e-5
(the float sums' order only) - PASS. The engine with it on q3q2u: `serve_smoke.py` ALL PASS, and a 4,013-token read's
first-token logits and tokens the installed engine's bit for bit (d2f4cc8962327208) - nothing else changes.

**The pack's checks** (2026-10-29): every shard's SHA-256 matched Hugging Face's while it streamed; `gguf_pack.py --verify 2`
read experts 0 and 383 of all 40 layers back from the GGUF - 80 records byte for byte, 0 problems. `serve_smoke.py` with its
config: ALL PASS (ready 94 s, the first reply 21.7 s after the request, the picture's 6.7 s). Its quality on our three tests
(`kld_iq2xxs.sh`) was not run - stopped at the user's word; mainline's figures stand for it (KL 0.170, perplexity +15.5% on
mixed text). DSpark on it (`--draft-dspark builtin --spec 3`, `DSPARK_ROWS=-1`; the user's runs): no gain noticed, off again;
window 2 left to try (DSV41_BACKLOG #123).

## 59. Every speed item of the audit of 2026-10-29 and the backlog: the plan (2026-10-29)

**The goal (the user, 2026-10-29):** "try EVERYTHING (from the audit and our backlogs) but keep the tests minimal avoiding
redundancy". Each item opt-in (`STRATA_DSV41_*`), the default path byte for byte as before; none that changes the output
beyond rounding. Left out: the lossy ones (#10), rewritten model files (#111, #110), backlog #53, the driver patch (#119),
the hardware levers (§10), the rows that are not speed (server, tools, robustness).

**Prefill (exact unless marked):**

| item | from | what |
|---|---|---|
| P1 | audit A2 | layer 39 in the cone: its ring rows only (no attention output, FFN, MoE; its experts out of the streams and the tails' read-ahead) |
| P2 | audit D2, #101 | the indexer's 32 heads, not V4's 64 with half of them zero (two query rows a block) |
| P3 | audit A3, §57 audit 1a | GPU 0 carves no candidate rows; the loan's context-sized buffers by the call's context, not `--max-context` |
| P4 | audit D1, #77 (rounding only) | layer 20 outside the cone: its compressor and index keys only |
| P5 | audit B3, #81 | the read-ahead into cache slots not held to the bounce slots' count |
| P6 | audit B1 | the long read's kept records shared out by layer (the drive reads through every layer of a window) |
| P7 | audit D3 | a window's attention halves before its FFN halves (the next layer's copies behind them) |
| P8 | #81 | more chunks a window at long context (the indexer scores' scratch) |
| P9 | #81, #101 | the first window's engram rows and each window boundary's idle drive |
| P10 | #68, #101 | the routing gap (a layer's RAM and cache copies first), the refill and the lend's copies off the critical path |
| P11 | §57 audit 2c | the tails' pass gated by the routing (two read starts: the pieces' routings first) |
| P12 | audit A4 | the long read's idle pinned buffers lent to the SSD cache between long reads |
| P13 | audit A5 | the engram rows: a call's rows deduplicated and sorted, read once |
| P14 | #8 (rounding only) | wider chunks (`--prefill-chunk 12288`) |
| P15 | #91, #96, §57 audit 2a/2b | the follow-up's idle drive at GPU 0's routing and the hand-off; the lend's displacement |
| P16 | audit A1, §57 audit 1b | fewer passes over GPU 0's layers in a long cold read - a traffic model first |

**Decode (exact):** #69 (`--threads 19`), #90 (`EG_LATE`, written), #97, #95, #92, #100's items, #40 (the VRAM experts'
K-quant kernel), #11; DSpark: #80's A/B (w2 / w3 against plain), #18, #4 (#70 last, very large). **Pictures:** #120.
**With the user's routing data only** (expert ids, recorded with their OK): audit A6, #67 (b-c), #88.
**Drive:** the episode line with the drive's temperature, the writes and the swap (audit C) - a diagnostic.

**Tests, minimal:** (1) exactness - the first token's logits and 4 greedy tokens of a 4,013-token read (`first_bits.py`, the
baseline d2f4cc8962327208) and of a long read with a checkpoint cut (windows, tails, cone), every exact switch on against
all off: equal bits pass them together, a difference is bisected; (2) one KLD run of the rounding items (P4, P14) against
§16 / §52's numbers; (3) speed after the IQ2_XXS download (its writes keep the drive in its slow mode): the follow-ups
(`serve_short_warm.py`), the cold long read, the decode (`serve_decode_ab.py`) - the config against all on.

**Results (2026-10-29).** Built, each opt-in: P1 `LAST_RING`, P2 `IDX32`, P3 `CAND_STAGE`, P5 `DEEP_AHEAD`, P6 `KEEP_SPREAD`,
P7 `LM_SPLIT`, P15 `FIRST_AHEAD` (N tokens and up; 1 = 2048), and the episode line (the drive's temperature, the MB written
and the pages swapped out over the 10 s before). Not built: P4 and P14 (rounding only: the user's "0 regression in quality"),
P11 (the tails' ungated pass costs little: ~5-9 GB an hour), P13 (the page cache already takes a call's repeated engram rows),
P12 (left: DSV41_BACKLOG #123); P8-P10, P16 and the decode items are DSV41_BACKLOG #121-#123.
- **Exact** (`first_bits.py`; the installed engine 566a1b1b against the dev build, every switch on): the 4,013-token read's
  first-token logits and tokens d2f4cc8962327208 (all off: the same); a 70,013-token read with its checkpoint cut at 65,536
  d935a9829c04b0e2 and a request resumed at that cut 6609fd30ff25518d - both the installed engine's. P2's kernel
  (`dsv41_kernels_parity`): the padded 64-head tile's bits in all 24 cases (full and over the candidates, F16 and packed keys,
  ratios 1 and 2, 1 and 37 rows). The checks found two faults in the first build, both fixed: P7 took a chunk end's copy of
  the running state after the FFN half - by then the next chunk's attention had moved the ring on (the resume step was added
  for it); P3 carved the loan without the candidate rows but laid the MoE buffers out with them, past the loan's end (an
  illegal access at the first read).
- **Speed** (q3q2u; the IQ2_XXS download over, the drive idle otherwise): the follow-ups (`serve_short_warm.py`, the server's
  config: the route bias, the adaptive tier, the 60 GiB cache), four of ~414 tokens - 75.2 -> 73.1 s, the drive's bytes
  151.9 -> 147.6 GB (-2.8% each; the first turn met an episode in both arms); the cold 44K read at 1M context
  (`serve_lm_speed.py CONFIG_ENV=1`: the config's switches, one tier) 123.9 -> 113.5 s (-8.4%; 15,056 -> 14,837 records from
  the drive; one window of 6 chunks in both, so P3's wider windows had nothing to save there); the decode
  (`serve_decode_ab.py`, exact routing, one tier): every arm within the drive's swing (7.06 / 8.47 tok/s the config, the
  passes' rate following their drive reads), `EG_LATE` (#90) the engram rows' time 1.50 -> 0.42 ms a token, `--threads 19`
  (#69) unresolved (its second pass met an episode). Every arm's first-token logits equal; the later tokens part in every arm,
  `EG_LATE`'s too, which changes no arithmetic - this decode is not reproducible past its first token run to run.
- **The follow-ups' idle drive**: 3.5-5.2 s of each follow-up's 15-22 s with no read in flight, waiting for a layer's routing
  (each layer's last MoE groups, the next layer's attention and router, the readback) - DSV41_BACKLOG #122.
- **The drive's episodes** (the new line): 42-45 C in every one (its warning point 89.85 C) and 0 pages swapped out; during
  the download's writes (15-20 MB/s) 1-4 an engine run, mostly with 146-277 MB written in the 10 s before; after it, one at
  each engine's load and first cold read (~160-190 GB), with 0-8 MB written - writes are not needed for one, but many of them
  make them far more frequent.
- **Installed**: `build-dsv41/strata-glm` 7333eb4d (566a1b1b kept as `strata-glm.566a1b1b`), the eight switches (the seven and
  `EG_LATE`) on in `strata-dsv41.json` and `strata-dsv41-iq2xxs.json` (56 each) - the user's call, 2026-10-29.

## 60. The thinking's loop guard and rows of numbers (2026-10-30)

**What happened** (the user, 2026-10-30: "IQ2_XXS got cut off again", then "I don't see any looping"): a reply on the IQ2_XXS
pack (a 60,252-token prompt, 60,218 of it reused) ended as `"length"` after 4,608 tokens of thinking. The reasoning look
(`"reasoning_loop_recovery": "stop"`, in strata-dsv41.json since 2026-10-19) read 0.271 at its ninth look, against its 0.25,
and the server's #530 line followed ("the reply reached max tokens while still thinking"), which read as max tokens. That
reply's thinking, read once with the user's leave (the copy deleted after): rows of numbers, drafted and revised. The look's
window held 248 repeated 12-word passages, none with a letter in it (0.000 counting only passages with words), and all but
15 of its last 300 words were new. No loop: the look counts every digit, comma, period and bracket as a word (half of that
thinking's 4,754), so 12 of them in a number array are ~15 characters, and rows that share values match by chance. The
server's other three replies (28,311, 21,741 and 1,554 tokens) read 0.000-0.028.

**The change** (`serve/server.py`): `"reasoning_loop_coverage"` in the config, the share of repeats that ends the thinking -
0.25 when absent, so every other config is as before. Both DeepSeek-V4.1 configs set 1.0, where only a window of nothing but
repeats ends it: a loop of words, of numbers or of one call block held in the thinking, or the periodic tail. A guard's stop
(the thinking's, a call's, one token's) no longer prints #530's line, and the thinking's line names the share.
`serve/test_reasoning_loop_recovery.py` (40 PASS, 6 new): drafts of a table as rows of numbers, each changing four rows of the
one before, read 0.59-0.74 - stopped at 0.25, delivered whole at 1.0. At 1.0 a loop of 80 words (551 characters a period,
past the tail's 64) still stops once it fills the window (~14,800 of the test's one-byte tokens), one of numbers (323
characters) at ~3,100, the tail's own loop at its first look, and 200 copies of one held call block as at 0.25. The key's bad
values stop the start.

**What it gives up:** a loop now ends once it fills the 2,000-word window (plus two of its periods), not a quarter of it -
some 1,500 words later - and one that changes a little each time (a counter in it) may never fill it: it runs to max_tokens
or the client's stop. The one-token guard (256 in a row) and the call guard are unchanged. It takes effect at the server's
next start. A look that counts only passages with words is DSV41_BACKLOG #124.

## 61. The Delta Inference Engine against Strata-X (2026-10-30)

**What it is.** github.com/musicurgy/delta-inference-engine at def776c4 (2026-10-27), cloned read-only to
~/AI/delta-inference-engine (nothing built or run): a Windows port of Xolotl (§3), 75 commits past the Xolotl read on
2026-10-18 (f7c561b7) - Xolotl's own to 10-21 and musicurgy's 50 (10-22 to 10-27). The engine is Xolotl's
`tools/v41/engine` (ds41_generate, ds41_moe, ds41_dense), not strata-glm: one NVIDIA GPU, every expert run on it (the VRAM
cache, pinned RAM over PCIe DMA, NVMe reads), a CPU tier beside it.

**Its numbers and what makes them** (its README; an RTX 3090 on PCIe 4.0, an i5-13600K, 192 GB of DDR5-5600, a 5.4 GB/s
PCIe 4.0 NVMe): writing 41 tok/s at 192 GB and 33 at 128 GB (six chats of up to 1,500 tokens, thinking on), 25-64K-token
prompts at 610 / 410 tok/s, on its default 366 GB "ds4 Q2" GGUF (2-bit experts; ~190 GB of it DeepSeek's Engram tables).
GSQ-RCO IQ2_XXS: 41 / 32 tok/s. What `Start DeepSeek.bat` turns on:
- lossy: route bias 0.28 in decode ("roughly +20% speed for ~1 point of top-1 agreement" on general text), the dense
  weights re-quantized to Q6_K at load (+2.5-8%), top_p 0.95, a 32,768-token thinking budget (the server forces the answer
  after it);
- exact: DSpark drafting (the MTP stack at IQ2_XXS, 4.2 GB, each block's 64 most-used experts; depth 5, the confidence
  head's gate at 0.6) with speculative sampling (`spec_accept`: a draft kept with its probability, else a token sampled
  without it), prompt-lookup drafts of up to 14 tokens in verify passes of up to 16 rows, the compressed KV and the index
  keys on their FP4 grids (`--kv-fp4`), 8 conversations kept in RAM, a follow-up of up to 2,048 tokens lending part of the
  expert cache and getting the same experts back.

**Against this box.** It does not run here: its models are GGUFs (366 GB for the default; the GSQ builds read their dense
weights and Engram from it), the drive has 126 GB free, and it drives one GPU. So the table compares an engine on its box
with ours on this one, not the engines:

| | Delta, its box | Strata-X, this box |
|---|---|---|
| the box | one RTX 3090 24 GB on PCIe 4.0; i5-13600K; 128-192 GB DDR5; a 5.4 GB/s NVMe | two RTX 3080 20 GB on PCIe 3.0 (the rings' copies 6.2 / 8.5 GB/s); Xeon Gold 6148; 125 GiB DDR4 (the CPU pool 62-72 GB/s); a DRAM-less NVMe 93% full (2.0-2.7 GB/s) |
| GSQ IQ2_XXS, writing | 32 tok/s (128 GB), 41 (192 GB): short chats, bias 0.28, Q6_K dense, DSpark | 26.3 tok/s, a new chat's 24,236-token reply; 23.6-41.2 (median 27.4) for eleven replies at 186-197K tokens of context (the user's server, 2026-10-30: bias 0.3 in decode, the dense weights as converted, lookup drafts) |
| a long read | 610 / 410 tok/s (ds4 Q2, 25-64K tokens) | q3q2u: 84K tokens in 140.4 s (601 tok/s), warm, no slow-mode episode (§31); IQ2_XXS not measured |

**Quality** (its README: DwarfStar's 112 official DeepSeek-API fixtures, thinking off, teacher-forced, top-1 agreement
with the API's top token, exact routing; "the test moves by about 0.2 points between identical runs"):

| | download | general | code + maths |
|---|---|---|---|
| ds4 Q2 (its default) | 366 GB | 91.5% | 93.0% |
| GSQ IQ2_XS | 159 GB + the default | 90.0% | 92.9% |
| GSQ IQ2_XXS | 142 GB + the default | 87.8% | 89.5% |
| GSQ IQ1_M | 125 GB + the default | 85.5% | - |
| Unsloth UD-IQ1_S | 233 GB (its own Engram at IQ4_NL) | 89.3% | 91.1% |

IQ2_XXS is 2.2 / 3.4 points under IQ2_XS there, and IQ2_XS was under q3q2u on our scale (§51: wikitext KLD 0.284 against
0.133). IQ2_XXS's own KLD (`kld_iq2xxs.sh`, ~13 minutes, the server stopped) is not run.

**UD-IQ1_S above GSQ IQ2_XXS with fewer bytes** (its five shards' headers, read 2026-10-30): its routed experts are 115.2 GB
at 1.70 bpw - gate/up IQ1_S in every layer but 30-34 and 39 (IQ2_XXS), the downs IQ1_S in layers 0-19 and IQ2_XXS in 20-39 -
against our IQ2_XXS pack's 132.1 GB at 1.94 bpw, whose downs are IQ1_M in 28 layers (the other tensors: attention Q6_K / Q5_K,
the shared experts Q6_K / Q8_0, the Engram tables IQ4_NL, 111.7 GB). Neither the dense weights nor the Engram explain it (Delta:
its own IQ4_NL tables 89.3% against DeepSeek's exact 89.0%). What does: the downs given the bits where ISTA's RCO takes them
away, and Unsloth's rounding - dsx (`FINDINGS.md:196-209`) measured UD's experts losing 0.99-1.14x from calibration text to
general text (its own GPTQ experts 1.2-1.7x), keeping their outputs' scale (0.92-0.99), and ggml's stock quantizer with UD's
exact layout and Unsloth's published imatrix 9-33% worse, sharing 6% of UD's scales and 71% of its grid indices. The layout
and the imatrix are public; the rounding is not. One test only (112 short prompts, top-1); no KL of both on one scale.

**What we have already:** the route bias's table follows the VRAM tier after every window's fills and swaps
(`maintain_cache`'s `route_pen_update`: Delta's `STRATA_BIAS_REFRESH`); the packed attention rows (§48: its `--kv-fp4`);
the lent slots given back from the SSD cache (§41, §45: its keep cache); the turn and deep checkpoints (its snapshot stack
and one-token rewind); forced tool calls and JSON output; the picture encoder on the GPU (§54: 1080p in 3.6 s; Delta encodes
on the CPU, a 1024 x 701 photo in ~8 s); parking (§57: built, off).

**Why DSpark pays there and not here.** Half of a Delta token is the dense weights on the GPU (Xolotl's 4090: ~19 of ~38 ms
at Q8_0), and under its bias most of the experts are in VRAM; a verify pass reads the dense weights once for all its rows,
so the drafts save most of that half. A q3q2u token here is the GPUs' ~15 ms, the CPU pool's ~16 and the drive's 9-12
(§43), and the pool and the drive pay for every row's experts: a window of 3 reads ~2.5x one token's experts for ~2.1
tokens (#80); warmed, w2 and w3 decoded as fast as plain (§11: 20.81-21.02 tok/s).

**What it adds to the backlog** (DSV41_BACKLOG.md):
- #126, parking on: Delta measured an agent switching between ~8K-token conversations at 1-3 s a switch instead of 16-18 s
  (a 9-turn session 95 -> 49 s). Here a conversation interrupted by another one reads its 200-600K tokens again. §57's
  `serve_park.py` is the check before the switch goes on.
- #125, lookup windows past 6: Xolotl's 1,200-token edit (greedy) went 32.0 -> 35.9 tok/s with 14 drafts instead of 6
  (1,040 of 1,077 accepted in 86 passes). Ours stop at 6 (`--lookup-window`, at most kTmax 8: the window's GPU kernels
  take at most 8 rows), and the draft policy takes a longer window only where its measured tokens per ms are better. Here
  each extra row costs ~3/4 of a token's pool and drive work, so past ~8 rows the estimate is no gain even at 96%.
- #18 (the window cut by DSpark's confidence head): Delta runs it by default (0.6, depth 5).

**Not for this box:** the experts' path over PCIe (our links 6.2 / 8.5 GB/s, the pool 62-72), the Q6_K dense weights (lossy
here: #10, same top -1.03 points with the bias), the 2-bit experts, the thinking budget, top_p 0.95, the Windows settings.

## 62. q3q2u as fast as it goes, nothing lossy: the round of 2026-10-30

**The goal** (the user, 2026-10-30): "Let's make the reliable quant run as fast as possible WITHOUT quality regressions" -
q3q2u stays the pack; every change exact (the same bits, or a different place to read the same bytes from), each opt-in and
measured before the config takes it.

**Parking on (#126, §57).** `checks/serve_park.py` on the installed engine (7333eb4d), the drive in its slow mode for part
of the run (0.89 GB/s through the first read):

| arm | A (16,013 tokens) | B (10,014) | A's follow-up (16,040) | B's follow-up (10,039) |
|---|---|---|---|---|
| U: A alone | read 135.4 s | - | resumed at 16,011, 53.4 s, a9d3230d302fa6c1 | - |
| P: parking (4,096 MiB) | read 59.5 s | read 54.8 s, A parked: 64 MiB in 63 ms | A put back in 7 ms, resumed at 16,011, **7.4 s**, **a9d3230d302fa6c1** | resumed at 10,012, 2.9 s |
| R: no parking | read 56.8 s | read 46.6 s | read from 0, **51.2 s** | - |

A parked conversation costs ~4 KB a position here (rows 14 MiB, the live end's state 25 MiB, its checkpoint 25 MiB at 16K;
§57's estimate for 600K: ~1.2 GB). R's follow-up logits differ (953fc4f13c2d6394): read in one pass from the start, not
resumed at the checkpoint - the prompt path's chunks against decode windows, rounding only, as any read from the root.
`--conversation-cache-mib 4096` is in `strata-dsv41.json` and `strata-dsv41-iq2xxs.json` (the slots, the 8,192-token
minimum and the 8 GiB free-RAM floor at their defaults); it takes effect at the server's next start.

**`--threads 19` (#69): no.** `serve_decode_ab.py` (the plan's question, 256 greedy tokens a pass, exact routing, one tier: 1,700
slots, the 30 GiB test cache - a tier this prompt finds only 21-24% of its routings in, 13-17 experts a token from the drive),
the same first-token logits in both arms (242e9e629628b0f5). The CPU pool's own part of a token (its time less the drive's):

| pass | 20 threads | 19 threads |
|---|---|---|
| 1 | 119.5 - 79.6 = 39.9 ms | 123.2 - 81.3 = 41.9 ms |
| 2 (the drive's slow mode in both) | 384.4 - 343.2 = 41.2 ms | 431.2 - 388.1 = 43.1 ms |
| 3 (... in the 19-thread arm) | 103.5 - 64.6 = 38.9 ms | 388.2 - 346.5 = 41.7 ms |

~5% slower with 19 in every pass: the 20th worker's bandwidth is worth more than its core. 20 stays.

**Prompt-lookup windows of 8 (#125): on.** `serve_decode_ab.py` (`_ARGS=` gives an arm its own engine flags,
`DECODE_QUESTION` another question), the same tier, exact routing:

| answer | `--lookup-window` | tok/s, passes 1 / 2 | windows for 256 tokens | drafts kept |
|---|---|---|---|---|
| copies the text ("Repeat the first four paragraphs ... word for word") | 6 (Strata's rule) | 7.20 / 7.93 | 60 / 90 | 196 / 166 |
| the same | **8** | **7.49 / 8.26** | 40 / 44 | 216 / 212 |
| the plan's question (few drafts) | 6 | 7.40 / 2.48 (an episode) / 8.40 | - | - |
| the same | 8 | 2.45 (an episode) / 8.01 / 8.50 | - | - |

+4% on the copying answer with the same greedy tokens in both passes and both arms (the first-token logits equal in every
arm); the draft policy took longer windows where they paid and otherwise ran as before. This tier reads 20-22 experts a token
from the drive, so a window's extra rows cost more here than in the server's warmed tier: 4% is the floor of what it gives
there. `--lookup-window 8` is in both DeepSeek configs (the engine's most; past it, #125).

**The F32 weights that hold BF16 values (#100's first item), `STRATA_DSV41_BF16_W=1`.** The routers are BF16 in the dense
GGUF, and the engine kept an F32 copy of each beside it (40 x 7.9 MB: 315 MB, the GGUF's BF16 left unused in the arena); the
compressors' kv and gate, the index keys' projection, the indexer's q_b and its weights' projection are F32 in the GGUF with
every lower half zero (checked: 0 of 2,621,440 / 2,621,440 / 5,242,880 values on layer 2's, and on every value at each load).
With the switch they stay BF16 on the GPUs and `Stage::gemv_dt` reads them: a decode window through `gemv_bf16_w` (gemv_f32_w's
two kernels with each weight widened as it loads - the same multiply-adds in the same order), a prompt sub-batch by widening
the weight into a per-GPU scratch (21 MB: the largest, q_b) and the same cuBLAS SGEMM on it. The same bits either way, ~280 MB
fewer read a decode token, ~440 MB of VRAM freed less the scratch.

**The follow-ups' idle drive, measured again (#122).** `serve_short_warm.py` at the config as it stands (60 GiB cache, the
adaptive tier, the bias), six follow-ups of 414 tokens, no slow-mode episode: each read 15.1-21.3 s, 2,174-3,059 records from
the drive, the drive busy 68-73%, and **3.1-5.2 s of each (17-24%) with no read in flight while a layer's routing came**. A
guess before the routing could fill it only as well as it guesses: on that run's routing trace (`STRATA_ROUTE_TRACE`, replayed
through `prompt_cache_sim.py`'s tiers), the top 20 of a layer's non-resident experts by the previous request's counts were
11.1 routed / 5.2 not, by a decayed count of the whole conversation 11.8 / 4.8, by the profile 12.0 / 8.0 - of ~88 drive reads
a layer in the replay. The engine's own guess (the next layer's router on this layer's rows, as the decode's read-ahead
guesses) is scored below.

**The engine's guess, scored (`STRATA_DSV41_PROMPT_GUESS_LOG=1`, a measurement only).** In a gated chunk's MoE (one chunk a
call, up to 4,096 rows), layer l's FFN rows normed with layer l+1's FFN norm and put through layer l+1's router and top-k (the
decode read-ahead's guess); when layer l+1 routes, its experts the stream reads from the drive (outside VRAM, the RAM copy and
the SSD cache), the most guessed first. GPU 0's 19 layers, `serve_short_warm.py`, the route bias off:

| read | drive reads a layer | the top 8 / 16 / 24 / 32 guessed: taken | ... of them routed |
|---|---|---|---|
| the first turn (3,009 rows) | 236.8 | 8.0 / 16.0 / 24.0 / 32.0 | 8.0 / 16.0 / 23.9 / 31.8 |
| follow-up 1 (412 rows) | 63.4 | 7.7 / 14.5 / 20.9 / 26.8 | 6.3 / 12.0 / 17.0 / 21.6 |
| follow-up 2 | 65.8 | 8.0 / 16.0 / 23.6 / 30.6 | 6.9 / 14.0 / 19.7 / 24.2 |
| follow-up 3 | 61.3 | 8.0 / 16.0 / 24.0 / 31.5 | 7.1 / 13.4 / 18.9 / 24.0 |
| follow-up 4 | 51.2 | 8.0 / 15.8 / 23.4 / 31.0 | 6.4 / 12.2 / 16.3 / 20.8 |

84% / 83% / 78% / 76% of the top 8 / 16 / 24 / 32 routed over the four follow-ups (history's best, above: 71% of 20) - about 18
of a layer's ~60 drive reads at 24 guessed, which the routing wait (~80-130 ms a layer, ~15-25 records) can hold.

**Reads on the guess (#122), `STRATA_DSV41_PROMPT_GUESS=N` (opt-in).** The issuer, while a gated chunk's layer routing comes,
reads up to N of the layer's guessed drive records (two or more of the chunk's rows guessed to them, the most guessed first)
into SSD cache slots through `prompt_keep_slot` - the routing's read of such a record then waits for that read (`ssd_cached`),
one it does not route stays cached. Exact: where a record is read from changes no result. A first A/B (`serve_short_warm.py`,
the route bias off, A without, B with N = 24; turns 0 and 3 of A met slow-mode episodes):

| follow-up | A | B | B's reads on the guess (routed of them) |
|---|---|---|---|
| 1 | 19.8 s, 3,022 drive records | 19.4 s, 3,036 | 257 (250) |
| 2 | 18.0 s, 2,864 | 17.6 s, 2,811 | 220 (208) |
| 4 | 13.7 s, 2,047 | 13.4 s, 2,027 | - |

-2% and 95-99% of the reads routed - but ~6 a layer, not 24: past the prompt's share the cache takes a read only when 1 / 26
of the chunk's rows route to it (PROMPT_HOT), and few guesses reach 16 of 412 rows. The routing waits stayed 3.0-3.5 s.
Turns 0 and 1 gave the same first-token logits in both arms; from turn 2 each turn's prompt holds the turn before's 48
generated tokens, which part at near-ties run to run (§14), so those are not compared. `STRATA_DSV41_PROMPT_GUESS_HOT=D` gives
the reads on the guess their own divisor; measured next with 8 tokens a turn (every turn's logits comparable).

**`STRATA_DSV41_BF16_W=1`, checked.** `bf16w_parity` (new: gemv_bf16_w against gemv_f32_w on the widened values at the five
shapes, 1-8 columns, 12 draws of three scales, and bf16_widen against the host's widening): 540 of 540 byte for byte, with and
without PRELOAD. `first_bits.py` (the 4,013-token read, 4 greedy tokens) with the slots pinned per GPU
(`STRATA_GLM_STAGE_SLOTS=1142,558` - unpinned, the freed VRAM moves the split of `--expert-slots 1700` to 1,158 / 542, and the
16 experts that change tiers change the logits by the CPU's rounding, #109): off 3278fb7b32159f1e, on 3278fb7b32159f1e, the
same tokens. 27 tensors kept as BF16; the dense arena 7.90 -> 7.79 GiB, and with the routers' F32 copies gone 700 -> 900 MiB
free on GPU 0, 6,365 -> 6,561 on GPU 1 at the same slots: ~396 MiB, ~28 expert slots at the server's sizes.

**... admitted past the share (`STRATA_DSV41_PROMPT_GUESS_HOT=400`: two guessed rows of 412 admit one), 8 tokens a turn**
(`DECODE_N=8`, so every turn's prompt is the same in both arms), A none, C 24 a layer:

| turn | A | C | C's reads on the guess (routed of them) | first-token logits, both |
|---|---|---|---|---|
| 0 (3,011) | 120.6 s (an episode) | 47.6 s | 503 (502) | d62eabebdedd51c2 |
| 1 (412) | 22.5 s, 3,182 drive records, routing waits 5.1 s | 21.9 s, 3,193, 5.6 s | 703 (639) | 4cbed05c48785bd9 |
| 2 | 21.1 s, 3,098, 4.3 s | 21.1 s, 3,132, 5.3 s | 843 (753) | ff263626d7010141 |
| 3 | 18.9 s, 2,717, 4.2 s | 19.9 s, 2,925, 4.7 s | 859 (770) | 5fa692b4d86289c7 |
| 4 | 16.5 s, 2,305, 3.9 s | 106.1 s (an episode), 2,469, 4.5 s | 840 (678) | 7cb256223a3252ec |

Exact on every turn, ~20 reads a layer at 72-98% routed (GPU 0's layers 72-80%, GPU 1's 92-98%) - and no faster: the drive
read 0.3-7.7% more records, and the host waited 0.5-1.0 s longer for the routings. The reads went out as soon as the issuer had
queued the layer before's records, while those were still on the drive, so they took the drive from records that layer's MoE was
waiting for. Next: only into an idle drive (none of the stage's reads in flight), in the stream's order.

**... only into an idle drive, in the stream's order (arm D).** The reads wait until none of the stage's reads is in flight, then
go out by expert id; the rest as C. D's tier held 835 slots on GPU 1 against A's 853 (the slots follow GPU 1's free VRAM, the
desktop's share of it moves run to run), so its logits are not A's - the arms of one tier agreed bit for bit (A / B at 861, A / C
at 853). Its follow-ups 2-4: 20.1 / 18.7 / 15.9 s against A's 21.1 / 18.9 / 16.5 (follow-up 1 met an episode), with 1.5-9% more
records from the drive (3,145 / 2,968 / 2,467 against 3,098 / 2,717 / 2,305). **Not taken:** 1-5% on a follow-up for 2-9% more of
the bytes the drive's slow mode follows. The switches stay in the engine, off; what would change it is the reads' timing - they
are only worth it in drive time nothing else wants.

**Installed** (2026-10-30): `build-dsv41/strata-glm` 833eeb62 (7333eb4d kept as `strata-glm.7333eb4d`). `first_bits.py` with the
slots pinned (`STRATA_GLM_STAGE_SLOTS=1142,558`): 7333eb4d 3278fb7b32159f1e, the new build with every new switch off the same,
with `STRATA_DSV41_BF16_W=1` the same. `serve_smoke.py` with the config as it stands: ALL PASS (ready 41 s, the reply 39.3 s
after the request, the picture's 16.5 s, the BF16 line in the engine's log). Both DeepSeek configs: `--conversation-cache-mib
4096`, `--lookup-window 8`, `STRATA_DSV41_BF16_W=1` (57 switches).

**A fresh expert profile** (the user, 2026-10-30: "Sure, go ahead"): the frozen profile the VRAM tier and the RAM copy start from
is the one learned on 2026-10-20 (12.6M decode routings). `strata-dsv41.json` now has `"expert_profile_save":
"~/AI/Strata-X-data/dsv41-expert-profile-q3q2u-learned.txt"` and `"expert_profile_save_every": 10`: the engine writes
what its tier learned every 10 minutes and at the stop (experts in VRAM first, then by this run's decode routings plus half the
count it started from), and the next start begins from it ("expert profile: the learned one ... in place of the config's") -
expert ids and counts only. The 10-20 file stays the config's `--expert-profile` and the fallback; removing the two keys
returns to it. The IQ2_XXS config is unchanged.

**A decode window's kernels now** (`checks/dec_layers.py`, new: the windows split at `route_publish_kernel` - #112 folded the
doorbell's publish into it - on an nsys trace, `--cuda-graph-trace=node`; `serve_decode_ab.py`'s tier, exact routing, 833eeb62
with the config's switches, BF16 on; ~122 windows): GPU 0 11.35 ms of kernels a window, GPU 1 11.96 (the tracing inflates the
small ones); a layer's work before its hand-off 399 / 433 us busy. A window, us: the dense projections (`native_small_mmvq`) 4,602
/ 5,248 (GPU 1's +646: the head), the VRAM experts 1,514 / 1,410 (beside the CPU pool), wo_a 1,092 / 1,114, the attention 599 / 640,
the CPU rows' combine 380 / 280, the BF16 gemvs 334 / 360, the norms 431 / 433, the quantize 217 / 218, the split mixers' gemv and
reduce 478 / 475, the routing and publish 245 / 246, the rope 160 / 160, the rest ~100 each. What #100 / #92 / #95 can still take
is therefore ~1.5-2.5 ms a token: the head split over both GPUs (~0.3-0.5 ms), the combine as the experts finish (~0.2-0.35 ms),
fusing the norms, quantize, rope and FP8 rounding into their neighbours (~0.5-1 ms over both GPUs).

**#121's traffic model** (`moex/tools/dsv41/long_read_model.py`, the measured rates: 77 GB of layers 0-20 a window today and
110 GB for the tails and the call's fixed reads, 105 GB for layers 0-20 read once, the drive 2.0 GB/s with a 72 s episode per
~170 GB, the rows 80 KB a token moved up and back each layer at 6.26 / ~6.0 GB/s, the GPU's compute 150-300 ms a (layer, chunk of
7,680) - the prompt read in the profile: 1.66 s of kernels on GPU 0 for 2,017 tokens x 20 layers, 9% busy over the read):

| cold read | today: windows, drive GB, time | one window: drive GB, the rows' copies, time | saved |
|---|---|---|---|
| 100K | 2, 264 GB, 4.1 min | 215 GB, 0.9 min, 3.3 min | 0.8 min (19%) |
| 200K | 4, 418 GB, 6.4 min | 215 GB, 1.8 min, 3.3 min | 3.1 min (49%) |
| 300K | 5, 495 GB, 7.6 min | 215 GB, 2.7 min, 3.3-4.0 min | 3.6-4.3 min (48-57%) |
| 552K | 9, 803 GB, 12.4 min | 215 GB, 4.9 min, 4.9-7.2 min | 5.2-7.4 min (42-60%) |
| 750K | 13, 1,111 GB, 17.1 min | 215 GB, 6.7 min, 6.7-9.8 min | 7.3-10.4 min (43-61%) |

Today's column is the drive and its episodes only (the user's 552K resume of 2026-10-28 took longer: ~805 GB read at the session's
1.86 TB/h is ~26 min with the engram rows' waits), so the saved minutes are a floor. One window is bounded by the rows' copies (H2D and
D2H not overlapped: ~half that with two copy engines at once) and the GPU's compute; the drive reads ~3.7x fewer bytes, ~1.3 episodes
against ~4.7 at 552K. The rows of a 750K read (61 GB) fill the SSD cache's 60 GiB of pinned blocks, which a single pass needs no
records in - past that, two windows. **Go**, as a project of its own: the layer-major path's windows, loans, rings, tails and
checkpoint cuts with the rows in host memory; what to measure first is the GPU's compute a (layer, chunk) at long context (one
profiled 70K read).

**#121's one uncertain input, measured** (`first_bits.py PROMPT_N=70000` under nsys, kernels and copies only, the installed
engine, the slots pinned): the 70,011-token cold read in 12 chunks of up to 6,144 in windows of 8 - 184.5 s, 230.2 GB from the
drive (1.22 GB/s, busy 81%; one slow-mode episode, 69.9 s), the GPU 12.8 s blocked on the engram rows (their fetches 103.8 s on
their worker, beside the layers: the serve line's "the engram rows" is that fetch time, not a wait). GPU 0's kernels 51.9 s over the read's 144.8 s (36%
busy): MMQ 30.2 s, the attention 8.7, the group copies 3.2, the indexer's scores 1.58, the rest ~8 - **35.9 us a token and
layer**, plus the indexer, which scores every earlier compressed position (1.58 s at 70K, ~98 s at 552K). H2D into GPU 0 193.6 GB
at 6.23 GB/s (every non-resident record of layers 0-19, once a window: two windows), D2H 4.1 GB at 6.53 GB/s; GPU 1 10% busy.
The model with it (`long_read_model.py`):

| cold read | today: drive GB, time (GPU 0's compute) | one window: the rows' copies, time | saved |
|---|---|---|---|
| 100K | 264 GB, 4.1 min (1.3) | 0.5-0.9 min, 3.3 min | 0.8 min (19%) |
| 200K | 418 GB, 6.4 min (2.6) | 0.9-1.8 min, 3.3 min | 3.1 min (49%) |
| 300K | 495 GB, 7.6 min (4.1) | 1.4-2.7 min, 4.1 min | 3.5 min (47%) |
| 552K | 803 GB, 12.4 min (8.2) | 2.5-4.9 min, 8.2 min | 4.1 min (33%) |
| 750K | 1,111 GB, 17.1 min (12.0) | 3.4-6.7 min, 12.0 min | 5.1 min (30%) |

Past ~300K one window is bound by GPU 0's own compute, not the drive or the rows' copies, so the saving levels off at 3-5 min a
read; it also takes ~3.7x fewer bytes from the drive (fewer slow-mode episodes). The model leaves out the engram rows: two
windows waited 12.8 s for them in this read, but one window needs every chunk's rows at layer 1 (§63, step 1: 302 s). Still a
go, but smaller than the first model said.

## 63. One layer-major window for a whole cold read: the plan (DSV41_BACKLOG #121, 2026-10-30)

**What it buys** (§62's model, GPU 0's compute measured): a cold read of 200K-750K tokens 3.1-5.1 min shorter (30-50%), 3.7x
fewer bytes from the drive. Past ~300K GPU 0's own compute (35.9 us a token and layer, plus the indexer) is the floor.

**What limits a window today** (`Impl::prompt`, `Stage::prompt_window`): GPU 0's loan holds each chunk's rows that cross layers
(`LmRows`: the streams R, 80 KB a token; the carried pre-mix, 96 B; layer 20's top-k lists, 2 KB; the candidate masks) and 2 K + 1
copies of the running state (~28 MB each) - 8 chunks of 6,144-7,680. Each window streams every non-resident record of layers 0-19
once into the ring (384 slots: a whole layer), which serves the window's chunks.

**The change** (opt-in `STRATA_DSV41_LM_HOST=1`; the default path byte for byte as now):
1. **The rows in host memory.** Each chunk's `LmRows` in pinned host memory, the GPU holding three staging sets: chunk c runs the
   layer in set c mod 3 while c + 1's rows copy up into the next set and c - 1's copy back from the one before (a copy stream each
   way, events between). The same bytes go up and come back: every (layer, chunk) runs on the rows today's window gives it.
2. **Where the host memory comes from.** A single pass reads each record once into the ring, so the SSD cache's 60 GiB of pinned
   blocks hold no record the pass will want again: the call takes the cache's slots (empties them, as a stop of the cache) and
   carves the rows from them, and gives them back empty at the end - 552K needs ~46 GB, 750K ~62 GB (past that, two windows). The
   RAM copy and the bounce slots are untouched. The cost: the reply after a long cold read starts with an empty SSD cache.
3. **The running state's copies** for each chunk end (the checkpoints there, a stop's way back) in host memory too: 2 K + 1 of
   ~28 MB, 4 GB at 72 chunks.
4. **The engram rows**: today two buffers of a window's rows each (`eg_lm_grow`); one window of the whole call needs one buffer of
   all its rows, sized first (to check: `m.eg->L x rows x row_bytes()` at 552K).
5. **`Impl::prompt`**: with the switch, one window of every chunk when the host memory holds it, else the fewest windows that fit.
   GPU 1 then gets the chunks after GPU 0's whole pass (today it overlaps the next window): ~1 min more at 552K, in the model's
   floor already.

**Checks.** CPU: the staging ring's order (a unit test of the up / run / back schedule). GPU, the server stopped: `first_bits.py`
on a 4,013-token read (one window: the switch changes nothing - the same bits as now) and on the 70,013-token read with its
checkpoint cut and a request resumed at it (two windows today, one with the switch: the same bits as today's
598ddcd7c221f866 with the slots pinned); a stop inside the window (the server's cancel test); then the speed: a 200K and a 300K
cold read, today against the switch, each a fresh engine.

**Effort.** The largest change since the layer-major read itself: the window loop's rows and copies, the cache's slots lent and
returned, the snapshots' home - several days of work, built and checked in steps (1, 3 and 5 first with the cache untouched and
a smaller host pool, to check the bits; then 2 and 4).

**Step 1, written (2026-10-30): the rows in host memory, the SSD cache untouched** (`STRATA_DSV41_LM_HOST=1`). `Impl::prompt` sizes
a window of as many chunks as the loan's copies of the state (`lm_extra_bytes(rows, 3, kw)`: three staging sets, 2 kw + 1 copies)
and MemAvailable less the RAM copy's 28 GiB headroom hold - every cut chunk of the call when they do - and gives each chunk a
pinned buffer `lmh[c]` of [R | the carried pre-mix | the top-k lists | the candidate masks], T rows each part.
`Stage::prompt_window` runs (layer, chunk) p in staging set p mod 3: at the window's first layer the chunk's embedding goes
straight into its set, at every later one its rows copy up on `lm_cs[0]` (after the set's last rows went back), the layer runs
as before, and the rows copy back on `lm_cs[1]` - one stream each way, so a later copy back implies every earlier one, and the
next pass's copy up of a chunk always reads what its last copy back wrote. LM_SPLIT's two passes a layer are off in such a window
(exact either way; it would copy the rows twice a layer). The engram's buffer holds every chunk's rows. The next stage reads each
chunk's [R | pre-mix] where it came back, and with two or more host windows finishes the one before before GPU 0 starts the next
(they share the buffers). The default path is the same code with the window's sets equal to its chunks.

**Step 1, measured** (`first_bits.py PROMPT_N=70000`, the slots pinned, `STRATA_DSV41_LM_HOST=1` with the test-only
`STRATA_DSV41_LM_HOST_KEEP_GIB=16`: the RAM copy fills MemAvailable down to its 28 GiB headroom, so until step 2 the rows find no
room beside it): one window of the 12 chunks (room for 14; 6.73 GiB of pinned rows; GPU 0's loan 9.06 GiB against 11.32) and
**the bits of today's two windows, 598ddcd7c221f866** - exact. 18,941 records streamed (12,452 from the drive) against 25,895
(16,292-16,529), **181.5 GB from the drive against 230.2-236.5 GB (-21%)**. But 442.7 s: GPU 0 stood at layer 1 for ~300 s. One
window needs every chunk's engram rows before its layer 2, and their 4 KB reads queued behind the records' and ran through three
slow-mode episodes (69, 147, 73 s; the guard's floor reads them one at a time): the GPU waited 302.0 s for them (§64).

**Step 2, written (2026-10-30): the rows in the SSD cache's blocks.** The cache keeps each pinned block's slots (`block_first`,
`block_n`); `Impl::cache_lend` lends whole blocks of a chunk's bytes or more - its rows, then its engram rows (`L x TM x
row_bytes`, mapped: the layers read them where they lie) - the least recently used blocks first, never one with a slot pinned
by a decode layer, held, used by the running prompt or reading; their records leave the cache and their slots are out of
`ssd_victim`'s reach until `cache_unlend` gives them back, empty, once every stage's streams, the staging copies and the engram
fetch are done (`LendEnd`, on every way out of the call). `Impl::prompt` sizes a host window from the blocks free to lend and
lends them before the call's own lend copies its VRAM experts into the cache; MemAvailable (with `LM_HOST_KEEP_GIB`) stays the
other source. A host window's engram rows are each chunk's own (`Stage::egc`), fetched as the window starts.

**Step 2, measured on the 70K read** (`LM_HOST=1`, `EG_URGENT=1`, the 30 GiB test cache, the slots pinned): one window of the 12
chunks from 34 lendable blocks (7.75 GiB), **598ddcd7c221f866** - exact; 144.4 s, 180.2 GB from the drive (1.22 GB/s, busy 50%),
no slow-mode episode in the read; the GPU waited 19.3 s for the engram rows (step 1: 302 s), every chunk's in by 40 s. GPU 0's pass
took ~100 s - its kernels ~50 s of it - with the drive idle 64.5 s behind the ring: a window holds one layer's records in the
ring while its 12 chunks run it, and the read-ahead's cache share per layer (`KEEP_SPREAD`) stops it about half a layer on. GPU 1's
pass then ~45 s, drive-bound. Two windows read 230.2-236.5 GB in 184.5-268.4 s, each with one to three episodes.

**One window's cost: every chunk's engram rows at layer 1.** (A first reading here - that in one window GPU 1's 20 layers follow
GPU 0's whole pass - was wrong: on the last stage `TAIL_BATCH` runs layer 20 a chunk, from a ring kept for the call, and layers
21-39 once for the tails at the call's end, in today's windows as in one, so one window loses only layer 20's overlap;
`long_read_model.py` keeps its first floor.) What one window does add: its layer 1 needs every chunk's engram rows. The 200K read in
one window (34 chunks, 18.8 GiB of rows from 34 lent blocks) stood at layer 1 for over 300 s, 25 of its 34 chunks' rows in,
through five slow-mode episodes - stopped there.

**Two sets of host windows (`STRATA_DSV41_LM_HOST=2`).** Windows of up to half the call's chunks - and of the blocks free to
lend - each with a set of blocks, the windows taking turns: GPU 1 runs window w from its set while GPU 0 runs w + 1 in the other
(`Stage::lmh_off`, `lmh_wk`: the window's set in `lmh` and `egc`), and the next window's engram rows are fetched into its set
while the window before runs - so only the first window's rows wait at its layer 1. Layers 0-20's records stream once a window,
but a window holds what the blocks hold rather than GPU 0's 8 chunks. It engages only when that means fewer windows than the
loan's (otherwise today's 8-chunk windows: the same count, with copies added). The model (GPU 0's pass the compute floor, as one
window's; a window's drive bytes as today's, the long read's keep in both): 200K two windows, 4.1 min against today's 6.4 (37%),
300K 4.1 against 7.6 (47%), 552K 8.2 against 12.4 (33%), 750K 12.0 against 17.1 (30%) - one window's minutes, with half its
first wait; 100K: today's two windows already, left as it is. In the 200K read GPU 0 began window 2 at 420 s while GPU 1 ran
window 1's layer 20: the two sets in use together, as designed.

**The 200K read with two sets, measured** (first_bits.py's 30 GiB test cache, the slots pinned, `EG_URGENT=1`): two windows of
17 chunks, 18.8 GiB of rows from 34 lent blocks - 711.0 s, 368 s of it in four slow-mode episodes (147, 70, 74, 77 s); the GPU
waited 170.2 s for window 1's engram rows (through the first two); first-token logits 5c6bb83e329883c1. But 473.6 GB from the
drive: the call lent all 34 of the cache's blocks, so only 344 records came from the cache, and the refill of GPU 0's 1,180 lent
VRAM slots read 1,008 records again - the lend left the cache no room for the slots the call's own lend copies into it, nor
for a later window's re-reads. **Fixed: at most half the cache's blocks lend** (`cache_lendable`): the server's 60 GiB, ~70
blocks, lends up to ~35 (one window of 215K tokens, or two sets of 17 chunks).

**... against today's windows** (the same 30 GiB test cache, a fresh engine, the engram tables dropped first): the same first-token
logits, **5c6bb83e329883c1** - exact.

| the 200K cold read | time | in slow-mode episodes | drive | records from the SSD / its cache | engram wait | "the ring" idle |
|---|---|---|---|---|---|---|
| today: 5 windows of 8 chunks | 593.7 s | 292 s (4) | 545.0 GB | 38,033 / 10,699 | 122.2 s | 85.3 s |
| two sets of 17 chunks | 711.0 s | 368 s (4) | 473.6 GB | 32,650 / 344 | 170.2 s | 193.2 s |

Outside the episodes 302 s against 343: fewer bytes, but a window of 17 chunks gives GPU 0 ~3.7 s of compute a layer against
~1.5 s of the drive's, and the read-ahead stops about half a layer on (`KEEP_SPREAD`'s share a layer, the bounce slots), so the
drive idles behind the ring (193 s against 85) - while 8 chunks a layer kept the two near even. The bytes saved would pay in
fewer episodes, and on this day the drive met one every few minutes whatever was read: 4 in each arm. **Not taken into the
config**: opt-in, exact; what it needs to pay is a read-ahead that keeps the drive busy several layers ahead in such a window
(the kept slots released a layer at a time, as the last window's are) and a drive in its usual state to measure it on.

**A one-window call's read-ahead, deeper** (written after it). In a call read in one host window - its only window is its last,
whose kept slots `lm_release` already gives back a layer at a time - the read-ahead may take the least recently used cache slot
past the prompt's share (a free one first), up to 3 layers' slots held at once (`RecordIo::deep_cap`, `dpin`, `deep_n`;
`prompt_keep_release` returns a layer's), and its queue goes to 16 x the bounce slots. The same records into other slots: exact.
The 70K read with it (the 30 GiB test cache): **598ddcd7c221f866**; 281.3 s with two episodes, one on the engram fetch at layer 1
(153.8 s waited); "the ring" 72.9 s against step 2's 64.5 - the read-ahead took 137 slots past the share of 1,152 allowed. The test
cache had no room for it: of its ~2,330 slots 768 were lent and the call's own lend held 1,174 for the refill, ~390 left; the
server's 60 GiB leaves ~2,700. The 4,013-token read with no switch, the same build: 3278fb7b32159f1e, the installed engine's.

**One window at the server's 60 GiB** (the 200K read, `LM_HOST=1`, `EG_URGENT=1`, `CACHE_GIB=60`): one window of the call's 34
chunks (37 cut, 20.5 GiB of rows from 37 lent blocks) - **5c6bb83e329883c1**, exact; but 893.9 s against today's 593.7 s (at 30
GiB), 427.3 GB from the drive against 545.0, and ~570 s of it in six episodes; the GPU waited **542.5 s** for the engram rows -
the episodes landed on the fetch every chunk's layer 1 waits for - and "the ring" idled 236.8 s. Fewer bytes, but no fewer
episodes on this day, and one window puts the whole call's engram fetch before its layer 2. **Verdict: `LM_HOST` stays opt-in
and off** (both forms, the lend and the one-window read-ahead with it); the baseline arm at 60 GiB was not run - it could not
change this. What one window would need first is an engram fetch that does not wait for the drive at layer 1.

## 64. The engram rows' fetch, and what bounds a short follow-up (2026-10-30)

**The engram fetch's rate, without the engine** (`moex/tools/dsv41/eg_fetch_rate.cpp`: N tokens x 2 layers x 24 rows uniform
at random over L01 / L14's weight tables, the tables dropped from the page cache first, the scales left out - they stay cached;
one run each, the server stopped):

| mode | 8,192 tokens alone | 4,096 tokens, 12 record readers meanwhile |
|---|---|---|
| mapped, 1,024 rows asked ahead (the config's) | 2.82 s, 139.6K rows/s | 13.3-17.0 s, 11.6-14.8K rows/s |
| O_DIRECT, 32 threads, the copies' order | 72.3 s, 5.4K/s (a slow-mode episode) | - |
| O_DIRECT, 32 threads, sorted and deduplicated | 3.02 s, 130.3K/s | - |
| ... 64 threads | 93.2 s, 4.2K/s (an episode) | - |
| ... 128 threads | 1.97 s, 199.5K/s | 29.9 s, 6.6K/s |

Alone, the mapped path is as fast as O_DIRECT; behind the record stream (12 readers of 4 MiB pieces, ~384 requests) the rows'
4 KB reads get a tenth of that, in proportion to their share of the queue. In the drive's slow mode, 4 KB reads ran at 5.4K a
second 32 at a time (6 ms each) and 4.2K at 64 (15 ms) - against ~1K a second one at a time, the guard's floor; a deeper one
served ~1K a second at 30 in flight (4 MB/s, the 200K reads of §63).

**The engram reads first while a GPU waits (`STRATA_DSV41_EG_URGENT=1`, opt-in, exact).** While a prompt stage is blocked on
its chunk's engram rows (`Impl::eg_await`, counted in `eg_need`), the streams' issuers start no record read from the drive
(`eg_hold`: their read-ahead stops, a record that needs the drive waits; the reads already queued finish), and the guard's
floor lets the fetch keep 32 pages in flight, not 1. Unlike `EG_FIRST` (§17: the records held through every fetch, even
while the GPUs needed them - 650 s against 204.5 s), the records wait only while a GPU does. The 70K read with it (two windows,
the slots pinned): **598ddcd7c221f866, today's bits**; the GPU waited 17.6 s for the rows (12.8 s without: the profiled read of
§62), their fetches took 43.5 s on their worker (103.8 s), the records were held 3.8 s; the read took 268.4 s, 228 s of it in
three slow-mode episodes (one, 70 s, in the read without). On today's two windows the wait is inside the episodes' swing; what
it is for is one window (§63), whose layer 1 waits for every chunk's rows. The 200K read on today's windows with it (the 30 GiB
test cache): **5c6bb83e329883c1**; 769.0 s against 593.7 s, the GPU 237.6 s on the engram rows against 122.2, the records held
80.6 s, 542.0 GB (545.0) - ~345 s outside its episodes against ~302. The long waits fell in the episodes, where the drive serves
~1,000 requests a second whatever is held back: holding the records there only idles their stream. **Not taken: opt-in, off.**

(§62's "103.8 s of it waiting for the engram rows" was the fetches' time on their worker - the serve line's "the engram rows" -
not a wait: the GPU waited 12.8 s. Corrected there.)

**What bounds a short follow-up: GPU 0's link.** The follow-ups' idle drive (§62, #122: 17-24% of each with no read in flight
"for the routing", nearly all of it GPU 0's) profiled: `serve_short_warm.py` (a 3,000-token first turn, ~400-token follow-ups,
8 tokens a reply) under nsys, the installed engine, the drive healthy through the follow-ups. Each layer cut at its routing's
readback (412 x 6 ids), its uploads and kernels (`checks/nsys_uploads.py`):

| a follow-up's layer | GPU 0 (layers 0-19) | GPU 1 (layers 20-39) |
|---|---|---|
| time | 310-700 ms (8.6 s for 20) | 510-840 ms (11.4 s for 17) |
| uploads | 1.9-3.7 GB, the copy engine busy 95-99% at 6.26 GB/s | 1.8-2.9 GB, busy ~30% at 12.2 GB/s |
| kernels | 31-39 ms | 28-35 ms |
| last upload to the next routing | 6-9 ms | 6-10 ms |

The routing comes 6-9 ms after a layer's last upload on both GPUs; what differs is the uploads. GPU 0's layers were mostly SSD-
cache hits (the first turn's reads filled the cache's free slots in stream order, layers 0-19 first: `CACHE_FREE`), so its x8
link set their pace while the drive idled; GPU 1's were mostly drive reads, its x16 link idle 70%. Follow-ups 23.1 / 22.1 s,
GPU 0's idle drive 6.1 / 5.3 s.

**... once the replies churn the cache.** The same conversation with 400-token replies (four follow-ups, no profiler): 17.1 /
39.9 (an episode) / 15.8 / 13.4 s, 2,815 / 2,854 / 2,712 / 2,227 records from the drive, GPU 0's idle drive 2.2 / 1.7 / 1.5 /
1.5 s; the replies 20.9 / 5.1 (the episode) / 23.1 / 25.6 tok/s. The decode's misses refill the cache across both GPUs' layers,
so the bias fades; the episode cost that turn ~86 s.

**A cache tilted to GPU 1's layers: not taken.** `prompt_cache_sim.py` with a time model (`POOL_W0=w`: GPU 0's layers' bytes
weighted by w in the pools' sizes, as the engine's start would size them; a prompt layer's time the larger of its uploads over
its GPU's link and its drive reads at 2.6 GB/s), on a trace of that conversation (8 follow-ups, 300-token replies;
`STRATA_ROUTE_TRACE`): the model's follow-ups 16.9 / 17.1 / 16.1 s against the engine's 18.0 / 16.9 / 15.6.

| pools | the 8 follow-ups' layers | the decode's drive reads a token |
|---|---|---|
| as today | 129.8 s | 4.64 |
| GPU 0's layers weighted 0.8 | 127.8 s (-1.5%) | 5.05 (+9%) |
| ... 0.6 | 128.1 s (-1.3%) | 5.92 (+28%) |

A follow-up already runs at the larger of the two per layer; what the tilt saves on GPU 1's drive reads GPU 0's layers read
back, and the decode, which routes evenly over both GPUs' layers, pays for it.

**The read-ahead's depth: not the limit.** `DEEP_AHEAD`'s queue at 16 x the bounce slots instead of 4 x (a check, reverted): the
70K read 598ddcd7c221f866, 191.7 s with two episodes, "the ring" 24.0 s with no read in flight against 17.6-18.8 s. A long read's
share of the cache is split evenly over its layers (`KEEP_SPREAD`), so a layer's read-ahead runs out of slots about half a layer
in, whatever the queue may hold.

**What is left, closed (2026-10-30).** Each lead this round named, with the number that closes it:

| lead | the number | verdict |
|---|---|---|
| A CPU share in a follow-up's layers (the pool computes lightly routed host-held experts, off GPU 0's link) | `prompt_cache_sim.py SHARE=1` on the traced conversation: the 8 follow-ups 129.8 -> 126.5 s (-2.5%), 1,612 experts to the pool - a steady follow-up is drive-bound | not built: new q8_1-activation CPU kernels (the pool's ggml-cpu kernels round activations as Q8_K, coarser than MMQ's q8_1) for 2.5% |
| The split moved toward GPU 0 (more layers on GPU 1's x16 link) | `SPLIT=15` / `10` in the time model: 129.8 -> 128.5 s (-1%) for follow-ups. And the engine will not place it: `--layer-split 15` and `10` start as "layer split 15 moved to 20 (V4.1's shared compressed rows stay on one GPU)" - layers 0-20 attend over the compressed rows layers 2, 8 and 14 write, which with the peers off cannot be shared across the cards | closed by the model's layout, not by effort |
| The engram's 4 KB reads merged into 128 KB ones during an episode (requests, not bytes, bound it there) | an episode serves 4 KB in ~1.0 ms and 128 KB in ~2.2 ms; an 8-chunk window's ~600K distinct rows a layer over 98 GB lie ~164 KB apart, so a 128 KB read holds ~1.8 of them: 1.2 ms a row against 1.0 | no gain on today's windows; it pays (~1.8x) only in windows of ~30 chunks, which lose on other counts (§63) |
| The VRAM experts' K-quant decode kernels (#40) | the decode profile: both GPUs in `doorbell_wait_kernel` 65-76% of a window, waiting for the host and the CPU pool | not taken: the GPUs' own kernels are off the decode's critical path |
| DSpark at w2 (its drafter instead of the prompt lookup's: `lk_on` requires no DSpark, #4) | `serve_decode_ab.py` as the server runs (the adaptive tier, the 60 GiB cache, three passes of 256 tokens): **11.38 / 11.27 tok/s against 12.91 / 13.96** on the measured passes (-15%, past the +-6% noise below); its windows read 8.9-9.4 experts a token from the drive against 6.7-7.7, and its drafter's 0.7 GiB of dense weights cost GPU 1 64 slots (817 against 881) | slower: closed (as the user's w3, 2026-10-29) |

| A long read's drive idle on today's windows (the ring full while GPU 0 runs a window's layer) | the 200K read at the server's 60 GiB (today's windows): 854.8 s, 8 episodes (574 s), 525.8 GB (36,638 records from the drive, 23,244 from the cache against 10,699 at 30 GiB) - outside the episodes ~281 s, ~1.9 GB/s against the drive's healthy 2.3-2.6: the drive already ~75-80% busy there, so ~60 s (~7% of the read) is the most any overlap could take back; the deeper read-ahead (§63) and the bigger windows aimed at it and lost; 5c6bb83e329883c1 | closed: ~7% ceiling, the attempts measured slower |

| A wider prompt batch (#8, `--prefill-chunk 12288`; set aside 2026-10-21 for a KLD check) | the 70K read with it: "12 chunks of up to 6144 tokens in windows of 8" and **598ddcd7c221f866**, today's bits - `choose_chunk` takes the largest chunk whose buffers fit every stage's loan, 6,144 at the 1M context, so the flag falls back; 262.0 s with two episodes (139.5 s), 230.7 GB. Its estimate (-15-19%) came from the chunk loop, where each chunk streamed the layers; a layer-major window streams them once for its chunks. What is left of it is MMQ's batch a (layer, expert) - ~96 rows at 6,144 - in the read's compute-bound part, with the drive ~75-80% busy outside episodes | closed: it does not fit, and its gain left with layer-major |
| A long read's work moved from GPU 0 to GPU 1 | the split is the model's (the row above: layers 0-20 on one card). And the bound, from the 200K read at 60 GiB: ~281 s outside its episodes, against the drive's own floor for its 525.8 GB (~202 s at a healthy 2.6 GB/s) and GPU 0's compute (35.9 us a token and layer, §63: ~144 s for 200K x 20 layers, beside the reads) - with every layer's compute gone the drive's 202 s stays, so ~79 s (~9% of the 854.8 s) is the most any move of compute could take back | closed: the layout forbids it, and the drive holds 91% of the read either way |
| More experts in VRAM | `tier_sim.py --engine-swaps` (new: the swaps as glm_tiers.cpp makes them with `SWAP_CACHE`, below) on the traced conversation, today's 1,176 + 881 slots: VRAM 166.0, RAM 24.1, the cache 48.3, the drive 1.5 a token - the engine's own decode lines on the follow-up runs: 69% of the routings in VRAM, 1.54-1.69 from the drive. +5% slots (1,235 + 925): 168.8 / 1.4; +10% (1,294 + 969): 171.6, 22.2, 44.8, 1.3 - at the decode's own costs (the pool's ~0.22 ms an expert from RAM or the cache, ~5.3 ms a drive record) ~2.3 ms of ~42 (~5%) for ~3.6 GiB that the 1M context's rows, the display's 2 GiB and the dense weights hold | closed: no VRAM to give it |
| A fixed VRAM tier (`--no-adaptive`) instead of the swaps | the same run: VRAM 149.9, RAM 35.3, the cache 53.2, the drive 1.5 - 16.1 routings a token more for the CPU pool (~3.5 ms) and no fewer drive reads. (`tier_sim.py` without `--engine-swaps` gave the adaptive tier 3.6 drive reads a token: it dropped a swap's victim and kept the incoming record's cache copy beside its VRAM one, which the engine does not since §44; the engine's lines read 1.5-1.7) | closed: the swaps win |
| The adaptive rule (heat decay, the round, its budget) | the same run over decay 0.5 / 0.7 / 0.85 and a round every 2 / 4 / 8 tokens, budget 7 / 14 / 28: the best (0.85 every 2 tokens) 169.1 VRAM routings a token against 166.0, for 7.2 swaps a token against 4.0 - each an 11-18 MB copy over a GPU's link out of the DRAM the pool drains at 65-71 of the 72.8 GB/s; the oracle profile (the evaluated text's own counts) 166.4 | not taken: ~1.6% before the swaps' own traffic, under the decode's noise |
| More accepted tokens a window | the prompt lookup on at 8 (+4% on a copying answer, the same tokens, §62); a window holds at most 8 (the engine's `kTmax`), and past 8 the lookup's drafts are estimated at <=1%; DSpark at w2 -15% (above), w3 no gain (the user's, 2026-10-29) | closed |

**The decode's own noise** (the split arms above ran the same layout, so three runs of one config): the measured pass 14.24 /
13.45 / 12.60 tok/s - +-6% run to run; a change under ~10% does not show in one A/B of this kind.

**A decode token, measured part by part** (the 400-token replies after warm follow-ups, `serve_short_warm.py`, 25.6 tok/s -
39 ms a token): the GPUs' chain (attention, the dense Q8_0 rows, the router, the VRAM experts, two host round trips a layer)
16.9 ms - and in the decode profile both GPUs wait in `doorbell_wait_kernel` 65-76% of a window, so their own kernels are
not what it waits on (the small ones ~1-1.2 ms a token in all, §62); the CPU pool's ~66 experts from RAM 14.3 ms, draining
65-71 GB/s against the 4 channels' measured 72.8; the drive's misses 5.7 ms (1.1 records a token: the cache's fit, which the
profile save of the q3q2u config refines over the user's sessions); the tier's upkeep, the head and the sampling 2.2 ms. The
parts run in the order the model's data does - the routing, the pool beside the VRAM experts, the combine, the next layer's
attention - so what is left is accepted tokens a window (the prompt lookup on; DSpark at w2 -15%, above) or more experts in
VRAM, which is full (+10% slots ~5%, the swaps' rule at its ceiling: the table above).

The decode is bound by its chain a layer - the routing to the host, the CPU pool's experts from RAM at 63-71 GB/s (near the 4
channels' 72.8), the combine - and by the drive's misses; a short follow-up by the drive, with GPU 0's link the larger only in
the few layers after a cold read; a long cold read by the drive and its episodes (read-disturb reclaim, §14). Every exact lever
the round found is in, measured and left off, or closed above.

## 65. A long read's pinned buffers given back when it ends (2026-11-01)

**The report.** The user's "used" (GNOME System Monitor: MemTotal - MemAvailable) climbed from 113.8 GB during a 140K read
(10-31) to 118.6, 120.6 and 122.0 GB the next day (12:02, 13:20, 14:23), the server up throughout. To 12:02 it was the
desktop's programs opened since (Chrome 2.4 GiB, Firefox's content processes 0.7, gnome-system-monitor itself ~49 MiB an
hour); the engine was flat. The step at 13:17 was the engine: dsh compacted the conversation at 843,424 tokens (its point:
95% of the 888,000 window), and the next request, 785,857 tokens, parted from it before token 131,072, so the engine read
720,321 tokens from the 65,536 checkpoint (2,549.7 s). In that read:

- the layer-major buffers grew to its windows (7 chunks of 9,363 tokens): `h_lm` 2 x 0.77 -> 5 x 0.72 GiB, `eg_lm_h`
  2 x 0.71 -> 2 x 0.77 GiB, `h_hand` 2 x 0.77 GiB (grow-only, from an earlier read) - the prompt path's pinned host memory
  4.83 -> 6.99 GiB, held until the engine stopped (`Impl::prompt` grew them; nothing freed them);
- parking (§57) took the old conversation: its positional rows (448 + 296 MiB: GPU 1's block is 368 B x its ~844K tokens),
  its running state (25 MiB) and the ~17 checkpoints the new prompt does not share (~25 MiB each), ~1.2 GiB in all;
- the kernel swapped out 1.7 GiB of the engine's cold heap.

**The change (on; `STRATA_GLM_KEEP_PROMPT_BUFS=1` keeps them as before).** `read_bufs_release` (glm_session.cpp), called in
`Engine::run_request` once a request's read is over, before its decode: each stage's prompt streams synchronized (as the
loan's end waits for them) and its engram fetch waited out (`eg_lm_drop`), it frees `h_lm` and `eg_lm_h`, and `h_hand` when
it holds more than 4,096 rows (a follow-up's chunk keeps its own). `h_cx` and `eg_pr_h` stay (0.19 + 0.12 GiB). The next long
read pins them again, sized as `Impl::prompt` always sized them; the answers do not change.

`first_bits.py`, the slots pinned (`STRATA_GLM_STAGE_SLOTS=1142,558`), a 70K read, then the first 88,000 tokens with another
question (resumed at the 65,536 cut), the change on and off:

| | given back (the change) | kept (`KEEP_PROMPT_BUFS=1`) |
|---|---|---|
| the 70,013-token read | **598ddcd7c221f866** (today's bits), 404.8 s | **598ddcd7c221f866**, 187.4 s |
| then given back | 4.91 GiB in 1,040 ms | - |
| the resume: 22,478 tokens read at 65,536 | **928ec34d8f57de40**, 126.7 s (a 71.7 s slow-mode episode) | **928ec34d8f57de40**, 53.2 s |
| then given back | 2.46 GiB in 452 ms | - |

The same four tokens in both. The release costs ~1 s before a long read's first token (0.26% of that 70K read); pinning the
buffers again in the next read is under the noise (the resume 55.0 s outside its episode, against 53.2 s). After a read like
the 720K one above it gives back ~6.7 of the 7.0 GiB (by the sizes above). Installed: build-dsv41/strata-glm 8a84b4cd
(833eeb62 kept).

**Found on the way.**

- The server's conversation-cache card (`/metrics`) counts parks from "conversation cache:" lines in the engine's log, which
  is /dev/null with `"diagnostics": false`: it showed parks 0 with the park above done (DSV41_BACKLOG #130).
- A client's compaction leaves a record that is never resumed: the compacted prompt parts from the old conversation early, and
  the old one's record (~1.2 GiB at 844K, its checkpoints in it) stays until the budget (4 GiB) drops the oldest (#131).
  Parking stays as it is (the user's call).
- RSS undercounts pinned memory: reclaim unmaps the pinned `/dev/zero` pages and puts them back on the active list (`h_hand`
  1.55 GiB mapped, 98 MiB resident). Sum the mappings' sizes; meminfo's Shmem matched them (7.58 against 7.48 GiB).
- With the server stopped "used" read 13.6 GB, none of it the engine (no process, Shmem 71 MB, GPU 0 at 15 MiB): the programs
  6.7 GB, the kernel's labelled memory 1.2, its reserve (watermarks, lowmem reserve) 1.2, and 4.45 GB in no meminfo field -
  the same with the engine up. Of that, `irdma` (the X722's iWARP RDMA: no carrier on either port, no users) held 3.0 GiB
  (`rmmod irdma`: +3,104 MiB available), and nvidia_uvm's struct pages for the cards' VRAM 632 MiB (10,354,688 pages x 64 B).

## 66. What upstream Strata and Maya had for this box: the audit of 2026-11-04

**The read.** Upstream Strata from our last merge (a0116b27, v0.1.43.1) to ffa80d73 (v0.2.10): 3,745 commits, among them
Project Maya's DeepSeek-V4.1 engine imported as `src/dsv41` on 10-20 (Maya v1.1.17) and 314 commits on it since. Maya from
v1.1.27 (4615f34) to v1.1.53 (c01aa8e): 467 commits, 155 on `src/ds4`. Three read-only reviews, each candidate then checked
in our code; nothing built or run of theirs. No upstream or Maya change touches V4.1's forward math against model.py. Not
merged as a whole: upstream's serve now has a `deepseek41` path of its own (request keys strata-glm refuses, other
switches) and defaults the user declined (a 32K thinking cap, an answer reserve, an optional API key, tool-call recovery);
the items below are ported by hand, each exact - the model reads the same tokens and gives the same bits.

**The plan.** Given to the user before any code was written (the user's goal: everything from the audit worth taking,
nothing that changes the model's input or output, minimal tests): the server's prompt memo, the fence fix and the metrics
guard; the engine's Ctrl+C handling, #53's restart half, the short-read limit, the two Maya fixes, the PLE fix; skipped: a
client's `reasoning_effort: false`, the 503, the RAM copy's order; the checks: an equivalence test of the tokenizer's path and
the serve suite, one GPU run (the 4,013-token bits, a simulated Ctrl+C, a restart against an uninterrupted engine). The
table below was written afterwards from that plan, with the checks as they were run. Two rows came later, after a review of
the round: the RAM copy's order, first skipped untested, was measured (its bar set before that run), and the fence fix got
a check of its own.

| item | where | the exactness it must keep | its check |
|---|---|---|---|
| a prompt's tokens from the last request's (+ the literal marking remembered) | serve | every prompt's ids = one full encode's | a randomized equivalence test, the serve suite, a timing |
| #53's restart half | engine | a restored conversation's follow-up = an uninterrupted engine's | `serve_restart.py`: save on SIGINT, restore, against a fresh engine |
| SIGINT / SIGTERM as QUIT in `--serve` | engine | no request's output changes; the save runs before the end | the same run (A ends on SIGINT) |
| V4.1's short-read limit by length | engine | today's usual path (96) | `first_bits.py`, slots pinned: the installed engine's 3278fb7b32159f1e |
| a failed sync fill ends its swap; MMQ's ids padded; the PLE reader; the metrics sampler | engine / serve | unreachable or edge cases; no bit of a normal run moves | the same `first_bits.py` run; the serve suite |
| the parser's fences (a call shown inside a ```` block stays text) | serve | the model's tokens untouched; the parse right wherever it changes | the parser before and after over 3,000 generated outputs, then CommonMark on the 208 that differ: **failed, reverted** (below) |
| Maya's per-layer RAM copy (`DS4_COPY_FAIR`) | measure first | - | a decode trace of a 9-turn conversation replayed in `tier_sim.py --engine-swaps` at the server's tiers, the profile its learned one: taken only if the decode's drive reads a token fall 10% or more (the decode's own noise is +-6%) |
| not taken: `reasoning_effort: false`, the 503, Q8MV_TC, #644, the lossy and other-hardware items | - | - | - (reasons below) |

**Taken.**

1. **A prompt's tokens from the last request's** (`serve/prompt_memo.py`; upstream f13a870d, Maya c25b1db). An agent resends
   its whole conversation every turn, and the server tokenized all of it in Python before the engine read a token. Timed on
   an 876K-token agent conversation built from repo files (no session content), the box idle:

   | the server's encode_prompt | before | after |
   |---|---|---|
   | first request of a conversation | 7.7-8.0 s | 2.3 s |
   | each later turn | 7.6-7.7 s | **0.13 s** (+0.14 s for the GEN line) |

   Where the time was: the literal marking's unmark scanned the prompt with a 1,288-way alternation of single private-use
   characters (5.5 s; a class of their code-point ranges finds the same matches in ~20 ms), the marking itself tested each
   message against ~1,290 control-token texts (0.7 s; now remembered per string, `LiteralMemo`), the tokenizer 1.9 s (now
   only the text after the last cut it can reuse). Why the reuse is exact: the tokenizer splits at its special tokens first
   and encodes each piece alone, so up to the end of any special token it emitted - a cut - the ids are the same in a text
   that starts the same way, provided the cut lies the longest special token's length before the first changed character and
   the plain spans before it agree. The rest is encoded piece by piece, which records the next request's cuts.
   `serve/test_prompt_memo.py`: growing turns, an edit in the middle, a compaction-like rewrite, two conversations taking
   turns, literals in messages, 60 random edit points against one full encode - every result the full encode's. The serve
   suite: 853 OK. `STRATA_PROMPT_MEMO=0` turns it off, `=verify` checks every prompt against a full encode. The GEN line's
   ids are joined with `map(str)` (the same text, ~2x faster).
2. **DSV41_BACKLOG #53's restart half** (`src/glm/glm_park_disk.cpp`; upstream cdc3a265, be26cb79, 6e53c7f8 for the idea).
   The config's `"conversation_dir"` (the server passes `--conversation-dir`; a key, not an argument, so the check scripts
   that start an engine from the config's args never write there): at a clean end - QUIT, the end of its input, SIGINT or
   SIGTERM - the engine writes the live conversation and each parked one (rows, the running state at its end, its
   checkpoints on its own line) to a file of its own, and a later engine puts one back through parking's restore when a prompt
   continues it 1,024+ tokens further than what it holds. The files hold hashes of the tokens (two independent 64-bit chains),
   not the tokens, and the identity of the engine that wrote them (the binary, every argument, the STRATA_* and
   CUDA_VISIBLE_DEVICES environment, the model files): another build's or configuration's file, a damaged one or one older
   than a week is deleted when the next engine looks; the sizes are checked against its geometry and a checksum against the
   data. At most 8 GiB (`--conversation-dir-mib`), never the disk's last 8 GiB, newest first.
   `checks/serve_restart.py` (first_bits.py's engine and prompt, `--conversation-cache-min-tokens 2048`):

   | engine (the installed build, the slots pinned) | what it did | result |
   |---|---|---|
   | A, fresh | the 4,013-token prompt, then SIGINT | logits **3278fb7b32159f1e** (first_bits.py's reference); exit 0 after 3.3 s, 4,016 tokens saved (54 MiB, 1 checkpoint, 84 ms) |
   | B, started the same way | the follow-up (4,028 tokens) | 4,016 tokens read back from disk, reused 4,016; logits **1addbaf23207727d**, tokens 671 5949 396 1733 270 13523 477 979 |
   | C, fresh, no directory | the prompt, then the follow-up in one run | the prompt 3278fb7b32159f1e; the follow-up **1addbaf23207727d**, the same 8 tokens |

   ALL PASS. (A first run without the slots pinned - each engine splitting them by its free VRAM, 1,158 / 542 - gave the
   same verdict, B = C at fba319870d7738f7, but A could not be held against the reference; the script pins
   `STRATA_GLM_STAGE_SLOTS=1142,558` now, as first_bits.py's runs do.) B also saved on QUIT, the server's way (4,035 tokens,
   110 ms). The identity holds across the server's starts: its arguments come from the config (the learned profile's path
   once it exists), and a manual `"layer_split"` keeps the cards' order.
3. **Ctrl+C no longer kills the engine before the server's QUIT.** The engine runs in the server's process group, so Ctrl+C
   reached it directly, and it had no handler: it died at once, before its profile save on QUIT. In `--serve` SIGINT and
   SIGTERM now act as QUIT (the running request stops, the conversations and the profile are saved, then the defaults are
   back: a second signal ends it); QUIT also stops a running request; SIGPIPE is ignored there (a gone server no longer ends
   the engine before its save; a request whose lines cannot be written stops).
4. **A short follow-up's way by its length alone** (upstream dfd1b0f2). `short_read_limit` lowered the 96 of the config's
   `--short-read` to `chunk_ms / rw_ms_tok` (at least 32) from this process's timings, so a 32-95-token follow-up went through
   decode windows one time and the prompt path (MMQ, other rounding) the next. On V4.1 it is the configured 96 now.
5. **A failed synchronous fill ends its swap** (Maya #646). `issue_incoming` answered a failed `fill_slot` as "in flight";
   the swap's never-recorded event then read as landed and the slot was made resident over bytes that were not the expert's
   (and a swap past its victim's copy-out looped in `finalize_swaps(true)`). Only reachable with
   `STRATA_DSV41_ASYNC_FILL=0` or after a read error; now the swap ends with the slot empty.
6. **MMQ's ids table padded** (Maya #645): llama.cpp's MMQ reads up to ~190 ids past a tile's last row; 512 entries more
   (2 KiB, filled) keep that read in the table - it landed in the next buffer of the same allocation before, so never a fault.
7. **The PLE reader after a failed read** (upstream ebe3383a): its next ticket waited for ever; only with
   `STRATA_DSV41_EG_READER=1` (off). **The metrics sampler** survives a failing sample (upstream b3db5a1c). (The family
   parser's fence rule, upstream 209dcadb, was taken and then reverted after its CommonMark check - below.)

**Measured first, not taken: Maya's per-layer RAM copy** (`DS4_COPY_FAIR`, 1243b70: the static RAM copy takes each
layer's hottest experts in turn instead of one order over all layers; Maya measured long generations ~3x faster on 2x RTX
3080 + 96 GB with a 155 GB GGUF and no SSD cache). `tier_sim.py --ram-fair` (new; `--profile FILE` reads a strata-glm
profile) on a decode trace of a 9-turn conversation recorded for it (`serve_short_warm.py`, TURNS 8, 200 tokens a turn,
exact routing - `NO_BIAS=1`, so the trace does not carry the bias's own placement; 1,877 tokens), the engine's tiers of that
run (VRAM 1,153 + 887 slots, the RAM copy 18.17 GiB = 1,366 records, the 60 GiB cache = 4,558), the q3q2u learned
profile, `--engine-swaps`:

| the RAM copy | a decode token: VRAM | RAM | the cache | SSD reads |
|---|---|---|---|---|
| one order (today) | 149.9 | 18.2 | 59.7 | **11.3** |
| by layer in turn | 149.9 | 17.5 | 60.2 | **11.3** |

The same drive reads (the bar was -10%): the 60 GiB cache holds what a layer's thin RAM share misses. The engine's own
lines on that run read 8.4-18.6 records a token from the SSD across its turns, the replay 11.3.

**Not taken.** A client's `reasoning_effort: false` read as the config's effort (e69cbfa5): no client here sends it, and the
fix changes what it means. A dropped connection instead of a 503 when the engine binary cannot start (c330caf4): nothing here
starts it that way. Maya's tensor-core q8 matrix-vector kernel: not exact with Strata's per-thread sums, and the GPUs wait on the
host anyway. Maya's "the same bits on the CPU lane and in VRAM" (#644): it changes the output and was 1.3-2.5x less exact on
q3q2u (§49). Turbo / capped experts, smaller Engram tables: lossy. `--parallel`, tensor-parallel, EXL3, AMD, Metal,
Windows: not this box's. If #132 comes back, Maya's IQ1_S CPU rows (2cbe538, a017efc) and MMQ loaders (5808aa8) are there.

**The bits.** `first_bits.py` with the slots pinned (`STRATA_GLM_STAGE_SLOTS=1142,558`) on the new build: the 4,013-token
read **3278fb7b32159f1e**, tokens 671 2831 344 304 - the installed engine's (§62, §64). The serve suite: 853 OK (with
upstream's own fix to its sampler test, b3db5a1c: it counted every sampler thread alive, and samplers now outlive a failing
sample). Installed: build-dsv41/strata-glm **d7ad8f88** (8a84b4cd kept); strata-dsv41.json `"conversation_dir":
"~/AI/Strata-X-data/dsv41-conversations"`.

**Two checks more, planned before they ran (2026-11-05, after a second review of the round).**

1. *The fence fix's 208 differing parses, against an independent reference.* Each differing output is parsed by
   markdown-it-py 3.0.0 (CommonMark): the call block counts as code when it starts inside a fenced or indented code block.
   The fix is kept only if the new parser agrees with CommonMark in every one of the 208 (the old one is expected to
   disagree there); any case where the new parser and CommonMark part is a regression, and the fix goes back out.
2. *#53's restart half through the real server.* A copy of strata-dsv41.json (diagnostics on, its own log, conversation
   folder and port; the installed engine): a ~70K-token conversation over the OpenAI endpoint (greedy, a short reply),
   then SIGINT to the server's process group - what Ctrl+C in its terminal sends - then the server started again and the
   follow-up sent. It passes if the engine saved the conversation on the stop, read it back for the follow-up and reused
   the whole of it (the identity held across two starts of the real server), and the follow-up took seconds, not a re-read.

*Check 1, the result: failed - the fence rule is reverted.* markdown-it-py (CommonMark) on the 6,000 parses: of the 208 that
differ, the new rule agrees with CommonMark in 194 and the old one in 14. The 14: the new rule closes a ```` block on a line
like "````md" - a run as long, with text after it - where CommonMark lets a closing fence carry nothing but spaces, so a call
CommonMark shows inside the block ran. (The old rule's toggling got those 14 right by chance and the other 194 wrong.) Of the
5,792 parses the two rules agree on, both agree with CommonMark in 5,270 and both part from it in 522 - the parser's
existing ways, untouched. As planned, the fix went back out: serve/family_parsers.py is c7bbbca9's again, byte for byte,
and its test is gone; the parser tests pass (74 OK). A rule that is CommonMark's would also change the 3-character fences
(a ```python line inside a ``` block no longer closing it) - not taken in this round.

## 67. Unsloth's UD-IQ3_XXS replaces q3q2u (2026-11-06)

**Why** (the user, 2026-11-06): the native release had served the day through its own config (`strata-dsv41-native.json`,
`run-deepseek-v4.1-flash-native.sh`, its own conversation folder and learned profile), at 11-12 tok/s from noon to 16h
with the conversation up to 233K tokens, flat - the adaptive tier and the SSD cache learn while it runs, so a restart
would not have made it faster. Then: "Let's delete q3q2u and download unsloth's Q2_K_XL or IQ3_XXS as a replacement. How
fast do they run?", and on the answer below, "IQ3_XXS, go ahead".

**The two quants** (unsloth/DeepSeek-V4.1-Flash-1017-GGUF, their v3.1 quants, the repo's lastModified 2026-11-02; each
GGUF's tensor headers read with range requests):

| | experts | bits a weight | MXFP4 downs (the release's) | Unsloth's KL / same top, wiki1100 | long17k | decode here (estimate) |
|---|---|---|---|---|---|---|
| UD-Q2_K_XL | 191.14 GB | 2.81 | layers 0-1, 15-19 | 0.138 / 91.5% | 0.0879 / 93.3% | ~18 tok/s (17-20) |
| UD-IQ3_XXS | 224.30 GB | 3.30 | layers 0-9, 15-34 | 0.0625 / 94.6% | 0.043 / 95.1% | ~16 tok/s (15-18) |

Unsloth's figures are theirs, against DeepSeek's reference with their own dense weights and Engram tables; q3q2u's KL
0.1328 (§16: wikitext-2 against the native release, the experts alone different) is another test and compares only
roughly. The estimates are not measurements: they come from the measured points - the GSQ IQ2_XXS pack's 132 GB at ~27
tok/s (§58), q3q2u's 212 GB at ~18, the native release's 289 GB at 11-12 - and §4's CPU rates, the IQ types slower on
this CPU than Q2_K / Q3_K / MXFP4 (IQ2_XXS 39.0, IQ2_S 46.9, IQ3_XXS 50.1 GB/s against 74-77). UD-Q2_K_XL has 10% fewer
bytes than q3q2u, but its gate/ups are all IQ types, which roughly cancels that; UD-IQ3_XXS has 6% more bytes and about
half Unsloth's KL. The user took UD-IQ3_XXS.

**gguf_pack.py** (§51's tool, two changes). MXFP4 (ggml type 39, 17 bytes / 32 values): the native records' own format
(fp4_repack.py writes the release's FP4 as ggml's block_mxfp4, as llama.cpp's conversion does), so a GGUF's MXFP4 expert
tensor copies as it is. A shard with no expert tensor - the metadata-only first shard, Unsloth's Engram tables in the last
- is not downloaded: 233.4 GB streamed of the quant's 394.7, its 161.3 GB of Q6_K Engram skipped (the engine serves the
release's FP8 Engram tables with every pack). DSV41_BACKLOG #132 needed the second.

**The pack** (2026-11-06 17:00-21:21, the user service `dsv41-udiq3xxs-pack` beside the native server, 14-15 MB/s; one
stream cut at 18:08, resumed in place): 224.31 GB of records (208.9 GiB) in `~/AI/Models/dsv41-1017/ud-iq3xxs`, 12.3-16.4
MB an expert. Gate/up per layer: IQ2_XS (0-1), IQ2_XXS (2-4), IQ2_S (5-14), IQ3_XXS (15-24, 35-37, 39), Q3_K (25-34, 38);
down: MXFP4 (0-9, 15-34), Q3_K (10-14, 38), IQ3_S (35-37, 39). The release's Engram tables and the drafter's experts are
links into `native` (`engram`, `experts/M0-M2.bin`), as q3q2u's were; the shards' SHA-256s and the layout are in
`recipes/ud-iq3xxs/README.txt` beside the pack. Every type has the engine's kernels - the CPU pool through ggml-cpu's
rows (AVX-512 rows for the IQ types), the decode's MMVQ, MMQ's tiles for the prompt path; IQ3_S (four layers' downs) has
not run on V4.1 before.

**The checks.** Every shard's SHA-256 matched Hugging Face's while it streamed. `gguf_pack.py --verify 2` (experts 0 and
383 of every layer read back from the GGUF, their gate, up and down byte for byte, the padding zero): 80 records, 0
problems. `checks/verify_pack_downs.py` (new; 4 experts a layer): the 40 layers of experts.tsv, each file 384 records of
its stride, every stride its types' size rounded up to 4 KiB, and the 120 MXFP4 downs checked (30 layers) equal to the
native release's bytes - all pass.

**The switch.** q3q2u deleted at the user's word (16:59; the drive 87% -> 75% full, 87% again with this pack);
`recipes/q3q2u/build.sh` rebuilds it from native with requant_pack (~64 min). `strata-dsv41.json`: `--pack
.../ud-iq3xxs`, and `--expert-profile` the same file as `"expert_profile_save"`
(`dsv41-expert-profile-udiq3xxs-learned.txt`, seeded with the native release's learned profile of 2026-11-06, 34.7M
decode routings). The server puts the learned file in place of the config's only when it is the same pack's (#57), so a
config whose `--expert-profile` names another file starts with one argument and continues with another, and the
conversation saves' identity (§66) changes between the first and second start; with both the same file, every start
passes the same arguments. (The native config has that change ahead: its next start swaps q3q2u's learned file for its
own, once.) The profile's save writes a `.tmp` and renames it, so a stop during the save leaves the last whole file.
`run-deepseek-v4.1-flash.sh` refuses to start while another Strata server runs, as the native script does - two engines
do not fit this box's memory. The general check scripts, `drivetest.py` and `eg_fetch_rate.cpp` point at ud-iq3xxs;
verify_q3q2u.py, round109.py, kld_q3q2u.sh and speed_q3q2u.sh stay as q3q2u's records.

**Against the native release** (2026-11-07, the native server stopped; binary c80230c5). `serve_smoke.py` on a copy of
strata-dsv41.json whose profile save and conversation folder point elsewhere (the seeded profile and the production folder
untouched): ALL PASS - ready 32 s after the start, the first reply 40.8 s after the request, the picture's 18.6 s; IQ3_S's
first run on V4.1. Then `PACKS=ud-iq3xxs kld_iq2xs.sh` (q3q2u's three tests, each a fresh engine; q3q2u's figures are
§16's of 2026-10-21 - its wikitext-2 run again on 2026-10-24 with a later binary gave 0.1307 +- 0.0079, so the binaries
since agree within the error):

| test | mean KLD | 99% | max | same top | PPL ratio |
|---|---|---|---|---|---|
| wikitext-2 16x512, prompt path: q3q2u | 0.1328 +- 0.0080 | 2.25 | 10.2 | 92.55% | 1.024 |
| the same: UD-IQ3_XXS | **0.0658 +- 0.0038** | **1.15** | **4.11** | **94.51%** | 1.037 |
| agent text, decode windows: q3q2u | 0.0317 +- 0.0015 | 0.318 | **0.73** | 94.38% | 1.004 |
| the same: UD-IQ3_XXS | **0.0240 +- 0.0026** | **0.224** | 4.94 | **94.92%** | 1.011 |
| agent text, route bias 0.3: q3q2u | 0.0397 +- 0.0021 | 0.448 | 1.54 | 93.01% | 1.011 |
| the same: UD-IQ3_XXS | **0.0306 +- 0.0015** | **0.312** | **1.13** | **93.50%** | 1.019 |

The KLD falls by half on wikitext and by a quarter on the agent text, at the production bias too, and the same-top rate
rises in all three; the perplexity ratio moves the other way in all three, by 0.7-1.3 points - the next token's own
probability a little lower on average than q3q2u gave it, while the whole distribution is nearer native's. With the
server's sampling (temperature 1.0, top_p 1.0: the whole distribution) the KLD is the measure of how far the sampled
text's odds are from native's. Not measured yet: the decode speed (from the user's runs, against the native release's
11-12 tok/s of 2026-11-06) and `first_bits.py`'s references for this pack (3278fb7b32159f1e and the others are q3q2u's).

## 68. DeepSeek-V4.1 from a standard GGUF set, and the head-to-head against upstream Strata and Maya (2026-11-07 to 2026-11-09)

**Why** (the user, 2026-11-07/08): Strata-X goes open source, so it should run DeepSeek-V4.1 straight from GGUF files anyone
can get, and be compared with upstream Strata and Project Maya on the same files - "a fully unquantized comparison on the
GGUF of the MXFP4 model". The set was made here from the release, with nothing downloaded ("We have the full model
downloaded locally and we have llama.cpp for conversion").

**The set** (`moex/tools/dsv41/gguf_convert.py`). It runs mainline llama.cpp's converter (7d7f0287,
conversion/deepseek41.py: the FP8 weights dequantized to BF16 exactly, the experts' MXFP4 kept bit for bit) with two of
its writers replaced:
- The experts are streamed from our native records, each record's crc32 checked against the manifest.
- The Engram is written as MXFP8 (ggml type 67: an e8m0 scale and 32 e4m3 values in a 33-byte block, the release's own
  format). The stock converter writes it as BF16, 699 GB, which neither Maya nor upstream reads.

Nine files, 508.4 GB:
- 1: metadata, plus the chat template (unsloth's port, as in upstream's fixtures);
- 2-7: the experts, 7 layers a file;
- 8: the Engram, 202.76 GB;
- 9: the dense part, 922 tensors in BF16/F32, 16.84 GB.

`--verify` read every expert slice and Engram row back with O_DIRECT and compared it with native: all identical. Files
2-7 and 9 have the SHA-256s of unsloth's UD-Q8_K_XL shards, so the local conversion is the published one byte for byte.
Its reads were paced at 50 MB/s while the server was busy.

**The engine.** Every part is exact.
- `STRATA_DSV41_ENGRAM_GGUF=FILE` takes the Engram from the GGUF's per_layer_token_embd; eg_rows splits each 264-byte row
  back into its 256 code bytes and 8 scales. Check: first_bits PROMPT_N=4000 with slots 1142,558 on UD-IQ3_XXS gives
  511ef09a0abcbd75 / [671, 2831, 344, 304] with either table.
- `--experts-gguf FILE` reads the routed experts from the set as virtual records: three O_DIRECT extents an expert
  (gate, up and down sit in tensors of their own), placed at 0, up_off and down_off of a padded slot (1 KiB of room on
  each side, the stride 4 KiB-aligned).
  - A slice that starts inside a sector is read from that sector's start.
  - The gate's first sector goes through a 512-byte buffer of the request's own, so nothing is written before the slot.
  - The last sector may run past the file's end (Seg.need).
  - It mixes with --pack layer by layer. Check: c78a0b99bad3b4b8, the native records' bits, with layers 0-27 from the
    set and with all 40.
- The load pass reads only the planned runs for the set's layers, as it does for records: 265 -> 69.8 GiB at the start.
  This was found from the user's question about the drive.
- `-m <file 1>` alone is enough:
  - the settings are mapped (the indexer's keys, the Engram's ple.* keys and token map, per-layer clamps);
  - the Engram tensors are bound;
  - the BF16 dense goes through gemv_dt (pj, heads_mm, the head), with a widening scratch of at most 48M floats and row
    blocks past it.
- `STRATA_DSV41_DENSE_Q8=1` quantizes the BF16 projections, engram_kv and the head to Q8_0 at load (ggml_quantize_chunk,
  names matched whole). It gives c78a0b99bad3b4b8, native's bits: the bits of our dense file. The BF16 dense as stored
  gives fd18c30dd37d9486 / [671, 2831, 344, 304], with 14.30 GiB of dense on the GPUs against 7.79.
- conv_identity stamps every file of the -m set, the Engram GGUF and the expert set.
- serve_smoke: ALL PASS with the GGUF Engram, and again with --experts-gguf.
- Two fixes found on the way:
  - choose_chunk now counts the stop snapshots' copies. The BF16 arm had failed with "the prompt path's buffers (6494 MiB)
    fit neither the lendable slots nor the free memory of GPU 1".
  - The swap cache's check of its first four victims compared the GGUF slot's gaps, which hold no byte of the expert. It
    printed UNLIKE in both runs while the slot check found nothing. It compares the three slices now: 4 of 4 byte for byte
    on a 4K run, the three experts flagged before among them.

**#53's restart half through the real server** (2026-11-06). The conversations' identity took the --vision-weights path,
and the server makes that temp folder anew at every start, so every start dropped the last one's conversations. The
identity now takes the manifest's lines and its weight files' stamps (glm_main.cpp; checks/serve_restart_e2e.py).

**Deleted after verification** (the user: "delete something ONLY AFTER fully verifying that it's no longer needed"):
- the UD-IQ3_XXS pack, 224 GB, at the user's word (strata-dsv41.json and run-deepseek-v4.1-flash.sh still name it:
  retired);
- native/engram, 203 GB (rows identical, bit check, smoke).

native/experts L00-L39 (269 GiB) wait in native/experts-held for a clean start without them. The native config (local)
has had the GGUF Engram and --experts-gguf since 2026-11-08 (installed dbd26018).

**The head-to-head** (checks/h2h_engines.py, h2h_table.py; results in ~/AI/Models/dsv41-1017/h2h/results-20261108).
- **The method:**
  - The same token ids go to each engine through the --serve protocol they all speak (GEN / T / DONE), decoded greedily.
  - The set's page cache is dropped first, so each engine starts cold.
  - CTX 262144: a 16,000-token first turn of this repo's docs, then 4 follow-ups of 400 new tokens, 128 tokens
    generated a turn.
  - Every engine starts from the same learned expert profile, and each leaves the display card ~0.5 GB free (ours:
    `--vram-reserve-later-mib 512`).
- **The other two:** upstream Strata's strata-dsv41 0.1.58 (146df3c4, built in a worktree) and Project Maya's maya-ds4
  1.1.56 (7bf15ac). Each runs on its own pack_gguf.py index of the set, as its setup writes it (--tp 2) and at --tp 1
  (each project's other candidate). No drafter (the set has none), no vision.
- **Ours:** the set alone, the dense in BF16, the native config's switches without the route bias.

| engine | ready | 16K read | follow-ups | decode tok/s (range) | drive read | slow-mode s |
|---|---|---|---|---|---|---|
| upstream --tp 1 | 62.6 s | 119.5 s | **28.0 s** | **5.70** (3.86-7.44) | 762 GB | 0 |
| Maya --tp 2 | 72.9 s | 230.3 s | 78.8 s | 5.15 (3.42-5.95) | 907 GB | 0 |
| Maya --tp 1 | 66.4 s | **75.2 s** | 42.1 s | 4.65 (3.73-5.45) | 693 GB | 0 |
| upstream --tp 2 | 67.7 s | 325.7 s | 57.2 s | 4.48 (3.06-5.09) | 1,385 GB | 0 |
| ours | **36.7 s** | 121.8 s | 34.1 s | 3.97 (2.96-4.72) | 894 GB | 0 |
| ours, a 97.85 GiB RAM copy | 93.4 s | **75.2 s** | 29.4 s | 3.40 (2.00-3.99) | 836 GB | 0 |

One run an arm.
- The last row gives ours the others' RAM budget (`--ssd-cache-gib 0 --resident-headroom-gib 12`): the profile's hottest
  96.17 GiB, fixed for the run. RAM peaked at 118.6 GiB used.
- First tokens: ours match Maya --tp 2's. Upstream differs at the 4th token (tp 1 and tp 2 differently) and Maya --tp 1 at
  the first. Greedy decoding parts at near-ties, so this is no verdict on quality.

**Where the decode's time goes.**
- **Ours** (the engine's own lines; the default arm, turns 1-4), one part after another:
  - 150-189 ms a token waiting for the drive's 21.6-26.6 experts (18.8 MB each: 406-499 MB at ~2.7 GB/s, the drive's
    ceiling here);
  - 27-31 ms waiting for the routing;
  - 35-44 ms of the CPU pool's expert math.
- **Upstream's**, from two sources: its serve-mode STAT lines, which the harness now keeps, and the report its one-shot
  path prints, run after each served request by a switch local to the worktree (DS4_SERVE_REPORT). Its own DS4_PROF
  stopped the engine at the first decoded token ("stack smashing detected").
  - Its CPU lane computes the experts VRAM does not hold: 147-164 a token. None are streamed to the GPU or computed from
    host RAM; its "CPU lane %" counts the prompt reads, where the GPUs do the work.
  - Its reads are on demand at ~2.6-2.7 GB/s, none ahead.
  - It misses 17.9 / 9.9 / 12.0 / 13.0 experts a token on turns 1-4, against our 26.6 / 22.3 / 21.6 / 24.4.

That is the gap: half the misses, on the same drive. Upstream's RAM store holds 5,456 experts (102.6 GB), filled with the
profile's hottest at the start (37 s), and it adapts as it goes (1,256-3,395 evictions a turn). Ours is a 20 GiB fixed
copy plus a 60 GiB cache that starts empty (~4,570 experts); the last row's fixed 97.85 GiB copy never adapts. Next
(§69): the cache filled from the profile at the start.

Upstream's turn 4 hit the day's first slow-mode episode on these files: 68 s, 245 MB a token at ~0.4 GB/s, 1.45 tok/s.

**Not compared:** long contexts (upstream's own doc says nothing past 256K was tested), drafters, and the modes that change
the answers (our route bias; upstream's DISK_SKIP_W and --experts). Each would need its quality measured with its speed.

## 69. The cache as the experts' RAM, and a warm start (2026-11-09)

**Why.** §68 found the decode's gap to upstream in the RAM tier. Upstream's store held 5,456 experts (102.6 GB) and adapted
as it went; ours was a 20 GiB fixed copy plus a 60 GiB cache that started empty. Upstream missed 9.9-17.9 experts a token,
ours 21.6-26.6.

**The warm start, measured first** (`Impl::ssd_warm`, opt-in `STRATA_DSV41_CACHE_WARM=1`). At the end of the expert load the
cache is filled with the profile's hottest experts that no VRAM slot and no RAM copy holds. The hottest go in last, so the
LRU gives them up last. On the default tiers (20 GiB copy + 60 GiB cache):
- drive reads a token on turns 1-3: 26.2 / 20.6 / 22.6, against 26.6 / 22.3 / 21.6 with the cache empty;
- the start took 25 s longer.

The decode replaced the profile's picks within a turn.

**tier_sim.py on that run's trace** (682 decode tokens, `--engine-swaps`, the q3q2u-learned profile) matches both engines'
runs: upstream's tiers simulate at 16.1 drive reads a token (15.2 measured), ours at 23.2 (26.5 measured). With perfect
knowledge of the conversation's counts, a static placement on our tiers would still miss 19.5 a token, so the size of the
tiers is the bound. Other policies at 4,814 slots (fixed VRAM):

| policy | drive reads a token |
|---|---|
| LRU | 24.6 |
| LFU | 23.6 |
| frequency-gated admission | 24.2 |
| Belady (the floor; needs the future) | 15.8 |

History says little about the next turn's experts; the size of the tiers does.

**The layout.** With the experts' RAM as the cache alone (no RAM copy), the decode's misses fell, and the warm start is what
that layout needs: it gives the start the profile's experts a fixed copy gave, and then adapts. Measured in tonight's
protocol (§70's table): the cache-only layout at 85 GiB decoded turns 1-4 at 4.26 / 5.70 / 5.93 / 5.69 tok/s, but turn 0 at
1.12 (empty); with the warm start, turn 0 at 3.37-3.40.

## 70. The release's FP8 projections back as FP8 (2026-11-09)

**Why.** The GGUF set holds the release's FP8 projections as BF16, written exactly by llama.cpp's converter (each value
times its block's power-of-two scale). BF16 costs 14.30 GiB on the GPUs. That is 6 GiB of expert slots, and on GPU 0 it
was the room layer-major needs: the loan was 10.66 of 9.70-10.58 GiB, so long reads ran chunk by chunk.

**The change** (opt-in `STRATA_DSV41_DENSE_FP8=1`).
- **Load** (load_dense): each run of 32 BF16 values of a row of those projections (DENSE_Q8's list without the head) becomes
  32 e4m3 codes and one e8m0 scale, the smallest power of two that holds the run.
  - The smallest scale is still exact: the release's scale is a power of two at least as large, and dividing by a smaller
    one only moves the values up.
  - Each value is checked as the GPU will read it back. A matrix that is not exactly FP8 refuses the start.
- **GPU** (internal type kDtFP8: values row-major, then the scales): gemv_dt, heads_mm and pj take it as they take BF16.
  - Decode: gemv_fp8_w. Each thread takes values t, t+256, ... as gemv_bf16_w's does, from rows read into shared memory by
    16-byte loads, widened branch-free.
  - At T = 1 with STRATA_DSV41_PRELOAD: gemv_fp8_w1_pre_kernel, gemv_bf16_w1_pre_kernel's order.
  - Prompts: fp8_widen into the same F32 scratch, then the same cuBLAS GEMM.

**Bit for bit.**
- first_bits PROMPT_N=4000 on the set, slots pinned 600,460: BF16 and FP8 both give fd18c30dd37d9486 / [671, 2831, 344, 304]
  - for the first kernel, the table one, and the preload/tiled one (binaries 3240fbe7, d8447305, d5b7aaf1).
- `moex/tools/dsv41/fp8_gemv_bench.cu`: every output on DeepSeek-V4.1's projection shapes is equal to gemv_bf16_w's, at T 1
  and 4, with and without the preload.
- 322 matrices; dense 14.30 -> 8.17 GiB (GPU 0 6.83 -> 3.62, GPU 1 7.47 -> 4.55).

**Speed** (fp8_gemv_bench, RTX 30-series, PRELOAD as production): T 1 at 0.83-0.98x of BF16's preload kernel. The first
version, without a preload kernel, ran at 0.34-0.55x there: +10 ms a token waiting for the GPU (§71's runs). These kernels
are bound by one block a row, not by bytes; a faster shape would change the summation order and lose BF16's bits.

**The kernels, a third step (2026-11-09 05:40).** The runs showed what the benchmark then confirmed: every one-row kernel (BF16's
and FP8's) runs the decode's projections at 150-500 GB/s. `gemv_fp8_rows_kernel<R>` (T = 1; `STRATA_DSV41_FP8_ROWS`, 4 by
default, 1: the one-row kernels) computes R rows a block:
- thread t takes values t, t + 256, ... of each row, in gemv_bf16_w's order, with the same warp sum and block sum, so the
  bits are the same;
- the block reads x once for its R rows, and the rows by 16-byte loads into shared memory, in tiles of 2,048 values.

fp8_gemv_bench at T = 1, against BF16's production path:

| projection | speed |
|---|---|
| attn_output_b | 2.11x |
| ffn_down_shexp | 1.25x |
| attn_output_a | 1.24x |
| ffn_gate/up_shexp | 1.16x |
| attn_q_b | 1.12x |
| attn_q_a | 1.07x |
| attn_kv (512 rows: too few blocks) | 0.64x |

Every output is bit for bit, and first_bits gives fd18c30dd37d9486 (binary 72e2cae9). In the run, the decode's wait for the
GPU fell from ~40 to 21.4-21.9 ms a token. The default is on for the GGUF set (`STRATA_DSV41_DENSE_FP8=0`: off): a matrix
whose values are not exactly FP8 keeps its BF16, checked at load.

## 71. Overnight: the three engines on equal terms, and the drive's episodes (2026-11-09)

**The goal** (the user, before an overnight away): Strata-X the fastest and most efficient of the three engines on
DeepSeek-V4.1 without compromising quality; test changes on all three over the common GGUF set; look for ways around the
drive's slow mode; DSpark if it helps; nothing that could take the PC down; ~1.5 GB free on the display card for every
engine.

**The protocol.**
- §68's harness on the same set and the same turns: cold page cache, a 16K first turn and 4 follow-ups of 400 tokens, 128
  generated a turn, greedy.
- Every engine leaves 1,536 MiB free on each card (`H2H_RESERVE_MIB`; theirs: `--reserve-mib`, which is per card).
- Every engine gives the experts 85 GiB of RAM (`H2H_RAM_GIB`: theirs as the RAM store, `--ssd-cache 91.3`; ours as the
  cache alone), leaving ~22 GiB free.
- A watchdog (a user service) would have stopped a test engine, never the desktop or the installed engine, below 8 GiB of
  MemAvailable, at memory pressure "full" >= 40%, or with GPU 1 under 1 GiB free. It never had to act.

**The episodes.** Episodes came often from 02:31 on: about 9 TB had been read from the set since it was written on
2026-11-08, and this drive's slow mode follows bytes read (§14).
- A turn counts as episode-hit when the harness counted slow seconds in it (>= 80 ms a read and < 2,000 reads a second, its
  1 s samples).
- That test misses our engine's episodes: at the guard's floor our engine keeps one 128 KB read in flight, which takes ~3 ms.
  So ours also counts a turn whose engine log has a drive-guard line in it.
- The tables below average the other turns ("clean"). Turn 0 is left out of the decode average for every engine (the first
  decode after a cold 16K read).

| engine (runs) | ready | 16K read (clean ones) | decode, clean turns | follow-up read, clean turns | read per run | episode-hit turns |
|---|---|---|---|---|---|---|
| Strata-X final: FP8 dense + 4-row kernel + §72, warm all-cache (2) | 79 s (141 s with an episode in the fill) | **75.5 s** | **6.57 tok/s** (5) | 28.2 s (5) | **619 GB** | 4 of 10 |
| Strata-X: FP8 dense + 4-row kernel, warm all-cache (3) | 81 s | 85.5 s | 6.62 tok/s (7) | **27.0 s** (7) | 672 GB | 8 of 15 |
| Strata-X: FP8 dense, warm all-cache (4) | 84 s | 85.5 s | 6.18 tok/s (13) | 27.4 s (13) | 666 GB | 5 of 20 |
| Strata-X: BF16, cache cold (1) | 42 s | 129.5 s | 5.39 tok/s (4) | 29.6 s (4) | 805 GB | 1 of 5 |
| upstream --tp 1 (3) | 176 s | none clean | 4.84 tok/s (7) | 30.3 s (7) | 904 GB | 8 of 15 |
| Maya --tp 2 (3) | 65 s | 421.8-442.5 s | 4.63 tok/s (4) | 68.7 s (4) | 1,431 GB | 8 of 15 |
| Maya --tp 1 (1) | 65 s | **73.7 s** | 5.53 tok/s (2) | 31.9 s (2) | 748 GB | 3 of 5 |

Against each engine's best shape:

| | decode | follow-ups | bytes read | 16K read | start |
|---|---|---|---|---|---|
With the final config (the 4-row kernel's 12 clean turns average 6.60 tok/s; §72 changes no decode), against each engine's
best shape:

| | decode | follow-ups | bytes read | 16K read | start |
|---|---|---|---|---|---|
| ours against upstream --tp 1 | 36% faster | 7-11% shorter | 32% fewer | - | upstream's 58-62 s against our 79 s |
| ours against Maya --tp 1 | 19% faster | 12-15% shorter | 17% fewer | a tie: 75.5 s against 73.7 s | Maya's 65 s against our 79 s |

The 35 s our start spends filling the cache buys the first reply's decode: 3.77 tok/s after the cold read. Maya --tp 1 has
2 clean turns from a night of episodes.

**Why ours decodes faster now** (the engines' own counters, clean turns):
- ours misses 11.3-13.1 experts a token; upstream 9.9-17.9 (its store is LRU, 5,456 slots of 104 GB at its own size);
- ours keeps 50-61% of the routings in VRAM; upstream 29-37%;
- FP8 freed 6.13 GiB for experts (1,462 VRAM slots against 1,112);
- the 4-row kernel halved the GPU's share of a token.

**The episodes themselves** (drive-watch-night.log, the guard's lines):
- Each lasts 55-80 s with any engine, and they come in clusters - one engine's turn met three in 4 minutes.
- At the guard's floor (QD1) the drive gave 336-343 reads a second at ~3 ms (~43 MB/s).
- At the new `STRATA_DSV41_GUARD_FLOOR=2` (16 requests in flight) it gave 358-469 a second at 39-48 ms each: 5-38% more,
  while another program's read would wait ~45 ms more. The default stays 0.
- The engines without a guard queue ~1,000 reads, and each waited 1-2 s.
- The lever is the bytes read: ours read 666-670 GB a conversation against upstream's 904 and Maya's 1,431.

**The long-context round** (a 100K-token first turn of the same docs, then 2 follow-ups of 400 tokens; the same protocol;
upstream and Maya in their faster prompt shape, --tp 1):

| engine | ready | 100K read | read in turn 0 | follow-ups | decode after the read / clean turns |
|---|---|---|---|---|---|
| Strata-X (72e2cae9) | 116 s | **274.3 s** (a 72.9 s episode in it) | **512 GB** | 32.2 / 26.9 s | 3.25 / **5.47, 7.22 tok/s** |
| upstream --tp 1 | 58 s | 1,229.1 s (548 s of the slow mode) | 2,162 GB | 37.8 / 30.1 s | 0.75 / 4.01 tok/s (1 clean) |
| Maya --tp 1 | 89 s | 384.1 s (143 s of the slow mode) | 745 GB | 33.2 / 97.2 s (the second episode-hit) | 1.18 / 5.43 tok/s (1 clean) |

- Ours read the 100K prompt layer-major: 11 chunks in windows of 4, the loan 12.64 of 12.91 GiB on GPU 0 (FP8 gave it back).
  So each layer's experts were streamed 3 times.
- Upstream's chunked path read 4.2x the bytes; without its slow-mode seconds it would still have taken ~680 s.
- Maya at --tp 1 read 1.46x ours (its --tp 2, the shape its setup picks, took 420-440 s for the 16K turn alone).
- At 100K, our follow-ups and decode stay at their 16K values.

**Not done tonight:**
- DSpark - §70's trace says a window of k drafts needs 408 (k = 2) / 690 (k = 4) distinct experts against 240 a token, so
  on this drive it breaks even at ~60% acceptance and gains ~11% at 70%, ~20% at 80%;
- a background warm fill (the 35 s at start), and why Maya --tp 1's cold 16K read beats ours by 14% (both read ~190-200 GB
  near the drive's rate: 2.6 against 2.35 GB/s);
- multi-row kernels for the BF16 path, which would change STRATA_DSV41_BF16_W's bits against F32 (§62).

## 72. A one-window prompt keeps no reads for later chunks (2026-11-09)

**Why.** With the warm cache of §69, the cold 16K read still took only 2,073 records from the cache against ~5,100 to expect:
4,816 warm records, about 85% of them routed in such a read, plus GPU 0's 1,005 lent ones. Maya --tp 1 read the same
16K in 73.7 s against our 85.5.

**The cause.** A call of two or more chunks sets `long_now` at plan time (STRATA_DSV41_PROMPT_KEEP_LONG=1): its drive reads
are kept in the cache "for its later chunks", up to all of it, each taking the least recently used slot of the replayed
layers first (prefer_ge = 21). It went off only when the deferred tails started. With layer-major, one window holds both of
a 16K read's chunks, so there is no later chunk to keep for. The kept reads evicted the warm records of layers 21-39, which
the tails then read from the drive.

**The change** (glm_prompt.cpp, after the window is sized). When one window holds every chunk, `long_now` is cleared, so
only the hot records stay (STRATA_DSV41_PROMPT_HOT). That is the one-window path §63 meant. Where a record is read from
changes no result. `STRATA_DSV41_KEEP_LONG_ONE=1` keeps the reads as before.

**Measured** (cold 16K read, the same binary cb4fc08f, the protocol of §71; the counters do not depend on the drive's
episodes, the times do):

| | from the drive | from the cache | turn 0 read | read time |
|---|---|---|---|---|
| kept as before (KEEP_LONG_ONE=1) | 10,434 | 2,069 | 277 GB | (4 episodes) |
| not kept (the default now) | **8,219** | **4,287** | **220 GB** | **74.2 s** (0 episodes; 85.5 s before) |

21% fewer drive reads in the cold read. The first decode after it went from 3.4 to 3.79 tok/s.

## 73. The start, the engram rows, and the daily config (2026-11-09 morning)

**The start** (from timestamped engine logs; the drive's episodes hit nearly every start, so the clean figures are each
step's own):

| step | time | status |
|---|---|---|
| FP8 check before planning | read the candidates twice, ~3 s clean | gone; the conversion checks as it writes, and a matrix that is not FP8 keeps its BF16 in a buffer of its own (`dense_extra`) |
| warm fill | after the VRAM load (13.1 + 37.1 s) | started before the load pass (by the VRAM plan) and finished after it: both phases 46-48 s together |
| cache pinning (5 s) | after the dense load | mapped at once, faulted in and registered by a thread beside the load, joined before READY (`STRATA_DSV41_PIN_ASYNC=0`: before) |

The pinning's safeguards:
- the RAM plan counts the bytes still being faulted in as used;
- the warm fill's reads land only in blocks already faulted in, because the fault-in writes a byte of every page.

Tried and dropped: O_DIRECT reads of the FP8 candidates. One read a tensor took 22.4 s, 8 threads 26 s, against 16 s
through the mapping with the kernel's read-ahead.

Clean start: 79.4 -> 71.9 s measured before the pinning change, ~65 s estimated with it (the dense load 15.2-15.5 s, then
~47 s). Maya --tp 1 starts in 63.6-65.2 s, upstream in 58-62 s. All three read ~133-135 GB at the start.

**The engram rows of a prompt chunk** (`STRATA_DSV41_EG_GGUF_DIRECT=0`: the mapped reads). The GGUF table's rows are now
read O_DIRECT and deduplicated, each row's 264 bytes from its one or two pages, split as `put()` splits them.
- The wait fell 7.5 -> 6.4 s in a cold 16K read (77.0 -> 75.2 s; one episode in each).
- The rows wait behind the expert reads on a full drive more than they wait for their own.
- first_bits: fd18c30dd37d9486 with this path, FP8 and the background pinning (binary 77e08fbf).

**The user's choice.** "Steady-state first": the start and the first read stay at parity (the warm fill before READY),
rather than a start of ~45 s that would slow the first read to ~78-83 s.

**The daily config** (strata-dsv41-native.json, 2026-11-09; a backup is in Strata-X-data/*.bak-20261109):
- `-m` is the GGUF set's file 1: the release's FP8 projections as FP8, exact (STRATA_DSV41_DENSE_FP8 is on by default).
- `--ssd-cache-gib` 90, cut to MemAvailable minus the 28 GiB headroom (86.4 GiB in the smoke run): the cache is the experts'
  RAM, with no fixed copy.
- `--expert-profile` is dsv41-expert-profile-native-learned.txt (111.6M routings of the user's sessions).
- `STRATA_DSV41_CACHE_WARM=1`.
- The route bias stays at 0.3 (the user's daily setting, which changes outputs).

Installed build-dsv41/strata-glm 77e08fbf (dbd26018 kept). serve_smoke on a copy with its own conversation folder and
profile save: ALL PASS - ready 77 s, the first reply 40.3 s after the request, the picture's 25.9 s. The first start
drops the 1.17 GB saved conversation once (a new identity), so start a fresh dsh session.

## 74. Where speed is left, read after the UD sets and FP8 dense: the audit of 2026-11-10 (read-only)

**Why.** The user asked for areas that could raise the prompt and decode speeds and cut the slow episodes. Two changes since
the last audits moved the prices: the dense part is the release's FP8 again (§70), and the UD sets keep most of their experts
in memory (UD-IQ1_M 123 GiB, UD-Q2_K_XL 178 GiB, against native's 269), so the drive weighs less and the GPUs' and the
CPU's parts more. The UD-IQ1_M token of 2026-11-09 is 41 ms: 22.8 ms the host waiting for the GPUs, 14.4 ms in the CPU pool
(~53 experts), 3.5 ms the rest.

**How.** Four read-only passes (the GPUs' part of a token; the CPU pool and the tiers; prompt reads and a turn's fixed cost;
episodes and stalls), each against this plan and the backlog first, then the code. Nothing was built, run or measured: the
user's server was up. Every gain below is an estimate from the measurements named; the lines the larger items rest on were
read a second time (the FP8 kernels and their dispatch, the two-chunk rule, the CPU's feature test, the engram rows at the
guard's floor, the huge-page pacing, the hashes on the host's path).

**The drive by set** (sysstat's 10-minute samples, sa02-sa10; slow time = the drive's busy time its bytes at 2.6 GB/s do not
explain):

| set (period) | read | per decoded token, all in | slow time |
|---|---|---|---|
| native (11-06, 11-08; 12 h each) | 3.6-4.3 TB/h | ~150 MB | 9.8-12.3%, ~5-6 episodes an hour |
| UD-Q2_K_XL (11-10, 2.2 h) | 0.33-1.2 TB/h when active | <= 13 MB | none seen |
| UD-IQ1_M (11-09 evening, 11-10 night) | 0.16-0.23 TB/h | 2.7 MB | 1.4-4.6%, 0.7-2.4 an hour (downloads wrote 59-208 GB in both periods) |

By day: 28-80 TB read and 9.9-12.6K s of slow time on 11-02 to 11-09; 3.1 TB and ~620 s on 11-10. The UD sets are the
largest cut of the episodes so far; sysstat's samples cannot show a single episode, nor split reads from writes.

**Decode, the GPUs' part** (DSV41_BACKLOG #137-#141, #149).
- FP8 dense cost GPU time: the wait was 13.3-17.4 ms a token with Q8_0 dense and is 21.4-22.8 with FP8, +4-8 ms across runs.
  The FP8 kernels keep BF16's summation order and are bound by operations a weight, not bytes: a shorter decode in the same
  order (#137, est. 2-4 ms), the batching the Q8_0 path had (#138, est. 0.7-1.4 ms), a multi-row kernel for verify windows
  (#139), the head over both GPUs and the engram projections off GPU 0's chain (#140, est. 1.3-1.9 ms together).
- The VRAM experts' IQ kernel runs four rows a block (#141, est. 0.3-0.8 ms on the UD sets).
- At its limit: launch overhead (one graph a GPU, the doorbell inside it; the gaps 0.11-0.2 ms a token, §34), the sampler, the
  indexer at depth; §38 and §39 stay dropped; a projection's rows over both cards (the mapped round trips eat it).

**Decode, the CPU pool and the tiers** (#142-#144).
- The pool's 14.4 ms is rows, not overhead: the same rate at 1.3 and at 3.4 CPU experts a layer; set-up 0.2-0.3 ms a token.
- The IQ formats are bound by their unpacking - 123-151 G weights a second against 190-295 at DRAM's rate - because this
  CPU lacks VNNI and VBMI, which the 512-bit kernels ask for (#142: est. 3-6.5 ms a token on UD-IQ1_M, 5-9 on UD-Q2_K_XL).
- The tier's upkeep on the host's path (#143, 0.5-0.6 ms); a few GiB more of RAM and VRAM for experts (#144).
- Slots: native 1,429, UD-Q2_K_XL 2,195, UD-IQ1_M 3,237; 1 GiB of VRAM is worth ~2.4 routings, ~0.65 ms a token. Nothing
  standing in VRAM can move to host memory; only the attention state (0.92 GiB) and the score rows are sized by the 1M context.
- The route bias penalises only experts in no memory tier (`glm_dsv41.cpp:1820-1828`). At their limit: the per-layer slot
  split (55.4% against 55.5% unconstrained), slots by heat per byte (+0.56 points on GPU 0), the cache's pools, `POOL_TASKS`.

**Prompt reads and a turn** (#145-#147).
- A follow-up replays nothing (a resumed chunk under 2,560 rows runs its own rows); its fixed seconds are the routed experts'
  bytes - from the drive on native (memory holds ~40% of them), over GPU 0's x8 link on the UD sets (~61% and ~89% held).
  The fit of 2026-11-08 is ~7-9 s a turn plus ~55 ms a new prompt token.
- A short follow-up that crosses a 65,536 mark becomes a two-chunk layer-major read (#145: est. 30-45 s on native).
- On the UD sets GPU 1 and its link idle through GPU 0's layers (#146: est. 3-4 s a follow-up); layer 1 waits for layer
  14's engram rows (#147a).
- At their limit: the VRAM experts are used in place; the refill 0.4 s; the routing gap 6-9 ms a layer; the engram fetch
  sorted, deduplicated and O_DIRECT (§73); the server's per-request work 0.27 s at 876K.

**Episodes and stalls** (#133, #147b, #148).
- The drive's episodes are the only class that costs more than seconds a day. Inside one, a UD-IQ1_M token's 48 engram rows
  fault one at a time at the guard's floor (#147b). The guard counts its episodes and reports them nowhere (#148).
- #133's cause on the UD sets: the huge-page pacing tests a buffer's fault rate only after a chunk that is not its last,
  and the cache's blocks are under 1 GiB there - one chunk each (`src/core/pinned.cu:319`).
- Nothing left in: the refill and the swaps (no drive reads), the checkpoint marks (+0.4-0.7 s), parking and saves (between
  requests), the server's loop look. Not measurable by reading: GPU 1's time shared with the display.

**#135, a likely cause.** A lone token's IQ2_S / IQ2_XS / IQ1_S rows take ggml's dot and a window's the grouped kernel
(`native_expert.cpp:184-214`), so they round differently alone than in a window - `STRATA_IQ_MT_MIN`'s known rule, not FP8.

**Not determined.** The split of the 22.8 ms between the chain and the GPU branch, the IQ kernels' and the head's own times
(one nsys trace through dec_layers.py on a UD config answers them); UD-Q2_K_XL's decode line and its bytes a turn; whether
the client returns the reasoning text (a windows-only turn's checkpoint depends on it); whether SMT siblings help the pool.


## 75. The exact items of §74's audit, built and measured (2026-11-10)

**The ask** (the user, after §74's list): address every item that causes no quality regression; the full MXFP4 set to gain
too; few tests. So: exact only. Every change below computes the same values in the same order, and each is checked that way -
the kernels against their references first, then the engine's own logits against the binary before it. §74's
output-changing options (#149) are untouched.

**How it was checked** (`moex/tools/dsv41/checks/speed_ab.py PREFIX [S] [X] [K] [R]`, new; `--cmp A.json B.json`). One
engine at a time through the serve protocol: no server, no conversation folder, no profile save.
- **S**, the config as deployed: three prompts (the plan's text, code, prose) through the family's template, greedy, 384
  tokens each after a 64-token settle; tok/s and the engine's own decode lines.
- **X**, bits: the route bias off, `--no-adaptive`, the slots pinned (`--expert-slots`, `STRATA_GLM_STAGE_SLOTS`), a 30 GiB
  cache, no warm fill - per request the first-token logits hash (`STRATA_GLM_FIRST_TOP`) and the greedy tokens, over a read,
  its reply, a short follow-up (decode windows) and a long one (a gated chunk). Two conversations, six requests.
- **K**: a 410-token follow-up across a checkpoint mark (`--prompt-cache-every 4096`); `K_EVERY=65536` is the same request
  with no mark in its way, the reference.
- **R**: the first conversation's three requests 12 times in ONE engine. From the second time on a request resumes at its
  checkpoint and decodes again, so many replies' windows are compared without a start a sample.
- `STRATA_GLM_EVERY_TOP=1` (new) prints every sampled row's hash, so two runs that part say at which token.

### The result

UD-Q2_K_XL as deployed (S, tok/s; the engine's wait for the GPUs a token in brackets). One run a row, and a row is good to
about +-3%: the lookup's drafts differ from run to run, and the two rows before the last differ by one matrix's rows a block.

| | plan | code | prose | all |
|---|---|---|---|---|
| the binary before (aaef063b) | 20.39 (23.6 ms) | 19.60 (19.7) | 17.93 (22.6) | 19.25 |
| + the CPU pool's 64-value decode | 20.53 (23.2) | 19.75 (21.8) | 18.57 (22.4) | 19.58 |
| + the FP8 kernels, as first ported | 21.49 (20.4) | 21.77 (18.3) | 19.00 (20.5) | 20.67 |
| + those kernels refined (marked blocks) | 22.39 (20.3) | 22.08 (15.9) | 19.06 (19.2) | 21.06 |
| + wo_b at 8 rows a block | 21.72 (19.3) | 20.53 (18.1) | 19.03 (19.5) | 20.37 |
| + layer 14's engram rows early, the slot check behind its copy (**the installed binary**, 9344f8c9) | 21.38 (19.9) | 21.81 (16.3) | 19.60 (19.7) | **20.89** |

**+8.5%** on the installed binary (19.25 -> 20.89), and +5.8% to +9.4% over the three runs with the final kernels (21.06,
20.37, 20.89). The steadier figures are the engine's own: the wait for the GPUs, the three prompts' mean, **22.0 -> 18.5-19.0
ms a token** (the installed binary: 18.6); the CPU pool's rate 46.0-47.3 -> 50.8-51.5 GB/s; the VRAM tier's upkeep 1.4-1.6 ->
0.8-1.1 ms a token.

The same bits: X's six requests give the first binary's hashes and tokens on the installed binary (`99fdbf0ac699c24b`,
`f9430ec883e12e1a`, `5e1e87134f0e0a2f`, `71876829eecebb79`, `f217ec310df34083`, `54e8d1864b89d33f`), and R's 36 requests gave
the first binary's table hash for hash (below; run on the build with the refined kernels).

### The CPU pool's IQ rows, 64 values at a time (#142)

This box has AVX-512 F / BW / VL / DQ / CD but neither VNNI nor VBMI, which `expert_layout.cpp:60-70` asks for, so every IQ
format ran the AVX-2 kernels (or ggml's dot) at 123-151 G weights a second where DRAM allows 190-295. The 512-bit kernels
that exist were written for VNNI machines and measured slower here: a scalar decode, and 512-bit products that lower the
clock.

`kIq256Wide` (`iq_avx2.cpp` `Fmt64<16/18/21/22>`, `iq_avx2_rows.inl` `row_dot64`): a group's **64** values decoded at once -
one 512-bit gather for both halves' grid entries, and the 64 signs as a **mask register** (`vpsubb` under a `k` mask in
place of the sign table's loads and shuffles; `pdep` folds IQ2_XXS's and IQ3_XXS's four 7-bit even-sign indices into bytes).
The products stay in 256-bit halves, into the same int32 lanes and the same float tail: the AVX-2 kernels' bits. No 512-bit
multiply, so the core keeps its AVX-2 clock. On where the CPU has AVX-512 F / BW / VL and BMI2 and no AVX-VNNI path;
`STRATA_IQ256_WIDE=0` turns it off. A lone token's IQ2_S, IQ2_XS and IQ1_S rows stay on ggml's dot, as before (their bits).

`iq_avx2_parity`: **0 failures**, the new variants (`wide`, `gather+wide`) among them, 1..8 tokens.

`native_expert_cpu_bench`, 20 threads, one token, UD-Q2_K_XL's own records (GB/s of weights; the plain read of the same
bytes: 77):

| layer | types (gate/up, down) | before | after |
|---|---|---|---|
| 3 | IQ2_XXS / IQ2_XXS | 39.07 | **50.80** |
| 8 | IQ2_XXS / IQ3_XXS | 43.43 | **55.88** |
| 20 | IQ2_S / IQ3_S | 49.97 | **52.66** |
| 26 | IQ3_XXS / IQ3_S | 51.60 | **65.08** |
| 35 | IQ3_XXS / IQ3_XXS | 49.64 | **65.00** |

Layers 26 and 35 go from 67% to 85% of the plain read's rate. In the engine the pool's own rate went 46.0-47.3 -> 50.8-51.5
GB/s a token. The decode gained less than the pool did (19.25 -> 19.58): the wait for the GPUs was the longer side, which
is what the FP8 work then shortened. The SMT siblings once more, 40 threads against 20: 64.76 against 65.22 GB/s - no gain.

### The FP8 dense kernels (#137, #138, #139)

§70 put the release's FP8 projections back and the decode's wait for the GPUs went 13.3-17.4 ms (the Q8_0 days) -> 21.4-22.8.
An nsys trace of this round (`--cuda-graph-trace=node`, `dec_layers.py`): on GPU 0 `gemv_fp8_rows` is 9.9 ms of a window's
18.9 ms of kernels, on GPU 1 9.75 of 21.0; then the VRAM experts' kernel (1.8 / 2.5 ms), the one-row window kernel (2.2 /
2.1 ms, with few drafts) and the head (2.1 ms, GPU 1).

- **Marked blocks (#137).** `fp8_block` sets bit 7 of a block's scale byte when the block holds an exponent-0 code (a zero
  or an e4m3 subnormal: 0.2-0.3% of the release's blocks, 0.008% of its codes - counted on the set's blk.5), and keeps a
  scale byte in [10, 127] (a matrix outside it keeps its BF16; all 322 still convert). A step whose blocks are unmarked
  builds each weight's F32 bits with **three integer instructions**: `q x 2^20 + sa` puts the exponent and the mantissa in
  place, with `sa = (S - 7) << 23` the scale's exponent, and `(q & 0x80) x 0xF00000` moves the sign from bit 27 to bit 31.
  A warp's lanes sit in one 32-block of each row, so "is any of this step's blocks marked" is the same for the whole warp:
  a branch that does not split it, taken for ~1% of the steps, to the general decode (`e4m3_sa`, out of line - inlined, the
  compiler turned the rare branch into instructions run for every weight). The power-of-two multiply is an integer add of the
  exponent field, so the value's bits are those of the float multiply. The three other readers of the scales mask the mark
  (`kFp8ScaleMask`: `gemv_fp8_w_kernel`, `gemv_fp8_w1_pre_kernel`, `fp8_widen_kernel`).
- **Windows (#139).** `gemv_fp8_rows_kernel<R, TT>` takes T = 1..8 columns, so a verify window (the prompt lookup's,
  DSpark's) no longer runs the one-row kernel, which §70 measured at 0.34-0.55x of BF16's.
- **The attention's heads in one launch (#138).** `gemv_fp8_heads` runs wo_a's 8 groups as one launch (grid.y = the group),
  the Q8_0 path's `WOA_BATCH` for FP8.
- **The quantize launches that fed nothing (#138).** `pj` and `heads_mm` run a BF16 or FP8 matrix on the F32 input and never
  read the quantized copy: four such launches a layer are gone (`pj_needs_q`). The shared expert's stays - the VRAM
  experts read its copy.
- **Rows a block.** 4, and 8 for a matrix of 8,192 values a row (wo_b: 120.5 -> 106.8 us a call), 4 again for windows of 5
  to 8 rows; `STRATA_DSV41_FP8_ROWS` names one for all.

`fp8_gemv_bench` - its shapes are the model's own now (n_embd 5,120, read from the set's blk.5; the first table's were 4,096
wide), its codes the release's share of exponent-0 blocks (uniform random codes put one in 87% of the blocks: the slow branch
of every decode), marked as the loader marks them. Against the same weights as BF16, **all 64 rows bit for bit** (8 shapes,
T = 1..8). A call's time, us, before this round -> now:

| shape | T = 1 | T = 4 | T = 8 |
|---|---|---|---|
| attn_q_a (5120 x 1280) | 23.4 -> **21.5** | 58.4 -> **30.3** | 76 -> **51.0** |
| attn_q_b (1280 x 32768) | 146 -> **129.6** | 495 -> **230.0** | 545 -> **380.2** |
| attn_kv (5120 x 512) | 11.7 -> **10.5** | 25.3 -> **15.5** | 32.4 -> **22.5** |
| shexp gate/up (5120 x 2304) | 40.6 -> **36.1** | 104 -> **52.2** | 135 -> **84.9** |
| shexp down (2304 x 5120) | 39.5 -> **36.0** | 112 -> **60.0** | 147 -> **96.1** |
| attn_output_b (8192 x 5120) | 133 -> **114.7** | 400 -> **193.8** | 500 -> **292.5** |
| attn_output_a (a group, 4096 x 1024) | 13.1 -> **11.5** | 36.5 -> **18.6** | 45 -> **29.6** |
| engram_kv (6144 x 25600) | ~511 -> **445.0** | - -> 790.9 | - -> 1169.8 |

One token 8-14% faster, windows 1.4-2.1x.

**The first port lost the one-token gain, and how it was found.** As first ported the kernel kept two tests a weight in its
inner loop - "is this row inside the matrix" and "is this column inside the window" - and decoded with the general function
inlined; `fp8_gemv_bench` then showed one token at 23.3 us where the kernel before ran 23.4 (the prototype: 21.4). The
user's reading of it ("revisit the port") was right. The body now computes all R rows with no test (a row past n_out gets a
tile of zeros and its sums are not written), takes one token by `if constexpr`, and marks the blocks. The same review found
the loader's bound one short: the general decode needs S >= 10, not 9 (at 9 an exponent-0 code's factor is 0.0, where
`e4m3_val` gives a denormal) - no weight of this model is near it.

**What bounds the shape now.** A probe kernel that only copies the tiles (no product at all) takes 60-66% of the whole
kernel's time: the global-to-shared copy is most of it. Reading each thread's values straight from global would skip that
copy, but a thread's values are then 16 consecutive ones instead of t, t + 256, ... - another summation order, other bits.
Two restructurings that keep the order were measured and neither helped: a tile's loads all issued before its decodes
(slower), and the rows stored four to a 32-bit word so one load serves four rows (within noise).

### The VRAM experts' IQ kernel (#141)

`mmvq_grouped_kernel` ran one warp a row, four rows a block, with no switch - `STRATA_MMVQ_GROUPED_ROWS` covers the K-quant
and MXFP4 kernels only (§20). It takes `STRATA_IQ_GROUPED_ROWS=4|8` now; one warp still computes a row, so every output is
the same bits. `mmvq_grouped_parity`: ALL PASS at 4 and at 8.

**No gain at 8, so 4 stays.** S with `STRATA_IQ_GROUPED_ROWS=8` against 4, the same build: the wait for the GPUs 18.0 / 19.7 ms
(code, prose) against 18.1 / 19.5. These kernels run while the CPU pool computes its experts, and the pool is the longer
side (25-28 ms a token against the VRAM experts' ~2), so their time is hidden behind it. The run's third prompt met a drive
episode (below).

### The turn that crossed a checkpoint mark (#145)

A periodic mark inside a short follow-up cut the read in two chunks, and a call of two chunks takes layer-major with the
tails deferred: every layer's experts streamed whole. `STRATA_DSV41_MARK_MIN` (4,096): a read of fewer new tokens keeps the
mark's deep checkpoint at its last turn token - a cut it makes anyway - and makes none at the mark.

K on UD-Q2_K_XL: the 410-token follow-up across the mark **37.9 -> 14.9 s**, its first-token logits and 32 greedy tokens
those of the same request with no mark in its way (`0dd3be5110726dd6`).

### The engram rows at the drive guard's floor (#147)

In an episode the guard asks for one row ahead, so a decode token's 48 rows fault one after the other. A window of 8 rows or
fewer (a decode token's, a short read's) asks for 32 pages whatever the guard's level: `STRATA_DSV41_EG_DECODE_ROWS` (32; 0:
the guard's own). The same rows in the same order: the same bytes.

**No gain shown, and §74's price for it was wrong.** §74 took ~3 ms a fault at the floor (144 ms a token). This round's runs
met episodes, with and without the change:

| request | binary | the token's drive wait | its engram rows |
|---|---|---|---|
| native, code (two episodes, 384 tokens in 157.8 s) | 77e08fbf, without | 348.8 ms | 9.13 ms |
| UD-Q2_K_XL, plan (one episode of 64.1 s, 384 tokens in 76.2 s) | with | 142.7 ms | 12.61 ms |
| native, plan (episodes, 384 tokens in 94.6 s) | with | 193.2 ms | 3.10 ms |

Without the change the rows were 2.6% of that request's drive wait, not the larger part: a 4 KB fault at the floor is far
under 3 ms. Three requests, each with its own share of episode, do not separate the two settings. What an episode costs is
the records: the guard's one 128 KB read at a time, 336-376 a second (~45 MB/s) while it lasts, whatever is asked.

### A start that stalls on fragmented memory (#133)

`fault_in_paced` tested the fault rate only after a chunk that was not its buffer's last, and the SSD cache's blocks are 64
slots - 557-861 MiB on UD-Q2_K_XL, one chunk each - so on the UD sets the test never ran and huge pages were never given up,
however slowly they faulted in. The rate is tested after a buffer's last chunk too (the caller passes one flag for all its
blocks, so the next block takes the decision). Logic only: the state that provoked #133 was not recreated; this round's
starts were as before (the cache's pinning 0.4 s, ready in 54-57 s on the test cache, 71-74 s deployed).

### The drive's episodes in the request line (#148)

`DriveGuard::episodes()` and `floor_s()` had no caller but the test. The DONE line carries two more fields - the episodes
begun during the request and the seconds the guard held the reads back - and the server keeps them per request
(`drive_episodes`, `drive_slow_s`), so the rate shows in the user's own runs with `"diagnostics": false`. The serve suite:
854 OK (11 skipped).

### Layer 14's engram rows, while the GPU waits (#140, in part)

The engram layers (1 and 14, both on GPU 0) dequantize the window's rows, run them through wkv - a 6,144 -> 25,600 gemv,
445 us a token with this round's kernels - and then gate the streams. The first two steps read the rows and nothing of the
streams. In a MoE layer's decode graph the GPU publishes the routing, runs the shared expert and the VRAM experts (~0.17 ms)
and then spins until the CPU pool's rows are in (the pool: 0.55-0.68 ms a layer on UD-Q2_K_XL, ~1.2 on native). Layer 14's
rows' half now runs in layer 13's graph just before that wait, into a buffer of its own, and layer 14 runs its gate alone
(`Stage::engram_rows`, `eg_pre_kv`; `STRATA_DSV41_EG_PRE=0`: as before). Never later than before: the old order cost
max(pool, GPU) + 445 us, the new one max(pool, GPU + 445). The same kernels on the same rows: X's six hashes.
S does not resolve it: the wait's three-prompt mean was 18.5 and 19.0 ms in the two runs before it and 18.6 with it, and its
most is 0.45 ms. It stays on the argument above and the bits.

Not moved: layer 1's own half. With `EG_LATE` its rows arrive while layer 0 runs, and the host tells the graph so only after
layer 0's experts; moving it needs the fetch's thread to tell the graph itself. And the head (2.1 ms of GPU 1 a token):
over both GPUs it is still open, and it cannot go back to FP8 as the projections did - 93.7% of `output.weight`'s BF16
values use more mantissa bits than e4m3 has (the first 12.8M values of the set's file 9).

### The slot check behind its copy (#143)

A swap's copy landed, and before the next window the host launched the check's hash kernel on the compute stream and waited
for it: "the slot check" of the decode line, 0.45-0.61 ms a token on UD-Q2_K_XL and native. The kernel is queued on the copy
stream right behind the copy now (`queue_slot_check`: the parts kernel over the slot's three slices, the sums into mapped
memory), so the copy's event completes with the sums written, and `finalize_swaps` compares them before the slot becomes
resident - a slot that differs never does: it stays empty, the expert keeps its RAM blob or cache record, and the CPU
computes it (before: it was resident until the check, then left). `STRATA_DSV41_SLOT_CHECK_ASYNC=0`: as before.
S on the installed binary: the slot check 0.52-0.61 -> **0.00** ms a token, the tier's upkeep 1.4-1.6 -> 0.8-1.1. The check
itself, tested: with `STRATA_GLM_SLOT_CHECK_FAULT=5` (the fifth such check's sum read as a mismatch) the engine printed one
"expert slot 430 (layer 11, expert 150) differs from the file in its gate rows - the slot stays out of the tier, the CPU
computes the expert" and went on decoding.

The other half of #143, the first-fill hashes (0.5-0.9 ms a token in a start's first replies), is a start's cost and not a
steady one: each expert is hashed once a run. Left as it is.

### Run to run: what repeats and what does not

- **R, 12 times the same three requests in one engine.** The reply's lookup drafts differ from one repetition to the next
  (9 of 15 kept, then 6 of 13, then 7 of 10 or 7 of 7) and so do the two follow-ups' logits: four draft patterns, four
  pairs of hashes - and every repetition with the same pattern gives the same pair (four of them one, six the other). The
  binary before gives **the same table, all 36 hashes**. So the output follows the windows the drafts made, in both
  binaries alike. This is #135's difference seen from the other side: a lone token's IQ2_S and IQ2_XS rows take ggml's dot
  and a window's the grouped kernel (`native_expert.cpp:184-214`), and the lookup's policy times its rounds
  (`glm_spec.cpp:426`), so the windows are not a function of the text alone.
- **X, a fresh engine each.** 16 runs of the first conversation across this round's builds: 15 the same at every hashed
  token - among them four with these kernels against two with the kernels before, every sampled row hashed, and one of
  each with another program holding GPU 1 at 100%.
- **One run parted** (the first full build's): its follow-ups' logits differed by 0.1-0.8 with the same draft pattern as the
  others and no line of its log unlike theirs but the timings. The same binary run again gave the others' bits, and no run
  since has parted. Not explained (DSV41_BACKLOG #150).

### The full MXFP4 set

`strata-dsv41-native.json`, S, the binary it ran (77e08fbf) and this round's build (before the last two changes), one run
each. The drive was in its slow mode for most of both: 3 episodes in the old binary's 9 minutes, 6-7 in the new one's 12
(the guard's lines). This round's ~40 engine starts each read the same hottest records again, and the guard's own count says
what that did. So tok/s compares only where a request met no episode:

| | plan | code | prose |
|---|---|---|---|
| 77e08fbf | **13.18** tok/s (wait 22.3 ms; pool 48.9, 23.0 of it the drive) | 2.43 (episodes; wait 19.2) | 11.30 (wait 21.8; pool 61.7 / 34.6) |
| this round | 4.06 (episodes; wait 19.5) | **12.84** (wait 17.6; pool 56.3 / 27.3) | stopped |

The wait for the GPUs: 22.3 -> 19.5 ms (plan), 19.2 -> 17.6 (code). The pool without its drive time is unchanged (25.9 -> 26.4
ms on plan): MXFP4's CPU kernels were at DRAM's rate and are untouched. No request ran clean in both binaries, so there is no
tok/s pair; 1.6-2.8 ms off a ~76 ms token is +2-4%, read from the counters.
Bits on this set (`speed_ab.py X` with `X_PROMPTS=prose X_LONG=0`: a 66-token conversation through decode windows alone, the slots pinned 600 + 450, a
30 GiB cache), 77e08fbf against the installed binary: both requests' logits hashes and greedy tokens the same
(`0ed559195dc6a5a6`, `152ea164ba15582b`) - the new binary's run through two drive episodes. This set's prompt path was not run
in this round: a first read streams most of its 269 GiB, and the drive was in the state above. That path's code is the sets'
common one, checked on UD-Q2_K_XL.

### Memory

The deployed start fills the cards to their reserves: 1,305 + 890 slots on UD-Q2_K_XL, 700 / 2,053 MiB left. Sampled every
2 s through an S run the cards' highest use was 19,435 / 18,076 MiB of 20,480. (The 2 GB and 3.5 GB seen free during this
round were the bit checks, which pin 1,200 + 800 slots.) GPU 1's 2 GiB is the display's and the image encoder's. GPU 0's 700
MiB (`--vram-reserve-mib`, Strata's value) was ~1,045 MiB free at the run's highest use: lowering it to 300 would give ~35
more slots on UD-Q2_K_XL (~22 on native), est. ~0.3 ms a token. Not changed: what the reserve covers at a 500K+ context
with pictures was not measured, and a request that fails for memory costs more than the slots give.

### Not done from §74's list

- **#140, the rest**: the head over both GPUs (est. 0.8 ms a token) and layer 1's engram rows (est. <= 0.45 ms).
- **#147 (a)**: layer 1 first in a prompt chunk's engram fetch (est. -3 s a cold 16K read).
- **#144**: (a) the `irdma` blacklist is the user's (sudo): `echo "blacklist irdma" | sudo tee /etc/modprobe.d/blacklist-irdma.conf`,
  then `sudo rmmod irdma` or a reboot - 3.0-4.2 GiB more for the cache; (b) the bounce slots lent to the cache in decode
  (1.7 GiB on UD-Q2_K_XL: ~2% more cache); (c) GPU 0's reserve (above).
- **#146**: a follow-up on the UD sets over both cards' links - large; its CPU share is rounding-only, so with #149.
- **#149**: the options that change the output - untouched, as asked.

(§76, the same evening: #140's rest and #147 (a) built; #144's bounce slots argued out of; #146 open with what it needs.)

## 76. The rest of §74's list: the head over both GPUs, the engram rows earlier, and a bit check without the lookup (2026-11-10, evening)

**Why.** §75 left five of §74's items open. The user's goal names every item, so this round takes them: two are built and
measured below (#140 whole, #147's other half), one is argued out of (#144's bounce slots), one stays open with what it
needs (#146). The same rule: exact only, checked on the engine's own logits.

### The bit check, without the prompt lookup - and #150 explained

The first run of this round's first build gave the plan conversation's two follow-ups other logits (`4971c4a9eb4d6adc`,
`5fd7d796fca1dc6e` for `f9430ec883e12e1a`, `5e1e87134f0e0a2f`) - **the very pair of §75's one unexplained run** (#150) - and
so did its second run. Two runs with every sampled row hashed (`STRATA_GLM_EVERY_TOP=1`), one of each outcome:

- the reply's rows are the same up to position 2671, its 49th token, and part at 2672;
- one outcome hashes rows 2672 and 2674 there, the other 2672 and 2673: the prompt lookup offered its drafts **one token
  later**. Each request's totals are the same in both (87 windows, 9 of 15 drafts kept, the same table by depth).

The lookup's policy is timed (`glm_spec.cpp:426`: it is told each round's milliseconds), and on a set with IQ2_S / IQ2_XS
rows a token computed alone takes ggml's dot where one computed in a window takes the grouped kernel (#135,
`native_expert.cpp:184-214`). So a token faster or slower can move a window by a position, and the bits from there on with
it. The head's 0.7 ms (below) moved it in 2 runs of 2; the builds before it moved it in 1 of 16. The outcome that moved is
itself the same bits across builds: this round's build gave §75's odd pair exactly.

So two binaries are compared **with the lookup off** from here on (`speed_ab.py`: `X_LOOKUP=1`, `--lookup-window 1`; 0 is
the engine's default size, not off). The old binary's six requests so: `99fdbf0ac699c24b`, `8e5183bcaad97c76`,
`a78dcbfae00647b4`, `71876829eecebb79`, `851dbbb88cffb28a`, `518f40357260364d` - and this round's binary gives the same six
(logits and tokens; binary 0be985bf). #135 stays what it was: with the lookup on, a reply's bits follow where its windows fall.

### The head over both GPUs (#140)

The head is BF16 (it has no FP8 values, §75): 129,280 rows of 5,120, 1.32 GB a token, 2.1 ms of GPU 1 after its layers
while GPU 0 has nothing to do. `load_dense` now loads it as two runs of rows - the first 45% (58,176 rows) on GPU 0, the
rest on GPU 1 (`STRATA_DSV41_HEAD_SPLIT=P`: P% on the first GPU, 0: whole on the last; only with two GPUs and no drafter:
those read the whole head on the last GPU). `Stage::head`: the window's normed rows go to GPU 0 through mapped memory behind
an event, each GPU runs `gemv_bf16_w` over its rows at once, GPU 0's logits come back the same way and the two runs are
joined (one row: GPU 1's straight into place). `gemv_bf16_w` is one block a row on either card, so each logit is the dot
product the whole head's launch gave it. VRAM moves with the rows: 1,305 + 890 expert slots became 1,253 + 942, the same
2,195.

S, UD-Q2_K_XL as deployed (tok/s; in brackets the wait for the GPUs, then a token's "rest": its other time less the tier's
upkeep - the head and the sampler's wait):

| | plan | code | prose | all |
|---|---|---|---|---|
| §75's binary (9344f8c9) | 21.38 (19.9; 3.0 ms) | 21.81 (16.3; 2.1) | 19.60 (19.7; 3.8) | 20.89 |
| + the head over both GPUs | 22.56 (19.8; 2.3) | 21.88 (16.9; 1.5) | 19.77 (19.1; 3.0) | 21.33 |
| + the engram rows earlier (**the installed binary**, 0be985bf) | 22.28 (20.0; 2.3) | 20.61 (18.2; 1.9) | 20.66 (19.6; 3.0) | **21.15** |

The head's part is the measured one: the "rest" **0.6-0.8 ms a token** lower. Against the binary before §75 (19.25 tok/s) the
two runs are +10-11%; a run is good to +-3%.

### The engram rows earlier (#140's other part, #147's)

**Layer 1's rows in layer 0's wait.** §75 moved layer 14's rows' half (the dequant and wkv, 445 us) into layer 13's graph;
layer 1's stayed, because its rows arrive while layer 0 runs (`EG_LATE`) and the host told the graph so only after layer
0's CPU experts. Now the fetch's own thread tells the graph (`eg_ring`; `STRATA_DSV41_EG_RING=0`: layer 14's alone): the
graph's wait for the rows sits in layer 0's graph before layer 1's rows' half, and it is released by whoever comes second -
the helper as its fetch ends, or the host when it arms the wait after launching the graph and finds the fetch done
(`eg_help_arm`, under the helper's lock; the word's value is taken then, so a window that fails before its launch leaves no
word behind, and a fetch that fails writes none: the host finds it and releases the GPUs as before).
S does not resolve the two layers' <= 0.9 ms: the wait for the GPUs, the three prompts' mean, was 18.6 ms on §75's binary and
18.6 and 19.3 on this round's two builds, a run good to ~1 ms (the code prompt's alone ran 16.3-18.2 with its drafts). They
rest on the argument (never later than before) and the bits. The path a failed fetch takes, tested (`speed_ab.py F`: the
40th fetch made to fail, `STRATA_GLM_EG_FAIL`): the request failed with the fetch's error and the engine ended with exit status 1 (as it does
after any failed request: the server starts it again), 40 s after the request was sent - no GPU left at a wait.

**A prompt chunk's rows layer by layer (#147).** A chunk's fetch read both engram layers' rows and layer 1 waited for all
of them. Where the table is the GGUF's and read O_DIRECT (`eg_split_ok`; `STRATA_DSV41_EG_SPLIT=0`: as before) `eg_rows`
takes a run of layers: the chunk loop's worker reads layer 1's rows, tells layer 1, then reads layer 14's (a second
future; the next chunk's fetch waits for it: one fetch at a time); a layer-major window's worker makes two passes over its
chunks - every chunk's layer-1 rows, then every chunk's layer-14 rows. Layer 14 waits for its own before it runs. The same
rows into the same places.
Measured on UD-Q2_K_XL: a 2.6K-token chunk waited 0.7 -> 0.4 s for its rows. A cold 16K read (two chunks, one layer-major
window; `speed_ab.py L`): 5.7 -> 5.1 s of waits, 47.1 -> 46.1 s in all, the same logits (`384f991bfacb6458`). Less than §74's
-3 s, and why: that read is drive-bound, and layer 14's rows, no longer waited for at layer 1, are read while the stream
reads its records - the drive's time either way (the fetch's own time went 6.8 -> 18.2 s beside the stream, and layer 14
waited for the rest of it). Where a read's records come from the cache the old wait was dead time; that is where this helps.

### Not built: the bounce slots as cache slots (#144)

The prompt stream's 128 bounce slots (1.7 GiB pinned on UD-Q2_K_XL, 2.2 on native) idle in decode, and the cache has a
state for slots that are away (`lent`, §63) that every slot choice already honours (`ssd_victim`, `block_free`, the warm
fill); the reader's thread waits out its own reads before it ends. So they could be cache slots between prompt calls. What
that buys, by estimate only: +1.8% of the cache's slots on UD-Q2_K_XL, +2.7% on native. Every streaming turn takes them
back and so empties them; of their records perhaps a fifth would have been hit again within the turn (a decode token
touches ~1% of the cache). With §74's one figure for the cache's size (+5.5% of cache, -18% of the drive-bound routings):
UD-Q2_K_XL saves ~29 reads a 300-token turn and loses ~25 - nothing; native saves 86-124 and loses ~26, est. 1-2 ms a
token and ~9% of the decode's drive bytes. Neither is measured, S cannot resolve them, and native's S was unusable today
(§75: the drive); the price of a mistake here is a cached record overwritten by a prompt's read, which no check catches on
the CPU's side. Not built on an estimate. What would decide it: the tier simulator on a native routing trace with 128 more
slots that empty at each turn. #144's (a), the `irdma` line, is the user's; (c), GPU 0's reserve, stays as §75 left it.

### Open: a follow-up over both GPUs' links (#146)

What a follow-up costs on the UD sets, from this round's logs (UD-Q2_K_XL, deployed): a 2.6K-token read streams 10,245
experts (110 GiB) in 24.6 s, a 600-token one ~8,200 (88 GiB): each routed expert's whole record goes to a GPU for ~18
tokens' rows. Two ways out:
- **The CPU's share** ((a)): the experts few rows route to computed by the CPU pool from the cache, as a decode token's are -
  no upload. The pool reads a record in ~0.2 ms; est. a 600-token follow-up from ~12 s to 5-6. It is **rounding-only**
  (the CPU's sums for the GPU's), the class §74 set aside and the user asks a KLD check for; and `moe_prompt_cpu` /
  `moe_prompt_share` read experts in place, which the SSD tier's records are not (`glm_prompt.cpp`: `rio` turns both off).
- **Exact, over both links** ((b)): `moe_prompt` keeps every routed pair's row (`mb.dm`) and combines them in routing
  order at the end, so a layer's experts can be computed on either card: GPU 1 would take two thirds of a GPU 0 layer's
  streamed experts through its x16 link and send their rows back. That is a second stream on GPU 1 for GPU 0's layers -
  its ring, its reader, its loan, the stop and failure paths - the largest change to the prompt path since layer-major.
  Not started.

### What was checked, and the install

On the installed binary (0be985bf; names are the first 8 hex digits of a binary's SHA-256), UD-Q2_K_XL, the lookup off:
X's six requests, K's crossing against the same request with no mark (`83eda8bbc8b9df22` both), L's 16K layer-major read -
the old binary's logits and tokens in each. The full MXFP4 set: the 66-token decode-only conversation, 77e08fbf against
0be985bf, both requests the same (`0ed559195dc6a5a6`, `152ea164ba15582b`); its prompt path was not run on that set (§75).
`iq_parity`, `iq_multi_parity`, `mmvq_grouped_parity` pass; `serve_smoke.py` ALL PASS (ready 73 s, the first reply 19.2 s,
the picture's 11.2 s). `strata-dsv41-ud-q2kxl.json`, its launcher and the daily `build-dsv41/strata-glm` (native) name it;
9344f8c9 (§75's), aaef063b and 77e08fbf stay beside it; `strata-dsv41-ud-iq1m.json` still names aaef063b (not checked on
that set).

**2026-11-15:** the UD-IQ1_M set went at the user's word (the model no longer used): its files 1-4 (139.8 GB), both its configs,
`launch-dsv4.1-flash-ud-iq1m.sh`, its expert profile, saved conversations and kernel-check folder, and
`moex/tools/dsv41/checks/dspark_ab.py`, which read both configs (DSV41_BACKLOG #132, #134). The engine's IQ1_S kernels, its IQ4_NL
engram reader and their checks stay; aaef063b stays as the binary before 0be985bf.
