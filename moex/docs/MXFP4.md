# MXFP4 at full speed: the full DeepSeek-V4-Flash and MiMo-V2.6 releases on this box (moex, from 2026-10-10)

**Status (2026-10-10 21:15): concluded - the user called it a failure.**
- On 125 GiB the exact releases reach ~45-50% of today's decode in the emulation; even hindsight-perfect placement reached 68%.
- The wall is the ~8 first-touch NVMe reads per token that no cache or predictor can serve, on a drive the desktop shares.
- The file-tier code (file tier, cold cache, burst, recency, route dump, emulation scripts, MXFP4 launchers) is NOT in the tree. It is kept in `git stash` ("moex: full-MXFP4-on-125GiB experiment ...").
- In the tree: the MXFP4 CPU kernels (§3b; mx_avx2 / mx_avx512, bitwise equal to ggml-cpu) and native_expert_cpu_bench.
- Next chapter: custom ~126-129 GB quants of both models (see the user's memory notes).

**Goal (user, 2026-10-10):** run the full MXFP4 releases of DeepSeek-V4-Flash(-Vision-Exp) and MiMo-V2.6-Flash-MOPD on
this box (125 GiB RAM: 4 x 32 GB in 4 of the Xeon Gold 6148's 6 channels; 2 x RTX 3080 20 GB; PCIe 3.0; one NVMe at
~2.3 GB/s) as fast as the ~3-bit quants run today, with no loss: the experts in their released MXFP4 bits.

## 1. The budgets

| | DSV4 today (UD "IQ3_S", IQ3_XXS experts) | DSV4 MXFP4 (ggml-org) | MiMo today (AesSedai BPW3.0) | MiMo MXFP4 (ggml-org) |
|---|---|---|---|---|
| routed experts | 99.7 GiB, 3.09 bpw | 137.1 GiB | 101.1 GiB, 2.87 bpw | 149.8 GiB |
| one expert (4096 x 2048 x 3) | 9.19 MiB | 12.75 MiB | ~8.6 MiB (mixed) | 12.75 MiB |
| routings per token | 43 x 6 = 258 | 258 | 47 x 8 = 376 | 376 |
| fits VRAM slots + pinned RAM (~26 + 82 GiB) | yes | no: ~28 GiB in place in the GGUF | yes | no: ~48 GiB in place |

- **Capacity:** the pinned RAM budget is MemAvailable minus the 28 GiB headroom (the user's rule after the 10-07 freezes).
  - Whatever doesn't fit is read in place from the file mapping: resident RAM mode's "what fits", glm_tiers.cpp.
  - Page-cache pages are reclaimable, so a page-cache tier inside the headroom keeps the rule's purpose: no pinned memory, no compaction.
- **CPU bytes per token:** the CPU pool computes every routing the VRAM tier misses, which is most of the decode time.
  - MXFP4 experts are 1.39x IQ3_XXS's size, and fewer of them fit in the same VRAM.
  - DSV4: ~1.2 GB/token -> ~1.7 GB. MiMo: ~1.9 -> ~3.2 GB.
  - At today's tokens/s, MiMo MXFP4 would need ~68 GB/s from four DDR4-2666 channels (85 GB/s theoretical, maybe ~65-70 usable).

## 2. Measurements (2026-10-10)

**CPU kernels** (build/native_expert_cpu_bench: real GGUF rows, the pool's blob-form entry points; one thread; the
user's server was busy, so these are noisy):

| tokens per expert | IQ3_XXS (DSV4, clamp 10) | MXFP4 |
|---|---|---|
| 1 | 3.40 ms per expert (2.84 GB/s) | 2.54 ms (5.26 GB/s), 3.69 ms with clamp 10 (recheck quiet) |
| 2 | 4.02 ms | 4.81 ms |
| 3 | 4.96 ms | 7.03 ms |

- The IQ kernels are compute-bound (lookups): 2.8 GB/s per core x 20 is ~57 GB/s, and the pool sees ~43.
- MXFP4 is cheap per byte, so twenty cores would outrun four channels: decode turns DRAM-bound.
- **Unmeasured, decisive:** the 20-thread GB/s of both on this box. scratchpad idle_bench.sh runs it when the server is idle.

**NVMe** (scratchpad fill_probe / pop_probe on uncached ranges of an unused GGUF; 4.46 MB ranges = one MXFP4 slice):
- MADV_WILLNEED never brought a range fully in, even after 10 s.
- Touch-faulting gives 0.73 / 1.6 / 2.1 / 2.3 GB/s at 1/4/8/16 parallel ranges.
- MADV_POPULATE_READ gives 0.99 / 1.09 / 1.74 / 1.96 GB/s: synchronous, every page resident and mapped.
- So an uncached expert faulted in by the pool costs ~6 ms. It must be populated ahead, or held in RAM.

## 3. Mechanisms built

1. **mx_avx2.cpp: multi-token MXFP4 rows.**
   - Bitwise equal to ggml-cpu's AVX2 `ggml_vec_dot_mxfp4_q8_0` per token: the same lanes, FMA order, horizontal sum and tail.
   - `build/mx_avx2_parity` checks nt 1..8, both expert shapes, the scalar tail, E8M0 0/1/2/160, and the native entry points with and without clamp.
   - Sharing the nibble lookup saves little (~460 ns per 4096-row x token either way), so it takes groups of >= 4 tokens (`STRATA_MX256_MIN`).
2. **glm_filetier.cpp: the file tier.** Active only when experts are read in place; the current ~3-bit configs fit, so it's inert for them.
   - *Demand:* a layer's in-place misses go to 8 threads (`MADV_POPULATE_READ`), and the pool claims them last (`Engine::Impl::file_first`, in both decode paths).
   - *Warm set:* the in-place experts, hottest first by the profile, held in the page cache up to MemAvailable - 6 GiB (`STRATA_GLM_FILE_RESERVE_GIB`). A nice-10 thread re-checks every 15 s and refills what the OS took back.
   - *Prompt look-ahead:* the stream issuer's next 16 entries are populated ahead of their pageable copies. The stream is (layer, id) ordered, so it is known in advance.
   - Switches: `STRATA_GLM_FILETIER=0`, `STRATA_GLM_FILE_THREADS`.

## 3b. The GPU window (2026-10-10 17:49-18:20; logs moex/out/mxfp4-window-*.log, mxfp4-emu-*.log)

**CPU, 20 threads, quiet box (real rows, clamp 10 where DSV4's):**

| per expert | IQ3_XXS (DSV4 today) | MXFP4, ggml-cpu rows | MXFP4, mx512 (new) |
|---|---|---|---|
| 1 token | 47.1 GB/s, 4,889 experts/s (66% of the read ceiling: compute-bound) | 68.7 GB/s, 5,136-5,197/s (96%: DRAM-bound) | 69.7 GB/s, 5,216/s |
| 2 tokens | 4,144/s | 3,565/s | **4,746/s** |
| 3 tokens | - | 2,536/s | **3,802/s** |

- A plain 20-thread read of the same memory: ~72 GB/s. That is this box's DRAM ceiling: 4 x DDR4-2666 = 85 GB/s theoretical.
- Single thread: MXFP4 5.6 GB/s vs IQ3_XXS 2.6; the clamp costs nothing.
- **mx_avx512.cpp:** two blocks per zmm, weights offset to unsigned with an exact lane correction. Bitwise = ggml for nt 1..8 (mx_avx2_parity). Default from 2 tokens (`STRATA_MX512_MIN`, `STRATA_NO_MX512`).
- So an MXFP4 expert costs the CPU 6% (1 token) to 15% (2 tokens) LESS than today's IQ3_XXS one.

**End to end: DeepSeek-V4 UD-IQ3_S as an MXFP4 file would be placed.**
- A 38 GiB anonymous balloon put ~28 GiB of experts in place in the GGUF, ~3,100 of them (28%). Real DSV4 MXFP4 would have ~2,450 (22%).
- 8K prompt, 384 tokens, MTP w2, page cache dropped first:

| run | prefill | decode | pool ms/token |
|---|---|---|---|
| today's placement | 548.6 T/s | 24.6 T/s (the first decode after load) | 20.6 |
| in place, file tier OFF | 153.3 | 15.6* | 45.5 |
| file tier ON (demand populates) | 261.5 | 14.2 | 47.3 |
| + pinned staging of in-place stream entries + parallel warm | 301.5 | 13.7 | 47.1 |
| + cold misses read O_DIRECT (private buffers), reserve 6 | 311.6 | 7.6 | 98.6 |
| + three slices per cold expert in parallel, reserve 12 | 305.6 | 10.2 | 65.7 |

*OFF's text accepted more drafts (2.34 vs 1.6 tokens per window). Per trunk pass, OFF was 150 ms vs ON's 115.

**The placement's ceiling (18:25-18:30, the same emulation, reserve 12):**

| placement | decode | cold reads |
|---|---|---|
| the learned profile (static) | 10.4 T/s | 53.3 GB |
| ORACLE: this text's own routing (--profile-save, then the profile) | 14.3 T/s (+38%) | 23.6 GB |

- Offline, `scratchpad profile_cross.py`: the base DSV4 profile's hottest 93.5% leaves 6.1% of the learned (real-usage) profile's routings outside, against 0.69% for the learned profile's own placement. Static placements miss a lot, and adaptivity is worth up to ~+38% here.
- But even perfect placement leaves ~58% of today's decode: the routing's long tail sends ~2% of routings to the ~12% of experts outside RAM, ~5 cold NVMe reads of ~10 MB per token.
- So on 125 GiB exact MXFP4 DSV4 tops out near 55-60% of today's decode, however adaptive the tiers are.

**The adaptive RAM tiers (18:52-19:02; a 512-token warm-up then a measured 384 in one load, the 8K prompt each time):**

| | measured decode | pool ms/token | cold reads | warm set at the end |
|---|---|---|---|---|
| static RAM tiers | 8.41 T/s | 90.8 | 15.1K (147 GB) | budget 8.84 of 15 GiB, 1,847 refills |
| adaptive (746 pinned <-> file exchanges, 4 re-ranks) | 8.84 T/s (+5%) | 85.3 | 16.0K (159 GB) | budget 12.17 of 15 GiB, 253 refills |

- With swap full (SwapFree ~5 MB), the page cache is all the kernel can reclaim. Of the 28 GiB headroom only ~9-12 GiB holds as a stable warm set; the rest is the desktop's and the kernel's working room.
- So the effective RAM for experts is ~82 pinned + ~10 cached + 25 VRAM = ~117 GiB against DSV4 MXFP4's 137 GiB.
- ~10 cold experts per token is ~134 MB, so ~17 T/s is the NVMe's (2.3 GB/s) ceiling with perfect overlap.
- Adaptivity helps (the oracle: +38%), but cannot lift that ceiling.

**The NVMe's ceiling (19:12, scratchpad direct_probe.c, O_DIRECT preads over an unused GGUF):**

| threads | 1 MiB reads | 4 MiB | 8 MiB |
|---|---|---|---|
| 8 | 2.78 GB/s | 2.73 | 2.72 |
| 16 | 2.77 | 2.72 | 2.70 |
| 32 | 2.69 | 2.72 | 2.69 |

- ~2.75 GB/s whatever the depth or size: the drive's own limit (PCIe 3.0 x4). The file tier's 8 threads already reach it.
- So the ~2.3 GB/s above (populate / faults) understated the O_DIRECT path by ~20%, and the cold tail's ceiling is ~20 T/s rather than ~17.

**Where the oracle's 62 ms per token goes (18:28 log):**
- The pool spends 27 ms computing, then waits ~18 ms per token for cold reads (~6 per token, ~61 MB).
- At 2.75 GB/s that is 22 ms of NVMe time per 62 ms token: the drive idles ~65% of the time, because a read is issued only once its layer's routing is known.
- The oracle's profile was 95% the prompt's own routing (8,192 prompt tokens against 384 generated: the prompt path counts its routings too).
  So "the oracle" is what placing by the conversation's own prompt would give, with no hindsight needed.
  Hence the prompt's burst (below) and, next, reads issued ahead of the routing.

**Learned:**
- The page cache cannot hold a stable warm set near its limit. At reserve 6 the kernel evicted it under ordinary pressure (2,000+ refills in a run); at reserve 12 it is stable (281) but smaller, so more cold reads.
- Either way ~6-12 experts per token come off the NVMe. One O_DIRECT stream reads ~1 GB/s, the drive ~2.3 GB/s, and a layer waits for its cold experts.
- With ~28 GiB of experts beyond the pinned budget, the file tier gets this box to ~40-58% of today's decode and ~56% of today's prefill. CAPACITY, not compute, is the wall.

**MiMo's own experts today (BPW3.0), 20 threads, 1 token (19:10):**
- Layer 30, IQ2_S gate/up + IQ3_S down, 8.56 MiB: 47.8 GB/s = 5,322 experts/s (66% of the read ceiling: compute-bound).
- Layer 10, IQ3_XXS gate/up + MXFP4 down, 10.38 MiB: 58.0 GB/s = 5,333/s.
- MXFP4 runs ~5,200/s on 4 channels: about the same per expert. On 6 channels its DRAM-bound rate scales (~+45%) while the compute-bound IQ kernels do not.
- So with the RDIMMs, MiMo MXFP4 should be ~today's speed or slightly faster, even with ~20-25% more experts on the CPU (fewer MXFP4 slots fit in VRAM).

## 4. Today's placements (the engine logs, strata-dsv4.log / strata-mimo.log, 2026-10-10)

| | VRAM slots | pinned RAM copy (MemAvailable 109-110 GiB, 28 GiB headroom) | lend region |
|---|---|---|---|
| DSV4 UD-IQ3_S | 1575 + 1269 = 2,844 x 9.19 MiB = 25.5 GiB (GPU 0 705 MiB free, GPU 1 2,054 MiB) | 81.9 GiB: complement 74.1 GiB | 730 experts |
| MiMo BPW3.0 | 1470 + 1332 = 2,802 x 8.56 MiB = 23.4 GiB | 81.3 GiB: complement 78.2 GiB | 271 experts |

Those layers' MXFP4 blobs (DSV4: 3 tensors; MiMo: 5 down tensors) were larger than the slots, so they never entered VRAM.
The GPU's MXFP4 decode kernels (`native_mxfp4_mmvq`, grouped too) are covered by `build/mxfp4_parity` (10-07).
In MXFP4, the same 25.5 GiB holds ~2,050 DSV4 experts (18.6%) and the same 23.4 GiB ~1,840 MiMo experts (15%).

## 5. Expected (estimates, to be replaced by runs)

**DSV4 MXFP4 on this box:**
- VRAM ~2,140 slots (19% of experts), pinned ~6,590 (60%), warm ~1,770 (16%). That's ~95% of experts in RAM or VRAM, ~99.7% of routings by the profile.
- About 0.6 cold experts per token from the NVMe (~4 ms unhidden).
- ~19% more routings on the CPU than today.
- If MXFP4 runs DRAM-bound at ~60-65 GB/s, an expert costs the CPU about what an IQ3_XXS one does now.
- That gives ~85-90% of today's decode. Prefill: about today's, with the stream ~1.5x the bytes overlapping compute.

**MiMo MXFP4:**
- ~1.7x the CPU bytes per token, at the DRAM ceiling, and 48 GiB in place with only ~20 GiB that can be warm.
- ~65-75% of today's speed.

**The cold tail, with real placements** (VRAM + pinned + warm ~22 GiB):
- **DSV4:** ~93.5% of experts in RAM or VRAM, so ~1.4 cold experts per token from the NVMe. That's up to ~10 ms per token unhidden, which trims the software-only estimate to ~75-85%.
- **MiMo:** ~82-85% in RAM or VRAM, so ~9-13 cold experts per token: 120-170 MB from a ~2 GB/s NVMe. NVMe-bound at ~10-12 T/s.
- MiMo needs ~+22 GiB of RAM for its cold tail. No exact software mechanism removes that.

**Rejected on paper:**
- A host-side router look-ahead: every token it would stream all of the next layer's router weights (90-180 MB) through the DRAM the pool is bound on.
- Lossless entropy coding of MXFP4. MEASURED 10-10 (scratchpad mxfp4_entropy.py, DSV4's own MXFP4 MTP-layer experts, 72 tensor-experts):
  - The E2M1 codes carry 3.876 of their 4 bits (3.853 given the previous code). The format's levels are used almost evenly.
  - The E8M0 scales carry only 0.96 of 8 bits.
  - Bound: 3.91 bits per weight vs 4.25 stored, so at best 8.1% smaller, most of it the scales.
  - Against a ~19% capacity gap, with a decoder in the hot path: dead.
- A GPU-side next-layer prediction is possible at ~zero cost, but hides only ~1 layer (~1 ms) of a ~6 ms cold read. Deferred until runs show the tail.

**If no RAM is added, MiMo's near-lossless option:** re-quantize only the coldest ~15% of experts by the profile (~2.4% of routings) to a compact imatrix format so the whole model fits RAM.
- The hot 85% stay in their released MXFP4 bits.
- Expected KLD vs MXFP4: ~the engine's own noise floor (measured 0.0064-0.0072 vs mainline).
- It needs per-expert formats inside a layer: two NativeFmts per layer, CPU jobs grouped by format, cold experts kept out of VRAM.
- It is not bit-exact MXFP4: the user decides.

## 5b. The prompt's burst (built 19:30, A/B queued)

- **Why:** the adaptive RAM tier moved 32 experts per 32 tokens, so after an 8K prompt it needed hundreds of tokens to reach the prompt's placement. In decode-runs mode (3-token prompts) it found +5%.
- **What:** `prompt()` sets `ram_burst`.
  - The next window's `ram_round` swaps every in-place expert that beats a pinned one by the prompt's routing at once, up to `STRATA_GLM_RAM_BURST` (4096; 0 = off), instead of 32.
  - Landed fills are committed every window during the burst (`ram_commit`).
- **Fills yield to the cold reads:**
  - At most `STRATA_GLM_RAM_FILL_THREADS` (2) at once, and always one worker left free.
  - Read in 1 MiB pieces, each only when no cold read is queued or in flight.
- **Page cache:**
  - A filled expert's page-cache copy is dropped, since it is a duplicate.
  - The warm set skips pinned and filling experts, and re-ranks when a burst is queued or lands.
- **Logs:**
  - `RAM tiers: the prompt's routing moves N experts ... (X GiB of fills queued ...)`
  - `... placement landed T s after it was queued`
- **A/B:** 8K prompt + 384 tokens, the same emulation as 18:25 (static learned: 10.4 T/s) and 18:28 (oracle: 14.3).
  - A = adaptive rounds without the burst, B = the burst.
  - The burst stays the default if B >= 12.5 T/s and B >= 1.15 x A.

**Next, the look-ahead study (queued after the A/B):**
- STRATA_GLM_ROUTE_DUMP=<file> (debug, glm_decode.cpp) appends every decode MoE layer's router input and ids (both decode paths), and every prompt chunk's ids.
- scratchpad pregate_study.py replays them offline against the GGUF's routers:
  - pre-gate d: layer L + d's router on x_L, the engine's sqrt(softplus) + bias top-K;
  - co-occurrence: the prompt's own P(e' at L + d | e at L);
  - reuse: the previous window's experts.
- Each reports recall and precision on the cold routings (the coldest 12% by the oracle profile, or by the learned one), and a "read at the first layer that predicts it" policy by lead.
- The script was validated on a synthetic dump (the d = 0 self-check reproduces the routing exactly).
- Build the read-ahead only if, at lead >= 3 layers, cold recall >= 50% and precision >= 50%.

## 5c. Measured 20:03-20:17: the burst, the oracle, the reserve (8K prompt + 384 plain tokens, w1, no MTP-acceptance confound)

The MXFP4 emulation (DSV4 UD-IQ3_S + 38 GiB balloon, ~28 GiB in place), one run each, nothing else reading the NVMe
(io_watchdog.sh: no 4 KiB probe above 300 ms in any of these runs):

| run | decode | pool ms/token | cold reads (per token) | warm set refills |
|---|---|---|---|---|
| A: adaptive rounds (the default) | 8.74 T/s | 75.2 | 5,725 (14.9) | 4 |
| B: + the prompt's burst (1,321 experts, 11.85 GiB, landed 10.1 s in) | 9.55 (+9%) | 67.9 | 4,816 (12.5) | 2 |
| O: the oracle's placement (this text's prompt + decode routing, hindsight) | 14.28 (+63%) | 37.9 | 2,026 (5.3) | 18 |
| R2: A with the warm set's reserve 12 -> 2 GiB (25 GiB warm instead of 15) | 7.75 (-11%) | 89.8 | 7,110 (18.5) | 1,672 |

- **The burst** captured 15% of the oracle's gain, below its +15% bar, so it is opt-in now (`STRATA_GLM_RAM_BURST=4096`).
- **My earlier reading of the oracle was wrong.** I said "95% of its profile is the prompt, so no hindsight". But the other 5%, the decode's own routing, is exactly the rare tail that makes the cold reads. The prompt predicts the bulk, not the tail.
- **The reserve:** the user's monitor showed a 107 GB peak, which counts only non-reclaimable memory.
  - During R2, page cache held another ~22.5 GiB, leaving ~3 GiB truly free.
  - The kernel then kept evicting the larger warm set (51 GB re-read), and the refills competed with the cold reads: slower.
  - Pinned memory comes from the same pool. So 12 GiB of breathing room (swap full) is the floor, and the slack is already holding experts.
- **The 19:52-19:59 desktop freeze** (MiMo's SHA-256 pass + arm A on the same NVMe, which also holds the full swap file): IO starvation, not memory (PSI io full avg300 22%, memory ~0). Heavy NVMe work is now serialized and run under io_watchdog.sh.

## 5d. The look-ahead study and the cold cache (20:18-20:29)

**Route dump** (today's DSV4, 8K prompt + 384 plain tokens; plain decode at today's placement: 20.99 T/s, the w1 baseline). The pregate_study.py self-check reproduced all 3,720 rows of the engine's routing.

| predictor of the cold routings | cold set = coldest 12% by the oracle (1.1% of routings) | ... by the learned profile (13.3%) |
|---|---|---|
| layer L + 3's router on x_L, K 8 | recall 50%, precision 16% | recall 68-86% at lead >= 3, precision 21-34% |
| policy "read at the first layer predicting it", lead >= 3, K 6 | recall 56%, precision 7% (21 reads per window for 2.7 needed) | |
| the prompt's co-occurrence | ~1% | |
| the previous window's experts | 22% | 54% |
| LRU of 64 / 128 / 256 cold experts | 52% / 61% / 66% | 68% / 81% / 90% |

- Read-ahead fails its bar (precision >= 50%): to catch the tail early it would read 3-7x the bytes.
- The tail recurs instead: what the decode routes rarely, it routes again within a few tokens.

**The cold cache (built 20:22, default on: `STRATA_GLM_COLD_LRU=128`, 0 = off):**
- A cold read's private buffer now keeps its expert after the layer, as an LRU on top of `STRATA_GLM_COLD_BUFS` (16) buffers for the reads in flight.
- A later layer that routes the expert computes from the buffer with the cached experts, in the first pass, with no read.
- Hits are claimed before misses evict. The warm set's budget gives up the cache's bytes (~1.6 GiB at MXFP4 size).

| run (same emulation, 8K + 384 w1) | decode | cold reads | cache hits |
|---|---|---|---|
| A (20:06) | 8.74 T/s | 5,725 | - |
| L: + the cold cache (128) | **10.21 T/s (+17%)** | 3,933 | 2,591 |

- The bar was 1.15 x A = 10.05, so it stays on.
- One watchdog probe reached 389 ms, during the load or the warm-set fill, not the decode.
- This emulation now runs at 49% of today's plain decode (20.99 T/s). The oracle placement reached 68%.

## 5e. Recency at scale (20:45-21:10): the page cache as an LRU, and a 1,024-expert private cache

| run (emulation, 8K + 384 w1) | in place | decode | cold reads | cache hits | notes |
|---|---|---|---|---|---|
| L3: the default (warm set 13.5 GiB + cold cache 128) | 30.3 GiB | 9.26 T/s | 4,323 | 2,926 | |
| K: cold reads through the page cache, warm once, no trim | 28.2 | 12.83 | - | - | INVALID: warm-up took 107 s while the kernel swapped GBs out (swap 2 -> 8 GiB); that RAM inflated it |
| K2: K + a MemFree >= 6 GiB trim | 26.9 | killed | | | watchdog: 16 page-ins took 1.3 s at the prompt; the trim's madvise storms are the suspect |
| K3: cold cache 1,024 in place of the warm set (2.3 GiB left) | 32.3 | 9.57 (+3%) | 3,209 | 8,708 | swap +3.4 GiB during the decode (the cache is anonymous memory) |

- The recency cache does catch what recurs: 73% of K3's cold accesses were hits.
- But what is left, ~8 first-touch reads per token (experts the decode has not used yet), is beyond any cache or predictor. At 2.75 GB/s it bounds the decode.
- K3 failed its bar (10.65 T/s), so the default stays the 128-expert cache.
- **vm.swappiness:** the box ran at 60, so the kernel swapped the desktop out rather than dropping our clean page cache. The user set it to 10 (until reboot).

## 6. The hardware levers (the physics' answer)

- **Two 32 GB DDR4-2666 ECC RDIMMs in the empty channels** (CPU1 slots C1 and F1, matching the four installed):
  - 192 GB, 6 channels: +64 GB and +50% bandwidth.
  - Both MXFP4 releases then fit pinned, the file tier goes unused, and the DRAM-bound MXFP4 pool runs ~1.5x faster than the IQ pool today.
  - Expected: parity or better for both.
  - `sudo dmidecode -t memory` gives the installed part numbers to match.
- **A second Xeon Gold 6148 in the empty socket:** +6 channels, +20 cores. Needs a NUMA-aware pool and arena (each socket computes the experts in its own memory).

## 6b. Ready for the switch (2026-10-10 19:10)

- **Configs and launchers:** strata-dsv4-mxfp4.json / strata-mimo-mxfp4.json + run-deepseek-v4-flash-mxfp4.sh / run-mimo-v2.6-flash-mxfp4.sh. They are the production configs with only the model files changed; same port and client names.
- **Integrity of the new byte paths:** STRATA_GLM_FILE_VERIFY=1 compares every direct cold read and RAM-tier fill with the mapping, and the prompt stager's direct reads too. 5,992 slices, 0 mismatches (DSV4 emulation, 19:05).
- **mx512's default layout (STRATA_MX512_V=1):** the activations repacked to the weights' unpack order. +7% at 1 token, +1-3% at 2-3, bitwise equal. Interleaving 2-4 rows was slower and was removed.

## 7. Next

1. The idle-window 20-thread CPU numbers (MXFP4 vs IQ3_XXS, DRAM ceiling).
2. MiMo MXFP4 end to end on the file tier once the download lands and the GPUs are free:
   - load line, warm set, demand stats;
   - decode at shallow and ~100K depth, prefill at 8K and long;
   - parity against mainline (the oracle and KLD scripts of PORT_AUDIT).
3. DSV4 MXFP4 (downloading after MiMo) the same way; then decide on:
   - the router look-ahead (decode: populate layer + 1's predicted in-place experts);
   - an MXFP4 kernel with software prefetch;
   - token_embd on the CPU (+0.5 GiB of slots).
