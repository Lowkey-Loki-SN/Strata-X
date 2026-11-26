# PORT_AUDIT_ALL.md — strata-glm (GLM-5.3-Flash, MiMo-V2.6-Flash, DeepSeek-V4-Flash) against Strata's Qwen design (2026-10-08)

The rule: match `moex/docs/STRATA_QWEN_DESIGN.md` item by item, except for a mechanism Strata does not have or an approach
that is objectively better (with the measurement). Never trade correctness for speed. The 189-item tables for the shared
machinery (process, serving, memory hierarchy, decode, MTP, prefill, sampling) are in `PORT_AUDIT_GLM.md` §5; this file adds
what the two later models and the 10-08 work changed.

## 1. Integrity (llama-perplexity `--kl-divergence-base`, mainline llama.cpp c479922ac; `strata-glm --kld`)

| model | test | mean KLD | same top-1 | PPL ratio | mainline's own noise floor |
|---|---|---|---|---|---|
| GLM-5.3 (Q2_K ref) | 512 x 8 | 0.0146 | 96.42% | 0.9925 | 0.0142 (ub 64 vs 512) |
| DeepSeek-V4 UD-IQ3_S | 512 x 8 | 0.0070 | 97.21% | 0.9994 | 0.0066 |
| DeepSeek-V4 | 4096 x 2 | 0.0140 | 96.51% | 0.9945 | 0.0132 (ub 128) |
| MiMo-V2.6 BPW3.0 | 512 x 8 | 0.0065-0.0068 | 96.5-96.8% | 0.997-1.000 | 0.0058 |
| MiMo-V2.6 | 4096 x 2 (new ref) | 0.0064 | 97.19% | 1.0017 | — |

Kernel parity suites (all pass): glm_kernels_parity (incl. the GQA tests added 10-08: decode split-K, FP32 tiles and the
tensor-core prompt kernel against a float64 reference, sinks, SWA rings, depth 65K), dsv4_kernels_parity (incl. non-causal
image spans), iq_avx2_parity (every AVX-2 variant bit-exact, gathered ones included), native_expert_parity (CPU rows vs
ggml vec_dot ~3e-8, width invariance 0 rows), suffix_drafter_test, draft_policy_test, serve unittest (204 + 30).  Added 10-08
evening for MiMo's DFlash drafter: non-causal GQA over a ring (5e-7 vs float64), the drafter's tensor-core Q8_0 projections
q8_mma (1-3e-7 vs float64, every shape incl. the 152576-row head), and the shared-K/V decode attention (1.3e-6 vs float64; a
row's output in a window of 1-8 rows BITWISE its output alone, also across a chunk-size doubling at 131072 keys).

Speculation exactness: greedy and seeded outputs are identical with and without drafting in a fixed placement (DSpark w6 ==
w1; coupled == argmax drafts == the same text). Different configurations can differ at near-ties because the expert
placement (CPU vs GPU, Q8_K vs Q8_1 activations) moves when a config takes VRAM - Strata's documented caveat; mainline shows
the same across offload placements.  DeepSeek-V4's NextN head (10-08): greedy windows of 1, 2, 3 and 4 rows print the same
160 ids on a frozen tier.  MiMo's DFlash drafter (10-08): plain vs drafted in one load (frozen tier, STRATA_IQ_MT_MIN=1) -
greedy with 7 drafts every window, seeded temp 1 with coupled drafts, seeded temp 1 with 7 argmax drafts every window - all
three BYTE-IDENTICAL (1477 / 1215 / 1215 chars).  Without STRATA_IQ_MT_MIN=1 the outputs part at a near-tie (char 497, char 43):
MiMo's gate/up experts are mostly IQ2_S, which the single-token rule below does not cover (its gathered kernel is slower for one
token here, 0.420 vs 0.352 ms), so a lone token's IQ2_S rows round differently (ggml's dot) than in a group - Strata's #152,
already reachable through prompt-lookup windows; mainline rounds by batch composition the same way.

GGUF reader fix (10-08): `block_geometry` had no MXFP4 (type 39), so `tensor_payload_bytes` read 0 and every MXFP4 expert slice
was copied as 0 bytes - the DSpark drafter's and the NextN head's routed experts were ZERO (their E8M0 bytes 0 = denormal scales,
so each pool call also crawled: 15 ms for 8 experts against 1.75 ms).  Both loaders now check each slice against the format.

## 2. Performance (8K-token prompt + 512-token decode at temp 1.0 / top_p 0.95 / top_k 20, 262144 context allocated)

| model (config) | prefill ours / mainline | decode ours / mainline |
|---|---|---|
| GLM-5.3 (`--mtp --coupled-draft --spec-min-p 0.7`) | ~555 / 167.6 = 3.3x | 27.9 / 7.75 = 3.6x |
| MiMo-V2.6 (plain) | 565 / 154.7 = 3.65x | 22.4 / 7.84 = 2.86x |
| MiMo-V2.6 (`--draft-dflash` Xiaomi's DFlash drafter, `--spec 8 --coupled-draft`, the cost-aware window; 10-08 evening) | 513 / 154.7 = 3.3x (plain 540 in the same conditions: the drafter's 1.5 GB = 175 slots, a smaller lend region) | prose 25.5 / 7.84 = 3.25x (plain 22.3 the same session, +14%); code 26.8-28.1 (+16-27%); at 78K context 26.4 vs plain 21.0 (+26%) |
| DeepSeek-V4 (`--draft-mtp` the original V4-Flash NextN head, `--spec 2 --coupled-draft --spec-min-p 0.7`) | 549 / 184.1 = 2.98x (plain 564) | 28.9 / 9.95 = 2.9x first request, 31.2 warm = 3.1x (plain 25.3 / 27.7) |

Mainline's best configuration per model: GLM plain (no GLM MTP in mainline), MiMo plain (its draft-mtp: 4.18 T/s at 12%
acceptance), DeepSeek-V4 plain (draft-dspark: 7.73 T/s at 39%; draft-mtp with the same NextN head as ours: 9.15 T/s at 47%
per draft, 3 drafts a step).  "Warm": the same 512-token request again after the tier has adapted (bench_minp); the DeepSeek-V4
arms ran back to back on one config each (plain C, pinned A, window 3 B, in place D, in VRAM E - below).

## 3. Deviations from the Qwen design, with the reason

| item | Strata | here | why (measured) |
|---|---|---|---|
| prompt chunk plan | the largest chunk <= 8192 | the same, but a segment's remainder <= 1/4 chunk is spread over its chunks (at most 1.25 x) | each chunk streams every non-resident expert once (~7 s a stage here): MiMo 8385 tokens 24.5 -> 16.5 s; one 16384 chunk is slower than 2 x 8192 (27.9 vs 25.5 s: no stage overlap) |
| prompt MoE buffers | rows x nf SwiGLU and rows x n down buffers | SwiGLU over the gate halves, down rows over gu when n <= 2 nf | bit-identical (frozen-tier KLD equal to all digits); ~1.2-1.6 GB less a GPU lent, MiMo 8385 tok 15.3 -> 14.2 s |
| post-prompt refill | per stage, one after the other | every stage's copies queued before any wait | 16384 tokens 25.5 -> 24.9 s |
| MiMo prompt attention | (Qwen: QSA prompt attention on tensor cores, FP16 hi+lo) | gqa_prefill_mma_kernel, the same numerics, plus per-chunk fresh accumulators + FP32 adds | ~10x the FP32 tiles; mma.sync's truncating FP32 accumulate drifted 1e-4 at 8K keys, 8e-4 at 64K when fed back through C - fixed (3.9e-6 at 65K) |
| MiMo decode attention (full layers) | (the FP32 split-K kernel) | gqa_decode_mma_kernel (10-08): one warp per (key split, KV head, token) - its 16 rows are the G = 16 heads sharing the KV head, one m16 tile; the prompt kernel's numerics; 128 key splits (STRATA_GLM_GQA_CHUNKS), STRATA_GLM_GQA_DECODE_MMA=0 = the FP32 kernel | a user session decoded at 20.6 T/s at 5K context but 9.6 at 135K: the FP32 kernel spent ~6 ms a full layer per token at 131K keys (9 layers ~54 ms); tensor cores 0.83-0.86 ms (1 token) and 4.1 ms vs 31.8 (4-token window); float64 parity 0.9-3.8e-6, 4.6e-6 to the FP32 kernel (glm_kernels_parity test_gqa_decode). In the user's session at 156.5K context: 14.5-15.5 T/s (was 8.5-9.4); the rest of the depth cost is the KV read itself (3.6 GB a token at 156K, ~5 ms at the cards' bandwidth; the kernel ~9 ms with 32-key tiles at 2 warps an SM; since 10-08 evening 16-key tiles (21.5 KB a warp, 4 an SM) and 64 splits: 0.57 ms a layer at 131K keys vs 0.82 (1 token), 2.46 vs 3.47 ms (4); STRATA_GLM_GQA_DECODE_KT=32 = the old tile) |
| single-token CPU expert rows | ggml vec_dot below 2 tokens (STRATA_IQ_MT_MIN=1 opt-in) | the gathered AVX-2 kernel from 1 token for IQ3_XXS / IQ3_S / IQ2_XXS where gathers are fast | DSV4 decode 24.5 -> 26.7 T/s (pool 36 -> 43 GB/s); rounding no longer depends on drafting (Strata's #152 goal); KLD unchanged |
| gather detection | Intel with AVX-VNNI only | + non-hybrid Intel AVX-512 cores unless the GDS microcode mitigation is on | Xeon Gold 6148 one-token IQ3_XXS 0.514 -> 0.418 ms |
| AVX-512 IQ kernels | VNNI + VBMI gate (the file needs neither) | built without them; BW-only CPUs opt in (STRATA_IQ512_BW=1) | slower here (license clocks, lookup-bound formats): 20.4-24.5 vs 26.3 T/s |
| coupled draft sampling | opt-in (--coupled-draft) | on in GLM's config | GLM acceptance 67 -> 81 %, decode 22.6 -> 25.3; with spec-min-p 0.7 (calibrated; Qwen on this box: 0.7) 27.9 |
| MiMo decode | (Qwen: MTP on) | plain | the checkpoint's MTP heads are weak: greedy first-draft acceptance 44%, later heads ~0-9% (GLM's NextN: 85-90%); at temp 1.0 55-57% -> MTP 18.6-19.7 T/s vs plain 22.4. Wiring verified: graph = mainline's (op oracle), [e|h] input order (the [h|e] order gives 0%), heads' catch-up input = the trunk's pre-norm hidden as mainline; mainline itself gets 12%. Only the heads a window can use are caught up now (moex) |
| short_read | 64 | 600 | GLM's 7.9 MiB experts: a prompt chunk costs ~12 s whatever its size |
| resident headroom / pages | 4-8 GiB, MADV_HUGEPAGE | 28 GiB, 4 KB pages | the desktop froze three times |
| GLM KDA rollback | GDN replay | per-row snapshots (~139 MiB a row) | TODO: replay (frees ~417 MiB) |
| DSpark drafter (DeepSeek-V4) | — (mainline mechanism) | --draft-dspark, opt-in, EXPERIMENTAL | lossless (w6 == w1); its earlier 16% acceptance and 150 ms steps were measured with ZERO routed experts (the MXFP4 reader bug above) and are void; not re-measured (the NextN head below drafts for this trunk) |
| DeepSeek-V4 MTP | (Qwen: its own MTP head) | the ORIGINAL DeepSeek-V4-Flash's single NextN block (ggml-org `mtp-DeepSeek-V4-Flash-MXFP4.gguf`, llama.cpp's `-md ... --spec-type draft-mtp`) on Vision-Exp's trunk, `--draft-mtp FILE` (src/glm/glm_dsv4_mtp.cpp, deepseek4.cpp graph_mtp) | Vision-Exp and 0731 ship only DSpark; their trunk weights are 7-26% (rel. L2) from the original's, yet the head's first draft is accepted 84-92% here (65-73% with the experts zeroed). Window costs measured with stand-in drafts (`--decode-runs fN`): 2 / 3 / 4 rows = 1.31 / 1.62 / 1.81 x one token, so window 2 (one draft) is best: 30.6-31.2 T/s vs window 3's 27.8-29.6 |
| the head's RMS eps | (one eps per model) | its own file's 1e-6 for every norm of its block, per stage (`Stage::eps_ovr`) | llama.cpp builds the draft model's graph with the draft's f_norm_rms_eps; Vision-Exp's trunk says 1e-20 |
| the head's experts | drafter experts resident in VRAM (Qwen: 512 in Q2_0, ~700 MB) | read IN PLACE from its GGUF mapping by the CPU pool (llama.cpp's mmap); `STRATA_GLM_MTP_VRAM=1` = Strata's layout, `STRATA_GLM_MTP_ARENA=1` = a pinned copy | measured, warm decode / prefill (8K) / first prompt: in place 31.15 / 549 / 16.3 s; in VRAM (3.19 GiB = 360 trunk slots and 340 lend-region experts) 31.95 / 517 / 23.2 s; pinned copy 30.65 / 530 / 22.5 s. In place finishes the bench request (8K + 512) 1.5% sooner and keeps the prefill at ~3x; the VRAM layout wins only decode-heavy work |
| MiMo drafter | (Qwen: its MTP head) | Xiaomi's DFlash drafter (MiMo-V2.6-Flash-*/dflash, mainline's convert --target-model-dir; `--draft-dflash FILE`, src/glm/glm_dflash.cpp = mainline's models/dflash.cpp graph<false> + common/speculative.cpp draft-dflash): five dense Qwen3-style blocks over the trunk's residual at layers 1/12/24/36/48, an 8-position non-causal noise block per step. Its experts-free 1.45 GiB on the last GPU; the trunk's head and embedding rows (byte-identical to the drafter's own but for the mask row: checked at load, its own otherwise); K / V rings of 1088 rows; features of target layers on GPU 0 carried in mapped memory (windows) / a pinned double buffer (each prompt chunk's last 1088 rows - mainline injects every row; only the window's are ever read); drafts by Strata's coupled sampling; one CUDA graph per (kept rows, coupled) step | acceptance at mainline's setting (7 drafts a step, argmax): prose temp 1 26.9% vs mainline 27.0%, code 48.2% vs 50.7%. The MTP heads it replaces were 44% / ~0-9% (greedy). Lossless (above). A step = 5.5-6 ms (8.0 with MMVQ, eager) |
| the drafter's projections | (MMVQ) | q8_mma_w: Q8_0 x up to 8 columns on int8 tensor cores (mma.m16n8k32, the GGUF layout staged by cp.async, split K), STRATA_GLM_DFLASH_MMVQ=1 = MMVQ; MMQ for a prompt chunk's tail | MMVQ (exact or llama.cpp's layout) is compute-bound past 4 columns: a step's projections 7.3-8.5 ms at 8 columns vs 3.2 at one; q8_mma 4.0-4.3 ms (MMQ 5.1). Drafts only: a verify is exact whatever the drafts (its sums differ from MMVQ's by ~1e-7) |
| DFlash's window | spec-min-p gate | cost-aware (`--spec-policy cost`, default; a request's strata_tune spec_policy 0 = the gate): per window n* = argmax E(n) - lam (verify(n + 1) + step), E from the drafts' measured acceptance by (depth, probability) cell (prior: the draft's probability, 1 window in 8 explores one draft more), verify costs by row count and lam = the long-run tokens per ms, all learned online | an extra verify row costs MiMo ~0.58 of a token here (24.9 ms vs 42.7: its experts mostly on the CPU), so a draft must be likely to pay for its row. Per-window E/cost (the first version) cut confident windows short: 23.1 vs the best fixed gate's 25.5 T/s; the renewal-reward form: 25.5 = the gate's at short context without a calibrated threshold, and it re-balances at depth |
| MiMo decode attention, windows | gqa_decode_mma_kernel (one warp per row) | gqa_decode_shared (10-08 evening, default; STRATA_GLM_GQA_SHARED=0 = per row): one block per (absolute key chunk, KV head), a warp per window row, each K / V tile read once per window; chunks of a power of two set by the row's own position (128 of them: 2048 keys at 131K), so a row's result is bitwise the same in any window | at 131K keys a full layer: 1 row 0.61 vs 0.58 ms, 2 rows 0.59 vs 1.05, 4 rows 1.14 vs 2.20, 8 rows 2.03 vs 4.37 - a verify row at depth no longer re-reads the cache (~3.6 GB a row at 156K) |
| vision | Strata's encoder process + GENI | the same design; GLM / MiMo / DeepSeek-V4 projectors | greedy image answers match mainline mtmd up to near-tie word choices; DeepSeek-V4's image span read non-causally in one batch (moex/docs/VISION.md) |
| chat_template_kwargs | enable_thinking: false honoured | true honoured too | DeepSeek-V4's template defaults to thinking off |

## 4. Open

1. ~~Prompt lookup~~: DONE 10-08 (Strata's SuffixDrafter + DraftPolicy in glm_spec.cpp, `--suffix-draft 3` on by default as in
   Strata; windows up to the MTP window + 2 / 6 alone; GLM-5.3 within its KDA snapshot rows). Lossless (seeded, frozen tier:
   identical text). MiMo code edit (rename + reprint a file) 17.8-18.9 -> 22.5-24.8 T/s (+27%, every lookup draft kept), prose
   20.75 -> 20.60 (noise: the policy took lookup in ~4% of windows). GLM: its MTP already copies code (398/399 drafts), lookup
   fires rarely - harmless.
2. GLM KDA rollback by replay.
3. Draft-vocab head subset (~2-4%).
4. The frozen-tier run-to-run difference seen once in four KLD runs: suspect cuBLAS SGEMM (router, T > 8) choosing algorithms
   under concurrent streams; check with CUBLAS_WORKSPACE_CONFIG=:4096:8.
5. DSpark: re-measure with its experts loaded (the MXFP4 fix) if it is ever wanted; the NextN head is the default drafter.
6. Product items from PORT_AUDIT_GLM.md §4 (watchdog, batch slots, session save/restore, setup integration, DONE fields 11-13).
7. DeepSeek-V4's head as a captured graph with its experts in VRAM (`STRATA_GLM_MTP_VRAM=1` is eager): ~1 ms a step more for
   decode-heavy work.
8. The head after a resumed prefix reads the ring the checkpoint restored (state_regions includes L[n_layer]); after an image
   span it reads the encoder's rows causally (llama.cpp skips its drafter on embedding batches).
9. MiMo + DFlash: (a) IQ2_S rows for one token in the multi-token kernel's rounding at no speed cost (then drafting is lossless
   by default, #152); (b) q8_mma's head at 7-8 columns runs ~500 GB/s (1.3 ms vs 0.92 at one column); (c) skip the draft step
   in stretches where the window policy keeps choosing no drafts (~30% of prose windows pay ~6 ms for nothing); (d) the decode
   graphs could also carry the drafter's prompt-tail inject (eager, ~25 ms a chunk).

## 5. Upstream Strata 0.1.43.1 merged (2026-10-13)

Merge 01f38213 (a0116b27). strata-glm's shared code was touched only on opt-in paths or CPUs without AVX2; the fixes that
reach the families are in the commits after it (NaN guard, pageable fallback of the prompt path's host buffers, cuBLAS
status, #537 and #2026 for the family parsers, the learned profile in Strata's #477 order).  Measured on this box
(2x RTX 3080, Xeon Gold 6148, the 10-13 GGUFs), with moex/experiments/glm_prompt_ab.py (follow-up reads at depth, arms
interleaved, medians of 3-4):

| short-read ring (9c3a8204) | 300 tokens | 1000 | 2000 | 4096 |
|---|---|---|---|---|
| GLM-5.3 UD-IQ3_XXS, the 160-slot ring / this rule | 5.48 / 3.55 s | 10.15 / 7.63 s | 14.33 / 11.16 s | 17.93 / 16.89 s (64) |
| MiMo-V2.6 SX-124G, the 144-slot ring / this rule | 3.45 / 2.95 s | 6.36 / 5.81 s | 9.89 / 8.66 s | 15.39 / 14.48 s |
| DeepSeek-V4 Q3_K (+ NextN head), 144 / this rule | 3.54 / 3.14 s | - | 9.20 / 7.95 s | - |

Decode unchanged in every arm.  Tried and not kept: upstream's --aux-cpus (the prompt path's helper threads off the host's
CPU: GLM 300-token follow-ups 5.12 vs 5.48 s within the noise, MiMo 3.03 vs 2.95 s); upstream's route tail skip (opt-in
there; it changes the output - dropped by choice, reliability first); MADV_WILLNEED on the lent experts' GGUF pages (below);
a near-tie rule for --layer-split auto (below).

Two findings, explained, not changed:
- **The first long read after a session is ~4-6 s slower** (GLM, an 8K fresh read after 300-token follow-ups: 22.3-26.7 s,
  the next ones 18.4 s).  Its process page-faulted 3.89 GiB from the disk, the next reads 0.01 / 0.00 GiB: the experts of
  the slots an 8K read lends (~460 a GPU) are streamed and refilled from the GGUF where they have no RAM copy, and the RAM
  copy holds only ~180-270 of them at GLM's default headroom (28 GiB); the adaptive swaps keep putting experts whose pages
  were never cached into those slots (found later, below - not, as first read, pages falling out of the page cache).
  MADV_WILLNEED when a slot is lent: 22.7 s, 3.69 GiB and 300-token follow-ups 5% slower - not kept (it fetched only the
  first 128 KiB of each part, one call's readahead cap; done right below).  One lever is RAM: strata-glm.json now passes --resident-headroom-gib 20 (the value the MiMo and DeepSeek-V4 configs use): the lend
  region in RAM 1145 experts instead of 180-270, the first 8K read after the session 17.6 s (0.78 GiB from the disk), the
  next ones 16.7-17.0 s (18.4 s at 28 GiB), MemAvailable 16.6-16.7 GiB through the run (the freezes were at Strata's
  4-8 GiB headroom).
- **The learned profile's restart (219744c8)**: GLM restarted from a profile the new save wrote (2561 experts in VRAM first)
  against the old one (prompt-heavy counts, no residency): the first request's VRAM hits 51.4% vs 38.9%, its decode 20.3 vs
  17.2 tok/s; the first three requests 48-51% vs 39-47% hits, the same from the fourth on (the adaptive tier converges).
- **Two identical runs differ** (GLM, --kld against this build's own --kld-save base, 4 x 512 tokens, windows of 8,
  --no-adaptive --share 0 --cpu-read 0): KLD 0.0087, top-1 95.6% - --layer-split auto chose layers 0-26 in one start and
  0-27 in the other (38.4% / 38.3% of the profile's routed mass: a near-tie decided by GPU 1's free VRAM), so other experts
  sat in VRAM and ran on the CPU path (Q8_K activations) instead of the GPU's (Q8_1).  The same split pinned
  (--layer-split 27, one slot of 2,846 different): KLD 0.00005, top-1 99.7%.  With the default flags (CPU share on) the
  same build in one process scored 0.0145 against itself.  For repeatable runs pin "layer_split" (and keep --no-adaptive,
  --share 0 --cpu-read 0); in normal use the adaptive tier moves experts anyway.  A tie rule (within 0.2% of the best mass,
  the split closest to the halves) held the split steady under VRAM drift but picked layers 0-24, where GLM decoded at
  19.9 vs 21.4 tok/s (layers 0-26, three requests each) - not kept; strata-glm.json pins "layer_split": "27" (layers 0-26
  on GPU 0, the split auto chose most often, measured above) so every start places the experts the same way.
- **MiMo and DeepSeek-V4 at the 20 GiB headroom their configs use** (four 300-token follow-ups, then 8K fresh reads): MiMo's
  auto split moved too (layers 0-26 in one start, 0-27 in two others, 44.8% / 44.6% of the routed mass); split 27 against
  28: follow-ups 2.7-3.1 / 2.8-3.0 s, decode median 26.2 / 24.2 tok/s, fresh reads the same - strata-mimo.json pins 27.
  DeepSeek-V4 chose layers 0-23 in all three starts - strata-dsv4.json pins 24, the same split.  Their first 8K read after
  the follow-ups still page-faults: MiMo 20.1 s (3.4 GiB from the disk) vs 16.1 s after, DeepSeek-V4 20.2 s (2.4 GiB) vs
  16.7 s - bigger models, less of the lend region in RAM.  A lower headroom would take it below the configs' own 20 GiB
  (MemAvailable 17.5 / 16.9 GiB through these runs), the margin that keeps the desktop responsive - fixed without it, below.
- **The first long read after a session, found and fixed** (the lent experts read in while the read runs): the cause is the
  adaptive tier.  A read lends each stage's top slots; their experts are streamed from their sources on every chunk and
  copied back afterwards, from the GGUF where they have no RAM copy.  The swaps put experts with no RAM copy into those slots
  (the exchange hands the incoming expert's blob to the victim), and the load never cached their pages (O_DIRECT).  MiMo with
  --no-adaptive: the 8K fresh read after the follow-ups 16.1 s with 0.02 GiB from the disk (19.7-20.1 s, 3.35 GiB with the
  swaps) - but decode 16-18 tok/s instead of 22-28, so the swaps stay.  Now, when the stages have lent, a thread reads those
  experts into the page cache in layer order while the read computes (POSIX_FADV_WILLNEED in 128 KiB pieces and a pread a
  part, ~2 GB/s; the read takes 16 s).  2x RTX 3080, the configs as committed, a 16K prefix, four 300-token follow-ups, two
  8K fresh reads; on / off (STRATA_GLM_WARM_LEND=0) on the same build, two runs a side:

  | model | first read after load (16K) | first 8K read after the follow-ups | the next 8K read | follow-ups |
  |---|---|---|---|---|
  | MiMo-V2.6 SX-124G | 26.2, 26.3 / 30.1, 29.6 s | 15.83, 15.96 / 19.49, 19.69 s | 15.79, 15.88 / 15.82, 15.85 s | 2.88, 2.88 / 2.86, 2.89 s |
  | DeepSeek-V4 Q3_K | 27.3, 27.2 / 32.2, 30.8 s | 16.40, 16.37 / 21.48, 20.21 s | 16.22, 16.23 / 16.41, 16.47 s | 3.01, 3.01 / 2.98, 2.99 s |
  | GLM-5.3 UD-IQ3_XXS | 27.7, 28.0 / 28.6, 28.8 s | 16.77, 16.88 / 18.56, 18.26 s | 16.72, 16.81 / 16.75, 16.78 s | 3.45, 3.51 / 3.47, 3.51 s |

  Decode on the same tokens: MiMo with its placement fixed (--no-adaptive --share 0 --cpu-read 0) generated identical
  tokens with the warm on and off, 4 x 128: 19.3 / 18.1 / 19.0 / 19.7 tok/s on, 19.3 / 18.1 / 19.5 / 19.4 off.  In normal
  runs any two runs' tokens part after 0-128 of 128 (read modes learned from timings, the swaps), so their decode tok/s
  compare different text.  No job runs during the 300-token follow-ups (their lent slots hold the lend region's experts);
  the jobs: 0.5-3.4 GiB in 0.3-2.0 s.  Another program's 4 KiB O_DIRECT reads beside the jobs: at most 1.5 ms (0.3 ms
  median; 6-13 ms during a model's own load) - the disk queue holds one part (~3 MiB).  Not kept on the way: one job after every request (MiMo: ~1.3 GiB read a request, mostly for nothing); touching
  the pages and MADV_DONTNEED-ing them (beside the stream's copies of the same pages: 0.02-0.3 GB/s, the thread on its CPU
  the whole time).
- **DFlash on SX-124G** (strata-mimo-dflash.json, untracked: SX-124G, 20 GiB headroom, split auto = layers 0-28 with the
  drafter on GPU 1): temp 1.0 / top_p 0.95 / top_k 64, an 8K code prefix, six 300-token prose follow-ups of 256 tokens, the
  same seeds, three sessions a side - decode median 25.3 / 25.0 / 23.1 vs plain 27.1 / 25.5 / 25.6 tok/s, follow-up reads
  2.76-2.84 vs 2.68-2.81 s; 587 of 801 drafts accepted, 1.62 tokens a window (the first session).  On BPW3.0 (10-08,
  section 2) it was +14% on prose; SX-124G decodes faster plain, and the drafter's step and verify rows now cost more than
  they return.  Its fresh 8K read after a session is slow: 28.0 / 21.2 / 24.3 s vs plain 15.9 / 15.8 / 16.1 - and 22.5 s with
  the warm off, so it predates the warm (which takes DFlash's first read after load from 22.0 to 16.9-17.3 s).  Not the read
  mode (one streamed chunk in both), not the warm job (done in 2.1 s), not experts dropping out of RAM (STRATA_GLM_SHARE_LOG=1:
  none in neither VRAM nor RAM and the 96 exchange blobs free at every read, plain and DFlash); not followed further -
  strata-mimo.json stays plain.
- The configs: strata-mimo.json / strata-dsv4.json pointed at GGUFs no longer on disk (BPW3.0, UD-IQ3_S) - now the files on
  disk with --resident-headroom-gib 20 (dee0294b); strata-mimo-mtp.json removed (no MiMo GGUF on disk has MTP blocks).
  tools/test_setup_engine_hash.py: its refusal tests asked GitHub for real and failed where the release has no such asset
  (52b142a7, the same on upstream's tree).
