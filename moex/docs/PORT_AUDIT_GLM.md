# PORT_AUDIT_GLM.md — strata-glm against Strata's Qwen design (2026-10-07)

The user's rule: the GLM-5.3-Flash port must match Strata's Qwen implementation wherever possible. The two exceptions are a
mechanism Strata does not have, and an approach that is objectively better. Speed is never traded for integrity.

This audit checks every item of `moex/docs/STRATA_QWEN_DESIGN.md` (189 items) against `src/glm/*`, `serve/*` and the CLI. The
first pass was done by three read-only audit agents, one per section group, with file:line evidence. Their full tables are
condensed below. Items fixed after the audit are marked **FIXED**, with the test that shows it.

## 1. Integrity: parity with llama.cpp (the yardstick)

Strata's own test: llama-perplexity `--kl-divergence` against llama.cpp's logits, teacher-forced. Strata's Qwen engine measured
KL 0.022 against llama.cpp; llama.cpp's own CPU and GPU builds differ by 0.058.

**GLM test.** `strata-glm --kld FILE` reads llama.cpp's `--kl-divergence-base` file (`src/glm/glm_main.cpp`, run_kld). It replays
every chunk: the first half through the prompt path, the scored half through decode windows. Each row is compared with
llama.cpp's own function (perplexity.cpp log_softmax, ported as is).
- Reference: `llama.cpp-glm5next-mtp` (the only tree that loads GLM-5.3; mainline does not yet), Q2_K, `-c 512 -b 512`, 8 chunks
  of a mixed corpus (wiki, code, technical, multilingual): `moex/out/glm-q2k-ref-mixed-512x8.kld`.
- A multi-sequence batch (`-b 2048`, 4 sequences per pass) gives NaN perplexity in llama.cpp's GLM PR code. The reference uses
  one sequence per pass.

| comparison | mean KLD | median | 99% | same top-1 | RMS Δp | PPL ratio |
|---|---|---|---|---|---|---|
| strata-glm vs llama.cpp | **0.0146 ± 0.0009** | 0.0022 | 0.198 | **96.42%** | 3.49% | 0.9925 |
| llama.cpp `-ub 64` vs llama.cpp `-ub 512` (noise floor) | 0.0142 ± 0.0010 | 0.0022 | 0.160 | 96.23% | 3.76% | 0.9999 |

**Verdict: at parity.** strata-glm differs from llama.cpp by as much as llama.cpp differs from itself when only its micro-batch
changes. The KLD is also below Strata's Qwen port (0.022).

**Other integrity checks**
- **Op-level oracle.** 4-layer slice, llama.cpp CPU with teacher forcing: Q8_0/f32 ops within 6e-7, router ids identical, CPU
  experts bitwise.
- **Kernel parity tests.** All pass. Grouped vs single-column MMVQ is bitwise for Q4_0 Q5_0 Q8_0 Q2_K..Q6_K IQ4_NL IQ4_XS
  MXFP4 IQ2_XXS IQ2_XS IQ3_XXS IQ3_S IQ2_S (`build/mmvq_grouped_parity`).
- **Speculation exactness.** With a frozen tier, greedy output is token-identical for plain decode and windows 2, 3 and 4
  (160 tokens). Outputs differ between runs only when the tier state differs: the same expert rounds differently on the GPU
  and the CPU, Strata's documented caveat (§3.5).
- **Slot integrity (FIXED today; Strata's verify_slot).** Every expert slice is hashed as it is read from the file. Every VRAM
  slot is hashed on its GPU and compared after the load (3,431 slots in 35 ms), and after every first-use fill, swap and
  post-prompt refill. A mismatch fails the load, or at runtime evicts the slot so the CPU computes that expert.

## 2. Fixed after the audit (2026-10-07)

| item | was | now | evidence |
|---|---|---|---|
| 1.5.2 stale checkpoints | Never deleted. After a conversation switch, a checkpoint could be restored over another conversation's latent cache: silently wrong output. | Strata's rule: drop checkpoints past `resume` or off the prompt's path; clear all on a read from 0. | `serve_protocol_test.py` 4d/4e: after a switch, resume 0 and output equal to a fresh engine |
| 1.5.3 root pile-up | Every root pinned forever (~146 MB each). | Same rule: a root not on the prompt's path is dropped. | 4d |
| 1.5.5 short reads per part | Decided per request, so the 1-20-token tail after the last cut cost a full prompt-path chunk (~12 s). | A final segment ≤ short_read goes through windows. | glm_session.cpp `tail_from` |
| 2.2 reserve | GPU0 1536 MiB (unjustified). | 700 (Strata's, and the live install's). | 1780 → 1886 slots on GPU0 |
| 2.3 read back after fill | Absent. | Hash-verified, at load and after every fill, swap and refill. | "3431 expert slots verified against the file" |
| 3.3 host thread | Unpinned (Strata measured 26.9 vs 36.32 GB/s pool unpinned). | Pinned once to the planned host core. | "the host thread pinned to core 0"; plain 26.7 → 27.0, MTP 34.4 → 35.0 T/s |
| 3.2 / 1.x comments | Claimed cross-layer overlap; wrong `experts_in_ram` docs; `--pin-gguf` silently unused. | Corrected. | — |
| 1.9 `--prefill auto:N` | Uncaught `stoi` exception. | Parsed (`auto[:N]`, `off`, N). | — |
| 1.2 config keys | `parallel`, `vram_elastic`, `split_skip_if_fits`, `expert_profile_save` made strata-glm exit 2 at start. | `families.check_config` refuses them by name before loading. | — |
| Host safety | The PC froze 3 times: the full 93 GiB pinned arena left 17 GB, and MADV_HUGEPAGE compaction stalled the desktop. | Strata's resident RAM mode (complement + lend region + exchanges, 28 GiB headroom), 4 KB pages, O_DIRECT load. | memory PSI 0, ~26 GiB available, load 2.5 GiB/s |
| 2.x layer split | "auto" was n_layer/2: GPU1 (the head, the drafter's region, the display) held 19% of its layers' experts, GPU0 32%. | Strata's `--layer-split auto` (generate.cpp: predict per placement, exhaustive search) with the measured profile mass instead of Strata's fitted rank curve: each K's per-GPU slots (free VRAM − reserve − dense − state − last-stage extras − a 336 MiB allowance measured on both stages), the profile filled hottest first, the most routed mass held wins. `Engine::Impl::choose_split`. | Layers 0-26 / 27-44: predicted slots 1776 + 1368, real 1778 + 1366; profile mass held 46.6% vs 45.9%; plain 27.0 → 27.8, MTP w3 35.0 → 36.0 T/s, hit rate 66.8% → 68.2%, the same output ids |

## 3. Deviations kept, with the measurement or reason

- **MTP window 3 by default** (Strata ships 4). Measured at steady state: w2 31.1, **w3 34.4-35.0**, w4 33.0 T/s; plain 27.0.
- **short_read 600** (Strata 64). GLM's 7.9 MiB experts make one prompt-path chunk cost ~12 s. Measured crossover: windows win
  up to ~600 tokens.
- **Prefill ring = 55% of a layer's experts** (Strata's byte budget). Measured: 64 / 160 / 320 / 480 slots → 624 / 745 / 722 /
  705 T/s.
- **Swap budget 256 MiB per round** (Strata: 96 swaps of its 1.4 MB blobs). 7.9 MiB swap DMAs contend with the doorbell.
- **Adaptive rounds every 4 tokens fed** (Strata: every 4 windows), so speculation's wider windows adapt at the same pace per
  token. Reasoned; measured hit rate only.
- **Step inputs through mapped memory** (window_load). Strata's design keeps per-step values device-resident. Here the H2D
  copies had queued behind swap DMAs on the copy engine: the drafter step took 7 ms instead of 2 ms.
- **Resident headroom 28 GiB** (Strata 4-8). This is a desktop: 17 GiB free froze it.
- **4 KB arena pages** (Strata: MADV_HUGEPAGE). Compaction froze the desktop (Strata's #771 records the same failure). Decode is
  within noise (28.5 vs 29.5 T/s).
- **Slot size maximising experts per byte.** Strata's packs have one blob size; mixed GGUFs (UD-IQ2_XXS) would lose about 23%
  of the tier with largest-blob slots. Larger-blob layers stay on the CPU.
- **The drafter's experts in their own VRAM region.** Their blob may not fit a trunk slot.
- **Decode graphs captured at load** (warm-up). Otherwise the first request would pay a few hundred ms of first-use captures.
- **KDA state snapshotted per window row** for rollback (Strata replays GDN). Costs ~139 MiB per row (~417 MiB at window 3).
  **TODO: port the replay.**
- **Dense projections through llama.cpp MMQ** (Strata: FP16 dequant + cuBLAS). Bit-faithful to llama.cpp's GPU path. Not
  measured against cuBLAS.
- **Exchange swaps are two-phase.** A VRAM-only victim is copied out first, while it is still resident. Correct by
  construction; one window of extra lag.

## 4. Open, ranked by impact (vs Qwen)

1. ~~Layer split by cost model~~: done (§2).
2. **Prompt lookup (SuffixDrafter + DraftPolicy).** +6-11% on code edits in Qwen.
3. **Calibration.** spec-min-p (Qwen on this box: 0.70; GLM: 0.5 uncalibrated) and pool workers.
4. **Coupled draft sampling.** The server samples GLM at temperature 1.0 by default; argmax drafts against sampled targets
   lower acceptance.
5. **KDA rollback by replay.** Frees ~417 MiB of VRAM (~53 slots) and the per-row snapshot writes.
6. **Draft-vocab subset** for the drafter's head (Qwen 106K of 248K ids).
7. **expert_profile_save (#477).** The learned tier survives a restart.
8. **Engine watchdog, and a liveness query in the host spin.** The spin timeout is 30 s, Strata's 20 s.
9. **Parking and session save/restore**, `--batch` slots, KV streaming / int8 KV for 262K contexts, VRAM elastic.
10. **Setup integration** (the config is hand-written), DONE fields 11-13 (RAM/file blobs), and a READY VRAM warning.

## 5. Condensed per-item status (the audit agents' tables)

### §0-1 Process and serving (58 items)
**MATCH**
- The Python front end owns HTTP, the template, tokenization and parsing.
- One persistent engine.
- Popen and logs.
- Child env.
- The readiness handshake.
- The port check.
- The min-free-VRAM guard.
- Restart on the next request.
- Shutdown.
- Framing and the reader thread.
- Sampling keys on GEN.
- Replies (INFO/READY/RESUME/PP/REUSED/T/DONE/ERR).
- EOS in the engine, stop strings in the server.
- STOP semantics.
- Cancel per PP.
- The context limit (n + max_new + 8).
- logprobs/n/best_of unsupported.
- The silence watchdog.
- The tokenizer (byte-level BPE; 0 differences from llama.cpp on ~180K strings plus a full Unicode sweep).
- Template rendering in Strata's sandboxed Jinja.
- Prompt hygiene.
- The reasoning machine.
- Tool-call rules #210/#211/#804/#1058 (the family parser equals Qwen's OutputParser event for event; 25 tests).
- Detokenization.
- Positional vs recurrent state.
- Byte-stable prompts.
- The fifo lock.
- Endpoints.
- Auth.
- Required config keys.
- The `/config` whitelist.
- A/B `--no-*` arms.

**FIXED:** checkpoint invalidation; the root pile-up; short reads per final part; config-key refusals.

**DEVIATION:**
- Engine defaults are product values (Strata's are A/B/off values).
- Family `--eos-ids`/`--turn-tokens` added by the server (justified).
- DONE fields 11-13 are 0 (TODO).
- `ckpt=0` keeps live state (harmless).
- The engine CLI is one GGUF with `-m` (no pack).
- `--threads`/`--no-pin` instead of `--pool-workers`/`--pool-affinity`.

**MISSING:**
- The suffix drafter.
- BGEN/batch.
- SAVE/RESTORE.
- Parking.
- `--kv` modes beyond fp16.
- KV streaming.
- The tail and message-boundary checkpoints.
- `--split-skip-if-fits` and `--pipeline-windows`.
- More than 2 GPUs.

**N/A:** the Windows job object; GENI (vision refused for families).

### §2-3.5 Memory hierarchy and decode (68 items)
**MATCH**
- Blob layout.
- Cache auto.
- Cache allocated last, EAGER modules.
- The zeroed slot arena.
- The residency table.
- No eviction inside the cache.
- Profile seeding in rank order.
- First-use fill as Strata's fallback.
- Slot lending (90% / 128 kept).
- Register before touch.
- Sliced registration fallback.
- Pin caps.
- mlock of the remainder.
- No NUMA.
- File tier through a plain mmap.
- Exchanges.
- Load reads 16 experts per read, a short read refuses.
- hits_ready all-or-nothing.
- The layer split at the router.
- No cross-layer pipelining.
- 5 mapped doorbell buffers.
- Publish as a captured kernel with a fence.
- In-graph spin waits.
- CPU rows read through a kernel.
- One host sync per window.
- The abort word releases the waits.
- Pool workers.
- Claims.
- Waiting.
- The work split.
- Activations quantized once.
- ggml vec_dot kernels.
- The router (GLM's own gating).
- Exact multi-column MMVQ.
- The GPU/CPU rounding caveat.

**FIXED:** the reserve; read-back after fill; host pinning.

**DEVIATION** (see §3):
- The complement always.
- 4 KB pages.
- Uniform slots by coverage.
- The text profile format (the content is equivalent).
- The swap byte budget.
- Swap timing (inline, two-phase).
- The O_DIRECT load.
- No host plan/flagA (the device picks hits).
- The shared expert on the main stream.
- x published every MoE layer.
- No 2 ms liveness query.
- Release fence only.
- Grouped hits per pair, not per expert.
- Counters 11-13.

**MISSING:**
- VRAM allocation hardening (¾ retry).
- Learned profile save.
- File-tier AIO and the router lookahead (matter only when the complement doesn't fit).
- PCIe share of misses (`--pcie-frac`; calibrated to 0.00 on this box for Qwen).

**N/A:** the canonical Q2_0 pack; Windows large pages; the canonical token graph.

### §3.6-9 MTP, prefill, sampling, ops (63 items)
**MATCH**
- spec-min-p gating, per request.
- One graph per chain step, catch-up through the accepted rows.
- Longest exact prefix, Philox, no rejection sampling.
- No tree drafting.
- Batched positions [0, n-1).
- Routing.
- Expert MMQ in groups of 16.
- The id-order walk.
- GDN/KDA prefill.
- QSA/DSA indexer and tensor-core prompt attention.
- The context limit.
- MTP K/V built by the prompt path.
- The sampler chain.
- Greedy ties to the lowest id.
- GPU sampling outside the graph.
- The Philox counter.
- Stats endpoints.
- Health.
- `--native`-style dense kernels.
- No allocations on the token path.
- Per-step values in device/mapped memory.
- The non-blocking stream.
- Sessions sized up front.
- The model file decides the MoE.
- Size VRAM before speed.
- Drift rules.

**DEVIATION** (see §3):
- Window 3.
- The drafter head on the full vocabulary (TODO).
- KDA snapshots (TODO).
- Chunk choice without 2048/1024 fallbacks.
- MMQ dense projections.
- Ring sizing.
- Routed-only streaming replaced by windows below 600.
- Flat K/V (no pages).
- Defaults sampling from the GGUF.
- VRAM reserve on the later GPU 2048 (the display's).
- Peer access enabled at load (eager path only).
- The trimmed stage weights.
- The log summary line.
- `return 1` instead of `_Exit` on CUDA faults.
- The GLM timeout text maps to 400, not 503 (TODO).
- Graphs captured at warm-up.

**MISSING:**
- The SuffixDrafter.
- Stager threads (file-tier only).
- int8/q4 KV.
- KV streaming.
- `--kv-grow`.
- Coupled draft sampling.
- Setup.
- `--calibrate`.
- Other second-GPU modes.
- The engine watchdog.
- The CPU ISA check.

**N/A:** fused int8 Q2_0 kernels; RoPE scaling; arch_defaults for gfx1151; PLE.
