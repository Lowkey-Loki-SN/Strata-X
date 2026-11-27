# ERNIE-5.5-Flash on Strata-X: the plan

The request (2026-11-24): support baidu/ERNIE-5.5-Flash-Preview-FP8 in the Strata-X engine, planned thoroughly and
implemented robustly, with vLLM's and SGLang's implementations as the references. The checkpoint downloads to
`~/AI/Models/HF/ERNIE-5.5-Flash-Preview-FP8` (pinned revision be26a1d0, 128 shards, every file SHA-256 checked by its
download script).

## 0. What the model is

From the release's config.json and the vLLM-ERNIE README (baidu/vLLM-ERNIE ba322a5):

- A sparse MoE of 280B parameters, 14B active a token, pre-trained on 33.5T tokens at up to 128K; inference to 1,048,576
  (YaRN, factor 16 over 65,536). 43 layers at hidden 4096, vocabulary 201,216. `experimental_attention_variant`
  `dsv4_hybrid`: DeepSeek-V4's architecture with Baidu's changes.
- **Attention, two kinds.** 37 layers of Heavily Compressed Attention (compress ratio 128: one cached row per 128
  tokens, plus a 128-token sliding window), MQA with 64 query heads of 512 (448 + 64 RoPE) over one shared KV head, a
  grouped low-rank output (8 groups of 1,024), sinks. 6 layers (7, 14, 21, 28, 35, 42; ratio -2) of MLA with DSA sparse
  attention (64 heads, qk 192 + 64, v 256, kv LoRA 512; the indexer's 64 heads of 128 keep the best 2,048 tokens).
  Every layer: a low-rank sigmoid output gate from the query LoRA, a low-rank post-mix across heads (VHA), QK norm.
- **The MoE.** Layer 0 dense (14,336); layers 1-42 a "multi-router latent MoE": 512 routed experts (top 10) and one
  shared expert, the routed ones in a 2,048-wide latent (projections in and out, a latent norm), SiTU-GLU activations,
  two router matrices, sigmoid scores with a correction bias, a learned scaling factor per expert.
- **mHC**: manifold-constrained hyper-connections, 4 streams, Sinkhorn 20 iterations.
- **MultiMax head**: the head with a learned piece-wise temperature (4 ranges).
- **Drafting.** The chat models bundle DSpark: 3 layers on the target's last 3 layers' hidden states, a block of 7
  drafts, a Markov head (rank 256), a confidence head. (ECHO, the base models' built-in draft, is the final layer.)
- **Weights**: FP8 e4m3 with 128 x 128 blocks of UE8M0 scales (stored as F32 `weight_scale_inv`), dynamic activation
  quantization in the references; BF16 for the embedding, the head, the routers, the compressors and the indexer's wk /
  weights_proj; F32 for the hyper-connections and small vectors.
- Baidu also publishes the same model in FP4 (~155 GB of weights) and BF16 (~560 GB); this plan is for the FP8 release.

## 1. The checkpoint in bytes

From every shard's safetensors header (`moex/tools/ernie55/inventory.py`; the records in
`~/AI/Strata-X-data/ernie55-meta/inventory.json`): 140,457 tensors, 301.8 GB.

| part | tensors | size | per decode token |
|---|---|---|---|
| routed experts, 42 layers x 512, three FP8 2048 x 2048 matrices each (w1, w2, w3) + F32 16 x 16 block scales | 21,504 experts | 270.6 GB (12.58 MB each) | 420 picks = 5.29 GB |
| DSpark's 3 stages' routed experts (512 each) | 1,536 experts | 19.3 GB | per draft step |
| the trunk's dense weights (attention, shared experts, latent projections, routers, mHC, the dense layer 0) | | ~8.7 GB | all of it |
| embedding (BF16 201,216 x 4096) / head (BF16) | 2 | 1.65 + 1.65 GB | one row / all |
| DSpark's dense weights (3 attention + shared stages, feature projection, Markov head, confidence head) | | ~1.0 GB | per draft step |

- The three weights of an expert are adjacent in one shard (w1, w2, w3: one 12 MiB run) for 21,419 of the 21,504; the
  other 85 straddle two shards. Their block scales sit apart, at the shards' starts (3 KiB an expert, 66 MB in all).
  So an expert is one extent (two for 85) of a file that is already on the drive: it can be read in place, with
  O_DIRECT, at the alignment around it - no converted copy (the drive has ~68 GB free once the download completes).
- The MLA + DSA layers: q_a (1024 x 4096), q_b (16,384 x 1024), kv_a_with_mqa (576 x 4096), kv_b (28,672 x 512), the
  indexer (wq_b 8,192 x 1024 FP8, wk 128 x 4096 BF16, weights_proj 64 x 4096 BF16, k_norm with weight and bias), a
  per-head softmax_offset (64, BF16).
- The HCA layers: linear_q_down (1024 x 4096), q_layernorm (1024), linear_q_up (32,768 x 1024), linear_kv (512 x 4096),
  kv_layernorm (512), the compressor (linear_wkv and linear_wgate 512 x 4096 BF16, ape 128 x 512 F32, norm 512),
  attn_sink (64, F32), linear_o_group_proj (8,192 x 4096), o_proj (4096 x 8,192).
- Every layer: gate_proj (8,192 x 1024: the output gate), vha_postmix_U / V (64 x 16, BF16), two hyper-connections
  (mapping_proj 16,384 x 24 F32, bias 24, alpha_pre / post / res), input / post-attention norms.
- Every MoE layer: gate.weight and gate.weight_1 (512 x 4096 BF16: the two routers), e_score_correction_bias (512),
  gate.routed_scaling_factor_param (512: a learned factor per expert), fc1_latent_proj (2048 x 4096) and
  fc2_latent_proj (4096 x 2048), latent_norm (2048), the shared expert (w1 / w3 2048 x 4096, w2 4096 x 2048).
- The head: lm_head with multimax_ranges and multimax_ts (4 each, BF16); mhc_contract (hc_head_fn 16,384 x 4, base,
  scale). DSpark: mtp.0-2 (each an HCA-type attention and a 512-expert MoE with its own mHC), feature_projection
  (4096 x 12,288 FP8: the 3 target layers), feature_norm, norm, markov_head (w1 256 x 201,216, w2 201,216 x 256),
  confidence_head (1 x 4,352).

## 2. The references

- **vLLM-ERNIE** (baidu/vLLM-ERNIE `ba322a5`, a vLLM plugin pinned to vllm 0.32.0; read against mainline vLLM
  `5df63a7`): `src/vllm_ernie/models/ernie5_5/` (the model, its mHC kernels, VHA, the multi-router MoE, the MLA + DSA
  attention, DSpark) and its tests (`tests/models/ernie5_5/`: torch references for mHC, the routing, VHA, the indexer).
  Baidu's own serving code: where the references differ, its choice is the one taken.
- **SGLang** (`f4284f2`, `python/sglang/srt/models/ernie55.py` and the DeepSeek-V4 layers it reuses): an independent port,
  read as the cross-check.
- Both are written up with file:line for every claim in `~/AI/Strata-X-data/ernie55-notes/vllm_spec.md` and
  `sglang_spec.md`; the map of strata-glm's DeepSeek-V4 / V4.1 / Hy4 machinery against them in `strata_dsv4_map.md`.
- llama.cpp has no ERNIE-5.5 code (local master 80325823d, 2026-11-21), so the oracle is our own reference model (§7),
  not a llama.cpp dump.

## 3. The forward pass

As the reference model computes it (moex/tools/ernie55/ref/model.py, every function citing the vLLM-ERNIE lines it
follows); fp32 throughout, the weights the release's FP8 values times their power-of-two block scales.

- **Embedding**: the BF16 row, broadcast to the 4 mHC streams.
- **mHC, each half of a layer** (DeepSeek-V4's, but for one factor): mixes = (fn . x) / (sqrt(mean(x^2)) + eps) over
  the flat 4 x 4096 streams (V4: rsqrt(mean + eps)); pre = sigmoid(mixes[0:4] a0 + b) + eps, post = 2 sigmoid(...),
  comb = softmax over a row, then 20 Sinkhorn steps; the input sum_j pre_j R_j, RMS-normed with the layer's norm; after
  the sublayer R'[k] = post[k] y + sum_j comb[j][k] R[j].
- **HCA attention (37 layers)**: DeepSeek-V4's ratio-128 layer - qr = RMS(q_a x); q = RMS_heads(q_b qr) with RoPE on
  each head's last 64 dims (GPT-J pairs, YaRN at 1.6e5, factor 16 over 65,536, magnitude 1); the K = V row
  RoPE(RMS(attn_kv x)); the compressor's row per completed block of 128 (softmax over the block's positions of
  wgate x + ape, the weighted sum of wkv x, RMS, RoPE at the block's first position); the attention with the head's sink
  over the last 128 positions and every completed row, scale 512^-0.5; then ERNIE's VHA (o[h] = sum_g (I + V U^T)[h, g]
  o[g], U and V 64 x 16), the de-RoPE, the grouped output z = concat_g wo_a[g] (heads 8g .. 8g + 7), and the output
  gate: out = wo_b (z sigmoid(gate qr)).
- **DSA attention (6 layers: 7, 14, 21, 28, 35, 42)**: MLA in the absorbed form - qc = RMS(q_a x), q = q_b qc
  (64 x [192 nope | 64 rope]); the cache row [RMS(kv_c) | RoPE(k_pe)] (plain RoPE at 8e6, GPT-J pairs); the lightning
  indexer: k = Hadamard(NeoX-RoPE(LayerNorm(wk x)) on dims 0-63), q = Hadamard(NeoX-RoPE(wq_b qc)) per head, both rounded
  to E4M3 with a power-of-two scale per 128; score[j] = sum_h w_h relu(q_h . k_j) with w = weights_proj x / sqrt(128 x 64)
  over the positions j < p + 1 - 128 (the window is not a candidate); the best 2,048 (all when fewer); the attention with
  the head's sink over those and the window [p - 127, p], scale 256^-0.5; the latent output through v_b; VHA; out =
  wo (o sigmoid(gate qc)).
- **FFN**: layer 0 dense SiTU-GLU (14,336): 4 tanh(g / 4) sigmoid(g) x 25 tanh(u / 25). Layers 1-42: score[e] =
  sigmoid(W0 x)[e] + sigmoid(W1 x)[e]; the top 10 of score + bias; weights = the scores at the ids over their sum, times
  the learned per-expert factor (the config's scalar 3.0 is applied nowhere); the routed experts on lat = fc1 x (2,048),
  each SiTU-GLU 2,048 -> 2,048 -> 2,048; the weighted sum RMS-normed (latent_norm), then fc2; plus the shared expert
  (SiTU-GLU, 2,048 wide) on x.
- **Head**: the streams' learned reduce (V4's hc_head: rsqrt(mean + eps), vLLM-ERNIE's), the final norm, the BF16 head
  into F32 logits, MultiMax y = x + t0 relu(r0 - x) + t1 relu(x - r1) + t2 relu(r2 - x)^2 + t3 relu(x - r3)^2.

Where the engine's arithmetic differs from vLLM-ERNIE's serving (which rounds to BF16 between ops and quantizes every FP8
linear's input to E4M3 per 128): strata-glm keeps F32 activations into the exact FP8 dense weights (kDtFP8), F16 caches
(the window, the compressed rows, the MLA rows, the index keys - the index keys' values are E4M3 times a power of two, so
F16 holds them exactly), and quantizes activations only into the quantized routed experts (Q8_1 / Q8_K, as llama.cpp).
The points the references leave open, and the choice: the head reduce's RMS factor (vLLM-ERNIE's rsqrt; SGLang's
1 / (sqrt + eps) differs by ~1e-6), the indexer's top-k ties (to the lower position), the DSpark stages' RoPE (vLLM's
plain 8e6; SGLang has YaRN) - DSpark comes after the trunk.

## 4. What strata-glm has, what is new

From a read of the DeepSeek-V4 / V4.1 path against ERNIE's config (`~/AI/Strata-X-data/ernie55-notes/strata_dsv4_map.md`,
file:line for every claim). ERNIE's trunk is DeepSeek-V4-Flash's shape (not V4.1's): 43 layers, hidden 4096, q LoRA
1024, 64 x 512 MQA heads with 64 RoPE dims, the 128 window, the 8 x 1024 grouped output, 4 mHC streams in V4's order
with V4's head reduce, 64 x 128 indexer heads with a Hadamard, the same YaRN on the compressed layers.

- **Reused as they are**: V4's ratio-128 layer (q LoRA and the weightless per-head q RMS, the K = V ring, the HCA
  compressor and its ring, the sparse attention with sinks over window + every compressed row, the de-RoPE, the
  grouped output); YaRN at 1.6e5 on the HCA rotations, plain RoPE at 8e6 elsewhere; mHC (V4's branch) and its head
  reduce; the MoE plumbing (VRAM slots, the adaptive tier, the pinned RAM copy, the CPU pool, the prompt path's MMQ,
  graphs, checkpoints); the lightning indexer's scores and top-k (any k); Hy4's MLA kernels for the 6 DSA layers - the
  same 512 + 64 rows, 64 heads, 192 + 64 -> 256 heads, sinks, per-row cell counts; the sigmoid-gate multiply, a
  LayerNorm with bias, the 128-point Hadamard, NeoX RoPE; the exact FP8 dense format (kDtFP8 holds 128 x 128
  power-of-two blocks bit for bit - every scale checked is a power of two); DSpark's machinery.
- **Generalized**: the arch's keys and layer kinds (128 = HCA, <= 0 = MLA + DSA); a second set of attention dims for
  the DSA layers; idx_topk 2048 and the indexer seeing only positions before the window; the index keys (LayerNorm of
  wk x, NeoX RoPE on dims 0-63 at 8e6, Hadamard, FP8 rounding per 128) and query; the attention gate from the q LoRA
  output (on the 8,192 grouped-output values in HCA layers, the 16,384 attention values in DSA layers); the latent MoE
  (Hy4's branch, plus an RMS norm of 2,048 between the combine and fc2, no post-norm); layer 0's dense FFN; BF16
  projections on the V4 path; the mHC flat-stream factor 1 / (sqrt(mean) + eps); DSpark with V4's head reduce, targets
  written as [41, 42, 43] (the outputs of 40-42), its Markov matrix transposed.
- **New**: VHA (o[h] = sum_g (I + V U^T)[h, g] o[g], U and V 64 x 16); SiTU-GLU (4 tanh(g/4) sigmoid(g) x 25 tanh(u/25)),
  at every SwiGLU site (GPU, grouped VRAM experts, the prompt path, the CPU pool's rows and AVX-512 kernels); the
  two-router score sigmoid(W0 x) + sigmoid(W1 x); a learned weight per expert after normalization (the scalar 3.0 is
  applied nowhere); MultiMax on the logits (y = x + t0 relu(r0 - x) + t1 relu(x - r1) + t2 relu(r2 - x)^2 +
  t3 relu(x - r3)^2), also on the drafter's.
- Open in the references themselves (to settle from their code paths and by measurement): the DSpark stages' RoPE
  (plain 8e6 in vLLM, YaRN 1.6e5 in SGLang), MultiMax before or after the Markov term, the head reduce's RMS factor,
  whether the draft block attends causally.

## 5. Where the bytes go on this box

The target (2026-11-24): a GGUF that runs wholly from VRAM and RAM - nothing read from the drive once it is loaded -
at the best quality that fits. Two RTX 3080 20 GB (sm_86), 125 GiB of RAM, a Xeon Gold 6148 (AVX-512 F/BW/DQ/VL, no
VNNI, no FP16/BF16 instructions).

- **VRAM (42.9 GB)**: the dense weights as the release's exact FP8 (6.71 GiB in kDtFP8) plus their BF16 / F32 tensors
  (0.77 GiB) and the BF16 head (1.54 GiB) - ~9.0 GiB; the reserves (700 MiB on GPU 0, 2 GiB on GPU 1, the display's);
  the buffers (~2 GiB); the caches, mapped as the context grows (Hy4's KV on demand: ~1 GB at 128K, ~8.3 GB at 1M);
  the rest holds experts: ~25 GB while the context is short, ~17 GB at 1M.
- **RAM (134 GB)**: the desktop and the OS (~10-14 GB), the engine's own buffers and the BF16 embedding (~5 GB), a
  little page cache; ~105-110 GB for experts.
- **So the experts may take ~120-125 GB** (RAM must hold all but the VRAM floor at 1M): 3.55-3.7 bits a weight over
  the 270.6 G expert values.

| experts as | bits a weight | size | fully resident |
|---|---|---|---|
| Q4_K everywhere | 4.5 | 152 GB | no (~30 GB over) |
| IQ4_XS everywhere | 4.25 | 144 GB | no (~20 GB over) |
| IQ3_S gate / up + IQ4_XS down | 3.71 | 125 GB | at the edge |
| IQ3_XXS gate / up + IQ4_XS down | 3.46 | 117 GB | yes, with room for DSpark's experts at ~2 bits |

The mix is chosen by measurement (§8 P4): an importance matrix from calibration text through the reference model, and
the KL divergence of each candidate against the FP8 release on a corpus; the bits go where the measurements say they
matter (the down matrices, the layers or experts the model is most sensitive to), within the budget.

No ERNIE-5.5 GGUF exists to download (2026-11-24): llama.cpp has no support (local master 80325823d, 2026-11-21), and
the only "Preview" GGUF repository (peasantsmith/...-Maya-GGUF) holds a README saying the files are work in progress.

## 6. The GGUF

Our own architecture string and keys (the DSV41_PORT.md pattern), written by our converter from the FP8 release:

- The dense FP8 tensors as BF16: an e4m3 value times its 128 x 128 block's power-of-two scale is a BF16 value exactly
  (the converter checks every value and stops on one that is not); strata-glm's loader turns them back into the release's
  FP8 codes and scales (kDtFP8), checking every value again. BF16 and F32 tensors as they are. (The plan was MXFP8,
  ggml type 67; BF16 costs 6.3 GiB more on the drive, none in memory, and any GGUF reader can read it.)
- The routed experts stacked per layer ([2048, 2048, 512] gate / up / down, as llama.cpp's `_exps`), each tensor in the
  type the plan gives it, quantized by ggml_quantize_chunk (llama-quantize's call) from the exactly dequantized FP8
  values, with the importance matrix's per-column weights.
- Re-laid at conversion: kv_b split into k_b / v_b (Hy4's MLA layout); mapping_proj transposed; the three alphas as V4's
  scale[0..2]; markov_w1 transposed; the two routers stacked; softmax_offset and the MultiMax parameters to F32.
- Split into files of ~48 GB, each with its own `general.alignment` (4,096); every tensor's bytes re-read through
  strata-glm's own GGUF reader (`strata::GgufModel`, the offsets a reader computes) and compared with what was written.
  Before 2026-11-25 only the first file carried the key: a reader takes GGUF's default 32 for the others, so it read
  every tensor of files 2 and 3 at the wrong offset (the converter's own re-read used its header size and did not see
  it; the engine failed at layer 15, the first in file 2: "router produced expert -1"). `fix_split_alignment.py` adds
  the key to such a set's later files in place - the header only, the data untouched.

## 7. The reference model

A PyTorch model of the forward pass on the CPU (`moex/tools/ernie55/ref/`), written from the two references where they
agree and from their kernels' arithmetic where only a kernel states it. It reads the FP8 release directly (exact
dequantization to fp32) and computes in fp32. Its uses: the oracle the engine is checked against op by op (with the
engine's own roundings switched on: F16 cache rows, F32 activations into FP8 weights); the source of the importance
matrix (per expert, the sums of squared inputs to gate / up and to down); and the reference distributions for the KL
divergence. Its own checks: each module against the references' torch functions and tests (vLLM-ERNIE's MoE routing,
indexer, VHA, DSpark tests), and the whole model producing coherent greedy text on a chat prompt.

## 8. Phases and checks

| phase | what | done when |
|---|---|---|
| P0 | inventory, references read, this plan | the specs reconciled; the open points listed |
| P1 | the reference model | its modules match the references' functions; coherent greedy replies |
| P2 | the converter; a check GGUF of a few layers (near-lossless types) | every tensor re-read equal; the reference model loads it |
| P3 | strata-glm `Arch::ernie55` on the GGUF | per-op parity with the reference model on the same weights (slices, then all 43 layers); the same greedy tokens |
| P4 | the importance matrix, the quant plan, the KL divergence of each candidate | a mix within the memory budget at the lowest measured KLD |
| P5 | the server family: template (encoding_ernie.py), tokenizer, reasoning and tool parsers, sampling (T 1.0, top_p 0.95) | prompts equal to the release's encoder; parsers on the references' own cases |
| P6 | speed: everything resident, KV on demand, graphs, the expert profile; DSpark | measured on the deployed config |

The GPU phases (P3, P4's KLD on the engine, P6) need the user's server stopped; the reference model runs on the CPU.

### Status (2026-11-25)

- P0, P1 done: the reference model gives "17 * 23 = 391" and "The sea shimmered under the golden sun, its waves whispering
  secrets to the shore." greedily on the release's chat format (`ref/run.py`); its module checks pass (`ref/test_ref.py`).
- P2: the converter is built; the 8-layer check GGUF (IQ3_XXS gate / up, IQ4_XS down, no imatrix: 25.75 GB) converts at
  ~190 s an IQ3_XXS expert tensor on 8 threads.
- P3: `Arch::ernie` written (src/glm/glm_ernie.cpp, src/glm/ernie_kernels.cu; the dispatch sites in glm_stage / decode /
  prompt / engine): the HCA and DSA attention, VHA, the gate, mHC's factor, the two-router routing, the latent norm,
  SiTU-GLU on every expert path (the kSituGlu "limit", strata/kernels/glu_act.hpp), MultiMax, the exact FP8 dense load.
  The kernels against CPU transcriptions (`ernie_kernels_check`): all pass - MultiMax, the E4M3 rounding, the cells and
  the top-2048 with the window excluded bit for bit; VHA 1.4e-7, SiTU-GLU 4.2e-7 against double. The op-level check
  against the reference model: `ref/oracle_dump.py` writes the reference's op outputs under the engine's probe names
  (the routed experts from the GGUF through `ernie55_expert_dq`, ggml's own decoders), `strata-glm --oracle` compares.
- P3 checked on the 8-layer GGUF (layers 0-7: the dense layer, six HCA layers, the first DSA layer), against the reference
  model on the same weights (its routed experts from the GGUF through `ernie55_expert_dq`), op by op with the engine taking
  the reference's values at every probe (`strata-glm --oracle`):
  - 40 tokens, every step: mHC, the projections, the gate, VHA, the routing (the same experts every step), the latent
    MoE's ends, layer 0's SiTU-GLU, the head and MultiMax within 1e-7..9e-7 (float rounding); the attention outputs 2-3e-4
    (the F16 caches); the index keys bit for bit, the index queries a rare E4M3 flip (mean 1e-5); the routed experts' rows
    1.0-1.1e-2 on the GPU (Q8_1 activations), 2.0-2.7e-2 on the CPU pool (Q8_K, llama.cpp's own CPU arithmetic).
  - `strata-glm --expert-check` (now with SiTU-GLU and the latent input): the pool's rows = llama.cpp's CPU arithmetic to
    1e-7..1e-5 (SiTU-GLU on every CPU kernel); GPU 1.0e-2, CPU 2.0e-2 from the exact float64 product.
  - 2,400 tokens, the first 2,390 through each side's prompt path, the last 10 checked: the HCA attention's difference
    grows with depth (2e-4 at layer 0, 7.6e-3 at layer 6: the prompt positions' own rows, not forced); layer 7's top-2048 of
    ~2,265 candidates differs in 5-12 positions at the cut; the engine's own logits within 0.08-0.58% of the reference's,
    the same top token at 9 of 10 positions (the tenth a near-tie).
  - decode graphs = eager windows: the same 24 tokens; decode windows of 8 against single positions on 2 x 1,024 tokens
    (the engine's own rows, `--kld-save` / `--kld --kld-window 8`): mean KLD 0.00028, the same top token at 99.2% - the
    windows' kernels summing in another order, as DeepSeek-V4.1's.
  - KV on demand (the DSA caches on CUDA VMM, on by default): grown from 1,024 to 2,487 positions in steps of 256 over the
    2,400-token prompt, the 24 generated tokens the same as with the caches allocated whole.
  - the prompt path with every expert off the GPU (`--expert-slots 0`: 1,730 experts streamed from the RAM copy to its
    ring): the later positions' differences at the levels of the VRAM run (hca_o-6 7.6e-3, kqv-7 7e-3 mean).
  - the server end to end on the slice (port 18234, the family's tokenizer, template, end tokens 8 / 2, the parser,
    streaming): a reply in the thinking with thinking on, in the answer with thinking off (the slice's 8 layers write
    gibberish); the serve suite 864 OK (14 skipped).
  - the shared code's other users: dsv4_kernels_parity, dsv41_kernels_parity, glm_kernels_parity, iq_avx2_parity,
    iq_avx1_parity, iq_multi_parity and mx_avx2_parity pass after the changes (the SiTU-GLU branch only on a negative
    limit, the top-k's exclusion 0 by default, Hy4's growth arithmetic the same).
- P5: the family `ernie55` (serve/families.py, family_templates/ernie55.jinja, family_parsers.py ErnieSyntax, the tokenizer
  `pre` ernie55 in tools/strata_tokenizer_families.py exported to $STRATA_X_DATA/tokenizers/ernie55): serve/test_ernie55.py
  renders 300 random conversations and the hand cases (every effort, thinking on / off, tools, past calls and results)
  the same as the release's encoding_ernie.py, parses the encoder's turns in every chunking as decoding_ernie.py does, and
  tokenizes rendered prompts as HF; moex/tools/ernie55/verify_tokenizer.py: 14,891 texts (every local corpus and the edge
  cases) id for id with HF tokenizers, encode and decode.
  - The server's request path (2026-11-25, serve/test_ernie55.py ServerPath): OpenAI and Anthropic bodies through
    frontend.openai_to_messages / anthropic_to_messages with the family's Dialect, then the template, give the encoder's
    string - an agent's loop with reasoning on every assistant turn (the release keeps every past turn's reasoning, tool
    calls' turns included), Anthropic thinking blocks. Three cases differed until the family took the release's reading:
    a past turn of reasoning alone was dropped (reasoning_turns), a system message after the first became a user turn
    (late_system), a past turn's reasoning under vLLM's / OpenRouter's `reasoning` was ignored (reasoning_field).
- P4: `ref/imatrix.py` collects the importance matrix and the KLD base in one layer-major pass of the reference model
  (each layer's weights read once): calibration = the DSV4 recipe's text, C4, code and multilingual sets (local);
  held out = wiki, technical and this repository's docs and code.
  - The set: IQ3_XXS gate / up and IQ4_XS down with the importance matrix (16 x 4,096 calibration tokens), the dense
    weights the release's FP8 exactly - `~/AI/Models/ernie55-gguf/ERNIE-5.5-Flash-IQ3XXS-IQ4XS-0000N-of-00003.gguf`,
    135.1 GB in 3 files, written in 6,544 s (16 threads). In memory: the experts 108.94 GiB, the dense 9.01 GiB.
  - Against the reference model's distributions (`ernie55-ref-6x4096.kld`: 6 held-out chunks of 4,096, 12,282 positions
    scored): mean KLD 0.0629 +- 0.0020, median 0.0067, 99th percentile 0.966, the same top token at 93.22%, perplexity
    3.1785 against the reference's 3.0853 (x1.030). Greedy on the reference's chat prompts: "391" for 17 x 23, token for
    token the reference's reply; the sea sentence parts from the reference's at its second token and stays fluent ("The
    vast, rhythmic sea shimmered under the golden sun, whispering secrets of ancient depths to the sandy shore.").
- P6 (2026-11-25, two RTX 3080 20 GB, the deployed config: 1M context, 28 GiB RAM headroom, the learned expert profile):
  - Memory: 2,737 + 2,286 expert slots (25.45 GiB) in VRAM, the rest of the experts (83.49 GiB, pinned) in RAM - decode
    reads nothing from the drive. A long prompt lends each GPU's top slots to its buffers (an 8K read: 2,062 slots, one
    chunk of 7,999); the lend region (the lent slots' RAM copies) holds 323-512 experts at 28 GiB headroom, so the rest
    of the lent experts (~1,550, 7.85 GiB) are read from the GGUF through the page cache - from the drive the first time
    (4.6 s, beside the read's compute).
  - Decode: 20.4-26.4 tok/s on short chats through the server, 23.0 at 78.8K context; a token 44.0 ms = the CPU pool's
    experts 18.2 ms + the wait for the GPUs 25.0 ms; the VRAM tier's hit rate 58-83%.
  - Prompt reads: 8,000 tokens in 26.6 s (300.7 tok/s), 78,814 in ~160 s (~490 tok/s).
  - The server end to end (a test server on port 18234; `moex/tools/ernie55/server_checks.py`, `needle.py`, `multiturn.py`): OpenAI chat with thinking on ("391") and off, a stream at
    effort low, a tool call and its follow-up, a streamed tool call, the Anthropic API plain and a streamed tool_use -
    8 of 8; a dropped stream, then a request at once (0.9 s); JSON mode; a fact placed at 40% of 78,814 tokens of this
    repository's docs found; a three-turn conversation over 50K tokens reuses 50,291 of 50,315 and 50,375 of 50,397
    prompt tokens (3.0 s and 1.6 s to the answer). A follow-up that changes the same user message reuses the last
    checkpoint below where they part (every 16,384 tokens; 65,536 of 78,808).
  - Long reads at the 28 GiB headroom, measured (the 8K read, the deployed arguments, the GGUF's pages dropped before each
    run): no chunk size keeps every lent expert in RAM - the lend region holds 169-281 experts, a 2,048-token chunk lends
    646-804 slots (its mixers' sub-batch buffers are ~1.6 MiB a row at ERNIE's 64 heads x 512, the indexer's score rows
    256 MiB a GPU at 1M positions).
    | headroom, chunk (sub-batch) | lend region | slots lent | read from the drive | 8K read | decode |
    |---|---|---|---|---|---|
    | 28 GiB, auto (one chunk of 7,999) | 169 | 2,158 | 10.08 GiB | 296 tok/s | 20.2 |
    | 28 GiB, 4,096 | 257 | 1,158 | 4.56 GiB | 227 tok/s | 19.5 |
    | 28 GiB, 2,048 | 246 | 804 | 2.83 GiB | 294 tok/s | 20.2 |
    | 28 GiB, 2,048 (sub 256) | 281 | 646 | 1.85 GiB | 290 tok/s | 20.4 |
    | 28 GiB, 1,024 (sub 256) | 268 | 500 | (under 1 s of reads) | 241 tok/s | 19.8 |
    | 20 GiB, auto | 1,844 | 2,158 | 1.59 GiB | 317 tok/s | 20.6 |
    The reads run beside the read's compute (10 GiB in 5.9 s of a 27 s read) and the page cache keeps them for the next
    read; decode reads nothing. Wholly resident long reads need the RAM: ~18 GiB headroom at the full chunk. Left at 28
    GiB (the user's choice).
  - Exact decode kernels (2026-11-25; each the old kernel's bits, the old one behind a switch): the profile of a decode
    token (nsys, the deployed arguments) had the FP8 projections at 15.3 ms of ~43 (190-460 GB/s: the row kernels stage
    each row in shared memory and sync twice a tile), mHC's F32 products 1.6 ms (one load a step), the routing 0.55 ms.
    - `gemv_fp8_warp_kernel` (glm_kernels.cu; STRATA_GLM_FP8_WARP=0 off): T 1, a warp a row, 8-byte loads, each lane the
      partials of 8 of the row kernels' threads, their xor tree as two shuffles and three adds in the lane, the 8 warps'
      totals in order - the same bits (moex/tools/ernie55/fp8_warp_bench.cu: 11 ERNIE shapes, the grouped wo_a and the
      DSA heads, against BF16 / the row kernels; moex/tools/dsv41/fp8_gemv_bench.cu: DeepSeek-V4.1's 8 shapes). Per call,
      the weights from DRAM: q_a 15.0 -> 10.9 us, q_b 126 -> 88, the gate 27 -> 18, wo_b 88 -> 71, the shared expert and
      the latent projections 26-27 -> 16-17, the DSA q_b 53 -> 30; wo_a (gemv_fp8_heads) 94 -> 74, the DSA k_b / v_b 54 /
      37 -> 23 / 18.
    - `gemv_f32_w1_chunk_kernel` (STRATA_GLM_F32_CHUNK=0 off): T 1, 16 loads a thread ahead of the same multiply-adds -
      the same bits (the bench's CPU emulation of the row kernel's order); mHC's 16,384 x 24 14.5 -> 6.8 us.
    - `route_warp_kernel` (ernie_kernels.cu; STRATA_ERNIE_ROUTE_WARP=0 off): the scores as before, the k rounds in one
      warp without block syncs - the same ids and weights (ernie_kernels_check: 2,000 tokens with ties, +-inf and NaN
      rows, E 512 / 1,024 / 100, k 1-32); 11.7 -> 11.1 us a decode token, 79 -> 56 us for 2,048 prompt rows. Only finite
      choices are taken (the block kernel could repeat an id once every other choice was NaN).
    - End to end, one layout and fixed paths (below): the same 256 greedy tokens as the binary before; decode 22.5 -> 26.1
      tok/s. The deployed arguments, two runs an arm alternating: 26.0 / 26.7 -> 29.3 / 31.3 tok/s (a token 35.1 / 34.4 ->
      30.9 / 28.8 ms, the GPUs' share 25.8 -> 19.5-20.1). Through the server: short requests 24.3-32.3 tok/s (18.3-26.3
      before, without the pool switches either), 27.6 tok/s at 78.8K (23.0); the 8 API checks and the 78.8K fact again,
      word for word the same replies.
    - The shared kernels' suites after the change: glm_kernels_parity, dsv41_kernels_parity, dsv4_kernels_parity,
      bf16w_parity, mmvq_grouped_parity, native_grouped_parity - all pass.
  - The CPU pool's switches (strata-ernie55.json "env": STRATA_POOL_SPREAD=1, STRATA_POOL_PIPELINE=1, STRATA_POOL_TASKS=252,
    Hy4-flash's): decode 23.99 / 25.30 -> 25.71 / 26.82 tok/s (alternating pairs), the same 256 tokens at one layout.
  - `STRATA_DSV41_HUGE_PIN=1` (strata-ernie55.json "env"; DeepSeek-V4.1's configs have it: the RAM copy on 2 MiB pages, its
    faults paced by fault_in_paced): GPU 1's copies from pinned memory 8.0 -> 12.1 GB/s on its x16 link (GPU 0's x8 link
    6.2 either way; h2d_bw, 1.75 MiB pieces), the 8K read 296.3 / 296.4 -> 298.8 / 299.0 tok/s, decode after it 24.69 /
    24.73 -> 25.05 / 25.11, the RAM copy pinned in 10.8 s instead of 28.4. The 8K read's profile: the GPUs busy 39-44% of
    it - the dense projections as FP32 SGEMM (FP8 widened; 4.9 / 4.2 s), the streamed experts' copies (GPU 0 53 GiB at
    6.2 GB/s, 9.3 s), the DSA layers' attention 1.2 / 1.3 s, the MoE's MMQ only 1.2 / 1.1 s.
  - Run to run, the tokens part (at 26-86 of 256 greedy tokens) even with the adaptive tier off: the prompt path's timed CPU
    share puts an expert on the CPU (Q8_K activations) or the GPU (Q8_1) by its timing - HY4_PORT.md's finding. With fixed
    paths and one layout (`STRATA_GLM_SHARE=0 STRATA_GLM_CPU_READ=0 STRATA_GLM_STAGE_SLOTS=2700,2250 --no-adaptive
    --layer-split 24 --expert-slots 4950`) four runs gave the same 256 tokens: the check for exact changes.
  - Prompt lookup (Strata's suffix drafter, on by default: `suffix_draft` 3, windows up to 6 rows) works for ERNIE through
    the server (the CLI's greedy loop runs without it): a 493-token verbatim repeat of a 1,600-character passage, greedy,
    408 of 410 drafts kept, decode 44.6-46.0 tok/s against 27.7 without drafts (the CLI, the same prompt), the reply the
    passage exactly. Its windows averaged ~5.8 rows at ~126 ms - 3.8x a single token: a window's routed experts barely
    overlap (the CPU pool's and the VRAM tier's reads grow with the rows), so a verify window is dear on this box.
  - `gemv_fp8_warp_t_kernel` (T 2 .. 8, the verify windows; STRATA_GLM_FP8_WARP=0 off): a warp a row, each step's weights
    decoded once for every column, each (row, column) in the row kernels' order - bit for bit at T 1-8 on DeepSeek-V4.1's
    shapes (fp8_gemv_bench); q_b at T 4 / 6 / 8 228 / 368 / 380 -> 184 / 263 / 318 us, wo_b 154 / 253 / 285 -> 161 / 245 /
    284 (the inputs' loads bound it now: 32 bytes of x for every 8 bytes of weights a column).
  - The verify cost of a window (`--decode-runs f1,f2,f4,f6,f8`: stand-in drafts, all rejected - one token a window; the
    sea prompt, short context, the deployed switches), the binary before -> the exact kernels: 1 row 36.4 -> 29.2 ms (27.5
    -> 34.2 tok/s), 2 rows 60.6 -> 44.0, 4 rows 69.1 -> 62.7, 6 rows 100.6 -> 88.5, 8 rows 113.2 -> 106.2 ms (3.6x a token).
    The windows' bits: `--kld ... --kld-chunks 1 --kld-window 8 --kld-save` at one layout and fixed paths, the binary before
    and the new one: the saved distributions (2,047 positions in windows of 8) byte for byte the same (823,811,084 bytes).
  - Not ported: DSpark (the release's drafter: three draft stages, each a whole 512-expert MoE layer, 9,358 tensors). On
    this box a verify window costs ~3.8x a token at ~6 rows (measured above, the prompt lookup's windows), so DSpark's
    8-row windows need more than ~5 of its 7 drafts kept a window to break even - before its own three MoE stages - and
    its 1,536 experts need ~5 GB more RAM at ~2 bits (the 28 GiB headroom down to ~23, or the trunk's experts from the
    drive); DeepSeek-V4.1's DSpark measured -20% here. The engine's V4.1 DSpark (block attention, the arena's experts on
    the CPU pool, the Markov and confidence heads) is the base if it is wanted: a converter for model.mtp.*, the drafter's
    blocks bound as ERNIE layers with ratio 0, ERNIE's latent MoE in arena_moe, its head reduce and MultiMax before the
    Markov steps, the capture at layers 41 / 42 / 43's inputs. The release has no MTP layer and no image input.

- **The hill-climb (2026-11-25, the inference-optimisation skill; ~/AI/Strata-X-data/ernie55-hillclimb: THEORY.md's ceilings,
  PLAN.md's log, ledger.jsonl)**: the skill's harness (bench.py against a test server on port 18234 with the run config minus
  the profile save; the panel's six prompts at 512 tokens greedy with thinking off, cold 2,663- and 20,582-token reads),
  controls around each candidate; engine levers first on the CLI at one layout with fixed paths.
  - Config levers, rejected (inside noise or worse): --threads 19, STRATA_POOL_SMT=1 at 38 threads (-2.3..-3.7%),
    STRATA_POOL_TASKS 126 / 504; in pool_decode_bench (moex/tools/ernie55: decode's call pattern on the GGUF's experts)
    chunked claims and a static split of the rows were slower than the pool (193 us a 2-expert layer; 219-286).
  - The decode profile (nsys, the deployed arguments): an HCA layer's part before the routing ~381 us on one stream - its
    eight big GEMVs 263.5 us for 149 MB (565 GB/s), ~30 small kernels ~120 us; the GPU's part of a token ~18 ms of ~33.
  - Exact kernel levers kept, each the kernels it replaces (an env switch turns each off): FP8 / BF16 GEMV pairs of one input
    (gemv_pair_w: q_a + the KV row, the compressor's two rows, the routers + routed_down; STRATA_GLM_PAIR=0); the combine's
    preload kernel at 10 slots and wherever the engine's preload default is set (STRATA_DSV41_PRELOAD=0); the routing and the
    doorbell publish in one launch (STRATA_ERNIE_ROUTE_PUBLISH=0); mHC's Sinkhorn chain on s3 beside the half, the read's
    pre coefficients and norm in one block, the mixer's rows as their 8 warps' totals (STRATA_ERNIE_MIX_SIDE=0,
    STRATA_GLM_F32_WARPS=0); rms_eo's loads ahead; VHA over 4 blocks a tile for windows up to 8 rows (STRATA_ERNIE_VHA_SPLIT=0;
    at every size it cost an 8K read 0.7-0.8%: each block reloads a prompt chunk's input tiles); one token's identity
    permutes skipped (STRATA_ERNIE_PERMUTE_COPY=1); the HCA query's / KV row's norm, RoPE and raw store in one launch
    (dsv4_norm_rope; STRATA_DSV4_NORM_ROPE=0); routed_up's unread Q8_1 copy skipped. At the fixed layout (the imatrix
    profile, 256 greedy tokens - six runs, the same tokens as f2cea209 - and verify windows of 8 after a 2,048-token prompt
    path, their saved distributions byte for byte): the GPU's wait before the routing 17.9 -> 16.4-16.5 ms a token, decode
    25.65-25.91 -> 26.66-26.95 tok/s. The shared parity suites pass (glm / dsv4 / dsv41_kernels_parity, ernie_kernels_check,
    preload_parity's three forms byte for byte, fp8_warp_bench's pairs bit for bit).
  - Through the server (bench.py, controls around each run): c1 decode 30.2 -> 31.5 tok/s (+4.3%; the old binary's 11
    controls 29.86-30.44, the new one's 8 runs 31.23-31.74), prefill 330.8 -> 328.5 tok/s at 2,663 tokens and 493.5 -> 490.9
    at 20,582 (-0.7 / -0.5%), TTFT 684 -> 686 ms. The VRAM tier's cadence and decay (--adapt-every 2 / 8, --adapt-decay 0.5 /
    0.85 - the last's first win did not repeat): all inside noise or worse, the defaults stay.
  - The head over both GPUs (DeepSeek-V4.1's §76, now ERNIE's too: `STRATA_DSV41_HEAD_SPLIT=P`, 45% of the 201,216 BF16 rows on
    GPU 0 by default, 0 = whole on GPU 1; the binding shared as bind_head_split, MultiMax after the two runs are joined): a
    token's "rest" was 2.8-3.2 ms with GPU 0 idle (a diagnosis run, the run config with diagnostics on). The same tokens at
    one layout (2,500 + 2,150 slots, five runs), verify windows of 8 byte for byte (with `--resident-headroom-gib 12` both runs
    pin the same RAM copy - the CLI's default 28 leaves the copy's edge moving with MemAvailable, so a few experts change
    between the blob form and the in-place form, which rounds otherwise). CLI decode 26.71 -> 27.47 tok/s (-1.2 ms a token);
    through the server 31.30 -> 32.39 tok/s (+3.5%, two runs, controls around each).
  - The pinned binary (882a9836) is that build with VHA's split for windows only: an 8K read 293.6 / 294.0 tok/s against the
    old binary's 293.9 / 294.6 (one layout, fixed paths; hc-k3, the split at every size, 291.5 / 292.4), the same tokens.
  - The gates on hc-k3 (the run config on port 18234): the 8 API checks (server_checks.py) PASS; a passphrase at 50%
    and at 95% of a 240,943-token document (760,000 characters of these docs, ~23% of the window) found exactly, the
    follow-ups on the cached text answered (needle.py).
  - The prompt path (2026-11-25, later): a 24K read's profile (nsys) - GPU 0 (27 layers, the x8 link) ~14.2 s a chunk, its
    kernels 11.8 s (FP32 SGEMM on the widened FP8 projections 6.4 s) against its copies 10.9 s (67 GB at 6.21 GB/s),
    overlapping in part; GPU 1 (16 layers) 8-10 s a chunk and idle between them. The SGEMMs run at ~11 TFLOPS because the
    cards sit at their 180 W limit under held FP32 math (915-1,170 MHz; BF16 tensor-core GEMMs hold 43-54 TFLOPS, not
    exact for F32 activations). The automatic split maximises the decode profile's mass in VRAM with a slot model from
    before the FP8 dense weights and the head split (1026 + 1065 slots predicted at 1M, 1626 + 1610 loaded); offline, the
    decode mass is flat from 23 to 27 layers on GPU 0 (47.37-47.60% at short context), and the DSA layers (7, 14, 21, 28,
    35, 42) stay three a GPU from 22 to 28.
  - The run config now says `--layer-split 23 --prefill-ring-frac 0.75` (the CLI, the 24K prompt, controls around each):
    the split 54.5 / 54.4 -> 49.8 s (+9.2%; 24: +6.3%, 22: no more); then the full ring at 384 slots instead of 288
    (0.55 of a layer's 512, GLM-5.3's measured best) 49.7 / 49.9 -> 47.0 s (+5.9%: the copies run further ahead through a
    layer's longer attention and dense half; 448 and 512 the same). `--prefill-ring-frac F` (EngineOptions::prefill_ring_frac,
    0 = 0.55) sets the full ring as a layer's share and keeps the short reads' smaller rings, which an explicit
    `--prefill-ring N` turns off. Through the server (hc-k7 = hc-k6's kernels, SASS identical, + that option): decode
    32.82 tok/s against 32.16-32.92, a cold 240,953-token read 532.3 -> 490.8 s (+8.5%) and the 95% needle's 175K new
    tokens 416.6 -> 380.8 s (+9.4%); the 20,582-token read the same (498.1 against 495.5-499.3: the server read it as
    four chunks shared with the CPU pool, no stream - the engine's learned share limit - so the ring and the split hardly
    enter); the 2,663-token read 328.5 against 330.8-334.1 (-0.7..-1.6%: four layers more on GPU 1, the display card, ~10%
    slower a layer). Gates on hc-k7: the 8 API checks PASS, the passphrase at 50% and 95% of the 240,953-token document
    found, the follow-ups answered. Not taken: `--prefill-sub 1024` (+1.3%; its SGEMMs pick other cuBLAS kernels, other bits);
    the projections widened once a layer and chunk instead of once a sub-batch (exact, byte for byte; the 24K read 48.1 /
    48.6 -> 48.9 s: the widening was hidden behind GPU 0's copies, and its 122 slots a GPU stream again every chunk).
  - The prompt's dense GEMMs on BF16 tensor cores (`STRATA_GLM_TC3=1`, opt-in in the engine, on in the run config): gemv_dt for
    T > 8 takes the weights' exact BF16 values (FP8 -> BF16 is exact: fp8_bf16) and x split exactly into hi + mid + lo BF16
    planes (split3_bf16), three cublasGemmEx with F32 accumulation, the small planes first - exact products, F32 sums in
    another order: not the SGEMM's bits. At the cards' 180 W limit FP32 SGEMM holds 9.5-11.9 TFLOPS, a BF16 GEMM 43-54
    (sgemm_bench). Quality against the FP8 reference (6 held-out chunks, 12,282 positions, one layout): mean KLD 0.062748
    +- 0.002018 against the default path's 0.062521 +- 0.002023, median 0.00668 / 0.00680, the same top 93.26 / 93.31%, PPL
    3.1750 / 3.1791 - inside noise. The 24K read 47.9 / 47.8 -> 44.6 s; with it split 22 beats 23 (42.5 against 43.7 /
    43.5 s; the DSA layers stay three a GPU). Through the server (the run config): decode 32.8 tok/s, the 2,663-token read
    381.8 tok/s (328.5-334.1 before), 20,582 tokens 591.0 (495.5-499.3); the 8 API checks PASS.
  - The FP8 warp GEMV against its read pattern (fp8_pattern_bench): the pattern alone reads at 670-705 GB/s on the big
    shapes, the kernel at 470-560. Three exact variants (the next step's loads issued first; deeper unrolls; E4M3 -> FP16 by
    moving bits, exact for every code, then widened and scaled) are bit for bit the kernel's and inside the bench's spread;
    an elimination run puts the gap in the integer work and loads per byte (no decode math -16%, no x loads -5%, one scale a
    row -5%), no single exact change removes it: the kernel stays. Rows a warp 3 and 8: slower or the same.
  - A bit-exact copy needs the SASS: rope_kernel's (x c - y s, x s + y c) compiled to two FFMAs (ptxas fused both), and a
    "line for line" copy rounded differently twice (tokens diverged at 47, then 73); both now spell that SASS with
    __fmaf_rn / __fmul_rn, and rope_kernel's SASS diffs to nothing against the pinned binary's.
  - The CPU pool: an ERNIE expert's IQ3_XXS gate/up rows run at 3.38 GB/s a thread, its IQ4_XS down rows at 6.31 - compute-
    bound (6.2 ns a 64-value group from L2 or DRAM; the sign decode's scalar ops and the gathers fill the ports); 67 GB/s at
    20 threads against a 76-81 GB/s read.

### Open (2026-11-25)

- **Long reads**: the run config now leaves 20 GiB (§9; was 28): an 8K read re-reads ~1.6 GiB of lent experts on a cold page
  cache (10 GiB at 28); decode reads nothing.
- **DSpark**: deferred (the user, 2026-11-25), its acceptance measured offline the same night (moex/tools/ernie55/ref/
  dspark_accept.py on the FP8 reference: the engine's greedy replies to four chat prompts, 1,610 blocks of 7): 3.13 of 7
  drafts kept, the first 81%, each later one 77-81% given the ones before; tokens a window of 4 rows 2.94 (code 3.60, math
  3.56, an explanation 2.60, a story 2.07). With a window's measured cost (4 rows 2.15 tokens' time) and the drafter's own
  ~0.5-0.63 tokens (three stages of a top-10 latent MoE for all 7 positions, from RAM on the CPU pool: 0.57-0.79 GB; 1.14 GB
  and 0.85 tokens if no two positions share an expert), greedy decoding would gain 6-11% on mixed text (-2% at 0.85), 28-36%
  on code and math, and lose 2-6% on an explanation and 22-26% on a story: a lever for code-heavy use with a window that
  follows the recent acceptance, not a flat win. At T 1.0 / top_p 0.95 (the run config) a draft is kept less often than
  greedy (not measured: the script with the target's probabilities and the drafter's routing, ~10 min of CPU, comes first
  if it is taken up). Its experts: 8.4 GB of RAM in the set's recipe (~5 GB at ~2 bits) out of the 20 GiB headroom. The
  base, when it is taken up, is the engine's V4.1 DSpark.
- **Quality**: the set's KLD 0.063 is what fits: a richer mix (IQ3_S gate / up on the sensitive layers, ~+2.6 GiB) would
  take the lend region's RAM, and a second 135 GB set does not fit the drive's 31 GB free.
  2026-11-27: measured - §10 (IQ3_S for all three is worse at the same size; IQ3_S gate/up with IQ4_XS down, +7.2%, ~-31%
  at the MoE's output).
- **The moex family tests** (moex.tests.test_family_parsers / props / merge): 60 failures in other families' cases
  (mimo2, glm5, deepseek4: quoted think ends, a tool call's trailing newline) - the same at f36bca4f without this work.

## 9. Running it, and the checks

- **The server**: `launch-ernie55-flash.sh` (port 1234, model name `ernie-5.5-flash`; it refuses while another Strata server
  runs; extra arguments go to serve/server.py unchanged, as the DeepSeek launcher's - `--host 172.17.0.1 --api-key KEY`
  for a Docker bridge, a later `--port` overrides 1234; tools/test_ernie55_launcher_args.py) with `strata-ernie55.json`: the pinned binary `build-dsv41/strata-glm.ernie55`, the GGUF set
  `~/AI/Models/ernie55-gguf/ERNIE-5.5-Flash-IQ3XXS-IQ4XS-0000N-of-00003.gguf`, the tokenizer directory
  `~/AI/Strata-X-data/tokenizers/ernie55`, the native 1M context, the release's sampling (T 1.0, top_p 0.95), thinking on
  by default (effort high; Strata's low / medium / high map to the release's, xhigh and max to its high).
- **The binary since 2026-11-26** (the engine renamed `strata-x`: moex/README.md's table): `build-dsv41/strata-x.ernie55`
  (md5 fce161c8), 8a352c35's source with the new names. The SASS of all 2,487 GPU functions is 8a352c35's (the namespace and
  the source files' names aside), the binary holds the same 264 `STRATA_*` names (STRATA_GLM_* as STRATA_X_*), and the CPU
  tests and the serve suite pass; it has not run on the GPUs yet. strata-ernie55.json names it, with `STRATA_X_TC3=1`. To go
  back: `"exe"` `build-dsv41/strata-glm.ernie55` and `STRATA_GLM_TC3`.
- **The RAM headroom** (2026-11-25, the user): strata-ernie55.json's args end with `--resident-headroom-gib 20` - the RAM the
  engine leaves the desktop when it plans the RAM copy. At 20 the lend region holds ~1,844 experts and an 8K read re-reads
  1.59 GiB of lent experts on a cold page cache (10.08 GiB at 28; P6's table). To go back, set it to 28 or delete the two
  entries (the engine's default is 28).
- **Sampling** (2026-11-25, the user): the release's T 1.0 / top_p 0.95 (generation_config.json; SGLang's ERNIE-5.5
  cookbook names those two; vLLM takes them as defaults and leaves top_k at 0, all tokens), with no top-k cut: the family
  has full_vocab (a request naming no positive top_k reaches the engine as top_k 0). Before, the engine's default top_k
  20 applied: on the FP8 reference's own distributions (the KLD base, 3,071 positions, T 1.0) the 0.95 nucleus is wider
  than 20 tokens at 21.0% of positions and wider than 64 at 12.0% (median 2, 90th percentile 89, 99th 818); the total
  variation from top_p 0.95 over the whole vocabulary 0.042 at top_k 20, 0.023 at the widest list (64).
  - 2026-11-25 night (the user: "correct and matching release"): the whole vocabulary's nucleus in the shared sampler
    (DSV41_BACKLOG #56), so top_k 0 with top_p < 1 samples as vLLM does - softmax(s / T) over every token (temperature first:
    the engine now orders ERNIE's sampling as DeepSeek-V4.1's, temp_first), the smallest set of the likeliest tokens holding
    top_p, the draw inside it. Each row's cut is a key from a radix select (sampler_nuc_max_kernel, sampler_nuc_kernel<0..4>:
    four passes of 8 key bits over the spans the split Gumbel-max uses, each span's histogram of the weights exp((s - max) / T)
    as 40-bit fixed point in warp-private 32-bit parts), then the split Gumbel-max skips the ids below it. Integer sums, so a
    row's cut is its logits' alone: a verify window's rows draw what single steps would. sampler_parity: frequencies equal HF
    top_p's in both orders (227 / 210 tokens kept on a flat row, total variation 0.012 / 0.011 over 204,800 draws, none
    outside), 8-row windows = single steps, at 129,280 tokens every row's pick = the host's nucleus + Gumbel-max (0 of 32
    rows, nuclei of ~46,600 tokens); the top_p 1 path's draws unchanged. One row at 201,216 tokens: 87 us (a narrow nucleus)
    to 157 us (a third of the vocabulary kept), against the 64-token list's 139-152 us. strata-ernie55.json's `sampling` has
    no top_k now (T 1.0, top_p 0.95: the family's, the release's). STRATA_FULL_NUCLEUS=0: the 64-token list, as before.
- **The prompt as the release's server builds it** (2026-11-25, night; the user: "fix all four"). A side-by-side of the
  server's whole prompt path - the conversion, the #537 marking of every special token's text inside a message, the
  template, the unmarking - against encoding_ernie.py's segments (the same text, and a special token at the same places:
  encode_segments makes one only where the renderer wrote one, and joins the ordinary text between them as our encode
  does, so the ids are the same) found four differences, now gone: (1) a past call's arguments were sorted by Jinja's
  case-folding `sort` (`a_b` before `aB`, `Path` last; the encoder's `sorted()` is case-sensitive) - the template sorts
  case-sensitively; (2) an assistant's reasoning written inline (`<ernie:think>R</ernie:think>answer`, no reasoning field)
  was marked before the template could split it, so the turn became an empty think block and the tags as text -
  Dialect.think_split now splits it before the marking, as the encoder does (the first close tag, the last open tag before
  it, the content after the last close tag; a reasoning field that is a string, even empty, keeps it whole); (3) a past
  turn's `thoughts` (the encoder's first field) was dropped - Dialect.thoughts_field; (4) `parallel_tool_calls: false`
  (and Anthropic's `disable_parallel_tool_use`) was ignored - Dialect.parallel_option passes it to the template, which then
  omits `parallel_tool_calls=True` as the release's vLLM plugin renders it, and the reply keeps its first call
  (FirstCallOnly; the release's server constrains the decoding to one invoke). The Responses API's converter does (2) and
  (4) too. serve/test_ernie55.py PromptPath / OneCall (5 tests; against the previous code the new cases fail on exactly
  these four): 18 OK in both venvs; the serve suite 872 OK (14 skipped). Matching before and after: the agent loop's 25
  special tokens, tags quoted in user, tool and argument text, the tool list's and nested values' JSON, late system and
  developer messages, the efforts (max reads as the release's high; vLLM's renderer would refuse max).
- **Queued for later** (the user, 2026-11-25: "we can do without it for now"): decode's remaining levers, each ~1-2% and 1-2 h -
  a free-order FP8 GEMV (16 B loads; ~+2%: fp8_free_bench in the hill-climb's notes), the ~25 small kernels a layer fused,
  the CPU pool's gate/up -> down handoff (~40 us a layer); ERNIE's DSpark drafter (not ported; measured offline: greedy
  +6-11% on mixed text, +28-36% on code and math, a loss on prose - the DSpark item above).
- **Engine switches** (defaults keep the release's arithmetic): `STRATA_ERNIE_KV_GROW=0` allocates the DSA layers' caches for
  the whole context at the start (default: mapped as a conversation grows, `_INIT` / `_STEP` 65,536 positions);
  `STRATA_ERNIE_IDX_FP8=0` scores the indexer exactly instead of on the references' E4M3 rounding (a check's switch);
  `STRATA_GLM_FP8_WARP=0`, `STRATA_GLM_F32_CHUNK=0`, `STRATA_ERNIE_ROUTE_WARP=0` the decode kernels before 2026-11-25 (the
  same bits, slower). The run config's "env" sets the CPU pool's switches (STRATA_POOL_SPREAD / PIPELINE / TASKS=252) and STRATA_DSV41_HUGE_PIN=1.
- **The binary**: `build-dsv41/strata-glm.ernie55` (md5 d7ca0a3d, 2026-11-25 night: 882a9836's kernels + the head over both GPUs +
  `--prefill-ring-frac` + the opt-in tensor-core prompt GEMMs (STRATA_GLM_TC3), built from the working tree - not committed yet;
  every earlier kernel's SASS unchanged, two added); the ones before kept beside it: `strata-glm.ernie55.3c9265ee` (no TC3),
  `.882a9836` (the exact kernel levers), `.f2cea209`, `.prev-20261125` (facd5800). The run config: `--layer-split 22
  --prefill-ring-frac 0.75` and "env" STRATA_GLM_TC3=1. To go back one step: copy `.3c9265ee` over it, split 23, drop the env.
- **The checks**, each without the server:
  - `build-dsv41-dev/ernie_kernels_check` - the new kernels against CPU transcriptions (a GPU, no model);
  - `ref/oracle_dump.py --gguf G --ids @F --out D [--probe-from N]` then `strata-glm -m G --tokens-file F --oracle D
    [--probe-from N]` - the engine op by op against the reference model on the same weights;
  - `strata-glm -m G --expert-check D` - the routed experts' GPU, CPU-pool and llama.cpp-CPU arithmetic against the exact
    float64 product (D: a dump with `ffn_moe_lat`);
  - `strata-glm -m G --kld ~/AI/Strata-X-data/ernie55-quality/ernie55-ref-6x4096.kld` - the quantized set against the
    reference model's distributions;
  - `serve/test_ernie55.py` (both venvs) and `moex/tools/ernie55/verify_tokenizer.py` - the family against the release's
    encoder, decoder and tokenizer.
- **Remaking the GGUF**: `convert_gguf_ernie55 --model ~/AI/Models/HF/ERNIE-5.5-Flash-Preview-FP8 --out PREFIX --types
  gu=iq3_xxs --types d=iq4_xs --imatrix ~/AI/Strata-X-data/ernie55-quality/ernie55-16x4096.imx --split-gb 48` (a layer
  range per type: `--types d=iq3_s:1-20`); `--layers 8` writes a test slice. The importance matrix and the KLD base:
  `ref/imatrix.py` (one layer-major pass of the reference model, ~2 h on the CPU).

## 10. The loop of 2026-11-26, and the recipe (2026-11-27)

The 41,661-token runaway of 2026-11-26 (the session the user named, decoded to a scratch folder for this and deleted
after) was a thinking that never closed: ~29K tokens of planning, then a cycle of six short lines that took the
thinking over (from reply token ~36.4K on, no other line). This section is what the night of 2026-11-27 measured about it,
and about the set's recipe. Every number below came from a run that night; the tools are in moex/tools/ernie55 (`kld_compare.py`,
`imx_tools.py`, `ref/kld_base.py`, `ref/recipe_check.py`, convert_gguf_ernie55's `--measure` and `--fakequant`).

- **The sequence**: the request rebuilt from the session (its system prompt, four user turns, 21 tools, effort max) renders
  to 7,931 tokens - the server log's count for that request - and the reasoning re-tokenizes to 41,652 of the 41,661
  generated. Together 49,583 positions; a KLD base of one chunk scores positions 24,791-49,581 (the reply's tokens
  16,861-41,651): the planning, the last ~4K tokens before the cycle and the cycle itself.
- **The engine at depth** (the live binary and run config, `--kld ... --kld-save`): the first half read through the prompt
  path and the scored half through decode windows of 8, against everything through decode windows (`--kld-mode windows`):
  mean KLD 0.004 (median 0.00000), the same top token at 98.9% of the positions; in the cycle 0.00003, 100%. The engine's
  own probability of `</ernie:think>` (id 20): one position above 1% in the planning (0.14, in the middle of a line), at the
  encoding's floor (~1e-7) through the last ~4K tokens and the whole cycle; the
  cycle's tokens at a mean probability of 0.977.
- **FP8 at depth** (`ref/kld_base.py`: the reference model, fp32 with the release's exact FP8 weights, over the same 49,583
  tokens; 7,447 s on the CPU; `kld_compare.py`): against the engine's rows, mean KL 0.067 over the scored half (median
  0.00001), the same top token at 96.0% - by region, the planning (reply tokens 16.9K-32.1K) 0.086 / 94.9%, the ~4K before
  the cycle 0.081 / 94.9%, the cycle 0.00025 / 99.96%. In bins of 2,048 positions from 24.8K to 45.3K: 0.03-0.12, with no
  growth (the set's KLD on the plain-text base is 0.063 at positions 2,048-4,094). The engine's two read paths stand at the
  same distance (0.067, 0.064).
- **What FP8 does on that text**: it never closes the thinking either - `</ernie:think>` below 0.0005 at every one of the
  planning's 1,137 line starts and the last ~4K's 400 (the engine's the same; FP8's highest anywhere 0.056, mid-line), at the
  encoding's floor through the last ~4K and the cycle (mean 1.3e-7 and 1.1e-7); the cycle's tokens at a mean probability of 0.977 (the engine 0.977). At the first cycle lines (from reply
  token ~30.9K) the two give the line's first token similar odds (FP8 0.084 / 0.605 / 0.189 / 0.642 ..., the engine 0.081 /
  0.670 / 0.123 / 0.928 ...); over the planning's 14 cycle lines the quant leaned more to them (0.74 against 0.54), over the
  last ~4K's 110 lines about the same (0.87 / 0.83).
- **So**: on this request the stall is the model's own - the exact weights in an fp32 implementation of the references'
  arithmetic plan without closing and continue the cycle as the set does; neither the engine nor the quantization made it.
  Not shown: whether FP8 would have reached such a cycle on a trajectory of its own (the text was sampled from the set).
  What ends it is the server: since this night `"reasoning_loop_recovery": "wrap"` in the run config (serve/server.py; at
  `"reasoning_loop_coverage"` 1.0 it would have fired at ~37.8K reply tokens on this text) closes the thinking with the
  family's wrap-up and lets the model answer, where `"stop"` ended the reply with none. On the real engine (the live
  binary, a test server): the same wrap-up path, reached through a thinking budget of 150 tokens, closed the thinking
  mid-sentence and the model answered in full (finish "stop", 587 tokens). A loop could not be provoked on demand: asked to
  write one line 400 times in its thinking, the model declined ("filler text doesn't accomplish anything") and answered.
- **"wrap" on the real loop** (2026-11-27 morning; the session decoded again for this and deleted after): the stalled
  request's own 7,931 prompt tokens and its reasoning up to reply token 36,000 (where the cycle was taking over), through the
  server's request code (Service.run, the request's 21 tools) on the real engine with the IQ3_S / IQ4_XS set and the run
  config's sampling. The model went on cycling; the guard read a fully repeated window at 2,560 generated tokens and wrapped
  the thinking up, and the model answered with one line and a tool call (`bash`) - its task's first step, as an agent
  goes on - where the original reply ran 41,661 tokens until it was cancelled. Again through the public checkout's code
  with its run config's own "wrap": fired at 3,072, the same answer shape. Two more checks: without the guard one resumed
  run left the cycle by itself after 683 tokens; closing the thinking at reply token 24,000 (as a 24K thinking budget would)
  gave the same one-line answer and `bash` call, so a budget is not needed for this request to go on.
- **The engine and the reference against the official code** (2026-11-27 morning, at the user's doubt that the engine was
  cleared only against our own reference): two audits read `ref/model.py` line by line, one against vLLM-ERNIE (ba322a5 on
  vLLM 5df63a7, the vLLM files the reference was written without - the sparse-attention backends, the compressor's helpers,
  the indexer's top-k - fetched at that commit), one against SGLang's port (f4284f2). No semantic difference on any
  long-context path: the compressor (a block completes at (p+1) % 128 == 0, the ape row p % 128, a softmax per channel over
  the block, the norm, the compressed row rotated at the block's first position 128k), the rows a query at p sees ((p+1) //
  128 compressed rows with no cap, the window p-127..p, one softmax with the sink, scale 512^-0.5), YaRN (theta 160000,
  original 65536, factor 16, beta 32 / 1, no attention factor: frequency and angle tables bit-equal to both up to position
  1,048,575), the DSA layers (the indexer's NeoX RoPE on 64 of 128 dims, Hadamard, the FP8 rounding, candidates 0..p-128,
  an exact top-2048, then the window). The differences are rounding: vLLM's README runs FP8 KV caches and rounds parts of
  the indexer to bf16 (the reference and the engine keep more); SGLang's fused indexer path scales its FP8 rounding by
  amax / 448 where vLLM-ERNIE and the reference use a power of two (only the candidates at the 2,048 cut, in the six DSA
  layers). Not read: the FlashMLA, FlashInfer and DeepGEMM CUDA kernels themselves (their sources are not in either repo).
- **Long context helps the engine** (the live binary and run config): the same final 2,047 tokens of
  moex/docs/DSV41_PLAN.md (one engineering log; its tokens 58,849-108,000) read with 4K, 16K and 48K of context before them -
  mean NLL 2.649 / 2.486 / 2.309 (perplexity 14.1 / 12.0 / 10.1): no plateau or rise to 48K. With the passphrases found at
  50% and 95% of a 240,943-token document (above), the engine and the reference are not where the stall comes from; what
  this box cannot test is Baidu's own model and serving.
- **No thinking budget** (2026-11-27, the user: "We already have the loop guard so why cap? Let the model think free"): the
  run config sets no `reasoning_budget_tokens`, so a thinking runs as long as the model takes it; what ends a cycle is the
  loop guard ("wrap", at its look: 2,560 and 3,072 generated tokens on the real loop). A budget of 32,768 was in the
  template for an hour that morning (public b0867eb8) and is gone again. For the record, from that hour: the dev server's
  log of 774 ERNIE-5.5 requests had generated tokens a reply median 376, 99th percentile 4,896, the longest normal one
  12,782, and only the stalled one past 16,384.
- **A stall in another order** (2026-11-27 afternoon): the guard's passage count catches a cycle only when a 12-word
  passage repeats - the stall's six short lines in their order, at 1,024 generated tokens on a scripted engine with this
  family's template and tokenizer, but the same lines shuffled, or shuffled with a new line every 20, ran 16K tokens with
  no look reading them as a repeat. The look now also reads a cycle of short lines in any order (serve/server.py
  `line_cycle`: of the thinking's last 256 non-empty lines, 90% copies of at most 12 distinct lines under 12 words): all
  four shapes end at 1,024 tokens, and the model's answer follows the wrap-up. Number drafts, code, a list, prose and
  short asides read as work (serve/test_reasoning_loop_recovery.py, LineCycle; the serve suite 899 OK). A thinking budget
  (`reasoning_budget_tokens`) would end such a planning earlier, but it also cuts a long thinking that is not looping: not set.

### The recipe

- **The routed experts' quantization error by type** (`convert_gguf_ernie55 --measure`, 16 experts a layer picked evenly
  along each layer's use, the set's importance matrix as the converter uses it; the error power the output sees,
  sum_j imx_j (w - q(w))^2 over sum_j imx_j w^2): nearly the same in every layer - gate and up IQ3_XXS 2.98-3.26% (mean
  3.20%), IQ3_S 1.83%, Q3_K 2.07%, IQ4_XS 0.57%, Q4_K 0.49%; down IQ3_XXS 1.46-2.88% (2.41%), IQ3_S 1.44%, IQ4_XS
  0.38-0.54% (0.47%). Moving bits between layers has little to gain from the weights themselves.
- **What reaches the MoE's output** (the reference model on 512 tokens of held-out text, this repository's docs; the routed
  experts' combined output recomputed with the set's gate/up, its down, or all three): layer 2 / 12 / 22 / 32 / 41 -
  gate+up 0.914 / 2.258 / 1.690 / 2.224 / 1.496%, down 0.214 / 0.316 / 0.205 / 0.238 / 0.132%, all three the sum. Gate and
  up carry 81-92% of the error.
- **IQ3_S for all three** (10.31 bits a value triple, the set 10.375): measured directly, not projected - the candidate's
  experts quantized by the converter's own code (`--fakequant`, bit for bit the set's quantization when given the set's
  recipe), the same tokens: layer 12 2.844% against the set's 2.568% (+10.7%), layer 22 2.121% / 1.887% (+12.4%), layer 41
  1.726% / 1.618% (+6.7%). Its IQ3_S gate/up alone 1.471 / 1.105 / 0.968% (the set's IQ3_XXS 2.258 / 1.690 / 1.496), its IQ3_S
  down alone 1.032 / 0.718 / 0.487% (the set's IQ4_XS 0.316 / 0.205 / 0.132). A projection from the weight errors had it
  13-23% better; it is worse at every layer measured. Spending fewer bits on gate/up instead (IQ2_S, 6.1% at the weights)
  would double the larger share. At the set's size its recipe stays.
- **A bigger calibration set** (the engine's importance matrix, `strata-x --imatrix`, after this night's fix for a latent MoE:
  over the reference's own 16 chunks it matched the reference's matrix - gate/up cosine 0.9993-1.0000, down 0.993-0.998;
  then 64 chunks of the same calibration files, none of the KLD yardstick's held-out ones, merged with the reference's 16 by
  `imx_tools.py`: `ernie55-plain80.imx`, 137.6M routed rows, 21,439 experts seen against 21,327): down's column importance
  moved 35-60% (relative L2) from the 16-chunk matrix, the quantization error at the MoE's output 0.1 / 0.3 / 2.0% at layers 12
  / 22 / 41 (the set's recipe, `recipe_check.py --no-bit-check`). The importance matrix's size is not a lever here.
- **IQ3_S gate/up with IQ4_XS down** (11.125 bits a triple, +7.2%; experts 125.45 GB against 117.0, the set 143.5 GB),
  measured as one recipe: layer 12 1.771% against the set's 2.568% (-31.1%), layer 22 1.301% / 1.887% (-31.1%), layer 41
  1.097% / 1.618% (-32.2%). The quality step this box can take, for ~8.4 GB more RAM or reads.
- **The set built** (`convert_gguf_ernie55 --types gu=iq3_s --types d=iq4_xs --imatrix ernie55-plain80.imx --reuse <the
  IQ3_XXS set>`: the 18.09 GB of dense tensors copied, the experts made; 5,533 s; every tensor re-read the same):
  `~/AI/Models/ernie55-gguf/ERNIE-5.5-Flash-IQ3S-IQ4XS-0000N-of-00003.gguf`, 143.52 GB. On the KLD yardstick (6 x 4,096,
  positions 2,048-4,094; the live binary and run config, the same night): **mean KLD 0.0389 +- 0.0013** (median 0.0039,
  99th percentile 0.65), the same top token at **95.07%**, perplexity 1.75% above the reference's - the IQ3_XXS set measured
  again beside it 0.0629 +- 0.0020 (median 0.0068, 99th 1.00), 93.20%, 2.93%. Its load at the same `--resident-headroom-gib
  20`: 4,569 experts in VRAM against 4,899 (each 7.2% larger), the RAM copy 94.69 GiB against 92.88 (the complement 91.99
  GiB, all of it; 0.00 GiB read from the GGUF), the lend region 401 experts against 1,634 - slots lent to the caches in a
  long conversation re-read more of their experts from the file. At depth (the loop's 49,583 tokens, its rows against the
  FP8 reference's): KL 0.0276 against the IQ3_XXS set's 0.0668 over positions 24.8K-49.6K, the planning 0.0365 / 0.0858,
  the ~4K before the cycle 0.0298 / 0.0813, the same top token in the planning 96.3% / 94.9%; in the cycle it continues as
  FP8 does (0.977, the close at the floor). Decode (`--decode-runs w1,w1,w1`, 512 greedy tokens after a warm-up of 256, the
  same prompt, back to back): 31.07 / 31.51 / 31.68 T/s against 33.19 / 32.79 / 32.28 (-4.1%), the CPU pool 10.8-11.8 ms a
  token against 10.3-11.0. The server's end-to-end checks (moex/tools/ernie55/server_checks.py, the public checkout's
  server): 8 of 8 PASS. Since 2026-11-27 both local run configs name it (`strata-ernie55.json.bak-iq3xxs-20261127` beside
  each is the one before). Since 2026-11-27 evening the template and the README's step 3 name it too. The Hugging Face
  repo carries both sets, one folder each, first named `IQ3_S-IQ4_XS/` and `IQ3_XXS-IQ4_XS/` (the IQ3_XXS files were
  moved there server-side, their hashes unchanged). Hugging Face labels a GGUF file by the last quantization name in its
  path (huggingface.js `parseGGUFQuantLabel`), so its GGUF panel showed the two sets as one IQ4_XS entry of 279 GB. On
  2026-11-27 the folders were renamed server-side to `IQ4_XS-IQ3_S/` and `IQ4_XS-IQ3_XXS/`, the down type first, the
  hashes again unchanged.
- **Which layers the output feels** (`ref/layer_sensitivity.py`: the reference model on two held-out chunks of 2,048 tokens -
  this repository's docs and wiki text, 2,046 positions scored - in 16 variants run at once: exact, one group of three MoE
  layers at a time with the IQ3_S / IQ4_XS set's experts, all layers; 6,193 s): mean KL from the exact model, by group -
  1-3 0.00325, 4-6 0.00415, 7-9 0.00358, 10-12 0.00520, 13-15 0.00617, 16-18 0.00359, 19-21 0.00216, 22-24 0.00201,
  25-27 0.00111, 28-30 0.00241, 31-33 0.00197, 34-36 0.00233, 37-39 0.00123, 40-42 0.00148; all layers at once 0.02311 (the
  groups' sum 0.04064: the costs do not add, the whole is 57% of the parts). The first 18 layers cost 2.4 times what the
  last 24 do, a layer for a layer (13-15 against 25-27: 5.6 times) - an uneven model, unlike the weights' own errors.
- **A reallocation at the same size** (`ref/alloc_check.py`, the same two chunks, the exact model, the set and the candidate in
  one pass; 5,947 s): gate/up IQ4_XS on layers 10-15 (the most sensitive groups) paid for by IQ3_XXS gate/up on the 13 least
  sensitive layers (22, 25-27, 31-33, 37-42; their experts from the IQ3_XXS set): mean KL 0.02712 against the uniform set's
  0.02313 (+17%), the same top token 94.72% / 94.62%. Predicted from the groups' numbers -7%: as with IQ3_S everywhere, a
  projection was wrong, and the measurement stands.
- **Its two halves apart** (the same pass, with checkpoints every 4 layers - two power outages that day; 2,046 positions):
  the uniform set 0.02313 (94.62%); layers 10-15 at IQ4_XS gate/up alone, +2.6 GB, 0.02144 (-7.3%, 94.92%); the 13
  least sensitive at IQ3_XXS gate/up alone, -2.6 GB, 0.02909 (+25.8%, 94.33%). For the same bytes the cheapest downgrade
  costs ~3.5 times what the best-placed upgrade gains - the most favourable pair the sensitivity profile allows - so no
  reallocation of gate/up bits between layers at the set's size beats the uniform set; the two halves add to about the
  combined +17%. The upgrade alone is a size step, not a free one: with experts of two sizes the engine sizes every
  expert slot for the larger (6.68 MB against 5.83), ~13% fewer experts in VRAM, and the RAM copy at the 20 GiB headroom no
  longer holds every expert outside VRAM (estimated from the slot rule and the IQ3_S / IQ4_XS set's load, not built). The
  IQ3_S / IQ4_XS set stays the run configs' set.
- **The upgrade built and measured** (2026-11-27 afternoon; the IQ3_XXS set deleted for the room, at the user's word):
  `ERNIE-5.5-Flash-IQ3S-IQ4XS-L10-15-IQ4XS-0000N-of-00004.gguf`, 146.13 GB (`--types gu=iq3_s --types d=iq4_xs --types
  gu=iq4_xs:10-15 --reuse <the IQ3_S / IQ4_XS set>`: 12 tensors made, 1,504 copied, 1,012 s). On the KLD yardstick 0.0370 +-
  0.0012 against 0.0389 (-4.7%), the same top token 95.33% / 95.07%. Its load at the same headroom: 3,951 experts in VRAM
  against 4,532 (slots sized for its 6.68 MB experts), the complement 97.35 GiB does not fit the RAM copy (95.07 GiB) -
  2.5-2.9 GiB read from the GGUF when needed, no lend region. Decode, back to back (512 greedy tokens x 3 after a warm-up):
  28.79 / 29.28 / 29.86 T/s against 32.04 / 31.68 / 31.47 (-7.6%), the CPU pool 12.8-13.8 ms a token against 10.8-11.5
  (the slot cap alone, simulated on the IQ3_S / IQ4_XS set with STRATA_X_STAGE_SLOTS, had cost 2.2%: the larger experts and
  the spill add the rest). 5% nearer the reference for 7.6% of decode and the RAM's margin: not better on this box. The
  IQ3_S / IQ4_XS set stays the run configs' set; the upgrade's files are kept for the user's choice.
