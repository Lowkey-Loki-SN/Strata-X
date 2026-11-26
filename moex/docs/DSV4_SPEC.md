# DeepSeek-V4-Flash forward pass, as implemented in llama.cpp (operator-level spec)

Source of truth: `~/AI/llama.cpp-glm5next-mtp` at `c92b4ae77` ("Merge branch 'pr27773' into integrate").
The oracle binary in `build/` was compiled from this tree (`-DLLAMA_COMMIT="c92b4ae77"`). It uses GCC 13.3,
`-O3 -march=native`, `GGML_NATIVE=ON`, `GGML_LLAMAFILE=ON`, `GGML_CPU_REPACK=ON`, `GGML_CUDA=ON`, and `-std=gnu11` / `-std=gnu++17`.
The host is a Xeon Gold 6148, so the build has AVX-512F/DQ/BW/VL, FMA and F16C, but no AVX512-BF16 and no VNNI.

GGUF: `~/AI/Models/GGUFs/dsv4-vision/UD-IQ3_S/DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_S-0000{1..4}-of-00004.gguf`.
I read the headers with a plain sequential header parser and never touched tensor data. The file has 1371 tensors.

## 0. Conventions and citation keys

Shapes use ggml order: `[ne0, ne1, ne2, ne3]`. `ne0` is the contiguous dimension.
For a weight stored as `[in, out]`, `ggml_mul_mat(W, x)` computes `y[o] = Σ_i W[i,o]·x[i]`.

| Symbol | Value | Meaning |
|---|---|---|
| T | — | Tokens in the ubatch |
| t | 0..T-1 | Token index |
| p_t | — | Position of token t |
| D | 4096 | n_embd |
| HC | 4 | Hyper-connection streams |
| H | 64 | Query heads |
| Dh | 512 | Head dim; K = V = one shared KV head |
| R | 64 | Rope dims |
| NOPE | 448 | Dh − R |
| Di | 128 | Indexer head dim |
| Hi | 64 | Indexer heads |
| E | 256 | Routed experts |
| k | 6 | Experts used per token |
| F | 2048 | Expert FFN dim |

Citation keys. All paths are relative to the tree root unless absolute.

| key | file |
|---|---|
| D4 | `src/models/deepseek4.cpp` |
| KVD | `src/llama-kv-cache-dsv4.cpp` (H: `.h`) |
| KV | `src/llama-kv-cache.cpp` |
| ISWA | `src/llama-kv-cache-iswa.cpp` |
| G / GH | `src/llama-graph.cpp` / `.h` |
| CTX | `src/llama-context.cpp` |
| MDL | `src/llama-model.cpp` |
| ARCH | `src/llama-arch.cpp` |
| HP / HPC | `src/llama-hparams.h` / `.cpp` |
| IMPL | `src/llama-impl.h` |
| GGML / GGMLH | `ggml/src/ggml.c` / `ggml/include/ggml.h` |
| OPS | `ggml/src/ggml-cpu/ops.cpp` |
| CPU | `ggml/src/ggml-cpu/ggml-cpu.c` |
| VEC / VECH | `ggml/src/ggml-cpu/vec.cpp` / `vec.h` |
| UOPS | `ggml/src/ggml-cpu/unary-ops.cpp` |
| X86Q | `ggml/src/ggml-cpu/arch/x86/quants.c` |
| QNT | `ggml/src/ggml-quants.c` |
| CONV | `conversion/deepseek.py` (converter, context only) |
| BATCH | `src/llama-batch.cpp` |
| IQP / REPACK / SGEMM | `ggml/src/ggml-cpu/iqp.cpp` / `ggml/src/ggml-cpu/repack.cpp` / `ggml/src/ggml-cpu/llamafile/sgemm.cpp` |

---

## 1. Hyperparameters

### 1.1 Keys read by the generic loader (`MDL:1200-1396`)

| GGUF key | value in this GGUF | field | meaning / default if absent |
|---|---|---|---|
| `general.architecture` | `deepseek4` | arch | — |
| `deepseek4.context_length` | 1048576 | n_ctx_train | required; also used as `-c 0` |
| `deepseek4.embedding_length` | 4096 | n_embd | required |
| `deepseek4.embedding_length_out` | 16384 | n_embd_out_impl | optional; **overwritten** to `hc*n_embd`=16384 at D4:53. Only the MTP/embedding paths use it |
| `deepseek4.block_count` | 43 | n_layer_all | required |
| `deepseek4.nextn_predict_layers` | absent | n_layer_nextn | default 0. This GGUF has no MTP block; D4:20-26 would also zero it if `blk.N.nextn.eh_proj` were missing |
| `deepseek4.expert_count` | 256 | n_expert | default 0 |
| `deepseek4.expert_used_count` | 6 | n_expert_used_arr[*] | default 0 |
| `deepseek4.expert_group_count` / `expert_group_used_count` | absent | 0 / 0 | no grouped routing (G:2028 skipped) |
| `deepseek4.feed_forward_length` | absent | n_ff_arr = 0 | unused |
| `deepseek4.attention.head_count` | 64 | n_head_arr | — |
| `deepseek4.attention.head_count_kv` | 1 | n_head_kv_arr | default = head_count |
| `deepseek4.rope.scaling.finetuned` | absent | false | unused |
| `deepseek4.rope.scaling.original_context_length` | 65536 | n_ctx_orig_yarn | default n_ctx_train |
| `deepseek4.rope.freq_base` | 10000.0 | rope_freq_base_train | default 10000. Used **only** by ratio-0 layers |
| `deepseek4.rope.scaling.type` | `yarn` | rope_scaling_type_train | default "linear" |
| `deepseek4.rope.scaling.factor` | 16.0 | rope_freq_scale_train = 1/16 | default 1 (factor 0 also gives 1) |
| `deepseek4.rope.scaling.attn_factor` | absent | rope_attn_factor = 1 | DSV4 never uses it (D4 builds its own attn_factor) |
| `deepseek4.rope.scaling.yarn_beta_fast` / `_slow` | 32.0 / 1.0 | **not read** | the hparams defaults 32 / 1 apply (HP:153-154). They match the file |
| `deepseek4.rope.scaling.yarn_log_multiplier` | absent, not read | 0 | — |
| `deepseek4.attention.key_length` | 512 | n_embd_head_k_full | default n_embd/n_head = 64 |
| `deepseek4.attention.value_length` | 512 | n_embd_head_v_full | default 64 |
| `deepseek4.rope.dimension_count` | 64 | n_rot_full | default key_length |
| `*.key_length_swa` / `*.value_length_swa` / `*.rope.dimension_count_swa` | absent | copy of full | all trunk layers are "SWA", so the code reads these copies (HPC:101-145) |
| `tokenizer.ggml.tokens` | 129280 entries | n_vocab | bos=0, eos=1, pad=2 |

### 1.2 Keys read by `llama_model_deepseek4::load_arch_hparams` (D4:19-80)

All are required unless noted. Key strings are at ARCH:191-313.

| GGUF key | value | field | meaning |
|---|---|---|---|
| `deepseek4.attention.layer_norm_rms_epsilon` | 9.99999968e-21 (f32(1e-20)) | f_norm_rms_eps | eps for **every** RMSNorm, including weightless ones: hc mixes, q heads, compressor, head. CONV:1072 confirms the LLM config value is 1e-20 |
| `deepseek4.attention.q_lora_rank` | 1024 | n_lora_q | q low-rank dim |
| `deepseek4.attention.sliding_window` | 128 | n_swa | raw window |
| `deepseek4.expert_feed_forward_length` | 2048 (scalar, broadcast to all layers) | n_ff_exp_arr | expert / shared-expert hidden dim |
| `deepseek4.expert_shared_count` | 1 | n_expert_shared | shared FFN width = 2048·1 |
| `deepseek4.expert_weights_scale` | 1.5 | expert_weights_scale | routed weight multiplier |
| `deepseek4.expert_weights_norm` | true | expert_weights_norm | normalise the top-k weights |
| `deepseek4.swiglu_clamp_exp` | [10.0]×43 | swiglu_clamp_exp[il] | routed-expert SwiGLU limit |
| `deepseek4.swiglu_clamp_shexp` | [10.0]×43 | swiglu_clamp_shexp[il] | optional; **default = swiglu_clamp_exp** (D4:37-39) |
| `deepseek4.attention.indexer.head_count` | 64 | indexer_n_head | — |
| `deepseek4.attention.indexer.key_length` | 128 | indexer_head_size | — |
| `deepseek4.attention.indexer.top_k` | 512 | indexer_top_k | — |
| `deepseek4.attention.output_group_count` | 8 | dsv4_o_group_count | — |
| `deepseek4.attention.output_lora_rank` | 1024 | dsv4_o_lora_rank | per-group rank |
| `deepseek4.attention.compress_rope_freq_base` | 160000.0 | dsv4_compress_rope_base | rope base of compressed layers. Not affected by `--rope-freq-base` |
| `deepseek4.hyper_connection.count` | 4 | dsv4_hc_mult | HC streams (asserted ==4 at D4:370) |
| `deepseek4.hyper_connection.sinkhorn_iterations` | 20 | dsv4_hc_sinkhorn_iters | — |
| `deepseek4.hyper_connection.epsilon` | 9.99999997e-07 | dsv4_hc_eps | — |
| `deepseek4.hash_layer_count` | 3 | dsv4_hash_layer_count | layers 0,1,2 are hash-routed |
| `deepseek4.attention.compress_ratios` | i32[46] = `[0,0,4,128,4,128,…,4,128,4,0,0,0]` | dsv4_compress_ratios | must have ≥ block_count entries (D4:55-61). Entries 43-45 are unused |
| `deepseek4.expert_gating_func` | 4 | expert_gating_func | must be 4 = `SQRT_SOFTPLUS` (HP:21), else throw (D4:63-66) |

Forced or derived settings (D4:67-73):
- `swa_type = STANDARD`.
- `set_swa_pattern(0)` makes **every** trunk layer "SWA" (HPC:8-22).
- `swa_full_non_causal = true`: image-span tokens see the whole span (HP:167-169).
- `n_embd_out_impl = 16384`.
- `rope_type = LLAMA_ROPE_TYPE_NORM = 0`, i.e. GGML_ROPE_TYPE_NORMAL with interleaved pairs (MDL:2909).

Layer kinds in this GGUF:

| ratio | layers | name in code |
|---|---|---|
| 0 | 0, 1 | raw (SWA-only) |
| 4 | 2, 4, 6, …, 42 (21 layers) | CSA (+ lightning indexer) |
| 128 | 3, 5, …, 41 (20 layers) | HCA |

Hash routing applies to layers 0, 1 and 2. Layer 2 is both CSA and hash.

### 1.3 Context-level parameters used by the graph (`CTX:113-213`, `G:1452-1495`)

| param | value (defaults) | source |
|---|---|---|
| freq_base | 10000 | cparams.rope_freq_base ← hparams |
| freq_scale | 1/16 = 0.0625 | rope_freq_scale_train |
| ext_factor | 1.0 | `yarn_ext_factor` < 0 → 1.0 because scaling type is YARN (CTX:172-174) |
| beta_fast / beta_slow | 32 / 1 | hparams defaults |
| n_ctx_orig | 65536 | n_ctx_orig_yarn |
| attn_factor (cparams) | computed at CTX:176-213 | **unused by DSV4**. D4 passes `dsv4_rope_attn_factor()` (D4:11-17) |
| norm_rms_eps | 1e-20 | — |
| n_ctx | padded to a multiple of 256 (CTX:288) | — |

---

## 2. Tensor contract

The table shows GGUF `ne` order, the type in **this** UD-IQ3_S file, and the expected shape in `load_arch_tensors` (D4:82-183). Names are at ARCH:463-688.

### 2.1 Global tensors

| name | GGUF ne | type | role |
|---|---|---|---|
| `token_embd.weight` | [4096, 129280] | Q6_K | embedding rows (`get_rows` → dequantised F32) |
| `output_hc_fn.weight` | [16384, 4] | F32 | head HC mixer (hc_head_fn) |
| `output_hc_base.weight` | [4] | F32 | head HC bias |
| `output_hc_scale.weight` | [1] | F32 | head HC scale |
| `output_norm.weight` | [4096] | F32 | final RMSNorm weight |
| `output.weight` | [4096, 129280] | Q6_K | LM head (not tied) |

### 2.2 Per-layer tensors present in every layer

| name (blk.N.) | GGUF ne | type | role |
|---|---|---|---|
| `attn_norm.weight` | [4096] | F32 | pre-attention RMSNorm weight |
| `attn_q_a.weight` | [4096, 1024] | Q6_K (Q8_0 in L26) | wq_a |
| `attn_q_a_norm.weight` | [1024] | F32 | q_a RMSNorm weight |
| `attn_q_b.weight` | [1024, 32768] | Q8_0 | wq_b → 64 heads × 512 |
| `attn_kv.weight` | [4096, 512] | Q8_0 | wkv: the single shared K=V head |
| `attn_kv_a_norm.weight` | [512] | F32 | kv RMSNorm weight (LLM_TENSOR_ATTN_KV_NORM, ARCH:520) |
| `attn_sinks.weight` | [64] | F32 | per-head sink logits |
| `attn_output_a.weight` | [4096, 8192] | Q8_0 | wo_a, **reshaped at load** to [4096, 1024, 8] (`TENSOR_ALLOW_RESHAPE`, D4:120-122). Group g uses file rows 1024g..1024g+1023 |
| `attn_output_b.weight` | [8192, 4096] | Q8_0 | wo_b |
| `hc_attn_fn.weight` | [16384, 24] | F32 | attention-side HC mixer |
| `hc_attn_base.weight` | [24] | F32 | — |
| `hc_attn_scale.weight` | [3] | F32 | — |
| `hc_ffn_fn.weight` | [16384, 24] | F32 | FFN-side HC mixer |
| `hc_ffn_base.weight` | [24] | F32 | — |
| `hc_ffn_scale.weight` | [3] | F32 | — |
| `ffn_norm.weight` | [4096] | F32 | — |
| `ffn_gate_inp.weight` | [4096, 256] | BF16 | router |
| `exp_probs_b_vl.bias` | [256] | F32 | optional (D4:163). Routing bias for **image (embd) ubatches only**. Present in all 43 layers here |
| `ffn_gate_exps.weight` | [4096, 2048, 256] | IQ3_XXS (MXFP4 in L26) | w1 |
| `ffn_up_exps.weight` | [4096, 2048, 256] | IQ3_XXS (MXFP4 in L26) | w3 |
| `ffn_down_exps.weight` | [2048, 4096, 256] | IQ3_XXS (MXFP4 in L26) | w2 |
| `ffn_gate_shexp.weight` | [4096, 2048] | Q6_K (Q8_0 in L26) | shared w1 |
| `ffn_up_shexp.weight` | [4096, 2048] | Q6_K (Q8_0 in L26) | shared w3 |
| `ffn_down_shexp.weight` | [2048, 4096] | Q6_K (Q8_0 in L26) | shared w2 |

### 2.3 Routing tensors that depend on the layer

| layers | tensor | GGUF ne | type | role |
|---|---|---|---|---|
| 0, 1, 2 (`il < hash_layer_count`) | `ffn_gate_tid2eid.weight` | [6, 129280] | I32 | token → 6 expert ids (D4:157-158) |
| 3..42 | `exp_probs_b.bias` | [256] | F32 | selection-only bias (D4:160). CONV:884-886 drops `gate.bias` for hash layers |

### 2.4 Extra tensors for ratio-4 (CSA) layers (D4:132-150)

`coff = 2` for ratio 4.

| name | GGUF ne | type | role |
|---|---|---|---|
| `attn_compressor_kv.weight` | [4096, 1024] | Q8_0 | compressor value proj (2×512: "prev" half then "cur" half) |
| `attn_compressor_gate.weight` | [4096, 1024] | Q8_0 | compressor score proj |
| `attn_compressor_ape.weight` | [1024, 4] | F32 | 4 rows (pos mod 4) × 1024 |
| `attn_compressor_norm.weight` | [512] | F32 | — |
| `indexer.attn_q_b.weight` | [1024, 8192] | Q8_0 | indexer q from qr → 64×128 |
| `indexer.proj.weight` | [4096, 64] | F32 | per-head indexer weights from x |
| `indexer_compressor_kv.weight` | [4096, 256] | Q8_0 | 2×128 |
| `indexer_compressor_gate.weight` | [4096, 256] | Q8_0 | — |
| `indexer_compressor_ape.weight` | [256, 4] | F32 | — |
| `indexer_compressor_norm.weight` | [128] | F32 | — |

### 2.5 Extra tensors for ratio-128 (HCA) layers

`coff = 1`.

| name | GGUF ne | type |
|---|---|---|
| `attn_compressor_kv.weight` | [4096, 512] | Q8_0 |
| `attn_compressor_gate.weight` | [4096, 512] | Q8_0 |
| `attn_compressor_ape.weight` | [512, 128] | F32 (row = pos mod 128) |
| `attn_compressor_norm.weight` | [512] | F32 |

Ratio-0 layers (0, 1) have no compressor or indexer tensors. Tensor count check: 6 global + 25×2 (L0, L1) + 35 (L2) + 35×20 (CSA) + 29×20 (HCA) = 1371.

---

## 3. Forward pass of one layer

The layer input is the HC state `X ∈ [4096, 4, T]` (F32). Stream s of token t is `X[:, s, t]`. The flattened view `[16384, T]` has element `(i, s) ↦ i + 4096·s`.

### 3.0 Exact primitive definitions (CPU semantics)

**RMS(v, w=None)** (OPS:3924-3987):
```
ss   = Σ_i (double)( (float)(v_i*v_i) )      // square in float, sum in double, sequential
mean = (float)(ss / n)
sc   = 1.0f / sqrtf(mean + eps)              // eps = 1e-20 (f32)
out_i = (v_i * sc) [* w_i]                   // two float multiplies, left to right
```
`build_norm(..., LLM_NORM_RMS)` = `ggml_rms_norm`, then `ggml_mul(w)` (G:1580-1613). The CPU may fuse RMS_NORM+MUL. The result is bit-identical (`x*scale*w`), but the intermediate `norm-<il>` tensor is then not separately materialised (CPU:3064-3090; disable with `GGML_CPU_DISABLE_FUSION=1`).

**ROPE(v[Dv], p, P, offs, inverse)**: `ggml_rope_ext` + `ggml_rope_set_offset`. Mode NORMAL, `n_dims = 64`. Only dims `[offs, offs+64)` rotate; all other dims are copied (GGMLH:2003-2011, OPS:6082-6238).
```
ts     = powf(base, -2.0f/64)                         // OPS:6143
θe_0   = (float)p ; θe_{k+1} = θe_k * ts              // iterative float products, k = pair index (OPS:5975-5989)
if ext != 0:                                          // rope_yarn, OPS:5958-5973
   ramp_k = 1 - min(1, max(0, (k - c0) / max(0.001f, c1 - c0)))   // k = i0/2 (integer), OPS:5951-5954
   mix    = ramp_k * ext
   θ_k    = (fs*θe_k)*(1-mix) + θe_k*mix
   msc    = attn_factor * (1.0f + 0.1f*logf(1.0f/fs))
else:
   θ_k = fs*θe_k ;  msc = attn_factor
cos_k = cosf(θ_k)*msc ;  sin_k = sinf(θ_k)*msc * (inverse ? -1 : +1)   // OPS:6171, compiled as sincosf
for k in 0..31:  a = v[offs+2k], b = v[offs+2k+1]                     // NORMAL = interleaved pairs, OPS:6206-6208
   v'[offs+2k]   = a*cos_k - b*sin_k
   v'[offs+2k+1] = a*sin_k + b*cos_k
```
`c0, c1` come from `ggml_rope_yarn_corr_dims` (GGML:4417-4429):
```
corr(nr)   = 64*logf(n_ctx_orig/(nr*2π)) / (2*logf(base))
[c0, c1]   = [max(0, floor(corr(β_fast))), min(63, ceil(corr(β_slow)))]
```
`ggml_rope_ext_back` is the same op with `inverse = true`, i.e. sin sign flipped and the same `msc` (GGML:4433-4447, OPS:6264-6285).

Parameter sets used by DSV4 (D4:911-918):

| set | used for | base | fs | ext | attn_factor | n_ctx_orig, β | effect |
|---|---|---|---|---|---|---|---|
| **P0** | q and raw kv of ratio-0 layers | 10000 | 1 | 0 | 1 | 0 | plain RoPE, `ts = 0.74989426` |
| **PC** | ratio-4/128 layers: q, raw kv, compressed entries, indexer q (D4:517-519, 590-592, 622-624) | 160000 | 1/16 | 1 | `1/(1+0.1·logf(16)) = 0.7829266` (0x3f486de1, D4:11-17) | 65536, 32/1 | `c0 = 15`, `c1 = 25`, `ts = 0.68765604`. Effective `msc = 0.7829266 × 1.2772589 = 1.0f` exactly (checked in f32): YaRN changes frequencies only, never magnitudes |

The same set is used for the de-rope of the attention output (D4:1213-1215).

**HADAMARD_n(v)**: `llama_mul_mat_hadamard` (IMPL:57-75). It is `ggml_mul_mat(H, v)` with hint `SRC0_IS_HADAMARD`, and on the CPU it runs as an FWHT (CPU:1262-1266, OPS:11990-12064):
```
v_i *= 1.0f/sqrtf(n)                  // first
for len = 1,2,4,…,n/2: for each pair (u = v[i+j], w = v[i+len+j]) → (u+w, u-w)   // f32, natural (Sylvester) order
```
This equals `H·v` with `H[i][j] = (-1)^popcount(i&j)/√n`. That matrix is the one `ggml_gen_hadamard` builds (KV:21-59). It is used per 128-dim row of the indexer q and indexer keys. For the raw/CSA/HCA K it is used only with a quantised `-ctk`; see §6.

**SOFTMAX_ROW(x, mask, scale, sink)**: CPU `soft_max` (OPS:5584-5694, VEC:531-596).
```
w_i = x_i*scale + mask_i ;  M = max_i w_i ; if sink: M = max(M, sink)
e_i = exp(w_i - M) ; S = Σ (double) e_i ; if sink: S += (double)expf(sink - M)
y_i = e_i * (float)(1.0/S)
```
`exp` = `ggml_v_expf` (AVX-512, 16-wide blocks; VECH:1172) for `i+15 < n`, and `expf` for the tail. **n = 8 (CSA compressor) is therefore all scalar `expf`.** Each 16-block's `_mm512_reduce_add_ps` is added to S in double. The sink logit is **not** scaled by `scale`.

### 3.1 mHC "pre" (attention side)

Code: D4:1258-1263, D4:356-413; fused CPU kernels at OPS:11110-11292.

1. `nrm_t = RMS(flat(X[:,:,t]))` over 16384 values, no weight. Then `mix_t = hc_attn_fn · nrm_t`, a `[24, T]` F32 matmul. **name `hc_mixes-l`**
2. `pre_t[s] = σ( (mix[s]*scale[0]) + base[s] ) + eps`, s = 0..3. The mul and add are separate ggml ops, each rounded. σ = `1/(1+expf(-x))` (UOPS:31). `+eps` uses `ggml_scale_bias(1, eps)`. **`hc_pre-l` [4, T]**
3. `post_t[s] = 2·σ( (mix[4+s]*scale[1]) + base[4+s] )`. **`hc_post-l` [4, T]**
4. `comb_t` (fused `ggml_dsv4_hc_comb`, default ON; GGMLH:2631-2650, OPS:11142-11227). Index convention: `comb[dst, src]`, ne0 = dst, flat index `8 + dst + 4·src`.
   ```
   L[d,s] = fmaf(mix[8+d+4s], scale[2], base[8+d+4s])
   for each s: m = max_d L ; e_d = expf(L[d,s]-m) ; Z = Σ_d e_d (float, d=0..3) ; C[d,s] = fmaf(e_d, 1/Z, eps)   // softmax over dst, +eps
   col(): for each d: Z = eps + C[d,0] + C[d,1] + C[d,2] + C[d,3] ; C[d,*] *= 1/Z                       // normalise over src
   row(): for each s: Z = eps + Σ_d C[d,s] ;                    C[*,s] *= 1/Z                       // normalise over dst
   col(); repeat 19×: { row(); col(); }                                                               // sinkhorn_iters=20
   ```
   **`hc_comb-l` [4, 4, T]**. The unfused fallback (D4:318-354) computes the same thing with `ggml_soft_max` / `sum_rows` / `div` (rounded differently).
5. `y_t[i] = Σ_{s=0..3} X[i,s,t]·pre_t[s]` (fused `ggml_dsv4_hc_pre`, compiled as an FMA chain starting from 0). **`hc_attn_pre-l` [4096, T]**

### 3.2 attn_norm

`a_t = RMS(y_t, attn_norm.weight)`. **`norm-l`, then `attn_norm-l` [4096, T]** (D4:1265-1266).

### 3.3 Query path

Code: D4:920-934.

1. `qa = attn_q_a · a` (4096→1024). **`qr-l`**
2. `qr = RMS(qa, attn_q_a_norm.weight)`. **`norm-l`, `qr_norm-l`**. This `qr` is also the indexer input.
3. `q = attn_q_b · qr` (1024→32768), reshaped to `[512, 64, T]`. Then a per-head `RMS` (no weight, eps 1e-20). **`q_norm-l`**
4. `q = ROPE(q, p_t, P_l, offs=448)`, where `P_l = P0` if ratio==0 else `PC`. **`q-l` [512, 64, T]**

### 3.4 Raw K=V path

Code: D4:936-944.

1. `kv = RMS(attn_kv · a, attn_kv_a_norm.weight)` (4096→512). **`norm-l`, `kv_norm-l` [512, 1, T]**
2. `kv = ROPE(kv, p_t, P_l, offs=448)`. **`kv-l`**. This vector is both the key and the value for its own position.
3. Store: `set_rows` into the raw SWA cache, F32→**F16 (RNE)**, at ring cells `k_idxs`. This happens after the optional Hadamard; see §6. Code: D4:744/800/854, KVD:1881-1908, KV:1324-1357.

### 3.5 KV compressors (ratio 4: CSA + LID; ratio 128: HCA)

Code: D4:946-1187 (graph), KVD:427-737 (per-ubatch plan), D4:479-597 (math).

The "x" input is the **attn_norm output `a`** (same tensor as the attention input).

Per compressor:

| | r | state ring S | coff | d | ape | norm weight |
|---|---|---|---|---|---|---|
| CSA | 4 | 8 | 2 | 512 | `attn_compressor_ape` [1024, 4] | `attn_compressor_norm` |
| LID (indexer keys) | 4 | 8 | 2 | 128 | `indexer_compressor_ape` [256, 4] | `indexer_compressor_norm` |
| HCA | 128 | 128 | 1 | 512 | `attn_compressor_ape` [512, 128] | `attn_compressor_norm` |

Constructors: KVD:1313-1329.

**Step 1: per-token projections (every token).**
```
Akv_t = W_ckv · a_t            ∈ R^{coff·d}                          name <c>_state_kv-l
Gs_t  = W_cgate · a_t          ∈ R^{coff·d}                          name <c>_state_score-l
Gs_t += ape[p_t mod r]  (get_rows on ape, row = p_t % r)             name <c>_state_score_ape-l
```
Here `<c>` is `csa`, `lid` or `hca` (D4:953-979, 1038-1048). The gate is **per channel**: one score per feature, not one per token.

**Step 2: source table** (KVD:473-488, D4:986-992, 547-548). A block may read rows from three places:
`[ persistent state ring (S rows/stream) | the T current-ubatch rows | 1 synthetic row ]`.
- Current ubatch: if position q is in this ubatch, use its fresh `(Akv, Gs)`.
- State ring: otherwise use row `q mod S`. It holds `(Akv, Gs+ape)` persisted by earlier ubatches. The state is F32 (KVD:964-966) and zero-initialised.
- Synthetic: if q < 0, use the appended row with `kv = 0`, `score = -inf` (`dsv4_append_zero_row`, D4:209-215). Only CSA/LID can reach q < 0 (the first block's previous half).

**Step 3: when a compressed entry is produced** (KVD:519-543).
- Entry b is produced at the token with `(p_t+1) % r == 0`. It is written to compressed-cache **row b = p_t / r**, i.e. a linear index with no ring.
- The rope position is the window start `s0 = p_t + 1 - r = b·r` (`state_write_pos`).
- Entry b covers positions `[b·r, b·r + r - 1]`. CSA/LID additionally look back at `[b·r - r, b·r - 1]`.
- A token at position p **sees entries `j < n_vis(p) = (p+1)/r`** (integer division; KVD:499-501). So the entry completed at p is visible to p itself.
- An incomplete window is never visible. It exists only in the F32 state ring and through the raw SWA window, which always covers it because r ≤ 128.

**Step 4a: HCA math** (no overlap; D4:479-524). For k in 0..511:
```
c[k] = Σ_{j=0..127} softmax_j( Gs_{s0+j}[k] ) · Akv_{s0+j}[k]
```
The graph does `get_rows` → reshape `[512, 128, nb]` → permute → `[128, 512, nb]` → `soft_max` (over the 128 positions, per channel) → `mul` → `sum_rows` (double, sequential j) → permute back to `[512, 1, nb]`.

**Step 4b: CSA/LID overlap math** (D4:526-597). Here `d` = 512 (CSA) or 128 (LID). The 8 slots per channel k in 0..d-1:
```
slot j=0..3 (prev window, FIRST half of features):  V_j = Akv_{s0-r+j}[k]   , G_j = Gs_{s0-r+j}[k]      (synthetic if s0-r+j<0)
slot j=4..7 (cur window, SECOND half of features):  V_j = Akv_{s0+j-4}[d+k] , G_j = Gs_{s0+j-4}[d+k]
c[k] = Σ_{j=0..7} softmax_j(G)·V_j
```
This matches the reference's `overlap_transform`. The ape row is the source token's own `pos mod 4`, added before pooling. Read-index layout: all blocks' prev indices first, then all blocks' cur indices (KVD:629-636).

**Step 5: finish** (D4:514-521, 587-594, 1005-1011, 1074-1080, 1134-1140).
```
e = RMS(c, norm_w)                         // eps 1e-20, over d
e = ROPE(e, s0, PC, offs = d-64)           // dims [448,512) for CSA/HCA, [64,128) for LID
LID only: e = HADAMARD_128(e)              // always (lid k_rot always exists: KV:327-332)
set_rows(cache, e) at row b, F32 -> F16 (RNE)
```

**Step 6: persist the state** (KVD:503-517, 643-651; D4:1027-1036, 1096-1105, 1177-1186). For every token, write `(Akv_t, Gs_t+ape)` into ring row `p_t mod S`. If several tokens in the ubatch map to the same row, the largest position wins. Ordering:
- Graph reads of the state happen **before** the persist writes in graph order, so compression sees the pre-ubatch ring.
- After the ubatch, the ring holds the last S positions.

**Dummy blocks** keep the graph shape fixed (KVD:547-627).
- HCA: if no HCA block completes in the ubatch, one dummy block is computed. It reads the first token's row ×128 and is written to row `kv_size-1` with rope pos 0. It is masked (asserted not visible).
- CSA/LID: padded to `ceil(n_seq_tokens/4)` writes per sequence. With the default single-sequence split (`split_simple`, `n_seq_tokens = 1`; BATCH:507,821) this means one dummy only when no block completes. Decode steps with `(p+1)%4 != 0` therefore produce a dummy CSA block and a dummy LID block, written to row `kv_size-1`.
- **In dumps, `*_state_compress*` tensors of such steps are garbage.** Ignore them. In a block step there are `nb = #completed blocks` rows (≥1).

**Other DSV4 compressor details:**
- Rollback snapshots and restore (`n_rs_seq`, default 0; KVD:653-716, D4:222-264) are no-ops when n_rs_seq = 0.
- Multi-sequence uses one stream per sequence; offsets come from `dsv4_stream_offset` (KVD:51-60).

### 3.6 Lightning indexer (ratio-4 layers only)

Code: D4:599-714. It runs inside `build_csa_lid_attention`, D4:731.

1. `iq = indexer.attn_q_b · qr` (1024→8192), reshaped to `[128, 64, T]`. **`lid_q-l`**. No normalisation of iq.
2. `iq = ROPE(iq, p_t, PC, offs = 64)`, i.e. dims [64, 128). **`lid_q_rope-l`**
3. `iq = HADAMARD_128(iq)` per head. **`lid_q_rot-l`**
4. `w_t = (indexer.proj · a_t) * (1.0f/sqrtf(8192))`. The weight is F32, so this is an F32 matmul. Scale `= 0.011048543 = 1/√(Di·Hi)`. **`lid_weights-l` [64, T]**
5. Keys: the LID cache rows `0..n_lid-1` (F16), already Hadamard-rotated at write time. **`lid_k-l` [128, 1, n_lid, 1]**
   - `n_lid = plan.n_kv = max(256, PAD256(max_t n_vis(p_t)))` (KVD:641).
6. Score. The default path is the fused `ggml_lightning_indexer` (OPS:12084-12164). `cparams.fused_lid = true` is not probed (CTX:236-237).
   ```
   score[j,t] = Σ_{h=0..63} max( iq[:,h,t] · f32(kI_j), 0 ) · w_t[h]  +  mask[j,t]
   mask[j,t]  = 0 if j < n_vis(p_t) else -inf          (F16 mask when fused; G:757-799, G:974)
   ```
   - Dot: `ggml_vec_dot_f32` over 128 values, with K converted F16→F32 and q in F32.
   - Heads accumulate sequentially as `fmaf(relu, w, score)`; the compiled code uses vfmadd231ss.
   - **`lid_score_masked-l` [n_lid, T, 1, 1]**
   - The unfused fallback (D4:658-678) uses `mul_mat` (q→F16), relu, mul, sum_rows, then `+mask`. Its tensors are named `lid_kq`, `lid_score`, `lid_score_masked`.
7. Top-k. `K = min(n_lid, 512)` (D4:680). `ggml_top_k` on the CPU (OPS:8550-8594):
   - Indices start as 0..n_lid-1.
   - `std::partial_sort(…, cmp: s[a] > s[b])`, keep the first K, then swap the first two entries.
   - Ties are broken by libstdc++'s heap-select and are not stable. Result order is irrelevant because it is used as a set.
   - **`lid_top_k-l` [K, T] I32**
8. Mask (D4:687-714):
   ```
   csa_mask[j,t] = (j ∈ topK_t ? 0 : -inf) + (j < n_vis(p_t) ? 0 : -inf)
   ```
   Implementation: `fill(-inf)` → `set_rows(zeros at topk)` → `add` kq_mask. **`csa_top_k_mask-l` [n_csa, T, 1, 1]**
   - Consequence: when `n_vis ≤ 512`, every visible block is selected, because masked rows score -inf and lose to every finite score. When `n_vis > 512`, the top 512 visible blocks are selected.

### 3.7 Attention

Code: D4:1189-1210. Dispatch: D4:769, D4:824, D4:860, G:2536-2672. Scale `= 1/sqrtf(512) = 0.044194173`.

Keys and values are the same tensor (K = V; build_attn_mha is called with `(q, k_all, k_all)`).

| layer | K/V set | mask |
|---|---|---|
| ratio 0 | raw ring view `[512, 1, n_raw]` (no separate name) | raw SWA mask. Output **`attn_raw`** |
| CSA | `concat(raw ring [n_raw], CSA rows [n_csa])` (**`csa_raw_k`, `csa_comp_k`, `csa_k_all`**) | `concat(raw_mask, csa_top_k_mask)` (**`csa_lid_kq_mask`**). Output **`attn_csa_lid`** |
| HCA | `concat(raw ring, HCA rows [n_hca])` (**`hca_raw_k`, `hca_comp_k`, `hca_k_all`**) | `concat(raw_mask, hca kq_mask)` (all `j < n_vis`; **`hca_kq_mask`**). Output **`attn_hca`** |

**Raw mask** (KV:1561-1709). Cell j is visible from token t iff:
- the cell is non-empty,
- it belongs to the same sequence,
- `p_j ≤ p_t` (causal), and
- `p_t − p_j < 128` (HP:462-466).

So it covers at most 128 raw keys, the current token included. For non-causal (image) ubatches, tokens of the current span ignore the window (KV:1690-1694).

Raw ring geometry:
- Ring size `n_raw = get_n_kv() = min(size_swa, max(256, PAD256(used_max_p1)))` (KV:1256-1270).
- `size_swa = PAD256(min(n_ctx, 128 + n_ubatch))`, which is 768 for `-ub 512` (ISWA:69-80).
- Compressed widths: `n_csa = n_hca = max(256, PAD256(max n_vis))` (KVD:641). Rows ≥ n_vis are masked.

**Math (all paths)** for token t, head h:
```
s_j   = (q_h · k_j) * 0.044194173 + m_j          (m_j ∈ {0, -inf})
M     = max(max_j s_j, sink_h)                   (sink = attn_sinks[h], unscaled)
P_j   = exp(s_j - M) / ( Σ_j exp(s_j - M) + exp(sink_h - M) )
o_h   = Σ_j P_j · k_j                            (k_j is the full 512-dim stored row, incl. rope dims)
```
CPU numerics differ by path; see §6.4.
- With `-fa off`: `kq` → `kq_soft_max` → `v_cont` → `kqv` (G:2605-2661). The output is the cont_2d `[32768, T]`.
- With FA: `ggml_flash_attn_ext` with sinks and `PREC_F32`. `n_kv_max` is set for CSA but **the CPU ignores it**; only CUDA uses it (G:2560-2603, GGML:5509-5516).
- With a quantised `-ctk`: `q`, `kv`, the compressed rows and the output are wrapped in HADAMARD_512 (D4:733-737, 770-772, 789-793, 825-827, 842-847, 862-863). This is mathematically transparent.

### 3.8 De-rope and grouped low-rank output

Code: D4:1212-1226.

1. `o = reshape([512, 64, T]); o = ROPE(o, p_t, P_l, offs=448, inverse)`. **`attn_derope-l`**
2. `o` is reshaped to `[4096 (=8 heads×512), 8 groups, T]` and permuted to `[4096, T, 8]`. Group g holds heads 8g..8g+7 concatenated head-major.
3. `oa[:, t, g] = wo_a[:, :, g] · o[:, t, g]` (4096→1024 per group, batched mul_mat). **`attn_wo_a-l` [1024, T, 8]**
4. Permute and cont to `[8192, T]` with index `r + 1024·g`. Then `out = attn_output_b · oa` (8192→4096). **`attn_out-l` [4096, T]**

### 3.9 mHC "post" (attention side)

Fused `ggml_dsv4_hc_post` (GGMLH:2663-2675, OPS:11313-11373):
```
X'[i,d,t] = attn_out[i,t]*post[d,t]  +  Σ_{s=0..3} X[i,s,t]*comb[d,s,t]
```
Compiled as `vmulss` then an FMA chain over s = 0..3. **`hc_attn_post-l` [4096, 4, T]** (D4:1270-1271).

### 3.10 mHC "pre" (FFN side) and ffn_norm

This repeats §3.1 on `X'` with `hc_ffn_fn/scale/base`. The names `hc_mixes-l`, `hc_pre-l`, `hc_post-l` and `hc_comb-l` appear a **second time**. Then:
- **`hc_ffn_pre-l`**
- `f = RMS(·, ffn_norm.weight)`, named **`norm-l`, `ffn_norm-l`**

Code: D4:1273-1286.

### 3.11 MoE

Code: D4:1288-1320, G:1938-2300.

1. Router logits. `ffn_gate_inp` is BF16, so on the CPU `f` is **rounded to BF16** (RNE) before the dot. `set_prec(F32)` has no CPU effect (G:1970-1973). **`ffn_moe_logits-l` [256, T]**
2. `probs = sqrtf(softplus(logits))`, with `softplus(x) = x > 20 ? x : logf(1+expf(x))` (UOPS:80-82, G:1998-2001). **`ffn_moe_probs-l`**
3. Expert selection (D4:1290-1301):

   | case | bias `b` used for selection | selection |
   |---|---|---|
   | Token ubatch, layer ≥ 3 | `exp_probs_b` | top-6 of biased scores |
   | Token ubatch, layer < 3 | none (`exp_probs_b = nullptr`) | **hash** |
   | **Embedding (image) ubatch, any layer** | `exp_probs_b_vl` (or `exp_probs_b` if the VL bias is absent; for hash layers this is nullptr, i.e. no bias) | top-6 of biased scores |

   - Hash layers ignore tid2eid for image tokens.
   - With a bias: `sel = probs + b`, named **`ffn_moe_probs_biased-l`**. Top-6 via `ggml_argsort_top_k` = `std::sort` descending (not stable, OPS:8489-8528) plus a view of the first 6 (GGML:5389-5403). Names **`ffn_moe_argsort-l` [256, T]** and **`ffn_moe_topk-l` [6, T]**.
   - Hash: `idx_t = tid2eid[:, token_t]`, the 6 ids in stored order. **`ffn_moe_topk-l`** (D4:1299).
4. Weights. These use the **unbiased** probs in both routing modes:
   ```
   wgt   = probs[idx]                             ffn_moe_weights-l           [1,6,T]
   S     = Σ wgt (sum_rows, double seq.)           ffn_moe_weights_sum-l       [1,T]
   S     = clamp(S, 6.103515625e-5, +inf)          ffn_moe_weights_sum_clamped-l
   wgt   = wgt / S                                 ffn_moe_weights_norm-l      [6,T]
   wgt  *= 1.5                                     ffn_moe_weights_scaled-l
   ```
   Code: G:2068-2097.
5. Experts, for each of the 6 selected e in selected order (G:2135-2186, OPS:3408-3465):
   ```
   up   = W_up[e]·f , gate = W_gate[e]·f          ffn_moe_up-l, ffn_moe_gate-l   [2048,6,T]
   h    = g/(1+expf(-g)) * u  with g = min(gate, 10), u = clamp(up, -10, 10)     ffn_moe_swiglu_limited-l
   down = W_down[e]·h                             ffn_moe_down-l                 [4096,6,T]
   down *= wgt[e]                                 ffn_moe_weighted-l   (weight applied AFTER down-proj)
   moe  = ((((d0 + d1) + d2) + d3) + d4) + d5     ffn_moe_out-l        (sequential adds, selection order)
   ```
   The clamp is asymmetric: the gate is clamped from above only, the up projection on both sides. Both clamps apply before SiLU.
6. Shared expert (`build_ffn`, PAR gate, G:1731-1798):
   ```
   u_s = W_up_sh·f (ffn_up-l), g_s = W_gate_sh·f (ffn_gate-l)
   h_s = silu(min(g_s,10))*clamp(u_s,-10,10)      ffn_swiglu_limited-l
   sh  = W_down_sh·h_s                            ffn_shexp-l
   ```
7. `ffn = moe + sh`. **`ffn_out-l`**
8. `X_next = hc_post(ffn, X', post_ffn, comb_ffn)` (§3.9 formula). There is no control vector here. **`l_last-l` [4096, 4, T]** (D4:1329-1334)

Layer 26 uses MXFP4 experts and Q8_0 for q_a and the shared expert. The formulas are the same; only the dequant and activation quantisation differ (§6).

---

## 4. Model head and embedding

**Embedding** (G:2303-2390, D4:1235-1245):
- `e_t = dequant_Q6_K(token_embd[:, token_t])` via `get_rows`. **No scaling** (`f_embedding_scale = 0`). Names: **`inp_tokens`, `embd`**.
- `X⁰[:, s, t] = e_t` for s = 0..3 (`ggml_repeat_4d`). **`hc_init`**
- For an image ubatch, `ubatch.embd` replaces the get_rows (`ggml_build_forward_select`).

**Head** (D4:1343-1365, 457-477). It runs only on output rows (`inp_out_ids`):
```
nrm  = RMS(flat X⁴³_t)  (16384 dims, no weight)
mix  = output_hc_fn · nrm  ∈ R^4                                hc_head_mixes
pre  = σ( (mix*hc_scale[0]) + hc_base ) + eps                   hc_head_pre
y    = ((x0*pre0 + x1*pre1) + x2*pre2) + x3*pre3                hc_head      (UNFUSED: il=-1 → separate mul/add ops, D4:301)
z    = RMS(y, output_norm.weight)                               norm, result_norm  (no "-il" suffix)
logits = output · z      (Q6_K; z → Q8_K)                       result_output [129280, n_out]
```
`res->t_embd = result_norm` (4096-d).

There are no post-hoc logit scaling or softcaps.

---

## 5. Cache and state between steps

Memory class: `llama_kv_cache_dsv4` (KVD:1210-1336; created at MDL:2426-2468). For each sequence (own stream; `kv_unified` is ignored) it holds:

| store | layers | row content | dtype | size (rows) | write rule |
|---|---|---|---|---|---|
| raw K ring (`kv_raw.swa`) | all 43 | rope'd `kv` (512) | type_k (F16 default) | `size_swa = PAD256(min(n_ctx, 128 + n_ubatch))` (768 @ ub512); `--swa-full` → n_ctx | ring. A cell is reused when empty or when `pos_cell ≤ pos_max+1−128` (KV:1042-1061) |
| `kv_raw.base` | none (0 layers) | — | — | n_ctx cells of bookkeeping | limits each seq to n_ctx tokens; a full slot → `llama_decode` fails |
| CSA cache | 21 | compressed K (512) | type_k | `PAD256(ceil(n_ctx/4))` | linear, row = block |
| LID cache | 21 | Hadamard-rotated indexer K (128) | type_k | `PAD256(ceil(n_ctx/4))` | linear |
| HCA cache | 20 | compressed K (512) | type_k | `PAD256(ceil(n_ctx/128))` | linear |
| CSA state | 21 | kv[1024], score+ape[1024] | F32 | 8 rows × streams × (1+n_rs_seq) planes | ring `pos mod 8` |
| LID state | 21 | kv[256], score+ape[256] | F32 | 8 rows | ring `pos mod 8` |
| HCA state | 20 | kv[512], score+ape[512] | F32 | 128 rows | ring `pos mod 128` |

**Per-token growth (F16):** CSA 21·1024/4 + LID 21·256/4 + HCA 20·1024/128 = 5376 + 1344 + 160 = **6880 B/token ≈ 6.7 KiB**.

Fixed parts:
- Raw ring: 768·43·1024 B = 33.8 MB.
- States: CSA 1.38 MB, HCA 10.5 MB, LID 0.34 MB (planes = 1).

Example at n_ctx = 65536: CSA 352 MB, LID 88 MB, HCA 10.5 MB.

**Positions and limits:**
- There is no context shift: `get_can_shift() == false` (KVD:1448-1452).
- Compressed rows are indexed `pos/ratio` with no wrap, so positions must stay < n_ctx_seq. The layer-less base cache enforces this through slot allocation (KVD:1351-1359, KV:1009-1083).
- `seq_rm` support (KVD:1459-1510):
  - Removing a whole sequence clears all compressed state.
  - Truncation `p0 > 0` is only allowed beyond the end, or within `n_rs_seq` tokens (rollback planes).
  - `seq_pos_min` returns pos_max, so the server checkpoints instead.
- All buffers are zeroed at creation and on clear (KV:293, KVD:990, 1335).
- `-ctk` must equal `-ctv` (CTX:3693).

---

## 6. Reduced precision and rounding on the CPU (what a bit-exact reference must mimic)

### 6.1 Storage roundings

- **raw K, CSA, LID, HCA rows**: `set_rows` F32 → type_k. With the default F16: `_mm512/_mm256_cvtps_ph(..., RNE)` (CPU:3480-3499).
  - Compressed state, ape, masks (0/-inf), hc tensors, norms and biases are all F32, so no rounding.
- **Quantised KV** (`-ctk q8_0` etc.): `attn_rot_k` turns on HADAMARD_512 for raw/CSA/HCA (KV:321-325, 1441-1458). LID always uses HADAMARD_128 (KV:327-332). The K row is then stored quantised.

### 6.2 Matmul activation conversion

`src1` is converted to the weight's `vec_dot_type` (CPU:216-400, 1322-1355).

| weight type (tensors here) | activation becomes | quantiser |
|---|---|---|
| Q8_0 (q_b, kv, compressor kv/gate, indexer q_b + compressor, wo_a, wo_b; L26 q_a/shexp) | **Q8_0**, 32-blocks | x86: `amax` over the block; `d = amax/127` stored **F16**; `id = 127/amax`; `q = round_ne(x*id)` (`_mm256_round_ps` NEAREST) (X86Q:302-380). The generic version uses `id = 1/d` and `roundf` (QNT:276-299); it is not used here |
| Q6_K (q_a, shexp, output), IQ3_XXS (experts) | **Q8_K**, 256-blocks | `max` = signed value with the largest abs; `iscale = -127/max`; `q = min(127, nearest_int(iscale*x))` (RNE magic-number trick); `d = 1/iscale` (F32); `bsums` (QNT:2768-2805, X86Q:505-507) |
| MXFP4 (L26 experts) | Q8_0 | as above |
| BF16 (ffn_gate_inp) | **BF16** | RNE (`ggml_compute_fp32_to_bf16`, ggml-impl.h:611-624) |
| F16 (KV in non-FA attention) | **F16** | RNE |
| F32 (hc fns, output_hc_fn, indexer.proj) | none | — |

Dot-product kernels:
- **Decode (`T = 1`)**: llamafile is skipped (`n < 2`, SGEMM:3820). The `ggml_vec_dot_*` kernels run.
- **Prefill**: tinyBLAS runs for F32/BF16/F16/Q8_0 weights (CPU:1296-1320, 1374-1394).
- IQ3_XXS with ≥8 rows per expert uses the IQ-panel GEMM (IQP:1053-1128; disable with `GGML_NO_IQ_PANEL=1`).
- MXFP4 on CPU buffers is repacked to `mxfp4_8x8` (REPACK:4688-4693; disable with `--no-repack`). Q8_0 and Q6_K are not repacked on x86.

The integer parts of these kernels are exact. The float accumulation order is kernel-specific, so expect ≤ a few ULP unless the same kernel is replicated.

### 6.3 Exp, log and trig

| op | implementation |
|---|---|
| `soft_max` | `ggml_v_expf` (AVX-512 polynomial, ≤1.45 ulp + 0.5) on 16-blocks, `expf` tail. Sums in double, reciprocal in double then cast to float (VEC:531-596, OPS:5667-5683). CSA compressor (n=8) uses pure `expf`; HCA (n=128) and attention use the vector exp |
| sigmoid / softplus / swiglu-silu / fused hc_comb softmax / FA one-chunk online softmax | libm `expf`, `logf`, scalar |
| RoPE | glibc `powf` (theta_scale), `logf`, `sincosf` |

### 6.4 Attention numerics by path

| path | when | Q | scores | probs → V | V accumulator | sink |
|---|---|---|---|---|---|---|
| non-FA (`-fa off`) | always | **→F16** (mul_mat conversion) | F16·F16, F32 accum | **P → F16** before `kqv` | F32 (dot) | in softmax |
| FA one_chunk | `T < 64`, or decode with n_kv < 512 | **→F16** (OPS:8732) | `vec_dot_f16` | online `expf`; masked keys skipped | **F16 (`VKQ16`), rounded every step** (OPS:8764-8778) | after the loop (OPS:8811-8827) |
| FA split-KV | `T = 1`, 1 stream, **n_kv ≥ 512** (CSA/HCA decode, or raw once the ring ≥ 512) | as one_chunk, per thread chunk of `ceil(n_kv/nth)` | as one_chunk | as one_chunk | F16 per chunk, then `reduce_partials` (expf) | first chunk |
| FA tiled | `T ≥ 64` (`GGML_FA_TILE_Q = 64`) | F32 | K→F32, `simd_gemm` | `ggml_vec_soft_max_f32` per 64-key tile | F32 | after the tiles (OPS:8852-9138) |

**Split-KV results depend on the thread count** (OPS:9261-9291).

FA is the default: `-fa auto` resolves to ON when the backend supports it (CTX:555-558). The probes `fused_dsv4_hc_{pre,comb,post}` and `fused_lid` default to ON (CTX:236-242).

### 6.5 Compiler FMA contraction

The CPU backend builds with `gnu11/gnu++17 -O3 -march=native`, so `-ffp-contract=fast`. The disassembly of `libggml-cpu.so` confirms contraction in:
- `dsv4_hc_pre`: `vfmadd231ss` chain.
- `dsv4_hc_post`: `vmulss` then `vfmadd231ss`.
- `dsv4_hc_comb`: `fmaf(mix, scale, base)`, `fmaf(e, inv, eps)`.
- `lightning_indexer` (score accumulation).
- `rope_flt`: theta blend, `1+0.1·logf` and `rotate_pairs` (`vfmsub231ss`/`vfmadd132ss`/`vfmaddsub`).
- `vec_mad1` (scale_bias, which is exact here).

To replicate bit-for-bit, port those scalar loops verbatim and compile them with the same GCC and flags.

`libllama` (D4) is built without `-march`, so `dsv4_rope_attn_factor` is computed without FMA. The resulting value is `0x3f486de1`.

### 6.6 Other order-sensitive points

- `rms_norm` squares in float.
- `sum_rows` / `vec_sum` accumulate in double, sequentially.
- MoE output sum: f32 sequential in selection order.
- Expert weighting is applied after down-projection. The reference applies it before w2, so results differ by rounding only.
- Top-k ties:
  - Indexer: `partial_sort`.
  - MoE: `std::sort`.
  - Neither is stable, and CUDA uses different algorithms.
  - **Compare `ffn_moe_topk` / `lid_top_k` first, and force the oracle's selection on a mismatch.**
- Raw-key summation order follows ring cell index, not position.

### 6.7 Recommended oracle settings for op-by-op dumps

- `-dev none` (or `CUDA_VISIBLE_DEVICES=`) so no op is offloaded to CUDA.
- Fixed `-t`, `-np 1`, fixed `-b/-ub`.
- `-fa off` for the simplest attention rounding (Q→F16, P→F16). Then also check `-fa on`.
- `GGML_NO_IQ_PANEL=1` and `--no-repack` to pin the matmul kernels.

---

## 7. Named intermediate tensors

Names follow the eval-callback / `ggml_format_name("%s-%d")` convention (CTX:2520-2545). `-l` is the layer index. Head tensors have no suffix.

Names are listed in **cb() order**. Graph execution order differs:
- In CSA/HCA layers the compressor branch runs **before** `qr`/`q`/`kv`. The first `ggml_build_forward_expand` in the layer is the compressed `cpy_k` (D4:1010/1139).
- q/kv are expanded at D4:739-740 / 795-796 / 849-850.

Several names repeat within a layer. Disambiguate by occurrence index and shape.

**Graph-level, before layer 0:** `inp_tokens`, `inp_embd`, `embd` [4096,T], `hc_init` [4096,4,T].

Inputs:
- `attn_inp_kq_mask` (raw).
- `dsv4_{csa,hca,lid}_{state_pos,state_persist_src_idxs,state_persist_dst_idxs,state_read_idxs,state_write_idxs,state_write_pos,kq_mask}`; restore/snapshot inputs only when n_rs_seq > 0.
- `attn_inp_k_rot` (LID Hadamard; also raw/CSA/HCA if `-ctk` is quantised).

Cache tensors:
- `cache_k_l<il>`. The raw, CSA, HCA and LID caches all use this same name pattern.
- `dsv4_<c>_state_kv_l<il>`, `dsv4_<c>_state_score_l<il>`.

### 7.1 Common prefix (every layer)

1. `hc_mixes-l` [24,T] → `hc_pre-l` [4,T] → `hc_post-l` [4,T] → `hc_comb-l` [4,4,T] → `hc_attn_pre-l` [4096,T]
2. `norm-l` [4096,T] (attn rms) → `attn_norm-l`
3. `qr-l` [1024,T] → `norm-l` → `qr_norm-l` → `q_norm-l` [512,64,T] → `q-l` [512,64,T]
4. `norm-l` [512,T] → `kv_norm-l` [512,1,T] → `kv-l` [512,1,T]

### 7.2 Ratio-0 layers (0, 1)

5. Non-FA only: `kq-l` [n_raw,T,64] → `kq_soft_max-l` → `v_cont-l` [n_raw,512] → `kqv-l` [512,T,64]
6. `attn_raw-l` [32768,T]

### 7.3 Ratio-128 (HCA) layers

5. `hca_state_kv-l` [512,T] → `hca_state_score-l` → `hca_state_score_ape-l`
6. `hca_state_compress-l` ×3: kv [512,128,nb], score [512,128,nb], pooled [512,1,nb]. Then `norm-l` [512,1,nb]. Then `hca_state_compress-l` ×2: normed, rope'd. Then `hca_state_compress_rot-l` only with quantised KV.
7. `hca_raw_k-l` [512,1,n_raw] → `hca_comp_k-l` [512,1,n_hca] → `hca_k_all-l` → `hca_kq_mask-l` [n_raw+n_hca,T]
8. Non-FA internals (as 7.2) → `attn_hca-l` [32768,T]

### 7.4 Ratio-4 (CSA) layers

5. `csa_state_kv-l` [1024,T] → `csa_state_score-l` → `csa_state_score_ape-l`
6. `csa_state_compress-l` ×3: kv_prev [512,4,nb], score_prev [512,4,nb], pooled [512,1,nb]. Then `norm-l`. Then `csa_state_compress-l` ×2: normed, rope'd. Then `csa_state_compress_rot-l` only with quantised KV.
7. `lid_state_kv-l` [256,T] → `lid_state_score-l` → `lid_state_score_ape-l`
8. `lid_state_compress-l` ×3 ([128,4,nb], [128,4,nb], [128,1,nb]) → `norm-l` → `lid_state_compress-l` ×2 → `lid_state_compress_rot-l` [128,1,nb] (always present)
9. `lid_q-l` [128,64,T] → `lid_q_rope-l` → `lid_q_rot-l` → `lid_weights-l` [64,T] → `lid_k-l` [128,1,n_lid]
10. Score:
    - Fused: `lid_score_masked-l` [n_lid,T,1,1].
    - Unfused: `lid_q-l` (permute), `lid_k-l` (permute), `lid_kq-l` ×2, `lid_score-l`, `lid_score_masked-l`.

    Then `lid_top_k-l` [K,T] I32.
11. `csa_raw_k-l` → `csa_comp_k-l` [512,1,n_csa] → `csa_k_all-l` → `csa_top_k_mask-l` [n_csa,T] → `csa_lid_kq_mask-l` [n_raw+n_csa,T]
12. Non-FA internals → `attn_csa_lid-l` [32768,T]

### 7.5 Common suffix (every layer)

13. `attn_derope-l` [512,64,T] → `attn_wo_a-l` [1024,T,8] → `attn_out-l` [4096,T] → `hc_attn_post-l` [4096,4,T]
14. `hc_mixes-l`, `hc_pre-l`, `hc_post-l`, `hc_comb-l` (2nd occurrence, FFN side) → `hc_ffn_pre-l` → `norm-l` → `ffn_norm-l`
15. MoE:
    `ffn_moe_logits-l` [256,T] → `ffn_moe_probs-l` → (`ffn_moe_probs_biased-l` → `ffn_moe_argsort-l` [256,T]: not for hash layers with token input) → `ffn_moe_topk-l` [6,T] I32
    → `ffn_moe_weights-l` [1,6,T] → `ffn_moe_weights_sum-l` [1,T] → `ffn_moe_weights_sum_clamped-l` → `ffn_moe_weights_norm-l` [6,T] → `ffn_moe_weights_scaled-l` [1,6,T]
    → `ffn_moe_up-l` [2048,6,T] → `ffn_moe_gate-l` → `ffn_moe_swiglu_limited-l` → `ffn_moe_down-l` [4096,6,T] → `ffn_moe_weighted-l` → `ffn_moe_out-l` [4096,T] (named twice, same tensor)
16. Shared expert and output: `ffn_up-l` [2048,T] → `ffn_gate-l` → `ffn_swiglu_limited-l` → `ffn_shexp-l` [4096,T] → `ffn_out-l` → `l_last-l` [4096,4,T]

**Head:** `hc_head_mixes` [4,n_out] → `hc_head_pre` → `hc_head` [4096,n_out] → `norm` → `result_norm` → `result_output` [129280,n_out].

Optional:
- `layer_inp-l` (HC mean) when `embeddings_layer_inp` is requested.
- `h_nextn` when `embeddings_nextn` is set.

---

## 8. Open questions and ambiguities

1. **FA kernel choice changes numerics.** Prefill ≥64 tokens uses the tiled path with F32 V; shorter prefill and decode use one_chunk with an F16 V accumulator. Decode with ≥512 keys uses split-KV, which depends on the thread count. A CPU reference needs to pick one path explicitly. `-fa off` is simplest.
2. **CUDA versus CPU.**
   - With GGML_CUDA, large-batch matmuls may be op-offloaded even at `-ngl 0`.
   - CUDA honours `n_kv_max` (sparse FA) and has its own top-k/argsort tie behaviour.
   - CUDA's BF16 router uses `prec F32`.
   - Dumps from GPU runs will not match a CPU reference bit-for-bit.
3. **What llama.cpp does not reproduce from the reference.**
   - There is no FP8 (`act_quant` on the kv nope part) or FP4 (indexer q/k) activation simulation. Caches are F16 instead.
   - Compressed rows are rounded once to F16 after RoPE (and after Hadamard for LID).
   - Decide whether your engine follows llama.cpp (the oracle) or the reference.
4. **YaRN.**
   - llama.cpp applies YaRN to compressed-layer RoPE unconditionally, at any context length.
   - It cancels the magnitude scaling to exactly 1.0.
   - The attention scale is a plain `1/√512`, with no `mscale²`.
   - Ratio-0 layers use base 10000 with no YaRN.
   - I could not check this against a reference `config.json`; no reference code is on disk.
5. **eps = 1e-20 is used for every RMS**, including the 16384-dim HC-mix norm and the output-HC norm. CONV:1072 implies the config's `rms_norm_eps` is 1e-20. Whether the reference uses a different eps for the HC norms is unverified.
6. **`hc_head` uses the unfused weighted sum** (separate mul/add ops). The per-layer hc_pre is fused (FMA chain). Do not reuse one implementation for both if you need exact bits.
7. **Dummy compressor blocks.** In decode steps that complete no block, `{csa,lid,hca}_state_compress*` dumps contain dummy garbage written to masked row `kv_size-1`. Corner case: when `n_ctx` is a multiple of 1024, that row becomes a real block at the final position. It is overwritten in the same graph before it is read.
8. **Media and hash routing.** Image (embd) ubatches in layers 0-2 use top-k on `probs + exp_probs_b_vl`, not tid2eid. Mixed text/image ubatches cannot occur because a batch is either tokens or embeddings.
9. **Top-k ties** (indexer `partial_sort`, MoE `std::sort`) are implementation-defined. They only matter for exact ties, e.g. several indexer scores of exactly 0 when all ReLUs vanish.
10. **Multi-sequence.** Per-sequence streams, `split_equal` and the CSA padding to `ceil(n_seq_tokens/4)` dummy blocks are only partly covered here. This spec assumes one sequence (`split_simple`, `n_seq_tokens = 1`, KVD:580; BATCH:507).
11. **Order of keys in the raw ring** (cell index) affects only the summation order. The masked-cell contents are zero-initialised, finite, and contribute exactly 0.
12. **MTP is not in this GGUF** (no `nextn_predict_layers`). If a later file has it, `graph_mtp` (D4:1371-1510) uses a plain iswa cache, ratio-0 style attention, and no hash routing.
