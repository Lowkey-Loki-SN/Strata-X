# Images (mmproj) on the family engines

Reference implementations: mainline llama.cpp's mtmd (`~/AI/Strata-X-data/llama.cpp-mainline`, c479922ac) and Strata's Qwen
vision path (`tools/vision/strata_vision.cpp`, `serve/server.py` `Vision` / `prepare`, the engine's `GENI`).  The encoder IS
mainline's code; the text side follows how mainline feeds mtmd's embeddings into each model.

## What runs where

| step | code | notes |
|---|---|---|
| decode + preprocess + ViT + projector | `strata-vision` (tools/vision) built against **mainline** mtmd | CPU by default (`"gpu": false`, 20 threads), mainline's FA default (auto = on) |
| pad expansion | `serve/server.py` `prepare()` with `families.FamilySpec.image_tokens` | one pad per image row, only for a pad that follows the family's start marker |
| rows into the model | `strata-glm --serve` `GENI` (src/glm/glm_serve.cpp) -> `Engine::run_request(..., images)` | SVE1 records, one per image, in prompt order |
| embedding substitution | `Stage::embd_rows` / `upload_embd` (`pos0`, `Impl::image_row`) | prompt path and decode windows alike |
| prefix reuse / checkpoints | `glm_session.cpp` `ImageKey` (start, n, FNV-1a hash) | Strata's ConversationImageKey rule |

### Per family

| family | projector (new files) | projector_type | markers (template) | positions | text-side special handling |
|---|---|---|---|---|---|
| glm5 (GLM-5.3-Flash) | `~/AI/Models/GGUFs/MMProj/strata-x/GLM-5.3-Flash-mmproj-f16-glm5v.gguf` | glm5v | `<\|begin_of_image\|>` + N x `<\|image\|>` (154854) + `<\|end_of_image\|>` | 1-D (the trunk has no RoPE) | none |
| mimo2 (MiMo-V2.6-Flash) | `~/AI/Models/GGUFs/MMProj/strata-x/MiMo-V2.6-Flash-mmproj-bf16.gguf` | mimovl | `<\|vision_start\|>` + N x `<\|image_pad\|>` (151655) + `<\|vision_end\|>` | 1-D NEOX | none |
| deepseek4 (DeepSeek-V4-Flash) | the user's `dsv4-vision/mmproj-DeepSeek-V4-Flash-Vision-Exp-BF16.gguf` (loads as-is) | deepseek4v | no vocab tokens: the rows carry their own sentinels | 1-D | see "DeepSeek-V4 hook" (wired 2026-10-08) |
| deepseek41 (DeepSeek-V4.1-Flash-1017) | none: the release's own encoder on the dense safetensors (`"mmproj": ~/AI/Models/dsv41-1017/dense`). `"gpu": true` (the config's): in the engine on a GPU's free VRAM, GPU 1's first (strata-glm VENC, `src/glm/dsv41_vision.cu`, fp32: against float64 0.81x the CPU encoder's error on average; a 1920x1080 picture 3.6 s, 640x480 0.3 s - DSV41_PLAN §54); `"gpu": false`: `serve/vision_dsv41.py` (bitwise its vision.py / image_processor.py) on the CPU, run by the interpreter the vision block's `"python"` names (15.5 s / 1.4 s) | the release's | `<｜deepseek_image｜>` pad runs, one per encoder row (up to 1,024 rows a picture, its delimiters included: vision_dsv41_plan.safe_resize) | 1-D, the span read causally | routed with `bias_vl`, no engram at image positions; text parts read with checkpoints, none inside a span (DSV41_PLAN §12-§13, DSV41_BACKLOG #43, #44, #49) |

The GLM file is the user's `GLM-5.3-Flash-mmproj-f16.gguf` with its METADATA rewritten (`moex/tools/gguf_kv_patch.py
--preset glm5v`, every tensor byte copied): buun-llama-cpp's converter named the projector `glm5next` with
`clip.vision.swiglu_limit` and no pixel budget; mainline's converter (conversion/qwen3vl.py `Glm5NextVisionModel`) writes
`glm5v`, `clip.vision.swiglu_clamp` 10.0 and `image_min_pixels` / `image_max_pixels` = 16 / 8000 tokens x 28^2.  The two
trees' graphs are the same code (buun's glm5next-vision.cpp calls the glm4v build; build_vit identical).

The MiMo file comes from the BF16 tower `~/AI/Models/extras/MiMo-v2.6-Flash-MOPD/vision-tower.safetensors` through
mainline's own converter class (`moex/tools/convert_mimo_mmproj.py`: `MiMoV2VisionAudioModel` with the audio half left
out - the class needs the audio tokenizer we do not have) and the HF `vision_config` (window types, token window 64, sinks).

## Building and running

    # the encoder against mainline (CPU; add -DSTRATA_VISION_CUDA=ON for a GPU encoder, 10-20 min build)
    cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DLLAMA_DIR=~/AI/Strata-X-data/llama.cpp-mainline
    cmake --build build-vision --target strata-vision -j 20

    # the config's "vision" section (strata-glm.json / strata-mimo.json)
    "vision": {"exe": ".../build-vision/bin/strata-vision", "mmproj": "<projector>", "model": "<text gguf, shard 1>",
               "gpu": false, "threads": 20}            # optional: "max_tokens" (mtmd image_max_tokens), "min_tokens"

The server passes `--image-pad <id>` to the engine (families.FamilySpec.image_tokens) when the config has a vision section.

Encode cost on the CPU (Xeon Gold 6148, 20 threads, FA on): GLM 414 tokens (640x480) 2.1 s, 4784 tokens (2560x1440
screenshot) 63.5 s; MiMo 300 / 3600 tokens 1.9 / 43 s; DeepSeek-V4 209 / 369 tokens 2.0 / 4.7 s.  mainline's default image
budget is the projector's (GLM up to 8000 tokens, MiMo up to 12845056 px): `"max_tokens"` trades detail for time (Strata's
Qwen setup ships 300 on the CPU); a GPU encoder would cut the time but its VRAM (weights + work buffers, ~2 GB) comes out
of the expert tier.

## Decisions (and why)

* **Embedding rows, nothing else**: mainline decodes an mtmd image chunk as an embedding batch through the same graph
  (`build_inp_embd`); glm5v and mimovl use mtmd's NORMAL positions (row i at pos_0 + i) and causal attention (only gemma3,
  gemma4v and deepseek4v are non-causal in `mtmd_decode_use_non_causal`).  So both engine paths only swap the embedding row.
* **The MTP drafter at image positions sees the image row of the next position** (as DeepSeek's MTP is trained: emb of the
  next input).  Mainline's draft-mtp skips embedding batches entirely (`common/speculative.cpp` "TODO: how to make it work
  with vision tokens?") and Strata's Qwen feeds the pad token's embedding; both are draft-quality choices only - verification
  is exact, outputs never depend on the drafts - and the row is the most faithful input (no holes in the drafter's caches).
* **Image keys**: two pictures of one size give the same ids, so every stored prefix (the live tokens, each checkpoint)
  carries the images inside it and is reused only when they are the same pictures (start, rows, FNV-1a of the record).
* **Pad runs**: each run of the pad id takes the next record's n rows; two pictures back to back form one run, taken record
  by record; a picture may not reach the last prompt token (it starts the first decode window as a token).

## Parity evidence (2026-10-08)

Mainline llama-server `--mmproj <same file> --no-mmproj-offload -fa on --load-mode none --fit on -fitt 2048,2048 -c 262144
-np 1 -t 20` (the projector on the CPU at FA on, as strata-vision) vs our server, greedy, same request
(`[image_url, "text"]`):

GLM-5.3-Flash (UD-IQ2_XXS):
* prompt tokens identical (434 / 4823 / 23): the template + pad expansion produce mainline's layout;
* the 2560x1440 screenshot question: the 82-token reasoning + answer identical character for character;
* 640x480 shapes: identical reasoning up to its last sentence, then ordinary drift;
* first generated token, log-probabilities ours vs mainline: shapes The -0.144 / -0.119, Let -2.41 / -2.58, Now -3.74 / -4.03,
  This -4.56 / -4.60; screenshot The -0.0069 / -0.0132, Looking -5.33 / -4.41; text-only control 17 -0.385 / -0.453, Simple
  -1.52 / -1.39, The -2.66 / -2.47 - the image prompts agree with mainline as closely as the text-only prompt does (the
  engine's own arithmetic, not the image path);
* reuse: the same screenshot again resumed 4821 of 4823 tokens (0.1 s); a same-size screenshot with a different number
  painted in was read from 0 and answered with the new number.

MiMo-V2.6-Flash (BPW3.0; mainline with `--fit off -ngl 99 --cpu-moe -c 32768`: its `--fit` over-asks for this model -
3.86 GiB compute buffer at -ub 4096, then 1.25 GiB KV at 262K, then 20502 MiB on a 20480 MiB card - so the experts sat
on the CPU; outputs do not depend on placement):
* prompt tokens identical (315 / 315 / 20);
* 640x480 shapes: the whole 160-token reasoning identical (641 characters); the same-size second picture (yellow square,
  green circle, "VISION 7") identical for 487 characters, then drift; the text-only control identical;
* MiMo's first token is always `<think>` (log-probability 0.0 in both), so its comparison is the greedy text;
* reuse with MiMo's sliding-window rings in the checkpoint: the same screenshot resumed 3631 of 3634 tokens; the painted-over
  one was read from 0 and answered "99,9 tok/s".

Engine protocol test (moex/tests/serve_geni_test.py, the 4-layer GLM slice): 16/16 - GENI refused without --image-pad;
pad-embedding rows give GEN's exact tokens on the window path and the prompt path (the substitution changes nothing else);
rows are deterministic and reach the model; live-sequence and checkpoint reuse only with the same image (a different
image: RESUME 0, and its continuation equals a fresh engine's); five malformed requests ERR with the engine serving on.
Server test (moex/tests/test_family_images.py, no GPU): pad expansion per family, two pictures in order, a marker typed as
text stays text, the --image-pad flag (DeepSeek-V4's "vision" was refused until 10-08).

## DeepSeek-V4 hook (wired 2026-10-08; the plan it followed below)

**Done:** (1) serve/server.py `prepare()` expands each 129264 at prompt index P into `encoded - (P % 4)` pads and trims the
SVE1 record by its first `P % 4` rows (`sve1_drop_rows`); (2) glm_session.cpp reads each image span through `prompt()` in
one batch with `attn_noncausal = image_batch = true` (dsv4_attn / dsv4_attn_mma take a non-causal flag: a span row sees
the span's later raw rows, older cells keep the 128 window; CSA/HCA/the indexer unchanged; no checkpoint inside a span);
(3) `Stage::route` with `image_batch` selects with `exp_probs_b_vl` (falling back to `exp_probs_b`) and skips `tid2eid`
on the hash layers; (5) families.py deepseek4 `vision_ready=True`.  The DSpark drafter is not run over image spans.
**Evidence:** dsv4_kernels_parity (non-causal raw T=5/8/300 and CSA T=200 against a float64 reference, tensor cores
included); greedy sessions against mainline llama-server + mtmd (vision-test, 10-08): prompt tokens equal (218 / 397 /
15); with the thinking mode matched (DeepSeek-V4's template defaults to thinking OFF, llama-server turns it ON - compare
with `chat_template_kwargs.enable_thinking` set) the reasoning text on the shapes image equals mainline's as far as it
was printed, the monitor screenshot's for ~60 tokens before a greedy divergence; the top first token agrees on every
request.  strata-dsv4.json has the vision section.

mainline (`src/models/deepseek4.cpp`, `tools/mtmd/mtmd.cpp`, `src/llama-kv-cache.cpp`):
1. **Lead pads**: strata-vision tokenizes the marker alone, so every record starts with 3 IMAGE_PAD rows; mtmd uses
   `lead_pad = 3 - (P % 4)` for an image block starting at prompt position P (START lands at a position = 3 mod 4).  The
   server must expand the template's `<｜deepseek_image｜>` (129264) at index P into n - (P % 4) pads and write the record
   trimmed by its first P % 4 rows (they are identical pad rows).
2. **Non-causal image span**: deepseek4's `non_causal_type` SWA_FULL - inside the span every row sees every other row of the
   span on the raw (SWA) attention (no future mask, no 128 window); older cells keep the window; CSA / HCA / the indexer
   ignore it (only completed compressed blocks are visible).  The whole span must be read in ONE batch (no decode windows,
   no chunk cut inside a span).  Unverified mainline caveat: CSA's sparse bound `n_kv_max` (deepseek4.cpp:759) may be small
   for image rows - do not copy it blindly.
3. **Routing of image rows**: every layer selects with `exp_probs_b_vl` instead of `exp_probs_b`, and the three hash layers
   do NOT use `tid2eid` for them (plain top-k with the vl bias) - `deepseek4.cpp` `is_media` (per ubatch there, per row here).
4. Embedding rows and 1-D positions as for the other families (this file's engine hooks already do it); the DSpark drafter's
   behaviour at image rows to be checked against mainline's draft-dspark.
5. Then: `families.py` deepseek4 `image_tokens=(None, "<｜deepseek_image｜>", 129264)` gets its expansion rule (no start
   token) and `vision_ready=True`, and run_request's arch check admits it.
