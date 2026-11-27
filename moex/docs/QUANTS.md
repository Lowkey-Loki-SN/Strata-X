# What each model runs from

`strata-x` reads GGUF files. A download runs only when two things hold: its architecture and layout are the ones the
engine reads for that model, and every tensor's type has a kernel on the path that tensor takes. This page says which files
those are, model by model. Check a file before you download it:

```bash
python3 serve/quant_check.py --hf ORG/REPO [FOLDER_OR_FILE]   # a Hugging Face set: its headers only, no download
python3 serve/quant_check.py FIRST_SHARD.gguf                 # files on disk (every shard of a split set)
```

The server runs the same check on a run config's `-m` set before it starts the engine (`STRATA_QUANT_CHECK=0` skips it):
a weight the engine has no kernel for stops the start with its name, and a slow path is said once. The check's rules are the
engine's own, each cited to its source in `serve/quant_check.py`.

## Quantization types, by where they run

| Type | GPU decode (experts in VRAM, and every dense weight) | Batched prompt path (experts) | CPU pool (experts in RAM) |
|---|---|---|---|
| Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | yes | yes | yes |
| IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS | yes | yes | yes |
| MXFP4, NVFP4, Q2_0 | yes | yes | yes |
| Q4_0, Q5_0 | yes | no | yes |
| Q5_1 | yes, but decode never reads its VRAM slots | no | yes |
| Q4_1, TQ1_0, TQ2_0, Q1_0 | no | no | yes |
| F16, BF16 experts | no | no | yes (not DeepSeek-V4.1's) |
| F32 experts | no | no | no |

What a "no" costs:

- An expert type without a GPU kernel stays in RAM, and the CPU computes it: decode slows with every such expert.
- One MoE layer whose expert types have no batched prompt kernel switches that path off for the whole model. Prompts are then
  read in 8-token decode windows: about 25 tok/s on GLM-5.3, against 745-990 with it (`moex/docs/MOEX.md`, §7c and §7e).
- A layer's gate and up experts must be one type.
- Q5_0 and Q5_1 get prompt tiles with the build options `STRATA_ORCA_Q4KS_MMQ` / `STRATA_MMQ_KQUANTS` (`CMakeLists.txt`);
  Q4_0 and Q4_1 have none.

Dense weights (attention, shared experts, routers, the head) run on the GPU only: F32 or a type with a GPU decode kernel,
and BF16 where a model's section says so. F16 is refused by every model. The token embedding is read on the CPU and can be
any type.

Sets that mix expert types per layer (Unsloth's UD quantizations) load. Only DeepSeek-V4.1 sizes VRAM slots per expert
(`STRATA_DSV41_SLOT_SIZES=1`); for the others the layers with the larger experts stay in RAM.

## GLM-5.3-Flash

- **Files**: llama.cpp's conversion - mainline's and Unsloth's (`general.architecture` `glm5-next`), or the llama.cpp PR
  tree's (`glm5next`).
- **Dense weights**: F32 or a GPU type; the router F32. A set with BF16 or F16 projections or head does not load.
- **Run here**: a Q2_K set (the engine within KLD 0.0146 of llama.cpp on the same file, `moex/docs/PORT_AUDIT_GLM.md`) and
  Unsloth's UD-IQ3_XXS (`moex/docs/PORT_AUDIT_ALL.md`).
- **Also**: drafting with its MTP block (`--mtp`); images with an mmproj of projector type `glm5v` (`moex/docs/VISION.md`).

## DeepSeek-V4-Flash

- **Files**: mainline llama.cpp's `deepseek4` conversion, with YaRN rope scaling and no NextN blocks in the file. The MTP
  drafter is a separate file (`--draft-mtp`, for example ggml-org's `mtp-DeepSeek-V4-Flash-MXFP4.gguf`).
- **Dense weights**: F32 or a GPU type; the router BF16 or F32; `indexer.proj`, `indexer_compressor_ape` and `output_hc_fn`
  F32 only.
- **Does not load as published**: ggml-org's MXFP4 set (its `indexer.proj` is Q8_0) and tarruda's Q3_K (BF16 head,
  compressor and indexer-compressor weights). Both ran here once those tensors were re-quantized.
- **Run here**: a UD-IQ3_S set of the Vision-Exp release (the engine within KLD 0.0070 of llama.cpp,
  `moex/docs/PORT_AUDIT_ALL.md`) and the re-quantized Q3_K.
- **Also**: images with the release's mmproj (`moex/docs/VISION.md`).

## DeepSeek-V4.1-Flash

Two ways to give it the model:

1. **A GGUF set**: mainline llama.cpp's `deepseek41` conversion, or Unsloth's sets. `-m` names the first file, and the
   experts are read from the set in place (`--experts-gguf` names another set for them). Such a set needs:
   - `STRATA_DSV41_BF16_W=1` in the run config's env: the sets store five compressor / indexer weights as BF16
     (`moex/configs/strata-dsv41-ud-q2kxl.json` sets it);
   - the Engram table (`per_layer_token_embd.weight`) as MXFP8 or IQ4_NL. Mainline's stock converter writes it as BF16 and
     Unsloth's UD-IQ3_XXS set as Q6_K: those need another file's table (`STRATA_DSV41_ENGRAM_GGUF=FILE`) or a pack;
   - a drive with 512-byte logical sectors (the experts are read with O_DIRECT; a 4Kn drive is refused).
2. **This repo's dense GGUF and an expert pack** (`moex/tools/dsv41/`: `write_dense_gguf.py`, then `gguf_pack.py` from a
   GGUF set's experts or `convert_pack.py` from the release): the daily config's way (`moex/configs/strata-dsv41.json`).

- **Dense weights**: F32, BF16 or a GPU type; BF16 matrices that hold exact FP8 values run as FP8. **Experts**: no F16 or BF16.
- **Run here**: the release's own MXFP4 experts, as a pack and as a GGUF set; Unsloth's UD-Q2_K_XL and UD-IQ1_M experts with
  the MXFP4 set's dense part and MXFP8 Engram table; Unsloth's UD-IQ3_XXS experts as a pack (KLD 0.0658 against the release,
  `moex/docs/DSV41_PLAN.md` §67). Not run here: an Unsloth set's own dense part.
- **Also**: images through the release's own encoder (`moex/docs/VISION.md`); DSpark drafting.

## MiMo-V2.6-Flash

- **Files**: llama.cpp's `mimo2` conversion, which keeps the release's fused `attn_qkv`. A file with separate Q / K / V or
  a shared expert does not load.
- **Dense weights**: F32 or a GPU type; the router and the sinks F32.
- **Run here**: AesSedai's BPW3.0 set (the engine within KLD 0.0065-0.0068 of llama.cpp, `moex/docs/MXFP4.md`) and an
  SX-124G set.
- **Also**: images with an mmproj made by `moex/tools/convert_mimo_mmproj.py` (`moex/docs/VISION.md`).

## Hy4-flash

- **Files**: AngelSlim's `Hy4-flash-GGUF` sets (llama.cpp PR #32517's layout), with their Engram shard: the table as
  F8_E4M3 or Q8_0.
- **Dense weights**: F32, BF16 or a GPU type; the router F32.
- **Run here**: Q4_K_M, checked op by op against llama.cpp (`moex/docs/HY4_PORT.md`). The Q3_K, IQ2_AH and SPARK-4Bit sets
  were checked on re-quantized slices only, never as whole sets.
- **Also**: the DFly2 drafter (`moex/tools/hy4/dfly_to_gguf.py`, from the release); images with AngelSlim's mmproj and
  `serve/vision_hy4.py`.

## ERNIE-5.5-Flash

- **Files**: only a set made by this repo's converter (`moex/tools/ernie55/convert_gguf.cpp`) from
  `baidu/ERNIE-5.5-Flash-Preview-FP8`. No other ERNIE-5.5 GGUF loads, and llama.cpp has no ERNIE-5.5 support. The
  converter writes the experts in any type ggml quantizes to and keeps the dense weights as the release's exact FP8 values
  (stored as BF16).
- **Tokenizer**: the GGUF carries none; export it from the release:
  `"$STRATA_PYTHON" tools/strata_tokenizer_families.py --hf-json <release>/tokenizer.json --pre ernie55 --roles bos=2,eos=8,unk=0,pad=0 --eog 2,8 --out "$STRATA_X_DATA/tokenizers/ernie55"`.
- **Run here**: two sets on Hugging Face, one folder each, both with IQ4_XS down and an importance matrix (KLD against
  the FP8 release; `moex/docs/ERNIE55_PLAN.md` §10). A folder's name gives the down type first: Hugging Face labels a
  GGUF file by the last quantization name in its path, so the sets show as IQ3_S and IQ3_XXS.
  - IQ3_S gate / up, `IQ4_XS-IQ3_S/`: 143.5 GB, KLD 0.0389, the run-config template's set;
  - IQ3_XXS gate / up, `IQ4_XS-IQ3_XXS/`: 135.1 GB, KLD 0.0629, a little faster, and more of its experts fit in VRAM and
    the RAM copy.

## Qwen3.8-Flash-Next (upstream Strata's engine)

Upstream's `setup.py` downloads the sets it supports, each pinned by its SHA-256: GSQ-RCO's Q2_0, IQ2_XS, IQ3_XXS and IQ3_S,
their Swift and Coder versions, and Unsloth's UD-IQ4_XS and UD-Q4_K_XL. For other files, see `docs/MODELS.md`.

## Not supported yet

The gaps a download can fall into, so that nobody downloads into them - each is work that would close it:

- BF16 or F16 dense weights on GLM-5.3, MiMo-V2.6 and DeepSeek-V4 (a BF16 decode path, as DeepSeek-V4.1 and ERNIE-5.5 have).
- Q4_0, Q4_1, Q5_0 and Q5_1 experts on the batched prompt path in the default build; Q4_1, TQ1_0, TQ2_0 and Q1_0 on the GPU.
- MiMo-V2.6 files with separate Q / K / V; DeepSeek-V4 files with NextN blocks.
- DeepSeek-V4.1 sets with a BF16 or Q6_K Engram table, without a second file's table.
- ERNIE-5.5 from any converter but this repo's.
