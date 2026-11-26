<h1 align="center">Strata-X</h1>

<p align="center"><b>GLM-5.3, DeepSeek-V4 / V4.1, MiMo-V2.6, Hy4 and ERNIE-5.5 on one PC</b><br>
NVIDIA graphics cards and system RAM · Linux · a fork of <a href="https://github.com/Niko1221/Strata">Strata</a> · MIT License</p>

Strata-X runs mixture-of-experts models larger than its graphics cards' memory on a desktop PC. It is a fork of
[Strata](https://github.com/Niko1221/Strata) and keeps Strata's design: the dense part of the model and its most-used
experts stay on the graphics cards, the other experts are pinned in RAM and the processor computes them at the same time,
and a prompt is read by streaming the experts through the cards. Its engine, `strata-x`, has run six model families here,
and for DeepSeek-V4.1 it also streams experts from an NVMe drive. The server in front of it is Strata's, with OpenAI- and
Anthropic-style APIs and a web app.

Strata-X is one person's fork, built and measured on one machine: two RTX 3080 20 GB cards (power-limited to 180 W), 125 GiB
of RAM, a 20-core Xeon Gold 6148 and one NVMe drive. Other machines are untested. There is no installer: you build the
engine, get a model's files and make its run config, as below.

## Models

| Model | Files run here | Context in its run config | Pictures | Drafter in its run config |
| --- | --- | --- | --- | --- |
| GLM-5.3-Flash | a Q2_K set, Unsloth's UD-IQ3_XXS | 256K | yes | its MTP block |
| DeepSeek-V4-Flash | a UD-IQ3_S set, tarruda's Q3_K once re-quantized | 256K | yes | its MTP file |
| DeepSeek-V4.1-Flash | the release's MXFP4 experts; Unsloth's UD-Q2_K_XL, UD-IQ3_XXS and UD-IQ1_M experts | 1M | yes | none |
| MiMo-V2.6-Flash | AesSedai's BPW3.0, an SX-124G set | 256K | yes | none |
| Hy4-flash | [AngelSlim's](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) Q4_K_M | 1M | yes | DFly2 |
| ERNIE-5.5-Flash | the set this repo's converter made ([on Hugging Face](https://huggingface.co/Lowkey-Loki/Ernie-5.5-Flash-Preview-Custom-IQ4_XS-GGUF)) | 1M | no | none |

The context a run config sets is not the length each model was tested to: the notes say how far each was run. A drafter
guesses a few tokens that the model then checks together; it pays where the guesses are often right. Other files can load
too, and some do not: `moex/docs/QUANTS.md` has the layouts and types the engine reads for each model.

**Before you download a model**, read [`moex/docs/QUANTS.md`](moex/docs/QUANTS.md): which files and quantization types
each model runs, which load only with a setting, and what is not supported yet. A Hugging Face set can be checked from
its headers alone, before you download it:

```bash
python3 serve/quant_check.py --hf ORG/REPO FOLDER
```

The server runs the same check on a run config's model files before it starts the engine. Its rules mirror the engine's
loaders as QUANTS.md lists them, so a set that passes can still fail on something the check does not cover.

## How fast is it here?

Measured on the machine above, with each run config's settings unless a note below says otherwise; other machines were
not measured. A token is roughly ¾ of an English word.

| Model, files | Writes an answer | Reads a prompt | Measured |
| --- | ---: | ---: | --- |
| ERNIE-5.5-Flash, the IQ3_XXS / IQ4_XS set | 32.8 tokens/s | 591 tokens/s (20,582 tokens) | 2026-11-25 |
| Hy4-flash, AngelSlim's Q4_K_M, DFly2, 250K tokens deep | 22.9-25.5 tokens/s | 416 tokens/s (217,741 tokens) | 2026-11-19 |
| DeepSeek-V4.1-Flash, Unsloth's UD-Q2_K_XL experts | 21.2 tokens/s | ~350 tokens/s (~16K tokens) ¹ | 2026-11-10 |
| DeepSeek-V4.1-Flash, the release's MXFP4 experts | 11.3-13.2 tokens/s ² | ~215 tokens/s (~16K tokens), 154.5 GB of experts from the SSD ² | 2026-11-09/10 |
| MiMo-V2.6-Flash, an SX-124G set | 26.2 tokens/s | 513-519 tokens/s (8,192 tokens) | 2026-10-13 |
| GLM-5.3-Flash, Unsloth's UD-IQ3_XXS, MTP | 21.4 tokens/s | 485-490 tokens/s (8,192 tokens) | 2026-10-13 |
| DeepSeek-V4-Flash, UD-IQ3_S, MTP | 24.6 tokens/s | 549 tokens/s (8K tokens) | 2026-10-10 |

¹ A cold read in a test layout (route bias off, 2,000 expert slots, a 30 GiB SSD cache, no warm start). How much of it
came from the SSD was not recorded. Over 2.2 hours of use, the drive read at most 13 MB per decoded token with this set,
prompts included.

² A cold first turn in the notes' comparison of three engines (route bias off): 8,219 expert records of 18.8 MB came from
the SSD and 4,287 from its RAM cache, and the drive read 220 GB over the whole turn. The decode figures count only the
requests the drive's slow mode did not hit (those it hit ran at 2.4-4.1 tokens/s). Over two 12-hour days, the drive read
~150 MB per decoded token with this set, prompts included.

The prompts, their lengths and the runs differ from row to row; the Hy4-flash row is a 250K-token conversation through
the server at its run config's sampling, the others shorter prompts. Each number, with how it was measured, is in the
notes: [ERNIE55_PLAN.md](moex/docs/ERNIE55_PLAN.md) §8, [HY4_PORT.md](moex/docs/HY4_PORT.md),
[DSV41_PLAN.md](moex/docs/DSV41_PLAN.md) §72-§76, [PORT_AUDIT_ALL.md](moex/docs/PORT_AUDIT_ALL.md) §5 and
[MXFP4.md](moex/docs/MXFP4.md).

## What you need

| | |
| --- | --- |
| **Graphics card** | **NVIDIA**, with CUDA. Built and run here on RTX 30 cards (compute capability 8.6); other generations are untested. The run configs here split the layers over two cards; one-card runs were used only for some of the engine's checks. |
| **RAM** | The experts that do not fit on the cards are pinned in RAM. ERNIE-5.5's set, 126 GiB of files, runs here on the two 20 GB cards and 125 GiB of RAM. DeepSeek-V4.1 can also read experts from an NVMe drive while it answers (its SSD tier). Less RAM is untested. |
| **System** | Linux. The engine and the fork's scripts were written and run on Linux only. |
| **To build** | The CUDA toolkit, CMake, Ninja, GCC and git (here CUDA 13.3, CMake 3.28 and GCC 13.3), and Python for the server (3.12 here; upstream Strata's server needs 3.10 or newer). |

## How to run

These steps run ERNIE-5.5-Flash, whose files are all on Hugging Face. They were followed from a fresh clone on the machine
above (2026-11-26), with the model's files from a local copy rather than a new download. The other models take the same
steps with their own files ([the table below](#the-other-models)).

**1. Build the engine.** CMake fetches the pinned llama.cpp it builds on, and applies `moex/patches/llama.cpp/` itself.
Set `CMAKE_CUDA_ARCHITECTURES` to your card's: 86 is the RTX 30 series.

```bash
git clone https://github.com/Lowkey-Loki-SN/Strata-X.git
cd Strata-X
cmake -S . -B build-dsv41 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_MOEX_KQUANTS=ON -DSTRATA_NATIVE_EXPERTS=ON
cmake --build build-dsv41 --target strata-x
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt huggingface_hub
```

**2. Set the folders**, in every new terminal. `moex/env.sh` finds the checkout and puts the rest beside it: `../Models`
for model files, `../Strata-X-data` for tokenizers, profiles and saved conversations, and the Python in `.venv`. Each can
be set before you source it ([Paths](moex/README.md#paths)).

```bash
. moex/env.sh
```

**3. Download the model** (three files, 135 GB) and the release's tokenizer. If Hugging Face asks, accept the release's
license on its page and log in with `.venv/bin/hf auth login`.

```bash
.venv/bin/hf download Lowkey-Loki/Ernie-5.5-Flash-Preview-Custom-IQ4_XS-GGUF --include "*.gguf" \
    --local-dir "$STRATA_MODELS/ernie55-gguf"
.venv/bin/hf download baidu/ERNIE-5.5-Flash-Preview-FP8 tokenizer.json \
    --local-dir "$STRATA_MODELS/HF/ERNIE-5.5-Flash-Preview-FP8"
```

**4. Export the tokenizer** into the folder the run config reads:

```bash
"$STRATA_PYTHON" tools/strata_tokenizer_families.py --hf-json "$STRATA_MODELS/HF/ERNIE-5.5-Flash-Preview-FP8/tokenizer.json" \
    --pre ernie55 --roles bos=2,eos=8,unk=0,pad=0 --eog 2,8 --out "$STRATA_X_DATA/tokenizers/ernie55"
```

**5. Make the run config** from its template in [`moex/configs/`](moex/configs/README.md). This fills in the folders; the
template already names the files of step 3.

```bash
sed -e "s#<STRATA_X_ROOT>#$STRATA_X_ROOT#g" -e "s#<STRATA_X_DATA>#$STRATA_X_DATA#g" \
    -e "s#<STRATA_MODELS>#$STRATA_MODELS#g" -e "s#<STRATA_AI_DIR>#$STRATA_AI_DIR#g" \
    moex/configs/strata-ernie55.json > strata-ernie55.json
```

Its settings (the layers split over two cards, expert slots, RAM headroom, threads) were tuned on the machine above and
are untested elsewhere; other hardware may need other values.

**6. Start the server:**

```bash
./launch-ernie55-flash.sh
```

The launcher first checks that the files, the tokenizer and the engine are there and that no other Strata server runs;
`"$STRATA_PYTHON" serve/server.py --engine strata --config strata-ernie55.json` starts the same server without those
checks. It serves on `127.0.0.1:1234` until you press Ctrl+C. The first start reads the model's files into RAM and has no
expert profile yet: the run learns which experts are used most, saves that, and later starts begin from it.

### The other models

Each has a template in `moex/configs/` that names its files and settings; if your files are somewhere else, change `-m`
and the other paths in the run config. These models were run here from the maintainer's checkout, not from a fresh clone.
Each one's tokenizer comes from its first GGUF file, into the folder below:

```bash
"$STRATA_PYTHON" tools/strata_tokenizer_families.py --gguf FIRST_FILE.gguf --out "$STRATA_X_DATA/tokenizers/FOLDER"
```

| Model | Template (launcher) | Tokenizer folder | The files it names | More |
| --- | --- | --- | --- | --- |
| Hy4-flash | `strata-hy4.json` (`launch-hy4-flash.sh`) | `hy4` | [AngelSlim/Hy4-flash-GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF)'s Q4_K_M; for pictures its mmproj and the release's `processor_config.json`; the DFly2 drafter, made from [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash)'s last file and `config.json` by `moex/tools/hy4/dfly_to_gguf.py` | [HY4_PORT.md](moex/docs/HY4_PORT.md) |
| DeepSeek-V4.1-Flash | `strata-dsv41-ud-q2kxl.json` (`launch-dsv4.1-flash-ud-q2kxl.sh`) | `deepseek4` | [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF)'s UD-Q2_K_XL experts, with the dense part and the MXFP8 Engram table of the release's GGUF conversion by `moex/tools/dsv41/gguf_convert.py` (mainline llama.cpp's converter, [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)); for pictures the release's own encoder | [QUANTS.md](moex/docs/QUANTS.md), [DSV41_PLAN.md](moex/docs/DSV41_PLAN.md) §68, [VISION.md](moex/docs/VISION.md) |
| DeepSeek-V4.1-Flash | `strata-dsv41-native.json` (`launch-dsv4.1-flash.sh`) | `deepseek4` | that conversion, with the release's MXFP4 experts as a pack (`moex/tools/dsv41/`) | the same |
| GLM-5.3-Flash | `strata-glm.json` | `glm5` | Unsloth's UD-IQ3_XXS, and an mmproj of type `glm5v` for pictures | [QUANTS.md](moex/docs/QUANTS.md), [VISION.md](moex/docs/VISION.md) |
| DeepSeek-V4-Flash | `strata-dsv4.json` | `deepseek4` | tarruda's Q3_K of the Vision-Exp release once its BF16 head, compressor and indexer-compressor weights are re-quantized, ggml-org's MTP file, and the release's mmproj | [QUANTS.md](moex/docs/QUANTS.md) |
| MiMo-V2.6-Flash | `strata-mimo.json` | `mimo2` | an SX-124G set, and an mmproj made by `moex/tools/convert_mimo_mmproj.py` | [QUANTS.md](moex/docs/QUANTS.md), [VISION.md](moex/docs/VISION.md) |

## Using it

- **Clients of the OpenAI API:** the base URL **`http://127.0.0.1:1234/v1`**, and as the model's name its run config's
  `model_name` (`ernie-5.5-flash`, `deepseek-v4.1-flash`, ...).
  - Anthropic's API: `http://127.0.0.1:1234/v1/messages`. The OpenAI Responses API: `/v1/responses`.
  - The fresh-clone test sent one request to each of the three, with ERNIE-5.5. Other clients and features are as upstream
    Strata's server documents them ([`docs/DETAILS.md`](docs/DETAILS.md#using-it)), untested here.
- **In the browser:** the server also serves Strata's web app, at `http://127.0.0.1:1234`.
- **Thinking:** `reasoning_effort` (`none`, `low`, `medium`, `high`), or `chat_template_kwargs` with `"enable_thinking":
  false`; each model's template takes the levels it has (`serve/families.py`).
- **Pictures:** attach them in the app or send them through the API, once the model's image encoder is set up
  ([`moex/docs/VISION.md`](moex/docs/VISION.md)).
- **From another PC or a container:** start the server with `--host ADDRESS --api-key SECRET`. Never serve beyond
  `127.0.0.1` without a key.

## DeepSeek-V4.1 from an SSD

DeepSeek-V4.1-Flash's routed experts do not fit on this machine: the release's MXFP4 experts alone are 288.8 GB, more than
its 40 GB of VRAM and 125 GiB of RAM together. The engine reads the experts it holds nowhere else from the NVMe drive while
it works, and much of the fork's DeepSeek-V4.1 work went into making that usable. The main parts, each measured on the
machine above, several on earlier expert packs than today's ([DSV41_PLAN.md](moex/docs/DSV41_PLAN.md) has the
measurements and their conditions):

- **Reads that bypass the page cache.** Experts in neither VRAM nor RAM are read with O_DIRECT straight into pinned memory
  (12 readers, 4 MiB pieces, demand reads first). A first version read them through mmap, which churned 2-3 GB of page
  cache a token until systemd-oomd stopped the session.
- **A RAM cache of the experts read** (`--ssd-cache-gib`, 90 in the templates), in place of a fixed RAM copy. With 60 GiB on
  an early pack, fresh coding text went from 1.20 to 3.78 tokens/s, and from 93.3 to 26.5 drive reads a token.
  `STRATA_DSV41_CACHE_WARM=1` fills it with the expert profile's hottest experts at the start: a first turn decoded at
  3.37-3.40 tokens/s instead of 1.12.
- **Route bias** (`STRATA_DSV41_ROUTE_BIAS=0.3` in the templates; **it changes the output**): during decode, an expert that
  would need a drive read has its routing score lowered a little. On an earlier pack, decode went from 6.97 / 10.44 / 12.04
  to 11.07 / 17.55 / 21.00 tokens/s over three passes of the same text, for a mean KL divergence 0.0054 higher on wikitext
  and 0.0083 higher on code and agent text.
- **The drive's slow mode.** Under this load the drive rewrote its most-read blocks (read disturb), dropping to ~490
  requests a second for ~64 s at a time and starving other programs' reads. The drive guard (`STRATA_DSV41_DRIVE_GUARD=1`)
  reads the disk's counters every 250 ms and drops the engine to one reader of 128 KB calls until the drive recovers:
  another program's 4 KB reads in an episode took a median 1.77 ms, against 1,020-2,084 ms a request in episodes without
  it.
- **Prompt reads.** A chunk reads only the experts its routing picked (a 32-token prompt 81.7 → 16.6 s). Layer-major windows
  read each layer's missing experts once per window (`STRATA_DSV41_LAYER_MAJOR=1`: a cold 44K read 166.9 → 125.3 s, 299 →
  250 GB from the drive). Rules for which reads stay in the cache cut the drive's bytes further
  (`STRATA_DSV41_PROMPT_KEEP_LONG=1`: -27%).
- **1M tokens of context** (`STRATA_DSV41_KV_PACK=1`): the compressed attention rows stored packed and read back bit for bit,
  945 MiB instead of 3,225 MiB at 1M, which leaves room for 175 more experts in VRAM.

The result is the two DeepSeek-V4.1 rows of the speed table. The switches are in the templates
(`moex/configs/strata-dsv41-*.json`), and the engine prints the DeepSeek switches it was started with.

## Switches for other setups

Most of the engine's switches are experimental. Each below was measured on the machine above, usually on one model, and
none on other hardware, so on yours another setting may be faster. A run config sets them: flags in its `"args"`,
environment variables in its `"env"`. DeepSeek-V4.1's SSD switches are in the section above.

| Switch | Default | What it does | Measured here |
| --- | --- | --- | --- |
| `--layer-split N` | auto | How many trunk layers, with their expert slots, go on the first card | ERNIE-5.5: a 24K read 54.5 → 49.8 s with 23 instead of auto |
| `--expert-slots N`, `--vram-reserve-mib M`, `--vram-reserve-later-mib M` | as many as fit, after 700 MiB kept free on the first card and 2,048 on the others | The experts held in VRAM, and the VRAM left to other programs | - |
| `--expert-profile F` and the config's `"expert_profile_save"` | none | The experts VRAM starts with; the server saves what a run learned and starts from it | GLM-5.3: the first request 17.2 → 20.3 tokens/s with a learned profile |
| `--resident-headroom-gib G` | 28 | The RAM left free when the RAM copy of the experts is sized | ERNIE-5.5 at 20: an 8K read 296 → 317 tokens/s, 1.59 GiB re-read from the drive instead of 10.08 |
| `STRATA_DSV41_HUGE_PIN=1` | off | Pinned RAM on 2 MiB pages (DeepSeek-V4.1's cache, and every model's RAM copy) | DeepSeek-V4.1 decode +7-9%; ERNIE-5.5's RAM copy pinned in 10.8 s instead of 28.4 |
| `STRATA_DSV41_SLOT_SIZES=1` | off | Each VRAM slot the size of its own expert (DeepSeek-V4.1's mixed sets) | 10.9% more experts in the same VRAM |
| `--threads N` | every physical core but the first | The CPU pool's threads | DeepSeek-V4.1: with 19 instead of 20, the pool's time a token ~5% longer |
| `STRATA_POOL_SPREAD=1`, `STRATA_POOL_PIPELINE=1`, `STRATA_POOL_TASKS=252` | off, off, 0 | How the pool's threads share out the experts' rows; the results are the same | ERNIE-5.5 decode 23.99 / 25.30 → 25.71 / 26.82 tokens/s; Hy4-flash +6%; not measured on GLM-5.3, MiMo or DeepSeek-V4 |
| `--max-context N` | 4,096 | The longest conversation the KV cache holds | - |
| `STRATA_HY4_KV_GROW=1` (ERNIE-5.5's cache grows this way by default) | off | The KV cache mapped as the conversation grows; the VRAM not yet needed holds experts meanwhile | Hy4-flash, greedy runs: a 48K read 27% faster, decode 15-25% faster |
| `STRATA_DSV41_KV_PACK=1` | off | DeepSeek-V4.1's compressed attention rows stored packed, bit for bit | At 1M: 945 MiB instead of 3,225, 175 more experts in VRAM |
| `--conversation-cache-mib N` and the config's `"conversation_dir"` | off | Parked conversations, in RAM and on disk, resumed without a re-read (DeepSeek-V4 and V4.1) | A 16K conversation resumed in 7.4 s instead of 51.2 s |
| `--prefill-ring-frac F` | 0.55 | The share of a layer's experts streamed ahead while a prompt is read | ERNIE-5.5 at 0.75: a 24K read 5.9% faster, a 240,953-token read 532 → 491 s |
| `STRATA_X_TC3=1` (other roundings) | off | The prompt's BF16 / FP8 matrix products on tensor cores | ERNIE-5.5: a 24K read 47.9 → 44.6 s, the KL divergence unchanged within its noise |
| `--mtp`, `--draft-mtp F`, `--draft-dfly F`, `--draft-dflash F`, `--draft-dspark F` | off | The model's own drafter: GLM-5.3's, DeepSeek-V4's, Hy4-flash's, MiMo's, DeepSeek-V4.1's | GLM-5.3 34.4-35.0 against 27.0 tokens/s plain; DeepSeek-V4 31.2 against 27.7; Hy4-flash +15% overall (code +43%, prose -3%); MiMo's and DeepSeek-V4.1's were slower here, so off |
| `--suffix-draft N`, `--lookup-window N` | on (matches of 3 tokens) | Drafts copied from the conversation itself | Where a reply repeats its prompt: MiMo +27% on a code edit; ERNIE-5.5 44.6-46.0 against 27.7 tokens/s on a verbatim repeat |

The notes have what each switch was measured against, and the ones tried and not kept.

## The repository

| Where | What |
| --- | --- |
| [`moex/README.md`](moex/README.md) | The fork's own README: what it adds, the paths, the tests and the index of the notes |
| [`moex/docs/QUANTS.md`](moex/docs/QUANTS.md) | Which model files and quantization types each model runs |
| `src/x/`, `include/strata/x/` | The engine, `strata-x` |
| `serve/` | The server (upstream's), each model family's chat template, reply parser and request dialect (`families.py`, `family_templates/`, `family_parsers.py`), and the quantization check (`quant_check.py`) |
| [`moex/configs/`](moex/configs/README.md) | The run-config templates |
| `launch-*.sh` | The launchers of the run configs this machine uses |
| `moex/docs/` | The engineering notes: each model's port, the measurements and the decisions, by date |
| `moex/tools/` | GGUF converters (ERNIE-5.5's), DeepSeek-V4.1's expert packs, checks and benchmarks |
| `moex/ref/`, `moex/tools/*/ref/` | The float reference models the engine's arithmetic is checked against |
| `docs/`, `setup.py`, `START-HERE.bat`, `src/core/`, ... | Upstream Strata's own: its engine, installer and docs |

**Tests:** `"$STRATA_PYTHON" -m unittest discover -s serve -p "test_*.py" -t .` runs the server's and the families' tests
without a GPU; `ctest --test-dir build-dsv41` runs the kernels' parity tests on the GPU. The rest:
[moex/README.md](moex/README.md#tests).

## Upstream Strata

Strata-X keeps upstream Strata's own engine, installer and docs as they were merged at v0.1.43.1 (2026-10-13), with some
later fixes ported by hand. For the models upstream Strata supports, use [Strata](https://github.com/Niko1221/Strata)
itself, with its own installer and its current releases.

## Reporting a problem

Open an [issue](https://github.com/Lowkey-Loki-SN/Strata-X/issues) with the form that fits, and say which model files and
run config you used. A security problem: report it privately ([SECURITY.md](SECURITY.md)). Before you change the code,
read [AGENTS.md](AGENTS.md): one change per pull request, every number with what it was measured on.

## Credits and license

Strata-X is built on [Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors, and on
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). The models are their makers': Z.ai's GLM-5.3-Flash,
DeepSeek's V4-Flash and V4.1-Flash, Xiaomi's MiMo-V2.6-Flash, Tencent's Hy4-flash and Baidu's ERNIE-5.5-Flash, each under
its own license. The GGUF sets `moex/docs/QUANTS.md` names are their makers' (Unsloth, AngelSlim, AesSedai, ggml-org and
tarruda). Each model's arithmetic was checked against llama.cpp or a float reference of its release; the notes say how,
and how far.

Strata-X is open source under the [MIT License](LICENSE), as Strata is.
