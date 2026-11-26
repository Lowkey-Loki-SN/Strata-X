<h1 align="center">Strata-X</h1>

<p align="center"><b>GLM-5.3, DeepSeek-V4 / V4.1, MiMo-V2.6, Hy4 and ERNIE-5.5 on one PC</b><br>
NVIDIA graphics cards and system RAM · Linux · a fork of <a href="https://github.com/Niko1221/Strata">Strata</a> · MIT License</p>

Strata-X runs large mixture-of-experts models, too big for any graphics card, on a desktop PC. It is a fork of
[Strata](https://github.com/Niko1221/Strata), which does this for Qwen3.8-Flash-Next, and it keeps Strata's design: the
dense part of the model and its most-used experts stay on the graphics cards, the other experts are pinned in RAM and the
processor computes them at the same time, and a prompt is read by streaming the experts through the cards. Its engine,
`strata-x`, runs six model families. The server in front of it is Strata's: OpenAI- and Anthropic-compatible, with a web
app.

Strata-X is one person's fork, built and measured on one machine: two RTX 3080 20 GB cards (power-limited to 180 W), 125 GiB
of RAM, a 20-core Xeon Gold 6148 and one NVMe drive. Other machines are untested. There is no installer: you build the
engine, get a model's files and make its run config, as below.

## Models

| Model | Files it runs from | Context | Images | Drafter in the run config |
| --- | --- | --- | --- | --- |
| GLM-5.3-Flash | llama.cpp's conversion (`glm5-next` / `glm5next`) | 256K | yes | its MTP block |
| DeepSeek-V4-Flash | mainline llama.cpp's `deepseek4` conversion | 256K | yes | its MTP file |
| DeepSeek-V4.1-Flash | a GGUF set (mainline's or Unsloth's), or this repo's dense GGUF and expert pack | 1M | yes | none |
| MiMo-V2.6-Flash | llama.cpp's `mimo2` conversion | 256K | yes | none |
| Hy4-flash | [AngelSlim's GGUF sets](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) | 1M | yes | DFly2 |
| ERNIE-5.5-Flash | this repo's converter, or [the set run here](https://huggingface.co/Lowkey-Loki/Ernie-5.5-Flash-Preview-Custom-IQ4_XS-GGUF) | 1M | no | none |

"Context" is what the run configs here set. A drafter guesses a few tokens that the model then checks at once: the same
text, sooner, where the guesses are good.

**Before you download a model**, read [`moex/docs/QUANTS.md`](moex/docs/QUANTS.md): which files and quantization types
each model runs, which load only with a setting, and what is not supported yet. A Hugging Face set can be checked from
its headers alone, before you download it:

```bash
python3 serve/quant_check.py --hf ORG/REPO FOLDER
```

The server runs the same check on a run config's model files before it starts the engine, and names any weight the engine
has no kernel for.

## How fast is it here?

Measured on the machine above, each in its run config's settings. A token is about ¾ of a word.

| Model, files | Writes an answer | Reads a prompt | Measured |
| --- | ---: | ---: | --- |
| ERNIE-5.5-Flash, the IQ3_XXS / IQ4_XS set | 32.8 tokens/s | 591 tokens/s (20,582 tokens) | 2026-11-25 |
| Hy4-flash, AngelSlim's Q4_K_M, DFly2 | 34.5-35.7 tokens/s | 627 tokens/s (250,509 tokens) | 2026-11-19 |
| DeepSeek-V4.1-Flash, Unsloth's UD-Q2_K_XL experts | 21.2 tokens/s | 16K tokens in 46.1 s | 2026-11-10 |
| DeepSeek-V4.1-Flash, the release's MXFP4 experts | 11.3-13.2 tokens/s | 16K tokens in 74.2 s | 2026-11-09/10 |
| MiMo-V2.6-Flash, an SX-124G set | 26.2 tokens/s | 8K tokens in 15.8-16.0 s | 2026-10-13 |
| GLM-5.3-Flash, Unsloth's UD-IQ3_XXS, MTP | 21.4 tokens/s | 8K tokens in 16.7-16.9 s | 2026-10-13 |
| DeepSeek-V4-Flash, UD-IQ3_S, MTP | 24.6 tokens/s | 549 tokens/s (8K tokens) | 2026-10-10 |

The prompts, their lengths and the runs differ from row to row, and a long conversation writes slower (Hy4-flash
25.4-32.8 tokens/s at 250K-316K tokens). Each number, with how it was measured, is in the notes:
[ERNIE55_PLAN.md](moex/docs/ERNIE55_PLAN.md) §8, [HY4_PORT.md](moex/docs/HY4_PORT.md),
[DSV41_PLAN.md](moex/docs/DSV41_PLAN.md) §72-§76, [PORT_AUDIT_ALL.md](moex/docs/PORT_AUDIT_ALL.md) §5 and
[MXFP4.md](moex/docs/MXFP4.md).

## What you need

| | |
| --- | --- |
| **Graphics card** | **NVIDIA**, with CUDA. Built and run here on RTX 30 cards (compute capability 8.6); other generations are untested. One card or two: the run configs here split the layers over two. |
| **RAM** | The experts that do not fit on the cards are pinned in RAM. ERNIE-5.5's set, 126 GiB of files, runs here on the two 20 GB cards and 125 GiB of RAM. DeepSeek-V4.1 can also read experts from an NVMe drive while it answers (its SSD tier). |
| **System** | Linux. The engine and the fork's scripts were written and run on Linux only. |
| **To build** | The CUDA toolkit, CMake, Ninja, GCC and git (here CUDA 13.3, CMake 3.28 and GCC 13.3), and Python 3.10 or newer for the server. |

## How to run

These steps run ERNIE-5.5-Flash, whose files are all on Hugging Face. The other models take the same steps with their own
files ([the table below](#the-other-models)).

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

Its settings (the layers split over two cards, expert slots, RAM headroom, threads) were tuned on the machine above. On
other hardware, start from them.

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
and the other paths in the run config. Its tokenizer comes from its first GGUF file, into the folder below:

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

- **Your apps and coding agents:** an OpenAI-compatible provider with the base URL **`http://127.0.0.1:1234/v1`**. The
  model's name is its run config's `model_name` (`ernie-5.5-flash`, `deepseek-v4.1-flash`, ...).
  - Apps that use Anthropic's API: `http://127.0.0.1:1234/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:1234`).
  - Apps that use the OpenAI Responses API: `/v1/responses`.
- **In the browser:** `http://127.0.0.1:1234` opens Strata's app: chat, and a live monitor of the model and the machine.
- **Thinking:** a client's `enable_thinking` and reasoning effort are mapped to the levels each model's chat template
  knows (`serve/families.py`).
- **Pictures:** attach them in the app or send them through the API, once the model's image encoder is set up
  ([`moex/docs/VISION.md`](moex/docs/VISION.md)).
- **From another PC or a container:** start the server with `--host ADDRESS --api-key SECRET`. Never serve beyond
  `127.0.0.1` without a key.

The server's API and settings are upstream Strata's: [`docs/DETAILS.md`](docs/DETAILS.md#using-it).

## The repository

| Where | What |
| --- | --- |
| [`moex/README.md`](moex/README.md) | The fork's own README: what it adds, the paths, the tests and the index of the notes |
| [`moex/docs/QUANTS.md`](moex/docs/QUANTS.md) | Which model files and quantization types each model runs |
| `src/x/`, `include/strata/x/` | The engine, `strata-x` |
| `serve/` | The server (upstream's), each model family's chat template, reply parser and request dialect (`families.py`, `family_templates/`, `family_parsers.py`), and the quantization check (`quant_check.py`) |
| [`moex/configs/`](moex/configs/README.md) | The run-config templates |
| `launch-*.sh` | The launchers of the run configs this machine uses |
| `moex/docs/` | The engineering notes: each model's port, every measurement and the decisions, by date |
| `moex/tools/` | GGUF converters (ERNIE-5.5's), DeepSeek-V4.1's expert packs, checks and benchmarks |
| `moex/ref/`, `moex/tools/*/ref/` | The float reference models the engine's arithmetic is checked against |
| `docs/`, `setup.py`, `START-HERE.bat`, `src/core/`, ... | Upstream Strata's own: its Qwen3.8-Flash-Next engine, installer and docs |

**Tests:** `"$STRATA_PYTHON" -m unittest discover -s serve -p "test_*.py" -t .` runs the server's and the families' tests
without a GPU; `ctest --test-dir build-dsv41` runs the kernels' parity tests on the GPU. The rest:
[moex/README.md](moex/README.md#tests).

## Strata and Qwen3.8-Flash-Next

Strata-X keeps upstream Strata's engine, installer and docs for Qwen3.8-Flash-Next as they were merged at v0.1.43.1
(2026-10-13), with some later fixes ported by hand. For that model, use [Strata](https://github.com/Niko1221/Strata)
itself: its installer sets everything up, and its releases are current.

## Reporting a problem

Open an [issue](https://github.com/Lowkey-Loki-SN/Strata-X/issues) with the form that fits, and say which model files and
run config you used. A security problem: report it privately ([SECURITY.md](SECURITY.md)). Before you change the code,
read [AGENTS.md](AGENTS.md): one change per pull request, every number with what it was measured on.

## Credits and license

Strata-X is built on [Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors, and on
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). The models are their makers': Z.ai's GLM-5.3-Flash,
DeepSeek's V4-Flash and V4.1-Flash, Xiaomi's MiMo-V2.6-Flash, Tencent's Hy4-flash and Baidu's ERNIE-5.5-Flash, each under
its own license. The GGUF sets `moex/docs/QUANTS.md` names are their makers' (Unsloth, AngelSlim, AesSedai, ggml-org and
tarruda). Each model's arithmetic is checked against llama.cpp's or its release's reference code (the notes say how).

Strata-X is open source under the [MIT License](LICENSE), as Strata is.
