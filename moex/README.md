# Strata-X (`moex`)

Strata-X is a fork of [Strata](https://github.com/Niko1221/Strata) that runs more of the large mixture-of-experts models
on one PC, with the same design: the dense part and the most-used experts in VRAM, the other experts pinned in RAM and
computed by a CPU pool, and the prompt read by streaming the experts through VRAM. Upstream's own path (Qwen3.8-Flash-Next,
`README.md`, `setup.py`) is kept as it is; this folder documents what the fork adds.

## What the fork adds

- **`strata-x`** (`src/x/`, `include/strata/x/`): one engine for six model families - GLM-5.3-Flash,
  DeepSeek-V4-Flash, DeepSeek-V4.1-Flash, MiMo-V2.6-Flash, Hy4-flash and ERNIE-5.5-Flash. Which model files each one
  loads, and which quantizations work, is in [`moex/docs/QUANTS.md`](docs/QUANTS.md) - read it before downloading anything.
- **The server's model families** (`serve/families.py`, `serve/family_templates/`, `serve/family_parsers.py`): each
  family's chat template, reply parser and request dialect, checked against the release's own encoder and decoder
  (`serve/test_deepseek41.py`, `serve/test_ernie55.py`, ...). The server, its OpenAI and Anthropic APIs and the web app are
  upstream's.
- **Run-config templates** ([`configs/`](configs/README.md)) and launchers in the checkout's folder (`launch-*.sh`).
- **Tools** (`tools/`): GGUF converters and inventories, tokenizer exports, parity and integrity checks, benches.
- **Reference models** (`ref/`, `tools/*/ref/`): float reference forward passes the engine's kernels are checked against.
- **Engineering notes** (`moex/docs/`, below).

## Build

`strata-x` is NVIDIA only (CUDA). CMake fetches the pinned llama.cpp it builds on and applies
`moex/patches/llama.cpp/*.patch` itself.

```bash
cmake -S . -B build-dsv41 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_MOEX_KQUANTS=ON -DSTRATA_NATIVE_EXPERTS=ON
cmake --build build-dsv41 --target strata-x
```

`CMAKE_CUDA_ARCHITECTURES` is your card's (86: RTX 30-series). The server needs Python 3.10+ with `requirements.txt`
(`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`).

## Run a model

1. Get model files that load (`moex/docs/QUANTS.md`; `python3 serve/quant_check.py --hf ORG/REPO FOLDER` checks a Hugging Face
   set from its headers before you download it).
2. Export its tokenizer: `python3 tools/strata_tokenizer_families.py --gguf FIRST_SHARD.gguf --out "$STRATA_X_DATA/tokenizers/<family>"`
   (ERNIE-5.5's GGUF carries none: `moex/docs/QUANTS.md` has its `--hf-json` command).
3. Make a run config from its template (`configs/README.md`).
4. Start the server, from the checkout's folder:

```bash
. moex/env.sh
"$STRATA_PYTHON" serve/server.py --engine strata --config strata-<model>.json     # or ./launch-<model>.sh
```

It serves on `127.0.0.1:1234` (OpenAI- and Anthropic-compatible, and the web app). Never bind another address without
`--api-key`.

## Paths

Every script here finds the checkout from its own location; the folders it uses beside the checkout default to the
layout the notes were written in, and each can be set beforehand (`moex/env.sh`):

| Variable | Default | Holds |
|---|---|---|
| `STRATA_X_ROOT` | this checkout | set by `moex/env.sh` |
| `STRATA_AI_DIR` | the checkout's parent folder | the folders below, unless set one by one |
| `STRATA_X_DATA` | `$STRATA_AI_DIR/Strata-X-data` | exported tokenizers, expert profiles, saved conversations, measurement folders |
| `STRATA_MODELS` | `$STRATA_AI_DIR/Models` | model files |
| `STRATA_PYTHON` | `./.venv`, else a Strata install's `$STRATA_AI_DIR/Strata/.venv` | the server's Python |

The notes write these folders as `~/AI/...` (`~/AI/Strata-X` is this checkout, `~/AI/Models` the models, and so on).

## Tests

- `python -m unittest discover -s serve -p "test_*.py" -t .` - the server, the families' templates and parsers (no GPU;
  the tests that need a release's own files skip without them).
- `moex/tests/test_family_*.py` - the families on the server's path (no GPU). `moex/tests/serve_*_test.py` and
  `profile_save_test.py` start the engine and need the GPUs.
- `ctest --test-dir build-dsv41` - the kernels' parity tests (GPU).
- `tools/test_ernie55_launcher_args.py` - a launcher's command line.

## Notes

`moex/docs/` is the fork's engineering log, kept as it was written: dated entries, every number with what it was measured on,
the decisions and the measurements that were rejected. Unless an entry says otherwise, the box is two RTX 3080 20 GB
(power-limited to 180 W), 125 GB of RAM, a 20-core Xeon Gold 6148 and one NVMe drive. "The user" in the notes is the
maintainer.

Until 2026-11-26 the engine was called `strata-glm`, and the notes written before then use its old names:

| In the notes | Now |
|---|---|
| `strata-glm` (the binary, the CMake target, the log lines) | `strata-x` |
| `src/glm/glm_<name>.cpp`, `src/glm/<name>.cu` | `src/x/<name>.cpp`, `src/x/<name>.cu` |
| `include/strata/glm/glm_<name>.hpp`, `include/strata/glm/<name>.hpp` | `include/strata/x/<name>.hpp` |
| `strata::glm` | `strata::x` |
| `STRATA_GLM_<NAME>` switches | `STRATA_X_<NAME>` (the engine says so when an old name is still set, and does not read it) |
| `glm_kernels_parity`, `glm_park_policy_test`, `strata_glm_kernels` | `x_kernels_parity`, `x_park_policy_test`, `strata_x_kernels` |

The model-specific names stay as they were: `glm5next`, `glm5.jinja`, `moex/ref/glm_ref.cpp`, the `strata-glm.json` run config
(GLM-5.3-Flash's) and the GLM experiments in `moex/experiments/`. File names cited with line numbers are as they were at the
time, so a line number can be out of date.

| Note | About |
|---|---|
| [`MOEX.md`](docs/MOEX.md) | The fork's design: Strata's mechanism for GLM-5.3, DeepSeek-V4 and MiMo-V2.6, and the port plan |
| [`PORT_AUDIT_GLM.md`](docs/PORT_AUDIT_GLM.md), [`PORT_AUDIT_ALL.md`](docs/PORT_AUDIT_ALL.md) | `strata-x` audited against Strata's Qwen design |
| [`STRATA_QWEN_DESIGN.md`](docs/STRATA_QWEN_DESIGN.md) | Upstream Strata's design inventory, the checklist the ports follow |
| [`DSV4_SPEC.md`](docs/DSV4_SPEC.md) | DeepSeek-V4-Flash's forward pass, operator by operator |
| [`MXFP4.md`](docs/MXFP4.md) | The MXFP4 releases of DeepSeek-V4 and MiMo-V2.6 at full speed |
| [`DSV41_PORT.md`](docs/DSV41_PORT.md) | DeepSeek-V4.1-Flash in `strata-x`: the port design |
| [`DSV41_PLAN.md`](docs/DSV41_PLAN.md), [`DSV41_BACKLOG.md`](docs/DSV41_BACKLOG.md) | DeepSeek-V4.1: the running plan and log, and everything deferred |
| [`HY4_PORT.md`](docs/HY4_PORT.md) | Hy4-flash: the port, its drafter, images and long context |
| [`ERNIE55_PLAN.md`](docs/ERNIE55_PLAN.md) | ERNIE-5.5-Flash: the port, our GGUF set, the server path and the speed work |
| [`VISION.md`](docs/VISION.md) | Images on the family engines |
| [`QUANTS.md`](docs/QUANTS.md) | Which model files and quantizations each model runs from |

## Port-planning tools

The first step of a new port (`moex/docs/MOEX.md`): describe the model, inventory its GGUF, and fit its experts to this box's
memory from a routing capture.

```bash
. moex/env.sh
python3 -I moex/tools/describe.py ~/AI/Models/EXL3/<model> --out moex/models
python3 -I moex/tools/gguf_inventory.py <first-shard.gguf>
"$STRATA_PYTHON" -I moex/tools/cache_capture.py <capture.npz> --gib 24.3 --bpw 2.6 3.06 --model moex/models/<type>.json \
    --json moex/out/capture-<type>.json
"$STRATA_PYTHON" -I moex/tools/fit.py --bpw 2.6 3.06
python3 -I moex/tools/port_plan.py
moex/experiments/capture_routing.sh all          # last: needs both GPUs free, and refuses otherwise
```
