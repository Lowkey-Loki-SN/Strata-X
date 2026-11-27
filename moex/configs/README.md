# Run configs

One `strata-<model>.json` per model this fork serves: the engine binary and its arguments, the tokenizer, sampling and the
server's options. The server reads the copy in the checkout's folder (`serve/server.py --engine strata --config
strata-<model>.json`). Those copies are per machine and stay out of git, as upstream's setup writes them; these are the
templates. Each names the engine as the build makes it (`build-dsv41/strata-x`, `moex/README.md`'s Build).

| Template | Model | Started by |
|---|---|---|
| `strata-dsv41.json` | DeepSeek-V4.1-Flash, dense part Q8_0, experts Unsloth UD-IQ3_XXS | `serve/server.py --config` |
| `strata-dsv41-native.json` | DeepSeek-V4.1-Flash, the release's own MXFP4 experts | `launch-dsv4.1-flash.sh` |
| `strata-dsv41-ud-q2kxl.json` | DeepSeek-V4.1-Flash from Unsloth's UD-Q2_K_XL GGUF files | `launch-dsv4.1-flash-ud-q2kxl.sh` |
| `strata-dsv4.json` | DeepSeek-V4-Flash | `serve/server.py --config` |
| `strata-glm.json` | GLM-5.3-Flash | `serve/server.py --config` |
| `strata-mimo.json` | MiMo-V2.6-Flash | `serve/server.py --config` |
| `strata-hy4.json` | Hy4-flash | `launch-hy4-flash.sh` |
| `strata-ernie55.json` | ERNIE-5.5-Flash (our IQ3_S / IQ4_XS GGUF set; for the IQ3_XXS set, its `-m`) | `launch-ernie55-flash.sh` |

The arguments (layer split, expert slots, RAM headroom, threads) were tuned on two RTX 3080 20 GB cards, 125 GB of RAM and a
20-core Xeon. On other hardware, start from them and leave the layout to the engine where it decides it itself.

To make one, fill in the folders (`moex/env.sh`, "Paths") and then point the model paths at your files:

```bash
. moex/env.sh
sed -e "s#<STRATA_X_ROOT>#$STRATA_X_ROOT#g" -e "s#<STRATA_X_DATA>#$STRATA_X_DATA#g" \
    -e "s#<STRATA_MODELS>#$STRATA_MODELS#g" -e "s#<STRATA_AI_DIR>#$STRATA_AI_DIR#g" \
    moex/configs/strata-ernie55.json > strata-ernie55.json
```
