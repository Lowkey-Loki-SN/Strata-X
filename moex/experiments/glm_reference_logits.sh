#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# Strata-X LOADING EXPERIMENT (run LAST, by hand, every model server stopped): the GLM-5.3-Flash parity oracle.
# Saves llama.cpp's full logits for a fixed corpus slice; the Strata-X GLM port is then judged by KL divergence against
# them (llama-perplexity --kl-divergence on the same file and text).  Quality of the Q2_K weights is irrelevant here:
# both engines read the SAME file, so any divergence beyond kernel rounding is a port bug.
#   ~10-20 min: 8 chunks x 512 tokens, experts on the CPU (102 GB in RAM), the rest on the GPUs.
set -euo pipefail
X=${STRATA_X_ROOT}; . "$X/moex/env.sh"
moex_gpus_free || exit 1
LC=${STRATA_AI_DIR}/llama.cpp-glm5next-mtp      # the only local build that loads arch "glm5next" (mainline: "glm5-next")
M=${STRATA_AI_DIR}/Models/GGUFs/glm53-flash/glm53-flash-q2k-fast-111.gguf
TEXT=${STRATA_AI_DIR}/exllamav3/exllamav3/conversion/standard_cal_data/code.utf8   # the routing captures' corpus
OUT=$X/moex/out; mkdir -p "$OUT"
if [ ! -x "$LC/build/bin/llama-perplexity" ]; then
    echo "building llama-perplexity in $LC (one target of the existing build)"
    cmake --build "$LC/build" --target llama-perplexity -j"$(nproc)"
fi
"$LC/build/bin/llama-perplexity" -m "$M" -f "$TEXT" -c 512 --chunks 8 -ngl 99 -ot 'ffn_.*_exps=CPU' -fa on -t 20 \
    --kl-divergence-base "$OUT/glm-q2k-ref-512x8.kld" 2>&1 | tee "$OUT/glm-reference.log"
echo "oracle: $OUT/glm-q2k-ref-512x8.kld ($(du -h "$OUT/glm-q2k-ref-512x8.kld" | cut -f1))"
