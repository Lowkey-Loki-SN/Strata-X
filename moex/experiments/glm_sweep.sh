#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# One loaded model, one prefill per --runs entry (ring:chunk[:sub]); the 24K-token prompt; top logits per run.
set -uo pipefail
. ${STRATA_X_ROOT}/moex/env.sh
moex_gpus_free || exit 1
X=${STRATA_X_ROOT}; M=${STRATA_AI_DIR}/Models/GGUFs/glm53-flash/glm53-flash-q2k-fast-111.gguf
tag=$1; runs=$2; shift 2
log=$X/moex/out/gpu/sweep-$tag.log
$X/build/strata-x -m $M --tokens-file ${PROMPT:-$STRATA_X_DATA/glm-prompt-20k.tok} -n 1 --ctx 32768 --top 8 \
    --expert-profile $STRATA_X_DATA/glm-expert-profile.txt --runs "$runs" "$@" 2>&1 | tee $log | grep -E 'RUN|rror|arena|loaded'
