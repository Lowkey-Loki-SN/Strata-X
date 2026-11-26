#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# Decode with graphs, the adaptive tier variants: one load each, same prompt, NGEN tokens.
set -uo pipefail
. ${STRATA_X_ROOT}/moex/env.sh
moex_gpus_free || exit 1
X=${STRATA_X_ROOT}; M=${STRATA_AI_DIR}/Models/GGUFs/glm53-flash/glm53-flash-q2k-fast-111.gguf
run() {
    local tag=$1; shift
    echo "=== $tag $(date +%T): $*"
    $X/build/strata-x -m $M --tokens-file ${PROMPT:-$STRATA_X_DATA/glm-p2.txt} -n ${NGEN:-192} --ctx 4096 \
        --expert-profile $STRATA_X_DATA/glm-expert-profile.txt "$@" 2>&1 | grep -E 'generated|per token|tier:|rror|graphs off'
}
run strata-rule
run static --no-adaptive
run every8-swaps32 --adapt-every 8 --adapt-swaps 32
