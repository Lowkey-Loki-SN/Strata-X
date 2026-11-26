#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# Full-model decode, eager vs graphs: same prompt, same tokens, one load each.
set -uo pipefail
. ${STRATA_X_ROOT}/moex/env.sh
moex_gpus_free || exit 1
X=${STRATA_X_ROOT}; M=${STRATA_AI_DIR}/Models/GGUFs/glm53-flash/glm53-flash-q2k-fast-111.gguf
for mode in "${@:-eager graphs}"; do
    flag=""; [ "$mode" = eager ] && flag="--no-graphs"
    log=$X/moex/out/gpu/decode-$mode.log
    echo "=== $mode $(date +%T)" | tee $log
    $X/build/strata-x -m $M --tokens-file ${PROMPT:-$STRATA_X_DATA/glm-p2.txt} -n ${NGEN:-96} --ctx 4096 \
        --expert-profile $STRATA_X_DATA/glm-expert-profile.txt $flag ${EXTRA:-} 2>&1 | tee -a $log | grep -E 'captured|graphs off|generated|per token|tier:|^[0-9]+,|rror'
done
