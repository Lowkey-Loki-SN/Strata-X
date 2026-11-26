#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# Full-model decode, plain vs MTP: same prompt, one load each.  Modes: plain | specN (window N) | specN:P (window N, min p P).
# PROMPT=file of ids, NGEN=tokens, EXTRA=more flags, MODEL=gguf.
set -uo pipefail
. ${STRATA_X_ROOT}/moex/env.sh
moex_gpus_free || exit 1
X=${STRATA_X_ROOT}; M=${MODEL:-${STRATA_AI_DIR}/Models/GGUFs/glm53-flash/glm53-flash-q2k-fast-111.gguf}
for mode in "${@:-plain spec2 spec4}"; do
    flag=""
    case "$mode" in
        spec*) w=${mode#spec}; p=""; if [[ "$w" == *:* ]]; then p="--spec-min-p ${w#*:}"; w=${w%%:*}; fi; flag="--spec $w $p";;
    esac
    log=$X/moex/out/gpu/mtp-ab-${mode//:/_}.log
    echo "=== $mode $(date +%T)" | tee $log
    $X/build/strata-x -m $M --tokens-file ${PROMPT:-$STRATA_X_DATA/glm-p2.txt} -n ${NGEN:-256} --ctx 4096 \
        --expert-profile $STRATA_X_DATA/glm-expert-profile.txt $flag ${EXTRA:-} 2>&1 | tee -a $log | grep -E 'captured|graphs off|drafter|generated|per token|tier:|MTP|draft|accept|rror|warm'
done
