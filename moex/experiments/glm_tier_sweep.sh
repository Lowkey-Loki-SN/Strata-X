#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# Decode with graphs under several adaptive-tier settings: one load each.  usage: glm_tier_sweep.sh "tag|flags" ...
set -uo pipefail
. ${STRATA_X_ROOT}/moex/env.sh
moex_gpus_free || exit 1
X=${STRATA_X_ROOT}; M=${STRATA_AI_DIR}/Models/GGUFs/glm53-flash/glm53-flash-q2k-fast-111.gguf
for v in "$@"; do
    tag=${v%%|*}; flags=${v#*|}
    echo "=== $tag ($flags) $(date +%T)"
    $X/build/strata-x -m $M --tokens-file ${PROMPT:-$STRATA_X_DATA/glm-p2.txt} -n ${NGEN:-384} --ctx 4096 \
        --expert-profile $STRATA_X_DATA/glm-expert-profile.txt $flags 2>&1 | grep -E 'generated|per token|CPU pool|tier:|rror|graphs off'
done
echo "=== done $(date +%T)"
