#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# strata-x prefill vs the prompt path's ring size, 24K-token prompt (3 chunks of 8192), production settings.
set -uo pipefail
X=${STRATA_X_ROOT}
for ring in "$@"; do
    NGEN=2 $X/moex/experiments/glm_prefill_bench.sh ring$ring ${STRATA_AI_DIR}/Strata-X-data/glm-prompt-20k.tok --prefill-ring $ring \
        2>&1 | grep -E 'prompt path|prefill 2|rror' | cut -c1-230
done
