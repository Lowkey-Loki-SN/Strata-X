#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# speed_q3q2u.sh OUT - a speed sanity check, the new pack against the current one: ab_every.py (a fresh engine, the production
# flags, the engram tables evicted first) on the cold 44K report prompt plus 512 tokens at the release's sampling, with
# strata-dsv41.json's switches (route bias 0.3, PROMPT_KEEP_LONG, EG_AHEAD).  One arm each: not a precise A/B (the drive's
# slow mode can land in either) - it is there to catch a regression.
set -u
cd ${STRATA_X_ROOT}
. moex/env.sh
C=${STRATA_X_ROOT}/moex/tools/dsv41/checks
M=$HOME/AI/Models/dsv41-1017
E=$M/ref-runs/engine-checks
PY=${STRATA_AI_DIR}/Strata/.venv/bin/python
{
  for p in q3q2 q3q2u; do
    $PY $C/pagecache.py --evict $M/native/engram/*.bin | tail -1
    moex_gpus_free || exit 1
    echo "=== $(date +%T) $p"
    systemd-run --user --scope --quiet --collect env ARM_ARGS="--pack $M/$p --prompt-cache-every 65536" \
      STRATA_DSV41_ROUTE_BIAS=0.3 STRATA_DSV41_PROMPT_KEEP_LONG=1 STRATA_DSV41_ROOT_AT_SYSTEM=1 STRATA_DSV41_EG_AHEAD=1 \
      PYTHONPATH=${STRATA_X_ROOT}:${STRATA_X_ROOT}/tools $PY $C/ab_every.py $E/speed-$p-1021.log $E/long-44k-prompt.json 512 7
    grep -a -h "expert slots verified\|expert blobs of\|RAM copy .* reserved" $E/speed-$p-1021.log | cut -c1-200
  done
  echo "=== $(date +%T) speed done"
} > "$1" 2>&1
