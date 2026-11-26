#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# strata-x prompt-path benchmark on the full Q2_K model: one load per config, prefill of a long prompt + a few tokens.
# usage: glm_prefill_bench.sh <tag> <tokens-file> <extra strata-x args...>
set -uo pipefail
. ${STRATA_X_ROOT}/moex/env.sh
moex_gpus_free || exit 1
X=${STRATA_X_ROOT}; M=${STRATA_AI_DIR}/Models/GGUFs/glm53-flash/glm53-flash-q2k-fast-111.gguf
tag=$1; toks=$2; shift 2
log=$X/moex/out/gpu/prefill-$tag.log
echo "=== $(date +%T) $tag: $*" | tee $log
free -g | sed -n 2p | tee -a $log
/usr/bin/time -v $X/build/strata-x -m $M --tokens-file $toks -n ${NGEN:-8} --ctx 32768 \
    --expert-profile $STRATA_X_DATA/glm-expert-profile.txt "$@" 2>&1 | grep -v -E '^\s+(Page size|Exit status|Socket|Signals|Swaps|File system|Average|Voluntary|Involuntary|Minor|Major|Percent|System time|User time|Command)' | tee -a $log | tail -30
echo "=== $(date +%T) done" | tee -a $log
