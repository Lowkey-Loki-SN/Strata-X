#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# kld_iq2xs.sh OUT - ISTA-DASLab's GSQ-RCO IQ2_XS experts (the pack in dsv41-1017/iq2xs, gguf_pack.py) through kld_q3q2u.sh's
# three tests, the production binary, the same dense GGUF and engram tables, each a fresh engine:
#   wikitext-2 16x512 through the prompt path (exact routing)
#   the generated agent text through decode windows, exact routing
#   the same with the route bias 0.3 (the production setting)
# q3q2u's own numbers for the same three are DSV41_PLAN §16's (2026-10-21); PACKS="q3q2u iq2xs" runs both again
set -u
cd ${STRATA_X_ROOT}
. moex/env.sh
M=$HOME/AI/Models/dsv41-1017
E=$M/ref-runs/engine-checks
Q=$M/quality
BIN=${STRATA_X_ROOT}/build-dsv41/strata-x
D=$(date +%m%d)
run() {   # run TAG ENV... -- ARGS...
  local tag=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  moex_gpus_free || exit 1
  echo "=== $(date +%T) $tag"
  systemd-run --user --scope --unit=strata-dsv41-$tag --collect -q env "${envs[@]}" $BIN \
    -m $M/dsv41-1017-dense-q8_0-dsp-eg.gguf --gpus 0,1 --layer-split 20 --ctx 4096 --threads 20 --ssd-cache-gib 60 "$@" \
    > $E/kld-$tag.log 2>&1
  echo "exit $?"
  grep -a -h "KLD RESULT\|experts.tsv: records\|expert blobs of\|expert slots verified" $E/kld-$tag.log | cut -c1-330
}
X=(STRATA_DSV41_HEADS_MMVQ=1 STRATA_DSV41_HC_SPLIT=1)
{
  for p in ${PACKS:-iq2xs}; do run wiki16-$p-$D "${X[@]}" -- --pack $M/$p --kld $Q/native-wiki16.kld; done
  for p in ${PACKS:-iq2xs}; do run agentgen-win-rb0-$p-$D "${X[@]}" -- --pack $M/$p --kld $Q/native-agentgen2x4096.kld --kld-mode windows; done
  for p in ${PACKS:-iq2xs}; do
    run agentgen-win-rb0.3-$p-$D STRATA_DSV41_ROUTE_BIAS=0.3 "${X[@]}" -- --pack $M/$p --kld $Q/native-agentgen2x4096.kld --kld-mode windows
  done
  echo "=== $(date +%T) kld done"
} > "$1" 2>&1
