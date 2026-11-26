#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# kld_q3q2u.sh OUT - the q3q2u pack against the current q3q2, the same day, the production binary, each arm a fresh engine
# (the flags of m_kld56.sh and the agentgen windows runs of 2026-10-19):
#   wikitext-2 16x512 through the prompt path (exact routing) - q3q2, q3q2u
#   the generated agent text through decode windows, exact routing - q3q2, q3q2u
#   the same with the route bias 0.3 (the production setting) - q3q2u (q3q2's: kld-q3q2-win-agentgen-rb0.3.log, 10-19)
set -u
cd ${STRATA_X_ROOT}
. moex/env.sh
M=$HOME/AI/Models/dsv41-1017
E=$M/ref-runs/engine-checks
Q=$M/quality
BIN=${STRATA_X_ROOT}/build-dsv41/strata-x
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
  for p in q3q2 q3q2u; do run wiki16-$p-1021 "${X[@]}" -- --pack $M/$p --kld $Q/native-wiki16.kld; done
  for p in q3q2 q3q2u; do run agentgen-win-rb0-$p-1021 "${X[@]}" -- --pack $M/$p --kld $Q/native-agentgen2x4096.kld --kld-mode windows; done
  run agentgen-win-rb0.3-q3q2u-1021 STRATA_DSV41_ROUTE_BIAS=0.3 "${X[@]}" -- --pack $M/q3q2u --kld $Q/native-agentgen2x4096.kld --kld-mode windows
  echo "=== $(date +%T) kld done"
} > "$1" 2>&1
