#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# kld_server.sh TAG ENVSET PACK MODE [WINDOW] - one KLD run of the installed engine against the native base (DSV41_PLAN §52)
#   ENVSET: server (strata-dsv41.json's switches at REV, 3d04e313 by default - the launch configuration - with the route bias
#           dropped, and its learned profile) | base (kld_q3q2u.sh's: HEADS_MMVQ + HC_SPLIT only)
#   PACK:   a pack directory under ~/AI/Models/dsv41-1017 (q3q2u, iq2xs)
#   MODE:   wiki (wikitext-2 16x512 through the prompt path) | agent (the generated agent text through decode windows)
#   WINDOW: MODE=agent's decode window rows (1: the server's one-token decode; 8: kld_q3q2u.sh's)
# EXTRA="A=1 B=0" adds switches; the log goes to ref-runs/engine-checks/kld-TAG.log
set -u
cd ${STRATA_X_ROOT}
. moex/env.sh
TAG=$1 ENVSET=$2 PACK=$3 MODE=$4 WINDOW=${5:-8} REV=${REV:-3d04e313}
M=$HOME/AI/Models/dsv41-1017
Q=$M/quality
E=$M/ref-runs/engine-checks
BIN=${STRATA_X_ROOT}/build-dsv41/strata-x
envs=()
prof=()
if [ "$ENVSET" = server ]; then
  while IFS= read -r kv; do envs+=("$kv"); done < <(git show $REV:strata-dsv41.json | python3 -I -c "
import json,sys
for k,v in json.load(sys.stdin)['env'].items():
    if k != 'STRATA_DSV41_ROUTE_BIAS': print(f'{k}={v}')")
  prof=(--expert-profile ${STRATA_AI_DIR}/Strata-X-data/dsv41-expert-profile-learned.txt)
else
  envs=(STRATA_DSV41_HEADS_MMVQ=1 STRATA_DSV41_HC_SPLIT=1)
fi
for kv in ${EXTRA:-}; do envs+=("$kv"); done
if [ "$MODE" = wiki ]; then
  args=(--kld $Q/native-wiki16.kld)
else
  args=(--kld $Q/native-agentgen2x4096.kld --kld-mode windows --kld-window "$WINDOW")
fi
moex_gpus_free || exit 1
echo "=== $(date +%T) $TAG: ${#envs[@]} switches, pack $PACK, $MODE, window $WINDOW"
systemd-run --user --scope --unit=strata-dsv41-$TAG --collect -q env "${envs[@]}" $BIN \
  -m $M/dsv41-1017-dense-q8_0-dsp-eg.gguf --pack $M/$PACK --gpus 0,1 --layer-split 20 --ctx 4096 --threads 20 \
  --ssd-cache-gib 60 "${prof[@]}" "${args[@]}" > $E/kld-$TAG.log 2>&1
echo "exit $? $(date +%T)"
grep -a -h "KLD RESULT\|in VRAM" $E/kld-$TAG.log | cut -c1-330
