#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# moex/tools/dsv41/phase1_checks.sh - DeepSeek-V4.1 phase 1's GPU checks, in order (DSV41_PORT.md S1-S4), from build-dsv41/.
#
#   moex/tools/dsv41/phase1_checks.sh [parity|oracle|kld|oracle-eg|kld-eg|dspark|all]      (default: all but dspark)
#
# Refuses to start unless moex_gpus_free passes (no model server, no GPU client): the live server is never disturbed.
#   parity     dsv41_kernels_parity (V4.1's kernels) and dsv4_kernels_parity (V4's, which must pass unchanged)
#   oracle     strata-x --oracle on the 12-step, 40-layer fp32 oracle without the engram (each op on the reference's
#              inputs), one GPU, the engram off (STRATA_DSV41_NO_ENGRAM=1)
#   kld        strata-x --kld against the 40-layer fp32 reference without the engram (128 tokens: the scored half
#              through decode windows), the engram off
#   oracle-eg  the same oracle with the engram on both sides (S4: layers 1 and 14's rows and gates)
#   kld-eg     strata-x --kld against the 1,024-token fp32 reference with the engram (the scored half through windows)
#   dspark     64 greedy tokens plain and with the DSpark drafter: the ids must be equal
# The GGUF holds the dense weights, the DSpark drafter and the engram projections (write_dense_gguf.py); the pack the
# expert records and the engram tables.
set -euo pipefail
here=$(cd "$(dirname "$0")/../../.." && pwd)
. "$here/moex/env.sh"
moex_gpus_free || exit 1
B=$here/build-dsv41
M=${STRATA_AI_DIR}/Models/dsv41-1017
GGUF=$M/dsv41-1017-dense-q8_0-dsp-eg.gguf
PACK=$M/native
OUT=$M/ref-runs/engine-checks
mkdir -p "$OUT"
what=${1:-all}
run() { echo "=== $(date '+%T') $*"; }
scoped() { systemd-run --user --scope --collect -q "$@"; }   # an engine run in its own scope (never the terminal's)

if [[ $what == parity || $what == all ]]; then
    run "dsv41_kernels_parity"
    "$B/dsv41_kernels_parity" 2>&1 | tee "$OUT/dsv41_kernels_parity.log" | tail -60
    run "dsv4_kernels_parity (V4 unchanged)"
    "$B/dsv4_kernels_parity" 2>&1 | tee "$OUT/dsv4_kernels_parity.log" | tail -15
fi
if [[ $what == oracle || $what == all ]]; then
    O=$M/ref-runs/oracle-12-fp32
    run "strata-x --oracle $O (engram off)"
    scoped env STRATA_DSV41_NO_ENGRAM=1 "$B/strata-x" -m "$GGUF" --pack "$PACK" --device 0 --ctx 4096 --expert-slots 0 \
        --experts-in-place --threads 20 --oracle "$O" --tol 0.05 -t "$(cat "$O/ids.txt")" 2>&1 | tee "$OUT/oracle-12-fp32.log" | tail -120
fi
if [[ $what == kld || $what == all ]]; then
    K=$M/ref-runs/noeng-128-fp32/kld-base.bin
    run "strata-x --kld $K (windows, engram off)"
    scoped env STRATA_DSV41_NO_ENGRAM=1 "$B/strata-x" -m "$GGUF" --pack "$PACK" --device 0 --ctx 4096 --expert-slots 0 \
        --experts-in-place --threads 20 --kld "$K" --kld-mode windows 2>&1 | tee "$OUT/kld-noeng-128-fp32-windows.log" | tail -30
fi
if [[ $what == oracle-eg || $what == all ]]; then
    O=$M/ref-runs/oracle-12-fp32-eg
    run "strata-x --oracle $O (engram on)"
    scoped "$B/strata-x" -m "$GGUF" --pack "$PACK" --device 0 --ctx 4096 --expert-slots 0 --experts-in-place --threads 20 \
        --oracle "$O" --tol 0.05 -t "$(cat "$O/ids.txt")" 2>&1 | tee "$OUT/oracle-12-fp32-eg.log" | tail -120
fi
if [[ $what == kld-eg || $what == all ]] && [[ $what == kld-eg || -f $M/ref-runs/eg-1024-fp32/kld-base.bin ]]; then
    # (DSV41_BACKLOG #57: that reference was dropped, DSV41_PLAN §11; `all` skips the step without it, `kld-eg` still fails)
    K=$M/ref-runs/eg-1024-fp32/kld-base.bin
    run "strata-x --kld $K (windows, engram on)"
    scoped "$B/strata-x" -m "$GGUF" --pack "$PACK" --device 0 --ctx 4096 --expert-slots 0 --experts-in-place --threads 20 \
        --kld "$K" --kld-mode windows 2>&1 | tee "$OUT/kld-eg-1024-fp32-windows.log" | tail -30
fi
if [[ $what == dspark ]]; then
    ids=$(cat "$M/ref-runs/oracle-12-fp32/ids.txt")
    run "64 greedy tokens, plain"
    scoped "$B/strata-x" -m "$GGUF" --pack "$PACK" --device 0 --ctx 4096 --expert-slots 0 --experts-in-place --threads 20 \
        -t "$ids" -n 64 --print-ids 2>&1 | tee "$OUT/greedy-plain.log" | tail -5
    run "64 greedy tokens, DSpark (window 2)"
    scoped "$B/strata-x" -m "$GGUF" --pack "$PACK" --device 0 --ctx 4096 --expert-slots 0 --experts-in-place --threads 20 \
        --draft-dspark builtin --spec 2 -t "$ids" -n 64 --print-ids 2>&1 | tee "$OUT/greedy-dspark.log" | tail -5
fi
