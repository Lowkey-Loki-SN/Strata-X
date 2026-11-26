#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# Strata-X LOADING EXPERIMENT (run LAST, by hand, with every model server stopped):
# per-prompt routing captures for DeepSeek V4 Flash and MiMo V2.6 Flash on the same 48-prompt code corpus as the GLM
# capture (4096-token prompts + 128 decode tokens), then the cache-capture analysis and the fit re-run, which replaces
# the GLM curve those two borrow today.  ~30-40 min per model (GLM's took that long at ~5 T/s decode).
#   moex/experiments/capture_routing.sh [dsv4|mimo|all]      (default: all)
set -euo pipefail
X=${STRATA_X_ROOT}; . "$X/moex/env.sh"
moex_gpus_free || exit 1                       # never while Strata / tabby / llama-server holds the GPUs
OUT=$X/moex/out; mkdir -p "$OUT"
PY=${STRATA_AI_DIR}/exllamav3/.venv/bin/python
export PYTHONPATH=${STRATA_AI_DIR}/exllamav3-next${PYTHONPATH:+:$PYTHONPATH}
export TORCH_EXTENSIONS_DIR=${HOME}/.cache/torch_extensions_next
export EXL3_MOE_STREAM_T=8 EXL3_MOE_CPU_WSLOT_MB=64 EXL3_MOE_CPU_WSLOTS=4 EXL3_MOE_CPU_STAGE_THREADS=8 EXL3_MOE_CPU_DYN=1
export EXL3_MOE_CPU_SWAP=0                     # routing is counted before placement; no mid-run swap sweeps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CORPUS=${STRATA_AI_DIR}/exllamav3/exllamav3/conversion/standard_cal_data/code.utf8
COMMON=(-chunk_size 4096 -cs 32768 -corpus "$CORPUS" -nprompts 48 -plen 4096 -gen 128)
cd /tmp                                        # never inside an exllamav3 checkout
run() {  # <type> <split> <residents> <model_dir> [wrapper...]
    local t=$1 mcs=$2 res=$3 m=$4; shift 4
    echo "=== $t: capture start $(date +%T)"
    "$PY" -u "$@" "$X/moex/experiments/profile_capture.py" -m "$m" -mcs "$mcs" -resident "$res" "${COMMON[@]}" \
        -o "$OUT/$t-code.npz" 2>&1 | tee "$OUT/$t-capture.log"
    "$PY" -I "$X/moex/tools/cache_capture.py" "$OUT/$t-code.npz" --frac 0.34 --gib 24.3 --bpw 2.6 3.06 4.5 \
        --model "$X/moex/models/$t.json" --json "$OUT/capture-$t.json"
}
what=${1:-all}
if [ "$what" = dsv4 ] || [ "$what" = all ]; then
    run deepseek_v4 192 64 ${STRATA_AI_DIR}/Models/EXL3/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw
fi
if [ "$what" = mimo ] || [ "$what" = all ]; then
    moex_gpus_free || exit 1
    # Yamz pack: the shim sorts the safetensors so the zz-e2e-step120 overlay wins; the uncensor edit stays OFF here
    EXL3_ABLIT_RUNTIME=off run mimo_v2 203 53 ${STRATA_AI_DIR}/Models/EXL3/MiMo-v2.6-Flash-MOPD-2.64bpw \
        ${STRATA_AI_DIR}/tabbyAPI/glmbench/yamz_hooked.py
fi
"$PY" -I "$X/moex/tools/fit.py" --bpw 2.6 3.06 --json "$OUT/fit.json"
