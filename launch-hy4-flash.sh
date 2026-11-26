#!/bin/sh
# Hy4-flash (Tencent's hy_v4_flash) from AngelSlim's Q4_K_M GGUF set (strata-hy4.json; moex/docs/HY4_PORT.md): the two files
# of ~/AI/Models/hy4-flash/q4_k_m, each SHA-256 checked as it downloaded (moex/tools/dsv41/hf_fetch.py; the repo commit is in
# SOURCE.txt).  The engine is the binary strata-hy4.json names (strata-x; here a pinned copy, so that rebuilding the dev tree
# never changes a running server).  Port 1234, the model name hy4-flash, the native 1M context (strata-hy4.json).
# One server at a time: two engines do not fit this box's memory, so this refuses while another Strata server runs, and
# before both files of the set are complete (the download gives a file its name only after its SHA-256 matched).
# the checkout this launcher is in; moex/env.sh sets STRATA_X_ROOT, STRATA_MODELS, STRATA_X_DATA, STRATA_PYTHON
cd "$(dirname "$(readlink -f "$0")")" || exit 1
. moex/env.sh
cfg="$STRATA_X_ROOT/strata-hy4.json"
me=launch-hy4-flash.sh
if pgrep -f "serve/server.py --engine strata" >/dev/null 2>&1; then
    echo "$me: a Strata server is running - stop it first (Ctrl+C in its terminal)" >&2
    exit 1
fi
set_="$STRATA_MODELS/hy4-flash/q4_k_m/Hy4-flash-Q4_K_M"
for i in 1 2; do
    if [ ! -f "$set_-0000$i-of-00002.gguf" ]; then
        echo "$me: file $i of the Q4_K_M set is not complete yet (~/AI/Models/hy4-flash/q4km-download.log)" >&2
        exit 1
    fi
done
# the DFly2 drafter (moex/tools/hy4/dfly_to_gguf.py from tencent/Hy4-flash's model-00148-of-00148.safetensors): drafts only,
# the trunk verifies every token - the output is the same with or without it
if [ ! -f "$STRATA_MODELS/hy4-flash/dfly2/hy4-dfly2-q8_0.gguf" ]; then
    echo "$me: the DFly2 drafter ~/AI/Models/hy4-flash/dfly2/hy4-dfly2-q8_0.gguf is missing (moex/docs/HY4_PORT.md)" >&2
    exit 1
fi
# the image encoder's weights (AngelSlim's mmproj, SHA-256 checked as it downloaded): without them the server reads text only
if [ ! -f "$STRATA_MODELS/hy4-flash/mmproj/mmproj-Hy4-flash-bf16.gguf" ]; then
    echo "$me: the mmproj ~/AI/Models/hy4-flash/mmproj/mmproj-Hy4-flash-bf16.gguf is missing: no image input this run" >&2
fi
# the engine the run config names (here a pinned copy of a build: rebuilding the dev tree never changes a running server)
exe=$("$STRATA_PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["exe"])' "$cfg" 2>/dev/null)
if [ ! -x "$exe" ]; then
    echo "$me: the engine ${exe:-(none)} that $(basename "$cfg") names is missing" >&2
    exit 1
fi
exec systemd-run --user --scope --quiet --collect -- "$STRATA_PYTHON" "$STRATA_X_ROOT/serve/server.py" "--engine" "strata" "--config" "$cfg" "--port" "1234"
