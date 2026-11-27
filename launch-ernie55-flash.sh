#!/bin/sh
# ERNIE-5.5-Flash (baidu/ERNIE-5.5-Flash-Preview-FP8, GGUF arch ernie55; strata-ernie55.json; moex/docs/ERNIE55_PLAN.md) from
# our own GGUF set (moex/tools/ernie55/convert_gguf.cpp: the release's FP8 dense weights exactly, the routed experts IQ3_XXS
# gate / up and IQ4_XS down with the reference model's importance matrix).  The engine is the binary the config names
# (strata-x; here a pinned copy, so that rebuilding the dev tree never changes a running server).  Port 1234, the model name
# ernie-5.5-flash, the native 1M context (the DSA layers' caches mapped as a conversation grows).  One server at a time:
# two engines do not fit this box's memory, so this refuses while another Strata server runs, and before the set the run config names is complete
# (the converter gives a file its name only after every tensor was re-read and matched).
# The RAM headroom is strata-ernie55.json's --resident-headroom-gib (20 since 2026-11-25; delete it or set 28 to go back).
# Optional server arguments are forwarded unchanged (as launch-dsv4.1-flash-ud-q2kxl.sh); no arguments keeps the config's
# host (127.0.0.1, this PC only).  A later --port overrides 1234.
# Local (unchanged): ./launch-ernie55-flash.sh
# Docker bridge: ./launch-ernie55-flash.sh --host 172.17.0.1 --api-key 'your-secret-key'
# Use a real API key for any non-loopback bind (the server only warns without one), and that same key in the client.
# the checkout this launcher is in; moex/env.sh sets STRATA_X_ROOT, STRATA_MODELS, STRATA_X_DATA, STRATA_PYTHON
cd "$(dirname "$(readlink -f "$0")")" || exit 1
. moex/env.sh
cfg="$STRATA_X_ROOT/strata-ernie55.json"
me=launch-ernie55-flash.sh
if pgrep -f "serve/server.py --engine strata" >/dev/null 2>&1; then
    echo "$me: a Strata server is running - stop it first (Ctrl+C in its terminal)" >&2
    exit 1
fi
if [ ! -f "$cfg" ]; then
    echo "$me: the run config $cfg is missing (make it from moex/configs/strata-ernie55.json)" >&2
    exit 1
fi
# the GGUF set the run config names: its "-m", the set's first file (NAME-00001-of-0000K.gguf names all K of them)
model=$("$STRATA_PYTHON" -c 'import json, sys; a = json.load(open(sys.argv[1]))["args"]; print(a[a.index("-m") + 1])' "$cfg" 2>/dev/null)
if [ -z "$model" ]; then
    echo "$me: $(basename "$cfg") names no model (\"-m\" in its args)" >&2
    exit 1
fi
case "$model" in
    *-00001-of-[0-9][0-9][0-9][0-9][0-9].gguf)
        base=${model%-00001-of-*}
        k=${model##*-of-}
        k=${k%.gguf}
        n=$(echo "$k" | sed 's/^0*//')
        i=1
        while [ "$i" -le "$n" ]; do
            if [ ! -f "$base-$(printf %05d "$i")-of-$k.gguf" ]; then
                echo "$me: file $i of $n of the GGUF set $(basename "$base") is not there yet (moex/tools/ernie55/convert_gguf.cpp writes it)" >&2
                exit 1
            fi
            i=$((i + 1))
        done ;;
    *)
        if [ ! -f "$model" ]; then
            echo "$me: the GGUF file $model that $(basename "$cfg") names is not there" >&2
            exit 1
        fi ;;
esac
if [ ! -d "$STRATA_X_DATA/tokenizers/ernie55" ]; then
    echo "$me: the tokenizer directory ~/AI/Strata-X-data/tokenizers/ernie55 is missing (tools/strata_tokenizer_families.py extract_hf)" >&2
    exit 1
fi
# the engine the run config names (here a pinned copy of a build: rebuilding the dev tree never changes a running server)
exe=$("$STRATA_PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["exe"])' "$cfg" 2>/dev/null)
if [ ! -x "$exe" ]; then
    echo "$me: the engine ${exe:-(none)} that $(basename "$cfg") names is missing" >&2
    exit 1
fi
exec systemd-run --user --scope --quiet --collect -- "$STRATA_PYTHON" "$STRATA_X_ROOT/serve/server.py" "--engine" "strata" "--config" "$cfg" "--port" "1234" "$@"
