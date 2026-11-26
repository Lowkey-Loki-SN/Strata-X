#!/bin/sh
# DeepSeek-V4.1-Flash with the release's own expert bits (strata-dsv41-native.json): the routed experts are the release's
# MXFP4 records (~/AI/Models/dsv41-1017/native, 269 GiB), the engram tables the release's FP8, the dense part the same Q8_0
# file as run-deepseek-v4.1-flash.sh - everything else (the route bias 0.3 in decode, every switch, the port, the model
# name) as that config, so a client connects to it the same way.  One server at a time: two engines do not fit this box's
# memory, and memory pressure has the user's slice killed - so this refuses while another Strata server runs.
# the checkout this launcher is in; moex/env.sh sets STRATA_X_ROOT, STRATA_MODELS, STRATA_X_DATA, STRATA_PYTHON
cd "$(dirname "$(readlink -f "$0")")" || exit 1
. moex/env.sh
if pgrep -f "serve/server.py --engine strata" >/dev/null 2>&1; then
    echo "launch-dsv4.1-flash.sh: a Strata server is running - stop it first (Ctrl+C in its terminal)" >&2
    exit 1
fi
# as run-deepseek-v4.1-flash.sh: the request limit above dsh's 256 MiB of pictures a request
export STRATA_MAX_BODY_MIB=384
exec systemd-run --user --scope --quiet --collect -- "$STRATA_PYTHON" "$STRATA_X_ROOT/serve/server.py" "--engine" "strata" "--config" "$STRATA_X_ROOT/strata-dsv41-native.json" "--port" "1234"
