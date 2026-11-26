#!/bin/sh
# DeepSeek-V4.1-Flash with unsloth's UD-Q2_K_XL routed experts, for testing (strata-dsv41-ud-q2kxl.json): the experts from the
# UD-Q2_K_XL set's files 2-5 (~/AI/Models/dsv41-1017/ud-q2kxl, repo commit 2688c0c3, each SHA-256 checked as it downloaded;
# the set's file 7, its Q5_K engram, is not downloaded); the dense part, the full engram table (the release's FP8 values: our
# set's file 8) and the settings from our MXFP4 set (gguf-mxfp4).  Everything else as strata-dsv41-native.json - the route
# bias 0.3, every switch, port 1234, the model name - so a client connects the same way; its own expert profile (seeded
# from the native one) and conversation folder.  The binary is 0be985bf (DSV41_PLAN §75, §76: aaef063b's output bit for bit on
# this set with the lookup off, 19.25 -> 21.15 tok/s); its GPU kernel checks ran when it was installed (checks-0be985bf/PASSED - without that
# marker they run again before the start).  It starts only when files 1-6 are all there: the saved conversations' identity
# stamps every file of the set, so a file arriving after the first start would drop them once.  One server at a time, as
# launch-dsv4.1-flash.sh.
# Optional server arguments are forwarded unchanged; no arguments keeps the current config/default host.
# Local (unchanged): ./launch-dsv4.1-flash-ud-q2kxl.sh
# Docker bridge: ./launch-dsv4.1-flash-ud-q2kxl.sh --host 172.17.0.1 --api-key 'your-secret-key'
# Use a real API key for any non-loopback bind, and use that same key in the client.
# the checkout this launcher is in; moex/env.sh sets STRATA_X_ROOT, STRATA_MODELS, STRATA_X_DATA, STRATA_PYTHON
cd "$(dirname "$(readlink -f "$0")")" || exit 1
. moex/env.sh
cfg="$STRATA_X_ROOT/strata-dsv41-ud-q2kxl.json"
me=launch-dsv4.1-flash-ud-q2kxl.sh
if pgrep -f "serve/server.py --engine strata" >/dev/null 2>&1; then
    echo "$me: a Strata server is running - stop it first (Ctrl+C in its terminal)" >&2
    exit 1
fi
set_="$STRATA_MODELS/dsv41-1017/ud-q2kxl/DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL"
for i in 1 2 3 4 5 6; do   # (the download gives a file its name only after its SHA-256 matched)
    if [ ! -f "$set_-0000$i-of-00007.gguf" ]; then
        echo "$me: file $i of the UD-Q2_K_XL set is not complete yet (~/AI/Models/dsv41-1017/ud-q2kxl-download.log)" >&2
        exit 1
    fi
done
chk="$STRATA_X_DATA/checks-0be985bf"
ok=1
run_check() {   # a check binary of this build and its arguments, on GPU 0
    n=$1
    shift
    if CUDA_VISIBLE_DEVICES=0 "$chk/$n" "$@" > "$chk/$n.log" 2>&1; then
        echo "  $n: pass"
    else
        echo "  $n: FAILED - $chk/$n.log" >&2
        ok=0
    fi
}
if [ ! -f "$chk/PASSED" ]; then
    echo "$me: the first start of this binary - its GPU kernel checks first (under a minute)"
    run_check iq_parity "$chk/fixtures"   # the GPU dequantizers against gguf-py's values
    run_check iq_multi_parity             # the MMVQ kernels against a double reference
    run_check mmvq_grouped_parity         # the grouped expert launch against single-column calls, bit for bit
    if [ "$ok" != 1 ]; then
        echo "$me: not started - a kernel check failed" >&2
        exit 1
    fi
    date > "$chk/PASSED"
fi
# as launch-dsv4.1-flash.sh: the request limit above dsh's 256 MiB of pictures a request
export STRATA_MAX_BODY_MIB=384
exec systemd-run --user --scope --quiet --collect -- "$STRATA_PYTHON" "$STRATA_X_ROOT/serve/server.py" "--engine" "strata" "--config" "$cfg" "--port" "1234" "$@"
