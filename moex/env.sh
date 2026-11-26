# Strata-X (branch moex): source this before running anything in this fork, from the checkout's folder:
#     . moex/env.sh
# It keeps the fork apart from a Strata install on the same machine (its folder, data folder and ~/.config/strata):
# Strata's setup remembers its data folder and calibration per user in $XDG_CONFIG_HOME/strata/settings.json,
# so without this a fork's setup run would rewrite the installed Strata's settings.
#
# Paths (moex/README.md, "Paths"): the checkout's own folder, and the folders beside it unless they are set first -
#     STRATA_X_DATA   tokenizers, expert profiles, conversations, measurement folders   default: ../Strata-X-data
#     STRATA_MODELS   model files                                                        default: ../Models
#     STRATA_PYTHON   the Python the launchers run the server with: ./.venv, else a Strata install's ../Strata/.venv
#     STRATA_AI_DIR   the folder that holds the checkout and the folders beside it        default: the checkout's parent
# POSIX sh (the launchers are /bin/sh): bash finds this file through $BASH_SOURCE, sh through the current folder.
if [ -n "${BASH_SOURCE:-}" ]; then
    STRATA_X_ROOT="$(cd "$(dirname "$BASH_SOURCE")/.." && pwd)"
else
    STRATA_X_ROOT="$(pwd)"
fi
export STRATA_X_ROOT
export STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"
export STRATA_X_DATA="${STRATA_X_DATA:-$STRATA_AI_DIR/Strata-X-data}"
export STRATA_MODELS="${STRATA_MODELS:-$STRATA_AI_DIR/Models}"
if [ -z "${STRATA_PYTHON:-}" ]; then
    if [ -x "$STRATA_X_ROOT/.venv/bin/python" ]; then STRATA_PYTHON="$STRATA_X_ROOT/.venv/bin/python"
    elif [ -x "$STRATA_AI_DIR/Strata/.venv/bin/python" ]; then STRATA_PYTHON="$STRATA_AI_DIR/Strata/.venv/bin/python"
    else STRATA_PYTHON=python3; fi
fi
export STRATA_PYTHON
export XDG_CONFIG_HOME="$STRATA_X_ROOT/.config"
export STRATA_X_PORT=1240          # a Strata install serves on 1234; tabby / llama-server use 1234 too

# moex_gpus_free: succeed only when no model server runs and both GPUs are (nearly) empty.
# Every loading experiment in moex/experiments calls it first: the live Strata server is never disturbed.
moex_gpus_free() {
    local busy
    busy=$(ps -eo comm,args | awk '$1=="strata" || /serve\/server\.py --engine strata/ || /tabbyAPI\/main\.py/ || $1=="llama-server"' | grep -v awk)
    if [ -n "$busy" ]; then
        echo "moex: a model server is running - refusing (stop it yourself first):" >&2
        echo "$busy" | cut -c1-140 >&2
        return 1
    fi
    # any compute client on a GPU (a CUDA app, or a video player's hardware decoding)
    local apps
    apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v '^$')
    if [ -n "$apps" ]; then
        echo "moex: a GPU has a compute client - refusing:" >&2
        echo "$apps" | cut -c1-140 >&2
        return 1
    fi
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    if [ "${used:-99999}" -gt 1500 ]; then
        echo "moex: a GPU still has ${used} MiB in use - refusing" >&2
        return 1
    fi
    return 0
}
