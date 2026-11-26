#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# moex/tools/chat_oracle/build_common.sh - build llama.cpp's `llama-common` (the chat template engine, the PEG output
# parsers and the auto-parser: common/chat*.cpp, common/jinja, common/peg-parser.cpp) CPU-only, in a build directory of its
# own, from the glm5next-mtp tree.  Not the oracle's build directory (llama-cpu-exact): that one stays as it is, because
# llama_dump and tok_oracle link its libllama.  No GPU backend is built, no weights are loaded by anything here.
# Idle priority on 4 cores, a memory cap, as moex/ref/build.sh.
set -euo pipefail
L=${STRATA_AI_DIR}/llama.cpp-glm5next-mtp
B=${STRATA_AI_DIR}/Strata-X-data/build/llama-cpu-common
RUN=(systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 -- chrt -i 0 nice -n 19 taskset -c 36-39)
if [ ! -e "$B/build.ninja" ]; then
    cmake -G Ninja -S "$L" -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
        -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=ON -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF
fi
"${RUN[@]}" ninja -C "$B" -j4 llama-common
echo "built $B"
