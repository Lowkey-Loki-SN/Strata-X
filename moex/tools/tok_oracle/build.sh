#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# moex/tools/tok_oracle/build.sh - build tok_oracle (the batch tokenizer oracle) against the SAME libllama as llama_dump:
# the CPU-only build of the glm5next-mtp llama.cpp tree in $STRATA_X_DATA/build/llama-cpu-exact (moex/ref/build.sh makes it).
# Vocabulary only: it never loads weights and never touches a GPU.  Idle priority on 4 cores, a memory cap.
set -euo pipefail
X=${STRATA_X_ROOT}
L=${STRATA_AI_DIR}/llama.cpp-glm5next-mtp
B=${STRATA_AI_DIR}/Strata-X-data/build/llama-cpu-exact
OUT=$X/moex/build
mkdir -p "$OUT"
[ -e "$B/bin/libllama.so" ] || { echo "missing $B/bin/libllama.so - run moex/ref/build.sh first" >&2; exit 1; }
RUN=(systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 -- chrt -i 0 nice -n 19 taskset -c 36-39)
"${RUN[@]}" g++ -O2 -std=c++17 -Wall -Wextra -I"$L/include" -I"$L/ggml/include" "$X/moex/tools/tok_oracle/tok_oracle.cpp" \
    -o "$OUT/tok_oracle" -L"$B/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$B/bin"
echo "built $OUT/tok_oracle"
