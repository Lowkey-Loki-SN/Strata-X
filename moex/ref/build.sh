#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# moex/ref/build.sh - build glm_ref against the CPU-only ggml of the glm5next llama.cpp tree (the oracle's own
# ggml, so the reference and llama_dump share every dot-product kernel).  Also builds llama_dump.
#
# The CPU build lives in $STRATA_X_DATA/build/llama-cpu-exact; this script configures and builds it if missing.
# "exact": GGML_LLAMAFILE=OFF and GGML_CPU_REPACK=OFF, so ggml_mul_mat on one token is ggml-cpu's plain vec_dot
# per row - the very function glm_ref calls - and the two differ only where an elementwise op is vectorised.
# Idle priority, 4 cores, a memory cap: the live Strata server keeps the machine.
set -euo pipefail
X=${STRATA_X_ROOT}
L=${STRATA_AI_DIR}/llama.cpp-glm5next-mtp
B=${STRATA_AI_DIR}/Strata-X-data/build/llama-cpu-exact
OUT=$X/moex/build
mkdir -p "$OUT"
RUN=(systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 -- chrt -i 0 nice -n 19 taskset -c 36-39)

if [ ! -e "$B/bin/libggml-cpu.so" ]; then
    cmake -G Ninja -S "$L" -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
        -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=ON -DGGML_LLAMAFILE=OFF -DGGML_CPU_REPACK=OFF
    "${RUN[@]}" ninja -C "$B" -j4 llama ggml
fi

INC=(-I"$L/include" -I"$L/ggml/include")
LIBS=(-L"$B/bin" -lggml-cpu -lggml-base -Wl,-rpath,"$B/bin")
"${RUN[@]}" g++ -O2 -march=native -std=c++17 -Wall -Wextra -Wno-unused-parameter "${INC[@]}" \
    "$X/moex/ref/gguf_file.cpp" "$X/moex/ref/recorder.cpp" "$X/moex/ref/glm_ref.cpp" "$X/moex/ref/glm_ref_main.cpp" \
    -o "$OUT/glm_ref" "${LIBS[@]}" -lpthread
"${RUN[@]}" g++ -O2 -std=c++17 "${INC[@]}" "$X/moex/tools/llama_dump/llama_dump.cpp" -o "$OUT/llama_dump" \
    -L"$B/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$B/bin"
"${RUN[@]}" g++ -O2 -march=native -std=c++17 -Wall -Wextra -Wno-unused-parameter "${INC[@]}" \
    "$X/moex/ref/gguf_file.cpp" "$X/moex/ref/recorder.cpp" "$X/moex/ref/glm_ref.cpp" "$X/moex/ref/mimo_ref.cpp" \
    "$X/moex/ref/ref_main.cpp" -o "$OUT/moex_ref" "${LIBS[@]}" -lpthread
echo "built $OUT/glm_ref $OUT/moex_ref $OUT/llama_dump"
