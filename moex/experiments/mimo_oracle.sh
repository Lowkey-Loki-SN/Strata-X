#!/usr/bin/env bash
STRATA_X_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"   # moex: this checkout (moex/README.md, Paths)
STRATA_AI_DIR="${STRATA_AI_DIR:-${STRATA_X_ROOT%/*}}"   # the folders beside it
# MiMo-V2.6-Flash: the 4-layer slice (layers 0-3: one global + dense layer, three SWA + MoE layers), llama.cpp's
# CPU dump of a short prompt, and the moex CPU reference checked op by op against it (teacher forcing).
# Needs shards 1-2 only (layers 0-3, the embedding and the head live there).  CPU only, idle priority.
set -uo pipefail
X=${STRATA_X_ROOT}; M=${STRATA_AI_DIR}/Models/GGUFs/mimo-v2.6-flash/BPW3.0; D=${STRATA_AI_DIR}/Strata-X-data/mimo-mini
RUN=(systemd-run --user --scope --quiet -p MemoryMax=24G -p MemorySwapMax=0 -- nice -n 10)
mkdir -p $D
if [ ! -f $D/mimo-mini-L0-3.gguf ]; then
    "${RUN[@]}" ${STRATA_AI_DIR}/llama.cpp/.venv/bin/python -I $X/moex/tools/gguf_slice.py \
        $M/MiMo-V2.6-Flash-MOPD-BPW3.0-00001-of-00004.gguf $M/MiMo-V2.6-Flash-MOPD-BPW3.0-00002-of-00004.gguf \
        $D/mimo-mini-L0-3.gguf --layers 0-3 2>&1 | tail -4
fi
P="The quick brown fox jumps over the lazy dog because it wanted to see what was on the other side of the hill"
$X/moex/build/llama_dump -m $D/mimo-mini-L0-3.gguf --tokenize "$P" 2>/dev/null > $D/p1.tok
echo "tokens: $(cat $D/p1.tok)"
rm -rf $D/dump-p1
"${RUN[@]}" $X/moex/build/llama_dump -m $D/mimo-mini-L0-3.gguf -o $D/dump-p1 -t "$(cat $D/p1.tok)" --ctx 512 \
    --threads 16 --no-repack --kv f32 2>&1 | grep -E '^llama_dump|rror'
cut -f2 $D/dump-p1/index.tsv | sed 's/-[0-9]*$//' | awk '!seen[$0]++' | tr '\n' ' ' | head -c 2500; echo
"${RUN[@]}" $X/moex/build/moex_ref -m $D/mimo-mini-L0-3.gguf --tokens-file $D/p1.tok --ctx 512 --threads 16 \
    --oracle $D/dump-p1 --tol 1e-5 2>&1 | tail -60
