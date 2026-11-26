"""moex/tools/gguf_vocab_only.py - a vocabulary-only copy of a GGUF: every metadata key, no tensors.

    python3 -I moex/tools/gguf_vocab_only.py SRC.gguf DST.gguf      (a venv with llama.cpp's gguf package)

llama.cpp's vocab_only load reads the metadata alone, so the copy serves tok_oracle (verify_tokenizer.py) and
export_tokenizer.py as the full file did. 2026-10-19: the DSV4 mini GGUF (11.3 GB) -> Strata-X-data/deepseek4-vocab-only.gguf
(5.3 MB, the same 71 keys; the export from it is byte-identical, the oracle check 0 differences) before the DSV4-Flash weights
were deleted.
"""
import sys

from gguf import GGUFReader, GGUFValueType, GGUFWriter

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
w = GGUFWriter(dst, r.fields["general.architecture"].contents())
n = 0
for name, f in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture":
        continue
    vt = f.types[0]
    w.add_key_value(name, f.contents(), vt, sub_type=f.types[-1] if vt == GGUFValueType.ARRAY else None)
    n += 1
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"{dst}: {n} keys, architecture {r.fields['general.architecture'].contents()}, {len(r.tensors)} tensors left out")
