#!/usr/bin/env python3
"""moex/tools/remote_gguf_types.py - whether strata-x runs a Hugging Face GGUF set, before downloading it (its headers only,
through HTTP range requests).  The rules are serve/quant_check.py's, the engine's own (moex/docs/QUANTS.md); this command
is kept for the notes that name it.
    python moex/tools/remote_gguf_types.py <hf-repo> [<folder-or-file>]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.quant_check import main  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        raise SystemExit(0)
    raise SystemExit(main(["--hf"] + sys.argv[1:]))
