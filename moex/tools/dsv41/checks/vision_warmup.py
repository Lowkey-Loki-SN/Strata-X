"""vision_warmup.py [PICTURE] - DSV41_BACKLOG #73 / #58: does the encoder's warm-up make a start's first real picture the
same as its later ones?  Three encoder processes on the config's vision block, 20 threads as the server runs them:
  A: no warm-up - the picture twice (its first pass against its second: #58's difference, if any);
  B, C: STRATA_DSV41_VISION_WARMUP=1 - the picture once each.
PASS when B's and C's rows are bitwise A's second pass.  CPU-heavy (~40 s a picture): run it with the server stopped."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

X = Path(_REPO)
pic = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(_os.path.join(_MODELS, 'dsv41-1017/ref-runs/engine-checks/v43-number.png'))
v = json.loads((X / "strata-dsv41.json").read_text())["vision"]
env_cfg = json.loads((X / "strata-dsv41.json").read_text()).get("env") or {}


def encode(times, warm):
    env = dict(os.environ)
    env.update({k: str(val) for k, val in env_cfg.items() if k.startswith("STRATA_DSV41_VISION")})
    env["STRATA_DSV41_VISION_WARMUP"] = "1" if warm else "0"
    p = subprocess.Popen([v["python"], v["exe"], "--mmproj", v["mmproj"], "--model", v["model"], "--threads", str(v["threads"])],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, env=env, cwd=str(X / "serve"))
    assert p.stdout.readline().strip() == "READY"
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for i in range(times):
            out = Path(d) / f"r{i}.sve"
            p.stdin.write(f"ENC {pic} {out}\n")
            p.stdin.flush()
            line = p.stdout.readline().strip()
            assert line.startswith("OK"), line
            rows.append(np.frombuffer(out.read_bytes()[20:], dtype=np.float32).copy())
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    p.wait(timeout=60)
    return rows


a = encode(2, warm=False)
b = encode(1, warm=True)[0]
c = encode(1, warm=True)[0]
print(f"A first vs A second: {'same' if np.array_equal(a[0], a[1]) else 'differ (max %.3g)' % np.abs(a[0] - a[1]).max()}")
print(f"B vs A second: {'same' if np.array_equal(b, a[1]) else 'differ (max %.3g)' % np.abs(b - a[1]).max()}")
print(f"C vs A second: {'same' if np.array_equal(c, a[1]) else 'differ (max %.3g)' % np.abs(c - a[1]).max()}")
print("PASS" if np.array_equal(b, a[1]) and np.array_equal(c, a[1]) else "FAIL - the warm-up does not fix the first pass")
