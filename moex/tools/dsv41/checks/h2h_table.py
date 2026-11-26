"""h2h_table.py DIR - the head-to-head's runs (h2h_engines.py's PREFIX.json files in DIR) side by side: ready, the first read,
the follow-ups, the decode, the drive."""
import json, pathlib, sys
d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
rows = []
for f in sorted(d.glob(sys.argv[2] if len(sys.argv) > 2 else "full-*.json")):
    r = json.loads(f.read_text())
    t = r["turns"]
    if not t:
        continue
    fu = t[1:]
    dec = [x["decode_tok_s"] for x in t if x["generated"] > 1]
    rows.append((f.stem.replace("full-", ""), r["ready_s"], t[0]["prompt"], t[0]["read_s"], t[0]["prompt"] / t[0]["read_s"] if t[0]["read_s"] else 0,
                 sum(x["read_s"] for x in fu) / len(fu) if fu else 0, sum(dec) / len(dec) if dec else 0,
                 min(dec) if dec else 0, max(dec) if dec else 0, sum(x["drive_gb"] for x in t), sum(x["slow_s"] for x in t),
                 t[0]["tokens"][:8]))
print("%-12s %7s %8s %9s %8s %10s %9s %13s %9s %6s  %s" % ("engine", "ready s", "turn 0", "read s", "tok/s", "follow-up s",
      "decode", "(min-max)", "drive GB", "slow s", "first tokens"))
for e, rd, p0, r0, tps, fus, dm, dmin, dmax, gb, sl, tk in rows:
    print("%-12s %7.1f %8d %9.1f %8.1f %10.1f %9.2f %6.2f-%-6.2f %9.1f %6d  %s" % (e, rd, p0, r0, tps, fus, dm, dmin, dmax, gb, sl, tk))
