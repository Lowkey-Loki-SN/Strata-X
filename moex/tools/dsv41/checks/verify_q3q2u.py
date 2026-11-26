"""verify_q3q2u.py - the built q3q2u pack: experts.tsv's 40 layers with the planned types and strides, every file exactly
384 records of its stride, and the MXFP4 down parts (layers 0-1, 15-19) bit-identical to the native records (8 experts a
layer, spread over the file)."""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import os
import sys

M = _os.path.join(_MODELS, 'dsv41-1017')
P = f"{M}/q3q2u"
Q2, Q3, MX = 10, 11, 39
plan = {}
for l in (0, 1, *range(15, 20)): plan[l] = (Q2, MX)
for l in range(2, 5): plan[l] = (Q2, Q2)
for l in (*range(5, 15), *range(20, 25)): plan[l] = (Q2, Q3)
for l in range(25, 40): plan[l] = (Q3, Q3)
row = {Q2: lambda n: n // 256 * 84, Q3: lambda n: n // 256 * 110, MX: lambda n: n // 32 * 17}
EMBD, FF, NE, SRC = 5120, 2304, 384, 18800640
last = {}
for line in open(f"{P}/experts.tsv"):
    il, ty, stride, rel = line.split()
    gu, _, d = ty.partition(",")
    last[int(il)] = (int(gu), int(d or gu), int(stride), rel)
bad = 0
for il in range(40):
    if il not in last:
        print(f"layer {il}: missing from experts.tsv"); bad += 1; continue
    gu, d, stride, rel = last[il]
    if (gu, d) != plan[il]:
        print(f"layer {il}: types {gu},{d} but the plan is {plan[il]}"); bad += 1
    blob = 2 * FF * row[gu](EMBD) + EMBD * row[d](FF)
    if stride != (blob + 4095) // 4096 * 4096:
        print(f"layer {il}: stride {stride} for a {blob}-byte blob"); bad += 1
    size = os.path.getsize(f"{P}/{rel}")
    if size != NE * stride:
        print(f"layer {il}: {size} bytes, not {NE} x {stride}"); bad += 1
    if d == MX:
        gu_new, gu_src, d_src = FF * row[gu](EMBD), FF * row[MX](EMBD), EMBD * row[MX](FF)
        with open(f"{M}/native/experts/L{il:02d}.bin", "rb") as fs, open(f"{P}/{rel}", "rb") as fn:
            for e in range(0, NE, NE // 8):
                fs.seek(e * SRC + 2 * gu_src); fn.seek(e * stride + 2 * gu_new)
                if fs.read(d_src) != fn.read(d_src):
                    print(f"layer {il} expert {e}: the MXFP4 down differs from the release"); bad += 1
total = sum(NE * last[il][2] for il in last)
print(f"{len(last)} layers, {total / 1e9:.2f} GB of records; {'all checks pass' if not bad else f'{bad} problems'}")
sys.exit(1 if bad else 0)
