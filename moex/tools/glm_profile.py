#!/usr/bin/env python3
"""moex/tools/glm_profile.py - an expert-routing profile for strata-x's VRAM expert tier.

    glm_profile.py <in.json> <out.txt>

<in.json> is exllamav3's split-stats file (EXL3_MOE_CPU_SPLIT_STATS, e.g. ~/AI/tabbyAPI/glm-routing-profile.json):
{"model.language_model.layers.<L>.mlp": [count per expert], ...}, decayed routing counts of the same GLM-5.3-Flash
measured on this box's real traffic.  The output ranks every (layer, expert) pair by count, hottest first, one
"<layer> <expert> <count>" line each - strata-x fills its slots from the top of the list (--expert-profile).
"""
import json, re, sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    d = json.load(open(sys.argv[1]))
    rows = []
    for k, counts in d.items():
        m = re.search(r"layers\.(\d+)\.mlp", k)
        if not m:
            continue
        layer = int(m.group(1))
        for e, c in enumerate(counts):
            rows.append((float(c), layer, e))
    rows.sort(key=lambda r: (-r[0], r[1], r[2]))
    # the model's shape in the header, as strata-x's own saved profile writes it: serve/server.py profile_shape reads it to
    # tell whether a learned profile (expert_profile_save) is of the same model, and without it the learned one was never used
    n_layer = max(r[1] for r in rows) + 1               # the trunk's layers: GLM-5.3-Flash 45 (the MTP block is not routed here)
    n_expert = max(len(c) for c in d.values() if isinstance(c, list))
    with open(sys.argv[2], "w") as f:
        f.write("# strata-x expert profile: n_layer %d n_expert %d, %d experts (layer expert count, hottest first; from %s)\n"
                % (n_layer, n_expert, len(rows), sys.argv[1]))
        for c, layer, e in rows:
            f.write(f"{layer} {e} {c:.3f}\n")
    layers = sorted({r[1] for r in rows})
    print(f"{len(rows)} pairs over {len(layers)} layers ({layers[0]}..{layers[-1]}) -> {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
