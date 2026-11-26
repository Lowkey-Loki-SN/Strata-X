#!/usr/bin/env python3
"""moex/tools/fit.py - which MoE model a Strata-style engine would run best on THIS box, and how fast.

A roofline per decode window, calibrated on Strata's own measured run here (moex/box.json):
    t_window = fixed + max(t_gpu, t_cpu)        (Strata overlaps the CPU experts with the GPU's work)
    t_gpu    = (dense bytes + hit expert bytes x union(window)) / GPU bandwidth
    t_cpu    = (missed expert bytes x union(window)) / CPU expert-kernel bandwidth
    decode   = tokens per window / t_window            (no MTP: window 1; MTP: Strata's measured Qwen windows)
Cache = usable VRAM - overhead - dense - one MTP layer - KV window, ranked over (layer, expert) pairs.  Hit rates come
from per-prompt routing captures (moex/out/capture-<type>.json, made by cache_capture.py); a model without one borrows
GLM's curve and says so.  Ranges, not points: the band covers CPU bandwidth, the fixed cost and live-adaptivity uplift.

    python moex/tools/fit.py [--bpw 3.06] [--models moex/models] [--captures moex/out]
"""
import argparse, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def curve(capfile):
    """(fractions, adaptive hit) from a capture json, sorted, for interpolation."""
    c = json.load(open(capfile))["capture"]
    pts = sorted((v["frac"], v["adaptive"]) for v in c.values())
    return [p[0] for p in pts], [p[1] for p in pts]

def predict(d, box, bpw, cap, borrowed):
    mt = d["model_type"]; g = d["derived"]
    eb = g["expert_params"] * bpw / 8                       # bytes per expert
    dense_gb = box["gpu_dense_gb"][mt]
    cache_gib = box["usable_vram_gib"] - box["overhead_gib"] - dense_gb / 1.0737 - box["mtp_layer_gib"][mt] - box["kv_window_gib"][mt]
    slots = cache_gib * 2**30 / eb
    frac = min(1.0, slots / g["experts_total"])
    xs, ys = cap
    proxy = float(np.interp(frac, xs, ys))                  # adaptive-proxy hit at this cache share
    act = g["routed_active_params_per_token"] * bpw / 8      # routed expert bytes per token
    m = box["mtp"]
    out = {}
    for mode, (tok, win) in {"no MTP": (1.0, 1.0), "MTP": (m["tokens_per_pass"], m["window"])}.items():
        if mode == "MTP" and not d["mtp_layers"]:
            continue
        u = 1 + m["union_per_extra_token"] * (win - 1)
        vals = []
        for ratio in box["live_adaptivity_miss_ratio"]:
            h = 1 - (1 - proxy) * ratio
            for bw in box["cpu_bw_gbs"]:
                for fx in box["pass_fixed_ms"]:
                    t_gpu = (dense_gb * 1e9 + h * act * u) / (box["gpu_bw_gbs"] * 1e9)
                    t_cpu = (1 - h) * act * u / (bw * 1e9)
                    vals.append(tok / (fx / 1e3 + max(t_gpu, t_cpu)))
        out[mode] = (min(vals), max(vals))
    routed_gb = g["routed_params_total"] * bpw / 8 / 1e9
    ram_need = routed_gb + box["ram_other_gb"]
    max_bpw = (box["ram_gb"] - box["ram_other_gb"]) * 8e9 / g["routed_params_total"]
    return {"cache_gib": cache_gib, "slots": slots, "frac": frac, "proxy_hit": proxy, "borrowed_curve": borrowed,
            "tps": out, "routed_gb": routed_gb, "ram_need_gb": ram_need, "ram_fits": ram_need <= box["ram_gb"],
            "max_bpw_all_in_ram": max_bpw, "cpu_bytes_per_token_gb": (act / 1e9)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpw", type=float, nargs="*", default=[2.6, 3.06])
    ap.add_argument("--models", default=os.path.join(ROOT, "models")); ap.add_argument("--captures", default=os.path.join(ROOT, "out"))
    ap.add_argument("--box", default=os.path.join(ROOT, "box.json")); ap.add_argument("--json")
    a = ap.parse_args()
    box = json.load(open(a.box))
    models = {json.load(open(f))["model_type"]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.models, "*.json")))}
    caps = {os.path.basename(f)[len("capture-"):-5]: curve(f) for f in glob.glob(os.path.join(a.captures, "capture-*.json"))}
    # calibration check: Strata's own model at its measured setup (Q4_K_XL ~5.1 bpw average over its experts)
    q = models.get("qwen4_exp_text")
    if q and "qwen4_exp_text" in caps:
        r = predict(q, box, 5.1, caps["qwen4_exp_text"], False)
        lo, hi = r["tps"]["MTP"]
        print(f"calibration: Qwen3.8-Flash-Next @ Q4_K_XL predicted {lo:.0f}-{hi:.0f} T/s with MTP "
              f"(cache {r['cache_gib']:.1f} GiB = {r['frac']:.0%} of pairs) vs Strata measured {box['strata_qwen_measured_tps']} T/s")
    rows = []
    for mt, d in models.items():
        if mt == "qwen4_exp_text":
            continue
        cap, borrowed = (caps[mt], False) if mt in caps else (caps["glm5_next_text"], True)
        for b in a.bpw:
            r = predict(d, box, b, cap, borrowed)
            best = max(v[1] for v in r["tps"].values()); worst = max(v[0] for v in r["tps"].values())
            today = box["measured_today_tps"][mt]
            rows.append((mt, b, r, worst, best, today))
    print(f"\n{'model':16s} {'bpw':>4s} {'cache':>9s} {'pairs':>6s} {'hit*':>5s}  {'decode T/s (no MTP | MTP)':27s} {'today':>5s} {'speed-up':>9s}  RAM (all experts in RAM)")
    for mt, b, r, worst, best, today in rows:
        t = r["tps"]; nm = f"{t['no MTP'][0]:.0f}-{t['no MTP'][1]:.0f}"; wm = f"{t['MTP'][0]:.0f}-{t['MTP'][1]:.0f}" if "MTP" in t else "-"
        print(f"{mt:16s} {b:4.2f} {r['cache_gib']:6.1f}GiB {r['frac']:6.1%} {r['proxy_hit']:5.0%}{'^' if r['borrowed_curve'] else ' '} "
              f"{nm:>8s} | {wm:<15s}  {today:5.1f} {worst / today:4.1f}-{best / today:.1f}x  "
              f"{r['routed_gb']:.0f} GB experts -> {'fits' if r['ram_fits'] else 'DOES NOT FIT'} (max {r['max_bpw_all_in_ram']:.2f} bpw)")
    print("\n* adaptive-proxy hit at that cache share (a floor; the band applies Strata's live uplift)   "
          "^ no capture for this model yet: GLM's curve borrowed (moex/experiments/capture_routing.sh fills it)")
    if a.json:
        json.dump([{"model": mt, "bpw": b, **{k: v for k, v in r.items()}} for mt, b, r, *_ in rows], open(a.json, "w"), indent=1, default=str)

if __name__ == "__main__":
    main()
