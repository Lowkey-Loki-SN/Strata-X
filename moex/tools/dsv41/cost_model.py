"""moex/tools/dsv41/cost_model.py - decode / prefill model for DeepSeek-V4.1-Flash-1017 on this box (DSV41_PLAN.md §4).

    python cost_model.py            # the plan's tables
    python cost_model.py --anchors  # the measured runs the model is checked against

Box: 2x RTX 3080 20 GB, 125 GiB DDR4 (4 of 6 channels), Xeon Gold 6148, one NVMe on the chipset. Every input is a
measurement on this box or a byte count from the checkpoint headers. The hit curve is the open question, so every
speed is a band: pessimistic (V4's static learned profile from this box), optimistic (0xBakeer's measured V4.1 LRU
curve, one task) and the middle (the geometric mean of the two miss rates). Xolotl's measured V4.1 hit rates (exact
routing, adaptive arena, 53% resident: 87-90%) sit at the middle curve (--anchors).

One decode step verifies k rows (DSpark, acceptance `accept`); the hottest experts fill VRAM, then RAM; RAM hits and
misses are computed by the CPU pool, VRAM hits by the GPU; misses are read from the SSD:
  t_step = fixed + draft + max(GPU + CPU, SSD) * overlap + (GPU + CPU + SSD) * (1 - overlap)
With k = 2 the rows touch `union` x one row's experts; the experts both rows picked run at the CPU kernel's 2-token
rate, the rest at its 1-token rate (measured per format below).
"""
from __future__ import annotations

import argparse
import math

N_EXP, LAYERS, TOPK, N_PER_LAYER = 15360, 40, 6, 384
PARAMS_PER_EXP = 3 * 2304 * 5120          # 35.39M
NATIVE_MB = 18.80                          # one expert, FP4 + E8M0 (headers)

# CPU pool, weight GB/s for one expert and 1 or 2 tokens through it: native_expert_cpu_bench, 20 threads, clamp 10,
# 48 experts cycled from DRAM, quiet box (2026-10-18, after a model running in the background was stopped; V4-Flash
# shapes 4096 x 2048 from the DSV4 Q3_K file, requantized per STRATA_BENCH_REQUANT). Plain read of the same blobs:
# 75.7-78.6 GB/s.
FORMATS = {
    "fp4":     dict(bpw=4.25,   cpu1=74.36, cpu2=70.90),   # MXFP4 mx512 (the release's bits)
    "q3_K":    dict(bpw=3.4375, cpu1=74.45, cpu2=48.97),   # ggml's generic Q3_K path (no Strata fast path)
    "iq3_xxs": dict(bpw=3.0625, cpu1=50.07, cpu2=42.35),
    "q2_K":    dict(bpw=2.625,  cpu1=76.52, cpu2=57.67),
    "iq2_s":   dict(bpw=2.5,    cpu1=46.86, cpu2=33.55),
    "iq2_xxs": dict(bpw=2.0625, cpu1=39.02, cpu2=31.74),
}

SSD_GBS = 2.40          # O_DIRECT, 4 MiB reads, while 16 threads read DRAM at 74 GB/s (2026-10-17)
PCIE_PREFILL_GBS = 4.8  # prompt streaming of RAM experts to the GPUs, both cards taking turns (~4.5 GiB/s, MOEX.md:227)
DENSE_GB = 8.52         # dense + head read per token (headers; Q8_0 dense is 1.06x the FP8 bytes, within noise here)
GPU_GBS = 600.0         # effective of 760 spec; the two cards run their layers one after another
T_FIXED = 0.006         # syncs, graphs, sampling (Strata-X DSV4 decode budget, approx)

VRAM_EXPERT_GB = 2 * 20.0 * 1.0737 - 0.7 - 2.0 - DENSE_GB - 7.2 - 0.7 - 3.0   # minus dense, DSpark, KV + work
RAM_EXPERT_GB = (125 - 20 - 8) * 1.0737   # 20 GiB headroom (the configs), 8 GiB for engram rows / OS slack

V41_LRU = [(0.0, 0.0), (0.195, 0.805), (0.260, 0.855), (0.326, 0.891), (0.391, 0.917)]
V4_STATIC = [(0, 0), (.05, .209), (.10, .314), (.20, .469), (.25, .533), (.30, .59), (.40, .69), (.50, .776),
             (.75, .933), (.80, .954), (.90, .986), (1, 1)]
# Xolotl, V4.1 lookups covered by the experts ranked by calibration counts (V41_RESUME_PROMPT.md:1468-1469)
V41_STATIC_XOLOTL = [(0, 0), (0.112, 0.42), (0.450, 0.78), (0.562, 0.86)]

# Xolotl: a +180-token turn touches ~7,000 distinct experts, ~2,600 of them in its NVMe tier (47% of the experts)
# (HOW_IT_WORKS.md:43-44). Fitted: tokens act like 0.2146 x n independent ones; touched experts lean resident.
TURN_TOKEN_FACTOR = 0.2146
TURN_NVME_LEAN = (2600 / 7000) / 0.47


def _interp(pts, x):
    for (a, ya), (b, yb) in zip(pts, pts[1:]):
        if x <= b:
            return ya + (yb - ya) * (x - a) / (b - a)
    return pts[-1][1]


def hit(frac: float, curve: str) -> float:
    """Share of routings served from VRAM + RAM when the hottest `frac` of the experts are resident."""
    if curve == "v41-lru":
        if frac <= 0.391:
            return _interp(V41_LRU, frac)
        return 1 - 0.083 * math.exp(-4.38 * (frac - 0.391))     # log-linear tail fitted on the 4 points
    if curve == "v4-static":
        return _interp(V4_STATIC, frac)
    if curve == "v41-static":
        if frac <= 0.562:
            return _interp(V41_STATIC_XOLOTL, frac)
        return 1 - 0.14 * math.exp(-4.11 * (frac - 0.562))      # the last segment's slope in ln(miss)
    if curve == "mid":
        return 1 - math.sqrt((1 - hit(frac, "v41-lru")) * (1 - hit(frac, "v4-static")))
    raise ValueError(curve)


def mb_of(fmt: str) -> float:
    return NATIVE_MB if fmt == "fp4" else PARAMS_PER_EXP * FORMATS[fmt]["bpw"] / 8 / 1e6


def resident_frac(hot: str) -> tuple[float, float]:
    mb = mb_of(hot)
    n_v, n_r = VRAM_EXPERT_GB * 1e3 / mb, RAM_EXPERT_GB * 1e3 / mb
    return min(1.0, (n_v + n_r) / N_EXP), min(1.0, n_v / N_EXP)


def decode(hot: str, cold: str, curve: str, ssd: float = SSD_GBS, overlap: float = 0.7, k: int = 2,
           accept: float = 0.85, union: float = 1.7, draft_ms: float = 3.0) -> dict:
    """Static per-expert formats: the hottest experts (by profile) in `hot` fill VRAM + RAM, the rest are on the SSD
    in `cold`. k=1 is plain decode (no draft)."""
    if k == 1:
        union, draft_ms = 1.0, 0.0
    mb_h, mb_c = mb_of(hot), mb_of(cold)
    fh, fc = FORMATS[hot], FORMATS[cold]
    frac, frac_v = resident_frac(hot)
    h, hv = hit(frac, curve), hit(frac_v, curve)
    row = LAYERS * TOPK
    uniq = row * union                       # distinct experts a step touches
    shared = row * k - uniq                  # experts both rows picked (k <= 2)
    single = uniq - shared
    gpu = DENSE_GB / GPU_GBS * (1 + 0.08 * (k - 1)) + uniq * hv * mb_h / 1e3 / GPU_GBS

    def cpu_cost(n, mb, f):                  # seconds for n experts' worth of picks, split by tokens per expert
        return n * (shared / uniq * mb / f["cpu2"] + single / uniq * mb / f["cpu1"]) / 1e3

    cpu = cpu_cost(uniq * (h - hv), mb_h, fh) + cpu_cost(uniq * (1 - h), mb_c, fc)
    ssd_t = uniq * (1 - h) * mb_c / 1e3 / ssd
    comp = gpu + cpu
    t = T_FIXED + draft_ms / 1e3 + max(comp, ssd_t) * overlap + (comp + ssd_t) * (1 - overlap)
    toks = sum(accept ** i for i in range(k))
    disk = N_EXP * (frac * mb_h + (1 - frac) * mb_c) / 1e3
    avg_bpw = disk * 1e3 / N_EXP / (PARAMS_PER_EXP / 8 / 1e6)
    return dict(tps=toks / t, hit=h, resident=frac, disk_gb=disk, avg_bpw=avg_bpw, t_gpu=gpu, t_cpu=cpu,
                t_ssd=ssd_t)


def band(hot, cold, **kw) -> tuple[float, float, float]:
    return tuple(decode(hot, cold, c, **kw)["tps"] for c in ("v4-static", "mid", "v41-lru"))


def full_pass_s(hot: str, cold: str, ssd: float = SSD_GBS, layers: int = LAYERS) -> dict:
    """One pass of a prompt over `layers` layers' experts (a chunk over ~300 tokens touches ~all of a layer's 384).
    The SSD reads the non-resident ones while the RAM ones stream to the GPUs over PCIe; the slower side sets it."""
    frac, frac_v = resident_frac(hot)
    n = layers * N_PER_LAYER
    ssd_gb = n * (1 - frac) * mb_of(cold) / 1e3
    pcie_gb = n * (frac - frac_v) * mb_of(hot) / 1e3 + ssd_gb   # everything not in VRAM crosses PCIe once
    return dict(ssd_s=ssd_gb / ssd, pcie_s=pcie_gb / PCIE_PREFILL_GBS, s=max(ssd_gb / ssd, pcie_gb / PCIE_PREFILL_GBS))


def prompt_s(tokens: int, hot: str, cold: str, chunk: int = 8192, ssd: float = SSD_GBS) -> float:
    """Expert-streaming floor of a fresh prompt: layers 0-20 once per chunk, layers 21-39 once (exact replay of the
    last 128 x 20 = 2,560 tokens, Xolotl ds41_generate.cpp:11-12). Compute is not counted (it overlaps)."""
    chunks = max(1, math.ceil(tokens / chunk))
    return chunks * full_pass_s(hot, cold, ssd, 21)["s"] + full_pass_s(hot, cold, ssd, 19)["s"]


def short_turn_s(tokens: int, hot: str, cold: str, ssd: float = SSD_GBS) -> float:
    """A small agent turn: the experts its tokens route, read once; the non-resident ones from the SSD (fitted on
    Xolotl's +180-token turn); the CPU computes the RAM ones (not counted: it overlaps the reads)."""
    frac, _ = resident_frac(hot)
    touched = N_EXP * (1 - (1 - TOPK / N_PER_LAYER) ** (TURN_TOKEN_FACTOR * tokens))
    from_ssd = touched * min(1.0, (1 - frac) * TURN_NVME_LEAN)
    return from_ssd * mb_of(cold) / 1e3 / ssd


def decode_mix(mix: list[tuple[float, str]], curve: str, ssd: float = SSD_GBS, overlap: float = 0.7, k: int = 2,
               accept: float = 0.85, union: float = 1.7, draft_ms: float = 3.0) -> dict:
    """Formats chosen per layer (or per expert, independently of how hot the expert is): mix = [(share of the
    experts, format)]. The hottest experts fill VRAM + RAM at their own sizes; hits, misses and CPU time are the
    share-weighted averages."""
    if k == 1:
        union, draft_ms = 1.0, 0.0
    avg_mb = sum(sh * mb_of(f) for sh, f in mix)
    frac = min(1.0, (VRAM_EXPERT_GB + RAM_EXPERT_GB) * 1e3 / avg_mb / N_EXP)
    frac_v = min(1.0, VRAM_EXPERT_GB * 1e3 / avg_mb / N_EXP)
    h, hv = hit(frac, curve), hit(frac_v, curve)
    row = LAYERS * TOPK
    uniq = row * union
    shared = row * k - uniq
    single = uniq - shared
    per_expert_cpu = sum(sh * (shared / uniq * mb_of(f) / FORMATS[f]["cpu2"] + single / uniq * mb_of(f) / FORMATS[f]["cpu1"])
                         for sh, f in mix) / 1e3
    gpu = DENSE_GB / GPU_GBS * (1 + 0.08 * (k - 1)) + uniq * hv * avg_mb / 1e3 / GPU_GBS
    cpu = uniq * (1 - hv) * per_expert_cpu
    ssd_t = uniq * (1 - h) * avg_mb / 1e3 / ssd
    comp = gpu + cpu
    t = T_FIXED + draft_ms / 1e3 + max(comp, ssd_t) * overlap + (comp + ssd_t) * (1 - overlap)
    toks = sum(accept ** i for i in range(k))
    return dict(tps=toks / t, hit=h, resident=frac, disk_gb=N_EXP * avg_mb / 1e3,
                avg_bpw=avg_mb * 1e6 * 8 / PARAMS_PER_EXP)


def mix_turn_s(tokens: int, mix, ssd: float = SSD_GBS) -> float:
    avg_mb = sum(sh * mb_of(f) for sh, f in mix)
    frac = min(1.0, (VRAM_EXPERT_GB + RAM_EXPERT_GB) * 1e3 / avg_mb / N_EXP)
    touched = N_EXP * (1 - (1 - TOPK / N_PER_LAYER) ** (TURN_TOKEN_FACTOR * tokens))
    return touched * min(1.0, (1 - frac) * TURN_NVME_LEAN) * avg_mb / 1e3 / ssd


# Xolotl's held-out output error per layer at ~2 bpw: layers 20-31 0.03-0.44%, 33-39 0.13-2.1% (insensitive);
# 4-19 1.9-12.9% and layer 32 14.8% (sensitive); 0-3 2.4-6.8% (tools/v41/hf/README.md:134-147).
SENSITIVE = 4 + 16 + 1          # layers 0-19 and 32
MIXES = [
    ("Q3_K everywhere (the DSV4 recipe)", [(1.0, "q3_K")]),
    ("Q3_K layers 0-19 + 32, Q2_K the other 19", [(SENSITIVE / 40, "q3_K"), (19 / 40, "q2_K")]),
    ("Q3_K layers 4-19 + 32, Q2_K the other 23", [(17 / 40, "q3_K"), (23 / 40, "q2_K")]),
    ("Q3_K layers 12-13 + 32, Q2_K the other 37", [(3 / 40, "q3_K"), (37 / 40, "q2_K")]),
    ("Q2_K everywhere", [(1.0, "q2_K")]),
    ("Q3_K 0-19 + 32, IQ2_XXS the other 19", [(SENSITIVE / 40, "q3_K"), (19 / 40, "iq2_xxs")]),
    ("Q2_K 0-19 + 32, IQ2_XXS the other 19 (~2.3 bpw)", [(SENSITIVE / 40, "q2_K"), (19 / 40, "iq2_xxs")]),
]

ROWS = [
    ("FP4 / FP4 (the release's bits, lossless)", "fp4", "fp4"),
    ("Q3_K / Q3_K (the DSV4 Q3_K recipe)", "q3_K", "q3_K"),
    ("Q3_K / Q2_K", "q3_K", "q2_K"),
    ("IQ3_XXS / IQ3_XXS", "iq3_xxs", "iq3_xxs"),
    ("Q3_K / IQ2_XXS", "q3_K", "iq2_xxs"),
    ("Q2_K / Q2_K", "q2_K", "q2_K"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", action="store_true")
    ap.add_argument("--hardware-levers", action="store_true",
                    help="reference only (DSV41_PLAN.md §10): the same packs with ~8.9 GB/s of NVMe")
    a = ap.parse_args()
    if a.anchors:
        print("0xBakeer V4.1 on a DGX Spark, 25.6% resident: measured hit 0.83, 0.92 GB read per token")
        print(f"   model (v41-lru): hit {hit(0.256, 'v41-lru'):.3f}")
        print("Strata-X V4 file-tier emulation: ~8 first-touch reads/token x 13.37 MB at 2.75 GB/s, base 20.99 tok/s")
        print(f"   model: {1 / (1 / 20.99 + 8 * 13.37e-3 / 2.75):.1f} tok/s, measured 10.21 (cold LRU)")
        print("Xolotl V4.1, exact routing, ~53% resident (VRAM ~1,450 + arena ~6,700): NVMe reads per token measured")
        print("   24-30 with the adaptive arena (hit 0.875-0.90), 48-85 with its static arena (0.65-0.80)")
        print(f"   model at 53%: mid {hit(0.53, 'mid'):.3f}, v4-static {hit(0.53, 'v4-static'):.3f}, "
              f"v41-lru {hit(0.53, 'v41-lru'):.3f}, Xolotl's calibration-count coverage {hit(0.53, 'v41-static'):.3f}")
        return
    if a.hardware_levers:
        print("REFERENCE ONLY (DSV41_PLAN.md §10): middle estimates with one NVMe vs ~8.9 GB/s of NVMe")
        for name, mix in MIXES:
            print(f"   {name:48s} decode {decode_mix(mix, 'mid')['tps']:5.1f} -> {decode_mix(mix, 'mid', ssd=8.9)['tps']:5.1f}"
                  f"   +200-token turn {mix_turn_s(200, mix):4.1f} -> {mix_turn_s(200, mix, 8.9):3.1f} s")
        return
    fits = (VRAM_EXPERT_GB + RAM_EXPERT_GB) * 1e3 / N_EXP
    print(f"expert memory: VRAM {VRAM_EXPERT_GB:.1f} GB + RAM {RAM_EXPERT_GB:.1f} GB; every expert resident needs "
          f"<= {fits:.2f} MB per expert = {fits * 1e6 * 8 / PARAMS_PER_EXP:.2f} bpw")
    print("\ndecode tok/s with DSpark (k=2), the NVMe at 2.4 GB/s: middle [pessimistic .. optimistic]; plain middle in ()")
    print(f"{'pack (hot = resident / cold = SSD)':40s} {'bpw':>5s} {'pack GB':>7s} {'resident':>8s} {'decode':>27s}")
    for name, hot, cold in ROWS:
        r = decode(hot, cold, "mid")
        p1 = decode(hot, cold, "mid", k=1)["tps"]
        b1 = band(hot, cold)
        print(f"{name:40s} {r['avg_bpw']:5.2f} {r['disk_gb']:7.0f} {r['resident']:8.0%} "
              f"{b1[1]:5.1f} [{b1[0]:4.1f}..{b1[2]:4.1f}] ({p1:4.1f})")
    print("\nper-layer mixes, decode tok/s with DSpark: middle [pessimistic .. optimistic]; plain middle in (); "
          "+200-token turn (s)")
    for name, mix in MIXES:
        r = decode_mix(mix, "mid")
        b = [decode_mix(mix, c)["tps"] for c in ("v4-static", "v41-lru")]
        p = decode_mix(mix, "mid", k=1)["tps"]
        print(f"   {name:48s} {r['avg_bpw']:4.2f} bpw {r['disk_gb']:4.0f} GB {r['resident']:4.0%} res  "
              f"{r['tps']:5.1f} [{b[0]:4.1f}..{b[1]:4.1f}] ({p:4.1f})   turn {mix_turn_s(200, mix):4.1f} s")
    print("\nprompts, expert-streaming floor (s): a +200-token agent turn, a fresh 8K and 32K prompt")
    for name, hot, cold in ROWS:
        print(f"   {name:40s} +200 tok {short_turn_s(200, hot, cold):5.1f}   8K {prompt_s(8192, hot, cold):5.1f}   "
              f"32K {prompt_s(32768, hot, cold):6.1f}")


if __name__ == "__main__":
    main()
