"""moex/tools/dsv41/speed_model.py - decode speed of a DeepSeek-V4.1 pack on this box, from measured parts (DSV41_PLAN §11).

    python speed_model.py --trace TRACE[:FROM:TO] --calib TRACE[:FROM:TO] [--packs native,q3q2-217,...] [--overlap 0|1]

The parts, each measured on this box (decode, plain windows of one token, the engine's tiers):
  - where each routed expert is: tier_sim.py replays a routing trace (STRATA_ROUTE_TRACE) against the pack's tiers -
    VRAM slots per GPU (a slot holds the pack's largest record), a fixed RAM copy and the LRU SSD cache (its slots are
    the largest record too) - giving per layer of every token the VRAM routings and the CPU experts from RAM, from the
    cache and read from the SSD;
  - the GPU's own time per token (attention, dense projections, VRAM experts, the host's doorbell round trips): the
    engine's router wait;
  - the CPU pool: bytes / its effective rate, plus a fixed cost per pool call (park, drain, re-park);
  - the SSD: bytes / its demand-read rate (O_DIRECT 4 MiB pieces, 12 readers).
Per layer the host computes the RAM experts - and, with --overlap 1 (run_cpu_jobs since 2026-10-18), the cache's
records too - while the missing records are read, then the read ones:
  t_layer = max(cpu(now), ssd(reads)) + cpu(read)        t_token = gpu + sum over layers of t_layer
The constants come from a calibration run of the engine (its printed per-token numbers); the model is then checked
against runs it was not fitted to.
"""
from __future__ import annotations

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tier_sim  # noqa: E402

MB = 1e6
# record bytes per format (requant_pack: [gate | up | down] padded to 4 KiB)
REC = {"fp4": 18_800_640, "q3_k": 15_208_448, "q2_k": 11_612_160}
PACKS = {   # per-layer formats (DSV41_PLAN §6; the user's choice: Q3_K on 4-19 and 32, Q2_K on the other 23)
    "native": ["fp4"] * 40,
    "q3q2-202": ["q3_k" if (4 <= il <= 19 or il == 32) else "q2_k" for il in range(40)],
    "q3q2-207": ["q3_k" if (il <= 19 or il == 32) else "q2_k" for il in range(40)],
    "q2-178": ["q2_k"] * 40,
    "q3q2-217": ["q2_k" if 20 <= il <= 31 else "q3_k" for il in range(40)],   # the pack after layers 33-39 went to Q3_K
}
# the CPU kernels' 1-token weight rates relative to MXFP4 (native_expert_cpu_bench, cost_model.py FORMATS)
CPU_REL = {"fp4": 1.0, "q3_k": 74.45 / 74.36, "q2_k": 76.52 / 74.36}


def tiers(pack: list[str], vram_slots=(1011, 783), ram_gib=15.37, cache_gib=60.0, split=20) -> dict:
    """The tier sizes in records, as the engine's log gives them: its VRAM slots per GPU ("GPU d: N expert slots"), its RAM
    copy's GiB ("... GiB in RAM (pinned)", the exchange blobs left out) filled by bytes in layer order after the VRAM fill
    (no profile: expert 0 of every layer, then 1, ...), and the SSD cache's per-size pools.  (Until 2026-10-19 the model took
    the native pack's 18.17 GiB and slots from GB figures: it predicted 4.1-4.8 SSD reads a token where the 217 GB pack's
    engine read 9.5 - with the engine's tiers and swap rules: 9.1.)"""
    nl = len(pack)
    vr = [[False] * tier_sim.NE for _ in range(nl)]
    used = [0, 0]
    for e in range(tier_sim.NE):
        for il in range(nl):
            g = 0 if il < split else 1
            if used[g] < vram_slots[g]:
                vr[il][e] = True
                used[g] += 1
    budget, ram = ram_gib * 2**30, 0
    for il in range(nl):
        for e in range(tier_sim.NE):
            if vr[il][e]:
                continue
            if budget < REC[pack[il]]:
                break
            budget -= REC[pack[il]]
            ram += 1
        else:
            continue
        break
    pools = tier_sim.PoolCache([REC[f] for f in pack], cache_gib)
    return dict(vram0=vram_slots[0], vram1=vram_slots[1], ram=ram, cache=sum(pools.cap.values()), cache_gib=cache_gib)


def replay(pack, calib, ev, t: dict, overlap: bool, c: dict) -> dict:
    """Replays `ev` after `calib` with the pack's tiers; the per-layer timing with constants `c`."""
    class A:   # tier_sim's options
        pass
    a = A()
    a.vram0, a.vram1, a.split, a.ram, a.cache = t["vram0"], t["vram1"], 20, t["ram"], t["cache"]
    a.vram_excl = t.get("excl", ())
    # the engine's swap rules (src/x/tiers.cpp swap_round): 17 a round for 14.5 MiB slots (256 MiB / slot, 8-96), at most 8 of
    # them bringing in an expert with no RAM copy
    a.adapt_every, a.decay, a.swaps, a.cache_decay, a.prior, a.margin, a.ram_decay = 4, 0.7, 17, 0.9999, 0.25, 1.0, 0.995
    a.ssd_swaps = 8
    a.pool_cache = tier_sim.PoolCache([REC[f] for f in pack], t.get("cache_gib", 60.0))
    sim = tier_sim.Sim(a, None, "static", "adaptive", ram=t["ram"], cache=t["cache"])
    acc = dict(t_layers=0.0, cpu=0.0, ssd=0.0, vram=0, ram=0, cache=0, reads=0, calls=0)

    def on_layer(il, T, nv, nr, nc, ns):
        b = REC[pack[il]]
        rate = c["cpu_gbs"] * CPU_REL[pack[il]] * 1e9
        now = nr + (nc if overlap else 0)
        later = ns + (0 if overlap else nc)
        t_now = (now * b / rate + c["call_ms"] / 1e3) if now else 0.0
        t_ssd = ns * b / (c["ssd_gbs"] * 1e9)
        t_later = (later * b / rate + c["call_ms"] / 1e3) if later else 0.0
        acc["t_layers"] += max(t_now, t_ssd) + t_later
        acc["cpu"] += t_now + t_later
        acc["ssd"] += t_ssd
        acc["vram"] += nv
        acc["ram"] += nr
        acc["cache"] += nc
        acc["reads"] += ns
        acc["calls"] += (1 if now else 0) + (1 if later else 0)

    for w in calib:
        sim.window(w, False)
    sim.on_layer = on_layer
    for w in ev:
        sim.window(w, True)
    n = max(1, sim.tokens)
    ms = {k: (v * 1e3 / n if isinstance(v, float) else v / n) for k, v in acc.items()}
    ms["token"] = c["gpu_ms"] + ms["t_layers"]
    ms["tok_s"] = 1e3 / ms["token"]
    return ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", nargs="*", default=[], help="the trace the tiers run through first (a server that has run)")
    ap.add_argument("--trace", nargs="+", required=True, help="the trace replayed and timed")
    ap.add_argument("--packs", default="native,q3q2-217")   # the production pack (DSV41_BACKLOG #57)
    ap.add_argument("--overlap", type=int, default=1)
    # the constants: R3 (2026-10-18, native, 60 GiB cache, coding text, plain): 265 ms a token, router wait 33.5 ms,
    # pool 231.5 ms of which 180 ms waiting for 26.5 SSD reads; 147 CPU experts a token
    ap.add_argument("--gpu-ms", type=float, default=33.5)
    ap.add_argument("--cpu-gbs", type=float, default=58.0)
    ap.add_argument("--call-ms", type=float, default=0.05)
    ap.add_argument("--ssd-gbs", type=float, default=2.76)
    ap.add_argument("--cache-gib", type=float, default=60.0)
    ap.add_argument("--ram-gib", type=float, default=17.29, help="the engine log's '... GiB in RAM (pinned)' (2026-10-19 21:14)")
    ap.add_argument("--vram-slots", default="979,815", help="the engine log's expert slots per GPU (2026-10-19 21:14)")
    a = ap.parse_args()
    calib = [w for p in a.calib for w in tier_sim.read_trace(p)]
    ev = [w for p in a.trace for w in tier_sim.read_trace(p)]
    c = dict(gpu_ms=a.gpu_ms, cpu_gbs=a.cpu_gbs, call_ms=a.call_ms, ssd_gbs=a.ssd_gbs)
    for name in a.packs.split(","):
        pack = PACKS[name]
        t = tiers(pack, vram_slots=tuple(int(x) for x in a.vram_slots.split(",")), ram_gib=a.ram_gib, cache_gib=a.cache_gib)
        r = replay(pack, calib, ev, t, bool(a.overlap), c)
        print(f"{name:10s} tiers {t} | per token: VRAM {r['vram']:.1f} RAM {r['ram']:.1f} cache {r['cache']:.1f} "
              f"SSD reads {r['reads']:.1f} | CPU {r['cpu']:.1f} ms, SSD {r['ssd']:.1f} ms, layers {r['t_layers']:.1f} ms "
              f"+ GPU {a.gpu_ms:.1f} = {r['token']:.1f} ms = {r['tok_s']:.2f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
