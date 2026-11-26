"""moex/tools/dsv41/tier_sim.py - replays decode routing traces against expert tier policies: SSD reads per token.

    python tier_sim.py --calib TRACE[:FROM:TO] ... --eval TRACE[:FROM:TO] ... [--vram0 842 --vram1 640 --split 20]
                       [--ram 4290] [--cache 64] [--policies all]

A trace is strata-x's STRATA_ROUTE_TRACE file: int32 records (layer, T, k, then T x k expert ids), one per MoE layer call
of each decode window (a prompt chunk's calls, marked layer + 1000, are skipped: prompt_cache_sim.py replays them). FROM:TO
picks windows by number (a window ends where the layer number drops). The calibration traces give the profile (routing
counts); the evaluation traces are replayed, after the calibration traces when a policy adapts
(the tiers start where the calibration left them, as a server that has run). Every policy counts, per evaluated token,
the routed experts in VRAM, in RAM, served by the SSD cache, and read from the SSD.

The tiers, as the engine has them (src/x/tiers.cpp, src/x/dsv41.cpp):
  - VRAM: per GPU, slots for its own layers (layers < split on GPU 0). Strata's adaptive rule: heat += 1 per routing,
    times `decay` every `adapt_every` tokens (Strata: 0.7 every 4); a swap round every `adapt_every` tokens trades the
    i-th hottest non-resident (heat >= 2) for the i-th coldest resident when hotter by 1.5, at most `swaps` per round.
  - RAM: a static set (the profile's hottest of the complement, or layer order without one), or adaptive (lru / heat).
  - SSD cache: LRU slots of whole records; a window's distinct misses per layer are read once.
  --engine-swaps (DSV41_PLAN §64): the swaps as the engine makes them (src/x/tiers.cpp swap_round, STRATA_DSV41_SWAP_CACHE=1):
    the round's plan cut to `swaps` first; an incoming expert with no RAM copy only from the SSD cache (one the cache does
    not hold waits for a later round), at most 8 such a round; its cache slot freed, and a victim with no RAM copy into a
    cache slot of its own.  Without it an incoming expert comes from nowhere and a victim is dropped (the runs before §64).
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import numpy as np

NE = 384
NL = 40


def read_trace(path: str):
    """[(window: list of (layer, T, ids[T, k]))] - a window ends where the layer number drops."""
    spec = path.split(":")
    p, lo, hi = spec[0], None, None
    if len(spec) == 3:
        lo, hi = int(spec[1]), int(spec[2])
    a = np.fromfile(p, dtype=np.int32)
    out, cur, last, i = [], [], -1, 0
    while i + 3 <= a.size:
        il, T, k = int(a[i]), int(a[i + 1]), int(a[i + 2])
        i += 3
        if i + T * k > a.size:   # a trace still being written
            break
        ids = a[i:i + T * k].reshape(T, k)
        i += T * k
        if il >= 1000:   # a prompt chunk's layer (strata-x traces them as layer + 1000 since 2026-10-22): not a decode window
            continue
        if il <= last and cur:
            out.append(cur)
            cur = []
        cur.append((il, T, ids))
        last = il
    if cur:
        out.append(cur)
    if lo is not None:
        out = out[lo:hi]
    return out


def read_profile(path: str) -> np.ndarray:
    """A strata-x expert profile file ('layer expert count' lines, '#' header) as routing counts (DSV41_PLAN §66)."""
    c = np.zeros(NL * NE, dtype=np.float64)
    for line in pathlib.Path(path).read_text().splitlines():
        f = line.split()
        if len(f) == 3 and not line.startswith("#"):
            il, e = int(f[0]), int(f[1])
            if 0 <= il < NL and 0 <= e < NE:
                c[il * NE + e] = float(f[2])
    return c


def counts_of(windows) -> np.ndarray:
    c = np.zeros(NL * NE, dtype=np.float64)
    for w in windows:
        for il, T, ids in w:
            np.add.at(c, il * NE + ids.reshape(-1), 1.0)
    return c


class PoolCache:
    """The engine's SSD cache with one LRU pool per record size (src/x/dsv41.cpp rec_io_init, --ssd-cache-gib): each pool its
    share of the bytes by its layers' record bytes; a layer's records go to its pool.  `layer_bytes`: [layer] record bytes."""

    def __init__(self, layer_bytes, total_gib: float):
        strides = sorted(set(layer_bytes))
        share = {s: sum(b for b in layer_bytes if b == s) for s in strides}
        tot = sum(share.values())
        self.cap = {s: int(total_gib * 2**30 * share[s] / tot / s) for s in strides}
        self.layer = list(layer_bytes)
        self.lru = {s: collections.OrderedDict() for s in strides}

    def _p(self, key):
        return self.lru[self.layer[key // NE]]

    def __contains__(self, key):
        return key in self._p(key)

    def move_to_end(self, key):
        self._p(key).move_to_end(key)

    def add(self, key):
        p = self._p(key)
        p[key] = None
        if len(p) > self.cap[self.layer[key // NE]]:
            p.popitem(last=False)

    def pop(self, key, default=None):
        return self._p(key).pop(key, default)

    def __len__(self):
        return sum(len(p) for p in self.lru.values())


class Sim:
    def __init__(self, a, profile: np.ndarray | None, ram_policy: str, vram_policy: str, ram: int | None = None,
                 cache: int | None = None, cache_policy: str = "lru"):
        self.a = a
        self.cache_policy = cache_policy
        # lfu: decayed routing counts of every key (time = routings fed); 2q: the probation FIFO, its ghost list, the LRU
        self.t = 0
        self.fc = np.zeros(NL * NE)
        self.ft = np.zeros(NL * NE)
        self.incache = np.zeros(NL * NE, dtype=bool)
        self.a1 = collections.OrderedDict()
        self.a1out = collections.OrderedDict()
        self.ram_n = a.ram if ram is None else ram
        self.cache_n = a.cache if cache is None else cache
        self.ram_policy = ram_policy
        self.vram_policy = vram_policy
        self.heat = np.zeros(NL * NE, dtype=np.float64)
        self.freq = (profile.copy() * a.prior if profile is not None else np.zeros(NL * NE)) if ram_policy == "lfu" else None
        rank = np.argsort(-profile, kind="stable") if profile is not None else np.arange(NL * NE)
        # VRAM: the profile's hottest per GPU (no profile: expert 0 of every layer, then 1, ... - src/x/tiers.cpp)
        self.vram = np.zeros(NL * NE, dtype=bool)
        cap = [a.vram0, a.vram1]
        used = [0, 0]
        gpu = lambda key: 0 if key // NE < a.split else 1
        self.excl = set(getattr(a, "vram_excl", ()) or ())
        order = rank if profile is not None else np.array([il * NE + e for e in range(NE) for il in range(NL)])
        for key in order:
            g = gpu(key)
            if int(key) // NE in self.excl:
                continue
            if used[g] < cap[g]:
                self.vram[key] = True
                used[g] += 1
        self.vcap = cap
        # RAM: the hottest of the complement by the profile (no profile: layer order)
        self.ram = np.zeros(NL * NE, dtype=bool)
        n = 0
        if getattr(a, "ram_fair", False) and profile is not None:
            # --ram-fair (Maya's DS4_COPY_FAIR, 1243b70; DSV41_PLAN §66): each layer's hottest of the complement in turn - the
            # layers' first, then their second, ... - instead of one order over all the layers
            per = [[] for _ in range(NL)]
            for key in rank:
                if not self.vram[key]:
                    per[int(key) // NE].append(int(key))
            r = 0
            while n < self.ram_n and any(r < len(q) for q in per):
                for q in per:
                    if r < len(q) and n < self.ram_n:
                        self.ram[q[r]] = True
                        n += 1
                r += 1
        for key in rank:
            if n >= self.ram_n:
                break
            if not self.vram[key] and not self.ram[key]:
                self.ram[key] = True
                n += 1
        self.ram_clock = collections.OrderedDict()   # ram lru: key -> None, oldest first
        for key in np.nonzero(self.ram)[0]:
            self.ram_clock[int(key)] = None
        self.cache = collections.OrderedDict()   # ssd cache lru (or a.pool_cache: the engine's per-size pools)
        if getattr(a, "pool_cache", None) is not None:
            self.cache = a.pool_cache
        self.tokens = 0
        self.stat = collections.Counter()
        self.on_layer = None   # f(il, T, vram routings, ram experts, cache experts, ssd reads): speed_model.py

    def window(self, w, count: bool):
        a = self.a
        T = max(t for _, t, _ in w)
        for il, T_, ids in w:
            keys = (il * NE + ids.reshape(-1)).tolist()
            if self.cache_policy == "lfu":
                for key in keys:
                    self.t += 1
                    self.fc[key] = self.fc[key] * a.cache_decay ** (self.t - self.ft[key]) + 1.0
                    self.ft[key] = self.t
            for key in keys:
                self.heat[key] += 1.0
            if self.freq is not None:
                np.add.at(self.freq, np.asarray(keys), 1.0)
            seen = set()
            lc = collections.Counter()
            for key in keys:
                if self.vram[key]:
                    where = "vram"
                elif self.ram[key]:
                    where = "ram"
                    if self.ram_policy == "lru":
                        self.ram_clock.move_to_end(key)
                elif key in seen:
                    where = "dup"   # a second token of the window: computed with the first's read
                elif key in self.cache or key in self.a1:
                    where = "cache"
                    if key in self.cache:
                        self.cache.move_to_end(key)
                else:
                    where = "ssd"
                    self.admit(key)
                    if self.ram_policy == "lru":   # the record is in a pinned cache slot: it trades places with the LRU RAM one
                        old, _ = self.ram_clock.popitem(last=False)
                        self.ram[old] = False
                        self.ram[key] = True
                        self.ram_clock[key] = None
                        if count:
                            self.stat["ram_promotions"] += 1
                    elif self.ram_policy == "lfu":   # ... with the RAM one of the lowest long-run count, when hotter by a margin
                        h = np.where(self.ram, self.freq, np.inf)
                        old = int(h.argmin())
                        if self.freq[key] > h[old] + a.margin:
                            self.ram[old] = False
                            self.ram[key] = True
                            if count:
                                self.stat["ram_promotions"] += 1
                seen.add(key)
                lc[where] += 1
                if count:
                    self.stat[where] += 1
            if count and self.on_layer is not None:
                self.on_layer(il, T_, lc["vram"], lc["ram"], lc["cache"], lc["ssd"])
        if count:
            self.tokens += T
        self.adapt_tick(T)

    def admit(self, key):
        """A record just read from the SSD: the cache policy keeps it or drops it."""
        if isinstance(self.cache, PoolCache):
            self.cache.add(key)
        elif self.cache_policy == "lru":
            self.cache[key] = None
            if len(self.cache) > self.cache_n:
                self.cache.popitem(last=False)
        elif self.cache_policy == "lfu":   # evict the cached key of the lowest decayed count, if the new one beats it
            if len(self.cache) < self.cache_n:
                self.cache[key] = None
                self.incache[key] = True
                return
            dec = self.fc * self.a.cache_decay ** (self.t - self.ft)
            h = np.where(self.incache, dec, np.inf)
            old = int(h.argmin())
            if dec[key] >= h[old]:
                del self.cache[old]
                self.incache[old] = False
                self.cache[key] = None
                self.incache[key] = True
        elif self.cache_policy == "2q":   # Johnson & Shasha: probation FIFO (25%), ghosts (50%), the main LRU (75%)
            n_in = max(1, self.cache_n // 4)
            if key in self.a1out:
                del self.a1out[key]
                self.cache[key] = None
                if len(self.cache) > self.cache_n - n_in:
                    self.cache.popitem(last=False)
            else:
                self.a1[key] = None
                if len(self.a1) > n_in:
                    old, _ = self.a1.popitem(last=False)
                    self.a1out[old] = None
                    if len(self.a1out) > self.cache_n // 2:
                        self.a1out.popitem(last=False)

    def adapt_tick(self, T):
        a = self.a
        self._fed = getattr(self, "_fed", 0) + T
        if self._fed < a.adapt_every:
            return
        self._fed = 0
        if self.vram_policy == "adaptive":
            self.swap_round()
        self.heat *= a.decay
        if self.freq is not None:
            self.freq *= a.ram_decay ** a.adapt_every

    def swap_round(self):
        a = self.a
        plan = []
        for il in range(NL):
            if il in self.excl:
                continue
            ks = np.arange(il * NE, (il + 1) * NE)
            res = self.vram[ks]
            h = self.heat[ks]
            vic = ks[res][np.argsort(h[res], kind="stable")]
            cand_mask = (~res) & (h >= 2.0)
            cand = ks[cand_mask][np.argsort(-h[cand_mask], kind="stable")]
            for i in range(min(len(cand), len(vic))):
                if self.heat[cand[i]] < self.heat[vic[i]] + 1.5:
                    break
                plan.append((self.heat[cand[i]] - self.heat[vic[i]], int(cand[i]), int(vic[i])))
        plan.sort(key=lambda x: -x[0])
        # the engine's rule (src/x/tiers.cpp swap_round): of a round's swaps, at most `ssd_swaps` bring in an expert with no RAM
        # copy (a synchronous fill, or since 2026-10-18 an asynchronous one from the SSD cache, under the same cap)
        eng = getattr(a, "engine_swaps", False)
        ssd_cap = 8 if eng else getattr(a, "ssd_swaps", None)
        if eng:
            plan = plan[:a.swaps]
        taken, ssd_in = [], 0
        for g, kin, kout in plan:
            if len(taken) >= a.swaps:
                break
            if ssd_cap is not None and not self.ram[kin]:
                if ssd_in >= ssd_cap:
                    continue
                if eng and kin not in self.cache and kin not in self.a1:
                    continue
                ssd_in += 1
            taken.append((g, kin, kout))
        for _, kin, kout in taken:
            from_cache = eng and not self.ram[kin]
            self.vram[kout] = False
            self.vram[kin] = True
            if from_cache:   # (SWAP_CACHE: the cache stays exclusive of VRAM)
                self.cache.pop(kin, None)
                self.a1.pop(kin, None)
                self.incache[kin] = False
                if not self.ram[kout] and kout not in self.cache and kout not in self.a1:
                    self.admit(kout)
            elif self.ram_policy != "static-keep":
                # the victim's RAM copy: the exchange keeps the RAM copy's size (the incoming one's blob is returned)
                if self.ram[kin]:
                    self.ram[kin] = False
                    self.ram[kout] = True
                    if self.ram_policy == "lru":
                        self.ram_clock.pop(kin, None)
                        self.ram_clock[kout] = None
            self.stat["swaps"] += 1

    def report(self, name):
        s, t = self.stat, max(1, self.tokens)
        tot = s["vram"] + s["ram"] + s["cache"] + s["ssd"] + s["dup"]
        print(f"{name:34s} per token: VRAM {s['vram']/t:6.1f}  RAM {s['ram']/t:6.1f}  cache {s['cache']/t:5.1f}  "
              f"SSD reads {s['ssd']/t:6.1f}  (window dups {s['dup']/t:4.1f}; routings {tot/t:.0f}; swaps {s['swaps']/t:.1f}"
              + (f"; RAM promotions {s['ram_promotions']/t:.1f}" if s["ram_promotions"] else "") + ")")
        return s["ssd"] / t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", nargs="*", default=[])
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--vram0", type=int, default=979)   # the 217 GB pack, 2026-10-19 (DSV41_BACKLOG #57)
    ap.add_argument("--vram1", type=int, default=815)
    ap.add_argument("--split", type=int, default=20)
    ap.add_argument("--ram", type=int, default=1300)   # 17.29 GiB of 11.1 / 14.5 MiB records
    ap.add_argument("--cache", type=int, default=4558)   # --ssd-cache-gib 60: 1,367 + 3,191 slots
    ap.add_argument("--cache-sweep", default="", help="also: static RAM shrunk by C and an LRU SSD cache of C slots, for each C")
    ap.add_argument("--cache-policies", default="lru", help="the cache sweep's policies: lru, lfu, 2q")
    ap.add_argument("--cache-decay", type=float, default=0.9999, help="lfu: a count's decay per routing fed (240 a token)")
    ap.add_argument("--adapt-every", type=int, default=4)
    ap.add_argument("--decay", type=float, default=0.7)
    ap.add_argument("--swaps", type=int, default=14)   # src/x/tiers.cpp: clamp(256 / slot MiB, 8, 96) = 14 for 17.9 MiB
    ap.add_argument("--policies", default="all")
    ap.add_argument("--profile", default="", help="the profile from a strata-x profile file instead of the calibration traces")
    ap.add_argument("--ram-fair", action="store_true", help="the static RAM set by layer in turn (Maya's DS4_COPY_FAIR)")
    ap.add_argument("--engine-swaps", action="store_true", help="the swaps as the engine makes them (see the top)")
    ap.add_argument("--ram-decay", type=float, default=0.995, help="lfu: the long-run count's decay per token")
    ap.add_argument("--prior", type=float, default=0.25, help="lfu: the profile's counts' weight at the start")
    ap.add_argument("--margin", type=float, default=1.0, help="lfu: a promotion's lead over the coldest RAM expert")
    a = ap.parse_args()
    calib = [w for p in a.calib for w in read_trace(p)]
    ev = [w for p in a.eval for w in read_trace(p)]
    prof = read_profile(a.profile) if a.profile else (counts_of(calib) if calib else None)
    print(f"calibration {len(calib)} windows, evaluation {len(ev)} windows "
          f"({sum(max(t for _, t, _ in w) for w in ev)} tokens)", file=sys.stderr)
    pol = {
        "none/static RAM (today)": (None, "static", "adaptive"),
        "profile/static RAM": (prof, "static", "adaptive"),
        "profile/static RAM, fixed VRAM": (prof, "static", "fixed"),
        "profile/LRU RAM": (prof, "lru", "adaptive"),
        "none/LRU RAM": (None, "lru", "adaptive"),
        "profile/LFU RAM": (prof, "lfu", "adaptive"),
        "none/LFU RAM": (None, "lfu", "adaptive"),
        "oracle (eval's own counts)/static": (counts_of(ev), "static", "adaptive"),
    }
    for name, (p, rp, vp) in pol.items():
        if a.policies != "all" and name.split("/")[0] not in a.policies.split(","):
            continue
        if p is None and name.startswith("profile"):
            continue
        sim = Sim(a, p, rp, vp)
        for w in calib:   # the tiers run through the calibration text first (a server that has run)
            sim.window(w, False)
        for w in ev:
            sim.window(w, True)
        sim.report(name)
    for c in [int(x) for x in a.cache_sweep.split(",") if x]:
        for cp in a.cache_policies.split(","):
            for pname, p in (("profile", prof), ("none", None)):
                if p is None and pname == "profile":
                    continue
                sim = Sim(a, p, "static", "adaptive", ram=a.ram + a.cache - c, cache=c, cache_policy=cp)
                for w in calib:
                    sim.window(w, False)
                for w in ev:
                    sim.window(w, True)
                sim.report(f"{pname}/RAM {a.ram + a.cache - c} + {cp} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
