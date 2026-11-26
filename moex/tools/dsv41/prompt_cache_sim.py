"""moex/tools/dsv41/prompt_cache_sim.py - a conversation's routing replayed against the SSD cache's policies (DSV41_PLAN §41):
the drive's reads of each prompt and of each decoded token.

    python -I prompt_cache_sim.py TRACE PROFILE [POLICY,...]

(DSV41_PLAN §64) Each prompt's layers are also timed: a layer the larger of its uploads over its GPU's link and its drive reads.
POOL_W0=w weights GPU 0's layers in the pools' sizes; SHARE=1 lets the CPU pool take host-held experts while that shortens a
layer; SPLIT=n puts the time model's GPU 1 from layer n (the tiers keep 20).

TRACE: strata-x's STRATA_ROUTE_TRACE file - int32 records (layer, T, k, then T x k expert ids), one per MoE layer call; a
prompt chunk's calls are marked layer + 1000.  A request is a run of prompt calls and the decode windows after it (a window
ends where the layer number drops).  PROFILE: the --expert-profile the engine started from (its order places the tiers).

The tiers, as the engine has them at strata-dsv41.json's settings (the q3q2u pack, --ssd-cache-gib 60, 512K context):
  - VRAM: 1,102 slots for GPU 0's layers (0-19), 856 for GPU 1's, filled in the profile's order; adaptive as tier_sim.py
    models src/x/tiers.cpp (heat += 1 a decode routing, x 0.7 every 4 tokens; a round trades the i-th hottest outsider for the
    i-th coldest resident when hotter by 1.5, at most 20 a round; an incoming RAM copy trades places with the victim).
  - the RAM copy: the profile's hottest of the rest, 20.84 GiB.
  - the SSD cache: one pool of slots per record size (349 / 1,746 / 815 / 1,746 of 11.6 / 12.8 / 14.0 / 15.2 MB records).
  - a prompt lends 325 VRAM slots for its first request, 184 after (the coldest by heat, half a GPU): their records go into
    the cache (most recent) for the prompt and its refill.
  - a decode window's routed experts outside VRAM and the RAM copy: a hit, or a read (its record kept, most recent); a prompt
    layer's distinct routed experts outside them: a hit, or a read the policy keeps or not.

POLICIES (all of them by default):
  lru           the config before §41: a prompt's reads not kept (its share is 0), its hits not refreshed
  lru+hits      ... a prompt's hits refresh (STRATA_DSV41_PROMPT_LRU=1)
  lru+free      ... a prompt's read takes a slot that holds nothing (STRATA_DSV41_CACHE_FREE=1)
  lru+hits+free both
  hot<D>        both, and a read that 1 / D or more of the chunk's rows route to takes the least recent slot
                (STRATA_DSV41_PROMPT_HOT=D)
  share         every prompt read kept, least recent evicted (STRATA_DSV41_PROMPT_KEEP_SHARE=1)
  lfu           the victim the lowest frequency (the profile's counts as ~52K tokens', then every access) of the pool's
                least recent quarter; a prompt's read kept only when more frequent than that victim; its hits refresh
  gdsf          GreedyDual-Size-Frequency (H = L + frequency, the lowest H evicted, L the evicted H); its hits refresh
  oracle        Belady's: the record used again last evicted, a read used again later than every cached one not kept - on
                the lru policy's sequence of cache lookups (the tiers above do not depend on the cache)
"""
import collections
import heapq
import sys

import numpy as np

NE, NL = 384, 40
REC = [14008320] * 2 + [11612160] * 3 + [12812288] * 10 + [14008320] * 5 + [12812288] * 5 + [15208448] * 15
SLOTS = {11612160: 349, 12812288: 1746, 14008320: 815, 15208448: 1746}
# moex (DSV41_PLAN §64): POOL_W0=w sizes the pools with GPU 0's layers' record bytes weighted by w in each size's share of the
# 60 GiB (1: as above; the engine sizes them by bytes alone - the tilt was tried here and not taken); the prompt's time a layer is
# the larger of its uploads over its GPU's link (layers 0-19 at 6.26 GB/s, 20-39 at 12.16) and its drive reads at 2.6 GB/s
import os as _os
_W0 = float(_os.environ.get("POOL_W0", "1"))
if _W0 != 1.0:
    _share = collections.Counter()
    for _il, _st in enumerate(REC):
        _share[_st] += _st * (_W0 if _il < 20 else 1.0)
    _bytes = sum(n * s for s, n in SLOTS.items())
    _all = sum(_share.values())
    SLOTS = {s: max(16, int(_bytes * _share[s] / _all / s)) for s in SLOTS}
LINK = (6.26e9, 12.16e9)
DRIVE = 2.6e9
# (§64) SHARE=1: a CPU share in the prompt's layers - experts held in host memory (the RAM copy, the cache, a record just read)
# computed by the CPU pool instead of uploaded, the fewest-routed first, while that shortens the layer: the pool's time an expert
# its record's bytes at POOL_BW (60 GB/s measured in decode) or its rows' arithmetic at POOL_OPS (int8 dot products, 1.5e12 a
# second assumed), the larger; the pool's rows uploaded (5,120 floats a routed pair)
SHARE = _os.environ.get("SHARE") == "1"
POOL_BW, POOL_OPS = 60e9, 1.5e12
SPLIT = int(_os.environ.get("SPLIT", "20"))   # (§64) the time model's first layer on GPU 1 (the tiers keep the engine's 20)
VR = (1102, 856)
RAM_BYTES = 20.84 * 2**30
SWAPS = 20
LENDS = [325] + [184] * 4096


def read_trace(path):
    a = np.fromfile(path, dtype=np.int32)
    reqs, cur, last, i = [], None, None, 0
    while i + 3 <= a.size:
        il, T, k = int(a[i]), int(a[i + 1]), int(a[i + 2])
        i += 3
        if i + T * k > a.size:   # a trace still being written
            break
        ids = a[i:i + T * k].reshape(T, k)
        i += T * k
        if il >= 1000:
            if cur is None or cur["dec"]:
                cur = {"pr": [], "dec": []}
                reqs.append(cur)
            cur["pr"].append((il - 1000, ids))
        else:
            if cur is None:
                cur = {"pr": [], "dec": []}
                reqs.append(cur)
            if not cur["dec"] or il <= last:
                cur["dec"].append([])
            cur["dec"][-1].append((il, ids))
        last = il % 1000
    return reqs


def load_profile(path):
    order, cnt = [], np.zeros(NL * NE)
    for line in open(path):
        if line.startswith("#"):
            continue
        l, e, c = line.split()
        key = int(l) * NE + int(e)
        order.append(key)
        cnt[key] = float(c)
    return order, cnt


class Sim:
    def __init__(self, order, prof, policy, nexts=None):
        self.policy = policy
        self.hot = int(policy[3:]) if policy.startswith("hot") else 0
        self.hits_refresh = policy in ("lru+hits", "lru+hits+free", "lfu", "gdsf", "oracle") or self.hot > 0
        self.free = policy in ("lru+free", "lru+hits+free") or self.hot > 0
        self.freq = prof / prof.sum() * 52000.0
        self.vram = np.zeros(NL * NE, bool)
        used = [0, 0]
        for key in order:
            g = 0 if key // NE < 20 else 1
            if used[g] < VR[g]:
                self.vram[key] = True
                used[g] += 1
        self.ram = np.zeros(NL * NE, bool)
        b = 0.0
        for key in order:
            if not self.vram[key] and b + REC[key // NE] <= RAM_BYTES:
                self.ram[key] = True
                b += REC[key // NE]
        self.heat = np.zeros(NL * NE)
        self.pool = {s: collections.OrderedDict() for s in SLOTS}   # key -> None, least recent first
        self.st = collections.Counter()
        self.fed = self.t = 0
        self.nexts, self.cur_next = nexts, {}
        self.heaps = {s: [] for s in SLOTS}
        self.H, self.L = {}, {s: 0.0 for s in SLOTS}
        self.seq = []   # the cache lookups, in order (the oracle's input)
        self.rows = 0

    def p(self, key):
        return self.pool[REC[key // NE]]

    def full(self, key):
        return len(self.p(key)) >= SLOTS[REC[key // NE]]

    def next_use(self, key):
        lst = self.nexts.get(key)
        while lst and lst[0] <= self.t:
            lst.popleft()
        return lst[0] if lst else 1 << 60

    def victim(self, key):
        p, sz = self.p(key), REC[key // NE]
        if self.policy == "lfu":
            n, best, bf = max(1, len(p) // 4), None, None
            for i, k2 in enumerate(p):
                if i >= n:
                    break
                if bf is None or self.freq[k2] < bf:
                    best, bf = k2, self.freq[k2]
            return best
        if self.policy in ("gdsf", "oracle"):
            h = self.heaps[sz]
            while h:
                v, k2 = h[0]
                if k2 in p and (self.H.get(k2) == v if self.policy == "gdsf" else self.cur_next.get(k2) == -v):
                    return k2
                heapq.heappop(h)
        return next(iter(p))

    def gd_set(self, key):
        sz = REC[key // NE]
        self.H[key] = self.L[sz] + self.freq[key]
        heapq.heappush(self.heaps[sz], (self.H[key], key))

    def insert(self, key, prompt, cnt=0):
        """A record just read: kept (a victim evicted) or not, by the policy."""
        p = self.p(key)
        if prompt and self.policy != "share" and not self.policy in ("lfu", "gdsf", "oracle"):
            if not self.free:
                return
            if self.full(key) and not (self.hot > 0 and cnt * self.hot >= self.rows):
                return
        if self.full(key):
            v = self.victim(key)
            if self.policy == "lfu" and prompt and self.freq[key] <= self.freq[v]:
                return
            if self.policy == "oracle" and self.next_use(key) >= self.next_use(v):
                return
            if self.policy == "gdsf":
                self.L[REC[key // NE]] = self.H.pop(v, 0.0)
            del p[v]
        p[key] = None
        if self.policy == "gdsf":
            self.gd_set(key)

    def lookup(self, key, prompt, cnt=0, count=True):
        self.t += 1
        self.seq.append(key)
        if self.policy in ("lfu", "gdsf"):
            self.freq[key] += 1.0
        if self.policy == "oracle":
            nt = self.next_use(key)
            self.cur_next[key] = nt
            heapq.heappush(self.heaps[REC[key // NE]], (-nt, key))
        if key in self.p(key):
            if not prompt or self.hits_refresh:
                self.p(key).move_to_end(key)
                if self.policy == "gdsf":
                    self.gd_set(key)
            self.st["p_hit" if prompt else "d_hit"] += count
            return
        self.st["p_read" if prompt else "d_read"] += count
        self.insert(key, prompt, cnt)

    def prompt(self, calls, lent_n):
        lent = []
        for g, n in ((0, lent_n // 2), (1, lent_n - lent_n // 2)):
            ks = np.nonzero(self.vram & ((np.arange(NL * NE) // NE < 20) == (g == 0)))[0]
            lent += ks[np.argsort(self.heat[ks], kind="stable")][:n].tolist()
        for key in lent:   # the lend: each lent expert's record into the cache (a prompt_keep_take)
            self.vram[key] = False
            if not self.ram[key]:
                if key in self.p(key):
                    self.p(key).move_to_end(key)
                else:
                    if self.full(key):
                        v = self.victim(key)
                        if self.policy == "gdsf":
                            self.L[REC[key // NE]] = self.H.pop(v, 0.0)
                        del self.p(key)[v]
                    self.p(key)[key] = None
                    if self.policy == "gdsf":
                        self.gd_set(key)
        for il, ids in calls:
            self.rows = ids.shape[0]
            cnt = collections.Counter((il * NE + ids.reshape(-1)).tolist())
            up = rd = 0
            r0 = self.st["p_read"]
            host = []   # (rows routed, bytes) of each expert outside VRAM: it is in host memory before it is uploaded
            for key in sorted(cnt):
                if not self.vram[key]:
                    up += REC[key // NE]
                    host.append((cnt[key], REC[key // NE]))
                    if not self.ram[key]:
                        self.lookup(key, True, cnt[key])
                        if self.st["p_read"] > r0:
                            rd += REC[key // NE]
                            r0 = self.st["p_read"]
            link = LINK[0 if il < SPLIT else 1]
            t = max(up / link, rd / DRIVE)
            if SHARE:   # the fewest-routed experts to the pool while the layer gets shorter
                host.sort()
                pool_b = pool_ops = 0.0
                up_s = up
                for rows, b in host:
                    nb, nops = pool_b + b, pool_ops + rows * 3 * 5120 * 2304 * 2
                    nup = up_s - b + rows * 5120 * 4
                    nt = max(nup / link, rd / DRIVE, nb / POOL_BW, nops / POOL_OPS)
                    if nt >= t:
                        break
                    pool_b, pool_ops, up_s, t = nb, nops, nup, nt
                    self.st["p_cpu"] += 1
            self.st["p_us"] += int(1e6 * t)
        for key in lent:
            self.vram[key] = True

    def window(self, w):
        T = max(ids.shape[0] for _, ids in w)
        for il, ids in w:
            seen = set()
            for key in (il * NE + ids.reshape(-1)).tolist():
                self.heat[key] += 1.0
                if not self.vram[key] and not self.ram[key] and key not in seen:
                    self.lookup(key, False)
                seen.add(key)
        self.st["tokens"] += T
        self.fed += T
        if self.fed >= 4:
            self.fed = 0
            self.swap_round()
            self.heat *= 0.7

    def swap_round(self):
        plan = []
        for il in range(NL):
            ks = np.arange(il * NE, (il + 1) * NE)
            res, h = self.vram[ks], self.heat[ks]
            vic = ks[res][np.argsort(h[res], kind="stable")]
            cm = (~res) & (h >= 2.0)
            cand = ks[cm][np.argsort(-h[cm], kind="stable")]
            for i in range(min(len(cand), len(vic))):
                if self.heat[cand[i]] < self.heat[vic[i]] + 1.5:
                    break
                plan.append((self.heat[cand[i]] - self.heat[vic[i]], int(cand[i]), int(vic[i])))
        plan.sort(key=lambda x: -x[0])
        for _, kin, kout in plan[:SWAPS]:
            self.vram[kout], self.vram[kin] = False, True
            if self.ram[kin]:
                self.ram[kin], self.ram[kout] = False, True


run_cpu = [0]


def run(reqs, order, prof, policy, nexts=None):
    s = Sim(order, prof, policy, nexts)
    per = []
    for ri, r in enumerate(reqs):
        a = dict(s.st)
        if r["pr"]:
            s.prompt(r["pr"], LENDS[ri])
        b = dict(s.st)
        for w in r["dec"]:
            s.window(w)
        c = dict(s.st)
        g = lambda d, k: d.get(k, 0)
        per.append((g(b, "p_read") - g(a, "p_read"), g(b, "p_hit") - g(a, "p_hit"), g(c, "d_read") - g(b, "d_read"),
                    g(c, "tokens") - g(b, "tokens"), (g(b, "p_us") - g(a, "p_us")) / 1e6))
    run_cpu[0] = s.st.get("p_cpu", 0)
    return per, s.seq


def main():
    reqs = read_trace(sys.argv[1])
    order, prof = load_profile(sys.argv[2])
    pols = sys.argv[3].split(",") if len(sys.argv) > 3 else ["lru", "lru+hits", "lru+free", "lru+hits+free", "hot26", "share",
                                                              "lfu", "gdsf", "oracle"]
    print(f"{len(reqs)} requests (prompt rows + decode windows): "
          + ", ".join(f"{max((ids.shape[0] for _, ids in r['pr']), default=0)}+{len(r['dec'])}" for r in reqs))
    nexts = None
    if "oracle" in pols:
        _, seq = run(reqs, order, prof, "lru")
        nexts = collections.defaultdict(collections.deque)
        for t, key in enumerate(seq, start=1):
            nexts[key].append(t)
    for pol in pols:
        per, _ = run(reqs, order, prof, pol, nexts if pol == "oracle" else None)
        pr = [x for x in per if x[0] + x[1] > 0]
        dr, dt = sum(x[2] for x in per), sum(x[3] for x in per)
        if SHARE:
            print(f"  (SHARE: {run_cpu[0]} experts to the pool in all)")
        print(f"  {pol:14s} each prompt's drive reads {' '.join(str(x[0]) for x in pr)} (cache hits {' '.join(str(x[1]) for x in pr)}); "
              f"its layers' time {' '.join(f'{x[4]:.1f}' for x in pr)} s; "
              f"decode {dr / max(1, dt):.2f} reads a token ({' '.join(f'{x[2] / max(1, x[3]):.2f}' for x in per if x[3])})",
              flush=True)


if __name__ == "__main__":
    main()
