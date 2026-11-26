"""DSV41_BACKLOG #121's traffic model: a cold read of N tokens, today (layer-major windows of at most 8 chunks: layers 0-20's
non-resident records stream once a WINDOW) against one window for the whole call (the chunks' rows parked in host memory between
layers: layers 0-20's records stream once a CALL, each (layer, chunk) moves its rows up and back).  Every input is a measured
number of this box (DSV41_PLAN §62's sources), the GPU's compute a range."""
import math
C = 7680                    # tokens a chunk at long context (7,424-8,192 measured)
WIN = 8                     # chunks a window today (LM_CHUNKS; GPU 0's loan)
ROW_B = 4 * 5120 * 4        # bytes a token's residual rows (4 mHC streams of 5,120 F32)
WIN_GB = 77.0               # drive GB a window for layers 0-20 (552K: ~805 GB over 9 windows, the rest the tails' run)
TAIL_GB = 110.0             # the tails' run and the call's fixed reads (the cold 44K read: 217.9 GB, layers 0-20 105.2 of it)
ONCE_GB = 105.0             # layers 0-20's records read once (the 44K read's)
DRIVE = 2.0                 # GB/s the drive reads when healthy (1.9-2.5 measured)
EPI_S, EPI_GB = 72.0, 170.0 # a slow-mode episode, and one per this many GB read (70-76 s; one per 160-190 GB)
H2D0 = 6.26                 # GB/s into GPU 0 (the profile's prompt read)
D2H0 = 6.0                  # GB/s back (assumed ~H2D)
LAYERS = 20                 # GPU 0's layers in every chunk of a long read
# GPU 0's compute in a cold read, measured (the 70,011-token read under nsys, 2026-10-30): 51.9 s of kernels for 20 layers -
# 35.9 us a token and layer outside the indexer, and the indexer's scores (lid_scores_mma, 1.58 s) growing with the position
TOK_LAYER_US = 35.9
IDX_US = 1.58e6 / (70011 * 70011 / 2)        # us a token per position: the scores over every earlier compressed position
def compute_s(N):
    return (N * LAYERS * TOK_LAYER_US + IDX_US * N * N / 2) / 1e6
print("GPU 0's compute measured: 35.9 us a token and layer, plus the indexer by position")
print(f"  {'N':>8} | today: windows, drive GB, drive + episodes, compute -> time | one window: drive GB, rows' copies (both ways"
      f" / overlapped), compute -> time | saved")
for N in (100_000, 200_000, 300_000, 552_000, 750_000):
    chunks = math.ceil(N / C)
    W = math.ceil(chunks / WIN)
    comp = compute_s(N)
    gb_now = W * WIN_GB + TAIL_GB
    drive_now = gb_now / DRIVE + gb_now / EPI_GB * EPI_S
    t_now = max(drive_now, comp)
    gb_new = ONCE_GB + TAIL_GB
    drive_new = gb_new / DRIVE + gb_new / EPI_GB * EPI_S
    rows = N * ROW_B / 1e9
    seq = LAYERS * rows * (1 / H2D0 + 1 / D2H0)
    ovl = LAYERS * rows * max(1 / H2D0, 1 / D2H0)
    # (DSV41_PLAN §63: GPU 1 runs layer 20 a chunk and layers 21-39 once for the tails at the call's end - in both - so one
    # window loses only layer 20's overlap; it does need every chunk's engram rows at layer 1, which this leaves out)
    t_new = max(drive_new, comp, seq)
    t_new_o = max(drive_new, comp, ovl)
    ram = "fits" if rows <= 60 * 1.074 else "over the cache's 60 GiB"
    print(f"  {N:>8} | {W:2d}, {gb_now:5.0f}, {drive_now / 60:5.1f} min, {comp / 60:4.1f} -> {t_now / 60:5.1f} min | "
          f"{gb_new:4.0f}, {seq / 60:4.1f} / {ovl / 60:4.1f} min ({rows:.0f} GB, {ram}), {comp / 60:4.1f} -> "
          f"{t_new_o / 60:4.1f}-{t_new / 60:4.1f} min | {(t_now - t_new) / 60:4.1f}-{(t_now - t_new_o) / 60:4.1f} min "
          f"({1 - t_new / t_now:.0%}-{1 - t_new_o / t_now:.0%})")
# (DSV41_PLAN §63) host windows with two sets of lent blocks (STRATA_DSV41_LM_HOST=2): GPU 1 runs window w's layer 20 while GPU 0
# runs w + 1 (its layers 21-39 run once for the tails at the end, as today) - so the compute is GPU 0's pass, as one window's;
# layers 0-20's records stream once a window, and only the first window's engram rows wait at its layer 1 - the best W a size
print("two sets of host windows (LM_HOST=2): the best number of windows W")
for N in (100_000, 200_000, 300_000, 552_000, 750_000):
    comp = compute_s(N)
    best = None
    W_now = math.ceil(math.ceil(N / C) / WIN)
    for W in range(2, max(3, W_now)):   # (fewer windows than today's; a window's bytes as today's, the long read's keep in both)
        gb = W * WIN_GB + TAIL_GB
        t = max(gb / DRIVE + gb / EPI_GB * EPI_S, comp)
        if best is None or t < best[0]:
            best = (t, W, gb)
    chunks = math.ceil(N / C)
    gb_now = math.ceil(chunks / WIN) * WIN_GB + TAIL_GB
    t_now = max(gb_now / DRIVE + gb_now / EPI_GB * EPI_S, comp)
    if W_now <= 2:
        print(f"  {N:>8} | today's {W_now} windows already: the switch leaves the read as it is")
        continue
    print(f"  {N:>8} | W {best[1]:2d} ({math.ceil(chunks / best[1])} chunks a window), {best[2]:5.0f} GB -> {best[0] / 60:4.1f} min "
          f"against today's {t_now / 60:4.1f} ({1 - best[0] / t_now:.0%})")
