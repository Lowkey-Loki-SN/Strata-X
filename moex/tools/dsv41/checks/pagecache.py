"""pagecache.py [--evict] FILE... - how much of each file sits in the page cache (mincore over a read-only mapping, 1 GiB at a
time: nothing is read in), and with --evict, drop its clean cached pages first (posix_fadvise DONTNEED: no root, no data
touched; pages another process has mapped stay).  For a cold A/B arm that reads through the mappings."""
import ctypes
import mmap
import os
import sys

import numpy as np

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
PAGE = mmap.PAGESIZE
CH = 1 << 30


def resident(fd, size):
    n = 0
    for off in range(0, size, CH):
        ln = min(CH, size - off)
        addr = libc.mmap(None, ln, mmap.PROT_READ, mmap.MAP_SHARED, fd, off)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap")
        vec = (ctypes.c_ubyte * ((ln + PAGE - 1) // PAGE))()
        if libc.mincore(addr, ln, vec) != 0:
            raise OSError(ctypes.get_errno(), "mincore")
        n += int((np.frombuffer(vec, dtype=np.uint8) & 1).sum())
        libc.munmap(addr, ln)
    return n


evict = "--evict" in sys.argv[1:]
total = 0
for path in [a for a in sys.argv[1:] if a != "--evict"]:
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    before = resident(fd, size)
    if evict:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    after = resident(fd, size)
    os.close(fd)
    total += after
    print(f"{path}: {size / 2**30:.1f} GiB, cached {before * PAGE / 2**20:.0f} MiB"
          + (f" -> {after * PAGE / 2**20:.0f} MiB after the eviction" if evict else ""))
print(f"cached now: {total * PAGE / 2**20:.0f} MiB")
