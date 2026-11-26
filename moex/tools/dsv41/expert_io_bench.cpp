// moex/tools/dsv41/expert_io_bench.cpp - expert_io.hpp on this NVMe: demand misses alone and under read-ahead load.
//
//   expert_io_bench FILE [threads] [seconds] [piece_mib] [record_bytes] [demand_only]
// Reads V4.1-sized experts (18,800,640 B by default, whole pages) at random page-aligned offsets of FILE, read-only
// O_DIRECT, in pieces of piece_mib (4 by default: ExpertIo's).  demand_only = 1: part A only (a sweep of piece size x
// readers, DSV41_BACKLOG #7).
//  A. demand only: batches of k misses (one layer's worth), the batch time and the drive's GB/s;
//  B. the same demand batches while a background thread keeps the speculative classes full, for each cap on the
//     reader threads the speculative classes may hold: the demand batch time and the total GB/s.
#include "expert_io.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace dsv41;
using Clock = std::chrono::steady_clock;
static uint64_t kExpert = 18'800'640;

static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s FILE [threads] [seconds]\n", argv[0]); return 2; }
    const int threads = argc > 2 ? std::atoi(argv[2]) : 8;
    const double run_s = argc > 3 ? std::atof(argv[3]) : 6.0;
    const uint64_t piece = (argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 4) << 20;
    if (argc > 5) kExpert = (std::strtoull(argv[5], nullptr, 10) + 4095) & ~4095ull;
    const bool demand_only = argc > 6 && std::atoi(argv[6]) == 1;
    const int fd = ExpertIo::open_direct(argv[1]);
    struct stat st;
    fstat(fd, &st);
    const uint64_t pages = (uint64_t(st.st_size) - kExpert) / 4096;
    const int slots = 64;
    uint8_t* mem = static_cast<uint8_t*>(std::aligned_alloc(4096, slots * kExpert));
    std::mt19937_64 rng(42);
    auto off = [&] { return (rng() % pages) * 4096; };

    auto demand_batches = [&](ExpertIo& io, int k, double dur, double* gbs) {
        std::vector<double> t;
        const auto t0 = Clock::now();
        uint64_t bytes = 0;
        while (secs(t0, Clock::now()) < dur) {
            const auto b0 = Clock::now();
            std::vector<ExpertIo::Handle> hs;
            for (int i = 0; i < k; ++i) hs.push_back(io.submit(fd, off(), kExpert, mem + uint64_t(i) * kExpert, IoClass::Demand));
            for (auto& h : hs) { io.wait(h); if (h->error) { std::fprintf(stderr, "read error %d\n", int(h->error)); std::exit(1); } }
            t.push_back(secs(b0, Clock::now()));
            bytes += uint64_t(k) * kExpert;
        }
        if (gbs) *gbs = bytes / secs(t0, Clock::now()) / 1e9;
        std::sort(t.begin(), t.end());
        return std::make_pair(t[t.size() / 2], t[t.size() * 9 / 10]);
    };

    std::printf("A. demand only, %d reader threads, %llu MiB pieces, %llu-byte records\n", threads,
                (unsigned long long) (piece >> 20), (unsigned long long) kExpert);
    for (int k : {1, 2, 4, 8}) {
        ExpertIo io(threads, 0, piece);
        double gbs = 0;
        auto [med, p90] = demand_batches(io, k, run_s / 2, &gbs);
        std::printf("   %d misses per batch: median %6.1f ms, p90 %6.1f ms, %.2f GB/s (%.1f ms per expert)\n", k, med * 1e3,
                    p90 * 1e3, gbs, med * 1e3 / k);
    }

    if (demand_only) { std::free(mem); close(fd); return 0; }
    std::printf("B. 2 demand misses per batch while read-ahead keeps the speculative classes full\n");
    for (int cap : {1, 2, 4, threads}) {  // cap 0 is part A
        ExpertIo io(threads, cap);
        std::atomic<bool> stop{false};
        std::atomic<uint64_t> spec_bytes{0};
        std::thread bg([&] {  // keeps ~16 speculative experts queued, as a read-ahead flood would
            std::mt19937_64 r2(7);
            std::deque<ExpertIo::Handle> inflight;
            int slot = 0;
            while (!stop) {
                while (inflight.size() < 16) {
                    const uint64_t o = (r2() % pages) * 4096;
                    inflight.push_back(io.submit(fd, o, kExpert, mem + uint64_t(8 + slot) * kExpert, IoClass::Ahead));
                    slot = (slot + 1) % (slots - 8);
                }
                io.wait(inflight.front());
                if (!inflight.front()->cancelled) spec_bytes += kExpert;
                inflight.pop_front();
            }
            for (auto& h : inflight) { io.cancel(h); io.wait(h); }
        });
        const auto t0 = Clock::now();
        double dgbs = 0;
        auto [med, p90] = demand_batches(io, 2, run_s / 2, &dgbs);
        const double el = secs(t0, Clock::now());
        stop = true;
        bg.join();
        std::printf("   speculative cap %2d of %d threads: demand batch median %6.1f ms, p90 %6.1f ms; demand %.2f + "
                    "speculative %.2f GB/s\n", cap, threads, med * 1e3, p90 * 1e3, dgbs, spec_bytes / el / 1e9);
    }
    std::free(mem);
    close(fd);
    return 0;
}
