// moex/tools/dsv41/drive_guard.hpp - the drive guard (DSV41_PLAN §18, opt-in STRATA_DSV41_DRIVE_GUARD=1; header-only).
//
// In its slow mode (DSV41_PLAN §14: the drive rewriting worn blocks, ~60-70 s at a time) this box's NVMe completed at most
// ~1,000 reads a second, and every request waited 1,020-2,084 ms behind the ~1,000 the engine had queued - the desktop's
// too (2026-10-21: / is on the same drive).  The guard keeps the engine's share of that queue at one read of each kind
// while the slow mode lasts, so another program's read waits behind one of the engine's.
//
// A thread reads the disk's counters (/sys/block/<disk>/stat: reads completed, the milliseconds they took) every 250 ms.
//  * The slow mode's sample: >= 80 ms a read and under 2,000 reads a second.  In its normal state the drive completed
//    >= 4,158 reads a second at every depth measured (1, 2, 8, 64, 256, 384 and ~1,400 in flight, floor_probe.py and the
//    engine's logs) and <= 39.4 ms a read; in the slow mode <= ~1,000 a second at >= 1,020 ms.  The guard drops to its
//    floor at once: one record reader reading 128 KB a call (one NVMe request), the engram rows' pages one at a time, the
//    start-up load on one reader (each one request in flight on the drive, floor_probe.py).
//  * Back up: two samples in a row at < 40 ms a read and >= 2,000 reads a second (the floor's own reads on a healthy drive
//    complete 4,158-16,588 a second), then a level every 250 ms while the samples stay that way, to no limit after kTop.  The
//    slow mode's sample again on the way, and it is back at the floor.  moex (DSV41_BACKLOG #99): a sample of 16 reads or more
//    whose reads the drive completed at >= 2,000 a second of its busy time (the counters' io_ticks) counts as healthy too - a
//    decode's reads at the floor come in bursts (a record's 104 requests one at a time, then the GPUs' part of a layer), so
//    the engine's own rate stayed under 2,000 a second on a healthy drive and the floor outlasted the episode (a 467-token
//    reply at 5.0 tok/s, 2026-10-22); the slow mode completes <= ~1,000 a busy second at every depth.
// STRATA_DSV41_DRIVE_GUARD=floor holds it at the floor for good (a check: the bytes read are the same at every level);
// STRATA_DSV41_DRIVE_GUARD_STAT=FILE reads the counters from FILE instead (a check feeds it a slow mode's counters).
#pragma once

#include <sys/stat.h>
#include <sys/sysmacros.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>
#include <thread>

namespace dsv41 {

class DriveGuard {
public:
    /// One level's limits: record readers at once, bytes a read call, engram rows asked for ahead of the copies.
    struct Depth {
        int readers;
        uint64_t call;
        int eg_rows;
    };
    static constexpr int kTop = 5;   ///< the level with no limit (the readers' own: 12 readers, 4 MiB calls, the engram's window)
    static Depth depth(int lv) {
        static constexpr Depth d[kTop] = {{1, 128u << 10, 1}, {1, 512u << 10, 4}, {2, 1u << 20, 16}, {2, 4u << 20, 64}, {8, 4u << 20, 256}};
        return lv >= kTop || lv < 0 ? Depth{INT_MAX, UINT64_MAX, INT_MAX} : d[lv];
    }

    /// moex (DSV41_PLAN §71): the level the slow mode drops to - STRATA_DSV41_GUARD_FLOOR=L (0, the floor, by default).  At the
    /// floor this drive took 3.01 ms a read in a slow mode, 340 reads a second (2026-11-09: 43 MB/s); the slow mode's own limit
    /// is ~1,000-2,000 reads a second at depth, so a level of 4-16 requests in flight (1: one 512 KB call, 2: two 1 MB calls)
    /// reads more of it while another program's read still waits behind no more than those few, not the ~1,000 of no guard
    static int slow_level() {
        static const int l = [] {
            const char* e = std::getenv("STRATA_DSV41_GUARD_FLOOR");
            return e ? std::clamp(std::atoi(e), 0, kTop - 1) : 0;
        }();
        return l;
    }

    /// The process's guard, or null (none started: every reader runs as without it).
    static DriveGuard* get() { return inst().load(std::memory_order_acquire); }

    /// Starts the guard once (a second call returns the first one): `stat` the counters' file, `floor` held there for good,
    /// `verbose` its lines on stderr.
    static DriveGuard* start(const std::string& stat, bool floor, bool verbose) {
        static std::atomic<bool> once{false};
        if (once.exchange(true)) return get();
        auto* g = new DriveGuard(stat, floor, verbose);   // the process's lifetime: never freed, its thread detached
        inst().store(g, std::memory_order_release);
        std::thread([g] { g->run(); }).detach();
        if (verbose)
            std::fprintf(stderr, "strata-x: the drive guard on (%s)%s: in the drive's slow mode (a 250 ms sample at >= 80 ms a "
                                 "read, under 2,000 reads a second) the engine keeps one read of each kind on it\n",
                         stat.c_str(), floor ? ", held at its floor (STRATA_DSV41_DRIVE_GUARD=floor)" : "");
        return g;
    }

    /// The stat file of the disk holding `dir` (/sys/dev/block/M:m/stat; the whole disk's for a partition), "" if none.
    static std::string disk_stat(const std::string& dir) {
        struct stat sb {};
        if (dir.empty() || ::stat(dir.c_str(), &sb) != 0) return "";
        const std::string base = "/sys/dev/block/" + std::to_string(major(sb.st_dev)) + ":" + std::to_string(minor(sb.st_dev));
        struct stat pb {};
        if (::stat((base + "/partition").c_str(), &pb) == 0) return base + "/../stat";
        if (::stat((base + "/stat").c_str(), &pb) == 0) return base + "/stat";
        return "";
    }

    int level() const { return lv_.load(std::memory_order_relaxed); }
    bool engaged() const { return level() < kTop; }
    int readers() const { return depth(level()).readers; }
    uint64_t call_bytes() const { return depth(level()).call; }
    int eg_rows() const { return depth(level()).eg_rows; }
    /// Reader k (0, 1, ...) of a pool waits while the level allows fewer readers (50 ms naps).
    void wait_reader(int k) const {
        while (k >= readers()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    int64_t episodes() const { return episodes_.load(std::memory_order_relaxed); }
    double floor_s() const { return (double) floor_ms_.load(std::memory_order_relaxed) / 1000.0; }   ///< time below kTop

private:
    DriveGuard(std::string stat, bool floor, bool verbose) : stat_(std::move(stat)), hold_(floor), verbose_(verbose) {
        lv_.store(floor ? 0 : kTop);
    }
    static std::atomic<DriveGuard*>& inst() {
        static std::atomic<DriveGuard*> p{nullptr};
        return p;
    }
    bool read(uint64_t& reads, uint64_t& ms, uint64_t& busy, uint64_t* wsect = nullptr) const {
        FILE* f = std::fopen(stat_.c_str(), "r");
        if (!f) return false;
        unsigned long long v[10] = {};
        const int got = std::fscanf(f, "%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5],
                                    &v[6], &v[7], &v[8], &v[9]);
        std::fclose(f);
        reads = v[0];
        ms = v[3];
        busy = got >= 10 ? v[9] : 0;   // io_ticks: the milliseconds with a request in flight (0: the busy rate unknown)
        if (wsect) *wsect = got >= 7 ? v[6] : 0;   // the sectors written
        return got >= 4;
    }
    /// moex (DSV41_PLAN §59, the audit's C): what an episode's line also gives - the drive's temperature (its hwmon "nvme"
    /// sensor, millidegrees), the MB written and the pages swapped out over the 10 s before it - so the episodes can be told
    /// apart by heat, by writes, by swap, or by none of them (the reads alone)
    static double nvme_temp_c() {
        for (int i = 0; i < 16; ++i) {
            const std::string d = "/sys/class/hwmon/hwmon" + std::to_string(i);
            char name[32] = {};
            if (FILE* f = std::fopen((d + "/name").c_str(), "r")) {
                const int ok = std::fscanf(f, "%31s", name);
                std::fclose(f);
                if (ok != 1 || std::string(name) != "nvme") continue;
                long mc = 0;
                if (FILE* t = std::fopen((d + "/temp1_input").c_str(), "r")) {
                    const int g = std::fscanf(t, "%ld", &mc);
                    std::fclose(t);
                    if (g == 1) return (double) mc / 1000.0;
                }
            }
        }
        return -1.0;
    }
    static uint64_t pswpout() {
        uint64_t v = 0;
        if (FILE* f = std::fopen("/proc/vmstat", "r")) {
            char key[64];
            unsigned long long x = 0;
            while (std::fscanf(f, "%63s %llu", key, &x) == 2)
                if (std::string(key) == "pswpout") {
                    v = x;
                    break;
                }
            std::fclose(f);
        }
        return v;
    }
    void run() {
        using clk = std::chrono::steady_clock;
        uint64_t r0 = 0, m0 = 0, b0 = 0;
        bool have = read(r0, m0, b0);
        int good = 0;                       // healthy samples in a row
        clk::time_point t_low{};            // when the level left kTop
        double low_ms = 0, low_n = 0;       // the drive's milliseconds and reads in the floor's samples, all but the healthy ones,
        int low_k = 0;                      // ... and those samples, from 2 s on (its lone reads' time)
        double drain_ms = 0, drain_n = 0;   // ... and in its first 2 s: the requests already queued draining
        int at_floor = 0;                   // samples at the floor so far
        std::deque<std::pair<uint64_t, uint64_t>> hist;   // (verbose) the sectors written, pswpout: the last 10 s
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            uint64_t r1 = 0, m1 = 0, b1 = 0, w1 = 0;
            if (!read(r1, m1, b1, verbose_ ? &w1 : nullptr)) {
                have = false;
                continue;
            }
            if (!have) {
                r0 = r1, m0 = m1, b0 = b1, have = true;
                continue;
            }
            if (verbose_) {   // (the 10 s before: 40 samples of the sectors written and the pages swapped out)
                hist.push_back({w1, pswpout()});
                if (hist.size() > 41) hist.pop_front();
            }
            const uint64_t n = r1 - r0, ms = (m1 - m0) & 0xffffffffull;   // the kernel prints the ticks as 32-bit
            const uint64_t bms = (b1 - b0) & 0xffffffffull;
            r0 = r1, m0 = m1, b0 = b1;
            if (hold_) continue;
            const double per = n ? (double) ms / (double) n : 0.0, rate = (double) n / 0.25;
            const bool slow = n > 0 && per >= 80.0 && rate < 2000.0;
            // (moex, #99: the drive's own rate while busy - a burst's reads, however few the sample holds)
            const bool busy_fast = n >= 16 && b1 > 0 && (bms == 0 || (double) n * 1000.0 / (double) bms >= 2000.0);
            const bool healthy = n > 0 && per < 40.0 && (rate >= 2000.0 || busy_fast);
            const int lv = lv_.load(std::memory_order_relaxed);
            if (slow) {
                if (lv == kTop) {
                    t_low = clk::now();
                    low_ms = low_n = drain_ms = drain_n = 0;
                    low_k = at_floor = 0;
                    episodes_.fetch_add(1, std::memory_order_relaxed);
                    if (verbose_) {
                        const double wmb = hist.size() >= 2 ? (double) (hist.back().first - hist.front().first) * 512.0 / 1e6 : -1.0;
                        const long long sw = hist.size() >= 2 ? (long long) (hist.back().second - hist.front().second) : -1;
                        std::fprintf(stderr, "strata-x: the drive guard: the drive's slow mode (%.0f ms a read, %.0f reads a second): "
                                             "the engine's reads at level %d (the drive at %.0f C; the 10 s before: %.0f MB written, "
                                             "%lld pages swapped out)\n", per, rate, slow_level(), nvme_temp_c(), wmb, sw);
                    }
                } else if (lv == slow_level()) {   // the floor's own samples in the slow mode (moex §71: the slow mode's level)
                    floor_sample(ms, n, at_floor, low_ms, low_n, low_k, drain_ms, drain_n);
                }
                lv_.store(slow_level(), std::memory_order_relaxed);
                good = 0;
                continue;
            }
            if (lv == kTop) continue;
            if (lv == slow_level() && !healthy) floor_sample(ms, n, at_floor, low_ms, low_n, low_k, drain_ms, drain_n);
            good = healthy ? good + 1 : 0;
            if (good >= (lv == slow_level() ? 2 : 1)) {
                lv_.store(lv + 1, std::memory_order_relaxed);
                if (lv + 1 == kTop) {
                    const double s = std::chrono::duration<double>(clk::now() - t_low).count();
                    floor_ms_.fetch_add((int64_t) (s * 1000.0), std::memory_order_relaxed);
                    if (verbose_)
                        std::fprintf(stderr, "strata-x: the drive guard: full depth again after %.1f s (at the slow mode's level the drive took "
                                             "%.2f ms a read, %.0f reads a second; in its first 2 s, the requests already queued "
                                             "draining, %.0f ms a read)\n",
                                     s, low_n > 0 ? low_ms / low_n : 0.0, low_k > 0 ? low_n / (0.25 * low_k) : 0.0,
                                     drain_n > 0 ? drain_ms / drain_n : 0.0);
                }
            }
        }
    }
    /// a sample at the floor: its first 8 (2 s) count as the queue the guard found draining, the rest as the floor's own
    static void floor_sample(uint64_t ms, uint64_t n, int& at_floor, double& low_ms, double& low_n, int& low_k, double& drain_ms,
                             double& drain_n) {
        if (++at_floor <= 8) {
            drain_ms += (double) ms, drain_n += (double) n;
        } else {
            low_ms += (double) ms, low_n += (double) n, ++low_k;
        }
    }
    const std::string stat_;
    const bool hold_, verbose_;
    std::atomic<int> lv_{kTop};
    std::atomic<int64_t> episodes_{0}, floor_ms_{0};
};

}  // namespace dsv41
