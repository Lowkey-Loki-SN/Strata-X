// drive_guard_test.cpp - DSV41_PLAN §18: the drive guard's rules on counters this test writes (no drive, no engine).
//   g++ -std=c++17 -O1 -pthread -I moex/tools/dsv41 moex/tools/dsv41/checks/drive_guard_test.cpp -o drive_guard_test && ./drive_guard_test
// (the counters' file in $TMPDIR, else /tmp)
// A healthy drive (8,000 reads a second at 0.2 ms), then the slow mode's counters (1,000 a second at 1,000 ms) for 6 s, a
// sample between the two (60 ms a read) that neither engages nor releases, healthy again, the slow mode again: the level
// after each step, how soon it dropped to the floor and came back, and the limits each level gives the readers.
#include "drive_guard.hpp"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace {
// 3 (moex, DSV41_BACKLOG #99): a decode's bursts at the floor on a healthy drive - 26 reads a 50 ms (520 a second) at 0.07 ms,
//   busy 2 ms of the 50; 4: the slow mode at the floor - 50 reads a 50 ms (1,000 a second) at 1 ms, busy the whole 50
std::atomic<int> mode{0};   // 0 healthy, 1 the slow mode, 2 between (60 ms a read, 3,000 a second)
std::atomic<bool> quit{false};
unsigned long long reads = 0, ms = 0, busy = 0;
const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/drive_guard_test_" +
                         std::to_string(getpid()) + ".stat";

void writer() {   // every 50 ms, as the kernel's counters move
    while (!quit) {
        const int m = mode.load();
        const unsigned long long n = m == 0 ? 400 : m == 1 ? 50 : m == 2 ? 150 : m == 3 ? 26 : 50;
        reads += n;
        ms += m == 0 ? n / 5 : m == 1 ? n * 1000 : m == 2 ? n * 60 : m == 3 ? 2 : n;
        busy += m == 3 ? 2 : 50;
        const std::string tmp = path + ".tmp";
        if (FILE* f = std::fopen(tmp.c_str(), "w")) {
            std::fprintf(f, "%llu 0 0 %llu 0 0 0 0 0 %llu 0\n", reads, ms, busy);
            std::fclose(f);
            std::rename(tmp.c_str(), path.c_str());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

int fails = 0;
void expect(bool ok, const char* what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what);
    fails += !ok;
}
double wait_for(const dsv41::DriveGuard& g, bool low, double limit_s) {   // seconds until the level is below / at kTop
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if ((g.level() < dsv41::DriveGuard::kTop) == low || s > limit_s) return s;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
}  // namespace

int main() {
    std::thread w(writer);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    dsv41::DriveGuard* g = dsv41::DriveGuard::start(path, false, true);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    expect(g->level() == dsv41::DriveGuard::kTop && g->readers() == INT_MAX && g->call_bytes() == UINT64_MAX &&
               g->eg_rows() == INT_MAX,
           "a healthy drive: full depth, no limits (the readers' own)");
    mode = 1;
    const double t_low = wait_for(*g, true, 3.0);
    std::printf("      the floor %.2f s after the slow mode's counters began\n", t_low);
    expect(t_low <= 0.6, "the slow mode: the floor within 0.6 s (a 250 ms sample, and the one it began in)");
    expect(g->level() == 0 && g->readers() == 1 && g->call_bytes() == (128u << 10) && g->eg_rows() == 1,
           "the floor: one reader, 128 KB a call, the engram rows one at a time");
    std::this_thread::sleep_for(std::chrono::seconds(5));
    expect(g->level() == 0, "the slow mode for 5 s more: still at the floor");
    mode = 2;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    expect(g->level() == 0, "60 ms a read at 3,000 a second (neither slow nor healthy): it holds the floor");
    mode = 0;
    const double t_full = wait_for(*g, false, 5.0);
    std::printf("      full depth %.2f s after the counters were healthy again\n", t_full);
    expect(t_full <= 2.0, "healthy again: full depth within 2 s (two samples, then a level a sample)");
    std::this_thread::sleep_for(std::chrono::seconds(1));
    mode = 1;
    const double t_low2 = wait_for(*g, true, 3.0);
    expect(t_low2 <= 0.6, "the slow mode again: the floor again within 0.6 s");
    std::this_thread::sleep_for(std::chrono::seconds(3));
    mode = 0;
    expect(wait_for(*g, false, 5.0) <= 2.0, "and back");
    expect(g->episodes() == 2, "two episodes counted");
    // moex (#99): a decode after an episode - its bursts on a healthy drive bring the guard back; the slow mode's floor holds
    std::this_thread::sleep_for(std::chrono::seconds(1));
    mode = 1;
    expect(wait_for(*g, true, 3.0) <= 0.6, "a third episode: the floor");
    std::this_thread::sleep_for(std::chrono::seconds(1));
    mode = 4;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    expect(g->level() == 0, "the slow mode at the floor (1,000 reads a busy second at 1 ms): it holds the floor");
    mode = 3;
    const double t_dec = wait_for(*g, false, 5.0);
    std::printf("      full depth %.2f s after a decode's bursts on a healthy drive began (520 reads a second, 13,000 a busy second)\n", t_dec);
    expect(t_dec <= 2.0, "a decode's bursts on a healthy drive: full depth within 2 s (#99: the drive's rate while busy)");
    quit = true;
    w.join();
    std::remove(path.c_str());
    std::printf("%s\n", fails ? "SOME FAILED" : "ALL PASS");
    return fails ? 1 : 0;
}
