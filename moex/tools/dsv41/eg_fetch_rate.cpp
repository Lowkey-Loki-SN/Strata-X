// moex/tools/dsv41/eg_fetch_rate.cpp - the engram fetch's rate without the engine (DSV41_PLAN §64): TOKENS positions x 2
// layers x 24 rows, uniform at random over L01 / L14's weight tables (the drive's cost a row, not the n-gram statistics; the
// scales are left out - EG_SCALES_PRELOAD keeps them in the page cache), the weight tables dropped from the page cache first.
//
//   eg_fetch_rate MODE TOKENS [RECORD_THREADS] [SEED]
// MODE  mapped:W   eg_rows' mapped path: one thread, the rows' pages asked for (MADV_WILLNEED) W rows ahead of the copies
//       direct:N   N threads, each row's 4 KB page read O_DIRECT, the rows in the copies' order
//       sorted:N   N threads, each layer's rows sorted and deduplicated first (block order on the drive)
// RECORD_THREADS  threads reading 4 MiB pieces of the q3q2u experts O_DIRECT meanwhile (the record stream)
// Prints the fetch's time, its rows a second, and the drive's counters (/sys/block/nvme0n1/stat) over it.  Reads only.
// Build: g++ -O2 -std=c++17 -pthread -o eg_fetch_rate moex/tools/dsv41/eg_fetch_rate.cpp
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
// Relative to the checkout's folder (run it from there): the models folder beside it, moex/env.sh's STRATA_MODELS default.
const char* kEg[2] = {"../Models/dsv41-1017/native/engram/L01.weight.bin",
                      "../Models/dsv41-1017/native/engram/L14.weight.bin"};
const char* kExperts = "../Models/dsv41-1017/ud-iq3xxs/experts";
constexpr int D = 256, C = 24;

struct Stat {
    uint64_t reads = 0, ms_read = 0, ms_busy = 0;
};
Stat stat_now() {
    Stat s;
    FILE* f = std::fopen("/sys/block/nvme0n1/stat", "r");
    if (!f) return s;
    unsigned long long v[11] = {};
    for (int i = 0; i < 11; ++i)
        if (std::fscanf(f, "%llu", &v[i]) != 1) break;
    std::fclose(f);
    s.reads = v[0];
    s.ms_read = v[3];
    s.ms_busy = v[9];
    return s;
}
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s mapped:W|direct:N|sorted:N TOKENS [RECORD_THREADS] [SEED]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    const int ntok = std::atoi(argv[2]);
    const int nrec = argc > 3 ? std::atoi(argv[3]) : 0;
    const uint64_t seed = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    const size_t colon = mode.find(':');
    const std::string kind = mode.substr(0, colon);
    const int arg = colon == std::string::npos ? 0 : std::atoi(mode.c_str() + colon + 1);
    if ((kind != "mapped" && kind != "direct" && kind != "sorted") || arg < 1 || ntok < 1) {
        std::fprintf(stderr, "bad mode or tokens\n");
        return 2;
    }
    int fd[2], dfd[2];
    uint64_t nrows[2];
    const uint8_t* map[2];
    for (int j = 0; j < 2; ++j) {
        fd[j] = open(kEg[j], O_RDONLY | O_CLOEXEC);
        dfd[j] = open(kEg[j], O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd[j] < 0 || dfd[j] < 0) {
            std::perror(kEg[j]);
            return 1;
        }
        struct stat st;
        fstat(fd[j], &st);
        nrows[j] = (uint64_t) st.st_size / D;
        posix_fadvise(fd[j], 0, 0, POSIX_FADV_DONTNEED);   // cold, as after a reboot
        void* m = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd[j], 0);
        if (m == MAP_FAILED) {
            std::perror("mmap");
            return 1;
        }
        madvise(m, (size_t) st.st_size, MADV_RANDOM);
        map[j] = static_cast<const uint8_t*>(m);
    }
    // the plan: rows in the copies' order (position t, layer j, column c)
    std::mt19937_64 rng(seed);
    const int64_t n = (int64_t) ntok * 2 * C;
    std::vector<uint64_t> rows((size_t) n);
    for (int64_t q = 0; q < n; ++q) rows[(size_t) q] = rng() % nrows[(q / C) % 2];
    std::vector<uint8_t> out((size_t) n * D);

    std::atomic<bool> stop{false};
    std::vector<std::thread> rec;
    std::vector<std::string> files;
    for (int l = 0; l < 43; ++l) {
        char p[512];
        std::snprintf(p, sizeof p, "%s/L%02d.bin", kExperts, l);
        if (access(p, R_OK) == 0) files.push_back(p);
    }
    for (int r = 0; r < nrec && !files.empty(); ++r)
        rec.emplace_back([&, r] {
            const size_t piece = (size_t) 4 << 20;
            void* buf = nullptr;
            if (posix_memalign(&buf, 4096, piece) != 0) return;
            std::mt19937_64 rr(seed ^ (0x9e3779b97f4a7c15ull * (uint64_t) (r + 1)));
            while (!stop.load(std::memory_order_relaxed)) {
                const std::string& f = files[rr() % files.size()];
                const int x = open(f.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
                if (x < 0) break;
                struct stat st;
                fstat(x, &st);
                const off_t off = (off_t) (rr() % ((uint64_t) st.st_size / piece)) * (off_t) piece;
                if (pread(x, buf, piece, off) < 0) {
                    close(x);
                    break;
                }
                close(x);
            }
            std::free(buf);
        });
    if (nrec > 0) std::this_thread::sleep_for(std::chrono::seconds(1));   // the record stream under way

    const Stat s0 = stat_now();
    const double t0 = now_s();
    int64_t reads = 0;
    if (kind == "mapped") {
        const uintptr_t pg = 4096;
        auto want = [&](int64_t q) {
            const uint8_t* a = map[(q / C) % 2] + rows[(size_t) q] * D;
            madvise((void*) ((uintptr_t) a & ~(pg - 1)), pg, MADV_WILLNEED);
        };
        int64_t asked = 0;
        for (int64_t q = 0; q < n; ++q) {
            for (const int64_t lim = std::min<int64_t>(n, q + arg); asked < lim; ++asked) want(asked);
            std::memcpy(out.data() + (size_t) q * D, map[(q / C) % 2] + rows[(size_t) q] * D, D);
        }
        reads = n;
    } else {
        // the reads: (layer, row, where its copies go) - in the copies' order, or each layer's sorted and deduplicated
        struct Rd {
            int j;
            uint64_t row;
        };
        std::vector<Rd> rd;
        rd.reserve((size_t) n);
        if (kind == "direct") {
            for (int64_t q = 0; q < n; ++q) rd.push_back({(int) ((q / C) % 2), rows[(size_t) q]});
        } else {
            for (int j = 0; j < 2; ++j) {
                std::vector<uint64_t> u;
                for (int64_t q = 0; q < n; ++q)
                    if ((q / C) % 2 == j) u.push_back(rows[(size_t) q]);
                std::sort(u.begin(), u.end());
                u.erase(std::unique(u.begin(), u.end()), u.end());
                for (uint64_t r : u) rd.push_back({j, r});
            }
        }
        std::vector<uint8_t> got(rd.size() * D);
        std::vector<std::thread> th;
        std::atomic<bool> bad{false};
        for (int w = 0; w < arg; ++w)
            th.emplace_back([&, w] {
                void* page = nullptr;
                if (posix_memalign(&page, 4096, 4096) != 0) {
                    bad = true;
                    return;
                }
                for (size_t k = (size_t) w; k < rd.size(); k += (size_t) arg) {
                    const uint64_t at = rd[k].row * D, p = at & ~(uint64_t) 4095;
                    if (pread(dfd[rd[k].j], page, 4096, (off_t) p) < (ssize_t) (at - p + D)) {
                        bad = true;
                        break;
                    }
                    std::memcpy(got.data() + k * D, static_cast<uint8_t*>(page) + (at - p), D);
                }
                std::free(page);
            });
        for (auto& x : th) x.join();
        if (bad) {
            std::fprintf(stderr, "a read failed\n");
            return 1;
        }
        reads = (int64_t) rd.size();
    }
    const double dt = now_s() - t0;
    const Stat s1 = stat_now();
    stop = true;
    for (auto& x : rec) x.join();
    const double dr = (double) (s1.reads - s0.reads);
    std::printf("%s, %d tokens: %lld rows, %lld page reads in %.2f s (%.0f rows/s, %.0f reads/s); the drive over it: %.0f "
                "requests/s, %.2f ms a request, busy %.0f%% (%d record threads)\n",
                mode.c_str(), ntok, (long long) n, (long long) reads, dt, n / dt, reads / dt, dr / dt,
                dr > 0 ? (double) (s1.ms_read - s0.ms_read) / dr : 0.0, 100.0 * (double) (s1.ms_busy - s0.ms_busy) / (dt * 1000.0), nrec);
    return 0;
}
