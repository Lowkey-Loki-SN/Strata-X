// moex (DSV41_PLAN §33): the CPU pool as decode drives it - a layer's few experts for one token through
// ExpertPool::run_split_multi_native (the gate/up rows in one phase, the down rows in another), a gap between layers
// as the GPU's part of a layer leaves the workers spinning, the experts from a pack's record file in 4 KB pages (the
// engine's RAM copy and SSD cache).  Prints the bytes over the drains' time (the serve decode line's "decode CPU pool").
//
//   pool_phase_bench RECORDS GU_TYPE D_TYPE [experts a layer 2] [tasks 252] [seconds 4] [gap us 500] [workers 20]
//
// RECORDS: one layer's record file of the pack (experts/L05.bin), GU_TYPE / D_TYPE its ggml types (experts.tsv).
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <sys/mman.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 4) {
        std::fprintf(stderr, "usage: pool_phase_bench RECORDS GU_TYPE D_TYPE [experts 2] [tasks 252] [seconds 4] [gap_us 500] [workers 20]\n");
        return 2;
    }
    const int gt = std::atoi(argv[2]), dt = std::atoi(argv[3]);
    const int per = argc > 4 ? std::atoi(argv[4]) : 2, tasks = argc > 5 ? std::atoi(argv[5]) : 252;
    const double secs = argc > 6 ? std::atof(argv[6]) : 4.0;
    const int gap_us = argc > 7 ? std::atoi(argv[7]) : 500, workers = argc > 8 ? std::atoi(argv[8]) : 20;
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gt, dt, 5120, 2304, f, err)) {
        std::printf("%s\n", err.c_str());
        return 1;
    }
    // 128 distinct experts (1.7 GB): the DRAM, not the caches; BENCH_EXPERTS=N more, copies of those 128 (the engine's RAM
    // copy and SSD cache span ~82 GB of 4 KB pages: their page tables do not fit the caches either)
    const int NE = std::getenv("BENCH_EXPERTS") ? std::max(128, std::atoi(std::getenv("BENCH_EXPERTS"))) : 128;
    const size_t stride = (f.bytes + 4095) / 4096 * 4096, total = (size_t) NE * stride;
    // BENCH_THP=1: 2 MiB pages (MADV_HUGEPAGE on a 2 MiB-aligned buffer), else 4 KB pages as the engine's
    const bool thp = std::getenv("BENCH_THP") && std::getenv("BENCH_THP")[0] == '1';
    const size_t alloc = thp ? (total + (2u << 20) - 1) / (2u << 20) * (2u << 20) : total;
    auto* blobs = (uint8_t*) std::aligned_alloc(thp ? (2u << 20) : 4096, alloc);
    if (!blobs) return 1;
    madvise(blobs, alloc, thp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    FILE* rf = std::fopen(argv[1], "rb");
    if (!rf) {
        std::printf("cannot open %s\n", argv[1]);
        return 1;
    }
    for (int e = 0; e < NE; ++e) {
        if (e >= 128) {
            std::memcpy(blobs + (size_t) e * stride, blobs + (size_t) (e % 128) * stride, f.bytes);
            continue;
        }
        const long long E = (e * 37 + 5) % 384;
        std::fseek(rf, (long) (E * (long long) stride), SEEK_SET);
        if (std::fread(blobs + (size_t) e * stride, 1, f.bytes, rf) != f.bytes) {
            std::printf("short read\n");
            return 1;
        }
    }
    std::fclose(rf);
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(5120);
    for (auto& v : x) v = nd(rng);
    std::vector<uint8_t> act(cpu::kNativeActBytes);
    cpu::native_quant_act(f, x.data(), act.data());
    std::vector<float> out((size_t) per * 5120);
    cpu::ExpertPool pool(workers, true, true, cpu::PoolAffinity::All, tasks);
    if (const char* e = std::getenv("BENCH_PIN_HOST"); e && e[0] == '1') {   // as the engine: the host on the first physical core
        const int hc = cpu::planned_host_core();
        const cpu::ThreadAffinity a = cpu::pin_current_thread(hc);
        std::printf("the host pinned to core %d: %s\n", hc, a.valid ? "yes" : "no");
    }
    std::vector<cpu::ExpertJobMulti> jobs((size_t) per);
    int next = 0;
    auto layer = [&] {
        for (int i = 0; i < per; ++i) {
            cpu::ExpertJobMulti& j = jobs[(size_t) i];
            j = cpu::ExpertJobMulti{};
            j.blob = blobs + (size_t) next * stride;
            next = (int) (((int64_t) next + 977) % NE);   // (a stride across the whole set)
            j.nt = 1;
            j.nact[0] = act.data();
            j.out[0] = out.data() + (size_t) i * 5120;
        }
        // BENCH_HOT=1: the same layer drained once untimed first - the experts then where that drain left them in the caches
        // (the best a read into the caches during the GPU's part of a layer could leave)
        static const bool hot = std::getenv("BENCH_HOT") && std::getenv("BENCH_HOT")[0] == '1';
        if (hot) pool.run_split_multi_native(f, jobs.data(), per);
        const double a = now_s();
        pool.run_split_multi_native(f, jobs.data(), per);
        return now_s() - a;
    };
    for (int i = 0; i < 200; ++i) layer();   // warm
    double drain = 0;
    long long n = 0;
    const double t_end = now_s() + secs;
    while (now_s() < t_end) {
        drain += layer();
        ++n;
        const double g0 = now_s();
        while ((now_s() - g0) * 1e6 < gap_us) {
        }
    }
    const double bytes = (double) n * per * (double) f.bytes;
    std::printf("%d experts a layer, %d tasks, %d workers + the host, gap %d us: %lld layers, %.1f us a layer's drain, %.2f GB/s "
                "(the expert %.2f MiB)\n", per, tasks, workers, gap_us, n, 1e6 * drain / (double) n, bytes / drain / 1e9,
                f.bytes / 1048576.0);
    std::free(blobs);
    return 0;
}
