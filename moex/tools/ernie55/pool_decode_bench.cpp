// moex (ERNIE55_PLAN.md): the CPU pool as ERNIE-5.5's decode drives it - a layer's few missed experts for one token
// through ExpertPool::run_split_multi_native, a gap between layers while the GPU runs the layer's dense part (the workers
// spin), the experts copied from the GGUF set into the blob form of the engine's RAM copy (2 MiB pages with BENCH_THP=1, as
// STRATA_DSV41_HUGE_PIN), the host pinned as the engine's.  Prints each experts-a-layer count's drain time (median, p90)
// and the bytes over it - the decode line's "GB/s over the drain" - next to the experts' bytes at the kernels' own rate.
//
//   pool_decode_bench GGUF_SHARD LAYER [experts a layer 2 | "mix"] [tasks 252] [seconds 4] [gap us 400] [workers 20]
//
// "mix": the count drawn per layer as the deployed tier's misses (10 routed, each a miss with BENCH_MISS, default 0.2).
// The pool's own switches are read from the environment as in the engine (STRATA_POOL_SPREAD / PIPELINE).  BENCH_STATIC=1:
// the same rows on the same kernels by 20 threads + the host with a fixed contiguous share each and barriers (no pool) - the
// hill-climb's check that the pool's schedule is not the gap (2026-11-25: 286 us against the pool's 193 at 2 experts).
// Build (the dev tree): g++ -O3 -std=c++20 -D_GNU_SOURCE -DSTRATA_NATIVE_EXPERTS=1 -Iinclude -I<ggml>/include ... with
// libstrata_kernels_cpu.a, ggml-cpu, ggml-base, -lpthread -fopenmp.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/glu_act.hpp"

#include "ggml.h"

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        std::fprintf(stderr, "usage: pool_decode_bench GGUF_SHARD LAYER [experts 2|mix] [tasks 252] [seconds 4] [gap_us 400] [workers 20]\n");
        return 2;
    }
    strata::GgufFile g(argv[1]);
    const int layer = std::atoi(argv[2]);
    const bool mix = argc > 3 && std::string(argv[3]) == "mix";
    const int per = argc > 3 && !mix ? std::atoi(argv[3]) : 2, tasks = argc > 4 ? std::atoi(argv[4]) : 252;
    const double secs = argc > 5 ? std::atof(argv[5]) : 4.0;
    const int gap_us = argc > 6 ? std::atoi(argv[6]) : 400, workers = argc > 7 ? std::atoi(argv[7]) : 20;
    const double p_miss = std::getenv("BENCH_MISS") ? std::atof(std::getenv("BENCH_MISS")) : 0.2;
    const strata::TensorInfo* t[3] = {};
    const char* roles[3] = {"gate", "up", "down"};
    for (int r = 0; r < 3; ++r) t[r] = g.find("blk." + std::to_string(layer) + ".ffn_" + roles[r] + "_exps.weight");
    if (!t[0] || !t[1] || !t[2]) {
        std::printf("layer %d: no gate/up/down expert tensors in %s\n", layer, argv[1]);
        return 1;
    }
    const int64_t n_embd = (int64_t) t[0]->shape[0], n_ff = (int64_t) t[0]->shape[1], n_exp = (int64_t) t[0]->shape[2];
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, n_embd, n_ff, f, err)) {
        std::printf("layer %d: %s\n", layer, err.c_str());
        return 1;
    }
    f.swiglu_limit = std::getenv("BENCH_CLAMP") ? (float) std::atof(std::getenv("BENCH_CLAMP")) : strata::kernels::kSituGlu;
    const size_t gsz = f.up_off, dsz = f.bytes - f.down_off;
    // BENCH_EXPERTS distinct experts (default 256, ~1.4 GB): the DRAM, not the caches
    const int NE = std::getenv("BENCH_EXPERTS") ? std::max(32, std::atoi(std::getenv("BENCH_EXPERTS"))) : 256;
    const size_t stride = (f.bytes + 4095) / 4096 * 4096, total = (size_t) NE * stride;
    const bool thp = std::getenv("BENCH_THP") && std::getenv("BENCH_THP")[0] == '1';
    const size_t alloc = thp ? (total + (2u << 20) - 1) / (2u << 20) * (2u << 20) : total;
    auto* blobs = (uint8_t*) std::aligned_alloc(thp ? (2u << 20) : 4096, alloc);
    if (!blobs) return 1;
    madvise(blobs, alloc, thp ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    {
        std::vector<std::thread> st;
        std::atomic<int> nx{0};
        for (int i = 0; i < 16; ++i)
            st.emplace_back([&] {
                for (int e; (e = nx.fetch_add(1)) < NE;) {
                    const size_t E = (size_t) ((e * 37 + 5) % n_exp);
                    uint8_t* b = blobs + (size_t) e * stride;
                    std::memcpy(b, g.tensor_data(*t[0]) + E * gsz, gsz);
                    std::memcpy(b + f.up_off, g.tensor_data(*t[1]) + E * gsz, gsz);
                    std::memcpy(b + f.down_off, g.tensor_data(*t[2]) + E * dsz, dsz);
                    std::memset(b + f.bytes, 0, stride - f.bytes);
                }
            });
        for (auto& x : st) x.join();
    }
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) n_embd);
    for (auto& v : x) v = nd(rng);
    std::vector<uint8_t> act(cpu::kNativeActBytes);
    cpu::native_quant_act(f, x.data(), act.data());
    const int maxper = mix ? 10 : per;
    std::vector<float> out((size_t) maxper * (size_t) n_embd);
    const bool stat_mode = std::getenv("BENCH_STATIC") && std::getenv("BENCH_STATIC")[0] == '1';
    std::unique_ptr<cpu::ExpertPool> pool_p;   // (BENCH_STATIC: none - its workers would spin on the team's cores)
    if (!stat_mode) pool_p = std::make_unique<cpu::ExpertPool>(workers, true, true, cpu::PoolAffinity::All, tasks);
    if (const char* e = std::getenv("BENCH_PIN_HOST"); !e || e[0] != '0') {   // as the engine: the host on the first physical core
        const int hc = cpu::planned_host_core();
        cpu::pin_current_thread(hc);
    }
    std::printf("layer %d: gate/up %s, down %s, %lld x %lld, expert %.3f MB, %d experts (%.0f MiB, %s pages), %d workers + the host, "
                "%d tasks, gap %d us, SPREAD %s PIPELINE %s\n",
                layer, ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), (long long) n_embd,
                (long long) n_ff, f.bytes / 1e6, NE, total / 1048576.0, thp ? "2 MiB" : "4 KiB", workers, tasks, gap_us,
                std::getenv("STRATA_POOL_SPREAD") ? std::getenv("STRATA_POOL_SPREAD") : "-",
                std::getenv("STRATA_POOL_PIPELINE") ? std::getenv("STRATA_POOL_PIPELINE") : "-");
    // BENCH_STATIC=1: the layer's work without the pool's scheduler - 20 threads on the pool's cores + the host, each a fixed
    // contiguous share of the gate/up rows (all experts' rows in a row), a barrier, each expert's h quantized by one thread, a
    // barrier, a fixed share of the down rows, a barrier: what a scheduler-free split of the same kernels takes
    const bool stat = std::getenv("BENCH_STATIC") && std::getenv("BENCH_STATIC")[0] == '1';
    struct Team {
        std::atomic<uint32_t> epoch{0}, gen{0};
        std::atomic<int> bar[3] = {0, 0, 0};
        std::atomic<bool> stop{false};
    };
    Team team;
    std::vector<std::thread> tthreads;
    const int NT = workers + 1;
    std::vector<std::vector<float>> sff;
    std::vector<std::vector<uint8_t>> shq;
    int cur_n = 0;
    std::vector<const uint8_t*> cur_blob;
    auto team_work = [&](int id, int g) {   // the share of thread id (0 = the host) in layer generation g
        const int n = cur_n;
        const int64_t R1 = (int64_t) n * f.n_ff, R2 = (int64_t) n * f.n_embd;
        auto rows = [&](int64_t R, int64_t per, auto fn) {
            const int64_t g0 = R * id / NT, g1 = R * (id + 1) / NT;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                fn(e, r0, r1);
                r += r1 - r0;
            }
        };
        auto barrier = [&](int b) {   // counts that only grow: layer g's barrier b is passed at NT x g arrivals
            team.bar[b].fetch_add(1, std::memory_order_acq_rel);
            while (team.bar[b].load(std::memory_order_acquire) < NT * g) _mm_pause();
        };
        rows(R1, f.n_ff, [&](int e, int r0, int r1) {
            const void* a[1] = {act.data()};
            float* ff[1] = {sff[(size_t) e].data()};
            cpu::native_gu_rows(f, cur_blob[(size_t) e], a, 1, ff, r0, r1);
        });
        barrier(0);
        if (id < n) cpu::native_quant_h(f, sff[(size_t) id].data(), shq[(size_t) id].data());
        barrier(1);
        rows(R2, f.n_embd, [&](int e, int r0, int r1) {
            const void* h[1] = {shq[(size_t) e].data()};
            float* o[1] = {out.data() + (size_t) e * (size_t) n_embd};
            cpu::native_down_rows(f, cur_blob[(size_t) e], h, 1, o, r0, r1);
        });
        barrier(2);
    };
    if (stat) {
        sff.assign(10, std::vector<float>((size_t) f.n_ff));
        shq.assign(10, std::vector<uint8_t>(f.h_bytes + 64));
        cur_blob.assign(10, nullptr);
        const cpu::CpuTopology topo = cpu::detect_cpu_topology(true, cpu::PoolAffinity::All);
        std::printf("static team: %zu worker cores from the topology:", topo.worker_cores.size());
        for (int c : topo.worker_cores) std::printf(" %d", c);
        std::printf("\n");
        for (int i = 1; i < NT; ++i)
            tthreads.emplace_back([&, i] {
                // the pool's placement: worker i - 1 on the topology's core i - 1 (a set mask: the host's pin is inherited otherwise)
                const int core = (size_t) (i - 1) < topo.worker_cores.size() ? topo.worker_cores[(size_t) (i - 1)] : i;
                cpu_set_t cs;
                CPU_ZERO(&cs);
                CPU_SET(core, &cs);
                if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0) std::printf("static team: pin to %d failed\n", core);
                uint32_t seen = 0;
                for (;;) {
                    uint32_t e;
                    while ((e = team.epoch.load(std::memory_order_acquire)) == seen) {
                        if (team.stop.load(std::memory_order_relaxed)) return;
                        _mm_pause();
                    }
                    seen = e;
                    team_work(i, (int) team.gen.load(std::memory_order_acquire));
                }
            });
    }
    std::vector<cpu::ExpertJobMulti> jobs((size_t) maxper);
    int next = 0;
    std::bernoulli_distribution miss(p_miss);
    auto layer_run = [&](int n) {
        for (int i = 0; i < n; ++i) {
            cpu::ExpertJobMulti& j = jobs[(size_t) i];
            j = cpu::ExpertJobMulti{};
            j.blob = blobs + (size_t) next * stride;
            next = (int) (((int64_t) next + 97) % NE);
            j.nt = 1;
            j.nact[0] = act.data();
            j.out[0] = out.data() + (size_t) i * (size_t) n_embd;
        }
        if (stat) {
            cur_n = n;
            for (int i = 0; i < n; ++i) cur_blob[(size_t) i] = jobs[(size_t) i].blob;
            const int g = (int) team.gen.fetch_add(1, std::memory_order_acq_rel) + 1;   // (before the epoch: a woken worker reads it)
            const double a = now_s();
            team.epoch.fetch_add(1, std::memory_order_acq_rel);
            team_work(0, g);
            return now_s() - a;
        }
        const double a = now_s();
        pool_p->run_split_multi_native(f, jobs.data(), n);
        return now_s() - a;
    };
    for (int i = 0; i < 300; ++i) layer_run(mix ? 2 : per);   // warm
    std::vector<std::vector<double>> by_n(11);
    const double t_end = now_s() + secs;
    while (now_s() < t_end) {
        int n = per;
        if (mix) {
            n = 0;
            for (int k = 0; k < 10; ++k) n += miss(rng) ? 1 : 0;
        }
        if (n > 0) by_n[(size_t) n].push_back(layer_run(n));
        const double g0 = now_s();
        while ((now_s() - g0) * 1e6 < gap_us) {
        }
    }
    double all_t = 0, all_b = 0;
    if (std::getenv("BENCH_DEBUG"))
        for (int n = 1; n <= 10; ++n)
            if (!by_n[(size_t) n].empty()) std::printf("  (debug) %d experts: %zu layers\n", n, by_n[(size_t) n].size());
    for (int n = 1; n <= 10; ++n) {
        auto& v = by_n[(size_t) n];
        if (v.size() < 20) continue;
        std::sort(v.begin(), v.end());
        double sum = 0;
        for (double d : v) sum += d;
        const double med = v[v.size() / 2], p90 = v[v.size() * 9 / 10], b = (double) n * (double) f.bytes;
        all_t += sum;
        all_b += b * (double) v.size();
        std::printf("  %2d experts: %6zu layers, drain median %7.1f us (p90 %7.1f, mean %7.1f), %.2f GB/s at the median; at 68.5 GB/s %6.1f us\n",
                    n, v.size(), 1e6 * med, 1e6 * p90, 1e6 * sum / (double) v.size(), b / med / 1e9, 1e6 * b / 68.5e9);
    }
    std::printf("  all: %.2f GB/s over the drains%s\n", all_b / all_t / 1e9, stat ? " (BENCH_STATIC: no pool)" : "");
    if (stat) {
        team.stop.store(true);
        for (auto& t2 : tthreads) t2.join();
    }
    std::free(blobs);
    return 0;
}
