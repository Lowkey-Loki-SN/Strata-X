// moex/tools/dsv41/hugepage_bench.cu - what 2 MiB pages change on this box (the dsx audit, 2026-10-23): host-to-device
// copies from pinned host memory - the IOMMU translates the GPUs' reads here (default domain "Translated", DMA-FQ) - and
// the CPU pool's streaming reads, 4 KiB pages against transparent huge pages, at the sizes the engine moves.
//
//   nvcc -O3 -std=c++17 -arch=sm_86 -Xcompiler -mavx2,-pthread moex/tools/dsv41/hugepage_bench.cu -o build-dsv41/hugepage_bench
//   moex_gpus_free && CUDA_DEVICE_ORDER=PCI_BUS_ID build-dsv41/hugepage_bench [GIB (8)]
//
// Three arms of GIB GiB each, one after the other:
//   4k     anonymous memory as STRATA_DSV41_FAST_PIN maps it today (no MADV_HUGEPAGE: 4 KiB pages with THP "madvise")
//   thp    the same 2 MiB-aligned, MADV_HUGEPAGE before a page is touched (transparent huge pages)
//   alloc  cudaHostAlloc, FAST_PIN's fallback
// faulted in by 20 threads, registered portable | mapped (as pin_blocks does), then:
//   - copies to each GPU and to both at once: whole records (12,812,288 bytes, q3q2u's Q2_K/Q3_K layers) and 4 MiB pieces,
//     from random 4 KiB-aligned offsets over the whole buffer (the engine's spread, not one hot region), twice each;
//   - the CPU's reads: 20 threads, one a physical core (CPUs 0-19), each streaming whole records at random offsets.
// Refuses to start while a GPU holds more than 1.5 GiB (a server running; asked of nvidia-smi before any CUDA call, so a busy
// GPU gets no context) or RAM is short.
#include <cuda_runtime.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        const cudaError_t e_ = (x);                                                                \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "hugepage_bench: %s: %s\n", #x, cudaGetErrorString(e_));          \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

static constexpr size_t kRecord = 12812288, kPiece = 4u << 20, kHuge = 2u << 20;
static constexpr int kThreads = 20;

static void pin(int cpu) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

/// the share of [p, p + n) that huge pages back (smaps' AnonHugePages of the mappings over it)
static double thp_share(const void* p, size_t n) {
    std::ifstream f("/proc/self/smaps");
    std::string line;
    const uintptr_t a = (uintptr_t) p, b = a + n;
    bool in = false;
    double huge = 0;
    while (std::getline(f, line)) {
        unsigned long lo = 0, hi = 0;
        if (std::sscanf(line.c_str(), "%lx-%lx ", &lo, &hi) == 2) {
            in = lo < b && hi > a;
            continue;
        }
        if (in && line.rfind("AnonHugePages:", 0) == 0) huge += std::atof(line.c_str() + 14) * 1024.0;
    }
    return huge / (double) n;
}

static uint64_t mem_available() {
    std::ifstream f("/proc/meminfo");
    std::string k;
    uint64_t v = 0;
    while (f >> k >> v) {
        if (k == "MemAvailable:") return v << 10;
        f.ignore(256, '\n');
    }
    return 0;
}

struct Arm {
    const char* name;
    uint8_t* p = nullptr;
    size_t n = 0;
    void* raw = nullptr;
    size_t raw_n = 0;
    double fault_s = 0, reg_s = 0, thp = 0;
};

static Arm make(const char* name, size_t n) {
    Arm m;
    m.name = name;
    m.n = n;
    if (std::strcmp(name, "alloc") == 0) {
        const auto t = clk::now();
        CK(cudaHostAlloc((void**) &m.p, n, cudaHostAllocPortable));
        m.reg_s = secs(t);
        return m;
    }
    m.raw_n = n + kHuge;
    m.raw = mmap(nullptr, m.raw_n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m.raw == MAP_FAILED) {
        std::perror("hugepage_bench: mmap");
        std::exit(1);
    }
    m.p = (uint8_t*) (((uintptr_t) m.raw + kHuge - 1) & ~(uintptr_t) (kHuge - 1));
    madvise(m.p, n, MADV_DONTFORK);
    if (std::strcmp(name, "thp") == 0 && madvise(m.p, n, MADV_HUGEPAGE) != 0) std::perror("hugepage_bench: MADV_HUGEPAGE");
    auto t = clk::now();
    std::vector<std::thread> ts;   // (pin_blocks' way: a write a 4 KiB page, by several threads)
    for (int k = 0; k < kThreads; ++k)
        ts.emplace_back([&, k] {
            pin(k);
            const size_t pages = n / 4096, per = (pages + kThreads - 1) / kThreads;
            volatile uint8_t* q = m.p;
            for (size_t pg = k * per; pg < std::min(pages, (k + 1) * per); ++pg) q[pg * 4096] = 1;
        });
    for (auto& x : ts) x.join();
    m.fault_s = secs(t);
    t = clk::now();
    CK(cudaHostRegister(m.p, n, cudaHostRegisterPortable | cudaHostRegisterMapped));
    m.reg_s = secs(t);
    m.thp = thp_share(m.p, n);
    return m;
}

static void drop(Arm& m) {
    if (m.raw) {
        CK(cudaHostUnregister(m.p));
        munmap(m.raw, m.raw_n);
    } else {
        CK(cudaFreeHost(m.p));
    }
}

/// GB/s of `count` copies of `piece` bytes from random 4 KiB-aligned offsets of the arm to GPU `dev`
static double h2d(const Arm& m, int dev, size_t piece, int count, uint64_t seed) {
    CK(cudaSetDevice(dev));
    void* d = nullptr;
    CK(cudaMalloc(&d, piece * 4));
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::mt19937_64 rng(seed);
    const size_t slots = (m.n - piece) / 4096;
    CK(cudaMemcpyAsync(d, m.p, piece, cudaMemcpyHostToDevice, s));   // (warm: the context, the first translation)
    CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < count; ++i)
        CK(cudaMemcpyAsync((uint8_t*) d + (size_t) (i % 4) * piece, m.p + (rng() % slots) * 4096, piece, cudaMemcpyHostToDevice, s));
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    CK(cudaStreamDestroy(s));
    CK(cudaFree(d));
    return (double) piece * count / (ms * 1e6);
}

/// GB/s of 20 pinned threads each streaming `per` whole records from random 4 KiB-aligned offsets
static double cpu_read(const Arm& m, int per, uint64_t seed) {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<uint64_t> sink(kThreads * 8);
    std::vector<std::thread> ts;
    const size_t slots = (m.n - kRecord) / 4096;
    for (int k = 0; k < kThreads; ++k)
        ts.emplace_back([&, k] {
            pin(k);
            std::mt19937_64 rng(seed + k);
            std::vector<size_t> offs(per);
            for (auto& o : offs) o = (rng() % slots) * 4096;
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) {
            }
            __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
            for (size_t o : offs) {
                const __m256i* q = (const __m256i*) (m.p + o);
                for (size_t j = 0; j + 4 <= kRecord / 32; j += 4) {
                    a0 = _mm256_add_epi64(a0, _mm256_load_si256(q + j));
                    a1 = _mm256_add_epi64(a1, _mm256_load_si256(q + j + 1));
                    a2 = _mm256_add_epi64(a2, _mm256_load_si256(q + j + 2));
                    a3 = _mm256_add_epi64(a3, _mm256_load_si256(q + j + 3));
                }
            }
            const __m256i a = _mm256_add_epi64(_mm256_add_epi64(a0, a1), _mm256_add_epi64(a2, a3));
            sink[(size_t) k * 8] = (uint64_t) _mm256_extract_epi64(a, 0);
        });
    while (ready.load() < kThreads) {
    }
    const auto t = clk::now();
    go.store(true, std::memory_order_release);
    for (auto& x : ts) x.join();
    const double s = secs(t);
    return (double) kThreads * per * kRecord / s / 1e9;
}

/// the most VRAM in use on any GPU, MiB, from nvidia-smi - before any CUDA call, so a running server's GPUs get no context
static long gpu_mib_used() {
    FILE* p = popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!p) return -1;
    long most = -1, v = 0;
    while (std::fscanf(p, "%ld", &v) == 1) most = std::max(most, v);
    pclose(p);
    return most;
}

static std::string first_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s;
}

int main(int argc, char** argv) {
    const size_t gib = argc > 1 ? (size_t) std::atoi(argv[1]) : 8;
    const size_t n = gib << 30;
    const long used = gpu_mib_used();
    if (used < 0 || used > 1536) {
        std::fprintf(stderr, "hugepage_bench: a GPU holds %ld MiB (or nvidia-smi failed) - a server is running; stop it first\n", used);
        return 1;
    }
    if (mem_available() < 2 * n + (4ull << 30)) {
        std::fprintf(stderr, "hugepage_bench: %.1f GiB of RAM available, %zu GiB needed\n", mem_available() / 1073741824.0,
                     2 * gib + 4);
        return 1;
    }
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    ndev = std::min(ndev, 2);
    std::printf("hugepage_bench: %zu GiB an arm; THP %s\n", gib,
                first_line("/sys/kernel/mm/transparent_hugepage/enabled").c_str());
    for (int d = 0; d < ndev; ++d) {
        char bus[32] = {};
        CK(cudaDeviceGetPCIBusId(bus, sizeof bus, d));
        for (char* c = bus; *c; ++c) *c = (char) std::tolower(*c);
        std::printf("  GPU %d: %s, IOMMU domain %s\n", d, bus,
                    first_line(std::string("/sys/bus/pci/devices/") + bus + "/iommu_group/type").c_str());
    }
    std::printf("GB/s, two runs each; copies of records (%zu bytes) and 4 MiB pieces from random 4 KiB-aligned offsets\n", kRecord);
    for (const char* name : {"4k", "thp", "alloc"}) {
        Arm m = make(name, n);
        if (m.raw)
            std::printf("%-5s  huge pages %3.0f%% of it; faulted in %.2f s, registered %.2f s\n", name, 100.0 * m.thp, m.fault_s, m.reg_s);
        else
            std::printf("%-5s  cudaHostAlloc %.2f s\n", name, m.reg_s);
        for (int g = 0; g < ndev; ++g) {
            const double a = h2d(m, g, kRecord, 300, 11), b = h2d(m, g, kRecord, 300, 12);
            const double c = h2d(m, g, kPiece, 900, 21), d = h2d(m, g, kPiece, 900, 22);
            std::printf("       GPU %d: records %5.2f %5.2f, 4 MiB pieces %5.2f %5.2f\n", g, a, b, c, d);
        }
        if (ndev == 2) {
            double sum[2] = {};
            for (int run = 0; run < 2; ++run) {
                double a = 0, b = 0;
                std::thread t0([&] { a = h2d(m, 0, kRecord, 300, 31 + run); });
                std::thread t1([&] { b = h2d(m, 1, kRecord, 300, 41 + run); });
                t0.join();
                t1.join();
                sum[run] = a + b;
            }
            std::printf("       both GPUs at once: records %5.2f %5.2f (the sum)\n", sum[0], sum[1]);
        }
        std::printf("       CPU, 20 threads streaming records: %5.1f %5.1f\n", cpu_read(m, 100, 51), cpu_read(m, 100, 61));
        std::fflush(stdout);
        drop(m);
    }
    return 0;
}
