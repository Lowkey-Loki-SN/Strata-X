// src/kernels/cpu/native_expert_cpu_bench.cpp - moex: the CPU pool's per-expert work by weight format, on real GGUF
// rows.  One routed expert = its gate/up rows with the SwiGLU (native_gu_rows), the h quantization and its down rows
// (native_down_rows): the blob form the pool runs for arena experts, so the same kernel choice as the engine.  The
// threads cycle through enough distinct experts that the weights stream from DRAM, like a decode step's misses.
//
//     build/native_expert_cpu_bench <shard.gguf> <layer> [experts 48] [tokens 1] [threads 1] [seconds 3] [clamp 0]
//
// Prints the weights' GB/s (the experts' bytes / wall time) next to a plain read of the same blobs by the same
// threads - the memory ceiling for that thread count.  `clamp` is the SwiGLU limit (DeepSeek-V4 / GLM: 10).  A layer
// whose expert tensors end past the end of the file (a download in progress) is refused, never touched.
// moex (DSV41_PLAN §29): STRATA_BENCH_RECORDS=<gate/up type>:<down type>:<n_embd>:<n_ff> (ggml type numbers) reads the
// experts from a DeepSeek-V4.1 pack's record file instead (the first argument; the layer is ignored): its records are
// the blob form, one per 4 KiB-rounded stride
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"

#include <immintrin.h>
#include <sys/mman.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

__attribute__((target("avx2"))) uint64_t read_bytes(const uint8_t* p, size_t n) {
    __m256i acc = _mm256_setzero_si256();
    for (size_t i = 0; i + 128 <= n; i += 128) {
        acc = _mm256_xor_si256(acc, _mm256_load_si256((const __m256i*) (p + i)));
        acc = _mm256_xor_si256(acc, _mm256_load_si256((const __m256i*) (p + i + 32)));
        acc = _mm256_xor_si256(acc, _mm256_load_si256((const __m256i*) (p + i + 64)));
        acc = _mm256_xor_si256(acc, _mm256_load_si256((const __m256i*) (p + i + 96)));
    }
    return (uint64_t) _mm256_extract_epi64(acc, 0) ^ (uint64_t) _mm256_extract_epi64(acc, 3);
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) {
        std::fprintf(stderr, "usage: native_expert_cpu_bench <shard.gguf> <layer> [experts] [tokens] [threads] [seconds] [clamp]\n");
        return 2;
    }
    const char* recs = std::getenv("STRATA_BENCH_RECORDS");
    if (recs && *recs) {   // moex: a record file's experts as they are
        int gt = -1, dt = -1;
        long long ne = 0, nf = 0;
        if (std::sscanf(recs, "%d:%d:%lld:%lld", &gt, &dt, &ne, &nf) != 4) { std::printf("STRATA_BENCH_RECORDS=gu:d:n_embd:n_ff\n"); return 2; }
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(gt, dt, ne, nf, f, err)) { std::printf("%s\n", err.c_str()); return 1; }
        const int NE = argc > 3 ? std::atoi(argv[3]) : 48, nt = argc > 4 ? std::atoi(argv[4]) : 1, nthr = argc > 5 ? std::atoi(argv[5]) : 1;
        const double secs = argc > 6 ? std::atof(argv[6]) : 3.0;
        f.swiglu_limit = argc > 7 ? (float) std::atof(argv[7]) : 0.0f;
        FILE* rf = std::fopen(argv[1], "rb");
        if (!rf) { std::printf("cannot open %s\n", argv[1]); return 1; }
        const size_t rstride = (f.bytes + 4095) / 4096 * 4096, stride = rstride, total = (size_t) NE * stride;
        auto* blobs = (uint8_t*) std::aligned_alloc(4096, total);
        madvise(blobs, total, MADV_NOHUGEPAGE);
        for (int e = 0; e < NE; ++e) {   // experts spread over the file, as a decode's misses
            const long long E = (e * 37 + 5) % 384;
            std::fseek(rf, (long) (E * (long long) rstride), SEEK_SET);
            if (std::fread(blobs + (size_t) e * stride, 1, f.bytes, rf) != f.bytes) { std::printf("short read\n"); return 1; }
            std::memset(blobs + (size_t) e * stride + f.bytes, 0, stride - f.bytes);
        }
        std::fclose(rf);
        std::printf("records %s: gate/up %s, down %s, %lld x %lld, expert %.2f MiB, %d experts (%.0f MiB), %d token(s), %d thread(s)\n",
                    argv[1], ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), ne, nf, f.bytes / 1048576.0,
                    NE, total / 1048576.0, nt, nthr);
        std::mt19937 rng(17);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<std::vector<uint8_t>> act(nt, std::vector<uint8_t>(f.act_bytes + 64));
        for (int k = 0; k < nt; ++k) {
            std::vector<float> x((size_t) ne);
            for (auto& v : x) v = nd(rng);
            cpu::native_quant_act(f, x.data(), act[k].data());
        }
        auto worker = [&](int tid, double t_end, std::atomic<long long>* done, uint64_t* sink) {
            std::vector<std::vector<float>> ff(nt, std::vector<float>((size_t) nf)), out(nt, std::vector<float>((size_t) ne));
            std::vector<std::vector<uint8_t>> hq(nt, std::vector<uint8_t>(f.h_bytes + 64));
            std::vector<const void*> a(nt), h(nt);
            std::vector<float*> pf(nt), po(nt);
            for (int k = 0; k < nt; ++k) { a[k] = act[k].data(); h[k] = hq[k].data(); pf[k] = ff[k].data(); po[k] = out[k].data(); }
            long long n = 0;
            uint64_t sm = 0;
            int e = tid;
            while (true) {
                const uint8_t* b = blobs + (size_t) e * stride;
                cpu::native_gu_rows(f, b, a.data(), nt, pf.data(), 0, (int) nf);
                for (int k = 0; k < nt; ++k) cpu::native_quant_h(f, ff[k].data(), hq[k].data());
                cpu::native_down_rows(f, b, h.data(), nt, po.data(), 0, (int) ne);
                sm += out[0][(size_t) (n % ne)] != 0.0f;
                ++n;
                e += nthr;
                if (e >= NE) e = tid;
                if ((n & 3) == 0 && now_s() >= t_end) break;
            }
            done->fetch_add(n);
            *sink += sm;
        };
        auto reader = [&](int tid, double t_end, std::atomic<long long>* done, uint64_t* sink) {
            long long n = 0;
            uint64_t sm = 0;
            int e = tid;
            while (true) {
                sm ^= read_bytes(blobs + (size_t) e * stride, f.bytes / 128 * 128);
                ++n;
                e += nthr;
                if (e >= NE) e = tid;
                if ((n & 15) == 0 && now_s() >= t_end) break;
            }
            done->fetch_add(n);
            *sink += sm;
        };
        auto run = [&](auto& fn, double dur) {
            std::atomic<long long> done{0};
            std::vector<uint64_t> sink((size_t) nthr * 8, 0);
            std::vector<std::thread> th;
            const double t0 = now_s(), t_end = t0 + dur;
            for (int i = 0; i < nthr; ++i) th.emplace_back(fn, i, t_end, &done, &sink[(size_t) i * 8]);
            for (auto& x : th) x.join();
            const double el = now_s() - t0;
            return std::make_pair((double) done.load() * (double) f.bytes / el, (double) done.load() / el);
        };
        run(worker, 0.3);
        const auto k = run(worker, secs);
        const auto rd = run(reader, secs / 2);
        std::printf("  kernel %6.2f GB/s  (%7.1f us per expert per thread, %.0f experts/s)   plain read %6.2f GB/s   kernel/read %.2f\n",
                    k.first / 1e9, 1e6 * nthr / k.second, k.second, rd.first / 1e9, k.first / rd.first);
        std::free(blobs);
        return 0;
    }
    strata::GgufFile g(argv[1]);
    const int layer = std::atoi(argv[2]);
    const int NE = argc > 3 ? std::atoi(argv[3]) : 48;
    const int nt = argc > 4 ? std::atoi(argv[4]) : 1;
    const int nthr = argc > 5 ? std::atoi(argv[5]) : 1;
    const double secs = argc > 6 ? std::atof(argv[6]) : 3.0;
    const float clamp = argc > 7 ? (float) std::atof(argv[7]) : 0.0f;
    if (NE < 1 || nt < 1 || nt > 8 || nthr < 1 || NE < nthr) {
        std::fprintf(stderr, "experts >= threads >= 1, tokens 1..8\n");
        return 2;
    }
    const strata::TensorInfo* t[3] = {};
    const char* roles[3] = {"gate", "up", "down"};
    for (int r = 0; r < 3; ++r) t[r] = g.find("blk." + std::to_string(layer) + ".ffn_" + roles[r] + "_exps.weight");
    if (!t[0] || !t[1] || !t[2]) {
        std::printf("layer %d: no separate gate/up/down expert tensors in %s\n", layer, argv[1]);
        return 1;
    }
    if (t[0]->type != t[1]->type || t[0]->shape.size() != 3 || t[2]->shape.size() != 3) {
        std::printf("layer %d: gate and up differ in type, or not 3-D\n", layer);
        return 1;
    }
    const int64_t n_embd = (int64_t) t[0]->shape[0], n_ff = (int64_t) t[0]->shape[1], n_exp = (int64_t) t[0]->shape[2];
    if ((int64_t) t[2]->shape[0] != n_ff || (int64_t) t[2]->shape[1] != n_embd || (int64_t) t[2]->shape[2] != n_exp) {
        std::printf("layer %d: down is not [n_ff, n_embd, n_expert]\n", layer);
        return 1;
    }
    // STRATA_BENCH_REQUANT=<gate/up type>:<down type> (ggml names, e.g. q3_K:mxfp4): each sampled expert re-quantized from
    // the file's rows (dequantized through ggml) into those formats - the kernels' speed for formats the file does not hold
    int gu_new = (int) t[0]->type, d_new = (int) t[2]->type;
    const char* rq = std::getenv("STRATA_BENCH_REQUANT");
    auto type_of = [](const std::string& nm) -> int {
        for (int i = 0; i < GGML_TYPE_COUNT; ++i)
            if (ggml_get_type_traits((ggml_type) i)->type_name && nm == ggml_get_type_traits((ggml_type) i)->type_name) return i;
        return -1;
    };
    if (rq && *rq) {
        const std::string sp(rq);
        const size_t c = sp.find(':');
        gu_new = type_of(sp.substr(0, c));
        d_new = type_of(c == std::string::npos ? sp : sp.substr(c + 1));
        if (gu_new < 0 || d_new < 0) { std::printf("STRATA_BENCH_REQUANT: unknown type in %s\n", rq); return 2; }
        ggml_quantize_init((ggml_type) gu_new);
        ggml_quantize_init((ggml_type) d_new);
    }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu_new, d_new, n_embd, n_ff, f, err)) {
        std::printf("layer %d: %s\n", layer, err.c_str());
        return 1;
    }
    f.swiglu_limit = clamp;
    const size_t gsz = f.up_off, dsz = f.bytes - f.down_off;
    const size_t gsz_src = ggml_row_size((ggml_type) t[0]->type, n_embd) * (size_t) n_ff, dsz_src = ggml_row_size((ggml_type) t[2]->type, n_ff) * (size_t) n_embd;
    for (int r = 0; r < 3; ++r) {
        const uint64_t end = g.data_start() + t[r]->offset + (uint64_t) n_exp * (r == 2 ? dsz_src : gsz_src);
        if (end > g.file_size()) {
            std::printf("layer %d: %s ends past the end of the file (%llu > %llu) - not downloaded yet?\n", layer,
                        t[r]->name.c_str(), (unsigned long long) end, (unsigned long long) g.file_size());
            return 1;
        }
    }
    // the blobs on 4 KB pages, as the engine's pinned arena by default (--arena-thp is opt-in)
    const size_t stride = (f.bytes + 4095) / 4096 * 4096, total = (size_t) NE * stride;
    auto* blobs = (uint8_t*) std::aligned_alloc(4096, total);
    if (blobs == nullptr) { std::printf("cannot allocate %zu bytes\n", total); return 1; }
    madvise(blobs, total, MADV_NOHUGEPAGE);
    auto setup_one = [&](int e) {
        const size_t E = (size_t) ((e * 37 + 5) % n_exp);
        uint8_t* b = blobs + (size_t) e * stride;
        if (rq && *rq) {   // dequantize the file's rows, quantize into the requested formats (imatrix: ones)
            auto requant = [&](const strata::TensorInfo& ti, size_t src_bytes, int64_t rows, int64_t cols, int dst_type, uint8_t* dst) {
                std::vector<float> fbuf((size_t) rows * (size_t) cols), ones((size_t) cols, 1.0f);
                ggml_get_type_traits((ggml_type) ti.type)->to_float(g.tensor_data(ti) + E * src_bytes, fbuf.data(), rows * cols);
                ggml_quantize_chunk((ggml_type) dst_type, fbuf.data(), dst, 0, rows, cols, ones.data());
            };
            requant(*t[0], gsz_src, n_ff, n_embd, gu_new, b);
            requant(*t[1], gsz_src, n_ff, n_embd, gu_new, b + f.up_off);
            requant(*t[2], dsz_src, n_embd, n_ff, d_new, b + f.down_off);
        } else {
            std::memcpy(b, g.tensor_data(*t[0]) + E * gsz, gsz);
            std::memcpy(b + f.up_off, g.tensor_data(*t[1]) + E * gsz, gsz);
            std::memcpy(b + f.down_off, g.tensor_data(*t[2]) + E * dsz, dsz);
        }
        std::memset(b + f.bytes, 0, stride - f.bytes);
    };
    {   // the setup in parallel (re-quantizing IQ formats is slow)
        std::vector<std::thread> st;
        std::atomic<int> next{0};
        const int nst = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
        for (int i = 0; i < nst; ++i)
            st.emplace_back([&] {
                for (int e; (e = next.fetch_add(1)) < NE;) setup_one(e);
            });
        for (auto& th : st) th.join();
    }
    std::printf("layer %d: gate/up %s, down %s, %lld x %lld, expert %.2f MiB, %d experts cycled (%.0f MiB), %d token(s), "
                "%d thread(s), clamp %g, multi-token from %d\n",
                layer, ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), (long long) n_embd,
                (long long) n_ff, f.bytes / 1048576.0, NE, total / 1048576.0, nt, nthr, (double) clamp,
                cpu::native_gu_mt_min(f.gu_type));

    // the activations: one quantized x per token (a layer's, shared by all its experts), random as in the parity tests
    std::mt19937 rng(17 + layer);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> act(nt, std::vector<uint8_t>(f.act_bytes + 64));
    for (int k = 0; k < nt; ++k) {
        std::vector<float> x((size_t) n_embd);
        for (auto& v : x) v = nd(rng);
        cpu::native_quant_act(f, x.data(), act[k].data());
    }

    auto worker = [&](int tid, double t_end, std::atomic<long long>* done, uint64_t* sink) {
        std::vector<std::vector<float>> ff(nt, std::vector<float>((size_t) n_ff)), out(nt, std::vector<float>((size_t) n_embd));
        std::vector<std::vector<uint8_t>> hq(nt, std::vector<uint8_t>(f.h_bytes + 64));
        std::vector<const void*> a(nt), h(nt);
        std::vector<float*> pf(nt), po(nt);
        for (int k = 0; k < nt; ++k) { a[k] = act[k].data(); h[k] = hq[k].data(); pf[k] = ff[k].data(); po[k] = out[k].data(); }
        long long n = 0;
        uint64_t s = 0;
        int e = tid;
        while (true) {
            const uint8_t* b = blobs + (size_t) e * stride;
            cpu::native_gu_rows(f, b, a.data(), nt, pf.data(), 0, (int) n_ff);
            for (int k = 0; k < nt; ++k) cpu::native_quant_h(f, ff[k].data(), hq[k].data());
            cpu::native_down_rows(f, b, h.data(), nt, po.data(), 0, (int) n_embd);
            s += out[0][(size_t) (n % n_embd)] != 0.0f;
            ++n;
            e += nthr;
            if (e >= NE) e = tid;
            if ((n & 3) == 0 && now_s() >= t_end) break;
        }
        done->fetch_add(n);
        *sink += s;
    };
    auto reader = [&](int tid, double t_end, std::atomic<long long>* done, uint64_t* sink) {
        long long n = 0;
        uint64_t s = 0;
        int e = tid;
        while (true) {
            s ^= read_bytes(blobs + (size_t) e * stride, f.bytes / 128 * 128);
            ++n;
            e += nthr;
            if (e >= NE) e = tid;
            if ((n & 15) == 0 && now_s() >= t_end) break;
        }
        done->fetch_add(n);
        *sink += s;
    };
    auto run = [&](auto& fn, double dur) {
        std::atomic<long long> done{0};
        std::vector<uint64_t> sink((size_t) nthr * 8, 0);
        std::vector<std::thread> th;
        const double t0 = now_s(), t_end = t0 + dur;
        for (int i = 0; i < nthr; ++i) th.emplace_back(fn, i, t_end, &done, &sink[(size_t) i * 8]);
        for (auto& x : th) x.join();
        const double el = now_s() - t0;
        return std::make_pair((double) done.load() * (double) f.bytes / el, (double) done.load() / el);
    };
    run(worker, 0.3);   // warm-up: page faults, clocks
    const auto k = run(worker, secs);
    const auto rd = run(reader, secs / 2);
    std::printf("  kernel %6.2f GB/s  (%7.1f us per expert per thread, %.0f experts/s)   plain read %6.2f GB/s   kernel/read %.2f\n",
                k.first / 1e9, 1e6 * nthr / k.second, k.second, rd.first / 1e9, k.first / rd.first);
    std::free(blobs);
    return 0;
}
