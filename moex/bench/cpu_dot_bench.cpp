// moex/bench/cpu_dot_bench.cpp - what one core and all cores can do with ggml-cpu's quantized dot, for a MoE expert's
// rows: the compute limit (rows in L2, re-read) against the streaming rate (rows from DRAM, a fresh blob each pass).
//
//     cpu_dot_bench <type> [threads]        type: q2_k q3_k q4_k q5_k q6_k q8_0 iq3_xxs iq3_s iq2_s mxfp4
//
// Built by moex/bench/build_cpu_dot.sh against the oracle's CPU-only ggml.  One "expert" = 3 matrices of 2048 rows x 4096
// (gate, up, down is 4096 rows x 2048 but the same bytes): rows of n_in values, dotted with one quantized activation.
#include "ggml-cpu.h"
#include "ggml.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    const std::string name = argc > 1 ? argv[1] : "q2_k";
    const int nthreads = argc > 2 ? atoi(argv[2]) : 1;
    ggml_cpu_init();
    ggml_type t = GGML_TYPE_Q2_K;
    if (name == "q3_k") t = GGML_TYPE_Q3_K;
    else if (name == "q4_k") t = GGML_TYPE_Q4_K;
    else if (name == "q5_k") t = GGML_TYPE_Q5_K;
    else if (name == "q6_k") t = GGML_TYPE_Q6_K;
    else if (name == "q8_0") t = GGML_TYPE_Q8_0;
    else if (name == "iq3_xxs") t = GGML_TYPE_IQ3_XXS;
    else if (name == "iq3_s") t = GGML_TYPE_IQ3_S;
    else if (name == "iq2_s") t = GGML_TYPE_IQ2_S;
    else if (name == "mxfp4") t = GGML_TYPE_MXFP4;
    const ggml_type_traits_cpu* tt = ggml_get_type_traits_cpu(t);
    const int n_in = 4096;
    const size_t rb = ggml_row_size(t, n_in);
    // the activation, quantized as ggml does for this weight type
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n_in);
    for (float& v : x) v = nd(rng);
    std::vector<uint8_t> xq(ggml_row_size(tt->vec_dot_type, n_in));
    ggml_get_type_traits_cpu(tt->vec_dot_type)->from_float(x.data(), xq.data(), n_in);
    // rows: random bytes with sane scales are not needed for timing (the arithmetic is data independent), but NaN
    // scales slow some paths: use a real quantization of random weights
    const int rows_big = 2048 * 3;                       // one expert's worth
    const size_t n_experts_stream = 24;                  // more than the L2 holds, so each pass streams from DRAM
    std::vector<float> wf((size_t) n_in);
    std::vector<uint8_t> w((size_t) rows_big * rb * n_experts_stream);
    {
        std::vector<float> imat((size_t) n_in, 1.0f);
        if (ggml_quantize_requires_imatrix(t)) ggml_quantize_init(t);
        for (size_t r = 0; r < (size_t) rows_big * 4; ++r) {
            for (float& v : wf) v = nd(rng) * 0.02f;
            ggml_quantize_chunk(t, wf.data(), w.data() + r * rb, 0, 1, n_in, ggml_quantize_requires_imatrix(t) ? imat.data() : nullptr);
        }
        for (size_t r = (size_t) rows_big * 4; r < (size_t) rows_big * n_experts_stream; ++r)
            memcpy(w.data() + r * rb, w.data() + (r % ((size_t) rows_big * 4)) * rb, rb);
    }
    std::vector<float> sink(nthreads * 16, 0.f);
    auto run = [&](bool stream, double seconds) {
        std::atomic<bool> stop{false};
        std::atomic<long long> rows{0};
        std::vector<std::thread> th;
        for (int i = 0; i < nthreads; ++i)
            th.emplace_back([&, i] {
                long long local = 0;
                size_t e = (size_t) i * 3 % n_experts_stream;
                float acc = 0, y;
                while (!stop.load(std::memory_order_relaxed)) {
                    const uint8_t* base = w.data() + (stream ? e : 0) * (size_t) rows_big * rb;
                    // a slice of this thread's own: rows [i * share, ...) of the expert, like the pool's row split
                    const int share = rows_big / std::max(1, nthreads);
                    const int r0 = stream ? i * share : 0, r1 = stream ? r0 + share : 256;
                    for (int r = r0; r < r1; ++r) {
                        tt->vec_dot(n_in, &y, 0, base + (size_t) r * rb, 0, xq.data(), 0, 1);
                        acc += y;
                    }
                    local += r1 - r0;
                    e = (e + nthreads) % n_experts_stream;
                }
                rows += local;
                sink[i * 16] = acc;
            });
        std::this_thread::sleep_for(std::chrono::milliseconds(300));   // warm up
        const long long r0 = rows.load();
        const double t0 = now();
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        stop = true;
        const double t1 = now();
        for (auto& x2 : th) x2.join();
        (void) r0;
        return (double) rows.load() * rb / (t1 - t0 + 0.3) / 1e9;   // GB/s of weight bytes (warm-up included in the count)
    };
    const double cached = run(false, 1.5), streamed = run(true, 2.5);
    printf("%-8s %2d thread%s: rows in cache %.2f GB/s (%.2f per core), streaming from DRAM %.2f GB/s (%.2f per core)\n",
           name.c_str(), nthreads, nthreads == 1 ? " " : "s", cached, cached / nthreads, streamed, streamed / nthreads);
    return 0;
}
