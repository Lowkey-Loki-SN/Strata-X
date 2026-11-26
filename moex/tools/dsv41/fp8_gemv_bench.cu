// moex/tools/dsv41/fp8_gemv_bench.cu - DSV41_PLAN §70: the decode's dense products from FP8 weights (gemv_fp8_w) against the
// same weights as BF16 (gemv_bf16_w), on DeepSeek-V4.1's projection shapes: each matrix's values are FP8 values times a
// power-of-two scale for every 32 of a row (as llama.cpp's set holds the release's FP8 projections), held both ways; the time
// a call (the median of 50 runs of 100 calls, T tokens) and every output compared bit for bit.
//
//   nvcc -O3 -std=c++17 -arch=sm_86 -Iinclude moex/tools/dsv41/fp8_gemv_bench.cu build-dsv41-dev/libstrata_x_kernels.a \
//        -o build-dsv41-dev/fp8_gemv_bench
//   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 build-dsv41-dev/fp8_gemv_bench
#include "strata/x/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                 \
    do {                                                                                      \
        const cudaError_t e_ = (x);                                                           \
        if (e_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "fp8_gemv_bench: %s: %s\n", #x, cudaGetErrorString(e_));     \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

static float e4m3(uint32_t q) {
    const uint32_t e = (q >> 3) & 15u, m = q & 7u;
    float v;
    if (e) {
        const uint32_t b = ((e + 120u) << 23) | (m << 20);
        std::memcpy(&v, &b, 4);
    } else {
        v = (float) m * 0.001953125f;
    }
    return (q & 0x80u) ? -v : v;
}

int main() {
    struct Shape { int n_in, n_out; const char* what; };
    // moex (DSV41_PLAN §75): DeepSeek-V4.1-Flash's own shapes (n_embd 5,120), read from the set's blk.5 tensors - the first
    // table's were 4,096-wide, which gave the blocks a different shape from the model's
    const Shape shapes[] = {{5120, 1280, "attn_q_a"}, {1280, 32768, "attn_q_b"}, {5120, 512, "attn_kv"}, {5120, 2304, "shexp gate/up"},
                            {2304, 5120, "shexp down"}, {8192, 5120, "attn_output_b"}, {4096, 1024, "attn_output_a (a group)"},
                            {6144, 25600, "engram_kv"}};
    std::mt19937 rng(7);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    std::printf("%-24s %6s %6s %2s | %9s %9s %6s | %s\n", "shape", "n_in", "n_out", "T", "BF16 us", "FP8 us", "speed", "outputs");
    // TS=1,4,8 (the widths to run; unset: every one) and ONLY=text (the shapes whose name holds it): a short run
    const char* ts_env = std::getenv("TS");
    const char* only = std::getenv("ONLY");
    auto want_t = [&](int T) {
        if (!ts_env || !*ts_env) return true;
        for (const char* c = ts_env; *c; ++c)
            if (*c - '0' == T) return true;
        return false;
    };
    for (const Shape& sh : shapes) {
        if (only && *only && !std::strstr(sh.what, only)) continue;
        const size_t nv = (size_t) sh.n_in * sh.n_out;
        std::vector<uint8_t> q(nv), sc(nv / 32);
        std::vector<uint16_t> bf(nv);
        for (size_t b = 0; b < nv / 32; ++b) {
            // moex (§75): the release's own shape - the scales near 2^-12 (its dense blocks) and an exponent-0 code in 0.3% of
            // the blocks (measured on the set's blk.5: 0.2-0.3% of the blocks, 0.008% of the codes).  Uniform random codes put
            // one in 87% of the blocks, which is the slow branch of every decode
            sc[b] = (uint8_t) (127 - 22 + (int) (rng() % 12));
            const float scf = std::ldexp(1.0f, (int) sc[b] - 127);
            const bool sub = rng() % 1000 < 3;
            bool any0 = false;
            for (int i = 0; i < 32; ++i) {
                uint8_t c = (uint8_t) (rng() & 0xFFu);
                if ((c & 0x78u) == 0) c |= 0x40u;             // exponent >= 1 ...
                if (sub && i == (int) (b % 32)) c &= 0x87u;   // ... but one code of a subnormal block
                if ((c & 0x7Fu) == 0x7Fu) c &= 0xFEu;   // (no NaN codes)
                q[b * 32 + i] = c;
                const float v = e4m3(c) * scf;
                uint32_t u;
                std::memcpy(&u, &v, 4);
                bf[b * 32 + i] = (uint16_t) (u >> 16);   // exact: 4 significant bits
                any0 = any0 || (c & 0x78u) == 0;
            }
            if (any0) sc[b] |= 0x80u;   // moex (§75): the loader's mark - this block holds an exponent-0 code (fp8_block)
        }
        for (int T : {1, 2, 3, 4, 5, 6, 7, 8}) {   // moex (§75): every verify window's width too (the prompt lookup's, DSpark's)
            if (!want_t(T)) continue;
            std::vector<float> x((size_t) T * sh.n_in);
            for (float& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.0f;
            uint8_t* dq;
            uint16_t* db;
            float *dx, *y1, *y2;
            CK(cudaMalloc(&dq, nv + nv / 32));
            CK(cudaMalloc(&db, nv * 2));
            CK(cudaMalloc(&dx, x.size() * 4));
            CK(cudaMalloc(&y1, (size_t) T * sh.n_out * 4));
            CK(cudaMalloc(&y2, (size_t) T * sh.n_out * 4));
            CK(cudaMemcpy(dq, q.data(), nv, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dq + nv, sc.data(), nv / 32, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(db, bf.data(), nv * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            auto timeit = [&](bool f8) {
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a));
                CK(cudaEventCreate(&b));
                std::vector<float> ms;
                for (int rep = 0; rep < 50; ++rep) {
                    CK(cudaEventRecord(a, s));
                    for (int k = 0; k < 100; ++k) {
                        if (f8) strata::x::gemv_fp8_w(dq, dq + nv, dx, y2, sh.n_in, sh.n_out, T, s);
                        else strata::x::gemv_bf16_w(db, dx, y1, sh.n_in, sh.n_out, T, s);
                    }
                    CK(cudaEventRecord(b, s));
                    CK(cudaEventSynchronize(b));
                    float t;
                    CK(cudaEventElapsedTime(&t, a, b));
                    ms.push_back(t * 10.0f);   // us a call
                }
                std::sort(ms.begin(), ms.end());
                return ms[ms.size() / 2];
            };
            const float tb = timeit(false), tf = timeit(true);
            std::vector<float> h1((size_t) T * sh.n_out), h2(h1.size());
            CK(cudaMemcpy(h1.data(), y1, h1.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h2.data(), y2, h2.size() * 4, cudaMemcpyDeviceToHost));
            const bool same = std::memcmp(h1.data(), h2.data(), h1.size() * 4) == 0;
            std::printf("%-24s %6d %6d %2d | %9.1f %9.1f %5.2fx | %s\n", sh.what, sh.n_in, sh.n_out, T, tb, tf, tb / tf,
                        same ? "bit for bit" : "DIFFER");
            CK(cudaFree(dq));
            CK(cudaFree(db));
            CK(cudaFree(dx));
            CK(cudaFree(y1));
            CK(cudaFree(y2));
        }
    }
    return 0;
}
