// moex/tools/ernie55/fp8_warp_bench.cu - ERNIE55_PLAN P6: the decode's FP8 projections (gemv_fp8_w at T 1) on ERNIE-5.5's own
// shapes, the warp kernel (default) against the row kernels (STRATA_DSV41_FP8_ROWS=4 / 8 picks them) and BF16 (gemv_bf16_w,
// the order both reproduce), every output compared bit for bit.  Each call reads a DIFFERENT copy of the weights (a ring of
// copies larger than the L2), as a decode token does: the weights come from DRAM.
//
//   nvcc -O3 -std=c++17 -arch=sm_86 -Iinclude moex/tools/ernie55/fp8_warp_bench.cu build-dsv41-dev/libstrata_x_kernels.a \
//        -o build-dsv41-dev/fp8_warp_bench
//   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 build-dsv41-dev/fp8_warp_bench
#include "strata/x/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                             \
    do {                                                                                  \
        const cudaError_t e_ = (x);                                                       \
        if (e_ != cudaSuccess) {                                                          \
            std::fprintf(stderr, "fp8_warp_bench: %s: %s\n", #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                 \
        }                                                                                 \
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

int main(int argc, char** argv) {
    struct Shape { int n_in, n_out; const char* what; };
    const Shape shapes[] = {{4096, 1024, "attn_q_a"},     {1024, 32768, "attn_q_b (HCA)"}, {1024, 8192, "attn_gate (HCA)"},
                            {4096, 512, "attn_kv (HCA)"}, {8192, 4096, "attn_output_b"},   {4096, 2048, "shexp gate/up, r_down"},
                            {2048, 4096, "shexp down, r_up"}, {1024, 16384, "attn_q_b (DSA)"}, {4096, 14336, "ffn gate/up (layer 0)"},
                            {14336, 4096, "ffn down (layer 0)"}, {4096, 576, "attn_kv_a_mqa (DSA)"}};
    const char* only = std::getenv("ONLY");
    std::mt19937 rng(11);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    bool all_same = true;
    {   // the clocks up first (~1.5 s of memsets over 256 MB): a short burst would run at the idle clocks
        void* wb = nullptr;
        CK(cudaMalloc(&wb, 256u << 20));
        const auto w0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 1.5) {
            for (int i = 0; i < 8; ++i) CK(cudaMemsetAsync(wb, i, 256u << 20, s));
            CK(cudaStreamSynchronize(s));
        }
        CK(cudaFree(wb));
    }
    std::printf("%-24s %6s %6s | %8s %8s %8s | %7s %7s | %s\n", "shape", "n_in", "n_out", "BF16 us", "rows us", "warp us",
                "GB/s", "speed", "outputs (warp vs BF16, rows vs BF16)");
    for (const Shape& sh : shapes) {
        if (only && *only && !std::strstr(sh.what, only)) continue;
        const size_t nv = (size_t) sh.n_in * sh.n_out;
        std::vector<uint8_t> q(nv), sc(nv / 32);
        std::vector<uint16_t> bf(nv);
        for (size_t b = 0; b < nv / 32; ++b) {   // fp8_gemv_bench's mix: scales near 2^-12, an exponent-0 code in 0.3% of blocks
            sc[b] = (uint8_t) (127 - 22 + (int) (rng() % 12));
            const float scf = std::ldexp(1.0f, (int) sc[b] - 127);
            const bool sub = rng() % 1000 < 3;
            bool any0 = false;
            for (int i = 0; i < 32; ++i) {
                uint8_t c = (uint8_t) (rng() & 0xFFu);
                if ((c & 0x78u) == 0) c |= 0x40u;
                if (sub && i == (int) (b % 32)) c &= 0x87u;
                if ((c & 0x7Fu) == 0x7Fu) c &= 0xFEu;
                q[b * 32 + i] = c;
                const float v = e4m3(c) * scf;
                uint32_t u;
                std::memcpy(&u, &v, 4);
                bf[b * 32 + i] = (uint16_t) (u >> 16);
                any0 = any0 || (c & 0x78u) == 0;
            }
            if (any0) sc[b] |= 0x80u;
        }
        std::vector<float> x(sh.n_in);
        for (float& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.0f;
        // a ring of copies past the L2 (5 MB): >= 64 MB of FP8 weights
        const size_t per = (nv + nv / 32 + 255) & ~(size_t) 255;
        const int copies = (int) std::max<size_t>(2, (64u << 20) / per + 1);
        uint8_t* dq;
        uint16_t* db;
        float *dx, *y0, *y1, *y2;
        CK(cudaMalloc(&dq, per * copies));
        CK(cudaMalloc(&db, nv * 2));
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMalloc(&y0, (size_t) sh.n_out * 4));
        CK(cudaMalloc(&y1, (size_t) sh.n_out * 4));
        CK(cudaMalloc(&y2, (size_t) sh.n_out * 4));
        for (int c = 0; c < copies; ++c) {
            CK(cudaMemcpy(dq + c * per, q.data(), nv, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dq + c * per + nv, sc.data(), nv / 32, cudaMemcpyHostToDevice));
        }
        CK(cudaMemcpy(db, bf.data(), nv * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        auto timeit = [&](int which, float* y) {   // 0 BF16, 1 FP8 rows (via env), 2 FP8 warp
            cudaEvent_t a, b;
            CK(cudaEventCreate(&a));
            CK(cudaEventCreate(&b));
            std::vector<float> ms;
            for (int rep = 0; rep < 30; ++rep) {
                CK(cudaEventRecord(a, s));
                for (int k = 0; k < 64; ++k) {
                    const uint8_t* w = dq + (size_t) (k % copies) * per;
                    if (which == 0) strata::x::gemv_bf16_w(db, dx, y, sh.n_in, sh.n_out, 1, s);
                    else strata::x::gemv_fp8_w(w, w + nv, dx, y, sh.n_in, sh.n_out, 1, s);
                }
                CK(cudaEventRecord(b, s));
                CK(cudaEventSynchronize(b));
                float t;
                CK(cudaEventElapsedTime(&t, a, b));
                ms.push_back(t * 1000.0f / 64.0f);
            }
            std::sort(ms.begin(), ms.end());
            return ms[ms.size() / 2];
        };
        const char* mode = argc > 1 ? argv[1] : "warp";   // "rows": this process runs the row kernels (the env set by the caller)
        const float tb = timeit(0, y0);
        const float tf = timeit(2, y2);
        std::vector<float> h0(sh.n_out), h2(sh.n_out);
        CK(cudaMemcpy(h0.data(), y0, h0.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h2.data(), y2, h2.size() * 4, cudaMemcpyDeviceToHost));
        const bool same = std::memcmp(h0.data(), h2.data(), h0.size() * 4) == 0;
        all_same = all_same && same;
        const double gbs = (double) (nv + nv / 32) / (tf * 1e-6) / 1e9;
        std::printf("%-24s %6d %6d | %8.1f %8s %8.1f | %7.0f %7s | %s (%s)\n", sh.what, sh.n_in, sh.n_out, tb, "-", tf, gbs, "-",
                    same ? "bit for bit" : "DIFFER", mode);
        CK(cudaFree(dq));
        CK(cudaFree(db));
        CK(cudaFree(dx));
        CK(cudaFree(y0));
        CK(cudaFree(y1));
        CK(cudaFree(y2));
    }
    // the grouped products (gemv_fp8_heads, one launch for H groups) against H BF16 calls
    struct HShape { int n_in, n_out, H; const char* what; };
    const HShape hs[] = {{4096, 1024, 8, "attn_output_a (HCA, 8 groups)"}, {192, 512, 64, "attn_k_b (DSA, 64 heads)"},
                         {512, 256, 64, "attn_v_b (DSA, 64 heads)"}};
    for (const HShape& sh : hs) {
        if (only && *only && !std::strstr(sh.what, only)) continue;
        const size_t nv = (size_t) sh.n_in * sh.n_out * sh.H;
        std::vector<uint8_t> q(nv), sc(nv / 32);
        std::vector<uint16_t> bf(nv);
        for (size_t b = 0; b < nv / 32; ++b) {
            sc[b] = (uint8_t) (127 - 22 + (int) (rng() % 12));
            const float scf = std::ldexp(1.0f, (int) sc[b] - 127);
            const bool sub = rng() % 1000 < 3;
            bool any0 = false;
            for (int i = 0; i < 32; ++i) {
                uint8_t c = (uint8_t) (rng() & 0xFFu);
                if ((c & 0x78u) == 0) c |= 0x40u;
                if (sub && i == (int) (b % 32)) c &= 0x87u;
                if ((c & 0x7Fu) == 0x7Fu) c &= 0xFEu;
                q[b * 32 + i] = c;
                const float v = e4m3(c) * scf;
                uint32_t u;
                std::memcpy(&u, &v, 4);
                bf[b * 32 + i] = (uint16_t) (u >> 16);
                any0 = any0 || (c & 0x78u) == 0;
            }
            if (any0) sc[b] |= 0x80u;
        }
        std::vector<float> x((size_t) sh.n_in * sh.H);
        for (float& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.0f;
        uint8_t* dq;
        uint16_t* db;
        float *dx, *y0, *y2;
        CK(cudaMalloc(&dq, nv + nv / 32));
        CK(cudaMalloc(&db, nv * 2));
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMalloc(&y0, (size_t) sh.n_out * sh.H * 4));
        CK(cudaMalloc(&y2, (size_t) sh.n_out * sh.H * 4));
        CK(cudaMemcpy(dq, q.data(), nv, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dq + nv, sc.data(), nv / 32, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(db, bf.data(), nv * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        for (int h = 0; h < sh.H; ++h)
            strata::x::gemv_bf16_w(db + (size_t) h * sh.n_in * sh.n_out, dx + (size_t) h * sh.n_in, y0 + (size_t) h * sh.n_out,
                                     sh.n_in, sh.n_out, 1, s);
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        std::vector<float> ms;
        for (int rep = 0; rep < 30; ++rep) {
            CK(cudaEventRecord(a, s));
            for (int k = 0; k < 64; ++k) strata::x::gemv_fp8_heads(dq, dq + nv, dx, y2, sh.n_in, sh.n_out, 1, sh.H, s);
            CK(cudaEventRecord(b, s));
            CK(cudaEventSynchronize(b));
            float t;
            CK(cudaEventElapsedTime(&t, a, b));
            ms.push_back(t * 1000.0f / 64.0f);
        }
        std::sort(ms.begin(), ms.end());
        std::vector<float> h0((size_t) sh.n_out * sh.H), h2(h0.size());
        CK(cudaMemcpy(h0.data(), y0, h0.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h2.data(), y2, h2.size() * 4, cudaMemcpyDeviceToHost));
        const bool same = std::memcmp(h0.data(), h2.data(), h0.size() * 4) == 0;
        all_same = all_same && same;
        std::printf("%-30s %5d %5d x%2d | heads %8.1f us (L2-warm) | %s\n", sh.what, sh.n_in, sh.n_out, sh.H, ms[ms.size() / 2],
                    same ? "bit for bit" : "DIFFER");
        CK(cudaFree(dq));
        CK(cudaFree(db));
        CK(cudaFree(dx));
        CK(cudaFree(y0));
        CK(cudaFree(y2));
    }
    // gemv_pair_w (two products of one input in one launch) against the two one-column calls, bit for bit: ERNIE's pairs -
    // q_a + the KV row (FP8), the compressor's two rows (BF16), the routers + routed_down (BF16 + FP8).  Run with
    // STRATA_DSV41_PRELOAD=1 (the BF16 preload kernel, the engine's default for ERNIE; without it the pair declines)
    {
        struct PShape { int n_in, na, nb; bool fa, fb; const char* what; };
        const PShape ps[] = {{4096, 1024, 512, true, true, "q_a + attn_kv (FP8 + FP8)"},
                             {4096, 512, 512, false, false, "compressor kv + gate (BF16 x 2)"},
                             {4096, 1024, 2048, false, true, "routers + routed_down (BF16 + FP8)"},
                             {1024, 32768, 8192, true, true, "q_b + attn_gate (FP8 + FP8)"},
                             {4096, 2048, 2048, true, true, "shared gate + up (FP8 + FP8)"}};
        for (const PShape& sh : ps) {
            if (only && *only && !std::strstr(sh.what, only)) continue;
            auto make = [&](int n_out, bool fp8, uint8_t*& dq, uint16_t*& db) {
                const size_t nv = (size_t) sh.n_in * n_out;
                std::vector<uint8_t> q(nv), sc(nv / 32);
                std::vector<uint16_t> bf(nv);
                for (size_t b = 0; b < nv / 32; ++b) {
                    sc[b] = (uint8_t) (127 - 22 + (int) (rng() % 12));
                    const float scf = std::ldexp(1.0f, (int) sc[b] - 127);
                    bool any0 = false;
                    for (int i = 0; i < 32; ++i) {
                        uint8_t c = (uint8_t) (rng() & 0xFFu);
                        if ((c & 0x78u) == 0) c |= 0x40u;
                        if (rng() % 1000 < 3) c &= 0x87u;
                        if ((c & 0x7Fu) == 0x7Fu) c &= 0xFEu;
                        q[b * 32 + i] = c;
                        const float v = e4m3(c) * scf;
                        uint32_t u;
                        std::memcpy(&u, &v, 4);
                        bf[b * 32 + i] = (uint16_t) (u >> 16);
                        any0 = any0 || (c & 0x78u) == 0;
                    }
                    if (any0) sc[b] |= 0x80u;
                }
                CK(cudaMalloc(&dq, nv + nv / 32));
                CK(cudaMalloc(&db, nv * 2));
                CK(cudaMemcpy(dq, q.data(), nv, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dq + nv, sc.data(), nv / 32, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(db, bf.data(), nv * 2, cudaMemcpyHostToDevice));
                (void) fp8;
            };
            uint8_t *qa, *qb;
            uint16_t *ba, *bb;
            make(sh.na, sh.fa, qa, ba);
            make(sh.nb, sh.fb, qb, bb);
            std::vector<float> x(sh.n_in);
            for (float& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.0f;
            float *dx, *ya, *yb, *ra, *rb;
            CK(cudaMalloc(&dx, x.size() * 4));
            CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            for (float** p : {&ya, &ra}) CK(cudaMalloc(p, (size_t) sh.na * 4));
            for (float** p : {&yb, &rb}) CK(cudaMalloc(p, (size_t) sh.nb * 4));
            const size_t nva = (size_t) sh.n_in * sh.na, nvb = (size_t) sh.n_in * sh.nb;
            auto one = [&](bool fp8, uint8_t* q, uint16_t* b, size_t nv, float* y, int n_out) {
                if (fp8) strata::x::gemv_fp8_w(q, q + nv, dx, y, sh.n_in, n_out, 1, s);
                else strata::x::gemv_bf16_w(b, dx, y, sh.n_in, n_out, 1, s);
            };
            one(sh.fa, qa, ba, nva, ra, sh.na);
            one(sh.fb, qb, bb, nvb, rb, sh.nb);
            const bool launched = strata::x::gemv_pair_w(sh.fa ? (const void*) qa : (const void*) ba, sh.fa ? qa + nva : nullptr, sh.fa, ya,
                                                           sh.na, sh.fb ? (const void*) qb : (const void*) bb, sh.fb ? qb + nvb : nullptr,
                                                           sh.fb, yb, sh.nb, dx, sh.n_in, 1, s);
            CK(cudaStreamSynchronize(s));
            std::vector<float> h1(sh.na), h2(sh.na), h3(sh.nb), h4(sh.nb);
            CK(cudaMemcpy(h1.data(), ya, h1.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h2.data(), ra, h2.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h3.data(), yb, h3.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h4.data(), rb, h4.size() * 4, cudaMemcpyDeviceToHost));
            const bool same = launched && std::memcmp(h1.data(), h2.data(), h1.size() * 4) == 0 && std::memcmp(h3.data(), h4.data(), h3.size() * 4) == 0;
            all_same = all_same && same;
            auto timeit = [&](bool pair) {
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a));
                CK(cudaEventCreate(&b));
                std::vector<float> ms;
                for (int rep = 0; rep < 30; ++rep) {
                    CK(cudaEventRecord(a, s));
                    for (int k = 0; k < 64; ++k) {
                        if (pair) strata::x::gemv_pair_w(sh.fa ? (const void*) qa : (const void*) ba, sh.fa ? qa + nva : nullptr, sh.fa, ya, sh.na,
                                                           sh.fb ? (const void*) qb : (const void*) bb, sh.fb ? qb + nvb : nullptr, sh.fb, yb, sh.nb,
                                                           dx, sh.n_in, 1, s);
                        else { one(sh.fa, qa, ba, nva, ra, sh.na); one(sh.fb, qb, bb, nvb, rb, sh.nb); }
                    }
                    CK(cudaEventRecord(b, s));
                    CK(cudaEventSynchronize(b));
                    float t;
                    CK(cudaEventElapsedTime(&t, a, b));
                    ms.push_back(t * 1000.0f / 64.0f);
                }
                std::sort(ms.begin(), ms.end());
                return ms[ms.size() / 2];
            };
            {   // the clocks up first: ~1.5 s of these launches (a short burst runs at the idle clocks)
                const auto w0 = std::chrono::steady_clock::now();
                while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < 1.5) timeit(false);
            }
            const float t2 = timeit(false), t1 = timeit(true), t2b = timeit(false), t1b = timeit(true);
            std::printf("   (again: two calls %6.1f us, pair %6.1f us)\n", t2b, t1b);
            std::printf("%-36s %5d -> %4d + %4d | two calls %6.1f us, pair %6.1f us (L2-warm) | %s\n", sh.what, sh.n_in, sh.na, sh.nb, t2, t1,
                        !launched ? "NOT LAUNCHED" : same ? "bit for bit" : "DIFFER");
            for (void* p : {(void*) qa, (void*) qb, (void*) ba, (void*) bb, (void*) dx, (void*) ya, (void*) yb, (void*) ra, (void*) rb}) CK(cudaFree(p));
        }
    }
    // the F32 one-column products (gemv_f32_w at T 1: mHC's mixes) against the row kernel's order emulated on the CPU - thread t's
    // fmaf chain over values t, t + 256, ..., warp_sum's xor tree, the 8 warps from 0.0f
    struct FShape { int n_in, n_out; const char* what; };
    const FShape fs[] = {{16384, 24, "mHC mixes (4 x 4096 -> 24)"}, {16384, 4, "mHC head (4 x 4096 -> 4)"}, {100, 7, "odd widths"}};
    for (const FShape& sh : fs) {
        if (only && *only && !std::strstr(sh.what, only)) continue;
        std::vector<float> w((size_t) sh.n_in * sh.n_out), x(sh.n_in), ref(sh.n_out);
        for (float& v : w) v = (float) ((int) (rng() % 20001) - 10000) / 7919.0f;
        for (float& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.0f;
        for (int r = 0; r < sh.n_out; ++r) {
            float part[256];
            for (int t = 0; t < 256; ++t) {
                float a = 0.0f;
                for (int i = t; i < sh.n_in; i += 256) a = std::fmaf(w[(size_t) r * sh.n_in + i], x[i], a);
                part[t] = a;
            }
            float tot = 0.0f;
            for (int wp = 0; wp < 8; ++wp) {
                float v[32];
                for (int l = 0; l < 32; ++l) v[l] = part[32 * wp + l];
                for (int o = 16; o > 0; o >>= 1) {
                    float nv[32];
                    for (int l = 0; l < 32; ++l) nv[l] = v[l] + v[l ^ o];
                    std::memcpy(v, nv, sizeof v);
                }
                tot += v[0];
            }
            ref[r] = tot;
        }
        float *dw, *dx, *dy;
        CK(cudaMalloc(&dw, w.size() * 4));
        CK(cudaMalloc(&dx, x.size() * 4));
        CK(cudaMalloc(&dy, ref.size() * 4));
        CK(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        std::vector<float> ms;
        for (int rep = 0; rep < 30; ++rep) {
            CK(cudaEventRecord(a, s));
            for (int k = 0; k < 64; ++k) strata::x::gemv_f32_w(dw, dx, dy, sh.n_in, sh.n_out, 1, s);
            CK(cudaEventRecord(b, s));
            CK(cudaEventSynchronize(b));
            float t;
            CK(cudaEventElapsedTime(&t, a, b));
            ms.push_back(t * 1000.0f / 64.0f);
        }
        std::sort(ms.begin(), ms.end());
        std::vector<float> got(sh.n_out);
        CK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
        const bool same = std::memcmp(got.data(), ref.data(), got.size() * 4) == 0;
        all_same = all_same && same;
        std::printf("%-30s %5d -> %3d | f32 %8.1f us (L2-warm) | %s\n", sh.what, sh.n_in, sh.n_out, ms[ms.size() / 2],
                    same ? "bit for bit (the row kernel's order)" : "DIFFER");
        CK(cudaFree(dw));
        CK(cudaFree(dx));
        CK(cudaFree(dy));
    }
    std::printf("%s\n", all_same ? "ALL bit for bit" : "SOME DIFFER");
    return all_same ? 0 : 1;
}
