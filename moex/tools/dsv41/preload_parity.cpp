// moex (DSV41_PLAN §34): the decode kernels' opt-in forms against their defaults, byte for byte.  One process runs one form
// (STRATA_DSV41_PRELOAD=1 / STRATA_DSV41_MIX_FUSED=1 set or not - the kernels read them once), so: run it twice, compare the
// two files.  The inputs are the same fixed-seed rows each run, at DeepSeek-V4.1's shapes (5,120 wide, 4 streams, 384 experts
// top-6, 24 mixer rows of 20,480 split 16 ways), many draws each:
//   preload_parity OUT [draws]
// The routing (ids and weights, with the model's bias and a penalty table, and without), the combine of six slots (a random
// mask between the GPU's rows and the CPU's), the norms of 5,120 and 20,480 floats (with a weight and without, in place), the
// mixer read (gemv_f32_split + mhc_coef_w + mhc_pre_w + the carried pre's copy + rms_norm, or mhc_mix_in_w1 in one launch).
#include "strata/x/dsv4_kernels.hpp"
#include "strata/x/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace g = strata::x;

#define CK(x)                                                                                              \
    do {                                                                                                   \
        const cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_));                                 \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

template <class T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    CK(cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: preload_parity OUT [draws]\n");
        return 2;
    }
    const int draws = argc > 2 ? std::atoi(argv[2]) : 200;
    std::FILE* out = std::fopen(argv[1], "wb");
    if (!out) return 1;
    auto put = [&](const void* p, size_t b) { std::fwrite(p, 1, b, out); };
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> coin(0, 1);
    auto randv = [&](size_t n, float sc) {
        std::vector<float> v(n);
        for (auto& x : v) x = nd(rng) * sc;
        return v;
    };
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    const int n = 5120, E = 384, k = 6, hcn = 4 * n;
    for (int d = 0; d < draws; ++d) {
        // the routing: logits, the model's bias, a penalty table (0 or 0.3), with and without the table
        {
            auto lg = randv(E, 3.0f), bias = randv(E, 0.1f);
            std::vector<float> pen(E);
            for (auto& p : pen) p = coin(rng) ? 0.3f : 0.0f;
            float *dl = dev(lg), *db = dev(bias), *dp = dev(pen), *dw = nullptr;
            int32_t* di = nullptr;
            CK(cudaMalloc(&dw, k * sizeof(float)));
            CK(cudaMalloc(&di, k * sizeof(int32_t)));
            for (int with_pen = 0; with_pen < 2; ++with_pen) {
                g::dsv4_route(dl, db, nullptr, E, k, 1.5f, di, dw, 1, s, with_pen ? dp : nullptr, true);
                CK(cudaStreamSynchronize(s));
                const auto ids = host(di, k);
                const auto w = host(dw, k);
                put(ids.data(), k * 4);
                put(w.data(), k * 4);
            }
            cudaFree(dl), cudaFree(db), cudaFree(dp), cudaFree(dw), cudaFree(di);
        }
        // the combine: six slots, a random mask between the GPU's rows and the CPU's, the shared expert's row
        {
            auto parts = randv((size_t) k * n, 1.0f), cpu = randv((size_t) k * n, 1.0f), wts = randv(k, 0.3f), sh = randv(n, 1.0f);
            std::vector<uint8_t> mask(k);
            for (auto& m : mask) m = (uint8_t) coin(rng);
            float *dpart = dev(parts), *dcpu = dev(cpu), *dwt = dev(wts), *dsh = dev(sh), *dout = nullptr;
            uint8_t* dm = dev(mask);
            CK(cudaMalloc(&dout, n * sizeof(float)));
            g::moe_combine_pairs(dpart, dcpu, dm, dwt, dsh, dout, n, k, 1, s);
            CK(cudaStreamSynchronize(s));
            const auto o = host(dout, n);
            put(o.data(), n * 4);
            cudaFree(dpart), cudaFree(dcpu), cudaFree(dwt), cudaFree(dsh), cudaFree(dout), cudaFree(dm);
        }
        // the f32 gemv of one column (the routers' 5,120 -> 384, the indexer's 512 -> 128 and 1,024 -> 64 x 128)
        {
            const int shapes[3][2] = {{5120, 384}, {512, 128}, {1024, 512}};
            for (const auto& sh : shapes) {
                auto W = randv((size_t) sh[0] * sh[1], 0.02f), x = randv(sh[0], 1.0f);
                float *dW = dev(W), *dx = dev(x), *dy = nullptr;
                CK(cudaMalloc(&dy, sh[1] * sizeof(float)));
                g::gemv_f32_w(dW, dx, dy, sh[0], sh[1], 1, s);
                CK(cudaStreamSynchronize(s));
                const auto y = host(dy, sh[1]);
                put(y.data(), (size_t) sh[1] * 4);
                cudaFree(dW), cudaFree(dx), cudaFree(dy);
            }
        }
        // the norms: 5,120 with a weight in place, 20,480 without
        {
            auto x1 = randv(n, 2.0f), w1 = randv(n, 0.5f), x2 = randv(hcn, 1.0f);
            float *dx1 = dev(x1), *dw1 = dev(w1), *dx2 = dev(x2), *dy2 = nullptr;
            CK(cudaMalloc(&dy2, hcn * sizeof(float)));
            g::rms_norm(dx1, dw1, dx1, 1, n, 1e-6f, s);
            g::rms_norm(dx2, nullptr, dy2, 1, hcn, 1e-6f, s);
            CK(cudaStreamSynchronize(s));
            const auto a = host(dx1, n), b = host(dy2, hcn);
            put(a.data(), n * 4);
            put(b.data(), (size_t) hcn * 4);
            cudaFree(dx1), cudaFree(dw1), cudaFree(dx2), cudaFree(dy2);
        }
        // the mixer read: rms_norm of the streams, the split gemv, then (default) reduce + coef + pre + copy + norm, or
        // (STRATA_DSV41_MIX_FUSED=1) the partials + mhc_mix_in_w1
        {
            auto R = randv(hcn, 1.0f), W = randv((size_t) 24 * hcn, 0.01f), sc = randv(3, 0.5f), bs = randv(24, 0.5f),
                 cpre = randv(24, 0.5f), nw = randv(n, 0.5f);
            float *dR = dev(R), *dW = dev(W), *dsc = dev(sc), *dbs = dev(bs), *dcpre = dev(cpre), *dnw = dev(nw);
            float *dflat = nullptr, *dmix = nullptr, *dcoef = nullptr, *dpart = nullptr, *dcur = nullptr;
            const int S = g::gemv_f32_split_parts(hcn);
            CK(cudaMalloc(&dflat, hcn * sizeof(float)));
            CK(cudaMalloc(&dmix, 24 * sizeof(float)));
            CK(cudaMalloc(&dcoef, 24 * sizeof(float)));
            CK(cudaMalloc(&dpart, (size_t) 24 * S * sizeof(float)));
            CK(cudaMalloc(&dcur, n * sizeof(float)));
            const float eps = 1e-6f, rms = 1e-6f;
            const int iters = 20;
            static const bool fused = [] {
                const char* e = std::getenv("STRATA_DSV41_MIX_FUSED");
                return e && e[0] == '1';
            }();
            g::rms_norm(dR, nullptr, dflat, 1, hcn, rms, s);
            if (fused) {
                g::gemv_f32_split_part(dW, dflat, dpart, hcn, 24, 1, s);
                g::mhc_mix_in_w1(dpart, S, dsc, dbs, dmix, dcoef, dcpre, dR, dcur, dnw, n, eps, iters, rms, s);
            } else {
                g::gemv_f32_split(dW, dflat, dmix, dpart, hcn, 24, 1, s);
                g::mhc_coef_w(dmix, dsc, dbs, dcoef, eps, iters, 1, s);
                g::mhc_pre_w(dR, dcpre, dcur, n, 1, s);
                CK(cudaMemcpyAsync(dcpre, dcoef, 24 * sizeof(float), cudaMemcpyDeviceToDevice, s));
                g::rms_norm(dcur, dnw, dcur, 1, n, rms, s);
            }
            CK(cudaStreamSynchronize(s));
            const auto m = host(dmix, 24), c = host(dcoef, 24), cp = host(dcpre, 24), cu = host(dcur, n);
            put(m.data(), 96), put(c.data(), 96), put(cp.data(), 96), put(cu.data(), (size_t) n * 4);
            cudaFree(dR), cudaFree(dW), cudaFree(dsc), cudaFree(dbs), cudaFree(dcpre), cudaFree(dnw), cudaFree(dflat), cudaFree(dmix);
            cudaFree(dcoef), cudaFree(dpart), cudaFree(dcur);
        }
    }
    std::fclose(out);
    std::printf("preload_parity: %d draws written to %s\n", draws, argv[1]);
    // the slot check's hash (§35): slices_hash_parts' 32 parts summed against slices_hash, one process, records' sizes
    {
        bool same = true;
        for (int d = 0; d < 20 && same; ++d) {
            const uint64_t lens[3] = {4718592ull + 8ull * d + (d % 3), 4718592ull, 5324800ull + (uint64_t) d};   // a tail on some
            std::vector<uint8_t> blob(lens[0] + lens[1] + lens[2] + 64);
            for (auto& b : blob) b = (uint8_t) (rng() & 0xff);
            uint8_t* db = dev(blob);
            std::vector<g::SliceJob> jobs = {{db, lens[0]}, {db + lens[0], lens[1]}, {db + lens[0] + lens[1], lens[2]}};
            g::SliceJob* dj = dev(jobs);
            uint64_t *o1 = nullptr, *o2 = nullptr;
            CK(cudaMalloc(&o1, 3 * sizeof(uint64_t)));
            CK(cudaMalloc(&o2, 3 * 32 * sizeof(uint64_t)));
            g::slices_hash(dj, 3, o1, s);
            g::slices_hash_parts(dj, 3, 32, o2, s);
            CK(cudaStreamSynchronize(s));
            const auto a = host(o1, 3), b = host(o2, 96);
            for (int j = 0; j < 3; ++j) {
                uint64_t h = 0;
                for (int q = 0; q < 32; ++q) h += b[(size_t) j * 32 + q];
                same = same && h == a[(size_t) j] && h == g::slices_hash_host(blob.data() + (j == 0 ? 0 : j == 1 ? lens[0] : lens[0] + lens[1]), lens[j]);
            }
            cudaFree(db), cudaFree(dj), cudaFree(o1), cudaFree(o2);
        }
        std::printf("preload_parity: slices_hash_parts against slices_hash and the host's: %s\n", same ? "PASS" : "FAIL");
        if (!same) return 1;
    }
    return 0;
}
