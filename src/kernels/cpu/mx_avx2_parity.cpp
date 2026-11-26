// src/kernels/cpu/mx_avx2_parity.cpp - moex: mx_avx2.cpp's multi-token MXFP4 rows against ggml-cpu's
// ggml_vec_dot_mxfp4_q8_0, bit for bit, for every token of groups of 1..8, on random blocks (E8M0 exponents across the
// whole range incl. the denormal 0 and 1) and Q8_0 activations from ggml's own quantizer; row lengths with an even
// block count (the experts' 4096 and 2048) and odd ones (ggml's scalar tail).  Then the native expert entry points
// (native_gu_rows / native_down_rows and the split forms) against ggml-cpu's rows with the kernel switched off by
// hand, incl. the SwiGLU clamp.  No GPU, no model.  --bench: ns per row and token on one thread, rows from DRAM.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/mx_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

std::vector<uint8_t> random_mx_rows(std::mt19937& rng, int rows, int n, bool edge_exponents) {
    const int nb = n / 32;
    std::vector<uint8_t> w((size_t) rows * nb * 17);
    std::uniform_int_distribution<int> byte(0, 255), e_mid(110, 135);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            uint8_t* blk = w.data() + ((size_t) r * nb + b) * 17;
            int e = e_mid(rng);
            if (edge_exponents && (b % 17) == 3) e = (b / 17) % 4 == 0 ? 0 : (b / 17) % 4 == 1 ? 1 : (b / 17) % 4 == 2 ? 2 : 160;
            blk[0] = (uint8_t) e;
            for (int i = 1; i < 17; ++i) blk[i] = (uint8_t) byte(rng);
        }
    return w;
}

std::vector<uint8_t> q8_0_rows(std::mt19937& rng, int nt, int n) {
    const auto* tq = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    const size_t rb = ggml_row_size(GGML_TYPE_Q8_0, n);
    std::vector<uint8_t> a((size_t) nt * rb);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) n);
    for (int t = 0; t < nt; ++t) {
        for (auto& v : x) v = nd(rng) * (t % 3 == 2 ? 30.f : 1.f);
        tq->from_float(x.data(), a.data() + (size_t) t * rb, n);
    }
    return a;
}

bool cpu512 = false;   // set in main: the AVX-512BW kernel is checked too where the CPU has it

int check_rows(std::mt19937& rng, int n, int rows, bool edges) {
    const auto* tm = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    const size_t row = ggml_row_size(GGML_TYPE_MXFP4, n), arb = ggml_row_size(GGML_TYPE_Q8_0, n);
    const auto w = random_mx_rows(rng, rows, n, edges);
    int bad = 0;
    for (int nt = 1; nt <= 8; ++nt) {
        const auto a = q8_0_rows(rng, nt, n);
        std::vector<const void*> ap(nt);
        std::vector<std::vector<float>> out(nt, std::vector<float>((size_t) rows, -1.f));
        std::vector<float*> op(nt);
        for (int t = 0; t < nt; ++t) { ap[t] = a.data() + (size_t) t * arb; op[t] = out[t].data(); }
        for (int kern = 0; kern < (cpu512 ? 2 : 1); ++kern) {
            if (kern == 0) cpu::mx256_rows(w.data(), row, n, ap.data(), nt, op.data(), 0, rows);
            else cpu::mx512_rows(w.data(), row, n, ap.data(), nt, op.data(), 0, rows);
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < rows; ++r) {
                    float ref = 0.f;
                    tm->vec_dot(n, &ref, 0, w.data() + (size_t) r * row, 0, ap[t], 0, 1);
                    if (std::memcmp(&ref, &out[t][r], 4) != 0) {
                        if (bad < 5)
                            std::printf("  %s n %d nt %d token %d row %d: %.9g vs ggml %.9g\n", kern ? "mx512" : "mx256", n, nt, t, r,
                                        out[t][r], ref);
                        ++bad;
                    }
                }
        }
    }
    std::printf("mx256%s_rows n %4d (%3d blocks)%s: %s\n", cpu512 ? " + mx512" : "", n, n / 32, edges ? " incl. E8M0 0/1/2/160" : "",
                bad ? "DIFFERS" : "bitwise equal to ggml-cpu for nt 1..8");
    return bad;
}

// the native entry points with the kernel against the same rows computed through ggml-cpu by hand (what they did before)
int check_native(std::mt19937& rng, int n_embd, int n_ff, float lim) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, n_embd, n_ff, f, err)) { std::printf("%s\n", err.c_str()); return 1; }
    f.swiglu_limit = lim;
    const auto* tm = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    const auto gate = random_mx_rows(rng, n_ff, n_embd, false), up = random_mx_rows(rng, n_ff, n_embd, false);
    const auto down = random_mx_rows(rng, n_embd, n_ff, false);
    std::vector<uint8_t> blob(f.bytes);
    std::memcpy(blob.data(), gate.data(), f.up_off);
    std::memcpy(blob.data() + f.up_off, up.data(), f.up_off);
    std::memcpy(blob.data() + f.down_off, down.data(), f.bytes - f.down_off);
    int bad = 0;
    for (int nt = 1; nt <= 8; ++nt) {
        std::vector<std::vector<uint8_t>> act(nt, std::vector<uint8_t>(f.act_bytes)), hq(nt, std::vector<uint8_t>(f.h_bytes));
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<const void*> ap(nt), hp(nt);
        for (int t = 0; t < nt; ++t) {
            std::vector<float> x((size_t) n_embd), h((size_t) n_ff);
            for (auto& v : x) v = nd(rng);
            for (auto& v : h) v = nd(rng) * 4.f;
            cpu::native_quant_act(f, x.data(), act[t].data());
            cpu::native_quant_h(f, h.data(), hq[t].data());
            ap[t] = act[t].data();
            hp[t] = hq[t].data();
        }
        std::vector<std::vector<float>> ff(nt, std::vector<float>((size_t) n_ff)), ff2 = ff, o(nt, std::vector<float>((size_t) n_embd)), o2 = o;
        std::vector<float*> fp(nt), fp2(nt), op(nt), op2(nt);
        for (int t = 0; t < nt; ++t) { fp[t] = ff[t].data(); fp2[t] = ff2[t].data(); op[t] = o[t].data(); op2[t] = o2[t].data(); }
        cpu::native_gu_rows(f, blob.data(), ap.data(), nt, fp.data(), 0, n_ff);
        cpu::native_gu_rows_split(f, blob.data(), blob.data() + f.up_off, ap.data(), nt, fp2.data(), 0, n_ff);
        cpu::native_down_rows(f, blob.data(), hp.data(), nt, op.data(), 0, n_embd);
        cpu::native_down_rows_at(f, blob.data() + f.down_off, hp.data(), nt, op2.data(), 0, n_embd);
        for (int t = 0; t < nt; ++t) {
            for (int r = 0; r < n_ff; ++r) {   // the pre-kernel ggml-cpu path, by hand
                float g = 0.f, u = 0.f;
                tm->vec_dot(n_embd, &g, 0, blob.data() + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                tm->vec_dot(n_embd, &u, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                if (f.swiglu_limit > 0.0f) {
                    g = std::min(g, f.swiglu_limit);
                    u = std::clamp(u, -f.swiglu_limit, f.swiglu_limit);
                }
                const float ref = (g / (1.f + std::exp(-g))) * u;
                if (std::memcmp(&ref, &ff[t][r], 4) != 0 || std::memcmp(&ref, &ff2[t][r], 4) != 0) ++bad;
            }
            for (int r = 0; r < n_embd; ++r) {
                float s = 0.f;
                tm->vec_dot(n_ff, &s, 0, blob.data() + f.down_off + (size_t) r * f.d_row, 0, hp[t], 0, 1);
                if (std::memcmp(&s, &o[t][r], 4) != 0 || std::memcmp(&s, &o2[t][r], 4) != 0) ++bad;
            }
        }
    }
    std::printf("native experts %d x %d MXFP4/MXFP4, clamp %g: %s\n", n_embd, n_ff, (double) lim,
                bad ? "DIFFERS" : "gate/up + SwiGLU and down bitwise equal to ggml-cpu's rows for nt 1..8 (blob and split forms)");
    return bad;
}

void bench(std::mt19937& rng) {
    const int n = 4096, rows = 2048 * 24;   // 24 experts' gate rows (~107 MB): from DRAM
    const auto* tm = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    const size_t row = ggml_row_size(GGML_TYPE_MXFP4, n), arb = ggml_row_size(GGML_TYPE_Q8_0, n);
    const auto w = random_mx_rows(rng, rows, n, false);
    for (int nt = 1; nt <= 4; ++nt) {
        const auto a = q8_0_rows(rng, nt, n);
        std::vector<const void*> ap(nt);
        std::vector<std::vector<float>> out(nt, std::vector<float>((size_t) rows));
        std::vector<float*> op(nt);
        for (int t = 0; t < nt; ++t) { ap[t] = a.data() + (size_t) t * arb; op[t] = out[t].data(); }
        auto t0 = std::chrono::steady_clock::now();
        cpu::mx256_rows(w.data(), row, n, ap.data(), nt, op.data(), 0, rows);
        const double k = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double k5 = 0.0;
        if (cpu::cpu_avx512bw_ok()) {
            t0 = std::chrono::steady_clock::now();
            cpu::mx512_rows(w.data(), row, n, ap.data(), nt, op.data(), 0, rows);
            k5 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < rows; ++r)
            for (int t = 0; t < nt; ++t) {
                float s;
                tm->vec_dot(n, &s, 0, w.data() + (size_t) r * row, 0, ap[t], 0, 1);
                out[t][r] = s;
            }
        const double g = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  nt %d: mx256 %.2f GB/s (%5.1f ns per row-token)   mx512 %.2f GB/s (%5.1f ns)   ggml-cpu rows %.2f GB/s (%5.1f ns)\n",
                    nt, rows * row / k / 1e9, 1e9 * k / rows / nt, k5 > 0 ? rows * row / k5 / 1e9 : 0.0,
                    k5 > 0 ? 1e9 * k5 / rows / nt : 0.0, rows * row / g / 1e9, 1e9 * g / rows / nt);
    }
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_cpu_init();
    cpu512 = cpu::cpu_avx512bw_ok();
    std::mt19937 rng(20261010);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        bench(rng);
        return 0;
    }
    int bad = 0;
    bad += check_rows(rng, 4096, 64, false);
    bad += check_rows(rng, 2048, 64, false);
    bad += check_rows(rng, 4096, 32, true);
    bad += check_rows(rng, 96, 16, true);   // 3 blocks: one pair + the scalar tail
    bad += check_rows(rng, 32, 16, false);  // 1 block: the tail only
    bad += check_native(rng, 4096, 2048, 0.0f);
    bad += check_native(rng, 4096, 2048, 10.0f);
    std::printf("mx_avx2_parity: %s\n", bad ? "FAILED" : "all bitwise equal");
    return bad ? 1 : 0;
}
