// src/kernels/cpu/native_expert.cpp - plan v0.3 P6: native (GGUF-form) experts on the CPU through ggml-cpu.
// See the header.  Nothing here is Strata arithmetic: the activation quantizers and the row dot products are
// ggml-cpu's, so an IQ expert computes what llama.cpp's CPU backend computes for it.
#include "strata/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/mx_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/glu_act.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <mutex>

namespace strata::kernels::cpu {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

// moex: MXFP4 rows (ggml type 39: the MXFP4 releases' routed experts) through mx_avx2.cpp's multi-token kernel, which is
// bit-exact against ggml-cpu's dot for every token - so any group size, no #152 rule, and the threshold is speed only.
// Sharing a block's nibble lookup across tokens saves little (the per-token maddubs/madd/scale/FMA dominate: ~460 ns
// per 4096-row and token either way, one busy-box thread, 2026-10-10) and ggml-cpu's own row is faster alone, so the
// kernel takes groups of STRATA_MX256_MIN (default 4) or more tokens.  STRATA_NO_MX256=1: always ggml-cpu's rows.
bool mx256_on(int type, int nt) {
    static const bool on = cpu_avx2_ok() && std::getenv("STRATA_NO_MX256") == nullptr;
    static const int mt_min = [] { const char* v = std::getenv("STRATA_MX256_MIN"); return v ? std::max(1, std::atoi(v)) : 4; }();
    static const bool on512 = cpu_avx512bw_ok() && std::getenv("STRATA_NO_MX512") == nullptr;
    static const int mt_min512 = [] { const char* v = std::getenv("STRATA_MX512_MIN"); return v ? std::max(1, std::atoi(v)) : 2; }();
    return type == 39 && nt <= MAXT && ((on && nt >= mt_min) || (on512 && nt >= mt_min512));
}

// moex: the MXFP4 rows' kernel for a group of nt tokens (mx256_on said one applies): the AVX-512BW one from
// STRATA_MX512_MIN (2) tokens - an expert shared by a window's tokens is compute-bound on this Skylake-SP - else AVX-2
void mx_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    static const bool on512 = cpu_avx512bw_ok() && std::getenv("STRATA_NO_MX512") == nullptr;
    static const int mt_min512 = [] { const char* v = std::getenv("STRATA_MX512_MIN"); return v ? std::max(1, std::atoi(v)) : 2; }();
    if (on512 && nt >= mt_min512) mx512_rows(w, row_bytes, n, act, nt, out, r0, r1);
    else mx256_rows(w, row_bytes, n, act, nt, out, r0, r1);
}

// MXFP4 gate/up rows [r0, r1) a chunk at a time: the chunk's gate and up dots for all tokens, then the SwiGLU exactly as
// the ggml-cpu path in native_gu_rows (the same expressions, the same clamp)
void mx_gu_rows(const NativeFmt& f, const uint8_t* gate, const uint8_t* up, const void* const* act, int nt,
                float* const* ff, int r0, int r1) {
    constexpr int R = 32;
    float gb[MAXT][R], ub[MAXT][R];
    float* gp[MAXT];
    float* upp[MAXT];
    for (int t = 0; t < nt; ++t) {
        gp[t] = gb[t];
        upp[t] = ub[t];
    }
    for (int r = r0; r < r1; r += R) {
        const int m = std::min(R, r1 - r);
        mx_rows(gate + (size_t) r * f.gu_row, f.gu_row, (int) f.n_embd, act, nt, gp, 0, m);
        mx_rows(up + (size_t) r * f.gu_row, f.gu_row, (int) f.n_embd, act, nt, upp, 0, m);
        for (int i = 0; i < m; ++i)
            for (int t = 0; t < nt; ++t) {
                float g = gb[t][i], u = ub[t][i];
                if (f.swiglu_limit < 0.0f) {   // moex (ERNIE-5.5): SiTU-GLU
                    ff[t][r + i] = strata::kernels::situ_glu(g, u);
                    continue;
                }
                if (f.swiglu_limit > 0.0f) {   // moex: ggml_compute_forward_swiglu_clamp_f32
                    g = std::min(g, f.swiglu_limit);
                    u = std::clamp(u, -f.swiglu_limit, f.swiglu_limit);
                }
                ff[t][r + i] = (g / (1.f + std::exp(-g))) * u;
            }
    }
}

}  // namespace

bool native_experts_available() noexcept { return true; }

// The "is this format handled by that kernel" questions, answered here and not next to the kernels: iq_avx512.cpp is
// compiled for AVX-512 and iq_avx2.cpp / kq_avx2.cpp for AVX2, and native_gu_rows asks these on every CPU before it
// has checked what the CPU can run.  A function in a wide-ISA file may use that ISA anywhere in its body, so an
// answer that is only a comparison must not come from one (#795).
bool iq512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22;
}

bool iq256_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23;
}

bool kq256_supported(int type) noexcept { return type == 12 || type == 7 || type == 8; }

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err) {
    init_once();
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    if (tg == nullptr || tg->vec_dot == nullptr || td == nullptr || td->vec_dot == nullptr) {
        err = "native experts: ggml-cpu has no dot product for type " + std::to_string(tg && tg->vec_dot ? d_type : gu_type);
        return false;
    }
    const ggml_type_traits_cpu* ag = traits(tg->vec_dot_type);
    const ggml_type_traits_cpu* ad = traits(td->vec_dot_type);
    if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "native experts: ggml-cpu cannot quantize an activation for this layer";
        return false;
    }
    if (n_embd % ggml_blck_size((ggml_type) gu_type) || n_ff % ggml_blck_size((ggml_type) d_type) ||
        n_embd % ggml_blck_size(tg->vec_dot_type) || n_ff % ggml_blck_size(td->vec_dot_type)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = (int) tg->vec_dot_type;
    f.d_act = (int) td->vec_dot_type;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    f.d_row = ggml_row_size((ggml_type) d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = 2 * f.up_off;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    f.act_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    f.h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

namespace {
// Q8_K (the activations of every i-quant row): ggml-cpu's x86 quantizer is the scalar reference, ~3 us per token
// and layer on the host before the pool can start; q8k_quant_avx2 writes the same bytes.  cpu_avx2_ok() too:
// iq_avx2.cpp is compiled for AVX2 (an AVX-only CPU, or STRATA_FORCE_ISA=avx, keeps ggml's).  STRATA_NO_Q8K_AVX2=1:
// ggml's on any CPU.
bool q8k_avx2(int type) {
    static const bool on = cpu_avx2_ok() && std::getenv("STRATA_NO_Q8K_AVX2") == nullptr;
    return on && type == (int) GGML_TYPE_Q8_K;
}
// the older CPUs' copy (iq_avx1.cpp): byte-identical like the AVX-2 one, and only where AVX2 is absent (same
// policy as STRATA_NO_IQ128 above; STRATA_NO_Q8K_AVX1=1 keeps ggml's scalar reference on those CPUs).
bool q8k_avx1(int type) {
    static const bool on = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("STRATA_NO_Q8K_AVX1") == nullptr;
    return on && type == (int) GGML_TYPE_Q8_K;
}
}  // namespace

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    if (q8k_avx2(f.gu_act)) { q8k_quant_avx2(x, dst, f.n_embd); return; }
    if (q8k_avx1(f.gu_act)) { q8k_quant_avx1(x, dst, f.n_embd); return; }
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    if (q8k_avx2(f.d_act)) { q8k_quant_avx2(h, dst, f.n_ff); return; }
    if (q8k_avx1(f.d_act)) { q8k_quant_avx1(h, dst, f.n_ff); return; }
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}

/// Whether the i-quant rows take iq_avx512.cpp: an AVX-512 CPU with VNNI and VBMI (Strata's gate), or - moex, opt-in
/// with STRATA_IQ512_BW=1 - one with F, BW, VL and DQ only (Skylake-SP: the file uses no VNNI / VBMI instruction and is
/// compiled without them).  STRATA_NO_IQ512 turns both off.  The formats without a 512-bit kernel keep following
/// cpu_avx512_ok() (`cpu512` below), so on a BW-only CPU they stay on iq256 as before.
static bool iq512_rows_ok() {
    static const bool ok = [] {
        if (std::getenv("STRATA_NO_IQ512") != nullptr) return false;
        if (cpu_avx512_ok()) return true;
        const char* bw = std::getenv("STRATA_IQ512_BW");
        return bw != nullptr && bw[0] == '1' && cpu_avx512bw_ok();
    }();
    return ok;
}

bool iq_one_token(int type);

int native_gu_mt_min(int gu_type) {
    // #152: from how many tokens the multi-token kernels run (ggml's vec_dot below that).  The default 2 is the
    // measured-fastest rule, but a token's expert rows then round differently alone than in a group, so greedy output
    // can depend on how many drafts a verify window held.  STRATA_IQ_MT_MIN=1 (opt-in, 0.1.30) uses the multi-token
    // kernels for every group: output independent of the drafting, at a measured -1..-3% decode on IQ3_S (AVX-512).
    static const char* env = std::getenv("STRATA_IQ_MT_MIN");
    static const int mt_min = env ? std::atoi(env) : 2;
    // IQ3_S where the AVX-2 kernel gathers its grid (cpu_gather_fast, unless STRATA_IQ256_GATHER=0) on a CPU without
    // AVX-512: there that kernel beats ggml's dot for ONE token too (14900KF P-core, an expert's gate/up rows: 0.34 ->
    // 0.19 ms; its E-cores, which keep the scalar decode: 0.71 -> 0.68), and in decode most CPU experts serve one token
    // of the window.  So IQ3_S takes it for every group, and those rows no longer depend on the drafting.  A machine-
    // wide rule, not a per-core one: every core rounds an expert the same.  Every probe here is in expert_layout.cpp
    // and cpu_avx2_ok() comes first, so no AVX2 code runs before the check.  STRATA_IQ_MT_MIN set keeps its rule.
    // Opt-in (STRATA_IQ3S_MT1=1): it changes a lone token's IQ3_S rounding on those CPUs, so the default stays 0.1.39's.
    static const bool iq3s_one = env == nullptr && std::getenv("STRATA_IQ3S_MT1") != nullptr && cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr &&
                                 !cpu_avx512_ok() && iq256_gather_setting() != 0 && cpu_gather_fast();
    if (iq3s_one && gu_type == 21) return 1;
    return env == nullptr && iq_one_token(gu_type) ? 1 : mt_min;
}

/// moex: the formats whose GATHERED AVX-2 decode beats ggml's dot for one token here, so every group takes it - the
/// 0.1.30 rule (STRATA_IQ_MT_MIN=1) per format, on a CPU whose gathers are fast (cpu_gather_fast, not turned off by
/// STRATA_IQ256_GATHER=0) and that runs the AVX-2 kernels for them (no iq512 rows).  Measured on a Xeon Gold 6148 (one
/// thread, one token, an expert's gate/up rows): IQ3_XXS 0.514 -> 0.418 ms, IQ3_S 0.627 -> 0.421, IQ2_XXS 0.365 -> 0.357;
/// IQ2_S stays on ggml's dot (0.352 against 0.420).  A token's rows then round the same alone as in a verify window
/// (#152): greedy output stops depending on the drafting.  STRATA_NO_IQ_ONE=1 restores Strata's default.
bool iq_one_token(int type) {
    static const bool on = std::getenv("STRATA_NO_IQ_ONE") == nullptr && cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr &&
                           !iq512_rows_ok() && iq256_gather_setting() != 0 && cpu_gather_fast();
    return on && (type == 18 || type == 21 || type == 16);
}

void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    // the multi-token kernels decode the weights once for all tokens: 2.0-2.4x ggml-cpu at three tokens, no faster
    // at one (all are bound by the codebook lookups, ~5 GB/s per core), measured by native_expert_parity.  AVX-512
    // first, then the AVX-2 one (Zen 2/3, Intel 12th-14th gen).  STRATA_NO_IQ512 drops an AVX-512 CPU to the
    // AVX-2 kernel, STRATA_NO_IQ256 drops the AVX-2 kernel; ggml-cpu's single-token vec_dot is reached only with
    // both set (and on a CPU without AVX-512, STRATA_NO_IQ512 changes nothing).
    static const bool avx512 = iq512_rows_ok();
    // STRATA_NO_IQ256 drops the AVX-2 kernel.  cpu_avx2_ok() is ALSO required: iq_avx2.cpp and kq_avx2.cpp
    // are compiled /arch:AVX2 and use AVX2 and FMA3, so calling them on a CPU without either is an
    // illegal instruction, not a slow path.  The gate sat next to `avx512` above, which does test
    // cpu_avx512_ok(), and that asymmetry is what let a pre-Haswell CPU reach iq256_gu_rows.
    // Without it this falls through to ggml-cpu's vec_dot below, which is compiled for whatever
    // baseline the build selected (AVX1 here) and covers the same types - IQ2_XXS and IQ2_S among
    // them.  That path loops over tokens itself, so it is correct for any `nt`, not just one.
    static const bool avx2 = cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr;
    // The older-CPU tier (STRATA_ISA_FLOOR=avx builds): without it every expert on such a CPU runs on ggml-cpu's
    // scalar _generic dots.  Only where AVX2 is absent, so a modern CPU's rounding never moves (and
    // STRATA_NO_IQ256=1 there still means ggml, as before).  iq_avx1.cpp is compiled for AVX and gated on
    // cpu_avx1_ok().  Measured on an E5-2470 v2 (iq_avx1_parity --bench): 1.2-3.1x ggml's dot from two tokens on,
    // at one token ggml's auto-vectorized generic loop matches the 128-bit kernel - so the same #152 mt_min rule
    // as the AVX-2 kernels, and STRATA_IQ_MT_MIN=1 takes the kernel for one token too (its rows are then
    // independent of the drafting: iq_avx1_parity checks every width bit for bit).
    static const bool avx1 = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("STRATA_NO_IQ128") == nullptr;
    const int mt_min = native_gu_mt_min(f.gu_type);   // #152
    // Unsloth UD-Q4_K_XL's Q4_K gate/up: the multi-token kernel is bit-exact against ggml's per-token dot (any group
    // size, no #152 rule).  Opt-in, STRATA_KQ256=1: measured no faster in the engine (a window's expert groups hold
    // ~1.4 tokens and the weights stay in L1 across ggml's per-token calls; 1.01-1.13x in native_expert_parity).
    static const bool kq = [] { const char* v = std::getenv("STRATA_KQ256"); return cpu_avx2_ok() && v != nullptr && std::atoi(v) != 0; }();
    if (kq && f.swiglu_limit == 0.0f && f.gu_type == 12 && nt >= 2) {   // one token: ggml's own dot below (the same bits, less overhead)
        kq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    // A format with only an AVX-2 kernel (IQ4_XS, #415) takes it on AVX-2 CPUs only: an AVX-512 CPU keeps ggml-cpu for
    // it, as before (its rows would round differently).  Each kernel only for the formats it implements: falling
    // through an empty switch would leave ff unwritten instead of falling back to ggml-cpu.
    static const bool cpu512 = cpu_avx512_ok();
    // moex: both multi-token kernels apply the SwiGLU limit too (GLM's experts clamp at 10, as the scalar path below)
    if (nt >= mt_min && avx512 && iq512_supported(f.gu_type)) {
        iq512_gu_rows_lim(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1, f.swiglu_limit);
        return;
    }
    if (nt >= mt_min && avx2 && iq256_supported(f.gu_type) && (iq512_supported(f.gu_type) || !cpu512)) {
        iq256_gu_rows_lim(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1, f.swiglu_limit);
        return;
    }
    if (mx256_on(f.gu_type, nt)) {   // moex: MXFP4, bit-exact against the rows below at any nt
        mx_gu_rows(f, blob, blob + f.up_off, act, nt, ff, r0, r1);
        return;
    }
    if (avx1 && nt >= mt_min && iq128_supported(f.gu_type)) {
        iq128_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1, f.swiglu_limit);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    const int n = (int) f.n_embd;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            if (f.swiglu_limit < 0.0f) {   // moex (ERNIE-5.5): SiTU-GLU
                ff[t][r] = strata::kernels::situ_glu(g, u);
                continue;
            }
            if (f.swiglu_limit > 0.0f) {   // moex: ggml_compute_forward_swiglu_clamp_f32
                g = std::min(g, f.swiglu_limit);
                u = std::clamp(u, -f.swiglu_limit, f.swiglu_limit);
            }
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; ggml-cpu's dot is single-token.  STRATA_NO_IQ4NL falls back to it.
    static const bool iq4nl_mt = std::getenv("STRATA_NO_IQ4NL") == nullptr;
    static const char* mt_env = std::getenv("STRATA_IQ_MT_MIN");
    static const int mt_min_all = mt_env ? std::atoi(mt_env) : 2;
    const int mt_min = mt_env == nullptr && iq_one_token(f.d_type) ? 1 : mt_min_all;   // moex: as the gate/up rows'
    static const bool kq = [] { const char* v = std::getenv("STRATA_KQ256"); return v != nullptr && std::atoi(v) != 0; }();
    // Both multi-token kernels below are /arch:AVX2 translation units (kq_avx2.cpp and iq_avx2.cpp),
    // so a CPU without AVX2 has to reach ggml-cpu's vec_dot instead - same reasoning as the gate/up
    // rows above, where `avx512` tested cpu_avx512_ok() and `avx2` did not.
    if (cpu_avx2_ok() && kq && nt >= 2 && (f.d_type == 7 || f.d_type == 8)) {   // Q5_1 / Q8_0 down: bit-exact, any group size
        kq256_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    if (cpu_avx2_ok() && nt >= mt_min && f.d_type == 20 && iq4nl_mt) {   // #152: the same rule as the gate/up rows
        iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    // moex: the i-quant down rows (UD GGUFs' IQ2_XXS / IQ3_XXS / IQ3_S, MiMo's) through the AVX-512 multi-token kernel too:
    // each row decoded once for all the expert's tokens (#152's rule: groups of mt_min or more)
    // (the gate/up rows' kernel rule: AVX-512 first, then AVX-2 - which an AVX-512 CPU takes only for the formats both have)
    static const bool avx512_d = iq512_rows_ok();
    if (avx512_d && nt >= mt_min && iq512_supported(f.d_type)) {
        iq512_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    static const bool avx2_d = cpu_avx2_ok() && std::getenv("STRATA_NO_IQ256") == nullptr;
    if (avx2_d && nt >= mt_min && iq256_supported(f.d_type) && (iq512_supported(f.d_type) || !cpu_avx512_ok())) {
        iq256_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    if (mx256_on(f.d_type, nt)) {   // moex: MXFP4, bit-exact against the rows below at any nt
        mx_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    // the older-CPU tier: IQ4_NL and Q2_0 down rows in 128-bit (ggml-cpu's dots for both are scalar here)
    static const bool avx1 = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("STRATA_NO_IQ128") == nullptr;
    if (avx1 && iq128_down_supported(f.d_type)) {
        iq128_down_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s;
        }
    }
}

void native_gu_rows_split(const NativeFmt& f, const uint8_t* gate, const uint8_t* up, const void* const* act, int nt,
                          float* const* ff, int r0, int r1) {
    if (mx256_on(f.gu_type, nt)) {   // moex: MXFP4 in place in a mapping too (bit-exact)
        mx_gu_rows(f, gate, up, act, nt, ff, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    const int n = (int) f.n_embd;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = gate + (size_t) r * f.gu_row;
        const uint8_t* ur = up + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            if (f.swiglu_limit < 0.0f) {   // moex (ERNIE-5.5): SiTU-GLU
                ff[t][r] = strata::kernels::situ_glu(g, u);
                continue;
            }
            if (f.swiglu_limit > 0.0f) {
                g = std::min(g, f.swiglu_limit);
                u = std::clamp(u, -f.swiglu_limit, f.swiglu_limit);
            }
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void native_down_rows_at(const NativeFmt& f, const uint8_t* down, const void* const* hq, int nt, float* const* out,
                         int r0, int r1) {
    if (mx256_on(f.d_type, nt)) {   // moex: MXFP4 in place in a mapping too (bit-exact)
        mx_rows(down, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = down + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s;
        }
    }
}

}  // namespace strata::kernels::cpu
