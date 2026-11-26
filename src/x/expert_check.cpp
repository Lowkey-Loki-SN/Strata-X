// src/x/expert_check.cpp - the routed experts' arithmetic against the exact product (moex integrity check).
//
//     strata-x -m MODEL.gguf --expert-check DUMP_DIR [--check-layers 1,2,3]
//
// For every MoE layer and step of a llama_dump (the oracle's), the experts' input (ffn_norm-L; a latent MoE's ffn_moe_lat-L,
// ERNIE-5.5's reference dump) and the experts it routed
// (ffn_moe_topk-L): each expert's gate / up / SwiGLU / down three ways -
//   EXACT  float64 over ggml's dequantized weights and the float activations;
//   CPU    llama.cpp's CPU arithmetic: the activations converted to the weight's vec_dot_type (Q8_K for the k- and i-quants,
//          Q8_0 for MXFP4), ggml's vec_dot per row, silu(min(g, L)) * clamp(u, -L, L) in float (SiTU-GLU for the kSituGlu
//          "limit", ERNIE-5.5's) (the dump's own values
//          check this transcription);
//   GPU    this engine's VRAM-hit path: Q8_1 activations and native MMVQ, the SwiGLU, Q8_1 again, native MMVQ;
//   POOL   this engine's CPU-pool path on a miss, in a group of two tokens (an MTP window's): native_gu_rows /
//          native_down_rows on the expert's [gate | up | down] blob - the multi-token kernels where this CPU has them
//          (they must agree with CPU to float rounding: same activations, same integer dots), else ggml-cpu's rows;
// and prints, per layer and expert format, each path's relative L2 error against EXACT (mean and worst over the experts).
// A GPU path that disagrees with llama.cpp's CPU by more than quantization noise is only acceptable when it is CLOSER to
// the exact product; this prints the evidence.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/x/kernels.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/glu_act.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace strata::x {

namespace kn = strata::kernels;
namespace kc = strata::kernels::cpu;

namespace {

struct DumpRec {
    std::string type;
    int64_t ne[4];
    uint64_t off;
};

double rel_l2(const std::vector<float>& a, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}
double rel_l2f(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - b[i];
        num += d * d;
        den += (double) b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

float silu_lim(float g, float u, float L) {   // ggml swiglu (L == 0) or swiglu_oai-style clamp (llama.cpp's limited SwiGLU)
    if (L < 0) return strata::kernels::situ_glu(g, u);   // moex (ERNIE-5.5): the kSituGlu sentinel
    if (L > 0) {
        g = std::min(g, L);
        u = std::max(-L, std::min(u, L));
    }
    return g / (1.0f + expf(-g)) * u;
}

/// rows x n_in weight block -> y[rows] = W x with the CPU's arithmetic (activation in vec_dot_type, ggml vec_dot)
void cpu_mv(int type, const uint8_t* w, int n_in, int rows, const float* x, float* y) {
    const ggml_type_traits_cpu* tc = ggml_get_type_traits_cpu((ggml_type) type);
    const ggml_type vt = tc->vec_dot_type;
    std::vector<uint8_t> xq(ggml_row_size(vt, n_in));
    ggml_get_type_traits_cpu(vt)->from_float(x, xq.data(), n_in);
    const size_t rb = ggml_row_size((ggml_type) type, n_in);
    for (int r = 0; r < rows; ++r) tc->vec_dot(n_in, y + r, 0, w + (size_t) r * rb, 0, xq.data(), 0, 1);
}

/// exact: float64 over the dequantized rows
void exact_mv(int type, const uint8_t* w, int n_in, int rows, const double* x, double* y) {
    const ggml_type_traits* t = ggml_get_type_traits((ggml_type) type);
    const size_t rb = ggml_row_size((ggml_type) type, n_in);
    std::vector<float> row(n_in);
    for (int r = 0; r < rows; ++r) {
        t->to_float(w + (size_t) r * rb, row.data(), n_in);
        double s = 0;
        for (int i = 0; i < n_in; ++i) s += (double) row[i] * x[i];
        y[r] = s;
    }
}

}  // namespace

int run_expert_check(const std::string& model, const std::string& dump, const std::vector<int>& only_layers) {
    ggml_cpu_init();   // ggml-cpu's tables (the vec_dot kernels' lookups, the fp16 table)
    std::map<std::string, DumpRec> recs;   // "step|name" (first occurrence)
    {
        std::ifstream idx(dump + "/index.tsv");
        if (!idx) {
            fprintf(stderr, "expert-check: no %s/index.tsv\n", dump.c_str());
            return 2;
        }
        std::string line;
        while (std::getline(idx, line)) {
            std::stringstream ss(line);
            std::string step, name, type;
            DumpRec r{};
            std::getline(ss, step, '\t');
            std::getline(ss, name, '\t');
            std::getline(ss, type, '\t');
            ss >> r.ne[0] >> r.ne[1] >> r.ne[2] >> r.ne[3] >> r.off;
            r.type = type;
            recs.emplace(step + "|" + name, r);
        }
    }
    FILE* bin = fopen((dump + "/dump.bin").c_str(), "rb");
    if (!bin) {
        fprintf(stderr, "expert-check: no dump.bin\n");
        return 2;
    }
    auto read_f32 = [&](const DumpRec& r, std::vector<float>& v) {
        v.resize((size_t) (r.ne[0] * r.ne[1] * r.ne[2] * r.ne[3]));
        fseeko(bin, (off_t) r.off, SEEK_SET);
        return fread(v.data(), sizeof(float), v.size(), bin) == v.size();
    };
    auto read_i32 = [&](const DumpRec& r, std::vector<int32_t>& v) {
        v.resize((size_t) (r.ne[0] * r.ne[1] * r.ne[2] * r.ne[3]));
        fseeko(bin, (off_t) r.off, SEEK_SET);
        return fread(v.data(), sizeof(int32_t), v.size(), bin) == v.size();
    };
    strata::GgufModel gm = strata::GgufModel::open(model);
    const std::string arch = gm.meta().get("general.architecture") ? gm.meta().get("general.architecture")->s : "";
    const strata::MetaValue* nl = gm.meta().get(arch + ".block_count");
    const strata::MetaValue* sw = gm.meta().get(arch + ".swiglu_clamp_exp");
    const int64_t n_layer = nl ? (int64_t) nl->num() : 0;
    cudaStream_t s = nullptr;
    cudaStreamCreate(&s);
    printf("%-6s %-22s %6s  %-24s %-24s %-24s %-12s %-24s\n", "layer", "format (gate/up, down)", "pairs", "GPU vs exact mean/worst",
           "CPU vs exact mean/worst", "GPU vs CPU mean/worst", "CPU vs dump", "POOL(2 tok) vs CPU mean/worst");
    for (int64_t il = 0; il < n_layer; ++il) {
        if (!only_layers.empty() && std::find(only_layers.begin(), only_layers.end(), (int) il) == only_layers.end()) continue;
        const std::string p = "blk." + std::to_string(il) + ".";
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* g = gm.find(p + "ffn_gate_exps.weight", &sg);
        const strata::TensorInfo* u = gm.find(p + "ffn_up_exps.weight", &su);
        const strata::TensorInfo* d = gm.find(p + "ffn_down_exps.weight", &sd);
        if (!g || !u || !d) continue;
        const int n = (int) g->shape[0], nf = (int) g->shape[1];
        const int gt = (int) g->type, dt = (int) d->type;
        float L = 0;
        if (sw && sw->type == strata::MetaType::ARRAY && (int64_t) sw->items.size() > il) L = (float) sw->items[(size_t) il].num();
        else if (sw) L = (float) sw->num();
        // moex (ERNIE-5.5): SiTU-GLU on every expert path - the engine's kSituGlu "limit"
        if (const strata::MetaValue* act = gm.meta().get(arch + ".activation"); act && act->s == "situ_glu") L = strata::kernels::kSituGlu;
        if (const char* lv = std::getenv("STRATA_X_CHECK_LIMIT")) L = (float) std::atof(lv);   // a test limit: makes the clamp bind
        const size_t gsl = ggml_row_size((ggml_type) gt, n) * nf, dsl = ggml_row_size((ggml_type) dt, nf) * n;
        const uint8_t* G = gm.shard(sg).tensor_data(*g);
        const uint8_t* U = gm.shard(su).tensor_data(*u);
        const uint8_t* D = gm.shard(sd).tensor_data(*d);
        if (!kn::native_mmvq_supported(gt) || !kn::native_mmvq_supported(dt)) {
            printf("%-6lld %-22s  (no GPU kernel)\n", (long long) il, (std::string(ggml_type_name((ggml_type) gt)) + "/" + ggml_type_name((ggml_type) dt)).c_str());
            continue;
        }
        uint8_t *dg = nullptr, *du = nullptr, *dd = nullptr, *xq = nullptr, *hq = nullptr;
        float *dx = nullptr, *yg = nullptr, *yu = nullptr, *yd = nullptr;
        cudaMalloc(&dg, gsl);
        cudaMalloc(&du, gsl);
        cudaMalloc(&dd, dsl);
        cudaMalloc(&dx, (size_t) n * 4);
        cudaMalloc(&xq, kn::native_q8_1_bytes(std::max(n, nf)));
        cudaMalloc(&hq, kn::native_q8_1_bytes(std::max(n, nf)));
        cudaMalloc(&yg, (size_t) nf * 4);
        cudaMalloc(&yu, (size_t) nf * 4);
        cudaMalloc(&yd, (size_t) n * 4);
        std::vector<double> e_gpu, e_cpu, e_gc, e_dump, s_gpu, s_cpu, e_pool;   // per expert; per step the weighted MoE sum
        kc::NativeFmt nf_fmt;
        std::string ferr;
        const bool pool_ok = kc::native_fmt(gt, dt, n, nf, nf_fmt, ferr);
        nf_fmt.swiglu_limit = L;
        std::vector<uint8_t> pblob(pool_ok ? nf_fmt.bytes : 0), pact(kc::kNativeActBytes), phq0(kc::kNativeHBytes), phq1(kc::kNativeHBytes);
        int pool_t1_diff = 0;
        for (int step = 0;; ++step) {
            // the experts' input: the FFN norm's output, or (a latent MoE: ERNIE-5.5's) the latent rows the dump has
            auto xin = recs.find(std::to_string(step) + "|ffn_moe_lat-" + std::to_string(il));
            if (xin == recs.end()) xin = recs.find(std::to_string(step) + "|ffn_norm-" + std::to_string(il));
            auto top = recs.find(std::to_string(step) + "|ffn_moe_topk-" + std::to_string(il));
            auto dn = recs.find(std::to_string(step) + "|ffn_moe_down-" + std::to_string(il));
            if (xin == recs.end() || top == recs.end()) break;
            std::vector<float> x, dref;
            std::vector<int32_t> ids;
            if (!read_f32(xin->second, x) || !read_i32(top->second, ids)) break;
            if (dn != recs.end()) read_f32(dn->second, dref);
            if ((int) x.size() != n) break;
            std::vector<double> xd(x.begin(), x.end());
            std::vector<float> wts;
            auto wr = recs.find(std::to_string(step) + "|ffn_moe_weights_scaled-" + std::to_string(il));
            if (wr == recs.end()) wr = recs.find(std::to_string(step) + "|ffn_moe_weights_norm-" + std::to_string(il));
            if (wr != recs.end()) read_f32(wr->second, wts);
            std::vector<double> sum_e(n, 0.0);
            std::vector<float> sum_g(n, 0.0f), sum_c(n, 0.0f);
            cudaMemcpyAsync(dx, x.data(), (size_t) n * 4, cudaMemcpyHostToDevice, s);
            kn::native_quantize_q8_1(dx, xq, n, 1, s);
            for (size_t j = 0; j < ids.size(); ++j) {
                const int e = ids[j];
                const uint8_t* ge = G + (size_t) e * gsl;
                const uint8_t* ue = U + (size_t) e * gsl;
                const uint8_t* de = D + (size_t) e * dsl;
                // exact
                std::vector<double> eg(nf), eu(nf), eh(nf), ed(n);
                exact_mv(gt, ge, n, nf, xd.data(), eg.data());
                exact_mv(gt, ue, n, nf, xd.data(), eu.data());
                for (int i = 0; i < nf; ++i) {
                    double gg = eg[i], uu = eu[i];
                    if (L < 0) {   // SiTU-GLU in float64
                        eh[i] = 4.0 * std::tanh(gg / 4.0) / (1.0 + std::exp(-gg)) * (25.0 * std::tanh(uu / 25.0));
                        continue;
                    }
                    if (L > 0) {
                        gg = std::min(gg, (double) L);
                        uu = std::max(-(double) L, std::min(uu, (double) L));
                    }
                    eh[i] = gg / (1.0 + std::exp(-gg)) * uu;
                }
                exact_mv(dt, de, nf, n, eh.data(), ed.data());
                // CPU (llama.cpp's)
                std::vector<float> cg(nf), cu(nf), ch(nf), cd(n);
                cpu_mv(gt, ge, n, nf, x.data(), cg.data());
                cpu_mv(gt, ue, n, nf, x.data(), cu.data());
                for (int i = 0; i < nf; ++i) ch[i] = silu_lim(cg[i], cu[i], L);
                cpu_mv(dt, de, nf, n, ch.data(), cd.data());
                // POOL (this engine's CPU miss, two tokens of one window): the same activation twice, so token 1 must equal
                // token 0 bit for bit and both must match CPU to float rounding
                if (pool_ok) {
                    std::memcpy(pblob.data(), ge, nf_fmt.up_off);
                    std::memcpy(pblob.data() + nf_fmt.up_off, ue, nf_fmt.up_off);
                    std::memcpy(pblob.data() + nf_fmt.down_off, de, dsl);
                    kc::native_quant_act(nf_fmt, x.data(), pact.data());
                    std::vector<float> f0(nf), f1(nf), p0(n), p1(n);
                    const void* acts[2] = {pact.data(), pact.data()};
                    float* ffs[2] = {f0.data(), f1.data()};
                    kc::native_gu_rows(nf_fmt, pblob.data(), acts, 2, ffs, 0, nf);
                    kc::native_quant_h(nf_fmt, f0.data(), phq0.data());
                    kc::native_quant_h(nf_fmt, f1.data(), phq1.data());
                    const void* hqs[2] = {phq0.data(), phq1.data()};
                    float* outs[2] = {p0.data(), p1.data()};
                    kc::native_down_rows(nf_fmt, pblob.data(), hqs, 2, outs, 0, n);
                    if (std::memcmp(p0.data(), p1.data(), (size_t) n * 4) != 0) ++pool_t1_diff;
                    e_pool.push_back(rel_l2f(p0, cd));
                }
                // GPU (this engine's VRAM hit)
                cudaMemcpyAsync(dg, ge, gsl, cudaMemcpyHostToDevice, s);
                cudaMemcpyAsync(du, ue, gsl, cudaMemcpyHostToDevice, s);
                cudaMemcpyAsync(dd, de, dsl, cudaMemcpyHostToDevice, s);
                kn::native_mmvq(gt, dg, xq, yg, n, nf, 1, s);
                kn::native_mmvq(gt, du, xq, yu, n, nf, 1, s);
                swiglu_clamp(yg, yu, yg, nf, L, s);
                kn::native_quantize_q8_1(yg, hq, nf, 1, s);
                kn::native_mmvq(dt, dd, hq, yd, nf, n, 1, s);
                std::vector<float> gd(n);
                cudaMemcpyAsync(gd.data(), yd, (size_t) n * 4, cudaMemcpyDeviceToHost, s);
                cudaStreamSynchronize(s);
                if (wts.size() == ids.size())
                    for (int i = 0; i < n; ++i) {
                        sum_e[i] += (double) wts[j] * ed[i];
                        sum_g[i] += wts[j] * gd[i];
                        sum_c[i] += wts[j] * cd[i];
                    }
                e_gpu.push_back(rel_l2(gd, ed));
                if (std::getenv("STRATA_X_CHECK_DETAIL") && rel_l2(gd, ed) > 0.1) {
                    double ng = 0, ne = 0;
                    for (int i = 0; i < n; ++i) {
                        ng += (double) gd[i] * gd[i];
                        ne += ed[i] * ed[i];
                    }
                    double hmax = 0, hn = 0;
                    for (int i = 0; i < nf; ++i) {
                        hmax = std::max(hmax, std::fabs(eh[i]));
                        hn += eh[i] * eh[i];
                    }
                    printf("  L%lld step %d e %d: gpu |y| %.4g exact |y| %.4g  rel %.3f  h: max %.4g rms %.4g\n", (long long) il, step, e,
                           std::sqrt(ng), std::sqrt(ne), rel_l2(gd, ed), hmax, std::sqrt(hn / nf));
                }
                e_cpu.push_back(rel_l2(cd, ed));
                e_gc.push_back(rel_l2f(gd, cd));
                if (!dref.empty()) {
                    std::vector<float> col(dref.begin() + (size_t) j * n, dref.begin() + (size_t) (j + 1) * n);
                    e_dump.push_back(rel_l2f(cd, col));
                }
            }
            if (wts.size() == ids.size()) {
                s_gpu.push_back(rel_l2(sum_g, sum_e));
                s_cpu.push_back(rel_l2(sum_c, sum_e));
            }
        }
        auto mean = [](const std::vector<double>& v) {
            double a = 0;
            for (double x : v) a += x;
            return v.empty() ? 0.0 : a / (double) v.size();
        };
        auto worst = [](const std::vector<double>& v) { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); };
        char fmt[64];
        snprintf(fmt, sizeof fmt, "%s/%s", ggml_type_name((ggml_type) gt), ggml_type_name((ggml_type) dt));
        printf("%-6lld %-22s %6zu  %9.3e / %9.3e    %9.3e / %9.3e    %9.3e / %9.3e    %9.3e    %9.3e / %9.3e%s\n", (long long) il, fmt,
               e_gpu.size(), mean(e_gpu), worst(e_gpu), mean(e_cpu), worst(e_cpu), mean(e_gc), worst(e_gc), worst(e_dump),
               mean(e_pool), worst(e_pool), pool_t1_diff ? "  (POOL token 1 != token 0!)" : "");
        printf("       the weighted MoE sum over %zu steps: GPU vs exact %9.3e / %9.3e   CPU (llama.cpp) vs exact %9.3e / %9.3e\n",
               s_gpu.size(), mean(s_gpu), worst(s_gpu), mean(s_cpu), worst(s_cpu));
        for (void* q : {(void*) dg, (void*) du, (void*) dd, (void*) dx, (void*) xq, (void*) hq, (void*) yg, (void*) yu, (void*) yd}) cudaFree(q);
    }
    fclose(bin);
    cudaStreamDestroy(s);
    return 0;
}

}  // namespace strata::x
