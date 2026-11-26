// moex/tools/hy4/hy4_requant.cpp - a Hy4-flash (hy_v4_flash) GGUF with some tensors re-quantized, for the engine's checks of
// a set's types before (or without) that set's download: the SPARK-4Bit sets' NVFP4 experts and dense projections, their
// Q8_0 stream mixers, indexer head weights and Engram table, out of a slice of the Q4_K_M set (moex/tools/hy4/hy4_slice.py).
//
//     hy4_requant IN.gguf OUT.gguf [--layers K] RULE...
//
// RULE is REGEX=TYPE: a tensor whose name the regex finds (std::regex_search) is decoded by ggml's to_float and quantized
// again by ggml_quantize_chunk to TYPE (a ggml type name: q8_0, nvfp4, bf16, iq2_xxs, ...; a type that needs an importance
// matrix gets a uniform one, as src/prefill/mmq_dense_parity.cpp quantizes); the first rule that matches wins; every other
// tensor is copied as it is.  --layers K keeps blocks 0 .. K-1 (block_count and the per-layer arrays cut to K; the
// Engram's modules must lie below K).  The values are the source's, re-rounded once: llama.cpp (the oracle) and strata-x
// read the same file, so they compare op for op on it - the file is a test fixture, not a model to use.
//
// Built against llama.cpp's ggml (the PR #32517 worktree's: it reads F8_E4M3, type 43), e.g.
//     g++ -O2 -std=c++17 -I$L/ggml/include moex/tools/hy4/hy4_requant.cpp -o $B/hy4_requant -L$B/bin -lggml-base \
//         -Wl,-rpath,$B/bin
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <regex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static ggml_type type_of(const std::string& n) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* nm = ggml_type_name((ggml_type) t);
        if (nm && n == nm && ggml_get_type_traits((ggml_type) t)->blck_size > 0) return (ggml_type) t;
    }
    std::fprintf(stderr, "hy4_requant: unknown ggml type %s\n", n.c_str());
    std::exit(2);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: hy4_requant IN.gguf OUT.gguf [--layers K] REGEX=TYPE ...\n");
        return 2;
    }
    const char* fin = argv[1];
    const char* fout = argv[2];
    int keep = -1;
    std::vector<std::pair<std::regex, ggml_type>> rules;
    std::vector<std::string> rule_text;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--layers" && i + 1 < argc) {
            keep = std::atoi(argv[++i]);
            continue;
        }
        const size_t eq = a.rfind('=');
        if (eq == std::string::npos) {
            std::fprintf(stderr, "hy4_requant: a rule is REGEX=TYPE, not %s\n", a.c_str());
            return 2;
        }
        rules.emplace_back(std::regex(a.substr(0, eq)), type_of(a.substr(eq + 1)));
        rule_text.push_back(a);
    }
    ggml_context* cin = nullptr;
    gguf_init_params ip{};
    ip.no_alloc = false;
    ip.ctx = &cin;
    gguf_context* in = gguf_init_from_file(fin, ip);
    if (!in) {
        std::fprintf(stderr, "hy4_requant: cannot read %s\n", fin);
        return 1;
    }
    const int64_t ka = gguf_find_key(in, "general.architecture");
    const std::string arch = ka >= 0 ? gguf_get_val_str(in, ka) : "";
    if (arch != "hy_v4_flash") {
        std::fprintf(stderr, "hy4_requant: %s is %s, not hy_v4_flash\n", fin, arch.c_str());
        return 1;
    }
    const std::string pre = arch + ".";
    gguf_context* out = gguf_init_empty();
    gguf_set_kv(out, in);
    const int64_t kb = gguf_find_key(in, (pre + "block_count").c_str());
    const uint32_t n_block = kb >= 0 ? gguf_get_val_u32(in, kb) : 0;
    if (keep >= 0) {
        if (keep < 1 || (uint32_t) keep > n_block) {
            std::fprintf(stderr, "hy4_requant: --layers %d: the file has %u blocks\n", keep, n_block);
            return 1;
        }
        gguf_set_val_u32(out, (pre + "block_count").c_str(), (uint32_t) keep);
        for (int64_t k = 0; k < gguf_get_n_kv(in); ++k) {   // the per-layer arrays (not the Engram's own keys)
            const std::string key = gguf_get_key(in, k);
            if (gguf_get_kv_type(in, k) != GGUF_TYPE_ARRAY || key.rfind(pre, 0) != 0 || key.rfind(pre + "ple.", 0) == 0) continue;
            if (gguf_get_arr_n(in, k) != n_block || gguf_get_arr_type(in, k) == GGUF_TYPE_STRING) continue;
            gguf_set_arr_data(out, key.c_str(), gguf_get_arr_type(in, k), gguf_get_arr_data(in, k), (size_t) keep);
        }
        const int64_t kl = gguf_find_key(in, (pre + "ple.layers").c_str());
        if (kl >= 0) {
            const size_t n = gguf_get_arr_n(in, kl);
            const gguf_type et = gguf_get_arr_type(in, kl);
            for (size_t i = 0; i < n; ++i) {
                const int64_t il = et == GGUF_TYPE_INT32 ? ((const int32_t*) gguf_get_arr_data(in, kl))[i]
                                                        : (int64_t) ((const uint32_t*) gguf_get_arr_data(in, kl))[i];
                if (il >= keep) {
                    std::fprintf(stderr, "hy4_requant: an Engram module at layer %lld, past --layers %d\n", (long long) il, keep);
                    return 1;
                }
            }
        }
    }
    gguf_set_val_str(out, "general.name", (std::string(gguf_get_val_str(in, gguf_find_key(in, "general.name"))) + " (moex hy4_requant)").c_str());
    // the tensors: first the bytes the re-quantized ones need, then each one decoded and quantized again
    struct Job {
        ggml_tensor* src;
        ggml_type to;
    };
    std::vector<Job> jobs;
    size_t need = 0;
    const int64_t nt = gguf_get_n_tensors(in);
    for (int64_t i = 0; i < nt; ++i) {
        ggml_tensor* t = ggml_get_tensor(cin, gguf_get_tensor_name(in, i));
        const std::string nm = t->name;
        if (keep >= 0 && nm.rfind("blk.", 0) == 0 && std::atoi(nm.c_str() + 4) >= keep) continue;
        ggml_type to = t->type;
        for (const auto& r : rules)
            if (std::regex_search(nm, r.first)) {
                to = r.second;
                break;
            }
        if (to != t->type) {
            if (t->ne[0] % ggml_blck_size(to)) {
                std::fprintf(stderr, "hy4_requant: %s: rows of %lld are not whole %s blocks\n", nm.c_str(), (long long) t->ne[0], ggml_type_name(to));
                return 1;
            }
            if (t->type != GGML_TYPE_F32 && !ggml_get_type_traits(t->type)->to_float) {
                std::fprintf(stderr, "hy4_requant: %s: ggml cannot decode %s\n", nm.c_str(), ggml_type_name(t->type));
                return 1;
            }
            need += ggml_row_size(to, t->ne[0]) * (size_t) ggml_nrows(t) + ggml_tensor_overhead() + 64;
        }
        jobs.push_back({t, to});
    }
    ggml_init_params gp{};
    gp.mem_size = need + ggml_tensor_overhead();
    gp.mem_buffer = nullptr;
    gp.no_alloc = false;
    ggml_context* cnew = ggml_init(gp);
    int n_changed = 0;
    std::vector<float> f32;
    for (const Job& j : jobs) {
        ggml_tensor* t = j.src;
        if (j.to == t->type) {
            gguf_add_tensor(out, t);
            continue;
        }
        const int64_t ne = ggml_nelements(t), ne0 = t->ne[0], nr = ggml_nrows(t);
        f32.resize((size_t) ne);
        if (t->type == GGML_TYPE_F32) std::memcpy(f32.data(), t->data, (size_t) ne * 4);
        else ggml_get_type_traits(t->type)->to_float(t->data, f32.data(), ne);
        ggml_tensor* n = ggml_new_tensor(cnew, j.to, ggml_n_dims(t), t->ne);
        ggml_set_name(n, t->name);
        if (j.to == GGML_TYPE_F32) {
            std::memcpy(n->data, f32.data(), (size_t) ne * 4);
        } else {
            std::vector<float> imat;
            if (ggml_quantize_requires_imatrix(j.to)) {
                ggml_quantize_init(j.to);
                imat.assign((size_t) ne0, 1.0f);
            }
            // rows in parallel (ggml_quantize_chunk writes rows [start / ne0, + nrows) of dst; the i-quants' searches are slow)
            const int nth = (int) std::min<int64_t>(16, std::max<int64_t>(1, nr / 64));
            std::vector<std::thread> th;
            for (int w = 0; w < nth; ++w) {
                const int64_t r0 = nr * w / nth, r1 = nr * (w + 1) / nth;
                th.emplace_back([&, r0, r1] {
                    ggml_quantize_chunk(j.to, f32.data(), n->data, r0 * ne0, r1 - r0, ne0, imat.empty() ? nullptr : imat.data());
                });
            }
            for (auto& x : th) x.join();
        }
        gguf_add_tensor(out, n);
        std::printf("  %-48s %-8s -> %s\n", t->name, ggml_type_name(t->type), ggml_type_name(j.to));
        ++n_changed;
    }
    if (!gguf_write_to_file(out, fout, false)) {
        std::fprintf(stderr, "hy4_requant: cannot write %s\n", fout);
        return 1;
    }
    std::printf("hy4_requant: %s: %zu tensors (%d re-quantized), %u -> %d blocks; rules:", fout, jobs.size(), n_changed, n_block,
                keep >= 0 ? keep : (int) n_block);
    for (const auto& r : rule_text) std::printf(" %s", r.c_str());
    std::printf("\n");
    gguf_free(out);
    gguf_free(in);
    ggml_free(cnew);
    ggml_free(cin);
    return 0;
}
