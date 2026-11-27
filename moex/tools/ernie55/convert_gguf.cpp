// moex/tools/ernie55/convert_gguf.cpp - ERNIE-5.5-Flash's FP8 release (Hugging Face safetensors) into a strata-x GGUF
// set, architecture "ernie55" (moex/docs/ERNIE55_PLAN.md §6).
//
//   convert_gguf --model DIR --out PREFIX [--types ROLE=TYPE[:LAYERS] ...] [--imatrix FILE] [--layers N] [--split-gb 48]
//                [--threads N] [--dry-run]
//
//   ROLE: gu (the routed experts' gate and up), d (their down); TYPE: a ggml type name (q8_0, q6_k, q5_k, q4_k, q3_k,
//   iq4_xs, iq4_nl, iq3_s, iq3_xxs, iq2_m, ...); LAYERS: a list like 1-6,40-42 (default: every MoE layer). Later --types
//   override earlier ones. Default: gu=q8_0, d=q8_0 (a near-lossless set, for checks).
//   --layers N: the first N layers only (a slice for the engine's checks; the head and its reduce kept).
//   --reuse SET.gguf (any file of an earlier set): every tensor the plan makes exactly as that set holds it (name, type, shape,
//   bytes; the routed experts only when the set records the same importance matrix file) is copied, not made again.
//   --fakequant (with --imatrix): no set; stdin "LAYER EXPERT GU_TYPE D_TYPE" lines, stdout that expert quantized in those
//   types and decoded (ernie55_expert_dq's format), for the reference model's checks of a recipe before it is written.
//   --measure OUT.tsv [--measure-types t1,t2,...] [--measure-experts K] [--measure-layers LIST] (with --imatrix): no set; per
//   MoE layer, role (gate / up / down) and type, the routed experts' quantization error weighted as the output sees it,
//   sum_rows sum_j imx_j (w - q(w))^2 / sum_rows sum_j imx_j w^2 (imx_j the column's mean squared input), over K experts
//   picked evenly along the layer's cumulative expert use (so their plain mean is the use-weighted mean), each quantized as
//   the set's own (ggml_quantize_chunk with the expert's importance weights) and decoded by ggml.
//   PREFIX-00001-of-0000K.gguf ...: llama.cpp's split layout (split.no / split.count / split.tensors.count in every file,
//   the model's keys in the first); tensors 4096-aligned, so every expert slice is whole pages (the loader's O_DIRECT reads).
//
// The tensors, ggml order (ne0 = the row length):
//   - The dense FP8 tensors -> BF16: e4m3 x a power-of-two block scale is a BF16 value exactly (checked, value by value);
//     strata-x's loader turns them back into its FP8 format (kDtFP8), bit for bit. BF16 tensors as they are; F32 kept.
//   - The norms, the hyper-connections, sinks, biases, the compressor's ape, the per-expert scales, MultiMax: F32.
//   - The routed experts stacked per layer: ffn_gate_exps / ffn_up_exps [2048, 2048, 512] (w1 / w3), ffn_down_exps
//     [2048, 2048, 512] (w2), each matrix dequantized exactly to f32 and quantized by ggml_quantize_chunk - llama-quantize's
//     call - with the importance matrix's per-column weights when one is given (an expert it never saw: none).
//   - Re-laid: mapping_proj / hc_head_fn transposed (stored [in, out]); the three alphas as scale[0..2]; the two routers
//     stacked [4096, 1024]; kv_b split per head into attn_k_b [192, 512, 64] (transposed) and attn_v_b [512, 256, 64]
//     (llama.cpp's MLA layout, Hy4's kernels').
//   - Not written (yet): DSpark's model.mtp.* (drafting only; a later step), the FlashIndexer add-on (off in the references).
//
// The importance matrix (--imatrix): "ERN55IMX", u32 version 1, u32 n_layer, u32 n_expert, u32 d_in, u32 d_ff, then per
// layer per expert f64 gate_up[d_in] (sum of squared latent inputs), f64 down[d_ff] (sum of squared SiTU-GLU outputs),
// u64 count - written by the reference model (moex/tools/ernie55/ref/imatrix.py).
//
// After writing, the set is re-read through strata's own GGUF reader (each file's header, alignment and offsets as the engine
// reads them) and each tensor's bytes compared with what was written (a hash a tensor).
#include "ggml.h"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/x/json_lite.hpp"
#include "strata/x/safetensors.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace st = strata::st;
namespace js = strata::json;

namespace {

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "convert_gguf: %s\n", m.c_str());
    std::exit(1);
}

constexpr uint64_t kAlign = 4096;

// ------------------------------------------------------------------------------------------------ e4m3 / bf16
float g_e4m3[256];

void init_e4m3() {
    for (int c = 0; c < 256; ++c) {
        const int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
        float v;
        if (e == 15 && m == 7) v = NAN;                       // (e4m3fn: one NaN code, no infinities)
        else if (e == 0) v = std::ldexp((float) m, -9);        // subnormal: m / 8 x 2^-6
        else v = std::ldexp(1.0f + m / 8.0f, e - 7);
        g_e4m3[c] = s ? -v : v;
    }
}

uint16_t bf16_exact(float f, bool& exact) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if (u & 0xFFFFu) exact = false;
    return (uint16_t) (u >> 16);
}

// ------------------------------------------------------------------------------------------------ the source
struct Src {
    st::Checkpoint ck;
    std::vector<int> fd;
    void open(const std::string& dir) {
        std::string err;
        if (!ck.open(dir, err, false)) die(err);
        fd.assign(ck.files().size(), -1);
        for (size_t f = 0; f < ck.files().size(); ++f) {
            fd[f] = ::open(ck.path((int) f).c_str(), O_RDONLY | O_CLOEXEC);
            if (fd[f] < 0) die(ck.path((int) f) + ": cannot open");
        }
    }
    const st::Tensor& t(const std::string& name) const {
        const st::Tensor* p = ck.find(name);
        if (!p) die("the release has no tensor " + name);
        return *p;
    }
    bool has(const std::string& name) const { return ck.find(name) != nullptr; }
    void bytes(const st::Tensor& t, void* dst) const {
        uint8_t* d = static_cast<uint8_t*>(dst);
        uint64_t off = t.offset, n = t.nbytes;
        while (n) {
            const ssize_t r = ::pread(fd[(size_t) t.file], d, (size_t) std::min<uint64_t>(n, 1u << 30), (off_t) off);
            if (r <= 0) die("a short read of a tensor");
            d += r;
            off += (uint64_t) r;
            n -= (uint64_t) r;
        }
    }
    /// a 2-D tensor as f32 [rows][cols] (FP8 dequantized with its _scale_inv; BF16 / F32 widened)
    std::vector<float> f32(const std::string& name, int64_t& rows, int64_t& cols) const {
        const st::Tensor& t = this->t(name);
        if (t.shape.size() == 1) {
            rows = 1;
            cols = t.shape[0];
        } else if (t.shape.size() == 2) {
            rows = t.shape[0];
            cols = t.shape[1];
        } else {
            die(name + ": not 1-D or 2-D");
        }
        std::vector<float> out((size_t) (rows * cols));
        std::vector<uint8_t> raw(t.nbytes);
        bytes(t, raw.data());
        if (t.dtype == st::DType::F8_E4M3) {
            const st::Tensor& s = this->t(name + "_scale_inv");
            const int64_t br = (rows + 127) / 128, bc = (cols + 127) / 128;
            if (s.dtype != st::DType::F32 || s.shape.size() != 2 || s.shape[0] != br || s.shape[1] != bc)
                die(name + "_scale_inv: not F32 [" + std::to_string(br) + ", " + std::to_string(bc) + "]");
            std::vector<float> sc((size_t) (br * bc));
            bytes(s, sc.data());
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t c = 0; c < cols; ++c)
                    out[(size_t) (r * cols + c)] = g_e4m3[raw[(size_t) (r * cols + c)]] * sc[(size_t) ((r / 128) * bc + c / 128)];
        } else if (t.dtype == st::DType::BF16) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(raw.data());
            for (size_t i = 0; i < out.size(); ++i) {
                const uint32_t u = (uint32_t) p[i] << 16;
                std::memcpy(&out[i], &u, 4);
            }
        } else if (t.dtype == st::DType::F32) {
            std::memcpy(out.data(), raw.data(), out.size() * 4);
        } else {
            die(name + ": dtype " + st::dtype_name(t.dtype));
        }
        return out;
    }
};

// ------------------------------------------------------------------------------------------------ the output tensors
struct OutT {
    std::string name;
    ggml_type type;
    std::vector<int64_t> ne;   ///< ggml order
    uint64_t nbytes = 0;
    int file = 0;
    uint64_t off = 0;          ///< in its file's data section
    std::function<void(uint8_t*)> fill;   ///< writes nbytes
    uint64_t hash = 0;
};

uint64_t fnv(const uint8_t* p, uint64_t n, uint64_t h = 1469598103934665603ull) {
    for (uint64_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

uint64_t row_bytes(ggml_type t, int64_t n) { return (uint64_t) ggml_row_size(t, n); }

ggml_type type_of(const std::string& s) {
    // (case aside: ggml names the k-quants q4_K, q6_K, ...; the usage line's q4_k is the same type)
    auto lower = [](std::string x) {
        for (char& ch : x) ch = (char) std::tolower((unsigned char) ch);
        return x;
    };
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char* n = ggml_type_name((ggml_type) i);
        if (n && lower(s) == lower(n)) return (ggml_type) i;
    }
    die("an unknown ggml type " + s);
}

// ------------------------------------------------------------------------------------------------ the GGUF writer
struct KV {
    std::string key;
    uint32_t type;   ///< GGUF value type
    std::vector<uint8_t> val;
};

enum : uint32_t { kU8 = 0, kI8 = 1, kU16 = 2, kI16 = 3, kU32 = 4, kI32 = 5, kF32 = 6, kBool = 7, kStr = 8, kArr = 9, kU64 = 10, kI64 = 11, kF64 = 12 };

template <class T> void put(std::vector<uint8_t>& b, T v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof v);
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
}

struct Meta {
    std::vector<KV> kv;
    void u32(const std::string& k, uint32_t v) { KV x{k, kU32, {}}; put(x.val, v); kv.push_back(x); }
    void i32(const std::string& k, int32_t v) { KV x{k, kI32, {}}; put(x.val, v); kv.push_back(x); }
    void u16(const std::string& k, uint16_t v) { KV x{k, kU16, {}}; put(x.val, v); kv.push_back(x); }
    void f32(const std::string& k, float v) { KV x{k, kF32, {}}; put(x.val, v); kv.push_back(x); }
    void b(const std::string& k, bool v) { KV x{k, kBool, {}}; x.val.push_back(v ? 1 : 0); kv.push_back(x); }
    void str(const std::string& k, const std::string& v) { KV x{k, kStr, {}}; put_str(x.val, v); kv.push_back(x); }
    void arr_i32(const std::string& k, const std::vector<int32_t>& v) {
        KV x{k, kArr, {}};
        put<uint32_t>(x.val, kI32);
        put<uint64_t>(x.val, v.size());
        for (int32_t e : v) put(x.val, e);
        kv.push_back(x);
    }
};

void write_all(int fd, const void* p, uint64_t n) {
    const uint8_t* c = static_cast<const uint8_t*>(p);
    while (n) {
        const ssize_t w = ::write(fd, c, (size_t) std::min<uint64_t>(n, 1u << 30));
        if (w <= 0) die("a short write");
        c += w;
        n -= (uint64_t) w;
    }
}

std::vector<uint8_t> header(const std::vector<KV>& kvs, const std::vector<OutT*>& ts) {
    std::vector<uint8_t> h;
    put<uint32_t>(h, 0x46554747u);   // "GGUF"
    put<uint32_t>(h, 3);
    put<uint64_t>(h, ts.size());
    put<uint64_t>(h, kvs.size());
    for (const KV& k : kvs) {
        put_str(h, k.key);
        put<uint32_t>(h, k.type);
        h.insert(h.end(), k.val.begin(), k.val.end());
    }
    for (const OutT* t : ts) {
        put_str(h, t->name);
        put<uint32_t>(h, (uint32_t) t->ne.size());
        for (int64_t d : t->ne) put<uint64_t>(h, (uint64_t) d);
        put<uint32_t>(h, (uint32_t) t->type);
        put<uint64_t>(h, t->off);
    }
    while (h.size() % kAlign) h.push_back(0);
    return h;
}

// ------------------------------------------------------------------------------------------------ the importance matrix
struct Imx {
    uint32_t n_layer = 0, n_expert = 0, d_in = 0, d_ff = 0;
    std::vector<double> gu, dn;
    std::vector<uint64_t> cnt;
    bool load(const std::string& p) {
        FILE* f = std::fopen(p.c_str(), "rb");
        if (!f) return false;
        char mg[8];
        uint32_t ver = 0;
        bool ok = std::fread(mg, 1, 8, f) == 8 && std::memcmp(mg, "ERN55IMX", 8) == 0 && std::fread(&ver, 4, 1, f) == 1 && ver == 1 &&
                  std::fread(&n_layer, 4, 1, f) == 1 && std::fread(&n_expert, 4, 1, f) == 1 && std::fread(&d_in, 4, 1, f) == 1 &&
                  std::fread(&d_ff, 4, 1, f) == 1;
        if (ok) {
            const size_t ne = (size_t) n_layer * n_expert;
            gu.resize(ne * d_in);
            dn.resize(ne * d_ff);
            cnt.resize(ne);
            for (size_t i = 0; ok && i < ne; ++i)
                ok = std::fread(&gu[i * d_in], 8, d_in, f) == d_in && std::fread(&dn[i * d_ff], 8, d_ff, f) == d_ff &&
                     std::fread(&cnt[i], 8, 1, f) == 1;
        }
        std::fclose(f);
        return ok;
    }
};

std::vector<int> parse_layers(const std::string& s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        const std::string part = s.substr(i, j - i);
        const size_t d = part.find('-');
        const int a = std::atoi(part.substr(0, d).c_str());
        const int b = d == std::string::npos ? a : std::atoi(part.substr(d + 1).c_str());
        for (int k = a; k <= b; ++k) v.push_back(k);
        i = j + 1;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, out, imx_path, reuse_path, measure_out, measure_types = "iq3_xxs,iq3_s,q3_k,iq4_xs,q4_k", measure_layers;
    int threads = (int) std::thread::hardware_concurrency(), n_layers_out = -1, measure_k = 16;
    double split_gb = 48.0;
    bool dry = false, fakequant = false;
    std::vector<std::string> types;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("a value is missing after " + a);
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--out") out = next();
        else if (a == "--types") types.push_back(next());
        else if (a == "--imatrix") imx_path = next();
        else if (a == "--layers") n_layers_out = std::atoi(next().c_str());
        else if (a == "--split-gb") split_gb = std::atof(next().c_str());
        else if (a == "--threads") threads = std::max(1, std::atoi(next().c_str()));
        else if (a == "--dry-run") dry = true;
        else if (a == "--reuse") reuse_path = next();
        else if (a == "--measure") measure_out = next();
        else if (a == "--fakequant") fakequant = true;
        else if (a == "--measure-types") measure_types = next();
        else if (a == "--measure-experts") measure_k = std::max(1, std::atoi(next().c_str()));
        else if (a == "--measure-layers") measure_layers = next();
        else die("unknown option " + a);
    }
    if ((!measure_out.empty() || fakequant) && out.empty()) out = "-";   // (--measure and --fakequant write no set)
    if (model.empty() || out.empty()) die("usage: --model DIR --out PREFIX [--types gu=TYPE[:LAYERS]] [--imatrix F] [--layers N]");
    init_e4m3();
    ggml_quantize_init(GGML_TYPE_IQ3_XXS);   // (the i-quants' grids, once, before the threads)

    // ---- the config
    std::string cfg_text;
    {
        FILE* f = std::fopen((model + "/config.json").c_str(), "rb");
        if (!f) die(model + "/config.json: cannot open");
        char buf[65536];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) cfg_text.append(buf, n);
        std::fclose(f);
    }
    js::Value cfg;
    std::string err;
    if (!js::parse(cfg_text.data(), cfg_text.size(), cfg, err)) die("config.json: " + err);
    auto ci = [&](const char* k) {
        const js::Value* v = cfg.find(k);
        if (!v || !v->is_num() || !v->is_int) die(std::string("config.json: no integer ") + k);
        return v->i;
    };
    auto cf = [&](const char* k) {
        const js::Value* v = cfg.find(k);
        if (!v || !v->is_num()) die(std::string("config.json: no number ") + k);
        return v->num;
    };
    const int n_layer = (int) ci("num_hidden_layers");
    const int64_t n = ci("hidden_size"), E = ci("n_routed_experts"), nf = ci("moe_intermediate_size"), lat = ci("moe_latent_size");
    std::vector<int32_t> ratios;
    for (const js::Value& v : cfg.find("compress_ratios")->arr) ratios.push_back((int32_t) v.i);
    if ((int) ratios.size() != n_layer) die("compress_ratios: not one a layer");
    const int nl = n_layers_out > 0 ? std::min(n_layers_out, n_layer) : n_layer;
    const int first_dense = (int) ci("first_k_dense_replace");

    // ---- the expert types
    std::map<int, ggml_type> t_gu, t_d;
    for (int il = first_dense; il < n_layer; ++il) t_gu[il] = t_d[il] = GGML_TYPE_Q8_0;
    for (const std::string& s : types) {
        const size_t eq = s.find('='), co = s.find(':');
        if (eq == std::string::npos) die("--types " + s + ": ROLE=TYPE[:LAYERS]");
        const std::string role = s.substr(0, eq), ty = s.substr(eq + 1, co == std::string::npos ? std::string::npos : co - eq - 1);
        const ggml_type t = type_of(ty);
        std::vector<int> ls;
        if (co == std::string::npos)
            for (int il = first_dense; il < n_layer; ++il) ls.push_back(il);
        else
            ls = parse_layers(s.substr(co + 1));
        for (int il : ls) {
            if (role == "gu") t_gu[il] = t;
            else if (role == "d") t_d[il] = t;
            else die("--types: the role is gu or d, not " + role);
        }
    }
    Imx imx;
    if (!imx_path.empty() && !imx.load(imx_path)) die(imx_path + ": not an ERN55IMX v1 file");
    if (!imx_path.empty() && (imx.n_layer != (uint32_t) n_layer || imx.n_expert != E || imx.d_in != lat || imx.d_ff != nf))
        die("the importance matrix's geometry is not the model's");
    for (int il = first_dense; il < nl; ++il)
        for (ggml_type t : {t_gu[il], t_d[il]})
            if (ggml_quantize_requires_imatrix(t) && imx_path.empty())
                die(std::string(ggml_type_name(t)) + " needs an importance matrix (--imatrix)");

    Src src;
    src.open(model);

    // ---- --measure: the routed experts' quantization error per layer, role and type (nothing written but the table)
    // ---- --fakequant: on stdin "LAYER EXPERT GU_TYPE D_TYPE" lines ("q" or EOF ends); on stdout that routed expert's gate,
    // up and down quantized as a set would hold them (ggml_quantize_chunk with the expert's importance weights) and decoded,
    // F32 row-major [out][in] each, after a 16-byte header {n_ff, n_lat, ok, 0} - ernie55_expert_dq's format, so the
    // reference model (ref/oracle_dump.py ExpertDQ) reads a recipe's experts without a set being written
    if (fakequant) {
        std::vector<float> y;
        std::vector<uint8_t> q;
        char line[256];
        while (std::fgets(line, sizeof line, stdin)) {
            if (line[0] == 'q') break;
            int il = -1;
            long long e = -1;
            char tg[64] = {0}, td[64] = {0};
            if (std::sscanf(line, "%d %lld %63s %63s", &il, &e, tg, td) != 4) continue;
            int32_t hdr[4] = {(int32_t) nf, (int32_t) lat, 0, 0};
            if (il < first_dense || il >= n_layer || e < 0 || e >= E) {
                std::fwrite(hdr, sizeof hdr, 1, stdout);
                std::fflush(stdout);
                continue;
            }
            hdr[2] = 1;
            std::fwrite(hdr, sizeof hdr, 1, stdout);
            const std::string m = "model.layers." + std::to_string(il) + ".block_sparse_moe.experts." + std::to_string(e) + ".";
            for (const char* w : {"w1", "w3", "w2"}) {
                const bool dn_role = std::string(w) == "w2";
                const ggml_type t = type_of(dn_role ? td : tg);
                int64_t rows, cols;
                std::vector<float> v = src.f32(m + w + ".weight", rows, cols);
                std::vector<float> imw((size_t) cols);
                const float* qw = nullptr;
                const size_t k = (size_t) il * imx.n_expert + (size_t) e;
                if (!imx.cnt.empty() && imx.cnt[k] > 0) {
                    const double* sm = dn_role ? &imx.dn[k * imx.d_ff] : &imx.gu[k * imx.d_in];
                    for (int64_t j = 0; j < cols; ++j) imw[(size_t) j] = (float) (sm[j] / (double) imx.cnt[k]);
                    qw = imw.data();
                }
                const size_t rb = row_bytes(t, cols);
                q.resize(rb * (size_t) rows);
                y.resize((size_t) (rows * cols));
                // the rows over the threads, as ggml_quantize_chunk splits nothing itself (each row's values are its own)
                const int nth = std::max(1, std::min<int>(threads, (int) rows));
                std::vector<std::thread> th;
                for (int wk = 0; wk < nth; ++wk)
                    th.emplace_back([&, wk] {
                        const int64_t r0 = rows * wk / nth, r1 = rows * (wk + 1) / nth;
                        if (r1 <= r0) return;
                        ggml_quantize_chunk(t, v.data() + r0 * cols, q.data() + (size_t) r0 * rb, 0, r1 - r0, cols, qw);
                        const auto* tr = ggml_get_type_traits(t);
                        for (int64_t r = r0; r < r1; ++r) tr->to_float(q.data() + (size_t) r * rb, y.data() + r * cols, cols);
                    });
                for (auto& x : th) x.join();
                std::fwrite(y.data(), sizeof(float), y.size(), stdout);
            }
            std::fflush(stdout);
        }
        return 0;
    }
    if (!measure_out.empty()) {
        if (imx.cnt.empty()) die("--measure needs --imatrix (the experts' use and their columns' weights)");
        std::vector<ggml_type> mts;
        for (size_t i = 0; i < measure_types.size();) {
            const size_t j = measure_types.find(',', i);
            const std::string ty = measure_types.substr(i, j == std::string::npos ? std::string::npos : j - i);
            if (!ty.empty()) mts.push_back(type_of(ty));
            if (j == std::string::npos) break;
            i = j + 1;
        }
        if (mts.empty()) die("--measure-types: no type");
        for (ggml_type t : mts) ggml_quantize_init(t);
        std::vector<int> mls;
        if (measure_layers.empty())
            for (int il = first_dense; il < n_layer; ++il) mls.push_back(il);
        else
            mls = parse_layers(measure_layers);
        FILE* fo = std::fopen(measure_out.c_str(), "w");
        if (!fo) die(measure_out + ": cannot write");
        std::fprintf(fo, "layer\trole\ttype\tbpw\trel_err\tsampled\n");
        const auto t0 = std::chrono::steady_clock::now();
        for (int il : mls) {
            if (il < first_dense || il >= n_layer) die("--measure-layers: " + std::to_string(il) + " is not a MoE layer");
            // the sample: measure_k positions spread evenly over the layer's cumulative expert use (the importance matrix's
            // row counts), so each pick stands for the same share of the routed rows and their plain mean is the use-weighted
            // mean (a hot expert can be picked more than once: its multiplicity)
            std::vector<double> cum((size_t) E);
            double tot = 0;
            for (int64_t e = 0; e < E; ++e) cum[(size_t) e] = tot += (double) imx.cnt[(size_t) il * imx.n_expert + (size_t) e];
            if (tot <= 0) die("layer " + std::to_string(il) + ": the importance matrix saw none of its experts");
            std::map<int64_t, int> mult;
            for (int k = 0; k < measure_k; ++k) {
                const double at = (k + 0.5) * tot / measure_k;
                const int64_t e = std::lower_bound(cum.begin(), cum.end(), at) - cum.begin();
                ++mult[std::min<int64_t>(e, E - 1)];
            }
            std::vector<int64_t> picks;
            for (const auto& [e, c] : mult) picks.push_back(e);
            for (const char* role : {"gate", "up", "down"}) {
                const std::string w = std::string(role) == "gate" ? "w1" : std::string(role) == "up" ? "w3" : "w2";
                const bool dn_role = std::string(role) == "down";
                const int64_t cols = dn_role ? nf : lat, rows = dn_role ? lat : nf;
                const std::string m = "model.layers." + std::to_string(il) + ".block_sparse_moe.";
                std::vector<std::vector<float>> vs(picks.size()), ws(picks.size());
                std::vector<double> sig(picks.size(), 0.0);
                std::vector<double> err(picks.size() * mts.size(), 0.0);
                std::atomic<size_t> next{0};
                auto read = [&]() {
                    for (size_t p; (p = next.fetch_add(1)) < picks.size();) {
                        int64_t r, c;
                        vs[p] = src.f32(m + "experts." + std::to_string(picks[p]) + "." + w + ".weight", r, c);
                        if (r != rows || c != cols) die(m + "experts: an expert of another shape");
                        const size_t k = (size_t) il * imx.n_expert + (size_t) picks[p];
                        const double* s = dn_role ? &imx.dn[k * imx.d_ff] : &imx.gu[k * imx.d_in];
                        ws[p].resize((size_t) cols);
                        for (int64_t j = 0; j < cols; ++j) ws[p][(size_t) j] = (float) (s[j] / (double) imx.cnt[k]);
                        double sg = 0;
                        for (int64_t i = 0; i < rows; ++i)
                            for (int64_t j = 0; j < cols; ++j) {
                                const double x = vs[p][(size_t) (i * cols + j)];
                                sg += (double) ws[p][(size_t) j] * x * x;
                            }
                        sig[p] = sg;
                    }
                };
                {
                    std::vector<std::thread> th;
                    for (int k = 0; k < std::min<int>(threads, (int) picks.size()); ++k) th.emplace_back(read);
                    for (auto& x : th) x.join();
                }
                next = 0;
                auto work = [&]() {   // one job: one picked expert in one type, quantized as the converter does, decoded by ggml
                    std::vector<uint8_t> q;
                    std::vector<float> y((size_t) cols);
                    for (size_t jb; (jb = next.fetch_add(1)) < picks.size() * mts.size();) {
                        const size_t p = jb / mts.size(), ti = jb % mts.size();
                        const ggml_type t = mts[ti];
                        const size_t rb = row_bytes(t, cols);
                        q.resize(rb * (size_t) rows);
                        const size_t got = ggml_quantize_chunk(t, vs[p].data(), q.data(), 0, rows, cols, ws[p].data());
                        if (got != rb * (size_t) rows) die("ggml_quantize_chunk wrote another size");
                        const auto* tr = ggml_get_type_traits(t);
                        double e2 = 0;
                        for (int64_t i = 0; i < rows; ++i) {
                            tr->to_float(q.data() + (size_t) i * rb, y.data(), cols);
                            const float* x = vs[p].data() + i * cols;
                            for (int64_t j = 0; j < cols; ++j) {
                                const double d = (double) x[j] - (double) y[(size_t) j];
                                e2 += (double) ws[p][(size_t) j] * d * d;
                            }
                        }
                        err[jb] = e2;
                    }
                };
                {
                    std::vector<std::thread> th;
                    for (int k = 0; k < threads; ++k) th.emplace_back(work);
                    for (auto& x : th) x.join();
                }
                for (size_t ti = 0; ti < mts.size(); ++ti) {
                    double rel = 0;
                    for (size_t p = 0; p < picks.size(); ++p)
                        rel += (double) mult[picks[p]] * (sig[p] > 0 ? err[p * mts.size() + ti] / sig[p] : 0.0);
                    rel /= (double) measure_k;
                    const double bpw = 8.0 * (double) row_bytes(mts[ti], cols) / (double) cols;
                    std::fprintf(fo, "%d\t%s\t%s\t%.4f\t%.6e\t%zu\n", il, role, ggml_type_name(mts[ti]), bpw, rel, picks.size());
                }
                std::fflush(fo);
            }
            std::fprintf(stderr, "convert_gguf: --measure layer %d done (%zu experts, %.0f s)\n", il, picks.size(),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        std::fclose(fo);
        return 0;
    }

    // ---- the plan of output tensors
    std::vector<std::unique_ptr<OutT>> plan;
    std::atomic<int> bf16_inexact{0};
    auto add = [&](const std::string& name, ggml_type type, std::vector<int64_t> ne, std::function<void(uint8_t*)> fill) {
        auto t = std::make_unique<OutT>();
        t->name = name;
        t->type = type;
        t->ne = ne;
        t->nbytes = row_bytes(type, ne[0]);
        for (size_t d = 1; d < ne.size(); ++d) t->nbytes *= (uint64_t) ne[d];
        t->fill = std::move(fill);
        plan.push_back(std::move(t));
    };
    // a 2-D [out, in] (or 1-D) tensor as BF16 / F32 in ggml order [in, out]; transposed: the stored [in, out]
    auto dense = [&](const std::string& src_name, const std::string& name, ggml_type type, bool transpose = false) {
        const st::Tensor& t = src.t(src_name);
        int64_t r = t.shape.size() == 2 ? t.shape[0] : 1, c = t.shape.size() == 2 ? t.shape[1] : t.shape[0];
        std::vector<int64_t> ne = t.shape.size() == 1 ? std::vector<int64_t>{c} : transpose ? std::vector<int64_t>{r, c} : std::vector<int64_t>{c, r};
        add(name, type, ne, [&src, &bf16_inexact, src_name, type, transpose](uint8_t* dst) {
            int64_t rows, cols;
            std::vector<float> v = src.f32(src_name, rows, cols);
            if (transpose) {
                std::vector<float> tv(v.size());
                for (int64_t i = 0; i < rows; ++i)
                    for (int64_t j = 0; j < cols; ++j) tv[(size_t) (j * rows + i)] = v[(size_t) (i * cols + j)];
                v.swap(tv);
            }
            if (type == GGML_TYPE_F32) {
                std::memcpy(dst, v.data(), v.size() * 4);
            } else {
                bool exact = true;
                uint16_t* d = reinterpret_cast<uint16_t*>(dst);
                for (size_t i = 0; i < v.size(); ++i) d[i] = bf16_exact(v[i], exact);
                if (!exact) ++bf16_inexact;
            }
        });
    };
    const ggml_type BF = GGML_TYPE_BF16, F = GGML_TYPE_F32;
    dense("model.embed_tokens.weight", "token_embd.weight", BF);
    for (int il = 0; il < nl; ++il) {
        const std::string p = "model.layers." + std::to_string(il) + ".", b = "blk." + std::to_string(il) + ".";
        dense(p + "input_layernorm.weight", b + "attn_norm.weight", F);
        dense(p + "post_attention_layernorm.weight", b + "ffn_norm.weight", F);
        for (const char* side : {"attn", "ffn"}) {
            const std::string hp = p + (std::string(side) == "attn" ? "self_attention_hyper_connection." : "mlp_hyper_connection.");
            dense(hp + "mapping_proj.weight", b + "hc_" + side + "_fn.weight", F, true);
            dense(hp + "bias", b + "hc_" + side + "_base.weight", F);
            add(b + "hc_" + side + "_scale.weight", F, {3}, [&src, hp](uint8_t* dst) {
                float* d = reinterpret_cast<float*>(dst);
                int64_t r, c;
                d[0] = src.f32(hp + "alpha_pre", r, c)[0];
                d[1] = src.f32(hp + "alpha_post", r, c)[0];
                d[2] = src.f32(hp + "alpha_res", r, c)[0];
            });
        }
        const std::string a = p + "self_attn.";
        dense(a + "gate_proj.weight", b + "attn_gate.weight", BF);
        dense(a + "vha_postmix_U", b + "attn_vha_u.weight", F);
        dense(a + "vha_postmix_V", b + "attn_vha_v.weight", F);
        if (ratios[(size_t) il] > 0) {   // HCA
            dense(a + "linear_q_down_proj.weight", b + "attn_q_a.weight", BF);
            dense(a + "q_layernorm.weight", b + "attn_q_a_norm.weight", F);
            dense(a + "linear_q_up_proj.weight", b + "attn_q_b.weight", BF);
            dense(a + "linear_kv_proj.weight", b + "attn_kv.weight", BF);
            dense(a + "kv_layernorm.weight", b + "attn_kv_a_norm.weight", F);
            dense(a + "core_attention.attn_sink", b + "attn_sinks.weight", F);
            dense(a + "linear_o_group_proj", b + "attn_output_a.weight", BF);
            dense(a + "o_proj.weight", b + "attn_output_b.weight", BF);
            const std::string cp = a + "core_attention.compressor.";
            dense(cp + "linear_wkv.weight", b + "attn_compressor_kv.weight", BF);
            dense(cp + "linear_wgate.weight", b + "attn_compressor_gate.weight", BF);
            dense(cp + "ape", b + "attn_compressor_ape.weight", F);
            dense(cp + "norm.weight", b + "attn_compressor_norm.weight", F);
        } else {   // MLA + DSA
            dense(a + "q_a_proj.weight", b + "attn_q_a.weight", BF);
            dense(a + "q_a_layernorm.weight", b + "attn_q_a_norm.weight", F);
            dense(a + "q_b_proj.weight", b + "attn_q_b.weight", BF);
            dense(a + "kv_a_proj_with_mqa.weight", b + "attn_kv_a_mqa.weight", BF);
            dense(a + "kv_a_layernorm.weight", b + "attn_kv_a_norm.weight", F);
            dense(a + "core_attention.softmax_offset", b + "attn_sinks.weight", F);
            dense(a + "o_proj.weight", b + "attn_output.weight", BF);
            const int64_t H = ci("hybrid_mla_num_attention_heads"), dn = ci("hybrid_mla_qk_nope_head_dim"),
                          dv = ci("hybrid_mla_v_head_dim"), L = ci("hybrid_mla_kv_lora_rank");
            const std::string kvb = a + "kv_b_proj.weight";
            auto kb = std::make_shared<std::vector<float>>();
            auto kb_load = [&src, kvb, kb]() {
                if (kb->empty()) {
                    int64_t r, c;
                    *kb = src.f32(kvb, r, c);
                }
            };
            add(b + "attn_k_b.weight", BF, {dn, L, H}, [=, &bf16_inexact](uint8_t* dst) {   // k_b[h][r][d] = kv_b[h (dn + dv) + d][r]
                kb_load();
                bool exact = true;
                uint16_t* d = reinterpret_cast<uint16_t*>(dst);
                for (int64_t h = 0; h < H; ++h)
                    for (int64_t r = 0; r < L; ++r)
                        for (int64_t k = 0; k < dn; ++k) *d++ = bf16_exact((*kb)[(size_t) ((h * (dn + dv) + k) * L + r)], exact);
                if (!exact) ++bf16_inexact;
            });
            add(b + "attn_v_b.weight", BF, {L, dv, H}, [=, &bf16_inexact](uint8_t* dst) {   // v_b[h][j][r] = kv_b[h (dn + dv) + dn + j][r]
                kb_load();
                bool exact = true;
                uint16_t* d = reinterpret_cast<uint16_t*>(dst);
                for (int64_t h = 0; h < H; ++h)
                    for (int64_t j = 0; j < dv; ++j)
                        for (int64_t r = 0; r < L; ++r) *d++ = bf16_exact((*kb)[(size_t) ((h * (dn + dv) + dn + j) * L + r)], exact);
                if (!exact) ++bf16_inexact;
                kb->clear();
                kb->shrink_to_fit();
            });
            const std::string ia = a + "core_attention.indexer.";
            dense(ia + "wq_b.weight", b + "indexer.attn_q_b.weight", BF);
            dense(ia + "wk.weight", b + "indexer.attn_k.weight", BF);
            dense(ia + "k_norm.weight", b + "indexer.k_norm.weight", F);
            dense(ia + "k_norm.bias", b + "indexer.k_norm.bias", F);
            dense(ia + "weights_proj.weight", b + "indexer.proj.weight", BF);
        }
        if (il < first_dense) {
            dense(p + "mlp.gate_proj.weight", b + "ffn_gate.weight", BF);
            dense(p + "mlp.up_proj.weight", b + "ffn_up.weight", BF);
            dense(p + "mlp.down_proj.weight", b + "ffn_down.weight", BF);
            continue;
        }
        const std::string m = p + "block_sparse_moe.";
        add(b + "ffn_gate_inp.weight", BF, {n, 2 * E}, [&src, m](uint8_t* dst) {   // the two routers stacked: rows 0..E-1, E..2E-1
            uint16_t* d = reinterpret_cast<uint16_t*>(dst);
            for (const char* g : {"gate.weight", "gate.weight_1"}) {
                const st::Tensor& t = src.t(m + g);
                if (t.dtype != st::DType::BF16) die(m + g + ": not BF16");
                src.bytes(t, d);
                d += t.nbytes / 2;
            }
        });
        dense(m + "e_score_correction_bias", b + "exp_probs_b.bias", F);
        dense(m + "gate.routed_scaling_factor_param", b + "ffn_exp_scale.weight", F);
        dense(m + "fc1_latent_proj.weight", b + "ffn_routed_down.weight", BF);
        dense(m + "fc2_latent_proj.weight", b + "ffn_routed_up.weight", BF);
        dense(m + "latent_norm.weight", b + "ffn_latent_norm.weight", F);
        dense(m + "shared_experts.w1.weight", b + "ffn_gate_shexp.weight", BF);
        dense(m + "shared_experts.w3.weight", b + "ffn_up_shexp.weight", BF);
        dense(m + "shared_experts.w2.weight", b + "ffn_down_shexp.weight", BF);
        for (const char* role : {"gate", "up", "down"}) {
            const std::string w = std::string(role) == "gate" ? "w1" : std::string(role) == "up" ? "w3" : "w2";
            const bool dn_role = std::string(role) == "down";
            const ggml_type t = dn_role ? t_d[il] : t_gu[il];
            const int64_t cols = dn_role ? nf : lat, rows = dn_role ? lat : nf;
            const uint64_t slice = row_bytes(t, cols) * (uint64_t) rows;
            add(b + "ffn_" + role + "_exps.weight", t, {cols, rows, E},
                [&src, &imx, m, w, il, t, cols, rows, slice, E, threads, dn_role](uint8_t* dst) {
                    std::atomic<int64_t> next{0};
                    auto work = [&]() {
                        std::vector<float> imw((size_t) cols);
                        for (int64_t e; (e = next.fetch_add(1)) < E;) {
                            int64_t r, c;
                            const std::vector<float> v = src.f32(m + "experts." + std::to_string(e) + "." + w + ".weight", r, c);
                            if (r != rows || c != cols) die(m + "experts: an expert of another shape");
                            const float* qw = nullptr;
                            if (!imx.cnt.empty()) {
                                const size_t k = (size_t) il * imx.n_expert + (size_t) e;
                                if (imx.cnt[k] > 0) {
                                    const double* s = dn_role ? &imx.dn[k * imx.d_ff] : &imx.gu[k * imx.d_in];
                                    for (int64_t j = 0; j < cols; ++j) imw[(size_t) j] = (float) (s[j] / (double) imx.cnt[k]);
                                    qw = imw.data();
                                }
                            }
                            const size_t got = ggml_quantize_chunk(t, v.data(), dst + (uint64_t) e * slice, 0, rows, cols, qw);
                            if (got != slice) die("ggml_quantize_chunk wrote another size");
                        }
                    };
                    std::vector<std::thread> th;
                    for (int k = 0; k < threads; ++k) th.emplace_back(work);
                    for (auto& x : th) x.join();
                });
        }
    }
    dense("model.mhc_contract.hc_head_fn", "output_hc_fn.weight", F, true);
    dense("model.mhc_contract.hc_head_base", "output_hc_base.weight", F);
    dense("model.mhc_contract.hc_head_scale", "output_hc_scale.weight", F);
    dense("model.norm.weight", "output_norm.weight", F);
    dense("lm_head.weight", "output.weight", BF);
    dense("model.lm_head.multimax_ranges", "output.multimax_ranges", F);
    dense("model.lm_head.multimax_ts", "output.multimax_ts", F);

    // ---- the metadata
    Meta md;
    const std::string A = "ernie55.";
    md.str("general.architecture", "ernie55");
    md.str("general.name", "ERNIE-5.5-Flash-Preview");
    md.u32("general.alignment", (uint32_t) kAlign);
    md.u32(A + "block_count", (uint32_t) nl);
    md.u32(A + "context_length", (uint32_t) ci("max_position_embeddings"));
    md.u32(A + "embedding_length", (uint32_t) n);
    md.u32(A + "feed_forward_length", (uint32_t) ci("intermediate_size"));
    md.u32(A + "leading_dense_block_count", (uint32_t) first_dense);
    md.u32(A + "attention.head_count", (uint32_t) ci("num_attention_heads"));
    md.u32(A + "attention.key_length", (uint32_t) ci("head_dim"));
    md.u32(A + "attention.value_length", (uint32_t) ci("v_head_dim"));
    md.u32(A + "rope.dimension_count", (uint32_t) ci("qk_rope_head_dim"));
    md.u32(A + "attention.q_lora_rank", (uint32_t) ci("q_lora_rank"));
    md.u32(A + "attention.sliding_window", (uint32_t) ci("sliding_window"));
    md.u32(A + "attention.output_group_count", (uint32_t) ci("o_groups"));
    md.u32(A + "attention.output_lora_rank", (uint32_t) ci("o_lora_rank"));
    md.arr_i32(A + "attention.compress_ratios", std::vector<int32_t>(ratios.begin(), ratios.begin() + nl));
    md.f32(A + "attention.compress_rope_freq_base", (float) cf("compress_rope_theta"));
    md.f32(A + "attention.layer_norm_rms_epsilon", (float) cf("rms_norm_eps"));
    md.u32(A + "attention.mla.head_count", (uint32_t) ci("hybrid_mla_num_attention_heads"));
    md.u32(A + "attention.mla.kv_lora_rank", (uint32_t) ci("hybrid_mla_kv_lora_rank"));
    md.u32(A + "attention.mla.q_lora_rank", (uint32_t) ci("hybrid_mla_q_lora_rank"));
    md.u32(A + "attention.mla.key_length_nope", (uint32_t) ci("hybrid_mla_qk_nope_head_dim"));
    md.u32(A + "attention.mla.rope_dimension_count", (uint32_t) ci("hybrid_mla_qk_rope_head_dim"));
    md.u32(A + "attention.mla.value_length", (uint32_t) ci("hybrid_mla_v_head_dim"));
    md.u32(A + "attention.indexer.head_count", (uint32_t) ci("index_n_heads"));
    md.u32(A + "attention.indexer.key_length", (uint32_t) ci("index_head_dim"));
    md.u32(A + "attention.indexer.top_k", (uint32_t) ci("index_topk"));
    md.f32(A + "rope.freq_base", (float) cf("rope_theta"));
    const js::Value* rs = cfg.find("rope_scaling");
    md.str(A + "rope.scaling.type", rs->str_or("type", rs->str_or("rope_type", "")));
    md.f32(A + "rope.scaling.factor", (float) rs->num_or("factor", 1.0));
    md.u32(A + "rope.scaling.original_context_length", (uint32_t) rs->int_or("original_max_position_embeddings", 0));
    md.f32(A + "rope.scaling.yarn_beta_fast", (float) rs->num_or("beta_fast", 32.0));
    md.f32(A + "rope.scaling.yarn_beta_slow", (float) rs->num_or("beta_slow", 1.0));
    md.u32(A + "hyper_connection.count", (uint32_t) ci("hc_mult"));
    md.u32(A + "hyper_connection.sinkhorn_iterations", (uint32_t) ci("hc_sinkhorn_iters"));
    md.f32(A + "hyper_connection.epsilon", (float) cf("hc_eps"));
    md.u32(A + "expert_count", (uint32_t) E);
    md.u32(A + "expert_used_count", (uint32_t) ci("num_experts_per_tok"));
    md.u32(A + "expert_feed_forward_length", (uint32_t) nf);
    md.u32(A + "expert_shared_count", (uint32_t) ci("n_shared_experts"));
    md.u32(A + "expert_latent_length", (uint32_t) lat);
    md.b(A + "expert_weights_norm", cfg.bool_or("norm_topk_prob", true));
    md.str(A + "expert_gating_func", "sigmoid2");   // sigmoid(W0 x) + sigmoid(W1 x), the two routers of ffn_gate_inp
    md.str(A + "activation", "situ_glu");
    md.f32(A + "activation.situ_beta", (float) cf("activation_situ_beta"));
    md.f32(A + "activation.situ_linear_beta", (float) cf("activation_situ_linear_beta"));
    md.u32("tokenizer.ggml.bos_token_id", (uint32_t) ci("bos_token_id"));
    md.u32("tokenizer.ggml.eos_token_id", 8);   // <|ernie:im_end|> (generation_config.json: eos [2, 8])
    md.u32("tokenizer.ggml.padding_token_id", (uint32_t) ci("pad_token_id"));
    {
        std::string desc;
        for (int il = first_dense; il < nl; ++il) desc += std::to_string(il) + ":" + ggml_type_name(t_gu[il]) + "/" + ggml_type_name(t_d[il]) + " ";
        md.str("ernie55.convert.expert_types", desc);
        // The importance matrix's file name only: a set's metadata must not carry a local path.
        md.str("ernie55.convert.imatrix", imx_path.empty() ? "none" : imx_path.substr(imx_path.find_last_of("/\\") + 1));
    }

    // ---- the split: files of at most split_gb, tensors in plan order, offsets 4096-aligned
    const uint64_t cap = (uint64_t) (split_gb * 1e9);
    std::vector<std::vector<OutT*>> files(1);
    uint64_t used = 0;
    for (auto& t : plan) {
        const uint64_t sz = (t->nbytes + kAlign - 1) / kAlign * kAlign;
        if (used > 0 && used + sz > cap) {
            files.emplace_back();
            used = 0;
        }
        t->file = (int) files.size() - 1;
        t->off = used;
        used += sz;
        files.back().push_back(t.get());
    }
    uint64_t total = 0;
    for (auto& t : plan) total += t->nbytes;
    std::fprintf(stderr, "convert_gguf: %zu tensors, %.2f GB, %zu files, %d layers%s\n", plan.size(), total / 1e9, files.size(), nl,
                 imx_path.empty() ? "" : ", with the importance matrix");
    // ---- --reuse SET: a tensor the plan makes exactly as an earlier set holds it (the same name, type, shape and bytes - and
    // for the routed experts the same importance matrix, by the file name the set records) is copied from it instead of being
    // made again; a tensor of another type is made. The copies are hashed and re-read like the rest.
    std::unique_ptr<strata::GgufModel> reuse;
    if (!reuse_path.empty()) {
        reuse = std::make_unique<strata::GgufModel>(strata::GgufModel::open(reuse_path));
        const strata::MetaValue* rv = reuse->meta().get("ernie55.convert.imatrix");
        const std::string want = imx_path.empty() ? "none" : imx_path.substr(imx_path.find_last_of("/\\") + 1);
        // (the published set's key reads "NAME: what it was made from" - moex/docs/ERNIE55_PLAN.md, the upload)
        const bool same_imx = rv && (rv->s == want || rv->s.rfind(want + ":", 0) == 0);
        int n_copy = 0, n_make = 0;
        uint64_t b_copy = 0;
        for (auto& t : plan) {
            size_t sh = 0;
            const strata::TensorInfo* ti = reuse->find(t->name, &sh);
            const bool exps = t->name.find("_exps") != std::string::npos;
            bool ok = ti && ti->type == (uint32_t) t->type && ti->shape.size() == t->ne.size() && strata::tensor_payload_bytes(*ti) == t->nbytes &&
                      reuse->in_bounds(*ti, sh) && (!exps || same_imx);
            for (size_t d = 0; ok && d < t->ne.size(); ++d) ok = ti->shape[d] == (uint64_t) t->ne[d];
            if (!ok) {
                ++n_make;
                continue;
            }
            const uint8_t* p = reuse->shard(sh).tensor_data(*ti);
            const uint64_t nb = t->nbytes;
            t->fill = [p, nb](uint8_t* dst) { std::memcpy(dst, p, nb); };
            ++n_copy;
            b_copy += nb;
        }
        std::fprintf(stderr, "convert_gguf: --reuse %s: %d tensors (%.2f GB) copied, %d made%s\n", reuse_path.c_str(), n_copy, b_copy / 1e9,
                     n_make, same_imx ? "" : " (its experts were quantized with another importance matrix: none of them copied)");
    }
    if (dry) {
        for (auto& t : plan)
            if (t->name.find("_exps") != std::string::npos)
                std::fprintf(stderr, "  %s %s %.3f GB\n", t->name.c_str(), ggml_type_name(t->type), t->nbytes / 1e9);
        return 0;
    }
    const int K = (int) files.size();
    auto fname = [&](int k) {
        char b[64];
        std::snprintf(b, sizeof b, "-%05d-of-%05d.gguf", k + 1, K);
        return out + b;
    };
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> buf;
    for (int k = 0; k < K; ++k) {
        std::vector<KV> kvs;
        if (k == 0) kvs = md.kv;
        Meta sm;
        // (2026-11-25) every file says its alignment: a reader takes GGUF's default 32 without it, and the data here starts
        // at the header padded to 4,096 - the later files of the first full set were read at the wrong offsets
        // (moex/tools/ernie55/fix_split_alignment.py patches such a set in place)
        if (k > 0) sm.u32("general.alignment", (uint32_t) kAlign);
        sm.u16("split.no", (uint16_t) k);
        sm.u16("split.count", (uint16_t) K);
        sm.i32("split.tensors.count", (int32_t) plan.size());
        kvs.insert(kvs.end(), sm.kv.begin(), sm.kv.end());
        const std::vector<uint8_t> h = header(kvs, files[(size_t) k]);
        const std::string path = fname(k) + ".tmp";
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0) die(path + ": cannot create");
        write_all(fd, h.data(), h.size());
        uint64_t at = 0;
        for (OutT* t : files[(size_t) k]) {
            if (t->off > at) {
                const std::vector<uint8_t> pad((size_t) (t->off - at), 0);
                write_all(fd, pad.data(), pad.size());
                at = t->off;
            }
            buf.assign(t->nbytes, 0);
            t->fill(buf.data());
            t->hash = fnv(buf.data(), t->nbytes);
            write_all(fd, buf.data(), t->nbytes);
            at += t->nbytes;
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (t->name.find("_exps") != std::string::npos)
                std::fprintf(stderr, "  %s %s %.2f GB (%.0f s)\n", t->name.c_str(), ggml_type_name(t->type), t->nbytes / 1e9, s);
        }
        if (::fsync(fd) != 0 || ::close(fd) != 0) die(path + ": cannot finish");
        if (std::rename(path.c_str(), fname(k).c_str()) != 0) die(fname(k) + ": cannot rename");
    }
    if (bf16_inexact.load()) die(std::to_string(bf16_inexact.load()) + " tensors were not exactly BF16 (FP8 x a power-of-two scale should be)");
    // ---- re-read and compare, through the engine's own GGUF reader (strata::GgufModel: each file's header, its alignment
    // and offsets as strata-x reads them - the converter's own layout would not show a header a reader takes otherwise)
    int bad = 0;
    {
        strata::GgufModel gm = strata::GgufModel::open(fname(0));
        if ((int) gm.size() != K) die("the set re-read has " + std::to_string(gm.size()) + " files, " + std::to_string(K) + " written");
        for (auto& t : plan) {
            size_t sh = 0;
            const strata::TensorInfo* ti = gm.find(t->name, &sh);
            if (!ti || (int) sh != t->file || strata::tensor_payload_bytes(*ti) != t->nbytes) {
                std::fprintf(stderr, "convert_gguf: %s is not where it was written in the re-read set\n", t->name.c_str());
                ++bad;
                continue;
            }
            if (fnv(gm.shard(sh).tensor_data(*ti), t->nbytes) != t->hash) {
                std::fprintf(stderr, "convert_gguf: %s differs on re-reading\n", t->name.c_str());
                ++bad;
            }
        }
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "convert_gguf: %s: %d files written in %.0f s; every tensor re-read %s\n", out.c_str(), K, s,
                 bad ? "WITH DIFFERENCES" : "the same");
    return bad ? 1 : 0;
}
