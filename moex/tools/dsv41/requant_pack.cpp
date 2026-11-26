// moex/tools/dsv41/requant_pack.cpp - DeepSeek-V4.1's routed experts requantized from the native pack's MXFP4 records into
// another ggml type per layer (the Q3_K / Q2_K mix of moex/docs/DSV41_PLAN.md section 6), written as a new pack.
//
//   requant_pack --src NATIVE_PACK --dst NEW_PACK --types q3_k:0-19,32 --types q2_k:20-31,33-39 [--imatrix FILE]
//                [--layers A-B] [--threads N]
//   (moex: --types GU/D:LAYERS gives the down matrices their own type, e.g. --types q2_k/q3_k:5-14 --types q2_k/mxfp4:0-1)
//
// Each record of the source (gate | up | down, ggml MXFP4, 18,800,640 bytes) is dequantized with ggml's own MXFP4 decoder and
// requantized with ggml_quantize_chunk - the call llama-quantize makes, with the imatrix's per-column importance when one is
// given - so every block is what llama.cpp would write from the same floats.  A part whose new type is MXFP4 is the source's
// own bytes, copied: the release's weights bit for bit (a pass through ggml's MXFP4 quantizer may pick other block scales).
// The new record is the blob [gate | up | down], padded to a 4 KiB multiple (the stride; O_DIRECT reads), at e * stride in
// NEW_PACK/experts/L<il>.bin.  NEW_PACK/experts.tsv lists "layer type stride file" for strata-x (it overrides the GGUF's
// deepseek41.expert_record.*); the type is "gu,d" where the down matrices' differs.
//
// The imatrix (optional): a little-endian file "DSV41IMX" u32 version 1, u32 n_layers, u32 n_expert, u32 n_embd, u32 n_ff,
// then per layer per expert: f32 gate_up[n_embd] (the sum of squared FFN inputs routed to the expert), f32 down[n_ff] (the sum
// of squared SwiGLU hidden values), u32 count - written by the engine's collection run.  An expert with no count is
// quantized without importance (ggml's plain rounding), as llama-quantize does for a tensor the imatrix misses.
//
// Nothing here touches a GPU; the CPU threads are --threads (default: all).
#include "ggml.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int64_t kEmbd = 5120, kFF = 2304, kExpert = 384, kLayers = 40;
constexpr uint64_t kSrcRecord = 18'800'640;   // MXFP4 [gate | up | down]

[[noreturn]] void die(const std::string& m) {
    fprintf(stderr, "requant_pack: %s\n", m.c_str());
    exit(1);
}

ggml_type parse_type(const std::string& s) {
    static const std::map<std::string, ggml_type> t = {{"q2_k", GGML_TYPE_Q2_K}, {"q3_k", GGML_TYPE_Q3_K}, {"q4_k", GGML_TYPE_Q4_K},
                                                       {"q5_k", GGML_TYPE_Q5_K}, {"q6_k", GGML_TYPE_Q6_K}, {"q8_0", GGML_TYPE_Q8_0},
                                                       {"iq2_xxs", GGML_TYPE_IQ2_XXS}, {"iq2_xs", GGML_TYPE_IQ2_XS}, {"iq2_s", GGML_TYPE_IQ2_S},
                                                       {"iq3_xxs", GGML_TYPE_IQ3_XXS}, {"iq3_s", GGML_TYPE_IQ3_S}, {"mxfp4", GGML_TYPE_MXFP4}};
    auto it = t.find(s);
    if (it == t.end()) die("unknown type " + s);
    return it->second;
}

/// "a-b,c,d-e" -> the layers
std::vector<int> parse_layers(const std::string& s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        const std::string part = s.substr(i, j - i);
        const size_t d = part.find('-');
        const int a = std::stoi(part.substr(0, d)), b = d == std::string::npos ? a : std::stoi(part.substr(d + 1));
        for (int l = a; l <= b; ++l) out.push_back(l);
        i = j + 1;
    }
    return out;
}

struct Imatrix {
    uint32_t n_layers = 0, n_expert = 0, n_embd = 0, n_ff = 0;
    std::vector<float> gu, dn;      // [layer][expert][n_embd], [layer][expert][n_ff]
    std::vector<uint32_t> count;    // [layer][expert]
    const float* gate_up(int il, int e) const { return count[(size_t) il * n_expert + e] ? &gu[((size_t) il * n_expert + e) * n_embd] : nullptr; }
    const float* down(int il, int e) const { return count[(size_t) il * n_expert + e] ? &dn[((size_t) il * n_expert + e) * n_ff] : nullptr; }
};

Imatrix load_imatrix(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) die("cannot open " + path);
    char magic[8];
    uint32_t ver = 0;
    Imatrix m;
    auto rd = [&](void* p, size_t n) {
        if (fread(p, 1, n, f) != n) die("short imatrix " + path);
    };
    rd(magic, 8);
    rd(&ver, 4);
    if (memcmp(magic, "DSV41IMX", 8) != 0 || ver != 1) die("not a DSV41IMX v1 imatrix: " + path);
    rd(&m.n_layers, 4);
    rd(&m.n_expert, 4);
    rd(&m.n_embd, 4);
    rd(&m.n_ff, 4);
    if (m.n_layers != kLayers || m.n_expert != kExpert || m.n_embd != kEmbd || m.n_ff != kFF) die("imatrix geometry differs: " + path);
    const size_t E = (size_t) m.n_layers * m.n_expert;
    m.gu.resize(E * m.n_embd);
    m.dn.resize(E * m.n_ff);
    m.count.resize(E);
    for (size_t k = 0; k < E; ++k) {
        rd(&m.gu[k * m.n_embd], 4 * m.n_embd);
        rd(&m.dn[k * m.n_ff], 4 * m.n_ff);
        rd(&m.count[k], 4);
    }
    fclose(f);
    return m;
}

/// The new blob's layout, gate and up of type tgu and down of type td: [gate (kFF rows of kEmbd) | up | down (kEmbd rows of
/// kFF)], each part's size.
struct Layout {
    size_t gu = 0, d = 0, blob = 0, stride = 0;
};
Layout layout_of(ggml_type tgu, ggml_type td) {
    Layout L;
    L.gu = ggml_row_size(tgu, kEmbd) * kFF;
    L.d = ggml_row_size(td, kFF) * kEmbd;
    L.blob = 2 * L.gu + L.d;
    L.stride = (L.blob + 4095) / 4096 * 4096;
    return L;
}

}  // namespace

int main(int argc, char** argv) {
    std::string src, dst, imx_path, layer_sel;
    std::map<int, std::pair<ggml_type, ggml_type>> type_of;   // layer -> (gate / up, down)
    int threads = (int) std::thread::hardware_concurrency();
    int n_experts = (int) kExpert;   // --experts N: only the first N (a test)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value after " + a);
            return argv[++i];
        };
        if (a == "--src") src = next();
        else if (a == "--dst") dst = next();
        else if (a == "--imatrix") imx_path = next();
        else if (a == "--layers") layer_sel = next();
        else if (a == "--threads") threads = std::max(1, std::stoi(next()));
        else if (a == "--experts") n_experts = std::max(1, std::min((int) kExpert, std::stoi(next())));
        else if (a == "--types") {
            const std::string v = next();
            const size_t c = v.find(':');
            if (c == std::string::npos) die("--types TYPE[/DOWN_TYPE]:LAYERS");
            const std::string ts = v.substr(0, c);
            const size_t sl = ts.find('/');
            const ggml_type tgu = parse_type(ts.substr(0, sl)), td = sl == std::string::npos ? tgu : parse_type(ts.substr(sl + 1));
            for (int l : parse_layers(v.substr(c + 1))) type_of[l] = {tgu, td};
        } else die("unknown argument " + a);
    }
    if (src.empty() || dst.empty() || type_of.empty())
        die("usage: --src DIR --dst DIR --types TYPE[/DOWN_TYPE]:LAYERS [...] [--imatrix F] [--layers A-B]");
    std::vector<int> layers;
    if (layer_sel.empty())
        for (int l = 0; l < kLayers; ++l) layers.push_back(l);
    else layers = parse_layers(layer_sel);
    for (int l : layers)
        if (!type_of.count(l)) die("no --types entry for layer " + std::to_string(l));
    Imatrix imx;
    if (!imx_path.empty()) imx = load_imatrix(imx_path);
    if (system(("mkdir -p '" + dst + "/experts'").c_str()) != 0) die("cannot create " + dst + "/experts");
    ggml_quantize_init(GGML_TYPE_Q2_K);
    for (const auto& [l, t] : type_of) {
        ggml_quantize_init(t.first);
        ggml_quantize_init(t.second);
    }
    const ggml_type_traits* mx = ggml_get_type_traits(GGML_TYPE_MXFP4);
    if (!mx->to_float) die("ggml has no MXFP4 decoder");
    const size_t gu_src = ggml_row_size(GGML_TYPE_MXFP4, kEmbd) * kFF, d_src = ggml_row_size(GGML_TYPE_MXFP4, kFF) * kEmbd;
    if (2 * gu_src + d_src != kSrcRecord) die("the MXFP4 record size does not match ggml's rows");

    // experts.tsv: every layer this run writes (a later run over other layers appends its own lines)
    FILE* tsv = fopen((dst + "/experts.tsv").c_str(), "a");
    if (!tsv) die("cannot write " + dst + "/experts.tsv");
    for (int il : layers) {
        const ggml_type tgu = type_of[il].first, td = type_of[il].second;
        const Layout L = layout_of(tgu, td);
        for (ggml_type t : {tgu, td})
            if (t != GGML_TYPE_MXFP4 && ggml_quantize_requires_imatrix(t) && imx_path.empty())
                die(std::string(ggml_type_name(t)) + " needs an imatrix");
        const std::string tname = tgu == td ? std::string(ggml_type_name(tgu)) : std::string(ggml_type_name(tgu)) + "/" + ggml_type_name(td);
        const std::string in = src + "/experts/L" + (il < 10 ? "0" : "") + std::to_string(il) + ".bin";
        const std::string rel = "experts/L" + std::string(il < 10 ? "0" : "") + std::to_string(il) + ".bin";
        const std::string out = dst + "/" + rel;
        const int fi = open(in.c_str(), O_RDONLY);
        if (fi < 0) die("cannot open " + in);
        const int fo = open((out + ".part").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fo < 0) die("cannot create " + out + ".part");
        const auto t0 = std::chrono::steady_clock::now();
        std::atomic<int> next_e{0};
        std::atomic<bool> bad{false};
        std::vector<std::thread> th;
        for (int w = 0; w < threads; ++w)
            th.emplace_back([&]() {
                std::vector<uint8_t> rec(kSrcRecord), blob(L.stride, 0);
                std::vector<float> f((size_t) kFF * kEmbd);
                for (int e; !bad && (e = next_e.fetch_add(1)) < n_experts;) {
                    if (pread(fi, rec.data(), kSrcRecord, (off_t) ((uint64_t) e * kSrcRecord)) != (ssize_t) kSrcRecord) {
                        bad = true;
                        return;
                    }
                    const float* wgu = imx_path.empty() ? nullptr : imx.gate_up(il, e);
                    const float* wd = imx_path.empty() ? nullptr : imx.down(il, e);
                    // gate, up: kFF rows of kEmbd; down: kEmbd rows of kFF
                    const uint8_t* parts[3] = {rec.data(), rec.data() + gu_src, rec.data() + 2 * gu_src};
                    const size_t outs[3] = {0, L.gu, 2 * L.gu};
                    for (int p = 0; p < 3; ++p) {
                        const int64_t rows = p < 2 ? kFF : kEmbd, cols = p < 2 ? kEmbd : kFF;
                        const ggml_type t = p < 2 ? tgu : td;
                        if (t == GGML_TYPE_MXFP4) {   // the release's own bytes (the same rows: ggml's MXFP4 is the source's)
                            memcpy(blob.data() + outs[p], parts[p], p < 2 ? gu_src : d_src);
                            continue;
                        }
                        mx->to_float(parts[p], f.data(), rows * cols);
                        ggml_quantize_chunk(t, f.data(), blob.data() + outs[p], 0, rows, cols, p < 2 ? wgu : wd);
                    }
                    if (pwrite(fo, blob.data(), L.stride, (off_t) ((uint64_t) e * L.stride)) != (ssize_t) L.stride) {
                        bad = true;
                        return;
                    }
                }
            });
        for (auto& x : th) x.join();
        posix_fadvise(fi, 0, 0, POSIX_FADV_DONTNEED);   // a 7 GB layer read once: out of the page cache (the desktop's)
        close(fi);
        if (bad || fsync(fo) != 0) die("reading or writing layer " + std::to_string(il));
        posix_fadvise(fo, 0, 0, POSIX_FADV_DONTNEED);   // ... and its new file, once it is on the disk
        if (close(fo) != 0) die("writing layer " + std::to_string(il));
        if (rename((out + ".part").c_str(), out.c_str()) != 0) die("cannot rename " + out);
        if (tgu == td) fprintf(tsv, "%d\t%d\t%zu\t%s\n", il, (int) tgu, L.stride, rel.c_str());
        else fprintf(tsv, "%d\t%d,%d\t%zu\t%s\n", il, (int) tgu, (int) td, L.stride, rel.c_str());
        fflush(tsv);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("layer %2d: %s, %zu-byte records (blob %zu), %d experts, %.2f GB in %.1f s%s\n", il, tname.c_str(), L.stride, L.blob,
               n_experts, (double) L.stride * n_experts / 1e9, s, imx_path.empty() ? "" : ", imatrix");
        fflush(stdout);
    }
    fclose(tsv);
    return 0;
}
