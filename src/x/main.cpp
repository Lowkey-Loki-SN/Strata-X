// src/x/main.cpp - strata-x, the fork's engine (moex/README.md; strata-glm until 2026-11-26): its command line and
// --serve.  It was first written for GLM-5.3-Flash (glm5next); version 1, as first written:
//
//     strata-x -m model.gguf (-t IDS | --tokens-file F) [-n N] [--ctx N] [--threads N] [--no-pin] [--device D]
//                [-o DIR] [--match RE] [--oracle DIR [--tol T]] [--print-ids]
//
// Feeds the prompt one token at a time (version 1 has no batched prefill), then generates N tokens greedily and
// prints their ids and the speed.  -o dumps every op output in llama_dump's format; --oracle checks every op on
// llama.cpp's own inputs against a llama_dump of the same tokens (moex/ref/recorder.hpp) and exits 1 when one
// is over --tol.  Tokenize / detokenize with moex/build/llama_dump's vocabulary (the ids are llama.cpp's).
#include "strata/x/engine.hpp"
#include "strata/x/serve.hpp"
#include "strata/artifact/gguf_split.hpp"

#include "recorder.hpp"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

extern char** environ;

/// moex (DSV41_BACKLOG #53): what a saved conversation must have been made by to be put back (EngineOptions::conv_identity) -
/// this binary, every argument in order, the environment's STRATA_* / CUDA_VISIBLE_DEVICES / CUDA_DEVICE_ORDER, and the
/// model files (device, inode, size, time): another build or configuration never reads another's state
static std::string conv_identity(int argc, char** argv, const std::string& model, const std::string& pack) {
    auto stamp = [](const std::string& path) {
        struct stat sb {};
        if (path.empty() || ::stat(path.c_str(), &sb) != 0) return std::string("-");
        return std::to_string((long long) sb.st_dev) + ":" + std::to_string((long long) sb.st_ino) + ":" +
               std::to_string((long long) sb.st_size) + ":" + std::to_string((long long) sb.st_mtim.tv_sec) + "." +
               std::to_string((long long) sb.st_mtim.tv_nsec);
    };
    std::string s = "strata-x conversations 1";
    s += "\x1f" "exe " + stamp("/proc/self/exe");
    s += "\x1f" "model " + stamp(model);
    // (moex: a GGUF set's other files too - its experts, engram table and dense weights live there)
    if (const std::vector<std::string> set = strata::gguf_split_paths(model, false); set.size() > 1)
        for (size_t i = 1; i < set.size(); ++i) s += "\x1f" "model " + stamp(set[i]);
    if (!pack.empty()) s += "\x1f" "pack " + stamp(pack + "/experts.tsv");
    // the GGUF file the engram tables come from (STRATA_DSV41_ENGRAM_GGUF; its path is in the environment below)
    if (const char* g = std::getenv("STRATA_DSV41_ENGRAM_GGUF"); g && *g) s += "\x1f" "engram gguf " + stamp(g);
    // the expert GGUF set's files (--experts-gguf names one of them; its path is in the arguments below)
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--experts-gguf")) {
            for (const std::string& p : strata::gguf_split_paths(argv[i + 1], false)) s += "\x1f" "experts gguf " + stamp(p);
        }
    // (moex: the DSpark drafter's own GGUF, --draft-dspark FILE - its blocks' state is saved with a conversation)
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--draft-dspark") && std::strcmp(argv[i + 1], "builtin")) s += "\x1f" "dspark gguf " + stamp(argv[i + 1]);
    for (int i = 1; i < argc; ++i) {
        // the server writes the image encoder's manifest into a new temp folder at every start (serve/server.py
        // EngineVision.work_dir), so its path differs from one start to the next while what it says does not - with the
        // path in the identity, every start found the last one's conversations "another engine's" and deleted them.  The
        // identity takes the manifest's lines and the stamps of the weight files they name (read as Dsv41Vision::load
        // reads them: five fields, then the file) in place of its path
        if (!std::strcmp(argv[i], "--vision-weights") && i + 1 < argc) {
            s += "\x1f" "--vision-weights";
            {   // moex: Hy4-flash's is the mmproj GGUF itself (a file of its own, not the server's temp manifest): its stamp
                std::ifstream g(argv[i + 1], std::ios::binary);
                char magic[4] = {0, 0, 0, 0};
                if (g && g.read(magic, 4) && !std::memcmp(magic, "GGUF", 4)) {
                    s += "\x1f" "vision gguf " + stamp(argv[++i]);
                    continue;
                }
            }
            std::ifstream f(argv[++i]);
            if (!f) s += "\x1f" "-";
            std::vector<std::string> files;
            std::string line;
            while (f && std::getline(f, line)) {
                s += "\x1f" + line;
                std::istringstream ls(line);
                std::string tag, name, off, bytes, shape, path;
                if ((ls >> tag >> name >> off >> bytes >> shape) && tag == "T" && std::getline(ls, path)) {
                    path.erase(0, path.find_first_not_of(' '));
                    files.push_back(path);
                }
            }
            std::sort(files.begin(), files.end());
            files.erase(std::unique(files.begin(), files.end()), files.end());
            for (const std::string& p : files) s += "\x1f" "vision file " + stamp(p);
            continue;
        }
        s += std::string("\x1f") + argv[i];
    }
    std::vector<std::string> env;
    for (char** e = environ; e && *e; ++e)
        if (!std::strncmp(*e, "STRATA_", 7) || !std::strncmp(*e, "CUDA_VISIBLE_DEVICES=", 21) || !std::strncmp(*e, "CUDA_DEVICE_ORDER=", 18))
            env.emplace_back(*e);
    std::sort(env.begin(), env.end());
    for (const auto& e : env) s += "\x1f" + e;
    return s;
}

static std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty()) out.push_back((int32_t) std::stol(item));
    return out;
}

/// moex: STRATA_X_QUIET=1 (the server's "diagnostics": false) - stderr becomes a stream that drops what is written to it
/// inside the process (glibc's fopencookie; stderr is an assignable variable there), so no diagnostic line is sent anywhere:
/// no file, no pipe, not /dev/null
static void quiet_stderr() {
#if defined(__GLIBC__)
    cookie_io_functions_t io{};
    io.write = [](void*, const char*, size_t n) -> ssize_t { return (ssize_t) n; };
    if (FILE* f = fopencookie(nullptr, "w", io)) stderr = f;
#endif
}

static void usage() {
    fprintf(stderr,
            "usage: strata-x -m model.gguf (-t IDS | --tokens-file F) [-n N] [--ctx N] [--threads N] [--no-pin]\n"
            "                  [--device D] [-o DIR] [--match RE] [--oracle DIR [--tol T]] [--print-ids]\n"
            "                  [--expert-slots N (-1 auto, 0 none)] [--vram-reserve-mib M] [--expert-profile F]\n"
            "                  [--no-adaptive] [--experts-in-place | --experts-in-ram] [--pack DIR] [--mtp | --spec N (window, 2..8)] [--spec-min-p P] [--coupled-draft]\n"
            "                  [--gpus 0,1] [--split K (trunk layers on the first GPU; 0 = one GPU)]\n"
            "                  [--vram-reserve-later-mib M] [--prefill-chunk N (0 auto up to 8192; -1 decode windows)]\n"
            "                  [--prefill-ring N (0: Strata's byte budget)] [--prefill-ring-frac F (the full ring, a layer's share)]\n"
            "                  [--prefill-sub N] [--short-read N]\n"
            "                  [--pin-gguf (in-place mode: register the mapping)]\n"
            "                  [--runs ring:chunk[:sub],... (one loaded model, one run per entry; reports each)]\n"
            "                  [--no-graphs (decode windows eagerly: no CUDA graphs, no doorbell)]\n"
            "                  [--serve --max-context N --eos IDS --turn-tokens IDS --prompt-cache 6 --prompt-cache-every 16384\n"
            "                   --prompt-cache-root 2048 --prompt-cache-deep 4 --image-pad ID]  (Strata's engine protocol on\n"
            "                   stdin/stdout; --image-pad: GENI; --prompt-cache-deep: the checkpoints at the multiples of\n"
            "                   --prompt-cache-every, kept apart from the turns' --prompt-cache - 0: every N fresh tokens among them)\n"
            "                  [--vision-weights MANIFEST (serve: VENC - DeepSeek-V4.1's image encoder on the first GPU)]\n"
            "                  [--conversation-cache-mib N --conversation-cache-slots 4 --conversation-cache-min-tokens 8192\n"
            "                   --conversation-cache-min-free-mib 8192]  (serve, DeepSeek-V4 / V4.1: a conversation a request\n"
            "                   leaves is parked in RAM and put back when continued - DSV41_PLAN §57; 0: off)\n"
            "                  [--conversation-dir DIR --conversation-dir-mib 8192]  (serve, with the above: the live and parked\n"
            "                   conversations saved there at a clean end and put back by a later engine started the same way)\n"
            "                  [--adapt-every 4] [--adapt-swaps 96] [--adapt-decay 0.7] (the adaptive tier, Strata's names)\n");
}

// ---- THE PARITY CHECK AGAINST llama.cpp (--kld FILE): llama-perplexity --kl-divergence-base writes, per chunk of n_ctx tokens
// evaluated from an empty context, the log-probabilities of positions [n_ctx/2, n_ctx - 1) (uint16 with a per-row scale and
// offset).  Here the same chunks run through this engine - the first half through the prompt path (or windows), the scored
// half through decode windows - and every row is compared with llama.cpp's own function (perplexity.cpp log_softmax, the
// kl_divergence_result accumulation), so the numbers mean exactly what llama-perplexity --kl-divergence prints.
namespace {
struct KldAcc {
    double sum_nll = 0, sum_nll_base = 0, sum_kld = 0, sum_kld2 = 0, sum_p_diff = 0, sum_p_diff2 = 0;
    float max_p_diff = 0;
    size_t n_same_top = 0, count = 0;
    std::vector<float> klds, pdiffs;
};
void kld_row(const float* logits, int n_vocab, const uint16_t* base_log_prob, int tok, KldAcc& k) {
    float max_logit = logits[0];
    int imax = 0;
    for (int i = 1; i < n_vocab; ++i)
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            imax = i;
        }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum_exp += expf(logits[i] - max_logit);
    const float log_sum_exp = (float) log(sum_exp);
    const float* d = (const float*) base_log_prob;
    const float scale = d[0], min_log_prob = d[1];
    base_log_prob += 4;
    const float nll = max_logit + log_sum_exp - logits[tok];
    k.sum_nll += nll;
    const float nll_base = -(scale * base_log_prob[tok] + min_log_prob);
    k.sum_nll_base += nll_base;
    max_logit += log_sum_exp;
    double sum = 0;
    int imax_base = -1;
    float p_log_base_max = 0;
    for (int i = 0; i < n_vocab; ++i) {
        const float p_log_base = scale * base_log_prob[i] + min_log_prob;
        if (i == 0 || p_log_base > p_log_base_max) {
            p_log_base_max = p_log_base;
            imax_base = i;
        }
        if (p_log_base > -16.f) {
            const float p_base = expf(p_log_base);
            sum += p_base * (p_log_base - logits[i] + max_logit);
        }
    }
    k.sum_kld += sum;
    k.sum_kld2 += sum * sum;
    ++k.count;
    if (imax == imax_base) ++k.n_same_top;
    const float p_diff = expf(-nll) - expf(-nll_base);
    k.sum_p_diff += p_diff;
    k.sum_p_diff2 += (double) p_diff * p_diff;
    k.max_p_diff = std::max(k.max_p_diff, std::fabs(p_diff));
    k.klds.push_back((float) sum);
    k.pdiffs.push_back(p_diff);
}

/// --kld-save: one row of llama-perplexity's --kl-divergence-base format from this engine's logits (the inverse of
/// kld_row's reading: scale and the min log-probability as two floats, then a uint16 per vocabulary entry).
void kld_encode(const float* logits, int n_vocab, uint16_t* out) {
    float max_logit = logits[0], min_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
        min_logit = std::min(min_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum_exp += expf(logits[i] - max_logit);
    const float log_sum_exp = (float) log(sum_exp);
    min_logit = std::max(min_logit, max_logit - 16.0f);
    const float scale = std::max(1e-12f, (max_logit - min_logit) / 65535.0f);
    float* d = (float*) out;
    d[0] = scale;
    d[1] = min_logit - max_logit - log_sum_exp;
    for (int i = 0; i < n_vocab; ++i) {
        const float q = std::nearbyint((std::max(logits[i], min_logit) - min_logit) / scale);
        out[4 + i] = (uint16_t) std::min(65535.0f, std::max(0.0f, q));
    }
}

/// `save` empty: compare with the file (llama.cpp's base, or one --kld-save wrote).  `save` set: the file only gives the
/// chunks' tokens, and this engine's own rows are written there - a base of this engine, so a change of build or setting is
/// measured against this one (and a second run against it shows the run-to-run floor).  `win`: the decode windows' rows (8 as
/// before; 1 scores every position alone, as a plain decode does).
int run_kld(strata::x::Engine& eng, const std::string& path, const std::string& mode, int max_chunks, const std::string& save = "",
            int win = 8) {
    win = std::clamp(win, 1, 8);
    std::ifstream in(path, std::ios::binary);
    char magic[8];
    int32_t n_ctx = 0, n_vocab = 0, n_chunk = 0;
    if (!in.read(magic, 8) || std::memcmp(magic, "_logits_", 8) != 0 || !in.read((char*) &n_ctx, 4) || !in.read((char*) &n_vocab, 4) ||
        !in.read((char*) &n_chunk, 4)) {
        fprintf(stderr, "strata-x: %s is not a llama-perplexity --kl-divergence-base file\n", path.c_str());
        return 2;
    }
    if (n_vocab != (int) eng.hparams().n_vocab) {
        fprintf(stderr, "strata-x: the file's vocabulary is %d, the model's %lld\n", n_vocab, (long long) eng.hparams().n_vocab);
        return 2;
    }
    std::vector<int32_t> tokens((size_t) n_chunk * n_ctx);
    in.read((char*) tokens.data(), (std::streamsize) (tokens.size() * 4));
    const int first = n_ctx / 2, n_tok = n_ctx - 1 - first, nv = 2 * ((n_vocab + 1) / 2) + 4;
    const int chunks = max_chunks > 0 ? std::min(max_chunks, n_chunk) : n_chunk;
    fprintf(stderr, "strata-x: KLD against %s: %d chunks of %d tokens (scored: positions %d-%d), the first half through %s\n",
            path.c_str(), chunks, n_ctx, first, n_ctx - 2, mode == "windows" ? "decode windows" : "the prompt path");
    std::vector<uint16_t> base((size_t) n_tok * nv);
    std::vector<float> lg;
    KldAcc k;
    std::string err;
    std::ofstream out;
    if (!save.empty()) {
        out.open(save, std::ios::binary);
        const int32_t hdr[3] = {n_ctx, n_vocab, chunks};
        out.write("_logits_", 8);
        out.write((const char*) hdr, sizeof hdr);
        out.write((const char*) tokens.data(), (std::streamsize) ((size_t) chunks * n_ctx * 4));
        if (!out) {
            fprintf(stderr, "strata-x: cannot write %s\n", save.c_str());
            return 2;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < chunks; ++c) {
        if (save.empty() && !in.read((char*) base.data(), (std::streamsize) (base.size() * 2))) {
            fprintf(stderr, "strata-x: the file ends in chunk %d\n", c);
            return 2;
        }
        const int32_t* tk = tokens.data() + (size_t) c * n_ctx;
        eng.reset();
        if (!eng.prefill(tk, first, 0, err)) {
            fprintf(stderr, "strata-x: chunk %d: %s\n", c, err.c_str());
            return 1;
        }
        for (int p = first; p < n_ctx - 1;) {
            const int T = std::min(win, n_ctx - 1 - p);
            if (!eng.window(tk + p, T, p, false, err)) {
                fprintf(stderr, "strata-x: chunk %d window %d: %s\n", c, p, err.c_str());
                return 1;
            }
            for (int r = 0; r < T; ++r) {
                if (!eng.logits(lg, err, r)) return 1;
                if (!save.empty()) kld_encode(lg.data(), n_vocab, base.data() + (size_t) (p + r - first) * nv);
                else kld_row(lg.data(), n_vocab, base.data() + (size_t) (p + r - first) * nv, tk[p + r + 1], k);
            }
            p += T;
        }
        if (!save.empty()) {
            out.write((const char*) base.data(), (std::streamsize) (base.size() * 2));
            fprintf(stderr, "strata-x:   chunk %d: this engine's rows written\n", c);
            continue;
        }
        const double n = (double) k.count;
        fprintf(stderr, "strata-x:   chunk %d: mean KLD so far %.6f, same top %.2f%%\n", c, k.sum_kld / n, 100.0 * k.n_same_top / n);
    }
    if (!save.empty()) {
        out.close();
        fprintf(stderr, "strata-x: KLD base of this engine written to %s (%d chunks of %d tokens, windows of %d)\n", save.c_str(),
                chunks, n_ctx, win);
        return out ? 0 : 1;
    }
    const double n = (double) k.count;
    std::vector<float> s = k.klds;
    std::sort(s.begin(), s.end());
    auto q = [&](double f) { return s.empty() ? 0.0 : (double) s[std::min(s.size() - 1, (size_t) (f * (double) (s.size() - 1)))]; };
    const double ppl = exp(k.sum_nll / n), ppl_base = exp(k.sum_nll_base / n);
    const double kld_mean = k.sum_kld / n, kld_sd = sqrt(std::max(0.0, k.sum_kld2 / n - kld_mean * kld_mean) / std::max(1.0, n - 1));
    fprintf(stderr,
            "strata-x: KLD RESULT (%s, %zu tokens, %.0f s): mean KLD %.6f +- %.6f | median %.6f | 99%% %.6f | max %.6f | same top %.2f%% | "
            "PPL %.4f vs llama.cpp %.4f (ratio %.5f) | RMS dp %.4f%% | max dp %.4f%%\n",
            mode.c_str(), k.count, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), kld_mean, kld_sd, q(0.5),
            q(0.99), s.empty() ? 0.0 : (double) s.back(), 100.0 * k.n_same_top / n, ppl, ppl_base, ppl / ppl_base,
            100.0 * sqrt(k.sum_p_diff2 / n), 100.0 * k.max_p_diff);
    return 0;
}
}  // namespace

namespace strata::x {
int run_expert_check(const std::string& model, const std::string& dump, const std::vector<int>& only_layers);
void dsv41_reads_by_layer(double tokens, int n_layer);   // dsv41.cpp (STRATA_DSV41_READS_BY_LAYER)
}

int main(int argc, char** argv) {
    // moex: no diagnostics whatsoever with STRATA_X_QUIET=1 - nothing on stderr (quiet_stderr), and serve mode gathers no
    // per-request statistics (serve.cpp); the protocol on stdout (READY, T, DONE, ERR ...) is unchanged
    if (const char* q = std::getenv("STRATA_X_QUIET"); q && q[0] == '1') quiet_stderr();
    // moex: the engine was strata-glm until 2026-11-26, its switches STRATA_GLM_*; one still set under that name is not read
    for (char** e = environ; e && *e; ++e)
        if (!std::strncmp(*e, "STRATA_GLM_", 11)) {
            const char* eq = std::strchr(*e, '=');
            const std::string name(*e, eq ? (size_t) (eq - *e) : std::strlen(*e));
            fprintf(stderr, "strata-x: %s is the old name of STRATA_X_%s and is not read\n", name.c_str(), name.c_str() + 11);
        }
    // Strata's engine loads every kernel at start: a kernel first used inside a graph capture (or mid-prompt) would
    // otherwise be loaded lazily where a CUDA call is not allowed / VRAM may be gone
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
    std::string model, ids, ids_file, out_dir, match = ".", oracle;
    int n_gen = 16;
    double tol = 1e-2;
    bool print_ids = false;
    int top_n = 0;
    int64_t probe_from = 0;   // moex (#36b): with -o, the first N prompt tokens through the prompt path, unrecorded
    std::string runs, decode_runs, warm_file, kld_file, kld_mode = "prompt";
    std::string imx_out;   // --imatrix FILE: the experts' importance over the prompt ids, in chunks (requant_pack's input)
    int imx_chunk = 4096;
    int kld_chunks = 0;
    std::string kld_save;
    int kld_window = 8;
    int warm_tokens = 256;
    bool serve_mode = false;
    std::vector<int32_t> eos_ids, turn_tokens;
    int ck_keep = 6, ck_deep = 4;
    int64_t ck_every = 16384, ck_root = 2048;
    strata::x::EngineOptions opt;
    std::string expert_check, profile_save;
    double profile_every = 10.0;
    int32_t image_pad = -1;   // moex vision: GENI's pad id (the server passes it with a "vision" config)
    std::vector<int> check_layers;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                exit(2);
            }
            return argv[++i];
        };
        if (a == "-m") model = next();
        else if (a == "--expert-check") expert_check = next();   // the routed experts' paths against the exact product
        else if (a == "--profile-save" || a == "--expert-profile-save") profile_save = next();   // the run's routing as an expert profile
        else if (a == "--expert-profile-save-every") profile_every = std::stod(next());          // minutes, serve mode (0: at the end only)
        else if (a == "--check-layers") {
            std::stringstream ls(next());
            for (std::string x; std::getline(ls, x, ',');) check_layers.push_back(std::stoi(x));
        }
        else if (a == "-t") ids = next();
        else if (a == "--tokens-file") ids_file = next();
        else if (a == "-n") n_gen = std::stoi(next());
        else if (a == "--ctx") opt.n_ctx = std::stoll(next());
        else if (a == "--threads") opt.threads = std::stoi(next());
        else if (a == "--no-pin") opt.pin = false;
        else if (a == "--device") {   // one GPU
            opt.device = std::stoi(next());
            opt.devices = {opt.device};
            opt.split = 0;
        }
        else if (a == "--gpus") {
            opt.devices.clear();
            for (int32_t d : parse_ids(next())) opt.devices.push_back(d);
        }
        else if (a == "--split") opt.split = std::stoi(next());
        else if (a == "--layer-split") {   // Strata's name: "auto", or the first layer of each later GPU's share (two GPUs: one number)
            const std::string v = next();
            if (v == "auto") opt.split = -1;
            else opt.split = std::stoi(v.substr(0, v.find(',')));
        }
        else if (a == "--vram-reserve-later-mib") opt.vram_reserve_later_mib = std::stoll(next());
        else if (a == "--vision-weights") opt.vision_manifest = next();   // moex (DSV41_PLAN §54): serve's VENC, the image encoder on GPU 0
        else if (a == "-o") out_dir = next();
        else if (a == "--match") match = next();
        else if (a == "--oracle") oracle = next();
        else if (a == "--tol") tol = std::stod(next());
        else if (a == "--print-ids") print_ids = true;
        else if (a == "--runs") runs = next();
        else if (a == "--decode-runs") decode_runs = next();   // wN[:P] / fN,... one loaded model: a warm-up, then each decode (fN: stand-in drafts, the verify cost of N rows)
        else if (a == "--warm-tokens") warm_tokens = std::stoi(next());
        else if (a == "--warm-file") warm_file = next();
        else if (a == "--kld") kld_file = next();                 // llama-perplexity --kl-divergence-base FILE: the parity check
        else if (a == "--kld-mode") kld_mode = next();            // prompt (default) | windows: how each chunk's first half is read
        else if (a == "--kld-chunks") kld_chunks = std::stoi(next());
        else if (a == "--kld-save") kld_save = next();             // write this engine's rows as the base (the tokens from --kld)
        else if (a == "--imatrix") imx_out = next();               // DeepSeek-V4.1's pack: collect the experts' imatrix over the ids
        else if (a == "--imatrix-chunk") imx_chunk = std::stoi(next());
        else if (a == "--kld-window") kld_window = std::stoi(next());   // decode window rows while scoring (8; 1: every position alone)
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context") opt.n_ctx = std::stoll(next());
        else if (a == "--eos" || a == "--eos-ids") eos_ids = parse_ids(next());
        else if (a == "--turn-tokens") turn_tokens = parse_ids(next());
        else if (a == "--image-pad") image_pad = (int32_t) std::stol(next());   // serve: GENI puts image rows at this id's runs
        else if (a == "--turn-token") turn_tokens.push_back((int32_t) std::stoll(next()));   // Strata's name (one id)
        else if (a == "--kv") {
            const std::string v = next();
            if (v != "fp16") {
                fprintf(stderr, "strata-x: --kv %s: only fp16 is implemented\n", v.c_str());
                return 2;
            }
        }
        else if (a == "--prefill") {   // Strata's name: "auto" or a chunk size
            const std::string v = next();
            // "auto" (Strata's auto:16384 / auto:32768 caps: this engine's auto stays <= 8192, the cap is taken as a request
            // for that chunk), "off" (windows), or a chunk size
            if (v.rfind("auto", 0) == 0) opt.prefill_chunk = v.size() > 5 && v[4] == ':' ? std::max(0, std::atoi(v.c_str() + 5)) : 0;
            else opt.prefill_chunk = v == "off" ? -1 : std::max(-1, std::atoi(v.c_str()));
        }
        else if (a == "--prompt-cache") ck_keep = std::stoi(next());
        else if (a == "--prompt-cache-every") ck_every = std::stoll(next());
        else if (a == "--prompt-cache-root") ck_root = std::stoll(next());
        else if (a == "--prompt-cache-deep") ck_deep = std::stoi(next());   // moex: deep checkpoints (session.cpp plan_cuts)
        else if (a == "--top") top_n = std::stoi(next());   // the last window's top logits (with -n 1: the prompt's)
        else if (a == "--probe-from") probe_from = std::stoll(next());   // -o: record only the tokens from N on (a deep check)
        else if (a == "--expert-slots") opt.expert_slots = std::stoll(next());
        else if (a == "--vram-reserve-mib") opt.vram_reserve_mib = std::stoll(next());
        else if (a == "--expert-profile") opt.expert_profile = next();
        else if (a == "--no-adaptive") opt.adaptive = false;
        else if (a == "--adapt-every") opt.adapt_every = std::stoi(next());
        else if (a == "--adapt-swaps") opt.adapt_swaps = std::stoi(next());
        else if (a == "--adapt-decay") opt.adapt_decay = std::stof(next());
        else if (a == "--experts-in-place") opt.experts_in_ram = 0;
        else if (a == "--ssd-cache-gib") opt.ssd_cache_gib = std::stod(next());   // DeepSeek-V4.1: the SSD cache (EngineOptions)
        // moex (DSV41_PLAN §57): parked conversations (DeepSeek-V4 / V4.1, --serve): the budget, the records, the smallest
        // conversation worth parking, the RAM left free after a copy
        else if (a == "--conversation-cache-mib") opt.conv_cache_mib = std::max(0LL, std::stoll(next()));
        else if (a == "--conversation-cache-slots") opt.conv_cache_slots = std::max(1, std::stoi(next()));
        else if (a == "--conversation-cache-min-tokens") opt.conv_cache_min_tokens = std::max(0LL, std::stoll(next()));
        else if (a == "--conversation-cache-min-free-mib") opt.conv_cache_min_free_mib = std::max(0LL, std::stoll(next()));
        else if (a == "--conversation-dir") opt.conv_dir = next();   // moex (#53): the conversations kept across restarts
        else if (a == "--conversation-dir-mib") opt.conv_dir_mib = std::max(0LL, std::stoll(next()));
        else if (a == "--pack") opt.pack = next();   // DeepSeek-V4.1: the pack directory (the expert records, the engram tables)
        else if (a == "--experts-gguf") opt.experts_gguf = next();   // moex: DeepSeek-V4.1's routed experts from a GGUF set
        else if (a == "--draft-dspark") opt.dspark = next();   // DeepSeek-V4's DSpark drafter (llama.cpp's -md with draft-dspark)
        else if (a == "--dspark-mask") opt.dspark_mask = std::stoi(next());
        else if (a == "--draft-mtp") opt.draft_mtp = next();   // DeepSeek-V4's NextN head GGUF (llama.cpp's -md with draft-mtp)
        else if (a == "--draft-dflash") opt.dflash = next();   // MiMo's DFlash drafter (llama.cpp's -md with draft-dflash)
        else if (a == "--draft-dfly") opt.dfly = next();       // moex: Hy4-flash's DFly2 drafter (moex/tools/hy4/dfly_to_gguf.py)
        else if (a == "--spec-policy") opt.spec_policy = next() == std::string("gate") ? 0 : 1;   // DFlash: cost (default) | gate
        else if (a == "--cpu-read") opt.cpu_read = std::max(0LL, std::stoll(next()));   // a prompt segment up to N: no stream
        else if (a == "--share") opt.prefill_share = std::max(0LL, std::stoll(next()));   // a segment up to N: the CPU shares (0 off)
        else if (a == "--mtp") opt.mtp_window = std::max(opt.mtp_window, 3);   // the NextN block drafts (window 3: measured best); --spec N sets it
        else if (a == "--spec") {
            const int v = std::stoi(next());
            opt.mtp_window = v >= 2 ? std::min(v, 8) : 0;
        }
        else if (a == "--spec-min-p") opt.spec_min_p = std::stof(next());
        else if (a == "--coupled-draft") opt.coupled_draft = true;      // Strata's coupled draft sampling (STRATA_SPEC_COUPLED=1)
        else if (a == "--no-coupled-draft") opt.coupled_draft = false;
        else if (a == "--suffix-draft") opt.suffix_draft = std::max(0, std::stoi(next()));   // Strata's prompt lookup (0: off)
        else if (a == "--lookup-window") opt.lookup_window = std::max(0, std::stoi(next()));
        else if (a == "--prefill-chunk") opt.prefill_chunk = std::stoi(next());
        else if (a == "--prefill-ring") opt.prefill_ring = std::stoi(next());
        else if (a == "--prefill-ring-frac") opt.prefill_ring_frac = std::stod(next());   // moex: the full ring, short reads kept
        else if (a == "--pin-gguf") opt.register_experts = true;
        else if (a == "--no-graphs") opt.graphs = false;
        else if (a == "--no-warmup") opt.warmup = false;
        else if (a == "--prefill-sub") opt.prefill_sub = std::stoi(next());
        else if (a == "--short-read") opt.short_read = std::stoll(next());
        else if (a == "--experts-in-ram") opt.experts_in_ram = 1;
        else if (a == "--resident-experts") opt.experts_in_ram = 1;   // Strata's name: the RAM copy with its 4 GiB headroom
        else if (a == "--resident-headroom-gib") opt.resident_headroom_gib = std::stod(next());
        else if (a == "--exchange-blobs") opt.exchange_blobs = std::stoi(next());
        else if (a == "--arena-thp") opt.arena_thp = true;
        else {
            usage();
            return 2;
        }
    }
    if (!ids_file.empty()) {
        std::ifstream in(ids_file);
        std::getline(in, ids);
    }
    if (!expert_check.empty() && !model.empty()) return strata::x::run_expert_check(model, expert_check, check_layers);
    std::vector<int32_t> prompt = parse_ids(ids);
    if (!imx_out.empty()) {   // the stream mode: every routed expert through the GPU products (the sums are taken there)
        opt.cpu_read = 0;
        opt.prefill_share = 0;
    }
    if (model.empty() || (prompt.empty() && !serve_mode && kld_file.empty())) {
        usage();
        return 2;
    }
    if (!opt.conv_dir.empty()) opt.conv_identity = conv_identity(argc, argv, model, opt.pack);
    opt.images = image_pad >= 0;   // moex vision: the server sends pictures (Hy4-flash keeps its M-RoPE table)
    strata::x::Engine eng;
    std::string err;
    if (!eng.load(model, opt, err)) {
        fprintf(stderr, "strata-x: %s\n", err.c_str());
        return 1;
    }
    if (!imx_out.empty()) {   // each chunk from an empty state through the prompt path, every routed expert on the GPU
        eng.set_short_read(0);
        if (!eng.imatrix_begin(err)) {
            fprintf(stderr, "strata-x: %s\n", err.c_str());
            return 1;
        }
        const auto ti = std::chrono::steady_clock::now();
        for (size_t c0 = 0; c0 < prompt.size(); c0 += (size_t) imx_chunk) {
            const int nt = (int) std::min<size_t>((size_t) imx_chunk, prompt.size() - c0);
            eng.reset();
            if (!eng.prefill(prompt.data() + c0, nt, 0, err)) {
                fprintf(stderr, "strata-x: imatrix chunk at %zu: %s\n", c0, err.c_str());
                return 1;
            }
            fprintf(stderr, "strata-x: imatrix: %zu of %zu tokens (%.0f s)\n", c0 + nt, prompt.size(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - ti).count());
            if (!eng.imatrix_write(imx_out, err)) {   // after every chunk: an interrupted run keeps what it read
                fprintf(stderr, "strata-x: %s\n", err.c_str());
                return 1;
            }
        }
        fprintf(stderr, "strata-x: imatrix written to %s\n", imx_out.c_str());
        return 0;
    }
    if (!kld_file.empty()) {
        if (kld_mode == "windows") eng.set_prompt_options(-1, 0, opt.prefill_sub);   // every position through decode windows
        else eng.set_short_read(0);                                                   // the prompt path for every first half
        return run_kld(eng, kld_file, kld_mode, kld_chunks, kld_save, kld_window);
    }
    if (serve_mode) {   // Strata's engine protocol: the Python server owns everything but the ids
        eng.set_checkpoint_policy(turn_tokens, ck_keep, ck_every, ck_root, ck_deep);
        strata::x::ServeOptions so;
        so.max_context = opt.n_ctx;
        so.eos = eos_ids;
        so.spec = opt.mtp_window;
        so.profile_save = profile_save;
        so.profile_every_min = profile_every;
        so.image_pad = image_pad;
        so.pack = opt.pack;
        if (opt.conv_cache_mib > 0) {   // moex (§57): the server's Monitor card reads these (the Qwen engine's names)
            if (eng.hparams().arch == strata::x::Arch::dsv4)
                so.info += "conversation_cache_mib=" + std::to_string(opt.conv_cache_mib) +
                           " conversation_cache_slots=" + std::to_string(opt.conv_cache_slots);
            else
                fprintf(stderr, "strata-x: --conversation-cache-mib parks DeepSeek-V4 / V4.1 conversations only: off\n");
        }
        if (so.eos.empty() && eng.hparams().eos_id >= 0) so.eos.push_back((int32_t) eng.hparams().eos_id);   // the GGUF's own
        return strata::x::serve(eng, so);
    }
    moex::Recorder rec;
    const bool probing = !out_dir.empty() || !oracle.empty();
    if (!out_dir.empty() && !rec.open(out_dir, match, err)) {
        fprintf(stderr, "strata-x: %s\n", err.c_str());
        return 1;
    }
    if (!oracle.empty() && !rec.attach_oracle(oracle, err)) {
        fprintf(stderr, "strata-x: %s\n", err.c_str());
        return 1;
    }
    if (probing) {
        eng.set_recorder(&rec);
        n_gen = 0;   // a check replays the oracle's tokens only
    }

    using clk = std::chrono::steady_clock;
    std::vector<int32_t> gen;
    const auto t0 = clk::now();
    int64_t pos = 0;
    if (probing) {   // the op-level check: one token at a time, like the oracle's dump
        if (probe_from > 0) {   // moex: a deep check - the first probe_from tokens through the prompt path, nothing recorded
            if (probe_from >= (int64_t) prompt.size()) {
                fprintf(stderr, "strata-x: --probe-from %lld leaves no token to record\n", (long long) probe_from);
                return 1;
            }
            eng.set_recorder(nullptr);
            eng.set_short_read(0);
            const auto tp = clk::now();
            if (!eng.prefill(prompt.data(), (int) probe_from, 0, err)) {
                fprintf(stderr, "strata-x: the prompt path to %lld: %s\n", (long long) probe_from, err.c_str());
                return 1;
            }
            fprintf(stderr, "strata-x: %lld tokens through the prompt path in %.1f s; recording from position %lld\n",
                    (long long) probe_from, std::chrono::duration<double>(clk::now() - tp).count(), (long long) probe_from);
            eng.set_recorder(&rec);
            pos = probe_from;
        }
        for (size_t i = (size_t) pos; i < prompt.size(); ++i, ++pos) {
            rec.set_step((int) pos);
            if (!eng.step(prompt[i], pos, err)) {
                fprintf(stderr, "strata-x: step %lld: %s\n", (long long) pos, err.c_str());
                return 1;
            }
            std::vector<float> lg;
            if (eng.logits(lg, err)) rec.f32("logits", -1, lg.data(), (int64_t) lg.size());
        }
    }
    if (!decode_runs.empty() && !probing) {
        // A DECODE SWEEP on one loaded model: a warm-up generation (another text, if given) so the tier starts converged, as a
        // long-running server's is; then each entry - wN = windows of up to N rows (w1: plain), :P = spec-min-p - on the same
        // prompt from a reset state, the tier adapting on.  Each prints one line.
        std::vector<int32_t> warm = prompt;
        if (!warm_file.empty()) {
            std::ifstream in(warm_file);
            std::string w;
            std::getline(in, w);
            warm = parse_ids(w);
        }
        // STRATA_PROFILE_RUN=K: the K-th decode run (1-based, after the warm-up) between cudaProfilerStart / Stop, for
        // nsys --capture-range=cudaProfilerApi (a profile of one run, not the load)
        static const int prof_run = std::getenv("STRATA_PROFILE_RUN") ? std::atoi(std::getenv("STRATA_PROFILE_RUN")) : 0;
        int run_no = 0;
        auto one = [&](const char* tag, const std::vector<int32_t>& pr, int ngen, int wcap, float mp, int stand_in = 0) -> bool {
            eng.reset();
            eng.set_spec(wcap, mp);
            eng.set_stand_in(stand_in);
            const auto s0 = eng.stats();
            strata::x::GenResult r;
            const bool prof = std::strcmp(tag, "warm-up") != 0 && ++run_no == prof_run;
            if (prof) cudaProfilerStart();
            const bool ok = eng.generate(pr, ngen, wcap >= 2, r, err);
            if (prof) cudaProfilerStop();
            if (!ok) {
                fprintf(stderr, "strata-x: %s\n", err.c_str());
                return false;
            }
            const auto s1 = eng.stats();
            const int64_t lk = (s1.hits - s0.hits) + (s1.misses - s0.misses);
            fprintf(stderr,
                    "strata-x: DECODE %-8s %4zu tokens in %5.2f s = %5.2f T/s | windows %lld (%.2f tok/win) accepted %lld/%lld "
                    "(%.0f%%) drafter %.1f ms/win | hit %.1f%% pool %.1f ms/tok wait %.1f ms/tok swaps %lld | prompt %zu in %.1f s | "
                    "ids %d,%d,%d\n",
                    tag, r.tokens.size(), r.ms_gen / 1000.0, r.tokens.size() / std::max(1e-9, r.ms_gen / 1000.0),
                    (long long) r.windows, (double) r.tokens.size() / std::max<int64_t>(1, r.windows), (long long) r.accepted,
                    (long long) r.drafted, 100.0 * r.accepted / std::max<double>(1.0, (double) r.drafted),
                    r.ms_draft / std::max<int64_t>(1, r.windows), 100.0 * (s1.hits - s0.hits) / std::max<int64_t>(1, lk),
                    (s1.ms_pool - s0.ms_pool) / std::max<double>(1.0, (double) r.tokens.size()),
                    (s1.ms_wait - s0.ms_wait) / std::max<double>(1.0, (double) r.tokens.size()), (long long) (s1.swaps - s0.swaps),
                    pr.size(), r.ms_prefill / 1000.0,
                    r.tokens.size() > 0 ? r.tokens[0] : -1, r.tokens.size() > 1 ? r.tokens[1] : -1, r.tokens.size() > 2 ? r.tokens[2] : -1);
            if (s1.cpu_ram + s1.cpu_ssd > s0.cpu_ram + s0.cpu_ssd) {   // DeepSeek-V4.1's SSD tier: where the CPU's experts came from
                const double nt = std::max<double>(1.0, (double) r.tokens.size());
                fprintf(stderr, "strata-x:          per token: %.1f VRAM hits, %.1f CPU from RAM, %.1f CPU from the SSD (%.0f ms waiting for it); "
                                "SSD reads %.1f, cache hits %.1f, read-ahead %.1f issued / %.1f used / %.1f wasted; VRAM fills %.2f from the "
                                "cache / %.2f read again\n",
                        (s1.hits - s0.hits) / nt, (s1.cpu_ram - s0.cpu_ram) / nt, (s1.cpu_ssd - s0.cpu_ssd) / nt,
                        (s1.ms_ssd_wait - s0.ms_ssd_wait) / nt, (s1.ssd_reads - s0.ssd_reads) / nt, (s1.ssd_hits - s0.ssd_hits) / nt,
                        (s1.ahead_issued - s0.ahead_issued) / nt, (s1.ahead_used - s0.ahead_used) / nt, (s1.ahead_wasted - s0.ahead_wasted) / nt,
                        (s1.fill_cached - s0.fill_cached) / nt, (s1.fill_reads - s0.fill_reads) / nt);
                static const bool by_layer = std::getenv("STRATA_DSV41_READS_BY_LAYER") != nullptr;
                if (by_layer) strata::x::dsv41_reads_by_layer(nt, (int) eng.hparams().n_layer);
            }
            if (s1.pool_bytes > s0.pool_bytes) {   // the CPU pool's time per token: its barriers against its drain
                const double nt = std::max<double>(1.0, (double) r.tokens.size());
                fprintf(stderr, "strata-x:          pool per token: park %.2f ms, drain %.2f ms, repark %.2f ms; %.2f GB at %.1f GB/s "
                                "over the drain\n",
                        (s1.pool_park_ms - s0.pool_park_ms) / nt, (s1.pool_drain_ms - s0.pool_drain_ms) / nt,
                        (s1.pool_repark_ms - s0.pool_repark_ms) / nt, (s1.pool_bytes - s0.pool_bytes) / nt / 1e9,
                        (s1.pool_bytes - s0.pool_bytes) / 1e9 / std::max(1e-9, (s1.pool_drain_ms - s0.pool_drain_ms) / 1000.0));
            }
            if (r.offered_at[0] > 0) {   // per draft depth: offered, accepted
                fprintf(stderr, "strata-x:          by depth:");
                for (int j = 0; j < 8 && r.offered_at[j] > 0; ++j)
                    fprintf(stderr, " d%d %lld/%lld (%.0f%%)", j, (long long) r.accepted_at[j], (long long) r.offered_at[j],
                            100.0 * r.accepted_at[j] / (double) r.offered_at[j]);
                fprintf(stderr, "\n");
            }
            if (std::getenv("STRATA_X_PRINT_IDS")) {
                for (size_t i = 0; i < r.tokens.size(); ++i) printf("%s%d", i ? "," : "", r.tokens[i]);
                printf("\n");
                fflush(stdout);
            }
            return true;
        };
        if (warm_tokens > 0 && !one("warm-up", warm, warm_tokens, opt.mtp_window >= 2 ? opt.mtp_window : 1, -1.0f)) return 1;
        std::stringstream ds(decode_runs);
        std::string item;
        while (std::getline(ds, item, ',')) {
            int wcap = 1, stand_in = 0;
            float mp = -1.0f;
            // fN: windows of N rows with stand-in drafts (Engine::set_stand_in) - the verify cost of N rows, no drafter needed
            if (!item.empty() && item[0] == 'f' && sscanf(item.c_str() + 1, "%d", &stand_in) == 1 && stand_in >= 1) {
                if (!one(item.c_str(), prompt, std::max(1, n_gen), 1, -1.0f, stand_in)) return 1;
                continue;
            }
            if (item.empty() || item[0] != 'w' || sscanf(item.c_str() + 1, "%d:%f", &wcap, &mp) < 1) {
                fprintf(stderr, "strata-x: --decode-runs wants wN[:P] or fN entries\n");
                return 2;
            }
            if (wcap > std::max(1, opt.mtp_window)) {
                fprintf(stderr, "strata-x: %s: the window is larger than the loaded --spec %d\n", item.c_str(), opt.mtp_window);
                return 2;
            }
            if (!one(item.c_str(), prompt, std::max(1, n_gen), wcap, mp)) return 1;
        }
        if (!profile_save.empty()) {   // the sweep's routing (the warm-up's text included) as an expert profile
            if (!eng.save_profile(profile_save, err)) fprintf(stderr, "strata-x: %s\n", err.c_str());
            else fprintf(stderr, "strata-x: expert profile written to %s\n", profile_save.c_str());
        }
        return 0;
    }
    if (!runs.empty() && !probing) {   // a benchmark sweep: the prompt path's knobs per run, one loaded model
        if (!warm_file.empty() && warm_tokens > 0) {
            // --warm-file with --runs: one generation on that text first, so the tiers (the SSD cache) start as a running
            // server's do - a long prompt takes every cached record instead of reading it
            std::ifstream in(warm_file);
            std::string w;
            std::getline(in, w);
            const std::vector<int32_t> warm = parse_ids(w);
            eng.reset();
            const auto sw0 = eng.stats();
            strata::x::GenResult r;
            if (!eng.generate(warm, warm_tokens, opt.mtp_window > 1, r, err)) {
                fprintf(stderr, "strata-x: %s\n", err.c_str());
                return 1;
            }
            const auto sw1 = eng.stats();
            fprintf(stderr, "strata-x: WARM-UP %zu prompt + %zu tokens in %.1f + %.1f s (SSD reads %lld)\n", warm.size(),
                    r.tokens.size(), r.ms_prefill / 1000.0, r.ms_gen / 1000.0, (long long) (sw1.ssd_reads - sw0.ssd_reads));
        }
        std::stringstream rs(runs);
        std::string item;
        while (std::getline(rs, item, ',')) {
            int ring = 0, chunk = 0, sub = opt.prefill_sub, plen = 0;
            if (sscanf(item.c_str(), "%d:%d:%d:%d", &ring, &chunk, &sub, &plen) < 2) {
                fprintf(stderr, "strata-x: --runs wants ring:chunk[:sub[:prompt length]] entries\n");
                return 2;
            }
            if (sub <= 0) sub = opt.prefill_sub;
            eng.set_prompt_options(chunk, ring, sub);
            eng.reset();
            const auto st0 = eng.stats();
            strata::x::GenResult r;
            const std::vector<int32_t> pr = plen > 0 && plen < (int) prompt.size() ? std::vector<int32_t>(prompt.begin(), prompt.begin() + plen) : prompt;
            if (!eng.generate(pr, std::max(1, n_gen), opt.mtp_window > 1, r, err)) {
                fprintf(stderr, "strata-x: %s\n", err.c_str());
                return 1;
            }
            const auto st1 = eng.stats();
            std::vector<float> lg;
            std::string top;
            if (top_n > 0 && eng.logits(lg, err)) {
                std::vector<int> idx(lg.size());
                for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
                std::partial_sort(idx.begin(), idx.begin() + std::min<size_t>(top_n, idx.size()), idx.end(),
                                  [&](int a, int b) { return lg[a] > lg[b]; });
                char b[64];
                for (int i = 0; i < top_n; ++i) {
                    snprintf(b, sizeof b, " %d:%.3f", idx[i], lg[idx[i]]);
                    top += b;
                }
            }
            fprintf(stderr,
                    "strata-x: RUN ring %d chunk %d sub %d: prefill %zu tokens in %.1f s = %.1f T/s | chunk %lld ring %lld "
                    "lent %lld streamed %lld (%lld read from the SSD, %lld from its cache, %lld kept; VRAM fills %lld from the cache / "
                    "%lld read) | moe wait %.1f s | engram rows %.1f s | top:%s | ids %d,%d,%d\n",
                    ring, chunk, sub, pr.size(), r.ms_prefill / 1000.0, pr.size() / std::max(1e-9, r.ms_prefill / 1000.0),
                    (long long) st1.prefill_chunk, (long long) st1.prefill_ring, (long long) st1.lent_slots,
                    (long long) (st1.streamed - st0.streamed), (long long) (st1.rec_reads - st0.rec_reads),
                    (long long) (st1.rec_cached - st0.rec_cached), (long long) (st1.prompt_kept - st0.prompt_kept),
                    (long long) (st1.fill_cached - st0.fill_cached), (long long) (st1.fill_reads - st0.fill_reads),
                    (st1.ms_prefill_wait - st0.ms_prefill_wait) / 1000.0, (st1.ms_eg_rows - st0.ms_eg_rows) / 1000.0, top.c_str(),
                    r.tokens.size() > 0 ? r.tokens[0] : -1, r.tokens.size() > 1 ? r.tokens[1] : -1,
                    r.tokens.size() > 2 ? r.tokens[2] : -1);
        }
        return 0;
    }
    double ms_gen = 0;
    if (!probing) {
        strata::x::GenResult r;
        if (!eng.generate(prompt, n_gen, opt.mtp_window > 1, r, err)) {
            fprintf(stderr, "strata-x: %s\n", err.c_str());
            return 1;
        }
        gen = r.tokens;
        ms_gen = r.ms_gen;
        fprintf(stderr, "strata-x: prefill %zu tokens in %.1f s (%.1f T/s)\n", prompt.size(), r.ms_prefill / 1000.0,
                prompt.size() / std::max(1e-9, r.ms_prefill / 1000.0));
        if (opt.mtp_window > 1)
            fprintf(stderr, "strata-x: MTP (window %d, min p %.2f): drafts accepted %lld of %lld (%.1f%%), %lld trunk windows, %.2f tokens per "
                            "window; drafter %.1f ms per window\n",
                    opt.mtp_window, opt.spec_min_p, (long long) r.accepted, (long long) r.drafted,
                    100.0 * r.accepted / std::max<double>(1.0, (double) r.drafted), (long long) r.windows,
                    (double) gen.size() / std::max<int64_t>(1, r.windows), r.ms_draft / std::max<int64_t>(1, r.windows));
    }
    const double ms_prompt = std::chrono::duration<double, std::milli>(clk::now() - t0).count() - ms_gen;
    const auto& st = eng.stats();
    fprintf(stderr, "strata-x: prompt %zu tokens in %.1f s (%.2f T/s)", prompt.size(), ms_prompt / 1000.0,
            prompt.size() / (ms_prompt / 1000.0));
    if (!gen.empty())
        fprintf(stderr, ", generated %zu in %.1f s (%.2f T/s)", gen.size(), ms_gen / 1000.0,
                gen.size() / std::max(1e-9, ms_gen / 1000.0));
    fprintf(stderr, "\nstrata-x: per token %.1f ms: pool %.1f ms, router wait %.1f ms\n",
            st.ms_total / std::max<int64_t>(1, st.tokens), st.ms_pool / std::max<int64_t>(1, st.tokens),
            st.ms_wait / std::max<int64_t>(1, st.tokens));
    if (st.ms_eg > 0)   // moex: Hy4-flash's Engram rows (the windows' host reads, every window since the start)
        fprintf(stderr, "strata-x: Engram rows read on the host: %.1f ms in all%s\n", st.ms_eg,
                st.hy4_eg_hits + st.hy4_eg_misses > 0
                    ? (" (the row cache held " + std::to_string((int) (100.0 * (double) st.hy4_eg_hits /
                                                                    (double) (st.hy4_eg_hits + st.hy4_eg_misses) + 0.5)) +
                       "% of " + std::to_string(st.hy4_eg_hits + st.hy4_eg_misses) + " rows)").c_str()
                    : "");
    if (st.pool_bytes > 0 && st.tokens > 0) {
        const double tk = (double) std::max<int64_t>(1, st.tokens);
        fprintf(stderr,
                "strata-x: CPU pool (%d workers + host) per token: gate/up %.1f ms, quantize %.1f ms, down %.1f ms, "
                "barriers %.1f + %.1f ms (park, repark); %.2f GB read -> %.1f GB/s over the drain\n",
                st.pool_workers, st.pool_gu_ms / tk, st.pool_q_ms / tk, st.pool_down_ms / tk, st.pool_park_ms / tk,
                st.pool_repark_ms / tk, st.pool_bytes / tk / 1e9,
                st.pool_bytes / 1e9 / std::max(1e-9, st.pool_drain_ms / 1000.0));
    }
    if (st.tokens_prefill)
        fprintf(stderr, "strata-x: prompt path: %lld tokens in %.1f s (%.1f T/s), chunk %lld, ring %lld, %lld slots lent; "
                        "%lld experts streamed; MoE host time %.1f s, of it waiting for the routing %.1f s\n",
                (long long) st.tokens_prefill, st.ms_prefill / 1000.0, st.tokens_prefill / std::max(1e-9, st.ms_prefill / 1000.0),
                (long long) st.prefill_chunk, (long long) st.prefill_ring, (long long) st.lent_slots,
                (long long) st.streamed, st.ms_prefill_moe / 1000.0, st.ms_prefill_wait / 1000.0);
    if (st.hits + st.misses > 0)
        fprintf(stderr, "strata-x: VRAM expert tier: %lld slots, hit rate %.1f%% (%lld of %lld), %lld swaps, %lld fills\n",
                (long long) st.slots, 100.0 * st.hits / (double) (st.hits + st.misses), (long long) st.hits,
                (long long) (st.hits + st.misses), (long long) st.swaps, (long long) st.fills);
    if (top_n > 0) {
        std::vector<float> lg;
        if (eng.logits(lg, err)) {
            std::vector<int> idx(lg.size());
            for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
            std::partial_sort(idx.begin(), idx.begin() + top_n, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            fprintf(stderr, "strata-x: top logits:");
            for (int i = 0; i < top_n; ++i) fprintf(stderr, " %d:%.4f", idx[i], lg[idx[i]]);
            fprintf(stderr, "\n");
        }
    }
    if (print_ids || !gen.empty()) {
        for (size_t i = 0; i < gen.size(); ++i) printf("%s%d", i ? "," : "", gen[i]);
        printf("\n");
    }
    if (!profile_save.empty()) {
        if (!eng.save_profile(profile_save, err)) fprintf(stderr, "strata-x: %s\n", err.c_str());
        else fprintf(stderr, "strata-x: expert profile written to %s\n", profile_save.c_str());
    }
    int bad = 0;
    if (!oracle.empty()) {
        bad = rec.report(tol, stdout);
        printf("%d op outputs over tol %.1e\n", bad, tol);
    }
    rec.close();
    return bad ? 1 : 0;
}
