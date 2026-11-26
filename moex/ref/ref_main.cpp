// moex/ref/ref_main.cpp - run a moex CPU reference (by the GGUF's architecture: glm5next, mimo2) over a token
// sequence, dumping what llama_dump dumps.
//
//     moex_ref -m model.gguf -t 1,2,3 | --tokens-file tokens.txt  [-o DIR] [--match REGEX] [--ctx N] [--threads N]
//             [--oracle DIR [--tol T]]
//
// Token by token, positions 0..n-1 of one sequence.  Prints each step's arg-max token and the top logit.
// --oracle: the op-level check (Recorder::attach_oracle) against a llama_dump of the same tokens; prints, per op
// output, the worst relative error given the oracle's own inputs, and exits 1 when any exceeds --tol.
#include "glm_ref.hpp"
#include "mimo_ref.hpp"

#include <functional>
#include <memory>

#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty()) out.push_back((int32_t) std::stol(item));
    return out;
}

int main(int argc, char** argv) {
    std::string model, out_dir, ids, ids_file, match = ".", oracle;
    double tol = 1e-4;
    int64_t n_ctx = 0;
    int threads = 4;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "-m") model = next();
        else if (a == "-o") out_dir = next();
        else if (a == "-t") ids = next();
        else if (a == "--tokens-file") ids_file = next();
        else if (a == "--match") match = next();
        else if (a == "--ctx") n_ctx = std::stoll(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--oracle") oracle = next();
        else if (a == "--tol") tol = std::stod(next());
        else {
            fprintf(stderr, "usage: moex_ref -m model.gguf (-t IDS | --tokens-file F) [-o DIR] [--match RE] [--ctx N] [--threads N]\n");
            return 2;
        }
    }
    if (!ids_file.empty()) {
        std::ifstream in(ids_file);
        std::getline(in, ids);
    }
    std::vector<int32_t> toks = parse_ids(ids);
    if (model.empty() || toks.empty()) {
        fprintf(stderr, "moex_ref: need -m and tokens\n");
        return 2;
    }
    ggml_cpu_init();

    moex::GgufFile f;
    std::string err;
    if (!f.open(model, err)) {
        fprintf(stderr, "moex_ref: %s\n", err.c_str());
        return 1;
    }
    const std::string arch = f.str("general.architecture");
    std::function<void(int32_t, int64_t, std::vector<float>&, moex::Recorder*)> step;
    std::unique_ptr<moex::GlmRef> glm;
    std::unique_ptr<moex::MimoRef> mimo;
    moex::GlmHparams ghp;
    moex::MimoHparams mhp;
    auto ctx_for = [&]() { return n_ctx > 0 ? n_ctx : std::max<int64_t>(256, ((int64_t) toks.size() + 255) / 256 * 256); };
    if (arch == "glm5next") {
        if (!ghp.load(f, err)) {
            fprintf(stderr, "moex_ref: %s\n", err.c_str());
            return 1;
        }
        fprintf(stderr, "moex_ref: glm5next, %lld layers, n_embd %lld, %zu tokens\n", (long long) ghp.n_layer,
                (long long) ghp.n_embd, toks.size());
        glm = std::make_unique<moex::GlmRef>(f, ghp, ctx_for(), threads);
        step = [&](int32_t t, int64_t p, std::vector<float>& l, moex::Recorder* r) { glm->step(t, p, l, r); };
    } else if (arch == "mimo2") {
        if (!mhp.load(f, err)) {
            fprintf(stderr, "moex_ref: %s\n", err.c_str());
            return 1;
        }
        fprintf(stderr, "moex_ref: mimo2, %lld layers (+%lld NextN), n_embd %lld, %lld experts top-%lld, %zu tokens\n",
                (long long) mhp.n_layer, (long long) (mhp.n_layer_all - mhp.n_layer), (long long) mhp.n_embd,
                (long long) mhp.n_expert, (long long) mhp.n_used, toks.size());
        mimo = std::make_unique<moex::MimoRef>(f, mhp, ctx_for(), threads);
        step = [&](int32_t t, int64_t p, std::vector<float>& l, moex::Recorder* r) { mimo->step(t, p, l, r); };
    } else {
        fprintf(stderr, "moex_ref: no reference for architecture %s\n", arch.c_str());
        return 1;
    }

    moex::Recorder rec;
    if (!out_dir.empty() && !rec.open(out_dir, match, err)) {
        fprintf(stderr, "moex_ref: %s\n", err.c_str());
        return 1;
    }
    if (!oracle.empty() && !rec.attach_oracle(oracle, err)) {
        fprintf(stderr, "moex_ref: %s\n", err.c_str());
        return 1;
    }
    std::vector<float> logits;
    for (size_t i = 0; i < toks.size(); ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        rec.set_step((int) i);
        step(toks[i], (int64_t) i, logits, rec.active() ? &rec : nullptr);
        if (rec.active()) rec.f32("logits", -1, logits.data(), (int64_t) logits.size());
        const auto best = std::max_element(logits.begin(), logits.end()) - logits.begin();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "step %3zu  token %6d  -> argmax %6lld  (%.4f)  %.0f ms\n", i, toks[i], (long long) best,
                logits[best], ms);
    }
    int bad = 0;
    if (!oracle.empty()) {
        bad = rec.report(tol, stdout);
        printf("%d op outputs over tol %.1e\n", bad, tol);
    }
    rec.close();
    return bad ? 1 : 0;
}
