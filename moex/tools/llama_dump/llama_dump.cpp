// moex/tools/llama_dump/llama_dump.cpp - llama.cpp as an oracle: every named graph tensor, per decode step, to disk.
//
//     llama_dump -m model.gguf -o DIR [-p "text" | -f FILE | -t 1,2,3] [--batch] [--match REGEX] [--kv f32|f16]
//     llama_dump -m model.gguf --tokenize TEXT | --detokenize IDS      (vocabulary only)
//                [--fa on|off|auto] [--threads N] [--ctx N] [--no-repack]
//
// Default is TOKEN-BY-TOKEN decode, one llama_decode per token: that is the graph a decoder runs (the
// autoregressive delta-net, the decode-sized indexer paths), so every record is what one step of an engine
// must reproduce.  --batch submits the whole prompt as one batch instead: the prefill graph (chunked delta-net).
//
// Output, in DIR:
//   dump.bin    the records' payloads back to back (f32 for F32/F16 tensors, i32 for I32), C order of ggml's ne
//               (ne[0] fastest);
//   index.tsv   step  name  type(f32|i32)  ne0 ne1 ne2 ne3  byte_offset  - one line per record;
//   tokens.txt  the token ids, one line;
//   logits are recorded per step under the name "logits" ([n_vocab], the step's last token).
//
// The CPU backend only (the model loads with n_gpu_layers = 0): the oracle must not need a GPU, and every
// tensor is then in host memory, so a record is a strided read of t->data.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

namespace {

struct Dump {
    FILE* bin = nullptr;
    FILE* idx = nullptr;
    std::regex match;
    int step = 0;
    uint64_t off = 0;
    std::vector<float> fbuf;
    std::vector<int32_t> ibuf;
    int64_t n_records = 0;
};

void write_record(Dump& d, const char* name, const char* type, const int64_t ne[4], const void* data, size_t bytes) {
    fwrite(data, 1, bytes, d.bin);
    fprintf(d.idx, "%d\t%s\t%s\t%lld\t%lld\t%lld\t%lld\t%llu\n", d.step, name, type, (long long) ne[0],
            (long long) ne[1], (long long) ne[2], (long long) ne[3], (unsigned long long) d.off);
    d.off += bytes;
    d.n_records++;
}

bool on_eval(struct ggml_tensor* t, bool ask, void* user) {
    Dump& d = *static_cast<Dump*>(user);
    const bool want = (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_I32) &&
                      t->name[0] != '\0' && std::regex_search(t->name, d.match);
    if (ask) return want;
    if (!want) return true;
    if (t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer)) {
        fprintf(stderr, "llama_dump: %s is not in host memory - skipped\n", t->name);
        return true;
    }
    const int64_t n = ggml_nelements(t);
    const char* base = static_cast<const char*>(t->data);
    if (t->type == GGML_TYPE_I32) {
        d.ibuf.resize(n);
        int64_t i = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
            for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
                for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0)
                        d.ibuf[i++] = *reinterpret_cast<const int32_t*>(base + i0 * t->nb[0] + i1 * t->nb[1] +
                                                                         i2 * t->nb[2] + i3 * t->nb[3]);
        write_record(d, t->name, "i32", t->ne, d.ibuf.data(), n * sizeof(int32_t));
    } else {
        d.fbuf.resize(n);
        int64_t i = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
            for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
                for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                        const char* p = base + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                        d.fbuf[i++] = t->type == GGML_TYPE_F32 ? *reinterpret_cast<const float*>(p)
                                                               : ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(p));
                    }
        write_record(d, t->name, "f32", t->ne, d.fbuf.data(), n * sizeof(float));
    }
    return true;
}

std::vector<llama_token> parse_tokens(const std::string& s) {
    std::vector<llama_token> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        out.push_back((llama_token) std::stol(s.substr(i, j - i)));
        i = j + 1;
    }
    return out;
}

void usage() {
    fprintf(stderr,
            "usage: llama_dump -m model.gguf -o DIR [-p TEXT | -f FILE | -t ID,ID,...] [--batch] [--match REGEX]\n"
            "                  [--kv f32|f16] [--fa on|off|auto] [--threads N] [--ctx N] [--no-repack]\n");
}

}  // namespace

// --tokenize TEXT / --detokenize IDS: the model's vocabulary only (no weights), for checking engines that work on ids.
static int vocab_mode(const std::string& model_path, const std::string& mode, const std::string& arg) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "llama_dump: vocab load failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (mode == "tok") {
        std::vector<llama_token> t(arg.size() + 16);
        const int n = llama_tokenize(vocab, arg.c_str(), (int) arg.size(), t.data(), (int) t.size(), false, true);
        for (int i = 0; i < n; ++i) printf("%s%d", i ? "," : "", t[i]);
        printf("\n");
    } else {
        std::vector<llama_token> ids = parse_tokens(arg);
        std::string text;
        char buf[512];
        for (llama_token id : ids) {
            const int n = llama_token_to_piece(vocab, id, buf, sizeof buf, 0, true);
            if (n > 0) text.append(buf, n);
        }
        printf("%s\n", text.c_str());
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 5 && (std::string(argv[3]) == "--tokenize" || std::string(argv[3]) == "--detokenize") &&
        std::string(argv[1]) == "-m")
        return vocab_mode(argv[2], std::string(argv[3]) == "--tokenize" ? "tok" : "detok", argv[4]);
    std::string model_path, out_dir, prompt, tok_list, match = ".", kv = "f32", fa = "off";
    bool batch = false, no_repack = false;
    int threads = 4, n_ctx = 0, n_gen = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); exit(2); }
            return argv[++i];
        };
        if (a == "-m") model_path = next();
        else if (a == "-o") out_dir = next();
        else if (a == "-p") prompt = next();
        else if (a == "-f") {
            FILE* pf = fopen(next().c_str(), "rb");
            if (!pf) { fprintf(stderr, "llama_dump: cannot read the prompt file\n"); return 1; }
            char buf[65536];
            size_t got;
            while ((got = fread(buf, 1, sizeof buf, pf)) > 0) prompt.append(buf, got);
            fclose(pf);
        }
        else if (a == "-t") tok_list = next();
        else if (a == "--batch") batch = true;
        else if (a == "--match") match = next();
        else if (a == "--kv") kv = next();
        else if (a == "--fa") fa = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--ctx") n_ctx = std::stoi(next());
        else if (a == "--no-repack") no_repack = true;
        else if (a == "--gen") n_gen = std::stoi(next());   // then N greedy tokens, printed to stdout
        else { usage(); return 2; }
    }
    if (model_path.empty() || out_dir.empty() || (prompt.empty() == tok_list.empty())) { usage(); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    if (no_repack) mp.use_extra_bufts = false;   // no CPU_REPACK weights: mul_mat is vec_dot per row
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "llama_dump: model load failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks;
    if (!tok_list.empty()) {
        toks = parse_tokens(tok_list);
    } else {
        toks.resize(prompt.size() + 16);
        int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), false, true);
        if (n < 0) { fprintf(stderr, "llama_dump: tokenize failed\n"); return 1; }
        toks.resize(n);
    }
    const int n_tok = (int) toks.size();

    Dump d;
    d.match = std::regex(match);
    std::string cmd = "mkdir -p '" + out_dir + "'";
    if (system(cmd.c_str()) != 0) return 1;
    d.bin = fopen((out_dir + "/dump.bin").c_str(), "wb");
    d.idx = fopen((out_dir + "/index.tsv").c_str(), "w");
    FILE* tf = fopen((out_dir + "/tokens.txt").c_str(), "w");
    if (!d.bin || !d.idx || !tf) { fprintf(stderr, "llama_dump: cannot write to %s\n", out_dir.c_str()); return 1; }
    for (int i = 0; i < n_tok; ++i) fprintf(tf, "%s%d", i ? "," : "", toks[i]);
    fprintf(tf, "\n");
    fclose(tf);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx > 0 ? n_ctx : std::max(256, ((n_tok + 64 + 255) / 256) * 256);
    cp.n_batch = batch ? std::max(n_tok, 1) : 1;
    cp.n_ubatch = cp.n_batch;
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.cb_eval = on_eval;
    cp.cb_eval_user_data = &d;
    cp.type_k = kv == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32;
    cp.type_v = cp.type_k;
    cp.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                       : fa == "auto" ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf = true;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "llama_dump: context init failed\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto record_logits = [&](int i) {
        if (!std::regex_search("logits", d.match)) return;   // --match excludes them too
        const float* lg = llama_get_logits_ith(ctx, i);
        int64_t ne[4] = {n_vocab, 1, 1, 1};
        write_record(d, "logits", "f32", ne, lg, (size_t) n_vocab * sizeof(float));
    };

    if (batch) {
        llama_batch b = llama_batch_init(n_tok, 0, 1);
        for (int i = 0; i < n_tok; ++i) {
            b.token[i] = toks[i];
            b.pos[i] = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
        b.n_tokens = n_tok;
        d.step = 0;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "llama_dump: decode failed\n"); return 1; }
        for (int i = 0; i < n_tok; ++i) { d.step = i; record_logits(i); }
        llama_batch_free(b);
    } else {
        for (int i = 0; i < n_tok; ++i) {
            d.step = i;
            llama_batch b = llama_batch_get_one(&toks[i], 1);
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "llama_dump: decode failed at %d\n", i); return 1; }
            record_logits(-1);
        }
    }
    if (n_gen > 0) {   // greedy continuation (ties to the lowest id), one token per decode
        const llama_vocab* gv = llama_model_get_vocab(model);
        const int nv = llama_vocab_n_tokens(gv);
        std::string outs;
        for (int g = 0; g < n_gen; ++g) {
            const float* lg = llama_get_logits_ith(ctx, -1);
            int best = 0;
            for (int v = 1; v < nv; ++v)
                if (lg[v] > lg[best]) best = v;
            outs += (g ? "," : "") + std::to_string(best);
            if (g + 1 == n_gen) break;
            d.step = n_tok + g;
            llama_token t = best;
            llama_batch b = llama_batch_get_one(&t, 1);
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "llama_dump: decode failed in generation\n"); return 1; }
            record_logits(-1);
        }
        printf("%s\n", outs.c_str());
    }
    fclose(d.bin);
    fclose(d.idx);
    fprintf(stderr, "llama_dump: %d tokens, %lld records, %.1f MiB in %s\n", n_tok, (long long) d.n_records,
            d.off / 1048576.0, out_dir.c_str());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
