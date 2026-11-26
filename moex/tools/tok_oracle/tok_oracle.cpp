// moex/tools/tok_oracle/tok_oracle.cpp - llama.cpp's own tokenizer as a BATCH oracle (vocabulary only: no weights, no GPU).
//
// `llama_dump --tokenize TEXT` starts a process per string.  This is the same libllama (the glm5next-mtp tree's CPU build,
// see moex/ref/build.sh) behind a pipe: the vocabulary loads once, then any number of requests are answered, and every
// llama_tokenize flag is open (llama_dump fixes add_special=false, parse_special=true).
//
//   tok_oracle -m model.gguf          requests on stdin, answers on stdout, llama.cpp's loader log on stderr
//
// Frames (binary safe).  A request is one header line, then (where it says so) N payload bytes:
//   TOK <add_special 0|1> <parse_special 0|1> <nbytes>\n<utf-8 bytes>   -> "id,id,id\n"  (an empty line: no tokens)
//   PIECE <id> <special 0|1>\n                                          -> "<nbytes>\n<bytes>"   (llama_token_to_piece, lstrip 0)
//   DETOK <remove_special 0|1> <unparse_special 0|1> <nbytes>\n<id,id,..>  -> "<nbytes>\n<bytes>"   (llama_detokenize)
//   INFO\n                                                              -> "key=value" lines, then "END\n"
//   ATTRS\n                                                             -> one line per token "id attr eog hex(text)", then "END\n"
//   QUIT\n
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<llama_token> parse_ids(const std::string & s) {
    std::vector<llama_token> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back((llama_token) std::stol(s.substr(i, j - i)));
        i = j + 1;
    }
    return out;
}

static bool read_exact(std::string & dst, size_t n) {
    dst.resize(n);
    size_t got = 0;
    while (got < n) {
        size_t r = fread(&dst[got], 1, n - got, stdin);
        if (r == 0) return false;
        got += r;
    }
    return true;
}

static bool read_line(std::string & line) {
    line.clear();
    int c;
    while ((c = fgetc(stdin)) != EOF) {
        if (c == '\n') return true;
        line.push_back((char) c);
    }
    return !line.empty();
}

int main(int argc, char ** argv) {
    std::string model_path;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-m" && i + 1 < argc) model_path = argv[++i];
        else { fprintf(stderr, "usage: tok_oracle -m model.gguf\n"); return 2; }
    }
    if (model_path.empty()) { fprintf(stderr, "usage: tok_oracle -m model.gguf\n"); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "tok_oracle: vocab load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    std::string line, payload;
    while (read_line(line)) {
        char cmd[16] = {0};
        sscanf(line.c_str(), "%15s", cmd);
        if (!strcmp(cmd, "QUIT")) break;
        if (!strcmp(cmd, "TOK")) {
            int add_special = 0, parse_special = 0;
            unsigned long nbytes = 0;
            sscanf(line.c_str(), "TOK %d %d %lu", &add_special, &parse_special, &nbytes);
            if (!read_exact(payload, nbytes)) break;
            std::vector<llama_token> toks(payload.size() + 16);
            int32_t n = llama_tokenize(vocab, payload.data(), (int32_t) payload.size(), toks.data(), (int32_t) toks.size(),
                                       add_special != 0, parse_special != 0);
            if (n < 0) {
                toks.resize((size_t) -n);
                n = llama_tokenize(vocab, payload.data(), (int32_t) payload.size(), toks.data(), (int32_t) toks.size(),
                                   add_special != 0, parse_special != 0);
            }
            std::string out;
            for (int32_t i = 0; i < n; ++i) {
                if (i) out.push_back(',');
                out += std::to_string(toks[i]);
            }
            out.push_back('\n');
            fwrite(out.data(), 1, out.size(), stdout);
            fflush(stdout);
        } else if (!strcmp(cmd, "PIECE")) {
            int id = 0, special = 1;
            sscanf(line.c_str(), "PIECE %d %d", &id, &special);
            std::string buf(512, '\0');
            int32_t n = llama_token_to_piece(vocab, id, &buf[0], (int32_t) buf.size(), 0, special != 0);
            if (n < 0) {
                buf.resize((size_t) -n);
                n = llama_token_to_piece(vocab, id, &buf[0], (int32_t) buf.size(), 0, special != 0);
            }
            if (n < 0) n = 0;
            printf("%d\n", n);
            fwrite(buf.data(), 1, (size_t) n, stdout);
            fflush(stdout);
        } else if (!strcmp(cmd, "DETOK")) {
            int rm = 0, unparse = 1;
            unsigned long nbytes = 0;
            sscanf(line.c_str(), "DETOK %d %d %lu", &rm, &unparse, &nbytes);
            if (!read_exact(payload, nbytes)) break;
            std::vector<llama_token> ids = parse_ids(payload);
            std::string buf(ids.size() * 64 + 64, '\0');
            int32_t n = llama_detokenize(vocab, ids.data(), (int32_t) ids.size(), &buf[0], (int32_t) buf.size(), rm != 0, unparse != 0);
            if (n < 0) {
                buf.resize((size_t) -n);
                n = llama_detokenize(vocab, ids.data(), (int32_t) ids.size(), &buf[0], (int32_t) buf.size(), rm != 0, unparse != 0);
            }
            if (n < 0) n = 0;
            printf("%d\n", n);
            fwrite(buf.data(), 1, (size_t) n, stdout);
            fflush(stdout);
        } else if (!strcmp(cmd, "INFO")) {
            printf("n_vocab=%d\n", n_vocab);
            printf("vocab_type=%d\n", (int) llama_vocab_type(vocab));
            printf("bos=%d\neos=%d\neot=%d\nsep=%d\nnl=%d\npad=%d\nmask=%d\n", llama_vocab_bos(vocab), llama_vocab_eos(vocab),
                   llama_vocab_eot(vocab), llama_vocab_sep(vocab), llama_vocab_nl(vocab), llama_vocab_pad(vocab),
                   llama_vocab_mask(vocab));
            printf("add_bos=%d\nadd_eos=%d\nadd_sep=%d\n", (int) llama_vocab_get_add_bos(vocab), (int) llama_vocab_get_add_eos(vocab),
                   (int) llama_vocab_get_add_sep(vocab));
            printf("fim_pre=%d\nfim_suf=%d\nfim_mid=%d\nfim_pad=%d\nfim_rep=%d\nfim_sep=%d\n", llama_vocab_fim_pre(vocab),
                   llama_vocab_fim_suf(vocab), llama_vocab_fim_mid(vocab), llama_vocab_fim_pad(vocab), llama_vocab_fim_rep(vocab),
                   llama_vocab_fim_sep(vocab));
            std::string eog;
            for (int32_t i = 0; i < n_vocab; ++i) {
                if (llama_vocab_is_eog(vocab, i)) { if (!eog.empty()) eog.push_back(','); eog += std::to_string(i); }
            }
            printf("eog=%s\n", eog.c_str());
            printf("END\n");
            fflush(stdout);
        } else if (!strcmp(cmd, "ATTRS")) {
            for (int32_t i = 0; i < n_vocab; ++i) {
                const char * t = llama_vocab_get_text(vocab, i);
                printf("%d %d %d ", i, (int) llama_vocab_get_attr(vocab, i), (int) llama_vocab_is_eog(vocab, i));
                for (const char * p = t; p && *p; ++p) printf("%02x", (unsigned char) *p);
                printf("\n");
            }
            printf("END\n");
            fflush(stdout);
        } else {
            fprintf(stderr, "tok_oracle: unknown command: %s\n", line.c_str());
            printf("ERR\n");
            fflush(stdout);
        }
    }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
