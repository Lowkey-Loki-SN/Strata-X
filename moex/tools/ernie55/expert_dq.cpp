// moex/tools/ernie55/expert_dq.cpp - the routed experts of an ernie55 GGUF, dequantized by ggml's own row decoders (the
// weights strata-x computes with), for the reference model's oracle runs (moex/tools/ernie55/ref/oracle_dump.py).
//
//     ernie55_expert_dq MODEL.gguf          (then on stdin: "LAYER EXPERT" lines; "q" or EOF ends)
//
// For each request it writes, on stdout, expert EXPERT of layer LAYER as F32 row-major matrices [out][in]: gate (w1)
// [n_ff][n_lat], up (w3) [n_ff][n_lat], down (w2) [n_lat][n_ff] - preceded by a 16-byte header {n_ff, n_lat, ok, 0}
// (int32; ok 0 = no such expert, no payload).  The model's shards are memory-mapped; nothing is cached.
#include "strata/artifact/gguf_reader.hpp"

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: ernie55_expert_dq MODEL.gguf  (stdin: \"LAYER EXPERT\" lines)\n");
        return 2;
    }
    strata::GgufModel gm = strata::GgufModel::open(argv[1]);
    std::vector<float> buf;
    char line[128];
    while (fgets(line, sizeof line, stdin)) {
        if (line[0] == 'q') break;
        int il = -1, e = -1;
        if (sscanf(line, "%d %d", &il, &e) != 2) continue;
        int32_t hdr[4] = {0, 0, 0, 0};
        const std::string b = "blk." + std::to_string(il) + ".";
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* tg = gm.find(b + "ffn_gate_exps.weight", &sg);
        const strata::TensorInfo* tu = gm.find(b + "ffn_up_exps.weight", &su);
        const strata::TensorInfo* td = gm.find(b + "ffn_down_exps.weight", &sd);
        if (!tg || !tu || !td || tg->shape.size() != 3 || td->shape.size() != 3 || e < 0 || (uint64_t) e >= tg->shape[2]) {
            fwrite(hdr, sizeof hdr, 1, stdout);
            fflush(stdout);
            continue;
        }
        const int64_t n_lat = (int64_t) tg->shape[0], n_ff = (int64_t) tg->shape[1];
        hdr[0] = (int32_t) n_ff;
        hdr[1] = (int32_t) n_lat;
        hdr[2] = 1;
        fwrite(hdr, sizeof hdr, 1, stdout);
        auto put = [&](const strata::TensorInfo* t, size_t sh, int64_t cols, int64_t rows) {
            const ggml_type ty = (ggml_type) t->type;
            const size_t rb = ggml_row_size(ty, cols);
            const uint8_t* base = gm.shard(sh).tensor_data(*t) + (size_t) e * rows * rb;
            buf.resize((size_t) (rows * cols));
            const auto* tr = ggml_get_type_traits(ty);
            // the rows over the threads (ggml's decoders are one row at a time; each row's values are its own)
            const int nth = (int) std::max<unsigned>(1, std::min<unsigned>(16, std::thread::hardware_concurrency()));
            std::vector<std::thread> th;
            for (int w = 0; w < nth; ++w)
                th.emplace_back([&, w] {
                    for (int64_t r = rows * w / nth; r < rows * (w + 1) / nth; ++r) {
                        if (ty == GGML_TYPE_F32) std::memcpy(buf.data() + r * cols, base + r * rb, (size_t) cols * 4);
                        else tr->to_float(base + r * rb, buf.data() + r * cols, cols);
                    }
                });
            for (auto& x : th) x.join();
            fwrite(buf.data(), sizeof(float), buf.size(), stdout);
        };
        put(tg, sg, n_lat, n_ff);
        put(tu, su, n_lat, n_ff);
        put(td, sd, n_ff, n_lat);
        fflush(stdout);
    }
    return 0;
}
