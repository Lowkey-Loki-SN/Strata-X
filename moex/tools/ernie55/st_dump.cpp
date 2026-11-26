// moex/tools/ernie55/st_dump.cpp - every tensor of a Hugging Face checkpoint as strata-x's reader sees it
// (include/strata/x/safetensors.hpp): one line each - name, shard, dtype, shape, absolute offset, bytes - sorted by name,
// for inventory.py --check to compare with its own reading of the same headers. A partial download is allowed (its
// shards not yet whole are counted and their tensors still listed when their header is on disk).
//
//   g++ -O2 -std=c++20 -Iinclude src/x/json_lite.cpp src/x/safetensors.cpp moex/tools/ernie55/st_dump.cpp -o st_dump
//   ./st_dump MODEL_DIR > tensors.tsv
#include "strata/x/safetensors.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    strata::st::Checkpoint ck;
    std::string err;
    if (!ck.open(argv[1], err, true)) {
        std::fprintf(stderr, "st_dump: %s\n", err.c_str());
        return 1;
    }
    std::vector<const std::pair<const std::string, strata::st::Tensor>*> v;
    for (const auto& kv : ck.tensors()) v.push_back(&kv);
    std::sort(v.begin(), v.end(), [](auto* a, auto* b) { return a->first < b->first; });
    for (const auto* kv : v) {
        const auto& t = kv->second;
        std::string sh;
        for (size_t i = 0; i < t.shape.size(); ++i) sh += (i ? "x" : "") + std::to_string(t.shape[i]);
        std::printf("%s\t%s\t%s\t%s\t%llu\t%llu\n", kv->first.c_str(), ck.files()[(size_t) t.file].c_str(), strata::st::dtype_name(t.dtype),
                    sh.c_str(), (unsigned long long) t.offset, (unsigned long long) t.nbytes);
    }
    std::fprintf(stderr, "st_dump: %zu shards (%d not whole yet), %zu tensors, %.2f GB\n", ck.files().size(), ck.n_partial(),
                 ck.tensors().size(), (double) ck.total_bytes() / 1e9);
    return 0;
}
