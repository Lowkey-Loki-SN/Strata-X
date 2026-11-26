// moex/tools/dsv41/engram_hash_dump.cpp - test driver for engram_hash.hpp (used by test_engram_hash.py).
//
//   engram_hash_dump TABLES IN OUT
// IN: u32 n_seq, then per sequence u32 n, u32 prefill, i32 tokens[n], u8 dead[n]. The first `prefill` tokens go in
// one append, the rest one token per append (decode). OUT: the int64 rows of every position, in order.
#include "engram_hash.hpp"

#include <cstdio>
#include <memory>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s TABLES IN OUT\n", argv[0]); return 2; }
    const dsv41::EngramTables t = dsv41::EngramTables::load(argv[1]);
    FILE* in = std::fopen(argv[2], "rb");
    FILE* out = std::fopen(argv[3], "wb");
    if (!in || !out) { std::perror("open"); return 1; }
    uint32_t n_seq = 0;
    if (std::fread(&n_seq, 4, 1, in) != 1) return 1;
    const size_t per = size_t(t.n_layers) * t.cols();
    for (uint32_t s = 0; s < n_seq; ++s) {
        uint32_t n = 0, prefill = 0;
        if (std::fread(&n, 4, 1, in) != 1 || std::fread(&prefill, 4, 1, in) != 1) return 1;
        std::vector<int32_t> tok(n);
        std::vector<uint8_t> dead8(n);
        if (std::fread(tok.data(), 4, n, in) != n || std::fread(dead8.data(), 1, n, in) != n) return 1;
        std::unique_ptr<bool[]> dead(new bool[n]);
        for (uint32_t i = 0; i < n; ++i) dead[i] = dead8[i] != 0;
        dsv41::EngramHasher h(t);
        std::vector<int64_t> rows(size_t(n) * per);
        h.append(tok.data(), dead.get(), prefill, rows.data());
        for (uint32_t i = prefill; i < n; ++i) h.append(&tok[i], &dead[i], 1, rows.data() + size_t(i) * per);
        std::fwrite(rows.data(), 8, rows.size(), out);
    }
    std::fclose(out);
    std::fclose(in);
    return 0;
}
