// moex/tools/dsv41/engram_hash.hpp - DeepSeek-V4.1 engram row numbers from token ids (header-only).
//
// The release's NgramHashState (inference/engram.py) for one sequence: every position hashes the 2..max_ngram
// grams of compressed ids ending there; the look-back stops at the sequence start and at dead tokens (image
// spans), whose slots and all older ones become the compressed pad id. Row = offset + (xor-fold of id * multiplier)
// mod prime, per (layer, n-gram size, head). The fixed tables come from engram_tables.py.
//
// The row numbers depend only on token ids, so the engine computes them as soon as a token is sampled or drafted
// and reads the rows from the SSD ahead of the engram layers.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsv41 {

struct EngramTables {
    uint32_t n_vocab = 0, n_compressed = 0, n_layers = 0, max_ngram = 0, n_heads = 0, head_dim = 0;
    int32_t pad_id = 0;
    std::vector<uint32_t> layer_ids;
    std::vector<uint64_t> num_embeddings;
    std::vector<int64_t> multipliers;  // [n_layers][max_ngram]
    std::vector<int64_t> primes;       // [n_layers][max_ngram - 1][n_heads]
    std::vector<int64_t> offsets;      // [n_layers][(max_ngram - 1) * n_heads]
    std::vector<int32_t> token_map;    // [n_vocab]

    uint32_t cols() const { return (max_ngram - 1) * n_heads; }  // rows per position per layer

    static EngramTables load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("engram tables: cannot open " + path);
        auto rd = [&](void* p, size_t n) {
            if (std::fread(p, 1, n, f) != n) { std::fclose(f); throw std::runtime_error("engram tables: short " + path); }
        };
        char magic[8]; uint32_t version = 0;
        rd(magic, 8); rd(&version, 4);
        if (std::memcmp(magic, "DSV41ENG", 8) != 0 || version != 1) {
            std::fclose(f); throw std::runtime_error("engram tables: bad header in " + path);
        }
        EngramTables t;
        rd(&t.n_vocab, 4); rd(&t.n_compressed, 4); rd(&t.pad_id, 4); rd(&t.n_layers, 4);
        rd(&t.max_ngram, 4); rd(&t.n_heads, 4); rd(&t.head_dim, 4);
        if (t.n_layers == 0 || t.n_layers > 64 || t.max_ngram < 2 || t.max_ngram > 16 || t.n_heads == 0 ||
            t.n_heads > 64 || t.n_vocab == 0 || t.n_vocab > (1u << 24)) {
            std::fclose(f); throw std::runtime_error("engram tables: implausible sizes in " + path);
        }
        t.layer_ids.resize(t.n_layers); rd(t.layer_ids.data(), 4 * t.n_layers);
        t.num_embeddings.resize(t.n_layers); rd(t.num_embeddings.data(), 8 * t.n_layers);
        t.multipliers.resize(size_t(t.n_layers) * t.max_ngram); rd(t.multipliers.data(), 8 * t.multipliers.size());
        t.primes.resize(size_t(t.n_layers) * t.cols()); rd(t.primes.data(), 8 * t.primes.size());
        t.offsets.resize(size_t(t.n_layers) * t.cols()); rd(t.offsets.data(), 8 * t.offsets.size());
        t.token_map.resize(t.n_vocab); rd(t.token_map.data(), 4 * t.token_map.size());
        std::fclose(f);
        for (uint32_t l = 0; l < t.n_layers; ++l) {  // every row a hash can produce lies inside the table
            for (uint32_t c = 0; c < t.cols(); ++c) {
                const int64_t p = t.primes[size_t(l) * t.cols() + c], o = t.offsets[size_t(l) * t.cols() + c];
                if (p <= 0 || o < 0 || uint64_t(o + p) > t.num_embeddings[l])
                    throw std::runtime_error("engram tables: a bucket range leaves the table in " + path);
            }
        }
        return t;
    }
};

// One sequence's look-back state: the compressed ids of every position so far (kDead for image tokens).
class EngramHasher {
public:
    static constexpr int32_t kDead = -1;

    explicit EngramHasher(const EngramTables& t) : t_(t) {}

    void reset() { ids_.clear(); }
    size_t size() const { return ids_.size(); }
    void truncate(size_t n) { if (n < ids_.size()) ids_.resize(n); }  // drafts rejected, or a checkpoint restored

    // Append `n` tokens (dead[i] true for image tokens) and write their rows: out[i][layer][col], col = (n-gram
    // size - 2) * n_heads + head, as in the reference's [B, L, n_layers, n_hash_cols].
    void append(const int32_t* tokens, const bool* dead, size_t n, int64_t* out) {
        const size_t start = ids_.size();
        for (size_t i = 0; i < n; ++i) {
            const int32_t tok = tokens[i];
            if (!dead || !dead[i]) {
                if (tok < 0 || uint32_t(tok) >= t_.n_vocab) throw std::runtime_error("engram: token id out of range");
                ids_.push_back(t_.token_map[tok]);
            } else {
                ids_.push_back(kDead);
            }
        }
        const uint32_t M = t_.max_ngram, C = t_.cols(), L = t_.n_layers, H = t_.n_heads;
        std::vector<int64_t> slot(M);
        for (size_t i = 0; i < n; ++i) {
            const size_t pos = start + i;
            bool blocked = false;
            for (uint32_t s = 0; s < M; ++s) {  // the reference clamps pos - s at 0 and blocks it via pos < s
                const int32_t src = pos >= s ? ids_[pos - s] : ids_[0];
                blocked = blocked || pos < s || src == kDead;
                slot[s] = blocked ? t_.pad_id : src;
            }
            int64_t* o = out + i * size_t(L) * C;
            for (uint32_t l = 0; l < L; ++l) {
                const int64_t* mul = &t_.multipliers[size_t(l) * M];
                int64_t rolling = slot[0] * mul[0];
                for (uint32_t g = 1; g < M; ++g) {
                    rolling ^= slot[g] * mul[g];
                    for (uint32_t h = 0; h < H; ++h) {
                        const size_t c = size_t(g - 1) * H + h;
                        const int64_t p = t_.primes[size_t(l) * C + c];
                        int64_t r = rolling % p;  // torch's % is floor-mod; rolling is never negative here
                        if (r < 0) r += p;
                        o[size_t(l) * C + c] = t_.offsets[size_t(l) * C + c] + r;
                    }
                }
            }
        }
    }

private:
    const EngramTables& t_;
    std::vector<int32_t> ids_;
};

}  // namespace dsv41
