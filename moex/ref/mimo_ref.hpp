// moex/ref/mimo_ref.hpp - MiMo-V2.6-Flash (GGUF arch `mimo2`) decode, one token at a time, on the CPU.
//
// THE SPECIFICATION IS LLAMA.CPP'S GRAPH: `src/models/mimo2.cpp` in ~/AI/llama.cpp-glm5next-mtp (c92b4ae77), on the
// CPU backend without flash attention.  A plain pre-norm decoder: RMSNorm; a fused QKV projection (Q/K heads of
// 192 with NEOX RoPE on the first 64 dims, V heads of 128); GQA attention over the full context (the 9 global
// layers, RoPE base 1e7) or a 128-token window (the SWA layers, base 1e4) with a learned per-head sink logit;
// the output projection scaled by 0.707; the residual; RMSNorm; a dense SwiGLU FFN (layer 0) or a sigmoid-routed
// top-8-of-256 MoE with a selection bias and normalized weights (no shared expert, no clamp); the residual.
// Projections run through ggml-cpu's vec_dot with the activation quantized to the weight's vec_dot_type, so
// against `llama_dump` the two differ in summation order only (ref_common.hpp).
#pragma once

#include "gguf_file.hpp"
#include "recorder.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace moex {

struct MimoHparams {
    int64_t n_embd = 0, n_layer = 0, n_layer_all = 0, n_vocab = 0;
    int64_t n_head = 0, head_k = 0, head_v = 0, n_rot = 0, n_swa = 0, n_ff = 0;
    int64_t n_expert = 0, n_used = 0, n_ff_exp = 0;
    float rms_eps = 0, rope_base = 0, rope_base_swa = 0, v_scale = 0, ew_scale = 0;
    std::vector<int64_t> head_kv;   ///< per layer (incl. the NextN layers)
    std::vector<int64_t> swa;       ///< per layer: 1 = sliding window
    bool is_swa(int64_t il) const { return swa[(size_t) il] != 0; }

    bool load(const GgufFile& f, std::string& err);
};

class MimoRef {
public:
    MimoRef(GgufFile& f, const MimoHparams& hp, int64_t n_ctx, int threads);
    /// One token at position `pos` (one sequence from 0).  Fills `logits` (n_vocab).
    void step(int32_t token, int64_t pos, std::vector<float>& logits, Recorder* rec);

private:
    GgufFile& f_;
    MimoHparams hp_;
    int64_t n_ctx_;
    int threads_;
    Recorder* rec_ = nullptr;

    struct Kv {
        std::vector<float> k, v;   ///< (n_ctx, n_head_kv, head_k) / (n_ctx, n_head_kv, head_v), f32 (--kv f32)
    };
    std::vector<Kv> kv_;

    std::string tn(int64_t il, const char* suffix) const;
    void matvec(const std::string& name, const float* x, float* y);
    void attention(int64_t il, int64_t pos, const float* cur, float* out);
    void swiglu_ffn(int64_t il, const std::string& gate, const std::string& up, const std::string& down, const float* cur,
                    float* out);
    void moe_ffn(int64_t il, const float* cur, float* out);
};

}  // namespace moex
