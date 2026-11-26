// moex/ref/glm_ref.hpp - GLM-5.3-Flash (GGUF arch `glm5next`) decode, one token at a time, on the CPU.
//
// THE SPECIFICATION IS LLAMA.CPP'S GRAPH FOR THIS ARCHITECTURE, AS IT RUNS BY DEFAULT on the CPU backend:
// `src/models/glm5-next.cpp` plus the fused kernels its context selects (`gated_delta_net`,
// `dsv4_hc_{pre,comb,post}`, `lightning_indexer`) in the `~/AI/llama.cpp-glm5next-mtp` tree.  Each step below
// names the line it transcribes.  Every projection runs through ggml-cpu's own `vec_dot` with the activation
// quantized to the weight's `vec_dot_type`, as `ggml_mul_mat` does, so against `llama_dump` the two differ in
// summation order only and a real difference is a SEMANTIC one.  That is the point of this program: the GPU
// port is built against it, kernel by kernel, and it must first be shown to compute what llama.cpp computes.
//
// Records use llama_dump's format and names (`name-<layer>`), so `compare_dumps.py` lines the two up.
#pragma once

#include "gguf_file.hpp"
#include "recorder.hpp"

#include <cstdint>
#include <cstdio>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace moex {

struct GlmHparams {
    int64_t n_embd = 0, n_layer = 0, n_layer_all = 0, n_vocab = 0;
    int64_t n_head = 0;          ///< KDA heads = MLA heads (64)
    int64_t hd_kda = 0;          ///< KDA head dim (128)
    int64_t d_conv = 0;          ///< 4
    int64_t n_dense = 0;         ///< leading dense layers (3)
    int64_t n_ff = 0;            ///< dense FFN width
    int64_t n_expert = 0, n_used = 0, n_ff_exp = 0;
    int64_t q_lora = 0, kv_lora = 0, k_mla = 0, v_mla = 0;
    int64_t idx_heads = 0, idx_dim = 0, idx_topk = 0, kpool = 0;
    bool select_tail = true;     ///< llama.cpp's default when the key is absent (a decode dump shows n_sel = 4k+3)
    int64_t hc = 0;
    int64_t sinkhorn_iters = 0;
    float hc_eps = 0, rms_eps = 0, ln_eps = 0, kda_lb = 0, ew_scale = 1;
    bool ew_norm = false;
    std::vector<int64_t> head_kv;          ///< 0: KDA layer, 1: DSA (MLA) layer
    std::vector<float> clamp_exp, clamp_sh;
    bool is_dsa(int64_t il) const { return head_kv[il] != 0; }

    bool load(const GgufFile& f, std::string& err);
};

class GlmRef {
public:
    GlmRef(GgufFile& f, const GlmHparams& hp, int64_t n_ctx, int threads);
    /// One token at position `pos` (= its cell: one sequence, starting at 0).  Fills `logits` (n_vocab).
    void step(int32_t token, int64_t pos, std::vector<float>& logits, Recorder* rec);

private:
    GgufFile& f_;
    GlmHparams hp_;
    int64_t n_ctx_;
    int threads_;
    Recorder* rec_ = nullptr;

    struct KdaState {
        std::vector<float> conv[3];   ///< per channel (d_conv - 1) samples, oldest first: [c * (d_conv-1) + t]
        std::vector<float> S;         ///< per head M[j][i] = S[i][j] (i key, j value): ggml's transposed layout
    };
    struct DsaState {
        std::vector<float> kv;        ///< (n_ctx, kv_lora): the latent cache, f32 (llama_dump runs with --kv f32)
        std::vector<float> ik, ig;    ///< (n_ctx, idx_dim): indexer key (after its LayerNorm) and pool gate
        std::vector<float> pooled;    ///< (n_ctx / kpool, idx_dim)
    };
    std::vector<KdaState> kda_;
    std::vector<DsaState> dsa_;

    // projections: y = W x, ggml_mul_mat's convention (W is ne0 = n_in contiguous rows, n_out of them)
    void matvec(const uint8_t* w, ggml_type t, int64_t n_in, int64_t n_out, const float* x, float* y) const;
    void matvec(const std::string& name, const float* x, float* y);
    std::string tn(int64_t il, const char* suffix) const;

    struct HcCoef {
        float pre[4], post[4], comb[16];   ///< comb[dst + 4*src]
    };
    HcCoef hc_coef(const std::vector<float>& R, int64_t il, const char* which);
    void hc_pre_apply(const std::vector<float>& R, const HcCoef& c, float* out) const;
    void hc_post(const float* x, const HcCoef& c, std::vector<float>& R) const;

    void kda_layer(int64_t il, const float* cur, float* out);
    void dsa_layer(int64_t il, int64_t pos, const float* cur, float* out);
    void dense_ffn(int64_t il, const float* cur, float* out);
    void moe_ffn(int64_t il, const float* cur, float* out);
    void swiglu_ffn(int64_t il, const std::string& gate, const std::string& up, const std::string& down, float limit,
                    const float* cur, float* out);
};

}  // namespace moex
