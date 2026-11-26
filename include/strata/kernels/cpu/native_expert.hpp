// include/strata/kernels/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows], and the arithmetic is ggml-cpu's
// own (`ggml_get_type_traits_cpu`): the activation is quantized with the weight type's `vec_dot_type` and each row
// is one `vec_dot`, exactly what llama.cpp's CPU backend computes for the same tensor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels::cpu {

/// Bytes of the largest quantized activation any native layer uses.  Upstream sized it for 2560 values as Q8_K
/// (10 x 292 = 2920); moex: 8192 covers n_embd 4096 (GLM-5.3-Flash, DeepSeek-V4-Flash, MiMo-V2.6: 16 x 292 = 4672).
inline constexpr size_t kNativeActBytes = 8192;
/// Bytes of the largest quantized down activation.  Upstream: 640 values (Q8_0 20 x 34, Q8_K 3 x 292); moex: 4096
/// covers n_ff 2048 (8 x 292 = 2336 as Q8_K, 64 x 34 = 2176 as Q8_0) and DeepSeek-V4.1's 2304 (2628 / 2448).
inline constexpr size_t kNativeHBytes = 4096;

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, n_ff of d_act)
    /// moex: the SwiGLU clamp of GLM-5.3-Flash / DeepSeek-V4 (`swiglu_clamp_exp`): silu(min(g, L)) * clamp(u, -L, L),
    /// as ggml's swiglu_clamp.  0 (every upstream pack): silu(g) * u.  Set after native_fmt; a layer with a clamp
    /// takes the AVX-512 multi-token kernel (it applies the clamp) or ggml-cpu's rows (the AVX-2 kernels have none).
    float swiglu_limit = 0.0f;
};

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err);

/// x (n_embd floats) -> the gate/up activation (act_bytes).
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes).
void native_quant_h(const NativeFmt& f, const float* h, void* dst);

/// From how many tokens native_gu_rows gives this gate/up type to a multi-token kernel (#152; ggml-cpu's per-token dot
/// below that).  1: a token's rows are the same alone and in any group.
int native_gu_mt_min(int gu_type);
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows r in [r0, r1), `nt` tokens.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);
/// moex: native_gu_rows with the gate and up rows at separate addresses (an expert in place in a GGUF mapping);
/// ggml-cpu's rows, the clamp when `f.swiglu_limit` is set.  Bitwise the blob form's ggml-cpu path.
void native_gu_rows_split(const NativeFmt& f, const uint8_t* gate, const uint8_t* up, const void* const* act, int nt,
                          float* const* ff, int r0, int r1);
/// moex: native_down_rows with the down rows at `down` (no `down_off`).
void native_down_rows_at(const NativeFmt& f, const uint8_t* down, const void* const* hq, int nt, float* const* out,
                         int r0, int r1);

}  // namespace strata::kernels::cpu
