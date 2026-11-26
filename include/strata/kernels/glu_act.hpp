// include/strata/kernels/glu_act.hpp - moex (ERNIE-5.5-Flash): the gated activations an expert path computes.
//
// The routed experts' activation travels through every expert path (the GPU's VRAM experts, the prompt path's grouped
// products, the CPU pool's rows) as one float, the "SwiGLU limit": 0 = SwiGLU, L > 0 = GLM's / DeepSeek's clamped SwiGLU
// at L, kSituGlu = ERNIE-5.5's SiTU-GLU (vLLM activation.py SiTUAndMul, vLLM-ERNIE ba322a5):
//   beta tanh(g / beta) sigmoid(g) * lin_beta tanh(u / lin_beta),  beta 4, lin_beta 25
// - the release's activation_situ_beta / activation_situ_linear_beta, which the engine checks at load.  A path that sees a
// negative limit computes SiTU-GLU; every other value keeps the path's code as it was.
#pragma once

#include <cmath>

namespace strata::kernels {

constexpr float kSituGlu = -1.0f;
constexpr float kSituBeta = 4.0f;
constexpr float kSituLinBeta = 25.0f;

/// SiTU-GLU on the host, in the reference's order: ((beta tanh(g / beta)) sigmoid(g)) (lin_beta tanh(u / lin_beta))
inline float situ_glu(float g, float u) {
    const float a = kSituBeta * std::tanh(g / kSituBeta);
    const float sg = 1.0f / (1.0f + std::exp(-g));
    const float b = kSituLinBeta * std::tanh(u / kSituLinBeta);
    return a * sg * b;
}

}  // namespace strata::kernels
