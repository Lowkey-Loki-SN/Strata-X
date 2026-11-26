// moex/ref/ref_common.hpp - the arithmetic the moex CPU references share, as ggml-cpu computes it.
//
// Each function transcribes the ggml-cpu op named beside it (~/AI/llama.cpp-glm5next-mtp, c92b4ae77), so a reference
// built from them differs from `llama_dump` in summation order at most.  glm_ref.cpp keeps its own copies (it was
// validated with them); the references written after it (mimo_ref, dsv4_ref) use these.
#pragma once

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace moex::ref {

// ggml_compute_forward_rms_norm_f32: double sum of float squares, float mean, 1/sqrtf.
inline void rms_norm(const float* x, int64_t n, float eps, float* y) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += (double) (x[i] * x[i]);
    const float mean = (float) (sum / (double) n);
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale;
}

inline void rms_norm_w(const float* x, int64_t n, float eps, const float* w, float* y) {
    rms_norm(x, n, eps, y);
    for (int64_t i = 0; i < n; ++i) y[i] = y[i] * w[i];
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
inline float siluf(float x) { return x / (1.0f + expf(-x)); }

// ggml_compute_forward_swiglu_clamp_f32: silu(min(gate, limit)) * clamp(up, -limit, limit); no limit: silu(g) * u
inline float swiglu(float gate, float up, float limit) {
    if (!(limit > 1e-6f)) return siluf(gate) * up;
    const float g = std::min(gate, limit);
    const float u = std::clamp(up, -limit, limit);
    return g / (1.f + expf(-g)) * u;
}

// ggml_vec_dot_f32 (the F32 x F32 mul_mat's dot)
inline float dot_f32(int64_t n, const float* a, const float* b) {
    float s = 0.0f;
    ggml_get_type_traits_cpu(GGML_TYPE_F32)->vec_dot((int) n, &s, 0, a, 0, b, 0, 1);
    return s;
}

// ggml_vec_sum_f32: a double sum, cast to float
inline float sum_f32(int64_t n, const float* x) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += (double) x[i];
    return (float) s;
}

// ggml_compute_forward_soft_max_f32 over n scaled values with an optional sink logit (ggml_soft_max_add_sinks):
// the max includes the sink, the exps are summed in double, the sink's exp joins the sum, then * (float) 1/sum.
// `w` holds the already scaled and masked scores (-INFINITY for masked).
inline void softmax_sink(float* w, int64_t n, const float* sink) {
    float mx = -INFINITY;
    for (int64_t i = 0; i < n; ++i) mx = std::max(mx, w[i]);
    if (sink) mx = std::max(mx, *sink);
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const float e = w[i] == -INFINITY ? 0.0f : expf(w[i] - mx);
        w[i] = e;
        sum += (double) e;
    }
    if (sink) sum += (double) expf(*sink - mx);
    const float inv = (float) (1.0 / sum);
    for (int64_t i = 0; i < n; ++i) w[i] *= inv;
}

/// ggml's rope cache (ggml_rope_cache_init + rope_yarn, ops.cpp 5951-5992) over `ne0` values (the head's width: the
/// theta keeps multiplying past n_dims) for position p.  cache[i0] = cos, cache[i0 + 1] = sin.
inline void rope_cache(int64_t p, int64_t ne0, int n_dims, float freq_base, float freq_scale, float ext_factor,
                       float attn_factor, const float corr_dims[2], std::vector<float>& cache) {
    cache.assign((size_t) ne0 + 2, 0.0f);
    const float theta_scale = powf(freq_base, -2.0f / (float) n_dims);
    float theta = (float) p;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float theta_extrap = theta;
        const float theta_interp = freq_scale * theta_extrap;
        float th = theta_interp, mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float y = ((float) (i0 / 2) - corr_dims[0]) / std::max(0.001f, corr_dims[1] - corr_dims[0]);
            const float ramp_mix = (1 - std::min(1.0f, std::max(0.0f, y))) * ext_factor;
            th = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        cache[(size_t) i0] = cosf(th) * mscale;
        cache[(size_t) i0 + 1] = sinf(th) * mscale;
        theta *= theta_scale;
    }
}

/// rotate_pairs (ops.cpp 6061): NEOX pairs (i, i + n_dims/2) of the first n_dims values, in place.
inline void rope_neox(float* x, int n_dims, const std::vector<float>& cache) {
    const int half = n_dims / 2;
    for (int i0 = 0; i0 < n_dims; i0 += 2) {
        const int ic = i0 / 2;
        const float c = cache[(size_t) i0], s = cache[(size_t) i0 + 1];
        const float x0 = x[ic], x1 = x[ic + half];
        x[ic] = x0 * c - x1 * s;
        x[ic + half] = x0 * s + x1 * c;
    }
}

/// y = W x with ggml_mul_mat's arithmetic: x quantized to the weight's vec_dot_type, one vec_dot per row.
inline void matvec(const uint8_t* w, ggml_type t, int64_t n_in, int64_t n_out, const float* x, float* y, int threads) {
    const ggml_type_traits_cpu* tt = ggml_get_type_traits_cpu(t);
    const ggml_type vt = tt->vec_dot_type;
    std::vector<uint8_t> xq;
    const void* xa = x;
    if (vt != GGML_TYPE_F32) {
        xq.resize(ggml_row_size(vt, n_in));
        ggml_get_type_traits_cpu(vt)->from_float(x, xq.data(), n_in);
        xa = xq.data();
    }
    const size_t rb = ggml_row_size(t, n_in);
    auto run = [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) tt->vec_dot((int) n_in, &y[r], 0, w + r * rb, 0, xa, 0, 1);
    };
    const int nt = n_out >= 1024 ? std::max(1, threads) : 1;
    if (nt == 1) {
        run(0, n_out);
        return;
    }
    std::vector<std::thread> pool;
    const int64_t chunk = (n_out + nt - 1) / nt;
    for (int i = 0; i < nt; ++i) {
        const int64_t r0 = i * chunk, r1 = std::min(n_out, r0 + chunk);
        if (r0 < r1) pool.emplace_back(run, r0, r1);
    }
    for (auto& th : pool) th.join();
}

}  // namespace moex::ref
