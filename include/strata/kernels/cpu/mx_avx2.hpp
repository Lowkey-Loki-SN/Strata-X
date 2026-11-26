// include/strata/kernels/cpu/mx_avx2.hpp - moex: AVX-2 multi-token dot products for MXFP4 rows (the routed experts of
// the MXFP4 releases: DeepSeek-V4-Flash, MiMo-V2.6, gpt-oss) against Q8_0 activations, bit-exact against ggml-cpu's
// AVX2 ggml_vec_dot_mxfp4_q8_0 for every token.  See src/kernels/cpu/mx_avx2.cpp.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// out[t][r] = w_r . a[t] for rows [r0, r1): MXFP4 rows of `n` values (n % 32 == 0), `row_bytes` apart; a[t] are Q8_0
/// rows of n values; 1 <= nt <= 8.  Each block's nibbles are looked up once for all nt tokens.
void mx256_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
                int r1);

/// The same with AVX-512BW (mx_avx512.cpp): two blocks per zmm, the weights offset to unsigned with an exact per-lane
/// correction; bitwise equal per token to ggml-cpu's AVX2 dot.  Callers gate on cpu_avx512bw_ok().
void mx512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0,
                int r1);

}  // namespace strata::kernels::cpu
