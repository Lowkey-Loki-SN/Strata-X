// prefill_mmq_iq1m_test - moex (DSV41_PLAN §58): the prompt path's MMQ products for IQ1_M, the DeepSeek-V4.1 GSQ-RCO
// IQ2_XXS build's expert format, at that model's expert shapes (gate and up together 4608 x 5120, down 5120 x 2304) and
// at rows that are not a multiple of 128 (the fallback tiles), on random IQ1_M blocks (every bit pattern is a valid
// block; a finite fp16 scale is set in each).  Two references, each a double-precision product of ggml's own
// dequantize_row_iq1_m weights: against the activations as given, the screen of prefill_mmq_kquant_test.cpp (MMQ rounds
// them to q8_1 by design); and against the activations rounded as quantize_mmq_q8_1 rounds them (per 32 values
// 127 / amax, roundf), where the tile's weights are ggml's exactly and only the float sums' order differs - a layout,
// sign, delta or scale slip shows there at 1e-2 and above.  Several experts per product, permuted rows, an all-zero row.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

// rows of random IQ1_M blocks (56 bytes a 256 values); the fp16 scale is the top nibbles of the four scale words (bytes
// 48..55), its sign and top exponent bits in byte 55's: 0x1 / 0x2 there (0x9 / 0xA negative) keeps it in 2^-11 .. 2^-4
std::vector<uint8_t> make_matrix(int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t rb = ggml_row_size(GGML_TYPE_IQ1_M, cols);
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += 56) {
        const uint8_t nib = (uint8_t) (((byte(rng) & 1) ? 0x8 : 0x0) | (1 + (byte(rng) & 1)));
        w[o + 55] = (uint8_t) ((w[o + 55] & 0x0F) | (nib << 4));
    }
    return w;
}

std::vector<float> make_x(int rows, int64_t cols, int trial) {
    std::vector<float> x((size_t) rows * (size_t) cols);
    for (int r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k)
            x[(size_t) r * (size_t) cols + (size_t) k] =
                0.45f * std::sin(0.009f * (float) (k + 1) + 0.23f * (float) r + 0.07f * (float) trial) +
                0.17f * std::cos(0.021f * (float) (k + 3) - 0.19f * (float) r) +
                0.002f * (float) (((k * 7 + r * 13 + trial * 5) % 19) - 9);
    std::fill(x.end() - cols, x.end(), 0.0f);   // an all-zero row: exactly zero out
    return x;
}

// the activations as quantize_mmq_q8_1 rounds them: per 32 values d_inv = 127 / amax, q = roundf(x d_inv), x' = q / d_inv
std::vector<float> q8_1_round(const std::vector<float>& x, int64_t cols) {
    std::vector<float> y(x.size(), 0.0f);
    for (size_t b = 0; b < x.size(); b += 32) {
        float amax = 0.0f;
        for (size_t j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b + j]));
        if (amax == 0.0f) continue;
        const float d_inv = 127.0f / amax, d = 1.0f / d_inv;
        for (size_t j = 0; j < 32; ++j) y[b + j] = std::roundf(x[b + j] * d_inv) * d;
    }
    (void) cols;
    return y;
}

std::vector<int32_t> perm(int n, int shift, bool reverse) {
    std::vector<int32_t> p((size_t) n);
    for (int i = 0; i < n; ++i) p[(size_t) i] = (int32_t) (((reverse ? n - 1 - i : i) + shift) % n);
    return p;
}

// one grouped product: experts side by side at one stride (a zeroed tail after them, as the prompt path stages them)
void product(mmq::Context& ctx, cudaStream_t s, const char* name, int64_t out_rows, int64_t cols,
             const std::vector<int>& counts, int trial) {
    const ggml_type t = GGML_TYPE_IQ1_M;
    if (!mmq::supported((int) t)) throw std::runtime_error(std::string(name) + ": IQ1_M is not covered by this build");
    if (!mmq::fits((int) t, out_rows)) throw std::runtime_error(std::string(name) + ": no MMQ tile for these rows");
    std::mt19937 rng(1234u + (unsigned) trial);
    const int n = (int) counts.size(), rows = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int32_t> bounds((size_t) n + 1, 0);
    for (int e = 0; e < n; ++e) bounds[(size_t) e + 1] = bounds[(size_t) e] + counts[(size_t) e];
    const auto src = perm(rows, trial + 1, false), dst = perm(rows, trial + 2, true);
    const std::vector<float> x = make_x(rows, cols, trial), xr8 = q8_1_round(x, cols);
    const size_t eb = mmq::matrix_bytes((int) t, out_rows, cols);
    if (eb != (size_t) out_rows * ggml_row_size(t, cols)) throw std::runtime_error(std::string(name) + ": matrix_bytes");
    std::vector<std::vector<uint8_t>> ws;
    std::vector<uint8_t> w((size_t) n * eb + 4096, 0);
    for (int e = 0; e < n; ++e) {
        ws.push_back(make_matrix(out_rows, cols, rng));
        std::copy(ws.back().begin(), ws.back().end(), w.begin() + (ptrdiff_t) ((size_t) e * eb));
    }
    Dev dx(x.size() * 4), dsrc((size_t) rows * 4), ddst((size_t) rows * 4), db(bounds.size() * 4), dw(w.size()),
        dxq(mmq::q8_bytes(rows, cols)), dy((size_t) rows * (size_t) out_rows * 4);
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dsrc.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaMemcpy(ddst.p, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst");
    ck(cudaMemcpy(db.p, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "bounds");
    ck(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMemset(dy.p, 0xff, (size_t) rows * (size_t) out_rows * 4), "sentinel");
    ck(cudaDeviceSynchronize(), "inputs");   // (the legacy stream's copies before the non-blocking stream's kernels)
    mmq::quantize((const float*) dx.p, (const int32_t*) dsrc.p, dxq.p, (int) t, cols, cols, rows, s);
    mmq::Product p;
    p.w = dw.p; p.type = (int) t; p.w_rows = out_rows; p.w_cols = cols; p.expert_bytes = eb; p.n = n;
    p.xq = dxq.p; p.bounds = (const int32_t*) db.p; p.ids = (const int32_t*) ddst.p; p.total_rows = rows;
    p.max_rows = *std::max_element(counts.begin(), counts.end()); p.dst = (float*) dy.p; p.ld_dst = out_rows;
    ctx.run(p, s);
    ck(cudaGetLastError(), "launch");
    ck(cudaStreamSynchronize(s), "sync");
    std::vector<float> got((size_t) rows * (size_t) out_rows);
    ck(cudaMemcpy(got.data(), dy.p, got.size() * 4, cudaMemcpyDeviceToHost), "y");

    const auto* tr = ggml_get_type_traits(t);
    std::vector<float> ref(got.size(), 0.0f), ref8(got.size(), 0.0f), wd((size_t) out_rows * (size_t) cols);
    for (int e = 0; e < n; ++e) {
        const size_t rb = ggml_row_size(t, cols);
        for (int64_t o = 0; o < out_rows; ++o)
            tr->to_float(ws[(size_t) e].data() + (size_t) o * rb, wd.data() + (size_t) o * (size_t) cols, cols);
        for (int r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r) {
            const float* xr = x.data() + (size_t) src[(size_t) r] * (size_t) cols;
            const float* x8 = xr8.data() + (size_t) src[(size_t) r] * (size_t) cols;
            for (int64_t o = 0; o < out_rows; ++o) {
                double acc = 0, acc8 = 0;
                const float* wr = wd.data() + (size_t) o * (size_t) cols;
                for (int64_t k = 0; k < cols; ++k) {
                    acc += (double) wr[k] * xr[k];
                    acc8 += (double) wr[k] * x8[k];
                }
                ref[(size_t) dst[(size_t) r] * (size_t) out_rows + (size_t) o] = (float) acc;
                ref8[(size_t) dst[(size_t) r] * (size_t) out_rows + (size_t) o] = (float) acc8;
            }
        }
    }
    double err2 = 0, ref2 = 0, max_abs = 0, err8 = 0, max8 = 0, zero_max = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) throw std::runtime_error(std::string(name) + ": unwritten or non-finite output");
        const double d = (double) got[i] - ref[i], d8 = (double) got[i] - ref8[i];
        err2 += d * d; ref2 += (double) ref[i] * ref[i]; max_abs = std::max(max_abs, std::fabs(d));
        err8 += d8 * d8; max8 = std::max(max8, std::fabs(d8));
    }
    for (int r = 0; r < rows; ++r)
        if (src[(size_t) r] == rows - 1)   // the zero row
            for (int64_t o = 0; o < out_rows; ++o)
                zero_max = std::max(zero_max, (double) std::fabs(got[(size_t) dst[(size_t) r] * (size_t) out_rows + (size_t) o]));
    const double rms = std::sqrt(ref2 / (double) ref.size()), rel = std::sqrt(err2 / (double) ref.size()) / rms,
                 rel8 = std::sqrt(err8 / (double) ref.size()) / rms;
    std::printf("%-26s rows %2d experts %d  ref_rms %.4g  rel_l2 %.5f  max/rms %.4f | as q8_1: rel_l2 %.2e  max/rms %.2e  "
                "| zero row %.3g\n", name, rows, n, rms, rel, max_abs / rms, rel8, max8 / rms, zero_max);
    if (rel > 0.04 || max_abs / rms > 0.35 || rel8 > 1e-3 || max8 / rms > 1e-2 || zero_max > std::max(1e-5, 1e-4 * rms))
        throw std::runtime_error(std::string(name) + ": outside the bounds");
}
}  // namespace

int main() {
    try {
        if (!mmq::built()) { std::printf("no MMQ in this build\n"); return 1; }
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        {
            mmq::Context ctx;
            const std::vector<std::vector<int>> batches{{1, 3, 3}, {4, 1, 2}, {2, 3}, {17}, {40, 9}};
            int trial = 0;
            for (const auto& counts : batches) {
                const std::string tag = "-" + std::to_string(trial);
                product(ctx, s, ("IQ1_M gate+up 4608x5120" + tag).c_str(), 4608, 5120, counts, trial);
                product(ctx, s, ("IQ1_M down 5120x2304" + tag).c_str(), 5120, 2304, counts, trial + 7);
                product(ctx, s, ("IQ1_M 1000x768 (fallback)" + tag).c_str(), 1000, 768, counts, trial + 13);
                ++trial;
            }
        }
        ck(cudaStreamDestroy(s), "destroy");
        std::printf("prefill MMQ IQ1_M parity passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill MMQ IQ1_M parity failed: %s\n", e.what());
        return 1;
    }
}
