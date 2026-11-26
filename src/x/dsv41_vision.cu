// src/x/dsv41_vision.cu - moex (DSV41_PLAN §54): DeepSeek-V4.1-Flash's image encoder on the GPU
// (include/strata/x/dsv41_vision.hpp).  serve/vision_dsv41.py is the CPU transcription of the release's vision.py that
// this follows step by step; dsv41_vision_parity checks the two against each other.
#include "strata/x/dsv41_vision.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

namespace strata::x {
namespace {

using V = Dsv41Vision;
constexpr int kQkv = 3 * V::kDim;          // 3,072: q | k | v, each 16 heads of 64
constexpr int kGU = 2 * V::kInter;         // 5,632: gate | up
constexpr int kRope = V::kHeadDim / 2;     // 32: a patch's angles - 16 by its row, then 16 by its column
constexpr int kChunk = 1024;               // the aligner's weights go up 1,024 output rows at a time
// A long sum is formed in blocks of kSumBlock terms, each block's sum added to the result in turn (cuBLAS's beta 1), as the
// CPU's GEMMs block their K and torch's fused attention its keys.  Against a float64 encode of a 1920x1080 picture (8,418
// patches; the CPU encoder 9.3e-6): one sum over each whole K 1.15e-4, blocks of 512 2.95e-5, of 128 6.0e-6 (DSV41_PLAN §54)
constexpr int kSumBlock = 128;
constexpr size_t kScoreBytes = 160ull << 20;   // a block of queries' scores, every head
constexpr float kEps = 1e-6f;
constexpr float kSqrtHalf = 0.70710678118654752440f;   // torch's M_SQRT1_2 in float (F.gelu, approximate "none")

// THE HOST LAYOUT (bf16 elements).  Piece 0: the patch embedding, the final norm and the aligner's biases.  Then the 32
// blocks, each contiguous in the order below (one copy a layer), the aligner's two weights (copied kChunk rows at a time)
// and the span's three delimiter rows (the host's: span_rows).
constexpr size_t kPeW = 0, kPeB = kPeW + (size_t) V::kDim * V::kPatchIn, kVNorm = kPeB + V::kDim, kA1B = kVNorm + V::kDim,
                 kA2B = kA1B + V::kOut, kPiece0 = kA2B + V::kOut;
constexpr size_t kBN1 = 0, kBWqkv = kBN1 + V::kDim, kBBqkv = kBWqkv + (size_t) kQkv * V::kDim, kBWo = kBBqkv + kQkv,
                 kBBo = kBWo + (size_t) V::kDim * V::kDim, kBN2 = kBBo + V::kDim, kBW1 = kBN2 + V::kDim,
                 kBW2 = kBW1 + (size_t) kGU * V::kDim, kBlock = kBW2 + (size_t) V::kDim * V::kInter;
constexpr size_t kBlocks = kPiece0, kA1W = kBlocks + (size_t) V::kLayers * kBlock, kA2W = kA1W + (size_t) V::kOut * V::kUnfold,
                 kDelim = kA2W + (size_t) V::kOut * V::kOut, kHostElems = kDelim + 3 * (size_t) V::kOut;
constexpr size_t kStageElems = std::max({kPiece0, kBlock, (size_t) kChunk * V::kUnfold});   // the largest piece
constexpr int kPieces = 1 + V::kLayers + 2 * (V::kOut / kChunk);   // piece 0, the blocks, the aligner's chunks
static_assert(V::kOut % kChunk == 0, "the aligner's chunks");

float bf16_float(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

/// The workspace of one picture: byte offsets into it
struct Plan {
    int64_t n = 0, L = 0;
    int qc = 0;   ///< queries a block of the attention
    size_t x = 0, t = 0, big = 0, sc = 0, st[2] = {0, 0}, wf = 0, hv = 0, rope = 0, total = 0;
};

size_t al256(size_t b) { return (b + 255) & ~(size_t) 255; }

Plan plan_of(int nh, int nw) {
    Plan p;
    if (nh < 1 || nw < 1 || (int64_t) nh * nw > V::kMaxPatches) return p;   // total 0: not a picture it takes
    p.n = (int64_t) nh * nw;
    p.L = V::cells(nh, nw);
    const size_t n = (size_t) p.n, L = (size_t) p.L;
    int qc = (int) std::min<size_t>(n, kScoreBytes / (sizeof(float) * V::kHeads * n));
    if (qc < (int) n) qc = std::max(32, qc / 32 * 32);
    p.qc = qc;
    // big holds in turn: the patches (fp32, their bf16 copy after them), qkv, gate | up, then the aligner's cells, its hidden
    // rows and its output rows
    const size_t big = std::max({n * kGU, L * (size_t) (V::kUnfold + 2 * V::kOut), n * V::kPatchIn + (n * V::kPatchIn + 1) / 2});
    size_t o = 0;
    auto take = [&](size_t bytes) {
        const size_t at = o;
        o += al256(bytes);
        return at;
    };
    p.x = take(n * V::kDim * sizeof(float));
    p.t = take(n * V::kDim * sizeof(float));
    p.big = take(big * sizeof(float));
    p.sc = take((size_t) V::kHeads * (size_t) qc * n * sizeof(float));
    p.st[0] = take(kStageElems * sizeof(uint16_t));
    p.st[1] = take(kStageElems * sizeof(uint16_t));
    p.wf = take(kStageElems * sizeof(float));
    p.hv = take((size_t) (V::kDim + 2 * V::kOut) * sizeof(float));
    p.rope = take(2 * n * kRope * sizeof(float));
    p.total = o;
    return p;
}

int grid_of(int64_t n, int threads) { return (int) std::min<int64_t>((n + threads - 1) / threads, 65535); }

// ------------------------------------------------------------------------------------------------ the kernels

__global__ void k_widen(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        out[i] = __uint_as_float((uint32_t) in[i] << 16);
}

/// y = x + b per row (F.linear's bias), in place
__global__ void k_bias(float* __restrict__ y, const float* __restrict__ b, int64_t rows, int width) {
    const int64_t n = rows * width;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = __fadd_rn(y[i], b[i % width]);
}

/// x = x + (t + b) (the block's residual over a biased linear), x = x + t without a bias
__global__ void k_residual(float* __restrict__ x, const float* __restrict__ t, const float* __restrict__ b, int64_t rows, int width) {
    const int64_t n = rows * width;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] = __fadd_rn(x[i], b ? __fadd_rn(t[i], b[i % width]) : t[i]);
}

/// vision.py RMSNorm on rows of kDim: x * (1 / sqrt(mean(x^2) + eps)), then w * that - the squares summed in a fixed order
/// (each thread's four, a warp's butterfly, the eight warps in turn)
__global__ void __launch_bounds__(256) k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y) {
    __shared__ float wsum[8];
    __shared__ float rs;
    const float* xr = x + (int64_t) blockIdx.x * V::kDim;
    float v[4], ss = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k] = xr[threadIdx.x + 256 * k];
        ss = __fadd_rn(ss, __fmul_rn(v[k], v[k]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
    if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t = __fadd_rn(t, wsum[i]);
        rs = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fmul_rn(t, 1.0f / V::kDim), kEps)));   // (1 / 1024: exact)
    }
    __syncthreads();
    const float s = rs;
    float* yr = y + (int64_t) blockIdx.x * V::kDim;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int j = threadIdx.x + 256 * k;
        yr[j] = __fmul_rn(w[j], __fmul_rn(v[k], s));
    }
}

/// qkv rows (n x 3072): the bias added, then q and k rotated as vision.py's rotary - per head the halves x1 (dims 0..31) and
/// x2 (32..63) become x1 cos - x2 sin and x2 cos + x1 sin, cos / sin of the patch's angle d.  A thread per q / k pair, or per
/// v value (bias only).
__global__ void k_qkv_rope(float* __restrict__ qkv, const float* __restrict__ b, const float* __restrict__ cs,
                           const float* __restrict__ sn, int64_t n) {
    const int64_t total = n * 2048;
    for (int64_t g = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; g < total; g += (int64_t) gridDim.x * blockDim.x) {
        const int64_t r = g >> 11;
        const int it = (int) (g & 2047);
        float* row = qkv + r * kQkv;
        if (it < 1024) {
            const int part = it >> 9, h = (it >> 5) & 15, d = it & 31;   // part 0: q, 1: k
            const int j1 = part * V::kDim + h * V::kHeadDim + d, j2 = j1 + kRope;
            const float x1 = __fadd_rn(row[j1], b[j1]), x2 = __fadd_rn(row[j2], b[j2]);
            const float c = cs[r * kRope + d], s = sn[r * kRope + d];
            row[j1] = __fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s));
            row[j2] = __fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s));
        } else {
            const int j = 2 * V::kDim + (it - 1024);
            row[j] = __fadd_rn(row[j], b[j]);
        }
    }
}

/// Rows of n scores, in place: exp(s - max) / sum.  A block a row, the row in shared memory (one read, one write), the sum
/// in a fixed order (each thread's stride, a warp's butterfly, the 16 warps' butterfly).
__global__ void __launch_bounds__(512) k_softmax(float* __restrict__ s, int n) {
    extern __shared__ float row[];
    __shared__ float red[16];
    __shared__ float res;
    float* g = s + (int64_t) blockIdx.x * n;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += 512) {
        const float v = g[i];
        row[i] = v;
        m = fmaxf(m, v);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < 16 ? red[threadIdx.x] : -INFINITY;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) res = v;
    }
    __syncthreads();
    const float mx = res;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += 512) {
        const float e = expf(__fsub_rn(row[i], mx));
        row[i] = e;
        sum = __fadd_rn(sum, e);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    __syncthreads();   // every thread read `res`
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < 16 ? red[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) res = v;
    }
    __syncthreads();
    const float tot = res;
    for (int i = threadIdx.x; i < n; i += 512) g[i] = __fdiv_rn(row[i], tot);
}

/// gate | up rows (row stride 5632): gate <- silu(gate) * up, silu(x) = x / (1 + exp(-x)) (torch's)
__global__ void k_silu_mul(float* __restrict__ gu, int64_t rows) {
    const int64_t n = rows * V::kInter;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        float* row = gu + (i / V::kInter) * kGU;
        const int j = (int) (i % V::kInter);
        const float x = row[j];
        row[j] = __fmul_rn(__fdiv_rn(x, __fadd_rn(1.0f, expf(-x))), row[V::kInter + j]);
    }
}

/// The aligner's hidden rows: v = h + b, then F.gelu's (v * 0.5) * (1 + erf(v * sqrt(1/2)))
__global__ void k_bias_gelu(float* __restrict__ h, const float* __restrict__ b, int64_t rows, int width) {
    const int64_t n = rows * width;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float v = __fadd_rn(h[i], b[i % width]);
        h[i] = __fmul_rn(__fmul_rn(v, 0.5f), __fadd_rn(1.0f, erff(__fmul_rn(v, kSqrtHalf))));
    }
}

/// The final rows (nh x nw patches of kDim) -> the aligner's cells (lh x lw of kUnfold): cell (bh, bw)'s column
/// c * 9 + kh * 3 + kw is patch (3 bh + kh, 3 bw + kw)'s value c, 0 past the picture's edge (F.pad, F.unfold(3, stride=3))
__global__ void k_unfold(const float* __restrict__ t, float* __restrict__ u, int nh, int nw, int lw, int64_t cells) {
    const int64_t n = cells * V::kUnfold;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t l = i / V::kUnfold;
        const int col = (int) (i % V::kUnfold), c = col / 9, kk = col % 9;
        const int y = 3 * (int) (l / lw) + kk / 3, x = 3 * (int) (l % lw) + kk % 3;
        u[i] = y < nh && x < nw ? t[((int64_t) y * nw + x) * V::kDim + c] : 0.0f;
    }
}

// ------------------------------------------------------------------------------------------------ the host side

struct Guard {   // the encode's end, however it ends: the streams drained, the events gone, the handle's math mode back
    cublasHandle_t h = nullptr;
    cudaStream_t s = nullptr, cs = nullptr;
    cublasMath_t math = CUBLAS_DEFAULT_MATH;
    bool math_set = false;
    cudaEvent_t ev[4] = {nullptr, nullptr, nullptr, nullptr};
    ~Guard() {
        if (s) cudaStreamSynchronize(s);
        if (cs) cudaStreamSynchronize(cs);
        for (cudaEvent_t e : ev)
            if (e) cudaEventDestroy(e);
        if (math_set) cublasSetMathMode(h, math);
    }
};

bool cuda_ok(const char* what, std::string& err) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    err = std::string("the image encoder on the GPU: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

bool blas_ok(cublasStatus_t st, const char* what, std::string& err) {
    if (st == CUBLAS_STATUS_SUCCESS) return true;
    err = std::string("the image encoder on the GPU: cuBLAS refused ") + what + " (status " + std::to_string((int) st) + ")";
    return false;
}

/// y (rows x out, row stride ldy) = x (rows x in, row stride ldx) W^T, W (out x in) row-major - the sum over `in` in blocks of
/// kSumBlock (the first block's product written, each later one's added: beta 1)
bool linear(cublasHandle_t h, const float* x, int ldx, const float* w, int in, int out, int64_t rows, float* y, int ldy,
            const char* what, std::string& err) {
    const float one = 1.0f, zero = 0.0f;
    for (int k0 = 0; k0 < in; k0 += kSumBlock) {
        const int kb = std::min(kSumBlock, in - k0);
        if (!blas_ok(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, out, (int) rows, kb, &one, w + k0, in, x + k0, ldx, k0 ? &one : &zero, y,
                                 ldy),
                     what, err))
            return false;
    }
    return true;
}

}  // namespace

Dsv41Vision::Dsv41Vision(HostAlloc alloc, HostFree free) : alloc_(std::move(alloc)), free_(std::move(free)) {}

Dsv41Vision::~Dsv41Vision() {
    if (host_ && free_) free_(host_);
}

size_t Dsv41Vision::workspace_bytes(int n_vit_h, int n_vit_w) { return plan_of(n_vit_h, n_vit_w).total; }

bool Dsv41Vision::load(const std::string& manifest, std::string& err) {
    if (host_) return true;
    struct Want {
        size_t at, elems;
        int d0, d1;   // d1 = 0: one-dimensional
    };
    std::map<std::string, Want> want;
    want["vision.patch_embed.proj.weight"] = {kPeW, (size_t) kDim * kPatchIn, kDim, kPatchIn};
    want["vision.patch_embed.proj.bias"] = {kPeB, (size_t) kDim, kDim, 0};
    want["vision.norm.weight"] = {kVNorm, (size_t) kDim, kDim, 0};
    want["aligner.w1.weight"] = {kA1W, (size_t) kOut * kUnfold, kOut, kUnfold};
    want["aligner.w1.bias"] = {kA1B, (size_t) kOut, kOut, 0};
    want["aligner.w2.weight"] = {kA2W, (size_t) kOut * kOut, kOut, kOut};
    want["aligner.w2.bias"] = {kA2B, (size_t) kOut, kOut, 0};
    want["image_start"] = {kDelim, (size_t) kOut, kOut, 0};
    want["image_newline"] = {kDelim + kOut, (size_t) kOut, kOut, 0};
    want["image_end"] = {kDelim + 2 * (size_t) kOut, (size_t) kOut, kOut, 0};
    for (int i = 0; i < kLayers; ++i) {
        const std::string b = "vision.blocks." + std::to_string(i) + ".";
        const size_t base = kBlocks + (size_t) i * kBlock;
        want[b + "norm1.weight"] = {base + kBN1, (size_t) kDim, kDim, 0};
        want[b + "attn.wqkv.weight"] = {base + kBWqkv, (size_t) kQkv * kDim, kQkv, kDim};
        want[b + "attn.wqkv.bias"] = {base + kBBqkv, (size_t) kQkv, kQkv, 0};
        want[b + "attn.wo.weight"] = {base + kBWo, (size_t) kDim * kDim, kDim, kDim};
        want[b + "attn.wo.bias"] = {base + kBBo, (size_t) kDim, kDim, 0};
        want[b + "norm2.weight"] = {base + kBN2, (size_t) kDim, kDim, 0};
        want[b + "mlp.w1.weight"] = {base + kBW1, (size_t) kGU * kDim, kGU, kDim};
        want[b + "mlp.w2.weight"] = {base + kBW2, (size_t) kDim * kInter, kDim, kInter};
    }
    std::ifstream f(manifest);
    std::string line;
    if (!f || !std::getline(f, line) || line != "SVM1") {
        err = "the image encoder's manifest " + manifest + " is missing or not an SVM1 file";
        return false;
    }
    uint16_t* host = static_cast<uint16_t*>(alloc_ ? alloc_(kHostElems * sizeof(uint16_t)) : nullptr);
    if (!host) {
        err = "the image encoder's weights: " + std::to_string((kHostElems * sizeof(uint16_t)) >> 20) + " MiB of pinned memory refused";
        return false;
    }
    std::map<std::string, int> fds;
    auto fail = [&](const std::string& why) {
        for (auto& kv : fds) ::close(kv.second);
        free_(host);
        err = "the image encoder's weights: " + why;
        return false;
    };
    std::map<std::string, bool> seen;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ls(line);
        std::string tag, name, shape, path;
        unsigned long long off = 0, bytes = 0;
        if (!(ls >> tag >> name >> off >> bytes >> shape) || tag != "T") return fail("a manifest line is not \"T <name> <offset> <bytes> <shape> <file>\"");
        std::getline(ls, path);
        path.erase(0, path.find_first_not_of(' '));
        const auto it = want.find(name);
        if (it == want.end()) return fail("the manifest names " + name + ", which the encoder does not take");
        if (seen.count(name)) return fail(name + " twice in the manifest");
        const Want& w = it->second;
        int d0 = 0, d1 = 0;
        const size_t x = shape.find('x');
        try {
            d0 = std::stoi(shape.substr(0, x));
            d1 = x == std::string::npos ? 0 : std::stoi(shape.substr(x + 1));
        } catch (const std::exception&) {
            return fail(name + "'s shape " + shape + " is not a number");
        }
        const bool vec_ok = w.d1 == 0 && ((d0 == w.d0 && d1 == 0) || (d0 == 1 && d1 == w.d0));   // [n] or [1, n]
        if (!(vec_ok || (d0 == w.d0 && d1 == w.d1))) return fail(name + " is " + shape + ", the encoder expects " + std::to_string(w.d0) +
                                                                (w.d1 ? "x" + std::to_string(w.d1) : std::string()));
        if (bytes != w.elems * sizeof(uint16_t)) return fail(name + " is " + std::to_string(bytes) + " bytes, not BF16 of its shape");
        int fd;
        if (const auto fit = fds.find(path); fit != fds.end()) {
            fd = fit->second;
        } else {
            fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) return fail("cannot open " + path + ": " + std::strerror(errno));
            fds[path] = fd;
        }
        struct stat sb {};
        if (::fstat(fd, &sb) != 0 || (unsigned long long) sb.st_size < off + bytes) return fail(path + " is shorter than " + name + "'s bytes");
        uint8_t* dst = reinterpret_cast<uint8_t*>(host + w.at);
        size_t got = 0;
        while (got < bytes) {
            const ssize_t r = ::pread(fd, dst + got, (size_t) bytes - got, (off_t) (off + got));
            if (r <= 0) return fail("reading " + name + " from " + path + ": " + (r < 0 ? std::strerror(errno) : "short"));
            got += (size_t) r;
        }
        seen[name] = true;
    }
    for (auto& kv : fds) ::close(kv.second);
    fds.clear();
    if (seen.size() != want.size()) {
        for (const auto& kv : want)
            if (!seen.count(kv.first)) return fail("the manifest lacks " + kv.first);
    }
    host_ = host;
    host_bytes_ = kHostElems * sizeof(uint16_t);
    return true;
}

void Dsv41Vision::span_rows(const float* rows, int n_llm_h, int n_llm_w, float* out) const {
    auto delim = [&](int k, float* o) {
        const uint16_t* d = host_ + kDelim + (size_t) k * kOut;
        for (int j = 0; j < kOut; ++j) o[j] = bf16_float(d[j]);
    };
    float* o = out;
    delim(0, o);
    o += kOut;
    for (int r = 0; r < n_llm_h; ++r) {
        std::memcpy(o, rows + (size_t) r * n_llm_w * kOut, (size_t) n_llm_w * kOut * sizeof(float));
        o += (size_t) n_llm_w * kOut;
        delim(1, o);
        o += kOut;
    }
    delim(2, o);
}

bool Dsv41Vision::encode(void* blas, void* stream, void* copy_stream, void* workspace, size_t ws_bytes, const uint16_t* patches,
                         int n_vit_h, int n_vit_w, int n_llm_h, int n_llm_w, float* rows, std::string& err) {
    if (!host_) {
        err = "the image encoder's weights are not loaded";
        return false;
    }
    const Plan p = plan_of(n_vit_h, n_vit_w);
    if (p.total == 0) {
        err = "a picture of " + std::to_string(n_vit_h) + " x " + std::to_string(n_vit_w) + " patches is not one the encoder takes (at most " +
              std::to_string(kMaxPatches) + ")";
        return false;
    }
    if ((int64_t) n_llm_h * n_llm_w != p.L || n_llm_h != (n_vit_h + kRatio - 1) / kRatio) {
        err = "the picture's cells (" + std::to_string(n_llm_h) + " x " + std::to_string(n_llm_w) + ") are not its patches' (" +
              std::to_string(n_vit_h) + " x " + std::to_string(n_vit_w) + ") in threes";
        return false;
    }
    if (ws_bytes < p.total) {
        err = "the image encoder's workspace is " + std::to_string(ws_bytes >> 20) + " MiB, the picture needs " + std::to_string(p.total >> 20);
        return false;
    }
    const int64_t n = p.n, L = p.L;
    Guard g;
    g.h = static_cast<cublasHandle_t>(blas);
    g.s = static_cast<cudaStream_t>(stream);
    g.cs = static_cast<cudaStream_t>(copy_stream);
    uint8_t* ws = static_cast<uint8_t*>(workspace);
    float* x = reinterpret_cast<float*>(ws + p.x);
    float* t = reinterpret_cast<float*>(ws + p.t);
    float* big = reinterpret_cast<float*>(ws + p.big);
    float* sc = reinterpret_cast<float*>(ws + p.sc);
    uint16_t* st[2] = {reinterpret_cast<uint16_t*>(ws + p.st[0]), reinterpret_cast<uint16_t*>(ws + p.st[1])};
    float* wf = reinterpret_cast<float*>(ws + p.wf);
    float* hv = reinterpret_cast<float*>(ws + p.hv);   // the final norm, then the aligner's two biases
    float* cs_t = reinterpret_cast<float*>(ws + p.rope);
    float* sn_t = cs_t + n * kRope;
    if (!blas_ok(cublasSetStream(g.h, g.s), "its stream", err) || !blas_ok(cublasGetMathMode(g.h, &g.math), "its math mode", err))
        return false;
    if (!blas_ok(cublasSetMathMode(g.h, CUBLAS_PEDANTIC_MATH), "pedantic math", err)) return false;
    g.math_set = true;
    for (cudaEvent_t& e : g.ev)
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) return cuda_ok("its events", err);
    cudaEvent_t* copied = g.ev;       // [2]: a staging buffer's piece has landed
    cudaEvent_t* widened = g.ev + 2;  // [2]: ... and been widened (the buffer may take the next piece)

    // THE PIECES: piece 0, block i at 1 + i, the aligner's chunks of w1, then of w2
    auto piece = [&](int k, size_t& elems) -> const uint16_t* {
        if (k == 0) {
            elems = kPiece0;
            return host_ + kPeW;
        }
        if (k <= kLayers) {
            elems = kBlock;
            return host_ + kBlocks + (size_t) (k - 1) * kBlock;
        }
        const int c = k - 1 - kLayers, nc = kOut / kChunk;
        if (c < nc) {
            elems = (size_t) kChunk * kUnfold;
            return host_ + kA1W + (size_t) c * kChunk * kUnfold;
        }
        elems = (size_t) kChunk * kOut;
        return host_ + kA2W + (size_t) (c - nc) * kChunk * kOut;
    };
    auto issue = [&](int k) {   // piece k's copy into staging buffer k % 2, once piece k - 2 there was widened
        size_t elems = 0;
        const uint16_t* src = piece(k, elems);
        if (k >= 2) cudaStreamWaitEvent(g.cs, widened[k & 1], 0);
        cudaMemcpyAsync(st[k & 1], src, elems * sizeof(uint16_t), cudaMemcpyHostToDevice, g.cs);
        cudaEventRecord(copied[k & 1], g.cs);
    };
    auto widen = [&](int k) -> bool {   // piece k into wf (piece 0's vectors into hv), then the next copy issued
        size_t elems = 0;
        piece(k, elems);
        cudaStreamWaitEvent(g.s, copied[k & 1], 0);
        if (k == 0) {
            k_widen<<<grid_of((int64_t) kVNorm, 256), 256, 0, g.s>>>(st[0], wf, (int64_t) kVNorm);   // the patch embedding
            k_widen<<<grid_of((int64_t) (kPiece0 - kVNorm), 256), 256, 0, g.s>>>(st[0] + kVNorm, hv, (int64_t) (kPiece0 - kVNorm));
        } else {
            k_widen<<<grid_of((int64_t) elems, 256), 256, 0, g.s>>>(st[k & 1], wf, (int64_t) elems);
        }
        cudaEventRecord(widened[k & 1], g.s);
        if (k + 2 < kPieces) issue(k + 2);
        return cuda_ok("a weight piece's widening", err);
    };
    issue(0);
    issue(1);

    // THE PATCHES (their bf16 copy after the fp32 rows in big) and the angles' cos / sin (vision.py vision_cos_sin: patch
    // (i, j)'s 16 row angles i / theta^(2k/32), then its 16 column angles; each angle the float product torch forms, its
    // cos / sin rounded from double)
    uint16_t* pb = reinterpret_cast<uint16_t*>(big + n * kPatchIn);
    cudaMemcpyAsync(pb, patches, (size_t) n * kPatchIn * sizeof(uint16_t), cudaMemcpyHostToDevice, g.s);
    k_widen<<<grid_of(n * kPatchIn, 256), 256, 0, g.s>>>(pb, big, n * kPatchIn);
    {
        std::vector<float> cv((size_t) n * kRope), sv((size_t) n * kRope);
        float inv[kRope / 2];
        for (int k = 0; k < kRope / 2; ++k) inv[k] = 1.0f / std::pow(10000.0f, (float) (2 * k) / (float) kRope);
        for (int64_t i = 0; i < n; ++i) {
            const float hp = (float) (i / n_vit_w), wp = (float) (i % n_vit_w);
            for (int k = 0; k < kRope; ++k) {
                const float a = (k < kRope / 2 ? hp : wp) * inv[k % (kRope / 2)];
                cv[(size_t) i * kRope + k] = (float) std::cos((double) a);
                sv[(size_t) i * kRope + k] = (float) std::sin((double) a);
            }
        }
        cudaMemcpyAsync(cs_t, cv.data(), cv.size() * sizeof(float), cudaMemcpyHostToDevice, g.s);
        cudaMemcpyAsync(sn_t, sv.data(), sv.size() * sizeof(float), cudaMemcpyHostToDevice, g.s);
        if (cudaStreamSynchronize(g.s) != cudaSuccess) return cuda_ok("the patches' upload", err);   // (cv, sv go out of scope)
    }
    if (!cuda_ok("the patches", err)) return false;

    // the patch embedding: x = patches W^T + b
    if (!widen(0)) return false;
    if (!linear(g.h, big, kPatchIn, wf, kPatchIn, kDim, n, x, kDim, "the patch embedding", err)) return false;
    k_bias<<<grid_of(n * kDim, 256), 256, 0, g.s>>>(x, wf + (kPeB - kPeW), n, kDim);
    if (!cuda_ok("the patch embedding's bias", err)) return false;

    const float one = 1.0f, zero = 0.0f, scale = 0.125f;   // 1 / sqrt(64): exact
    const size_t smem = (size_t) n * sizeof(float);
    for (int il = 0; il < kLayers; ++il) {
        if (!widen(1 + il)) return false;
        const float *n1 = wf + kBN1, *wqkv = wf + kBWqkv, *bqkv = wf + kBBqkv, *wo = wf + kBWo, *bo = wf + kBBo, *n2 = wf + kBN2,
                    *w1 = wf + kBW1, *w2 = wf + kBW2;
        // h = norm1(x); q | k | v = h Wqkv^T + b; q, k rotated
        k_rmsnorm<<<(unsigned) n, 256, 0, g.s>>>(x, n1, t);
        if (!linear(g.h, t, kDim, wqkv, kDim, kQkv, n, big, kQkv, "the attention's qkv", err)) return false;
        k_qkv_rope<<<grid_of(n * 2048, 256), 256, 0, g.s>>>(big, bqkv, cs_t, sn_t, n);
        if (!cuda_ok("the qkv bias and rope", err)) return false;
        // o = softmax(q k^T / 8) v, a block of queries at a time, every head in one call (into t: h is consumed)
        const float* q = big;
        const float* k = big + kDim;
        const float* v = big + 2 * kDim;
        for (int64_t q0 = 0; q0 < n; q0 += p.qc) {
            const int qb = (int) std::min<int64_t>(p.qc, n - q0);
            if (!blas_ok(cublasSgemmStridedBatched(g.h, CUBLAS_OP_T, CUBLAS_OP_N, (int) n, qb, kHeadDim, &scale, k, kQkv, kHeadDim,
                                                   q + q0 * kQkv, kQkv, kHeadDim, &zero, sc, (int) n, (long long) qb * n, kHeads),
                         "the attention's scores", err))
                return false;
            k_softmax<<<(unsigned) (kHeads * qb), 512, smem, g.s>>>(sc, (int) n);
            if (!cuda_ok("the attention's softmax", err)) return false;
            for (int64_t j0 = 0; j0 < n; j0 += kSumBlock) {   // o = p v, the keys kSumBlock at a time (beta 1 after the first)
                const int kb = (int) std::min<int64_t>(kSumBlock, n - j0);
                if (!blas_ok(cublasSgemmStridedBatched(g.h, CUBLAS_OP_N, CUBLAS_OP_N, kHeadDim, qb, kb, &one, v + j0 * kQkv, kQkv, kHeadDim,
                                                       sc + j0, (int) n, (long long) qb * n, j0 ? &one : &zero, t + q0 * kDim, kDim,
                                                       kHeadDim, kHeads),
                             "the attention's output", err))
                    return false;
            }
        }
        // x = x + (o Wo^T + bo)   (o Wo^T into big: qkv is consumed)
        if (!linear(g.h, t, kDim, wo, kDim, kDim, n, big, kDim, "the attention's wo", err)) return false;
        k_residual<<<grid_of(n * kDim, 256), 256, 0, g.s>>>(x, big, bo, n, kDim);
        // x = x + w2(silu(gate) * up), gate | up = norm2(x) W1^T   (silu(gate) * up over the gate half; its product into t)
        k_rmsnorm<<<(unsigned) n, 256, 0, g.s>>>(x, n2, t);
        if (!linear(g.h, t, kDim, w1, kDim, kGU, n, big, kGU, "the mlp's w1", err)) return false;
        k_silu_mul<<<grid_of(n * kInter, 256), 256, 0, g.s>>>(big, n);
        if (!linear(g.h, big, kGU, w2, kInter, kDim, n, t, kDim, "the mlp's w2", err)) return false;
        k_residual<<<grid_of(n * kDim, 256), 256, 0, g.s>>>(x, t, nullptr, n, kDim);
        if (!cuda_ok("a block", err)) return false;
    }
    // the final norm, the 3x3 cells, the aligner: gelu(cells W1^T + b1) W2^T + b2, its weights kChunk output rows at a time
    k_rmsnorm<<<(unsigned) n, 256, 0, g.s>>>(x, hv, t);
    float* u = big;
    float* hid = big + L * kUnfold;
    float* out = hid + L * kOut;
    k_unfold<<<grid_of(L * kUnfold, 256), 256, 0, g.s>>>(t, u, n_vit_h, n_vit_w, n_llm_w, L);
    if (!cuda_ok("the final norm and the cells", err)) return false;
    const int nc = kOut / kChunk;
    for (int c = 0; c < nc; ++c) {
        if (!widen(1 + kLayers + c)) return false;
        if (!linear(g.h, u, kUnfold, wf, kUnfold, kChunk, L, hid + (size_t) c * kChunk, kOut, "the aligner's w1", err)) return false;
    }
    k_bias_gelu<<<grid_of(L * kOut, 256), 256, 0, g.s>>>(hid, hv + kDim, L, kOut);
    for (int c = 0; c < nc; ++c) {
        if (!widen(1 + kLayers + nc + c)) return false;
        if (!linear(g.h, hid, kOut, wf, kOut, kChunk, L, out + (size_t) c * kChunk, kOut, "the aligner's w2", err)) return false;
    }
    k_bias<<<grid_of(L * kOut, 256), 256, 0, g.s>>>(out, hv + kDim + kOut, L, kOut);
    if (!cuda_ok("the aligner", err)) return false;
    cudaMemcpyAsync(rows, out, (size_t) L * kOut * sizeof(float), cudaMemcpyDeviceToHost, g.s);
    if (cudaStreamSynchronize(g.s) != cudaSuccess || cudaStreamSynchronize(g.cs) != cudaSuccess) return cuda_ok("the rows' copy", err);
    return cuda_ok("the encode", err);
}

}  // namespace strata::x
