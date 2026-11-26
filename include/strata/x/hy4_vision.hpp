// include/strata/x/hy4_vision.hpp - moex: Hy4-flash's image encoder on the GPU - the release's vision tower and projector
// (AngelSlim's mmproj-Hy4-flash-bf16.gguf; serve/vision_hy4.py is the CPU encoder this follows step by step, in fp32):
//   * the patch embedding (the 16 x 16 conv as a GEMM over each patch's 768 values) and its bias, then the learned 128 x 128
//     position grid resampled to the picture's patches (F.interpolate bilinear, scale_factor (g + 0.1) / 128, align_corners
//     False; on the host, the grid kept there);
//   * 27 pre-LN blocks: LayerNorm (eps 1e-5), q | k | v with their biases, attention over every patch (16 heads of 72,
//     scale 1 / sqrt(72)), the output projection and its bias, LayerNorm, the MLP (4,304, GELU tanh: the release's
//     bias_activation_fusion; STRATA_HY4_VISION_GELU=erf takes vLLM's erf, as the CPU encoder does) - the linear layers as
//     cuBLAS SGEMM in pedantic math (no TF32), their sums in blocks of 128 terms as the DeepSeek-V4.1 encoder forms them;
//   * the projector: RMS norm, the 2 x 2 conv of stride 2 (each merged cell's 4,608 inputs, (channel, kh, kw)) and its bias,
//     GELU erf, the 1 x 1 conv and its bias, then the newline row after each merged row.
// Deterministic: the same patches give the same rows, bit for bit.
//
// The block weights stay in pinned host memory as released (bf16, 0.82 GB; the rest f32, 0.22 GB) and go to the GPU a block
// at a time on a copy stream while the previous block computes; every device buffer is carved from the caller's workspace.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::x {

class Hy4Vision {
public:
    static constexpr int kLayers = 27, kDim = 1152, kHeads = 16, kHeadDim = 72, kInter = 4304, kPatch = 16, kMerge = 2,
                         kOut = 3072, kGrid = 128;
    static constexpr int kPatchIn = 3 * kPatch * kPatch;       ///< 768: a patch's values, (channel, y, x)
    static constexpr int kMergeIn = kDim * kMerge * kMerge;     ///< 4,608: a merged cell's inputs (channel, kh, kw)
    static constexpr int kMergeHid = 4608;                      ///< the 2 x 2 conv's outputs
    static constexpr int kMaxPatches = 16384;                   ///< the processor's max_pixels / 16^2: a softmax row in shared memory

    using HostAlloc = std::function<void*(size_t)>;   ///< pinned host memory (the engine's host_buf)
    using HostFree = std::function<void(void*)>;
    /// A tensor of the mmproj by its GGUF name: its ggml type (0 F32, 30 BF16), its GGUF shape (dim 0 fastest) and its data;
    /// false when the file has no such tensor
    using TensorSource = std::function<bool(const std::string& name, uint32_t& type, std::vector<uint64_t>& shape,
                                            const uint8_t*& data)>;

    Hy4Vision(HostAlloc alloc, HostFree free);
    ~Hy4Vision();
    Hy4Vision(const Hy4Vision&) = delete;
    Hy4Vision& operator=(const Hy4Vision&) = delete;

    /// Every tensor the encoder reads copied into pinned memory, each checked for its type and shape.  Once per run.
    bool load(const TensorSource& src, std::string& err);
    bool loaded() const { return host_ != nullptr; }
    size_t host_bytes() const { return host_bytes_; }

    /// The rows of a picture of gh x gw patches: (gh / 2) merged rows of (gw / 2) cells and the newline
    static int64_t rows_of(int gh, int gw) { return (int64_t) (gh / kMerge) * (gw / kMerge + 1); }
    /// The device workspace encode() needs for such a picture (0: not a picture it takes - both sides even, at most
    /// kMaxPatches patches)
    static size_t workspace_bytes(int gh, int gw);

    /// One picture: `patches` (gh * gw rows of kPatchIn floats in reading order, serve/vision_hy4_plan.py patches_of) ->
    /// `rows` (rows_of() x kOut floats: each merged row's cells, then the newline row).  On GPU `stream` with the cuBLAS handle
    /// `blas` (its stream set, its math mode pedantic for the call and restored), the weights copied on `copy_stream`;
    /// `workspace` holds workspace_bytes().  Synchronous: `rows` is written when it returns.
    bool encode(void* blas, void* stream, void* copy_stream, void* workspace, size_t ws_bytes, const float* patches, int gh, int gw,
                float* rows, std::string& err);

private:
    HostAlloc alloc_;
    HostFree free_;
    uint8_t* host_ = nullptr;   ///< the weights in the layout of hy4_vision.cu (Layout)
    size_t host_bytes_ = 0;
    bool gelu_erf_ = false;     ///< STRATA_HY4_VISION_GELU=erf: the MLP's GELU as vLLM's
};

}  // namespace strata::x
