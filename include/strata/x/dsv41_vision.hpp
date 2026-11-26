// include/strata/x/dsv41_vision.hpp - moex (DSV41_PLAN §54): DeepSeek-V4.1-Flash's image encoder on the GPU - the
// release's ViT and aligner (inference/vision.py; serve/vision_dsv41.py is the CPU encoder) in fp32 throughout, as the CPU
// encoder computes them with STRATA_DSV41_VISION_DTYPE=fp32:
//   * the linear layers: cuBLAS SGEMM in pedantic math (no TF32), the bias and the residual added after it, as F.linear's
//     addmm and the block's `x + ...` round them;
//   * the attention (16 heads of 64, every patch to every patch, scale 1/8): per block of queries, S = Q K^T / 8 and
//     O = P V as strided-batched SGEMMs over the heads, the row softmax between them (a row of scores in shared memory: one
//     read, one write, a fixed summation order);
//   * RMSNorm (eps 1e-6), the 2-D RoPE (its cos / sin rounded from double), SiLU, erf GELU: torch's order of roundings,
//     every product and sum rounded on its own (no contraction into an FMA);
//   * the 3x3 unfold of the final rows into the aligner's 9,216 inputs (F.pad + F.unfold).
// Deterministic: the same patches give the same rows, bit for bit (the engine's prefix keys hash a picture's rows).
//
// The weights stay in pinned host memory as released (bf16, 0.97 GB) and go to the GPU a layer at a time on a copy stream
// while the previous layer computes; every device buffer is carved from the caller's workspace (workspace_bytes()).
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace strata::x {

class Dsv41Vision {
public:
    // the release's inference/config.json vision_* values (load() checks the weights' shapes against them)
    static constexpr int kLayers = 32, kDim = 1024, kHeads = 16, kHeadDim = 64, kInter = 2816, kPatch = 14, kRatio = 3,
                         kOut = 5120;
    static constexpr int kPatchIn = 3 * kPatch * kPatch;      ///< 588: a patch's pixels, (channel, y, x)
    static constexpr int kUnfold = kDim * kRatio * kRatio;    ///< 9,216: an aligner cell's inputs
    static constexpr int kMaxPatches = 11264;                 ///< the softmax's row in shared memory (the plan's most: 9,189)

    using HostAlloc = std::function<void*(size_t)>;           ///< pinned host memory (the engine's host_buf)
    using HostFree = std::function<void(void*)>;
    Dsv41Vision(HostAlloc alloc, HostFree free);
    ~Dsv41Vision();
    Dsv41Vision(const Dsv41Vision&) = delete;
    Dsv41Vision& operator=(const Dsv41Vision&) = delete;

    /// The weights `manifest` names (serve/vision_dsv41_plan.py write_manifest: a line "SVM1", then one per tensor -
    /// "T <name> <offset> <bytes> <shape, e.g. 3072x1024> <file>", the file the rest of the line) read into pinned memory,
    /// every tensor checked for its name, BF16 size and shape.  Once per run (loaded() after).
    bool load(const std::string& manifest, std::string& err);
    bool loaded() const { return host_ != nullptr; }
    size_t host_bytes() const { return host_bytes_; }

    /// The aligner cells of a picture of n_vit_h x n_vit_w patches: ceil(n_vit_h / 3) x ceil(n_vit_w / 3)
    static int64_t cells(int n_vit_h, int n_vit_w) {
        return (int64_t) ((n_vit_h + kRatio - 1) / kRatio) * ((n_vit_w + kRatio - 1) / kRatio);
    }
    /// The device workspace encode() needs for such a picture (0: not a picture it takes)
    static size_t workspace_bytes(int n_vit_h, int n_vit_w);

    /// One picture: `patches` (n_vit_h * n_vit_w rows of kPatchIn bf16 values, image_processor's order and rounding) ->
    /// `rows` (cells() x kOut floats, the aligner's rows in reading order; span_rows() adds the delimiters).  On GPU
    /// `stream` with the cuBLAS handle `blas` (its stream set, its math mode pedantic for the call and restored), the weights
    /// copied on `copy_stream`; `workspace` holds workspace_bytes().  Synchronous: `rows` is written when it returns.
    bool encode(void* blas, void* stream, void* copy_stream, void* workspace, size_t ws_bytes, const uint16_t* patches,
                int n_vit_h, int n_vit_w, int n_llm_h, int n_llm_w, float* rows, std::string& err);

    /// image_processor.image_token_types around the aligner's rows (model.py merge_image_embeddings): the image_start row,
    /// each of the n_llm_h rows of n_llm_w cells followed by image_newline, then image_end - n_llm_h (n_llm_w + 1) + 2 rows
    /// of kOut floats into `out`
    void span_rows(const float* rows, int n_llm_h, int n_llm_w, float* out) const;
    static int64_t span_count(int n_llm_h, int n_llm_w) { return (int64_t) n_llm_h * (n_llm_w + 1) + 2; }

private:
    HostAlloc alloc_;
    HostFree free_;
    uint16_t* host_ = nullptr;   ///< the weights, bf16, in the layout of dsv41_vision.cu (Layout)
    size_t host_bytes_ = 0;
};

}  // namespace strata::x
