// src/x/vision.cpp - moex (DSV41_PLAN §54): DeepSeek-V4.1-Flash's image encoder in the engine.  The server's VENC
// command between requests (serve.cpp) hands it each new picture's patches (serve/vision_dsv41_plan.py write_patches); the
// pictures are encoded on GPU 0 (strata/x/dsv41_vision.hpp) and each picture's span goes back as an SVE1 rows file, the
// same records serve/vision_dsv41.py writes on the CPU - the server's picture cache, GENI and the prompt path are unchanged.
//
// The workspace (at most ~0.6 GB) is borrowed for the call, from the first GPU of these that has it with kFreeMargin to spare:
// the later GPUs' free VRAM (they keep vram_reserve_later_mib free - the display's 2 GiB on this box: 2,050 MiB after the load),
// then GPU 0's (~0.7 GiB after the load, less after a request); else GPU 0's top slots, lent as a prompt lends them (their
// experts' records into the SSD cache first with STRATA_DSV41_PROMPT_KEEP) and refilled after (STRATA_DSV41_VENC_LEND=1 always
// lends: the check's second arm).  The two GPUs are the same card: either gives a picture the same rows, bit for bit.  Nothing
// of the encoder stays in VRAM between calls; its weights (0.97 GB, bf16) stay in pinned RAM from the first picture on.
#include "engine_impl.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::x {
namespace {

constexpr int32_t kSvp = 0x31505653;   // "SVP1": int32 {magic, n_vit_h, n_vit_w, n_llm_h, n_llm_w, kPatchIn}, then the bf16 patches
constexpr int32_t kSve = 0x31455653;   // "SVE1": int32 {magic, rows, nx, ny, width}, then rows x width floats (read_images)
constexpr int32_t kSvh = 0x31485653;   // "SVH1": int32 {magic, gh, gw, 768}, then Hy4-flash's gh x gw patches of 768 floats
constexpr size_t kFreeMargin = 512ull << 20;   // free VRAM the encoder leaves untouched when it runs there (the desktop's, GPU 1)

struct Picture {
    int nh = 0, nw = 0, lh = 0, lw = 0;
    std::vector<uint16_t> patches;
};

bool read_patches(const std::string& path, Picture& p, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open the picture's patches " + path;
        return false;
    }
    int32_t h[6] = {};
    const bool head = std::fread(h, sizeof(int32_t), 6, f) == 6;
    if (!head || h[0] != kSvp || h[1] < 1 || h[2] < 1 || h[3] < 1 || h[4] < 1 || h[5] != Dsv41Vision::kPatchIn ||
        (int64_t) h[1] * h[2] > Dsv41Vision::kMaxPatches) {
        std::fclose(f);
        err = "the picture's patches " + path + " are not an SVP1 file the encoder takes";
        return false;
    }
    p.nh = h[1], p.nw = h[2], p.lh = h[3], p.lw = h[4];
    const size_t cnt = (size_t) p.nh * p.nw * Dsv41Vision::kPatchIn;
    p.patches.resize(cnt);
    const bool body = std::fread(p.patches.data(), sizeof(uint16_t), cnt, f) == cnt;
    const bool end = body && std::fgetc(f) == EOF;
    std::fclose(f);
    if (!body || !end) {
        err = "the picture's patches " + path + (body ? " run past their header's size" : " are short");
        return false;
    }
    if (Dsv41Vision::cells(p.nh, p.nw) != (int64_t) p.lh * p.lw || p.lh != (p.nh + Dsv41Vision::kRatio - 1) / Dsv41Vision::kRatio) {
        err = "the picture's patches " + path + ": cells and patches disagree";
        return false;
    }
    return true;
}

/// moex: Hy4-flash's picture (serve/vision_hy4_plan.py write_patches)
struct HyPicture {
    int gh = 0, gw = 0;
    std::vector<float> patches;
};

bool read_hy_patches(const std::string& path, HyPicture& p, std::string& err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open the picture's patches " + path;
        return false;
    }
    int32_t h[4] = {};
    const bool head = std::fread(h, sizeof(int32_t), 4, f) == 4;
    if (!head || h[0] != kSvh || h[3] != Hy4Vision::kPatchIn || Hy4Vision::workspace_bytes(h[1], h[2]) == 0) {
        std::fclose(f);
        err = "the picture's patches " + path + " are not an SVH1 file the encoder takes";
        return false;
    }
    p.gh = h[1], p.gw = h[2];
    const size_t cnt = (size_t) p.gh * p.gw * Hy4Vision::kPatchIn;
    p.patches.resize(cnt);
    const bool body = std::fread(p.patches.data(), sizeof(float), cnt, f) == cnt;
    const bool end = body && std::fgetc(f) == EOF;
    std::fclose(f);
    if (!body || !end) {
        err = "the picture's patches " + path + (body ? " run past their header's size" : " are short");
        return false;
    }
    return true;
}

/// The rows file, written whole before its name appears (a reader never sees half of one): n rows of `width`, the record's grid
/// nx x ny (n x 1: a span, DeepSeek-V4.1's)
bool write_rows(const std::string& path, const float* rows, int64_t n, std::string& err, int64_t nx = 0, int64_t ny = 1,
                int width = Dsv41Vision::kOut) {
    const std::string part = path + ".part";
    FILE* f = std::fopen(part.c_str(), "wb");
    if (!f) {
        err = "cannot write the picture's rows " + part;
        return false;
    }
    if (nx <= 0) nx = n, ny = 1;
    const int32_t h[5] = {kSve, (int32_t) n, (int32_t) nx, (int32_t) ny, width};
    const size_t cnt = (size_t) n * width;
    const bool ok = std::fwrite(h, sizeof(int32_t), 5, f) == 5 && std::fwrite(rows, sizeof(float), cnt, f) == cnt;
    if (std::fclose(f) != 0 || !ok || std::rename(part.c_str(), path.c_str()) != 0) {
        std::remove(part.c_str());
        err = "cannot write the picture's rows " + path;
        return false;
    }
    return true;
}

}  // namespace

bool Engine::Impl::encode_pictures(const std::vector<std::pair<std::string, std::string>>& files, std::vector<int64_t>& rows,
                                   double& ms, std::string& how, std::string& err) {
    const auto t0 = clk::now();
    rows.clear();
    ms = 0.0;
    how.clear();
    if (opt.vision_manifest.empty()) return fail(err, "this engine was started without --vision-weights (no image encoder)");
    if (stages.empty() || files.empty()) return fail(err, "no picture to encode");
    // every picture read and checked before anything is borrowed: a bad file refuses the call as it is
    const bool hy = hp.arch == Arch::hy4;   // Hy4-flash: SVH1 patches, Hy4Vision (the mmproj GGUF's weights)
    std::vector<Picture> pics(hy ? 0 : files.size());
    std::vector<HyPicture> hpics(hy ? files.size() : 0);
    size_t need = 0;
    int64_t most = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        if (hy) {
            if (!read_hy_patches(files[i].first, hpics[i], err)) return false;
            need = std::max(need, Hy4Vision::workspace_bytes(hpics[i].gh, hpics[i].gw));
            most = std::max(most, Hy4Vision::rows_of(hpics[i].gh, hpics[i].gw));
            continue;
        }
        if (!read_patches(files[i].first, pics[i], err)) return false;
        need = std::max(need, Dsv41Vision::workspace_bytes(pics[i].nh, pics[i].nw));
        most = std::max(most, Dsv41Vision::cells(pics[i].nh, pics[i].nw));
    }
    if (hy) {
        if (!hvision) hvision = std::make_unique<Hy4Vision>([](size_t b) { return host_buf(b); }, [](void* p) { host_free(p); });
        if (!hvision->loaded()) {
            const auto tl = clk::now();
            try {
                const strata::GgufFile g(opt.vision_manifest);
                auto src = [&](const std::string& name, uint32_t& type, std::vector<uint64_t>& shape, const uint8_t*& data) {
                    const strata::TensorInfo* t = g.find(name);
                    if (!t) return false;
                    type = t->type;
                    shape = t->shape;
                    if (g.data_start() + t->offset + t->elements() * (t->type == 30 ? 2 : 4) > g.file_size()) return false;
                    data = g.tensor_data(*t);
                    return true;
                };
                if (!hvision->load(src, err)) return false;
            } catch (const std::exception& e) {
                return fail(err, "the image encoder's weights: " + opt.vision_manifest + ": " + e.what());
            }
            fprintf(stderr, "strata-x: the image encoder's weights read in %.0f ms (%.0f MiB, pinned)\n", ms_since(tl),
                    hvision->host_bytes() / 1048576.0);
        }
    } else {
        if (!vision) vision = std::make_unique<Dsv41Vision>([](size_t b) { return host_buf(b); }, [](void* p) { host_free(p); });
        if (!vision->loaded()) {
            const auto tl = clk::now();
            if (!vision->load(opt.vision_manifest, err)) return false;
            fprintf(stderr, "strata-x: the image encoder's weights read in %.0f ms (%.0f MiB, pinned)\n", ms_since(tl),
                    vision->host_bytes() / 1048576.0);
        }
    }
    // the workspace: a GPU's free VRAM with room to spare, the later GPUs first; else GPU 0's top slots
    static const bool force_lend = std::getenv("STRATA_DSV41_VENC_LEND") && std::getenv("STRATA_DSV41_VENC_LEND")[0] == '1';
    void* ws = nullptr;
    Stage* gw = nullptr;   // the GPU whose free VRAM holds the workspace (null: GPU 0's lent slots)
    std::string frees;     // each GPU's free VRAM, for a refusal's message
    for (size_t k = stages.size(); k-- > 0 && !gw && !force_lend;) {
        Stage& c = *stages[k];
        cudaSetDevice(c.dev);
        size_t fr = 0, tot = 0;
        if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) {
            cudaGetLastError();
            continue;
        }
        frees += (frees.empty() ? "" : ", ") + std::string("GPU ") + std::to_string(c.dev) + " " + std::to_string(fr >> 20) + " MiB";
        if (fr < need + kFreeMargin) continue;
        if (cudaMalloc(&ws, need) != cudaSuccess) {
            cudaGetLastError();   // (refused after all: the next GPU, or the slots)
            ws = nullptr;
            continue;
        }
        gw = &c;
    }
    Stage& g = gw ? *gw : *stages[0];
    cudaSetDevice(g.dev);
    if (cudaStreamSynchronize(g.s) != cudaSuccess || cudaStreamSynchronize(g.cs) != cudaSuccess) {
        if (gw) cudaFree(ws);
        return fail(err, "a stream of GPU " + std::to_string(g.dev) + " failed before the image encoder");
    }
    int64_t lent = 0;
    if (!gw) {
        if (g.lent != 0) return fail(err, "GPU " + std::to_string(g.dev) + "'s slots are lent already");
        const std::string room = "the image encoder's workspace (" + std::to_string(need >> 20) + " MiB) fits neither a GPU's free VRAM (" +
                                 (frees.empty() ? std::string("not asked") : frees) + ", " + std::to_string(kFreeMargin >> 20) +
                                 " MiB kept spare) nor GPU " + std::to_string(g.dev) + "'s ";
        if (g.n_slots <= 0 || slot_bytes == 0) return fail(err, room + "expert slots (none)");
        finalize_swaps(true);   // a copy in flight into a slot lands before the slot is lent
        prompt_keep_reset();
        bool failed = false;
        ws = g.lend_top(need, g.lendable(), failed, err);
        lent = g.lent;
        if (!ws) {
            if (g.lent > 0) {
                g.refill_issue();
                g.refill_finish();
            }
            prompt_keep_reset();
            return failed ? false : fail(err, room + "lendable slots");
        }
    }
    std::vector<float> cells((size_t) most * (hy ? Hy4Vision::kOut : Dsv41Vision::kOut)), span;
    bool ok = true;
    for (size_t i = 0; i < hpics.size() && ok; ++i) {   // Hy4-flash: the rows are the record, its grid (ow + 1) x oh
        const HyPicture& p = hpics[i];
        const auto tp = clk::now();
        ok = hvision->encode(g.blas, g.s, g.cs, ws, need, p.patches.data(), p.gh, p.gw, cells.data(), err);
        if (!ok) break;
        const int64_t n = Hy4Vision::rows_of(p.gh, p.gw);
        ok = write_rows(files[i].second, cells.data(), n, err, p.gw / Hy4Vision::kMerge + 1, p.gh / Hy4Vision::kMerge, Hy4Vision::kOut);
        rows.push_back(n);
        fprintf(stderr, "strata-x: a picture of %d x %d patches -> %lld rows on GPU %d in %.0f ms\n", p.gh, p.gw, (long long) n, g.dev,
                ms_since(tp));
    }
    for (size_t i = 0; i < pics.size() && ok; ++i) {
        const Picture& p = pics[i];
        const auto tp = clk::now();
        ok = vision->encode(g.blas, g.s, g.cs, ws, need, p.patches.data(), p.nh, p.nw, p.lh, p.lw, cells.data(), err);
        if (!ok) break;
        const int64_t n = Dsv41Vision::span_count(p.lh, p.lw);
        span.resize((size_t) n * Dsv41Vision::kOut);
        vision->span_rows(cells.data(), p.lh, p.lw, span.data());
        ok = write_rows(files[i].second, span.data(), n, err);
        rows.push_back(n);
        fprintf(stderr, "strata-x: a picture of %d x %d patches -> %lld rows on GPU %d in %.0f ms\n", p.nh, p.nw, (long long) n, g.dev,
                ms_since(tp));
    }
    if (gw) {
        cudaFree(ws);
    } else {
        tail_ahead_stop();   // (a no-op between requests: the prompt's read-ahead is over)
        g.refill_issue();
        g.refill_finish();
        prompt_keep_reset();
    }
    how = gw ? "free=gpu" + std::to_string(g.dev) : "lent=" + std::to_string(lent);
    ms = ms_since(t0);
    if (const cudaError_t ce = cudaGetLastError(); ok && ce != cudaSuccess) {
        ok = false;
        err = std::string("the image encoder: ") + cudaGetErrorString(ce);
    }
    if (!ok && err.empty()) err = "the image encoder failed";
    return ok;
}

}  // namespace strata::x
