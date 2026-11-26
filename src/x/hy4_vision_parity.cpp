// src/x/hy4_vision_parity.cpp - moex: Hy4-flash's image encoder on the GPU (strata/x/hy4_vision.hpp) for one picture's
// patches, outside the engine, so a check can hold its rows against the CPU encoder's (serve/vision_hy4.py) - the check is
// moex/tools/hy4/vision_gpu_parity.py.
//
//   hy4_vision_parity MMPROJ.gguf PATCHES.svh OUT.sve [device]   (exit 0: written; 1: refused, the reason on stderr)
//
// OUT: one SVE1 record, as the engine writes it (int32 {magic, rows, ow + 1, oh, 3072}, then the rows).  The weights' load
// and the encode's times go to stderr; the encode is run twice and the two outputs compared bit for bit (deterministic).
#include "strata/artifact/gguf_reader.hpp"
#include "strata/x/hy4_vision.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using strata::x::Hy4Vision;

static int die(const std::string& why) {
    std::fprintf(stderr, "hy4_vision_parity: %s\n", why.c_str());
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 4) return die("usage: hy4_vision_parity MMPROJ.gguf PATCHES.svh OUT.sve [device]");
    const int dev = argc > 4 ? std::atoi(argv[4]) : 0;
    if (cudaSetDevice(dev) != cudaSuccess) return die("no GPU " + std::to_string(dev));
    FILE* f = std::fopen(argv[2], "rb");
    int32_t h[4] = {};
    if (!f || std::fread(h, 4, 4, f) != 4 || h[0] != 0x31485653 || h[3] != Hy4Vision::kPatchIn) return die("not an SVH1 file");
    const int gh = h[1], gw = h[2];
    std::vector<float> patches((size_t) gh * gw * Hy4Vision::kPatchIn);
    if (std::fread(patches.data(), 4, patches.size(), f) != patches.size()) return die("short patches");
    std::fclose(f);
    Hy4Vision enc([](size_t b) { void* p = nullptr; return cudaMallocHost(&p, b) == cudaSuccess ? p : nullptr; },
                  [](void* p) { cudaFreeHost(p); });
    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    {
        const strata::GgufFile g(argv[1]);
        auto src = [&](const std::string& name, uint32_t& type, std::vector<uint64_t>& shape, const uint8_t*& data) {
            const strata::TensorInfo* t = g.find(name);
            if (!t) return false;
            type = t->type;
            shape = t->shape;
            data = g.tensor_data(*t);
            return true;
        };
        if (!enc.load(src, err)) return die(err);
    }
    auto ms = [](auto a) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count(); };
    std::fprintf(stderr, "weights %.0f MiB pinned in %.0f ms\n", enc.host_bytes() / 1048576.0, ms(t0));
    const size_t need = Hy4Vision::workspace_bytes(gh, gw);
    if (need == 0) return die("a picture the encoder does not take");
    void* ws = nullptr;
    if (cudaMalloc(&ws, need) != cudaSuccess) return die("no " + std::to_string(need >> 20) + " MiB of workspace on GPU " + std::to_string(dev));
    cublasHandle_t blas = nullptr;
    cudaStream_t s = nullptr, cs = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess)
        return die("cuBLAS / streams");
    const int64_t n = Hy4Vision::rows_of(gh, gw);
    std::vector<float> rows((size_t) n * Hy4Vision::kOut), again(rows.size());
    for (int run = 0; run < 2; ++run) {
        t0 = std::chrono::steady_clock::now();
        if (!enc.encode(blas, s, cs, ws, need, patches.data(), gh, gw, run ? again.data() : rows.data(), err)) return die(err);
        std::fprintf(stderr, "encode %d: %d x %d patches -> %lld rows in %.1f ms (workspace %.0f MiB)\n", run, gh, gw, (long long) n, ms(t0),
                     need / 1048576.0);
    }
    if (std::memcmp(rows.data(), again.data(), rows.size() * 4)) return die("two encodes of the same patches differ");
    FILE* o = std::fopen(argv[3], "wb");
    const int32_t hh[5] = {0x31455653, (int32_t) n, gw / 2 + 1, gh / 2, Hy4Vision::kOut};
    if (!o || std::fwrite(hh, 4, 5, o) != 5 || std::fwrite(rows.data(), 4, rows.size(), o) != rows.size() || std::fclose(o)) return die("write");
    std::fprintf(stderr, "deterministic: the two encodes are bit for bit the same\n");
    return 0;
}
