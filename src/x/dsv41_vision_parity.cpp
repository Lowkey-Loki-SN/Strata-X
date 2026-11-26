// src/x/dsv41_vision_parity.cpp - moex (DSV41_PLAN §54): DeepSeek-V4.1's image encoder on the GPU against the CPU
// encoder's rows for the same picture (moex/tools/dsv41/checks/vision_gpu_parity.py writes the inputs and runs it):
//
//     dsv41_vision_parity MANIFEST PATCHES.svp REFERENCE.sve [TOL [OUT.sve]]
//
// MANIFEST: serve/vision_dsv41_plan.py write_manifest's; PATCHES: its write_patches' for the picture; REFERENCE: the SVE1
// record serve/vision_dsv41.py wrote for it (fp32, the fused attention); OUT: the GPU's span written as an SVE1 record (for a
// comparison against a float64 encode: vision_fp64_ref.py).  The picture is encoded twice in one workspace: the
// span's relative L2 error against the reference (at most TOL, default 1e-5), its largest difference, both encodes' times,
// and whether the second encode gave the first one's bytes.  Exit 0 when within TOL and the same twice, 1 otherwise, 2 on an
// input or CUDA error.
#include "strata/x/dsv41_vision.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using strata::x::Dsv41Vision;

namespace {

bool read_file(const char* path, std::vector<char>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(n > 0 ? (size_t) n : 0);
    const bool ok = n >= 0 && std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

int die(const std::string& why) {
    std::fprintf(stderr, "dsv41_vision_parity: %s\n", why.c_str());
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) return die("usage: dsv41_vision_parity MANIFEST PATCHES.svp REFERENCE.sve [TOL]");
    const double tol = argc > 4 ? std::atof(argv[4]) : 1e-5;
    std::vector<char> svp, sve;
    if (!read_file(argv[2], svp) || svp.size() < 24) return die(std::string("cannot read ") + argv[2]);
    if (!read_file(argv[3], sve) || sve.size() < 20) return die(std::string("cannot read ") + argv[3]);
    int32_t ph[6], rh[5];
    std::memcpy(ph, svp.data(), sizeof ph);
    std::memcpy(rh, sve.data(), sizeof rh);
    if (ph[0] != 0x31505653 || ph[5] != Dsv41Vision::kPatchIn) return die("the patches are not an SVP1 file");
    const int nh = ph[1], nw = ph[2], lh = ph[3], lw = ph[4];
    const size_t np = (size_t) nh * nw * Dsv41Vision::kPatchIn;
    if (svp.size() != 24 + np * 2) return die("the patches file's size is not its header's");
    const int64_t span = Dsv41Vision::span_count(lh, lw);
    if (rh[0] != 0x31455653 || rh[1] != span || rh[4] != Dsv41Vision::kOut || sve.size() != 20 + (size_t) span * Dsv41Vision::kOut * 4)
        return die("the reference is not an SVE1 record of this picture's " + std::to_string(span) + " rows");
    const float* ref = reinterpret_cast<const float*>(sve.data() + 20);

    Dsv41Vision enc(
        [](size_t b) -> void* {
            void* p = nullptr;
            return cudaHostAlloc(&p, b, cudaHostAllocPortable) == cudaSuccess ? p : nullptr;
        },
        [](void* p) { cudaFreeHost(p); });
    std::string err;
    const auto tl = std::chrono::steady_clock::now();
    if (!enc.load(argv[1], err)) return die(err);
    const double ms_load = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tl).count();
    cudaStream_t s = nullptr, cs = nullptr;
    cublasHandle_t h = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess || cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) != cudaSuccess ||
        cublasCreate(&h) != CUBLAS_STATUS_SUCCESS)
        return die("no CUDA stream or cuBLAS handle");
    const size_t need = Dsv41Vision::workspace_bytes(nh, nw);
    void* ws = nullptr;
    if (need == 0 || cudaMalloc(&ws, need) != cudaSuccess) return die("the workspace (" + std::to_string(need >> 20) + " MiB) refused");
    const int64_t cells = Dsv41Vision::cells(nh, nw);
    std::vector<float> rows((size_t) cells * Dsv41Vision::kOut), a((size_t) span * Dsv41Vision::kOut), b(a.size());
    const uint16_t* patches = reinterpret_cast<const uint16_t*>(svp.data() + 24);
    double ms[2] = {0.0, 0.0};
    for (int pass = 0; pass < 2; ++pass) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!enc.encode(h, s, cs, ws, need, patches, nh, nw, lh, lw, rows.data(), err)) return die(err);
        ms[pass] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        enc.span_rows(rows.data(), lh, lw, pass == 0 ? a.data() : b.data());
    }
    const bool same = std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    if (argc > 5) {
        FILE* f = std::fopen(argv[5], "wb");
        const int32_t h[5] = {0x31455653, (int32_t) span, (int32_t) span, 1, Dsv41Vision::kOut};
        if (!f || std::fwrite(h, sizeof(int32_t), 5, f) != 5 || std::fwrite(a.data(), sizeof(float), a.size(), f) != a.size())
            return die(std::string("cannot write ") + argv[5]);
        std::fclose(f);
    }
    double num = 0.0, den = 0.0, worst = 0.0;
    bool finite = true;
    for (size_t i = 0; i < a.size(); ++i) {
        finite &= std::isfinite(a[i]);
        const double d = (double) a[i] - (double) ref[i];
        num += d * d;
        den += (double) ref[i] * ref[i];
        worst = std::max(worst, std::fabs(d));
    }
    const double rel = den > 0.0 ? std::sqrt(num / den) : (num > 0.0 ? 1.0 : 0.0);
    const bool pass = finite && same && rel <= tol;
    std::printf("%d x %d patches -> %lld rows: rel_l2 %.3g (tol %.3g), max_abs %.3g, encode %.1f / %.1f ms, workspace %.1f MiB, "
                "weights read in %.0f ms, second encode %s - %s\n",
                nh, nw, (long long) span, rel, tol, worst, ms[0], ms[1], need / 1048576.0, ms_load, same ? "the same bytes" : "DIFFERENT",
                pass ? "PASS" : "FAIL");
    cudaFree(ws);
    cublasDestroy(h);
    cudaStreamDestroy(s);
    cudaStreamDestroy(cs);
    return pass ? 0 : 1;
}
