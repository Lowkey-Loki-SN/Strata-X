// moex/ref/recorder.hpp - the dump writer (llama_dump's format) and the op-level oracle check, shared by the CPU
// reference (glm_ref) and the GPU engine's probe mode (strata-x --dump / --oracle).  Plain C++ and POSIX.
#pragma once

#include <cstdint>
#include <cstdio>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace moex {

/// The dump writer (llama_dump's format), and the OP-LEVEL CHECK.
///
/// With an oracle dump attached (`attach_oracle`), every recorded tensor is compared with the oracle's record of
/// the same (step, name, occurrence) and then OVERWRITTEN with it, before the reference uses it.  So each op runs
/// on the oracle's own inputs and its error is its own: the end-to-end comparison cannot show that, because one
/// ulp in an activation that is re-quantized to Q8_0/Q8_K can flip a rounding and the flip then propagates (step
/// 14 of the first comparison: one element of block 63).  The recorded tensors are the op boundaries, so forcing
/// them also makes every matvec's input bitwise the oracle's.
class Recorder {
public:
    bool open(const std::string& dir, const std::string& match, std::string& err);
    bool attach_oracle(const std::string& dir, std::string& err);
    void close();
    void set_step(int s);
    void f32(const char* name, int il, float* d, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1);
    void i32(const char* name, int il, int32_t* d, int64_t ne0, int64_t ne1 = 1);
    bool active() const { return bin_ != nullptr || oracle_bin_ != nullptr; }
    /// moex: whether a dump would write record `name` of layer `il` (no oracle checks it) - a large probe asks first
    bool wants(const char* name, int il) const;
    /// Per record name, the worst relative error over the steps; returns the number over `tol`.
    int report(double tol, FILE* out) const;

private:
    FILE* bin_ = nullptr;
    FILE* idx_ = nullptr;
    std::regex match_;
    bool match_all_ = true;
    int step_ = 0;
    uint64_t off_ = 0;
    bool want(const std::string& n) const;
    void write(const std::string& name, const char* type, const int64_t ne[4], const void* d, size_t bytes);

    // the oracle: (step, name, occurrence) -> (offset, element count, is_i32), payloads read with pread
    struct ORec { uint64_t off; int64_t n; bool i32; };
    std::unordered_map<std::string, ORec> oracle_;
    int oracle_fd_ = -1;
    const char* oracle_bin_ = nullptr;
    std::unordered_map<std::string, int> occ_;   ///< this step's occurrence counts
    struct Stat { double worst = 0; int step = -1; int n = 0; double sum = 0; };
    std::vector<std::string> order_;
    std::unordered_map<std::string, Stat> stats_;
    std::string key(const std::string& n);       ///< counts the occurrence, returns "step|name|occ"
    void check(const std::string& stat_name, double rel);
};

}  // namespace moex
