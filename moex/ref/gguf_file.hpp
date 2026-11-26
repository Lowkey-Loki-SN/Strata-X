// moex/ref/gguf_file.hpp - a GGUF's tensors and keys for the CPU reference: tensor bytes read on demand (pread),
// cached per tensor, metadata through ggml's gguf API.  No ggml context, no backend: the reference does its
// own arithmetic and only borrows ggml-cpu's dot products.
#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct gguf_context;

namespace moex {

struct TensorInfo {
    std::string name;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
    uint64_t off = 0;      ///< absolute file offset of the first byte
    uint64_t nbytes = 0;
    uint64_t row_bytes() const { return ggml_row_size(type, ne[0]); }
};

class GgufFile {
public:
    ~GgufFile();
    bool open(const std::string& path, std::string& err);

    const TensorInfo* info(const std::string& name) const;
    /// The whole tensor, read once and kept.  Aborts (with the name) when the tensor does not exist, because a
    /// missing tensor in the reference is a wiring mistake, never a condition to continue from.
    const uint8_t* data(const std::string& name);
    /// `n` bytes at `off` inside the tensor, uncached (for slices of tensors too big to keep, e.g. experts).
    void read(const TensorInfo& t, uint64_t off, uint64_t n, void* dst) const;
    /// Drop a cached tensor.
    void release(const std::string& name);

    bool has_key(const std::string& key) const;
    uint32_t u32(const std::string& key) const;
    float f32(const std::string& key) const;
    bool boolean(const std::string& key) const;
    std::string str(const std::string& key) const;
    /// Integer or float arrays (and scalars, as a one-element array), converted.
    std::vector<int64_t> iarr(const std::string& key) const;
    std::vector<float> farr(const std::string& key) const;

private:
    gguf_context* g_ = nullptr;
    int fd_ = -1;
    std::unordered_map<std::string, TensorInfo> t_;
    std::unordered_map<std::string, std::vector<uint8_t>> cache_;
    int64_t key_id(const std::string& key) const;
};

}  // namespace moex
