// include/strata/x/safetensors.hpp - moex (2026-11-24): a Hugging Face checkpoint read as it is published - its
// model.safetensors.index.json and every shard's header (or one model.safetensors) - so strata-x can load a release's
// dense tensors and read its experts in place, with no converted copy (moex/docs/ERNIE55_PLAN.md §6).
//
// A shard is 8 bytes of little-endian header length, the JSON header ({name: {dtype, shape, data_offsets [a, b)}},
// "__metadata__" aside), then the data; a tensor's bytes are at 8 + header + a. Every tensor is checked: a known dtype,
// b - a = its elements x the dtype's bytes, inside its file; every name of the index in the shard it names; no name twice.
// A shard shorter than its header implies is "not whole" (a download in progress): an error unless the caller allows
// partial checkpoints, and then its tensors are still listed (whole() says which files can be read).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::st {

enum class DType : uint8_t { BOOL, U8, I8, I16, I32, I64, U16, U32, U64, F16, BF16, F32, F64, F8_E4M3, F8_E5M2, F8_E8M0, Unknown };

size_t dtype_bytes(DType d);   ///< 0 for Unknown
DType dtype_of(const std::string& s);
const char* dtype_name(DType d);

struct Tensor {
    int file = -1;
    DType dtype = DType::Unknown;
    std::vector<int64_t> shape;
    uint64_t offset = 0;   ///< the data's absolute byte offset in its file
    uint64_t nbytes = 0;
    int64_t numel() const;
};

class Checkpoint {
public:
    /// dir holds model.safetensors.index.json and its shards, or a single model.safetensors
    bool open(const std::string& dir, std::string& err, bool allow_partial = false);
    const Tensor* find(const std::string& name) const;
    const std::unordered_map<std::string, Tensor>& tensors() const { return t_; }
    const std::vector<std::string>& files() const { return files_; }
    std::string path(int file) const { return dir_ + "/" + files_[(size_t) file]; }
    uint64_t file_size(int file) const { return size_[(size_t) file]; }
    /// on disk with every byte its header implies (a shard still downloading, or not there yet, is not)
    bool whole(int file) const { return present_[(size_t) file] && size_[(size_t) file] >= data_end_[(size_t) file]; }
    int n_partial() const;
    uint64_t total_bytes() const;
    /// a tensor's bytes into dst (t.nbytes of them), buffered reads; false and err on a short read or a partial file
    bool read(const Tensor& t, void* dst, std::string& err) const;

private:
    std::string dir_;
    std::vector<std::string> files_;
    std::vector<uint64_t> size_, data_end_;
    std::vector<char> present_;   ///< its header read
    std::unordered_map<std::string, Tensor> t_;
    bool read_header(int file, std::string& err);
};

}  // namespace strata::st
