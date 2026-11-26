// moex/ref/gguf_file.cpp - see gguf_file.hpp.
#include "gguf_file.hpp"

#include "gguf.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace moex {

GgufFile::~GgufFile() {
    if (fd_ >= 0) ::close(fd_);
    if (g_) gguf_free(g_);
}

bool GgufFile::open(const std::string& path, std::string& err) {
    ggml_context* meta = nullptr;
    gguf_init_params p{};
    p.no_alloc = true;
    p.ctx = &meta;
    g_ = gguf_init_from_file(path.c_str(), p);
    if (!g_) {
        err = "gguf_init_from_file failed: " + path;
        return false;
    }
    const uint64_t data_off = gguf_get_data_offset(g_);
    const int64_t n = gguf_get_n_tensors(g_);
    for (int64_t i = 0; i < n; ++i) {
        TensorInfo t;
        t.name = gguf_get_tensor_name(g_, i);
        t.type = gguf_get_tensor_type(g_, i);
        t.off = data_off + gguf_get_tensor_offset(g_, i);
        t.nbytes = gguf_get_tensor_size(g_, i);
        const ggml_tensor* mt = ggml_get_tensor(meta, t.name.c_str());
        for (int d = 0; d < 4; ++d) t.ne[d] = mt->ne[d];
        t_[t.name] = t;
    }
    ggml_free(meta);
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        err = "open failed: " + path;
        return false;
    }
    return true;
}

const TensorInfo* GgufFile::info(const std::string& name) const {
    auto it = t_.find(name);
    return it == t_.end() ? nullptr : &it->second;
}

void GgufFile::read(const TensorInfo& t, uint64_t off, uint64_t n, void* dst) const {
    if (off + n > t.nbytes) {
        fprintf(stderr, "moex::GgufFile: read past %s (%llu + %llu > %llu)\n", t.name.c_str(),
                (unsigned long long) off, (unsigned long long) n, (unsigned long long) t.nbytes);
        abort();
    }
    uint8_t* p = static_cast<uint8_t*>(dst);
    uint64_t done = 0;
    while (done < n) {
        const ssize_t r = ::pread(fd_, p + done, n - done, (off_t) (t.off + off + done));
        if (r <= 0) {
            fprintf(stderr, "moex::GgufFile: pread failed on %s\n", t.name.c_str());
            abort();
        }
        done += (uint64_t) r;
    }
}

const uint8_t* GgufFile::data(const std::string& name) {
    auto c = cache_.find(name);
    if (c != cache_.end()) return c->second.data();
    const TensorInfo* t = info(name);
    if (!t) {
        fprintf(stderr, "moex::GgufFile: no tensor %s\n", name.c_str());
        abort();
    }
    std::vector<uint8_t>& v = cache_[name];
    v.resize(t->nbytes);
    read(*t, 0, t->nbytes, v.data());
    return v.data();
}

void GgufFile::release(const std::string& name) { cache_.erase(name); }

int64_t GgufFile::key_id(const std::string& key) const {
    const int64_t id = gguf_find_key(g_, key.c_str());
    if (id < 0) {
        fprintf(stderr, "moex::GgufFile: no key %s\n", key.c_str());
        abort();
    }
    return id;
}

bool GgufFile::has_key(const std::string& key) const { return gguf_find_key(g_, key.c_str()) >= 0; }

uint32_t GgufFile::u32(const std::string& key) const {
    const int64_t id = key_id(key);
    switch (gguf_get_kv_type(g_, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g_, id);
        case GGUF_TYPE_INT32: return (uint32_t) gguf_get_val_i32(g_, id);
        case GGUF_TYPE_UINT64: return (uint32_t) gguf_get_val_u64(g_, id);
        default: fprintf(stderr, "moex::GgufFile: %s is not an integer\n", key.c_str()); abort();
    }
}

float GgufFile::f32(const std::string& key) const {
    const int64_t id = key_id(key);
    if (gguf_get_kv_type(g_, id) == GGUF_TYPE_FLOAT64) return (float) gguf_get_val_f64(g_, id);
    return gguf_get_val_f32(g_, id);
}

bool GgufFile::boolean(const std::string& key) const { return gguf_get_val_bool(g_, key_id(key)); }

std::string GgufFile::str(const std::string& key) const { return gguf_get_val_str(g_, key_id(key)); }

template <typename T>
static std::vector<T> read_arr(const gguf_context* g, int64_t id) {
    std::vector<T> out;
    const gguf_type kt = gguf_get_kv_type(g, id);
    if (kt != GGUF_TYPE_ARRAY) {
        switch (kt) {
            case GGUF_TYPE_UINT32: out.push_back((T) gguf_get_val_u32(g, id)); break;
            case GGUF_TYPE_INT32: out.push_back((T) gguf_get_val_i32(g, id)); break;
            case GGUF_TYPE_FLOAT32: out.push_back((T) gguf_get_val_f32(g, id)); break;
            case GGUF_TYPE_BOOL: out.push_back((T) gguf_get_val_bool(g, id)); break;
            default: abort();
        }
        return out;
    }
    const size_t n = gguf_get_arr_n(g, id);
    const void* d = gguf_get_arr_data(g, id);
    const gguf_type at = gguf_get_arr_type(g, id);
    for (size_t i = 0; i < n; ++i) {
        switch (at) {
            case GGUF_TYPE_UINT32: out.push_back((T) static_cast<const uint32_t*>(d)[i]); break;
            case GGUF_TYPE_INT32: out.push_back((T) static_cast<const int32_t*>(d)[i]); break;
            case GGUF_TYPE_FLOAT32: out.push_back((T) static_cast<const float*>(d)[i]); break;
            case GGUF_TYPE_BOOL: out.push_back((T) static_cast<const int8_t*>(d)[i]); break;
            default: abort();
        }
    }
    return out;
}

std::vector<int64_t> GgufFile::iarr(const std::string& key) const { return read_arr<int64_t>(g_, key_id(key)); }
std::vector<float> GgufFile::farr(const std::string& key) const { return read_arr<float>(g_, key_id(key)); }

}  // namespace moex
