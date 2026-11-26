// src/x/safetensors.cpp - moex (2026-11-24): include/strata/x/safetensors.hpp's reader.
#include "strata/x/safetensors.hpp"

#include "strata/x/json_lite.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <set>

namespace strata::st {

namespace {

constexpr uint64_t kMaxHeader = 256ull << 20;   // the largest header read (ERNIE-5.5's are ~0.1-3.5 MB)

struct Fd {
    int fd = -1;
    ~Fd() {
        if (fd >= 0) ::close(fd);
    }
};

bool pread_all(int fd, void* dst, uint64_t n, uint64_t off, std::string& err) {
    uint8_t* d = static_cast<uint8_t*>(dst);
    while (n > 0) {
        const size_t want = (size_t) std::min<uint64_t>(n, 1ull << 30);
        const ssize_t r = ::pread(fd, d, want, (off_t) off);
        if (r < 0) {
            if (errno == EINTR) continue;
            err = std::string("read: ") + std::strerror(errno);
            return false;
        }
        if (r == 0) {
            err = "a short read (the file ends early)";
            return false;
        }
        d += r;
        off += (uint64_t) r;
        n -= (uint64_t) r;
    }
    return true;
}

bool read_file(const std::string& path, std::string& out, std::string& err) {
    Fd f;
    f.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (f.fd < 0) {
        err = path + ": " + std::strerror(errno);
        return false;
    }
    struct stat st {};
    if (::fstat(f.fd, &st) != 0) {
        err = path + ": " + std::strerror(errno);
        return false;
    }
    out.resize((size_t) st.st_size);
    if (!pread_all(f.fd, out.data(), (uint64_t) st.st_size, 0, err)) {
        err = path + ": " + err;
        return false;
    }
    return true;
}

}  // namespace

size_t dtype_bytes(DType d) {
    switch (d) {
        case DType::BOOL: case DType::U8: case DType::I8: case DType::F8_E4M3: case DType::F8_E5M2: case DType::F8_E8M0: return 1;
        case DType::I16: case DType::U16: case DType::F16: case DType::BF16: return 2;
        case DType::I32: case DType::U32: case DType::F32: return 4;
        case DType::I64: case DType::U64: case DType::F64: return 8;
        default: return 0;
    }
}

DType dtype_of(const std::string& s) {
    static const std::pair<const char*, DType> k[] = {
        {"BOOL", DType::BOOL}, {"U8", DType::U8}, {"I8", DType::I8}, {"I16", DType::I16}, {"I32", DType::I32},
        {"I64", DType::I64}, {"U16", DType::U16}, {"U32", DType::U32}, {"U64", DType::U64}, {"F16", DType::F16},
        {"BF16", DType::BF16}, {"F32", DType::F32}, {"F64", DType::F64}, {"F8_E4M3", DType::F8_E4M3},
        {"F8_E5M2", DType::F8_E5M2}, {"F8_E8M0", DType::F8_E8M0}};
    for (const auto& [n, d] : k)
        if (s == n) return d;
    return DType::Unknown;
}

const char* dtype_name(DType d) {
    switch (d) {
        case DType::BOOL: return "BOOL";
        case DType::U8: return "U8";
        case DType::I8: return "I8";
        case DType::I16: return "I16";
        case DType::I32: return "I32";
        case DType::I64: return "I64";
        case DType::U16: return "U16";
        case DType::U32: return "U32";
        case DType::U64: return "U64";
        case DType::F16: return "F16";
        case DType::BF16: return "BF16";
        case DType::F32: return "F32";
        case DType::F64: return "F64";
        case DType::F8_E4M3: return "F8_E4M3";
        case DType::F8_E5M2: return "F8_E5M2";
        case DType::F8_E8M0: return "F8_E8M0";
        default: return "unknown";
    }
}

int64_t Tensor::numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

const Tensor* Checkpoint::find(const std::string& name) const {
    const auto it = t_.find(name);
    return it == t_.end() ? nullptr : &it->second;
}

int Checkpoint::n_partial() const {
    int n = 0;
    for (size_t f = 0; f < files_.size(); ++f) n += whole((int) f) ? 0 : 1;
    return n;
}

uint64_t Checkpoint::total_bytes() const {
    uint64_t s = 0;
    for (const auto& kv : t_) s += kv.second.nbytes;
    return s;
}

bool Checkpoint::read_header(int file, std::string& err) {
    const std::string p = path(file);
    Fd f;
    f.fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC);
    if (f.fd < 0) {
        err = p + ": " + std::strerror(errno);
        return false;
    }
    struct stat st {};
    if (::fstat(f.fd, &st) != 0) {
        err = p + ": " + std::strerror(errno);
        return false;
    }
    const uint64_t size = (uint64_t) st.st_size;
    uint8_t len8[8];
    if (size < 8 || !pread_all(f.fd, len8, 8, 0, err)) {
        err = p + ": no header length (" + std::to_string(size) + " bytes on disk)";
        return false;
    }
    uint64_t n = 0;
    for (int k = 7; k >= 0; --k) n = (n << 8) | len8[k];
    if (n == 0 || n > kMaxHeader) {
        err = p + ": a header length of " + std::to_string(n) + " bytes";
        return false;
    }
    if (8 + n > size) {
        err = p + ": its " + std::to_string(n) + "-byte header is not on disk yet (" + std::to_string(size) + " bytes)";
        return false;
    }
    std::string h((size_t) n, '\0');
    if (!pread_all(f.fd, h.data(), n, 8, err)) {
        err = p + ": " + err;
        return false;
    }
    json::Value v;
    if (!json::parse(h.data(), h.size(), v, err) || !v.is_obj()) {
        err = p + ": the header: " + (err.empty() ? "not an object" : err);
        return false;
    }
    const uint64_t base = 8 + n;
    uint64_t end = base;
    for (const auto& [name, tv] : v.obj) {
        if (name == "__metadata__") continue;
        const json::Value* dt = tv.find("dtype");
        const json::Value* sh = tv.find("shape");
        const json::Value* off = tv.find("data_offsets");
        if (!dt || !dt->is_str() || !sh || !sh->is_arr() || !off || !off->is_arr() || off->arr.size() != 2) {
            err = p + ": tensor " + name + ": no dtype, shape or data_offsets";
            return false;
        }
        Tensor t;
        t.file = file;
        t.dtype = dtype_of(dt->str);
        if (t.dtype == DType::Unknown) {
            err = p + ": tensor " + name + ": an unknown dtype " + dt->str;
            return false;
        }
        for (const json::Value& d : sh->arr) {
            if (!d.is_num() || !d.is_int || d.i < 0) {
                err = p + ": tensor " + name + ": a bad shape";
                return false;
            }
            t.shape.push_back(d.i);
        }
        const json::Value &a = off->arr[0], &b = off->arr[1];
        if (!a.is_int || !b.is_int || a.i < 0 || b.i < a.i) {
            err = p + ": tensor " + name + ": bad data_offsets";
            return false;
        }
        t.offset = base + (uint64_t) a.i;
        t.nbytes = (uint64_t) (b.i - a.i);
        if (t.nbytes != (uint64_t) t.numel() * dtype_bytes(t.dtype)) {
            err = p + ": tensor " + name + ": " + std::to_string(t.nbytes) + " bytes for " + std::to_string(t.numel()) + " x " +
                  dt->str;
            return false;
        }
        end = std::max(end, t.offset + t.nbytes);
        if (!t_.emplace(name, std::move(t)).second) {
            err = p + ": tensor " + name + " is in two shards";
            return false;
        }
    }
    size_[(size_t) file] = size;
    data_end_[(size_t) file] = end;
    present_[(size_t) file] = 1;
    return true;
}

bool Checkpoint::open(const std::string& dir, std::string& err, bool allow_partial) {
    dir_ = dir;
    files_.clear();
    t_.clear();
    std::string idx;
    std::vector<std::pair<std::string, std::string>> wmap;   // (tensor, shard) from the index
    std::string ierr;
    if (read_file(dir + "/model.safetensors.index.json", idx, ierr)) {
        json::Value v;
        if (!json::parse(idx.data(), idx.size(), v, err)) {
            err = "model.safetensors.index.json: " + err;
            return false;
        }
        const json::Value* wm = v.find("weight_map");
        if (!wm || !wm->is_obj() || wm->obj.empty()) {
            err = "model.safetensors.index.json: no weight_map";
            return false;
        }
        std::set<std::string> fs;
        for (const auto& [name, f] : wm->obj) {
            if (!f.is_str() || f.str.empty() || f.str.find('/') != std::string::npos || f.str.find("..") != std::string::npos) {
                err = "model.safetensors.index.json: tensor " + name + ": a bad shard name";
                return false;
            }
            fs.insert(f.str);
            wmap.emplace_back(name, f.str);
        }
        files_.assign(fs.begin(), fs.end());
    } else {
        struct stat st {};
        if (::stat((dir + "/model.safetensors").c_str(), &st) != 0) {
            err = dir + ": neither model.safetensors.index.json nor model.safetensors";
            return false;
        }
        files_ = {"model.safetensors"};
    }
    size_.assign(files_.size(), 0);
    data_end_.assign(files_.size(), 0);
    present_.assign(files_.size(), 0);
    std::vector<std::string> partial;
    for (size_t f = 0; f < files_.size(); ++f) {
        struct stat st {};
        if (::stat(path((int) f).c_str(), &st) != 0) {   // not there yet (a download keeps it under another name until done)
            partial.push_back(files_[f]);
            continue;
        }
        std::string herr;
        if (!read_header((int) f, herr)) {
            if (allow_partial && herr.find("not on disk yet") != std::string::npos) {   // (its header still arriving)
                partial.push_back(files_[f]);
                continue;
            }
            err = herr;
            return false;
        }
        if (!whole((int) f)) partial.push_back(files_[f]);
    }
    if (!partial.empty() && !allow_partial) {
        err = std::to_string(partial.size()) + " of " + std::to_string(files_.size()) + " shards are not whole yet (" + partial[0] +
              (partial.size() > 1 ? ", ..." : "") + "): the download is not complete";
        return false;
    }
    std::unordered_map<std::string, int> fidx;
    for (size_t f = 0; f < files_.size(); ++f) fidx[files_[f]] = (int) f;
    for (const auto& [name, shard] : wmap) {
        const auto it = t_.find(name);
        const int f = fidx[shard];
        if (it == t_.end()) {
            if (allow_partial && !whole(f)) continue;   // (its header is not on disk yet)
            err = "tensor " + name + ": in the index, not in " + shard;
            return false;
        }
        if (it->second.file != f) {
            err = "tensor " + name + ": the index names " + shard + ", it is in " + files_[(size_t) it->second.file];
            return false;
        }
    }
    return true;
}

bool Checkpoint::read(const Tensor& t, void* dst, std::string& err) const {
    if (t.file < 0 || (size_t) t.file >= files_.size() || !whole(t.file)) {
        err = "a tensor of a shard that is not whole";
        return false;
    }
    Fd f;
    f.fd = ::open(path(t.file).c_str(), O_RDONLY | O_CLOEXEC);
    if (f.fd < 0) {
        err = path(t.file) + ": " + std::strerror(errno);
        return false;
    }
    if (!pread_all(f.fd, dst, t.nbytes, t.offset, err)) {
        err = path(t.file) + ": " + err;
        return false;
    }
    return true;
}

}  // namespace strata::st
