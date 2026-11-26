// src/x/park_disk.cpp - moex (DSV41_BACKLOG #53, the restart half): CONVERSATIONS ON DISK (--conversation-dir).
//
// Parking (park.cpp) keeps a conversation the engine would overwrite in RAM.  This keeps the live and parked ones when
// the engine ends: at a clean end (QUIT, the end of its input, SIGINT / SIGTERM in --serve - serve.cpp) each goes to a
// file of its own, and a later engine started the same way finds them (disk_index) and, when a prompt continues one 1,024+
// tokens further than what it holds, reads it into the parked records (disk_offer) for park_try to put back: the same bytes
// in the same buffers, so the request goes on as if the engine had never stopped.  A freeze or a kill writes nothing.
//
// A file: a header (its engine's identity, its length, the hashes of its tokens and of each checkpoint's, the pictures'
// keys, the byte counts), the positional rows per stage, the running state at its end per stage, each checkpoint's state per
// stage, and a trailer (a checksum of the data, the end mark).  The tokens themselves are not written: a prompt matches a
// record when the hashes of its first len tokens are the record's (two independent 64-bit streams), and the record's tokens
// are then the prompt's.  The identity (main.cpp: the binary, its arguments, the STRATA_* and CUDA_VISIBLE_DEVICES
// environment, the model files) keeps another build's or configuration's state out: such a file is deleted at the index, as
// is a damaged one or one older than a week.  The rows' and states' sizes are checked against this engine's geometry too.
#include "engine_impl.hpp"
#include "park_policy.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

namespace strata::x {

namespace {

constexpr char kMagic[8] = {'S', 'X', 'C', 'O', 'N', 'V', '0', '1'};
constexpr char kEnd[8] = {'S', 'X', 'C', 'O', 'N', 'V', 'E', 'N'};
constexpr double kMiB = 1048576.0;
constexpr int64_t kMaxAgeS = 7 * 24 * 3600;
constexpr uint64_t kMinFreeDisk = 8ull << 30;   // a save never leaves the disk with less
constexpr size_t kIo = 16u << 20;               // read / write pieces

uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

/// The hashes of a token list, two independent 64-bit chains.
struct TokHash {
    uint64_t a = 0x243F6A8885A308D3ull, b = 0x13198A2E03707344ull;
    void add(int32_t t) {
        const uint64_t x = (uint32_t) t;
        a = mix64(a ^ (x + 0x9E3779B97F4A7C15ull));
        b = mix64((b + x) * 0xD6E8FEB86659FD93ull);
    }
};

/// The hashes of ids[0, len) for each len of `lens` (ascending, each <= ids.size()).
std::vector<std::array<uint64_t, 2>> tok_hashes(const std::vector<int32_t>& ids, const std::vector<int64_t>& lens) {
    std::vector<std::array<uint64_t, 2>> out(lens.size());
    TokHash h;
    size_t li = 0;
    while (li < lens.size() && lens[li] <= 0) out[li++] = {h.a, h.b};
    for (int64_t i = 0; i < (int64_t) ids.size() && li < lens.size(); ++i) {
        h.add(ids[(size_t) i]);
        while (li < lens.size() && lens[li] == i + 1) out[li++] = {h.a, h.b};
    }
    return out;
}

std::array<uint64_t, 2> text_hash(const std::string& s) {
    TokHash h;
    for (unsigned char c : s) h.add((int32_t) c);
    h.add((int32_t) s.size());
    return {h.a, h.b};
}

/// A checksum of a byte stream, independent of how it is cut into pieces.
struct Sum {
    uint64_t h = 0xCBF29CE484222325ull;
    uint8_t carry[8] = {};
    size_t nc = 0;
    void word(uint64_t w) {
        h = (h ^ w) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 29;
    }
    void add(const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        while (n && nc) {
            carry[nc++] = *b++;
            --n;
            if (nc == 8) {
                uint64_t w;
                std::memcpy(&w, carry, 8);
                word(w);
                nc = 0;
            }
        }
        for (; n >= 8; b += 8, n -= 8) {
            uint64_t w;
            std::memcpy(&w, b, 8);
            word(w);
        }
        while (n) {
            carry[nc++] = *b++;
            --n;
        }
    }
    uint64_t fin() const {
        uint64_t r = h;
        for (size_t i = 0; i < nc; ++i) r = (r ^ carry[i]) * 0x100000001B3ull;
        return mix64(r ^ nc);
    }
};

std::vector<ImageKey> below(const std::vector<ImageKey>& v, int64_t len) {   // the pictures that start inside [0, len)
    std::vector<ImageKey> out;
    for (const ImageKey& k : v)
        if (k.start < len) out.push_back(k);
    return out;
}

bool prefix_of(const std::vector<int32_t>& a, const std::vector<int32_t>& b, size_t n) {   // a[0, n) == b[0, n)
    return a.size() >= n && b.size() >= n && std::equal(a.begin(), a.begin() + (std::ptrdiff_t) n, b.begin());
}

uint64_t mem_available() {
    std::ifstream f("/proc/meminfo");
    std::string key, unit;
    uint64_t kb = 0;
    while (f >> key >> kb >> unit)
        if (key == "MemAvailable:") return kb * 1024;
    return UINT64_MAX;
}

struct Writer {   // the header, built in memory
    std::vector<uint8_t> b;
    template <class T> void put(const T& v) {
        const auto* p = reinterpret_cast<const uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof(T));
    }
    void imgs(const std::vector<ImageKey>& v) {
        put<uint32_t>((uint32_t) v.size());
        for (const ImageKey& k : v) {
            put<int64_t>(k.start);
            put<int64_t>(k.n);
            put<uint64_t>(k.hash);
        }
    }
};

struct Reader {   // ... and parsed from the file's first bytes
    const std::vector<uint8_t>& b;
    size_t at = 0;
    bool ok = true;
    template <class T> T get() {
        T v{};
        if (at + sizeof(T) > b.size()) {
            ok = false;
            return v;
        }
        std::memcpy(&v, b.data() + at, sizeof(T));
        at += sizeof(T);
        return v;
    }
    std::vector<ImageKey> imgs() {
        std::vector<ImageKey> v;
        const uint32_t n = get<uint32_t>();
        if (!ok || n > 65536) {
            ok = false;
            return v;
        }
        for (uint32_t i = 0; i < n && ok; ++i) {
            ImageKey k;
            k.start = get<int64_t>();
            k.n = get<int64_t>();
            k.hash = get<uint64_t>();
            v.push_back(k);
        }
        return v;
    }
};

bool write_all(int fd, const void* p, size_t n, Sum* sum) {
    if (sum) sum->add(p, n);
    const uint8_t* b = static_cast<const uint8_t*>(p);
    while (n) {
        const ssize_t w = ::write(fd, b, std::min(n, kIo));
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return false;
        b += w;
        n -= (size_t) w;
    }
    return true;
}

bool read_all(int fd, void* p, size_t n, Sum* sum) {
    uint8_t* b = static_cast<uint8_t*>(p);
    size_t left = n;
    while (left) {
        const ssize_t r = ::read(fd, b, std::min(left, kIo));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        b += r;
        left -= (size_t) r;
    }
    if (sum) sum->add(p, n);
    return true;
}

bool skip_all(int fd, uint64_t n, std::vector<uint8_t>& scratch, Sum& sum) {   // read and only hashed
    while (n) {
        const size_t k = (size_t) std::min<uint64_t>(n, scratch.size());
        if (!read_all(fd, scratch.data(), k, &sum)) return false;
        n -= k;
    }
    return true;
}

void sync_dir(const std::string& dir) {
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
}

bool make_dirs(const std::string& dir) {   // mkdir -p, the last one private
    std::string at;
    for (size_t i = 0; i <= dir.size(); ++i)
        if (i == dir.size() || (dir[i] == '/' && i > 0)) {
            at = dir.substr(0, i);
            if (::mkdir(at.c_str(), i == dir.size() ? 0700 : 0755) != 0 && errno != EEXIST) return false;
        }
    struct stat sb {};
    return ::stat(dir.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
}

}  // namespace

// ------------------------------------------------------------------------------------------------ the index

void Engine::Impl::disk_index() {
    disk_indexed = true;
    disk.clear();
    if (opt.conv_dir.empty() || !park_on()) return;
    DIR* d = ::opendir(opt.conv_dir.c_str());
    if (!d) return;
    const auto id = text_hash(opt.conv_identity);
    const int64_t now = (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
    uint64_t total = 0;
    int dropped = 0;
    std::vector<std::string> names;
    while (dirent* e = ::readdir(d)) names.emplace_back(e->d_name);
    ::closedir(d);
    for (const std::string& name : names) {
        if (name.rfind("conv-", 0) != 0) continue;
        const std::string path = opt.conv_dir + "/" + name;
        if (name.find(".tmp") != std::string::npos) {   // a write that did not finish
            ::unlink(path.c_str());
            continue;
        }
        if (name.size() < 5 || name.compare(name.size() - 4, 4, ".bin") != 0) continue;
        DiskRec r;
        r.path = path;
        std::string why;
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        struct stat sb {};
        if (fd < 0 || ::fstat(fd, &sb) != 0) {
            if (fd >= 0) ::close(fd);
            continue;
        }
        r.file_bytes = (uint64_t) sb.st_size;
        r.mtime = (int64_t) sb.st_mtime;
        std::vector<uint8_t> head((size_t) std::min<uint64_t>(r.file_bytes, 4u << 20));
        const bool got = read_all(fd, head.data(), head.size(), nullptr);
        char end[8] = {};
        const bool tail = r.file_bytes >= 16 && ::pread(fd, end, 8, (off_t) (r.file_bytes - 8)) == 8;
        ::close(fd);
        Reader rd{head};
        char magic[8] = {};
        if (got) std::memcpy(magic, head.data(), std::min<size_t>(8, head.size()));
        rd.at = 8;
        const uint64_t ih0 = rd.get<uint64_t>(), ih1 = rd.get<uint64_t>();
        const uint32_t ns = rd.get<uint32_t>();
        r.len = rd.get<int64_t>();
        r.h[0] = rd.get<uint64_t>();
        r.h[1] = rd.get<uint64_t>();
        r.turn_pos = rd.get<int64_t>();
        r.imgs = rd.imgs();
        if (!got || std::memcmp(magic, kMagic, 8) != 0 || !tail || std::memcmp(end, kEnd, 8) != 0) why = "not a whole file";
        else if (!rd.ok || ih0 != id[0] || ih1 != id[1]) why = "another engine's (its binary, arguments or model files differ)";
        else if (ns != stages.size() || r.len < 1 || r.len >= opt.n_ctx) why = "another layout";
        else if (now - r.mtime > kMaxAgeS) why = "older than a week";
        if (why.empty()) {
            uint64_t data = 0;
            for (uint32_t si = 0; si < ns; ++si) r.row_bytes.push_back(rd.get<uint64_t>());
            for (uint32_t si = 0; si < ns; ++si) r.snap_bytes.push_back(rd.get<uint64_t>());
            const uint32_t nck = rd.get<uint32_t>();
            for (uint32_t j = 0; j < nck && rd.ok && j < 4096; ++j) {
                DiskCk c;
                c.len = rd.get<int64_t>();
                c.h[0] = rd.get<uint64_t>();
                c.h[1] = rd.get<uint64_t>();
                c.root = rd.get<uint8_t>() != 0;
                c.deep = rd.get<uint8_t>() != 0;
                c.imgs = rd.imgs();
                for (uint32_t si = 0; si < ns; ++si) c.bytes.push_back(rd.get<uint64_t>());
                r.checks.push_back(std::move(c));
            }
            r.data_off = rd.at;
            // the sizes this engine's geometry gives for that length
            bool geo = rd.ok && nck == r.checks.size();
            for (size_t si = 0; si < stages.size() && geo; ++si) {
                std::vector<std::pair<void*, size_t>> reg;
                stages[si]->positional_regions(r.len, reg);
                uint64_t rb = 0;
                for (const auto& x : reg) rb += x.second;
                const uint64_t sb2 = stages[si]->run_state_bytes();
                geo = rb == r.row_bytes[si] && sb2 == r.snap_bytes[si];
                data += r.row_bytes[si] + r.snap_bytes[si];
                for (const DiskCk& c : r.checks) {
                    geo = geo && c.bytes[si] == sb2 && c.len >= 1 && c.len < r.len;
                    data += c.bytes[si];
                }
            }
            r.data_bytes = data;
            if (!geo) why = "another layout";
            else if (r.data_off + data + 16 != r.file_bytes) why = "not a whole file";
        }
        if (!why.empty()) {
            std::fprintf(stderr, "strata-x serve: conversation cache: %s dropped (%s)\n", name.c_str(), why.c_str());
            ::unlink(path.c_str());
            ++dropped;
            continue;
        }
        total += r.file_bytes;
        disk.push_back(std::move(r));
    }
    if (!disk.empty() || dropped)
        std::fprintf(stderr, "strata-x serve: conversation cache: %zu saved conversations on disk (%.0f MiB) in %s\n", disk.size(),
                     total / kMiB, opt.conv_dir.c_str());
}

// ------------------------------------------------------------------------------------------------ putting one back

void Engine::Impl::disk_offer(const std::vector<int32_t>& prompt, const std::vector<ImageKey>& keys, int64_t n, int64_t cur) {
    if (!disk_indexed) disk_index();
    if (disk.empty()) return;
    std::vector<int64_t> lens;
    for (const DiskRec& r : disk) {
        if (r.used) continue;
        if (r.len <= n - 1) lens.push_back(r.len);
        for (const DiskCk& c : r.checks)
            if (c.len <= n - 1) lens.push_back(c.len);
    }
    if (lens.empty()) return;
    std::sort(lens.begin(), lens.end());
    lens.erase(std::unique(lens.begin(), lens.end()), lens.end());
    const auto hs = tok_hashes(prompt, lens);
    auto same = [&](int64_t len, const uint64_t h[2]) {
        const auto it = std::lower_bound(lens.begin(), lens.end(), len);
        if (it == lens.end() || *it != len) return false;
        const auto& x = hs[(size_t) (it - lens.begin())];
        return x[0] == h[0] && x[1] == h[1];
    };
    int best = -1, best_ck = -1;   // best_ck: -1 its end, else the checkpoint it gives
    int64_t best_avail = 0;
    for (size_t i = 0; i < disk.size(); ++i) {
        const DiskRec& r = disk[i];
        if (r.used) continue;
        int64_t avail = 0;
        int ck = -2;
        if (r.len <= n - 1 && same(r.len, r.h) && below(keys, r.len) == r.imgs) {
            avail = r.len;
            ck = -1;
        }
        if (ck != -1)
            for (size_t j = r.checks.size(); j-- > 0;) {
                const DiskCk& c = r.checks[j];
                if (c.len <= n - 1 && c.len > avail && same(c.len, c.h) && below(keys, c.len) == c.imgs) {
                    avail = c.len;
                    ck = (int) j;
                    break;
                }
            }
        if (avail > best_avail) {
            best = (int) i;
            best_avail = avail;
            best_ck = ck;
        }
    }
    if (best < 0 || !park::restore_pays(best_avail, cur)) return;
    for (const auto& p : parked) {   // a parked one that gives as much is put back from RAM (park_try)
        int64_t avail = 0;
        if ((int64_t) p->ids.size() <= n - 1 && prefix_of(p->ids, prompt, p->ids.size()) && below(keys, (int64_t) p->ids.size()) == p->imgs)
            avail = (int64_t) p->ids.size();
        for (const auto& c : p->checks)
            if ((int64_t) c->ids.size() <= n - 1 && (int64_t) c->ids.size() > avail && prefix_of(c->ids, prompt, c->ids.size()) &&
                below(keys, (int64_t) c->ids.size()) == c->imgs)
                avail = (int64_t) c->ids.size();
        if (avail >= best_avail) return;
    }
    DiskRec& r = disk[(size_t) best];
    r.used = true;   // one try: a record that fails to load is not offered again
    const auto t0 = clk::now();
    const int64_t len = best_ck < 0 ? r.len : r.checks[(size_t) best_ck].len;
    // the rows for `len`: each region a prefix of the record's (positional_regions: the same buffers, fewer rows)
    std::vector<std::vector<size_t>> keep(stages.size()), full(stages.size());
    uint64_t need = 0;
    for (size_t si = 0; si < stages.size(); ++si) {
        std::vector<std::pair<void*, size_t>> a, b;
        stages[si]->positional_regions(r.len, a);
        stages[si]->positional_regions(len, b);
        if (a.size() != b.size()) {
            std::fprintf(stderr, "strata-x serve: conversation cache: %s does not fit this engine's regions; not used\n", r.path.c_str());
            return;
        }
        for (size_t j = 0; j < a.size(); ++j) {
            full[si].push_back(a[j].second);
            keep[si].push_back(std::min(a[j].second, b[j].second));
            need += keep[si].back();
        }
        need += r.snap_bytes[si];
        for (const DiskCk& c : r.checks)
            if (c.len < len) need += c.bytes[si];
    }
    const uint64_t floor = (uint64_t) opt.conv_cache_min_free_mib << 20, avail_ram = mem_available();
    if (avail_ram != UINT64_MAX && avail_ram < need + floor) {
        std::fprintf(stderr, "strata-x serve: conversation cache: %lld saved tokens not read back (%.0f MiB, %.0f MiB available, the "
                             "floor %lld MiB)\n",
                     (long long) len, need / kMiB, avail_ram / kMiB, (long long) opt.conv_cache_min_free_mib);
        return;
    }
    const int fd = ::open(r.path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    struct Closer {
        int fd;
        ~Closer() {
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // a one-time read: not kept in the page cache
            ::close(fd);
        }
    } closer{fd};
    Sum sum;
    std::vector<uint8_t> scratch(kIo);
    bool ok = ::lseek(fd, (off_t) r.data_off, SEEK_SET) == (off_t) r.data_off;
    auto P = std::make_unique<Parked>();
    P->ids.assign(prompt.begin(), prompt.begin() + len);
    P->imgs = best_ck < 0 ? r.imgs : r.checks[(size_t) best_ck].imgs;
    P->turn_pos = std::min(r.turn_pos, len);
    P->rows.resize(stages.size());
    P->row_bytes.assign(stages.size(), 0);
    for (size_t si = 0; si < stages.size() && ok; ++si) {   // the rows, each region's first part kept
        uint64_t rb = 0;
        for (size_t x : keep[si]) rb += x;
        P->row_bytes[si] = rb;
        P->rows[si].reset(new (std::nothrow) uint8_t[std::max<uint64_t>(rb, 1)]);
        ok = P->rows[si] != nullptr;
        uint64_t off = 0;
        for (size_t j = 0; j < full[si].size() && ok; ++j) {
            ok = read_all(fd, P->rows[si].get() + off, keep[si][j], &sum) && skip_all(fd, full[si][j] - keep[si][j], scratch, sum);
            off += keep[si][j];
        }
    }
    auto pinned_ck = [&](const std::vector<uint64_t>& bytes, bool read_it) -> std::unique_ptr<Checkpoint> {
        auto c = std::make_unique<Checkpoint>();
        c->blob.assign(stages.size(), nullptr);
        c->bytes.assign(stages.size(), 0);
        for (size_t si = 0; si < stages.size() && ok; ++si) {
            if (!read_it) {
                ok = skip_all(fd, bytes[si], scratch, sum);
                continue;
            }
            set_dev(stages[si]->dev);
            if (cudaMallocHost(&c->blob[si], std::max<uint64_t>(bytes[si], 1)) != cudaSuccess) {
                cudaGetLastError();
                c->blob[si] = nullptr;
                ok = false;
                break;
            }
            c->bytes[si] = (size_t) bytes[si];
            ok = read_all(fd, c->blob[si], bytes[si], &sum);
        }
        return c;
    };
    auto snap_end = pinned_ck(r.snap_bytes, best_ck < 0);   // the running state at its end (used when its end matched)
    std::vector<std::unique_ptr<Checkpoint>> cks;
    for (size_t j = 0; j < r.checks.size() && ok; ++j) {
        const DiskCk& dc = r.checks[j];
        auto c = pinned_ck(dc.bytes, dc.len <= len);
        if (dc.len <= len) {
            c->ids.assign(prompt.begin(), prompt.begin() + dc.len);
            c->imgs = dc.imgs;
            c->root = dc.root;
            c->deep = dc.deep;
            cks.push_back(std::move(c));
        }
    }
    uint64_t want = 0;
    char end[8] = {};
    ok = ok && read_all(fd, &want, sizeof want, nullptr) && read_all(fd, end, 8, nullptr) && std::memcmp(end, kEnd, 8) == 0 &&
         want == sum.fin();
    if (!ok) {
        std::fprintf(stderr, "strata-x serve: conversation cache: %s could not be read back (short, damaged or no pinned RAM); "
                             "deleted\n",
                     r.path.c_str());
        ::unlink(r.path.c_str());
        return;
    }
    if (best_ck < 0) {
        P->snap = std::move(snap_end);
    } else {   // its checkpoint at `len` is the running state at the restored end
        for (size_t j = 0; j < cks.size(); ++j)
            if ((int64_t) cks[j]->ids.size() == len) {
                P->snap = std::move(cks[j]);
                cks.erase(cks.begin() + (std::ptrdiff_t) j);
                break;
            }
    }
    if (!P->snap) return;
    P->snap->ids = P->ids;
    P->snap->imgs = P->imgs;
    for (auto& c : cks) P->checks.push_back(std::move(c));
    P->used = ++park_clock;
    const size_t nck = P->checks.size();
    const double mib = P->bytes() / kMiB;
    parked.push_back(std::move(P));
    std::fprintf(stderr, "strata-x serve: conversation cache: read %lld tokens back from disk (%.0f MiB, %zu checkpoints) in %.0f ms\n",
                 (long long) len, mib, nck, ms_since(t0));
}

// ------------------------------------------------------------------------------------------------ the clean end's save

bool Engine::Impl::disk_save(std::string& err) {
    if (opt.conv_dir.empty() || !park_on()) return true;
    if (!disk_indexed) disk_index();
    if (!make_dirs(opt.conv_dir)) {
        err = "cannot make the conversation directory " + opt.conv_dir;
        std::fprintf(stderr, "strata-x serve: conversation cache: %s\n", err.c_str());
        return false;
    }
    const auto id = text_hash(opt.conv_identity);
    struct Item {
        const Parked* p = nullptr;   // null: the live conversation
        const std::vector<int32_t>* ids = nullptr;
        uint64_t bytes = 0, stamp = 0;
    };
    std::vector<Item> items;
    auto own_checks = [&](const std::vector<int32_t>& ids, const std::vector<ImageKey>& imgs, auto& src) {
        std::vector<const Checkpoint*> v;   // its checkpoints on its own line, ascending
        for (const auto& c : src)
            if (!c->ids.empty() && c->ids.size() < ids.size() && prefix_of(c->ids, ids, c->ids.size()) &&
                below(imgs, (int64_t) c->ids.size()) == c->imgs) {
                bool whole = true;
                for (size_t si = 0; si < stages.size(); ++si) whole = whole && si < c->blob.size() && c->blob[si];
                if (whole) v.push_back(c.get());
            }
        std::sort(v.begin(), v.end(), [](const Checkpoint* a, const Checkpoint* b) { return a->ids.size() < b->ids.size(); });
        v.erase(std::unique(v.begin(), v.end(), [](const Checkpoint* a, const Checkpoint* b) { return a->ids.size() == b->ids.size(); }),
                v.end());
        return v;
    };
    const int64_t min_tokens = std::max<int64_t>(1, opt.conv_cache_min_tokens);
    if (live_valid && (int64_t) live.size() >= min_tokens && (int64_t) live.size() < opt.n_ctx) {
        Item it;
        it.ids = &live;
        it.stamp = UINT64_MAX;
        for (const auto& g : stages) {
            std::vector<std::pair<void*, size_t>> reg;
            g->positional_regions((int64_t) live.size(), reg);
            for (const auto& x : reg) it.bytes += x.second;
            it.bytes += g->run_state_bytes();
        }
        for (const Checkpoint* c : own_checks(live, live_imgs, checks))
            for (size_t x : c->bytes) it.bytes += x;
        items.push_back(it);
    }
    for (const auto& p : parked)
        if ((int64_t) p->ids.size() >= min_tokens && p->snap) items.push_back(Item{p.get(), &p->ids, p->bytes(), p->used});
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.stamp > b.stamp; });
    const uint64_t budget = (uint64_t) std::max<int64_t>(0, opt.conv_dir_mib) << 20;
    uint64_t used = 0;
    std::vector<std::string> written;
    uint8_t* bounce = nullptr;   // the live rows' way out of VRAM
    struct BounceFree {
        uint8_t*& b;
        ~BounceFree() {
            if (b) cudaFreeHost(b);
        }
    } bounce_free{bounce};
    bool all_ok = true;
    for (const Item& it : items) {
        const std::vector<int32_t>& ids = *it.ids;
        const int64_t len = (int64_t) ids.size();
        struct statvfs fs {};
        const uint64_t free_disk = ::statvfs(opt.conv_dir.c_str(), &fs) == 0 ? (uint64_t) fs.f_bavail * fs.f_frsize : 0;
        if (used + it.bytes > budget || free_disk < it.bytes + kMinFreeDisk) {
            std::fprintf(stderr, "strata-x serve: conversation cache: %lld tokens not saved (%.0f MiB: over the %lld MiB of "
                                 "--conversation-dir-mib, or the disk's last %.0f GiB)\n",
                         (long long) len, it.bytes / kMiB, (long long) opt.conv_dir_mib, kMinFreeDisk / (kMiB * 1024));
            continue;
        }
        const auto t0 = clk::now();
        const std::vector<ImageKey>& imgs = it.p ? it.p->imgs : live_imgs;
        const int64_t turn_pos = it.p ? it.p->turn_pos : live_turn_pos;
        std::vector<const Checkpoint*> cks = it.p ? own_checks(ids, imgs, it.p->checks) : own_checks(ids, imgs, checks);
        std::vector<int64_t> lens;
        for (const Checkpoint* c : cks) lens.push_back((int64_t) c->ids.size());
        lens.push_back(len);
        const auto hs = tok_hashes(ids, lens);
        Writer w;
        for (char c : kMagic) w.put<char>(c);
        w.put<uint64_t>(id[0]);
        w.put<uint64_t>(id[1]);
        w.put<uint32_t>((uint32_t) stages.size());
        w.put<int64_t>(len);
        w.put<uint64_t>(hs.back()[0]);
        w.put<uint64_t>(hs.back()[1]);
        w.put<int64_t>(turn_pos);
        w.imgs(imgs);
        std::vector<std::vector<std::pair<void*, size_t>>> reg(stages.size());
        for (size_t si = 0; si < stages.size(); ++si) {
            stages[si]->positional_regions(len, reg[si]);
            uint64_t rb = 0;
            for (const auto& x : reg[si]) rb += x.second;
            w.put<uint64_t>(it.p ? (uint64_t) it.p->row_bytes[si] : rb);
        }
        for (size_t si = 0; si < stages.size(); ++si)
            w.put<uint64_t>(it.p ? (uint64_t) it.p->snap->bytes[si] : (uint64_t) stages[si]->run_state_bytes());
        w.put<uint32_t>((uint32_t) cks.size());
        for (size_t j = 0; j < cks.size(); ++j) {
            w.put<int64_t>((int64_t) cks[j]->ids.size());
            w.put<uint64_t>(hs[j][0]);
            w.put<uint64_t>(hs[j][1]);
            w.put<uint8_t>(cks[j]->root ? 1 : 0);
            w.put<uint8_t>(cks[j]->deep ? 1 : 0);
            w.imgs(cks[j]->imgs);
            for (size_t si = 0; si < stages.size(); ++si) w.put<uint64_t>(cks[j]->bytes[si]);
        }
        char hex[40];
        std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long) hs.back()[0]);
        const std::string fin = opt.conv_dir + "/conv-" + hex + ".bin", tmp = fin + ".tmp." + std::to_string((long long) ::getpid());
        const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        bool ok = fd >= 0 && write_all(fd, w.b.data(), w.b.size(), nullptr);
        Sum sum;
        // the rows: a parked record's from RAM, the live one's out of VRAM through a pinned bounce buffer
        for (size_t si = 0; si < stages.size() && ok; ++si) {
            if (it.p) {
                ok = write_all(fd, it.p->rows[si].get(), it.p->row_bytes[si], &sum);
                continue;
            }
            Stage& g = *stages[si];
            set_dev(g.dev);
            if (!bounce && cudaMallocHost(&bounce, kIo) != cudaSuccess) {
                cudaGetLastError();
                bounce = nullptr;
                ok = false;
                break;
            }
            ok = cudaStreamSynchronize(g.s) == cudaSuccess;
            for (const auto& x : reg[si]) {
                for (size_t off = 0; off < x.second && ok; off += kIo) {
                    const size_t k = std::min(kIo, x.second - off);
                    ok = cudaMemcpyAsync(bounce, static_cast<const uint8_t*>(x.first) + off, k, cudaMemcpyDeviceToHost, g.s) ==
                             cudaSuccess &&
                         cudaStreamSynchronize(g.s) == cudaSuccess && write_all(fd, bounce, k, &sum);
                }
                if (!ok) break;
            }
        }
        // the running state at its end
        std::unique_ptr<Checkpoint> snap_live;
        if (ok && !it.p) {
            snap_live = std::make_unique<Checkpoint>();
            snap_live->blob.assign(stages.size(), nullptr);
            snap_live->bytes.assign(stages.size(), 0);
            for (size_t si = 0; si < stages.size(); ++si) snapshot(*snap_live, (int) si);
        }
        const Checkpoint* snap = it.p ? it.p->snap.get() : snap_live.get();
        for (size_t si = 0; si < stages.size() && ok; ++si)
            ok = snap && si < snap->blob.size() && snap->blob[si] && write_all(fd, snap->blob[si], snap->bytes[si], &sum);
        for (const Checkpoint* c : cks)
            for (size_t si = 0; si < stages.size() && ok; ++si) ok = write_all(fd, c->blob[si], c->bytes[si], &sum);
        const uint64_t sv = sum.fin();
        ok = ok && write_all(fd, &sv, sizeof sv, nullptr) && write_all(fd, kEnd, 8, nullptr) && ::fsync(fd) == 0;
        if (fd >= 0) ::close(fd);
        ok = ok && ::rename(tmp.c_str(), fin.c_str()) == 0;
        if (!ok) {
            cudaGetLastError();
            ::unlink(tmp.c_str());
            std::fprintf(stderr, "strata-x serve: conversation cache: %lld tokens could not be saved (%s)\n", (long long) len,
                         std::strerror(errno));
            all_ok = false;
            continue;
        }
        used += it.bytes;
        written.push_back(fin);
        std::fprintf(stderr, "strata-x serve: conversation cache: saved %lld tokens (%.0f MiB, %zu checkpoints) in %.0f ms to %s\n",
                     (long long) len, it.bytes / kMiB, cks.size(), ms_since(t0), fin.c_str());
    }
    sync_dir(opt.conv_dir);
    // the files of an earlier run: the ones put back are superseded; the others stay while the budget holds them (newest first)
    std::vector<const DiskRec*> old;
    for (const DiskRec& r : disk)
        if (std::find(written.begin(), written.end(), r.path) == written.end()) old.push_back(&r);
    std::sort(old.begin(), old.end(), [](const DiskRec* a, const DiskRec* b) { return a->mtime > b->mtime; });
    for (const DiskRec* r : old) {
        if (!r->used && used + r->file_bytes <= budget) {
            used += r->file_bytes;
            continue;
        }
        ::unlink(r->path.c_str());
    }
    disk_indexed = false;   // the directory has changed
    if (!all_ok) err = "some conversations could not be saved";
    return all_ok;
}

}  // namespace strata::x
