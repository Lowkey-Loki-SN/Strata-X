// src/x/serve.cpp - see include/strata/x/serve.hpp.  The protocol lives here; everything about a request
// (reuse, checkpoints, reading, sampling) is Engine::run_request.
#include "strata/x/serve.hpp"

#include "drive_guard.hpp"   // moex/tools/dsv41: the drive guard's counters for the DONE line (DSV41_BACKLOG #148)

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/sysmacros.h>
#endif

#include <atomic>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace strata::x {

namespace {

/// moex (DSV41_BACKLOG #53): SIGINT and SIGTERM end --serve the way QUIT does - the running request stops, the conversations
/// and the profile are saved, the engine ends.  Ctrl+C in the server's terminal reached the engine too (the same process
/// group) and killed it before the server's QUIT, so neither was ever saved; and a shutdown's SIGTERM did the same.  The
/// handler only sets flags (the serve loop looks every 0.5 s); once saved, the defaults are back (a second signal ends it).
std::atomic<bool> g_quit_signal{false};
std::atomic<std::atomic<bool>*> g_req_stop{nullptr};
extern "C" void on_quit_signal(int) {
    g_quit_signal.store(true);
    if (std::atomic<bool>* s = g_req_stop.load()) s->store(true);
}

struct Commands {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
};

bool parse_ids(const char* s, std::vector<int32_t>& out, std::string& err) {
    out.clear();
    while (*s == ' ') ++s;
    if (!*s) {
        err = "no ids";
        return false;
    }
    char* end = nullptr;
    for (;;) {
        const long long v = std::strtoll(s, &end, 10);
        if (end == s) {
            err = "an id is not a number";
            return false;
        }
        if (v < 0 || v > 0x7fffffffLL) {
            err = "an id is out of range";
            return false;
        }
        out.push_back((int32_t) v);
        s = end;
        while (*s == ' ') ++s;
        if (*s == ',') {
            ++s;
            continue;
        }
        break;
    }
    if (*s) {
        err = "trailing text after the ids";
        return false;
    }
    return true;
}

uint64_t fnv1a(const void* p, size_t n, uint64_t h) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

/// moex (DSV41_PLAN §56): a picture's rows file mapped read-only for one request (a manifest's spans point into it).
struct MappedRows {
    void* base = nullptr;
    size_t size = 0;
    MappedRows() = default;
    MappedRows(void* b, size_t s) : base(b), size(s) {}
    MappedRows(const MappedRows&) = delete;
    MappedRows& operator=(const MappedRows&) = delete;
    MappedRows(MappedRows&& o) noexcept : base(o.base), size(o.size) { o.base = nullptr, o.size = 0; }
    ~MappedRows() {
        if (base) munmap(base, size);
    }
};

/// moex (DSV41_PLAN §56): a GENI manifest - "SVM1", then a line a picture in prompt order: `skip n nx ny key path` (the rows
/// to leave out at the record's start, the rows used, the record's grid after the skip, a 64-bit key of the record's content
/// in hex, the SVE1 file).  Each file's header is checked against its line and the file mapped: only the rows the prompt
/// path reads are ever read, and the server's key replaces hashing the rows.  -> the records, rows pointing into `maps`.
struct ImgRec {
    int64_t n, nx, ny;
    const float* rows;
    size_t off;     // the SVE1 file's: the offset into `rows` (its pointer is set once the file is read)
    bool keyed;     // a manifest's: `key` is the server's
    uint64_t key;
};

bool read_manifest(FILE* f, const std::string& path, int n_embd, std::vector<MappedRows>& maps, std::vector<ImgRec>& recs,
                   std::string& err) {
    char line[4352];
    int ln = 1;
    while (std::fgets(line, sizeof line, f)) {
        ++ln;
        size_t len = std::strlen(line);
        if (len == sizeof line - 1 && line[len - 1] != '\n') {
            err = "a line of the image manifest " + path + " is too long";
            return false;
        }
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        long long skip = -1, n = -1, nx = -1, ny = -1;
        unsigned long long key = 0;
        int used = 0;
        if (std::sscanf(line, "%lld %lld %lld %lld %llx %n", &skip, &n, &nx, &ny, &key, &used) != 5 || used <= 0 ||
            line[used] == '\0') {
            err = "line " + std::to_string(ln) + " of the image manifest " + path + " is not `skip n nx ny key path`";
            return false;
        }
        const std::string file(line + used);
        const int fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            err = "cannot open the image rows file " + file;
            return false;
        }
        struct stat sb {};
        int32_t h[5] = {0, 0, 0, 0, 0};
        const bool head = ::fstat(fd, &sb) == 0 && ::pread(fd, h, sizeof h, 0) == (ssize_t) sizeof h;
        const int64_t total = head ? h[1] : 0;
        const bool ok = head && h[0] == 0x31455653 && total >= 1 && h[2] >= 1 && h[3] >= 1 && (int64_t) h[2] * h[3] == total &&
                        h[4] == n_embd && (int64_t) sb.st_size == 20 + total * n_embd * (int64_t) sizeof(float) && skip >= 0 &&
                        skip < total && n == total - skip &&
                        (skip == 0 ? nx == h[2] && ny == h[3] : nx == n && ny == 1);   // a skip leaves a 1-D record
        if (!ok) {
            ::close(fd);
            err = "the image rows file " + file + " does not match its manifest line (" + std::to_string(ln) + ")";
            return false;
        }
        void* base = ::mmap(nullptr, (size_t) sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (base == MAP_FAILED) {
            err = "cannot map the image rows file " + file;
            return false;
        }
        maps.emplace_back(base, (size_t) sb.st_size);
        const float* rows = reinterpret_cast<const float*>(static_cast<const uint8_t*>(base) + 20) + (size_t) skip * n_embd;
        recs.push_back({n, nx, ny, rows, 0, true, (uint64_t) key});
    }
    if (std::ferror(f)) {
        err = "cannot read the image manifest " + path;
        return false;
    }
    return true;
}

/// moex vision, GENI: the embeddings file (Strata's SVE1 records, one per image in prompt order: int32 {'SVE1', n, nx, ny,
/// n_embd} then n x n_embd floats) against the prompt's pad runs - each run of `pad` ids takes the next record's n rows
/// (two pictures back to back make one run, taken record by record); every record used, none left over, no picture on the
/// last prompt token (it is read by the first decode window as a token).  A picture's key is FNV-1a over {n, nx, ny} and its
/// rows, as Strata's GENI.  `rows` owns the floats the spans point into.  moex (DSV41_PLAN §56): or a manifest of the
/// pictures' own files ("SVM1", read_manifest), the spans pointing into `maps` and keyed by the server.
bool read_images(const std::string& path, const std::vector<int32_t>& ids, int32_t pad, int n_embd, std::vector<float>& rows,
                 std::vector<MappedRows>& maps, std::vector<ImageSpan>& spans, std::string& err) {
    std::vector<ImgRec> recs;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open the embeddings file " + path;
        return false;
    }
    char magic[5] = {0, 0, 0, 0, 0};
    const bool manifest = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "SVM1", 4) == 0;
    if (manifest) {
        const int c = std::fgetc(f);   // the rest of the first line
        const bool ok = (c == '\n' || (c == '\r' && std::fgetc(f) == '\n')) && read_manifest(f, path, n_embd, maps, recs, err);
        std::fclose(f);
        if (!ok) {
            if (err.empty()) err = "the image manifest " + path + " does not start with a line of its own";
            return false;
        }
    } else {
        std::rewind(f);
    }
    for (; !manifest;) {
        int32_t h[5];
        const size_t got = std::fread(h, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || h[0] != 0x31455653 || h[1] < 1 || h[2] < 1 || h[3] < 1 || (int64_t) h[2] * h[3] != h[1]) {
            std::fclose(f);
            err = "the embeddings file is not a list of SVE1 records";
            return false;
        }
        if (h[4] != n_embd) {
            std::fclose(f);
            err = "an image's rows are " + std::to_string(h[4]) + " wide, the model's embedding " + std::to_string(n_embd);
            return false;
        }
        const size_t off = rows.size(), cnt = (size_t) h[1] * (size_t) n_embd;
        rows.resize(off + cnt);
        if (std::fread(rows.data() + off, sizeof(float), cnt, f) != cnt) {
            std::fclose(f);
            err = "the embeddings file is short";
            return false;
        }
        recs.push_back({h[1], h[2], h[3], nullptr, off, false, 0});
    }
    if (!manifest) std::fclose(f);
    size_t k = 0;
    const int64_t n = (int64_t) ids.size();
    for (int64_t i = 0; i < n;) {
        if (ids[(size_t) i] != pad) {
            ++i;
            continue;
        }
        if (k == recs.size()) {
            err = "the prompt and its images do not match (more pad tokens than image rows)";
            return false;
        }
        const ImgRec& r = recs[k++];
        if (i + r.n > n - 1) {
            err = "an image reaches the prompt's last token";
            return false;
        }
        for (int64_t j = 0; j < r.n; ++j)
            if (ids[(size_t) (i + j)] != pad) {
                err = "the prompt and its images do not match (a pad run is shorter than its image)";
                return false;
            }
        uint64_t hv = r.key;
        if (!r.keyed) {
            hv = 1469598103934665603ull;
            const int64_t dims[3] = {r.n, r.nx, r.ny};
            hv = fnv1a(dims, sizeof dims, hv);
            hv = fnv1a(rows.data() + r.off, (size_t) r.n * n_embd * sizeof(float), hv);
        }
        spans.push_back(ImageSpan{i, r.n, hv, nullptr, r.nx, r.ny});
        i += r.n;
    }
    if (k != recs.size()) {
        err = "the prompt and its images do not match (image rows left over)";
        return false;
    }
    for (size_t j = 0; j < spans.size(); ++j)   // rows no longer grows
        spans[j].rows = recs[j].keyed ? recs[j].rows : rows.data() + recs[j].off;
    return true;
}

/// moex: the drive and the box while a request's prompt is read (Linux) - the counters of the pack's disk
/// (/sys/dev/block/M:m/stat, the whole disk's when the pack is on a partition: fields 1, 3, 4, 7 and 10) and the CPU and
/// memory pressure stall totals (/proc/pressure, the "some" lines, microseconds).  A slow read then shows from the log
/// whether the drive was slow (its rate down while it was busy), something else wrote, or the box ran short of CPU or memory.
struct BoxStat {
    bool disk = false;
    unsigned long long rd_ios = 0, rd_sec = 0, rd_ms = 0, wr_sec = 0, io_ms = 0, cpu_us = 0, mem_us = 0;
    std::chrono::steady_clock::time_point t;
};

std::string disk_stat_path(const std::string& dir) {
#ifdef __linux__
    struct stat sb {};
    if (dir.empty() || ::stat(dir.c_str(), &sb) != 0) return "";
    const std::string base = "/sys/dev/block/" + std::to_string(major(sb.st_dev)) + ":" + std::to_string(minor(sb.st_dev));
    struct stat pb {};
    if (::stat((base + "/partition").c_str(), &pb) == 0) return base + "/../stat";   // the link's target's parent: the disk
    if (::stat((base + "/stat").c_str(), &pb) == 0) return base + "/stat";
#else
    (void) dir;
#endif
    return "";
}

BoxStat box_stat(const std::string& disk) {
    BoxStat b;
    b.t = std::chrono::steady_clock::now();
#ifdef __linux__
    if (FILE* f = disk.empty() ? nullptr : std::fopen(disk.c_str(), "r")) {
        unsigned long long v[10] = {};
        b.disk = std::fscanf(f, "%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5],
                             &v[6], &v[7], &v[8], &v[9]) == 10;
        std::fclose(f);
        b.rd_ios = v[0], b.rd_sec = v[2], b.rd_ms = v[3], b.wr_sec = v[6], b.io_ms = v[9];
    }
    auto some = [](const char* path) {
        unsigned long long us = 0;
        if (FILE* f = std::fopen(path, "r")) {
            char line[256];
            if (std::fgets(line, sizeof line, f))
                if (const char* p = std::strstr(line, "total=")) us = std::strtoull(p + 6, nullptr, 10);
            std::fclose(f);
        }
        return us;
    };
    b.cpu_us = some("/proc/pressure/cpu");
    b.mem_us = some("/proc/pressure/memory");
#else
    (void) disk;
#endif
    return b;
}

}  // namespace

int serve(Engine& eng, const ServeOptions& so) {
    // stdout is the protocol: one line per message, flushed (the server reads it on a thread and answers heartbeats)
    setvbuf(stdout, nullptr, _IOLBF, 0);
    // moex: STRATA_X_QUIET=1 (the server's "diagnostics": false) - a request's statistics (the drive's and the box's
    // counters, its lines) are neither gathered nor written; main.cpp drops stderr itself
    static const bool quiet = std::getenv("STRATA_X_QUIET") && std::getenv("STRATA_X_QUIET")[0] == '1';
    const std::string disk = quiet ? std::string() : disk_stat_path(so.pack);
    Commands q;
    std::atomic<bool> stop{false};
    // stdin is read with read(2) on a detached thread: a blocked stdio read would hold the stdin lock that exit() waits for
    std::thread([&q, &stop] {
        std::string line;
        char buf[65536];
        for (;;) {
            const ssize_t n = ::read(0, buf, sizeof buf);
            if (n <= 0) break;
            for (ssize_t i = 0; i < n; ++i) {
                if (buf[i] != '\n') {
                    line.push_back(buf[i]);
                    continue;
                }
                while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
                if (line == "STOP") {
                    stop = true;   // never queued: a STOP is for the request that is running
                } else if (!line.empty()) {
                    if (line == "QUIT") stop = true;   // moex: the running request ends first (the server is leaving)
                    std::lock_guard<std::mutex> lk(q.mu);
                    q.lines.push_back(line);
                    q.cv.notify_one();
                }
                line.clear();
            }
        }
        std::lock_guard<std::mutex> lk(q.mu);
        q.eof = true;
        q.cv.notify_one();
    }).detach();

    g_req_stop.store(&stop);
    {
        struct sigaction sa {};
        sa.sa_handler = on_quit_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);
        // a server already gone must not end the engine before its save (a write to the pipe fails instead: below, a
        // request whose lines cannot be written stops)
        std::signal(SIGPIPE, SIG_IGN);
    }
    const EngineStats st0 = eng.stats();
    // moex (DSV41_PLAN §56): geni=manifest - a GENI file may be a manifest of the pictures' own rows files (read_manifest)
    std::printf("INFO context=%lld kv=fp16 expert_slots=%lld expert_cache_mib=%lld spec=%d pool_workers=%d %s geni=manifest "
                "engine=strata-x\n",
                (long long) so.max_context, (long long) st0.slots, (long long) (st0.slots * st0.slot_mib), so.spec, st0.pool_workers,
                so.info.c_str());
    // moex (DSV41_BACKLOG #52), STRATA_DSV41_CACHE_REFILL=1: the SSD cache of the last run (its list, saved with the expert
    // profile) read back in the background at the reads' lowest priority - the first requests after a restart find it in RAM
    static const bool cache_refill = std::getenv("STRATA_DSV41_CACHE_REFILL") && std::getenv("STRATA_DSV41_CACHE_REFILL")[0] == '1';
    const std::string cache_keys = cache_refill && !so.profile_save.empty() ? so.profile_save + ".ssd-cache" : std::string();
    if (!cache_keys.empty())
        if (const int64_t nr = eng.refill_cache(cache_keys); nr > 0)
            std::fprintf(stderr, "strata-x: SSD cache: %lld records of the last run being read back (%s)\n", (long long) nr,
                         cache_keys.c_str());
    std::printf("READY %lld stop\n", (long long) so.max_context);
    std::fflush(stdout);

    // #477: the learned profile, at the end and every profile_every_min minutes (between requests only)
    auto last_save = std::chrono::steady_clock::now();
    auto save_profile = [&](const char* why) {
        if (so.profile_save.empty()) return;
        std::string err;
        if (!eng.save_profile(so.profile_save, err)) std::fprintf(stderr, "strata-x: profile save: %s\n", err.c_str());
        else std::fprintf(stderr, "strata-x: expert profile saved to %s (%s)\n", so.profile_save.c_str(), why);
        if (!cache_keys.empty() && !eng.save_cache_keys(cache_keys, err))
            std::fprintf(stderr, "strata-x: SSD cache list: %s\n", err.c_str());
        last_save = std::chrono::steady_clock::now();
    };
    // moex (DSV41_BACKLOG #53): the clean end - the conversations first (minutes of reading each), then the profile, then
    // the signals' defaults back
    auto finish = [&](const char* why) {
        std::string err;
        eng.save_conversations(err);
        save_profile(why);
        g_req_stop.store(nullptr);
        std::signal(SIGINT, SIG_DFL);
        std::signal(SIGTERM, SIG_DFL);
    };
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(q.mu);
            const auto ready = [&] { return !q.lines.empty() || q.eof || g_quit_signal.load(); };
            for (;;) {
                if (ready()) break;
                // every 0.5 s a look at the signal flag (a handler cannot wake the condition variable)
                const auto now = std::chrono::steady_clock::now(), poll = now + std::chrono::milliseconds(500);
                const bool periodic = !so.profile_save.empty() && so.profile_every_min > 0;
                const auto due = periodic ? last_save + std::chrono::milliseconds((int64_t) (so.profile_every_min * 60000.0)) : poll;
                if (q.cv.wait_until(lk, std::min(due, poll), ready)) break;
                if (periodic && std::chrono::steady_clock::now() >= due) {
                    lk.unlock();
                    save_profile("periodic");
                    lk.lock();
                }
            }
            if (g_quit_signal.load()) {   // SIGINT / SIGTERM: as QUIT
                lk.unlock();
                finish("signal");
                break;
            }
            if (q.lines.empty()) {   // EOF
                lk.unlock();
                finish("end");
                break;
            }
            line = q.lines.front();
            q.lines.pop_front();
        }
        stop = false;   // a STOP that arrived between requests is stale
        if (line == "QUIT" || g_quit_signal.load()) {
            finish("quit");
            break;
        }
        // moex (DSV41_PLAN §54): VENC <n> <patches> <rows> ... - a request's new pictures encoded on the first GPU between
        // requests (vision.cpp; the server sends it in the request queue's turn) -> VOK <rows,...> <ms> <how>, or ERR
        if (line.rfind("VENC ", 0) == 0) {
            std::istringstream ls(line.substr(5));
            long long n = 0;
            std::vector<std::pair<std::string, std::string>> files;
            if (ls >> n && n >= 1 && n <= 4096) {
                std::string a, b;
                while ((long long) files.size() < n && ls >> a >> b) files.emplace_back(a, b);
            }
            std::string extra;
            if (n < 1 || (long long) files.size() != n || (ls >> extra)) {
                std::printf("ERR expected: VENC <n> <patches> <rows> ... (n pairs, no spaces in the paths)\n");
                continue;
            }
            std::vector<int64_t> rows;
            double ms = 0.0;
            std::string how, e;
            if (!eng.encode_pictures(files, rows, ms, how, e)) {
                std::printf("ERR %s\n", e.c_str());
                continue;
            }
            std::string list;
            for (size_t i = 0; i < rows.size(); ++i) list += (i ? "," : "") + std::to_string(rows[i]);
            std::printf("VOK %s %.1f %s\n", list.c_str(), ms, how.c_str());
            continue;
        }
        // GEN <max_new> [key=value ...] <ids>, or (moex vision) GENI <max_new> [key=value ...] <embeddings file> <ids>
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            continue;
        }
        char* endp = nullptr;
        const long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &endp, 10);
        SamplingParams sp;
        bool ckpt = true;
        if (endp != nullptr) {
            for (;;) {
                while (*endp == ' ') ++endp;
                const char* start = endp;
                while (*endp != '\0' && *endp != ' ') ++endp;
                if (endp == start) break;
                const std::string tok(start, (size_t) (endp - start));
                const size_t eq = tok.find('=');
                if (eq == std::string::npos) {
                    endp = const_cast<char*>(start);   // the ids start at the first token without '='
                    break;
                }
                const std::string key = tok.substr(0, eq), val = tok.substr(eq + 1);
                const float fv = std::strtof(val.c_str(), nullptr);
                if (key == "temperature") sp.temperature = fv;
                else if (key == "top_p") sp.top_p = fv;
                else if (key == "top_k") sp.top_k = std::atoi(val.c_str());
                else if (key == "min_p") sp.min_p = fv;
                else if (key == "penalty_last_n") sp.penalty_last_n = std::atoi(val.c_str());
                else if (key == "penalty_repeat") sp.penalty_repeat = fv;
                else if (key == "penalty_freq") sp.penalty_freq = fv;
                else if (key == "penalty_present") sp.penalty_present = fv;
                else if (key == "seed") sp.seed = std::strtoull(val.c_str(), nullptr, 10);
                else if (key == "spec_min_p") sp.spec_min_p = std::clamp(fv, 0.0f, 1.0f);
                else if (key == "spec_coupled") sp.spec_coupled = std::atoi(val.c_str()) != 0 ? 1 : 0;
                else if (key == "spec_window") sp.spec_window = std::clamp(std::atoi(val.c_str()), 0, 8);
                else if (key == "dspark_conf") sp.dspark_conf = std::clamp(fv, 0.0f, 1.0f);   // moex: per request (an A/B key)
                else if (key == "dspark_cum") sp.dspark_cum = std::clamp(fv, 0.0f, 1.0f);
                else if (key == "spec_policy") sp.spec_policy = std::atoi(val.c_str()) != 0 ? 1 : 0;
                else if (key == "read_short") sp.read_short = std::max(0LL, std::atoll(val.c_str()));
                else if (key == "read_cpu") sp.read_cpu = std::max(0LL, std::atoll(val.c_str()));
                else if (key == "read_share") sp.read_share = std::max(0LL, std::atoll(val.c_str()));
                else if (key == "read_ring") sp.read_ring = std::max(0LL, std::atoll(val.c_str()));
                else if (key == "ckpt") ckpt = std::atoi(val.c_str()) != 0;
                // unknown keys (cvec, pcie_frac, ...) are skipped
            }
        }
        std::string emb_path;   // GENI: the first token without '=' is the embeddings file (no spaces: Strata's server writes none)
        if (geni && endp != nullptr) {
            while (*endp == ' ') ++endp;
            const char* start = endp;
            while (*endp != '\0' && *endp != ' ') ++endp;
            emb_path.assign(start, (size_t) (endp - start));
        }
        std::vector<int32_t> ids;
        std::string pe;
        if (geni && emb_path.empty()) pe = "no embeddings file";
        if (max_new < 1 || endp == nullptr || !pe.empty() || !parse_ids(endp, ids, pe)) {
            std::printf("ERR bad request: %s\n", pe.empty() ? "max_new" : pe.c_str());
            continue;
        }
        if ((int64_t) ids.size() + max_new + 8 > so.max_context) {
            std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) ids.size(), max_new,
                        (long long) so.max_context);
            continue;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t >= eng.hparams().n_vocab;
        if (bad) {
            std::printf("ERR a token id is outside the vocabulary\n");
            continue;
        }
        std::vector<float> img_rows;    // the request's image rows (the spans point into them until the request ends)
        std::vector<MappedRows> img_maps;   // moex (§56): or into its pictures' own files, mapped until it ends
        std::vector<ImageSpan> spans;
        if (geni) {
            if (so.image_pad < 0) {
                std::printf("ERR this engine was started without --image-pad (no image encoder)\n");
                continue;
            }
            std::string ie;
            if (!read_images(emb_path, ids, so.image_pad, (int) eng.hparams().n_embd, img_rows, img_maps, spans, ie)) {
                std::printf("ERR %s\n", ie.c_str());
                continue;
            }
        }
        RequestHooks hooks;
        int64_t resumed = 0;
        hooks.resume = [&](int64_t r) {
            resumed = r;
            std::printf("RESUME %lld\n", (long long) r);
            std::fflush(stdout);
        };
        hooks.progress = [&](int64_t reached, int64_t total, double ms) {
            const double read = (double) std::max<int64_t>(1, reached - resumed);
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) reached, (long long) total, ms, read / std::max(1e-3, ms / 1000.0));
            std::fflush(stdout);
            return !stop.load() && !std::ferror(stdout);
        };
        hooks.work = [](int64_t done, int64_t all) {   // moex (DSV41_PLAN §18): a long read's progress inside its chunks
            std::printf("PL %lld %lld\n", (long long) done, (long long) all);
            std::fflush(stdout);
        };
        BoxStat bx0, bx1;   // the drive and the box at the request's start and when its prompt is read (the line below)
        EngineStats stp;    // ... and the engine's counters then: the reply's own swaps and fills stay out of the prompt's line
        bool have_stp = false;
        hooks.reused = [&](int64_t r) {
            if (!quiet) {
                bx1 = box_stat(disk);
                stp = eng.stats();
                have_stp = true;
            }
            std::printf("REUSED %lld\n", (long long) r);
            std::fflush(stdout);
        };
        hooks.token = [&](int32_t id) {
            std::printf("T %d\n", id);
            std::fflush(stdout);
            return !stop.load() && !std::ferror(stdout);
        };
        hooks.stop = [&] { return stop.load(); };
        RequestStats rs;
        std::string err;
        // moex (DSV41_BACKLOG #148, DSV41_PLAN §75): the drive guard's episodes begun and its seconds below full depth during
        // this request - the DONE line's last two fields, which the server keeps per request with diagnostics off too
        const dsv41::DriveGuard* dgd = dsv41::DriveGuard::get();
        const int64_t dg_ep0 = dgd ? dgd->episodes() : 0;
        const double dg_fl0 = dgd ? dgd->floor_s() : 0.0;
        const EngineStats st0 = quiet ? EngineStats{} : eng.stats();   // the request's prompt path, from the counters'
        if (!quiet) bx0 = box_stat(disk);                                // difference (logged below)
        if (!eng.run_request(ids, (int) max_new, sp, so.eos, hooks, ckpt, rs, err, geni ? &spans : nullptr)) {
            // a failure past validation may have left the state half written: say so and end (the server starts it again)
            std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
            std::fprintf(stderr, "strata-x serve: the request failed: %s\n", err.c_str());
            eng.forget();
            g_req_stop.store(nullptr);   // (nothing is saved after a failure: its state may be half written)
            std::signal(SIGINT, SIG_DFL);
            std::signal(SIGTERM, SIG_DFL);
            return 1;
        }
        std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld %lld %.1f\n", (long long) rs.generated,
                    (long long) rs.prompt, rs.prompt_ms, rs.decode_ms, rs.finish.c_str(), (long long) rs.drafts_accepted,
                    (long long) rs.drafts_offered, (long long) rs.reused, (long long) rs.hits, (long long) rs.lookups, 0LL, 0LL, 0.0,
                    (long long) rs.prompt_read, 0LL, (long long) (dgd ? dgd->episodes() - dg_ep0 : 0),
                    dgd ? dgd->floor_s() - dg_fl0 : 0.0);
        std::fflush(stdout);
        if (quiet) continue;   // the request's lines below: none
        std::fprintf(stderr, "strata-x serve: %lld prompt tokens (%lld reused, %lld read) in %.1f s, %lld generated in %.1f s (%.1f T/s), %s; "
                             "%lld windows, drafts %lld/%lld, lookup windows %lld (%lld kept)\n",
                     (long long) rs.prompt, (long long) rs.reused, (long long) rs.prompt_read, rs.prompt_ms / 1000.0,
                     (long long) rs.generated, rs.decode_ms / 1000.0, rs.generated / std::max(1e-3, rs.decode_ms / 1000.0), rs.finish.c_str(),
                     (long long) rs.windows, (long long) rs.drafts_accepted, (long long) rs.drafts_offered, (long long) rs.lookup_windows,
                     (long long) rs.lookup_accepted);
        if (const EngineStats st9 = eng.stats(); st9.hy4_eg_hits + st9.hy4_eg_misses > st0.hy4_eg_hits + st0.hy4_eg_misses) {
            // moex (2026-11-19): Hy4-flash's Engram rows this request looked up, and the share its row cache held
            const double h = (double) (st9.hy4_eg_hits - st0.hy4_eg_hits), q = h + (double) (st9.hy4_eg_misses - st0.hy4_eg_misses);
            std::fprintf(stderr, "strata-x serve: the Engram's rows: %.0f looked up, %.1f%% from the row cache\n", q, 100.0 * h / q);
        }
        if (const EngineStats st1 = have_stp ? stp : eng.stats(); st1.tokens_prefill > st0.tokens_prefill && st1.rec_reads + st1.ssd_reads > 0) {
            // DeepSeek-V4.1's SSD tier: where the request's prompt path spent its time (its chunk, its reads, its waits), and the
            // drive and the box meanwhile (from the request's start to the end of its prompt)
            char box[320] = "";
            // moex (DSV41_BACKLOG #59): the engram tables' n-gram reader, when on (STRATA_DSV41_EG_READER)
            char egr[160] = "";
            if (const long long q = st1.eg_reader_rows - st0.eg_reader_rows; q > 0)
                std::snprintf(egr, sizeof egr, " (the n-gram reader: %lld rows, %.0f%% from its cache, %lld reads)", q,
                              100.0 * (double) (st1.eg_reader_hits - st0.eg_reader_hits) / (double) q,
                              (long long) (st1.eg_reader_reads - st0.eg_reader_reads));
            else if (const long long qa = st1.eg_dedup_asked - st0.eg_dedup_asked; qa > 0)   // moex (#103)
                std::snprintf(egr, sizeof egr, " (%lld of its %lld rows distinct, read once; the scales from the page cache)",
                              (long long) (st1.eg_dedup_read - st0.eg_dedup_read), qa);
            bool drive_slow = false;   // moex (DSV41_BACKLOG #59): a read the drive's slow mode took, by its numbers below
            if (bx0.disk && bx1.disk && bx1.t > bx0.t) {
                const double s = std::chrono::duration<double>(bx1.t - bx0.t).count(), gb = (bx1.rd_sec - bx0.rd_sec) * 512.0 / 1e9;
                const unsigned long long ios = bx1.rd_ios - bx0.rd_ios;
                // the kernel prints the time fields as 32-bit milliseconds (summed over the requests in flight, so a busy disk's
                // read time passes 2^32 ms within weeks): a difference modulo 2^32
                auto ms32 = [](unsigned long long a, unsigned long long b) { return (double) ((b - a) & 0xffffffffull); };
                std::snprintf(box, sizeof box, "; the drive meanwhile: %.1f GB read in %.1f s (%.2f GB/s, %.2f ms a request, busy %.0f%%), "
                                               "%.0f MB written; stalls: CPU %.1f%%, memory %.1f%%",
                              gb, s, gb / s, ios ? ms32(bx0.rd_ms, bx1.rd_ms) / (double) ios : 0.0,
                              100.0 * ms32(bx0.io_ms, bx1.io_ms) / (s * 1e3), (bx1.wr_sec - bx0.wr_sec) * 512.0 / 1e6,
                              100.0 * (double) (bx1.cpu_us - bx0.cpu_us) / (s * 1e6), 100.0 * (double) (bx1.mem_us - bx0.mem_us) / (s * 1e6));
                drive_slow = s >= 10.0 && gb / s < 1.2 && 100.0 * ms32(bx0.io_ms, bx1.io_ms) / (s * 1e3) >= 85.0;
            }
            std::fprintf(stderr, "strata-x serve: prompt path: %lld tokens, chunk %lld (ring %lld, %lld slots lent) in %.1f s; %lld "
                                 "experts streamed: %lld read from the SSD, %lld from its cache; refill %lld from the cache / %lld read; "
                                 "waits: each layer's routing %.1f s, the engram rows %.1f s%s; read-ahead: %lld read on the spot, "
                                 "%lld unused; no read in flight: lending %.1f s, for the routing %.1f s (GPU 0 %.1f, GPU 1 %.1f), the "
                                 "ring %.1f s, bounce slots %.1f s, refilling %.1f s%s\n",
                         (long long) (st1.tokens_prefill - st0.tokens_prefill), (long long) st1.prefill_chunk, (long long) st1.prefill_ring,
                         (long long) st1.lent_slots, (st1.ms_prefill - st0.ms_prefill) / 1000.0, (long long) (st1.streamed - st0.streamed),
                         (long long) (st1.rec_reads - st0.rec_reads), (long long) (st1.rec_cached - st0.rec_cached),
                         (long long) (st1.fill_cached - st0.fill_cached), (long long) (st1.fill_reads - st0.fill_reads),
                         (st1.ms_prefill_wait - st0.ms_prefill_wait) / 1000.0, (st1.ms_eg_rows - st0.ms_eg_rows) / 1000.0, egr,
                         (long long) (st1.prompt_sync - st0.prompt_sync), (long long) (st1.prompt_orphan - st0.prompt_orphan),
                         (st1.ms_idle_lend - st0.ms_idle_lend) / 1000.0, (st1.ms_idle_gate - st0.ms_idle_gate) / 1000.0,
                         (st1.ms_idle_gate_dev[0] - st0.ms_idle_gate_dev[0]) / 1000.0, (st1.ms_idle_gate_dev[1] - st0.ms_idle_gate_dev[1]) / 1000.0,
                         (st1.ms_idle_ring - st0.ms_idle_ring) / 1000.0, (st1.ms_idle_bounce - st0.ms_idle_bounce) / 1000.0,
                         (st1.ms_idle_refill - st0.ms_idle_refill) / 1000.0, box);
            if (st1.prompt_adm_free + st1.prompt_adm_hot > st0.prompt_adm_free + st0.prompt_adm_hot)   // moex (DSV41_PLAN §41)
                std::fprintf(stderr, "strata-x serve: prompt path: the SSD cache kept %lld of its reads past its share: %lld in free "
                                     "slots (STRATA_DSV41_CACHE_FREE), %lld routed by 1 / %s of the rows or more (STRATA_DSV41_PROMPT_HOT)\n",
                             (long long) (st1.prompt_adm_free + st1.prompt_adm_hot - st0.prompt_adm_free - st0.prompt_adm_hot),
                             (long long) (st1.prompt_adm_free - st0.prompt_adm_free), (long long) (st1.prompt_adm_hot - st0.prompt_adm_hot),
                             std::getenv("STRATA_DSV41_PROMPT_HOT") ? std::getenv("STRATA_DSV41_PROMPT_HOT") : "-");
            // moex (DSV41_BACKLOG #59): blocks the engine reads this often are rewritten by the drive in its lulls, and while it
            // does that (~64 s) it serves ~490 requests a second - three such reads within an hour, and the log counts them
            static std::deque<std::chrono::steady_clock::time_point> slow_reads;
            static std::chrono::steady_clock::time_point hinted{};
            if (drive_slow) {
                const auto now = std::chrono::steady_clock::now();
                slow_reads.push_back(now);
                while (now - slow_reads.front() > std::chrono::hours(1)) slow_reads.pop_front();
                if (slow_reads.size() >= 3 && (hinted == std::chrono::steady_clock::time_point{} || now - hinted > std::chrono::hours(1))) {
                    hinted = now;
                    std::fprintf(stderr, "strata-x serve: %zu prompt reads in the last hour ran under 1.2 GB/s with the drive busy - "
                                         "its slow mode (DSV41_PLAN §14, §18: the drive guard's lines give each episode)\n",
                                 slow_reads.size());
                }
            }
        }
        if (have_stp && rs.generated >= 64) {
            // moex: where the reply's time went, from the prompt's end (the counters above) to its last token - a slow reply
            // says whether it waited for the drive, the CPU pool or the GPUs
            const EngineStats st2 = eng.stats();
            const double g = (double) rs.generated, lk = (double) ((st2.hits - stp.hits) + (st2.misses - stp.misses));
            std::fprintf(stderr, "strata-x serve: decode: %lld tokens in %.1f s; a token: %.1f ms waiting for the routing, %.1f ms in the "
                                 "CPU pool (%.1f ms of it for the drive), %.2f ms reading its engram rows; %.2f experts read from the SSD "
                                 "and %.2f from its cache, %.1f from RAM; %.0f%% of the routings in VRAM; %lld swaps, %lld fills read from "
                                 "the SSD\n",
                         (long long) rs.generated, rs.decode_ms / 1000.0, (st2.ms_wait - stp.ms_wait) / g, (st2.ms_pool - stp.ms_pool) / g,
                         (st2.ms_ssd_wait - stp.ms_ssd_wait) / g, (st2.ms_eg_rows - stp.ms_eg_rows + st2.ms_eg - stp.ms_eg) / g,
                         (double) (st2.ssd_reads - stp.ssd_reads) / g,
                         (double) (st2.ssd_hits - stp.ssd_hits) / g, (double) (st2.cpu_ram - stp.cpu_ram) / g,
                         lk > 0 ? 100.0 * (double) (st2.hits - stp.hits) / lk : 0.0, (long long) (st2.swaps - stp.swaps),
                         (long long) (st2.fill_reads - stp.fill_reads));
            // moex (DSV41_PLAN §34): the rest of a token - the head (the last layer's tail and the logits), the tier's upkeep,
            // and what is left (the window's inputs, its engram rows, the sampling, the reply's lines)
            std::fprintf(stderr, "strata-x serve: decode: the VRAM tier's upkeep a token: landed copies %.2f ms, first-use fills %.2f, the "
                                 "slot check %.2f, the swap rounds %.2f, the route bias's table %.2f; of it %.2f ms hashing %.2f first-time "
                                 "records\n",
                         (st2.ms_up_final - stp.ms_up_final) / g, (st2.ms_up_fill - stp.ms_up_fill) / g, (st2.ms_up_verify - stp.ms_up_verify) / g,
                         (st2.ms_up_round - stp.ms_up_round) / g, (st2.ms_up_pen - stp.ms_up_pen) / g, (st2.ms_up_khash - stp.ms_up_khash) / g,
                         (double) (st2.up_khash - stp.up_khash) / g);
            std::fprintf(stderr, "strata-x serve: decode: a token's other %.1f ms: the head %.1f, the VRAM tier's upkeep %.1f, the rest %.1f\n",
                         rs.decode_ms / g - (st2.ms_wait - stp.ms_wait) / g - (st2.ms_pool - stp.ms_pool) / g, (st2.ms_head - stp.ms_head) / g,
                         (st2.ms_upkeep - stp.ms_upkeep) / g,
                         rs.decode_ms / g - (st2.ms_wait - stp.ms_wait) / g - (st2.ms_pool - stp.ms_pool) / g - (st2.ms_head - stp.ms_head) / g -
                             (st2.ms_upkeep - stp.ms_upkeep) / g);
            if (st2.pool_bytes > stp.pool_bytes) {   // moex (DSV41_PLAN §29): where the CPU pool's time goes (main.cpp's line)
                const double gb = (double) (st2.pool_bytes - stp.pool_bytes) / 1e9, dr = st2.pool_drain_ms - stp.pool_drain_ms;
                std::fprintf(stderr, "strata-x serve: decode CPU pool: a token %.3f GB over %.2f ms of its drains (%.1f GB/s): gate/up "
                                     "%.2f ms, quantize %.2f, down %.2f, barriers %.2f + %.2f (park, repark)\n", gb / g, dr / g,
                             dr > 0 ? gb / (dr / 1000.0) : 0.0, (st2.pool_gu_ms - stp.pool_gu_ms) / g, (st2.pool_q_ms - stp.pool_q_ms) / g,
                             (st2.pool_down_ms - stp.pool_down_ms) / g, (st2.pool_park_ms - stp.pool_park_ms) / g,
                             (st2.pool_repark_ms - stp.pool_repark_ms) / g);
            }
            if (st2.ahead_issued > stp.ahead_issued)   // moex (DSV41_PLAN §23): the opt-in read-ahead's (in the SSD reads above)
                std::fprintf(stderr, "strata-x serve: decode read-ahead: a token %.2f records read ahead, %.2f of them used, %.2f "
                                     "evicted unused\n", (double) (st2.ahead_issued - stp.ahead_issued) / g,
                             (double) (st2.ahead_used - stp.ahead_used) / g, (double) (st2.ahead_wasted - stp.ahead_wasted) / g);
        }
        if (rs.offered_at[0] > 0) {   // per draft depth: accepted / offered
            std::fprintf(stderr, "strata-x serve: drafts by depth:");
            for (int j = 0; j < 8 && rs.offered_at[j] > 0; ++j)
                std::fprintf(stderr, " %d: %lld/%lld (%.0f%%)", j + 1, (long long) rs.accepted_at[j], (long long) rs.offered_at[j],
                             100.0 * rs.accepted_at[j] / rs.offered_at[j]);
            std::fprintf(stderr, "\n");
        }
        // moex (PORT_AUDIT_ALL §6): the RAM copy - its state now and this request's events (its read included), every request
        // of an engine with a RAM copy; then, with STRATA_X_TIER_CHECK=1, its invariants (the engine idle)
        if (const Engine::RamCopyState rc = eng.ram_copy_state(); rc.blobs > 0) {
            const EngineStats sr = eng.stats();
            std::fprintf(stderr, "strata-x serve: the RAM copy: %lld exchange blobs free, %lld experts only in the file, %lld doubled "
                                 "in VRAM; this request: blobs taken %lld by swaps and %lld by growths, %lld given back, %lld released "
                                 "from doubles, %lld swap attempts waited for an owed blob; for want of a blob %lld victims and %lld growths' "
                                 "experts to the file; %lld victims to the file in place of an expert read from it\n",
                         (long long) rc.free, (long long) rc.file_only, (long long) rc.doubled,
                         (long long) (sr.ram_take_swap - st0.ram_take_swap), (long long) (sr.ram_take_grow - st0.ram_take_grow),
                         (long long) (sr.ram_give - st0.ram_give), (long long) (sr.ram_recycled - st0.ram_recycled),
                         (long long) (sr.ram_wait - st0.ram_wait), (long long) (sr.ram_fallback - st0.ram_fallback),
                         (long long) (sr.ram_file_grow - st0.ram_file_grow), (long long) (sr.ram_file_swap - st0.ram_file_swap));
        }
        static const bool tier_check = std::getenv("STRATA_X_TIER_CHECK") && std::getenv("STRATA_X_TIER_CHECK")[0] == '1';
        if (tier_check) eng.tier_check();
    }
    return 0;
}

}  // namespace strata::x
