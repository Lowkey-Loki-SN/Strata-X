// moex/tools/dsv41/expert_io.hpp - priority O_DIRECT reads for experts streamed from the SSD (header-only).
//
// One pool of reader threads serves four classes, highest first: demand misses (a layer has routed and waits),
// layer-ahead predictions, the prefill sweep, background promotions. A request is split into pieces of at most
// `piece` bytes (4 MiB: larger calls queue hundreds of NVMe requests and stall the desktop, src/x/tiers.cpp:495), and
// the classes below demand may hold at most `spec_cap` reader threads at a time, so a demand piece waits behind
// at most that many speculative pieces. A queued speculative request can be cancelled, or promoted to demand when
// the router proves it was needed. The reads go straight into the caller's (pinned, 4 KiB aligned) slot memory.
// moex (DSV41_PLAN §18): with the drive guard on (drive_guard.hpp) at most its readers() at a time, each read call at
// most its call_bytes(); without it, as before.
#pragma once

#include "drive_guard.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace dsv41 {

enum class IoClass : int { Demand = 0, Ahead = 1, Sweep = 2, Background = 3 };
constexpr int kIoClasses = 4;

class ExpertIo {
public:
    /// moex (the GGUF set): one extent of a multi-extent request - `length` bytes of the file at `offset` (both multiples
    /// of 512, the logical block) to `dst` (aligned as the device's dma_alignment asks: the caller checks it)
    struct Seg {
        uint64_t offset = 0, length = 0;
        uint8_t* dst = nullptr;
        int fd = -1;   ///< its file (-1: the request's) - a layer's three tensors may sit in two files of a set
        /// the bytes of it that must be read (0: all): past them the last sector may run beyond the file's end, where an
        /// O_DIRECT read comes back short (a GGUF ends where its last tensor does, mid-sector)
        uint64_t need = 0;
    };
    static constexpr int kMaxSegs = 4;
    /// The head of a part that starts inside a sector: segment `seg` (one 512-byte sector) is read into `buf` (owned by
    /// the request), then `len` bytes from buf + `skip` are copied to `dst` - so no byte is written before the part
    struct Fix {
        int seg = -1;
        uint8_t* dst = nullptr;
        uint32_t skip = 0, len = 0;
    };
    struct Request {
        int fd = -1;
        uint64_t offset = 0;
        uint64_t length = 0;
        uint8_t* dst = nullptr;
        Seg seg[kMaxSegs];            ///< moex: a multi-extent request's extents (nseg 0: the one extent above)
        int nseg = 0;
        Fix fix;
        uint8_t* fix_buf = nullptr;   ///< 512 bytes, 4096-aligned (std::aligned_alloc; freed with the request)
        ~Request() { std::free(fix_buf); }
        std::atomic<int> cls{0};
        std::atomic<uint32_t> pending{0};   // pieces not yet read
        std::atomic<bool> cancelled{false};
        std::atomic<int> error{0};
        std::atomic<bool> done{false};
    };
    using Handle = std::shared_ptr<Request>;

    ExpertIo(int threads, int spec_cap, uint64_t piece = 4ull << 20)
        : piece_(piece), spec_cap_(spec_cap) {
        if (threads < 1 || spec_cap < 0 || spec_cap > threads || piece % 4096) throw std::invalid_argument("ExpertIo");
        for (int i = 0; i < threads; ++i) workers_.emplace_back([this] { run(); });
    }
    ~ExpertIo() {
        { std::lock_guard<std::mutex> g(mu_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    // offset, length and dst must be 4 KiB aligned (O_DIRECT).
    Handle submit(int fd, uint64_t offset, uint64_t length, uint8_t* dst, IoClass cls) {
        if ((offset | length | reinterpret_cast<uintptr_t>(dst)) & 4095) throw std::invalid_argument("unaligned read");
        auto r = std::make_shared<Request>();
        r->fd = fd; r->offset = offset; r->length = length; r->dst = dst; r->cls = int(cls);
        const uint32_t n = uint32_t((length + piece_ - 1) / piece_);
        r->pending = n;
        if (n == 0) { r->done = true; return r; }
        {
            std::lock_guard<std::mutex> g(mu_);
            for (uint32_t i = 0; i < n; ++i) q_[int(cls)].push_back({r, uint64_t(i) * piece_});
        }
        cv_.notify_all();
        return r;
    }

    // moex (the GGUF set): several extents of one file as one request (a GGUF expert's gate, up and down slices, each into
    // its place in the record's slot), done when all are read; `fix` (optional): its segment is a 512-byte sector read into
    // the request's own buffer, whose bytes [skip, skip + len) then go to fix.dst.  Offsets and lengths: multiples of 512;
    // the destinations' alignment: the caller's check (the device's dma_alignment)
    Handle submit(int fd, const Seg* segs, int nseg, const Fix* fix, IoClass cls) {
        if (nseg < 1 || nseg > kMaxSegs) throw std::invalid_argument("ExpertIo: segments");
        auto r = std::make_shared<Request>();
        r->fd = fd; r->cls = int(cls); r->nseg = nseg;
        uint32_t n = 0;
        for (int s = 0; s < nseg; ++s) {
            r->seg[s] = segs[s];
            if ((segs[s].offset | segs[s].length) & 511) throw std::invalid_argument("unaligned read");
            n += uint32_t((segs[s].length + piece_ - 1) / piece_);
        }
        if (fix && fix->seg >= 0) {
            if (fix->seg >= nseg || segs[fix->seg].length != 512 || fix->skip + fix->len > 512) throw std::invalid_argument("ExpertIo: fix");
            r->fix = *fix;
            r->fix_buf = static_cast<uint8_t*>(std::aligned_alloc(4096, 4096));
            if (!r->fix_buf) throw std::bad_alloc();
            r->seg[fix->seg].dst = r->fix_buf;
        }
        r->pending = n;
        if (n == 0) { r->done = true; return r; }
        {
            std::lock_guard<std::mutex> g(mu_);
            for (int s = 0; s < nseg; ++s)
                for (uint64_t at = 0; at < segs[s].length; at += piece_) q_[int(cls)].push_back({r, (uint64_t(s) << kSegShift) | at});
        }
        cv_.notify_all();
        return r;
    }

    // The router needs it now: move its queued pieces to the demand queue.
    void promote(const Handle& r) {
        std::lock_guard<std::mutex> g(mu_);
        const int c = r->cls.exchange(int(IoClass::Demand));
        if (c == int(IoClass::Demand)) return;
        auto& from = q_[c];
        for (auto it = from.begin(); it != from.end();) {
            if (it->req == r) { q_[0].push_back(*it); it = from.erase(it); } else { ++it; }
        }
        cv_.notify_all();
    }

    // Not needed after all: pieces not started are dropped; the request completes with `cancelled` set.
    void cancel(const Handle& r) { r->cancelled = true; }

    void wait(const Handle& r) {
        if (r->done.load(std::memory_order_acquire)) return;
        std::unique_lock<std::mutex> g(done_mu_);
        done_cv_.wait(g, [&] { return r->done.load(std::memory_order_acquire); });
    }

    static int open_direct(const std::string& path) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("open " + path + ": " + std::strerror(errno));
        return fd;
    }

private:
    struct Piece { Handle req; uint64_t at; };

    bool pick(Piece& p) {  // under mu_
        if (g_ && running_ >= g_->readers()) return false;   // the drive guard's limit (its floor: one reader)
        if (!q_[0].empty()) { p = q_[0].front(); q_[0].pop_front(); running_ += g_ != nullptr; return true; }
        if (spec_running_ >= spec_cap_) return false;
        for (int c = 1; c < kIoClasses; ++c) {
            if (!q_[c].empty()) {
                p = q_[c].front(); q_[c].pop_front(); ++spec_running_; p.at |= kSpecBit; running_ += g_ != nullptr; return true;
            }
        }
        return false;
    }

    void run() {
        for (;;) {
            Piece p;
            {
                std::unique_lock<std::mutex> g(mu_);
                if (g_) {   // the guard's level can rise with nothing queued: look again every 50 ms
                    while (!cv_.wait_for(g, std::chrono::milliseconds(50), [&] { return stop_ || pick(p); })) {}
                } else {
                    cv_.wait(g, [&] { return stop_ || pick(p); });
                }
                if (stop_ && !p.req) return;
            }
            const bool spec = p.at & kSpecBit;
            const int si = r_seg(p);   // moex: a multi-extent request's segment (-1: the single extent)
            const uint64_t at = p.at & ~kSpecBit & kOffMask;
            Request& r = *p.req;
            if (!r.cancelled.load(std::memory_order_relaxed)) {
                const uint64_t base = si < 0 ? r.offset : r.seg[si].offset;
                uint8_t* const dst = si < 0 ? r.dst : r.seg[si].dst;
                const int fd = si >= 0 && r.seg[si].fd >= 0 ? r.seg[si].fd : r.fd;
                const uint64_t len = std::min<uint64_t>(piece_, (si < 0 ? r.length : r.seg[si].length) - at);
                const uint64_t call = g_ ? std::min<uint64_t>(len, g_->call_bytes()) : len;   // the guard's floor: 128 KB
                uint64_t got = 0;
                while (got < len) {
                    const ssize_t n = ::pread(fd, dst + at + got, std::min<uint64_t>(call, len - got), off_t(base + at + got));
                    if (n <= 0) {
                        if (n < 0 && errno == EINTR) continue;
                        if (n == 0 && si >= 0 && r.seg[si].need && at + got >= r.seg[si].need) break;   // moex: the file's end
                        r.error = n < 0 ? errno : EIO;
                        break;
                    }
                    got += uint64_t(n);
                }
                if (si >= 0 && si == r.fix.seg && !r.error.load(std::memory_order_relaxed))
                    std::memcpy(r.fix.dst, r.fix_buf + r.fix.skip, r.fix.len);
            }
            if (spec || g_) {
                std::lock_guard<std::mutex> g(mu_);
                if (spec) --spec_running_;
                if (g_) --running_;
            }
            cv_.notify_one();
            if (r.pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                { std::lock_guard<std::mutex> g(done_mu_); r.done.store(true, std::memory_order_release); }
                done_cv_.notify_all();
            }
        }
    }

    static constexpr uint64_t kSpecBit = 1ull << 63;
    // moex: a multi-extent request's piece carries its segment in bits 56..58 (a piece offset stays far below 2^56)
    static constexpr int kSegShift = 56;
    static constexpr uint64_t kOffMask = (1ull << kSegShift) - 1;
    static int r_seg(const Piece& p) { return p.req->nseg ? int(((p.at & ~kSpecBit) >> kSegShift) & 7) : -1; }
    const uint64_t piece_;
    const int spec_cap_;
    DriveGuard* const g_ = DriveGuard::get();   ///< moex (§18): the drive guard, or null
    int spec_running_ = 0;
    int running_ = 0;                            ///< pieces being read (counted with the guard only)
    bool stop_ = false;
    std::mutex mu_, done_mu_;
    std::condition_variable cv_, done_cv_;
    std::deque<Piece> q_[kIoClasses];
    std::vector<std::thread> workers_;
};

}  // namespace dsv41
