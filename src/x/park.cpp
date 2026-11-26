// src/x/park.cpp - moex (DSV41_PLAN §57): PARKED CONVERSATIONS (DeepSeek-V4 / V4.1, --conversation-cache-mib).
//
// The engine holds one conversation's state.  A request that leaves the live conversation - its prompt parts from the live
// tokens before their last prompt's final turn: another session, a subagent, a client's side request - would overwrite its
// positional rows and delete its checkpoints, and its next request would read it again from the root.  Here that state goes
// to RAM first: the positional rows (Stage::positional_regions), the running state at the live end (a checkpoint of it) and
// the conversation's own checkpoints (the ones the leaving request does not share - today's rule deletes them).  A request
// that continues a parked conversation further than the live state and the checkpoints can puts it back: the same bytes in
// the same buffers, the running state and the engram's ids through Impl::restore, its tokens, pictures and checkpoints - and
// the request goes on as if the conversation had never left.  Rows past the restored length hold another conversation's
// bytes, and every one is written before a read reaches it (a visible row is a completed block): the invariant a
// checkpoint's restore relies on.  The rules are park_policy.hpp's; the budget's order: the least recently used records,
// then the record's oldest checkpoints; under it nothing is parked and the engine does what it did before.
#include "engine_impl.hpp"
#include "park_policy.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <new>
#include <string>

namespace strata::x {

namespace {
bool prefix_of(const std::vector<int32_t>& a, const std::vector<int32_t>& b, size_t n) {   // a[0, n) == b[0, n)
    return a.size() >= n && b.size() >= n && std::equal(a.begin(), a.begin() + (std::ptrdiff_t) n, b.begin());
}

std::vector<ImageKey> below(const std::vector<ImageKey>& v, int64_t len) {   // the pictures that start inside [0, len)
    std::vector<ImageKey> out;
    for (const ImageKey& k : v)
        if (k.start < len) out.push_back(k);
    return out;
}

/// The common prefix of tokens a and b, ended at the first picture inside it that the other lacks (its rows differ).
int64_t common_prefix(const std::vector<int32_t>& a, const std::vector<ImageKey>& ai, const std::vector<int32_t>& b,
                      const std::vector<ImageKey>& bi) {
    const size_t lim = std::min(a.size(), b.size());
    size_t c = 0;
    while (c < lim && a[c] == b[c]) ++c;
    for (const auto* side : {&ai, &bi}) {
        const auto& other = side == &ai ? bi : ai;
        for (const ImageKey& k : *side) {   // (sorted by start)
            if (k.start >= (int64_t) c) break;
            if (std::find(other.begin(), other.end(), k) == other.end()) {
                c = (size_t) k.start;
                break;
            }
        }
    }
    return (int64_t) c;
}

uint64_t mem_available() {   // /proc/meminfo's MemAvailable in bytes; unknown: no floor
    std::ifstream f("/proc/meminfo");
    std::string key, unit;
    uint64_t kb = 0;
    while (f >> key >> kb >> unit)
        if (key == "MemAvailable:") return kb * 1024;
    return UINT64_MAX;
}

size_t ck_bytes(const Checkpoint& c) {
    size_t b = 0;
    for (size_t x : c.bytes) b += x;
    return b;
}

constexpr double kMiB = 1048576.0;
}  // namespace

size_t Parked::bytes() const {
    size_t b = 0;
    for (size_t x : row_bytes) b += x;
    if (snap) b += ck_bytes(*snap);
    for (const auto& c : checks) b += ck_bytes(*c);
    return b;
}

uint64_t Engine::Impl::parked_bytes() const {
    uint64_t b = 0;
    for (const auto& p : parked) b += p->bytes();
    return b;
}

void Engine::Impl::park_note_turn(const std::vector<int32_t>& prompt) {
    int64_t pos = (int64_t) prompt.size();   // no turn token: the prompt's end
    for (int64_t i = (int64_t) prompt.size() - 1; i >= 1; --i)
        if (std::find(turn_tokens.begin(), turn_tokens.end(), prompt[(size_t) i]) != turn_tokens.end()) {
            pos = i;
            break;
        }
    live_turn_pos = pos;
}

bool Engine::Impl::park_live(const std::vector<int32_t>& prompt, const std::vector<ImageKey>& keys, const Parked* keep) {
    const auto t0 = clk::now();
    const int64_t len = (int64_t) live.size();
    // a record that extends the live conversation holds it already; the records it extends are superseded
    for (const auto& p : parked)
        if ((int64_t) p->ids.size() >= len && prefix_of(live, p->ids, (size_t) len) && below(p->imgs, len) == live_imgs) {
            std::fprintf(stderr, "strata-x serve: conversation cache: skipped %lld tokens (a parked conversation extends them); "
                                 "parked=%zu bytes=%llu\n",
                         (long long) len, parked.size(), (unsigned long long) parked_bytes());
            return false;
        }
    for (size_t i = 0; i < parked.size();) {
        const Parked& p = *parked[i];
        if (&p != keep && (int64_t) p.ids.size() <= len && prefix_of(p.ids, live, p.ids.size()) &&
            below(live_imgs, (int64_t) p.ids.size()) == p.imgs) {
            parked.erase(parked.begin() + (std::ptrdiff_t) i);
            ++park_evictions;
        } else {
            ++i;
        }
    }
    // what the record takes: new copies of the rows and of the running state at the live end, and the checkpoints it keeps
    std::vector<std::vector<std::pair<void*, size_t>>> reg(stages.size());
    std::vector<size_t> rb(stages.size(), 0);
    uint64_t need = 0;
    for (size_t si = 0; si < stages.size(); ++si) {
        stages[si]->positional_regions(len, reg[si]);
        for (const auto& r : reg[si]) rb[si] += r.second;
        need += rb[si] + stages[si]->run_state_bytes();
    }
    std::vector<size_t> own;   // the conversation's own checkpoints: the ones the new prompt does not share, newest first
    for (size_t i = 0; i < checks.size(); ++i) {
        const Checkpoint& c = *checks[i];
        const bool shared =
            c.ids.size() <= prompt.size() && prefix_of(c.ids, prompt, c.ids.size()) && below(keys, (int64_t) c.ids.size()) == c.imgs;
        if (!shared) own.push_back(i);
    }
    std::sort(own.begin(), own.end(), [&](size_t a, size_t b) { return checks[a]->used > checks[b]->used; });
    uint64_t own_bytes = 0;
    for (size_t i : own) own_bytes += ck_bytes(*checks[i]);
    const uint64_t budget = (uint64_t) opt.conv_cache_mib << 20, floor = (uint64_t) opt.conv_cache_min_free_mib << 20;
    const size_t slots = (size_t) std::max(1, opt.conv_cache_slots);
    // room: the least recently used records out, never `keep`
    while (!parked.empty() && (parked.size() >= slots || parked_bytes() + need + own_bytes > budget)) {
        std::vector<uint64_t> st;
        int ki = -1;
        for (size_t i = 0; i < parked.size(); ++i) {
            st.push_back(parked[i]->used);
            if (parked[i].get() == keep) ki = (int) i;
        }
        const int v = park::lru_victim(st, ki);
        if (v < 0) break;
        parked.erase(parked.begin() + v);
        ++park_evictions;
    }
    // still over the budget: its oldest checkpoints stay behind (deleted, as before), the newest goes with it
    while (own.size() > 1 && parked_bytes() + need + own_bytes > budget) {
        own_bytes -= ck_bytes(*checks[own.back()]);
        own.pop_back();
    }
    const uint64_t avail = mem_available();
    if (parked.size() >= slots || !park::fits(parked_bytes() + own_bytes, need, budget, avail, floor)) {
        std::fprintf(stderr, "strata-x serve: conversation cache: skipped %lld tokens (%.0f MiB: over the %lld MiB budget, the slots "
                             "or the %lld MiB free-RAM floor - %.0f MiB available); parked=%zu bytes=%llu\n",
                     (long long) len, (need + own_bytes) / kMiB, (long long) opt.conv_cache_mib,
                     (long long) opt.conv_cache_min_free_mib, avail == UINT64_MAX ? -1.0 : avail / kMiB, parked.size(),
                     (unsigned long long) parked_bytes());
        return false;
    }
    auto P = std::make_unique<Parked>();
    P->rows.resize(stages.size());
    P->row_bytes = rb;
    for (size_t si = 0; si < stages.size(); ++si) {
        Stage& g = *stages[si];
        set_dev(g.dev);
        P->rows[si].reset(new (std::nothrow) uint8_t[std::max<size_t>(rb[si], 1)]);
        bool ok = P->rows[si] != nullptr && cudaStreamSynchronize(g.s) == cudaSuccess;
        size_t off = 0;
        for (const auto& r : reg[si]) {
            if (!ok) break;
            ok = cudaMemcpyAsync(P->rows[si].get() + off, r.first, r.second, cudaMemcpyDeviceToHost, g.s) == cudaSuccess;
            off += r.second;
        }
        if (!ok || cudaStreamSynchronize(g.s) != cudaSuccess) {
            cudaGetLastError();
            std::fprintf(stderr, "strata-x serve: conversation cache: skipped %lld tokens (the rows' copy failed); parked=%zu "
                                 "bytes=%llu\n",
                         (long long) len, parked.size(), (unsigned long long) parked_bytes());
            return false;
        }
    }
    P->snap = std::make_unique<Checkpoint>();
    P->snap->ids = live;
    P->snap->imgs = live_imgs;
    P->snap->blob.assign(stages.size(), nullptr);
    P->snap->bytes.assign(stages.size(), 0);
    for (size_t si = 0; si < stages.size(); ++si) snapshot(*P->snap, (int) si);
    for (size_t si = 0; si < stages.size(); ++si)
        if (!P->snap->blob[si]) {
            std::fprintf(stderr, "strata-x serve: conversation cache: skipped %lld tokens (no pinned RAM for its running state); "
                                 "parked=%zu bytes=%llu\n",
                         (long long) len, parked.size(), (unsigned long long) parked_bytes());
            return false;
        }
    std::vector<size_t> take(own.begin(), own.end());   // its checkpoints go with it (erased from the back)
    std::sort(take.begin(), take.end(), std::greater<size_t>());
    for (size_t i : take) {
        P->checks.push_back(std::move(checks[i]));
        checks.erase(checks.begin() + (std::ptrdiff_t) i);
    }
    P->ids = live;
    P->imgs = live_imgs;
    P->turn_pos = live_turn_pos;
    P->used = ++park_clock;
    size_t rows_total = 0;
    for (size_t x : P->row_bytes) rows_total += x;
    const size_t nck = P->checks.size();
    const double snap_mib = ck_bytes(*P->snap) / kMiB, mib = P->bytes() / kMiB;
    parked.push_back(std::move(P));
    std::fprintf(stderr, "strata-x serve: conversation cache: parked %lld tokens (%.0f MiB: rows %.0f, the live end's state %.0f, "
                         "%zu checkpoints) in %.0f ms; parked=%zu bytes=%llu evictions=%lld\n",
                 (long long) len, mib, rows_total / kMiB, snap_mib, nck, ms_since(t0), parked.size(),
                 (unsigned long long) parked_bytes(), (long long) park_evictions);
    return true;
}

void Engine::Impl::park_leaving(const std::vector<int32_t>& prompt, const std::vector<ImageKey>& keys, const Parked* keep) {
    if (!live_valid || live.empty()) return;
    const int64_t common = common_prefix(live, live_imgs, prompt, keys);
    if (!park::worth_parking((int64_t) live.size(), common, live_turn_pos, opt.conv_cache_min_tokens)) return;
    park_live(prompt, keys, keep);
}

void Engine::Impl::unpark(size_t i) {
    const auto t0 = clk::now();
    std::unique_ptr<Parked> P = std::move(parked[i]);
    parked.erase(parked.begin() + (std::ptrdiff_t) i);
    const int64_t len = (int64_t) P->ids.size();
    bool ok = true;
    for (size_t si = 0; si < stages.size() && ok; ++si) {
        Stage& g = *stages[si];
        std::vector<std::pair<void*, size_t>> reg;
        g.positional_regions(len, reg);
        set_dev(g.dev);
        ok = cudaStreamSynchronize(g.s) == cudaSuccess;
        size_t off = 0;
        for (const auto& r : reg) {
            if (!ok) break;
            ok = cudaMemcpyAsync(r.first, P->rows[si].get() + off, r.second, cudaMemcpyHostToDevice, g.s) == cudaSuccess;
            off += r.second;
        }
        ok = ok && off == P->row_bytes[si] && cudaStreamSynchronize(g.s) == cudaSuccess;
    }
    if (!ok) {   // the rows are half written: no state of any conversation is trusted, the request reads from 0
        cudaGetLastError();
        live.clear();
        live_imgs.clear();
        live_valid = false;
        checks.clear();
        std::fprintf(stderr, "strata-x serve: conversation cache: the restore of %lld tokens failed - the state is reset; parked=%zu "
                             "bytes=%llu\n",
                     (long long) len, parked.size(), (unsigned long long) parked_bytes());
        return;
    }
    restore(*P->snap);   // the running state at its live end, and the engram's ids for [0, len)
    live = P->ids;
    live_imgs = P->imgs;
    live_turn_pos = P->turn_pos;
    live_valid = true;
    const size_t nck = P->checks.size();
    for (auto& c : P->checks) {
        c->used = ++ck_clock;
        checks.push_back(std::move(c));
    }
    std::fprintf(stderr, "strata-x serve: conversation cache: restored %lld tokens (%zu checkpoints) in %.0f ms; parked=%zu "
                         "bytes=%llu\n",
                 (long long) len, nck, ms_since(t0), parked.size(), (unsigned long long) parked_bytes());
}

bool Engine::Impl::park_try(const std::vector<int32_t>& prompt, const std::vector<ImageKey>& keys, int64_t n, int64_t cur) {
    int best = -1;
    int64_t best_avail = 0;
    for (size_t i = 0; i < parked.size(); ++i) {
        const Parked& p = *parked[i];
        int64_t avail = 0;   // its live end when the prompt continues it, else its longest checkpoint the prompt shares
        if ((int64_t) p.ids.size() <= n - 1 && prefix_of(p.ids, prompt, p.ids.size()) && below(keys, (int64_t) p.ids.size()) == p.imgs)
            avail = (int64_t) p.ids.size();
        for (const auto& c : p.checks)
            if ((int64_t) c->ids.size() <= n - 1 && (int64_t) c->ids.size() > avail && prefix_of(c->ids, prompt, c->ids.size()) &&
                below(keys, (int64_t) c->ids.size()) == c->imgs)
                avail = (int64_t) c->ids.size();
        if (avail > best_avail) {
            best = (int) i;
            best_avail = avail;
        }
    }
    if (best < 0 || !park::restore_pays(best_avail, cur)) return false;
    const Parked* want = parked[(size_t) best].get();
    park_leaving(prompt, keys, want);   // the live one first, by the leave rule - never evicting the one to restore
    for (size_t i = 0; i < parked.size(); ++i)
        if (parked[i].get() == want) {
            unpark(i);
            return live_valid;
        }
    return false;
}

}  // namespace strata::x
