// src/x/park_policy.hpp - moex (DSV41_PLAN §57): the parked conversations' rules as pure functions - park.cpp uses
// them, park_policy_test.cpp walks them by hand.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::x::park {

/// A request LEAVES the live conversation when its prompt parts from the live tokens (`common`: their common prefix, the
/// pictures' keys included) before the live conversation's last turn token (`turn_pos`: where its last prompt's final turn
/// began).  A follow-up shares at least that much: with tools it extends the live tokens (DeepSeek-V4.1's encoder keeps every
/// turn's reasoning then), and in a chat without tools the encoder drops the reasoning of turns before the last user turn, so
/// it parts right after that turn token - never before it.  Another conversation parts long before.
inline bool leaves(int64_t common, int64_t turn_pos) { return common < turn_pos; }

/// Worth parking: the request leaves a conversation of at least `min_tokens`, and at least that many of its tokens would be
/// lost (a long shared prefix stays in the engine for both).
inline bool worth_parking(int64_t live_n, int64_t common, int64_t turn_pos, int64_t min_tokens) {
    return live_n >= min_tokens && leaves(common, turn_pos) && live_n - common >= min_tokens;
}

/// A parked conversation is put back when it gives the prompt at least `gain` tokens more than the live state and the
/// engine's checkpoints can (`avail` against `cur`).
inline bool restore_pays(int64_t avail, int64_t cur, int64_t gain = 1024) { return avail >= cur + gain; }

/// The record to evict first: the least recently used of `stamps`, never index `keep` (-1: none); -1 when there is none.
inline int lru_victim(const std::vector<uint64_t>& stamps, int keep = -1) {
    int v = -1;
    for (int i = 0; i < (int) stamps.size(); ++i)
        if (i != keep && (v < 0 || stamps[(size_t) i] < stamps[(size_t) v])) v = i;
    return v;
}

/// `need` new bytes fit: the budget holds `held` (the records, and the checkpoints the new one keeps) plus `need`, and
/// MemAvailable stays at `floor` or above after the copy.
inline bool fits(uint64_t held, uint64_t need, uint64_t budget, uint64_t mem_avail, uint64_t floor) {
    return held + need <= budget && mem_avail >= need + floor;
}

}  // namespace strata::x::park
