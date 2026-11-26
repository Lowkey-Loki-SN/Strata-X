// src/x/park_policy_test.cpp - moex (DSV41_PLAN §57): the parked conversations' rules, walked by hand (no GPU).
#include "park_policy.hpp"

#include <cstdio>

using namespace strata::x::park;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("%-78s %s\n", what, ok ? "PASS" : "FAIL");
    g_fail += ok ? 0 : 1;
}

int main() {
    // a follow-up: the live conversation is 600,000 tokens, its last prompt's final turn began at 599,000; a chat without tools
    // drops the last reply's reasoning (with tools every turn keeps it), so the follow-up parts right after that turn token
    check(!leaves(599001, 599000), "a follow-up parting after the last turn token stays");
    check(!leaves(599000, 599000), "... parting at the turn token itself stays");
    check(!worth_parking(600000, 599001, 599000, 8192), "... and is not parked");
    // another conversation: it shares the system prompt only
    check(leaves(4731, 599000), "another conversation (the system prompt shared) leaves");
    check(worth_parking(600000, 4731, 599000, 8192), "... and the 600K one is parked");
    check(!worth_parking(6000, 100, 5900, 8192), "a conversation under min_tokens is not parked");
    check(!worth_parking(600000, 595000, 599000, 8192), "a request sharing all but 5,000 tokens: too little lost to park");
    check(worth_parking(600000, 590000, 599000, 8192), "... all but 10,000: parked");
    // restore: a parked record that gives 1,024 more tokens than the live state and the checkpoints
    check(restore_pays(600000, 4731), "a parked 600K conversation against the root: restored");
    check(!restore_pays(5000, 4731), "a parked record only a little longer than what the engine has: not restored");
    check(restore_pays(5755, 4731) && !restore_pays(5754, 4731), "the 1,024-token edge");
    // the budget's order
    check(lru_victim({5, 2, 9}) == 1, "the least recently used record goes first");
    check(lru_victim({5, 2, 9}, 1) == 0, "... never the one being restored");
    check(lru_victim({}) == -1 && lru_victim({3}, 0) == -1, "no record to evict");
    const uint64_t MiB = 1u << 20;
    check(fits(1000 * MiB, 600 * MiB, 4096 * MiB, 20000 * MiB, 8192 * MiB), "600 MiB into a 4 GiB budget with 20 GB free fits");
    check(!fits(3600 * MiB, 600 * MiB, 4096 * MiB, 20000 * MiB, 8192 * MiB), "... over the budget does not");
    check(!fits(0, 600 * MiB, 4096 * MiB, 8500 * MiB, 8192 * MiB), "... nor below the free-RAM floor after the copy");
    std::printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
