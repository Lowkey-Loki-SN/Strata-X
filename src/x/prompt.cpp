// src/x/prompt.cpp - STRATA'S PROMPT PATH, for GLM-5.3-Flash (src/prefill/prefill.cpp; the audit is
// moex/docs/STRATA_QWEN_DESIGN.md section 4).  What carries over, decision by decision:
//
//   * THE CHUNK: Strata's --prefill auto - the largest chunk up to 8192 tokens (a 256 grid) whose buffers and ring
//     fit in the expert slots the stage can LEND (at most 90% of them, at least 128 kept), and a request lends only
//     what min(chunk, its segment) needs.  The slots go back, refilled with queued copies and one sync, afterwards.
//   * THE STREAM: every non-resident expert of every MoE layer of the stage, in (layer, id) order, through a ring of
//     Strata's byte budget (384 Q2_0 blobs = 506 MiB, here as GLM blobs), issued by ITS OWN HOST THREAD whatever the
//     routing, so the copy engine keeps working through the attention halves; slot k % ring is reused once the entry
//     `ring` before it is gathered (its `used` event), the consumer gives entries back a whole MMQ group at a time.
//   * THE MoE: per layer one sync for the routing, a counting sort by expert (token order kept), the activations
//     quantized once in that order, then ONE id-order walk over the routed experts - resident ones from their slots,
//     streamed ones from the ring, skipped stream entries only give their slot back - in MMQ groups of 16 gathered
//     in one launch: gate+up product, SwiGLU (GLM's clamp), q8_1, down product; the combine through the row map.
//   * THE STAGES: a layer split's stages are pipelined over the chunks - a stage hands its residual rows to the next
//     through a pinned double buffer and runs the next stage's chunk on another thread while it reads the next chunk.
//   * THE LAST TOKEN goes through the first decode window (no head here); the NextN block's caches are filled here.
//
// GLM's own: the mixers (KDA / DSA) and the dense FFN run in sub-batches (EngineOptions::prefill_sub) inside the chunk
// - their per-token buffers are ~10x Qwen's (a 64 x 512 absorbed query per token) - only the MoE needs the chunk.
#include "engine_impl.hpp"
#include "strata/x/dsv41_kernels.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <thread>

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

namespace strata::x {

namespace {
constexpr uint64_t kQ2_0Blob = 1382400ull;   ///< the blob Strata's ring budget was measured with
constexpr int kRingCap = 512;               ///< Strata's ring_cap() without the fused Q2_0 kernels
constexpr size_t kMmqTail = 4096;            ///< the zeroed tail after a group's last expert (MMQ reads past it)
/// moex: the SwiGLU over the gate halves and the down rows over gu (Stage::lend); STRATA_X_MOE_ALIAS=0 = the separate
/// rows x nf and rows x n buffers, for an A/B (the same arithmetic: outputs must be bit-identical with a frozen tier)
bool moe_alias() {
    static const bool on = !(std::getenv("STRATA_X_MOE_ALIAS") && std::getenv("STRATA_X_MOE_ALIAS")[0] == '0');
    return on;
}

/// gu's row stride (floats).  moex (DSV41_PLAN §33), opt-in STRATA_DSV41_DM_ALIAS=1: the down rows over gu also where a down
/// row is longer than a gate+up row (n > 2 nf: DeepSeek-V4.1's 5120 against 2 x 2304) - gu's rows n floats apart, so row r's
/// down output lies in gu's row r, consumed (its group's SwiGLU quantized) before its group's down product writes there and
/// past every earlier group's: R x 2 nf floats less a chunk (1.04 GB at 9,439 rows), the same arithmetic at other addresses
int64_t gu_ld(const Hparams& hp) {
    static const bool wide = std::getenv("STRATA_DSV41_DM_ALIAS") && std::getenv("STRATA_DSV41_DM_ALIAS")[0] == '1';
    const int64_t g = 2 * hp.n_ff_exp;
    return wide && moe_alias() ? std::max<int64_t>(g, hp.n_embd) : g;
}

/// moex (DSV41_BACKLOG #81): the prompt heartbeat's drive - the stat file of the disk holding `dir` (/sys/dev/block/M:m/stat,
/// the whole disk's for a partition: serve.cpp's statistics read the same) and its counters
std::string hb_disk_path(const std::string& dir) { return dsv41::DriveGuard::disk_stat(dir); }
struct HbDisk {
    bool ok = false;
    unsigned long long rd_ios = 0, rd_sec = 0, rd_ms = 0, in_flight = 0, io_ms = 0;
};
HbDisk hb_disk(const std::string& path) {
    HbDisk d;
    if (FILE* f = path.empty() ? nullptr : std::fopen(path.c_str(), "r")) {
        unsigned long long v[10] = {};
        d.ok = std::fscanf(f, "%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6],
                           &v[7], &v[8], &v[9]) == 10;
        std::fclose(f);
        d.rd_ios = v[0], d.rd_sec = v[2], d.rd_ms = v[3], d.io_ms = v[9];
    }
    // the requests in flight from the disk's inflight file: the stat file's own field read 0 throughout on this box's kernel
    // (7.0, 2026-10-22: 0 in 871 samples while the inflight file counted 1-2 reads)
    const std::string ip = path.size() > 4 ? path.substr(0, path.size() - 4) + "inflight" : "";
    if (FILE* f = ip.empty() ? nullptr : std::fopen(ip.c_str(), "r")) {
        unsigned long long rd = 0, wr = 0;
        if (std::fscanf(f, "%llu %llu", &rd, &wr) == 2) d.in_flight = rd + wr;
        std::fclose(f);
    }
    return d;
}

/// moex (DSV41_PLAN §18): the bytes this process has read from storage (/proc/self/io read_bytes: O_DIRECT reads and the
/// page cache's misses), or -1
int64_t io_read_bytes() {
    int64_t v = -1;
    if (FILE* f = std::fopen("/proc/self/io", "r")) {
        char key[64];
        long long x = 0;
        while (std::fscanf(f, "%63s %lld", key, &x) == 2)
            if (std::strcmp(key, "read_bytes:") == 0) v = x;
        std::fclose(f);
    }
    return v;
}

struct Lay {
    uint8_t* base;
    size_t off = 0;
    template <typename T>
    T* take(size_t n) {
        off = (off + 255) & ~(size_t) 255;
        T* p = base ? reinterpret_cast<T*>(base + off) : nullptr;
        off += std::max<size_t>(n, 1) * sizeof(T);
        return p;
    }
};
}  // namespace

/// A chunk's stream: the entries, where each layer's start, the issuer thread's and the consumer's positions.
struct Stream {
    struct Entry {
        int32_t l, e;
    };
    std::vector<Entry> seq;
    std::vector<size_t> start;   ///< [il]: the first entry of layer il (start[il + 1]: its end)
    std::atomic<size_t> issued{0}, consumed{0};
    std::atomic<bool> stop{false};
    std::thread issuer;
    size_t k = 0;                ///< the consumer's next entry
    /// DeepSeek-V4.1's SSD tier, a short chunk (Engine::Impl::stream_gate): the issuer reads only the routed experts - each
    /// layer's entries wait for its routing, which moe_prompt publishes, and the others are skipped (their slots go back
    /// unread).  Streaming every non-resident expert ahead read the whole SSD tier (~9,500 records) for a 32-token prompt.
    bool gate = false;
    /// ... and with STRATA_DSV41_GATE_AHEAD=N (opt-in, off: its A/B lay inside the drive's run-to-run swing, DSV41_BACKLOG #31)
    /// a gated chunk of N tokens or more reads the next
    /// layer's records before its routing is known: the drive idles otherwise while each layer's routing is computed (5.9 s
    /// of a 9,918-token chunk's 58.6 s, 2026-10-19), and a long chunk routes nearly every expert - a record it does not
    /// route is dropped once its read has landed (counted as unused)
    bool spec = false;
    /// moex (DSV41_PLAN §59 P15), STRATA_DSV41_FIRST_AHEAD=N (opt-in; N tokens and up, 2048 if 1): the stream's first layer read
    /// ahead of its routing as `spec` reads each next one - the drive is otherwise idle from the stream's start (on GPU 1: the
    /// hand-off from GPU 0) to its first layer's routing; such a chunk routes nearly every expert, the others are dropped
    int64_t first_l = -1;
    std::vector<uint8_t> routed;             ///< [layer * E + e]: the chunk's rows routed to it, at most 255 (valid up to routed_upto)
    std::atomic<int> rows{0};                ///< ... the chunk's rows (moex, §41: STRATA_DSV41_PROMPT_HOT's share of them)
    std::atomic<int64_t> routed_upto{-1};    ///< the last layer whose routing is in `routed`
    /// moex (DSV41_PLAN §62), STRATA_DSV41_PROMPT_GUESS: the issuer's - the layer whose guess it read ahead on, those experts,
    /// and the chunk's counts (read ahead, routed of them)
    int guess_l = -1;
    std::vector<int> guess_e;
    int64_t guess_read = 0, guess_used = 0;
    /// moex (DSV41_BACKLOG #76): a layer-major window's chunk before the last - its MoE reads the layer's streamed experts and
    /// keeps their ring slots for the chunks after it (the last one gives them back)
    bool hold = false;
    /// moex (DSV41_PLAN §33): the ring's records from the call's earlier chunks (Stage::ring_have) - the issuer skips those
    /// entries and marks the ones it adds; the MoE waits for each of a group's copies (they landed at different times)
    std::vector<uint8_t>* have = nullptr;
    /// DeepSeek-V4.1's engram rows of the chunk, fetched on a worker while the stream starts and layer 0 runs (the first engram
    /// layer waits for them): "" or the error.  Declared last, so a chunk that ends early waits for it before the rest goes
    std::future<std::string> eg;
    /// moex (DSV41_BACKLOG #147, DSV41_PLAN §76): where the chunk is fetched layer by layer (Impl::eg_split_ok) `eg` is the first
    /// engram layer's rows and `eg2` the others' - the first engram layer waits for `eg`, a later one for `eg2` - both set by
    /// one worker, `eg_job` (declared after them: a chunk that ends early waits for the worker first)
    std::shared_future<std::string> eg2;
    std::future<void> eg_job;
};

// ------------------------------------------------------------------ the loan

/// The prompt path's device bytes for a chunk of `rows` tokens: the chunk's scratch (mixers in sub-batches of `sub`),
/// the MoE's buffers and tables, the group buffers, the ring.
size_t Engine::Impl::ring_blob(int di) const {
    static const bool own = std::getenv("STRATA_DSV41_STAGE_RING") && std::getenv("STRATA_DSV41_STAGE_RING")[0] == '1';
    size_t b = 0;
    if (own && di >= 0) {   // moex (§25): GPU 0's layers of q3q2u hold 11.1-13.4 MiB experts, the largest blob is 14.5 MiB
        for (int64_t il = hp.n_dense; il < n_xl && il < (int64_t) ldev.size(); ++il)
            if (ldev[(size_t) il] == di) b = std::max<size_t>(b, L[(size_t) il].fmt.bytes);
    }
    if (b == 0) b = blob_max;
    return (b + 255) & ~(size_t) 255;
}

size_t Stage::prompt_bytes(const Engine::Impl& m, int rows, int sub, int ring, bool last, int di, size_t stride, bool cand_rows) {
    const Hparams& hp = m.hp;
    const int64_t n = hp.n_embd, nf = hp.n_ff_exp, E = hp.n_expert, R = (int64_t) rows * hp.n_used;
    const int64_t max_in = max_proj_in(hp);
    size_t b = scratch_bytes(m, rows, sub, cand_rows);
    Lay c{nullptr};
    c.take<uint8_t>(std::max(mmq::q8_bytes(R, n), mmq::q8_bytes(sub, max_in)));   // xq (the routings; dense MMQ)
    c.take<uint8_t>(mmq::q8_bytes(R, nf));                                         // hq
    const int64_t ldg = gu_ld(hp);
    c.take<float>((size_t) (R * ldg));                                             // gu (the SwiGLU in its gate halves)
    if (!moe_alias()) c.take<float>((size_t) (R * nf));                            // hb (A/B only)
    if (n > ldg || !moe_alias()) c.take<float>((size_t) (R * n));                  // dm (else over gu)
    c.take<uint8_t>(kGroup * m.gub_max + kMmqTail);                                // grp_gu
    c.take<uint8_t>(kGroup * m.db_max + kMmqTail);                                 // grp_d
    c.take<int32_t>((size_t) std::max<int64_t>(R, max_in) + 512);                  // ident (+ MMQ's read past its rows, #645)
    c.take<int32_t>((size_t) (2 * R + (E + 1) + ((E + kGroup - 1) / kGroup) * (kGroup + 1)));   // grp
    c.take<uint8_t>((size_t) ring * (stride ? stride : m.ring_blob(di)));          // ring
    if (last) c.take<uint8_t>(dflash_tail_bytes(m, rows));                        // DFlash: a chunk's tail features
    return b + c.off + 256;
}

namespace {
/// moex (DSV41_PLAN §63): /proc/meminfo's MemAvailable in bytes (0 when unreadable)
uint64_t lm_mem_available() {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (fgets(line, sizeof line, f))
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            kb = strtoull(line + 13, nullptr, 10);
            break;
        }
    fclose(f);
    return kb * 1024;
}

/// DeepSeek-V4.1's candidate masks: one row's bytes (the kernels' stride, replay_shift's)
size_t lm_cand_row(const Engine::Impl& m) {
    return (size_t) dsv41_cand_bytes((int) m.dsv41_score_rows(), std::max(1, (int) m.hp.cand_bsize));
}
}  // namespace

/// moex (DSV41_BACKLOG #76): what a layer-major window of k chunks of `rows` adds to the loan - the rows that cross layers of
/// every chunk after the first (Stage::LmRows, carved as lend carves them) and 2 k + 1 copies of the running state
size_t Stage::lm_extra_bytes(int rows, int k, int kw) const {
    if (k <= 0) return 0;
    const size_t snap = ((run_state_bytes() + 255) & ~(size_t) 255) + 256;
    const size_t per = (size_t) rows * hp.hc * hp.n_embd * sizeof(float) + (size_t) rows * 24 * sizeof(float) +
                       (size_t) rows * hp.idx_topk * sizeof(int32_t) + (cand_rows() ? (size_t) rows * lm_cand_row(m) : 0) +
                       4 * 256;
    // moex (§63): with the rows in host memory the loan carves k staging sets, and a copy of the state for each chunk of kw
    return (size_t) (k - 1) * per + (size_t) (2 * std::max(k, kw) + 1) * snap;
}

/// moex (DSV41_PLAN §30): a deferred tail piece's rows - replay_shift's list (the streams, the carried pre-mix, layer 20's
/// top-k lists, its candidate masks)
size_t Stage::tail_rows_bytes(const Engine::Impl& m, int rows) {
    const Hparams& hp = m.hp;
    return (size_t) rows * ((size_t) hp.hc * hp.n_embd * sizeof(float) + 24 * sizeof(float) + (size_t) hp.idx_topk * sizeof(int32_t) +
                            lm_cand_row(m)) + 4 * 256;
}

uint8_t* Stage::lend_top(size_t need, int64_t cap, bool& failed, std::string& err) {
    failed = false;
    int64_t k_lend = (int64_t) ((need + m.slot_bytes - 1) / m.slot_bytes);
    if (!slot_off.empty()) {   // moex (§24): sized slots - the fewest top slots whose bytes hold it
        k_lend = 0;
        while (k_lend < n_slots && lend_bytes(k_lend) < need) ++k_lend;
        if (lend_bytes(k_lend) < need) k_lend = n_slots + 1;
    }
    if (k_lend > cap) return nullptr;
    lent = k_lend;
    lent_keys.assign((size_t) lent, -1);
    bool to_cache = false;   // STRATA_DSV41_PROMPT_KEEP: lent experts' records copied into the SSD cache (below)
    for (int64_t i = 0; i < lent; ++i) {
        const int64_t slot = slot_base + n_slots - lent + i;
        const int32_t key = m.owner[(size_t) slot];
        lent_keys[(size_t) i] = key;
        if (key >= 0) m.resident[(size_t) key] = -1;
        if (key >= 0 && m.rio && m.rio->keep_on) {   // STRATA_DSV41_PROMPT_KEEP: its refill reads it again - keep it
            if (m.lent_key.empty()) m.lent_key.assign((size_t) m.n_xl * hp.n_expert, 0);
            m.lent_key[(size_t) key] = 1;
            // ... and while the slot still holds it, its record copied into an SSD cache slot: the prompt streams it from
            // RAM and the refill takes it from there (both read it from the SSD: ~956 records at 8K tokens, ~5 s each way)
            if (m.on_ssd(key))
                if (const int ci = m.prompt_keep_take(key); ci >= 0) {
                    cudaMemcpyAsync(m.rio->cs[(size_t) ci].p, m.slot_ptr((int32_t) slot), L[(size_t) (key / hp.n_expert)].fmt.bytes,
                                    cudaMemcpyDeviceToHost, s);
                    to_cache = true;
                }
        }
        m.res_dirty = true;
        m.owner[(size_t) slot] = -1;
    }
    if (to_cache && cudaStreamSynchronize(s) != cudaSuccess) {   // the copies out land before the slots become buffers
        failed = true;
        m.fail(err, "copying the lent experts to the SSD cache failed");
        return nullptr;
    }
    st.lent_slots = lent;
    return slots + (slot_off.empty() ? (size_t) (n_slots - lent) * m.slot_bytes : slot_off[(size_t) (n_slots - lent)]);
}

bool Stage::lend(int rows, int sub, int ring_slots, std::string& err) {
    cudaSetDevice(dev);
    if (cudaStreamSynchronize(s) != cudaSuccess || cudaStreamSynchronize(cs) != cudaSuccess)
        return m.fail(err, "a stream failed before the prompt");
    // moex (DSV41_BACKLOG #78): the running state's copies for a stop inside a chunk, after the rest of the loan; (#76) a
    // layer-major window's: 2 k + 1 of them (its start and every chunk end of two windows), and its other chunks' rows
    const int lmk = std::max(0, lm_k);
    const int lmw_n = lmk > 0 ? std::max(lmk, lm_win) : 0;   // (moex, §63: a window of more chunks than the sets)
    const int n_snap = lmk > 0 ? 2 * lmw_n + 1 : m.stop_layers ? run_snap_count() : 0;
    const size_t snap_bytes = n_snap > 0 ? (run_state_bytes() + 255) & ~(size_t) 255 : 0;
    size_t need = prompt_bytes(m, rows, sub, ring_slots, last, this->di, 0, cand_rows()) +
                  (lmk > 0 ? lm_extra_bytes(rows, lmk, lm_win) : (size_t) n_snap * (snap_bytes + 256));
    // moex (DSV41_PLAN §30): a call that defers its tails also holds their run (the chunk part carved again for the longest
    // tail with a ring of a whole layer) and, after it, every piece's rows and state copy; a loan without the room runs the
    // tails as before
    tail_chunk_bytes = 0;
    tails.clear();
    tail_snaps.clear();
    tail_rows_used = 0;
    ring_have.clear();   // (moex, §33: the ring is carved again)
    ring_kept_l = -1;
    keep_stride = 0;
    if (tail_batch && tail_cap > 0) {
        const int tr = (int) m.replay_tail;
        const size_t runner = prompt_bytes(m, tr, std::min(sub, tr), (int) hp.n_expert, last, this->di, 0, cand_rows()) + 256;
        const size_t full = (run_state_bytes() + 255) & ~(size_t) 255;
        const size_t chunk_part = (std::max(need, runner) + 255) & ~(size_t) 255;   // (the pieces' buffers 256-aligned)
        const size_t with = chunk_part + (size_t) tail_snap_cap * full + tail_rows_bytes(m, tail_rows_cap) + (size_t) tail_cap * 4 * 256;
        if (n_slots > 0 && m.slot_bytes > 0 && lend_bytes(lendable()) >= with) {
            tail_chunk_bytes = chunk_part;
            need = with;
        } else {
            tail_batch = false;
        }
        // moex (DSV41_PLAN §33), STRATA_DSV41_RING_KEEP=1: the call's chunks stream only this stage's layers before
        // replay_layer (GPU 1's layer 20) - a ring of a whole layer at their blob, so each record is read once a call
        // (prompt_chunk), not once a chunk; without the room, the ring it had
        static const bool keep_env = std::getenv("STRATA_DSV41_RING_KEEP") && std::getenv("STRATA_DSV41_RING_KEEP")[0] == '1';
        if (tail_batch && keep_env && last && m.rio && m.replay_layer > lb && m.replay_layer <= le) {
            size_t b = 0;
            for (int64_t il = lb; il < m.replay_layer; ++il)
                if (hp.is_moe(il)) b = std::max<size_t>(b, L[(size_t) il].fmt.bytes);
            b = (b + 255) & ~(size_t) 255;
            const size_t need_k = prompt_bytes(m, rows, sub, (int) hp.n_expert, last, this->di, b, cand_rows()) +
                                  (size_t) n_snap * (snap_bytes + 256);
            const size_t chunk_k = (std::max(need_k, runner) + 255) & ~(size_t) 255;
            const size_t with_k = chunk_k + (size_t) tail_snap_cap * full + tail_rows_bytes(m, tail_rows_cap) + (size_t) tail_cap * 4 * 256;
            const size_t room = lend_bytes(lendable());
            if (b > 0 && room >= with_k) {
                keep_stride = b;
                ring_slots = (int) hp.n_expert;
                tail_chunk_bytes = chunk_k;
                need = with_k;
            }
            fprintf(stderr, "strata-x: GPU %d: the call's ring %s layer %lld's records (%lld x %.2f MiB, the loan %.2f of %.2f GiB; "
                            "STRATA_DSV41_RING_KEEP)\n", dev, keep_stride ? "keeps" : "has no room to keep", (long long) lb,
                    (long long) hp.n_expert, b / 1048576.0, with_k / 1073741824.0, room / 1073741824.0);
        }
    }
    uint8_t* base = nullptr;
    lent = 0;
    static const bool no_lend = std::getenv("STRATA_X_NO_LEND") && std::getenv("STRATA_X_NO_LEND")[0] == '1';   // A/B, debugging
    if (n_slots > 0 && m.slot_bytes > 0 && !no_lend) {
        bool failed = false;
        base = lend_top(need, lmk > 0 ? lendable_lm() : lendable(), failed, err);   // (moex, §28: a layer-major read may lend more)
        if (failed) return false;
    }
    if (!base) {   // nothing to lend (no tier, or it is too small): the prompt path's own buffers
        if (own_prompt_bytes < need) {
            if (own_prompt) cudaFree(own_prompt);
            own_prompt = nullptr;
            own_prompt_bytes = 0;
            if (cudaMalloc(&own_prompt, need) != cudaSuccess) {
                cudaGetLastError();
                return m.fail(err, "the prompt path's buffers (" + std::to_string(need >> 20) + " MiB) fit neither the " +
                                       "lendable slots nor the free memory of GPU " + std::to_string(dev));
            }
            own_prompt_bytes = need;
        }
        base = own_prompt;
    }
    loan_base = base;   // (moex, §30: the tails' run carves the chunk part again)
    call_sub = sub;
    if (tail_batch) {   // the cuts' state copies right after the chunk part, the pieces' rows after them (tail_keep)
        const size_t full = (run_state_bytes() + 255) & ~(size_t) 255;
        tail_snaps.assign((size_t) tail_snap_cap, RunSnap{});
        for (int i = 0; i < tail_snap_cap; ++i) tail_snaps[(size_t) i].d = base + tail_chunk_bytes + (size_t) i * full;
    }
    // the layout: the chunk's scratch, then the MoE's buffers, then the ring
    carve(pr, base, rows, sub);
    Lay c{base, scratch_bytes(m, rows, sub, cand_rows())};   // (§59 P3: the carve's own size)
    const int64_t n = hp.n_embd, nf = hp.n_ff_exp, E = hp.n_expert, R = (int64_t) rows * hp.n_used;
    const int64_t max_in = max_proj_in(hp);
    mb.rows = R;
    mb.xq = c.take<uint8_t>(std::max(mmq::q8_bytes(R, n), mmq::q8_bytes(sub, max_in)));
    mb.hq = c.take<uint8_t>(mmq::q8_bytes(R, nf));
    mb.ld_gu = gu_ld(hp);
    mb.gu = c.take<float>((size_t) (R * mb.ld_gu));
    // moex: no rows x nf SwiGLU buffer (written over the gate halves) and, when a down row fits in a gate+up row
    // (n <= 2 nf: MiMo, DSV4, GLM; DeepSeek-V4.1 with gu's rows n apart, gu_ld), the down rows over gu too - row r at
    // gu + r n, inside rows <= r of gu, every one of them consumed (its group's SwiGLU quantized) before its group's down
    // product writes there.  ~1.2-1.6 GB less a GPU at 8192 rows: as many slots fewer lent, so fewer experts streamed per chunk
    mb.hb = moe_alias() ? nullptr : c.take<float>((size_t) (R * nf));
    mb.dm = n <= mb.ld_gu && moe_alias() ? mb.gu : c.take<float>((size_t) (R * n));
    mb.grp_gu = c.take<uint8_t>(kGroup * m.gub_max + kMmqTail);
    mb.grp_d = c.take<uint8_t>(kGroup * m.db_max + kMmqTail);
    const int64_t n_ident = std::max<int64_t>(R, max_in);
    // moex (Maya #645): llama.cpp's MMQ reads up to ~190 ids past a tile's last row; 512 more entries keep that read inside
    // this table (it landed in the next buffer of the same allocation, so it never faulted) - the same bits, 2 KiB
    mb.ident = c.take<int32_t>((size_t) n_ident + 512);
    mb.grp = c.take<int32_t>((size_t) (2 * R + (E + 1) + ((E + kGroup - 1) / kGroup) * (kGroup + 1)));
    ring_stride = keep_stride ? keep_stride : m.ring_blob(this->di);   // (moex, §33: the kept layer's blob)
    ring_n = ring_slots;
    ring = c.take<uint8_t>((size_t) ring_n * ring_stride);
    if (last && m.dfl) {   // DFlash: the chunk's last rows' features, their fusion and K / V (dflash_inject's tail buffers)
        const int64_t r = std::min(rows, m.dfl->cap), nt = (int64_t) m.dfl->tl.size(), kvw = m.dfl->nkv * m.dfl->hd;
        df_feat_pr = c.take<float>((size_t) (r * nt * n));
        df_g_pr = c.take<float>((size_t) (r * n));
        df_k_pr = c.take<float>((size_t) (2 * r * kvw));
    }
    run_snaps.assign((size_t) n_snap, RunSnap{});   // moex (#78): none taken yet in this call
    run_snap_next = 0;
    for (RunSnap& r : run_snaps) r.d = c.take<uint8_t>(snap_bytes);
    lmw.assign((size_t) lmk, LmRows{});   // moex (#76): chunk 0 of a window keeps the scratch's rows, the others their own
    for (int k = 0; k < lmk; ++k) {
        LmRows& w = lmw[(size_t) k];
        if (k == 0) {
            w = {pr.R, pr.cpre, pr.dsel, pr.dcand};
            continue;
        }
        w.R = c.take<float>((size_t) rows * hp.hc * hp.n_embd);
        w.cpre = c.take<float>((size_t) rows * 24);
        w.dsel = c.take<int32_t>((size_t) rows * hp.idx_topk);
        w.dcand = cand_rows() ? c.take<uint8_t>((size_t) rows * lm_cand_row(m)) : nullptr;   // (§59 P3)
    }
    mmq::iota(mb.ident, n_ident + 512, s);   // (the pad too: ids MMQ may read, never uses)
    if (m.sync_check) {   // debugging: where everything of the loan lies
        auto rel = [&](const void* p) { return (long long) ((const uint8_t*) p - (const uint8_t*) base); };
        fprintf(stderr, "strata-x: GPU %d loan: %lld slots of %zu B = %lld B from slot %lld (n_slots %lld); scratch [0, %zu); xq %lld hq %lld gu %lld "
                        "hb %lld dm %lld grp_gu %lld grp_d %lld ident %lld grp %lld; ring [%lld, %lld) = %d x %zu; carve end %zu; need %zu\n",
                dev, (long long) lent, m.slot_bytes, (long long) lend_bytes(lent), (long long) (n_slots - lent), (long long) n_slots,
                scratch_bytes(m, rows, sub, cand_rows()), rel(mb.xq), rel(mb.hq), rel(mb.gu), rel(mb.hb), rel(mb.dm), rel(mb.grp_gu), rel(mb.grp_d),
                rel(mb.ident), rel(mb.grp), rel(ring), rel(ring + (size_t) ring_n * ring_stride), ring_n, ring_stride, c.off, need);
        fprintf(stderr, "strata-x: GPU %d scratch: R %lld cur %lld out %lld xq %lld xq2 %lld fg %lld fu %lld qkv %lld att %lld gpart %lld\n", dev,
                rel(pr.R), rel(pr.cur), rel(pr.out), rel(pr.xq), rel(pr.xq2), rel(pr.fg), rel(pr.fu), rel(pr.qkv), rel(pr.att), rel(pr.gpart));
    }
    if (copied.empty()) {
        copied.resize(kRingCap);
        used.resize(kRingCap);
        for (int i = 0; i < kRingCap; ++i)
            if (cudaEventCreateWithFlags(&copied[(size_t) i], cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&used[(size_t) i], cudaEventDisableTiming) != cudaSuccess)
                return m.fail(err, "ring events");
    }
    used_of.assign(kRingCap, 0);
    for (int i = 0; i < kRingCap; ++i) used_of[(size_t) i] = i;
    live.assign(kRingCap, 0);
    // the pinned host tables, a chunk's worth
    const size_t ints = (size_t) (2 * R + (E + 1) + ((E + kGroup - 1) / kGroup) * (kGroup + 1));
    if (h_grp_cap < ints) {
        host_free(h_grp);
        host_free(h_ids);
        h_grp_cap = 0;
        h_grp = static_cast<int32_t*>(host_buf(ints * sizeof(int32_t)));
        h_ids = static_cast<int32_t*>(host_buf((size_t) R * sizeof(int32_t)));
        if (!h_grp || !h_ids) return m.fail(err, "host tables (out of memory)");
        h_grp_cap = ints;
    }
    if (cudaStreamSynchronize(s) != cudaSuccess) return m.fail(err, "prompt buffers");
    return true;
}

/// The lent slots back: their experts copied in again (queued, one sync), the tables restored.
void Stage::give_back_slots() {
    refill_issue();
    refill_finish();
}

/// give_back_slots' first half: the lent slots' copies queued on this stage's stream (moex: Engine::Impl::prompt queues
/// every stage's before it waits for any, so the two GPUs' refills share the time instead of running one after the other).
void Stage::refill_issue() {
    refilled.clear();
    if (lent == 0) return;
    static const bool refill_async = [] {   // moex (DSV41_PLAN §45), opt-in STRATA_DSV41_REFILL_ASYNC=1
        const char* e = std::getenv("STRATA_DSV41_REFILL_ASYNC");
        return e && e[0] == '1';
    }();
    cudaSetDevice(dev);
    std::string err;
    if (m.sync_check && !sync_check("before the lent slots' refill", (int) lb, err)) fprintf(stderr, "strata-x: %s\n", err.c_str());
    cudaStreamSynchronize(cs);
    cudaStreamSynchronize(s);
    const int64_t NE = hp.n_expert;
    for (int64_t i = 0; i < lent; ++i) {
        const int32_t key = lent_keys[(size_t) i];
        if (key < 0) continue;
        const int32_t slot = (int32_t) (slot_base + n_slots - lent + i);
        const int64_t il = key / NE, e = key % NE;
        const Layer& l = L[(size_t) il];
        if (m.src(key).pinned) {
            cudaMemcpyAsync(m.slot_ptr(slot), m.src(key).g, l.fmt.bytes, cudaMemcpyHostToDevice, s);
            m.owner[(size_t) slot] = key;
            m.resident[(size_t) key] = slot;
        } else if (int cs = refill_async && m.rio && m.on_ssd(key) && m.khash_ok[(size_t) key] ? m.ssd_hold(key) : -1; cs >= 0) {
            // moex (DSV41_PLAN §45): queued from its held SSD cache slot like a RAM copy - fill_slot synced the stream before and
            // after each one, so a GPU's refills ran one at a time and GPU 1's waited for GPU 0's (3,402 after an 84K read)
            cudaMemcpyAsync(m.slot_ptr(slot), m.rio->cs[(size_t) cs].p, l.fmt.bytes, cudaMemcpyHostToDevice, s);
            m.owner[(size_t) slot] = key;
            m.resident[(size_t) key] = slot;
            ++m.rio->fill_cached;
            refill_held.push_back(cs);
        } else {
            m.fill_slot(slot, il, e, err);
        }
        refilled.push_back({slot, key});
    }
}

/// give_back_slots' second half: the copies waited for, verified, the window buffers back in use.
void Stage::refill_finish() {
    if (lent == 0) {
        use(win);
        return;
    }
    cudaSetDevice(dev);
    std::string err;
    if (m.sync_check && !sync_check("the lent slots' refill", (int) lb, err)) fprintf(stderr, "strata-x: %s\n", err.c_str());
    cudaStreamSynchronize(s);
    for (const int cs : refill_held) m.ssd_release(cs);   // (moex, §45: the copies from them have landed)
    refill_held.clear();
    if (!m.verify_slots(refilled, false, err)) fprintf(stderr, "strata-x: the refilled slots' check: %s\n", err.c_str());
    refilled.clear();
    lent = 0;
    lent_keys.clear();
    use(win);
}

// ------------------------------------------------------------------ the stream's issuer

namespace {
/// DeepSeek-V4.1's SSD tier: the issuer's records read ahead into the stage's bounce slots, all but one of them in flight,
/// in stream order - so the drive always has the next records queued while a copy is issued (one record at a time, the
/// gated 200-token prompt read at ~2.2 GB/s of the drive's ~2.7).  A gated chunk reads ahead only as far as its routing is
/// known.  The bounce slots are taken round robin and freed in the same order (each by its copy's event).
struct Prefetch {
    struct P {
        size_t idx;
        int b;   ///< its bounce slot; -1 - s: SSD cache slot s (STRATA_DSV41_PROMPT_KEEP)
        dsv41::ExpertIo::Handle h;
    };
    std::deque<P> q;
    size_t next = 0;   ///< the next stream entry to look at
    size_t bounces() const {   // the reads in q into bounce slots (the others land in SSD cache slots)
        size_t k = 0;
        for (const P& x : q) k += x.b >= 0;
        return k;
    }
    bool idle() const {   // none of its reads in flight: the drive has no work of this stage's (the idle stats)
        for (const P& x : q)
            if (x.h && !x.h->done.load(std::memory_order_acquire)) return false;
        return true;
    }
};

/// the issuer's wait `what` (a spin's one turn) added to `acc` when none of the stage's reads is in flight meanwhile
template <class F> void idle_turn(Stage& g, const Prefetch& pf, std::atomic<int64_t> Engine::Impl::RecordIo::*acc, F&& what) {
    const bool idle = g.m.rio && pf.idle();
    const auto t = clk::now();
    what();
    if (idle) {
        const int64_t us = (int64_t) (ms_since(t) * 1000.0);
        (g.m.rio.get()->*acc).fetch_add(us, std::memory_order_relaxed);
        if (acc == &Engine::Impl::RecordIo::us_gate) g.us_gate_own.fetch_add(us, std::memory_order_relaxed);
    }
}

/// Records read ahead for stream entries before `end` that nothing took (an expert a read before the routing guessed and
/// the routing left out, or one issue_entry served from elsewhere): each read lands - its bounce slot is reused only then -
/// and leaves the queue, which so stays in step with the stream (a stale front would make every later record a read on
/// the spot)
void drop_before(Stage& g, Prefetch& pf, size_t end) {
    while (!pf.q.empty() && pf.q.front().idx < end) {
        g.m.rio->io->wait(pf.q.front().h);
        pf.q.pop_front();
        g.m.rio->prompt_orphan.fetch_add(1, std::memory_order_relaxed);
    }
}

/// moex (DSV41_BACKLOG #51), STRATA_DSV41_EG_AHEAD=1: the next chunk's engram rows fetched while this chunk runs
bool eg_ahead() {
    static const bool v = std::getenv("STRATA_DSV41_EG_AHEAD") && std::getenv("STRATA_DSV41_EG_AHEAD")[0] == '1';
    return v;
}

/// STRATA_DSV41_GATE_AHEAD=N: the shortest chunk that reads each layer's records before its routing (Stream::spec); 0: off
int64_t gate_ahead() {
    static const int64_t v = [] {
        const char* e = std::getenv("STRATA_DSV41_GATE_AHEAD");
        return e ? std::max<int64_t>(0, std::atoll(e)) : (int64_t) 0;
    }();
    return v;
}

void issue_entry(Stage& g, const Stream::Entry& en, int sl, size_t idx, Prefetch& pf) {
    const Layer& l = g.L[(size_t) en.l];
    uint8_t* dst = g.ring + (size_t) sl * g.ring_stride;
    if (g.m.sync_check && (sl < 0 || sl >= g.ring_n || l.fmt.bytes > g.ring_stride)) {
        fprintf(stderr, "strata-x: RING WRITE OUT OF RANGE: GPU %d slot %d of %d, layer %d expert %d, %zu B into a stride of %zu B\n", g.dev, sl,
                g.ring_n, en.l, en.e, l.fmt.bytes, g.ring_stride);
        abort();
    }
    if (g.live[(size_t) sl]) cudaStreamWaitEvent(g.cs, g.used[(size_t) g.used_of[(size_t) sl]], 0);
    drop_before(g, pf, idx);   // (none expected: the issuer drops a skipped entry's read itself)
    const int64_t key = (int64_t) en.l * g.hp.n_expert + en.e;
    const Engine::Impl::Src sr = g.m.src(key);
    if (sr.pinned) {   // the RAM copy: one blob
        cudaMemcpyAsync(dst, sr.g, l.fmt.bytes, cudaMemcpyHostToDevice, g.cs);
    } else if (!pf.q.empty() && pf.q.front().idx == idx && pf.q.front().b < 0) {   // read ahead into an SSD cache slot
        // (STRATA_DSV41_PROMPT_KEEP): it stays there for the refill, the decode and the next prompts; the running prompt keeps
        // the slot from eviction, so the copy reads it safely
        const int ci = -1 - pf.q.front().b;
        g.m.rio->io->wait(pf.q.front().h);
        const bool ok = pf.q.front().h->error.load() == 0;
        pf.q.pop_front();
        if (ok) {
            cudaMemcpyAsync(dst, g.m.rio->cs[(size_t) ci].p, l.fmt.bytes, cudaMemcpyHostToDevice, g.cs);
        } else {   // a failed read leaves the cache, and must not feed stale weights.  moex (DSV41_BACKLOG #47): the record is
                   // read once more, into a bounce slot; a second failure zeroes the slot and ends the chunk (prompt_layer)
            fprintf(stderr, "strata-x: the prompt stream: an O_DIRECT read of layer %d expert %d failed; reading it again\n", en.l,
                    en.e);
            g.m.prompt_keep_drop(ci);
            const int b = g.rbounce_next;
            g.rbounce_next = (b + 1) % (int) g.rbounce.size();
            cudaEventSynchronize(g.rbounce_ev[b]);   // its previous copy has left it
            std::string rerr;
            const int64_t e = en.e;
            if (g.m.rec_read(en.l, &e, 1, &g.rbounce[b], rerr)) {
                cudaMemcpyAsync(dst, g.rbounce[b], l.fmt.bytes, cudaMemcpyHostToDevice, g.cs);
            } else {
                g.read_failed.store(1 + key);
                cudaMemsetAsync(dst, 0, l.fmt.bytes, g.cs);
            }
            cudaEventRecord(g.rbounce_ev[b], g.cs);
        }
    } else if (const uint8_t* rc = g.m.on_ssd(key) ? g.m.ssd_cached(key) : nullptr; rc) {   // DeepSeek-V4.1's SSD tier: a
        cudaMemcpyAsync(dst, rc, l.fmt.bytes, cudaMemcpyHostToDevice, g.cs);   // record the decode read lately, from its cache slot
    } else if (g.m.on_ssd(key)) {   // ... else the record read ahead into a pinned bounce slot (O_DIRECT), then the copy
        int b;
        bool ok;
        if (!pf.q.empty() && pf.q.front().idx == idx) {
            b = pf.q.front().b;
            g.m.rio->io->wait(pf.q.front().h);
            ok = pf.q.front().h->error.load() == 0;
            pf.q.pop_front();
        } else {   // not read ahead (the read-ahead takes every SSD entry in order; kept for safety): one read now
            g.m.rio->prompt_sync.fetch_add(1, std::memory_order_relaxed);
            b = g.rbounce_next;
            g.rbounce_next = (b + 1) % (int) g.rbounce.size();
            cudaEventSynchronize(g.rbounce_ev[b]);   // its previous copy has left it
            std::string err;
            const int64_t e = en.e;
            ok = g.m.rec_read(en.l, &e, 1, &g.rbounce[b], err);
        }
        if (!ok) {   // moex (DSV41_BACKLOG #47): once more; a second failure zeroes the slot (no stale weights) and ends the chunk
            fprintf(stderr, "strata-x: the prompt stream: an O_DIRECT read of layer %d expert %d failed; reading it again\n", en.l,
                    en.e);
            std::string rerr;
            const int64_t e = en.e;
            if (!g.m.rec_read(en.l, &e, 1, &g.rbounce[b], rerr)) {
                g.read_failed.store(1 + key);
                memset(g.rbounce[b], 0, l.fmt.bytes);
            }
        }
        cudaMemcpyAsync(dst, g.rbounce[b], l.fmt.bytes, cudaMemcpyHostToDevice, g.cs);
        cudaEventRecord(g.rbounce_ev[b], g.cs);
    } else {   // in place in the GGUF mapping: the three slices
        cudaMemcpyAsync(dst, sr.g, l.sg, cudaMemcpyHostToDevice, g.cs);
        cudaMemcpyAsync(dst + l.fmt.up_off, sr.u, l.su, cudaMemcpyHostToDevice, g.cs);
        cudaMemcpyAsync(dst + l.fmt.down_off, sr.d, l.sd, cudaMemcpyHostToDevice, g.cs);
    }
    cudaEventRecord(g.copied[(size_t) sl], g.cs);
    g.live[(size_t) sl] = 1;
    drop_before(g, pf, idx + 1);   // a record read ahead for this entry that the branches above did not take (none expected)
}

/// The read-ahead's next records from `from` on: SSD entries that issue_entry will read (not in RAM, not in the SSD cache,
/// routed when the chunk is gated), while the queue has room and the routing is known.  `block` = false (the ring is full):
/// only into bounce slots whose copy has landed - a copy queued behind the full ring waits for the consumer.
void read_ahead(Stage& g, Stream& str, Prefetch& pf, size_t from, bool block) {
    if (!g.m.rio) return;
    const int64_t E = g.hp.n_expert;
    if (pf.next < from) pf.next = from;
    // moex (DSV41_PLAN §59 P5), STRATA_DSV41_DEEP_AHEAD=1: the reads into SSD cache slots (PROMPT_KEEP) are not held to the
    // bounce slots' count - only the bounce reads are (a slot is reused round by round), and the queue to 4 x that: the drive
    // keeps reading while the ring is full.  The same records, the same slots: exact
    static const bool deep = std::getenv("STRATA_DSV41_DEEP_AHEAD") && std::getenv("STRATA_DSV41_DEEP_AHEAD")[0] == '1';
    // (§63: in a one-window call, whose read-ahead may take held cache slots past the share, the queue to 16 x)
    const size_t qmax = deep ? (g.m.rio->deep_cap > 0 ? 16 : 4) * g.rbounce.size() : g.rbounce.size();
    while (pf.q.size() + 1 < qmax && pf.next < str.seq.size()) {
        if (g.m.eg_hold()) break;   // moex (#76, §64): the engram's reads first
        if (str.have && (*str.have)[pf.next]) {   // moex (§33): in the ring from an earlier chunk
            ++pf.next;
            continue;
        }
        const Stream::Entry& en = str.seq[pf.next];
        if (str.gate) {
            const int64_t known = str.routed_upto.load(std::memory_order_acquire);
            if (known < en.l) {   // the layer's routing is not known yet: wait for it - or, a long chunk (Stream::spec), read
                // the next layer's records before it - or the stream's first layer's (§59 P15)
                if (!(str.spec && en.l == known + 1) && en.l != str.first_l) break;
            } else if (!str.routed[(size_t) en.l * E + en.e]) {
                ++pf.next;
                continue;
            }
        }
        const int64_t key = (int64_t) en.l * E + en.e;
        if (g.m.on_ssd(key) && g.m.rio->keep_on) {   // STRATA_DSV41_PROMPT_KEEP: into a cache slot while the prompt's share lasts
            const bool known = str.gate && str.routed_upto.load(std::memory_order_acquire) >= en.l;   // (§41: its routing count)
            const int ci = g.m.prompt_keep_slot(key, known ? str.routed[(size_t) en.l * E + en.e] : 0,
                                                 known ? str.rows.load(std::memory_order_relaxed) : 0);
            if (ci >= 0) {
                pf.q.push_back({pf.next, -1 - ci, g.m.rio->cs[(size_t) ci].h});
                ++pf.next;
                continue;
            }
            if (ci == -2) {   // cached already: issue_entry copies it from its slot
                ++pf.next;
                continue;
            }
        }
        if (g.m.on_ssd(key) && (g.m.rio->keep_on || !g.m.rio->where.count(key))) {   // between decodes the cache does not change
            if (deep && pf.bounces() + 1 >= g.rbounce.size()) break;   // (§59 P5) every bounce slot holds a read in flight
            const int b = g.rbounce_next;
            if (!block && cudaEventQuery(g.rbounce_ev[b]) != cudaSuccess) {
                cudaGetLastError();
                break;
            }
            g.rbounce_next = (b + 1) % (int) g.rbounce.size();
            if (cudaEventQuery(g.rbounce_ev[b]) != cudaSuccess) {   // its previous copy has not left it yet
                cudaGetLastError();
                idle_turn(g, pf, &Engine::Impl::RecordIo::us_bounce, [&] { cudaEventSynchronize(g.rbounce_ev[b]); });
            }
            pf.q.push_back({pf.next, b, g.m.rec_submit(en.l, en.e, g.rbounce[b])});
        }
        ++pf.next;
    }
}

/// moex (DSV41_PLAN §62), STRATA_DSV41_PROMPT_GUESS=N (opt-in; 0 off): the largest number of a layer's drive records a gated
/// chunk's issuer reads into SSD cache slots on the layer's guessed routing (Stage::moe_prompt's guess, one layer before)
/// while the routing comes - the drive is idle then.  Only with PROMPT_KEEP: a record read so stays in its slot, the
/// routing's read of it waits for that read (ssd_cached), and one it does not route stays cached.  Exact: where a record is
/// read from changes no result
int prompt_guess() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_DSV41_PROMPT_GUESS");
        return e ? std::max(0, std::atoi(e)) : 0;
    }();
    return v;
}
/// ... STRATA_DSV41_PROMPT_GUESS_HOT=D: such a read past the prompt's share takes the least recently used slot when D x its
/// guessed rows >= the chunk's rows (PROMPT_HOT's rule with its own divisor; unset: PROMPT_HOT's)
int prompt_guess_hot() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_DSV41_PROMPT_GUESS_HOT");
        return e ? std::max(0, std::atoi(e)) : -1;
    }();
    return v;
}

/// ... the reads, once the guess for layer L has landed and the drive has none of the stage's reads: the layer's experts
/// outside VRAM, the RAM copy and the SSD cache that two or more of the chunk's rows were guessed to - the N most guessed,
/// read in the order the MoE takes them (by id)
void guess_reads(Stage& g, Stream& str, const Prefetch& pf, int L) {
    if (prompt_guess() <= 0 || !g.m.rio || !g.m.rio->keep_on || str.guess_l == L) return;
    if (!pf.idle()) return;   // only once the layer before has every read landed: the drive idle, nothing to delay
    if (g.m.eg_hold()) return;   // (#76, §64: the engram's reads first)
    const int b = L & 1, E = (int) g.hp.n_expert;
    std::vector<int32_t> votes((size_t) E, 0);
    int rows = 0;
    {
        std::lock_guard<std::mutex> lk(g.gs_mu);
        if (g.gs_for[b] != L || g.gs_str[b] != &str) return;
        if (cudaEventQuery(g.gs_ev[b]) != cudaSuccess) {
            cudaGetLastError();
            return;
        }
        rows = g.gs_rows[b];
        const int32_t* gh = g.gs_hb[b];
        for (int i = 0; i < rows * (int) g.hp.n_used; ++i)
            if (gh[i] >= 0 && gh[i] < E) votes[(size_t) gh[i]]++;
    }
    str.guess_l = L;
    str.guess_e.clear();
    std::vector<int> cand;
    {
        std::lock_guard<std::mutex> lk(g.m.rio->mu);
        for (size_t i = str.start[(size_t) L]; i < str.start[(size_t) L + 1]; ++i) {
            const int e = str.seq[i].e;
            const int64_t key = (int64_t) L * E + e;
            if (votes[(size_t) e] >= 2 && !(str.have && (*str.have)[i]) && !g.m.src(key).pinned && g.m.on_ssd(key) &&
                !g.m.rio->where.count(key))
                cand.push_back(e);
        }
    }
    std::stable_sort(cand.begin(), cand.end(), [&](int a, int c) { return votes[(size_t) a] > votes[(size_t) c]; });
    if ((int) cand.size() > prompt_guess()) cand.resize((size_t) prompt_guess());
    std::sort(cand.begin(), cand.end());   // (the stream's order: the MoE's groups go by expert id)
    for (int e : cand)
        if (g.m.prompt_keep_slot((int64_t) L * E + e, votes[(size_t) e], rows, prompt_guess_hot()) >= 0) str.guess_e.push_back(e);
    str.guess_read += (int64_t) str.guess_e.size();
}

void start_issuer(Stage& g, Stream& str) {
    if (str.seq.empty()) return;
    str.issuer = std::thread([&g, &str] {
        cudaSetDevice(g.dev);
        Prefetch pf;
        struct Drain {   // reads still in flight land before the bounce slots are used again
            Stage& g;
            Prefetch& pf;
            ~Drain() {
                for (const auto& x : pf.q) g.m.rio->io->wait(x.h);
            }
        } drain{g, pf};
        for (size_t idx = 0; idx < str.seq.size(); ++idx) {
            while (idx >= str.consumed.load(std::memory_order_acquire) + (size_t) g.ring_n) {
                if (str.stop.load(std::memory_order_acquire)) return;
                read_ahead(g, str, pf, idx, false);   // the drive works while the ring is full
                idle_turn(g, pf, &Engine::Impl::RecordIo::us_ring, [] { std::this_thread::yield(); });
            }
            if (str.have && (*str.have)[idx]) {   // moex (§33): in the ring from an earlier chunk of the call
                str.issued.store(idx + 1, std::memory_order_release);
                g.hb_issued.store(idx + 1, std::memory_order_relaxed);
                continue;
            }
            const Stream::Entry& en = str.seq[idx];
            if (str.gate) {   // a short chunk on the SSD tier: the layer's routing first, then its routed experts only
                while (str.routed_upto.load(std::memory_order_acquire) < en.l) {
                    if (str.stop.load(std::memory_order_acquire)) return;
                    if (str.spec || en.l == str.first_l) read_ahead(g, str, pf, idx, false);   // the layer's records while its
                                                                                                // routing comes
                    guess_reads(g, str, pf, (int) en.l);   // (moex, §62: or its guessed ones)
                    idle_turn(g, pf, &Engine::Impl::RecordIo::us_gate, [] { std::this_thread::yield(); });
                }
                if (str.guess_l == (int) en.l && !str.guess_e.empty()) {   // (§62) the guess's reads the routing took
                    for (int e : str.guess_e) str.guess_used += str.routed[(size_t) en.l * g.hp.n_expert + e] > 0;
                    str.guess_e.clear();
                }
                if (!str.routed[(size_t) en.l * g.hp.n_expert + en.e]) {   // the consumer gives its slot back unread
                    drop_before(g, pf, idx + 1);   // ... and a read of it before the routing leaves the queue
                    str.issued.store(idx + 1, std::memory_order_release);
                    continue;
                }
            }
            if (g.m.eg_first || g.m.eg_urgent) {   // moex (#76, §64): while the engram's fetch has the drive (eg_hold), a record
                                                    // that needs a read of it waits
                auto needs_drive = [&] {
                    const int64_t key = (int64_t) en.l * g.hp.n_expert + en.e;
                    if (g.m.src(key).pinned || (!pf.q.empty() && pf.q.front().idx == idx)) return false;
                    return g.m.on_ssd(key) && !g.m.ssd_cached(key);
                };
                while (g.m.eg_hold() && needs_drive()) {
                    if (str.stop.load(std::memory_order_acquire)) return;
                    const auto t = clk::now();
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                    (g.m.eg_first ? g.m.us_eg_first : g.m.us_eg_urgent).fetch_add((int64_t) (ms_since(t) * 1000.0), std::memory_order_relaxed);
                }
            }
            read_ahead(g, str, pf, idx, true);
            issue_entry(g, en, (int) (idx % (size_t) g.ring_n), idx, pf);
            if (str.have) (*str.have)[idx] = 1;
            str.issued.store(idx + 1, std::memory_order_release);
            g.hb_issued.store(idx + 1, std::memory_order_relaxed);   // moex (#81): the heartbeat's view
            g.hb_ahead.store((int) pf.q.size(), std::memory_order_relaxed);
        }
        if (str.guess_read > 0)   // (moex, §62)
            fprintf(stderr, "strata-x: prompt guess (GPU %d): %lld records read ahead of their layers' routing on its guess, %lld of "
                            "them routed\n", g.dev, (long long) str.guess_read, (long long) str.guess_used);
    });
}

void stop_issuer(Stream& str) {
    if (!str.issuer.joinable()) return;
    str.stop.store(true, std::memory_order_release);
    str.issuer.join();
}
}  // namespace

// ------------------------------------------------------------------ the MoE of a chunk

bool Stage::moe_prompt(int il, int T, Stream& str, std::string& err) {
    const Layer& l = L[(size_t) il];
    const kc::NativeFmt& f = l.fmt;
    const int n = (int) hp.n_embd, E = (int) hp.n_expert, k = (int) hp.n_used, nf = (int) hp.n_ff_exp;
    // Hy4-flash's and ERNIE-5.5's latent MoE (hy4.cpp): the router and the shared expert read cur, the routed experts xl, nx wide
    const bool lat = (bool) l.r_down;
    const int nx = lat ? (int) hp.ex() : n;
    const int R = T * k;
    const auto t0 = clk::now();
    gemv_dt(l.gate_inp, cur, router, n, (int) l.gate_inp.n_out(), T);   // (ERNIE-5.5: 2 E logits)
    route(il, T, d_toks, m.prompt_pen_for(il));   // DeepSeek-V4's hash layers: the chunk's tokens (prompt_chunk uploaded them)
    if (!sync_check("MoE routing", il, err)) return false;
    cudaMemcpyAsync(h_ids, ids, (size_t) R * sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    cudaEventRecord(ev, s);
    // the shared expert while the host waits for the routing
    for (int b0 = 0; hp.shexp && b0 < T; b0 += SB) {
        const int tn = std::min(SB, T - b0);
        swiglu_ffn(il, tn, l.sh_gate, l.sh_up, l.sh_down, hp.clamp_sh[(size_t) il], cur + (size_t) b0 * n,
                   shared + (size_t) b0 * n);
    }
    if (lat) hy4_moe_in(il, T);   // ... and the experts' latent input, meanwhile too
    const auto tw = clk::now();
    if (cudaEventSynchronize(ev) != cudaSuccess) return m.fail(err, "the routing's readback failed");
    st.ms_prefill_wait += ms_since(tw);
    m.trace_route(1000 + il, T, k, h_ids);   // (STRATA_ROUTE_TRACE: a prompt chunk's layer, marked + 1000)
    // the counting sort by expert (token order kept within an expert)
    std::vector<int32_t> cnt((size_t) E, 0), off((size_t) E + 1, 0);
    for (int i = 0; i < R; ++i) {
        const int e = h_ids[i];
        if (e < 0 || e >= E) return m.fail(err, "router produced expert " + std::to_string(e));
        cnt[(size_t) e]++;
    }
    for (int e = 0; e < E; ++e) off[(size_t) e + 1] = off[(size_t) e] + cnt[(size_t) e];
    // moex (§62), STRATA_DSV41_PROMPT_GUESS_LOG=1: the guess one layer before, scored - of this layer's experts the stream would
    // read from the drive (not in VRAM, the RAM copy or the SSD cache), the ones with the most guessed rows first: how many of
    // the top K this layer routes.  STRATA_DSV41_PROMPT_GUESS=N: the issuer reads on the guess (guess_reads).  Then the next
    // layer's guess: its FFN norm, router and top-k on this layer's FFN rows (the decode read-ahead's guess, ahead_compute)
    static const bool guess_log = [] {
        const char* e = std::getenv("STRATA_DSV41_PROMPT_GUESS_LOG");
        return e && e[0] == '1';
    }();
    // (a call of one chunk at a time: a layer-major window runs each layer over its chunks before the next layer)
    if ((guess_log || prompt_guess() > 0) && str.gate && m.rio && !str.hold && T <= 4096) {
        if (guess_log && gs_for[il & 1] == il && gs_str[il & 1] == &str && gs_rows[il & 1] == T) {   // (its copy landed: the routing's sync came after it)
            const int32_t* gh = gs_hb[il & 1];
            std::vector<int32_t> votes((size_t) E, 0);
            for (int i = 0; i < R; ++i)
                if (gh[i] >= 0 && gh[i] < E) votes[(size_t) gh[i]]++;
            std::vector<int> cand;
            {
                std::lock_guard<std::mutex> lk(m.rio->mu);   // (where, not ssd_cached: the probe pins and ages nothing)
                for (size_t i = str.start[(size_t) il]; i < str.start[(size_t) il + 1]; ++i) {
                    const int e = str.seq[i].e;
                    const int64_t key = (int64_t) il * E + e;
                    if (!m.src(key).pinned && m.on_ssd(key) && !m.rio->where.count(key)) cand.push_back(e);
                }
            }
            std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return votes[(size_t) a] > votes[(size_t) b]; });
            int64_t drive = 0;
            for (int e : cand) drive += cnt[(size_t) e] > 0;
            gs_n[0]++;
            gs_n[1] += drive;
            const int Ks[4] = {8, 16, 24, 32};
            for (int q = 0; q < 4; ++q)
                for (int j = 0; j < (int) cand.size() && j < Ks[q] && votes[(size_t) cand[(size_t) j]] > 0; ++j) {
                    gs_n[2 + 2 * q]++;
                    gs_n[3 + 2 * q] += cnt[(size_t) cand[(size_t) j]] > 0;
                }
        }
        if (il + 1 < le && hp.is_moe(il + 1) && L[(size_t) il + 1].gate_inp) {
            const int b = (int) ((il + 1) & 1);
            {
                std::lock_guard<std::mutex> lk(gs_mu);   // the issuer leaves this buffer now
                gs_for[b] = -1;
                if (gs_cap < T) gs_for[0] = gs_for[1] = -1;
            }
            if (gs_cap < T) {   // (both buffers out of the issuer's reach above)
                const int64_t cap = std::max<int64_t>(T, 1024);
                for (int i = 0; i < 2; ++i) {
                    if (gs_hb[i]) cudaFreeHost(gs_hb[i]);
                    gs_hb[i] = nullptr;
                    if (!gs_ev[i] && cudaEventCreateWithFlags(&gs_ev[i], cudaEventDisableTiming) != cudaSuccess)
                        return m.fail(err, "the prompt guess's events");
                    if (cudaMallocHost(&gs_hb[i], (size_t) cap * k * sizeof(int32_t)) != cudaSuccess)
                        return m.fail(err, "the prompt guess's host buffers");
                }
                if (!(gs_x = dalloc<float>((size_t) cap * n, err)) ||
                    !(gs_ids = reinterpret_cast<int32_t*>(dalloc<float>((size_t) cap * k, err))) || !(gs_wts = dalloc<float>((size_t) cap * k, err)))
                    return m.fail(err, "the prompt guess's buffers");
                gs_cap = cap;
            }
            const Layer& nl = L[(size_t) il + 1];
            rms_norm(cur, nl.ffn_norm.f(), gs_x, T, n, rms_eps(), s);
            gemv_dt(nl.gate_inp, gs_x, router, n, E, T);
            dsv4_route(router, nl.probs_b ? nl.probs_b.f() : nullptr, nullptr, E, k, hp.ew_scale, gs_ids, gs_wts, T, s,
                       m.prompt_pen_for(il + 1), true);
            cudaMemcpyAsync(gs_hb[b], gs_ids, (size_t) R * sizeof(int32_t), cudaMemcpyDeviceToHost, s);
            cudaEventRecord(gs_ev[b], s);
            std::lock_guard<std::mutex> lk(gs_mu);
            gs_for[b] = (int) il + 1;
            gs_str[b] = &str;
            gs_rows[b] = T;
        }
        // the stage's last MoE layer (LAST_RING: the last layer runs no MoE): the chunk's line
        if (guess_log && (il + 1 == le || (il + 2 == le && m.last_ring_only(le - 1))) && gs_n[0] > 0) {
            const double ly = (double) gs_n[0];
            fprintf(stderr, "strata-x: prompt guess (GPU %d, %d rows): %lld layers, %.1f drive reads a layer; the guess's top 8 / 16 / "
                            "24 / 32 of them: %.1f / %.1f / %.1f / %.1f taken, %.1f / %.1f / %.1f / %.1f routed a layer\n",
                    dev, T, (long long) gs_n[0], gs_n[1] / ly, gs_n[2] / ly, gs_n[4] / ly, gs_n[6] / ly, gs_n[8] / ly, gs_n[3] / ly,
                    gs_n[5] / ly, gs_n[7] / ly, gs_n[9] / ly);
            std::fill(std::begin(gs_n), std::end(gs_n), 0);
        }
    }
    if (str.gate) {   // the issuer waits for this: a short chunk on the SSD tier reads only the routed experts
        for (int e = 0; e < E; ++e) str.routed[(size_t) il * E + e] = (uint8_t) std::min<int32_t>(255, cnt[(size_t) e]);
        str.rows.store(T, std::memory_order_relaxed);
        str.routed_upto.store(il, std::memory_order_release);
    }
    int32_t* src_h = h_grp;
    int32_t* slot_h = h_grp + R;
    {
        std::vector<int32_t> fill(off.begin(), off.end() - 1);
        for (int i = 0; i < R; ++i) {
            const int e = h_ids[i];
            const int32_t p = fill[(size_t) e]++;
            slot_h[i] = p;
            src_h[p] = i / k;
        }
    }
    std::vector<int32_t> order;
    for (int e = 0; e < E; ++e)
        if (cnt[(size_t) e] > 0) order.push_back(e);
    const int nu = (int) order.size(), ngr = (nu + kGroup - 1) / kGroup;
    int32_t* abs_h = slot_h + R;
    int32_t* rel_h = abs_h + (nu + 1);
    for (int j = 0; j < nu; ++j) abs_h[j] = off[(size_t) order[(size_t) j]];
    abs_h[nu] = R;
    for (int gi = 0; gi < ngr; ++gi)
        for (int qi = 0; qi <= kGroup; ++qi)
            rel_h[gi * (kGroup + 1) + qi] = abs_h[std::min(nu, gi * kGroup + qi)] - abs_h[gi * kGroup];
    const size_t ints = (size_t) (2 * R + (nu + 1) + ngr * (kGroup + 1));
    cudaMemcpyAsync(mb.grp, h_grp, ints * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    const int32_t* d_src = mb.grp;
    const int32_t* d_slot = mb.grp + R;
    const int32_t* d_abs = d_slot + R;
    const int32_t* d_rel = d_abs + (nu + 1);
    mmq::quantize(lat ? hxl : cur, d_src, mb.xq, f.gu_type, nx, nx, R, s);   // the activations once, in expert order
    if (m.imx) {   // the imatrix collection: each routed expert's squared inputs (its tokens' FFN rows)
        cudaMemcpyAsync(imx_order, order.data(), (size_t) nu * sizeof(int32_t), cudaMemcpyHostToDevice, s);   // pageable: staged
        // moex (2026-11-27): a latent MoE's experts (Hy4-flash, ERNIE-5.5) read the latent rows, nx wide - the hidden rows
        // were summed here before; elsewhere nx = n and the rows are cur, as before
        dsv41_imx_sumsq(lat ? hxl : cur, nx, d_src, d_abs, imx_order, nu, nx, m.imx->gu[(size_t) il], s);
        for (int e = 0; e < E; ++e) m.imx->cnt[(size_t) il * E + e] += (uint64_t) cnt[(size_t) e];
    }
    // the walk: the routed experts in id order; resident from their slots, streamed from the ring
    const size_t gu_half = mmq::matrix_bytes(f.gu_type, nf, nx), d_bytes = mmq::matrix_bytes(f.d_type, nx, nf);
    // THE GROUP STRIDE IS THIS LAYER'S: MMQ takes an expert's stride in blocks of its type (expert_bytes / type_size), and a
    // mixed GGUF's largest matrices (gub_max: MiMo's IQ3_XXS gate+up, 6,422,528 B) are no whole number of another type's blocks
    // (Q2_K's 84 B) - the stride truncated, every expert after a group's first read at a misaligned offset (NaN scales); the
    // group buffers are sized for the largest layer, so this layer's exact matrices always fit
    const size_t gub = 2 * gu_half, db = d_bytes;
    const size_t kend = str.start[(size_t) il + 1];
    mmq::GatherGroup gg;
    int gslots[kGroup];
    int gn = 0;   // the open group's streamed slots not yet gathered
    auto flush = [&]() {
        if (gg.n <= gg.first) return;
        if (gn > 0) cudaStreamWaitEvent(s, copied[(size_t) gslots[gn - 1]], 0);   // the copies land in order
        if (str.have)   // moex (§33): ... but a kept ring's were issued by different chunks
            for (int i = 0; i + 1 < gn; ++i) cudaStreamWaitEvent(s, copied[(size_t) gslots[i]], 0);
        if (!mmq::gather_native_group(gg, f.up_off, gu_half, f.down_off, d_bytes, mb.grp_gu, gub, mb.grp_d, db, s))
            for (int i = gg.first; i < gg.n; ++i) {
                const uint8_t* b = gg.blob[i];
                mmq::gather_native(b, b + f.up_off, gu_half, b + f.down_off, d_bytes, mb.grp_gu + (size_t) i * gub,
                                   mb.grp_d + (size_t) i * db, s);
            }
        if (gn > 0 && !str.hold) {   // one event releases the group's slots (moex #76: not while a window holds them)
            const int rel = gslots[gn - 1];
            cudaEventRecord(used[(size_t) rel], s);
            for (int i = 0; i < gn; ++i) used_of[(size_t) gslots[i]] = rel;
        }
        gn = 0;
        gg.first = gg.n;
    };
    auto products = [&](int j0, int ngx, int gi) {
        const int32_t r0 = abs_h[j0], nr = abs_h[j0 + ngx] - r0;
        int32_t maxr = 0;
        for (int i = j0; i < j0 + ngx; ++i) maxr = std::max(maxr, cnt[(size_t) order[(size_t) i]]);
        cudaMemsetAsync(mb.grp_gu + (size_t) ngx * gub, 0, kMmqTail, s);
        cudaMemsetAsync(mb.grp_d + (size_t) ngx * db, 0, kMmqTail, s);
        mmq::Product gu;
        gu.w = mb.grp_gu;
        gu.type = f.gu_type;
        gu.w_rows = 2 * nf;
        gu.w_cols = nx;
        gu.expert_bytes = gub;
        gu.n = ngx;
        gu.xq = mb.xq;
        gu.bounds = d_abs + j0;
        gu.ids = mb.ident;
        gu.total_rows = R;
        gu.max_rows = maxr;
        gu.dst = mb.gu;
        gu.ld_dst = mb.ld_gu;
        mq->run(gu, s);
        if (mb.hb) {   // STRATA_X_MOE_ALIAS=0
            swiglu_clamp_split(mb.gu + (size_t) r0 * mb.ld_gu, mb.hb + (size_t) r0 * nf, nr, nf, f.swiglu_limit, s);
            mmq::quantize(mb.hb + (size_t) r0 * nf, nullptr, mb.hq, f.d_type, nf, nf, nr, s);
        } else {
            swiglu_clamp_inplace(mb.gu + (size_t) r0 * mb.ld_gu, nr, nf, f.swiglu_limit, s, mb.ld_gu);
            mmq::quantize(mb.gu + (size_t) r0 * mb.ld_gu, nullptr, mb.hq, f.d_type, nf, mb.ld_gu, nr, s);
        }
        if (m.imx)   // the imatrix: the group's SwiGLU rows (absolute rows; read before the down product reuses gu)
            dsv41_imx_sumsq(mb.hb ? mb.hb : mb.gu, mb.hb ? nf : mb.ld_gu, nullptr, d_abs + j0, imx_order + j0, ngx, nf,
                            m.imx->dn[(size_t) il], s);
        mmq::Product dn;
        dn.w = mb.grp_d;
        dn.type = f.d_type;
        dn.w_rows = nx;
        dn.w_cols = nf;
        dn.expert_bytes = db;
        dn.n = ngx;
        dn.xq = mb.hq;
        dn.bounds = d_rel + gi * (kGroup + 1);
        dn.ids = mb.ident;
        dn.total_rows = nr;
        dn.max_rows = maxr;
        dn.dst = mb.dm + (size_t) r0 * nx;
        dn.ld_dst = nx;
        mq->run(dn, s);
        if (m.sync_check) {
            std::string e2;
            if (!sync_check("MoE group products", il, e2)) fprintf(stderr, "strata-x: %s (types %d / %d, group of %d)\n", e2.c_str(), f.gu_type, f.d_type, ngx);
        }
    };
    auto give_back = [&]() {
        if (!str.hold) str.consumed.store(str.k, std::memory_order_release);
    };
    auto release_to = [&](int32_t e_stop) {   // stream entries the routing did not pick: only their slots go back
        while (str.k < kend && str.seq[str.k].e < e_stop) {
            if (str.hold) {   // moex (#76): the window's later chunks may route them - they stay
                ++str.k;
                continue;
            }
            if (gn > 0) flush();
            const int sl = (int) (str.k % (size_t) ring_n);
            cudaEventRecord(used[(size_t) sl], s);
            used_of[(size_t) sl] = sl;
            ++str.k;
            if (gn == 0) give_back();
        }
    };
    for (int j = 0; j < nu; ++j) {
        const int32_t e = order[(size_t) j];
        release_to(e);
        const int qi = j % kGroup;
        const uint8_t* blob;
        if (str.k < kend && str.seq[str.k].e == e) {
            const int sl = (int) (str.k % (size_t) ring_n);
            while (str.issued.load(std::memory_order_acquire) <= str.k) std::this_thread::yield();
            blob = ring + (size_t) sl * ring_stride;
            gslots[gn++] = sl;
            ++str.k;
            if (!str.hold) st.streamed++;
        } else {
            const int32_t slot = m.resident[(size_t) il * E + e];
            if (slot < 0) return m.fail(err, "an expert is neither resident nor streamed");
            blob = m.slot_ptr(slot);
        }
        gg.blob[qi] = blob;
        gg.n = qi + 1;
        if (qi + 1 == kGroup || j + 1 == nu) {
            flush();
            gg = mmq::GatherGroup{};
            give_back();   // the group is gathered: its ring slots are free for the issuer
            products(j - qi, qi + 1, j / kGroup);
        }
    }
    release_to(E);
    give_back();
    if (lat) {   // the sum in the latent, then routed_up, the shared expert and the MLP's post-norm
        moe_combine_rows(mb.dm, d_slot, wts, nullptr, hml, nx, k, T, s);
        hy4_moe_out(il, T, out);
    } else {
        moe_combine_rows(mb.dm, d_slot, wts, hp.shexp ? shared : nullptr, out, n, k, T, s);
    }
    st.ms_prefill_moe += ms_since(t0);
    return true;
}

/// A chunk's MoE with no stream (EngineOptions::cpu_read, a mid-size prompt segment): the routing to the host as moe_prompt's,
/// then the routed experts split by the VRAM tier - the resident ones' rows first in the row map, walked as MMQ groups from
/// their slots on the GPU (moe_prompt's products), the others' rows after them, computed by the CPU pool at the same time: each
/// expert once over all of its tokens (jobs of up to MAXT tokens: the multi-token kernels decode a row once per job) from its
/// RAM copy - then one upload of the CPU's rows and moe_prompt's combine.  The arithmetic per routed pair is the decode
/// windows' (the GPU's MMQ or the pool's), only the grouping differs.
bool Stage::moe_prompt_cpu(int il, int T, std::string& err) {
    const Layer& l = L[(size_t) il];
    const kc::NativeFmt& f = l.fmt;
    const int n = (int) hp.n_embd, E = (int) hp.n_expert, k = (int) hp.n_used, nf = (int) hp.n_ff_exp;
    const bool lat = (bool) l.r_down;   // Hy4-flash's and ERNIE-5.5's latent MoE (moe_prompt's)
    const int nx = lat ? (int) hp.ex() : n;
    const int R = T * k;
    const auto t0 = clk::now();
    // the host's buffers for this chunk (pinned rows in, the CPU's expert rows out)
    if (h_cx_rows < T) {
        // moex (2026-11-26): the share path's last copy into h_cx (sh_ev[2]) may still be in flight - with no CPU rows nothing
        // waited for it - and the buffer is freed here
        if (sh_ev[2]) cudaEventSynchronize(sh_ev[2]);
        host_free(h_cx);
        h_cx_rows = 0;
        if (!(h_cx = static_cast<float*>(host_buf((size_t) T * n * sizeof(float))))) return m.fail(err, "CPU-read rows (out of memory)");   // (nx <= n)
        h_cx_rows = T;
    }
    if (h_cdm_rows < R) {
        host_free(h_cdm);
        h_cdm_rows = 0;
        if (!(h_cdm = static_cast<float*>(host_buf((size_t) R * n * sizeof(float))))) return m.fail(err, "CPU expert rows (out of memory)");
        h_cdm_rows = R;
    }
    gemv_dt(l.gate_inp, cur, router, n, (int) l.gate_inp.n_out(), T);   // (ERNIE-5.5: 2 E logits)
    route(il, T, d_toks, m.prompt_pen_for(il));
    if (!sync_check("MoE routing", il, err)) return false;
    cudaMemcpyAsync(h_ids, ids, (size_t) R * sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    if (lat) hy4_moe_in(il, T);
    cudaMemcpyAsync(h_cx, lat ? hxl : cur, (size_t) T * nx * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaEventRecord(ev, s);
    for (int b0 = 0; hp.shexp && b0 < T; b0 += SB) {   // the shared expert while the host waits for the routing
        const int tn = std::min(SB, T - b0);
        swiglu_ffn(il, tn, l.sh_gate, l.sh_up, l.sh_down, hp.clamp_sh[(size_t) il], cur + (size_t) b0 * n, shared + (size_t) b0 * n);
    }
    const auto tw = clk::now();
    if (cudaEventSynchronize(ev) != cudaSuccess) return m.fail(err, "the routing's readback failed");
    st.ms_prefill_wait += ms_since(tw);
    m.trace_route(1000 + il, T, k, h_ids);   // (STRATA_ROUTE_TRACE: a prompt chunk's layer, marked + 1000)
    // the counting sort, the resident experts' rows first
    std::vector<int32_t> cnt((size_t) E, 0), off((size_t) E, 0);
    for (int i = 0; i < R; ++i) {
        const int e = h_ids[i];
        if (e < 0 || e >= E) return m.fail(err, "router produced expert " + std::to_string(e));
        cnt[(size_t) e]++;
    }
    std::vector<int32_t> gpu_order, cpu_order;
    for (int e = 0; e < E; ++e)
        if (cnt[(size_t) e] > 0) (m.n_slots && m.resident[(size_t) il * E + e] >= 0 ? gpu_order : cpu_order).push_back(e);
    int32_t p = 0;
    for (int e : gpu_order) off[(size_t) e] = p, p += cnt[(size_t) e];
    const int32_t R_gpu = p;
    for (int e : cpu_order) off[(size_t) e] = p, p += cnt[(size_t) e];
    int32_t* src_h = h_grp;
    int32_t* slot_h = h_grp + R;
    {
        std::vector<int32_t> fill(off);
        for (int i = 0; i < R; ++i) {
            const int e = h_ids[i];
            const int32_t q = fill[(size_t) e]++;
            slot_h[i] = q;
            src_h[q] = i / k;
        }
    }
    const int nu = (int) gpu_order.size(), ngr = (nu + kGroup - 1) / kGroup;
    int32_t* abs_h = slot_h + R;
    int32_t* rel_h = abs_h + (nu + 1);
    for (int j = 0; j < nu; ++j) abs_h[j] = off[(size_t) gpu_order[(size_t) j]];
    abs_h[nu] = R_gpu;
    for (int gi = 0; gi < ngr; ++gi)
        for (int qi = 0; qi <= kGroup; ++qi)
            rel_h[gi * (kGroup + 1) + qi] = abs_h[std::min(nu, gi * kGroup + qi)] - abs_h[gi * kGroup];
    const size_t ints = (size_t) (2 * R + (nu + 1) + ngr * (kGroup + 1));
    cudaMemcpyAsync(mb.grp, h_grp, ints * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    const int32_t* d_src = mb.grp;
    const int32_t* d_slot = mb.grp + R;
    const int32_t* d_abs = d_slot + R;
    const int32_t* d_rel = d_abs + (nu + 1);
    // 1. the GPU's part, queued: the resident experts as MMQ groups from their slots (moe_prompt's walk without the ring)
    if (nu > 0) {
        mmq::quantize(lat ? hxl : cur, d_src, mb.xq, f.gu_type, nx, nx, R_gpu, s);
        const size_t gu_half = mmq::matrix_bytes(f.gu_type, nf, nx), d_bytes = mmq::matrix_bytes(f.d_type, nx, nf);
        const size_t gub = 2 * gu_half, db = d_bytes;
        for (int gi = 0; gi < ngr; ++gi) {
            const int j0 = gi * kGroup, ngx = std::min(kGroup, nu - j0);
            mmq::GatherGroup gg;
            for (int qi = 0; qi < ngx; ++qi) gg.blob[qi] = m.slot_ptr(m.resident[(size_t) il * E + gpu_order[(size_t) (j0 + qi)]]);
            gg.n = ngx;
            if (!mmq::gather_native_group(gg, f.up_off, gu_half, f.down_off, d_bytes, mb.grp_gu, gub, mb.grp_d, db, s))
                for (int i = 0; i < ngx; ++i) {
                    const uint8_t* b = gg.blob[i];
                    mmq::gather_native(b, b + f.up_off, gu_half, b + f.down_off, d_bytes, mb.grp_gu + (size_t) i * gub,
                                       mb.grp_d + (size_t) i * db, s);
                }
            const int32_t r0 = abs_h[j0], nr = abs_h[j0 + ngx] - r0;
            int32_t maxr = 0;
            for (int i = j0; i < j0 + ngx; ++i) maxr = std::max(maxr, cnt[(size_t) gpu_order[(size_t) i]]);
            cudaMemsetAsync(mb.grp_gu + (size_t) ngx * gub, 0, kMmqTail, s);
            cudaMemsetAsync(mb.grp_d + (size_t) ngx * db, 0, kMmqTail, s);
            mmq::Product gu;
            gu.w = mb.grp_gu;
            gu.type = f.gu_type;
            gu.w_rows = 2 * nf;
            gu.w_cols = nx;
            gu.expert_bytes = gub;
            gu.n = ngx;
            gu.xq = mb.xq;
            gu.bounds = d_abs + j0;
            gu.ids = mb.ident;
            gu.total_rows = R_gpu;
            gu.max_rows = maxr;
            gu.dst = mb.gu;
            gu.ld_dst = mb.ld_gu;
            mq->run(gu, s);
            if (mb.hb) {   // STRATA_X_MOE_ALIAS=0
                swiglu_clamp_split(mb.gu + (size_t) r0 * mb.ld_gu, mb.hb + (size_t) r0 * nf, nr, nf, f.swiglu_limit, s);
                mmq::quantize(mb.hb + (size_t) r0 * nf, nullptr, mb.hq, f.d_type, nf, nf, nr, s);
            } else {
                swiglu_clamp_inplace(mb.gu + (size_t) r0 * mb.ld_gu, nr, nf, f.swiglu_limit, s, mb.ld_gu);
                mmq::quantize(mb.gu + (size_t) r0 * mb.ld_gu, nullptr, mb.hq, f.d_type, nf, mb.ld_gu, nr, s);
            }
            mmq::Product dn;
            dn.w = mb.grp_d;
            dn.type = f.d_type;
            dn.w_rows = nx;
            dn.w_cols = nf;
            dn.expert_bytes = db;
            dn.n = ngx;
            dn.xq = mb.hq;
            dn.bounds = d_rel + gi * (kGroup + 1);
            dn.ids = mb.ident;
            dn.total_rows = nr;
            dn.max_rows = maxr;
            dn.dst = mb.dm + (size_t) r0 * nx;
            dn.ld_dst = nx;
            mq->run(dn, s);
        }
    }
    // 2. meanwhile the CPU's part: every other routed expert once over its tokens, from its RAM copy (or the file's mapping)
    if (R_gpu < R) {
        const auto tc = clk::now();
        if (cact.size() < (size_t) T * kc::kNativeActBytes) cact.resize((size_t) T * kc::kNativeActBytes);
        for (int t = 0; t < T; ++t) kc::native_quant_act(f, h_cx + (size_t) t * nx, cact.data() + (size_t) t * kc::kNativeActBytes);
        std::vector<kc::ExpertJobMulti> jobs;
        jobs.reserve(cpu_order.size() * 2);
        for (int e : cpu_order) {
            const Engine::Impl::Src sr = m.src((int64_t) il * E + e);
            const bool one_blob = sr.u == sr.g + f.up_off && sr.d == sr.g + f.down_off;
            for (int32_t q0 = 0; q0 < cnt[(size_t) e]; q0 += kc::MAXT) {
                kc::ExpertJobMulti jb;
                jb.blob = sr.g;
                jb.up = one_blob ? nullptr : sr.u;
                jb.down = one_blob ? nullptr : sr.d;
                jb.nt = std::min<int32_t>(kc::MAXT, cnt[(size_t) e] - q0);
                for (int qi = 0; qi < jb.nt; ++qi) {
                    const int32_t row = off[(size_t) e] + q0 + qi;
                    jb.nact[qi] = cact.data() + (size_t) src_h[row] * kc::kNativeActBytes;
                    jb.out[qi] = h_cdm + (size_t) (row - R_gpu) * nx;
                }
                jobs.push_back(jb);
            }
        }
        {
            std::lock_guard<std::mutex> lk(m.pool_mu);   // the stages' threads share the pool (a multi-chunk read)
            m.pool->run_split_multi_native(f, jobs.data(), (int) jobs.size());
        }
        cudaMemcpyAsync(mb.dm + (size_t) R_gpu * nx, h_cdm, (size_t) (R - R_gpu) * nx * sizeof(float), cudaMemcpyHostToDevice, s);
        st.ms_prefill_cpu += ms_since(tc);
    }
    if (lat) {
        moe_combine_rows(mb.dm, d_slot, wts, nullptr, hml, nx, k, T, s);
        hy4_moe_out(il, T, out);
    } else {
        moe_combine_rows(mb.dm, d_slot, wts, hp.shexp ? shared : nullptr, out, n, k, T, s);
    }
    // the copy above reads h_cdm, the next layer's rows overwrite it: wait for it (the GPU's part is long done by now)
    if (R_gpu < R && cudaStreamSynchronize(s) != cudaSuccess) return m.fail(err, "the CPU expert rows' upload failed");
    st.ms_prefill_moe += ms_since(t0);
    return true;
}

namespace {
/// the share's priors (Stage::moe_prompt_share) until this process has measured: a GPU's copies (bytes per ms; this box's x8
/// links carry ~6.3 / ~8.1 GB/s) and the pool on prompt jobs (blob bytes per ms per job unit; MiMo's 8.4K-token CPU read,
/// 10-08: ~0.6 ms a job of 8 tokens, a 1-token job ~0.25 ms)
constexpr double kShareLinkPrior = 6.5e6, kShareCpuPrior = 30.0e6;
/// a routed expert's pool cost in job units: jobs of up to MAXT tokens, each 1 + 0.2 per token past the first (a job decodes
/// each row once; every token adds its dot products)
double share_units(int32_t tokens) {
    const int32_t jobs = (tokens + kc::MAXT - 1) / kc::MAXT;
    return (double) jobs + 0.2 * (double) (tokens - jobs);
}
}  // namespace

/// moex: Strata 0.1.41's CPU share (src/prefill/prefill.cpp, STRATA_PREFILL_CPU_SHARE), in this engine's prompt path.  The
/// chunk's MoE with the CPU pool and the stream SHARING the routed experts the VRAM tier does not hold.  Per layer: the routing
/// to the host first (and the chunk's FFN inputs, for the pool), then the split - those experts by their token counts, the
/// fewest first, to the pool while that shortens the longer of the two sides (its estimated time against the GPU's copies of
/// the rest); the GPU's streamed through the ring by an issuer thread (only routed experts: the stream does not run ahead of
/// the routing) and walked after the resident ones as moe_prompt's MMQ groups, while the pool computes the CPU's on another
/// thread (moe_prompt_cpu's jobs); then the CPU's rows uploaded and moe_prompt's combine.  Both speeds are measured: the
/// pool's on these jobs (Engine::Impl::share_cpu_rate) and this GPU's on its copies (share_link: its link), EMAs over the
/// shared layers.  Upstream hands the pool a measured fraction of the experts of at most MAXT tokens; here an expert costs
/// the pool its jobs, so the split goes as far as the two sides balance - on this box (x8 links of ~7 GB/s a GPU, a 20-core
/// pool) the pool computes a lightly routed expert from RAM in a fraction of the time its copy takes.  A routed pair's
/// arithmetic is a decode window's: the GPU's MMQ or the pool's kernels, as the VRAM tier decides in decode - so, as
/// upstream's, the share changes which rounding a pair gets, not what is computed.
bool Stage::moe_prompt_share(int il, int T, std::string& err) {
    const Layer& l = L[(size_t) il];
    const kc::NativeFmt& f = l.fmt;
    const int n = (int) hp.n_embd, E = (int) hp.n_expert, k = (int) hp.n_used, nf = (int) hp.n_ff_exp;
    const bool lat = (bool) l.r_down;   // Hy4-flash's and ERNIE-5.5's latent MoE (moe_prompt's)
    const int nx = lat ? (int) hp.ex() : n;
    const int R = T * k;
    const auto t0 = clk::now();
    if (h_cx_rows < T) {
        // moex (2026-11-26): the share path's last copy into h_cx (sh_ev[2]) may still be in flight - with no CPU rows nothing
        // waited for it - and the buffer is freed here
        if (sh_ev[2]) cudaEventSynchronize(sh_ev[2]);
        host_free(h_cx);
        h_cx_rows = 0;
        if (!(h_cx = static_cast<float*>(host_buf((size_t) T * n * sizeof(float))))) return m.fail(err, "CPU-share rows (out of memory)");
        h_cx_rows = T;
    }
    if (!sh_ev[0])
        for (cudaEvent_t& e : sh_ev)
            if (cudaEventCreate(&e) != cudaSuccess) return m.fail(err, "the share's events");
    gemv_dt(l.gate_inp, cur, router, n, (int) l.gate_inp.n_out(), T);   // (ERNIE-5.5: 2 E logits)
    route(il, T, d_toks, m.prompt_pen_for(il));
    if (!sync_check("MoE routing", il, err)) return false;
    cudaMemcpyAsync(h_ids, ids, (size_t) R * sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    cudaEventRecord(ev, s);
    if (lat) hy4_moe_in(il, T);
    cudaMemcpyAsync(h_cx, lat ? hxl : cur, (size_t) T * nx * sizeof(float), cudaMemcpyDeviceToHost, s);   // the pool's inputs
    cudaEventRecord(sh_ev[2], s);
    for (int b0 = 0; hp.shexp && b0 < T; b0 += SB) {   // the shared expert while the host waits for the routing
        const int tn = std::min(SB, T - b0);
        swiglu_ffn(il, tn, l.sh_gate, l.sh_up, l.sh_down, hp.clamp_sh[(size_t) il], cur + (size_t) b0 * n, shared + (size_t) b0 * n);
    }
    const auto tw = clk::now();
    if (cudaEventSynchronize(ev) != cudaSuccess) return m.fail(err, "the routing's readback failed");
    st.ms_prefill_wait += ms_since(tw);
    m.trace_route(1000 + il, T, k, h_ids);   // (STRATA_ROUTE_TRACE: a prompt chunk's layer, marked + 1000)
    // the previous shared layer's copies: done (this stream waited for them before its products), so their span is final
    if (sh_pend_bytes > 0.0) {
        float ms = 0.0f;
        if (cudaEventSynchronize(sh_ev[1]) == cudaSuccess && cudaEventElapsedTime(&ms, sh_ev[0], sh_ev[1]) == cudaSuccess && ms > 0.05f) {
            const double v = sh_pend_bytes / ms;
            share_link = share_link > 0.0 ? share_link + 0.3 * (v - share_link) : v;
        }
        sh_pend_bytes = 0.0;
    }
    std::vector<int32_t> cnt((size_t) E, 0), off((size_t) E, 0);
    for (int i = 0; i < R; ++i) {
        const int e = h_ids[i];
        if (e < 0 || e >= E) return m.fail(err, "router produced expert " + std::to_string(e));
        cnt[(size_t) e]++;
    }
    // THE SPLIT: the routed experts not in VRAM, the fewest tokens first, to the pool while that shortens the longer side
    std::vector<int32_t> res_order, str_order, cpu_order;
    std::vector<std::pair<int32_t, int32_t>> cand;   // (tokens, expert)
    for (int e = 0; e < E; ++e) {
        if (cnt[(size_t) e] == 0) continue;
        if (m.n_slots && m.resident[(size_t) il * E + e] >= 0) res_order.push_back(e);
        else cand.emplace_back(cnt[(size_t) e], e);
    }
    std::sort(cand.begin(), cand.end());
    const double bytes = (double) f.bytes;
    const double link = share_link > 0.0 ? share_link : kShareLinkPrior;
    const double rate0 = m.share_cpu_rate.load(std::memory_order_relaxed), rate = rate0 > 0.0 ? rate0 : kShareCpuPrior;
    const double g = bytes / link;   // ms a streamed expert
    size_t take = 0;
    static const double fixed = [] {   // STRATA_X_SHARE_FRAC=x (A/B): a fixed fraction of them to the pool, the fewest first
        const char* e = std::getenv("STRATA_X_SHARE_FRAC");
        return e ? std::clamp(std::atof(e), 0.0, 1.0) : -1.0;
    }();
    if (fixed >= 0.0) {
        take = std::min(cand.size(), (size_t) std::llround(fixed * (double) cand.size()));
    } else {
        double C = 0.0, G = g * (double) cand.size();
        for (; take < cand.size(); ++take) {
            const double c = share_units(cand[take].first) * bytes / rate;
            if (std::max(C + c, G - g) >= std::max(C, G)) break;
            C += c;
            G -= g;
        }
    }
    double units = 0.0;
    for (size_t i = 0; i < cand.size(); ++i) {
        if (i < take) {
            cpu_order.push_back(cand[i].second);
            units += share_units(cand[i].first);
        } else {
            str_order.push_back(cand[i].second);
        }
    }
    std::sort(cpu_order.begin(), cpu_order.end());
    std::sort(str_order.begin(), str_order.end());
    // the row map: the resident experts' rows, the streamed ones', then the pool's
    int32_t p = 0;
    for (int e : res_order) off[(size_t) e] = p, p += cnt[(size_t) e];
    for (int e : str_order) off[(size_t) e] = p, p += cnt[(size_t) e];
    const int32_t R_gpu = p, R_cpu = R - R_gpu;
    for (int e : cpu_order) off[(size_t) e] = p, p += cnt[(size_t) e];
    int32_t* src_h = h_grp;
    int32_t* slot_h = h_grp + R;
    {
        std::vector<int32_t> fill(off);
        for (int i = 0; i < R; ++i) {
            const int e = h_ids[i];
            const int32_t q = fill[(size_t) e]++;
            slot_h[i] = q;
            src_h[q] = i / k;
        }
    }
    std::vector<int32_t> gpu_order(res_order);
    gpu_order.insert(gpu_order.end(), str_order.begin(), str_order.end());
    const int nres = (int) res_order.size(), nu = (int) gpu_order.size(), ngr = (nu + kGroup - 1) / kGroup;
    int32_t* abs_h = slot_h + R;
    int32_t* rel_h = abs_h + (nu + 1);
    for (int j = 0; j < nu; ++j) abs_h[j] = off[(size_t) gpu_order[(size_t) j]];
    abs_h[nu] = R_gpu;
    for (int gi = 0; gi < ngr; ++gi)
        for (int qi = 0; qi <= kGroup; ++qi)
            rel_h[gi * (kGroup + 1) + qi] = abs_h[std::min(nu, gi * kGroup + qi)] - abs_h[gi * kGroup];
    const size_t ints = (size_t) (2 * R + (nu + 1) + ngr * (kGroup + 1));
    cudaMemcpyAsync(mb.grp, h_grp, ints * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    const int32_t* d_src = mb.grp;
    const int32_t* d_slot = mb.grp + R;
    const int32_t* d_abs = d_slot + R;
    const int32_t* d_rel = d_abs + (nu + 1);
    // 1. this layer's stream: the GPU's non-resident experts in id order, copied by the issuer thread as the ring allows
    Stream str;
    for (int e : str_order) str.seq.push_back({(int32_t) il, (int32_t) e});
    if (!str.seq.empty()) cudaEventRecord(sh_ev[0], cs);
    struct Join {
        Stream& s;
        ~Join() { stop_issuer(s); }
    } join{str};
    start_issuer(*this, str);
    // 2. the pool's experts on their own thread, from their RAM copies (or the file's mapping), as moe_prompt_cpu's jobs
    double cpu_ms = 0.0;   // before cpu_fut: the thread writes it, and a future from std::async waits for it when destroyed
    std::future<bool> cpu_fut;
    if (R_cpu > 0) {
        if (h_cdm_rows < R_cpu) {
            const int64_t want = std::max<int64_t>(R_cpu, h_cdm_rows + h_cdm_rows / 2);   // grow by half: few reallocations
            host_free(h_cdm);
            h_cdm_rows = 0;
            if (!(h_cdm = static_cast<float*>(host_buf((size_t) want * n * sizeof(float)))))
                return m.fail(err, "CPU-share expert rows (out of memory)");
            h_cdm_rows = want;
        }
        cpu_fut = std::async(std::launch::async, [this, &f, &cnt, &off, &cpu_order, &cpu_ms, src_h, R_gpu, T, n = nx, il, E, bytes, units]() -> bool {
            const auto tc = clk::now();
            cudaSetDevice(dev);
            if (cudaEventSynchronize(sh_ev[2]) != cudaSuccess) return false;   // the FFN inputs are on the host
            if (cact.size() < (size_t) T * kc::kNativeActBytes) cact.resize((size_t) T * kc::kNativeActBytes);
            std::vector<char> need((size_t) T, 0);
            for (int e : cpu_order)
                for (int32_t q = 0; q < cnt[(size_t) e]; ++q) need[(size_t) src_h[off[(size_t) e] + q]] = 1;
            for (int t = 0; t < T; ++t)
                if (need[(size_t) t]) kc::native_quant_act(f, h_cx + (size_t) t * n, cact.data() + (size_t) t * kc::kNativeActBytes);
            std::vector<kc::ExpertJobMulti> jobs;
            jobs.reserve(cpu_order.size() * 2);
            for (int e : cpu_order) {
                const Engine::Impl::Src sr = m.src((int64_t) il * E + e);
                const bool one_blob = sr.u == sr.g + f.up_off && sr.d == sr.g + f.down_off;
                for (int32_t q0 = 0; q0 < cnt[(size_t) e]; q0 += kc::MAXT) {
                    kc::ExpertJobMulti jb;
                    jb.blob = sr.g;
                    jb.up = one_blob ? nullptr : sr.u;
                    jb.down = one_blob ? nullptr : sr.d;
                    jb.nt = std::min<int32_t>(kc::MAXT, cnt[(size_t) e] - q0);
                    for (int qi = 0; qi < jb.nt; ++qi) {
                        const int32_t row = off[(size_t) e] + q0 + qi;
                        jb.nact[qi] = cact.data() + (size_t) src_h[row] * kc::kNativeActBytes;
                        jb.out[qi] = h_cdm + (size_t) (row - R_gpu) * n;
                    }
                    jobs.push_back(jb);
                }
            }
            std::lock_guard<std::mutex> lk(m.pool_mu);   // the stages' threads share the pool (a multi-chunk read)
            const auto tp = clk::now();
            m.pool->run_split_multi_native(f, jobs.data(), (int) jobs.size());
            const double pool_ms = ms_since(tp);
            cpu_ms = ms_since(tc);
            if (pool_ms > 0.5) {   // the pool's speed on these jobs, for the next layers' splits (under the lock: both stages')
                const double v = units * bytes / cpu_ms, r = m.share_cpu_rate.load(std::memory_order_relaxed);
                m.share_cpu_rate.store(r > 0.0 ? r + 0.3 * (v - r) : v, std::memory_order_relaxed);
            }
            return true;
        });
    }
    // 3. the GPU's walk: the resident experts' groups first (no copy to wait for), then the streamed ones as they land
    if (nu > 0) {
        mmq::quantize(lat ? hxl : cur, d_src, mb.xq, f.gu_type, nx, nx, R_gpu, s);
        const size_t gu_half = mmq::matrix_bytes(f.gu_type, nf, nx), d_bytes = mmq::matrix_bytes(f.d_type, nx, nf);
        const size_t gub = 2 * gu_half, db = d_bytes;   // THIS layer's stride (moe_prompt's note)
        mmq::GatherGroup gg;
        int gslots[kGroup];
        int gn = 0;
        auto flush = [&]() {
            if (gg.n <= gg.first) return;
            if (gn > 0) cudaStreamWaitEvent(s, copied[(size_t) gslots[gn - 1]], 0);   // the copies land in order
            if (str.have)   // moex (§33): ... but a kept ring's were issued by different chunks
                for (int i = 0; i + 1 < gn; ++i) cudaStreamWaitEvent(s, copied[(size_t) gslots[i]], 0);
            if (!mmq::gather_native_group(gg, f.up_off, gu_half, f.down_off, d_bytes, mb.grp_gu, gub, mb.grp_d, db, s))
                for (int i = gg.first; i < gg.n; ++i) {
                    const uint8_t* b = gg.blob[i];
                    mmq::gather_native(b, b + f.up_off, gu_half, b + f.down_off, d_bytes, mb.grp_gu + (size_t) i * gub,
                                       mb.grp_d + (size_t) i * db, s);
                }
            if (gn > 0) {   // one event releases the group's slots
                const int rel = gslots[gn - 1];
                cudaEventRecord(used[(size_t) rel], s);
                for (int i = 0; i < gn; ++i) used_of[(size_t) gslots[i]] = rel;
            }
            gn = 0;
            gg.first = gg.n;
        };
        for (int j = 0; j < nu; ++j) {
            const int32_t e = gpu_order[(size_t) j];
            const int qi = j % kGroup;
            const uint8_t* blob;
            if (j >= nres) {
                const int sl = (int) (str.k % (size_t) ring_n);
                while (str.issued.load(std::memory_order_acquire) <= str.k) std::this_thread::yield();
                blob = ring + (size_t) sl * ring_stride;
                gslots[gn++] = sl;
                ++str.k;
                st.streamed++;
            } else {
                blob = m.slot_ptr(m.resident[(size_t) il * E + e]);
            }
            gg.blob[qi] = blob;
            gg.n = qi + 1;
            if (qi + 1 == kGroup || j + 1 == nu) {
                flush();
                gg = mmq::GatherGroup{};
                str.consumed.store(str.k, std::memory_order_release);   // the group is gathered: its ring slots are free
                const int j0 = j - qi, ngx = qi + 1, gi = j / kGroup;
                const int32_t r0 = abs_h[j0], nr = abs_h[j0 + ngx] - r0;
                int32_t maxr = 0;
                for (int i = j0; i < j0 + ngx; ++i) maxr = std::max(maxr, cnt[(size_t) gpu_order[(size_t) i]]);
                cudaMemsetAsync(mb.grp_gu + (size_t) ngx * gub, 0, kMmqTail, s);
                cudaMemsetAsync(mb.grp_d + (size_t) ngx * db, 0, kMmqTail, s);
                mmq::Product gu;
                gu.w = mb.grp_gu;
                gu.type = f.gu_type;
                gu.w_rows = 2 * nf;
                gu.w_cols = nx;
                gu.expert_bytes = gub;
                gu.n = ngx;
                gu.xq = mb.xq;
                gu.bounds = d_abs + j0;
                gu.ids = mb.ident;
                gu.total_rows = R_gpu;
                gu.max_rows = maxr;
                gu.dst = mb.gu;
                gu.ld_dst = mb.ld_gu;
                mq->run(gu, s);
                if (mb.hb) {   // STRATA_X_MOE_ALIAS=0
                    swiglu_clamp_split(mb.gu + (size_t) r0 * mb.ld_gu, mb.hb + (size_t) r0 * nf, nr, nf, f.swiglu_limit, s);
                    mmq::quantize(mb.hb + (size_t) r0 * nf, nullptr, mb.hq, f.d_type, nf, nf, nr, s);
                } else {
                    swiglu_clamp_inplace(mb.gu + (size_t) r0 * mb.ld_gu, nr, nf, f.swiglu_limit, s, mb.ld_gu);
                    mmq::quantize(mb.gu + (size_t) r0 * mb.ld_gu, nullptr, mb.hq, f.d_type, nf, mb.ld_gu, nr, s);
                }
                mmq::Product dn;
                dn.w = mb.grp_d;
                dn.type = f.d_type;
                dn.w_rows = nx;
                dn.w_cols = nf;
                dn.expert_bytes = db;
                dn.n = ngx;
                dn.xq = mb.hq;
                dn.bounds = d_rel + gi * (kGroup + 1);
                dn.ids = mb.ident;
                dn.total_rows = nr;
                dn.max_rows = maxr;
                dn.dst = mb.dm + (size_t) r0 * nx;
                dn.ld_dst = nx;
                mq->run(dn, s);
                if (m.sync_check) {
                    std::string e2;
                    if (!sync_check("MoE group products (shared)", il, e2)) fprintf(stderr, "strata-x: %s (types %d / %d, group of %d)\n", e2.c_str(), f.gu_type, f.d_type, ngx);
                }
            }
        }
    }
    stop_issuer(str);   // every entry was issued (the walk consumed them all)
    if (!str.seq.empty()) {
        cudaEventRecord(sh_ev[1], cs);
        sh_pend_bytes = bytes * (double) str.seq.size();
    }
    st.prefill_share_experts += (int64_t) cand.size();
    if (cpu_fut.valid()) {   // 4. the pool's rows after the GPU's, then the combine
        if (!cpu_fut.get()) return m.fail(err, "the CPU share's inputs did not arrive");
        cudaMemcpyAsync(mb.dm + (size_t) R_gpu * nx, h_cdm, (size_t) R_cpu * nx * sizeof(float), cudaMemcpyHostToDevice, s);
        st.ms_prefill_cpu += cpu_ms;
        st.prefill_cpu_experts += (int64_t) cpu_order.size();
    }
    if (lat) {
        moe_combine_rows(mb.dm, d_slot, wts, nullptr, hml, nx, k, T, s);
        hy4_moe_out(il, T, out);
    } else {
        moe_combine_rows(mb.dm, d_slot, wts, hp.shexp ? shared : nullptr, out, n, k, T, s);
    }
    // the upload above reads h_cdm and the next layer's pool writes it: wait for it
    if (R_cpu > 0 && cudaStreamSynchronize(s) != cudaSuccess) return m.fail(err, "the CPU share's rows' upload failed");
    st.ms_prefill_moe += ms_since(t0);
    return true;
}

bool Stage::moe_chunk(int il, int T, Stream& str, std::string& err) {
    if (cpu_moe) return moe_prompt_cpu(il, T, err);
    if (share) return moe_prompt_share(il, T, err);
    return moe_prompt(il, T, str, err);
}

// ------------------------------------------------------------------ a chunk through the stage's layers

bool Stage::sync_check(const char* what, int il, std::string& err) {
    if (!m.sync_check) return true;
    cudaError_t e = cudaStreamSynchronize(s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(cs);   // the copy stream too (the prompt path's streamed experts)
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    return m.fail(err, std::string("sync check: ") + what + ", layer " + std::to_string(il) + ": " + cudaGetErrorString(e));
}

bool Stage::nan_check(const char* what, int il, const float* d, size_t n, std::string& err) {
    static const bool on = std::getenv("STRATA_X_SYNC_CHECK") && std::getenv("STRATA_X_SYNC_CHECK")[0] == '2';
    if (!on) return true;
    if (!sync_check(what, il, err)) return false;
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(h[i]))
            return m.fail(err, std::string("nan check: ") + what + ", layer " + std::to_string(il) + ": element " + std::to_string(i) +
                                   " of " + std::to_string(n) + " is " + (std::isnan(h[i]) ? "NaN" : "Inf"));
    return true;
}

bool Stage::prompt_layer(int il, int T, int64_t pos0, Stream& str, std::string& err, int half) {
    const int n = (int) hp.n_embd;
    img_pos0 = pos0;   // (moex, #113: route()'s and the engram's rows by position)
    // moex (DSV41_PLAN §59 P7): half 1 runs the attention half alone, half 2 the FFN half alone (a layer-major window's two
    // passes: what crosses them is the chunk's own R and carried pre-mix, LmRows), 0 both
    const size_t hcn = (size_t) hp.hc * n;
    if (half != 2) {
        if (m.dsp && last) {   // DSpark: the features of the chunk's last rows (the drafter attends a 128-position window)
            const int tail = std::min(T, dsp_rows);
            dspark_capture(il, T - tail, tail);
        }
        if (m.dfl) {   // DFlash: the chunk's last rows' residual at a target layer's input (any stage; its window + the block)
            const int tail = std::min(T, m.dfl->cap);
            dflash_capture(il, T - tail, tail, true);
        }
        // the attention half, sub-batch by sub-batch (the KDA state, the conv state and the DSA caches carry over)
        for (int b0 = 0; b0 < T; b0 += SB) {
            const int tn = std::min(SB, T - b0);
            float* Rr = R + (size_t) b0 * hcn;
            float* cr = cur + (size_t) b0 * n;
            float* orr = out + (size_t) b0 * n;
            float* co = coef + (size_t) b0 * 24;
            if (m.sync_check && !m.dense_hash.empty() && b0 == 0 && !m.verify_dense(il, "before the layer runs", err)) return false;
            // DeepSeek-V4.1's engram (the chunk's rows; their fetch may still run beside the stream).  An image span's rows get none:
            // model.py shuts the gate there (engram_mask), which leaves the streams as they are
            if (m.eg_index(il) >= 0 && !image_batch) {
                if (str.eg.valid()) {
                    const auto tw = clk::now();
                    const std::string e2 = m.eg_await(str.eg);
                    m.us_eg_wait.fetch_add((int64_t) (ms_since(tw) * 1000.0), std::memory_order_relaxed);
                    if (!e2.empty()) return m.fail(err, e2);
                    // moex (DSV41_BACKLOG #51): this chunk's rows are in - the next chunk's are fetched from here on, beside this
                    // chunk's layers and the stream, into the second buffer (their positions are noted first: none below pos0 + T)
                    if (eg_ahead() && eg_next_T > 0 && eg_next_toks && !eg_nx.valid()) {
                        if (!eg_nx_grow(eg_next_T, err)) return false;
                        m.eg_note(eg_next_toks, eg_next_pos, eg_next_T);
                        eg_nx_pos = eg_next_pos;
                        eg_nx_T = eg_next_T;
                        // (moex #147: after this chunk's later layers' rows, if those are still being read - one fetch at a time)
                        eg_nx = std::async(std::launch::async,
                                           [this, p = eg_next_pos, t = eg_next_T, h = eg_nx_h, cap = eg_nx_cap, prev = str.eg2] {
                                               if (prev.valid()) prev.wait();
                                               std::string e3;
                                               return m.eg_rows(p, t, h, cap, 0, e3) ? std::string()
                                                                                    : (e3.empty() ? std::string("engram rows") : e3);
                                           });
                    }
                }
                if (m.eg_index(il) >= 1 && str.eg2.valid()) {   // moex (#147): a later engram layer - its rows are in first
                    const auto tw = clk::now();
                    const std::string e2 = m.eg_await(str.eg2);
                    m.us_eg_wait.fetch_add((int64_t) (ms_since(tw) * 1000.0), std::memory_order_relaxed);
                    str.eg2 = std::shared_future<std::string>();
                    if (!e2.empty()) return m.fail(err, e2);
                }
                if (!img_mix) engram(il, b0, tn, eg_pr_d, eg_pr_cap);
                else   // moex (#113): the text rows' runs only - a picture's rows get none (the shut gate)
                    for (int r0 = b0; r0 < b0 + tn;) {
                        const bool pic = m.image_row(pos0 + r0) != nullptr;
                        int r1 = r0 + 1;
                        while (r1 < b0 + tn && (m.image_row(pos0 + r1) != nullptr) == pic) ++r1;
                        if (!pic) engram(il, r0, r1 - r0, eg_pr_d, eg_pr_cap);
                        r0 = r1;
                    }
            }
            if (const int j = m.hy4_eg_index(il); j >= 0 && m.heg)   // Hy4-flash's Engram (hy4_eg_chunk uploaded the rows)
                hy4_engram(il, j - hy_egp_j0, b0, tn, hy_egp_rows, hy_egp_valid, hy_egp_cap);   // (this stage's slots alone)
            mix_in(il, tn, Rr, cr, co, true);
            if (!sync_check("attention norm", il, err) || !nan_check("attention norm", il, cr, (size_t) tn * n, err)) return false;
            attention(il, pos0 + b0, tn, cr, orr, false);
            if (!sync_check("attention", il, err)) return false;
            if (hp.arch == Arch::mimo2 && !nan_check("attention qkv", il, qkv, (size_t) tn * hp.qkv_width(il), err)) return false;
            if (hp.arch == Arch::mimo2 && !nan_check("attention heads", il, att, (size_t) tn * hp.n_head * hp.hd_v, err)) return false;
            if (!nan_check("attention out", il, orr, (size_t) tn * n, err)) return false;
            mix_out(il, tn, orr, Rr, co, true);
        }
    }
    if (half == 1) return true;
    if (m.last_ring_only(il)) return true;   // moex (§59 P1): the last layer's ring rows are in; its output is never read
    // the FFN half: the reads in sub-batches, the routed experts over the whole chunk
    for (int b0 = 0; b0 < T; b0 += SB) {
        const int tn = std::min(SB, T - b0);
        mix_in(il, tn, R + (size_t) b0 * hcn, cur + (size_t) b0 * n, coef + (size_t) b0 * 24, false);
    }
    if (!sync_check("FFN norm", il, err) || !nan_check("FFN norm", il, cur, (size_t) T * n, err)) return false;
    if (hp.is_moe(il)) {
        if (!moe_chunk(il, T, str, err)) return false;
        if (const int64_t k = read_failed.exchange(0); k > 0)   // moex (DSV41_BACKLOG #47): never a chunk without an expert
            return m.fail(err, "an O_DIRECT read of layer " + std::to_string((k - 1) / hp.n_expert) + " expert " +
                                   std::to_string((k - 1) % hp.n_expert) +
                                   " failed twice: the prompt is not read with that expert missing");
        if (!sync_check("MoE", il, err) || !nan_check("MoE out", il, out, (size_t) T * n, err)) return false;
    } else {
        const Layer& l = L[(size_t) il];
        for (int b0 = 0; b0 < T; b0 += SB) {
            const int tn = std::min(SB, T - b0);
            swiglu_ffn(il, tn, l.ffn_gate, l.ffn_up, l.ffn_down, hp.clamp_sh[(size_t) il], cur + (size_t) b0 * n,
                       out + (size_t) b0 * n);
        }
        if (l.ffn_post_norm) rms_norm(out, l.ffn_post_norm.f(), out, T, n, hp.rms_eps, s);   // Hy4-flash's sandwich norm
    }
    mix_out(il, T, out, R, coef, false);
    return true;
}

bool Stage::prompt_chunk(const int32_t* toks, const float* hand_in, int T, int64_t pos0, const int32_t* next_toks,
                         int n_next, std::string& err) {
    cudaSetDevice(dev);
    use(pr);
    chunk_stopped = false;
    const int n = (int) hp.n_embd, E = (int) hp.n_expert;
    bool eg_now = false;   // DeepSeek-V4.1: the chunk's engram rows are to be fetched beside the stream
    bool eg_took = false;  // ... or were, by the previous chunk (STRATA_DSV41_EG_AHEAD): its fetch becomes this chunk's
    try {
        if (toks) {
            if (!upload_embd(toks, T, cur, pos0)) return m.fail(err, "token out of range");
            if (hp.arch == Arch::dsv4 && !upload_toks(toks, T, err)) return false;   // the hash layers' ids (this stage's)
            for (int hh = 0; hh < (int) hp.hc; ++hh)   // the residual streams (mHC's four, MiMo's one) start as the embedding
                cudaMemcpy2DAsync(R + (size_t) hh * n, (size_t) hp.hc * n * sizeof(float), cur, (size_t) n * sizeof(float),
                                  (size_t) n * sizeof(float), (size_t) T, cudaMemcpyDeviceToDevice, s);
            if (hp.v41) v41_pre_reset(T);   // DeepSeek-V4.1: layer 0's attention reads stream 0
            if (m.eg) {   // DeepSeek-V4.1's engram: the chunk's rows (this stage holds the engram layers)
                // moex (DSV41_BACKLOG #51): rows fetched ahead for exactly this chunk's start (and at least its length) are
                // taken over with their buffer; a guess for another chunk is waited out and left unused
                if (eg_nx_pos >= 0) {
                    if (eg_nx_pos == pos0 && T <= eg_nx_T && !image_batch) {
                        cudaStreamSynchronize(s);   // the previous chunk's kernels may still read the buffer given back
                        std::swap(eg_pr_h, eg_nx_h);
                        std::swap(eg_pr_d, eg_nx_d);
                        std::swap(eg_pr_cap, eg_nx_cap);
                        eg_took = true;
                    } else if (eg_nx.valid()) {
                        eg_nx.get();
                    }
                    eg_nx_pos = -1;
                }
                if (!eg_took && !eg_pr_grow(T, err)) return false;
                cudaStreamSynchronize(s);   // the previous chunk's kernels may still read the buffer
                m.eg_note(toks, pos0, T);   // an image span's positions: DEAD (eg_note)
                eg_now = !image_batch && !eg_took;   // fetched below, beside the stream (STRATA_DSV41_EG_SYNC=1: here, before
                                                     // it); an image span has none to fetch (its rows get no engram)
                static const bool eg_sync = std::getenv("STRATA_DSV41_EG_SYNC") && std::getenv("STRATA_DSV41_EG_SYNC")[0] == '1';
                if (eg_sync && eg_now) {
                    eg_now = false;
                    if (!m.eg_rows(pos0, T, eg_pr_h, eg_pr_cap, 0, err)) return false;
                }
            }
        } else {
            cudaMemcpyAsync(R, hand_in, (size_t) T * hp.hc * n * sizeof(float), cudaMemcpyHostToDevice, s);
            if (hp.v41)   // ... and a later stage's the previous stage's last FFN pre-mix, after the rows
                cudaMemcpyAsync(cpre, hand_in + (size_t) T * hp.hc * n, (size_t) T * 24 * sizeof(float), cudaMemcpyHostToDevice, s);
        }
        // the MTP block(s)' caches along the chunk (DSpark injects its rows below instead)
        const bool nextn = last && m.mtp_on && !m.dsp && !m.dfl && !m.dfy && n_next > 0 && next_toks != nullptr;   // (DFly2: none)
        // this chunk's stream: every non-resident expert of the stage's MoE layers, layer by layer in id order
        Stream str;
        str.start.assign((size_t) m.n_load + 1, 0);
        auto add_layer = [&](int64_t il) {
            str.start[(size_t) il] = str.seq.size();
            for (int32_t e = 0; e < E; ++e)
                if (m.resident[(size_t) il * E + e] < 0) str.seq.push_back({(int32_t) il, e});
            str.start[(size_t) il + 1] = str.seq.size();
        };
        // (the CPU read and the share stream nothing ahead of the routing: moe_chunk; the CPU read has no ring at all)
        const bool ahead = !cpu_moe && !share;
        bool first_ahead_on = false;   // (§59 P15: Stream::first_l once the stream's layers are in)
        if (ahead && m.rio && T <= m.stream_gate()) {   // DeepSeek-V4.1's SSD tier: a short chunk reads its routed experts only
            str.gate = true;
            str.routed.assign((size_t) m.n_load * E, 0);
            str.spec = gate_ahead() > 0 && T >= gate_ahead();
            static const int64_t first_ahead = [] {   // (§59 P15)
                const char* e = std::getenv("STRATA_DSV41_FIRST_AHEAD");
                const int64_t v = e ? std::atoll(e) : 0;
                return v == 1 ? (int64_t) 2048 : v;
            }();
            first_ahead_on = first_ahead > 0 && T >= first_ahead;
        }
        // bounded replay: the rows [r0, T) the reads after this chunk need run the layers from replay_layer on; the others
        // stop before it (a chunk with none of them streams none of those layers' experts)
        int r0 = 0;
        int64_t tail_at = -1;   // (moex, §30) the read start this chunk's tail rows end at
        if (m.replay_layer >= 0 && m.replay_layer < le && !m.replay_ends.empty()) {
            r0 = T;
            const auto it = std::upper_bound(m.replay_ends.begin(), m.replay_ends.end(), pos0);   // the first read start after pos0
            if (it != m.replay_ends.end()) {
                r0 = (int) std::clamp<int64_t>(*it - m.replay_tail - pos0, 0, T);
                tail_at = *it;
            }
        }
        // (moex, §30: a deferred tail streams those layers in the tails' run, not here)
        const bool defer = tail_batch && m.replay_layer >= 0 && m.replay_layer < le && r0 < T && tail_at >= 0;
        // moex (#108): nor a layer whose cone holds none of this chunk's rows (pos0 + T at or before cut - 128 (40 - il), and so
        // every later one) - the layer loop stops there, and its entries left in the stream held the issuer at a full ring and
        // the chunk's last wait below forever (a call of 8,192 + 2,049-2,559 tokens with one read start: the audit of 2026-10-24)
        auto cone_empty = [&](int64_t il) {
            return replay_cone() && m.replay_layer >= 0 && il >= m.replay_layer && tail_at >= 0 && cone_rows(il, pos0, T, tail_at) <= 0;
        };
        for (int64_t il = lb; il < le; ++il)   // only the stage's MoE layers' (and the NextN's) starts are read
            if (hp.is_moe(il) && ahead && !(m.replay_layer >= 0 && il >= m.replay_layer && (r0 >= T || defer)) && !cone_empty(il) &&
                !m.last_ring_only(il))
                add_layer(il);
        if (nextn && hp.arch == Arch::glm5 && ahead) add_layer(hp.n_layer);
        if (first_ahead_on && !str.seq.empty()) str.first_l = str.seq.front().l;   // (§59 P15)
        // moex (DSV41_PLAN §33), STRATA_DSV41_RING_KEEP=1: a call's chunks that stream one layer, the tails deferred - the
        // records stay in the ring for the call's later chunks, each of which reads only the routed ones it lacks (the slot of
        // an entry is its index: the ring holds the whole layer)
        static const bool keep_env = std::getenv("STRATA_DSV41_RING_KEEP") && std::getenv("STRATA_DSV41_RING_KEEP")[0] == '1';
        const bool one_layer = keep_env && keep_stride > 0 && tail_batch && last && m.rio && ahead && !str.seq.empty() &&
                               str.seq.size() <= (size_t) ring_n && str.seq.front().l == str.seq.back().l;
        size_t had = 0;
        if (one_layer) {
            if (ring_kept_l != str.seq.front().l || ring_have.size() != str.seq.size()) {
                ring_have.assign(str.seq.size(), 0);
                ring_kept_l = str.seq.front().l;
            }
            had = (size_t) std::count(ring_have.begin(), ring_have.end(), (uint8_t) 1);
            str.hold = true;
            str.have = &ring_have;
        }
        struct Join {
            Stream& s;
            ~Join() { stop_issuer(s); }
        } join{str};
        hb_total = str.seq.size();
        hb_issued = 0;
        start_issuer(*this, str);
        if (eg_took) str.eg = std::move(eg_nx);   // moex (#51): the fetch begun by the previous chunk (layer 1 waits for it)
        if (eg_now && m.eg_split_ok(T)) {
            // moex (DSV41_BACKLOG #147, DSV41_PLAN §76): the first engram layer's rows, then the others' - layer 1 waits for
            // its own alone (the whole fetch was 1.3 s at 2K tokens, 2.5 s at 8K, half of it rows layer 14 reads 13 layers on)
            auto p1 = std::make_shared<std::promise<std::string>>(), p2 = std::make_shared<std::promise<std::string>>();
            str.eg = p1->get_future();
            str.eg2 = p2->get_future().share();
            str.eg_job = std::async(std::launch::async, [this, pos0, T, p1, p2] {
                std::string e1, e2;
                const bool ok1 = m.eg_rows(pos0, T, eg_pr_h, eg_pr_cap, 0, e1, 0, 1);
                p1->set_value(ok1 ? std::string() : (e1.empty() ? std::string("engram rows") : e1));
                const bool ok2 = ok1 && m.eg_rows(pos0, T, eg_pr_h, eg_pr_cap, 0, e2, 1, -1);
                p2->set_value(ok2 ? std::string() : (e2.empty() ? std::string("engram rows") : e2));
            });
        } else if (eg_now)   // the engram rows' page reads (1.3 s at 2K tokens, 2.5 s at 8K) beside the stream and layer 0, not before them
            str.eg = std::async(std::launch::async, [this, pos0, T] {
                std::string e2;
                return m.eg_rows(pos0, T, eg_pr_h, eg_pr_cap, 0, e2) ? std::string() : (e2.empty() ? std::string("engram rows") : e2);
            });
        int Tl = T;              // the rows the layers from here on run over (bounded replay: the chunk's tail)
        int64_t pl = pos0;
        // moex (DSV41_BACKLOG #78): a stop is looked for before each layer, and the running state as the chunk finds it is kept
        // for it (Impl::prompt puts it back once the chain is drained, if the last stage did not get past this chunk's start)
        const bool can_stop = m.stop_layers && !run_snaps.empty();
        if (can_stop) run_snap_take(pos0);
        int64_t stopped_at = -1;   // the layer a stop came before
        for (int64_t il = lb; il < le; ++il) {
            if (can_stop && m.stop_asked()) {
                stopped_at = il;
                if (il > lb) run_pos = -1;   // layers ran: the state is inside the chunk
                break;
            }
            if (m.replay_layer >= 0 && il == m.replay_layer && r0 > 0) {
                if (r0 >= T) {   // none of this chunk's rows is read later: the replayed layers skip it
                    m.pl_add(le - il);
                    break;
                }
                replay_shift(T, T - r0);
                Tl = T - r0;
                pl = pos0 + r0;
            }
            if (defer && il == m.replay_layer) {   // moex (§30): the tail's rows wait for the tails' run (its layers' units too)
                if (!tail_keep(Tl, pl, tail_at, pos0 + T == tail_at, err)) return false;
                break;
            }
            if (replay_cone() && m.replay_layer >= 0 && il >= m.replay_layer && tail_at >= 0) {   // moex (#108)
                const int need = cone_rows(il, pl, Tl, tail_at);
                if (need <= 0) {   // none of the rows left is read at this layer or later: the rest of the layers skip them
                    m.pl_add(le - il);
                    break;
                }
                if (need < Tl) {
                    replay_shift(Tl, need);
                    pl += Tl - need;
                    Tl = need;
                }
            }
            if (hp.is_moe(il)) str.k = str.start[(size_t) il];
            hb_il = (int) il;
            if (!prompt_layer((int) il, Tl, pl, str, err)) return false;
            m.pl_add(1);
        }
        if (str.have && stopped_at < 0) {   // moex (§33): the issuer done with the chunk's entries before the next chunk's starts
            while (str.issued.load(std::memory_order_acquire) < str.seq.size()) std::this_thread::yield();
            stop_issuer(str);
            st.streamed += (int64_t) ((size_t) std::count(ring_have.begin(), ring_have.end(), (uint8_t) 1) - had);
        }
        if (stopped_at >= 0) {   // the rest of the chunk does not run: its stream stops (the reads in flight land first)
            stop_issuer(str);
            chunk_stopped = true;
            fprintf(stderr, "strata-x: a stop before layer %lld of the chunk at %lld (GPU %d)%s\n", (long long) stopped_at,
                    (long long) pos0, dev, run_pos < 0 ? ": its running state goes back" : "");
            const cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) return m.fail(err, std::string("prompt chunk: ") + cudaGetErrorString(ce));
            return true;
        }
        if (m.dsp && last && !(m.replay_layer >= 0 && r0 >= T)) {   // DSpark: the trunk's output too, then the chunk's last
            const int tail = std::min(Tl, dsp_rows);                  // rows into the drafter's rings
            dspark_capture((int) hp.n_layer, Tl - tail, tail);
            if (!dspark_inject(tail, pl + Tl - tail, err)) return false;
        }
        if (m.dfl && last) {   // DFlash: the same (every chunk's: a checkpoint or a short next chunk may read from here)
            const int tail = std::min(T, m.dfl->cap);
            dflash_capture((int) hp.n_layer, T - tail, tail, true);
            if (!dflash_inject(tail, pos0 + T - tail, true, err)) return false;
        }
        if (nextn && hp.arch == Arch::dsv4) {   // DeepSeek-V4's NextN head: the rows its first decode window reads
            if (!prompt_mtp_dsv4(T, n_next, pos0, next_toks, err)) return false;
        } else if (nextn && hp.arch == Arch::mimo2) {
            if (!prompt_mtp_mimo(T, n_next, pos0, next_toks, err)) return false;
        } else if (nextn) {
            str.k = str.start[(size_t) hp.n_layer];
            if (!prompt_nextn(T, n_next, pos0, next_toks, str, err)) return false;
        }
        // the issuer is done once every entry was consumed; wait for it before the stream's state goes away
        while (str.issuer.joinable() && str.issued.load(std::memory_order_acquire) < str.seq.size()) std::this_thread::yield();
        stop_issuer(str);
    } catch (const std::exception& e) {
        return m.fail(err, e.what());
    }
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return m.fail(err, std::string("prompt chunk: ") + cudaGetErrorString(ce));
    run_pos = pos0 + T;
    st.tokens_prefill += T;
    return true;
}

// ------------------------------------------------------------------ moex (DSV41_BACKLOG #108): the replay's dependency cone

/// opt-in STRATA_DSV41_REPLAY_CONE=1: each replayed layer over the rows the reads after it still need, not all replay_tail of them.
/// A read start at `cut` needs each layer's KV ring there (its last 128 positions), and layer L's rows there need layer L - 1's
/// 127 positions before them: layer L runs [cut - 128 (n_layer - L), cut) - 2,432 rows at layer 21, 128 at layer 39, half the
/// replay's 19 x 2,560.  Off with a drafter (DSpark, NextN, DFlash read more of the trunk's last rows)
bool Stage::replay_cone() const {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_DSV41_REPLAY_CONE");
        return e && e[0] == '1';
    }();
    return on && hp.v41 && !m.dsp && !m.dfl && !m.mtp_on;
}

/// rows [pos, pos + T) going into layer il, serving the read start `cut` (the first after them): how many of the last ones it needs
int Stage::cone_rows(int64_t il, int64_t pos, int T, int64_t cut) const {
    const int64_t from = cut - (int64_t) 128 * (hp.n_layer - il);
    return (int) std::clamp<int64_t>(pos + T - from, 0, T);
}

// ------------------------------------------------------------------ moex (DSV41_PLAN §30): the deferred tails

bool Stage::tail_keep(int T, int64_t pos, int64_t cut, bool at_cut, std::string& err) {
    const int tr = (int) m.replay_tail;
    int snaps_used = 0;
    for (const TailRows& t : tails) snaps_used += t.snap != nullptr;
    if ((int) tails.size() >= tail_cap || T > tr || T <= 0 || (at_cut && snaps_used >= tail_snap_cap))
        return m.fail(err, "a deferred tail the loan has no room for");
    const size_t full = (run_state_bytes() + 255) & ~(size_t) 255;
    Lay c{loan_base + tail_chunk_bytes + (size_t) tail_snap_cap * full + tail_rows_used};
    TailRows t;
    const size_t hcn = (size_t) hp.hc * hp.n_embd, cb = lm_cand_row(m);
    t.R = c.take<float>((size_t) T * hcn);
    t.cpre = c.take<float>((size_t) T * 24);
    t.dsel = c.take<int32_t>((size_t) T * hp.idx_topk);
    t.dcand = c.take<uint8_t>((size_t) T * cb);
    tail_rows_used += (c.off + 255) & ~(size_t) 255;
    cudaMemcpyAsync(t.R, R, (size_t) T * hcn * sizeof(float), cudaMemcpyDeviceToDevice, s);
    if (cpre) cudaMemcpyAsync(t.cpre, cpre, (size_t) T * 24 * sizeof(float), cudaMemcpyDeviceToDevice, s);
    else t.cpre = nullptr;
    if (dsel) cudaMemcpyAsync(t.dsel, dsel, (size_t) T * hp.idx_topk * sizeof(int32_t), cudaMemcpyDeviceToDevice, s);
    else t.dsel = nullptr;
    if (dcand) cudaMemcpyAsync(t.dcand, dcand, (size_t) T * cb, cudaMemcpyDeviceToDevice, s);
    else t.dcand = nullptr;
    t.T = T;
    t.pos = pos;
    t.cut = cut;
    if (at_cut) {   // the chunk ends at the cut: the running state there, its replayed layers completed by the tails' run
        RunSnap& r = tail_snaps[(size_t) snaps_used];
        std::vector<std::pair<void*, size_t>> reg;
        state_regions(reg);
        size_t off = 0;
        for (const auto& x : reg) {
            cudaMemcpyAsync(r.d + off, x.first, x.second, cudaMemcpyDeviceToDevice, s);
            off += x.second;
        }
        r.pos = cut;
        t.snap = &r;
    }
    tails.push_back(t);
    return true;
}

/// The call's deferred tails through the layers from replay_layer on, once its chunks are done: each layer's non-resident
/// experts stream once into a ring that holds the whole layer (the chunk part of the loan carved again for the longest
/// piece, with the chunks' own sub-batch), and every piece runs the layer from it in position order (Stream::hold until the
/// last) as its chunk would have run it - so the layers' rings end as the call's last tail leaves them, and a cut's state copy
/// takes each layer as the cut's last piece finishes it.  A stop is looked for before each (layer, piece): the call's
/// checkpoints at its cuts then lack this stage (not kept) and the state is exact nowhere (Impl::prompt).  past_stop: a call
/// stopped among its chunks runs the tails it has to the end, so the checkpoints at the cuts it passed are kept
bool Stage::prompt_tails(std::string& err, bool past_stop) {
    if (tails.empty()) return true;
    cudaSetDevice(dev);
    chunk_stopped = false;
    const int E = (int) hp.n_expert;
    int TM = 0;
    for (const TailRows& t : tails) TM = std::max(TM, t.T);
    const int sub = std::min(call_sub, TM);
    try {
        // the chunks' copies into the ring (a stopped chunk's may still be queued) land before the loan is carved again
        if (cudaStreamSynchronize(cs) != cudaSuccess) return m.fail(err, "the ring's copies failed before the deferred tails");
        uint8_t* base = loan_base;
        carve(pr, base, TM, sub);
        Lay c{base, scratch_bytes(m, TM, sub, cand_rows())};
        const int64_t n = hp.n_embd, nf = hp.n_ff_exp, R_ = (int64_t) TM * hp.n_used;
        const int64_t max_in = max_proj_in(hp);
        mb.rows = R_;
        mb.xq = c.take<uint8_t>(std::max(mmq::q8_bytes(R_, n), mmq::q8_bytes(sub, max_in)));
        mb.hq = c.take<uint8_t>(mmq::q8_bytes(R_, nf));
        mb.ld_gu = gu_ld(hp);
        mb.gu = c.take<float>((size_t) (R_ * mb.ld_gu));
        mb.hb = moe_alias() ? nullptr : c.take<float>((size_t) (R_ * nf));
        mb.dm = n <= mb.ld_gu && moe_alias() ? mb.gu : c.take<float>((size_t) (R_ * n));
        mb.grp_gu = c.take<uint8_t>(kGroup * m.gub_max + kMmqTail);
        mb.grp_d = c.take<uint8_t>(kGroup * m.db_max + kMmqTail);
        const int64_t n_ident = std::max<int64_t>(R_, max_in);
        mb.ident = c.take<int32_t>((size_t) n_ident + 512);   // (moex, Maya #645: MMQ's read past the last row)
        mb.grp = c.take<int32_t>((size_t) (2 * R_ + (E + 1) + ((E + kGroup - 1) / kGroup) * (kGroup + 1)));
        ring_stride = m.ring_blob(this->di);
        ring_n = E;
        ring = c.take<uint8_t>((size_t) ring_n * ring_stride);
        if (c.off > tail_chunk_bytes) return m.fail(err, "the tails' run does not fit the loan's chunk part");
        mmq::iota(mb.ident, n_ident + 512, s);
        use(pr);
        Stream str;
        str.start.assign((size_t) m.n_load + 1, 0);
        for (int64_t il = m.replay_layer; il < le; ++il) {
            if (!hp.is_moe(il) || m.last_ring_only(il)) continue;
            str.start[(size_t) il] = str.seq.size();
            for (int32_t e = 0; e < E; ++e)
                if (m.resident[(size_t) il * E + e] < 0) str.seq.push_back({(int32_t) il, e});
            str.start[(size_t) il + 1] = str.seq.size();
        }
        struct Join {
            Stream& s;
            ~Join() { stop_issuer(s); }
        } join{str};
        hb_total = str.seq.size();
        hb_issued = 0;
        start_issuer(*this, str);
        bool stopped = false;
        for (int64_t il = m.replay_layer; il < le && !stopped; ++il) {
            for (size_t ti = 0; ti < tails.size(); ++ti) {
                if (!past_stop && m.stop_asked()) {
                    stopped = true;
                    break;
                }
                TailRows& t = tails[ti];
                R = t.R;
                cpre = t.cpre;
                dsel = t.dsel;
                dcand = t.dcand;
                if (replay_cone() && t.T > 0) {   // moex (#108): the piece's rows this layer and the later ones still read
                    const int need = cone_rows(il, t.pos, t.T, t.cut);
                    if (need < t.T) {
                        if (need > 0) replay_shift(t.T, need);
                        t.pos += t.T - std::max(need, 0);
                        t.T = std::max(need, 0);
                    }
                }
                if (t.T <= 0) {   // (a piece before its cut that no layer from here on reads: its units counted)
                    m.pl_add(1);
                    continue;
                }
                if (hp.is_moe(il)) {
                    str.k = str.start[(size_t) il];
                    str.hold = ti + 1 < tails.size();
                }
                hb_il = (int) il;
                hb_c = (int) ti;
                if (!prompt_layer((int) il, t.T, t.pos, str, err)) return false;
                m.pl_add(1);
                if (t.snap) run_snap_layer(il, *t.snap);
            }
        }
        if (stopped) {
            stop_issuer(str);
            chunk_stopped = true;
            fprintf(stderr, "strata-x: a stop in the deferred tails' run (GPU %d): the call's checkpoints at its cuts and its state "
                            "go\n", dev);
        } else {
            while (str.issuer.joinable() && str.issued.load(std::memory_order_acquire) < str.seq.size()) std::this_thread::yield();
            stop_issuer(str);
        }
    } catch (const std::exception& e) {
        use(pr);
        return m.fail(err, e.what());
    }
    use(pr);
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return m.fail(err, std::string("the deferred tails: ") + cudaGetErrorString(ce));
    return true;
}

// ------------------------------------------------------------------ moex (DSV41_BACKLOG #76): a layer-major window

void Stage::lm_rows(int c) {
    const LmRows& w = lmw[(size_t) c];
    R = w.R;
    cpre = w.cpre;
    dsel = w.dsel;
    dcand = w.dcand;
}

bool Stage::eg_lm_grow(int k, int rows, std::string& err) {
    (void) k;   // (two buffers, each a whole window's rows)
    if (eg_lm_h[0] && eg_lm_cap >= rows) return true;
    eg_lm_drop();
    cudaSetDevice(dev);
    cudaStreamSynchronize(s);   // the old buffers may still be read
    for (int b = 0; b < 2; ++b) {
        if (eg_lm_h[b]) cudaFreeHost(eg_lm_h[b]);
        eg_lm_h[b] = eg_lm_d[b] = nullptr;
    }
    eg_lm_cap = 0;
    const size_t bytes = (size_t) m.eg->L * rows * m.eg->row_bytes();
    for (int b = 0; b < 2; ++b)
        if (cudaHostAlloc((void**) &eg_lm_h[b], bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &eg_lm_d[b], eg_lm_h[b], 0) != cudaSuccess) {
            cudaGetLastError();
            return m.fail(err, "the engram's buffers of a layer-major window");
        }
    eg_lm_cap = rows;
    return true;
}

/// The rows of window w into buffer b on a worker, chunk by chunk (each chunk's own promise: its first engram layer waits for
/// that one only), after the other buffer's fetch has finished - one fetch at a time: two at once took each other's drive
void Stage::eg_lm_fetch(int b, const std::vector<LmChunk>& w) {
    if (eg_lm_job[b].valid()) eg_lm_job[b].get();   // (a fetch into it no window took: waited out)
    eg_lm_p[b] = std::vector<std::promise<std::string>>(w.size());
    eg_lm_cf[b].clear();
    for (auto& pr : eg_lm_p[b]) eg_lm_cf[b].push_back(pr.get_future().share());
    eg_lm_p2[b] = std::vector<std::promise<std::string>>(w.size());   // (moex #147: the engram layers after the first)
    eg_lm_cf2[b].clear();
    for (auto& pr : eg_lm_p2[b]) eg_lm_cf2[b].push_back(pr.get_future().share());
    std::vector<int> off;
    int T = 0;
    for (const LmChunk& c : w) {
        m.eg_note(c.toks, c.pos0, c.T);   // its positions noted first (none below this window's: the hashes read back only)
        off.push_back(T);
        T += c.T;
    }
    eg_lm_pos[b] = w.front().pos0;
    eg_lm_T[b] = T;
    m.hb_eg_queued += (int) w.size();
    const std::shared_future<std::string> before = eg_lm_cf2[1 - b].empty() ? std::shared_future<std::string>() : eg_lm_cf2[1 - b].back();
    // moex (DSV41_BACKLOG #147, DSV41_PLAN §76): two passes over the window's chunks where every chunk can be fetched layer by
    // layer - the first engram layer's rows of each chunk, then the other layers' - so the window's layer 1 has waited for
    // half the rows by the time it runs (layer 14's are read while layers 1-13 run); otherwise one pass, both promises a chunk
    bool split = true;
    for (const LmChunk& c : w) split = split && m.eg_split_ok(c.T);
    eg_lm_job[b] = std::async(std::launch::async, [this, b, w, off, before, split] {
        if (before.valid()) before.wait();
        std::vector<char> good(w.size(), 1);
        for (int pass = 0; pass < (split ? 2 : 1); ++pass)
            for (size_t c = 0; c < w.size(); ++c) {
                std::string e;
                const int j0 = split ? pass : 0, j1 = split && pass == 0 ? 1 : -1;
                // (moex, §63 step 2: a host window's chunk c into its lent block's own engram rows)
                const bool ok = !good[c] ? false
                                : egc.empty() ? m.eg_rows(w[c].pos0, w[c].T, eg_lm_h[b], eg_lm_cap, off[c], e, j0, j1)
                                              : m.eg_rows(w[c].pos0, w[c].T, egc[(size_t) b * lmh_wk + c], egc_cap, 0, e, j0, j1);
                good[c] = ok ? 1 : 0;
                const std::string res = ok ? std::string() : (e.empty() ? std::string("engram rows") : e);
                if (!split || pass == 0) eg_lm_p[b][c].set_value(res);
                if (!split || pass == 1) {
                    eg_lm_p2[b][c].set_value(res);
                    ++m.hb_eg_done;
                }
            }
    });
}

void Stage::eg_lm_drop() {
    for (int b = 0; b < 2; ++b)
        if (eg_lm_job[b].valid()) eg_lm_job[b].get();
    for (int b = 0; b < 2; ++b) {
        eg_lm_cf[b].clear();
        eg_lm_p[b].clear();
        eg_lm_cf2[b].clear();
        eg_lm_p2[b].clear();
        eg_lm_pos[b] = -1;
        eg_lm_T[b] = 0;
    }
}

/// The window's chunks through this stage's layers, layer by layer (Impl::prompt: the first stage of DeepSeek-V4.1; it hands
/// the chunks on to the next stage after).  Every (layer, chunk) runs as prompt_chunk runs it - the same rows, positions,
/// sub-batches, routing and MoE groups, the layer's caches written in position order - so each chunk leaves the window as the
/// chunk loop leaves it.  The layer's non-resident experts stream once into the ring, which holds a whole layer, and stay until
/// the window's last chunk has run the layer (Stream::hold).  The running state at each chunk end is put together in a copy
/// of its own, layer by layer: the checkpoints there and a stop's way back read it.
bool Stage::prompt_window(const std::vector<LmChunk>& w, std::string& err) {
    cudaSetDevice(dev);
    use(pr);
    chunk_stopped = false;
    const int K = (int) w.size(), n = (int) hp.n_embd, E = (int) hp.n_expert;
    // moex (§63): more chunks than the loan's sets - their rows in host memory, the sets their staging ring
    const bool host = K > (int) lmw.size();
    const int NS = (int) lmw.size();
    if (K < 1 || (host && (K > lm_win || (int) lmh.size() < K || NS < 3)))
        return m.fail(err, "a layer-major window of " + std::to_string(K) + " chunks");
    const size_t rb = (size_t) hp.hc * n * sizeof(float), pb = 24 * sizeof(float), sb = (size_t) hp.idx_topk * sizeof(int32_t),
                 cb = cand_rows() ? lm_cand_row(m) : 0;
    // chunk c's rows <-> set k (the parts each T rows long, one after the other in its host buffer)
    auto rows_copy = [&](int c, int k, bool up, cudaStream_t cst) {
        const int T = w[(size_t) c].T;
        uint8_t* hb = lmh[(size_t) (lmh_off + c)];
        const LmRows& d = lmw[(size_t) k];
        const struct { void* dev; size_t row; } parts[] = {{d.R, rb}, {d.cpre, pb}, {d.dsel, sb}, {d.dcand, cb}};
        size_t off = 0;
        for (const auto& pt : parts) {
            const size_t bytes = (size_t) T * pt.row;
            if (pt.dev && bytes)
                cudaMemcpyAsync(up ? pt.dev : (void*) (hb + off), up ? (const void*) (hb + off) : pt.dev, bytes,
                                up ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, cst);
            off += bytes;
        }
    };
    int TM = 0;
    for (const LmChunk& c : w) TM = std::max(TM, c.T);
    // the chunk loop's engram buffer comes back after the window: the window's chunks read theirs from its buffer
    struct EgKeep {
        Stage& g;
        uint8_t *h, *d;
        int cap;
        ~EgKeep() {
            g.eg_pr_h = h;
            g.eg_pr_d = d;
            g.eg_pr_cap = cap;
            g.eg_rbase = 0;
        }
    } eg_keep{*this, eg_pr_h, eg_pr_d, eg_pr_cap};
    int eg_b = -1;                          // the buffer of this window's engram rows
    std::vector<int> eg_off((size_t) K, 0);   // each chunk's first row in it
    std::vector<char> eg_in((size_t) K, 0);   // ... and whether they are in
    std::vector<char> eg_in2((size_t) K, 0);  // (moex #147: ... its later engram layers')
    for (int c = 1; c < K; ++c) eg_off[(size_t) c] = eg_off[(size_t) c - 1] + w[(size_t) c - 1].T;
    int64_t stopped_il = -1;
    int stopped_c = -1;
    try {
        // 1. each chunk's streams start as its embedding, its carried pre-mix one-hot (prompt_chunk's first stage) - with the rows
        //    in host memory (§63), as each chunk comes to the window's first layer
        auto embed = [&](int c) -> bool {
            if (!upload_embd(w[(size_t) c].toks, w[(size_t) c].T, cur, w[(size_t) c].pos0)) return false;
            for (int hh = 0; hh < (int) hp.hc; ++hh)
                cudaMemcpy2DAsync(R + (size_t) hh * n, (size_t) hp.hc * n * sizeof(float), cur, (size_t) n * sizeof(float),
                                  (size_t) n * sizeof(float), (size_t) w[(size_t) c].T, cudaMemcpyDeviceToDevice, s);
            v41_pre_reset(w[(size_t) c].T);
            return true;
        };
        for (int c = 0; c < K && !host; ++c) {
            lm_rows(c);
            if (!embed(c)) return m.fail(err, "token out of range");
        }
        if (host) {   // (§63) the staging ring's copy streams and events, once
            for (cudaStream_t& cst : lm_cs)
                if (!cst && cudaStreamCreateWithFlags(&cst, cudaStreamNonBlocking) != cudaSuccess) return m.fail(err, "a copy stream");
            for (auto* evs : {&lm_up_ev, &lm_done_ev, &lm_dn_ev})
                while ((int) evs->size() < NS) {
                    cudaEvent_t ev = nullptr;
                    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) return m.fail(err, "an event");
                    evs->push_back(ev);
                }
        }
        // 2. the engram rows, a window's in a buffer of its own, fetched chunk by chunk on a worker: this window's - queued
        //    behind the window before's, or now - and then the next window's (each chunk's first engram layer waits for its
        //    own).  A chunk's own fetch beside a whole layer's stream queued its page reads behind the records: 116.7 s of
        //    waits on the cold 44K read, against the chunk loop's 26.0; one fetch for a whole window, 196.9 s (2026-10-21)
        if (m.eg) {
            // (§63: every chunk's; step 2: each chunk's in its lent block, egc - the next window's then fetched as it starts,
            // its blocks the same)
            if (egc.empty() && !eg_lm_grow(K, (host ? K : (int) lmw.size()) * pr.rows, err)) return false;
            if (!egc.empty() && (int) egc.size() < lmh_off + K) return m.fail(err, "a host window's engram rows: fewer blocks than chunks");
            int Tw = 0;
            for (const LmChunk& c : w) Tw += c.T;
            if (!egc.empty() && lmh_wk > 0) {   // (two sets: the window's own, fetched ahead into it while the window before ran)
                eg_b = lmh_off / lmh_wk;
                if (!(eg_lm_pos[eg_b] == w[0].pos0 && eg_lm_T[eg_b] == Tw)) eg_lm_fetch(eg_b, w);
            } else {
                for (int bb = 0; bb < 2 && eg_b < 0; ++bb)
                    if (eg_lm_pos[bb] == w[0].pos0 && eg_lm_T[bb] == Tw) eg_b = bb;
                if (eg_b < 0) {
                    eg_b = eg_lm_pos[0] < 0 ? 0 : eg_lm_pos[1] < 0 ? 1 : 0;
                    eg_lm_fetch(eg_b, w);
                }
            }
            if (!lm_next.empty() && (egc.empty() || lmh_wk > 0)) eg_lm_fetch(1 - eg_b, lm_next);
        }
        eg_next_T = 0;   // (no fetch of prompt_layer's own: the window's are under way)
        // 3. the stream: every non-resident expert of the stage's MoE layers, layer by layer in id order, ungated (the window
        //    routes nearly every expert, and the issuer reads the next layer while this one's chunks run)
        Stream str;
        str.start.assign((size_t) m.n_load + 1, 0);
        for (int64_t il = lb; il < le; ++il) {
            if (!hp.is_moe(il)) continue;
            str.start[(size_t) il] = str.seq.size();
            for (int32_t e = 0; e < E; ++e)
                if (m.resident[(size_t) il * E + e] < 0) str.seq.push_back({(int32_t) il, e});
            str.start[(size_t) il + 1] = str.seq.size();
            if (str.start[(size_t) il + 1] - str.start[(size_t) il] > (size_t) ring_n)
                return m.fail(err, "a layer-major window: layer " + std::to_string(il) + " streams more experts than the ring holds");
        }
        struct Join {
            Stream& s;
            ~Join() { stop_issuer(s); }
        } join{str};
        hb_total = str.seq.size();
        hb_issued = 0;
        start_issuer(*this, str);
        // 4. the running state's copies: the window's start (when no copy holds it yet: a call's first window) and each
        //    chunk end, put together as each layer finishes that chunk
        if (!run_snap_at(w[0].pos0)) run_snap_take(w[0].pos0);
        std::vector<RunSnap*> ends((size_t) K, nullptr);
        for (int c = 0; c < K; ++c) ends[(size_t) c] = run_snap_new(w[(size_t) c].pos0 + w[(size_t) c].T);
        // 5. layer by layer, chunk by chunk; a stop is looked for before each (layer, chunk)
        // (moex, §37: the call's last window, its tails read ahead - each layer's kept cache slots go once its chunks' MoE has
        // run, which waited for every copy from them)
        if (lm_release && rel_ev.empty()) {
            rel_ev.assign((size_t) (le - lb), nullptr);
            for (auto& ev : rel_ev) cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        }
        std::deque<int64_t> rel_q;
        auto release_done = [&](bool wait) {
            while (!rel_q.empty()) {
                cudaEvent_t ev = rel_ev[(size_t) (rel_q.front() - lb)];
                if (wait) cudaEventSynchronize(ev);
                else if (cudaEventQuery(ev) != cudaSuccess) {
                    cudaGetLastError();
                    break;
                }
                m.prompt_keep_release(rel_q.front());
                rel_q.pop_front();
            }
        };
        // moex (DSV41_PLAN §59 P7), STRATA_DSV41_LM_SPLIT=1: each layer's attention halves of the window's chunks, then their
        // FFN halves - the next layer's records copy into the ring behind the K attention halves, not behind one; a stop only
        // at a layer's start (every chunk through the layer before).  The same kernels on the same rows: exact
        static const bool lm_split_env = std::getenv("STRATA_DSV41_LM_SPLIT") && std::getenv("STRATA_DSV41_LM_SPLIT")[0] == '1';
        const bool lm_split = lm_split_env && !host;   // (§63: a host window runs each layer whole - its rows go up once a layer)
        for (int64_t il = lb; il < le && stopped_il < 0; ++il) {
            if (lm_release) release_done(false);
            for (int pass = 0; pass < (lm_split ? 2 : 1) && stopped_il < 0; ++pass)
            for (int c = 0; c < K; ++c) {
                if (pass == 0 && (!lm_split || c == 0) && m.stop_asked()) {
                    stopped_il = il;
                    stopped_c = c;
                    break;
                }
                int set = c;
                if (host) {   // (§63) the chunk's rows into its set: its embedding at the window's first layer, else up from host memory
                    set = (int) (((il - lb) * K + c) % NS);
                    lm_rows(set);
                    if (il == lb) {
                        cudaStreamWaitEvent(s, lm_dn_ev[(size_t) set], 0);   // the set's last rows are back
                        if (!embed(c)) return m.fail(err, "token out of range");
                    } else {
                        cudaStreamWaitEvent(lm_cs[0], lm_dn_ev[(size_t) set], 0);
                        rows_copy(c, set, true, lm_cs[0]);
                        cudaEventRecord(lm_up_ev[(size_t) set], lm_cs[0]);
                        cudaStreamWaitEvent(s, lm_up_ev[(size_t) set], 0);
                    }
                } else {
                    lm_rows(c);
                }
                if (m.eg) {
                    eg_pr_h = egc.empty() ? eg_lm_h[eg_b] : egc[(size_t) (lmh_off + c)];
                    eg_pr_d = egc.empty() ? eg_lm_d[eg_b] : egc_d[(size_t) (lmh_off + c)];
                    eg_pr_cap = egc.empty() ? eg_lm_cap : egc_cap;
                    eg_rbase = egc.empty() ? eg_off[(size_t) c] : 0;
                    if (m.eg_index(il) >= 0 && !eg_in[(size_t) c]) {   // its first engram layer: the chunk's rows are in first
                        const auto tw = clk::now();
                        const std::string e2 = m.eg_await(eg_lm_cf[eg_b][(size_t) c]);
                        m.us_eg_wait.fetch_add((int64_t) (ms_since(tw) * 1000.0), std::memory_order_relaxed);
                        if (!e2.empty()) return m.fail(err, e2);
                        eg_in[(size_t) c] = 1;
                    }
                    if (m.eg_index(il) >= 1 && !eg_in2[(size_t) c]) {   // moex (#147): a later engram layer's rows
                        const auto tw = clk::now();
                        const std::string e2 = m.eg_await(eg_lm_cf2[eg_b][(size_t) c]);
                        m.us_eg_wait.fetch_add((int64_t) (ms_since(tw) * 1000.0), std::memory_order_relaxed);
                        if (!e2.empty()) return m.fail(err, e2);
                        eg_in2[(size_t) c] = 1;
                    }
                }
                const int half = !lm_split ? 0 : pass == 0 ? 1 : 2;
                if (half != 1 && hp.is_moe(il)) {
                    str.k = str.start[(size_t) il];
                    str.hold = c + 1 < K;
                }
                hb_il = (int) il;
                hb_c = c;
                if (!prompt_layer((int) il, w[(size_t) c].T, w[(size_t) c].pos0, str, err, half)) return false;
                // the chunk end's copy of the layer's running state (its ring, its compressor's rows): the attention half
                // writes it, so in the split it is taken after that half - before the next chunk's moves it on
                if (half != 2 && ends[(size_t) c]) run_snap_layer(il, *ends[(size_t) c]);
                if (host) {   // (§63) and back to host memory once its layer has run
                    cudaEventRecord(lm_done_ev[(size_t) set], s);
                    cudaStreamWaitEvent(lm_cs[1], lm_done_ev[(size_t) set], 0);
                    rows_copy(c, set, false, lm_cs[1]);
                    cudaEventRecord(lm_dn_ev[(size_t) set], lm_cs[1]);
                }
                if (half == 1) continue;
                m.pl_add(1);
            }
            if (lm_release && stopped_il < 0 && hp.is_moe(il)) {
                cudaEventRecord(rel_ev[(size_t) (il - lb)], s);
                rel_q.push_back(il);
            }
        }
        if (lm_release) release_done(true);
        if (host && cudaStreamSynchronize(lm_cs[1]) != cudaSuccess) return m.fail(err, "a host window's rows back");   // (§63)
        if (stopped_il >= 0) {   // the rest of the window does not run: its stream stops (the reads in flight land first)
            stop_issuer(str);
            chunk_stopped = true;
            if (stopped_il > lb || stopped_c > 0) run_pos = -1;   // layers ran: the state is inside the window
            fprintf(stderr, "strata-x: a stop before layer %lld of the layer-major window's chunk %d at %lld (GPU %d)%s\n",
                    (long long) stopped_il, stopped_c, (long long) w[(size_t) stopped_c].pos0, dev,
                    run_pos < 0 ? ": its running state goes back" : "");
        } else {
            while (str.issuer.joinable() && str.issued.load(std::memory_order_acquire) < str.seq.size()) std::this_thread::yield();
            stop_issuer(str);
        }
    } catch (const std::exception& e) {
        return m.fail(err, e.what());
    }
    use(pr);
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return m.fail(err, std::string("a layer-major window: ") + cudaGetErrorString(ce));
    if (chunk_stopped) return true;
    run_pos = w.back().pos0 + w.back().T;
    for (const LmChunk& c : w) st.tokens_prefill += c.T;
    return true;
}

/// The NextN block over a chunk's positions pos0..pos0 + n_next - 1 (their next tokens known), to fill its caches:
/// the trunk's final hidden (the streams' mean, the output norm) and the next token's embedding, eh_proj, the DSA
/// attention, the MoE.  No head: drafts are not read during the prompt.
bool Stage::prompt_nextn(int T, int n_next, int64_t pos0, const int32_t* next_toks, Stream& str, std::string& err) {
    const int n = (int) hp.n_embd, il = (int) hp.n_layer;
    const Layer& l = L[(size_t) il];
    mhc_mean_w(R, h, n, T, s);
    rms_norm(h, m.output_norm.f(), h, T, n, hp.rms_eps, s);
    float* z = R;   // the streams are no longer needed: z (n_next, n) lives in their first rows
    const int Tm = std::min(T, n_next);
    for (int b0 = 0; b0 < Tm; b0 += SB) {
        const int tn = std::min(SB, Tm - b0);
        float* cr = cur + (size_t) b0 * n;
        float* orr = out + (size_t) b0 * n;
        // the drafter's input at position p is the NEXT position's embedding: an image row there when p + 1 is in a picture
        if (!upload_embd(next_toks + b0, tn, cr, pos0 + b0 + 1)) return m.fail(err, "MTP token out of range");
        rms_norm(cr, m.enorm.f(), cr, tn, n, hp.rms_eps, s);
        rms_norm(h + (size_t) b0 * n, m.hnorm.f(), orr, tn, n, hp.rms_eps, s);
        cudaMemcpy2DAsync(flat, (size_t) 2 * n * sizeof(float), cr, (size_t) n * sizeof(float), (size_t) n * sizeof(float),
                          (size_t) tn, cudaMemcpyDeviceToDevice, s);
        cudaMemcpy2DAsync(flat + n, (size_t) 2 * n * sizeof(float), orr, (size_t) n * sizeof(float),
                          (size_t) n * sizeof(float), (size_t) tn, cudaMemcpyDeviceToDevice, s);
        quant(flat, xq, 2 * n, tn);
        pj(m.eh_proj, flat, xq, z + (size_t) b0 * n, tn);
    }
    for (int b0 = 0; b0 < Tm; b0 += SB) {
        const int tn = std::min(SB, Tm - b0);
        rms_norm(z + (size_t) b0 * n, l.attn_norm.f(), cur + (size_t) b0 * n, tn, n, hp.rms_eps, s);
        dsa(il, pos0 + b0, tn, cur + (size_t) b0 * n, out + (size_t) b0 * n);
    }
    add_f32(z, out, z, Tm * n, s);
    rms_norm(z, l.ffn_norm.f(), cur, Tm, n, hp.rms_eps, s);
    return moe_chunk(il, Tm, str, err);
}

/// MiMo's MTP blocks over a chunk's positions pos0 .. pos0 + n_next - 1 (their next tokens known), to fill their caches as
/// mainline's draft-mtp does after every target batch: each block over the trunk's final residual (h_nextn, still in R) and
/// the next tokens' embeddings.  No head: drafts are not read during the prompt.
bool Stage::prompt_mtp_mimo(int T, int n_next, int64_t pos0, const int32_t* next_toks, std::string& err) {
    const int n = (int) hp.n_embd, Tm = std::min(T, n_next);
    float* z = shared;   // MiMo has no shared expert: its rows are free here
    for (int hd = 0; hd < mtp_heads(); ++hd) {
        const int il = (int) hp.n_layer + hd;
        const Layer& l = L[(size_t) il];
        for (int b0 = 0; b0 < Tm; b0 += SB) {
            const int tn = std::min(SB, Tm - b0);
            float* e = cur + (size_t) b0 * n;
            float* hh = out + (size_t) b0 * n;
            float* zz = z + (size_t) b0 * n;
            if (!upload_embd(next_toks + b0, tn, e, pos0 + b0 + 1)) return m.fail(err, "MTP token out of range");
            rms_norm(e, l.enorm.f(), e, tn, n, hp.rms_eps, s);
            rms_norm(R + (size_t) b0 * n, l.hnorm.f(), hh, tn, n, hp.rms_eps, s);
            concat_rows(e, hh, flat, n, tn, s);
            quant(flat, xq, 2 * n, tn);
            pj(l.eh_proj, flat, xq, zz, tn);
            rms_norm(zz, l.attn_norm.f(), e, tn, n, hp.rms_eps, s);
            gqa(il, pos0 + b0, tn, e, hh);
            add_f32(zz, hh, zz, tn * n, s);
            rms_norm(zz, l.ffn_norm.f(), e, tn, n, hp.rms_eps, s);
            swiglu_ffn(il, tn, l.ffn_gate, l.ffn_up, l.ffn_down, 0.0f, e, hh);   // the block's output is not needed: only its caches
        }
    }
    return true;
}

// ------------------------------------------------------------------ the chunk choice and the pipeline

namespace {
int ring_slots_for(const Engine::Impl& m) {
    if (m.ring_req > 0) return std::clamp((int) m.ring_req, 2 * kGroup, kRingCap);   // a request's read_ring (an A/B key)
    if (m.opt.prefill_ring > 0) return std::clamp(m.opt.prefill_ring, 2 * kGroup, kRingCap);
    // Strata sizes its ring in BYTES (prefill.cpp ring_bytes: 384 Q2_0 blobs = 506 MiB, "96 slots 1153 tok/s, 384
    // 1294") because the right depth is a matter of TIME: the copy engine must be able to run through the attention
    // half of a layer while the compute stream is not consuming experts.  GLM's blobs are 5.7x bigger and its layers 288
    // experts; measured here (24K prompt, two 3080s, chunk 8192, 4 KB-page-or-THP arena): 64 slots 624 T/s, 160 -> 745,
    // 320 -> 722, 480 -> 705 - past ~0.55 of a layer every slot costs a resident expert (a lent slot is streamed again
    // on each chunk) for no copy-ahead it can use.  So: 55% of a layer's experts, in whole groups.  Not fully pinned
    // (the in-place mode): Strata's quarter budget, 96 Q2_0 blobs.
    if (!m.arena_pinned) {
        const int64_t r = m.blob_max ? (int64_t) (96ull * kQ2_0Blob / m.blob_max) : 0;
        return (int) std::clamp<int64_t>(r, 2 * kGroup, kRingCap);
    }
    // moex (ERNIE55_PLAN): EngineOptions::prefill_ring_frac in place of the 0.55 - ERNIE-5.5's layer has a longer attention and
    // dense half (FP32 SGEMMs at the cards' power limit) for the copies to run through: 24K read 49.7 s at 0.55 (288 slots),
    // 47.0 s at 448 slots
    const double frac = m.opt.prefill_ring_frac > 0.0 ? std::min(m.opt.prefill_ring_frac, 1.0) : 0.55;
    const int64_t r = ((int64_t) (frac * m.hp.n_expert) + kGroup / 2) / kGroup * kGroup;
    return (int) std::clamp<int64_t>(r, 2 * kGroup, kRingCap);
}
/// The ring of a read whose largest chunk is TM rows: ring_slots_for's, smaller for a short read.  Every lent slot is an
/// expert out of VRAM for the read and a refill after it, and a short chunk has little compute for the copy-ahead to hide
/// behind - Strata stages a read below 1024 tokens through 8 slots.  Measured on GLM-5.3-Flash UD-IQ3_XXS, 2x RTX 3080,
/// follow-up reads at 33K-103K context (medians of 3-4, moex/experiments/glm_prompt_ab.py): 300 tokens 5.48 s at the
/// 160-slot ring, 3.55 s at 32; 1000: 10.15 / 7.63 (64: 8.31); 2000: 14.33 / 11.16 (64: 11.72); 4096: 17.93 / 17.85,
/// 64: 16.89.  MiMo-V2.6 SX-124G (144 slots / this rule): 300 3.45 / 2.95 s, 1000 6.36 / 5.81, 2000 9.89 / 8.66, 4096
/// 15.39 / 14.48; DeepSeek-V4 Q3_K: 300 3.54 / 3.14, 2000 9.20 / 7.95.  So 32 slots below 3072 rows, 64 below 6144, the
/// full ring from there (the 8192-row chunk's measured best).
/// An explicit ring (a request's read_ring, --prefill-ring) wins; STRATA_X_SHORT_RING=0 keeps the full ring always.
int ring_for_read(const Engine::Impl& m, int TM) {
    const int full = ring_slots_for(m);
    static const bool on = [] { const char* e = std::getenv("STRATA_X_SHORT_RING"); return !(e && e[0] == '0'); }();
    if (!on || m.ring_req > 0 || m.opt.prefill_ring > 0 || !m.arena_pinned) return full;
    if (TM < 3072) return std::min(full, 2 * kGroup);
    if (TM < 6144) return std::min(full, 4 * kGroup);
    return full;
}
int round256(int v) { return (v + 255) / 256 * 256; }
}  // namespace

/// moex: THE LENT EXPERTS READ IN WHILE THE PROMPT RUNS.  A read lends each stage's top slots: their experts are streamed
/// from their sources on every chunk and copied back afterwards - from their RAM copies (the lend region), the rest from the
/// GGUF through the page cache (issue_entry's three slices, fill_slot).  The adaptive swaps put experts with no RAM copy into
/// those slots (the exchange hands the incoming expert's blob to the victim), and the load never cached their pages
/// (O_DIRECT): the first long read after a session read them from the disk, page fault by page fault.  MiMo-V2.6 SX-124G,
/// 20 GiB headroom, 2x RTX 3080, 8K fresh read after four 300-token follow-ups: 19.7-20.1 s with 3.35 GiB from the disk,
/// 16.1 s on the next read or with --no-adaptive (0.02 GiB).  So when the stages have lent, a thread reads those experts
/// into the page cache, in layer order, while the read computes: one part (gate / up / down) at a time - mincore, and a part
/// with a page missing gets POSIX_FADV_WILLNEED in 128 KiB pieces (one call reads at most the readahead window, 128 KiB on
/// this NVMe: an earlier MADV_WILLNEED try asked for whole parts and got their first 128 KiB) and a pread that waits for it.
/// The thread stays out of the page tables: touching the pages and MADV_DONTNEED-ing them, beside the stream's copies of
/// the same pages, ran at 0.02-0.3 GB/s instead of 2.2 (DeepSeek-V4, the sampled thread on its CPU the whole time).  The
/// disk queue holds one part (~3 MiB), never the job, and nothing is read that the read would not read.
/// STRATA_X_WARM_LEND=0: off; STRATA_X_WARM_LOG=1: every job in the log (else those of a second or more).
void Engine::Impl::warm_start() {
    static const bool on = [] { const char* e = std::getenv("STRATA_X_WARM_LEND"); return !(e && e[0] == '0'); }();
    if (!on || n_slots == 0 || slot_bytes == 0 || warm_thr.joinable()) return;
    struct Shard {
        const uint8_t* base;
        uint64_t size;
        int fd;
    };
    std::vector<Shard> sh;
    for (size_t i = 0; i < gm->size(); ++i) {
        const strata::GgufFile& f = gm->shard(i);
        if (f.tensors().empty()) continue;
        const strata::TensorInfo& t0 = f.tensors().front();
        const int fd = ::open(f.path().c_str(), O_RDONLY | O_CLOEXEC);
        if (fd >= 0) sh.push_back({f.tensor_data(t0) - f.data_start() - t0.offset, f.file_size(), fd});
    }
    if (sh.empty()) return;
    auto work = [this, sh] {
        pthread_setname_np(pthread_self(), "strata-warm");
        static const bool log_all = std::getenv("STRATA_X_WARM_LOG") && std::getenv("STRATA_X_WARM_LOG")[0] == '1';
        const uintptr_t pg = 4096, kPiece = 128u << 10;
        std::vector<unsigned char> vec;
        std::vector<uint8_t> buf;
        for (;;) {
            std::vector<std::pair<const uint8_t*, size_t>> job;
            {
                std::unique_lock<std::mutex> lk(warm_mu);
                warm_cv.wait(lk, [&] { return warm_new.load() || warm_stop.load(); });
                if (warm_stop.load()) break;
                job.swap(warm_job);
                warm_new = false;
            }
            const auto t0 = clk::now();
            size_t parts = 0;
            uint64_t missing = 0;
            for (const auto& [p, n] : job) {
                if (warm_stop.load() || warm_new.load()) break;   // a newer job holds every expert still lent
                const uintptr_t a = (uintptr_t) p & ~(pg - 1), e = ((uintptr_t) p + n + pg - 1) & ~(pg - 1);
                vec.assign((e - a) / pg, 0);
                if (mincore((void*) a, e - a, vec.data()) == 0 &&
                    std::all_of(vec.begin(), vec.end(), [](unsigned char v) { return (v & 1) != 0; }))
                    continue;                                     // cached
                const Shard* s = nullptr;
                for (const Shard& x : sh)
                    if (p >= x.base && p < x.base + x.size) s = &x;
                if (!s) continue;
                const off_t off = (off_t) (a - (uintptr_t) s->base);
                for (uintptr_t c = 0; c < e - a; c += kPiece)
                    posix_fadvise(s->fd, off + (off_t) c, (off_t) std::min(kPiece, e - a - c), POSIX_FADV_WILLNEED);
                if (buf.size() < e - a) buf.resize(e - a);
                if (pread(s->fd, buf.data(), e - a, off) < 0) continue;   // waits for the part: the queue stays one part deep
                for (unsigned char v : vec) missing += (v & 1) ? 0 : pg;
                ++parts;
            }
            const double t = ms_since(t0) / 1000.0;
            if (opt.verbose && (t >= 1.0 || (log_all && !job.empty())))
                fprintf(stderr, "strata-x: the lent experts' file pages: %zu of %zu expert parts read in (%.2f GiB) in %.1f s\n",
                        parts, job.size(), missing / 1073741824.0, t);
        }
        for (const Shard& x : sh) ::close(x.fd);
    };
    try {
        warm_thr = std::thread(work);
    } catch (const std::exception&) {   // no thread (out of resources): the reads fault the pages in themselves, as before
        for (const Shard& x : sh) ::close(x.fd);
    }
}

void Engine::Impl::warm_lend() {
    if (!warm_thr.joinable()) return;
    const int64_t NE = hp.n_expert;
    std::vector<int32_t> keys;
    for (const auto& gp : stages)
        for (const int32_t key : gp->lent_keys)
            if (key >= 0 && (ram_off.empty() || ram_off[(size_t) key] < 0)) keys.push_back(key);   // not refilled from RAM
    if (keys.empty()) return;   // the slots of a short read hold the lend region's experts
    // by layer: the order the chunks stream them in (a lent expert is streamed from its source too), and the file's
    std::sort(keys.begin(), keys.end());
    std::vector<std::pair<const uint8_t*, size_t>> job;
    for (const int32_t key : keys) {
        const int64_t il = key / NE, e = key % NE;
        const Layer& l = L[(size_t) il];
        if (l.rec_stride) continue;   // a pack's record: never through the page cache (the refill reads it with O_DIRECT)
        job.push_back({l.eg + (size_t) e * l.sg, l.sg});
        job.push_back({l.eu + (size_t) e * l.su, l.su});
        job.push_back({l.ed + (size_t) e * l.sd, l.sd});
    }
    {
        std::lock_guard<std::mutex> lk(warm_mu);
        warm_job.swap(job);
        warm_new = true;
    }
    warm_cv.notify_one();
}

/// Strata's --prefill auto: the largest chunk on a 256 grid up to 8192 (or EngineOptions::prefill_chunk) whose
/// buffers and ring fit every stage's lendable slots - or, with nothing to lend, its free memory; a segment lends
/// only what min(chunk, segment) needs.
int Engine::Impl::choose_chunk(int n, int cap) const {
    const int want = std::min(cap > 0 ? cap : opt.prefill_chunk > 0 ? opt.prefill_chunk : 8192, round256(n));
    auto fits = [&](int T) {
        const int ring = ring_for_read(*this, T);
        const int sub = std::min(opt.prefill_sub, T);
        for (const auto& gp : stages) {
            const Stage& g = *gp;
            // (moex: and the stop-inside-a-chunk copies of the running state Stage::lend adds - without them the chunk could fill
            // a stage's lendable slots to the byte and its lend fail: BF16 dense on GPU 1, 362 slots = 6,494 MiB, 2026-11-09)
            const size_t snaps = stop_layers ? (size_t) g.run_snap_count() * (((g.run_state_bytes() + 255) & ~(size_t) 255) + 256) : 0;
            const size_t need = Stage::prompt_bytes(*this, T, sub, ring, g.last, g.di, 0, g.cand_rows()) + snaps;
            uint64_t room = 0;
            if (g.n_slots > 0 && slot_bytes > 0) {
                room = (uint64_t) g.lend_bytes(g.lendable());
            }
            if (room < need) {
                size_t fr = 0, tot = 0;
                cudaSetDevice(g.dev);
                cudaMemGetInfo(&fr, &tot);
                room = std::max<uint64_t>(room, fr > (512ull << 20) ? fr - (512ull << 20) : 0);
            }
            if (room < need) return false;
        }
        return true;
    };
    for (int T = want; T >= 256; T -= 256)
        if (fits(T)) return T;
    return 256;
}

/// moex (DSV41_PLAN §18): k more of the call's (chunk, layer) units done - a progress line (pctl->work) when pl_on; the
/// stages' threads call it
void Engine::Impl::pl_add(int64_t k) {
    if (!pl_on) return;
    const int64_t d = pl_done.fetch_add(k, std::memory_order_relaxed) + k;
    pctl->work(std::min(d, pl_total), pl_total);
}

bool Engine::Impl::prompt(const int32_t* toks, int n, int64_t pos0, int n_avail, std::string& err) {
    if (n <= 0) return true;
    // moex (STRATA_HY4_KV_GROW): the sparse layers' caches mapped through the read (before any slot is lent)
    if ((hp.arch == Arch::hy4 || hp.arch == Arch::ernie) && kv_grow && !hy4_kv_ensure(pos0 + n + kTmax + 16, err)) return false;
    prompt_end = pos0 + n;
    finalize_swaps(true);   // a copy in flight into a slot must land before the slot is lent
    // STRATA_DSV41_PROMPT_KEEP: the slots this call reads into or copies from stay put until it returns (its refill included)
    prompt_keep_reset();
    struct KeepEnd {
        Engine::Impl& m;
        ~KeepEnd() { m.prompt_keep_reset(); }
    } keep_end{*this};
    const auto t_all = clk::now();
    // a small / mid-size segment: no stream, the non-resident experts on the CPU pool (EngineOptions::cpu_read; -1: auto, the
    // learned crossover; a request's read_cpu or STRATA_X_CPU_READ=N: fixed)
    static const int64_t cpu_env = std::getenv("STRATA_X_CPU_READ") ? std::atoll(std::getenv("STRATA_X_CPU_READ")) : -1;
    const int64_t cpu_read = cpu_read_req >= 0 ? cpu_read_req : cpu_env >= 0 ? cpu_env : opt.cpu_read;
    // moex: a segment up to the share's limit reads with the CPU sharing the experts (Stage::moe_prompt_share; it replaces the
    // auto CPU read - a request's read_share, STRATA_X_SHARE=N or EngineOptions::prefill_share: fixed, 0 off); an explicit
    // CPU read (cpu_read > 0) still wins below its limit, for A/B
    static const int64_t share_env = std::getenv("STRATA_X_SHARE") ? std::atoll(std::getenv("STRATA_X_SHARE")) : -1;
    const int64_t share_n = share_req >= 0 ? share_req : share_env >= 0 ? share_env : opt.prefill_share;
    // moex 10-09: the mode is each CHUNK's, not the segment's.  A checkpoint cut near the end (the last turn token) leaves a
    // chunk of a few tokens; streamed, it cost a whole pass over the experts (an 8K prompt read in 23.7 s instead of 14.9 s
    // once decode windows no longer read such tails, short_read 0) - shared or on the CPU it costs its tokens.
    const int64_t share_lim = pool && !rio && share_n != 0 ? (share_n < 0 ? share_limit() : share_n) : 0;
    // (DeepSeek-V4.1's SSD tier: the CPU read and the share read experts in place - not with records: the stream only)
    const int64_t cpu_lim = !pool || rio ? 0 : cpu_read > 0 ? cpu_read : share_n == 0 && cpu_read < 0 ? cpu_read_limit() : 0;
    auto mode_of = [&](int64_t T) -> int {   // 0 the stream, 1 the CPU read, 2 shared
        if (cpu_lim > 0 && T <= cpu_lim) return 1;
        if (share_lim > 0 && T <= share_lim) return 2;
        return 0;
    };
    std::atomic<int> n_mode[3] = {{0}, {0}, {0}};   // the first stage's chunks by mode (this call's costs, below)
    struct CpuOff {
        std::vector<std::unique_ptr<Stage>>& st;
        ~CpuOff() {
            for (auto& gp : st) gp->cpu_moe = gp->share = false;
        }
    } cpu_off{stages};
    const int TM0 = choose_chunk(n);
    // THE PLAN (moex): chunks of TM0 that end at each checkpoint position - and a segment whose last chunk would be a small
    // remainder (a quarter of a chunk or less) spread over its other chunks instead.  Every chunk streams each non-resident
    // expert once, ~7 s a stage on this box whatever its length, so a remainder cost a whole chunk: MiMo's 8385 tokens read in
    // 24.5 s as 8192 + 193 and in 16.5 s as one chunk of 8448.  No larger than 1.25 x: past that a chunk loses the stages'
    // overlap (16384 tokens: 25.5 s as 2 x 8192, 27.9 s as one) and lends slots beyond the RAM copies.  Auto mode only.
    std::vector<int> plan;
    {
        std::vector<int64_t> bounds = {0, (int64_t) n};
        if (pctl)
            for (int64_t cut : pctl->cuts)
                if (cut - pos0 > 0 && cut - pos0 < n) bounds.push_back(cut - pos0);
        std::sort(bounds.begin(), bounds.end());
        bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
        const int merge_cap = opt.prefill_chunk > 0 ? 0 : choose_chunk(std::min(n, TM0 + TM0 / 4), TM0 + TM0 / 4);
        for (size_t b = 0; b + 1 < bounds.size(); ++b) {
            const int64_t len = bounds[b + 1] - bounds[b];
            int64_t nc = (len + TM0 - 1) / TM0, c = TM0;
            if (nc >= 2 && merge_cap > TM0) {
                const int64_t c2 = (len + nc - 2) / (nc - 1);   // nc - 1 chunks
                if (c2 <= merge_cap) {
                    --nc;
                    c = c2;
                }
            }
            for (int64_t p = 0; p < len; p += c) plan.push_back((int) std::min(c, len - p));
        }
    }
    const int TM = *std::max_element(plan.begin(), plan.end());
    last_prompt_chunks = (int) plan.size();
    if (rio && rio->keep_on && !rio->keep_cap_long.empty() && rio->keep_cap_long[0] > 0 && plan.size() >= 2 &&
        replay_layer >= 0) {   // STRATA_DSV41_PROMPT_KEEP_LONG: a long prompt's chunks share the layers below
        std::lock_guard<std::mutex> lk(rio->mu);                         // the replayed ones - kept for its later chunks
        rio->long_now = true;
        rio->prefer_ge = (int) replay_layer;
    }
    replay_ends.clear();   // bounded replay: where later reads start - this call's end and its checkpoint cuts
    if (replay_layer >= 0) {
        if (pctl)
            for (int64_t cut : pctl->cuts)
                if (cut > pos0 && cut < pos0 + n) replay_ends.push_back(cut);
        replay_ends.push_back(pos0 + n);
        std::sort(replay_ends.begin(), replay_ends.end());
    }
    reset_dev_cache();   // choose_chunk read every GPU's free memory
    const int sub = std::min(opt.prefill_sub, TM);
    bool any_ring = false;   // a chunk streams (shared or not): the ring is lent
    for (int T : plan) any_ring |= mode_of(T) != 1;
    const int ring = any_ring ? ring_for_read(*this, TM) : 0;
    // moex (DSV41_BACKLOG #76), STRATA_DSV41_LAYER_MAJOR=1: the first stage reads windows of up to STRATA_DSV41_LM_CHUNKS
    // chunks (default 4) layer by layer - as many as its loan holds: a ring of a whole layer's experts, the window's other
    // chunks' rows, the copies of the running state - for a call of two or more streamed chunks (DeepSeek-V4.1 on two or more
    // stages, its SSD tier, a call the serving layer controls)
    static const bool lm_env = std::getenv("STRATA_DSV41_LAYER_MAJOR") && std::getenv("STRATA_DSV41_LAYER_MAJOR")[0] == '1';
    static const int lm_max = std::getenv("STRATA_DSV41_LM_CHUNKS") ? std::max(2, std::atoi(std::getenv("STRATA_DSV41_LM_CHUNKS"))) : 4;
    int lm_k = 0, lm_w = 0;
    bool lm_lend = false;           // (moex, §63 step 2: the host window's rows in the SSD cache's lent blocks)
    bool lm_dbl = false;            // ... two sets of them, one a window (STRATA_DSV41_LM_HOST=2)
    size_t lm_chunk_b = 0;          // ... a block's share: a chunk's rows, then its engram rows
    std::vector<uint8_t*> lm_blk;   // ... the blocks, lent as the window is sized (before the lend's own cache copies)
    // moex (§63 step 2): the lent blocks back once every GPU is done with them (the rows' copies, the next stage's reads of
    // them, the engram fetch into them) - on every way out of the call
    struct LendEnd {
        Engine::Impl& m;
        bool on = false;
        ~LendEnd() {
            if (!on) return;
            for (auto& gp : m.stages) {
                cudaSetDevice(gp->dev);
                cudaStreamSynchronize(gp->s);
                cudaStreamSynchronize(gp->cs);
                for (cudaStream_t cst : gp->lm_cs)
                    if (cst) cudaStreamSynchronize(cst);
            }
            m.stages[0]->eg_lm_drop();
            Stage& g0 = *m.stages[0];
            if (g0.lmh_lent) {   // (a call that failed before taking them still holds its own host rows: they stay)
                g0.lmh.clear();
                g0.lmh_bytes = 0;
                g0.lmh_lent = false;
            }
            g0.egc.clear();
            g0.egc_d.clear();
            g0.egc_cap = 0;
            g0.lmh_off = g0.lmh_wk = 0;
            m.cache_unlend();
            m.reset_dev_cache();
        }
    } lend_end{*this};
   // (moex, §63: lm_w - a window's chunks when its rows are in host memory)
    const int lm_ring = (int) hp.n_expert;   // a layer's experts, every one of them lent at worst
    if (lm_env && hp.v41 && stages.size() >= 2 && pctl && rio && plan.size() >= 2 && lm_ring <= kRingCap &&
        std::all_of(plan.begin(), plan.end(), [&](int T) { return mode_of(T) == 0; })) {
        const Stage& g0 = *stages[0];
        const size_t cap = g0.lend_bytes(g0.lendable_lm()),
                     base = Stage::prompt_bytes(*this, TM, sub, lm_ring, g0.last, g0.di, 0, g0.cand_rows());
        for (int k = std::min<int>(lm_max, (int) plan.size()); k >= 2 && lm_k == 0; --k)
            if (base + g0.lm_extra_bytes(TM, k) <= cap) lm_k = k;
        fprintf(stderr, "strata-x: layer-major: %zu chunks of up to %d tokens in windows of %d (GPU %d's loan: %.2f of %.2f GiB)\n",
                plan.size(), TM, lm_k, g0.dev, lm_k > 0 ? (double) (base + g0.lm_extra_bytes(TM, lm_k)) / 1073741824.0 : 0.0,
                (double) cap / 1073741824.0);
        // moex (DSV41_PLAN §72): one window holds every chunk - the reads STRATA_DSV41_PROMPT_KEEP_LONG keeps "for its later
        // chunks" serve no later chunk then, and each took the least recently used slot of the replayed layers first (prefer_ge):
        // with a warm cache, the records the tails were about to read (2,073 cache hits in a 16K read against ~5,100 to expect).
        // Not kept; the hot ones still are (STRATA_DSV41_PROMPT_HOT).  Where a record is read from changes no result.
        // STRATA_DSV41_KEEP_LONG_ONE=1: kept as before
        static const bool keep_long_one = std::getenv("STRATA_DSV41_KEEP_LONG_ONE") && std::getenv("STRATA_DSV41_KEEP_LONG_ONE")[0] == '1';
        if (lm_k >= (int) plan.size() && !keep_long_one) {
            std::lock_guard<std::mutex> lk(rio->mu);
            if (rio->long_now) {
                rio->long_now = false;
                rio->prefer_ge = -1;
                fprintf(stderr, "strata-x: layer-major: one window holds every chunk: the reads are not kept for later chunks "
                                "(STRATA_DSV41_KEEP_LONG_ONE=1: kept)\n");
            }
        }
        // moex (DSV41_PLAN §63), opt-in STRATA_DSV41_LM_HOST=1: more chunks than a window holds - the window's rows in pinned host
        // memory, the loan three staging sets and a copy of the state for each chunk: as many chunks a window as the loan and
        // MemAvailable (less the RAM copy's headroom) hold - every chunk of the call when they do, so each layer's records stream
        // once a call
        // (=2: two windows of half the call each, with a set of blocks each, so GPU 1 runs one while GPU 0 runs the other)
        static const int lm_host_env = std::getenv("STRATA_DSV41_LM_HOST") ? std::atoi(std::getenv("STRATA_DSV41_LM_HOST")) : 0;
        if (lm_host_env >= 1 && (int) plan.size() > std::max(lm_k, 3)) {
            const size_t tok = (size_t) hp.hc * hp.n_embd * sizeof(float) + 24 * sizeof(float) + (size_t) hp.idx_topk * sizeof(int32_t) +
                               (g0.cand_rows() ? lm_cand_row(*this) : 0);
            // step 2: a chunk's rows and its engram rows in one of the SSD cache's pinned blocks, lent for the call (a single pass
            // reads each record once, into the ring: the blocks' records are the least recently used ones, dropped) - else
            // MemAvailable less the RAM copy's headroom (which the RAM copy fills; STRATA_DSV41_LM_HOST_KEEP_GIB: a check's floor)
            const size_t egb = eg ? ((size_t) eg->L * TM * eg->row_bytes() + 255) & ~(size_t) 255 : 0;
            lm_chunk_b = (((size_t) TM * tok + 255) & ~(size_t) 255) + egb;
            const int lendable = rio ? cache_lendable(lm_chunk_b) : 0;
            static const double keep_gib = std::getenv("STRATA_DSV41_LM_HOST_KEEP_GIB") ? std::atof(std::getenv("STRATA_DSV41_LM_HOST_KEEP_GIB"))
                                                                                         : opt.resident_headroom_gib;
            const uint64_t avail = lm_mem_available(), keep = (uint64_t) (keep_gib * 1073741824.0);
            const uint64_t room = avail > keep ? avail - keep : 0;
            int kw = 0;   // (a cut at each checkpoint can split a chunk: the window is counted in the cut chunks, each <= TM)
            const int nck = (int) plan.size() + (int) pctl->cuts.size();
            if (lm_host_env == 2 && nck >= 8) {   // two sets of up to half the call's chunks - and of the blocks free to lend -,
                                                 // one a window, the windows taking turns
                int half = std::min((nck + 1) / 2, lendable / 2);
                while (half >= 4 && base + g0.lm_extra_bytes(TM, 3, half) > cap) --half;
                // (only when it means fewer windows than GPU 0's loan holds: the same count is today's read with copies added)
                if (half >= 4 && (lm_k < 2 || (nck + half - 1) / half < (nck + lm_k - 1) / lm_k)) {
                    kw = half;
                    lm_lend = true;
                    lm_dbl = true;
                }
            }
            for (int k = nck; k > 3 && kw == 0; --k) {
                if (base + g0.lm_extra_bytes(TM, 3, k) > cap) continue;
                if (k <= lendable) {
                    kw = k;
                    lm_lend = true;
                } else if ((uint64_t) k * TM * tok <= room) {
                    kw = k;
                }
            }
            if (kw > lm_k) {
                lm_k = 3;
                lm_w = kw;
            } else {
                lm_lend = false;
                lm_dbl = false;
            }
            if (lm_lend) {
                lend_end.on = true;
                const int nb = lm_dbl ? 2 * lm_w : lm_w;
                if (cache_lend(nb, lm_chunk_b, lm_blk) < nb) {   // (counted just now under the same lock: not expected)
                    cache_unlend();
                    lm_blk.clear();
                    lm_lend = false;
                    lm_dbl = false;
                    lend_end.on = false;
                    lm_k = 0;
                    lm_w = 0;
                    fprintf(stderr, "strata-x: layer-major: the SSD cache could not lend the host window's blocks; windows as before\n");
                    for (int k = std::min<int>(lm_max, (int) plan.size()); k >= 2 && lm_k == 0; --k)
                        if (base + g0.lm_extra_bytes(TM, k) <= cap) lm_k = k;
                }
            }
            if (lm_w > 0 && !lm_dbl && lm_w >= nck && rio) {   // one window: its read-ahead may hold 3 layers' slots past the
                std::lock_guard<std::mutex> lk(rio->mu);          // share, a layer's back once its chunks have run it
                rio->deep_cap = 3 * (int) hp.n_expert;
            }
            fprintf(stderr, "strata-x: layer-major, the rows in host memory (STRATA_DSV41_LM_HOST): windows of %d chunks of up to %d "
                            "tokens, %.2f GiB of pinned host rows (%s), 3 staging sets (GPU %d's loan: %.2f of %.2f GiB)\n",
                    lm_w, TM, (double) (lm_dbl ? 2 : 1) * lm_w * (lm_lend ? lm_chunk_b : (size_t) TM * tok) / 1073741824.0,
                    lm_lend ? ("the SSD cache's blocks: " + std::to_string(lendable) + " free to lend").c_str()
                            : ("MemAvailable " + std::to_string((int) (avail / 1073741824.0)) + " GiB").c_str(),
                    g0.dev, lm_w > 0 ? (double) (base + g0.lm_extra_bytes(TM, 3, lm_w)) / 1073741824.0 : 0.0, (double) cap / 1073741824.0);
        }
    }
    // moex (DSV41_BACKLOG #78), STRATA_DSV41_STOP_LAYERS=1 (and a layer-major window, whose copies it uses): a stop between a
    // chunk's layers (DeepSeek-V4.1 on two or more stages, a call the serving layer controls: never an image span, which is
    // read whole)
    static const bool stop_env = std::getenv("STRATA_DSV41_STOP_LAYERS") && std::getenv("STRATA_DSV41_STOP_LAYERS")[0] == '1';
    stop_layers = (stop_env || lm_k > 0) && hp.v41 && stages.size() >= 2 && pctl != nullptr;
    // moex (#76): opt-in STRATA_DSV41_EG_FIRST=1, a layer-major call's engram reads first (Impl::eg_first).  Off: with the
    // record reads held while each fetch ran, the cold 44K read took 650.4 s against 204.5 s without (the drive in its slow
    // mode, 2026-10-21) - every read waited behind the slowest
    static const bool eg_first_env = std::getenv("STRATA_DSV41_EG_FIRST") && std::getenv("STRATA_DSV41_EG_FIRST")[0] == '1';
    eg_first = lm_k > 0 && eg && eg_first_env;
    struct EgFirstEnd {
        Engine::Impl& m;
        ~EgFirstEnd() { m.eg_first = false; }
    } eg_first_end{*this};
    // moex (DSV41_PLAN §64): opt-in STRATA_DSV41_EG_URGENT=1, any call's record reads held while a stage waits for its
    // chunk's engram rows (Impl::eg_urgent)
    static const bool eg_urgent_env = std::getenv("STRATA_DSV41_EG_URGENT") && std::getenv("STRATA_DSV41_EG_URGENT")[0] == '1';
    eg_urgent = eg && eg_urgent_env;
    // moex (DSV41_PLAN §30), opt-in STRATA_DSV41_TAIL_BATCH=1: the last stage defers the call's tails (two or more pieces: a
    // chunk with rows in the last replay_tail before a read start, as the chunk loop below cuts them) to one run after its
    // chunks, each replayed layer streamed once for all of them (Stage::prompt_tails); the lend gives it up without the room
    static const bool tb_env = std::getenv("STRATA_DSV41_TAIL_BATCH") && std::getenv("STRATA_DSV41_TAIL_BATCH")[0] == '1';
    for (auto& gp : stages) {
        gp->tail_batch = false;
        gp->tail_cap = 0;
    }
    if (tb_env && hp.v41 && replay_layer >= 0 && stages.size() >= 2 && pctl && !dsp && !dfl && !mtp_on && replay_ends.size() >= 2 &&
        replay_layer >= stages.back()->lb && replay_layer < stages.back()->le) {
        int pieces = 0, rows = 0, at_cut = 0;
        size_t pj = 0;
        for (int c0 = 0; c0 < n; ++pj) {
            int T = pj < plan.size() ? plan[pj] : std::min(TM, n - c0);
            for (int64_t cut : pctl->cuts) {
                const int64_t rel = cut - pos0 - c0;
                if (rel > 0 && rel < T) T = (int) rel;
            }
            const int64_t p0 = pos0 + c0;
            const auto it = std::upper_bound(replay_ends.begin(), replay_ends.end(), p0);
            if (it != replay_ends.end() && *it - replay_tail - p0 < T) {
                ++pieces;
                rows += T - (int) std::clamp<int64_t>(*it - replay_tail - p0, 0, T);
                at_cut += p0 + T == *it;
            }
            c0 += T;
        }
        stages.back()->tail_batch = pieces >= 2;
        stages.back()->tail_cap = pieces;
        stages.back()->tail_rows_cap = rows;
        stages.back()->tail_snap_cap = at_cut;
    }
    const auto t_lend = clk::now();
    for (size_t si = 0; si < stages.size(); ++si) {
        Stage& g = *stages[si];
        g.lm_k = si == 0 ? lm_k : 0;
        g.lm_win = si == 0 ? lm_w : 0;
        if (!g.lend(TM, sub, g.lm_k > 0 ? lm_ring : ring, err)) {
            for (auto& g2 : stages) {
                g2->give_back_slots();
                g2->lm_k = 0;
                g2->lm_win = 0;
            }
            return false;
        }
    }
    if (rio) rio->us_lend.fetch_add((int64_t) (ms_since(t_lend) * 1000.0), std::memory_order_relaxed);
    warm_lend();   // the lent experts without a RAM copy: read in while the chunks run, for the refill
    if (route_bias_prompt) route_pen_update();   // the opt-in prompt route bias: the residency after the lend
    const size_t row_bytes = (size_t) hp.hc * hp.n_embd * sizeof(float);
    const size_t pre_bytes = hp.v41 ? 24 * sizeof(float) : 0;   // DeepSeek-V4.1: a row's carried pre-mix rides after the rows
    for (size_t si = 0; si + 1 < stages.size(); ++si) {   // the hand-off double buffers, a chunk's rows
        Stage& g = *stages[si];
        if (g.h_hand_rows < TM) {
            g.h_hand_rows = 0;
            for (float*& hb : g.h_hand) {
                host_free(hb);
                if (!(hb = static_cast<float*>(host_buf((size_t) TM * (row_bytes + pre_bytes))))) {
                    for (auto& g2 : stages) g2->give_back_slots();
                    return fail(err, "hand-off buffers (out of memory)");
                }
            }
            g.h_hand_rows = TM;
        }
    }
    if (lm_w > 0 && lm_lend) {   // (§63 step 2) the rows and engram rows of chunk c in lent block c
        Stage& g0 = *stages[0];
        const size_t bytes = (size_t) TM * (row_bytes + pre_bytes + (size_t) hp.idx_topk * sizeof(int32_t) +
                                            (g0.cand_rows() ? lm_cand_row(*this) : 0));
        if (!g0.lmh_lent)
            for (uint8_t* hb : g0.lmh) host_free(hb);   // (a host window's own rows from an earlier call)
        g0.lmh.clear();
        g0.lmh_bytes = 0;
        g0.lmh_lent = true;
        const size_t eg_at = (bytes + 255) & ~(size_t) 255;
        for (uint8_t* b : lm_blk) {
            g0.lmh.push_back(b);
            if (eg) {
                void* dp = nullptr;
                if (cudaHostGetDevicePointer(&dp, b + eg_at, 0) != cudaSuccess || !dp) {
                    cudaGetLastError();
                    for (auto& g2 : stages) g2->give_back_slots();
                    return fail(err, "a host window's engram rows: a lent block is not mapped on the GPU");
                }
                g0.egc.push_back(b + eg_at);
                g0.egc_d.push_back(static_cast<uint8_t*>(dp));
            }
        }
        g0.lmh_bytes = bytes;
        g0.egc_cap = TM;
    } else if (lm_w > 0) {   // moex (§63): each chunk of a host window its pinned rows - [R | pre-mix | top-k lists | candidate
        Stage& g0 = *stages[0];   // masks], T rows each: the first two the hand-off's layout, so the next stage reads them where
        const size_t bytes = (size_t) TM * (row_bytes + pre_bytes + (size_t) hp.idx_topk * sizeof(int32_t) +   // they are
                                            (g0.cand_rows() ? lm_cand_row(*this) : 0));
        if (g0.lmh_lent) {   // (never: the call that lent them gave them back and cleared them)
            g0.lmh.clear();
            g0.lmh_lent = false;
        }
        if ((int) g0.lmh.size() < lm_w || g0.lmh_bytes < bytes) {
            for (uint8_t* hb : g0.lmh) host_free(hb);
            g0.lmh.assign((size_t) lm_w, nullptr);
            g0.lmh_bytes = 0;
            for (uint8_t*& hb : g0.lmh)
                if (!(hb = static_cast<uint8_t*>(host_buf(bytes)))) {
                    for (uint8_t* p : g0.lmh) host_free(p);
                    g0.lmh.clear();
                    for (auto& g2 : stages) g2->give_back_slots();
                    return fail(err, "a host window's rows (out of memory)");
                }
            g0.lmh_bytes = bytes;
        }
    } else if (lm_k > 0) {   // moex (#76): a window's chunks handed on together - each chunk's rows a pinned buffer of its own: the
        Stage& g0 = *stages[0];   // chunk loop's two (unused in a layer-major call), then ones of the window's
        if ((int) g0.h_lm.size() < std::max(0, lm_k - 2) || g0.h_lm_rows < TM) {
            for (float* hb : g0.h_lm) host_free(hb);
            g0.h_lm.assign((size_t) std::max(0, lm_k - 2), nullptr);
            g0.h_lm_rows = 0;
            for (float*& hb : g0.h_lm)
                if (!(hb = static_cast<float*>(host_buf((size_t) TM * (row_bytes + pre_bytes))))) {
                    for (auto& g2 : stages) g2->give_back_slots();
                    return fail(err, "a layer-major window's hand-off buffers (out of memory)");
                }
            g0.h_lm_rows = TM;
        }
    }
    for (auto& gp : stages) {
        gp->st.prefill_chunk = TM;
        gp->st.prefill_ring = gp->lm_k > 0 ? lm_ring : ring;
        gp->run_pos = pos0;   // moex (#78): every stage's running state is at the call's start
    }
    // moex (DSV41_BACKLOG #81), diagnostics on: a line every 20 s while a read of 4096 tokens or more runs - each GPU's chunk or
    // window, layer, stream and reads ahead, the window's engram fetches, the drive - so a read that stalls says where (the
    // statistics' line comes at its end, and only after a chunk is done)
    static const bool hb_quiet = std::getenv("STRATA_X_QUIET") && std::getenv("STRATA_X_QUIET")[0] == '1';
    std::atomic<bool> hb_stop{false};
    std::thread hb_th;
    struct HbJoin {
        std::atomic<bool>& s;
        std::thread& t;
        ~HbJoin() {
            s = true;
            if (t.joinable()) t.join();
        }
    } hb_join{hb_stop, hb_th};
    hb_win = -1;
    hb_eg_done = 0;
    hb_eg_queued = 0;
    for (auto& gp : stages) {
        gp->hb_il = -1;
        gp->hb_c = -1;
        gp->hb_issued = 0;
        gp->hb_total = 0;
        gp->hb_ahead = 0;
    }
    // moex (DSV41_PLAN §18): the call's progress lines (a read of 4096 tokens or more, under a PromptCtl with `work`): one as
    // each (chunk, layer) is done (pl_add), and one every 30 s while this process reads from storage - in the drive's slow
    // mode one layer can outlast the server's silence limit (300 s; the 670 s read of 2026-10-21 printed nothing), while an
    // engine that is stuck reads nothing and still meets it
    pl_on = pctl && pctl->work && n >= 4096;
    pl_done = 0;
    pl_total = (int64_t) plan.size() * hp.n_layer;
    const bool hb_on = !hb_quiet && n >= 4096;
    if (hb_on || pl_on)
        hb_th = std::thread([this, &hb_stop, n, lm_k, hb_on] {
            static const std::string disk = hb_disk_path(opt.pack);
            const auto t_start = clk::now();
            HbDisk d0 = hb_disk(disk);
            auto tp = clk::now();
            int64_t io0 = io_read_bytes();
            for (int tick = 1;; ++tick) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (hb_stop.load()) break;
                if (pl_on && tick % 300 == 0) {   // 30 s
                    const int64_t io1 = io_read_bytes();
                    if (io1 > io0) pctl->work(std::min(pl_done.load(), pl_total), pl_total);
                    io0 = io1;
                }
                if (!hb_on || tick % 200 != 0) continue;   // the heartbeat: 20 s, diagnostics on
                const HbDisk d1 = hb_disk(disk);
                const double dt = ms_since(tp) / 1000.0;
                tp = clk::now();
                char drive[200] = "";
                if (d0.ok && d1.ok && dt > 0) {
                    auto ms32 = [](unsigned long long a, unsigned long long b) { return (double) ((b - a) & 0xffffffffull); };
                    const unsigned long long ios = d1.rd_ios - d0.rd_ios;
                    std::snprintf(drive, sizeof drive, "; the drive: %.0f MB/s, %.1f ms a request, busy %.0f%%, %llu in flight",
                                  (double) (d1.rd_sec - d0.rd_sec) * 512.0 / 1e6 / dt, ios ? ms32(d0.rd_ms, d1.rd_ms) / (double) ios : 0.0,
                                  100.0 * ms32(d0.io_ms, d1.io_ms) / (dt * 1e3), d1.in_flight);
                }
                d0 = d1;
                std::string gpus;
                for (const auto& gp : stages) {
                    char one[200];
                    std::snprintf(one, sizeof one, "; GPU %d at layer %d%s, its stream %zu of %zu records issued, %d read ahead", gp->dev,
                                  gp->hb_il.load(), gp->lm_k > 0 ? (" chunk " + std::to_string(gp->hb_c.load())).c_str() : "",
                                  gp->hb_issued.load(), gp->hb_total.load(), gp->hb_ahead.load());
                    gpus += one;
                }
                char eg[96] = "";
                if (lm_k > 0) std::snprintf(eg, sizeof eg, "; the engram rows: %d of %d chunks fetched", hb_eg_done.load(), hb_eg_queued.load());
                fprintf(stderr, "strata-x: prompt heartbeat at %.0f s (%d tokens): %s %d of %d%s%s%s\n", ms_since(t_start) / 1000.0, n,
                        lm_k > 0 ? "layer-major window" : "chunk", hb_win.load() + 1, hb_nwin.load(), gpus.c_str(), eg, drive);
            }
        });
    const auto t0 = clk::now();
    // stage si runs chunk c; hands its rows on and starts stage si + 1's run of the chunk on its own thread
    std::vector<std::future<bool>> next(stages.size());
    std::vector<std::string> errs(stages.size());
    std::vector<int> hb(stages.size(), 0);
    std::function<bool(size_t, const int32_t*, const float*, int, int64_t, int)> run =
        [&](size_t si, const int32_t* tk, const float* hin, int T, int64_t p0, int c0) -> bool {
        Stage& g = *stages[si];
        const int md = mode_of(T);   // a stage runs its chunks one after the other: its flags are this chunk's
        g.cpu_moe = md == 1;
        g.share = md == 2;
        if (si == 0) n_mode[md]++;
        const int n_next = g.last ? std::max(0, std::min(T, n_avail - c0 - 1)) : 0;
        g.dfl_par = hb[si];   // DFlash: this chunk's tails go to the hand-off buffer's twin
        // Hy4-flash's Engram: the chunk's rows for this stage's layers, queued before its layers (toks: the whole prompt's)
        if (heg && !hy4_eg_chunk(g, toks + c0, p0, T, errs[si])) return false;
        if (!g.prompt_chunk(tk, hin, T, p0, g.last ? toks + c0 + 1 : nullptr, n_next, errs[si])) return false;
        if (g.chunk_stopped) {   // moex (#78): a stop inside the chunk - no cut, no progress, nothing handed on
            pctl->stopped = true;
            return true;
        }
        if (pctl) {   // the serving layer: a checkpoint position this stage has reached, the whole pipeline's progress
            const int64_t pe = p0 + T;
            if (pctl->on_cut && !g.tail_batch && std::binary_search(pctl->cuts.begin(), pctl->cuts.end(), pe))   // (§30: the
                pctl->on_cut((int) si, pe);                                                                      // tails' run's)
            if (g.last) {
                pctl->done = pe;
                if (pctl->progress && !pctl->progress(pe)) pctl->stopped = true;
            }
        }
        if (si + 1 == stages.size()) return true;
        float* hbuf = g.h_hand[hb[si]];
        cudaMemcpyAsync(hbuf, g.R, (size_t) T * row_bytes, cudaMemcpyDeviceToHost, g.s);
        if (pre_bytes) cudaMemcpyAsync(reinterpret_cast<uint8_t*>(hbuf) + (size_t) T * row_bytes, g.cpre, (size_t) T * pre_bytes,
                                       cudaMemcpyDeviceToHost, g.s);
        if (cudaStreamSynchronize(g.s) != cudaSuccess) {
            errs[si] = "the layer split's hand-off failed";
            return false;
        }
        if (next[si + 1].valid() && !next[si + 1].get()) return false;   // the next stage's previous chunk
        if (stop_layers && pctl->stopped.load()) return true;   // moex (#78): after a stop nothing more is handed on
        const float* fin = g.h_dfl[hb[si]];   // DFlash: this chunk's tails of this stage's target layers (null: none)
        next[si + 1] = std::async(std::launch::async, [&, si, hbuf, fin, T, p0, c0] {
            stages[si + 1]->dfl_in = fin;
            return run(si + 1, nullptr, hbuf, T, p0, c0);
        });
        hb[si] ^= 1;
        return true;
    };
    bool ok = true;
    size_t pi = 0;
    if (lm_k > 0) {   // moex (DSV41_BACKLOG #76): windows of lm_k chunks through the first stage, layer by layer
        struct Ck {
            int c0, T;
        };
        std::vector<Ck> cks;   // the chunks as the chunk loop below cuts them
        for (int c0 = 0; c0 < n; ++pi) {
            int T = pi < plan.size() ? plan[pi] : std::min(TM, n - c0);
            for (int64_t cut : pctl->cuts) {
                const int64_t rel = cut - pos0 - c0;
                if (rel > 0 && rel < T) T = (int) rel;
            }
            cks.push_back({c0, T});
            c0 += T;
        }
        Stage& g0 = *stages[0];
        std::future<bool> s1;   // the next stage's run of the window before: its chunks one after the other
        const size_t wk = (size_t) (lm_w > 0 ? lm_w : lm_k);   // (moex, §63: a host window's chunks)
        for (size_t a = 0; a < cks.size() && ok; a += wk) {
            if (pctl->stopped.load() || (pctl->stop && pctl->stop())) {
                pctl->stopped = true;
                break;
            }
            const size_t b = std::min(cks.size(), a + wk);
            hb_win = (int) (a / wk);
            hb_nwin = (int) ((cks.size() + wk - 1) / wk);
            std::vector<Stage::LmChunk> w;
            for (size_t i = a; i < b; ++i) w.push_back({toks + cks[i].c0, cks[i].T, pos0 + cks[i].c0});
            g0.lm_next.clear();   // the next window (its engram rows fetched while this one runs)
            for (size_t i = b; i < std::min(cks.size(), b + wk); ++i)
                g0.lm_next.push_back({toks + cks[i].c0, cks[i].T, pos0 + cks[i].c0});
            set_dev(g0.dev);
            // moex (DSV41_PLAN §37), STRATA_DSV41_TAIL_AHEAD=1: the last window - the deferred tails' records read ahead meanwhile
            static const bool tail_ahead_env = std::getenv("STRATA_DSV41_TAIL_AHEAD") && std::getenv("STRATA_DSV41_TAIL_AHEAD")[0] == '1';
            g0.lm_release = false;
            if (tail_ahead_env && b == cks.size() && rio && rio->keep_on && stages.back()->tail_batch && replay_layer >= 0) {
                tail_ahead_start(replay_layer, stages.back()->le);
                g0.lm_release = true;
            }
            // (moex, §63) a host window's rows come back into the buffers the next stage reads the window before from: it finishes
            // that one first - with two sets (LM_HOST=2) the window before's are the other set's, and the next stage runs that
            // window while this one does
            const size_t hoff = lm_dbl ? ((a / wk) % 2) * wk : 0;
            g0.lmh_off = (int) hoff;
            g0.lmh_wk = lm_dbl ? (int) wk : 0;
            if (lm_w > 0 && !lm_dbl && s1.valid() && !s1.get()) {
                ok = false;
                break;
            }
            const bool w_ok = g0.prompt_window(w, errs[0]);
            g0.lm_release = false;
            if (!w_ok) {
                ok = false;
                break;
            }
            if (g0.chunk_stopped) {
                pctl->stopped = true;
                break;
            }
            // the window's checkpoints: this stage's part from its copy of the state at each cut in it
            for (size_t i = a; i < b; ++i) {
                const int64_t pe = pos0 + cks[i].c0 + cks[i].T;
                if (!pctl->on_cut || !std::binary_search(pctl->cuts.begin(), pctl->cuts.end(), pe)) continue;
                if (const Stage::RunSnap* r = g0.run_snap_at(pe)) {
                    snap_src = r->d;
                    snap_src_si = 0;
                    pctl->on_cut(0, pe);
                    snap_src = nullptr;
                    snap_src_si = -1;
                }
            }
            // the hand-off: the next stage must be done with the window before (it reads the same buffers)
            if (s1.valid() && !s1.get()) {
                ok = false;
                break;
            }
            if (pctl->stopped.load()) break;
            const bool hostw = b - a > g0.lmw.size();   // (moex, §63: its rows came back to host memory in the hand-off's layout)
            auto hand = [&, hoff](size_t k) {   // chunk k of the window's buffer (a host call's: its host rows, whatever its size)
                return lm_w > 0 ? reinterpret_cast<float*>(g0.lmh[hoff + k]) : k < 2 ? g0.h_hand[k] : g0.h_lm[k - 2];
            };
            for (size_t i = a; i < b && !hostw; ++i) {
                const Stage::LmRows& rw = g0.lmw[i - a];
                float* hbuf = hand(i - a);
                cudaMemcpyAsync(hbuf, rw.R, (size_t) cks[i].T * row_bytes, cudaMemcpyDeviceToHost, g0.s);
                cudaMemcpyAsync(reinterpret_cast<uint8_t*>(hbuf) + (size_t) cks[i].T * row_bytes, rw.cpre, (size_t) cks[i].T * pre_bytes,
                                cudaMemcpyDeviceToHost, g0.s);
            }
            if (cudaStreamSynchronize(g0.s) != cudaSuccess) {
                errs[0] = "the layer split's hand-off failed";
                ok = false;
                break;
            }
            s1 = std::async(std::launch::async, [&, a, b, hand] {
                for (size_t i = a; i < b; ++i) {
                    if (!run(1, nullptr, hand(i - a), cks[i].T, pos0 + cks[i].c0, cks[i].c0)) return false;
                    if (pctl->stopped.load()) return true;
                }
                return true;
            });
        }
        if (s1.valid() && !s1.get()) ok = false;
        g0.lm_next.clear();
        g0.eg_lm_drop();   // a fetch for a window this call did not read: waited out; no buffer names a window any more
        if (eg_first)
            fprintf(stderr, "strata-x: layer-major: the record reads waited %.1f s for the engram's (STRATA_DSV41_EG_FIRST)\n",
                    us_eg_first.exchange(0) / 1e6);
        if (rio && rio->deep_cap > 0)   // (§63)
            fprintf(stderr, "strata-x: layer-major, one window: its read-ahead took %lld cache slots past the share (at most %d "
                            "held)\n", (long long) rio->adm_deep.exchange(0), rio->deep_cap);
    }
    for (int c0 = 0; c0 < n && ok && lm_k == 0; ++pi) {
        int T = pi < plan.size() ? plan[pi] : std::min(TM, n - c0);
        if (pctl) {   // a chunk ends exactly at each checkpoint position (the plan's do); a stop ends the reading between chunks
            for (int64_t cut : pctl->cuts) {
                const int64_t rel = cut - pos0 - c0;
                if (rel > 0 && rel < T) T = (int) rel;
            }
            if (pctl->stopped.load() || (pctl->stop && pctl->stop())) {
                pctl->stopped = true;
                break;
            }
        }
        set_dev(stages[0]->dev);
        if (eg || heg) {   // moex (DSV41_BACKLOG #51; Hy4-flash 2026-11-19): the next chunk for stage 0's engram rows ahead (the plan's
                           // length: an upper bound)
            Stage& g0 = *stages[0];
            const int64_t nx = c0 + T;
            g0.eg_next_T = nx < n ? (int) std::min<int64_t>(pi + 1 < plan.size() ? plan[pi + 1] : TM, n - nx) : 0;
            g0.eg_next_toks = toks + nx;
            g0.eg_next_pos = pos0 + nx;
        }
        hb_win = (int) pi;
        hb_nwin = (int) plan.size();
        ok = run(0, toks + c0, nullptr, T, pos0 + c0, c0);
        c0 += T;
    }
    if (eg) {   // moex (#51): a fetch for a chunk this call did not read is waited out and dropped
        Stage& g0 = *stages[0];
        g0.eg_next_T = 0;
        if (g0.eg_nx.valid()) g0.eg_nx.get();
        g0.eg_nx_pos = -1;
        if (n >= 2048) {
            char held[160] = "";
            if (eg_urgent)
                std::snprintf(held, sizeof held, "; the record reads held %.1f s meanwhile (STRATA_DSV41_EG_URGENT)", us_eg_urgent.load() / 1e6);
            fprintf(stderr, "strata-x: prompt path: %d tokens waited %.1f s for their engram rows (%s)%s\n", n,
                    us_eg_wait.exchange(0) / 1e6, eg_ahead() ? "each chunk's fetched while the one before ran" : "fetched with each chunk",
                    held);
        } else {
            us_eg_wait.store(0);
        }
        us_eg_urgent.store(0);
    }
    // drain the chain
    for (size_t si = 1; si < stages.size(); ++si)
        if (next[si].valid() && !next[si].get()) ok = false;
    if (heg) {   // moex (2026-11-19): Hy4-flash's gatherings ahead into the row cache - any left (a stop, a cut) waited out and dropped
        stages[0]->eg_next_T = 0;
        std::lock_guard<std::mutex> lk(eg_pf_mu);
        for (auto* mp : {&eg_pf, &eg_pf0})
            for (auto& kv : *mp)
                if (kv.second.valid()) kv.second.wait();
        eg_pf.clear();
        eg_pf0.clear();
    }
    reset_dev_cache();
    for (auto& gp : stages) {
        cudaSetDevice(gp->dev);
        if (cudaStreamSynchronize(gp->s) != cudaSuccess) ok = false;
    }
    // moex (DSV41_PLAN §30): the last stage's deferred tails, then its checkpoints at the call's cuts (their state copies, the
    // replayed layers completed by the tails' run); a call stopped before or inside them keeps none of its cuts (a checkpoint
    // without every stage is not kept) and no exact state
    if (!stages.empty()) {
        Stage& gl = *stages.back();
        const int sl = (int) stages.size() - 1;
        if (ok && gl.tail_batch && !gl.tails.empty()) {
            // a call stopped among its chunks runs the tails it has (each piece's chunk finished its layers before replay_layer)
            // to the end first: the cuts it passed keep their checkpoints, as without the deferral; its live state goes
            const bool was_stopped = pctl->stopped.load();
            tail_ahead_stop();   // (moex, §37: the reads still queued promoted: the tails' run reads the records from the cache)
            set_dev(gl.dev);
            const auto t_tails = clk::now();
            bool ran = false;
            if (!gl.prompt_tails(errs[(size_t) sl], was_stopped)) ok = false;
            else ran = !gl.chunk_stopped;
            if (cudaStreamSynchronize(gl.s) != cudaSuccess) ok = false;
            fprintf(stderr, "strata-x: the deferred tails: %zu pieces through layers %lld-%lld in %.1f s (STRATA_DSV41_TAIL_BATCH)%s\n",
                    gl.tails.size(), (long long) replay_layer, (long long) gl.le - 1, ms_since(t_tails) / 1000.0,
                    was_stopped ? ", the call stopped: its cuts' checkpoints first" : "");
            if (ok && ran) {
                for (const Stage::TailRows& t : gl.tails)
                    if (t.snap && pctl->on_cut && std::binary_search(pctl->cuts.begin(), pctl->cuts.end(), t.cut)) {
                        snap_src = t.snap->d;
                        snap_src_si = sl;
                        pctl->on_cut(sl, t.cut);
                        snap_src = nullptr;
                        snap_src_si = -1;
                    }
            }
            if (ok && (was_stopped || !ran)) {   // the live state: the stop's way back (#78) went with the tails' loan
                pctl->stopped = true;
                pctl->torn = true;
                fprintf(stderr, "strata-x: a stop with deferred tails: the state is exact nowhere (the live tokens go; the next read "
                                "starts at a checkpoint)\n");
            }
        }
        gl.tail_batch = false;
        gl.tails.clear();
        reset_dev_cache();
    }
    // moex (DSV41_BACKLOG #78): after a stop every stage's running state goes back to where the last stage got - the start
    // of the chunk it did not finish, which each stage ahead of it kept (a stage that never got there keeps its own)
    const bool stopped_layers = ok && stop_layers && pctl->stopped.load();
    if (stopped_layers) {
        const int64_t at = std::max<int64_t>(pos0, pctl->done);
        for (auto& gp : stages) {
            if (gp->run_pos == at) continue;
            cudaSetDevice(gp->dev);
            if (gp->run_snap_put(at)) {
                fprintf(stderr, "strata-x: after the stop, GPU %d's running state is back at %lld\n", gp->dev, (long long) at);
            } else {
                pctl->torn = true;
                fprintf(stderr, "strata-x: a stop inside a chunk: GPU %d kept no copy of its state at %lld - the state is exact "
                                "nowhere (the live tokens go)\n", gp->dev, (long long) at);
            }
            if (cudaStreamSynchronize(gp->s) != cudaSuccess) ok = false;
        }
    }
    // moex (§37): the tails' read-ahead ends before the refill reads the SSD cache's map without its lock - a call whose last
    // window stopped or failed never reached the tails' run, which stops it (the audit of 2026-10-24); a no-op when not running
    tail_ahead_stop();
    const auto t_refill = clk::now();
    for (auto& gp : stages) gp->refill_issue();   // every stage's copies queued, then waited for
    for (auto& gp : stages) gp->refill_finish();
    for (auto& gp : stages) {   // moex (#76): the window's buffers were the loan's
        gp->lm_k = 0;
        gp->lm_win = 0;
    }
    if (rio) rio->us_refill.fetch_add((int64_t) (ms_since(t_refill) * 1000.0), std::memory_order_relaxed);
    reset_dev_cache();
    if (!ok) {
        for (const auto& e : errs)
            if (!e.empty()) return fail(err, e);
        return fail(err, "the prompt path failed");
    }
    stages[0]->st.ms_prefill += ms_since(t0);
    // the paths' costs (cpu_read_limit, share_limit): a token through the CPU-assisted read, a shared read, a chunk through the
    // stream
    const double ms = ms_since(t_all);
    const int nc = n_mode[0] + n_mode[1] + n_mode[2];   // a call whose chunks all took one mode measures it
    if (stopped_layers) {   // moex (#78): a chunk cut short by a stop measures nothing
    } else if (n_mode[1] == nc && n >= 128) {
        const double v = ms / n;
        cpu_ms_tok = cpu_ms_tok > 0.0 ? cpu_ms_tok + 0.3 * (v - cpu_ms_tok) : v;
    } else if (n_mode[2] == nc && n >= 32) {
        share_observe((double) n, ms);
    } else if (n_mode[0] == nc && nc == 1 && n <= 4096) {   // one small chunk: the stream's (nearly) fixed cost
        stream_chunk_ms = stream_chunk_ms > 0.0 ? stream_chunk_ms + 0.3 * (ms - stream_chunk_ms) : ms;
    }
    static const bool share_log = std::getenv("STRATA_X_SHARE_LOG") && std::getenv("STRATA_X_SHARE_LOG")[0] == '1';
    if (share_log) {
        EngineStats a;
        for (auto& gp : stages) {
            a.prefill_cpu_experts += gp->st.prefill_cpu_experts;
            a.prefill_share_experts += gp->st.prefill_share_experts;
        }
        {   // the experts in neither VRAM nor RAM (read from the GGUF when used), and the exchange blobs left
            int64_t file_only = 0;
            for (int64_t key = hp.n_dense * hp.n_expert; key < n_xl * hp.n_expert; ++key)
                if (resident[(size_t) key] < 0 && (ram_off.empty() || ram_off[(size_t) key] < 0)) ++file_only;
            fprintf(stderr, "strata-x: experts in neither VRAM nor RAM: %lld; exchange blobs free: %zu\n", (long long) file_only,
                    xpool.size());
        }
        fprintf(stderr, "strata-x: read %d tokens in %.0f ms (%s) | share: limit %lld, links %s MB/s, pool %.0f MB/s a unit; "
                        "experts to the pool so far %lld of %lld\n",
                n, ms, [&] {
                    std::string t;
                    const char* nm[3] = {"stream", "CPU read", "shared"};
                    for (int i = 0; i < 3; ++i)
                        if (n_mode[i] > 0) t += (t.empty() ? "" : " + ") + std::to_string(n_mode[i].load()) + " " + nm[i];
                    return t;
                }().c_str(), (long long) share_limit(),
                [&] {
                    std::string t;
                    for (auto& gp : stages) t += (t.empty() ? "" : "/") + std::to_string((int) (gp->share_link / 1e3));
                    return t;
                }().c_str(),
                share_cpu_rate.load() / 1e3, (long long) a.prefill_cpu_experts, (long long) a.prefill_share_experts);
    }
    return true;
}

void Engine::Impl::share_observe(double n, double ms, double w) {
    constexpr double keep = 0.85;   // ~6 reads' memory: the depth a conversation reads at moves
    for (double& v : sh_fit) v *= keep;
    sh_fit[0] += w;
    sh_fit[1] += w * n;
    sh_fit[2] += w * n * n;
    sh_fit[3] += w * ms;
    sh_fit[4] += w * n * ms;
}

int64_t Engine::Impl::share_limit() const {
    // where a shared read of n tokens (a + b n) costs a streamed chunk.  The fit's priors: MiMo here, estimated 10-09 from the
    // link and pool speeds, on the slow side - ~5 s at 256 tokens, ~9 s at 2048 (every layer touches ~220 non-resident experts
    // once n is past ~100: their pool or copy time is the floor; the tokens add their attention and pool rows)
    constexpr double kN0 = 256.0, kMs0 = 5000.0, kN1 = 2048.0, kMs1 = 9000.0, kStreamChunkPrior = 12000.0;
    double f[5];
    for (int i = 0; i < 5; ++i) f[i] = sh_fit[i];
    const double wp = 0.3;   // the priors' weight: two observations' worth once the fit has decayed to a few reads
    f[0] += 2 * wp;
    f[1] += wp * (kN0 + kN1);
    f[2] += wp * (kN0 * kN0 + kN1 * kN1);
    f[3] += wp * (kMs0 + kMs1);
    f[4] += wp * (kN0 * kMs0 + kN1 * kMs1);
    const double det = f[0] * f[2] - f[1] * f[1];
    double b = det > 1e-9 ? (f[0] * f[4] - f[1] * f[3]) / det : (kMs1 - kMs0) / (kN1 - kN0);
    b = std::max(b, 0.1);   // a read never gets cheaper with more tokens
    const double a = (f[3] - b * f[1]) / f[0];
    // a streamed chunk: as measured (one small chunk through the stream), else the copies it makes - every non-resident
    // expert of every MoE layer - at the links' speeds the shared layers measured (copies bound it on this box: MiMo's
    // 86 GB at 6.3 / 8.1 GB/s = 12.2 s, its measured chunks ~12 s), else MiMo's measured chunk
    double sc = stream_chunk_ms;
    if (sc <= 0.0) {
        const int64_t E = hp.n_expert;
        double est = 0.0;
        for (const auto& gp : stages) {
            const Stage& g = *gp;
            if (g.share_link <= 0.0) {
                est = 0.0;
                break;
            }
            double bytes = 0.0;
            for (int64_t il = g.lb; il < g.le; ++il) {
                if (!hp.is_moe(il)) continue;
                int64_t nr = 0;
                for (int64_t e = 0; e < E; ++e) nr += resident[(size_t) (il * E + e)] < 0;
                bytes += (double) nr * (double) L[(size_t) il].fmt.bytes;
            }
            est += bytes / g.share_link;
        }
        sc = est > 0.0 ? est : kStreamChunkPrior;
    }
    return std::clamp<int64_t>((int64_t) ((sc - a) / b), 64, 8192);
}

int64_t Engine::Impl::cpu_read_limit() const {
    // where the CPU-assisted read of n tokens (~n x cpu_ms_tok) costs one small streamed chunk; until this process has
    // measured both, MiMo's on this box (10-08: the CPU read 25.7 ms a token over 8.4K tokens; a small chunk's stream ~12-14 s)
    constexpr double kCpuTokPrior = 25.7, kStreamChunkPrior = 12000.0;
    const double ct = cpu_ms_tok > 0.0 ? cpu_ms_tok : kCpuTokPrior, sc = stream_chunk_ms > 0.0 ? stream_chunk_ms : kStreamChunkPrior;
    return std::clamp<int64_t>((int64_t) (sc / ct), 64, 8192);
}

}  // namespace strata::x
