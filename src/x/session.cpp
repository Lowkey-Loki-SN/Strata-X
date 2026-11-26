// src/x/session.cpp - SERVING: one request on the live sequence (Strata's generate.cpp --serve, STRATA_QWEN_DESIGN.md
// 1.5 and 5, for GLM).
//
//   * WHAT THE ENGINE HOLDS splits in two (Strata 1.5): the POSITIONAL state - the latent caches, the pooled indexer keys -
//     is reused in place and never copied ("rewinding to a position just means writing from there again"); the RUNNING state
//     - every KDA layer's recurrent matrix and conv history, every DSA layer's open-pool indexer rings - is what a
//     checkpoint copies (~146 MB, pinned host memory).
//   * REUSE is an exact stored prefix, not a fuzzy longest-common-prefix: the live token list when it is a prefix of the
//     prompt (length <= n - 1: "the last prompt token always starts the next verify window"), or the longest checkpoint
//     whose ids are a prefix; the longest wins; otherwise the state is zeroed and the prompt is read from the start.
//   * CHECKPOINTS (Strata's policy): before the last turn token of the prompt; at the root - the first turn boundary at
//     least `root` (2048) tokens in, pinned; every 16384 fresh tokens; six kept, the rest by least recent use.  They are
//     taken DURING reading: the prompt path ends a chunk exactly there and each stage snapshots its own state when it has
//     finished that chunk (the stages run chunks concurrently, so each at its own moment).
//   * SAMPLING is Strata's sampler on the GPU (penalties -> top_k -> top_p -> min_p -> temperature -> pick), Philox keyed
//     by (seed, position): the draw for the token after position p is counter p, so a run is reproducible and independent of
//     how the windows were cut.  Greedy with no penalties is the plain arg-max.
//   * A STOP polled between prompt chunks and decode windows ends the request ("cancel"); an EOS id ends it ("stop") after
//     being emitted; max_new ends it ("length").
//   * IMAGES (moex, GENI): the request's pictures replace the embeddings of their pad runs (Impl::img, read by embd_rows on
//     both the prompt path and the windows); every stored prefix (the live tokens, a checkpoint) remembers the images inside
//     it, and is reused only when they are the same pictures (Strata's ConversationImageKey rule: ids alone are the same for
//     any two pictures of one size).
// What the live sequence is afterwards: the prompt and every generated token except the last (it was sampled, never fed).
#include "engine_impl.hpp"

#include "strata/kernels/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::x {

Checkpoint::~Checkpoint() {
    for (uint8_t* b : blob)
        if (b) cudaFreeHost(b);
}

// ------------------------------------------------------------------ the running state

int64_t Engine::Impl::short_read_limit() const {
    // Strata's short_read: a segment this short reads through decode windows - unless the windows have measured slower than
    // one prompt chunk for it (windows slow down with depth while a chunk costs its streamed experts whatever its length:
    // MiMo at ~100K, 302 tokens through windows 13.4 s, 685 through one chunk 14.4 s)
    // moex (upstream dfd1b0f2): DeepSeek-V4.1 picks the way by the segment's length alone (--short-read, 96 in its config).
    // The timings moved the limit between runs - a drive episode, a cheap chunk - so a 32-95-token follow-up went through
    // decode windows one time and the prompt path (MMQ, other rounding) the next: the same request, other bits
    if (hp.v41) return prompt_short;
    if (rw_ms_tok <= 0.0 || chunk_ms <= 0.0) return prompt_short;
    return std::min<int64_t>(prompt_short, std::max<int64_t>(32, (int64_t) (chunk_ms / rw_ms_tok)));
}

void Stage::state_regions_layer(int64_t il, std::vector<std::pair<void*, size_t>>& out) const {
    const int64_t din = hp.n_head * hp.hd_kda;
    const Layer& l = L[(size_t) il];
    if (l.S) out.push_back({l.S, (size_t) hp.hd_kda * din * sizeof(float)});
    if (l.conv) out.push_back({l.conv, (size_t) 3 * din * (hp.d_conv - 1) * sizeof(float)});
    if (l.ik_ring) out.push_back({l.ik_ring, (size_t) m.ring_rows * hp.idx_dim * sizeof(float)});
    if (l.ig_ring) out.push_back({l.ig_ring, (size_t) m.ring_rows * hp.idx_dim * sizeof(float)});
    // Hy4-flash: a sliding-window layer's MLA rows are a ring (running state, as MiMo's below); a sparse layer's are positional
    if (hp.arch == Arch::hy4 && l.kv_cache && hp.is_swa(il)) out.push_back({l.kv_cache, (size_t) l.kv_rows * 576 * 2});
    // MiMo: a sliding-window layer's ring is running state (it holds only the last positions); a full layer's rows are
    // positional
    if (l.k_cache && hp.is_swa(il)) {
        out.push_back({l.k_cache, (size_t) l.kv_rows * hp.n_kv(il) * hp.hd_k * 2});
        out.push_back({l.v_cache, (size_t) l.kv_rows * hp.n_kv(il) * hp.hd_v * 2});
    }
    // DeepSeek-V4: the raw window's ring and the compressors' state rings hold only the last positions (running state);
    // the compressed and indexer rows are positional (row = block), rewritten as positions return
    if (l.raw) {
        const int r = hp.cratio[(size_t) il];
        out.push_back({l.raw, (size_t) l.raw_rows * hp.hd_k * 2});
        if (r > 0 && l.c_ring_kv) {   // (DeepSeek-V4.1: only the ratio-2 KV sources pool)
            const size_t ring = (size_t) m.dsv4_ring(r) * (r == 4 ? 2 : 1) * hp.hd_k * sizeof(float);
            out.push_back({l.c_ring_kv, ring});
            out.push_back({l.c_ring_sc, ring});
        }
        if (r == 4) {
            const size_t ring = (size_t) m.dsv4_ring(4) * 2 * hp.idx_dim * sizeof(float);
            out.push_back({l.l_ring_kv, ring});
            out.push_back({l.l_ring_sc, ring});
        }
    }
}

size_t Stage::state_layer_offset(int64_t il) const {
    std::vector<std::pair<void*, size_t>> reg;
    for (int64_t j = lb; j < il; ++j) state_regions_layer(j, reg);
    size_t off = 0;
    for (const auto& r : reg) off += r.second;
    return off;
}

void Stage::state_regions(std::vector<std::pair<void*, size_t>>& out) const {
    auto add = [&](int64_t il) { state_regions_layer(il, out); };
    for (int64_t il = lb; il < le; ++il) add(il);
    if (last && m.mtp_on)   // the NextN / MTP blocks' caches and rings live with the head
        for (int64_t il = hp.n_layer; il < m.n_load; ++il) add(il);
    if (last && m.dsp)      // DSpark's blocks' K = V rings (L[base ..], ratio 0: running state)
        for (int64_t il = m.dsp->base; il < m.dsp->base + m.dsp->n; ++il) add(il);
    if (last && m.dmtp)     // DeepSeek-V4's NextN head: its block's K = V ring (L[n_layer], ratio 0)
        add(hp.n_layer);
    if (last && m.dfl)      // MiMo's DFlash drafter: its blocks' K / V rings (the window's rows: running state)
        for (const auto& b : m.dfl->b) {
            const size_t bytes = (size_t) m.dfl->ring * m.dfl->nkv * m.dfl->hd * 2;
            out.push_back({b.kc, bytes});
            out.push_back({b.vc, bytes});
        }
}

void Engine::Impl::snapshot(Checkpoint& c, int si) {
    Stage& g = *stages[(size_t) si];
    std::vector<std::pair<void*, size_t>> reg;
    g.state_regions(reg);
    size_t total = 0;
    for (const auto& r : reg) total += r.second;
    set_dev(g.dev);
    if (c.blob.size() <= (size_t) si) {
        c.blob.resize(stages.size(), nullptr);
        c.bytes.resize(stages.size(), 0);
    }
    if (!c.blob[(size_t) si] && cudaMallocHost(&c.blob[(size_t) si], std::max<size_t>(total, 1)) != cudaSuccess) {
        cudaGetLastError();
        c.blob[(size_t) si] = nullptr;
        c.bytes[(size_t) si] = 0;
        return;
    }
    if (snap_src && snap_src_si == si) {   // moex (#76): a layer-major window's copy of the state at this chunk end
        cudaMemcpyAsync(c.blob[(size_t) si], snap_src, total, cudaMemcpyDeviceToHost, g.s);
    } else {
        size_t off = 0;
        for (const auto& r : reg) {
            cudaMemcpyAsync(c.blob[(size_t) si] + off, r.first, r.second, cudaMemcpyDeviceToHost, g.s);
            off += r.second;
        }
    }
    cudaStreamSynchronize(g.s);
    c.bytes[(size_t) si] = total;
}

// moex (DSV41_BACKLOG #78): the running state's copies a stop inside a prompt chunk puts back (Impl::stop_layers) - the same
// regions as a checkpoint's, device to device in the prompt path's loan

size_t Stage::run_state_bytes() const {
    std::vector<std::pair<void*, size_t>> reg;
    state_regions(reg);
    size_t total = 0;
    for (const auto& r : reg) total += r.second;
    return total;
}

int Stage::run_snap_count() const {
    int k = 1;
    for (const auto& gp : m.stages) k += gp->lb > lb;
    return k;
}

void Stage::run_snap_take(int64_t pos) {
    if (run_snaps.empty()) return;
    RunSnap& r = run_snaps[(size_t) run_snap_next];
    run_snap_next = (run_snap_next + 1) % (int) run_snaps.size();
    std::vector<std::pair<void*, size_t>> reg;
    state_regions(reg);
    size_t off = 0;
    for (const auto& x : reg) {
        cudaMemcpyAsync(r.d + off, x.first, x.second, cudaMemcpyDeviceToDevice, s);
        off += x.second;
    }
    r.pos = pos;
}

Stage::RunSnap* Stage::run_snap_new(int64_t pos) {
    if (run_snaps.empty()) return nullptr;
    RunSnap& r = run_snaps[(size_t) run_snap_next];
    run_snap_next = (run_snap_next + 1) % (int) run_snaps.size();
    r.pos = pos;
    return &r;
}

void Stage::run_snap_layer(int64_t il, RunSnap& r) {
    std::vector<std::pair<void*, size_t>> reg;
    state_regions_layer(il, reg);
    size_t off = state_layer_offset(il);
    for (const auto& x : reg) {
        cudaMemcpyAsync(r.d + off, x.first, x.second, cudaMemcpyDeviceToDevice, s);
        off += x.second;
    }
}

const Stage::RunSnap* Stage::run_snap_at(int64_t pos) const {
    for (const RunSnap& r : run_snaps)
        if (r.pos == pos) return &r;
    return nullptr;
}

bool Stage::run_snap_put(int64_t pos) {
    for (const RunSnap& r : run_snaps) {
        if (r.pos != pos) continue;
        std::vector<std::pair<void*, size_t>> reg;
        state_regions(reg);
        size_t off = 0;
        for (const auto& x : reg) {
            cudaMemcpyAsync(x.first, r.d + off, x.second, cudaMemcpyDeviceToDevice, s);
            off += x.second;
        }
        run_pos = pos;
        return true;
    }
    return false;
}

void Engine::Impl::restore(const Checkpoint& c) {
    eg_note(c.ids.data(), 0, (int64_t) c.ids.size());   // DeepSeek-V4.1's engram hashes the positions before the next read
    if (heg) hy4_eg_note(c.ids.data(), 0, (int) c.ids.size());   // ... and Hy4-flash's
    for (size_t si = 0; si < stages.size(); ++si) {
        Stage& g = *stages[si];
        std::vector<std::pair<void*, size_t>> reg;
        g.state_regions(reg);
        set_dev(g.dev);
        size_t off = 0;
        for (const auto& r : reg) {
            cudaMemcpyAsync(r.first, c.blob[si] + off, r.second, cudaMemcpyHostToDevice, g.s);
            off += r.second;
        }
        cudaStreamSynchronize(g.s);
    }
}

void Engine::Impl::reset_state() {
    for (auto& gp : stages) {
        Stage& g = *gp;
        set_dev(g.dev);
        cudaStreamSynchronize(g.s);
        std::vector<std::pair<void*, size_t>> reg;
        g.state_regions(reg);
        for (const auto& r : reg) cudaMemsetAsync(r.first, 0, r.second, g.s);
        cudaStreamSynchronize(g.s);
    }
    live.clear();
    live_imgs.clear();
}

void Engine::set_checkpoint_policy(std::vector<int32_t> turn_tokens, int keep, int64_t every, int64_t root, int deep) {
    p_->turn_tokens = std::move(turn_tokens);
    p_->ck_keep = keep;
    p_->ck_every = every;
    p_->ck_root = root;
    p_->ck_deep = std::max(0, deep);
}

void Engine::forget() {
    p_->checks.clear();
    p_->live.clear();
    p_->live_imgs.clear();
    p_->live_valid = false;
}

// ------------------------------------------------------------------ sampling

/// A parity probe (moex vision): STRATA_X_FIRST_TOP=N prints a request's first sampled distribution's top N as
/// id:log-probability (llama-server's top_logprobs) to stderr.  `row`: that distribution's logits (device, last stage).
void Engine::Impl::first_top_probe(const float* row, int64_t counter_pos) {
    static const int first_top = std::getenv("STRATA_X_FIRST_TOP") ? std::atoi(std::getenv("STRATA_X_FIRST_TOP")) : 0;
    // moex (DSV41_PLAN §75, for checks): STRATA_X_EVERY_TOP=1 - every sampled row's hash too ("token at position N | logits
    // H"), so two runs that part say at which token they do
    static const bool every_top = [] {
        const char* e = std::getenv("STRATA_X_EVERY_TOP");
        return e && e[0] == '1';
    }();
    if (first_top <= 0 || (!first_top_pending && !every_top)) return;
    const bool first = first_top_pending;
    first_top_pending = false;
    Stage& g = last_stage();
    std::vector<float> lg((size_t) hp.n_vocab);
    cudaMemcpyAsync(lg.data(), row, lg.size() * sizeof(float), cudaMemcpyDeviceToHost, g.s);
    cudaStreamSynchronize(g.s);
    if (!first) {   // (a later token: the hash alone)
        uint64_t h2 = 1469598103934665603ull;
        for (const uint8_t* b = reinterpret_cast<const uint8_t*>(lg.data()), *e = b + lg.size() * sizeof(float); b < e; ++b)
            h2 = (h2 ^ *b) * 1099511628211ull;
        fprintf(stderr, "strata-x: token at position %lld | tok-logits %016llx\n", (long long) counter_pos, (unsigned long long) h2);
        return;
    }
    double mx = -1e300, z = 0;
    for (float v : lg) mx = std::max(mx, (double) v);
    for (float v : lg) z += std::exp((double) v - mx);
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    const int k = std::min<int>(first_top, (int) idx.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
    fprintf(stderr, "strata-x: first-token top at position %lld:", (long long) counter_pos);
    for (int i = 0; i < k; ++i) fprintf(stderr, " %d:%.5f", idx[(size_t) i], (double) lg[(size_t) idx[(size_t) i]] - mx - std::log(z));
    // moex: the whole row's bits as one FNV-1a hash - two runs left in equal states print the same (the bitwise checks)
    uint64_t h = 1469598103934665603ull;
    for (const uint8_t* b = reinterpret_cast<const uint8_t*>(lg.data()), *e = b + lg.size() * sizeof(float); b < e; ++b)
        h = (h ^ *b) * 1099511628211ull;
    fprintf(stderr, " | logits %016llx\n", (unsigned long long) h);
}

/// The token after the window's last row: arg-max when greedy without penalties, Strata's sampler otherwise.
/// `counter_pos` is the position of the FED token (Philox counter); hist[0, hist_n) the most recent tokens, last = newest.
int32_t Engine::Impl::sample_last(const SamplingParams& sp, int64_t counter_pos, const int32_t* hist, int hist_n, std::string& err) {
    Stage& g = last_stage();
    set_dev(g.dev);
    const int T = std::max(1, logit_rows);
    const float* row = g.logits + (size_t) (T - 1) * hp.n_vocab;
    first_top_probe(row, counter_pos);
    const bool greedy = sp.temperature <= 0.0f;
    const bool pen = sp.penalty_last_n > 0 && (sp.penalty_repeat != 1.0f || sp.penalty_freq != 0.0f || sp.penalty_present != 0.0f);
    if (greedy && !pen) {
        argmax_w(row, (int) hp.n_vocab, g.amax, 1, g.s);
        cudaMemcpyAsync(h_amax, g.amax, sizeof(int32_t), cudaMemcpyDeviceToHost, g.s);
    } else {
        kn::SamplerParams p;
        p.top_k = sp.top_k;
        p.full_vocab = sp.top_k <= 0;   // as spec.cpp's chain_of
        p.temp_first = sp.temp_first;
        p.top_p = sp.top_p;
        p.min_p = sp.min_p;
        p.temperature = sp.temperature;
        p.penalty_last_n = pen ? std::min(sp.penalty_last_n, samp_hist_cap) : 0;
        p.penalty_repeat = sp.penalty_repeat;
        p.penalty_freq = sp.penalty_freq;
        p.penalty_present = sp.penalty_present;
        p.seed = sp.seed;
        p.counter = (uint64_t) counter_pos;
        p.greedy = greedy;
        const int* dh = nullptr;
        int hl = 0;
        if (p.penalty_last_n > 0) {
            hl = p.penalty_last_n;
            for (int i = 0; i < hl; ++i) {   // the newest `hl` tokens, -1 in the unused front slots
                const int src = hist_n - hl + i;
                h_hist[i] = src >= 0 ? hist[src] : -1;
            }
            cudaMemcpyAsync(samp_hist, h_hist, (size_t) hl * sizeof(int32_t), cudaMemcpyHostToDevice, g.s);
            dh = samp_hist;
        }
        kn::sample_tokens(row, 1, (int) hp.n_vocab, dh, hl, p, samp_out, g.s);
        cudaMemcpyAsync(h_amax, samp_out, sizeof(int32_t), cudaMemcpyDeviceToHost, g.s);
    }
    if (cudaStreamSynchronize(g.s) != cudaSuccess) {
        err = "sampling failed";
        return -1;
    }
    return h_amax[0];
}

// ------------------------------------------------------------------ a request

namespace {
bool is_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b, size_t n) {   // a[0, n) == b[0, n)
    return a.size() >= n && b.size() >= n && std::equal(a.begin(), a.begin() + (std::ptrdiff_t) n, b.begin());
}
/// The images that start inside [0, L) (both lists sorted by start).
std::vector<ImageKey> images_below(const std::vector<ImageKey>& v, int64_t L) {
    std::vector<ImageKey> out;
    for (const ImageKey& k : v)
        if (k.start < L) out.push_back(k);
    return out;
}
/// moex vision: a stored prefix of length L (whose ids match) holds the same pictures as the request.
bool images_match(const std::vector<ImageKey>& held, const std::vector<ImageKey>& req, int64_t L) {
    return images_below(held, L) == images_below(req, L);
}
/// moex: a request's read is over - the long read's pinned buffers go back to the system.  Impl::prompt grows them to the
/// largest read and kept them until the engine stopped: a layer-major window's hand-off rows (h_lm) and engram rows
/// (eg_lm_h), and the chunk hand-off rows (h_hand) - 6.7 GiB after a 720K-token DeepSeek-V4.1 read (chunks of 9,363 tokens,
/// windows of 7, 2026-11-01), while a follow-up's chunk is a few hundred rows.  h_hand stays when it holds no more than 4096
/// rows (a follow-up's chunk); the next long read pins the rest again (Impl::prompt sizes them as before), and the answers are
/// the same.  STRATA_X_KEEP_PROMPT_BUFS=1: kept, as before
void read_bufs_release(Engine::Impl& m) {
    static const bool keep = std::getenv("STRATA_X_KEEP_PROMPT_BUFS") && std::getenv("STRATA_X_KEEP_PROMPT_BUFS")[0] == '1';
    constexpr int kKeepRows = 4096;
    if (keep) return;
    bool any = false;
    for (auto& gp : m.stages) any |= !gp->h_lm.empty() || gp->eg_lm_h[0] || gp->eg_lm_h[1] || gp->h_hand_rows > kKeepRows;
    if (!any) return;
    const auto t0 = clk::now();
    const size_t row = (size_t) m.hp.hc * m.hp.n_embd * sizeof(float) + (m.hp.v41 ? 24 * sizeof(float) : 0);   // Impl::prompt's
    size_t freed = 0;
    for (auto& gp : m.stages) {
        Stage& g = *gp;
        cudaSetDevice(g.dev);
        // nothing may still use them: the prompt path's streams done (as its LendEnd waits), every engram fetch waited out
        if (g.s) cudaStreamSynchronize(g.s);
        if (g.cs) cudaStreamSynchronize(g.cs);
        for (cudaStream_t cst : g.lm_cs)
            if (cst) cudaStreamSynchronize(cst);
        g.eg_lm_drop();
        for (float* hb : g.h_lm) host_free(hb);
        freed += g.h_lm.size() * (size_t) g.h_lm_rows * row;
        g.h_lm.clear();
        g.h_lm_rows = 0;
        for (int b = 0; b < 2; ++b) {
            if (g.eg_lm_h[b]) {
                cudaFreeHost(g.eg_lm_h[b]);
                freed += (size_t) m.eg->L * g.eg_lm_cap * m.eg->row_bytes();
            }
            g.eg_lm_h[b] = g.eg_lm_d[b] = nullptr;
        }
        g.eg_lm_cap = 0;
        if (g.h_hand_rows > kKeepRows) {
            for (float*& hb : g.h_hand) {
                host_free(hb);
                hb = nullptr;
            }
            freed += 2 * (size_t) g.h_hand_rows * row;
            g.h_hand_rows = 0;
        }
    }
    m.cur_dev = -1;
    fprintf(stderr, "strata-x serve: the read's pinned buffers given back: %.2f GiB in %.0f ms\n", freed / 1073741824.0,
            ms_since(t0));
}
}  // namespace

bool Engine::run_request(const std::vector<int32_t>& prompt, int max_new, const SamplingParams& sp_in,
                         const std::vector<int32_t>& eos, const RequestHooks& hooks, bool ckpt, RequestStats& rs,
                         std::string& err, const std::vector<ImageSpan>* images) {
    Impl& m = *p_;
    rs = RequestStats{};
    const int64_t n = (int64_t) prompt.size();
    rs.prompt = n;
    if (n < 1 || max_new < 1) return m.fail(err, "an empty prompt or max_new < 1");
    // the request's own read-path settings (setup's calibration, the strata_tune keys), for this request only
    m.cpu_read_req = sp_in.read_cpu;
    m.share_req = sp_in.read_share;
    m.ring_req = sp_in.read_ring;
    struct ReadReqOff {
        Impl& m;
        ~ReadReqOff() { m.cpu_read_req = m.share_req = m.ring_req = -1; }
    } read_req_off{m};
    const int64_t req_short = sp_in.read_short;
    if (n + max_new + 8 > m.opt.n_ctx)
        return m.fail(err, "prompt (" + std::to_string(n) + " tokens) + max_new (" + std::to_string(max_new) +
                               ") exceeds the context (" + std::to_string(m.opt.n_ctx) + ")");
    for (int32_t t : prompt)
        if (t < 0 || t >= m.hp.n_vocab) return m.fail(err, "a token id is outside the vocabulary");
    // moex vision: the request's images (GENI) - sorted, disjoint, inside the read part of the prompt (the last token goes
    // through the first decode window as a token, so it is never an image row)
    std::vector<ImageKey> keys;
    if (images && !images->empty()) {
        // the text sides read images as mainline's mtmd feeds them: 1-D positions, the embedding batch through the same graph;
        // causal for glm5v and mimovl, DeepSeek-V4 (deepseek4v) reads each span as one non-causal batch (below).  Hy4-flash:
        // causal, with the release's M-RoPE (each picture's grid; Impl::hy4_mpos_set)
        int64_t at = 0;
        for (const ImageSpan& im : *images) {
            if (im.n < 1 || !im.rows || im.start < at || im.start + im.n > n - 1) return m.fail(err, "the images do not fit the prompt");
            at = im.start + im.n;
            keys.push_back(ImageKey{im.start, im.n, im.hash});
        }
        // DeepSeek-V4.1 routes an image's rows with bias_vl (model.py Gate.forward): a GGUF without it would route them as text
        if (m.hp.v41)
            for (int64_t il = 0; il < m.hp.n_layer; ++il)
                if (m.hp.is_moe(il) && !m.L[(size_t) il].probs_b_vl)
                    return m.fail(err, "this GGUF has no blk." + std::to_string(il) + ".exp_probs_b_vl: DeepSeek-V4.1 cannot route an "
                                       "image's rows without it (rewrite the GGUF with write_dense_gguf.py)");
    }
    m.first_top_pending = true;
    struct ImageScope {   // the rows are the caller's: never kept past this request
        Impl& m;
        ~ImageScope() { m.img.clear(); }
    } image_scope{m};
    m.img = images ? *images : std::vector<ImageSpan>{};
    // Hy4-flash reads its pictures with M-RoPE (vLLM's get_mrope_input_positions): the request's table to every stage
    if (m.hp.arch == Arch::hy4 && !m.hy4_mpos_set(err)) return false;
    SamplingParams sp = sp_in;
    if (sp.seed == 0) sp.seed = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    // DeepSeek-V4.1: a client's top_p / min_p act on the tempered distribution, as in the serving stacks its release names
    // (HF, vLLM, SGLang); with the release's own sampling (T 1.0, top_p 1.0, the whole vocabulary) there is nothing to order.
    // moex (ERNIE55_PLAN): ERNIE-5.5 too - its release serves on vLLM / SGLang (T 1.0, top_p 0.95 over every token)
    if (m.hp.v41 || m.hp.arch == Arch::ernie) sp.temp_first = true;
    const auto t0 = clk::now();
    m.finalize_swaps(true);

    // 1. what can be reused: the live sequence, or the longest checkpoint that is an exact prefix
    int64_t resume = 0;
    Checkpoint* ck = nullptr;
    auto find_reuse = [&] {
        resume = 0;
        ck = nullptr;
        if (m.live_valid && (int64_t) m.live.size() <= n - 1 && is_prefix(m.live, prompt, m.live.size()) &&
            images_match(m.live_imgs, keys, (int64_t) m.live.size()))
            resume = (int64_t) m.live.size();
        for (auto& c : m.checks)
            if ((int64_t) c->ids.size() <= n - 1 && (int64_t) c->ids.size() > resume && is_prefix(c->ids, prompt, c->ids.size()) &&
                images_match(c->imgs, keys, (int64_t) c->ids.size())) {
                ck = c.get();
                resume = (int64_t) c->ids.size();
            }
    };
    find_reuse();
    // moex (DSV41_PLAN §57, --conversation-cache-mib): a parked conversation this prompt continues further is put back (the live
    // one parked first if the prompt leaves it); else the live one is parked when the prompt leaves it.  Then the reuse is
    // found again - on the restored state, or without the checkpoints that went with the parked one
    if (m.park_on()) {
        if (!m.opt.conv_dir.empty()) m.disk_offer(prompt, keys, n, resume);   // (#53: a conversation an earlier engine saved)
        const bool restored = m.park_try(prompt, keys, n, resume);
        if (!restored && m.live_valid && resume < (int64_t) m.live.size()) m.park_leaving(prompt, keys);
        find_reuse();
        m.park_note_turn(prompt);
    }
    // moex (DSV41_BACKLOG #53, a measurement): how much of the tokens the engine holds this prompt really shares - when far more
    // than it can reuse (no checkpoint near where they part: a client that compacts its context, a dropped reasoning), the
    // log says where they part, which decides between a checkpoint policy and keeping more state
    if (!m.live.empty()) {
        const size_t lim = std::min(m.live.size(), prompt.size());
        size_t common = 0;
        while (common < lim && m.live[common] == prompt[common]) ++common;
        if ((int64_t) common > resume + 512)
            std::fprintf(stderr, "strata-x serve: this prompt shares %zu tokens with the engine's %zu but reuses %lld (no state kept "
                                 "nearer where they part)\n",
                         common, m.live.size(), (long long) resume);
    }
    m.live_valid = false;   // until this request has left the engine consistent
    if (ck) {
        m.restore(*ck);
        ck->used = ++m.ck_clock;
        m.live.assign(ck->ids.begin(), ck->ids.end());
        m.live_imgs = ck->imgs;
    } else if (resume == 0) {
        m.reset_state();
    }   // else: the live sequence continues
    // Strata's rule (generate.cpp's serve loop): this request rewrites every cell from `resume` on, so a checkpoint past it
    // (or not on this prompt's path) no longer has its cells - restoring it later would attend over another conversation's
    // latent cache.  The ones kept are prefixes of both the old tokens and the new.  Read from 0: none is kept (the root too).
    m.checks.erase(std::remove_if(m.checks.begin(), m.checks.end(),
                                  [&](const std::unique_ptr<Checkpoint>& c) {
                                      return (int64_t) c->ids.size() > resume || !is_prefix(c->ids, prompt, c->ids.size()) ||
                                             !images_match(c->imgs, keys, (int64_t) c->ids.size());
                                  }),
                   m.checks.end());
    if (resume == 0) m.checks.clear();
    rs.reused = resume;
    // moex (STRATA_HY4_KV_GROW): a request that keeps far fewer positions than the caches hold mapped gives the rest back, and
    // the expert slots they took return to the tier (the rows it keeps: the reused prefix, within its read)
    if ((m.hp.arch == Arch::hy4 || m.hp.arch == Arch::ernie) && m.kv_grow && !m.hy4_kv_release(n + kTmax + 16, err)) return false;
    if (hooks.resume) hooks.resume(resume);

    // DeepSeek-V4.1's route bias is the decode's: while the prompt is read (windows included, up to and with the first
    // decode window, which carries the last prompt token) its table holds zeros; the update after that window restores it
    struct PenPrompt {
        Impl& m;
        explicit PenPrompt(Impl& mm) : m(mm) {
            m.pen_prompt = true;
            m.route_pen_update();
        }
        ~PenPrompt() { m.pen_prompt = false; }
    } pen_prompt{m};

    // 2. read the prompt tokens [resume, n - 1); the last one goes through the first decode window
    const int64_t to_read = (n - 1) - resume;
    bool cancelled = false;
    rs.prompt_read = n - resume;   // as Strata counts it (generate.cpp's read_n): the fresh tokens, the decode-fed last one too
    // decode windows over [from, to): Strata's short reads ("a window costs ~16 ms a token"); with MTP the drafter's cells along
    // moex (the audit of 2026-10-19, #34b): the live tokens are always the positions the windows wrote - a cancel by the progress
    // hook used to let the loop run on and leave `live` where it was, up to a short read (96 rows) behind the state, and a resend
    // from there pools a ratio-2 block over a compressor ring row (16 rows) a later position overwrote
    auto read_windows = [&](int64_t from, int64_t to) -> bool {
        const auto tw = clk::now();
        int64_t i = from;
        while (i < to && !cancelled) {
            if (hooks.stop && hooks.stop()) {
                cancelled = true;
                break;
            }
            const int T = (int) std::min<int64_t>(kTmax, to - i);
            if (!window(prompt.data() + i, T, i, false, err)) return false;
            // the drafter's cells for these positions (their next tokens are in the prompt; DSpark: the rows' features)
            if (m.mtp_on && !m.drafter_read(T, prompt.data() + i + 1, i, err)) return false;
            i += T;
            if (hooks.progress && !hooks.progress(i, n, ms_since(t0))) cancelled = true;
        }
        m.live.assign(prompt.begin(), prompt.begin() + i);
        if (!cancelled && to - from >= 16) {   // what a token through windows costs at this depth (short_read_limit)
            const double v = ms_since(tw) / (double) (to - from);
            m.rw_ms_tok = m.rw_ms_tok > 0.0 ? m.rw_ms_tok + 0.3 * (v - m.rw_ms_tok) : v;
        }
        return true;
    };
    // the prompt path's cost per chunk (short_read_limit): its streamed experts dominate, whatever the chunk's length
    auto note_chunks = [&](double ms) {
        if (m.last_prompt_chunks <= 0) return;
        const double v = ms / m.last_prompt_chunks;
        m.chunk_ms = m.chunk_ms > 0.0 ? m.chunk_ms + 0.3 * (v - m.chunk_ms) : v;
    };
    // checkpoint positions: before the last turn token, the root, every `ck_every` fresh tokens; positions in (resume, n - 1]
    // (a chunk ends exactly there).  moex (DSV41_BACKLOG #43): planned the same way for V4.1's requests with pictures
    bool cuts_off_pics = false;   // moex (#113): no cut inside a picture's span (a merged read's chunks would make one)
    auto plan_cuts = [&](std::vector<int64_t>& cuts, std::vector<std::unique_ptr<Checkpoint>>& made) {
        auto want = [&](int64_t pos, bool root, bool deep) {
            if (pos <= resume || pos > n - 1) return;
            if (cuts_off_pics)
                for (const ImageSpan& im : m.img)
                    if (pos > im.start && pos < im.start + im.n) return;
            for (auto& c : m.checks)
                if ((int64_t) c->ids.size() == pos && is_prefix(c->ids, prompt, (size_t) pos)) {   // already held
                    c->deep |= deep;
                    return;
                }
            if (std::find(cuts.begin(), cuts.end(), pos) != cuts.end()) {
                for (auto& c : made)
                    if ((int64_t) c->ids.size() == pos) c->deep |= deep;
                return;
            }
            cuts.push_back(pos);
            auto c = std::make_unique<Checkpoint>();
            c->ids.assign(prompt.begin(), prompt.begin() + pos);
            c->imgs = images_below(keys, pos);
            c->root = root;
            c->deep = deep;
            c->blob.assign(m.stages.size(), nullptr);
            c->bytes.assign(m.stages.size(), 0);
            made.push_back(std::move(c));
        };
        if (ckpt && m.ck_keep > 0) {
            int64_t first_turn_after_root = -1, last_turn = -1, first_turns[2] = {-1, -1};
            for (int64_t i = 1; i < n; ++i)
                if (std::find(m.turn_tokens.begin(), m.turn_tokens.end(), prompt[(size_t) i]) != m.turn_tokens.end()) {
                    if (first_turn_after_root < 0 && i >= m.ck_root) first_turn_after_root = i;
                    if (first_turns[0] < 0) first_turns[0] = i;
                    else if (first_turns[1] < 0) first_turns[1] = i;
                    last_turn = i;
                }
            // moex (DSV41_BACKLOG #42), opt-in: STRATA_DSV41_ROOT_AT_SYSTEM=1 takes the root only where the prompt's first message
            // ends (its system prompt: the first or second turn boundary), the prefix later conversations share, or where the
            // read is cut anyway (the last turn token).  Further in, the root is one conversation's and its cut costs the SSD
            // tier a streaming pass: 3,570 fresh tokens read in 62.7 s as 2,873 + 697 at the root, 3,261 in one part in 35.4 s
            static const bool root_at_system = [] {
                const char* e = std::getenv("STRATA_DSV41_ROOT_AT_SYSTEM");
                return e && std::atoi(e) != 0;
            }();
            if (root_at_system && first_turn_after_root > 0 && first_turn_after_root != first_turns[0] &&
                first_turn_after_root != first_turns[1] && first_turn_after_root != last_turn) {
                if (first_turn_after_root > resume && first_turn_after_root <= n - 1)
                    std::fprintf(stderr, "strata-x serve: no root checkpoint at %lld, inside the conversation (its first message "
                                         "ends at %lld; STRATA_DSV41_ROOT_AT_SYSTEM)\n",
                                 (long long) first_turn_after_root, (long long) first_turns[1]);
                first_turn_after_root = -1;
            }
            bool have_root = false;
            for (auto& c : m.checks) have_root |= c->root && is_prefix(c->ids, prompt, c->ids.size());
            if (!have_root && first_turn_after_root > 0) want(first_turn_after_root, true, false);
            if (last_turn > 0) want(last_turn, false, false);
            // moex: the periodic checkpoints at the multiples of ck_every (deep ones), so a conversation read a turn at a time
            // gets them as it grows.  Counted from each read's start, it got none, and the ones a long read made went with the
            // turns' LRU five turns later: 2026-10-20's sessions read 128-155K tokens again from the 4,731-token root where
            // they shared 67-142K with the engine.  ck_deep 0: the old placement, in the LRU
            if (m.ck_every > 0) {
                const bool deep = m.ck_deep > 0;
                // moex (DSV41_BACKLOG #145, DSV41_PLAN §75): a short read that crosses a mark keeps the mark's deep checkpoint
                // at its last turn token - a cut it makes anyway, a turn boundary past the mark - and makes no cut at the mark
                // itself.  The cut split a follow-up of a few hundred tokens in two chunks, and a call of two chunks runs
                // layer-major with the tails deferred: every layer's experts streamed whole (~13,000 records for ~7,400
                // gated, §41), as the root's cut cost a streaming pass (ROOT_AT_SYSTEM above).  The read then computes what
                // any follow-up of its length computes.  STRATA_DSV41_MARK_MIN=N: reads of N new tokens or more cut at the
                // mark as before (4,096 by default; 0: every read)
                static const int64_t mark_min = [] {
                    const char* e = std::getenv("STRATA_DSV41_MARK_MIN");
                    return e ? std::max<int64_t>(0, std::atoll(e)) : 4096;
                }();
                const bool at_turn = deep && m.hp.v41 && n - 1 - resume < mark_min;
                for (int64_t pos = deep ? (resume / m.ck_every + 1) * m.ck_every : resume + m.ck_every; pos < n - 1; pos += m.ck_every) {
                    if (!at_turn) want(pos, false, deep);
                    else if (last_turn > pos && last_turn <= n - 1) want(last_turn, false, true);   // the planned cut, kept deep
                    // (no turn token past the mark in a read this short: no cut at all - the reply's own deep checkpoint of
                    //  the live state covers the mark, #62 below)
                }
            }
        }
        std::sort(cuts.begin(), cuts.end());
    };
    // the checkpoints that were completed up to `reached` (every stage has written its part): keep them
    auto keep_made = [&](std::vector<std::unique_ptr<Checkpoint>>& made, int64_t reached) {
        for (auto& c : made) {
            if ((int64_t) c->ids.size() > reached) continue;
            bool whole = true;
            for (size_t si = 0; si < m.stages.size(); ++si) whole &= c->blob.size() > si && c->blob[si] != nullptr;
            if (!whole) continue;
            c->used = ++m.ck_clock;
            m.checks.push_back(std::move(c));
        }
        // at most ck_keep of the root and the turns': the root stays, the rest by least recent use (moex: the deep ones are
        // not theirs to push out)
        auto is_deep = [](const std::unique_ptr<Checkpoint>& c) { return c->deep && !c->root; };
        auto held = [&](bool deep) {
            int k = 0;
            for (auto& c : m.checks) k += is_deep(c) == deep;
            return k;
        };
        while (held(false) > m.ck_keep) {
            size_t victim = m.checks.size();
            for (size_t i = 0; i < m.checks.size(); ++i)
                if (!m.checks[i]->root && !is_deep(m.checks[i]) &&
                    (victim == m.checks.size() || m.checks[i]->used < m.checks[victim]->used))
                    victim = i;
            if (victim == m.checks.size()) break;
            m.checks.erase(m.checks.begin() + (std::ptrdiff_t) victim);
        }
        // moex: at most ck_deep deep ones - the one whose neighbours lie closest together goes (never the deepest), so the
        // rest stay spread over the conversation
        while (held(true) > m.ck_deep) {
            std::vector<size_t> d;
            for (size_t i = 0; i < m.checks.size(); ++i)
                if (is_deep(m.checks[i])) d.push_back(i);
            std::sort(d.begin(), d.end(), [&](size_t a, size_t b) { return m.checks[a]->ids.size() < m.checks[b]->ids.size(); });
            size_t victim = m.checks.size();
            int64_t best = INT64_MAX;
            for (size_t k = 0; k + 1 < d.size(); ++k) {
                const int64_t lo = k ? (int64_t) m.checks[d[k - 1]]->ids.size() : 0;
                const int64_t gap = (int64_t) m.checks[d[k + 1]]->ids.size() - lo;
                if (gap < best) {
                    best = gap;
                    victim = d[k];
                }
            }
            if (victim == m.checks.size()) break;
            m.checks.erase(m.checks.begin() + (std::ptrdiff_t) victim);
        }
    };
    // moex (DSV41_BACKLOG #62): a mark of ck_every that the live tokens passed in a reply or a short read (the decode and
    // windows make no cut; plan_cuts plans marks after `resume` only) is kept now as a deep checkpoint of the live state - exact
    // here, the request continues it: one copy of the stages' state (~28 MB on V4.1), no cut and no stream pass.  2026-10-20's
    // traffic crossed ~58% of its marks so, and a rewrite past them read from the root
    if (ckpt && m.ck_keep > 0 && m.ck_deep > 0 && m.ck_every > 0 && !ck && resume > 0 && resume == (int64_t) m.live.size()) {
        const int64_t mark = resume / m.ck_every * m.ck_every;
        bool held = mark == 0;
        for (auto& c : m.checks) held = held || (c->deep && (int64_t) c->ids.size() >= mark && (int64_t) c->ids.size() <= resume);
        if (!held) {
            std::vector<std::unique_ptr<Checkpoint>> snap;
            auto c = std::make_unique<Checkpoint>();
            c->ids.assign(m.live.begin(), m.live.end());
            c->imgs = m.live_imgs;
            c->deep = true;
            c->blob.assign(m.stages.size(), nullptr);
            c->bytes.assign(m.stages.size(), 0);
            for (size_t si = 0; si < m.stages.size(); ++si) m.snapshot(*c, (int) si);
            snap.push_back(std::move(c));
            keep_made(snap, resume);   // kept when every stage's copy was made, under the deep cap
        }
    }
    if (to_read > 0 && m.hp.arch == Arch::dsv4 && !keys.empty()) {
        // DeepSeek-V4 reads each image span as ONE batch with causal attention off - mainline's mtmd decodes an image chunk as
        // its own ubatch with llama_set_causal_attn(false) (deepseek4's non_causal_type SWA_FULL: the span's rows see each
        // other past the 128-position window, older cells keep it; the compressed rows and the indexer see completed blocks
        // only) - its rows routed with exp_probs_b_vl and no tid2eid (deepseek4.cpp is_media).  The text around the spans reads
        // as usual, without checkpoints: a span must never be cut by a chunk or a window.  DeepSeek-V4.1 (moex vision) reads its
        // span the same way but CAUSALLY - the release's model.py runs an image span through the same causal attention as text,
        // routed with bias_vl (Gate.forward's image_mask) and given no engram (eg_note's DEAD positions, prompt_layer).
        // moex (DSV41_BACKLOG #43): V4.1's text parts read with the prompt path's control - progress lines, STOP, and the text
        // path's checkpoints (plan_cuts), each remembering the pictures below it; a cut inside a span is never made (a span
        // reads as one batch without it).  Before, a request with a picture read with none of them: a follow-up whose live
        // tokens were no prefix read the whole conversation again, and a long silent part could outlast the server's patience
        const int64_t short_lim = req_short >= 0 ? req_short : m.short_read_limit();
        const bool ictl_on = m.hp.v41;
        static const bool pic_merge_env = std::getenv("STRATA_DSV41_PIC_MERGE") && std::getenv("STRATA_DSV41_PIC_MERGE")[0] == '1';
        const bool pic_merge = pic_merge_env && m.hp.v41 && !m.mtp_on && !m.dsp && !m.dfl && m.opt.prefill_chunk != -1 &&
                               m.gub_max > 0 && m.mmq_all && mmq::built();
        cuts_off_pics = pic_merge;
        PromptCtl ictl;
        std::vector<int64_t> icuts;
        std::vector<std::unique_ptr<Checkpoint>> imade;
        if (ictl_on) {
            plan_cuts(icuts, imade);
            ictl.cuts = icuts;
            ictl.on_cut = [&](int si, int64_t pos) {
                for (auto& c : imade)
                    if ((int64_t) c->ids.size() == pos) m.snapshot(*c, si);
            };
            ictl.progress = [&](int64_t reached) { return !(hooks.progress && !hooks.progress(reached, n, ms_since(t0))); };
            ictl.stop = hooks.stop;
            ictl.work = hooks.work;
        }
        int64_t ireached = resume;   // how far the reading has left the state exact (the checkpoints up to it are kept)
        auto read_text = [&](int64_t from, int64_t to) -> bool {
            if (to <= from) return true;
            if (to - from > short_lim && m.opt.prefill_chunk != -1 && m.gub_max > 0 && m.mmq_all && mmq::built()) {
                const auto tp = clk::now();
                if (ictl_on) m.pctl = &ictl;
                const bool ok = m.prompt(prompt.data() + from, (int) (to - from), from, m.mtp_on ? (int) (to - from) + 1 : 0, err);
                m.pctl = nullptr;
                if (!ok) return false;
                if (ictl_on && ictl.stopped.load()) {   // a stop between chunks: #34a's rule - the live tokens only at a read start
                    cancelled = true;
                    const int64_t reached = std::max<int64_t>(from, std::min<int64_t>(to, ictl.done));
                    const bool exact = !ictl.torn && (m.replay_layer < 0 || reached == from || reached == to ||
                                                      std::binary_search(icuts.begin(), icuts.end(), reached));
                    m.live.assign(prompt.begin(), prompt.begin() + (exact ? reached : 0));
                    ireached = reached;
                    return true;
                }
                note_chunks(ms_since(tp));
                m.live.assign(prompt.begin(), prompt.begin() + to);
                ireached = to;
                return true;
            }
            if (!read_windows(from, to)) return false;
            ireached = (int64_t) m.live.size();
            return true;
        };
        // moex (DSV41_BACKLOG #113), opt-in STRATA_DSV41_PIC_MERGE=1: the text and the pictures as one read - each row routes
        // and takes the engram by its position (a picture's: exp_probs_b_vl, no hash, no engram), so the experts they share
        // stream once and the short text around a picture reads in its chunk, not through windows.  Rounding only: the
        // chunks' shapes change.  Off with a drafter (its blocks route a picture's rows as text)
        struct MixOff {
            Engine::Impl& m;
            ~MixOff() {
                for (auto& gp : m.stages) gp->img_mix = false;
            }
        };
        auto read_merged = [&](int64_t from, int64_t to) -> bool {
            const auto tp = clk::now();
            MixOff off{m};
            for (auto& gp : m.stages) gp->img_mix = true;
            m.pctl = &ictl;
            const bool ok = m.prompt(prompt.data() + from, (int) (to - from), from, 0, err);
            m.pctl = nullptr;
            if (!ok) return false;
            if (ictl.stopped.load()) {   // a stop between chunks: as read_text's
                cancelled = true;
                const int64_t reached = std::max<int64_t>(from, std::min<int64_t>(to, ictl.done));
                const bool exact = !ictl.torn && (m.replay_layer < 0 || reached == from || reached == to ||
                                                  std::binary_search(icuts.begin(), icuts.end(), reached));
                m.live.assign(prompt.begin(), prompt.begin() + (exact ? reached : 0));
                ireached = reached;
                return true;
            }
            note_chunks(ms_since(tp));
            m.live.assign(prompt.begin(), prompt.begin() + to);
            ireached = to;
            return true;
        };
        int64_t pos = resume;
        if (pic_merge) {   // moex (DSV41_BACKLOG #113): the new text and its pictures in one prompt call, up to the short tail
            int64_t img_end = pos;
            for (const ImageSpan& im : m.img) {
                if (im.start < resume && im.start + im.n > resume) return m.fail(err, "a reused prefix ends inside an image");
                img_end = std::max(img_end, im.start + im.n);
            }
            int64_t to = n - 1;
            if (!icuts.empty() && icuts.back() > pos && icuts.back() >= img_end && (n - 1) - icuts.back() <= short_lim) to = icuts.back();
            if (img_end > pos && to > pos) {
                if (!read_merged(pos, to)) return false;
                if (!cancelled) pos = to;
            }
        }
        for (const ImageSpan& im : m.img) {
            if (pic_merge) break;
            if (im.start + im.n <= resume) continue;   // inside the reused prefix (its images matched)
            if (im.start < resume) return m.fail(err, "a reused prefix ends inside an image");   // spans are read whole
            if (!read_text(pos, im.start)) return false;
            if (cancelled) break;
            if (!(m.opt.prefill_chunk != -1 && m.gub_max > 0 && m.mmq_all && mmq::built()))
                return m.fail(err, "DeepSeek-V4 reads an image as one batch: the prompt path is off");
            for (auto& gp : m.stages) {
                gp->attn_noncausal = !m.hp.v41;
                gp->image_batch = true;
            }
            const bool ok = m.prompt(prompt.data() + im.start, (int) im.n, im.start, m.mtp_on ? (int) im.n + 1 : 0, err);
            for (auto& gp : m.stages) gp->attn_noncausal = gp->image_batch = false;
            if (!ok) return false;
            m.live.assign(prompt.begin(), prompt.begin() + im.start + im.n);
            pos = im.start + im.n;
            ireached = pos;
            // moex (DSV41_BACKLOG #63): a cut planned at the span's end - a user turn that ends with a picture has its last-turn
            // cut there - ends no chunk of read_text, so it is snapshot here: a prompt call's end is a read start, exact
            if (ictl_on)
                for (auto& c : imade)
                    if (c && (int64_t) c->ids.size() == pos)
                        for (size_t si = 0; si < m.stages.size(); ++si)
                            if (c->blob.size() <= si || !c->blob[si]) m.snapshot(*c, (int) si);
            if (hooks.progress && !hooks.progress(pos, n, ms_since(t0))) cancelled = true;
            if (cancelled) break;
        }
        if (!cancelled) {   // the last part; as the text path, a short tail behind the last cut goes through windows
            int64_t tail_from = n - 1;
            if (!icuts.empty() && icuts.back() > pos && (n - 1) - icuts.back() <= short_lim) tail_from = icuts.back();
            if (!read_text(pos, tail_from)) return false;
            // a last cut behind a short text (read through windows: no chunk ends there) is snapshot here - windows leave
            // every layer exact at their end.  Without it a picture question's follow-up found no checkpoint below the
            // picture-free tail and read everything again
            if (ictl_on && !cancelled && tail_from < n - 1 && ireached == tail_from)
                for (auto& c : imade)
                    if ((int64_t) c->ids.size() == tail_from)
                        for (size_t si = 0; si < m.stages.size(); ++si)
                            if (c->blob.size() <= si || !c->blob[si]) m.snapshot(*c, (int) si);
            if (!cancelled && tail_from < n - 1 && !read_windows(tail_from, n - 1)) return false;
        }
        if (ictl_on) keep_made(imade, ireached);
        if (cancelled) {
            rs.finish = "cancel";
            rs.prompt_ms = ms_since(t0);
            rs.prompt_read = std::max<int64_t>(0, (int64_t) m.live.size() - resume);
            m.live_imgs = images_below(keys, (int64_t) m.live.size());
            m.live_valid = true;
            return true;
        }
    } else if (to_read > 0) {
        const int64_t short_lim = req_short >= 0 ? req_short : m.short_read_limit();
        const bool batched = to_read > short_lim && m.opt.prefill_chunk != -1 && m.gub_max > 0 && m.mmq_all && mmq::built();
        if (batched) {
            PromptCtl ctl;
            // checkpoint positions: before the last turn token, the root, every `ck_every` fresh tokens; positions in
            // (resume, n - 1] (a chunk ends exactly there)
            std::vector<int64_t> cuts;
            std::vector<std::unique_ptr<Checkpoint>> made;
            plan_cuts(cuts, made);
            ctl.cuts = cuts;
            ctl.on_cut = [&](int si, int64_t pos) {
                for (auto& c : made)
                    if ((int64_t) c->ids.size() == pos) m.snapshot(*c, si);
            };
            ctl.progress = [&](int64_t reached) {
                if (hooks.progress && !hooks.progress(reached, n, ms_since(t0))) return false;
                return true;
            };
            ctl.stop = hooks.stop;
            ctl.work = hooks.work;
            m.pctl = &ctl;
            // with MTP the drafter's cells are built along: every position read has its next token in the prompt
            // Strata decides short reads per PART: a last segment (after the last cut: usually the generation prompt behind the
            // last turn token) of at most short_read tokens goes through windows - through the prompt path it would be a chunk
            // of its own, streaming every non-resident expert (~12 s on GLM) for a handful of tokens
            int64_t tail_from = n - 1;
            if (!cuts.empty() && cuts.back() > resume && (n - 1) - cuts.back() <= short_lim) tail_from = cuts.back();
            const int path_n = (int) (tail_from - resume);
            const auto tp = clk::now();
            const bool ok = m.prompt(prompt.data() + resume, path_n, resume, m.mtp_on ? path_n + 1 : 0, err);
            if (ok && !ctl.stopped.load()) note_chunks(ms_since(tp));
            m.pctl = nullptr;
            if (!ok) return false;
            cancelled = ctl.stopped.load();
            const int64_t reached = std::max<int64_t>(resume, std::min<int64_t>(tail_from, ctl.done));
            // moex (#34a): with bounded replay (DeepSeek-V4.1) the layers from replay_layer on are exact only at a read start - a
            // checkpoint cut or this call's end, each with its replayed tail; a stop at another chunk end leaves them behind the
            // rest, and the state cannot be rolled back to a cut (the compressor rings moved on).  The live tokens are dropped
            // then: the next request starts from the last checkpoint kept below (or from 0) instead of reading on from there
            const bool exact = !ctl.torn && (m.replay_layer < 0 || reached == resume || reached == tail_from ||
                                             std::binary_search(cuts.begin(), cuts.end(), reached));
            m.live.assign(prompt.begin(), prompt.begin() + (exact ? reached : 0));
            keep_made(made, reached);
            if (!cancelled && tail_from < n - 1 && !read_windows(tail_from, n - 1)) return false;   // the short tail
            if (cancelled) {
                rs.finish = "cancel";
                rs.prompt_ms = ms_since(t0);
                rs.prompt_read = std::max<int64_t>(0, std::max<int64_t>((int64_t) m.live.size(), reached) - resume);
                m.live_imgs = images_below(keys, (int64_t) m.live.size());
                m.live_valid = true;
                return true;
            }
        } else {   // a short segment: decode windows ("a window costs ~16 ms a token"), no checkpoints
            if (!read_windows(resume, n - 1)) return false;
            if (cancelled) {
                rs.finish = "cancel";
                rs.prompt_ms = ms_since(t0);
                rs.prompt_read = std::max<int64_t>(0, (int64_t) m.live.size() - resume);   // as far as the read got
                m.live_imgs = images_below(keys, (int64_t) m.live.size());
                m.live_valid = true;
                return true;
            }
        }
    }
    if (hooks.reused) hooks.reused(resume);
    read_bufs_release(m);   // moex: the read is over - its pinned buffers back before the decode

    // 3. the decode (spec.cpp): the first window is the last prompt token at position n - 1; with MTP the drafter's
    //    chain and the verify windows follow it (Strata's serial loop)
    std::vector<int32_t> hist(prompt.end() - std::min<int64_t>(n, m.samp_hist_cap), prompt.end());   // the penalty window's tail
    const int64_t hits0 = [&] { return stats().hits; }(), miss0 = stats().misses;
    const auto td = clk::now();
    rs.prompt_ms = ms_since(t0);
    Impl::SpecRun run;
    run.sp = &sp;
    run.eos = &eos;
    run.max_new = max_new;
    run.pos = n - 1;
    run.x = prompt.back();
    run.ctx = prompt.data();
    run.ctx_n = n;
    run.hist = &hist;
    run.token = [&](int32_t y) { return hooks.token ? hooks.token(y) : true; };
    run.stop = hooks.stop;
    m.pen_prompt = false;   // the table keeps its zeros through the first window (the last prompt token); then the decode's
    if (!m.spec_decode(run, err)) return false;
    if (run.nonfinite) {   // #879: the poison may sit in any state this request wrote or reused - none of it is kept
        rs.finish = "nonfinite";
        rs.generated = run.produced;
        rs.decode_ms = ms_since(td);
        rs.drafts_accepted = run.accepted;
        rs.drafts_offered = run.offered;
        rs.windows = run.windows;
        forget();
        return true;
    }
    if (cancelled || run.cancelled) rs.finish = "cancel";
    else if (run.eos_hit) rs.finish = "stop";
    rs.generated = run.produced;
    rs.decode_ms = ms_since(td);
    rs.hits = stats().hits - hits0;
    rs.lookups = rs.hits + (stats().misses - miss0);
    rs.drafts_accepted = run.accepted;
    rs.drafts_offered = run.offered;
    rs.windows = run.windows;
    rs.lookup_windows = run.lookup_windows;
    rs.lookup_accepted = run.lookup_accepted;
    for (int j = 0; j < kTmax && j < 8; ++j) {
        rs.offered_at[j] = run.offered_at[j];
        rs.accepted_at[j] = run.accepted_at[j];
    }
    rs.draft_ms = run.draft_ms;
    m.live_imgs = images_below(keys, (int64_t) m.live.size());   // the images ride along with the prompt part of the live tokens
    m.live_valid = true;   // the live tokens are exactly the fed ones: the prompt, and every generated token but the last
    return true;
}

}  // namespace strata::x
