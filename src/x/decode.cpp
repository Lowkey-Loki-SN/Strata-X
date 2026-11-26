// src/x/decode.cpp - THE DECODE GRAPHS, Strata's Verifier structure (STRATA_QWEN_DESIGN.md section 3) for GLM:
//
//   * one CUDA graph per stage (a GPU's layers) per window variant (T tokens, keep), captured on a variant's second use
//     (the first runs eagerly and loads every kernel; CUDA_MODULE_LOADING=EAGER, as Strata's engine sets it);
//   * every per-step value lives in device memory, never in a kernel argument: the window's position (d_pos), the VRAM
//     tier's residency table (d_res), the doorbell's counters - the graph is replayed at every position;
//   * each MoE layer, inside the graph: the router, then the DOORBELL (a kernel copies the ids and the layer's
//     activations into mapped host memory and bumps a sequence word), the shared expert and the VRAM tier's HITS - chosen
//     on the device from d_res and computed by grouped launches (one per projection over every routed pair) - and then a
//     spin kernel that waits for the host's flag before the combine reads the CPU's rows from mapped memory;
//   * the host launches ONE graph per stage and serves the layers: spin on the sequence word (no driver call), classify
//     the pairs by the same residency table, run the misses on the CPU pool, write the rows into the mapped block, raise
//     the flag.  Within a layer the GPU computes the shared expert and the VRAM hits while the CPU computes the misses;
//     the chain is serial across layers (Strata: "no cross-layer pipelining" - the next layer needs this one's output).
// The eager path (stage.cpp's layer_window / moe_window) stays: the probe mode, the NextN block, T > 8.
#include "engine_impl.hpp"

#include "ggml.h"

#include <emmintrin.h>

#include <algorithm>
#include <cstring>

namespace strata::x {

bool Stage::alloc_decode(std::string& err) {
    cudaSetDevice(dev);
    const int64_t n = hp.n_embd, k = hp.n_used, nf = hp.n_ff_exp, E = hp.n_expert;
    auto up = [](size_t v) { return (v + 255) & ~(size_t) 255; };
    const size_t off_ids = 256, off_x = off_ids + up((size_t) kTmax * k * 4), off_rows = off_x + up((size_t) kTmax * n * 4),
                 off_mask = off_rows + up((size_t) kTmax * k * n * 4), total = off_mask + 256;
    if (cudaHostAlloc(&db_block, total, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
        return m.fail(err, "the doorbell's mapped host block");
    std::memset(db_block, 0, total);
    void* dp = nullptr;
    if (cudaHostGetDevicePointer(&dp, db_block, 0) != cudaSuccess) return m.fail(err, "the doorbell's device view");
    auto hv = [&](size_t off) { return static_cast<uint8_t*>(db_block) + off; };
    auto dv = [&](size_t off) { return static_cast<uint8_t*>(dp) + off; };
    db_seq = reinterpret_cast<volatile uint32_t*>(hv(0));
    db_flag = reinterpret_cast<volatile uint32_t*>(hv(64));
    db_abort = reinterpret_cast<volatile uint32_t*>(hv(128));
    dd_seq = reinterpret_cast<volatile uint32_t*>(dv(0));
    dd_flag = reinterpret_cast<volatile uint32_t*>(dv(64));
    dd_abort = reinterpret_cast<volatile uint32_t*>(dv(128));
    db_egf = reinterpret_cast<volatile uint32_t*>(hv(192));   // (moex, §42: the engram rows' word, eg_late)
    dd_egf = reinterpret_cast<volatile uint32_t*>(dv(192));
    db_ids = reinterpret_cast<int32_t*>(hv(off_ids));
    dd_ids = reinterpret_cast<int32_t*>(dv(off_ids));
    db_x = reinterpret_cast<float*>(hv(off_x));
    dd_x = reinterpret_cast<float*>(dv(off_x));
    db_rows = reinterpret_cast<float*>(hv(off_rows));
    dd_rows = reinterpret_cast<float*>(dv(off_rows));
    db_mask = hv(off_mask);
    dd_mask = dv(off_mask);
    if (!(d_pub_cnt = dalloc<uint32_t>(1, err)) || !(d_wait_cnt = dalloc<uint32_t>(1, err)) || !(d_eg_cnt = dalloc<uint32_t>(1, err)) ||
        !(d_pos = dalloc<int64_t>(1, err)) ||
        !(d_res = dalloc<int32_t>((size_t) m.n_load * E, err)) || !(d_hslot = dalloc<int32_t>((size_t) kTmax * k, err)) ||
        !(hit_g = dalloc<float>((size_t) kTmax * k * nf, err)) || !(hit_u = dalloc<float>((size_t) kTmax * k * nf, err)) ||
        !(hit_parts = dalloc<float>((size_t) kTmax * k * n, err)) ||
        !(hit_xq = dalloc<uint8_t>(kn::native_q8_1_bytes((int) (kTmax * k * nf)), err)))
        return false;
    cudaMemset(d_res, 0xFF, (size_t) m.n_load * E * sizeof(int32_t));   // -1: nothing resident
    if (m.rio && m.rio->ahead_on) {   // DeepSeek-V4.1's read-ahead: the next layer's router on this layer's FFN rows
        void* hp_ = nullptr;
        if (!(apred_x = dalloc<float>((size_t) kTmax * n, err)) || !(apred_logits = dalloc<float>((size_t) kTmax * E, err)) ||
            !(apred_wts = dalloc<float>((size_t) kTmax * k, err)) || !(apred_ids = dalloc<int32_t>((size_t) kTmax * k, err)) ||
            cudaHostAlloc(&hp_, (size_t) kTmax * k * sizeof(int32_t), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &dd_pred, hp_, 0) != cudaSuccess)
            return m.fail(err, "the read-ahead's buffers");
        db_pred = static_cast<int32_t*>(hp_);
        void* hs = nullptr;   // moex (§32): the guess's own published word and device count
        if (!(d_pred_cnt = dalloc<uint32_t>(1, err)) ||
            cudaHostAlloc(&hs, sizeof(uint32_t), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &dd_pred_seq, hs, 0) != cudaSuccess)
            return m.fail(err, "the read-ahead's published word");
        db_pred_seq = static_cast<volatile uint32_t*>(hs);
        *db_pred_seq = 0u;
        cudaMemset(d_pred_cnt, 0, sizeof(uint32_t));
        pred_expect = 0u;
    }
    if (cudaMallocHost(&h_pos, sizeof(int64_t)) != cudaSuccess) return m.fail(err, "pinned position word");
    // the step's inputs in mapped memory (Portable: the next GPU reads this stage's out_rows)
    auto mapped = [&](size_t bytes, void** hp_, void** dp_) {
        if (cudaHostAlloc(hp_, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return false;
        std::memset(*hp_, 0, bytes);
        return cudaHostGetDevicePointer(dp_, *hp_, 0) == cudaSuccess;
    };
    if (!mapped(sizeof(int64_t), (void**) &in_pos_h, (void**) &in_pos_d) ||
        !mapped((size_t) kTmax * n * sizeof(float), (void**) &in_rows_h, (void**) &in_rows_d) ||
        !mapped((size_t) m.n_load * E * sizeof(int32_t), (void**) &in_res_h, (void**) &in_res_d) ||
        // (DeepSeek-V4.1: the residuals, then the rows' carried pre-mix at kTmax * hc * n)
        !mapped((size_t) kTmax * (hp.hc * n + (hp.v41 ? 24 : 0)) * sizeof(float), (void**) &out_rows_h, (void**) &out_rows_d))
        return m.fail(err, "the step's mapped input blocks");
    if (hp.arch == Arch::dsv4 && di == 0 && !mapped((size_t) kTmax * sizeof(int32_t), (void**) &in_tok_h, (void**) &in_tok_d))
        return m.fail(err, "the window's mapped token ids");
    for (int64_t i = 0; i < m.n_load * E; ++i) in_res_h[i] = -1;
    res_lo = lb * E;
    res_n = (le - lb) * E;
    if (last && m.mtp_on &&
        (!mapped(sizeof(int64_t), (void**) &mtp_in_pos_h, (void**) &mtp_in_pos_d) ||
         !mapped((size_t) kTmax * n * sizeof(float), (void**) &mtp_in_rows_h, (void**) &mtp_in_rows_d) ||
         !mapped(sizeof(int32_t), (void**) &mtp_out_tok_h, (void**) &mtp_out_tok_d) ||
         !mapped(sizeof(float), (void**) &mtp_out_prob_h, (void**) &mtp_out_prob_d) ||
         !mapped(sizeof(int32_t), (void**) &cd_cell_h, (void**) &cd_cell_d)))
        return m.fail(err, "the drafter's mapped blocks");
    hit_ok.assign((size_t) m.n_load, 0);
    if (last) {   // the sampler's buffers (session.cpp)
        if (!(m.samp_hist = dalloc<int32_t>((size_t) kTmax * m.samp_hist_cap, err)) || !(m.samp_out = dalloc<int32_t>(kTmax, err))) return false;
        if (cudaMallocHost(&m.h_hist, (size_t) kTmax * m.samp_hist_cap * sizeof(int32_t)) != cudaSuccess)
            return m.fail(err, "the sampler's pinned history");
    }
    return true;
}

void Stage::release_gpu_waits() {
    if (!db_abort) return;
    *db_abort = 1u;   // the wait kernels exit; the stream drains
    cudaSetDevice(dev);
    cudaStreamSynchronize(s);
    *db_abort = 0u;
    // moex (DSV41_BACKLOG #56): the rest of the window ran on - each publish and each released wait still moved its device
    // counter - so the host takes the device's counts: otherwise the next window reads the last publish's ids before its own
    // publish, and its wait kernel spins on a flag the host never writes
    if (d_pub_cnt && d_wait_cnt) {
        uint32_t c[2] = {0u, 0u};
        if (cudaMemcpy(&c[0], d_pub_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(&c[1], d_wait_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess) {
            host_seq = c[0];
            host_flag = c[1];
            *db_flag = host_flag;
        }
    }
    if (d_eg_cnt) {   // (moex, §42: the engram rows' wait, released with the others)
        uint32_t c = 0u;
        if (cudaMemcpy(&c, d_eg_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess) {
            host_egf = c;
            *db_egf = host_egf;
        }
    }

    if (d_pred_cnt) {   // (moex, §32: the guesses' count too)
        uint32_t c = 0u;
        if (cudaMemcpy(&c, d_pred_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess) pred_expect = c;
    }
}

/// One layer's GPU work as captured: the same ops as layer_window / moe_window, with the position from d_pos, the
/// routing handed to the host through the doorbell, the VRAM hits as grouped launches chosen by d_res, and the CPU's rows
/// taken from mapped memory.  Probes are off (the graph path never runs with a recorder).
void Stage::layer_graph_body(int il, int T, bool keep) {
    const Layer& l = L[(size_t) il];
    const int n = (int) hp.n_embd, E = (int) hp.n_expert, k = (int) hp.n_used, nf = (int) hp.n_ff_exp;
    dspark_capture(il, 0, T);   // DSpark: the streams' mean at a target layer's input (in the graph)
    dflash_capture(il, 0, T, false);   // DFlash: the residual there (an earlier stage's: into its mapped block)
    if (m.eg_index(il) >= 0) {   // DeepSeek-V4.1's engram (the rows: mapped)
        // (moex, §42: the host's copies first; with eg_ring, §76, the wait is in the layer before, below)
        if (m.eg_late && il == m.eg_late_layer && !m.eg_ring) doorbell_wait(dd_egf, dd_abort, d_eg_cnt, s);
        // (moex, #140: its rows' half may have run in the layer before, below - eg_pre_kv)
        engram(il, 0, T, eg_win_d, kTmax, eg_pre_kv.empty() ? nullptr : eg_pre_kv[(size_t) m.eg_index(il)]);
    }
    // Hy4-flash's Engram: its rows were uploaded to hy_eg_rows on this stream before the launch (forward_graph)
    if (const int j = m.hy4_eg_index(il); j >= 0 && m.heg) hy4_engram(il, j, 0, T, hy_eg_rows, hy_eg_valid, kTmax);
    mix_in(il, T, R, cur, coef, true);
    attention(il, 0, T, cur, out, keep);
    mix_out(il, T, out, R, coef, true);
    mix_in(il, T, R, cur, coef, false);
    if (!hp.is_moe(il)) {
        swiglu_ffn(il, T, l.ffn_gate, l.ffn_up, l.ffn_down, hp.clamp_sh[(size_t) il], cur, out);
        if (l.ffn_post_norm) rms_norm(out, l.ffn_post_norm.f(), out, T, n, hp.rms_eps, s);   // Hy4-flash's sandwich norm
    } else {
        // Hy4-flash's latent MoE (moe_window's): the experts read xl = routed_down(cur), nx wide
        const bool lat = (bool) l.r_down;   // (Hy4-flash's and ERNIE-5.5's latent MoE)
        const int nx = lat ? (int) hp.ex() : n;
        // moex (the hill-climb): ERNIE-5.5's routers and routed_down in one launch (gemv_pair_w: each row by its own kernel's
        // code; hy4_moe_in's Q8_1 copy of cur is not read before the latent's overwrites it)
        const bool rd_pair = hp.arch == Arch::ernie && lat && l.gate_inp.type == GGML_TYPE_BF16 && l.r_down.type == kDtFP8 &&
                             gemv_pair_w(l.gate_inp.d, nullptr, false, router, (int) l.gate_inp.n_out(), l.r_down.d,
                                         static_cast<const uint8_t*>(l.r_down.s), true, hxl, nx, cur, n, T, s);
        if (!rd_pair) {
            gemv_dt(l.gate_inp, cur, router, n, (int) l.gate_inp.n_out(), T);   // (ERNIE-5.5: 2 E logits)
            if (lat) hy4_moe_in(il, T);
        }
        if (!route_publish(il, T, in_tok_d, m.route_pen_win(il))) {   // (moex, #112: one launch, STRATA_DSV41_ROUTE_PUBLISH)
            route(il, T, in_tok_d, m.route_pen_win(il));   // DeepSeek-V4's hash layers read the window's tokens from mapped memory (the first stage's)
            doorbell_publish(ids, T * k, lat ? hxl : cur, T * nx, dd_ids, dd_x, dd_seq, d_pub_cnt, s);
        }
        if (m.ahead_from(il)) {   // DeepSeek-V4.1: the experts d layers on guessed after the doorbell (moex, §32), published so
            ahead_compute(il, T);  // (the ids, then their own count)
            doorbell_publish(apred_ids, T * (int) hp.k_of(il + m.ahead_d), nullptr, 0, dd_pred, nullptr, dd_pred_seq, d_pred_cnt, s);
        }
        if (hp.shexp) swiglu_ffn(il, T, l.sh_gate, l.sh_up, l.sh_down, hp.clamp_sh[(size_t) il], cur, shared);   // leaves cur's Q8_1 columns in xq
        else quant(cur, xq, n, T);
        if (lat) quant(hxl, xq, nx, T);   // the hits' columns: the latent's
        const kc::NativeFmt& f = l.fmt;
        const bool hits = n_slots > 0 && kn::native_mmvq_grouped_supported(f.gu_type, nx) &&
                          kn::native_mmvq_grouped_supported(f.d_type, nf);
        hit_ok[(size_t) il] = hits ? 1 : 0;
        if (hits) {
            hit_slots(ids, d_res + (size_t) il * E, d_hslot, T * k, s);
            // (moex, §24: with sized slots the residency table holds the slots' offsets in kSlotUnit units)
            const size_t sstride = m.slot_sized ? Engine::Impl::kSlotUnit : m.slot_bytes;
            kn::native_mmvq_grouped(f.gu_type, slots, sstride, 0, f.up_off, d_hslot, xq, k, hit_g, (size_t) nf, hit_u,
                                    (size_t) nf, nx, nf, T * k, s);
            swiglu_clamp(hit_g, hit_u, hit_g, T * k * nf, f.swiglu_limit, s);
            quant(hit_g, hit_xq, T * k * nf, 1);
            kn::native_mmvq_grouped(f.d_type, slots, sstride, f.down_off, 0, d_hslot, hit_xq, 1, hit_parts, (size_t) nx,
                                    nullptr, 0, nf, nx, T * k, s);
        }
        // moex (DSV41_BACKLOG #140, DSV41_PLAN §75): the next layer's engram rows' half (the dequant and wkv: they read the
        // window's rows, nothing of the streams) here, where the GPU would only spin until the CPU's rows are in - the same
        // kernels on the same input into a buffer of their own, so the same bits; that layer then runs its gate alone
        for (size_t ej = 0; ej < eg_pre_kv.size(); ++ej)
            if (eg_pre_kv[ej] && hp.eg_layers[ej] == il + 1) {
                // (eg_ring: the first engram layer's - the window's rows are still being fetched: the helper's word first)
                if (m.eg_ring && il + 1 == m.eg_late_layer) doorbell_wait(dd_egf, dd_abort, d_eg_cnt, s);
                engram_rows(il + 1, 0, T, eg_win_d, kTmax, eg_pre_att, eg_pre_kv[ej]);
            }
        doorbell_wait(dd_flag, dd_abort, d_wait_cnt, s);
        if (lat) {   // the sum in the latent, routed_up, the shared expert, the post-norm (hy4.cpp)
            moe_combine_pairs(hit_parts, dd_rows, dd_mask, wts, nullptr, hml, nx, k, T, s);
            hy4_moe_out(il, T, out);
        } else {
            moe_combine_pairs(hit_parts, dd_rows, dd_mask, wts, hp.shexp ? shared : nullptr, out, n, k, T, s);
        }
    }
    mix_out(il, T, out, R, coef, false);
}

bool Stage::capture_window(int T, bool keep, std::string& err) {
    cudaSetDevice(dev);
    use(win);
    gpos = d_pos;
    cudaGraph_t g = nullptr;
    std::string why;
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        gpos = nullptr;
        cudaGetLastError();
        return m.fail(err, "cannot begin a graph capture on GPU " + std::to_string(dev));
    }
    try {
        const int n = (int) hp.n_embd;
        // the inputs (mapped): the position and this stage's residency slice; the first stage's embeddings broadcast to the four
        // mHC streams, a later stage's the previous stage's residuals
        const int hcn = (int) hp.hc * n;   // the residual streams (mHC's 4, or MiMo's one)
        if (di == 0) window_load(in_pos_d, in_rows_d, in_res_d + res_lo, d_pos, d_res + res_lo, (int) res_n, R, T, hcn, n, s);
        else window_load(in_pos_d, m.stages[(size_t) di - 1]->out_rows_d, in_res_d + res_lo, d_pos, d_res + res_lo, (int) res_n, R, T,
                         hcn, hcn, s);
        if (hp.v41) {   // DeepSeek-V4.1's carried pre-mix: stream 0 before layer 0, else the previous GPU's
            if (di == 0) v41_pre_reset(T);
            else store_mapped(m.stages[(size_t) di - 1]->out_rows_d + (size_t) kTmax * hcn, cpre, (size_t) T * 24, s);
        }
        const int64_t pen_le = std::min<int64_t>(le, hp.n_layer);   // moex (§33): the route bias's rows, once a window
        if (!m.route_pen_dev.empty() && m.route_pen_dev[(size_t) di] && pen_le > lb)
            store_mapped(m.route_pen + lb * hp.n_expert, m.route_pen_dev[(size_t) di] + lb * hp.n_expert, (size_t) (pen_le - lb) * hp.n_expert, s);
        for (int64_t il = lb; il < le; ++il) layer_graph_body((int) il, T, keep);
        if (!last) {
            store_mapped(R, out_rows_d, (size_t) T * hcn, s);   // the residuals for the next GPU
            if (hp.v41) store_mapped(cpre, out_rows_d + (size_t) kTmax * hcn, (size_t) T * 24, s);
        }
        else {
            dspark_capture((int) hp.n_layer, 0, T);                      // DSpark: the trunk's output
            dflash_capture((int) hp.n_layer, 0, T, false);               // DFlash: ... and its
        }
    } catch (const std::exception& e) {
        why = e.what();
    }
    const cudaError_t ce = cudaStreamEndCapture(s, &g);
    gpos = nullptr;
    if (!why.empty() || ce != cudaSuccess || !g) {
        cudaGetLastError();
        if (g) cudaGraphDestroy(g);
        return m.fail(err, "graph capture on GPU " + std::to_string(dev) + " failed: " +
                               (why.empty() ? std::string(cudaGetErrorString(ce)) : why));
    }
    cudaGraphExec_t ex = nullptr;
    const cudaError_t ie = cudaGraphInstantiate(&ex, g, 0);
    cudaGraphDestroy(g);
    if (ie != cudaSuccess) {
        cudaGetLastError();
        return m.fail(err, std::string("graph instantiation failed: ") + cudaGetErrorString(ie));
    }
    gx[T * 2 + (keep ? 1 : 0)] = ex;
    if (m.opt.verbose)
        fprintf(stderr, "strata-x: GPU %d: captured the decode graph for windows of %d%s (layers %lld-%lld)\n", dev, T,
                keep ? " + MTP snapshots" : "", (long long) lb, (long long) le - 1);
    return true;
}

bool Stage::graph_ready(int T, bool keep, std::string& err) {
    const int key = T * 2 + (keep ? 1 : 0);
    if (gx.count(key)) return true;
    if (++gx_uses[key] < 2) return false;   // the first use of a variant runs eagerly: it loads every kernel
    return capture_window(T, keep, err);
}

/// The host's half of MoE layer il for a window of T tokens: the doorbell, the pairs classified by the residency table,
/// the misses on the CPU pool (rows straight into the mapped block), the flag.
bool Stage::service_moe(int il, int T, std::string& err) {
    const Layer& l = L[(size_t) il];
    const int E = (int) hp.n_expert, k = (int) hp.n_used;
    const int n = l.r_down ? (int) hp.ex() : (int) hp.n_embd;   // the experts' width (Hy4-flash, ERNIE-5.5: the latent)
    const uint32_t want = ++host_seq;
    const auto tw = clk::now();
    uint32_t spins = 0;
    while ((int32_t) (*db_seq - want) < 0) {   // memory only: no driver call
        _mm_pause();
        if ((++spins & 1023u) == 0 && ms_since(tw) > 30000.0) {
            release_gpu_waits();
            return m.fail(err, "the GPU never published layer " + std::to_string(il) + "'s routing (30 s)");
        }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    st.ms_wait += ms_since(tw);
    m.trace_route(il, T, k, db_ids);
    struct Miss {
        int e, nt;
        int t[kc::MAXT], j[kc::MAXT];
    };
    std::vector<Miss> misses;
    const bool hits_on = hit_ok[(size_t) il] != 0;
    // DeepSeek-V4.1's read-ahead: the next layer's guessed experts (published with this layer's routing) - their reads queue
    // behind this layer's demand reads, which run_cpu_jobs submits below
    std::vector<int32_t> pred;
    const bool want_pred = m.ahead_from(il) && db_pred && db_pred_seq;
    if (want_pred) ++pred_expect;
    // (moex, §32) the guess comes after the doorbell: waited for where it is read, at most 2 ms (a late one reads none ahead)
    auto take_pred = [&] {
        if (!want_pred) return;
        const auto tq = clk::now();
        while ((int32_t) (*db_pred_seq - pred_expect) < 0)
            if (ms_since(tq) > 2.0) return;
        std::atomic_thread_fence(std::memory_order_acquire);
        pred.assign(db_pred, db_pred + T * (int) hp.k_of(il + m.ahead_d));
    };
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < k; ++j) {
            const int e = db_ids[t * k + j];
            if (e < 0 || e >= E) {
                release_gpu_waits();
                return m.fail(err, "router produced expert " + std::to_string(e));
            }
            const size_t key = (size_t) il * E + (size_t) e;
            m.heat[key] += 1.0f;
            m.routed[key]++;
            if (hits_on && m.n_slots && m.resident[key] >= 0) {
                db_mask[t * k + j] = 0;
                st.hits++;
                continue;
            }
            db_mask[t * k + j] = 1;
            st.misses++;
            m.token_misses.push_back((int32_t) key);
            auto it = std::find_if(misses.begin(), misses.end(), [&](const Miss& x) { return x.e == e && x.nt < kc::MAXT; });
            if (it == misses.end()) {
                misses.push_back(Miss{e, 0, {}, {}});
                it = misses.end() - 1;
            }
            it->t[it->nt] = t;
            it->j[it->nt] = j;
            it->nt++;
        }
    const auto t1 = clk::now();
    if (!misses.empty()) {
        for (int t = 0; t < T; ++t)
            kc::native_quant_act(l.fmt, db_x + (size_t) t * n, m.act_q.data() + (size_t) t * kc::kNativeActBytes);
        std::vector<kc::ExpertJobMulti> jobs(misses.size());
        std::vector<int64_t> keys(misses.size());
        for (size_t mi = 0; mi < misses.size(); ++mi) {
            kc::ExpertJobMulti& jb = jobs[mi];
            const int e = misses[mi].e;
            keys[mi] = (int64_t) il * E + e;
            const Engine::Impl::Src sr = m.src((int64_t) il * E + e);
            jb.blob = sr.g;
            // one [g | u | d] blob (the RAM arena's): the pool's blob form, whose multi-token kernels decode each row once
            // for all the expert's tokens; an expert in place in the file mapping: the split form (ggml-cpu's rows)
            const bool one_blob = sr.u == sr.g + l.fmt.up_off && sr.d == sr.g + l.fmt.down_off;
            jb.up = one_blob ? nullptr : sr.u;
            jb.down = one_blob ? nullptr : sr.d;
            jb.nt = misses[mi].nt;
            for (int i = 0; i < jb.nt; ++i) {
                jb.nact[i] = m.act_q.data() + (size_t) misses[mi].t[i] * kc::kNativeActBytes;
                jb.out[i] = db_rows + ((size_t) misses[mi].t[i] * k + misses[mi].j[i]) * n;
            }
        }
        // this layer's demand reads first, then the next layer's guesses queued behind them (run_cpu_jobs)
        take_pred();
        if (!m.run_cpu_jobs(il, jobs, keys, err, pred.empty() ? nullptr : &pred, il + m.ahead_d)) return false;
        pred.clear();
    } else {
        take_pred();
    }
    if (!pred.empty()) m.ssd_ahead(il + m.ahead_d, pred.data(), (int) pred.size());   // no CPU experts here: the guesses at once
    st.ms_pool += ms_since(t1);
    std::atomic_thread_fence(std::memory_order_release);
    *db_flag = ++host_flag;   // the GPU's wait kernel sees it and the combine reads the rows
    return true;
}

}  // namespace strata::x
