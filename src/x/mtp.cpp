// src/x/mtp.cpp - THE DRAFTER: Strata's MtpDrafter (src/core/mtp.cpp, STRATA_QWEN_DESIGN.md 3.6) for GLM's NextN block.
//
//   cell i of the block's sequence pairs the trunk's post-norm hidden at position i with the token at position i + 1:
//       z = eh_proj([enorm(emb(tok)) | hnorm(h)])  ->  one pre-norm block (the DSA attention over the block's own caches,
//       its own MoE)  ->  shared_head_norm  ->  the shared head;     its output predicts the token at position i + 2, and
//       its own RESIDUAL feeds the next draft step (the chain).
//
// What makes Strata's draft step cheap, and what this file does the same way:
//   * the drafter's experts are ALL in VRAM for good (Strata: "all 512 experts resident in Q2_0, ~700 MB"; here the 288 of
//     the NextN layer in pinned tier slots): a draft step never waits for the host, so
//   * the step is ONE CAPTURED GRAPH per row count (a catch-up over the accepted rows, or one chain step), its inputs -
//     the embeddings, the hidden rows, the first position - in device buffers the host fills before each launch;
//   * the head runs on the last row only (the rows before it are tokens already known), and the same launch gives the
//     draft's PROBABILITY, which is what spec-min-p gates the chain on.
// A speculative cell (a chain step's, a rejected window row's) is just overwritten when its position is processed again:
// every cache the block has is positional.
#include "engine_impl.hpp"

#include <cstdio>
#include <cstring>

namespace strata::x {

/// The block's MoE with every routed expert in this GPU's slots: the router, the shared expert, the pairs' slots from the
/// residency table, the grouped launches, the combine - layer_graph_body's MoE without the doorbell (no CPU rows).
void Stage::mtp_moe_resident(int il, int T) {
    const Layer& l = L[(size_t) il];
    const int n = (int) hp.n_embd, E = (int) hp.n_expert, k = (int) hp.n_used, nf = (int) hp.n_ff_exp;
    const kc::NativeFmt& f = l.fmt;
    gemv_f32_w(l.gate_inp.f(), cur, router, n, E, T, s);
    moe_route_w(router, l.probs_b.f(), E, k, hp.ew_norm, hp.ew_scale, ids, wts, T, s);
    swiglu_ffn(il, T, l.sh_gate, l.sh_up, l.sh_down, hp.clamp_sh[(size_t) il], cur, shared);   // leaves cur's Q8_1 columns in xq
    hit_slots(ids, d_res + (size_t) il * E, d_hslot, T * k, s);
    kn::native_mmvq_grouped(f.gu_type, m.nx_base, m.nx_bytes, 0, f.up_off, d_hslot, xq, k, hit_g, (size_t) nf, hit_u, (size_t) nf, n, nf,
                            T * k, s);
    swiglu_clamp(hit_g, hit_u, hit_g, T * k * nf, f.swiglu_limit, s);
    quant(hit_g, hit_xq, T * k * nf, 1);
    kn::native_mmvq_grouped(f.d_type, m.nx_base, m.nx_bytes, f.down_off, 0, d_hslot, hit_xq, 1, hit_parts, (size_t) n, nullptr, 0, nf, n,
                            T * k, s);
    moe_combine_pairs(hit_parts, hit_parts, d_zero_mask, wts, shared, out, n, k, T, s);
}

bool Stage::mtp_body(int head, int Tm, int64_t pos0, bool last_head, bool res, std::string& err) {
    if (hp.arch == Arch::dsv4) return dsv4_mtp_body(Tm, pos0, last_head, err);   // dsv4_mtp.cpp
    const int n = (int) hp.n_embd, il = (int) hp.n_layer + head;
    const Layer& l = L[(size_t) il];
    if (hp.arch == Arch::mimo2) {   // mimo2.cpp graph_mtp: [enorm(e) | hnorm(h)] -> eh_proj -> GQA -> dense FFN -> out norm -> head
        try {
            rms_norm(mtp_e, l.enorm.f(), mtp_e, Tm, n, hp.rms_eps, s);
            rms_norm(mtp_hin, l.hnorm.f(), mtp_h, Tm, n, hp.rms_eps, s);
            // [e | h], mainline's order (checked 10-08: MiMo-7B's [h | e] order takes the acceptance from 16% to 0%)
            concat_rows(mtp_e, mtp_h, mtp_cat, n, Tm, s);
            quant(mtp_cat, xq, 2 * n, Tm);
            pj(l.eh_proj, mtp_cat, xq, mtp_z, Tm);
            rms_norm(mtp_z, l.attn_norm.f(), cur, Tm, n, hp.rms_eps, s);
            gqa(il, pos0, Tm, cur, out);
            add_f32(mtp_z, out, mtp_z, Tm * n, s);
            rms_norm(mtp_z, l.ffn_norm.f(), cur, Tm, n, hp.rms_eps, s);
            swiglu_ffn(il, Tm, l.ffn_gate, l.ffn_up, l.ffn_down, 0.0f, cur, out);
            add_f32(mtp_z, out, mtp_z, Tm * n, s);
            const float* zl = mtp_z + (size_t) (Tm - 1) * n;
            cudaMemcpyAsync(mtp_chain + (size_t) head * n, zl, (size_t) n * sizeof(float), cudaMemcpyDeviceToDevice, s);
            if (last_head) {
                rms_norm(zl, l.out_norm.f(), mtp_hnlast, 1, n, hp.rms_eps, s);
                quant(mtp_hnlast, xq, n, 1);
                mm_t(m.output, xq, mtp_logits, 1);
                draft_pick();
            }
        } catch (const std::exception& e) {
            return m.fail(err, e.what());
        }
        const cudaError_t ce = cudaGetLastError();
        if (ce != cudaSuccess) return m.fail(err, cudaGetErrorString(ce));
        return true;
    }
    try {
        rms_norm(mtp_e, m.enorm.f(), mtp_e, Tm, n, hp.rms_eps, s);
        rms_norm(mtp_hin, m.hnorm.f(), mtp_h, Tm, n, hp.rms_eps, s);
        concat_rows(mtp_e, mtp_h, mtp_cat, n, Tm, s);
        quant(mtp_cat, xq, 2 * n, Tm);
        pj(m.eh_proj, mtp_cat, xq, mtp_z, Tm);
        rms_norm(mtp_z, l.attn_norm.f(), cur, Tm, n, hp.rms_eps, s);
        dsa(il, pos0, Tm, cur, out);
        add_f32(mtp_z, out, mtp_z, Tm * n, s);
        rms_norm(mtp_z, l.ffn_norm.f(), cur, Tm, n, hp.rms_eps, s);
        if (res) mtp_moe_resident(il, Tm);
        else if (!moe_window(il, Tm, err)) return false;
        add_f32(mtp_z, out, mtp_z, Tm * n, s);
        if (last_head) {
            const float* zl = mtp_z + (size_t) (Tm - 1) * n;
            cudaMemcpyAsync(mtp_zlast, zl, (size_t) n * sizeof(float), cudaMemcpyDeviceToDevice, s);   // the chain's hidden
            rms_norm(zl, m.shead_norm.f(), mtp_hnlast, 1, n, hp.rms_eps, s);
            quant(mtp_hnlast, xq, n, 1);
            mm_t(m.output, xq, mtp_logits, 1);
            draft_pick();
        } else {
            rms_norm(mtp_z, m.shead_norm.f(), mtp_h, Tm, n, hp.rms_eps, s);
            quant(mtp_h, xq, n, Tm);
            mm_t(m.output, xq, mtp_logits, Tm);
            argmax_w(mtp_logits, (int) hp.n_vocab, mtp_amax, Tm, s);
        }
    } catch (const std::exception& e) {
        return m.fail(err, e.what());
    }
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) return m.fail(err, cudaGetErrorString(ce));
    return true;
}

/// The draft of the drafter's last row (mtp_logits[0]) into mtp_amax[0] / mtp_prob[0]: its argmax and that token's
/// softmax probability, or - coupled draft sampling, cd_on - a sample with the request's chain and the verify row's own
/// uniform (strata::kernels::coupled_draft_sample, counter cd_cell + 1) and its probability under the final distribution.
void Stage::draft_pick() {
    if (cd_on)
        strata::kernels::coupled_draft_sample(mtp_logits, (int) hp.n_vocab, nullptr, nullptr, (int) hp.n_vocab, cd_params, cd_ring,
                                              kCdCap, 0, cd_cell_d, cd_scratch, mtp_amax, mtp_prob, s);
    else
        argmax_prob_w(mtp_logits, (int) hp.n_vocab, mtp_amax, mtp_prob, 1, s);
}

bool Stage::mtp_capture(int head, int Tm, bool draft, std::string& err) {
    cudaSetDevice(dev);
    use(win);
    gpos = mtp_pos;
    cudaGraph_t g = nullptr;
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        gpos = nullptr;
        cudaGetLastError();
        return m.fail(err, "cannot begin the drafter's graph capture");
    }
    std::string why;
    const int n = (int) hp.n_embd;
    window_load(mtp_in_pos_d, mtp_in_rows_d, nullptr, mtp_pos, nullptr, 0, mtp_e, Tm, n, n, s);   // the inputs (mapped)
    const bool ok = mtp_body(head, Tm, 0, draft, true, why);
    if (ok && draft) store_draft(mtp_amax, mtp_prob, mtp_out_tok_d, mtp_out_prob_d, s);             // the output (mapped)
    const cudaError_t ce = cudaStreamEndCapture(s, &g);
    gpos = nullptr;
    if (!ok || ce != cudaSuccess || !g) {
        cudaGetLastError();
        if (g) cudaGraphDestroy(g);
        return m.fail(err, "the drafter's graph capture failed: " + (why.empty() ? std::string(cudaGetErrorString(ce)) : why));
    }
    cudaGraphExec_t ex = nullptr;
    const cudaError_t ie = cudaGraphInstantiate(&ex, g, 0);
    cudaGraphDestroy(g);
    if (ie != cudaSuccess) {
        cudaGetLastError();
        return m.fail(err, std::string("the drafter's graph instantiation failed: ") + cudaGetErrorString(ie));
    }
    gx_mtp[mtp_key(head, Tm, draft)] = ex;
    if (m.opt.verbose)
        fprintf(stderr, "strata-x: GPU %d: captured the drafter graph for block %d, %d row%s%s\n", dev, head, Tm, Tm == 1 ? "" : "s",
                draft ? " + its draft" : "");
    return true;
}

bool Stage::mtp_run(int head, int Tm, const int32_t* toks, const float* h_src, int64_t pos0, bool draft, int32_t& tok, float& prob,
                    std::string& err) {
    cudaSetDevice(dev);
    use(win);
    const int n = (int) hp.n_embd;
    if (Tm < 1 || Tm > kTmax) return m.fail(err, "drafter rows");
    // the inputs through mapped memory and a copy kernel: nothing here waits behind the tier's swap DMAs on a copy engine
    const bool own_embd = m.dmtp && m.dmtp->embd;   // DeepSeek-V4's head reads its own embedding table (mainline's draft model)
    if (!(own_embd ? mtp_embd_rows(toks, Tm, mtp_in_rows_h, pos0 + 1) : embd_rows(toks, Tm, mtp_in_rows_h, pos0 + 1)))
        return m.fail(err, "MTP token out of range");   // toks[0] sits at pos0 + 1
    *mtp_in_pos_h = pos0;
    // the draft's MTP cell: its last row's position (it predicts the token two past it); the verify row that checks it
    // draws Philox(seed, cell + 1) - the next window's row j for chain step j (Strata's coupled_draft_counter)
    if (cd_cell_h) *(volatile int32_t*) cd_cell_h = (int32_t) (pos0 + Tm - 1);
    if (h_src != mtp_hin) copy_f32(h_src, mtp_hin, (size_t) Tm * mtp_hw(), s);
    const int key = mtp_key(head, Tm, draft);
    bool graph = m.graphs_on && mtp_resident && !m.mtp_graphs_off;
    if (graph && !gx_mtp.count(key)) {
        if (++gx_mtp_uses[key] >= 2) {   // the first use of a variant runs eagerly
            std::string cerr;
            if (!mtp_capture(head, Tm, draft, cerr)) {
                fprintf(stderr, "strata-x: drafter graphs off: %s\n", cerr.c_str());
                m.mtp_graphs_off = true;
            }
        }
        graph = gx_mtp.count(key) != 0;
    }
    static const bool timing = std::getenv("STRATA_X_MTP_TIMING") != nullptr;
    static cudaEvent_t e0 = nullptr, e1 = nullptr;
    static double sum_gpu = 0, sum_host = 0;
    static int cnt = 0;
    const auto th = clk::now();
    static double sum_launch = 0;
    if (timing && !e0) {
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
    }
    if (timing) cudaEventRecord(e0, s);
    if (graph) {
        if (cudaGraphLaunch(gx_mtp[key], s) != cudaSuccess) return m.fail(err, "the drafter's graph launch failed");
    } else {
        window_load(mtp_in_pos_d, mtp_in_rows_d, nullptr, mtp_pos, nullptr, 0, mtp_e, Tm, n, n, s);
        if (!mtp_body(head, Tm, pos0, draft, mtp_resident, err)) return false;
        if (draft) store_draft(mtp_amax, mtp_prob, mtp_out_tok_d, mtp_out_prob_d, s);
    }
    if (timing) {
        cudaEventRecord(e1, s);
        sum_launch += ms_since(th);
    }
    if (!draft) return true;   // a catch-up that only fills a block's caches: nothing to read back, nothing to wait for
    if (cudaStreamSynchronize(s) != cudaSuccess) return m.fail(err, "the draft readback failed");
    tok = *(volatile int32_t*) mtp_out_tok_h;
    prob = *(volatile float*) mtp_out_prob_h;
    if (timing) {
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        sum_gpu += ms;
        sum_host += ms_since(th);
        if (++cnt % 50 == 0)
            fprintf(stderr, "strata-x: drafter step (%s, %d rows): GPU %.2f ms, launch call %.2f ms, launch to readback %.2f ms (mean of %d)\n",
                    graph ? "graph" : "eager", Tm, sum_gpu / cnt, sum_launch / cnt, sum_host / cnt, cnt);
    }
    return true;
}

int Stage::mtp_heads() const {
    if (m.dsp) return m.dsp->block;   // DSpark: one block of drafts per step
    if (m.dfl) return m.dfl->block - 1;   // DFlash: the block's positions after the anchor
    if (m.dfy) return m.dfy->block - 1;   // DFly2: the block's positions after the bonus row
    return hp.arch == Arch::mimo2 ? (int) (m.n_load - hp.n_layer) : 1;
}

bool Engine::Impl::drafter_read(int T, const int32_t* next_toks, int64_t pos0, std::string& err) {
    Stage& lg = last_stage();
    if (dsp) return lg.dspark_inject(T, pos0, err);
    if (dfl) return lg.dflash_inject(T, pos0, false, err);
    if (dfy) return true;   // DFly2 reads the trunk's own caches: nothing of the prompt to take in
    int32_t tok = 0;
    float prob = 0;
    return lg.mtp_catchup(T, next_toks, lg.mtp_hsrc(), pos0, tok, prob, err);
}

bool Stage::mtp_catchup(int T, const int32_t* toks, const float* h_src, int64_t pos0, int32_t& tok, float& prob, std::string& err) {
    if (m.dsp) {   // DSpark: the window's kept rows into the rings, then the next block of drafts after its last emitted token
        // STRATA_DSV41_DSPARK_ROWS (opt-in, A/B): N > 0 drafts at most N positions, -1 as many as the window verifies (--spec W:
        // W - 1); unset, DSpark's whole block (5).  A window of 3 reads 2 drafts: the other rows cost the drafter's three blocks
        // (their experts on the CPU pool) and Markov-head syncs for nothing - but the block attends across its rows, so the
        // drafts themselves can change (the trunk verifies them: the output does not)
        static const int rows_env = std::getenv("STRATA_DSV41_DSPARK_ROWS") ? std::atoi(std::getenv("STRATA_DSV41_DSPARK_ROWS")) : 0;
        int nd = (int) m.dsp->block;
        if (rows_env > 0) nd = std::min(nd, rows_env);
        else if (rows_env < 0 && m.spec_cap > 1) nd = std::min(nd, (m.req_cap > 1 ? std::min(m.req_cap, m.spec_cap) : m.spec_cap) - 1);
        if (!dspark_inject(T, pos0, err) || !dspark_draft(toks[T - 1], pos0 + T, nd, err)) return false;
        tok = dsp_n > 0 ? dsp_tok[0] : 0;
        prob = dsp_n > 0 ? dsp_prob[0] : 0.0f;
        return true;
    }
    if (m.dfl) {   // DFlash: the window's kept rows into the rings, then the block after its last emitted token
        if (!dflash_step(T, toks[T - 1], pos0, err)) return false;
        tok = df_n > 0 ? df_tok[0] : 0;
        prob = df_n > 0 ? df_prob[0] : 0.0f;
        return true;
    }
    if (m.dfy) {   // DFly2: the block after the window's last emitted token, from its last kept row
        if (!dfly_step(T, toks[T - 1], pos0, err)) return false;
        tok = dfy_n > 0 ? dfy_tok[0] : 0;
        prob = dfy_n > 0 ? dfy_prob[0] : 0.0f;
        return true;
    }
    // moex: only the blocks a window can draft with keep their caches (mainline's draft-mtp catches up every block whatever
    // its n_max): with --spec W, blocks 0 .. W - 2; MiMo at --spec 2 skips two of its three block passes a window
    const int cap_w = m.spec_cap > 0 ? std::min(m.spec_cap, m.snap_T) : m.snap_T;
    const int nh = std::max(1, std::min(mtp_heads(), cap_w - 1));
    for (int hd = nh - 1; hd >= 0; --hd)   // block 0 last: its draft is the one read back (the others only fill their caches)
        if (!mtp_run(hd, T, toks, h_src, pos0, hd == 0, tok, prob, err)) return false;
    return true;
}

bool Stage::mtp_chain_step(int j, const int32_t* d, int64_t pos, int32_t& tok, float& prob, std::string& err) {
    if (m.dsp) {   // DSpark drafted the whole block in the catch-up
        if (j >= dsp_n) return m.fail(err, "a DSpark chain step past its block");
        tok = dsp_tok[j];
        prob = dsp_prob[j];
        return true;
    }
    if (m.dfl) {   // ... and DFlash
        if (j >= df_n) return m.fail(err, "a DFlash chain step past its block");
        tok = df_tok[j];
        prob = df_prob[j];
        return true;
    }
    if (m.dfy) {   // ... and DFly2
        if (j >= dfy_n) return m.fail(err, "a DFly2 chain step past its block");
        tok = dfy_tok[j];
        prob = dfy_prob[j];
        return true;
    }
    if (hp.arch != Arch::mimo2) {
        // DeepSeek-V4: the block's four output streams always (llama.cpp's t_h_nextn); GLM: its residual or its normed form
        const float* chain_h = m.mtp_chain_res || hp.arch == Arch::dsv4 ? mtp_zlast : mtp_hnlast;
        return mtp_run(0, 1, &d[j - 1], chain_h, pos + j - 1, true, tok, prob, err);
    }
    if (j >= mtp_heads()) return m.fail(err, "a chain step past the MTP blocks");
    // block j over rows pos .. pos + j - 1: draft i (the token after row i) with block i's residual at its last row
    return mtp_run(j, j, d, mtp_chain, pos, true, tok, prob, err);
}

}  // namespace strata::x
