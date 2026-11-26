// src/x/spec.cpp - THE DECODE LOOP: Strata's serial speculative window loop (src/program/generate.cpp's serve loop, STRATA_QWEN_DESIGN.md 3.6)
// for GLM.
//
//   window = [x, d_0 .. d_(nd-1)]: the fed-back token and the drafts that passed spec-min-p (draft j enters while drafts
//            0..j each have probability >= p; the chain stops launching steps once one falls below)
//   verify = one trunk pass over the window (a captured graph per row count, the KDA state snapshotted after every row)
//   accept = the longest prefix of drafts the trunk's OWN token at that row equals - the arg-max, or the sampler's draw
//            keyed by (seed, position) - so a draft changes the speed, never the output; the window emits a + 1 tokens
//   commit = the KDA state rolls back to the last kept row; the drafter catches up over the kept rows (its last row is the
//            next window's first draft) - the positional caches of both just get overwritten when a position returns
#include "engine_impl.hpp"

#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::x {

namespace {
/// The request's sampler chain as the kernels take it (sample_rows' and the coupled drafter's: the same one).
kn::SamplerParams chain_of(const SamplingParams& sp, int hist_cap, bool pen) {
    kn::SamplerParams p;
    p.top_k = sp.top_k;
    p.full_vocab = sp.top_k <= 0;   // no top_k cut: the whole vocabulary when top_p and min_p cut nothing either
    p.temp_first = sp.temp_first;
    p.top_p = sp.top_p;
    p.min_p = sp.min_p;
    p.temperature = sp.temperature;
    p.penalty_last_n = pen ? std::min(sp.penalty_last_n, hist_cap) : 0;
    p.penalty_repeat = sp.penalty_repeat;
    p.penalty_freq = sp.penalty_freq;
    p.penalty_present = sp.penalty_present;
    p.seed = sp.seed;
    p.greedy = sp.temperature <= 0.0f;
    return p;
}
bool penalized(const SamplingParams& sp) {
    return sp.penalty_last_n > 0 && (sp.penalty_repeat != 1.0f || sp.penalty_freq != 0.0f || sp.penalty_present != 0.0f);
}
}  // namespace

// ------------------------------------------------------------------ DFlash's cost-aware window

int Engine::Impl::DflashGate::pbin(float p) {
    static const float edges[kP - 1] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.85f, 0.9f, 0.93f, 0.96f, 0.98f, 0.99f, 0.995f};
    int b = 0;
    while (b < kP - 1 && p >= edges[b]) ++b;
    return b;
}

double Engine::Impl::DflashGate::q(int depth, float p) const {
    const int dc = std::min(depth, kD - 1), pb = pbin(p);
    // the prior of a cell: the draft's own probability, worth three drafts (measured on MiMo at temperature 1 with coupled
    // drafts: acceptance tracks p bucket by bucket - 0.96-0.98 at p >= 0.995, ~0.63 at 0.6-0.7)
    constexpr double w0 = 3.0;
    return (ok[dc][pb] + w0 * (double) p) / (all[dc][pb] + w0);
}

double Engine::Impl::DflashGate::c(int T) const {
    T = std::clamp(T, 1, kTmax);
    if (cost_n[T] > 0) return cost[T];
    // an unseen row count: the first row's cost times the extra rows' measured slope (the prior: 0.6 of a row each)
    double c1 = cost_n[1] > 0 ? cost[1] : 0.0, slope = 0.6;
    for (int t = kTmax; t >= 2; --t)
        if (cost_n[t] > 0) {
            if (c1 <= 0.0) c1 = cost[t] / (1.0 + slope * (t - 1));
            slope = std::max(0.05, (cost[t] / c1 - 1.0) / (t - 1));
            break;
        }
    if (c1 <= 0.0) c1 = 1.0;   // nothing measured yet: relative units
    return c1 * (1.0 + slope * (T - 1));
}

int Engine::Impl::DflashGate::choose(const float* p, int nmax) {
    // the long run's tokens per ms are what counts, not each window's: a window takes n* = argmax E(n) - lam cost(n), lam the
    // rate the rounds achieve (the renewal-reward optimum; maximising each window's own ratio cuts the confident windows short,
    // which measured 23.1 against a fixed gate's 25.5 T/s on prose)
    const double d = draft_n > 0 && cost_n[1] > 0 ? draft_ms : 0.0;   // the draft step after the window: paid whatever n is
    const double l = lam > 0.0 ? lam : 1.0 / (c(1) + d);
    double E = 1.0, P = 1.0, best = E - l * (c(1) + d);
    int bn = 0;
    for (int j = 0; j < nmax; ++j) {
        P *= q(j, p[j]);
        E += P;
        const double v = E - l * (c(j + 2) + d);
        if (v > best) {
            best = v;
            bn = j + 1;
        }
    }
    // exploration: one window in eight verifies one draft more than the choice, so the cells the choice avoids (low
    // probabilities, deeper drafts) keep being measured - otherwise a cell's estimate could only ever be its prior
    if (++windows % 8 == 0 && bn < nmax) {
        ++bn;
        ++explored;
    }
    hist[std::min(bn, kTmax - 1)]++;
    return bn;
}

void Engine::Impl::DflashGate::print() const {
    fprintf(stderr, "strata-x: DFlash window policy: verify ms by rows");
    for (int T = 1; T <= kTmax; ++T) fprintf(stderr, " %d:%.1f%s", T, c(T), cost_n[T] > 0 ? "" : "*");
    fprintf(stderr, ", draft step %.2f ms, rate %.2f T/s; drafts chosen (n: windows)", draft_ms, lam * 1000.0);
    for (int n = 0; n < kTmax; ++n)
        if (hist[n]) fprintf(stderr, " %d:%lld", n, (long long) hist[n]);
    fprintf(stderr, " (%lld explored)\n", (long long) explored);
    for (int dc = 0; dc < kD; ++dc) {
        fprintf(stderr, "strata-x:   depth %d%s acceptance by draft probability:", dc + 1, dc == kD - 1 ? "+" : "");
        for (int b = 0; b < kP; ++b)
            if (all[dc][b] >= 0.5) fprintf(stderr, " [%d] %.2f (%.0f)", b, ok[dc][b] / all[dc][b], all[dc][b]);
        fprintf(stderr, "\n");
    }
}

void Engine::Impl::DflashGate::observe(const float* p, int nd, int accepted) {
    for (auto& row : ok)
        for (double& v : row) v *= 0.995;
    for (auto& row : all)
        for (double& v : row) v *= 0.995;
    for (int j = 0; j < nd && j <= accepted; ++j) {   // draft j was reached when the ones before it held
        const int dc = std::min(j, kD - 1), pb = pbin(p[j]);
        all[dc][pb] += 1.0;
        if (j < accepted) ok[dc][pb] += 1.0;
    }
}

void Engine::Impl::DflashGate::observe_rate(int tokens, double round_ms) {
    if (round_ms <= 0.0) return;
    if (ms_ema <= 0.0) {
        tok_ema = (double) tokens;
        ms_ema = round_ms;
    } else {
        tok_ema += 0.02 * ((double) tokens - tok_ema);
        ms_ema += 0.02 * (round_ms - ms_ema);
    }
    lam = tok_ema / ms_ema;   // a ratio of means (the long run's), not a mean of the rounds' ratios
}

void Engine::Impl::DflashGate::observe_cost(int T, double verify_ms, double step_ms) {
    T = std::clamp(T, 1, kTmax);
    cost[T] = cost_n[T] > 0 ? cost[T] + 0.1 * (verify_ms - cost[T]) : verify_ms;
    cost_n[T] += 1.0;
    if (step_ms > 0.0) {
        draft_ms = draft_n > 0 ? draft_ms + 0.1 * (step_ms - draft_ms) : step_ms;
        draft_n += 1.0;
    }
}

/// The #879 guard's flag words (mapped, on the last stage's GPU), set up at the first window; false: the guard is off
/// (STRATA_NAN_GUARD=0, or no mapped memory - said once, the windows then run unchecked as before).
bool Engine::Impl::nan_guard_ready(Stage& g) {
    if (nf_on == 0) {
        const char* e = std::getenv("STRATA_NAN_GUARD");
        nf_on = e && e[0] == '0' ? -1 : 1;
        if (nf_on > 0) {
            const size_t bytes = (size_t) kTmax * kn::kNonfiniteBlocks * sizeof(uint32_t);
            void* hp_ = nullptr;
            void* dp_ = nullptr;
            if (cudaHostAlloc(&hp_, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                cudaHostGetDevicePointer(&dp_, hp_, 0) != cudaSuccess) {
                cudaGetLastError();
                if (hp_) cudaFreeHost(hp_);
                fprintf(stderr, "strata-x: the non-finite logits guard is off: no mapped memory for its flags\n");
                nf_on = -1;
            } else {
                nf_h = static_cast<uint32_t*>(hp_);
                nf_d = static_cast<uint32_t*>(dp_);
                std::memset(nf_h, 0, bytes);
            }
        }
    }
    (void) g;
    return nf_on > 0;
}

namespace {
/// STRATA_NAN_INJECT=n[,n...] (a test hook, as upstream's): the n-th window the guard checks (from 1) reports row 0.
bool nan_inject(int64_t window) {
    static const std::vector<int64_t> at = [] {
        std::vector<int64_t> v;
        if (const char* e = std::getenv("STRATA_NAN_INJECT"))
            for (const char* q = e; *q;) {
                char* end = nullptr;
                const long long x = std::strtoll(q, &end, 10);
                if (end == q) break;
                if (x > 0) v.push_back(x);
                q = *end == ',' ? end + 1 : end;
            }
        return v;
    }();
    return std::find(at.begin(), at.end(), window) != at.end();
}
}  // namespace

bool Engine::Impl::sample_rows(const SamplingParams& sp, int64_t counter_pos, const int32_t* win, int T, const int32_t* hist,
                               int hist_n, int32_t* out, std::string& err) {
    Stage& g = last_stage();
    set_dev(g.dev);
    nf_row = -1;
    const bool guard = nan_guard_ready(g);
    if (guard) kn::logits_nonfinite_rows(g.logits, T, (int) hp.n_vocab, nf_d, g.s);   // before the sampler reads the logits
    first_top_probe(g.logits, counter_pos);   // STRATA_X_FIRST_TOP (a parity probe; off by default)
    const bool greedy = sp.temperature <= 0.0f;
    const bool pen = penalized(sp);
    if (greedy && !pen) {
        argmax_w(g.logits, (int) hp.n_vocab, g.amax, T, g.s);
        cudaMemcpyAsync(h_amax, g.amax, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost, g.s);
    } else {
        kn::SamplerParams p = chain_of(sp, samp_hist_cap, pen);
        p.counter = (uint64_t) counter_pos;
        const int* dh = nullptr;
        int hl = 0;
        if (p.penalty_last_n > 0) {
            hl = p.penalty_last_n;
            // row t follows the window's drafts 1..t: the tail (hist without its last token, win[0]) then win[0..t]
            kn::penalty_rows(hist, std::max(0, hist_n - 1), win, T, hl, h_hist);
            cudaMemcpyAsync(samp_hist, h_hist, (size_t) T * hl * sizeof(int32_t), cudaMemcpyHostToDevice, g.s);
            dh = samp_hist;
        }
        kn::sample_tokens(g.logits, T, (int) hp.n_vocab, dh, hl, p, samp_out, g.s);
        cudaMemcpyAsync(h_amax, samp_out, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost, g.s);
    }
    if (cudaStreamSynchronize(g.s) != cudaSuccess) return fail(err, "sampling failed");
    for (int t = 0; t < T; ++t) out[t] = h_amax[t];
    if (guard) {
        ++nf_windows;
        for (int t = 0; t < T && nf_row < 0; ++t)
            for (int b = 0; b < kn::kNonfiniteBlocks; ++b)
                if (nf_h[(size_t) t * kn::kNonfiniteBlocks + b] != 0) {
                    nf_row = t;
                    break;
                }
        if (nf_row < 0 && nan_inject(nf_windows)) nf_row = 0;
    }
    return true;
}

bool Engine::Impl::spec_decode(SpecRun& r, std::string& err) {
    Stage& lg = last_stage();
    const SamplingParams& sp = *r.sp;
    int cap_w = spec_cap > 0 ? std::min(spec_cap, snap_T) : snap_T;
    if (sp.spec_window > 0) cap_w = std::min(cap_w, sp.spec_window);   // a request's own cap (1: no drafting)
    const bool use_mtp = mtp_on && cap_w >= 2 && !r.plain;
    const int spec = use_mtp ? cap_w : 1;                                  // the longest window
    const float min_p = sp.spec_min_p >= 0.0f ? sp.spec_min_p : opt.spec_min_p;
    // moex: DeepSeek-V4.1's DSpark verifies a draft only while the drafter's own confidence head holds (model.py
    // DSparkConfidenceHead; its logit through a sigmoid, as upstream's and Maya's --dspark-conf): each draft's at least
    // STRATA_DSV41_DSPARK_CONF, and the product of the confidences so far at least STRATA_DSV41_DSPARK_CUM (the cumulative
    // cut).  Unset: 0, off.  The trunk verifies every draft it is given, so the tokens are the same - only how many rows a
    // window verifies changes.  Beside the draft-probability floor (--spec-min-p), which still applies
    static const float dsp_conf_min = [] { const char* e = std::getenv("STRATA_DSV41_DSPARK_CONF"); return e && *e ? (float) std::atof(e) : 0.0f; }();
    static const float dsp_cum_min = [] { const char* e = std::getenv("STRATA_DSV41_DSPARK_CUM"); return e && *e ? (float) std::atof(e) : 0.0f; }();
    const float conf_min = sp.dspark_conf >= 0.0f ? sp.dspark_conf : dsp_conf_min;   // (a request's dspark_conf / dspark_cum keys)
    const float cum_min = sp.dspark_cum >= 0.0f ? sp.dspark_cum : dsp_cum_min;
    const bool dsp_cut = dsp && dsp->v41 && (conf_min > 0.0f || cum_min > 0.0f);
    // (STRATA_DSV41_DSPARK_ROWS=-1 drafts what the window can verify: this request's cap, not only the engine's)
    req_cap = use_mtp ? cap_w : 0;
    struct ReqCapOff {
        int& c;
        ~ReqCapOff() { c = 0; }
    } req_cap_off{req_cap};
    // coupled draft sampling for this request (sampled, no penalties, the MTP drafter's own heads): the chain on the device
    // once, before any draft of the request; argmax drafts again after it
    static const bool cd_env = [] { const char* e = std::getenv("STRATA_SPEC_COUPLED"); return e && *e && std::strcmp(e, "0") != 0; }();
    static const bool cd_env_off = [] { const char* e = std::getenv("STRATA_SPEC_COUPLED"); return e && std::strcmp(e, "0") == 0; }();
    lg.cd_on = false;
    // moex: DFly2's drafts are coupled by default (they never change the output; STRATA_SPEC_COUPLED=0 keeps them argmax)
    const bool cd_want = sp.spec_coupled >= 0 ? sp.spec_coupled != 0 : (opt.coupled_draft || cd_env || (dfy && !cd_env_off));
    if (use_mtp && !dsp && lg.cd_params && cd_want && sp.temperature > 0.0f && !penalized(sp)) {
        const kn::SamplerParams cp = chain_of(sp, samp_hist_cap, false);
        set_dev(lg.dev);
        lg.cd_on = cudaMemcpyAsync(lg.cd_params, &cp, sizeof cp, cudaMemcpyHostToDevice, lg.s) == cudaSuccess &&
                   cudaStreamSynchronize(lg.s) == cudaSuccess;
    }
    struct CdOff {
        Stage& g;
        ~CdOff() { g.cd_on = false; }
    } cd_off{lg};
    // Strata's prompt lookup (EngineOptions::suffix_draft): the request's context into the suffix drafter, then every token
    // emitted; per window the policy may take the lookup's continuation instead of the MTP's drafts (generate.cpp's serial
    // loop: only where the MTP's own first draft agrees, never the request's first window)
    const bool lk_on = lookup_S >= 2 && sfx && policy && !r.plain && !dsp;
    // DFlash's window: the cost-aware choice (EngineOptions::spec_policy, a request's spec_policy) or the spec-min-p gate
    const bool cost_gate = (dfl || dfy) && use_mtp && (sp.spec_policy >= 0 ? sp.spec_policy : opt.spec_policy) == 1;
    int32_t lk_buf[kTmax] = {0};
    if (lk_on) {
        sfx->reset();
        if (r.ctx && r.ctx_n > 0) sfx->append(r.ctx, (size_t) r.ctx_n);
        else sfx->append(r.x);
    }
    int64_t pos = r.pos;
    int32_t x = r.x;
    bool have_draft = false;                                               // d_tok[0] is the draft for the token after x
    int32_t d_tok[kTmax] = {0}, win[kTmax] = {0}, y[kTmax] = {0};
    float d_prob[kTmax] = {0};
    while (r.produced < r.max_new) {
        if (r.stop && r.stop()) {
            r.cancelled = true;
            break;
        }
        // 1. the window: x, then the drafts that pass the gate (the chain steps run only as far as the gate lets them)
        const auto t_round = clk::now();                                   // the policy's round: verify + commit + draft
        const int room = r.max_new - (int) r.produced;                     // a window of T rows emits at most T tokens
        int nd = 0;
        if (use_mtp && have_draft) {
            const int cap = std::max(0, std::min({spec - 1, room - 1, hp.arch == Arch::glm5 || dmtp ? kTmax : lg.mtp_heads()}));
            const auto td = clk::now();
            if (cost_gate) {   // DFlash: the whole block is in; its expected tokens per ms decide (a request's spec_min_p: a floor)
                int avail = 0;
                while (avail < cap) {
                    if (avail > 0 && !lg.mtp_chain_step(avail, d_tok, pos, d_tok[avail], d_prob[avail], err)) return false;
                    if (sp.spec_min_p >= 0.0f && d_prob[avail] < sp.spec_min_p) break;
                    ++avail;
                }
                nd = dgate.choose(d_prob, avail);
            } else {
                float cum = 1.0f;
                while (nd < cap) {
                    if (d_prob[nd] < min_p) break;
                    if (dsp_cut && nd < lg.dsp_n) {   // (the block's confidences came with its drafts: dspark_draft)
                        const float c = 1.0f / (1.0f + std::exp(-lg.dsp_conf[nd]));
                        if (c < conf_min || (cum *= c) < cum_min) break;
                    }
                    ++nd;
                    if (nd < cap && !lg.mtp_chain_step(nd, d_tok, pos, d_tok[nd], d_prob[nd], err)) return false;
                }
            }
            r.draft_ms += ms_since(td);
        }
        int T = nd + 1;
        // prompt lookup: a repeat of earlier context whose continuation starts with the MTP's first draft (any, without MTP)
        bool from_sfx = false;
        int sfx_match = 0;
        if (lk_on && r.windows > 0 && room >= 2) {
            const int k = sfx->propose(std::min(lookup_S, room) - 1, lk_buf);
            sfx_match = sfx->last_match();
            if (k > 0 && (!use_mtp || !have_draft || lk_buf[0] == d_tok[0])) {
                const strata::spec::DraftPolicy::Pick pk = policy->choose(T, k, sfx_match);
                if (pk.lookup && pk.t >= 2) {
                    T = std::min(pk.t, k + 1);
                    from_sfx = true;
                }
            }
        }
        // a measurement (Engine::set_stand_in, --decode-runs fN): the window filled to N rows with the context's last tokens
        // as drafts - rejected as a rule, so each window emits one token and the run's speed is the verify cost of N rows
        if (stand_in >= 2 && !use_mtp && !from_sfx && room >= 2) {
            T = std::min({stand_in, room, kTmax});
            const std::vector<int32_t>& hs = *r.hist;
            for (int j = 0; j < T - 1; ++j) {
                const int64_t at = (int64_t) hs.size() - (T - 1) + j;
                lk_buf[j] = at >= 0 ? hs[(size_t) at] : x;
            }
            from_sfx = true;
        }
        nd = T - 1;
        win[0] = x;
        for (int j = 0; j < nd; ++j) win[1 + j] = from_sfx ? lk_buf[j] : d_tok[j];
        // keep: the KDA snapshots a rollback needs (GLM-5.3 only; MiMo / DeepSeek-V4 rewrite rejected rows in place)
        const auto tv = clk::now();
        if (!run_window(win, T, pos, T > 1 && hp.arch == Arch::glm5, err)) return false;
        ++r.windows;
        r.offered += nd;
        // 2. the trunk's own token after every row
        if (!sample_rows(sp, pos, win, T, r.hist->data(), (int) r.hist->size(), y, err)) return false;
        if (nf_row >= 0) {   // #879: none of this window's tokens; run_request ends the request and forgets every state
            fprintf(stderr, "strata-x: non-finite logits at position %lld (window row %d of %d, #879): the window's tokens are "
                            "dropped and the cached state too; the next request reads its prompt from token 0\n",
                    (long long) (pos + nf_row), nf_row, T);
            r.nonfinite = true;
            break;
        }
        int a = 0;
        while (a < T - 1 && y[a] == win[a + 1]) ++a;
        r.accepted += a;
        if ((dfl || dfy) && use_mtp) {   // DFlash's (DFly2's) window policy learns this machine's verify costs and acceptance
            const double vms = ms_since(tv), sms = have_draft ? (dfl ? lg.df_last_ms : lg.dfy_last_ms) : 0.0;
            dgate.observe_cost(T, vms, sms);
            dgate.observe_rate(a + 1, vms + sms);
            if (have_draft && !from_sfx) dgate.observe(d_prob, nd, a);
        }
        if (from_sfx) {
            ++r.lookup_windows;
            r.lookup_accepted += a;
        }
        for (int j = 0; j < nd; ++j) {
            r.offered_at[j]++;
            r.accepted_at[j] += j < a ? 1 : 0;
        }
        // 3. emit win[1..a] and y[a]; a stop (an EOS id, the limit, a cancel) keeps only the rows before the stopping token
        int kept = a + 1;
        bool stopped = false;
        int32_t emitted[kTmax];
        for (int i = 0; i <= a; ++i) {
            const int32_t e = i < a ? win[i + 1] : y[a];
            emitted[i] = e;
            if (lk_on) sfx->append(e);
            r.hist->push_back(e);
            if ((int) r.hist->size() > samp_hist_cap)
                r.hist->erase(r.hist->begin(), r.hist->begin() + (std::ptrdiff_t) (r.hist->size() - samp_hist_cap));
            ++r.produced;
            const bool is_eos = std::find(r.eos->begin(), r.eos->end(), e) != r.eos->end();
            const bool more = r.token ? r.token(e) : true;
            if (is_eos || !more || r.produced >= r.max_new) {
                r.eos_hit = is_eos;
                r.cancelled = !is_eos && !more;
                kept = i + 1;
                stopped = true;
                break;
            }
        }
        // 4. commit: the recurrent state after the last kept row, the live sequence, the drafter over the kept rows
        if (kept < T) rollback_to(kept);
        for (int j = 0; j < kept; ++j) live.push_back(win[j]);
        if (use_mtp) {
            const auto td = clk::now();
            int32_t nt[kTmax];                                             // the token after each kept row = what was emitted
            for (int j = 0; j < kept; ++j) nt[j] = emitted[j];
            if (!lg.mtp_catchup(kept, nt, lg.mtp_hsrc(), pos, d_tok[0], d_prob[0], err)) return false;
            have_draft = true;
            r.draft_ms += ms_since(td);
        }
        if (lk_on && !stopped && r.windows > 1) policy->observe(from_sfx, T, a, sfx_match, ms_since(t_round));
        pos += kept;
        if (stopped) break;
        x = y[a];
    }
    static const bool gate_print = std::getenv("STRATA_X_DFLASH_GATE") != nullptr;
    if (gate_print && cost_gate) dgate.print();
    return true;
}

/// Captures every decode graph the loop will use - the trunk's windows of 1 .. spec rows and the drafter's - before the first
/// request: a variant's first use runs eagerly and its second is captured (decode.cpp), which would otherwise cost the
/// first request a few hundred ms.  The state the windows leave (the recurrent states, the tier's counts) is cleared after.
void Engine::Impl::warm_decode() {
    if (!graphs_on) return;
    std::vector<EngineStats> saved;
    for (auto& gp : stages) saved.push_back(gp->st);
    std::string err;
    int32_t tok[kTmax];
    for (int i = 0; i < kTmax; ++i) tok[i] = 1;
    const auto t0 = clk::now();
    auto go = [&](int T, bool keep) {
        for (int rep = 0; rep < 2; ++rep)
            if (!forward(tok, T, 0, keep, err)) {
                fprintf(stderr, "strata-x: warm-up window of %d failed: %s\n", T, err.c_str());
                err.clear();
                return;
            }
    };
    go(1, false);
    // the verify windows the loop runs: with snapshots on GLM-5.3 (run_window's keep), without on MiMo and DeepSeek-V4
    const bool keep_w = hp.arch == Arch::glm5;
    if (mtp_on && (dsp || dfl || dfy)) {   // DSpark, DFlash, DFly2: the trunk's verify windows (the drafter runs eagerly)
        for (int T = 2; T <= snap_T; ++T) go(T, keep_w);
    } else if (mtp_on) {
        for (int T = 2; T <= snap_T; ++T) go(T, keep_w);
        Stage& lg = last_stage();
        int32_t tk = 0;
        float pr = 0;
        // every drafter variant the loop uses: each block's catch-up over 1 .. snap_T rows (block 0's with its draft), and
        // MiMo's chain steps (block j over j rows, with a draft)
        const int nh = lg.mtp_heads();
        for (int hd = 0; hd < nh; ++hd)
            for (int Tm = 1; Tm <= snap_T; ++Tm)
                for (int dr = 0; dr < 2; ++dr) {
                    const bool draft = dr == 1;
                    if (hd > 0 && draft && !(hp.arch == Arch::mimo2 && Tm == hd)) continue;   // a variant the loop never runs
                    if (hd == 0 && !draft) continue;
                    for (int rep = 0; rep < 2; ++rep)
                        if (!lg.mtp_run(hd, Tm, tok, hd == 0 ? lg.mtp_hsrc() : lg.mtp_chain, 0, draft, tk, pr, err)) {
                            fprintf(stderr, "strata-x: warm-up drafter step (block %d, %d rows) failed: %s\n", hd, Tm, err.c_str());
                            err.clear();
                        }
                }
    }
    for (auto& gp : stages) {
        set_dev(gp->dev);
        cudaStreamSynchronize(gp->s);
    }
    reset_state();
    std::fill(heat.begin(), heat.end(), 0.0f);
    std::fill(routed.begin(), routed.end(), 0);   // the warm-up's windows are not this run's routing (the saved profile)
    token_misses.clear();
    for (size_t i = 0; i < stages.size(); ++i) stages[i]->st = saved[i];
    windows_seen = 0;
    last_T = 0;
    if (opt.verbose) fprintf(stderr, "strata-x: decode graphs captured in %.1f s\n", ms_since(t0) / 1000.0);
}

}  // namespace strata::x
