// src/x/tiers.cpp - THE EXPERTS' TIERS: the VRAM slots, the pinned RAM copy, the GGUF in place.
//
// Strata's resident RAM mode (src/core/expert_source.cpp pin_cache_complement, #403 / #467 / #577), for GLM:
//   * the VRAM tier is planned first - the drafter's experts pinned in the last GPU's first slots, then the expert profile,
//     hottest first, into each GPU's slots;
//   * the RAM copy (Strata's pinned arena) holds the CACHE COMPLEMENT - every expert the VRAM tier does not - and then, as
//     far as the budget allows, the prompt path's LEND REGION (the experts in the slots a prompt borrows, from the top slot
//     down, so a prompt need not read them from the file and their refill afterwards is a pinned copy);
//   * the budget is the available RAM minus a headroom (EngineOptions::resident_headroom_gib); when the complement itself
//     does not fit, the hottest of it by the profile is kept (Strata's "what fits") and the rest is read from the GGUF;
//   * one O_DIRECT pass over the files at load routes every expert to its slot (through pinned bounce buffers) and/or its
//     RAM blob - nothing goes through the page cache;
//   * the adaptive tier EXCHANGES: an expert entering VRAM hands its RAM blob to the one leaving (the leaving one is copied
//     out to a free exchange blob first, while it is still resident; Strata's reserve_exchanges), so the RAM copy keeps its
//     size.  An expert with a RAM copy (the lend region's) simply leaves.
#include "engine_impl.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>

namespace strata::x {

namespace {
uint64_t meminfo_kb(const char* key) {
    uint64_t v = 0;
    if (FILE* f = fopen("/proc/meminfo", "r")) {
        char line[256];
        const size_t kl = strlen(key);
        while (fgets(line, sizeof line, f))
            if (strncmp(line, key, kl) == 0 && line[kl] == ':') {
                v = strtoull(line + kl + 1, nullptr, 10);
                break;
            }
        fclose(f);
    }
    return v;
}
}  // namespace

/// The routed experts' formats and their place in the GGUF mapping - the file tier every expert has.  The RAM copy and the
/// VRAM slots are built by init_cache, once the slots are sized.
bool Engine::Impl::load_experts(std::string& err) {
    // DeepSeek-V4.1: the pack's record files, or (moex, --experts-gguf) a GGUF set's virtual records (dsv41.cpp)
    if (hp.v41) return opt.experts_gguf.empty() ? load_experts_records(err) : load_experts_gguf(err);
    for (int64_t il = hp.n_dense; il < n_xl; ++il) {
        Layer& l = L[(size_t) il];
        auto n = [&](const char* sfx) { return "blk." + std::to_string(il) + "." + sfx; };
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* g = gm->find(n("ffn_gate_exps.weight"), &sg);
        const strata::TensorInfo* u = gm->find(n("ffn_up_exps.weight"), &su);
        const strata::TensorInfo* d = gm->find(n("ffn_down_exps.weight"), &sd);
        if (!g || !u || !d) return fail(err, n("ffn_*_exps.weight missing"));
        if (g->type != u->type) return fail(err, n("gate and up experts of different types"));
        // (Hy4-flash: the experts' width is the latent's, hp.ex(); n_embd everywhere else)
        if ((int64_t) g->shape[0] != hp.ex() || (int64_t) g->shape[1] != hp.n_ff_exp || (int64_t) g->shape[2] != hp.n_expert ||
            (int64_t) d->shape[0] != hp.n_ff_exp || (int64_t) d->shape[1] != hp.ex())
            return fail(err, n("expert shapes"));
        if (!kc::native_fmt((int) g->type, (int) d->type, hp.ex(), hp.n_ff_exp, l.fmt, err)) return false;
        l.fmt.swiglu_limit = hp.clamp_exp[(size_t) il];
        l.eg = gm->shard(sg).tensor_data(*g);
        l.eu = gm->shard(su).tensor_data(*u);
        l.ed = gm->shard(sd).tensor_data(*d);
        l.sg = l.su = l.fmt.gu_row * (size_t) hp.n_ff_exp;
        l.sd = l.fmt.d_row * (size_t) hp.ex();
        l.blob_contiguous = false;
        l.f_shard[0] = sg;
        l.f_shard[1] = su;
        l.f_shard[2] = sd;
        l.f_off[0] = gm->shard(sg).data_start() + g->offset;
        l.f_off[1] = gm->shard(su).data_start() + u->offset;
        l.f_off[2] = gm->shard(sd).data_start() + d->offset;
        blob_max = std::max(blob_max, l.fmt.bytes);
        if (mmq::built() && mmq::supported(l.fmt.gu_type) && mmq::supported(l.fmt.d_type)) {
            gub_max = std::max(gub_max, mmq::matrix_bytes(l.fmt.gu_type, 2 * hp.n_ff_exp, hp.ex()));
            db_max = std::max(db_max, mmq::matrix_bytes(l.fmt.d_type, hp.ex(), hp.n_ff_exp));
        } else {
            mmq_all = false;   // a layer without MMQ tiles: no prompt path (its MoE would have no product)
        }
    }
    // moex (2026-11-19): STRATA_POOL_TASKS=N for these experts too (DeepSeek-V4.1's record loader reads it, dsv41.cpp
    // load_experts_tail): each row phase in N tasks, not 3 a thread - every row stays one thread's dot product, no result changes
    const char* pt = std::getenv("STRATA_POOL_TASKS");
    const int pool_tasks = pt ? std::clamp(std::atoi(pt), 0, kc::ExpertPool::kMaxTasks) : 0;
    pool = std::make_unique<kc::ExpertPool>(opt.threads, opt.pin, true, kc::PoolAffinity::All, pool_tasks);
    if (opt.verbose)
        fprintf(stderr, "strata-x: CPU expert pool, %d workers + the host%s\n", pool->workers(),
                pool_tasks > 0 ? (", " + std::to_string(pool_tasks) + " tasks a phase (STRATA_POOL_TASKS)").c_str() : "");
    act_q.resize((size_t) kTmax * kc::kNativeActBytes);
    return true;
}

/// The run's expert profile (Strata's expert_profile_save, #477, in rank_learned_profile's order): every expert of the trunk's
/// MoE layers - the ones in VRAM now first (a swap in flight counts as landed: where this run's tier ended), then the rest;
/// within each by this run's decode routing (descending), then by the profile the run started from (an expert this run
/// never routed keeps its old place), then by index - so a start from it begins where this one ended, as Strata's does.
/// "<layer> <expert> <count>" a line, in that order, which --expert-profile reads to plan the VRAM tier and the RAM copy.
/// The count (choose_split's routed mass, and the next run's prior) is this run's decode routings plus half the count the
/// loaded profile gave: a short run moves it little, an old one fades out.
bool Engine::Impl::save_profile(const std::string& path, std::string& err) const {
    const int64_t NE = hp.n_expert, k0 = hp.n_dense * NE, k1 = std::min<int64_t>(hp.n_layer * NE, (int64_t) routed.size());
    if (k1 <= k0) {
        err = "no routed layers to save";
        return false;
    }
    std::vector<uint8_t> res((size_t) (k1 - k0), 0);
    for (int64_t key = k0; key < k1; ++key) res[(size_t) (key - k0)] = key < (int64_t) resident.size() && resident[(size_t) key] >= 0;
    for (const PendingSwap& ps : pending) {
        if (ps.key >= k0 && ps.key < k1) res[(size_t) (ps.key - k0)] = 1;
        if (ps.key_out >= k0 && ps.key_out < k1) res[(size_t) (ps.key_out - k0)] = 0;
    }
    auto prk = [&](int64_t key) -> int64_t {
        return key < (int64_t) prior_rank.size() && prior_rank[(size_t) key] >= 0 ? (int64_t) prior_rank[(size_t) key] : INT64_MAX;
    };
    std::vector<int64_t> keys((size_t) (k1 - k0));
    for (int64_t key = k0; key < k1; ++key) keys[(size_t) (key - k0)] = key;
    std::stable_sort(keys.begin(), keys.end(), [&](int64_t a, int64_t b) {
        const bool ra = res[(size_t) (a - k0)] != 0, rb = res[(size_t) (b - k0)] != 0;
        if (ra != rb) return ra;
        if (routed[(size_t) a] != routed[(size_t) b]) return routed[(size_t) a] > routed[(size_t) b];
        const int64_t pa = prk(a), pb = prk(b);
        if (pa != pb) return pa < pb;
        return a < b;
    });
    FILE* f = fopen((path + ".tmp").c_str(), "w");
    if (!f) {
        err = "cannot write " + path;
        return false;
    }
    uint64_t total = 0, in_vram = 0;
    for (int64_t k = k0; k < k1; ++k) {
        total += routed[(size_t) k];
        in_vram += res[(size_t) (k - k0)];
    }
    // moex (DSV41_BACKLOG #57): a pack's profile names it - FNV-1a 64 of its experts.tsv - so the server starts from it only
    // on the same pack (every V4.1 pack has the same shape).  The builds before 2026-10-20's fix took 1469598103934665603 for
    // the offset basis (its last digit lost); the server accepts their digest too
    std::string pack_id;
    if (!opt.pack.empty())
        if (FILE* tf = fopen((opt.pack + "/experts.tsv").c_str(), "rb")) {
            uint64_t h = 14695981039346656037ull;
            for (int c; (c = fgetc(tf)) != EOF;) h = (h ^ (uint64_t) (unsigned char) c) * 1099511628211ull;
            fclose(tf);
            char hex[40];
            snprintf(hex, sizeof hex, "; pack fnv64:%016llx", (unsigned long long) h);
            pack_id = hex;
        }
    fprintf(f, "# strata-x expert profile: n_layer %lld n_expert %lld, %zu experts, %llu decode routings this run, %llu in VRAM "
               "(layer expert count; the VRAM tier first, then the hottest - Strata's #477 order)%s\n",
            (long long) hp.n_layer, (long long) hp.n_expert, keys.size(), (unsigned long long) total, (unsigned long long) in_vram,
            pack_id.c_str());
    for (int64_t k : keys) {
        const double prior = k < (int64_t) prior_count.size() ? prior_count[(size_t) k] : 0.0;
        fprintf(f, "%lld %lld %.1f\n", (long long) (k / NE), (long long) (k % NE), (double) routed[(size_t) k] + 0.5 * prior);
    }
    if (fclose(f) != 0 || rename((path + ".tmp").c_str(), path.c_str()) != 0) {
        err = "cannot write " + path;
        return false;
    }
    return true;
}

Engine::Impl::Src Engine::Impl::src(int64_t key) const {
    const int64_t NE = hp.n_expert, il = key / NE, e = key % NE;
    const Layer& l = L[(size_t) il];
    if (!ram_off.empty() && ram_off[(size_t) key] >= 0) {
        const uint8_t* p = arena->data() + ram_off[(size_t) key];
        return {p, p + l.fmt.up_off, p + l.fmt.down_off, arena_pinned};
    }
    if (l.rec_stride && !l.rec_gguf) {   // a pack's record: [gate | up | down] at e * stride (a GGUF set's: the slices below)
        const uint8_t* p = l.eg + (size_t) e * l.rec_stride;
        return {p, p + l.fmt.up_off, p + l.fmt.down_off, false};
    }
    return {l.eg + (size_t) e * l.sg, l.eu + (size_t) e * l.su, l.ed + (size_t) e * l.sd, false};
}

uint8_t* Engine::Impl::slot_ptr(int32_t slot) const {
    if (slot >= n_slots) return nx_base + (size_t) (slot - n_slots) * nx_bytes;   // the drafter's region
    const Stage& g = *stages[(size_t) slot_dev[(size_t) slot]];
    if (!g.slot_off.empty()) return g.slots + g.slot_off[(size_t) (slot - g.slot_base)];   // moex (§24): sized slots
    return g.slots + (size_t) (slot - g.slot_base) * slot_bytes;
}

size_t Engine::Impl::slot_cap(int32_t slot) const {
    if (slot >= n_slots) return nx_bytes;
    const Stage& g = *stages[(size_t) slot_dev[(size_t) slot]];
    if (g.slot_off.empty()) return slot_bytes;
    const size_t k = (size_t) (slot - g.slot_base);
    return g.slot_off[k + 1] - g.slot_off[k];
}

bool Engine::Impl::init_cache(std::string& err) {
    const int64_t NE = hp.n_expert, NK = n_load * NE, NKX = n_xl * NE;   // NKX: the keys of the layers with experts
    resident.assign((size_t) NK, -1);
    heat.assign((size_t) NK, 0.0f);
    routed.assign((size_t) NK, 0);
    prior_rank.assign((size_t) NK, -1);
    prior_count.assign((size_t) NK, 0.0);
    ram_off.assign((size_t) NK, -1);
    ram_cap.assign((size_t) NK, 0);
    khash.assign((size_t) NK * 3, 0);
    khash_ok.assign((size_t) NK, 0);
    xpool.clear();
    // THE SLOT SIZE: uniform slots (Strata's), sized to hold the most experts per byte of VRAM.  Strata's packs have one blob
    // size; a GGUF mixes them (unsloth's UD-IQ2_XXS: 7.2 MiB in 39 layers, 8.4-9.4 MiB in three): slots of the largest blob
    // would waste ~23% of the tier, so the size maximising (experts it can hold) / (bytes per slot) wins and a layer whose
    // blob is larger stays on the CPU (the checks below).  One blob size: that size, as before.
    {
        std::map<size_t, int64_t> by_size;   // blob size -> trunk experts of that size (GPU-supported formats)
        int64_t all = 0;
        for (int64_t il = hp.n_dense; il < hp.n_layer; ++il) {
            const kc::NativeFmt& f = L[(size_t) il].fmt;
            if (!kn::native_mmvq_supported(f.gu_type) || !kn::native_mmvq_supported(f.d_type)) continue;   // CPU only
            by_size[(f.bytes + 255) & ~(size_t) 255] += NE;
            all += NE;
        }
        double best = -1.0;
        int64_t covered = 0;
        for (const auto& [sz, cnt] : by_size) {   // ascending sizes: a slot of `sz` holds every expert up to it
            covered += cnt;
            const double score = (double) covered / (double) sz;
            if (score > best) {
                best = score;
                slot_bytes = sz;
            }
        }
        if (opt.verbose && by_size.size() > 1)
            fprintf(stderr, "strata-x: expert blobs of %zu sizes: slots of %.2f MiB hold %.0f%% of the trunk's experts; the larger layers' "
                            "experts stay on the CPU\n",
                    by_size.size(), slot_bytes / 1048576.0, 100.0 * [&] {
                        int64_t c = 0;
                        for (const auto& [sz, cnt] : by_size) c += sz <= slot_bytes ? cnt : 0;
                        return (double) c / (double) std::max<int64_t>(1, all);
                    }());
    }
    if (slot_bytes > 0 && cudaMallocHost(&staging, std::max({slot_bytes, (size_t) blob_max, (size_t) (rio ? rio->slot : 0)})) != cudaSuccess)
        return fail(err, "cannot allocate the slot staging buffer");
    // THE DRAFTER'S REGION (Strata: "all 512 experts resident", allocated before the cache): the NextN layer's experts, all
    // of them, on the last GPU in an allocation of their own (their blob may not fit a trunk slot)
    nx_count = 0;
    if (mtp_on && hp.arch == Arch::mimo2) last_stage().mtp_resident = true;   // its MTP blocks have dense FFNs: nothing to place
    if (mtp_on && hp.arch == Arch::glm5 && opt.expert_slots != 0) {
        Stage& lg = last_stage();
        const kc::NativeFmt& f = L[(size_t) hp.n_layer].fmt;
        const bool grouped = kn::native_mmvq_grouped_supported(f.gu_type, (int) hp.n_embd) &&
                             kn::native_mmvq_grouped_supported(f.d_type, (int) hp.n_ff_exp);
        cudaSetDevice(lg.dev);
        nx_bytes = (f.bytes + 255) & ~(size_t) 255;
        if (grouped && cudaMalloc(&nx_base, (size_t) NE * nx_bytes) == cudaSuccess) {
            nx_count = NE;
            lg.mtp_resident = true;
        } else {
            cudaGetLastError();
            nx_base = nullptr;
            if (opt.verbose)
                fprintf(stderr, "strata-x: the drafter's experts are not resident in VRAM (%s): draft steps run eagerly on the CPU pool\n",
                        !grouped ? "no grouped kernels for their type" : "no room");
        }
        cur_dev = -1;
    }

    // ---- 1. the VRAM slots, allocated LAST (Strata): what each GPU has left once its reserve stays free
    int64_t total = 0;
    std::vector<int32_t> plan_slot((size_t) NK, -1);   // [key]: its planned slot (2. below; 1' places them itself)
    // moex (DSV41_PLAN §24, for checks): STRATA_X_STAGE_SLOTS=N0,N1,... - at most Ni slots on stage i, so two layouts can
    // hold the same experts (an absent or empty entry: no cap)
    auto stage_cap = [](size_t di) -> int64_t {
        const char* e = std::getenv("STRATA_X_STAGE_SLOTS");
        for (size_t i = 0; e && *e; ++i) {
            if (i == di) return *e != ',' ? std::atoll(e) : INT64_MAX;
            e = std::strchr(e, ',');
            if (e) ++e;
        }
        return INT64_MAX;
    };
    // moex (DSV41_PLAN §24), opt-in STRATA_DSV41_SLOT_SIZES=1: 1' - each slot the size of its expert's blob.  The plan's order
    // (the profile's lines, then expert e of every layer for e = 0, 1, ...) placed GPU by GPU while the next expert fits what
    // the GPU has left (its reserve kept) and --expert-slots: a uniform slot of the largest blob left GPU 0, whose layers
    // hold none that large, 8-24% of each slot empty (q3q2u: 11.1-13.4 MiB experts in 14.5 MiB slots)
    if (const char* e = std::getenv("STRATA_DSV41_SLOT_SIZES"); e && e[0] == '1' && hp.v41 && nx_count == 0 && opt.expert_slots != 0 &&
                                                                 slot_bytes > 0)
        slot_sized = true;
    if (slot_sized) {
        std::vector<int64_t> order;
        std::vector<char> seen((size_t) NK, 0);
        if (!opt.expert_profile.empty())
            if (FILE* fp = fopen(opt.expert_profile.c_str(), "r")) {
                char line[256];
                while (fgets(line, sizeof line, fp)) {
                    long long il, e2;
                    double c;
                    if (line[0] == '#' || sscanf(line, "%lld %lld %lf", &il, &e2, &c) != 3) continue;
                    if (il < hp.n_dense || il >= n_xl || e2 < 0 || e2 >= NE) continue;
                    const int64_t key = il * NE + e2;
                    if (!seen[(size_t) key]) {
                        seen[(size_t) key] = 1;
                        order.push_back(key);
                    }
                }
                fclose(fp);
            }
        for (int64_t e2 = 0; e2 < NE; ++e2)
            for (int64_t il = hp.n_dense; il < n_xl; ++il)
                if (!seen[(size_t) (il * NE + e2)]) order.push_back(il * NE + e2);
        for (auto& gp : stages) {
            Stage& g = *gp;
            cudaSetDevice(g.dev);
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const size_t reserve = (size_t) (g.di == 0 ? opt.vram_reserve_mib : opt.vram_reserve_later_mib) << 20;
            const size_t room = free_b > reserve ? free_b - reserve : 0;
            const int64_t cap = std::min(opt.expert_slots > 0 ? std::max<int64_t>(0, opt.expert_slots - total) : NK, stage_cap((size_t) g.di));
            g.slot_base = total;
            g.slot_off.assign(1, 0);
            for (int64_t key : order) {
                const int64_t il = key / NE;
                if (ldev[(size_t) il] != g.di) continue;
                const kc::NativeFmt& f = L[(size_t) il].fmt;
                if (!kn::native_mmvq_supported(f.gu_type) || !kn::native_mmvq_supported(f.d_type) || f.bytes > slot_bytes) continue;
                const size_t sz = (f.bytes + kSlotUnit - 1) / kSlotUnit * kSlotUnit;
                if (g.slot_off.back() + sz > room || (int64_t) g.slot_off.size() - 1 >= cap) break;   // the order's next does not fit
                plan_slot[(size_t) key] = (int32_t) (g.slot_base + (int64_t) g.slot_off.size() - 1);
                g.slot_off.push_back(g.slot_off.back() + sz);
            }
            g.n_slots = (int64_t) g.slot_off.size() - 1;
            if (g.n_slots == 0) {
                g.slot_off.clear();
            } else if (cudaMalloc(&g.slots, g.slot_off.back()) != cudaSuccess) {
                return fail(err, "cannot allocate the expert slots on GPU " + std::to_string(g.dev));
            } else {
                cudaMemset(g.slots, 0, g.slot_off.back());   // a slot read before it is filled: zeros
            }
            total += g.n_slots;
            if (opt.verbose)
                fprintf(stderr, "strata-x: GPU %d: %lld expert slots of their experts' sizes (%.2f GiB, %.2f MiB on average; "
                                "STRATA_DSV41_SLOT_SIZES), %zu MiB left free\n", g.dev, (long long) g.n_slots,
                        (g.n_slots ? g.slot_off.back() : 0) / 1073741824.0,
                        g.n_slots ? (double) g.slot_off.back() / (double) g.n_slots / 1048576.0 : 0.0,
                        (size_t) ((free_b - (g.n_slots ? g.slot_off.back() : 0)) >> 20));
        }
    }
    if (opt.expert_slots != 0 && slot_bytes > 0 && !slot_sized)
        for (auto& gp : stages) {
            Stage& g = *gp;
            cudaSetDevice(g.dev);
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const int64_t reserve = (g.di == 0 ? opt.vram_reserve_mib : opt.vram_reserve_later_mib) << 20;
            int64_t fit = free_b > (size_t) reserve ? (int64_t) ((free_b - (size_t) reserve) / slot_bytes) : 0;
            int64_t mine = 0;
            for (int64_t il = hp.n_dense; il < hp.n_layer; ++il)
                if (ldev[(size_t) il] == g.di && L[(size_t) il].fmt.bytes <= slot_bytes) mine += NE;
            int64_t want = std::min(fit, mine);
            if (opt.expert_slots > 0) want = std::min(want, std::max<int64_t>(0, opt.expert_slots - total));
            want = std::min(want, stage_cap((size_t) g.di));
            if (want > 0 && kv_grow) {
                // moex (2026-11-19, STRATA_HY4_KV_GROW): the slots on VMM, so the top ones can give their memory to the sparse
                // layers' caches as the context grows (hy4_retire_slots); the floor: what fits beside the whole context's caches
                if (slot_vmm.size() < stages.size()) slot_vmm.resize(stages.size());
                if (slot_floor.size() < stages.size()) slot_floor.assign(stages.size(), 0);
                if (slot_n0.size() < stages.size()) slot_n0.assign(stages.size(), 0);
                slot_n0[g.di] = want;
                if (!vmm_reserve(slot_vmm[g.di], g.dev, (size_t) want * slot_bytes, err) ||
                    !vmm_map_to(slot_vmm[g.di], (size_t) want * slot_bytes, err))
                    return fail(err, "the expert slots on GPU " + std::to_string(g.dev) + ": " + err);
                g.slots = reinterpret_cast<uint8_t*>(slot_vmm[g.di].va);
                uint64_t later = 0;   // this stage's caches still to be mapped, at the whole context
                for (int64_t il = 0; il < n_load && il < (int64_t) kv_vmm.size(); ++il)
                    if (kv_vmm[(size_t) il].va && kv_vmm[(size_t) il].dev == g.dev)
                        later += (kv_vmm[(size_t) il].reserved - kv_vmm[(size_t) il].mapped) +
                                 (pool_vmm[(size_t) il].reserved - pool_vmm[(size_t) il].mapped);
                slot_floor[g.di] = std::max<int64_t>(0, want - (int64_t) ((later + (64ull << 20) + slot_bytes - 1) / slot_bytes));
                if (opt.verbose)
                    fprintf(stderr, "strata-x: GPU %d: %lld of its expert slots give their memory to the caches as the context grows "
                                    "(STRATA_HY4_KV_GROW; %lld stay at the whole context)\n", g.dev, (long long) (want - slot_floor[g.di]),
                            (long long) slot_floor[g.di]);
            } else if (want > 0 && cudaMalloc(&g.slots, (size_t) want * slot_bytes) != cudaSuccess) {
                return fail(err, "cannot allocate the expert slots on GPU " + std::to_string(g.dev));
            }
            if (want > 0) cudaMemset(g.slots, 0, (size_t) want * slot_bytes);   // a slot read before it is filled: zeros
            g.n_slots = want > 0 ? want : 0;
            g.slot_base = total;
            total += g.n_slots;
            if (opt.verbose)
                fprintf(stderr, "strata-x: GPU %d: %lld expert slots (%.2f GiB), %zu MiB left free\n", g.dev, (long long) g.n_slots,
                        (double) g.n_slots * slot_bytes / 1073741824.0, (size_t) ((free_b - (size_t) g.n_slots * slot_bytes) >> 20));
        }
    cur_dev = -1;
    n_slots = total;
    const int64_t n_all = n_slots + nx_count;   // the trunk's slots, then the drafter's region
    owner.assign((size_t) n_all, -1);
    slot_pending.assign((size_t) n_all, 0);
    key_pending.assign((size_t) NK, 0);
    slot_dev.assign((size_t) n_all, (int) stages.size() - 1);
    slot_pinned.assign((size_t) n_all, 0);
    for (auto& gp : stages)
        for (int64_t k = 0; k < gp->n_slots; ++k) slot_dev[(size_t) (gp->slot_base + k)] = gp->di;
    for (int64_t k = 0; k < nx_count; ++k) slot_pinned[(size_t) (n_slots + k)] = 1;

    // ---- 2. the VRAM plan: the drafter's experts pinned in the last GPU's first slots (Strata: "all 512 experts resident"),
    //         then the expert profile, hottest first, each pair on its layer's GPU while that GPU has slots
    std::vector<int64_t> rank;   // every key, the profile's order first (the RAM budget's order)
    std::vector<char> ranked((size_t) NK, 0);
    for (int64_t e = 0; e < nx_count; ++e) plan_slot[(size_t) (hp.n_layer * NE + e)] = (int32_t) (n_slots + e);
    int64_t filled = slot_sized ? n_slots : 0;
    {
        std::vector<int64_t> next(stages.size(), 0);
        if (slot_sized)   // 1' placed them: every slot is planned, this pass only ranks the profile
            for (size_t di = 0; di < stages.size(); ++di) next[di] = stages[di]->n_slots;
        if (!opt.expert_profile.empty()) {
            FILE* fp = fopen(opt.expert_profile.c_str(), "r");
            if (!fp) return fail(err, "cannot read " + opt.expert_profile);
            char line[256];
            while (fgets(line, sizeof line, fp)) {
                long long il, e;
                double c;
                if (line[0] == '#' || sscanf(line, "%lld %lld %lf", &il, &e, &c) != 3) continue;
                if (il < hp.n_dense || il >= n_xl || e < 0 || e >= NE) continue;
                const int64_t key = il * NE + e;
                if (!ranked[(size_t) key]) {
                    ranked[(size_t) key] = 1;
                    prior_rank[(size_t) key] = (int32_t) rank.size();   // the saved profile's tie-break (save_profile)
                    prior_count[(size_t) key] = c;
                    rank.push_back(key);
                }
                if (plan_slot[(size_t) key] >= 0 || L[(size_t) il].fmt.bytes > slot_bytes || n_slots == 0) continue;
                const int di = ldev[(size_t) il];
                Stage& g = *stages[(size_t) di];
                if (next[(size_t) di] >= g.n_slots) continue;
                plan_slot[(size_t) key] = (int32_t) (g.slot_base + next[(size_t) di]++);
                ++filled;   // usage starts at 0: the profile places the experts, this run's own traffic ranks them (Strata)
            }
            fclose(fp);
        }
        for (int64_t key = hp.n_dense * NE; key < NKX; ++key)
            if (!ranked[(size_t) key]) rank.push_back(key);   // the rest in layer order
        // NO PROFILE (or a short one): the slots left are spread over the layers, expert 0 of every layer, then expert 1, ... -
        // nothing is known about the routing, but every slot holds an expert, so the RAM copy (the complement) is no bigger
        // than with a profile and fits as well; the adaptive tier then trades them for this run's hot ones (exchanges)
        bool any = true;
        for (int64_t e = 0; any && n_slots > 0; ++e) {
            any = false;
            for (int64_t il = hp.n_dense; il < n_xl && e < NE; ++il) {
                const int64_t key = il * NE + e;
                const int di = ldev[(size_t) il];
                Stage& g = *stages[(size_t) di];
                if (next[(size_t) di] >= g.n_slots) continue;
                any = true;
                if (plan_slot[(size_t) key] >= 0 || L[(size_t) il].fmt.bytes > slot_bytes) continue;
                plan_slot[(size_t) key] = (int32_t) (g.slot_base + next[(size_t) di]++);
                ++filled;
            }
            if (e >= NE) break;
        }
    }

    // ---- 3. the RAM plan (pin_cache_complement): the complement, then the lend region, within the budget
    const uint64_t xblob = ((uint64_t) blob_max + 4095) & ~(uint64_t) 4095;
    const uint64_t xbytes = opt.experts_in_ram == 0 ? 0 : (uint64_t) std::max(0, opt.exchange_blobs) * xblob;
    uint64_t comp = 0;
    for (int64_t key = hp.n_dense * NE; key < NKX; ++key)
        if (plan_slot[(size_t) key] < 0) comp += L[(size_t) (key / NE)].fmt.bytes;
    const uint64_t avail_now = meminfo_kb("MemAvailable") * 1024ull,   // (moex §73: less the cache's bytes still being faulted in)
                   pend = rio ? rio->pin_pending.load(std::memory_order_acquire) : 0, avail = avail_now > pend ? avail_now - pend : 0;
    const double head_gib = opt.experts_in_ram == 1 ? std::min(opt.resident_headroom_gib, 4.0) : opt.resident_headroom_gib;
    const uint64_t headroom = (uint64_t) (head_gib * 1073741824.0);
    const uint64_t budget = opt.experts_in_ram == 0 ? 0 : avail > headroom ? avail - headroom : 0;
    std::vector<char> in_ram((size_t) NK, 0);
    uint64_t want_ram = 0;
    bool what_fits = false, lend_kept_all = true;
    int64_t lend_kept = 0;
    if (budget > 0 && comp + xbytes <= budget) {
        for (int64_t key = hp.n_dense * NE; key < NKX; ++key)
            if (plan_slot[(size_t) key] < 0) in_ram[(size_t) key] = 1;
        want_ram = comp;
        // the lend region: from each stage's top slot down, alternating the stages, while it fits
        std::vector<int64_t> at(stages.size());
        for (size_t di = 0; di < stages.size(); ++di) at[di] = stages[di]->n_slots - 1;
        std::vector<int32_t> slot_key((size_t) n_slots, -1);   // the trunk's slots (the drafter's region is never lent)
        for (int64_t key = 0; key < NK; ++key)
            if (plan_slot[(size_t) key] >= 0 && plan_slot[(size_t) key] < n_slots) slot_key[(size_t) plan_slot[(size_t) key]] = (int32_t) key;
        for (bool more = true; more;) {
            more = false;
            for (size_t di = 0; di < stages.size(); ++di) {
                Stage& g = *stages[di];
                const int64_t floor_ = g.n_slots - g.lendable();
                if (at[di] < floor_) continue;
                const int64_t slot = g.slot_base + at[di]--;
                more = more || at[di] >= floor_;
                const int32_t key = slot_key[(size_t) slot];
                if (key < 0 || slot_pinned[(size_t) slot]) continue;
                const uint64_t b = L[(size_t) (key / NE)].fmt.bytes;
                if (want_ram + b + xbytes > budget) {
                    lend_kept_all = false;
                    continue;
                }
                in_ram[(size_t) key] = 1;
                want_ram += b;
                ++lend_kept;
            }
        }
    } else if (budget > xbytes) {   // Strata's "what fits": the hottest of the complement by the profile, the rest from the files
        what_fits = true;
        lend_kept_all = false;
        for (int64_t key : rank) {
            if (plan_slot[(size_t) key] >= 0) continue;
            const uint64_t b = L[(size_t) (key / NE)].fmt.bytes;
            if (want_ram + b + xbytes > budget) continue;
            in_ram[(size_t) key] = 1;
            want_ram += b;
        }
    }

    // ---- 4. the arena: layer by layer (per-layer registration bounds), the planned blobs in expert order, then the exchanges
    std::vector<uint64_t> bounds;
    uint64_t off = 0;
    for (int64_t il = hp.n_dense; il < n_xl; ++il) {
        bounds.push_back(off);
        for (int64_t e = 0; e < NE; ++e)
            if (in_ram[(size_t) (il * NE + e)]) {
                ram_off[(size_t) (il * NE + e)] = (int64_t) off;
                ram_cap[(size_t) (il * NE + e)] = L[(size_t) il].fmt.bytes;
                off += L[(size_t) il].fmt.bytes;
            }
    }
    off = (off + 4095) & ~(uint64_t) 4095;
    const uint64_t x0 = off;
    if (want_ram > 0 || xbytes > 0) off += xbytes;
    bounds.push_back(off);
    ram_bytes = off;
    if (ram_bytes > 0) {
        const auto tr = clk::now();
        // PinnedArena's switch (see EngineOptions::arena_thp); moex (DSV41_BACKLOG #102): STRATA_DSV41_HUGE_PIN=1 asks for them,
        // faulted in paced (with FAST_PIN's prefault: strata::core::fault_in_paced stops the request once compaction shows)
        if (!opt.arena_thp && !huge_pin()) setenv("STRATA_NO_ARENA_THP", "1", 1);
        // moex (DSV41_PLAN §21): STRATA_DSV41_FAST_PIN=1 faults the RAM copy in before its registration too (PinnedArena's
        // STRATA_ARENA_PREFAULT), as it does the SSD cache's blocks (dsv41.cpp pin_blocks)
        if (const char* e = std::getenv("STRATA_DSV41_FAST_PIN"); e && e[0] == '1') setenv("STRATA_ARENA_PREFAULT", "1", 1);
        arena = std::make_unique<strata::core::PinnedArena>(ram_bytes, bounds);
        if (!arena->valid()) {
            if (opt.verbose) fprintf(stderr, "strata-x: the RAM copy could not be reserved (%s): every expert from the file\n", arena->note.c_str());
            arena.reset();
            ram_off.assign((size_t) NK, -1);
            ram_bytes = 0;
        } else {
            arena_pinned = arena->registered_bytes >= ram_bytes;
            for (int64_t i = 0; i < opt.exchange_blobs && xbytes > 0; ++i) xpool.push_back({(int64_t) (x0 + (uint64_t) i * xblob), xblob});
            if (opt.verbose)
                fprintf(stderr, "strata-x: RAM copy %.2f GiB reserved in %.1f s (%s): MemAvailable %.1f GiB, headroom %.0f GiB\n",
                        ram_bytes / 1073741824.0, ms_since(tr) / 1000.0, arena->note.c_str(), avail / 1073741824.0, head_gib);
        }
    }

    // moex (DSV41_PLAN §69, §73): the SSD cache's warm fill - the profile's hottest experts that neither the VRAM plan nor the RAM
    // copy holds - its reads started now, beside the pass below (rank's first entries are the profile's lines, hottest first)
    if (rio && hp.v41) ssd_warm(rank, (size_t) std::count(ranked.begin(), ranked.end(), (char) 1), &plan_slot);
    // ---- 5. one pass over the files: 16 experts per read per role, O_DIRECT, the slices to their RAM blobs and (through
    //         pinned bounce buffers) their VRAM slots
    const auto t0 = clk::now();
    std::atomic<int64_t> next_l{hp.n_dense};
    std::atomic<bool> bad{false};
    std::atomic<int> buffered{0};
    std::atomic<uint64_t> read_bytes{0};
    std::atomic<int64_t> t_read{0}, t_copy{0}, t_issue{0}, t_wait{0};   // ns, summed over the readers
    std::string bad_msg;
    size_t max_slice = 0;
    for (int64_t il = hp.n_dense; il < n_xl; ++il)
        max_slice = std::max({max_slice, L[(size_t) il].sg, L[(size_t) il].sd, (size_t) L[(size_t) il].rec_stride});
    const size_t bounce_bytes = 16 * max_slice + 8192;
    const int nth = std::max(1, std::min(8, (int) std::thread::hardware_concurrency() / 4));
    const dsv41::DriveGuard* gd = dsv41::DriveGuard::get();   // moex (DSV41_PLAN §18): DeepSeek-V4.1's drive guard, or null
    std::vector<std::thread> th;
    for (int w = 0; w < nth; ++w)
        th.emplace_back([&, w]() {
            uint8_t* bounce[2] = {nullptr, nullptr};
            cudaStream_t cst[2] = {nullptr, nullptr};          // per stage index
            cudaEvent_t cev[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
            int busy[2] = {-1, -1};                             // the stage whose copies still read bounce[k]
            for (int k = 0; k < 2; ++k)
                if (!(bounce[k] = alloc_pinned(bounce_bytes))) {   // moex (#102): on 2 MiB pages with STRATA_DSV41_HUGE_PIN=1
                    bad = true;
                    bad_msg = "the load's pinned bounce buffers";
                }
            int k = 0;
            // moex (DSV41_PLAN §18): in the drive's slow mode one reader takes the next layer - the others wait here, holding
            // no layer (the one each was reading it finishes at the guard's 128 KB a call), and leave once every layer is taken
            auto my_turn = [&] {
                while (gd && w >= gd->readers() && !bad && next_l.load() < n_xl) std::this_thread::sleep_for(std::chrono::milliseconds(50));
                return true;
            };
            for (int64_t il; !bad && my_turn() && (il = next_l.fetch_add(1)) < n_xl;) {
                const Layer& l = L[(size_t) il];
                const int di = ldev[(size_t) il];
                bool any_vram = false, any = false;
                for (int64_t e = 0; e < NE; ++e) {
                    any_vram |= plan_slot[(size_t) (il * NE + e)] >= 0;
                    any |= plan_slot[(size_t) (il * NE + e)] >= 0 || ram_off[(size_t) (il * NE + e)] >= 0;
                }
                if (!any) continue;
                if (any_vram && !cst[di]) {
                    cudaSetDevice(stages[(size_t) di]->dev);
                    cudaStreamCreateWithFlags(&cst[di], cudaStreamNonBlocking);
                    for (auto& ev : cev[di]) cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
                }
                // a GGUF: three roles (the gate / up / down tensors, an expert's slice of each); a pack's record file
                // (DeepSeek-V4.1): one role, the whole record [gate | up | down] at e * stride, the blob copied once
                const bool rec = l.rec_stride > 0 && !l.rec_gguf;   // (moex: a GGUF set's virtual record: its three slices)
                const size_t slice[3] = {rec ? (size_t) l.rec_stride : l.sg, l.su, l.sd};   // the file's stride per expert
                const size_t clen[3] = {rec ? l.fmt.bytes : l.sg, l.su, l.sd};             // the bytes copied per expert
                const size_t part[3] = {0, l.fmt.up_off, l.fmt.down_off};
                for (int r = 0; r < (rec ? 1 : 3) && !bad; ++r) {
                    const std::string path = rec ? opt.pack + "/" + hp.rec_file[(size_t) il]
                                                 : l.rec_gguf ? xg[l.f_shard[r]]->path() : gm->shard(l.f_shard[r]).path();
                    bool direct = true;
                    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
                    if (fd < 0) {
                        direct = false;
                        buffered++;
                        fd = open(path.c_str(), O_RDONLY);
                    }
                    if (fd < 0) {
                        bad = true;
                        bad_msg = "cannot open " + path;
                        break;
                    }
                    for (int64_t e0 = 0, ne = 0; e0 < NE && !bad; e0 += ne) {
                        auto planned = [&](int64_t e) {
                            return plan_slot[(size_t) (il * NE + e)] >= 0 || ram_off[(size_t) (il * NE + e)] >= 0;
                        };
                        if (rec || l.rec_gguf) {
                            // moex: a GGUF set's virtual records the same way, each role's slices of the planned runs only - the
                            // 16-slice runs of any planned one read 249 GiB at a start to load ~42 (serve_smoke, 2026-11-08)
                            // moex (DSV41_BACKLOG #52): a pack's records are read whole one by one, so only the planned ones are:
                            // a run of up to 16 consecutive ones.  A 16-record run whenever one of them was planned read 195-199
                            // GiB in 80-86 s to keep 41-45 GiB (~77 planned records a layer touch ~97% of the runs)
                            ne = 1;
                            if (!planned(e0)) continue;
                            while (ne < 16 && e0 + ne < NE && planned(e0 + ne)) ++ne;
                        } else {
                            ne = std::min<int64_t>(16, NE - e0);
                            bool need = false;
                            for (int64_t i = 0; i < ne; ++i) need |= planned(e0 + i);
                            if (!need) continue;
                        }
                        auto tq = clk::now();
                        if (busy[k] >= 0) {   // the bounce buffer's previous copies must be done before it is read into again
                            cudaEventSynchronize(cev[busy[k]][k]);
                            busy[k] = -1;
                        }
                        t_wait += (int64_t) (ms_since(tq) * 1e6);
                        tq = clk::now();
                        const size_t len = (size_t) (ne - 1) * slice[r] + clen[r];
                        const uint64_t foff = (rec ? 0 : l.f_off[r]) + (uint64_t) e0 * slice[r];
                        const uint64_t a0 = direct ? foff & ~(uint64_t) 4095 : foff;
                        const size_t want = direct ? (size_t) (((foff + len + 4095) & ~(uint64_t) 4095) - a0) : len;
                        const size_t needb = (size_t) (foff - a0) + len;
                        size_t got = 0;
                        // at most 4 MiB per call: one call of a 16-expert run (~70 MB) queues ~560 requests of the NVMe's 128 KB on
                        // the device, eight readers thousands, and every other process's read waits behind them - the desktop
                        // stalled for a moment on a MiMo load (2026-10-07, the NVMe's scheduler is "none": no I/O priorities); 32
                        // requests per call keep the device's queue deep enough for its full rate
                        constexpr size_t kReadCall = 4u << 20;
                        while (got < needb) {
                            // moex (§18): in the drive's slow mode 128 KB a call (one NVMe request)
                            const size_t call = gd ? (size_t) std::min<uint64_t>(kReadCall, gd->call_bytes()) : kReadCall;
                            const ssize_t rd = pread(fd, bounce[k] + got, std::min(call, want - got), (off_t) (a0 + got));
                            if (rd <= 0) break;   // an O_DIRECT read past the file's end comes back short: fine once `needb` is in
                            got += (size_t) rd;
                        }
                        if (got < needb) {   // a short read is a wrong load, not a slow one (Strata's rule)
                            bad = true;
                            bad_msg = "short read in " + path;
                            break;
                        }
                        read_bytes += got;
                        t_read += (int64_t) (ms_since(tq) * 1e6);
                        const uint8_t* sb = bounce[k] + (foff - a0);
                        for (int64_t i = 0; i < ne; ++i) {   // what the file holds for each expert: the slots are checked against it
                            const uint8_t* xb = sb + (size_t) i * slice[r];
                            if (rec) {   // the record's three parts, as a GGUF's three slices are hashed
                                khash[(size_t) (il * NE + e0 + i) * 3 + 0] = slices_hash_host(xb, l.sg);
                                khash[(size_t) (il * NE + e0 + i) * 3 + 1] = slices_hash_host(xb + l.fmt.up_off, l.su);
                                khash[(size_t) (il * NE + e0 + i) * 3 + 2] = slices_hash_host(xb + l.fmt.down_off, l.sd);
                            } else {
                                khash[(size_t) (il * NE + e0 + i) * 3 + r] = slices_hash_host(xb, slice[r]);
                            }
                            // hashed: the record's three parts at once, a GGUF's with its last slice (a run is read for all
                            // three roles or none, so its first two slices were hashed too)
                            if (rec || r == 2) khash_ok[(size_t) (il * NE + e0 + i)] = 1;
                        }
                        bool issued = false;
                        for (int64_t i = 0; i < ne; ++i) {
                            const int64_t key = il * NE + e0 + i;
                            tq = clk::now();
                            if (ram_off[(size_t) key] >= 0)
                                memcpy(arena->data() + ram_off[(size_t) key] + part[r], sb + (size_t) i * slice[r], clen[r]);
                            t_copy += (int64_t) (ms_since(tq) * 1e6);
                            const int32_t slot = plan_slot[(size_t) key];
                            if (slot >= 0) {
                                tq = clk::now();
                                if (!issued) cudaSetDevice(stages[(size_t) di]->dev);
                                cudaMemcpyAsync(slot_ptr(slot) + part[r], sb + (size_t) i * slice[r], clen[r], cudaMemcpyHostToDevice, cst[di]);
                                issued = true;
                                t_issue += (int64_t) (ms_since(tq) * 1e6);
                            }
                        }
                        if (issued) {
                            cudaEventRecord(cev[di][k], cst[di]);
                            busy[k] = di;
                        }
                        k ^= 1;
                    }
                    close(fd);
                }
                // (only the runs read are hashed: an expert left on the SSD tier stays unhashed until its first read)
            }
            for (int s2 = 0; s2 < 2; ++s2)
                if (cst[s2]) {   // each stream on its own GPU: a sync from the other GPU's context fails, and the bounce buffers
                                 // below would be freed under copies still reading them
                    cudaSetDevice(stages[(size_t) s2]->dev);
                    cudaStreamSynchronize(cst[s2]);
                    for (auto& ev : cev[s2]) cudaEventDestroy(ev);
                    cudaStreamDestroy(cst[s2]);
                }
            for (auto* b : bounce) free_pinned(b);
        });
    for (auto& x : th) x.join();
    cur_dev = -1;
    if (bad) return fail(err, "loading the experts: " + bad_msg);

    // ---- 6. the tables
    for (int64_t key = 0; key < NK; ++key) {
        const int32_t slot = plan_slot[(size_t) key];
        if (slot < 0) continue;
        owner[(size_t) slot] = (int32_t) key;
        resident[(size_t) key] = slot;
        stages[(size_t) slot_dev[(size_t) slot]]->st.fills++;
    }
    for (auto& gp : stages) gp->st.slots = gp->n_slots;
    res_dirty = true;
    {   // every slot read back on its GPU and compared with the file (Strata's verify_slot, at startup: a mismatch refuses the run)
        std::vector<std::pair<int32_t, int32_t>> sk;
        for (int64_t key = 0; key < NK; ++key)
            if (plan_slot[(size_t) key] >= 0) sk.push_back({plan_slot[(size_t) key], (int32_t) key});
        const auto tv = clk::now();
        if (!verify_slots(sk, true, err)) return false;
        if (opt.verbose)
            fprintf(stderr, "strata-x: %zu expert slots verified against the file (%.0f ms)\n", sk.size(), ms_since(tv));
    }
    if (opt.verbose) {
        const double secs = std::max(1e-3, ms_since(t0) / 1000.0);
        fprintf(stderr, "strata-x: load pass, summed over the %d readers: reads %.1f s, RAM copies %.1f s, slot copies issued %.1f s, "
                        "waits for the bounce buffers %.1f s\n", nth, t_read / 1e9, t_copy / 1e9, t_issue / 1e9, t_wait / 1e9);
        fprintf(stderr, "strata-x: experts loaded in %.1f s (%.2f GiB read, %.2f GiB/s, %d readers%s): %lld in VRAM (%lld from the "
                        "profile%s), %.2f GiB in RAM%s\n",
                secs, read_bytes.load() / 1073741824.0, read_bytes.load() / 1073741824.0 / secs, nth,
                buffered > 0 ? ", some buffered" : "", (long long) (filled + nx_count), (long long) filled,
                nx_count ? " + the drafter's, in their own region" : "", want_ram / 1073741824.0, arena_pinned ? " (pinned)" : " (pageable)");
        uint64_t file_only = 0;
        for (int64_t key = hp.n_dense * NE; key < NKX; ++key)
            if (plan_slot[(size_t) key] < 0 && ram_off[(size_t) key] < 0) file_only += L[(size_t) (key / NE)].fmt.bytes;
        // moex (DSV41_BACKLOG #48): say so when there is no RAM copy at all, and where the rest comes from on a pack
        fprintf(stderr, "strata-x: resident RAM mode: the complement %.2f GiB %s, the lend region %lld experts%s, %zu exchange blobs; "
                        "%.2f GiB read from %s when needed\n",
                comp / 1073741824.0,
                what_fits ? "does NOT fit: the hottest of it by the profile"
                          : budget == 0 ? "NOT in RAM: no RAM copy (MemAvailable is within the headroom)" : "in RAM",
                (long long) lend_kept, lend_kept_all ? "" : " (not all of it)", xpool.size(), file_only / 1073741824.0,
                rio ? (opt.experts_gguf.empty() ? "the pack's records" : "the records (the GGUF set's slices)") : "the GGUF");
    }
    // moex (DSV41_PLAN §69, §73): the warm fill's reads, started before the load pass, waited for and published here; the cache's
    // fault-in and registration joined (before READY: a GPU copies from these slots from now on)
    if (rio && hp.v41) ssd_warm_finish();
    if (rio && rio->pin_thr.joinable()) rio->pin_thr.join();
    return true;
}

/// Hashes slots on their GPUs (after everything queued on the stage's streams) and compares with the file's hashes.  `fatal`:
/// a mismatch fails (the load); otherwise the slot leaves the tier - the expert is computed by the CPU from its own copy -
/// and the mismatch is reported.
bool Engine::Impl::verify_slots(const std::vector<std::pair<int32_t, int32_t>>& sk, bool fatal, std::string& err) {
    if (sk.empty()) return true;
    const int64_t NE = hp.n_expert;
    const size_t need = sk.size() * 3;
    // moex (DSV41_PLAN §35), opt-in STRATA_DSV41_HASH_WIDE=1: a few slots' check (the swaps a token brought in) in kWideParts
    // parts a slice, summed here - the same hashes; one launch of 3 blocks a slot held the host 1.9 ms between windows
    static const bool wide = [] {
        const char* e = std::getenv("STRATA_DSV41_HASH_WIDE");
        return e && e[0] == '1';
    }();
    constexpr int kWideParts = 32;
    const bool wide_now = wide && sk.size() <= 64;   // (the load's check, ~2,000 slots, fills the GPUs as it is)
    if (need > vcap) {
        if (vjobs_h) cudaFreeHost(vjobs_h);
        if (vout_h) cudaFreeHost(vout_h);
        if (vpart_h) cudaFreeHost(vpart_h);
        vjobs_h = nullptr;
        vout_h = nullptr;
        vpart_h = nullptr;
        const size_t cap = std::max<size_t>(need, 4096);
        if (cudaHostAlloc((void**) &vjobs_h, cap * sizeof(SliceJob), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaHostAlloc((void**) &vout_h, cap * sizeof(uint64_t), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &vjobs_d, vjobs_h, 0) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &vout_d, vout_h, 0) != cudaSuccess)
            return fail(err, "the slot check's mapped buffers");
        vcap = cap;
    }
    if (wide_now && !vpart_h) {   // 64 slots' 192 slices, kWideParts each
        if (cudaHostAlloc((void**) &vpart_h, (size_t) 192 * kWideParts * sizeof(uint64_t), cudaHostAllocMapped | cudaHostAllocPortable) !=
                cudaSuccess ||
            cudaHostGetDevicePointer((void**) &vpart_d, vpart_h, 0) != cudaSuccess)
            return fail(err, "the slot check's mapped buffers");
    }
    // the jobs, grouped by stage (one launch each)
    size_t at = 0;
    std::vector<size_t> first(stages.size() + 1, 0);
    std::vector<std::pair<int32_t, int32_t>> order;
    for (size_t di = 0; di < stages.size(); ++di) {
        first[di] = at;
        for (const auto& [slot, key] : sk) {
            if (slot_dev[(size_t) slot] != (int) di) continue;
            const Layer& l = L[(size_t) (key / NE)];
            uint8_t* p = slot_ptr(slot);
            vjobs_h[at++] = {p, (uint64_t) l.sg};
            vjobs_h[at++] = {p + l.fmt.up_off, (uint64_t) l.su};
            vjobs_h[at++] = {p + l.fmt.down_off, (uint64_t) l.sd};
            order.push_back({slot, key});
        }
    }
    first[stages.size()] = at;
    for (size_t di = 0; di < stages.size(); ++di) {
        const int n = (int) (first[di + 1] - first[di]);
        if (n == 0) continue;
        Stage& g = *stages[di];
        set_dev(g.dev);
        // on the compute stream, idle between windows: after the fills queued there (fill_slot), and the swaps' copies on the
        // copy stream have landed already (their events completed) - the copies still in flight there are not waited for
        if (wide_now) slices_hash_parts(vjobs_d + first[di], n, kWideParts, vpart_d + first[di] * kWideParts, g.s);
        else slices_hash(vjobs_d + first[di], n, vout_d + first[di], g.s);
    }
    for (size_t di = 0; di < stages.size(); ++di) {
        if (first[di + 1] == first[di]) continue;
        set_dev(stages[di]->dev);
        if (cudaStreamSynchronize(stages[di]->s) != cudaSuccess) return fail(err, "the slot check failed on the GPU");
    }
    if (wide_now)   // each slice's parts summed (modulo 2^64): slices_hash's value
        for (size_t i = 0; i < at; ++i) {
            uint64_t h = 0;
            for (int p = 0; p < kWideParts; ++p) h += ((volatile uint64_t*) vpart_h)[i * kWideParts + p];
            vout_h[i] = h;
        }
    for (size_t i = 0; i < order.size(); ++i) {
        const auto [slot, key] = order[i];
        ++verify_checked;
        if (!khash_ok[(size_t) key]) continue;   // never read at load (its layer had nothing planned): nothing to compare with
        std::string parts;   // every part that differs (gate / up / down)
        for (int r = 0; r < 3; ++r)
            if (((volatile uint64_t*) vout_h)[i * 3 + r] != khash[(size_t) key * 3 + r])
                parts += (parts.empty() ? "" : "+") + std::string(r == 0 ? "gate" : r == 1 ? "up" : "down");
        if (parts.empty()) continue;
        ++verify_bad;
        char msg[200];
        snprintf(msg, sizeof msg, "expert slot %d (layer %lld, expert %lld) differs from the file in its %s rows", (int) slot,
                 (long long) (key / NE), (long long) (key % NE), parts.c_str());
        if (fatal) return fail(err, msg);
        fprintf(stderr, "strata-x: ERROR: %s - the slot leaves the tier, the CPU computes the expert\n", msg);
        if (resident[(size_t) key] == slot) resident[(size_t) key] = -1;
        if (owner[(size_t) slot] == key) owner[(size_t) slot] = -1;
        res_dirty = true;
    }
    cur_dev = -1;
    return true;
}

/// Copies (il, e)'s blob into `slot` on that slot's GPU (on its compute stream, after the work queued before: the slot's
/// previous expert may be in use by it) and updates the tables.  From the pinned RAM copy: a queued DMA; otherwise the three
/// slices from the file through the pinned staging blob, synchronously.
namespace {
void record_hashes(Engine::Impl& m, const uint8_t* rec, const Layer& l, uint64_t* out);   // (below: moex, §35)
}  // namespace

bool Engine::Impl::fill_slot(int32_t slot, int64_t il, int64_t e, std::string& err) {
    const int64_t NE = hp.n_expert;
    const Layer& l = L[(size_t) il];
    Stage& g = *stages[(size_t) slot_dev[(size_t) slot]];
    set_dev(g.dev);
    const Src sr = src(il * NE + e);
    if (sr.pinned) {
        if (cudaMemcpyAsync(slot_ptr(slot), sr.g, l.fmt.bytes, cudaMemcpyHostToDevice, g.s) != cudaSuccess)
            return fail(err, "slot fill failed");
    } else if (on_ssd(il * NE + e)) {   // DeepSeek-V4.1's SSD tier: the record from the SSD cache when a window read it (a swap's
                                        // incoming expert usually missed just now), else with O_DIRECT - then one copy
        if (cudaStreamSynchronize(g.s) != cudaSuccess) return fail(err, "stream failed before a slot fill");
        const uint8_t* rec = nullptr;
        if (auto it = rio->where.find(il * NE + e); it != rio->where.end()) {   // between windows: no layer uses the cache
            RecordIo::Slot& c = rio->cs[(size_t) it->second];
            if (c.h) rio->io->wait(c.h);
            if (!c.h || !c.h->error.load()) rec = c.p;
        }
        if (rec) {
            ++rio->fill_cached;
        } else {
            if (!rec_read(il, &e, 1, &staging, err)) return false;
            rec = staging;
            ++rio->fill_reads;
        }
        if (!khash_ok[(size_t) (il * NE + e)]) {   // its first read: the hashes the slot checks compare with from now on
            record_hashes(*this, rec, l, &khash[(size_t) (il * NE + e) * 3]);
            khash_ok[(size_t) (il * NE + e)] = 1;
        }
        if (cudaMemcpyAsync(slot_ptr(slot), rec, l.fmt.bytes, cudaMemcpyHostToDevice, g.s) != cudaSuccess ||
            cudaStreamSynchronize(g.s) != cudaSuccess)
            return fail(err, "slot fill failed");
    } else {
        if (cudaStreamSynchronize(g.s) != cudaSuccess) return fail(err, "stream failed before a slot fill");
        memcpy(staging, sr.g, l.sg);
        memcpy(staging + l.fmt.up_off, sr.u, l.su);
        memcpy(staging + l.fmt.down_off, sr.d, l.sd);
        if (cudaMemcpyAsync(slot_ptr(slot), staging, l.fmt.bytes, cudaMemcpyHostToDevice, g.s) != cudaSuccess ||
            cudaStreamSynchronize(g.s) != cudaSuccess)
            return fail(err, "slot fill failed");
    }
    const int32_t prev = owner[(size_t) slot];
    if (prev >= 0) resident[(size_t) prev] = -1;
    owner[(size_t) slot] = (int32_t) (il * NE + e);
    resident[(size_t) (il * NE + e)] = slot;
    g.st.fills++;
    res_dirty = true;
    return true;
}

/// THE ADAPTIVE TIER (Strata's rule, see EngineOptions): between windows, finished copies become resident, empty slots
/// take this window's misses (a first-use fill, when no profile seeded them), and every `adapt_every` tokens a swap round.
void Engine::Impl::maintain_cache() {
    struct PenUpdate {   // the opt-in route bias's table, once this window's fills and swaps are in the tables
        Engine::Impl& m;
        ~PenUpdate() {
            const auto t = clk::now();
            m.route_pen_update();
            m.stages[0]->st.ms_up_pen += ms_since(t);
        }
    } pen_update{*this};
    if (n_slots == 0 || !opt.adaptive) {
        token_misses.clear();
        if (!verify_queue.empty()) {
            std::string verr;
            std::vector<std::pair<int32_t, int32_t>> q;
            q.swap(verify_queue);
            verify_slots(q, false, verr);
        }
        return;
    }
    std::string err;
    const int64_t NE = hp.n_expert;
    EngineStats& ust = stages[0]->st;   // (moex, §35: where the upkeep's time goes)
    auto tp = clk::now();
    finalize_swaps(false);
    ust.ms_up_final += ms_since(tp);
    tp = clk::now();
    std::vector<int64_t> next_empty(stages.size());
    for (size_t di = 0; di < stages.size(); ++di) next_empty[di] = stages[di]->slot_base;
    auto slot_end = [&](size_t di) { return stages[di]->slot_base + stages[di]->n_slots; };
    for (int32_t key : token_misses) {
        if (resident[(size_t) key] >= 0 || key_pending[(size_t) key]) continue;
        const int64_t il = key / NE, e = key % NE;
        const size_t di = (size_t) ldev[(size_t) il];
        while (next_empty[di] < slot_end(di) && (owner[(size_t) next_empty[di]] >= 0 || slot_pending[(size_t) next_empty[di]]))
            ++next_empty[di];
        if (next_empty[di] >= slot_end(di) || L[(size_t) il].fmt.bytes > slot_bytes ||
            L[(size_t) il].fmt.bytes > slot_cap((int32_t) next_empty[di]))   // moex (§24): a sized slot holds its own size
            continue;
        if (fill_slot((int32_t) next_empty[di], il, e, err)) verify_queue.push_back({(int32_t) next_empty[di], key});
    }
    token_misses.clear();
    ust.ms_up_fill += ms_since(tp);
    tp = clk::now();
    if (!verify_queue.empty()) {   // every fill read back before a window uses it ... after it lands: the tier's own check
        std::string verr;
        std::vector<std::pair<int32_t, int32_t>> q;
        q.swap(verify_queue);
        verify_slots(q, false, verr);
    }
    ust.ms_up_verify += ms_since(tp);
    tp = clk::now();
    struct RoundTime {
        EngineStats& st;
        clk::time_point t;
        ~RoundTime() { st.ms_up_round += ms_since(t); }
    } round_time{ust, tp};
    // a round every `adapt_every` TOKENS fed (a plain decode's windows; a verify window of T rows counts T), so speculation's
    // fewer, wider windows keep the tier adapting at the same pace per token
    if (opt.adapt_every > 0) {
        windows_seen += std::max(1, last_T);
        if (windows_seen >= opt.adapt_every) {
            windows_seen = 0;
            swap_round();
        }
    }
}

namespace {
/// moex (DSV41_PLAN §35): slices_hash_host's value of a record's gate / up / down slices (`out[3]`), with
/// STRATA_DSV41_HASH_POOL=1 summed over kHashParts word ranges a slice on the CPU pool's threads (idle between windows) - the
/// hash is a sum of products modulo 2^64, so the parts' sum is the one-thread value: a first-time swapped-in record's 14.5
/// MiB took the host ~2.2 ms between windows, up to 3.4 ms a token
constexpr int kHashParts = 16;
struct RecHashJob {
    const uint8_t* p[3];
    uint64_t len[3];
    uint64_t part[3][kHashParts];
};
void rec_hash_task(void* a, int t) {
    RecHashJob& j = *static_cast<RecHashJob*>(a);
    const int s = t / kHashParts, q = t % kHashParts;
    const uint8_t* p = j.p[s];
    const uint64_t nw = j.len[s] / 8, tail = j.len[s] % 8;
    const uint64_t i0 = nw * (uint64_t) q / kHashParts, i1 = nw * (uint64_t) (q + 1) / kHashParts;
    uint64_t h = 0;
    for (uint64_t i = i0; i < i1; ++i) {
        uint64_t w;
        memcpy(&w, p + i * 8, 8);
        h += w * (2 * i + 1);
    }
    if (tail && q == kHashParts - 1) {
        uint64_t w = 0;
        for (uint64_t b = 0; b < tail; ++b) w |= (uint64_t) p[nw * 8 + b] << (8 * b);
        h += w * (2 * nw + 1);
    }
    j.part[s][q] = h;
}
void record_hashes(Engine::Impl& m, const uint8_t* rec, const Layer& l, uint64_t* out) {
    static const bool par = [] {
        const char* e = std::getenv("STRATA_DSV41_HASH_POOL");
        return e && e[0] == '1';
    }();
    if (!par || !m.pool) {
        out[0] = slices_hash_host(rec, l.sg);
        out[1] = slices_hash_host(rec + l.fmt.up_off, l.su);
        out[2] = slices_hash_host(rec + l.fmt.down_off, l.sd);
        return;
    }
    RecHashJob j{};
    j.p[0] = rec;
    j.p[1] = rec + l.fmt.up_off;
    j.p[2] = rec + l.fmt.down_off;
    j.len[0] = l.sg;
    j.len[1] = l.su;
    j.len[2] = l.sd;
    {
        std::lock_guard<std::mutex> lk(m.pool_mu);
        m.pool->run_tasks(rec_hash_task, &j, 3 * kHashParts);
    }
    for (int s = 0; s < 3; ++s) {
        uint64_t h = 0;
        for (int q = 0; q < kHashParts; ++q) h += j.part[s][q];
        out[s] = h;
    }
    static std::atomic<bool> checked{false};   // the first record also the one-thread way: the same three values
    if (!checked.exchange(true)) {
        const bool same = out[0] == slices_hash_host(rec, l.sg) && out[1] == slices_hash_host(rec + l.fmt.up_off, l.su) &&
                          out[2] == slices_hash_host(rec + l.fmt.down_off, l.sd);
        fprintf(stderr, "strata-x: the records' hashes on the CPU pool (STRATA_DSV41_HASH_POOL): %s the one-thread values\n",
                same ? "equal to" : "UNLIKE");
    }
}
}  // namespace

/// moex (DSV41_BACKLOG #143, DSV41_PLAN §75; STRATA_DSV41_SLOT_CHECK_ASYNC=0: the check after the copy lands, as before): a
/// swap's slot check queued on the copy stream right behind its copy - verify_slots' parts kernel over the slot's three
/// slices, the sums into mapped memory - so the copy's event completes with the sums written and finalize_swaps compares them
/// before the slot becomes resident. The host launched the same kernel on the compute stream between windows and waited for
/// it: 0.45-0.56 ms a token on UD-Q2_K_XL and native (the decode line's "the slot check"). ps.vq: its area, -1: none (the
/// buffers failed, or all kVq in flight) - then the old way.
static void queue_slot_check(Engine::Impl& m, Engine::Impl::PendingSwap& ps, Stage& g, const Layer& l) {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_DSV41_SLOT_CHECK_ASYNC");
        return !(e && e[0] == '0');
    }();
    ps.vq = -1;
    if (!on) return;
    constexpr int kVq = Engine::Impl::kVq, P = Engine::Impl::kVqParts;
    if (!m.vq_tried) {
        m.vq_tried = true;
        if (cudaHostAlloc((void**) &m.vq_jobs_h, (size_t) kVq * 3 * sizeof(SliceJob), cudaHostAllocMapped | cudaHostAllocPortable) !=
                cudaSuccess ||
            cudaHostAlloc((void**) &m.vq_out_h, (size_t) kVq * 3 * P * sizeof(uint64_t), cudaHostAllocMapped | cudaHostAllocPortable) !=
                cudaSuccess ||
            cudaHostGetDevicePointer((void**) &m.vq_jobs_d, m.vq_jobs_h, 0) != cudaSuccess ||
            cudaHostGetDevicePointer((void**) &m.vq_out_d, m.vq_out_h, 0) != cudaSuccess) {
            cudaGetLastError();
            if (m.vq_jobs_h) cudaFreeHost(m.vq_jobs_h);
            if (m.vq_out_h) cudaFreeHost(m.vq_out_h);
            m.vq_jobs_h = nullptr;
            m.vq_out_h = nullptr;
        } else {
            for (int i = kVq - 1; i >= 0; --i) m.vq_free.push_back(i);
            if (const char* e = std::getenv("STRATA_X_SLOT_CHECK_FAULT")) m.vq_fault = std::atoll(e);
        }
    }
    if (!m.vq_jobs_h || m.vq_free.empty()) return;
    ps.vq = m.vq_free.back();
    m.vq_free.pop_back();
    SliceJob* j = m.vq_jobs_h + (size_t) ps.vq * 3;
    uint8_t* p = m.slot_ptr(ps.slot);
    j[0] = {p, (uint64_t) l.sg};
    j[1] = {p + l.fmt.up_off, (uint64_t) l.su};
    j[2] = {p + l.fmt.down_off, (uint64_t) l.sd};
    std::atomic_thread_fence(std::memory_order_release);
    slices_hash_parts(m.vq_jobs_d + (size_t) ps.vq * 3, 3, P, m.vq_out_d + (size_t) ps.vq * 3 * P, g.cs);
}

/// ... read once the copy's event has completed: true = the slot holds the file's bytes (or the expert has no hashes to
/// compare with, verify_slots' rule). A mismatch is reported and counted as there, and the caller leaves the slot empty.
static bool slot_check_read(Engine::Impl& m, const Engine::Impl::PendingSwap& ps) {
    constexpr int P = Engine::Impl::kVqParts;
    ++m.verify_checked;
    if (!m.khash_ok[(size_t) ps.key]) return true;
    std::atomic_thread_fence(std::memory_order_acquire);
    const volatile uint64_t* out = m.vq_out_h + (size_t) ps.vq * 3 * P;
    std::string parts;
    for (int r = 0; r < 3; ++r) {
        uint64_t h = 0;
        for (int q = 0; q < P; ++q) h += out[r * P + q];   // each slice's parts summed (modulo 2^64): slices_hash's value
        if (r == 0 && m.vq_fault > 0 && --m.vq_fault == 0) h ^= 1u;   // (test: STRATA_X_SLOT_CHECK_FAULT)
        if (h != m.khash[(size_t) ps.key * 3 + r]) parts += (parts.empty() ? "" : "+") + std::string(r == 0 ? "gate" : r == 1 ? "up" : "down");
    }
    if (parts.empty()) return true;
    ++m.verify_bad;
    const int64_t NE = m.hp.n_expert;
    fprintf(stderr, "strata-x: ERROR: expert slot %d (layer %lld, expert %lld) differs from the file in its %s rows - the slot "
                    "stays out of the tier, the CPU computes the expert\n", (int) ps.slot, (long long) (ps.key / NE),
            (long long) (ps.key % NE), parts.c_str());
    return false;
}

/// The incoming copy of a swap, queued after the compute stream's work (a captured graph may still read the slot).
/// -> 0: in flight (ps.ev recorded), 1: done (a synchronous fill from the file), -1: that synchronous fill failed - nothing
/// recorded; the caller ends the swap with the slot empty.  (moex, Maya #646: a failed fill used to read as "in flight", and
/// its never-recorded event as landed, so the slot was made resident over bytes that were not the expert's - and a swap
/// past its victim's copy-out went through that step again, looping in finalize_swaps(true).)
static int issue_incoming(Engine::Impl& m, Engine::Impl::PendingSwap& ps) {
    Stage& g = *m.stages[(size_t) ps.di];
    const int64_t NE = m.hp.n_expert, il = ps.key / NE, e = ps.key % NE;
    const Layer& l = m.L[(size_t) il];
    m.set_dev(g.dev);
    const Engine::Impl::Src sr = m.src(ps.key);
    if (!sr.pinned && ps.cslot >= 0) {   // DeepSeek-V4.1: from its SSD cache slot (pinned, held), queued like a RAM copy's
        const uint8_t* rec = m.rio->cs[(size_t) ps.cslot].p;
        if (!m.khash_ok[(size_t) ps.key]) {   // its first fill: the hashes the slot check compares with from now on
            const auto tk = clk::now();
            record_hashes(m, rec, l, &m.khash[(size_t) ps.key * 3]);
            m.khash_ok[(size_t) ps.key] = 1;
            m.stages[0]->st.ms_up_khash += ms_since(tk);
            m.stages[0]->st.up_khash++;
        }
        cudaEventRecord(g.swap_ev, g.s);
        cudaStreamWaitEvent(g.cs, g.swap_ev, 0);
        cudaMemcpyAsync(m.slot_ptr(ps.slot), rec, l.fmt.bytes, cudaMemcpyHostToDevice, g.cs);
        queue_slot_check(m, ps, g, l);   // (moex, #143: behind the copy, before its event)
        cudaEventRecord(ps.ev, g.cs);
        ++m.rio->fill_cached;
        ps.phase = 1;
        return 0;   // in flight
    }
    if (!sr.pinned) {   // from the file: a synchronous fill through the staging blob
        std::string err;
        if (!m.fill_slot(ps.slot, il, e, err)) {
            cudaGetLastError();
            fprintf(stderr, "strata-x: a swap's fill of layer %lld expert %lld failed (%s); the slot stays empty\n", (long long) il,
                    (long long) e, err.c_str());
            return -1;
        }
        m.slot_pending[(size_t) ps.slot] = 0;
        m.key_pending[(size_t) ps.key] = 0;
        g.st.swaps++;
        return 1;   // done: no event to wait for
    }
    cudaEventRecord(g.swap_ev, g.s);
    cudaStreamWaitEvent(g.cs, g.swap_ev, 0);
    cudaMemcpyAsync(m.slot_ptr(ps.slot), sr.g, l.fmt.bytes, cudaMemcpyHostToDevice, g.cs);
    queue_slot_check(m, ps, g, l);   // (moex, #143)
    cudaEventRecord(ps.ev, g.cs);
    ps.phase = 1;
    return 0;   // in flight
}

/// Copies that have landed (or, `wait`, all of them) move on: a victim's copy out (phase 0) lets the victim leave - its RAM
/// copy is now the exchange blob - and starts the incoming copy; an incoming copy (phase 1) makes the expert resident and
/// returns its RAM blob to the exchange pool (the exchange: the RAM copy keeps its size).
void Engine::Impl::finalize_swaps(bool wait) {
    for (bool again = true; again;) {
        again = false;
        for (size_t i = 0; i < pending.size();) {
            PendingSwap& ps = pending[i];
            set_dev(stages[(size_t) ps.di]->dev);
            if (wait) cudaEventSynchronize(ps.ev);
            else if (cudaEventQuery(ps.ev) != cudaSuccess) {
                cudaGetLastError();
                ++i;
                continue;
            }
            if (ps.phase == 0) {   // the victim is out: it leaves VRAM with the exchange blob as its RAM copy
                resident[(size_t) ps.key_out] = -1;
                if (ps.vslot >= 0) {   // (moex, §44: ... or with its record in a cache slot)
                    ssd_publish(ps.vslot, ps.key_out);
                } else {
                    ram_off[(size_t) ps.key_out] = ps.xb;
                    ram_cap[(size_t) ps.key_out] = ps.xb_cap;
                }
                owner[(size_t) ps.slot] = -1;
                res_dirty = true;
                const int r = issue_incoming(*this, ps);
                if (r == 0) {   // the incoming copy is in flight now
                    again = again || wait;
                    ++i;
                    continue;
                }
                if (r < 0) {   // its synchronous fill failed: the swap ends, the slot empty (the victim has left)
                    slot_pending[(size_t) ps.slot] = 0;
                    key_pending[(size_t) ps.key] = 0;
                }
            } else {
                // moex (#143): the check that ran behind the copy - a slot that differs from the file never becomes resident
                // (it stays empty, the expert keeps its RAM blob or cache record, and the CPU computes it)
                bool good = true;
                if (ps.vq >= 0) {
                    good = slot_check_read(*this, ps);
                    vq_free.push_back(ps.vq);
                }
                slot_pending[(size_t) ps.slot] = 0;
                key_pending[(size_t) ps.key] = 0;
                if (ps.cslot >= 0) ssd_release(ps.cslot);   // the copy from the SSD cache slot has landed
                if (good) {
                    owner[(size_t) ps.slot] = ps.key;
                    resident[(size_t) ps.key] = ps.slot;
                    if (ps.cslot >= 0 && swap_cache()) ssd_free_dup(ps.cslot, ps.key);   // (moex, §44: VRAM holds it now)
                    if (ps.vq < 0) verify_queue.push_back({ps.slot, ps.key});   // checked before the next window (maintain_cache)
                    if (ps.ret >= 0) {   // the exchange: the incoming expert's RAM blob is free now
                        ram_off[(size_t) ps.key] = -1;
                        ram_cap[(size_t) ps.key] = 0;
                        xpool.push_back({ps.ret, ps.ret_cap});
                    }
                }
                res_dirty = true;
            }
            cudaEventDestroy(ps.ev);
            pending[i] = pending.back();
            pending.pop_back();
        }
    }
}

void Engine::Impl::swap_round() {
    const int64_t E = hp.n_expert;
    struct Cand {
        float gain;
        int il, e;
        int32_t slot;
    };
    std::vector<Cand> plan;
    for (int64_t il = hp.n_dense; il < n_xl; ++il) {
        if (stage_of(il).n_slots == 0 || L[(size_t) il].fmt.bytes > slot_bytes) continue;
        if (il >= hp.n_layer && nx_count > 0) continue;   // the drafter's experts are pinned in their region
        std::vector<std::pair<float, int>> cands, victims;   // (usage, expert)
        for (int e = 0; e < (int) E; ++e) {
            const size_t key = (size_t) (il * E + e);
            if (resident[key] >= 0) {
                if (!slot_pinned[(size_t) resident[key]] && !slot_pending[(size_t) resident[key]]) victims.push_back({heat[key], e});
            } else if (heat[key] >= 2.0f && !key_pending[key]) {
                cands.push_back({heat[key], e});
            }
        }
        std::sort(cands.begin(), cands.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::sort(victims.begin(), victims.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < cands.size() && i < victims.size(); ++i) {
            if (cands[i].first < victims[i].first + 1.5f) break;   // the i-th hottest against the i-th coldest
            plan.push_back({cands[i].first - victims[i].first, (int) il, cands[i].second,
                            resident[(size_t) (il * E + victims[i].second)]});
        }
    }
    const int budget = opt.adapt_swaps > 0 ? opt.adapt_swaps
                                           : std::clamp((int) (256.0 / ((double) slot_bytes / 1048576.0)), 8, 96);
    if ((int) plan.size() > budget) {
        std::partial_sort(plan.begin(), plan.begin() + budget, plan.end(),
                          [](const Cand& a, const Cand& b) { return a.gain > b.gain; });
        plan.resize((size_t) budget);
    }
    std::vector<char> ordered(stages.size(), 0);
    int sync_left = 8;   // an incoming expert read from the file is a synchronous fill
    for (const Cand& c : plan) {
        Stage& g = *stages[(size_t) slot_dev[(size_t) c.slot]];
        const int32_t key_in = (int32_t) (c.il * E + c.e), vkey = owner[(size_t) c.slot];
        const Src s_in = src(key_in);
        // DeepSeek-V4.1's SSD tier: the incoming record from its SSD cache slot, copied asynchronously like a RAM copy (it was a
        // synchronous fill - a stream sync, the 14.5 MB copy and another sync on the host between windows, ~1.3 ms each, ~2.3 a
        // token warmed); one the cache does not hold waits for a later round (its read would stall the window ~5 ms)
        // (They keep the synchronous fills' cap of 8 a round: without it the tier swapped ~4 a token instead of ~2.4, and the
        // victims it dropped were read again - 2.2 SSD reads a token instead of 0.9, 2026-10-18.)
        int cslot = -1;
        if (!s_in.pinned && sync_left <= 0) continue;
        if (!s_in.pinned && rio && on_ssd(key_in) && ssd_async_fill()) {
            cslot = ssd_hold(key_in);
            if (cslot < 0) continue;
        }
        PendingSwap ps;
        ps.slot = c.slot;
        ps.key = key_in;
        ps.di = g.di;
        ps.cslot = cslot;
        ps.ret = s_in.pinned && arena && ram_off[(size_t) key_in] >= 0 ? ram_off[(size_t) key_in] : -1;
        ps.ret_cap = ps.ret >= 0 ? ram_cap[(size_t) key_in] : 0;
        set_dev(g.dev);
        if (vkey >= 0 && ram_off[(size_t) vkey] < 0 && cslot >= 0 && swap_cache() && on_ssd(vkey)) {
            // moex (DSV41_PLAN §44): the incoming record from its cache slot - the victim into a cache slot of its own (phase 0,
            // while it stays resident; published when its copy lands), not an exchange blob nothing would hand back
            const int vs = ssd_take_for(vkey);
            if (vs >= 0) {
                ps.vslot = vs;
                ps.key_out = vkey;
                ps.phase = 0;
                if (!ordered[(size_t) g.di]) {
                    cudaEventRecord(g.swap_ev, g.s);
                    cudaStreamWaitEvent(g.cs, g.swap_ev, 0);
                    ordered[(size_t) g.di] = 1;
                }
                cudaMemcpyAsync(rio->cs[(size_t) vs].p, slot_ptr(c.slot), L[(size_t) c.il].fmt.bytes, cudaMemcpyDeviceToHost, g.cs);
                cudaEventCreateWithFlags(&ps.ev, cudaEventDisableTiming);
                cudaEventRecord(ps.ev, g.cs);
                slot_pending[(size_t) c.slot] = 1;
                key_pending[(size_t) key_in] = 1;
                pending.push_back(ps);
                g.st.swaps++;
                g.st.fills++;
                continue;
            }
        }
        if (vkey >= 0 && ram_off[(size_t) vkey] < 0) {
            // the victim exists only in VRAM: copy it out to an exchange blob first, while it stays resident (phase 0); with no
            // exchange blob left it falls back to the file (it is still in the GGUF) and the incoming one keeps its RAM copy
            const uint64_t need = L[(size_t) c.il].fmt.bytes;
            size_t best = xpool.size();   // the smallest free blob that holds the victim
            for (size_t i = 0; i < xpool.size(); ++i)
                if (xpool[i].second >= need && (best == xpool.size() || xpool[i].second < xpool[best].second)) best = i;
            if (best == xpool.size() || !arena_pinned) {
                ps.ret = -1;
            } else {
                ps.xb = xpool[best].first;
                ps.xb_cap = xpool[best].second;
                xpool[best] = xpool.back();
                xpool.pop_back();
                ps.key_out = vkey;
                ps.phase = 0;
                if (!ordered[(size_t) g.di]) {
                    cudaEventRecord(g.swap_ev, g.s);
                    cudaStreamWaitEvent(g.cs, g.swap_ev, 0);
                    ordered[(size_t) g.di] = 1;
                }
                cudaMemcpyAsync(arena->data() + ps.xb, slot_ptr(c.slot), L[(size_t) c.il].fmt.bytes, cudaMemcpyDeviceToHost, g.cs);
                cudaEventCreateWithFlags(&ps.ev, cudaEventDisableTiming);
                cudaEventRecord(ps.ev, g.cs);
                slot_pending[(size_t) c.slot] = 1;
                key_pending[(size_t) key_in] = 1;
                pending.push_back(ps);
                g.st.swaps++;
                g.st.fills++;
                continue;
            }
        }
        // the victim has a RAM copy (or falls back to the file): it leaves at once - the CPU computes it meanwhile - and the
        // incoming expert keeps its RAM copy (no exchange: the lend region's slots keep theirs)
        if (vkey >= 0 && ram_off[(size_t) vkey] >= 0) ps.ret = -1;
        if (vkey >= 0) resident[(size_t) vkey] = -1;
        owner[(size_t) c.slot] = -1;
        res_dirty = true;
        slot_pending[(size_t) c.slot] = 1;
        key_pending[(size_t) key_in] = 1;
        cudaEventCreateWithFlags(&ps.ev, cudaEventDisableTiming);
        if (!s_in.pinned) --sync_left;
        if (const int r = issue_incoming(*this, ps); r != 0) {   // a synchronous fill from the file: done, or failed
            if (r < 0) {   // the slot stays empty (its victim has left already)
                slot_pending[(size_t) c.slot] = 0;
                key_pending[(size_t) key_in] = 0;
            }
            cudaEventDestroy(ps.ev);
            continue;
        }
        pending.push_back(ps);
        g.st.swaps++;
        g.st.fills++;
    }
    for (auto& hv : heat) hv *= opt.adapt_decay;
}

}  // namespace strata::x
