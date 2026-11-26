"""moex/tools/ernie55/ref/model.py - a reference model of ERNIE-5.5-Flash's forward pass on the CPU (PyTorch), read from the
FP8 release as Baidu publishes it (moex/docs/ERNIE55_PLAN.md §7).

Written from the two serving references - Baidu's vLLM plugin (vLLM-ERNIE ba322a5 on vLLM 5df63a7) and SGLang (f4284f2) -
as their kernels compute (the specs with file:line: ~/AI/Strata-X-data/ernie55-notes/vllm_spec.md, sglang_spec.md). It
computes the model's function, not one GPU's shortcuts:

- weights: the FP8 values times their power-of-two block scales, exactly, in fp32 (BF16 / F32 tensors as stored);
- arithmetic: fp32 throughout - no FP8 rounding of activations before the FP8 linears (the references' W8A8 GEMMs do
  that with GPU-dependent scale formats), no BF16 rounding between ops, fp32 caches;
- kept as the references do, because it decides which tokens are attended: the lightning indexer's queries and keys
  rounded to E4M3 with one power-of-two scale per 128 values (UE8M0), the window excluded from its candidates.

Where the references differ it takes vLLM-ERNIE's choice and the docstring of the function says so (the head reduce's
RMS factor; plain RoPE at 8e6 for DSpark's stages). Per-layer caches make the same code run a prompt chunk and a
one-token decode step: `Model.forward(ids, cache)` appends to the cache.

Use the vLLM venv (torch with float8_e4m3fn, safetensors): ~/AI/vLLM/.venv/bin/python -I ...
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field

import torch
from safetensors import safe_open

F32 = torch.float32
IDX_FP8 = True   # the indexer's E4M3 rounding of q and k (the references'); False: exact scores (a check's switch)


# ------------------------------------------------------------------------------------------------ config
@dataclass
class Cfg:
    n_layer: int
    hidden: int
    vocab: int
    ratios: list[int]                 # per layer: 128 HCA, <= 0 MLA + DSA
    first_dense: int                  # layers before it have a dense FFN
    ff_dense: int
    n_expert: int
    top_k: int
    ff_exp: int
    latent: int
    n_head: int                       # HCA query heads
    head_dim: int                     # HCA head (448 nope + 64 rope)
    rope_dim: int
    q_lora: int
    o_groups: int
    o_lora: int
    window: int
    # DSA (MLA) layers
    mla_heads: int
    mla_kv_lora: int
    mla_nope: int
    mla_rope: int
    mla_v: int
    idx_heads: int
    idx_dim: int
    idx_topk: int
    # RoPE
    rope_theta: float
    crope_theta: float
    yarn_factor: float
    yarn_orig: int
    beta_fast: float
    beta_slow: float
    # mHC
    hc: int
    sinkhorn: int
    hc_eps: float
    rms_eps: float
    # SiTU
    situ_beta: float
    situ_lin_beta: float
    # DSpark
    dspark_block: int
    dspark_rank: int
    dspark_noise: int
    dspark_targets: list[int] = field(default_factory=list)

    @staticmethod
    def load(path: pathlib.Path) -> "Cfg":
        c = json.loads(path.read_text())
        rs = c["rope_scaling"]
        assert rs.get("type", rs.get("rope_type")) == "yarn", rs
        assert c["hc_mult"] == 4 and c["hidden_act"] == "situ" and c["gated_linear_unit"] and c["scoring_func"] == "sigmoid"
        return Cfg(
            n_layer=c["num_hidden_layers"], hidden=c["hidden_size"], vocab=c["vocab_size"], ratios=list(c["compress_ratios"]),
            first_dense=c["first_k_dense_replace"], ff_dense=c["intermediate_size"], n_expert=c["n_routed_experts"],
            top_k=c["num_experts_per_tok"], ff_exp=c["moe_intermediate_size"], latent=c["moe_latent_size"],
            n_head=c["num_attention_heads"], head_dim=c["head_dim"], rope_dim=c["qk_rope_head_dim"], q_lora=c["q_lora_rank"],
            o_groups=c["o_groups"], o_lora=c["o_lora_rank"], window=c["sliding_window"],
            mla_heads=c["hybrid_mla_num_attention_heads"], mla_kv_lora=c["hybrid_mla_kv_lora_rank"],
            mla_nope=c["hybrid_mla_qk_nope_head_dim"], mla_rope=c["hybrid_mla_qk_rope_head_dim"], mla_v=c["hybrid_mla_v_head_dim"],
            idx_heads=c["index_n_heads"], idx_dim=c["index_head_dim"], idx_topk=c["index_topk"],
            rope_theta=float(c["rope_theta"]), crope_theta=float(c["compress_rope_theta"]), yarn_factor=float(rs["factor"]),
            yarn_orig=int(rs["original_max_position_embeddings"]), beta_fast=float(rs["beta_fast"]), beta_slow=float(rs["beta_slow"]),
            hc=c["hc_mult"], sinkhorn=c["hc_sinkhorn_iters"], hc_eps=float(c["hc_eps"]), rms_eps=float(c["rms_norm_eps"]),
            situ_beta=float(c["activation_situ_beta"]), situ_lin_beta=float(c["activation_situ_linear_beta"]),
            dspark_block=c["dspark_block_size"], dspark_rank=c["dspark_markov_rank"], dspark_noise=c["dspark_noise_token_id"],
            dspark_targets=list(c["dspark_target_layer_ids"]))

    def is_dsa(self, il: int) -> bool:
        """vLLM-ERNIE's rule (config.py:156-160): a layer is MLA + DSA when its ratio is <= 0; layers past the table
        (DSpark's stages) are window-only HCA-class blocks"""
        return il < len(self.ratios) and self.ratios[il] <= 0


# ------------------------------------------------------------------------------------------------ weights
class Weights:
    """The release's tensors by name: model.safetensors.index.json (in the model directory, or `index` when the download
    has not written it yet) and its shards, opened lazily. f32() dequantizes an FP8 tensor with its `<name>_scale_inv`
    (F32, one scale per 128 x 128 block, ceil-divided at the edges) exactly."""

    def __init__(self, model_dir: str | pathlib.Path, index: str | pathlib.Path | None = None):
        self.dir = pathlib.Path(model_dir)
        ip = pathlib.Path(index) if index else self.dir / "model.safetensors.index.json"
        self.map: dict[str, str] = json.loads(ip.read_text())["weight_map"]
        self._h: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self.map

    def raw(self, name: str) -> torch.Tensor:
        shard = self.map[name]
        h = self._h.get(shard)
        if h is None:
            h = safe_open(str(self.dir / shard), framework="pt", device="cpu")
            self._h[shard] = h
        return h.get_tensor(name)

    def f32(self, name: str) -> torch.Tensor:
        t = self.raw(name)
        if t.dtype == torch.float8_e4m3fn:
            s = self.raw(name + "_scale_inv").to(F32)
            out, inn = t.shape
            if out % 128 == 0 and inn % 128 == 0:   # (every block whole: the same products, no expanded scale tensor)
                return (t.to(F32).view(out // 128, 128, inn // 128, 128) * s.view(out // 128, 1, inn // 128, 1)).view(out, inn)
            sb = s.repeat_interleave(128, 0)[:out].repeat_interleave(128, 1)[:, :inn]
            return t.to(F32) * sb
        return t.to(F32)


# ------------------------------------------------------------------------------------------------ small ops
def rms(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm, eps inside the root: x * rsqrt(mean(x^2) + eps) (* w)"""
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y if w is None else y * w


def situ_glu(g: torch.Tensor, u: torch.Tensor, beta: float, lin_beta: float) -> torch.Tensor:
    """SiTU-GLU: beta tanh(g / beta) sigmoid(g) * lin_beta tanh(u / lin_beta) (vLLM activation.py:162-202; the CUDA kernel
    writes sigmoid as (1 + tanh(g / 2)) / 2 - the same function)"""
    return beta * torch.tanh(g / beta) * torch.sigmoid(g) * (lin_beta * torch.tanh(u / lin_beta))


def multimax(x: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The head's piece-wise temperature (vLLM-ERNIE nvidia/model.py:97-159), left to right in fp32:
    x + t0 relu(r0 - x) + t1 relu(x - r1) + t2 relu(r2 - x)^2 + t3 relu(x - r3)^2"""
    y = x + t[0] * torch.relu(r[0] - x)
    y = y + t[1] * torch.relu(x - r[1])
    y = y + t[2] * torch.relu(r[2] - x).square()
    return y + t[3] * torch.relu(x - r[3]).square()


def fp8_fake(x: torch.Tensor, group: int = 128) -> torch.Tensor:
    """E4M3 with one power-of-two scale per `group` values along the last dim (UE8M0, the indexer's format: vLLM
    per_token_group_quant_fp8 with use_ue8m0): s = 2^ceil(log2(max(amax, 1e-10) / 448)), q = e4m3(clamp(x / s, +-448)),
    returned dequantized"""
    sh = x.shape
    g = x.reshape(-1, group)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (g / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(F32)
    return (q * s).reshape(sh)


_HAD: dict[int, torch.Tensor] = {}


def hadamard(x: torch.Tensor) -> torch.Tensor:
    """The orthonormal Walsh-Hadamard transform over the last dim (Sylvester order, / sqrt(n)). The indexer applies the
    same transform to q and k, so their dot product is unchanged; it spreads outliers before the FP8 rounding."""
    n = x.shape[-1]
    h = _HAD.get(n)
    if h is None:
        h = torch.ones(1, 1)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        h = h / math.sqrt(n)
        _HAD[n] = h
    return x @ h.T


# ------------------------------------------------------------------------------------------------ RoPE
def inv_freq_plain(dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=F32) / dim))


def inv_freq_yarn(dim: int, theta: float, factor: float, orig: int, beta_fast: float, beta_slow: float) -> torch.Tensor:
    """DeepSeek YaRN as vLLM's DeepseekV4ScalingRotaryEmbedding builds it (deepseek_scaling_rope.py:73-101, common.py:34-76):
    the dims below the correction range keep their frequency, those above are divided by `factor`, a linear ramp
    between; the magnitude correction is 1 (mscale = mscale_all_dim = 0)"""
    def corr(rot: float) -> float:
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(theta))
    low, high = math.floor(corr(beta_fast)), math.ceil(corr(beta_slow))
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    base = theta ** (torch.arange(0, dim, 2, dtype=F32) / dim)
    extrap, interp = 1.0 / base, 1.0 / (factor * base)
    ramp = ((torch.arange(dim // 2, dtype=F32) - low) / (high - low)).clamp(0, 1)
    return interp * ramp + extrap * (1 - ramp)


def rope_gptj(x: torch.Tensor, pos: torch.Tensor, inv: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """GPT-J (interleaved) rotation of the last dim's pairs (2i, 2i+1) at positions `pos` (x [T, ..., d], pos [T]); the
    angle in fp32 as the references' tables have it. inverse: rotation by -theta (the HCA output's de-RoPE)"""
    ang = pos.to(F32)[:, None] * inv[None, :]
    c, s = torch.cos(ang), torch.sin(ang)
    if inverse:
        s = -s
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [inv.shape[0]]
    c, s = c.reshape(shape), s.reshape(shape)
    a, b = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


def rope_neox(x: torch.Tensor, pos: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    """NeoX (halves) rotation of the last dim: pairs (i, i + d/2)"""
    ang = pos.to(F32)[:, None] * inv[None, :]
    c, s = torch.cos(ang), torch.sin(ang)
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [inv.shape[0]]
    c, s = c.reshape(shape), s.reshape(shape)
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return torch.cat([a * c - b * s, b * c + a * s], -1)


# ------------------------------------------------------------------------------------------------ mHC
@dataclass
class HcParams:
    fn: torch.Tensor      # [24, 4 * hidden]
    base: torch.Tensor    # [24]
    scale: torch.Tensor   # [3]: alpha_pre, alpha_post, alpha_res


def hc_params(w: Weights, p: str) -> HcParams:
    return HcParams(fn=w.f32(p + ".mapping_proj.weight").T.contiguous(), base=w.f32(p + ".bias"),
                    scale=torch.cat([w.f32(p + ".alpha_pre"), w.f32(p + ".alpha_post"), w.f32(p + ".alpha_res")]))


def mhc_pre(R: torch.Tensor, hp: HcParams, cfg: Cfg, hook=None, il: int = -1):
    """One sublayer's pre (vLLM-ERNIE mhc_kernels.py:262-317; its torch reference tests/models/ernie5_5/test_mhc.py:18-33):
    the mixer's RMS factor 1 / (sqrt(mean) + eps) - ERNIE's, eps outside the root. R [T, 4, n] -> (the collapsed input
    [T, n], post [T, 4], comb [T, 4, 4] with row j the source stream and column k the destination)"""
    T, S, n = R.shape
    x = R.reshape(T, S * n)
    mixes = (x @ hp.fn.T) / (x.square().mean(-1, keepdim=True).sqrt() + cfg.rms_eps)
    if hook:
        hook("hc_mixes", il, mixes)
    pre = torch.sigmoid(mixes[:, :S] * hp.scale[0] + hp.base[:S]) + cfg.hc_eps
    post = torch.sigmoid(mixes[:, S:2 * S] * hp.scale[1] + hp.base[S:2 * S]) * 2.0
    comb = (mixes[:, 2 * S:] * hp.scale[2] + hp.base[2 * S:]).view(T, S, S)
    comb = torch.softmax(comb, -1) + cfg.hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + cfg.hc_eps)
    for _ in range(cfg.sinkhorn - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + cfg.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + cfg.hc_eps)
    x_in = (pre.unsqueeze(-1) * R).sum(1)
    if hook:
        hook("hc_pre", il, pre)
        hook("hc_post", il, post)
        hook("hc_comb", il, comb)
    return x_in, post, comb


def mhc_post(y: torch.Tensor, R: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """R'[k] = post[k] y + sum_j comb[j][k] R[j]"""
    return post.unsqueeze(-1) * y.unsqueeze(1) + torch.einsum("tjk,tjh->tkh", comb, R)


def hc_head(R: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, cfg: Cfg, hook=None) -> torch.Tensor:
    """The streams' final reduce (vLLM's mainline kernel, tilelang_kernels.py:1021-1115): rsqrt(mean + eps) - vLLM-ERNIE's
    choice; SGLang's ERNIE port has 1 / (sqrt(mean) + eps) here (a ~1e-6 relative difference)"""
    T, S, n = R.shape
    x = R.reshape(T, S * n)
    mix = (x @ fn.T) * torch.rsqrt(x.square().mean(-1, keepdim=True) + cfg.rms_eps)
    if hook:
        hook("hc_head_mixes", -1, mix)
    pre = torch.sigmoid(mix * scale[0] + base) + cfg.hc_eps
    return (pre.unsqueeze(-1) * R).sum(1)


# ------------------------------------------------------------------------------------------------ the cache
@dataclass
class LayerCache:
    # HCA: every position's K = V row (the window reads the last 128), the compressor's per-position state, the
    # compressed rows (one per completed block of 128)
    kv: list = field(default_factory=list)           # [T, 512] chunks
    c_kv: list = field(default_factory=list)         # [T, 512] the compressor's kv state
    c_sc: list = field(default_factory=list)         # [T, 512] its score state (+ ape)
    comp: list = field(default_factory=list)         # [512] rows
    # DSA: the latent rows [kv_c 512 | k_pe 64] and the indexer's keys (FP8-rounded, Hadamard'd, RoPE'd)
    lat: list = field(default_factory=list)
    ikey: list = field(default_factory=list)

    @staticmethod
    def cat(lst: list, width: int) -> torch.Tensor:
        return torch.cat(lst, 0) if lst else torch.zeros(0, width)


@dataclass
class Cache:
    layers: list
    n: int = 0          # positions held
    capture: dict = field(default_factory=dict)   # DSpark: layer output (stream mean) of the last call, by capture index


# ------------------------------------------------------------------------------------------------ the model
class Model:
    def __init__(self, model_dir: str, index: str | None = None, verbose: bool = False):
        self.dir = pathlib.Path(model_dir)
        self.cfg = Cfg.load(self.dir / "config.json")
        self.w = Weights(model_dir, index)
        self.verbose = verbose
        c = self.cfg
        self.inv_hca = inv_freq_yarn(c.rope_dim, c.crope_theta, c.yarn_factor, c.yarn_orig, c.beta_fast, c.beta_slow)
        self.inv_plain = inv_freq_plain(c.mla_rope, c.rope_theta)
        self.inv_idx = inv_freq_plain(c.rope_dim, c.rope_theta)
        self.hooks: dict = {}       # name -> callable(name, layer, tensor): per-op dumps for the engine's checks
        self._lw: dict = {}         # per-layer dense weights in fp32 (filled on first use; clear() frees them)
        self._big: dict = {}        # the embedding and the head (BF16, 1.65 GB each), read once
        self.imatrix = None         # an Imatrix: the routed experts' input statistics, when collecting
        self.expert_src = None      # callable (layer, expert) -> (w1, w3, w2) F32: the routed experts from elsewhere (a GGUF)
        self.keep_dense = False     # keep every layer's dense weights between calls (the oracle's step-by-step runs)

    # -------------------------------------------------------------------------------- helpers
    def new_cache(self) -> Cache:
        return Cache(layers=[LayerCache() for _ in range(self.cfg.n_layer)])

    def _hook(self, name: str, il: int, t: torch.Tensor) -> None:
        f = self.hooks.get(name) or self.hooks.get("*")
        if f:
            f(name, il, t)

    def W(self, name: str) -> torch.Tensor:
        """a dense tensor in fp32, kept while its layer runs"""
        t = self._lw.get(name)
        if t is None:
            t = self.w.f32(name)
            self._lw[name] = t
        return t

    def clear(self) -> None:
        if not self.keep_dense:
            self._lw.clear()

    def expert(self, il: int, e: int, b: str):
        """expert e's (w1, w3, w2) [out, in] in F32: the release's exact values, or expert_src's"""
        if self.expert_src is not None:
            return self.expert_src(il, e)
        ep = f"{b}experts.{e}."
        return self.w.f32(ep + "w1.weight"), self.w.f32(ep + "w3.weight"), self.w.f32(ep + "w2.weight")

    def big(self, name: str) -> torch.Tensor:
        t = self._big.get(name)
        if t is None:
            t = self.w.raw(name)
            self._big[name] = t
        return t

    # -------------------------------------------------------------------------------- attention: HCA
    def attn_hca(self, il: int, x: torch.Tensor, pos: torch.Tensor, lc: LayerCache, pre: str) -> torch.Tensor:
        c = self.cfg
        T = x.shape[0]
        W = self.W
        a = pre + "self_attn."
        qr = rms(x @ W(a + "linear_q_down_proj.weight").T, W(a + "q_layernorm.weight"), c.rms_eps)
        self._hook("q_resid", il, qr)
        kv = rms(x @ W(a + "linear_kv_proj.weight").T, W(a + "kv_layernorm.weight"), c.rms_eps)
        g = qr @ W(a + "gate_proj.weight").T                                   # [T, 8192]
        q = (qr @ W(a + "linear_q_up_proj.weight").T).view(T, c.n_head, c.head_dim)
        q = rms(q, None, c.rms_eps)                                            # the weightless per-head norm
        nope = c.head_dim - c.rope_dim
        q = torch.cat([q[..., :nope], rope_gptj(q[..., nope:], pos, self.inv_hca)], -1)
        kv = torch.cat([kv[..., :nope], rope_gptj(kv[..., nope:], pos, self.inv_hca)], -1)
        self._hook("hca_q", il, q)
        self._hook("hca_kv", il, kv)
        has_comp = il < len(c.ratios) and c.ratios[il] > 1                    # (DSpark's stages: window only)
        if has_comp:   # the compressor's state (fp32 kv_score from the BF16 weights), and the blocks completed in this call
            cp = a + "core_attention.compressor."
            ks = x @ torch.cat([W(cp + "linear_wkv.weight"), W(cp + "linear_wgate.weight")], 0).T
            ape = W(cp + "ape")
            lc.c_kv.append(ks[:, :512])
            lc.c_sc.append(ks[:, 512:] + ape[pos % 128])
            ckv, csc = LayerCache.cat(lc.c_kv, 512), LayerCache.cat(lc.c_sc, 512)
            n_done = (int(pos[-1]) + 1) // 128
            while len(lc.comp) < n_done:
                k = len(lc.comp)
                wts = torch.softmax(csc[128 * k:128 * k + 128], 0)            # per channel, over the block's positions
                row = (ckv[128 * k:128 * k + 128] * wts).sum(0, keepdim=True)
                row = rms(row, W(cp + "norm.weight"), c.rms_eps)
                row = torch.cat([row[..., :nope], rope_gptj(row[..., nope:], torch.tensor([128 * k]), self.inv_hca)], -1)
                lc.comp.append(row)
                self._hook("hca_comp", il, row)
        lc.kv.append(kv)
        allkv = LayerCache.cat(lc.kv, c.head_dim)
        comp = LayerCache.cat(lc.comp, c.head_dim)
        sink = W(a + "core_attention.attn_sink")
        scale = c.head_dim ** -0.5
        o = torch.empty(T, c.n_head, c.head_dim)
        for t in range(T):
            p = int(pos[t])
            nc = (p + 1) // 128 if has_comp else 0
            rows = torch.cat([comp[:nc], allkv[max(0, p - c.window + 1):p + 1]], 0)    # K = V
            s = (q[t] @ rows.T) * scale                                       # [H, n]
            m = torch.maximum(s.amax(-1), sink)
            e = torch.exp(s - m[:, None])
            o[t] = (e @ rows) / (e.sum(-1) + torch.exp(sink - m))[:, None]
        self._hook("hca_o", il, o)
        o = self.vha(o, a)
        o = torch.cat([o[..., :nope], rope_gptj(o[..., nope:], pos, self.inv_hca, inverse=True)], -1)
        wa = W(a + "linear_o_group_proj").view(c.o_groups, c.o_lora, -1)       # [8, 1024, 8 x 512]
        z = torch.einsum("tgr,gdr->tgd", o.reshape(T, c.o_groups, -1), wa).reshape(T, -1)
        self._hook("hca_z", il, z)
        u = z * torch.sigmoid(g)
        return u @ W(a + "o_proj.weight").T

    def vha(self, o: torch.Tensor, a: str) -> torch.Tensor:
        """VHA, the low-rank post-mix across heads: o'[h] = sum_g (I + V U^T)[h, g] o[g] (vLLM-ERNIE attention.py:136-172;
        fp32 here: TP=1 serving rounds the [64, 64] matrix to BF16 once)"""
        U, V = self.W(a + "vha_postmix_U"), self.W(a + "vha_postmix_V")
        m = torch.eye(U.shape[0]) + V @ U.T
        return torch.einsum("hg,tgd->thd", m, o)

    # -------------------------------------------------------------------------------- attention: MLA + DSA
    def attn_dsa(self, il: int, x: torch.Tensor, pos: torch.Tensor, lc: LayerCache, pre: str) -> torch.Tensor:
        c = self.cfg
        T = x.shape[0]
        W = self.W
        a = pre + "self_attn."
        H, dn, dr, dv, L = c.mla_heads, c.mla_nope, c.mla_rope, c.mla_v, c.mla_kv_lora
        qc = rms(x @ W(a + "q_a_proj.weight").T, W(a + "q_a_layernorm.weight"), c.rms_eps)
        self._hook("q_resid", il, qc)
        kvl = x @ W(a + "kv_a_proj_with_mqa.weight").T
        g = qc @ W(a + "gate_proj.weight").T                                   # [T, 16384]
        q = (qc @ W(a + "q_b_proj.weight").T).view(T, H, dn + dr)
        kv_c = rms(kvl[:, :L], W(a + "kv_a_layernorm.weight"), c.rms_eps)
        k_pe = rope_gptj(kvl[:, L:].unsqueeze(1), pos, self.inv_plain).squeeze(1)
        q_pe = rope_gptj(q[..., dn:], pos, self.inv_plain)
        q_nope = q[..., :dn]
        self._hook("kv_cmpr", il, kv_c)
        self._hook("k_pe", il, k_pe)
        lc.lat.append(torch.cat([kv_c, k_pe], -1))
        # the lightning indexer: keys and queries rounded as the references round them
        ia = a + "core_attention.indexer."
        qi = (qc @ W(ia + "wq_b.weight").T).view(T, c.idx_heads, c.idx_dim)
        k = x @ W(ia + "wk.weight").T
        wts = x @ W(ia + "weights_proj.weight").T                              # [T, 64]
        k = torch.nn.functional.layer_norm(k, (c.idx_dim,), W(ia + "k_norm.weight"), W(ia + "k_norm.bias"), 1e-6)
        hr = c.rope_dim
        qi = torch.cat([rope_neox(qi[..., :hr], pos, self.inv_idx), qi[..., hr:]], -1)
        k = torch.cat([rope_neox(k[:, :hr].unsqueeze(1), pos, self.inv_idx).squeeze(1), k[:, hr:]], -1)
        qi, k = hadamard(qi), hadamard(k)
        if IDX_FP8:
            qi, k = fp8_fake(qi), fp8_fake(k)
        self._hook("idx_k", il, k)
        self._hook("idx_q", il, qi)
        lc.ikey.append(k)
        keys = LayerCache.cat(lc.ikey, c.idx_dim)
        lat = LayerCache.cat(lc.lat, L + dr)
        wscale = (c.idx_dim ** -0.5) * (c.idx_heads ** -0.5)
        self._hook("idx_w", il, wts * wscale)
        kvb = W(a + "kv_b_proj.weight").view(H, dn + dv, L)
        w_uk, w_uv = kvb[:, :dn], kvb[:, dn:]                                  # [H, 192, 512], [H, 256, 512]
        ql = torch.einsum("thd,hdl->thl", q_nope, w_uk)                       # the absorbed query [T, H, 512]
        qf = torch.cat([ql, q_pe], -1)
        sink = W(a + "core_attention.softmax_offset")
        scale = (dn + dr) ** -0.5
        out = torch.empty(T, H, dv)
        for t in range(T):
            p = int(pos[t])
            n_cand = max(p + 1 - c.window, 0)
            sel = torch.empty(0, dtype=torch.long)
            if n_cand > 0:
                if n_cand <= c.idx_topk:
                    sel = torch.arange(n_cand)
                else:
                    logit = (wts[t] * wscale) @ torch.relu(qi[t] @ keys[:n_cand].T)   # [n_cand]
                    sel = torch.topk(logit, c.idx_topk).indices.sort().values
                    self._hook("dsa_sel", il, sel)
            if self.hooks:   # the engine's record: the indexer's top_k entries for the token, -1 past the chosen ones
                rec = torch.full((c.idx_topk,), -1, dtype=torch.int32)
                rec[:sel.numel()] = sel.to(torch.int32)
                self._hook("dsa_sel_rec", il, rec)
            idx = torch.cat([sel, torch.arange(max(0, p - c.window + 1), p + 1)])
            rows = lat[idx]                                                    # [n, 576]
            s = (qf[t] @ rows.T) * scale
            m = torch.maximum(s.amax(-1), sink)
            e = torch.exp(s - m[:, None])
            o_lat = (e @ rows[:, :L]) / (e.sum(-1) + torch.exp(sink - m))[:, None]
            self._hook("kqv", il, o_lat)
            out[t] = torch.einsum("hl,hvl->hv", o_lat, w_uv)
        self._hook("dsa_o", il, out)
        o = self.vha(out, a).reshape(T, -1)
        return (o * torch.sigmoid(g)) @ W(a + "o_proj.weight").T

    # -------------------------------------------------------------------------------- FFN
    def ffn(self, il: int, x: torch.Tensor, pre: str) -> torch.Tensor:
        c = self.cfg
        W = self.W
        if il < c.first_dense and il < c.n_layer:
            m = pre + "mlp."
            return self._glu(il, x, W(m + "gate_proj.weight"), W(m + "up_proj.weight"), W(m + "down_proj.weight"))
        b = pre + "block_sparse_moe."
        T = x.shape[0]
        lg = x @ torch.cat([W(b + "gate.weight"), W(b + "gate.weight_1")], 0).T          # [T, 1024]
        self._hook("ffn_moe_logits", il, lg[:, :c.n_expert])
        sc = torch.sigmoid(lg[:, :c.n_expert]) + torch.sigmoid(lg[:, c.n_expert:])
        ids = torch.topk(sc + W(b + "e_score_correction_bias"), c.top_k, -1).indices
        wk = sc.gather(1, ids)
        wk = wk / wk.sum(-1, keepdim=True).clamp_min(1e-20)
        wk = wk * W(b + "gate.routed_scaling_factor_param")[ids]
        self._hook("moe_ids", il, ids)
        self._hook("moe_w", il, wk)
        self._hook("ffn_moe_topk", il, ids.to(torch.int32))
        self._hook("ffn_moe_weights_scaled", il, wk)
        lat = x @ W(b + "fc1_latent_proj.weight").T                                       # [T, 2048]
        self._hook("ffn_moe_lat", il, lat)   # (the experts' input: strata-x --expert-check reads it)
        yl = torch.zeros(T, c.latent)
        down = torch.zeros(T, c.top_k, c.latent) if self.hooks else None                  # (the engine's ffn_moe_down)
        for e in torch.unique(ids).tolist():
            tok, slot = (ids == e).nonzero(as_tuple=True)
            w1, w3, w2 = self.expert(il, e, b)
            xe = lat[tok]
            hg = xe @ w1.T
            hu = xe @ w3.T
            act = situ_glu(hg, hu, c.situ_beta, c.situ_lin_beta)
            if self.imatrix is not None:
                self.imatrix.add(il, e, xe, act)
            y = act @ w2.T
            if down is not None:
                down[tok, slot] = y
            yl.index_add_(0, tok, y * wk[tok, slot][:, None])
        if down is not None:
            self._hook("ffn_moe_down", il, down)
        self._hook("ffn_moe_out", il, yl)
        r = rms(yl, W(b + "latent_norm.weight"), c.rms_eps) @ W(b + "fc2_latent_proj.weight").T
        s = b + "shared_experts."
        sh = self._glu(il, x, W(s + "w1.weight"), W(s + "w3.weight"), W(s + "w2.weight"))
        self._hook("ffn_shexp", il, sh)
        return sh + r

    def _glu(self, il: int, x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor) -> torch.Tensor:
        """a dense SiTU-GLU FFN (layer 0, the shared experts), its products under the engine's swiglu_ffn names"""
        c = self.cfg
        u = x @ wu.T
        g = x @ wg.T
        self._hook("ffn_up", il, u)
        self._hook("ffn_gate", il, g)
        a = situ_glu(g, u, c.situ_beta, c.situ_lin_beta)
        self._hook("ffn_swiglu", il, a)
        return a @ wd.T

    # -------------------------------------------------------------------------------- the trunk
    def embed(self, ids: list[int] | torch.Tensor) -> torch.Tensor:
        """the tokens' rows broadcast to the 4 streams: [T, 4, 4096]"""
        ids = torch.as_tensor(ids, dtype=torch.long)
        emb = self.big("model.embed_tokens.weight")[ids].to(F32)
        return emb.unsqueeze(1).expand(ids.shape[0], self.cfg.hc, self.cfg.hidden).contiguous()

    def layer(self, il: int, R: torch.Tensor, pos: torch.Tensor, lc: LayerCache) -> torch.Tensor:
        """layer il over the streams R [T, 4, 4096] at positions pos, appending to its cache lc"""
        return self.ffn_half(il, self.attn_half(il, R, pos, lc))

    def attn_half(self, il: int, R: torch.Tensor, pos: torch.Tensor, lc: LayerCache) -> torch.Tensor:
        """layer il's attention half (mHC's read and write around it): one sequence's tokens"""
        c = self.cfg
        pre = f"model.layers.{il}."
        ha = hc_params(self.w, pre + "self_attention_hyper_connection")
        xa, post, comb = mhc_pre(R, ha, c, self._hook if self.hooks else None, il)
        self._hook("hc_attn_pre", il, xa)
        xa = rms(xa, self.W(pre + "input_layernorm.weight"), c.rms_eps)
        self._hook("attn_norm", il, xa)
        ya = self.attn_dsa(il, xa, pos, lc, pre) if c.is_dsa(il) else self.attn_hca(il, xa, pos, lc, pre)
        self._hook("attn_out", il, ya)
        R = mhc_post(ya, R, post, comb)
        self._hook("hc_attn_post", il, R)
        return R

    def ffn_half(self, il: int, R: torch.Tensor) -> torch.Tensor:
        """layer il's FFN half: every row on its own, so the rows of several sequences may run together"""
        c = self.cfg
        pre = f"model.layers.{il}."
        hf = hc_params(self.w, pre + "mlp_hyper_connection")
        xf, post, comb = mhc_pre(R, hf, c, self._hook if self.hooks else None, il)
        self._hook("hc_ffn_pre", il, xf)
        xf = rms(xf, self.W(pre + "post_attention_layernorm.weight"), c.rms_eps)
        self._hook("ffn_norm", il, xf)
        yf = self.ffn(il, xf, pre)
        self._hook("ffn_out", il, yf)
        R = mhc_post(yf, R, post, comb)
        self._hook("l_out", il, R)
        return R

    def forward(self, ids: list[int] | torch.Tensor, cache: Cache, logits: str = "last") -> torch.Tensor:
        """The trunk over `ids` at positions cache.n .. cache.n + T - 1, appending to the cache. logits: "last" (the last
        position's, [V]), "all" ([T, V]) or "none" (the hidden states only: returns the final [T, 4096])."""
        c = self.cfg
        ids = torch.as_tensor(ids, dtype=torch.long)
        T = ids.shape[0]
        pos = torch.arange(cache.n, cache.n + T)
        emb = self.big("model.embed_tokens.weight")[ids].to(F32)
        R = emb.unsqueeze(1).expand(T, c.hc, c.hidden).contiguous()                       # the 4 streams
        cache.capture = {}
        for il in range(c.n_layer):
            R = self.layer(il, R, pos, cache.layers[il])
            if il in c.dspark_targets:   # (DSpark reads layer il's output, the streams' mean)
                cache.capture[il] = R.mean(1)
            self.clear()
            if self.verbose:
                print(f"  layer {il} done", flush=True)
        cache.n += T
        h = self.final_norm(R)
        if logits == "none":
            return h
        if logits == "last":
            h = h[-1:]
        out = self.logits(h)
        return out[0] if logits == "last" else out

    def final_norm(self, R: torch.Tensor) -> torch.Tensor:
        """the streams' reduce and the final norm: [T, 4, 4096] -> [T, 4096]"""
        c = self.cfg
        h = hc_head(R, self.w.f32("model.mhc_contract.hc_head_fn").T.contiguous(), self.w.f32("model.mhc_contract.hc_head_base"),
                    self.w.f32("model.mhc_contract.hc_head_scale"), c, self._hook if self.hooks else None)
        self._hook("hc_head", -1, h)
        h = rms(h, self.w.f32("model.norm.weight"), c.rms_eps)
        self._hook("result_norm", -1, h)
        return h

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """the head and MultiMax: [T, 4096] -> [T, V] in fp32"""
        lm = self.big("lm_head.weight")
        out = torch.cat([h @ lm[i:i + 16384].to(F32).T for i in range(0, lm.shape[0], 16384)], -1)
        out = multimax(out, self.w.f32("model.lm_head.multimax_ranges"), self.w.f32("model.lm_head.multimax_ts"))
        self._hook("result_output", -1, out)
        return out


class Imatrix:
    """The routed experts' input statistics for the converter's quantization (requant_pack.cpp's DSV41IMX layout): per
    layer and expert, the sum of squared inputs per column for gate / up (the latent) and for down (the SiTU-GLU output),
    and the number of rows"""

    def __init__(self, n_layer: int, n_expert: int, d_in: int, d_ff: int):
        self.gu = torch.zeros(n_layer, n_expert, d_in, dtype=torch.float64)
        self.dn = torch.zeros(n_layer, n_expert, d_ff, dtype=torch.float64)
        self.cnt = torch.zeros(n_layer, n_expert, dtype=torch.int64)

    def add(self, il: int, e: int, xin: torch.Tensor, act: torch.Tensor) -> None:
        self.gu[il, e] += xin.double().square().sum(0)
        self.dn[il, e] += act.double().square().sum(0)
        self.cnt[il, e] += xin.shape[0]

    def save(self, path: str | pathlib.Path) -> None:
        """convert_gguf.cpp's --imatrix: "ERN55IMX", u32 version 1, u32 n_layer / n_expert / d_in / d_ff, then per layer per
        expert f64 gate_up[d_in], f64 down[d_ff], u64 count (written beside, then renamed)"""
        import struct
        L, E, di = self.gu.shape
        df = self.dn.shape[2]
        tmp = pathlib.Path(str(path) + ".tmp")
        with open(tmp, "wb") as f:
            f.write(b"ERN55IMX" + struct.pack("<5I", 1, L, E, di, df))
            for il in range(L):
                for e in range(E):
                    f.write(self.gu[il, e].numpy().astype("<f8").tobytes())
                    f.write(self.dn[il, e].numpy().astype("<f8").tobytes())
                    f.write(struct.pack("<Q", int(self.cnt[il, e])))
        tmp.replace(path)

    @staticmethod
    def load(path: str | pathlib.Path) -> "Imatrix":
        import struct
        b = pathlib.Path(path).read_bytes()
        assert b[:8] == b"ERN55IMX", path
        ver, L, E, di, df = struct.unpack("<5I", b[8:28])
        assert ver == 1
        m = Imatrix(L, E, di, df)
        o = 28
        for il in range(L):
            for e in range(E):
                m.gu[il, e] = torch.frombuffer(bytearray(b[o:o + 8 * di]), dtype=torch.float64)
                o += 8 * di
                m.dn[il, e] = torch.frombuffer(bytearray(b[o:o + 8 * df]), dtype=torch.float64)
                o += 8 * df
                m.cnt[il, e] = struct.unpack("<Q", b[o:o + 8])[0]
                o += 8
        return m
