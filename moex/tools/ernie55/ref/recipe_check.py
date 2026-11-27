"""moex/tools/ernie55/ref/recipe_check.py - a quantization recipe's routed experts against a GGUF set's, on real activations,
before the recipe is written as a set. The reference model (model.py, fp32, exact FP8) reads a short text; at chosen layers
the routed experts' combined output (before latent_norm) is recomputed with (a) the set's experts (ernie55_expert_dq), (b)
the candidate recipe's (convert_gguf_ernie55 --fakequant: the converter's own quantization, with the given importance
matrix), (c) / (d) the candidate's gate/up or down alone (the other exact) - each as relative error power against the
exact output. First, once: the fake-quant of the set's own recipe must equal the set's decoded experts bit for bit (skip
it with --no-bit-check when the candidate's importance matrix is not the set's).

    ~/AI/vLLM/.venv/bin/python -I moex/tools/ernie55/ref/recipe_check.py --gguf G.gguf --imatrix I.imx (--text F | --ids F)
        --cand GU_TYPE,D_TYPE [--cur GU_TYPE,D_TYPE] [--layers 12,22,41] [--tokens 512] [--threads N] [--no-bit-check]

Measured with it (2026-11-27, moex/docs/ERNIE55_PLAN.md section 10): ~8 minutes a layer at 512 tokens (12 threads, the
candidate's 512 experts quantized at each chosen layer). It measures one layer's output, not the model's: a recipe it
favours still wants a KLD of the written set.
"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
_MODELS = _os.environ.get("STRATA_MODELS") or _os.path.join(_AI, "Models")
import argparse, collections, pathlib, struct, subprocess, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model as M                     # noqa: E402
from oracle_dump import ExpertDQ      # noqa: E402
from tokenizers import Tokenizer      # noqa: E402

class FakeQ(ExpertDQ):
    def __init__(self, conv, model, imx, threads):
        self.p = subprocess.Popen([conv, '--model', model, '--imatrix', imx, '--fakequant', '--threads', str(threads)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.cache, self.keep, self.n_read = collections.OrderedDict(), 0, 0

    def get(self, il, e, tg, td):
        self.p.stdin.write(f'{il} {e} {tg} {td}\n'.encode())
        self.p.stdin.flush()
        nf, nl, ok, _ = struct.unpack('<4i', self._read(16))
        assert ok, (il, e)
        n = nf * nl * 4
        g = torch.frombuffer(bytearray(self._read(n)), dtype=torch.float32).view(nf, nl)
        u = torch.frombuffer(bytearray(self._read(n)), dtype=torch.float32).view(nf, nl)
        d = torch.frombuffer(bytearray(self._read(n)), dtype=torch.float32).view(nl, nf)
        return g, u, d

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', required=True)
    ap.add_argument('--imatrix', required=True)
    ap.add_argument('--text', default='')
    ap.add_argument('--ids', default='', help='token ids instead of --text: a file of comma-separated ids (one chunk a line: the first)')
    ap.add_argument('--no-bit-check', action='store_true', help='the candidate uses another importance matrix than the set')
    ap.add_argument('--tokens', type=int, default=512)
    ap.add_argument('--offset', type=int, default=20000)
    ap.add_argument('--layers', default='12,22,41')
    ap.add_argument('--cand', default='iq3_s,iq3_s', help='GU_TYPE,D_TYPE')
    ap.add_argument('--cur', default='iq3_xxs,iq4_xs', help="the set's own recipe (for the bit check)")
    ap.add_argument('--model', default=_os.path.join(_MODELS, 'HF/ERNIE-5.5-Flash-Preview-FP8'))
    ap.add_argument('--tool', default=_os.path.join(_REPO, 'build-dsv41-dev', 'ernie55_expert_dq'))
    ap.add_argument('--conv', default=_os.path.join(_REPO, 'build-dsv41-dev', 'convert_gguf_ernie55'))
    ap.add_argument('--threads', type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(str(pathlib.Path(a.model) / 'tokenizer.json'))
    if a.ids:
        ids = [int(x) for x in pathlib.Path(a.ids).read_text().splitlines()[0].split(',') if x.strip()][:a.tokens]
    else:
        ids = [2] + tok.encode(pathlib.Path(a.text).read_text(errors='replace'), add_special_tokens=False).ids[a.offset:a.offset + a.tokens - 1]
    want = {int(x) for x in a.layers.split(',')}
    cg, cd = a.cand.split(',')
    ug, ud = a.cur.split(',')
    m = M.Model(a.model)
    c = m.cfg
    dq = ExpertDQ(pathlib.Path(a.tool), a.gguf, keep=0)
    fq = FakeQ(a.conv, a.model, a.imatrix, max(1, a.threads))
    cap = {}
    m.hooks = {n: (lambda name, il, t: cap.__setitem__(name, t.clone()) if il in want else None)
               for n in ('ffn_moe_lat', 'moe_ids', 'moe_w', 'ffn_moe_out')}
    t0 = time.time()
    checked = a.no_bit_check
    with torch.inference_mode():
        R = m.embed(ids)
        pos = torch.arange(len(ids))
        for il in range(c.n_layer):
            cap.clear()
            R = m.attn_half(il, R, pos, M.LayerCache())
            R = m.ffn_half(il, R)
            if il in want and 'ffn_moe_lat' in cap:
                lat, eids, wk, yref = cap['ffn_moe_lat'], cap['moe_ids'], cap['moe_w'], cap['ffn_moe_out']
                b = f'model.layers.{il}.block_sparse_moe.'
                keys = ('set', 'cand', 'cand_gu', 'cand_d', 'exact')
                outs = {k: torch.zeros_like(yref) for k in keys}
                for e in torch.unique(eids).tolist():
                    tk, slot = (eids == e).nonzero(as_tuple=True)
                    xe = lat[tk]
                    w1, w3, w2 = m.w.f32(b + f'experts.{e}.w1.weight'), m.w.f32(b + f'experts.{e}.w3.weight'), m.w.f32(b + f'experts.{e}.w2.weight')
                    s1, s3, s2 = dq(il, e)
                    if not checked:   # the fake-quant of the set's own recipe = the set's experts, bit for bit
                        f1, f3, f2 = fq.get(il, e, ug, ud)
                        same = [bool(torch.equal(x, y)) for x, y in ((f1, s1), (f3, s3), (f2, s2))]
                        print(f'bit check, layer {il} expert {e}: gate/up/down identical {same}', flush=True)
                        assert all(same), 'the fake-quant is not the set\'s quantization'
                        checked = True
                    k1, k3, k2 = fq.get(il, e, cg, cd)
                    wts = wk[tk, slot][:, None]
                    for k, (g, u, d) in {'exact': (w1, w3, w2), 'set': (s1, s3, s2), 'cand': (k1, k3, k2),
                                         'cand_gu': (k1, k3, w2), 'cand_d': (w1, w3, k2)}.items():
                        act = M.situ_glu(xe @ g.T, xe @ u.T, c.situ_beta, c.situ_lin_beta)
                        outs[k].index_add_(0, tk, (act @ d.T) * wts)
                ref = outs['exact']
                den = ref.double().square().sum()
                r = {k: float((outs[k] - ref).double().square().sum() / den) * 100 for k in keys[:-1]}
                print(f'layer {il}: MoE output rel err power: the set ({a.cur}) {r["set"]:.3f}% | candidate ({a.cand}) {r["cand"]:.3f}% '
                      f'({100 * (r["cand"] / r["set"] - 1):+.1f}%) | its gate/up alone {r["cand_gu"]:.3f}%, its down alone {r["cand_d"]:.3f}% '
                      f'({time.time() - t0:.0f} s)', flush=True)
            m.clear()
    fq.p.stdin.write(b'q\n'); fq.p.stdin.flush()
    dq.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
