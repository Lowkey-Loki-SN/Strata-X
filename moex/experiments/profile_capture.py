#!/usr/bin/env python3
"""Run ~/AI/exl3-profiles/util/moe_profile_build.py on today's exllamav3-next (v1.5.4).

The builder was written against exllamav3 1.4.8 + PR #315; its routing hook (BlockSparseMLP.routing_fn(bsz, cfg, z,
params)) and model_init's -mcs/-cs/-chunk_size still match v1.5.4, but it imports exllamav3.model.moe_profile
(model_fingerprint) for the .meta.json, a module v1.5.4 no longer has.  This stubs that module, then runs the builder
unchanged.  usage: profile_capture.py <moe_profile_build.py args...>
"""
import os as _os  # moex: paths from this checkout's place (moex/README.md, Paths)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
_AI = _os.environ.get("STRATA_AI_DIR") or _os.path.dirname(_REPO)
import runpy, sys, types, hashlib, json, os
BUILDER = _os.path.join(_AI, 'exl3-profiles/util/moe_profile_build.py')
stub = types.ModuleType("exllamav3.model.moe_profile")
def model_fingerprint(config, n_experts):
    fp = {"architecture": getattr(config, "architecture", "?"), "experts": n_experts,
          "layers": getattr(config, "num_hidden_layers", None), "hidden_size": getattr(config, "hidden_size", None)}
    fp["sha"] = hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return fp
stub.model_fingerprint = model_fingerprint
import exllamav3.model  # noqa: E402  (the real package first, then the stub under it)
sys.modules["exllamav3.model.moe_profile"] = stub
sys.argv = [BUILDER] + sys.argv[1:]
sys.path.insert(0, os.path.dirname(BUILDER))
runpy.run_path(BUILDER, run_name="__main__")
