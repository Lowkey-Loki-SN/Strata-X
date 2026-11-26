#!/usr/bin/env python3
"""moex/tools/convert_mimo_mmproj.py - MiMo-V2.6-Flash's vision tower (BF16 safetensors) -> a mimovl mmproj GGUF, with
mainline llama.cpp's own converter code (conversion/mimo.py MiMoV2VisionAudioModel), vision only.

    <python with torch + safetensors> -I -B convert_mimo_mmproj.py --mainline LLAMA_CPP_TREE --config CONFIG.json \\
        --preprocessor PREPROCESSOR.json --tower vision-tower.safetensors --out OUT.gguf [--outtype bf16|f16|f32]

Why a script: mainline's class converts MiMo's vision AND audio encoders into one file and needs the audio tokenizer
(audio_tokenizer/config.json + model.safetensors) and the full HF repo; we have the vision tower only (the BF16 `visual.*`
tensors, byte-copied from benthecarman/MiMo-V2.6-Flash-RL-exl3) and the HF config (vision_config: vit_window_attn_types,
visual_token_window_size, use_sink, fullatt_block_indexes ... - the keys mtmd's mimovl loader requires).  The subclass below
keeps every VISION line of the class unchanged (tensor mapping, the Conv3D split, the window pattern, the pixel budget)
and only leaves the audio encoder out: has_audio_encoder = False, no audio keys, no audio tensors.  The mainline tree is
only read (-B: no .pyc written into it); its gguf-py comes first on sys.path so the writer matches the converter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mainline", required=True, help="mainline llama.cpp tree (read-only)")
    ap.add_argument("--config", required=True, help="the model's HF config.json (MiMoV2ForCausalLM, with vision_config)")
    ap.add_argument("--preprocessor", required=True, help="preprocessor_config.json of the vision tower")
    ap.add_argument("--tower", required=True, help="the vision tower's safetensors (visual.* tensors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--outtype", default="bf16", choices=["bf16", "f16", "f32"])
    ap.add_argument("--name", default="MiMo V2.6 Flash", help="general.name (else the converter names it after the temp dir)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        print(f"{out} exists: refusing to overwrite", file=sys.stderr)
        return 2
    tree = Path(a.mainline).resolve()
    sys.path.insert(0, str(tree / "gguf-py"))
    sys.path.insert(0, str(tree))
    import gguf  # noqa: E402 - mainline's gguf-py
    from conversion.base import MmprojModel  # noqa: E402
    from conversion.mimo import MiMoV2VisionAudioModel  # noqa: E402

    class MiMoV2VisionOnly(MiMoV2VisionAudioModel):
        has_audio_encoder = False

        def get_audio_config(self):
            return None

        def set_gguf_parameters(self):
            MmprojModel.set_gguf_parameters(self)
            # MiMoV2VisionAudioModel.set_gguf_parameters' vision half, line for line (conversion/mimo.py)
            self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.MIMOVL)
            self.gguf_writer.add_vision_use_silu(True)
            self.gguf_writer.add_vision_head_count_kv(self.num_kv_heads)
            self.gguf_writer.add_vision_spatial_merge_size(self.spatial_merge_size)
            self.gguf_writer.add_uint32(gguf.Keys.ClipVision.WINDOW_SIZE, self.visual_token_window_size)
            self.gguf_writer.add_vision_wa_pattern_mode(self.vit_window_attn_types)
            self.gguf_writer.add_vision_attention_layernorm_eps(self.rms_norm_eps)
            self.gguf_writer.add_vision_min_pixels(int(self.preprocessor_config["min_pixels"]))
            self.gguf_writer.add_vision_max_pixels(int(self.preprocessor_config["max_pixels"]))

        def generate_extra_tensors(self):
            return iter(())   # the audio tokenizer's encoder (the class's only extra tensors)

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    if cfg.get("architectures") != ["MiMoV2ForCausalLM"] or "vision_config" not in cfg:
        raise SystemExit(f"{a.config}: not a MiMoV2ForCausalLM config with a vision_config")
    ftype = {"bf16": gguf.LlamaFileType.MOSTLY_BF16, "f16": gguf.LlamaFileType.MOSTLY_F16,
             "f32": gguf.LlamaFileType.ALL_F32}[a.outtype]
    with tempfile.TemporaryDirectory(prefix="mimo-mmproj-") as d:
        d = Path(d)
        (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (d / "preprocessor_config.json").write_text(Path(a.preprocessor).read_text(encoding="utf-8"), encoding="utf-8")
        os.symlink(Path(a.tower).resolve(), d / "model.safetensors")
        tmp_out = out.with_name(out.name + ".part")
        model = MiMoV2VisionOnly(d, ftype, tmp_out, eager=False, hparams=None, model_name=a.name)
        model.write()
        written = model.fname_out
        os.rename(written, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
