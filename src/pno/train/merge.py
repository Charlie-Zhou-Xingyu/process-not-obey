"""Merge a LoRA adapter into the base model and export it in the base checkpoint's own layout.

We train on ``Qwen3_5ForCausalLM`` (text-only view of the Qwen3.5 checkpoint), whose weight
names are ``model.layers.*``; the released checkpoint is a ``Qwen3_5ForConditionalGeneration``
with ``model.language_model.layers.*`` plus vision and MTP weights. To give vLLM a drop-in
replacement for the base model, we copy every original tensor and overwrite only those the
adapter changed, keeping the original file names, config and index. The tokenizer's chat
template is patched to accept the ``input`` role (see ``pno.formatting``).

    python -m pno.train.merge --base Qwen/Qwen3.5-2B --adapter runs/C3/seed0 --out merged/C3_seed0
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from pno.formatting import qwen_template_with_input_role

WEIGHT_SUFFIXES = (".safetensors",)


def resolve_local(base: str) -> Path:
    p = Path(base)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(base))


def patch_chat_template(out_dir: Path) -> None:
    tc_path = out_dir / "tokenizer_config.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        if isinstance(tc.get("chat_template"), str):
            tc["chat_template"] = qwen_template_with_input_role(tc["chat_template"])
            tc_path.write_text(json.dumps(tc, ensure_ascii=False, indent=2))
    jinja = out_dir / "chat_template.jinja"
    if jinja.exists():
        jinja.write_text(qwen_template_with_input_role(jinja.read_text()))


def export_merged(base: str, adapter: str, out: str, check_tied: bool = True) -> dict:
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    base_dir = resolve_local(base)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    merged = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    orig: dict[str, tuple[str, "torch.Tensor"]] = {}
    for f in sorted(base_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():
                orig[k] = (f.name, sf.get_tensor(k))

    def to_orig(k: str) -> str | None:
        for c in (k, k.replace("model.", "model.language_model.", 1)):
            if c in orig:
                return c
        return None

    replaced, unchanged, skipped = [], 0, []
    for k, v in merged.items():
        ok = to_orig(k)
        if ok is None:
            skipped.append(k)  # e.g. lm_head.weight when tied to the embeddings
            continue
        fname, t = orig[ok]
        if tuple(t.shape) != tuple(v.shape):
            raise ValueError(f"shape mismatch for {k} -> {ok}: {tuple(v.shape)} vs {tuple(t.shape)}")
        v = v.to(t.dtype).contiguous()
        if torch.equal(t, v):
            unchanged += 1
        else:
            orig[ok] = (fname, v)
            replaced.append(ok)
    bad_skips = [k for k in skipped if k != "lm_head.weight"]
    if bad_skips:
        raise KeyError(f"merged weights with no counterpart in the base checkpoint: {bad_skips[:5]}")

    by_file: dict[str, dict] = defaultdict(dict)
    for k, (fname, t) in orig.items():
        by_file[fname][k] = t
    for fname, tensors in by_file.items():
        save_file(tensors, str(out_dir / fname), metadata={"format": "pt"})
    for f in base_dir.iterdir():
        if f.is_file() and not f.name.endswith(WEIGHT_SUFFIXES):
            shutil.copy(f, out_dir / f.name)
    patch_chat_template(out_dir)
    info = {"base": str(base), "adapter": str(adapter), "replaced": len(replaced), "unchanged": unchanged,
            "skipped": skipped, "examples": replaced[:5]}
    (out_dir / "pno_merge_info.json").write_text(json.dumps(info, indent=2))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(export_merged(a.base, a.adapter, a.out), indent=1))


if __name__ == "__main__":
    main()
