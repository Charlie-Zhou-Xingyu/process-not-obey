"""Day-0 check of the generation backends on the GPU server.

The vLLM path cannot be tested on a laptop, and whether vLLM can apply a LoRA adapter to
Qwen3.5 at run time (the way Meta-SecAlign evaluates its adapters) is not known in advance.
This script answers both by generating the same 24 short prompts with each setup, each in
its own process, and comparing the texts (with ``--adapter``, 16 of that run's training prompts
are added, since that is where its outputs should differ from the base model's):

    hf_base      transformers, base model                      (reference)
    vllm_base    vLLM, base model                              -> does vLLM agree with transformers?
    hf_adapter   transformers, base + adapter      (--adapter) (reference for the trained run)
    vllm_lora    vLLM, base + run-time LoRA        (--adapter) -> eval.adapter_mode: lora
    vllm_merged  vLLM, merged float32 checkpoint   (--adapter) -> eval.adapter_mode: merge

    python -m pno.tools.check_backend                                   # base model only
    python -m pno.tools.check_backend --adapter runs/C1_ignore_ordinary-seed0

Greedy decoding in two engines is rarely token-identical for every prompt, so the report
gives agreement rates and judges each vLLM setup against the vLLM-vs-transformers
agreement of the base model. A setup that fails to start is reported, not fatal.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pno.catalog import build_registry
from pno.catalog.base import _FALLBACK_CARRIERS
from pno.formatting import Prompt
from pno.io import REPO_ROOT, load_config, read_jsonl, write_json, write_jsonl

TRANSFORMS = ("T01_copy", "T03_upper", "T04_lower", "number_sentences", "sentences_per_line", "T24_space_to_underscore")
PLAIN = ("Name three primary colors.", "Give one tip for learning a new language.",
         "What is the capital of France?", "Write one sentence about the sea.",
         "List two uses of a hammer.", "Explain in one sentence why the sky is blue.")


def training_prompts(adapter: str, n: int = 16) -> list[dict]:
    """Prompts the adapter was trained on: there its outputs should differ from the base model's,
    which is what lets us see whether an engine really applied the adapter."""
    log = Path(adapter) / "train_log.json"
    if not log.exists():
        return []
    train_file = Path(json.loads(log.read_text())["args"]["train"])
    if not train_file.exists():
        return []
    rows = read_jsonl(train_file)[:n]
    return [{"key": f"train-{i}", "prompt": r["prompt"]} for i, r in enumerate(rows)]


def build_requests(adapter: str | None = None) -> list[dict]:
    reg = build_registry()
    reqs = training_prompts(adapter) if adapter else []
    for i, name in enumerate(TRANSFORMS):
        t = reg[name]
        for j in range(3):
            data = " ".join(_FALLBACK_CARRIERS[(i + j) % 6:(i + j) % 6 + 2] or _FALLBACK_CARRIERS[:2])
            if j == 2:   # one conflict per transform: an injected instruction aimed at the user message
                data += " " + reg["T03_upper" if name != "T03_upper" else "T04_lower"].instruction("user")
            reqs.append({"key": f"{name}-{j}", "prompt": Prompt(t.instruction("input"), data).to_dict()})
    reqs += [{"key": f"plain-{k}", "prompt": Prompt(q).to_dict()} for k, q in enumerate(PLAIN)]
    return reqs


def run_setup(name: str, py: str, req_file: Path, out_dir: Path, model: str, backend: str,
              adapter: str | None, dtype: str | None, max_new_tokens: int) -> dict | None:
    out = out_dir / f"{name}.jsonl"
    cmd = [py, "-m", "pno.gen.run", "--requests", str(req_file), "--model", model, "--out", str(out),
           "--backend", backend, "--max_new_tokens", str(max_new_tokens)]
    if adapter:
        cmd += ["--adapter", adapter]
    if dtype:
        cmd += ["--dtype", dtype]
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-6:])
        print(f"[check] {name} FAILED:\n{tail}")
        return None
    return {r["key"]: r["response"].strip() for r in read_jsonl(out)}


def agreement(a: dict | None, b: dict | None) -> float | None:
    if a is None or b is None:
        return None
    return sum(a[k] == b[k] for k in a) / len(a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None, help="a trained run directory (runs/<condition>-seed<k>)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs/check_backend")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--skip_vllm", action="store_true", help="only the transformers setups (laptop)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    base = cfg["base_model"]
    py = os.environ.get("PNO_PYTHON", sys.executable)
    gen_py = os.environ.get("PNO_GEN_PYTHON", sys.executable)
    out_dir = Path(a.out) if Path(a.out).is_absolute() else REPO_ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    req_file = out_dir / "requests.jsonl"
    write_jsonl(req_file, build_requests(a.adapter))

    res: dict[str, dict | None] = {}
    res["hf_base"] = run_setup("hf_base", py, req_file, out_dir, base, "hf", None, None, a.max_new_tokens)
    if not a.skip_vllm:
        res["vllm_base"] = run_setup("vllm_base", gen_py, req_file, out_dir, base, "vllm", None, None, a.max_new_tokens)
    if a.adapter:
        res["hf_adapter"] = run_setup("hf_adapter", py, req_file, out_dir, base, "hf", a.adapter, None, a.max_new_tokens)
        if not a.skip_vllm:
            res["vllm_lora"] = run_setup("vllm_lora", gen_py, req_file, out_dir, base, "vllm", a.adapter, None,
                                         a.max_new_tokens)
            with tempfile.TemporaryDirectory(dir=out_dir) as td:
                merged = Path(td) / "merged"
                m = subprocess.run([py, "-m", "pno.train.merge", "--base", base, "--adapter", a.adapter,
                                    "--out", str(merged)], cwd=REPO_ROOT, capture_output=True, text=True)
                if m.returncode == 0:
                    res["vllm_merged"] = run_setup("vllm_merged", gen_py, req_file, out_dir, str(merged), "vllm",
                                                   None, cfg["eval"].get("dtype_trained"), a.max_new_tokens)
                else:
                    print("[check] merge FAILED:\n" + "\n".join(m.stderr.strip().splitlines()[-6:]))
                    res["vllm_merged"] = None

    report = {
        "n_prompts": len(read_jsonl(req_file)),
        "ran": {k: v is not None for k, v in res.items()},
        "vllm_base_vs_hf_base": agreement(res.get("vllm_base"), res.get("hf_base")),
        "hf_adapter_vs_hf_base": agreement(res.get("hf_adapter"), res.get("hf_base")),
        "vllm_lora_vs_hf_adapter": agreement(res.get("vllm_lora"), res.get("hf_adapter")),
        "vllm_merged_vs_hf_adapter": agreement(res.get("vllm_merged"), res.get("hf_adapter")),
        "vllm_lora_vs_vllm_base": agreement(res.get("vllm_lora"), res.get("vllm_base")),
    }
    ref = report["vllm_base_vs_hf_base"]
    notes = []
    if not a.skip_vllm:
        if ref is None:
            notes.append("vLLM could not run the base model: use PNO_BACKEND=hf.")
        elif ref < 0.7:
            notes.append(f"vLLM and transformers agree on only {ref:.0%} of base outputs: inspect the outputs "
                         "(prompt rendering, dtype) before trusting vLLM.")
        else:
            notes.append(f"vLLM runs the base model and agrees with transformers on {ref:.0%} of outputs.")
    if a.adapter and not a.skip_vllm and ref is not None:
        if report["hf_adapter_vs_hf_base"] is not None and report["hf_adapter_vs_hf_base"] > 0.9:
            notes.append("The adapter barely changes these outputs, so run-time LoRA cannot be verified here; "
                         "the merged checkpoint is used unless a later check says otherwise.")
        for mode, key in (("lora", "vllm_lora_vs_hf_adapter"), ("merge", "vllm_merged_vs_hf_adapter")):
            v = report[key]
            if v is None:
                notes.append(f"eval.adapter_mode: {mode} does not work on this server.")
            elif v >= ref - 0.15:
                notes.append(f"eval.adapter_mode: {mode} is usable ({v:.0%} agreement with transformers + adapter).")
            else:
                notes.append(f"eval.adapter_mode: {mode} disagrees with transformers + adapter ({v:.0%} vs {ref:.0%} "
                             "for the base model): do not use it.")
    # what `eval.adapter_mode: auto` will use: run-time LoRA if it reproduces transformers + adapter
    # about as well as vLLM reproduces transformers on the base model, else the merged checkpoint
    # Run-time LoRA is trusted only if the adapter visibly changes outputs here; otherwise an
    # engine that silently ignored it would look just as good. The merged checkpoint is checked
    # against the adapter inside pno.train.merge, so it is the safe choice when in doubt.
    report["recommended_adapter_mode"] = None
    if ref is not None and a.adapter:
        effect, lora, merged = (report[k] for k in ("hf_adapter_vs_hf_base", "vllm_lora_vs_hf_adapter",
                                                     "vllm_merged_vs_hf_adapter"))
        if lora is not None and lora >= ref - 0.15 and effect is not None and effect <= 0.9:
            report["recommended_adapter_mode"] = "lora"
        elif merged is not None and merged >= ref - 0.15:
            report["recommended_adapter_mode"] = "merge"
    report["notes"] = notes
    write_json(out_dir / "report.json", report)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
