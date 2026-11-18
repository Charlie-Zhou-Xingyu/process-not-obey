"""Evaluate one model on a set of benchmarks: one model load, greedy decoding, then scoring.

    python -m pno.eval.run --name C3_process_ordinary-seed0 --model merged/C3_process_ordinary-seed0 \
        --benchmarks secfid sep transfer ifeval conflict

Outputs under ``<out>/<name>/``: ``<bench>.jsonl`` (items + raw outputs),
``<bench>_scores.jsonl`` (per-item scores) and ``<bench>_summary.json``. Generation is
skipped for benchmarks whose outputs already exist, so the script is restartable;
``--score_only`` rescoring never loads the model.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from pno.eval.common import load_outputs, render_items, save_outputs, save_scores
from pno.io import load_config

BENCHMARKS = ("secfid", "secfid_clean", "sep", "transfer", "ifeval", "conflict")


def get_benchmark(name: str):
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; choose from {BENCHMARKS}")
    return importlib.import_module(f"pno.eval.{name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="model name used for the output directory")
    ap.add_argument("--model", help="HF id or local path (not needed with --score_only)")
    ap.add_argument("--adapter", default=None, help="hf backend only: LoRA adapter to load on top of --model")
    ap.add_argument("--tokenizer", default=None,
                    help="render with this tokenizer's chat template instead of the model's (renderer=hf)")
    ap.add_argument("--renderer", default="chatml", choices=["chatml", "hf"])
    ap.add_argument("--placement", default="input_role", choices=["input_role", "user_concat"])
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--backend", default=None, choices=["vllm", "hf"])
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16"],
                    help="weights dtype for generation (merged runs store their update in float32)")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N items per benchmark (smoke tests)")
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--no_score", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8, help="hf backend only")
    ap.add_argument("--device", default=None, help="hf backend only")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    ecfg = cfg["eval"]
    out_dir = Path(a.out) / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    benches = {b: get_benchmark(b) for b in a.benchmarks}

    todo = {}
    if not a.score_only:
        for b, mod in benches.items():
            if (out_dir / f"{b}.jsonl").exists():
                print(f"[eval] {a.name}/{b}: outputs exist, skipping generation")
                continue
            items = mod.load(cfg)
            todo[b] = items[: a.limit] if a.limit else items
    if todo:
        if not a.model:
            raise SystemExit("--model is required to generate")
        from pno.gen.backends import GenConfig, make_generator

        backend = a.backend or ecfg["backend"]
        if backend == "hf":
            gen = make_generator("hf", a.model, adapter_path=a.adapter, batch_size=a.batch_size, device=a.device,
                                 dtype=a.dtype)
        else:
            gen = make_generator("vllm", a.model, max_model_len=ecfg["max_model_len"], tokenizer_path=a.tokenizer,
                                 dtype=a.dtype or "bfloat16")
        for b, items in todo.items():
            tok = gen.tok
            if a.tokenizer and a.renderer == "hf" and backend == "hf":
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(a.tokenizer)
            prompts = render_items(items, a.renderer, tok, a.placement)
            max_new = ecfg.get(b, ecfg["secfid"] if b == "secfid_clean" else {})["max_new_tokens"]
            outs = gen.generate_full(prompts, GenConfig(max_new_tokens=max_new))
            save_outputs(out_dir / f"{b}.jsonl", items, outs)
            print(f"[eval] {a.name}/{b}: generated {len(outs)} outputs "
                  f"({sum(o['truncated'] for o in outs)} hit the {max_new}-token limit)")

    if a.no_score:
        return
    for b, mod in benches.items():
        path = out_dir / f"{b}.jsonl"
        if not path.exists():
            print(f"[eval] {a.name}/{b}: no outputs to score")
            continue
        items, outs = load_outputs(path)
        per_item = mod.score(items, outs, cfg)
        summary = {"model": a.name, "benchmark": b, **mod.summarize(per_item),
                   "truncated_outputs": sum(bool(it.get("truncated")) for it in items)}
        save_scores(out_dir, b, per_item, summary)
        print(f"[eval] {a.name}/{b}: " + json.dumps(summary))


if __name__ == "__main__":
    main()
