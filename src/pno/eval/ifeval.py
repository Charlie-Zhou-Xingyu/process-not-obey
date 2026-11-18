"""IFEval [10]: instruction following without any injection (utility check).

Prompts come from ``google/IFEval`` (541 items) and are scored with the reference
checkers shipped in lm-eval (``lm_eval.tasks.ifeval.utils``). We generate with our own
renderer so that non-thinking mode and the chat format match every other evaluation.
Headline number: prompt-level strict accuracy; the per-item 0/1 feeds the bootstrap.
"""
from __future__ import annotations

from pno.eval.common import mean
from pno.formatting import Prompt

NAME = "ifeval"
HF_ID = "google/IFEval"


def load(cfg: dict | None = None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(HF_ID, split="train")
    items = []
    for r in ds:
        items.append({"id": f"ifeval-{r['key']}", "prompt": Prompt(r["prompt"]).to_dict(),
                      "doc": {"key": r["key"], "prompt": r["prompt"],
                              "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]}})
    return items


def score(items: list[dict], outputs: list[str], cfg: dict | None = None) -> list[dict]:
    from lm_eval.tasks.ifeval import utils

    per_item = []
    for it, out in zip(items, outputs):
        r = utils.process_results(it["doc"], [out])
        per_item.append({
            "id": it["id"],
            "prompt_strict": float(r["prompt_level_strict_acc"]),
            "prompt_loose": float(r["prompt_level_loose_acc"]),
            "inst_strict": [float(x) for x in r["inst_level_strict_acc"]],
            "inst_loose": [float(x) for x in r["inst_level_loose_acc"]],
        })
    return per_item


def summarize(per_item: list[dict]) -> dict:
    flat = lambda k: [x for p in per_item for x in p[k]]  # noqa: E731
    out = {"n": len(per_item),
           "prompt_strict": mean(p["prompt_strict"] for p in per_item),
           "prompt_loose": mean(p["prompt_loose"] for p in per_item),
           "inst_strict": mean(flat("inst_strict")),
           "inst_loose": mean(flat("inst_loose"))}
    out["mean4"] = mean(out[k] for k in ("prompt_strict", "prompt_loose", "inst_strict", "inst_loose"))  # as in [4]
    return out


PRIMARY = "prompt_strict"
