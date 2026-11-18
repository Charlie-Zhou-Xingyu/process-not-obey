"""Shared plumbing for benchmarks.

A benchmark module exposes::

    NAME: str
    load(cfg) -> list[dict]                 # items: {"id", "prompt": {...}, ...scoring fields}
    score(items, outputs, cfg) -> list[dict] # one dict of numeric per-item scores per item
    summarize(per_item) -> dict              # headline numbers

Generation and scoring are separate steps so that a scorer needing its own model (an
embedding model or a judge) can run after the GPU is free.
"""
from __future__ import annotations

from pathlib import Path

from pno.formatting import Prompt, render_prompt
from pno.io import read_jsonl, write_json, write_jsonl


def render_items(items: list[dict], renderer: str, tokenizer=None, placement: str = "input_role") -> list[str]:
    return [render_prompt(Prompt.from_dict(it["prompt"]), renderer=renderer, tokenizer=tokenizer,
                          placement=placement) for it in items]


def save_outputs(path: Path, items: list[dict], outputs: list[dict]) -> None:
    """``outputs`` come from ``generate_full``: ``{"text", "truncated"}`` per item."""
    write_jsonl(path, ({**it, "output": o["text"], "truncated": o["truncated"]} for it, o in zip(items, outputs)))


def load_outputs(path: Path) -> tuple[list[dict], list[str]]:
    rows = read_jsonl(path)
    return rows, [r["output"] for r in rows]


def mean(xs) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float("nan")


def save_scores(out_dir: Path, name: str, per_item: list[dict], summary: dict) -> None:
    write_jsonl(out_dir / f"{name}_scores.jsonl", per_item)
    write_json(out_dir / f"{name}_summary.json", summary)
