"""Alpaca-cleaned loading and the disjoint pools every builder draws from.

All data builders take items from fixed, disjoint pools so that, for example, the
instructions injected in ignore-style pairs never also appear as benign tasks, and carrier
sentences never come from the same rows as host tasks. Pools depend only on the pool seed
(``data.pool_seed`` in the config), not on the training seed.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from pno.io import read_jsonl, write_jsonl

ALPACA_HF_ID = "yahma/alpaca-cleaned"

_ASCII_OK = re.compile(r"^[A-Za-z0-9 ,.;:'\"!?()\-%$&/]+$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_ABBREV_END = re.compile(r"\b(dr|mr|mrs|ms|prof|st|vs|etc|inc|ltd|jr|sr|no|fig|e\.g|i\.e|u\.s|a\.m|p\.m|[a-z])\.$", re.I)


def one_sentence(s: str) -> bool:
    """True if the sentence transforms (pno.catalog.base.split_sentences) see exactly one
    sentence and it does not end in an abbreviation (a fragment such as "Call Dr.")."""
    from pno.catalog.base import split_sentences

    return split_sentences(s) == [s] and not _ABBREV_END.search(s)


_ASSISTANT_TALK = re.compile(r"\b(AI|language model|assistant|I'm sorry|I am sorry|I cannot|I can't|I am unable|"
                             r"I'm unable|I apologize)\b")

# Leading words that make a sentence read as an instruction. Carrier sentences must not be
# instruction-like, otherwise the "no conflict" competence check is not conflict-free.
IMPERATIVE_STARTS = set("""
add analyze answer apply arrange ask avoid be bring build calculate call change check choose classify
come compare compose compute consider convert copy count create cut decide define delete describe design
determine develop do don't draw edit eliminate enter estimate evaluate explain extract fill find fix
follow forget format generate get give go help identify ignore imagine include insert keep let list
look make mark name note open organize output paraphrase pick place plan please predict prepare print
provide put rank read recommend remember remove rename reorder repeat rephrase replace reply respond
return reverse rewrite say select send separate show sort spell start state stop suggest summarize take
tell think transform translate try turn type underline use verify visit wait write
""".split())


@dataclass
class AlpacaRow:
    idx: int
    instruction: str
    input: str
    output: str


def load_alpaca(cache: str | Path | None = None) -> list[AlpacaRow]:
    """Load alpaca-cleaned from a local JSONL cache, or from the HF hub (then cache it)."""
    if cache and Path(cache).exists():
        rows = read_jsonl(cache)
    else:
        from datasets import load_dataset

        ds = load_dataset(ALPACA_HF_ID, split="train")
        rows = [dict(r) for r in ds]
        if cache:
            write_jsonl(cache, rows)
    return [AlpacaRow(i, r["instruction"].strip(), (r.get("input") or "").strip(), r["output"].strip())
            for i, r in enumerate(rows)]


def is_instruction_like(sentence: str) -> bool:
    words = sentence.strip().split()
    if not words:
        return False
    first = words[0].lower().strip(",.:;!?\"'")
    return first in IMPERATIVE_STARTS or sentence.rstrip().endswith("?")


def carrier_sentences(rows: list[AlpacaRow], min_words: int = 6, max_words: int = 18) -> list[str]:
    """Plain declarative English sentences taken from Alpaca outputs (prose lines only)."""
    out, seen = [], set()
    for r in rows:
        for line in r.output.split("\n"):
            line = line.strip()
            if not line or not line[0].isupper():      # skips list items, code, numbered steps
                continue
            for s in _SENT_SPLIT.split(line):
                s = s.strip()
                n = len(s.split())
                if not (min_words <= n <= max_words) or not s.endswith(".") or not s[0].isupper():
                    continue
                if not _ASCII_OK.match(s) or is_instruction_like(s) or "you" in s.lower().split():
                    continue
                if _ASSISTANT_TALK.search(s) or not one_sentence(s):
                    continue
                if s not in seen:
                    seen.add(s)
                    out.append(s)
    return out


def single_sentence_instructions(rows: list[AlpacaRow], min_words: int = 4, max_words: int = 16) -> list[str]:
    """Self-contained one-sentence Alpaca instructions (no input) usable as injected text."""
    bad = re.compile(r"\b(this|these|following|above|below|given|input|text|sentence|passage|paragraph|"
                     r"user|system|message|provided)\b", re.I)
    out = []
    for r in rows:
        ins = r.instruction
        if r.input or ins[-1:] not in ".?" or "\n" in ins or not _ASCII_OK.match(ins):
            continue
        if len(_SENT_SPLIT.split(ins)) != 1 or bad.search(ins) or not one_sentence(ins):
            continue
        if min_words <= len(ins.split()) <= max_words and is_instruction_like(ins):
            out.append(ins)
    return out


def split_holdout(items: list[str], frac: float = 0.1, seed: int = 182) -> tuple[list[str], list[str]]:
    """Deterministic train / held-out split (held-out carriers are used only by evaluations)."""
    items = sorted(set(items))
    random.Random(seed).shuffle(items)
    k = max(1, int(len(items) * frac))
    return items[k:], items[:k]


@dataclass
class Pools:
    """Disjoint row-index pools over alpaca-cleaned."""

    host_with_input: list[int] = field(default_factory=list)   # ignore-style hosts (task + data)
    injected_rows: list[int] = field(default_factory=list)     # rows injected into hosts
    benign_rows: list[int] = field(default_factory=list)       # ordinary benign tasks
    carrier_rows: list[int] = field(default_factory=list)      # source of carrier sentences
    fidelity_rows: list[int] = field(default_factory=list)     # source of instruction-like content


def make_pools(rows: list[AlpacaRow], seed: int = 182, max_input_words: int = 200) -> Pools:
    """Meta-SecAlign uses every alpaca-cleaned row with a non-empty input as a host and injects
    a random row of the original Alpaca file (with or without input). We keep that shape but
    make the roles disjoint: no row is a host, an injection, a benign task and a text source
    at once."""
    rng = random.Random(seed)
    with_input = [r.idx for r in rows if r.input and len(r.input.split()) <= max_input_words]
    no_input = [r.idx for r in rows if not r.input]
    rng.shuffle(with_input)
    rng.shuffle(no_input)
    a, b = int(0.50 * len(with_input)), int(0.85 * len(with_input))
    q = len(no_input) // 4
    p = Pools()
    p.host_with_input = with_input[:a]
    p.injected_rows = with_input[b:] + no_input[:q]
    p.benign_rows = with_input[a:b] + no_input[q:2 * q]
    p.carrier_rows = no_input[2 * q:3 * q]
    p.fidelity_rows = no_input[3 * q:]
    rng.shuffle(p.injected_rows)
    rng.shuffle(p.benign_rows)
    return p
