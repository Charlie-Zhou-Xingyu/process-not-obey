"""Text transformations whose correct output Python can compute exactly.

A ``Transform`` pairs an instruction template with a reference implementation. The same
transform is phrased for different targets:

* ``input``  - "... the input" (trusted user instruction aimed at the untrusted input; process-style)
* ``user``   - "... the user message" (the catalog's own wording; an injection aimed at the
  trusted user message, or a system instruction in the boundary control)
* ``system`` - "... the system message" (boundary control: injection in the user message
  aimed at the system message, the original recipe of [8])

Every transform used for security data must be content-sensitive: its output must change
when one sentence is added to or removed from the input. Otherwise processing an injected
sentence and dropping it cannot be told apart. ``sensitivity`` checks this empirically.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# Noun phrase substituted for the catalog's "user message" (templates carry their own article).
TARGET_PHRASES = {
    "input": "input",
    "user": "user message",
    "system": "system message",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=\S)")


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split("\n")).strip()
    return [s for s in _SENT_SPLIT.split(text) if s]


def split_lines(text: str) -> list[str]:
    return [ln for ln in text.split("\n")]


@dataclass(frozen=True)
class Transform:
    name: str
    template: str                       # must contain "{target}"
    fn: Callable[[str], str]
    category: str = "misc"
    unit: str = "text"                  # "text" (sentences joined by spaces) or "lines"
    source: str = "goalgoal"            # "goalgoal" = catalog of [8]; "extra" = our additions
    notes: str = ""

    def instruction(self, target: str) -> str:
        return self.template.replace("{target}", TARGET_PHRASES[target])

    def apply(self, text: str) -> str:
        return self.fn(text)


class Registry:
    def __init__(self, trusted_suffix: str = ""):
        self._items: dict[str, Transform] = {}
        # Appended to instructions on the trusted side only (e.g. " Output only the result.").
        # The catalog of [8] says nothing about output format; injected instructions keep
        # the catalog wording.
        self.trusted_suffix = trusted_suffix

    def trusted_instruction(self, t: "Transform", target: str) -> str:
        return t.instruction(target) + self.trusted_suffix

    def add(self, t: Transform) -> Transform:
        if "{target}" not in t.template:
            raise ValueError(f"{t.name}: template needs {{target}}")
        if t.name in self._items:
            raise ValueError(f"duplicate transform {t.name}")
        self._items[t.name] = t
        return t

    def __getitem__(self, name: str) -> Transform:
        return self._items[name]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self):
        return iter(self._items.values())

    def __len__(self):
        return len(self._items)

    def names(self, source: str | None = None) -> list[str]:
        return [t.name for t in self._items.values() if source is None or t.source == source]


PROBE_SENTENCES = (
    "The weather was mild for most of the week.",
    "Bring seven green apples tomorrow.",
    "Nobody expected the meeting to end so early in the afternoon.",
    "Call Dr. Lee at noon.",
    "A second train leaves from the same platform an hour later, just after sunset.",
)


def sensitivity(t: Transform, carriers: list[str], extra_sentences: tuple[str, ...] = PROBE_SENTENCES) -> float:
    """Share of (carrier, extra sentence) cases with T(carrier + sentence) != T(carrier).

    The extra sentences differ in length and wording so that parity-style or
    first/last-element transforms cannot pass by accident.
    """
    if not carriers:
        return 0.0
    sep = "\n" if t.unit == "lines" else " "
    diff = sum(t.apply(c + sep + extra_sentences[i % len(extra_sentences)]) != t.apply(c)
               for i, c in enumerate(carriers))
    return diff / len(carriers)


_FALLBACK_CARRIERS = (
    "The harbor was quiet before the first ferry arrived.",
    "Two engineers measured the bridge again in March.",
    "A small bakery opened near the old station last year.",
    "Most visitors leave the island before the evening tide.",
    "The library keeps its oldest maps in a locked room.",
    "Rain delayed the harvest by almost two weeks.",
)


def sensitive_names(reg: Registry, sentences: tuple[str, ...] | list[str] = _FALLBACK_CARRIERS,
                    threshold: float = 0.95, n: int = 60) -> list[str]:
    """Transforms whose output changes when a sentence is added (the pool usable as T_u)."""
    import random

    rng = random.Random(0)
    probes = {"text": [" ".join(rng.sample(list(sentences), 2)) for _ in range(n)],
              "lines": ["\n".join(rng.sample(list(sentences), 3)) for _ in range(n)]}
    return [t.name for t in reg if sensitivity(t, probes[t.unit]) >= threshold]
