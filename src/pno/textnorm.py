"""Normalization used when checking transformation outputs against the Python reference."""
from __future__ import annotations

import re

_FENCE = re.compile(r"^```[a-zA-Z]*\n(.*?)\n?```$", re.S)
_WS = re.compile(r"[ \t]+")


def strip_wrappers(text: str) -> str:
    """Remove harmless wrappers a model may add around an otherwise exact answer."""
    t = text.strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1).strip()
    for prefix in ("Output:", "Result:", "Answer:", "Here is the result:", "Here is the transformed text:"):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):].strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'`":
        t = t[1:-1].strip()
    return t


def normalize(text: str) -> str:
    """Whitespace-insensitive at line ends, sensitive to everything else (case, punctuation)."""
    t = strip_wrappers(text).replace("\r\n", "\n")
    lines = [_WS.sub(" ", ln).strip() for ln in t.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def exact_match(output: str, reference: str) -> bool:
    return normalize(output) == normalize(reference)
