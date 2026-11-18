"""Comparison of a model output with the Python reference of a transformation.

The reference is compared as is (only line-end whitespace and surrounding blank space are
ignored). If the raw output does not match, we retry after removing wrappers a model may
add around an otherwise exact answer (a code fence, an "Output:" prefix, surrounding
quotes). Wrappers are never removed from the reference: for a transform whose correct
output is itself quoted (JSON string, CSV field), an unquoted copy must not match.
"""
from __future__ import annotations

import re

_FENCE = re.compile(r"^```[a-zA-Z]*\n(.*?)\n?```$", re.S)
_PREFIXES = ("Output:", "Result:", "Answer:", "Here is the result:", "Here is the transformed text:")


def strip_wrappers(text: str) -> str:
    t = text.strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1).strip()
    for prefix in _PREFIXES:
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):].strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'`":
        t = t[1:-1].strip()
    return t


def normalize(text: str) -> str:
    """Drop surrounding blank space and whitespace at line ends; keep everything else."""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").strip().split("\n")]
    return "\n".join(lines)


def exact_match(output: str, reference: str) -> bool:
    ref = normalize(reference)
    return normalize(output) == ref or normalize(strip_wrappers(output)) == ref
