"""Read constants and datasets from the pinned third-party clones without importing them.

Python files are parsed with ``ast`` and only literal assignments are evaluated, so no
third-party code runs. ``scripts/fetch_third_party.sh`` creates the clones.
"""
from __future__ import annotations

import ast
import json
import os
from functools import lru_cache
from pathlib import Path

from pno.io import REPO_ROOT

PINNED = {
    "sec-fid": "c316d5d32cad1413de22b4c55fce6ad1fe2eb4e3",
    "SEP": "7606c0696f20f5aa433169fd2221f76852d1d4f5",
    "Meta_SecAlign": "2031502bdf5a046b437ec835c6b05ff468aa0699",
    "SecPO": "da0372a0e598badf4e51c98c175a962147cac2ac",
}


def root() -> Path:
    return Path(os.environ.get("PNO_THIRD_PARTY", REPO_ROOT / "third_party"))


def repo(name: str) -> Path:
    p = root() / name
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; run: bash scripts/fetch_third_party.sh")
    return p


def literal_assignments(py_file: Path, names: set[str]) -> dict:
    """Values of module-level ``NAME = <literal>`` assignments (plain or annotated)."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id in names:
                out[t.id] = ast.literal_eval(value)
    missing = names - set(out)
    if missing:
        raise KeyError(f"{py_file}: literals not found: {sorted(missing)}")
    return out


@lru_cache(maxsize=None)
def secalign_constants() -> dict:
    """``OTHER_DELM_TOKENS`` (fake delimiters) and ``IGNORE_ATTACK_SENTENCES`` of Meta-SecAlign."""
    return literal_assignments(repo("Meta_SecAlign") / "config.py", {"OTHER_DELM_TOKENS", "IGNORE_ATTACK_SENTENCES"})


@lru_cache(maxsize=None)
def secpo_seed_templates() -> tuple[str, ...]:
    f = repo("SecPO") / "src" / "secpo" / "backends" / "llama31_current_policy_pair.py"
    return tuple(literal_assignments(f, {"DEFAULT_INITIAL_TEMPLATES"})["DEFAULT_INITIAL_TEMPLATES"])


@lru_cache(maxsize=None)
def sep_rows() -> list[dict]:
    """SEP in the form Meta-SecAlign evaluates it (setup.py): the probe is whatever
    ``prompt_instructed`` adds to ``prompt_clean``, minus one leading space or newline."""
    raw = json.loads((repo("SEP") / "datasets" / "SEP_dataset.json").read_text(encoding="utf-8"))
    rows = []
    for i, d in enumerate(raw):
        inj = d["prompt_instructed"].replace(d["prompt_clean"], "")
        if inj.startswith(" ") or inj.startswith("\n"):
            inj = inj[1:]
        rows.append({"idx": i, "instruction": d["system_prompt_clean"], "input": d["prompt_clean"],
                     "injection": inj, "witness": d["witness"], "type": d["info"]["type"],
                     "insistent": d["info"]["is_insistent"]})
    return rows
