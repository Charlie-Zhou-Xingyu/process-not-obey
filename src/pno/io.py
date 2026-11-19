"""Small I/O helpers: JSONL, YAML config, stable hashing."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


CONFIG_ENV = "PNO_CONFIG"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """The experiment config: ``path`` if given, else ``$PNO_CONFIG``, else configs/experiment.yaml.

    ``pno.pipeline --config X`` sets ``PNO_CONFIG`` so that every step it launches (training,
    generation, scoring) reads the same file."""
    path = Path(path) if path else Path(os.environ.get(CONFIG_ENV) or REPO_ROOT / "configs" / "experiment.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def stable_hash(*parts: Any, n: int = 12) -> str:
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()
    return h[:n]


def revision_for(model: str | os.PathLike, cfg: dict | None = None) -> str | None:
    """Pinned commit of a hub model (config ``revisions``); None for local paths and unpinned ids."""
    if Path(model).exists():
        return None
    try:
        cfg = cfg or load_config()
    except FileNotFoundError:
        return None
    return (cfg.get("revisions") or {}).get(str(model))
