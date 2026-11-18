"""Paired bootstrap over test items (proposal, Evaluation section).

For each condition we have a per-item score matrix of shape (n_seeds, n_items), e.g. 1 if
the item was safely processed. A comparison is a linear contrast over conditions
(``{"C3": 1, "C5": -1}`` for the primary comparison; ``{"C4": 1, "C3": -1, "C2": -1,
"C1": 1}`` for the 2x2 interaction). We average each condition over its seeds, form the
per-item contrast, and resample items with replacement. Items are the unit of resampling
because every model answers the same items.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def _as_matrix(x) -> np.ndarray:
    m = np.asarray(x, dtype=float)
    return m[None, :] if m.ndim == 1 else m


def contrast_bootstrap(scores: Mapping[str, np.ndarray], weights: Mapping[str, float],
                       n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0,
                       chunk: int = 1_000) -> dict:
    per_item = None
    for cond, w in weights.items():
        v = _as_matrix(scores[cond]).mean(axis=0) * w
        per_item = v if per_item is None else per_item + v
    assert per_item is not None
    n = per_item.shape[0]
    rng = np.random.default_rng(seed)
    boots = []
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(k, n))
        boots.append(per_item[idx].mean(axis=1))
    b = np.concatenate(boots)
    lo, hi = np.quantile(b, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(per_item.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n_items": int(n), "n_boot": int(n_boot), "weights": dict(weights)}


def paired_diff(scores: Mapping[str, np.ndarray], a: str, b: str, **kw) -> dict:
    """Mean difference a - b with a paired item bootstrap CI."""
    return contrast_bootstrap(scores, {a: 1.0, b: -1.0}, **kw)


def noninferior(result: dict, margin: float, higher_is_better: bool = True) -> bool:
    """True if the CI excludes a loss larger than ``margin`` (e.g. 0.02 = 2 points)."""
    return result["ci_low"] > -margin if higher_is_better else result["ci_high"] < margin
