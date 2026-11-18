"""Transformation catalog: the goal-goal catalog of [8] re-implemented, plus our extra
sentence/line operations. ``build_registry()`` returns both."""
from __future__ import annotations

from pno.catalog.base import Registry, Transform


def build_registry(include: tuple[str, ...] = ("goalgoal", "extra")) -> Registry:
    reg = Registry()
    if "goalgoal" in include:
        from pno.catalog import goalgoal

        goalgoal.register(reg)
    if "extra" in include:
        from pno.catalog import extra

        extra.register(reg)
    return reg


__all__ = ["Registry", "Transform", "build_registry"]
