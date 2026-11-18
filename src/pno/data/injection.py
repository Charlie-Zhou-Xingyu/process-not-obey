"""Where an injected instruction sits inside untrusted data (Meta-SecAlign's SecAlign++ scheme).

Every security pair, Alpaca-based or catalog-based, draws its placement from the same
distribution, so the conditions do not differ in attack format:

* ``start`` (45%)       injection + separator + data
* ``end`` (45%)         data + separator + injection
* ``completion`` (10%)  data, then a fake answer to the trusted task and the injection, each
  behind a randomized fake delimiter (``utils.create_injection_for_completion`` in the
  Meta-SecAlign repo): a mark style, a role word plus an Output/Command/Data-type word, and
  random upper / lower / unchanged casing.

The delimiter vocabulary is read from the pinned Meta-SecAlign clone
(``pno.third_party.secalign_constants()["OTHER_DELM_TOKENS"]``).
"""
from __future__ import annotations

import random

DEFAULT_WEIGHTS = {"start": 0.45, "end": 0.45, "completion": 0.10}


def sample_kind(rng: random.Random, weights: dict[str, float] | None = None) -> str:
    weights = weights or DEFAULT_WEIGHTS
    kinds = list(weights)
    return rng.choices(kinds, weights=[weights[k] for k in kinds], k=1)[0]


def sample_marks(rng: random.Random, delm_tokens: dict) -> dict[str, str]:
    """One mark style shared by the three delimiters; role, word and casing drawn per delimiter."""
    mark = rng.choice(delm_tokens["mark"]) + ":"

    def delm(name: str) -> str:
        role = rng.choice(delm_tokens["user" if name in ("inst", "inpt") else "asst"])
        s = role + rng.choice(delm_tokens[name])
        p = rng.random()
        return s.upper() if p < 1 / 3 else s.lower() if p < 2 / 3 else s

    return {name: mark.format(s=delm(name)) for name in ("resp", "inst", "inpt")}


def completion_block(marks: dict[str, str], fake_response: str, instruction: str, inpt: str = "") -> str:
    text = marks["resp"] + "\n\n" + fake_response
    text += "\n\n" + marks["inst"] + "\n\n" + instruction
    if inpt:
        text += "\n\n" + marks["inpt"] + "\n\n" + inpt
    return text


def place(data: str, injection: str, kind: str, sep: str = " ", marks: dict | None = None,
          fake_response: str = "", injection_input: str = "") -> str:
    """``injection`` is the full injected text for start/end; for completion it is the injected
    instruction and ``injection_input`` its optional input."""
    if kind == "start":
        return injection + sep + data
    if kind == "end":
        return data + sep + injection
    if kind == "completion":
        if marks is None:
            raise ValueError("completion placement needs marks")
        return data + "\n\n" + completion_block(marks, fake_response, injection, injection_input)
    raise ValueError(f"unknown placement {kind!r}")
