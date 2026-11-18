"""Catalog-based pairs: process-style, target control, boundary control, fidelity.

All four are computed exactly by Python (no model generation). One ``ConflictSpec`` fixes
the transform pair, carrier text, and where the injected sentence sits; the three security
variants are different renderings of the same spec, so they share prompts and rejected
responses wherever the design says they should:

==================  =========================================  ===============================  ======================
condition           prompt                                     preferred                        rejected
==================  =========================================  ===============================  ======================
process-style       user: T_u("... the input");                T_u(whole input)                 T_i(user message)
                    input: carrier + T_i("... the user message")
target control      same as process-style                      T_u(carrier)  (injection cut)    same as process-style
boundary control    system: T_u("... the user message");       T_u(user message)                T_i(system message)
                    user: carrier + T_i("... the system message")
fidelity (benign)   user: T_u("... the input");                T_u(carrier + sentence)          T_u(carrier)  (dropped)
                    input: carrier + instruction-like sentence
==================  =========================================  ===============================  ======================

Unlike [8], whose two messages contain nothing but the two instructions, our untrusted
side also carries ordinary text (the carrier). Without it the target control would have
nothing left to transform once the injected sentence is removed.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from pno.catalog.base import Registry, Transform
from pno.data import injection as inj_
from pno.formatting import Prompt

BENIGN_POSITIONS = ("start", "middle", "end")
# fixed probe texts used only to group transforms that behave identically on plain prose
_CLASS_PROBES = (
    "The harbor was quiet before the first ferry arrived.",
    "Two engineers measured the bridge again in March.",
    "A small bakery opened near the old station last year.",
    "Most visitors leave the island before the evening tide.",
)


def unit_sep(unit: str) -> str:
    return "\n" if unit == "lines" else " "


def join_units(units: list[str], unit: str) -> str:
    return unit_sep(unit).join(units)


def insert_unit(units: list[str], sentence: str, position: str, rng: random.Random) -> tuple[list[str], int]:
    """Insert ``sentence`` as one more sentence/line. Returns new units and its index."""
    if position == "start":
        k = 0
    elif position == "end":
        k = len(units)
    else:
        k = rng.randint(1, len(units) - 1) if len(units) >= 2 else len(units)
    return units[:k] + [sentence] + units[k:], k


class CarrierSampler:
    def __init__(self, sentences: list[str], rng: random.Random,
                 n_text: tuple[int, int] = (1, 2), n_lines: tuple[int, int] = (2, 4)):
        if len(sentences) < 10:
            raise ValueError("need more carrier sentences")
        self.sentences, self.rng = sentences, rng
        self.n_text, self.n_lines = n_text, n_lines

    def sample(self, unit: str) -> list[str]:
        lo, hi = self.n_lines if unit == "lines" else self.n_text
        return self.rng.sample(self.sentences, self.rng.randint(lo, hi))


class TransformSampler:
    """Uniform over behavior classes, then uniform inside the class.

    Several catalog instructions do the same thing to plain single-spaced ASCII prose
    (copy, NFC/NFD/NFKC, strip, CRLF->LF all return the text unchanged). Sampling
    instructions uniformly would make "return the input unchanged" a large share of the
    data whenever these pass the competence filter, so we balance by behavior instead.
    """

    def __init__(self, reg: Registry, names: list[str], rng: random.Random, balance: bool = True):
        if not names:
            raise ValueError("empty transform pool")
        self.rng = rng
        if not balance:
            self.classes = [[n] for n in sorted(names)]
            return
        groups: dict[tuple, list[str]] = {}
        for n in sorted(names):
            t = reg[n]
            probes = [unit_sep(t.unit).join(_CLASS_PROBES[i:i + 2]) for i in range(len(_CLASS_PROBES) - 1)]
            groups.setdefault((t.unit, tuple(t.apply(p) for p in probes)), []).append(n)
        self.classes = list(groups.values())

    def sample(self, exclude: str | None = None) -> str:
        classes = [[n for n in c if n != exclude] for c in self.classes]
        classes = [c for c in classes if c]
        return self.rng.choice(self.rng.choice(classes))


@dataclass
class ConflictSpec:
    t_user: str            # transform the trusted side asks for
    t_inj: str             # transform the injected sentence asks for
    carrier: list[str]     # carrier sentences (unit=text) or lines (unit=lines)
    kind: str              # start | end | completion  (pno.data.injection)
    marks: dict | None
    seed_tag: str

    def to_dict(self) -> dict:
        return asdict(self)


def _render(spec: ConflictSpec, tu: Transform, injected_sentence: str) -> str:
    clean = join_units(spec.carrier, tu.unit)
    return inj_.place(clean, injected_sentence, spec.kind, sep=unit_sep(tu.unit), marks=spec.marks,
                      fake_response=tu.apply(clean))


def process_pair(reg: Registry, spec: ConflictSpec) -> dict:
    tu, ti = reg[spec.t_user], reg[spec.t_inj]
    user = tu.instruction("input")
    data = _render(spec, tu, ti.instruction("user"))
    return {"kind": "process", "prompt": Prompt(user, data).to_dict(),
            "chosen": tu.apply(data), "rejected": ti.apply(user),
            "meta": {**spec.to_dict(), "injected_sentence": ti.instruction("user")}}


def target_pair(reg: Registry, spec: ConflictSpec) -> dict:
    tu = reg[spec.t_user]
    row = process_pair(reg, spec)
    row["kind"] = "target"
    row["chosen"] = tu.apply(join_units(spec.carrier, tu.unit))
    return row


def boundary_pair(reg: Registry, spec: ConflictSpec) -> dict:
    tu, ti = reg[spec.t_user], reg[spec.t_inj]
    system = tu.instruction("user")
    user_msg = _render(spec, tu, ti.instruction("system"))
    return {"kind": "boundary", "prompt": Prompt(user_msg, None, system).to_dict(),
            "chosen": tu.apply(user_msg), "rejected": ti.apply(system),
            "meta": {**spec.to_dict(), "injected_sentence": ti.instruction("system")}}


def _clean_text(x: str, max_chars: int) -> bool:
    return bool(x) and x == x.strip() and len(x) <= max_chars


def sample_conflict(reg: Registry, user_sampler: TransformSampler, inj_sampler: TransformSampler,
                    carriers: CarrierSampler, rng: random.Random, seed_tag: str, delm_tokens: dict,
                    weights: dict | None = None, max_chars: int = 1500, max_tries: int = 200) -> ConflictSpec:
    """Sample a spec whose processed / ignored / executed responses are pairwise different,
    non-empty, free of leading or trailing whitespace and not longer than ``max_chars``."""
    for _ in range(max_tries):
        tu = reg[user_sampler.sample()]
        ti = reg[inj_sampler.sample(exclude=tu.name)]
        kind = inj_.sample_kind(rng, weights)
        marks = inj_.sample_marks(rng, delm_tokens) if kind == "completion" else None
        spec = ConflictSpec(tu.name, ti.name, carriers.sample(tu.unit), kind, marks, seed_tag)
        p, t, b = process_pair(reg, spec), target_pair(reg, spec), boundary_pair(reg, spec)
        texts = [p["chosen"], t["chosen"], p["rejected"], b["chosen"], b["rejected"]]
        if len({p["chosen"], t["chosen"], p["rejected"]}) == 3 and len({b["chosen"], t["chosen"], b["rejected"]}) == 3 \
                and all(_clean_text(x, max_chars) for x in texts):
            return spec
    raise RuntimeError("could not sample a valid conflict; check the transform pools")


def fidelity_pair(reg: Registry, t_user: str, carrier: list[str], sentence: str, position: str,
                  rng: random.Random, seed_tag: str, max_chars: int = 1500) -> dict | None:
    tu: Transform = reg[t_user]
    units, k = insert_unit(carrier, sentence, position, rng)
    data = join_units(units, tu.unit)
    chosen, rejected = tu.apply(data), tu.apply(join_units(carrier, tu.unit))
    if chosen == rejected or not _clean_text(chosen, max_chars) or not _clean_text(rejected, max_chars):
        return None
    return {"kind": "fidelity", "prompt": Prompt(tu.instruction("input"), data).to_dict(),
            "chosen": chosen, "rejected": rejected,
            "meta": {"t_user": t_user, "carrier": carrier, "position": position, "insert_at": k,
                     "content_sentence": sentence, "seed_tag": seed_tag}}
