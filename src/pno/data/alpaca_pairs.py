"""Alpaca-based pairs with self-generated targets: ignore-style security and ordinary benign.

Ignore-style follows Meta-SecAlign's SecAlign++ recipe (``utils.generate_preference_dataset``):

* host = an alpaca-cleaned row with a non-empty input (trusted instruction + data);
* injection = another row's instruction, plus its input if it has one;
* placement = 45% before the data, 45% after it, 10% completion-style (after a fake answer,
  behind randomized fake delimiters; the fake answer is the host row's reference output);
* training prompt = [user: host instruction, input: data with injection];
* preferred = the base model's own reply to the clean task (instruction + blank line + data,
  as one user message); rejected = the base model's reply to the injected task on its own.

Ordinary benign: an uninjected Alpaca task; preferred = the base model's reply to it,
rejected = the base model's reply to a different task.

Differences from the original, all deliberate: rows play one role only (disjoint pools);
both hosts and injections come from alpaca-cleaned; targets are sampled at temperature 0.8
like the original but with a fixed seed and at most ``data.gen_max_new_tokens`` tokens;
replies that hit the token limit are discarded rather than used as truncated targets.

Building is two-phase because targets come from the base model: ``requests_for`` lists
the prompts that need a reply (keyed by a stable hash), the server generates them with
``pno.gen.run``, and ``assemble_*`` look the replies up.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from pno.data import injection as inj_
from pno.data.alpaca import AlpacaRow
from pno.formatting import Prompt
from pno.io import stable_hash

GEN_PLACEMENT = "user_concat"   # one user message: instruction + "\n\n" + data (Meta-SecAlign's code)


def prompt_key(p: Prompt) -> str:
    return stable_hash(p.system, p.instruction, p.data, n=16)


def injected_task_text(r: AlpacaRow) -> str:
    return r.instruction + (" " + r.input if r.input else "")


@dataclass
class IgnoreSpec:
    host: int
    injected: int
    kind: str
    marks: dict | None


def sample_ignore_specs(hosts: list[int], injected_pool: list[int], n: int, rng: random.Random,
                        delm_tokens: dict, weights: dict | None = None) -> list[IgnoreSpec]:
    hs, inj = rng.sample(hosts, n), rng.sample(injected_pool, n)
    specs = []
    for h, i in zip(hs, inj):
        kind = inj_.sample_kind(rng, weights)
        specs.append(IgnoreSpec(h, i, kind, inj_.sample_marks(rng, delm_tokens) if kind == "completion" else None))
    return specs


def ignore_prompts(rows: list[AlpacaRow], s: IgnoreSpec) -> dict[str, Prompt]:
    host, inj = rows[s.host], rows[s.injected]
    if s.kind == "completion":
        attacked = inj_.place(host.input, inj.instruction, "completion", marks=s.marks,
                              fake_response=host.output, injection_input=inj.input)
    else:
        attacked = inj_.place(host.input, injected_task_text(inj), s.kind)
    return {
        "attacked": Prompt(host.instruction, attacked),
        "clean": Prompt(host.instruction, host.input),
        "injected_alone": Prompt(injected_task_text(inj)),
    }


def benign_prompt(r: AlpacaRow) -> Prompt:
    return Prompt(r.instruction, r.input or None)


def requests_for(prompts: list[Prompt]) -> list[dict]:
    seen, out = set(), []
    for p in prompts:
        k = prompt_key(p)
        if k not in seen:
            seen.add(k)
            out.append({"key": k, "prompt": p.to_dict(), "placement": GEN_PLACEMENT})
    return out


class Responses:
    """Generated replies keyed by prompt hash; truncated or empty replies are unusable."""

    def __init__(self, rows: list[dict]):
        self._ok = {r["key"]: r["response"].strip() for r in rows
                    if not r.get("truncated") and r["response"].strip()}
        self.n_total = len(rows)

    def get(self, p: Prompt) -> str | None:
        return self._ok.get(prompt_key(p))

    def __len__(self):
        return len(self._ok)


def assemble_ignore(rows: list[AlpacaRow], specs: list[IgnoreSpec], responses: Responses, n: int) -> list[dict]:
    out = []
    for s in specs:
        ps = ignore_prompts(rows, s)
        chosen, rejected = responses.get(ps["clean"]), responses.get(ps["injected_alone"])
        if chosen is None or rejected is None or chosen == rejected:
            continue
        out.append({"kind": "ignore", "prompt": ps["attacked"].to_dict(), "chosen": chosen, "rejected": rejected,
                    "meta": {"host": s.host, "injected": s.injected, "placement": s.kind, "marks": s.marks,
                             "injected_text": injected_task_text(rows[s.injected])}})
        if len(out) == n:
            return out
    raise RuntimeError(f"only {len(out)}/{n} usable ignore-style pairs; raise data.oversample")


def assemble_ordinary(rows: list[AlpacaRow], idxs: list[int], responses: Responses, n: int,
                      rng: random.Random) -> list[dict]:
    ok = [i for i in idxs if responses.get(benign_prompt(rows[i])) is not None]
    if len(ok) < n + 1:
        raise RuntimeError(f"only {len(ok)} usable benign rows for {n} pairs; raise data.oversample")
    out = []
    for i in ok[:n]:
        resp = responses.get(benign_prompt(rows[i]))
        for _ in range(50):
            j = rng.choice(ok)   # rejected = the base model's reply to a different task
            other = responses.get(benign_prompt(rows[j]))
            if j != i and other != resp:
                break
        else:
            raise RuntimeError("could not find a distinct rejected response")
        out.append({"kind": "ordinary", "prompt": benign_prompt(rows[i]).to_dict(), "chosen": resp,
                    "rejected": other, "meta": {"row": i, "rejected_from": j}})
    return out
