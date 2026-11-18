"""Build the six training conditions (proposal Table 1 plus two controls).

    # 1) list the prompts whose targets the base model must generate
    python -m pno.data.build requests
    # 2) generate them on the server (greedy, one user message per prompt)
    python -m pno.gen.run --requests data/generated/requests.jsonl --model Qwen/Qwen3.5-2B \
        --out data/generated/responses.jsonl --max_new_tokens 1024
    # 3) assemble 6 conditions x seeds
    python -m pno.data.build assemble

``preview`` prints one example of each generation-free component; ``assemble
--allow_missing_generation`` writes only the conditions that need no generated targets.

Per seed, every component is sampled once and shared by all conditions that use it:
the ordinary benign half is identical in C1, C3, C5, C6; process-style, target control and
boundary control are renderings of the same conflict specs; the fidelity half is identical
in C2 and C4; the ignore-style half is identical in C1 and C2. So conditions differ only in
the factor they are meant to vary.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from pno import third_party
from pno.catalog import build_registry
from pno.data import alpaca_pairs as ap_
from pno.data.alpaca import (carrier_sentences, load_alpaca, make_pools, single_sentence_instructions,
                             split_holdout)
from pno.data.conflicts import (BENIGN_POSITIONS, CarrierSampler, TransformSampler, boundary_pair, fidelity_pair,
                                process_pair, sample_conflict, target_pair)
from pno.formatting import Prompt
from pno.io import REPO_ROOT, load_config, read_jsonl, stable_hash, write_json, write_jsonl


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_keep(path: str | None, reg) -> list[str]:
    p = _resolve(path)
    if p and p.exists():
        keep = json.loads(p.read_text())["keep"]
        missing = [k for k in keep if k not in reg]
        if missing:
            raise KeyError(f"keep.json names unknown transforms: {missing}")
        return keep
    print(f"[build] WARNING: no competence keep-list at {path!r}; using every transform that passes the "
          "content-sensitivity check (fine for smoke tests, not for the study)")
    from pno.catalog.base import sensitivity
    from pno.data.conflicts import _CLASS_PROBES, unit_sep

    keep = []
    for t in reg:
        probes = [unit_sep(t.unit).join(_CLASS_PROBES[i:i + 2]) for i in range(3)] * 4
        if sensitivity(t, probes) >= 0.95:
            keep.append(t.name)
    return keep


class Builder:
    def __init__(self, cfg: dict, keep_path: str | None = None, rows=None, delm_tokens: dict | None = None):
        self.cfg = cfg
        d = self.d = cfg["data"]
        self.reg = build_registry(trusted_suffix=d.get("trusted_suffix", ""))
        self.keep = load_keep(keep_path or d.get("keep_file"), self.reg)
        self.inj_pool = self.keep if d.get("injected_from", "keep") == "keep" else self.reg.names()
        self.rows = rows if rows is not None else load_alpaca(_resolve(d.get("alpaca_cache")))
        self.pools = make_pools(self.rows, d["pool_seed"])
        # 10% of carriers and content sentences are held out for evaluation (pno.eval.conflict)
        self.sentences, self.heldout_sentences = split_holdout(
            carrier_sentences([self.rows[i] for i in self.pools.carrier_rows]), 0.1, d["pool_seed"])
        self.fid_sentences, self.heldout_fid_sentences = split_holdout(
            single_sentence_instructions([self.rows[i] for i in self.pools.fidelity_rows]), 0.1, d["pool_seed"])
        self.delm = delm_tokens or third_party.secalign_constants()["OTHER_DELM_TOKENS"]
        self.weights = d.get("injection_placement")
        self.max_chars = d.get("max_target_chars", 1500)

    def rng(self, seed, part: str) -> random.Random:
        return random.Random(f"{self.d['pool_seed']}|{seed}|{part}")

    def samplers(self, rng: random.Random, sentences: list[str]):
        bal = self.d.get("balance_behavior_classes", True)
        return (TransformSampler(self.reg, self.keep, rng, bal), TransformSampler(self.reg, self.inj_pool, rng, bal),
                CarrierSampler(sentences, rng, tuple(self.d["carrier_sentences_text"]), tuple(self.d["carrier_lines"])))

    # ---- generation-free components -------------------------------------------------
    def conflict_specs(self, seed, n: int, sentences: list[str] | None = None, part: str = "conflict"):
        rng = self.rng(seed, part)
        us, js, cs = self.samplers(rng, sentences or self.sentences)
        return [sample_conflict(self.reg, us, js, cs, rng, f"s{seed}-c{k}", self.delm, self.weights, self.max_chars)
                for k in range(n)]

    def conflict_rows(self, seed) -> dict[str, list[dict]]:
        specs = self.conflict_specs(seed, self.d["security_pairs"])
        return {"process": [process_pair(self.reg, s) for s in specs],
                "target": [target_pair(self.reg, s) for s in specs],
                "boundary": [boundary_pair(self.reg, s) for s in specs]}

    def fidelity_rows(self, seed, n: int | None = None, sentences=None, fid_sentences=None,
                      part: str = "fidelity") -> list[dict]:
        rng = self.rng(seed, part)
        us, _, cs = self.samplers(rng, sentences or self.sentences)
        fid = fid_sentences or self.fid_sentences
        rows, n = [], n or self.d["benign_pairs"]
        while len(rows) < n:
            t = us.sample()
            r = fidelity_pair(self.reg, t, cs.sample(self.reg[t].unit), rng.choice(fid),
                              rng.choice(BENIGN_POSITIONS), rng, f"s{seed}-f{len(rows)}", self.max_chars)
            if r is not None:
                rows.append(r)
        return rows

    # ---- components with self-generated targets -------------------------------------
    def ignore_specs(self, seed):
        n = int(self.d["security_pairs"] * self.d.get("oversample", 1.25))
        return ap_.sample_ignore_specs(self.pools.host_with_input, self.pools.injected_rows, n,
                                       self.rng(seed, "ignore"), self.delm, self.weights)

    def benign_idxs(self, seed) -> list[int]:
        n = int(self.d["benign_pairs"] * self.d.get("oversample", 1.25))
        return self.rng(seed, "ordinary").sample(self.pools.benign_rows, n)

    def requests(self, seeds: list[int]) -> list[dict]:
        prompts: list[Prompt] = []
        for s in seeds:
            for spec in self.ignore_specs(s):
                ps = ap_.ignore_prompts(self.rows, spec)
                prompts += [ps["clean"], ps["injected_alone"]]
            prompts += [ap_.benign_prompt(self.rows[i]) for i in self.benign_idxs(s)]
        return ap_.requests_for(prompts)

    # ---- assembly ---------------------------------------------------------------------
    def components(self, seed, responses: ap_.Responses | None) -> dict[str, list[dict]]:
        comp = self.conflict_rows(seed)
        comp["fidelity"] = self.fidelity_rows(seed)
        if responses is not None:
            comp["ignore"] = ap_.assemble_ignore(self.rows, self.ignore_specs(seed), responses,
                                                 self.d["security_pairs"])
            comp["ordinary"] = ap_.assemble_ordinary(self.rows, self.benign_idxs(seed), responses,
                                                     self.d["benign_pairs"], self.rng(seed, "ordinary-rejected"))
        return comp

    def condition(self, comp: dict[str, list[dict]], spec: dict, seed) -> list[dict]:
        rows = comp[spec["security"]] + comp[spec["benign"]]
        rows = [dict(r, id=f"s{seed}-{r['kind']}-{i}") for i, r in enumerate(rows)]
        random.Random(f"mix|{seed}").shuffle(rows)
        return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["requests", "assemble", "preview"])
    p.add_argument("--out", default="data/generated")
    p.add_argument("--config", default=None)
    p.add_argument("--keep", default=None, help="competence keep.json (default: data.keep_file)")
    p.add_argument("--responses", default=None, help="default: <out>/responses.jsonl")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--allow_missing_generation", action="store_true",
                   help="assemble only the conditions that need no generated targets")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    seeds = a.seeds if a.seeds is not None else cfg["seeds"]
    b = Builder(cfg, a.keep)
    out = _resolve(a.out)
    print(f"[build] {len(b.keep)} transforms in the pool, {len(b.sentences)} carrier sentences, "
          f"{len(b.fid_sentences)} instruction-like content sentences")

    if a.stage == "requests":
        reqs = b.requests(seeds)
        n = write_jsonl(out / "requests.jsonl", reqs)
        print(f"[build] wrote {n} generation requests -> {out / 'requests.jsonl'}")
        return

    if a.stage == "preview":
        comp = b.components(seeds[0], None)
        for kind, rows in comp.items():
            r = rows[0]
            print(f"\n=== {kind} ===\n" + json.dumps({k: r[k] for k in ("prompt", "chosen", "rejected")},
                                                   indent=1, ensure_ascii=False))
        return

    responses = None
    rpath = _resolve(a.responses) if a.responses else out / "responses.jsonl"
    if rpath.exists():
        responses = ap_.Responses(read_jsonl(rpath))
        print(f"[build] {len(responses)}/{responses.n_total} generated replies usable")
    elif not a.allow_missing_generation:
        raise SystemExit(f"missing {rpath}; run the generation step or pass --allow_missing_generation")

    manifest = {"config": cfg, "keep": b.keep, "seeds": {}}
    for s in seeds:
        comp = b.components(s, responses)
        sd = out / f"seed{s}"
        for kind, rows in comp.items():
            write_jsonl(sd / "components" / f"{kind}.jsonl", rows)
        manifest["seeds"][s] = {}
        for name, spec in cfg["conditions"].items():
            if spec["security"] not in comp or spec["benign"] not in comp:
                print(f"[build] seed {s}: skip {name} (needs generated targets)")
                continue
            rows = b.condition(comp, spec, s)
            write_jsonl(sd / f"{name}.jsonl", rows)
            manifest["seeds"][s][name] = {
                "n": len(rows), "kinds": dict(Counter(r["kind"] for r in rows)),
                "sha": stable_hash(*(json.dumps(r, sort_keys=True) for r in rows), n=16)}
            print(f"[build] seed {s} {name}: {manifest['seeds'][s][name]}")
    write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
