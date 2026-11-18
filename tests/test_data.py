import copy
import random

import pytest

from pno.data import alpaca_pairs as ap_
from pno.data import injection as inj_
from pno.data.alpaca import carrier_sentences, make_pools, single_sentence_instructions
from pno.data.build import Builder
from pno.data.conflicts import TransformSampler
from pno.formatting import Prompt, render_prompt
from pno.io import load_config

DELM = {"mark": ["{s}", "[{s}]", "<|{s}|>"], "inst": ["Command", "Task"], "inpt": ["Data"], "resp": ["Answer"],
        "user": ["", "User "], "asst": ["", "Assistant "]}


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config())
    c["data"].update(security_pairs=120, benign_pairs=120, keep_file=None, alpaca_cache=None)
    c["seeds"] = [0, 1]
    return c


@pytest.fixture(scope="module")
def builder(cfg, toy_alpaca):
    return Builder(cfg, rows=toy_alpaca, delm_tokens=DELM)


def fake_responses(builder, seeds):
    reqs = builder.requests(seeds)
    return ap_.Responses([{"key": r["key"], "response": "REPLY to: " + r["prompt"]["instruction"][:60] +
                           " | " + (r["prompt"]["data"] or "")[:40], "truncated": False} for r in reqs]), reqs


def test_pools_are_disjoint(toy_alpaca):
    p = make_pools(toy_alpaca, 182)
    parts = [p.host_with_input, p.injected_rows, p.benign_rows, p.carrier_rows, p.fidelity_rows]
    allidx = [i for part in parts for i in part]
    assert len(allidx) == len(set(allidx)) and all(parts)
    assert all(toy_alpaca[i].input for i in p.host_with_input)
    sents = carrier_sentences([toy_alpaca[i] for i in p.carrier_rows])
    assert sents and all(s.endswith(".") and "\n" not in s for s in sents)
    assert single_sentence_instructions([toy_alpaca[i] for i in p.fidelity_rows])


def test_placement_distribution_and_marks():
    rng = random.Random(0)
    kinds = [inj_.sample_kind(rng) for _ in range(4000)]
    share = {k: kinds.count(k) / len(kinds) for k in ("start", "end", "completion")}
    assert abs(share["start"] - 0.45) < 0.04 and abs(share["completion"] - 0.10) < 0.03
    m = inj_.sample_marks(rng, DELM)
    assert set(m) == {"resp", "inst", "inpt"} and all(v.endswith(":") for v in m.values())
    text = inj_.place("DATA", "Do it.", "completion", marks=m, fake_response="ANSWER")
    assert text.startswith("DATA\n\n" + m["resp"] + "\n\nANSWER\n\n" + m["inst"] + "\n\nDo it.")
    assert inj_.place("DATA", "Do it.", "start") == "Do it. DATA"
    assert inj_.place("DATA", "Do it.", "end", sep="\n") == "DATA\nDo it."


def test_conflict_variants_share_what_they_should(builder):
    comp = builder.conflict_rows(0)
    reg = builder.reg
    assert len(comp["process"]) == len(comp["target"]) == len(comp["boundary"]) == 120
    for p, t, b in zip(comp["process"], comp["target"], comp["boundary"]):
        tu, ti = reg[p["meta"]["t_user"]], reg[p["meta"]["t_inj"]]
        # target control: identical prompt and rejected response, different preferred response
        assert p["prompt"] == t["prompt"] and p["rejected"] == t["rejected"] and p["chosen"] != t["chosen"]
        user, data = p["prompt"]["instruction"], p["prompt"]["data"]
        injected = p["meta"]["injected_sentence"]
        assert user == tu.instruction("input") and injected == ti.instruction("user") and injected in data
        assert p["chosen"] == tu.apply(data)                  # the whole input, injection included
        assert p["rejected"] == ti.apply(user)                # the injection carried out
        sep = "\n" if tu.unit == "lines" else " "
        assert t["chosen"] == tu.apply(sep.join(p["meta"]["carrier"]))   # injection removed
        # boundary control: same conflict one level up (system vs user), no input role
        assert b["prompt"]["data"] is None and b["prompt"]["system"] == tu.instruction("user")
        assert ti.instruction("system") in b["prompt"]["instruction"]
        assert b["chosen"] == tu.apply(b["prompt"]["instruction"]) and b["rejected"] == ti.apply(b["prompt"]["system"])
        assert b["meta"]["carrier"] == p["meta"]["carrier"] and b["meta"]["kind"] == p["meta"]["kind"]
        for x in (p["chosen"], p["rejected"], t["chosen"], b["chosen"], b["rejected"]):
            assert x and x == x.strip()
    kinds = {p["meta"]["kind"] for p in comp["process"]}
    assert kinds == {"start", "end", "completion"}


def test_fidelity_pairs(builder):
    rows = builder.fidelity_rows(0)
    assert len(rows) == 120
    for r in rows:
        tu = builder.reg[r["meta"]["t_user"]]
        assert r["meta"]["content_sentence"] in r["prompt"]["data"]
        assert r["chosen"] == tu.apply(r["prompt"]["data"])
        sep = "\n" if tu.unit == "lines" else " "
        assert r["rejected"] == tu.apply(sep.join(r["meta"]["carrier"])) and r["chosen"] != r["rejected"]


def test_seeds_differ_and_are_reproducible(builder, cfg, toy_alpaca):
    a0, a1 = builder.conflict_rows(0), builder.conflict_rows(1)
    assert [r["prompt"] for r in a0["process"]] != [r["prompt"] for r in a1["process"]]
    again = Builder(cfg, rows=toy_alpaca, delm_tokens=DELM).conflict_rows(0)
    assert again == a0


def test_requests_and_assembly(builder, cfg):
    responses, reqs = fake_responses(builder, cfg["seeds"])
    assert len({r["key"] for r in reqs}) == len(reqs)
    assert all(r["placement"] == "user_concat" for r in reqs)
    comp = builder.components(0, responses)
    assert {k: len(v) for k, v in comp.items()} == {k: 120 for k in
                                                    ("process", "target", "boundary", "fidelity", "ignore", "ordinary")}
    for r in comp["ignore"]:
        host, inj = builder.rows[r["meta"]["host"]], builder.rows[r["meta"]["injected"]]
        assert r["prompt"]["instruction"] == host.instruction and inj.instruction in r["prompt"]["data"]
        assert host.input in r["prompt"]["data"]
        assert r["chosen"] == responses.get(Prompt(host.instruction, host.input))   # answer to the clean task
        assert r["rejected"] == responses.get(Prompt(ap_.injected_task_text(inj)))  # answer to the injection
    for r in comp["ordinary"]:
        assert r["chosen"] != r["rejected"] and r["meta"]["row"] != r["meta"]["rejected_from"]
    # every condition = 120 security + 120 benign, sharing components across conditions
    conds = {name: builder.condition(comp, spec, 0) for name, spec in cfg["conditions"].items()}
    for name, rows in conds.items():
        spec = cfg["conditions"][name]
        kinds = [r["kind"] for r in rows]
        assert kinds.count(spec["security"]) == 120 and kinds.count(spec["benign"]) == 120
        assert len({r["id"] for r in rows}) == 240
    ordinary = lambda n: sorted(r["prompt"]["instruction"] for r in conds[n] if r["kind"] == "ordinary")  # noqa: E731
    assert ordinary("C1_ignore_ordinary") == ordinary("C3_process_ordinary") == ordinary("C5_target_ordinary")
    prompts = lambda n, k: sorted(str(r["prompt"]) for r in conds[n] if r["kind"] == k)  # noqa: E731
    assert prompts("C3_process_ordinary", "process") == prompts("C5_target_ordinary", "target")
    assert prompts("C2_ignore_fidelity", "fidelity") == prompts("C4_process_fidelity", "fidelity")


def test_truncated_or_empty_replies_are_dropped(builder, cfg):
    responses, reqs = fake_responses(builder, [0])
    bad = ap_.Responses([{"key": r["key"], "response": "x", "truncated": i % 2 == 0} for i, r in enumerate(reqs)])
    assert len(bad) == len(reqs) // 2
    with pytest.raises(RuntimeError):
        ap_.assemble_ignore(builder.rows, builder.ignore_specs(0), bad, 120)


def test_behavior_classes_balance_identity_transforms(builder):
    names = ["T01_copy", "T08_nfc", "T09_nfd", "T10_nfkc", "T11_strip", "T14_crlf_to_lf", "T03_upper", "T04_lower"]
    s = TransformSampler(builder.reg, names, random.Random(0), balance=True)
    assert len(s.classes) == 3           # {six identity-like}, {upper}, {lower}
    draws = [s.sample() for _ in range(3000)]
    assert abs(draws.count("T03_upper") / 3000 - 1 / 3) < 0.05
    assert s.sample(exclude="T03_upper") != "T03_upper"
    flat = TransformSampler(builder.reg, names, random.Random(0), balance=False)
    assert len(flat.classes) == len(names)


def test_training_rows_render(builder):
    for r in builder.conflict_rows(0)["process"][:20] + builder.conflict_rows(0)["boundary"][:20]:
        s = render_prompt(Prompt.from_dict(r["prompt"]))
        assert s.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert ("<|im_start|>input\n" in s) == (r["prompt"]["data"] is not None)
        assert s.count("<|im_start|>") == 3
