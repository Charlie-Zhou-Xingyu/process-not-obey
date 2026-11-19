"""Regression tests for defects found in the 2026-11-18 code review."""
import copy
import random

import numpy as np
import pytest

from pno.catalog import build_registry
from pno.catalog.base import sensitive_names, split_sentences
from pno.data.alpaca import AlpacaRow, carrier_sentences, one_sentence, single_sentence_instructions
from pno.data.build import Builder
from pno.data.conflicts import TransformSampler
from pno.io import load_config, write_jsonl
from pno.textnorm import exact_match

DELM = {"mark": ["{s}", "[{s}]"], "inst": ["Command"], "inpt": ["Data"], "resp": ["Answer"],
        "user": ["", "User "], "asst": ["", "Assistant "]}
TEXT = "The harbor was quiet. Two engineers measured the bridge."


def test_exact_match_never_unwraps_the_reference():
    reg = build_registry()
    for name in ("T88_json_string", "T91_csv_field"):
        ref = reg[name].apply(TEXT)
        assert exact_match(ref, ref) and exact_match(ref + "\n", ref)
        assert not exact_match(TEXT, ref)                      # an unquoted copy is not the quoted answer
    doubled = reg["T25_double_chars"].apply(TEXT)
    assert not exact_match(doubled.replace("  ", " "), doubled)   # internal spacing is part of the answer
    # wrappers around an otherwise exact output are still forgiven
    up = TEXT.upper()
    assert exact_match(f"```\n{up}\n```", up) and exact_match(f"Output: {up}", up) and exact_match(f'"{up}"', up)
    assert exact_match("1. a  \n2. b\n\n", "1. a\n2. b") and not exact_match("1. a\n2. b", "1. a 2. b")


def test_carriers_are_single_sentences_for_the_sentence_transforms():
    assert not one_sentence("Call Dr. Lee at noon.") and not one_sentence("He spoke to Dr.")
    assert not one_sentence("The U.S. economy grew.") and one_sentence("The price rose 3.5 percent in May.")
    rows = [AlpacaRow(0, "x", "", "The clinic is run by Dr. Lee and two nurses. The town has a small harbor and a market."),
            AlpacaRow(1, "Name the capital of the U.S. state of Ohio.", "", "y"),
            AlpacaRow(2, "List three benefits of regular exercise.", "", "y")]
    sents = carrier_sentences(rows)
    assert sents == ["The town has a small harbor and a market."]
    assert all(split_sentences(s) == [s] for s in sents)
    assert single_sentence_instructions(rows) == ["List three benefits of regular exercise."]


def test_no_completion_placement_for_sentence_transforms(toy_alpaca):
    c = copy.deepcopy(load_config())
    c["data"].update(security_pairs=400, benign_pairs=10, keep_file=None, alpaca_cache=None,
                     injection_placement={"start": 0.1, "end": 0.1, "completion": 0.8})
    b = Builder(c, rows=toy_alpaca, delm_tokens=DELM)
    specs = b.conflict_specs(0, 400)
    by_cat = {}
    for s in specs:
        by_cat.setdefault(b.reg[s.t_user].category, set()).add(s.kind)
    assert by_cat["sentence"] <= {"start", "end"}
    assert any("completion" in kinds for cat, kinds in by_cat.items() if cat != "sentence")


def test_sampler_reports_an_exhausted_pool():
    reg = build_registry()
    s = TransformSampler(reg, ["T03_upper"], random.Random(0))
    assert s.sample() == "T03_upper"
    with pytest.raises(ValueError, match="at least two"):
        s.sample(exclude="T03_upper")


def test_sensitive_names_is_the_shared_filter():
    names = set(sensitive_names(build_registry()))
    assert {"T01_copy", "T03_upper", "number_sentences", "T72_number_lines"} <= names
    assert not {"T19_keep_decimals", "T67_first_run", "T87_even_length", "T74_odd_lines"} & names


def test_report_warns_about_partial_models(tmp_path):
    from pno.stats.report import build_report, to_markdown, hypothesis_check

    cfg = load_config()
    for name, n in (("C3_process_ordinary-seed0", 100), ("C5_target_ordinary-seed0", 100), ("base_qwen35_2b", 20)):
        d = tmp_path / name
        d.mkdir()
        write_jsonl(d / "sep_scores.jsonl", [{"id": f"s{i}", "secure": 1.0} for i in range(n)])
    rep = build_report(tmp_path, cfg)
    assert len(rep["warnings"]) == 1 and "base_qwen35_2b" in rep["warnings"][0] and "20 items" in rep["warnings"][0]
    assert "## Warnings" in to_markdown(rep, hypothesis_check(rep["contrasts"], 0.02))


def test_secfid_clean_counts_either_label_on_embedding_items():
    from pno import third_party
    try:
        third_party.repo("sec-fid")
    except FileNotFoundError:
        pytest.skip("third_party/sec-fid not fetched")
    from pno.eval import secfid, secfid_clean

    raw = {r["id"]: r for r in secfid.raw_rows()}
    items = [i for i in secfid_clean.load() if i["task"] == "translation"][:2] + \
            [i for i in secfid_clean.load() if i["id"] == "counting-000000"]
    first = lambda v: v if isinstance(v, str) else v[0]  # noqa: E731
    outs = [first(raw[items[0]["id"]]["eval"]["references"]["processed"]), "zzz qqq", "There are 4."]

    def bow(texts):
        import zlib
        m = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):
            for w in t.split():
                m[i, zlib.crc32(w.encode()) % 256] += 1
            m[i] /= max(np.linalg.norm(m[i]), 1e-9)
        return m

    per = secfid_clean.score(items, outs, embedder=bow)
    assert [p["correct"] for p in per] == [1.0, 0.0, 0.0]      # counting needs the clean count (3), not 4


def test_adapter_mode_auto_follows_the_backend_check(tmp_path, monkeypatch):
    import json

    from pno import pipeline

    cfg = copy.deepcopy(load_config())
    cfg["eval"]["adapter_mode"] = "auto"
    monkeypatch.setattr(pipeline, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("PNO_ADAPTER_MODE", raising=False)
    monkeypatch.setenv("PNO_BACKEND", "vllm")
    assert pipeline.adapter_mode(cfg) == "lora"                      # no report yet
    rep = tmp_path / "outputs/check_backend/report.json"
    rep.parent.mkdir(parents=True)
    rep.write_text(json.dumps({"recommended_adapter_mode": "merge"}))
    assert pipeline.adapter_mode(cfg) == "merge" and pipeline.trained_dtype(cfg) == "float32"
    rep.write_text(json.dumps({"recommended_adapter_mode": "lora"}))
    assert pipeline.adapter_mode(cfg) == "lora" and pipeline.trained_dtype(cfg) is None
    monkeypatch.setenv("PNO_BACKEND", "hf")                          # transformers applies adapters natively
    rep.write_text(json.dumps({"recommended_adapter_mode": "merge"}))
    assert pipeline.adapter_mode(cfg) == "lora"
    monkeypatch.setenv("PNO_ADAPTER_MODE", "merge")                  # explicit override wins
    assert pipeline.adapter_mode(cfg) == "merge"
    monkeypatch.setenv("PNO_ADAPTER_MODE", "bogus")
    with pytest.raises(SystemExit):
        pipeline.adapter_mode(cfg)


# ---- found on the server: CUDA-only kernels on CPU ----------------------------------------

def test_cpu_runs_hide_the_cuda_only_qwen_kernels():
    import importlib

    from pno.gen import backends

    assert backends.pick_device("cpu") == "cpu"
    for name in backends.GPU_ONLY_KERNELS:
        with pytest.raises(ImportError):
            importlib.import_module(name)


def test_reference_kernels_refuse_once_the_fast_ones_are_bound(monkeypatch):
    import sys
    import types

    from pno.gen import backends

    monkeypatch.setitem(sys.modules, "causal_conv1d", types.ModuleType("causal_conv1d"))
    monkeypatch.setitem(sys.modules, backends.QWEN35_MODELING, types.ModuleType("modeling"))
    with pytest.raises(RuntimeError, match="CUDA-only"):
        backends.use_reference_kernels()


def test_cpu_quota_reads_the_cgroup_limit(tmp_path):
    from pno.gen.backends import cpu_quota

    f = tmp_path / "cpu.max"
    f.write_text("2633142 100000\n")
    assert cpu_quota(str(f), visible=192) == 26       # the rented server: 192 visible cores, 26 usable
    assert cpu_quota(str(f), visible=16) is None      # quota above the visible cores: nothing to cap
    f.write_text("max 100000\n")
    assert cpu_quota(str(f), visible=192) is None     # no quota
    assert cpu_quota(str(tmp_path / "missing"), visible=192) is None


def test_loss_spec_accepts_one_loss_or_a_weighted_list():
    import pytest

    from pno.train.dpo import loss_spec

    assert loss_spec({}) == (["sigmoid"], None)
    assert loss_spec({"loss_type": "sigmoid"}) == (["sigmoid"], None)
    assert loss_spec({"loss_type": ["sigmoid", "sft"]}) == (["sigmoid", "sft"], None)       # TRL: weights of 1
    assert loss_spec({"loss_type": ["sigmoid", "sft"], "loss_weights": [1, 0.5]}) == (["sigmoid", "sft"], [1.0, 0.5])
    with pytest.raises(ValueError):
        loss_spec({"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0]})
    with pytest.raises(ValueError):
        loss_spec({"loss_type": []})



def test_a_condition_half_can_be_shared_by_two_components():
    import pytest

    from pno.data.build import Builder, half_rows

    comp = {k: [{"kind": k, "prompt": {"instruction": f"{k}{i}"}, "chosen": "a", "rejected": "b"} for i in range(4)]
            for k in ("ignore", "process", "ordinary", "fidelity")}
    assert half_rows(comp, "ignore") == comp["ignore"]                              # the six conditions: unchanged
    mixed = half_rows(comp, ["ignore", "process"])
    assert [r["prompt"]["instruction"] for r in mixed] == ["ignore0", "ignore1", "process0", "process1"]
    rows = Builder.condition(None, comp, {"security": ["ignore", "process"], "benign": "ordinary"}, 0)
    kinds = sorted(r["kind"] for r in rows)
    assert kinds == ["ignore"] * 2 + ["ordinary"] * 4 + ["process"] * 2 and len({r["id"] for r in rows}) == 8
    same = Builder.condition(None, comp, {"security": "process", "benign": "fidelity"}, 0)
    assert sorted(r["kind"] for r in same) == ["fidelity"] * 4 + ["process"] * 4
    with pytest.raises(ValueError):
        half_rows({"a": [1, 2, 3], "b": [1, 2, 3]}, ["a", "b"])                     # 3 rows cannot be halved
