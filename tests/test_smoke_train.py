"""End-to-end smoke test on a tiny random Qwen3.5: build pairs -> LoRA+DPO -> merge -> generate -> score.

Runs on CPU in about a minute. Needs a local Qwen3.5-2B snapshot for the config and
tokenizer (no weights are loaded from it); skipped otherwise.

    pytest tests/test_smoke_train.py -m slow -q
"""
import copy
import json

import pytest

from pno.io import load_config, read_jsonl, write_jsonl
from pno.train import dpo

pytestmark = pytest.mark.slow

DELM = {"mark": ["{s}", "[{s}]"], "inst": ["Command"], "inpt": ["Data"], "resp": ["Answer"],
        "user": ["", "User "], "asst": ["", "Assistant "]}


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("smoke")


@pytest.fixture(scope="module")
def tiny(qwen_snapshot, workdir):
    from tiny_model import make_tiny_qwen

    return make_tiny_qwen(qwen_snapshot, workdir / "tiny-qwen")


@pytest.fixture(scope="module")
def cfg_path(workdir):
    c = copy.deepcopy(load_config())
    c["data"].update(security_pairs=16, benign_pairs=16, keep_file=None, alpaca_cache=None)
    c["training"].update(pairs_per_condition=32, per_device_batch_size=4, effective_batch_size=8, max_length=384,
                         lora_r=8)
    c["eval"]["conflict"].update(n_per_type=4, max_new_tokens=16)
    p = workdir / "experiment.yaml"
    import yaml

    p.write_text(yaml.safe_dump(c))
    return str(p)


def test_train_merge_generate(tiny, cfg_path, workdir, toy_alpaca):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pno.data.build import Builder
    from pno.formatting import Prompt, render_prompt
    from pno.gen.backends import GenConfig, HFGenerator
    from pno.train import dpo, merge

    cfg = load_config(cfg_path)
    b = Builder(cfg, rows=toy_alpaca, delm_tokens=DELM)
    comp = b.components(0, None)
    rows = b.condition({"process": comp["process"], "fidelity": comp["fidelity"]},
                       {"security": "process", "benign": "fidelity"}, 0)
    train_file = workdir / "train.jsonl"
    write_jsonl(train_file, rows)

    run_dir = workdir / "run"
    dpo.main(["--train", str(train_file), "--model", tiny, "--out", str(run_dir), "--config", cfg_path,
              "--device", "cpu", "--seed", "0"])
    log = json.loads((run_dir / "train_log.json").read_text())
    assert log["max_steps"] == 4 and log["effective_batch"] == 8          # ceil(32 / 8)
    assert log["n_after_prepare"] == 32
    assert (run_dir / "adapter_config.json").exists()
    loss = log["train_metrics"]["train_loss"]
    assert loss == loss and 0 < loss < 10                                  # finite, not NaN

    # merged export keeps the base checkpoint layout and matches base + adapter
    merged_dir = workdir / "merged"
    info = merge.export_merged(tiny, str(run_dir), str(merged_dir))
    assert info["replaced"] > 0 and info["skipped"] in ([], ["lm_head.weight"])
    assert info["verify_max_abs_logit_diff"] < 1e-3 and info["verify_same_top1"]
    tok = AutoTokenizer.from_pretrained(merged_dir)
    msgs = Prompt("Uppercase the input.", "hello there").messages()
    assert tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) == \
        render_prompt(Prompt("Uppercase the input.", "hello there"))       # patched template accepts "input"

    prompt = render_prompt(Prompt.from_dict(rows[0]["prompt"]))
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
    base = AutoModelForCausalLM.from_pretrained(tiny, dtype=torch.float32)
    with torch.no_grad():
        base_logits = base(**ids).logits[0, -1]
        peft_logits = PeftModel.from_pretrained(base, str(run_dir))(**ids).logits[0, -1]
        merged_logits = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float32)(**ids).logits[0, -1]
    assert torch.allclose(peft_logits, merged_logits, atol=1e-3)           # float32 merge is near exact
    assert not torch.allclose(peft_logits, base_logits, atol=1e-4)         # the adapter did change the model

    gen = HFGenerator(str(merged_dir), device="cpu", batch_size=4)
    full = gen.generate_full([prompt, render_prompt(Prompt("Say hi."))], GenConfig(max_new_tokens=8))
    assert len(full) == 2 and all(set(r) == {"text", "truncated"} for r in full)


def test_train_with_dpo_plus_sft_loss(tiny, cfg_path, workdir):
    """The combined loss ([sigmoid, sft] with weights) is accepted by the installed TRL."""
    import yaml

    train_file = workdir / "train.jsonl"
    if not train_file.exists():
        pytest.skip("training smoke test did not run")
    c = yaml.safe_load(open(cfg_path))
    c["training"].update(loss_type=["sigmoid", "sft"], loss_weights=[1.0, 1.0])
    p = workdir / "experiment_sft.yaml"
    p.write_text(yaml.safe_dump(c))
    run_dir = workdir / "run_sft"
    dpo.main(["--train", str(train_file), "--model", tiny, "--out", str(run_dir), "--config", str(p),
              "--device", "cpu", "--seed", "0", "--max_steps", "2"])
    log = json.loads((run_dir / "train_log.json").read_text())
    assert log["training_cfg"]["loss_type"] == ["sigmoid", "sft"]
    loss = log["train_metrics"]["train_loss"]
    plain = json.loads((workdir / "run" / "train_log.json").read_text())["log_history"][0]["loss"]
    first = log["log_history"][0]["loss"]
    assert loss == loss and first > plain + 0.5          # ln 2 from DPO plus the response's NLL (random model: large)


def test_eval_runner_with_adapter(tiny, cfg_path, workdir, toy_alpaca, monkeypatch):
    """pno.eval.run end to end on the held-out conflict benchmark (hf backend, adapter loaded)."""
    from pno.data import build as build_mod
    from pno.eval import run as eval_run

    real_builder = build_mod.Builder
    monkeypatch.setattr(build_mod, "Builder", lambda cfg, *a, **k: real_builder(cfg, rows=toy_alpaca, delm_tokens=DELM))
    run_dir = workdir / "run"
    if not (run_dir / "adapter_config.json").exists():
        pytest.skip("training smoke test did not run")
    out = workdir / "eval"
    eval_run.main(["--name", "tiny-seed0", "--model", tiny, "--adapter", str(run_dir), "--backend", "hf",
                   "--device", "cpu", "--benchmarks", "conflict", "--out", str(out), "--config", cfg_path])
    scores = read_jsonl(out / "tiny-seed0" / "conflict_scores.jsonl")
    assert len(scores) == 12 and {s["type"] for s in scores} == {"user_data", "system_user", "fidelity"}
    summ = json.loads((out / "tiny-seed0" / "conflict_summary.json").read_text())
    assert summ["user_data"]["n"] == 4
    # restartable: a second call skips generation and only rescoring happens
    eval_run.main(["--name", "tiny-seed0", "--benchmarks", "conflict", "--out", str(out), "--config", cfg_path,
                   "--score_only"])
