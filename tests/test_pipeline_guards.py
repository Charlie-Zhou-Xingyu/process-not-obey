"""Guards against silently reusing stale runs, outputs or configs (code review, Nov 19)."""
import hashlib
import json
import os
import time

import pytest

from pno import io, pipeline
from pno.catalog import build_registry
from pno.data.build import load_keep


def _cfg():
    return io.load_config()


def _run(tmp_path, training_cfg, data_bytes=b"rows\n", record_sha=True):
    data = tmp_path / "C1.jsonl"
    data.write_bytes(data_bytes)
    run = tmp_path / "run"
    run.mkdir()
    log = {"training_cfg": training_cfg}
    if record_sha:
        log["train_sha256"] = hashlib.sha256(data_bytes).hexdigest()
    (run / "train_log.json").write_text(json.dumps(log))
    return data, run


def test_matching_run_is_reused(tmp_path):
    cfg = _cfg()
    data, run = _run(tmp_path, dict(cfg["training"]))
    assert pipeline.stale_run_reason(cfg, data, run) is None


def test_logging_options_do_not_count(tmp_path):
    cfg = _cfg()
    data, run = _run(tmp_path, dict(cfg["training"], logging_steps=99, report_to=["wandb"]))
    assert pipeline.stale_run_reason(cfg, data, run) is None


def test_changed_loss_is_refused(tmp_path):
    cfg = _cfg()
    data, run = _run(tmp_path, dict(cfg["training"]))
    new = dict(cfg, training=dict(cfg["training"], loss_type=["sigmoid", "sft"], loss_weights=[1.0, 1.0]))
    why = pipeline.stale_run_reason(new, data, run)
    assert why and "loss_type" in why and "loss_weights" in why


def test_changed_data_is_refused(tmp_path):
    cfg = _cfg()
    data, run = _run(tmp_path, dict(cfg["training"]))
    data.write_bytes(b"other rows\n")
    assert "data changed" in pipeline.stale_run_reason(cfg, data, run)


def test_old_logs_without_data_hash_are_accepted(tmp_path):
    cfg = _cfg()
    data, run = _run(tmp_path, dict(cfg["training"]), record_sha=False)
    data.write_bytes(b"other rows\n")
    assert pipeline.stale_run_reason(cfg, data, run) is None


def test_pipeline_config_reaches_child_steps(tmp_path, monkeypatch):
    other = tmp_path / "variant.yaml"
    cfg = _cfg()
    cfg["training"]["loss_type"] = ["sigmoid", "sft"]
    import yaml

    other.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv(io.CONFIG_ENV, "")      # registers the variable, so teardown restores the original
    monkeypatch.delenv(io.CONFIG_ENV)
    pipeline.main(["status", "--config", str(other)])
    assert os.environ[io.CONFIG_ENV] == str(other.resolve())
    assert io.load_config()["training"]["loss_type"] == ["sigmoid", "sft"]   # what a child process reads
    monkeypatch.delenv(io.CONFIG_ENV)
    assert io.load_config()["training"]["loss_type"] == "sigmoid"


def test_regenerated_outputs_are_rescored(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "eval_dir", lambda name: tmp_path / name)
    out = tmp_path / "m"
    out.mkdir()
    for b in ("sep", "ifeval"):
        (out / f"{b}.jsonl").write_text("{}\n")
        (out / f"{b}_summary.json").write_text("{}")
    later = time.time() + 5
    os.utime(out / "sep.jsonl", (later, later))            # sep was regenerated after scoring
    (out / "secfid_summary.json").write_text("{}")          # secfid: output deleted, summary left
    pipeline.evaluate_one(_cfg(), "m", "model", "chatml", ["sep", "ifeval", "secfid"], dry=True)
    cmds = capsys.readouterr().out.splitlines()
    score = [c for c in cmds if "--score_only" in c]
    assert len(score) == 1 and "sep" in score[0] and "secfid" in score[0] and "ifeval" not in score[0]


def test_outputs_older_than_the_run_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "eval_dir", lambda name: tmp_path / "eval" / name)
    out = tmp_path / "eval" / "r"
    out.mkdir(parents=True)
    (out / "secfid.jsonl").write_text("{}\n")
    run = tmp_path / "run"
    run.mkdir()
    (run / "train_log.json").write_text("{}")
    later = time.time() + 5
    os.utime(run / "train_log.json", (later, later))       # retrained after the outputs were made
    with pytest.raises(SystemExit, match="older than the run"):
        pipeline.evaluate_one(_cfg(), "r", "model", "chatml", ["secfid"], dry=False, adapter=str(run))


def test_keep_list_required_when_asked(tmp_path):
    reg = build_registry()
    with pytest.raises(FileNotFoundError):
        load_keep(str(tmp_path / "missing.json"), reg, required=True)
    assert load_keep(str(tmp_path / "missing.json"), reg)      # smoke tests may fall back
