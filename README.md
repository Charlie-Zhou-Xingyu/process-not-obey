# Learning to Process, Not Obey

CS182 Fall 2026 final project (Topic 3, prompt-injection security). We ask whether security
training that teaches a model to process injected instructions as content, instead of
ignoring them, keeps security while losing less content. Base model: Qwen3.5-2B, LoRA + DPO,
six training conditions, three seeds. Main benchmark: SecFid.

- `PLAN.md` - the working plan: milestones, commands, go/no-go checks (in Chinese).
- `docs/DESIGN.md` - what each setting is and where it comes from.
- `configs/experiment.yaml` - every number the scripts use.

## Layout

```
src/pno/
  catalog/     99 instructions of the goal-goal paper + 14 sentence/line operations
  data/        pools, injection placement, six conditions (build.py)
  gen/         greedy generation: vLLM (server) or transformers (laptop)
  train/       dpo.py (TRL DPO + LoRA), merge.py (export in the base checkpoint layout)
  eval/        secfid, secfid_clean, sep, transfer, ifeval, conflict, competence; run.py
  stats/       paired item bootstrap, report.py
  pipeline.py  train / eval / status / gates over the whole grid
scripts/       00_setup_server .. 50_report
tests/         unit tests + CPU end-to-end smoke test on a tiny random Qwen3.5
```

## Laptop (no GPU needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,eval,dev]"
bash scripts/fetch_third_party.sh
pytest -q -m "not slow"                 # unit tests
pytest -q tests/test_smoke_train.py     # ~1 min on CPU; needs a local Qwen3.5-2B snapshot (PNO_QWEN_LOCAL)
python -m pno.data.build preview        # one real example of each generation-free component
```

## Server

```bash
bash scripts/00_setup_server.sh
hf auth login                 # access to meta-llama/Llama-3.1-8B-Instruct and google/embeddinggemma-300m
bash scripts/10_gates.sh      # four go/no-go checks; prints PASS / what to do otherwise
bash scripts/30_train.sh --seeds 0
bash scripts/40_eval.sh  --seeds 0 --cleanup_merged
bash scripts/50_report.sh     # outputs/report/report.md
```

Everything is restartable: finished generations, runs and scores are skipped.
`python -m pno.pipeline status` shows what exists; add `--dry_run` to `train` or `eval` to
print the commands without running them.

Two environments. If vLLM's pinned packages clash with the training stack, install vLLM
(plus `pip install -e .` and `datasets`) in a second venv and export
`PNO_GEN_PYTHON=/path/to/that/python`; generation then runs there and training and scoring
in the main one. `PNO_BACKEND=hf` avoids vLLM altogether (slower).

Disk. Each merged checkpoint is about 4.6 GB (18 runs). `--cleanup_merged` deletes a run's
merged weights once its outputs exist; adapters in `runs/` are kept and re-merged on demand.

## Third-party material

Nothing from other repositories is vendored. `scripts/fetch_third_party.sh` clones, at
pinned commits: SecFid (code MIT, data CC BY-SA 4.0), SEP, Meta-SecAlign (CC BY-NC 4.0) and
SecPO (Apache-2.0). We read their data files and parse literal constants; we do not import
their code. `src/pno/catalog/goalgoal_catalog.json` transcribes the instruction catalog of
arXiv:2610.35093 (CC BY 4.0); the implementations are ours. The SecFid scoring rules in
`src/pno/eval/secfid.py` re-implement the logic of that benchmark's `evaluate.py`
(MIT License, Copyright (c) 2026 The SecFid Authors).
