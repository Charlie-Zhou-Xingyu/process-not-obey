# Learning to Process, Not Obey

CS182 Fall 2026 final project (Topic 3, prompt-injection security). We ask whether security
training that teaches a model to process injected instructions as content, instead of
ignoring them, keeps security while losing less content. Base model: Qwen3.5-2B, LoRA + DPO,
six training conditions, three seeds. Main benchmark: SecFid.

- `PLAN.md` - the working plan: milestones, commands, go/no-go checks (in Chinese).
- `docs/WALKTHROUGH.md` - a guided tour of the code (in Chinese).
- `docs/DESIGN.md` - what each setting is and where it comes from.
- `configs/experiment.yaml` - every number the scripts use.

The layout follows the two repositories this work builds on, Meta-SecAlign and SecPO:
environment setup, a `demo.py`, data construction, training, evaluation.

# Environment setup

Hardware: one GPU with at least 24 GB of memory, Ampere or newer (bfloat16), 48 GB of RAM
and 100 GB of disk covers training Qwen3.5-2B and evaluating the three 8B references. The
NVIDIA driver must support CUDA 13.0 or newer: vLLM 0.34.0 pins torch 2.14.0, whose wheels
ship the CUDA 13.0 runtime (it also pins transformers below 5.23 and needs Python 3.11+).
A laptop is enough for the unit tests, the rehearsal and the data that needs no model.

```bash
git clone https://github.com/Charlie-Zhou-Xingyu/process-not-obey.git && cd process-not-obey
bash scripts/00_setup_server.sh     # venv, vLLM + TRL + PEFT, pinned third-party repos, unit tests
hf auth login                       # access to meta-llama/Llama-3.1-8B-Instruct and google/embeddinggemma-300m
```

Laptop:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,eval,dev]"
bash scripts/fetch_third_party.sh
pytest -q -m "not slow"                 # unit tests
pytest -q tests/test_smoke_train.py     # end to end on a tiny random Qwen3.5 (CPU; needs PNO_QWEN_LOCAL)
bash scripts/rehearsal.sh               # every server script, start to finish, on the tiny model (~10 min, CPU)
```

`scripts/rehearsal.sh` copies the repo to a temporary directory, shrinks every size in the
config (16 pairs per condition, a dozen items per benchmark) and then runs the real
`10_gates`, `30_train`, `40_eval` and `50_report` scripts with transformers on CPU. Run it
after changing code, and once on the server before the real runs, there as
`PNO_DEVICE=cuda bash scripts/rehearsal.sh`: the same run on the GPU, with bfloat16,
gradient checkpointing and the optimized Qwen3.5 kernels that real training uses.

The optimized kernels (`causal-conv1d`, `flash-linear-attention`, installed by
`00_setup_server.sh`) are CUDA-only, and transformers calls them on every device once they
are importable. Whenever our code picks a device other than CUDA (and for the merge, which
runs on CPU) it hides the two packages first, so that transformers uses its own PyTorch
implementation.

Model revisions are pinned in `configs/experiment.yaml` (`revisions`), as SecPO pins its
base model.

# Demo

`demo.py` runs two injection examples through the base model and, with `--adapter`, through
a trained run. For each it prints what executing, processing and ignoring the injected
sentence would look like, then the model's output and its label.

```bash
python demo.py
python demo.py --adapter runs/C3_process_ordinary-seed0
```

# Data

Six conditions, 2,000 security pairs + 2,000 benign pairs each (`docs/DESIGN.md`, section 1).

```bash
python -m pno.data.build preview    # one real example of each generation-free component
bash scripts/20_build_data.sh       # after gate 1: base-model targets, then all conditions and seeds
```

Process-style, target-control, boundary-control and fidelity pairs are computed by Python
from the instruction catalog (`src/pno/catalog/`). Ignore-style and ordinary pairs follow
Meta-SecAlign's SecAlign++ recipe and need the base model's own replies as targets.

# Training

LoRA + DPO with the settings of the goal-goal paper, which match SecPO's Qwen configuration
(rank 64, alpha 8, dropout 0.1, beta 0.1, learning rate 1.6e-4, one epoch, effective batch
64, linear schedule with 3% warmup).

```bash
bash scripts/10_gates.sh             # backend check + the proposal's four go/no-go checks
bash scripts/30_train.sh --seeds 0   # six conditions, one seed
```

# Evaluation

```bash
bash scripts/40_eval.sh --seeds 0    # trained runs, the base model and three references
bash scripts/50_report.sh            # outputs/report/report.md: tables, bootstrap contrasts, hypothesis check
```

- Benchmarks: SecFid (headline: safe processing), SecFid on clean inputs, SEP with the six
  static attacks of Meta-SecAlign and SecPO, a transfer attack built from SecPO's released
  injections, IFEval, and held-out goal-goal conflicts.
- Trained runs are served the way Meta-SecAlign evaluates its adapters: the base model in
  vLLM plus the LoRA adapter at run time. If vLLM cannot apply LoRA to Qwen3.5, merged
  float32 checkpoints are evaluated instead (about 7 GB each; `--cleanup_merged` deletes
  them once outputs exist). `python -m pno.tools.check_backend --adapter runs/<run>` (part
  of gate 4) finds out which works, and `eval.adapter_mode: auto` follows its answer;
  `PNO_ADAPTER_MODE=lora|merge` overrides it.
- `PNO_BACKEND=hf` replaces vLLM by plain transformers (slower); `PNO_DEVICE=cpu` forces a device. If vLLM's pinned packages
  clash with the training stack, install vLLM in a second venv and point
  `PNO_GEN_PYTHON` at it.
- Everything is restartable; `python -m pno.pipeline status` shows what exists and
  `--dry_run` prints commands without running them.

# Layout

```
src/pno/
  catalog/     99 instructions of the goal-goal paper + 14 sentence/line operations
  data/        pools, injection placement, six conditions (build.py)
  gen/         generation: vLLM (server) or transformers (laptop)
  train/       dpo.py (TRL DPO + LoRA), merge.py (merged export, merge mode only)
  eval/        secfid, secfid_clean, sep, transfer, ifeval, conflict, competence; run.py
  stats/       paired item bootstrap, report.py
  tools/       check_backend.py
  pipeline.py  train / eval / status / gates over the whole grid
scripts/       00_setup_server .. 50_report
tests/         unit tests + CPU end-to-end smoke test on a tiny random Qwen3.5
demo.py
```

# Acknowledgements and third-party material

The data recipe, prompt format and SEP protocol follow
[Meta-SecAlign](https://github.com/facebookresearch/Meta_SecAlign) (CC BY-NC 4.0) and
[SecPO](https://github.com/Sizhe-Chen/SecPO) (Apache-2.0); the benchmark is
[SecFid](https://github.com/mhermon/sec-fid) (code MIT, data CC BY-SA 4.0); SEP data come
from [its repository](https://github.com/egozverev/Should-It-Be-Executed-Or-Processed).
Nothing from these repositories is vendored: `scripts/fetch_third_party.sh` clones them at
pinned commits, and we read their data files and parse literal constants without importing
their code. `src/pno/catalog/goalgoal_catalog.json` transcribes the instruction catalog of
arXiv:2610.35093 (CC BY 4.0); the implementations are ours. The SecFid scoring rules in
`src/pno/eval/secfid.py` re-implement the logic of that benchmark's `evaluate.py`
(MIT License, Copyright (c) 2026 The SecFid Authors).
