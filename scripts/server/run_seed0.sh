#!/usr/bin/env bash
# One-seed study without the gated Hugging Face models (no token on the server yet):
# gate 4 first, then the other five conditions, generation for every benchmark, scoring of
# everything that needs no gated model, and the partial SecFid scores. Every step is restartable.
set -euo pipefail
cd /workspace/process-not-obey
set +u; source .venv/bin/activate; set -u
step() { echo; echo "=================== $(date -u +%H:%M:%S) $* ==================="; }
REFS="base_qwen35_2b meta_secalign_8b secpo_llama31_8b"

step "gate 1 diagnostic (PLAN.md 6.13): trusted instructions with an output-format suffix"
[ -f outputs/competence/qwen3.5-2b_suffix/keep.json ] || python -m pno.eval.competence --model Qwen/Qwen3.5-2B \
    --backend vllm --out outputs/competence/qwen3.5-2b_suffix --trusted_suffix " Output only the result."
step "train C1 seed 0 (gate 4)"
python -m pno.pipeline train --seeds 0 --conditions C1_ignore_ordinary
step "backend check with the adapter: run-time LoRA or merged checkpoint?"
python -m pno.tools.check_backend --adapter runs/C1_ignore_ordinary-seed0
step "SEP reference replies"
[ -f data/generated/sep_reference.jsonl ] || python -m pno.eval.sep prepare --model Qwen/Qwen3.5-2B --backend vllm
step "gates 3 and 4: SecFid outputs of the base model and C1"
python -m pno.pipeline eval --seeds 0 --conditions C1_ignore_ordinary --models base_qwen35_2b --benchmarks secfid secfid_clean --no_score
python -m pno.eval.secfid partial outputs/eval/C1_ignore_ordinary-seed0 outputs/eval/base_qwen35_2b
step "train the other five conditions"
python -m pno.pipeline train --seeds 0
step "generate: six runs, base model, the two open references"
python -m pno.pipeline eval --seeds 0 --models $REFS --no_score
step "score the benchmarks that need no gated model"
python -m pno.pipeline eval --seeds 0 --models $REFS --benchmarks sep transfer ifeval conflict
step "partial SecFid scores"
python -m pno.eval.secfid partial outputs/eval/*
step "report"
bash scripts/50_report.sh
step "done"
