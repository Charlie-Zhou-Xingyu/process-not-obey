#!/usr/bin/env bash
# After the one-seed run: what needed the Hugging Face login. Full SecFid scoring of every
# model, the undefended Llama reference on every benchmark, report, gate summary.
set -euo pipefail
cd /workspace/process-not-obey
set +u; source .venv/bin/activate; set -u
step() { echo; echo "=================== $(date -u +%H:%M:%S) $* ==================="; }
step "full evaluation (SecFid scoring for every model, Llama-3.1-8B-Instruct on every benchmark)"
bash scripts/40_eval.sh --seeds 0
step "report"
bash scripts/50_report.sh
step "gates"
python -m pno.pipeline gates
step "done"
