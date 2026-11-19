#!/usr/bin/env bash
# Last job of the queue: the three 8B reference models on the remaining benchmarks (not needed
# for the six-condition comparison), then anything still missing, the report and the gate
# summary. A failing step is reported and the next one still runs.
set -uo pipefail
cd /workspace/process-not-obey
set +u; source .venv/bin/activate; set -u
step() { echo; echo "=================== $(date -u +%H:%M:%S) $* ==================="; }
fail=0
run() { "$@" || { echo "STEP FAILED (exit $?): $*"; fail=1; }; }
P="python -m pno.pipeline eval"
step "Meta-SecAlign-8B: remaining benchmarks"
run $P --models meta_secalign_8b
step "undefended Llama-3.1-8B-Instruct: remaining benchmarks"
run $P --models llama31_8b_instr
step "SecPO reference: every benchmark"
run $P --models secpo_llama31_8b
step "anything still missing (six runs, base model, references)"
run bash scripts/40_eval.sh --seeds 0
step "report"
run bash scripts/50_report.sh
step "gates"
run python -m pno.pipeline gates
step "done (failed steps: $fail)"
exit $fail
