#!/usr/bin/env bash
# Rest of the one-seed run, reordered (Nov 19, 05:10 UTC) so that the headline benchmark comes
# first: SecFid for the six runs (primary comparison C3 vs C5 before the others), then gate 2
# (the two published Llama models on SecFid), then the other benchmarks of the six runs and of
# the base model. A failing step is reported and the next one still runs; every step skips
# what is already done. The 8B reference models' other benchmarks are in run_after.sh (last).
set -uo pipefail
cd /workspace/process-not-obey
set +u; source .venv/bin/activate; set -u
step() { echo; echo "=================== $(date -u +%H:%M:%S) $* ==================="; }
fail=0
run() { "$@" || { echo "STEP FAILED (exit $?): $*"; fail=1; }; }
P="python -m pno.pipeline eval"
SF="--benchmarks secfid secfid_clean"
REST="--benchmarks sep transfer ifeval conflict"

step "SecFid, primary comparison first: C3 and C5 (generate + score)"
run $P --seeds 0 --conditions C3_process_ordinary C5_target_ordinary --models $SF
step "SecFid: the other runs (generate + score)"
run $P --seeds 0 --models $SF
step "gate 2: Meta-SecAlign-8B on SecFid (generate + score)"
run $P --models meta_secalign_8b $SF
step "gate 2: undefended Llama-3.1-8B-Instruct on SecFid (generate + score)"
run $P --models llama31_8b_instr $SF
run python -m pno.pipeline gates
step "other benchmarks: six runs (generate + score)"
run $P --seeds 0 --models $REST
step "other benchmarks: base model"
run $P --models base_qwen35_2b $REST
step "partial SecFid scores"
run python -m pno.eval.secfid partial outputs/eval/*
step "report"
run bash scripts/50_report.sh
step "done (failed steps: $fail)"
exit $fail
