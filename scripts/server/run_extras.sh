#!/usr/bin/env bash
# After the one-seed run and the no-count diagnostic.
#  1. Gate 3 diagnostic: the base model on SecFid with instruction and data in ONE user message
#     (our runs put the data in a separate "input" turn). Is the low executed rate due to the format?
#  2. Gate 2 diagnostic: the same for the undefended Llama-3.1-8B-Instruct with its own chat
#     template. With the input turn it executed 14.8% of the probes; the paper reports 49.1%
#     and does not say how undefended models were prompted.
#  3. Gate 4 fallback named in the proposal: C1 with twice the security pairs (4,000 + 2,000).
# A failing step is reported and the next one still runs.
set -uo pipefail
cd /workspace/process-not-obey
set +u; source .venv/bin/activate; set -u
step() { echo; echo "=================== $(date -u +%H:%M:%S) $* ==================="; }
fail=0
# A failing step is tried once more after five minutes (the server's network dropped twice on
# Nov 18 for a few minutes; every step skips what is already done), then reported.
run() {
  "$@" && return 0
  echo "STEP FAILED (exit $?), retrying in 5 minutes: $*"; sleep 300
  "$@" || { echo "STEP FAILED again (exit $?): $*"; fail=1; }
}
step "base model, single user message"
run python -m pno.eval.run --name base_qwen35_2b_concat --renderer chatml --placement user_concat \
    --model Qwen/Qwen3.5-2B --backend vllm --benchmarks secfid secfid_clean
step "undefended Llama-3.1-8B-Instruct, single user message, its own chat template"
run python -m pno.eval.run --name llama31_8b_instr_concat --renderer hf --placement user_concat \
    --model meta-llama/Llama-3.1-8B-Instruct --backend vllm --benchmarks secfid secfid_clean
step "C1 with twice the security pairs"
run bash /workspace/run_c1_variant.sh sec2x 4000 1 1.25
step "done (failed steps: $fail)"
exit $fail
