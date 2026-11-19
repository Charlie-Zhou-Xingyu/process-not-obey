#!/usr/bin/env bash
# After the one-seed run and its full scoring.
#  1. Gate 3 diagnostic: the base model on SecFid with instruction and data in ONE user message
#     (our runs put the data in a separate "input" turn). Is the low executed rate due to the format?
#  2. Gate 4 fallback named in the proposal: C1 with twice the security pairs (4,000 + 2,000).
set -euo pipefail
cd /workspace/process-not-obey
set +u; source .venv/bin/activate; set -u
echo; echo "=================== $(date -u +%H:%M:%S) base model, single user message ==================="
python -m pno.eval.run --name base_qwen35_2b_concat --renderer chatml --placement user_concat \
    --model Qwen/Qwen3.5-2B --backend vllm --benchmarks secfid secfid_clean
bash /workspace/run_c1_variant.sh sec2x 4000 1 1.25
