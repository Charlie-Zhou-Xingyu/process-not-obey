#!/usr/bin/env bash
# Results of the no-count diagnostic next to the main study's runs (run on the server).
cd /workspace/process-not-obey; set +u; source .venv/bin/activate; set -u
V=/workspace/pno-nocount/outputs/eval
echo "nocount marker: $(cat .nocount_done 2>/dev/null || echo missing)"; grep "===================\|STEP FAILED\|Traceback" nocount.log | cut -c1-150
python3 /workspace/summarize.py /workspace/pno-nocount
python3 /workspace/summarize.py /workspace/process-not-obey | grep "model\|base_qwen35_2b \|C3_process\|C4_process\|C5_target"
python3 /workspace/bytask.py /workspace/pno-nocount
M="base_qwen35_2b C5_target_ordinary-seed0 C3_process_ordinary-seed0 $V/C3_process_ordinary-seed0 C4_process_fidelity-seed0 $V/C4_process_fidelity-seed0"
python /workspace/count_bias.py $M 2>&1 | sed -n '/=== counting/,/^--- clean item/p' | grep -v "^--- clean"
python /workspace/count_dist.py $M 2>&1 | grep -v "^---"
python /workspace/pair_diff.py $V/C3_process_ordinary-seed0 C3_process_ordinary-seed0 $V/C4_process_fidelity-seed0 C4_process_fidelity-seed0 $V/C3_process_ordinary-seed0 C5_target_ordinary-seed0
