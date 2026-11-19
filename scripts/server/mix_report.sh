#!/usr/bin/env bash
# Results of the two mixed conditions next to the main study's runs (run on the server).
cd /workspace/process-not-obey; set +u; source .venv/bin/activate; set -u
V=/workspace/pno-mix/outputs/eval
echo "mix marker: $(cat .mix_done 2>/dev/null || echo missing)"; grep "===================\|STEP FAILED\|Traceback" mix.log | cut -c1-150
python3 /workspace/summarize.py /workspace/pno-mix
python3 /workspace/summarize.py /workspace/process-not-obey 2>/dev/null | grep "model\|base_qwen35_2b \|C1_ignore\|C2_ignore\|C3_process\|C4_process"
python3 /workspace/bytask.py /workspace/pno-mix
python /workspace/pair_diff.py $V/C7_mix_ordinary-seed0 C1_ignore_ordinary-seed0 $V/C7_mix_ordinary-seed0 C3_process_ordinary-seed0 $V/C8_mix_fidelity-seed0 C2_ignore_fidelity-seed0 $V/C8_mix_fidelity-seed0 C4_process_fidelity-seed0 $V/C8_mix_fidelity-seed0 $V/C7_mix_ordinary-seed0
python /workspace/sep_truncation.py $V/C7_mix_ordinary-seed0 $V/C8_mix_fidelity-seed0 2>&1 | cut -c1-200
python /workspace/count_dist.py base_qwen35_2b C1_ignore_ordinary-seed0 C2_ignore_fidelity-seed0 $V/C7_mix_ordinary-seed0 $V/C8_mix_fidelity-seed0 2>&1 | grep -v "^---"
python /workspace/conflict_matrix.py C1_ignore_ordinary-seed0 C2_ignore_fidelity-seed0 $V/C7_mix_ordinary-seed0 $V/C8_mix_fidelity-seed0 2>&1 | cut -c1-200
