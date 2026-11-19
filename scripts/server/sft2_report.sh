#!/usr/bin/env bash
# Results of the two anchored-loss follow-up runs (C1 and C7 with DPO + SFT), next to their plain-DPO counterparts.
cd /workspace/process-not-obey; set +u; source .venv/bin/activate; set -u
S=/workspace/pno-sft/outputs/eval; X=/workspace/pno-mixsft/outputs/eval; M=/workspace/pno-mix/outputs/eval
echo "sft2 marker: $(cat .sft2_done 2>/dev/null || echo missing)"; grep "===================\|STEP FAILED\|Traceback" sft2.log | cut -c1-150
python3 /workspace/summarize.py /workspace/pno-sft
python3 /workspace/summarize.py /workspace/pno-mixsft
python3 /workspace/summarize.py /workspace/pno-mix
python3 /workspace/summarize.py /workspace/process-not-obey 2>/dev/null | grep "model\|base_qwen35_2b \|C1_ignore\|C2_ignore"
python3 /workspace/bytask.py /workspace/pno-mixsft
python3 /workspace/bytask.py /workspace/pno-sft | grep "C1_ignore\|by task\|persuasion\|clean inputs"
python /workspace/pair_diff.py $S/C1_ignore_ordinary-seed0 C1_ignore_ordinary-seed0 $S/C1_ignore_ordinary-seed0 base_qwen35_2b $X/C7_mix_ordinary-seed0 $S/C1_ignore_ordinary-seed0 $X/C7_mix_ordinary-seed0 $M/C7_mix_ordinary-seed0 $X/C7_mix_ordinary-seed0 $S/C2_ignore_fidelity-seed0
python /workspace/sep_truncation.py $S/C1_ignore_ordinary-seed0 $X/C7_mix_ordinary-seed0 2>&1 | cut -c1-200
python /workspace/count_dist.py base_qwen35_2b C1_ignore_ordinary-seed0 $S/C1_ignore_ordinary-seed0 $X/C7_mix_ordinary-seed0 2>&1 | grep -v "^---"
python /workspace/conflict_matrix.py C1_ignore_ordinary-seed0 $S/C1_ignore_ordinary-seed0 $M/C7_mix_ordinary-seed0 $X/C7_mix_ordinary-seed0 2>&1 | grep "^\[\|trusted\|^ALL" | cut -c1-200
