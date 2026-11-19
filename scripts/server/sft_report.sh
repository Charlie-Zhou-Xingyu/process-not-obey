#!/usr/bin/env bash
# Results of the anchored-loss diagnostic (DPO + SFT) next to the main study's runs (run on the server).
cd /workspace/process-not-obey; set +u; source .venv/bin/activate; set -u
V=/workspace/pno-sft/outputs/eval
echo "sft marker: $(cat .sft_done 2>/dev/null || echo missing)"; grep "===================\|STEP FAILED\|Traceback" sft.log | cut -c1-150
python3 /workspace/summarize.py /workspace/pno-sft
python3 /workspace/summarize.py /workspace/process-not-obey 2>/dev/null | grep "model\|base_qwen35_2b \|C2_ignore\|C4_process"
python3 /workspace/bytask.py /workspace/pno-sft
M="base_qwen35_2b C2_ignore_fidelity-seed0 $V/C2_ignore_fidelity-seed0 C4_process_fidelity-seed0 $V/C4_process_fidelity-seed0"
python /workspace/count_bias.py $M 2>&1 | sed -n '/=== counting/,/^--- clean item/p' | grep -v "^--- clean"
python /workspace/count_dist.py $M 2>&1 | grep -v "^---"
python /workspace/conflict_matrix.py C2_ignore_fidelity-seed0 $V/C2_ignore_fidelity-seed0 C4_process_fidelity-seed0 $V/C4_process_fidelity-seed0 2>&1 | cut -c1-200
python3 - <<'PY'
import json
for c in ("C4_process_fidelity", "C2_ignore_fidelity"):
    for root in ("/workspace/process-not-obey", "/workspace/pno-sft"):
        try:
            j = json.load(open(f"{root}/runs/{c}-seed0/train_log.json")); r = [x for x in j["log_history"] if "loss" in x][-10:]
            print(f"{root.split('/')[-1]:18s} {c}: last-10 loss {sum(x['loss'] for x in r)/10:.3f}  rewards/chosen {sum(x['rewards/chosen'] for x in r)/10:+.2f}  rewards/rejected {sum(x['rewards/rejected'] for x in r)/10:+.2f}  logps/chosen {sum(x['logps/chosen'] for x in r)/10:.1f}  seconds {j['seconds']:.0f}")
        except Exception as e:
            print(root, c, "no log:", e)
PY
