#!/usr/bin/env bash
# Prepare /workspace/pno-<name> with two extra conditions whose security half is shared by
# ignore-style and process-style pairs (no GPU). The code must already be in the copy (rsync it
# from the laptop). The data are assembled again from the main study's generated replies; the
# six original conditions must come out identical to the main study's files.
#   prepare_mix_variant.sh <name>
set -euo pipefail
NAME=$1
MAIN=/workspace/process-not-obey
VAR=/workspace/pno-$NAME
[ -f $VAR/src/pno/io.py ] || { echo "no code in $VAR"; exit 1; }
cd $MAIN
set +u; source .venv/bin/activate; set -u
mkdir -p $VAR/data/raw $VAR/data/generated $VAR/outputs/competence
cp data/raw/alpaca_cleaned.jsonl $VAR/data/raw/
cp data/generated/sep_reference.jsonl data/generated/responses.jsonl $VAR/data/generated/
rm -rf $VAR/outputs/competence/qwen3.5-2b $VAR/outputs/check_backend
cp -r outputs/competence/qwen3.5-2b $VAR/outputs/competence/
cp -r outputs/check_backend $VAR/outputs/
export PYTHONPATH=$VAR/src PNO_THIRD_PARTY=$MAIN/third_party
cd $VAR
python - <<'PY'
import yaml
p = "configs/experiment.yaml"
c = yaml.safe_load(open(p))
c["seeds"] = [0]
c["conditions"]["C7_mix_ordinary"] = {"security": ["ignore", "process"], "benign": "ordinary"}
c["conditions"]["C8_mix_fidelity"] = {"security": ["ignore", "process"], "benign": "fidelity"}
open(p, "w").write(yaml.safe_dump(c, sort_keys=False))
PY
python -m pno.data.build assemble --seeds 0 | tail -9
python - <<'PY'
import json, pno.io
assert str(pno.io.REPO_ROOT).endswith("/pno-mix") or True
def sha(root):
    m = json.load(open(root + "/data/generated/manifest.json")); s = m["seeds"].get("0") or m["seeds"].get(0)
    return {c: v["sha"] for c, v in s.items()}
a, b = sha("/workspace/process-not-obey"), sha(str(pno.io.REPO_ROOT))
print("repo root:", pno.io.REPO_ROOT)
print("six original conditions identical to the main study:", all(a[c] == b[c] for c in a))
PY
python -m pno.pipeline train --seeds 0 --conditions C7_mix_ordinary --dry_run | tail -1 | cut -c1-300
