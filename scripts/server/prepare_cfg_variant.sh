#!/usr/bin/env bash
# Prepare /workspace/pno-<name> for run_cfg_variant.sh (no GPU). The code must already be there
# (rsync it from the laptop, so that code newer than the main copy on the server can be tried
# without touching the main copy). This script adds the main study's training data and gate
# outputs and applies config overrides given as  section.key=<yaml value>.
#   prepare_cfg_variant.sh <name> 'training.loss_type=[sigmoid, sft]' 'training.loss_weights=[1.0, 1.0]'
set -euo pipefail
NAME=$1; shift
MAIN=/workspace/process-not-obey
VAR=/workspace/pno-$NAME
[ -f $VAR/src/pno/io.py ] || { echo "no code in $VAR"; exit 1; }
cd $MAIN
set +u; source .venv/bin/activate; set -u
mkdir -p $VAR/data/raw $VAR/data/generated $VAR/outputs/competence
cp data/raw/alpaca_cleaned.jsonl $VAR/data/raw/
cp data/generated/sep_reference.jsonl data/generated/manifest.json $VAR/data/generated/
rm -rf $VAR/data/generated/seed0 $VAR/outputs/competence/qwen3.5-2b $VAR/outputs/check_backend
cp -r data/generated/seed0 $VAR/data/generated/
cp -r outputs/competence/qwen3.5-2b $VAR/outputs/competence/
cp -r outputs/check_backend $VAR/outputs/
export PYTHONPATH=$VAR/src PNO_THIRD_PARTY=$MAIN/third_party
cd $VAR
python - "$@" <<'PY'
import sys, yaml
p = "configs/experiment.yaml"
c = yaml.safe_load(open(p))
c["seeds"] = [0]
for ov in sys.argv[1:]:
    path, val = ov.split("=", 1)
    sec, key = path.split(".", 1)
    c[sec][key] = yaml.safe_load(val)
    print("override:", path, "=", c[sec][key])
open(p, "w").write(yaml.safe_dump(c, sort_keys=False))
PY
python - <<'PY'
import hashlib, pno.io
from pno.io import load_config
from pno.train.dpo import loss_spec
print("repo root:", pno.io.REPO_ROOT)
print("loss:", loss_spec(load_config()["training"]))
for c in ("C3_process_ordinary", "C4_process_fidelity"):
    a = hashlib.sha256(open(f"/workspace/process-not-obey/data/generated/seed0/{c}.jsonl", "rb").read()).hexdigest()[:16]
    b = hashlib.sha256(open(f"data/generated/seed0/{c}.jsonl", "rb").read()).hexdigest()[:16]
    print(f"{c}: training file identical to the main study: {a == b}")
PY
python -m pno.pipeline train --seeds 0 --conditions C4_process_fidelity --dry_run | tail -1 | cut -c1-300
