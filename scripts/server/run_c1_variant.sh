#!/usr/bin/env bash
# The ignore-style baseline (C1, seed 0) at a different training size, in its own copy of the
# repo (/workspace/pno-<name>) so that the main study's data and runs stay untouched.
#   run_c1_variant.sh <name> <security_pairs> <epochs> <oversample>
#   PREPARE_ONLY=1 run_c1_variant.sh ...     # stop after listing the generation requests (no GPU)
set -euo pipefail
NAME=$1; SEC=$2; EPOCHS=$3; OVER=$4
MAIN=/workspace/process-not-obey
VAR=/workspace/pno-$NAME
cd $MAIN
set +u; source .venv/bin/activate; set -u
step() { echo; echo "=================== $(date -u +%H:%M:%S) [$NAME] $* ==================="; }

step "workspace"
mkdir -p $VAR
rsync -a --exclude /.git --exclude /.venv --exclude /third_party --exclude /outputs --exclude /runs \
      --exclude /merged --exclude /data --exclude '__pycache__' --exclude '*.egg-info' --exclude '*.log' ./ $VAR/
mkdir -p $VAR/data/raw $VAR/data/generated $VAR/outputs/competence
cp data/raw/alpaca_cleaned.jsonl $VAR/data/raw/
cp data/generated/sep_reference.jsonl $VAR/data/generated/
rm -rf $VAR/outputs/competence/qwen3.5-2b $VAR/outputs/check_backend
cp -r outputs/competence/qwen3.5-2b $VAR/outputs/competence/
cp -r outputs/check_backend $VAR/outputs/
export PYTHONPATH=$VAR/src PNO_THIRD_PARTY=$MAIN/third_party
cd $VAR
python - "$SEC" "$EPOCHS" "$OVER" <<'PY'
import sys, yaml
sec, epochs, over = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
p = "configs/experiment.yaml"
c = yaml.safe_load(open(p))
c["seeds"] = [0]
c["data"].update(security_pairs=sec, oversample=over)
c["training"].update(pairs_per_condition=sec + c["data"]["benign_pairs"], epochs=epochs)
open(p, "w").write(yaml.safe_dump(c, sort_keys=False))
print({k: c["data"][k] for k in ("security_pairs", "benign_pairs", "oversample")},
      {k: c["training"][k] for k in ("pairs_per_condition", "epochs", "learning_rate")})
PY
step "targets"
python -m pno.data.build requests --seeds 0
if [ -n "${PREPARE_ONLY:-}" ]; then echo "prepared: $(wc -l < data/generated/requests.jsonl) requests"; exit 0; fi
[ -f data/generated/responses.jsonl ] || python -m pno.gen.run --requests data/generated/requests.jsonl \
    --model Qwen/Qwen3.5-2B --out data/generated/responses.jsonl --backend vllm --max_new_tokens 1024 --temperature 0.8 --seed 0
python -m pno.data.build assemble --seeds 0
step "train"
python -m pno.pipeline train --seeds 0 --conditions C1_ignore_ordinary
step "evaluate"
python -m pno.pipeline eval --seeds 0 --conditions C1_ignore_ordinary --models --benchmarks secfid secfid_clean sep
step "done"
