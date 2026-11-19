#!/usr/bin/env bash
# One-seed runs with some transforms removed from the training pool, in a copy of the repo
# (/workspace/pno-<name>) so that the main study's data and runs stay untouched. Only the
# transform-based components change (process, target, boundary, fidelity). The ignore-style
# and ordinary components do not depend on the pool, so the main study's generated replies
# are reused and nothing is generated here.
#   run_keep_variant.sh <name> <regex of transform names to drop> <condition> [<condition> ...]
#   PREPARE_ONLY=1 run_keep_variant.sh ...     # stop after assembling the data (no GPU)
set -uo pipefail
NAME=$1; DROP=$2; shift 2
MAIN=/workspace/process-not-obey
VAR=/workspace/pno-$NAME
cd $MAIN
set +u; source .venv/bin/activate; set -u
step() { echo; echo "=================== $(date -u +%H:%M:%S) [$NAME] $* ==================="; }
fail=0
run() { "$@" || { echo "STEP FAILED (exit $?): $*"; fail=1; }; }

step "workspace"
mkdir -p $VAR
rsync -a --exclude /.git --exclude /.venv --exclude /third_party --exclude /outputs --exclude /runs \
      --exclude /merged --exclude /data --exclude /server_results --exclude '__pycache__' --exclude '*.egg-info' \
      --exclude '*.log' ./ $VAR/ || exit 1
mkdir -p $VAR/data/raw $VAR/data/generated $VAR/outputs/competence
cp data/raw/alpaca_cleaned.jsonl $VAR/data/raw/ || exit 1
cp data/generated/sep_reference.jsonl data/generated/responses.jsonl $VAR/data/generated/ || exit 1
rm -rf $VAR/outputs/competence/qwen3.5-2b $VAR/outputs/check_backend
cp -r outputs/competence/qwen3.5-2b $VAR/outputs/competence/ || exit 1
cp -r outputs/check_backend $VAR/outputs/ || exit 1
export PYTHONPATH=$VAR/src PNO_THIRD_PARTY=$MAIN/third_party
cd $VAR
python - "$DROP" <<'PY' || exit 1
import json, re, sys, yaml
drop = re.compile(sys.argv[1])
p = "outputs/competence/qwen3.5-2b/keep.json"
k = json.load(open(p))
before = list(k["keep"])
k["keep"] = [n for n in before if not drop.search(n)]
k["dropped_for_variant"] = [n for n in before if drop.search(n)]
k["n_keep"] = len(k["keep"])
assert k["dropped_for_variant"], "the pattern matched no kept transform"
assert len(k["keep"]) >= 4, "too few transforms left"
json.dump(k, open(p, "w"), indent=1)
c = yaml.safe_load(open("configs/experiment.yaml"))
c["seeds"] = [0]
open("configs/experiment.yaml", "w").write(yaml.safe_dump(c, sort_keys=False))
print("dropped:", k["dropped_for_variant"])
print("kept   :", k["keep"])
PY
step "data"
python -m pno.data.build assemble --seeds 0 || exit 1
python -c "
import pno.io, sys
assert str(pno.io.REPO_ROOT) == '$VAR', pno.io.REPO_ROOT       # everything below must stay inside the copy
"  || exit 1
if [ -n "${PREPARE_ONLY:-}" ]; then echo "prepared: $(ls data/generated/seed0/*.jsonl | wc -l) condition files in $VAR"; exit 0; fi
for c in "$@"; do
  step "train $c"
  run python -m pno.pipeline train --seeds 0 --conditions "$c"
  step "evaluate $c"
  run python -m pno.pipeline eval --seeds 0 --conditions "$c" --models --benchmarks secfid secfid_clean
done
step "done (failed steps: $fail)"
exit $fail
