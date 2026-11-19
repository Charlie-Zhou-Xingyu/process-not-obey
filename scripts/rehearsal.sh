#!/usr/bin/env bash
# Dress rehearsal: run every stage of the study end to end on a tiny random Qwen3.5 (CPU,
# roughly ten minutes), in a throwaway copy of the repo. The commands are the real ones
# (10_gates, 30_train, 40_eval, 50_report); only sizes and model paths differ. Run it after
# changing code and once on the server before spending GPU time.
#
#   bash scripts/rehearsal.sh [workdir]        # default: a new temporary directory
#   REHEARSAL_BENCHMARKS="secfid sep conflict" bash scripts/rehearsal.sh    # e.g. offline, without IFEval
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PYBIN=${PNO_PYTHON:-$ROOT/.venv/bin/python}
QWEN=${PNO_QWEN_LOCAL:-Qwen/Qwen3.5-2B}
[ -d "$QWEN" ] && QWEN=$(cd "$QWEN" && pwd)       # a local snapshot: make the path absolute
WORK=${1:-$(mktemp -d)}/pno-rehearsal
[ -d "$ROOT/third_party/sec-fid" ] || bash scripts/fetch_third_party.sh

rm -rf "$WORK" && mkdir -p "$WORK"
# leading "/" anchors the excludes at the repo root (src/pno/data must be copied)
rsync -a --exclude /.git --exclude /.venv --exclude /third_party --exclude /outputs --exclude /runs \
      --exclude /merged --exclude /data --exclude '__pycache__' --exclude '*.egg-info' ./ "$WORK"/
export PNO_THIRD_PARTY=$ROOT/third_party PYTHONPATH=$WORK/src PNO_PYTHON=$PYBIN PNO_GEN_PYTHON=$PYBIN
export PNO_BACKEND=hf PNO_DEVICE=cpu TOKENIZERS_PARALLELISM=false
cd "$WORK"
"$PYBIN" scripts/make_rehearsal.py "$WORK" "$QWEN"
if [ -n "${REHEARSAL_BENCHMARKS:-}" ]; then
  "$PYBIN" - "$REHEARSAL_BENCHMARKS" <<'PY'
import sys, yaml
c = yaml.safe_load(open("configs/experiment.yaml")); c["eval"]["benchmarks"] = sys.argv[1].split()
open("configs/experiment.yaml", "w").write(yaml.safe_dump(c, sort_keys=False))
PY
fi

step() { echo; echo "=================== $* ==================="; }
step "gates (backend check, competence, SecFid scoring, base model, C1)"; bash scripts/10_gates.sh
step "train the six conditions";                                          bash scripts/30_train.sh --seeds 0
step "evaluate runs, base model and references";                          bash scripts/40_eval.sh --seeds 0
step "report";                                                            bash scripts/50_report.sh
step "merge mode on one run"
rm -f outputs/eval/C2_ignore_fidelity-seed0/conflict*
PNO_ADAPTER_MODE=merge "$PYBIN" -m pno.pipeline eval --seeds 0 --conditions C2_ignore_fidelity --models \
    --benchmarks conflict --cleanup_merged
step "demo";  "$PYBIN" demo.py --model "$WORK/rehearsal_models/tiny-qwen" --adapter runs/C3_process_ordinary-seed0 \
    --device cpu --max_new_tokens 6 | tail -4
step "checks"
"$PYBIN" - <<'PY'
import json
from pathlib import Path
from pno.io import load_config
cfg = load_config()
benches = cfg["eval"].get("benchmarks") or ["secfid", "secfid_clean", "sep", "transfer", "ifeval", "conflict"]
missing = []
for cond in cfg["conditions"]:
    for b in benches:
        if not Path(f"outputs/eval/{cond}-seed0/{b}_summary.json").exists():
            missing.append(f"{cond}-seed0/{b}")
for m, spec in cfg["models"].items():
    for b in benches:
        if b == "conflict" and spec["renderer"] != "chatml":
            continue
        if not Path(f"outputs/eval/{m}/{b}_summary.json").exists():
            missing.append(f"{m}/{b}")
assert not missing, f"missing summaries: {missing}"
hyp = json.loads(Path("outputs/report/hypothesis.json").read_text())
need = {"secfid", "sep", "transfer"} <= set(benches)
assert hyp["complete"] or not need, hyp
assert Path("outputs/report/report.md").read_text().count("C1_ignore_ordinary") >= 1
assert json.loads(Path("merged/C2_ignore_fidelity-seed0/pno_merge_info.json").read_text())["verify_same_top1"]
assert not list(Path("merged/C2_ignore_fidelity-seed0").glob("*.safetensors")), "cleanup_merged left weights behind"
print(f"REHEARSAL PASSED: {len(cfg['conditions'])} runs and {len(cfg['models'])} models, benchmarks: {' '.join(benches)}")
PY
echo "workspace: $WORK"
