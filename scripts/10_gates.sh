#!/usr/bin/env bash
# The four go/no-go checks of the proposal. Each step is restartable.
#   bash scripts/10_gates.sh            # all gates
#   GATES="1 3" bash scripts/10_gates.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
PY=${PNO_PYTHON:-python}; GEN_PY=${PNO_GEN_PYTHON:-python}; BACKEND=${PNO_BACKEND:-vllm}
BASE=$($PY -c "from pno.io import load_config; print(load_config()['base_model'])")
GATES=${GATES:-"1 2 3 4"}

for g in $GATES; do case $g in
1)  # which catalog transforms can the base model do? (also its untrained conflict behavior)
    [ -f outputs/competence/qwen3.5-2b/keep.json ] || \
      $GEN_PY -m pno.eval.competence --model "$BASE" --backend "$BACKEND" --n 25 --conflict \
        --out outputs/competence/qwen3.5-2b ;;
2)  # does our SecFid scoring reproduce the published Meta-SecAlign-8B / Llama-3.1-8B numbers?
    $PY -m pno.pipeline eval --models meta_secalign_8b llama31_8b_instr --benchmarks secfid ;;
3)  # is the base model vulnerable on SecFid, and can it do the tasks on clean inputs?
    $PY -m pno.pipeline eval --models base_qwen35_2b --benchmarks secfid secfid_clean ;;
4)  # does ignore-style training reproduce content dropping at this scale? (C1, seed 0)
    bash scripts/20_build_data.sh
    $PY -m pno.pipeline train --seeds 0 --conditions C1_ignore_ordinary
    $PY -m pno.pipeline eval --seeds 0 --conditions C1_ignore_ordinary --models --benchmarks secfid secfid_clean
    $PY -m pno.pipeline eval --models base_qwen35_2b --benchmarks secfid ;;
esac; done
$PY -m pno.pipeline gates
