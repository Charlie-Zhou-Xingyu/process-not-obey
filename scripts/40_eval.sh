#!/usr/bin/env bash
# Evaluate trained runs and the untrained / reference models on every benchmark.
#   bash scripts/40_eval.sh --seeds 0
#   bash scripts/40_eval.sh --models base_qwen35_2b        # only that model
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
PY=${PNO_PYTHON:-python}; GEN_PY=${PNO_GEN_PYTHON:-python}; BACKEND=${PNO_BACKEND:-vllm}
BASE=$($PY -c "from pno.io import load_config; print(load_config()['base_model'])")
# reference replies used as the fake answer in SEP completion attacks (once, shared by all models)
[ -f data/generated/sep_reference.jsonl ] || $GEN_PY -m pno.eval.sep prepare --model "$BASE" --backend "$BACKEND"
$PY -m pno.pipeline eval "$@"
