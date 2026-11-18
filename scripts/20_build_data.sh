#!/usr/bin/env bash
# Build the six conditions for every seed. Needs gate 1 (keep.json) first.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
PY=${PNO_PYTHON:-python}; GEN_PY=${PNO_GEN_PYTHON:-python}; BACKEND=${PNO_BACKEND:-vllm}
read -r BASE MAXTOK TEMP <<<"$($PY -c "
from pno.io import load_config; c = load_config()
print(c['base_model'], c['data']['gen_max_new_tokens'], c['data']['gen_temperature'])")"

[ -f outputs/competence/qwen3.5-2b/keep.json ] || { echo "run gate 1 first: GATES=1 bash scripts/10_gates.sh"; exit 1; }
$PY -m pno.data.build requests
[ -f data/generated/responses.jsonl ] || \
  $GEN_PY -m pno.gen.run --requests data/generated/requests.jsonl --model "$BASE" \
    --out data/generated/responses.jsonl --backend "$BACKEND" --max_new_tokens "$MAXTOK" --temperature "$TEMP"
$PY -m pno.data.build assemble
