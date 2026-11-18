#!/usr/bin/env bash
# Tables, bootstrap contrasts and the hypothesis check from whatever has been evaluated so far.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
${PNO_PYTHON:-python} -m pno.stats.report --eval_dir outputs/eval --out outputs/report
${PNO_PYTHON:-python} -m pno.pipeline status
