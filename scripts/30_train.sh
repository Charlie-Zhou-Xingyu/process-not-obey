#!/usr/bin/env bash
# Train + merge. Default: every condition and seed in the config.
#   bash scripts/30_train.sh --seeds 0                     # the six conditions, one seed
#   bash scripts/30_train.sh --seeds 1 2
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then set +u; source .venv/bin/activate; set -u; fi
${PNO_PYTHON:-python} -m pno.pipeline train "$@"
