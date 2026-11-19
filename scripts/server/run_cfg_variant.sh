#!/usr/bin/env bash
# Train and evaluate conditions in a prepared copy of the repo (/workspace/pno-<name>) whose
# training config or code differs from the main study, on the main study's training data.
# The copy is prepared by prepare_cfg_variant.sh; nothing is synced from the main repo here.
#   run_cfg_variant.sh <name> "<benchmarks>" <condition> [<condition> ...]
set -uo pipefail
NAME=$1; BENCH=$2; shift 2
MAIN=/workspace/process-not-obey
VAR=/workspace/pno-$NAME
cd $MAIN
set +u; source .venv/bin/activate; set -u
step() { echo; echo "=================== $(date -u +%H:%M:%S) [$NAME] $* ==================="; }
fail=0
# A failing step is tried once more after five minutes (the server's network dropped twice on
# Nov 18 for a few minutes; every step skips what is already done), then reported.
run() {
  "$@" && return 0
  echo "STEP FAILED (exit $?), retrying in 5 minutes: $*"; sleep 300
  "$@" || { echo "STEP FAILED again (exit $?): $*"; fail=1; }
}
export PYTHONPATH=$VAR/src PNO_THIRD_PARTY=$MAIN/third_party
cd $VAR || exit 1
python -c "
import pno.io
assert str(pno.io.REPO_ROOT) == '$VAR', pno.io.REPO_ROOT       # everything below must stay inside the copy
" || exit 1
for c in "$@"; do
  step "train $c"
  run python -m pno.pipeline train --seeds 0 --conditions "$c"
  step "evaluate $c"
  run python -m pno.pipeline eval --seeds 0 --conditions "$c" --models --benchmarks $BENCH
done
step "done (failed steps: $fail)"
exit $fail
