#!/usr/bin/env bash
# Full SecFid scoring of the base model and C1 as soon as their outputs exist (gates 3 and 4).
set -uo pipefail
cd /workspace/process-not-obey
set +u; source .venv/bin/activate; set -u
while ! grep -q "train the other five conditions" seed0.log 2>/dev/null; do [ -f .seed0_done ] && break; sleep 20; done
while ! grep -q "gemma=0" .dl_gated_done 2>/dev/null; do sleep 10; done
python -m pno.pipeline eval --seeds 0 --conditions C1_ignore_ordinary --models base_qwen35_2b --benchmarks secfid secfid_clean \
  && python -m pno.pipeline gates
