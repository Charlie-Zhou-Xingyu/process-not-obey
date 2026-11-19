#!/usr/bin/env bash
# One-time setup on the GPU server (Linux, CUDA). Run from the repo root:
#   bash scripts/00_setup_server.sh
# Then: hf auth login   (token needs access to meta-llama/Llama-3.1-8B-Instruct and
#                        google/embeddinggemma-300m; accept both licenses on the Hub first)
set -euo pipefail
cd "$(dirname "$0")/.."

# vLLM needs Python >= 3.11; take the first of 3.11, 3.12, 3.13 (override with PYBIN=...)
if [ -z "${PYBIN:-}" ]; then
  for c in python3.11 python3.12 python3.13; do
    if command -v "$c" >/dev/null 2>&1; then PYBIN=$c; break; fi
  done
fi
[ -n "${PYBIN:-}" ] || { echo "no python3.11+ found; install one or set PYBIN"; exit 1; }
echo "using $($PYBIN --version) at $(command -v "$PYBIN")"
$PYBIN -m venv .venv
set +u; source .venv/bin/activate; set -u
python -m pip install -U pip

# vLLM first: it pins torch and transformers (vLLM 0.34.0: torch==2.14.0 with the CUDA 13.0
# runtime, transformers<5.23), so the NVIDIA driver must support CUDA 13.0 or newer. If its pins clash with the training stack
# (pip reports a conflict below), create a second venv for vLLM and export
# PNO_GEN_PYTHON=<that venv>/bin/python (README, Evaluation).
pip install "vllm>=0.34"
pip install -e ".[train,eval,dev]" "sentence-transformers>=3.0"
# optimized Gated DeltaNet kernels for Qwen3.5 in transformers (training speed); optional
pip install flash-linear-attention causal-conv1d || echo "WARN: optimized Qwen3.5 kernels not installed (training will be slower)"

bash scripts/fetch_third_party.sh
python -m pytest -q -m "not slow"
python - <<'PY'
import torch, transformers, trl, peft
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__)
PY
echo "Setup done. Next: hf auth login, then bash scripts/10_gates.sh"
