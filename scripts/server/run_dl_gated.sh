#!/usr/bin/env bash
# Gated models, after `hf auth login`: the SecFid scorer and the undefended reference.
set -uo pipefail
cd /workspace/process-not-obey
set +u; source .venv/bin/activate; set -u
hf download google/embeddinggemma-300m > dl_embeddinggemma.log 2>&1; echo "gemma=$?" > .dl_gated_done
# safetensors only: the repo also holds the original .pth weights (another 16 GB)
hf download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*" > dl_llama.log 2>&1; echo "llama=$?" >> .dl_gated_done
