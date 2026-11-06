#!/usr/bin/env bash
# Start the Gemma 4 26B A4B classifier server (OpenAI-compatible API on 127.0.0.1:8000).
# Settings approved for this project; see docs/TECHNICAL.md section 4 for what each flag does.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM="${ROOT}/.venv/bin/vllm"
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
PORT="${PORT:-8000}"

# FlashInfer's sampler JIT-compiles CUDA code and needs nvcc, which this machine does not
# have (no CUDA toolkit, driver only). Use vLLM's PyTorch sampler instead; with
# temperature 0 the result is identical.
export VLLM_USE_FLASHINFER_SAMPLER=0

exec "${VLLM}" serve "${MODEL}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384 \
  --limit-mm-per-prompt.image 0 \
  --generation-config vllm \
  --async-scheduling \
  --api-server-count 4 \
  --host 127.0.0.1 \
  --port "${PORT}"
