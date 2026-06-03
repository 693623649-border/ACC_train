#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/acc_qwen3_h20_fp8_sp2.yaml}"

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ACCELERATE_MIXED_PRECISION=fp8
export ACCELERATE_FP8_BACKEND=te

accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision fp8 \
  --fp8_backend te \
  -m acc_train.train \
  --config "${CONFIG}" \
  "$@"
