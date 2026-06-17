#!/usr/bin/env bash
# =============================================================================
# A800 80G · 原生 BF16 · 131K · ZeRO-3 + Ulysses SP2 训练启动脚本
# 与 A100 BF16 路径环境变量一致（BF16 不依赖 FP8 backend）；
# 仅切换默认 CONFIG / ACCELERATE_CONFIG 指向 A800 配置。
# 用法： bash scripts/launch_train_a800_bf16_sp2.sh [--override ...]
# =============================================================================
set -euo pipefail

CONFIG="${CONFIG:-configs/acc_qwen3_a800_bf16_sp2.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_a800_bf16_ds.yaml}"

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"   # 利于 SP2 all-to-all 重叠
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # 长序列防碎片
export ACCELERATE_MIXED_PRECISION=bf16
unset ACCELERATE_FP8_BACKEND      # A800 不走 FP8
unset ACCELERATE_FP8_FORMAT

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  -m acc_train.train \
  --config "${CONFIG}" \
  "$@"
