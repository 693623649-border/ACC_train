# ACC 4500: 2xA100 BF16 SP2 Long-Context Training

本仓库用于在单机 2 张 NVIDIA A100-SXM4-40GB 上训练 ACC 4500 子集。默认训练路径是：

```text
Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
+ BF16 training compute on A100 through Accelerate + DeepSpeed
+ frozen Qwen FP8 checkpoint backbone
+ LoRA/router trainable parameters
+ DeepSpeed ZeRO-3 without CPU offload by default
+ DeepSpeed-Ulysses SP2
+ container image 2.11.0-cuda12.8-py3.12.3-devel
+ max_seq_length 131072
```

训练语义保持为：一条 ACC 样本是一条完整 trajectory sequence，只是在底层由 SP2 沿 sequence 维切到 2 张 A100 上共同训练。不做 packing，不把所有样本强行 pad 到 128K。

> 注意：2xA100 40GB 严格保持 131072 token 属于高显存风险配置。本仓库默认采用速度优先策略（ZeRO-3 + SP2，不启用 CPU offload），正式训练前必须跑 8K 与最长桶 128K smoke test；若 128K smoke OOM，再考虑增加显存资源、改用更多 sequence parallel 分片，或最后手动启用 CPU offload。本配置不会自动降级上下文长度。

## 1. 构建环境

默认 Dockerfile 基于服务器镜像：

```text
2.11.0-cuda12.8-py3.12.3-devel
```

```bash
docker build -t acc-qwen3-a100-bf16-sp2 -f docker/Dockerfile .
```

如果服务器上的完整镜像名包含 registry/namespace，或 `2.11.0-cuda12.8-py3.12.3-devel` 是某个镜像的 tag，用 `BASE_IMAGE` 覆盖：

```bash
docker build -t acc-qwen3-a100-bf16-sp2 \
  --build-arg BASE_IMAGE=<registry>/<image>:2.11.0-cuda12.8-py3.12.3-devel \
  -f docker/Dockerfile .
```

Dockerfile 默认安装 `flash-attn>=2.8.3`。如果服务器镜像已经内置匹配 Torch 2.11 / CUDA 12.8 / Python 3.12 的 flash-attn，可以跳过重装：

```bash
docker build -t acc-qwen3-a100-bf16-sp2 \
  --build-arg INSTALL_FLASH_ATTN=0 \
  -f docker/Dockerfile .
```

进入容器：

```bash
docker run --gpus all --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/workspace/ACC_Train" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/ACC_Train \
  -it acc-qwen3-a100-bf16-sp2
```

登录 Hugging Face 和 W&B：

```bash
huggingface-cli login
wandb login
```

离线 W&B：

```bash
export WANDB_MODE=offline
```

## 2. 检查 A100 BF16 运行时

A100 是 Ampere 架构，支持 BF16，但不支持本仓库原 H20 路径所需的原生 FP8 训练。当前默认路径仍以 `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8` 为 base model repo，但训练 compute 使用 `mixed_precision=bf16`，并通过 DeepSpeed ZeRO-3 做单机双卡分片。

```bash
python scripts/check_a100_bf16_runtime.py
```

期望输出包含：

```text
expected_container_image=2.11.0-cuda12.8-py3.12.3-devel
expected_python=3.12
expected_torch=2.11
expected_torch_cuda=12.8
python_312=true
torch_211=true
torch_cuda_128=true
expected_base_model=Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
native_fp8_compute_required=false
required_device_profile=2xA100-SXM4-40GB
required_mixed_precision=bf16
required_parallelism=deepspeed_zero3_sp2
all_visible_gpus_a100=true
all_visible_gpus_40gb_class=true
cuda_bf16_supported=true
a100_bf16_training_ready=true
```

## 3. 下载模型非权重资产

只下载 tokenizer/config/index/README，不下载真实权重分片：

```bash
python scripts/download_model_assets.py
```

默认输出：

```text
model_assets/Qwen3-30B-A3B-Thinking-2507-FP8-nonweights
```

确认没有权重文件：

```bash
find model_assets/Qwen3-30B-A3B-Thinking-2507-FP8-nonweights \
  -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" -o -name "*.gguf" \)
```

该命令应无输出。`model.safetensors.index.json` 只是索引文件，可以保留。

## 4. 下载 ACC 4500 子集

采样 manifest 已放在 `manifests/ACC_subset_4500_manifest.json`，不依赖 `paper/` 目录。

```bash
python scripts/download_acc_subset.py \
  --manifest manifests/ACC_subset_4500_manifest.json \
  --output-dir data/acc_subset_4500
```

期望：

```text
search_agent_data_1k.jsonl: 1000
sql_agent_data_1500.jsonl: 1500
swe_agent_data_2k.jsonl: 2000
total: 4500
```

## 5. Tokenize 与分桶

使用 Qwen chat template，优先使用 assistant mask 标注 labels；不支持时使用 offset fallback。保留 `<think>`，只训练 assistant token。超过 131072 token 的样本写入 reject report。

```bash
python scripts/tokenize_acc_subset.py \
  --manifest manifests/ACC_subset_4500_manifest.json \
  --raw-dir data/acc_subset_4500 \
  --output-dir data/tokenized_acc_4500_qwen3_fp8_128k \
  --model-name-or-path Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 \
  --max-seq-length 131072
```

产物：

```text
data/tokenized_acc_4500_qwen3_fp8_128k/bucket_le_*.jsonl
data/tokenized_acc_4500_qwen3_fp8_128k/index.jsonl
data/tokenized_acc_4500_qwen3_fp8_128k/metadata.json
data/tokenized_acc_4500_qwen3_fp8_128k/rejected.jsonl
```

`index.jsonl` 用于训练时快速读取 offset/length，避免初始化时解析巨大 token 数组。

## 6. 训练配置

主配置：

```text
configs/acc_qwen3_a100_bf16_sp2.yaml
configs/deepspeed_zero3_bf16_a100_sp2.json
configs/accelerate_a100_bf16_ds.yaml
```

关键参数：

```text
model: Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
precision_mode: native_bf16_ampere
mixed_precision: bf16
TrainingArguments bf16/fp16: true/false
DeepSpeed bf16/fp16: true/false
max_seq_length: 131072
num_train_epochs: 1
global_batch_size: 16
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
sp_size: 2
dp_replicate_size: 1
learning_rate: 1e-5
min_learning_rate: 1e-6
warmup_ratio: 0.05
cross_entropy_chunk_size: 1024
LoRA: r=8, alpha=16, dropout=0.05, q/k/v/o
router: model.layers.*.mlp.gate trainable, dtype=auto
ZeRO-3: optimizer/param CPU offload disabled by default
report_to: wandb
run_name: acc-qwen3-a100-bf16-sp2
```

## 7. Smoke Test

短上下文 2 step：

```bash
bash scripts/launch_train_a100_bf16_sp2.sh \
  --override data.tokenized_dir=data/tokenized_acc_4500_qwen3_fp8_128k \
  --override training.output_dir=outputs/smoke_a100_bf16_8k \
  --override training.max_steps=2 \
  --override data.max_seq_length=8192
```

最长桶 128K 2 step：

```bash
bash scripts/launch_train_a100_bf16_sp2.sh \
  --override data.tokenized_dir=data/tokenized_acc_4500_qwen3_fp8_128k \
  --override training.output_dir=outputs/smoke_a100_bf16_128k \
  --override training.max_steps=2 \
  --override data.min_seq_length=114689
```

如果最长桶 smoke test OOM，先不要启动正式训练；优先增加显存资源或提高 sequence parallel 分片数，最后再手动启用 CPU offload，或显式降低上下文长度。

## 8. 正式 1 Epoch 训练

```bash
bash scripts/launch_train_a100_bf16_sp2.sh \
  --override data.tokenized_dir=data/tokenized_acc_4500_qwen3_fp8_128k \
  --override training.output_dir=outputs/acc_qwen3_a100_bf16_sp2
```

恢复训练：

```bash
bash scripts/launch_train_a100_bf16_sp2.sh \
  --resume-from-checkpoint outputs/acc_qwen3_a100_bf16_sp2/checkpoint-50
```

输出：

```text
outputs/acc_qwen3_a100_bf16_sp2/
  adapter_config.json
  adapter_model.safetensors
  router_gates.safetensors
  tokenizer.json
  tokenizer_config.json
  acc_training_config.json
```

## 9. 实现要点

- `acc_train/precision.py` 支持 `native_bf16_ampere` 和旧的 `native_fp8_transformer_engine` 两条精度路径。
- A100 路径要求 `mixed_precision=bf16`、`bf16=true`、`fp16=false`，并拒绝非 Ampere/BF16-capable GPU。
- A100 路径使用 `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8` 作为 base model repo，但不启用原生 FP8 compute 或 Transformer Engine FP8 backend。
- `scripts/launch_train_a100_bf16_sp2.sh` 使用 `accelerate launch --mixed_precision bf16`，不传 FP8 backend。
- `configs/deepspeed_zero3_bf16_a100_sp2.json` 显式开启 DeepSpeed `bf16`，关闭 `fp16`，使用 ZeRO-3，默认不启用 optimizer/param CPU offload。
- `requirements.txt` 是 A100 BF16 默认依赖；旧 H20 FP8 路径如需 Transformer Engine，使用 `requirements-fp8.txt`。
- backbone 显式冻结，只开启 LoRA 和 router gate；trainable 参数 dtype 默认为 `auto`。
- loss 使用 SP-aware chunked CE：优先消费 Ulysses 注入的 `shift_labels`，避免 shard 内自行 shift 丢失边界 token。
- 每个 CE chunk 通过 checkpoint 重算 `lm_head`，避免 autograd 为所有 chunk 保留 logits 激活。
- 全 -100 label chunk 也会执行 dummy `lm_head`，并把 chunk 次数跨 rank 对齐，降低 ZeRO-3 参数 gather 死锁风险。
- 跨 SP rank 的 loss 按有效 label token 加权聚合，适配 ACC/SFT 中 prompt token 被标记为 `-100` 的不均匀分布。
- ZeRO-3 下 router gate 保存只在 main process 执行，并在保存前 gather 分片参数。

## 10. 已知服务器验证风险

- 2xA100-SXM4-40GB 上的 131072 token 正式训练有 OOM 风险，必须先跑 8K 和最长桶 128K smoke test。
- 当前服务器镜像目标为 `2.11.0-cuda12.8-py3.12.3-devel`；如果 Torch/CUDA/Python 版本不匹配，先修正镜像再跑 smoke test。
- `flash-attn` 必须与 Torch 2.11、CUDA 12.8、Python 3.12 匹配；默认 Dockerfile 会安装 `flash-attn>=2.8.3`，但服务器内置版本优先通过检查脚本确认。
- Qwen3 MoE BF16 checkpoint + PEFT LoRA + ZeRO-3 + Ulysses SP2 的组合需要在目标服务器验证。
- DeepSpeed-Ulysses 与 Qwen3Moe 的版本组合依赖较新 `transformers/accelerate/deepspeed`。
- 本地静态测试不能替代 A100 上的 smoke test。
- 旧 H20 FP8 配置和脚本仍保留在仓库中，用于需要原 FP8 路径时参考或回退。
