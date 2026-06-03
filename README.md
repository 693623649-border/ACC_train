# ACC 4500: 2xH20 FP8 SP2 Long-Context Training

本仓库用于在 2 张 NVIDIA H20 服务器上训练 ACC 4500 子集。默认训练路径是：

```text
Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
+ Native FP8 compute through Accelerate + Transformer Engine
+ frozen FP8 checkpoint backbone
+ LoRA/router trainable parameters without BF16 training switches
+ DeepSpeed-Ulysses SP2
+ max_seq_length 131072
```

训练语义保持为：一条 ACC 样本是一条完整 trajectory sequence，只是在底层由 SP2 沿 sequence 维切到 2 张 H20 上共同训练。不做 packing，不把所有样本强行 pad 到 128K。

## 1. 构建环境

```bash
docker build -t acc-qwen3-h20-fp8-sp2 -f docker/Dockerfile .
```

进入容器：

```bash
docker run --gpus all --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/workspace/ACC_Train" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/ACC_Train \
  -it acc-qwen3-h20-fp8-sp2
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

## 2. 检查 H20 原生 FP8 支持

Qwen 官方 FP8 checkpoint 要求 GPU compute capability 大于 8.9。当前训练路径还要求 NVIDIA Transformer Engine，因为本仓库使用 Accelerate `mixed_precision=fp8` + TE backend，而不是 BF16 autocast。

```bash
python scripts/check_qwen_fp8_runtime.py
```

期望所有可见 GPU 输出：

```text
required_mixed_precision=fp8
required_fp8_backend=te
transformer_engine_available=true
qwen_fp8_runtime_supported=true
all_visible_gpus_qwen_fp8_supported=true
native_fp8_training_ready=true
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
configs/acc_qwen3_h20_fp8_sp2.yaml
configs/deepspeed_zero3_fp8_sp2.json
configs/accelerate_h20_fp8_te.yaml
```

关键参数：

```text
model: Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
precision_mode: native_fp8_transformer_engine
mixed_precision: fp8
fp8_backend: TE
fp8_format: HYBRID
TrainingArguments bf16/fp16: false/false
DeepSpeed bf16/fp16: false/false
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
report_to: wandb
run_name: acc-qwen3-h20-native-fp8-sp2
```

## 7. Smoke Test

短上下文 2 step：

```bash
bash scripts/launch_train_h20_fp8_sp2.sh \
  --override data.tokenized_dir=data/tokenized_acc_4500_qwen3_fp8_128k \
  --override training.output_dir=outputs/smoke_h20_fp8_8k \
  --override training.max_steps=2 \
  --override data.max_seq_length=8192
```

最长桶 2 step：

```bash
bash scripts/launch_train_h20_fp8_sp2.sh \
  --override data.tokenized_dir=data/tokenized_acc_4500_qwen3_fp8_128k \
  --override training.output_dir=outputs/smoke_h20_fp8_128k \
  --override training.max_steps=2 \
  --override data.min_seq_length=114689
```

## 8. 正式 1 Epoch 训练

```bash
bash scripts/launch_train_h20_fp8_sp2.sh \
  --override data.tokenized_dir=data/tokenized_acc_4500_qwen3_fp8_128k \
  --override training.output_dir=outputs/acc_qwen3_h20_native_fp8_sp2
```

恢复训练：

```bash
bash scripts/launch_train_h20_fp8_sp2.sh \
  --resume-from-checkpoint outputs/acc_qwen3_h20_native_fp8_sp2/checkpoint-50
```

输出：

```text
outputs/acc_qwen3_h20_native_fp8_sp2/
  adapter_config.json
  adapter_model.safetensors
  router_gates.safetensors
  tokenizer.json
  tokenizer_config.json
  acc_training_config.json
```

## 9. 实现要点

- `acc_train/precision.py` 是精度实现模块：强制 `mixed_precision=fp8`、`fp8_backend=TE`，并拒绝 `bf16=true/fp16=true`。
- `scripts/launch_train_h20_fp8_sp2.sh` 使用 `accelerate launch --mixed_precision fp8 --fp8_backend te`，不再使用 `--mixed_precision bf16`。
- `configs/deepspeed_zero3_fp8_sp2.json` 显式关闭 DeepSpeed `bf16` 和 `fp16`，ZeRO-3 只负责分片和优化器状态管理。
- FP8 checkpoint 会检查所有可见 GPU 的 compute capability，非 FP8-capable GPU 会直接报错。
- backbone 显式冻结，只开启 LoRA 和 router gate；trainable 参数 dtype 默认为 `auto`，不再强制 cast 到 BF16。
- loss 使用 SP-aware chunked CE：优先消费 Ulysses 注入的 `shift_labels`，避免 shard 内自行 shift 丢失边界 token。
- 每个 CE chunk 通过 checkpoint 重算 `lm_head`，避免 autograd 为所有 chunk 保留 logits 激活。
- 全 -100 label chunk 也会执行 dummy `lm_head`，并把 chunk 次数跨 rank 对齐，降低 ZeRO-3 参数 gather 死锁风险。
- 跨 SP rank 的 loss 按有效 label token 加权聚合，适配 ACC/SFT 中 prompt token 被标记为 `-100` 的不均匀分布。
- ZeRO-3 下 router gate 保存只在 main process 执行，并在保存前 gather 分片参数。
- DeepSpeed-Ulysses SP2 负责长序列 attention 并行。

## 10. 已知服务器验证风险

- FP8 checkpoint + PEFT LoRA + Transformer Engine 对 Qwen3Moe quantized/linear modules 的兼容性必须通过 smoke test 验证。
- DeepSpeed-Ulysses 与 Qwen3Moe 的版本组合依赖较新 `transformers/accelerate/deepspeed`。
- 本地静态测试不能替代 H20 上的 8K 和 128K smoke test。
