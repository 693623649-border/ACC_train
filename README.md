# ACC 4500: 2xA800 80G NVLink Long-Context Training

本仓库用于在 2 张 A800 80G NVLink 服务器上训练 ACC 4500 子集。训练语义保持为：

```text
一条 ACC 样本 = 一条完整 trajectory sequence
底层由 DeepSpeed-Ulysses SP2 沿 sequence 维切到 2 张卡共同训练
```

默认主路径使用 `Qwen/Qwen3-30B-A3B-Thinking-2507` BF16 基座。`Qwen/Qwen3-30B-A3B-Thinking-2507-FP8` 只作为非权重资产和未来支持 FP8 硬件的 smoke check 参考；A800 不默认使用 FP8 runtime。

## 1. 构建环境

基础镜像按需求使用：

```bash
docker build -t acc-qwen3-a800-sp2 -f docker/Dockerfile .
```

进入容器：

```bash
docker run --gpus all --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/workspace/ACC_Train" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/ACC_Train \
  -it acc-qwen3-a800-sp2
```

登录 Hugging Face：

```bash
huggingface-cli login
```

登录 Weights & Biases，用于训练曲线和指标展示：

```bash
wandb login
```

默认配置会把训练日志上报到 W&B，run 名称为 `acc-qwen3-a800-sp2`。如果服务器不能联网，可以使用离线模式：

```bash
export WANDB_MODE=offline
```

如果 `flash-attn` 在 CUDA13 镜像中编译失败，先确认容器里的 `nvcc --version` 与 `python -c "import torch; print(torch.__version__, torch.version.cuda)"` 一致，再重试：

```bash
MAX_JOBS=8 python -m pip install --no-build-isolation --force-reinstall "flash-attn>=2.7.0"
```

## 2. 预下载模型非权重内容

这一步可以在本机或服务器执行，只下载 tokenizer/config/index/README，不下载 safetensors 权重：

```bash
python scripts/download_model_assets.py \
  --repo-id Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --output-dir model_assets/Qwen3-30B-A3B-Thinking-2507-nonweights
```

如果也想顺手下载 FP8 reference 仓库的非权重内容，用：

```bash
python scripts/download_model_assets.py --include-fp8-reference
```

校验没有权重文件：

```bash
find model_assets/Qwen3-30B-A3B-Thinking-2507-nonweights \
  -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" -o -name "*.gguf" \)
```

该命令应无输出。

服务器上可以先检查当前 GPU 是否适合 Qwen FP8 runtime：

```bash
python scripts/check_qwen_fp8_runtime.py
```

A800/A100 预期会显示不支持 Qwen FP8 runtime，因此正式训练继续使用 BF16 主路径。

## 3. 下载 ACC 4500 子集

子集固定由 `paper/ACC_subset_4500_manifest.json` 决定：

```bash
python scripts/download_acc_subset.py \
  --manifest paper/ACC_subset_4500_manifest.json \
  --output-dir data/acc_subset_4500
```

期望输出：

```text
data/acc_subset_4500/search_agent_data_1k.jsonl
data/acc_subset_4500/sql_agent_data_1500.jsonl
data/acc_subset_4500/swe_agent_data_2k.jsonl
data/acc_subset_4500/manifest.json
```

快速计数：

```bash
wc -l data/acc_subset_4500/*.jsonl
```

期望 Search 1000、SQL 1500、SWE 2000，总数 4500。

## 4. Tokenize 与长度分桶

使用 Qwen chat template，保留 `<think>`，只训练 assistant token；超过 131072 token 的样本写入 reject report。

```bash
python scripts/tokenize_acc_subset.py \
  --manifest paper/ACC_subset_4500_manifest.json \
  --raw-dir data/acc_subset_4500 \
  --output-dir data/tokenized_acc_4500_qwen3_128k \
  --model-name-or-path Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --max-seq-length 131072
```

主要产物：

```text
data/tokenized_acc_4500_qwen3_128k/bucket_le_*.jsonl
data/tokenized_acc_4500_qwen3_128k/metadata.json
data/tokenized_acc_4500_qwen3_128k/rejected.jsonl
```

训练时动态 padding 到 batch 内最大长度的 8 倍数，不会把所有样本强行 pad 到 128K。

## 5. 训练配置

主配置在：

```text
configs/acc_qwen3_a800_sp2.yaml
configs/deepspeed_zero3_sp2.json
```

关键参数：

```text
model: Qwen/Qwen3-30B-A3B-Thinking-2507
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
optimizer: AdamW, beta1=0.9, beta2=0.999, weight_decay=0.1
LoRA: r=8, alpha=16, dropout=0.05, q/k/v/o
router: model.layers.*.mlp.gate trainable
report_to: wandb
run_name: acc-qwen3-a800-sp2
```

注意：A800 主训练不使用 FP8 checkpoint。代码会阻止在不支持 FP8 compute 的硬件上误把 FP8 repo 当训练模型。

## 6. Smoke Test

先跑短上下文 2 step，确认依赖、DeepSpeed、Trainer、LoRA/router 都能连起来：

```bash
bash scripts/launch_train_sp2.sh \
  --override data.tokenized_dir=data/tokenized_acc_4500_qwen3_128k \
  --override training.output_dir=outputs/smoke_8k \
  --override training.max_steps=2 \
  --override data.max_seq_length=8192
```

再跑最长桶 2 step。这个测试会暴露 131K + SP2 + FlashAttention2 的显存和通信问题：

```bash
bash scripts/launch_train_sp2.sh \
  --override data.tokenized_dir=data/tokenized_acc_4500_qwen3_128k \
  --override training.output_dir=outputs/smoke_128k \
  --override training.max_steps=2 \
  --override data.min_seq_length=114689
```

如果短测通过、长测 OOM，先不要改训练语义；优先检查：

```text
NCCL P2P 是否走 NVLink
flash-attn 是否真实启用
DeepSpeed-Ulysses 是否按 sp_size=2 生效
tokenized 样本是否存在异常超长
```

## 7. 正式 1 Epoch 训练

```bash
bash scripts/launch_train_sp2.sh \
  --override data.tokenized_dir=data/tokenized_acc_4500_qwen3_128k \
  --override training.output_dir=outputs/acc_qwen3_a800_sp2
```

从 checkpoint 恢复：

```bash
bash scripts/launch_train_sp2.sh \
  --resume-from-checkpoint outputs/acc_qwen3_a800_sp2/checkpoint-50
```

训练输出：

```text
outputs/acc_qwen3_a800_sp2/
  adapter_config.json
  adapter_model.safetensors
  router_gates.safetensors
  tokenizer.json
  tokenizer_config.json
  acc_training_config.json
```

## 8. 关键实现说明

- `scripts/download_acc_subset.py` 只按 manifest 行号筛选，不重新随机。
- `scripts/tokenize_acc_subset.py` 不截断长样本；超过 131072 直接拒收。
- `acc_train.dataset.ACCDataCollator` 做 batch 内动态 padding，并返回 `position_ids`。
- `acc_train.train` 使用 `ParallelismConfig(sp_backend="deepspeed", sp_size=2)`，让 2 卡共同训练一条 sequence。
- `acc_train.modeling` 默认冻结 backbone，只打开 LoRA 和 router gate。
- `configs/deepspeed_zero3_sp2.json` 使用 ZeRO-3 分片；SP2 负责长序列 activation 压力，ZeRO-3 负责参数/优化器状态压力。

## 9. 已知风险

- A800 不适合官方 Qwen FP8 runtime；默认使用 BF16 基座是为了可落地。
- DeepSpeed-Ulysses 的 HF 集成依赖较新版本 `transformers/accelerate/deepspeed`，镜像构建后应优先跑 smoke test。
- 论文原设置是 sequence parallelism 8 和 4 epochs；本方案按 2 卡设备改为 SP2 和 1 epoch，目标是工程可运行而不是完全复现论文算力规模。
