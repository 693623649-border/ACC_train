# 计划：双卡 A800 80G · BF16 原生 · 131K 不 OOM · 无 CPU offload

**状态：✅ 已执行（配置阶段，2026-06-17）** — §3 四个 A800 配置文件已创建 + `requirements.txt` 补齐 torch/flash-attn；代码兼容性已验证（无需改 `acc_train/*.py`）。tokenize / smoke / 正式训练按用户要求**未在本地执行**。
**日期：2026-06-17**
**分支：feature/h20-fp8-sp2**

---

## 1. 需求摘要（含硬约束）

| # | 约束 | 来源 |
|---|---|---|
| R1 | 模型基座改为 `Qwen/Qwen3-30B-A3B-Thinking-2507`（**非 FP8**，无需运行时反量化） | 用户明确 |
| R2 | 设备：双卡 **A800 80G**（Ampere sm80，架构/显存同 A100，不支持 FP8） | 用户明确 |
| R3 | **`max_seq_length` 必须保持 131072（128K）**，不得降低 | 用户明确 |
| R4 | **禁止 CPU offload**（`offload_optimizer/offload_param = none`） | 用户明确 |
| R5 | 修改 **GPU 分片策略** 防止 OOM | 用户明确 |
| R6 | 修改 **LoRA 可训练参数** 配合防 OOM | 用户明确 |
| R7 | 保留论文对齐能力（attention LoRA 对应 Figure 5a-b 注意力重构；router 对应 Figure 5c-d 专家特化） | 论文方法 |

---

## 2. 显存预算分析（可行性证明 + 唯一分片路径）

### 2.1 为什么只能选 ZeRO-3（排除 ZeRO-2/ZeRO-1）

Qwen3-30B-A3B BF16 权重 ≈ **61 GB**（30.5B × 2 bytes）。在 R3/R4 约束下逐项排除：

| 分片方案 | 单卡权重占用 | 单卡 80G 是否可行 | 结论 |
|---|---|---|---|
| ZeRO-0/1（仅分片优化器） | 61 GB（完整权重） | 80-61=19G 给激活 → **OOM** | ❌ |
| ZeRO-2（分片优化器+梯度） | 61 GB（完整权重） | 同上 → **OOM** | ❌ |
| **ZeRO-3（分片权重+梯度+优化器）** | **~30.5 GB**（61÷2） | 80-30.5=49.5G 给激活+通信 | ✅ **唯一可行** |
| TP2（张量并行） | 30.5 GB | 理论可行，但仓库无 TP 实现，需大改 | ⚠️ 超出范围 |

→ **结论：R4 禁止 offload 后，ZeRO-3 是唯一能让 61GB BF16 权重装进 80GB 的分片方案。** 用户 R5"修改分片策略"= 调优 ZeRO-3 的 partition 参数，而非更换 stage。

### 2.2 单卡峰值显存预算（131K, SP2→每卡 64K, ZeRO-3, grad-ckpt, chunked CE）

| 项 | 占用 | 依据 |
|---|---|---|
| 权重 ZeRO-3 分片 | **30.5 GB** | 61GB ÷ 2 卡 |
| 激活（grad-ckpt 存 48 层边界） | **~12.3 GB** | 64K × 2048 × 2B × 48 层 |
| 单层重算峰值（MoE dispatch+attn） | ~3-4 GB | flash-attn2 降 O(L²)→O(L)；grad-ckpt 层内独立重算 |
| ZeRO-3 param gather 临时 + bucket（调小后） | ~2-3 GB | reduce/prefetch bucket 2e7 |
| 可训练参数优化器状态（LoRA+gate ≈ 6.6-19M） | <0.3 GB | AdamW fp32，量级极小 |
| chunked CE logits（chunk=1024） | ~0.3 GB | 1024 × 151936 × 2B，checkpoint 重算 |
| CUDA context + 碎片 | ~2-3 GB | |
| **峰值合计** | **~51-57 GB** | **80GB 余量 ~23-29 GB** |

**判断：理论可行，余量充足。** 但最长桶（114K-131K）的 MoE dispatch buffer 与 ZeRO gather 临时峰值是主要风险点，必须靠 §3.2 的分片参数调优压低。

### 2.3 R6（减少 LoRA 参数）的真实收益——诚实评估

| 可训练配置 | 参数量 | 优化器状态 | 相对省显存 |
|---|---|---|---|
| q/k/v/o + router（现状） | 19.2 M | ~230 MB | 基准 |
| q/v + router | 15.9 M | ~190 MB | 省 ~40 MB |
| q/v only（关 router） | 3.3 M | ~40 MB | 省 ~190 MB |

→ **减少 LoRA 参数对 80GB 显存的收益是百 MB 级，不是 GB 级。** R6 的真实价值是"榨干每一 MB 的保守策略 + 降低 ZeRO-3 对 trainable param 的全副本开销"，而非主要防 OOM 手段。主防 OOM 的是 §3.2 分片调优。计划仍按 R6 执行收敛，但会在风险栏如实标注。

---

## 3. 实施步骤（具体文件 + 配置 diff）

> 规划阶段不改源文件；以下 diff 为计划内容，批准后由 executor 应用。
> 策略：**新建 A800 专用配置**（不动 A100/H20 原文件，保留可复现性）。

### 步骤 1：新建主配置 `configs/acc_qwen3_a800_bf16_sp2.yaml`

基于 [configs/acc_qwen3_a100_bf16_sp2.yaml](configs/acc_qwen3_a100_bf16_sp2.yaml) 修改 4 处：

```yaml
# (1) R1: 换原生 BF16 checkpoint，去除运行时反量化
model:
  name_or_path: Qwen/Qwen3-30B-A3B-Thinking-2507      # 去掉 -FP8 后缀
  precision_mode: native_bf16_ampere
  torch_dtype: bfloat16
  attn_implementation: flash_attention_2
  use_cache: false
  gradient_checkpointing: true                         # R3 防OOM 必备

# (2) 换 checkpoint 后需重新 tokenize（不同 tokenizer 路径）
data:
  tokenized_dir: data/tokenized_acc_4500_qwen3_bf16_128k
  max_seq_length: 131072                               # R3 保持 128K
  bucket_boundaries: [8192,16384,32768,49152,65536,81920,98304,114688,131072]

# (3) R6→合理最大化：attention 全投影 + router gate + rank 16（见 §9 LoRA 最大化决策）
lora:
  enabled: true
  r: 16                                      # 8→16，提升单层表达力（长上下文学习受益）
  alpha: 32                                  # 2×r 保持缩放
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]   # 论文 Figure 5a-b 注意力重构
  bias: none
router:
  train_router_gates: true                   # 开启，论文 Figure 5c-d 专家特化对齐

# (4) 指向新的 ZeRO-3 调优配置
training:
  deepspeed_config: configs/deepspeed_zero3_bf16_a800_sp2.json
  bf16: true
  fp16: false
  tf32: true                                # A800 支持 TF32，免费加速 matmul
  cross_entropy_chunk_size: 1024            # 保持，控 logits 峰值
  optim: adamw_torch                        # 可训练参数少，无需 8-bit

assets:
  non_weight_dir: model_assets/Qwen3-30B-A3B-Thinking-2507-nonweights
```

> **R6 决策说明**：关 router gate 会牺牲论文 Figure 5c-d 的"专家特化"对齐。若用户更看重论文完全对齐，可在批准时改为 `train_router_gates: true` + `target_modules: [q_proj, k_proj, v_proj, o_proj]`（回到 19.2M），§2.2 预算显示仍可行。本计划默认走保守防 OOM 档。

### 步骤 2：新建 ZeRO-3 调优配置 `configs/deepspeed_zero3_bf16_a800_sp2.json`

基于 [configs/deepspeed_zero3_bf16_a100_sp2.json](configs/deepspeed_zero3_bf16_a100_sp2.json)，**核心是压低 ZeRO-3 gather 的临时显存峰值**：

```jsonc
{
  "bf16": { "enabled": true },
  "fp16": { "enabled": false },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "none" },        // R4 禁止 CPU
    "offload_param":    { "device": "none" },         // R4 禁止 CPU
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 20000000,                   // 5e7→2e7，降通信峰值
    "stage3_prefetch_bucket_size": 20000000,          // 5e7→2e7
    "stage3_param_persistence_threshold": 100000,     // 1e6→1e5，更多大参数被分片不常驻
    "stage3_max_live_parameters": 300000000,          // 1e9→3e8，限制同时存活参数
    "stage3_max_reuse_distance": 300000000,           // 1e9→3e8
    "stage3_gather_16bit_weights_on_model_save": false
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "train_batch_size": "auto",
  "sequence_parallel_size": 2,                        // SP2 切序列，每卡 64K
  "zero_allow_untested_optimizer": true,
  "wall_clock_breakdown": false
}
```

### 步骤 3：新建 accelerate 配置 `configs/accelerate_a800_bf16_ds.yaml`

基于 [configs/accelerate_a100_bf16_ds.yaml](configs/accelerate_a100_bf16_ds.yaml)，仅改 deepspeed_config 指向：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
mixed_precision: bf16
num_processes: 2
num_machines: 1
deepspeed_config:
  deepspeed_config_file: configs/deepspeed_zero3_bf16_a800_sp2.json
  zero_stage: 3
  zero3_init_flag: true
  zero3_save_16bit_model: false
  offload_optimizer_device: none
  offload_param_device: none
```

### 步骤 4：新建启动脚本 `scripts/launch_train_a800_bf16_sp2.sh`

基于 [scripts/launch_train_a100_bf16_sp2.sh](scripts/launch_train_a100_bf16_sp2.sh)，改默认 CONFIG/ACCELERATE_CONFIG：

```bash
CONFIG="${CONFIG:-configs/acc_qwen3_a800_bf16_sp2.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_a800_bf16_ds.yaml}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"   # 长序列防碎片
export ACCELERATE_MIXED_PRECISION=bf16
unset ACCELERATE_FP8_BACKEND
unset ACCELERATE_FP8_FORMAT
```

### 步骤 5：适配 `scripts/download_model_assets.py`

支持下载 `Qwen3-30B-A3B-Thinking-2507`（非 FP8）的非权重资产到 `model_assets/Qwen3-30B-A3B-Thinking-2507-nonweights`。需确认该 checkpoint 在 HuggingFace 的 repo 名（预期即 `Qwen/Qwen3-30B-A3B-Thinking-2507`）。

### 步骤 6：重新 tokenize（换 checkpoint 后必做）

`tokenizer` 路径变化，需重跑 `scripts/tokenize_acc_subset.py`，输出到 `data/tokenized_acc_4500_qwen3_bf16_128k`。max_seq_length 保持 131072，超长样本进 rejected.jsonl。

### 步骤 7：代码兼容性确认（无需改源码，仅验证）

| 检查点 | 文件:行 | 结果 |
|---|---|---|
| 非 FP8 checkpoint 跳过硬件断言 | [modeling.py:45-46](acc_train/modeling.py#L45) | ✅ `"FP8" not in name` 直接 return |
| BF16 精度路径 | [precision.py:104-134](acc_train/precision.py#L104) | ✅ A800 sm80 满足 `visible_gpus_support_bf16` |
| SP2 整除校验 | [train.py:267-270](acc_train/train.py#L267) | ✅ num_heads(32)%2=0, num_kv_heads(4)%2=0 |
| pad_multiple(8)%sp(2) | [train.py:272](acc_train/train.py#L272) | ✅ |
| grad-ckpt + 冻结 backbone 反传 | [modeling.py:119-120](acc_train/modeling.py#L119) | ✅ `enable_input_require_grads` 已调用 |
| SP-aware chunked CE | [train.py:78-137](acc_train/train.py#L78) | ✅ 边界 token 不丢 |

---

## 4. 验收标准（testable）

- [ ] AC1: `configs/acc_qwen3_a800_bf16_sp2.yaml` 中 `name_or_path` 不含 `-FP8`，`offload_*` 均为 `none`
- [ ] AC2: `max_seq_length == 131072`，未被降低
- [ ] AC3: ZeRO stage=3，且 `reduce_bucket_size` / `prefetch_bucket_size` ≤ 2e7
- [ ] AC4: `python scripts/check_qwen_fp8_runtime.py` 在 A800 上不阻塞 BF16 路径（BF16 不依赖该脚本，但需确认 precision.py 不误报）
- [ ] AC5: **Smoke test 8K 2-step 跑通**，单卡峰值显存 < 60 GB（验证基本管线）
- [ ] AC6: **Smoke test 最长桶（min_seq_length=114689）2-step 跑通且不 OOM**，记录峰值显存（核心验收，证明 R3 可行）
- [ ] AC7: `trainable_parameters` 打印值 ≈ 3.3M（q/v LoRA，关 router）或按批准档位
- [ ] AC8: 正式 1-epoch 训练首 step loss 正常下降，无 NaN

---

## 5. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 最长桶 MoE dispatch buffer 峰值 OOM | 中 | 训练中断 | AC6 smoke test 先验证；若 OOM，阶梯降级见下 |
| ZeRO-3 init 阶段一次性加载权重 OOM | 低 | 启动失败 | `zero3_init_flag:true` + HfDeepSpeedConfig 已分区；`expandable_segments` 防碎片 |
| R6 关 router gate 牺牲论文专家特化对齐 | 中 | 效果略降 | 批准时可切回 `train_router_gates:true`（§2.2 显示仍可行） |
| 换 checkpoint 后 tokenizer/config 微小差异 | 低 | tokenize 偏差 | 步骤 6 重新 tokenize；对比 rejected 数量 |
| A800 NVLink 带宽低于 A100（特供版）致 SP2 通信慢 | 中 | 训练慢（非 OOM） | 不影响可行性；可接受速度代价 |

**OOM 降级阶梯**（仅在 AC6 失败时触发，每级保留 R3=131K、R4=无 offload）：
1. 进一步调小 bucket：2e7 → 1e7；`stage3_max_live_parameters` → 1e8
2. R6 再收紧：`target_modules: [q_proj]` only（单 attention 投影）
3. `gradient_checkpointing` 确认 `use_reentrant=False`（已是，[train.py:133](acc_train/train.py#L133)）
4. 最后兜底：与用户重新协商 R3（降 max_seq_length）——本计划不主动突破硬约束

---

## 6. 验证步骤（执行顺序）

1. `python scripts/download_model_assets.py`（步骤 5 适配后）→ 确认非权重资产就位、无 `.safetensors`
2. `python scripts/tokenize_acc_subset.py --max-seq-length 131072`（步骤 6）
3. **AC5 smoke**：
   ```bash
   bash scripts/launch_train_a800_bf16_sp2.sh \
     --override training.max_steps=2 --override data.max_seq_length=8192
   ```
4. **AC6 最长桶 smoke（核心）**：
   ```bash
   bash scripts/launch_train_a800_bf16_sp2.sh \
     --override training.max_steps=2 --override data.min_seq_length=114689
   ```
   记录 `nvidia-smi` 峰值；若 < 80GB 则 R3/R4 约束满足
5. AC7 确认日志 `trainable_parameters=...`
6. 通过后正式训练

---

## 7. 关键技术判断（给决策者）

1. **R4（无 CPU offload）+ R3（131K）+ 双卡 80G 的唯一解是 ZeRO-3 + SP2 + grad-ckpt + chunked CE 全开**，无其他分片方案可行（§2.1 已排除 ZeRO-1/2）。这不是"一种选择"，是约束求解的唯一解。
2. **减少 LoRA 参数（R6）对 80GB 显存收益是百 MB 级**，主要靠它防 OOM 是不现实的；它的真实作用是降低 ZeRO-3 trainable 副本 + 保守榨显存。真正防 OOM 的是 §3.2 的 ZeRO-3 partition 调优（压低 gather 临时峰值）。
3. 若 AC6 最长桶 smoke 实测峰值 > 75GB，应在降级阶梯 §5 内调整，而非突破 R3/R4。

---

## 8. 变更范围清单（批准后由 executor 执行）

| 动作 | 文件 | 类型 |
|---|---|---|
| 新建 | `configs/acc_qwen3_a800_bf16_sp2.yaml` | 新文件 |
| 新建 | `configs/deepspeed_zero3_bf16_a800_sp2.json` | 新文件 |
| 新建 | `configs/accelerate_a800_bf16_ds.yaml` | 新文件 |
| 新建 | `scripts/launch_train_a800_bf16_sp2.sh` | 新文件 |
| 适配 | `scripts/download_model_assets.py` | 改动（支持非 FP8 repo） |
| 不动 | `acc_train/*.py` | 无需改（代码已兼容 BF16） |
| 不动 | A100/H20 原配置 | 保留可复现性 |

---

## 9. LoRA 合理最大化决策（响应"合理最大化可训练 LoRA 参数"）

§2.2 显示 131K 下有 23-29GB 余量，LoRA 优化器状态仅百 MB 级，**显存不构成瓶颈**——"合理最大化"的边界由**训练合理性**而非显存决定。

**候选档位（按参数量递增）**：

| 档 | 配置 | 参数量 | 论文对齐 | 取舍 |
|---|---|---|---|---|
| 1 | q/v LoRA r=8 + 关 router | 3.34M | 部分（仅 attention 重构，丢专家特化） | 保守防 OOM |
| 2 | q/k/v/o r=8 + router | 19.22M | ✅ 论文原档（Figure 5 全对齐） | 基准 |
| **3** | **q/k/v/o r=16 + router** | **~26.0M** | ✅ 完整覆盖 + rank 翻倍 | **推荐（合理最大化）** |
| 4 | 档3 + MoE expert LoRA | 数 GB 级 | ❌ 超出合理边界 | 不推荐 |

**档3 参数量核算**（rank=16, heads=32, kv_heads=4, head_dim=128, hidden=2048, layers=48）：
- 每层 LoRA：q 98304 + k 40960 + v 40960 + o 98304 = 278528；×48 层 = **13.37M**
- router gate：每层 2048×128=262144；×48 层 = **12.58M**
- 合计 = **~25.95M（0.085% of 30.5B）**；AdamW fp32 优化器状态 ~416 MB，对 §2.2 余量无影响。

**为何档3 是合理最大化边界**：
1. attention 全投影（q/k/v/o）= 论文 Figure 5a-b 注意力重构，**必选**。
2. router gate = 论文 Figure 5c-d 专家特化，**必选**。
3. rank 8→16 = LoRA 常见的合理容量翻倍，对长上下文学习表达力提升；继续升 rank（32+）收益递减。
4. **不给 128 个 MoE experts 加 LoRA** —— 参数爆炸到 GB 级、与"冻结 backbone"理念冲突、收益不明确。

→ **计划已采用档3**（§3 步骤1）。若用户偏好更保守，回退档2（19.22M）；显存上两档均可。

---

## 10. 速度调优（无 CPU offload 约束下）

### 10.1 免费加速（默认已开启，无需权衡）
| 项 | 状态 | 收益 |
|---|---|---|
| `tf32: true` | ✅ | A800 sm80 支持，matmul ~1.5×，BF16 精度无损 |
| `attn_implementation: flash_attention_2` | ✅ | 长序列必需，attention O(L²)→O(L) |
| `overlap_comm` + `contiguous_gradients` | ✅ | ZeRO-3 通信/计算重叠 |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | ✅ | 利于 SP2 all-to-all 重叠 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | ✅ | 防碎片，减少显存浪费 |
| `gradient_checkpointing`（use_reentrant=False） | ✅ | 131K 防激活 OOM 必备（代价：~33% 前向重算） |

### 10.2 OOM↔速度 trade-off（smoke test 实测后反向调参）
§3 步骤2 为防 OOM 把 ZeRO-3 的 bucket/live_params 调小。§2.2 显示 23-29GB 余量，AC6 smoke 通过后这些"保守旋钮"可逐步放大提速：

| 旋钮 | 当前（防OOM） | 速度档 | 作用 |
|---|---|---|---|
| `reduce_bucket_size` | 2e7 | 5e7（若峰值<70G） | 减少 reduce round trips |
| `stage3_prefetch_bucket_size` | 2e7 | 5e7 | 减少 gather 次数 |
| `stage3_max_live_parameters` | 3e8 | 5e8 | 参数复用，减少重复 gather |
| `stage3_max_reuse_distance` | 3e8 | 5e8 | 同上 |
| `cross_entropy_chunk_size` | 1024 | 2048（logits 峰值有余量时） | 减少 lm_head 重算次数 |

**调参方法**：每级只调一个旋钮 → 跑 AC6 最长桶 smoke → 记录 `nvidia-smi` 峰值。峰值 < 72GB 则保留该档；逼近 75GB 则回退。目标峰值 ~70-72GB（留 8GB 碎片/波动余量）。这是把"防 OOM 的保守预算"逐步退让为"速度优先"的核心手段。

### 10.3 A800 特有瓶颈
A800 NVLink 带宽 400 GB/s（vs A100 的 600），SP2 的 all-to-all 与 ZeRO-3 gather 受影响 → 长序列通信占比升高。缓解：保持 `overlap_comm` + 合理 prefetch bucket（§10.2 的放大直接降通信占比）。

---

## 11. token 总量 / step 数 / 训练时间估算

> ⚠️ 数据未下载/未 tokenize（`data/` 为空），以下为**基于论文 Figure 3 分布的区间估算**，需以 §6 smoke 实测单 step 墙钟 × step 数 校准。

### 11.1 token 总量（估算）
4500 子集构成：Search 1000, SWE 2000, SQL 1500。按 Figure 3 长度特征估算平均序列长度：
- Search（多跳问答、证据长、长尾）：~35K
- SWE（代码 patch+干扰文件、偏短）：~12K
- SQL（表内容、中等）：~20K

总量 = 1000×35K + 2000×12K + 1500×20K = **~89M tokens**

| 场景 | 总量 | 说明 |
|---|---|---|
| 保守 | ~60M | 样本偏短 |
| 中位 | ~89M | 上述估算 |
| 乐观 | ~130M | Search 长尾占比高 |

### 11.2 step 数（确定值，与 token 无关）
- global_batch_size=16（per_device 1 × grad_accum 16 × dp 1；SP2 不增 batch 维度）
- **optimizer steps/epoch = ⌈4500/16⌉ = 282**
- micro-batch forward 次数/epoch = 4500（每样本一次前向）
- 论文训 4 epoch → 1128 steps；本子集默认 1 epoch = 282 steps。

### 11.3 训练时间估算（A800 80G BF16）
- 算力：A800 SXM = A100 同 die，312 TFLOPS dense BF16
- MoE active params/token = 3B（A3B）
- 每 token 前向+反向+grad-ckpt 重算 ≈ 6×3B×(1+0.33) ≈ **24 GFLOPS/token**
- 总 FLOPS ≈ 89M × 24G ≈ **2.14 EFLOPS**
- 实际 MFU（ZeRO-3+SP2+grad-ckpt+A800 NVLink 瓶颈）：~30-40%
- 有效算力 ≈ 312 × 0.35 ≈ 109 TFLOPS；纯计算 ≈ 2.14e18/1.09e14 ≈ **5.4h**

含通信/碎片墙钟（A800 NVLink 比 A100 低 33%）：

| 场景 | 1 epoch 耗时 | 条件 |
|---|---|---|
| 乐观 | ~6h | 短样本多、MFU 40%、通信重叠好 |
| 中位 | ~9-11h | 上述估算 |
| 保守 | ~14h | Search 长尾重、MFU 30%、A800 带宽瓶颈 |

4 epoch（论文设置）≈ 中位 ×4 ≈ 36-44h。**实际以 AC6 smoke 单 step 实测墙钟 × 282（×4 for 4-epoch）校准。**

---

**待批准。批准后按 §6 顺序执行，AC6 最长桶 smoke 不 OOM 即视为 R3/R4 核心约束达成。**
