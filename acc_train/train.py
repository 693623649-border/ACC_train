from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from acc_train.config import load_yaml_config, parse_override, set_by_dotted_key
from acc_train.dataset import ACCDataCollator, LengthBucketSampler, TokenizedJsonlDataset
from acc_train.modeling import (
    apply_lora_and_router_policy,
    count_trainable_parameters,
    load_causal_lm,
    save_router_gates,
)
from acc_train.precision import configure_native_fp8_precision, print_precision_plan


class ACCBucketTrainer(Trainer):
    def __init__(
        self,
        *args,
        bucket_boundaries: list[int] | None = None,
        seed: int = 42,
        min_learning_rate: float | None = None,
        base_learning_rate: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bucket_boundaries = bucket_boundaries or []
        self.bucket_seed = seed
        self.min_learning_rate = min_learning_rate
        self.base_learning_rate = base_learning_rate
        self.cross_entropy_chunk_size = int(getattr(self.args, "cross_entropy_chunk_size", 0) or 0)

    def _get_train_sampler(self):
        if self.train_dataset is None:
            return None
        if not hasattr(self.train_dataset, "lengths"):
            return super()._get_train_sampler()
        return LengthBucketSampler(
            lengths=self.train_dataset.lengths,
            boundaries=self.bucket_boundaries,
            seed=self.bucket_seed,
            shuffle=True,
        )

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        optimizer = optimizer or self.optimizer
        if optimizer is None:
            raise RuntimeError("Optimizer must be created before scheduler.")

        base_lr = float(self.base_learning_rate or self.args.learning_rate)
        min_lr = float(self.min_learning_rate or 0.0)
        min_ratio = min(max(min_lr / base_lr, 0.0), 1.0) if base_lr > 0 else 0.0
        warmup_steps = self.args.get_warmup_steps(num_training_steps)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_ratio + (1.0 - min_ratio) * cosine

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        chunk_size = int(getattr(self, "cross_entropy_chunk_size", 0) or 0)
        if chunk_size <= 0:
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            except TypeError:
                return super().compute_loss(model, inputs, return_outputs=return_outputs)

        labels = inputs["labels"]
        shift_labels = inputs.get("shift_labels")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        position_ids = inputs.get("position_ids")
        causal_lm = unwrap_causal_lm(model)
        transformer = getattr(causal_lm, "model", None)
        lm_head = getattr(causal_lm, "lm_head", None)
        if transformer is None or lm_head is None:
            raise RuntimeError("Chunked CE requires a CausalLM with `.model` and `.lm_head` attributes.")

        outputs = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if shift_labels is None:
            shift_hidden = hidden_states[:, :-1, :]
            shift_labels = labels[:, 1:]
        else:
            shift_hidden = hidden_states
        shift_hidden = shift_hidden.contiguous()
        shift_labels = shift_labels.contiguous()

        total_loss = shift_hidden.new_zeros(())
        total_tokens = shift_labels.ne(-100).sum()
        seq_len = shift_hidden.shape[1]
        num_chunks = max(math.ceil(seq_len / chunk_size), 1)
        num_chunks = max(num_chunks, distributed_max_int(num_chunks))
        for chunk_index in range(num_chunks):
            start = chunk_index * chunk_size
            end = min(start + chunk_size, seq_len)
            if start < seq_len:
                chunk_hidden = shift_hidden[:, start:end, :]
                chunk_labels = shift_labels[:, start:end]
            else:
                chunk_hidden = shift_hidden[:, :1, :] * 0
                chunk_labels = shift_labels.new_full((shift_labels.shape[0], 1), -100)
            loss_fn = lambda hidden, labels: chunked_lm_head_ce_sum(hidden, labels, lm_head)
            total_loss = total_loss + checkpoint(loss_fn, chunk_hidden, chunk_labels, use_reentrant=False)
        loss = normalize_sequence_parallel_loss(total_loss, total_tokens, self.sequence_parallel_group())
        if return_outputs:
            return loss, {"loss": loss}
        return loss

    def sequence_parallel_group(self):
        accelerator = getattr(self, "accelerator", None)
        mesh = getattr(accelerator, "torch_device_mesh", None)
        if mesh is not None:
            try:
                return mesh["sp"].get_group()
            except Exception:
                pass
        try:
            from deepspeed.utils import groups
        except Exception:
            return None
        try:
            return groups._get_sequence_parallel_group()
        except Exception:
            return None


def unwrap_causal_lm(model):
    current = model
    while hasattr(current, "module"):
        current = current.module
    if hasattr(current, "get_base_model"):
        current = current.get_base_model()
    while hasattr(current, "module"):
        current = current.module
    return current


def chunked_lm_head_ce_sum(hidden_states: torch.Tensor, labels: torch.Tensor, lm_head) -> torch.Tensor:
    logits = lm_head(hidden_states)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


def distributed_max_int(value: int) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return int(value)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor(int(value), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def normalize_sequence_parallel_loss(loss_sum: torch.Tensor, good_tokens: torch.Tensor, sp_group) -> torch.Tensor:
    local_good_tokens = good_tokens.to(device=loss_sum.device, dtype=loss_sum.dtype)
    local_loss = loss_sum / local_good_tokens.clamp_min(1)
    if sp_group is None or not dist.is_available() or not dist.is_initialized():
        return local_loss
    try:
        sp_world_size = dist.get_world_size(group=sp_group)
    except Exception:
        return local_loss
    if sp_world_size <= 1:
        return local_loss

    losses_per_rank = dist_nn.all_gather(local_loss, group=sp_group)
    tokens_per_rank = dist_nn.all_gather(local_good_tokens, group=sp_group)
    total_loss = loss_sum.new_zeros(())
    total_good_tokens = loss_sum.new_zeros(())
    for rank in range(sp_world_size):
        rank_tokens = tokens_per_rank[rank]
        total_good_tokens = total_good_tokens + rank_tokens
        if rank_tokens.item() > 0:
            total_loss = total_loss + losses_per_rank[rank] * rank_tokens
    return total_loss / total_good_tokens.clamp_min(1)


def build_parallelism_config(config: dict[str, Any]):
    parallelism = config.get("parallelism", {})
    if parallelism.get("backend") != "deepspeed_ulysses":
        return None
    try:
        from accelerate.utils import DeepSpeedSequenceParallelConfig, ParallelismConfig
    except ImportError as exc:
        raise RuntimeError(
            "DeepSpeed-Ulysses SP requires accelerate with ParallelismConfig support. "
            "Install accelerate>=1.12.0 in the CUDA13/Torch2.7 image."
        ) from exc

    return ParallelismConfig(
        sp_backend=parallelism.get("sp_backend", "deepspeed"),
        sp_size=int(parallelism.get("sp_size", 2)),
        dp_replicate_size=int(parallelism.get("dp_replicate_size", 1)),
        sp_handler=DeepSpeedSequenceParallelConfig(
            sp_seq_length_is_variable=bool(parallelism.get("sp_seq_length_is_variable", True)),
            sp_attn_implementation=parallelism.get("sp_attn_implementation", "flash_attention_2"),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3 MoE on ACC 4500 with SP2 long-context SFT.")
    parser.add_argument("--config", default="configs/acc_qwen3_h20_fp8_sp2.yaml")
    parser.add_argument("--override", action="append", default=[], help="Dotted YAML override, e.g. training.max_steps=2")
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for raw in overrides:
        key, value = parse_override(raw)
        set_by_dotted_key(config, key, value)
    return config


def validate_training_config(config: dict[str, Any], model) -> None:
    training_cfg = config["training"]
    parallelism = config.get("parallelism", {})
    per_device = int(training_cfg["per_device_train_batch_size"])
    grad_accum = int(training_cfg["gradient_accumulation_steps"])
    dp_size = int(parallelism.get("dp_replicate_size", 1))
    expected_global = per_device * grad_accum * dp_size
    configured_global = int(training_cfg.get("global_batch_size", expected_global))
    if expected_global != configured_global:
        raise ValueError(
            "global_batch_size should count data-parallel replicas, not SP ranks. "
            f"Expected {expected_global} = per_device({per_device}) * grad_accum({grad_accum}) * dp({dp_size}), "
            f"got {configured_global}."
        )

    sp_size = int(parallelism.get("sp_size", 1))
    num_heads = int(getattr(model.config, "num_attention_heads", 0))
    num_kv_heads = int(getattr(model.config, "num_key_value_heads", sp_size))
    if num_heads and num_heads % sp_size != 0:
        raise ValueError(f"num_attention_heads={num_heads} must be divisible by sp_size={sp_size}.")
    if num_kv_heads and num_kv_heads % sp_size != 0:
        raise ValueError(f"num_key_value_heads={num_kv_heads} must be divisible by sp_size={sp_size}.")
    pad_to_multiple_of = int(config.get("data", {}).get("pad_to_multiple_of", 1))
    if sp_size > 1 and pad_to_multiple_of % sp_size != 0:
        raise ValueError(f"pad_to_multiple_of={pad_to_multiple_of} must be divisible by sp_size={sp_size}.")


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_yaml_config(args.config), args.override)
    precision_plan = configure_native_fp8_precision(config)
    set_seed(int(config.get("seed", 42)))
    print_precision_plan(precision_plan)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name_or_path"],
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TokenizedJsonlDataset(
        config["data"]["tokenized_dir"],
        min_length=config["data"].get("min_seq_length"),
        max_length=config["data"].get("max_seq_length"),
    )
    data_collator = ACCDataCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=int(config["data"].get("pad_to_multiple_of", 8)),
    )

    model = load_causal_lm(config)
    validate_training_config(config, model)
    model = apply_lora_and_router_policy(model, config)
    trainable, total = count_trainable_parameters(model)
    print(f"trainable_parameters={trainable} total_parameters={total} ratio={trainable / total:.6f}")

    training_cfg = config["training"]
    parallelism_config = build_parallelism_config(config)

    training_args = TrainingArguments(
        output_dir=training_cfg["output_dir"],
        overwrite_output_dir=False,
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        max_steps=int(training_cfg.get("max_steps", -1)),
        learning_rate=float(training_cfg["learning_rate"]),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.05)),
        optim=training_cfg.get("optim", "adamw_torch"),
        adam_beta1=float(training_cfg.get("adam_beta1", 0.9)),
        adam_beta2=float(training_cfg.get("adam_beta2", 0.999)),
        weight_decay=float(training_cfg.get("weight_decay", 0.1)),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        bf16=precision_plan.bf16,
        fp16=precision_plan.fp16,
        tf32=precision_plan.tf32,
        run_name=training_cfg.get("run_name"),
        logging_dir=training_cfg.get("logging_dir"),
        logging_steps=int(training_cfg.get("logging_steps", 1)),
        save_steps=int(training_cfg.get("save_steps", 50)),
        save_total_limit=int(training_cfg.get("save_total_limit", 3)),
        dataloader_num_workers=int(training_cfg.get("dataloader_num_workers", 2)),
        remove_unused_columns=bool(training_cfg.get("remove_unused_columns", False)),
        report_to=[] if str(training_cfg.get("report_to", "none")).lower() == "none" else training_cfg.get("report_to"),
        deepspeed=training_cfg.get("deepspeed_config"),
        parallelism_config=parallelism_config,
    )
    setattr(training_args, "cross_entropy_chunk_size", int(training_cfg.get("cross_entropy_chunk_size", 0)))

    trainer = ACCBucketTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        bucket_boundaries=list(config["data"]["bucket_boundaries"]),
        seed=int(config.get("seed", 42)),
        min_learning_rate=float(training_cfg.get("min_learning_rate", 0.0)),
        base_learning_rate=float(training_cfg["learning_rate"]),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(training_cfg["output_dir"])
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_cfg["output_dir"])
    router_path = save_router_gates(model, training_cfg["output_dir"])

    if trainer.is_world_process_zero():
        with Path(training_cfg["output_dir"], "acc_training_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        print(f"router_gates={router_path}")


if __name__ == "__main__":
    main()
