from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


def torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def cuda_supports_qwen_fp8() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return (major, minor) > (8, 9)


def assert_supported_model_choice(model_name_or_path: str, allow_unsupported_fp8: bool = False) -> None:
    if "FP8" not in model_name_or_path.upper():
        return
    if cuda_supports_qwen_fp8() or allow_unsupported_fp8:
        return
    capability = "no CUDA"
    if torch.cuda.is_available():
        capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
    raise RuntimeError(
        "The selected Qwen FP8 checkpoint is not the default A800 training path. "
        "Qwen documents FP8 computation for GPUs with compute capability > 8.9; "
        f"this runtime reports {capability}. Use Qwen/Qwen3-30B-A3B-Thinking-2507 "
        "for the A800 BF16 path, or pass allow_unsupported_fp8=true only for a "
        "smoke test on supported hardware."
    )


def load_causal_lm(config: dict[str, Any]):
    model_cfg = config["model"]
    model_name = model_cfg["name_or_path"]
    assert_supported_model_choice(model_name, bool(model_cfg.get("allow_unsupported_fp8", False)))
    dtype = torch_dtype_from_name(model_cfg.get("torch_dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        device_map=None,
    )
    model.config.use_cache = bool(model_cfg.get("use_cache", False))
    if bool(model_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
    return model


def apply_lora_and_router_policy(model, config: dict[str, Any]):
    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", True):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            bias=str(lora_cfg.get("bias", "none")),
        )
        model = get_peft_model(model, peft_config)

    router_cfg = config.get("router", {})
    if router_cfg.get("train_router_gates", True):
        pattern = re.compile(str(router_cfg.get("name_regex", r"model\.layers\.\d+\.mlp\.gate")))
        router_dtype = torch_dtype_from_name(router_cfg.get("dtype", "bfloat16"))
        marked = 0
        for name, module in model.named_modules():
            if pattern.search(name):
                module.to(dtype=router_dtype)
                for param in module.parameters(recurse=False):
                    param.requires_grad = True
                    marked += param.numel()
        if marked == 0:
            raise RuntimeError(f"No router gate parameters matched regex: {pattern.pattern}")
    return model


def count_trainable_parameters(model) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def save_router_gates(model, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    router_state = {}
    for name, tensor in model.state_dict().items():
        if re.search(r"model\.layers\.\d+\.mlp\.gate", name):
            router_state[name] = tensor.detach().cpu()
    if not router_state:
        raise RuntimeError("No router gate tensors found while saving router weights.")
    path = output / "router_gates.safetensors"
    save_file(router_state, str(path))
    return path
