from __future__ import annotations

import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


def torch_dtype_from_name(name: str) -> torch.dtype | str:
    normalized = str(name).lower()
    if normalized == "auto":
        return "auto"
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
    return all_visible_gpus_support_qwen_fp8()


def all_visible_gpus_support_qwen_fp8() -> bool:
    if not torch.cuda.is_available():
        return False
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        if (major, minor) <= (8, 9):
            return False
    return True


def assert_supported_model_choice(
    model_name_or_path: str,
    allow_unsupported_fp8: bool = False,
    require_native_fp8_runtime: bool = True,
) -> None:
    if "FP8" not in model_name_or_path.upper():
        return
    if not require_native_fp8_runtime:
        return
    if all_visible_gpus_support_qwen_fp8() or allow_unsupported_fp8:
        return
    capability = "no CUDA devices"
    if torch.cuda.is_available():
        capability = ", ".join(
            f"cuda:{index} sm{major}{minor}"
            for index in range(torch.cuda.device_count())
            for major, minor in [torch.cuda.get_device_capability(index)]
        )
    raise RuntimeError(
        "The selected Qwen FP8 checkpoint requires Hopper/Ada-or-newer FP8-capable GPUs. "
        "Qwen documents FP8 computation for GPUs with compute capability > 8.9; "
        f"this runtime reports {capability}. The H20 FP8 path should report sm90-class "
        "devices. Set allow_unsupported_fp8=true only for an intentional diagnostic run."
    )


def config_requires_native_fp8_runtime(config: dict[str, Any]) -> bool:
    precision_cfg = config.get("precision", {})
    training_cfg = config.get("training", {})
    mode = str(precision_cfg.get("mode", config.get("model", {}).get("precision_mode", ""))).lower()
    mixed_precision = str(precision_cfg.get("mixed_precision", training_cfg.get("mixed_precision", ""))).lower()
    return mode == "native_fp8_transformer_engine" or mixed_precision == "fp8"


def build_hf_deepspeed_config(config: dict[str, Any]):
    ds_path = config.get("training", {}).get("deepspeed_config")
    if not ds_path:
        return None
    try:
        from transformers.integrations import HfDeepSpeedConfig
    except ImportError:
        try:
            from transformers.deepspeed import HfDeepSpeedConfig
        except ImportError as exc:
            raise RuntimeError("ZeRO-3 init-time partitioning requires HfDeepSpeedConfig.") from exc

    with Path(ds_path).open("r", encoding="utf-8") as handle:
        ds_config = json.load(handle)
    if ds_config.get("zero_optimization", {}).get("stage") != 3:
        return None
    return HfDeepSpeedConfig(ds_config)


def load_causal_lm(config: dict[str, Any]):
    model_cfg = config["model"]
    model_name = model_cfg["name_or_path"]
    assert_supported_model_choice(
        model_name,
        bool(model_cfg.get("allow_unsupported_fp8", False)),
        require_native_fp8_runtime=config_requires_native_fp8_runtime(config),
    )
    dtype = torch_dtype_from_name(model_cfg.get("torch_dtype", "auto"))
    hf_deepspeed_config = build_hf_deepspeed_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        device_map=None,
    )
    if hf_deepspeed_config is not None:
        model._hf_deepspeed_config = hf_deepspeed_config
    model.config.use_cache = bool(model_cfg.get("use_cache", False))
    if bool(model_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
    return model


def apply_lora_and_router_policy(model, config: dict[str, Any]):
    for param in model.parameters():
        param.requires_grad = False

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
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    router_cfg = config.get("router", {})
    if router_cfg.get("train_router_gates", True):
        pattern = re.compile(str(router_cfg.get("name_regex", r"model\.layers\.\d+\.mlp\.gate")))
        router_dtype = torch_dtype_from_name(router_cfg.get("dtype", "auto"))
        marked = 0
        for name, module in model.named_modules():
            if pattern.search(name):
                if router_dtype != "auto":
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


def is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def gathered_parameter(param: torch.nn.Parameter):
    try:
        import deepspeed
    except ImportError:
        return nullcontext(param)
    zero = getattr(deepspeed, "zero", None)
    if zero is None or not hasattr(zero, "GatheredParameters"):
        return nullcontext(param)
    return zero.GatheredParameters([param], modifier_rank=0)


def save_router_gates(model, output_dir: str | Path) -> Path | None:
    output = Path(output_dir)
    router_state = {}
    for name, param in model.named_parameters():
        if re.search(r"model\.layers\.\d+\.mlp\.gate", name):
            with gathered_parameter(param):
                if is_main_process():
                    router_state[name] = param.detach().cpu().clone()
    if not is_main_process():
        return None
    if not router_state:
        raise RuntimeError("No router gate tensors found while saving router weights.")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "router_gates.safetensors"
    save_file(router_state, str(path))
    return path
