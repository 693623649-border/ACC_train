from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PrecisionPlan:
    mode: str
    mixed_precision: str
    fp8_backend: str
    fp8_format: str
    trainable_parameter_dtype: str
    bf16: bool
    fp16: bool
    tf32: bool


def _normal(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _visible_gpu_capabilities() -> list[tuple[int, int]]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count())]


def visible_gpus_support_fp8() -> bool:
    capabilities = _visible_gpu_capabilities()
    return bool(capabilities) and all(capability > (8, 9) for capability in capabilities)


def visible_gpus_support_bf16() -> bool:
    capabilities = _visible_gpu_capabilities()
    if not capabilities:
        return False
    return all(capability >= (8, 0) for capability in capabilities) and torch.cuda.is_bf16_supported()


def build_precision_plan(config: dict[str, Any]) -> PrecisionPlan:
    precision_cfg = config.get("precision", {})
    training_cfg = config.get("training", {})
    mode = _normal(precision_cfg.get("mode", config.get("model", {}).get("precision_mode")), "native_fp8_transformer_engine")
    default_mixed_precision = "bf16" if mode == "native_bf16_ampere" else "fp8"
    mixed_precision = _normal(
        precision_cfg.get("mixed_precision", training_cfg.get("mixed_precision")),
        default_mixed_precision,
    ).lower()
    default_fp8_backend = "disabled" if mixed_precision == "bf16" else "TE"
    default_fp8_format = "disabled" if mixed_precision == "bf16" else "HYBRID"
    fp8_backend = _normal(precision_cfg.get("fp8_backend"), default_fp8_backend).upper()
    fp8_format = _normal(precision_cfg.get("fp8_format"), default_fp8_format).upper()
    trainable_dtype = _normal(precision_cfg.get("trainable_parameter_dtype"), "auto").lower()
    return PrecisionPlan(
        mode=mode,
        mixed_precision=mixed_precision,
        fp8_backend=fp8_backend,
        fp8_format=fp8_format,
        trainable_parameter_dtype=trainable_dtype,
        bf16=bool(training_cfg.get("bf16", False)),
        fp16=bool(training_cfg.get("fp16", False)),
        tf32=bool(training_cfg.get("tf32", False)),
    )


def assert_native_fp8_precision(config: dict[str, Any], plan: PrecisionPlan) -> None:
    if plan.mixed_precision != "fp8":
        raise ValueError(f"Native FP8 run requires mixed_precision=fp8, got {plan.mixed_precision!r}.")
    if plan.fp8_backend != "TE":
        raise ValueError(f"This repository is pinned to Transformer Engine FP8, got backend={plan.fp8_backend!r}.")
    if plan.bf16 or plan.fp16:
        raise ValueError("Native FP8 run must keep TrainingArguments bf16=false and fp16=false.")
    if not visible_gpus_support_fp8() and not bool(config.get("model", {}).get("allow_unsupported_fp8", False)):
        if torch.cuda.is_available():
            details = ", ".join(
                f"cuda:{index} sm{major}{minor}"
                for index, (major, minor) in enumerate(_visible_gpu_capabilities())
            )
        else:
            details = "no CUDA devices"
        raise RuntimeError(
            "Native FP8 training requires all visible GPUs to have compute capability > 8.9; "
            f"this runtime reports {details}."
        )
    if bool(config.get("precision", {}).get("require_transformer_engine", True)):
        if not _package_available("transformer_engine"):
            raise RuntimeError(
                "Native FP8 training requires NVIDIA Transformer Engine. "
                "Install `transformer-engine[pytorch]` in the CUDA13/Torch image."
            )


def assert_native_bf16_precision(config: dict[str, Any], plan: PrecisionPlan) -> None:
    if plan.mixed_precision != "bf16":
        raise ValueError(f"A100 BF16 run requires mixed_precision=bf16, got {plan.mixed_precision!r}.")
    if not plan.bf16 or plan.fp16:
        raise ValueError("A100 BF16 run must keep TrainingArguments bf16=true and fp16=false.")
    if not visible_gpus_support_bf16() and not bool(config.get("precision", {}).get("allow_unsupported_bf16", False)):
        if torch.cuda.is_available():
            details = ", ".join(
                f"cuda:{index} sm{major}{minor}"
                for index, (major, minor) in enumerate(_visible_gpu_capabilities())
            )
        else:
            details = "no CUDA devices"
        raise RuntimeError(
            "Native BF16 training on this path requires Ampere-or-newer GPUs with BF16 support; "
            f"this runtime reports {details}."
        )


def configure_precision(config: dict[str, Any]) -> PrecisionPlan:
    plan = build_precision_plan(config)
    if plan.mode == "native_fp8_transformer_engine" or plan.mixed_precision == "fp8":
        assert_native_fp8_precision(config, plan)
        os.environ["ACCELERATE_MIXED_PRECISION"] = "fp8"
        os.environ["ACCELERATE_FP8_BACKEND"] = plan.fp8_backend.lower()
        os.environ["ACCELERATE_FP8_FORMAT"] = plan.fp8_format
    elif plan.mode == "native_bf16_ampere" or plan.mixed_precision == "bf16":
        assert_native_bf16_precision(config, plan)
        os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
        os.environ.pop("ACCELERATE_FP8_BACKEND", None)
        os.environ.pop("ACCELERATE_FP8_FORMAT", None)
    else:
        raise ValueError(f"Unsupported precision mode={plan.mode!r} mixed_precision={plan.mixed_precision!r}.")
    torch.backends.cuda.matmul.allow_tf32 = plan.tf32
    torch.backends.cudnn.allow_tf32 = plan.tf32
    return plan


def configure_native_fp8_precision(config: dict[str, Any]) -> PrecisionPlan:
    return configure_precision(config)


def print_precision_plan(plan: PrecisionPlan) -> None:
    print(f"precision_mode={plan.mode}")
    print(f"accelerate_mixed_precision={plan.mixed_precision}")
    print(f"fp8_backend={plan.fp8_backend}")
    print(f"fp8_format={plan.fp8_format}")
    print(f"training_args_bf16={str(plan.bf16).lower()}")
    print(f"training_args_fp16={str(plan.fp16).lower()}")
    print(f"tf32={str(plan.tf32).lower()}")
    print(f"trainable_parameter_dtype={plan.trainable_parameter_dtype}")
