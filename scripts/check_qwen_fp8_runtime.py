#!/usr/bin/env python3
from __future__ import annotations

import importlib.util

import torch


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    transformer_engine_available = package_available("transformer_engine")
    print("required_mixed_precision=fp8")
    print("required_fp8_backend=te")
    print(f"transformer_engine_available={str(transformer_engine_available).lower()}")
    if not torch.cuda.is_available():
        print("cuda_available=false")
        print("qwen_fp8_runtime_supported=false")
        print("all_visible_gpus_qwen_fp8_supported=false")
        print("native_fp8_training_ready=false")
        return
    all_supported = True
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        name = torch.cuda.get_device_name(index)
        supported = (major, minor) > (8, 9)
        all_supported = all_supported and supported
        print(
            f"gpu={index} name={name} compute_capability={major}.{minor} "
            f"qwen_fp8_runtime_supported={str(supported).lower()}"
        )
    print(f"all_visible_gpus_qwen_fp8_supported={str(all_supported).lower()}")
    ready = all_supported and transformer_engine_available
    print(f"native_fp8_training_ready={str(ready).lower()}")
    print("H20/Hopper-class GPUs should report sm90-class capability and use Accelerate FP8 with Transformer Engine.")


if __name__ == "__main__":
    main()
