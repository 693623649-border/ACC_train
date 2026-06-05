#!/usr/bin/env python3
from __future__ import annotations

import importlib.util

try:
    import torch
except ImportError:
    torch = None


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    print("required_device_profile=2xA100-SXM4-40GB")
    print("required_mixed_precision=bf16")
    print("required_parallelism=deepspeed_zero3_sp2")
    if torch is None:
        print("torch_available=false")
        print("a100_bf16_training_ready=false")
        return
    print("torch_available=true")
    print(f"torch_version={torch.__version__}")
    print(f"accelerate_available={str(package_available('accelerate')).lower()}")
    print(f"deepspeed_available={str(package_available('deepspeed')).lower()}")
    print(f"flash_attn_available={str(package_available('flash_attn')).lower()}")
    print(f"transformer_engine_required=false")

    if not torch.cuda.is_available():
        print("cuda_available=false")
        print("a100_bf16_training_ready=false")
        return

    print("cuda_available=true")
    print(f"visible_gpu_count={torch.cuda.device_count()}")
    cuda_bf16_supported = torch.cuda.is_bf16_supported()
    all_a100 = True
    all_ampere_or_newer = True
    all_memory_40gb_class = True

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        total_gb = props.total_memory / 1024**3
        is_a100 = "A100" in props.name
        ampere_or_newer = (major, minor) >= (8, 0)
        memory_40gb_class = total_gb >= 39.0
        all_a100 = all_a100 and is_a100
        all_ampere_or_newer = all_ampere_or_newer and ampere_or_newer
        all_memory_40gb_class = all_memory_40gb_class and memory_40gb_class
        print(
            f"gpu={index} name={props.name} total_memory_gb={total_gb:.1f} "
            f"compute_capability={major}.{minor} is_a100={str(is_a100).lower()} "
            f"ampere_or_newer={str(ampere_or_newer).lower()} "
            f"memory_40gb_class={str(memory_40gb_class).lower()}"
        )

    two_process_target = torch.cuda.device_count() >= 2
    package_ready = (
        package_available("accelerate")
        and package_available("deepspeed")
        and package_available("flash_attn")
    )
    ready = (
        two_process_target
        and all_a100
        and all_ampere_or_newer
        and all_memory_40gb_class
        and cuda_bf16_supported
        and package_ready
    )
    print(f"all_visible_gpus_a100={str(all_a100).lower()}")
    print(f"all_visible_gpus_ampere_or_newer={str(all_ampere_or_newer).lower()}")
    print(f"all_visible_gpus_40gb_class={str(all_memory_40gb_class).lower()}")
    print(f"cuda_bf16_supported={str(cuda_bf16_supported).lower()}")
    print(f"two_process_target_available={str(two_process_target).lower()}")
    print(f"a100_bf16_training_ready={str(ready).lower()}")
    print("Run 8K and longest-bucket 128K smoke tests before the full 131072-token run.")


if __name__ == "__main__":
    main()
