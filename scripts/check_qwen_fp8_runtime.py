#!/usr/bin/env python3
from __future__ import annotations

import torch


def main() -> None:
    if not torch.cuda.is_available():
        print("cuda_available=false")
        print("qwen_fp8_runtime_supported=false")
        print("all_visible_gpus_qwen_fp8_supported=false")
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
    print("H20/Hopper-class GPUs are expected to report sm90-class capability and support the FP8 path.")


if __name__ == "__main__":
    main()
