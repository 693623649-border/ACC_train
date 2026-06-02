#!/usr/bin/env python3
from __future__ import annotations

import torch


def main() -> None:
    if not torch.cuda.is_available():
        print("cuda_available=false")
        print("qwen_fp8_runtime_supported=false")
        return
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        name = torch.cuda.get_device_name(index)
        supported = (major, minor) > (8, 9)
        print(
            f"gpu={index} name={name} compute_capability={major}.{minor} "
            f"qwen_fp8_runtime_supported={str(supported).lower()}"
        )
    print("A800/A100 class GPUs are expected to report sm80 and should use the BF16 training path.")


if __name__ == "__main__":
    main()
