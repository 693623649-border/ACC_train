#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"
DEFAULT_OUTPUT_DIR = "model_assets/Qwen3-30B-A3B-Thinking-2507-nonweights"

ALLOW_PATTERNS = [
    "*.json",
    "*.md",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "merges.txt",
    "vocab.json",
    "tokenizer.*",
    "chat_template*",
    "*.py",
]

IGNORE_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.gguf",
    "*.onnx",
]

BLOCKED_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download only non-weight files from a Hugging Face model repo.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def download_non_weight_assets(repo_id: str, output_dir: str | Path, revision: str | None = None) -> Path:
    output_dir = Path(output_dir)
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )
    blocked = [p for p in Path(path).rglob("*") if p.is_file() and p.suffix in BLOCKED_SUFFIXES]
    if blocked:
        details = "\n".join(str(p) for p in blocked[:20])
        raise RuntimeError(f"Downloaded weight-like files unexpectedly:\n{details}")
    print(f"downloaded_non_weight_assets={path}")
    return Path(path)


def main() -> None:
    args = parse_args()
    download_non_weight_assets(args.repo_id, args.output_dir, args.revision)


if __name__ == "__main__":
    main()
