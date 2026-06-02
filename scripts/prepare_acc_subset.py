#!/usr/bin/env python3
"""Build a deterministic ACC subset by streaming official Hugging Face JSONL files.

This script does not require downloading the full ACC dataset to disk first. It streams
the official files and writes only the requested subset files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


REPO_BASE = "https://huggingface.co/datasets/groundhogLLM/ACC-dataset/resolve/main"
DEFAULT_SEED = "acc-subset-v1-20260602"

SPECS = {
    "search_agent": {
        "filename": "search_agent_data.jsonl",
        "total": 3369,
        "take": 1000,
        "output": "search_agent_data_1k.jsonl",
    },
    "sql_agent": {
        "filename": "sql_agent_data.jsonl",
        "total": 3065,
        "take": 1500,
        "output": "sql_agent_data_1500.jsonl",
    },
    "swe_agent": {
        "filename": "swe_agent_data.jsonl",
        "total": 4368,
        "take": 2000,
        "output": "swe_agent_data_2k.jsonl",
    },
}


def score(seed: str, config_name: str, line_index: int) -> int:
    key = f"{seed}|{config_name}|{line_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest(), "big")


def selected_indices(seed: str, config_name: str, total: int, take: int | str) -> list[int]:
    if take == "all":
        return list(range(total))
    ranked = sorted(range(total), key=lambda idx: score(seed, config_name, idx))
    return sorted(ranked[: int(take)])


def stream_filter(url: str, output_path: Path, keep: set[int] | None, expected_total: int) -> tuple[int, int]:
    seen = 0
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, output_path.open("wb") as out:
        for idx, line in enumerate(response):
            seen += 1
            if keep is None or idx in keep:
                out.write(line)
                written += 1
            if seen % 500 == 0:
                print(f"  streamed={seen} written={written}", flush=True)
    if seen != expected_total:
        raise RuntimeError(f"Expected {expected_total} lines from {url}, got {seen}.")
    return seen, written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="acc_subset_4500", help="Directory to write subset JSONL files.")
    parser.add_argument("--seed", default=DEFAULT_SEED, help="Stable random seed string.")
    parser.add_argument("--dry-run", action="store_true", help="Only print selection summary; do not stream files.")
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional path to write the sampling manifest without requiring dataset download.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest = {
        "dataset": "groundhogLLM/ACC-dataset",
        "seed": args.seed,
        "selection_rule": "Select line indices with the smallest sha256(seed|config_name|line_index) scores.",
        "line_index_base": "zero_based",
        "length_policy": "Use official ACC 2K-128K data range; no local token retokenization or truncation in selection.",
        "configs": {},
    }

    print(f"seed={args.seed}")
    print(f"output_dir={output_dir}")
    start = time.time()
    for config_name, spec in SPECS.items():
        take = spec["take"]
        total = int(spec["total"])
        indices = selected_indices(args.seed, config_name, total, take)
        output_name = str(spec["output"])
        url = f"{REPO_BASE}/{spec['filename']}"
        print(f"\n[{config_name}] source={spec['filename']} total={total} take={take} output={output_name}")
        print(f"  selected_count={len(indices)} first20={indices[:20]} last20={indices[-20:]}")

        config_manifest = {
            "source_url": url,
            "source_file": spec["filename"],
            "output_file": output_name,
            "source_total_lines": total,
            "selected_lines": len(indices),
            "selected_indices_zero_based": indices,
        }

        if not args.dry_run:
            keep = None if take == "all" else set(indices)
            _, written = stream_filter(url, output_dir / output_name, keep, total)
            if written != len(indices):
                raise RuntimeError(f"{config_name}: expected to write {len(indices)} lines, wrote {written}.")
            print(f"  done written={written}")

        manifest["configs"][config_name] = config_manifest

    total_selected = sum(cfg["selected_lines"] for cfg in manifest["configs"].values())
    manifest["total_selected_lines"] = total_selected
    manifest["expected_files"] = [cfg["output_file"] for cfg in manifest["configs"].values()]
    print(f"\ntotal_selected_lines={total_selected}")

    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"sampling_manifest={manifest_path}")

    if not args.dry_run:
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"manifest={manifest_path}")
        print(f"elapsed_seconds={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
