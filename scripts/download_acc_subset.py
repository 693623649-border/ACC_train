#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ACC subset lines according to a fixed manifest.")
    parser.add_argument("--manifest", default="manifests/ACC_subset_4500_manifest.json")
    parser.add_argument("--output-dir", default="data/acc_subset_4500")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stream_selected_lines(url: str, output_path: Path, selected: set[int], expected_total: int) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    seen = 0
    written = 0
    with urllib.request.urlopen(url, timeout=180) as response, tmp_path.open("wb") as out:
        for idx, line in enumerate(response):
            seen += 1
            if idx in selected:
                out.write(line)
                written += 1
            if seen % 500 == 0:
                print(f"  streamed={seen} written={written}", flush=True)
    if seen != expected_total:
        raise RuntimeError(f"Expected {expected_total} source lines from {url}, got {seen}.")
    tmp_path.replace(output_path)
    return seen, written


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    print(f"dataset={manifest['dataset']}")
    print(f"seed={manifest['seed']}")
    print(f"output_dir={output_dir}")
    started = time.time()

    for config_name, spec in manifest["configs"].items():
        output_path = output_dir / spec["output_file"]
        selected = set(int(idx) for idx in spec["selected_indices_zero_based"])
        expected_selected = int(spec["selected_lines"])
        expected_total = int(spec["source_total_lines"])
        if len(selected) != expected_selected:
            raise ValueError(f"{config_name}: manifest selected index count mismatch.")

        print(f"\n[{config_name}] selected={expected_selected} output={output_path}")
        if args.dry_run:
            print(f"  source={spec['source_url']}")
            print(f"  first10={sorted(selected)[:10]} last10={sorted(selected)[-10:]}")
            continue
        if output_path.exists() and not args.overwrite:
            print("  exists, skipping; pass --overwrite to regenerate")
            continue
        _, written = stream_selected_lines(spec["source_url"], output_path, selected, expected_total)
        if written != expected_selected:
            raise RuntimeError(f"{config_name}: expected {expected_selected}, wrote {written}.")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest_path, output_dir / "manifest.json")
    print(f"\nelapsed_seconds={time.time() - started:.1f}")


if __name__ == "__main__":
    main()
