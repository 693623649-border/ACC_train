#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize ACC subset with assistant-only labels.")
    parser.add_argument("--manifest", default="manifests/ACC_subset_4500_manifest.json")
    parser.add_argument("--raw-dir", default="data/acc_subset_4500")
    parser.add_argument("--output-dir", default="data/tokenized_acc_4500_qwen3_bf16_128k")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-30B-A3B-Thinking-2507")
    parser.add_argument("--max-seq-length", type=int, default=131072)
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument(
        "--bucket-boundaries",
        default="8192,16384,32768,49152,65536,81920,98304,114688,131072",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def normalize_message(message: dict[str, Any]) -> dict[str, str]:
    role = message.get("role") or message.get("from") or message.get("speaker")
    content = message.get("content")
    if content is None:
        content = message.get("value") or message.get("text") or ""
    if role in {"human", "user"}:
        role = "user"
    elif role in {"gpt", "assistant", "model"}:
        role = "assistant"
    elif role in {"system"}:
        role = "system"
    else:
        role = str(role or "user")
    return {"role": role, "content": str(content)}


def apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def encode_with_template_assistant_mask(tokenizer, messages: list[dict[str, str]]) -> tuple[list[int], list[int]] | None:
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except Exception:
        return None
    input_ids = list(encoded["input_ids"])
    mask = encoded.get("assistant_masks")
    if mask is None:
        mask = encoded.get("assistant_tokens_mask")
    if mask is None:
        return None
    labels = [token_id if int(flag) == 1 else -100 for token_id, flag in zip(input_ids, mask)]
    if all(label == -100 for label in labels):
        return None
    return input_ids, labels


def encode_with_offset_assistant_labels(tokenizer, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
    full_text = apply_chat_template(tokenizer, messages)
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = encoded.input_ids
    labels = [-100] * len(full_ids)
    cursor = 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or not message["content"]:
            continue
        prefix_text = apply_chat_template(tokenizer, messages[:index] + [{"role": "assistant", "content": ""}])
        search_from = max(0, min(len(full_text), len(prefix_text), cursor))
        start_char = full_text.find(message["content"], search_from)
        if start_char < 0:
            start_char = full_text.find(message["content"], cursor)
        if start_char < 0:
            continue
        end_char = start_char + len(message["content"])
        cursor = end_char
        for token_index, (start, end) in enumerate(encoded.offset_mapping):
            if start < end_char and end > start_char:
                labels[token_index] = full_ids[token_index]
    return full_ids, labels


def encode_with_assistant_labels(tokenizer, messages: list[dict[str, str]]) -> tuple[list[int], list[int]]:
    masked = encode_with_template_assistant_mask(tokenizer, messages)
    if masked is not None:
        return masked
    if tokenizer.is_fast:
        return encode_with_offset_assistant_labels(tokenizer, messages)

    full_text = apply_chat_template(tokenizer, messages)
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    labels = [-100] * len(full_ids)

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix_messages = list(messages[:index]) + [{"role": "assistant", "content": ""}]
        prefix_text = apply_chat_template(tokenizer, prefix_messages)
        end_text = apply_chat_template(tokenizer, messages[: index + 1])
        prefix_len = len(tokenizer(prefix_text, add_special_tokens=False).input_ids)
        end_len = len(tokenizer(end_text, add_special_tokens=False).input_ids)
        prefix_len = min(prefix_len, len(full_ids))
        end_len = min(end_len, len(full_ids))
        if end_len > prefix_len:
            labels[prefix_len:end_len] = full_ids[prefix_len:end_len]

    return full_ids, labels


def bucket_for_length(length: int, boundaries: list[int]) -> int:
    for boundary in boundaries:
        if length <= boundary:
            return boundary
    return boundaries[-1]


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries = [int(item) for item in args.bucket_boundaries.split(",") if item.strip()]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bucket_handles = {
        boundary: (output_dir / f"bucket_le_{boundary}.jsonl").open("wb")
        for boundary in boundaries
    }
    index_handle = (output_dir / "index.jsonl").open("w", encoding="utf-8")
    reject_path = output_dir / "rejected.jsonl"
    reject_handle = reject_path.open("w", encoding="utf-8")
    counts = Counter()
    lengths: list[int] = []

    try:
        for config_name, spec in manifest["configs"].items():
            input_path = raw_dir / spec["output_file"]
            if not input_path.exists():
                raise FileNotFoundError(f"Missing subset file: {input_path}")
            with input_path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(tqdm(handle, desc=f"tokenize:{config_name}"), start=1):
                    record = json.loads(line)
                    messages = [normalize_message(message) for message in record["dialogs"]]
                    try:
                        input_ids, labels = encode_with_assistant_labels(tokenizer, messages)
                    except Exception as exc:
                        reject_handle.write(
                            json.dumps(
                                {
                                    "id": record.get("id"),
                                    "source": config_name,
                                    "line_no": line_no,
                                    "reason": f"chat_template_error:{type(exc).__name__}:{exc}",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        counts["rejected_template"] += 1
                        continue

                    length = len(input_ids)
                    if length > args.max_seq_length:
                        reject_handle.write(
                            json.dumps(
                                {
                                    "id": record.get("id"),
                                    "source": config_name,
                                    "line_no": line_no,
                                    "reason": "too_long",
                                    "length": length,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        counts["rejected_too_long"] += 1
                        continue
                    if all(label == -100 for label in labels):
                        reject_handle.write(
                            json.dumps(
                                {
                                    "id": record.get("id"),
                                    "source": config_name,
                                    "line_no": line_no,
                                    "reason": "no_assistant_labels",
                                    "length": length,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        counts["rejected_no_labels"] += 1
                        continue

                    bucket = bucket_for_length(length, boundaries)
                    payload = {
                        "id": record.get("id"),
                        "source": config_name,
                        "task_type": record.get("task_type"),
                        "length": length,
                        "bucket": bucket,
                        "input_ids": input_ids,
                        "attention_mask": [1] * length,
                        "labels": labels,
                    }
                    bucket_path = output_dir / f"bucket_le_{bucket}.jsonl"
                    payload_line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                    offset = bucket_handles[bucket].tell()
                    bucket_handles[bucket].write(payload_line)
                    index_handle.write(
                        json.dumps(
                            {
                                "relative_path": bucket_path.name,
                                "offset": offset,
                                "length": length,
                                "id": record.get("id"),
                                "source": config_name,
                                "bucket": bucket,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    counts[f"accepted_{config_name}"] += 1
                    counts[f"bucket_le_{bucket}"] += 1
                    counts["accepted_total"] += 1
                    lengths.append(length)
    finally:
        for handle in bucket_handles.values():
            handle.close()
        index_handle.close()
        reject_handle.close()

    metadata = {
        "model_name_or_path": args.model_name_or_path,
        "max_seq_length": args.max_seq_length,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        "bucket_boundaries": boundaries,
        "counts": dict(counts),
        "length_min": min(lengths) if lengths else None,
        "length_max": max(lengths) if lengths else None,
        "length_avg": sum(lengths) / len(lengths) if lengths else None,
        "rejected_path": str(reject_path),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
