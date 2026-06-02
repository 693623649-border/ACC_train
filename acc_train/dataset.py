from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class JsonlRecord:
    path: Path
    offset: int
    length: int


class TokenizedJsonlDataset(Dataset):
    """Lazy JSONL dataset for tokenized ACC sequences.

    Each line is expected to contain input_ids, labels, attention_mask, and length.
    The dataset keeps only byte offsets in memory, which is important for 128K-token
    examples where loading every token array at construction time is wasteful.
    """

    def __init__(
        self,
        tokenized_dir: str | Path,
        pattern: str = "bucket_*.jsonl",
        min_length: int | None = None,
        max_length: int | None = None,
    ) -> None:
        self.tokenized_dir = Path(tokenized_dir)
        self.min_length = min_length
        self.max_length = max_length
        self.files = sorted(self.tokenized_dir.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No tokenized files matching {pattern} under {self.tokenized_dir}")

        self.records: list[JsonlRecord] = []
        for path in self.files:
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at {path}:{offset}") from exc
                    length = int(payload.get("length", len(payload.get("input_ids", []))))
                    if self.min_length is not None and length < self.min_length:
                        continue
                    if self.max_length is not None and length > self.max_length:
                        continue
                    self.records.append(JsonlRecord(path=path, offset=offset, length=length))

        if not self.records:
            raise ValueError(
                f"No examples found under {self.tokenized_dir} "
                f"with min_length={self.min_length} max_length={self.max_length}"
            )
        self._handles: dict[Path, Any] = {}

    def __len__(self) -> int:
        return len(self.records)

    @property
    def lengths(self) -> list[int]:
        return [record.length for record in self.records]

    def _get_handle(self, path: Path):
        handle = self._handles.get(path)
        if handle is None or handle.closed:
            handle = path.open("rb")
            self._handles[path] = handle
        return handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        handle = self._get_handle(record.path)
        handle.seek(record.offset)
        payload = json.loads(handle.readline())
        return {
            "input_ids": payload["input_ids"],
            "attention_mask": payload.get("attention_mask", [1] * len(payload["input_ids"])),
            "labels": payload["labels"],
            "length": payload.get("length", len(payload["input_ids"])),
            "id": payload.get("id"),
            "source": payload.get("source"),
            "bucket": payload.get("bucket"),
        }


class LengthBucketSampler(Sampler[int]):
    """Sampler that shuffles within fixed length buckets.

    It preserves one sample per batch semantics in Trainer while giving the collator
    batches with similar lengths when per-device batch size is increased for smoke
    tests. For the production SP2 run, per_device_train_batch_size stays 1.
    """

    def __init__(
        self,
        lengths: list[int],
        boundaries: list[int],
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.lengths = lengths
        self.boundaries = sorted(int(boundary) for boundary in boundaries)
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.buckets: dict[int, list[int]] = {boundary: [] for boundary in self.boundaries}
        self.buckets[math.inf] = []
        for idx, length in enumerate(lengths):
            bucket = next((boundary for boundary in self.boundaries if length <= boundary), math.inf)
            self.buckets[bucket].append(idx)

    def __len__(self) -> int:
        return len(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        bucket_keys = [key for key, values in self.buckets.items() if values]
        if self.shuffle:
            rng.shuffle(bucket_keys)
        for key in bucket_keys:
            values = list(self.buckets[key])
            if self.shuffle:
                rng.shuffle(values)
            yield from values


class ACCDataCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = int(pad_to_multiple_of)

    def _target_length(self, lengths: list[int]) -> int:
        max_len = max(lengths)
        multiple = max(1, self.pad_to_multiple_of)
        return ((max_len + multiple - 1) // multiple) * multiple

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        lengths = [len(feature["input_ids"]) for feature in features]
        target_len = self._target_length(lengths)
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []
        batch_position_ids: list[list[int]] = []

        for feature in features:
            input_ids = list(feature["input_ids"])
            labels = list(feature["labels"])
            attention_mask = list(feature.get("attention_mask", [1] * len(input_ids)))
            pad_len = target_len - len(input_ids)
            if pad_len < 0:
                raise ValueError("Feature is longer than computed target length.")
            batch_input_ids.append(input_ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(attention_mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)
            batch_position_ids.append(list(range(len(input_ids))) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "position_ids": torch.tensor(batch_position_ids, dtype=torch.long),
        }
