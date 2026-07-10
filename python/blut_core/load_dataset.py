"""Materialize a HuggingFace dataset split as deterministic JSONL."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def materialize(
    name: str,
    split: str,
    output: Path,
    *,
    subset: str | None = None,
    max_samples: int | None = None,
) -> int:
    from datasets import load_dataset

    dataset = load_dataset(name, subset, split=split)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=output.parent, prefix=output.name + ".", suffix=".tmp")
    temporary = Path(temporary_name)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in dataset:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if count == 0:
        output.unlink(missing_ok=True)
        raise ValueError(f"dataset {name!r} split {split!r} is empty")
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--subset")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(
        args.name,
        args.split,
        args.output,
        subset=args.subset,
        max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
