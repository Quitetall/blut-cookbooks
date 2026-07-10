"""Generic data ingredient specs.

Provides dataset loaders for common formats: HuggingFace datasets and
CSV/JSONL files. Domain cookbooks override with their own data loaders
(e.g. LamQuant's LMA loader).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from blut_core.registry import register_ingredient
from blut_core.spec import IngredientSpec


# ---- HuggingFace dataset ------------------------------------------------
@dataclass(frozen=True)
class HfDataConfig:
    path: str                       # HF dataset name or local path
    split: str = "train"
    subset: Optional[str] = None    # dataset config/subset name
    streaming: bool = False
    max_samples: Optional[int] = None
    text_field: str = "text"
    label_field: Optional[str] = None


def _build_hf_data(cfg: HfDataConfig):
    """Return a ``datasets.Dataset`` or ``datasets.IterableDataset``."""
    from datasets import load_dataset
    ds = load_dataset(
        cfg.path,
        name=cfg.subset,
        split=cfg.split,
        streaming=cfg.streaming,
    )
    if cfg.max_samples is not None:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))
    return ds


@register_ingredient
def _hf_data():
    return IngredientSpec(
        name="huggingface", kind="data", config_cls=HfDataConfig,
        cache_relevant=True,
        build=_build_hf_data,
        requires=("pkg:datasets",),
    )


# ---- CSV / JSONL file ---------------------------------------------------
@dataclass(frozen=True)
class CsvJsonlConfig:
    path: str
    format: str = "auto"            # "csv", "json", "jsonl", or "auto"
    text_field: str = "text"
    label_field: Optional[str] = None
    max_samples: Optional[int] = None
    split: str = "train"


def _build_csv_jsonl(cfg: CsvJsonlConfig):
    """Return a ``datasets.Dataset`` from a local CSV/JSONL file."""
    from datasets import load_dataset
    fmt = cfg.format
    if fmt == "auto":
        if cfg.path.endswith(".csv"):
            fmt = "csv"
        elif cfg.path.endswith(".jsonl") or cfg.path.endswith(".json"):
            fmt = "json"
        else:
            fmt = "json"  # default
    ds = load_dataset(fmt, data_files=cfg.path, split=cfg.split)
    if cfg.max_samples is not None:
        ds = ds.select(range(min(cfg.max_samples, len(ds))))
    return ds


@register_ingredient
def _csv_jsonl_data():
    return IngredientSpec(
        name="csv_jsonl", kind="data", config_cls=CsvJsonlConfig,
        cache_relevant=True,
        build=_build_csv_jsonl,
        requires=("pkg:datasets",),
    )
