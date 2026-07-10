"""Shared Python runtime and ingredient registry for BLUT cookbooks.

The package contains domain-agnostic building blocks reusable by any cookbook:

- ``runctx``       — run identity + filesystem anchors from the BLUT env (P10)
- ``MetricLog``    — atomic per-epoch CSV/Parquet metric writer
- ``read_metric``  — verbatim metric/log reader (no LLM in the path; ADR 0038)
- ``status``       — emit/read the StatusUpdate wire protocol (P4)
- ``RunManifest``  — run provenance, written even on crash (P8)
- ``checkpoint``   — corruption-safe save/load + resume payload contract (P7)
- ``sysgauge``     — best-effort GPU/host snapshot for the metric stream (P10)
- ingredient specs and builders used by generic training recipes

``torch`` is lazy-imported only inside ``checkpoint``/``sysgauge`` functions, so
``import blut_core`` stays cheap and dependency-light. See ``blut/docs/metrics.md``
and decisions/0037, 0044.
"""

from blut_core.registry import (
    build_ingredient,
    get_spec,
    list_ingredients,
    register_ingredient,
)
from blut_core.spec import KINDS, IngredientSpec

# Register generic ingredient specs on package import.
from blut_core.ingredients import _specs as _ingredient_specs  # noqa: F401

from . import checkpoint, runctx, status, sysgauge
from .metric_log import MetricLog
from .run_manifest import RunManifest

# read_metric is a CLI module (`python -m blut_core.read_metric`); it is NOT
# eager-imported here (that triggers a runpy double-import warning under -m).
# `from blut_core import read_metric` still works (submodule import).

__all__ = [
    "IngredientSpec",
    "KINDS",
    "MetricLog",
    "RunManifest",
    "build_ingredient",
    "checkpoint",
    "get_spec",
    "list_ingredients",
    "read_metric",
    "register_ingredient",
    "runctx",
    "status",
    "sysgauge",
]
