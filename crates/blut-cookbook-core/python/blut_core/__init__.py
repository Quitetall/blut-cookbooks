"""blut_core — the shared foundation for all BLUT cookbooks.

Provides the ingredient system (``IngredientSpec``, registry, KINDS) and
generic ML ingredient specs (optimizers, schedulers, losses, etc.) that
domain cookbooks build on.

Domain cookbooks import from ``blut_core`` rather than duplicating these types::

    from blut_core import IngredientSpec, register_ingredient, build_ingredient
    from blut_core.spec import KINDS
"""

from blut_core.registry import (
    build_ingredient,
    get_spec,
    list_ingredients,
    register_ingredient,
)
from blut_core.spec import KINDS, IngredientSpec

# Register the built-in generic ingredient specs on package import.
from blut_core.ingredients import _specs as _ingredient_specs  # noqa: F401

# Runtime utilities (merged from blut-core backends).
from . import runctx, status, sysgauge, checkpoint
from .metric_log import MetricLog
from .run_manifest import RunManifest

__all__ = [
    "build_ingredient",
    "get_spec",
    "list_ingredients",
    "register_ingredient",
    "IngredientSpec",
    "KINDS",
    "runctx", "status", "sysgauge", "checkpoint", "read_metric",
    "MetricLog", "RunManifest",
]
