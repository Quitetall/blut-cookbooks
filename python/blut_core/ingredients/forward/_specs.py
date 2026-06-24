"""Generic forward pass ingredient specs.

Provides a standard forward pass wrapper. Domain cookbooks can extend with
domain-specific forward patterns (e.g. LamQuant's MAE masked forward).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from blut_core.registry import register_ingredient
from blut_core.spec import IngredientSpec


# ---- Standard forward pass ----------------------------------------------
@dataclass(frozen=True)
class StandardForwardConfig:
    """Standard forward pass — just calls model(batch)."""
    pass


def _build_standard_forward(cfg: StandardForwardConfig):
    """Return a forward function that calls model(batch) directly."""
    def forward_fn(model, batch, **kwargs):
        return model(batch)
    return forward_fn


@register_ingredient
def _standard_forward():
    return IngredientSpec(
        name="standard", kind="forward", config_cls=StandardForwardConfig,
        cache_relevant=False,
        build=_build_standard_forward,
    )
