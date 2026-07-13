"""Generic EMA ingredient specs.

Provides exponential moving average and simple model averaging.
Domain cookbooks can extend with domain-specific EMA variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from blut_core.registry import register_ingredient
from blut_core.spec import IngredientSpec


# ---- Exponential moving average -----------------------------------------
@dataclass(frozen=True)
class ExponentialEmaConfig:
    decay: float = 0.999
    enabled: bool = True


def _build_exponential_ema(cfg: ExponentialEmaConfig, model):
    """Return an EMA-wrapped model, or None when disabled."""
    if not cfg.enabled:
        return None
    if not 0.0 < cfg.decay < 1.0:
        raise ValueError(f"ema decay must be in (0, 1), got {cfg.decay}")
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.decay))


@register_ingredient
def _exponential_ema():
    return IngredientSpec(
        name="exponential", kind="ema", config_cls=ExponentialEmaConfig,
        cache_relevant=True,
        build=_build_exponential_ema,
    )


# ---- Simple model averaging --------------------------------------------
@dataclass(frozen=True)
class ModelAverageConfig:
    enabled: bool = True
    avg_fn: Optional[str] = None  # "mean" or None (default AveragedModel)


def _build_model_average(cfg: ModelAverageConfig, model):
    """Return an AveragedModel with simple averaging, or None when disabled."""
    if not cfg.enabled:
        return None
    from torch.optim.swa_utils import AveragedModel
    return AveragedModel(model)


@register_ingredient
def _model_average():
    return IngredientSpec(
        name="model_average", kind="ema", config_cls=ModelAverageConfig,
        cache_relevant=True,
        build=_build_model_average,
    )
