"""Generic sampler ingredient specs.

Provides common samplers: random, sequential, and weighted.
Domain cookbooks can extend with domain-specific samplers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from blut_core.registry import register_ingredient
from blut_core.spec import IngredientSpec


# ---- Random sampler -----------------------------------------------------
@dataclass(frozen=True)
class RandomSamplerConfig:
    seed: int = 42
    replacement: bool = False
    num_samples: Optional[int] = None


def _build_random_sampler(cfg: RandomSamplerConfig, dataset):
    """Return a torch Sampler for random sampling."""
    from torch.utils.data import RandomSampler, WeightedRandomSampler
    generator = __import__("torch").Generator().manual_seed(cfg.seed)
    if cfg.replacement:
        return WeightedRandomSampler(
            weights=[1.0] * len(dataset),
            num_samples=cfg.num_samples or len(dataset),
            replacement=True,
        )
    return RandomSampler(
        dataset,
        num_samples=cfg.num_samples,
        generator=generator,
    )


@register_ingredient
def _random_sampler():
    return IngredientSpec(
        name="random", kind="sampler", config_cls=RandomSamplerConfig,
        cache_relevant=True,
        build=_build_random_sampler,
    )


# ---- Sequential sampler ------------------------------------------------
@dataclass(frozen=True)
class SequentialSamplerConfig:
    pass


def _build_sequential_sampler(cfg: SequentialSamplerConfig, dataset):
    from torch.utils.data import SequentialSampler
    return SequentialSampler(dataset)


@register_ingredient
def _sequential_sampler():
    return IngredientSpec(
        name="sequential", kind="sampler", config_cls=SequentialSamplerConfig,
        cache_relevant=True,
        build=_build_sequential_sampler,
    )


# ---- Weighted sampler --------------------------------------------------
@dataclass(frozen=True)
class WeightedSamplerConfig:
    weights: tuple  # per-sample weights
    num_samples: Optional[int] = None
    replacement: bool = True


def _build_weighted_sampler(cfg: WeightedSamplerConfig, dataset):
    from torch.utils.data import WeightedRandomSampler
    return WeightedRandomSampler(
        weights=list(cfg.weights),
        num_samples=cfg.num_samples or len(cfg.weights),
        replacement=cfg.replacement,
    )


@register_ingredient
def _weighted_sampler():
    return IngredientSpec(
        name="weighted", kind="sampler", config_cls=WeightedSamplerConfig,
        cache_relevant=True,
        build=_build_weighted_sampler,
    )
