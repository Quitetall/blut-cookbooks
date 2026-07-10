"""Generic logging ingredient specs.

Provides the BLUT_METRIC stdout emitter — the standard observability sink
for all BLUT trainers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from blut_core.registry import register_ingredient
from blut_core.spec import IngredientSpec


@dataclass(frozen=True)
class BlutMetricConfig:
    pass


def _build_blut_metric(cfg):
    def emit(metrics, *, kind="epoch", phase=None):
        """Emit one runner-parseable ``BLUT_METRIC <json>`` line.

        Args:
            metrics: a dict of per-epoch metrics.
            kind: the event tag (``'epoch'`` for the per-epoch line).
            phase: optional phase tag; omitted when None.

        Returns the emitted JSON string (also printed + flushed to stdout).
        """
        payload = {k: v for k, v in metrics.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        payload['kind'] = kind
        if phase is not None:
            payload['phase'] = phase
        line = 'BLUT_METRIC ' + json.dumps(payload)
        print(line, flush=True)
        return line

    return emit


@register_ingredient
def _blut_metric_spec():
    return IngredientSpec(
        name="blut_metric", kind="logging", config_cls=BlutMetricConfig,
        cache_relevant=False,
        build=_build_blut_metric,
    )
