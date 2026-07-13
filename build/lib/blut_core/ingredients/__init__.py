"""Generic ingredient specs — the core training primitives.

Importing this module registers all generic specs into the global registry.
Domain cookbooks import ``blut_core`` first (registering these), then register
their own domain-specific specs on top.
"""

# Import all _specs modules to trigger their @register_ingredient decorators.
from blut_core.ingredients.data import _specs as _data_specs  # noqa: F401
from blut_core.ingredients.model import _specs as _model_specs  # noqa: F401
from blut_core.ingredients.optimizer import _specs as _optimizer_specs  # noqa: F401
from blut_core.ingredients.scheduler import _specs as _scheduler_specs  # noqa: F401
from blut_core.ingredients.loss import _specs as _loss_specs  # noqa: F401
from blut_core.ingredients.step import _specs as _step_specs  # noqa: F401
from blut_core.ingredients.ema import _specs as _ema_specs  # noqa: F401
from blut_core.ingredients.checkpoint import _specs as _checkpoint_specs  # noqa: F401
from blut_core.ingredients.eval import _specs as _eval_specs  # noqa: F401
from blut_core.ingredients.sampler import _specs as _sampler_specs  # noqa: F401
from blut_core.ingredients.logging import _specs as _logging_specs  # noqa: F401
from blut_core.ingredients.forward import _specs as _forward_specs  # noqa: F401
