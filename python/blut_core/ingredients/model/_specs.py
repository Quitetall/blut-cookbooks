"""Generic model ingredient specs.

Provides model loaders for common patterns: HuggingFace pretrained models,
custom PyTorch modules, and LoRA adapters. Domain cookbooks override with
their own architectures (e.g. LamQuant's JointCodec).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from blut_core.registry import register_ingredient
from blut_core.spec import IngredientSpec


# ---- HuggingFace pretrained model ---------------------------------------
@dataclass(frozen=True)
class FromPretrainedConfig:
    model_name: str                     # HF model id or local path
    torch_dtype: str = "float32"        # "float32", "float16", "bfloat16"
    trust_remote_code: bool = False
    device_map: Optional[str] = None    # "auto", "cpu", None


def _build_from_pretrained(cfg: FromPretrainedConfig):
    """Return a HuggingFace model loaded from pretrained weights."""
    from transformers import AutoModel, AutoModelForCausalLM
    dtype_map = {
        "float32": __import__("torch").float32,
        "float16": __import__("torch").float16,
        "bfloat16": __import__("torch").bfloat16,
    }
    dtype = dtype_map.get(cfg.torch_dtype)
    # Try causal LM first, fall back to generic model
    try:
        return AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            device_map=cfg.device_map,
        )
    except (ValueError, OSError):
        return AutoModel.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
            device_map=cfg.device_map,
        )


@register_ingredient
def _from_pretrained():
    return IngredientSpec(
        name="from_pretrained", kind="model", config_cls=FromPretrainedConfig,
        cache_relevant=True,
        build=_build_from_pretrained,
        requires=("pkg:transformers",),
    )


# ---- Custom PyTorch module ----------------------------------------------
@dataclass(frozen=True)
class CustomModuleConfig:
    module_path: str        # e.g. "myproject.models.MyNet"
    init_kwargs: dict = None  # kwargs to pass to the constructor

    def __post_init__(self):
        if self.init_kwargs is None:
            object.__setattr__(self, 'init_kwargs', {})


def _build_custom_module(cfg: CustomModuleConfig):
    """Load a PyTorch module by Python dotted path."""
    import importlib
    parts = cfg.module_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"module_path must be 'package.module.ClassName', "
            f"got {cfg.module_path!r}")
    mod = importlib.import_module(parts[0])
    cls = getattr(mod, parts[1])
    return cls(**cfg.init_kwargs)


@register_ingredient
def _custom_module():
    return IngredientSpec(
        name="custom_module", kind="model", config_cls=CustomModuleConfig,
        cache_relevant=True,
        build=_build_custom_module,
    )


# ---- LoRA adapter -------------------------------------------------------
@dataclass(frozen=True)
class LoraAdapterConfig:
    base_model: str                 # HF model id or local path
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj")
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    torch_dtype: str = "float16"


def _build_lora_adapter(cfg: LoraAdapterConfig):
    """Return a PEFT-wrapped model with LoRA adapters."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    import torch
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(cfg.torch_dtype, torch.float16)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=dtype)
    lora_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias=cfg.bias,
        task_type=cfg.task_type,
    )
    return get_peft_model(base, lora_config)


@register_ingredient
def _lora_adapter():
    return IngredientSpec(
        name="lora_adapter", kind="model", config_cls=LoraAdapterConfig,
        cache_relevant=True,
        build=_build_lora_adapter,
        requires=("pkg:peft", "pkg:transformers"),
    )
