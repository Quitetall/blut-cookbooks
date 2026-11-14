"""LoRA fine-tuning of image-text models (AutoModelForImageTextToText) on a
streamed text-prompt dataset, with mid-run checkpoint and exact resume.

Selected by `arch == "image_text"` in the job spec. Runs under the separate
image-text venv profile (transformers 5.x); see `venv.rs`. Behavioural
reference: KnightLM's `model/train/harness.py` (completion-only loss, LoRA on
the text decoder's q/v, rank 8, finite-gradient and adapter-reload checks).

Why a hand-written loop instead of `transformers.Trainer`: a Trainer resume
needs a map-style dataset or its own sampler skipping, and does not promise
the same subsequent losses. Here the data position is one integer (see
`hf_image_text_data`), so a resumed run replays the identical batches.

Defaults, for ~210k single-turn rows, a 0.5B model and one RTX 4090:
  * bf16 base weights, fp32 LoRA weights (what peft makes of a bf16 base).
  * r=8 / alpha=16 and lr=2e-4 are the harness values; 2e-4 is the usual
    LoRA learning rate. Warmup 3% then cosine to 0 (the common recipe for a
    single pass); grad clip 1.0 (the harness value).
  * Effective batch ~32 = `batch_size` x `grad_accum` (e.g. 8 x 4): at 210k
    rows that is ~6.5k optimizer steps, enough for the cosine to matter and
    small enough to stay stable at lr 2e-4.
  * Checkpoint every 500 steps (~8% of an epoch), keep the last 3.
Nothing here was tuned on a holdout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import hf_image_text_data as data

DEFAULTS: Dict[str, Any] = {
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.03,
    "max_grad_norm": 1.0,
    "save_steps": 500,
    "keep_checkpoints": 3,
    "shuffle_buffer": 10_000,
    "logging_steps": 10,
}

# Only these `extra` keys mean anything to the image-text path; anything else
# is refused rather than silently ignored.
_NO_EXTRA = "image_text jobs do not take `extra` keys; use the typed fields"

# LoRA targets by model_type. SmolVLM (Idefics3): the harness regex.
_LORA_REGEX = {
    "idefics3": r"model\.text_model\.layers\.\d+\.self_attn\.[qv]_proj",
    # Qwen3.5 (Gated-DeltaNet hybrid): q/v on the full-attention layers (every
    # 4th), and the fused q/k/v input projection on the linear-attention layers,
    # the closest analogue of q/v there. The vision tower is not touched.
    "qwen3_5": (r"model\.language_model\.layers\.\d+\."
                r"(self_attn\.[qv]_proj|linear_attn\.in_proj_qkv)"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def opt(spec: Dict[str, Any], key: str) -> Any:
    value = spec.get(key)
    return DEFAULTS[key] if value is None else value


def resolve_lora_regex(spec: Dict[str, Any], model_type: str, module_names: List[str]) -> str:
    """Explicit regex wins; else the per-architecture default; else refuse."""
    import re

    regex = spec.get("lora_target_regex")
    if not regex:
        regex = _LORA_REGEX.get(model_type)
    if not regex:
        raise ValueError(
            f"no default LoRA target for model_type {model_type!r}; "
            "set lora_target_regex"
        )
    pat = re.compile(regex)
    hits = [n for n in module_names if pat.fullmatch(n)]
    if not hits:
        raise ValueError(f"lora_target_regex {regex!r} matches no module")
    return regex


# --------------------------------------------------------------------------
# checkpoints


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def list_checkpoints(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("step-")),
        key=lambda p: int(p.name.split("-", 1)[1]),
    )


def save_checkpoint(root: Path, step: int, consumed: int, model, optimizer, scheduler,
                    extra_state: Dict[str, Any], keep: int) -> Path:
    """Write adapter + optimizer + scheduler + RNG + data position atomically.

    Everything lands in `.tmp-step-N` first; one rename publishes it, so a
    crash leaves either the previous checkpoint or this one, never a mix.
    """
    import torch

    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step-{step:08d}"
    tmp = root / f".tmp-step-{step:08d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    model.save_pretrained(tmp / "adapter")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": {
                "python": list(random.getstate()[1]),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        tmp / "train_state.pt",
    )
    (tmp / "state.json").write_text(
        json.dumps({"step": step, "rows_consumed": consumed, **extra_state})
    )
    for f in [*(tmp / "adapter").iterdir(), tmp / "train_state.pt", tmp / "state.json"]:
        with open(f, "rb") as fh:
            os.fsync(fh.fileno())
    _fsync_dir(tmp / "adapter")
    _fsync_dir(tmp)
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    _fsync_dir(root)
    for old in list_checkpoints(root)[:-max(1, keep)]:
        shutil.rmtree(old, ignore_errors=True)
    return final


def load_checkpoint(path: Path, model, optimizer, scheduler) -> Dict[str, Any]:
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(model, load_file(path / "adapter" / "adapter_model.safetensors"))
    st = torch.load(path / "train_state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(st["optimizer"])
    scheduler.load_state_dict(st["scheduler"])
    rng = st["rng"]
    random.setstate((3, tuple(rng["python"]), None))
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return json.loads((path / "state.json").read_text())


# --------------------------------------------------------------------------
# training


def _tensors(batch, pad_id: int, device: str):
    import torch

    cols = data.collate(batch, pad_id)
    return {k: torch.tensor(v, device=device) for k, v in cols.items()}


def run(spec: Dict[str, Any], emit: Callable[[Dict[str, Any]], None]) -> None:
    import peft
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor, get_scheduler

    if spec.get("extra"):
        raise ValueError(f"{_NO_EXTRA}: {sorted(spec['extra'])}")
    pc = spec.get("peft")
    if not pc or pc.get("method") != "lora":
        raise ValueError("image_text supports method 'lora' only (not qlora/full)")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("image_text training requires CUDA with bfloat16 support")

    seed = int(spec["seed"])
    bs, ga = int(spec["batch_size"]), int(spec["grad_accum"])
    epochs, seq_len = int(spec["epochs"]), int(spec["seq_len"])
    out_dir = Path(spec["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(spec.get("checkpoint_root") or out_dir / "checkpoints")
    shards = [Path(p) for p in (spec.get("dataset_shards") or [spec["train_dataset_path"]])]
    for s in shards:
        if not s.is_file():
            raise FileNotFoundError(f"dataset shard missing: {s}")

    base = spec["base_model"]
    expected = spec.get("base_sha256")
    if expected:
        got = sha256_file(Path(base) / "model.safetensors")
        if got != expected:
            raise ValueError(f"base weights sha256 {got} != expected {expected}")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()

    local = Path(base).is_dir()
    processor = AutoProcessor.from_pretrained(base, local_files_only=local)
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def load_base():
        return AutoModelForImageTextToText.from_pretrained(
            base, local_files_only=local, dtype=torch.bfloat16
        ).to("cuda")

    base_model = load_base()
    model_type = base_model.config.model_type
    regex = resolve_lora_regex(spec, model_type, [n for n, _ in base_model.named_modules()])
    lora_cfg = peft.LoraConfig(
        r=int(pc["rank"]), lora_alpha=int(pc["alpha"]), target_modules=regex, bias="none"
    )
    model = peft.get_peft_model(base_model, lora_cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(params, lr=float(spec["lr"]))

    n_rows = data.count_rows(shards)
    per_step = bs * ga
    total = spec.get("max_steps") or (epochs * (n_rows // per_step))
    total = int(total)
    if total < 1:
        raise ValueError(f"{n_rows} rows x {epochs} epochs is less than one step of {per_step} rows")
    warmup = math.ceil(float(opt(spec, "warmup_ratio")) * total)
    scheduler = get_scheduler(
        opt(spec, "lr_scheduler"), optimizer, num_warmup_steps=warmup, num_training_steps=total
    )

    step, consumed, dropped = 0, 0, 0
    if spec.get("resume"):
        ckpts = list_checkpoints(ckpt_root)
        if ckpts:
            st = load_checkpoint(ckpts[-1], model, optimizer, scheduler)
            step, consumed = int(st["step"]), int(st["rows_consumed"])
            dropped = int(st.get("dropped_long", 0))
            emit({"kind": "resumed", "step": step, "path": str(ckpts[-1])})

    stream = data.RowStream(shards, seed, int(opt(spec, "shuffle_buffer")), epochs, skip=consumed)
    batcher = data.Batcher(stream, processor, bs, seq_len)
    batcher.dropped_long = dropped
    save_steps, keep = int(opt(spec, "save_steps")), int(opt(spec, "keep_checkpoints"))
    log_steps, max_norm = int(opt(spec, "logging_steps")), float(opt(spec, "max_grad_norm"))

    model.train()
    first_params = [p.detach().cpu().clone() for p in params]
    win_tokens = 0
    win_start = time.monotonic()
    last_loss: Optional[float] = None
    max_seen_norm = 0.0
    while step < total:
        window = []
        for _ in range(ga):
            b = batcher.next_batch()
            if b is None:
                break
            window.append(b)
        if len(window) < ga:
            break  # data exhausted before `total`: drop the partial window
        n_tok = sum(len(ids) - len(p) for b in window for p, ids in b)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for b in window:
            inputs = _tensors(b, pad_id, "cuda")
            count = int(inputs["labels"].ne(-100).sum())
            loss = model(**inputs, use_cache=False).loss
            lv = float(loss.detach())
            if not math.isfinite(lv):
                raise RuntimeError(f"non-finite training loss at step {step + 1}")
            # token-weighted mean over the whole accumulation window
            (loss * (count / n_tok)).backward()
            loss_sum += lv * count
            win_tokens += int(inputs["attention_mask"].sum())
        norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm, error_if_nonfinite=True))
        max_seen_norm = max(max_seen_norm, norm)
        lr_now = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
        step += 1
        last_loss = loss_sum / n_tok
        if step == 1 or step % log_steps == 0 or step == total:
            dt = max(time.monotonic() - win_start, 1e-9)
            emit({
                "kind": "step", "step": step, "total": total, "loss": last_loss,
                "lr": lr_now, "grad_norm": norm, "tokens_per_second": win_tokens / dt,
            })
            win_tokens = 0
            win_start = time.monotonic()
        if save_steps and step % save_steps == 0 and step < total:
            path = save_checkpoint(ckpt_root, step, stream.consumed, model, optimizer,
                                   scheduler, {"dropped_long": batcher.dropped_long}, keep)
            emit({"kind": "saved", "path": str(path)})

    if step == 0 and last_loss is None:
        raise RuntimeError("no optimizer step ran")
    if step > 0 and max_seen_norm <= 0:
        raise RuntimeError("gradients were all zero; LoRA did not train")
    final = save_checkpoint(ckpt_root, step, stream.consumed, model, optimizer, scheduler,
                            {"dropped_long": batcher.dropped_long}, keep)
    emit({"kind": "saved", "path": str(final)})
    model.save_pretrained(out_dir)
    changed = any(not torch.equal(a, p.detach().cpu()) for a, p in zip(first_params, params))
    if not changed and step > 0 and not spec.get("resume"):
        raise RuntimeError("LoRA training did not update adapter weights")

    reload_err = None
    if spec.get("verify_reload", True):
        reload_err = _verify_reload(model, out_dir, load_base, batcher, pad_id)
    summary = {
        "steps": step, "total_steps": total, "rows_consumed": stream.consumed,
        "dropped_long": batcher.dropped_long, "trainable_parameters": trainable,
        "lora_target_regex": regex, "model_type": model_type,
        "max_grad_norm_seen": max_seen_norm, "adapter_reload_max_logit_error": reload_err,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "transformers": transformers.__version__, "peft": peft.__version__,
        "torch": torch.__version__,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    emit({"kind": "done", "checkpoint_dir": str(out_dir), "final_loss": last_loss, **{
        k: summary[k] for k in ("steps", "rows_consumed", "dropped_long")}})


def _verify_reload(model, out_dir: Path, load_base, batcher, pad_id) -> float:
    """Saved adapter on a fresh base must give bit-identical logits."""
    import peft
    import torch

    prompt = "Reply with one word."
    pids, ids = data.encode(prompt, "ok", batcher.processor)
    probe = _tensors([(pids, ids)], pad_id, "cuda")
    probe.pop("labels")
    model.eval()
    with torch.inference_mode():
        expected = model(**probe, use_cache=False).logits[:, -1, :].float().cpu()
    reloaded = peft.PeftModel.from_pretrained(load_base(), out_dir).eval()
    with torch.inference_mode():
        actual = reloaded(**probe, use_cache=False).logits[:, -1, :].float().cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return float((actual - expected).abs().max())


def run_merge(spec: Dict[str, Any], emit: Callable[[Dict[str, Any]], None]) -> None:
    """Fold a LoRA adapter into an image-text base model (`task == "merge"`)."""
    import peft
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base, adapter = spec["base_model"], spec["adapter_path"]
    out = Path(spec["output_dir"])
    local = Path(base).is_dir()
    model = AutoModelForImageTextToText.from_pretrained(
        base, local_files_only=local, dtype=torch.bfloat16
    )
    merged = peft.PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    AutoProcessor.from_pretrained(base, local_files_only=local).save_pretrained(out)
    emit({"kind": "done", "checkpoint_dir": str(out), "final_loss": None})
