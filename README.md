# BLUT Cookbooks

Reusable cookbooks for the [BLUT](https://github.com/Quitetall/blut) ML
orchestrator: concrete stages, training backends, recipes, and the Python
runtime they launch.

| Crate | Binary | What it holds |
|---|---|---|
| `blut-cookbook-standard` | `blut-standard` | Generic-LLM stages (materialize, filter, split, SFT/DPO training, LoRA merge, GGUF conversion, evaluation) and the `hf_trainer` and `lamu` training backends |
| `blut-cookbook-core` | `blut-core` | Generic ML recipes — `train_from_dataset`, `finetune_pretrained`, `eval_only` — built from swappable Python ingredients |

`python/blut_core` is the Python runtime both crates launch: the ingredient
registry (models, optimizers, schedulers, losses, training steps), the generic
trainer, and the run context, metric log, and checkpoint contract. Install it,
or put this repository's `python` directory on `PYTHONPATH`.

The engine comes from crates.io (`blut = "=0.2.0-alpha.1"`); this repository
builds with no sibling checkouts. LamQuant may add untracked local Cargo
overrides for editable development.

## Build and test

```sh
cargo test --workspace --all-targets --locked
PYTHONPATH=python python -m pytest -q python
```

CI runs both, plus `rustfmt`, `clippy -D warnings`, `cargo-deny` (advisories,
licenses, sources), and a compile check on the declared minimum Rust
version, 1.88.

## Train a model

`train_from_dataset` loads a HuggingFace dataset, splits it, trains, and
evaluates. For a HuggingFace causal LM, text rows are tokenized with the
model's own tokenizer and packed into fixed-length blocks; use the `causal_lm`
loss.

```sh
blut-core recipe run train_from_dataset --args '{
  "hf_name": "wikitext", "subset": "wikitext-2-raw-v1", "max_samples": 2000,
  "model":     {"kind": "model", "name": "from_pretrained",
                "config": {"model_name": "HuggingFaceTB/SmolLM2-135M"}},
  "optimizer": {"kind": "optimizer", "name": "adamw", "config": {"lr": 1e-4}},
  "scheduler": {"kind": "scheduler", "name": "constant", "config": {}},
  "loss":      {"kind": "loss", "name": "causal_lm", "config": {}},
  "epochs": 1, "batch_size": 8, "max_seq_len": 256
}'
```

`blut-core recipe show train_from_dataset` prints the full argument schema.

## Distributed training

Set `nproc_per_node` for several GPUs on one machine. Set `nnodes` as well to
span machines; every node then needs the rendezvous in its environment:

| Variable | Meaning |
|---|---|
| `MASTER_ADDR` | Address of the rank-0 node — required |
| `NODE_RANK` | This node's index, `0` to `nnodes - 1` — required |
| `MASTER_PORT` | Rendezvous port; defaults to `29500` |

The Slurm launcher exports all three. For a manual launch, run the same recipe
on every node with its own `NODE_RANK`. A missing `MASTER_ADDR` or `NODE_RANK`
fails the stage immediately rather than leaving each node waiting at a
rendezvous of its own.

`parallel_strategy` selects `ddp` (replicate the model, the default) or `fsdp`
(FSDP2: shard parameters, gradients, and optimizer state). The `hf_sft_train`
and `hf_dpo_train` stages take the same `nproc_per_node` and `nnodes`.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
