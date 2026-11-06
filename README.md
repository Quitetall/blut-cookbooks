# BLUT Cookbooks

Reusable cookbooks for the [BLUT](https://github.com/Quitetall/blut) ML
orchestrator: concrete stages, training backends, recipes, and the Python
runtime they launch.

| Crate | Binary | What it holds |
|---|---|---|
| `blut-cookbook-standard` | `blut-standard` | Generic-LLM stages (materialize, filter, split, SFT/DPO training, LoRA merge, GGUF conversion, evaluation, the adapters that put a dataset behind the engine's checks) and the `hf_trainer` and `lamu` training backends |
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
version, 1.99.

## Train a model

`train_from_dataset` loads a HuggingFace dataset, splits it, trains, and
evaluates. For a HuggingFace causal LM, text rows are tokenized with the
model's own tokenizer and packed into fixed-length blocks; use the `causal_lm`
loss.

```sh
blut-core recipe run train_from_dataset --args '{
  "hf_name": "Salesforce/wikitext", "subset": "wikitext-2-raw-v1", "max_samples": 2000,
  "model":     {"kind": "model", "name": "from_pretrained",
                "config": {"model_name": "HuggingFaceTB/SmolLM2-135M"}},
  "optimizer": {"kind": "optimizer", "name": "adamw", "config": {"lr": 1e-4}},
  "scheduler": {"kind": "scheduler", "name": "constant", "config": {}},
  "loss":      {"kind": "loss", "name": "causal_lm", "config": {}},
  "epochs": 1, "batch_size": 8, "max_seq_len": 256
}'
```

`blut-core recipe show train_from_dataset` prints the full argument schema.

The `constant`, `cosine`, `linear` and `wsd` schedulers set each epoch's learning
rate before that epoch trains. Their warmup is in epochs and starts with the
first one: `constant` defaults to `warmup_epochs: 5`, so the run above, at one
epoch, trains at a fifth of the optimizer's `lr`. Set `"warmup_epochs": 0` for
none.

## Guarding a dataset

The engine's `check_jsonl` (row-count bounds) and `assert` (a scalar
predicate) stages stop a plan before bad data reaches training. They take the
engine's `checks.jsonl` artifact, and the data stages here emit
`dataset.jsonl`, so a guard sits between `to_checks_jsonl` and
`from_checks_jsonl`:

```text
load_dataset → to_checks_jsonl → check_jsonl → from_checks_jsonl → split_train_eval
```

Neither adapter copies the file. `from_checks_jsonl` hands on the row count the
guard enforced (non-blank lines, as `load_dataset` counts them), so the
consumer gets the dataset it would have had without the guard. Put the guard
on the path to the consumer, as here: a guard on a side branch does not stop
the consumer.

`crates/blut-cookbook-core/examples/guarded_dataset.json` is this plan as a
declared recipe. Set its dataset path and row bounds, then:

```sh
blut-core recipe declare crates/blut-cookbook-core/examples/guarded_dataset.json --run
blut-core checks report <job>
```

With `on_breach: "block"` (the default) a breach fails the guard and nothing
after it runs; `"warn"` records the breach and continues. `check_jsonl` checks
only the row count, so a malformed row that keeps the count in bounds passes.
`assert` compares the `actual` value written in its arguments; nothing feeds
it a measured one. Compiled recipes of either backend can use the same stages.

## Distributed training

Set `nproc_per_node` for several GPUs on one machine. Set `nnodes` as well to
span machines; every node then needs the rendezvous in its environment:

| Variable | Meaning |
|---|---|
| `MASTER_ADDR` | Address of the rank-0 node — required |
| `NODE_RANK` | This node's index, `0` to `nnodes - 1` — required |
| `MASTER_PORT` | Rendezvous port; defaults to `29500` |

For a manual launch, run the same recipe on every node with its own
`NODE_RANK`. A missing `MASTER_ADDR` or `NODE_RANK` fails the stage immediately
rather than leaving each node waiting at a rendezvous of its own.

With `--launcher slurm`, run the recipe once, from a host that can submit jobs.
Each training stage becomes one `srun` step with `--nodes=<nnodes>`, one task per
node, and `--gpus-per-node=<nproc_per_node>` unless `device` is `cpu`. Every
node takes its rank from `SLURM_NODEID` and the coordinator from the first host
in the allocation (or `MASTER_ADDR`, if set). The step is named
`blut-<job>-<node>`, the name the engine records with the BLUT job, so
`blut-core cancel` can stop it and `plan resume` can see it after the process
that submitted it is gone. `BLUT_SLURM_PARTITION`, `_MEM`,
`_CPUS`, `_TIME` and `_GPUS` pass through. A `BLUT_SLURM_NODES` or
`_NTASKS_PER_NODE` that contradicts the stage is an error. The job directory
must be on a filesystem every node can read. This holds for `train_model`,
`hf_sft_train`, `hf_dpo_train`, and the lamu backend's `sft_train`. Other
launchers are refused by the training stages rather than run locally.

`parallel_strategy` selects `ddp` (replicate the model, the default) or `fsdp`
(FSDP2: shard parameters, gradients, and optimizer state). The `hf_sft_train`
and `hf_dpo_train` stages take the same `nproc_per_node` and `nnodes`.

## Following a training stage

`blut-core log <job>` prints a training stage's progress and `blut-core jobs`
its latest step and loss, read from the updates the stage forwards in the
engine's trainer format. `vram` is the GPU memory the trainer's process holds
(0 off the GPU). No value a trainer did not measure is filled in.

- `train_model` and `train_model_on_split` forward each step logged by
  `log_every_steps` (its loss, the optimizer's learning rate), each checkpoint
  save, every epoch as `… epoch E/N loss=<mean>`, and, as the final loss, the
  last epoch's mean loss. Without `log_every_steps` there are no step lines.
  The trainer writes these to stdout as `BLUT_STATUS <json>` lines, which the
  stage forwards and keeps off the terminal; the `[trainer]` lines are
  unchanged.
- `sft_train`, `hf_sft_train` and `hf_dpo_train` forward each step, save, and
  the final loss. The HF stages also forward each evaluation HF logs, as
  `eval @<step>  loss=<eval_loss>`. An HF log with no loss or learning rate
  (the end-of-run summary) appears as `… step N/M`, and a run that trained no
  step reports no final loss.

A step whose loss is not a finite number, or whose learning rate cannot be
read, appears as `… step N/M` as well. A non-finite loss from `train_model`, or
from the HF stages, also carries the string `"nan"`, `"inf"` or `"-inf"` on the
forwarded update. That is how the engine recognizes a diverged run: `blut hpo`
kills such a trial, and so does any plan under `BLUT_EXECUTOR=parallel` with
`BLUT_KILL_ON_NAN=1`. It is seen when a loss is logged: at `log_every_steps`,
otherwise at the end of the epoch.

The checkpoint these stages produce records the final loss the trainer
reported, or none when it reported none (a run that trained no step).

## Resuming an interrupted training stage

With BLUT 0.2.0-alpha.2 or later, `train_model` and `train_model_on_split`
checkpoint during training and resume from the newest checkpoint when the stage
runs again: after a retry, after `blut-core plan resume <job>`, or when a Slurm
batch job that ran the recipe is requeued and runs it again. The engine keys the
checkpoint directory by the stage's arguments and inputs, so a changed config or
dataset starts from step 0. See the engine's `CONTRACT.md` §8.

| Argument | Meaning |
|---|---|
| `checkpoint_every_steps` | Checkpoint every N steps |
| `checkpoint_every_secs` | Otherwise every N seconds (default 600) |
| `log_every_steps` | Print and report the loss every N steps |
| `train_attempts` | Attempts for the training stage, the first included (default 1) |

A resumed run continues the step count, the data order, the optimizer and RNG
state, and the loss, so it ends with the weights an uninterrupted run would
have. The trainer keeps the two newest checkpoints and skips one that is
partial, corrupt, or written for another config or world size. Under DDP only
rank 0 reads and writes checkpoints; the directory (`$BLUT_CHECKPOINTS_DIR`,
default beside the jobs directory) must be on rank 0's node. FSDP runs do not
checkpoint mid-stage and restart from step 0.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
