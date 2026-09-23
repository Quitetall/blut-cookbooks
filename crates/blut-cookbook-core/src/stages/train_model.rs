//! Stage — `train_model`.
//!
//! Generic training stage that invokes the ingredient-based trainer
//! (`blut_core.trainer`). Takes a `DatasetJsonl` input, writes an
//! ingredient config JSON, shells out to the Python trainer, and
//! returns an `HfCheckpoint`.
//!
//! Nondeterministic (training is stochastic).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use blut::artifacts::{DatasetJsonl, DatasetSplit, HfCheckpoint};
use blut::framework::artifact::ContentHash;
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

use super::shared::IngredientCfg;
use blut_backends::distributed::{Rendezvous, torchrun_args};

pub struct TrainModel;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Model ingredient (e.g. {"kind": "model", "name": "from_pretrained", "config": {...}}).
    pub model: IngredientCfg,
    /// Optimizer ingredient.
    pub optimizer: IngredientCfg,
    /// Scheduler ingredient.
    pub scheduler: IngredientCfg,
    /// Loss ingredient.
    pub loss: IngredientCfg,
    /// Training step ingredient (default: "standard").
    #[serde(default = "default_step")]
    pub step: IngredientCfg,
    /// Number of training epochs.
    pub epochs: u32,
    /// Batch size per GPU.
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    /// Random seed.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Device to train on.
    #[serde(default = "default_device")]
    pub device: String,
    /// Not read by the trainer: LoRA rank is set in the `lora_adapter` model
    /// ingredient's own config. Kept so existing arguments still deserialize
    /// and cache keys do not move; setting it is rejected rather than
    /// silently ignored.
    #[serde(default)]
    pub lora_rank: Option<u32>,
    /// Causal-LM training: the dataset column holding the text. The trainer
    /// reads `"text"` when unset. Only consulted for a HuggingFace model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_field: Option<String>,
    /// Causal-LM training: tokens per packed training block. The trainer uses
    /// 512 when unset. Only consulted for a HuggingFace model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_seq_len: Option<u32>,
    /// Number of GPUs for DDP. 1 = single-GPU (default). >1 = torchrun DDP.
    #[serde(default = "default_nproc")]
    pub nproc_per_node: u32,
    /// Number of DDP nodes (for multi-node). Default 1.
    #[serde(default = "default_nnodes")]
    pub nnodes: u32,
    /// Multi-GPU parallel strategy: `Ddp` (default, replicate) or `Fsdp`
    /// (FSDP2 fully_shard — shard params/grads/optimizer state, ZeRO-3).
    /// Only meaningful when nproc_per_node > 1. A typed enum so a typo is
    /// rejected at deserialize time, not at the Python boundary.
    #[serde(default)]
    pub parallel_strategy: ParallelStrategy,
}

/// Multi-GPU parallel strategy. `Ddp` (default) replicates the model per rank;
/// `Fsdp` (FSDP2 fully_shard) shards params/grads/optimizer state (ZeRO-3).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum ParallelStrategy {
    #[default]
    Ddp,
    Fsdp,
}

impl ParallelStrategy {
    /// Serde `skip_serializing_if` for the default strategy.
    pub fn is_ddp(&self) -> bool {
        *self == ParallelStrategy::Ddp
    }
}

fn default_nproc() -> u32 {
    1
}
fn default_nnodes() -> u32 {
    1
}

fn default_step() -> IngredientCfg {
    IngredientCfg {
        kind: "step".into(),
        name: "standard".into(),
        config: serde_json::Value::Object(Default::default()),
    }
}
fn default_batch_size() -> u32 {
    32
}
fn default_seed() -> u64 {
    42
}
fn default_device() -> String {
    "cuda".to_string()
}

#[async_trait]
impl Stage for TrainModel {
    const NAME: &'static str = "train_model";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const DETERMINISTIC: bool = false;
    type Input = DatasetJsonl;
    type Output = HfCheckpoint;
    type Args = Args;

    /// DDP: hold N GPU permits when nproc_per_node > 1.
    fn gpu_permits(&self, args: &Args) -> u32 {
        args.nproc_per_node.max(1)
    }

    /// DDP: scale memory reservation by nproc.
    fn memory_gib_for(&self, args: &Args) -> u32 {
        4 * args.nproc_per_node.max(1)
    }

    async fn run(
        &self,
        ctx: &StageContext,
        input: DatasetJsonl,
        args: &Args,
    ) -> Result<HfCheckpoint, StageError> {
        // Validate inputs
        if args.epochs == 0 {
            return Err(StageError::BadInput("epochs must be > 0".into()));
        }
        if args.batch_size == 0 {
            return Err(StageError::BadInput("batch_size must be > 0".into()));
        }
        if args.lora_rank.is_some() {
            return Err(StageError::BadInput(
                "lora_rank is not read by the trainer; set the rank in the \
                 lora_adapter model ingredient's config instead"
                    .into(),
            ));
        }
        if !input.path.exists() {
            return Err(StageError::BadInput(format!(
                "dataset not found: {}",
                input.path.display()
            )));
        }

        let output_dir = ctx.stage_dir.join("checkpoint");
        std::fs::create_dir_all(&output_dir).map_err(|source| StageError::Io {
            path: output_dir.clone(),
            source,
        })?;

        // Write ingredient config JSON
        let config = serde_json::json!({
            "dataset_path": input.path,
            "output_dir": output_dir,
            "model": args.model,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "loss": args.loss,
            "step": args.step,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": args.device,
            "lora_rank": args.lora_rank,
            "parallel_strategy": args.parallel_strategy,
            "text_field": args.text_field,
            "max_seq_len": args.max_seq_len,
        });
        let config_path = ctx.stage_dir.join("train_config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap()).map_err(
            |source| StageError::Io {
                path: config_path.clone(),
                source,
            },
        )?;

        // Invoke the generic trainer (via torchrun when the run is distributed)
        let nproc = args.nproc_per_node.max(1);
        let nnodes = args.nnodes.max(1);
        let rdzv = if nnodes > 1 {
            Some(Rendezvous::from_env().map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?)
        } else {
            None
        };
        let mut cmd = std::process::Command::new("python3");
        if let Some(launch) = torchrun_args(nproc, nnodes, rdzv.as_ref()) {
            cmd.args(&launch);
        }
        cmd.args([
            "-m",
            "blut_core.trainer",
            "--config",
            config_path.to_str().unwrap(),
        ]);
        cmd.current_dir(&ctx.stage_dir);

        // Pin only when this node runs a single process. With several local
        // ranks torchrun assigns devices through LOCAL_RANK and a pin would
        // collapse every rank onto one card. One rank per node (the usual
        // multi-node shape) still pins: LOCAL_RANK is 0 there, so device 0
        // must be the card the scheduler actually granted.
        if nproc <= 1 {
            if let Some(dev) = ctx.device_index {
                cmd.env("CUDA_VISIBLE_DEVICES", dev.to_string());
            }
        }

        let status = cmd.status().map_err(|e| {
            StageError::Backend(anyhow::anyhow!("failed to run blut_core.trainer: {e}"))
        })?;

        if !status.success() {
            return Err(StageError::Backend(anyhow::anyhow!(
                "blut_core.trainer exited with {status}"
            )));
        }

        // Verify output exists
        if !output_dir.exists()
            || output_dir
                .read_dir()
                .map_or(true, |mut d| d.next().is_none())
        {
            return Err(StageError::Backend(anyhow::anyhow!(
                "trainer did not produce checkpoint in {}",
                output_dir.display()
            )));
        }

        let content_hash = ContentHash::hash_dir(&output_dir)
            .map_err(|e| StageError::Backend(anyhow::anyhow!("hash error: {e}")))?;

        Ok(HfCheckpoint {
            path: output_dir,
            base_model: args.model.name.clone(),
            method_tag: "full".into(),
            content_hash,
            final_loss: 0.0, // filled by trainer
        })
    }
}

/// `train_model` over the train half of a `DatasetSplit`.
///
/// Exists so a recipe can fork a split into this training edge and a
/// `take_eval` edge, then merge the checkpoint with the held-out rows. The
/// linear alternative — `take_train` then `train_model` — drops the eval half
/// on the floor, which left `train_from_dataset` with nothing held out to
/// evaluate on and let it quietly evaluate on its own training split instead.
pub struct TrainModelOnSplit;

#[async_trait]
impl Stage for TrainModelOnSplit {
    const NAME: &'static str = "train_model_on_split";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    const DETERMINISTIC: bool = false;
    type Input = DatasetSplit;
    type Output = HfCheckpoint;
    type Args = Args;

    fn gpu_permits(&self, args: &Args) -> u32 {
        TrainModel.gpu_permits(args)
    }

    fn memory_gib_for(&self, args: &Args) -> u32 {
        TrainModel.memory_gib_for(args)
    }

    async fn run(
        &self,
        ctx: &StageContext,
        input: DatasetSplit,
        args: &Args,
    ) -> Result<HfCheckpoint, StageError> {
        if input.train.n_examples <= 0 {
            return Err(StageError::BadInput(format!(
                "train_model_on_split: train half has {} examples",
                input.train.n_examples
            )));
        }
        TrainModel.run(ctx, input.train, args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `lora_rank` used to be accepted and then ignored: the trainer never
    /// read it. A knob that silently does nothing must be refused, and before
    /// any Python is spawned.
    #[tokio::test]
    async fn lora_rank_is_refused_not_ignored() {
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("stage")).unwrap();
        let ctx = StageContext::for_test(td.path().to_path_buf(), td.path().join("stage"));
        let data = td.path().join("train.jsonl");
        std::fs::write(&data, r#"{"text":"hi"}"#).unwrap();
        let input = DatasetJsonl {
            path: data,
            content_hash: ContentHash::of_bytes(b"x"),
            n_examples: 1,
        };
        let mut args: Args = serde_json::from_value(serde_json::json!({
            "model": {"kind": "model", "name": "from_pretrained", "config": {}},
            "optimizer": {"kind": "optimizer", "name": "adamw", "config": {}},
            "scheduler": {"kind": "scheduler", "name": "constant", "config": {}},
            "loss": {"kind": "loss", "name": "causal_lm", "config": {}},
            "epochs": 1,
        }))
        .unwrap();
        args.lora_rank = Some(16);
        let r = TrainModel.run(&ctx, input, &args).await;
        assert!(
            matches!(&r, Err(StageError::BadInput(m)) if m.contains("lora_adapter")),
            "got {r:?}"
        );
    }

    #[test]
    fn args_schema_is_valid() {
        let schema = schemars::schema_for!(Args);
        assert!(schema.schema.object.is_some());
    }

    #[test]
    fn parallel_strategy_defaults_to_ddp() {
        // Omitted in JSON → "ddp" (back-compat: existing specs are unchanged).
        let json = serde_json::json!({
            "model": {"kind": "model", "name": "from_pretrained", "config": {}},
            "optimizer": {"kind": "optimizer", "name": "adamw", "config": {}},
            "scheduler": {"kind": "scheduler", "name": "constant", "config": {}},
            "loss": {"kind": "loss", "name": "cross_entropy", "config": {}},
            "epochs": 1,
        });
        let args: Args = serde_json::from_value(json).unwrap();
        assert_eq!(args.parallel_strategy, ParallelStrategy::Ddp);
        assert_eq!(args.nproc_per_node, 1);
    }

    #[test]
    fn parallel_strategy_fsdp_roundtrips() {
        let json = serde_json::json!({
            "model": {"kind": "model", "name": "from_pretrained", "config": {}},
            "optimizer": {"kind": "optimizer", "name": "adamw", "config": {}},
            "scheduler": {"kind": "scheduler", "name": "constant", "config": {}},
            "loss": {"kind": "loss", "name": "cross_entropy", "config": {}},
            "epochs": 1,
            "nproc_per_node": 2,
            "parallel_strategy": "fsdp",
        });
        let args: Args = serde_json::from_value(json).unwrap();
        assert_eq!(args.parallel_strategy, ParallelStrategy::Fsdp);
        assert_eq!(args.nproc_per_node, 2);
    }

    #[test]
    fn parallel_strategy_rejects_unknown() {
        // A typo is caught by serde at deserialize, not deferred to Python.
        let bad: Result<ParallelStrategy, _> = serde_json::from_value(serde_json::json!("zero3"));
        assert!(bad.is_err());
    }

    #[test]
    fn ingredient_cfg_serde_roundtrip() {
        let cfg = IngredientCfg {
            kind: "optimizer".into(),
            name: "adamw".into(),
            config: serde_json::json!({"lr": 1e-3}),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: IngredientCfg = serde_json::from_str(&json).unwrap();
        assert_eq!(back.kind, "optimizer");
        assert_eq!(back.name, "adamw");
    }
}
