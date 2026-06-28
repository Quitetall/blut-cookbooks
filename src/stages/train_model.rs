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

use blut::artifacts::{DatasetJsonl, HfCheckpoint};
use blut::framework::artifact::ContentHash;
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

use super::shared::IngredientCfg;

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
    /// Optional LoRA rank (if using lora_adapter model ingredient).
    #[serde(default)]
    pub lora_rank: Option<u32>,
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ParallelStrategy {
    #[default]
    Ddp,
    Fsdp,
}

fn default_nproc() -> u32 { 1 }
fn default_nnodes() -> u32 { 1 }

fn default_step() -> IngredientCfg {
    IngredientCfg {
        kind: "step".into(),
        name: "standard".into(),
        config: serde_json::Value::Object(Default::default()),
    }
}
fn default_batch_size() -> u32 { 32 }
fn default_seed() -> u64 { 42 }
fn default_device() -> String { "cuda".to_string() }

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
        let base = Self::MEMORY_GIB.max(4);  // minimum 4 GiB per process
        base * args.nproc_per_node.max(1)
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
        if !input.path.exists() {
            return Err(StageError::BadInput(format!(
                "dataset not found: {}", input.path.display()
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
        });
        let config_path = ctx.stage_dir.join("train_config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap())
            .map_err(|source| StageError::Io {
                path: config_path.clone(),
                source,
            })?;

        // Invoke the generic trainer (via torchrun when DDP)
        let nproc = args.nproc_per_node.max(1);
        let nnodes = args.nnodes.max(1);
        let mut cmd = std::process::Command::new("python3");
        if nproc > 1 {
            // DDP mode: launch via torchrun
            cmd.args(["-m", "torch.distributed.run"]);
            if nnodes <= 1 {
                // Single-node DDP
                cmd.args(["--standalone", "--nproc_per_node", &nproc.to_string()]);
            } else {
                // Multi-node DDP: read rendezvous from env (set by Slurm launcher)
                let master_addr = std::env::var("MASTER_ADDR")
                    .unwrap_or_else(|_| {
                        tracing::warn!("MASTER_ADDR not set for multi-node DDP, using 127.0.0.1");
                        "127.0.0.1".into()
                    });
                let master_port = std::env::var("MASTER_PORT")
                    .unwrap_or_else(|_| "29500".into());
                let node_rank = std::env::var("NODE_RANK")
                    .unwrap_or_else(|_| {
                        tracing::warn!("NODE_RANK not set for multi-node DDP, using 0");
                        "0".into()
                    });
                cmd.args([
                    "--nnodes", &nnodes.to_string(),
                    "--nproc_per_node", &nproc.to_string(),
                    "--rdzv_backend", "c10d",
                    "--rdzv_endpoint", &format!("{master_addr}:{master_port}"),
                    "--node_rank", &node_rank,
                ]);
            }
        }
        cmd.args([
            "-m", "blut_core.trainer",
            "--config", config_path.to_str().unwrap(),
        ]);
        cmd.current_dir(&ctx.stage_dir);

        // For DDP: don't pin to a single device — torchrun manages LOCAL_RANK
        if nproc <= 1 {
            if let Some(dev) = ctx.device_index {
                cmd.env("CUDA_VISIBLE_DEVICES", dev.to_string());
            }
        }

        let status = cmd.status().map_err(|e| StageError::Backend(
            anyhow::anyhow!("failed to run blut_core.trainer: {}", e)
        ))?;

        if !status.success() {
            return Err(StageError::Backend(
                anyhow::anyhow!("blut_core.trainer exited with {}", status)
            ));
        }

        // Verify output exists
        if !output_dir.exists() || output_dir.read_dir().map_or(true, |mut d| d.next().is_none()) {
            return Err(StageError::Backend(
                anyhow::anyhow!("trainer did not produce checkpoint in {}", output_dir.display())
            ));
        }

        let content_hash = ContentHash::hash_dir(&output_dir)
            .map_err(|e| StageError::Backend(anyhow::anyhow!("hash error: {}", e)))?;

        Ok(HfCheckpoint {
            path: output_dir,
            base_model: args.model.name.clone(),
            method_tag: "full".into(),
            content_hash,
            final_loss: 0.0,  // filled by trainer
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let bad: Result<ParallelStrategy, _> =
            serde_json::from_value(serde_json::json!("zero3"));
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
