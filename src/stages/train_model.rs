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
    /// Batch size.
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
}

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
        });
        let config_path = ctx.stage_dir.join("train_config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap())
            .map_err(|source| StageError::Io {
                path: config_path.clone(),
                source,
            })?;

        // Invoke the generic trainer
        let status = std::process::Command::new("python3")
            .args([
                "-m", "blut_core.trainer",
                "--config", config_path.to_str().unwrap(),
            ])
            .current_dir(&ctx.stage_dir)
            .status()
            .map_err(|e| StageError::Backend(
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
