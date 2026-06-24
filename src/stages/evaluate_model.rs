//! Stage — `evaluate_model`.
//!
//! Generic evaluation stage that invokes the ingredient-based evaluator
//! (`blut_core.evaluator`). Takes a checkpoint, runs evaluation against
//! a dataset specified in Args, and returns an `EvalReport`.
//!
//! Deterministic — same checkpoint + same dataset → identical metrics.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use blut::artifacts::{EvalReport, HfCheckpoint};
use blut::framework::artifact::ContentHash;
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

use super::shared::IngredientCfg;

pub struct EvaluateModel;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Path to the evaluation dataset (JSONL).
    pub dataset_path: PathBuf,
    /// Eval ingredient (e.g. {"kind": "eval", "name": "loss_eval", "config": {}}).
    pub eval: IngredientCfg,
    /// Batch size for evaluation.
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    /// Device to evaluate on.
    #[serde(default = "default_device")]
    pub device: String,
}

fn default_batch_size() -> u32 { 64 }
fn default_device() -> String { "cuda".to_string() }

#[async_trait]
impl Stage for EvaluateModel {
    const NAME: &'static str = "evaluate_model";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    type Input = HfCheckpoint;
    type Output = EvalReport;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: HfCheckpoint,
        args: &Args,
    ) -> Result<EvalReport, StageError> {
        if !input.path.exists() {
            return Err(StageError::BadInput(format!(
                "checkpoint not found: {}", input.path.display()
            )));
        }
        if !args.dataset_path.exists() {
            return Err(StageError::BadInput(format!(
                "dataset not found: {}", args.dataset_path.display()
            )));
        }

        // Write eval config JSON
        let config = serde_json::json!({
            "checkpoint_path": input.path,
            "dataset_path": args.dataset_path,
            "eval": args.eval,
            "batch_size": args.batch_size,
            "device": args.device,
        });
        let config_path = ctx.stage_dir.join("eval_config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config).unwrap())
            .map_err(|source| StageError::Io {
                path: config_path.clone(),
                source,
            })?;

        // Invoke the generic evaluator
        let output_path = ctx.stage_dir.join("eval_report.json");
        let status = std::process::Command::new("python3")
            .args([
                "-m", "blut_core.evaluator",
                "--config", config_path.to_str().unwrap(),
                "--output", output_path.to_str().unwrap(),
            ])
            .current_dir(&ctx.stage_dir)
            .status()
            .map_err(|e| StageError::Backend(
                anyhow::anyhow!("failed to run blut_core.evaluator: {}", e)
            ))?;

        if !status.success() {
            return Err(StageError::Backend(
                anyhow::anyhow!("blut_core.evaluator exited with {}", status)
            ));
        }

        // Parse the eval report
        let report_json = std::fs::read_to_string(&output_path)
            .map_err(|source| StageError::Io {
                path: output_path.clone(),
                source,
            })?;
        let metrics: serde_json::Value = serde_json::from_str(&report_json)
            .map_err(|e| StageError::Backend(
                anyhow::anyhow!("failed to parse eval report: {}", e)
            ))?;

        let content_hash = ContentHash::hash_file(&output_path)
            .map_err(|source| StageError::Io {
                path: output_path.clone(),
                source,
            })?;

        Ok(EvalReport {
            path: output_path,
            evaluator: "blut_core.evaluator".into(),
            metrics,
            content_hash,
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
}
