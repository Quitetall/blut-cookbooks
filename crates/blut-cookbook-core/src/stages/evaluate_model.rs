//! Generic checkpoint evaluation stages.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use blut::artifacts::{DatasetJsonl, EvalReport, HfCheckpoint};
use blut::framework::artifact::ContentHash;
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

use super::shared::IngredientCfg;

pub struct EvaluateModel;
pub struct EvaluateLoadedDataset;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Local evaluation dataset. Exclusive with `hf_name`.
    #[serde(default)]
    pub dataset_path: Option<PathBuf>,
    /// HuggingFace dataset name. Exclusive with `dataset_path`.
    #[serde(default)]
    pub hf_name: Option<String>,
    /// HuggingFace split used when `hf_name` is selected.
    #[serde(default = "default_split")]
    pub split: String,
    /// Evaluation ingredient.
    pub eval: IngredientCfg,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_device")]
    pub device: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct LoadedDatasetArgs {
    pub checkpoint_path: PathBuf,
    pub eval: IngredientCfg,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_device")]
    pub device: String,
}

fn default_split() -> String {
    "test".into()
}

fn default_batch_size() -> u32 {
    64
}

fn default_device() -> String {
    "cuda".into()
}

fn resolve_dataset(ctx: &StageContext, args: &Args) -> Result<PathBuf, StageError> {
    match (&args.dataset_path, &args.hf_name) {
        (Some(_), Some(_)) => Err(StageError::BadInput(
            "evaluate_model: pass exactly one of dataset_path / hf_name, not both".into(),
        )),
        (None, None) => Err(StageError::BadInput(
            "evaluate_model: one of dataset_path / hf_name is required".into(),
        )),
        (Some(path), None) => {
            if !path.is_file() {
                return Err(StageError::BadInput(format!(
                    "dataset not found: {}",
                    path.display()
                )));
            }
            Ok(path.clone())
        }
        (None, Some(name)) => {
            let output = ctx.stage_dir.join("evaluation_dataset.jsonl");
            let status = std::process::Command::new("python3")
                .args(["-m", "blut_core.load_dataset", "--name", name, "--split"])
                .arg(&args.split)
                .arg("--output")
                .arg(&output)
                .status()
                .map_err(|error| {
                    StageError::Backend(anyhow::anyhow!(
                        "failed to run blut_core.load_dataset: {error}"
                    ))
                })?;
            if !status.success() {
                return Err(StageError::Backend(anyhow::anyhow!(
                    "blut_core.load_dataset exited with {status}"
                )));
            }
            if !output.is_file() {
                return Err(StageError::Backend(anyhow::anyhow!(
                    "blut_core.load_dataset did not produce {}",
                    output.display()
                )));
            }
            Ok(output)
        }
    }
}

fn run_evaluator(
    ctx: &StageContext,
    checkpoint_path: &Path,
    dataset_path: &Path,
    eval: &IngredientCfg,
    batch_size: u32,
    device: &str,
) -> Result<EvalReport, StageError> {
    if !checkpoint_path.exists() {
        return Err(StageError::BadInput(format!(
            "checkpoint not found: {}",
            checkpoint_path.display()
        )));
    }
    if !dataset_path.is_file() {
        return Err(StageError::BadInput(format!(
            "dataset not found: {}",
            dataset_path.display()
        )));
    }

    let config = serde_json::json!({
        "checkpoint_path": checkpoint_path,
        "dataset_path": dataset_path,
        "eval": eval,
        "batch_size": batch_size,
        "device": device,
    });
    let config_path = ctx.stage_dir.join("eval_config.json");
    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|error| StageError::Backend(anyhow::anyhow!("serialize eval config: {error}")))?;
    std::fs::write(&config_path, config_json).map_err(|source| StageError::Io {
        path: config_path.clone(),
        source,
    })?;

    let output_path = ctx.stage_dir.join("eval_report.json");
    let status = std::process::Command::new("python3")
        .args(["-m", "blut_core.evaluator", "--config"])
        .arg(&config_path)
        .arg("--output")
        .arg(&output_path)
        .current_dir(&ctx.stage_dir)
        .status()
        .map_err(|error| {
            StageError::Backend(anyhow::anyhow!(
                "failed to run blut_core.evaluator: {error}"
            ))
        })?;
    if !status.success() {
        return Err(StageError::Backend(anyhow::anyhow!(
            "blut_core.evaluator exited with {status}"
        )));
    }

    let report_json = std::fs::read_to_string(&output_path).map_err(|source| StageError::Io {
        path: output_path.clone(),
        source,
    })?;
    let metrics = serde_json::from_str(&report_json).map_err(|error| {
        StageError::Backend(anyhow::anyhow!("failed to parse eval report: {error}"))
    })?;
    let content_hash = ContentHash::hash_file(&output_path).map_err(|source| StageError::Io {
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

#[async_trait]
impl Stage for EvaluateModel {
    const NAME: &'static str = "evaluate_model";
    const SCHEMA: u32 = 2;
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
        let dataset_path = resolve_dataset(ctx, args)?;
        run_evaluator(
            ctx,
            &input.path,
            &dataset_path,
            &args.eval,
            args.batch_size,
            &args.device,
        )
    }
}

#[async_trait]
impl Stage for EvaluateLoadedDataset {
    const NAME: &'static str = "evaluate_loaded_dataset";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    type Input = DatasetJsonl;
    type Output = EvalReport;
    type Args = LoadedDatasetArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: DatasetJsonl,
        args: &LoadedDatasetArgs,
    ) -> Result<EvalReport, StageError> {
        run_evaluator(
            ctx,
            &args.checkpoint_path,
            &input.path,
            &args.eval,
            args.batch_size,
            &args.device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemas_are_valid() {
        assert!(schemars::schema_for!(Args).schema.object.is_some());
        assert!(
            schemars::schema_for!(LoadedDatasetArgs)
                .schema
                .object
                .is_some()
        );
    }

    #[test]
    fn local_dataset_source_resolves() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("eval.jsonl");
        std::fs::write(&path, "{}\n").unwrap();
        let ctx = StageContext::for_test(temp.path().into(), temp.path().join("stage"));
        let args = Args {
            dataset_path: Some(path.clone()),
            hf_name: None,
            split: "test".into(),
            eval: IngredientCfg {
                kind: "eval".into(),
                name: "loss_eval".into(),
                config: serde_json::json!({}),
            },
            batch_size: 1,
            device: "cpu".into(),
        };
        assert_eq!(resolve_dataset(&ctx, &args).unwrap(), path);
    }
}
