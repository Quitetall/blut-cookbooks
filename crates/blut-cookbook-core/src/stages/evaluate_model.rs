//! Generic checkpoint evaluation stages.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use blut::artifacts::{DatasetJsonl, EvalReport, HfCheckpoint};
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

use super::shared::IngredientCfg;

pub struct EvaluateModel;
pub struct EvaluateLoadedDataset;
/// Evaluate a checkpoint on held-out rows that arrive as a plan edge.
///
/// The merge end of a `split -> fork(train, take_eval) -> merge` recipe.
/// `evaluate_model` fetches its own rows from a dataset name, and the recipe
/// that used it passed the TRAINING split's name — so its report measured
/// the model on the data it had just trained on.
pub struct EvaluateHeldOut;

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
    /// HuggingFace dataset config/subset name, for datasets that publish
    /// several (`wikitext` has no loadable default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subset: Option<String>,
    /// Cap the rows loaded. Unset takes the whole split.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_samples: Option<usize>,
    /// Evaluation ingredient.
    pub eval: IngredientCfg,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_device")]
    pub device: String,
    /// Causal-LM packing, as for training (evaluator default `"text"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_field: Option<String>,
    /// Causal-LM packing, as for training (evaluator default 512).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_seq_len: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct LoadedDatasetArgs {
    pub checkpoint_path: PathBuf,
    pub eval: IngredientCfg,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_device")]
    pub device: String,
    /// Causal-LM packing, as for training (evaluator default `"text"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_field: Option<String>,
    /// Causal-LM packing, as for training (evaluator default 512).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_seq_len: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct HeldOutArgs {
    /// Evaluation ingredient.
    pub eval: IngredientCfg,
    /// The loss ingredient training used; `loss_eval` scores with it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss: Option<IngredientCfg>,
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_device")]
    pub device: String,
    /// Causal-LM packing, as for training (evaluator default `"text"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_field: Option<String>,
    /// Causal-LM packing, as for training (evaluator default 512).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_seq_len: Option<u32>,
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
            let mut cmd = std::process::Command::new("python3");
            cmd.args(["-m", "blut_core.load_dataset", "--name", name, "--split"])
                .arg(&args.split)
                .arg("--output")
                .arg(&output);
            if let Some(subset) = &args.subset {
                cmd.args(["--subset", subset]);
            }
            if let Some(max) = args.max_samples {
                cmd.arg("--max-samples").arg(max.to_string());
            }
            let status = cmd.status().map_err(|error| {
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

#[allow(clippy::too_many_arguments)]
fn run_evaluator(
    ctx: &StageContext,
    checkpoint_path: &Path,
    dataset_path: &Path,
    eval: &IngredientCfg,
    loss: Option<&IngredientCfg>,
    batch_size: u32,
    device: &str,
    text_field: Option<&str>,
    max_seq_len: Option<u32>,
) -> Result<EvalReport, StageError> {
    let to_json = |cfg: &IngredientCfg| {
        serde_json::to_value(cfg)
            .map_err(|error| StageError::Backend(anyhow::anyhow!("serialize ingredient: {error}")))
    };
    let eval = to_json(eval)?;
    let loss = loss.map(to_json).transpose()?;
    blut_backends::evaluator::run_evaluator(
        ctx,
        &blut_backends::evaluator::EvalRequest {
            checkpoint_path,
            dataset_path,
            eval,
            loss,
            batch_size,
            device,
            text_field,
            max_seq_len,
        },
        "blut_core.evaluator",
    )
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
            None,
            args.batch_size,
            &args.device,
            args.text_field.as_deref(),
            args.max_seq_len,
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
            None,
            args.batch_size,
            &args.device,
            args.text_field.as_deref(),
            args.max_seq_len,
        )
    }
}

#[async_trait]
impl Stage for EvaluateHeldOut {
    const NAME: &'static str = "evaluate_held_out";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    type Input = (HfCheckpoint, DatasetJsonl);
    type Output = EvalReport;
    type Args = HeldOutArgs;

    async fn run(
        &self,
        ctx: &StageContext,
        input: (HfCheckpoint, DatasetJsonl),
        args: &HeldOutArgs,
    ) -> Result<EvalReport, StageError> {
        let (checkpoint, held_out) = input;
        if held_out.n_examples <= 0 {
            return Err(StageError::BadInput(format!(
                "evaluate_held_out: held-out split has {} examples",
                held_out.n_examples
            )));
        }
        run_evaluator(
            ctx,
            &checkpoint.path,
            &held_out.path,
            &args.eval,
            args.loss.as_ref(),
            args.batch_size,
            &args.device,
            args.text_field.as_deref(),
            args.max_seq_len,
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
            subset: None,
            max_samples: None,
            text_field: None,
            max_seq_len: None,
        };
        assert_eq!(resolve_dataset(&ctx, &args).unwrap(), path);
    }
}
