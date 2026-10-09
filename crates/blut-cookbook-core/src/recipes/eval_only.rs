//! Recipe — `eval_only`.
//!
//! Evaluation-only pipeline:
//!
//!   load_dataset → evaluate_model
//!
//! Loads a dataset and runs evaluation against a checkpoint. No training.

use serde::{Deserialize, Serialize};

use blut::framework::error::RecipeError;
use blut::framework::plan::Plan;
use blut::recipes::recipe::Recipe;

use crate::recipes::shared::DatasetSelect;
use crate::stages::IngredientCfg;
use crate::stages::evaluate_model::{EvaluateLoadedDataset, LoadedDatasetArgs};
use crate::stages::load_dataset::{Args as LoadArgs, LoadDataset};

#[derive(Default)]
pub struct EvalOnly;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// HuggingFace dataset name. Exclusive with `dataset_path`.
    #[serde(default)]
    pub hf_name: Option<String>,
    /// Path to a local JSONL/CSV dataset. Exclusive with `hf_name`.
    #[serde(default)]
    pub dataset_path: Option<std::path::PathBuf>,
    /// Dataset split (default: "test").
    #[serde(default = "default_split")]
    pub split: String,
    /// Path to the checkpoint to evaluate.
    pub checkpoint_path: std::path::PathBuf,
    /// Eval ingredient config.
    #[serde(default = "default_eval")]
    pub eval: IngredientCfg,
    /// Batch size.
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    /// Device to evaluate on.
    #[serde(default = "default_device")]
    pub device: String,
    /// `subset` and `max_samples`.
    #[serde(flatten)]
    pub dataset: DatasetSelect,
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
fn default_eval() -> IngredientCfg {
    IngredientCfg {
        kind: "eval".into(),
        name: "loss_eval".into(),
        config: serde_json::Value::Object(Default::default()),
    }
}
fn default_batch_size() -> u32 {
    64
}
fn default_device() -> String {
    "cuda".into()
}

impl Recipe for EvalOnly {
    type Backend = blut_backends::LamuTrainerBackend;
    const NAME: &'static str = "eval_only";
    const DESCRIPTION: &'static str =
        "Evaluation-only: load dataset → evaluate checkpoint. No training.";
    type Args = Args;
    const CATEGORY: blut::recipes::recipe::RecipeCategory =
        blut::recipes::recipe::RecipeCategory::Eval;
    const INPUT_KINDS: &'static [&'static str] = &[];
    const OUTPUT_KIND: &'static str = "eval.report";

    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError> {
        match (&args.hf_name, &args.dataset_path) {
            (Some(_), Some(_)) => {
                return Err(RecipeError::InvalidArgs {
                    field: None,
                    message: "pass exactly one of hf_name / dataset_path, not both".into(),
                });
            }
            (None, None) => {
                return Err(RecipeError::InvalidArgs {
                    field: None,
                    message: "one of hf_name / dataset_path is required".into(),
                });
            }
            _ => {}
        }
        if !args.checkpoint_path.exists() {
            return Err(RecipeError::InvalidArgs {
                field: Some("checkpoint_path"),
                message: format!("not found: {}", args.checkpoint_path.display()),
            });
        }

        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;

        // The eval_only recipe needs a special flow: load_dataset produces
        // DatasetJsonl, but evaluate_model needs (HfCheckpoint, DatasetJsonl).
        // We use a fork pattern: load the dataset, then pair it with the
        // checkpoint in the evaluate stage.
        //
        // For now, we use a simplified approach: load_dataset → evaluate_model
        // where the checkpoint path is embedded in the eval args.
        let plan = Plan::new(Self::NAME, recipe_args_json)
            .start(
                LoadDataset,
                LoadArgs {
                    path: args.dataset_path.clone(),
                    hf_name: args.hf_name.clone(),
                    split: args.split.clone(),
                    subset: args.dataset.subset.clone(),
                    max_samples: args.dataset.max_samples,
                },
            )
            .then(
                EvaluateLoadedDataset,
                LoadedDatasetArgs {
                    checkpoint_path: args.checkpoint_path,
                    eval: args.eval,
                    batch_size: args.batch_size,
                    device: args.device,
                    text_field: args.text_field,
                    max_seq_len: args.max_seq_len,
                },
            )
            .finish();
        Ok(plan)
    }
}

blut::register_recipe!(EvalOnly);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_schema_is_valid() {
        let schema = schemars::schema_for!(Args);
        assert!(schema.schema.object.is_some());
    }

    #[test]
    fn compiles_to_load_then_evaluate() {
        let temp = tempfile::tempdir().unwrap();
        let checkpoint = temp.path().join("checkpoint");
        std::fs::create_dir(&checkpoint).unwrap();
        let dataset = temp.path().join("eval.jsonl");
        std::fs::write(&dataset, "{}\n").unwrap();
        let plan = EvalOnly
            .compile(Args {
                hf_name: None,
                dataset_path: Some(dataset),
                split: "test".into(),
                checkpoint_path: checkpoint,
                eval: default_eval(),
                batch_size: 1,
                device: "cpu".into(),
                dataset: DatasetSelect::default(),
                text_field: None,
                max_seq_len: None,
            })
            .unwrap()
            .into_compiled();
        assert_eq!(plan.n_nodes(), 2);
        assert_eq!(plan.n_edges(), 1);
    }
}
