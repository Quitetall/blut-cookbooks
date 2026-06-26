//! Recipe — `finetune_pretrained`.
//!
//! Fine-tuning pipeline with LoRA:
//!
//!   load_dataset → split_train_eval → take_train → train_model
//!     → merge_lora → convert_gguf → register_model
//!
//! Uses the ingredient system for the training loop. The model ingredient
//! should be `lora_adapter` for LoRA fine-tuning.

use serde::{Deserialize, Serialize};

use blut::framework::error::RecipeError;
use blut::framework::plan::Plan;
use blut::recipes::recipe::Recipe;
use blut_backends::stages::take_train::TakeTrain;
use blut_backends::stages::{
    convert_gguf::{Args as ConvertArgs, ConvertGguf},
    merge_lora::{Args as MergeArgs, MergeLora},
    register_model::{Args as RegArgs, RegisterModel},
    split_train_eval::{Args as SplitArgs, SplitTrainEval},
};

use crate::stages::load_dataset::{Args as LoadArgs, LoadDataset};
use crate::stages::train_model::{Args as TrainArgs, TrainModel};
use crate::stages::IngredientCfg;

#[derive(Default)]
pub struct FinetunePretrained;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// HuggingFace dataset name. Exclusive with `dataset_path`.
    #[serde(default)]
    pub hf_name: Option<String>,
    /// Path to a local JSONL/CSV dataset. Exclusive with `hf_name`.
    #[serde(default)]
    pub dataset_path: Option<std::path::PathBuf>,
    /// Dataset split (default: "train").
    #[serde(default = "default_split")]
    pub split: String,
    /// Model ingredient config (should be lora_adapter for LoRA fine-tuning).
    pub model: IngredientCfg,
    /// Optimizer ingredient config.
    pub optimizer: IngredientCfg,
    /// Scheduler ingredient config.
    pub scheduler: IngredientCfg,
    /// Loss ingredient config.
    pub loss: IngredientCfg,
    /// Training step ingredient (default: standard).
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
    /// Train/eval split ratio.
    #[serde(default = "default_eval_ratio")]
    pub eval_ratio: f32,
    /// Device to train on.
    #[serde(default = "default_device")]
    pub device: String,
    /// Output model name for registry.
    pub output_name: String,
    /// GGUF quantization (default: Q4_K_M).
    #[serde(default = "default_quant")]
    pub quant: String,
    /// Notes for model registry.
    #[serde(default)]
    pub notes: String,
}

fn default_split() -> String { "train".into() }
fn default_step() -> IngredientCfg {
    IngredientCfg { kind: "step".into(), name: "standard".into(), config: serde_json::Value::Object(Default::default()) }
}
fn default_batch_size() -> u32 { 1 }
fn default_seed() -> u64 { 42 }
fn default_eval_ratio() -> f32 { 0.1 }
fn default_device() -> String { "cuda".into() }
fn default_quant() -> String { "Q4_K_M".into() }

impl Recipe for FinetunePretrained {
    type Backend = blut_backends::LamuTrainerBackend;
    const NAME: &'static str = "finetune_pretrained";
    const DESCRIPTION: &'static str = "Fine-tune a pretrained model with LoRA: load dataset → split → train → merge → convert → register.";
    type Args = Args;
    const CATEGORY: blut::recipes::recipe::RecipeCategory =
        blut::recipes::recipe::RecipeCategory::Train;
    const INPUT_KINDS: &'static [&'static str] = &[];
    const OUTPUT_KIND: &'static str = "model.gguf";

    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError> {
        match (&args.hf_name, &args.dataset_path) {
            (Some(_), Some(_)) => return Err(RecipeError::InvalidArgs(
                "pass exactly one of hf_name / dataset_path, not both".into(),
            )),
            (None, None) => return Err(RecipeError::InvalidArgs(
                "one of hf_name / dataset_path is required".into(),
            )),
            _ => {}
        }
        if args.output_name.is_empty() {
            return Err(RecipeError::InvalidArgs("output_name is empty".into()));
        }
        if args.epochs == 0 {
            return Err(RecipeError::InvalidArgs("epochs must be > 0".into()));
        }

        let recipe_args_json = serde_json::to_value(&args)
            .map_err(|e| RecipeError::CompileFailed(format!("serialize args: {e}")))?;

        let plan = Plan::new(Self::NAME, recipe_args_json)
            .start(
                LoadDataset,
                LoadArgs {
                    path: args.dataset_path.clone(),
                    hf_name: args.hf_name.clone(),
                    split: args.split.clone(),
                    subset: None,
                    max_samples: None,
                },
            )
            .then(
                SplitTrainEval,
                SplitArgs {
                    eval_ratio: args.eval_ratio,
                    seed: args.seed,
                },
            )
            .then(TakeTrain, blut_backends::stages::take_train::Args::default())
            .then(
                TrainModel,
                TrainArgs {
                    model: args.model.clone(),
                    optimizer: args.optimizer.clone(),
                    scheduler: args.scheduler.clone(),
                    loss: args.loss.clone(),
                    step: args.step.clone(),
                    epochs: args.epochs,
                    batch_size: args.batch_size,
                    seed: args.seed,
                    device: args.device.clone(),
                    lora_rank: None,
                    nproc_per_node: 1,
                    nnodes: 1,
                },
            )
            .then(MergeLora, MergeArgs::default())
            .then(
                ConvertGguf,
                ConvertArgs {
                    quant: args.quant.clone(),
                    name: args.output_name.clone(),
                },
            )
            .then(
                RegisterModel,
                RegArgs {
                    name: args.output_name.clone(),
                    notes: args.notes.clone(),
                    arch: "finetuned".into(),
                },
            )
            .finish();
        Ok(plan)
    }
}

blut::register_recipe!(FinetunePretrained);

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            hf_name: Some("imdb".into()),
            dataset_path: None,
            split: "train".into(),
            model: IngredientCfg {
                kind: "model".into(),
                name: "lora_adapter".into(),
                config: serde_json::json!({"base_model": "meta-llama/Llama-3-8B", "r": 16}),
            },
            optimizer: IngredientCfg {
                kind: "optimizer".into(),
                name: "adamw".into(),
                config: serde_json::json!({"lr": 2e-4}),
            },
            scheduler: IngredientCfg {
                kind: "scheduler".into(),
                name: "cosine".into(),
                config: serde_json::json!({"total_epochs": 3}),
            },
            loss: IngredientCfg {
                kind: "loss".into(),
                name: "cross_entropy".into(),
                config: serde_json::Value::Object(Default::default()),
            },
            step: default_step(),
            epochs: 3,
            batch_size: 1,
            seed: 42,
            eval_ratio: 0.1,
            device: "cuda".into(),
            output_name: "my-finetuned-model".into(),
            quant: "Q4_K_M".into(),
            notes: String::new(),
        }
    }

    #[test]
    fn compiles_to_7_node_plan() {
        let plan = FinetunePretrained.compile(args()).unwrap().into_compiled();
        assert_eq!(plan.n_nodes(), 7);
        assert_eq!(plan.n_edges(), 6);
    }

    #[test]
    fn rejects_empty_output_name() {
        let mut a = args();
        a.output_name = String::new();
        let r = FinetunePretrained.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs(_))));
    }
}
