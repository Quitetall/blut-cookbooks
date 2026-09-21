//! Recipe — `train_from_dataset`.
//!
//! Generic training pipeline:
//!
//!   load_dataset → split_train_eval → take_train → train_model → evaluate_model
//!
//! Works with any dataset (HuggingFace or local CSV/JSONL) and any model
//! (via the ingredient system). Zero domain code required.

use serde::{Deserialize, Serialize};

use blut::framework::error::RecipeError;
use blut::framework::plan::Plan;
use blut::recipes::recipe::Recipe;
use blut_backends::stages::split_train_eval::{Args as SplitArgs, SplitTrainEval};
use blut_backends::stages::take_train::TakeTrain;

use crate::stages::IngredientCfg;
use crate::stages::evaluate_model::{Args as EvalArgs, EvaluateModel};
use crate::stages::load_dataset::{Args as LoadArgs, LoadDataset};
use crate::stages::train_model::{Args as TrainArgs, ParallelStrategy, TrainModel};

#[derive(Default)]
pub struct TrainFromDataset;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// HuggingFace dataset name (e.g. "imdb"). Exclusive with `dataset_path`.
    #[serde(default)]
    pub hf_name: Option<String>,
    /// Path to a local JSONL/CSV dataset. Exclusive with `hf_name`.
    #[serde(default)]
    pub dataset_path: Option<std::path::PathBuf>,
    /// Dataset split (default: "train").
    #[serde(default = "default_split")]
    pub split: String,
    /// Model ingredient config.
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
    /// Processes (GPUs) per node. 1 = single process. >1 = local DDP.
    #[serde(default = "default_nproc")]
    pub nproc_per_node: u32,
    /// Nodes in the job. >1 requires MASTER_ADDR and NODE_RANK in the
    /// environment of every node; the Slurm launcher exports both.
    #[serde(default = "default_nnodes")]
    pub nnodes: u32,
    /// `ddp` (replicate, default) or `fsdp` (FSDP2 shard). Only meaningful
    /// once the run is distributed.
    #[serde(default)]
    pub parallel_strategy: ParallelStrategy,
}

fn default_split() -> String {
    "train".into()
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
fn default_eval_ratio() -> f32 {
    0.1
}
fn default_device() -> String {
    "cuda".into()
}

impl Recipe for TrainFromDataset {
    type Backend = blut_backends::LamuTrainerBackend;
    const NAME: &'static str = "train_from_dataset";
    const DESCRIPTION: &'static str = "Generic training pipeline: load dataset → split → train → evaluate. \
        Works with any HuggingFace or local dataset and any model via the ingredient system.";
    type Args = Args;
    const CATEGORY: blut::recipes::recipe::RecipeCategory =
        blut::recipes::recipe::RecipeCategory::Train;
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
        if args.epochs == 0 {
            return Err(RecipeError::InvalidArgs {
                field: Some("epochs"),
                message: "must be > 0".into(),
            });
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
            .then(
                TakeTrain,
                blut_backends::stages::take_train::Args::default(),
            )
            .then(TrainModel, train_args(&args))
            .then(
                EvaluateModel,
                EvalArgs {
                    dataset_path: args.dataset_path.clone(),
                    hf_name: args.hf_name.clone(),
                    split: args.split.clone(),
                    eval: IngredientCfg {
                        kind: "eval".into(),
                        name: "loss_eval".into(),
                        config: serde_json::Value::Object(Default::default()),
                    },
                    batch_size: args.batch_size * 2,
                    device: args.device.clone(),
                },
            )
            .finish();
        Ok(plan)
    }
}

/// Map the recipe's arguments onto the `train_model` stage.
///
/// Named and separate so the mapping is testable: the engine keeps compiled
/// plan nodes `pub(crate)`, so a test outside the engine cannot read a stage's
/// arguments back off a plan. The distributed fields in particular were once
/// hard-coded to `1` here while the stage supported more, which no test could
/// have caught through `compile` alone.
fn train_args(args: &Args) -> TrainArgs {
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
        nproc_per_node: args.nproc_per_node,
        nnodes: args.nnodes,
        parallel_strategy: args.parallel_strategy,
    }
}

blut::register_recipe!(TrainFromDataset);

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            hf_name: Some("imdb".into()),
            dataset_path: None,
            split: "train".into(),
            nproc_per_node: 1,
            nnodes: 1,
            parallel_strategy: ParallelStrategy::Ddp,
            model: IngredientCfg {
                kind: "model".into(),
                name: "from_pretrained".into(),
                config: serde_json::json!({"model_name": "distilbert-base-uncased"}),
            },
            optimizer: IngredientCfg {
                kind: "optimizer".into(),
                name: "adamw".into(),
                config: serde_json::json!({"lr": 3e-4}),
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
            batch_size: 32,
            seed: 42,
            eval_ratio: 0.1,
            device: "cuda".into(),
        }
    }

    #[test]
    fn compiles_to_5_node_plan() {
        let plan = TrainFromDataset.compile(args()).unwrap().into_compiled();
        assert_eq!(plan.n_nodes(), 5);
        assert_eq!(plan.n_edges(), 4);
    }

    #[test]
    fn rejects_both_dataset_specifiers() {
        let mut a = args();
        a.dataset_path = Some("/tmp/x.jsonl".into());
        let r = TrainFromDataset.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs { .. })));
    }

    #[test]
    fn rejects_neither_dataset_specifier() {
        let mut a = args();
        a.hf_name = None;
        let r = TrainFromDataset.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs { .. })));
    }

    /// The recipe used to hard-code `nproc_per_node: 1, nnodes: 1`, so the
    /// stage's multi-node support was unreachable from every shipped recipe:
    /// you could rent two machines and have no supported way to ask for them.
    #[test]
    fn the_distributed_shape_reaches_the_train_stage() {
        let mut a = args();
        a.nnodes = 2;
        a.nproc_per_node = 4;
        a.parallel_strategy = ParallelStrategy::Fsdp;
        let t = train_args(&a);
        assert_eq!(t.nnodes, 2);
        assert_eq!(t.nproc_per_node, 4);
        assert_eq!(t.parallel_strategy, ParallelStrategy::Fsdp);
        // And the plan still compiles with the distributed shape in it.
        assert_eq!(
            TrainFromDataset
                .compile(a)
                .unwrap()
                .into_compiled()
                .n_nodes(),
            5
        );
    }

    /// Defaults must keep a plain single-GPU run exactly as it was.
    #[test]
    fn the_default_shape_is_a_single_local_process() {
        let t = train_args(&args());
        assert_eq!(t.nnodes, 1);
        assert_eq!(t.nproc_per_node, 1);
        assert_eq!(t.parallel_strategy, ParallelStrategy::Ddp);
    }

    #[test]
    fn rejects_zero_epochs() {
        let mut a = args();
        a.epochs = 0;
        let r = TrainFromDataset.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs { .. })));
    }
}
