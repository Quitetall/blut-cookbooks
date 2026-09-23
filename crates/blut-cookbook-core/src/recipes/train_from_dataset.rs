//! Recipe — `train_from_dataset`.
//!
//! Generic training pipeline:
//!
//!   load_dataset → split_train_eval ─┬→ train_model_on_split ─┬→ evaluate_held_out
//!                                    └→ take_eval ────────────┘
//!
//! Works with any dataset (HuggingFace or local CSV/JSONL) and any model
//! (via the ingredient system). Zero domain code required.

use serde::{Deserialize, Serialize};

use blut::framework::error::RecipeError;
use blut::framework::plan::Plan;
use blut::recipes::recipe::Recipe;
use blut_backends::stages::split_train_eval::{Args as SplitArgs, SplitTrainEval};
use blut_backends::stages::take_eval::TakeEval;

use crate::recipes::shared::{DatasetSelect, TrainShape};
use crate::stages::IngredientCfg;
use crate::stages::evaluate_model::{EvaluateHeldOut, HeldOutArgs};
use crate::stages::load_dataset::{Args as LoadArgs, LoadDataset};
use crate::stages::train_model::{Args as TrainArgs, TrainModelOnSplit};

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
    /// `subset` and `max_samples`.
    #[serde(flatten)]
    pub dataset: DatasetSelect,
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
    /// `nproc_per_node`, `nnodes`, `parallel_strategy`, `text_field`,
    /// `max_seq_len`.
    #[serde(flatten)]
    pub shape: TrainShape,
}

fn default_split() -> String {
    "train".into()
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
            .start(LoadDataset, load_args(&args))
            .then(
                SplitTrainEval,
                SplitArgs {
                    eval_ratio: args.eval_ratio,
                    seed: args.seed,
                },
            )
            // Train on one half, hold the other out, and evaluate the trained
            // checkpoint on the half it never saw. This recipe used to chain
            // `take_train -> train_model -> evaluate_model`, which dropped the
            // held-out half and had `evaluate_model` reload the dataset by
            // name — the training split — so the report scored the model on
            // its own training data.
            .fork(
                TrainModelOnSplit,
                train_args(&args),
                TakeEval,
                blut_backends::stages::take_eval::Args::default(),
            )
            .merge(EvaluateHeldOut, held_out_args(&args))
            .finish();
        Ok(plan)
    }
}

/// Map the recipe's arguments onto the `load_dataset` stage.
///
/// Named for the same reason as [`train_args`]: a compiled plan does not let a
/// test outside the engine read a stage's arguments back.
fn load_args(args: &Args) -> LoadArgs {
    LoadArgs {
        path: args.dataset_path.clone(),
        hf_name: args.hf_name.clone(),
        split: args.split.clone(),
        subset: args.dataset.subset.clone(),
        max_samples: args.dataset.max_samples,
    }
}

/// Map the recipe's arguments onto the held-out evaluation. It scores with the
/// same loss ingredient and the same text packing that training used, so the
/// held-out number is comparable to the training loss.
fn held_out_args(args: &Args) -> HeldOutArgs {
    HeldOutArgs {
        eval: IngredientCfg {
            kind: "eval".into(),
            name: "loss_eval".into(),
            config: serde_json::Value::Object(Default::default()),
        },
        loss: Some(args.loss.clone()),
        batch_size: args.batch_size.saturating_mul(2),
        device: args.device.clone(),
        text_field: args.shape.text_field.clone(),
        max_seq_len: args.shape.max_seq_len,
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
        nproc_per_node: args.shape.nproc_per_node,
        nnodes: args.shape.nnodes,
        parallel_strategy: args.shape.parallel_strategy,
        text_field: args.shape.text_field.clone(),
        max_seq_len: args.shape.max_seq_len,
    }
}

blut::register_recipe!(TrainFromDataset);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::train_model::ParallelStrategy;

    fn args() -> Args {
        Args {
            hf_name: Some("imdb".into()),
            dataset_path: None,
            split: "train".into(),
            dataset: DatasetSelect::default(),
            shape: TrainShape::default(),
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
        // load -> split -> {train_model_on_split, take_eval} -> evaluate_held_out
        let plan = TrainFromDataset.compile(args()).unwrap().into_compiled();
        assert_eq!(plan.n_nodes(), 5);
        assert_eq!(
            plan.n_edges(),
            5,
            "the split fans out to two edges that rejoin"
        );
    }

    /// Evaluation must use the loss training used, not a default.
    #[test]
    fn held_out_evaluation_scores_with_the_training_loss() {
        let a = args();
        let h = held_out_args(&a);
        assert_eq!(
            h.loss.as_ref().map(|l| l.name.as_str()),
            Some(a.loss.name.as_str())
        );
        assert_eq!(h.eval.name, "loss_eval");
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
        a.shape = TrainShape {
            nproc_per_node: 4,
            nnodes: 2,
            parallel_strategy: ParallelStrategy::Fsdp,
            text_field: Some("body".into()),
            max_seq_len: Some(1024),
        };
        let t = train_args(&a);
        assert_eq!(t.text_field.as_deref(), Some("body"));
        assert_eq!(t.max_seq_len, Some(1024));
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

    /// `subset` and `max_samples` were hard-coded to `None` alongside the
    /// distributed fields. A dataset that publishes several configs — wikitext
    /// among them — has no loadable default, so those were simply unopenable
    /// from this recipe, and every other corpus loaded in full.
    #[test]
    fn the_dataset_selectors_reach_the_load_stage() {
        let mut a = args();
        a.dataset = DatasetSelect {
            subset: Some("wikitext-2-raw-v1".into()),
            max_samples: Some(256),
        };
        let l = load_args(&a);
        assert_eq!(l.subset.as_deref(), Some("wikitext-2-raw-v1"));
        assert_eq!(l.max_samples, Some(256));
    }

    /// The recipe's arguments are serialized into the plan's identity. A run
    /// that sets none of the fields added for distributed or causal-LM
    /// training must serialize exactly as it did before they existed, or every
    /// such run misses the cache on arguments that change nothing.
    #[test]
    fn defaulted_new_fields_leave_the_plan_identity_unchanged() {
        let json = serde_json::to_value(args()).unwrap();
        for key in [
            "nproc_per_node",
            "nnodes",
            "parallel_strategy",
            "text_field",
            "max_seq_len",
            "subset",
            "max_samples",
        ] {
            assert!(json.get(key).is_none(), "{key} serialized at its default");
        }
    }

    /// The shared structs are flattened: users still write every field at the
    /// top level of the recipe's arguments.
    #[test]
    fn distributed_and_dataset_fields_are_read_from_the_top_level() {
        let mut json = serde_json::to_value(args()).unwrap();
        json["nnodes"] = 2.into();
        json["subset"] = "wikitext-2-raw-v1".into();
        let a: Args = serde_json::from_value(json).unwrap();
        assert_eq!(a.shape.nnodes, 2);
        assert_eq!(a.dataset.subset.as_deref(), Some("wikitext-2-raw-v1"));
    }

    #[test]
    fn rejects_zero_epochs() {
        let mut a = args();
        a.epochs = 0;
        let r = TrainFromDataset.compile(a);
        assert!(matches!(r, Err(RecipeError::InvalidArgs { .. })));
    }
}
