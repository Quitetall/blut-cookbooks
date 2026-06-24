//! **blut-cookbook-core** — the core BLUT cookbook (generic ML primitives).
//!
//! Provides generic ingredients, stages, and recipes that work for any ML
//! training task out of the box. Domain cookbooks (lamquant, lamu, eagle)
//! extend it by overriding what they need and inheriting everything else.
//!
//! The ingredient system (`blut_core` Python package) lives in
//! `python/blut_core/` and provides 24 generic ingredient specs across
//! 12 kinds (data, model, optimizer, scheduler, loss, step, ema,
//! checkpoint, eval, sampler, logging, forward).

pub mod recipes;
pub mod stages;

pub use blut::framework::{Cookbook, Registry};
pub use recipes::CORE_RECIPES;

use std::sync::Arc;

use blut::framework::stage::{ErasedStageCtor, Stage, StageDyn};
use blut::recipes::recipe::RecipeDef;
use crate::stages::*;

/// Executable erased stage constructors for the core cookbook stages.
/// Each ctor yields a fresh `Arc<dyn StageDyn>` keyed by `Stage::NAME`.
static CORE_STAGES_ERASED: &[(&str, ErasedStageCtor)] = &[
    (<LoadDataset as Stage>::NAME, || Arc::new(LoadDataset) as Arc<dyn StageDyn>),
    (<TrainModel as Stage>::NAME, || Arc::new(TrainModel) as Arc<dyn StageDyn>),
    (<EvaluateModel as Stage>::NAME, || Arc::new(EvaluateModel) as Arc<dyn StageDyn>),
];

/// The core BLUT cookbook — generic ML training primitives.
pub struct CoreCookbook;

impl Cookbook for CoreCookbook {
    fn name(&self) -> &'static str {
        "core"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        CORE_RECIPES
    }
    fn default_args(&self, _recipe: &str) -> Option<String> {
        None
    }
    fn stages_erased(&self) -> &'static [(&'static str, ErasedStageCtor)] {
        CORE_STAGES_ERASED
    }
}

/// A [`Registry`] containing the core cookbook + the standard cookbook
/// (blut-backends). This is the minimal set for generic ML training.
pub fn registry() -> Registry {
    let mut r = Registry::new();
    r.register(Box::new(CoreCookbook));
    r.register(Box::new(blut_backends::StandardCookbook));
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_cookbook_registers_and_lists_recipes() {
        let r = registry();
        assert!(r.find("train_from_dataset").is_some());
        assert!(r.find("finetune_pretrained").is_some());
        assert!(r.find("eval_only").is_some());
        assert_eq!(
            r.all().count(),
            3 + blut_backends::StandardCookbook.recipes().len(),
            "3 core recipes + standard cookbook recipes"
        );
    }

    #[test]
    fn stages_erased_is_complete_unique_and_self_consistent() {
        let names: Vec<&str> = CORE_STAGES_ERASED.iter().map(|(n, _)| *n).collect();
        assert_eq!(names.len(), 3, "all core stages registered");
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "stage names must be unique");
        for (name, ctor) in CORE_STAGES_ERASED {
            assert_eq!(ctor().name(), *name, "ctor for '{name}' yields a mismatched stage");
        }
    }

    #[test]
    fn cookbook_name_is_core() {
        assert_eq!(CoreCookbook.name(), "core");
    }
}
