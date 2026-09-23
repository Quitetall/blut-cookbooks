//! Core cookbook recipes — generic ML training pipelines.

pub mod eval_only;
pub mod finetune_pretrained;
pub mod shared;
pub mod train_from_dataset;

use blut::recipes::recipe::RecipeDef;

pub static CORE_RECIPES: &[&RecipeDef] = &[
    &train_from_dataset::DEF,
    &finetune_pretrained::DEF,
    &eval_only::DEF,
];
