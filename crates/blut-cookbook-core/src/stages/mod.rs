//! Core cookbook stages — generic ML training primitives.

mod compat_impls;
pub mod evaluate_model;
pub mod load_dataset;
pub mod shared;
pub mod train_model;

pub use evaluate_model::{EvaluateLoadedDataset, EvaluateModel};
pub use load_dataset::LoadDataset;
pub use shared::IngredientCfg;
pub use train_model::TrainModel;
