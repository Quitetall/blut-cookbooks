//! Core cookbook stages — generic ML training primitives.

mod compat_impls;
pub mod shared;
pub mod load_dataset;
pub mod train_model;
pub mod evaluate_model;

pub use load_dataset::LoadDataset;
pub use train_model::TrainModel;
pub use evaluate_model::EvaluateModel;
pub use shared::IngredientCfg;
