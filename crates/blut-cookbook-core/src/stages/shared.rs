//! Shared types for core cookbook stages.

use serde::{Deserialize, Serialize};

/// Ingredient config for one training/eval component.
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct IngredientCfg {
    pub kind: String,
    pub name: String,
    #[serde(default)]
    pub config: serde_json::Value,
}
