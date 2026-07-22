// SPDX-License-Identifier: MIT
//! Deterministic, CPU-only acceptance recipe for ADR 0109.
//!
//! This is deliberately tiny but exercises production seams: typed recipe
//! compilation, dynamic resource envelopes, ordinary executor caching, live
//! metrics, broker refusal, and multi-objective study artifacts.

use std::path::Path;

use async_trait::async_trait;
use blut::backends::TrainingBackend;
use blut::framework::error::RecipeError;
use blut::framework::{
    Artifact, Compatible, ContentHash, Plan, Resource, Stage, StageContext, StageError, StageEvent,
};
use blut::recipes::recipe::{Recipe, RecipeCategory};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

pub struct StudyFixtureBackend;

impl TrainingBackend for StudyFixtureBackend {
    const ID: &'static str = "blut-study-fixture";
    const DESCRIPTION: &'static str = "CPU-only ADR 0109 acceptance backend";
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Args {
    /// Selects one of two deterministic non-dominated metric vectors.
    pub shape: u32,
    /// Declared RAM; the acceptance grid includes an oversized dynamic trial.
    pub ram_gib: u32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            shape: 0,
            ram_gib: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StudyFixtureResult {
    shape: u32,
    ram_gib: u32,
}

impl Artifact for StudyFixtureResult {
    const KIND: &'static str = "blut.study_fixture";
    const SCHEMA: u32 = 1;

    fn content_hash(&self) -> ContentHash {
        let mut bytes = self.shape.to_le_bytes().to_vec();
        bytes.extend(self.ram_gib.to_le_bytes());
        ContentHash::of_bytes(&bytes)
    }

    fn primary_path(&self) -> &Path {
        Path::new("")
    }
}

pub struct StudyFixtureStage;

#[async_trait]
impl Stage for StudyFixtureStage {
    const NAME: &'static str = "blut_study_fixture";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = ();
    type Output = StudyFixtureResult;
    type Args = Args;

    fn memory_gib_for(&self, args: &Self::Args) -> u32 {
        args.ram_gib
    }

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &Self::Args,
    ) -> Result<Self::Output, StageError> {
        let (quality, latency_ms, cpu_seconds) = match args.shape {
            0 => (0.92, 15.0, 1.0),
            1 => (0.84, 9.0, 1.5),
            _ => return Err(StageError::BadInput("shape must be 0 or 1".into())),
        };
        let _ = ctx.status_tx.send(StageEvent::StageStep {
            node_idx: ctx.node_idx,
            stage_name: Self::NAME.into(),
            update: json!({
                "epoch": 1,
                "quality": quality,
                "latency_ms": latency_ms,
                "cpu_seconds": cpu_seconds,
            }),
        });
        Ok(StudyFixtureResult {
            shape: args.shape,
            ram_gib: args.ram_gib,
        })
    }
}

impl Compatible<StudyFixtureBackend> for StudyFixtureStage {}

#[derive(Default)]
pub struct StudyFixtureRecipe;

impl Recipe for StudyFixtureRecipe {
    type Backend = StudyFixtureBackend;
    const NAME: &'static str = "blut_study_fixture";
    const DESCRIPTION: &'static str =
        "Deterministic CPU-only multi-objective study acceptance fixture.";
    const CATEGORY: RecipeCategory = RecipeCategory::Eval;
    const OUTPUT_KIND: &'static str = StudyFixtureResult::KIND;
    type Args = Args;

    fn compile(&self, args: Self::Args) -> Result<Plan<(), Self::Backend>, RecipeError> {
        if args.shape > 1 {
            return Err(RecipeError::InvalidArgs {
                field: Some("shape"),
                message: "shape must be 0 or 1".into(),
            });
        }
        if args.ram_gib == 0 {
            return Err(RecipeError::InvalidArgs {
                field: Some("ram_gib"),
                message: "ram_gib must be greater than zero".into(),
            });
        }
        let recipe_args = serde_json::to_value(&args)
            .map_err(|error| RecipeError::CompileFailed(error.to_string()))?;
        Ok(Plan::new(Self::NAME, recipe_args)
            .start(StudyFixtureStage, args)
            .finish())
    }
}

blut::register_recipe!(StudyFixtureRecipe);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_compiles_with_a_dynamic_declared_envelope() {
        let args = Args {
            shape: 1,
            ram_gib: 3,
        };
        StudyFixtureRecipe
            .compile(args.clone())
            .expect("compile fixture");
        let envelope = StudyFixtureStage.resource_envelope(&args);
        assert_eq!(envelope.ram_bytes, 3 * 1024 * 1024 * 1024);
    }
}
