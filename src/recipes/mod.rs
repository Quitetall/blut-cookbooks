//! Small built-in recipes owned by the standard cookbook.

mod study_fixture;

pub use study_fixture::{StudyFixtureRecipe, StudyFixtureStage};

use blut::recipes::recipe::RecipeDef;

/// Standard recipes available to every composed BLUT binary.
pub static STANDARD_RECIPES: &[&RecipeDef] = &[&study_fixture::DEF];
