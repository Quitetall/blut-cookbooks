//! blut-cookbook-standard — the standard ML cookbook for the BLUT engine.
//!
//! Domain-agnostic ML training ingredients (materialize, filter, split,
//! train, merge, convert, register, eval) plus concrete training-backend
//! identities (`HfTrainerBackend`, `LamuTrainerBackend`). This is the
//! "standard kitchen" that domain-specific cookbooks (blut-lamu,
//! blut-lamquant) build on.
//!
//! Implements the `Cookbook` trait so it registers with the BLUT engine
//! as `"standard"`. Its ingredients are available to declarative `.toml`
//! recipes via `stages_erased()`.
//!
//! Public surface:
//!
//!   - `stages::*` — concrete generic-LLM ingredient impls (materialize,
//!     filter, split, sft/dpo/distill train, merge_lora, convert_gguf,
//!     register_model, eval_*).
//!   - `backends::*` — the concrete training-backend identities
//!     (`HfTrainerBackend`, `LamuTrainerBackend`) + their subprocess
//!     runners / venv management / backend-specific ingredients. Re-exports
//!     the abstract `TrainingBackend` trait from `blut::backends`.
//!   - `backend` — the `TrainBackend` trait + `TrainArtifact` /
//!     `StatusFn` (the runtime contract a concrete trainer implements).
//!   - `convert` — HF checkpoint → GGUF conversion (llama.cpp tools).
//!   - `conversations` — lamu conversation-DB → JSONL dump.
//!   - `StandardCookbook` — the `Cookbook` impl + `registry()` entry point.
//!
//! The engine primitives the ingredients wire (framework, artifacts, config,
//! spec/protocol job schema, python_kill subprocess lifecycle, paths,
//! registry) live in the `blut` crate and are reached via `blut::*`.

// Production code is unsafe-free EXCEPT the subprocess pre_exec paths,
// which the engine's `python_kill` already gates; the moved backends
// merely call into it. Deny by default, allow per-call-site; tests get
// a blanket allow.
#![cfg_attr(not(test), deny(unsafe_code))]

pub mod backend;
pub mod backends;
pub mod conversations;
pub mod convert;
pub mod stages;

/// Process-wide lock for tests that mutate environment variables.
/// Mirrors the engine's `TEST_ENV_LOCK`: several moved test modules
/// touch `LAMU_TRAIN_*` env vars; without a shared mutex parallel test
/// execution races on the global env.
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Convenience re-exports mirroring the engine's old top-level surface,
// so callers that named these directly keep a single import path.
pub use backend::{StatusFn, TrainArtifact, TrainBackend};
pub use backends::{HfTrainerBackend, LamuTrainerBackend, TrainingBackend};
pub use backends::lamu::python_backend::PythonTrainBackend;

// The job/status persistence schema (TrainSpec / StatusUpdate) + the
// TrainError type stay in the engine (the framework's `jobs.rs` reads
// them); re-export for ergonomic single-path access from cookbooks.
pub use blut::error::TrainError;
pub use blut::protocol::StatusUpdate;
pub use blut::spec::{DatasetSource, Method, Optim, TrainSpec};

// ── Cookbook registration ────────────────────────────────────────────

use blut::framework::{Cookbook, Registry};
use blut::recipes::recipe::RecipeDef;

/// The standard ML cookbook. Registers all generic-LLM ingredients
/// with the BLUT engine so declarative `.toml` recipes can reference
/// them by name.
pub struct StandardCookbook;

/// Erased ingredient constructors for declarative `.toml` recipe support.
/// Each entry maps a name → `Arc<dyn StageDyn>` constructor.
static STANDARD_STAGES_ERASED: &[(&str, blut::framework::stage::ErasedStageCtor)] = &[
    ("materialize_conversations", || std::sync::Arc::new(stages::MaterializeConversations)),
    ("materialize_dataset_path", || std::sync::Arc::new(stages::MaterializeDatasetPath)),
    ("materialize_for_eval", || std::sync::Arc::new(stages::MaterializeForEval)),
    ("filter_dataset", || std::sync::Arc::new(stages::FilterDataset)),
    ("split_train_eval", || std::sync::Arc::new(stages::SplitTrainEval)),
    ("register_dataset", || std::sync::Arc::new(stages::RegisterDataset)),
    ("take_train", || std::sync::Arc::new(stages::TakeTrain)),
    ("sft_train", || std::sync::Arc::new(stages::SftTrain)),
    ("dpo_train", || std::sync::Arc::new(stages::DpoTrain)),
    ("distill_train", || std::sync::Arc::new(stages::DistillTrain)),
    ("merge_lora", || std::sync::Arc::new(stages::MergeLora)),
    ("convert_gguf", || std::sync::Arc::new(stages::ConvertGguf)),
    ("register_model", || std::sync::Arc::new(stages::RegisterModel)),
    ("eval_loss", || std::sync::Arc::new(stages::EvalLoss)),
    ("eval_lm_harness", || std::sync::Arc::new(stages::EvalLmHarness)),
    ("eval_judge", || std::sync::Arc::new(stages::EvalJudge)),
    ("merge_reports", || std::sync::Arc::new(stages::MergeReports)),
    // HF-backend-specific ingredients
    ("hf_sft_train", || std::sync::Arc::new(backends::hf_trainer::stages::HfSftTrain)),
    ("hf_dpo_train", || std::sync::Arc::new(backends::hf_trainer::stages::HfDpoTrain)),
];

impl Cookbook for StandardCookbook {
    fn name(&self) -> &'static str {
        "standard"
    }
    fn recipes(&self) -> &'static [&'static RecipeDef] {
        // No built-in recipes — domain cookbooks (blut-lamu, blut-lamquant)
        // define recipes that compose these ingredients.
        &[]
    }
    fn stages_erased(&self) -> &'static [(&'static str, blut::framework::stage::ErasedStageCtor)] {
        STANDARD_STAGES_ERASED
    }
}

/// Build a `Registry` containing just the standard cookbook.
/// Downstream binaries register their domain cookbook on top.
pub fn registry() -> Registry {
    let mut r = Registry::new();
    r.register(Box::new(StandardCookbook));
    r
}
