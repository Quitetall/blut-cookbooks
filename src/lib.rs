//! blut-backends — the generic-LLM cookbook layer for the BLUT engine.
//!
//! Extracted from the `blut` engine crate at the engine-only v1.0 carve
//! (the keystone of the public release): the engine became a PURE
//! framework (Stage / Plan / Recipe / Registry / Cookbook + CLI/TUI
//! orchestration + the job/status persistence schema), and everything
//! LLM-domain-specific moved here.
//!
//! Public surface:
//!
//!   - `stages::*` — concrete generic-LLM stage impls (materialize,
//!     filter, split, sft/dpo/distill train, merge_lora, convert_gguf,
//!     register_model, eval_*).
//!   - `backends::*` — the concrete training-backend identities
//!     (`HfTrainerBackend`, `LamuTrainerBackend`) + their subprocess
//!     runners / venv management / backend-specific stages. Re-exports
//!     the abstract `TrainingBackend` trait from `blut::backends`.
//!   - `backend` — the `TrainBackend` trait + `TrainArtifact` /
//!     `StatusFn` (the runtime contract a concrete trainer implements).
//!   - `convert` — HF checkpoint → GGUF conversion (llama.cpp tools).
//!   - `conversations` — lamu conversation-DB → JSONL dump.
//!
//! The engine primitives the stages wire (framework, artifacts, config,
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
