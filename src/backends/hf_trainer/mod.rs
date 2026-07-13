//! HuggingFace Trainer backend.
//!
//! Subprocesses a Python wrapper that drives
//! `transformers.Trainer` (SFT, distillation) and
//! `trl.DPOTrainer` (preference fine-tuning). Manages its own
//! venv at `~/.local/share/blut/hf-venv/` — auto-provisioned on
//! first use so users don't fight Python environment drift.
//!
//! The runner and SFT/DPO stages execute real Transformers/TRL training. Its
//! Python wrapper is embedded in the Rust crate and materialized privately at
//! launch, so installed packages do not depend on a retained source checkout.

pub mod runner;
pub mod stages;
pub mod venv;

pub use runner::{DpoConfig, HfRunArtifact, HfTrainerJob, HfTrainerRunner, PeftConfig, StatusLine};
pub use stages::{HfDpoTrain, HfSftTrain};
pub use venv::{VenvError, ensure_venv, venv_root};
