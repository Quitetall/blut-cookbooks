//! HuggingFace Trainer backend.
//!
//! Subprocesses a Python wrapper that drives
//! `transformers.Trainer` (SFT, distillation) and
//! `trl.DPOTrainer` (preference fine-tuning). Manages its own
//! venv at `~/.local/share/blut/hf-venv/` — auto-provisioned on
//! first use so users don't fight Python environment drift.
//!
//! Land status (BB-1): stub. Real runner + stages + recipes land
//! in BB-4 + BB-5. This module exists now so the typed `Plan`
//! machinery can name `HfTrainerBackend` at compile time
//! everywhere it needs to without forward-declaring.

pub mod image_text;
pub mod runner;
pub mod stages;
pub mod venv;

pub use image_text::{ARCH_CAUSAL_LM, ARCH_IMAGE_TEXT, ImageTextOpts};
pub use runner::{DpoConfig, HfRunArtifact, HfTrainerJob, HfTrainerRunner, PeftConfig, StatusLine};
pub use stages::{HfDpoTrain, HfSftTrain};
pub use venv::{
    CAUSAL_LM_PROFILE, IMAGE_TEXT_PROFILE, VenvError, VenvProfile, ensure_venv,
    ensure_venv_profile, venv_root,
};
