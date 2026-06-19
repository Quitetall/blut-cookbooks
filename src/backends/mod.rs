//! Concrete training-backend identities for the generic-LLM cookbooks.
//!
//! The ABSTRACT [`TrainingBackend`] marker trait is the engine's public
//! 1.0 backend-identity seam and lives in `blut::backends`. This module
//! re-exports it (so downstream `blut_backends::backends::TrainingBackend`
//! keeps working) and supplies the two concrete identity structs that the
//! generic-LLM stages + cookbooks wire: [`HfTrainerBackend`] and
//! [`LamuTrainerBackend`]. Each brings its own subprocess runner, its own
//! typed stages (under `backends/<id>/stages/`), and its own identity.
//!
//! Identity matters because the typed `Plan<Out, B>` is parameterized
//! over a backend `B`. A recipe declares `type Backend = HfTrainerBackend`;
//! the compiler then refuses to wire a `LamuTrainerBackend`-tagged stage
//! into its plan.

pub mod hf_trainer;
pub mod lamu;

/// Re-export the engine's abstract backend-identity trait. Concrete
/// backends below implement it; cookbook stages bound on it.
pub use blut::backends::TrainingBackend;

/// HuggingFace Trainer backend. Subprocesses a Python wrapper
/// that drives `transformers.Trainer` (and `trl.DPOTrainer` for
/// preference-pair fine-tuning). Manages its own venv at
/// `~/.local/share/blut/hf-venv/` on first use.
///
/// This is BLUT's blessed default for LLM-style training. New
/// `hf_*` recipes ship under it.
pub struct HfTrainerBackend;
impl TrainingBackend for HfTrainerBackend {
    const ID: &'static str = "hf_trainer";
    const DESCRIPTION: &'static str = "HuggingFace Trainer (transformers + trl). Auto-managed venv. Default for SFT/DPO/distillation.";
}

/// LAMU's `trainer.py` wire. The original BLUT backend — emits
/// `StatusUpdate` JSON lines, expects a `TrainSpec` blob on argv.
/// Kept for back-compat with the lamu cookbook's SFT recipes; new
/// recipes should prefer `HfTrainerBackend`.
pub struct LamuTrainerBackend;
impl TrainingBackend for LamuTrainerBackend {
    const ID: &'static str = "lamu";
    const DESCRIPTION: &'static str =
        "LAMU trainer.py (TrainSpec JSON / StatusUpdate wire). Original backend.";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_ids_are_unique() {
        let ids = [HfTrainerBackend::ID, LamuTrainerBackend::ID];
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len(), "backend IDs must be unique");
    }

    #[test]
    fn ids_are_stable_strings() {
        // Lock the wire identifiers — bumping these invalidates
        // every cache + audit reference. Catch unintentional
        // renames here.
        assert_eq!(HfTrainerBackend::ID, "hf_trainer");
        assert_eq!(LamuTrainerBackend::ID, "lamu");
    }
}
