//! Stage 8 — `dpo_train`. Wraps trainer_dpo.py.
//!
//! Full DPO implementation pending in a follow-up. Rust-side
//! typed contract is complete; the Python side currently emits
//! Failed for non-self-check invocations. Recipe + executor wiring
//! works end-to-end as a smoke target.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use blut::artifacts::{HfCheckpoint, PreferenceJsonl};
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

pub struct DpoTrain;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub base_model: String,
    pub output_name: String,
    /// DPO temperature. Smaller = stronger preference signal.
    pub beta: f32,
    pub lr: f32,
    pub epochs: u32,
    pub batch_size: u32,
    pub grad_accum: u32,
    pub seq_len: u32,
    pub seed: u64,
}

#[async_trait]
impl Stage for DpoTrain {
    const NAME: &'static str = "dpo_train";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu, Resource::Network];
    type Input = PreferenceJsonl;
    type Output = HfCheckpoint;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: PreferenceJsonl,
        args: &Args,
    ) -> Result<HfCheckpoint, StageError> {
        // ADR 0037 Stage 5: the trainer_dpo.py stub was deleted rather
        // than completed ("delete, don't complete"). This legacy stage is
        // retained only so old recipes fail with guidance instead of a
        // resolver mystery; it never trained anything real.
        let _ = (ctx, input, args);
        Err(StageError::BadInput(
            "the trainer_dpo.py stub was removed (ADR 0037 Stage 5) — use hf_dpo_train (trl DPOTrainer) instead"
                .into(),
        ))
    }
}
