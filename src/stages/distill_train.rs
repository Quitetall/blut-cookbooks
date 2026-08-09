//! Stage 9 — `distill_train`. Wraps trainer_distill.py.
//!
//! Two-input stage: takes (HfCheckpoint teacher, DatasetJsonl) and
//! produces an HfCheckpoint student. Teacher's outputs are sampled
//! into the dataset path before training (the Python side handles
//! that). Currently uses trainer_distill.py stub.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use blut::artifacts::{DatasetJsonl, HfCheckpoint};
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

pub struct DistillTrain;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub student_base: String,
    pub output_name: String,
    pub kl_weight: f32,
    pub lr: f32,
    pub epochs: u32,
    pub batch_size: u32,
    pub grad_accum: u32,
    pub seq_len: u32,
    pub seed: u64,
}

#[async_trait]
impl Stage for DistillTrain {
    const NAME: &'static str = "distill_train";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu, Resource::Network];
    type Input = (HfCheckpoint, DatasetJsonl);
    type Output = HfCheckpoint;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: (HfCheckpoint, DatasetJsonl),
        args: &Args,
    ) -> Result<HfCheckpoint, StageError> {
        // ADR 0037 Stage 5: the trainer_distill.py stub was deleted rather
        // than completed ("delete, don't complete"). This legacy stage is
        // retained only so old recipes fail with guidance instead of a
        // resolver mystery; it never trained anything real.
        let _ = (ctx, input, args);
        Err(StageError::BadInput(
            "the trainer_distill.py stub was removed (ADR 0037 Stage 5) — use the blut-lamu distill_bitnet stage (ternary) or hf_sft_train (HF-base) instead"
                .into(),
        ))
}
}
