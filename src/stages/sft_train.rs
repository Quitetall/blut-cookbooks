//! Stage 7 — `sft_train`.
//!
//! Wraps `python_backend::PythonTrainBackend` (the existing SFT
//! trainer.py runner). Converts a typed `DatasetJsonl` input + Args
//! into a legacy `TrainSpec`, runs through the python subprocess,
//! returns a typed `HfCheckpoint`.
//!
//! Resource: Gpu + Network. The cross-process scheduler_lock from
//! lamu-core is acquired by recipes (or the executor's commit-6
//! Resource::Gpu enforcement); this stage only concerns itself
//! with running the trainer once it has the GPU.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::backend::TrainBackend;
use crate::backends::lamu::python_backend::PythonTrainBackend;
use blut::artifacts::{DatasetJsonl, HfCheckpoint};
use blut::framework::artifact::ContentHash;
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};
use blut::paths;
use blut::spec::{DatasetSource, Method, Optim, TrainSpec};

pub struct SftTrain;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub base_model: String,
    pub output_name: String,
    /// `qlora`, `lora`, or `full`.
    pub method: String,
    pub rank: u32,
    pub alpha: u32,
    pub optimizer: String,
    pub lr: f32,
    pub epochs: u32,
    pub batch_size: u32,
    pub grad_accum: u32,
    pub seq_len: u32,
    pub seed: u64,
}

#[async_trait]
impl Stage for SftTrain {
    const NAME: &'static str = "sft_train";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu, Resource::Network];
    type Input = DatasetJsonl;
    type Output = HfCheckpoint;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: DatasetJsonl,
        args: &Args,
    ) -> Result<HfCheckpoint, StageError> {
        // R21 pre: arg + input sanity.
        debug_assert!(input.n_examples > 0, "dataset must have examples");
        if args.epochs == 0 || args.batch_size == 0 || args.grad_accum == 0 {
            return Err(StageError::BadInput(
                "epochs/batch_size/grad_accum must be > 0".into(),
            ));
        }
        if !(args.lr > 0.0 && args.lr.is_finite()) {
            return Err(StageError::BadInput(format!(
                "lr must be positive finite; got {}",
                args.lr
            )));
        }
        let output_dir = ctx.stage_dir.join("checkpoint");
        std::fs::create_dir_all(&output_dir).map_err(|source| StageError::Io {
            path: output_dir.clone(),
            source,
        })?;

        let method = match args.method.as_str() {
            "qlora" => Method::QLora {
                rank: args.rank,
                alpha: args.alpha,
            },
            "lora" => Method::Lora {
                rank: args.rank,
                alpha: args.alpha,
            },
            "full" => Method::Full,
            other => {
                return Err(StageError::BadInput(format!(
                    "unknown method '{other}', expected qlora|lora|full"
                )));
            }
        };
        let optimizer = match args.optimizer.as_str() {
            "adam_w" | "adamw" => Optim::AdamW,
            "adam_w8bit" | "adamw8bit" => Optim::AdamW8bit,
            "apollo_mini" => Optim::ApolloMini,
            "apollo_rank4" | "apollo" => Optim::ApolloRank4,
            other => {
                return Err(StageError::BadInput(format!("unknown optimizer '{other}'")));
            }
        };

        let spec = TrainSpec {
            base_model: args.base_model.clone(),
            output_name: args.output_name.clone(),
            output_dir: output_dir.clone(),
            method,
            dataset: DatasetSource::JsonlPath {
                path: input.path.clone(),
            },
            optimizer,
            lr: args.lr,
            epochs: args.epochs,
            batch_size: args.batch_size,
            grad_accum: args.grad_accum,
            seq_len: args.seq_len,
            seed: args.seed,
            quant: "Q4_K_M".into(), // unused by the trainer; convert_gguf owns quant
            skip_convert: true,
            nproc_per_node: 1,
            nnodes: 1,
            dpo_beta: None,
        };
        spec.validate()
            .map_err(|e| StageError::BadInput(format!("{e}")))?;

        let python =
            paths::resolve_python().map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;
        let trainer_script =
            paths::resolve_trainer_script().map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;
        let mut backend = trainer_backend(ctx, python, trainer_script);

        // Forward StageStep events from trainer.py's per-step
        // status updates so the unified status.jsonl carries them.
        let stage_name_owned = SftTrain::NAME.to_string();
        let status_tx = ctx.status_tx.clone();
        // This stage's topo position — stamped into every StageStep so a consumer
        // attributes steps to the right node. The HPO scheduler maps
        // StageStep.node_idx → trial; a hardcoded 0 would collapse all concurrent
        // trial nodes onto node 0 and mis-target early-stop decisions.
        let node_idx = ctx.node_idx;
        let on_status: crate::backend::StatusFn = Box::new(move |update| {
            if let Ok(value) = serde_json::to_value(&update) {
                let _ = status_tx.send(blut::framework::status::StageEvent::StageStep {
                    node_idx,
                    stage_name: stage_name_owned.clone(),
                    update: value,
                });
            }
        });

        let artifact = backend
            .run(spec.clone(), on_status)
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;

        let hash = ContentHash::hash_dir(&output_dir).map_err(|source| StageError::Io {
            path: output_dir.clone(),
            source,
        })?;

        Ok(HfCheckpoint {
            path: output_dir,
            base_model: args.base_model.clone(),
            method_tag: args.method.clone(),
            content_hash: hash,
            final_loss: artifact.final_loss,
        })
    }
}

/// The trainer for this stage, placed where `--launcher` says. Under Slurm its
/// job carries `ctx.slurm_job_name()`, the name the engine records with the
/// BLUT job, so `blut cancel` can stop, and `plan resume` can see, a trainer
/// left running by a BLUT process that was killed.
fn trainer_backend(
    ctx: &StageContext,
    python: std::path::PathBuf,
    trainer_script: std::path::PathBuf,
) -> PythonTrainBackend {
    PythonTrainBackend::new(python, trainer_script)
        .with_launch_target(ctx.launch_target)
        .with_job_name(ctx.slurm_job_name())
}

#[cfg(test)]
mod tests {
    use super::*;
    use blut::config::launcher::LaunchTarget;

    /// Under `--launcher slurm` the trainer's step is named after the job and
    /// node, as `train_model`'s and the HF stages' are.
    #[test]
    fn a_slurm_step_is_named_after_the_job_and_node() {
        let td = tempfile::tempdir().unwrap();
        let job_dir = td.path().join("20261030-120000-1");
        let mut ctx = StageContext::for_test(job_dir.clone(), job_dir.join("stage"));
        ctx.launch_target = LaunchTarget::Slurm;
        ctx.node_idx = 2;
        let spec = TrainSpec {
            base_model: "org/m".into(),
            output_name: "named".into(),
            output_dir: td.path().join("out"),
            method: Method::Full,
            dataset: DatasetSource::JsonlPath {
                path: td.path().join("x.jsonl"),
            },
            optimizer: Optim::AdamW,
            lr: 2e-4,
            epochs: 1,
            batch_size: 1,
            grad_accum: 1,
            seq_len: 512,
            seed: 42,
            quant: "Q4_K_M".into(),
            skip_convert: true,
            nproc_per_node: 1,
            nnodes: 1,
            dpo_beta: None,
        };
        let cmd = trainer_backend(&ctx, "py".into(), "trainer.py".into())
            .command(&spec, "{}")
            .unwrap();
        assert_eq!(cmd[0], "srun");
        assert!(
            cmd.contains(&"--job-name=blut-20261030-120000-1-2".to_string()),
            "{cmd:?}"
        );
    }
}
