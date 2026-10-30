//! HF Trainer backend stages.
//!
//! Each stage here drives `HfTrainerRunner` (the subprocess +
//! venv runner) with a task-specific job spec. They mirror the
//! lamu trainer.py stages (`sft_train`, `dpo_train`,
//! `distill_train`) in their typed I/O so recipes can swap
//! backends with minimal change beyond the recipe-level
//! `type Backend` tag.

pub mod hf_dpo_train;
pub mod hf_sft_train;

pub use hf_dpo_train::HfDpoTrain;
pub use hf_sft_train::HfSftTrain;

use blut::framework::stage::StageContext;
use blut::framework::status::StageEvent;

use super::StatusLine;

/// The status callback an HF stage hands its runner: each wrapper line goes to
/// the executor as a `StageStep` whose `update` is the engine's trainer update
/// ([`StatusLine::status_update`]). That is the format `blut log` and
/// `blut jobs` read from a stage's forwarded progress; the `hf_step` and
/// `hf_saved` payloads these stages used to send were in no format they
/// parse, so an HF stage's progress never reached either.
pub(crate) fn forward_status(
    ctx: &StageContext,
    stage_name: &'static str,
) -> Box<dyn Fn(StatusLine) + Send + Sync> {
    let status_tx = ctx.status_tx.clone();
    // This stage's topo position, so a consumer (`blut log`, the HPO
    // scheduler) attributes each update to the right node.
    let node_idx = ctx.node_idx;
    Box::new(move |line: StatusLine| {
        let Some(update) = line.status_update() else {
            return;
        };
        match serde_json::to_value(&update) {
            Ok(update) => {
                let _ = status_tx.send(StageEvent::StageStep {
                    node_idx,
                    stage_name: stage_name.to_string(),
                    update,
                });
            }
            Err(e) => tracing::warn!("{stage_name}: cannot serialize a trainer update: {e}"),
        }
    })
}

/// Each HF stage, run against a stub trainer that prints what the wrapper
/// prints, with its events persisted by the engine's own status writer and
/// read back by the engine's own readers: `read_status` (what `blut log`
/// renders) and `list_jobs` (what `blut jobs` prints). No torch, no GPU: the
/// "venv" is a directory whose `bin/python` is `/bin/sh`.
#[cfg(all(test, unix))]
mod tests {
    use std::future::Future;
    use std::path::{Path, PathBuf};

    use blut::artifacts::{DatasetJsonl, HfCheckpoint, PreferenceJsonl};
    use blut::framework::artifact::ContentHash;
    use blut::framework::error::StageError;
    use blut::framework::stage::{Stage, StageContext};
    use blut::framework::status::{StatusHub, spawn_status_writer_checked};
    use blut::jobs::JobSummary;
    use blut::protocol::StatusUpdate;

    use super::{HfDpoTrain, HfSftTrain};

    const JOB: &str = "20261030-120000-000000001";

    struct EnvGuard(Vec<(&'static str, Option<std::ffi::OsString>)>);

    impl EnvGuard {
        fn set(vars: &[(&'static str, &Path)]) -> Self {
            let saved = vars
                .iter()
                .map(|(k, _)| (*k, std::env::var_os(k)))
                .collect();
            for (k, v) in vars {
                // SAFETY: serialized by crate::TEST_ENV_LOCK, held by the caller.
                unsafe { std::env::set_var(k, v) };
            }
            Self(saved)
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in self.0.drain(..) {
                // SAFETY: as in `set`.
                unsafe {
                    match v {
                        Some(v) => std::env::set_var(k, v),
                        None => std::env::remove_var(k),
                    }
                }
            }
        }
    }

    fn executable(path: &Path, body: &str) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::write(path, body).unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    /// What the engine reads back from one stage's run.
    struct Relayed {
        /// `read_status`: every trainer update, as `blut log` sees them.
        updates: Vec<StatusUpdate>,
        /// The job's row in `blut jobs`.
        summary: JobSummary,
        /// `blut log`'s rendering.
        log: String,
        /// The checkpoint directory the stub reported.
        ckpt: PathBuf,
    }

    /// Run one stage against the stub and return what the engine reads back.
    #[allow(clippy::await_holding_lock)]
    async fn relayed<F, Fut>(run: F) -> Relayed
    where
        F: FnOnce(StageContext, &Path) -> Fut,
        Fut: Future<Output = Result<HfCheckpoint, StageError>>,
    {
        let _lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let td = tempfile::tempdir().unwrap();
        let root = td.path();

        // A provisioned "venv" whose interpreter runs the stub as a shell script.
        let venv = root.join("venv");
        std::fs::create_dir_all(venv.join("bin")).unwrap();
        executable(&venv.join("bin/python"), "#!/bin/sh\nexec /bin/sh \"$@\"\n");
        std::fs::write(
            venv.join(".blut_hf_venv_version"),
            crate::backends::hf_trainer::venv::VENV_VERSION,
        )
        .unwrap();

        // The wrapper's stdout for a short run: an evaluation-style log with
        // no learning rate, two measured steps, a save, and the end.
        let ckpt = root.join("ckpt");
        std::fs::create_dir_all(&ckpt).unwrap();
        std::fs::write(ckpt.join("adapter_model.safetensors"), b"weights").unwrap();
        let stub = root.join("hf_trainer_runner.sh");
        let c = ckpt.display();
        executable(
            &stub,
            &format!(
                "echo '{{\"kind\":\"step\",\"step\":1,\"total\":4,\"loss\":null,\"lr\":null,\"vram_mb\":0}}'\n\
                 echo '{{\"kind\":\"step\",\"step\":2,\"total\":4,\"loss\":2.5,\"lr\":0.0002,\"vram_mb\":812}}'\n\
                 echo '{{\"kind\":\"step\",\"step\":4,\"total\":4,\"loss\":1.5,\"lr\":0.0001,\"vram_mb\":812}}'\n\
                 echo '{{\"kind\":\"saved\",\"path\":\"{c}/checkpoint-4\"}}'\n\
                 echo '{{\"kind\":\"done\",\"checkpoint_dir\":\"{c}\",\"final_loss\":1.75}}'\n"
            ),
        );

        let jobs = root.join("jobs");
        let _env = EnvGuard::set(&[
            ("BLUT_HF_VENV", &venv),
            ("BLUT_HF_TRAINER_RUNNER", &stub),
            ("LAMU_TRAIN_JOBS_DIR", &jobs),
        ]);
        let job_dir = blut::paths::job_dir(JOB).unwrap();

        // The engine's status writer, fed the way the executor feeds it.
        let (hub, lifecycle) = StatusHub::new();
        let writer = spawn_status_writer_checked(&hub, lifecycle, &job_dir).unwrap();
        let mut ctx = StageContext::for_test(job_dir.clone(), job_dir.join("stage"));
        ctx.node_idx = 3;
        ctx.status_tx = hub.broadcast_sender();
        std::fs::create_dir_all(&ctx.stage_dir).unwrap();
        let out = run(ctx, root).await.expect("the stage completes");
        assert_eq!(out.path, ckpt);
        drop(hub);
        writer.await.unwrap().expect("status.jsonl written");

        let updates = blut::jobs::read_status(JOB).unwrap();
        let summary = blut::jobs::list_jobs()
            .unwrap()
            .into_iter()
            .find(|j| j.id == JOB)
            .expect("the job is listed");
        let log = blut::jobs::tail_log(JOB, 100).unwrap();
        Relayed {
            updates,
            summary,
            log,
            ckpt,
        }
    }

    fn assert_progress_is_visible(run: &Relayed) {
        assert_eq!(
            run.updates,
            [
                StatusUpdate::Heartbeat {
                    phase: Some("step 1/4".into()),
                    vram_mb: Some(0),
                },
                StatusUpdate::Step {
                    step: 2,
                    total: 4,
                    loss: 2.5,
                    lr: 0.0002,
                    vram_mb: 812,
                },
                StatusUpdate::Step {
                    step: 4,
                    total: 4,
                    loss: 1.5,
                    lr: 0.0001,
                    vram_mb: 812,
                },
                StatusUpdate::Saved {
                    path: run.ckpt.join("checkpoint-4"),
                },
                StatusUpdate::Done {
                    final_loss: 1.75,
                    checkpoint_dir: run.ckpt.clone(),
                },
            ]
        );
        // `blut jobs`
        assert_eq!(run.summary.last_step, Some(4));
        assert_eq!(run.summary.last_loss, Some(1.5));
        assert_eq!(run.summary.final_loss, Some(1.75));
        // `blut log`
        let log = &run.log;
        assert!(log.contains("… step 1/4"), "{log}");
        assert!(log.contains("step 4/4  loss=1.5000"), "{log}");
        assert!(log.contains("done  final_loss=1.7500"), "{log}");
    }

    #[tokio::test]
    async fn hf_sft_progress_reaches_blut_log_and_blut_jobs() {
        let run = relayed(|ctx, root| {
            let data = root.join("train.jsonl");
            std::fs::write(&data, "{\"text\":\"hello\"}\n").unwrap();
            let input = DatasetJsonl {
                content_hash: ContentHash::hash_file(&data).unwrap(),
                path: data,
                n_examples: 1,
            };
            let args = serde_json::from_value(
                serde_json::json!({"base_model": "org/model", "lr": 1e-4, "epochs": 1}),
            )
            .unwrap();
            async move { HfSftTrain.run(&ctx, input, &args).await }
        })
        .await;
        assert_progress_is_visible(&run);
    }

    #[tokio::test]
    async fn hf_dpo_progress_reaches_blut_log_and_blut_jobs() {
        let run = relayed(|ctx, root| {
            let data = root.join("prefs.jsonl");
            std::fs::write(&data, "{\"prompt\":\"p\",\"chosen\":\"c\",\"rejected\":\"r\"}\n")
                .unwrap();
            let input = PreferenceJsonl {
                content_hash: ContentHash::hash_file(&data).unwrap(),
                path: data,
                n_pairs: 1,
            };
            let args = serde_json::from_value(
                serde_json::json!({"base_model": "org/model", "beta": 0.1, "lr": 5e-6, "epochs": 1}),
            )
            .unwrap();
            async move { HfDpoTrain.run(&ctx, input, &args).await }
        })
        .await;
        assert_progress_is_visible(&run);
    }
}
