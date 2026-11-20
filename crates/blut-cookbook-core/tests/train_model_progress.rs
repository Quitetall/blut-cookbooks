//! `train_model`'s progress reaches `blut log` and `blut jobs`.
//!
//! The stage runs a stub in place of `python -m blut_core.trainer` (no torch,
//! no GPU) that prints what the trainer prints: `[trainer]` lines,
//! `BLUT_METRIC` lines, and its updates marked `BLUT_STATUS`. The engine's own
//! status writer persists what the stage forwards, and the engine's own readers
//! read it back: `read_status` (what `blut log` renders), `list_jobs` (what
//! `blut jobs` prints) and `tail_log`.
#![cfg(unix)]

use std::path::Path;

use blut::artifacts::{DatasetJsonl, HfCheckpoint};
use blut::framework::artifact::ContentHash;
use blut::framework::stage::{Stage, StageContext};
use blut::framework::status::{StatusHub, spawn_status_writer_checked};
use blut::protocol::StatusUpdate;
use blut_cookbook_core::stages::train_model::{Args, TrainModel, TrainModelOnSplit};

const JOB: &str = "20261031-090000-000000001";

/// One test at a time touches the process environment.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn executable(path: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, body).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// What the trainer prints for a two-step epoch logged every step: the second
/// step's loss was NaN, which it reports as a heartbeat.
const TRAINER_STDOUT: &str = r#"#!/bin/sh
out="$(dirname "$4")/checkpoint"
echo "[trainer] starting training: 1 epochs, device=cpu"
echo 'BLUT_METRIC {"step": 1, "loss": 2.5, "epoch": 1, "kind": "step"}'
echo "[trainer] step 1/2 loss=2.5000"
echo 'BLUT_STATUS {"kind":"step","step":1,"total":2,"loss":2.5,"lr":0.001,"vram_mb":0}'
echo "[trainer] step 2/2 loss=nan"
echo 'BLUT_STATUS {"kind":"heartbeat","phase":"step 2/2 loss=nan","vram_mb":0,"loss":"nan"}'
echo "[trainer] epoch 1/1: loss=2.0000"
echo 'BLUT_STATUS {"kind":"heartbeat","phase":"epoch 1/1 loss=2.0000"}'
echo weights > "$out/model.pt"
echo "BLUT_STATUS {\"kind\":\"saved\",\"path\":\"$out/model.pt\"}"
echo "[trainer] training complete"
echo "BLUT_STATUS {\"kind\":\"done\",\"final_loss\":2.0,\"checkpoint_dir\":\"$out\"}"
"#;

fn args() -> Args {
    serde_json::from_value(serde_json::json!({
        "model": {"kind": "model", "name": "from_pretrained", "config": {}},
        "optimizer": {"kind": "optimizer", "name": "adamw", "config": {}},
        "scheduler": {"kind": "scheduler", "name": "constant", "config": {}},
        "loss": {"kind": "loss", "name": "causal_lm", "config": {}},
        "epochs": 1,
        "device": "cpu",
        "log_every_steps": 1,
    }))
    .unwrap()
}

struct EnvVars(Vec<(&'static str, Option<std::ffi::OsString>)>);

impl EnvVars {
    fn set(vars: &[(&'static str, &Path)]) -> Self {
        let saved = vars
            .iter()
            .map(|(k, _)| (*k, std::env::var_os(k)))
            .collect();
        for (k, v) in vars {
            // SAFETY: tests in this binary hold ENV_LOCK while the variables are set.
            unsafe { std::env::set_var(k, v) };
        }
        Self(saved)
    }
}

impl Drop for EnvVars {
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

/// Run `stage` against the stub `trainer`; return the updates `read_status`
/// reads back, the job's `list_jobs` row, `tail_log`'s rendering, and the
/// checkpoint.
#[allow(clippy::await_holding_lock)]
async fn run_with_stub(
    stage: &'static str,
    trainer: &str,
) -> (
    Vec<StatusUpdate>,
    blut::jobs::JobSummary,
    String,
    HfCheckpoint,
) {
    let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let td = tempfile::tempdir().unwrap();
    let stub = td.path().join("python");
    executable(&stub, trainer);
    let jobs = td.path().join("jobs");
    let _env = EnvVars::set(&[("BLUT_PYTHON", &stub), ("LAMU_TRAIN_JOBS_DIR", &jobs)]);

    let job_dir = blut::paths::job_dir(JOB).unwrap();
    let (hub, lifecycle) = StatusHub::new();
    let writer = spawn_status_writer_checked(&hub, lifecycle, &job_dir).unwrap();
    let mut ctx = StageContext::for_test(job_dir.clone(), job_dir.join("stage"));
    ctx.node_idx = 2;
    ctx.status_tx = hub.broadcast_sender();
    std::fs::create_dir_all(&ctx.stage_dir).unwrap();

    let data = td.path().join("train.jsonl");
    std::fs::write(&data, "{\"text\":\"a\"}\n{\"text\":\"b\"}\n").unwrap();
    let dataset = DatasetJsonl {
        content_hash: ContentHash::hash_file(&data).unwrap(),
        path: data,
        n_examples: 2,
    };
    let ckpt = match stage {
        "train_model" => TrainModel.run(&ctx, dataset, &args()).await,
        "train_model_on_split" => {
            let split = blut::artifacts::DatasetSplit {
                train: dataset.clone(),
                eval: dataset,
            };
            TrainModelOnSplit.run(&ctx, split, &args()).await
        }
        other => unreachable!("{other}"),
    }
    .expect("the stage completes");
    drop(ctx);
    drop(hub);
    writer.await.unwrap().expect("status.jsonl written");

    let updates = blut::jobs::read_status(JOB).unwrap();
    let summary = blut::jobs::list_jobs()
        .unwrap()
        .into_iter()
        .find(|j| j.id == JOB)
        .expect("the job is listed");
    let log = blut::jobs::tail_log(JOB, 100).unwrap();
    // The forwarded lines name the stage that sent them.
    let status = std::fs::read_to_string(job_dir.join("status.jsonl")).unwrap();
    let steps = status
        .lines()
        .filter(|l| l.contains("\"stage_step\""))
        .collect::<Vec<_>>();
    assert_eq!(steps.len(), updates.len(), "{status}");
    for l in steps {
        assert!(
            l.contains(&format!("\"node_idx\":2,\"stage_name\":\"{stage}\"")),
            "{l}"
        );
    }
    (updates, summary, log, ckpt)
}

fn assert_progress_is_visible(
    updates: &[StatusUpdate],
    summary: &blut::jobs::JobSummary,
    log: &str,
    ckpt: &HfCheckpoint,
) {
    let dir = ckpt.path.as_path();
    assert_eq!(
        updates,
        [
            StatusUpdate::Step {
                step: 1,
                total: 2,
                loss: 2.5,
                lr: 0.001,
                vram_mb: 0,
            },
            StatusUpdate::Heartbeat {
                phase: Some("step 2/2 loss=nan".into()),
                vram_mb: Some(0),
            },
            StatusUpdate::Heartbeat {
                phase: Some("epoch 1/1 loss=2.0000".into()),
                vram_mb: None,
            },
            StatusUpdate::Saved {
                path: dir.join("model.pt"),
            },
            StatusUpdate::Done {
                final_loss: 2.0,
                checkpoint_dir: dir.to_path_buf(),
            },
        ]
    );
    // `blut jobs`
    assert_eq!(summary.last_step, Some(1));
    assert_eq!(summary.last_loss, Some(2.5));
    assert_eq!(summary.final_loss, Some(2.0));
    // `blut log`
    assert!(log.contains("step 1/2  loss=2.5000"), "{log}");
    assert!(log.contains("… epoch 1/1 loss=2.0000"), "{log}");
    assert!(log.contains("done  final_loss=2.0000"), "{log}");
    // The terminal's lines are not progress.
    assert!(!log.contains("[trainer]"), "{log}");
    // The checkpoint records the loss the trainer reported, not 0.0.
    assert_eq!(ckpt.final_loss, Some(2.0));
}

#[tokio::test]
async fn train_model_progress_reaches_blut_log_and_blut_jobs() {
    let (updates, summary, log, ckpt) = run_with_stub("train_model", TRAINER_STDOUT).await;
    assert_progress_is_visible(&updates, &summary, &log, &ckpt);
}

#[tokio::test]
async fn train_model_on_split_progress_is_its_own() {
    let (updates, summary, log, ckpt) = run_with_stub("train_model_on_split", TRAINER_STDOUT).await;
    assert_progress_is_visible(&updates, &summary, &log, &ckpt);
}

/// An attempt that trained no epoch reports no `done`: the checkpoint records
/// no final loss rather than a made-up one.
#[tokio::test]
async fn a_run_that_reports_no_final_loss_records_none() {
    let trainer = TRAINER_STDOUT
        .lines()
        .filter(|l| !l.contains("done"))
        .collect::<Vec<_>>()
        .join("\n");
    let (_, summary, _, ckpt) = run_with_stub("train_model", &trainer).await;
    assert_eq!(ckpt.final_loss, None);
    assert_eq!(summary.final_loss, None);
}
