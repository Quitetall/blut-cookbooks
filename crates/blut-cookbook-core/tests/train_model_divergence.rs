//! A diverged trainer reaches the engine's kill path.
//!
//! The engine kills a branch when a stage's forwarded step reports a
//! non-finite value (`KillOnNaN`, which `blut hpo` layers under its scheduler).
//! The stub stands in for `python -m blut_core.trainer` (no torch, no GPU),
//! prints what the trainer prints for a NaN loss, and then hangs, as a real
//! diverged run keeps training. Only the kill can end it within the test.
#![cfg(unix)]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use blut::framework::control::KillOnNaN;
use blut::framework::executor::{ExecCtx, ParallelExecutor};
use blut::framework::plan_spec::PlanSpec;
use blut::framework::status::StageEvent;

fn executable(path: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, body).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn alive(pid: i32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|s| {
        s.rsplit(')')
            .next()
            .is_some_and(|t| !t.trim().starts_with('Z'))
    })
}

fn plan(dataset: &Path) -> PlanSpec {
    let ingredient =
        |kind: &str, name: &str| serde_json::json!({"kind": kind, "name": name, "config": {}});
    serde_json::from_value(serde_json::json!({
        "name": "diverges",
        "nodes": [
            {"stage": "load_dataset", "args": {"path": dataset}},
            {"stage": "train_model", "args": {
                "model": ingredient("model", "custom_module"),
                "optimizer": ingredient("optimizer", "adamw"),
                "scheduler": ingredient("scheduler", "constant"),
                "loss": ingredient("loss", "mse"),
                "epochs": 1, "device": "cpu", "log_every_steps": 1,
            }},
        ],
        "edges": [[0, 1]],
    }))
    .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_nan_loss_kills_the_stage_as_a_divergence() {
    let td = tempfile::tempdir().unwrap();
    let pid_file = td.path().join("trainer.pid");
    let stub = td.path().join("python");
    executable(
        &stub,
        &format!(
            r#"#!/bin/sh
echo $$ > "{}"
echo 'BLUT_STATUS {{"kind":"step","step":1,"total":8,"loss":2.5,"lr":0.001,"vram_mb":0}}'
echo "[trainer] step 2/8 loss=nan"
echo 'BLUT_STATUS {{"kind":"heartbeat","phase":"step 2/8 loss=nan","vram_mb":0,"loss":"nan"}}'
exec sleep 600
"#,
            pid_file.display()
        ),
    );
    // SAFETY: the only test in this binary, before anything reads the variable.
    unsafe { std::env::set_var("BLUT_PYTHON", &stub) };

    let data = td.path().join("train.jsonl");
    std::fs::write(&data, "{\"text\":\"a\"}\n").unwrap();
    let mut reg = blut_cookbook_core::registry();
    blut::checks::register(&mut reg);
    let compiled = plan(&data).compile(&reg).expect("kinds line up");

    let job_dir = td.path().join("20261106-090000-000000001");
    std::fs::create_dir_all(&job_dir).unwrap();
    let ctx = ExecCtx::new(job_dir).with_control(Arc::new(KillOnNaN));
    let mut rx = ctx.status.subscribe();
    let result = tokio::time::timeout(
        Duration::from_secs(120),
        ParallelExecutor::execute(compiled, ctx),
    )
    .await
    .expect("the kill ends the run; the trainer alone would sleep for 600 s");
    let error = result
        .expect_err("a diverged stage fails the plan")
        .to_string();

    let mut failed = None;
    loop {
        match rx.try_recv() {
            Ok(StageEvent::StageFailed {
                stage_name, error, ..
            }) => failed = Some((stage_name, error)),
            Ok(_) | Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {}
            Err(_) => break,
        }
    }
    let (stage, detail) = failed.expect("the stage failure was reported");
    assert_eq!(stage, "train_model");
    assert!(
        detail.contains("diverged") && detail.contains("nan"),
        "a divergence, not a plain cancel: {detail} (plan: {error})"
    );
    let pid: i32 = std::fs::read_to_string(&pid_file)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    for _ in 0..300 {
        if !alive(pid) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("trainer {pid} survived the kill");
}
