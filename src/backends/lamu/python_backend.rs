//! `PythonTrainBackend` — runs `trainer.py` as a subprocess.
//!
//! Wire format: stdin is unused; one `StatusUpdate` JSON line per
//! line of stdout. The reader is a dedicated tokio task that
//! streams stdout into the `on_status` callback so the trainer
//! never blocks on a Rust-side queue.
//!
//! Cancellation: SIGTERM, 10s grace, SIGKILL. The grace period
//! lets the trainer flush partial checkpoints + close any open file
//! handles. Modeled on `lamu_core::backends::graceful_kill`; kept
//! local so this crate has no upward dep on lamu-core.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::oneshot;

use crate::backend::{StatusFn, TrainArtifact, TrainBackend};
use blut::error::{Result, TrainError};
use blut::protocol::StatusUpdate;
use blut::spec::TrainSpec;

/// Where to find the python interpreter and the trainer script.
///
/// Resolution is the caller's responsibility — `PythonTrainBackend`
/// takes both as explicit paths so tests can point at a stdlib-only
/// python and the bundled `trainer.py --self-check` mode without
/// any environment magic. The CLI binary (step 5) wires the
/// production resolver: `$LAMU_TRAIN_PYTHON` env > `~/local-llm/.venv/bin/python`
/// > `~/.local/share/lamu/train-venv/bin/python` > system `python3`.
#[derive(Clone, Debug)]
pub struct PythonTrainBackend {
    pub python: PathBuf,
    pub trainer_script: PathBuf,
    /// Extra env passed to the trainer subprocess (PYTHONPATH, etc.).
    pub env: Vec<(String, String)>,
    /// D3 liveness watchdog: if `Some`, a trainer that emits NO status
    /// line (Step/Eval/Saved/Heartbeat) AND NO stderr line for this long
    /// is presumed hung and SIGTERM'd. `None` (default) = no watchdog,
    /// so a chatty-or-heartbeating trainer is never falsely killed and
    /// old trainers (no heartbeats) are unaffected.
    pub liveness_timeout: Option<Duration>,
    /// Where to place the trainer (#3). `Local` (default) spawns it directly
    /// here; `Slurm` runs it as one `srun` step, placed by
    /// `distributed::training_argv_named` like every other training process,
    /// with the SAME streaming/cancel path. Any other target is refused.
    launch_target: blut::config::launcher::LaunchTarget,
    /// The Slurm job name: the stage's `StageContext::slurm_job_name()`, which
    /// the engine records with the BLUT job. `None` keeps the generic
    /// `blut-train`, which `blut cancel` and `plan resume` cannot find.
    job_name: Option<String>,
    child_pid: Arc<Mutex<Option<u32>>>,
}

impl PythonTrainBackend {
    pub fn new(python: PathBuf, trainer_script: PathBuf) -> Self {
        Self {
            python,
            trainer_script,
            env: Vec::new(),
            liveness_timeout: None,
            launch_target: blut::config::launcher::LaunchTarget::Local,
            job_name: None,
            child_pid: Arc::new(Mutex::new(None)),
        }
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Enable the D3 liveness watchdog with the given idle timeout.
    pub fn with_liveness_timeout(mut self, timeout: Duration) -> Self {
        self.liveness_timeout = Some(timeout);
        self
    }

    /// Place the trainer on `target` (#3). `Local` = direct spawn here.
    pub fn with_launch_target(mut self, target: blut::config::launcher::LaunchTarget) -> Self {
        self.launch_target = target;
        self
    }

    /// Name the Slurm job. A stage passes `ctx.slurm_job_name()`, so
    /// `blut cancel` can stop, and `plan resume` can see, a trainer the BLUT
    /// process that started it left running.
    pub fn with_job_name(mut self, name: impl Into<String>) -> Self {
        self.job_name = Some(name.into());
        self
    }

    /// The trainer's command line for `spec_json`:
    /// `python [torchrun prefix] trainer.py <spec>`, placed where
    /// `launch_target` says (`srun --job-name=<job_name> …` under Slurm).
    pub(crate) fn command(&self, spec: &TrainSpec, spec_json: &str) -> Result<Vec<String>> {
        let inner = [
            self.trainer_script.display().to_string(),
            spec_json.to_string(),
        ];
        // DDP: when nproc_per_node > 1 the trainer runs under torchrun, which
        // initializes the process group; under Slurm each node resolves its
        // own rank from the allocation.
        let (program, argv) = crate::distributed::training_argv_named(
            self.launch_target,
            &self.python.display().to_string(),
            spec.nproc_per_node,
            spec.nnodes,
            true,
            &inner,
            self.job_name.as_deref().unwrap_or("blut-train"),
        )
        .map_err(TrainError::other)?;
        let mut cmdline = vec![program];
        cmdline.extend(argv);
        Ok(cmdline)
    }
}

#[async_trait]
impl TrainBackend for PythonTrainBackend {
    async fn run(&mut self, spec: TrainSpec, on_status: StatusFn) -> Result<TrainArtifact> {
        spec.validate()?;
        let spec_json = serde_json::to_string(&spec)
            .map_err(|e| TrainError::other(format!("serialize TrainSpec for trainer.py: {e}")))?;

        let cmdline = self.command(&spec, &spec_json)?;
        // For DDP: don't pin CUDA_VISIBLE_DEVICES — torchrun manages LOCAL_RANK
        let env = self.env.clone();

        // A local run is contained (memory cap, peak, tree kill; the tree
        // also dies if this future is dropped), and on Unix the child leads
        // its own process group (KILL-1), so cancel reaches DataLoader workers
        // and torchrun ranks; see crate::local_stage.
        let mut child = crate::local_stage::spawn(
            self.launch_target,
            "blut-lamu_trainer",
            &cmdline,
            None,
            &env,
            |cmd| {
                cmd.stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true);
            },
        )
        .await
        .map_err(|e| {
            TrainError::Trainer(format!(
                "spawn {} {}: {}",
                self.python.display(),
                self.trainer_script.display(),
                e
            ))
        })?;
        if let Some(pid) = child.id() {
            *self.child_pid.lock() = Some(pid);
            // KILL-2: publish the child's identity so the in-process
            // cancel handler + a cross-process `blut cancel` (via the
            // job pid file) can reach the whole group.
            blut::python_kill::register_child(blut::python_kill::capture_identity(pid));
        }

        let stdout =
            child.child_mut().stdout.take().ok_or_else(|| {
                TrainError::Trainer("trainer subprocess stdout pipe missing".into())
            })?;

        let (artifact_tx, artifact_rx) = oneshot::channel();
        let started = Instant::now();
        let on_status: Arc<StatusFn> = Arc::new(on_status);

        // D3 liveness: every status line AND every stderr line refreshes
        // `last_activity`; the watchdog kills a trainer that goes silent
        // on BOTH for `liveness_timeout`.
        let last_activity = Arc::new(Mutex::new(Instant::now()));
        let watchdog_reason: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        // Stdout reader: forwards every parsed StatusUpdate to the
        // caller's callback. Captures the terminal Done/Failed and
        // ships an artifact down the oneshot channel. Heartbeats (D4)
        // refresh liveness but are SWALLOWED (no status.jsonl noise).
        let on_status_for_reader = Arc::clone(&on_status);
        let activity_stdout = Arc::clone(&last_activity);
        let stdout_reader = tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            let mut last_done: Option<(f32, PathBuf)> = None;
            let mut last_failed: Option<String> = None;
            while let Ok(Some(line)) = reader.next_line().await {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<StatusUpdate>(line) {
                    Ok(u) => {
                        *activity_stdout.lock() = Instant::now();
                        if matches!(u, StatusUpdate::Heartbeat { .. }) {
                            continue; // liveness only — don't forward
                        }
                        if let StatusUpdate::Done {
                            final_loss,
                            checkpoint_dir,
                        } = &u
                        {
                            last_done = Some((*final_loss, checkpoint_dir.clone()));
                        }
                        if let StatusUpdate::Failed { error } = &u {
                            last_failed = Some(error.clone());
                        }
                        (on_status_for_reader)(u);
                    }
                    Err(e) => {
                        // Malformed lines never stall the run — log + continue.
                        tracing::warn!(
                            "trainer.py emitted unparseable status: {} ({} bytes): {}",
                            e,
                            line.len(),
                            line.chars().take(200).collect::<String>()
                        );
                    }
                }
            }
            let _ = artifact_tx.send((last_done, last_failed));
        });

        // Stderr drain. Forward to tracing so a buggy trainer's
        // python traceback doesn't disappear into a closed pipe. Stderr
        // chatter (e.g. tqdm) is also a liveness signal.
        // The pump also records the systemd backend's `Memory peak:` line.
        let activity_stderr = Arc::clone(&last_activity);
        if !child.pump_stderr(move |line| {
            *activity_stderr.lock() = Instant::now();
            tracing::info!(target: "blut::trainer_stderr", "{}", line);
        }) {
            return Err(TrainError::Trainer(
                "trainer subprocess stderr pipe missing".into(),
            ));
        }

        // D3 watchdog task (only when configured). Ticks at min(timeout/4,
        // 30s); on idle > timeout it records a reason and fires
        // `watchdog_kill`, and the wait below stops the whole tree
        // (local_stage::stop_tree: the backend's teardown, or SIGTERM then
        // SIGKILL to the process group). Aborted below once the child exits.
        let watchdog_kill = tokio_util::sync::CancellationToken::new();
        let watchdog = self.liveness_timeout.map(|timeout| {
            let activity = Arc::clone(&last_activity);
            let reason = Arc::clone(&watchdog_reason);
            let kill = watchdog_kill.clone();
            tokio::spawn(async move {
                let tick = std::cmp::min(timeout / 4, Duration::from_secs(30))
                    .max(Duration::from_millis(100));
                loop {
                    tokio::time::sleep(tick).await;
                    let idle = activity.lock().elapsed();
                    if idle <= timeout {
                        continue;
                    }
                    *reason.lock() = Some(format!(
                        "liveness watchdog: no output for {idle:?} (limit {timeout:?})"
                    ));
                    tracing::warn!(
                        "liveness watchdog: no trainer output for {idle:?} (> {timeout:?}); \
                         stopping it"
                    );
                    kill.cancel();
                    return;
                }
            })
        });

        // `None`: the watchdog fired and the tree was killed.
        let exit_status = crate::local_stage::wait_or_cancel(&mut child, &watchdog_kill)
            .await
            .map_err(|e| TrainError::Trainer(format!("wait for trainer.py: {e}")))?;

        // Child exited — stop the watchdog if it's still ticking.
        if let Some(w) = &watchdog {
            w.abort();
        }

        // Reader tasks finish once their pipes hit EOF (they always
        // do once the child exits). Awaiting here serializes the
        // final on_status callback before we return.
        let _ = stdout_reader.await;

        if let Some(pid) = self.child_pid.lock().take() {
            blut::python_kill::unregister_child(pid);
        }
        let elapsed = started.elapsed();

        // If the watchdog killed the trainer, surface THAT (a hang), not
        // the generic "exited with no Done" — it's the actionable cause.
        if let Some(reason) = watchdog_reason.lock().take() {
            return Err(TrainError::Trainer(reason));
        }

        let (last_done, last_failed) = artifact_rx
            .await
            .map_err(|_| TrainError::Trainer("status reader dropped before report".into()))?;

        if let Some(error) = last_failed {
            return Err(TrainError::Trainer(error));
        }
        let exit_status = exit_status.ok_or_else(|| {
            TrainError::Trainer("trainer.py was killed by the liveness watchdog".into())
        })?;
        if !exit_status.success() {
            return Err(TrainError::Trainer(format!(
                "trainer.py exited with {exit_status} and emitted no Failed status"
            )));
        }
        let (final_loss, checkpoint_dir) = last_done.ok_or_else(|| {
            TrainError::Trainer("trainer.py exited successfully but emitted no Done status".into())
        })?;

        Ok(TrainArtifact {
            checkpoint_dir,
            gguf_path: None,
            final_loss,
            elapsed,
        })
    }

    async fn cancel(&mut self) -> Result<()> {
        // Atomic take() rather than read-then-clear so two concurrent
        // cancel() calls don't both fire the kill sequence.
        let pid = match self.child_pid.lock().take() {
            Some(p) => p,
            None => return Ok(()),
        };
        graceful_kill(pid).await;
        Ok(())
    }
}

async fn graceful_kill(pid: u32) {
    blut::python_kill::graceful_kill_pid(pid, Duration::from_secs(10)).await
}

/// Where the trainer runs, and what its Slurm job is called.
#[cfg(test)]
mod placement_tests {
    use super::*;
    use blut::config::launcher::LaunchTarget;
    use blut::spec::{DatasetSource, Method, Optim, TrainSpec};

    fn spec() -> TrainSpec {
        TrainSpec {
            base_model: "org/m".into(),
            output_name: "placement".into(),
            output_dir: PathBuf::from("out"),
            method: Method::Full,
            dataset: DatasetSource::JsonlPath {
                path: PathBuf::from("x.jsonl"),
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
            dpo_beta: None,
            nproc_per_node: 1,
            nnodes: 1,
        }
    }

    fn backend() -> PythonTrainBackend {
        PythonTrainBackend::new("py".into(), "trainer.py".into())
    }

    #[test]
    fn a_local_run_is_the_bare_trainer() {
        assert_eq!(
            backend().command(&spec(), "{}").unwrap(),
            ["py", "trainer.py", "{}"]
        );
    }

    /// The name the stage hands over is the Slurm job's name, which the
    /// engine records with the BLUT job: `blut cancel` and `plan resume` find
    /// the job by it. The generic `blut-train` they cannot.
    #[test]
    fn a_slurm_step_carries_the_name_it_was_given() {
        let cmd = backend()
            .with_launch_target(LaunchTarget::Slurm)
            .with_job_name("blut-20261030-120000-1-2")
            .command(&spec(), "{}")
            .unwrap();
        assert_eq!(cmd[0], "srun");
        assert!(
            cmd.contains(&"--job-name=blut-20261030-120000-1-2".to_string()),
            "{cmd:?}"
        );
        assert!(!cmd.iter().any(|a| a == "--job-name=blut-train"), "{cmd:?}");
        assert_eq!(cmd[cmd.len() - 3..], ["py", "trainer.py", "{}"]);
    }

    /// A placement this backend cannot honor is an error, never a local run.
    #[test]
    fn another_launcher_is_refused() {
        let err = backend()
            .with_launch_target(LaunchTarget::Ray)
            .command(&spec(), "{}")
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("training cannot run under --launcher"),
            "{err}"
        );
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::backend::StatusFn;
    use blut::spec::{DatasetSource, Method, Optim, TrainSpec};

    fn python3() -> Option<PathBuf> {
        which::which("python3").ok()
    }

    fn spec() -> TrainSpec {
        TrainSpec {
            base_model: "org/m".into(),
            output_name: "wd-test".into(),
            output_dir: PathBuf::from("/tmp/blut-wd-test"),
            method: Method::QLora {
                rank: 16,
                alpha: 32,
            },
            dataset: DatasetSource::JsonlPath {
                path: PathBuf::from("/tmp/x.jsonl"),
            },
            optimizer: Optim::AdamW8bit,
            lr: 2e-4,
            epochs: 1,
            batch_size: 1,
            grad_accum: 1,
            seq_len: 512,
            seed: 42,
            quant: "Q4_K_M".into(),
            skip_convert: true,
            dpo_beta: None,
            nproc_per_node: 1,
            nnodes: 1,
        }
    }

    /// Write a python script that prints one Step line, then runs the
    /// given tail (`silent` = sleep forever; `heartbeat` = ping then Done).
    fn trainer_script(td: &std::path::Path, tail: &str) -> PathBuf {
        let body = format!(
            "import json,sys,time\n\
             print(json.dumps({{'kind':'step','step':1,'total':10,'loss':1.0,'lr':0.0,'vram_mb':0}}),flush=True)\n\
             {tail}\n"
        );
        let p = td.join("fake_trainer.py");
        std::fs::write(&p, body).unwrap();
        p
    }

    #[tokio::test]
    async fn liveness_watchdog_kills_a_silent_trainer() {
        let Some(py) = python3() else {
            eprintln!("skip: python3 not found");
            return;
        };
        let td = tempfile::tempdir().unwrap();
        // Prints one Step, then goes silent for 60s — the watchdog must
        // kill it well before that.
        let script = trainer_script(td.path(), "time.sleep(60)");
        let mut backend =
            PythonTrainBackend::new(py, script).with_liveness_timeout(Duration::from_millis(500));
        let on_status: StatusFn = Box::new(|_u| {});
        let fut = backend.run(spec(), on_status);
        let r = tokio::time::timeout(Duration::from_secs(10), fut)
            .await
            .expect("watchdog must fire long before the 10s guard");
        match r {
            Err(TrainError::Trainer(msg)) => {
                assert!(msg.contains("liveness watchdog"), "got: {msg}");
            }
            other => panic!("expected a liveness-watchdog error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn heartbeating_trainer_survives_the_watchdog() {
        let Some(py) = python3() else {
            eprintln!("skip: python3 not found");
            return;
        };
        let td = tempfile::tempdir().unwrap();
        let ckpt = td.path().join("ckpt");
        std::fs::create_dir_all(&ckpt).unwrap();
        // Heartbeats every 100ms for ~1.5s (keeping liveness fresh under
        // a 500ms timeout), then Done.
        let tail = format!(
            "for _ in range(15):\n\
             \x20 print(json.dumps({{'kind':'heartbeat','phase':'load'}}),flush=True)\n\
             \x20 time.sleep(0.1)\n\
             print(json.dumps({{'kind':'done','final_loss':0.5,'checkpoint_dir':'{}'}}),flush=True)",
            ckpt.display()
        );
        let script = trainer_script(td.path(), &tail);
        let mut backend =
            PythonTrainBackend::new(py, script).with_liveness_timeout(Duration::from_millis(500));
        let on_status: StatusFn = Box::new(|_u| {});
        let r = tokio::time::timeout(Duration::from_secs(10), backend.run(spec(), on_status))
            .await
            .expect("must complete");
        assert!(r.is_ok(), "heartbeats must keep the trainer alive: {r:?}");
    }
}
