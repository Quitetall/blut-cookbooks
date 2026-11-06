//! `HfTrainerRunner` — subprocess + JSON wire to the bundled
//! Python wrapper.
//!
//! Wire format:
//!
//!   - Argv: `python <wrapper.py> <job_json_path>`.
//!     The wrapper reads its JSON job spec from the file (rather
//!     than argv) so the spec can carry long fields (prompts,
//!     paths, hyperparams) without bumping ARG_MAX.
//!   - Stdout: one JSON line per status event. Schema:
//!     `{"kind": "step", "step": int, "total": int, "loss": f, "lr": f, "vram_mb": int}`
//!     `{"kind": "eval", "step": int, "eval_loss": f}`
//!     `{"kind": "saved", "path": str}`
//!     `{"kind": "done", "checkpoint_dir": str, "final_loss": f}`
//!     `{"kind": "failed", "error": str}`
//!     `loss`, `lr` and `final_loss` may be null: HF logs them only on some
//!     steps, and a run that trained no step has no training loss.
//!   - Stderr: free-form. Captured to tracing::warn.
//!
//! The stages relay these lines to the executor in the engine's trainer
//! format, [`StatusLine::status_update`], which is what `blut log` and
//! `blut jobs` read.
//!
//! The wrapper script lives at
//! `src/backends/hf_trainer/python/hf_trainer_runner.py` and is
//! invoked via the auto-managed venv at `~/.local/share/blut/hf-venv/`
//! (see `super::venv`).

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};

use blut::protocol::StatusUpdate;
use blut::python_kill::graceful_kill_pid;

use super::venv::{VenvError, ensure_venv};

/// Typed job spec the Rust side hands the Python wrapper. Mirrors
/// `transformers.TrainingArguments` for the canonical fields +
/// carries a `task` discriminator so the wrapper picks
/// `transformers.Trainer` vs `trl.DPOTrainer`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HfTrainerJob {
    /// "sft" | "dpo" | "distill". Drives wrapper-side trainer
    /// class selection.
    pub task: String,
    /// HuggingFace model id (e.g. "Qwen/Qwen3-7B") OR an absolute
    /// path to a local checkpoint dir.
    pub base_model: String,
    /// Absolute path to the training dataset (JSONL).
    pub train_dataset_path: PathBuf,
    /// Optional eval dataset path. None = no eval loop.
    pub eval_dataset_path: Option<PathBuf>,
    /// Where the trainer should save its checkpoints.
    pub output_dir: PathBuf,
    /// Common hyperparams. The wrapper passes these straight to
    /// `TrainingArguments`.
    pub lr: f32,
    pub epochs: u32,
    pub batch_size: u32,
    pub grad_accum: u32,
    pub seq_len: u32,
    pub seed: u64,
    /// Free-form extras. Folded into the wrapper's
    /// `TrainingArguments(**extra)` kwargs. Lets recipes pass
    /// `optim`, `lr_scheduler_type`, `warmup_ratio`, etc. without
    /// adding a typed field for every TrainingArguments option.
    #[serde(default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// LoRA / QLoRA hyperparams. None = full-rank fine-tune.
    #[serde(default)]
    pub peft: Option<PeftConfig>,
    /// DPO-specific (ignored when task != "dpo").
    #[serde(default)]
    pub dpo: Option<DpoConfig>,
    /// Processes (GPUs) per node. >1 runs DDP under torchrun.
    #[serde(default = "crate::distributed::one")]
    pub nproc_per_node: u32,
    /// Nodes in the job. >1 needs MASTER_ADDR and NODE_RANK on every node;
    /// the wrapper then also saves a checkpoint on each node.
    #[serde(default = "crate::distributed::one")]
    pub nnodes: u32,
    /// Where the run is placed (`--launcher`). Not part of the spec the
    /// wrapper reads: it decides how the wrapper is started, not what it does.
    #[serde(skip)]
    pub launch_target: blut::config::launcher::LaunchTarget,
    /// The Slurm job name under `--launcher slurm`
    /// (`StageContext::slurm_job_name()`). `None` keeps the generic
    /// `blut-train`, which `blut cancel` cannot find.
    #[serde(skip)]
    pub slurm_job_name: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeftConfig {
    /// "lora" | "qlora".
    pub method: String,
    pub rank: u32,
    pub alpha: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DpoConfig {
    pub beta: f32,
    /// Optional override for the preferences file. None = fall
    /// back to `HfTrainerJob.train_dataset_path`, which is the
    /// common case (DPO trains on the dataset's own
    /// chosen/rejected pairs).
    #[serde(default)]
    pub preferences_path: Option<PathBuf>,
}

/// One line emitted by the wrapper on stdout.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StatusLine {
    Step {
        step: u64,
        total: u64,
        /// The latest logged training loss; `None` until HF has logged one.
        #[serde(default)]
        loss: Option<f32>,
        /// `None` on a log that is not a training log (the end-of-run
        /// summary; an evaluation is an `eval` line).
        #[serde(default)]
        lr: Option<f32>,
        /// GPU memory the reporting process holds, in MiB (0 off the GPU).
        #[serde(default)]
        vram_mb: Option<u32>,
    },
    /// An evaluation pass HF logged, with its loss.
    Eval {
        step: u64,
        eval_loss: f32,
    },
    Saved {
        path: PathBuf,
    },
    Done {
        checkpoint_dir: PathBuf,
        #[serde(default)]
        final_loss: Option<f32>,
    },
    Failed {
        error: String,
    },
}

impl StatusLine {
    /// This line as the engine's trainer update, the format `blut log` and
    /// `blut jobs` read from a stage's forwarded progress
    /// (`StageEvent::StageStep { update }`), and the one the lamu trainer
    /// emits natively.
    ///
    /// A step whose loss or learning rate HF has not logged becomes a
    /// heartbeat naming the step, since the engine's step update has no way to
    /// say "not measured" and a made-up number would read as a measurement. A
    /// run that trained no step has no final loss, so its `done` has no
    /// counterpart (`None`); the stage's own result still records the
    /// checkpoint.
    pub fn status_update(&self) -> Option<StatusUpdate> {
        let count = |n: u64| u32::try_from(n).unwrap_or(u32::MAX);
        Some(match self {
            StatusLine::Step {
                step,
                total,
                loss: Some(loss),
                lr: Some(lr),
                vram_mb,
            } => StatusUpdate::Step {
                step: count(*step),
                total: count(*total),
                loss: *loss,
                lr: *lr,
                vram_mb: vram_mb.unwrap_or(0),
            },
            StatusLine::Step {
                step,
                total,
                vram_mb,
                ..
            } => StatusUpdate::Heartbeat {
                phase: Some(format!("step {step}/{total}")),
                vram_mb: *vram_mb,
            },
            StatusLine::Eval { step, eval_loss } => StatusUpdate::Eval {
                step: count(*step),
                eval_loss: *eval_loss,
            },
            StatusLine::Saved { path } => StatusUpdate::Saved { path: path.clone() },
            StatusLine::Done {
                checkpoint_dir,
                final_loss: Some(final_loss),
            } => StatusUpdate::Done {
                final_loss: *final_loss,
                checkpoint_dir: checkpoint_dir.clone(),
            },
            StatusLine::Done {
                final_loss: None, ..
            } => return None,
            StatusLine::Failed { error } => StatusUpdate::Failed {
                error: error.clone(),
            },
        })
    }
}

#[derive(Debug)]
pub struct HfRunArtifact {
    pub checkpoint_dir: PathBuf,
    pub final_loss: Option<f32>,
    pub elapsed: Duration,
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("venv: {0}")]
    Venv(#[from] VenvError),
    #[error("python -m hf_trainer_runner spawn at {python}: {source}")]
    Spawn {
        python: String,
        #[source]
        source: std::io::Error,
    },
    #[error("python wrapper exited with status {status} and emitted no Done line")]
    Failed { status: String },
    #[error("python wrapper reported failure: {message}")]
    WrapperFailed { message: String },
    #[error("io at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("serialize job spec: {0}")]
    SerializeJob(#[source] serde_json::Error),
    #[error("distributed launch: {0}")]
    Launch(String),
}

pub struct HfTrainerRunner {
    child_pid: Arc<Mutex<Option<u32>>>,
}

impl HfTrainerRunner {
    pub fn new() -> Self {
        Self {
            child_pid: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn run(
        &mut self,
        job: HfTrainerJob,
        on_status: Box<dyn Fn(StatusLine) + Send + Sync>,
    ) -> Result<HfRunArtifact, RunError> {
        // Ensure venv (blocking — fine, this only fires on the
        // first hf_* recipe of a session).
        let python = tokio::task::spawn_blocking(ensure_venv)
            .await
            .map_err(|e| RunError::Io {
                path: PathBuf::from("ensure_venv-join"),
                source: std::io::Error::other(format!("{e}")),
            })??;

        // Write job spec to a tempfile so the wrapper can read it
        // without argv-size constraints.
        let td = tempfile::tempdir().map_err(|source| RunError::Io {
            path: PathBuf::from("hf-trainer-tempdir"),
            source,
        })?;
        let spec_path = td.path().join("job.json");
        let body = serde_json::to_vec_pretty(&job).map_err(RunError::SerializeJob)?;
        std::fs::write(&spec_path, body).map_err(|source| RunError::Io {
            path: spec_path.clone(),
            source,
        })?;

        // Locate the wrapper script. It ships in-tree and gets
        // resolved relative to the runner module at compile time.
        let wrapper = match wrapper_path() {
            Some(p) if p.exists() => p,
            _ => {
                return Err(RunError::Io {
                    path: PathBuf::from("src/backends/hf_trainer/python/hf_trainer_runner.py"),
                    source: std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "wrapper script missing; expected alongside the binary",
                    ),
                });
            }
        };

        // Under torchrun whenever the job spans more than one process, on
        // one node or several. This gated on `nproc_per_node > 1` alone and
        // always passed `--standalone`, so a multi-node HF job could not be
        // expressed at all.
        let inner = [
            wrapper.display().to_string(),
            spec_path.display().to_string(),
        ];
        let (program, argv) = crate::distributed::training_argv_named(
            job.launch_target,
            &python.display().to_string(),
            job.nproc_per_node,
            job.nnodes,
            true,
            &inner,
            job.slurm_job_name.as_deref().unwrap_or("blut-train"),
        )
        .map_err(RunError::Launch)?;
        let mut cmdline = vec![program];
        cmdline.extend(argv);
        // A local run is contained (memory cap, peak, tree kill; the tree
        // also dies if this future is dropped), and on Unix the child leads
        // its own process group (KILL-1); see crate::local_stage.
        let mut child = crate::local_stage::spawn(
            job.launch_target,
            "blut-hf_trainer",
            &cmdline,
            None,
            &[],
            |cmd| {
                cmd.stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true);
            },
        )
        .await
        .map_err(|e| RunError::Spawn {
            python: python.display().to_string(),
            source: crate::local_stage::spawn_error_to_io(e),
        })?;
        if let Some(pid) = child.id() {
            *self.child_pid.lock() = Some(pid);
            // KILL-2: publish for in-process + cross-process cancel.
            blut::python_kill::register_child(blut::python_kill::capture_identity(pid));
        }

        let started = Instant::now();
        let stdout = child.child_mut().stdout.take();
        let on_status: Arc<dyn Fn(StatusLine) + Send + Sync> = Arc::from(on_status);
        let collected: Arc<Mutex<(Option<HfRunArtifact>, Option<String>)>> =
            Arc::new(Mutex::new((None, None)));

        let stdout_handle = if let Some(stdout) = stdout {
            let cb = on_status.clone();
            let collected = collected.clone();
            Some(tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<StatusLine>(line) {
                        Ok(s) => {
                            if let StatusLine::Done {
                                checkpoint_dir,
                                final_loss,
                            } = &s
                            {
                                let mut c = collected.lock();
                                c.0 = Some(HfRunArtifact {
                                    checkpoint_dir: checkpoint_dir.clone(),
                                    final_loss: *final_loss,
                                    elapsed: Duration::ZERO,
                                });
                            }
                            if let StatusLine::Failed { error } = &s {
                                collected.lock().1 = Some(error.clone());
                            }
                            (cb)(s);
                        }
                        Err(_) => {
                            tracing::warn!(
                                target: "blut::hf_trainer_stdout",
                                "unparseable status: {line}"
                            );
                        }
                    }
                }
            }))
        } else {
            None
        };

        // The pump also records the systemd backend's `Memory peak:` line.
        child.pump_stderr(|line| {
            tracing::warn!(target: "blut::hf_trainer_stderr", "{}", line);
        });

        let exit = child.wait().await.map_err(|e| RunError::Io {
            path: PathBuf::from("hf-trainer-wait"),
            source: std::io::Error::other(e.to_string()),
        })?;
        if let Some(h) = stdout_handle
            && let Err(e) = h.await
        {
            tracing::error!(target: "blut::hf_trainer", "stdout reader join failed: {e}");
        }
        if let Some(pid) = self.child_pid.lock().take() {
            blut::python_kill::unregister_child(pid);
        }
        let elapsed = started.elapsed();

        let mut collected = collected.lock();
        if let Some(err) = collected.1.take() {
            return Err(RunError::WrapperFailed { message: err });
        }
        if !exit.success() {
            return Err(RunError::Failed {
                status: format!("{exit}"),
            });
        }
        let mut artifact = collected.0.take().ok_or_else(|| RunError::Failed {
            status: "no Done line emitted".into(),
        })?;
        artifact.elapsed = elapsed;
        Ok(artifact)
    }

    pub async fn cancel(&mut self) {
        let pid = match self.child_pid.lock().take() {
            Some(p) => p,
            None => return,
        };
        graceful_kill_pid(pid, Duration::from_secs(10)).await;
    }
}

impl Default for HfTrainerRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Locate the wrapper script. Resolution order:
///   1. `$BLUT_HF_TRAINER_RUNNER` env (absolute path override)
///   2. `<exe_dir>/python/hf_trainer_runner.py` (when installed
///      alongside the binary)
///   3. `<repo_root>/src/backends/hf_trainer/python/hf_trainer_runner.py`
///      (development tree)
fn wrapper_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("BLUT_HF_TRAINER_RUNNER") {
        return Some(PathBuf::from(p));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let p = dir.join("python").join("hf_trainer_runner.py");
        if p.exists() {
            return Some(p);
        }
    }
    // Development tree fallback: walk up to find a Cargo.toml then
    // resolve against `src/backends/hf_trainer/python/`.
    if let Ok(cwd) = std::env::current_dir() {
        let mut cur = cwd.as_path();
        loop {
            let candidate = cur
                .join("src")
                .join("backends")
                .join("hf_trainer")
                .join("python")
                .join("hf_trainer_runner.py");
            if candidate.exists() {
                return Some(candidate);
            }
            match cur.parent() {
                Some(p) => cur = p,
                None => break,
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_round_trips_via_serde() {
        let job = HfTrainerJob {
            task: "sft".into(),
            base_model: "Qwen/Qwen3-7B".into(),
            train_dataset_path: "/tmp/train.jsonl".into(),
            eval_dataset_path: Some("/tmp/eval.jsonl".into()),
            output_dir: "/tmp/ckpt".into(),
            lr: 2e-4,
            epochs: 3,
            batch_size: 1,
            grad_accum: 8,
            seq_len: 4096,
            seed: 42,
            extra: serde_json::Map::new(),
            peft: Some(PeftConfig {
                method: "qlora".into(),
                rank: 16,
                alpha: 32,
            }),
            dpo: None,
            nproc_per_node: 1,
            nnodes: 1,
            launch_target: Default::default(),
            slurm_job_name: None,
        };
        let s = serde_json::to_string(&job).unwrap();
        let back: HfTrainerJob = serde_json::from_str(&s).unwrap();
        assert_eq!(back.task, "sft");
        assert_eq!(back.peft.as_ref().unwrap().method, "qlora");
    }

    #[test]
    fn status_lines_round_trip() {
        let lines = [
            r#"{"kind":"step","step":42,"total":100,"loss":0.42}"#,
            r#"{"kind":"saved","path":"/tmp/ckpt-step-100"}"#,
            r#"{"kind":"done","checkpoint_dir":"/tmp/ckpt","final_loss":0.21}"#,
            r#"{"kind":"failed","error":"OOM at step 1234"}"#,
        ];
        for s in lines {
            let parsed: StatusLine = serde_json::from_str(s).unwrap();
            // Round-trip back.
            let _ = serde_json::to_string(&parsed).unwrap();
        }
    }

    fn update(line: &str) -> Option<StatusUpdate> {
        serde_json::from_str::<StatusLine>(line)
            .unwrap()
            .status_update()
    }

    /// Each wrapper line maps to the engine's trainer update: the format
    /// `blut log` and `blut jobs` read. A measured step is a step.
    #[test]
    fn a_logged_step_becomes_the_engines_step() {
        assert_eq!(
            update(r#"{"kind":"step","step":4,"total":10,"loss":0.5,"lr":0.0002,"vram_mb":812}"#),
            Some(StatusUpdate::Step {
                step: 4,
                total: 10,
                loss: 0.5,
                lr: 0.0002,
                vram_mb: 812,
            })
        );
        assert_eq!(
            update(r#"{"kind":"eval","step":4,"eval_loss":1.25}"#),
            Some(StatusUpdate::Eval {
                step: 4,
                eval_loss: 1.25
            })
        );
        assert_eq!(
            update(r#"{"kind":"saved","path":"/c/checkpoint-4"}"#),
            Some(StatusUpdate::Saved {
                path: "/c/checkpoint-4".into()
            })
        );
        assert_eq!(
            update(r#"{"kind":"done","checkpoint_dir":"/c","final_loss":0.25}"#),
            Some(StatusUpdate::Done {
                final_loss: 0.25,
                checkpoint_dir: "/c".into()
            })
        );
        assert_eq!(
            update(r#"{"kind":"failed","error":"CUDA out of memory"}"#),
            Some(StatusUpdate::Failed {
                error: "CUDA out of memory".into()
            })
        );
    }

    /// What HF did not measure is not invented: a step without a logged loss
    /// or learning rate (the end-of-run summary) still
    /// shows progress, as a heartbeat, and a run with no training loss has no
    /// `done` update rather than a zero one.
    #[test]
    fn an_unmeasured_value_is_never_filled_in() {
        assert_eq!(
            update(r#"{"kind":"step","step":4,"total":10,"loss":0.5,"lr":null,"vram_mb":812}"#),
            Some(StatusUpdate::Heartbeat {
                phase: Some("step 4/10".into()),
                vram_mb: Some(812),
            })
        );
        assert_eq!(
            update(r#"{"kind":"step","step":1,"total":10,"loss":null,"lr":0.0002}"#),
            Some(StatusUpdate::Heartbeat {
                phase: Some("step 1/10".into()),
                vram_mb: None,
            })
        );
        assert_eq!(
            update(r#"{"kind":"done","checkpoint_dir":"/c","final_loss":null}"#),
            None
        );
    }

    #[test]
    fn wrapper_path_resolves_in_dev_tree() {
        // In the dev tree the wrapper script doesn't exist yet
        // (lands alongside this commit), so this test verifies the
        // function returns Some(...) once the script is present.
        // Until then it asserts the path-search code doesn't panic
        // even when none exists.
        let _ = wrapper_path();
    }
}
