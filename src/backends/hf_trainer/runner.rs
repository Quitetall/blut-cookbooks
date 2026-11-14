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
//!     `{"kind": "step", "step": int, "total": int, "loss": f}`
//!     `{"kind": "saved", "path": str}`
//!     `{"kind": "done", "checkpoint_dir": str, "final_loss": f}`
//!     `{"kind": "failed", "error": str}`
//!   - Stderr: free-form. Captured to tracing::warn.
//!
//! The wrapper script lives at
//! `src/backends/hf_trainer/python/hf_trainer_runner.py` and is
//! invoked via the auto-managed venv at `~/.local/share/blut/hf-venv/`
//! (see `super::venv`).

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

use blut::python_kill::graceful_kill_pid;

use super::image_text::ImageTextOpts;
use super::venv::{IMAGE_TEXT_PROFILE, VenvError, ensure_venv, ensure_venv_profile};

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
    /// Image-text / streaming / resume options. Flattened: top-level keys on
    /// the wire, all omitted for a causal-LM job.
    #[serde(flatten)]
    pub opts: ImageTextOpts,
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
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StatusLine {
    Step {
        step: u64,
        total: u64,
        #[serde(default)]
        loss: Option<f32>,
        #[serde(default)]
        lr: Option<f32>,
        /// Pre-clip gradient norm (image_text path only).
        #[serde(default)]
        grad_norm: Option<f32>,
        #[serde(default)]
        tokens_per_second: Option<f32>,
    },
    /// Training continued from a checkpoint (image_text path only).
    Resumed {
        step: u64,
        #[serde(default)]
        path: Option<PathBuf>,
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
        // image_text jobs run in their own transformers-5 profile; every
        // other job keeps the original venv.
        let image_text = job.opts.is_image_text();
        let python = tokio::task::spawn_blocking(move || {
            if image_text {
                ensure_venv_profile(&IMAGE_TEXT_PROFILE)
            } else {
                ensure_venv()
            }
        })
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
        let prefix = crate::distributed::launch_prefix(job.nproc_per_node, job.nnodes)
            .map_err(RunError::Launch)?;
        let mut cmd = Command::new(&python);
        cmd.args(&prefix);
        cmd.arg(&wrapper).arg(&spec_path);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // KILL-1: new session/process group — see python_kill::pre_exec_setsid.
        #[cfg(unix)]
        {
            // tokio's Command exposes `pre_exec` inherently.
            // SAFETY: setsid is async-signal-safe and allocates nothing.
            #[allow(unsafe_code)]
            unsafe {
                cmd.pre_exec(blut::python_kill::pre_exec_setsid);
            }
        }

        let mut child = cmd.spawn().map_err(|source| RunError::Spawn {
            python: python.display().to_string(),
            source,
        })?;
        if let Some(pid) = child.id() {
            *self.child_pid.lock() = Some(pid);
            // KILL-2: publish for in-process + cross-process cancel.
            blut::python_kill::register_child(blut::python_kill::capture_identity(pid));
        }

        let started = Instant::now();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
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

        let stderr_handle = stderr.map(|stderr| {
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::warn!(target: "blut::hf_trainer_stderr", "{}", line);
                }
            })
        });

        let exit = child.wait().await.map_err(|source| RunError::Io {
            path: PathBuf::from("hf-trainer-wait"),
            source,
        })?;
        if let Some(h) = stdout_handle
            && let Err(e) = h.await
        {
            tracing::error!(target: "blut::hf_trainer", "stdout reader join failed: {e}");
        }
        if let Some(h) = stderr_handle
            && let Err(e) = h.await
        {
            tracing::error!(target: "blut::hf_trainer", "stderr reader join failed: {e}");
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
///   3. the source checkout this crate was compiled from
///      (`CARGO_MANIFEST_DIR`, baked in at build time). This is what finds
///      the script for a binary built from a git dependency, whose checkout
///      lives under `~/.cargo/git/checkouts`.
///   4. `<repo_root>/src/backends/hf_trainer/python/hf_trainer_runner.py`
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
    let built_from = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/backends/hf_trainer/python/hf_trainer_runner.py");
    if built_from.exists() {
        return Some(built_from);
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
            opts: ImageTextOpts::default(),
        };
        let s = serde_json::to_string(&job).unwrap();
        let back: HfTrainerJob = serde_json::from_str(&s).unwrap();
        assert_eq!(back.task, "sft");
        assert_eq!(back.peft.as_ref().unwrap().method, "qlora");
    }

    /// A causal-LM job's wire spec must not grow any key: the Python wrapper
    /// for existing recipes sees exactly what it saw before.
    #[test]
    fn causal_lm_job_wire_has_no_new_keys() {
        let job: HfTrainerJob = serde_json::from_value(serde_json::json!({
            "task": "sft", "base_model": "m", "train_dataset_path": "/t",
            "eval_dataset_path": null, "output_dir": "/o", "lr": 0.1, "epochs": 1,
            "batch_size": 1, "grad_accum": 1, "seq_len": 8, "seed": 1
        }))
        .unwrap();
        let v = serde_json::to_value(&job).unwrap();
        let mut keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "base_model",
                "batch_size",
                "dpo",
                "epochs",
                "eval_dataset_path",
                "extra",
                "grad_accum",
                "lr",
                "nnodes",
                "nproc_per_node",
                "output_dir",
                "peft",
                "seed",
                "seq_len",
                "task",
                "train_dataset_path"
            ]
        );
    }

    #[test]
    fn image_text_job_flattens_options_onto_the_wire() {
        let job: HfTrainerJob = serde_json::from_value(serde_json::json!({
            "task": "sft", "base_model": "m", "train_dataset_path": "/t",
            "eval_dataset_path": null, "output_dir": "/o", "lr": 0.1, "epochs": 1,
            "batch_size": 8, "grad_accum": 4, "seq_len": 512, "seed": 1,
            "arch": "image_text", "resume": true, "dataset_shards": ["/s0", "/s1"]
        }))
        .unwrap();
        assert!(job.opts.is_image_text() && job.opts.resume);
        let v = serde_json::to_value(&job).unwrap();
        assert_eq!(v["arch"], "image_text");
        assert_eq!(v["dataset_shards"][1], "/s1");
    }

    #[test]
    fn step_line_accepts_new_and_old_shapes() {
        let old: StatusLine =
            serde_json::from_str(r#"{"kind":"step","step":1,"total":2,"loss":0.5,"lr":1e-4}"#)
                .unwrap();
        assert!(matches!(
            old,
            StatusLine::Step {
                grad_norm: None,
                ..
            }
        ));
        let new: StatusLine = serde_json::from_str(
            r#"{"kind":"step","step":1,"total":2,"loss":0.5,"lr":1e-4,"grad_norm":0.7,"tokens_per_second":1234.5}"#,
        )
        .unwrap();
        assert!(matches!(
            new,
            StatusLine::Step {
                grad_norm: Some(_),
                ..
            }
        ));
        let r: StatusLine =
            serde_json::from_str(r#"{"kind":"resumed","step":500,"path":"/c/step-500"}"#).unwrap();
        assert!(matches!(r, StatusLine::Resumed { step: 500, .. }));
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

    #[test]
    fn wrapper_path_resolves_in_dev_tree() {
        // In the dev tree the wrapper script doesn't exist yet
        // (lands alongside this commit), so this test verifies the
        // function returns Some(...) once the script is present.
        // Until then it asserts the path-search code doesn't panic
        // even when none exists.
        let _ = wrapper_path();
    }

    /// End to end through the runner with a stub wrapper and a fake venv:
    /// an image_text job picks the image-text profile's interpreter (the
    /// original venv is pointed at a path that does not exist, so a wrong
    /// pick fails), and the new `resumed` and extended `step` lines parse.
    #[cfg(unix)]
    #[test]
    fn image_text_job_uses_its_own_venv_and_new_events_flow() {
        use std::os::unix::fs::PermissionsExt;
        let _guard = crate::TEST_ENV_LOCK.lock().expect("test env lock poisoned");
        let td = tempfile::tempdir().unwrap();
        let venv = td.path().join("it-venv");
        std::fs::create_dir_all(venv.join("bin")).unwrap();
        // A "python" that records its argv[0..] then runs the real one.
        let py = venv.join("bin/python");
        std::fs::write(
            &py,
            "#!/bin/sh\ntouch \"$(dirname \"$0\")/used\"\nexec python3 \"$@\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::write(
            venv.join(".blut_hf_venv_version"),
            super::super::venv::IMAGE_TEXT_VENV_VERSION,
        )
        .unwrap();
        let wrapper = td.path().join("stub.py");
        std::fs::write(
            &wrapper,
            r#"
import json, sys
spec = json.load(open(sys.argv[1]))
assert spec["arch"] == "image_text" and spec["resume"] is True, spec
for l in (
    {"kind": "resumed", "step": 500, "path": "/c/step-500"},
    {"kind": "step", "step": 501, "total": 900, "loss": 0.5, "lr": 1e-4,
     "grad_norm": 0.9, "tokens_per_second": 1234.0},
    {"kind": "saved", "path": "/c/step-1000"},
    {"kind": "done", "checkpoint_dir": spec["output_dir"], "final_loss": 0.25, "steps": 900},
):
    print(json.dumps(l), flush=True)
"#,
        )
        .unwrap();
        let prev: Vec<_> = [
            "BLUT_HF_VENV",
            "BLUT_HF_VENV_IMAGE_TEXT",
            "BLUT_HF_TRAINER_RUNNER",
        ]
        .iter()
        .map(|k| (*k, std::env::var(k).ok()))
        .collect();
        // SAFETY: serialized by TEST_ENV_LOCK.
        unsafe {
            std::env::set_var("BLUT_HF_VENV", td.path().join("no-such-venv"));
            std::env::set_var("BLUT_HF_VENV_IMAGE_TEXT", &venv);
            std::env::set_var("BLUT_HF_TRAINER_RUNNER", &wrapper);
        }
        let job: HfTrainerJob = serde_json::from_value(serde_json::json!({
            "task": "sft", "base_model": "m", "train_dataset_path": "/t",
            "eval_dataset_path": null, "output_dir": td.path().join("out"), "lr": 0.1,
            "epochs": 1, "batch_size": 1, "grad_accum": 1, "seq_len": 8, "seed": 1,
            "arch": "image_text", "resume": true
        }))
        .unwrap();
        let seen: Arc<Mutex<Vec<StatusLine>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let res = rt.block_on(async {
            HfTrainerRunner::new()
                .run(job, Box::new(move |s| sink.lock().push(s)))
                .await
        });
        unsafe {
            for (k, v) in prev {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
        let art = res.expect("run");
        assert!(
            venv.join("bin/used").exists(),
            "image-text interpreter not used"
        );
        assert_eq!(art.final_loss, Some(0.25));
        let seen = seen.lock();
        assert!(matches!(seen[0], StatusLine::Resumed { step: 500, .. }));
        assert!(matches!(
            seen[1],
            StatusLine::Step {
                grad_norm: Some(_),
                tokens_per_second: Some(_),
                ..
            }
        ));
    }
}
