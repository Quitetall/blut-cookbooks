//! Local stage processes, contained wherever they run.
//!
//! Every stage in this crate and in `blut-cookbook-core` that runs a Python
//! or tool process on this machine spawns it through [`spawn`] (or [`run`]),
//! which hands a local run to the engine's `LocalSystemd::spawn_contained`.
//! The engine picks the backend and logs one line naming it:
//!
//! - **Linux with a systemd user manager:** a transient `systemd-run --user`
//!   service with the launcher's `MemoryMax`, the tree's peak reported at
//!   exit, and `systemctl --user stop` killing the whole tree.
//! - **Linux without one** (containers, CI, Kubernetes pods): a delegated
//!   cgroup v2 leaf where one is writable, otherwise `rlimit` with no cap. The
//!   stage still runs; it is never refused only because systemd is missing.
//! - **Windows:** a Job Object (commit cap, peak, tree kill).
//! - **macOS:** `rlimit`, with no cap.
//! - **`BLUT_NO_CONTAIN=1`:** `bare`, named as such in the log line.
//!
//! On Unix every stage process also leads its own process group, so where
//! the backend has no tree teardown of its own (`rlimit`, `bare`) cancelling
//! the stage reaches the whole tree through the group: [`stop_tree`] sends
//! SIGTERM to the group first, which lets `torchrun` stop its ranks and `srun`
//! cancel its step, then SIGKILL.
//!
//! A process placed on a cluster (`--launcher slurm`) is the scheduler's
//! client (`srun`), not the workload, so it is never put in a local
//! container: the cluster enforces the job's memory. It is spawned bare, as a
//! group leader, and stopped with the same SIGTERM-then-SIGKILL.
//!
//! A trainer's checkpoint marker carries the pid the trainer itself reports
//! (`os.getpid()`), so slot liveness checks see the trainer, not the
//! `systemd-run` client that is this process's direct child.

use std::path::Path;
use std::process::ExitStatus;
use std::time::Duration;

use blut::config::launcher::{LaunchTarget, LocalSystemd};
use blut::containment::{ContainedChild, Containment, bare::Bare};
use blut::framework::error::StageError;
use blut::framework::stage::StageContext;

/// Environment variable naming the Python interpreter for stage processes.
pub const PYTHON_ENV: &str = "BLUT_PYTHON";

/// How long a stage's process group gets between SIGTERM and SIGKILL when
/// it is stopped through the group.
pub const STOP_GRACE: Duration = Duration::from_secs(10);

/// The Python interpreter stage processes run: `$BLUT_PYTHON` when set, else
/// `python` on Windows (where `python3` is usually absent) and `python3`
/// elsewhere.
pub fn python() -> String {
    std::env::var(PYTHON_ENV)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default_python().to_string())
}

fn default_python() -> &'static str {
    if cfg!(windows) { "python" } else { "python3" }
}

/// Whether a stage process placed on `target` runs contained: every local
/// run. A cluster placement is the cluster's to contain.
pub fn contained(target: LaunchTarget) -> bool {
    target == LaunchTarget::Local
}

/// Spawn one stage process: `inner` is the program and its arguments, `cwd`
/// its working directory (this process's when `None`), `env` added to the
/// inherited environment. `configure` sets pipes and similar before the
/// spawn; it must not start a new session, which this function does. `unit`
/// names the run in logs (and in the unit name where a backend uses one).
///
/// The returned [`ContainedChild`] must be kept until the process exits:
/// [`wait_or_cancel`] waits and kills the whole tree on cancellation, and
/// dropping it early kills the tree too.
pub async fn spawn(
    target: LaunchTarget,
    unit: &str,
    inner: &[String],
    cwd: Option<&Path>,
    env: &[(String, String)],
    configure: impl FnOnce(&mut tokio::process::Command),
) -> blut::error::Result<ContainedChild> {
    let cwd = match cwd {
        Some(dir) => dir.to_path_buf(),
        None => std::env::current_dir().map_err(|source| blut::TrainError::Io {
            path: ".".into(),
            source,
        })?,
    };
    let configure = |c: &mut tokio::process::Command| {
        // Lead a process group, so the tree can be signalled as one.
        #[cfg(unix)]
        // SAFETY: setsid is async-signal-safe and allocates nothing; sound
        // between fork and exec.
        #[allow(unsafe_code)]
        unsafe {
            c.pre_exec(blut::python_kill::pre_exec_setsid);
        }
        configure(c);
    };
    if contained(target) {
        return LocalSystemd::default()
            .spawn_contained(unit, inner, &cwd, env, configure)
            .await;
    }
    let Some((program, args)) = inner.split_first() else {
        return Err(blut::TrainError::other(format!(
            "{unit}: empty stage command"
        )));
    };
    let run = Bare.wrap_command(
        Path::new(program),
        args,
        &cwd,
        env,
        unit,
        &Default::default(),
    )?;
    ContainedChild::spawn(Box::new(Bare), run, unit, configure).await
}

/// The OS error behind a failed [`spawn`], for callers whose error type
/// carries an `io::Error`: a failed `spawn()` keeps its kind (so a missing
/// interpreter is still `NotFound`); any other failure, such as the backend
/// refusing to contain the child, becomes `ErrorKind::Other` with its message.
pub fn spawn_error_to_io(error: blut::TrainError) -> std::io::Error {
    match error {
        blut::TrainError::Io { source, .. } => source,
        other => std::io::Error::other(other.to_string()),
    }
}

/// Kill the whole tree of a stage process and reap it.
///
/// A backend that tears the tree down itself (systemd, cgroup v2, Job
/// Object) does so. Otherwise, on Unix, the stage's process group gets
/// SIGTERM, up to [`STOP_GRACE`] for the group leader to exit, then SIGKILL:
/// `srun` cancels its step and `torchrun` stops its ranks on SIGTERM, and
/// anything that ignores it is killed.
pub async fn stop_tree(child: &mut ContainedChild) -> blut::error::Result<ExitStatus> {
    #[cfg(unix)]
    if !child.tears_down_tree()
        && let Some(pid) = child.id()
    {
        use nix::sys::signal::{Signal, killpg};
        use nix::unistd::Pid;
        let group = Pid::from_raw(pid as i32);
        let _ = killpg(group, Signal::SIGTERM);
        let _ = tokio::time::timeout(STOP_GRACE, child.child_mut().wait()).await;
        let _ = killpg(group, Signal::SIGKILL);
    }
    child.kill_tree().await
}

/// Wait for a stage process, or stop its whole tree ([`stop_tree`]) when
/// `cancel` fires first. `Ok(None)` means it was cancelled.
pub async fn wait_or_cancel(
    child: &mut ContainedChild,
    cancel: &tokio_util::sync::CancellationToken,
) -> blut::error::Result<Option<ExitStatus>> {
    let cancelled = tokio::select! {
        status = child.child_mut().wait() => {
            status.map_err(|e| blut::TrainError::other(format!("wait for {}: {e}", child.unit())))?;
            false
        }
        () = cancel.cancelled() => true,
    };
    if cancelled {
        tracing::warn!(
            "{}: cancelled; stopping its process tree ({})",
            child.unit(),
            child.kind()
        );
        stop_tree(child).await?;
        return Ok(None);
    }
    // Exited: this reaps nothing new, tears down what is left of the tree and
    // reads the peak.
    child.wait().await.map(Some)
}

/// Run one stage process to completion, contained per [`spawn`] and
/// cancellable: when `ctx.cancel` fires (a cancel, a soft stage timeout) the
/// whole tree is stopped and the stage fails with [`StageError::Cancelled`].
/// Dropping the stage future (a hard stage timeout) kills the tree too.
///
/// Stdout is inherited. Stderr is relayed line by line to this process's
/// stderr, which is how the systemd backend's `Memory peak:` line reaches the
/// engine's peak report.
pub async fn run(
    ctx: &StageContext,
    target: LaunchTarget,
    unit: &str,
    inner: &[String],
    cwd: Option<&Path>,
    env: &[(String, String)],
) -> Result<ExitStatus, StageError> {
    let mut child = spawn(target, unit, inner, cwd, env, |c| {
        c.stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
    })
    .await
    .map_err(|e| StageError::Backend(anyhow::anyhow!("{unit}: {e}")))?;
    child.pump_stderr(|line| eprintln!("{line}"));
    match wait_or_cancel(&mut child, &ctx.cancel)
        .await
        .map_err(|e| StageError::Backend(anyhow::anyhow!("{unit}: {e}")))?
    {
        Some(status) => Ok(status),
        None => Err(StageError::Cancelled),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_defaults_per_platform() {
        assert_eq!(
            default_python(),
            if cfg!(windows) { "python" } else { "python3" }
        );
    }

    #[test]
    fn every_local_run_is_contained_and_no_cluster_one() {
        assert!(contained(LaunchTarget::Local));
        assert!(!contained(LaunchTarget::Slurm));
        assert!(!contained(LaunchTarget::Ray));
    }

    #[tokio::test]
    async fn exit_status_and_output_pass_through() {
        let tmp = tempfile::tempdir().unwrap();
        let (prog, flag, script) = if cfg!(windows) {
            ("cmd", "/C", "echo hi& exit 3")
        } else {
            ("sh", "-c", "echo hi; exit 3")
        };
        let inner = [prog, flag, script].map(String::from);
        let mut child = spawn(
            LaunchTarget::Local,
            "local-stage-test",
            &inner,
            Some(tmp.path()),
            &[],
            |c| {
                c.stdout(std::process::Stdio::piped());
            },
        )
        .await
        .expect("spawn");
        let mut out = String::new();
        let mut pipe = child.child_mut().stdout.take().unwrap();
        tokio::io::AsyncReadExt::read_to_string(&mut pipe, &mut out)
            .await
            .unwrap();
        let status = child.wait().await.expect("wait");
        assert_eq!(status.code(), Some(3));
        assert_eq!(out.trim(), "hi");
        // A local run goes through the engine's selection, which names the
        // unit after the stage.
        assert!(child.unit().starts_with("local-stage-test"));
    }

    #[tokio::test]
    async fn a_slurm_placement_is_spawned_bare() {
        let tmp = tempfile::tempdir().unwrap();
        let (prog, flag, script) = if cfg!(windows) {
            ("cmd", "/C", "exit 0")
        } else {
            ("sh", "-c", "exit 0")
        };
        let inner = [prog, flag, script].map(String::from);
        let mut child = spawn(
            LaunchTarget::Slurm,
            "slurm-client",
            &inner,
            Some(tmp.path()),
            &[],
            |_| {},
        )
        .await
        .expect("spawn");
        assert_eq!(child.kind(), "bare");
        assert!(child.wait().await.expect("wait").success());
    }

    /// The fallback tree kill: a bare stage (a Slurm client here; also what
    /// `BLUT_NO_CONTAIN=1` gives a local one) whose child started a
    /// grandchild. Cancelling must take the grandchild too, through the
    /// process group.
    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_a_bare_stage_kills_its_grandchild() {
        let tmp = tempfile::tempdir().unwrap();
        let pidfile = tmp.path().join("grandchild.pid");
        let script = format!("sleep 600 & echo $! > {}; wait", pidfile.display());
        let inner = ["sh".to_string(), "-c".to_string(), script];
        let mut child = spawn(
            LaunchTarget::Slurm,
            "tree",
            &inner,
            Some(tmp.path()),
            &[],
            |c| {
                c.kill_on_drop(true);
            },
        )
        .await
        .expect("spawn");
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while std::fs::read_to_string(&pidfile).map_or(true, |s| s.trim().is_empty()) {
            assert!(
                std::time::Instant::now() < deadline,
                "grandchild never started"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let grandchild: i32 = std::fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let alive =
            |pid: i32| nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok();
        // Falsify the probe first.
        assert!(alive(grandchild), "grandchild {grandchild} was not running");

        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();
        let outcome =
            tokio::time::timeout(Duration::from_secs(60), wait_or_cancel(&mut child, &cancel))
                .await
                .expect("stopped within 60 s")
                .expect("wait_or_cancel");
        assert!(outcome.is_none(), "reported as cancelled");
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while alive(grandchild) {
            assert!(
                std::time::Instant::now() < deadline,
                "grandchild {grandchild} survived the cancel"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    #[tokio::test]
    async fn an_empty_command_is_an_error() {
        let err = spawn(LaunchTarget::Slurm, "empty", &[], None, &[], |_| {})
            .await
            .expect_err("empty");
        assert!(err.to_string().contains("empty stage command"), "{err}");
    }
}
