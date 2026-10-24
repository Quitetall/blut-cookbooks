//! Local stage processes, contained where the engine can contain them.
//!
//! Every stage in this crate and in `blut-cookbook-core` that runs a Python
//! or tool process on this machine spawns it through [`spawn`].
//!
//! - **Windows:** the process runs inside a Job Object, through the engine's
//!   `LocalSystemd::spawn_contained`. The launcher's `mem_max` caps the
//!   committed memory of the whole tree, the tree's peak is logged when it
//!   exits, and the tree is killed when the stage is cancelled, when a hard
//!   stage timeout drops the stage, or when the driver process dies (the job
//!   closes with it).
//! - **Elsewhere:** the process is spawned bare, as these stages always were
//!   on Unix. Callers that need to kill a whole tree there put the child in
//!   its own session and register it with `blut::python_kill`.
//!
//! A process placed on a cluster (`--launcher slurm`) is the scheduler's
//! client (`srun`), not the workload, so it is never contained here: the
//! cluster enforces the job's memory.

use std::path::Path;

use blut::config::launcher::{LaunchTarget, LocalSystemd};
use blut::containment::{ContainedChild, Containment, bare::Bare};
use blut::framework::error::StageError;
use blut::framework::stage::StageContext;

/// Environment variable naming the Python interpreter for stage processes.
pub const PYTHON_ENV: &str = "BLUT_PYTHON";

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

/// Whether a stage process placed on `target` runs contained. Only a local
/// run on Windows: see the module docs.
pub fn contained(target: LaunchTarget) -> bool {
    cfg!(windows) && target == LaunchTarget::Local
}

/// Spawn one stage process: `inner` is the program and its arguments, `cwd`
/// its working directory (this process's when `None`), `env` added to the
/// inherited environment. `configure` sets pipes and similar before the
/// spawn. `unit` names the run in logs (and is the unit name where a backend
/// uses one).
///
/// The returned [`ContainedChild`] must be kept until the process exits:
/// [`ContainedChild::wait_or_cancel`] waits and kills the tree on
/// cancellation, and on Windows dropping it kills the tree.
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

/// Run one stage process to completion with inherited stdio, as
/// `std::process::Command::status` did, but contained per [`spawn`] and
/// cancellable: when `ctx.cancel` fires the whole tree is killed and the
/// stage fails with [`StageError::Cancelled`]. The child is also killed if
/// the stage future is dropped (a hard stage timeout).
pub async fn run(
    ctx: &StageContext,
    target: LaunchTarget,
    unit: &str,
    inner: &[String],
    cwd: Option<&Path>,
    env: &[(String, String)],
) -> Result<std::process::ExitStatus, StageError> {
    let mut child = spawn(target, unit, inner, cwd, env, |c| {
        c.kill_on_drop(true);
    })
    .await
    .map_err(|e| StageError::Backend(anyhow::anyhow!("{unit}: {e}")))?;
    match child
        .wait_or_cancel(&ctx.cancel)
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
    fn only_a_local_windows_run_is_contained() {
        assert_eq!(contained(LaunchTarget::Local), cfg!(windows));
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
        assert_eq!(
            child.kind(),
            if cfg!(windows) {
                "windows-job-object"
            } else {
                "bare"
            }
        );
    }

    #[tokio::test]
    async fn an_empty_command_is_an_error() {
        let err = spawn(LaunchTarget::Slurm, "empty", &[], None, &[], |_| {})
            .await
            .expect_err("empty");
        assert!(err.to_string().contains("empty stage command"), "{err}");
    }
}
