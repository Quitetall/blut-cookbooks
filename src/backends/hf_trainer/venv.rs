//! Auto-managed HF Trainer venv.
//!
//! BLUT's first-class HF Trainer backend ships with its own
//! Python environment so users don't have to chase
//! `transformers` / `trl` / `peft` / `bitsandbytes` version drift.
//! The venv lives at `~/.local/share/blut/hf-venv/` (override with
//! `BLUT_HF_VENV`) and is provisioned on first use via
//! `python -m venv` + `pip install`.
//!
//! Reentrancy: `ensure_venv()` is safe to call concurrently — if
//! the marker file exists with a matching version stamp, all
//! callers short-circuit. The first caller takes a sentinel lock
//! file; concurrent callers wait via filesystem polling.

use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// Provisioning schema version. The marker also includes the requirements-lock
/// digest, so changing the lock automatically invalidates existing venvs.
pub const VENV_VERSION: &str = "2";

/// A complete, hash-locked Linux/x86-64 resolution. The HF trainer is a GPU
/// server backend; unsupported platforms fail closed at pip's wheel check
/// instead of falling back to an unreviewed source build.
const REQUIREMENTS_LOCK: &str = include_str!("requirements-hf.lock");
const REQUIREMENTS_LOCK_FILE: &str = ".blut_hf_requirements.lock";
const LOCKED_PYTHON_MINOR: &str = "3.12";
const LOCKED_PLATFORM: &str = "linux-x86_64";

include!(concat!(env!("OUT_DIR"), "/hf_requirements.rs"));

/// Resolve the venv root. `$BLUT_HF_VENV` env wins; default
/// `~/.local/share/blut/hf-venv/`.
pub fn venv_root() -> PathBuf {
    if let Ok(p) = std::env::var("BLUT_HF_VENV") {
        return PathBuf::from(p);
    }
    let home = dirs::data_local_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    home.join("blut").join("hf-venv")
}

/// Path to the venv's Python interpreter. Caller-side; resolved
/// AFTER `ensure_venv()` returns.
pub fn python_path(venv_root: &Path) -> PathBuf {
    venv_root.join("bin").join("python")
}

/// Marker file storing `VENV_VERSION` after a successful build.
fn marker_path(venv_root: &Path) -> PathBuf {
    venv_root.join(".blut_hf_venv_version")
}

/// Mutex file held during provisioning. Concurrent callers see
/// this and poll the marker instead of racing pip.
fn lock_path(venv_root: &Path) -> PathBuf {
    let name = venv_root
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("hf-venv");
    venv_root.with_file_name(format!(".{name}.blut-provision.lock"))
}

fn expected_marker() -> String {
    let digest = Sha256::digest(REQUIREMENTS_LOCK.as_bytes());
    format!("{VENV_VERSION}:{LOCKED_PYTHON_MINOR}:{LOCKED_PLATFORM}:{digest:x}")
}

fn marker_matches(marker: &Path, py: &Path, expected: &str) -> bool {
    marker.exists()
        && py.exists()
        && std::fs::read_to_string(marker)
            .map(|contents| contents.trim() == expected)
            .unwrap_or(false)
}

#[derive(Debug, thiserror::Error)]
pub enum VenvError {
    #[error("Python 3.12 not on PATH; install it or set BLUT_HF_PYTHON to bootstrap {venv_root}")]
    PythonMissing { venv_root: PathBuf },
    #[error("HF trainer lock supports Python {expected}, but {executable} reports Python {actual}")]
    UnsupportedPython {
        executable: PathBuf,
        expected: &'static str,
        actual: String,
    },
    #[error("HF trainer lock supports {expected}, but this binary targets {actual}")]
    UnsupportedPlatform {
        expected: &'static str,
        actual: String,
    },
    #[error("failed to inspect Python at {executable}: {status}")]
    PythonProbeFailed { executable: PathBuf, status: String },
    #[error("`python -m venv` failed at {venv_root}: {status}")]
    VenvCreateFailed { venv_root: PathBuf, status: String },
    #[error("pip install failed: {status}")]
    PipFailed { status: String },
    #[error("io at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("timed out waiting for concurrent venv provisioning ({path})")]
    LockTimeout { path: PathBuf },
}

/// Ensure the HF venv exists with the current pinned dep set.
/// Returns the absolute path to the venv's Python interpreter.
///
/// Behavior:
///   1. Marker exists + matches `VENV_VERSION` → return python_path.
///   2. Lock file exists → poll for marker (up to 30 min for
///      first-time install — pip download of torch is slow).
///   3. Neither → take lock, run `python -m venv`, `pip install`,
///      write marker, release lock.
pub fn ensure_venv() -> Result<PathBuf, VenvError> {
    let actual_platform = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    if actual_platform != LOCKED_PLATFORM {
        return Err(VenvError::UnsupportedPlatform {
            expected: LOCKED_PLATFORM,
            actual: actual_platform,
        });
    }
    let root = venv_root();
    let marker = marker_path(&root);
    let py = python_path(&root);
    let expected = expected_marker();

    // Fast path: marker says we're good.
    if marker_matches(&marker, &py, &expected) {
        return Ok(py);
    }

    // Take the lock atomically. `create_new(true)` returns
    // AlreadyExists if another process beat us to it — at which
    // point we wait for the marker (with stale-lock recovery on
    // top of plain timeout). `create_new` is symlink-safe: the
    // syscall fails rather than following a pre-existing symlink
    // that would otherwise let `std::fs::write` overwrite an
    // attacker-targeted file.
    let lock = lock_path(&root);
    if let Some(parent) = lock.parent() {
        std::fs::create_dir_all(parent).map_err(|source| VenvError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let acquired_lock = match std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&lock)
    {
        Ok(mut f) => {
            use std::io::Write;
            let _ = write!(f, "{}", std::process::id());
            true
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => false,
        Err(source) => {
            return Err(VenvError::Io {
                path: lock.clone(),
                source,
            });
        }
    };
    if !acquired_lock {
        // Someone else is provisioning. If their PID is dead,
        // break the lock (recover from a crashed prior caller)
        // and retry once. Otherwise poll for the marker.
        if pid_in_lock_is_dead(&lock) {
            tracing::warn!(
                target: "blut::hf_venv",
                "stale lock at {} (holder PID dead); breaking and retrying",
                lock.display()
            );
            let _ = std::fs::remove_file(&lock);
            return ensure_venv();
        }
        wait_for_marker(&marker, &py, &expected, Duration::from_secs(30 * 60))?;
        return Ok(py);
    }

    let result = (|| -> Result<(), VenvError> {
        provision(&root)?;
        std::fs::write(&marker, &expected).map_err(|source| VenvError::Io {
            path: marker.clone(),
            source,
        })?;
        Ok(())
    })();

    // Release lock regardless of outcome.
    let _ = std::fs::remove_file(&lock);
    result?;
    Ok(py)
}

/// Read the PID stored inside the lock file and check whether
/// that process is still alive. Returns true if the lock can be
/// reasonably reclaimed (file missing, PID unreadable, or process
/// gone). On non-Unix targets always returns false — we don't
/// have a portable cross-platform pid-alive check, so we wait
/// the full timeout instead of breaking locks blind.
#[cfg(unix)]
fn pid_in_lock_is_dead(lock: &Path) -> bool {
    use nix::errno::Errno;
    use nix::sys::signal::kill;
    use nix::unistd::Pid;
    let body = match std::fs::read_to_string(lock) {
        Ok(b) => b,
        Err(_) => return true,
    };
    let pid: i32 = match body.trim().parse() {
        Ok(v) => v,
        Err(_) => return true,
    };
    // Signal 0: probe without delivering. Ok = alive,
    // ESRCH = gone, EPERM = alive-but-not-ours.
    match kill(Pid::from_raw(pid), None) {
        Ok(()) => false,
        Err(Errno::ESRCH) => true,
        Err(Errno::EPERM) => false,
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn pid_in_lock_is_dead(_lock: &Path) -> bool {
    false
}

fn wait_for_marker(
    marker: &Path,
    py: &Path,
    expected: &str,
    timeout: Duration,
) -> Result<(), VenvError> {
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if marker_matches(marker, py, expected) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    Err(VenvError::LockTimeout {
        path: marker.to_path_buf(),
    })
}

fn provision(root: &Path) -> Result<(), VenvError> {
    tracing::info!(target: "blut::hf_venv", "provisioning HF venv at {}", root.display());

    let python3 = if let Ok(path) = std::env::var("BLUT_HF_PYTHON") {
        PathBuf::from(path)
    } else {
        which::which("python3.12")
            .or_else(|_| which::which("python3"))
            .map_err(|_| VenvError::PythonMissing {
                venv_root: root.to_path_buf(),
            })?
    };
    let actual_python = python_minor(&python3)?;
    if actual_python != LOCKED_PYTHON_MINOR {
        return Err(VenvError::UnsupportedPython {
            executable: python3,
            expected: LOCKED_PYTHON_MINOR,
            actual: actual_python,
        });
    }

    let venv_status = Command::new(&python3)
        .arg("-m")
        .arg("venv")
        .arg("--clear")
        .arg(root)
        .status()
        .map_err(|source| VenvError::Io {
            path: root.to_path_buf(),
            source,
        })?;
    if !venv_status.success() {
        return Err(VenvError::VenvCreateFailed {
            venv_root: root.to_path_buf(),
            status: format!("{venv_status}"),
        });
    }

    let requirements = root.join(REQUIREMENTS_LOCK_FILE);
    std::fs::write(&requirements, REQUIREMENTS_LOCK).map_err(|source| VenvError::Io {
        path: requirements.clone(),
        source,
    })?;

    let python = python_path(root);
    let mut cmd = Command::new(&python);
    cmd.args(pip_install_args(&requirements));
    let status = cmd.status().map_err(|source| VenvError::Io {
        path: python,
        source,
    })?;
    if !status.success() {
        return Err(VenvError::PipFailed {
            status: format!("{status}"),
        });
    }

    tracing::info!(target: "blut::hf_venv", "HF venv ready at {}", root.display());
    Ok(())
}

fn python_minor(python: &Path) -> Result<String, VenvError> {
    let output = Command::new(python)
        .args([
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ])
        .output()
        .map_err(|source| VenvError::Io {
            path: python.to_path_buf(),
            source,
        })?;
    if !output.status.success() {
        return Err(VenvError::PythonProbeFailed {
            executable: python.to_path_buf(),
            status: format!("{}", output.status),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn pip_install_args(requirements: &Path) -> Vec<OsString> {
    [
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--only-binary=:all:",
        "--require-hashes",
        "--no-compile",
        "--requirement",
    ]
    .into_iter()
    .map(OsString::from)
    .chain(std::iter::once(requirements.as_os_str().to_owned()))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn venv_root_honors_env_override() {
        let _guard = crate::TEST_ENV_LOCK.lock().expect("test env lock poisoned");
        // SAFETY: tests are serialized via TEST_ENV_LOCK at the
        // crate root for env-mutating tests. This single-shot test
        // grabs + releases without checking other state.
        let prev = std::env::var("BLUT_HF_VENV").ok();
        // SAFETY: see #[allow] in lib.rs for env-test serialization.
        unsafe {
            std::env::set_var("BLUT_HF_VENV", "/tmp/blut-hf-venv-test");
        }
        assert_eq!(venv_root(), PathBuf::from("/tmp/blut-hf-venv-test"));
        match prev {
            Some(v) => unsafe { std::env::set_var("BLUT_HF_VENV", v) },
            None => unsafe { std::env::remove_var("BLUT_HF_VENV") },
        }
    }

    #[test]
    fn default_path_is_under_data_local_dir() {
        let _guard = crate::TEST_ENV_LOCK.lock().expect("test env lock poisoned");
        // Make sure we're not pointing at /tmp by default.
        let prev = std::env::var("BLUT_HF_VENV").ok();
        unsafe {
            std::env::remove_var("BLUT_HF_VENV");
        }
        let root = venv_root();
        assert!(root.ends_with("blut/hf-venv") || root.ends_with("blut\\hf-venv"));
        if let Some(value) = prev {
            unsafe { std::env::set_var("BLUT_HF_VENV", value) };
        }
    }

    #[test]
    fn python_path_layout_matches_unix_venv() {
        let root = PathBuf::from("/tmp/x");
        assert_eq!(python_path(&root), PathBuf::from("/tmp/x/bin/python"));
    }

    #[test]
    fn provisioning_lock_is_a_sibling_of_the_venv() {
        let root = PathBuf::from("/tmp/blut/hf-venv");
        assert_eq!(
            lock_path(&root),
            PathBuf::from("/tmp/blut/.hf-venv.blut-provision.lock")
        );
    }

    #[test]
    fn pip_install_is_noninteractive_binary_only_and_hash_locked() {
        let args = pip_install_args(Path::new("/tmp/requirements.lock"));
        let args: Vec<_> = args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            [
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--only-binary=:all:",
                "--require-hashes",
                "--no-compile",
                "--requirement",
                "/tmp/requirements.lock",
            ]
        );
    }

    #[test]
    fn marker_binds_schema_and_requirements_lock() {
        let marker = expected_marker();
        assert!(marker.starts_with("2:3.12:linux-x86_64:"));
        assert_eq!(marker.rsplit(':').next().map(str::len), Some(64));
    }

    #[test]
    fn requirements_are_exact_and_hash_locked() {
        assert!(REQUIREMENTS_LOCK.contains("torch=="));
        assert!(REQUIREMENTS_LOCK.contains("pip=="));
        assert!(REQUIREMENTS_LOCK.contains("--hash=sha256:"));
        assert!(!REQUIREMENTS_LOCK.contains(">="));
        for requirement in REQUIRED_PKGS {
            assert!(REQUIREMENTS_LOCK.contains(requirement));
        }
    }
}
