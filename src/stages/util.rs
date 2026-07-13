//! Shared stage utilities.
//!
//! Functions used by multiple stages live here so cross-module
//! `super::` calls don't have to reach into a specific stage's
//! private surface.

/// Pretty-print a JSON value to the given path. Creates the parent
/// directory if missing. Used by every eval stage + `merge_reports`.
pub(crate) fn write_report(
    path: &std::path::Path,
    metrics: &serde_json::Value,
) -> Result<(), blut::framework::error::StageError> {
    use blut::framework::error::StageError;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| StageError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let body = serde_json::to_vec_pretty(metrics)
        .map_err(|e| StageError::Backend(anyhow::anyhow!("serialize eval report: {e}")))?;
    std::fs::write(path, body).map_err(|source| StageError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Resolve a trainer owned by this cookbook.
///
/// Public installs use the `blut_standard` Python package. Source checkouts
/// use the in-tree module. Legacy engine resolution remains last for existing
/// deployments and explicit overrides.
pub(crate) fn resolve_trainer_script(
    python: &std::path::Path,
    filename: &str,
) -> Result<std::path::PathBuf, blut::framework::error::StageError> {
    use blut::framework::error::StageError;

    let (module, override_name) = match filename {
        "trainer.py" => ("blut_standard.trainer", "BLUT_STANDARD_TRAINER_PY"),
        "trainer_dpo.py" => ("blut_standard.trainer_dpo", "BLUT_STANDARD_TRAINER_DPO_PY"),
        "trainer_distill.py" => (
            "blut_standard.trainer_distill",
            "BLUT_STANDARD_TRAINER_DISTILL_PY",
        ),
        other => {
            return Err(StageError::BadInput(format!(
                "unsupported standard trainer script: {other}"
            )));
        }
    };

    if let Ok(path) = std::env::var(override_name) {
        let path = std::path::PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(StageError::BadInput(format!(
            "${override_name} does not name a file: {}",
            path.display()
        )));
    }

    let source_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("blut_standard")
        .join(filename);
    if source_path.is_file() {
        return Ok(source_path);
    }

    let snippet = format!(
        "import importlib.util; s=importlib.util.find_spec({module:?}); print(s.origin if s and s.origin else '')"
    );
    // Spawn errors fall through deliberately: legacy resolution below returns
    // the existing actionable interpreter/script error contract.
    if let Ok(output) = std::process::Command::new(python)
        .args(["-c", &snippet])
        .output()
        && output.status.success()
    {
        let path = std::path::PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
        if path.is_file() {
            return Ok(path);
        }
    }

    let legacy = if filename == "trainer.py" {
        blut::paths::resolve_trainer_script()
    } else {
        blut::paths::resolve_trainer_script_named(filename)
    };
    legacy.map_err(|error| {
        StageError::Backend(anyhow::anyhow!(
            "{module} not installed for {}; install blut-cookbook-standard into that interpreter or set ${override_name}: {error}",
            python.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::{resolve_trainer_script, write_report};

    #[test]
    fn write_report_creates_parent_and_writes_pretty_json() {
        let dir = std::env::temp_dir().join(format!("blut_util_test_{}", std::process::id()));
        let path = dir.join("nested").join("report.json");
        let _ = std::fs::remove_dir_all(&dir);
        let metrics = serde_json::json!({"r": 0.5, "prd": 94.4});
        write_report(&path, &metrics).expect("write_report should succeed");
        assert!(path.exists(), "report not written");
        let back: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(back, metrics, "round-trip mismatch");
        // pretty-printed → multi-line
        assert!(std::fs::read_to_string(&path).unwrap().contains('\n'));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn source_trainer_resolves_from_cookbook_owner() {
        let path = resolve_trainer_script(std::path::Path::new("python3"), "trainer.py")
            .expect("source trainer");
        assert!(path.ends_with("python/blut_standard/trainer.py"));
    }

    #[test]
    fn unknown_trainer_is_rejected() {
        let result = resolve_trainer_script(std::path::Path::new("python3"), "other.py");
        assert!(matches!(
            result,
            Err(blut::framework::error::StageError::BadInput(_))
        ));
    }
}
