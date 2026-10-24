//! HuggingFace checkpoint → GGUF → quantized GGUF.
//!
//! Wraps llama.cpp's `convert_hf_to_gguf.py` and `llama-quantize`
//! binaries (located via `blut::config::llama_tool`). Two-step
//! pipeline:
//!
//!   1. `python3 convert_hf_to_gguf.py <checkpoint_dir> --outfile <name.f16.gguf>`
//!   2. `llama-quantize <name.f16.gguf> <name.<quant>.gguf> <quant>`
//!
//! The intermediate f16 file is removed after a successful quantize
//! so disk pressure stays bounded — a 7B f16 is ~14 GB; the Q4_K_M
//! is ~4 GB. Failed conversions leave the f16 in place so the user
//! can inspect / re-run the quantize step manually.
//!
//! `quant == "f16"` skips the quantize step and returns the f16
//! file as the final artifact.

use std::path::{Path, PathBuf};

use blut::config::launcher::LaunchTarget;
use blut::error::{Result, TrainError};

/// Run one conversion tool to completion with inherited stdio, contained on
/// Windows (see [`crate::local_stage`]). The child dies with this future.
async fn run_tool(unit: &str, cmdline: &[String]) -> Result<std::process::ExitStatus> {
    let mut child = crate::local_stage::spawn(LaunchTarget::Local, unit, cmdline, None, &[], |c| {
        c.kill_on_drop(true);
    })
    .await?;
    child.wait().await
}

/// Run the HF → GGUF pipeline. Returns the final on-disk path
/// (quantized GGUF if `quant != "f16"`, otherwise the f16 GGUF).
///
/// The output files land next to `checkpoint_dir`'s parent — i.e.
/// alongside the HF checkpoint folder, not inside it. This keeps
/// the HF and GGUF artifacts at the same directory level so the
/// registry can reference one cleanly while leaving the other on
/// disk for re-quantization.
pub async fn convert_to_gguf(checkpoint_dir: &Path, name: &str, quant: &str) -> Result<PathBuf> {
    if !is_safe_filename(name) {
        return Err(TrainError::Convert(format!(
            "name '{name}' must be a non-empty bare filename \
             ([A-Za-z0-9_.-]+, no leading dot/dash, no '..', no path separators)"
        )));
    }
    if !checkpoint_dir.exists() {
        return Err(TrainError::Convert(format!(
            "checkpoint_dir does not exist: {}",
            checkpoint_dir.display()
        )));
    }
    let parent = checkpoint_dir.parent().ok_or_else(|| {
        TrainError::Convert(format!(
            "checkpoint_dir '{}' has no parent",
            checkpoint_dir.display()
        ))
    })?;
    let f16_path = parent.join(format!("{name}.f16.gguf"));

    let convert = blut::config::llama_tool("convert_hf_to_gguf.py")
        .map_err(|e| TrainError::Convert(format!("locate convert_hf_to_gguf.py: {e}")))?;

    let python = crate::local_stage::python();
    let convert_status = run_tool(
        "blut-convert_hf_to_gguf",
        &[
            python.clone(),
            convert.display().to_string(),
            checkpoint_dir.display().to_string(),
            "--outfile".to_string(),
            f16_path.display().to_string(),
        ],
    )
    .await
    .map_err(|e| TrainError::Convert(format!("spawn {python} {}: {e}", convert.display())))?;
    if !convert_status.success() {
        return Err(TrainError::Convert(format!(
            "convert_hf_to_gguf.py exited with {convert_status}"
        )));
    }
    if !f16_path.exists() {
        return Err(TrainError::Convert(format!(
            "convert succeeded but produced no file at {}",
            f16_path.display()
        )));
    }

    if quant == "f16" {
        return Ok(f16_path);
    }

    let quantize = blut::config::llama_tool("llama-quantize")
        .map_err(|e| TrainError::Convert(format!("locate llama-quantize: {e}")))?;
    let q_path = parent.join(format!("{name}.{quant}.gguf"));
    let q_status = run_tool(
        "blut-llama-quantize",
        &[
            quantize.display().to_string(),
            f16_path.display().to_string(),
            q_path.display().to_string(),
            quant.to_string(),
        ],
    )
    .await
    .map_err(|e| TrainError::Convert(format!("spawn {}: {e}", quantize.display())))?;
    if !q_status.success() {
        return Err(TrainError::Convert(format!(
            "llama-quantize exited with {q_status}; \
             intermediate f16 left at {} for manual retry",
            f16_path.display()
        )));
    }
    if !q_path.exists() {
        return Err(TrainError::Convert(format!(
            "quantize succeeded but produced no file at {}",
            q_path.display()
        )));
    }

    // Quant succeeded; reclaim disk by removing the f16 intermediate.
    if let Err(e) = std::fs::remove_file(&f16_path) {
        tracing::warn!(
            "failed to remove intermediate f16 {}: {} (artifact remains usable)",
            f16_path.display(),
            e
        );
    }

    Ok(q_path)
}

/// Bare filename safety check. Mirrors `spec::is_safe_registry_name`
/// shape so converted-model names stay registry-safe AND
/// filesystem-safe. Catches: empty, leading `.` / `-`, any path
/// separator (`/` or `\\`), `..` substring, NUL.
fn is_safe_filename(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && !name.starts_with('-')
        && !name.contains("..")
        && !name.contains('/')
        && !name.contains('\\')
        && !name.contains('\0')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_filename_accepts_normal_names() {
        for n in ["model", "test-7b", "qwen3.6", "personal_v2"] {
            assert!(is_safe_filename(n), "{n} should be safe");
        }
    }

    #[test]
    fn safe_filename_rejects_path_traversal() {
        for n in [
            "../etc/passwd",
            "..",
            "a/b",
            "a\\b",
            "a..b",
            ".hidden",
            "-leading",
            "",
            "name\0bad",
            "name with space",
        ] {
            assert!(!is_safe_filename(n), "{n} should be rejected");
        }
    }
}
