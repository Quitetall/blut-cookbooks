//! Run the `blut_core.evaluator` Python module over a checkpoint and a
//! dataset, and read its report back as an `EvalReport`.
//!
//! Shared by the standard cookbook's `eval_loss` stage and the core
//! cookbook's evaluation stages, which each carried — or lacked — their own
//! copy of this subprocess plumbing.

use std::path::Path;

use blut::artifacts::EvalReport;
use blut::framework::artifact::ContentHash;
use blut::framework::error::StageError;
use blut::framework::stage::StageContext;

/// One evaluation: which checkpoint, on which rows, measured how.
#[derive(Clone, Debug)]
pub struct EvalRequest<'a> {
    /// Directory the trainer wrote: `hf/` (a HuggingFace model) and `model.pt`.
    pub checkpoint_path: &'a Path,
    /// JSONL rows to evaluate on.
    pub dataset_path: &'a Path,
    /// The eval ingredient, as `{"kind": "eval", "name": ..., "config": {...}}`.
    pub eval: serde_json::Value,
    /// The loss ingredient training used, for eval ingredients that score with
    /// it (`loss_eval`). `None` lets the evaluator pick the model's own loss.
    pub loss: Option<serde_json::Value>,
    pub batch_size: u32,
    /// `"cuda"` or `"cpu"`. The evaluator runs on the CPU when CUDA is absent.
    pub device: &'a str,
    /// Causal-LM packing, as for training. `None` uses the evaluator's
    /// defaults (`"text"`, 512).
    pub text_field: Option<&'a str>,
    pub max_seq_len: Option<u32>,
}

/// Run the evaluator in `ctx.stage_dir` and return its report.
///
/// `evaluator` names the report's producer (the `EvalReport::evaluator`
/// field). Every failure is a stage error: a report is only returned when the
/// evaluator exited cleanly and wrote one.
pub fn run_evaluator(
    ctx: &StageContext,
    req: &EvalRequest<'_>,
    evaluator: &str,
) -> Result<EvalReport, StageError> {
    if !req.checkpoint_path.exists() {
        return Err(StageError::BadInput(format!(
            "checkpoint not found: {}",
            req.checkpoint_path.display()
        )));
    }
    if !req.dataset_path.is_file() {
        return Err(StageError::BadInput(format!(
            "dataset not found: {}",
            req.dataset_path.display()
        )));
    }
    let config = serde_json::json!({
        "checkpoint_path": req.checkpoint_path,
        "dataset_path": req.dataset_path,
        "eval": req.eval,
        "loss": req.loss,
        "batch_size": req.batch_size,
        "device": req.device,
        "text_field": req.text_field,
        "max_seq_len": req.max_seq_len,
    });
    let config_path = ctx.stage_dir.join("eval_config.json");
    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|error| StageError::Backend(anyhow::anyhow!("serialize eval config: {error}")))?;
    std::fs::write(&config_path, config_json).map_err(|source| StageError::Io {
        path: config_path.clone(),
        source,
    })?;

    let output_path = ctx.stage_dir.join("eval_report.json");
    let status = std::process::Command::new("python3")
        .args(["-m", "blut_core.evaluator", "--config"])
        .arg(&config_path)
        .arg("--output")
        .arg(&output_path)
        .current_dir(&ctx.stage_dir)
        .status()
        .map_err(|error| {
            StageError::Backend(anyhow::anyhow!(
                "failed to run blut_core.evaluator: {error}"
            ))
        })?;
    if !status.success() {
        return Err(StageError::Backend(anyhow::anyhow!(
            "blut_core.evaluator exited with {status}"
        )));
    }

    let report_json = std::fs::read_to_string(&output_path).map_err(|source| StageError::Io {
        path: output_path.clone(),
        source,
    })?;
    let metrics = serde_json::from_str(&report_json).map_err(|error| {
        StageError::Backend(anyhow::anyhow!("failed to parse eval report: {error}"))
    })?;
    let content_hash = ContentHash::hash_file(&output_path).map_err(|source| StageError::Io {
        path: output_path.clone(),
        source,
    })?;
    Ok(EvalReport {
        path: output_path,
        evaluator: evaluator.into(),
        metrics,
        content_hash,
    })
}
