//! Stage 14 — `eval_lm_harness`.
//!
//! Wraps EleutherAI lm-eval-harness as a subprocess. Input is a
//! checkpoint, output is an `EvalReport` keyed by task name.
//! Resource declaration is `[Gpu, Network]` because the harness
//! downloads benchmark datasets on first use.
//!
//! The harness subprocess is not wired yet. Until it is, this stage refuses
//! to run. It used to return a score derived from the checkpoint's content
//! hash — `0.3 + (hash_byte / 255 + i * 0.05) % 0.6` per task — shaped
//! exactly like a real result and marked only by a `"synthetic": true` key
//! that nothing downstream read. A benchmark number that was never measured
//! is worse than no number.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use blut::artifacts::{DatasetJsonl, EvalReport, HfCheckpoint};
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

pub struct EvalLmHarness;

/// Why this stage refuses to run. It used to fabricate a result instead.
const NOT_IMPLEMENTED: &str = "eval_lm_harness is not implemented: it would run EleutherAI lm-evaluation-harness, and that subprocess is not wired yet. It no longer returns placeholder scores. Run lm_eval against the checkpoint directly until it lands.";

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub tasks: Vec<String>,
    #[serde(default = "default_num_fewshot")]
    pub num_fewshot: u32,
}
fn default_num_fewshot() -> u32 {
    0
}

#[async_trait]
impl Stage for EvalLmHarness {
    const NAME: &'static str = "eval_lm_harness";
    // 2: stopped emitting synthetic scores; cached v1 reports are invalid.
    const SCHEMA: u32 = 2;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu, Resource::Network];
    // Tuple input: shares the (HfCheckpoint, DatasetJsonl) shape
    // with `eval_loss` + `eval_judge` so the `eval_suite` recipe
    // can fork3 from a single materialization. The dataset half
    // is currently unused by lm-eval-harness itself, but recipes
    // that want to feed a custom eval split (e.g. via the harness's
    // `--include_path` future flag) will have it available.
    type Input = (HfCheckpoint, DatasetJsonl);
    type Output = EvalReport;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: Self::Input,
        args: &Args,
    ) -> Result<EvalReport, StageError> {
        let (ckpt, _dataset) = input;
        // R21 + R23.
        debug_assert!(!ckpt.base_model.is_empty(), "ckpt base_model required");
        if args.tasks.is_empty() {
            return Err(StageError::BadInput(
                "eval_lm_harness: tasks list must be non-empty".into(),
            ));
        }
        Err(StageError::BadInput(NOT_IMPLEMENTED.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blut::framework::artifact::ContentHash;
    use std::path::PathBuf;

    fn ckpt() -> HfCheckpoint {
        HfCheckpoint {
            path: PathBuf::from("/tmp/ckpt"),
            base_model: "Qwen/Qwen3-7B".into(),
            method_tag: "qlora_merged".into(),
            content_hash: ContentHash::of_bytes(b"ckpt"),
            final_loss: 0.4,
        }
    }
    fn ds() -> DatasetJsonl {
        DatasetJsonl {
            path: PathBuf::from("/tmp/x.jsonl"),
            content_hash: ContentHash::of_bytes(b""),
            n_examples: 10,
        }
    }
    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn refuses_rather_than_inventing_scores() {
        let td = tempfile::tempdir().unwrap();
        let r = EvalLmHarness
            .run(
                &ctx(td.path()),
                (ckpt(), ds()),
                &Args {
                    tasks: vec!["hellaswag".into()],
                    num_fewshot: 5,
                },
            )
            .await;
        assert!(
            matches!(&r, Err(StageError::BadInput(m)) if m.contains("not implemented")),
            "got {r:?}"
        );
    }

    #[tokio::test]
    async fn rejects_empty_tasks() {
        let td = tempfile::tempdir().unwrap();
        let r = EvalLmHarness
            .run(
                &ctx(td.path()),
                (ckpt(), ds()),
                &Args {
                    tasks: vec![],
                    num_fewshot: 0,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
