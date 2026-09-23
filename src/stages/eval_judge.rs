//! Stage 15 — `eval_judge`.
//!
//! Pair-of-prompts evaluation by a judge model. Generates a
//! response from `input.0` (the candidate checkpoint) on each
//! prompt in `args.prompts`, sends candidate + prompt to a judge
//! model via HTTP (not via the MCP server to keep this crate's
//! dependency tree shallow), and aggregates judge scores.
//!
//! Input is `(HfCheckpoint, DatasetJsonl)`. The dataset's first
//! `args.n_samples` lines are read as eval prompts when
//! `args.prompts` is empty — keeps recipes that already have an
//! eval split from needing to duplicate prompt strings.
//!
//! The judge call is not wired yet. Until it is, this stage refuses to run.
//! It used to return a mean score derived from the checkpoint's content
//! hash, shaped exactly like a real result and marked only by a
//! `"synthetic": true` key that nothing downstream read.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use blut::artifacts::{DatasetJsonl, EvalReport, HfCheckpoint};
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

pub struct EvalJudge;

/// Why this stage refuses to run. It used to fabricate a result instead.
const NOT_IMPLEMENTED: &str = "eval_judge is not implemented: the judge-model call is not wired yet. It no longer returns placeholder scores. Use eval_loss for a measured held-out loss and perplexity.";

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    pub judge_model: String,
    #[serde(default)]
    pub prompts: Vec<String>,
    #[serde(default = "default_n_samples")]
    pub n_samples: u32,
}
fn default_n_samples() -> u32 {
    20
}

#[async_trait]
impl Stage for EvalJudge {
    const NAME: &'static str = "eval_judge";
    // 2: stopped emitting synthetic scores; cached v1 reports are invalid.
    const SCHEMA: u32 = 2;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu, Resource::Network];
    type Input = (HfCheckpoint, DatasetJsonl);
    type Output = EvalReport;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: Self::Input,
        args: &Args,
    ) -> Result<EvalReport, StageError> {
        let (_ckpt, ds) = input;
        // R23: hard guards (release-mode too). Negative n_examples
        // would wrap on the `as u32` cast below; large n_samples
        // could overflow into negative via `as i64`. Reject both
        // up front rather than silently producing garbage `n`.
        if ds.n_examples < 0 {
            return Err(StageError::BadInput(format!(
                "eval_judge: dataset n_examples {} is negative",
                ds.n_examples
            )));
        }
        if args.judge_model.is_empty() {
            return Err(StageError::BadInput(
                "eval_judge: judge_model is empty".into(),
            ));
        }
        if args.n_samples == 0 && args.prompts.is_empty() {
            return Err(StageError::BadInput(
                "eval_judge: either n_samples > 0 or prompts must be non-empty".into(),
            ));
        }
        // After the ds.n_examples >= 0 + n_samples: u32 checks,
        // `args.n_samples as i64` is lossless and the min() result
        // is bounded by u32::MAX, so the `as u32` cast below is
        // well-defined.
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
    fn ds(n: i64) -> DatasetJsonl {
        DatasetJsonl {
            path: PathBuf::from("/tmp/x.jsonl"),
            content_hash: ContentHash::of_bytes(b""),
            n_examples: n,
        }
    }
    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn refuses_rather_than_inventing_scores() {
        let td = tempfile::tempdir().unwrap();
        let r = EvalJudge
            .run(
                &ctx(td.path()),
                (ckpt(), ds(100)),
                &Args {
                    judge_model: "a-judge".into(),
                    prompts: vec![],
                    n_samples: 10,
                },
            )
            .await;
        assert!(
            matches!(&r, Err(StageError::BadInput(m)) if m.contains("not implemented")),
            "got {r:?}"
        );
    }

    #[tokio::test]
    async fn rejects_empty_judge_model() {
        let td = tempfile::tempdir().unwrap();
        let r = EvalJudge
            .run(
                &ctx(td.path()),
                (ckpt(), ds(10)),
                &Args {
                    judge_model: "".into(),
                    prompts: vec![],
                    n_samples: 1,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
