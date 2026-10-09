//! Stage 13 — `eval_loss`.
//!
//! Cross-entropy + perplexity over an eval split, evaluated against
//! a checkpoint. Input is `(HfCheckpoint, DatasetJsonl)` — the
//! checkpoint + the eval split. Output is an `EvalReport` with
//! `{loss, perplexity, n_examples}` at the metric root.
//!
//! Runs the `blut_core.evaluator` Python module with the `perplexity` eval
//! ingredient: text rows are packed into `max_seq`-token blocks with the
//! checkpoint's own tokenizer, exactly as training packs them, and the
//! model's next-token loss is averaged over them.
//!
//! Until this change the stage computed `loss = 0.5 + hash_byte / 512` from
//! the checkpoint's content hash and returned it as a measurement, marked
//! only by a `"synthetic": true` key that nothing downstream read.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use blut::artifacts::{DatasetJsonl, EvalReport, HfCheckpoint};
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

pub struct EvalLoss;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    #[serde(default = "default_max_seq")]
    pub max_seq: u32,
}
fn default_batch_size() -> u32 {
    1
}
fn default_max_seq() -> u32 {
    4096
}

#[async_trait]
impl Stage for EvalLoss {
    const NAME: &'static str = "eval_loss";
    // 2: measures instead of deriving a number from the checkpoint hash;
    // cached v1 reports are invalid.
    const SCHEMA: u32 = 2;
    const RESOURCES: &'static [Resource] = &[Resource::Gpu];
    type Input = (HfCheckpoint, DatasetJsonl);
    type Output = EvalReport;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: Self::Input,
        args: &Args,
    ) -> Result<EvalReport, StageError> {
        // R21 + R23: arg + input sanity.
        if input.1.n_examples <= 0 {
            return Err(StageError::BadInput(format!(
                "eval_loss: eval dataset has {} examples",
                input.1.n_examples
            )));
        }
        if args.batch_size == 0 || args.max_seq == 0 {
            return Err(StageError::BadInput(
                "batch_size + max_seq must be > 0".into(),
            ));
        }
        let (ckpt, ds) = input;
        let mut report = crate::evaluator::run_evaluator(
            ctx,
            &crate::evaluator::EvalRequest {
                checkpoint_path: &ckpt.path,
                dataset_path: &ds.path,
                eval: serde_json::json!({"kind": "eval", "name": "perplexity", "config": {}}),
                loss: None,
                batch_size: args.batch_size,
                device: "cuda",
                text_field: None,
                max_seq_len: Some(args.max_seq),
            },
            "eval_loss",
        )?;
        if let Some(metrics) = report.metrics.as_object_mut() {
            metrics.insert("n_examples".into(), ds.n_examples.into());
        }
        Ok(report)
    }
}

impl Default for Args {
    fn default() -> Self {
        Args {
            batch_size: default_batch_size(),
            max_seq: default_max_seq(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blut::framework::artifact::ContentHash;
    use std::path::PathBuf;

    fn ckpt(byte: u8) -> HfCheckpoint {
        HfCheckpoint {
            path: PathBuf::from("/tmp/ckpt"),
            base_model: "Qwen/Qwen3-7B".into(),
            method_tag: "qlora_merged".into(),
            content_hash: ContentHash([byte; 32]),
            final_loss: 0.4,
        }
    }
    fn ds() -> DatasetJsonl {
        DatasetJsonl {
            path: PathBuf::from("/tmp/eval.jsonl"),
            content_hash: ContentHash::of_bytes(b"eval"),
            n_examples: 50,
        }
    }
    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    /// It used to report a loss for a checkpoint that does not exist. A
    /// measurement needs something to measure.
    #[tokio::test]
    async fn refuses_a_checkpoint_that_does_not_exist() {
        let td = tempfile::tempdir().unwrap();
        let data = td.path().join("eval.jsonl");
        std::fs::write(&data, "{\"text\": \"hello\"}\n").unwrap();
        let mut d = ds();
        d.path = data;
        let mut c = ckpt(0);
        c.path = td.path().join("no-such-checkpoint");
        let r = EvalLoss
            .run(&ctx(td.path()), (c, d), &Args::default())
            .await;
        assert!(
            matches!(&r, Err(StageError::BadInput(m)) if m.contains("checkpoint not found")),
            "got {r:?}"
        );
    }

    #[tokio::test]
    async fn refuses_an_empty_eval_split() {
        let td = tempfile::tempdir().unwrap();
        let mut d = ds();
        d.n_examples = 0;
        let r = EvalLoss
            .run(&ctx(td.path()), (ckpt(0), d), &Args::default())
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
