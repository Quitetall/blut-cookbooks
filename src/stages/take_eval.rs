//! Adapter stage — `take_eval`.
//!
//! Projects the eval half of a `DatasetSplit` into a bare `DatasetJsonl`: the
//! counterpart of `take_train`. A recipe forks the split into a training edge
//! and this edge, then merges the trained checkpoint with the held-out rows
//! for evaluation.
//!
//! Pure projection — no work, no IO. The returned artifact shares the eval
//! file's path and hash with the upstream split, so downstream cache keys are
//! stable across re-runs.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use blut::artifacts::{DatasetJsonl, DatasetSplit};
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

pub struct TakeEval;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {}

#[async_trait]
impl Stage for TakeEval {
    const NAME: &'static str = "take_eval";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = DatasetSplit;
    type Output = DatasetJsonl;
    type Args = Args;

    async fn run(
        &self,
        _ctx: &StageContext,
        input: DatasetSplit,
        _args: &Args,
    ) -> Result<DatasetJsonl, StageError> {
        if input.eval.n_examples <= 0 {
            return Err(StageError::BadInput(format!(
                "take_eval: eval half has {} examples",
                input.eval.n_examples
            )));
        }
        Ok(input.eval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blut::framework::artifact::ContentHash;
    use std::path::PathBuf;

    fn half(name: &str, n: i64) -> DatasetJsonl {
        DatasetJsonl {
            path: PathBuf::from(format!("/tmp/{name}.jsonl")),
            content_hash: ContentHash::of_bytes(name.as_bytes()),
            n_examples: n,
        }
    }

    fn ctx(td: &std::path::Path) -> StageContext {
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn projects_the_eval_half() {
        let td = tempfile::tempdir().unwrap();
        let split = DatasetSplit {
            train: half("train", 90),
            eval: half("eval", 10),
        };
        let out = TakeEval
            .run(&ctx(td.path()), split, &Args {})
            .await
            .unwrap();
        assert_eq!(out.path, PathBuf::from("/tmp/eval.jsonl"));
        assert_eq!(out.n_examples, 10);
    }

    #[tokio::test]
    async fn refuses_an_empty_eval_half() {
        let td = tempfile::tempdir().unwrap();
        let split = DatasetSplit {
            train: half("train", 90),
            eval: half("eval", 0),
        };
        let r = TakeEval.run(&ctx(td.path()), split, &Args {}).await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
