//! Adapter stages — `to_checks_jsonl` and `from_checks_jsonl`.
//!
//! The engine's built-in guards, `check_jsonl` (row-count bounds) and `assert`
//! (a scalar predicate), take and return `blut::checks::JsonlArtifact`, kind
//! `checks.jsonl`, and a plan kind-checks every edge by exact kind. Every data
//! stage in these cookbooks emits `DatasetJsonl` (kind `dataset.jsonl`), and
//! every trainer, splitter and evaluator takes it, so no guard could sit on a
//! data path. These two stages carry the same file across that boundary and
//! back:
//!
//! ```text
//! load_dataset → to_checks_jsonl → check_jsonl → assert → from_checks_jsonl → train_model
//! ```
//!
//! A stage has one output type, so making the data stages emit
//! `JsonlArtifact` instead would have broken each of those typed edges, and
//! emitting both is not expressible. The adapters leave every existing edge as
//! it is and make a guard two extra nodes where a recipe wants one. Because
//! the guard sits on the path, a blocking breach fails the plan before the
//! consumer runs; a guard on a side branch would not stop it.
//!
//! Neither stage copies or rewrites the file. `to_checks_jsonl` hashes it,
//! as the engine's `JsonlArtifact::from_file` does; the checks pass it through
//! unchanged; `from_checks_jsonl` counts its rows the way `load_dataset` and
//! `check_jsonl` do (non-blank lines), so the count the guard enforced is the
//! `n_examples` the consumer sees. For a dataset `load_dataset` produced, the
//! `DatasetJsonl` that comes out equals the one that went in.

use async_trait::async_trait;

use blut::artifacts::DatasetJsonl;
use blut::checks::JsonlArtifact;
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext, StageExecutionBoundary};

/// `dataset.jsonl` → `checks.jsonl`: the same file, so `check_jsonl` and
/// `assert` can guard it.
pub struct ToChecksJsonl;

#[async_trait]
impl Stage for ToChecksJsonl {
    const NAME: &'static str = "to_checks_jsonl";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = DatasetJsonl;
    type Output = JsonlArtifact;
    type Args = ();

    async fn run(
        &self,
        _ctx: &StageContext,
        input: DatasetJsonl,
        _args: &(),
    ) -> Result<JsonlArtifact, StageError> {
        // Hashing reads the whole file: off the async runtime.
        let path = input.path.clone();
        tokio::task::spawn_blocking(move || JsonlArtifact::from_file(&path))
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!("to_checks_jsonl join: {e}")))?
            .map_err(|source| StageError::Io {
                path: input.path,
                source,
            })
    }
}

/// `checks.jsonl` → `dataset.jsonl`: the same file, with its row count, so a
/// guarded dataset reaches the stages that take a `DatasetJsonl`.
pub struct FromChecksJsonl;

#[async_trait]
impl Stage for FromChecksJsonl {
    const NAME: &'static str = "from_checks_jsonl";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    const EXECUTION_BOUNDARY: StageExecutionBoundary = StageExecutionBoundary::InProcess;
    type Input = JsonlArtifact;
    type Output = DatasetJsonl;
    type Args = ();

    async fn run(
        &self,
        _ctx: &StageContext,
        input: JsonlArtifact,
        _args: &(),
    ) -> Result<DatasetJsonl, StageError> {
        let path = input.path.clone();
        let n_examples = tokio::task::spawn_blocking(move || count_rows(&path))
            .await
            .map_err(|e| StageError::Backend(anyhow::anyhow!("from_checks_jsonl join: {e}")))?
            .map_err(|source| StageError::Io {
                path: input.path.clone(),
                source,
            })?;
        let n_examples = i64::try_from(n_examples).map_err(|_| {
            StageError::BadInput(format!(
                "{}: {n_examples} rows do not fit a dataset count",
                input.path.display()
            ))
        })?;
        Ok(DatasetJsonl {
            path: input.path,
            // The hash `to_checks_jsonl` took of these bytes, passed through
            // the checks unchanged.
            content_hash: input.content_hash,
            n_examples,
        })
    }
}

/// Non-blank lines: the rule `load_dataset` counts `n_examples` by and
/// `check_jsonl` enforces its bounds by. Streams, so a large file is never
/// held in memory.
fn count_rows(path: &std::path::Path) -> std::io::Result<u64> {
    use std::io::BufRead;
    let mut n = 0u64;
    for line in std::io::BufReader::new(std::fs::File::open(path)?).lines() {
        if !line?.trim().is_empty() {
            n += 1;
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use blut::framework::artifact::ContentHash;

    fn ctx(td: &std::path::Path) -> StageContext {
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    /// The round trip hands a consumer exactly the dataset it would have had
    /// without the guard: same path, same hash, same count. A blank line is
    /// not a row on either side.
    #[tokio::test]
    async fn round_trip_returns_the_dataset_unchanged() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("ds.jsonl");
        std::fs::write(
            &path,
            "{\"text\":\"a\"}\n\n{\"text\":\"b\"}\n  \n{\"text\":\"c\"}\n",
        )
        .unwrap();
        let dataset = DatasetJsonl {
            path: path.clone(),
            content_hash: ContentHash::hash_file(&path).unwrap(),
            n_examples: 3,
        };

        let guarded = ToChecksJsonl
            .run(&ctx(td.path()), dataset.clone(), &())
            .await
            .unwrap();
        assert_eq!(guarded.path, path);
        assert_eq!(guarded.content_hash, dataset.content_hash);

        let back = FromChecksJsonl
            .run(&ctx(td.path()), guarded, &())
            .await
            .unwrap();
        assert_eq!(back.path, dataset.path);
        assert_eq!(back.content_hash, dataset.content_hash);
        assert_eq!(back.n_examples, 3);
    }

    /// The hash is taken of the bytes on disk, not copied from the input: a
    /// guard must describe the file it guards.
    #[tokio::test]
    async fn to_checks_hashes_the_file_it_points_at() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("ds.jsonl");
        std::fs::write(&path, "{\"text\":\"a\"}\n").unwrap();
        let stale = DatasetJsonl {
            path: path.clone(),
            content_hash: ContentHash::of_bytes(b"not these bytes"),
            n_examples: 1,
        };
        let guarded = ToChecksJsonl
            .run(&ctx(td.path()), stale, &())
            .await
            .unwrap();
        assert_eq!(guarded.content_hash, ContentHash::hash_file(&path).unwrap());
    }

    #[tokio::test]
    async fn a_missing_file_is_an_io_error_naming_it() {
        let td = tempfile::tempdir().unwrap();
        let path = td.path().join("gone.jsonl");
        let r = ToChecksJsonl
            .run(
                &ctx(td.path()),
                DatasetJsonl {
                    path: path.clone(),
                    content_hash: ContentHash::of_bytes(b""),
                    n_examples: 0,
                },
                &(),
            )
            .await;
        assert!(matches!(r, Err(StageError::Io { path: p, .. }) if p == path));
        let r = FromChecksJsonl
            .run(
                &ctx(td.path()),
                JsonlArtifact {
                    path: path.clone(),
                    content_hash: ContentHash::of_bytes(b""),
                },
                &(),
            )
            .await;
        assert!(matches!(r, Err(StageError::Io { path: p, .. }) if p == path));
    }
}
