//! Materializer — `materialize_dataset_path`.
//!
//! Graph-input stage that turns a path or registered dataset name
//! into a typed `DatasetJsonl`. The split between
//! `materialize_conversations` (pulls from lamu-mcp's conversation
//! db) and this stage isolates the source-of-truth choice from the
//! rest of the pipeline — recipe 1 wires the former, recipe 2 the
//! latter.
//!
//! Args carry exactly one of `path` or `registered_name`. Recipes
//! that have a path call this with `path` set. Recipes that ship a
//! dataset name pre-registered in `datasets_db` call this with
//! `registered_name` set; the stage resolves the name to its
//! recorded `source_path`. The selected bytes are copied into stage-owned storage;
//! count and content hash describe that retained copy, not a mutable source locator.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use blut::artifacts::DatasetJsonl;
use blut::datasets_db;
use blut::framework::artifact::ContentHash;
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

pub struct MaterializeDatasetPath;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Path to a JSONL dataset on disk. Exclusive with `registered_name`.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Registered dataset name in `datasets_db`. Exclusive with `path`.
    #[serde(default)]
    pub registered_name: Option<String>,
}

#[async_trait]
impl Stage for MaterializeDatasetPath {
    const NAME: &'static str = "materialize_dataset_path";
    // Invalidate cached locator-only outputs from behavior schema 1.
    const SCHEMA: u32 = 2;
    const RESOURCES: &'static [Resource] = &[Resource::Disk];
    type Input = ();
    type Output = DatasetJsonl;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        _input: (),
        args: &Args,
    ) -> Result<DatasetJsonl, StageError> {
        // R30: reject path traversal on the explicit-path branch.
        if let Some(p) = &args.path
            && p.components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(StageError::BadInput(format!(
                "path '{}' contains '..' — refusing",
                p.display()
            )));
        }
        let resolved_path = match (&args.path, &args.registered_name) {
            (Some(p), None) => p.clone(),
            (None, Some(name)) => {
                let conn =
                    datasets_db::open().map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;
                let rec = datasets_db::get_by_name(&conn, name)
                    .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?
                    .ok_or_else(|| {
                        StageError::BadInput(format!("registered dataset '{name}' not found"))
                    })?;
                rec.source_path
            }
            (Some(_), Some(_)) => {
                return Err(StageError::BadInput(
                    "materialize_dataset_path: pass exactly one of path / registered_name, not both"
                        .into(),
                ));
            }
            (None, None) => {
                return Err(StageError::BadInput(
                    "materialize_dataset_path: one of path / registered_name is required".into(),
                ));
            }
        };

        if !resolved_path.exists() {
            return Err(StageError::BadInput(format!(
                "dataset path not found: {}",
                resolved_path.display()
            )));
        }

        // The engine packages backing files from the stage's owned directory.
        // Returning the caller's locator both fails that ownership check and lets
        // later source edits change a supposedly materialized artifact.
        let output_path = ctx.stage_dir.join("dataset.jsonl");
        let mut input = std::fs::File::open(&resolved_path).map_err(|source| StageError::Io {
            path: resolved_path.clone(),
            source,
        })?;
        let mut pending =
            tempfile::NamedTempFile::new_in(&ctx.stage_dir).map_err(|source| StageError::Io {
                path: ctx.stage_dir.clone(),
                source,
            })?;
        std::io::copy(&mut input, pending.as_file_mut()).map_err(|source| StageError::Io {
            path: output_path.clone(),
            source,
        })?;
        let content_hash =
            ContentHash::hash_file(pending.path()).map_err(|source| StageError::Io {
                path: output_path.clone(),
                source,
            })?;
        let n_examples = count_jsonl_examples(pending.path())?;
        pending
            .as_file()
            .sync_all()
            .map_err(|source| StageError::Io {
                path: output_path.clone(),
                source,
            })?;
        // Atomic replacement does not follow an existing destination symlink.
        // Failed reads/validation leave any previous output intact.
        pending
            .persist(&output_path)
            .map_err(|error| StageError::Io {
                path: output_path.clone(),
                source: error.error,
            })?;

        Ok(DatasetJsonl {
            path: output_path,
            content_hash,
            n_examples,
        })
    }
}

fn count_jsonl_examples(path: &std::path::Path) -> Result<i64, StageError> {
    use std::io::BufRead;
    let f = std::fs::File::open(path).map_err(|source| StageError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut n = 0i64;
    for line in std::io::BufReader::new(f).lines() {
        let line = line.map_err(|source| StageError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if !line.trim().is_empty() {
            n += 1;
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(td: &std::path::Path) -> StageContext {
        std::fs::create_dir_all(td.join("stage")).unwrap();
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    #[tokio::test]
    async fn loads_dataset_from_path() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("ds.jsonl");
        std::fs::write(&p, "{\"a\":1}\n{\"b\":2}\n{\"c\":3}\n").unwrap();
        let out = MaterializeDatasetPath
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    path: Some(p.clone()),
                    registered_name: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(out.path, td.path().join("stage/dataset.jsonl"));
        assert_eq!(out.n_examples, 3);
        let original = std::fs::read(&p).unwrap();
        std::fs::remove_file(&p).unwrap();
        assert_eq!(std::fs::read(&out.path).unwrap(), original);
        assert_eq!(ContentHash::hash_file(&out.path).unwrap(), out.content_hash);
    }

    #[tokio::test]
    async fn unreadable_dataset_preserves_previous_output() {
        let td = tempfile::tempdir().unwrap();
        let context = ctx(td.path());
        let output = context.stage_dir.join("dataset.jsonl");
        std::fs::write(&output, "previous\n").unwrap();
        let source = td.path().join("invalid.jsonl");
        std::fs::write(&source, [0xff, b'\n']).unwrap();
        let result = MaterializeDatasetPath
            .run(
                &context,
                (),
                &Args {
                    path: Some(source),
                    registered_name: None,
                },
            )
            .await;
        assert!(result.is_err());
        assert_eq!(std::fs::read(&output).unwrap(), b"previous\n");
        assert_eq!(std::fs::read_dir(&context.stage_dir).unwrap().count(), 1);
    }

    #[tokio::test]
    async fn failed_publication_preserves_destination_and_removes_temporary_copy() {
        let td = tempfile::tempdir().unwrap();
        let context = ctx(td.path());
        let source = td.path().join("source.jsonl");
        std::fs::write(&source, "{}\n").unwrap();
        let destination = context.stage_dir.join("dataset.jsonl");
        std::fs::create_dir(&destination).unwrap();
        std::fs::write(destination.join("existing"), "retained").unwrap();
        let result = MaterializeDatasetPath
            .run(
                &context,
                (),
                &Args {
                    path: Some(source.clone()),
                    registered_name: None,
                },
            )
            .await;
        assert!(matches!(result, Err(StageError::Io { .. })));
        assert_eq!(
            std::fs::read(destination.join("existing")).unwrap(),
            b"retained"
        );
        assert_eq!(std::fs::read(&source).unwrap(), b"{}\n");
        assert_eq!(std::fs::read_dir(&context.stage_dir).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn output_symlink_does_not_redirect_materialization() {
        let td = tempfile::tempdir().unwrap();
        let context = ctx(td.path());
        let source = td.path().join("source.jsonl");
        let outside = td.path().join("untouched.jsonl");
        std::fs::write(&source, "{}\n").unwrap();
        std::fs::write(&outside, "outside\n").unwrap();
        std::os::unix::fs::symlink(&outside, context.stage_dir.join("dataset.jsonl")).unwrap();
        let output = MaterializeDatasetPath
            .run(
                &context,
                (),
                &Args {
                    path: Some(source.clone()),
                    registered_name: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(output.path, context.stage_dir.join("dataset.jsonl"));
        assert!(!output.path.is_symlink());
        assert_eq!(std::fs::read(&output.path).unwrap(), b"{}\n");
        assert_eq!(std::fs::read(&outside).unwrap(), b"outside\n");
        assert_eq!(std::fs::read(&source).unwrap(), b"{}\n");
    }

    #[tokio::test]
    async fn rejects_missing_path() {
        let td = tempfile::tempdir().unwrap();
        let r = MaterializeDatasetPath
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    path: Some(td.path().join("missing")),
                    registered_name: None,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_both_path_and_name() {
        let td = tempfile::tempdir().unwrap();
        let r = MaterializeDatasetPath
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    path: Some(td.path().join("x")),
                    registered_name: Some("y".into()),
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_neither_path_nor_name() {
        let td = tempfile::tempdir().unwrap();
        let r = MaterializeDatasetPath
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    path: None,
                    registered_name: None,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
