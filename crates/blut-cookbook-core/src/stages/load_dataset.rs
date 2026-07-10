//! Stage — `load_dataset`.
//!
//! Graph-input stage that loads a dataset from a HuggingFace dataset name
//! or a local CSV/JSONL file path. Produces a typed `DatasetJsonl` artifact.
//!
//! Deterministic — same dataset path/name → identical output.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use blut::artifacts::DatasetJsonl;
use blut::framework::artifact::ContentHash;
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

pub struct LoadDataset;

#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Path to a local JSONL/CSV dataset file. Exclusive with `hf_name`.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// HuggingFace dataset name (e.g. "imdb", "squad"). Exclusive with `path`.
    #[serde(default)]
    pub hf_name: Option<String>,
    /// Dataset split to load (default: "train").
    #[serde(default = "default_split")]
    pub split: String,
    /// Optional subset/config name for HuggingFace datasets.
    #[serde(default)]
    pub subset: Option<String>,
    /// Maximum number of samples to load (None = all).
    #[serde(default)]
    pub max_samples: Option<usize>,
}

fn default_split() -> String {
    "train".to_string()
}

#[async_trait]
impl Stage for LoadDataset {
    const NAME: &'static str = "load_dataset";
    const SCHEMA: u32 = 1;
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
        match (&args.path, &args.hf_name) {
            (Some(p), None) => {
                // Local file path
                if p.components()
                    .any(|c| matches!(c, std::path::Component::ParentDir))
                {
                    return Err(StageError::BadInput(format!(
                        "path '{}' contains '..' — refusing",
                        p.display()
                    )));
                }
                if !p.exists() {
                    return Err(StageError::BadInput(format!(
                        "dataset path not found: {}",
                        p.display()
                    )));
                }
                let content_hash = ContentHash::hash_file(p).map_err(|source| StageError::Io {
                    path: p.clone(),
                    source,
                })?;
                let n_examples = count_jsonl_examples(p)?;
                Ok(DatasetJsonl {
                    path: p.clone(),
                    content_hash,
                    n_examples,
                })
            }
            (None, Some(hf_name)) => {
                // HuggingFace dataset — download and convert to JSONL
                let output_path = ctx.stage_dir.join("dataset.jsonl");
                let mut cmd = std::process::Command::new("python3");
                cmd.args([
                    "-m",
                    "blut_core.load_dataset",
                    "--name",
                    hf_name,
                    "--split",
                    &args.split,
                    "--output",
                    output_path.to_str().unwrap(),
                ]);
                if let Some(subset) = &args.subset {
                    cmd.args(["--subset", subset]);
                }
                if let Some(max) = args.max_samples {
                    cmd.args(["--max-samples", &max.to_string()]);
                }
                let status = cmd.status().map_err(|e| {
                    StageError::Backend(anyhow::anyhow!(
                        "failed to run blut_core.load_dataset: {}",
                        e
                    ))
                })?;
                if !status.success() {
                    return Err(StageError::Backend(anyhow::anyhow!(
                        "blut_core.load_dataset exited with {}",
                        status
                    )));
                }
                if !output_path.exists() {
                    return Err(StageError::Backend(anyhow::anyhow!(
                        "blut_core.load_dataset did not produce output"
                    )));
                }
                let content_hash =
                    ContentHash::hash_file(&output_path).map_err(|source| StageError::Io {
                        path: output_path.clone(),
                        source,
                    })?;
                let n_examples = count_jsonl_examples(&output_path)?;
                Ok(DatasetJsonl {
                    path: output_path,
                    content_hash,
                    n_examples,
                })
            }
            (Some(_), Some(_)) => Err(StageError::BadInput(
                "load_dataset: pass exactly one of path / hf_name, not both".into(),
            )),
            (None, None) => Err(StageError::BadInput(
                "load_dataset: one of path / hf_name is required".into(),
            )),
        }
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
        std::fs::write(&p, "{\"text\":\"hello\"}\n{\"text\":\"world\"}\n").unwrap();
        let out = LoadDataset
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    path: Some(p.clone()),
                    hf_name: None,
                    split: "train".into(),
                    subset: None,
                    max_samples: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(out.path, p);
        assert_eq!(out.n_examples, 2);
    }

    #[tokio::test]
    async fn rejects_missing_path() {
        let td = tempfile::tempdir().unwrap();
        let r = LoadDataset
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    path: Some(td.path().join("missing")),
                    hf_name: None,
                    split: "train".into(),
                    subset: None,
                    max_samples: None,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_both_path_and_name() {
        let td = tempfile::tempdir().unwrap();
        let r = LoadDataset
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    path: Some(td.path().join("x")),
                    hf_name: Some("imdb".into()),
                    split: "train".into(),
                    subset: None,
                    max_samples: None,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }

    #[tokio::test]
    async fn rejects_neither_path_nor_name() {
        let td = tempfile::tempdir().unwrap();
        let r = LoadDataset
            .run(
                &ctx(td.path()),
                (),
                &Args {
                    path: None,
                    hf_name: None,
                    split: "train".into(),
                    subset: None,
                    max_samples: None,
                },
            )
            .await;
        assert!(matches!(r, Err(StageError::BadInput(_))));
    }
}
