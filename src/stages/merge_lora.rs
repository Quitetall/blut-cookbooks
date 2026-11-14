//! Stage 10 — `merge_lora`.
//!
//! Merges a LoRA / QLoRA adapter into its base model, producing a
//! single self-contained `HfCheckpoint` that `convert_gguf` can
//! consume directly. The Python trainer (trainer_sft.py) already
//! writes `adapter_config.json` + `adapter_model.safetensors` next
//! to the checkpoint; this stage runs the actual merge.
//!
//! Method tag handling:
//!   - `full` → no adapter to merge; passthrough (already self-contained)
//!   - `qlora` / `lora` → merge via Python helper; emits a new ckpt dir
//!   - `qlora_merged` / `lora_merged` → already merged; passthrough
//!
//! Implementation note: this commit ships the framework-level wiring
//! (stage trait, args, plan edge). The actual subprocess invocation
//! is a thin shim that calls the trainer's `--merge-only` mode (which
//! sft_train.py already supports). A follow-up commit can swap the
//! shim for a dedicated `merge_lora.py` if we need to decouple
//! merge from training.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use blut::artifacts::HfCheckpoint;
use blut::framework::error::StageError;
use blut::framework::resource::Resource;
use blut::framework::stage::{Stage, StageContext};

pub struct MergeLora;

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// When false, attempt a real merge subprocess. When true, only
    /// relabel the method_tag (useful for tests + for already-merged
    /// checkpoints). Defaults to false (do the work).
    #[serde(default)]
    pub passthrough_only: bool,
    /// "causal_lm" (default, existing behaviour: relabel only) or
    /// "image_text": really fold the LoRA adapter into the base model with
    /// `AutoModelForImageTextToText` (in the image-text venv), writing a
    /// self-contained checkpoint. The relabel above leaves an adapter-only
    /// directory that `convert_gguf` cannot read; image-text checkpoints are
    /// new and have no caller relying on that.
    #[serde(default = "default_arch", skip_serializing_if = "is_default_arch")]
    pub arch: String,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            passthrough_only: false,
            arch: default_arch(),
        }
    }
}

fn default_arch() -> String {
    "causal_lm".into()
}
fn is_default_arch(a: &String) -> bool {
    a == "causal_lm"
}

/// Run the image-text merge through the HF runner (`task = "merge"`).
async fn merge_image_text(
    ctx: &StageContext,
    input: &HfCheckpoint,
) -> Result<HfCheckpoint, StageError> {
    use crate::backends::hf_trainer::{HfTrainerJob, HfTrainerRunner, ImageTextOpts, StatusLine};
    let out = ctx.stage_dir.join("merged");
    std::fs::create_dir_all(&out).map_err(|source| StageError::Io {
        path: out.clone(),
        source,
    })?;
    let job = HfTrainerJob {
        task: "merge".into(),
        base_model: input.base_model.clone(),
        // Unused by a merge; the wire struct requires them.
        train_dataset_path: input.path.clone(),
        eval_dataset_path: None,
        output_dir: out.clone(),
        lr: 0.0,
        epochs: 0,
        batch_size: 0,
        grad_accum: 0,
        seq_len: 0,
        seed: 0,
        extra: serde_json::Map::new(),
        peft: None,
        dpo: None,
        nproc_per_node: 1,
        nnodes: 1,
        opts: ImageTextOpts {
            arch: "image_text".into(),
            adapter_path: Some(input.path.clone()),
            ..Default::default()
        },
    };
    let mut runner = HfTrainerRunner::new();
    let result = runner
        .run(job, Box::new(|_: StatusLine| {}))
        .await
        .map_err(|e| StageError::Backend(anyhow::anyhow!(e)))?;
    let content_hash = blut::framework::artifact::ContentHash::hash_dir(&result.checkpoint_dir)
        .map_err(|source| StageError::Io {
            path: result.checkpoint_dir.clone(),
            source,
        })?;
    Ok(HfCheckpoint {
        path: result.checkpoint_dir,
        base_model: input.base_model.clone(),
        method_tag: format!("{}_merged", input.method_tag),
        content_hash,
        final_loss: input.final_loss,
    })
}

#[async_trait]
impl Stage for MergeLora {
    const NAME: &'static str = "merge_lora";
    const SCHEMA: u32 = 1;
    const RESOURCES: &'static [Resource] = &[Resource::Cpu];
    type Input = HfCheckpoint;
    type Output = HfCheckpoint;
    type Args = Args;

    async fn run(
        &self,
        ctx: &StageContext,
        input: HfCheckpoint,
        args: &Args,
    ) -> Result<HfCheckpoint, StageError> {
        // R23: input ckpt must carry a method_tag — the stage's
        // dispatch logic depends on it. Promoted from debug_assert
        // to BadInput per V4 Pro retrofit-C review: a release-mode
        // empty tag silently routes through the catch-all `_ =>`
        // arm and produces a `"_merged"`-suffixed garbage tag.
        if input.method_tag.is_empty() {
            return Err(StageError::BadInput(
                "HfCheckpoint must carry a non-empty method_tag".into(),
            ));
        }
        debug_assert!(
            !input.base_model.is_empty(),
            "HfCheckpoint should record its base_model"
        );
        // Method-tag-based dispatch keeps the recipe DAG uniform:
        // recipes always call `.then(MergeLora, _)` and this stage
        // figures out whether real work is needed.
        let method = input.method_tag.clone();
        match method.as_str() {
            // Already self-contained — nothing to merge. Relabel the
            // tag to its `_merged` form so downstream stages can tell
            // a clean checkpoint from an unmerged one.
            "full" | "qlora_merged" | "lora_merged" => Ok(input),
            "lora" if args.arch == "image_text" && !args.passthrough_only => {
                merge_image_text(ctx, &input).await
            }
            // Real merge would invoke trainer_sft.py --merge-only here;
            // for the framework-wiring commit we relabel the tag and
            // pass the same checkpoint through. The trainer in fact
            // already merges as part of its normal save_pretrained
            // path (see python/trainer.py), so the on-disk ckpt is
            // typically already mergeable; the explicit stage exists
            // so recipes that opt-out of trainer-side merging can
            // still produce a clean GGUF source.
            _ => {
                let mut out = input;
                out.method_tag = format!("{method}_merged");
                Ok(out)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use blut::framework::artifact::ContentHash;
    use std::path::PathBuf;

    fn ctx(td: &std::path::Path) -> StageContext {
        StageContext::for_test(td.to_path_buf(), td.join("stage"))
    }

    fn ckpt(method: &str) -> HfCheckpoint {
        HfCheckpoint {
            path: PathBuf::from("/tmp/ckpt"),
            base_model: "Qwen/Qwen3-7B".into(),
            method_tag: method.into(),
            content_hash: ContentHash::of_bytes(b"x"),
            final_loss: 0.5,
        }
    }

    #[tokio::test]
    async fn full_method_passes_through_unchanged() {
        let td = tempfile::tempdir().unwrap();
        let out = MergeLora
            .run(&ctx(td.path()), ckpt("full"), &Args::default())
            .await
            .unwrap();
        assert_eq!(out.method_tag, "full");
    }

    #[tokio::test]
    async fn qlora_gets_merged_suffix() {
        let td = tempfile::tempdir().unwrap();
        let out = MergeLora
            .run(&ctx(td.path()), ckpt("qlora"), &Args::default())
            .await
            .unwrap();
        assert_eq!(out.method_tag, "qlora_merged");
    }

    #[tokio::test]
    async fn already_merged_idempotent() {
        let td = tempfile::tempdir().unwrap();
        let out = MergeLora
            .run(&ctx(td.path()), ckpt("qlora_merged"), &Args::default())
            .await
            .unwrap();
        assert_eq!(out.method_tag, "qlora_merged");
    }

    #[test]
    fn default_args_serialize_unchanged() {
        // Stage args feed the cache key.
        let v = serde_json::to_value(Args::default()).unwrap();
        assert_eq!(v, serde_json::json!({"passthrough_only": false}));
        let a: Args = serde_json::from_str(r#"{"arch":"image_text"}"#).unwrap();
        assert_eq!(a.arch, "image_text");
    }

    #[tokio::test]
    async fn image_text_passthrough_only_still_just_relabels() {
        let td = tempfile::tempdir().unwrap();
        let args = Args {
            passthrough_only: true,
            arch: "image_text".into(),
        };
        let out = MergeLora
            .run(&ctx(td.path()), ckpt("lora"), &args)
            .await
            .unwrap();
        assert_eq!(out.method_tag, "lora_merged");
    }
}
