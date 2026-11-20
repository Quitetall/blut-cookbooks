//! Optional job fields for image-text fine-tuning (`arch = "image_text"`) and
//! for the streaming / checkpoint / resume machinery that path uses.
//!
//! Every field is optional and skipped on serialization at its default, so a
//! causal-LM job (the only kind that existed before) serializes to the same
//! bytes as it always did: stage arguments feed the cache key, and the wire
//! spec the Python wrapper reads is unchanged for existing recipes.
//!
//! The struct is `#[serde(flatten)]`ed into both the stage `Args` and the
//! wire `HfTrainerJob`, so on the wire these are ordinary top-level keys.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const ARCH_CAUSAL_LM: &str = "causal_lm";
pub const ARCH_IMAGE_TEXT: &str = "image_text";

fn default_arch() -> String {
    ARCH_CAUSAL_LM.into()
}
fn is_default_arch(a: &String) -> bool {
    a == ARCH_CAUSAL_LM
}
fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ImageTextOpts {
    /// "causal_lm" (default; `AutoModelForCausalLM`, the original path) or
    /// "image_text" (`AutoModelForImageTextToText` + `AutoProcessor`, loss on
    /// the completion only, runs in the separate transformers-5 venv).
    #[serde(default = "default_arch", skip_serializing_if = "is_default_arch")]
    pub arch: String,
    /// Regex (full match on module names) choosing the LoRA targets. Default
    /// per model_type; SmolVLM/Idefics3: text-decoder q_proj and v_proj only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_target_regex: Option<String>,
    /// "cosine" (default) | "linear" | "constant_with_warmup".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lr_scheduler: Option<String>,
    /// Fraction of total steps spent warming up. Default 0.03.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup_ratio: Option<f64>,
    /// Gradient-norm clip. Default 1.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_grad_norm: Option<f64>,
    /// Checkpoint every N optimizer steps. Default 500; 0 = only at the end.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_steps: Option<u32>,
    /// Checkpoints kept (newest). Default 3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_checkpoints: Option<u32>,
    /// Continue from the newest checkpoint if one exists (else start fresh).
    #[serde(default, skip_serializing_if = "is_false")]
    pub resume: bool,
    /// Ordered JSONL shards, streamed in this order. Empty = the single
    /// `train_dataset_path`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dataset_shards: Vec<PathBuf>,
    /// Shuffle-buffer rows held in memory. Default 10000; 1 = no shuffle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shuffle_buffer: Option<u32>,
    /// Stop after this many optimizer steps (default: epochs x rows / batch).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<u64>,
    /// Emit a `step` event every N steps. Default 10.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logging_steps: Option<u32>,
    /// sha256 of `<base_model>/model.safetensors`; refuse to train on a
    /// mismatch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_sha256: Option<String>,
    /// Where checkpoints go. Default `<output_dir>/checkpoints`. Point it at a
    /// stable directory when a restarted stage gets a fresh `output_dir`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_root: Option<PathBuf>,
    /// Reload the saved adapter on a fresh base and require identical logits.
    /// Default true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify_reload: Option<bool>,
    /// `task = "merge"` only: the adapter directory to fold into the base.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_path: Option<PathBuf>,
}

impl Default for ImageTextOpts {
    fn default() -> Self {
        Self {
            arch: default_arch(),
            lora_target_regex: None,
            lr_scheduler: None,
            warmup_ratio: None,
            max_grad_norm: None,
            save_steps: None,
            keep_checkpoints: None,
            resume: false,
            dataset_shards: Vec::new(),
            shuffle_buffer: None,
            max_steps: None,
            logging_steps: None,
            base_sha256: None,
            checkpoint_root: None,
            verify_reload: None,
            adapter_path: None,
        }
    }
}

impl ImageTextOpts {
    pub fn is_image_text(&self) -> bool {
        self.arch == ARCH_IMAGE_TEXT
    }

    /// Checks that depend only on these fields plus the training method and
    /// `extra`. Returns a message suitable for `StageError::BadInput`.
    pub fn validate(
        &self,
        method: &str,
        extra: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), String> {
        if self.arch != ARCH_CAUSAL_LM && self.arch != ARCH_IMAGE_TEXT {
            return Err(format!("arch '{}' must be causal_lm|image_text", self.arch));
        }
        if !self.is_image_text() {
            // The new fields mean nothing to the original path; refuse
            // rather than silently ignore a setting the user expects to act.
            let set = self.set_field_names();
            if !set.is_empty() {
                return Err(format!(
                    "{} only apply to arch = \"image_text\"",
                    set.join(", ")
                ));
            }
            return Ok(());
        }
        if method != "lora" {
            return Err(format!(
                "arch image_text supports method 'lora' only; got '{method}'"
            ));
        }
        if !extra.is_empty() {
            return Err("arch image_text does not take `extra`; use the typed fields".into());
        }
        if let Some(s) = &self.lr_scheduler
            && !matches!(s.as_str(), "cosine" | "linear" | "constant_with_warmup")
        {
            return Err(format!(
                "lr_scheduler '{s}' must be cosine|linear|constant_with_warmup"
            ));
        }
        if let Some(w) = self.warmup_ratio
            && !(0.0..1.0).contains(&w)
        {
            return Err(format!("warmup_ratio must be in [0, 1); got {w}"));
        }
        if let Some(g) = self.max_grad_norm
            && !(g > 0.0 && g.is_finite())
        {
            return Err(format!("max_grad_norm must be positive finite; got {g}"));
        }
        if self.shuffle_buffer == Some(0) {
            return Err("shuffle_buffer must be >= 1 (1 = no shuffle)".into());
        }
        if self.max_steps == Some(0) || self.logging_steps == Some(0) {
            return Err("max_steps and logging_steps must be > 0".into());
        }
        if self.keep_checkpoints == Some(0) {
            return Err("keep_checkpoints must be >= 1".into());
        }
        if let Some(re) = &self.lora_target_regex
            && re.is_empty()
        {
            return Err("lora_target_regex must be non-empty when set".into());
        }
        Ok(())
    }

    fn set_field_names(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        let mut add = |on: bool, n: &'static str| {
            if on {
                v.push(n)
            }
        };
        add(self.lora_target_regex.is_some(), "lora_target_regex");
        add(self.lr_scheduler.is_some(), "lr_scheduler");
        add(self.warmup_ratio.is_some(), "warmup_ratio");
        add(self.max_grad_norm.is_some(), "max_grad_norm");
        add(self.save_steps.is_some(), "save_steps");
        add(self.keep_checkpoints.is_some(), "keep_checkpoints");
        add(self.resume, "resume");
        add(!self.dataset_shards.is_empty(), "dataset_shards");
        add(self.shuffle_buffer.is_some(), "shuffle_buffer");
        add(self.max_steps.is_some(), "max_steps");
        add(self.logging_steps.is_some(), "logging_steps");
        add(self.base_sha256.is_some(), "base_sha256");
        add(self.checkpoint_root.is_some(), "checkpoint_root");
        add(self.verify_reload.is_some(), "verify_reload");
        add(self.adapter_path.is_some(), "adapter_path");
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn none() -> serde_json::Map<String, serde_json::Value> {
        serde_json::Map::new()
    }

    #[test]
    fn defaults_serialize_to_nothing() {
        let v = serde_json::to_value(ImageTextOpts::default()).unwrap();
        assert_eq!(v, json!({}));
    }

    #[test]
    fn missing_fields_deserialize_to_defaults() {
        let o: ImageTextOpts = serde_json::from_value(json!({})).unwrap();
        assert_eq!(o, ImageTextOpts::default());
        assert!(!o.is_image_text());
    }

    #[test]
    fn all_fields_round_trip() {
        let v = json!({
            "arch": "image_text", "lora_target_regex": "a.*", "lr_scheduler": "linear",
            "warmup_ratio": 0.05, "max_grad_norm": 0.5, "save_steps": 100,
            "keep_checkpoints": 2, "resume": true, "dataset_shards": ["/a", "/b"],
            "shuffle_buffer": 64, "max_steps": 9, "logging_steps": 1,
            "base_sha256": "ab", "checkpoint_root": "/c", "verify_reload": false,
            "adapter_path": "/d"
        });
        let o: ImageTextOpts = serde_json::from_value(v.clone()).unwrap();
        assert!(o.is_image_text() && o.resume);
        assert_eq!(o.dataset_shards.len(), 2);
        assert_eq!(serde_json::to_value(&o).unwrap(), v);
    }

    #[test]
    fn causal_lm_refuses_image_text_only_fields() {
        let o = ImageTextOpts {
            resume: true,
            save_steps: Some(10),
            ..Default::default()
        };
        let e = o.validate("lora", &none()).unwrap_err();
        assert!(e.contains("resume") && e.contains("save_steps"), "{e}");
        assert!(ImageTextOpts::default().validate("qlora", &none()).is_ok());
    }

    #[test]
    fn image_text_requires_lora_and_no_extra() {
        let o = ImageTextOpts {
            arch: "image_text".into(),
            ..Default::default()
        };
        assert!(o.validate("lora", &none()).is_ok());
        assert!(o.validate("qlora", &none()).is_err());
        assert!(o.validate("full", &none()).is_err());
        let mut extra = none();
        extra.insert("optim".into(), json!("adamw"));
        assert!(o.validate("lora", &extra).is_err());
    }

    #[test]
    fn image_text_rejects_bad_values() {
        let base = ImageTextOpts {
            arch: "image_text".into(),
            ..Default::default()
        };
        let bad = [
            ImageTextOpts {
                lr_scheduler: Some("exp".into()),
                ..base.clone()
            },
            ImageTextOpts {
                warmup_ratio: Some(1.0),
                ..base.clone()
            },
            ImageTextOpts {
                max_grad_norm: Some(0.0),
                ..base.clone()
            },
            ImageTextOpts {
                max_grad_norm: Some(f64::NAN),
                ..base.clone()
            },
            ImageTextOpts {
                shuffle_buffer: Some(0),
                ..base.clone()
            },
            ImageTextOpts {
                keep_checkpoints: Some(0),
                ..base.clone()
            },
            ImageTextOpts {
                max_steps: Some(0),
                ..base.clone()
            },
            ImageTextOpts {
                lora_target_regex: Some(String::new()),
                ..base.clone()
            },
            ImageTextOpts {
                arch: "vision".into(),
                ..Default::default()
            },
        ];
        for o in bad {
            assert!(o.validate("lora", &none()).is_err(), "{o:?}");
        }
    }
}
