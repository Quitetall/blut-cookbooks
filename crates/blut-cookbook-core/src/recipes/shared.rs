//! Arguments shared by the recipes that load a dataset and end in
//! `train_model`.
//!
//! `train_from_dataset` and `finetune_pretrained` both hard-coded the dataset
//! slice (`subset: None, max_samples: None`) and the training layout
//! (`nproc_per_node: 1, nnodes: 1`) when building their stages, so neither a
//! multi-config dataset nor a multi-GPU run was reachable from either one.
//! These structs are flattened into both recipes' arguments: the JSON a user
//! writes is unchanged, and the two recipes cannot drift apart again.
//!
//! Every field is omitted from serialization at its default. Recipe arguments
//! are part of a plan's identity, so a run that sets none of them keeps the
//! identity — and the cache entries — it had before they existed.

use serde::{Deserialize, Serialize};

use blut_backends::distributed::one;

use crate::stages::train_model::ParallelStrategy;

/// Which slice of a HuggingFace dataset to load.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DatasetSelect {
    /// HuggingFace dataset config/subset name. Required by datasets that
    /// publish several configs — `wikitext` has no loadable default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subset: Option<String>,
    /// Cap the rows loaded. Unset takes the whole split.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_samples: Option<usize>,
}

/// How training is laid out across processes, and how text becomes tokens.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TrainShape {
    /// Processes (GPUs) per node. >1 runs local DDP or FSDP under torchrun.
    #[serde(
        default = "blut_backends::distributed::one",
        skip_serializing_if = "blut_backends::distributed::is_one"
    )]
    pub nproc_per_node: u32,
    /// Nodes in the job. >1 needs `MASTER_ADDR` and `NODE_RANK` in the
    /// environment of every node; the Slurm launcher exports both.
    #[serde(
        default = "blut_backends::distributed::one",
        skip_serializing_if = "blut_backends::distributed::is_one"
    )]
    pub nnodes: u32,
    /// `ddp` (replicate, default) or `fsdp` (FSDP2 shard). Only meaningful
    /// once the run is distributed.
    #[serde(default, skip_serializing_if = "ParallelStrategy::is_ddp")]
    pub parallel_strategy: ParallelStrategy,
    /// Causal-LM training: the dataset column holding the text (trainer
    /// default `"text"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_field: Option<String>,
    /// Causal-LM training: tokens per packed block (trainer default 512).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_seq_len: Option<u32>,
}

impl Default for TrainShape {
    fn default() -> Self {
        Self {
            nproc_per_node: one(),
            nnodes: one(),
            parallel_strategy: ParallelStrategy::default(),
            text_field: None,
            max_seq_len: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_serialize_to_nothing() {
        assert_eq!(
            serde_json::to_value(DatasetSelect::default()).unwrap(),
            serde_json::json!({})
        );
        assert_eq!(
            serde_json::to_value(TrainShape::default()).unwrap(),
            serde_json::json!({})
        );
    }

    #[test]
    fn an_empty_object_deserializes_to_the_defaults() {
        let shape: TrainShape = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(shape, TrainShape::default());
        assert_eq!((shape.nproc_per_node, shape.nnodes), (1, 1));
    }

    #[test]
    fn set_fields_round_trip() {
        let shape = TrainShape {
            nproc_per_node: 4,
            nnodes: 2,
            parallel_strategy: ParallelStrategy::Fsdp,
            text_field: Some("body".into()),
            max_seq_len: Some(1024),
        };
        let json = serde_json::to_value(&shape).unwrap();
        assert_eq!(serde_json::from_value::<TrainShape>(json).unwrap(), shape);
    }
}
