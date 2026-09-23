//! `Compatible<B>` impls for core cookbook stages.
//!
//! Core stages are generic — they work with any backend.

use blut::backends::TrainingBackend;
use blut::framework::compat::Compatible;

use super::{
    EvaluateHeldOut, EvaluateLoadedDataset, EvaluateModel, LoadDataset, TrainModel,
    TrainModelOnSplit,
};

impl<B: TrainingBackend> Compatible<B> for LoadDataset {}
impl<B: TrainingBackend> Compatible<B> for TrainModel {}
impl<B: TrainingBackend> Compatible<B> for TrainModelOnSplit {}
impl<B: TrainingBackend> Compatible<B> for EvaluateModel {}
impl<B: TrainingBackend> Compatible<B> for EvaluateLoadedDataset {}
impl<B: TrainingBackend> Compatible<B> for EvaluateHeldOut {}
