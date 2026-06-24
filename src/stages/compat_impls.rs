//! `Compatible<B>` impls for core cookbook stages.
//!
//! Core stages are generic — they work with any backend.

use blut::framework::compat::Compatible;
use blut::backends::TrainingBackend;

use super::{LoadDataset, TrainModel, EvaluateModel};

impl<B: TrainingBackend> Compatible<B> for LoadDataset {}
impl<B: TrainingBackend> Compatible<B> for TrainModel {}
impl<B: TrainingBackend> Compatible<B> for EvaluateModel {}
