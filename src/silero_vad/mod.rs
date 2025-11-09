//! Core Silero VAD building blocks shared across the Rust crate.
//!
//! This module groups the bundled ONNX weights (`data`), model-loading helpers
//! (`model`) and post-processing utilities (`utils_vad`) exposed by the crate.

pub mod data;
pub mod model;
pub mod utils_vad;

use thiserror::Error;

/// Unified error type returned by Silero VAD helpers.
#[derive(Debug, Error)]
pub enum SileroError {
    /// Arbitrary message produced by downstream crates or custom guards.
    #[error("{0}")]
    Message(String),
}

/// Convenience alias for results returned by public Silero VAD functions.
pub type Result<T> = std::result::Result<T, SileroError>;

impl From<ort::Error> for SileroError {
    fn from(value: ort::Error) -> Self {
        Self::Message(value.to_string())
    }
}

impl From<ndarray::ShapeError> for SileroError {
    fn from(value: ndarray::ShapeError) -> Self {
        Self::Message(value.to_string())
    }
}
