pub mod data;
pub mod model;
pub mod utils_vad;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SileroError {
    #[error("{0}")]
    Message(String),
}

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
