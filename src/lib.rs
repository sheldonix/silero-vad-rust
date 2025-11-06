pub mod silero_vad;

pub use silero_vad::model::load_silero_vad;
pub use silero_vad::utils_vad::{
    VadIterator, collect_chunks, drop_chunks, get_speech_timestamps, read_audio, save_audio,
};
