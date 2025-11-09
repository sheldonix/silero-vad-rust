//! Rust port of the Silero Voice Activity Detector with ONNX Runtime helpers.
//!
//! The crate re-exports the most common entry points from [`silero_vad::model`]
//! and [`silero_vad::utils_vad`] so downstream crates can call into the
//! high-level helpers without navigating the internal module tree.
//!
//! # Quick Start
//!
//! ## Offline Pass
//!
//! ```no_run
//! use silero_vad_rust::{get_speech_timestamps, load_silero_vad, read_audio};
//! use silero_vad_rust::silero_vad::utils_vad::VadParameters;
//!
//! fn main() -> anyhow::Result<()> {
//!     let audio = read_audio("samples/test.wav", 16_000)?;
//!     let mut model = load_silero_vad()?; // defaults to ONNX opset 16
//!     let params = VadParameters {
//!         return_seconds: true,
//!         ..Default::default()
//!     };
//!
//!     let speech = get_speech_timestamps(&audio, &mut model, &params)?;
//!     println!("Detected segments: {speech:?}");
//!     Ok(())
//! }
//! ```
//!
//! ## Streaming Chunks
//!
//! ```no_run
//! use silero_vad_rust::{load_silero_vad, read_audio};
//!
//! fn stream_chunks() -> anyhow::Result<()> {
//!     let audio = read_audio("samples/long.wav", 16_000)?;
//!     let mut model = load_silero_vad()?;
//!     let chunk_size = 512; // 16 kHz window
//!
//!     for frame in audio.chunks(chunk_size) {
//!         let padded = if frame.len() == chunk_size {
//!             frame.to_vec()
//!         } else {
//!             let mut tmp = vec![0.0f32; chunk_size];
//!             tmp[..frame.len()].copy_from_slice(frame);
//!             tmp
//!         };
//!
//!         let probs = model.forward_chunk(&padded, 16_000)?;
//!         println!("frame prob={:.3}", probs[[0, 0]]);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Segment Trimming & Muting
//!
//! ```no_run
//! use silero_vad_rust::{
//!     collect_chunks, drop_chunks, get_speech_timestamps, load_silero_vad, read_audio, save_audio,
//! };
//! use silero_vad_rust::silero_vad::utils_vad::VadParameters;
//!
//! fn trim_audio() -> anyhow::Result<()> {
//!     let audio = read_audio("samples/raw.wav", 16_000)?;
//!     let mut model = load_silero_vad()?;
//!     let params = VadParameters {
//!         return_seconds: false,
//!         ..Default::default()
//!     };
//!     let speech = get_speech_timestamps(&audio, &mut model, &params)?;
//!
//!     let voice_only = collect_chunks(&speech, &audio, false, None)?;
//!     save_audio("out_voice.wav", &voice_only, 16_000)?;
//!
//!     let muted_voice = drop_chunks(&speech, &audio, false, None)?;
//!     save_audio("out_silence.wav", &muted_voice, 16_000)?;
//!     Ok(())
//! }
//! ```
//!
//! ## Event-Driven Iterator
//!
//! ```no_run
//! use silero_vad_rust::{
//!     load_silero_vad, read_audio,
//!     silero_vad::utils_vad::{VadEvent, VadIterator, VadIteratorParams},
//! };
//!
//! fn iterate_events() -> anyhow::Result<()> {
//!     let audio = read_audio("samples/live.wav", 16_000)?;
//!     let model = load_silero_vad()?;
//!     let params = VadIteratorParams {
//!         threshold: 0.55,
//!         ..Default::default()
//!     };
//!     let mut iterator = VadIterator::new(model, params)?;
//!
//!     for frame in audio.chunks(512) {
//!         let event = iterator.process_chunk(frame, true, 1)?;
//!         if let Some(VadEvent::Start(ts)) = event {
//!             println!("speech started at {ts}s");
//!         } else if let Some(VadEvent::End(ts)) = event {
//!             println!("speech ended at {ts}s");
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Enabling GPU Runtime
//!
//! ```no_run
//! use silero_vad_rust::silero_vad::model::{load_silero_vad_with_options, LoadOptions};
//!
//! fn load_gpu_model() -> anyhow::Result<()> {
//!     let options = LoadOptions {
//!         opset_version: 15,
//!         force_onnx_cpu: false, // allow custom providers (GPU, NNAPI, etc.)
//!         ..Default::default()
//!     };
//!     let _model = load_silero_vad_with_options(options)?;
//!     Ok(())
//! }
//! ```
//!
//! ## Tuning Parameters
//!
//! ```no_run
//! use silero_vad_rust::{get_speech_timestamps, load_silero_vad, read_audio};
//! use silero_vad_rust::silero_vad::utils_vad::VadParameters;
//!
//! fn compare_thresholds() -> anyhow::Result<()> {
//!     let audio = read_audio("samples/noisy.wav", 16_000)?;
//!     let mut model = load_silero_vad()?;
//!
//!     let mut strict = VadParameters::default();
//!     strict.threshold = 0.65;
//!     strict.min_speech_duration_ms = 400;
//!
//!     let mut permissive = VadParameters::default();
//!     permissive.threshold = 0.4;
//!     permissive.min_speech_duration_ms = 150;
//!
//!     let strict_segments = get_speech_timestamps(&audio, &mut model, &strict)?;
//!     model.reset_states();
//!     let permissive_segments = get_speech_timestamps(&audio, &mut model, &permissive)?;
//!
//!     println!("strict count: {}", strict_segments.len());
//!     println!("permissive count: {}", permissive_segments.len());
//!     Ok(())
//! }
//! ```

pub mod silero_vad;

/// Loads the default Silero VAD ONNX model (opset 16, CPU provider).
pub use silero_vad::model::load_silero_vad;
/// Convenience re-exports for the high-level audio helpers.
pub use silero_vad::utils_vad::{
    VadIterator, collect_chunks, drop_chunks, get_speech_timestamps, read_audio, save_audio,
};
