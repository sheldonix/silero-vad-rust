use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use silero_vad_rust::{
    get_speech_timestamps, load_silero_vad, read_audio,
    silero_vad::{model::load_silero_vad_with_options, utils_vad::VadParameters},
};

fn data_path(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(file)
}

fn process_audio_sample(file: &str) -> Result<()> {
    let path = data_path(file);
    let audio = read_audio(&path, 16_000)
        .with_context(|| format!("failed to read audio sample {}", file))?;
    assert!(
        audio.len() > 512,
        "audio sample should contain multiple frames"
    );

    let mut model = load_silero_vad().context("failed to load ONNX model")?;
    let params = VadParameters {
        return_seconds: true,
        ..Default::default()
    };

    let timestamps = get_speech_timestamps(&audio, &mut model, &params)
        .context("failed to compute speech timestamps")?;
    assert!(
        !timestamps.is_empty(),
        "expected non-empty timestamps for {}",
        file
    );

    let output = model
        .audio_forward(&audio, 16_000)
        .context("failed to run audio_forward")?;
    assert_eq!(
        output.nrows(),
        1,
        "model output should have batch dimension 1"
    );
    assert!(
        output.ncols() > 0,
        "model output should contain probability values"
    );

    Ok(())
}

#[test]
fn onnx_model_processes_wav_audio() -> Result<()> {
    process_audio_sample("test.wav")
}

#[test]
fn torchscript_models_are_not_supported() {
    let mut options = silero_vad_rust::silero_vad::model::LoadOptions::default();
    options.use_onnx = false;
    let error = load_silero_vad_with_options(options)
        .expect_err("TorchScript models are intentionally unsupported");
    assert!(
        error
            .to_string()
            .contains("TorchScript models are not supported"),
        "returned error should state TorchScript support is missing"
    );
}
