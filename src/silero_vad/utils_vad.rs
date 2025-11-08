use crate::silero_vad::model::OnnxModel;
use crate::silero_vad::{Result, SileroError};

use std::path::Path;
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SpeechTimestamp {
    pub start: f64,
    pub end: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct VadParameters {
    pub threshold: f32,
    pub sampling_rate: u32,
    pub min_speech_duration_ms: u32,
    pub max_speech_duration_s: f32,
    pub min_silence_duration_ms: u32,
    pub speech_pad_ms: u32,
    pub return_seconds: bool,
    pub time_resolution: u32,
    pub visualize_probs: bool,
    pub neg_threshold: Option<f32>,
    pub window_size_samples: Option<usize>,
    pub min_silence_at_max_speech: u32,
    pub use_max_possible_silence: bool,
}

impl Default for VadParameters {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            sampling_rate: 16_000,
            min_speech_duration_ms: 250,
            max_speech_duration_s: f32::INFINITY,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            return_seconds: false,
            time_resolution: 1,
            visualize_probs: false,
            neg_threshold: None,
            window_size_samples: None,
            min_silence_at_max_speech: 98,
            use_max_possible_silence: true,
        }
    }
}

fn make_error(message: impl Into<String>) -> SileroError {
    SileroError::Message(message.into())
}

#[allow(dead_code)]
pub fn read_audio<P: AsRef<Path>>(path: P, sampling_rate: u32) -> Result<Vec<f32>> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(make_error(format!(
            "Audio file not found: {}",
            path.display()
        )));
    }
    if sampling_rate == 0 {
        return Err(make_error("Target sampling rate must be greater than zero"));
    }

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());

    let (mut samples, source_sr) = match extension.as_deref() {
        Some("wav") => read_wav_file(path)?,
        _ => {
            return Err(make_error(format!(
                "Unsupported audio container `{}`. Only WAV files are supported.",
                path.display()
            )));
        }
    };

    if source_sr != sampling_rate {
        samples = resample_linear(&samples, source_sr, sampling_rate)?;
    }

    Ok(samples)
}

#[allow(dead_code)]
pub fn save_audio<P: AsRef<Path>>(path: P, samples: &[f32], sampling_rate: u32) -> Result<()> {
    let path = path.as_ref();
    if sampling_rate == 0 {
        return Err(make_error("Sampling rate must be greater than zero"));
    }

    #[cfg(feature = "audio-wav")]
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sampling_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec).map_err(|e| {
            make_error(format!(
                "Failed to create WAV file {}: {}",
                path.display(),
                e
            ))
        })?;

        for sample in samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let quantized = (clamped * i16::MAX as f32).round() as i16;
            writer
                .write_sample(quantized)
                .map_err(|e| make_error(format!("Failed to write WAV sample: {}", e)))?;
        }

        writer
            .finalize()
            .map_err(|e| make_error(format!("Failed to finalize WAV file: {}", e)))?;
        Ok(())
    }

    #[cfg(not(feature = "audio-wav"))]
    {
        let _ = (samples, sampling_rate);
        Err(make_error(
            "Saving audio requires the `audio-wav` feature to be enabled",
        ))
    }
}

#[allow(dead_code)]
pub fn collect_chunks(
    timestamps: &[SpeechTimestamp],
    wav: &[f32],
    seconds: bool,
    sampling_rate: Option<u32>,
) -> Result<Vec<f32>> {
    if seconds && sampling_rate.is_none() {
        return Err(make_error(
            "sampling_rate must be provided when seconds is true",
        ));
    }

    if timestamps.is_empty() {
        return Ok(Vec::new());
    }

    let sr = if seconds { sampling_rate.unwrap() } else { 0 };
    let mut result = Vec::new();

    for ts in timestamps {
        let mut start = timestamp_to_index(ts.start, seconds, sr);
        let mut end = timestamp_to_index(ts.end, seconds, sr);
        if end <= start {
            continue;
        }
        start = start.min(wav.len());
        end = end.min(wav.len());
        if start < end {
            result.extend_from_slice(&wav[start..end]);
        }
    }

    Ok(result)
}

#[allow(dead_code)]
pub fn drop_chunks(
    timestamps: &[SpeechTimestamp],
    wav: &[f32],
    seconds: bool,
    sampling_rate: Option<u32>,
) -> Result<Vec<f32>> {
    if seconds && sampling_rate.is_none() {
        return Err(make_error(
            "sampling_rate must be provided when seconds is true",
        ));
    }

    if timestamps.is_empty() {
        return Ok(wav.to_vec());
    }

    let sr = if seconds { sampling_rate.unwrap() } else { 0 };
    let mut result = Vec::with_capacity(wav.len());
    let mut cursor = 0usize;

    for ts in timestamps {
        let start = timestamp_to_index(ts.start, seconds, sr).min(wav.len());
        let end = timestamp_to_index(ts.end, seconds, sr).min(wav.len());
        if start > cursor {
            result.extend_from_slice(&wav[cursor..start]);
        }
        cursor = cursor.max(end);
    }

    if cursor < wav.len() {
        result.extend_from_slice(&wav[cursor..]);
    }

    Ok(result)
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct VadIteratorParams {
    pub threshold: f32,
    pub sampling_rate: u32,
    pub min_silence_duration_ms: u32,
    pub speech_pad_ms: u32,
}

impl Default for VadIteratorParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            sampling_rate: 16_000,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum VadEvent {
    Start(f64),
    End(f64),
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct VadIterator {
    pub model: OnnxModel,
    pub params: VadIteratorParams,
    triggered: bool,
    temp_end: Option<usize>,
    current_sample: usize,
    min_silence_samples: f64,
    speech_pad_samples: f64,
}

impl VadIterator {
    #[allow(dead_code)]
    pub fn new(model: OnnxModel, params: VadIteratorParams) -> Result<Self> {
        if !matches!(params.sampling_rate, 8_000 | 16_000) {
            return Err(make_error(
                "VADIterator does not support sampling rates other than [8000, 16000]",
            ));
        }

        let mut iterator = Self {
            model,
            params,
            triggered: false,
            temp_end: None,
            current_sample: 0,
            min_silence_samples: 0.0,
            speech_pad_samples: 0.0,
        };
        iterator.reset_states();
        Ok(iterator)
    }

    #[allow(dead_code)]
    pub fn reset_states(&mut self) {
        self.model.reset_states();
        self.triggered = false;
        self.temp_end = None;
        self.current_sample = 0;
        self.min_silence_samples =
            self.params.sampling_rate as f64 * self.params.min_silence_duration_ms as f64 / 1000.0;
        self.speech_pad_samples =
            self.params.sampling_rate as f64 * self.params.speech_pad_ms as f64 / 1000.0;
    }

    #[allow(dead_code)]
    pub fn process_chunk(
        &mut self,
        chunk: &[f32],
        return_seconds: bool,
        time_resolution: u32,
    ) -> Result<Option<VadEvent>> {
        if chunk.is_empty() {
            return Ok(None);
        }

        let chunk_len = chunk.len();
        let output = self.model.forward_chunk(chunk, self.params.sampling_rate)?;
        self.current_sample += chunk_len;
        let speech_prob = output[[0, 0]];

        if speech_prob >= self.params.threshold {
            if self.temp_end.is_some() {
                self.temp_end = None;
            }
            if !self.triggered {
                self.triggered = true;
                let pad = self.speech_pad_samples.floor() as isize;
                let mut start_index = self.current_sample as isize - pad - chunk_len as isize;
                if start_index < 0 {
                    start_index = 0;
                }
                let position = format_position(
                    start_index as usize,
                    return_seconds,
                    self.params.sampling_rate,
                    time_resolution,
                );
                return Ok(Some(VadEvent::Start(position)));
            }
        }

        let neg_threshold = self.params.threshold - 0.15;
        if speech_prob < neg_threshold && self.triggered {
            if self.temp_end.is_none() {
                self.temp_end = Some(self.current_sample);
            }

            let temp_end = self.temp_end.unwrap();
            let silence_duration = self.current_sample - temp_end;
            if (silence_duration as f64) < self.min_silence_samples {
                return Ok(None);
            }

            let pad = self.speech_pad_samples.floor() as isize;
            let mut end_index = temp_end as isize + pad - chunk_len as isize;
            if end_index < 0 {
                end_index = 0;
            }

            self.temp_end = None;
            self.triggered = false;

            let position = format_position(
                end_index as usize,
                return_seconds,
                self.params.sampling_rate,
                time_resolution,
            );
            return Ok(Some(VadEvent::End(position)));
        }

        Ok(None)
    }
}

#[allow(dead_code)]
pub fn get_speech_timestamps(
    audio: &[f32],
    model: &mut OnnxModel,
    params: &VadParameters,
) -> Result<Vec<SpeechTimestamp>> {
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    let mut audio_vec = audio.to_vec();
    let mut sampling_rate = params.sampling_rate;
    if sampling_rate == 0 {
        return Err(make_error("Sampling rate must be greater than zero"));
    }

    let mut step = 1usize;
    if sampling_rate > 16_000 && sampling_rate % 16_000 == 0 {
        let factor = (sampling_rate / 16_000) as usize;
        step = factor.max(1);
        sampling_rate = 16_000;
        audio_vec = audio_vec.into_iter().step_by(step).collect();
    }

    if sampling_rate != 8_000 && sampling_rate != 16_000 {
        return Err(make_error(
            "Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates",
        ));
    }

    let window_size_samples = if sampling_rate == 16_000 { 512 } else { 256 };

    model.reset_states();
    let min_speech_samples = sampling_rate as f64 * params.min_speech_duration_ms as f64 / 1000.0;
    let speech_pad_samples = sampling_rate as f64 * params.speech_pad_ms as f64 / 1000.0;
    let max_speech_samples = if params.max_speech_duration_s.is_finite() {
        sampling_rate as f64 * params.max_speech_duration_s as f64
            - window_size_samples as f64
            - 2.0 * speech_pad_samples
    } else {
        f64::INFINITY
    };
    let min_silence_samples = sampling_rate as f64 * params.min_silence_duration_ms as f64 / 1000.0;
    let min_silence_samples_at_max_speech =
        sampling_rate as f64 * params.min_silence_at_max_speech as f64 / 1000.0;

    let audio_length_samples = audio_vec.len();
    let mut speech_probs =
        Vec::with_capacity((audio_length_samples + window_size_samples - 1) / window_size_samples);

    let mut idx = 0usize;
    while idx < audio_length_samples {
        let end = (idx + window_size_samples).min(audio_length_samples);
        let mut chunk = vec![0.0_f32; window_size_samples];
        chunk[..(end - idx)].copy_from_slice(&audio_vec[idx..end]);
        let output = model.forward_chunk(&chunk, sampling_rate)?;
        speech_probs.push(output[[0, 0]]);
        idx += window_size_samples;
    }

    let mut triggered = false;
    let mut current_start = 0usize;
    let mut has_current_start = false;
    let mut temp_end = 0usize;
    let mut prev_end = 0usize;
    let mut next_start = 0usize;
    let mut possible_ends: Vec<(usize, usize)> = Vec::new();
    let neg_threshold = params
        .neg_threshold
        .unwrap_or_else(|| (params.threshold - 0.15).max(0.01));

    let mut speeches: Vec<(usize, usize)> = Vec::new();

    for (i, &speech_prob) in speech_probs.iter().enumerate() {
        let cur_sample = window_size_samples * i;

        if speech_prob >= params.threshold && temp_end != 0 {
            let sil_dur = cur_sample.saturating_sub(temp_end);
            if (sil_dur as f64) > min_silence_samples_at_max_speech {
                possible_ends.push((temp_end, sil_dur));
            }
            temp_end = 0;
            if next_start < prev_end {
                next_start = cur_sample;
            }
        }

        if speech_prob >= params.threshold && !triggered {
            triggered = true;
            current_start = cur_sample;
            has_current_start = true;
            continue;
        }

        if triggered
            && has_current_start
            && (cur_sample.saturating_sub(current_start) as f64) > max_speech_samples
        {
            if params.use_max_possible_silence && !possible_ends.is_empty() {
                let (best_end, dur) = possible_ends
                    .iter()
                    .cloned()
                    .max_by_key(|(_, dur)| *dur)
                    .unwrap();
                speeches.push((current_start, best_end));
                has_current_start = false;
                next_start = best_end + dur;
                if next_start < best_end + cur_sample {
                    current_start = next_start;
                    has_current_start = true;
                    triggered = true;
                } else {
                    triggered = false;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
                continue;
            } else if prev_end != 0 {
                speeches.push((current_start, prev_end));
                has_current_start = false;
                if next_start < prev_end {
                    triggered = false;
                } else {
                    current_start = next_start;
                    has_current_start = true;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
                continue;
            } else {
                speeches.push((current_start, cur_sample));
                has_current_start = false;
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                possible_ends.clear();
                continue;
            }
        }

        if speech_prob < neg_threshold && triggered {
            if temp_end == 0 {
                temp_end = cur_sample;
            }
            let sil_dur_now = cur_sample.saturating_sub(temp_end);

            if !params.use_max_possible_silence
                && (sil_dur_now as f64) > min_silence_samples_at_max_speech
            {
                prev_end = temp_end;
            }

            if (sil_dur_now as f64) < min_silence_samples {
                continue;
            } else {
                let end = temp_end;
                if has_current_start
                    && (end.saturating_sub(current_start) as f64) > min_speech_samples
                {
                    speeches.push((current_start, end));
                }
                triggered = false;
                has_current_start = false;
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
                continue;
            }
        }
    }

    if has_current_start
        && (audio_length_samples.saturating_sub(current_start) as f64) > min_speech_samples
    {
        speeches.push((current_start, audio_length_samples));
    }

    if speeches.is_empty() {
        return Ok(Vec::new());
    }

    let mut segments = speeches;
    let speech_pad = speech_pad_samples.floor() as usize;

    for i in 0..segments.len() {
        if i == 0 {
            segments[i].0 = segments[i].0.saturating_sub(speech_pad);
        }

        if i != segments.len() - 1 {
            let silence_duration = segments[i + 1].0.saturating_sub(segments[i].1);
            if silence_duration < 2 * speech_pad {
                let adjust = silence_duration / 2;
                segments[i].1 = (segments[i].1 + adjust).min(audio_length_samples);
                segments[i + 1].0 = segments[i + 1].0.saturating_sub(adjust);
            } else {
                segments[i].1 = (segments[i].1 + speech_pad).min(audio_length_samples);
                segments[i + 1].0 = segments[i + 1].0.saturating_sub(speech_pad);
            }
        } else {
            segments[i].1 = (segments[i].1 + speech_pad).min(audio_length_samples);
        }
    }

    let mut result = Vec::with_capacity(segments.len());

    if params.return_seconds {
        let sr_f64 = sampling_rate as f64;
        let audio_length_seconds = audio_length_samples as f64 / sr_f64;
        for (start, end) in segments {
            if end <= start {
                continue;
            }
            let start_sec =
                round_with_resolution(start as f64 / sr_f64, params.time_resolution).max(0.0);
            let mut end_sec = round_with_resolution(end as f64 / sr_f64, params.time_resolution);
            if end_sec > audio_length_seconds {
                end_sec = audio_length_seconds;
            }
            if end_sec <= start_sec {
                continue;
            }
            result.push(SpeechTimestamp {
                start: start_sec,
                end: end_sec,
            });
        }
    } else {
        let scale = step as f64;
        for (start, end) in segments {
            if end <= start {
                continue;
            }
            result.push(SpeechTimestamp {
                start: start as f64 * scale,
                end: end as f64 * scale,
            });
        }
    }

    Ok(result)
}

fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(make_error("Sampling rate must be greater than zero"));
    }
    if samples.is_empty() || src_rate == dst_rate {
        return Ok(samples.to_vec());
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let mut output_len = ((samples.len() as f64) * ratio).round() as usize;
    if output_len == 0 {
        output_len = 1;
    }

    let mut output = Vec::with_capacity(output_len);
    let last_index = samples.len() - 1;

    for i in 0..output_len {
        let pos = (i as f64) / ratio;
        let base = pos.floor() as usize;
        let base = base.min(last_index);
        let frac = pos - base as f64;
        let next = if base >= last_index {
            last_index
        } else {
            base + 1
        };
        let v0 = samples[base];
        let v1 = samples[next];
        let frac_f32 = frac as f32;
        let interpolated = v0 * (1.0_f32 - frac_f32) + v1 * frac_f32;
        output.push(interpolated);
    }

    Ok(output)
}

#[cfg(feature = "audio-wav")]
fn read_wav_file(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| make_error(format!("Failed to open WAV file {}: {}", path.display(), e)))?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    if sample_rate == 0 {
        return Err(make_error("WAV file reports zero sample rate"));
    }
    let channels = spec.channels.max(1) as usize;

    let mut interleaved = Vec::with_capacity(reader.duration() as usize * channels);

    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                interleaved.push(
                    sample.map_err(|e| make_error(format!("Failed to read WAV sample: {}", e)))?,
                );
            }
        }
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            if bits <= 8 {
                for sample in reader.samples::<i8>() {
                    let value = sample
                        .map_err(|e| make_error(format!("Failed to read WAV sample: {}", e)))?;
                    interleaved.push(value as f32 / i8::MAX as f32);
                }
            } else if bits <= 16 {
                for sample in reader.samples::<i16>() {
                    let value = sample
                        .map_err(|e| make_error(format!("Failed to read WAV sample: {}", e)))?;
                    interleaved.push(value as f32 / i16::MAX as f32);
                }
            } else {
                let max_amplitude = (1u32 << (bits - 1)) as f32;
                for sample in reader.samples::<i32>() {
                    let value = sample
                        .map_err(|e| make_error(format!("Failed to read WAV sample: {}", e)))?;
                    interleaved.push(value as f32 / max_amplitude);
                }
            }
        }
    }

    let mono = interleaved_to_mono(interleaved, channels);
    Ok((mono, sample_rate))
}

#[cfg(not(feature = "audio-wav"))]
fn read_wav_file(_path: &Path) -> Result<(Vec<f32>, u32)> {
    Err(make_error(
        "WAV support is disabled. Enable the `audio-wav` feature to read WAV files.",
    ))
}

fn interleaved_to_mono(samples: Vec<f32>, channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples;
    }

    let mut mono = Vec::with_capacity(samples.len() / channels);
    for frame in samples.chunks(channels) {
        let mut sum = 0.0_f32;
        for sample in frame {
            sum += *sample;
        }
        mono.push(sum / channels as f32);
    }
    mono
}

fn timestamp_to_index(value: f64, seconds: bool, sampling_rate: u32) -> usize {
    if !value.is_finite() {
        return 0;
    }

    let scaled = if seconds {
        value * sampling_rate as f64
    } else {
        value
    };

    if scaled <= 0.0 {
        0
    } else {
        scaled.round() as usize
    }
}

fn format_position(index: usize, seconds: bool, sampling_rate: u32, time_resolution: u32) -> f64 {
    if seconds {
        let sr = sampling_rate as f64;
        round_with_resolution(index as f64 / sr, time_resolution)
    } else {
        index as f64
    }
}

fn round_with_resolution(value: f64, resolution: u32) -> f64 {
    if resolution == 0 {
        return value.round();
    }
    let factor = 10f64.powi(resolution as i32);
    (value * factor).round() / factor
}
