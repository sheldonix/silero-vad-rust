use crate::silero_vad::data::{ONNX_MODELS, SILERO_VAD_ONNX, SILERO_VAD_OP15_ONNX};
use crate::silero_vad::{Result, SileroError};

use ndarray::{Array1, Array2, ArrayD, Axis, s};
use ort::{execution_providers::CPUExecutionProvider, session::Session, value::Tensor};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub use_onnx: bool,
    pub opset_version: u32,
    pub force_onnx_cpu: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            use_onnx: true,
            opset_version: 16,
            force_onnx_cpu: true,
        }
    }
}

#[derive(Debug)]
pub struct OnnxModel {
    session: Session,
    state: ArrayD<f32>,
    context: Option<Array2<f32>>,
    sample_rates: Vec<u32>,
    last_sr: Option<u32>,
    last_batch_size: Option<usize>,
}

impl OnnxModel {
    pub fn from_path(path: impl AsRef<Path>, force_cpu: bool) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(SileroError::Message(format!(
                "Model file not found: {}",
                path.display()
            )));
        }

        let mut builder = Session::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?;
        if force_cpu {
            builder =
                builder.with_execution_providers([CPUExecutionProvider::default().build()])?;
        }
        let session = builder.commit_from_file(path)?;

        let sample_rates = if path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.contains("16k"))
            .unwrap_or(false)
        {
            vec![16_000]
        } else {
            vec![8_000, 16_000]
        };

        Ok(Self {
            session,
            state: ArrayD::<f32>::zeros(ndarray::IxDyn(&[2, 1, 128])),
            context: None,
            sample_rates,
            last_sr: None,
            last_batch_size: None,
        })
    }

    #[inline]
    fn context_size(sr: u32) -> usize {
        match sr {
            16_000 => 64,
            8_000 => 32,
            other => {
                if other % 16_000 == 0 {
                    64
                } else {
                    32
                }
            }
        }
    }

    #[inline]
    fn chunk_size(sr: u32) -> usize {
        match sr {
            16_000 => 512,
            8_000 => 256,
            other if other % 16_000 == 0 => 512,
            _ => 256,
        }
    }

    fn ensure_state(&mut self, batch_size: usize, sr: u32) {
        let desired_shape = [2, batch_size, 128];
        if self.state.shape() != &desired_shape {
            self.state = ArrayD::<f32>::zeros(ndarray::IxDyn(&desired_shape));
        }

        let context_size = Self::context_size(sr);
        match &self.context {
            Some(ctx) if ctx.nrows() == batch_size && ctx.ncols() == context_size => {}
            _ => {
                self.context = Some(Array2::<f32>::zeros((batch_size, context_size)));
            }
        }
    }

    fn normalize_input(&self, mut input: Array2<f32>, sr: u32) -> Result<(Array2<f32>, u32)> {
        let mut sr = sr;
        if sr != 16_000 && sr % 16_000 == 0 {
            let step = (sr / 16_000) as usize;
            let cols = input.ncols();
            let new_cols = (cols + step - 1) / step;
            let mut downsampled = Array2::<f32>::zeros((input.nrows(), new_cols));
            for row in 0..input.nrows() {
                let mut dst_col = 0;
                let mut src_col = 0;
                while src_col < cols {
                    downsampled[(row, dst_col)] = input[(row, src_col)];
                    dst_col += 1;
                    src_col += step;
                }
            }
            input = downsampled;
            sr = 16_000;
        }

        if !self.sample_rates.contains(&sr) {
            return Err(SileroError::Message(format!(
                "Supported sampling rates: {:?} (or multiples of 16000)",
                self.sample_rates
            )));
        }

        let sr_per_sample = (sr as f32) / (input.ncols() as f32);
        if sr_per_sample > 31.25 {
            return Err(SileroError::Message(
                "Input audio chunk is too short".to_string(),
            ));
        }

        Ok((input, sr))
    }

    pub fn reset_states(&mut self) {
        self.context = None;
        self.state = ArrayD::<f32>::zeros(ndarray::IxDyn(&[2, 1, 128]));
        self.last_sr = None;
        self.last_batch_size = None;
    }

    pub fn forward_chunk(&mut self, chunk: &[f32], sr: u32) -> Result<Array2<f32>> {
        let array = Array2::from_shape_vec((1, chunk.len()), chunk.to_vec())?;
        self.forward(array, sr)
    }

    pub fn forward(&mut self, input: Array2<f32>, sr: u32) -> Result<Array2<f32>> {
        let (input, sr) = self.normalize_input(input, sr)?;
        let batch_size = input.nrows();
        let chunk_size = Self::chunk_size(sr);

        if input.ncols() != chunk_size {
            return Err(SileroError::Message(format!(
                "Provided number of samples is {} (Supported values: {} for 16000 sample rate, {} for 8000)",
                input.ncols(),
                Self::chunk_size(16_000),
                Self::chunk_size(8_000)
            )));
        }

        if self.last_batch_size != Some(batch_size) || self.last_sr != Some(sr) {
            self.state = ArrayD::<f32>::zeros(ndarray::IxDyn(&[2, batch_size, 128]));
            self.context = None;
        }
        self.ensure_state(batch_size, sr);

        let context_size = Self::context_size(sr);
        let sr_array = Array1::<i64>::from_elem(1, sr as i64);

        let context = self
            .context
            .get_or_insert_with(|| Array2::<f32>::zeros((batch_size, context_size)));

        let mut concatenated = Array2::<f32>::zeros((batch_size, context_size + chunk_size));
        concatenated
            .slice_mut(s![.., 0..context_size])
            .assign(context);
        concatenated
            .slice_mut(s![.., context_size..])
            .assign(&input);

        let input_tensor = Tensor::from_array(concatenated.clone())?;
        let state_tensor = Tensor::from_array(self.state.clone())?;
        let sr_tensor = Tensor::from_array(sr_array)?;
        let inputs = ort::inputs![input_tensor, state_tensor, sr_tensor];

        let outputs = self.session.run(inputs)?;

        let state_key = if outputs.contains_key("stateN") {
            "stateN"
        } else if outputs.contains_key("state") {
            "state"
        } else {
            outputs
                .iter()
                .nth(1)
                .map(|(name, _)| name)
                .unwrap_or("state")
        };

        let (state_shape, state_data) = outputs[state_key].try_extract_tensor::<f32>()?;
        self.state = ArrayD::<f32>::from_shape_vec(state_shape.to_ixdyn(), state_data.to_vec())?;

        let output_key = if outputs.contains_key("output") {
            "output"
        } else {
            outputs
                .iter()
                .next()
                .map(|(name, _)| name)
                .unwrap_or("output")
        };

        let (_output_shape, output_data) = outputs[output_key].try_extract_tensor::<f32>()?;
        let total = output_data.len();
        if batch_size == 0 {
            return Err(SileroError::Message(
                "Batch size must be greater than zero".to_string(),
            ));
        }
        let columns = if total % batch_size == 0 {
            total / batch_size
        } else {
            return Err(SileroError::Message(format!(
                "Unexpected output shape: elements ({total}) not divisible by batch size {batch_size}"
            )));
        };
        let columns = columns.max(1);
        let output = Array2::<f32>::from_shape_vec((batch_size, columns), output_data.to_vec())?;

        let new_context = input
            .slice(s![.., (chunk_size - context_size)..])
            .to_owned();
        *context = new_context;
        self.last_sr = Some(sr);
        self.last_batch_size = Some(batch_size);

        Ok(output)
    }

    pub fn audio_forward(&mut self, audio: &[f32], sr: u32) -> Result<Array2<f32>> {
        let array = Array2::from_shape_vec((1, audio.len()), audio.to_vec())?;
        let (mut array, sr) = self.normalize_input(array, sr)?;
        self.state = ArrayD::<f32>::zeros(ndarray::IxDyn(&[2, 1, 128]));
        self.context = None;
        self.last_sr = None;
        self.last_batch_size = None;

        let chunk_size = Self::chunk_size(sr);
        let remainder = array.ncols() % chunk_size;
        if remainder != 0 {
            let pad = chunk_size - remainder;
            let mut padded = Array2::<f32>::zeros((1, array.ncols() + pad));
            padded.slice_mut(s![.., 0..array.ncols()]).assign(&array);
            array = padded;
        }

        let mut outputs = Vec::new();
        for start in (0..array.ncols()).step_by(chunk_size) {
            let end = start + chunk_size;
            let chunk = array.slice(s![.., start..end]).to_owned();
            let chunk_out = self.forward(chunk, sr)?;
            outputs.push(chunk_out);
        }

        let views: Vec<_> = outputs.iter().map(|arr| arr.view()).collect();
        let concatenated = ndarray::concatenate(Axis(1), &views)?;
        Ok(concatenated)
    }

    pub fn sample_rates(&self) -> &[u32] {
        &self.sample_rates
    }
}

pub fn load_silero_vad() -> Result<OnnxModel> {
    load_silero_vad_with_options(LoadOptions::default())
}

pub fn load_silero_vad_with_options(options: LoadOptions) -> Result<OnnxModel> {
    if !options.use_onnx {
        return Err(SileroError::Message(
            "TorchScript models are not supported in the Rust port yet".into(),
        ));
    }

    let available_ops = [15u32, 16u32];
    if !available_ops.contains(&options.opset_version) {
        return Err(SileroError::Message(format!(
            "Available ONNX opset_version: {:?}",
            available_ops
        )));
    }

    let model_name = if options.opset_version == 16 {
        SILERO_VAD_ONNX
    } else {
        SILERO_VAD_OP15_ONNX
    };

    if !ONNX_MODELS.contains(&model_name) {
        return Err(SileroError::Message(
            "Requested model is not available".into(),
        ));
    }

    let model_path = resolve_model_path(model_name)?;
    OnnxModel::from_path(model_path, options.force_onnx_cpu)
}

fn resolve_model_path(model_name: &str) -> Result<PathBuf> {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/silero_vad/data");
    if !base.exists() {
        return Err(SileroError::Message(format!(
            "Model directory not found: {}",
            base.display()
        )));
    }
    let path = base.join(model_name);
    if !path.exists() {
        return Err(SileroError::Message(format!(
            "Model file not found: {}",
            path.display()
        )));
    }
    Ok(path)
}
