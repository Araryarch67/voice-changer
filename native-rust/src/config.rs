use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Noise suppression: `Sup1` cleans the microphone with RNNoise; `Sup2` also mutes
/// non-speech and cleans the converted output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Noise {
    #[default]
    Off,
    Sup1,
    Sup2,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F0Detector {
    /// The RMVPE model named by the manifest.
    #[default]
    Rmvpe,
    /// CREPE ONNX models, `crepe_full.onnx` or `crepe_tiny.onnx` beside the RMVPE file.
    CrepeFull,
    CrepeTiny,
    /// WORLD's Harvest: accurate and slow.
    Harvest,
    /// WORLD's DIO: fast and rough.
    Dio,
    /// Built-in YIN: no model run, less robust than RMVPE.
    Yin,
}
/// Where ONNX sessions run. Applied when a model is loaded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    #[default]
    Cpu,
    /// CUDA execution provider; needs a GPU build of ONNX Runtime.
    Gpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub bypass: bool,
    pub input_gain: f32,
    pub output_gain: f32,
    pub semitones: f32,
    pub speaker_id: i64,
    pub block_ms: u32,
    pub context_ms: u32,
    pub overlap_ms: u32,
    pub search_ms: u32,
    pub silence_threshold: f32,
    pub index_ratio: f32,
    pub protect: f32,
    pub noise: Noise,
    /// Acoustic echo cancellation for use with speakers instead of headphones.
    pub echo: bool,
    pub f0_detector: F0Detector,
    pub device: Device,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            bypass: true,
            input_gain: 1.0,
            output_gain: 1.0,
            semitones: 0.0,
            speaker_id: 0,
            block_ms: 320,
            context_ms: 500,
            overlap_ms: 40,
            search_ms: 12,
            silence_threshold: 0.00001,
            index_ratio: 0.0,
            protect: 0.5,
            noise: Noise::Off,
            echo: false,
            f0_detector: F0Detector::Rmvpe,
            device: Device::Cpu,
        }
    }
}
impl Settings {
    pub fn validate(&self) -> Result<()> {
        if !(20..=1000).contains(&self.block_ms)
            || self.context_ms > 5000
            || self.overlap_ms == 0
            || self.overlap_ms > self.block_ms
            || self.search_ms > 100
        {
            bail!("Invalid block/context/overlap/search duration");
        }
        if !self.input_gain.is_finite()
            || !(0.0..=4.0).contains(&self.input_gain)
            || !self.output_gain.is_finite()
            || !(0.0..=4.0).contains(&self.output_gain)
            || !self.semitones.is_finite()
            || !(-48.0..=48.0).contains(&self.semitones)
            || !self.index_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.index_ratio)
            || !self.protect.is_finite()
            || !(0.0..=0.5).contains(&self.protect)
            || !self.silence_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.silence_threshold)
            || self.speaker_id < 0
        {
            bail!("Invalid gain, pitch, speaker, index ratio, protect, or silence threshold");
        }
        Ok(())
    }
}

/// Explicit model contract: arbitrary ONNX exports are not interchangeable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelManifest {
    pub name: String,
    pub model: PathBuf,
    pub embedder: PathBuf,
    pub pitch: PathBuf,
    pub sample_rate: u32,
    pub channels: usize,
    pub use_f0: bool,
    pub embedder_input: String,
    pub embedder_output: String,
    pub output_scale: f32,
    pub feature_bank: Option<PathBuf>,
    pub intra_threads: usize,
}
impl Default for ModelManifest {
    fn default() -> Self {
        Self {
            name: "RVC voice".into(),
            model: "voice.onnx".into(),
            embedder: "contentvec.onnx".into(),
            pitch: "rmvpe.onnx".into(),
            sample_rate: 40000,
            channels: 768,
            use_f0: true,
            embedder_input: "source".into(),
            embedder_output: "".into(),
            output_scale: 1.0,
            feature_bank: None,
            intra_threads: 4,
        }
    }
}
impl ModelManifest {
    pub fn load(path: &Path) -> Result<Self> {
        let mut model: Self = serde_json::from_slice(
            &std::fs::read(path).with_context(|| format!("Read {}", path.display()))?,
        )?;
        if ![16000, 24000, 32000, 40000, 44100, 48000].contains(&model.sample_rate)
            || ![256, 768].contains(&model.channels)
            || !model.output_scale.is_finite()
            || model.output_scale <= 0.0
            || !(1..=32).contains(&model.intra_threads)
        {
            bail!("Invalid model sample rate, embedding channels, or output scale");
        }
        let base = path.parent().unwrap_or(Path::new("."));
        for file in [&mut model.model, &mut model.embedder] {
            if file.is_relative() {
                *file = base.join(&*file);
            }
            if !file.is_file() {
                bail!("Model file not found: {}", file.display());
            }
        }
        if model.use_f0 {
            if model.pitch.is_relative() {
                model.pitch = base.join(&model.pitch);
            }
            if !model.pitch.is_file() {
                bail!("Pitch model not found: {}", model.pitch.display());
            }
        }
        if let Some(file) = &mut model.feature_bank {
            if file.is_relative() {
                *file = base.join(&*file);
            }
        }
        Ok(model)
    }
}
