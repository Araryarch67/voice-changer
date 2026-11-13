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
    /// FCPE ONNX model, `fcpe.onnx` beside the RMVPE file.
    Fcpe,
    /// WORLD's Harvest: accurate and slow.
    Harvest,
    /// WORLD's DIO: fast and rough.
    Dio,
    /// Built-in YIN: no model run, less robust than RMVPE.
    Yin,
}
impl F0Detector {
    /// The model file this detector needs beside the RMVPE model, if any.
    pub fn model_file(self) -> Option<&'static str> {
        match self {
            F0Detector::CrepeFull => Some("crepe_full.onnx"),
            F0Detector::CrepeTiny => Some("crepe_tiny.onnx"),
            F0Detector::Fcpe => Some("fcpe.onnx"),
            F0Detector::Rmvpe | F0Detector::Harvest | F0Detector::Dio | F0Detector::Yin => None,
        }
    }
}
/// Where ONNX sessions run. Applied when a model is loaded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    #[default]
    Cpu,
    /// CUDA execution provider (NVIDIA); needs a CUDA build of ONNX Runtime with CUDA
    /// and cuDNN installed.
    Gpu,
    /// DirectML execution provider: any DirectX 12 GPU on Windows, no extra install.
    DirectMl,
}
impl Device {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU (CUDA)",
            Self::DirectMl => "GPU (DirectML)",
        }
    }
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
            // -50 dBFS on the loudest 20 ms: speech passes, a quiet room does not, so
            // the GPU rests between sentences. The reference's 0.00001 (-100 dB)
            // converts silence too.
            silence_threshold: 0.0032,
            index_ratio: 0.0,
            protect: 0.5,
            noise: Noise::Off,
            echo: false,
            f0_detector: F0Detector::Rmvpe,
            device: Device::Cpu,
        }
    }
}
/// The longest chunk: the reference client's CHUNK 1024 (131072 samples at 48 kHz).
/// Its longer choices, 5 seconds and up, are of no use live.
pub const MAX_BLOCK_MS: u32 = 2731;
impl Settings {
    pub fn validate(&self) -> Result<()> {
        if !(20..=MAX_BLOCK_MS).contains(&self.block_ms)
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

/// Gives a voice a new index (copied into its folder) or, with `None`, none. The
/// manifest is restored if the result does not load.
pub fn set_feature_bank(manifest: &Path, index: Option<&Path>) -> Result<()> {
    let original = std::fs::read_to_string(manifest)
        .with_context(|| format!("Read {}", manifest.display()))?;
    let mut value: serde_json::Value =
        serde_json::from_str(&original).context("The voice's model.json is not valid JSON")?;
    let folder = manifest.parent().context("The manifest has no folder")?;
    let object = value
        .as_object_mut()
        .context("The voice's model.json is not an object")?;
    match index {
        Some(index) => {
            anyhow::ensure!(
                index
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("index")),
                "Choose a .index file"
            );
            let name = index.file_name().context("The index has no file name")?;
            let target = folder.join(name);
            if index.canonicalize().ok() != target.canonicalize().ok() {
                std::fs::copy(index, &target)
                    .with_context(|| format!("Copy {}", index.display()))?;
            }
            object.insert(
                "feature_bank".into(),
                serde_json::Value::String(name.to_string_lossy().into_owned()),
            );
        }
        None => {
            object.remove("feature_bank");
        }
    }
    std::fs::write(manifest, serde_json::to_string_pretty(&value)?)?;
    if let Err(error) = ModelManifest::load(manifest) {
        std::fs::write(manifest, original)?;
        return Err(error.context("The voice did not load with that index; nothing was changed"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changes_and_removes_a_voice_index() {
        let root = std::env::temp_dir().join(format!("vc-bank-{}", std::process::id()));
        let (voice, shared) = (root.join("voice"), root.join("shared"));
        std::fs::create_dir_all(&voice).unwrap();
        std::fs::create_dir_all(&shared).unwrap();
        for file in [
            voice.join("voice.onnx"),
            shared.join("e.onnx"),
            shared.join("p.onnx"),
        ] {
            std::fs::write(file, b"").unwrap();
        }
        let manifest = voice.join("model.json");
        let mut model = ModelManifest {
            model: "voice.onnx".into(),
            embedder: "../shared/e.onnx".into(),
            pitch: "../shared/p.onnx".into(),
            ..ModelManifest::default()
        };
        model.feature_bank = None;
        std::fs::write(&manifest, serde_json::to_string(&model).unwrap()).unwrap();
        let elsewhere = root.join("added_IVF1_Flat.index");
        std::fs::write(&elsewhere, b"bank").unwrap();

        set_feature_bank(&manifest, Some(&elsewhere)).unwrap();
        assert!(voice.join("added_IVF1_Flat.index").is_file());
        let loaded = ModelManifest::load(&manifest).unwrap();
        assert_eq!(
            loaded.feature_bank,
            Some(voice.join("added_IVF1_Flat.index"))
        );

        set_feature_bank(&manifest, None).unwrap();
        assert_eq!(ModelManifest::load(&manifest).unwrap().feature_bank, None);
        assert!(set_feature_bank(&manifest, Some(&voice.join("voice.onnx"))).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
