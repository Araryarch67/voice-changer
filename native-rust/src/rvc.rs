use crate::{
    config::{Device, F0Detector, ModelManifest, Settings},
    dsp::{self, Sola},
    engine::Converter,
    retrieval::FeatureBank,
};
use anyhow::{bail, Context, Result};
use ort::{execution_providers::CUDAExecutionProvider, session::Session, value::Tensor};
use std::path::Path;

fn open_session(file: &Path, threads: usize, device: Device) -> Result<Session> {
    let mut builder = Session::builder()?.with_intra_threads(threads)?;
    if device == Device::Gpu {
        builder = builder
            .with_execution_providers([CUDAExecutionProvider::default()
                .build()
                .error_on_failure()])
            .context("GPU is selected but this ONNX Runtime cannot use CUDA; install a GPU build or switch the device to CPU")?;
    }
    Ok(builder.commit_from_file(file)?)
}

/// RVC ONNX with waveform-in ContentVec and waveform-in RMVPE exports.
pub struct Rvc {
    manifest: ModelManifest,
    voice: Session,
    embedder: Session,
    pitch: Option<Session>,
    /// Loaded on first use of a CREPE detector, with the detector it was loaded for.
    crepe: Option<(F0Detector, Session)>,
    /// Blocks still converted after the input fell under the noise gate.
    gate_hold: u32,
    device: Device,
    history: Vec<f32>,
    sola: Sola,
    geometry: Option<(u32, u32, u32, u32, u32)>,
    bank: Option<FeatureBank>,
}
impl Rvc {
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_on(path, Device::Cpu)
    }
    pub fn load_on(path: &Path, device: Device) -> Result<Self> {
        // ort load-dynamic can panic when its shared library cannot be located.
        std::panic::catch_unwind(|| Self::load_inner(path, device)).map_err(|_| anyhow::anyhow!(
            "Cannot initialize ONNX Runtime. Set ORT_DYLIB_PATH to a compatible ONNX Runtime 1.22 library."
        ))?
    }
    fn load_inner(path: &Path, device: Device) -> Result<Self> {
        let manifest = ModelManifest::load(path)?;
        crate::runtime::initialize()?;
        if device == Device::Gpu {
            crate::runtime::preload_gpu_libraries();
        }
        let session = |file: &Path| open_session(file, manifest.intra_threads, device);
        let voice = session(&manifest.model).context("Load RVC ONNX")?;
        let embedder = session(&manifest.embedder).context("Load ContentVec/HuBERT ONNX")?;
        for required in ["feats", "p_len", "sid"] {
            anyhow::ensure!(
                voice.inputs.iter().any(|i| i.name == required),
                "RVC export missing input {required}"
            );
        }
        let expected = (if manifest.use_f0 { 5 } else { 3 })
            + usize::from(voice.inputs.iter().any(|i| i.name == "convert_length"));
        anyhow::ensure!(
            voice.inputs.len() == expected,
            "Unsupported RVC inputs: expected {expected}"
        );
        if manifest.use_f0 {
            for required in ["pitch", "pitchf"] {
                anyhow::ensure!(
                    voice.inputs.iter().any(|i| i.name == required),
                    "RVC export missing input {required}"
                );
            }
        }
        anyhow::ensure!(
            embedder
                .inputs
                .iter()
                .any(|i| i.name == manifest.embedder_input),
            "Embedder input '{}' not found",
            manifest.embedder_input
        );
        anyhow::ensure!(
            embedder
                .inputs
                .iter()
                .all(|i| i.name == manifest.embedder_input || i.name == "padding_mask"),
            "Unsupported embedder inputs; export a waveform-in graph with optional padding_mask"
        );
        let pitch = if manifest.use_f0 {
            let model = session(&manifest.pitch).context("Load RMVPE ONNX")?;
            anyhow::ensure!(model.inputs.iter().any(|i| i.name == "waveform")
                && model.inputs.iter().all(|i| i.name == "waveform" || i.name == "threshold"),
                "RMVPE must accept waveform [1,N] and optional threshold [1]; mel-input exports are not supported");
            Some(model)
        } else {
            None
        };
        let bank = manifest
            .feature_bank
            .as_deref()
            .map(|file| FeatureBank::load(file, manifest.channels))
            .transpose()?;
        Ok(Self {
            manifest,
            voice,
            embedder,
            pitch,
            crepe: None,
            gate_hold: 0,
            device,
            history: Vec::new(),
            sola: Sola::default(),
            geometry: None,
            bank,
        })
    }

    fn features(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        let source = Tensor::from_array(([1, audio.len()], audio.to_vec().into_boxed_slice()))?;
        let has_mask = self
            .embedder
            .inputs
            .iter()
            .any(|i| i.name == "padding_mask");
        let outputs = if has_mask {
            let mask = Tensor::from_array((
                [1, audio.len()],
                vec![false; audio.len()].into_boxed_slice(),
            ))?;
            self.embedder.run(ort::inputs! { self.manifest.embedder_input.as_str() => source, "padding_mask" => mask })?
        } else {
            self.embedder
                .run(ort::inputs! { self.manifest.embedder_input.as_str() => source })?
        };
        let output = if self.manifest.embedder_output.is_empty() {
            &outputs[0]
        } else {
            outputs
                .get(self.manifest.embedder_output.as_str())
                .context("Embedder output not found")?
        };
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .context("Embedder must output float32 features")?;
        anyhow::ensure!(shape.len() == 3 && shape[0] == 1 && shape[2] == self.manifest.channels as i64,
            "Expected embedding [1,T,{}], got {:?}. RVC v1 needs projected layer 9 / 256 channels; v2 layer 12 / 768 channels.", self.manifest.channels, shape);
        anyhow::ensure!(
            !values.is_empty() && values.iter().all(|v| v.is_finite()),
            "Empty/non-finite embedding"
        );
        Ok(values.to_vec())
    }

    fn pitches(
        &mut self,
        audio: &[f32],
        frames: usize,
        semitones: f32,
        detector: F0Detector,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        if self.pitch.is_none() {
            return Ok((Vec::new(), Vec::new()));
        }
        let detected = match detector {
            F0Detector::Yin => crate::pitch::yin(audio, frames),
            #[cfg(feature = "world")]
            F0Detector::Harvest => crate::pitch::harvest(audio),
            #[cfg(feature = "world")]
            F0Detector::Dio => crate::pitch::dio(audio),
            #[cfg(not(feature = "world"))]
            F0Detector::Harvest | F0Detector::Dio => {
                bail!("This build does not include the Harvest and DIO detectors")
            }
            F0Detector::CrepeFull | F0Detector::CrepeTiny => {
                if self
                    .crepe
                    .as_ref()
                    .is_none_or(|(loaded, _)| *loaded != detector)
                {
                    let name = if detector == F0Detector::CrepeFull {
                        "crepe_full.onnx"
                    } else {
                        "crepe_tiny.onnx"
                    };
                    let file = self.manifest.pitch.with_file_name(name);
                    anyhow::ensure!(
                        file.is_file(),
                        "CREPE model not found: {}. Put {name} beside the RMVPE model, or choose another F0 detector",
                        file.display()
                    );
                    let model = open_session(&file, self.manifest.intra_threads, self.device)
                        .context("Load CREPE ONNX")?;
                    anyhow::ensure!(
                        model.inputs.iter().any(|i| i.name == "frames"),
                        "CREPE model must take a `frames` input of [N,1024]"
                    );
                    self.crepe = Some((detector, model));
                }
                let (_, model) = self.crepe.as_mut().expect("loaded above");
                let (input, count) = crate::pitch::crepe_frames(audio);
                let input = Tensor::from_array(([count, 1024], input.into_boxed_slice()))?;
                let output = model.run(ort::inputs! { "frames" => input })?;
                let (_, activations) = output[0].try_extract_tensor::<f32>()?;
                crate::pitch::crepe_decode(activations)
            }
            F0Detector::Rmvpe => {
                let model = self.pitch.as_mut().expect("checked above");
                let waveform =
                    Tensor::from_array(([1, audio.len()], audio.to_vec().into_boxed_slice()))?;
                let output = if model.inputs.iter().any(|i| i.name == "threshold") {
                    let threshold = Tensor::from_array(([1], vec![0.3_f32].into_boxed_slice()))?;
                    model.run(ort::inputs! { "waveform" => waveform, "threshold" => threshold })?
                } else {
                    model.run(ort::inputs! { "waveform" => waveform })?
                };
                output[0].try_extract_tensor::<f32>()?.1.to_vec()
            }
        };
        anyhow::ensure!(
            !detected.is_empty() && detected.iter().all(|v| v.is_finite() && *v >= 0.0),
            "Invalid pitch detector output"
        );
        let shift = 2.0_f32.powf(semitones / 12.0);
        let pitchf: Vec<_> = (0..frames)
            .map(|i| detected[(i * detected.len() / frames).min(detected.len() - 1)] * shift)
            .collect();
        let pitch = pitchf.iter().map(|v| dsp::coarse_pitch(*v)).collect();
        Ok((pitch, pitchf))
    }

    fn retrieve(&self, features: &mut [f32], pitch: &[f32], settings: &Settings) -> Result<()> {
        if settings.index_ratio == 0.0 {
            return Ok(());
        }
        let bank = self
            .bank
            .as_ref()
            .context("Index ratio is nonzero but no feature_bank is configured")?;
        let channels = self.manifest.channels;
        for (i, frame) in features.chunks_exact_mut(channels).enumerate() {
            let best = bank.nearest(frame);
            let unvoiced = !pitch.is_empty() && pitch[i] <= 0.0;
            let protection = if unvoiced && settings.protect < 0.5 {
                settings.protect
            } else {
                1.0
            };
            let ratio = settings.index_ratio * protection;
            for (sample, retrieved) in frame.iter_mut().zip(best) {
                *sample = *sample * (1.0 - ratio) + retrieved * ratio;
            }
        }
        Ok(())
    }
}
impl Converter for Rvc {
    fn process(&mut self, input: &[f32], rate: u32, settings: &Settings) -> Result<Vec<f32>> {
        let geometry = (
            rate,
            settings.block_ms,
            settings.context_ms,
            settings.overlap_ms,
            settings.search_ms,
        );
        if self.geometry != Some(geometry) {
            self.reset();
            self.geometry = Some(geometry);
        }
        let new_audio = dsp::resample(input, rate, 16000);
        let overlap = settings.overlap_ms as usize * 16;
        let search = settings.search_ms as usize * 16;
        let context = settings.context_ms as usize * 16;
        let required = (new_audio.len() + overlap + search + context).div_ceil(160) * 160;
        self.history.extend(new_audio);
        if self.history.len() < required {
            let mut padding = vec![0.0; required - self.history.len()];
            padding.append(&mut self.history);
            self.history = padding;
        } else if self.history.len() > required {
            self.history.drain(..self.history.len() - required);
        }
        // Gate on the loudest 20 ms rather than the block average, so a word that
        // starts or ends inside the block is kept, and convert one more block after
        // the last loud one so trailing sounds are not cut.
        let loudest = input
            .chunks((rate as usize / 50).max(1))
            .map(dsp::rms)
            .fold(0.0, f32::max);
        if loudest >= settings.silence_threshold {
            self.gate_hold = 1;
        } else if self.gate_hold > 0 {
            self.gate_hold -= 1;
        } else {
            self.sola.reset();
            return Ok(vec![0.0; input.len()]);
        }
        let audio = self.history.clone();
        let profile_start = std::time::Instant::now();
        let extracted = self.features(&audio)?;
        let feature_ms = profile_start.elapsed().as_secs_f64() * 1000.0;
        let source_frames = extracted.len() / self.manifest.channels;
        let frames = (required / 160).min(source_frames * 2);
        anyhow::ensure!(frames > 0, "Embedding contains no frames");
        // Matches nearest-neighbor x2 interpolation in the reference RVC pipeline.
        let mut features = Vec::with_capacity(frames * self.manifest.channels);
        for frame in 0..frames {
            let offset = (frame / 2) * self.manifest.channels;
            features.extend_from_slice(&extracted[offset..offset + self.manifest.channels]);
        }
        let pitch_start = std::time::Instant::now();
        let (pitch, pitchf) =
            self.pitches(&audio, frames, settings.semitones, settings.f0_detector)?;
        let pitch_ms = pitch_start.elapsed().as_secs_f64() * 1000.0;
        self.retrieve(&mut features, &pitchf, settings)?;
        let feats = Tensor::from_array((
            [1, frames, self.manifest.channels],
            features.into_boxed_slice(),
        ))?;
        let p_len = Tensor::from_array(([1], vec![frames as i64].into_boxed_slice()))?;
        let sid = Tensor::from_array(([1], vec![settings.speaker_id].into_boxed_slice()))?;
        let voice_start = std::time::Instant::now();
        let output = if self.manifest.use_f0 {
            let pitch = Tensor::from_array(([1, frames], pitch.into_boxed_slice()))?;
            let pitchf = Tensor::from_array(([1, frames], pitchf.into_boxed_slice()))?;
            if self.voice.inputs.iter().any(|i| i.name == "convert_length") {
                let convert_frames =
                    (input.len() as u64 * 16000 / rate as u64 + overlap as u64 + search as u64)
                        .div_ceil(160);
                let convert_samples = convert_frames * self.manifest.sample_rate as u64 / 100;
                let convert_length =
                    Tensor::from_array(([1], vec![convert_samples as i64].into_boxed_slice()))?;
                self.voice.run(ort::inputs! { "feats" => feats, "p_len" => p_len, "pitch" => pitch, "pitchf" => pitchf, "sid" => sid, "convert_length" => convert_length })?
            } else {
                self.voice.run(ort::inputs! { "feats" => feats, "p_len" => p_len, "pitch" => pitch, "pitchf" => pitchf, "sid" => sid })?
            }
        } else {
            if settings.semitones != 0.0 {
                bail!("Pitch shifting requires an f0-enabled RVC model");
            }
            self.voice
                .run(ort::inputs! { "feats" => feats, "p_len" => p_len, "sid" => sid })?
        };
        let (_, values) = output[0]
            .try_extract_tensor::<f32>()
            .context("RVC must output float32 audio")?;
        anyhow::ensure!(
            values.iter().all(|v| v.is_finite()),
            "Non-finite RVC output"
        );
        let values: Vec<f32> = values
            .iter()
            .map(|v| v * self.manifest.output_scale)
            .collect();
        drop(output);
        if std::env::var_os("VC_PROFILE").is_some() {
            eprintln!(
                "features={feature_ms:.1}ms pitch={pitch_ms:.1}ms voice={:.1}ms",
                voice_start.elapsed().as_secs_f64() * 1000.0
            );
        }
        let converted = dsp::resample(&values, self.manifest.sample_rate, rate);
        self.sola.process(
            &converted,
            input.len(),
            settings.overlap_ms as usize * rate as usize / 1000,
            settings.search_ms as usize * rate as usize / 1000,
        )
    }
    fn reset(&mut self) {
        self.history.clear();
        self.sola.reset();
        self.geometry = None;
        self.gate_hold = 0;
    }
}
