use crate::{
    config::{Device, F0Detector, ModelManifest, Settings},
    dsp::{self, Sola},
    engine::Converter,
    retrieval::FeatureBank,
};
use anyhow::{bail, Context, Result};
use ort::{
    execution_providers::{CUDAExecutionProvider, DirectMLExecutionProvider},
    session::Session,
    value::Tensor,
};
use std::path::Path;

/// Pitch is detected again for this much audio before the part that is converted
/// (10 ms frames), so the detector has some context; older frames keep the pitch
/// found in earlier blocks, as the reference's "silence front" does.
const PITCH_MARGIN_FRAMES: usize = 16;

fn open_session(file: &Path, threads: usize, device: Device) -> Result<Session> {
    anyhow::ensure!(
        crate::runtime::available_devices().contains(&device),
        "{} is not available with this ONNX Runtime on this system; choose another device",
        device.label()
    );
    let mut builder = Session::builder()?.with_intra_threads(threads)?;
    // A chosen GPU that cannot start is an error, never a quiet switch to the CPU.
    builder = match device {
        Device::Cpu => builder,
        Device::Gpu => builder
            // TF32 makes RTX GPUs faster but changes the voice: output came out about
            // 12% quieter than on the CPU. Full precision matches the CPU and is still
            // several times faster than real time.
            .with_execution_providers([CUDAExecutionProvider::default()
                .with_tf32(false)
                .build()
                .error_on_failure()])
            .context("GPU (CUDA) is selected but CUDA could not start; press Set up CUDA in Settings to download its NVIDIA libraries (or install CUDA 12 and cuDNN 9), or choose another device")?,
        // DirectML needs sequential execution without memory patterns.
        Device::DirectMl => builder
            .with_memory_pattern(false)?
            .with_parallel_execution(false)?
            .with_execution_providers([DirectMLExecutionProvider::default()
                .with_device_id(crate::runtime::directml_device())
                .build()
                .error_on_failure()])
            .context("GPU (DirectML) is selected but DirectML could not start; update the graphics driver (DirectX 12 is required) or choose another device")?,
    };
    Ok(builder.commit_from_file(file)?)
}

/// RVC ONNX with waveform-in ContentVec and waveform-in RMVPE exports.
pub struct Rvc {
    manifest: ModelManifest,
    voice: Session,
    embedder: Session,
    pitch: Option<Session>,
    /// Loaded on first use of a CREPE or FCPE detector, with the detector it was
    /// loaded for.
    detector_model: Option<(F0Detector, Session)>,
    /// Blocks still converted after the input fell under the noise gate.
    gate_hold: u32,
    /// Whether the noise gate is currently open (needs a smooth fade-out before muting).
    gate_open: bool,
    device: Device,
    history: Vec<f32>,
    samples_seen: i64,
    pitch_tracker: crate::pitch::PitchTracker,
    sola: Sola,
    geometry: Option<(u32, u32, u32, u32, u32)>,
    bank: Option<FeatureBank>,
    /// Detected F0 for every 10 ms frame of `history`, before smoothing and the
    /// pitch shift, and the detector that found it; reused for older frames.
    pitch_track: Vec<f32>,
    pitch_track_detector: Option<F0Detector>,
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
        let started = std::time::Instant::now();
        let manifest = ModelManifest::load(path)?;
        log::info!(
            "Loading voice \"{}\" on {} ({} Hz, {} features, {})",
            manifest.name,
            device.label(),
            manifest.sample_rate,
            manifest.channels,
            if manifest.use_f0 {
                "with pitch"
            } else {
                "no pitch"
            }
        );
        crate::runtime::initialize()?;
        if device == Device::Gpu {
            crate::runtime::preload_gpu_libraries();
        }
        let session = |file: &Path| {
            let opened = std::time::Instant::now();
            let session = open_session(file, manifest.intra_threads, device);
            if session.is_ok() {
                log::info!(
                    "  loaded {} in {} ms",
                    file.file_name().unwrap_or_default().to_string_lossy(),
                    opened.elapsed().as_millis()
                );
            }
            session
        };
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
        match &manifest.feature_bank {
            Some(file) => log::info!(
                "  index {}",
                file.file_name().unwrap_or_default().to_string_lossy()
            ),
            None => log::info!("  no index"),
        }
        log::info!("Voice ready in {} ms", started.elapsed().as_millis());
        Ok(Self {
            manifest,
            voice,
            embedder,
            pitch,
            detector_model: None,
            gate_hold: 0,
            gate_open: false,
            device,
            history: Vec::new(),
            samples_seen: 0,
            pitch_tracker: crate::pitch::PitchTracker::default(),
            sola: Sola::default(),
            geometry: None,
            bank,
            pitch_track: Vec::new(),
            pitch_track_detector: None,
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

    /// The session of `detector`'s own model (`F0Detector::model_file` beside the
    /// RMVPE model), loaded on first use; `input` is the input it must take.
    fn detector_model(&mut self, detector: F0Detector, input: &str) -> Result<&mut Session> {
        if self
            .detector_model
            .as_ref()
            .is_none_or(|(loaded, _)| *loaded != detector)
        {
            let name = detector.model_file().expect("detectors with a model");
            let file = self.manifest.pitch.with_file_name(name);
            anyhow::ensure!(
                file.is_file(),
                "{detector:?} model not found: {}. Put {name} beside the RMVPE model, download it in Settings under Processing, or choose another F0 detector",
                file.display()
            );
            let model = open_session(&file, self.manifest.intra_threads, self.device)
                .with_context(|| format!("Load {name}"))?;
            anyhow::ensure!(
                model.inputs.iter().any(|i| i.name == input),
                "{name} must take an `{input}` input"
            );
            self.detector_model = Some((detector, model));
        }
        Ok(&mut self.detector_model.as_mut().expect("loaded above").1)
    }

    /// F0 per frame for `frames` 10 ms frames of `audio`, from `detector`.
    fn detect(&mut self, audio: &[f32], frames: usize, detector: F0Detector) -> Result<Vec<f32>> {
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
                let model = self.detector_model(detector, "frames")?;
                let (input, count) = crate::pitch::crepe_frames(audio);
                let input = Tensor::from_array(([count, 1024], input.into_boxed_slice()))?;
                let output = model.run(ort::inputs! { "frames" => input })?;
                let (_, activations) = output[0].try_extract_tensor::<f32>()?;
                crate::pitch::crepe_decode(activations)
            }
            F0Detector::Fcpe => {
                let model = self.detector_model(detector, "audio")?;
                let input =
                    Tensor::from_array(([1, audio.len()], audio.to_vec().into_boxed_slice()))?;
                let output = model.run(ort::inputs! { "audio" => input })?;
                let (_, salience) = output[0].try_extract_tensor::<f32>()?;
                crate::pitch::fcpe_decode(salience)
            }
            F0Detector::Rmvpe => {
                let model = self.pitch.as_mut().expect("checked by the caller");
                let waveform =
                    Tensor::from_array(([1, audio.len()], audio.to_vec().into_boxed_slice()))?;
                let output = if model.inputs.iter().any(|i| i.name == "threshold") {
                    let threshold = Tensor::from_array(([1], vec![0.03_f32].into_boxed_slice()))?;
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
        Ok((0..frames)
            .map(|i| detected[(i * detected.len() / frames).min(detected.len() - 1)])
            .collect())
    }

    /// Coarse and fine pitch for `frames` frames of `audio`. `fresh_frames` is how
    /// many frames at the end are new since the last block (`None` when the blocks
    /// do not line up with 10 ms frames) and `tail_frames` how many at the end must
    /// be detected again; the rest come from the previous block's track.
    fn pitches(
        &mut self,
        audio: &[f32],
        history_start: i64,
        frames: usize,
        fresh_frames: Option<usize>,
        tail_frames: usize,
        settings: &Settings,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        if self.pitch.is_none() {
            return Ok((Vec::new(), Vec::new()));
        }
        let detector = settings.f0_detector;
        let tail = tail_frames.min(frames);
        let reusable = fresh_frames.filter(|&fresh| {
            self.pitch_track_detector == Some(detector)
                && self.pitch_track.len() == frames
                && fresh <= tail
                && audio.len() >= tail * 160
        });
        let track = match reusable {
            Some(fresh) if tail < frames => {
                let detected = self.detect(&audio[audio.len() - tail * 160..], tail, detector)?;
                let mut track = self.pitch_track[fresh..fresh + frames - tail].to_vec();
                track.extend(detected);
                track
            }
            _ => self.detect(audio, frames, detector)?,
        };
        self.pitch_track.clone_from(&track);
        self.pitch_track_detector = Some(detector);
        let clean = crate::pitch::stabilize_f0(&track);
        let pitchf = self.pitch_tracker.process(
            history_start,
            &clean,
            frames,
            settings.semitones,
            settings.pitch_correct,
        );
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
        let mut retrieved = vec![0.0_f32; channels];
        // Frames come in identical pairs (x2 interpolation); search once per pair.
        let mut searched: Option<usize> = None;
        for (i, frame) in features.chunks_exact_mut(channels).enumerate() {
            let unvoiced = !pitch.is_empty() && pitch[i] <= 0.0;
            let protection = if unvoiced && settings.protect < 0.5 {
                settings.protect
            } else {
                1.0
            };
            let ratio = settings.index_ratio * protection;
            if ratio <= 0.0 {
                continue;
            }
            if searched != Some(i / 2) {
                bank.blend_nearest(frame, 8, &mut retrieved);
                searched = Some(i / 2);
            }
            for (sample, &blended) in frame.iter_mut().zip(&retrieved) {
                *sample = *sample * (1.0 - ratio) + blended * ratio;
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
        // New 10 ms frames this block; pitch for older ones is reused only when the
        // block is a whole number of frames, so frames line up between blocks.
        let fresh_frames = new_audio
            .len()
            .is_multiple_of(160)
            .then_some(new_audio.len() / 160);
        let convert_frames = (new_audio.len()
            + settings.overlap_ms as usize * 16
            + settings.search_ms as usize * 16)
            .div_ceil(160);
        let overlap = settings.overlap_ms as usize * 16;
        let search = settings.search_ms as usize * 16;
        let context = settings.context_ms as usize * 16;
        let required = (new_audio.len() + overlap + search + context).div_ceil(160) * 160;
        self.samples_seen += new_audio.len() as i64;
        let history_start = self.samples_seen - required as i64;
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
        let mut closing = false;
        if loudest > 0.0 && loudest >= settings.silence_threshold {
            self.gate_hold = 1;
            self.gate_open = true;
        } else if self.gate_hold > 0 {
            self.gate_hold -= 1;
        } else if self.gate_open {
            closing = true;
            self.gate_open = false;
        } else {
            self.sola.reset();
            // The pitch track no longer lines up with the history after a gap.
            self.pitch_track.clear();
            self.pitch_tracker.reset();
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
        let (pitch, pitchf) = self.pitches(
            &audio,
            history_start,
            frames,
            fresh_frames,
            convert_frames + PITCH_MARGIN_FRAMES,
            settings,
        )?;
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
        log::trace!(
            "block: features {feature_ms:.1} ms, pitch {pitch_ms:.1} ms, voice {:.1} ms",
            voice_start.elapsed().as_secs_f64() * 1000.0
        );
        let converted = dsp::resample(&values, self.manifest.sample_rate, rate);
        let mut out = self.sola.process(
            &converted,
            input.len(),
            settings.overlap_ms as usize * rate as usize / 1000,
            settings.search_ms as usize * rate as usize / 1000,
        )?;
        if closing {
            // Smoothly fade the tail block to zero before muting, avoiding a click.
            dsp::fade(&mut out, 1.0, 0.0);
            self.sola.reset();
        }
        Ok(out)
    }
    fn reset(&mut self) {
        self.history.clear();
        self.samples_seen = 0;
        self.pitch_tracker.reset();
        self.sola.reset();
        self.geometry = None;
        self.gate_hold = 0;
        self.gate_open = false;
        self.pitch_track.clear();
        self.pitch_track_detector = None;
    }
}
