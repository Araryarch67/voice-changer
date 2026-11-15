//! so-vits-svc voices (svc-develop-team/so-vits-svc 4.0, as the reference's
//! `SoVitsSvc40` engine runs them, and 4.1 checkpoints with the same architecture).
//!
//! A voice is a generator checkpoint (`G_*.pth`, its weights under `model`) and the
//! `config.json` it was trained with. Import fills one of the graph templates in
//! `templates/` (`tools/dev/make_sovits_templates.py`: the reference's
//! `SynthesizerTrn` inference path) with the checkpoint's weights, as RVC import does.
//! The graph takes ContentVec features, F0 with unvoiced frames filled in, the
//! voiced flags, the coarse pitch, the speaker and the text encoder's noise, and
//! returns 44.1 kHz audio from its NSF-HiFiGAN decoder.
//!
//! Features: `vec256l9` (ContentVec layer 9 through its final projection, 256
//! channels, so-vits-svc 4.0's default) and `vec768l12` (layer 12, 768 channels,
//! 4.1's default). The reference reads layer 9 for 768-channel models, but the
//! shared ContentVec model has no 768-channel layer-9 output and 4.1 trains on layer
//! 12, so 768-channel voices use layer 12.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// What the voice's graph needs, stored in its `metadata` entry as `{"sovits": …}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Info {
    pub sample_rate: u32,
    /// Audio samples per frame.
    pub hop: usize,
    /// ContentVec channels: 256 (layer 9, projected) or 768 (layer 12).
    pub ssl_dim: usize,
    /// Speaker embeddings in the checkpoint.
    pub speakers: usize,
    /// The speakers config.json names, by id.
    pub names: Vec<(String, i64)>,
}

#[cfg(feature = "export")]
pub use export::{export, looks_like_sovits, Exported};

#[cfg(feature = "export")]
mod export {
    use super::*;
    use crate::pth::{self, Tensor, Value};
    use anyhow::bail;
    use std::{collections::HashMap, path::Path};

    /// (ssl_dim, gin_channels, template): so-vits-svc 4.0's and 4.1's defaults.
    const TEMPLATES: [(usize, usize, &[u8]); 2] = [
        (
            256,
            256,
            include_bytes!("../templates/sovits-256-256.template.onnx.zz"),
        ),
        (
            768,
            768,
            include_bytes!("../templates/sovits-768-768.template.onnx.zz"),
        ),
    ];
    /// The config.json keys of the template's config list, in order.
    const CONFIG_KEYS: [&str; 15] = [
        "inter_channels",
        "hidden_channels",
        "filter_channels",
        "n_heads",
        "n_layers",
        "kernel_size",
        "p_dropout",
        "resblock",
        "resblock_kernel_sizes",
        "resblock_dilation_sizes",
        "upsample_rates",
        "upsample_initial_channel",
        "upsample_kernel_sizes",
        "ssl_dim",
        "gin_channels",
    ];

    /// What [`export`] wrote.
    #[derive(Debug, Clone)]
    pub struct Exported {
        pub model: std::path::PathBuf,
        pub info: Info,
    }

    /// The `config.json` beside `checkpoint`, if it is a so-vits-svc config (its
    /// model section has `ssl_dim`); RVC training folders hold a config.json too.
    pub fn looks_like_sovits(checkpoint: &Path) -> Option<std::path::PathBuf> {
        let config = checkpoint.parent()?.join("config.json");
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config).ok()?).ok()?;
        value.get("model")?.get("ssl_dim")?;
        Some(config)
    }

    fn number_equal(a: &serde_json::Value, b: &serde_json::Value) -> bool {
        match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => (x - y).abs() < 1e-9,
            _ => match (a.as_array(), b.as_array()) {
                (Some(x), Some(y)) => {
                    x.len() == y.len() && x.iter().zip(y).all(|(a, b)| number_equal(a, b))
                }
                _ => a == b,
            },
        }
    }

    /// Builds `voice.onnx` in `folder` from a so-vits-svc generator checkpoint and the
    /// config.json it was trained with.
    pub fn export(checkpoint: &Path, config: &Path, folder: &Path) -> Result<Exported> {
        let config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(config).with_context(|| format!("Read {}", config.display()))?,
        )
        .context("The so-vits-svc config.json is not valid JSON")?;
        let model = config
            .get("model")
            .context("config.json has no model section; is it the voice's so-vits-svc config?")?;
        let data = config
            .get("data")
            .context("config.json has no data section")?;
        let int = |section: &serde_json::Value, key: &str| -> Result<u64> {
            section
                .get(key)
                .and_then(|v| v.as_f64())
                .map(|v| v as u64)
                .with_context(|| format!("config.json has no {key}"))
        };
        let (sample_rate, hop) = (
            int(data, "sampling_rate")? as u32,
            int(data, "hop_length")? as usize,
        );
        ensure!(
            sample_rate == 44_100 && hop == 512,
            "This model is for {sample_rate} Hz with {hop} samples a frame; so-vits-svc voices at 44100 Hz and 512 are supported"
        );
        let ssl_dim = int(model, "ssl_dim")? as usize;
        let gin = int(model, "gin_channels")? as usize;
        if let Some(encoder) = model.get("speech_encoder").and_then(|v| v.as_str()) {
            ensure!(
                matches!((encoder, ssl_dim), ("vec256l9", 256) | ("vec768l12", 768)),
                "This model uses the {encoder} speech encoder; vec256l9 and vec768l12 (ContentVec) are supported"
            );
        }
        if let Some(vocoder) = model.get("vocoder_name").and_then(|v| v.as_str()) {
            ensure!(
                vocoder == "nsf-hifigan",
                "This model uses the {vocoder} vocoder; nsf-hifigan is supported"
            );
        }
        for option in [
            "vol_embedding",
            "use_depthwise_conv",
            "use_transformer_flow",
            "flow_share_parameter",
            "speaker_embedding",
        ] {
            ensure!(
                !model.get(option).and_then(|v| v.as_bool()).unwrap_or(false),
                "This model was trained with {option} on, which is not supported"
            );
        }
        if let Some(flows) = model.get("n_flow_layer").and_then(|v| v.as_u64()) {
            ensure!(
                flows == 4,
                "This model has {flows} flow layers; 4 are supported"
            );
        }
        let (.., compressed) = TEMPLATES
            .iter()
            .find(|t| (t.0, t.1) == (ssl_dim, gin))
            .with_context(|| format!("No so-vits-svc template for ssl_dim {ssl_dim} with gin_channels {gin}; 256/256 (4.0) and 768/768 (4.1) are supported"))?;

        let root = pth::load(checkpoint)?;
        let Some(Value::Dict(items)) = root.get("model") else {
            bail!("Not a so-vits-svc generator checkpoint: it has no model weights (choose the G_*.pth, not D_*.pth)");
        };
        let mut weights: HashMap<&str, &Tensor> = HashMap::new();
        for (key, value) in items {
            if let (Value::Str(key), Value::Tensor(tensor)) = (key, value) {
                weights.insert(key.as_str(), tensor);
            }
        }
        let speakers = weights
            .get("emb_g.weight")
            .map(|t| t.shape[0])
            .context("The checkpoint has no speaker embedding")?;
        let mut names: Vec<(String, i64)> = config
            .get("spk")
            .and_then(|v| v.as_object())
            .map(|spk| {
                spk.iter()
                    .filter_map(|(name, id)| Some((name.clone(), id.as_i64()?)))
                    .filter(|(_, id)| (0..speakers as i64).contains(id))
                    .collect()
            })
            .unwrap_or_default();
        names.sort_by_key(|(_, id)| *id);
        let info = Info {
            sample_rate,
            hop,
            ssl_dim,
            speakers,
            names,
        };
        let metadata = serde_json::json!({ "sovits": info }).to_string();
        let bytes = crate::export::fill_template(
            compressed,
            &weights,
            &["enc_q.", "f0_decoder."],
            &metadata,
            |template| {
                let differs: Vec<&str> = CONFIG_KEYS
                    .iter()
                    .zip(&template.config)
                    .filter(|(key, want)| {
                        model
                            .get(**key)
                            .is_none_or(|have| !number_equal(have, want))
                    })
                    .map(|(key, _)| *key)
                    .collect();
                ensure!(
                    differs.is_empty(),
                    "This model's architecture differs from the standard so-vits-svc one in {}",
                    differs.join(", ")
                );
                Ok(())
            },
        )?;
        std::fs::create_dir_all(folder)?;
        let model_path = folder.join("voice.onnx");
        std::fs::write(&model_path, bytes)
            .with_context(|| format!("Write {}", model_path.display()))?;
        Ok(Exported {
            model: model_path,
            info,
        })
    }
}

/// Frames for `samples` 16 kHz samples at the voice's hop.
pub fn frame_count(samples: usize, info: &Info) -> usize {
    let at_rate = samples as u64 * info.sample_rate as u64 / 16_000;
    (at_rate as usize).div_ceil(info.hop).max(1)
}

/// F0 for the graph, as the reference's `interpolate_f0`: voiced flags, and the
/// unvoiced frames filled linearly between voiced ones (held at the ends), shifted
/// by `semitones`.
pub fn f0_and_uv(detected: &[f32], semitones: f32) -> (Vec<f32>, Vec<i64>) {
    let uv: Vec<i64> = detected
        .iter()
        .map(|hz| i64::from(hz.is_finite() && *hz > 0.0))
        .collect();
    let voiced: Vec<usize> = (0..detected.len()).filter(|&i| uv[i] == 1).collect();
    let factor = 2f32.powf(semitones / 12.0);
    let f0 = (0..detected.len())
        .map(|i| {
            if voiced.is_empty() {
                return 0.0;
            }
            let next = voiced.partition_point(|&v| v < i);
            let hz = if next < voiced.len() && voiced[next] == i {
                detected[i]
            } else if next == 0 {
                detected[voiced[0]]
            } else if next == voiced.len() {
                detected[voiced[next - 1]]
            } else {
                let (a, b) = (voiced[next - 1], voiced[next]);
                detected[a] + (detected[b] - detected[a]) * (i - a) as f32 / (b - a) as f32
            };
            hz * factor
        })
        .collect();
    (f0, uv)
}

#[cfg(any(feature = "onnx", feature = "onnx-static"))]
pub use engine::SoVitsSvc;

#[cfg(any(feature = "onnx", feature = "onnx-static"))]
mod engine {
    use super::*;
    use crate::{
        config::{Device, ModelManifest, Settings},
        dsp::{self, Sola},
        engine::Converter,
        rvc::{embed, open_session, Detectors},
    };
    use ort::{session::Session, value::Tensor};
    use std::path::Path;

    /// A so-vits-svc voice, converting block by block like the RVC engine: the same
    /// history, noise gate and SOLA crossfade.
    pub struct SoVitsSvc {
        manifest: ModelManifest,
        info: Info,
        voice: Session,
        embedder: Session,
        detectors: Detectors,
        history: Vec<f32>,
        sola: Sola,
        geometry: Option<(u32, u32, u32, u32, u32)>,
        gate_hold: u32,
        gate_open: bool,
        noise: crate::diffusion_svc::Noise,
    }

    impl SoVitsSvc {
        pub fn load_on(path: &Path, device: Device) -> Result<Self> {
            std::panic::catch_unwind(|| Self::load_inner(path, device)).map_err(|_| {
                anyhow::anyhow!(
                    "Cannot initialize ONNX Runtime. Set ORT_DYLIB_PATH to a compatible ONNX Runtime 1.22 library."
                )
            })?
        }

        fn load_inner(path: &Path, device: Device) -> Result<Self> {
            let manifest = ModelManifest::load(path)?;
            crate::runtime::initialize()?;
            if device == Device::Gpu {
                crate::runtime::preload_gpu_libraries();
            }
            let threads = manifest.intra_threads;
            let open = |file: &Path, what: &str| {
                open_session(file, threads, device).with_context(|| format!("Load {what}"))
            };
            let voice = open(&manifest.model, "the so-vits-svc voice")?;
            let embedder = open(&manifest.embedder, "ContentVec")?;
            let rmvpe = open(&manifest.pitch, "RMVPE")?;
            let info: Info = {
                let metadata = voice.metadata()?;
                let text = metadata
                    .custom("metadata")?
                    .context("The voice has no so-vits-svc metadata; import it again")?;
                let value: serde_json::Value = serde_json::from_str(&text)?;
                serde_json::from_value(
                    value
                        .get("sovits")
                        .cloned()
                        .context("The voice has no so-vits-svc metadata; import it again")?,
                )
                .context("Read the voice's so-vits-svc metadata")?
            };
            for input in ["feats", "f0", "uv", "pitch", "sid", "noise"] {
                ensure!(
                    voice.inputs.iter().any(|i| i.name == input),
                    "The so-vits-svc voice has no `{input}` input; import it again"
                );
            }
            log::info!(
                "so-vits-svc voice \"{}\" on {} ({} channels, {} speakers)",
                manifest.name,
                device.label(),
                info.ssl_dim,
                info.speakers
            );
            let detectors = Detectors::new(rmvpe, &manifest.pitch, threads, device);
            Ok(Self {
                manifest,
                info,
                voice,
                embedder,
                detectors,
                history: Vec::new(),
                sola: Sola::default(),
                geometry: None,
                gate_hold: 0,
                gate_open: false,
                noise: crate::diffusion_svc::Noise::new(0x50_7175),
            })
        }

        /// The voice's audio (44.1 kHz) for 16 kHz `audio`.
        fn convert(&mut self, audio: &[f32], settings: &Settings) -> Result<Vec<f32>> {
            let info = self.info.clone();
            let frames = frame_count(audio.len(), &info);
            let features = embed(
                &mut self.embedder,
                &self.manifest.embedder_input,
                &self.manifest.embedder_output,
                audio,
                info.ssl_dim,
            )?;
            let feats = crate::diffusion_svc::stretch(&features, info.ssl_dim, frames);
            let detected = self.detectors.detect(audio, frames, settings.f0_detector)?;
            let (f0, uv) = f0_and_uv(&detected, settings.semitones);
            let pitch: Vec<i64> = f0.iter().map(|hz| dsp::coarse_pitch(*hz)).collect();
            let scale = settings.noise_scale;
            let noise: Vec<f32> = self
                .noise
                .fill(192 * frames)
                .into_iter()
                .map(|v| v * scale)
                .collect();
            let speaker = settings.speaker_id.clamp(0, info.speakers as i64 - 1);
            let outputs = self.voice.run(ort::inputs! {
                "feats" => Tensor::from_array(([1, frames, info.ssl_dim], feats.into_boxed_slice()))?,
                "f0" => Tensor::from_array(([1, frames], f0.into_boxed_slice()))?,
                "uv" => Tensor::from_array(([1, frames], uv.into_boxed_slice()))?,
                "pitch" => Tensor::from_array(([1, frames], pitch.into_boxed_slice()))?,
                "sid" => Tensor::from_array(([1], vec![speaker].into_boxed_slice()))?,
                "noise" => Tensor::from_array(([1, 192, frames], noise.into_boxed_slice()))?,
            })?;
            let (_, audio) = outputs[0].try_extract_tensor::<f32>()?;
            ensure!(
                audio.iter().all(|v| v.is_finite()),
                "Non-finite so-vits-svc output"
            );
            Ok(audio
                .iter()
                .map(|v| (v * self.manifest.output_scale).clamp(-1.0, 1.0))
                .collect())
        }
    }

    impl Converter for SoVitsSvc {
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
            let new_audio = dsp::resample(input, rate, 16_000);
            let overlap = settings.overlap_ms as usize * 16;
            let search = settings.search_ms as usize * 16;
            let context = settings.context_ms as usize * 16;
            let required = new_audio.len() + overlap + search + context;
            self.history.extend(new_audio);
            if self.history.len() < required {
                let mut padding = vec![0.0; required - self.history.len()];
                padding.append(&mut self.history);
                self.history = padding;
            } else if self.history.len() > required {
                self.history.drain(..self.history.len() - required);
            }
            // The same gate as the RVC engine: the loudest 20 ms decides, and one
            // more block is converted after the last loud one.
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
                return Ok(vec![0.0; input.len()]);
            }
            let audio = self.history.clone();
            let mut voice = self.convert(&audio, settings)?;
            // The last frame reaches past the history; keep the part that lines up.
            let aligned = (audio.len() as u64 * self.info.sample_rate as u64 / 16_000) as usize;
            voice.truncate(aligned);
            let converted = dsp::resample(&voice, self.info.sample_rate, rate);
            let mut out = self.sola.process(
                &converted,
                input.len(),
                settings.overlap_ms as usize * rate as usize / 1000,
                settings.search_ms as usize * rate as usize / 1000,
            )?;
            if closing {
                dsp::fade(&mut out, 1.0, 0.0);
                self.sola.reset();
            }
            Ok(out)
        }

        fn reset(&mut self) {
            self.history.clear();
            self.sola.reset();
            self.geometry = None;
            self.gate_hold = 0;
            self.gate_open = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unvoiced_frames_are_filled_and_flagged() {
        let (f0, uv) = f0_and_uv(&[0.0, 200.0, 0.0, 0.0, 260.0, 0.0], 12.0);
        assert_eq!(uv, vec![0, 1, 0, 0, 1, 0]);
        let expected = [200.0, 200.0, 220.0, 240.0, 260.0, 260.0].map(|v| v * 2.0);
        for (got, want) in f0.iter().zip(expected) {
            assert!((got - want).abs() < 1e-3, "{f0:?}");
        }
        let (silent, flags) = f0_and_uv(&[0.0; 3], 0.0);
        assert_eq!((silent, flags), (vec![0.0; 3], vec![0; 3]));
    }

    #[test]
    fn frames_cover_the_audio_at_the_voice_rate() {
        let info = Info {
            sample_rate: 44_100,
            hop: 512,
            ssl_dim: 768,
            speakers: 1,
            names: vec![],
        };
        // One second at 16 kHz is 44100 samples, 86.1 frames of 512.
        assert_eq!(frame_count(16_000, &info), 87);
        assert_eq!(frame_count(0, &info), 1);
    }
}
