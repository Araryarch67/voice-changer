//! Diffusion-SVC voices (CNChTu/Diffusion-SVC, as the reference's `DiffusionSVC`
//! engine runs them): a combo checkpoint (`.ptc`) holds a "naive" model, which
//! turns ContentVec units, F0 and loudness straight into a mel spectrogram, and a
//! shallow diffusion model, which refines that mel; NSF-HiFiGAN turns the mel into
//! 44.1 kHz audio.
//!
//! Import builds three ONNX graphs from the checkpoint
//! (`crate::diffusion_svc_export`): `naive.onnx`, `condition.onnx` (the diffusion
//! model's conditioning) and `denoiser.onnx` (one WaveNet step). The sampler that calls the denoiser, DPM-Solver++ (multistep,
//! second order, time-uniform steps, as `GaussianDiffusion` uses it), runs here in
//! Rust.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[cfg(feature = "export")]
pub use crate::diffusion_svc_export::{export, Exported};

/// Mel bins of the NSF-HiFiGAN vocoder the models are trained for.
pub const MEL_BINS: usize = 128;
/// The vocoder in the shared models folder.
pub use crate::config::VOCODER_FILE;

/// Predicts the noise in a sample at a model input time: the denoiser.
pub type Denoise<'a> = dyn FnMut(&[f32], f32) -> Result<Vec<f32>> + 'a;

/// What the graphs and the sampler need to know, stored in `denoiser.onnx`'s
/// metadata (`diffusion_svc`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Info {
    pub sample_rate: u32,
    /// Audio samples per mel frame.
    pub block_size: usize,
    /// ContentVec channels the models take.
    pub units: usize,
    pub speakers: usize,
    /// The most diffusion steps the shallow model was trained for, and the
    /// default.
    pub k_step_max: usize,
    pub spec_min: f32,
    pub spec_max: f32,
    /// The diffusion model's noise schedule (1000 steps).
    pub betas: Vec<f32>,
    pub alphas_cumprod: Vec<f32>,
}

impl Info {
    /// The default speed-up, as the reference's slot generator picks it: the
    /// second largest divisor of `k_step_max`.
    pub fn default_speedup(&self) -> usize {
        let n = self.k_step_max.max(1);
        (1..n).rev().find(|d| n.is_multiple_of(*d)).unwrap_or(1)
    }
}

// ------------------------------------------------------------------------------
// Sampling
// ------------------------------------------------------------------------------

/// The discrete-time VP noise schedule of DPM-Solver (`NoiseScheduleVP` with
/// `schedule='discrete'`), on the first `k_step` betas.
#[derive(Debug, Clone)]
pub struct Schedule {
    t_array: Vec<f64>,
    log_alpha: Vec<f64>,
}

/// Piecewise-linear interpolation through `(xp, yp)` (ascending `xp`), extended
/// linearly beyond both ends, as DPM-Solver's `interpolate_fn`.
fn interpolate(x: f64, xp: &[f64], yp: &[f64]) -> f64 {
    let k = xp.len();
    let i = if x <= xp[0] {
        0
    } else if x >= xp[k - 1] {
        k - 2
    } else {
        xp.partition_point(|&v| v < x).saturating_sub(1).min(k - 2)
    };
    yp[i] + (x - xp[i]) * (yp[i + 1] - yp[i]) / (xp[i + 1] - xp[i])
}

impl Schedule {
    pub fn new(betas: &[f32]) -> Self {
        let mut log_alpha = Vec::with_capacity(betas.len());
        let mut sum = 0.0f64;
        for beta in betas {
            sum += 0.5 * (1.0 - *beta as f64).ln();
            log_alpha.push(sum);
        }
        // numerical_clip_alpha: drop the tail whose half log-SNR is below -5.1.
        let below = log_alpha
            .iter()
            .filter(|&&a| a - 0.5 * (1.0 - (2.0 * a).exp()).ln() < -5.1)
            .count();
        log_alpha.truncate(log_alpha.len() - below);
        let n = log_alpha.len();
        let t_array = (1..=n).map(|i| i as f64 / n as f64).collect();
        Self { t_array, log_alpha }
    }

    pub fn total(&self) -> usize {
        self.log_alpha.len()
    }

    fn log_mean(&self, t: f64) -> f64 {
        interpolate(t, &self.t_array, &self.log_alpha)
    }

    fn alpha(&self, t: f64) -> f64 {
        self.log_mean(t).exp()
    }

    fn sigma(&self, t: f64) -> f64 {
        (1.0 - (2.0 * self.log_mean(t)).exp()).sqrt()
    }

    fn lambda(&self, t: f64) -> f64 {
        let log_mean = self.log_mean(t);
        log_mean - 0.5 * (1.0 - (2.0 * log_mean).exp()).ln()
    }
}

/// DPM-Solver++ (multistep, order 2, time-uniform steps, lower order at the end
/// for fewer than 10 steps) from `x` at t = 1 down to t = 1/N. `noise` predicts
/// the noise for `x` at a model input time.
pub fn dpm_solver(
    schedule: &Schedule,
    mut x: Vec<f32>,
    steps: usize,
    noise: &mut Denoise,
) -> Result<Vec<f32>> {
    ensure!(steps >= 2, "DPM-Solver needs at least two steps");
    let n = schedule.total() as f64;
    let (t_start, t_end) = (1.0f64, 1.0 / n);
    // torch.linspace in float32.
    let times: Vec<f64> = (0..=steps)
        .map(|i| (t_start + (t_end - t_start) * i as f64 / steps as f64) as f32 as f64)
        .collect();
    let mut data_prediction = |x: &[f32], t: f64| -> Result<Vec<f32>> {
        let predicted = noise(x, ((t - 1.0 / n) * n) as f32)?;
        ensure!(predicted.len() == x.len(), "The denoiser changed the shape");
        let (alpha, sigma) = (schedule.alpha(t) as f32, schedule.sigma(t) as f32);
        Ok(x.iter()
            .zip(&predicted)
            .map(|(x, e)| (x - sigma * e) / alpha)
            .collect())
    };
    let first_update = |x: &[f32], model: &[f32], s: f64, t: f64| -> Vec<f32> {
        let h = schedule.lambda(t) - schedule.lambda(s);
        let ratio = (schedule.sigma(t) / schedule.sigma(s)) as f32;
        let coefficient = (schedule.alpha(t) * (-h).exp_m1()) as f32;
        x.iter()
            .zip(model)
            .map(|(x, m)| ratio * x - coefficient * m)
            .collect()
    };
    let second_update = |x: &[f32], previous: (&[f32], f64), last: (&[f32], f64), t: f64| {
        let (lambda_1, lambda_0, lambda_t) = (
            schedule.lambda(previous.1),
            schedule.lambda(last.1),
            schedule.lambda(t),
        );
        let (h_0, h) = (lambda_0 - lambda_1, lambda_t - lambda_0);
        let r0 = h_0 / h;
        let ratio = (schedule.sigma(t) / schedule.sigma(last.1)) as f32;
        let phi = (schedule.alpha(t) * (-h).exp_m1()) as f32;
        let d = (1.0 / r0) as f32;
        x.iter()
            .zip(last.0.iter().zip(previous.0))
            .map(|(x, (m0, m1))| ratio * x - phi * m0 - 0.5 * phi * d * (m0 - m1))
            .collect::<Vec<f32>>()
    };
    let mut previous_model = data_prediction(&x, times[0])?;
    let mut previous_time = times[0];
    x = first_update(&x, &previous_model, times[0], times[1]);
    let mut last_model = data_prediction(&x, times[1])?;
    let mut last_time = times[1];
    for (step, &t) in times.iter().enumerate().skip(2) {
        let order = if steps < 10 {
            2.min(steps + 1 - step)
        } else {
            2
        };
        x = if order == 1 {
            first_update(&x, &last_model, last_time, t)
        } else {
            second_update(
                &x,
                (&previous_model, previous_time),
                (&last_model, last_time),
                t,
            )
        };
        previous_model = std::mem::take(&mut last_model);
        previous_time = last_time;
        last_time = t;
        if step < steps {
            last_model = data_prediction(&x, t)?;
        }
    }
    Ok(x)
}

/// Gaussian noise from a small deterministic generator (SplitMix64 and
/// Box-Muller); `seed` 0 gives zeros, for comparisons with a noiseless
/// reference.
pub struct Noise {
    state: u64,
    spare: Option<f32>,
}

impl Noise {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn sample(&mut self) -> f32 {
        if self.state == 0 && self.spare.is_none() {
            return 0.0;
        }
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let uniform = |n: &mut Self| ((n.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let (u1, u2) = (uniform(self), uniform(self));
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        self.spare = Some((radius * angle.sin()) as f32);
        (radius * angle.cos()) as f32
    }

    pub fn fill(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.sample()).collect()
    }
}

/// Shallow diffusion from the naive model's mel (`[T,128]`, frame-major): noise
/// it to step `k_step`, then denoise with `speedup` (DPM-Solver++ with `k_step /
/// speedup` steps, or every step when `speedup` is 1). Returns `[T,128]`.
pub fn shallow_diffusion(
    info: &Info,
    naive_mel: &[f32],
    k_step: usize,
    speedup: usize,
    rng: &mut Noise,
    denoise: &mut Denoise,
) -> Result<Vec<f32>> {
    ensure!(
        (1..=info.k_step_max).contains(&k_step),
        "Diffusion steps must be 1 to {}",
        info.k_step_max
    );
    let frames = naive_mel.len() / MEL_BINS;
    let range = info.spec_max - info.spec_min;
    // [T,128] -> normalized [128,T] (the denoiser's [1,1,128,T]).
    let mut x0 = vec![0.0; naive_mel.len()];
    for f in 0..frames {
        for m in 0..MEL_BINS {
            x0[m * frames + f] = (naive_mel[f * MEL_BINS + m] - info.spec_min) / range * 2.0 - 1.0;
        }
    }
    let t = k_step - 1;
    let (a, b) = (
        (info.alphas_cumprod[t] as f64).sqrt() as f32,
        (1.0 - info.alphas_cumprod[t] as f64).sqrt() as f32,
    );
    let noise = rng.fill(x0.len());
    let mut x: Vec<f32> = x0.iter().zip(&noise).map(|(x, e)| a * x + b * e).collect();
    if speedup > 1 {
        let schedule = Schedule::new(&info.betas[..k_step]);
        let steps = k_step / speedup;
        x = dpm_solver(&schedule, x, steps, denoise)?;
    } else {
        x = ddpm(info, x, k_step, rng, denoise)?;
    }
    let mut mel = vec![0.0; x.len()];
    for f in 0..frames {
        for m in 0..MEL_BINS {
            mel[f * MEL_BINS + m] = (x[m * frames + f] + 1.0) / 2.0 * range + info.spec_min;
        }
    }
    Ok(mel)
}

/// The full ancestral sampler (`p_sample` for every step), for a speed-up of 1.
fn ddpm(
    info: &Info,
    mut x: Vec<f32>,
    k_step: usize,
    rng: &mut Noise,
    denoise: &mut Denoise,
) -> Result<Vec<f32>> {
    for t in (0..k_step).rev() {
        let predicted = denoise(&x, t as f32)?;
        let alpha_bar = info.alphas_cumprod[t] as f64;
        let alpha_bar_prev = if t == 0 {
            1.0
        } else {
            info.alphas_cumprod[t - 1] as f64
        };
        let beta = info.betas[t] as f64;
        let recip = (1.0 / alpha_bar).sqrt();
        let recip_m1 = (1.0 / alpha_bar - 1.0).sqrt();
        let coef1 = beta * alpha_bar_prev.sqrt() / (1.0 - alpha_bar);
        let coef2 = (1.0 - alpha_bar_prev) * (1.0 - beta).sqrt() / (1.0 - alpha_bar);
        let variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar);
        let deviation = if t == 0 {
            0.0
        } else {
            (0.5 * variance.max(1e-20).ln()).exp()
        };
        x = x
            .iter()
            .zip(&predicted)
            .map(|(x, e)| {
                let x0 = (recip * *x as f64 - recip_m1 * *e as f64).clamp(-1.0, 1.0);
                let mean = coef1 * x0 + coef2 * *x as f64;
                (mean + deviation * rng.sample() as f64) as f32
            })
            .collect();
    }
    Ok(x)
}

// ------------------------------------------------------------------------------
// Audio preparation, as the reference's Diffusion-SVC pipeline does it
// ------------------------------------------------------------------------------

/// Frames of `samples` 16 kHz samples at the model's hop (`block_size` at
/// `sample_rate`): `len // hop + 1`.
pub fn frame_count(samples: usize, info: &Info) -> usize {
    let hop = info.block_size as f64 * 16_000.0 / info.sample_rate as f64;
    (samples as f64 / hop).floor() as usize + 1
}

/// RMS loudness per frame, as `VolumeExtractor.extract_t`: windows of the hop
/// rounded down, padded to `frames` with the last value.
pub fn volume(audio: &[f32], frames: usize, info: &Info) -> Vec<f32> {
    let hop = (info.block_size * 16_000 / info.sample_rate as usize).max(1);
    let mut volume: Vec<f32> = audio
        .chunks_exact(hop)
        .map(|w| (w.iter().map(|v| v * v).sum::<f32>() / hop as f32).sqrt())
        .take(frames)
        .collect();
    let last = volume.last().copied().unwrap_or(0.0);
    volume.resize(frames, last);
    volume
}

/// The sample mask of `get_mask_from_volume_t`: frames above -60 dB, widened by
/// four frames each side, then linearly upsampled to `block_size` samples a
/// frame.
pub fn mask(volume: &[f32], block_size: usize) -> Vec<f32> {
    mask_below(volume, block_size, -60.0)
}

/// [`mask`] with its threshold in dB (DDSP-SVC's is -45).
pub fn mask_below(volume: &[f32], block_size: usize, db: f32) -> Vec<f32> {
    let threshold = 10f32.powf(db / 20.0);
    let on: Vec<f32> = volume
        .iter()
        .map(|v| if *v > threshold { 1.0 } else { 0.0 })
        .collect();
    let n = on.len();
    let at = |i: isize| on[i.clamp(0, n as isize - 1) as usize];
    let widened: Vec<f32> = (0..n as isize)
        .map(|i| (i - 4..=i + 4).map(at).fold(0.0, f32::max))
        .collect();
    let mut samples = Vec::with_capacity(n * block_size);
    for i in 0..n {
        let (from, to) = (widened[i], widened[(i + 1).min(n - 1)]);
        for j in 0..block_size {
            samples.push(from + (to - from) * j as f32 / block_size as f32);
        }
    }
    samples
}

/// F0 ready for the models: unvoiced frames filled by linear interpolation
/// between voiced ones (held at the ends), at least 50 Hz, then shifted by
/// `semitones`.
pub fn fill_f0(detected: &[f32], semitones: f32) -> Vec<f32> {
    let voiced: Vec<usize> = (0..detected.len())
        .filter(|&i| detected[i].is_finite() && detected[i] > 0.0)
        .collect();
    let factor = 2f32.powf(semitones / 12.0);
    (0..detected.len())
        .map(|i| {
            let hz = if voiced.is_empty() {
                0.0
            } else {
                let next = voiced.partition_point(|&v| v < i);
                if next < voiced.len() && voiced[next] == i {
                    detected[i]
                } else if next == 0 {
                    detected[voiced[0]]
                } else if next == voiced.len() {
                    detected[voiced[next - 1]]
                } else {
                    let (a, b) = (voiced[next - 1], voiced[next]);
                    let t = (i - a) as f32 / (b - a) as f32;
                    detected[a] + (detected[b] - detected[a]) * t
                }
            };
            hz.max(50.0) * factor
        })
        .collect()
}

/// `[source_frames, channels]` features stretched to `frames` by nearest
/// neighbour, as `F.interpolate(mode="nearest")`.
pub fn stretch(features: &[f32], channels: usize, frames: usize) -> Vec<f32> {
    let source = features.len() / channels;
    let mut out = Vec::with_capacity(frames * channels);
    for i in 0..frames {
        let s = (i * source / frames).min(source - 1);
        out.extend_from_slice(&features[s * channels..(s + 1) * channels]);
    }
    out
}

/// The diffusion depth (k_step) and speed-up for the settings' depth and sampler
/// steps (0 for the model's defaults: its maximum depth and its default
/// speed-up). DPM-Solver needs at least two steps, so a speed-up never leaves
/// fewer.
pub fn steps_for(info: &Info, depth: u32, sampler_steps: u32) -> (usize, usize) {
    let k_step = match depth as usize {
        0 => info.k_step_max,
        k => k.min(info.k_step_max),
    };
    let speedup = match sampler_steps as usize {
        0 => info.default_speedup(),
        steps => (k_step / steps.max(1)).max(1),
    };
    let speedup = if speedup > 1 && k_step / speedup < 2 {
        (k_step / 2).max(1)
    } else {
        speedup
    };
    (k_step, speedup)
}

#[cfg(any(feature = "onnx", feature = "onnx-static"))]
pub use engine::DiffusionSvc;

#[cfg(any(feature = "onnx", feature = "onnx-static"))]
mod engine {
    use super::*;
    use crate::{
        config::{Device, ModelManifest, Settings},
        dsp::{self, Sola},
        engine::Converter,
        rvc::{embed, open_session, Detectors},
    };
    use anyhow::Context;
    use ort::{session::Session, value::Tensor};
    use std::path::Path;

    /// A Diffusion-SVC voice, converting block by block like the RVC engine: the
    /// same history, noise gate and SOLA crossfade.
    pub struct DiffusionSvc {
        manifest: ModelManifest,
        info: Info,
        naive: Session,
        condition: Session,
        denoiser: Session,
        vocoder: Session,
        embedder: Session,
        detectors: Detectors,
        history: Vec<f32>,
        sola: Sola,
        geometry: Option<(u32, u32, u32, u32, u32)>,
        gate_hold: u32,
        gate_open: bool,
        noise: Noise,
    }

    impl DiffusionSvc {
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
            let file =
                |f: &Option<std::path::PathBuf>| f.clone().expect("checked by ModelManifest::load");
            let naive = open(&manifest.model, "the Diffusion-SVC naive model")?;
            let condition = open(&file(&manifest.condition), "the Diffusion-SVC conditioning")?;
            let denoiser = open(&file(&manifest.denoiser), "the Diffusion-SVC denoiser")?;
            let vocoder = open(&file(&manifest.vocoder), "NSF-HiFiGAN")?;
            let embedder = open(&manifest.embedder, "ContentVec")?;
            let rmvpe = open(&manifest.pitch, "RMVPE")?;
            let info: Info = {
                let metadata = denoiser.metadata()?;
                let text = metadata
                    .custom("diffusion_svc")?
                    .context("The denoiser has no diffusion_svc metadata; import the .ptc again")?;
                serde_json::from_str(&text).context("Read the denoiser's diffusion_svc metadata")?
            };
            for (session, input) in [(&vocoder, "mel"), (&vocoder, "f0"), (&naive, "units")] {
                anyhow::ensure!(
                    session.inputs.iter().any(|i| i.name == input),
                    "A Diffusion-SVC model has no `{input}` input"
                );
            }
            log::info!(
                "Diffusion-SVC voice \"{}\" on {} ({} speakers, up to {} diffusion steps)",
                manifest.name,
                device.label(),
                info.speakers,
                info.k_step_max
            );
            let detectors = Detectors::new(rmvpe, &manifest.pitch, threads, device);
            Ok(Self {
                manifest,
                info,
                naive,
                condition,
                denoiser,
                vocoder,
                embedder,
                detectors,
                history: Vec::new(),
                sola: Sola::default(),
                geometry: None,
                gate_hold: 0,
                gate_open: false,
                noise: Noise::new(0x5eed_d1ff),
            })
        }

        /// The k_step and speed-up to use for `settings`.
        fn steps(&self, settings: &Settings) -> (usize, usize) {
            steps_for(
                &self.info,
                settings.diffusion_steps,
                settings.diffusion_sampler_steps,
            )
        }

        /// The voice's audio (44.1 kHz) for 16 kHz `audio`.
        fn convert(&mut self, audio: &[f32], settings: &Settings) -> Result<Vec<f32>> {
            let info = self.info.clone();
            let frames = frame_count(audio.len(), &info);
            let volume = volume(audio, frames, &info);
            let features = embed(
                &mut self.embedder,
                &self.manifest.embedder_input,
                &self.manifest.embedder_output,
                audio,
                info.units,
            )?;
            let units = stretch(&features, info.units, frames);
            let detected = self.detectors.detect(audio, frames, settings.f0_detector)?;
            let f0 = fill_f0(&detected, settings.semitones);
            let speaker = settings.speaker_id.min(info.speakers as i64 - 1);
            let inputs = |session: &Session| -> Result<
                Vec<(
                    std::borrow::Cow<'static, str>,
                    ort::session::SessionInputValue<'static>,
                )>,
            > {
                let mut feeds = ort::inputs! {
                    "units" => Tensor::from_array(([1, frames, info.units], units.clone().into_boxed_slice()))?,
                    "f0" => Tensor::from_array(([1, frames, 1], f0.clone().into_boxed_slice()))?,
                    "volume" => Tensor::from_array(([1, frames, 1], volume.clone().into_boxed_slice()))?,
                };
                if session.inputs.iter().any(|i| i.name == "speaker") {
                    feeds.push((
                        "speaker".into(),
                        Tensor::from_array(([1], vec![speaker].into_boxed_slice()))?.into(),
                    ));
                }
                Ok(feeds)
            };
            let feeds = inputs(&self.naive)?;
            let mut mel = self.naive.run(feeds)?[0]
                .try_extract_tensor::<f32>()?
                .1
                .to_vec();
            if settings.diffusion {
                let feeds = inputs(&self.condition)?;
                let condition = self.condition.run(feeds)?[0]
                    .try_extract_tensor::<f32>()?
                    .1
                    .to_vec();
                let hidden = condition.len() / frames;
                let (k_step, speedup) = self.steps(settings);
                let denoiser = &mut self.denoiser;
                let mut denoise = |x: &[f32], step: f32| -> Result<Vec<f32>> {
                    let outputs = denoiser.run(ort::inputs! {
                        "x" => Tensor::from_array(([1, 1, MEL_BINS, frames], x.to_vec().into_boxed_slice()))?,
                        "step" => Tensor::from_array(([1], vec![step].into_boxed_slice()))?,
                        "condition" => Tensor::from_array(([1, hidden, frames], condition.clone().into_boxed_slice()))?,
                    })?;
                    Ok(outputs[0].try_extract_tensor::<f32>()?.1.to_vec())
                };
                mel =
                    shallow_diffusion(&info, &mel, k_step, speedup, &mut self.noise, &mut denoise)?;
            }
            let outputs = self.vocoder.run(ort::inputs! {
                "mel" => Tensor::from_array(([1, frames, MEL_BINS], mel.into_boxed_slice()))?,
                "f0" => Tensor::from_array(([1, frames], f0.into_boxed_slice()))?,
            })?;
            let waveform = outputs[0].try_extract_tensor::<f32>()?.1;
            let gate = mask(&volume, info.block_size);
            anyhow::ensure!(
                waveform.iter().all(|v| v.is_finite()),
                "Non-finite Diffusion-SVC output"
            );
            Ok(waveform
                .iter()
                .zip(gate.iter().chain(std::iter::repeat(&1.0)))
                .map(|(v, m)| (v * m * self.manifest.output_scale).clamp(-1.0, 1.0))
                .collect())
        }
    }

    impl Converter for DiffusionSvc {
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

    fn linear_betas(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (1e-4 + (0.02 - 1e-4) * i as f64 / (n - 1) as f64) as f32)
            .collect()
    }

    #[test]
    fn interpolation_extends_linearly_beyond_the_keypoints() {
        let (xp, yp) = ([1.0, 2.0, 4.0], [10.0, 20.0, 0.0]);
        assert_eq!(interpolate(1.5, &xp, &yp), 15.0);
        assert_eq!(interpolate(3.0, &xp, &yp), 10.0);
        assert_eq!(interpolate(0.0, &xp, &yp), 0.0);
        assert_eq!(interpolate(5.0, &xp, &yp), -10.0);
        assert_eq!(interpolate(2.0, &xp, &yp), 20.0);
    }

    #[test]
    fn the_schedule_matches_its_definition() {
        let betas = linear_betas(1000);
        let schedule = Schedule::new(&betas[..100]);
        assert_eq!(schedule.total(), 100);
        // At t = k / N, log alpha is half the log of the cumulative product.
        let product: f64 = betas[..50].iter().map(|b| 1.0 - *b as f64).product();
        assert!((schedule.log_mean(0.5) - 0.5 * product.ln()).abs() < 1e-9);
        let t = 0.37;
        let (alpha, sigma) = (schedule.alpha(t), schedule.sigma(t));
        assert!((alpha * alpha + sigma * sigma - 1.0).abs() < 1e-12);
        assert!((schedule.lambda(t) - (alpha / sigma).ln()).abs() < 1e-9);
    }

    #[test]
    fn dpm_solver_recovers_data_from_an_exact_noise_predictor() {
        // With x_t = alpha x0 + sigma e and a predictor that knows x0, the solver
        // must land on x0 whatever the number of steps.
        let schedule = Schedule::new(&linear_betas(1000)[..100]);
        let x0 = [0.3_f32, -0.7, 0.1, 0.9];
        let e = [1.2_f32, -0.4, 0.8, -1.5];
        let (alpha, sigma) = (schedule.alpha(1.0) as f32, schedule.sigma(1.0) as f32);
        let start: Vec<f32> = x0
            .iter()
            .zip(&e)
            .map(|(x, e)| alpha * x + sigma * e)
            .collect();
        let n = schedule.total() as f64;
        for steps in [2, 5, 10, 20] {
            let mut predictor = |x: &[f32], t_input: f32| -> Result<Vec<f32>> {
                let t = t_input as f64 / n + 1.0 / n;
                let (a, s) = (schedule.alpha(t) as f32, schedule.sigma(t) as f32);
                Ok(x.iter().zip(&x0).map(|(x, x0)| (x - a * x0) / s).collect())
            };
            let out = dpm_solver(&schedule, start.clone(), steps, &mut predictor).unwrap();
            for (got, want) in out.iter().zip(&x0) {
                // At t = 1/N the sample still carries sigma(1/N) of noise.
                assert!((got - want).abs() < 0.02, "{steps} steps: {out:?}");
            }
        }
    }

    #[test]
    fn noise_is_standard_normal_and_seed_zero_is_silent() {
        let samples = Noise::new(7).fill(20_000);
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let var = samples.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / samples.len() as f32;
        assert!(
            mean.abs() < 0.03 && (var - 1.0).abs() < 0.05,
            "{mean} {var}"
        );
        assert!(Noise::new(0).fill(10).iter().all(|v| *v == 0.0));
    }

    #[test]
    fn steps_follow_the_settings_and_keep_two_sampler_steps() {
        let info = Info {
            sample_rate: 44_100,
            block_size: 512,
            units: 768,
            speakers: 1,
            k_step_max: 100,
            spec_min: -12.0,
            spec_max: 2.0,
            betas: vec![],
            alphas_cumprod: vec![],
        };
        assert_eq!(steps_for(&info, 0, 0), (100, 50));
        assert_eq!(steps_for(&info, 0, 10), (100, 10));
        assert_eq!(steps_for(&info, 40, 4), (40, 10));
        assert_eq!(steps_for(&info, 500, 0), (100, 50));
        assert_eq!(steps_for(&info, 3, 0), (3, 1));
        assert_eq!(steps_for(&info, 100, 100), (100, 1));
    }

    #[test]
    fn default_speedup_is_the_second_largest_divisor() {
        let info = |k| Info {
            sample_rate: 44_100,
            block_size: 512,
            units: 768,
            speakers: 1,
            k_step_max: k,
            spec_min: -12.0,
            spec_max: 2.0,
            betas: vec![],
            alphas_cumprod: vec![],
        };
        assert_eq!(info(100).default_speedup(), 50);
        assert_eq!(info(20).default_speedup(), 10);
        assert_eq!(info(7).default_speedup(), 1);
    }
}
