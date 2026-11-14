//! DDSP-SVC voices (yxlllc/DDSP-SVC 3.0, as the reference's `DDSP_SVC` engine runs
//! them). A DDSP model turns ContentVec units, F0, phase and loudness into the
//! controls of a differentiable synthesizer (`Unit2Control`, a PCmer); the
//! synthesizer, written here in Rust, turns them into 44.1 kHz audio:
//!
//! - `Sins`: additive sinusoids with a group-delay filter, plus filtered noise;
//! - `CombSub`: a combtooth excitation through a group-delay filter and a
//!   pitch-adaptive harmonic filter, plus filtered noise;
//! - `CombSubFast`: the same excitation and noise filtered frame by frame in the
//!   frequency domain.
//!
//! The result can then be refined, as the reference does, either by a separately
//! trained shallow-diffusion model (the DDSP audio's mel is noised and denoised,
//! then NSF-HiFiGAN renders it) or by the NSF-HiFiGAN "enhancer" alone.
//!
//! Import builds `ddsp.onnx` (Unit2Control) and, with a diffusion model, the
//! same `condition.onnx` and `denoiser.onnx` Diffusion-SVC voices use
//! (`crate::ddsp_svc_export`).
use crate::diffusion_svc::{self, Info, MEL_BINS};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[cfg(feature = "export")]
pub use crate::ddsp_svc_export::{export, Exported};

/// The reference's `threshold` setting (dB): frames quieter than this, and four
/// frames around them, are muted.
pub const MASK_DB: f32 = -45.0;
/// The reference's default diffusion depth (`kStep`) and speed-up (`diffAcc`).
pub const DEFAULT_K_STEP: usize = 120;
pub const DEFAULT_SPEEDUP: usize = 20;
/// ContentVec's hop at 16 kHz.
const ENCODER_HOP: f32 = 320.0 / 16_000.0;
/// NSF-HiFiGAN's analysis (its `config.json`): 44.1 kHz, 2048-point STFT, hop
/// 512, 128 mels from 40 Hz to 16 kHz.
pub const VOCODER_RATE: u32 = 44_100;
pub const VOCODER_HOP: usize = 512;
const N_FFT: usize = 2048;
const MEL_FMIN: f64 = 40.0;
const MEL_FMAX: f64 = 16_000.0;

/// The synthesizer a DDSP model drives (`model.type` in its config).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Synth {
    Sins,
    CombSub,
    CombSubFast,
}

/// What the synthesizer needs to know, stored in `ddsp.onnx`'s metadata
/// (`ddsp_svc`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DdspInfo {
    pub sample_rate: u32,
    /// Audio samples per frame.
    pub block_size: usize,
    /// ContentVec channels the model takes.
    pub units: usize,
    pub speakers: usize,
    pub synth: Synth,
    /// The sizes of the model's three controls, in its order: `Sins` amplitudes,
    /// group delay and noise; `CombSub` group delay, harmonic magnitude and
    /// noise; `CombSubFast` harmonic magnitude, harmonic phase and noise.
    pub controls: [usize; 3],
    /// The diffusion model's vocoder is `nsf-hifigan-log10`: its mel is scaled
    /// to log10 before NSF-HiFiGAN.
    #[serde(default)]
    pub log10_mel: bool,
}

impl DdspInfo {
    /// Values a frame of the model's output.
    pub fn control_size(&self) -> usize {
        self.controls.iter().sum()
    }
}

// ------------------------------------------------------------------------------
// Feature preparation, as the reference's DDSP-SVC pipeline does it
// ------------------------------------------------------------------------------

/// Frames of `samples` model-rate samples: `len // block_size + 1`.
pub fn frame_count(samples: usize, block_size: usize) -> usize {
    samples / block_size + 1
}

/// RMS loudness per frame, as `Volume_Extractor`: the squared audio reflected
/// by half a hop at both ends, averaged over hop-long windows.
pub fn volume(audio: &[f32], block_size: usize) -> Vec<f32> {
    let frames = frame_count(audio.len(), block_size);
    let n = audio.len() as isize;
    if n < 2 {
        return vec![0.0; frames];
    }
    let left = (block_size / 2) as isize;
    let at = |i: isize| {
        let mut j = i - left;
        // numpy's "reflect": the edge sample is not repeated.
        while j < 0 || j >= n {
            j = if j < 0 { -j } else { 2 * (n - 1) - j };
        }
        let v = audio[j as usize];
        v * v
    };
    (0..frames)
        .map(|f| {
            let start = (f * block_size) as isize;
            let sum: f64 = (start..start + block_size as isize)
                .map(|i| at(i) as f64)
                .sum();
            (sum / block_size as f64).sqrt() as f32
        })
        .collect()
}

/// ContentVec's 20 ms units (`[U, channels]`) picked for each model frame, as
/// `Units_Encoder.encode` aligns them: unit `round(i * ratio)`.
pub fn align_units(
    features: &[f32],
    channels: usize,
    frames: usize,
    sample_rate: u32,
    block_size: usize,
) -> Vec<f32> {
    let source = (features.len() / channels).max(1);
    let ratio = (block_size as f32 / sample_rate as f32) / ENCODER_HOP;
    let mut out = Vec::with_capacity(frames * channels);
    for i in 0..frames {
        let s = ((ratio * i as f32).round_ties_even() as usize).min(source - 1);
        match features.get(s * channels..(s + 1) * channels) {
            Some(row) => out.extend_from_slice(row),
            None => out.extend(std::iter::repeat_n(0.0, channels)),
        }
    }
    out
}

/// `[frames, channels]` values linearly interpolated to `factor` samples a frame,
/// as the reference's `upsample` (the last frame holds its value).
pub fn upsample(frames: &[f32], channels: usize, factor: usize) -> Vec<f32> {
    let n = frames.len() / channels;
    let mut out = vec![0.0; n * factor * channels];
    for i in 0..n {
        let next = (i + 1).min(n - 1);
        for j in 0..factor {
            let t = j as f32 / factor as f32;
            let row = (i * factor + j) * channels;
            for c in 0..channels {
                let (a, b) = (frames[i * channels + c], frames[next * channels + c]);
                out[row + c] = a + (b - a) * t;
            }
        }
    }
    out
}

/// The excitation's position in its cycle at every sample (`x`, in [-0.5, 0.5])
/// and the upsampled F0: the running sum of `f0 / sample_rate`, wrapped.
pub struct Excitation {
    pub cycle: Vec<f32>,
    pub f0: Vec<f32>,
}

impl Excitation {
    pub fn new(f0_frames: &[f32], info: &DdspInfo) -> Self {
        let f0 = upsample(f0_frames, 1, info.block_size);
        let rate = info.sample_rate as f64;
        let mut sum = 0.0_f64;
        let cycle = f0
            .iter()
            .map(|f| {
                sum += *f as f64 / rate;
                (sum - sum.round_ties_even()) as f32
            })
            .collect();
        Self { cycle, f0 }
    }

    /// The phase (radians) at the first sample of each frame: the model's
    /// `phase` input.
    pub fn phase_frames(&self, block_size: usize) -> Vec<f32> {
        self.cycle
            .iter()
            .step_by(block_size)
            .map(|x| 2.0 * std::f32::consts::PI * x)
            .collect()
    }
}

/// F0 for the enhancer's "auto" key: the semitones above 760 Hz the highest F0
/// reaches, rounded up (0 when it stays below).
pub fn adaptive_key(f0: &[f32]) -> u32 {
    let highest = f0.iter().copied().fold(0.0_f32, f32::max) as f64;
    if highest <= 0.0 {
        return 0;
    }
    (12.0 * (highest / 760.0).log2()).ceil().max(0.0) as u32
}

/// The rate the enhancer treats the audio as having for `key`: NSF-HiFiGAN's
/// rate raised by `key` semitones, to a multiple of 100 Hz.
pub fn adaptive_rate(key: u32) -> u32 {
    let factor = 2f64.powf(-(key as f64) / 12.0);
    100 * (VOCODER_RATE as f64 / factor / 100.0).round_ties_even() as u32
}

/// F0 for audio at `adaptive` Hz, as the enhancer resamples it: scaled and
/// interpolated onto the vocoder's frames (held at the ends).
pub fn enhancer_f0(
    f0: &[f32],
    block_size: usize,
    sample_rate: u32,
    adaptive: u32,
    frames: usize,
) -> Vec<f32> {
    let real_factor = VOCODER_RATE as f64 / adaptive as f64;
    let scaled: Vec<f64> = f0.iter().map(|v| *v as f64 * real_factor).collect();
    let step = block_size as f64 / sample_rate as f64 / real_factor;
    let frame = VOCODER_HOP as f64 / VOCODER_RATE as f64;
    let (first, last) = (scaled[0], scaled[scaled.len() - 1]);
    (0..frames)
        .map(|j| {
            let t = frame * j as f64;
            let position = t / step;
            if position <= 0.0 {
                first
            } else if position >= (scaled.len() - 1) as f64 {
                last
            } else {
                let i = position.floor() as usize;
                let (t0, t1) = (step * i as f64, step * (i + 1) as f64);
                scaled[i] + (scaled[i + 1] - scaled[i]) * (t - t0) / (t1 - t0)
            }
        })
        .map(|v| v as f32)
        .collect()
}

/// The diffusion depth (k_step) and speed-up for the settings' depth and sampler
/// steps; 0 takes the reference's defaults (depth 120, speed-up 20: 6 steps).
pub fn steps_for(info: &Info, depth: u32, sampler_steps: u32) -> (usize, usize) {
    let k_step = match depth as usize {
        0 => DEFAULT_K_STEP.min(info.k_step_max),
        k => k.min(info.k_step_max),
    };
    let steps = match sampler_steps as usize {
        0 => (k_step / DEFAULT_SPEEDUP).max(1),
        steps => steps,
    };
    diffusion_svc::steps_for(info, k_step as u32, steps as u32)
}

// ------------------------------------------------------------------------------
// Synthesis
// ------------------------------------------------------------------------------

/// How an impulse response is windowed, as `frequency_impulse_response`.
#[derive(Clone, Copy)]
enum Window {
    /// None: only put in causal form.
    Rect,
    /// A Hann window over the whole response.
    Hann,
    /// The pitch-adaptive raised cosine of `apply_dynamic_window_to_impulse_response`
    /// with this half width (samples). As in the reference, taps beyond the half
    /// width on the right get weight 1.
    Dynamic(f32),
}

/// The DDSP synthesizers, with cached FFT plans.
pub struct Synthesizer {
    planner: FftPlanner<f64>,
}

impl Default for Synthesizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Synthesizer {
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
        }
    }

    fn fft(&mut self, data: &mut [Complex<f64>]) {
        self.planner.plan_fft_forward(data.len()).process(data);
    }

    /// The real signal of length `2 * (bins - 1)` with the half spectrum `bins`
    /// (the imaginary parts of the first and last bins are ignored), as
    /// `torch.fft.irfft`.
    fn irfft(&mut self, bins: &[Complex<f64>]) -> Vec<f64> {
        let n = 2 * (bins.len() - 1);
        let mut full = vec![Complex::new(0.0, 0.0); n];
        for (k, bin) in bins.iter().enumerate() {
            let bin = if k == 0 || k == n / 2 {
                Complex::new(bin.re, 0.0)
            } else {
                *bin
            };
            full[k] = bin;
            if k != 0 && k != n / 2 {
                full[n - k] = bin.conj();
            }
        }
        self.planner.plan_fft_inverse(n).process(&mut full);
        full.iter().map(|v| v.re / n as f64).collect()
    }

    /// One frame's impulse response from its frequency response, windowed and in
    /// causal form (peak in the middle).
    fn impulse(&mut self, response: &[Complex<f64>], window: Window) -> Vec<f64> {
        let ir = self.irfft(response);
        let n = ir.len();
        let half = n / 2;
        (0..n)
            .map(|i| {
                let v = ir[(i + n - half) % n];
                match window {
                    Window::Rect => v,
                    Window::Hann => v * (0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()),
                    Window::Dynamic(half_width) => {
                        let mut w = (i as f32 - half as f32) / half_width;
                        if w > 1.0 {
                            w = 0.0;
                        }
                        v * ((1.0 + (std::f32::consts::PI * w).cos()) / 2.0) as f64
                    }
                }
            })
            .collect()
    }

    /// Filters `audio` with one impulse response per frame, as `fft_convolve`:
    /// 50% overlapped Bartlett-windowed frames, each convolved with its frame's
    /// response, overlap-added and shifted by half the response.
    fn convolve(&mut self, audio: &[f64], responses: &[Vec<f64>]) -> Vec<f64> {
        let frames = responses.len();
        let hop = audio.len() / frames;
        let ir_size = responses[0].len();
        let frame = 2 * hop;
        let size = frame + ir_size - 1;
        let bartlett: Vec<f64> = (0..frame)
            .map(|i| 1.0 - (2.0 * i as f64 / frame as f64 - 1.0).abs())
            .collect();
        let mut out = vec![0.0; frames * hop + size];
        let mut a = vec![Complex::new(0.0, 0.0); size];
        let mut b = vec![Complex::new(0.0, 0.0); size];
        for k in 0..=frames {
            a.iter_mut().for_each(|v| *v = Complex::new(0.0, 0.0));
            b.iter_mut().for_each(|v| *v = Complex::new(0.0, 0.0));
            for (i, w) in bartlett.iter().enumerate() {
                let p = k * hop + i;
                if p >= hop && p - hop < audio.len() {
                    a[i].re = audio[p - hop] * w;
                }
            }
            for (slot, v) in b.iter_mut().zip(&responses[k.min(frames - 1)]) {
                slot.re = *v;
            }
            self.fft(&mut a);
            self.fft(&mut b);
            for (x, y) in a.iter_mut().zip(&b) {
                *x *= y;
            }
            self.planner.plan_fft_inverse(size).process(&mut a);
            for (i, v) in a.iter().enumerate() {
                out[k * hop + i] += v.re / size as f64;
            }
        }
        let start = hop + ir_size / 2;
        out[start..start + audio.len()].to_vec()
    }

    /// Per-frame responses from `[frames, bins]` values.
    fn responses(
        &mut self,
        frames: usize,
        bins: usize,
        mut value: impl FnMut(usize, usize) -> Complex<f64>,
        window: impl Fn(usize) -> Window,
    ) -> Vec<Vec<f64>> {
        (0..frames)
            .map(|t| {
                let response: Vec<Complex<f64>> = (0..bins).map(|k| value(t, k)).collect();
                self.impulse(&response, window(t))
            })
            .collect()
    }

    /// The group-delay all-pass filter: response `exp(i * cumsum(pi * tanh(c)))`.
    fn group_delay(&mut self, audio: &[f64], controls: &[f32], frames: usize) -> Vec<f64> {
        let bins = controls.len() / frames;
        let responses = self.responses(
            frames,
            bins,
            |t, k| {
                let row = &controls[t * bins..t * bins + k + 1];
                let sum: f64 = row.iter().map(|c| PI * (*c as f64).tanh()).sum();
                Complex::from_polar(1.0, sum)
            },
            |_| Window::Rect,
        );
        self.convolve(audio, &responses)
    }

    /// Uniform noise in [-1, 1) through the filter `exp(c) / 128`.
    fn noise(&mut self, uniform: &[f32], controls: &[f32], frames: usize) -> Vec<f64> {
        let bins = controls.len() / frames;
        let noise: Vec<f64> = uniform.iter().map(|u| *u as f64 * 2.0 - 1.0).collect();
        let responses = self.responses(
            frames,
            bins,
            |t, k| Complex::new((controls[t * bins + k] as f64).exp() / 128.0, 0.0),
            |_| Window::Hann,
        );
        self.convolve(&noise, &responses)
    }

    /// Synthesizes `frames * block_size` samples from the model's controls
    /// (`[frames, control_size]`), its F0 (`[frames]`, Hz) and uniform noise in
    /// [0, 1) (`frames * block_size` values, the reference's `torch.rand_like`).
    pub fn synthesize(
        &mut self,
        info: &DdspInfo,
        controls: &[f32],
        f0_frames: &[f32],
        uniform: &[f32],
    ) -> Vec<f32> {
        let frames = f0_frames.len();
        let size = info.control_size();
        let bs = info.block_size;
        let rate = info.sample_rate as f32;
        let split = |which: usize| -> Vec<f32> {
            let start: usize = info.controls[..which].iter().sum();
            let width = info.controls[which];
            (0..frames)
                .flat_map(|t| {
                    controls[t * size + start..t * size + start + width]
                        .iter()
                        .copied()
                })
                .collect()
        };
        let excitation = Excitation::new(f0_frames, info);
        let signal: Vec<f64> = match info.synth {
            Synth::Sins => {
                let harmonics = info.controls[0];
                let amplitudes: Vec<f32> = split(0)
                    .chunks(harmonics)
                    .zip(f0_frames)
                    .flat_map(|(row, f0)| {
                        row.iter().enumerate().map(move |(k, c)| {
                            let pitch = f0 * (k + 1) as f32;
                            let audible = if pitch < rate / 2.0 { 1.0 } else { 0.0 };
                            c.exp() / 128.0 * (audible + 1e-7)
                        })
                    })
                    .collect();
                let amplitudes = upsample(&amplitudes, harmonics, bs);
                let sinusoids: Vec<f64> = excitation
                    .cycle
                    .iter()
                    .enumerate()
                    .map(|(s, x)| {
                        let phase = 2.0 * std::f32::consts::PI * x;
                        let row = &amplitudes[s * harmonics..(s + 1) * harmonics];
                        row.iter()
                            .enumerate()
                            .map(|(k, a)| ((phase * (k + 1) as f32).sin() * a) as f64)
                            .sum()
                    })
                    .collect();
                let harmonic = self.group_delay(&sinusoids, &split(1), frames);
                let noise = self.noise(uniform, &split(2), frames);
                harmonic.iter().zip(&noise).map(|(h, n)| h + n).collect()
            }
            Synth::CombSub => {
                let comb = combtooth(&excitation, rate);
                let harmonic = self.group_delay(&comb, &split(0), frames);
                let magnitudes = split(1);
                let bins = info.controls[1];
                let responses = self.responses(
                    frames,
                    bins,
                    |t, k| Complex::new((magnitudes[t * bins + k] as f64).exp(), 0.0),
                    |t| Window::Dynamic(1.5 * rate / (f0_frames[t] + 1e-3)),
                );
                let harmonic = self.convolve(&harmonic, &responses);
                let noise = self.noise(uniform, &split(2), frames);
                harmonic.iter().zip(&noise).map(|(h, n)| h + n).collect()
            }
            Synth::CombSubFast => {
                let comb = combtooth(&excitation, rate);
                let (magnitude, phase, noise_magnitude) = (split(0), split(1), split(2));
                let bins = bs + 1;
                let size = 2 * bs;
                let window: Vec<f64> = (0..size)
                    .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / size as f64).cos()).sqrt())
                    .collect();
                let mut out = vec![0.0; (frames + 2) * bs];
                for k in 0..=frames {
                    let t = k.min(frames - 1);
                    let frame = |signal: &dyn Fn(usize) -> f64| -> Vec<Complex<f64>> {
                        (0..size)
                            .map(|i| {
                                let p = k * bs + i;
                                let v = if p >= bs && p - bs < frames * bs {
                                    signal(p - bs)
                                } else {
                                    0.0
                                };
                                Complex::new(v * window[i], 0.0)
                            })
                            .collect()
                    };
                    let mut comb_fft = frame(&|i| comb[i]);
                    let mut noise_fft = frame(&|i| uniform[i] as f64 * 2.0 - 1.0);
                    self.fft(&mut comb_fft);
                    self.fft(&mut noise_fft);
                    let spectrum: Vec<Complex<f64>> = (0..bins)
                        .map(|b| {
                            let m = magnitude[t * bins + b] as f64;
                            let p = phase[t * bins + b] as f64;
                            let source = Complex::from_polar(m.exp(), PI * p);
                            let noise = (noise_magnitude[t * bins + b] as f64).exp() / 128.0;
                            comb_fft[b] * source + noise_fft[b] * noise
                        })
                        .collect();
                    let frame_out = self.irfft(&spectrum);
                    for (i, v) in frame_out.iter().enumerate() {
                        out[k * bs + i] += v * window[i];
                    }
                }
                out[bs..bs + frames * bs].to_vec()
            }
        };
        signal.into_iter().map(|v| v as f32).collect()
    }
}

/// The combtooth excitation: `sinc(sample_rate * x / (f0 + 1e-3))`.
fn combtooth(excitation: &Excitation, rate: f32) -> Vec<f64> {
    excitation
        .cycle
        .iter()
        .zip(&excitation.f0)
        .map(|(x, f0)| {
            let v = rate * x / (f0 + 1e-3);
            if v == 0.0 {
                1.0
            } else {
                let pv = std::f32::consts::PI * v;
                (pv.sin() / pv) as f64
            }
        })
        .collect()
}

// ------------------------------------------------------------------------------
// NSF-HiFiGAN's mel spectrogram
// ------------------------------------------------------------------------------

/// The log-mel spectrogram NSF-HiFiGAN takes, as its `STFT.get_mel`: a 2048-point
/// Hann STFT with hop 512 (reflect padding of 768 at both ends, not centred),
/// magnitudes through librosa's Slaney mel filters (40 Hz to 16 kHz), then
/// `ln(max(x, 1e-5))`.
pub struct MelExtractor {
    basis: Vec<f64>,
    window: Vec<f64>,
    planner: FftPlanner<f64>,
}

impl Default for MelExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl MelExtractor {
    pub fn new() -> Self {
        Self {
            basis: mel_basis(VOCODER_RATE as f64, N_FFT, MEL_BINS, MEL_FMIN, MEL_FMAX),
            window: (0..N_FFT)
                .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / N_FFT as f64).cos())
                .collect(),
            planner: FftPlanner::new(),
        }
    }

    /// `[frames, 128]` for 44.1 kHz `audio`.
    pub fn extract(&mut self, audio: &[f32]) -> Vec<f32> {
        let n = audio.len() as isize;
        let left = ((N_FFT - VOCODER_HOP) / 2) as isize;
        let right = ((N_FFT - VOCODER_HOP).div_ceil(2) as isize).max(N_FFT as isize - n - left);
        let reflect = right < n;
        let at = |i: isize| -> f64 {
            let mut j = i - left;
            if reflect {
                while j < 0 || j >= n {
                    j = if j < 0 { -j } else { 2 * (n - 1) - j };
                }
                audio[j as usize] as f64
            } else if j < 0 || j >= n {
                0.0
            } else {
                audio[j as usize] as f64
            }
        };
        let padded = n + left + right;
        let frames = ((padded - N_FFT as isize) / VOCODER_HOP as isize + 1).max(0) as usize;
        let bins = N_FFT / 2 + 1;
        let fft = self.planner.plan_fft_forward(N_FFT);
        let mut buffer = vec![Complex::new(0.0, 0.0); N_FFT];
        let mut magnitudes = vec![0.0; bins];
        let mut mel = Vec::with_capacity(frames * MEL_BINS);
        for f in 0..frames {
            for (i, slot) in buffer.iter_mut().enumerate() {
                *slot = Complex::new(at((f * VOCODER_HOP + i) as isize) * self.window[i], 0.0);
            }
            fft.process(&mut buffer);
            for (m, v) in magnitudes.iter_mut().zip(&buffer) {
                *m = (v.re * v.re + v.im * v.im + 1e-9).sqrt();
            }
            for row in self.basis.chunks(bins) {
                let value: f64 = row.iter().zip(&magnitudes).map(|(w, m)| w * m).sum();
                mel.push(value.max(1e-5).ln() as f32);
            }
        }
        mel
    }
}

fn hz_to_mel(hz: f64) -> f64 {
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        hz / f_sp
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_sp * mel
    }
}

/// librosa's `filters.mel` (Slaney scale and area normalization), `[mels, bins]`.
fn mel_basis(rate: f64, n_fft: usize, mels: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    let bins = n_fft / 2 + 1;
    let (low, high) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let points: Vec<f64> = (0..mels + 2)
        .map(|i| mel_to_hz(low + (high - low) * i as f64 / (mels + 1) as f64))
        .collect();
    let mut basis = vec![0.0; mels * bins];
    for m in 0..mels {
        let (left, centre, right) = (points[m], points[m + 1], points[m + 2]);
        let norm = 2.0 / (right - left);
        for k in 0..bins {
            let hz = rate / 2.0 * k as f64 / (bins - 1) as f64;
            let lower = (hz - left) / (centre - left);
            let upper = (right - hz) / (right - centre);
            basis[m * bins + k] = lower.min(upper).max(0.0) * norm;
        }
    }
    // librosa builds the filters in float32.
    basis.iter().map(|v| *v as f32 as f64).collect()
}

/// Uniform values in [0, 1) from SplitMix64, for the synthesizer's noise.
pub struct Uniform {
    state: u64,
}

impl Uniform {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn fill(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| {
                self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = self.state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                (z >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect()
    }
}

#[cfg(any(feature = "onnx", feature = "onnx-static"))]
pub use engine::DdspSvc;

#[cfg(any(feature = "onnx", feature = "onnx-static"))]
mod engine {
    use super::*;
    use crate::{
        config::{Device, ModelManifest, Settings},
        diffusion_svc::{fill_f0, mask_below, shallow_diffusion, Noise},
        dsp::{self, Sola},
        engine::Converter,
        rvc::{embed, open_session, Detectors},
    };
    use anyhow::{Context, Result};
    use ort::{session::Session, value::Tensor};
    use std::path::Path;

    struct Diffusion {
        info: Info,
        condition: Session,
        denoiser: Session,
    }

    /// A DDSP-SVC voice, converting block by block like the RVC engine: the same
    /// history, noise gate and SOLA crossfade.
    pub struct DdspSvc {
        manifest: ModelManifest,
        info: DdspInfo,
        ddsp: Session,
        diffusion: Option<Diffusion>,
        vocoder: Session,
        embedder: Session,
        detectors: Detectors,
        synthesizer: Synthesizer,
        mel: MelExtractor,
        uniform: Uniform,
        noise: Noise,
        history: Vec<f32>,
        sola: Sola,
        geometry: Option<(u32, u32, u32, u32, u32)>,
        gate_hold: u32,
        gate_open: bool,
    }

    fn metadata<T: serde::de::DeserializeOwned>(session: &Session, key: &str) -> Result<T> {
        let metadata = session.metadata()?;
        let text = metadata
            .custom(key)?
            .with_context(|| format!("The model has no {key} metadata; import the voice again"))?;
        serde_json::from_str(&text).with_context(|| format!("Read the model's {key} metadata"))
    }

    impl DdspSvc {
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
            let ddsp = open(&manifest.model, "the DDSP-SVC model")?;
            let info: DdspInfo = metadata(&ddsp, "ddsp_svc")?;
            let diffusion = match (&manifest.condition, &manifest.denoiser) {
                (Some(condition), Some(denoiser)) => {
                    let condition = open(condition, "the DDSP-SVC diffusion conditioning")?;
                    let denoiser = open(denoiser, "the DDSP-SVC diffusion denoiser")?;
                    let info: Info = metadata(&denoiser, "diffusion_svc")?;
                    Some(Diffusion {
                        info,
                        condition,
                        denoiser,
                    })
                }
                _ => None,
            };
            let vocoder = open(
                manifest
                    .vocoder
                    .as_deref()
                    .context("No vocoder in model.json")?,
                "NSF-HiFiGAN",
            )?;
            let embedder = open(&manifest.embedder, "ContentVec")?;
            let rmvpe = open(&manifest.pitch, "RMVPE")?;
            log::info!(
                "DDSP-SVC voice \"{}\" on {} ({:?}, {} speakers, {})",
                manifest.name,
                device.label(),
                info.synth,
                info.speakers,
                if diffusion.is_some() {
                    "with a diffusion model"
                } else {
                    "no diffusion model"
                }
            );
            let detectors = Detectors::new(rmvpe, &manifest.pitch, threads, device);
            Ok(Self {
                manifest,
                info,
                ddsp,
                diffusion,
                vocoder,
                embedder,
                detectors,
                synthesizer: Synthesizer::new(),
                mel: MelExtractor::new(),
                uniform: Uniform::new(0xdd5b_5eed),
                noise: Noise::new(0x5eed_d1ff),
                history: Vec::new(),
                sola: Sola::default(),
                geometry: None,
                gate_hold: 0,
                gate_open: false,
            })
        }

        fn vocode(&mut self, mel: Vec<f32>, f0: &[f32]) -> Result<Vec<f32>> {
            let frames = mel.len() / MEL_BINS;
            let f0: Vec<f32> = (0..frames)
                .map(|i| f0.get(i).or(f0.last()).copied().unwrap_or(0.0))
                .collect();
            let outputs = self.vocoder.run(ort::inputs! {
                "mel" => Tensor::from_array(([1, frames, MEL_BINS], mel.into_boxed_slice()))?,
                "f0" => Tensor::from_array(([1, frames], f0.into_boxed_slice()))?,
            })?;
            Ok(outputs[0].try_extract_tensor::<f32>()?.1.to_vec())
        }

        /// The NSF-HiFiGAN enhancer, as the reference's `Enhancer.enhance` with
        /// the "auto" key: audio whose pitch rises above 760 Hz is treated as
        /// slower, so the vocoder stays in its range.
        fn enhance(&mut self, audio: &[f32], f0: &[f32]) -> Result<Vec<f32>> {
            let key = adaptive_key(f0);
            let rate = adaptive_rate(key);
            if rate == self.info.sample_rate && self.info.block_size == VOCODER_HOP {
                let mel = self.mel.extract(audio);
                return self.vocode(mel, f0);
            }
            let resampled = dsp::resample(audio, self.info.sample_rate, rate);
            let frames = resampled.len() / VOCODER_HOP + 1;
            let f0 = enhancer_f0(
                f0,
                self.info.block_size,
                self.info.sample_rate,
                rate,
                frames,
            );
            let mel = self.mel.extract(&resampled);
            let out = self.vocode(mel, &f0)?;
            Ok(dsp::resample(&out, rate, VOCODER_RATE))
        }

        /// The voice's audio (44.1 kHz) for 16 kHz `audio`.
        fn convert(&mut self, audio: &[f32], settings: &Settings) -> Result<Vec<f32>> {
            let info = self.info.clone();
            let model_audio = dsp::resample(audio, 16_000, info.sample_rate);
            let frames = frame_count(model_audio.len(), info.block_size);
            let volume = volume(&model_audio, info.block_size);
            let features = embed(
                &mut self.embedder,
                &self.manifest.embedder_input,
                &self.manifest.embedder_output,
                audio,
                info.units,
            )?;
            let units = align_units(
                &features,
                info.units,
                frames,
                info.sample_rate,
                info.block_size,
            );
            let detected = self.detectors.detect(audio, frames, settings.f0_detector)?;
            let f0 = fill_f0(&detected, settings.semitones);
            let excitation = Excitation::new(&f0, &info);
            let phase = excitation.phase_frames(info.block_size);
            let speaker = settings.speaker_id.clamp(0, info.speakers as i64 - 1);
            let feeds = |session: &Session,
                         phase: Option<&[f32]>|
             -> Result<
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
                if let Some(phase) = phase {
                    feeds.push((
                        "phase".into(),
                        Tensor::from_array(([1, frames, 1], phase.to_vec().into_boxed_slice()))?
                            .into(),
                    ));
                }
                if session.inputs.iter().any(|i| i.name == "speaker") {
                    feeds.push((
                        "speaker".into(),
                        Tensor::from_array(([1], vec![speaker].into_boxed_slice()))?.into(),
                    ));
                }
                Ok(feeds)
            };
            let inputs = feeds(&self.ddsp, Some(&phase))?;
            let controls = self.ddsp.run(inputs)?[0]
                .try_extract_tensor::<f32>()?
                .1
                .to_vec();
            anyhow::ensure!(
                controls.len() == frames * info.control_size(),
                "The DDSP-SVC model returned {} values for {frames} frames",
                controls.len()
            );
            let uniform = self.uniform.fill(frames * info.block_size);
            let signal = self.synthesizer.synthesize(&info, &controls, &f0, &uniform);
            let gate = mask_below(&volume, info.block_size, MASK_DB);
            let masked = |audio: Vec<f32>| -> Vec<f32> {
                audio
                    .iter()
                    .zip(gate.iter().chain(std::iter::repeat(&1.0)))
                    .map(|(v, m)| v * m)
                    .collect()
            };
            let output = match self.diffusion.as_mut().filter(|_| settings.diffusion) {
                Some(diffusion) => {
                    let mel = self.mel.extract(&signal);
                    let mel_frames = mel.len() / MEL_BINS;
                    let mel_rows = mel_frames.min(frames);
                    let gt = mel[..mel_rows * MEL_BINS].to_vec();
                    let inputs = feeds(&diffusion.condition, None)?;
                    let condition = diffusion.condition.run(inputs)?[0]
                        .try_extract_tensor::<f32>()?
                        .1
                        .to_vec();
                    let hidden = condition.len() / frames;
                    // The mel has a frame per model frame; cut the conditioning to it
                    // should the two ever differ.
                    let mut cut = vec![0.0; hidden * mel_rows];
                    for h in 0..hidden {
                        cut[h * mel_rows..(h + 1) * mel_rows]
                            .copy_from_slice(&condition[h * frames..h * frames + mel_rows]);
                    }
                    let (k_step, speedup) = steps_for(
                        &diffusion.info,
                        settings.diffusion_steps,
                        settings.diffusion_sampler_steps,
                    );
                    let denoiser = &mut diffusion.denoiser;
                    let mut denoise = |x: &[f32], step: f32| -> Result<Vec<f32>> {
                        let outputs = denoiser.run(ort::inputs! {
                            "x" => Tensor::from_array(([1, 1, MEL_BINS, mel_rows], x.to_vec().into_boxed_slice()))?,
                            "step" => Tensor::from_array(([1], vec![step].into_boxed_slice()))?,
                            "condition" => Tensor::from_array(([1, hidden, mel_rows], cut.clone().into_boxed_slice()))?,
                        })?;
                        Ok(outputs[0].try_extract_tensor::<f32>()?.1.to_vec())
                    };
                    let info = diffusion.info.clone();
                    let mut mel = shallow_diffusion(
                        &info,
                        &gt,
                        k_step,
                        speedup,
                        &mut self.noise,
                        &mut denoise,
                    )?;
                    if self.info.log10_mel {
                        mel.iter_mut().for_each(|v| *v *= std::f32::consts::LOG10_E);
                    }
                    masked(self.vocode(mel, &f0)?)
                }
                None if settings.enhancer => {
                    let signal = masked(signal);
                    self.enhance(&signal, &f0)?
                }
                None => masked(signal),
            };
            anyhow::ensure!(
                output.iter().all(|v| v.is_finite()),
                "Non-finite DDSP-SVC output"
            );
            Ok(output
                .iter()
                .map(|v| (v * self.manifest.output_scale).clamp(-1.0, 1.0))
                .collect())
        }
    }

    impl Converter for DdspSvc {
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
            let aligned = (audio.len() as u64 * VOCODER_RATE as u64 / 16_000) as usize;
            voice.truncate(aligned);
            let converted = dsp::resample(&voice, VOCODER_RATE, rate);
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

    fn info(synth: Synth, controls: [usize; 3]) -> DdspInfo {
        DdspInfo {
            sample_rate: 44_100,
            block_size: 64,
            units: 4,
            speakers: 1,
            synth,
            controls,
            log10_mel: false,
        }
    }

    #[test]
    fn volume_reflects_half_a_hop() {
        // A constant signal has its own level in every frame.
        let v = volume(&[0.5; 100], 10);
        assert_eq!(v.len(), 11);
        assert!(v.iter().all(|x| (x - 0.5).abs() < 1e-6), "{v:?}");
        // numpy's reflect at the start: [3,2,1,0,1,2,...] squared.
        let ramp: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let v = volume(&ramp, 4);
        let want = ((4.0 + 1.0 + 0.0 + 1.0) / 4.0f32).sqrt();
        assert!((v[0] - want).abs() < 1e-5, "{} vs {want}", v[0]);
    }

    #[test]
    fn units_are_picked_by_rounded_time() {
        let features: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let units = align_units(&features, 1, 6, 44_100, 512);
        // ratio = (512 / 44100) / 0.02 = 0.5805
        assert_eq!(units, vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn the_excitation_wraps_once_a_cycle() {
        let info = info(Synth::CombSubFast, [65, 65, 65]);
        // 441 Hz at 44.1 kHz: one cycle every 100 samples.
        let e = Excitation::new(&[441.0; 10], &info);
        assert_eq!(e.cycle.len(), 640);
        assert!((e.cycle[49] - 0.5).abs() < 1e-4 || (e.cycle[49] + 0.5).abs() < 1e-4);
        assert!(e.cycle[99].abs() < 1e-4);
        assert!(e.cycle.iter().all(|x| x.abs() <= 0.5 + 1e-6));
    }

    #[test]
    fn mel_filters_match_librosa() {
        // Values from librosa.filters.mel(sr=44100, n_fft=2048, n_mels=128, fmin=40,
        // fmax=16000) (librosa 0.10).
        let basis = mel_basis(44_100.0, 2048, 128, 40.0, 16_000.0);
        let row = |m: usize| &basis[m * 1025..(m + 1) * 1025];
        let (peak, height) =
            row(0).iter().enumerate().fold(
                (0, 0.0),
                |best, (k, v)| if *v > best.1 { (k, *v) } else { best },
            );
        assert_eq!(peak, 3);
        assert!((height - 0.030_752_37).abs() < 1e-7, "{height}");
        assert_eq!(row(127).iter().rposition(|v| *v > 0.0), Some(743));
        let sum: f64 = row(64).iter().sum();
        assert!((sum - 0.046_984_59).abs() < 1e-6, "{sum}");
        assert!(basis.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn synthesizers_are_silent_without_excitation_or_noise() {
        // Zero noise (uniform 0.5) and very low harmonic magnitudes: near silence,
        // and each synthesizer returns frames * block_size samples.
        let frames = 5;
        for (synth, controls) in [
            (Synth::Sins, [8, 6, 6]),
            (Synth::CombSub, [6, 6, 6]),
            (Synth::CombSubFast, [65, 65, 65]),
        ] {
            let info = info(synth, controls);
            let mut c = vec![-30.0; frames * info.control_size()];
            if synth == Synth::CombSubFast {
                // harmonic phase: zero
                for t in 0..frames {
                    for k in 65..130 {
                        c[t * 195 + k] = 0.0;
                    }
                }
            }
            let out =
                Synthesizer::new().synthesize(&info, &c, &[220.0; 5], &vec![0.5; frames * 64]);
            assert_eq!(out.len(), frames * 64, "{synth:?}");
            assert!(out.iter().all(|v| v.abs() < 1e-6), "{synth:?}: {out:?}");
        }
    }

    #[test]
    fn the_enhancer_key_follows_the_highest_pitch() {
        assert_eq!(adaptive_key(&[200.0, 500.0]), 0);
        assert_eq!(adaptive_key(&[761.0]), 1);
        assert_eq!(adaptive_rate(0), 44_100);
        assert_eq!(adaptive_rate(1), 46_700);
    }

    #[test]
    fn default_diffusion_is_120_deep_in_6_steps() {
        let info = Info {
            sample_rate: 44_100,
            block_size: 512,
            units: 768,
            speakers: 1,
            k_step_max: 1000,
            spec_min: -12.0,
            spec_max: 2.0,
            betas: vec![0.01; 1000],
            alphas_cumprod: vec![0.5; 1000],
        };
        assert_eq!(steps_for(&info, 0, 0), (120, 20));
        assert_eq!(steps_for(&info, 200, 10), (200, 20));
    }
}
