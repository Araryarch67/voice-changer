use std::collections::VecDeque;

/// Computes root-mean-square amplitude of `audio`, returning `0.0` when empty.
pub fn rms(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt()
}

/// Applies a raised-cosine fade from `start_gain` to `end_gain` in place.
pub fn fade(samples: &mut [f32], start_gain: f32, end_gain: f32) {
    let len = samples.len();
    if len == 0 {
        return;
    }
    if len == 1 {
        samples[0] *= 0.5 * (start_gain + end_gain);
        return;
    }
    let denom = (len - 1) as f32;
    for (i, sample) in samples.iter_mut().enumerate() {
        let t = i as f32 / denom;
        let curve = (t * std::f32::consts::FRAC_PI_2).sin().powi(2);
        *sample *= start_gain + (end_gain - start_gain) * curve;
    }
}

/// Windowed-sinc resampling for contextual inference windows (not the audio callback).
pub fn resample(audio: &[f32], from: u32, to: u32) -> Vec<f32> {
    assert!(from > 0 && to > 0);
    if from == to || audio.is_empty() {
        return audio.to_vec();
    }
    let len = (audio.len() as u64 * to as u64 / from as u64) as usize;
    let cutoff = (to as f64 / from as f64).min(1.0) * 0.95;
    // Scale kernel half-width inversely with cutoff so steep downsampling (e.g. 192k -> 16k)
    // retains ~16 sinc lobes for anti-aliasing while clamping extreme ratios.
    let half_width = (16.0 / cutoff).clamp(16.0, 512.0);
    let radius = half_width.ceil() as isize;
    let max_index = audio.len() as isize - 1;
    (0..len)
        .map(|i| {
            let position = i as f64 * from as f64 / to as f64;
            let center = position.floor() as isize;
            let start = (center - radius).max(0);
            let end = (center + radius).min(max_index);
            let mut sum = 0.0;
            let mut weights = 0.0;
            for j in start..=end {
                let distance = position - j as f64;
                if distance.abs() > half_width {
                    continue;
                }
                let x = std::f64::consts::PI * distance * cutoff;
                let sinc = if x.abs() < 1e-8 { 1.0 } else { x.sin() / x };
                let window = 0.5 + 0.5 * (std::f64::consts::PI * distance / half_width).cos();
                let weight = sinc * window * cutoff;
                sum += audio[j as usize] as f64 * weight;
                weights += weight;
            }
            if weights.abs() < 1e-8 {
                0.0
            } else {
                (sum / weights) as f32
            }
        })
        .collect()
}

fn catmull_rom(y0: f32, y1: f32, y2: f32, y3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * y1)
        + (-y0 + y2) * t
        + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t2
        + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t3)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.max(1)
}

/// Continuous rate converter for independent input/output device clocks.
///
/// Uses 4-point Catmull-Rom cubic Hermite interpolation with exact rational
/// phase tracking so output never drifts across arbitrary chunk boundaries.
pub struct StreamResampler {
    buffer: VecDeque<f32>,
    pos: usize,
    num: u64,
    step: u64,
    den: u64,
    passthrough: bool,
}
impl StreamResampler {
    /// Creates a stream resampler from `from` Hz to `to` Hz.
    pub fn new(from: u32, to: u32) -> Self {
        assert!(from > 0 && to > 0);
        let g = gcd(from as u64, to as u64);
        let mut buffer = VecDeque::new();
        buffer.push_back(0.0);
        Self {
            buffer,
            pos: 1,
            num: 0,
            step: from as u64 / g,
            den: to as u64 / g,
            passthrough: from == to,
        }
    }

    /// Clears buffered history and resets phase to the start of a stream.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.buffer.push_back(0.0);
        self.pos = 1;
        self.num = 0;
    }

    /// Appends input `samples` and returns all newly interpolated output samples.
    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        if self.passthrough {
            return samples.to_vec();
        }
        self.buffer.extend(samples);
        let mut out = Vec::new();
        while self.pos + 2 < self.buffer.len() {
            let t = self.num as f32 / self.den as f32;
            out.push(catmull_rom(
                self.buffer[self.pos - 1],
                self.buffer[self.pos],
                self.buffer[self.pos + 1],
                self.buffer[self.pos + 2],
                t,
            ));
            let next = self.num + self.step;
            self.pos += (next / self.den) as usize;
            self.num = next % self.den;
        }
        let consumed = self.pos.saturating_sub(1).min(self.buffer.len());
        self.buffer.drain(..consumed);
        self.pos -= consumed;
        out
    }
}

/// Stateful 2nd-order Butterworth high-pass biquad filter.
///
/// Strips DC offset, plosives, and sub-bass rumble while preserving the speech passband.
/// Coefficients and filter state use `f64` for stability at high sample rates (e.g. 192 kHz).
#[derive(Clone, Debug)]
pub struct HighPass {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl HighPass {
    /// Default voice high-pass cutoff in Hz (~80 Hz strips rumble without thinning vocals).
    pub const DEFAULT_VOICE_CUTOFF_HZ: f32 = 80.0;

    /// Creates a 2nd-order Butterworth high-pass filter at `cutoff_hz` for `sample_rate`.
    ///
    /// Non-positive or non-finite cutoffs configure a transparent passthrough.
    pub fn new(cutoff_hz: f32, sample_rate: u32) -> Self {
        assert!(sample_rate > 0);
        if !cutoff_hz.is_finite() || cutoff_hz <= 0.0 {
            return Self {
                b0: 1.0,
                b1: 0.0,
                b2: 0.0,
                a1: 0.0,
                a2: 0.0,
                z1: 0.0,
                z2: 0.0,
            };
        }
        let sr = sample_rate as f64;
        let fc = (cutoff_hz as f64).min(sr * 0.49);
        let w0 = 2.0 * std::f64::consts::PI * fc / sr;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 * std::f64::consts::FRAC_1_SQRT_2;
        let a0 = 1.0 + alpha;
        let b0 = ((1.0 + cos_w0) * 0.5) / a0;
        Self {
            b0,
            b1: -(1.0 + cos_w0) / a0,
            b2: b0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Creates a high-pass filter tuned for voice input (~80 Hz cutoff).
    pub fn default_voice(sample_rate: u32) -> Self {
        Self::new(Self::DEFAULT_VOICE_CUTOFF_HZ, sample_rate)
    }

    /// Filters `samples` in place, preserving state across consecutive chunks.
    pub fn process(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            let x = *sample as f64;
            let y = self.b0 * x + self.z1;
            self.z1 = self.b1 * x - self.a1 * y + self.z2;
            self.z2 = self.b2 * x - self.a2 * y;
            *sample = y as f32;
        }
    }

    /// Resets internal filter delay state to zero.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

pub fn coarse_pitch(f0: f32) -> i64 {
    let min = 1127.0_f32 * (1.0_f32 + 50.0 / 700.0).ln();
    let max = 1127.0_f32 * (1.0_f32 + 1100.0 / 700.0).ln();
    if f0 <= 0.0 {
        return 1;
    }
    let mel = 1127.0 * (1.0 + f0 / 700.0).ln();
    (((mel - min) * 254.0 / (max - min)) + 1.0)
        .round()
        .clamp(1.0, 255.0) as i64
}

#[derive(Default)]
pub struct Sola {
    tail: Vec<f32>,
}
impl Sola {
    pub fn reset(&mut self) {
        self.tail.clear();
    }
    /// `audio` contains overlap + alignment search + the new block at the output rate.
    pub fn process(
        &mut self,
        audio: &[f32],
        block: usize,
        overlap: usize,
        search: usize,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(
            block > 0 && overlap > 0 && overlap <= block,
            "Invalid SOLA dimensions"
        );
        anyhow::ensure!(
            audio.len() >= block + overlap + search,
            "Model output too short for SOLA: {} < {}",
            audio.len(),
            block + overlap + search
        );
        let audio = &audio[audio.len() - (block + overlap + search)..];
        let mut offset = 0;
        if self.tail.len() == overlap {
            let mut best = f32::NEG_INFINITY;
            for shift in 0..=search {
                let candidate = &audio[shift..shift + overlap];
                let energy = candidate.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-8;
                let score = candidate
                    .iter()
                    .zip(&self.tail)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / energy;
                if score > best {
                    best = score;
                    offset = shift;
                }
            }
        }
        let mut result = audio[offset..offset + block].to_vec();
        if self.tail.len() == overlap {
            for (i, value) in result.iter_mut().take(overlap).enumerate() {
                let blend = if overlap == 1 {
                    1.0
                } else {
                    i as f32 / (overlap - 1) as f32
                };
                let blend = (blend * std::f32::consts::FRAC_PI_2).sin().powi(2);
                *value = *value * blend + self.tail[i] * (1.0 - blend);
            }
        }
        self.tail = audio[offset + block..offset + block + overlap].to_vec();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq_hz: f32, sample_rate: u32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn continuous_rate_conversion_does_not_drift_between_chunks() {
        let input: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.02).sin()).collect();
        let mut whole = StreamResampler::new(44100, 48000);
        let expected = whole.push(&input);
        let mut stream = StreamResampler::new(44100, 48000);
        let actual: Vec<_> = input.chunks(137).flat_map(|s| stream.push(s)).collect();
        assert_eq!(actual.len(), expected.len());
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-5));
    }

    #[test]
    fn stream_resampler_small_chunks_and_cubic_accuracy() {
        let input = sine_wave(1000.0, 44100, 4410);
        let mut whole = StreamResampler::new(44100, 48000);
        let expected = whole.push(&input);

        let mut single = StreamResampler::new(44100, 48000);
        assert!(single.push(&[]).is_empty());
        let mut single_out = Vec::new();
        for &sample in &input {
            single_out.extend(single.push(&[sample]));
        }
        assert_eq!(single_out, expected);

        // Downsampling in small chunks also stays bit-identical across chunking.
        let high = sine_wave(1000.0, 192000, 1920);
        let mut down_whole = StreamResampler::new(192000, 48000);
        let down_expected = down_whole.push(&high);
        let mut down_chunked = StreamResampler::new(192000, 48000);
        let down_actual: Vec<_> = high.chunks(3).flat_map(|c| down_chunked.push(c)).collect();
        assert_eq!(down_actual, down_expected);

        // Verify Catmull-Rom accuracy against the analytic sine at output sample times.
        let interior = &expected[16..expected.len() - 16];
        let max_err = interior
            .iter()
            .enumerate()
            .map(|(idx, &val)| {
                let out_idx = idx + 16;
                let ideal = (2.0 * PI * 1000.0 * out_idx as f32 / 48000.0).sin();
                (val - ideal).abs()
            })
            .fold(0.0_f32, f32::max);
        assert!(max_err < 1e-3, "max cubic error was {max_err}");

        // Passthrough rate conversion returns input unchanged.
        let mut same = StreamResampler::new(48000, 48000);
        assert_eq!(same.push(&[0.1, -0.2, 0.3]), vec![0.1, -0.2, 0.3]);
    }

    #[test]
    fn resampler_preserves_dc_and_rejects_aliasing() {
        let dc = resample(&vec![0.25; 4800], 48000, 16000);
        assert_eq!(dc.len(), 1600);
        assert!(dc.iter().all(|v| (v - 0.25).abs() < 1e-5));
        let input = sine_wave(14000.0, 48000, 4800);
        let output = resample(&input, 48000, 16000);
        assert!(rms(&output[32..output.len() - 32]) < 0.02);
    }

    #[test]
    fn resampler_high_rate_anti_aliasing_and_passband() {
        // 192 kHz -> 16 kHz (12x downsample)
        let dc_192 = resample(&vec![0.5; 19200], 192000, 16000);
        assert_eq!(dc_192.len(), 1600);
        assert!(dc_192.iter().all(|v| (v - 0.5).abs() < 1e-5));

        let passband = sine_wave(1000.0, 192000, 19200);
        let pass_out = resample(&passband, 192000, 16000);
        let pass_rms = rms(&pass_out[64..pass_out.len() - 64]);
        let expected_rms = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (pass_rms - expected_rms).abs() < 0.01,
            "passband RMS drifted: {pass_rms}"
        );

        for alias_hz in [10000.0, 24000.0, 60000.0] {
            let alias = sine_wave(alias_hz, 192000, 19200);
            let out = resample(&alias, 192000, 16000);
            let alias_rms = rms(&out[64..out.len() - 64]);
            assert!(
                alias_rms < 0.01,
                "192k -> 16k leaked {alias_hz} Hz tone: rms={alias_rms}"
            );
        }

        // 96 kHz -> 16 kHz (6x downsample)
        let alias_96 = sine_wave(12000.0, 96000, 9600);
        let out_96 = resample(&alias_96, 96000, 16000);
        assert!(rms(&out_96[64..out_96.len() - 64]) < 0.01);
    }

    #[test]
    fn high_pass_rejects_dc_and_sub_bass_preserving_speech() {
        for &rate in &[48000_u32, 192000_u32] {
            let mut hp = HighPass::default_voice(rate);
            let mut dc = vec![0.4; rate as usize];
            hp.process(&mut dc);
            assert!(rms(&dc[rate as usize / 2..]) < 1e-4);

            hp.reset();
            let mut rumble = sine_wave(20.0, rate, rate as usize);
            hp.process(&mut rumble);
            let rumble_gain = rms(&rumble[rate as usize / 2..]) / std::f32::consts::FRAC_1_SQRT_2;
            assert!(
                rumble_gain < 0.08,
                "20 Hz gain too high at {rate}: {rumble_gain}"
            );

            hp.reset();
            let speech = sine_wave(1000.0, rate, rate as usize);
            let mut whole = speech.clone();
            hp.process(&mut whole);
            let speech_gain = rms(&whole[rate as usize / 4..]) / std::f32::consts::FRAC_1_SQRT_2;
            assert!(
                speech_gain > 0.99 && speech_gain < 1.01,
                "1 kHz passband distorted at {rate}: {speech_gain}"
            );

            // Chunked stateful processing matches single-buffer processing.
            hp.reset();
            let mut chunked = speech;
            for chunk in chunked.chunks_mut(113) {
                hp.process(chunk);
            }
            assert!(whole.iter().zip(chunked).all(|(a, b)| (a - b).abs() < 1e-6));
        }

        // Non-positive cutoff acts as transparent passthrough.
        let mut bypass = HighPass::new(0.0, 48000);
        let mut samples = vec![0.25, -0.5, 0.75];
        bypass.process(&mut samples);
        assert_eq!(samples, vec![0.25, -0.5, 0.75]);
    }

    #[test]
    fn fade_applies_smooth_raised_cosine_envelope() {
        let mut empty: [f32; 0] = [];
        fade(&mut empty, 0.0, 1.0);

        let mut one = [2.0];
        fade(&mut one, 0.0, 1.0);
        assert!((one[0] - 1.0).abs() < 1e-6);

        let mut buf = vec![1.0; 11];
        fade(&mut buf, 0.0, 1.0);
        assert!(buf[0].abs() < 1e-6);
        assert!((buf[5] - 0.5).abs() < 1e-5);
        assert!((buf[10] - 1.0).abs() < 1e-6);
        assert!(buf.windows(2).all(|w| w[0] <= w[1] + 1e-6));
    }

    #[test]
    fn sola_aligns_shifted_periodic_audio() {
        let wave: Vec<_> = (0..400).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut sola = Sola::default();
        let first = sola.process(&wave[..140], 100, 20, 20).unwrap();
        // Model's next window starts seven samples before the expected boundary.
        let second = sola.process(&wave[93..233], 100, 20, 20).unwrap();
        assert_eq!(first.len(), 100);
        assert!(second
            .iter()
            .zip(&wave[100..200])
            .all(|(a, b)| (a - b).abs() < 1e-5));
        assert!(sola.process(&[], 100, 20, 20).is_err());
    }

    #[test]
    fn pitch_quantization_bounds() {
        assert_eq!(coarse_pitch(0.0), 1);
        assert_eq!(coarse_pitch(50.0), 1);
        assert_eq!(coarse_pitch(1100.0), 255);
        assert_eq!(coarse_pitch(10000.0), 255);
    }
}
