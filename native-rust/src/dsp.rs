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
        let tail_energy = self.tail.iter().map(|v| v * v).sum::<f32>().sqrt();
        if self.tail.len() == overlap && tail_energy > 1e-6 {
            let mut best = f32::NEG_INFINITY;
            for shift in 0..=search {
                let candidate = &audio[shift..shift + overlap];
                let cand_energy = candidate.iter().map(|v| v * v).sum::<f32>().sqrt();
                if cand_energy <= 1e-6 {
                    continue;
                }
                let corr = candidate
                    .iter()
                    .zip(&self.tail)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / (cand_energy * tail_energy);
                if corr > best {
                    best = corr;
                    offset = shift;
                }
            }
            if best < 0.15 {
                offset = 0;
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

/// Stateful post-HPF vocal dynamics leveler: gentle voice-gated AGC, soft-knee downward
/// compression for belt/transient control, and a noise-floor-aware soft downward expander.
pub struct VocalDynamics {
    rate: u32,
    sc_b0: f32,
    sc_b1: f32,
    sc_a1: f32,
    sc_x1: f32,
    sc_y1: f32,
    rms_power: f32,
    fast_power: f32,
    noise_floor_db: f32,
    quiet_ref_db: f32,
    quiet_steady_samples: u32,
    agc_gain_db: f32,
    smoothed_gain_db: f32,
    below_close_samples: u32,
    gate_closed: bool,
    gate_atten_db: f32,
}

impl VocalDynamics {
    pub fn new(rate: u32) -> Self {
        let rate = rate.max(1);
        // 1st-order 700 Hz ±2.5 dB presence tilt for the sidechain detector only (keeps chest boom
        // from over-compressing upper-mid consonants while leaving the audio path uncolored).
        let amp = 10.0_f64.powf(5.0 / 40.0);
        let k = (std::f64::consts::PI * (700.0 / rate as f64).min(0.45)).tan();
        let a0 = amp * k + 1.0;
        let sc_b0 = ((k + amp) / a0) as f32;
        let sc_b1 = ((k - amp) / a0) as f32;
        let sc_a1 = ((amp * k - 1.0) / a0) as f32;
        Self {
            rate,
            sc_b0,
            sc_b1,
            sc_a1,
            sc_x1: 0.0,
            sc_y1: 0.0,
            rms_power: 1e-10,
            fast_power: 1e-10,
            noise_floor_db: -55.0,
            quiet_ref_db: -55.0,
            quiet_steady_samples: 0,
            agc_gain_db: 0.0,
            smoothed_gain_db: 0.0,
            below_close_samples: 0,
            gate_closed: false,
            gate_atten_db: 0.0,
        }
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.rate);
    }

    /// Processes `audio` in place with `amount` in `0.0..=1.0`.
    pub fn process(&mut self, audio: &mut [f32], amount: f32) {
        let amount = amount.clamp(0.0, 1.0);
        if amount <= 0.0 || audio.is_empty() {
            return;
        }
        let sr = self.rate as f32;
        let alpha = |tau: f32| (-1.0 / (sr * tau)).exp();

        let a_rms_atk = alpha(0.010);
        let a_rms_rel = alpha(0.080);
        let a_fast_atk = alpha(0.005);
        let a_fast_rel = alpha(0.180);
        let a_agc_up = alpha(0.800);
        let a_agc_dn = alpha(0.080);
        let a_noise_up = alpha(2.500);
        let a_noise_dn = alpha(0.250);
        let a_steady_ref = alpha(0.300);
        let a_gate_open = alpha(0.004);
        let a_gate_close = alpha(0.140);
        let a_gain_up = alpha(0.025);
        let a_gain_dn = alpha(0.004);
        let hold_samples = (0.160 * sr) as u32;
        let steady_hold_samples = (0.450 * sr) as u32;

        const TARGET_ACTIVE_DB: f32 = -20.0;
        const AGC_MIN_DB: f32 = -3.0;
        const AGC_MAX_DB: f32 = 8.0;
        const COMP_THRESH_DB: f32 = -18.0;
        const COMP_RATIO: f32 = 3.0;
        const COMP_KNEE_DB: f32 = 6.0;
        const EXP_KNEE_DB: f32 = 4.0;
        const EXP_RATIO: f32 = 1.5;
        const MAX_GATE_ATTEN_DB: f32 = 12.0;

        for sample in audio.iter_mut() {
            let x = *sample;
            let sc = self.sc_b0 * x + self.sc_b1 * self.sc_x1 - self.sc_a1 * self.sc_y1;
            self.sc_x1 = x;
            self.sc_y1 = sc;
            let p = sc * sc;
            let a_rms = if p > self.rms_power {
                a_rms_atk
            } else {
                a_rms_rel
            };
            self.rms_power = a_rms * self.rms_power + (1.0 - a_rms) * p;

            let a_fast = if p > self.fast_power {
                a_fast_atk
            } else {
                a_fast_rel
            };
            self.fast_power = a_fast * self.fast_power + (1.0 - a_fast) * p;

            let level_db = 10.0 * (self.rms_power + 1e-12).log10();
            let fast_db = 10.0 * (self.fast_power + 1e-12).log10();

            // Detect sustained low-variance, low-crest stationary background in (-70, -38) dBFS so
            // changed room noise or recovery after digital silence reacquires without learning speech.
            if level_db > -70.0 && level_db < -38.0 && (fast_db - level_db) < 6.0 {
                if self.quiet_steady_samples == 0 {
                    self.quiet_ref_db = level_db;
                    self.quiet_steady_samples = 1;
                } else if (level_db - self.quiet_ref_db).abs() <= 2.5 {
                    self.quiet_ref_db =
                        a_steady_ref * self.quiet_ref_db + (1.0 - a_steady_ref) * level_db;
                    self.quiet_steady_samples = self.quiet_steady_samples.saturating_add(1);
                } else {
                    self.quiet_ref_db = level_db;
                    self.quiet_steady_samples = 1;
                }
            } else {
                self.quiet_steady_samples = 0;
            }
            let stationary_quiet = self.quiet_steady_samples >= steady_hold_samples;

            // Adapt noise floor downward whenever input drops below it, and upward either slowly
            // near the current estimate or via confirmed stationary background reacquisition.
            if level_db < self.noise_floor_db {
                let target_n = level_db.max(-72.0);
                self.noise_floor_db =
                    a_noise_dn * self.noise_floor_db + (1.0 - a_noise_dn) * target_n;
            } else if stationary_quiet {
                let target_n = level_db.min(-40.0);
                if target_n > self.noise_floor_db {
                    self.noise_floor_db =
                        a_noise_dn * self.noise_floor_db + (1.0 - a_noise_dn) * target_n;
                }
            } else if level_db < -38.0 && level_db < self.noise_floor_db + 4.5 {
                let target_n = level_db.min(-40.0);
                self.noise_floor_db =
                    a_noise_up * self.noise_floor_db + (1.0 - a_noise_up) * target_n;
            }

            // Continuous hysteresis gate: require `hold_samples` strictly below `close_thresh`
            // without accumulation across intermediate-band excursions.
            let open_thresh = self.noise_floor_db + 5.0;
            let close_thresh = self.noise_floor_db + 2.0;
            if level_db > open_thresh {
                self.below_close_samples = 0;
                self.gate_closed = false;
            } else if level_db < close_thresh {
                self.below_close_samples = self.below_close_samples.saturating_add(1);
                if self.below_close_samples >= hold_samples {
                    self.gate_closed = true;
                }
            } else {
                self.below_close_samples = 0;
            }
            let gate_target = if self.gate_closed {
                MAX_GATE_ATTEN_DB
            } else {
                0.0
            };
            let a_gate = if gate_target < self.gate_atten_db {
                a_gate_open
            } else {
                a_gate_close
            };
            self.gate_atten_db = a_gate * self.gate_atten_db + (1.0 - a_gate) * gate_target;

            // Soft downward expander below T_e.
            let t_exp = (self.noise_floor_db + 5.0).min(-38.0);
            let exp_atten_db = if level_db > t_exp + EXP_KNEE_DB * 0.5 {
                0.0
            } else if level_db >= t_exp - EXP_KNEE_DB * 0.5 {
                let d = t_exp + EXP_KNEE_DB * 0.5 - level_db;
                (EXP_RATIO - 1.0) * (d * d) / (2.0 * EXP_KNEE_DB)
            } else {
                ((EXP_RATIO - 1.0) * (t_exp - level_db)).min(MAX_GATE_ATTEN_DB)
            };
            let floor_atten_db = exp_atten_db.max(self.gate_atten_db);

            // Freeze AGC during pauses and confirmed stationary background so noise is never boosted.
            let confidently_active = level_db > self.noise_floor_db + 6.5
                && self.gate_atten_db < 1.0
                && !stationary_quiet;
            if confidently_active {
                let agc_target = (TARGET_ACTIVE_DB - level_db).clamp(AGC_MIN_DB, AGC_MAX_DB);
                let a_agc = if agc_target > self.agc_gain_db {
                    a_agc_up
                } else {
                    a_agc_dn
                };
                self.agc_gain_db = a_agc * self.agc_gain_db + (1.0 - a_agc) * agc_target;
            }

            // Downward soft-knee compressor reacts on the fast peak/RMS envelope after AGC.
            let comp_in_db = fast_db.max(level_db) + self.agc_gain_db;
            let comp_out_db = if comp_in_db < COMP_THRESH_DB - COMP_KNEE_DB * 0.5 {
                comp_in_db
            } else if comp_in_db <= COMP_THRESH_DB + COMP_KNEE_DB * 0.5 {
                let d = comp_in_db - COMP_THRESH_DB + COMP_KNEE_DB * 0.5;
                comp_in_db + (1.0 / COMP_RATIO - 1.0) * (d * d) / (2.0 * COMP_KNEE_DB)
            } else {
                COMP_THRESH_DB + (comp_in_db - COMP_THRESH_DB) / COMP_RATIO
            };
            let comp_gr_db = comp_out_db - comp_in_db;

            // Fade out AGC boost smoothly when gating down during silence.
            let agc_blend = (1.0 - self.gate_atten_db / MAX_GATE_ATTEN_DB).clamp(0.0, 1.0);
            let target_gain_db =
                (self.agc_gain_db * agc_blend + comp_gr_db - floor_atten_db).clamp(-18.0, 8.0);
            let a_gain = if target_gain_db < self.smoothed_gain_db {
                a_gain_dn
            } else {
                a_gain_up
            };
            self.smoothed_gain_db =
                a_gain * self.smoothed_gain_db + (1.0 - a_gain) * target_gain_db;

            *sample *= 10.0_f32.powf((amount * self.smoothed_gain_db) / 20.0);
        }
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

    #[test]
    fn sola_handles_silent_tail_without_nan() {
        let mut sola = Sola::default();
        let silence = vec![0.0_f32; 140];
        let first = sola.process(&silence, 100, 20, 20).unwrap();
        let second = sola.process(&silence, 100, 20, 20).unwrap();
        assert!(first.iter().all(|v| v.is_finite() && *v == 0.0));
        assert!(second.iter().all(|v| v.is_finite() && *v == 0.0));
    }

    #[test]
    fn vocal_dynamics_lifts_quiet_signal_tames_loud_and_holds_during_silence() {
        let mut dyn_proc = VocalDynamics::new(48000);
        // Low-level room noise (-50 dBFS) for 0.8 s should never cause gain climb above unity.
        let noise: Vec<f32> = (0..38400)
            .map(|i| 0.00316 * (i as f32 * 0.37).sin())
            .collect();
        let mut noisy = noise.clone();
        dyn_proc.process(&mut noisy, 1.0);
        assert!(
            rms(&noisy[19200..]) <= rms(&noise[19200..]) * 1.05,
            "AGC climbed on background noise"
        );

        // Quiet singing (-32 dBFS -> amp ~0.0355) for 1.2 s receives gentle upward boost.
        let mut dyn_voice = VocalDynamics::new(48000);
        let mut quiet = sine_wave(220.0, 48000, 57600)
            .into_iter()
            .map(|s| s * 0.0355)
            .collect::<Vec<_>>();
        let quiet_in_rms = rms(&quiet[38400..]);
        dyn_voice.process(&mut quiet, 1.0);
        let quiet_out_rms = rms(&quiet[38400..]);
        assert!(
            quiet_out_rms > quiet_in_rms * 1.4,
            "expected gentle lift on quiet phrase: {quiet_in_rms} -> {quiet_out_rms}"
        );

        // Loud belt transient (-4 dBFS peak -> amp ~0.63) is compressed rather than boosted.
        let mut loud = sine_wave(220.0, 48000, 24000)
            .into_iter()
            .map(|s| s * 0.63)
            .collect::<Vec<_>>();
        let loud_in_rms = rms(&loud[12000..]);
        dyn_voice.process(&mut loud, 1.0);
        let loud_out_rms = rms(&loud[12000..]);
        assert!(
            loud_out_rms < loud_in_rms,
            "expected downward compression on loud phrase: {loud_in_rms} -> {loud_out_rms}"
        );

        // Dynamics scale consistently across 44.1 kHz and 48 kHz.
        let mut d44 = VocalDynamics::new(44100);
        let mut q44 = sine_wave(220.0, 44100, 44100)
            .into_iter()
            .map(|s| s * 0.0355)
            .collect::<Vec<_>>();
        d44.process(&mut q44, 1.0);
        assert!(rms(&q44[22050..]) > 0.035);
    }

    #[test]
    fn vocal_dynamics_reacquires_stationary_floor_and_requires_continuous_gate_hold() {
        // Stationary -45 dBFS background for 3 s must reacquire noise floor and avoid boost.
        let mut loud_room = VocalDynamics::new(48000);
        let amp_45 = 10.0_f32.powf(-45.0 / 20.0) * std::f32::consts::SQRT_2;
        let bg_45: Vec<f32> = (0..48000 * 3)
            .map(|i| amp_45 * (2.0 * std::f32::consts::PI * 190.0 * i as f32 / 48000.0).sin())
            .collect();
        let mut proc_45 = bg_45.clone();
        loud_room.process(&mut proc_45, 0.75);
        assert!(
            rms(&proc_45[48000 * 2..]) < rms(&bg_45[48000 * 2..]) * 0.85,
            "expected stationary -45 dBFS floor to reacquire and attenuate"
        );

        // 1 s digital silence followed by 3 s of -55 dBFS background must recover from -72 dB.
        let mut after_silence = VocalDynamics::new(48000);
        let mut silence_then_bg = vec![0.0_f32; 48000];
        let amp_55 = 10.0_f32.powf(-55.0 / 20.0) * std::f32::consts::SQRT_2;
        let bg_55: Vec<f32> = (0..48000 * 3)
            .map(|i| amp_55 * (2.0 * std::f32::consts::PI * 210.0 * i as f32 / 48000.0).sin())
            .collect();
        silence_then_bg.extend_from_slice(&bg_55);
        after_silence.process(&mut silence_then_bg, 0.75);
        assert!(
            rms(&silence_then_bg[48000 * 3..]) < rms(&bg_55[48000 * 2..]) * 0.85,
            "expected silence-to-background recovery without boosting -55 dBFS noise"
        );

        // Interrupted gate hold: ~80 ms below close, 120 ms in [close, open] band, ~135 ms below
        // close (cumulative > 210 ms, each continuous interval < 160 ms) must NOT close the gate,
        // whereas continuing without interruption past 160 ms must close it.
        let mut gated = VocalDynamics::new(48000);
        let amp_active = 10.0_f32.powf(-35.0 / 20.0) * std::f32::consts::SQRT_2;
        let amp_low = 10.0_f32.powf(-60.0 / 20.0) * std::f32::consts::SQRT_2;
        let amp_mid = 10.0_f32.powf(-49.5 / 20.0) * std::f32::consts::SQRT_2; // safely in [close, open] with sidechain tilt
        let make_tone = |len: usize, amp: f32| -> Vec<f32> {
            (0..len)
                .map(|i| amp * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 48000.0).sin())
                .collect()
        };
        gated.process(&mut make_tone(48000 / 2, amp_active), 1.0);
        // ~360 ms RMS decay from -35 dBFS + ~80 ms strictly below close_thresh (~ -53.5 dBFS).
        gated.process(&mut make_tone(48000 * 440 / 1000, amp_low), 1.0);
        assert!(
            gated.below_close_samples > 0 && !gated.gate_closed,
            "expected active hold counter without closing: below={} closed={} floor={}",
            gated.below_close_samples,
            gated.gate_closed,
            gated.noise_floor_db
        );
        // Entering [close_thresh, open_thresh] resets hold counter without clearing hysteresis.
        gated.process(&mut make_tone(48000 * 120 / 1000, amp_mid), 1.0);
        assert_eq!(gated.below_close_samples, 0);
        assert!(!gated.gate_closed);
        // Another <160 ms strictly below close (~25 ms RMS decay + ~135 ms below close).
        gated.process(&mut make_tone(48000 * 160 / 1000, amp_low), 1.0);
        assert!(
            !gated.gate_closed,
            "interrupted intervals should not close 160 ms continuous hold (below={})",
            gated.below_close_samples
        );
        // Continuing below close crosses 160 ms continuously and closes the gate.
        gated.process(&mut make_tone(48000 * 150 / 1000, amp_low), 1.0);
        assert!(
            gated.gate_closed,
            "continuous interval exceeding 160 ms below close_thresh should close gate"
        );
    }
}
