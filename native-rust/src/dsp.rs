use std::collections::VecDeque;

pub fn rms(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt()
}

/// Windowed-sinc resampling for contextual inference windows (not the audio callback).
pub fn resample(audio: &[f32], from: u32, to: u32) -> Vec<f32> {
    assert!(from > 0 && to > 0);
    if from == to {
        return audio.to_vec();
    }
    let len = (audio.len() as u64 * to as u64 / from as u64) as usize;
    let cutoff = (to as f64 / from as f64).min(1.0) * 0.95;
    (0..len)
        .map(|i| {
            let position = i as f64 * from as f64 / to as f64;
            let center = position.floor() as isize;
            let mut sum = 0.0;
            let mut weights = 0.0;
            for j in (center - 16)..=(center + 16) {
                if j < 0 || j >= audio.len() as isize {
                    continue;
                }
                let distance = position - j as f64;
                if distance.abs() > 16.0 {
                    continue;
                }
                let x = std::f64::consts::PI * distance * cutoff;
                let sinc = if x.abs() < 1e-8 { 1.0 } else { x.sin() / x };
                let window = 0.5 + 0.5 * (std::f64::consts::PI * distance / 16.0).cos();
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

/// Continuous rate converter for independent input/output device clocks.
pub struct StreamResampler {
    buffer: VecDeque<f32>,
    phase: f64,
    step: f64,
}
impl StreamResampler {
    pub fn new(from: u32, to: u32) -> Self {
        assert!(from > 0 && to > 0);
        Self {
            buffer: VecDeque::new(),
            phase: 0.0,
            step: from as f64 / to as f64,
        }
    }
    pub fn push(&mut self, samples: &[f32]) -> Vec<f32> {
        self.buffer.extend(samples);
        let mut out = Vec::new();
        while self.phase + 1.0 < self.buffer.len() as f64 {
            let i = self.phase.floor() as usize;
            let mix = (self.phase - i as f64) as f32;
            out.push(self.buffer[i] * (1.0 - mix) + self.buffer[i + 1] * mix);
            self.phase += self.step;
        }
        let consumed = (self.phase.floor() as usize).min(self.buffer.len());
        self.buffer.drain(..consumed);
        self.phase -= consumed as f64;
        out
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
    fn resampler_preserves_dc_and_rejects_aliasing() {
        let dc = resample(&vec![0.25; 4800], 48000, 16000);
        assert_eq!(dc.len(), 1600);
        assert!(dc.iter().all(|v| (v - 0.25).abs() < 1e-5));
        let input: Vec<_> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 14000.0 * i as f32 / 48000.0).sin())
            .collect();
        let output = resample(&input, 48000, 16000);
        assert!(rms(&output[32..output.len() - 32]) < 0.02);
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
