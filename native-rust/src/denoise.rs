//! RNNoise noise suppression (nnnoiseless) for arbitrary block sizes and sample rates.
use crate::dsp;
use nnnoiseless::DenoiseState;
use std::collections::VecDeque;

const FRAME: usize = DenoiseState::FRAME_SIZE;
/// Frames the voice gate stays open after speech was last detected (300 ms).
const HOLD_FRAMES: u32 = 30;

pub struct Denoiser {
    state: Box<DenoiseState<'static>>,
    pending: Vec<f32>,
    ready: VecDeque<f32>,
    gain: f32,
    hold: u32,
    fresh: bool,
}
impl Default for Denoiser {
    fn default() -> Self {
        Self {
            state: DenoiseState::new(),
            pending: Vec::new(),
            // One frame of lead so every call can return as many samples as it was given.
            ready: std::iter::repeat_n(0.0, FRAME).collect(),
            gain: 1.0,
            hold: HOLD_FRAMES,
            fresh: true,
        }
    }
}
impl Denoiser {
    pub fn reset(&mut self) {
        if !self.fresh {
            *self = Self::default();
        }
    }
    /// Returns `audio.len()` cleaned samples, delayed by one 10 ms frame. With `gate`,
    /// frames RNNoise classifies as non-speech are faded out entirely.
    pub fn process(&mut self, audio: &[f32], rate: u32, gate: bool) -> Vec<f32> {
        self.fresh = false;
        let wide = dsp::resample(audio, rate, 48000);
        // RNNoise works on 48 kHz frames in 16-bit sample scale.
        self.pending.extend(wide.iter().map(|s| s * 32768.0));
        let mut clean = [0.0; FRAME];
        let mut consumed = 0;
        for frame in self.pending.as_chunks::<FRAME>().0 {
            let voice = self.state.process_frame(&mut clean, frame);
            if voice >= 0.5 {
                self.hold = HOLD_FRAMES;
            } else {
                self.hold = self.hold.saturating_sub(1);
            }
            let target = if gate && self.hold == 0 { 0.0 } else { 1.0 };
            let step = (target - self.gain) / FRAME as f32;
            for sample in clean {
                self.gain += step;
                self.ready.push_back(sample / 32768.0 * self.gain);
            }
            self.gain = target;
            consumed += FRAME;
        }
        self.pending.drain(..consumed);
        let available = wide.len().min(self.ready.len());
        let mut output: Vec<f32> = self.ready.drain(..available).collect();
        output.resize(wide.len(), 0.0);
        let mut output = dsp::resample(&output, 48000, rate);
        output.resize(audio.len(), 0.0);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white_noise(samples: usize) -> Vec<f32> {
        let mut seed = 0x2545_f491_u32;
        (0..samples)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) as f32 / 8_388_608.0 - 1.0) * 0.05
            })
            .collect()
    }

    #[test]
    fn keeps_block_length_at_any_rate() {
        let mut denoiser = Denoiser::default();
        for (rate, len) in [(48000, 15360), (44100, 14112), (16000, 333), (48000, 7)] {
            assert_eq!(denoiser.process(&vec![0.1; len], rate, false).len(), len);
        }
    }

    #[test]
    fn suppresses_steady_noise() {
        let noise = white_noise(48000 * 2);
        let mut denoiser = Denoiser::default();
        let mut output = Vec::new();
        for block in noise.chunks(15360) {
            output.extend(denoiser.process(block, 48000, false));
        }
        // Compare the second half, after RNNoise has adapted.
        let half = noise.len() / 2;
        assert!(dsp::rms(&output[half..]) < dsp::rms(&noise[half..]) * 0.5);
    }

    #[test]
    fn gate_mutes_non_speech() {
        let noise = white_noise(48000 * 2);
        let mut denoiser = Denoiser::default();
        let mut output = Vec::new();
        for block in noise.chunks(15360) {
            output.extend(denoiser.process(block, 48000, true));
        }
        assert!(dsp::rms(&output[noise.len() * 3 / 4..]) < 1e-4);
    }
}
