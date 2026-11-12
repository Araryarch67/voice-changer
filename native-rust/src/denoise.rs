//! RNNoise noise suppression (nnnoiseless) for arbitrary block sizes and sample rates.
use crate::dsp;
use nnnoiseless::DenoiseState;
use std::collections::VecDeque;

const FRAME: usize = DenoiseState::FRAME_SIZE;
/// RNNoise voice probability that opens the gate.
const VOICE: f32 = 0.5;
/// Frames the gate looks ahead, so it is already open when a word starts (30 ms).
const LOOKAHEAD: usize = 3;
/// Frames the gate stays open after speech was last detected (500 ms).
const HOLD_FRAMES: u32 = 50;
/// Frames the gate takes to close (100 ms); it opens within one frame.
const RELEASE_FRAMES: f32 = 10.0;

pub struct Denoiser {
    state: Box<DenoiseState<'static>>,
    gate: bool,
    pending: Vec<f32>,
    /// Cleaned frames with their voice probability, waiting out the gate's lookahead.
    held: VecDeque<([f32; FRAME], f32)>,
    ready: VecDeque<f32>,
    gain: f32,
    hold: u32,
    fresh: bool,
}
impl Default for Denoiser {
    fn default() -> Self {
        Self::new(false)
    }
}
impl Denoiser {
    fn new(gate: bool) -> Self {
        let lookahead = if gate { LOOKAHEAD } else { 0 };
        Self {
            state: DenoiseState::new(),
            gate,
            pending: Vec::new(),
            held: VecDeque::new(),
            // Enough lead that every call can return as many samples as it was given.
            ready: std::iter::repeat_n(0.0, FRAME * (1 + lookahead)).collect(),
            gain: 1.0,
            hold: HOLD_FRAMES,
            fresh: true,
        }
    }
    pub fn reset(&mut self) {
        if !self.fresh {
            *self = Self::new(self.gate);
        }
    }
    /// Returns `audio.len()` cleaned samples, delayed by 10 ms. With `gate`, stretches
    /// RNNoise classifies as non-speech are also faded out, at 30 ms more delay.
    pub fn process(&mut self, audio: &[f32], rate: u32, gate: bool) -> Vec<f32> {
        if gate != self.gate {
            *self = Self::new(gate);
        }
        self.fresh = false;
        let lookahead = if gate { LOOKAHEAD } else { 0 };
        let wide = dsp::resample(audio, rate, 48000);
        // RNNoise works on 48 kHz frames in 16-bit sample scale.
        self.pending.extend(wide.iter().map(|s| s * 32768.0));
        let mut consumed = 0;
        for frame in self.pending.as_chunks::<FRAME>().0 {
            let mut clean = [0.0; FRAME];
            let voice = self.state.process_frame(&mut clean, frame);
            self.held.push_back((clean, voice));
            consumed += FRAME;
            if self.held.len() <= lookahead {
                continue;
            }
            let speaking = self.held.iter().any(|(_, voice)| *voice >= VOICE);
            let (clean, _) = self.held.pop_front().expect("longer than the lookahead");
            if speaking {
                self.hold = HOLD_FRAMES;
            } else {
                self.hold = self.hold.saturating_sub(1);
            }
            let target = if !gate || self.hold > 0 {
                1.0
            } else {
                (self.gain - 1.0 / RELEASE_FRAMES).max(0.0)
            };
            let step = (target - self.gain) / FRAME as f32;
            for sample in clean {
                self.gain += step;
                self.ready.push_back(sample / 32768.0 * self.gain);
            }
            self.gain = target;
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
