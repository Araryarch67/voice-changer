//! Acoustic echo cancellation: removes the converted voice the microphone picks up
//! from the speakers, using the Rust port of WebRTC's AEC3.
use crate::dsp;
use aec3::{
    nodes::audio::AudioFormat,
    pipelines::linear::{self, LinearPipeline},
};
use anyhow::Result;
use std::collections::VecDeque;

pub struct EchoCanceller {
    pipeline: LinearPipeline,
    /// Rate AEC3 runs at; 48 kHz unless the stream is already at a rate it accepts.
    rate: u32,
    frame: usize,
    capture: Vec<f32>,
    render: Vec<f32>,
    ready: VecDeque<f32>,
}
impl EchoCanceller {
    pub fn new(stream_rate: u32) -> Result<Self> {
        let rate = if [16000, 32000, 48000].contains(&stream_rate) {
            stream_rate
        } else {
            48000
        };
        let format = AudioFormat::ten_ms(rate, 1);
        // Only the canceller: noise suppression and gain are separate settings here.
        let pipeline = linear::builder(format, format)
            .enable_noise_suppression(false)
            .enable_gain_controller2(false)
            .enable_post_filter(false)
            .build()
            .map_err(|error| anyhow::anyhow!("Create echo canceller: {error}"))?;
        let frame = format.sample_count();
        Ok(Self {
            pipeline,
            rate,
            frame,
            capture: Vec::new(),
            render: Vec::new(),
            // One frame of lead so every call can return as many samples as it was given.
            ready: std::iter::repeat_n(0.0, frame).collect(),
        })
    }
    /// `capture` is a microphone block and `render` the audio sent to the output for
    /// the block before it; AEC3 estimates the remaining delay itself. Returns
    /// `capture.len()` samples, delayed by one 10 ms frame.
    pub fn process(
        &mut self,
        capture: &[f32],
        render: &[f32],
        stream_rate: u32,
    ) -> Result<Vec<f32>> {
        let wide = dsp::resample(capture, stream_rate, self.rate);
        let mut played = dsp::resample(render, stream_rate, self.rate);
        played.resize(wide.len(), 0.0);
        self.capture.extend_from_slice(&wide);
        self.render.extend_from_slice(&played);
        let mut clean = vec![0.0; self.frame];
        let mut consumed = 0;
        while self.capture.len() - consumed >= self.frame {
            let span = consumed..consumed + self.frame;
            self.pipeline
                .handle_render_frame(&self.render[span.clone()])
                .map_err(|error| anyhow::anyhow!("Echo canceller render: {error}"))?;
            let produced = self
                .pipeline
                .process_capture_frame(&self.capture[span], &mut clean)
                .map_err(|error| anyhow::anyhow!("Echo canceller capture: {error}"))?;
            if produced {
                self.ready.extend(clean.iter().copied());
            } else {
                self.ready.extend(std::iter::repeat_n(0.0, self.frame));
            }
            consumed += self.frame;
        }
        self.capture.drain(..consumed);
        self.render.drain(..consumed);
        let available = wide.len().min(self.ready.len());
        let mut output: Vec<f32> = self.ready.drain(..available).collect();
        output.resize(wide.len(), 0.0);
        let mut output = dsp::resample(&output, self.rate, stream_rate);
        output.resize(capture.len(), 0.0);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Speech-like test signal: noise bursts with pauses.
    fn bursts(samples: usize, rate: usize) -> Vec<f32> {
        let mut seed = 0x9e37_79b9_u32;
        (0..samples)
            .map(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (seed >> 8) as f32 / 8_388_608.0 - 1.0;
                let envelope = if (i / (rate / 4)) % 3 == 2 { 0.0 } else { 0.3 };
                noise * envelope
            })
            .collect()
    }

    #[test]
    fn removes_a_delayed_echo_of_the_output() {
        let rate = 48000;
        let far = bursts(rate * 8, rate);
        // The microphone hears only the speakers: 120 ms later and quieter.
        let delay = rate * 120 / 1000;
        let mic: Vec<f32> = (0..far.len())
            .map(|i| {
                if i >= delay {
                    far[i - delay] * 0.5
                } else {
                    0.0
                }
            })
            .collect();
        let mut canceller = EchoCanceller::new(rate as u32).unwrap();
        let mut output = Vec::new();
        for (capture, render) in mic.chunks(15360).zip(far.chunks(15360)) {
            output.extend(canceller.process(capture, render, rate as u32).unwrap());
        }
        assert_eq!(output.len(), mic.len());
        // After convergence the echo is at least 10 dB down.
        let tail = mic.len() * 3 / 4;
        let (before, after) = (dsp::rms(&mic[tail..]), dsp::rms(&output[tail..]));
        assert!(after < before * 0.316, "echo {before} -> {after}");
    }

    #[test]
    fn keeps_block_length_at_other_rates() {
        let mut canceller = EchoCanceller::new(44100).unwrap();
        for len in [14112, 333, 7] {
            let block = vec![0.05; len];
            assert_eq!(canceller.process(&block, &block, 44100).unwrap().len(), len);
        }
    }
}
