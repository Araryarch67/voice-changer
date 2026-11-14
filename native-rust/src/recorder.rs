//! Records the microphone for a voice cloning dataset: mono samples at the device's
//! own rate, with the current level for a meter. Nothing is converted while
//! recording; [`crate::dataset::save_take`] trims and normalizes the take.
use anyhow::{bail, Result};
use cpal::{
    traits::{DeviceTrait, StreamTrait},
    FromSample, SampleFormat, SizedSample,
};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc, Mutex,
};

/// Longest take kept (ten minutes); recording stops adding after it.
const MAX_SECONDS: usize = 600;

struct Shared {
    samples: Mutex<Vec<f32>>,
    /// Peak of the last callback, as `f32` bits.
    level: AtomicU32,
}

pub struct Recorder {
    stream: cpal::Stream,
    shared: Arc<Shared>,
    rate: u32,
}

impl Recorder {
    /// Starts recording `device` (an input device id), or the default microphone.
    pub fn start(device: &Option<String>) -> Result<Self> {
        let host = cpal::default_host();
        let device = crate::audio::select_device(&host, device, true)?;
        let config = device.default_input_config()?;
        let rate = config.sample_rate();
        log::info!(
            "Recording {} ({} Hz, {} ch)",
            device,
            rate,
            config.channels()
        );
        let shared = Arc::new(Shared {
            samples: Mutex::new(Vec::with_capacity(rate as usize * 60)),
            level: AtomicU32::new(0),
        });
        let limit = rate as usize * MAX_SECONDS;
        let format = config.sample_format();
        let config: cpal::StreamConfig = config.into();
        let stream = match format {
            SampleFormat::F32 => build::<f32>(&device, &config, shared.clone(), limit)?,
            SampleFormat::I16 => build::<i16>(&device, &config, shared.clone(), limit)?,
            SampleFormat::I32 => build::<i32>(&device, &config, shared.clone(), limit)?,
            SampleFormat::U16 => build::<u16>(&device, &config, shared.clone(), limit)?,
            format => bail!("Unsupported input sample format: {format:?}"),
        };
        stream.play()?;
        Ok(Self {
            stream,
            shared,
            rate,
        })
    }

    /// The input's recent peak, 0 to 1.
    pub fn level(&self) -> f32 {
        f32::from_bits(self.shared.level.load(Ordering::Relaxed))
    }

    /// Seconds recorded so far.
    pub fn seconds(&self) -> f64 {
        self.shared.samples.lock().unwrap().len() as f64 / self.rate as f64
    }

    /// Stops and returns the sample rate and the recording.
    pub fn finish(self) -> (u32, Vec<f32>) {
        let _ = self.stream.pause();
        drop(self.stream);
        let samples = std::mem::take(&mut *self.shared.samples.lock().unwrap());
        (self.rate, samples)
    }
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: Arc<Shared>,
    limit: usize,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels.max(1) as usize;
    Ok(device.build_input_stream(
        *config,
        move |data: &[T], _| {
            let mut peak = 0.0f32;
            // A short lock; the GUI only takes it to read the length.
            if let Ok(mut samples) = shared.samples.lock() {
                for frame in data.chunks_exact(channels) {
                    let sample =
                        frame.iter().map(|s| s.to_sample::<f32>()).sum::<f32>() / channels as f32;
                    let sample = if sample.is_finite() { sample } else { 0.0 };
                    peak = peak.max(sample.abs());
                    if samples.len() < limit {
                        samples.push(sample);
                    }
                }
            }
            shared.level.store(peak.to_bits(), Ordering::Relaxed);
        },
        |error| log::warn!("Recording: {error}"),
        None,
    )?)
}
