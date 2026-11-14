use crate::{
    config::{Device, Noise, Settings},
    denoise::Denoiser,
    dsp,
    echo::EchoCanceller,
};
use anyhow::{bail, Result};

pub trait Converter: Send {
    fn process(&mut self, audio: &[f32], sample_rate: u32, settings: &Settings)
        -> Result<Vec<f32>>;
    fn reset(&mut self);
}

pub struct Engine {
    converter: Option<Box<dyn Converter>>,
    was_bypass: bool,
    high_pass: Option<(u32, dsp::HighPass)>,
    vocal_dynamics: Option<dsp::VocalDynamics>,
    input_denoise: Denoiser,
    output_denoise: Denoiser,
    echo: Option<EchoCanceller>,
    /// The previous block as it was sent to the output: the echo canceller's reference.
    played: Vec<f32>,
}
impl Engine {
    pub fn reset(&mut self) {
        if let Some(converter) = &mut self.converter {
            converter.reset();
        }
        if let Some((_, hp)) = &mut self.high_pass {
            hp.reset();
        }
        if let Some(dyn_proc) = &mut self.vocal_dynamics {
            dyn_proc.reset();
        }
        self.input_denoise.reset();
        self.output_denoise.reset();
        self.echo = None;
        self.played.clear();
    }
    /// Replaces the voice; the next block starts it afresh.
    pub fn set_converter(&mut self, converter: Box<dyn Converter>) {
        self.converter = Some(converter);
        self.reset();
    }
    pub fn new(converter: Option<Box<dyn Converter>>) -> Self {
        Self {
            converter,
            was_bypass: true,
            high_pass: None,
            vocal_dynamics: None,
            input_denoise: Denoiser::default(),
            output_denoise: Denoiser::default(),
            echo: None,
            played: Vec::new(),
        }
    }
    pub fn process(&mut self, audio: &[f32], rate: u32, settings: &Settings) -> Result<Vec<f32>> {
        settings.validate()?;
        anyhow::ensure!(rate > 0, "Sample rate must be positive");
        // Reset stateful conversion stages on transition *before* filtering block 1,
        // otherwise block 1 runs through the HPF and immediately gets wiped for block 2.
        if !settings.bypass && self.was_bypass {
            if let Some(converter) = &mut self.converter {
                converter.reset();
            }
            if let Some((_, hp)) = &mut self.high_pass {
                hp.reset();
            }
            if let Some(dyn_proc) = &mut self.vocal_dynamics {
                dyn_proc.reset();
            }
        } else if settings.bypass && !self.was_bypass {
            if let Some(converter) = &mut self.converter {
                converter.reset();
            }
        }
        let mut input: Vec<_> = audio.iter().map(|s| s * settings.input_gain).collect();
        if settings.echo {
            if self.echo.is_none() {
                self.echo = Some(EchoCanceller::new(rate)?);
            }
            let canceller = self.echo.as_mut().expect("created above");
            input = canceller.process(&input, &self.played, rate)?;
        } else {
            self.echo = None;
            self.played.clear();
        }
        if !settings.bypass {
            let hp = match &mut self.high_pass {
                Some((hp_rate, hp)) if *hp_rate == rate => hp,
                slot => {
                    *slot = Some((rate, dsp::HighPass::default_voice(rate)));
                    &mut slot.as_mut().expect("initialized above").1
                }
            };
            hp.process(&mut input);
        } else if let Some((_, hp)) = &mut self.high_pass {
            hp.reset();
        }
        match settings.noise {
            Noise::Off => self.input_denoise.reset(),
            mode => {
                input = self
                    .input_denoise
                    .process(&input, rate, mode == Noise::Sup2)
            }
        }
        if !settings.bypass && settings.vocal_leveling > 0.0 {
            let dyn_proc = match &mut self.vocal_dynamics {
                Some(d) if d.rate() == rate => d,
                slot => slot.insert(dsp::VocalDynamics::new(rate)),
            };
            dyn_proc.process(&mut input, settings.vocal_leveling);
        } else if let Some(dyn_proc) = &mut self.vocal_dynamics {
            dyn_proc.reset();
        }
        let mut out = if settings.bypass {
            input
        } else {
            let Some(converter) = &mut self.converter else {
                bail!("Load a model or enable bypass before starting");
            };
            converter.process(&input, rate, settings)?
        };
        if settings.noise == Noise::Sup2 && !settings.bypass {
            out = self.output_denoise.process(&out, rate, false);
        } else {
            self.output_denoise.reset();
        }
        self.was_bypass = settings.bypass;
        anyhow::ensure!(out.len() == audio.len(), "Converter changed block duration");
        anyhow::ensure!(
            out.iter().all(|s| s.is_finite()),
            "Model produced non-finite audio"
        );
        for s in &mut out {
            let gained = *s * settings.output_gain;
            *s = if settings.bypass {
                gained.clamp(-1.0, 1.0)
            } else {
                soft_limit(gained)
            };
        }
        if settings.echo {
            self.played.clone_from(&out);
        }
        Ok(out)
    }
    pub fn input_level(audio: &[f32]) -> f32 {
        dsp::rms(audio)
    }

    /// Runs one non-silent synthetic block with gate bypassed so lazy ONNX Runtime kernels
    /// initialize before live capture starts, then resets all stateful DSP history.
    pub fn warmup(&mut self, rate: u32, settings: &Settings) -> Result<()> {
        settings.validate()?;
        anyhow::ensure!(rate > 0, "Sample rate must be positive");
        if self.converter.is_none() {
            return Ok(());
        }
        let mut warmup_settings = settings.clone();
        warmup_settings.bypass = false;
        warmup_settings.echo = false;
        warmup_settings.silence_threshold = 0.0;
        warmup_settings.input_gain = 1.0;
        let samples = (rate as usize * settings.block_ms as usize / 1000).max(1);
        let synth: Vec<f32> = (0..samples)
            .map(|i| 0.1 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / rate as f32).sin())
            .collect();
        let _ = self.process(&synth, rate, &warmup_settings)?;
        self.reset();
        self.was_bypass = true;
        Ok(())
    }
}

/// Transparent soft-knee limiter above 0.9 to prevent harsh digital clipping on loud transients.
fn soft_limit(sample: f32) -> f32 {
    const KNEE: f32 = 0.9;
    let abs = sample.abs();
    if abs <= KNEE {
        sample
    } else {
        let head = 1.0 - KNEE;
        let shaped = KNEE + head * ((abs - KNEE) / head).tanh();
        (sample.signum() * shaped).clamp(-1.0, 1.0)
    }
}

pub fn load_converter(path: &std::path::Path) -> Result<Box<dyn Converter>> {
    load_converter_on(path, Device::Cpu)
}
pub fn load_converter_on(path: &std::path::Path, device: Device) -> Result<Box<dyn Converter>> {
    #[cfg(any(feature = "onnx", feature = "onnx-static"))]
    {
        Ok(Box::new(crate::rvc::Rvc::load_on(path, device)?))
    }
    #[cfg(not(any(feature = "onnx", feature = "onnx-static")))]
    {
        let _ = (path, device);
        bail!("This build does not include the onnx feature")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PassthroughConverter;
    impl Converter for PassthroughConverter {
        fn process(&mut self, audio: &[f32], _: u32, _: &Settings) -> Result<Vec<f32>> {
            Ok(audio.to_vec())
        }
        fn reset(&mut self) {}
    }

    #[test]
    fn the_longest_chunk_is_accepted_and_one_past_it_is_not() {
        let mut engine = Engine::new(Some(Box::new(PassthroughConverter)));
        let longest = Settings {
            bypass: false,
            block_ms: crate::config::MAX_BLOCK_MS,
            ..Settings::default()
        };
        let block = vec![0.1; 48 * longest.block_ms as usize];
        let out = engine.process(&block, 48000, &longest).unwrap();
        assert_eq!(out.len(), block.len());
        let too_long = Settings {
            block_ms: longest.block_ms + 1,
            ..longest
        };
        assert!(engine.process(&block, 48000, &too_long).is_err());
    }

    #[test]
    fn active_path_applies_high_pass_and_soft_limits_peaks() {
        let mut engine = Engine::new(Some(Box::new(PassthroughConverter)));
        let settings = Settings {
            bypass: false,
            output_gain: 1.5,
            ..Settings::default()
        };

        // Pure DC input should settle near zero after the 80 Hz high-pass filter,
        // and the second consecutive block must not suffer a late HPF reset transient.
        let dc = vec![0.4; 4800];
        let filtered1 = engine.process(&dc, 48000, &settings).unwrap();
        assert!(dsp::rms(&filtered1[2400..]) < 1e-3);
        let filtered2 = engine.process(&dc, 48000, &settings).unwrap();
        assert!(
            dsp::rms(&filtered2) < 1e-3,
            "second DC block saw unexpected HPF transient: {}",
            dsp::rms(&filtered2)
        );

        // Normal speech band (220 Hz) passes through and hot peaks are soft-limited inside [-1, 1].
        engine.reset();
        let speech: Vec<f32> = (0..4800)
            .map(|i| 0.8 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 48000.0).sin())
            .collect();
        let out = engine.process(&speech, 48000, &settings).unwrap();
        assert!(out
            .iter()
            .all(|s| s.is_finite() && (-1.0..=1.0).contains(s)));
        assert!(dsp::rms(&out[2400..]) > 0.5);

        // Soft limiter is linear within [-0.9, 0.9] and smooth above 0.9.
        assert_eq!(soft_limit(0.5), 0.5);
        assert_eq!(soft_limit(-0.9), -0.9);
        assert!(soft_limit(1.0) < 1.0 && soft_limit(1.0) > 0.95);
        assert!(soft_limit(-1.0) > -1.0 && soft_limit(-1.0) < -0.95);
        assert!(soft_limit(2.0) <= 1.0 && soft_limit(2.0) > 0.99);
        assert!(soft_limit(-2.0) >= -1.0 && soft_limit(-2.0) < -0.99);
    }

    #[test]
    fn warmup_runs_non_zero_audio_and_resets_state() {
        use std::sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc,
        };
        struct SpyConverter {
            saw_nonzero: Arc<AtomicBool>,
            resets: Arc<AtomicUsize>,
        }
        impl Converter for SpyConverter {
            fn process(&mut self, audio: &[f32], _: u32, _: &Settings) -> Result<Vec<f32>> {
                if dsp::rms(audio) > 0.01 {
                    self.saw_nonzero.store(true, Ordering::Relaxed);
                }
                Ok(audio.to_vec())
            }
            fn reset(&mut self) {
                self.resets.fetch_add(1, Ordering::Relaxed);
            }
        }
        let saw_nonzero = Arc::new(AtomicBool::new(false));
        let resets = Arc::new(AtomicUsize::new(0));
        let mut engine = Engine::new(Some(Box::new(SpyConverter {
            saw_nonzero: saw_nonzero.clone(),
            resets: resets.clone(),
        })));
        let settings = Settings {
            bypass: true,
            silence_threshold: 0.9,
            block_ms: 40,
            ..Settings::default()
        };
        engine.warmup(48000, &settings).unwrap();
        assert!(saw_nonzero.load(Ordering::Relaxed));
        assert_eq!(resets.load(Ordering::Relaxed), 2);
        assert!(engine.was_bypass);
    }
}
