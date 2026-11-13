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
            input_denoise: Denoiser::default(),
            output_denoise: Denoiser::default(),
            echo: None,
            played: Vec::new(),
        }
    }
    pub fn process(&mut self, audio: &[f32], rate: u32, settings: &Settings) -> Result<Vec<f32>> {
        settings.validate()?;
        anyhow::ensure!(rate > 0, "Sample rate must be positive");
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
        let mut out = if settings.bypass {
            if !self.was_bypass {
                if let Some(converter) = &mut self.converter {
                    converter.reset();
                }
            }
            input
        } else {
            if self.was_bypass {
                if let Some(converter) = &mut self.converter {
                    converter.reset();
                }
                if let Some((_, hp)) = &mut self.high_pass {
                    hp.reset();
                }
            }
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
    fn active_path_applies_high_pass_and_soft_limits_peaks() {
        let mut engine = Engine::new(Some(Box::new(PassthroughConverter)));
        let settings = Settings {
            bypass: false,
            output_gain: 1.5,
            ..Settings::default()
        };

        // Pure DC input should settle near zero after the 80 Hz high-pass filter.
        let dc = vec![0.4; 4800];
        let filtered = engine.process(&dc, 48000, &settings).unwrap();
        assert!(dsp::rms(&filtered[2400..]) < 1e-3);

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
}
