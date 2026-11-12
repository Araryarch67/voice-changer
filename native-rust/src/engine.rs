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
        self.input_denoise.reset();
        self.output_denoise.reset();
        self.echo = None;
        self.played.clear();
    }
    pub fn new(converter: Option<Box<dyn Converter>>) -> Self {
        Self {
            converter,
            was_bypass: true,
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
            *s = (*s * settings.output_gain).clamp(-1.0, 1.0);
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
