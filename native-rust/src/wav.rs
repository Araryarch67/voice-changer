use crate::{config::Settings, engine::Engine};
use anyhow::{Context, Result};
use std::path::Path;

/// Reads a WAV file into mono `f32` samples normalized to `[-1.0, 1.0]` along with its sample rate.
pub fn read_mono(path: &Path) -> Result<(u32, Vec<f32>)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("Open input WAV {}", path.display()))?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.channels > 0 && spec.sample_rate > 0,
        "Invalid WAV format"
    );
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2.0_f32.powi(spec.bits_per_sample as i32 - 1);
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono: Vec<f32> = samples
        .chunks_exact(spec.channels as usize)
        .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
        .collect();
    Ok((spec.sample_rate, mono))
}

/// Converts mono `samples` at `sample_rate` block by block, as live audio is;
/// `on_block` hears how many blocks are done out of how many.
pub fn convert_samples(
    engine: &mut Engine,
    samples: &[f32],
    sample_rate: u32,
    settings: &Settings,
    on_block: &mut dyn FnMut(usize, usize),
) -> Result<Vec<f32>> {
    settings.validate()?;
    let block = (sample_rate as usize * settings.block_ms as usize / 1000).max(1);
    let blocks = samples.len().div_ceil(block);
    let mut converted = Vec::with_capacity(samples.len());
    for (index, chunk) in samples.chunks(block).enumerate() {
        let mut padded = chunk.to_vec();
        padded.resize(block, 0.0);
        let audio = engine.process(&padded, sample_rate, settings)?;
        converted.extend_from_slice(&audio[..chunk.len()]);
        on_block(index + 1, blocks);
    }
    Ok(converted)
}

pub fn convert(
    engine: &mut Engine,
    input: &Path,
    output: &Path,
    settings: &Settings,
) -> Result<()> {
    settings.validate()?;
    anyhow::ensure!(input != output, "Input and output must be different files");
    if output.exists() {
        anyhow::ensure!(
            std::fs::canonicalize(input)? != std::fs::canonicalize(output)?,
            "Input and output refer to the same file"
        );
    }
    let (sample_rate, mono) = read_mono(input)?;
    // Process before creating output so a model failure cannot leave a partial result.
    let converted = convert_samples(engine, &mono, sample_rate, settings, &mut |_, _| {})?;
    let mut writer = hound::WavWriter::create(
        output,
        hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for sample in converted {
        writer.write_sample((sample * 32767.0).round() as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
