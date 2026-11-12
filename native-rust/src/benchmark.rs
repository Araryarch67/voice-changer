use crate::{
    config::Settings,
    engine::{load_converter_on, Engine},
};
use anyhow::Result;
use serde::Serialize;
use std::{path::Path, time::Instant};

#[derive(Debug, Serialize)]
pub struct Benchmark {
    pub block_ms: u32,
    pub load_ms: f64,
    pub first_block_ms: f64,
    pub median_ms: f64,
    pub max_ms: f64,
    pub realtime_factor: f64,
    pub output_rms: f32,
}
pub fn run(path: &Path, settings: &Settings, blocks: usize) -> Result<Benchmark> {
    settings.validate()?;
    anyhow::ensure!(
        (3..=100).contains(&blocks),
        "Benchmark requires 3..=100 blocks"
    );
    let start = Instant::now();
    let mut engine = Engine::new(Some(load_converter_on(path, settings.device)?));
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let rate = 48000;
    let size = rate as usize * settings.block_ms as usize / 1000;
    let mut times = Vec::new();
    let mut output_rms = 0.0;
    let settings = Settings {
        bypass: false,
        ..settings.clone()
    };
    for block in 0..blocks {
        let audio: Vec<_> = (0..size)
            .map(|i| {
                let t = (block * size + i) as f32 / rate as f32;
                (2.0 * std::f32::consts::PI * 180.0 * t).sin() * 0.2
            })
            .collect();
        let start = Instant::now();
        let output = engine.process(&audio, rate, &settings)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        output_rms = crate::dsp::rms(&output);
    }
    let first_block_ms = times.remove(0);
    times.sort_by(f64::total_cmp);
    let median_ms = times[times.len() / 2];
    Ok(Benchmark {
        block_ms: settings.block_ms,
        load_ms,
        first_block_ms,
        median_ms,
        max_ms: *times.last().unwrap(),
        realtime_factor: median_ms / settings.block_ms as f64,
        output_rms,
    })
}
