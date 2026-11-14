//! Vocal separation with the real Kim_Vocal_2 model the voice engine dialog offers.
//! Run with `MDX_ONNX=path/to/kim_vocal_2.onnx cargo test --test separate -- --ignored --nocapture`.
#![cfg(any(feature = "onnx", feature = "onnx-static"))]
use std::path::PathBuf;
use voice_changer::{
    audio_file::Stereo,
    config::Device,
    separate::{separate, Progress, RATE},
};

fn model() -> PathBuf {
    let path = PathBuf::from(std::env::var_os("MDX_ONNX").expect("set MDX_ONNX"));
    assert!(path.is_file(), "{}", path.display());
    path
}

const SECONDS: usize = 8;
const TAU: f32 = std::f32::consts::TAU;

/// A sung "ah": a glottal pulse train with vibrato through vowel formants.
fn voice() -> Vec<f32> {
    let n = RATE as usize * SECONDS;
    let mut phase = 0.0_f32;
    let source: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            let f0 = 220.0 * (1.0 + 0.02 * (TAU * 5.5 * t).sin());
            phase = (phase + f0 / RATE as f32) % 1.0;
            // A sharp pulse per period, like vocal folds closing.
            (-(phase * 40.0)).exp() - 0.025
        })
        .collect();
    // Formants of "ah": 800, 1150 and 2900 Hz, as resonators.
    let mut out = vec![0.0_f32; n];
    for (formant, bandwidth, gain) in [
        (800.0, 80.0, 1.0),
        (1150.0, 90.0, 0.6),
        (2900.0, 120.0, 0.25),
    ] {
        let r = (-std::f32::consts::PI * bandwidth / RATE as f32).exp();
        let c = 2.0 * r * (TAU * formant / RATE as f32).cos();
        let (mut y1, mut y2) = (0.0_f32, 0.0_f32);
        for (i, x) in source.iter().enumerate() {
            let y = x * gain + c * y1 - r * r * y2;
            out[i] += y;
            (y2, y1) = (y1, y);
        }
    }
    let peak = out.iter().fold(0.0_f32, |p, s| p.max(s.abs()));
    out.iter().map(|s| s / peak * 0.3).collect()
}

/// A bass line and a kick drum: no voice in it.
fn band() -> Vec<f32> {
    (0..RATE as usize * SECONDS)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            let note = [55.0, 65.4, 73.4, 49.0][(t * 2.0) as usize % 4];
            let bass = (TAU * note * t).sin() * 0.25 + (TAU * note * 2.0 * t).sin() * 0.08;
            let beat = t % 0.5;
            let kick = (-beat * 30.0).exp()
                * (TAU * (60.0 + 90.0 * (-beat * 40.0).exp()) * beat).sin()
                * 0.4;
            bass + kick
        })
        .collect()
}

fn stereo(samples: Vec<f32>) -> Stereo {
    Stereo {
        rate: RATE,
        channels: [samples.clone(), samples],
    }
}

fn energy(audio: &Stereo) -> f32 {
    audio.channels.iter().flatten().map(|s| s * s).sum::<f32>()
}

#[test]
#[ignore = "requires ONNX Runtime 1.22 (ORT_DYLIB_PATH) and MDX_ONNX"]
fn the_model_hears_a_voice_and_not_a_band() {
    voice_changer::runtime::initialize().unwrap();
    let model = model();
    let share = |song: &Stereo| {
        let progress = Progress::default();
        let started = std::time::Instant::now();
        let stems = separate(&model, Device::Cpu, song, &progress).unwrap();
        assert_eq!(stems.vocals.len(), song.len());
        assert_eq!(stems.instrumental.len(), song.len());
        assert_eq!(progress.fraction(), 1.0);
        let share = energy(&stems.vocals) / energy(song).max(1e-12);
        println!(
            "vocal share {share:.3} ({:.1} s for {SECONDS} s of audio)",
            started.elapsed().as_secs_f32()
        );
        (share, stems)
    };
    let (voice_share, _) = share(&stereo(voice()));
    let (band_share, _) = share(&stereo(band()));
    let mix: Vec<f32> = voice().iter().zip(band()).map(|(v, b)| v + b).collect();
    let (_, stems) = share(&stereo(mix));
    // The vocals of the mix are closer to the voice than to the band.
    let voice_alone = stereo(voice());
    let difference: f32 = stems
        .vocals
        .channels
        .iter()
        .flatten()
        .zip(voice_alone.channels.iter().flatten())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    println!(
        "voice {voice_share:.3}, band {band_share:.3}, mix vocals off the voice by {:.3} of its energy",
        difference / energy(&voice_alone)
    );
    assert!(voice_share > 0.5, "voice share {voice_share}");
    assert!(band_share < 0.1, "band share {band_share}");
    assert!(difference < energy(&voice_alone) * 0.5);
    let silence = separate(
        &model,
        Device::Cpu,
        &stereo(vec![0.0; RATE as usize]),
        &Progress::default(),
    )
    .unwrap();
    assert!(silence.vocals.peak() < 1e-4 && silence.instrumental.peak() < 1e-4);
}
