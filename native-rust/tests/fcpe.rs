//! The FCPE detector with the real model the voice engine dialog downloads.
//! Run with `FCPE_ONNX=path/to/fcpe.onnx cargo test --test fcpe -- --ignored`.
#![cfg(any(feature = "onnx", feature = "onnx-static"))]
use ort::{session::Session, value::Tensor};
use std::path::{Path, PathBuf};
use voice_changer::{
    config::{F0Detector, Settings},
    engine::{load_converter, Engine},
    pitch::fcpe_decode,
};

fn model() -> PathBuf {
    let path = PathBuf::from(std::env::var_os("FCPE_ONNX").expect("set FCPE_ONNX"));
    assert!(path.is_file(), "{}", path.display());
    path
}

/// A tone with eight harmonics, closer to a voice than a sine: FCPE is unsure of a
/// bare sine below about 120 Hz (peak salience 0.02-0.05), as a voice has harmonics.
fn sine(hz: f32, rate: f32, samples: usize) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * hz * i as f32 / rate;
            (1..=8)
                .map(|k| (phase * k as f32).sin() / k as f32)
                .sum::<f32>()
                * 0.15
        })
        .collect()
}

fn detect(session: &mut Session, audio: &[f32]) -> Vec<f32> {
    let input = Tensor::from_array(([1, audio.len()], audio.to_vec().into_boxed_slice())).unwrap();
    let output = session.run(ort::inputs! { "audio" => input }).unwrap();
    let (shape, salience) = output[0].try_extract_tensor::<f32>().unwrap();
    assert_eq!(shape[shape.len() - 1], 360);
    fcpe_decode(salience)
}

#[test]
#[ignore = "requires ONNX Runtime 1.22 (ORT_DYLIB_PATH) and FCPE_ONNX"]
fn fcpe_follows_a_tone_and_is_silent_on_silence() {
    voice_changer::runtime::initialize().unwrap();
    let mut session = Session::builder()
        .unwrap()
        .commit_from_file(model())
        .unwrap();
    for hz in [110.0, 220.0, 440.0] {
        let f0 = detect(&mut session, &sine(hz, 16000.0, 16000));
        assert_eq!(f0.len(), 16000 / 160 + 1);
        let mut voiced: Vec<f32> = f0[5..f0.len() - 5].to_vec();
        voiced.sort_by(f32::total_cmp);
        let median = voiced[voiced.len() / 2];
        assert!((median / hz - 1.0).abs() < 0.01, "{hz} Hz read as {median}");
    }
    assert!(detect(&mut session, &vec![0.0; 16000])
        .iter()
        .all(|f| *f == 0.0));
}

#[test]
#[ignore = "requires ONNX Runtime 1.22 (ORT_DYLIB_PATH) and FCPE_ONNX"]
fn the_pipeline_runs_with_fcpe_beside_the_rmvpe_model() {
    // The synthetic test voice, with fcpe.onnx beside its pitch model.
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let folder = std::env::temp_dir().join(format!("vc-fcpe-{}", std::process::id()));
    std::fs::create_dir_all(&folder).unwrap();
    for file in ["model.json", "voice.onnx", "embedder.onnx", "pitch.onnx"] {
        std::fs::copy(fixtures.join(file), folder.join(file)).unwrap();
    }
    std::fs::copy(model(), folder.join("fcpe.onnx")).unwrap();
    let mut engine = Engine::new(Some(load_converter(&folder.join("model.json")).unwrap()));
    let settings = Settings {
        bypass: false,
        f0_detector: F0Detector::Fcpe,
        ..Settings::default()
    };
    let input = sine(220.0, 48000.0, 15360);
    for _ in 0..3 {
        let output = engine.process(&input, 48000, &settings).unwrap();
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|v| v.is_finite()));
    }
    let _ = std::fs::remove_dir_all(&folder);
}
