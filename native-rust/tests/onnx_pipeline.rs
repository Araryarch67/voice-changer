#![cfg(any(feature = "onnx", feature = "onnx-static"))]
use std::path::Path;
use voice_changer_native::{
    config::Settings,
    engine::{load_converter, Engine},
};

fn engine() -> Engine {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model.json");
    Engine::new(Some(load_converter(&manifest).unwrap()))
}

#[test]
#[ignore = "requires ONNX Runtime 1.22; set ORT_DYLIB_PATH"]
fn onnx_pipeline_preserves_duration_and_pitch_controls_reach_model() {
    let mut engine = engine();
    let mut settings = Settings {
        bypass: false,
        ..Settings::default()
    };
    let input = vec![0.2; 7680];
    let first = engine.process(&input, 48000, &settings).unwrap();
    assert_eq!(first.len(), input.len());
    assert!(first.iter().all(|v| v.is_finite() && *v > 0.1));
    settings.semitones = 12.0;
    engine.process(&input, 48000, &settings).unwrap();
    let shifted = engine.process(&input, 48000, &settings).unwrap();
    assert!(shifted[shifted.len() - 1] > first[first.len() - 1] + 0.1);
    settings.bypass = true;
    assert_eq!(engine.process(&input, 48000, &settings).unwrap(), input);
    settings.bypass = false;
    let reset = engine.process(&input, 48000, &settings).unwrap();
    assert_eq!(reset.len(), input.len());
}

#[test]
#[ignore = "requires ONNX Runtime 1.22; set ORT_DYLIB_PATH"]
fn silence_and_missing_retrieval_bank_are_handled_explicitly() {
    let mut engine = engine();
    let mut settings = Settings {
        bypass: false,
        ..Settings::default()
    };
    assert_eq!(
        engine.process(&vec![0.0; 7680], 48000, &settings).unwrap(),
        vec![0.0; 7680]
    );
    settings.index_ratio = 0.5;
    let error = engine
        .process(&vec![0.2; 7680], 48000, &settings)
        .unwrap_err();
    assert!(error.to_string().contains("feature_bank"));
}

#[test]
fn missing_manifest_is_an_error() {
    assert!(load_converter(Path::new("does-not-exist.json")).is_err());
}
