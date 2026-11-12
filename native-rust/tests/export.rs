#![cfg(feature = "export")]
use std::path::{Path, PathBuf};
use voice_changer_native::pth::{self, Value};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// `tiny.pth` and its expected values were written by `torch.save` (PyTorch zip format).
#[test]
fn checkpoint_reader_matches_torch_for_dtypes_strides_and_plain_data() {
    let root = pth::load(&fixture("tiny.pth")).unwrap();
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture("tiny-expected.json")).unwrap()).unwrap();
    for name in ["half", "transposed", "sliced", "bf16", "double"] {
        let Some(Value::Tensor(tensor)) = root.get("weight").unwrap().get(name) else {
            panic!("{name} is not a tensor");
        };
        let shape: Vec<usize> = serde_json::from_value(expected["shapes"][name].clone()).unwrap();
        let data: Vec<f32> = serde_json::from_value(expected[name].clone()).unwrap();
        assert_eq!((&tensor.shape, &tensor.data), (&shape, &data), "{name}");
    }
    assert_eq!(
        root.get("config").and_then(Value::to_json).unwrap(),
        serde_json::json!([1025, "1", [3, 7, 11], [[1, 3, 5]], 0, 0.5, 48000])
    );
    assert_eq!(root.get("version").and_then(Value::as_str), Some("v2"));
    assert_eq!(
        root.get("info").and_then(Value::as_str).map(str::len),
        Some(300)
    );
    assert_eq!(root.get("flag"), Some(&Value::Bool(true)));
    assert_eq!(root.get("none"), Some(&Value::None));
    assert_eq!(root.get("big").and_then(Value::as_int), Some(1 << 40));
    assert_eq!(root.get("neg").and_then(Value::as_int), Some(-5));
}

#[test]
fn non_rvc_or_non_zip_checkpoints_are_rejected() {
    let out = std::env::temp_dir().join("vc-export-reject.onnx");
    let error = voice_changer_native::export::export_rvc(&fixture("tiny.pth"), &out, false)
        .unwrap_err()
        .to_string();
    assert!(error.contains("speaker embedding"), "{error}");
    assert!(pth::load(&fixture("model.json")).is_err());
}

/// Exports `../voice-rvc` checkpoints in Rust and runs them through the realtime pipeline.
#[cfg(any(feature = "onnx", feature = "onnx-static"))]
#[test]
#[ignore = "requires ../voice-rvc checkpoints, models/shared ONNX and ONNX Runtime 1.22"]
fn real_checkpoints_export_and_convert() {
    use voice_changer_native::{
        config::Settings,
        engine::{load_converter, Engine},
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("vc-export-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (checkpoint, rate) in [("kobo/kobov2.pth", 48000), ("zeta/zetaTest.pth", 40000)] {
        let voice = dir.join(format!("{rate}.onnx"));
        let exported = voice_changer_native::export::export_rvc(
            &root.join("../voice-rvc").join(checkpoint),
            &voice,
            true,
        )
        .unwrap();
        assert_eq!((exported.sample_rate, exported.speakers), (rate, 109));
        let manifest = dir.join(format!("{rate}.json"));
        let shared = root.join("models/shared");
        std::fs::write(
            &manifest,
            serde_json::json!({
                "model": voice, "embedder": shared.join("contentvec.onnx"), "pitch": shared.join("rmvpe.onnx"),
                "sample_rate": rate, "channels": 768, "use_f0": true,
                "embedder_input": "audio", "embedder_output": "unit12",
            })
            .to_string(),
        )
        .unwrap();
        let mut engine = Engine::new(Some(load_converter(&manifest).unwrap()));
        let settings = Settings {
            bypass: false,
            ..Settings::default()
        };
        let input: Vec<f32> = (0..15360)
            .map(|i| 0.3 * (i as f32 * 2.0 * std::f32::consts::PI * 200.0 / 48000.0).sin())
            .collect();
        for _ in 0..2 {
            let output = engine.process(&input, 48000, &settings).unwrap();
            assert_eq!(output.len(), input.len());
            assert!(output.iter().all(|v| v.is_finite()));
            assert!(
                output.iter().any(|v| v.abs() > 1e-3),
                "{checkpoint} produced silence"
            );
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
