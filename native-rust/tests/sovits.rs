//! A so-vits-svc voice block by block through the engine. Make one with
//! `tools/dev/verify_sovits_export.py WORK BINARY` (it leaves random-weight voices in
//! WORK/models), then run
//! `SOVITS_VOICE=WORK/models/verify-768-0/voice.onnx ORT_DYLIB_PATH=... cargo test --test sovits -- --ignored --nocapture`.
//! The synthetic embedder and pitch models from `tests/fixtures` stand in for
//! ContentVec and RMVPE, so this checks plumbing, not sound.
#![cfg(any(feature = "onnx", feature = "onnx-static"))]
use std::path::{Path, PathBuf};
use voice_changer::{config::Settings, engine::Engine};

#[test]
#[ignore = "needs SOVITS_VOICE from tools/dev/verify_sovits_export.py and ONNX Runtime"]
fn a_voice_converts_blocks_end_to_end() {
    let voice = PathBuf::from(std::env::var_os("SOVITS_VOICE").expect("set SOVITS_VOICE"));
    let synthetic = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let folder = std::env::temp_dir().join(format!("vc-sovits-test-{}", std::process::id()));
    std::fs::create_dir_all(&folder).unwrap();
    let manifest = serde_json::json!({
        "engine": "so-vits-svc",
        "name": "Random-weight so-vits-svc (not a voice)",
        "model": voice,
        "embedder": synthetic.join("embedder.onnx"),
        "pitch": synthetic.join("pitch.onnx"),
        "sample_rate": 44100,
        "channels": 768,
        "use_f0": true,
    });
    let path = folder.join("model.json");
    std::fs::write(&path, manifest.to_string()).unwrap();
    let converter = voice_changer::engine::load_converter(&path).unwrap();
    let mut engine = Engine::new(Some(converter));
    let rate = 48_000;
    let settings = Settings {
        bypass: false,
        semitones: 5.0,
        speaker_id: 3,
        ..Default::default()
    };
    let block = rate as usize * settings.block_ms as usize / 1000;
    let mut energy = 0.0;
    for b in 0..4 {
        let input: Vec<f32> = (0..block)
            .map(|i| {
                let t = (b * block + i) as f32 / rate as f32;
                (std::f32::consts::TAU * 220.0 * t).sin() * 0.3
            })
            .collect();
        let started = std::time::Instant::now();
        let out = engine.process(&input, rate, &settings).unwrap();
        assert_eq!(out.len(), block);
        assert!(out.iter().all(|v| v.is_finite()));
        energy += out.iter().map(|v| v * v).sum::<f32>();
        println!("block {b} in {} ms", started.elapsed().as_millis());
    }
    assert!(energy > 0.0, "silent output");
    std::fs::remove_dir_all(&folder).unwrap();
}
