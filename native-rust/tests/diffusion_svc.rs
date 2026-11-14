//! Diffusion-SVC graphs and sampler against the reference's PyTorch classes.
//! Write the fixtures with `tools/dev/verify_diffusion_svc.py OUT_DIR`, then run
//! `DSVC_FIXTURES=OUT_DIR ORT_DYLIB_PATH=... cargo test --test diffusion_svc -- --ignored --nocapture`.
#![cfg(all(feature = "export", any(feature = "onnx", feature = "onnx-static")))]
use ort::{session::Session, value::Tensor};
use std::path::{Path, PathBuf};
use voice_changer::diffusion_svc::{self, Info, Noise, MEL_BINS};

fn fixtures() -> PathBuf {
    PathBuf::from(std::env::var_os("DSVC_FIXTURES").expect("set DSVC_FIXTURES"))
}

fn read(folder: &Path, name: &str) -> Vec<f32> {
    std::fs::read(folder.join(format!("{name}.f32")))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// Largest difference relative to the reference's peak.
fn relative_error(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    let peak = want.iter().fold(0.0_f32, |m, v| m.max(v.abs())).max(1e-6);
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
        / peak
}

fn session(file: &Path) -> Session {
    Session::builder()
        .unwrap()
        .with_intra_threads(2)
        .unwrap()
        .commit_from_file(file)
        .unwrap()
}

fn info_of(denoiser: &Session) -> Info {
    let metadata = denoiser.metadata().unwrap();
    serde_json::from_str(&metadata.custom("diffusion_svc").unwrap().unwrap()).unwrap()
}

#[test]
#[ignore = "needs DSVC_FIXTURES from tools/dev/verify_diffusion_svc.py and ONNX Runtime"]
fn graphs_and_sampler_match_the_reference() {
    voice_changer::runtime::initialize().unwrap();
    for name in ["single", "speakers"] {
        let folder = fixtures().join(name);
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(folder.join("meta.json")).unwrap()).unwrap();
        let frames = meta["frames"].as_u64().unwrap() as usize;
        let hidden = meta["hidden"].as_u64().unwrap() as usize;
        let speaker = meta["speaker"].as_i64().unwrap();
        let out = tempfile_dir(name);
        let exported = diffusion_svc::export(&folder.join("combo.ptc"), &out).unwrap();
        let (units, f0, volume) = (
            read(&folder, "units"),
            read(&folder, "f0"),
            read(&folder, "volume"),
        );
        let inputs = |session: &Session| {
            let mut feeds = ort::inputs! {
                "units" => Tensor::from_array(([1, frames, 768], units.clone().into_boxed_slice())).unwrap(),
                "f0" => Tensor::from_array(([1, frames, 1], f0.clone().into_boxed_slice())).unwrap(),
                "volume" => Tensor::from_array(([1, frames, 1], volume.clone().into_boxed_slice())).unwrap(),
            };
            if session.inputs.iter().any(|i| i.name == "speaker") {
                feeds.push((
                    "speaker".into(),
                    Tensor::from_array(([1], vec![speaker].into_boxed_slice()))
                        .unwrap()
                        .into(),
                ));
            }
            feeds
        };

        let mut naive = session(&exported.naive);
        let feeds = inputs(&naive);
        let naive_mel = naive.run(feeds).unwrap()[0]
            .try_extract_tensor::<f32>()
            .unwrap()
            .1
            .to_vec();
        let error = relative_error(&naive_mel, &read(&folder, "naive"));
        println!("{name}: naive mel {error:.2e}");
        assert!(error < 1e-4, "{name}: naive mel differs by {error}");

        let mut condition_session = session(&exported.condition);
        let feeds = inputs(&condition_session);
        let condition = condition_session.run(feeds).unwrap()[0]
            .try_extract_tensor::<f32>()
            .unwrap()
            .1
            .to_vec();
        let error = relative_error(&condition, &read(&folder, "condition"));
        println!("{name}: condition {error:.2e}");
        assert!(error < 1e-5, "{name}: condition differs by {error}");

        let mut denoiser = session(&exported.denoiser);
        let info = info_of(&denoiser);
        assert_eq!(info, exported.info);
        let mut denoise = |x: &[f32], step: f32| -> anyhow::Result<Vec<f32>> {
            let outputs = denoiser.run(ort::inputs! {
                "x" => Tensor::from_array(([1, 1, MEL_BINS, frames], x.to_vec().into_boxed_slice()))?,
                "step" => Tensor::from_array(([1], vec![step].into_boxed_slice()))?,
                "condition" => Tensor::from_array(([1, hidden, frames], condition.clone().into_boxed_slice()))?,
            })?;
            Ok(outputs[0].try_extract_tensor::<f32>()?.1.to_vec())
        };
        let step = meta["step"].as_f64().unwrap() as f32;
        let noise = denoise(&read(&folder, "denoise_in"), step).unwrap();
        let error = relative_error(&noise, &read(&folder, "denoise_out"));
        println!("{name}: denoiser step {error:.2e}");
        assert!(error < 1e-5, "{name}: denoiser differs by {error}");

        for (i, run) in meta["runs"].as_array().unwrap().iter().enumerate() {
            let k_step = run["k_step"].as_u64().unwrap() as usize;
            let speedup = run["speedup"].as_u64().unwrap() as usize;
            let mel = diffusion_svc::shallow_diffusion(
                &info,
                &read(&folder, "naive"),
                k_step,
                speedup,
                &mut Noise::new(0),
                &mut denoise,
            )
            .unwrap();
            let error = relative_error(&mel, &read(&folder, &format!("mel{i}")));
            println!("{name}: shallow diffusion k_step {k_step} speedup {speedup}: {error:.2e}");
            assert!(error < 1e-4, "{name}: mel {i} differs by {error}");
        }
    }
}

fn tempfile_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vc-dsvc-test-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The whole voice, block by block through the engine: the random-weight combo
/// with the synthetic embedder and pitch models from `tests/fixtures` and the real
/// NSF-HiFiGAN vocoder (`NSF_HIFIGAN_ONNX`). Checks plumbing, not sound.
#[test]
#[ignore = "needs DSVC_FIXTURES, NSF_HIFIGAN_ONNX and ONNX Runtime"]
fn a_voice_converts_blocks_end_to_end() {
    use voice_changer::{config::Settings, engine::Engine};
    let vocoder =
        PathBuf::from(std::env::var_os("NSF_HIFIGAN_ONNX").expect("set NSF_HIFIGAN_ONNX"));
    let folder = tempfile_dir("voice");
    let exported = diffusion_svc::export(&fixtures().join("single/combo.ptc"), &folder).unwrap();
    let synthetic = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let manifest = serde_json::json!({
        "engine": "diffusion-svc",
        "name": "Random-weight Diffusion-SVC (not a voice)",
        "model": exported.naive,
        "condition": exported.condition,
        "denoiser": exported.denoiser,
        "vocoder": vocoder,
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
    for diffusion in [false, true] {
        let settings = Settings {
            bypass: false,
            diffusion,
            block_ms: 320,
            ..Default::default()
        };
        let block = rate as usize * 320 / 1000;
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
            println!(
                "diffusion {diffusion}: block {b} in {} ms",
                started.elapsed().as_millis()
            );
        }
        assert!(energy > 0.0, "diffusion {diffusion}: silent output");
    }
}
