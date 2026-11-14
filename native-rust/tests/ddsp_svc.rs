//! DDSP-SVC graphs, synthesizers, mel and diffusion against the reference's
//! PyTorch classes. Write the fixtures with `tools/dev/verify_ddsp_svc.py OUT_DIR`,
//! then run
//! `DDSP_FIXTURES=OUT_DIR ORT_DYLIB_PATH=... cargo test --test ddsp_svc -- --ignored --nocapture`.
#![cfg(all(feature = "export", any(feature = "onnx", feature = "onnx-static")))]
use ort::{session::Session, value::Tensor};
use std::path::{Path, PathBuf};
use voice_changer::{
    ddsp_svc::{self, DdspInfo, Excitation, MelExtractor, Synthesizer},
    diffusion_svc::{self, Info, Noise, MEL_BINS},
};

fn fixtures() -> PathBuf {
    PathBuf::from(std::env::var_os("DDSP_FIXTURES").expect("set DDSP_FIXTURES"))
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

fn check(what: &str, got: &[f32], want: &[f32], tolerance: f32) {
    let error = relative_error(got, want);
    println!("{what}: {error:.2e}");
    assert!(error < tolerance, "{what}: relative error {error}");
}

fn session(file: &Path) -> Session {
    Session::builder()
        .unwrap()
        .with_intra_threads(2)
        .unwrap()
        .commit_from_file(file)
        .unwrap()
}

fn run(
    session: &mut Session,
    frames: usize,
    inputs: &[(&str, &[f32], usize)],
    speaker: Option<i64>,
) -> Vec<f32> {
    let mut feeds = Vec::new();
    for (name, data, width) in inputs {
        feeds.push((
            std::borrow::Cow::Owned(name.to_string()),
            ort::session::SessionInputValue::from(
                Tensor::from_array(([1, frames, *width], data.to_vec().into_boxed_slice()))
                    .unwrap(),
            ),
        ));
    }
    if let Some(speaker) = speaker {
        if session.inputs.iter().any(|i| i.name == "speaker") {
            feeds.push((
                "speaker".into(),
                Tensor::from_array(([1], vec![speaker].into_boxed_slice()))
                    .unwrap()
                    .into(),
            ));
        }
    }
    session.run(feeds).unwrap()[0]
        .try_extract_tensor::<f32>()
        .unwrap()
        .1
        .to_vec()
}

/// A stand-in for NSF-HiFiGAN with its inputs and output shapes: each frame's
/// mean mel value, squashed and held for the 512 samples of the frame.
fn stand_in_vocoder(file: &Path) {
    use voice_changer::onnx_build::{Attr, Dim, Graph};
    let mut g = Graph::new();
    let frames = Dim::Named("frames");
    let mel = g.input("mel", &[Dim::Fixed(1), frames, Dim::Fixed(128)]);
    let _f0 = g.input("f0", &[Dim::Fixed(1), frames]);
    let mean = g.op(
        "ReduceMean",
        &[&mel],
        &[("axes", Attr::Ints(vec![2])), ("keepdims", Attr::Int(1))],
    );
    let level = g.op("Tanh", &[&mean], &[]);
    let repeats = g.ints(&[1, 1, 512]);
    let held = g.op("Tile", &[&level, &repeats], &[]);
    let shape = g.ints(&[1, -1]);
    let audio = g.op("Reshape", &[&held, &shape], &[]);
    g.output("audio", &audio, &[Dim::Fixed(1), Dim::Named("samples")]);
    std::fs::write(file, g.finish("stand_in_vocoder", &[])).unwrap();
}

/// The whole voice, block by block through the engine, in its three modes
/// (diffusion, enhancer, the synthesizer alone): the random-weight CombSub model
/// and its diffusion model with the synthetic embedder and pitch models from
/// `tests/fixtures`, and NSF-HiFiGAN (`NSF_HIFIGAN_ONNX`) or, without it, a
/// stand-in. Checks plumbing, not sound.
#[test]
#[ignore = "needs DDSP_FIXTURES from tools/dev/verify_ddsp_svc.py and ONNX Runtime"]
fn a_voice_converts_blocks_end_to_end() {
    use voice_changer::{config::Settings, engine::Engine};
    let folder = std::env::temp_dir().join(format!("ddsp-svc-voice-{}", std::process::id()));
    let case = fixtures().join("combsub");
    let exported = ddsp_svc::export(
        &case.join("ddsp").join("model_1000.pt"),
        Some(&case.join("diff").join("model_1000.pt")),
        &folder,
    )
    .unwrap();
    let vocoder = match std::env::var_os("NSF_HIFIGAN_ONNX") {
        Some(file) => PathBuf::from(file),
        None => {
            let file = folder.join("stand-in-vocoder.onnx");
            stand_in_vocoder(&file);
            file
        }
    };
    let synthetic = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let manifest = serde_json::json!({
        "engine": "ddsp-svc",
        "name": "Random-weight DDSP-SVC (not a voice)",
        "model": exported.ddsp,
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
    for (diffusion, enhancer) in [(true, true), (false, true), (false, false)] {
        let settings = Settings {
            bypass: false,
            diffusion,
            enhancer,
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
                "diffusion {diffusion}, enhancer {enhancer}: block {b} in {} ms",
                started.elapsed().as_millis()
            );
        }
        assert!(
            energy > 0.0,
            "diffusion {diffusion}, enhancer {enhancer}: silent output"
        );
    }
    let _ = std::fs::remove_dir_all(&folder);
}

#[test]
#[ignore = "needs DDSP_FIXTURES from tools/dev/verify_ddsp_svc.py"]
fn volume_and_mask_match_the_reference() {
    let folder = fixtures();
    let audio = read(&folder, "audio");
    let volume = ddsp_svc::volume(&audio, 512);
    check("volume", &volume, &read(&folder, "volume"), 1e-5);
    let mask = diffusion_svc::mask_below(&volume, 512, ddsp_svc::MASK_DB);
    check("mask", &mask, &read(&folder, "mask"), 1e-6);
}

#[test]
#[ignore = "needs DDSP_FIXTURES from tools/dev/verify_ddsp_svc.py and ONNX Runtime"]
fn graphs_synthesizers_and_diffusion_match_the_reference() {
    voice_changer::runtime::initialize().unwrap();
    for name in ["sins", "combsub", "combsubfast"] {
        let folder = fixtures().join(name);
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(folder.join("meta.json")).unwrap()).unwrap();
        let frames = meta["frames"].as_u64().unwrap() as usize;
        let speaker = meta["speaker"].as_i64().unwrap();
        let out = std::env::temp_dir().join(format!("ddsp-svc-test-{name}-{}", std::process::id()));
        let exported = ddsp_svc::export(
            &folder.join("ddsp").join("model_1000.pt"),
            Some(&folder.join("diff").join("model_1000.pt")),
            &out,
        )
        .unwrap();
        let info: DdspInfo = exported.info.clone();
        let (units, f0, volume) = (
            read(&folder, "units"),
            read(&folder, "f0"),
            read(&folder, "volume"),
        );

        let phase = Excitation::new(&f0, &info).phase_frames(info.block_size);
        check(
            &format!("{name}: phase"),
            &phase,
            &read(&folder, "phase"),
            1e-5,
        );

        let mut ddsp = session(&exported.ddsp);
        let metadata: DdspInfo = serde_json::from_str(
            &ddsp
                .metadata()
                .unwrap()
                .custom("ddsp_svc")
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(metadata, info);
        let controls = run(
            &mut ddsp,
            frames,
            &[
                ("units", &units, info.units),
                ("f0", &f0, 1),
                ("phase", &read(&folder, "phase"), 1),
                ("volume", &volume, 1),
            ],
            Some(speaker),
        );
        let want_controls = read(&folder, "controls");
        check(
            &format!("{name}: controls"),
            &controls,
            &want_controls,
            1e-4,
        );

        let signal =
            Synthesizer::new().synthesize(&info, &want_controls, &f0, &read(&folder, "uniform"));
        let want_signal = read(&folder, "signal");
        check(&format!("{name}: signal"), &signal, &want_signal, 2e-4);

        let mel = MelExtractor::new().extract(&want_signal);
        let want_mel = read(&folder, "mel");
        check(&format!("{name}: mel"), &mel, &want_mel, 1e-4);

        let mut condition = session(exported.condition.as_ref().unwrap());
        let mut denoiser = session(exported.denoiser.as_ref().unwrap());
        let diffusion: Info = exported.diffusion.clone().unwrap();
        assert_eq!(diffusion.k_step_max, 1000);
        let cond = run(
            &mut condition,
            frames,
            &[
                ("units", &units, info.units),
                ("f0", &f0, 1),
                ("volume", &volume, 1),
            ],
            Some(speaker),
        );
        let hidden = cond.len() / frames;
        let k_step = meta["k_step"].as_u64().unwrap() as usize;
        let speedup = meta["speedup"].as_u64().unwrap() as usize;
        assert_eq!(ddsp_svc::steps_for(&diffusion, 0, 0), (k_step, speedup));
        let mut denoise = |x: &[f32], step: f32| -> anyhow::Result<Vec<f32>> {
            let outputs = denoiser.run(ort::inputs! {
                "x" => Tensor::from_array(([1, 1, MEL_BINS, frames], x.to_vec().into_boxed_slice()))?,
                "step" => Tensor::from_array(([1], vec![step].into_boxed_slice()))?,
                "condition" => Tensor::from_array(([1, hidden, frames], cond.clone().into_boxed_slice()))?,
            })?;
            Ok(outputs[0].try_extract_tensor::<f32>()?.1.to_vec())
        };
        let diffused = diffusion_svc::shallow_diffusion(
            &diffusion,
            &want_mel,
            k_step,
            speedup,
            &mut Noise::new(0),
            &mut denoise,
        )
        .unwrap();
        check(
            &format!("{name}: diffused"),
            &diffused,
            &read(&folder, "diffused"),
            1e-4,
        );
        let _ = std::fs::remove_dir_all(&out);
    }
}
