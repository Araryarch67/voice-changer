use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use voice_changer::{
    analyze,
    config::{Device, F0Detector, Noise, Settings},
    engine::{self, Engine},
};

fn parse_noise(name: &str) -> Result<Noise> {
    match name.to_ascii_lowercase().as_str() {
        "off" => Ok(Noise::Off),
        "sup1" => Ok(Noise::Sup1),
        "sup2" => Ok(Noise::Sup2),
        other => bail!("Unknown noise mode {other}; use off, sup1, or sup2"),
    }
}

fn parse_f0(name: &str) -> Result<F0Detector> {
    match name.to_ascii_lowercase().as_str() {
        "rmvpe" => Ok(F0Detector::Rmvpe),
        "crepe_full" | "crepe" => Ok(F0Detector::CrepeFull),
        "crepe_tiny" => Ok(F0Detector::CrepeTiny),
        "harvest" => Ok(F0Detector::Harvest),
        "dio" => Ok(F0Detector::Dio),
        "yin" => Ok(F0Detector::Yin),
        other => bail!(
            "Unknown F0 detector {other}; use rmvpe, crepe_full, crepe_tiny, harvest, dio, or yin"
        ),
    }
}

fn default_report_prefix(wav: &Path) -> PathBuf {
    let stem = wav.file_stem().unwrap_or_default().to_string_lossy();
    wav.with_file_name(stem.as_ref())
}

fn run_analyze(args: &[String]) -> Result<()> {
    if args.is_empty() {
        bail!("Usage: --analyze IN.wav [OUT.wav] [--out PREFIX]");
    }
    let mut positional = Vec::new();
    let mut prefix: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                let value = args.get(i).context("--out requires a path prefix")?;
                prefix = Some(PathBuf::from(value));
            }
            arg if arg.starts_with('-') => bail!("Unknown option {arg} for --analyze"),
            arg => positional.push(PathBuf::from(arg)),
        }
        i += 1;
    }
    if !(1..=2).contains(&positional.len()) {
        bail!("Usage: --analyze IN.wav [OUT.wav] [--out PREFIX]");
    }
    let second = positional.get(1).map(PathBuf::as_path);
    let prefix = prefix.unwrap_or_else(|| default_report_prefix(second.unwrap_or(&positional[0])));
    let (report, json_path, png_path) =
        analyze::analyze_wavs_to_disk(&positional[0], second, &prefix)?;
    println!("{}", analyze::format_summary(&report));
    println!("Report JSON: {}", json_path.display());
    if let Some(png) = png_path {
        println!("Report PNG: {}", png.display());
    }
    Ok(())
}

fn run_convert(args: &[String]) -> Result<()> {
    if args.len() < 3 {
        bail!("Usage: --convert MODEL.json IN.wav OUT.wav [SEMITONES [INDEX_RATIO]] [--gpu] [--block-ms MS] [--context-ms MS] [--f0 METHOD] [--protect VAL] [--analyze]");
    }
    let model = Path::new(&args[0]);
    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    let mut settings = Settings {
        bypass: false,
        ..Settings::default()
    };
    let mut do_analyze = false;
    let mut positional_idx = 0usize;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--gpu" | "--cuda" => settings.device = Device::Gpu,
            "--directml" => settings.device = Device::DirectMl,
            "--cpu" => settings.device = Device::Cpu,
            "--analyze" => do_analyze = true,
            "--block-ms" => {
                i += 1;
                settings.block_ms = args
                    .get(i)
                    .context("--block-ms requires a value")?
                    .parse()?;
            }
            "--context-ms" => {
                i += 1;
                settings.context_ms = args
                    .get(i)
                    .context("--context-ms requires a value")?
                    .parse()?;
            }
            "--f0" => {
                i += 1;
                settings.f0_detector = parse_f0(args.get(i).context("--f0 requires a method")?)?;
            }
            "--protect" => {
                i += 1;
                settings.protect = args.get(i).context("--protect requires a value")?.parse()?;
            }
            "--index" => {
                i += 1;
                settings.index_ratio = args.get(i).context("--index requires a value")?.parse()?;
            }
            "--overlap-ms" => {
                i += 1;
                settings.overlap_ms = args
                    .get(i)
                    .context("--overlap-ms requires a value")?
                    .parse()?;
            }
            "--search-ms" => {
                i += 1;
                settings.search_ms = args
                    .get(i)
                    .context("--search-ms requires a value")?
                    .parse()?;
            }
            "--silence-thresh" => {
                i += 1;
                settings.silence_threshold = args
                    .get(i)
                    .context("--silence-thresh requires a value")?
                    .parse()?;
            }
            "--noise" => {
                i += 1;
                settings.noise =
                    parse_noise(args.get(i).context("--noise requires off|sup1|sup2")?)?;
            }
            "--input-gain" => {
                i += 1;
                settings.input_gain = args
                    .get(i)
                    .context("--input-gain requires a value")?
                    .parse()?;
            }
            "--output-gain" => {
                i += 1;
                settings.output_gain = args
                    .get(i)
                    .context("--output-gain requires a value")?
                    .parse()?;
            }
            arg if !arg.starts_with("--") && positional_idx < 2 => {
                if positional_idx == 0 {
                    settings.semitones = arg.parse().context("Invalid semitones")?;
                } else {
                    settings.index_ratio = arg.parse().context("Invalid index ratio")?;
                }
                positional_idx += 1;
            }
            other => bail!("Unknown option {other} for --convert"),
        }
        i += 1;
    }
    let converter = engine::load_converter_on(model, settings.device)?;
    voice_changer::wav::convert(&mut Engine::new(Some(converter)), input, output, &settings)?;
    if do_analyze {
        let prefix = default_report_prefix(output);
        let model_label = model.display().to_string();
        let (report, json_path, png_path) = analyze::analyze_wavs_with_context(
            input,
            Some(output),
            &prefix,
            Some(&model_label),
            Some(&settings),
        )?;
        println!("{}", analyze::format_summary(&report));
        println!("Report JSON: {}", json_path.display());
        if let Some(png) = png_path {
            println!("Report PNG: {}", png.display());
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    // The terminal shows what the app does; RUST_LOG=debug shows more, RUST_LOG=warn less.
    env_logger::Builder::from_env(
        // Graphics libraries only report real failures; their warnings are noise here.
        env_logger::Env::default().default_filter_or(
            "warn,voice_changer=info,wgpu_hal=error,wgpu_core=error,naga=error,epaint=error,egui_winit=error,eframe=error",
        ),
    )
    .format_timestamp_millis()
    .init();
    log::info!(
        "Voice Changer {} on {} {}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--benchmark") if (2..=5).contains(&args.len()) => {
            let block_ms = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(Settings::default().block_ms);
            let blocks = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(6);
            let device = match args.get(4).map(String::as_str) {
                None | Some("cpu") => Device::Cpu,
                Some("gpu" | "cuda") => Device::Gpu,
                Some("directml") => Device::DirectMl,
                Some(other) => bail!("Unknown device {other}; use cpu, cuda or directml"),
            };
            let settings = Settings { block_ms, device, ..Settings::default() };
            println!("{}", serde_json::to_string_pretty(&voice_changer::benchmark::run(Path::new(&args[1]), &settings, blocks)?)?);
        }
        Some("--analyze") => run_analyze(&args[1..])?,
        Some("--help" | "-h") => println!(
            "Voice Changer\n  (no arguments)              Open egui app\n  --devices                   List audio device IDs\n  --convert MODEL.json IN.wav OUT.wav [SEMITONES [INDEX_RATIO]] [--gpu|--directml|--cpu] [--block-ms MS] [--context-ms MS] [--f0 METHOD] [--protect VAL] [--analyze]\n  --analyze IN.wav [OUT.wav] [--out PREFIX]  Generate JSON + PNG waveform/spectrogram/pitch diagnostics\n  --benchmark MODEL.json [BLOCK_MS [BLOCKS [cpu|cuda|directml]]]  Time conversion on a device\n  --bypass IN.wav OUT.wav      Test audio pipeline without models\n  --export-rvc MODEL.pth OUT.onnx [--realtime]  Convert an RVC v2 checkpoint to ONNX\n\nONNX: set ORT_DYLIB_PATH to ONNX Runtime 1.22 shared library."
        ),
        Some("--export-rvc") if args.len() == 3 || (args.len() == 4 && args[3] == "--realtime") => {
            #[cfg(feature = "export")]
            println!("{}", serde_json::to_string(&voice_changer::export::export_rvc(Path::new(&args[1]), Path::new(&args[2]), args.len() == 4)?)?);
            #[cfg(not(feature = "export"))]
            bail!("Build with the export feature");
        }
        Some("--devices") => {
            #[cfg(feature = "audio")]
            for device in voice_changer::audio::devices()? { println!("{}\t{}\t{}", if device.input { "input" } else { "output" }, device.id, device.name); }
            #[cfg(not(feature = "audio"))]
            bail!("Build with the audio feature");
        }
        Some("--bypass") if args.len() == 3 => {
            voice_changer::wav::convert(&mut Engine::new(None), Path::new(&args[1]), Path::new(&args[2]), &Settings::default())?;
        }
        Some("--convert") => run_convert(&args[1..])?,
        None => {
            #[cfg(feature = "gui")]
            voice_changer::run(eframe::NativeOptions {
                // The app id matches voice-changer.desktop, so Wayland taskbars and docks
                // show that entry's icon; the window icon covers X11 and Windows.
                viewport: eframe::egui::ViewportBuilder::default()
                    .with_title("Voice Changer")
                    .with_app_id("voice-changer")
                    .with_icon(eframe::icon_data::from_png_bytes(include_bytes!("../assets/hicolor/256x256/apps/voice-changer.png")).expect("bundled icon is a valid PNG"))
                    .with_inner_size([1060.0, 760.0])
                    .with_min_inner_size([360.0, 540.0]),
                // In the middle of the screen the first time; after that the window
                // comes back where it was left.
                centered: true,
                ..Default::default()
            }).map_err(|e| anyhow::anyhow!("{e}"))?;
            #[cfg(not(feature = "gui"))]
            bail!("GUI disabled; use --help for CLI options");
        }
        _ => bail!("Unknown arguments; use --help"),
    }
    Ok(())
}
