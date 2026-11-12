use anyhow::{bail, Context, Result};
use std::path::Path;
use voice_changer_native::{
    config::Settings,
    engine::{self, Engine},
};

fn main() -> Result<()> {
    // The terminal shows what the app does; RUST_LOG=debug shows more, RUST_LOG=warn less.
    env_logger::Builder::from_env(
        // Graphics libraries only report real failures; their warnings are noise here.
        env_logger::Env::default().default_filter_or(
            "warn,voice_changer_native=info,wgpu_hal=error,wgpu_core=error,naga=error,epaint=error,egui_winit=error,eframe=error",
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
                None | Some("cpu") => voice_changer_native::config::Device::Cpu,
                Some("gpu" | "cuda") => voice_changer_native::config::Device::Gpu,
                Some("directml") => voice_changer_native::config::Device::DirectMl,
                Some(other) => bail!("Unknown device {other}; use cpu, cuda or directml"),
            };
            let settings = Settings { block_ms, device, ..Settings::default() };
            println!("{}", serde_json::to_string_pretty(&voice_changer_native::benchmark::run(Path::new(&args[1]), &settings, blocks)?)?);
        }
        Some("--help" | "-h") => println!("Voice Changer\n  (no arguments)              Open egui app\n  --devices                   List audio device IDs\n  --convert MODEL.json IN.wav OUT.wav [SEMITONES [INDEX_RATIO]]\n  --benchmark MODEL.json [BLOCK_MS [BLOCKS [cpu|cuda|directml]]]  Time conversion on a device\n  --bypass IN.wav OUT.wav      Test audio pipeline without models\n  --export-rvc MODEL.pth OUT.onnx [--realtime]  Convert an RVC v2 checkpoint to ONNX\n\nONNX: set ORT_DYLIB_PATH to ONNX Runtime 1.22 shared library."),
        Some("--export-rvc") if args.len() == 3 || (args.len() == 4 && args[3] == "--realtime") => {
            #[cfg(feature = "export")]
            println!("{}", serde_json::to_string(&voice_changer_native::export::export_rvc(Path::new(&args[1]), Path::new(&args[2]), args.len() == 4)?)?);
            #[cfg(not(feature = "export"))]
            bail!("Build with the export feature");
        }
        Some("--devices") => {
            #[cfg(feature = "audio")]
            for device in voice_changer_native::audio::devices()? { println!("{}\t{}\t{}", if device.input { "input" } else { "output" }, device.id, device.name); }
            #[cfg(not(feature = "audio"))]
            bail!("Build with the audio feature");
        }
        Some("--bypass") if args.len() == 3 => {
            voice_changer_native::wav::convert(&mut Engine::new(None), Path::new(&args[1]), Path::new(&args[2]), &Settings::default())?;
        }
        Some("--convert") if (4..=6).contains(&args.len()) => {
            let converter = engine::load_converter(Path::new(&args[1]))?;
            let settings = Settings { bypass: false, semitones: args.get(4).map(|s| s.parse()).transpose().context("Invalid semitones")?.unwrap_or(0.0), index_ratio: args.get(5).map(|s| s.parse()).transpose().context("Invalid index ratio")?.unwrap_or(0.0), ..Settings::default() };
            voice_changer_native::wav::convert(&mut Engine::new(Some(converter)), Path::new(&args[2]), Path::new(&args[3]), &settings)?;
        }
        None => {
            #[cfg(feature = "gui")]
            voice_changer_native::run(eframe::NativeOptions {
                // The app id matches voice-changer.desktop, so Wayland taskbars and docks
                // show that entry's icon; the window icon covers X11 and Windows.
                viewport: eframe::egui::ViewportBuilder::default()
                    .with_title("Voice Changer")
                    .with_app_id("voice-changer")
                    .with_icon(eframe::icon_data::from_png_bytes(include_bytes!("../assets/hicolor/256x256/apps/voice-changer.png")).expect("bundled icon is a valid PNG"))
                    .with_inner_size([1060.0, 760.0])
                    .with_min_inner_size([360.0, 540.0]),
                ..Default::default()
            }).map_err(|e| anyhow::anyhow!("{e}"))?;
            #[cfg(not(feature = "gui"))]
            bail!("GUI disabled; use --help for CLI options");
        }
        _ => bail!("Unknown arguments; use --help"),
    }
    Ok(())
}
