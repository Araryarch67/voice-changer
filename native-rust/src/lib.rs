pub mod analyze;
#[cfg(feature = "gui")]
pub mod app;
#[cfg(feature = "audio")]
pub mod audio;
pub mod benchmark;
#[cfg(feature = "catalog")]
pub mod catalog;
pub mod config;
pub mod denoise;
pub mod dsp;
pub mod echo;
pub mod engine;
#[cfg(feature = "export")]
pub mod export;
#[cfg(feature = "gui")]
pub mod gpu_info;
#[cfg(feature = "gui")]
pub mod i18n;
#[cfg(feature = "export")]
pub mod import;
#[cfg(feature = "gui")]
pub mod m3;
pub mod pitch;
pub mod platform;
#[cfg(feature = "export")]
pub mod pth;
pub mod retrieval;
pub mod runtime;
#[cfg(any(feature = "onnx", feature = "onnx-static"))]
pub mod rvc;
#[cfg(feature = "catalog")]
pub mod setup;
#[cfg(all(feature = "audio", target_os = "linux"))]
pub mod virtual_mic;
pub mod wav;
#[cfg(windows)]
pub mod windows_mic;

#[cfg(feature = "gui")]
pub fn run(options: eframe::NativeOptions) -> eframe::Result {
    eframe::run_native(
        "Voice Changer",
        options,
        Box::new(|cc| Ok(Box::new(app::VoiceApp::new(cc)))),
    )
}

#[cfg(all(target_os = "android", feature = "android"))]
#[no_mangle]
pub fn android_main(android_app: android_activity::AndroidApp) {
    platform::android_init(android_app.clone());
    let options = eframe::NativeOptions {
        android_app: Some(android_app),
        ..Default::default()
    };
    if let Err(error) = run(options) {
        eprintln!("Voice Changer: {error}");
    }
}

/// Called on the main thread by the iOS application launcher.
#[cfg(all(target_os = "ios", feature = "gui"))]
#[no_mangle]
pub extern "C" fn voice_changer_ios_main() -> i32 {
    match run(eframe::NativeOptions::default()) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("Voice Changer: {error}");
            1
        }
    }
}
