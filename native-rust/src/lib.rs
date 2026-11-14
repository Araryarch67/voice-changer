pub mod analyze;
#[cfg(feature = "gui")]
pub mod app;
#[cfg(feature = "audio")]
pub mod audio;
pub mod audio_file;
pub mod benchmark;
#[cfg(feature = "catalog")]
pub mod catalog;
pub mod cli;
pub mod config;
#[cfg(feature = "catalog")]
pub mod cuda_setup;
pub mod dataset;
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
#[cfg(feature = "audio")]
pub mod recorder;
pub mod retrieval;
pub mod runtime;
#[cfg(any(feature = "onnx", feature = "onnx-static"))]
pub mod rvc;
#[cfg(any(feature = "onnx", feature = "onnx-static"))]
pub mod separate;
#[cfg(feature = "catalog")]
pub mod setup;
#[cfg(all(feature = "audio", target_os = "linux"))]
pub mod virtual_mic;
pub mod wav;
#[cfg(windows)]
pub mod windows_mic;

#[cfg(feature = "gui")]
pub fn run(options: eframe::NativeOptions) -> eframe::Result {
    let options = eframe::NativeOptions {
        wgpu_options: wgpu_options(),
        ..options
    };
    eframe::run_native(
        "Voice Changer",
        options,
        Box::new(|cc| Ok(Box::new(app::VoiceApp::new(cc)))),
    )
}

/// egui's GPU setup, asking only for limits the GPU has: egui asks for wgpu's
/// defaults (64 KB uniform buffers, 8192 px textures), which some phone GPUs and
/// the Android emulator do not offer, and the window then never opens. Such a GPU
/// gets the limits it reports instead, which egui's small buffers fit.
#[cfg(feature = "gui")]
fn wgpu_options() -> eframe::egui_wgpu::WgpuConfiguration {
    use eframe::egui_wgpu::{WgpuConfiguration, WgpuSetup, WgpuSetupCreateNew};
    let default = WgpuSetupCreateNew::default();
    let egui_device = default.device_descriptor.clone();
    WgpuConfiguration {
        wgpu_setup: WgpuSetup::CreateNew(WgpuSetupCreateNew {
            device_descriptor: std::sync::Arc::new(move |adapter| {
                let mut descriptor = egui_device(adapter);
                let supported = adapter.limits();
                if !descriptor.required_limits.check_limits(&supported) {
                    log::warn!("The GPU offers less than wgpu's default limits; using its own");
                    descriptor.required_limits = supported;
                }
                descriptor
            }),
            ..default
        }),
        ..Default::default()
    }
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
