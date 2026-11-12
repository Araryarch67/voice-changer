#[cfg(any(feature = "onnx", feature = "onnx-static"))]
use anyhow::{Context, Result};
use std::path::PathBuf;

pub fn library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(any(target_os = "macos", target_os = "ios")) {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

pub fn model_directories() -> Vec<PathBuf> {
    let mut directories = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            directories.push(parent.join("models"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        directories.push(cwd.join("models"));
        directories.push(cwd.join("native-rust/models"));
    }
    // Development fallback; distributed builds use paths beside their executable.
    directories.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models"));
    directories
}

/// The models folder that holds the shared embedder and pitch models; imported voices
/// are created beside them.
pub fn models_root() -> Option<PathBuf> {
    model_directories().into_iter().find(|directory| {
        directory.join("shared/contentvec.onnx").is_file()
            && directory.join("shared/rmvpe.onnx").is_file()
    })
}

pub fn discover_models() -> Vec<(String, PathBuf)> {
    let mut models = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for directory in model_directories() {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let path = if path.is_dir() {
                path.join("model.json")
            } else {
                path
            };
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Ok(canonical) = path.canonicalize() else {
                continue;
            };
            if !seen.insert(canonical.clone()) {
                continue;
            }
            if let Ok(model) = crate::config::ModelManifest::load(&canonical) {
                models.push((model.name, canonical));
            }
        }
    }
    models.sort_by(|a, b| a.0.cmp(&b.0));
    models
}

/// ONNX Runtime looks for its provider libraries beside the path it was loaded from,
/// so a symlinked runtime must be opened at its real location.
#[cfg(all(feature = "onnx", not(feature = "onnx-static")))]
fn resolved(path: PathBuf) -> PathBuf {
    path.canonicalize().unwrap_or(path)
}

#[cfg(any(feature = "onnx", feature = "onnx-static"))]
pub fn initialize() -> Result<()> {
    #[cfg(all(feature = "onnx", not(feature = "onnx-static")))]
    {
        if let Some(path) = std::env::var_os("ORT_DYLIB_PATH") {
            let path = PathBuf::from(path);
            anyhow::ensure!(path.is_file(), "ONNX Runtime not found: {}", path.display());
            let path = resolved(path);
            ort::init_from(path.to_string_lossy())
                .with_name("voice-changer")
                .commit()
                .context("Initialize ONNX Runtime from ORT_DYLIB_PATH")?;
            return Ok(());
        }
        let mut candidates = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                candidates.extend([
                    parent.join("lib").join(library_name()),
                    parent.join(library_name()),
                ]);
            }
        }
        candidates.push(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("runtime")
                .join(library_name()),
        );
        for path in candidates {
            if path.is_file() {
                let path = resolved(path);
                ort::init_from(path.to_string_lossy())
                    .with_name("voice-changer")
                    .commit()
                    .context("Initialize bundled ONNX Runtime")?;
                return Ok(());
            }
        }
    }
    ort::init()
        .with_name("voice-changer")
        .commit()
        .context("Initialize system ONNX Runtime")?;
    Ok(())
}

/// Distribution builds of the CUDA provider (Arch's `onnxruntime-cuda`) do not link
/// cuDNN themselves, so its symbols must already be visible process-wide when the
/// provider library is opened. Does nothing where cuDNN is absent.
pub fn preload_gpu_libraries() {
    #[cfg(target_os = "linux")]
    {
        use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
        static LOADED: std::sync::Once = std::sync::Once::new();
        LOADED.call_once(|| {
            // SAFETY: cuDNN's initialisers have no preconditions; the handle is leaked
            // so the library stays mapped for the life of the process.
            if let Ok(library) =
                unsafe { Library::open(Some("libcudnn.so.9"), RTLD_NOW | RTLD_GLOBAL) }
            {
                std::mem::forget(library);
            }
        });
    }
}

/// Whether the loaded ONNX Runtime offers the CUDA execution provider. Checked once.
pub fn gpu_available() -> bool {
    #[cfg(any(feature = "onnx", feature = "onnx-static"))]
    {
        use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};
        static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            // ort load-dynamic can panic when its shared library cannot be located.
            std::panic::catch_unwind(|| {
                initialize().is_ok()
                    && CUDAExecutionProvider::default()
                        .is_available()
                        .unwrap_or(false)
            })
            .unwrap_or(false)
        })
    }
    #[cfg(not(any(feature = "onnx", feature = "onnx-static")))]
    {
        false
    }
}
