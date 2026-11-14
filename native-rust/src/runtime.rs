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

/// The folder deleted with a voice: the folder holding its `model.json`, directly
/// inside one of `roots` (the models folders), and never `shared` or a hidden one.
/// `None` for a manifest anywhere else, whose files are left alone.
pub fn voice_folder(manifest: &std::path::Path, roots: &[PathBuf]) -> Option<PathBuf> {
    let manifest = manifest.canonicalize().ok()?;
    if manifest.file_name()? != "model.json" {
        return None;
    }
    let folder = manifest.parent()?;
    let name = folder.file_name()?.to_str()?;
    if name == "shared" || name.starts_with('.') {
        return None;
    }
    let parent = folder.parent()?;
    roots
        .iter()
        .filter_map(|root| root.canonicalize().ok())
        .any(|root| root == parent)
        .then(|| folder.to_path_buf())
}

/// Deletes the voice of `manifest` from disk when it lives in a models folder (see
/// [`voice_folder`]), and returns the folder deleted; `None` when there was none to
/// delete.
pub fn delete_voice(manifest: &std::path::Path) -> std::io::Result<Option<PathBuf>> {
    let Some(folder) = voice_folder(manifest, &model_directories()) else {
        return Ok(None);
    };
    std::fs::remove_dir_all(&folder)?;
    log::info!("Deleted {}", folder.display());
    Ok(Some(folder))
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

#[cfg(all(feature = "onnx", not(feature = "onnx-static")))]
fn log_runtime(path: &std::path::Path) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| log::info!("ONNX Runtime loaded from {}", path.display()));
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
            log_runtime(&path);
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
                log_runtime(&path);
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
/// On Windows the CUDA provider's dependencies are found through `PATH`, which the
/// cuDNN 9 installer does not extend, so its folders (and `lib\` beside the app, for
/// DLLs copied there) are added for this process.
pub fn preload_gpu_libraries() {
    // GPUs newer than the runtime's prebuilt kernels (RTX 50 with ONNX Runtime 1.22)
    // compile them from PTX on first use. CUDA caches the result, but its default
    // 256 MiB cache is too small for ONNX Runtime and would recompile every launch.
    static CACHE: std::sync::Once = std::sync::Once::new();
    CACHE.call_once(|| {
        log::info!("Preparing CUDA libraries");
        if std::env::var_os("CUDA_CACHE_MAXSIZE").is_none() {
            std::env::set_var("CUDA_CACHE_MAXSIZE", "4294967296");
        }
    });
    #[cfg(target_os = "windows")]
    {
        static ADDED: std::sync::Once = std::sync::Once::new();
        ADDED.call_once(|| {
            let mut folders = cuda_folders();
            if let Some(path) = std::env::var_os("PATH") {
                folders.extend(std::env::split_paths(&path));
            }
            if let Ok(joined) = std::env::join_paths(folders) {
                std::env::set_var("PATH", joined);
            }
        });
    }
    #[cfg(target_os = "linux")]
    {
        use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
        // Not once: the app's CUDA folder can be filled while it runs.
        preload_cuda_folder();
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

/// The folder the app downloads NVIDIA's CUDA 12 and cuDNN 9 libraries into
/// (`cuda_setup`), per user so it is writable wherever the app is installed:
/// `%LOCALAPPDATA%\voice-changer\cuda12` on Windows,
/// `${XDG_DATA_HOME:-~/.local/share}/voice-changer/cuda12` on Linux.
pub fn cuda_library_folder() -> Option<PathBuf> {
    let data = if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "linux") {
        std::env::var_os("XDG_DATA_HOME")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
            })
    } else {
        None
    }?;
    Some(data.join("voice-changer").join("cuda12"))
}

/// The CUDA and cuDNN libraries the CUDA build of ONNX Runtime 1.22 loads, in an
/// order where each one's dependencies come before it. From the imports of
/// `onnxruntime_providers_cuda` (Windows: cuBLAS, cuFFT, the CUDA runtime and cuDNN;
/// Linux adds cuRAND and NVRTC), plus the parts cuDNN loads by itself.
pub fn cuda_libraries() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &[
            "cudart64_12.dll",
            "cublasLt64_12.dll",
            "cublas64_12.dll",
            "cufft64_11.dll",
            "cudnn_graph64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_adv64_9.dll",
            "cudnn64_9.dll",
        ]
    } else if cfg!(target_os = "linux") {
        &[
            "libcudart.so.12",
            "libcublasLt.so.12",
            "libcublas.so.12",
            "libcurand.so.10",
            "libcufft.so.11",
            "libnvrtc-builtins.so.12.9",
            "libnvrtc.so.12",
            "libcudnn_graph.so.9",
            "libcudnn_engines_precompiled.so.9",
            "libcudnn_engines_runtime_compiled.so.9",
            "libcudnn_heuristic.so.9",
            "libcudnn_ops.so.9",
            "libcudnn_cnn.so.9",
            "libcudnn_adv.so.9",
            "libcudnn.so.9",
        ]
    } else {
        &[]
    }
}

/// The libraries the CUDA provider needs that cannot be found; empty when CUDA
/// has everything. The cuDNN parts it loads by itself are only checked through
/// cuDNN, as a separate install may split them differently.
pub fn missing_cuda_libraries() -> Vec<&'static str> {
    preload_gpu_libraries();
    let needed = cuda_libraries()
        .iter()
        .copied()
        .filter(|name| !name.contains("cudnn_") && !name.contains("nvrtc-builtins"));
    #[cfg(target_os = "windows")]
    {
        // Windows also searches the executable's folder.
        let folders: Vec<PathBuf> = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(PathBuf::from))
            .into_iter()
            .chain(
                std::env::var_os("PATH")
                    .map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
                    .unwrap_or_default(),
            )
            .collect();
        needed
            .filter(|name| !folders.iter().any(|folder| folder.join(name).is_file()))
            .collect()
    }
    #[cfg(target_os = "linux")]
    {
        use libloading::os::unix::{Library, RTLD_LAZY};
        // SAFETY: only CUDA's own libraries, whose initialisers have no preconditions.
        needed
            .filter(|name| unsafe { Library::open(Some(*name), RTLD_LAZY) }.is_err())
            .collect()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        needed.collect()
    }
}

/// Whether an NVIDIA driver is installed: the CUDA driver library it ships.
pub fn nvidia_driver_present() -> bool {
    #[cfg(target_os = "windows")]
    {
        let system = std::env::var_os("SystemRoot")
            .map_or_else(|| PathBuf::from(r"C:\Windows"), PathBuf::from)
            .join("System32");
        system.join("nvcuda.dll").is_file()
    }
    #[cfg(target_os = "linux")]
    {
        use libloading::os::unix::{Library, RTLD_LAZY};
        // SAFETY: the driver library's initialisers have no preconditions.
        unsafe { Library::open(Some("libcuda.so.1"), RTLD_LAZY) }.is_ok()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        false
    }
}

/// Starts the CUDA execution provider without a model, which loads its libraries;
/// the error says what is missing.
#[cfg(any(feature = "onnx", feature = "onnx-static"))]
pub fn check_cuda() -> Result<()> {
    use ort::{execution_providers::CUDAExecutionProvider, session::Session};
    std::panic::catch_unwind(|| -> Result<()> {
        initialize()?;
        preload_gpu_libraries();
        Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
            .context("Start CUDA")?;
        Ok(())
    })
    .unwrap_or_else(|_| anyhow::bail!("ONNX Runtime stopped while starting CUDA"))
}

/// Loads the libraries in the app's CUDA folder by their full paths, so the CUDA
/// provider finds them by name (`LD_LIBRARY_PATH` is only read at start). Each stays
/// loaded for the life of the process. One whose dependencies are not loaded yet is
/// tried again after the others.
#[cfg(target_os = "linux")]
fn preload_cuda_folder() {
    use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
    use std::sync::Mutex;
    static LOADED: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let Some(folder) = cuda_library_folder() else {
        return;
    };
    let Ok(mut loaded) = LOADED.lock() else {
        return;
    };
    loop {
        let mut progressed = false;
        for name in cuda_libraries() {
            let path = folder.join(name);
            if loaded.contains(name) || !path.is_file() {
                continue;
            }
            // SAFETY: NVIDIA's libraries from the pinned archives; their initialisers
            // have no preconditions.
            if let Ok(library) = unsafe { Library::open(Some(&path), RTLD_NOW | RTLD_GLOBAL) } {
                std::mem::forget(library);
                loaded.push(name);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
}

/// Folders that hold the CUDA and cuDNN DLLs on Windows, newest cuDNN first.
#[cfg(target_os = "windows")]
fn cuda_folders() -> Vec<PathBuf> {
    let mut folders = Vec::new();
    // Searched even before it exists, so libraries downloaded later are found.
    folders.extend(cuda_library_folder());
    if let Some(ort_dir) = std::env::var_os("ORT_DYLIB_PATH")
        .map(PathBuf::from)
        .and_then(|path| path.parent().map(PathBuf::from))
    {
        folders.push(ort_dir);
    }
    if let Some(lib) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|parent| parent.join("lib")))
    {
        folders.push(lib);
    }
    let manifest_runtime = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime");
    if manifest_runtime.is_dir() {
        folders.push(manifest_runtime);
    }
    let program_files = std::env::var_os("ProgramFiles")
        .map_or_else(|| PathBuf::from(r"C:\Program Files"), PathBuf::from);
    folders.extend(cudnn_folders(&program_files.join(r"NVIDIA\CUDNN")));
    if let Some(cuda) = std::env::var_os("CUDA_PATH") {
        folders.push(PathBuf::from(cuda).join("bin"));
    }
    folders
}

/// cuDNN 9 installs as `v9.x\bin\12.x\` (or `v9.x\bin\` in early releases).
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
fn cudnn_folders(root: &std::path::Path) -> Vec<PathBuf> {
    let children = |dir: &std::path::Path, prefix: &str| -> Vec<PathBuf> {
        let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_dir()
                    && path
                        .file_name()
                        .is_some_and(|name| name.to_string_lossy().starts_with(prefix))
            })
            .collect();
        found.sort();
        found.reverse();
        found
    };
    let mut folders = Vec::new();
    for version in children(root, "v9") {
        let bin = version.join("bin");
        folders.extend(children(&bin, "12"));
        folders.push(bin);
    }
    folders
}

static DIRECTML_DEVICE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// The DirectX 12 adapter DirectML runs on; the UI sets it to the discrete GPU.
pub fn set_directml_device(index: i32) {
    DIRECTML_DEVICE.store(index, std::sync::atomic::Ordering::Relaxed);
}

pub fn directml_device() -> i32 {
    DIRECTML_DEVICE.load(std::sync::atomic::Ordering::Relaxed)
}

/// The devices the loaded ONNX Runtime can use here, CPU first. Checked once.
pub fn available_devices() -> &'static [crate::config::Device] {
    use crate::config::Device;
    static AVAILABLE: std::sync::OnceLock<Vec<Device>> = std::sync::OnceLock::new();
    AVAILABLE.get_or_init(|| {
        let devices: Vec<Device> = std::iter::once(Device::Cpu).chain(gpu_devices()).collect();
        log::info!(
            "Devices available: {}",
            devices
                .iter()
                .map(|device| device.label())
                .collect::<Vec<_>>()
                .join(", ")
        );
        devices
    })
}

/// The GPU execution providers the loaded ONNX Runtime offers here.
#[cfg(any(feature = "onnx", feature = "onnx-static"))]
fn gpu_devices() -> Vec<crate::config::Device> {
    use crate::config::Device;
    use ort::execution_providers::{DirectMLExecutionProvider, ExecutionProvider};
    // ort load-dynamic can panic when its shared library cannot be located.
    let offered = |provider: &dyn ExecutionProvider| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            provider.supported_by_platform()
                && initialize().is_ok()
                && provider.is_available().unwrap_or(false)
        }))
        .unwrap_or(false)
    };
    let mut devices = Vec::new();
    if gpu_available() {
        devices.push(Device::Gpu);
    }
    if offered(&DirectMLExecutionProvider::default()) {
        devices.push(Device::DirectMl);
    }
    devices
}

/// A build without ONNX Runtime has no GPU to offer.
#[cfg(not(any(feature = "onnx", feature = "onnx-static")))]
fn gpu_devices() -> Vec<crate::config::Device> {
    Vec::new()
}

/// The GPU to use when the user has not chosen: CUDA, then DirectML.
pub fn preferred_gpu() -> Option<crate::config::Device> {
    use crate::config::Device;
    [Device::Gpu, Device::DirectMl]
        .into_iter()
        .find(|device| available_devices().contains(device))
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
                preload_gpu_libraries();
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

#[cfg(test)]
mod tests {
    #[test]
    fn only_a_voice_folder_inside_models_is_deleted() {
        use super::voice_folder;
        let base = std::env::temp_dir().join(format!("vc-delete-{}", std::process::id()));
        let models = base.join("models");
        let elsewhere = base.join("elsewhere");
        for folder in ["aria", "shared", ".downloads"] {
            std::fs::create_dir_all(models.join(folder)).unwrap();
            std::fs::write(models.join(folder).join("model.json"), "{}").unwrap();
        }
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::fs::write(elsewhere.join("model.json"), "{}").unwrap();
        std::fs::write(models.join("loose.json"), "{}").unwrap();
        let roots = [models.clone()];
        let found = voice_folder(&models.join("aria/model.json"), &roots).unwrap();
        assert_eq!(found, models.join("aria").canonicalize().unwrap());
        for manifest in [
            models.join("shared/model.json"),
            models.join(".downloads/model.json"),
            models.join("loose.json"),
            elsewhere.join("model.json"),
            models.join("missing/model.json"),
        ] {
            assert_eq!(
                voice_folder(&manifest, &roots),
                None,
                "{}",
                manifest.display()
            );
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn finds_cudnn_folders_newest_first() {
        let root = std::env::temp_dir().join(format!("vc-cudnn-{}", std::process::id()));
        for folder in [
            "v9.1/bin/12.8",
            "v9.1/bin/11.8",
            "v9.10/bin/12.9",
            "v8.9/bin",
        ] {
            std::fs::create_dir_all(root.join(folder)).unwrap();
        }
        let found = super::cudnn_folders(&root);
        assert_eq!(
            found,
            vec![
                root.join("v9.10/bin/12.9"),
                root.join("v9.10/bin"),
                root.join("v9.1/bin/12.8"),
                root.join("v9.1/bin"),
            ]
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
