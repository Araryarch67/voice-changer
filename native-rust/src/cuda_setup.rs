//! CUDA setup: downloads the NVIDIA libraries the CUDA build of ONNX Runtime 1.22
//! needs (the CUDA 12 runtime, cuBLAS, cuFFT and cuDNN 9; on Linux also cuRAND and
//! NVRTC) from NVIDIA's redistributable archives into the app's own folder
//! (`runtime::cuda_library_folder`), so the GPU (CUDA) download works without
//! installing CUDA and cuDNN by hand. Only the libraries ONNX Runtime loads are kept.
//!
//! Versions are pinned from NVIDIA's `redistrib_*.json` manifests: CUDA 12.9.1
//! (any CUDA 12 runtime works with ONNX Runtime's CUDA 12 build, and 12.8 or newer
//! knows RTX 50 GPUs) and cuDNN 9.12.0, the smallest recent cuDNN 9 for CUDA 12.
//! Each archive is checked against its published size and SHA-256 before anything
//! is taken from it.
use crate::catalog::{agent, Progress};
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

/// One NVIDIA redistributable archive and the libraries taken from it.
pub struct Package {
    pub title: &'static str,
    pub url: &'static str,
    pub bytes: u64,
    /// SHA-256 as NVIDIA's manifest lists it.
    pub sha256: &'static str,
    /// Library names to keep, from the archive's `bin/` or `lib/`. A versioned file
    /// (`libcublas.so.12.9.1.4`) is saved under the name it is loaded by
    /// (`libcublas.so.12`).
    pub files: &'static [&'static str],
}

#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
const WINDOWS: [Package; 4] = [
    Package {
        title: "CUDA runtime 12.9",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.9.79-archive.zip",
        bytes: 3_521_238,
        sha256: "179e9c43b0735ffe67207b3da556eb5a0c50f3047961882b7657d3b822d34ef8",
        files: &["cudart64_12.dll"],
    },
    Package {
        title: "cuBLAS 12.9",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.9.1.4-archive.zip",
        bytes: 549_755_186,
        sha256: "d534d98b0b453a98914dbf3adf47d7e84b55037abf02f87466439e1dcef581ed",
        files: &["cublas64_12.dll", "cublasLt64_12.dll"],
    },
    Package {
        title: "cuFFT 11.4",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/libcufft/windows-x86_64/libcufft-windows-x86_64-11.4.1.4-archive.zip",
        bytes: 198_361_265,
        sha256: "f26f80bb9abff3269c548e1559e8c2b4ba58ccb8acc6095bbc6404fc962d4b80",
        files: &["cufft64_11.dll"],
    },
    Package {
        title: "cuDNN 9.12",
        url: "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.12.0.46_cuda12-archive.zip",
        bytes: 550_446_499,
        sha256: "47e83e940946a13d9707cbdde6886b8750110185354e8a3d9b5e9de52e4bec2e",
        files: &[
            "cudnn64_9.dll",
            "cudnn_adv64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_graph64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_ops64_9.dll",
        ],
    },
];

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const LINUX: [Package; 6] = [
    Package {
        title: "CUDA runtime 12.9",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.9.79-archive.tar.xz",
        bytes: 1_514_676,
        sha256: "1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99",
        files: &["libcudart.so.12"],
    },
    Package {
        title: "cuBLAS 12.9",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/linux-x86_64/libcublas-linux-x86_64-12.9.1.4-archive.tar.xz",
        bytes: 933_611_504,
        sha256: "546addc4a9d82b8f23aa9ba9274b6bc0429a63008a31c759884ac24880796057",
        files: &["libcublas.so.12", "libcublasLt.so.12"],
    },
    Package {
        title: "cuFFT 11.4",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/libcufft/linux-x86_64/libcufft-linux-x86_64-11.4.1.4-archive.tar.xz",
        bytes: 470_942_192,
        sha256: "b0e65af59b0c2f6c8ed9f5552a9b375890855b7926ae2c0404d15dcf2565bda4",
        files: &["libcufft.so.11"],
    },
    Package {
        title: "cuRAND 10.3",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/libcurand/linux-x86_64/libcurand-linux-x86_64-10.3.10.19-archive.tar.xz",
        bytes: 89_582_256,
        sha256: "48281b4caadb1cf790d44ac76b23c77d06f474c0b1799814f314aafec9258ad6",
        files: &["libcurand.so.10"],
    },
    Package {
        title: "NVRTC 12.9",
        url: "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-12.9.86-archive.tar.xz",
        bytes: 114_231_976,
        sha256: "82913658363892dbc0f2638b070476234476e06e084fed60db861cb7e161a6af",
        files: &["libnvrtc.so.12", "libnvrtc-builtins.so.12.9"],
    },
    Package {
        title: "cuDNN 9.12",
        url: "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.12.0.46_cuda12-archive.tar.xz",
        bytes: 845_538_764,
        sha256: "21fad1b3a8a3db018868184b2082bd492633ae5101752432d655782bd108b19e",
        files: &[
            "libcudnn.so.9",
            "libcudnn_adv.so.9",
            "libcudnn_cnn.so.9",
            "libcudnn_engines_precompiled.so.9",
            "libcudnn_engines_runtime_compiled.so.9",
            "libcudnn_graph.so.9",
            "libcudnn_heuristic.so.9",
            "libcudnn_ops.so.9",
        ],
    },
];

/// NVIDIA's license agreements for the downloaded libraries.
pub const LICENSES: [(&str, &str); 2] = [
    (
        "CUDA Toolkit EULA",
        "https://docs.nvidia.com/cuda/eula/index.html",
    ),
    (
        "cuDNN Software License Agreement",
        "https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html",
    ),
];

/// The archives for this system; empty where there are none (macOS, ARM).
pub fn packages() -> &'static [Package] {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        &WINDOWS
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        &LINUX
    } else {
        &[]
    }
}

/// Download progress, and whether the current archive is being unpacked.
#[derive(Default)]
pub struct CudaProgress {
    pub download: Progress,
    pub unpacking: AtomicBool,
}

fn installed(folder: &Path, package: &Package) -> bool {
    package.files.iter().all(|file| folder.join(file).is_file())
}

/// Bytes still to download into `folder`.
pub fn remaining_bytes(folder: &Path) -> u64 {
    packages()
        .iter()
        .filter(|package| !installed(folder, package))
        .map(|package| package.bytes)
        .sum()
}

/// Whether the app should offer the download: the loaded ONNX Runtime has the CUDA
/// provider, an NVIDIA driver is installed, there are archives for this system, and
/// a library CUDA needs cannot be found. Checks the disk, so call it once in a while,
/// not every frame.
#[cfg(any(feature = "onnx", feature = "onnx-static"))]
pub fn needed() -> bool {
    !packages().is_empty()
        && crate::runtime::gpu_available()
        && crate::runtime::nvidia_driver_present()
        && !crate::runtime::missing_cuda_libraries().is_empty()
}

/// Downloads and unpacks every archive whose libraries `folder` does not have yet.
/// Cancelling stops at the next chunk; partial files are removed.
pub fn download(folder: &Path, progress: &CudaProgress) -> Result<()> {
    anyhow::ensure!(
        !packages().is_empty(),
        "NVIDIA publishes no CUDA libraries for this system"
    );
    let staging = folder.join(".download");
    std::fs::create_dir_all(&staging)
        .with_context(|| format!("Cannot create {}", staging.display()))?;
    let missing: Vec<&Package> = packages()
        .iter()
        .filter(|package| !installed(folder, package))
        .collect();
    progress.download.total.store(
        missing.iter().map(|package| package.bytes).sum(),
        Ordering::Relaxed,
    );
    for package in missing {
        let name = package.url.rsplit('/').next().unwrap_or("archive");
        let archive = staging.join(format!("{name}.partial"));
        let result = fetch(package, &archive, &progress.download).and_then(|()| {
            progress.unpacking.store(true, Ordering::Relaxed);
            log::info!("Unpacking {}", package.title);
            let unpacked = unpack(package, &archive, folder, &progress.download.cancel);
            progress.unpacking.store(false, Ordering::Relaxed);
            unpacked
        });
        let _ = std::fs::remove_file(&archive);
        result.with_context(|| package.title)?;
        log::info!("Installed {} in {}", package.title, folder.display());
    }
    let _ = std::fs::remove_dir(&staging);
    Ok(())
}

/// Downloads `package` to `path`, checking its size and SHA-256.
fn fetch(package: &Package, path: &Path, progress: &Progress) -> Result<()> {
    log::info!("Downloading {}", package.url);
    let response = agent()
        .get(package.url)
        .call()
        .with_context(|| format!("Could not download {}", package.url))?;
    let mut file =
        std::fs::File::create(path).with_context(|| format!("Cannot write {}", path.display()))?;
    let mut reader = response.into_reader();
    let mut buffer = vec![0; 1 << 16];
    let mut received = 0u64;
    let mut hash = Sha256::new();
    loop {
        if progress.cancel.load(Ordering::Relaxed) {
            bail!("Download cancelled");
        }
        let read = reader.read(&mut buffer).context("Download interrupted")?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])
            .context("Could not save the download; is the disk full?")?;
        hash.update(&buffer[..read]);
        received += read as u64;
        progress.received.fetch_add(read as u64, Ordering::Relaxed);
    }
    check(package, received, &hash.finalize())
}

fn check(package: &Package, received: u64, digest: &[u8]) -> Result<()> {
    if received != package.bytes {
        bail!(
            "check failed: arrived incomplete ({received} of {} bytes)",
            package.bytes
        );
    }
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    if hex != package.sha256 {
        bail!("check failed: the archive is not the expected one (SHA-256 {hex})");
    }
    Ok(())
}

/// The name `entry` (a path inside an archive) is kept under, if it is one of
/// `package`'s libraries: a file in `bin/` or `lib/` named like a wanted library,
/// or a versioned file of one.
fn wanted(package: &Package, entry: &str) -> Option<&'static str> {
    let mut parts = entry.rsplit(['/', '\\']);
    let name = parts.next()?;
    if !matches!(parts.next(), Some("bin" | "lib")) {
        return None;
    }
    package.files.iter().copied().find(|file| {
        name == *file
            || name
                .strip_prefix(file)
                .is_some_and(|rest| rest.starts_with('.'))
    })
}

/// Copies `reader` to `folder/name`, through a `.partial` file.
fn save(reader: &mut dyn Read, folder: &Path, name: &str, cancel: &AtomicBool) -> Result<()> {
    let partial = folder.join(format!("{name}.partial"));
    let written = (|| -> Result<()> {
        let mut file = std::fs::File::create(&partial)
            .with_context(|| format!("Cannot write {}", partial.display()))?;
        let mut buffer = vec![0; 1 << 16];
        loop {
            if cancel.load(Ordering::Relaxed) {
                bail!("Download cancelled");
            }
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                return Ok(());
            }
            file.write_all(&buffer[..read])
                .context("Could not save the library; is the disk full?")?;
        }
    })();
    if let Err(error) = written {
        let _ = std::fs::remove_file(&partial);
        return Err(error);
    }
    let target = folder.join(name);
    // A library already loaded by this process cannot be replaced on Windows.
    let _ = std::fs::remove_file(&target);
    std::fs::rename(&partial, &target).with_context(|| format!("Could not save {name}"))
}

/// Takes `package`'s libraries out of the checked archive at `archive`.
fn unpack(package: &Package, archive: &Path, folder: &Path, cancel: &AtomicBool) -> Result<()> {
    let mut found: Vec<&str> = Vec::new();
    if package.url.ends_with(".zip") {
        let file = std::fs::File::open(archive)?;
        let mut zip = zip::ZipArchive::new(file).context("Unreadable archive")?;
        for index in 0..zip.len() {
            let mut entry = zip.by_index(index)?;
            if !entry.is_file() {
                continue;
            }
            if let Some(name) = wanted(package, entry.name()) {
                save(&mut entry, folder, name, cancel)?;
                found.push(name);
            }
        }
    } else {
        let mut tar = tar::Archive::new(XzReader::new(archive)?);
        for entry in tar.entries().context("Unreadable archive")? {
            let mut entry = entry.context("Unreadable archive")?;
            if !entry.header().entry_type().is_file() {
                continue;
            }
            let path = entry.path()?.to_string_lossy().into_owned();
            if let Some(name) = wanted(package, &path) {
                save(&mut entry, folder, name, cancel)?;
                found.push(name);
            }
        }
    }
    let absent: Vec<&str> = package
        .files
        .iter()
        .copied()
        .filter(|file| !found.contains(file))
        .collect();
    if !absent.is_empty() {
        bail!("the archive does not contain {}", absent.join(", "));
    }
    Ok(())
}

/// Reads an `.xz` file as it is decompressed on another thread (lzma-rs only
/// writes its output).
struct XzReader {
    chunks: std::sync::mpsc::Receiver<std::io::Result<Vec<u8>>>,
    current: std::io::Cursor<Vec<u8>>,
}
impl XzReader {
    fn new(path: &Path) -> Result<Self> {
        struct Sender(std::sync::mpsc::SyncSender<std::io::Result<Vec<u8>>>);
        impl Write for Sender {
            fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
                self.0
                    .send(Ok(data.to_vec()))
                    .map_err(|_| std::io::Error::other("reader closed"))?;
                Ok(data.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let file = std::fs::File::open(path)?;
        let (tx, rx) = std::sync::mpsc::sync_channel(64);
        std::thread::Builder::new()
            .name("xz".into())
            .spawn(move || {
                let mut input = std::io::BufReader::new(file);
                // Buffered so the channel carries large chunks.
                let mut output = std::io::BufWriter::with_capacity(1 << 20, Sender(tx.clone()));
                let result = lzma_rs::xz_decompress(&mut input, &mut output)
                    .map_err(|error| std::io::Error::other(format!("{error:?}")))
                    .and_then(|()| output.flush());
                if let Err(error) = result {
                    let _ = tx.send(Err(error));
                }
            })?;
        Ok(Self {
            chunks: rx,
            current: Default::default(),
        })
    }
}
impl Read for XzReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        loop {
            let read = self.current.read(buffer)?;
            if read > 0 || buffer.is_empty() {
                return Ok(read);
            }
            match self.chunks.recv() {
                Ok(chunk) => self.current = std::io::Cursor::new(chunk?),
                // The decompressor finished.
                Err(_) => return Ok(0),
            }
        }
    }
}

/// Where the libraries go; `None` where the system has no per-user data folder.
pub fn folder() -> Option<PathBuf> {
    crate::runtime::cuda_library_folder()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUDA: &str = "https://developer.download.nvidia.com/compute/cuda/redist";
    const CUDNN: &str = "https://developer.download.nvidia.com/compute/cudnn/redist";

    #[test]
    fn packages_are_pinned_nvidia_archives_with_every_library() {
        for (packages, libraries) in [(&WINDOWS[..], ".dll"), (&LINUX[..], ".so.")] {
            let mut files: Vec<&str> = Vec::new();
            for package in packages {
                assert!(
                    package.url.starts_with(CUDA) || package.url.starts_with(CUDNN),
                    "{}",
                    package.url
                );
                assert_eq!(package.sha256.len(), 64);
                assert!(package.bytes > 0);
                files.extend(package.files);
            }
            assert!(files.iter().all(|file| file.contains(libraries)));
            files.sort();
            files.dedup();
            assert_eq!(
                files.len(),
                packages.iter().map(|p| p.files.len()).sum::<usize>()
            );
        }
        // Every library the runtime looks for is downloaded here.
        let mut needed = crate::runtime::cuda_libraries().to_vec();
        let mut offered: Vec<&str> = packages()
            .iter()
            .flat_map(|p| p.files.iter().copied())
            .collect();
        needed.sort();
        offered.sort();
        assert_eq!(needed, offered);
    }

    #[test]
    fn keeps_libraries_under_their_load_names() {
        let cublas = &LINUX[1];
        assert_eq!(
            wanted(
                cublas,
                "libcublas-linux-x86_64-12.9.1.4-archive/lib/libcublas.so.12.9.1.4"
            ),
            Some("libcublas.so.12")
        );
        assert_eq!(
            wanted(
                cublas,
                "libcublas-linux-x86_64-12.9.1.4-archive/lib/libcublasLt.so.12.9.1.4"
            ),
            Some("libcublasLt.so.12")
        );
        assert_eq!(wanted(cublas, "x/lib/libcublas_static.a"), None);
        assert_eq!(wanted(cublas, "x/include/libcublas.so.12"), None);
        let nvrtc = &LINUX[4];
        assert_eq!(
            wanted(nvrtc, "x/lib/libnvrtc-builtins.so.12.9.86"),
            Some("libnvrtc-builtins.so.12.9")
        );
        assert_eq!(
            wanted(nvrtc, "x/lib/libnvrtc.so.12.9.86"),
            Some("libnvrtc.so.12")
        );
        let cudnn = &WINDOWS[3];
        assert_eq!(
            wanted(
                cudnn,
                "cudnn-windows-x86_64-9.12.0.46_cuda12-archive/bin/cudnn64_9.dll"
            ),
            Some("cudnn64_9.dll")
        );
        assert_eq!(wanted(cudnn, "x/lib/x64/cudnn64_9.lib"), None);
        assert_eq!(wanted(&WINDOWS[1], "x/bin/nvblas64_12.dll"), None);
    }

    #[test]
    fn the_check_rejects_wrong_size_or_content() {
        let package = &WINDOWS[0];
        let right: Vec<u8> = (0..64)
            .step_by(2)
            .map(|i| u8::from_str_radix(&package.sha256[i..i + 2], 16).unwrap())
            .collect();
        assert!(check(package, package.bytes, &right).is_ok());
        assert!(check(package, package.bytes - 1, &right).is_err());
        assert!(check(package, package.bytes, &[0u8; 32]).is_err());
    }

    /// Builds a small archive like NVIDIA's and unpacks it.
    fn unpacks(package: &Package, archive: &Path, folder: &Path) {
        std::fs::create_dir_all(folder).unwrap();
        unpack(package, archive, folder, &AtomicBool::new(false)).unwrap();
        for file in package.files {
            assert_eq!(std::fs::read(folder.join(file)).unwrap(), file.as_bytes());
        }
        let stray: Vec<_> = std::fs::read_dir(folder)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name())
            .filter(|name| !package.files.iter().any(|file| name.to_str() == Some(file)))
            .collect();
        assert!(stray.is_empty(), "{stray:?}");
    }

    #[test]
    fn unpacks_only_the_libraries_from_a_zip() {
        let base = std::env::temp_dir().join(format!("vc-cuda-zip-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let package = &WINDOWS[1];
        let archive = base.join("a.zip");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&archive).unwrap());
        let options = zip::write::SimpleFileOptions::default();
        for (name, data) in [
            ("libcublas-archive/LICENSE", "license"),
            ("libcublas-archive/bin/nvblas64_12.dll", "no"),
            ("libcublas-archive/bin/cublas64_12.dll", "cublas64_12.dll"),
            (
                "libcublas-archive/bin/cublasLt64_12.dll",
                "cublasLt64_12.dll",
            ),
            ("libcublas-archive/lib/x64/cublas.lib", "no"),
        ] {
            zip.start_file(name, options).unwrap();
            zip.write_all(data.as_bytes()).unwrap();
        }
        zip.finish().unwrap();
        unpacks(package, &archive, &base.join("out"));
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn unpacks_only_the_libraries_from_a_tar_xz() {
        let base = std::env::temp_dir().join(format!("vc-cuda-xz-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let package = &LINUX[1];
        let mut tar = tar::Builder::new(Vec::new());
        let mut add = |name: &str, data: &[u8]| {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, name, data).unwrap();
        };
        add("x/LICENSE", b"license");
        add("x/lib/libcublas.so.12.9.1.4", b"libcublas.so.12");
        add("x/lib/libcublasLt.so.12.9.1.4", b"libcublasLt.so.12");
        add("x/lib/libcublas_static.a", b"no");
        let mut link = tar::Header::new_gnu();
        link.set_entry_type(tar::EntryType::Symlink);
        link.set_size(0);
        tar.append_link(&mut link, "x/lib/libcublas.so.12", "libcublas.so.12.9.1.4")
            .unwrap();
        let tar = tar.into_inner().unwrap();
        let mut xz = Vec::new();
        lzma_rs::xz_compress(&mut &tar[..], &mut xz).unwrap();
        let archive = base.join("a.tar.xz");
        std::fs::write(&archive, xz).unwrap();
        unpacks(package, &archive, &base.join("out"));
        // A library missing from the archive is an error.
        let mut short = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_size(1);
        header.set_cksum();
        short
            .append_data(&mut header, "x/lib/libcublas.so.12.9.1.4", &b"a"[..])
            .unwrap();
        let mut xz = Vec::new();
        lzma_rs::xz_compress(&mut &short.into_inner().unwrap()[..], &mut xz).unwrap();
        std::fs::write(&archive, xz).unwrap();
        let out = base.join("short");
        std::fs::create_dir_all(&out).unwrap();
        assert!(unpack(package, &archive, &out, &AtomicBool::new(false)).is_err());
        std::fs::remove_dir_all(&base).unwrap();
    }
}
