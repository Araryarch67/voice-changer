//! Voice engine setup: downloads the shared models into `models/shared/`, from the
//! same place the reference voice changer fetches them. ContentVec and RMVPE are
//! required by every voice; the CREPE pitch detectors are optional. Each download
//! is checked against its size and SHA-256 before it is kept.
use crate::catalog::{agent, Progress};
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::Ordering,
};

pub struct SharedModel {
    pub file: &'static str,
    pub title: &'static str,
    /// What it does, for people who do not know the term.
    pub purpose: &'static str,
    pub url: &'static str,
    pub bytes: u64,
    /// SHA-256 of the file, as Hugging Face lists it.
    pub sha256: &'static str,
    /// Every voice needs it.
    pub required: bool,
    /// Offered ticked: worth having for most people.
    pub recommended: bool,
}

pub const SHARED: [SharedModel; 4] = [
    SharedModel {
        file: "contentvec.onnx",
        title: "ContentVec",
        purpose: "Hears what you say, separately from how your voice sounds",
        url: "https://huggingface.co/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx",
        bytes: 378_550_151,
        sha256: "4b31ed3d95a568fab7952de923ff7f7d3d17128ea6fce69f665509d24c3156db",
        required: true,
        recommended: true,
    },
    SharedModel {
        file: "rmvpe.onnx",
        title: "RMVPE",
        purpose: "Follows your pitch so the new voice stays in tune",
        url: "https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx",
        bytes: 362_003_174,
        sha256: "84f0586308e36157f75b77c8591bf636d6719c0c4ba95f8faf3df479e7566219",
        required: true,
        recommended: true,
    },
    SharedModel {
        file: "crepe_tiny.onnx",
        title: "CREPE tiny",
        purpose:
            "A light pitch detector for slower computers; pick it as the F0 detector in Studio",
        url: "https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/tiny.onnx",
        bytes: 1_955_762,
        sha256: "91fc2a0fd10f965dbf7775995daf50e99273caedd7efd00001f23be649da1bc3",
        required: false,
        recommended: true,
    },
    SharedModel {
        file: "crepe_full.onnx",
        title: "CREPE full",
        purpose: "A precise pitch detector for singing; heavy, best with a GPU",
        url: "https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/full.onnx",
        bytes: 88_984_790,
        sha256: "119845c72c702e052e5262430f9d120bce46176689aa226c39d09dea5cc3a610",
        required: false,
        recommended: false,
    },
];

/// Where the models are downloaded to: the working directory's `models/` when it
/// exists (the Linux packages start in a writable data folder, the Windows shortcut
/// in the install folder), otherwise `models/` beside the executable.
pub fn target_root() -> PathBuf {
    let cwd = std::env::current_dir().ok().map(|cwd| cwd.join("models"));
    if let Some(folder) = cwd.as_ref().filter(|folder| folder.is_dir()) {
        return folder.clone();
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|parent| parent.join("models")))
        .or(cwd)
        .unwrap_or_else(|| PathBuf::from("models"))
}

/// Whether `model` is already in `folder`.
pub fn installed(folder: &Path, model: &SharedModel) -> bool {
    folder.join(model.file).is_file()
}

/// Downloads the chosen models that `root/shared` does not have yet. `progress`
/// counts them together; cancelling stops at the next chunk and removes the partial
/// file. A model that fails its check is not kept, and the others still download;
/// the error names every one that failed.
pub fn download(root: &Path, chosen: &[&'static str], progress: &Progress) -> Result<()> {
    let folder = root.join("shared");
    std::fs::create_dir_all(&folder)
        .with_context(|| format!("Cannot create {}", folder.display()))?;
    let missing: Vec<&SharedModel> = SHARED
        .iter()
        .filter(|model| chosen.contains(&model.file) && !installed(&folder, model))
        .collect();
    progress.total.store(
        missing.iter().map(|model| model.bytes).sum(),
        Ordering::Relaxed,
    );
    let mut failed = Vec::new();
    for model in missing {
        if let Err(error) = fetch(model, &folder, progress) {
            if progress.cancel.load(Ordering::Relaxed) {
                return Err(error);
            }
            log::error!("{}: {error:#}", model.title);
            failed.push(format!("{}: {error:#}", model.title));
        }
    }
    if !failed.is_empty() {
        bail!("{}", failed.join("; "));
    }
    Ok(())
}

fn fetch(model: &SharedModel, folder: &Path, progress: &Progress) -> Result<()> {
    log::info!("Downloading {} from {}", model.file, model.url);
    let response = agent()
        .get(model.url)
        .call()
        .with_context(|| format!("Could not download {}", model.title))?;
    let partial = folder.join(format!("{}.part", model.file));
    let mut file = std::fs::File::create(&partial)
        .with_context(|| format!("Cannot write {}", partial.display()))?;
    let mut reader = response.into_reader();
    let mut buffer = vec![0; 1 << 16];
    let mut received = 0u64;
    let mut hash = Sha256::new();
    let outcome = loop {
        if progress.cancel.load(Ordering::Relaxed) {
            break Err(anyhow::anyhow!("Download cancelled"));
        }
        let read = match reader.read(&mut buffer) {
            Ok(0) => break Ok(()),
            Ok(read) => read,
            Err(error) => {
                break Err(anyhow::Error::new(error).context("Download interrupted"));
            }
        };
        if let Err(error) = file.write_all(&buffer[..read]) {
            break Err(anyhow::Error::new(error).context("Could not save the download"));
        }
        hash.update(&buffer[..read]);
        received += read as u64;
        progress.received.fetch_add(read as u64, Ordering::Relaxed);
    };
    drop(file);
    let checked = outcome.and_then(|()| check(model, received, &hash.finalize()));
    if let Err(error) = checked {
        let _ = std::fs::remove_file(&partial);
        return Err(error);
    }
    std::fs::rename(&partial, folder.join(model.file))
        .with_context(|| format!("Could not save {}", model.file))?;
    log::info!("Downloaded and checked {}", model.file);
    Ok(())
}

/// The model check: the expected size and SHA-256.
fn check(model: &SharedModel, received: u64, digest: &[u8]) -> Result<()> {
    if received != model.bytes {
        bail!(
            "check failed: arrived incomplete ({received} of {} bytes)",
            model.bytes
        );
    }
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    if hex != model.sha256 {
        bail!("check failed: the file is not the expected model (SHA-256 {hex})");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_come_from_the_reference_weights() {
        for model in &SHARED {
            assert!(
                model.url.starts_with("https://huggingface.co/wok000/"),
                "{}",
                model.url
            );
            assert!(model.file.ends_with(".onnx"));
            assert_eq!(model.sha256.len(), 64);
            assert!(!model.required || model.recommended);
        }
        // runtime::models_root and rvc.rs look for these names.
        assert_eq!(SHARED[0].file, "contentvec.onnx");
        assert_eq!(SHARED[1].file, "rmvpe.onnx");
        assert!(SHARED.iter().any(|m| m.file == "crepe_full.onnx"));
        assert!(SHARED.iter().any(|m| m.file == "crepe_tiny.onnx"));
    }

    #[test]
    fn the_check_rejects_wrong_size_or_content() {
        let model = &SHARED[2];
        let right = hex_to_bytes(model.sha256);
        assert!(check(model, model.bytes, &right).is_ok());
        assert!(check(model, model.bytes - 1, &right).is_err());
        assert!(check(model, model.bytes, &[0u8; 32]).is_err());
    }

    fn hex_to_bytes(hex: &str) -> Vec<u8> {
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect()
    }
}
