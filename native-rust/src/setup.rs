//! First-start setup: downloads the shared ContentVec and RMVPE models every voice
//! needs into `models/shared/`, from the same place the reference voice changer
//! fetches them on its first start.
use crate::catalog::{agent, Progress};
use anyhow::{bail, Context, Result};
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
}

pub const SHARED: [SharedModel; 2] = [
    SharedModel {
        file: "contentvec.onnx",
        title: "ContentVec",
        purpose: "Hears what you say, separately from how your voice sounds",
        url: "https://huggingface.co/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx",
        bytes: 378_550_151,
    },
    SharedModel {
        file: "rmvpe.onnx",
        title: "RMVPE",
        purpose: "Follows your pitch so the new voice stays in tune",
        url: "https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx",
        bytes: 362_003_174,
    },
];

/// Total size of both models.
pub fn total_bytes() -> u64 {
    SHARED.iter().map(|model| model.bytes).sum()
}

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

/// Downloads the models missing from `root/shared`. `progress` counts both files
/// together; cancelling stops at the next chunk and removes the partial file.
pub fn download(root: &Path, progress: &Progress) -> Result<()> {
    let folder = root.join("shared");
    std::fs::create_dir_all(&folder)
        .with_context(|| format!("Cannot create {}", folder.display()))?;
    let missing: Vec<&SharedModel> = SHARED
        .iter()
        .filter(|model| !folder.join(model.file).is_file())
        .collect();
    progress.total.store(
        missing.iter().map(|model| model.bytes).sum(),
        Ordering::Relaxed,
    );
    for model in missing {
        fetch(model, &folder, progress)?;
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
        received += read as u64;
        progress.received.fetch_add(read as u64, Ordering::Relaxed);
    };
    drop(file);
    if let Err(error) = outcome.and_then(|()| {
        if received != model.bytes {
            bail!(
                "{} arrived incomplete ({received} of {} bytes)",
                model.title,
                model.bytes
            );
        }
        Ok(())
    }) {
        let _ = std::fs::remove_file(&partial);
        return Err(error);
    }
    std::fs::rename(&partial, folder.join(model.file))
        .with_context(|| format!("Could not save {}", model.file))?;
    log::info!("Downloaded {}", model.file);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const WEIGHTS: &str = "https://huggingface.co/wok000/weights_gpl/resolve/main";

    #[test]
    fn urls_come_from_the_reference_weights() {
        for model in &SHARED {
            assert!(model.url.starts_with(WEIGHTS), "{}", model.url);
            assert!(model.file.ends_with(".onnx"));
        }
        assert_eq!(total_bytes(), 740_553_325);
    }
}
