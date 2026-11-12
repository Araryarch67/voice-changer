//! Turns an RVC checkpoint (`.pth`), or a zip holding one, into a ready voice folder:
//! ONNX export, the matching `.index` if there is one, and the `model.json` manifest.
use crate::{config::ModelManifest, export, runtime};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Imports `source` into the models folder and returns the new manifest's path.
pub fn import_voice(source: &Path) -> Result<PathBuf> {
    let root = runtime::models_root().context(
        "No models folder with shared/contentvec.onnx and shared/rmvpe.onnx was found; add those two files first",
    )?;
    import_into(&root, source)
}

fn import_into(root: &Path, source: &Path) -> Result<PathBuf> {
    let extension = source
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    match extension.as_str() {
        "pth" => {
            let index = source.parent().and_then(find_index);
            import_checkpoint(root, source, index.as_deref(), false)
        }
        "zip" => {
            let unpacked = root.join(format!(".import-{}", std::process::id()));
            let result = unpack(source, &unpacked).and_then(|checkpoint| {
                let index = find_index(&unpacked);
                import_checkpoint(root, &checkpoint, index.as_deref(), true)
            });
            let _ = std::fs::remove_dir_all(&unpacked);
            result
        }
        _ => bail!(
            "Choose an RVC .pth checkpoint, a .zip that contains one, or a model.json manifest"
        ),
    }
}

/// The training index beside a checkpoint: the `added_*.index` file if there is one,
/// otherwise the only `.index` file.
fn find_index(directory: &Path) -> Option<PathBuf> {
    let mut indexes: Vec<PathBuf> = std::fs::read_dir(directory)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|e| e == "index"))
        .collect();
    indexes.sort();
    let added = indexes.iter().position(|path| {
        path.file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("added_"))
    });
    match added {
        Some(found) => Some(indexes.swap_remove(found)),
        None if indexes.len() == 1 => indexes.pop(),
        None => None,
    }
}

/// Extracts the checkpoint and any index from an archive into `target`, flattened.
/// Only `.pth` and `.index` entries are written, under their bare file names.
fn unpack(archive: &Path, target: &Path) -> Result<PathBuf> {
    let file =
        std::fs::File::open(archive).with_context(|| format!("Open {}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(file).context("Read zip archive")?;
    std::fs::create_dir_all(target)?;
    let mut checkpoints = Vec::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let Some(name) = entry.enclosed_name().and_then(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        }) else {
            continue;
        };
        let lower = name.to_lowercase();
        // G_*/D_* are training states, not voices.
        let training = lower.starts_with("g_") || lower.starts_with("d_");
        if entry.is_dir() || !(lower.ends_with(".index") || lower.ends_with(".pth") && !training) {
            continue;
        }
        let path = target.join(&name);
        std::io::copy(&mut entry, &mut std::fs::File::create(&path)?)
            .with_context(|| format!("Extract {name}"))?;
        if lower.ends_with(".pth") {
            checkpoints.push(path);
        }
    }
    checkpoints.sort();
    match checkpoints.len() {
        0 => bail!("The archive contains no RVC .pth checkpoint"),
        1 => Ok(checkpoints.remove(0)),
        count => bail!(
            "The archive contains {count} checkpoints; extract it and import the one you want"
        ),
    }
}

fn import_checkpoint(
    root: &Path,
    checkpoint: &Path,
    index: Option<&Path>,
    take_index: bool,
) -> Result<PathBuf> {
    let stem = checkpoint
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let slug: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let slug = slug.trim_matches('-');
    let slug = if slug.is_empty() { "voice" } else { slug };
    let folder = (1..)
        .map(|n| {
            root.join(if n == 1 {
                slug.to_owned()
            } else {
                format!("{slug}-{n}")
            })
        })
        .find(|folder| !folder.exists())
        .expect("an unused folder name exists");
    std::fs::create_dir_all(&folder)?;
    let result = build(&folder, checkpoint, index, take_index, &stem);
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&folder);
    }
    result
}

fn build(
    folder: &Path,
    checkpoint: &Path,
    index: Option<&Path>,
    take_index: bool,
    stem: &str,
) -> Result<PathBuf> {
    let exported = export::export_rvc(checkpoint, &folder.join("voice-realtime.onnx"), true)?;
    let feature_bank = match index {
        Some(index) => {
            let name = index.file_name().context("Index has no file name")?;
            // An index unpacked from an archive is moved; one beside a checkpoint is
            // copied, so the voice folder stands on its own.
            if !take_index || std::fs::rename(index, folder.join(name)).is_err() {
                std::fs::copy(index, folder.join(name)).context("Copy index")?;
            }
            Some(PathBuf::from(name))
        }
        None => None,
    };
    let manifest = ModelManifest {
        name: stem.replace(['_', '-'], " "),
        model: "voice-realtime.onnx".into(),
        embedder: "../shared/contentvec.onnx".into(),
        pitch: "../shared/rmvpe.onnx".into(),
        sample_rate: exported.sample_rate,
        channels: exported.channels,
        use_f0: exported.f0,
        embedder_input: "audio".into(),
        // v1 voices use the projected layer-9 features, v2 the layer-12 ones.
        embedder_output: if exported.channels == 256 {
            "units9"
        } else {
            "unit12"
        }
        .into(),
        output_scale: 1.0,
        feature_bank,
        intra_threads: std::thread::available_parallelism().map_or(4, |n| n.get().min(8)),
    };
    let path = folder.join("model.json");
    std::fs::write(&path, serde_json::to_string_pretty(&manifest)?)?;
    ModelManifest::load(&path).context("The imported voice did not pass validation")?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_added_index_or_the_only_one() {
        let directory =
            std::env::temp_dir().join(format!("vc-import-index-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        assert_eq!(find_index(&directory), None);
        std::fs::write(directory.join("trained_IVF1_Flat.index"), b"").unwrap();
        assert_eq!(
            find_index(&directory),
            Some(directory.join("trained_IVF1_Flat.index"))
        );
        std::fs::write(directory.join("other.index"), b"").unwrap();
        assert_eq!(find_index(&directory), None);
        std::fs::write(directory.join("added_IVF1_Flat.index"), b"").unwrap();
        assert_eq!(
            find_index(&directory),
            Some(directory.join("added_IVF1_Flat.index"))
        );
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn rejects_other_files_and_empty_archives() {
        let root = std::env::temp_dir().join(format!("vc-import-reject-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        assert!(import_into(&root, Path::new("voice.wav")).is_err());
        let archive = root.join("empty.zip");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&archive).unwrap());
        zip.start_file("readme.txt", zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.start_file("G_2333.pth", zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.finish().unwrap();
        let error = import_into(&root, &archive).unwrap_err().to_string();
        assert!(error.contains("no RVC .pth"), "{error}");
        // Nothing is left behind.
        let left: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(left, vec![std::ffi::OsString::from("empty.zip")]);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
