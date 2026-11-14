//! Voice cloning datasets: recordings of one speaker kept as WAV takes in
//! `datasets/<name>/`, checked for length and level, and packed into a ZIP that the
//! Colab notebook (`colab/train_voice.ipynb`) trains an RVC v2 voice from.
//!
//! Each take is trimmed of leading and trailing silence and its peak set to -1 dBFS
//! when it is saved; slicing into short clips is left to the trainer (Applio's
//! preprocess), which also filters and resamples.
use anyhow::{bail, ensure, Context, Result};
use std::path::{Path, PathBuf};

/// Less than this is refused when packing: the voice would not be usable.
pub const MINIMUM_SECONDS: f64 = 60.0;
/// Enough for a good voice; more helps up to about half an hour.
pub const RECOMMENDED_SECONDS: f64 = 600.0;
/// The file in the ZIP that tells the notebook the voice's name.
pub const MANIFEST: &str = "voice-changer-dataset.json";
/// The notebook that trains a voice from a packed dataset, at this app's version.
pub fn colab_url() -> String {
    format!(
        "https://colab.research.google.com/github/Araryarch67/voice-changer/blob/v{}/native-rust/colab/train_voice.ipynb",
        env!("CARGO_PKG_VERSION")
    )
}

/// Peak a take is normalized to (-1 dBFS).
const PEAK: f32 = 0.891;
/// A quieter take is not boosted more than this (+24 dB): it would raise the noise.
const MAX_GAIN: f32 = 15.85;
/// Frames louder than this (-45 dBFS) count as voice when trimming.
const VOICE_RMS: f32 = 0.005_6;

/// The folder datasets are kept in: beside the models folder.
pub fn root() -> PathBuf {
    crate::runtime::model_directories()
        .into_iter()
        .find(|directory| directory.is_dir())
        .and_then(|models| models.parent().map(Path::to_path_buf))
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_default()
        .join("datasets")
}

/// `name` as a folder and model name: letters, digits, `-` and `_`, other characters
/// turned into `_`. Empty when nothing is left.
pub fn safe_name(name: &str) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    cleaned.trim_matches('_').chars().take(48).collect()
}

/// One saved recording.
#[derive(Clone, Debug, PartialEq)]
pub struct Take {
    pub path: PathBuf,
    pub seconds: f64,
}

/// What saving a take found about it.
#[derive(Clone, Debug, PartialEq)]
pub struct TakeReport {
    pub take: Take,
    /// The input reached full scale: the microphone gain is too high.
    pub clipped: bool,
    /// The voice was very quiet: the microphone gain is too low or it is too far.
    pub quiet: bool,
}

/// The takes in `folder`, in the order they were recorded.
pub fn takes(folder: &Path) -> Vec<Take> {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return Vec::new();
    };
    let mut takes: Vec<Take> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("wav"))
        })
        .filter_map(|path| {
            let reader = hound::WavReader::open(&path).ok()?;
            let spec = reader.spec();
            let seconds = reader.duration() as f64 / spec.sample_rate.max(1) as f64;
            Some(Take { path, seconds })
        })
        .collect();
    takes.sort_by(|a, b| a.path.cmp(&b.path));
    takes
}

/// Total length of `takes` in seconds.
pub fn total_seconds(takes: &[Take]) -> f64 {
    takes.iter().map(|take| take.seconds).sum()
}

/// The next free `take-NNN.wav` in `folder`.
fn next_take(folder: &Path) -> PathBuf {
    (1..)
        .map(|n| folder.join(format!("take-{n:03}.wav")))
        .find(|path| !path.exists())
        .expect("a free take name")
}

/// Trims silence and normalizes `samples` (mono, `rate` Hz), then saves them as the
/// next take in `folder`. Fails when there is under a second of voice.
pub fn save_take(folder: &Path, rate: u32, samples: &[f32]) -> Result<TakeReport> {
    ensure!(rate > 0, "Invalid sample rate");
    let peak_in = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let clipped = samples.iter().filter(|s| s.abs() >= 0.999).count() > rate as usize / 1000;
    let voiced = trim(samples, rate);
    ensure!(
        voiced.len() >= rate as usize,
        "No voice was heard; check the microphone and speak closer to it"
    );
    let gain = if peak_in > 0.0 {
        (PEAK / peak_in).min(MAX_GAIN)
    } else {
        1.0
    };
    let quiet = peak_in < 0.05;
    std::fs::create_dir_all(folder)
        .with_context(|| format!("Cannot create {}", folder.display()))?;
    let path = next_take(folder);
    write_wav(&path, rate, voiced.iter().map(|s| s * gain))?;
    log::info!("Saved {}", path.display());
    Ok(TakeReport {
        take: Take {
            seconds: voiced.len() as f64 / rate as f64,
            path,
        },
        clipped,
        quiet,
    })
}

/// Adds an existing WAV recording of the speaker to `folder` as a take.
pub fn add_file(folder: &Path, source: &Path) -> Result<TakeReport> {
    let (rate, samples) = crate::wav::read_mono(source)?;
    save_take(folder, rate, &samples)
}

/// `samples` without the silence before the first and after the last voiced frame,
/// keeping a quarter second around the voice.
fn trim(samples: &[f32], rate: u32) -> &[f32] {
    let frame = (rate as usize / 50).max(1);
    let loud = |chunk: &[f32]| {
        (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32).sqrt() > VOICE_RMS
    };
    let frames: Vec<bool> = samples.chunks(frame).map(loud).collect();
    let (Some(first), Some(last)) = (
        frames.iter().position(|&v| v),
        frames.iter().rposition(|&v| v),
    ) else {
        return &[];
    };
    let pad = rate as usize / 4;
    let start = (first * frame).saturating_sub(pad);
    let end = ((last + 1) * frame + pad).min(samples.len());
    &samples[start..end]
}

fn write_wav(path: &Path, rate: u32, samples: impl Iterator<Item = f32>) -> Result<()> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .with_context(|| format!("Cannot write {}", path.display()))?;
    for sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Packs the takes in `folder` into `output` for the notebook: the WAVs under
/// `dataset/` and [`MANIFEST`] with the voice's `name`. Returns the length packed.
#[cfg(feature = "export")]
pub fn pack(folder: &Path, name: &str, output: &Path) -> Result<f64> {
    use std::io::Write;
    let name = safe_name(name);
    ensure!(!name.is_empty(), "Give the voice a name");
    let takes = takes(folder);
    let seconds = total_seconds(&takes);
    if seconds < MINIMUM_SECONDS {
        bail!(
            "Only {:.0} seconds recorded; record at least {:.0} (10 minutes is recommended)",
            seconds,
            MINIMUM_SECONDS
        );
    }
    let partial = output.with_extension("zip.partial");
    let written = (|| -> Result<()> {
        let file = std::fs::File::create(&partial)
            .with_context(|| format!("Cannot write {}", partial.display()))?;
        let mut zip = zip::ZipWriter::new(file);
        // WAV barely compresses; storing keeps packing fast.
        let stored = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        let manifest = serde_json::json!({
            "name": name,
            "seconds": seconds.round(),
            "takes": takes.len(),
            "app": env!("CARGO_PKG_VERSION"),
        });
        zip.start_file(MANIFEST, zip::write::SimpleFileOptions::default())?;
        zip.write_all(serde_json::to_string_pretty(&manifest)?.as_bytes())?;
        for take in &takes {
            let file_name = take
                .path
                .file_name()
                .context("Take without a name")?
                .to_string_lossy();
            zip.start_file(format!("dataset/{file_name}"), stored)?;
            let mut source = std::fs::File::open(&take.path)?;
            std::io::copy(&mut source, &mut zip)?;
        }
        zip.finish()?;
        Ok(())
    })();
    if let Err(error) = written {
        let _ = std::fs::remove_file(&partial);
        return Err(error);
    }
    let _ = std::fs::remove_file(output);
    std::fs::rename(&partial, output)
        .with_context(|| format!("Cannot write {}", output.display()))?;
    log::info!("Packed {} ({seconds:.0} s)", output.display());
    Ok(seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, seconds: f32, amplitude: f32) -> Vec<f32> {
        (0..(rate as f32 * seconds) as usize)
            .map(|i| amplitude * (i as f32 * 220.0 * std::f32::consts::TAU / rate as f32).sin())
            .collect()
    }

    #[test]
    fn names_are_safe_for_folders() {
        assert_eq!(safe_name("  My Voice! "), "My_Voice");
        assert_eq!(safe_name("suara-ku_2"), "suara-ku_2");
        assert_eq!(safe_name("../.."), "");
    }

    #[test]
    fn takes_are_trimmed_normalized_and_packed() {
        let folder = std::env::temp_dir().join(format!("vc-dataset-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&folder);
        let rate = 16_000;
        let mut samples = vec![0.0; rate as usize * 2];
        samples.extend(tone(rate, 3.0, 0.2));
        samples.extend(vec![0.0; rate as usize * 2]);
        let report = save_take(&folder, rate, &samples).unwrap();
        assert!(!report.clipped && !report.quiet);
        // Three seconds of voice and a quarter second either side.
        assert!(
            (report.take.seconds - 3.5).abs() < 0.05,
            "{}",
            report.take.seconds
        );
        let (_, saved) = crate::wav::read_mono(&report.take.path).unwrap();
        let peak = saved.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!((peak - PEAK).abs() < 0.01, "{peak}");

        assert!(save_take(&folder, rate, &vec![0.0; rate as usize * 3]).is_err());
        let loud = save_take(&folder, rate, &tone(rate, 2.0, 1.0)).unwrap();
        assert!(loud.clipped);
        assert_eq!(takes(&folder).len(), 2);
        assert!(loud.take.path.ends_with("take-002.wav"));

        #[cfg(feature = "export")]
        {
            let output = folder.join("voice.zip");
            assert!(pack(&folder, "voice", &output).is_err(), "too short");
            for _ in 0..12 {
                save_take(&folder, rate, &tone(rate, 5.0, 0.3)).unwrap();
            }
            let seconds = pack(&folder, "My Voice", &output).unwrap();
            assert!(seconds >= MINIMUM_SECONDS);
            let mut zip = zip::ZipArchive::new(std::fs::File::open(&output).unwrap()).unwrap();
            assert_eq!(zip.len(), 15);
            let manifest: serde_json::Value =
                serde_json::from_reader(zip.by_name(MANIFEST).unwrap()).unwrap();
            assert_eq!(manifest["name"], "My_Voice");
            assert!(zip.by_name("dataset/take-001.wav").is_ok());
        }
        let _ = std::fs::remove_dir_all(&folder);
    }
}
