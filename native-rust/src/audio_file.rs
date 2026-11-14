//! Reads songs and writes stems: any of WAV, MP3, FLAC, OGG Vorbis and M4A/AAC in,
//! WAV out. Used by vocal separation and covers, which take whole songs.
use anyhow::{bail, Context, Result};
use std::path::Path;
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, errors::Error as DecodeError,
    formats::FormatOptions, io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
};

/// File extensions [`read_stereo`] opens, for file pickers.
pub const EXTENSIONS: [&str; 6] = ["wav", "mp3", "flac", "ogg", "m4a", "aac"];

/// Stereo audio: two channels of equal length at `rate`.
#[derive(Debug, Clone, PartialEq)]
pub struct Stereo {
    pub rate: u32,
    pub channels: [Vec<f32>; 2],
}
impl Stereo {
    pub fn len(&self) -> usize {
        self.channels[0].len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The largest absolute sample of either channel.
    pub fn peak(&self) -> f32 {
        self.channels
            .iter()
            .flatten()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()))
    }
    pub fn seconds(&self) -> f64 {
        self.len() as f64 / self.rate.max(1) as f64
    }
}

/// Decodes an audio file into stereo `f32` samples. Mono files are copied to both
/// channels; extra channels past the first two are dropped.
pub fn read_stereo(path: &Path) -> Result<Stereo> {
    let file = std::fs::File::open(path).with_context(|| format!("Open {}", path.display()))?;
    let source = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("{} is not an audio file this app can read", path.display()))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|track| track.codec_params.sample_rate.is_some())
        .with_context(|| format!("No audio track in {}", path.display()))?;
    let track_id = track.id;
    let mut rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .with_context(|| format!("Cannot decode the audio in {}", path.display()))?;
    let (mut left, mut right) = (Vec::new(), Vec::new());
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(DecodeError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(DecodeError::ResetRequired) => break,
            Err(error) => return Err(error).context("Read the audio"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A damaged frame is skipped, as players do.
            Err(DecodeError::DecodeError(_)) => continue,
            Err(error) => return Err(error).context("Decode the audio"),
        };
        let spec = *decoded.spec();
        rate = spec.rate;
        let count = spec.channels.count();
        if count == 0 {
            continue;
        }
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buffer.copy_interleaved_ref(decoded);
        for frame in buffer.samples().chunks_exact(count) {
            left.push(frame[0]);
            right.push(if count > 1 { frame[1] } else { frame[0] });
        }
    }
    if left.is_empty() || rate == 0 {
        bail!("{} holds no audio", path.display());
    }
    Ok(Stereo {
        rate,
        channels: [left, right],
    })
}

/// Writes 32-bit float stereo WAV, which keeps samples above full scale.
pub fn write_float(path: &Path, audio: &Stereo) -> Result<()> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 2,
            sample_rate: audio.rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .with_context(|| format!("Create {}", path.display()))?;
    for (left, right) in audio.channels[0].iter().zip(&audio.channels[1]) {
        writer.write_sample(*left)?;
        writer.write_sample(*right)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Writes 16-bit stereo WAV for sharing, turned down as a whole when it would clip.
pub fn write_pcm16(path: &Path, audio: &Stereo) -> Result<()> {
    let peak = audio.peak();
    let gain = if peak > 0.99 { 0.99 / peak } else { 1.0 };
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 2,
            sample_rate: audio.rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .with_context(|| format!("Create {}", path.display()))?;
    let pcm = |sample: f32| (sample * gain * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
    for (left, right) in audio.channels[0].iter().zip(&audio.channels[1]) {
        writer.write_sample(pcm(*left))?;
        writer.write_sample(pcm(*right))?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trips_and_mono_fills_both_channels() {
        let folder = std::env::temp_dir().join(format!("vc-audio-file-{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let stereo = Stereo {
            rate: 22050,
            channels: [vec![0.5, -0.25, 1.5], vec![0.0, 0.125, -1.5]],
        };
        let float = folder.join("float.wav");
        write_float(&float, &stereo).unwrap();
        assert_eq!(read_stereo(&float).unwrap(), stereo);

        // 16-bit output is turned down to fit, keeping the balance.
        let pcm = folder.join("pcm.wav");
        write_pcm16(&pcm, &stereo).unwrap();
        let read = read_stereo(&pcm).unwrap();
        assert!((read.peak() - 0.99).abs() < 1e-3, "{}", read.peak());
        assert!((read.channels[0][0] / read.channels[0][2] - 1.0 / 3.0).abs() < 1e-3);

        let mono = folder.join("mono.wav");
        let mut writer = hound::WavWriter::create(
            &mono,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for sample in [0_i16, 16384, -16384] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        let read = read_stereo(&mono).unwrap();
        assert_eq!(read.rate, 16000);
        assert_eq!(read.channels[0], read.channels[1]);
        assert!((read.channels[0][1] - 0.5).abs() < 1e-4);
        let _ = std::fs::remove_dir_all(&folder);
    }

    #[test]
    fn a_file_that_is_not_audio_is_refused() {
        let file = std::env::temp_dir().join(format!("vc-not-audio-{}.mp3", std::process::id()));
        std::fs::write(&file, b"not audio at all").unwrap();
        assert!(read_stereo(&file).is_err());
        let _ = std::fs::remove_file(&file);
    }
}
