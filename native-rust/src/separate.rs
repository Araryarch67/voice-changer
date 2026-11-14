//! Vocal separation with an MDX-Net model (Kim_Vocal_2), done as Ultimate Vocal
//! Remover and AICoverGen do it: the song at 44.1 kHz, peak-normalised, cut into
//! overlapping chunks, each chunk's spectrum (STFT) run through the model, which
//! returns the vocals' spectrum, and turned back into audio. The instrumental is the
//! song minus the vocals times the model's compensation.
use crate::{audio_file::Stereo, config::Device};
use anyhow::{Context, Result};
use ort::value::Tensor;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
};

/// The model file, in the shared models folder.
pub const MODEL_FILE: &str = "kim_vocal_2.onnx";

/// Sample rate the model works at.
pub const RATE: u32 = 44_100;
/// Kim_Vocal_2's settings in UVR's model list (`mdx_model_data`): `n_fft` 7680,
/// `dim_f` 3072, `dim_t` 2^8, `compensate` 1.009; UVR's hop is 1024.
const N_FFT: usize = 7680;
const HOP: usize = 1024;
const DIM_F: usize = 3072;
const DIM_T: usize = 256;
const COMPENSATE: f32 = 1.009;
const N_BINS: usize = N_FFT / 2 + 1;
/// Samples per chunk the model sees, and the edge of each chunk that is discarded.
const CHUNK: usize = HOP * (DIM_T - 1);
const TRIM: usize = N_FFT / 2;
const STEP: usize = CHUNK - 2 * TRIM;

/// How far a separation or a cover has got, for a progress bar: the stage, and
/// how much of it is done. `cancel` stops it.
#[derive(Default)]
pub struct Progress {
    pub stage: AtomicUsize,
    pub done: AtomicUsize,
    pub total: AtomicUsize,
    pub cancel: AtomicBool,
}
/// [`Progress::stage`] values.
pub const SEPARATING: usize = 0;
pub const CONVERTING: usize = 1;
pub const MIXING: usize = 2;
pub const MEASURING: usize = 3;
impl Progress {
    fn begin(&self, stage: usize, total: usize) {
        self.stage.store(stage, Ordering::Relaxed);
        self.total.store(total, Ordering::Relaxed);
        self.done.store(0, Ordering::Relaxed);
    }
    pub fn fraction(&self) -> f32 {
        let total = self.total.load(Ordering::Relaxed).max(1);
        (self.done.load(Ordering::Relaxed) as f32 / total as f32).min(1.0)
    }
}

/// The installed vocal separation model, from the shared models folder.
pub fn model_path() -> Option<std::path::PathBuf> {
    crate::runtime::model_directories()
        .into_iter()
        .map(|folder| folder.join("shared").join(MODEL_FILE))
        .find(|file| file.is_file())
}

/// The model, or an error that says how to get it.
pub fn require_model() -> Result<std::path::PathBuf> {
    model_path().with_context(|| {
        format!("The vocal separation model ({MODEL_FILE}) is not installed; download Kim Vocal 2 in Settings under Processing (Manage models), or run `voice-changer-native setup download kim_vocal_2`")
    })
}

/// Where a split writes its stems: `<song>-vocals.wav` and `<song>-instrumental.wav`
/// in `folder` (beside the song when `None`).
pub fn stem_paths(song: &Path, folder: Option<&Path>) -> (std::path::PathBuf, std::path::PathBuf) {
    let stem = song.file_stem().unwrap_or_default().to_string_lossy();
    let folder = folder
        .map(Path::to_path_buf)
        .or_else(|| song.parent().map(Path::to_path_buf))
        .unwrap_or_default();
    (
        folder.join(format!("{stem}-vocals.wav")),
        folder.join(format!("{stem}-instrumental.wav")),
    )
}

/// Splits the song at `input` into the two WAV files `vocals` and `instrumental`
/// (32-bit float, 44.1 kHz stereo). Returns the song's length in seconds.
pub fn split_file(
    model: &Path,
    device: Device,
    input: &Path,
    vocals: &Path,
    instrumental: &Path,
    progress: &Progress,
) -> Result<f64> {
    anyhow::ensure!(
        vocals != input && instrumental != input,
        "The stems must not overwrite the song"
    );
    let song = crate::audio_file::read_stereo(input)?;
    let stems = separate(model, device, &song, progress)?;
    progress.begin(MIXING, 1);
    crate::audio_file::write_float(vocals, &stems.vocals)?;
    crate::audio_file::write_float(instrumental, &stems.instrumental)?;
    progress.done.store(1, Ordering::Relaxed);
    Ok(song.seconds())
}

/// What a cover is made with.
pub struct CoverJob<'a> {
    pub separation_model: &'a Path,
    pub voice: &'a Path,
    pub settings: &'a crate::config::Settings,
    pub input: &'a Path,
    pub output: &'a Path,
    /// Loudness of the new voice and of the music in the mix (1 keeps it).
    pub vocal_gain: f32,
    pub music_gain: f32,
    /// Also write the vocals, the instrumental and the converted vocals beside the
    /// output.
    pub keep_stems: bool,
    /// Choose Tune from the song's own pitch for a voice in this register instead
    /// of using `settings.semitones`.
    pub auto_tune: Option<Register>,
}

pub use crate::config::Register;

impl Register {
    /// The middle of a melody sung comfortably in this register: E4 and E3.
    fn centre_hz(self) -> f32 {
        match self {
            Register::Female => 330.0,
            Register::Male => 165.0,
        }
    }

    /// Above this (A5 and A4) high notes start to thin out or break up in voices
    /// trained on this register.
    fn ceiling_hz(self) -> f32 {
        match self {
            Register::Female => 880.0,
            Register::Male => 440.0,
        }
    }
}

/// How high a song's vocals are sung.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SongPitch {
    /// The middle of the sung notes.
    pub median_hz: f32,
    /// The high notes: 95 % of sung frames are at or below this.
    pub high_hz: f32,
}

/// The pitch of separated vocals (`rate` Hz, mono): each 10 ms frame that is sung,
/// that is voiced and within 30 dB of the loudest frame so bleed from the music
/// does not count. `None` when less than a second is sung.
pub fn song_pitch(vocals: &[f32], rate: u32) -> Option<SongPitch> {
    let audio = crate::dsp::resample(vocals, rate, 16_000);
    #[cfg(feature = "world")]
    let f0 = crate::pitch::harvest(&audio);
    #[cfg(not(feature = "world"))]
    let f0 = crate::pitch::yin(&audio, audio.len() / 160);
    let loudness: Vec<f32> = (0..f0.len())
        .map(|frame| {
            let start = (frame * 160).min(audio.len());
            crate::dsp::rms(&audio[start..(start + 320).min(audio.len())])
        })
        .collect();
    let loudest = loudness.iter().copied().fold(0.0, f32::max);
    let mut sung: Vec<f32> = f0
        .iter()
        .zip(&loudness)
        .filter(|(hz, level)| **hz > 0.0 && **level > loudest * 0.03)
        .map(|(hz, _)| *hz)
        .collect();
    if sung.len() < 100 {
        return None;
    }
    sung.sort_by(f32::total_cmp);
    let at = |share: usize| sung[(sung.len() - 1) * share / 100];
    Some(SongPitch {
        median_hz: at(50),
        high_hz: at(95),
    })
}

/// Tune for singing `pitch` in a voice of `register`: whole octaves only, because
/// the music keeps its key; the octave that brings the melody's middle nearest the
/// register's, then lower while the high notes would go above what it sings well.
pub fn automatic_semitones(pitch: &SongPitch, register: Register) -> f32 {
    let octaves = (register.centre_hz() / pitch.median_hz).log2().round();
    let mut octaves = octaves.clamp(-2.0, 2.0);
    while octaves > -2.0 && pitch.high_hz * 2.0_f32.powf(octaves) > register.ceiling_hz() {
        octaves -= 1.0;
    }
    octaves * 12.0
}

/// The Studio settings as a cover uses them. The separated vocals are already
/// clean, so the microphone's clean-up stays off: echo cancellation would take
/// the voice's own previous block out of the next one, and noise suppression
/// can thin out long held notes.
pub fn song_settings(studio: &crate::config::Settings) -> crate::config::Settings {
    crate::config::Settings {
        bypass: false,
        echo: false,
        noise: crate::config::Noise::Off,
        ..studio.clone()
    }
}

/// Files a cover wrote.
pub struct CoverFiles {
    pub cover: std::path::PathBuf,
    pub stems: Vec<std::path::PathBuf>,
    pub seconds: f64,
    /// The Tune the voice sang with.
    pub semitones: f32,
    /// The song's pitch, when Tune was chosen from it.
    pub pitch: Option<SongPitch>,
}

/// A cover of a song in another voice: the vocals are separated, converted with
/// the RVC voice, and mixed back over the instrumental (16-bit stereo WAV).
pub fn cover(job: &CoverJob, progress: &Progress) -> Result<CoverFiles> {
    anyhow::ensure!(
        job.output != job.input,
        "The cover must not overwrite the song"
    );
    let song = crate::audio_file::read_stereo(job.input)?;
    let stems = separate(job.separation_model, job.settings.device, &song, progress)?;
    // The voice model is mono; it sings the mid of the vocals.
    let vocals: Vec<f32> = stems.vocals.channels[0]
        .iter()
        .zip(&stems.vocals.channels[1])
        .map(|(left, right)| (left + right) / 2.0)
        .collect();
    let mut settings = song_settings(job.settings);
    let mut pitch = None;
    if let Some(register) = job.auto_tune {
        progress.begin(MEASURING, 1);
        let measured = song_pitch(&vocals, RATE)
            .context("No singing was found in the song to choose Tune from; set Tune yourself")?;
        settings.semitones = automatic_semitones(&measured, register);
        pitch = Some(measured);
    }
    let converter = crate::engine::load_converter_on(job.voice, settings.device)?;
    let mut engine = crate::engine::Engine::new(Some(converter));
    progress.begin(CONVERTING, 1);
    let converted =
        crate::wav::convert_samples(&mut engine, &vocals, RATE, &settings, &mut |done, total| {
            progress.total.store(total, Ordering::Relaxed);
            progress.done.store(done, Ordering::Relaxed);
        })?;
    anyhow::ensure!(!progress.cancel.load(Ordering::Relaxed), "Cover cancelled");
    progress.begin(MIXING, 1);
    let mix: [Vec<f32>; 2] = std::array::from_fn(|c| {
        stems.instrumental.channels[c]
            .iter()
            .zip(&converted)
            .map(|(music, voice)| music * job.music_gain + voice * job.vocal_gain)
            .collect()
    });
    let cover = Stereo {
        rate: RATE,
        channels: mix,
    };
    crate::audio_file::write_pcm16(job.output, &cover)?;
    let mut written = Vec::new();
    if job.keep_stems {
        let (vocals_path, instrumental_path) = stem_paths(job.output, None);
        let converted_path = vocals_path.with_file_name(format!(
            "{}-new-vocals.wav",
            job.output.file_stem().unwrap_or_default().to_string_lossy()
        ));
        crate::audio_file::write_float(&vocals_path, &stems.vocals)?;
        crate::audio_file::write_float(&instrumental_path, &stems.instrumental)?;
        crate::audio_file::write_float(
            &converted_path,
            &Stereo {
                rate: RATE,
                channels: [converted.clone(), converted],
            },
        )?;
        written = vec![vocals_path, instrumental_path, converted_path];
    }
    progress.done.store(1, Ordering::Relaxed);
    Ok(CoverFiles {
        cover: job.output.to_path_buf(),
        stems: written,
        seconds: song.seconds(),
        semitones: settings.semitones,
        pitch,
    })
}

/// The two parts of a song, at [`RATE`].
pub struct Stems {
    pub vocals: Stereo,
    pub instrumental: Stereo,
}

/// The STFT and its inverse, as `torch.stft`/`torch.istft` with a periodic Hann
/// window, `center=True` and reflect padding.
struct Spectrum {
    window: Vec<f32>,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
}
impl Spectrum {
    fn new() -> Self {
        let window = (0..N_FFT)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos())
            .collect();
        let mut planner = FftPlanner::new();
        Self {
            window,
            forward: planner.plan_fft_forward(N_FFT),
            inverse: planner.plan_fft_inverse(N_FFT),
        }
    }

    /// `[DIM_T][N_BINS]` spectrum of one channel of a chunk of [`CHUNK`] samples.
    fn stft(&self, samples: &[f32]) -> Vec<Complex32> {
        debug_assert_eq!(samples.len(), CHUNK);
        let pad = N_FFT / 2;
        // Reflect padding: x[pad], ..., x[1] before and x[n-2], ... after.
        let at = |i: isize| -> f32 {
            let n = samples.len() as isize;
            let mut j = i - pad as isize;
            if j < 0 {
                j = -j;
            }
            if j >= n {
                j = 2 * (n - 1) - j;
            }
            samples[j as usize]
        };
        let mut out = Vec::with_capacity(DIM_T * N_BINS);
        let mut frame = vec![Complex32::new(0.0, 0.0); N_FFT];
        for t in 0..DIM_T {
            for (n, value) in frame.iter_mut().enumerate() {
                *value = Complex32::new(at((t * HOP + n) as isize) * self.window[n], 0.0);
            }
            self.forward.process(&mut frame);
            out.extend_from_slice(&frame[..N_BINS]);
        }
        out
    }

    /// One channel of [`CHUNK`] samples from a `[DIM_T][N_BINS]` spectrum.
    fn istft(&self, spectrum: &[Complex32]) -> Vec<f32> {
        let pad = N_FFT / 2;
        let length = CHUNK + 2 * pad;
        let (mut sum, mut weight) = (vec![0.0_f32; length], vec![0.0_f32; length]);
        let mut frame = vec![Complex32::new(0.0, 0.0); N_FFT];
        for t in 0..DIM_T {
            let bins = &spectrum[t * N_BINS..(t + 1) * N_BINS];
            frame[..N_BINS].copy_from_slice(bins);
            // A real signal: the DC and Nyquist bins are real, the rest mirrored.
            frame[0].im = 0.0;
            frame[N_BINS - 1].im = 0.0;
            for k in 1..N_FFT - N_BINS + 1 {
                frame[N_FFT - k] = bins[k].conj();
            }
            self.inverse.process(&mut frame);
            for n in 0..N_FFT {
                let w = self.window[n];
                sum[t * HOP + n] += frame[n].re / N_FFT as f32 * w;
                weight[t * HOP + n] += w * w;
            }
        }
        (pad..pad + CHUNK)
            .map(|i| {
                if weight[i] > 1e-11 {
                    sum[i] / weight[i]
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// The model input for a stereo chunk: `[1, 4, DIM_F, DIM_T]`, channels
/// left real, left imaginary, right real, right imaginary.
fn model_input(spectra: [&[Complex32]; 2]) -> Vec<f32> {
    let mut input = vec![0.0_f32; 4 * DIM_F * DIM_T];
    for (channel, spectrum) in spectra.iter().enumerate() {
        for t in 0..DIM_T {
            for f in 0..DIM_F {
                let value = spectrum[t * N_BINS + f];
                input[((channel * 2) * DIM_F + f) * DIM_T + t] = value.re;
                input[((channel * 2 + 1) * DIM_F + f) * DIM_T + t] = value.im;
            }
        }
    }
    input
}

/// Back from `[1, 4, DIM_F, DIM_T]` to two `[DIM_T][N_BINS]` spectra; the bins
/// above `DIM_F` are zero, as UVR pads them.
fn model_output(output: &[f32]) -> [Vec<Complex32>; 2] {
    std::array::from_fn(|channel| {
        let mut spectrum = vec![Complex32::new(0.0, 0.0); DIM_T * N_BINS];
        for t in 0..DIM_T {
            for f in 0..DIM_F {
                spectrum[t * N_BINS + f] = Complex32::new(
                    output[((channel * 2) * DIM_F + f) * DIM_T + t],
                    output[((channel * 2 + 1) * DIM_F + f) * DIM_T + t],
                );
            }
        }
        spectrum
    })
}

/// Separates `song` into vocals and instrumental with the MDX model at `model`.
pub fn separate(model: &Path, device: Device, song: &Stereo, progress: &Progress) -> Result<Stems> {
    anyhow::ensure!(!song.is_empty(), "The song is empty");
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut session = crate::rvc::open_session(model, threads, device)
        .with_context(|| format!("Load the vocal separation model {}", model.display()))?;
    anyhow::ensure!(
        session.inputs.len() == 1,
        "{} is not an MDX-Net vocal model",
        model.display()
    );
    let input_name = session.inputs[0].name.clone();
    let channels: [Vec<f32>; 2] = std::array::from_fn(|c| {
        if song.rate == RATE {
            song.channels[c].clone()
        } else {
            crate::dsp::resample(&song.channels[c], song.rate, RATE)
        }
    });
    let length = channels[0].len();
    // A normalised input separates better; the stems are scaled back.
    let peak = channels
        .iter()
        .flatten()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()))
        .max(1e-8);
    // TRIM of silence before, and after the end enough to fill the last chunk.
    let pad = STEP - length % STEP;
    let padded: [Vec<f32>; 2] = std::array::from_fn(|c| {
        let mut samples = vec![0.0; TRIM];
        samples.extend(channels[c].iter().map(|sample| sample / peak));
        samples.resize(TRIM + length + pad + TRIM, 0.0);
        samples
    });
    let chunks = (length + pad) / STEP;
    progress.begin(SEPARATING, chunks);
    let spectrum = Spectrum::new();
    let mut vocals: [Vec<f32>; 2] = [
        Vec::with_capacity(length + pad),
        Vec::with_capacity(length + pad),
    ];
    for chunk in 0..chunks {
        anyhow::ensure!(
            !progress.cancel.load(Ordering::Relaxed),
            "Separation cancelled"
        );
        let start = chunk * STEP;
        let spectra: [Vec<Complex32>; 2] =
            std::array::from_fn(|c| spectrum.stft(&padded[c][start..start + CHUNK]));
        let input = model_input([&spectra[0], &spectra[1]]);
        let input = Tensor::from_array(([1, 4, DIM_F, DIM_T], input.into_boxed_slice()))?;
        let output = session.run(ort::inputs! { input_name.as_str() => input })?;
        let (shape, values) = output[0].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            values.len() == 4 * DIM_F * DIM_T,
            "Unexpected vocal model output {shape:?}"
        );
        for (c, spectrum_out) in model_output(values).iter().enumerate() {
            let audio = spectrum.istft(spectrum_out);
            vocals[c].extend_from_slice(&audio[TRIM..TRIM + STEP]);
        }
        progress.done.store(chunk + 1, Ordering::Relaxed);
    }
    let vocals: [Vec<f32>; 2] = std::array::from_fn(|c| {
        vocals[c][..length]
            .iter()
            .map(|sample| sample * peak)
            .collect()
    });
    let instrumental: [Vec<f32>; 2] = std::array::from_fn(|c| {
        channels[c]
            .iter()
            .zip(&vocals[c])
            .map(|(mix, vocal)| mix - vocal * COMPENSATE)
            .collect()
    });
    anyhow::ensure!(
        vocals.iter().flatten().all(|sample| sample.is_finite()),
        "The vocal model returned invalid audio"
    );
    Ok(Stems {
        vocals: Stereo {
            rate: RATE,
            channels: vocals,
        },
        instrumental: Stereo {
            rate: RATE,
            channels: instrumental,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_tune_moves_whole_octaves_to_the_voice() {
        let pitch = |median_hz, high_hz| SongPitch { median_hz, high_hz };
        // A low female singer (measured: "From The Start", median 310 Hz, high 449 Hz).
        let laufey = pitch(310.0, 449.0);
        assert_eq!(automatic_semitones(&laufey, Register::Female), 0.0);
        assert_eq!(automatic_semitones(&laufey, Register::Male), -12.0);
        // A male singer in a female voice goes up, unless the high notes would be
        // too high.
        assert_eq!(
            automatic_semitones(&pitch(160.0, 260.0), Register::Female),
            12.0
        );
        assert_eq!(
            automatic_semitones(&pitch(240.0, 480.0), Register::Female),
            0.0
        );
        // A high soprano in a male voice drops two octaves at most.
        assert_eq!(
            automatic_semitones(&pitch(700.0, 1400.0), Register::Male),
            -24.0
        );
    }

    #[test]
    fn song_pitch_reads_a_sung_tone_and_ignores_quiet_bleed() {
        let rate = 44_100;
        let tone = |hz: f32, level: f32, seconds: f32| -> Vec<f32> {
            (0..(rate as f32 * seconds) as usize)
                .map(|i| {
                    let t = i as f32 / rate as f32;
                    (1..=6)
                        .map(|h| (std::f32::consts::TAU * hz * h as f32 * t).sin() / h as f32)
                        .sum::<f32>()
                        * level
                })
                .collect()
        };
        let mut vocals = tone(220.0, 0.3, 1.5);
        vocals.extend(tone(440.0, 0.3, 0.5));
        vocals.extend(tone(1000.0, 0.003, 1.0));
        let pitch = song_pitch(&vocals, rate).expect("sung");
        assert!((pitch.median_hz - 220.0).abs() < 5.0, "{pitch:?}");
        assert!((pitch.high_hz - 440.0).abs() < 10.0, "{pitch:?}");
        assert!(song_pitch(&vec![0.0; rate as usize], rate).is_none());
    }

    #[test]
    fn a_cover_keeps_the_tuning_but_not_the_microphone_clean_up() {
        let studio = crate::config::Settings {
            bypass: true,
            echo: true,
            noise: crate::config::Noise::Sup2,
            semitones: -5.0,
            index_ratio: 0.5,
            ..Default::default()
        };
        let song = song_settings(&studio);
        assert!(!song.bypass && !song.echo);
        assert_eq!(song.noise, crate::config::Noise::Off);
        assert_eq!((song.semitones, song.index_ratio), (-5.0, 0.5));
    }

    #[test]
    fn the_spectrum_turns_back_into_the_same_audio() {
        let spectrum = Spectrum::new();
        let samples: Vec<f32> = (0..CHUNK)
            .map(|i| (i as f32 * 0.013).sin() * 0.4 + (i as f32 * 0.0007).cos() * 0.2)
            .collect();
        let back = spectrum.istft(&spectrum.stft(&samples));
        let worst = samples
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 1e-3, "worst difference {worst}");
    }

    #[test]
    fn the_model_layout_round_trips() {
        let spectra: [Vec<Complex32>; 2] = std::array::from_fn(|c| {
            (0..DIM_T * N_BINS)
                .map(|i| {
                    let (t, f) = (i / N_BINS, i % N_BINS);
                    if f < DIM_F {
                        Complex32::new((c * 7 + t) as f32, f as f32)
                    } else {
                        Complex32::new(0.0, 0.0)
                    }
                })
                .collect()
        });
        let input = model_input([&spectra[0], &spectra[1]]);
        assert_eq!(model_output(&input), spectra);
        // Channel 1 (left imaginary), bin 5, frame 3.
        assert_eq!(input[(DIM_F + 5) * DIM_T + 3], 5.0);
    }
}
