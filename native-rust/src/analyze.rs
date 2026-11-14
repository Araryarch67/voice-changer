//! Objective audio diagnostics, spectral/pitch inspection, and PNG plot generation
//! for analyzing input vs converted RVC audio quality.

use crate::{config::Settings, dsp, wav};
use anyhow::{Context, Result};
use serde::Serialize;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

const ANALYSIS_RATE: u32 = 16_000;
const HOP_SAMPLES: usize = 160; // 10 ms at 16 kHz
const YIN_WINDOW: usize = 640; // 40 ms at 16 kHz
const FFT_SIZE: usize = 1024;
const SILENCE_RMS: f32 = 0.005; // ~ -46 dBFS
const MAX_TIMELINE_BINS: usize = 400;
const MAX_EVENTS: usize = 20;

#[derive(Debug, Clone, Serialize)]
pub struct SpectralBands {
    pub sub_80hz_pct: f32,
    pub body_80_300hz_pct: f32,
    pub low_mid_300_1khz_pct: f32,
    pub presence_1k_4khz_pct: f32,
    pub brilliance_4k_8khz_pct: f32,
    pub air_over_8khz_pct: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SignalStats {
    pub sample_rate: u32,
    pub samples: usize,
    pub duration_s: f32,
    pub rms_db: f32,
    pub active_rms_db: f32,
    pub peak_db: f32,
    pub crest_factor_db: f32,
    pub dc_offset: f32,
    pub clipped_samples: usize,
    pub near_clip_samples: usize,
    pub silence_fraction: f32,
    pub max_step_jump: f32,
    pub click_count: usize,
    pub voiced_fraction: f32,
    pub f0_min_hz: Option<f32>,
    pub f0_p10_hz: Option<f32>,
    pub f0_median_hz: Option<f32>,
    pub f0_p90_hz: Option<f32>,
    pub f0_max_hz: Option<f32>,
    pub octave_jumps: usize,
    pub pitch_jitter_cents: Option<f32>,
    pub spectral_centroid_hz: f32,
    pub spectral_rolloff_85_hz: f32,
    pub spectral_flatness: f32,
    pub bands: SpectralBands,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonStats {
    pub estimated_latency_ms: Option<f32>,
    pub envelope_correlation: f32,
    pub measured_pitch_shift_semitones: Option<f32>,
    pub pitch_shift_std_semitones: Option<f32>,
    pub voiced_dropout_fraction: Option<f32>,
    pub active_drop_fraction: Option<f32>,
    pub noise_floor_db: Option<f32>,
    pub brightness_delta_hz: f32,
    pub rolloff_delta_hz: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticEvent {
    pub time_s: f32,
    pub kind: String,
    pub detail: String,
    pub severity: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimelineBin {
    pub t_s: f32,
    pub in_rms_db: Option<f32>,
    pub out_rms_db: f32,
    pub in_f0_hz: Option<f32>,
    pub out_f0_hz: Option<f32>,
    pub out_centroid_hz: f32,
    pub clicks: usize,
    pub clips: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveHealthStats {
    pub blocks: usize,
    pub block_ms: u32,
    pub worker_p50_ms: f32,
    pub worker_p95_ms: f32,
    pub worker_max_ms: f32,
    pub dropped_input_samples: u64,
    pub dropped_output_samples: u64,
    pub underrun_samples: u64,
    pub queue_peak_ms: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalysisReport {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub settings: Option<Settings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub live_health: Option<LiveHealthStats>,
    pub input: Option<SignalStats>,
    pub output: SignalStats,
    pub comparison: Option<ComparisonStats>,
    pub diagnostics: Vec<String>,
    pub events: Vec<DiagnosticEvent>,
    pub timeline: Vec<TimelineBin>,
}

struct TrackFrames {
    stats: SignalStats,
    rms_frames: Vec<f32>,
    /// High-passed 80+ Hz RMS envelope so sub-bass hum does not count as speech activity.
    speech_rms_frames: Vec<f32>,
    f0_frames: Vec<Option<f32>>,
    centroid_frames: Vec<f32>,
    click_frames: Vec<usize>,
    clip_frames: Vec<usize>,
    spectrogram: Vec<Vec<f32>>,
    events: Vec<DiagnosticEvent>,
    duration_s: f32,
}

/// Analyzes a single audio buffer (assumed to be the output/target recording).
pub fn analyze_samples(samples: &[f32], rate: u32, label: &str) -> AnalysisReport {
    analyze_single_with_frames(samples, rate, label).0
}

fn analyze_single_with_frames(
    samples: &[f32],
    rate: u32,
    label: &str,
) -> (AnalysisReport, TrackFrames) {
    let out = analyze_track(samples, rate, "output");
    let mut diagnostics = Vec::new();
    collect_single_diagnostics(&out.stats, "Output", &mut diagnostics);
    if diagnostics.is_empty() {
        diagnostics.push("No major signal anomalies detected.".to_owned());
    }
    let timeline = build_timeline(None, &out, 0);
    let mut events = out.events.clone();
    trim_events(&mut events);
    let report = AnalysisReport {
        label: label.to_owned(),
        model: None,
        settings: None,
        live_health: None,
        input: None,
        output: out.stats.clone(),
        comparison: None,
        diagnostics,
        events,
        timeline,
    };
    (report, out)
}

/// Analyzes an input and converted output audio pair together.
pub fn analyze_pair_samples(
    input: &[f32],
    in_rate: u32,
    output: &[f32],
    out_rate: u32,
    label: &str,
) -> AnalysisReport {
    analyze_pair_with_frames(input, in_rate, output, out_rate, label).0
}

fn analyze_pair_with_frames(
    input: &[f32],
    in_rate: u32,
    output: &[f32],
    out_rate: u32,
    label: &str,
) -> (AnalysisReport, TrackFrames, TrackFrames) {
    let in_track = analyze_track(input, in_rate, "input");
    let out_track = analyze_track(output, out_rate, "output");
    let (comparison, lag_frames, mut pair_events) = compare_tracks(&in_track, &out_track);

    let mut diagnostics = Vec::new();
    collect_single_diagnostics(&in_track.stats, "Input mic", &mut diagnostics);
    collect_single_diagnostics(&out_track.stats, "Converted output", &mut diagnostics);
    collect_comparison_diagnostics(&comparison, &mut diagnostics);
    if diagnostics.is_empty() {
        diagnostics.push("Input and converted output look healthy.".to_owned());
    }

    let timeline = build_timeline(Some(&in_track), &out_track, lag_frames);
    let mut events = out_track.events.clone();
    events.append(&mut pair_events);
    trim_events(&mut events);

    let report = AnalysisReport {
        label: label.to_owned(),
        model: None,
        settings: None,
        live_health: None,
        input: Some(in_track.stats.clone()),
        output: out_track.stats.clone(),
        comparison: Some(comparison),
        diagnostics,
        events,
        timeline,
    };
    (report, in_track, out_track)
}

/// Loads and analyzes a single WAV file.
pub fn analyze_wav(path: &Path) -> Result<AnalysisReport> {
    let (rate, samples) = wav::read_mono(path)?;
    let label = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    Ok(analyze_samples(&samples, rate, &label))
}

/// Loads and analyzes an input/output WAV file pair.
pub fn analyze_wav_pair(input: &Path, output: &Path) -> Result<AnalysisReport> {
    let (in_rate, in_samples) = wav::read_mono(input)?;
    let (out_rate, out_samples) = wav::read_mono(output)?;
    let label = format!(
        "{} -> {}",
        input.file_name().unwrap_or_default().to_string_lossy(),
        output.file_name().unwrap_or_default().to_string_lossy()
    );
    Ok(analyze_pair_samples(
        &in_samples,
        in_rate,
        &out_samples,
        out_rate,
        &label,
    ))
}

/// Writes `<prefix>-report.json` and (when PNG encoding is built in) `<prefix>-report.png`.
pub fn write_reports(
    report: &AnalysisReport,
    prefix: &Path,
    input: Option<(&[f32], u32)>,
    output: (&[f32], u32),
) -> Result<(PathBuf, Option<PathBuf>)> {
    let in_track = input.map(|(samples, rate)| analyze_track(samples, rate, "input"));
    let out_track = analyze_track(output.0, output.1, "output");
    write_report_files(report, prefix, in_track.as_ref(), &out_track)
}

fn write_report_files(
    report: &AnalysisReport,
    prefix: &Path,
    in_track: Option<&TrackFrames>,
    out_track: &TrackFrames,
) -> Result<(PathBuf, Option<PathBuf>)> {
    if let Some(parent) = prefix.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Create report directory {}", parent.display()))?;
        }
    }
    let json_path = suffixed_path(prefix, "report.json");
    let json = serde_json::to_string_pretty(report).context("Serialize diagnostic report")?;
    std::fs::write(&json_path, json).with_context(|| format!("Write {}", json_path.display()))?;

    #[cfg(any(feature = "gui", test))]
    let png_path = {
        let png = suffixed_path(prefix, "report.png");
        render_diagnostic_png(report, in_track, out_track, &png)?;
        Some(png)
    };
    #[cfg(not(any(feature = "gui", test)))]
    let png_path = {
        let _ = (in_track, out_track);
        None
    };

    Ok((json_path, png_path))
}

/// Convenience helper for WAV inputs: analyzes a single WAV or pair and writes both report files.
pub fn analyze_wavs_to_disk(
    first: &Path,
    second: Option<&Path>,
    prefix: &Path,
) -> Result<(AnalysisReport, PathBuf, Option<PathBuf>)> {
    analyze_wavs_with_context(first, second, prefix, None, None)
}

/// Like [`analyze_wavs_to_disk`], attaching the active model path and [`Settings`] snapshot.
pub fn analyze_wavs_with_context(
    first: &Path,
    second: Option<&Path>,
    prefix: &Path,
    model: Option<&str>,
    settings: Option<&Settings>,
) -> Result<(AnalysisReport, PathBuf, Option<PathBuf>)> {
    analyze_wavs_with_live_health(first, second, prefix, model, settings, None)
}

/// Like [`analyze_wavs_with_context`], also attaching live stream playback health telemetry.
pub fn analyze_wavs_with_live_health(
    first: &Path,
    second: Option<&Path>,
    prefix: &Path,
    model: Option<&str>,
    settings: Option<&Settings>,
    live_health: Option<LiveHealthStats>,
) -> Result<(AnalysisReport, PathBuf, Option<PathBuf>)> {
    let (out_rate, out_samples) = wav::read_mono(second.unwrap_or(first))?;
    let in_loaded = second
        .is_some()
        .then(|| wav::read_mono(first))
        .transpose()?;
    let (mut report, in_track, out_track) = match &in_loaded {
        Some((in_rate, in_samples)) => {
            let (report, in_t, out_t) = analyze_pair_with_frames(
                in_samples,
                *in_rate,
                &out_samples,
                out_rate,
                &format!(
                    "{} -> {}",
                    first.file_name().unwrap_or_default().to_string_lossy(),
                    second
                        .unwrap_or(first)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                ),
            );
            (report, Some(in_t), out_t)
        }
        None => {
            let (report, out_t) = analyze_single_with_frames(
                &out_samples,
                out_rate,
                &first.file_name().unwrap_or_default().to_string_lossy(),
            );
            (report, None, out_t)
        }
    };
    report.model = model.map(str::to_owned);
    report.settings = settings.cloned();
    report.live_health = live_health;
    let (json_path, png_path) = write_report_files(&report, prefix, in_track.as_ref(), &out_track)?;
    Ok((report, json_path, png_path))
}

/// Formats a concise human-readable diagnostic summary.
pub fn format_summary(report: &AnalysisReport) -> String {
    let o = &report.output;
    let mut parts = vec![format!(
        "{} ({:.2}s @ {} Hz): RMS {:.1} dB, active {:.1} dB, peak {:.1} dB, clips {}, clicks {}",
        report.label,
        o.duration_s,
        o.sample_rate,
        o.rms_db,
        o.active_rms_db,
        o.peak_db,
        o.clipped_samples,
        o.click_count
    )];
    if let Some(f0) = o.f0_median_hz {
        parts.push(format!(
            "Output F0 median {:.1} Hz ({:.0}% voiced, {} octave jumps)",
            f0,
            o.voiced_fraction * 100.0,
            o.octave_jumps
        ));
    }
    if let Some(cmp) = &report.comparison {
        let shift = cmp
            .measured_pitch_shift_semitones
            .map(|s| format!("{s:+.2} st"))
            .unwrap_or_else(|| "n/a".to_owned());
        let drop = cmp
            .voiced_dropout_fraction
            .map(|d| format!("{:.0}%", d * 100.0))
            .unwrap_or_else(|| "n/a".to_owned());
        let lat = cmp
            .estimated_latency_ms
            .map(|ms| format!("{ms:.0} ms"))
            .unwrap_or_else(|| "n/a".to_owned());
        parts.push(format!(
            "Shift {shift} | Latency {lat} | Voiced dropouts {drop} | Centroid delta {:+.0} Hz",
            cmp.brightness_delta_hz
        ));
    }
    for diag in &report.diagnostics {
        parts.push(format!("  - {diag}"));
    }
    parts.join("\n")
}

fn suffixed_path(prefix: &Path, suffix: &str) -> PathBuf {
    let base = prefix.file_name().unwrap_or_default().to_string_lossy();
    prefix.with_file_name(format!("{base}-{suffix}"))
}

fn analyze_track(samples: &[f32], rate: u32, role: &str) -> TrackFrames {
    let rate = rate.max(1);
    let duration_s = samples.len() as f32 / rate as f32;
    let resampled = if rate == ANALYSIS_RATE {
        samples.to_vec()
    } else {
        dsp::resample(samples, rate, ANALYSIS_RATE)
    };
    let mut hp_resampled = resampled.clone();
    dsp::HighPass::default_voice(ANALYSIS_RATE).process(&mut hp_resampled);

    let n_frames = (resampled.len() / HOP_SAMPLES).max(1);
    let mut rms_frames = Vec::with_capacity(n_frames);
    let mut speech_rms_frames = Vec::with_capacity(n_frames);
    let mut f0_frames = Vec::with_capacity(n_frames);
    let mut centroid_frames = Vec::with_capacity(n_frames);
    let mut spectrogram = Vec::with_capacity(n_frames);
    let mut events = Vec::new();

    // Precompute per-frame native clip and click counts aligned to the 10 ms frame grid.
    let mut click_frames = vec![0usize; n_frames];
    let mut clip_frames = vec![0usize; n_frames];
    let native_hop = ((rate as usize) / 100).max(1);

    let (
        dc_offset,
        clipped_samples,
        near_clip_samples,
        max_step_jump,
        click_count,
        mut native_events,
    ) = scan_time_domain(
        samples,
        rate,
        native_hop,
        &mut click_frames,
        &mut clip_frames,
    );
    if role == "output" {
        events.append(&mut native_events);
    }

    let hann = hann_window(FFT_SIZE);
    let mut active_bins = 0usize;
    let mut total_energy = 0.0_f64;
    let mut active_energy = 0.0_f64;
    let mut peak = 0.0_f32;
    for &s in samples {
        let a = s.abs();
        peak = peak.max(a);
        total_energy += f64::from(s) * f64::from(s);
    }

    for frame_idx in 0..n_frames {
        let center = frame_idx * HOP_SAMPLES + HOP_SAMPLES / 2;
        let rms = window_rms(&resampled, center, HOP_SAMPLES);
        let speech_rms = window_rms(&hp_resampled, center, HOP_SAMPLES);
        rms_frames.push(rms);
        speech_rms_frames.push(speech_rms);

        let f0 = if speech_rms >= SILENCE_RMS {
            active_bins += 1;
            active_energy += f64::from(rms) * f64::from(rms);
            yin_f0(&hp_resampled, center)
        } else {
            None
        };
        f0_frames.push(f0);

        let mag = stft_frame(&resampled, center, &hann);
        let centroid = frame_centroid(&mag, ANALYSIS_RATE);
        centroid_frames.push(centroid);
        spectrogram.push(mag);
    }
    smooth_isolated_f0_glitches(&mut f0_frames);

    // Native-rate spectral bands so >8 kHz aliasing/air is captured when rate >= 22.05 kHz.
    let (bands, spectral_centroid_hz, spectral_rolloff_85_hz, spectral_flatness) =
        compute_native_spectrum(samples, rate);

    let rms = if samples.is_empty() {
        0.0
    } else {
        (total_energy / samples.len() as f64).sqrt() as f32
    };
    let active_rms = if active_bins > 0 {
        (active_energy / active_bins as f64).sqrt() as f32
    } else {
        rms
    };
    let rms_db = to_db(rms);
    let active_rms_db = to_db(active_rms);
    let peak_db = to_db(peak);

    let mut f0 = summarize_f0(&f0_frames, active_bins);
    if role == "output" {
        events.append(&mut f0.events);
    }

    let stats = SignalStats {
        sample_rate: rate,
        samples: samples.len(),
        duration_s,
        rms_db,
        active_rms_db,
        peak_db,
        crest_factor_db: peak_db - rms_db,
        dc_offset,
        clipped_samples,
        near_clip_samples,
        silence_fraction: 1.0 - (active_bins as f32 / n_frames as f32),
        max_step_jump,
        click_count,
        voiced_fraction: f0.voiced_fraction,
        f0_min_hz: f0.min_hz,
        f0_p10_hz: f0.p10_hz,
        f0_median_hz: f0.median_hz,
        f0_p90_hz: f0.p90_hz,
        f0_max_hz: f0.max_hz,
        octave_jumps: f0.octave_jumps,
        pitch_jitter_cents: f0.jitter_cents,
        spectral_centroid_hz,
        spectral_rolloff_85_hz,
        spectral_flatness,
        bands,
    };

    TrackFrames {
        stats,
        rms_frames,
        speech_rms_frames,
        f0_frames,
        centroid_frames,
        click_frames,
        clip_frames,
        spectrogram,
        events,
        duration_s,
    }
}

fn scan_time_domain(
    samples: &[f32],
    rate: u32,
    hop: usize,
    click_frames: &mut [usize],
    clip_frames: &mut [usize],
) -> (f32, usize, usize, f32, usize, Vec<DiagnosticEvent>) {
    if samples.is_empty() {
        return (0.0, 0, 0, 0.0, 0, Vec::new());
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s)).sum();
    let dc_offset = (sum / samples.len() as f64) as f32;

    let mut clipped = 0usize;
    let mut near_clip = 0usize;
    let mut max_step = 0.0_f32;
    let mut clicks = 0usize;
    let mut events = Vec::new();
    let merge_gap = (rate as usize / 40).max(1); // 25 ms cooldown
    let mut last_click_idx: Option<usize> = None;
    let mut last_clip_idx: Option<usize> = None;

    // Precompute 10 ms local RMS per hop so step threshold scales cleanly with speech loudness.
    let hop_rms: Vec<f32> = samples
        .chunks(hop)
        .map(|chunk| {
            let e: f32 = chunk.iter().map(|&x| x * x).sum();
            (e / chunk.len().max(1) as f32).sqrt()
        })
        .collect();

    let short_win = (rate as usize / 500).max(8); // ~2 ms one-sided neighborhood

    for (i, &s) in samples.iter().enumerate() {
        let frame = (i / hop).min(click_frames.len().saturating_sub(1));
        let abs = s.abs();
        if abs >= 0.999 {
            clipped += 1;
            clip_frames[frame] += 1;
            if last_clip_idx.is_none_or(|prev| i - prev >= merge_gap) {
                last_clip_idx = Some(i);
                events.push(DiagnosticEvent {
                    time_s: i as f32 / rate as f32,
                    kind: "clip".to_owned(),
                    detail: format!("Sample clipped at {:.3}", s),
                    severity: 0.9,
                });
            }
        } else if abs >= 0.98 {
            near_clip += 1;
        }
        if i >= 1 {
            let step = (s - samples[i - 1]).abs();
            max_step = max_step.max(step);
            let accel = if i >= 2 {
                (s - 2.0 * samples[i - 1] + samples[i - 2]).abs()
            } else {
                step
            };
            let hop_local = hop_rms.get(i / hop).copied().unwrap_or(0.0);
            let is_click = if step > 0.18 || accel > 0.28 {
                // Evaluate immediate +/- 2 ms energy excluding [i-1..=i] so sharp
                // high-frequency sibilant/plosive bursts near a 10 ms bin boundary
                // don't false-trigger as discontinuity clicks.
                let before_start = i.saturating_sub(1 + short_win);
                let before_end = i.saturating_sub(1);
                let after_start = (i + 1).min(samples.len());
                let after_end = (i + 1 + short_win).min(samples.len());
                let rms_slice = |sl: &[f32]| -> f32 {
                    if sl.is_empty() {
                        0.0
                    } else {
                        (sl.iter().map(|&x| x * x).sum::<f32>() / sl.len() as f32).sqrt()
                    }
                };
                let local = hop_local
                    .max(rms_slice(&samples[before_start..before_end]))
                    .max(rms_slice(&samples[after_start..after_end]));
                (step > 0.18 && step > 4.5 * local) || (accel > 0.28 && accel > 6.0 * local)
            } else {
                false
            };
            if is_click && last_click_idx.is_none_or(|prev| i - prev >= merge_gap) {
                last_click_idx = Some(i);
                clicks += 1;
                click_frames[frame] += 1;
                events.push(DiagnosticEvent {
                    time_s: i as f32 / rate as f32,
                    kind: "click".to_owned(),
                    detail: format!("Discontinuity jump {step:.3} (local RMS {hop_local:.3})"),
                    severity: (step * 2.0).clamp(0.4, 1.0),
                });
            }
        }
    }

    (dc_offset, clipped, near_clip, max_step, clicks, events)
}

struct F0Summary {
    voiced_fraction: f32,
    min_hz: Option<f32>,
    p10_hz: Option<f32>,
    median_hz: Option<f32>,
    p90_hz: Option<f32>,
    max_hz: Option<f32>,
    octave_jumps: usize,
    jitter_cents: Option<f32>,
    events: Vec<DiagnosticEvent>,
}

/// Removes isolated 1-frame YIN subharmonic/harmonic jumps without altering real >= 2-frame
/// register breaks or pitch steps.
fn smooth_isolated_f0_glitches(f0_frames: &mut [Option<f32>]) {
    let raw = f0_frames.to_vec();
    let len = raw.len();
    let cents_diff = |a: f32, b: f32| (1200.0 * (a / b).log2()).abs();
    for i in 0..len {
        let Some(cur) = raw[i] else {
            continue;
        };
        let prev1 = i.checked_sub(1).and_then(|j| raw[j]);
        let next1 = raw.get(i + 1).copied().flatten();
        match (prev1, next1) {
            (Some(a), Some(b)) => {
                if cents_diff(a, b) <= 500.0
                    && cents_diff(cur, a) >= 900.0
                    && cents_diff(cur, b) >= 900.0
                {
                    f0_frames[i] = Some(0.5 * (a + b));
                }
            }
            (None, Some(n1)) => {
                if let Some(n2) = raw.get(i + 2).copied().flatten() {
                    if cents_diff(n1, n2) <= 500.0 && cents_diff(cur, n1) >= 900.0 {
                        f0_frames[i] = Some(n1);
                    }
                }
            }
            (Some(p1), None) => {
                if let Some(p2) = i.checked_sub(2).and_then(|j| raw[j]) {
                    if cents_diff(p1, p2) <= 500.0 && cents_diff(cur, p1) >= 900.0 {
                        f0_frames[i] = Some(p1);
                    }
                }
            }
            (None, None) => {}
        }
    }
}

fn summarize_f0(f0_frames: &[Option<f32>], active_bins: usize) -> F0Summary {
    let mut voiced: Vec<f32> = f0_frames.iter().filter_map(|&v| v).collect();
    let voiced_fraction = if active_bins > 0 {
        (voiced.len() as f32 / active_bins as f32).min(1.0)
    } else {
        0.0
    };
    if voiced.is_empty() {
        return F0Summary {
            voiced_fraction,
            min_hz: None,
            p10_hz: None,
            median_hz: None,
            p90_hz: None,
            max_hz: None,
            octave_jumps: 0,
            jitter_cents: None,
            events: Vec::new(),
        };
    }
    voiced.sort_by(f32::total_cmp);
    let quantile = |q: f32| -> f32 {
        let idx = ((voiced.len() - 1) as f32 * q).round() as usize;
        voiced[idx.min(voiced.len() - 1)]
    };

    let mut octave_jumps = 0usize;
    let mut deltas = Vec::new();
    let mut events = Vec::new();
    let mut prev: Option<(usize, f32)> = None;
    for (idx, &item) in f0_frames.iter().enumerate() {
        let Some(curr) = item else {
            continue;
        };
        if let Some((prev_idx, prev_hz)) = prev {
            if idx - prev_idx <= 3 {
                let cents = (1200.0 * (curr / prev_hz).log2()).abs();
                deltas.push(cents);
                if cents >= 1000.0 {
                    octave_jumps += 1;
                    events.push(DiagnosticEvent {
                        time_s: idx as f32 * 0.01,
                        kind: "octave_jump".to_owned(),
                        detail: format!("Pitch jumped {prev_hz:.1} -> {curr:.1} Hz"),
                        severity: 0.85,
                    });
                }
            }
        }
        prev = Some((idx, curr));
    }
    deltas.sort_by(f32::total_cmp);
    let jitter = if deltas.is_empty() {
        None
    } else {
        Some(deltas[deltas.len() / 2])
    };

    F0Summary {
        voiced_fraction,
        min_hz: Some(voiced[0]),
        p10_hz: Some(quantile(0.10)),
        median_hz: Some(quantile(0.50)),
        p90_hz: Some(quantile(0.90)),
        max_hz: Some(*voiced.last().unwrap()),
        octave_jumps,
        jitter_cents: jitter,
        events,
    }
}

fn compare_tracks(
    input: &TrackFrames,
    output: &TrackFrames,
) -> (ComparisonStats, isize, Vec<DiagnosticEvent>) {
    let (lag, corr) = estimate_latency_lag(&input.speech_rms_frames, &output.speech_rms_frames);
    let estimated_latency_ms = if corr >= 0.3 {
        Some(lag as f32 * 10.0)
    } else {
        None
    };
    let effective_lag = if corr >= 0.3 { lag } else { 0 };

    let mut shifts = Vec::new();
    let mut in_voiced = 0usize;
    let mut voiced_dropped = 0usize;
    let mut in_active = 0usize;
    let mut active_dropped = 0usize;
    let mut noise_energy = 0.0_f64;
    let mut noise_count = 0usize;
    let mut events = Vec::new();
    let mut last_drop_frame: Option<usize> = None;

    for out_i in 0..output.rms_frames.len() {
        let in_i = out_i as isize - effective_lag;
        if in_i < 0 || (in_i as usize) >= input.speech_rms_frames.len() {
            continue;
        }
        let in_idx = in_i as usize;
        let in_rms = input.speech_rms_frames[in_idx];
        let out_rms = output.rms_frames[out_i];
        if in_rms >= SILENCE_RMS {
            in_active += 1;
            if out_rms < SILENCE_RMS * 0.4 {
                active_dropped += 1;
                if last_drop_frame.is_none_or(|prev| out_i - prev >= 20) {
                    last_drop_frame = Some(out_i);
                    events.push(DiagnosticEvent {
                        time_s: out_i as f32 * 0.01,
                        kind: "active_drop".to_owned(),
                        detail: "Input had speech but output went silent".to_owned(),
                        severity: 0.75,
                    });
                }
            }
            if let Some(in_f0) = input.f0_frames[in_idx] {
                in_voiced += 1;
                if let Some(out_f0) = output.f0_frames[out_i] {
                    shifts.push(12.0 * (out_f0 / in_f0).log2());
                } else if out_rms >= SILENCE_RMS {
                    voiced_dropped += 1;
                }
            }
        } else {
            noise_energy += f64::from(out_rms) * f64::from(out_rms);
            noise_count += 1;
        }
    }

    shifts.sort_by(f32::total_cmp);
    let measured_pitch_shift_semitones = if shifts.is_empty() {
        None
    } else {
        Some(shifts[shifts.len() / 2])
    };
    let pitch_shift_std_semitones = if shifts.len() >= 2 {
        let mean = shifts.iter().sum::<f32>() / shifts.len() as f32;
        let var =
            shifts.iter().map(|&s| (s - mean) * (s - mean)).sum::<f32>() / shifts.len() as f32;
        Some(var.sqrt())
    } else {
        None
    };

    let comparison = ComparisonStats {
        estimated_latency_ms,
        envelope_correlation: corr,
        measured_pitch_shift_semitones,
        pitch_shift_std_semitones,
        voiced_dropout_fraction: (in_voiced > 0).then(|| voiced_dropped as f32 / in_voiced as f32),
        active_drop_fraction: (in_active > 0).then(|| active_dropped as f32 / in_active as f32),
        noise_floor_db: (noise_count > 0)
            .then(|| to_db((noise_energy / noise_count as f64).sqrt() as f32)),
        brightness_delta_hz: output.stats.spectral_centroid_hz - input.stats.spectral_centroid_hz,
        rolloff_delta_hz: output.stats.spectral_rolloff_85_hz - input.stats.spectral_rolloff_85_hz,
    };

    (comparison, effective_lag, events)
}

fn estimate_latency_lag(in_env: &[f32], out_env: &[f32]) -> (isize, f32) {
    let total_in: f64 = in_env.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let total_out: f64 = out_env.iter().map(|&y| f64::from(y) * f64::from(y)).sum();
    let denom = (total_in * total_out).sqrt();
    if denom <= 1e-9 {
        return (0, 0.0);
    }
    let max_lag = 200_isize.min(out_env.len() as isize); // up to 2000 ms at 10 ms/frame
    let mut best_lag = 0_isize;
    let mut best_corr = 0.0_f32;
    for lag in -20_isize..=max_lag {
        let mut dot = 0.0_f64;
        for (out_i, &o) in out_env.iter().enumerate() {
            let in_i = out_i as isize - lag;
            if in_i < 0 || (in_i as usize) >= in_env.len() {
                continue;
            }
            dot += f64::from(in_env[in_i as usize]) * f64::from(o);
        }
        let corr = (dot / denom) as f32;
        if corr > best_corr + 1e-4
            || ((corr - best_corr).abs() <= 1e-4 && lag.abs() < best_lag.abs())
        {
            best_corr = corr;
            best_lag = lag;
        }
    }
    (best_lag, best_corr)
}

fn collect_single_diagnostics(stats: &SignalStats, prefix: &str, out: &mut Vec<String>) {
    if stats.clipped_samples > 0 {
        out.push(format!(
            "{prefix} clipped {} samples (peak {:.1} dB) — lower Gain or watch mic headroom.",
            stats.clipped_samples, stats.peak_db
        ));
    }
    if stats.click_count > 0 {
        out.push(format!(
            "{prefix} has {} sudden waveform discontinuities/clicks (max step {:.2}).",
            stats.click_count, stats.max_step_jump
        ));
    }
    if stats.dc_offset.abs() > 0.02 {
        out.push(format!(
            "{prefix} carries DC offset ({:+.3}).",
            stats.dc_offset
        ));
    }
    if stats.bands.sub_80hz_pct > 15.0 {
        out.push(format!(
            "{prefix} has heavy sub-80 Hz rumble ({:.1}% energy).",
            stats.bands.sub_80hz_pct
        ));
    }
    if stats.octave_jumps > 2 {
        out.push(format!(
            "{prefix} has {} pitch octave jumps — try RMVPE/FCPE or stabilize mic level.",
            stats.octave_jumps
        ));
    }
    if let Some(jitter) = stats.pitch_jitter_cents {
        if jitter > 90.0 {
            out.push(format!(
                "{prefix} pitch contour is shaky ({jitter:.0} cents median frame jitter)."
            ));
        }
    }
    if stats.silence_fraction > 0.95 {
        out.push(format!(
            "{prefix} is nearly all silent ({:.0}% silence).",
            stats.silence_fraction * 100.0
        ));
    }
}

fn collect_comparison_diagnostics(cmp: &ComparisonStats, out: &mut Vec<String>) {
    if let Some(drop) = cmp.voiced_dropout_fraction {
        if drop > 0.15 {
            out.push(format!(
                "High voiced dropout ({:.0}% of voiced input frames became unvoiced/raspy in output).",
                drop * 100.0
            ));
        }
    }
    if let Some(drop) = cmp.active_drop_fraction {
        if drop > 0.12 {
            out.push(format!(
                "{:.0}% of active speech was gated/dropped in output — lower S.Thresh.",
                drop * 100.0
            ));
        }
    }
    if cmp.brightness_delta_hz < -600.0 || cmp.rolloff_delta_hz < -1200.0 {
        out.push(format!(
            "Converted output is darker/muffled than input (centroid {:+.0} Hz, rolloff {:+.0} Hz).",
            cmp.brightness_delta_hz, cmp.rolloff_delta_hz
        ));
    }
    if let Some(std) = cmp.pitch_shift_std_semitones {
        if std > 3.0 {
            out.push(format!(
                "Inconsistent pitch shift (+/- {:.1} st std dev) — check F0 detector or Octave lock.",
                std
            ));
        }
    }
}

fn build_timeline(
    input: Option<&TrackFrames>,
    output: &TrackFrames,
    lag_frames: isize,
) -> Vec<TimelineBin> {
    let total = output.rms_frames.len().max(1);
    let step = total.div_ceil(MAX_TIMELINE_BINS).max(1);
    let mut bins = Vec::with_capacity(total.div_ceil(step));
    for start in (0..total).step_by(step) {
        let end = (start + step).min(total);
        let out_rms = average_slice(&output.rms_frames[start..end]);
        let out_centroid = average_slice(&output.centroid_frames[start..end]);
        let out_f0 = median_opt(&output.f0_frames[start..end]);
        let clicks: usize = output.click_frames[start..end].iter().sum();
        let clips: usize = output.clip_frames[start..end].iter().sum();

        let (in_rms_db, in_f0_hz) = if let Some(inp) = input {
            let mut rms_acc = Vec::new();
            let mut f0_acc = Vec::new();
            for i in start..end {
                let in_i = i as isize - lag_frames;
                if in_i >= 0 && (in_i as usize) < inp.rms_frames.len() {
                    let idx = in_i as usize;
                    rms_acc.push(inp.rms_frames[idx]);
                    f0_acc.push(inp.f0_frames[idx]);
                }
            }
            let r = (!rms_acc.is_empty()).then(|| to_db(average_slice(&rms_acc)));
            let f = median_opt(&f0_acc);
            (r, f)
        } else {
            (None, None)
        };

        bins.push(TimelineBin {
            t_s: start as f32 * 0.01,
            in_rms_db,
            out_rms_db: to_db(out_rms),
            in_f0_hz,
            out_f0_hz: out_f0,
            out_centroid_hz: out_centroid,
            clicks,
            clips,
        });
    }
    bins
}

fn trim_events(events: &mut Vec<DiagnosticEvent>) {
    events.sort_by(|a, b| {
        b.severity
            .total_cmp(&a.severity)
            .then_with(|| a.time_s.total_cmp(&b.time_s))
    });
    events.truncate(MAX_EVENTS);
}

fn compute_native_spectrum(samples: &[f32], rate: u32) -> (SpectralBands, f32, f32, f32) {
    let mut bands = SpectralBands {
        sub_80hz_pct: 0.0,
        body_80_300hz_pct: 0.0,
        low_mid_300_1khz_pct: 0.0,
        presence_1k_4khz_pct: 0.0,
        brilliance_4k_8khz_pct: 0.0,
        air_over_8khz_pct: 0.0,
    };
    if samples.is_empty() {
        return (bands, 0.0, 0.0, 1.0);
    }
    let nfft = 2048usize;
    let hann = hann_window(nfft);
    let hop = ((rate as usize) / 50).max(1); // 20 ms
    let mut avg = vec![0.0_f64; nfft / 2 + 1];
    let mut frames = 0usize;
    for center in (hop / 2..samples.len()).step_by(hop) {
        if window_rms(samples, center, hop) < SILENCE_RMS {
            continue;
        }
        let mag = stft_frame_size(samples, center, &hann, nfft);
        for (dst, &src) in avg.iter_mut().zip(mag.iter()) {
            *dst += f64::from(src);
        }
        frames += 1;
    }
    if frames == 0 {
        return (bands, 0.0, 0.0, 1.0);
    }
    for v in &mut avg {
        *v /= frames as f64;
    }

    let bin_hz = rate as f64 / nfft as f64;
    let mut total_power = 0.0_f64;
    let mut weighted_hz = 0.0_f64;
    let mut log_sum = 0.0_f64;
    let mut sub = 0.0_f64;
    let mut body = 0.0_f64;
    let mut low_mid = 0.0_f64;
    let mut presence = 0.0_f64;
    let mut brilliance = 0.0_f64;
    let mut air = 0.0_f64;

    for (bin, &mag) in avg.iter().enumerate().skip(1) {
        let hz = bin as f64 * bin_hz;
        let power = mag * mag;
        total_power += power;
        weighted_hz += hz * power;
        log_sum += (power + 1e-18).ln();
        if hz < 80.0 {
            sub += power;
        } else if hz < 300.0 {
            body += power;
        } else if hz < 1000.0 {
            low_mid += power;
        } else if hz < 4000.0 {
            presence += power;
        } else if hz < 8000.0 {
            brilliance += power;
        } else {
            air += power;
        }
    }
    if total_power <= 1e-18 {
        return (bands, 0.0, 0.0, 1.0);
    }

    let pct = |v: f64| (v * 100.0 / total_power) as f32;
    bands = SpectralBands {
        sub_80hz_pct: pct(sub),
        body_80_300hz_pct: pct(body),
        low_mid_300_1khz_pct: pct(low_mid),
        presence_1k_4khz_pct: pct(presence),
        brilliance_4k_8khz_pct: pct(brilliance),
        air_over_8khz_pct: pct(air),
    };
    let centroid = (weighted_hz / total_power) as f32;
    let mut cum = 0.0_f64;
    let mut rolloff = 0.0_f32;
    for (bin, &mag) in avg.iter().enumerate().skip(1) {
        cum += mag * mag;
        if cum >= 0.85 * total_power {
            rolloff = (bin as f64 * bin_hz) as f32;
            break;
        }
    }
    let bins = (avg.len() - 1) as f64;
    let flatness = ((log_sum / bins).exp() / (total_power / bins)) as f32;
    (bands, centroid, rolloff, flatness.clamp(0.0, 1.0))
}

fn yin_f0(audio: &[f32], center: usize) -> Option<f32> {
    let start = center.saturating_sub(YIN_WINDOW / 2);
    if start + YIN_WINDOW > audio.len() {
        return None;
    }
    let half = YIN_WINDOW / 2;
    let min_tau = 16usize; // 1000 Hz at 16 kHz
    let max_tau = half.min(291); // ~55 Hz at 16 kHz
    if max_tau <= min_tau + 2 {
        return None;
    }

    let slice = &audio[start..start + YIN_WINDOW];
    let head = &slice[..half];
    let mut cmnd = vec![1.0_f32; max_tau + 1];
    let mut running_sum = 0.0_f32;
    for tau in 1..=max_tau {
        let shifted = &slice[tau..tau + half];
        let diff: f32 = head
            .iter()
            .zip(shifted)
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum();
        running_sum += diff;
        cmnd[tau] = if running_sum > 1e-12 {
            diff * tau as f32 / running_sum
        } else {
            1.0
        };
    }

    let mut chosen = None;
    for tau in min_tau..max_tau {
        if cmnd[tau] < 0.15 {
            let mut best = tau;
            while best < max_tau && cmnd[best + 1] < cmnd[best] {
                best += 1;
            }
            // Disambiguate weak borderline 2nd-harmonic triggers only when 2 * tau is at least
            // 4x deeper (< 0.25 * cmnd[best]), avoiding false octave-halvings on clean fundamentals.
            if cmnd[best] >= 0.09 {
                let dbl = best * 2;
                if dbl <= max_tau {
                    let lo = dbl.saturating_sub(2).max(min_tau);
                    let hi = (dbl + 2).min(max_tau);
                    if let Some((offset, &dbl_val)) = cmnd[lo..=hi]
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(b.1))
                    {
                        if dbl_val < cmnd[best] * 0.25 {
                            best = lo + offset;
                        }
                    }
                }
            }
            chosen = Some(best);
            break;
        }
    }
    if chosen.is_none() {
        let (best_tau, &best_val) = cmnd[min_tau..=max_tau]
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))?;
        if best_val < 0.25 {
            chosen = Some(min_tau + best_tau);
        }
    }
    let tau = chosen?;
    let refined = if tau > min_tau && tau < max_tau {
        let a = cmnd[tau - 1];
        let b = cmnd[tau];
        let c = cmnd[tau + 1];
        let denom = 2.0 * (a - 2.0 * b + c);
        if denom.abs() > 1e-6 {
            tau as f32 + ((a - c) / denom).clamp(-0.5, 0.5)
        } else {
            tau as f32
        }
    } else {
        tau as f32
    };
    Some(ANALYSIS_RATE as f32 / refined)
}

fn hann_window(size: usize) -> Vec<f32> {
    let denom = (size.saturating_sub(1)).max(1) as f64;
    (0..size)
        .map(|i| (0.5 * (1.0 - (2.0 * PI * i as f64 / denom).cos())) as f32)
        .collect()
}

fn stft_frame(samples: &[f32], center: usize, hann: &[f32]) -> Vec<f32> {
    stft_frame_size(samples, center, hann, FFT_SIZE)
}

fn stft_frame_size(samples: &[f32], center: usize, hann: &[f32], nfft: usize) -> Vec<f32> {
    let half = nfft / 2;
    let mut re = vec![0.0_f64; nfft];
    let mut im = vec![0.0_f64; nfft];
    let scale = 2.0 / nfft as f64;
    for i in 0..nfft {
        let pos = center as isize - half as isize + i as isize;
        if pos >= 0 && (pos as usize) < samples.len() {
            re[i] = f64::from(samples[pos as usize] * hann[i]);
        }
    }
    fft_in_place(&mut re, &mut im);
    (0..=half)
        .map(|k| ((re[k] * re[k] + im[k] * im[k]).sqrt() * scale) as f32)
        .collect()
}

fn fft_in_place(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let angle = -2.0 * PI / len as f64;
        let wlen_re = angle.cos();
        let wlen_im = angle.sin();
        for start in (0..n).step_by(len) {
            let mut w_re = 1.0_f64;
            let mut w_im = 0.0_f64;
            for k in 0..(len / 2) {
                let u_re = re[start + k];
                let u_im = im[start + k];
                let v_re = re[start + k + len / 2] * w_re - im[start + k + len / 2] * w_im;
                let v_im = re[start + k + len / 2] * w_im + im[start + k + len / 2] * w_re;
                re[start + k] = u_re + v_re;
                im[start + k] = u_im + v_im;
                re[start + k + len / 2] = u_re - v_re;
                im[start + k + len / 2] = u_im - v_im;
                let next_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_re;
            }
        }
        len <<= 1;
    }
}

fn frame_centroid(mag: &[f32], rate: u32) -> f32 {
    let bin_hz = rate as f32 / ((mag.len() - 1) * 2) as f32;
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (i, &m) in mag.iter().enumerate().skip(1) {
        let p = f64::from(m) * f64::from(m);
        num += f64::from(i as f32 * bin_hz) * p;
        den += p;
    }
    if den > 1e-18 {
        (num / den) as f32
    } else {
        0.0
    }
}

fn window_rms(samples: &[f32], center: usize, width: usize) -> f32 {
    let half = width / 2;
    let start = center.saturating_sub(half);
    let end = (start + width).min(samples.len());
    if end <= start {
        return 0.0;
    }
    dsp::rms(&samples[start..end])
}

fn average_slice(slice: &[f32]) -> f32 {
    if slice.is_empty() {
        0.0
    } else {
        slice.iter().sum::<f32>() / slice.len() as f32
    }
}

fn median_opt(slice: &[Option<f32>]) -> Option<f32> {
    let mut vals: Vec<f32> = slice.iter().filter_map(|&v| v).collect();
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(f32::total_cmp);
    Some(vals[vals.len() / 2])
}

fn to_db(amp: f32) -> f32 {
    20.0 * amp.max(1e-6).log10()
}

#[cfg(any(feature = "gui", test))]
fn render_diagnostic_png(
    report: &AnalysisReport,
    input: Option<&TrackFrames>,
    output: &TrackFrames,
    path: &Path,
) -> Result<()> {
    use image::{Rgb, RgbImage};

    const W: u32 = 1200;
    const H: u32 = 860;
    const MARGIN_L: u32 = 68;
    const MARGIN_R: u32 = 24;
    const PLOT_W: u32 = W - MARGIN_L - MARGIN_R;

    let bg = Rgb([18, 19, 24]);
    let panel_bg = Rgb([26, 28, 36]);
    let border = Rgb([52, 56, 70]);
    let text = Rgb([228, 226, 232]);
    let muted = Rgb([152, 154, 170]);
    let cyan = Rgb([77, 208, 225]);
    let coral = Rgb([255, 138, 101]);
    let warn = Rgb([255, 82, 82]);
    let jump_col = Rgb([255, 213, 79]);

    let mut img = RgbImage::from_pixel(W, H, bg);

    // Header text
    draw_text(&mut img, 20, 12, &report.label, text, 2);
    let summary_line = match &report.comparison {
        Some(cmp) => {
            let shift = cmp
                .measured_pitch_shift_semitones
                .map(|s| format!("{s:+.2} ST"))
                .unwrap_or_else(|| "N/A".to_owned());
            let lat = cmp
                .estimated_latency_ms
                .map(|ms| format!("{ms:.0} MS"))
                .unwrap_or_else(|| "N/A".to_owned());
            let drop = cmp
                .voiced_dropout_fraction
                .map(|d| format!("{:.0}%", d * 100.0))
                .unwrap_or_else(|| "N/A".to_owned());
            format!(
                "DUR {:.2}S | SHIFT {shift} | LATENCY {lat} | DROPOUTS {drop} | CLIPS {} | CLICKS {}",
                output.duration_s, report.output.clipped_samples, report.output.click_count
            )
        }
        None => format!(
            "DUR {:.2}S | RMS {:.1} DB | F0 MED {:?} | CLIPS {} | CLICKS {}",
            output.duration_s,
            report.output.active_rms_db,
            report.output.f0_median_hz.map(|v| format!("{v:.0}HZ")),
            report.output.clipped_samples,
            report.output.click_count
        ),
    };
    draw_text(&mut img, 20, 32, &summary_line, muted, 1);
    let diag_y = if let Some(s) = &report.settings {
        let line = format!(
            "CFG: TUNE {:+.1} | BLK {}MS CTX {}MS OVL {}MS | F0 {:?} | IDX {:.0}% PROT {:.2} | STH {:.4} GAIN {:.2}/{:.2}",
            s.semitones,
            s.block_ms,
            s.context_ms,
            s.overlap_ms,
            s.f0_detector,
            s.index_ratio * 100.0,
            s.protect,
            s.silence_threshold,
            s.input_gain,
            s.output_gain,
        );
        draw_text(&mut img, 20, 44, &line, cyan, 1);
        56
    } else {
        48
    };
    if let Some(first_diag) = report.diagnostics.first() {
        let diag_color = if first_diag.contains("look healthy") || first_diag.contains("No major") {
            image::Rgb([129, 199, 132])
        } else {
            coral
        };
        draw_text(&mut img, 20, diag_y, first_diag, diag_color, 1);
    }

    let wave_top = 74u32;
    let wave_h = 160u32;
    let (in_spec_top, in_spec_h, out_spec_top, out_spec_h) = if input.is_some() {
        (252u32, 180u32, 450u32, 180u32)
    } else {
        (252u32, 0u32, 252u32, 378u32)
    };
    let pitch_top = 648u32;
    let pitch_h = 180u32;

    draw_panel(
        &mut img, MARGIN_L, wave_top, PLOT_W, wave_h, panel_bg, border,
    );
    draw_text(&mut img, 10, wave_top + 6, "RMS DB", muted, 1);
    draw_text(
        &mut img,
        MARGIN_L + 8,
        wave_top + 6,
        "ENVELOPE (CYAN=IN, CORAL=OUT, RED=CLICK/CLIP)",
        text,
        1,
    );

    let max_dur = output
        .duration_s
        .max(input.map_or(0.0, |t| t.duration_s))
        .max(0.05);

    // Draw RMS envelopes in -60..0 dBFS
    if let Some(inp) = input {
        draw_curve(
            &mut img,
            MARGIN_L,
            wave_top + 20,
            PLOT_W,
            wave_h - 26,
            &inp.rms_frames
                .iter()
                .map(|&r| Some(((to_db(r) + 60.0) / 60.0).clamp(0.0, 1.0)))
                .collect::<Vec<_>>(),
            cyan,
        );
    }
    draw_curve(
        &mut img,
        MARGIN_L,
        wave_top + 20,
        PLOT_W,
        wave_h - 26,
        &output
            .rms_frames
            .iter()
            .map(|&r| Some(((to_db(r) + 60.0) / 60.0).clamp(0.0, 1.0)))
            .collect::<Vec<_>>(),
        coral,
    );
    for ev in &report.events {
        if ev.kind == "click" || ev.kind == "clip" {
            let px =
                MARGIN_L + ((ev.time_s / max_dur).clamp(0.0, 1.0) * (PLOT_W - 1) as f32) as u32;
            draw_vline(&mut img, px, wave_top + 2, wave_top + wave_h - 2, warn);
        }
    }

    if let Some(inp) = input {
        draw_panel(
            &mut img,
            MARGIN_L,
            in_spec_top,
            PLOT_W,
            in_spec_h,
            panel_bg,
            border,
        );
        draw_spectrogram(
            &mut img,
            MARGIN_L + 1,
            in_spec_top + 1,
            PLOT_W - 2,
            in_spec_h - 2,
            &inp.spectrogram,
        );
        draw_text(&mut img, 10, in_spec_top + 6, "0-8KHZ", muted, 1);
        draw_text(
            &mut img,
            MARGIN_L + 8,
            in_spec_top + 6,
            "INPUT SPECTROGRAM",
            text,
            1,
        );
    }

    draw_panel(
        &mut img,
        MARGIN_L,
        out_spec_top,
        PLOT_W,
        out_spec_h,
        panel_bg,
        border,
    );
    draw_spectrogram(
        &mut img,
        MARGIN_L + 1,
        out_spec_top + 1,
        PLOT_W - 2,
        out_spec_h - 2,
        &output.spectrogram,
    );
    draw_text(&mut img, 10, out_spec_top + 6, "0-8KHZ", muted, 1);
    draw_text(
        &mut img,
        MARGIN_L + 8,
        out_spec_top + 6,
        "OUTPUT SPECTROGRAM",
        text,
        1,
    );

    draw_panel(
        &mut img, MARGIN_L, pitch_top, PLOT_W, pitch_h, panel_bg, border,
    );
    draw_text(&mut img, 10, pitch_top + 6, "F0 HZ", muted, 1);
    draw_text(
        &mut img,
        MARGIN_L + 8,
        pitch_top + 6,
        "PITCH CONTOUR (60-800 HZ LOG, YELLOW=OCTAVE JUMP)",
        text,
        1,
    );

    let norm_f0 = |f0: Option<f32>| -> Option<f32> {
        let hz = f0?;
        let lo = 60.0_f32.ln();
        let hi = 800.0_f32.ln();
        Some(((hz.clamp(60.0, 800.0).ln() - lo) / (hi - lo)).clamp(0.0, 1.0))
    };
    if let Some(inp) = input {
        let vals: Vec<_> = inp.f0_frames.iter().copied().map(norm_f0).collect();
        draw_curve(
            &mut img,
            MARGIN_L,
            pitch_top + 20,
            PLOT_W,
            pitch_h - 28,
            &vals,
            cyan,
        );
    }
    let out_vals: Vec<_> = output.f0_frames.iter().copied().map(norm_f0).collect();
    draw_curve(
        &mut img,
        MARGIN_L,
        pitch_top + 20,
        PLOT_W,
        pitch_h - 28,
        &out_vals,
        coral,
    );
    for ev in &report.events {
        if ev.kind == "octave_jump" {
            let px =
                MARGIN_L + ((ev.time_s / max_dur).clamp(0.0, 1.0) * (PLOT_W - 1) as f32) as u32;
            draw_vline(
                &mut img,
                px,
                pitch_top + 18,
                pitch_top + pitch_h - 4,
                jump_col,
            );
        }
    }

    // Time ticks along bottom
    for step in 0..=4 {
        let frac = step as f32 / 4.0;
        let x = MARGIN_L + (frac * (PLOT_W - 40) as f32) as u32;
        draw_text(
            &mut img,
            x,
            H - 18,
            &format!("{:.1}S", max_dur * frac),
            muted,
            1,
        );
    }

    img.save(path)
        .with_context(|| format!("Save diagnostic plot {}", path.display()))?;
    Ok(())
}

#[cfg(any(feature = "gui", test))]
fn draw_panel(
    img: &mut image::RgbImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    fill: image::Rgb<u8>,
    border: image::Rgb<u8>,
) {
    for yy in y..(y + h).min(img.height()) {
        for xx in x..(x + w).min(img.width()) {
            let is_edge = yy == y || yy + 1 == y + h || xx == x || xx + 1 == x + w;
            img.put_pixel(xx, yy, if is_edge { border } else { fill });
        }
    }
}

#[cfg(any(feature = "gui", test))]
fn draw_vline(img: &mut image::RgbImage, x: u32, y0: u32, y1: u32, color: image::Rgb<u8>) {
    if x >= img.width() {
        return;
    }
    for y in y0..=y1.min(img.height() - 1) {
        img.put_pixel(x, y, color);
    }
}

#[cfg(any(feature = "gui", test))]
fn draw_curve(
    img: &mut image::RgbImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    norm_vals: &[Option<f32>],
    color: image::Rgb<u8>,
) {
    if norm_vals.is_empty() || w < 2 || h < 2 {
        return;
    }
    let mut prev: Option<(i32, i32)> = None;
    for (idx, &val) in norm_vals.iter().enumerate() {
        let Some(v) = val else {
            prev = None;
            continue;
        };
        let px = x as i32 + ((idx as f32 / norm_vals.len().max(1) as f32) * (w - 1) as f32) as i32;
        let py = y as i32 + ((1.0 - v.clamp(0.0, 1.0)) * (h - 1) as f32) as i32;
        if let Some((lx, ly)) = prev {
            if (px - lx).abs() <= 16 {
                draw_line(img, lx, ly, px, py, color);
            }
        }
        draw_dot(img, px, py, color);
        prev = Some((px, py));
    }
}

#[cfg(any(feature = "gui", test))]
fn draw_dot(img: &mut image::RgbImage, cx: i32, cy: i32, color: image::Rgb<u8>) {
    for dy in -1..=1 {
        for dx in -1..=1 {
            let x = cx + dx;
            let y = cy + dy;
            if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }
}

#[cfg(any(feature = "gui", test))]
fn draw_line(img: &mut image::RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, color: image::Rgb<u8>) {
    let steps = (x1 - x0).abs().max((y1 - y0).abs()).max(1);
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let x = (x0 as f32 + (x1 - x0) as f32 * t).round() as i32;
        let y = (y0 as f32 + (y1 - y0) as f32 * t).round() as i32;
        if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
            img.put_pixel(x as u32, y as u32, color);
        }
    }
}

#[cfg(any(feature = "gui", test))]
fn draw_spectrogram(
    img: &mut image::RgbImage,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    frames: &[Vec<f32>],
) {
    if frames.is_empty() || w == 0 || h == 0 {
        return;
    }
    let bins = frames[0].len();
    let row_bins: Vec<usize> = (0..h)
        .map(|py| {
            let frac = 1.0 - (py as f32 / (h - 1).max(1) as f32);
            ((frac * (bins - 1) as f32).round() as usize).min(bins - 1)
        })
        .collect();
    let mut cached_fi = usize::MAX;
    let mut col_colors = vec![image::Rgb([14, 12, 28]); h as usize];
    for px in 0..w {
        let fi = ((px as usize * frames.len()) / w as usize).min(frames.len() - 1);
        if fi != cached_fi {
            cached_fi = fi;
            let spec = &frames[fi];
            for (py, &bi) in row_bins.iter().enumerate() {
                let db = (20.0 * spec[bi].max(1e-5).log10() + 85.0) / 70.0;
                col_colors[py] = colormap_inferno(db.clamp(0.0, 1.0));
            }
        }
        for (py, &color) in col_colors.iter().enumerate() {
            img.put_pixel(x + px, y + py as u32, color);
        }
    }
}

#[cfg(any(feature = "gui", test))]
fn colormap_inferno(t: f32) -> image::Rgb<u8> {
    let stops: [(f32, [f32; 3]); 5] = [
        (0.0, [14.0, 12.0, 28.0]),
        (0.25, [76.0, 22.0, 108.0]),
        (0.5, [182.0, 54.0, 84.0]),
        (0.75, [242.0, 134.0, 34.0]),
        (1.0, [252.0, 248.0, 164.0]),
    ];
    for w in stops.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let k = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            return image::Rgb([
                (c0[0] + (c1[0] - c0[0]) * k) as u8,
                (c0[1] + (c1[1] - c0[1]) * k) as u8,
                (c0[2] + (c1[2] - c0[2]) * k) as u8,
            ]);
        }
    }
    image::Rgb([252, 248, 164])
}

#[cfg(any(feature = "gui", test))]
fn draw_text(
    img: &mut image::RgbImage,
    mut x: u32,
    y: u32,
    text: &str,
    color: image::Rgb<u8>,
    scale: u32,
) {
    for ch in text.chars() {
        let glyph = glyph_5x7(ch.to_ascii_uppercase());
        for (row, &bits) in glyph.iter().enumerate() {
            for col in 0..5u32 {
                if (bits >> (4 - col)) & 1 == 1 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = x + col * scale + sx;
                            let py = y + row as u32 * scale + sy;
                            if px < img.width() && py < img.height() {
                                img.put_pixel(px, py, color);
                            }
                        }
                    }
                }
            }
        }
        x += 6 * scale;
        if x + 6 * scale >= img.width() {
            break;
        }
    }
}

#[cfg(any(feature = "gui", test))]
fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x08],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '+' => [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
        '/' => [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
        '=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        '|' => [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '%' => [0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F],
        '>' => [0x10, 0x08, 0x04, 0x02, 0x04, 0x08, 0x10],
        '<' => [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
        _ => [0x00; 7],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq_hz: f32, rate: u32, seconds: f32, amp: f32) -> Vec<f32> {
        let len = (rate as f32 * seconds) as usize;
        (0..len)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq_hz * i as f32 / rate as f32).sin())
            .collect()
    }

    #[test]
    fn yin_and_pair_comparison_measure_octave_shift_and_latency() {
        let rate = 48_000_u32;
        let mut input = vec![0.0_f32; rate as usize / 10]; // 100 ms lead silence
        input.extend(sine_wave(220.0, rate, 0.8, 0.25));
        let mut output = vec![0.0_f32; rate as usize / 5]; // 200 ms latency -> +100 ms lag
        output.extend(sine_wave(440.0, rate, 0.8, 0.25));

        let report = analyze_pair_samples(&input, rate, &output, rate, "pitch-test");
        let cmp = report.comparison.expect("comparison stats");
        let shift = cmp.measured_pitch_shift_semitones.expect("shift");
        assert!((shift - 12.0).abs() < 0.35, "expected +12 st, got {shift}");
        let lat = cmp.estimated_latency_ms.expect("latency");
        assert!(
            (lat - 100.0).abs() <= 20.0,
            "expected ~100 ms lag, got {lat}"
        );
        assert_eq!(report.output.clipped_samples, 0);
        assert_eq!(report.output.click_count, 0);
    }

    #[test]
    fn detects_clicks_clipping_and_muffling_and_writes_reports() {
        let rate = 48_000;
        let input = sine_wave(1500.0, rate, 0.5, 0.25);
        let mut output = sine_wave(220.0, rate, 0.5, 0.25);
        // Inject a hard discontinuity click and a clipped sample.
        output[rate as usize / 4] = 1.0;
        output[rate as usize / 4 + 1] = -0.5;

        let report = analyze_pair_samples(&input, rate, &output, rate, "artifact-test");
        assert!(report.output.clipped_samples >= 1);
        assert!(report.output.click_count >= 1);
        assert!(report.comparison.as_ref().unwrap().brightness_delta_hz < -500.0);

        let dir = std::env::temp_dir().join("voice_changer_analyze_test");
        let prefix = dir.join("unit");
        let (json_path, png_path) =
            write_reports(&report, &prefix, Some((&input, rate)), (&output, rate)).unwrap();
        assert!(json_path.is_file());
        let png = png_path.expect("png generated in tests");
        assert!(png.is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn analyze_wavs_to_disk_reads_distinct_input_and_output_files() {
        let dir = std::env::temp_dir().join("voice_changer_analyze_wav_pair_test");
        std::fs::create_dir_all(&dir).unwrap();
        let in_wav = dir.join("in.wav");
        let out_wav = dir.join("out.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let write_tone = |path: &Path, hz: f32| {
            let mut w = hound::WavWriter::create(path, spec).unwrap();
            for s in sine_wave(hz, 16_000, 0.5, 0.25) {
                w.write_sample((s * 32767.0).round() as i16).unwrap();
            }
            w.finalize().unwrap();
        };
        write_tone(&in_wav, 220.0);
        write_tone(&out_wav, 440.0);

        let prefix = dir.join("pair");
        let (report, json_path, png_path) =
            analyze_wavs_to_disk(&in_wav, Some(&out_wav), &prefix).unwrap();
        let shift = report
            .comparison
            .and_then(|c| c.measured_pitch_shift_semitones)
            .expect("pitch shift");
        assert!((shift - 12.0).abs() < 0.35, "expected +12 st, got {shift}");
        assert!(json_path.is_file());
        assert!(png_path.unwrap().is_file());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ignores_sibilant_burst_and_repairs_isolated_yin_halvings() {
        let mut f0 = vec![
            None,
            Some(95.0),
            Some(190.0),
            Some(192.0),
            Some(96.0),
            Some(191.0),
            Some(380.0),
            Some(382.0),
            Some(191.0),
            None,
        ];
        smooth_isolated_f0_glitches(&mut f0);
        // Onset 1-frame halving (95.0) and interior 1-frame halving (96.0) are smoothed,
        // while the real >=2-frame octave jump (380, 382) and trailing edge stay intact.
        assert_eq!(f0[1], Some(190.0));
        assert!((f0[4].unwrap() - 191.5).abs() < 0.01);
        assert_eq!(f0[6], Some(380.0));
        assert_eq!(f0[7], Some(382.0));

        // High-frequency 7.5 kHz burst across a 10 ms bin boundary should not count as a click.
        let rate = 48_000_u32;
        let mut sig = vec![0.0_f32; rate as usize / 2];
        let burst_start = 480 * 10 + 420;
        for (k, slot) in sig[burst_start..burst_start + 360].iter_mut().enumerate() {
            *slot = 0.22 * (2.0 * std::f32::consts::PI * 7500.0 * k as f32 / rate as f32).sin();
        }
        let report = analyze_samples(&sig, rate, "sibilant-test");
        assert_eq!(report.output.click_count, 0);

        // Dominant 2nd harmonic (strong 360 Hz + weaker 180 Hz fundamental) must resolve to 180 Hz,
        // not falsely lock onto 2*f0 = 360 Hz.
        let harm: Vec<f32> = (0..ANALYSIS_RATE as usize / 2)
            .map(|i| {
                let t = i as f32 / ANALYSIS_RATE as f32;
                0.28 * (2.0 * std::f32::consts::PI * 180.0 * t).sin()
                    + 0.72 * (2.0 * std::f32::consts::PI * 360.0 * t).sin()
            })
            .collect();
        let f0 = yin_f0(&harm, ANALYSIS_RATE as usize / 4).expect("detected fundamental");
        assert!((f0 - 180.0).abs() < 4.0, "expected ~180 Hz f0, got {f0}");
    }
}
