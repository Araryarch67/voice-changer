//! Built-in pitch tracking that needs no model file.

const RATE: f32 = 16000.0;
const HOP: usize = 160;
const WINDOW: usize = 640;
/// Lag range for 50 Hz to 1100 Hz at 16 kHz.
const MIN_LAG: usize = 14;
const MAX_LAG: usize = 320;

/// YIN pitch tracking on 16 kHz audio: one value in Hz per 10 ms frame, 0 for unvoiced.
/// Less robust than RMVPE on noisy or breathy input.
pub fn yin(audio: &[f32], frames: usize) -> Vec<f32> {
    let margin = (WINDOW + MAX_LAG) / 2;
    let mut padded = vec![0.0; margin];
    padded.extend_from_slice(audio);
    padded.resize(padded.len().max(frames * HOP + margin) + margin, 0.0);
    let mut difference = [0.0_f32; MAX_LAG + 1];
    let mut pitch: Vec<f32> = (0..frames)
        .map(|frame| {
            // Centre the analysis span on the frame.
            let start = frame * HOP + HOP / 2;
            let span = &padded[start..start + WINDOW + MAX_LAG];
            let energy = span[..WINDOW].iter().map(|s| s * s).sum::<f32>() / WINDOW as f32;
            if energy < 1e-6 {
                return 0.0;
            }
            for (lag, slot) in difference.iter_mut().enumerate().skip(1) {
                *slot = span[..WINDOW]
                    .iter()
                    .zip(&span[lag..lag + WINDOW])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
            }
            // Cumulative mean normalised difference.
            let mut running = 0.0;
            for (lag, slot) in difference.iter_mut().enumerate().skip(1) {
                running += *slot;
                *slot = if running > 0.0 {
                    *slot * lag as f32 / running
                } else {
                    1.0
                };
            }
            let Some(mut lag) = (MIN_LAG..MAX_LAG).find(|lag| difference[*lag] < 0.15) else {
                return 0.0;
            };
            while lag + 1 < MAX_LAG && difference[lag + 1] < difference[lag] {
                lag += 1;
            }
            // Parabolic interpolation around the dip for sub-sample precision.
            let (before, at, after) = (difference[lag - 1], difference[lag], difference[lag + 1]);
            let curve = before - 2.0 * at + after;
            let shift = if curve.abs() > 1e-9 {
                (0.5 * (before - after) / curve).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            RATE / (lag as f32 + shift)
        })
        .collect();
    // A three-frame median removes isolated octave jumps.
    let raw = pitch.clone();
    for i in 1..frames.saturating_sub(1) {
        let mut window = [raw[i - 1], raw[i], raw[i + 1]];
        window.sort_by(f32::total_cmp);
        if raw[i] > 0.0 && window[1] > 0.0 {
            pitch[i] = window[1];
        }
    }
    pitch
}

const CREPE_WINDOW: usize = 1024;
const CREPE_BINS: usize = 360;
/// CREPE's bin 0 in cents above 10 Hz; bins are 20 cents apart.
const CREPE_CENTS_OFFSET: f64 = 1_997.379_408_437_619;

/// CREPE model input for 16 kHz audio: one centred 1024-sample frame per 10 ms, as a
/// flat `[count, 1024]` matrix. The ONNX graph normalises the frames itself.
pub fn crepe_frames(audio: &[f32]) -> (Vec<f32>, usize) {
    let mut padded = vec![0.0; CREPE_WINDOW / 2];
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + CREPE_WINDOW / 2, 0.0);
    let count = 1 + (padded.len() - CREPE_WINDOW) / HOP;
    let frames = (0..count)
        .flat_map(|frame| {
            padded[frame * HOP..frame * HOP + CREPE_WINDOW]
                .iter()
                .copied()
        })
        .collect();
    (frames, count)
}

/// Decodes CREPE's `[count, 360]` output as the reference does: weighted average
/// around the strongest bin within 50 Hz to 1100 Hz, a three-frame median, and
/// unvoiced (0) where the peak activation is under 0.1.
pub fn crepe_decode(activations: &[f32]) -> Vec<f32> {
    let bin_of = |hz: f64| (1200.0 * (hz / 10.0).log2() - CREPE_CENTS_OFFSET) / 20.0;
    let (low, high) = (
        bin_of(50.0).floor() as usize,
        bin_of(1100.0).ceil() as usize,
    );
    let (mut pitch, mut confidence) = (Vec::new(), Vec::new());
    for frame in activations.as_chunks::<CREPE_BINS>().0 {
        let peak = (low..high)
            .max_by(|a, b| frame[*a].total_cmp(&frame[*b]))
            .unwrap_or(low);
        let (mut weighted, mut total) = (0.0_f64, 0.0_f64);
        let window = peak.saturating_sub(4).max(low)..(peak + 5).min(high);
        for (bin, activation) in frame.iter().enumerate().take(window.end).skip(window.start) {
            let weight = f64::from(activation.max(0.0));
            weighted += (20.0 * bin as f64 + CREPE_CENTS_OFFSET) * weight;
            total += weight;
        }
        pitch.push(if total > 0.0 {
            (10.0 * 2.0_f64.powf(weighted / total / 1200.0)) as f32
        } else {
            0.0
        });
        confidence.push(frame[peak]);
    }
    let (pitch, confidence) = (median3(&pitch, false), median3(&confidence, false));
    pitch
        .iter()
        .zip(&confidence)
        .map(|(hz, sure)| if *sure < 0.1 { 0.0 } else { *hz })
        .collect()
}

const FCPE_BINS: usize = 360;
/// FCPE's peak salience under which a frame is unvoiced: torchfcpe's default, which
/// the reference uses.
const FCPE_THRESHOLD: f32 = 0.006;

/// Decodes FCPE's `[frames, 360]` salience as torchfcpe's `local_argmax` decoder
/// does for the reference: the salience-weighted mean of the nine bins around the
/// peak, bins evenly spaced in cents from 32.7 Hz to 1975.5 Hz, and unvoiced (0)
/// where the peak is under 0.006. No smoothing, as in the reference.
pub fn fcpe_decode(salience: &[f32]) -> Vec<f32> {
    let cents = |hz: f64| 1200.0 * (hz / 10.0).log2();
    let (first, last) = (cents(32.7), cents(1975.5));
    let bin_cents = |bin: usize| first + (last - first) * bin as f64 / (FCPE_BINS - 1) as f64;
    salience
        .as_chunks::<FCPE_BINS>()
        .0
        .iter()
        .map(|frame| {
            let finite = |v: f32| if v.is_finite() { v } else { 0.0 };
            let peak = (0..FCPE_BINS)
                .max_by(|a, b| finite(frame[*a]).total_cmp(&finite(frame[*b])))
                .unwrap_or(0);
            if finite(frame[peak]) < FCPE_THRESHOLD {
                return 0.0;
            }
            // As torch.gather with clamped indices: an edge bin may count twice.
            let (mut weighted, mut total) = (0.0_f64, 0.0_f64);
            for offset in 0..9 {
                let bin = (peak + offset).saturating_sub(4).min(FCPE_BINS - 1);
                let weight = f64::from(finite(frame[bin]));
                weighted += bin_cents(bin) * weight;
                total += weight;
            }
            (10.0 * 2.0_f64.powf(weighted / total / 1200.0)) as f32
        })
        .collect()
}

/// Three-point median. Edges either use the two values there are, or are padded
/// with zeros like `scipy.signal.medfilt`.
fn median3(values: &[f32], zero_padded: bool) -> Vec<f32> {
    (0..values.len())
        .map(|i| {
            let before = i.checked_sub(1).map(|j| values[j]);
            let after = values.get(i + 1).copied();
            match (before, after) {
                (Some(a), Some(b)) => {
                    let mut window = [a, values[i], b];
                    window.sort_by(f32::total_cmp);
                    window[1]
                }
                (edge, other) if zero_padded => {
                    let mut window = [0.0, values[i], edge.or(other).unwrap_or(0.0)];
                    window.sort_by(f32::total_cmp);
                    window[1]
                }
                (edge, other) => match edge.or(other) {
                    Some(neighbour) => (values[i] + neighbour) / 2.0,
                    None => values[i],
                },
            }
        })
        .collect()
}

/// Stabilizes an F0 contour in Hz (`0.0` = unvoiced): preserves all unvoiced
/// `0.0` frames, suppresses isolated interior 1-frame voiced noise blips, and
/// smooths spurious single/double-frame pitch spikes and octave jumps without
/// blurring monotonic pitch slides or normal vibrato.
pub fn stabilize_f0(f0: &[f32]) -> Vec<f32> {
    let len = f0.len();
    let mut out = Vec::with_capacity(len);
    for (i, &cur) in f0.iter().enumerate() {
        if !cur.is_finite() || cur <= 0.0 {
            out.push(0.0);
            continue;
        }
        let start = i.saturating_sub(2);
        let end = (i + 3).min(len);
        let mut voiced = [0.0_f32; 5];
        let mut n = 0;
        for &v in &f0[start..end] {
            if v.is_finite() && v > 0.0 {
                voiced[n] = v;
                n += 1;
            }
        }
        // Mute isolated interior 1-frame voiced blips surrounded by silence/unvoiced.
        if n == 1 && i > 0 && i + 1 < len {
            out.push(0.0);
            continue;
        }
        let mut fixed = cur;
        let prev = i
            .checked_sub(1)
            .map(|j| f0[j])
            .filter(|v| v.is_finite() && *v > 0.0);
        let next = f0.get(i + 1).copied().filter(|v| v.is_finite() && *v > 0.0);
        if let (Some(a), Some(b)) = (prev, next) {
            let mut w3 = [a, cur, b];
            w3.sort_by(f32::total_cmp);
            let med3 = w3[1];
            let ratio3 = (cur / med3).max(med3 / cur);
            if ratio3 > 1.15 {
                fixed = med3;
            }
        }
        if n >= 3 {
            voiced[..n].sort_by(f32::total_cmp);
            let med5 = voiced[n / 2];
            let ratio5 = (fixed / med5).max(med5 / fixed);
            if ratio5 > 1.4 {
                fixed = med5;
            }
        }
        out.push(fixed);
    }
    out
}

/// WORLD's Harvest with StoneMask refinement and a three-frame median, on 16 kHz
/// audio: one value per 10 ms frame, 0 for unvoiced. Accurate and slow.
#[cfg(feature = "world")]
pub fn harvest(audio: &[f32]) -> Vec<f32> {
    let Some(samples) = world_input(audio) else {
        return vec![0.0; audio.len() / HOP + 1];
    };
    let mut option = rsworld_sys::HarvestOption::new();
    option.f0_ceil = 1100.0;
    option.frame_period = 10.0;
    let (times, coarse) = rsworld::harvest(&samples, RATE as i32, &option);
    let refined = rsworld::stonemask(&samples, RATE as i32, &times, &coarse);
    let refined: Vec<f32> = refined.iter().map(|hz| *hz as f32).collect();
    median3(&refined, true)
}

/// WORLD's DIO with StoneMask refinement, on 16 kHz audio. Fast and rough.
#[cfg(feature = "world")]
pub fn dio(audio: &[f32]) -> Vec<f32> {
    let Some(samples) = world_input(audio) else {
        return vec![0.0; audio.len() / HOP + 1];
    };
    let mut option = rsworld_sys::DioOption::new();
    option.f0_floor = 50.0;
    option.f0_ceil = 1100.0;
    option.channels_in_octave = 2.0;
    option.frame_period = 10.0;
    let (times, coarse) = rsworld::dio(&samples, RATE as i32, &option);
    rsworld::stonemask(&samples, RATE as i32, &times, &coarse)
        .iter()
        .map(|hz| *hz as f32)
        .collect()
}

/// WORLD needs a few analysis periods of signal; shorter input is treated as unvoiced.
#[cfg(feature = "world")]
fn world_input(audio: &[f32]) -> Option<Vec<f64>> {
    (audio.len() >= 1600).then(|| audio.iter().map(|s| f64::from(*s)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_a_sine() {
        for frequency in [110.0_f32, 220.0, 440.0] {
            let audio: Vec<f32> = (0..16000)
                .map(|i| (2.0 * std::f32::consts::PI * frequency * i as f32 / RATE).sin() * 0.3)
                .collect();
            let pitch = yin(&audio, 100);
            assert_eq!(pitch.len(), 100);
            for value in &pitch[5..95] {
                assert!(
                    (value - frequency).abs() < frequency * 0.01,
                    "{value} vs {frequency}"
                );
            }
        }
    }

    /// Harvest returns nothing for a pure sine (so does the reference library), so the
    /// WORLD test uses a tone with harmonics, like a voice.
    #[cfg(feature = "world")]
    #[test]
    fn world_detectors_track_a_harmonic_tone() {
        for frequency in [110.0_f32, 220.0, 440.0] {
            let tone: Vec<f32> = (0..16000)
                .map(|i| {
                    let phase = 2.0 * std::f32::consts::PI * frequency * i as f32 / RATE;
                    (1..=8)
                        .map(|k| (phase * k as f32).sin() / k as f32)
                        .sum::<f32>()
                        * 0.2
                })
                .collect();
            for (name, pitch) in [("harvest", harvest(&tone)), ("dio", dio(&tone))] {
                assert_eq!(pitch.len(), 101, "{name}");
                for value in &pitch[10..90] {
                    assert!(
                        (value - frequency).abs() < frequency * 0.02,
                        "{name}: {value} vs {frequency}"
                    );
                }
            }
        }
        assert!(harvest(&[0.0; 100]).iter().all(|v| *v == 0.0));
    }

    #[test]
    fn fcpe_decoding_matches_the_bin_scale() {
        // A peak on bin 0 and bin 359 decodes to the ends of FCPE's range; a peak
        // between two bins lands between their pitches; a weak frame is unvoiced.
        let frame = |bins: &[(usize, f32)]| {
            let mut frame = vec![0.0; FCPE_BINS];
            for &(bin, value) in bins {
                frame[bin] = value;
            }
            frame
        };
        let decoded = fcpe_decode(
            &[
                frame(&[(0, 0.9)]),
                frame(&[(359, 0.9)]),
                frame(&[(180, 0.8), (181, 0.8)]),
                frame(&[(200, 0.005)]),
            ]
            .concat(),
        );
        assert!((decoded[0] - 32.7).abs() < 0.01, "{decoded:?}");
        assert!((decoded[1] - 1975.5).abs() < 0.1, "{decoded:?}");
        let between = (
            fcpe_decode(&frame(&[(180, 0.9)]))[0],
            fcpe_decode(&frame(&[(181, 0.9)]))[0],
        );
        assert!(
            between.0 < decoded[2] && decoded[2] < between.1,
            "{decoded:?} {between:?}"
        );
        assert_eq!(decoded[3], 0.0);
        assert_eq!(fcpe_decode(&frame(&[(100, f32::NAN)])), vec![0.0]);
    }

    #[test]
    fn crepe_decoding_matches_the_bin_scale() {
        let (frames, count) = crepe_frames(&[0.0; 16000]);
        assert_eq!((count, frames.len()), (101, 101 * CREPE_WINDOW));
        // Bin 120 is 20 * 120 + 1997.38 cents above 10 Hz, about 126.8 Hz.
        let expected = 10.0 * 2.0_f32.powf((20.0 * 120.0 + 1_997.379_4) / 1200.0);
        let mut frame = vec![0.0; CREPE_BINS];
        frame[119] = 0.5;
        frame[120] = 1.0;
        frame[121] = 0.5;
        let voiced: Vec<f32> = frame.iter().copied().cycle().take(CREPE_BINS * 5).collect();
        for value in crepe_decode(&voiced) {
            assert!((value - expected).abs() < 0.01, "{value} vs {expected}");
        }
        // Weak activations are unvoiced, and peaks outside 50..1100 Hz are ignored.
        let mut weak = vec![0.0; CREPE_BINS];
        weak[120] = 0.05;
        weak[350] = 1.0;
        assert_eq!(crepe_decode(&weak), vec![0.0]);
    }

    #[test]
    fn silence_is_unvoiced() {
        assert!(yin(&[0.0; 16000], 100).iter().all(|v| *v == 0.0));
        assert_eq!(yin(&[], 3), vec![0.0; 3]);
    }

    #[test]
    fn stabilize_f0_removes_spikes_while_preserving_unvoiced_and_slides() {
        assert!(stabilize_f0(&[]).is_empty());
        assert_eq!(stabilize_f0(&[220.0]), vec![220.0]);

        // Unvoiced 0.0 frames are preserved, and isolated interior 1-frame voiced blips are muted.
        let blips = [0.0, 0.0, 320.0, 0.0, 0.0, 200.0, 202.0, 0.0];
        assert_eq!(
            stabilize_f0(&blips),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 200.0, 202.0, 0.0]
        );

        // Single-frame and double-frame octave spikes are snapped without blurring monotonic slides.
        let spiked = [
            0.0, 200.0, 205.0, 410.0, 210.0, 215.0, 430.0, 434.0, 220.0, 225.0, 0.0,
        ];
        let fixed = stabilize_f0(&spiked);
        assert_eq!(fixed[0], 0.0);
        assert_eq!(fixed[10], 0.0);
        assert!((fixed[3] - 210.0).abs() < 1e-3, "1-frame spike: {fixed:?}");
        assert!(
            fixed[6] < 230.0 && fixed[7] < 230.0,
            "2-frame spike: {fixed:?}"
        );

        // Monotonic pitch slide is preserved unchanged.
        let slide = [0.0, 180.0, 195.0, 210.0, 228.0, 246.0, 265.0, 0.0];
        assert_eq!(stabilize_f0(&slide), slide);

        // Gentle vibrato is preserved unchanged.
        let vibrato = [200.0, 208.0, 216.0, 208.0, 200.0, 192.0, 200.0];
        assert_eq!(stabilize_f0(&vibrato), vibrato);
    }
}
