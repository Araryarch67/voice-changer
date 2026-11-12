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

    #[test]
    fn silence_is_unvoiced() {
        assert!(yin(&[0.0; 16000], 100).iter().all(|v| *v == 0.0));
        assert_eq!(yin(&[], 3), vec![0.0; 3]);
    }
}
