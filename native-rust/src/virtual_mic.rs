//! Built-in virtual cable for Linux: a "Voice Changer Output" sink whose monitor is
//! exposed as a "Voice Changer Mic" source, so Discord, OBS or a game can pick the
//! converted voice as a microphone. Uses `pactl`, which works with PipeWire
//! (pipewire-pulse) and PulseAudio alike.
use anyhow::{bail, Context, Result};
use std::process::Command;

pub const SINK: &str = "voice_changer_output";
pub const SOURCE: &str = "voice_changer_mic";

fn pactl(args: &[&str]) -> Result<String> {
    let output = Command::new("pactl")
        .args(args)
        // Parsed output must not be translated.
        .env("LC_ALL", "C")
        .output()
        .context(
            "The virtual microphone needs pactl; install pipewire-pulse (or PulseAudio) and its tools",
        )?;
    if !output.status.success() {
        bail!(
            "pactl {} failed: {}",
            args.first().copied().unwrap_or_default(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// `key="value"` for a sink or source property list, quoted so spaces survive both
/// PulseAudio's and PipeWire's argument parsers.
fn description(kind: &str, text: &str) -> String {
    format!("{kind}_properties=\"device.description=\\\"{text}\\\"\"")
}

/// Makes sure Voice Changer Output and Voice Changer Mic exist. Devices already there
/// are kept: they stay after the app closes, because an app such as Discord that loses
/// its chosen microphone falls back to the real one and does not switch back.
pub fn ensure() -> Result<()> {
    let modules = pactl(&["list", "short", "modules"])?;
    let has = |arg: String| modules.lines().any(|line| line.contains(&arg));
    let (sink, source) = (
        has(format!("sink_name={SINK}")),
        has(format!("source_name={SOURCE}")),
    );
    if sink && source {
        log::info!("Voice Changer Mic is ready (kept from before)");
        return Ok(());
    }
    log::info!("Creating Voice Changer Output and Voice Changer Mic");
    // Half a pair is rebuilt whole, so the microphone always reads this sink.
    remove();
    load(&[
        "module-null-sink",
        &format!("sink_name={SINK}"),
        &description("sink", "Voice Changer Output"),
    ])?;
    load(&[
        "module-remap-source",
        &format!("master={SINK}.monitor"),
        &format!("source_name={SOURCE}"),
        &description("source", "Voice Changer Mic"),
    ])
}

fn load(args: &[&str]) -> Result<()> {
    let mut command = vec!["load-module"];
    command.extend_from_slice(args);
    let id = pactl(&command)?;
    id.trim()
        .parse::<u32>()
        .with_context(|| format!("pactl returned no module id: {id}"))?;
    Ok(())
}

/// Removes both devices: when the option is turned off, or left over while it is off.
pub fn remove() {
    if let Ok(list) = pactl(&["list", "short", "modules"]) {
        let ids = removal_order(&list);
        if !ids.is_empty() {
            log::info!("Removing Voice Changer Mic");
        }
        for id in ids {
            let _ = pactl(&["unload-module", &id.to_string()]);
        }
    }
}

/// The microphone first: while it exists without its sink, the sound server would
/// connect it to the real microphone. Module ids do not say which was loaded first.
fn removal_order(list: &str) -> Vec<u32> {
    let mut ids: Vec<(bool, u32)> = list
        .lines()
        .filter_map(|line| {
            let id = line.split('\t').next()?.trim().parse().ok()?;
            let ours = line.contains(&format!("sink_name={SINK}"))
                || line.contains(&format!("source_name={SOURCE}"));
            ours.then(|| (!line.contains("module-remap-source"), id))
        })
        .collect();
    ids.sort_unstable();
    ids.into_iter().map(|(_, id)| id).collect()
}

/// Names the app's playback streams carry, so the sound server keeps their routes
/// apart: the converted voice and the monitor.
pub const PLAYBACK_NAME: &str = "Voice Changer";
pub const MONITOR_NAME: &str = "Voice Changer monitor";

/// Which stream is about to be opened.
pub enum Stream {
    /// The converted voice; `true` sends it straight to the virtual microphone.
    Output(bool),
    Monitor,
}

/// Runs `open` (which opens one ALSA stream) with the environment that names that
/// stream and, for the virtual microphone, targets it from the start. PipeWire's ALSA
/// plugin reads `PIPEWIRE_ALSA` and `PIPEWIRE_NODE`, PulseAudio's reads `PULSE_PROP`
/// and `PULSE_SINK`, both when the stream opens; the previous values are restored.
pub fn open_stream<T>(stream: Stream, open: impl FnOnce() -> T) -> T {
    let (name, node, target) = match stream {
        Stream::Output(to_mic) => (PLAYBACK_NAME, "voice_changer_playback", to_mic),
        Stream::Monitor => (MONITOR_NAME, "voice_changer_monitor", false),
    };
    let mut values = vec![
        (
            "PIPEWIRE_ALSA",
            Some(format!(
                "{{ application.name = \"{name}\" node.name = \"{node}\" node.description = \"{name}\" }}"
            )),
        ),
        ("PULSE_PROP", Some(format!("application.name='{name}'"))),
        ("PIPEWIRE_NODE", target.then(|| SINK.to_owned())),
        ("PULSE_SINK", target.then(|| SINK.to_owned())),
    ];
    let previous: Vec<_> = values
        .iter()
        .map(|(key, _)| (*key, std::env::var_os(key)))
        .collect();
    // Only the audio worker sets these, right around opening its own streams.
    for (key, value) in values.drain(..) {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
    let result = open();
    for (key, value) in previous {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
    result
}

/// Puts the converted voice back on the virtual sink if anything moved it (a device
/// change, the sound server restoring an old route, the user). Returns how many of the
/// app's voice streams exist.
pub fn ensure_routed() -> Result<usize> {
    let sink = pactl(&["list", "short", "sinks"])?
        .lines()
        .find(|line| line.split('\t').nth(1) == Some(SINK))
        .and_then(|line| line.split('\t').next()?.parse::<u32>().ok())
        .context("The Voice Changer Mic device is gone; turn the virtual microphone off and on")?;
    let streams = voice_streams(&pactl(&["list", "sink-inputs"])?);
    for (index, on) in &streams {
        if *on != sink {
            log::warn!("The voice stream left Voice Changer Mic; moving it back");
            pactl(&["move-sink-input", &index.to_string(), SINK])?;
        }
    }
    keep_mic_on_sink()?;
    Ok(streams.len())
}

/// Checks Voice Changer Mic every second while it lives, so it never carries another
/// source even when no session is running. Dropping it stops the checks.
pub struct Guard {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The latest problem, taken by the UI.
    pub error: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}
impl Guard {
    pub fn start() -> Result<Self> {
        use std::sync::{atomic::AtomicBool, Arc, Mutex};
        let stop = Arc::new(AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None));
        let (thread_stop, thread_error) = (stop.clone(), error.clone());
        std::thread::Builder::new()
            .name("virtual-mic-guard".into())
            .spawn(move || {
                let mut reported = false;
                while !thread_stop.load(std::sync::atomic::Ordering::Acquire) {
                    match keep_mic_on_sink() {
                        Ok(()) => reported = false,
                        Err(problem) if !reported => {
                            reported = true;
                            *thread_error.lock().unwrap() = Some(format!("{problem:#}"));
                        }
                        Err(_) => {}
                    }
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
            })?;
        Ok(Self { stop, error })
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// On PipeWire the microphone is a loopback whose capture side is an ordinary stream.
/// If the sound server re-links it, it lands on the default (real) microphone, and
/// Voice Changer Mic would carry the raw voice; it is put back on the sink's monitor.
pub fn keep_mic_on_sink() -> Result<()> {
    static MUTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    use std::sync::atomic::Ordering;
    let monitor = format!("{SINK}.monitor");
    let monitor_index = pactl(&["list", "short", "sources"])?
        .lines()
        .find(|line| line.split('\t').nth(1) == Some(monitor.as_str()))
        .and_then(|line| line.split('\t').next()?.parse::<u32>().ok());
    let mut result = Ok(());
    for (index, source) in loopback_captures(&pactl(&["list", "source-outputs"])?) {
        // Not yet linked (no source) is harmless; any other source is not.
        if source == u32::MAX || Some(source) == monitor_index {
            continue;
        }
        log::warn!("Voice Changer Mic was connected to another source; moving it back");
        let moved = match monitor_index {
            Some(_) => pactl(&["move-source-output", &index.to_string(), &monitor]).map(drop),
            None => Err(anyhow::anyhow!("Voice Changer Output is missing")),
        };
        if let Err(error) = moved {
            // Last resort: a silent microphone rather than the real voice.
            log::error!("Could not move Voice Changer Mic back ({error:#}); muting it");
            let _ = pactl(&["set-source-mute", SOURCE, "1"]);
            MUTED.store(true, Ordering::Release);
            result = Err(error
                .context("Voice Changer Mic was connected to another source and has been muted"));
        }
    }
    if result.is_ok() && MUTED.swap(false, Ordering::AcqRel) {
        let _ = pactl(&["set-source-mute", SOURCE, "0"]);
    }
    result
}

/// (source-output, source) for Voice Changer Mic's loopback capture streams.
fn loopback_captures(list: &str) -> Vec<(u32, u32)> {
    let wanted = "media.name = \"Voice Changer Mic input\"";
    let mut found = Vec::new();
    let (mut index, mut source) = (None, None);
    for line in list.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Source Output #") {
            index = rest.parse().ok();
            source = None;
        } else if let Some(rest) = line.strip_prefix("Source: ") {
            source = rest.parse().ok();
        } else if line == wanted {
            if let (Some(index), Some(source)) = (index.take(), source) {
                found.push((index, source));
            }
        }
    }
    found
}

/// (sink-input, sink) for the converted-voice streams in `pactl list sink-inputs`.
fn voice_streams(list: &str) -> Vec<(u32, u32)> {
    let wanted = format!("application.name = \"{PLAYBACK_NAME}\"");
    let mut found = Vec::new();
    let (mut index, mut sink) = (None, None);
    for line in list.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Sink Input #") {
            index = rest.parse().ok();
            sink = None;
        } else if let Some(rest) = line.strip_prefix("Sink: ") {
            sink = rest.parse().ok();
        } else if line == wanted {
            if let (Some(index), Some(sink)) = (index.take(), sink) {
                found.push((index, sink));
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_voice_streams_and_their_sinks() {
        let list = "Sink Input #41\n\tSink: 65\n\tProperties:\n\t\tapplication.name = \"Voice Changer\"\n\nSink Input #42\n\tSink: 65\n\tProperties:\n\t\tapplication.name = \"Voice Changer monitor\"\n\nSink Input #43\n\tSink: 70\n\tProperties:\n\t\tapplication.name = \"Voice Changer\"\n";
        assert_eq!(voice_streams(list), vec![(41, 65), (43, 70)]);
    }

    #[test]
    fn finds_the_mic_loopback_capture() {
        let list = "Source Output #194\n\tSource: 66\n\tProperties:\n\t\tmedia.name = \"RecordStream\"\n\nSource Output #1842\n\tSource: 66\n\tProperties:\n\t\tmedia.name = \"Voice Changer Mic input\"\n";
        assert_eq!(loopback_captures(list), vec![(1842, 66)]);
    }

    #[test]
    fn removes_the_microphone_before_its_sink() {
        let list = "536870916\tmodule-remap-source\tmaster=voice_changer_output.monitor source_name=voice_changer_mic\n536870917\tmodule-null-sink\tsink_name=voice_changer_output\n536870918\tmodule-null-sink\tsink_name=other\n";
        assert_eq!(removal_order(list), vec![536870916, 536870917]);
        let list = "7\tmodule-null-sink\tsink_name=voice_changer_output\n9\tmodule-remap-source\tsource_name=voice_changer_mic\n";
        assert_eq!(removal_order(list), vec![9, 7]);
    }

    #[test]
    fn finds_leftover_modules() {
        let list = "536870913\tmodule-null-sink\tsink_name=voice_changer_output sink_properties=x\n536870914\tmodule-remap-source\tmaster=voice_changer_output.monitor source_name=voice_changer_mic\n536870915\tmodule-null-sink\tsink_name=other\n";
        assert_eq!(removal_order(list), vec![536870914, 536870913]);
    }
}
