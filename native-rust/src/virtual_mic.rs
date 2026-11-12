//! Built-in virtual cable for Linux: a "Voice Changer Output" sink whose monitor is
//! exposed as a "Voice Changer Mic" source, so Discord, OBS or a game can pick the
//! converted voice as a microphone. Uses `pactl`, which works with PipeWire
//! (pipewire-pulse) and PulseAudio alike.
use anyhow::{bail, Context, Result};
use std::process::Command;

pub const SINK: &str = "voice_changer_output";
pub const SOURCE: &str = "voice_changer_mic";

/// The sink and source exist while this lives; dropping it removes them.
pub struct VirtualMic {
    modules: Vec<u32>,
}

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

impl VirtualMic {
    pub fn create() -> Result<Self> {
        // Modules left by a run that did not exit cleanly are replaced.
        for id in stale_modules(&pactl(&["list", "short", "modules"])?) {
            let _ = pactl(&["unload-module", &id.to_string()]);
        }
        let mut mic = Self {
            modules: Vec::new(),
        };
        mic.load(&[
            "module-null-sink",
            &format!("sink_name={SINK}"),
            &description("sink", "Voice Changer Output"),
        ])?;
        mic.load(&[
            "module-remap-source",
            &format!("master={SINK}.monitor"),
            &format!("source_name={SOURCE}"),
            &description("source", "Voice Changer Mic"),
        ])?;
        Ok(mic)
    }
    fn load(&mut self, args: &[&str]) -> Result<()> {
        let mut command = vec!["load-module"];
        command.extend_from_slice(args);
        let id = pactl(&command)?;
        self.modules.push(
            id.trim()
                .parse()
                .with_context(|| format!("pactl returned no module id: {id}"))?,
        );
        Ok(())
    }
}
impl Drop for VirtualMic {
    fn drop(&mut self) {
        for id in self.modules.iter().rev() {
            let _ = pactl(&["unload-module", &id.to_string()]);
        }
    }
}

fn stale_modules(list: &str) -> Vec<u32> {
    list.lines()
        .filter(|line| {
            line.contains(&format!("sink_name={SINK}"))
                || line.contains(&format!("source_name={SOURCE}"))
        })
        .filter_map(|line| line.split('\t').next()?.trim().parse().ok())
        .collect()
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
            pactl(&["move-sink-input", &index.to_string(), SINK])?;
        }
    }
    Ok(streams.len())
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

/// Removes devices left by a run that did not exit cleanly, while the option is off.
pub fn remove_stale() {
    if let Ok(list) = pactl(&["list", "short", "modules"]) {
        for id in stale_modules(&list) {
            let _ = pactl(&["unload-module", &id.to_string()]);
        }
    }
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
    fn finds_leftover_modules() {
        let list = "536870913\tmodule-null-sink\tsink_name=voice_changer_output sink_properties=x\n536870914\tmodule-remap-source\tmaster=voice_changer_output.monitor source_name=voice_changer_mic\n536870915\tmodule-null-sink\tsink_name=other\n";
        assert_eq!(stale_modules(list), vec![536870913, 536870914]);
    }
}
