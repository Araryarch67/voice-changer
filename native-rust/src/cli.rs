//! The command line: every task the GUI does without a window, for scripts and AI
//! agents. Subcommands have fixed names, `--json` prints one machine-readable result,
//! nothing ever waits for input, and the exit code tells success from failure:
//! [`EXIT_OK`], [`EXIT_FAILURE`] (the command ran and failed) and [`EXIT_USAGE`] (the
//! command line itself is wrong, including a missing `--yes`).
use crate::config::{Device, F0Detector, ModelManifest, Noise, Settings};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::{
    fmt,
    io::Write,
    path::{Path, PathBuf},
};

pub const EXIT_OK: i32 = 0;
pub const EXIT_FAILURE: i32 = 1;
pub const EXIT_USAGE: i32 = 2;

/// The first arguments of the older flag-style commands, still handled by `main.rs`.
pub const LEGACY_FLAGS: [&str; 6] = [
    "--convert",
    "--devices",
    "--analyze",
    "--benchmark",
    "--bypass",
    "--export-rvc",
];

/// A mistake in the command line, as opposed to a failure while running it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsageError(pub String);
impl fmt::Display for UsageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for UsageError {}

fn usage<T>(message: impl Into<String>) -> Result<T, UsageError> {
    Err(UsageError(message.into()))
}

/// A parsed command line.
#[derive(Debug, Clone)]
pub struct Invocation {
    /// Print one JSON object on stdout instead of text.
    pub json: bool,
    /// Log progress (`info`) on stderr.
    pub verbose: bool,
    /// No progress lines on stderr.
    pub quiet: bool,
    pub command: Command,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Input,
    Output,
}

/// Conversion settings from the command line. `auto_device` picks a GPU when one is
/// available, which can only be known once ONNX Runtime is loaded.
#[derive(Debug, Clone)]
pub struct Tuning {
    pub settings: Settings,
    pub auto_device: bool,
}

#[derive(Debug, Clone)]
pub struct LiveArgs {
    pub voice: String,
    pub input: Option<String>,
    pub output: Option<String>,
    /// `Some(None)` is the default output device.
    pub monitor: Option<Option<String>>,
    pub virtual_mic: bool,
    pub duration: Option<f64>,
    pub status_every: Option<f64>,
    pub tuning: Tuning,
}

#[derive(Debug, Clone)]
pub enum Command {
    Help(Option<String>),
    Version,
    Status,
    VoicesList,
    VoicesShow {
        voice: String,
    },
    VoicesImport {
        source: PathBuf,
        index: Option<PathBuf>,
        name: Option<String>,
    },
    VoicesDelete {
        voice: String,
    },
    VoicesSetIndex {
        voice: String,
        index: Option<PathBuf>,
    },
    Devices {
        direction: Option<Direction>,
    },
    SetupStatus,
    SetupDownload {
        models: Vec<&'static str>,
        root: Option<PathBuf>,
    },
    Convert {
        voice: String,
        input: PathBuf,
        output: PathBuf,
        tuning: Tuning,
        analyze: bool,
        overwrite: bool,
    },
    Live(LiveArgs),
    Split {
        input: PathBuf,
        out_dir: Option<PathBuf>,
        tuning: Tuning,
        overwrite: bool,
    },
    Cover {
        voice: String,
        input: PathBuf,
        output: Option<PathBuf>,
        tuning: Tuning,
        vocal_gain: f32,
        music_gain: f32,
        keep_stems: bool,
        overwrite: bool,
    },
    Analyze {
        input: PathBuf,
        output: Option<PathBuf>,
        prefix: Option<PathBuf>,
    },
    ExportRvc {
        checkpoint: PathBuf,
        output: PathBuf,
        realtime: bool,
    },
    Benchmark {
        voice: String,
        tuning: Tuning,
        blocks: usize,
    },
    DatasetList,
    DatasetShow {
        name: String,
    },
    DatasetAdd {
        name: String,
        files: Vec<PathBuf>,
    },
    DatasetRecord {
        name: String,
        seconds: f64,
        input: Option<String>,
    },
    DatasetPack {
        name: String,
        output: Option<PathBuf>,
        overwrite: bool,
    },
    DatasetDelete {
        name: String,
    },
    DatasetColab,
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------

const HELP: &str = "\
Voice Changer - real-time RVC voice changer

Usage:
  voice-changer-native                     Open the GUI
  voice-changer-native <COMMAND> [OPTIONS] [--json]

Commands:
  status                 Version, models folder, voice engine and ONNX Runtime state
  voices list            List installed voices
  voices show VOICE      Details of one voice
  voices import FILE     Import an RVC .pth checkpoint or a .zip holding one
  voices set-index VOICE (INDEX | --none)
                         Give a voice another .index file, or remove its index
  voices delete VOICE --yes
                         Delete a voice's folder from disk
  devices                List audio input and output devices
  setup status           Which shared models (ContentVec, RMVPE, CREPE, FCPE, Kim Vocal 2)
                         are installed
  setup download [MODEL...]
                         Download the shared models the voices need
  convert VOICE IN.wav OUT.wav [SETTINGS]
                         Convert a WAV file with a voice
  live VOICE [SETTINGS]  Convert the microphone in real time until stopped
  split SONG             Separate a song's vocals from its music (two WAV files)
  cover VOICE SONG [OUT.wav] [SETTINGS]
                         Sing a song in a voice: split, convert the vocals, mix back
  analyze IN.wav [OUT.wav]
                         Write JSON and PNG diagnostics for a recording
  export-rvc MODEL.pth OUT.onnx [--realtime]
                         Convert an RVC checkpoint to ONNX only (no voice folder)
  benchmark VOICE [SETTINGS]
                         Time conversion of synthetic audio on a device
  dataset list|show|add|record|pack|delete|colab
                         Clone a voice: collect recordings, pack them for the
                         Colab training notebook (see `dataset --help`)
  version                Print the version
  help [COMMAND]         Help for a command (also COMMAND --help)

Global options (anywhere on the line):
  --json                 Print one JSON object on stdout: {\"ok\":true,\"data\":...} or
                         {\"ok\":false,\"error\":{\"kind\":\"usage\"|\"failure\",\"message\":...}}
  -v, --verbose          Log progress on stderr (RUST_LOG also works)
  -q, --quiet            No progress lines on stderr
  -h, --help             Help

VOICE is a voice's name or folder name as `voices list` shows them (case does not
matter), a voice folder, or a path to its model.json.

Exit codes: 0 success, 1 the command failed, 2 the command line is wrong (unknown
option, bad value, missing --yes).

Nothing asks questions: destructive commands need --yes, and output files are only
replaced with --overwrite.

Older flag-style commands still work: --convert, --devices, --analyze, --benchmark,
--bypass, --export-rvc.";

const SETTINGS_HELP: &str = "\
Settings (all optional; the GUI's names in brackets):
  --settings FILE.json   Start from a JSON settings object (the fields below, by
                         their snake_case names); flags override it
  --semitones N          Pitch shift in semitones, -48..48 [TUNE] (default 0)
  --index-ratio R        How much of the voice's index to use, 0..1 [INDEX]
                         (default 0; needs a voice with an index)
  --f0 METHOD            Pitch detector [F0 Det.]: rmvpe (default, recommended),
                         fcpe, crepe_tiny, crepe_full, harvest, dio, yin.
                         fcpe and crepe need `setup download fcpe|crepe_tiny|crepe_full`
  --protect P            Keep breaths and consonants unconverted, 0..0.5 (default 0.5)
  --block-ms MS          Block length, 20..2731 [CHUNK] (default 320)
  --context-ms MS        Extra past audio per block, 0..5000 [EXTRA] (default 500)
  --overlap-ms MS        Crossfade between blocks (default 40)
  --search-ms MS         SOLA search window, 0..100 (default 12)
  --silence-threshold T  Blocks quieter than this are not converted, 0..1 [S.Thresh]
                         (default 0.0032)
  --noise MODE           off, sup1 (clean the microphone) or sup2 (also mute non-speech
                         and clean the output) [NOISE] (default off)
  --input-gain G         0..4 (default 1)  [GAIN in]
  --output-gain G        0..4 (default 1)  [GAIN out]
  --speaker-id N         Speaker of a multi-speaker voice (default 0)
  --echo                 Echo cancellation, for speakers instead of headphones
  --device DEV           cpu (default), cuda, directml or auto (a GPU when available)
                         [GPU]";

fn command_help(topic: &str) -> Option<String> {
    let text = match topic {
        "status" => "\
Usage: status [--json]

Shows the version, the folders searched for voices, the models folder in use, whether
the voice engine (shared ContentVec and RMVPE models) is installed, how many voices
there are, and which execution devices (cpu, cuda, directml) ONNX Runtime offers.
Start here: if engine_ready is false, run `setup download`."
            .to_owned(),
        "voices" => "\
Usage:
  voices list [--json]
  voices show VOICE [--json]
  voices import FILE [--index FILE.index] [--name NAME] [--json]
  voices set-index VOICE (INDEX.index | --none) [--json]
  voices delete VOICE --yes [--json]

list       Every voice found, with its name, id (folder name), manifest path, sample
           rate, version (v1/v2), pitch support, index and whether it can be deleted.
show       The same for one voice.
import     FILE is an RVC .pth checkpoint or a .zip holding one. The checkpoint is
           exported to ONNX into models/<name>/ with a model.json. The index is taken
           from --index, else from the zip, else the one beside the .pth. --name sets
           the voice's name (default: the file name). Needs `setup download` first.
set-index  Copies INDEX.index into the voice's folder and uses it; --none removes it.
           The change is undone if the voice does not load with it.
delete     Removes the voice's whole folder from disk. Only folders directly inside a
           models folder are deleted, never `shared`. Requires --yes; cannot be undone."
            .to_owned(),
        "devices" => "\
Usage: devices [--input | --output] [--json]

Lists audio devices with their ID and name. Pass an ID (or the exact name) to
`live --input/--output/--monitor`."
            .to_owned(),
        "setup" => "\
Usage:
  setup status [--json]
  setup download [MODEL...] [--recommended | --all] [--root DIR] [--json]

Shared models live in <models>/shared/ and every voice uses them:
  contentvec   required   ContentVec embedder (378 MB)
  rmvpe        required   RMVPE pitch detector, recommended (362 MB)
  crepe_tiny   optional   light CREPE pitch detector (2 MB)
  crepe_full   optional   precise CREPE pitch detector (89 MB)
  fcpe         optional   FCPE pitch detector (48 MB)
  kim_vocal_2  optional   vocal separation for `split` and `cover` (67 MB)

download fetches the named models (or, with none named, --recommended: contentvec,
rmvpe and crepe_tiny), skipping those already installed, and checks each against its
size and SHA-256. --all downloads every model. --root chooses the models folder (default:
the one in use, else ./models when it exists, else models/ beside the program).
Progress goes to stderr."
            .to_owned(),
        "convert" => format!(
            "\
Usage: convert VOICE IN.wav OUT.wav [SETTINGS] [--analyze] [--overwrite] [--json]
       convert --voice VOICE --input IN.wav --output OUT.wav ...

Converts IN.wav (any rate, downmixed to mono) to OUT.wav (mono 16-bit at the input's
rate) in blocks, as the live voice changer would. OUT.wav is not replaced unless
--overwrite is given. --analyze also writes OUT.json and OUT.png diagnostics.

{SETTINGS_HELP}"
        ),
        "live" => format!(
            "\
Usage: live VOICE [--input DEVICE] [--output DEVICE] [--monitor DEVICE|default]
            [--virtual-mic] [--duration SECONDS] [--status-every SECONDS] [SETTINGS]
            [--json]

Converts the microphone in real time. DEVICE is an ID or exact name from `devices`;
without one the system default is used. --monitor also plays the voice on a second
device. --virtual-mic sends the voice to the virtual microphone (Linux: the built-in
\"Voice Changer Mic\"; Windows: VB-Cable) instead of --output.

Runs for --duration seconds, or until the process is stopped (Ctrl+C or a kill
signal). With --json, each line on stdout is a JSON object: {{\"event\":\"ready\"}} once
audio flows, {{\"event\":\"status\",...}} every --status-every seconds (default 1 with
--json, 5 otherwise; 0 turns them off), and the final {{\"ok\":...}} result when it ends
by --duration or an error.

{SETTINGS_HELP}"
        ),
        "split" => "\
Usage: split SONG [--out-dir DIR] [--device DEV] [--overwrite] [--json]

Separates SONG (WAV, MP3, FLAC, OGG Vorbis or M4A/AAC) into SONG-vocals.wav and
SONG-instrumental.wav (32-bit float, 44.1 kHz stereo) beside it, or in --out-dir.
Uses the Kim Vocal 2 MDX-Net model as Ultimate Vocal Remover does; install it with
`setup download kim_vocal_2`. --device is cpu (default), cuda, directml or auto.
Existing stems are only replaced with --overwrite. Progress goes to stderr."
            .to_owned(),
        "cover" => format!(
            "\
Usage: cover VOICE SONG [OUT.wav] [--voice-gain G] [--music-gain G] [--keep-stems]
             [SETTINGS] [--overwrite] [--json]

Sings SONG in VOICE: its vocals are separated (as `split` does), converted with the
voice, and mixed back over the instrumental into OUT.wav (16-bit stereo, 44.1 kHz;
default SONG-cover.wav beside the song, turned down as a whole if it would clip).
--voice-gain and --music-gain (0..2, default 1) set the balance. --keep-stems also
writes OUT-vocals.wav, OUT-instrumental.wav and OUT-new-vocals.wav. Use --semitones
to move the song to the voice's range and --f0 rmvpe (the default) for singing.
Needs `setup download kim_vocal_2`. Progress goes to stderr.

{SETTINGS_HELP}"
        ),
        "analyze" => "\
Usage: analyze IN.wav [OUT.wav] [--out PREFIX] [--json]

With one file, describes it; with two, compares the input and the converted output
(level, pitch, clicks, clipping, timing). Writes PREFIX.json and PREFIX.png (default
prefix: the last file's path without its extension)."
            .to_owned(),
        "export-rvc" => "\
Usage: export-rvc MODEL.pth OUT.onnx [--realtime] [--json]

Converts an RVC v1/v2 checkpoint to ONNX without making a voice folder. --realtime
exports the decoder the live pipeline uses. Most users want `voices import` instead."
            .to_owned(),
        "benchmark" => format!(
            "\
Usage: benchmark VOICE [--blocks N] [SETTINGS] [--json]

Loads the voice on --device and converts N (3..100, default 6) blocks of synthetic
audio, reporting load time, per-block time and the real-time factor (below 1 keeps up).

{SETTINGS_HELP}"
        ),
        "dataset" => format!(
            "\
Usage:
  dataset list [--json]
  dataset show NAME [--json]
  dataset add NAME FILE.wav... [--json]
  dataset record NAME --seconds N [--input DEVICE] [--json]
  dataset pack NAME [--out FILE.zip] [--overwrite] [--json]
  dataset delete NAME --yes [--json]   (also removes <datasets>/NAME.zip)
  dataset colab [--json]

Clones a voice in three steps, like the GUI's Clone page:
  1. Collect at least {min} s (10 minutes or more is recommended) of one speaker in
     <datasets>/NAME/: `add` copies WAV files in, `record` records the microphone.
     Each take is trimmed of silence and its peak set to -1 dBFS.
  2. `pack` writes <datasets>/NAME.zip (or --out). Open the notebook that `colab`
     prints, choose a T4 GPU, run its four steps and upload the ZIP in step 2.
     Training takes about 1-3 hours on Colab's free GPU.
  3. Step 4 downloads NAME-rvc.zip; `voices import NAME-rvc.zip` installs it.

NAME may contain letters, digits, - and _; other characters become _. Only clone
voices you have the right to use.",
            min = crate::dataset::MINIMUM_SECONDS
        ),
        "version" => "Usage: version [--json]".to_owned(),
        "help" => "Usage: help [COMMAND]".to_owned(),
        _ => return None,
    };
    Some(text)
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Arguments after the command, with `--flag=value` split in two.
struct Cursor {
    items: std::collections::VecDeque<String>,
}
impl Cursor {
    fn new(args: &[String]) -> Self {
        let mut items = std::collections::VecDeque::new();
        for arg in args {
            match arg.split_once('=') {
                Some((flag, value)) if arg.starts_with("--") => {
                    items.push_back(flag.to_owned());
                    items.push_back(value.to_owned());
                }
                _ => items.push_back(arg.clone()),
            }
        }
        Self { items }
    }
    fn next(&mut self) -> Option<String> {
        self.items.pop_front()
    }
    fn value(&mut self, flag: &str) -> Result<String, UsageError> {
        match self.items.pop_front() {
            Some(value) => Ok(value),
            None => usage(format!("{flag} requires a value")),
        }
    }
}

fn number<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, UsageError> {
    value
        .parse()
        .map_err(|_| UsageError(format!("{flag}: `{value}` is not a valid number")))
}

pub fn parse_f0(name: &str) -> Result<F0Detector, UsageError> {
    match name.to_ascii_lowercase().replace('-', "_").as_str() {
        "rmvpe" => Ok(F0Detector::Rmvpe),
        "crepe_full" | "crepe" => Ok(F0Detector::CrepeFull),
        "crepe_tiny" => Ok(F0Detector::CrepeTiny),
        "fcpe" => Ok(F0Detector::Fcpe),
        "harvest" => Ok(F0Detector::Harvest),
        "dio" => Ok(F0Detector::Dio),
        "yin" => Ok(F0Detector::Yin),
        other => usage(format!(
            "Unknown F0 detector `{other}`; use rmvpe, fcpe, crepe_tiny, crepe_full, harvest, dio or yin"
        )),
    }
}

pub fn parse_noise(name: &str) -> Result<Noise, UsageError> {
    match name.to_ascii_lowercase().as_str() {
        "off" => Ok(Noise::Off),
        "sup1" => Ok(Noise::Sup1),
        "sup2" => Ok(Noise::Sup2),
        other => usage(format!(
            "Unknown noise mode `{other}`; use off, sup1 or sup2"
        )),
    }
}

/// `None` is `auto`.
pub fn parse_device(name: &str) -> Result<Option<Device>, UsageError> {
    match name.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Some(Device::Cpu)),
        "cuda" | "gpu" => Ok(Some(Device::Gpu)),
        "directml" | "dml" => Ok(Some(Device::DirectMl)),
        "auto" => Ok(None),
        other => usage(format!(
            "Unknown device `{other}`; use cpu, cuda, directml or auto"
        )),
    }
}

/// The command-line name of an execution device.
pub fn device_name(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Gpu => "cuda",
        Device::DirectMl => "directml",
    }
}

fn f0_name(detector: F0Detector) -> &'static str {
    match detector {
        F0Detector::Rmvpe => "rmvpe",
        F0Detector::CrepeFull => "crepe_full",
        F0Detector::CrepeTiny => "crepe_tiny",
        F0Detector::Fcpe => "fcpe",
        F0Detector::Harvest => "harvest",
        F0Detector::Dio => "dio",
        F0Detector::Yin => "yin",
    }
}

/// Collects the settings flags; applied over `--settings FILE` once all are read, so
/// the order on the line does not matter.
#[derive(Default)]
struct TuningFlags {
    file: Option<PathBuf>,
    flags: Vec<(String, String)>,
    echo: bool,
}
impl TuningFlags {
    /// Takes `flag` (and its value) when it is a settings flag.
    fn take(&mut self, flag: &str, cursor: &mut Cursor) -> Result<bool, UsageError> {
        match flag {
            "--settings" => self.file = Some(cursor.value(flag)?.into()),
            "--echo" => self.echo = true,
            "--semitones"
            | "--tune"
            | "--index-ratio"
            | "--f0"
            | "--f0-detector"
            | "--protect"
            | "--block-ms"
            | "--context-ms"
            | "--overlap-ms"
            | "--search-ms"
            | "--silence-threshold"
            | "--noise"
            | "--input-gain"
            | "--output-gain"
            | "--speaker-id"
            | "--device" => {
                let value = cursor.value(flag)?;
                self.flags.push((flag.to_owned(), value));
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn build(self) -> Result<Tuning, UsageError> {
        let mut settings = match &self.file {
            Some(file) => {
                let text = std::fs::read_to_string(file).map_err(|error| {
                    UsageError(format!(
                        "--settings: cannot read {}: {error}",
                        file.display()
                    ))
                })?;
                serde_json::from_str::<Settings>(&text).map_err(|error| {
                    UsageError(format!(
                        "--settings: {} is not a settings object: {error}",
                        file.display()
                    ))
                })?
            }
            None => Settings::default(),
        };
        settings.bypass = false;
        let mut auto_device = false;
        for (flag, value) in &self.flags {
            let flag = flag.as_str();
            match flag {
                "--semitones" | "--tune" => settings.semitones = number(flag, value)?,
                "--index-ratio" => settings.index_ratio = number(flag, value)?,
                "--f0" | "--f0-detector" => settings.f0_detector = parse_f0(value)?,
                "--protect" => settings.protect = number(flag, value)?,
                "--block-ms" => settings.block_ms = number(flag, value)?,
                "--context-ms" => settings.context_ms = number(flag, value)?,
                "--overlap-ms" => settings.overlap_ms = number(flag, value)?,
                "--search-ms" => settings.search_ms = number(flag, value)?,
                "--silence-threshold" => settings.silence_threshold = number(flag, value)?,
                "--noise" => settings.noise = parse_noise(value)?,
                "--input-gain" => settings.input_gain = number(flag, value)?,
                "--output-gain" => settings.output_gain = number(flag, value)?,
                "--speaker-id" => settings.speaker_id = number(flag, value)?,
                "--device" => match parse_device(value)? {
                    Some(device) => {
                        settings.device = device;
                        auto_device = false;
                    }
                    None => auto_device = true,
                },
                _ => unreachable!("only settings flags are collected"),
            }
        }
        if self.echo {
            settings.echo = true;
        }
        settings
            .validate()
            .map_err(|error| UsageError(format!("{error:#} (see `help convert` for ranges)")))?;
        Ok(Tuning {
            settings,
            auto_device,
        })
    }
}

fn unknown(flag: &str, command: &str) -> UsageError {
    UsageError(format!(
        "Unknown option `{flag}` for `{command}`; see `{command} --help`"
    ))
}

fn positional_count(
    command: &str,
    positional: &[String],
    range: std::ops::RangeInclusive<usize>,
) -> Result<(), UsageError> {
    if range.contains(&positional.len()) {
        return Ok(());
    }
    let extra = positional.get(*range.end()).map_or_else(
        || "missing arguments".to_owned(),
        |arg| format!("unexpected argument `{arg}`"),
    );
    usage(format!("`{command}`: {extra}; see `{command} --help`"))
}

/// A positional argument, or the named one given instead (`--voice V`).
fn pick(named: Option<String>, positional: &mut std::vec::IntoIter<String>) -> Option<String> {
    named.or_else(|| positional.next())
}

/// Parses the command line after the program name.
pub fn parse(args: &[String]) -> Result<Invocation, UsageError> {
    let mut json = false;
    let mut verbose = false;
    let mut quiet = false;
    let mut help = false;
    let mut rest = Vec::new();
    for arg in args {
        match arg.as_str() {
            "--json" => json = true,
            "-v" | "--verbose" => verbose = true,
            "-q" | "--quiet" => quiet = true,
            "-h" | "--help" => help = true,
            _ => rest.push(arg.clone()),
        }
    }
    let invocation = |command| Invocation {
        json,
        verbose,
        quiet,
        command,
    };
    let Some(name) = rest.first().cloned() else {
        return Ok(invocation(Command::Help(None)));
    };
    let mut cursor = Cursor::new(&rest[1..]);
    if help {
        let topic = (command_help(&name).is_some()).then_some(name);
        return Ok(invocation(Command::Help(topic)));
    }
    let command = match name.as_str() {
        "help" => {
            let topic = cursor.next();
            if let Some(topic) = &topic {
                if command_help(topic).is_none() {
                    return usage(format!("No command named `{topic}`; see `help`"));
                }
            }
            Command::Help(topic)
        }
        "version" | "--version" | "-V" => Command::Version,
        "status" => {
            if let Some(arg) = cursor.next() {
                return Err(unknown(&arg, "status"));
            }
            Command::Status
        }
        "voices" => parse_voices(&mut cursor)?,
        "dataset" | "datasets" => parse_dataset(&mut cursor)?,
        "devices" => {
            let mut direction = None;
            while let Some(arg) = cursor.next() {
                direction = Some(match arg.as_str() {
                    "--input" => Direction::Input,
                    "--output" => Direction::Output,
                    _ => return Err(unknown(&arg, "devices")),
                });
            }
            Command::Devices { direction }
        }
        "setup" => parse_setup(&mut cursor)?,
        "convert" => parse_convert(&mut cursor)?,
        "live" => parse_live(&mut cursor)?,
        "split" => parse_split(&mut cursor)?,
        "cover" => parse_cover(&mut cursor)?,
        "analyze" => {
            let mut positional = Vec::new();
            let mut prefix = None;
            while let Some(arg) = cursor.next() {
                match arg.as_str() {
                    "--out" => prefix = Some(cursor.value(&arg)?.into()),
                    flag if flag.starts_with("--") => return Err(unknown(flag, "analyze")),
                    _ => positional.push(arg),
                }
            }
            positional_count("analyze", &positional, 1..=2)?;
            let mut positional = positional.into_iter().map(PathBuf::from);
            Command::Analyze {
                input: positional.next().expect("counted"),
                output: positional.next(),
                prefix,
            }
        }
        "export-rvc" => {
            let mut positional = Vec::new();
            let mut realtime = false;
            while let Some(arg) = cursor.next() {
                match arg.as_str() {
                    "--realtime" => realtime = true,
                    flag if flag.starts_with("--") => return Err(unknown(flag, "export-rvc")),
                    _ => positional.push(arg),
                }
            }
            positional_count("export-rvc", &positional, 2..=2)?;
            let mut positional = positional.into_iter().map(PathBuf::from);
            Command::ExportRvc {
                checkpoint: positional.next().expect("counted"),
                output: positional.next().expect("counted"),
                realtime,
            }
        }
        "benchmark" => {
            let mut tuning = TuningFlags::default();
            let mut positional = Vec::new();
            let mut voice = None;
            let mut blocks = 6;
            while let Some(arg) = cursor.next() {
                match arg.as_str() {
                    "--voice" => voice = Some(cursor.value(&arg)?),
                    "--blocks" => {
                        blocks = number(&arg, &cursor.value(&arg)?)?;
                        if !(3..=100).contains(&blocks) {
                            return usage("--blocks must be 3..100");
                        }
                    }
                    flag if tuning.take(flag, &mut cursor)? => {}
                    flag if flag.starts_with("--") => return Err(unknown(flag, "benchmark")),
                    _ => positional.push(arg),
                }
            }
            let wanted = usize::from(voice.is_none());
            positional_count("benchmark", &positional, wanted..=wanted)?;
            let mut positional = positional.into_iter();
            Command::Benchmark {
                voice: pick(voice, &mut positional).ok_or_else(|| {
                    UsageError("`benchmark` needs a VOICE; see `voices list`".into())
                })?,
                tuning: tuning.build()?,
                blocks,
            }
        }
        other if other.starts_with('-') => {
            return usage(format!("Unknown option `{other}`; see `--help`"))
        }
        other => return usage(format!("Unknown command `{other}`; see `--help`")),
    };
    Ok(invocation(command))
}

fn parse_dataset(cursor: &mut Cursor) -> Result<Command, UsageError> {
    let Some(action) = cursor.next() else {
        return usage("`dataset` needs an action: list, show, add, record, pack, delete or colab");
    };
    let action = match action.as_str() {
        "ls" => "list",
        "info" => "show",
        "remove" | "rm" => "delete",
        other => other,
    }
    .to_owned();
    let command = format!("dataset {action}");
    let mut positional = Vec::new();
    let (mut seconds, mut input, mut output) = (None, None, None);
    let (mut overwrite, mut yes) = (false, false);
    while let Some(arg) = cursor.next() {
        match (action.as_str(), arg.as_str()) {
            ("record", "--seconds") => seconds = Some(number(&arg, &cursor.value(&arg)?)?),
            ("record", "--input") => input = Some(cursor.value(&arg)?),
            ("pack", "--out") => output = Some(PathBuf::from(cursor.value(&arg)?)),
            ("pack", "--overwrite") => overwrite = true,
            ("delete", "--yes" | "-y") => yes = true,
            (_, flag) if flag.starts_with("--") => return Err(unknown(flag, &command)),
            _ => positional.push(arg),
        }
    }
    let name = |positional: &[String]| -> Result<String, UsageError> {
        let name = crate::dataset::safe_name(&positional[0]);
        if name.is_empty() {
            return usage(format!("`{command}`: NAME needs letters or digits"));
        }
        Ok(name)
    };
    Ok(match action.as_str() {
        "list" => {
            positional_count(&command, &positional, 0..=0)?;
            Command::DatasetList
        }
        "colab" => {
            positional_count(&command, &positional, 0..=0)?;
            Command::DatasetColab
        }
        "show" => {
            positional_count(&command, &positional, 1..=1)?;
            Command::DatasetShow {
                name: name(&positional)?,
            }
        }
        "add" => {
            positional_count(&command, &positional, 2..=usize::MAX)?;
            Command::DatasetAdd {
                name: name(&positional)?,
                files: positional[1..].iter().map(PathBuf::from).collect(),
            }
        }
        "record" => {
            positional_count(&command, &positional, 1..=1)?;
            let Some(seconds) = seconds else {
                return usage("`dataset record` needs --seconds N (how long to record)");
            };
            if !(1.0..=600.0).contains(&seconds) {
                return usage("`dataset record`: --seconds must be 1..600");
            }
            Command::DatasetRecord {
                name: name(&positional)?,
                seconds,
                input,
            }
        }
        "pack" => {
            positional_count(&command, &positional, 1..=1)?;
            Command::DatasetPack {
                name: name(&positional)?,
                output,
                overwrite,
            }
        }
        "delete" => {
            positional_count(&command, &positional, 1..=1)?;
            if !yes {
                return usage(
                    "`dataset delete` removes the recordings from disk; add --yes to confirm",
                );
            }
            Command::DatasetDelete {
                name: name(&positional)?,
            }
        }
        other => {
            return usage(format!(
            "Unknown action `dataset {other}`; use list, show, add, record, pack, delete or colab"
        ))
        }
    })
}

fn parse_voices(cursor: &mut Cursor) -> Result<Command, UsageError> {
    let Some(action) = cursor.next() else {
        return usage("`voices` needs an action: list, show, import, set-index or delete");
    };
    let action = match action.as_str() {
        "ls" => "list",
        "info" => "show",
        "add" => "import",
        "remove" | "rm" => "delete",
        other => other,
    }
    .to_owned();
    let command = format!("voices {action}");
    let mut positional = Vec::new();
    let mut index = None;
    let mut name = None;
    let mut none = false;
    let mut yes = false;
    while let Some(arg) = cursor.next() {
        match (action.as_str(), arg.as_str()) {
            ("import", "--index") => index = Some(PathBuf::from(cursor.value(&arg)?)),
            ("import", "--name") => name = Some(cursor.value(&arg)?),
            ("set-index", "--none") => none = true,
            ("delete", "--yes" | "-y") => yes = true,
            (_, flag) if flag.starts_with("--") => return Err(unknown(flag, &command)),
            _ => positional.push(arg),
        }
    }
    let mut args = positional.clone().into_iter();
    Ok(match action.as_str() {
        "list" => {
            positional_count(&command, &positional, 0..=0)?;
            Command::VoicesList
        }
        "show" => {
            positional_count(&command, &positional, 1..=1)?;
            Command::VoicesShow {
                voice: args.next().expect("counted"),
            }
        }
        "import" => {
            positional_count(&command, &positional, 1..=1)?;
            Command::VoicesImport {
                source: args.next().expect("counted").into(),
                index,
                name,
            }
        }
        "set-index" => {
            let wanted = 2 - usize::from(none);
            positional_count(&command, &positional, wanted..=wanted)?;
            let voice = args.next().expect("counted");
            let index = args.next().map(PathBuf::from);
            if none == index.is_some() {
                return usage("`voices set-index` takes either an INDEX file or --none");
            }
            Command::VoicesSetIndex { voice, index }
        }
        "delete" => {
            positional_count(&command, &positional, 1..=1)?;
            if !yes {
                return usage(
                    "`voices delete` removes the voice's folder from disk and cannot be undone; pass --yes to confirm",
                );
            }
            Command::VoicesDelete {
                voice: args.next().expect("counted"),
            }
        }
        other => {
            return usage(format!(
                "Unknown action `voices {other}`; use list, show, import, set-index or delete"
            ))
        }
    })
}

/// Shared model names on the command line: the file without `.onnx`.
#[cfg(feature = "catalog")]
fn shared_model(name: &str) -> Option<&'static crate::setup::SharedModel> {
    let wanted = name
        .to_ascii_lowercase()
        .replace('-', "_")
        .trim_end_matches(".onnx")
        .to_owned();
    crate::setup::SHARED
        .iter()
        .find(|model| model.file.trim_end_matches(".onnx") == wanted)
}

fn parse_setup(cursor: &mut Cursor) -> Result<Command, UsageError> {
    let Some(action) = cursor.next() else {
        return usage("`setup` needs an action: status or download");
    };
    match action.as_str() {
        "status" | "check" => {
            if let Some(arg) = cursor.next() {
                return Err(unknown(&arg, "setup status"));
            }
            Ok(Command::SetupStatus)
        }
        "download" | "install" => {
            #[cfg(feature = "catalog")]
            {
                use crate::setup::SHARED;
                let mut models: Vec<&'static str> = Vec::new();
                let mut all = false;
                let mut root = None;
                while let Some(arg) = cursor.next() {
                    match arg.as_str() {
                        "--all" => all = true,
                        "--recommended" => models.extend(
                            SHARED
                                .iter()
                                .filter(|m| m.required || m.recommended)
                                .map(|m| m.file),
                        ),
                        "--root" => root = Some(PathBuf::from(cursor.value(&arg)?)),
                        flag if flag.starts_with("--") => {
                            return Err(unknown(flag, "setup download"))
                        }
                        name => match shared_model(name) {
                            Some(model) => models.push(model.file),
                            None => {
                                return usage(format!(
                                "Unknown model `{name}`; use contentvec, rmvpe, crepe_tiny, crepe_full or fcpe"
                            ))
                            }
                        },
                    }
                }
                if all {
                    models = SHARED.iter().map(|m| m.file).collect();
                } else if models.is_empty() {
                    models = SHARED
                        .iter()
                        .filter(|m| m.required || m.recommended)
                        .map(|m| m.file)
                        .collect();
                }
                // In SHARED's order, once each.
                let models = SHARED
                    .iter()
                    .map(|m| m.file)
                    .filter(|file| models.contains(file))
                    .collect();
                Ok(Command::SetupDownload { models, root })
            }
            #[cfg(not(feature = "catalog"))]
            usage("This build cannot download models (built without the catalog feature)")
        }
        other => usage(format!(
            "Unknown action `setup {other}`; use status or download"
        )),
    }
}

fn parse_convert(cursor: &mut Cursor) -> Result<Command, UsageError> {
    let mut tuning = TuningFlags::default();
    let mut positional = Vec::new();
    let (mut voice, mut input, mut output) = (None, None, None);
    let (mut analyze, mut overwrite) = (false, false);
    while let Some(arg) = cursor.next() {
        match arg.as_str() {
            "--voice" => voice = Some(cursor.value(&arg)?),
            "--input" | "-i" => input = Some(cursor.value(&arg)?),
            "--output" | "-o" => output = Some(cursor.value(&arg)?),
            "--analyze" => analyze = true,
            "--overwrite" | "--force" => overwrite = true,
            flag if tuning.take(flag, cursor)? => {}
            flag if flag.starts_with("--") => return Err(unknown(flag, "convert")),
            _ => positional.push(arg),
        }
    }
    let wanted = [&voice, &input, &output]
        .iter()
        .filter(|v| v.is_none())
        .count();
    positional_count("convert", &positional, wanted..=wanted)?;
    let mut positional = positional.into_iter();
    let voice = pick(voice, &mut positional);
    let input = pick(input, &mut positional);
    let output = pick(output, &mut positional);
    let (Some(voice), Some(input), Some(output)) = (voice, input, output) else {
        return usage("`convert` needs VOICE IN.wav OUT.wav; see `convert --help`");
    };
    Ok(Command::Convert {
        voice,
        input: input.into(),
        output: output.into(),
        tuning: tuning.build()?,
        analyze,
        overwrite,
    })
}

fn parse_split(cursor: &mut Cursor) -> Result<Command, UsageError> {
    let mut tuning = TuningFlags::default();
    let mut positional = Vec::new();
    let (mut out_dir, mut overwrite) = (None, false);
    while let Some(arg) = cursor.next() {
        match arg.as_str() {
            "--out-dir" | "-o" => out_dir = Some(cursor.value(&arg)?.into()),
            "--overwrite" | "--force" => overwrite = true,
            // Only the device matters for a split.
            "--device" => {
                tuning.take(&arg, cursor)?;
            }
            flag if flag.starts_with("--") => return Err(unknown(flag, "split")),
            _ => positional.push(arg),
        }
    }
    positional_count("split", &positional, 1..=1)?;
    Ok(Command::Split {
        input: positional.remove(0).into(),
        out_dir,
        tuning: tuning.build()?,
        overwrite,
    })
}

fn parse_cover(cursor: &mut Cursor) -> Result<Command, UsageError> {
    let mut tuning = TuningFlags::default();
    let mut positional = Vec::new();
    let (mut voice, mut vocal_gain, mut music_gain) = (None, 1.0_f32, 1.0_f32);
    let (mut keep_stems, mut overwrite) = (false, false);
    let gain = |flag: &str, value: String| -> Result<f32, UsageError> {
        let gain: f32 = number(flag, &value)?;
        if !(0.0..=2.0).contains(&gain) {
            return usage(format!("{flag} must be 0..2"));
        }
        Ok(gain)
    };
    while let Some(arg) = cursor.next() {
        match arg.as_str() {
            "--voice" => voice = Some(cursor.value(&arg)?),
            "--voice-gain" => vocal_gain = gain(&arg, cursor.value(&arg)?)?,
            "--music-gain" => music_gain = gain(&arg, cursor.value(&arg)?)?,
            "--keep-stems" => keep_stems = true,
            "--overwrite" | "--force" => overwrite = true,
            flag if tuning.take(flag, cursor)? => {}
            flag if flag.starts_with("--") => return Err(unknown(flag, "cover")),
            _ => positional.push(arg),
        }
    }
    let least = 1 + usize::from(voice.is_none());
    positional_count("cover", &positional, least..=least + 1)?;
    let mut positional = positional.into_iter();
    let voice = pick(voice, &mut positional)
        .ok_or_else(|| UsageError("`cover` needs VOICE SONG; see `cover --help`".into()))?;
    let input = positional.next().expect("counted").into();
    Ok(Command::Cover {
        voice,
        input,
        output: positional.next().map(PathBuf::from),
        tuning: tuning.build()?,
        vocal_gain,
        music_gain,
        keep_stems,
        overwrite,
    })
}

fn parse_live(cursor: &mut Cursor) -> Result<Command, UsageError> {
    let mut tuning = TuningFlags::default();
    let mut positional = Vec::new();
    let mut voice = None;
    let mut args = LiveArgs {
        voice: String::new(),
        input: None,
        output: None,
        monitor: None,
        virtual_mic: false,
        duration: None,
        status_every: None,
        tuning: Tuning {
            settings: Settings::default(),
            auto_device: false,
        },
    };
    let seconds = |flag: &str, value: String| -> Result<f64, UsageError> {
        let value: f64 = number(flag, &value)?;
        if !value.is_finite() || value < 0.0 {
            return usage(format!("{flag} must be a number of seconds, 0 or more"));
        }
        Ok(value)
    };
    while let Some(arg) = cursor.next() {
        match arg.as_str() {
            "--voice" => voice = Some(cursor.value(&arg)?),
            "--input" | "-i" => args.input = Some(cursor.value(&arg)?),
            "--output" | "-o" => args.output = Some(cursor.value(&arg)?),
            "--monitor" => {
                let value = cursor.value(&arg)?;
                args.monitor = Some((!value.eq_ignore_ascii_case("default")).then_some(value));
            }
            "--virtual-mic" => args.virtual_mic = true,
            "--duration" => args.duration = Some(seconds(&arg, cursor.value(&arg)?)?),
            "--status-every" => args.status_every = Some(seconds(&arg, cursor.value(&arg)?)?),
            flag if tuning.take(flag, cursor)? => {}
            flag if flag.starts_with("--") => return Err(unknown(flag, "live")),
            _ => positional.push(arg),
        }
    }
    if voice.is_none() && positional.is_empty() {
        return usage("`live` needs a VOICE; see `voices list`");
    }
    positional_count("live", &positional, usize::from(voice.is_none())..=1)?;
    if voice.is_some() && !positional.is_empty() {
        return usage("`live`: give the voice once, as VOICE or --voice");
    }
    let mut positional = positional.into_iter();
    args.voice = pick(voice, &mut positional)
        .ok_or_else(|| UsageError("`live` needs a VOICE; see `voices list`".into()))?;
    args.tuning = tuning.build()?;
    Ok(Command::Live(args))
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

/// What a command produced: the JSON `data` and the text shown without `--json`.
pub struct Report {
    pub data: Value,
    pub text: String,
}

/// The JSON printed for a success.
pub fn success_json(data: Value) -> Value {
    json!({ "ok": true, "data": data })
}

/// The JSON printed for a failure; `kind` is `usage` or `failure`.
pub fn error_json(kind: &str, message: &str) -> Value {
    json!({ "ok": false, "error": { "kind": kind, "message": message } })
}

fn print_line(value: &Value) {
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{value}");
    let _ = stdout.flush();
}

/// Reports a usage error the way `json` asks and returns [`EXIT_USAGE`].
pub fn report_usage_error(json: bool, error: &UsageError) -> i32 {
    if json {
        print_line(&error_json("usage", &error.0));
    } else {
        eprintln!("error: {error}");
    }
    EXIT_USAGE
}

/// Runs a parsed command, prints its result and returns the exit code.
pub fn execute(invocation: &Invocation) -> i32 {
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(invocation)))
        .unwrap_or_else(|_| {
            Err(anyhow::anyhow!(
                "Internal error (panic); run with -v for details"
            ))
        });
    match outcome {
        Ok(report) => {
            if invocation.json {
                print_line(&success_json(report.data));
            } else if !report.text.is_empty() {
                println!("{}", report.text);
            }
            EXIT_OK
        }
        Err(error) => {
            if let Some(usage) = error.downcast_ref::<UsageError>() {
                return report_usage_error(invocation.json, usage);
            }
            let message = format!("{error:#}");
            if invocation.json {
                print_line(&error_json("failure", &message));
            } else {
                eprintln!("error: {message}");
            }
            EXIT_FAILURE
        }
    }
}

fn run(invocation: &Invocation) -> Result<Report> {
    match &invocation.command {
        Command::Help(topic) => {
            let text = match topic {
                Some(topic) => command_help(topic).unwrap_or_else(|| HELP.to_owned()),
                None => HELP.to_owned(),
            };
            Ok(Report {
                data: json!({ "command": topic, "help": text }),
                text,
            })
        }
        Command::Version => Ok(Report {
            data: json!({ "version": env!("CARGO_PKG_VERSION") }),
            text: format!("voice-changer-native {}", env!("CARGO_PKG_VERSION")),
        }),
        Command::Status => status(),
        Command::VoicesList => voices_list(),
        Command::VoicesShow { voice } => {
            let manifest = resolve_voice(voice)?;
            let entry = voice_entry(&manifest);
            Ok(Report {
                text: voice_text(&entry),
                data: entry,
            })
        }
        Command::VoicesImport {
            source,
            index,
            name,
        } => voices_import(source, index.as_deref(), name.as_deref()),
        Command::VoicesDelete { voice } => voices_delete(voice),
        Command::VoicesSetIndex { voice, index } => {
            let manifest = resolve_voice(voice)?;
            crate::config::set_feature_bank(&manifest, index.as_deref())?;
            let entry = voice_entry(&manifest);
            let text = match index {
                Some(_) => format!(
                    "{} now uses {}",
                    entry["name"].as_str().unwrap_or_default(),
                    entry["index"].as_str().unwrap_or_default()
                ),
                None => format!(
                    "{} no longer uses an index",
                    entry["name"].as_str().unwrap_or_default()
                ),
            };
            Ok(Report { data: entry, text })
        }
        Command::Devices { direction } => devices(*direction),
        Command::SetupStatus => setup_status(),
        Command::SetupDownload { models, root } => {
            setup_download(models, root.as_deref(), invocation.quiet)
        }
        Command::Convert {
            voice,
            input,
            output,
            tuning,
            analyze,
            overwrite,
        } => convert(voice, input, output, tuning, *analyze, *overwrite),
        Command::Live(args) => live(args, invocation),
        Command::Split {
            input,
            out_dir,
            tuning,
            overwrite,
        } => split(
            input,
            out_dir.as_deref(),
            tuning,
            *overwrite,
            invocation.quiet,
        ),
        #[cfg(not(any(feature = "onnx", feature = "onnx-static")))]
        Command::Cover { .. } => {
            bail!("This build cannot make covers (built without ONNX Runtime)")
        }
        #[cfg(any(feature = "onnx", feature = "onnx-static"))]
        Command::Cover {
            voice,
            input,
            output,
            tuning,
            vocal_gain,
            music_gain,
            keep_stems,
            overwrite,
        } => cover(
            &CoverArgs {
                voice,
                input,
                output: output.as_deref(),
                tuning,
                vocal_gain: *vocal_gain,
                music_gain: *music_gain,
                keep_stems: *keep_stems,
                overwrite: *overwrite,
            },
            invocation.quiet,
        ),
        Command::Analyze {
            input,
            output,
            prefix,
        } => analyze(input, output.as_deref(), prefix.as_deref()),
        Command::ExportRvc {
            checkpoint,
            output,
            realtime,
        } => export_rvc(checkpoint, output, *realtime),
        Command::DatasetList => dataset_list(),
        Command::DatasetShow { name } => dataset_show(name),
        Command::DatasetAdd { name, files } => dataset_add(name, files),
        Command::DatasetRecord {
            name,
            seconds,
            input,
        } => dataset_record(name, *seconds, input.as_deref(), invocation.quiet),
        Command::DatasetPack {
            name,
            output,
            overwrite,
        } => dataset_pack(name, output.as_deref(), *overwrite),
        Command::DatasetDelete { name } => dataset_delete(name),
        Command::DatasetColab => {
            let url = crate::dataset::colab_url();
            Ok(Report {
                text: format!(
                    "{url}\nChoose Runtime > Change runtime type > T4 GPU, run the four steps and upload the ZIP from `dataset pack` in step 2. Then `voices import NAME-rvc.zip`."
                ),
                data: json!({ "url": url }),
            })
        }
        Command::Benchmark {
            voice,
            tuning,
            blocks,
        } => {
            let manifest = resolve_voice(voice)?;
            let settings = resolved_settings(tuning);
            let result = crate::benchmark::run(&manifest, &settings, *blocks)?;
            let text = format!(
                "{} on {}: load {:.0} ms, first block {:.0} ms, median {:.1} ms, worst {:.1} ms per {} ms block, real-time factor {:.2}",
                display(&manifest),
                device_name(settings.device),
                result.load_ms,
                result.first_block_ms,
                result.median_ms,
                result.max_ms,
                result.block_ms,
                result.realtime_factor
            );
            let mut data = serde_json::to_value(&result)?;
            data["device"] = json!(device_name(settings.device));
            data["manifest"] = json!(display(&manifest));
            Ok(Report { data, text })
        }
    }
}

/// A path as people and agents expect it: no `\\?\` prefix from canonicalising on
/// Windows.
pub fn display(path: &Path) -> String {
    let text = path.display().to_string();
    match text.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => rest.to_owned(),
        _ => text,
    }
}

/// [`display`] of the absolute path, so results do not depend on the working folder.
fn absolute(path: &Path) -> String {
    display(&std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()))
}

/// Finds a voice by path, name or folder name among `voices` (name, manifest).
pub fn match_voice(spec: &str, voices: &[(String, PathBuf)]) -> Result<PathBuf> {
    let path = Path::new(spec);
    if path.is_dir() && path.join("model.json").is_file() {
        return Ok(path.join("model.json").canonicalize()?);
    }
    if path.is_file() {
        return Ok(path.canonicalize()?);
    }
    let folder_name = |manifest: &Path| {
        manifest
            .parent()
            .and_then(Path::file_name)
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let mut found: Vec<&PathBuf> = voices
        .iter()
        .filter(|(name, manifest)| {
            name.eq_ignore_ascii_case(spec) || folder_name(manifest).eq_ignore_ascii_case(spec)
        })
        .map(|(_, manifest)| manifest)
        .collect();
    found.dedup();
    match found.as_slice() {
        [one] => Ok((*one).clone()),
        [] => bail!("No voice named `{spec}`; run `voices list` to see the voices"),
        many => bail!(
            "`{spec}` matches {} voices; give the path of one: {}",
            many.len(),
            many.iter()
                .map(|path| display(path))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn resolve_voice(spec: &str) -> Result<PathBuf> {
    match_voice(spec, &crate::runtime::discover_models())
}

fn voice_entry(manifest: &Path) -> Value {
    let folder = manifest.parent().unwrap_or(Path::new("."));
    let id = folder
        .file_name()
        .map(|name| name.to_string_lossy().into_owned());
    let deletable =
        crate::runtime::voice_folder(manifest, &crate::runtime::model_directories()).is_some();
    match ModelManifest::load(manifest) {
        Ok(model) => json!({
            "name": model.name,
            "id": id,
            "manifest": display(manifest),
            "folder": display(folder),
            "sample_rate": model.sample_rate,
            "version": if model.channels == 256 { "v1" } else { "v2" },
            "use_f0": model.use_f0,
            "index": model.feature_bank.as_deref().map(display),
            "deletable": deletable,
        }),
        Err(error) => json!({
            "name": id,
            "id": id,
            "manifest": display(manifest),
            "folder": display(folder),
            "error": format!("{error:#}"),
            "deletable": deletable,
        }),
    }
}

fn voice_text(entry: &Value) -> String {
    let field = |key: &str| match &entry[key] {
        Value::String(text) => text.clone(),
        Value::Null => "-".into(),
        other => other.to_string(),
    };
    if entry.get("error").is_some() {
        return format!(
            "{}\n  manifest: {}\n  error: {}",
            field("name"),
            field("manifest"),
            field("error")
        );
    }
    format!(
        "{}\n  id: {}\n  manifest: {}\n  {} Hz, RVC {}, pitch {}\n  index: {}\n  deletable: {}",
        field("name"),
        field("id"),
        field("manifest"),
        field("sample_rate"),
        field("version"),
        if entry["use_f0"] == json!(true) {
            "yes"
        } else {
            "no"
        },
        field("index"),
        field("deletable"),
    )
}

fn voices_list() -> Result<Report> {
    let voices: Vec<Value> = crate::runtime::discover_models()
        .iter()
        .map(|(_, manifest)| voice_entry(manifest))
        .collect();
    let text = if voices.is_empty() {
        "No voices found. Import one with `voices import FILE.pth` (after `setup download`)."
            .to_owned()
    } else {
        voices
            .iter()
            .map(|voice| {
                format!(
                    "{}\t{}\t{}",
                    voice["id"].as_str().unwrap_or_default(),
                    voice["name"].as_str().unwrap_or_default(),
                    voice["manifest"].as_str().unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    Ok(Report {
        data: json!({ "voices": voices }),
        text,
    })
}

fn voices_import(source: &Path, index: Option<&Path>, name: Option<&str>) -> Result<Report> {
    #[cfg(feature = "export")]
    {
        anyhow::ensure!(source.is_file(), "No such file: {}", source.display());
        let manifest = crate::import::import_with(source, index, name)?;
        let entry = voice_entry(&manifest);
        let text = format!("Imported\n{}", voice_text(&entry));
        Ok(Report { data: entry, text })
    }
    #[cfg(not(feature = "export"))]
    {
        let _ = (source, index, name);
        bail!("This build cannot import voices (built without the export feature)")
    }
}

fn voices_delete(voice: &str) -> Result<Report> {
    let manifest = resolve_voice(voice)?;
    let entry = voice_entry(&manifest);
    match crate::runtime::delete_voice(&manifest)
        .with_context(|| format!("Could not delete {}", display(&manifest)))?
    {
        Some(folder) => Ok(Report {
            text: format!("Deleted {}", display(&folder)),
            data: json!({ "deleted": display(&folder), "voice": entry }),
        }),
        None => bail!(
            "{} is not in a models folder, so its files are left alone; delete them yourself",
            display(&manifest)
        ),
    }
}

fn devices(direction: Option<Direction>) -> Result<Report> {
    #[cfg(feature = "audio")]
    {
        let devices: Vec<Value> = crate::audio::devices()?
            .into_iter()
            .filter(|device| match direction {
                Some(Direction::Input) => device.input,
                Some(Direction::Output) => !device.input,
                None => true,
            })
            .map(|device| {
                json!({
                    "direction": if device.input { "input" } else { "output" },
                    "id": device.id,
                    "name": device.name,
                })
            })
            .collect();
        let text = devices
            .iter()
            .map(|d| {
                format!(
                    "{}\t{}\t{}",
                    d["direction"].as_str().unwrap_or_default(),
                    d["id"].as_str().unwrap_or_default(),
                    d["name"].as_str().unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(Report {
            data: json!({ "devices": devices }),
            text,
        })
    }
    #[cfg(not(feature = "audio"))]
    {
        let _ = direction;
        bail!("This build has no audio devices (built without the audio feature)")
    }
}

/// Finds an audio device by ID, exact name, or a unique part of its name.
#[cfg(feature = "audio")]
fn resolve_device(spec: &str, input: bool) -> Result<String> {
    let devices: Vec<_> = crate::audio::devices()?
        .into_iter()
        .filter(|d| d.input == input)
        .collect();
    if let Some(device) = devices.iter().find(|d| d.id == spec) {
        return Ok(device.id.clone());
    }
    if let Some(device) = devices.iter().find(|d| d.name.eq_ignore_ascii_case(spec)) {
        return Ok(device.id.clone());
    }
    let lower = spec.to_lowercase();
    let partial: Vec<_> = devices
        .iter()
        .filter(|d| d.name.to_lowercase().contains(&lower))
        .collect();
    let kind = if input { "input" } else { "output" };
    match partial.as_slice() {
        [one] => Ok(one.id.clone()),
        [] => bail!("No {kind} device `{spec}`; run `devices --{kind}`"),
        many => bail!(
            "`{spec}` matches {} {kind} devices ({}); give its ID from `devices`",
            many.len(),
            many.iter()
                .map(|d| d.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn status() -> Result<Report> {
    let root = crate::runtime::models_root();
    let voices = crate::runtime::discover_models().len();
    let mut directories: Vec<String> = Vec::new();
    for directory in crate::runtime::model_directories() {
        let shown = display(&directory);
        if !directories.contains(&shown) {
            directories.push(shown);
        }
    }
    #[cfg(any(feature = "onnx", feature = "onnx-static"))]
    let (runtime_ok, runtime_error) = match std::panic::catch_unwind(crate::runtime::initialize) {
        Ok(Ok(())) => (true, None),
        Ok(Err(error)) => (false, Some(format!("{error:#}"))),
        Err(_) => (false, Some("ONNX Runtime could not be loaded".to_owned())),
    };
    #[cfg(not(any(feature = "onnx", feature = "onnx-static")))]
    let (runtime_ok, runtime_error) = (false, Some("Built without ONNX Runtime".to_owned()));
    let execution: Vec<&str> = if runtime_ok {
        crate::runtime::available_devices()
            .iter()
            .map(|d| device_name(*d))
            .collect()
    } else {
        Vec::new()
    };
    let data = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "model_directories": directories,
        "models_root": root.as_deref().map(display),
        "engine_ready": root.is_some(),
        "voices": voices,
        "onnx_runtime": { "loaded": runtime_ok, "error": runtime_error },
        "execution_devices": execution,
    });
    let text = format!(
        "Voice Changer {} ({} {})\nModels folder: {}\nVoice engine: {}\nVoices: {voices}\nONNX Runtime: {}\nDevices: {}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        root.as_deref().map_or_else(|| "none yet".to_owned(), display),
        if root.is_some() {
            "ready"
        } else {
            "not set up; run `setup download`"
        },
        if runtime_ok {
            "loaded".to_owned()
        } else {
            runtime_error.unwrap_or_default()
        },
        if execution.is_empty() {
            "-".to_owned()
        } else {
            execution.join(", ")
        },
    );
    Ok(Report { data, text })
}

#[cfg(feature = "catalog")]
fn shared_status(folder: &Path) -> Vec<Value> {
    crate::setup::SHARED
        .iter()
        .map(|model| {
            json!({
                "name": model.file.trim_end_matches(".onnx"),
                "file": model.file,
                "title": model.title,
                "purpose": model.purpose,
                "required": model.required,
                "recommended": model.recommended,
                "bytes": model.bytes,
                "installed": crate::setup::installed(folder, model),
            })
        })
        .collect()
}

#[cfg(feature = "catalog")]
fn shared_text(root: &Path, models: &[Value]) -> String {
    let mut lines = vec![format!("Models folder: {}", display(root))];
    for model in models {
        lines.push(format!(
            "  {:<11} {:<9} {:<10} {}",
            model["name"].as_str().unwrap_or_default(),
            if model["installed"] == json!(true) {
                "installed"
            } else {
                "missing"
            },
            if model["required"] == json!(true) {
                "required"
            } else {
                "optional"
            },
            model["title"].as_str().unwrap_or_default()
        ));
    }
    lines.join("\n")
}

fn setup_status() -> Result<Report> {
    #[cfg(feature = "catalog")]
    {
        let root = crate::runtime::models_root().unwrap_or_else(crate::setup::target_root);
        let models = shared_status(&root.join("shared"));
        let ready = crate::runtime::models_root().is_some();
        let mut text = shared_text(&root, &models);
        if !ready {
            text.push_str("\nThe voice engine is not set up; run `setup download`.");
        }
        Ok(Report {
            data: json!({ "root": display(&root), "engine_ready": ready, "models": models }),
            text,
        })
    }
    #[cfg(not(feature = "catalog"))]
    bail!("This build cannot download models (built without the catalog feature)")
}

#[cfg(feature = "catalog")]
fn setup_download(models: &[&'static str], root: Option<&Path>, quiet: bool) -> Result<Report> {
    use std::sync::{atomic::Ordering, Arc};
    let root = root
        .map(Path::to_path_buf)
        .unwrap_or_else(|| crate::runtime::models_root().unwrap_or_else(crate::setup::target_root));
    let progress = Arc::new(crate::catalog::Progress::default());
    let worker = {
        let (root, models, progress) = (root.clone(), models.to_vec(), progress.clone());
        std::thread::spawn(move || crate::setup::download(&root, &models, &progress))
    };
    let mut last = std::time::Instant::now();
    while !worker.is_finished() {
        std::thread::sleep(std::time::Duration::from_millis(200));
        if !quiet && last.elapsed() >= std::time::Duration::from_secs(2) {
            last = std::time::Instant::now();
            let total = progress.total.load(Ordering::Relaxed);
            let received = progress.received.load(Ordering::Relaxed);
            if total > 0 {
                eprintln!(
                    "Downloading: {:.0}% ({} of {} MB)",
                    received as f64 * 100.0 / total as f64,
                    received / 1_000_000,
                    total / 1_000_000
                );
            }
        }
    }
    worker
        .join()
        .map_err(|_| anyhow::anyhow!("The download stopped unexpectedly"))??;
    let status = shared_status(&root.join("shared"));
    let ready =
        root.join("shared/contentvec.onnx").is_file() && root.join("shared/rmvpe.onnx").is_file();
    Ok(Report {
        text: format!("{}\nDone.", shared_text(&root, &status)),
        data: json!({
            "root": display(&root),
            "requested": models.iter().map(|f| f.trim_end_matches(".onnx")).collect::<Vec<_>>(),
            "engine_ready": ready,
            "models": status,
        }),
    })
}

#[cfg(not(feature = "catalog"))]
fn setup_download(_: &[&'static str], _: Option<&Path>, _: bool) -> Result<Report> {
    bail!("This build cannot download models (built without the catalog feature)")
}

fn resolved_settings(tuning: &Tuning) -> Settings {
    let mut settings = tuning.settings.clone();
    if tuning.auto_device {
        settings.device = crate::runtime::preferred_gpu().unwrap_or(Device::Cpu);
    }
    settings
}

fn settings_json(settings: &Settings) -> Value {
    let mut value = serde_json::to_value(settings).unwrap_or(Value::Null);
    value["device"] = json!(device_name(settings.device));
    value["f0_detector"] = json!(f0_name(settings.f0_detector));
    value
}

/// Fails early, with the fix, when the pitch detector needs a model that is missing.
fn check_detector(settings: &Settings, manifest: &Path) -> Result<()> {
    let Some(file) = settings.f0_detector.model_file() else {
        return Ok(());
    };
    let model = ModelManifest::load(manifest)?;
    if !model.use_f0 {
        return Ok(());
    }
    let beside = model.pitch.parent().unwrap_or(Path::new(".")).join(file);
    anyhow::ensure!(
        beside.is_file(),
        "The {} pitch detector needs {}; run `setup download {}`",
        f0_name(settings.f0_detector),
        display(&beside),
        file.trim_end_matches(".onnx")
    );
    Ok(())
}

fn convert(
    voice: &str,
    input: &Path,
    output: &Path,
    tuning: &Tuning,
    analyze: bool,
    overwrite: bool,
) -> Result<Report> {
    let manifest = resolve_voice(voice)?;
    anyhow::ensure!(input.is_file(), "No such input file: {}", input.display());
    check_overwrite(&[output], overwrite)?;
    let settings = resolved_settings(tuning);
    check_detector(&settings, &manifest)?;
    let started = std::time::Instant::now();
    let converter = crate::engine::load_converter_on(&manifest, settings.device)?;
    crate::wav::convert(
        &mut crate::engine::Engine::new(Some(converter)),
        input,
        output,
        &settings,
    )?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let duration_s = hound::WavReader::open(output)
        .map(|reader| reader.duration() as f64 / reader.spec().sample_rate.max(1) as f64)
        .unwrap_or(0.0);
    let mut data = json!({
        "voice": voice_entry(&manifest),
        "input": absolute(input),
        "output": absolute(output),
        "duration_s": duration_s,
        "elapsed_ms": elapsed_ms,
        "settings": settings_json(&settings),
    });
    let mut text = format!(
        "Converted {} -> {} ({duration_s:.1} s of audio in {:.1} s on {})",
        input.display(),
        output.display(),
        elapsed_ms / 1000.0,
        device_name(settings.device)
    );
    if analyze {
        let prefix = output.with_extension("");
        let label = display(&manifest);
        let (report, json_path, png_path) = crate::analyze::analyze_wavs_with_context(
            input,
            Some(output),
            &prefix,
            Some(&label),
            Some(&settings),
        )?;
        data["analysis"] = json!({
            "json": absolute(&json_path),
            "png": png_path.as_deref().map(absolute),
            "diagnostics": report.diagnostics,
        });
        text.push_str(&format!(
            "\n{}\nReport JSON: {}",
            crate::analyze::format_summary(&report),
            json_path.display()
        ));
        if let Some(png) = png_path {
            text.push_str(&format!("\nReport PNG: {}", png.display()));
        }
    }
    Ok(Report { data, text })
}

/// Refuses to replace `paths` unless `overwrite`.
fn check_overwrite(paths: &[&Path], overwrite: bool) -> Result<()> {
    if let Some(path) = paths
        .iter()
        .find(|path| path.exists())
        .filter(|_| !overwrite)
    {
        return Err(UsageError(format!(
            "{} already exists; pass --overwrite to replace it",
            path.display()
        ))
        .into());
    }
    Ok(())
}

/// Runs `work` on a thread and prints its progress on stderr every two seconds.
#[cfg(any(feature = "onnx", feature = "onnx-static"))]
fn with_progress<T: Send>(
    quiet: bool,
    progress: &crate::separate::Progress,
    work: impl FnOnce() -> Result<T> + Send,
) -> Result<T> {
    use crate::separate::{CONVERTING, SEPARATING};
    use std::sync::atomic::Ordering;
    std::thread::scope(|scope| {
        let worker = scope.spawn(work);
        let mut last = std::time::Instant::now();
        while !worker.is_finished() {
            std::thread::sleep(std::time::Duration::from_millis(200));
            if !quiet && last.elapsed() >= std::time::Duration::from_secs(2) {
                last = std::time::Instant::now();
                let stage = match progress.stage.load(Ordering::Relaxed) {
                    SEPARATING => "Separating the vocals",
                    CONVERTING => "Converting the vocals",
                    _ => "Saving",
                };
                eprintln!("{stage}: {:.0}%", progress.fraction() * 100.0);
            }
        }
        worker
            .join()
            .map_err(|_| anyhow::anyhow!("The job stopped unexpectedly"))?
    })
}

#[cfg(any(feature = "onnx", feature = "onnx-static"))]
fn split(
    input: &Path,
    out_dir: Option<&Path>,
    tuning: &Tuning,
    overwrite: bool,
    quiet: bool,
) -> Result<Report> {
    anyhow::ensure!(input.is_file(), "No such song: {}", input.display());
    let model = crate::separate::require_model()?;
    let (vocals, instrumental) = crate::separate::stem_paths(input, out_dir);
    check_overwrite(&[&vocals, &instrumental], overwrite)?;
    if let Some(folder) = out_dir {
        std::fs::create_dir_all(folder)?;
    }
    let device = resolved_settings(tuning).device;
    let started = std::time::Instant::now();
    let progress = crate::separate::Progress::default();
    let seconds = with_progress(quiet, &progress, || {
        crate::separate::split_file(&model, device, input, &vocals, &instrumental, &progress)
    })?;
    let elapsed = started.elapsed().as_secs_f64();
    Ok(Report {
        text: format!(
            "Split {} ({seconds:.1} s in {elapsed:.1} s on {})\nVocals: {}\nInstrumental: {}",
            input.display(),
            device_name(device),
            vocals.display(),
            instrumental.display()
        ),
        data: json!({
            "input": absolute(input),
            "vocals": absolute(&vocals),
            "instrumental": absolute(&instrumental),
            "duration_s": seconds,
            "elapsed_ms": elapsed * 1000.0,
            "device": device_name(device),
            "model": display(&model),
        }),
    })
}

#[cfg(not(any(feature = "onnx", feature = "onnx-static")))]
fn split(_: &Path, _: Option<&Path>, _: &Tuning, _: bool, _: bool) -> Result<Report> {
    bail!("This build cannot separate vocals (built without ONNX Runtime)")
}

/// What `cover` was asked for.
#[cfg(any(feature = "onnx", feature = "onnx-static"))]
struct CoverArgs<'a> {
    voice: &'a str,
    input: &'a Path,
    output: Option<&'a Path>,
    tuning: &'a Tuning,
    vocal_gain: f32,
    music_gain: f32,
    keep_stems: bool,
    overwrite: bool,
}

#[cfg(any(feature = "onnx", feature = "onnx-static"))]
fn cover(args: &CoverArgs, quiet: bool) -> Result<Report> {
    let manifest = resolve_voice(args.voice)?;
    anyhow::ensure!(
        args.input.is_file(),
        "No such song: {}",
        args.input.display()
    );
    let model = crate::separate::require_model()?;
    let output = args.output.map(Path::to_path_buf).unwrap_or_else(|| {
        let stem = args.input.file_stem().unwrap_or_default().to_string_lossy();
        args.input.with_file_name(format!("{stem}-cover.wav"))
    });
    check_overwrite(&[&output], args.overwrite)?;
    let settings = resolved_settings(args.tuning);
    check_detector(&settings, &manifest)?;
    let started = std::time::Instant::now();
    let progress = crate::separate::Progress::default();
    let files = with_progress(quiet, &progress, || {
        crate::separate::cover(
            &crate::separate::CoverJob {
                separation_model: &model,
                voice: &manifest,
                settings: &settings,
                input: args.input,
                output: &output,
                vocal_gain: args.vocal_gain,
                music_gain: args.music_gain,
                keep_stems: args.keep_stems,
            },
            &progress,
        )
    })?;
    let elapsed = started.elapsed().as_secs_f64();
    let mut text = format!(
        "Cover of {} in {} -> {} ({:.1} s in {elapsed:.1} s on {})",
        args.input.display(),
        display(&manifest),
        files.cover.display(),
        files.seconds,
        device_name(settings.device)
    );
    for stem in &files.stems {
        text.push_str(&format!("\n{}", stem.display()));
    }
    Ok(Report {
        text,
        data: json!({
            "voice": voice_entry(&manifest),
            "input": absolute(args.input),
            "output": absolute(&files.cover),
            "stems": files.stems.iter().map(|p| absolute(p)).collect::<Vec<_>>(),
            "duration_s": files.seconds,
            "elapsed_ms": elapsed * 1000.0,
            "voice_gain": args.vocal_gain,
            "music_gain": args.music_gain,
            "settings": settings_json(&settings),
        }),
    })
}

fn analyze(input: &Path, output: Option<&Path>, prefix: Option<&Path>) -> Result<Report> {
    let prefix = prefix
        .map(Path::to_path_buf)
        .unwrap_or_else(|| output.unwrap_or(input).with_extension(""));
    let (report, json_path, png_path) =
        crate::analyze::analyze_wavs_to_disk(input, output, &prefix)?;
    let mut text = format!(
        "{}\nReport JSON: {}",
        crate::analyze::format_summary(&report),
        json_path.display()
    );
    if let Some(png) = &png_path {
        text.push_str(&format!("\nReport PNG: {}", png.display()));
    }
    Ok(Report {
        data: json!({
            "json": absolute(&json_path),
            "png": png_path.as_deref().map(absolute),
            "report": report,
        }),
        text,
    })
}

/// A dataset folder as JSON: its name, folder, takes and length.
fn dataset_entry(name: &str) -> Value {
    let folder = crate::dataset::root().join(name);
    let takes = crate::dataset::takes(&folder);
    let seconds = crate::dataset::total_seconds(&takes);
    json!({
        "name": name,
        "folder": display(&folder),
        "takes": takes.iter().map(|take| json!({
            "file": display(&take.path),
            "seconds": (take.seconds * 10.0).round() / 10.0,
        })).collect::<Vec<_>>(),
        "seconds": seconds.round(),
        "ready": seconds >= crate::dataset::MINIMUM_SECONDS,
        "recommended_seconds": crate::dataset::RECOMMENDED_SECONDS,
    })
}

fn dataset_line(entry: &Value) -> String {
    let seconds = entry["seconds"].as_f64().unwrap_or_default();
    format!(
        "{}: {} takes, {}:{:02}{}",
        entry["name"].as_str().unwrap_or_default(),
        entry["takes"].as_array().map_or(0, Vec::len),
        seconds as u64 / 60,
        seconds as u64 % 60,
        if entry["ready"].as_bool().unwrap_or_default() {
            ""
        } else {
            " (too short to train)"
        }
    )
}

fn dataset_list() -> Result<Report> {
    let root = crate::dataset::root();
    let mut names: Vec<String> = std::fs::read_dir(&root)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.path().is_dir())
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    let entries: Vec<Value> = names.iter().map(|name| dataset_entry(name)).collect();
    let text = if entries.is_empty() {
        format!("No datasets in {}", display(&root))
    } else {
        entries
            .iter()
            .map(dataset_line)
            .collect::<Vec<_>>()
            .join("\n")
    };
    Ok(Report {
        data: json!({ "root": display(&root), "datasets": entries }),
        text,
    })
}

fn dataset_show(name: &str) -> Result<Report> {
    let entry = dataset_entry(name);
    let mut text = dataset_line(&entry);
    for take in entry["takes"].as_array().into_iter().flatten() {
        text.push_str(&format!(
            "\n  {} ({} s)",
            take["file"].as_str().unwrap_or_default(),
            take["seconds"]
        ));
    }
    Ok(Report { data: entry, text })
}

fn dataset_add(name: &str, files: &[PathBuf]) -> Result<Report> {
    let folder = crate::dataset::root().join(name);
    let mut added = Vec::new();
    for file in files {
        let report = crate::dataset::add_file(&folder, file)
            .with_context(|| format!("Could not add {}", display(file)))?;
        added.push(take_json(&report));
    }
    let entry = dataset_entry(name);
    Ok(Report {
        text: format!("Added {} file(s). {}", added.len(), dataset_line(&entry)),
        data: json!({ "added": added, "dataset": entry }),
    })
}

fn take_json(report: &crate::dataset::TakeReport) -> Value {
    json!({
        "file": display(&report.take.path),
        "seconds": (report.take.seconds * 10.0).round() / 10.0,
        "clipped": report.clipped,
        "quiet": report.quiet,
    })
}

#[cfg(feature = "audio")]
fn dataset_record(name: &str, seconds: f64, input: Option<&str>, quiet: bool) -> Result<Report> {
    let device = input.map(|spec| resolve_device(spec, true)).transpose()?;
    let recorder = crate::recorder::Recorder::start(&device)?;
    if !quiet {
        eprintln!("Recording for {seconds:.0} s; speak now.");
    }
    std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
    let (rate, samples) = recorder.finish();
    let folder = crate::dataset::root().join(name);
    let report = crate::dataset::save_take(&folder, rate, &samples)?;
    let entry = dataset_entry(name);
    let mut text = format!(
        "Saved {} ({:.1} s of voice). {}",
        display(&report.take.path),
        report.take.seconds,
        dataset_line(&entry)
    );
    if report.clipped {
        text.push_str("\nwarning: the take clipped; lower the microphone gain");
    } else if report.quiet {
        text.push_str("\nwarning: the take is very quiet; raise the microphone gain");
    }
    Ok(Report {
        text,
        data: json!({ "take": take_json(&report), "dataset": entry }),
    })
}
#[cfg(not(feature = "audio"))]
fn dataset_record(_: &str, _: f64, _: Option<&str>, _: bool) -> Result<Report> {
    bail!("This build cannot record (built without the audio feature)")
}

#[cfg(feature = "export")]
fn dataset_pack(name: &str, output: Option<&Path>, overwrite: bool) -> Result<Report> {
    let folder = crate::dataset::root().join(name);
    anyhow::ensure!(
        folder.is_dir(),
        "No dataset named {name}; see `dataset list`"
    );
    let output = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| crate::dataset::root().join(format!("{name}.zip")));
    if output.exists() && !overwrite {
        return Err(UsageError(format!(
            "{} exists; add --overwrite to replace it",
            display(&output)
        ))
        .into());
    }
    let seconds = crate::dataset::pack(&folder, name, &output)?;
    let url = crate::dataset::colab_url();
    Ok(Report {
        text: format!(
            "Packed {seconds:.0} s into {}\nUpload it in step 2 of {url}",
            display(&output)
        ),
        data: json!({ "zip": display(&output), "seconds": seconds.round(), "colab": url }),
    })
}
#[cfg(not(feature = "export"))]
fn dataset_pack(_: &str, _: Option<&Path>, _: bool) -> Result<Report> {
    bail!("This build cannot pack datasets (built without the export feature)")
}

fn dataset_delete(name: &str) -> Result<Report> {
    let folder = crate::dataset::root().join(name);
    anyhow::ensure!(
        folder.is_dir(),
        "No dataset named {name}; see `dataset list`"
    );
    std::fs::remove_dir_all(&folder)
        .with_context(|| format!("Could not delete {}", display(&folder)))?;
    // The ZIP `dataset pack` made by default goes with it.
    let packed = crate::dataset::root().join(format!("{name}.zip"));
    let _ = std::fs::remove_file(&packed);
    Ok(Report {
        text: format!("Deleted {}", display(&folder)),
        data: json!({ "deleted": display(&folder) }),
    })
}

fn export_rvc(checkpoint: &Path, output: &Path, realtime: bool) -> Result<Report> {
    #[cfg(feature = "export")]
    {
        let exported = crate::export::export_rvc(checkpoint, output, realtime)?;
        let mut data = serde_json::to_value(&exported)?;
        data["output"] = json!(absolute(output));
        Ok(Report {
            text: format!(
                "Exported {} -> {} (RVC {}, {} Hz, {} channels, pitch {})",
                checkpoint.display(),
                output.display(),
                exported.version,
                exported.sample_rate,
                exported.channels,
                if exported.f0 { "yes" } else { "no" }
            ),
            data,
        })
    }
    #[cfg(not(feature = "export"))]
    {
        let _ = (checkpoint, output, realtime);
        bail!("This build cannot export checkpoints (built without the export feature)")
    }
}

#[cfg(feature = "audio")]
fn live(args: &LiveArgs, invocation: &Invocation) -> Result<Report> {
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};
    let manifest = resolve_voice(&args.voice)?;
    let settings = resolved_settings(&args.tuning);
    check_detector(&settings, &manifest)?;
    let input = args
        .input
        .as_deref()
        .map(|spec| resolve_device(spec, true))
        .transpose()?;
    let output = args
        .output
        .as_deref()
        .map(|spec| resolve_device(spec, false))
        .transpose()?;
    let monitor = match &args.monitor {
        Some(Some(spec)) => Some(Some(resolve_device(spec, false)?)),
        other => other.clone(),
    };
    let session = crate::audio::AudioSession::start(crate::audio::AudioOptions {
        input: input.clone(),
        output: output.clone(),
        manifest: Some(manifest.clone()),
        settings: settings.clone(),
        record_dir: std::env::temp_dir(),
        virtual_mic: args.virtual_mic,
        monitor,
    })?;
    let metrics = session.metrics.clone();
    let level = |cell: &std::sync::atomic::AtomicU32| f32::from_bits(cell.load(Ordering::Relaxed));
    let status_every = args
        .status_every
        .unwrap_or(if invocation.json { 1.0 } else { 5.0 });
    let started = Instant::now();
    let mut announced = false;
    let mut last_status = Instant::now();
    let status = |kind: &str| {
        json!({
            "event": kind,
            "elapsed_s": started.elapsed().as_secs_f64(),
            "input_level": level(&metrics.input_level),
            "output_level": level(&metrics.output_level),
            "processing_ms": level(&metrics.processing_ms),
            "block_ms": settings.block_ms,
            "input_rate": metrics.input_rate.load(Ordering::Relaxed),
            "output_rate": metrics.output_rate.load(Ordering::Relaxed),
            "dropped_input": metrics.dropped_input.load(Ordering::Relaxed),
            "dropped_output": metrics.dropped_output.load(Ordering::Relaxed),
            "underrun_samples": metrics.underrun_samples.load(Ordering::Relaxed),
        })
    };
    loop {
        std::thread::sleep(Duration::from_millis(100));
        if !metrics.running.load(Ordering::Acquire) {
            let error = metrics.error.lock().unwrap().clone();
            bail!(
                "{}",
                error.unwrap_or_else(|| "Audio stopped unexpectedly".to_owned())
            );
        }
        if !announced && metrics.ready.load(Ordering::Acquire) {
            announced = true;
            last_status = Instant::now();
            if invocation.json {
                let mut event = status("ready");
                event["voice"] = json!(display(&manifest));
                event["settings"] = settings_json(&settings);
                print_line(&event);
            } else if !invocation.quiet {
                eprintln!(
                    "Live: {} on {}{}. Stop with Ctrl+C.",
                    display(&manifest),
                    device_name(settings.device),
                    args.duration
                        .map(|d| format!(" for {d} s"))
                        .unwrap_or_default()
                );
            }
        }
        if announced && status_every > 0.0 && last_status.elapsed().as_secs_f64() >= status_every {
            last_status = Instant::now();
            let event = status("status");
            if invocation.json {
                print_line(&event);
            } else if !invocation.quiet {
                eprintln!(
                    "in {:.3}  out {:.3}  {:.1} ms per {} ms block  dropped {}/{}",
                    event["input_level"].as_f64().unwrap_or_default(),
                    event["output_level"].as_f64().unwrap_or_default(),
                    event["processing_ms"].as_f64().unwrap_or_default(),
                    settings.block_ms,
                    event["dropped_input"],
                    event["dropped_output"],
                );
            }
        }
        if args
            .duration
            .is_some_and(|d| started.elapsed().as_secs_f64() >= d)
        {
            break;
        }
    }
    let mut summary = status("stopped");
    session.stop();
    let deadline = Instant::now() + Duration::from_secs(5);
    while metrics.running.load(Ordering::Acquire) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    if let Some(error) = metrics.error.lock().unwrap().clone() {
        bail!("{error}");
    }
    summary["voice"] = json!(display(&manifest));
    summary["input_device"] = json!(input);
    summary["output_device"] = json!(output);
    summary["settings"] = settings_json(&settings);
    let text = format!(
        "Stopped after {:.1} s",
        summary["elapsed_s"].as_f64().unwrap_or_default()
    );
    Ok(Report {
        data: summary,
        text,
    })
}

#[cfg(not(feature = "audio"))]
fn live(_: &LiveArgs, _: &Invocation) -> Result<Report> {
    bail!("This build has no audio devices (built without the audio feature)")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_owned).collect()
    }

    fn parsed(line: &str) -> Invocation {
        parse(&args(line)).unwrap_or_else(|e| panic!("{line}: {e}"))
    }

    fn rejected(line: &str) -> String {
        parse(&args(line)).map(|_| ()).unwrap_err().0
    }

    #[test]
    fn split_and_cover_read_their_options() {
        match parsed("split song.mp3 --out-dir stems --device cpu --overwrite").command {
            Command::Split {
                input,
                out_dir,
                tuning,
                overwrite,
            } => {
                assert_eq!(input, PathBuf::from("song.mp3"));
                assert_eq!(out_dir, Some(PathBuf::from("stems")));
                assert_eq!(tuning.settings.device, Device::Cpu);
                assert!(overwrite);
            }
            _ => panic!("not a split"),
        }
        match parsed(
            "cover Aria song.flac out.wav --semitones 12 --voice-gain 1.25 --keep-stems --json",
        )
        .command
        {
            Command::Cover {
                voice,
                input,
                output,
                tuning,
                vocal_gain,
                music_gain,
                keep_stems,
                overwrite,
            } => {
                assert_eq!(voice, "Aria");
                assert_eq!(input, PathBuf::from("song.flac"));
                assert_eq!(output, Some(PathBuf::from("out.wav")));
                assert_eq!(tuning.settings.semitones, 12.0);
                assert_eq!((vocal_gain, music_gain), (1.25, 1.0));
                assert!(keep_stems && !overwrite);
            }
            _ => panic!("not a cover"),
        }
        assert!(matches!(
            parsed("cover --voice Aria song.mp3").command,
            Command::Cover { output: None, .. }
        ));
        assert!(rejected("split").contains("split"));
        assert!(rejected("split a.mp3 b.mp3").contains("split"));
        assert!(rejected("split a.mp3 --semitones 3").contains("--semitones"));
        assert!(rejected("cover Aria").contains("cover"));
        assert!(rejected("cover Aria song.mp3 --music-gain 3").contains("0..2"));
        assert!(command_help("split").is_some() && command_help("cover").is_some());
    }

    #[test]
    fn dataset_commands_parse() {
        assert!(matches!(parsed("dataset ls").command, Command::DatasetList));
        match parsed("dataset add My.Voice a.wav b.wav").command {
            Command::DatasetAdd { name, files } => {
                assert_eq!(name, "My_Voice");
                assert_eq!(files.len(), 2);
            }
            other => panic!("{other:?}"),
        }
        match parsed("dataset record me --seconds=30 --input mic").command {
            Command::DatasetRecord { seconds, input, .. } => {
                assert_eq!(seconds, 30.0);
                assert_eq!(input.as_deref(), Some("mic"));
            }
            other => panic!("{other:?}"),
        }
        assert!(rejected("dataset record me").contains("--seconds"));
        assert!(rejected("dataset delete me").contains("--yes"));
        assert!(rejected("dataset add me").contains("missing"));
        assert!(rejected("dataset show ..").contains("NAME"));
        assert!(matches!(
            parsed("dataset pack me --overwrite").command,
            Command::DatasetPack {
                overwrite: true,
                ..
            }
        ));
    }

    #[test]
    fn global_flags_go_anywhere() {
        let invocation = parsed("voices --json list -v");
        assert!(invocation.json && invocation.verbose && !invocation.quiet);
        assert!(matches!(invocation.command, Command::VoicesList));
        assert!(matches!(parsed("").command, Command::Help(None)));
        assert!(matches!(parsed("--version").command, Command::Version));
    }

    #[test]
    fn help_for_every_command() {
        for topic in [
            "status",
            "voices",
            "devices",
            "setup",
            "convert",
            "live",
            "analyze",
            "export-rvc",
            "benchmark",
            "dataset",
            "version",
            "help",
        ] {
            assert!(command_help(topic).is_some(), "{topic}");
            match parsed(&format!("{topic} --help")).command {
                Command::Help(Some(found)) => assert_eq!(found, topic),
                other => panic!("{topic}: {other:?}"),
            }
            assert!(
                HELP.contains(topic),
                "{topic} is missing from the main help"
            );
        }
        assert!(matches!(parsed("help convert").command, Command::Help(Some(t)) if t == "convert"));
        assert!(rejected("help nonsense").contains("nonsense"));
    }

    #[test]
    fn convert_takes_positional_or_named_arguments() {
        let command = parsed(
            "convert aria in.wav out.wav --semitones -12 --index-ratio=0.75 --f0 FCPE --protect 0.33 --device cuda --analyze",
        )
        .command;
        let Command::Convert {
            voice,
            input,
            output,
            tuning,
            analyze,
            overwrite,
        } = command
        else {
            panic!("not convert");
        };
        assert_eq!(voice, "aria");
        assert_eq!(input, PathBuf::from("in.wav"));
        assert_eq!(output, PathBuf::from("out.wav"));
        assert!(analyze && !overwrite);
        let s = tuning.settings;
        assert!(!s.bypass);
        assert_eq!(s.semitones, -12.0);
        assert_eq!(s.index_ratio, 0.75);
        assert_eq!(s.f0_detector, F0Detector::Fcpe);
        assert_eq!(s.protect, 0.33);
        assert_eq!(s.device, Device::Gpu);
        assert!(!tuning.auto_device);

        let Command::Convert {
            voice,
            input,
            output,
            tuning,
            ..
        } = parsed("convert -o out.wav --voice aria --device auto in.wav").command
        else {
            panic!("not convert");
        };
        assert_eq!(
            (voice.as_str(), input, output),
            ("aria", "in.wav".into(), "out.wav".into())
        );
        assert!(tuning.auto_device);
    }

    #[test]
    fn bad_command_lines_are_usage_errors() {
        assert!(rejected("convert aria in.wav").contains("missing"));
        assert!(rejected("convert aria in.wav out.wav extra").contains("extra"));
        assert!(rejected("convert a b c --semitones high").contains("--semitones"));
        assert!(rejected("convert a b c --semitones 99").contains("Invalid"));
        assert!(rejected("convert a b c --index-ratio 2").contains("Invalid"));
        assert!(rejected("convert a b c --f0 pyin").contains("pyin"));
        assert!(rejected("convert a b c --noise loud").contains("loud"));
        assert!(rejected("convert a b c --device tpu").contains("tpu"));
        assert!(rejected("convert a b c --wat").contains("--wat"));
        assert!(rejected("convert a b c --protect").contains("requires a value"));
        assert!(rejected("frobnicate").contains("frobnicate"));
        assert!(rejected("voices").contains("action"));
        assert!(rejected("voices dance").contains("dance"));
        assert!(rejected("devices --both").contains("--both"));
        assert!(rejected("benchmark aria --blocks 1").contains("3..100"));
        assert!(rejected("live").contains("VOICE"));
        assert!(rejected("live aria --duration -1").contains("seconds"));
    }

    #[test]
    fn delete_needs_yes() {
        let error = rejected("voices delete aria");
        assert!(error.contains("--yes"), "{error}");
        assert!(
            matches!(parsed("voices delete aria --yes").command, Command::VoicesDelete { voice } if voice == "aria")
        );
        assert!(matches!(
            parsed("voices rm aria -y").command,
            Command::VoicesDelete { .. }
        ));
    }

    #[test]
    fn voices_actions() {
        let Command::VoicesImport {
            source,
            index,
            name,
        } = parsed("voices import kobo.pth --index added.index --name Kobo").command
        else {
            panic!("not import");
        };
        assert_eq!(source, PathBuf::from("kobo.pth"));
        assert_eq!(index, Some("added.index".into()));
        assert_eq!(name.as_deref(), Some("Kobo"));
        assert!(matches!(
            parsed("voices set-index aria --none").command,
            Command::VoicesSetIndex { index: None, .. }
        ));
        assert!(matches!(
            parsed("voices set-index aria new.index").command,
            Command::VoicesSetIndex { index: Some(_), .. }
        ));
        assert!(rejected("voices set-index aria").contains("set-index"));
        assert!(rejected("voices set-index aria new.index --none").contains("set-index"));
        assert!(rejected("voices list extra").contains("extra"));
    }

    #[cfg(feature = "catalog")]
    #[test]
    fn setup_download_picks_models() {
        let models = |line: &str| match parsed(line).command {
            Command::SetupDownload { models, .. } => models,
            other => panic!("{other:?}"),
        };
        assert_eq!(
            models("setup download"),
            ["contentvec.onnx", "rmvpe.onnx", "crepe_tiny.onnx"]
        );
        assert_eq!(
            models("setup download fcpe RMVPE fcpe"),
            ["rmvpe.onnx", "fcpe.onnx"]
        );
        assert_eq!(
            models("setup download --all").len(),
            crate::setup::SHARED.len()
        );
        assert!(rejected("setup download pyin").contains("pyin"));
    }

    #[test]
    fn live_options() {
        let Command::Live(live) = parsed(
            "live --voice aria --input 0 --output Speakers --monitor default --virtual-mic --duration 2.5 --status-every 0 --echo --noise sup2",
        )
        .command
        else {
            panic!("not live");
        };
        assert_eq!(live.voice, "aria");
        assert_eq!(live.input.as_deref(), Some("0"));
        assert_eq!(live.output.as_deref(), Some("Speakers"));
        assert_eq!(live.monitor, Some(None));
        assert!(live.virtual_mic);
        assert_eq!(live.duration, Some(2.5));
        assert_eq!(live.status_every, Some(0.0));
        assert!(live.tuning.settings.echo);
        assert_eq!(live.tuning.settings.noise, Noise::Sup2);
        assert!(rejected("live aria --voice aria").contains("once"));
    }

    #[test]
    fn settings_file_is_overridden_by_flags() {
        let file =
            std::env::temp_dir().join(format!("vc-cli-settings-{}.json", std::process::id()));
        std::fs::write(
            &file,
            r#"{"semitones": 5, "protect": 0.2, "f0_detector": "harvest"}"#,
        )
        .unwrap();
        let line = format!(
            "convert a in.wav out.wav --semitones 7 --settings {}",
            file.display()
        );
        let Command::Convert { tuning, .. } =
            parse(&line.split(' ').map(str::to_owned).collect::<Vec<_>>())
                .unwrap()
                .command
        else {
            panic!("not convert");
        };
        assert_eq!(tuning.settings.semitones, 7.0);
        assert_eq!(tuning.settings.protect, 0.2);
        assert_eq!(tuning.settings.f0_detector, F0Detector::Harvest);
        assert!(!tuning.settings.bypass);
        std::fs::write(&file, "not json").unwrap();
        let line = format!("convert a b c --settings {}", file.display());
        assert!(
            parse(&line.split(' ').map(str::to_owned).collect::<Vec<_>>())
                .unwrap_err()
                .0
                .contains("not a settings object")
        );
        std::fs::remove_file(&file).unwrap();
    }

    #[test]
    fn json_envelopes() {
        assert_eq!(
            success_json(json!({ "voices": [] })).to_string(),
            r#"{"data":{"voices":[]},"ok":true}"#
        );
        let error = error_json("usage", "bad");
        assert_eq!(error["ok"], json!(false));
        assert_eq!(error["error"]["kind"], json!("usage"));
        assert_eq!(error["error"]["message"], json!("bad"));
        let tuned = settings_json(&Settings {
            device: Device::Gpu,
            f0_detector: F0Detector::CrepeTiny,
            ..Settings::default()
        });
        assert_eq!(tuned["device"], json!("cuda"));
        assert_eq!(tuned["f0_detector"], json!("crepe_tiny"));
        assert_eq!(tuned["block_ms"], json!(320));
    }

    #[test]
    fn help_and_version_run_with_json() {
        for line in ["help", "help convert", "version"] {
            let report = run(&parsed(line)).unwrap();
            assert!(!report.text.is_empty());
            assert!(report.data.is_object());
        }
        assert_eq!(
            run(&parsed("version")).unwrap().data["version"],
            json!(env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn voices_are_found_by_name_folder_or_path() {
        let root = std::env::temp_dir().join(format!("vc-cli-voices-{}", std::process::id()));
        for folder in ["aria", "bram", "bram-2"] {
            std::fs::create_dir_all(root.join(folder)).unwrap();
            std::fs::write(root.join(folder).join("model.json"), "{}").unwrap();
        }
        let voices = vec![
            ("Aria Soft".to_owned(), root.join("aria/model.json")),
            ("Bram".to_owned(), root.join("bram/model.json")),
            ("Bram".to_owned(), root.join("bram-2/model.json")),
        ];
        assert_eq!(
            match_voice("aria soft", &voices).unwrap(),
            root.join("aria/model.json")
        );
        assert_eq!(
            match_voice("ARIA", &voices).unwrap(),
            root.join("aria/model.json")
        );
        assert_eq!(
            match_voice("bram-2", &voices).unwrap(),
            root.join("bram-2/model.json")
        );
        let folder = root.join("bram").display().to_string();
        assert_eq!(
            match_voice(&folder, &voices).unwrap(),
            root.join("bram/model.json").canonicalize().unwrap()
        );
        let error = match_voice("Bram", &voices).unwrap_err().to_string();
        assert!(error.contains("matches 2 voices"), "{error}");
        let error = match_voice("nobody", &voices).unwrap_err().to_string();
        assert!(error.contains("voices list"), "{error}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn paths_lose_the_verbatim_prefix() {
        assert_eq!(display(Path::new(r"\\?\C:\models\a")), r"C:\models\a");
        assert_eq!(
            display(Path::new(r"\\?\UNC\server\share")),
            r"\\?\UNC\server\share"
        );
        assert_eq!(display(Path::new("models/a")), "models/a");
    }

    #[test]
    fn legacy_flags_are_not_commands() {
        for flag in LEGACY_FLAGS {
            assert!(rejected(flag).contains("Unknown option"), "{flag}");
        }
    }
}
